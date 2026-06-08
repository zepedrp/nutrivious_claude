"""
app/slices/biomechanical_tissue/observation.py

Observation Model -- h(x, theta) -> y for the Biomechanical Tissue Slice

Maps the 5-dimensional biomechanical state to three clinically accessible
proxy measurements.

Physiological basis of each observable
----------------------------------------
y[0]  Pain_VAS [0-10, Visual Analogue Scale]
      Self-reported pain intensity is the most accessible proximal marker of
      tendon and bone microdamage in the field (Alfredson 1998; de Vries 2015).
      Modelled as a weighted sum of tendon and bone microdamage via M-M
      saturation, reflecting the composite load on nociceptors:
          VAS = VAS_scale * (w_tend * TendDmg / (k_tend_sat + TendDmg)
                           + w_bone * BoneDmg / (k_bone_sat + BoneDmg))
      Measurement noise: sigma_VAS ~ HalfNormal(1.0) points (subjective NRS).

y[1]  Ultrasound_Echogenicity [au, 0-1]
      Greyscale tendon ultrasonography (GS-US) echogenicity provides a
      non-invasive proxy for tendon fibre organisation and stiffness
      (Leung 2017; Westh 2006 BJSM). Echogenicity correlates positively
      with tendon stiffness and inversely with microdamage:
          Echo = Echo_ref * Tendon_Stiffness / (1 + k_echo_dmg * TendDmg)
      Normalised to [0, 1] via population reference.
      sigma_Echo ~ HalfNormal(0.08).

y[2]  DEXA_ZScore_proxy [au]
      Dual-energy X-ray absorptiometry bone mineral density, expressed as a
      Z-score relative to the athlete's onboarding reference:
          Z_proxy = (Bone_Density - BMD_ref) / sigma_BMD_ref
      Sparse (measured once per 6-12 months at most); the UKF inflates
      R[2,2] to 1e8 on the ~360/365 days without a DEXA scan.
      sigma_DEXA ~ HalfNormal(0.25) Z-score units.

Design invariants
------------------
  * Pure JAX: no Python conditionals on traced values.
  * @jax.jit and jax.vmap safe.
  * Fail-Loud: NaN propagates.

References
----------
  Alfredson H. et al. (1998) Am J Sports Med 26:360-366       [VAS tendon pain]
  de Vries A.J. et al. (2015) BJSM 49:1554-1559              [VAS pain proxy]
  Leung J.L.Y. et al. (2017) Ultrasound Med Biol 43:1491      [US echogenicity]
  Westh E. et al. (2006) BJSM 40:814-818                      [US vs stiffness]
  Nattiv A. et al. (2023) JBMR Plus 7:e10729                  [DEXA Z-score]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.biomechanical_tissue.ode import (
    IDX_TEND_DMG,
    IDX_TEND_STIFF,
    IDX_BONE_DMG,
    IDX_BONE_DENS,
    STATE_DIM,
    OBS_DIM,
    BiomechanicalParams,
    DEFAULT_BIO_PARAMS,
)


# -- Observation parameter container ------------------------------------------

class BioObsParams(NamedTuple):
    """
    Observation model parameters for the Biomechanical Tissue slice.

    Population priors (literature-anchored)
    ----------------------------------------
    sigma_VAS      ~ HalfNormal(1.0) pts    -- Alfredson 1998: inter-rater NRS SD
    k_tend_sat     ~ LogN(log 0.50, 0.3^2) -- M-M half-saturation for tendon pain
    k_bone_sat     ~ LogN(log 0.50, 0.3^2) -- M-M half-saturation for bone pain
    VAS_scale      ~ N(8.0, 1.0^2)         -- max achievable VAS under severe damage
    w_tend         ~ Beta(3, 2)            -- tendon vs bone weight in VAS
    w_bone         ~ Beta(2, 3)            -- bone contribution (complementary)
    Echo_ref       ~ N(0.75, 0.1^2)        -- healthy tendon echogenicity au
    k_echo_dmg     ~ LogN(log 1.0, 0.4^2) -- damage -> echogenicity suppression
    sigma_Echo     ~ HalfNormal(0.08)
    sigma_DEXA     ~ HalfNormal(0.25)      -- Z-score measurement noise
    BMD_ref        ~ 1.15 g/cm2            -- population lumbar reference
    sigma_BMD_ref  ~ 0.10 g/cm2            -- reference population SD (DEXA)
    """
    sigma_VAS:      float = 1.0     # pts    -- VAS subjective measurement noise
    k_tend_sat:     float = 0.50    # au     -- M-M half-saturation (tendon -> VAS)
    k_bone_sat:     float = 0.50    # au     -- M-M half-saturation (bone -> VAS)
    VAS_scale:      float = 8.0     # pts    -- maximum VAS amplitude
    w_tend:         float = 0.60    # au     -- tendon weight in composite VAS
    w_bone:         float = 0.40    # au     -- bone weight in composite VAS
    Echo_ref:       float = 0.75    # au     -- healthy tendon echogenicity (normalised)
    k_echo_dmg:     float = 1.00    # au^-1  -- damage suppression of echogenicity
    sigma_Echo:     float = 0.08    # au     -- echogenicity measurement noise
    sigma_DEXA:     float = 0.25    # au     -- DEXA Z-score noise
    BMD_ref:        float = 1.15    # g/cm2  -- lumbar spine population reference
    sigma_BMD_ref:  float = 0.10    # g/cm2  -- DEXA reference population SD


DEFAULT_BIO_OBS_PARAMS: BioObsParams = BioObsParams()

# Nominal observation noise covariance R (diagonal, 3 x 3)
#   y[0] Pain_VAS [0-10]:              sigma^2 = 1.0^2  = 1.0
#   y[1] Ultrasound_Echogenicity [au]: sigma^2 = 0.08^2 = 0.0064
#   y[2] DEXA_ZScore_proxy [au]:       sigma^2 = 0.25^2 = 0.0625  (1e8 when absent)
R_BIO_DEFAULT: jax.Array = jnp.diag(jnp.array([
    1.0,       # Pain_VAS variance [pts^2]
    0.0064,    # Echogenicity variance [au^2]
    0.0625,    # DEXA Z-score variance [au^2] -- inflated to 1e8 on missing days
], dtype=jnp.float32))


# -- Observation function -----------------------------------------------------

@jax.jit
def h_bio(
    x:          jax.Array,
    obs_params: BioObsParams,
    bio_params: BiomechanicalParams = DEFAULT_BIO_PARAMS,
) -> jax.Array:
    """
    Biomechanical observation function: h(x, theta) -> y.

    Maps 5-state biomechanical vector to
    [Pain_VAS, Ultrasound_Echogenicity, DEXA_ZScore_proxy].
    Noiseless -- measurement noise applied externally by the filter.

    Parameters
    ----------
    x          : shape (STATE_DIM,) -- biomechanical state vector
    obs_params : BioObsParams
    bio_params : BiomechanicalParams (unused at present; reserved for future
                 genetic-scale adjustments to observation scaling)

    Returns
    -------
    y : shape (OBS_DIM,) = (3,)
        y[0] Pain_VAS [0-10, NRS]
        y[1] Ultrasound_Echogenicity [au, 0-1]
        y[2] DEXA_ZScore_proxy [au]

    Notes
    -----
    Pain_VAS: weighted M-M saturation of tendon + bone microdamage.
              VAS -> 0 when both damage states -> 0 (healthy rest).
              VAS -> VAS_scale when damage saturates both M-M functions.

    Ultrasound_Echogenicity: healthy tendon = Echo_ref; decreases with
              microdamage via suppression factor 1/(1 + k_echo_dmg*dmg).
              Tracks Tendon_Stiffness normalised by Echo_ref.

    DEXA_ZScore_proxy: (BMD - BMD_ref) / sigma_BMD_ref.
              Z = 0 at population mean; Z = -2 at significant deficit.
              Sparse; R[2,2] inflated to 1e8 on days without DEXA.
    """
    tend_dmg   = jnp.maximum(x[IDX_TEND_DMG],   jnp.float32(0.0))
    tend_stiff = jnp.maximum(x[IDX_TEND_STIFF],  jnp.float32(0.0))
    bone_dmg   = jnp.maximum(x[IDX_BONE_DMG],    jnp.float32(0.0))
    bone_dens  = x[IDX_BONE_DENS]

    # -- y[0]: Pain_VAS -------------------------------------------------------
    tend_pain  = tend_dmg / (obs_params.k_tend_sat + tend_dmg)
    bone_pain  = bone_dmg / (obs_params.k_bone_sat + bone_dmg)
    pain_vas   = obs_params.VAS_scale * (
        obs_params.w_tend * tend_pain + obs_params.w_bone * bone_pain
    )

    # -- y[1]: Ultrasound_Echogenicity ----------------------------------------
    # Echogenicity tracks stiffness, suppressed by active microdamage
    echo_suppression  = jnp.float32(1.0) / (
        jnp.float32(1.0) + obs_params.k_echo_dmg * tend_dmg
    )
    echo              = obs_params.Echo_ref * tend_stiff * echo_suppression

    # -- y[2]: DEXA_ZScore_proxy ----------------------------------------------
    dexa_z = (bone_dens - obs_params.BMD_ref) / jnp.maximum(
        obs_params.sigma_BMD_ref, jnp.float32(1e-4)
    )

    return jnp.array([pain_vas, echo, dexa_z], dtype=jnp.float32)


@jax.jit
def h_bio_sigma(
    sigma_pts:  jax.Array,
    obs_params: BioObsParams,
    bio_params: BiomechanicalParams = DEFAULT_BIO_PARAMS,
) -> jax.Array:
    """
    Vectorised observation function over UKF sigma points.

    Parameters
    ----------
    sigma_pts  : shape (2*STATE_DIM + 1, STATE_DIM)
    obs_params : BioObsParams
    bio_params : BiomechanicalParams

    Returns
    -------
    y_sigma : shape (2*STATE_DIM + 1, OBS_DIM)
    """
    return jax.vmap(h_bio, in_axes=(0, None, None))(sigma_pts, obs_params, bio_params)


# -- Quality-flag-aware R inflation -------------------------------------------

def inflate_R_bio(
    quality_flags: tuple[int, int, int],
    R_nominal:     jax.Array = R_BIO_DEFAULT,
) -> jax.Array:
    """
    Inflate R for the biomechanical observation based on data quality.

    Parameters
    ----------
    quality_flags : (flag_VAS, flag_Echo, flag_DEXA)
                    Each in [0, 4] per CanonicalObservation quality_flag:
                      0 -- excellent (direct measurement)
                      1 -- good      (indirect proxy)
                      2 -- moderate  (estimated / carry-forward)
                      3 -- poor      (imputed)
                      4 -- missing   -> R x 1e8 (predict-only)
                    flag_DEXA is 4 on the ~360/365 days without DEXA.
    R_nominal     : shape (OBS_DIM, OBS_DIM) = (3, 3)

    Returns
    -------
    R_inflated : shape (3, 3)
    """
    scale_map: dict[int, float] = {0: 1.0, 1: 1.5, 2: 4.0, 3: 16.0, 4: 1e8}
    s_vas   = scale_map.get(quality_flags[0], 1e8)
    s_echo  = scale_map.get(quality_flags[1], 1e8)
    s_dexa  = scale_map.get(quality_flags[2], 1e8)
    diag_scale = jnp.array([s_vas, s_echo, s_dexa], dtype=jnp.float32)
    return R_nominal * jnp.diag(diag_scale)
