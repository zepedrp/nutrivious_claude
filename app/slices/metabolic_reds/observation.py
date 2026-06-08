"""
app/slices/metabolic_reds/observation.py — Module 13

Observation Model — Metabolic RED-S / Thyroid Slice (L2/L3)

Maps the 5-state metabolic_reds state vector x to observable outputs:

    y[0]  fT3_obs    [pmol/L]   — laboratory Free T3 (direct assay)
    y[1]  fT4_obs    [pmol/L]   — laboratory Free T4 (direct assay)
    y[2]  RMR_Proxy  [kcal/day] — resting metabolic rate proxy from wearable
                                   (doubly-labelled water analogue: RMR_ref × RMR_Multiplier)

OBS_DIM = 3.

Physiological mappings
──────────────────────
fT3_obs / fT4_obs  (linear; state already in clinical units):
    The ODE states Free_T3 and Free_T4 are maintained in pmol/L throughout,
    matching the clinical assay output directly.  scale factors = 1.0.

    Observation noise:
      σ_fT3 = 0.5 pmol/L  (intra-assay CV ~10% at 5 pmol/L; Thienpont 2010)
      σ_fT4 = 1.5 pmol/L  (intra-assay CV ~9% at 16 pmol/L; Thienpont 2010)
    Lab observations are episodic (quality_flag = 4 between draws → R inflated).

RMR_Proxy  (wearable-inferred basal metabolic rate):
    Modern optical HR wearables estimate daily TDEE with ~10–15% error
    (Dooley 2017; Shcherbina 2017).  From TDEE, RMR is back-calculated by
    subtracting estimated exercise EE.  The proxy is therefore:

        RMR_Proxy = RMR_ref × RMR_Multiplier

    where RMR_ref ≈ 1700 kcal/day is the population reference (Harris-Benedict
    mean across sex/age/height for typical athletic population).
    σ_RMR_proxy = 150 kcal/day reflects wearable estimation uncertainty.

Sparse assimilation:
    fT3/fT4 lab draws are infrequent (weeks to months).  Use inflate_R_mr()
    to set quality_flag=4 for non-draw days → R_channel × 1e8, effectively
    switching the UKF update off for those channels.
    RMR_Proxy is available daily from the wearable (flag 0–3 based on coverage).

References
──────────
    Dooley E.E. et al. (2017) JMIR Mhealth Uhealth 5(9):e116
    Shcherbina A. et al. (2017) J Pers Med 7(2):3
    Thienpont L.M. et al. (2010) Clin Chem 56(6):912–920
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.metabolic_reds.ode import (
    IDX_FREE_T3,
    IDX_FREE_T4,
    IDX_RMR_MULT,
    STATE_DIM,
    OBS_DIM,
)


# ── Observation model parameters ─────────────────────────────────────────────
class MetabolicRedsObsParams(NamedTuple):
    """Observation model parameters for the metabolic_reds slice."""
    scale_fT3:  float = 1.0      # [pmol/L / pmol/L] state already in clinical units
    scale_fT4:  float = 1.0      # [pmol/L / pmol/L] state already in clinical units
    RMR_ref:    float = 1700.0   # [kcal/day] population RMR reference (Harris-Benedict mean)


DEFAULT_MR_OBS_PARAMS: MetabolicRedsObsParams = MetabolicRedsObsParams()


# ── Observation noise covariance (population default) ─────────────────────────
# R_MR_DEFAULT is a 3×3 diagonal matrix.
# σ_fT3    = 0.5  pmol/L  (intra-assay CV ~10%;  Thienpont 2010)
# σ_fT4    = 1.5  pmol/L  (intra-assay CV ~9%;   Thienpont 2010)
# σ_RMR    = 150  kcal/day (wearable TDEE error ~10%; Shcherbina 2017)
R_MR_DEFAULT: jax.Array = jnp.diag(
    jnp.array([0.5 ** 2, 1.5 ** 2, 150.0 ** 2], dtype=jnp.float32)
)


# ── Observation function ──────────────────────────────────────────────────────
def h_mr(
    x:          jax.Array,
    obs_params: MetabolicRedsObsParams = DEFAULT_MR_OBS_PARAMS,
) -> jax.Array:
    """
    Observation function h(x) → y ∈ ℝ³ for the metabolic_reds slice.

    Pure JAX — JIT + vmap safe.

    Parameters
    ----------
    x          : shape (STATE_DIM,) = (5,)
    obs_params : MetabolicRedsObsParams

    Returns
    -------
    y : shape (OBS_DIM,) = (3,) — [fT3_obs [pmol/L], fT4_obs [pmol/L], RMR_Proxy [kcal/day]]
    """
    fT3_c = jnp.maximum(x[IDX_FREE_T3], jnp.float32(0.0))
    fT4_c = jnp.maximum(x[IDX_FREE_T4], jnp.float32(0.0))
    rmr_c = jnp.clip(x[IDX_RMR_MULT], jnp.float32(0.5), jnp.float32(1.2))

    fT3_obs   = jnp.float32(obs_params.scale_fT3) * fT3_c
    fT4_obs   = jnp.float32(obs_params.scale_fT4) * fT4_c
    rmr_proxy = jnp.float32(obs_params.RMR_ref)  * rmr_c

    return jnp.stack([fT3_obs, fT4_obs, rmr_proxy])


def h_mr_sigma(
    sigma_pts:  jax.Array,
    obs_params: MetabolicRedsObsParams = DEFAULT_MR_OBS_PARAMS,
) -> jax.Array:
    """
    Apply h_mr over (2n+1) UKF sigma points.

    Parameters
    ----------
    sigma_pts  : shape (2*STATE_DIM+1, STATE_DIM) = (11, 5)
    obs_params : MetabolicRedsObsParams

    Returns
    -------
    y_sigma    : shape (2*STATE_DIM+1, OBS_DIM) = (11, 3)
    """
    return jax.vmap(h_mr, in_axes=(0, None))(sigma_pts, obs_params)


# ── R inflation (sparse assimilation) ─────────────────────────────────────────
def inflate_R_mr(
    quality_flags: tuple[int, int, int],
    R:             jax.Array = R_MR_DEFAULT,
) -> jax.Array:
    """
    Inflate observation noise R for unavailable channels.

    quality_flags : (flag_fT3, flag_fT4, flag_RMR_proxy)
        flag = 4 → channel not available this time step → R_channel × 1e8

    Channel availability:
      fT3_obs    — episodic lab draw only; flag=4 on non-draw days
      fT4_obs    — episodic lab draw only; flag=4 on non-draw days
      RMR_Proxy  — daily from wearable; flag 0–3 based on wear-time coverage

    Returns
    -------
    R_inflated : (OBS_DIM, OBS_DIM)
    """
    factor = jnp.ones(OBS_DIM, dtype=jnp.float32)
    for i, flag in enumerate(quality_flags):
        if flag == 4:
            factor = factor.at[i].set(jnp.float32(1e8))
    return R * jnp.diag(factor)
