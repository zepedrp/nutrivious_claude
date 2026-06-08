"""
app/slices/neuromuscular_tissue/observation.py

Observation Model V4.0 -- Neuromuscular Tissue Slice (L2/L3)

Maps the 6-dimensional intra-session state to two non-invasive real-time
observables:

  y[0]  EMG_Amplitude_mV  -- surface EMG amplitude [mV]
  y[1]  SmO2_pct          -- muscle O2 saturation via NIRS [%]

Physiological basis
-------------------
EMG_Amplitude_mV:
  Surface EMG amplitude correlates with the number and firing rate of active
  motor units.  At low load, primarily slow MUs (Type 1) are recruited;
  at high load, fast MUs (Type 2) add to the signal.
  EMG = k_EMG * (R1 + R2)
  At full bilateral recruitment (R1=R2=1): EMG_max = 2*k_EMG = 4.0 mV.
  Physiological range: 0-5 mV for surface electrodes (Enoka & Duchateau 2008).
  Measurement noise: sigma_EMG ~ 1.0 mV (motion artifact + skin impedance).

SmO2_pct:
  Near-infrared spectroscopy (NIRS) measures local O2 saturation of myoglobin
  and capillary Hb in the active muscle.  SmO2 falls when O2 demand exceeds
  local delivery, reflected by:
    (a) ATP depletion: lower ATP signals impaired oxidative phosphorylation
    (b) High total fiber recruitment: more fibers competing for available O2
  SmO2 = max(0, SmO2_base
               - k_SmO2_atp * (ATP_rest - ATP) / ATP_rest
               - k_SmO2_rec * (R1 + R2))
  At rest (ATP=ATP_rest, R1=R2=0): SmO2 = SmO2_base = 85 %.
  At VO2max (ATP depleted 25%, R1+R2=1.5): SmO2 ~ 35-50 % (Ferrari 2011 EJAP).
  Measurement noise: sigma_SmO2 ~ 1.0 % (inter-optode variability; Bhambhani 2004).

  Note: Glycogen depletion (Bonking) affects SmO2 indirectly through ATP
  depletion (captured by the ATP deficit term) and reduced T2 recruitment
  (captured by the total_rec term).  No explicit glycogen term is required.

Observation noise (base, diagonal)
------------------------------------
  R[0,0] = sigma_EMG^2  = 1.0   [mV^2]
  R[1,1] = sigma_SmO2^2 = 1.0   [%^2]

Quality flag inflation (flag=4 -> R * 1e8 -> predict-only channel)

References
----------
  Enoka R.M., Duchateau J. (2008) J Physiol 586:37-45     [EMG-recruitment]
  Ferrari M. et al. (2011) Eur J Appl Physiol 111:2461    [NIRS SmO2 exercise]
  Bhambhani Y.N. (2004) Sports Med 34:255-269             [NIRS SmO2 reliability]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.neuromuscular_tissue.ode import (
    IDX_ATP,
    IDX_R1,
    IDX_R2,
    STATE_DIM,
    OBS_DIM,
    NMv4Params,
    DEFAULT_V4_PARAMS,
)


# -- Observation parameter container -------------------------------------------

class NMv4ObsParams(NamedTuple):
    """
    Observation model parameters for the Neuromuscular Tissue V4.0 slice.

    Population priors (literature-anchored)
    ----------------------------------------
    k_EMG        : 2.0 mV per unit total recruitment (R1+R2)
    SmO2_base    : 85.0 % -- resting muscle oxygenation
    k_SmO2_atp   : 10.0 % per unit normalized ATP deficit
    k_SmO2_rec   : 15.0 % per unit total recruitment
    sigma_EMG    : 1.0 mV -- measurement noise
    sigma_SmO2   : 1.0 % -- NIRS measurement noise
    """
    k_EMG:       float = 2.0    # mV per unit (R1+R2)
    SmO2_base:   float = 85.0   # % resting saturation
    k_SmO2_atp:  float = 10.0   # % per unit ATP deficit fraction
    k_SmO2_rec:  float = 15.0   # % per unit (R1+R2)
    sigma_EMG:   float = 1.0    # mV -- observation noise
    sigma_SmO2:  float = 1.0    # % -- observation noise


DEFAULT_V4_OBS_PARAMS: NMv4ObsParams = NMv4ObsParams()

# Nominal R (2x2 diagonal)
R_NM_V4_DEFAULT: jax.Array = jnp.diag(jnp.array([
    1.0,   # EMG  variance [mV^2]
    1.0,   # SmO2 variance [%^2]
], dtype=jnp.float32))


# -- Observation function ------------------------------------------------------

@jax.jit
def h_nm_v4(
    x:          jax.Array,
    obs_params: NMv4ObsParams = DEFAULT_V4_OBS_PARAMS,
    nm_params:  NMv4Params    = DEFAULT_V4_PARAMS,
) -> jax.Array:
    """
    Neuromuscular V4.0 observation function: h(x, theta) -> y.

    Noiseless -- measurement noise applied by the filter externally.
    Glycogen state is not directly observed; affects y through R2 dynamics.

    Parameters
    ----------
    x          : shape (STATE_DIM,) = (6,) -- [ATP, Ca, R1, R2, RyR1, Glycogen]
    obs_params : NMv4ObsParams
    nm_params  : NMv4Params (for ATP_rest)

    Returns
    -------
    y : shape (OBS_DIM,) = (2,)
        y[0] EMG_Amplitude_mV [mV]   -- rises with total recruitment
        y[1] SmO2_pct         [%]    -- falls with ATP depletion and high recruitment
    """
    ATP = jnp.maximum(x[IDX_ATP], jnp.float32(0.0))
    R1  = jnp.clip(x[IDX_R1], jnp.float32(0.0), jnp.float32(1.0))
    R2  = jnp.clip(x[IDX_R2], jnp.float32(0.0), jnp.float32(1.0))

    total_rec = R1 + R2

    # EMG amplitude: proportional to total active fiber recruitment
    emg = jnp.float32(obs_params.k_EMG) * total_rec

    # SmO2: depletes with ATP deficit and high fiber recruitment
    atp_rest      = jnp.float32(nm_params.ATP_rest)
    atp_def_frac  = jnp.maximum(atp_rest - ATP, jnp.float32(0.0)) / atp_rest
    smo2 = jnp.float32(obs_params.SmO2_base) \
         - jnp.float32(obs_params.k_SmO2_atp) * atp_def_frac \
         - jnp.float32(obs_params.k_SmO2_rec) * total_rec
    smo2 = jnp.clip(smo2, jnp.float32(0.0), jnp.float32(100.0))

    return jnp.array([emg, smo2], dtype=jnp.float32)


# -- Vectorised version (sigma-point propagation) ------------------------------

def h_nm_v4_sigma(
    sigma_pts:  jax.Array,
    obs_params: NMv4ObsParams = DEFAULT_V4_OBS_PARAMS,
    nm_params:  NMv4Params    = DEFAULT_V4_PARAMS,
) -> jax.Array:
    """
    Vectorised observation function over UKF sigma points.

    Parameters
    ----------
    sigma_pts : shape (2*STATE_DIM+1, STATE_DIM) = (13, 6)

    Returns
    -------
    y_sigma : shape (2*STATE_DIM+1, OBS_DIM) = (13, 2)
    """
    return jax.vmap(h_nm_v4, in_axes=(0, None, None))(sigma_pts, obs_params, nm_params)


# -- Quality-flag-aware R inflation -------------------------------------------

def inflate_R_nm_v4(
    quality_flags:  tuple[int, int],
    R_nominal:      jax.Array = R_NM_V4_DEFAULT,
) -> jax.Array:
    """
    Inflate R per channel based on data quality flags.

    Parameters
    ----------
    quality_flags : (flag_EMG, flag_SmO2)
                    Each in [0, 4]:
                      0 -- excellent (direct measurement)
                      1 -- good      (indirect proxy)
                      2 -- moderate  (estimated)
                      3 -- poor      (imputed)
                      4 -- missing   -> R * 1e8 (predict-only)
    R_nominal     : shape (OBS_DIM, OBS_DIM) = (2, 2)

    Returns
    -------
    R_inflated : shape (2, 2)
    """
    _scale_map: dict[int, float] = {0: 1.0, 1: 1.5, 2: 4.0, 3: 16.0, 4: 1e8}
    s_emg  = _scale_map.get(quality_flags[0], 1e8)
    s_smo2 = _scale_map.get(quality_flags[1], 1e8)
    diag_scale = jnp.array([s_emg, s_smo2], dtype=jnp.float32)
    return R_nominal * jnp.diag(diag_scale)
