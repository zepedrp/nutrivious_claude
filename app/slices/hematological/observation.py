"""
app/slices/hematological/observation.py  Observation model h(x, theta) -> y

Observation vector y in R^3  (blood panel, sparse):
  y[0] Hgb_g_dL      Hemoglobin concentration   [g/dL]
  y[1] Hematocrit_pct Hematocrit                [%]
  y[2] Ferritin_lab   Ferritin                  [ug/L = ng/mL]

Derivation:
  RBC_Vol_mL  = RBC_Mass_g * 0.88
  PV_mL       = Plasma_Vol_L * 1000
  total_mL    = RBC_Vol_mL + PV_mL
  Hgb_g_dL   = (RBC_Mass_g * 29.04) / total_mL   (MCHC 33% x 0.88 x 100)
  Hct_pct     = RBC_Vol_mL / total_mL * 100
  Ferritin    = x[IDX_FERRITIN]

Observation noise R (diagonal):
  sigma_Hgb      = 1.0 g/dL   (base; quality_flag=4 -> *1e8)
  sigma_Hct      = 1.0 %      (base)
  sigma_Ferritin = 1.0 ug/L   (base)
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.hematological.ode import (
    IDX_RBC_MASS, IDX_PLASMA_VOL, IDX_FERRITIN,
    OBS_DIM,
)

_RBC_ML_PER_G  = 0.88
_MCHC_FRAC_100 = 29.04    # RBC_Mass_g * this / total_mL = Hgb_g_dL

_QUALITY_INFLATE = jnp.array([1.0, 2.0, 4.0, 10.0, 1e8, 1e8], dtype=jnp.float32)


class HemObsParams(NamedTuple):
    sigma_Hgb:      float = 1.0
    sigma_Hct:      float = 1.0
    sigma_Ferritin: float = 1.0


DEFAULT_HEM_OBS_PARAMS = HemObsParams()


def h_hem(
    x:          jax.Array,
    obs_params: HemObsParams = DEFAULT_HEM_OBS_PARAMS,
) -> jax.Array:
    """h: R^5 -> R^3.  JAX-traceable; no data-dependent branches."""
    RBC_Mass = jnp.maximum(x[IDX_RBC_MASS],   0.0)
    PV_L     = jnp.maximum(x[IDX_PLASMA_VOL], 0.0)
    Ferritin = jnp.maximum(x[IDX_FERRITIN],   0.0)

    rbc_vol_mL = RBC_Mass * _RBC_ML_PER_G
    pv_mL      = PV_L * 1000.0
    total_mL   = rbc_vol_mL + pv_mL + 1e-6

    hgb_g_dL = (RBC_Mass * _MCHC_FRAC_100) / total_mL
    hct_pct  = rbc_vol_mL / total_mL * 100.0

    return jnp.array([hgb_g_dL, hct_pct, Ferritin])


def observation_noise_R(
    obs_params:   HemObsParams = DEFAULT_HEM_OBS_PARAMS,
    quality_flag: int          = 0,
) -> jax.Array:
    """Diagonal R (3x3) with scalar inflation. quality_flag=4 -> R * 1e8."""
    inflate = _QUALITY_INFLATE[jnp.clip(quality_flag, 0, 5)]
    sigmas  = jnp.array([
        obs_params.sigma_Hgb,
        obs_params.sigma_Hct,
        obs_params.sigma_Ferritin,
    ], dtype=jnp.float32)
    return jnp.diag((sigmas ** 2) * inflate)


def inflate_R_hem(
    obs_params: HemObsParams         = DEFAULT_HEM_OBS_PARAMS,
    flags:      tuple[int, int, int] = (0, 0, 0),
) -> jax.Array:
    """Per-channel R inflation. flag=4 -> that channel predict-only."""
    inflate = jnp.array([
        _QUALITY_INFLATE[jnp.clip(flags[0], 0, 5)],
        _QUALITY_INFLATE[jnp.clip(flags[1], 0, 5)],
        _QUALITY_INFLATE[jnp.clip(flags[2], 0, 5)],
    ], dtype=jnp.float32)
    sigmas = jnp.array([
        obs_params.sigma_Hgb,
        obs_params.sigma_Hct,
        obs_params.sigma_Ferritin,
    ], dtype=jnp.float32)
    return jnp.diag((sigmas ** 2) * inflate)
