"""
app/slices/thermo_renal/observation.py  —  Thermo-Renal V2 Observation Model

Extracts 2 observable quantities from the 5-state vector.

  y[0]  Core_Temp_obs [°C]       — ingestible pill / oesophageal patch
  y[1]  Body_Mass_Drop_kg        — pre/post weigh-in proxy for total water loss
                                    (1 L ≈ 1 kg)

OBS_DIM = 2

Quality-flag convention
  flag=0 : good (weigh-in done / pill active)
  flag=4 : missing → inflate R[channel] × 1e8 (predict-only)

References
  Niedermann R. et al. (2014) Ann Occup Hyg 58(8):1000–1013   [skin/core temp error]
  Montain S.J., Coyle E.F. (1992) J Appl Physiol             [BW as fluid proxy]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.thermo_renal.ode import (
    IDX_CORE_TEMP,
    IDX_PLASMA_VOL,
    IDX_INTERS_VOL,
    OBS_DIM,
    ThermoRenalParams,
    DEFAULT_TR_PARAMS,
)


# ── Observation parameters ────────────────────────────────────────────────────

class TRObsParams(NamedTuple):
    """
    Observation model parameters for the Thermo-Renal V2 slice.

    sigma_core_temp : HalfNormal(0.3) °C   — ingestible pill / patch
    sigma_bw_drop   : HalfNormal(0.1) kg   — digital scale precision
    TBW_ref_L       : PV_ref + IV_ref [L] = 4.2 + 12.0 = 16.2 L (reference total)
    """
    sigma_core_temp: float = 0.3    # [°C]
    sigma_bw_drop:   float = 0.1    # [kg]
    TBW_ref_L:       float = 16.2   # PV_ref + IV_ref [L]


DEFAULT_TR_OBS_PARAMS: TRObsParams = TRObsParams()

# Nominal R (diagonal, 2×2)
R_TR_DEFAULT: jax.Array = jnp.diag(jnp.array([
    0.09,   # Core_Temp_obs variance [°C²]   σ = 0.3°C
    0.01,   # Body_Mass_Drop variance [kg²]   σ = 0.1 kg
], dtype=jnp.float32))


# ── Observation function ──────────────────────────────────────────────────────

@jax.jit
def h_tr(
    x:          jax.Array,
    obs_params: TRObsParams,
    tr_params:  ThermoRenalParams = DEFAULT_TR_PARAMS,
) -> jax.Array:
    """
    h(x, θ) → [Core_Temp_obs, Body_Mass_Drop_kg]

    Parameters
    ----------
    x          : (STATE_DIM=5,)
    obs_params : TRObsParams
    tr_params  : ThermoRenalParams (unused directly but kept for API parity)

    Returns
    -------
    y : (OBS_DIM=2,)
        y[0]  Core_Temp_obs [°C]
        y[1]  Body_Mass_Drop_kg [kg]   positive = dehydrated
    """
    T_core = x[IDX_CORE_TEMP]
    PV     = jnp.maximum(x[IDX_PLASMA_VOL], jnp.float32(0.1))
    IV     = jnp.maximum(x[IDX_INTERS_VOL], jnp.float32(0.1))

    # Total tracked volume; drop relative to reference
    total_vol   = PV + IV
    water_loss  = jnp.float32(obs_params.TBW_ref_L) - total_vol   # [L ≈ kg]
    bw_drop     = jnp.maximum(water_loss, jnp.float32(0.0))

    return jnp.array([T_core, bw_drop], dtype=jnp.float32)


@jax.jit
def h_tr_sigma(
    sigma_pts:  jax.Array,
    obs_params: TRObsParams,
    tr_params:  ThermoRenalParams = DEFAULT_TR_PARAMS,
) -> jax.Array:
    """Vectorised h_tr over UKF sigma points. Shape (2n+1, OBS_DIM)."""
    return jax.vmap(h_tr, in_axes=(0, None, None))(sigma_pts, obs_params, tr_params)


# ── Quality-flag-aware R inflation ───────────────────────────────────────────

def inflate_R_tr(
    quality_flags: tuple[int, int],
    R_nominal:     jax.Array = R_TR_DEFAULT,
) -> jax.Array:
    """
    Scale R diagonals by quality flag.

    quality_flags : (flag_core_temp, flag_bw_drop)
        0 — excellent (pill active / weighed)
        4 — missing   → R × 1e8 (predict-only)
    """
    scale_map = {0: 1.0, 1: 1.5, 2: 4.0, 3: 16.0, 4: 1e8}
    s0 = scale_map.get(quality_flags[0], 1e8)
    s1 = scale_map.get(quality_flags[1], 1e8)
    diag_scale = jnp.array([s0, s1], dtype=jnp.float32)
    return R_nominal * jnp.diag(diag_scale)
