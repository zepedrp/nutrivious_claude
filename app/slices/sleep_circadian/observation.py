"""
app/slices/sleep_circadian/observation.py  —  Observation model V2.0

Maps the 5-dim state to 2 wearable-observable signals.

Observation vector y ∈ ℝ²
──────────────────────────
  y[0]  SWS_Duration_proxy  —  slow-wave sleep fraction [0–1]
                                from smart-ring motion+HRV (Oura/Garmin)
  y[1]  CBT_Nadir_Phase     —  circadian phase proxy [rad, −π..π]
                                inferred from distal skin temperature nadir
                                (Oura Ring 4 / RCT skin-temp feature)

Observation function h(x, θ):
  y[0] = x[IDX_SWS]                               + ε_sws
  y[1] = arctan2(x[IDX_SCN_Y], x[IDX_SCN_X])     + ε_phase

Noise model (diagonal R):
  σ_sws   = 0.15  (wearable SWS fraction vs PSG MAE ~0.12–0.18; Chinoy 2021)
  σ_phase = 0.30  rad  (wrist-temp circadian phase vs DLMO assay ~0.25–0.35 rad;
                         Rüger 2018 J Sleep Res; corresponds to ~1.1 h SD at 24 h period)

Quality-flag R inflation (canonical):
  flag 0  → R_nominal
  flag 1  → ×2
  flag 2  → ×4
  flag 3  → ×10
  flag 4  → ×1e8  (missing — predict-only)

References
──────────
  Chinoy E.D. et al. (2021) Nat Sci Sleep 13:285–296  (Oura/Garmin/Apple PSG validation)
  Rüger M. et al. (2018) J Sleep Res 27(2):e12581  (wrist temp vs DLMO)
  Casiraghi L.P. et al. (2021) Curr Biol 31(9):1837–1847  (smartphone circadian phase)
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from app.slices.sleep_circadian.ode import (
    IDX_SCN_Y,
    IDX_SCN_X,
    IDX_SWS,
    STATE_DIM,
    OBS_DIM,
)

# ── Observation parameters ─────────────────────────────────────────────────────

class SleepObsParams(NamedTuple):
    """
    Observation noise standard deviations (fixed at population level).

    sigma_sws    [frac]  Chinoy 2021 — wearable SWS fraction vs PSG
    sigma_phase  [rad]   Rüger 2018  — skin-temp nadir vs DLMO assay
    """
    sigma_sws:   float = 0.15   # fraction
    sigma_phase: float = 0.30   # rad


DEFAULT_OBS_PARAMS = SleepObsParams()

# Quality-flag R inflation multipliers (indices 0..5; clamp to 0..5)
_QUALITY_INFLATE = jnp.array([1.0, 2.0, 4.0, 10.0, 1e8, 1e8])


# ── Observation function ───────────────────────────────────────────────────────

def h_sleep(
    x: jnp.ndarray,
    obs_params: SleepObsParams = DEFAULT_OBS_PARAMS,
) -> jnp.ndarray:
    """
    h(x) → y ∈ ℝ²  — JAX-traceable, no data-dependent control flow.

    y[0] = Sleep_Drive_SWS  (direct observation proxy)
    y[1] = arctan2(SCN_y, SCN_x)  (circadian phase proxy)
    """
    sws_proxy   = x[IDX_SWS]
    phase_proxy = jnp.arctan2(x[IDX_SCN_Y], x[IDX_SCN_X])
    return jnp.array([sws_proxy, phase_proxy])


# ── Noise covariance ───────────────────────────────────────────────────────────

def observation_noise_R(
    obs_params: SleepObsParams = DEFAULT_OBS_PARAMS,
    quality_flag: int = 0,
) -> jnp.ndarray:
    """
    Diagonal R matrix [OBS_DIM, OBS_DIM] with quality-flag inflation.

    quality_flag 0 → nominal; 4 → ×1e8 (predict-only).
    """
    inflate = _QUALITY_INFLATE[jnp.clip(quality_flag, 0, 5)]
    sigmas  = jnp.array([obs_params.sigma_sws, obs_params.sigma_phase])
    return jnp.diag((sigmas ** 2) * inflate)
