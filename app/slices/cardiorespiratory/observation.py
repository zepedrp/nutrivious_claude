"""
app/slices/cardiorespiratory/observation.py

L2/L4 Observation Model — Cardiorespiratory Slice
h(x, θ) → y = [HR_obs_bpm, VO2_obs_mLkgmin, RMSSD_obs_ms]

Maps the 6-dim cardiorespiratory state vector to the three wearable-observable
signals used by the UKF to assimilate real-world data.

Observation equations
─────────────────────
y[0]  HR_obs  = Heart_Rate + ε_HR        ε_HR  ~ N(0, σ²_HR)   [bpm]
y[1]  VO2_obs = V_O2       + ε_VO2       ε_VO2 ~ N(0, σ²_VO2)  [mL/kg/min]
y[2]  RMSSD   = RMSSD_ref × Autonomic_Tone + ε_AT
                                           ε_AT  ~ N(0, σ²_AT)  [ms]

The RMSSD model is physiologically grounded: vagal efferent activity directly
drives RR-interval variability (Malik 1996), and Autonomic_Tone (∈ [0,1]) tracks
the vagal component. RMSSD_ref is the population-level RMSSD at full vagal tone
(35 ms mean; Malik et al. 1996 Eur Heart J).

Missing-observation protocol (quality_flag R inflation)
────────────────────────────────────────────────────────
VO2 and RMSSD are typically sparse (metabolic cart → intermittent; RMSSD →
morning-only). HR is dense (beat-to-beat wearable). When a signal is absent
(NaN in the observation dict), R is inflated by 1e8 for that channel, making
the innovation effectively zero — a pure predict-only step for that dimension.
This follows the HLD §4.3 missing-observation handling pattern.

Quality-flag scaling (HLD §5.2):
    quality_flag = 0 → R × 1.0  (clean signal)
    quality_flag = 1 → R × 2.0  (minor artefact)
    quality_flag = 2 → R × 10.0 (interpolated)
    quality_flag = 3 → R × 100. (poor quality)
    quality_flag = 4 → R × 1e8  (missing / invalid)

References
──────────
  Malik M. et al. (1996) Eur Heart J 17:354-381     [RMSSD population reference]
  Plews D.J. et al. (2013) Int J Sports Physiol Perform 8:641–645 [RMSSD CV]
  Buchheit M. (2014) Int J Sports Physiol Perform 9:701-14        [HRV monitoring]

Design invariants
─────────────────
  • Pure JAX: @jax.jit and jax.vmap safe (sigma-point propagation).
  • NaN propagates — no silent substitution.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.cardiorespiratory.ode import (
    IDX_HR,
    IDX_VO2,
    IDX_AT,
    STATE_DIM,
    OBS_DIM,
    DEFAULT_CARDIO_SLICE_PARAMS,
)

# ── Quality-flag R inflation table ────────────────────────────────────────────

_QUALITY_INFLATE: tuple[float, ...] = (1.0, 2.0, 10.0, 100.0, 1.0e8)


def inflate_R_for_quality(quality_flag: int, R_base: jax.Array) -> jax.Array:
    """
    Scale the observation noise matrix for the overall session quality flag.

    quality_flag ∈ [0, 4]; flag=4 → 1e8× (predict-only; no update).
    """
    flag  = max(0, min(4, quality_flag))
    scale = _QUALITY_INFLATE[flag]
    return R_base * jnp.float32(scale)


# ── Observation parameter container ──────────────────────────────────────────

class CardioObsParams(NamedTuple):
    """
    Observation model parameters for the cardiorespiratory slice.

    Population-level defaults anchored to literature (personalised
    by NLME if wearable calibration data is available).

    Fields
    ------
    sigma_HR_bpm    : float — HR monitor additive noise SD [bpm]
                      Typical optical PPG σ ≈ 3 bpm (Kristiansen 2011)
    sigma_VO2       : float — metabolic cart / indirect calorimetry noise [mL/kg/min]
                      Typical σ ≈ 2 mL/kg/min (Macfarlane 2001)
    sigma_RMSSD_ms  : float — RMSSD measurement noise SD [ms]
                      Plews 2013 reports 5-day CV ≈ 8–15% of mean;
                      at RMSSD_ref = 35 ms → σ ≈ 5 ms
    RMSSD_ref_ms    : float — RMSSD at full vagal tone (Autonomic_Tone = 1.0)
                      Malik 1996 population mean: 35 ms
    """
    sigma_HR_bpm:   float = 3.0    # bpm — PPG HR monitor noise
    sigma_VO2:      float = 2.0    # mL/kg/min — metabolic cart noise
    sigma_RMSSD_ms: float = 5.0    # ms — RMSSD HRV monitor noise
    RMSSD_ref_ms:   float = 35.0   # ms — RMSSD at AT = 1.0 (Malik 1996)


DEFAULT_OBS_PARAMS = CardioObsParams(
    sigma_HR_bpm   = 3.0,
    sigma_VO2      = 2.0,
    sigma_RMSSD_ms = 5.0,
    RMSSD_ref_ms   = DEFAULT_CARDIO_SLICE_PARAMS.RMSSD_ref_ms,
)

# ── Default R matrix (diagonal) ───────────────────────────────────────────────

_p = DEFAULT_OBS_PARAMS
R_DEFAULT: jax.Array = jnp.diag(jnp.array([
    _p.sigma_HR_bpm   ** 2,   # HR    [bpm²]
    _p.sigma_VO2      ** 2,   # VO2   [(mL/kg/min)²]
    _p.sigma_RMSSD_ms ** 2,   # RMSSD [ms²]
], dtype=jnp.float32))


# ── Observation functions (JIT + vmap safe) ───────────────────────────────────

@jax.jit
def h_cardio(
    x:          jax.Array,
    obs_params: CardioObsParams,
) -> jax.Array:
    """
    Observation function h(x, θ): state → observable signals.

    Parameters
    ----------
    x          : shape (STATE_DIM,) — current state
    obs_params : CardioObsParams — noise / reference parameters

    Returns
    -------
    y : shape (OBS_DIM,) = [HR_bpm, VO2_mLkgmin, RMSSD_ms]

    Notes
    -----
    RMSSD is nonlinear in the state (product of RMSSD_ref and AT) which gives
    the UKF a meaningful cross-covariance between AT and the HRV observation.
    """
    HR    = x[IDX_HR]
    VO2   = x[IDX_VO2]
    AT    = x[IDX_AT]
    RMSSD = jnp.float32(obs_params.RMSSD_ref_ms) * AT
    return jnp.stack([HR, VO2, RMSSD])


def h_cardio_sigma(
    sigma_pts:  jax.Array,
    obs_params: CardioObsParams,
) -> jax.Array:
    """
    Apply h_cardio to all 2n+1 sigma points.

    Parameters
    ----------
    sigma_pts  : shape (2n+1, STATE_DIM)
    obs_params : CardioObsParams

    Returns
    -------
    y_sigma : shape (2n+1, OBS_DIM)
    """
    return jax.vmap(h_cardio, in_axes=(0, None))(sigma_pts, obs_params)


# ── Missing-observation handling ──────────────────────────────────────────────

def obs_dict_to_array(obs_dict: dict[str, float]) -> jax.Array:
    """
    Convert observation dict to a (OBS_DIM,) JAX array.

    Missing signals must be passed as float('nan') or omitted.
    NaN values trigger R×1e8 inflation in the UKF update step.

    Keys
    ----
    "HR_obs_bpm"   : float — heart rate observation [bpm]
    "VO2_obs"      : float — VO2 observation [mL/kg/min]; optional
    "RMSSD_obs_ms" : float — RMSSD observation [ms]; optional
    """
    hr    = float(obs_dict.get("HR_obs_bpm",   float("nan")))
    vo2   = float(obs_dict.get("VO2_obs",      float("nan")))
    rmssd = float(obs_dict.get("RMSSD_obs_ms", float("nan")))
    return jnp.array([hr, vo2, rmssd], dtype=jnp.float32)


def inflate_R_per_channel(
    R_base: jax.Array,
    y_obs:  jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Inflate R per-channel for NaN observations and return a safe y_obs.

    NaN observation → R channel inflated ×1e8; y_obs channel replaced by
    the predicted value (zero innovation ≡ predict-only for that channel).

    Parameters
    ----------
    R_base : (OBS_DIM, OBS_DIM) — baseline observation noise
    y_obs  : (OBS_DIM,) — raw observations (NaN for missing)

    Returns
    -------
    (R_inflated, y_safe) : (OBS_DIM, OBS_DIM), (OBS_DIM,)
        y_safe  — NaN → 0 placeholder (overwritten by predict-only in update)
        R_inflated — per-channel scaling
    """
    nan_mask = jnp.isnan(y_obs)                                # (OBS_DIM,) bool
    scale    = jnp.where(nan_mask, jnp.float32(1e8), jnp.float32(1.0))
    R_inflated = R_base * jnp.diag(scale)
    # Replace NaN with 0 so downstream arithmetic doesn't propagate NaN
    y_safe = jnp.where(nan_mask, jnp.float32(0.0), y_obs)
    return R_inflated, y_safe
