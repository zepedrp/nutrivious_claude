"""
app/engine/observation/aerobic_observer.py

L4 Aerobic/HRV Observation Model — h(x, θ) → y_obs

Maps the 9-dim continuous ODE state (Mod 3 Cardiorespiratory + W'_bal from
Mod 1 Metabolic) to the 2-dim wearable observable space captured by Fase 4
sensors (Garmin/Oura/Polar/Whoop).

State x ∈ ℝ⁹
─────────────────────────────────────────────────────────────────────────────
  x[0]  NE          Norepinephrine         [adim., τ≈7 min]
  x[1]  E           Epinephrine            [adim., τ≈8 min]
  x[2]  V_vagal     Vagal tone             [adim., ∈(0,1), τ≈20 min]
  x[3]  P_a         Mean arterial pressure [mmHg,  τ≈0.3 min]
  x[4]  PaCO2       Arterial PCO2          [mmHg,  τ≈5 min]
  x[5]  PbCO2       Brain PCO2             [mmHg,  τ≈3.3 min]
  x[6]  SpO2        O2 saturation          [frac,  ∈(0,1)]
  x[7]  V_E         Ventilation            [L·min⁻¹]
  x[8]  W_prime_bal W' balance             [kJ]
─────────────────────────────────────────────────────────────────────────────

Observations y ∈ ℝ²: [HR_bpm, RMSSD_ms]

Mathematical mappings
─────────────────────────────────────────────────────────────────────────────
HR_bpm  = HR_intr − k_chron_vag · V_vagal + k_chron_cat · (NE + E)         (1)
  — Berntson et al. 1994 J Auton Nerv Syst (DOI 10.1016/0165-1838(94)90168-6)
  — SA-node intrinsic rate modified by parasympathetic withdrawal and
    sympathetic drive; linearised around the physiological operating point.

RMSSD_ms = RMSSD_ref · exp(k_rmssd · (V_vagal − 0.5))                      (2)
  — Hoshi et al. 2013 Heart (DOI 10.1136/heartjnl-2012-302085)
  — Buchheit 2014 BJSM (DOI 10.1136/bjsports-2013-093010)
  — Shaffer & Ginsberg 2017 Front. Neurosci. (DOI 10.3389/fnins.2017.00258)
  — Log-normal RMSSD distribution; exponential vagal–RMSSD relationship
    validated in endurance athletes (Plews et al. 2013 IJSPP).
─────────────────────────────────────────────────────────────────────────────

Design invariants
─────────────────────────────────────────────────────────────────────────────
  • Pure JAX — no Python-level conditionals on array values.
  • @jax.jit and jax.vmap safe (sigma-point propagation for UKF).
  • Parameters are population-level priors; NLME η_i personalises them.
  • Fail-Loud: no silent substitution — NaN propagates visibly.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── State vector indices ──────────────────────────────────────────────────────
IDX_NE        = 0   # Norepinephrine    [adim.]
IDX_E         = 1   # Epinephrine       [adim.]
IDX_V_VAGAL   = 2   # Vagal tone        [adim., ∈(0,1)]
IDX_P_A       = 3   # MAP               [mmHg]
IDX_PACO2     = 4   # Arterial PCO2     [mmHg]
IDX_PBCO2     = 5   # Brain PCO2        [mmHg]
IDX_SPO2      = 6   # SpO2              [frac]
IDX_V_E       = 7   # Ventilation       [L·min⁻¹]
IDX_W_PRIME   = 8   # W'_balance        [kJ]

STATE_DIM: int = 9
OBS_DIM:   int = 2   # [HR_bpm, RMSSD_ms]


# ── Observation parameter container ──────────────────────────────────────────

class AerobicObserverParams(NamedTuple):
    """
    Population-level parameters for h(x) observation model.

    All fields are NamedTuple leaves → valid JAX pytree (JIT-traced).
    NLME η_i shifts these per individual; the filter operates on the deviations.

    Population priors (literature-anchored):
        HR_intr      ~ N(110, 12²) bpm  — Berntson 1994 (denervated heart rate)
        k_chron_vag  ~ N(50,  10²) bpm  — Katona 1970 (vagal chronotropy range)
        k_chron_cat  ~ N(35,   7²) bpm  — Goldberger 1999 (sympathetic gain)
        RMSSD_ref    ~ LogN(ln60, 0.25²) ms — Plews 2013 endurance athlete
        k_rmssd      ~ N(2.5,  0.5²)    — Hoshi 2013 (log-linear slope)
        HR_floor     = 30 bpm (hard physiological floor)
        RMSSD_floor  = 5  ms  (artefact threshold)
    """
    HR_intr:     float   # intrinsic SA-node rate [bpm]
    k_chron_vag: float   # vagal chronotropy gain [bpm per unit V_vagal]
    k_chron_cat: float   # catecholamine chronotropy gain [bpm per (NE+E)]
    RMSSD_ref:   float   # reference RMSSD at V_vagal = 0.5 [ms]
    k_rmssd:     float   # log-linear vagal–RMSSD slope [adim.]
    HR_floor:    float   # absolute physiological HR floor [bpm]
    RMSSD_floor: float   # artefact / absolute RMSSD floor [ms]


DEFAULT_OBSERVER_PARAMS = AerobicObserverParams(
    HR_intr     = 110.0,   # Berntson 1994 — autonomically-denervated intrinsic rate
    k_chron_vag = 50.0,    # 50 bpm suppression at full vagal tone (V_vagal=1.0)
    k_chron_cat = 35.0,    # 35 bpm elevation per unit catecholamine
    RMSSD_ref   = 60.0,    # ms; endurance athlete population mean (Plews 2013)
    k_rmssd     = 2.5,     # calibrated: RMSSD~25ms at V_vagal=0.2 (post-exercise)
    HR_floor    = 30.0,    # bpm; absolute physiological minimum
    RMSSD_floor = 5.0,     # ms; below this = likely artefact
)


# ── Core observation function h(x, θ) ────────────────────────────────────────

@jax.jit
def h_observer(
    x: jax.Array,
    params: AerobicObserverParams,
) -> jax.Array:
    """
    Pure observation function h: ℝ⁹ → ℝ².

    JIT-compiled and vmap-safe. No Python-level branching on array values.
    Uses jnp.maximum / jnp.clip exclusively (smooth, differentiable).

    Parameters
    ----------
    x      : shape (STATE_DIM,) = (9,) — aerobic ODE state
    params : AerobicObserverParams

    Returns
    -------
    y : shape (OBS_DIM,) = (2,) — [HR_bpm, RMSSD_ms]

    Notes
    -----
    Equation (1):  HR = HR_intr − k_chron_vag·V_vagal + k_chron_cat·(NE+E)
    Equation (2):  RMSSD = RMSSD_ref · exp(k_rmssd · (V_vagal − 0.5))

    V_vagal is clamped to [0.01, 0.99] before exp() to prevent overflow/underflow
    in the exponent. The clamp is JIT-safe (no Python branching).
    """
    NE      = x[IDX_NE]
    E       = x[IDX_E]
    V_vagal = x[IDX_V_VAGAL]

    # (1) Heart rate — autonomic balance (Berntson 1994; Levy & Martin 1979)
    HR_raw = (
        params.HR_intr
        - params.k_chron_vag * V_vagal
        + params.k_chron_cat * (NE + E)
    )
    HR_bpm = jnp.maximum(params.HR_floor, HR_raw)

    # (2) RMSSD — log-linear vagal model (Hoshi 2013; Buchheit 2014)
    #   V_vagal = 0.5 → RMSSD = RMSSD_ref  (reference resting midpoint)
    #   V_vagal → 1.0 → RMSSD increases  (high parasympathetic = high HRV)
    #   V_vagal → 0.0 → RMSSD decreases  (sympathetic dominance = low HRV)
    V_c       = jnp.clip(V_vagal, 0.01, 0.99)
    RMSSD_raw = params.RMSSD_ref * jnp.exp(params.k_rmssd * (V_c - 0.5))
    RMSSD_ms  = jnp.maximum(params.RMSSD_floor, RMSSD_raw)

    return jnp.stack([HR_bpm, RMSSD_ms])


def h_observer_sigma(
    sigma_pts: jax.Array,
    params: AerobicObserverParams,
) -> jax.Array:
    """
    Vectorised observer for UKF sigma-point propagation.

    Applies h_observer to each row of sigma_pts via jax.vmap.
    No Python loop — a single fused JIT kernel for all 2n+1 points.

    Parameters
    ----------
    sigma_pts : shape (n_sigma, STATE_DIM) = (2n+1, 9)
    params    : AerobicObserverParams

    Returns
    -------
    y_sigma : shape (n_sigma, OBS_DIM) = (2n+1, 2)
    """
    return jax.vmap(h_observer, in_axes=(0, None))(sigma_pts, params)
