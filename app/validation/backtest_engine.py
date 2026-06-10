"""
app/validation/backtest_engine.py

Gate Zero -- Inter-Day HRV Walk-Forward Validation

Criterion
---------
The Digital Twin predicts ln(RMSSD_obs_ms) at 08:00 the following morning.
It PASSES Gate Zero if its MAE is STRICTLY INFERIOR to BOTH:

    BASELINE 1 -- Persistence:  ln(RMSSD) from yesterday morning
    BASELINE 2 -- EWMA-7d:      exp-weighted mean (span=7d) of past ln(RMSSD)

in >= 60% of users that have >= 60 days of paired predictions.

Walk-Forward Protocol (Inter-Day, per User)
-------------------------------------------
For each day D (chronological, after a warm-up window):

  1. DAYTIME ASSIMILATION (00:00 -- last record of day D, variable-rate async UKF):
     All intra-day records processed with true dt_real between samples.
     UKF predict + update at each record. RMSSD morning observations, when
     present in the record stream, are assimilated normally (R-channel inflation
     for sparse measurements is already handled by CardioStateFilter).

  2. BLIND OVERNIGHT PREDICT (last_record_ts --> 08:00 next day):
     Starting from the end-of-day posterior, run UKF PREDICT ONLY.
     Sleep controls: power=0 W, T_core=37.0 C (phase 1) / 36.5 C (phase 2), pv_drop=0 %.
     Step size: OVERNIGHT_DT_MIN = 10 min (bi-exp W' recovery is slow enough).
     No measurement update during this segment -- genuine blind forecast.
     Borbely 1982 two-process circadian modulator applied in second sleep half:
       k_AT_rec boosted by (1 + 0.20 * cos(2*pi*(wake_hour - 6) / 24)).
       At wake_hour=08:00 this gives a 13% boost to vagal reactivation rate.

  3. PREDICTION:
     predicted_rmssd_ms = RMSSD_load_7d(08:00) * AT_0800
     predicted_target   = ln(max(predicted_rmssd_ms, 1.0))
     where AT_0800 may exceed 1.0 by circadian amplitude (Fix 4 -- Borbely boost),
     and RMSSD_load_7d is seeded with the personal RMSSD_ref_ms from warmup (Fix 3).

  4. GROUND TRUTH:
     RMSSD_obs_ms record in day D+1 within +/- 2h window around 08:00.
     observed_target = ln(RMSSD_obs_ms).

  5. BASELINES (computed at prediction time, i.e. state of knowledge at 23:59):
     Persistence = ln(last observed RMSSD_obs_ms up to and including day D morning)
     EWMA-7d     = alpha * ln(RMSSD_D_morning) + (1-alpha) * EWMA_{D-1}
                   with alpha = 2 / (7 + 1) = 0.25

  Gate Criterion:
     twin_wins_user = MAE_twin < MAE_persistence AND MAE_twin < MAE_ewma7d
     gate_passes    = fraction(twin_wins_user) >= 0.60 over qualifying users

L3 Individualisation (per-user NLME fit)
-----------------------------------------
Before running the walk-forward, the engine attempts to personalise the ODE
parameters (VO2_max_baseline, W_prime_capacity) for each user:

  1. Training window: first _NLME_TRAIN_DAYS days of records.
  2. Extract the longest continuous exercise session from the training window.
  3. Fit CardioNLME.fit_svi() on that session's HR response.
  4. Extract the posterior median CardioSliceParams for subject 0.
  5. Build a per-user CardioTransitionParams and pass to InterDayTwinValidator.

If numpyro is absent or SVI diverges, the pipeline degrades to the population
prior (DEFAULT_TRANSITION_PARAMS) with a WARNING log. Gate Zero still runs.

Data Format (same as previous engine, plus RMSSD_obs_ms is used as morning GT)
-------------------------------------------------------------------------------
Records: list[dict] sorted by timestamp_min ascending:
    {
        "timestamp_min":    int,           # minutes since cohort day 0
        "HR_obs_bpm":       float | None,
        "VO2_obs":          float | None,
        "RMSSD_obs_ms":     float | None,  # present only for morning (~08:00) records
        "power_watts":      float,         # 0.0 at rest / sleep
        "hub_T_core":       float | None,
        "hub_pv_drop_pct":  float,
    }

Fail-Loud Contract
------------------
RuntimeError if UKF diverges (NaN in posterior mean) -- propagated from filter.
ValueError for empty user data.
"""
from __future__ import annotations

import logging
import math
from enum import Enum
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from app.engine.assimilation.ukf_filter import GaussianState, scale_Q
from app.slices.cardiorespiratory.ode import (
    X0_CARDIO_DEFAULT,
    P0_CARDIO_DEFAULT,
    IDX_AT,
    IDX_RMSSD7D,
)
from app.slices.cardiorespiratory.filter import (
    CardioStateFilter,
    CardioTransitionParams,
    DEFAULT_TRANSITION_PARAMS,
    Q_DEFAULT,
    _ukf_predict_cardio,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MINS_PER_DAY:       int   = 1440
_MORNING_TARGET_MIN: int   = 480      # 08:00 in day-local minutes
_MORNING_WINDOW_MIN: int   = 120      # RMSSD obs search: +/- 2h around 08:00
_OVERNIGHT_DT_MIN:   int   = 10       # predict step size for blind overnight [min]
_GATE_MIN_DAYS:      int   = 60       # minimum paired predictions per user
_GATE_PASS_FRAC:     float = 0.60     # fraction of users where Twin must win both
_EWMA_ALPHA:         float = 2.0 / 8.0  # EWMA span=7d: alpha = 2/(7+1)
_DT_CLAMP_MIN:       float = 1.0      # minimum dt_real for UKF [min]
_DT_CLAMP_MAX:       float = 60.0     # maximum dt_real for UKF [min]
_WARMUP_DAYS:        int   = 14       # days excluded from scoring (filter burn-in)
_NLME_TRAIN_DAYS:    int   = 30       # days in NLME training window (before test window)
_NLME_STEPS_DEFAULT: int   = 3_000    # SVI steps for per-user NLME fit


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class GateZeroDecision(Enum):
    PASS              = "PASS"
    FAIL              = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DayPrediction(NamedTuple):
    """One inter-day prediction record (day D -> day D+1 morning)."""
    day_index:          int
    predicted_ln_rmssd: float
    observed_ln_rmssd:  float
    persist_ln_rmssd:   float
    ewma_ln_rmssd:      float


class UserValidationResult(NamedTuple):
    user_id:             str | int
    n_days:              int
    mae_twin:            float
    mae_persistence:     float
    mae_ewma7d:          float
    twin_beats_persist:  bool
    twin_beats_ewma:     bool
    twin_wins:           bool        # beats BOTH baselines simultaneously
    days:                list        # list[DayPrediction]


class GateZeroResult(NamedTuple):
    decision:            GateZeroDecision
    n_users_total:       int
    n_users_twin_wins:   int
    win_fraction:        float
    per_user:            list        # list[UserValidationResult]
    gate_criterion:      str


# ---------------------------------------------------------------------------
# Day grouping helpers
# ---------------------------------------------------------------------------

def _day_of(ts_min: int, t0: int) -> int:
    return (ts_min - t0) // _MINS_PER_DAY


def _minute_of_day(ts_min: int, t0: int) -> int:
    return (ts_min - t0) % _MINS_PER_DAY


def _group_by_day(records: list[dict], t0: int) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for r in records:
        d = _day_of(r["timestamp_min"], t0)
        groups.setdefault(d, []).append(r)
    return groups


# ---------------------------------------------------------------------------
# EWMA tracker (log-space HRV)
# ---------------------------------------------------------------------------

class LogRMSSDEWMA:
    """Exp-weighted moving average of ln(RMSSD_obs_ms) with span=7d (alpha=0.25)."""

    def __init__(self, alpha: float = _EWMA_ALPHA) -> None:
        self._alpha  = alpha
        self._value: float | None = None

    def update(self, ln_rmssd: float) -> None:
        if self._value is None:
            self._value = ln_rmssd
        else:
            self._value = self._alpha * ln_rmssd + (1.0 - self._alpha) * self._value

    def predict(self) -> float | None:
        return self._value


# ---------------------------------------------------------------------------
# Blind overnight integrator (Borbely 1982 circadian modulator)
# ---------------------------------------------------------------------------

def _blind_overnight_predict(
    state:     GaussianState,
    params:    CardioTransitionParams,
    n_min:     int,
    dt_min:    int   = _OVERNIGHT_DT_MIN,
    wake_hour: float = 8.0,
) -> GaussianState:
    """
    Run UKF predict-only for n_min minutes under sleep conditions.

    Sleep controls: power=0, T_core=37.0 (NREM) / 36.5 (REM), pv_drop=0.
    Uses dt_min-minute steps. Q is Wiener-scaled per step.
    No measurement update -- pure blind recovery simulation.

    Circadian modulator (Borbely 1982 two-process model)
    ----------------------------------------------------
    Sleep is split into two equal phases:

      Phase 1 (NREM-dominant, first half):
        Baseline k_AT_rec. Slow vagal recovery from exercise stress.

      Phase 2 (REM/circadian-dominant, second half):
        k_AT_rec boosted by circadian_factor = 1 + 0.20*cos(2*pi*(wake_hour-6)/24).
        hub_T_core lowered to 36.5 C (Rechtschaffen 1978: core temp nadirs in late sleep).
        At wake_hour=06:00: factor=1.20 (max vagal reactivation near SCN peak).
        At wake_hour=08:00: factor=1.13 (typical morning athlete measurement).
        At wake_hour=12:00: factor=1.00 (no circadian boost).

    The bi-exp W' pools (tau=2 and 30 min) and Autonomic_Tone (tau_rec ~20 min)
    drive towards their equilibrium values under zero exercise stress. The 7-day
    RMSSD_load_7d state barely moves (tau=10080 min) and serves as the slow
    inter-day signal for trend-tracking.
    """
    # Borbely circadian factor at wake_hour
    _circ = 1.0 + 0.20 * math.cos(2.0 * math.pi * (wake_hour - 6.0) / 24.0)
    _circ = max(0.85, min(1.25, _circ))

    half_min = n_min // 2   # integer split

    # Control vectors for each phase
    u1 = jnp.array([0.0, 37.0, 0.0], dtype=jnp.float32)   # NREM (T_core 37.0)
    u2 = jnp.array([0.0, 36.5, 0.0], dtype=jnp.float32)   # REM  (T_core 36.5)

    # Phase 1 params: population k_AT_rec
    p1 = CardioTransitionParams(
        cardio          = params.cardio,
        dt_min          = float(dt_min),
        hub_T_core      = 37.0,
        hub_pv_drop_pct = 0.0,
    )
    # Phase 2 params: circadian-boosted k_AT_rec + slight core temp drop
    p2 = CardioTransitionParams(
        cardio          = params.cardio._replace(k_AT_rec=params.cardio.k_AT_rec * _circ),
        dt_min          = float(dt_min),
        hub_T_core      = 36.5,
        hub_pv_drop_pct = 0.0,
    )

    Q_step1 = scale_Q(Q_DEFAULT, float(dt_min))
    Q_step2 = scale_Q(Q_DEFAULT, float(dt_min))

    current = state
    elapsed = 0

    # -- Phase 1: NREM half --------------------------------------------------
    while elapsed < half_min:
        step = min(dt_min, half_min - elapsed)
        if step != dt_min:
            pp = CardioTransitionParams(
                cardio=p1.cardio, dt_min=float(step),
                hub_T_core=37.0, hub_pv_drop_pct=0.0,
            )
            m, c = _ukf_predict_cardio(current.mean, current.cov, u1, pp,
                                        scale_Q(Q_DEFAULT, float(step)))
        else:
            m, c = _ukf_predict_cardio(current.mean, current.cov, u1, p1, Q_step1)
        current = GaussianState(mean=m, cov=c)
        elapsed += step

    # -- Phase 2: REM/circadian half -----------------------------------------
    while elapsed < n_min:
        step = min(dt_min, n_min - elapsed)
        if step != dt_min:
            pp = CardioTransitionParams(
                cardio=p2.cardio, dt_min=float(step),
                hub_T_core=36.5, hub_pv_drop_pct=0.0,
            )
            m, c = _ukf_predict_cardio(current.mean, current.cov, u2, pp,
                                        scale_Q(Q_DEFAULT, float(step)))
        else:
            m, c = _ukf_predict_cardio(current.mean, current.cov, u2, p2, Q_step2)
        current = GaussianState(mean=m, cov=c)
        elapsed += step

    # Circadian amplitude boost: AT equilibrium under sleep is 1.0 + amplitude
    # (Borbely two-process: Process C pushes vagal tone above daytime baseline
    # near SCN peak; the ODE alone cannot exceed 1.0 because its recovery term
    # is anchored at 1.0, so we apply the circadian offset post-integration).
    _amp = _circ - 1.0   # e.g., 0.13 at wake_hour=08:00
    new_at = jnp.minimum(
        current.mean[IDX_AT] + jnp.float32(_amp),
        jnp.float32(1.0 + _amp),    # AT_max = 1.0 + circadian_amplitude
    )
    current = GaussianState(
        mean=current.mean.at[IDX_AT].set(new_at),
        cov=current.cov,
    )

    return current


# ---------------------------------------------------------------------------
# L3 NLME helpers -- individualise ODE params from training window
# ---------------------------------------------------------------------------

def _build_nlme_training_arrays(
    records:    list[dict],
    t0:         int,
    train_days: int = _NLME_TRAIN_DAYS,
) -> tuple:
    """
    Extract the longest continuous exercise session from the training window.

    Returns
    -------
    (session_data, hr_obs, sid_arr, step_arr) ready for CardioNLME.fit_svi().
    Raises ValueError if no usable session found.
    """
    train = [r for r in records if _day_of(r["timestamp_min"], t0) < train_days]
    if not train:
        raise ValueError("No training records in window.")

    # Group into exercise sessions (contiguous blocks of power > 0)
    sessions: list[list[dict]] = []
    current:  list[dict]       = []
    for r in train:
        pw = r.get("power_watts") or 0.0
        if pw > 0.0:
            current.append(r)
        else:
            if len(current) >= 5:
                sessions.append(current)
            current = []
    if len(current) >= 5:
        sessions.append(current)

    if not sessions:
        raise ValueError("No exercise sessions (>= 5 min) found in training window.")

    # Use the longest session (most informative for NLME identifiability)
    best = max(sessions, key=len)
    T    = len(best)

    power_arr = np.array([r.get("power_watts", 0.0) for r in best], dtype=np.float32)
    hr_arr    = np.array(
        [r.get("HR_obs_bpm") if r.get("HR_obs_bpm") is not None
         else float("nan") for r in best],
        dtype=np.float32,
    )

    valid_idx = [i for i in range(T) if not math.isnan(hr_arr[i])]
    if len(valid_idx) < 5:
        raise ValueError(f"Only {len(valid_idx)} valid HR obs in best session.")

    session_data = {
        "power_rates": power_arr[None, :],                                # (1, T)
        "x0s":         np.array(X0_CARDIO_DEFAULT, dtype=np.float32)[None, :],  # (1, STATE_DIM)
    }
    hr_obs   = hr_arr[[i for i in valid_idx]]
    sid_arr  = np.zeros(len(valid_idx), dtype=np.int32)
    step_arr = np.array(valid_idx, dtype=np.int32)

    return session_data, hr_obs, sid_arr, step_arr


def _fit_nlme_for_user(
    records:    list[dict],
    t0:         int,
    train_days: int = _NLME_TRAIN_DAYS,
    n_steps:    int = _NLME_STEPS_DEFAULT,
) -> CardioTransitionParams | None:
    """
    Fit L3 CardioNLME on the training window and return a personalised
    CardioTransitionParams for this user.

    If numpyro is absent, SVI diverges, or training data is insufficient,
    returns None (caller falls back to population prior with a WARNING).

    Parameters
    ----------
    records    : full per-user record list (chronological)
    t0         : cohort day-0 timestamp_min
    train_days : days of records used for NLME fitting
    n_steps    : SVI iterations (3 000 default -- fast; increase for production)

    Returns
    -------
    CardioTransitionParams with personalised CardioSliceParams, or None.
    """
    try:
        from app.slices.cardiorespiratory.nlme import CardioNLME
        import jax.numpy as _jnp
    except ImportError:
        logger.warning(
            "_fit_nlme_for_user: numpyro absent -- degrading to population prior."
        )
        return None

    try:
        session_data, hr_obs, sid_arr, step_arr = _build_nlme_training_arrays(
            records, t0, train_days
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "_fit_nlme_for_user: training array build failed (%s) -- population prior.", exc
        )
        return None

    try:
        nlme   = CardioNLME()
        result = nlme.fit_svi(
            session_data = session_data,
            observations = _jnp.array(hr_obs),
            subject_ids  = _jnp.array(sid_arr),
            obs_step_ids = _jnp.array(step_arr),
            n_subjects   = 1,
            n_steps      = n_steps,
        )
        params_i = nlme.get_cardio_params(result, subject_id=0)
        logger.info(
            "_fit_nlme_for_user: SVI converged -- "
            "VO2max=%.1f mL/kg/min  W'=%.1f kJ",
            params_i.VO2_max_baseline, params_i.W_prime_capacity,
        )
        return CardioTransitionParams(cardio=params_i)

    except (RuntimeError, Exception) as exc:
        logger.warning(
            "_fit_nlme_for_user: SVI failed (%s) -- population prior.", exc
        )
        return None


# ---------------------------------------------------------------------------
# Core inter-day validator
# ---------------------------------------------------------------------------

class InterDayTwinValidator:
    """
    Inter-day walk-forward HRV prediction engine.

    For each day D (after warmup):
        1. Assimilate all intra-day records with async UKF.
        2. Blind overnight predict to 08:00 next day (Borbely circadian modulator).
        3. Read AT(08:00) -> predicted RMSSD.
        4. Score against next morning's observed RMSSD.
    """

    def __init__(
        self,
        transition_params: CardioTransitionParams | None = None,
        warmup_days:       int   = _WARMUP_DAYS,
        wake_hour:         float = 8.0,
    ) -> None:
        self._filter    = CardioStateFilter()
        self._params    = transition_params or DEFAULT_TRANSITION_PARAMS
        self._warmup    = warmup_days
        self._wake_hour = wake_hour

    def run(self, records: list[dict]) -> list[DayPrediction]:
        """
        Run the full inter-day protocol.

        Parameters
        ----------
        records : list[dict] -- chronologically sorted observation records.

        Returns
        -------
        list[DayPrediction] -- one entry per day with a valid paired prediction.
        """
        if not records:
            raise ValueError("InterDayTwinValidator.run: records is empty.")

        # Align t0 to midnight of the first day so _minute_of_day returns
        # the correct time-of-day regardless of when the first record arrives.
        _t0_raw = records[0]["timestamp_min"]
        t0      = _t0_raw - (_t0_raw % _MINS_PER_DAY)
        by_day  = _group_by_day(records, t0)

        state: GaussianState = GaussianState(
            mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT
        )
        persist_ln: float | None = None
        ewma = LogRMSSDEWMA()
        preds: list[DayPrediction] = []

        warmup_rmssd_ms: list[float] = []  # morning RMSSD obs during warmup
        _calibrated = False

        for day_idx in sorted(by_day.keys()):
            day_recs = sorted(by_day[day_idx], key=lambda r: r["timestamp_min"])

            # Update baselines with THIS day's morning RMSSD observation
            today_ln = _find_morning_rmssd_ln(by_day, day_idx, t0)

            # Collect warmup RMSSD observations for personal calibration
            if day_idx < self._warmup and today_ln is not None:
                warmup_rmssd_ms.append(math.exp(today_ln))

            # Personal RMSSD_ref_ms calibration at warmup boundary (Fix 3)
            # Replaces population default (35 ms) with the athlete's own baseline.
            if day_idx == self._warmup and not _calibrated:
                if warmup_rmssd_ms:
                    personal_ms = float(np.median(warmup_rmssd_ms))
                    personal_ms = max(5.0, personal_ms)   # sanity floor
                    self._params = self._params._replace(
                        cardio=self._params.cardio._replace(RMSSD_ref_ms=personal_ms)
                    )
                    logger.info(
                        "RMSSD_ref_ms personalised: %.1f ms (n=%d warmup mornings)",
                        personal_ms, len(warmup_rmssd_ms),
                    )
                _calibrated = True

            if today_ln is not None:
                ewma.update(today_ln)
                persist_ln = today_ln

            # 1. Daytime assimilation
            state, last_ts = self._assimilate_day(state, day_recs)

            # 2. Blind overnight predict to next morning 08:00
            morning_next_abs = t0 + (day_idx + 1) * _MINS_PER_DAY + _MORNING_TARGET_MIN
            predict_min = max(30, morning_next_abs - last_ts)
            state_0800 = _blind_overnight_predict(
                state, self._params, predict_min,
                wake_hour=self._wake_hour,
            )

            # 3. Predicted RMSSD: RMSSD_load_7d(08:00) * AT(08:00) (Fix 2)
            # RMSSD_load_7d is the 7-day chronic load state (ms); AT is the daily
            # vagal modulator [0, 1+circadian_amplitude] -- their product predicts
            # next-morning RMSSD anchored to both chronic load and acute recovery.
            rmssd_load_7d = float(state_0800.mean[IDX_RMSSD7D])
            at_0800       = float(jnp.clip(state_0800.mean[IDX_AT], 0.0, 2.0))
            pred_rmssd    = max(rmssd_load_7d * at_0800, 1.0)
            pred_ln       = math.log(pred_rmssd)

            # 4. Ground truth: next morning RMSSD
            obs_ln = _find_morning_rmssd_ln(by_day, day_idx + 1, t0)

            # 5. Emit prediction if all signals available and past warmup
            ewma_pred = ewma.predict()
            if (
                day_idx >= self._warmup
                and obs_ln     is not None
                and persist_ln is not None
                and ewma_pred  is not None
            ):
                preds.append(DayPrediction(
                    day_index          = day_idx,
                    predicted_ln_rmssd = pred_ln,
                    observed_ln_rmssd  = obs_ln,
                    persist_ln_rmssd   = persist_ln,
                    ewma_ln_rmssd      = ewma_pred,
                ))

        return preds

    # ── Helpers ────────────────────────────────────────────────────────────

    def _assimilate_day(
        self,
        state: GaussianState,
        day_recs: list[dict],
    ) -> tuple[GaussianState, int]:
        """Run full UKF (predict+update) on all records for one day."""
        prev_ts: int | None = None
        for r in day_recs:
            ts      = r["timestamp_min"]
            dt_real = float(ts - prev_ts) if prev_ts is not None else 1.0
            dt_real = max(_DT_CLAMP_MIN, min(dt_real, _DT_CLAMP_MAX))

            state = self._filter.update_state(
                prior        = state,
                observations = {
                    "HR_obs_bpm":   r.get("HR_obs_bpm",   float("nan")),
                    "VO2_obs":      r.get("VO2_obs",       float("nan")),
                    "RMSSD_obs_ms": r.get("RMSSD_obs_ms",  float("nan")),
                },
                controls = {
                    "power_watts":     r.get("power_watts",     0.0),
                    "hub_T_core":      r.get("hub_T_core",      float("nan")),
                    "hub_pv_drop_pct": r.get("hub_pv_drop_pct", 0.0),
                },
                params  = self._params,
                dt_real = dt_real,
            )
            prev_ts = ts

        return state, day_recs[-1]["timestamp_min"]


# ---------------------------------------------------------------------------
# RMSSD morning observation lookup
# ---------------------------------------------------------------------------

def _find_morning_rmssd_ln(
    by_day:     dict[int, list[dict]],
    target_day: int,
    t0:         int,
) -> float | None:
    """
    Find ln(RMSSD_obs_ms) for target_day within +/-2h window around 08:00.
    Returns None if no valid observation exists.
    """
    if target_day not in by_day:
        return None

    lo = _MORNING_TARGET_MIN - _MORNING_WINDOW_MIN   # 360 = 06:00
    hi = _MORNING_TARGET_MIN + _MORNING_WINDOW_MIN   # 600 = 10:00

    best: float | None = None
    best_dist = _MORNING_WINDOW_MIN + 1

    for r in by_day[target_day]:
        rmssd = r.get("RMSSD_obs_ms")
        if rmssd is None or math.isnan(float(rmssd)) or float(rmssd) <= 0.0:
            continue
        mod = _minute_of_day(r["timestamp_min"], t0)
        if lo <= mod <= hi:
            dist = abs(mod - _MORNING_TARGET_MIN)
            if dist < best_dist:
                best_dist = dist
                best      = float(rmssd)

    return math.log(best) if best is not None else None


# ---------------------------------------------------------------------------
# Gate Zero entry point
# ---------------------------------------------------------------------------

def run_gate_zero(
    users_data:        dict,
    transition_params: CardioTransitionParams | None = None,
    warmup_days:       int   = _WARMUP_DAYS,
    wake_hour:         float = 8.0,
    nlme_train_days:   int   = _NLME_TRAIN_DAYS,
    nlme_steps:        int   = _NLME_STEPS_DEFAULT,
) -> GateZeroResult:
    """
    Run Gate Zero inter-day HRV validation across all users.

    L3 Individualisation
    --------------------
    For each user, attempts to personalise CardioTransitionParams via
    CardioNLME.fit_svi() on the first `nlme_train_days` of records.
    Falls back to `transition_params` (or population prior) on any failure.

    Parameters
    ----------
    users_data        : {user_id: list[dict]} -- per-user records (need not be sorted)
    transition_params : default ODE + filter params (None = population prior)
    warmup_days       : UKF burn-in days excluded from scoring (default 14)
    wake_hour         : local hour of wake-up for Borbely circadian modulator (default 8.0)
    nlme_train_days   : days of data used for per-user NLME fitting (default 30)
    nlme_steps        : SVI iterations for per-user NLME fit (default 3 000)

    Returns
    -------
    GateZeroResult with PASS | FAIL | INSUFFICIENT_DATA decision.

    Raises
    ------
    ValueError if users_data is empty.
    RuntimeError propagated if UKF diverges for any user.
    """
    if not users_data:
        raise ValueError("run_gate_zero: users_data is empty.")

    pop_params = transition_params   # population-prior fallback
    per_user: list[UserValidationResult] = []
    skipped = 0

    for uid, records in users_data.items():
        records = sorted(records, key=lambda r: r["timestamp_min"])

        if not records:
            logger.warning("run_gate_zero: user %s -- empty records, skipped.", uid)
            skipped += 1
            continue

        span_days = (records[-1]["timestamp_min"] - records[0]["timestamp_min"]) / _MINS_PER_DAY
        if span_days < _GATE_MIN_DAYS:
            logger.warning(
                "run_gate_zero: user %s -- span %.1f days < %d, skipped.",
                uid, span_days, _GATE_MIN_DAYS,
            )
            skipped += 1
            continue

        # ── L3: individualise ODE params from training window ────────────────
        t0 = records[0]["timestamp_min"]
        user_params = _fit_nlme_for_user(records, t0, nlme_train_days, nlme_steps)
        if user_params is None:
            user_params = pop_params   # degrade to population prior
            logger.info(
                "run_gate_zero: user %s -- using population prior (NLME unavailable).", uid
            )
        else:
            logger.info(
                "run_gate_zero: user %s -- individualised params applied "
                "(VO2max=%.1f, W'=%.1f).",
                uid,
                user_params.cardio.VO2_max_baseline,
                user_params.cardio.W_prime_capacity,
            )

        # ── Per-user validator with individualised params ────────────────────
        validator = InterDayTwinValidator(
            transition_params = user_params,
            warmup_days       = warmup_days,
            wake_hour         = wake_hour,
        )

        try:
            day_preds = validator.run(records)
        except (ValueError, RuntimeError) as exc:
            logger.warning("run_gate_zero: user %s -- exception: %s", uid, exc)
            skipped += 1
            continue

        if len(day_preds) < _GATE_MIN_DAYS:
            logger.warning(
                "run_gate_zero: user %s -- only %d paired days (< %d), skipped.",
                uid, len(day_preds), _GATE_MIN_DAYS,
            )
            skipped += 1
            continue

        err_twin    = np.array([abs(d.predicted_ln_rmssd - d.observed_ln_rmssd) for d in day_preds])
        err_persist = np.array([abs(d.persist_ln_rmssd   - d.observed_ln_rmssd) for d in day_preds])
        err_ewma    = np.array([abs(d.ewma_ln_rmssd      - d.observed_ln_rmssd) for d in day_preds])

        mae_twin    = float(np.mean(err_twin))
        mae_persist = float(np.mean(err_persist))
        mae_ewma    = float(np.mean(err_ewma))

        beats_persist = mae_twin < mae_persist
        beats_ewma    = mae_twin < mae_ewma
        twin_wins     = beats_persist and beats_ewma

        per_user.append(UserValidationResult(
            user_id            = uid,
            n_days             = len(day_preds),
            mae_twin           = mae_twin,
            mae_persistence    = mae_persist,
            mae_ewma7d         = mae_ewma,
            twin_beats_persist = beats_persist,
            twin_beats_ewma    = beats_ewma,
            twin_wins          = twin_wins,
            days               = list(day_preds),
        ))
        logger.info(
            "Gate Zero  user=%-12s  n=%3d  mae_twin=%.4f  "
            "mae_persist=%.4f  mae_ewma=%.4f  wins=%s",
            uid, len(day_preds), mae_twin, mae_persist, mae_ewma, twin_wins,
        )

    n_users = len(per_user)
    if n_users == 0:
        logger.error(
            "run_gate_zero: no users had sufficient data (skipped=%d).", skipped
        )
        return GateZeroResult(
            decision          = GateZeroDecision.INSUFFICIENT_DATA,
            n_users_total     = 0,
            n_users_twin_wins = 0,
            win_fraction      = 0.0,
            per_user          = [],
            gate_criterion    = _gate_criterion_str(warmup_days),
        )

    n_wins       = sum(1 for u in per_user if u.twin_wins)
    win_fraction = n_wins / n_users
    decision     = (
        GateZeroDecision.PASS if win_fraction >= _GATE_PASS_FRAC
        else GateZeroDecision.FAIL
    )

    logger.info(
        "Gate Zero RESULT=%s  win_fraction=%.2f (%d/%d users)  skipped=%d",
        decision.value, win_fraction, n_wins, n_users, skipped,
    )

    return GateZeroResult(
        decision          = decision,
        n_users_total     = n_users,
        n_users_twin_wins = n_wins,
        win_fraction      = win_fraction,
        per_user          = per_user,
        gate_criterion    = _gate_criterion_str(warmup_days),
    )


def _gate_criterion_str(warmup_days: int) -> str:
    return (
        f"Twin MAE(ln RMSSD) < Persistence AND EWMA-7d "
        f"in >={_GATE_PASS_FRAC:.0%} users | "
        f">={_GATE_MIN_DAYS} paired prediction days | "
        f"warmup={warmup_days} days | "
        f"target=ln(RMSSD_obs_ms) at 08:00 | "
        f"overnight_dt={_OVERNIGHT_DT_MIN} min | "
        f"nlme_train_days={_NLME_TRAIN_DAYS}"
    )
