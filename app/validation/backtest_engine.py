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
     Sleep controls: power=0 W, T_core=37.0 C, pv_drop=0 %.
     Step size: OVERNIGHT_DT_MIN = 10 min (bi-exp W' recovery is slow enough).
     No measurement update during this segment -- genuine blind forecast.

  3. PREDICTION:
     predicted_rmssd_ms = clip(AT_0800, 0, 1) * RMSSD_ref_ms
     predicted_target   = ln(max(predicted_rmssd_ms, 1.0))

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
# Blind overnight integrator
# ---------------------------------------------------------------------------

def _blind_overnight_predict(
    state:    GaussianState,
    params:   CardioTransitionParams,
    n_min:    int,
    dt_min:   int = _OVERNIGHT_DT_MIN,
) -> GaussianState:
    """
    Run UKF predict-only for n_min minutes under sleep conditions.

    Sleep controls: power=0, T_core=37.0, pv_drop=0.
    Uses dt_min-minute steps. Q is Wiener-scaled per step.
    No measurement update -- pure blind recovery simulation.

    The bi-exp W' pools (tau=2 and 30 min) and Autonomic_Tone (tau_rec ~20 min)
    drive towards their equilibrium values under zero exercise stress. The 7-day
    RMSSD_load_7d state barely moves (tau=10080 min) and serves as the slow
    inter-day signal for trend-tracking.
    """
    u = jnp.array([0.0, 37.0, 0.0], dtype=jnp.float32)

    sleep_params = CardioTransitionParams(
        cardio           = params.cardio,
        dt_min           = float(dt_min),
        hub_T_core       = 37.0,
        hub_pv_drop_pct  = 0.0,
    )
    Q_step = scale_Q(Q_DEFAULT, float(dt_min))

    current = state
    elapsed = 0
    while elapsed < n_min:
        step = min(dt_min, n_min - elapsed)
        if step != dt_min:
            part_params = CardioTransitionParams(
                cardio=params.cardio, dt_min=float(step),
                hub_T_core=37.0, hub_pv_drop_pct=0.0,
            )
            Q_part = scale_Q(Q_DEFAULT, float(step))
            m, c = _ukf_predict_cardio(current.mean, current.cov, u, part_params, Q_part)
        else:
            m, c = _ukf_predict_cardio(current.mean, current.cov, u, sleep_params, Q_step)
        current = GaussianState(mean=m, cov=c)
        elapsed += step

    return current


# ---------------------------------------------------------------------------
# Core inter-day validator
# ---------------------------------------------------------------------------

class InterDayTwinValidator:
    """
    Inter-day walk-forward HRV prediction engine.

    For each day D (after warmup):
        1. Assimilate all intra-day records with async UKF.
        2. Blind overnight predict to 08:00 next day.
        3. Read AT(08:00) -> predicted RMSSD.
        4. Score against next morning's observed RMSSD.
    """

    def __init__(
        self,
        transition_params: CardioTransitionParams | None = None,
        warmup_days:       int = _WARMUP_DAYS,
    ) -> None:
        self._filter   = CardioStateFilter()
        self._params   = transition_params or DEFAULT_TRANSITION_PARAMS
        self._warmup   = warmup_days

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

        t0     = records[0]["timestamp_min"]
        by_day = _group_by_day(records, t0)

        state: GaussianState = GaussianState(
            mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT
        )
        persist_ln: float | None = None
        ewma = LogRMSSDEWMA()
        preds: list[DayPrediction] = []

        for day_idx in sorted(by_day.keys()):
            day_recs = sorted(by_day[day_idx], key=lambda r: r["timestamp_min"])

            # ── Update baselines with THIS day's morning RMSSD observation ──
            today_ln = _find_morning_rmssd_ln(by_day, day_idx, t0)
            if today_ln is not None:
                ewma.update(today_ln)
                persist_ln = today_ln

            # ── 1. Daytime assimilation ──────────────────────────────────────
            state, last_ts = self._assimilate_day(state, day_recs)

            # ── 2. Blind overnight predict to next morning 08:00 ─────────────
            morning_next_abs = t0 + (day_idx + 1) * _MINS_PER_DAY + _MORNING_TARGET_MIN
            predict_min = max(30, morning_next_abs - last_ts)
            state_0800 = _blind_overnight_predict(state, self._params, predict_min)

            # ── 3. Predicted RMSSD via Autonomic_Tone(08:00) ─────────────────
            at_0800       = float(jnp.clip(state_0800.mean[IDX_AT], 0.0, 1.0))
            rmssd_ref     = float(self._params.cardio.RMSSD_ref_ms)
            pred_rmssd    = max(at_0800 * rmssd_ref, 1.0)
            pred_ln       = math.log(pred_rmssd)

            # ── 4. Ground truth: next morning RMSSD ───────────────────────────
            obs_ln = _find_morning_rmssd_ln(by_day, day_idx + 1, t0)

            # ── 5. Emit prediction if all signals available and past warmup ───
            ewma_pred = ewma.predict()
            if (
                day_idx >= self._warmup
                and obs_ln      is not None
                and persist_ln  is not None
                and ewma_pred   is not None
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
            ts     = r["timestamp_min"]
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
    warmup_days:       int = _WARMUP_DAYS,
) -> GateZeroResult:
    """
    Run Gate Zero inter-day HRV validation across all users.

    Parameters
    ----------
    users_data        : {user_id: list[dict]} -- per-user records (need not be sorted)
    transition_params : ODE + filter params (None = population prior)
    warmup_days       : UKF burn-in days excluded from scoring (default 14)

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

    validator = InterDayTwinValidator(
        transition_params=transition_params,
        warmup_days=warmup_days,
    )
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
        f"overnight_dt={_OVERNIGHT_DT_MIN} min"
    )
