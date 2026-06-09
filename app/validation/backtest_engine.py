"""
app/validation/backtest_engine.py

Gate Zero — Walk-Forward Out-of-Sample Validation Harness

Criterion (HLD §5, Compass Passo 13)
──────────────────────────────────────
The digital twin must beat the 7-day rolling mean (naive baseline) on MAE
in ≥60% of users at ≥60 days of data.

    GateZeroDecision.PASS   — twin_beats_baseline in ≥60% users
    GateZeroDecision.FAIL   — twin_beats_baseline in < 60% users
    GateZeroDecision.INSUFFICIENT_DATA — < 60 days of data for any user

Walk-forward protocol
──────────────────────
For each user:
    1. Train on [0, T_train) — fit NLME posterior (or use population prior)
    2. Test on [T_train, T_train + 60 days) — one-step-ahead prediction
    3. Baseline: 7-day rolling mean of the training window's HR target
    4. Twin: UKF predict-step mean at each test step
    5. Compute MAE(twin) and MAE(baseline) on HR channel

Data format
────────────
Each user's data is a list[dict] of 1-min observation records:
    {
        "timestamp_min": int,          # minutes since day 0
        "HR_obs_bpm":    float | None,
        "VO2_obs":       float | None,
        "RMSSD_obs_ms":  float | None,
        "power_watts":   float,        # 0.0 if rest
        "hub_T_core":    float | None,
        "hub_pv_drop_pct": float,      # 0.0 if unknown
    }

Fail-Loud contract
───────────────────
RuntimeError if twin produces NaN predictions (filter divergence).
All exceptions from CardioStateFilter propagate unmodified.
Empty user data raises ValueError loudly.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from enum import Enum
from typing import NamedTuple

import numpy as np

from app.engine.assimilation.ukf_filter import GaussianState
from app.slices.cardiorespiratory.ode import (
    X0_CARDIO_DEFAULT,
    P0_CARDIO_DEFAULT,
    IDX_HR,
)
from app.slices.cardiorespiratory.filter import (
    CardioStateFilter,
    CardioTransitionParams,
    DEFAULT_TRANSITION_PARAMS,
)

logger = logging.getLogger(__name__)

# Gate Zero criterion (HLD §5)
_GATE_MIN_DAYS:         int   = 60      # minimum test window [days]
_GATE_PASS_FRACTION:    float = 0.60   # fraction of users where twin must win
_ROLLING_WINDOW_DAYS:   int   = 7      # rolling mean baseline window [days]
_MINS_PER_DAY:          int   = 1440


# ── Decision enum ─────────────────────────────────────────────────────────────

class GateZeroDecision(Enum):
    PASS              = "PASS"
    FAIL              = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# ── Per-user result ───────────────────────────────────────────────────────────

class UserValidationResult(NamedTuple):
    user_id:        str | int
    n_test_steps:   int
    mae_twin:       float   # MAE of UKF predict-step HR [bpm]
    mae_baseline:   float   # MAE of 7-day rolling mean HR [bpm]
    twin_wins:      bool    # mae_twin < mae_baseline


# ── Gate Zero aggregate result ────────────────────────────────────────────────

class GateZeroResult(NamedTuple):
    decision:           GateZeroDecision
    n_users_total:      int
    n_users_twin_wins:  int
    win_fraction:       float
    per_user:           list   # list[UserValidationResult]
    gate_criterion:     str    # human-readable threshold description


# ── Walk-forward splitter ─────────────────────────────────────────────────────

class WalkForwardSplitter:
    """
    Splits a chronological observation list into train/test at a fixed train fraction.

    Default: 50% train, 50% test. Test must be ≥ 60 days.
    """

    def __init__(self, train_fraction: float = 0.50) -> None:
        if not (0.0 < train_fraction < 1.0):
            raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
        self.train_fraction = train_fraction

    def split(
        self, records: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Split records into (train, test).

        Records must be sorted by timestamp_min ascending.

        Raises
        ------
        ValueError if fewer than 60 days of records, or if test window < 60 days.
        """
        if not records:
            raise ValueError("WalkForwardSplitter: records is empty.")

        t_min = records[0]["timestamp_min"]
        t_max = records[-1]["timestamp_min"]
        total_days = (t_max - t_min) / _MINS_PER_DAY

        if total_days < _GATE_MIN_DAYS:
            raise ValueError(
                f"WalkForwardSplitter: total span {total_days:.1f} days < "
                f"required {_GATE_MIN_DAYS} days."
            )

        split_min = t_min + self.train_fraction * (t_max - t_min)
        train = [r for r in records if r["timestamp_min"] <  split_min]
        test  = [r for r in records if r["timestamp_min"] >= split_min]

        if not train:
            raise ValueError("WalkForwardSplitter: train set is empty.")
        if not test:
            raise ValueError("WalkForwardSplitter: test set is empty.")

        test_days = (test[-1]["timestamp_min"] - test[0]["timestamp_min"]) / _MINS_PER_DAY
        if test_days < _GATE_MIN_DAYS:
            raise ValueError(
                f"WalkForwardSplitter: test window {test_days:.1f} days < "
                f"required {_GATE_MIN_DAYS} days."
            )

        return train, test


# ── Rolling mean baseline ─────────────────────────────────────────────────────

class RollingMeanBaseline:
    """
    7-day rolling mean baseline for HR prediction.

    Computes the mean of all HR observations in the trailing 7-day window.
    Initialised from the training set; updated as test observations are consumed.
    """

    def __init__(self, window_days: int = _ROLLING_WINDOW_DAYS) -> None:
        self._window_min = window_days * _MINS_PER_DAY
        self._buffer:    deque[tuple[int, float]] = deque()  # (timestamp_min, hr)
        self._sum:       float = 0.0
        self._n:         int   = 0

    def fit(self, train_records: list[dict]) -> None:
        """Load training window HR observations into rolling buffer."""
        for r in train_records:
            hr = r.get("HR_obs_bpm")
            if hr is not None and not math.isnan(float(hr)):
                self._buffer.append((r["timestamp_min"], float(hr)))
                self._sum += float(hr)
                self._n   += 1

    def predict(self, timestamp_min: int) -> float:
        """
        Return the rolling mean HR as of `timestamp_min`.

        Evicts observations older than window_days from the buffer.

        Returns float('nan') if no observations are in the window.
        """
        cutoff = timestamp_min - self._window_min
        while self._buffer and self._buffer[0][0] < cutoff:
            _, evicted_hr = self._buffer.popleft()
            self._sum -= evicted_hr
            self._n   -= 1

        return self._sum / self._n if self._n > 0 else float("nan")

    def observe(self, timestamp_min: int, hr_obs: float) -> None:
        """Feed an observed HR into the rolling buffer after predicting for this step."""
        if not math.isnan(hr_obs):
            self._buffer.append((timestamp_min, hr_obs))
            self._sum += hr_obs
            self._n   += 1


# ── Twin validator ────────────────────────────────────────────────────────────

class TwinValidator:
    """
    One-step-ahead HR prediction via the cardiorespiratory UKF.

    Uses only the predict step (no update) to produce out-of-sample forecasts.
    The prior state is the posterior from the previous step in the training set,
    then rolled forward through the test set with UKF update.

    Walk-forward protocol:
        1. Warm-up: run full UKF (predict + update) on training set.
        2. Test: for each test step:
               a. Record predict-step HR mean (BEFORE update) as the forecast.
               b. Run UKF update to propagate state.
               c. Compare forecast vs HR_obs.
    """

    def __init__(
        self,
        transition_params: CardioTransitionParams | None = None,
    ) -> None:
        self._filter = CardioStateFilter()
        self._trans  = transition_params or DEFAULT_TRANSITION_PARAMS

    def warm_up(self, train_records: list[dict]) -> GaussianState:
        """
        Run full UKF on training records. Returns final posterior state.

        Raises RuntimeError if UKF diverges (NaN in posterior_mean).
        """
        state = GaussianState(mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT)
        for r in train_records:
            state = self._filter.update_state(
                prior        = state,
                observations = self._obs_from_record(r),
                controls     = self._ctrl_from_record(r),
                params       = self._trans,
            )
        return state

    def evaluate_test(
        self,
        warm_state:   GaussianState,
        test_records: list[dict],
    ) -> tuple[list[float], list[float]]:
        """
        Walk-forward evaluation on test records.

        Returns
        -------
        (twin_preds, true_hrs)
            twin_preds : list[float] — UKF predict-step HR means [bpm]
            true_hrs   : list[float] — observed HRs (NaN excluded from both lists)
        """
        state     = warm_state
        preds:    list[float] = []
        actuals:  list[float] = []

        for r in test_records:
            hr_obs = r.get("HR_obs_bpm")
            if hr_obs is None or math.isnan(float(hr_obs)):
                # No ground truth for this step — skip prediction (no information)
                # Still run UKF update so state stays synchronised
                state = self._filter.update_state(
                    prior        = state,
                    observations = self._obs_from_record(r),
                    controls     = self._ctrl_from_record(r),
                    params       = self._trans,
                )
                continue

            # ── Predict step: HR mean BEFORE assimilation ─────────────────
            # Run predict only (no update) to get the genuine out-of-sample forecast
            from app.slices.cardiorespiratory.filter import _ukf_predict_cardio, Q_DEFAULT
            u = np.array([
                r.get("power_watts",      0.0),
                r.get("hub_T_core",       float("nan")),
                r.get("hub_pv_drop_pct",  0.0),
            ], dtype=np.float32)
            import jax.numpy as jnp
            mean_pred, _ = _ukf_predict_cardio(
                state.mean, state.cov,
                jnp.array(u, dtype=jnp.float32),
                self._trans, Q_DEFAULT,
            )
            hr_pred = float(mean_pred[IDX_HR])

            if math.isnan(hr_pred):
                raise RuntimeError(
                    "TwinValidator: NaN in UKF predict-step HR. "
                    "Filter diverged — check Q/R and initial state."
                )

            preds.append(hr_pred)
            actuals.append(float(hr_obs))

            # ── Update step: assimilate actual observation ─────────────────
            state = self._filter.update_state(
                prior        = state,
                observations = self._obs_from_record(r),
                controls     = self._ctrl_from_record(r),
                params       = self._trans,
            )

        return preds, actuals

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _obs_from_record(r: dict) -> dict[str, float]:
        return {
            "HR_obs_bpm":    r.get("HR_obs_bpm",   float("nan")),
            "VO2_obs":       r.get("VO2_obs",       float("nan")),
            "RMSSD_obs_ms":  r.get("RMSSD_obs_ms",  float("nan")),
        }

    @staticmethod
    def _ctrl_from_record(r: dict) -> dict[str, float]:
        return {
            "power_watts":      r.get("power_watts",      0.0),
            "hub_T_core":       r.get("hub_T_core",       float("nan")),
            "hub_pv_drop_pct":  r.get("hub_pv_drop_pct",  0.0),
        }


# ── Gate Zero entry point ─────────────────────────────────────────────────────

def run_gate_zero(
    users_data: dict[str | int, list[dict]],
    transition_params: CardioTransitionParams | None = None,
    train_fraction: float = 0.50,
) -> GateZeroResult:
    """
    Run the Gate Zero walk-forward validation across all users.

    Parameters
    ----------
    users_data        : {user_id: list[dict]} — per-user chronological records
    transition_params : ODE + filter params; None = population prior
    train_fraction    : fraction of each user's data used for warm-up

    Returns
    -------
    GateZeroResult with PASS | FAIL | INSUFFICIENT_DATA decision

    Raises
    ------
    ValueError if users_data is empty.
    RuntimeError if any user's UKF diverges (propagated).
    """
    if not users_data:
        raise ValueError("run_gate_zero: users_data is empty.")

    splitter  = WalkForwardSplitter(train_fraction=train_fraction)
    per_user: list[UserValidationResult] = []
    skipped   = 0

    for uid, records in users_data.items():
        # Sort chronologically
        records = sorted(records, key=lambda r: r["timestamp_min"])

        try:
            train, test = splitter.split(records)
        except ValueError as exc:
            logger.warning("run_gate_zero: user %s skipped — %s", uid, exc)
            skipped += 1
            continue

        # ── Baseline ──────────────────────────────────────────────────────
        baseline = RollingMeanBaseline()
        baseline.fit(train)

        # ── Twin warm-up ──────────────────────────────────────────────────
        twin = TwinValidator(transition_params=transition_params)
        warm_state = twin.warm_up(train)

        # ── Walk-forward evaluation ───────────────────────────────────────
        twin_preds, actuals = twin.evaluate_test(warm_state, test)

        if not twin_preds:
            logger.warning("run_gate_zero: user %s has no valid HR observations in test set.", uid)
            skipped += 1
            continue

        # Baseline predictions (aligned to the same steps as twin_preds)
        # Re-run baseline in order on test set to get aligned predictions
        baseline2 = RollingMeanBaseline()
        baseline2.fit(train)
        hr_test_records = [r for r in test if (
            r.get("HR_obs_bpm") is not None and
            not math.isnan(float(r.get("HR_obs_bpm", float("nan"))))
        )]
        base_preds: list[float] = []
        for r in hr_test_records:
            base_pred = baseline2.predict(r["timestamp_min"])
            base_preds.append(base_pred if not math.isnan(base_pred) else float(actuals[0]) if actuals else 70.0)
            baseline2.observe(r["timestamp_min"], float(r["HR_obs_bpm"]))

        n = min(len(twin_preds), len(base_preds), len(actuals))
        if n == 0:
            skipped += 1
            continue

        mae_twin     = float(np.mean(np.abs(np.array(twin_preds[:n]) - np.array(actuals[:n]))))
        mae_baseline = float(np.mean(np.abs(np.array(base_preds[:n]) - np.array(actuals[:n]))))
        twin_wins    = mae_twin < mae_baseline

        per_user.append(UserValidationResult(
            user_id      = uid,
            n_test_steps = n,
            mae_twin     = mae_twin,
            mae_baseline = mae_baseline,
            twin_wins    = twin_wins,
        ))

        logger.info(
            "run_gate_zero user=%s n=%d mae_twin=%.3f mae_baseline=%.3f twin_wins=%s",
            uid, n, mae_twin, mae_baseline, twin_wins,
        )

    n_users     = len(per_user)
    if n_users == 0:
        logger.error("run_gate_zero: no users had sufficient data (skipped=%d).", skipped)
        return GateZeroResult(
            decision          = GateZeroDecision.INSUFFICIENT_DATA,
            n_users_total     = 0,
            n_users_twin_wins = 0,
            win_fraction      = 0.0,
            per_user          = [],
            gate_criterion    = (
                f"twin MAE < baseline (7-day rolling mean) in "
                f"≥{_GATE_PASS_FRACTION:.0%} of users | "
                f"≥{_GATE_MIN_DAYS} days test window"
            ),
        )

    n_wins       = sum(1 for u in per_user if u.twin_wins)
    win_fraction = n_wins / n_users

    if win_fraction >= _GATE_PASS_FRACTION:
        decision = GateZeroDecision.PASS
    else:
        decision = GateZeroDecision.FAIL

    logger.info(
        "run_gate_zero RESULT=%s win_fraction=%.2f (%d/%d users) skipped=%d",
        decision.value, win_fraction, n_wins, n_users, skipped,
    )

    return GateZeroResult(
        decision          = decision,
        n_users_total     = n_users,
        n_users_twin_wins = n_wins,
        win_fraction      = win_fraction,
        per_user          = per_user,
        gate_criterion    = (
            f"twin MAE < baseline (7-day rolling mean) in "
            f"≥{_GATE_PASS_FRACTION:.0%} of users | "
            f"≥{_GATE_MIN_DAYS} days test window"
        ),
    )
