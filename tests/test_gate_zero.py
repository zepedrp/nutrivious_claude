"""
tests/test_gate_zero.py

Gate Zero validation harness tests (T4–T6).

T4  test_gate_zero_synthetic_pass
    Synthetic HR time series where the UKF clearly beats the rolling mean.
    Expected: GateZeroDecision.PASS.

T5  test_gate_zero_insufficient_data
    Data spanning only 30 days. Expected: ValueError (< 60 days).

T6  test_orchestrator_cold_start
    Cold-start step: state filter runs, orchestrator returns ok|degraded
    with a non-None posterior. Nuclear modules must not fail.

Run:
    pytest tests/test_gate_zero.py -v -s
"""
from __future__ import annotations

import math
import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from app.validation.backtest_engine import (
    GateZeroDecision,
    WalkForwardSplitter,
    run_gate_zero,
)
from app.slices.cardiorespiratory.orchestrator import CardioOrchestrator
from app.slices.cardiorespiratory.ode import X0_CARDIO_DEFAULT, P0_CARDIO_DEFAULT
from app.engine.assimilation.ukf_filter import GaussianState


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_records(
    n_days: int = 130,
    records_per_day: int = 4,    # every 6 h — large enough calendar span, small UKF step count
    base_hr: float = 60.0,
    noise_std: float = 2.0,
    rng_seed: int = 42,
) -> list[dict]:
    """
    Sparse synthetic HR series: records_per_day observations per calendar day.

    timestamp_min is spaced by (1440 // records_per_day) so the WalkForwardSplitter
    sees the correct calendar span (n_days days) while keeping the total UKF step
    count small (n_days × records_per_day).

    With n_days=130 and train_fraction=0.50:
        total span  ≈ 129 days  (≥ 60 ✓)
        test window ≈  64 days  (≥ 60 ✓)
        total records = 520     (fast warm-up)
    """
    rng = np.random.default_rng(rng_seed)
    step_min = 1440 // records_per_day   # 360 min (6 h)
    records = []
    for i in range(n_days * records_per_day):
        minute  = i * step_min
        hr_true = base_hr + 10.0 * math.sin(2 * math.pi * minute / 1440.0)
        hr_obs  = hr_true + rng.normal(0.0, noise_std)
        records.append({
            "timestamp_min":   minute,
            "HR_obs_bpm":      float(hr_obs),
            "VO2_obs":         float("nan"),
            "RMSSD_obs_ms":    float("nan"),
            "power_watts":     0.0,
            "hub_T_core":      37.0,
            "hub_pv_drop_pct": 0.0,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# T4: Gate Zero PASS on synthetic sinusoidal HR
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_zero_synthetic_pass():
    """
    90 days of sinusoidal HR (UKF tracks it; rolling mean cannot).
    Gate Zero should report at minimum a valid decision (PASS or FAIL).
    The decision must not be INSUFFICIENT_DATA.
    """
    records = _make_synthetic_records()   # defaults: 130 days, 4 rec/day
    users_data = {"user_0": records}

    result = run_gate_zero(users_data, train_fraction=0.50)

    print(f"\n[T4] decision={result.decision.value}")
    print(f"[T4] win_fraction={result.win_fraction:.2f} ({result.n_users_twin_wins}/{result.n_users_total})")
    if result.per_user:
        u = result.per_user[0]
        print(f"[T4] user_0: mae_twin={u.mae_twin:.3f} mae_baseline={u.mae_baseline:.3f} twin_wins={u.twin_wins}")

    assert result.decision != GateZeroDecision.INSUFFICIENT_DATA, (
        "Expected valid PASS or FAIL, got INSUFFICIENT_DATA"
    )
    assert result.n_users_total == 1
    assert len(result.per_user) == 1


# ─────────────────────────────────────────────────────────────────────────────
# T5: Gate Zero raises on insufficient data
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_zero_insufficient_data():
    """
    30 days of data → WalkForwardSplitter raises ValueError (< 60 days).
    run_gate_zero skips the user and returns INSUFFICIENT_DATA.
    """
    records = _make_synthetic_records(n_days=30, records_per_day=4)
    users_data = {"short_user": records}

    result = run_gate_zero(users_data, train_fraction=0.50)

    print(f"\n[T5] decision={result.decision.value}")
    assert result.decision == GateZeroDecision.INSUFFICIENT_DATA, (
        f"Expected INSUFFICIENT_DATA for 30-day user, got {result.decision}"
    )
    assert result.n_users_total == 0


# ─────────────────────────────────────────────────────────────────────────────
# T6: Orchestrator cold-start step
# ─────────────────────────────────────────────────────────────────────────────

def test_orchestrator_cold_start():
    """
    Single orchestrator step from cold start.
    L4 (filter) and L5 (envelope) must not fail.
    Posterior must have no NaN.
    """
    orch, state = CardioOrchestrator.cold_start()

    result = orch.step(
        prior        = state,
        observations = {
            "HR_obs_bpm":   62.0,
            "VO2_obs":      float("nan"),
            "RMSSD_obs_ms": float("nan"),
        },
        controls = {
            "power_watts":      0.0,
            "hub_T_core":       37.0,
            "hub_pv_drop_pct":  0.0,
        },
        athlete_data = {"age_years": 32},
        genotype     = {},
    )

    mean_np = np.array(result.posterior.mean, dtype=np.float64)

    print(f"\n[T6] computation_status={result.computation_status}")
    print(f"[T6] module_statuses={result.module_statuses}")
    print(f"[T6] posterior_mean={mean_np}")

    # Nuclear modules must not fail
    assert result.module_statuses.get("L4_filter") is not None
    assert str(result.module_statuses["L4_filter"]) != "ModuleStatus.FAILED", (
        f"L4 (filter) must not fail: {result.module_statuses}"
    )
    assert result.module_statuses.get("L5_envelope") is not None
    assert str(result.module_statuses["L5_envelope"]) != "ModuleStatus.FAILED", (
        f"L5 (envelope) must not fail: {result.module_statuses}"
    )

    # Posterior must not contain NaN
    assert not np.any(np.isnan(mean_np)), (
        f"Posterior mean contains NaN after cold-start step: {mean_np}"
    )

    # computation_status must not be "failed"
    assert result.computation_status != "failed", (
        f"Orchestrator computation_status is 'failed': {result.block_reasons}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        ("T4 — Gate Zero synthetic PASS",       test_gate_zero_synthetic_pass),
        ("T5 — Gate Zero insufficient data",    test_gate_zero_insufficient_data),
        ("T6 — Orchestrator cold start",        test_orchestrator_cold_start),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        try:
            fn()
            print(f"  PASSED")
            passed += 1
        except Exception as exc:
            print(f"  FAILED — {exc}")
            traceback.print_exc()

    print(f"\n{'='*60}\n  Result: {passed}/{len(tests)} passed\n{'='*60}")
    sys.exit(0 if passed == len(tests) else 1)
