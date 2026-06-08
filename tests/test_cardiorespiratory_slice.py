"""
tests/test_cardiorespiratory_slice.py

Gate Zero — Cardiorespiratory Slice (L2-L4)

Three targeted tests covering the core physics and filter stability
of the cardiorespiratory performance slice.

T1  test_cardiovascular_drift
    Constant power, hub_pv_drop rises (plasma volume loss) + hub_T_core rises
    (thermal load from Mod 10). Physics: Frank-Starling + Coyle 2001.
    Expected: dSV/dt more negative, dHR/dt more positive.

T2  test_w_prime_respiratory_steal
    Power >> CP: W'_bal drains (Clarke-Skiba), Resp_Fatigue rises (Dempsey
    metaboreflex). Constraints fire on exhausted state.
    Expected: dW'/dt < 0, dRF/dt > 0, ≥1 constraint violation.

T3  test_ukf_stability
    Assimilate 30 steps of HR_obs only (VO2 = NaN, RMSSD = NaN).
    Expected: no NaNs in posterior, covariance PSD, physical clamps obeyed.

Run:
    pytest tests/test_cardiorespiratory_slice.py -v -s
"""
from __future__ import annotations

import math
import sys
import numpy as np

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp

from app.slices.cardiorespiratory.ode import (
    cardiorespiratory_slice_ode,
    DEFAULT_CARDIO_SLICE_PARAMS,
    X0_CARDIO_DEFAULT,
    P0_CARDIO_DEFAULT,
    IDX_VO2, IDX_HR, IDX_SV, IDX_WPRIME, IDX_RF, IDX_AT,
)
from app.slices.cardiorespiratory.envelope import (
    build_cardiorespiratory_envelope,
    check_hard_constraints,
    check_all_constraints,
)
from app.slices.cardiorespiratory.filter import (
    CardioStateFilter,
    CardioTransitionParams,
)
from app.engine.assimilation.ukf_filter import GaussianState


# ─────────────────────────────────────────────────────────────────────────────
# T1: Cardiovascular Drift
# ─────────────────────────────────────────────────────────────────────────────

def test_cardiovascular_drift():
    """
    Inject hub_pv_drop_pct (plasma-volume loss) and hub_T_core (thermal load).
    Both channels must depress SV and accelerate HR per the Coyle 2001 mechanism.

    Physics verified at ODE derivative level (no full integration required):
      • hub_pv_drop = 8% → drift_factor = 0.88 → SV_target ↓ → dSV/dt more negative
      • hub_T_core = 38.5°C → HR_Tcore = +12 bpm → HR_target ↑ → dHR/dt larger
    """
    params = DEFAULT_CARDIO_SLICE_PARAMS

    # Exercising steady state (200 W, 30+ min in): SV at reference (peak Frank-Starling)
    x_ss = jnp.array([40.0, 155.0, params.SV_ref, 12.0, 0.10, 0.40], dtype=jnp.float32)

    # Baseline: NaN T_core (→ 37°C internally), 0% PV drop
    dx_base = cardiorespiratory_slice_ode(
        jnp.float32(0.0), x_ss,
        (params, 200.0, float("nan"), 0.0)
    )

    # CV drift: T_core = 38.5°C (thermal load) + 8% PV drop (plasma volume loss)
    dx_drift = cardiorespiratory_slice_ode(
        jnp.float32(0.0), x_ss,
        (params, 200.0, 38.5, 8.0)
    )

    dSV_base  = float(dx_base[IDX_SV])
    dSV_drift = float(dx_drift[IDX_SV])
    dHR_base  = float(dx_base[IDX_HR])
    dHR_drift = float(dx_drift[IDX_HR])

    print(f"\n[T1] dSV_base={dSV_base:.5f} L/min  dSV_drift={dSV_drift:.5f} L/min")
    print(f"[T1] dHR_base={dHR_base:.3f} bpm/min  dHR_drift={dHR_drift:.3f} bpm/min")

    # SV falls faster under PV loss (Frank-Starling; Coyle 1986)
    assert dSV_drift < dSV_base, (
        f"Expected dSV/dt more negative under CV drift: "
        f"base={dSV_base:.5f}, drift={dSV_drift:.5f}"
    )

    # HR rises faster to compensate (thermal tachycardia; Coyle 2001)
    assert dHR_drift > dHR_base, (
        f"Expected dHR/dt larger under thermal + PV drift: "
        f"base={dHR_base:.3f}, drift={dHR_drift:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T2: W' Depletion + Respiratory Steal + Constraint Check
# ─────────────────────────────────────────────────────────────────────────────

def test_w_prime_respiratory_steal():
    """
    Power >> CP (400 W > 250 W): W'_bal drains and Resp_Fatigue climbs.
    Exhausted-state constraint check: HR ceiling and W' floor violations.

    Two-part test:
    (a) ODE derivatives confirm correct direction of state change.
    (b) Phase3Envelope constraint evaluation fires on the exhausted state.
    """
    params = DEFAULT_CARDIO_SLICE_PARAMS

    # Near-exhaustion state: VO2 at max, HR high, W'_bal nearly gone, RF high
    x_exhaust = jnp.array([46.0, 182.0, 0.060, 0.5, 0.70, 0.10], dtype=jnp.float32)

    # Power well above CP: 400 W vs CP = 250 W
    dx = cardiorespiratory_slice_ode(
        jnp.float32(0.0), x_exhaust,
        (params, 400.0, 37.0, 0.0)
    )

    dW_dt  = float(dx[IDX_WPRIME])
    dRF_dt = float(dx[IDX_RF])

    print(f"\n[T2] dW'/dt={dW_dt:.4f} kJ/min  (expected < 0)")
    print(f"[T2] dRF/dt={dRF_dt:.4f} adim/min  (expected > 0)")

    # W'_bal must drain above CP (Clarke-Skiba 2013)
    assert dW_dt < 0, (
        f"W'_bal must drain at P >> CP; got dW'/dt = {dW_dt:.4f} kJ/min"
    )

    # Resp_Fatigue must increase at 400 W / 350 W_ref = 1.14 > RF_threshold (0.85)
    assert dRF_dt > 0, (
        f"Resp_Fatigue must rise above RF_threshold; got dRF/dt = {dRF_dt:.4f} adim/min"
    )

    # ── Part (b): constraint evaluation on exhausted state ─────────────────
    env = build_cardiorespiratory_envelope({})

    # State where HR exceeds ceiling (200 > HR_max×1.05 = 194.25)
    # and W'_bal is at zero (< 0.05 kJ hard floor)
    state_exhausted = {
        "Hub_HR_bpm":         200.0,   # > 185 × 1.05 = 194.25 → hard ceiling violated
        "Hub_W_prime_kJ":     0.0,     # < 0.05 kJ hard floor  → task failure violated
        "Hub_Resp_Fatigue":   0.90,    # > 0.85 → chance constraint violated
    }

    violations = check_all_constraints(env, state_exhausted)
    print(f"[T2] Constraint violations on exhausted state: {violations}")

    assert len(violations) >= 1, (
        f"Expected ≥1 constraint violation for exhausted state, got: {violations}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T3: UKF Stability under Dense HR + Sparse VO2/RMSSD
# ─────────────────────────────────────────────────────────────────────────────

def test_ukf_stability():
    """
    Assimilate 30 minutes of resting HR only (VO2 = NaN, RMSSD = NaN).
    The filter must remain numerically stable and respect physical clamps.

    Checks:
        (a) No NaN in posterior_mean after 30 steps
        (b) W'_bal ≥ 0 (mandatory clamp via jnp.maximum)
        (c) V_O2 ≥ 0 (mandatory clamp)
        (d) Covariance is positive semi-definite (all eigenvalues ≥ −1e-4)
    """
    filt  = CardioStateFilter()
    state = GaussianState(mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT)
    trans = CardioTransitionParams(
        cardio  = DEFAULT_CARDIO_SLICE_PARAMS,
        dt_min  = 1.0,
    )

    rng = np.random.default_rng(42)
    # 30 minutes of resting HR (around 55 bpm with small noise)
    hr_series = 55.0 + rng.normal(0.0, 3.0, 30)

    for step, hr_obs in enumerate(hr_series):
        state = filt.update_state(
            prior        = state,
            observations = {
                "HR_obs_bpm":    float(hr_obs),
                "VO2_obs":       float("nan"),   # missing — predict-only
                "RMSSD_obs_ms":  float("nan"),   # missing — predict-only
            },
            controls     = {
                "power_watts":     0.0,   # at rest
                "hub_T_core":     37.0,
                "hub_pv_drop_pct": 0.0,
            },
            params = trans,
        )

    mean_np = np.array(state.mean, dtype=np.float64)
    cov_np  = np.array(state.cov,  dtype=np.float64)

    print(f"\n[T3] posterior_mean = {mean_np}")
    print(f"[T3] W'_bal = {mean_np[IDX_WPRIME]:.4f} kJ   VO2 = {mean_np[IDX_VO2]:.4f} mL/kg/min")

    # (a) No NaN
    assert not np.any(np.isnan(mean_np)), (
        f"NaN in posterior_mean after 30 UKF steps: {mean_np}"
    )

    # (b) W'_bal ≥ 0 (mandatory clamp)
    assert mean_np[IDX_WPRIME] >= 0.0, (
        f"W'_bal went negative: {mean_np[IDX_WPRIME]:.6f} kJ"
    )

    # (c) V_O2 ≥ 0 (mandatory clamp)
    assert mean_np[IDX_VO2] >= 0.0, (
        f"V_O2 went negative: {mean_np[IDX_VO2]:.6f} mL/kg/min"
    )

    # (d) Covariance PSD (numerical symmetrisation allows tiny negative eigvals)
    eigvals = np.linalg.eigvalsh(cov_np)
    min_eigval = float(eigvals.min())
    print(f"[T3] min covariance eigenvalue = {min_eigval:.2e}")
    assert min_eigval >= -1e-4, (
        f"Covariance not PSD: min eigenvalue = {min_eigval:.2e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner (also usable as: python tests/test_cardiorespiratory_slice.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback, sys

    tests = [
        ("T1 — Cardiovascular Drift",         test_cardiovascular_drift),
        ("T2 — W' Depletion + Steal",          test_w_prime_respiratory_steal),
        ("T3 — UKF Stability (sparse obs)",   test_ukf_stability),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            fn()
            print(f"  PASSED ✓")
            passed += 1
        except Exception as exc:
            print(f"  FAILED ✗ — {exc}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{len(tests)} tests passed")
    print(f"{'='*60}")
    sys.exit(0 if passed == len(tests) else 1)
