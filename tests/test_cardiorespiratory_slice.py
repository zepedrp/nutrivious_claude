"""
tests/test_cardiorespiratory_slice.py

Gate Zero -- Cardiorespiratory Slice (8-state, FASE 2)

T1  test_cardiovascular_drift
    Constant power; hub_pv_drop rises + hub_T_core rises (Coyle 2001).
    Expected: dSV/dt more negative, dHR/dt more positive.

T2  test_w_prime_respiratory_steal
    Power >> CP: both W_fast and W_slow drain; Resp_Fatigue rises.
    Exhausted-state constraint check fires >= 1 violation.

T3  test_ukf_stability
    Assimilate 30 steps of HR_obs only (VO2=NaN, RMSSD=NaN).
    Expected: no NaN, W_fast>=0, W_slow>=0, covariance PSD.

T4  test_caen_2021_biphasic_recovery
    Depleted state at rest: verify W_fast recovers visibly faster than W_slow.
    Ratio of recovered fractions must exceed 3x (reflects tau_slow/tau_fast >= 15x).

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
import diffrax

from app.slices.cardiorespiratory.ode import (
    cardiorespiratory_slice_ode,
    DEFAULT_CARDIO_SLICE_PARAMS,
    X0_CARDIO_DEFAULT,
    P0_CARDIO_DEFAULT,
    IDX_VO2, IDX_HR, IDX_SV,
    IDX_WFAST, IDX_WSLOW,
    IDX_RF, IDX_AT, IDX_RMSSD7D,
    STATE_DIM,
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rest_state_8(params=DEFAULT_CARDIO_SLICE_PARAMS) -> jax.Array:
    """8-element resting state at population-mean capacity."""
    phi = params.W_prime_phi
    W   = params.W_prime_capacity
    return jnp.array([
        params.VO2_rest,
        params.HR_basal,
        params.SV_ref,
        phi * W,              # W_fast fully charged
        (1.0 - phi) * W,      # W_slow fully charged
        0.01,                 # RF near zero
        1.00,                 # AT full vagal tone
        params.RMSSD_ref_ms,  # RMSSD_7d at rest
    ], dtype=jnp.float32)


# ─────────────────────────────────────────────────────────────────────────────
# T1: Cardiovascular Drift
# ─────────────────────────────────────────────────────────────────────────────

def test_cardiovascular_drift():
    """
    Inject hub_pv_drop_pct (plasma-volume loss) and hub_T_core (thermal load).
    Both channels must depress SV and accelerate HR per Coyle 2001.

    Verified at ODE derivative level -- no full integration required.
    """
    params = DEFAULT_CARDIO_SLICE_PARAMS

    # Exercising steady state (200 W, 30 min in): W split proportionally
    phi = params.W_prime_phi
    W   = params.W_prime_capacity
    x_ss = jnp.array(
        [40.0, 155.0, params.SV_ref, phi*12.0, (1-phi)*12.0, 0.10, 0.40, 30.0],
        dtype=jnp.float32,
    )

    dx_base = cardiorespiratory_slice_ode(
        jnp.float32(0.0), x_ss,
        (params, 200.0, float("nan"), 0.0)
    )
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

    assert dSV_drift < dSV_base, (
        f"Expected dSV/dt more negative under CV drift: "
        f"base={dSV_base:.5f}, drift={dSV_drift:.5f}"
    )
    assert dHR_drift > dHR_base, (
        f"Expected dHR/dt larger under thermal + PV drift: "
        f"base={dHR_base:.3f}, drift={dHR_drift:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T2: W' Depletion + Respiratory Steal + Constraint Check
# ─────────────────────────────────────────────────────────────────────────────

def test_w_prime_respiratory_steal():
    """
    Power >> CP (400 W > 250 W): both W_fast and W_slow drain; RF climbs.
    Exhausted-state constraint evaluation fires on HR ceiling.
    """
    params = DEFAULT_CARDIO_SLICE_PARAMS
    phi    = params.W_prime_phi

    # Near-exhaustion: W_fast and W_slow both nearly gone
    x_exhaust = jnp.array(
        [46.0, 182.0, 0.060, 0.2, 0.3, 0.70, 0.10, 28.0],
        dtype=jnp.float32,
    )

    dx = cardiorespiratory_slice_ode(
        jnp.float32(0.0), x_exhaust,
        (params, 400.0, 37.0, 0.0)
    )

    dWfast_dt = float(dx[IDX_WFAST])
    dWslow_dt = float(dx[IDX_WSLOW])
    dRF_dt    = float(dx[IDX_RF])

    print(f"\n[T2] dW_fast/dt={dWfast_dt:.4f} kJ/min  (expected < 0)")
    print(f"[T2] dW_slow/dt={dWslow_dt:.4f} kJ/min  (expected < 0)")
    print(f"[T2] dRF/dt={dRF_dt:.4f} adim/min  (expected > 0)")

    assert dWfast_dt < 0, f"W_fast must drain at P >> CP; got {dWfast_dt:.4f}"
    assert dWslow_dt < 0, f"W_slow must drain at P >> CP; got {dWslow_dt:.4f}"
    assert dRF_dt > 0,    f"Resp_Fatigue must rise above RF_threshold; got {dRF_dt:.4f}"

    # Constraint evaluation on exhausted hub state
    env = build_cardiorespiratory_envelope({})
    state_exhausted = {
        "Hub_HR_bpm":         200.0,
        "Hub_W_prime_kJ":     0.0,
        "Hub_Resp_Fatigue":   0.90,
    }
    violations = check_all_constraints(env, state_exhausted)
    print(f"[T2] Constraint violations: {violations}")
    assert len(violations) >= 1, (
        f"Expected >= 1 constraint violation; got: {violations}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T3: UKF Stability (8-state, sparse observations)
# ─────────────────────────────────────────────────────────────────────────────

def test_ukf_stability():
    """
    Assimilate 30 minutes of resting HR only (VO2=NaN, RMSSD=NaN).
    8-state filter must remain numerically stable and obey physical clamps.

    Checks:
        (a) No NaN in posterior mean after 30 steps
        (b) W_fast >= 0 (mandatory clamp)
        (c) W_slow >= 0 (mandatory clamp)
        (d) V_O2  >= 0
        (e) Covariance PSD (min eigenvalue >= -1e-4)
    """
    filt  = CardioStateFilter()
    state = GaussianState(mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT)
    trans = CardioTransitionParams(
        cardio  = DEFAULT_CARDIO_SLICE_PARAMS,
        dt_min  = 1.0,
    )

    rng = np.random.default_rng(42)
    hr_series = 55.0 + rng.normal(0.0, 3.0, 30)

    for step, hr_obs in enumerate(hr_series):
        state = filt.update_state(
            prior        = state,
            observations = {
                "HR_obs_bpm":   float(hr_obs),
                "VO2_obs":      float("nan"),
                "RMSSD_obs_ms": float("nan"),
            },
            controls = {
                "power_watts":     0.0,
                "hub_T_core":      37.0,
                "hub_pv_drop_pct": 0.0,
            },
            params = trans,
        )

    mean_np = np.array(state.mean, dtype=np.float64)
    cov_np  = np.array(state.cov,  dtype=np.float64)

    print(f"\n[T3] posterior_mean = {mean_np}")
    print(f"[T3] W_fast={mean_np[IDX_WFAST]:.4f} kJ  W_slow={mean_np[IDX_WSLOW]:.4f} kJ")
    print(f"[T3] V_O2={mean_np[IDX_VO2]:.4f} mL/kg/min")

    assert not np.any(np.isnan(mean_np)), f"NaN in posterior_mean: {mean_np}"
    assert mean_np[IDX_WFAST] >= 0.0,    f"W_fast negative: {mean_np[IDX_WFAST]:.6f}"
    assert mean_np[IDX_WSLOW] >= 0.0,    f"W_slow negative: {mean_np[IDX_WSLOW]:.6f}"
    assert mean_np[IDX_VO2]   >= 0.0,    f"V_O2 negative: {mean_np[IDX_VO2]:.6f}"

    eigvals = np.linalg.eigvalsh(cov_np)
    min_eig = float(eigvals.min())
    print(f"[T3] min covariance eigenvalue = {min_eig:.2e}")
    assert min_eig >= -1e-4, f"Covariance not PSD: min eigenvalue = {min_eig:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# T4: Caen 2021 Biphasic Recovery
# ─────────────────────────────────────────────────────────────────────────────

def test_caen_2021_biphasic_recovery():
    """
    Caen et al. (2021) W' bi-exponential: fast pool recovers faster than slow.

    Setup:
        - Both pools are heavily depleted: W_fast = 10% cap, W_slow = 10% cap.
        - Apply 0 W (pure rest) for 5 minutes to observe recovery.
        - Recovery fraction = (W_i_final - W_i_0) / (W_i_cap - W_i_0).

    Expected: frac_fast / frac_slow >> 1  (tau_slow/tau_fast = 30/2 = 15x).
    Minimum ratio asserted: 3x (conservative; actual ~8-10x over 5 min).

    Physiological interpretation: The PCr-linked fast pool (tau~2 min) is
    substantially recharged within 5 min, while the metabolic slow pool
    (tau~30 min) barely moves -- consistent with post-sprint PCr resynthesis
    kinetics (Harris 1976; Caen 2021 Table 1).
    """
    params = DEFAULT_CARDIO_SLICE_PARAMS
    phi    = params.W_prime_phi
    W_cap  = params.W_prime_capacity

    W_fast_cap = phi * W_cap            # 0.40 * 18 = 7.2 kJ
    W_slow_cap = (1.0 - phi) * W_cap    # 0.60 * 18 = 10.8 kJ

    # Depleted start: both pools at 10% of capacity
    W_fast_0 = 0.10 * W_fast_cap   # 0.72 kJ
    W_slow_0 = 0.10 * W_slow_cap   # 1.08 kJ

    x_depleted = jnp.array([
        params.VO2_rest,
        params.HR_basal,
        params.SV_ref,
        W_fast_0,
        W_slow_0,
        0.01,
        0.50,
        params.RMSSD_ref_ms * 0.5,
    ], dtype=jnp.float32)

    # Integrate 5 min at rest (P=0, no thermal or PV stress)
    sol = diffrax.diffeqsolve(
        terms   = diffrax.ODETerm(cardiorespiratory_slice_ode),
        solver  = diffrax.Tsit5(),
        t0      = jnp.float32(0.0),
        t1      = jnp.float32(5.0),
        dt0     = jnp.float32(0.1),
        y0      = x_depleted,
        args    = (params, 0.0, 37.0, 0.0),
        saveat  = diffrax.SaveAt(t1=True),
        max_steps = 128,
    )
    x_final = sol.ys[0]

    W_fast_f = float(x_final[IDX_WFAST])
    W_slow_f = float(x_final[IDX_WSLOW])

    deficit_fast = W_fast_cap - float(W_fast_0)
    deficit_slow = W_slow_cap - float(W_slow_0)

    frac_fast = (W_fast_f - float(W_fast_0)) / max(deficit_fast, 1e-9)
    frac_slow = (W_slow_f - float(W_slow_0)) / max(deficit_slow, 1e-9)

    ratio = frac_fast / max(frac_slow, 1e-9)

    print(f"\n[T4] W_fast: {W_fast_0:.3f} -> {W_fast_f:.3f} kJ  (cap {W_fast_cap:.1f} kJ)")
    print(f"[T4] W_slow: {W_slow_0:.3f} -> {W_slow_f:.3f} kJ  (cap {W_slow_cap:.1f} kJ)")
    print(f"[T4] Recovery fraction fast={frac_fast:.4f}  slow={frac_slow:.4f}")
    print(f"[T4] frac_fast / frac_slow = {ratio:.2f}x  (expected >= 3.0x)")

    # Fast pool must actually recover
    assert W_fast_f > float(W_fast_0), (
        f"W_fast must increase at rest; got {W_fast_f:.4f} <= {W_fast_0:.4f}"
    )
    # Slow pool must also recover (just slower)
    assert W_slow_f > float(W_slow_0), (
        f"W_slow must increase at rest; got {W_slow_f:.4f} <= {W_slow_0:.4f}"
    )
    # Biphasic ratio: fast pool recovers at least 3x more than slow pool
    assert ratio >= 3.0, (
        f"Caen 2021 biphasic recovery: expected fast/slow fraction >= 3.0x; "
        f"got {ratio:.2f}x  (fast={frac_fast:.4f}, slow={frac_slow:.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback, sys

    tests = [
        ("T1 -- Cardiovascular Drift",          test_cardiovascular_drift),
        ("T2 -- W' Depletion + Steal",           test_w_prime_respiratory_steal),
        ("T3 -- UKF Stability (sparse obs)",    test_ukf_stability),
        ("T4 -- Caen 2021 Biphasic Recovery",   test_caen_2021_biphasic_recovery),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            fn()
            print(f"  PASSED")
            passed += 1
        except Exception as exc:
            print(f"  FAILED -- {exc}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{len(tests)} tests passed")
    print(f"{'='*60}")
    sys.exit(0 if passed == len(tests) else 1)
