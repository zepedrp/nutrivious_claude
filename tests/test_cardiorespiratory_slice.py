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
# T5: T-UKF Boundary Sticking Test (Simon 2010 variance_floor proof)
# ─────────────────────────────────────────────────────────────────────────────

def test_tukf_boundary_sticking():
    """
    Simon 2010 variance floor: filter must NOT get stuck at physical boundary.

    Physical background
    -------------------
    The W_fast / W_slow pools have a sigma-point artifact: large initial covariance
    (P0 var(W_fast) = 4.0 kJ^2) spreads sigma points well above the mean, so the
    UKF mean does NOT converge to zero during depletion -- high-W sigma points
    resist drain proportionally.  The real clinical boundary-sticking risk in this
    ODE is the AUTONOMIC TONE (AT in [0,1]), which is clamped by range_clamp_moments
    and can become overconfident near AT=0 (full autonomic collapse).

    Protocol
    --------
    Use a TIGHT covariance (var(AT) = 1e-4, below the floor = 1e-6 is enforced
    by the filter PROCESS NOISE not the floor, so we test this differently):

      Phase A -- exhaustion (100 steps @400W, starting from X0 resting state):
        - AT drops from 1.0 toward its equilibrium value at 400W (~0.13).
        - var(AT) is maintained above _VAR_FLOOR[IDX_AT] = 1e-6 at all times.
        - var(W_fast) and var(W_slow) are maintained above their floors.
        - No NaN in posterior after 100 steps.
        - All covariance eigenvalues >= -1e-4 (PSD guarantee).

      Phase B -- recovery (1 step @0W):
        - AT must INCREASE immediately (no overconfidence / sticking).
        - W_fast must INCREASE (ODE recovery kinetics tracked by filter).

    Key assertions
    --------------
    (a) AT_100 < 0.40  : exhaustion reached (AT near equilibrium ~0.13 at 400W).
    (b) AT_100 > 0.05  : AT stays physiologically positive (no collapse to zero).
    (c) var(AT_100) >= _VAR_FLOOR[IDX_AT]  : floor maintained on AT dimension.
    (d) var(W_fast_100) >= _VAR_FLOOR[IDX_WFAST]  : floor maintained on W_fast.
    (e) var(W_slow_100) >= _VAR_FLOOR[IDX_WSLOW]  : floor maintained on W_slow.
    (f) posterior eigenvalues >= -1e-4  : covariance PSD (no runaway errors).
    (g) AT_101 > AT_100  : recovery tracked immediately at P=0 W.
    (h) W_fast_101 > W_fast_100  : W_fast recovery begins in step 101.
    """
    from app.slices.cardiorespiratory.filter import _VAR_FLOOR

    filt  = CardioStateFilter()
    trans = CardioTransitionParams(
        cardio  = DEFAULT_CARDIO_SLICE_PARAMS,
        dt_min  = 1.0,
    )

    # Start from X0 resting (AT=1.0, W pools fully charged)
    state = GaussianState(mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT)

    # 100 steps at 400W (far above CP=250W); HR observed every 5 steps
    for step in range(100):
        hr_obs = 185.0 if step % 5 == 0 else float("nan")
        state = filt.update_state(
            prior        = state,
            observations = {
                "HR_obs_bpm":   hr_obs,
                "VO2_obs":      float("nan"),
                "RMSSD_obs_ms": float("nan"),
            },
            controls = {
                "power_watts":     400.0,
                "hub_T_core":      38.5,
                "hub_pv_drop_pct": 3.0,
            },
            params = trans,
        )

    cov_np     = np.array(state.cov,  dtype=np.float64)
    mean_np    = np.array(state.mean, dtype=np.float64)
    at_step100  = float(mean_np[IDX_AT])
    wf_step100  = float(mean_np[IDX_WFAST])
    var_at      = float(cov_np[IDX_AT,    IDX_AT])
    var_wf      = float(cov_np[IDX_WFAST, IDX_WFAST])
    var_ws      = float(cov_np[IDX_WSLOW, IDX_WSLOW])
    eigvals     = np.linalg.eigvalsh(cov_np)
    min_eig     = float(eigvals.min())

    print(f"\n[T5] After 100 steps @400W:")
    print(f"     AT={at_step100:.4f}  W_fast={wf_step100:.4f} kJ")
    print(f"     var(AT)={var_at:.2e}  var(W_fast)={var_wf:.2e}  var(W_slow)={var_ws:.2e}")
    print(f"     min eigenvalue={min_eig:.2e}")

    # (a) AT must be depressed -- equilibrium at 400W is ~0.13
    assert at_step100 < 0.40, (
        f"AT must be depressed after 100 steps @400W (exhaust equilibrium ~0.13); "
        f"got AT={at_step100:.4f}"
    )
    # (b) AT must stay physically positive
    assert at_step100 > 0.05, (
        f"AT collapsed below physiological floor; got AT={at_step100:.4f}"
    )
    # (c) variance_floor on AT dimension
    floor_at = float(_VAR_FLOOR[IDX_AT])
    assert var_at >= floor_at, (
        f"Simon 2010 variance_floor violated for AT: "
        f"var={var_at:.2e} < floor={floor_at:.2e}  (overconfidence at boundary)"
    )
    # (d) variance_floor on W_fast
    floor_wf = float(_VAR_FLOOR[IDX_WFAST])
    assert var_wf >= floor_wf, (
        f"Simon 2010 variance_floor violated for W_fast: "
        f"var={var_wf:.2e} < floor={floor_wf:.2e}"
    )
    # (e) variance_floor on W_slow
    floor_ws = float(_VAR_FLOOR[IDX_WSLOW])
    assert var_ws >= floor_ws, (
        f"Simon 2010 variance_floor violated for W_slow: "
        f"var={var_ws:.2e} < floor={floor_ws:.2e}"
    )
    # (f) Covariance PSD (Higham 1988 nearest_psd repair intact)
    assert min_eig >= -1e-4, (
        f"Posterior covariance not PSD after 100 exhaustion steps: "
        f"min eigenvalue={min_eig:.2e}"
    )

    # Phase B: one recovery step at P=0W
    state_rec = filt.update_state(
        prior        = state,
        observations = {
            "HR_obs_bpm":   float("nan"),
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

    at_step101  = float(state_rec.mean[IDX_AT])
    wf_step101  = float(state_rec.mean[IDX_WFAST])

    print(f"[T5] After step 101 @0W:")
    print(f"     AT={at_step101:.4f}  (delta={at_step101 - at_step100:+.5f})")
    print(f"     W_fast={wf_step101:.4f} kJ  (delta={wf_step101 - wf_step100:+.5f})")

    # (g) AT recovers immediately -- no sticking at exhaustion boundary
    assert at_step101 > at_step100, (
        f"AT must recover at P=0 (no boundary sticking at AT={at_step100:.4f}); "
        f"AT_100={at_step100:.4f}  AT_101={at_step101:.4f}"
    )
    # (h) W_fast begins recovery
    assert wf_step101 > wf_step100, (
        f"W_fast must begin recovery at P=0; "
        f"W_fast_100={wf_step100:.4f}  W_fast_101={wf_step101:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T6: Gate Zero -- 60-day Realistic Micro-Cycle Data
# ─────────────────────────────────────────────────────────────────────────────

def _generate_micro_cycle_records(
    n_days: int = 75,
    seed:   int = 42,
) -> list[dict]:
    """
    Generate synthetic athlete records for a HIIT/Z2/Rest micro-cycle.

    Week structure (repeating):
        Mon  HIIT  6x3min @350W (18 min total above CP)
        Tue  Z2    45min @200W  (below CP -- aerobic base)
        Wed  Rest  no exercise
        Thu  HIIT  6x3min @350W
        Fri  Z2    45min @200W
        Sat  Z2    45min @200W
        Sun  Rest  no exercise

    Per-day records:
        - Morning (08:00): RMSSD_obs_ms (HRV wearable; derived from AT state)
        - Session (10:00 onward): 1 record per minute with HR_obs_bpm + power_watts

    RMSSD generation uses an AR(1) fatigue-recovery model:
        - Hard training day: AT suppressed proportional to training stress
        - Rest day: AT recovers toward 1.0 with time constant 1 day
        - RMSSD = AT * RMSSD_BASE + N(0, 5) ms

    Returns
    -------
    list[dict] sorted by timestamp_min; total span >= n_days days.
    """
    rng        = np.random.default_rng(seed)
    RMSSD_BASE = 35.0     # ms population mean (Malik 1996)
    MINS_PER_DAY  = 1440
    SESSION_START = 600   # 10:00

    # Weekly session blocks: list of (power_W, duration_min)
    HIIT_SESSION = [(350.0, 3), (100.0, 3)] * 3   # 3x(3min on / 3min off) = 18 min
    Z2_SESSION   = [(200.0, 45)]
    REST_SESSION: list = []

    WEEK = [
        HIIT_SESSION, Z2_SESSION, REST_SESSION,
        HIIT_SESSION, Z2_SESSION, Z2_SESSION, REST_SESSION,
    ]

    # Simple HR model: HR = HR_rest + intensity_fraction * HR_range
    def _hr(power: float) -> float:
        HR_REST, HR_MAX = 58.0, 190.0
        frac = min(power / 350.0, 1.0)
        return HR_REST + frac * (HR_MAX - HR_REST)

    records: list[dict] = []
    at_state = 1.0   # AR(1) autonomous tone proxy [0, 1]

    for day in range(n_days):
        day_offset = day * MINS_PER_DAY

        # Morning RMSSD record (HR and VO2 absent -- omit keys so .get() returns nan)
        rmssd_obs = max(1.0, at_state * RMSSD_BASE + float(rng.normal(0.0, 5.0)))
        records.append({
            "timestamp_min": day_offset + 480,   # 08:00
            "RMSSD_obs_ms":  float(rmssd_obs),
            "power_watts":   0.0,
            "hub_T_core":    37.0,
            "hub_pv_drop_pct": 0.0,
        })

        # Exercise session
        dow     = day % 7
        session = WEEK[dow]
        ts      = day_offset + SESSION_START
        session_stress = 0.0

        for power, dur_min in session:
            hr_base = _hr(power)
            t_core  = 37.5 if power > 250 else 37.0
            pv_drop = 2.0  if power > 250 else 0.5
            for _ in range(dur_min):
                hr_obs = hr_base + float(rng.normal(0.0, 3.0))
                # RMSSD absent during exercise -- omit key so .get() returns nan
                records.append({
                    "timestamp_min":   ts,
                    "HR_obs_bpm":      float(hr_obs),
                    "power_watts":     float(power),
                    "hub_T_core":      float(t_core),
                    "hub_pv_drop_pct": float(pv_drop),
                })
                ts += 1
            session_stress += power * dur_min / (350.0 * 18.0)   # normalised load

        # Update AR(1) AT state (fatigue = suppression, rest = recovery)
        if session_stress > 0.0:
            at_state = max(0.15, at_state * (1.0 - 0.30 * session_stress))
        else:
            at_state = min(1.00, at_state + 0.18 * (1.0 - at_state))   # rest recovery

    return sorted(records, key=lambda r: r["timestamp_min"])


def test_gate_zero_micro_cycle_60d():
    """
    Gate Zero with a 60-day HIIT/Z2/Rest micro-cycle.

    Validates that:
    (a) Pipeline completes without error (not INSUFFICIENT_DATA).
    (b) Exactly 1 user processed with sufficient paired predictions.
    (c) n_days >= 60 paired (predict, observe) pairs scored.
    (d) MAE values are finite (no NaN/Inf leaking through the pipeline).
    (e) Circadian and L3 NLME paths both execute without crash
        (NLME falls back gracefully if numpyro absent).

    The gate outcome (PASS / FAIL) is informational only -- it depends on ODE
    parameter accuracy relative to the AR(1) synthetic HRV, which is not
    required to match perfectly.  The test probes pipeline correctness, not
    model performance.
    """
    from app.validation.backtest_engine import run_gate_zero, GateZeroDecision

    records = _generate_micro_cycle_records(n_days=75, seed=42)
    print(f"\n[T6] Generated {len(records)} records over 75 days.")

    result = run_gate_zero(
        users_data      = {"athlete_synthetic": records},
        warmup_days     = 14,
        wake_hour       = 8.0,
        nlme_train_days = 30,
        nlme_steps      = 200,    # minimal steps -- just proves wiring, not convergence
    )

    print(f"[T6] Gate Zero decision: {result.decision.value}")
    print(f"[T6] Users total={result.n_users_total}  twin_wins={result.n_users_twin_wins}")
    if result.per_user:
        u = result.per_user[0]
        print(
            f"[T6] n_days={u.n_days}  "
            f"mae_twin={u.mae_twin:.4f}  "
            f"mae_persist={u.mae_persistence:.4f}  "
            f"mae_ewma={u.mae_ewma7d:.4f}"
        )

    # (a) Pipeline completed -- not starved of data
    assert result.decision != GateZeroDecision.INSUFFICIENT_DATA, (
        "Gate Zero returned INSUFFICIENT_DATA -- data generator or pipeline broken."
    )

    # (b) Exactly 1 user processed
    assert result.n_users_total == 1, (
        f"Expected 1 user; got {result.n_users_total}"
    )

    # (c) Sufficient scored days
    n_days = result.per_user[0].n_days
    assert n_days >= 60, (
        f"Expected >= 60 paired prediction days; got {n_days}"
    )

    # (d) All MAE values finite
    u = result.per_user[0]
    assert math.isfinite(u.mae_twin),        f"mae_twin is not finite: {u.mae_twin}"
    assert math.isfinite(u.mae_persistence), f"mae_persistence is not finite: {u.mae_persistence}"
    assert math.isfinite(u.mae_ewma7d),      f"mae_ewma7d is not finite: {u.mae_ewma7d}"
    assert u.mae_twin > 0.0,                 f"mae_twin is zero -- predictions are perfect/constant"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback, sys

    tests = [
        ("T1 -- Cardiovascular Drift",             test_cardiovascular_drift),
        ("T2 -- W' Depletion + Steal",              test_w_prime_respiratory_steal),
        ("T3 -- UKF Stability (sparse obs)",       test_ukf_stability),
        ("T4 -- Caen 2021 Biphasic Recovery",      test_caen_2021_biphasic_recovery),
        ("T5 -- T-UKF Boundary Sticking",          test_tukf_boundary_sticking),
        ("T6 -- Gate Zero 60d Micro-Cycle",        test_gate_zero_micro_cycle_60d),
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
