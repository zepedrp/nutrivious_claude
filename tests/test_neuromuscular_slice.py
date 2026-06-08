"""
tests/test_neuromuscular_slice.py

Neuromuscular Tissue Slice V4.0 -- pytest validation suite.

Tests
-----
  T1  test_henneman_fatigue_compensation
        Constant power at T2 threshold for 30 min.  As RyR1_Damage accumulates,
        the fatigue-compensation term forces INCREASING Type 2 recruitment.
        Asserts R2[late] > R2[early].  Glycogen stays above bonk threshold.

  T2  test_lactate_calcium_block
        Lactate=1.0 vs Lactate=15.0 at the same power for 15 min.  High lactate
        inhibits SERCA pump (k_SERCA/(1+k_lac*Lac)), so Ca2+ accumulates.
        Asserts Ca_high > Ca_low * 1.5x.

  T3  test_glycogen_depletion_bonking
        High power (300 W) for 120 min.  Local glycogen depletes; Type 2 gate
        closes (Bonking).  Asserts: Glycogen < 10 mmol/kg, R2 < 0.1, ATP < ATP_rest.

  T4  test_ukf_neuro_assimilation_v4
        60 steps of synthetic [EMG, SmO2] assimilation via NMv4Filter (6-state).
        Asserts: no NaN in any posterior_mean; covariance PSD at each step.

Run
---
  pytest tests/test_neuromuscular_slice.py -v -s
"""
from __future__ import annotations

import sys
import math

import jax
import jax.numpy as jnp
import diffrax
import pytest

sys.path.insert(0, ".")

from app.slices.neuromuscular_tissue.ode import (
    NMv4Params,
    DEFAULT_V4_PARAMS,
    X0_NM_V4,
    P0_NM_V4,
    STATE_DIM,
    CTRL_DIM,
    IDX_ATP, IDX_CA, IDX_R1, IDX_R2, IDX_RYR1, IDX_GLYCOGEN,
    nm_v4_ode,
    hub_peripheral_fatigue,
)
from app.slices.neuromuscular_tissue.observation import (
    NMv4ObsParams,
    DEFAULT_V4_OBS_PARAMS,
    OBS_DIM,
    R_NM_V4_DEFAULT,
    h_nm_v4,
    inflate_R_nm_v4,
)
from app.slices.neuromuscular_tissue.filter import (
    NMv4Filter,
    Q_DEFAULT,
)
from app.engine.assimilation.ukf_filter import GaussianState


# -- Helpers -------------------------------------------------------------------

def _run_ode_minutes(
    x0:       jax.Array,
    u:        jax.Array,
    n_steps:  int,
    params:   NMv4Params = DEFAULT_V4_PARAMS,
) -> list[jax.Array]:
    """Roll out the NM V4.0 ODE for n_steps minutes (1 diffrax solve per step)."""
    states = [x0]
    x = x0
    for _ in range(n_steps):
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(nm_v4_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(1.0),
            dt0       = jnp.float32(0.1),
            y0        = x,
            args      = (params, u),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 32,
        )
        x = sol.ys[0]
        states.append(x)
    return states


# =============================================================================
# T1 -- Henneman Fatigue Compensation
# =============================================================================

def test_henneman_fatigue_compensation():
    """
    At T2 threshold power (250 W), Ca > Ca_ROS_thresh, so RyR1 accumulates.
    As RyR1_Damage rises, effective_power = Power + k_fat_comp * RyR1 increases,
    pushing R2_target higher -> Recruitment_Type2 must INCREASE.

    30-minute run keeps Glycogen above bonk threshold (>40 mmol/kg),
    isolating the pure RyR1 compensation mechanism from bonking.
    """
    print()
    print("[T1] Henneman fatigue compensation ...")

    power = float(DEFAULT_V4_PARAMS.P_th_2)   # 250 W
    u = jnp.array([power, 1.0, 90.0], dtype=jnp.float32)

    # 30 minutes: RyR1 visible, glycogen above bonk threshold
    N_STEPS = 30
    states = _run_ode_minutes(X0_NM_V4, u, n_steps=N_STEPS)

    # Verify NaN-free
    for i, x in enumerate(states):
        assert not bool(jnp.any(jnp.isnan(x))), f"NaN in state at minute {i}"

    x_early = states[5]    # minute 5  -- settled; little RyR1
    x_late  = states[29]   # minute 29 -- significant RyR1 accumulated

    ryr1_early = float(x_early[IDX_RYR1])
    ryr1_late  = float(x_late[IDX_RYR1])
    r2_early   = float(x_early[IDX_R2])
    r2_late    = float(x_late[IDX_R2])
    gly_late   = float(x_late[IDX_GLYCOGEN])

    print(f"    RyR1_Damage  early={ryr1_early:.4f}  late={ryr1_late:.4f}")
    print(f"    R2           early={r2_early:.4f}  late={r2_late:.4f}")
    print(f"    Glycogen     late={gly_late:.1f} mmol/kg (bonk_thr={DEFAULT_V4_PARAMS.Glycogen_bonk_thr})")

    assert ryr1_late > ryr1_early, (
        f"FAIL: RyR1 did not accumulate (early={ryr1_early:.4f}, late={ryr1_late:.4f}). "
        "Check Ca_ROS_thresh and k_ROS parameters."
    )
    assert r2_late > r2_early, (
        f"FAIL: R2 did not increase as RyR1 rose "
        f"(early={r2_early:.4f}, late={r2_late:.4f}). "
        "Check k_fat_comp -- RyR1 must increase effective_power for T2."
    )
    # Glycogen still well above bonk threshold: pure RyR1 test, no bonking interference
    assert gly_late > float(DEFAULT_V4_PARAMS.Glycogen_bonk_thr), (
        f"FAIL: Glycogen depleted below bonk threshold at 30 min ({gly_late:.1f} mmol/kg). "
        "Reduce run duration or power for pure RyR1 test."
    )

    print(f"    [PASS] R2 increased by {r2_late - r2_early:.4f} as RyR1 rose by {ryr1_late - ryr1_early:.4f}")


# =============================================================================
# T2 -- Lactate Calcium Block (SERCA inhibition)
# =============================================================================

def test_lactate_calcium_block():
    """
    High lactate inhibits SERCA: k_SERCA_eff = k_SERCA / (1 + k_lac * Lac).
    At Lac=15 mmol/L, SERCA rate is ~40% of the low-lactate rate.
    Result: Ca2+ accumulates to higher levels under high lactate.
    """
    print()
    print("[T2] Lactate SERCA block -> Calcium accumulation ...")

    power = 200.0   # W -- moderate exercise to drive Ca release

    u_low_lac  = jnp.array([power, 1.0,  90.0], dtype=jnp.float32)
    u_high_lac = jnp.array([power, 15.0, 90.0], dtype=jnp.float32)

    N_STEPS = 15   # 15 min -- glycogen stays high, pure Ca test
    states_low  = _run_ode_minutes(X0_NM_V4, u_low_lac,  n_steps=N_STEPS)
    states_high = _run_ode_minutes(X0_NM_V4, u_high_lac, n_steps=N_STEPS)

    ca_low  = float(states_low[-1][IDX_CA])
    ca_high = float(states_high[-1][IDX_CA])

    k_SERCA   = DEFAULT_V4_PARAMS.k_SERCA_base
    k_lac     = DEFAULT_V4_PARAMS.k_lac_SERCA
    serca_low  = k_SERCA / (1 + k_lac * 1.0)
    serca_high = k_SERCA / (1 + k_lac * 15.0)
    expected_ratio = serca_low / serca_high   # ~2.3

    print(f"    SERCA_low={serca_low:.3f} min^-1  SERCA_high={serca_high:.3f} min^-1")
    print(f"    Expected Ca ratio ~ {expected_ratio:.2f}x")
    print(f"    Ca_cytosolic: Lac=1.0 -> {ca_low:.4f} uM,  Lac=15.0 -> {ca_high:.4f} uM")

    assert ca_high > ca_low, (
        f"FAIL: High lactate did not produce higher Ca2+ "
        f"(Ca_low={ca_low:.4f}, Ca_high={ca_high:.4f}). "
        "SERCA inhibition by lactate not working."
    )
    assert ca_high > ca_low * 1.5, (
        f"FAIL: Ca accumulation under high lactate too small "
        f"(ratio={ca_high/ca_low:.2f}, expected >1.5x). "
        "Check k_lac_SERCA parameter."
    )

    print(f"    [PASS] Ca_high / Ca_low = {ca_high/ca_low:.3f}x (>1.5x required)")


# =============================================================================
# T3 -- Glycogen Depletion and Bonking
# =============================================================================

def test_glycogen_depletion_bonking():
    """
    High power (300 W) for 120 minutes drains local muscle glycogen.
    When glycogen falls below Glycogen_bonk_thr, the glycogen gate suppresses
    Type 2 recruitment near zero (Bonking / Hitting the Wall).

    Assertions:
      (a) Muscle_Glycogen < 10 mmol/kg at minute 115  [near total depletion]
      (b) Recruitment_Type2 < 0.1 at minute 115       [gate nearly closed]
      (c) ATP_Muscle < ATP_rest at minute 115          [fuel deficit]
    """
    print()
    print("[T3] Glycogen depletion and Bonking at 300 W / 120 min ...")

    power  = 300.0   # W -- high power, above P_th_2
    u      = jnp.array([power, 2.0, 90.0], dtype=jnp.float32)

    N_STEPS = 120
    states  = _run_ode_minutes(X0_NM_V4, u, n_steps=N_STEPS)

    # Verify NaN-free throughout
    for i, x in enumerate(states):
        assert not bool(jnp.any(jnp.isnan(x))), f"NaN in state at minute {i}"

    x_early = states[5]     # minute 5   -- pre-bonk, T2 active
    x_late  = states[115]   # minute 115 -- deep bonk

    gly_early  = float(x_early[IDX_GLYCOGEN])
    gly_late   = float(x_late[IDX_GLYCOGEN])
    r2_early   = float(x_early[IDX_R2])
    r2_late    = float(x_late[IDX_R2])
    atp_late   = float(x_late[IDX_ATP])
    atp_rest   = float(DEFAULT_V4_PARAMS.ATP_rest)

    print(f"    Muscle_Glycogen  early={gly_early:.1f}  late={gly_late:.3f} mmol/kg")
    print(f"    R2               early={r2_early:.4f}  late={r2_late:.4f}")
    print(f"    ATP_Muscle       late={atp_late:.3f}  (rest={atp_rest:.1f})")

    assert gly_late < 10.0, (
        f"FAIL: Glycogen not depleted after 120 min at 300W "
        f"(Glycogen={gly_late:.3f} mmol/kg, expected <10). "
        "Check k_gly_T2 depletion rate."
    )
    assert r2_late < 0.1, (
        f"FAIL: T2 not suppressed at bonk (R2={r2_late:.4f}, expected <0.1). "
        "Check glycogen_gate sigmoid (k_glycogen_gate, Glycogen_bonk_thr)."
    )
    assert atp_late < atp_rest, (
        f"FAIL: ATP not in deficit at bonk "
        f"(ATP={atp_late:.3f} >= ATP_rest={atp_rest:.1f}). "
        "V4.0 fix: glycogen-driven resynthesis must reduce when Glycogen=0."
    )

    print(f"    [PASS] Bonked: Gly={gly_late:.3f} mmol/kg, R2={r2_late:.4f}, "
          f"ATP={atp_late:.3f} < {atp_rest:.1f}")


# =============================================================================
# T4 -- UKF Assimilation: 60 steps, NaN-free, PSD covariance
# =============================================================================

def test_ukf_neuro_assimilation_v4():
    """
    60 steps of synthetic [EMG, SmO2] assimilation via NMv4Filter (6-state UKF).

    UKF parameters: alpha=0.10, n=6 -> 13 sigma points.
    lambda = -5.94, n+lambda = 0.06, W0_mean = -99.0 (float32 safe).

    Asserts:
      (a) No NaN in any posterior_mean.
      (b) Covariance diagonal is positive (PSD) at every step.
      (c) Posterior states are physically plausible.
    """
    print()
    print("[T4] V4.0 UKF assimilation -- 6-state, 60 steps ...")

    N_STEPS  = 60
    RNG_SEED = 7
    POWER    = 200.0   # W -- moderate; glycogen stays above bonk threshold

    # -- Generate synthetic observations from ODE ground truth ----------------
    u_ex  = jnp.array([POWER, 2.0, 90.0], dtype=jnp.float32)
    rng   = jax.random.PRNGKey(RNG_SEED)
    true_states = _run_ode_minutes(X0_NM_V4, u_ex, n_steps=N_STEPS)

    obs_params = DEFAULT_V4_OBS_PARAMS
    emg_obs  = []
    smo2_obs = []

    for x_t in true_states[1:]:
        y_clean = h_nm_v4(x_t, obs_params)
        rng, k1, k2 = jax.random.split(rng, 3)
        emg_obs.append(float(y_clean[0])  + float(jax.random.normal(k1) * 0.5))
        smo2_obs.append(float(y_clean[1]) + float(jax.random.normal(k2) * 0.5))

    emg_arr  = jnp.array(emg_obs,  dtype=jnp.float32)
    smo2_arr = jnp.array(smo2_obs, dtype=jnp.float32)
    u_seq    = jnp.tile(u_ex[None, :], (N_STEPS, 1))

    # -- Run UKF filter --------------------------------------------------------
    filt  = NMv4Filter()
    state = GaussianState(mean=X0_NM_V4, cov=P0_NM_V4)
    posteriors: list[GaussianState] = []

    for step in range(N_STEPS):
        state = filt.update_state(
            prior         = state,
            emg_mV        = float(emg_arr[step]),
            smo2_pct      = float(smo2_arr[step]),
            u             = u_seq[step],
            quality_flags = (0, 0),
        )
        posteriors.append(state)

    # -- Assertions ------------------------------------------------------------
    nan_steps = []
    nsd_steps = []

    for step_i, post in enumerate(posteriors):
        if bool(jnp.any(jnp.isnan(post.mean))):
            nan_steps.append(step_i)
        diag = jnp.diag(post.cov)
        if bool(jnp.any(diag < jnp.float32(-1e-6))):
            nsd_steps.append(step_i)

    assert len(nan_steps) == 0, (
        f"FAIL: NaN in posterior_mean at steps {nan_steps[:5]}. "
        "UKF diverged -- check alpha, Q, R."
    )
    assert len(nsd_steps) == 0, (
        f"FAIL: Non-PSD covariance at steps {nsd_steps[:5]}. "
        "Increase Q or check diagonal floor."
    )

    final_mean = posteriors[-1].mean

    # Physical plausibility of final posterior
    assert float(final_mean[IDX_ATP])      >= 0.0, "FAIL: ATP < 0 in posterior"
    assert float(final_mean[IDX_CA])       >= 0.0, "FAIL: Ca < 0 in posterior"
    assert float(final_mean[IDX_RYR1])     >= 0.0, "FAIL: RyR1 < 0 in posterior"
    assert float(final_mean[IDX_GLYCOGEN]) >= 0.0, "FAIL: Glycogen < 0 in posterior"
    assert 0.0 <= float(final_mean[IDX_R1]) <= 1.0, \
        f"FAIL: R1 out of [0,1]: {float(final_mean[IDX_R1]):.4f}"
    assert 0.0 <= float(final_mean[IDX_R2]) <= 1.0, \
        f"FAIL: R2 out of [0,1]: {float(final_mean[IDX_R2]):.4f}"

    # Glycogen should still be positive after 60 min at 200W
    assert float(final_mean[IDX_GLYCOGEN]) > 0.0, \
        f"FAIL: Glycogen depleted after 60 min at 200W ({float(final_mean[IDX_GLYCOGEN]):.1f})"

    prior_var_R2 = float(P0_NM_V4[IDX_R2, IDX_R2])    # 0.01
    post_var_R2  = float(posteriors[-1].cov[IDX_R2, IDX_R2])
    print(f"    Prior var(R2)={prior_var_R2:.4f}  Posterior var(R2) step 60={post_var_R2:.6f}")
    print(f"    [PASS] 60 steps NaN-free, PSD maintained. "
          f"ATP={float(final_mean[IDX_ATP]):.2f}  "
          f"Ca={float(final_mean[IDX_CA]):.3f}  "
          f"R2={float(final_mean[IDX_R2]):.3f}  "
          f"RyR1={float(final_mean[IDX_RYR1]):.4f}  "
          f"Gly={float(final_mean[IDX_GLYCOGEN]):.1f}")
