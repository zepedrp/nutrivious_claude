"""
tests/test_sleep_circadian_slice.py  —  Gate-Zero tests for Sleep-Circadian V2.0

Four invariants that must hold before the slice is promoted to production:

  T1  test_light_induced_phase_shift
        10 000 lux at midnight suppresses melatonin AND delays the SCN phase.

  T2  test_cortisol_insomnia_block
        Chronic-stress cortisol (1 000 nmol/L) inhibits melatonin secretion,
        weakening the mel_gate so Sleep_Drive_SWS is lower than in the control.

  T3  test_adenosine_training_accumulation
        hub_training_stress=1.0 during 10 h of simulated wakefulness leaves
        significantly more Adenosine than the no-stress baseline.

  T4  test_ukf_sleep_assimilation
        48 half-hour UKF steps (24 h) complete without NaN and keep P PSD.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from app.slices.sleep_circadian.ode import (
    HubInputs,
    DEFAULT_SLEEP_PARAMS,
    IDX_ADENOSINE,
    IDX_MELATONIN,
    IDX_SWS,
    integrate_hours,
    simulate_nsteps,
    initial_state,
    hub_circadian_phase,
)
from app.slices.sleep_circadian.filter import filter_nsteps
from app.slices.sleep_circadian.observation import h_sleep, DEFAULT_OBS_PARAMS


# ─────────────────────────────────────────────────────────────────────────────
# T1  Light-induced phase shift + melatonin suppression
# ─────────────────────────────────────────────────────────────────────────────

def test_light_induced_phase_shift():
    """
    Starting at midnight (initial_state default), integrate 4 hours under
    10 000 lux vs complete darkness.

    Mechanistic expectations:
      • mel_light_gate = max(0, 1 − lux/100) → 0 at 10 000 lux → melatonin
        secretion stops entirely → Melatonin decays faster in the light arm.
      • Photic drive B = G_phot * clip(lux/10000, 0, 1) perturbs the FJK
        oscillator; over 4 h the cumulative effect delays the circadian phase
        (phase_light < phase_dark), confirmed numerically.
    """
    params = DEFAULT_SLEEP_PARAMS
    x0 = initial_state(params)

    hubs_dark  = HubInputs(hub_light_lux=0.0,      hub_training_stress=0.0, hub_Cortisol_nmolL=100.0)
    hubs_light = HubInputs(hub_light_lux=10_000.0, hub_training_stress=0.0, hub_Cortisol_nmolL=100.0)

    T_HOURS = 4.0
    x_dark  = integrate_hours(x0, hubs=hubs_dark,  params=params, t_hours=T_HOURS)
    x_light = integrate_hours(x0, hubs=hubs_light, params=params, t_hours=T_HOURS)

    mel_dark  = float(x_dark[IDX_MELATONIN])
    mel_light = float(x_light[IDX_MELATONIN])

    # Melatonin strongly suppressed by bright light
    assert mel_light < mel_dark, (
        f"Melatonin not suppressed: light={mel_light:.3f}, dark={mel_dark:.3f}"
    )
    assert mel_light < 0.70 * mel_dark, (
        f"Suppression insufficient: ratio light/dark = {mel_light / mel_dark:.3f} (need < 0.70)"
    )

    phase_dark  = float(hub_circadian_phase(x_dark))
    phase_light = float(hub_circadian_phase(x_light))

    # Phase delay: 4 h of midnight light leaves SCN behind the dark trajectory
    assert phase_light < phase_dark, (
        f"Expected phase delay (light < dark): light={phase_light:.4f} rad, dark={phase_dark:.4f} rad"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T2  Cortisol insomnia block
# ─────────────────────────────────────────────────────────────────────────────

def test_cortisol_insomnia_block():
    """
    Chronic-stress cortisol (1 000 nmol/L) vs normal nighttime level (100 nmol/L).
    Both scenarios start deep in subjective night with abundant Adenosine (1.20)
    and an initial Melatonin bolus (50 pg/mL).

    Mechanism: cort_inhib = 1 / (1 + (cort/K_cort)²)
      cort=100  → cort_inhib ≈ 0.96  → M_drive largely intact → Mel stays higher
      cort=1000 → cort_inhib = 0.20  → M_drive cut 5×      → Mel decays faster

    A lower Mel weakens the mel_gate → lower SWS_eq → lower Sleep_Drive_SWS.
    """
    params = DEFAULT_SLEEP_PARAMS

    # Deep subjective night: SCN_x strongly negative → large melatonin drive.
    # Adenosine well above threshold; Melatonin at initial burst peak.
    x0 = jnp.array([
        -0.80,  # SCN_x
         0.10,  # SCN_y
         1.20,  # Adenosine (above Aden_thr=0.50)
        50.0,   # Melatonin [pg/mL] — initial peak
         0.20,  # SWS — early consolidation
    ])

    hubs_ctrl   = HubInputs(hub_light_lux=0.0, hub_training_stress=0.0, hub_Cortisol_nmolL=100.0)
    hubs_stress = HubInputs(hub_light_lux=0.0, hub_training_stress=0.0, hub_Cortisol_nmolL=1000.0)

    T_HOURS = 3.0
    x_ctrl   = integrate_hours(x0, hubs=hubs_ctrl,   params=params, t_hours=T_HOURS)
    x_stress = integrate_hours(x0, hubs=hubs_stress, params=params, t_hours=T_HOURS)

    mel_ctrl   = float(x_ctrl[IDX_MELATONIN])
    mel_stress = float(x_stress[IDX_MELATONIN])
    sws_ctrl   = float(x_ctrl[IDX_SWS])
    sws_stress = float(x_stress[IDX_SWS])

    # High cortisol suppresses melatonin secretion
    assert mel_stress < mel_ctrl, (
        f"Cortisol should suppress melatonin: stress={mel_stress:.2f}, ctrl={mel_ctrl:.2f}"
    )

    # Weaker melatonin gate → lower SWS drive in stress scenario
    assert sws_stress < sws_ctrl, (
        f"Cortisol stress should block SWS: stress={sws_stress:.3f}, ctrl={sws_ctrl:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T3  Adenosine training accumulation
# ─────────────────────────────────────────────────────────────────────────────

def test_adenosine_training_accumulation():
    """
    hub_training_stress=1.0 during 10 simulated hours of daytime wakefulness
    (20 half-hour steps, daytime lux=500 so melatonin is suppressed → SWS≈0
    → clearance term negligible → accumulation dominates).

    Mechanism: stress_mult = 1 + k_stress * stress = 1 + 0.5 * 1.0 = 1.5
    → 50 % faster accumulation in wake.

    Expected: aden_stress / aden_ctrl > 1.20 after 10 h.
    """
    params = DEFAULT_SLEEP_PARAMS

    # Morning rested state: low Adenosine, awake (SWS ≈ 0)
    x0 = jnp.array([
         0.60,  # SCN_x — post-dawn
        -0.30,  # SCN_y
         0.10,  # Adenosine — low after overnight glymphatic clearance
         5.0,   # Melatonin — daytime nadir
         0.02,  # SWS — awake
    ])

    T_STEPS = 20        # 20 × 0.5 h = 10 h
    DAYTIME_LUX = 500.0
    CORT_DAY = 150.0    # nmol/L — normal daytime cortisol

    hubs_ctrl   = jnp.tile(jnp.array([DAYTIME_LUX, 0.0, CORT_DAY]), (T_STEPS, 1))
    hubs_stress = jnp.tile(jnp.array([DAYTIME_LUX, 1.0, CORT_DAY]), (T_STEPS, 1))

    traj_ctrl   = simulate_nsteps(x0, hubs_ctrl,   params=params, dt_hours=0.5)
    traj_stress = simulate_nsteps(x0, hubs_stress, params=params, dt_hours=0.5)

    aden_ctrl   = float(traj_ctrl[-1,   IDX_ADENOSINE])
    aden_stress = float(traj_stress[-1, IDX_ADENOSINE])

    assert aden_stress > aden_ctrl, (
        f"Training stress should increase Adenosine: stress={aden_stress:.4f}, ctrl={aden_ctrl:.4f}"
    )

    ratio = aden_stress / max(aden_ctrl, 1e-9)
    assert ratio > 1.20, (
        f"Expected ≥20 %% more adenosine with training stress; got ratio={ratio:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T4  UKF assimilation — 48 steps, no NaN, covariance PSD maintained
# ─────────────────────────────────────────────────────────────────────────────

def test_ukf_sleep_assimilation():
    """
    Run the UKF for 48 half-hour steps (24 h) assimilating synthetic wearable
    observations generated from the forward ODE trajectory.

    Assertions (Fail-Loud contract):
      1. No NaN in any posterior mean.
      2. min_eigval(P_t) >= -1e-4 at every step (covariance remains PSD).
    """
    params    = DEFAULT_SLEEP_PARAMS
    obs_params = DEFAULT_OBS_PARAMS

    T   = 48
    rng = np.random.default_rng(42)
    x0  = initial_state(params)

    # Build hub sequence: first half = daytime (lux, cortisol high),
    #                     second half = night (dark, low cortisol)
    hubs_arr = np.zeros((T, 3))
    hubs_arr[:24, 0] = 200.0    # daytime lux
    hubs_arr[:24, 2] = 200.0    # daytime cortisol [nmol/L]
    hubs_arr[24:, 0] = 0.0      # dark night
    hubs_arr[24:, 2] = 80.0     # nighttime cortisol

    # Forward ODE trajectory as synthetic ground truth
    traj = simulate_nsteps(x0, jnp.array(hubs_arr), params=params, dt_hours=0.5)
    # traj shape: (T+1, STATE_DIM); traj[0] = x0

    # Synthetic observations: h(x_{t+1}) + small Gaussian noise
    y_obs = np.zeros((T, 2))
    for t in range(T):
        y_true    = np.array(h_sleep(traj[t + 1], obs_params=obs_params))
        noise     = rng.normal(0.0, [obs_params.sigma_sws * 0.5, obs_params.sigma_phase * 0.5])
        y_obs[t]  = y_true + noise

    # Run the UKF
    means, covs = filter_nsteps(
        y_obs_sequence = jnp.array(y_obs),
        hubs_sequence  = jnp.array(hubs_arr),
        params         = params,
        obs_params     = obs_params,
        x0             = x0,
    )

    assert means.shape == (T, 5),    f"Unexpected means shape: {means.shape}"
    assert covs.shape  == (T, 5, 5), f"Unexpected covs shape:  {covs.shape}"

    # No NaN in posterior means
    means_np = np.array(means)
    assert not np.any(np.isnan(means_np)), (
        f"NaN detected in UKF posterior means at steps: "
        f"{np.where(np.any(np.isnan(means_np), axis=1))[0].tolist()}"
    )

    # Covariance PSD at every step
    covs_np = np.array(covs)
    for t in range(T):
        eigvals  = np.linalg.eigvalsh(covs_np[t])
        min_eig  = float(eigvals.min())
        assert min_eig >= -1e-4, (
            f"Step {t}: covariance not PSD — min_eigval={min_eig:.2e}"
        )
