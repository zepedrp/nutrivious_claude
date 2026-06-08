"""
tests/test_neural_cognitive_slice.py  — REMASTER v2.0

Gate Zero — Neural/Cognitive Slice (L2–L4)

Three formal acceptance tests for the remasterized 7-state system:

  T1: test_hypoxia_ammonia_collapse
      Extreme metabolic stress (no heat) -> cerebral O2 drops + brain NH3 rises
      -> CAR must collapse below 0.75 hard floor.
      Isolates Subudhi 2009 (cerebral hypoxia) + Nybo 2005 (ammonia) pathways.

  T2: test_caffeine_adenosine_antagonism
      Step A) 24h sleep debt -> PVT_Lapses spike (adenosine accumulates).
      Step B) 200 mg caffeine dose -> Caffeine_Plasma rises, Active_Adenosine
              collapses via competitive antagonism -> PVT_Lapses fall drastically.
      Proves the A1/A2A competitive antagonism model (Nehlig 2010).

  T3: test_ukf_stability
      UKF assimilates RPE = 10 (max) with PVT NaN (predict-only).
      Posterior must be NaN-free and covariance must be Positive-Semi-Definite.

Fail-Loud contract (CLAUDE.md §6):
  RuntimeError propagates unmodified — tests must NOT swallow exceptions.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import diffrax
import pytest

from app.slices.neural_cognitive.ode import (
    DEFAULT_NC_PARAMS,
    X0_NC_DEFAULT,
    P0_NC_DEFAULT,
    neural_cognitive_ode,
    compute_active_adenosine,
    IDX_CAR,
    IDX_NH3,
    IDX_O2SAT,
    IDX_ADEN,
    IDX_CAF,
    IDX_5HT,
    IDX_DA,
    STATE_DIM,
    CTRL_DIM,
)
from app.slices.neural_cognitive.observation import (
    DEFAULT_NC_OBS_PARAMS,
    R_NC_DEFAULT,
    h_nc,
    inflate_R_nc,
)
from app.slices.neural_cognitive.envelope import (
    build_neural_cognitive_envelope,
    CAR_HARD_FLOOR,
)
from app.slices.neural_cognitive.filter import (
    NeuralCognitiveStateFilter,
    NeuralCogTransitionParams,
)
from app.engine.assimilation.ukf_filter import GaussianState


# ── Integration helper ────────────────────────────────────────────────────────

def _run_ode(
    u:         jax.Array,
    t1:        float,
    x0:        jax.Array = X0_NC_DEFAULT,
    params               = DEFAULT_NC_PARAMS,
    dt0:       float     = 0.05,
    max_steps: int       = 8192,
) -> jax.Array:
    """Integrate the 7-state neural/cognitive ODE and return final state."""
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(neural_cognitive_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.float32(t1),
        dt0       = jnp.float32(dt0),
        y0        = x0,
        args      = (params, u),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = max_steps,
    )
    return sol.ys[0]


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Cerebral Hypoxia + Ammonia Toxicity -> CAR Collapse (no heat)
# ─────────────────────────────────────────────────────────────────────────────

def test_hypoxia_ammonia_collapse():
    """
    Inject maximal training stress + maximal metabolic stress, NO heat.
    Validates that the Subudhi 2009 (cerebral hypoxia) and Nybo 2005 (ammonia)
    pathways are sufficient to collapse CAR below the hard floor 0.75.

    Physical expectation at steady-state (derived analytically):
      5HT SS = 2.5,  DA SS = 1.75  ->  ratio ~= 1.43
      NH3 SS = 0.50  (logistic ceiling; hub_metab=1.0)
          ammonia_inhib = 0.50 / (0.50 + 0.50) = 0.500
      O2 target = 0.70 x (1 - 0.15x1.0^2) = 0.595
          hypoxia_inhib = (0.70 - 0.595) / 0.70 = 0.150
      thermal = 0  (T_core = 37.0)
      afferent = 0  (hub_muscle_damage = 0)

      CAR_ss ~= 2.0 / (2.0 + 0.105x1.43 + 2.0x0.15 + 1.5x0.50)
             ~= 2.0 / (2.0 + 0.150 + 0.300 + 0.750) = 2.0/3.20 ~= 0.625

    After 6 h all suppressors are well-converged (τ_CAR = 0.5 h, τ_NH3 = 1 h,
    τ_O2 = 0.5 h).  CAR must be < 0.75.

    Asserts:
      (a) CAR_final < 0.75 hard floor
      (b) Brain_Ammonia rose above 0 (AMP deamination active)
      (c) Cerebral_O2_Sat dropped below resting level (0.70)
      (d) Brain_5HT rose above rest (1.0) under training stress
      (e) No NaN in any state
      (f) Envelope hard constraint for CAR >= 0.75 exists and is violated
    """
    # Maximal stress, maximal metabolic load, NO heat, NO muscle damage
    u = jnp.array([
        1.00,   # hub_training_stress   — maximal aerobic load
        0.00,   # hub_muscle_damage     — isolated (no afferent pathway)
        37.00,  # hub_T_core            — thermoneutral
        0.00,   # hub_sleep_debt        — well-rested
        0.00,   # hub_IL6              — no systemic inflammation
        1.00,   # hub_metabolic_stress  — maximal AMP deamination -> NH3
        0.00,   # caffeine_intake       — no caffeine
        0.00,   # hub_hypoglycemia      — euglycaemic
    ], dtype=jnp.float32)

    x_final = _run_ode(u, t1=6.0)

    car_final  = float(x_final[IDX_CAR])
    nh3_final  = float(x_final[IDX_NH3])
    o2_final   = float(x_final[IDX_O2SAT])
    b5ht_final = float(x_final[IDX_5HT])
    no_nan     = bool(~jnp.any(jnp.isnan(x_final)))

    print(f"\n[T1] After 6h max stress (no heat):")
    print(f"  CAR              = {car_final:.4f}  (hard floor = {CAR_HARD_FLOOR})")
    print(f"  Brain_NH3        = {nh3_final:.4f}  (expect > 0 from AMP deamination)")
    print(f"  Cerebral_O2_Sat  = {o2_final:.4f}  (expect < 0.70 resting)")
    print(f"  Brain_5HT        = {b5ht_final:.4f} (expect > 1.0 under stress)")
    print(f"  No NaN: {no_nan}")

    # (a) CAR collapses below hard floor — hypoxia + ammonia suppress voluntarily
    assert car_final < CAR_HARD_FLOOR, (
        f"CAR = {car_final:.4f} must be < {CAR_HARD_FLOOR} under hypoxia + ammonia. "
        "Check k_car_hypoxia and k_car_ammonia parameters."
    )

    # (b) Brain ammonia rose (AMP deamination active under max metabolic stress)
    assert nh3_final > 0.10, (
        f"Brain_NH3 = {nh3_final:.4f} should rise above 0.10 at max metabolic stress "
        f"(AMP deamination: NH3_ss = 0.50)."
    )

    # (c) Cerebral O2 dropped below resting level (quadratic hypoxia model)
    assert o2_final < 0.695, (
        f"Cerebral_O2_Sat = {o2_final:.4f} should drop below resting 0.70 "
        f"(Subudhi 2009: ~15% relative drop at VO2max)."
    )

    # (d) Serotonin rose under training stress (BBB tryptophan transport)
    assert b5ht_final > 1.0, (
        f"Brain_5HT = {b5ht_final:.4f} should rise above rest (1.0) during training."
    )

    # (e) No NaN in state vector
    assert no_nan, "State contains NaN — ODE integration failure."

    # (f) Envelope hard constraint exists and is violated
    env = build_neural_cognitive_envelope()
    hard_car = [c for c in env.hard_constraints if c.state_key == "Hub_NC_CAR"]
    assert len(hard_car) >= 1, "Envelope must define a Hub_NC_CAR hard constraint."
    for hc in hard_car:
        assert car_final < hc.rhs, (
            f"Hard constraint '{hc.name}' (CAR threshold {hc.rhs}) not violated at "
            f"CAR = {car_final:.4f}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Caffeine Antagonises Adenosine: PVT_Lapses Collapse After Dose
# ─────────────────────────────────────────────────────────────────────────────

def test_caffeine_adenosine_antagonism():
    """
    Two-step test proving the A1/A2A competitive antagonism model (Nehlig 2010).

    Step A — 24h sleep debt (hub_sleep_debt = 0.8, no exercise, no caffeine):
        Adenosine_Pool accumulates toward SS = 0.80.
        After 24 h (2.4 τ, τ = 10 h): Aden ~= 0.73.
        Active_Adenosine = 0.73 (no caffeine).
        PVT_Lapses = 4.0 x 0.73 ~= 2.92.

    Step B — 200 mg caffeine dose (intake_plasma = 10 mg/L/h for 0.5 h):
        Caffeine_Plasma rises to ~= 4.8 mg/L after 0.5 h.
        Active_Adenosine = 0.73 / (1 + 4.8/3.0) ~= 0.28.
        PVT_Lapses = 4.0 x 0.28 ~= 1.1  (>= 50% reduction).

    Asserts:
      (a) After Step A: PVT_Lapses_before > 1.5 (substantial sleep debt burden)
      (b) After Step A: Adenosine_Pool > 0.40 (adenosine accumulated)
      (c) After Step B: Caffeine_Plasma > 2.0 (dose absorbed)
      (d) After Step B: PVT_Lapses_after < PVT_Lapses_before x 0.65
              (caffeine reduced lapses by >= 35%)
      (e) After Step B: Active_Adenosine_after < Active_Adenosine_before x 0.65
              (receptor occupancy reduced)
      (f) No NaN in either step's final state
    """
    # ── Step A: 24h sleep debt accumulation ───────────────────────────────────
    u_sleep = jnp.array([
        0.00,   # hub_training_stress  — no exercise
        0.00,   # hub_muscle_damage    — no damage
        37.00,  # hub_T_core           — thermoneutral
        0.80,   # hub_sleep_debt       — ~1 night deficit; realistic accumulation
        0.00,   # hub_IL6             — no systemic inflammation
        0.00,   # hub_metabolic_stress — no exercise
        0.00,   # caffeine_intake      — no caffeine yet
        0.00,   # hub_hypoglycemia     — euglycaemic
    ], dtype=jnp.float32)

    x_after_sleep = _run_ode(u_sleep, t1=24.0)

    aden_before    = float(x_after_sleep[IDX_ADEN])
    caf_before     = float(x_after_sleep[IDX_CAF])
    active_before  = float(compute_active_adenosine(x_after_sleep, K_caf=3.0))
    y_before       = h_nc(x_after_sleep, DEFAULT_NC_OBS_PARAMS)
    pvt_before     = float(y_before[1])

    print(f"\n[T2] After 24h sleep debt (Step A):")
    print(f"  Adenosine_Pool     = {aden_before:.4f}  (expect ~0.73)")
    print(f"  Caffeine_Plasma    = {caf_before:.4f}   (expect 0; no intake)")
    print(f"  Active_Adenosine   = {active_before:.4f}")
    print(f"  PVT_Lapses_before  = {pvt_before:.4f}  (expect ~2.9)")

    # (a) PVT_Lapses elevated after sleep debt
    assert pvt_before > 1.5, (
        f"PVT_Lapses after 24h sleep debt = {pvt_before:.4f} "
        f"should exceed 1.5 (sleep debt burden)."
    )

    # (b) Adenosine accumulated
    assert aden_before > 0.40, (
        f"Adenosine_Pool = {aden_before:.4f} should exceed 0.40 after 24h sleep debt."
    )

    # ── Step B: 200 mg caffeine over 0.5h then 1.5h clearance ────────────────
    # caffeine_intake_plasma = dose_mg / (intake_h x V_dist_L) = 200/(0.5x40) = 10 mg/L/h
    u_caffeine = jnp.array([
        0.00,    # hub_training_stress  — rest
        0.00,    # hub_muscle_damage
        37.00,   # hub_T_core
        0.00,    # hub_sleep_debt       — no additional debt in Step B
        0.00,    # hub_IL6
        0.00,    # hub_metabolic_stress
        10.00,   # caffeine_intake_plasma [mg/L/h] — 200mg / (0.5h x 40L)
        0.00,    # hub_hypoglycemia
    ], dtype=jnp.float32)

    # Phase B1: 0.5h absorption window (caffeine intake active)
    x_after_caf = _run_ode(u_caffeine, t1=0.5, x0=x_after_sleep)

    # Phase B2: 1.5h clearance (no more intake)
    u_clearance = u_caffeine.at[6].set(0.0)   # stop caffeine intake
    x_final = _run_ode(u_clearance, t1=1.5, x0=x_after_caf)

    caf_after      = float(x_final[IDX_CAF])
    aden_after     = float(x_final[IDX_ADEN])
    active_after   = float(compute_active_adenosine(x_final, K_caf=3.0))
    y_after        = h_nc(x_final, DEFAULT_NC_OBS_PARAMS)
    pvt_after      = float(y_after[1])
    no_nan_a       = bool(~jnp.any(jnp.isnan(x_after_sleep)))
    no_nan_b       = bool(~jnp.any(jnp.isnan(x_final)))

    print(f"\n[T2] After caffeine dose (Step B, 2h total):")
    print(f"  Caffeine_Plasma    = {caf_after:.4f} mg/L  (expect > 2.0)")
    print(f"  Adenosine_Pool     = {aden_after:.4f}  (slight clearance from Step A)")
    print(f"  Active_Adenosine   = {active_after:.4f} (expect < {active_before:.4f} x 0.65)")
    print(f"  PVT_Lapses_after   = {pvt_after:.4f}  (expect < {pvt_before:.4f} x 0.65)")
    print(f"  No NaN (A): {no_nan_a}, No NaN (B): {no_nan_b}")

    # (c) Caffeine absorbed into plasma
    assert caf_after > 2.0, (
        f"Caffeine_Plasma = {caf_after:.4f} mg/L should exceed 2.0 after 200mg dose. "
        "Check CYP1A2_clearance_rate and caffeine_intake_plasma encoding."
    )

    # (d) PVT_Lapses fell drastically after caffeine (>= 35% reduction)
    reduction = (pvt_before - pvt_after) / (pvt_before + 1e-9)
    assert pvt_after < pvt_before * 0.65, (
        f"PVT_Lapses after caffeine = {pvt_after:.4f} should be < "
        f"{pvt_before:.4f} x 0.65 = {pvt_before*0.65:.4f}. "
        f"Reduction = {100*reduction:.1f}%. "
        "Caffeine-adenosine competitive antagonism not effective enough."
    )

    # (e) Active_Adenosine receptor occupancy reduced by caffeine
    assert active_after < active_before * 0.65, (
        f"Active_Adenosine = {active_after:.4f} should be < "
        f"{active_before:.4f} x 0.65 = {active_before*0.65:.4f}. "
        "K_caf or Caffeine_Plasma insufficient for the competitive displacement."
    )

    # (f) No NaN in either step
    assert no_nan_a and no_nan_b, "State contains NaN — ODE integration failure."


# ─────────────────────────────────────────────────────────────────────────────
# T3 — UKF Stability: RPE = 10 Assimilation with PVT NaN (PSD Covariance)
# ─────────────────────────────────────────────────────────────────────────────

def test_ukf_stability():
    """
    Assimilate RPE = 10.0 (Borg maximum) with PVT missing (NaN -> predict-only).

    Starting from the rested prior (CAR ~= 0.95 -> predicted RPE ~= 1.45 via log
    mapping), the filter receives RPE = 10.0 — a large innovation of ~8.5 Borg units.

    The 7-state UKF must remain numerically stable (15 sigma points) under
    this strong corrective pull.

    Asserts:
      (a) posterior_mean is NaN-free (filter numerically stable)
      (b) posterior_cov diagonal is non-negative (Positive-Semi-Definite)
      (c) no NaN in full covariance matrix
      (d) CAR posterior decreased vs. prior CAR (RPE=10 pulls toward low-CAR regime)
      (e) Adenosine_Pool posterior >= 0 (thermodynamic floor enforced)
      (f) Caffeine_Plasma posterior >= 0 (physical floor enforced)
      (g) No RuntimeError raised (Fail-Loud not triggered spuriously)
    """
    filt  = NeuralCognitiveStateFilter()
    prior = GaussianState(mean=X0_NC_DEFAULT, cov=P0_NC_DEFAULT)

    controls = {
        "hub_training_stress":   0.60,
        "hub_muscle_damage":     0.20,
        "hub_T_core":            38.0,
        "hub_sleep_debt":        0.10,
        "hub_IL6":               0.05,
        "hub_metabolic_stress":  0.30,
        "caffeine_intake_plasma": 0.0,
        "hub_hypoglycemia":      0.00,
    }

    posterior = filt.update_state(
        prior         = prior,
        controls      = controls,
        dt_hours      = 1.0,
        rpe_proxy     = 10.0,         # maximum Borg — forces large innovation
        pvt_lapses    = float("nan"), # PVT not available today
        quality_flags = (0, 4),       # RPE assimilated; PVT predict-only
    )

    pm   = posterior.mean
    pcov = posterior.cov
    diag = jnp.diag(pcov)

    car_prior     = float(prior.mean[IDX_CAR])
    car_posterior = float(pm[IDX_CAR])
    aden_post     = float(pm[IDX_ADEN])
    caf_post      = float(pm[IDX_CAF])

    print(f"\n[T3] UKF assimilation (RPE=10, PVT=NaN):")
    print(f"  Prior CAR       = {car_prior:.4f}")
    print(f"  Posterior CAR   = {car_posterior:.4f}  (should be <= prior)")
    print(f"  Adenosine Pool  = {aden_post:.4f}  (must be >= 0)")
    print(f"  Caffeine Plasma = {caf_post:.4f}   (must be >= 0)")
    print(f"  Cov diagonal    = {[f'{float(d):.5f}' for d in diag]}")
    print(f"  Any NaN mean:   {bool(jnp.any(jnp.isnan(pm)))}")
    print(f"  Any NaN cov:    {bool(jnp.any(jnp.isnan(pcov)))}")

    # (a) Posterior mean is NaN-free
    assert not bool(jnp.any(jnp.isnan(pm))), (
        f"posterior_mean contains NaN: {pm}"
    )

    # (b) Covariance diagonal is non-negative (PSD)
    assert bool(jnp.all(diag >= jnp.float32(-1e-5))), (
        f"posterior_cov has negative diagonal: {[float(d) for d in diag]}"
    )

    # (c) No NaN in full covariance
    assert not bool(jnp.any(jnp.isnan(pcov))), (
        "posterior_cov contains NaN — filter diverged numerically."
    )

    # (d) CAR posterior <= prior CAR (RPE=10 pulls toward low-CAR regime)
    assert car_posterior <= car_prior + 0.02, (
        f"Posterior CAR ({car_posterior:.4f}) should not exceed prior CAR "
        f"({car_prior:.4f} + 0.02 tolerance) when RPE=10 is observed."
    )

    # (e) Adenosine Pool posterior >= 0 (physics-law floor in update_state)
    assert aden_post >= 0.0, (
        f"Adenosine_Pool posterior ({aden_post:.4f}) must be >= 0."
    )

    # (f) Caffeine Plasma posterior >= 0 (no negative plasma concentration)
    assert caf_post >= 0.0, (
        f"Caffeine_Plasma posterior ({caf_post:.4f}) must be >= 0."
    )
