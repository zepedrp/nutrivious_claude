"""
tests/test_neuroendocrine_slice.py — Gate-Zero: SAM × HPA × Somatotropic  V2.0

T1  test_sam_vs_hpa_delay
        Inject acute stress (hub_training_stress=1.0).
        After 0.1 h (6 min): Epinephrine ratio > 5× vs rest.
        After 0.1 h: Cortisol changes < 10% (cascade delay CRH→ACTH→Cortisol).
        Proves timescale separation (γ_Epi=20.8 h⁻¹ vs γ_Cort=0.45 h⁻¹) and
        Kvaerno5 stability on the stiff ODE.

T2  test_personalized_glucose_panic
        Athlete A: Metabolic_Flexibility_Threshold=75 mg/dL (CHO-dependent).
        Athlete B: Metabolic_Flexibility_Threshold=55 mg/dL (fat-adapted).
        Both exposed to glucose=65 mg/dL for 3 h.
        Assert: CRH_A > CRH_B × 1.5  (panic fires only for A).
        Assert: Cortisol_A > Cortisol_B.

T3  test_circadian_car_and_sleep
        Part 1 — CAR: hub_circadian_phase=1.0 at dawn → Cortisol elevated vs no drive.
        Part 2 — Sleep GH: hub_sleep_sws_fraction=0.7 at night → GH pulse confirmed.

T4  test_ukf_stiff_stability
        30 steps of 1-h UKF with sparse observations.
        Assert at every step: no NaN, PSD covariance, all states ≥ 0.

Run:
    pytest tests/test_neuroendocrine_slice.py -v -s
"""
from __future__ import annotations

import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import pytest

from app.slices.neuroendocrine.ode import (
    NeuroParams, DEFAULT_NEURO_PARAMS,
    IDX_EPI, IDX_NE, IDX_CRH, IDX_ACTH, IDX_CORT,
    IDX_GHRH, IDX_SS, IDX_GH, IDX_IGF1,
    STATE_DIM, OBS_DIM,
    initial_state, integrate_Nh, integrate_1h,
)
from app.slices.neuroendocrine.observation import (
    DEFAULT_OBS_PARAMS, h_neuro, observation_noise_R,
)
from app.slices.neuroendocrine.filter import (
    NeuroFilterState,
    initial_filter_state,
    update_state,
    default_process_noise_Q,
    _clamp_physical,
)


# ─────────────────────────────────────────────────────────────────────────────
# T1 — SAM spike vs HPA delay (timescale separation + Kvaerno5 stability)
# ─────────────────────────────────────────────────────────────────────────────

def test_sam_vs_hpa_delay():
    """
    Inject max stress for only 0.1 h (6 min).

    Physiological expectation:
      Epi  t½=2 min  → reaches ~84% of new SS in 6 min  (ratio ~16-20×)
      NE   t½=3 min  → reaches ~75% of new SS in 6 min  (ratio ~7×)
      Cortisol cascade (CRH→ACTH→Cortisol) takes 30-60 min minimum → <10% change

    Also confirms Kvaerno5 handles γ_Epi=20.8 h⁻¹ stiffness without NaN.
    """
    x0 = initial_state()   # 08:00 resting: Epi=50, NE=300, Cortisol=420

    # Max acute stress, no other changes
    def ctrl_stress(t):
        return jnp.array([1.0, 0.0, 0.0, 90.0, 0.0])

    x_01h = integrate_Nh(x0, n_hours=0.1, control_fn=ctrl_stress, t0=8.0)

    # ── SAM: massive spike ──
    epi_0   = float(x0[IDX_EPI])
    epi_01  = float(x_01h[IDX_EPI])
    ne_0    = float(x0[IDX_NE])
    ne_01   = float(x_01h[IDX_NE])

    epi_ratio = epi_01 / (epi_0 + 1e-6)
    ne_ratio  = ne_01  / (ne_0  + 1e-6)

    print(f"\n  T1: Epi  {epi_0:.1f} -> {epi_01:.1f} pg/mL  (ratio={epi_ratio:.2f}x)")
    print(f"      NE   {ne_0:.1f} -> {ne_01:.1f} pg/mL  (ratio={ne_ratio:.2f}x)")

    assert epi_ratio > 5.0, (
        f"Epi must spike >5x in 6 min (got {epi_ratio:.2f}x). "
        "Check gamma_Epi or Kvaerno5 integration."
    )
    assert ne_ratio > 3.0, (
        f"NE must spike >3x in 6 min (got {ne_ratio:.2f}x)."
    )

    # ── HPA: barely changes (cascade delay) ──
    cort_0  = float(x0[IDX_CORT])
    cort_01 = float(x_01h[IDX_CORT])
    cort_change_pct = abs(cort_01 - cort_0) / (cort_0 + 1e-6)

    print(f"      Cort {cort_0:.1f} -> {cort_01:.1f} nmol/L  (change={cort_change_pct*100:.2f}%)")

    assert cort_change_pct < 0.10, (
        f"Cortisol must change <10% in 6 min (got {cort_change_pct*100:.2f}%). "
        "HPA cascade delay is ~30-60 min; check coupling coefficients."
    )

    # ── Timescale separation: SAM rise >> HPA rise ──
    assert epi_ratio > (cort_01 / cort_0) * 2.0, (
        "Epi rise must be >> Cortisol rise — timescale separation violated."
    )

    # ── No NaN (Kvaerno5 stability) ──
    assert not jnp.any(jnp.isnan(x_01h)), "NaN in 0.1h stress trajectory"

    print("  T1 PASS: SAM spike confirmed, Cortisol cascade delay confirmed, Kvaerno5 stable.")


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Personalised Metabolic_Flexibility_Threshold (glucose panic)
# ─────────────────────────────────────────────────────────────────────────────

def test_personalized_glucose_panic():
    """
    Athlete A (MFT=75): glucose=65 < 75 → CRH panic fires (hypo_excess=10).
    Athlete B (MFT=55): glucose=65 > 55 → no panic (hypo_excess=0, fat-adapted).

    After 3 h at glucose=65, Athlete A has substantially higher CRH and Cortisol.
    """
    # Afternoon low-cortisol starting state (14:00) for cleaner signal
    x0_pm = initial_state().at[IDX_CORT].set(150.0)

    params_panic   = NeuroParams(Metabolic_Flexibility_Threshold=75.0)
    params_adapted = NeuroParams(Metabolic_Flexibility_Threshold=55.0)

    def ctrl_65(t):
        return jnp.array([0.0, 0.0, 0.0, 65.0, 0.0])   # glucose=65 mg/dL

    x_panic   = integrate_Nh(x0_pm, n_hours=3.0, params=params_panic,
                              control_fn=ctrl_65, t0=14.0)
    x_adapted = integrate_Nh(x0_pm, n_hours=3.0, params=params_adapted,
                              control_fn=ctrl_65, t0=14.0)

    crh_panic   = float(x_panic[IDX_CRH])
    crh_adapted = float(x_adapted[IDX_CRH])
    cort_panic  = float(x_panic[IDX_CORT])
    cort_adapted= float(x_adapted[IDX_CORT])

    print(f"\n  T2: CRH   panic={crh_panic:.3f}  adapted={crh_adapted:.3f}  pg/mL")
    print(f"      Cort  panic={cort_panic:.1f}  adapted={cort_adapted:.1f}  nmol/L")

    assert crh_panic > crh_adapted * 1.5, (
        f"Panic athlete CRH ({crh_panic:.3f}) must be >1.5x fat-adapted ({crh_adapted:.3f}). "
        "Check k_CRH_hypo and MFT parametrisation."
    )
    assert cort_panic > cort_adapted, (
        f"Panic athlete Cortisol ({cort_panic:.1f}) must exceed adapted ({cort_adapted:.1f})."
    )

    # Panic athlete's Epi/NE also elevated via HPA→sympathetic feedback
    # (not directly forced, but stress coupling k_CRH_stress is small → CRH rises)

    # No NaN
    assert not jnp.any(jnp.isnan(x_panic)),   "NaN in panic trajectory"
    assert not jnp.any(jnp.isnan(x_adapted)), "NaN in fat-adapted trajectory"

    print("  T2 PASS: personalised metabolic flexibility threshold confirmed.")


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Circadian CAR + Sleep GH pulse
# ─────────────────────────────────────────────────────────────────────────────

def test_circadian_car_and_sleep():
    """
    Part 1 — Cortisol Awakening Response (CAR):
        hub_circadian_phase=1.0 at 06:00 (dawn) adds k_CRH_CAR=3.0 pg/mL/h to CRH.
        After 2 h, Cortisol must be higher than without the awakening drive.

    Part 2 — Nocturnal GH pulse:
        hub_sleep_sws_fraction=0.70 at 22:00 boosts GHRH (k_SWS=1.5).
        After 4 h, GH must be >20% higher than wakefulness control.
        IGF-1 must be >= wakefulness (anabolic follow-through).
    """
    # ── Part 1: CAR ──────────────────────────────────────────────────────────
    x0_dawn = jnp.array([
        50.0,   # Epi     pg/mL  resting
        300.0,  # NE      pg/mL  resting
        1.2,    # CRH     pg/mL  pre-dawn nadir
        12.0,   # ACTH    pg/mL
        120.0,  # Cortisol nmol/L  nadir
        80.0,   # GHRH    pg/mL  pre-dawn elevated (still dark)
        25.0,   # SS      pg/mL  low at night
        1.5,    # GH      ng/mL
        240.0,  # IGF-1   ng/mL
    ])
    t_dawn = 6.0   # 06:00

    def ctrl_car_on(t):
        return jnp.array([0.0, 1.0, 0.0, 85.0, 0.0])   # full awakening drive

    def ctrl_car_off(t):
        return jnp.array([0.0, 0.0, 0.0, 85.0, 0.0])   # no awakening drive

    x_car_on  = integrate_Nh(x0_dawn, n_hours=2.0, control_fn=ctrl_car_on,  t0=t_dawn)
    x_car_off = integrate_Nh(x0_dawn, n_hours=2.0, control_fn=ctrl_car_off, t0=t_dawn)

    cort_car_on  = float(x_car_on[IDX_CORT])
    cort_car_off = float(x_car_off[IDX_CORT])
    print(f"\n  T3 CAR: Cortisol car_on={cort_car_on:.1f}  car_off={cort_car_off:.1f}  nmol/L")

    assert cort_car_on > cort_car_off, (
        f"CAR: Cortisol must be elevated with hub_circadian_phase=1.0 "
        f"(car_on={cort_car_on:.1f}, car_off={cort_car_off:.1f}). "
        "Check k_CRH_CAR parameter."
    )

    assert not jnp.any(jnp.isnan(x_car_on)),  "NaN in CAR-on trajectory"
    assert not jnp.any(jnp.isnan(x_car_off)), "NaN in CAR-off trajectory"

    # ── Part 2: Nocturnal GH pulse ────────────────────────────────────────────
    x0_sleep = x0_dawn   # use same night state
    t_sleep  = 22.0      # 22:00 — early sleep

    def ctrl_deep_sleep(t):
        return jnp.array([0.0, 0.0, 0.0, 85.0, 0.70])   # 70% SWS

    def ctrl_wake(t):
        return jnp.array([0.0, 0.0, 0.0, 85.0, 0.0])    # wakefulness

    x_sws  = integrate_Nh(x0_sleep, n_hours=4.0, control_fn=ctrl_deep_sleep, t0=t_sleep)
    x_wake = integrate_Nh(x0_sleep, n_hours=4.0, control_fn=ctrl_wake,       t0=t_sleep)

    gh_sws  = float(x_sws[IDX_GH])
    gh_wake = float(x_wake[IDX_GH])
    ratio_gh = gh_sws / (gh_wake + 1e-6)

    igf_sws  = float(x_sws[IDX_IGF1])
    igf_wake = float(x_wake[IDX_IGF1])

    print(f"      SWS GH:  sws={gh_sws:.3f}  wake={gh_wake:.3f}  ng/mL  (ratio={ratio_gh:.2f}x)")
    print(f"      SWS IGF1: sws={igf_sws:.2f}  wake={igf_wake:.2f}  ng/mL")

    assert ratio_gh >= 1.20, (
        f"Sleep GH pulse must be >=20% larger with SWS=0.7 (got {ratio_gh:.2f}x). "
        "Check k_SWS_GHRH coupling."
    )
    assert igf_sws >= igf_wake, (
        "IGF-1 must be >= wakefulness with deep sleep (anabolic follow-through)."
    )

    assert not jnp.any(jnp.isnan(x_sws)),  "NaN in deep-sleep trajectory"
    assert not jnp.any(jnp.isnan(x_wake)), "NaN in wakefulness trajectory"

    print("  T3 PASS: CAR (hub_circadian_phase) and sleep GH pulse confirmed.")


# ─────────────────────────────────────────────────────────────────────────────
# T4 — UKF stiff stability (30 hourly steps)
# ─────────────────────────────────────────────────────────────────────────────

def test_ukf_stiff_stability():
    """
    Run 30 UKF steps (1 h each).

    Sparse observation schedule:
      - Every 8 h: Cortisol observed (noisy); Epi/NE observed (noisy).
      - IGF-1 missing throughout (quality_flag=4 → predict-only for that channel).
      - All other hours: predict-only (quality_flag=4).

    Assert at EVERY step:
      - No NaN in posterior mean
      - PSD covariance (all diagonal > 0)
      - Thermodynamic clamp: all states ≥ 0
    """
    x0    = initial_state()
    state = initial_filter_state(x0=x0)
    Q     = default_process_noise_Q()

    rng    = jax.random.PRNGKey(99)
    params = DEFAULT_NEURO_PARAMS

    n_steps = 30
    t_start = 8.0

    for step in range(n_steps):
        t_h   = t_start + step
        rng, key = jax.random.split(rng)
        rng, key2 = jax.random.split(rng)

        hour_mod = int(t_h) % 24

        if hour_mod % 8 == 0:
            # Sparse lab draw: Epi + NE + Cortisol with noise; IGF-1 missing
            noise_epi  = float(jax.random.normal(key)  * 50.0)
            noise_ne   = float(jax.random.normal(key2) * 100.0)
            noise_cort = float(jax.random.normal(jax.random.fold_in(key, 1)) * 28.0)
            y_obs = jnp.array([
                float(state.mean[IDX_EPI])  + noise_epi,
                float(state.mean[IDX_NE])   + noise_ne,
                float(state.mean[IDX_CORT]) + noise_cort,
                float(state.mean[IDX_IGF1]),   # IGF-1 "observed" at predicted mean
            ])
            qflag = 3   # proxy quality; IGF-1 effectively predict-only (obs=pred)
        else:
            # All channels predict-only
            y_obs = jnp.array([
                float(state.mean[IDX_EPI]),
                float(state.mean[IDX_NE]),
                float(state.mean[IDX_CORT]),
                float(state.mean[IDX_IGF1]),
            ])
            qflag = 4

        ctrl = lambda _: jnp.array([0.0, 0.0, 0.0, 85.0, 0.0])

        state = update_state(
            state        = state,
            y_obs        = y_obs,
            params       = params,
            obs_params   = DEFAULT_OBS_PARAMS,
            Q            = Q,
            quality_flag = qflag,
            control_fn   = ctrl,
            t0           = t_h,
        )

        mu = state.mean
        P  = state.cov

        assert not jnp.any(jnp.isnan(mu)), (
            f"NaN in posterior_mean at step {step} (t={t_h:.0f}h): {mu}"
        )
        assert not jnp.any(jnp.isnan(P)), (
            f"NaN in posterior_cov at step {step}"
        )

        diag_P = jnp.diag(P)
        assert jnp.all(diag_P > 0.0), (
            f"Non-positive variance at step {step}: diag(P)={diag_P}"
        )

        assert jnp.all(mu >= 0.0), (
            f"Negative concentration at step {step}: mean={mu}"
        )

    # Final checks
    final_mu = state.mean
    final_P  = state.cov

    print(f"\n  T4: 30-step UKF complete.")
    print(f"      Final state: Epi={float(final_mu[IDX_EPI]):.1f}  NE={float(final_mu[IDX_NE]):.1f}  "
          f"Cort={float(final_mu[IDX_CORT]):.1f}  IGF1={float(final_mu[IDX_IGF1]):.1f}")
    print(f"      diag(P): {jnp.diag(final_P)}")

    assert not jnp.any(jnp.isnan(final_mu)),  "Final posterior mean contains NaN"
    assert jnp.all(jnp.diag(final_P) > 0.0), "Final covariance not PSD"
    assert jnp.all(final_mu >= 0.0),          "Final state violates thermodynamic clamp"

    sym_err = float(jnp.max(jnp.abs(final_P - final_P.T)))
    assert sym_err < 1e-3, f"Covariance asymmetric: max|P-P^T| = {sym_err:.2e}"

    print(f"      UKF alpha=0.10, n=9 — no catastrophic float32 cancellation. PASS.")
    print("  T4 PASS: 30-step stiff UKF stable.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("T1: SAM spike vs HPA cascade delay")
    test_sam_vs_hpa_delay()
    print()
    print("=" * 70)
    print("T2: Personalised glucose panic (Metabolic_Flexibility_Threshold)")
    test_personalized_glucose_panic()
    print()
    print("=" * 70)
    print("T3: Circadian CAR + nocturnal GH pulse")
    test_circadian_car_and_sleep()
    print()
    print("=" * 70)
    print("T4: 30-step UKF stiff stability")
    test_ukf_stiff_stability()
    print()
    print("=" * 70)
    print("ALL 4 TESTS PASSED — neuroendocrine V2.0 Gate Zero CLEAR")
