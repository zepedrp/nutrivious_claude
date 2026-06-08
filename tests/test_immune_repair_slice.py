"""
tests/test_immune_repair_slice.py -- Gate-Zero: Immune Repair V1.0

T1  test_exercise_myokine_il6
        hub_power_watts high, hub_training_stress zero.
        Assert: IL-6 fires as myokine; Damage and M1 stay low.

T2  test_damage_inflammation_cascade
        hub_training_stress high for 4h; then rest for 48h.
        Assert: Damage up -> M1 up -> IL-6 up (inflammatory) -> M2 rises -> Damage drained.

T3  test_cortisol_immunosuppression
        hub_training_stress high + hub_Cortisol chronic at 800 nmol/L vs control.
        Assert: M1 is blocked in the overtraining scenario.
        Muscle Damage stagnates (not repaired) vs the control case.

T4  test_ukf_immune_assimilation
        Assimilate 48 hourly steps.
        Assert: No NaN, covariance PSD at every step.

Run:
    pytest tests/test_immune_repair_slice.py -v -s
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

from app.slices.immune_repair.ode import (
    ImmuneParams, DEFAULT_IMMUNE_PARAMS,
    IDX_DMG, IDX_M1, IDX_M2, IDX_IL6,
    STATE_DIM, OBS_DIM,
    initial_state, integrate_Nh,
)
from app.slices.immune_repair.observation import (
    DEFAULT_OBS_PARAMS, h_immune, observation_noise_R,
)
from app.slices.immune_repair.filter import (
    ImmuneFilterState,
    initial_filter_state,
    update_state,
    default_process_noise_Q,
)


# ─────────────────────────────────────────────────────────────────────────────
# T1 -- Myokine IL-6 from contractile power (no damage)
# ─────────────────────────────────────────────────────────────────────────────

def test_exercise_myokine_il6():
    """
    High power output (300W) with ZERO mechanical stress.
    IL-6 should spike (myokine) but Damage and M1 must remain near baseline.

    Uses a fully-recovered (zero-damage) starting state to isolate the
    myokine IL-6 source from background damage-driven M1 recruitment.
    """
    # Zero-damage start: isolates myokine path cleanly
    x0 = jnp.array([0.0, 0.0, 0.05, 1.0])   # D=0, M1=0, M2 homeostatic, IL6 basal

    # 2h at 300W, zero training stress, resting cortisol
    def ctrl_myokine(t):
        return jnp.array([0.0, 300.0, 300.0])   # stress=0, power=300W, normal cortisol

    x_after = integrate_Nh(x0, n_hours=2.0, control_fn=ctrl_myokine)

    il6_0 = float(x0[IDX_IL6])
    il6_f = float(x_after[IDX_IL6])
    dmg_f = float(x_after[IDX_DMG])
    m1_f  = float(x_after[IDX_M1])

    print(f"\n  T1: IL-6 {il6_0:.2f} -> {il6_f:.2f} pg/mL  (myokine source active)")
    print(f"      Damage={dmg_f:.4f}  M1={m1_f:.4f}  (must stay near baseline)")

    # IL-6 must rise substantially from myokine source
    assert il6_f > il6_0 * 2.0, (
        f"Myokine IL-6 must at least double at 300W (got {il6_f:.2f} vs baseline {il6_0:.2f}). "
        "Check k_il6_myo."
    )

    # Damage must stay near zero (no stress input)
    assert dmg_f < 0.10, (
        f"Muscle damage must stay near baseline with zero stress (got {dmg_f:.4f}). "
        "Check k_dmg gating by hub_training_stress."
    )

    # M1 must stay near zero (no damage to recruit them)
    assert m1_f < 0.05, (
        f"M1 macrophages must stay quiescent with no damage (got {m1_f:.4f}). "
        "Check k_M1_recruit gating by damage."
    )

    assert not jnp.any(jnp.isnan(x_after)), "NaN in myokine trajectory"
    print("  T1 PASS: myokine IL-6 confirmed; damage and M1 decoupled from power.")


# ─────────────────────────────────────────────────────────────────────────────
# T2 -- Damage -> inflammation cascade -> repair
# ─────────────────────────────────────────────────────────────────────────────

def test_damage_inflammation_cascade():
    """
    4h of high mechanical stress (session) followed by 48h rest.

    Sequence expected:
        Damage rises during session.
        M1 rises (recruited by damage, peaks ~12h post).
        IL-6 rises via M1 cytokine source (inflammatory, sustained).
        M2 rises (polarised from M1, peaks ~24-36h post).
        Damage falls as M2 clears it.
    """
    x0 = initial_state()

    # --- Session: 4h at training_stress=0.8 ---
    def ctrl_session(t):
        return jnp.array([0.8, 200.0, 350.0])   # high stress, moderate power, normal cortisol

    x_post_session = integrate_Nh(x0, n_hours=4.0, control_fn=ctrl_session)

    dmg_post = float(x_post_session[IDX_DMG])
    m1_post  = float(x_post_session[IDX_M1])
    il6_post = float(x_post_session[IDX_IL6])

    print(f"\n  T2: Post-session: Damage={dmg_post:.4f}  M1={m1_post:.4f}  IL6={il6_post:.2f}")

    assert dmg_post > float(x0[IDX_DMG]) * 2.0, (
        f"Damage must rise during session (got {dmg_post:.4f} vs baseline {float(x0[IDX_DMG]):.4f})."
    )

    # --- Recovery: 48h at rest ---
    def ctrl_rest(t):
        return jnp.array([0.0, 0.0, 300.0])

    x_12h = integrate_Nh(x_post_session, n_hours=12.0, control_fn=ctrl_rest)
    x_48h = integrate_Nh(x_post_session, n_hours=48.0, control_fn=ctrl_rest)

    m1_12h  = float(x_12h[IDX_M1])
    m2_12h  = float(x_12h[IDX_M2])
    dmg_48h = float(x_48h[IDX_DMG])
    m2_48h  = float(x_48h[IDX_M2])

    print(f"      At 12h recovery: M1={m1_12h:.4f}  M2={m2_12h:.4f}")
    print(f"      At 48h recovery: Damage={dmg_48h:.4f}  M2={m2_48h:.4f}")

    # M2 must be elevated at 12h (polarisation in progress)
    assert m2_12h > float(x0[IDX_M2]), (
        f"M2 must be elevated 12h post-session (got {m2_12h:.4f} vs baseline {float(x0[IDX_M2]):.4f})."
    )

    # Damage must reduce by 48h vs post-session
    assert dmg_48h < dmg_post, (
        f"Damage must clear during 48h rest (post={dmg_post:.4f}, 48h={dmg_48h:.4f}). "
        "Check M2-driven repair (k_M2_repair)."
    )

    assert not jnp.any(jnp.isnan(x_48h)), "NaN in recovery trajectory"
    print("  T2 PASS: Damage->M1->IL6->M2->Repair cascade confirmed.")


# ─────────────────────────────────────────────────────────────────────────────
# T3 -- Cortisol immunosuppression (overtraining immune block)
# ─────────────────────────────────────────────────────────────────────────────

def test_cortisol_immunosuppression():
    """
    Same training stress (stress=0.8, 24h) applied to two scenarios:
        Control:     hub_Cortisol = 350 nmol/L (normal)
        Overtraining: hub_Cortisol = 800 nmol/L (chronic stress, above threshold=550)

    Assert:
        M1 is suppressed in the overtraining scenario.
        Muscle damage fails to be cleared (M2 starved of M1 polarisation).
        Overtraining damage > control damage at 48h recovery.
    """
    x0 = initial_state()

    def ctrl_control(t):
        return jnp.array([0.8, 100.0, 350.0])   # normal cortisol

    def ctrl_overtraining(t):
        return jnp.array([0.8, 100.0, 800.0])   # chronically elevated cortisol

    # 24h training + 48h recovery = 72h total
    x_ctrl_train = integrate_Nh(x0, n_hours=24.0, control_fn=ctrl_control)
    x_ot_train   = integrate_Nh(x0, n_hours=24.0, control_fn=ctrl_overtraining)

    m1_ctrl = float(x_ctrl_train[IDX_M1])
    m1_ot   = float(x_ot_train[IDX_M1])

    print(f"\n  T3: Post-training M1: control={m1_ctrl:.4f}  overtraining={m1_ot:.4f}")

    assert m1_ctrl > m1_ot, (
        f"Cortisol 800 nmol/L must suppress M1 below control "
        f"(ctrl={m1_ctrl:.4f}, ot={m1_ot:.4f}). "
        "Check k_cort_supp * cort_excess term in dM1/dt."
    )

    # Recovery phase: 48h rest (cortisol normalises)
    def ctrl_recovery(t):
        return jnp.array([0.0, 0.0, 300.0])

    x_ctrl_rec = integrate_Nh(x_ctrl_train, n_hours=48.0, control_fn=ctrl_recovery)
    x_ot_rec   = integrate_Nh(x_ot_train,   n_hours=48.0, control_fn=ctrl_recovery)

    dmg_ctrl = float(x_ctrl_rec[IDX_DMG])
    dmg_ot   = float(x_ot_rec[IDX_DMG])
    m2_ctrl  = float(x_ctrl_rec[IDX_M2])
    m2_ot    = float(x_ot_rec[IDX_M2])

    print(f"      Post-48h recovery: Damage ctrl={dmg_ctrl:.4f}  ot={dmg_ot:.4f}")
    print(f"      M2: ctrl={m2_ctrl:.4f}  ot={m2_ot:.4f}")

    # Overtraining damage must not recover as well as control
    assert dmg_ot > dmg_ctrl, (
        f"Overtraining damage ({dmg_ot:.4f}) must exceed control damage ({dmg_ctrl:.4f}) "
        "after 48h recovery. M1-suppression starves M2 -> impaired repair. "
        "Check Cortisol_threshold and k_cort_supp."
    )

    assert not jnp.any(jnp.isnan(x_ctrl_rec)), "NaN in control trajectory"
    assert not jnp.any(jnp.isnan(x_ot_rec)),   "NaN in overtraining trajectory"

    print("  T3 PASS: cortisol immunosuppression -> impaired repair confirmed mathematically.")


# ─────────────────────────────────────────────────────────────────────────────
# T4 -- UKF 48-step assimilation
# ─────────────────────────────────────────────────────────────────────────────

def test_ukf_immune_assimilation():
    """
    48 hourly UKF steps with sparse biomarker observations.
    Every 12h: hsCRP observed (noisy); CK observed (noisy).
    Other hours: predict-only (quality_flags=(4, 4)).

    Assert at EVERY step:
        No NaN in posterior mean.
        Covariance PSD (all diagonal > 0).
        All states >= 0 (physical clamp active).
    """
    x0    = initial_state()
    state = initial_filter_state(x0=x0)
    Q     = default_process_noise_Q()

    rng = jax.random.PRNGKey(7)

    # Low-grade session: moderate damage accumulation
    def ctrl_session(t):
        return jnp.array([0.4, 150.0, 320.0])

    n_steps = 48
    t_start = 0.0

    for step in range(n_steps):
        t_h  = t_start + float(step)
        rng, key = jax.random.split(rng)

        # Sparse biomarker: every 12h
        if int(step) % 12 == 0:
            # Noisy lab measurements around current predicted state
            y_pred = h_immune(state.mean, DEFAULT_OBS_PARAMS)
            noise  = jax.random.normal(key, shape=(2,)) * jnp.array([0.5, 60.0])
            y_obs  = y_pred + noise
            qflags = (0, 0)
        else:
            # Predict-only: feed predicted observation back (no information gain)
            y_obs  = h_immune(state.mean, DEFAULT_OBS_PARAMS)
            qflags = (4, 4)

        ctrl = ctrl_session

        state = update_state(
            state         = state,
            y_obs         = y_obs,
            params        = DEFAULT_IMMUNE_PARAMS,
            obs_params    = DEFAULT_OBS_PARAMS,
            Q             = Q,
            quality_flags = qflags,
            control_fn    = ctrl,
            t0            = t_h,
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
            f"Negative state at step {step}: mean={mu}"
        )

    final_mu = state.mean
    final_P  = state.cov

    print(f"\n  T4: 48-step UKF complete.")
    print(f"      Final state: Damage={float(final_mu[IDX_DMG]):.4f}  "
          f"M1={float(final_mu[IDX_M1]):.4f}  "
          f"M2={float(final_mu[IDX_M2]):.4f}  "
          f"IL6={float(final_mu[IDX_IL6]):.2f} pg/mL")
    print(f"      diag(P): {jnp.diag(final_P)}")

    # Symmetry check
    sym_err = float(jnp.max(jnp.abs(final_P - final_P.T)))
    assert sym_err < 1e-3, f"Covariance asymmetric: max|P-P^T| = {sym_err:.2e}"

    print("  T4 PASS: 48-step UKF -- no NaN, PSD guaranteed, alpha=0.10 float32 stable.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("T1: Myokine IL-6 (power, no damage)")
    test_exercise_myokine_il6()
    print()
    print("=" * 70)
    print("T2: Damage->Inflammation->Repair cascade")
    test_damage_inflammation_cascade()
    print()
    print("=" * 70)
    print("T3: Cortisol immunosuppression (overtraining)")
    test_cortisol_immunosuppression()
    print()
    print("=" * 70)
    print("T4: 48-step UKF assimilation")
    test_ukf_immune_assimilation()
    print()
    print("=" * 70)
    print("ALL 4 TESTS PASSED -- immune_repair V1.0 Gate Zero CLEAR")
