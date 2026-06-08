"""
tests/test_hematological_slice.py  Gate Zero -- Hematological Slice V3.0

T1  test_sports_anemia_il6_block
    High iron + altitude EPO, but IL6=1 (healthy) vs IL6=50 (inflamed).
    Inflamed: Ferritin does NOT rise (hepcidin blocks iron absorption).
    Inflamed: Erythropoiesis collapses (iron-limited marrow).

T2  test_footstrike_hemolysis
    High constant load (runner) vs zero load (swimmer).
    Runner: RBC_Mass much lower, Hemolysis_Tox spikes.

T3  test_plasma_volume_expansion
    Continuous load -> Plasma_Vol expands -> Hematocrit FALLS
    (pseudo-anemia by dilution; RBC constant to isolate the effect).

T4  test_ukf_hematological_assimilation
    48 UKF steps; blood tests only at hour 24.
    Posterior is NaN-free; covariance is PSD throughout.
"""
import jax
import jax.numpy as jnp
import pytest

from app.slices.hematological.ode import (
    IDX_RBC_MASS, IDX_PLASMA_VOL, IDX_EPO,
    IDX_FERRITIN, IDX_HEM_TOX,
    STATE_DIM, OBS_DIM,
    HematologicalParams, DEFAULT_HEM_PARAMS,
    X0_HEM_DEFAULT,
    compute_hematocrit, integrate_1h,
)
from app.slices.hematological.observation import (
    HemObsParams, DEFAULT_HEM_OBS_PARAMS, h_hem,
)
from app.slices.hematological.filter import (
    HemFilterState, initial_filter_state, update_state,
)

jax.config.update("jax_enable_x64", False)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _simulate(x0, u, n_hours=200, params=DEFAULT_HEM_PARAMS):
    """Run n_hours of constant-control ODE integration."""
    x = x0
    xs = [x0]
    for _ in range(n_hours):
        x = integrate_1h(x, params=params, u=u, t0=0.0)
        xs.append(x)
    return jnp.stack(xs)   # (n_hours+1, STATE_DIM)


def _erythropoiesis(x, p=DEFAULT_HEM_PARAMS):
    EPO  = jnp.maximum(x[IDX_EPO],      0.0)
    Ferr = jnp.maximum(x[IDX_FERRITIN], 0.0)
    return p.k_epo_rbc * EPO * (Ferr / (p.Km_Fe + Ferr))


# ── T1: IL-6 / Hepcidin Iron Block ────────────────────────────────────────────

def test_sports_anemia_il6_block():
    """
    Altitude camp: high EPO drive (Hypoxia=20%) + high iron intake (18 mg/h).
    Healthy (IL6=1.0):   iron absorbed normally -> Ferritin stable/rising.
    Inflamed (IL6=50.0): hepcidin proxy blocks absorption -> Ferritin collapses,
                         erythropoiesis collapses (iron-limited).
    """
    u_healthy   = jnp.array([20.0, 0.0, 1.0,  18.0], dtype=jnp.float32)
    u_inflamed  = jnp.array([20.0, 0.0, 50.0, 18.0], dtype=jnp.float32)

    x0 = X0_HEM_DEFAULT   # Ferritin=80 ug/L at start

    traj_h = _simulate(x0, u_healthy,  n_hours=200)
    traj_i = _simulate(x0, u_inflamed, n_hours=200)

    ferr_healthy  = float(traj_h[-1, IDX_FERRITIN])
    ferr_inflamed = float(traj_i[-1, IDX_FERRITIN])
    ery_healthy   = float(_erythropoiesis(traj_h[-1]))
    ery_inflamed  = float(_erythropoiesis(traj_i[-1]))

    print(f"Ferritin: healthy={ferr_healthy:.1f}  inflamed={ferr_inflamed:.1f} ug/L")
    print(f"Erythropoiesis: healthy={ery_healthy:.4f}  inflamed={ery_inflamed:.4f} g/h")

    # Inflamed Ferritin must be substantially lower than healthy
    assert ferr_inflamed < ferr_healthy * 0.5, (
        f"Expected inflamed Ferritin << healthy, got "
        f"{ferr_inflamed:.1f} vs {ferr_healthy:.1f}"
    )

    # Erythropoiesis must collapse in inflamed scenario
    assert ery_inflamed < ery_healthy * 0.5, (
        f"Expected erythropoiesis collapse in IL6=50, got "
        f"{ery_inflamed:.4f} vs healthy {ery_healthy:.4f}"
    )


# ── T2: Footstrike Hemolysis ───────────────────────────────────────────────────

def test_footstrike_hemolysis():
    """
    Runner (Load=1.0) vs swimmer (Load=0.0).
    Runner: higher hemolysis rate depletes RBC_Mass and accumulates Hemolysis_Tox.
    """
    u_runner  = jnp.array([0.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    u_swimmer = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)

    x0 = X0_HEM_DEFAULT

    traj_r = _simulate(x0, u_runner,  n_hours=200)
    traj_s = _simulate(x0, u_swimmer, n_hours=200)

    rbc_runner   = float(traj_r[-1, IDX_RBC_MASS])
    rbc_swimmer  = float(traj_s[-1, IDX_RBC_MASS])
    tox_runner   = float(traj_r[-1, IDX_HEM_TOX])
    tox_swimmer  = float(traj_s[-1, IDX_HEM_TOX])

    print(f"RBC_Mass: runner={rbc_runner:.0f}g  swimmer={rbc_swimmer:.0f}g")
    print(f"HemTox:   runner={tox_runner:.4f}  swimmer={tox_swimmer:.4f}")

    # Runner must have substantially lower RBC mass
    assert rbc_runner < rbc_swimmer * 0.85, (
        f"Expected runner RBC << swimmer, got {rbc_runner:.0f} vs {rbc_swimmer:.0f}"
    )

    # Runner hemolysis toxin must be much higher
    assert tox_runner > tox_swimmer * 3.0, (
        f"Expected runner HemTox >> swimmer, got {tox_runner:.4f} vs {tox_swimmer:.4f}"
    )


# ── T3: Plasma Volume Expansion (pseudo-anemia) ────────────────────────────────

def test_plasma_volume_expansion():
    """
    Continuous load (Load=0.8) drives plasma volume expansion.
    Hematocrit FALLS even though RBC_Mass changes slower -- pseudo-anemia.
    """
    u_load = jnp.array([0.0, 0.8, 1.0, 0.0], dtype=jnp.float32)

    x0 = X0_HEM_DEFAULT

    traj = _simulate(x0, u_load, n_hours=120)

    pv_initial = float(x0[IDX_PLASMA_VOL])
    pv_final   = float(traj[-1, IDX_PLASMA_VOL])
    hct_initial = float(compute_hematocrit(x0))
    hct_final   = float(compute_hematocrit(traj[-1]))

    print(f"Plasma Vol: initial={pv_initial:.3f}L  final={pv_final:.3f}L")
    print(f"Hematocrit: initial={hct_initial:.2f}%  final={hct_final:.2f}%")

    # Plasma volume must expand with continuous load
    assert pv_final > pv_initial + 0.05, (
        f"Expected PV to expand by >0.05L, got delta={pv_final-pv_initial:.3f}L"
    )

    # Hematocrit must fall (dilution effect)
    assert hct_final < hct_initial - 0.5, (
        f"Expected Hct to drop by >0.5%, got delta={hct_final-hct_initial:.2f}%"
    )


# ── T4: UKF Assimilation (sparse blood panel) ─────────────────────────────────

def test_ukf_hematological_assimilation():
    """
    48-step UKF (hourly); blood tests arrive only at hour 24.
    Steps 0-23 and 25-47: all flags=4 (predict-only).
    Step 24: all flags=0 (full assimilation).
    Asserts: no NaN in any posterior mean, covariance PSD at end.
    """
    # Constant control: slight hypoxia + normal iron
    u = jnp.array([5.0, 0.0, 1.0, 0.75], dtype=jnp.float32)

    state = initial_filter_state()

    # Synthetic observation from default state (reasonable lab values)
    y_obs = h_hem(X0_HEM_DEFAULT)

    all_means = []
    for step in range(48):
        flags = (0, 0, 0) if step == 24 else (4, 4, 4)
        state = update_state(
            state         = state,
            y_obs         = y_obs,
            quality_flags = flags,
            u             = u,
        )
        all_means.append(state.mean)

    means_stack = jnp.stack(all_means)   # (48, STATE_DIM)

    # No NaN in any posterior mean
    assert not jnp.any(jnp.isnan(means_stack)), (
        f"NaN detected in posterior means at steps: "
        f"{jnp.where(jnp.any(jnp.isnan(means_stack), axis=1))[0].tolist()}"
    )

    # Final covariance PSD (all diagonal elements >= 0)
    diag_P = jnp.diag(state.cov)
    assert jnp.all(diag_P >= 0.0), (
        f"Non-PSD covariance diagonal at end: {diag_P.tolist()}"
    )

    print(f"UKF 48-step complete. Final state: {[f'{v:.2f}' for v in state.mean.tolist()]}")
    print(f"Final cov diagonal: {[f'{v:.4f}' for v in diag_P.tolist()]}")
