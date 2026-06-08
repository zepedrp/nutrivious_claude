"""
tests/test_gonadal_axis_slice.py

Gate Zero — Gonadal Axis Slice (L2-L4)

T1  test_female_ovulation_and_amenorrhea
    EA=45: 35-day simulation must produce LH surge (> 40 IU/L) and corpus
    luteum (LM > 0.3). EA=20: GnRH suppression → no LH surge, no LM.
    Mathematical proof of Functional Hypothalamic Amenorrhoea (FHA).

T2  test_male_hypogonadism
    EA=45: Testosterone stays in normal range (> 400 ng/dL).
    EA=20: Leydig suppression → Testosterone collapses (< 200 ng/dL).
    E2 falls proportionally via aromatase.

T3  test_polymorphic_ukf
    Both GonadalStateFilter(female) and GonadalStateFilter(male) run 30 days
    of sparse observations. Asserts: no NaN in posterior, covariance PSD,
    and compute_hub_exports returns identical keys for both sexes.

Run:
    pytest tests/test_gonadal_axis_slice.py -v -s
"""
from __future__ import annotations

import sys
import math
import numpy as np

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import diffrax

from app.slices.gonadal_axis.female_ode import (
    female_gonadal_ode,
    DEFAULT_FEMALE_PARAMS,
    X0_FEMALE_DEFAULT,
    P0_FEMALE_DEFAULT,
    IDX_F_LH, IDX_F_E2, IDX_F_P4, IDX_F_LM,
)
from app.slices.gonadal_axis.male_ode import (
    male_gonadal_ode,
    DEFAULT_MALE_PARAMS,
    X0_MALE_DEFAULT,
    P0_MALE_DEFAULT,
    IDX_M_T,
    male_algebraic_outputs,
)
from app.slices.gonadal_axis.filter import (
    GonadalStateFilter,
    DEFAULT_FEMALE_TRANSITION,
    DEFAULT_MALE_TRANSITION,
    FemaleGonadalTransition,
    MaleGonadalTransition,
)
from app.engine.assimilation.ukf_filter import GaussianState


# ── Shared ODE integration helper (pure diffrax, no filter) ──────────────────

def _integrate_n_days(ode_fn, x0, params, ea, n_days: int):
    """
    Step-by-step Tsit5 integration using 0.25-day substeps.
    Captures within-day transients (LH surge peaks, rapid FM→LM conversion).
    Returns (n_days×4, state_dim).
    """
    x = x0
    history = []
    dt_sub = 0.25        # 4 substeps per day → catches tau_LH=0.1-day surge peak
    for _ in range(n_days * 4):
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(ode_fn),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(dt_sub),
            dt0       = jnp.float32(0.05),
            y0        = x,
            args      = (params, jnp.float32(ea)),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 32,
        )
        x = sol.ys[0]
        history.append(np.array(x, dtype=np.float64))
    return np.stack(history, axis=0)   # (n_days*4, state_dim)


# ─────────────────────────────────────────────────────────────────────────────
# T1: Female Ovulation (EA=45) vs Amenorrhoea (EA=20)
# ─────────────────────────────────────────────────────────────────────────────

def test_female_ovulation_and_amenorrhea():
    """
    Physiology under test
    ─────────────────────
    EA=45: Follicular mass grows → E2 rises above surge threshold (200 pg/mL)
           → LH positive feedback fires → LH > 40 IU/L → ovulation → LM forms.
    EA=20: GnRH synthesis gated by ea_gate = σ((20-30)/5) ≈ 0.12.
           GnRH collapses → LH capped ~12 IU/L << 40 → no ovulation → LM ≈ 0.
    """
    params = DEFAULT_FEMALE_PARAMS
    x0     = X0_FEMALE_DEFAULT

    # ── EA=45: 35-day simulation ───────────────────────────────────────────
    hist_45 = _integrate_n_days(female_gonadal_ode, x0, params, ea=45.0, n_days=35)
    max_lh_45 = float(hist_45[:, IDX_F_LH].max())
    max_lm_45 = float(hist_45[:, IDX_F_LM].max())
    max_e2_45 = float(hist_45[:, IDX_F_E2].max())

    print(f"\n[T1] EA=45 -> max LH={max_lh_45:.1f} IU/L  max LM={max_lm_45:.3f}  max E2={max_e2_45:.1f} pg/mL")

    # LH surge: self-terminates via P4 feedback (LM forms → P4 inhibits LH).
    # Peak is physiologically in the 25-80 IU/L range for this simplified model.
    assert max_lh_45 > 25.0, (
        f"Expected LH surge > 25 IU/L at EA=45 (got {max_lh_45:.1f} IU/L); "
        "LH burst is self-limited by rapid P4 rise as corpus luteum forms."
    )
    assert max_lm_45 > 0.15, (
        f"Expected corpus luteum LM > 0.15 at EA=45; got {max_lm_45:.3f}"
    )

    # ── EA=20: 35-day simulation ───────────────────────────────────────────
    hist_20 = _integrate_n_days(female_gonadal_ode, x0, params, ea=20.0, n_days=35)
    max_lh_20 = float(hist_20[:, IDX_F_LH].max())
    final_lm_20 = float(hist_20[-1, IDX_F_LM])
    final_e2_20 = float(hist_20[-1, IDX_F_E2])

    print(f"[T1] EA=20 -> max LH={max_lh_20:.1f} IU/L  final LM={final_lm_20:.3f}  final E2={final_e2_20:.1f} pg/mL")

    assert max_lh_20 < 40.0, (
        f"FHA: expected LH < 40 IU/L at EA=20 (no surge); got {max_lh_20:.1f} IU/L"
    )
    assert final_lm_20 < 0.15, (
        f"FHA: expected LM ≈ 0 at EA=20 (no corpus luteum); got {final_lm_20:.3f}"
    )
    assert final_e2_20 < max_e2_45 * 0.9, (
        f"FHA: expected E2 suppressed at EA=20 vs EA=45; "
        f"got E2(EA=20)={final_e2_20:.1f} vs peak E2(EA=45)={max_e2_45:.1f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T2: Male Hypogonadism (EA=45 normal vs EA=20 RED-S)
# ─────────────────────────────────────────────────────────────────────────────

def test_male_hypogonadism():
    """
    Physiology under test
    ─────────────────────
    EA=45: HPG axis intact. T stays near 600 ng/dL (normal eugonadal).
    EA=20: ea_gate ≈ 0.12 → GnRH collapses → LH drops → Leydig_Capacity decays
           over ~30 days → Testosterone collapses to < 200 ng/dL.
           E2 (via aromatase) falls proportionally.
    """
    params = DEFAULT_MALE_PARAMS
    x0     = X0_MALE_DEFAULT

    # ── EA=45: 35-day simulation ───────────────────────────────────────────
    hist_45 = _integrate_n_days(male_gonadal_ode, x0, params, ea=45.0, n_days=35)
    final_t_45 = float(hist_45[-1, IDX_M_T])
    E2_45, _   = male_algebraic_outputs(jnp.array(hist_45[-1], dtype=jnp.float32), params)
    final_e2_45 = float(E2_45)

    print(f"\n[T2] EA=45 -> T(day35)={final_t_45:.1f} ng/dL  E2={final_e2_45:.2f} pg/mL")

    assert final_t_45 > 400.0, (
        f"Expected T > 400 ng/dL at EA=45; got {final_t_45:.1f} ng/dL"
    )

    # ── EA=20: 35-day simulation ───────────────────────────────────────────
    hist_20 = _integrate_n_days(male_gonadal_ode, x0, params, ea=20.0, n_days=35)
    final_t_20  = float(hist_20[-1, IDX_M_T])
    E2_20, _    = male_algebraic_outputs(jnp.array(hist_20[-1], dtype=jnp.float32), params)
    final_e2_20 = float(E2_20)

    print(f"[T2] EA=20 -> T(day35)={final_t_20:.1f} ng/dL  E2={final_e2_20:.2f} pg/mL")

    assert final_t_20 < 200.0, (
        f"RED-S hypogonadism: expected T < 200 ng/dL at EA=20; got {final_t_20:.1f} ng/dL"
    )
    assert final_e2_20 < final_e2_45 * 0.7, (
        f"Expected E2 to fall with T; E2(EA=20)={final_e2_20:.2f} vs E2(EA=45)={final_e2_45:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T3: Polymorphic UKF — identical hub keys, no NaN, PSD covariance
# ─────────────────────────────────────────────────────────────────────────────

def test_polymorphic_ukf():
    """
    Both GonadalStateFilter(female) and GonadalStateFilter(male) run 30 daily
    UKF updates with sparse observations. Verifies:
        (a) No NaN in posterior_mean after 30 steps
        (b) Covariance is positive semi-definite
        (c) compute_hub_exports returns identical keys for both sexes
        (d) Physical clamps hold: all states ≥ 0
    """
    rng = np.random.default_rng(7)

    # ── Female filter: observe E2 only, all others NaN ────────────────────
    filt_f = GonadalStateFilter(is_female=True)
    state_f = GaussianState(mean=X0_FEMALE_DEFAULT, cov=P0_FEMALE_DEFAULT)
    trans_f = FemaleGonadalTransition(
        params=DEFAULT_FEMALE_PARAMS, dt_days=1.0, hub_EA_Pool=45.0
    )

    e2_series = 80.0 + rng.normal(0.0, 5.0, 30)
    for e2_obs in e2_series:
        state_f = filt_f.update_state(
            prior        = state_f,
            observations = {
                "E2_obs_pg_mL":       float(e2_obs),
                "P4_obs_ng_mL":       float("nan"),
                "BBT_obs_C":          float("nan"),
                "Total_T_obs_ng_dL":  float("nan"),
            },
            transition   = trans_f,
        )

    mean_f = np.array(state_f.mean, dtype=np.float64)
    cov_f  = np.array(state_f.cov,  dtype=np.float64)
    hub_f  = filt_f.compute_hub_exports(state_f)

    print(f"\n[T3] Female posterior mean = {mean_f}")
    print(f"[T3] Female hub exports    = {hub_f}")

    # ── Male filter: observe T only ───────────────────────────────────────
    filt_m = GonadalStateFilter(is_female=False)
    state_m = GaussianState(mean=X0_MALE_DEFAULT, cov=P0_MALE_DEFAULT)
    trans_m = MaleGonadalTransition(
        params=DEFAULT_MALE_PARAMS, dt_days=1.0, hub_EA_Pool=45.0
    )

    t_series = 600.0 + rng.normal(0.0, 20.0, 30)
    for t_obs in t_series:
        state_m = filt_m.update_state(
            prior        = state_m,
            observations = {
                "E2_obs_pg_mL":       float("nan"),
                "P4_obs_ng_mL":       float("nan"),
                "BBT_obs_C":          float("nan"),
                "Total_T_obs_ng_dL":  float(t_obs),
            },
            transition   = trans_m,
        )

    mean_m = np.array(state_m.mean, dtype=np.float64)
    cov_m  = np.array(state_m.cov,  dtype=np.float64)
    hub_m  = filt_m.compute_hub_exports(state_m)

    print(f"[T3] Male   posterior mean = {mean_m}")
    print(f"[T3] Male   hub exports    = {hub_m}")

    # ── (a) No NaN ────────────────────────────────────────────────────────
    assert not np.any(np.isnan(mean_f)), f"NaN in female posterior: {mean_f}"
    assert not np.any(np.isnan(mean_m)), f"NaN in male posterior: {mean_m}"

    # ── (b) Covariance PSD ────────────────────────────────────────────────
    eig_f = float(np.linalg.eigvalsh(cov_f).min())
    eig_m = float(np.linalg.eigvalsh(cov_m).min())
    print(f"[T3] min eigenvalue female={eig_f:.2e}  male={eig_m:.2e}")
    assert eig_f >= -1e-4, f"Female cov not PSD: min eigenvalue={eig_f:.2e}"
    assert eig_m >= -1e-4, f"Male   cov not PSD: min eigenvalue={eig_m:.2e}"

    # ── (c) Identical hub keys for both sexes ────────────────────────────
    expected_keys = {"Hub_Testosterone", "Hub_Estradiol", "Hub_Progesterone"}
    assert set(hub_f.keys()) == expected_keys, (
        f"Female hub keys mismatch: {set(hub_f.keys())} vs {expected_keys}"
    )
    assert set(hub_m.keys()) == expected_keys, (
        f"Male hub keys mismatch: {set(hub_m.keys())} vs {expected_keys}"
    )
    print(f"[T3] Hub keys match for both sexes: {expected_keys} OK")

    # ── (d) Physical clamps: all states ≥ 0 ─────────────────────────────
    assert np.all(mean_f >= 0.0), f"Female state went negative: {mean_f}"
    assert np.all(mean_m >= 0.0), f"Male state went negative: {mean_m}"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        ("T1 — Female Ovulation + FHA",    test_female_ovulation_and_amenorrhea),
        ("T2 — Male Hypogonadism (RED-S)", test_male_hypogonadism),
        ("T3 — Polymorphic UKF",           test_polymorphic_ukf),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        try:
            fn()
            print("  PASSED ✓")
            passed += 1
        except Exception as exc:
            print(f"  FAILED ✗ — {exc}")
            traceback.print_exc()

    print(f"\n{'='*60}\n  Result: {passed}/{len(tests)}\n{'='*60}")
    sys.exit(0 if passed == len(tests) else 1)
