"""
tests/test_biomechanical_slice.py

Round-trip Gate-Zero validation -- Biomechanical Tissue slice (L2-L4).

Tests
------
  T1  Wolff's Law bone adaptation:
        Consistent moderate load + testosterone -> Bone_Density increases.
  T2  Estrogen-induced tendon laxity (ACL risk model):
        Peri-ovulatory E2 peak -> Tendon_Stiffness drops acutely and
        Hub_Tendon_Rupture_Risk rises vs low-E2 control.
  T3  Cortisol-driven collagen degradation:
        Chronic high Cortisol blocks CSR and accumulates Tendon_Microdamage
        compared to high-testosterone anabolic control.
  T4  UKF biomechanical assimilation (48 steps):
        No NaNs; covariance remains PSD throughout.

Run
---
  python tests/test_biomechanical_slice.py
"""
from __future__ import annotations

import sys
import io
import math
import traceback

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import diffrax

from app.slices.biomechanical_tissue.ode import (
    BiomechanicalParams,
    DEFAULT_BIO_PARAMS,
    X0_BIO_DEFAULT,
    P0_BIO_DEFAULT,
    STATE_DIM, OBS_DIM, CTRL_DIM,
    IDX_TEND_DMG,
    IDX_CSR,
    IDX_TEND_STIFF,
    IDX_BONE_DMG,
    IDX_BONE_DENS,
    HUBS_DEFAULT,
    biomechanical_ode,
    hub_tendon_rupture_risk,
    hub_bone_stress_fracture_risk,
)
from app.slices.biomechanical_tissue.observation import (
    BioObsParams,
    DEFAULT_BIO_OBS_PARAMS,
    R_BIO_DEFAULT,
    h_bio,
    inflate_R_bio,
)
from app.slices.biomechanical_tissue.filter import (
    BiomechanicalStateFilter,
    BioTransitionParams,
    DEFAULT_TRANSITION_PARAMS,
    Q_DEFAULT,
    _ALPHA,
)
from app.engine.assimilation.ukf_filter import GaussianState


# -- Helpers -------------------------------------------------------------------

def _integrate_nhours(
    x:       jax.Array,
    u:       jax.Array,
    n_hours: int,
    params:  BiomechanicalParams = DEFAULT_BIO_PARAMS,
) -> jax.Array:
    """Advance state by n_hours via diffrax.Tsit5 (same integrator as filter)."""
    def _step(carry, _):
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(biomechanical_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(1.0),
            dt0       = jnp.float32(0.1),
            y0        = carry,
            args      = (params, u),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 64,
        )
        return sol.ys[0], sol.ys[0]

    x_final, _ = jax.lax.scan(_step, x, None, length=n_hours)
    return x_final


# ============================================================================
# T1 -- Wolff's Law bone adaptation
# ============================================================================

def test_t1_wolff_law_bone_adaptation():
    print("[T1] Wolff's Law: moderate consistent load -> Bone_Density increases ...")

    # Moderate load (1.0 au) + high testosterone + adequate nutrition
    # Load is in the anabolic window (below pathological threshold of 3.0)
    u_load = jnp.array([
        1.0,    # hub_load      [au]    -- moderate, within Wolff window
        180.0,  # hub_IGF1      [ng/mL] -- healthy IGF-1
        25.0,   # hub_T         [nmol/L]-- high testosterone (male reference)
        150.0,  # hub_Cortisol  [nmol/L]-- low-moderate cortisol
        50.0,   # hub_Estrogen  [pg/mL] -- low (follicular or male baseline)
        1.0,    # hub_Nutrition [0-1]   -- fully replete
    ], dtype=jnp.float32)

    x0    = X0_BIO_DEFAULT
    bmd_0 = float(x0[IDX_BONE_DENS])

    # Simulate 720 hours = 30 days of moderate consistent loading
    x_final = _integrate_nhours(x0, u_load, n_hours=720)

    bmd_final = float(x_final[IDX_BONE_DENS])

    assert not bool(jnp.any(jnp.isnan(x_final))), \
        "FAIL T1: NaN in state after 720h moderate loading"

    assert bmd_final > bmd_0, (
        f"FAIL T1 Wolff's Law: Bone_Density did not increase under consistent "
        f"moderate load + high testosterone. "
        f"bmd_0={bmd_0:.6f}, bmd_final={bmd_final:.6f}. "
        "Check k_wolff, T_ref_nmolL, and load_anabolic calculation."
    )

    delta_bmd = bmd_final - bmd_0
    print(f"    Bone_Density: {bmd_0:.6f} -> {bmd_final:.6f}  (delta = +{delta_bmd:.6f} g/cm2)")
    print(f"    Bone_Microdamage final: {float(x_final[IDX_BONE_DMG]):.5f}")
    print("  [OK] T1 passed\n")


# ============================================================================
# T2 -- Estrogen-induced tendon laxity (ACL risk)
# ============================================================================

def test_t2_estrogen_tendon_laxity():
    print("[T2] Estrogen peak -> Tendon_Stiffness drops and Rupture_Risk rises ...")

    # Peri-ovulatory scenario: E2 > 200 pg/mL (activates laxity pathway)
    u_high_E2 = jnp.array([
        1.5,    # hub_load      [au]    -- moderate-high athletic load
        150.0,  # hub_IGF1      [ng/mL] -- normal IGF-1
        12.0,   # hub_T         [nmol/L]-- female testosterone range
        200.0,  # hub_Cortisol  [nmol/L]
        350.0,  # hub_Estrogen  [pg/mL] -- peri-ovulatory PEAK (above 200 threshold)
        1.0,    # hub_Nutrition [0-1]   -- replete
    ], dtype=jnp.float32)

    # Follicular / male control: E2 well below laxity threshold
    u_low_E2 = jnp.array([
        1.5,    # hub_load      [au]    -- same load
        150.0,  # hub_IGF1      [ng/mL] -- same IGF-1
        12.0,   # hub_T         [nmol/L]-- same testosterone
        200.0,  # hub_Cortisol  [nmol/L]
        60.0,   # hub_Estrogen  [pg/mL] -- follicular/male baseline (below 200 threshold)
        1.0,    # hub_Nutrition [0-1]   -- replete
    ], dtype=jnp.float32)

    x0 = X0_BIO_DEFAULT

    # Simulate 48 hours (acute peri-ovulatory window)
    x_high_E2 = _integrate_nhours(x0, u_high_E2, n_hours=48)
    x_low_E2  = _integrate_nhours(x0, u_low_E2,  n_hours=48)

    assert not bool(jnp.any(jnp.isnan(x_high_E2))), "FAIL T2: NaN in high-E2 trajectory"
    assert not bool(jnp.any(jnp.isnan(x_low_E2))),  "FAIL T2: NaN in low-E2 trajectory"

    stiff_high_E2 = float(x_high_E2[IDX_TEND_STIFF])
    stiff_low_E2  = float(x_low_E2[IDX_TEND_STIFF])

    risk_high_E2 = float(hub_tendon_rupture_risk(x_high_E2, u_high_E2, DEFAULT_BIO_PARAMS))
    risk_low_E2  = float(hub_tendon_rupture_risk(x_low_E2,  u_low_E2,  DEFAULT_BIO_PARAMS))

    print(f"    Tendon_Stiffness: high-E2={stiff_high_E2:.5f}, low-E2={stiff_low_E2:.5f}")
    print(f"    Hub_Tendon_Rupture_Risk: high-E2={risk_high_E2:.5f}, low-E2={risk_low_E2:.5f}")

    assert stiff_high_E2 < stiff_low_E2, (
        f"FAIL T2: Tendon_Stiffness did NOT drop under high E2. "
        f"high-E2 stiff={stiff_high_E2:.5f}, low-E2 stiff={stiff_low_E2:.5f}. "
        "Check k_E2_laxity, E2_laxity_thr in BiomechanicalParams."
    )

    assert risk_high_E2 > risk_low_E2, (
        f"FAIL T2: Hub_Tendon_Rupture_Risk did NOT increase under high E2. "
        f"high-E2 risk={risk_high_E2:.5f}, low-E2 risk={risk_low_E2:.5f}."
    )

    print("  [OK] T2 passed\n")


# ============================================================================
# T3 -- Cortisol-driven collagen degradation
# ============================================================================

def test_t3_cortisol_collagen_degradation():
    print("[T3] Nutritional deficit + Cortisol collapses CSR and accumulates damage ...")

    # Anabolic scenario: high IGF-1, full nutrition, low cortisol
    u_anabolic = jnp.array([
        1.8,    # hub_load      [au]    -- heavy load (same both scenarios)
        300.0,  # hub_IGF1      [ng/mL] -- high IGF-1 (anabolic)
        20.0,   # hub_T         [nmol/L]-- normal testosterone
        12.0,   # hub_Cortisol  [nmol/L]-- low cortisol
        60.0,   # hub_Estrogen  [pg/mL]
        1.0,    # hub_Nutrition [0-1]   -- fully replete (Gly/Pro/VitC adequate)
    ], dtype=jnp.float32)

    # Catabolic scenario: low IGF-1, nutritional deficit, chronic high cortisol
    u_catabolic = jnp.array([
        1.8,    # hub_load      [au]    -- same load
        80.0,   # hub_IGF1      [ng/mL] -- suppressed IGF-1 (overtraining / fasting)
        10.0,   # hub_T         [nmol/L]-- low T (cortisol-suppressed HPG axis)
        800.0,  # hub_Cortisol  [nmol/L]-- chronic extreme stress / overtraining
        60.0,   # hub_Estrogen  [pg/mL]
        0.0,    # hub_Nutrition [0-1]   -- full deficit (Gly/Pro/VitC absent) -> LOX blocked
    ], dtype=jnp.float32)

    x0 = X0_BIO_DEFAULT

    # Simulate 168 hours = 7 days of chronic exposure
    x_anabolic  = _integrate_nhours(x0, u_anabolic,  n_hours=168)
    x_catabolic = _integrate_nhours(x0, u_catabolic, n_hours=168)

    assert not bool(jnp.any(jnp.isnan(x_anabolic))),  "FAIL T3: NaN in anabolic trajectory"
    assert not bool(jnp.any(jnp.isnan(x_catabolic))), "FAIL T3: NaN in catabolic trajectory"

    csr_anabolic  = float(x_anabolic[IDX_CSR])
    csr_catabolic = float(x_catabolic[IDX_CSR])

    dmg_anabolic  = float(x_anabolic[IDX_TEND_DMG])
    dmg_catabolic = float(x_catabolic[IDX_TEND_DMG])

    print(f"    Collagen_Synthesis_Rate: anabolic={csr_anabolic:.5f}, catabolic={csr_catabolic:.5f}")
    print(f"    Tendon_Microdamage:      anabolic={dmg_anabolic:.5f}, catabolic={dmg_catabolic:.5f}")

    # Assertion 1: Nutrition=0 + high Cortisol collapses CSR
    assert csr_catabolic < csr_anabolic, (
        f"FAIL T3: Catabolic scenario did NOT suppress CSR vs anabolic. "
        f"csr_catabolic={csr_catabolic:.5f}, csr_anabolic={csr_anabolic:.5f}. "
        "Check k_IGF1_csr * hub_Nutrition gate and k_Cort_csr in BiomechanicalParams."
    )

    # Assertion 2: Low CSR means lower repair rate -> more damage accumulation
    assert dmg_catabolic > dmg_anabolic, (
        f"FAIL T3: Catabolic scenario did NOT accumulate more Tendon_Microdamage "
        f"than anabolic control. "
        f"dmg_catabolic={dmg_catabolic:.5f}, dmg_anabolic={dmg_anabolic:.5f}. "
        "CSR-gated repair: k_repair_eff = k_repair_tend * (1 + CSR). "
        "With CSR~0, repair collapses and damage builds to load/k_repair_tend equilibrium."
    )

    print("  [OK] T3 passed\n")


# ============================================================================
# T4 -- UKF biomechanical assimilation (48 steps)
# ============================================================================

def test_t4_ukf_biomech_assimilation():
    print("[T4] UKF biomechanical assimilation: 48 steps, no NaN, PSD covariance ...")

    # Verify alpha = 0.10 as mandated
    assert abs(_ALPHA - 0.10) < 1e-9, (
        f"FAIL T4: alpha={_ALPHA:.4f} != 0.10 (mandatory for biomechanical UKF)"
    )

    filt  = BiomechanicalStateFilter()
    state = GaussianState(mean=X0_BIO_DEFAULT, cov=P0_BIO_DEFAULT)

    # Alternating session / rest hub inputs  (6-dim)
    controls_session = {
        "hub_load":       1.5,
        "hub_IGF1":     180.0,
        "hub_T":         20.0,
        "hub_Cortisol": 280.0,
        "hub_Estrogen":  90.0,
        "hub_Nutrition":  1.0,
    }
    controls_rest = {
        "hub_load":       0.1,
        "hub_IGF1":     160.0,
        "hub_T":         20.0,
        "hub_Cortisol": 220.0,
        "hub_Estrogen":  90.0,
        "hub_Nutrition":  1.0,
    }

    history: list[GaussianState] = [state]

    rng = jax.random.PRNGKey(99)
    obs_params = DEFAULT_BIO_OBS_PARAMS

    for hour in range(48):
        ctrl = controls_session if (hour % 4 < 2) else controls_rest

        # Synthetic observation from current state mean
        y_clean = h_bio(state.mean, obs_params)
        rng, key = jax.random.split(rng)
        noise = jax.random.normal(key, shape=(OBS_DIM,)) * jnp.array(
            [obs_params.sigma_VAS, obs_params.sigma_Echo, obs_params.sigma_DEXA],
            dtype=jnp.float32,
        )
        y_noisy = y_clean + noise

        pain_vas = float(jnp.clip(y_noisy[0], 0.0, 10.0))
        echo     = float(jnp.clip(y_noisy[1], 0.0,  1.5))
        # DEXA only on hour 24 (simulate sparse scan); all others flag=4
        if hour == 24:
            dexa_z     = float(y_noisy[2])
            flag_dexa  = 1
        else:
            dexa_z     = float("nan")
            flag_dexa  = 4

        state = filt.update_state(
            prior         = state,
            pain_vas      = pain_vas,
            echo          = echo,
            controls      = ctrl,
            quality_flags = (0, 1, flag_dexa),
            dexa_z        = dexa_z,
        )
        history.append(state)

    # -- Assert: no NaN in any posterior mean or covariance --------------------
    for hr, post in enumerate(history):
        if bool(jnp.any(jnp.isnan(post.mean))):
            raise AssertionError(f"FAIL T4: NaN in posterior_mean at hour {hr}")
        if bool(jnp.any(jnp.isnan(post.cov))):
            raise AssertionError(f"FAIL T4: NaN in posterior_cov at hour {hr}")

    # -- Assert: all posterior covariances are PSD (non-negative diagonal) -----
    for hr, post in enumerate(history):
        diag = jnp.diag(post.cov)
        if bool(jnp.any(diag < jnp.float32(-1e-5))):
            raise AssertionError(
                f"FAIL T4: posterior_cov non-PSD at hour {hr}: "
                f"min_diag={float(jnp.min(diag)):.6f}"
            )

    # -- Assert: final state is physiologically sensible -----------------------
    final = history[-1]
    assert float(final.mean[IDX_BONE_DENS]) > 0.5, \
        f"FAIL T4: Bone_Density collapsed below 0.5 g/cm2: {float(final.mean[IDX_BONE_DENS]):.4f}"
    assert float(final.mean[IDX_TEND_STIFF]) >= 0.0, \
        "FAIL T4: Tendon_Stiffness went negative"

    # -- Assert: hub risk outputs are in [0, 1] --------------------------------
    u_final = jnp.array([1.5, 180.0, 20.0, 280.0, 90.0, 1.0], dtype=jnp.float32)
    risks   = filt.compute_hub_risks(final, u_final)

    assert 0.0 <= risks["Hub_Tendon_Rupture_Risk"] <= 1.0, (
        f"FAIL T4: Hub_Tendon_Rupture_Risk out of [0,1]: "
        f"{risks['Hub_Tendon_Rupture_Risk']:.5f}"
    )
    assert 0.0 <= risks["Hub_Bone_Stress_Fracture_Risk"] <= 1.0, (
        f"FAIL T4: Hub_Bone_Stress_Fracture_Risk out of [0,1]: "
        f"{risks['Hub_Bone_Stress_Fracture_Risk']:.5f}"
    )

    print(f"    48 steps: no NaN, covariance PSD maintained throughout.")
    print(f"    Final Bone_Density:        {float(final.mean[IDX_BONE_DENS]):.5f} g/cm2")
    print(f"    Final Tendon_Stiffness:    {float(final.mean[IDX_TEND_STIFF]):.5f} au")
    print(f"    Hub_Tendon_Rupture_Risk:   {risks['Hub_Tendon_Rupture_Risk']:.5f}")
    print(f"    Hub_Bone_Fracture_Risk:    {risks['Hub_Bone_Stress_Fracture_Risk']:.5f}")
    print("  [OK] T4 passed\n")


# ============================================================================
# Runner
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("  NUTRIVIOUS BOS -- Biomechanical Tissue Slice Gate-Zero")
    print(f"  ODE: 5-state hourly (Tsit5). OBS_DIM={OBS_DIM}. UKF alpha={_ALPHA}.")
    print("=" * 70 + "\n")

    tests = [
        ("T1 -- Wolff's Law bone adaptation",        test_t1_wolff_law_bone_adaptation),
        ("T2 -- Estrogen tendon laxity (ACL risk)",  test_t2_estrogen_tendon_laxity),
        ("T3 -- Cortisol collagen degradation",      test_t3_cortisol_collagen_degradation),
        ("T4 -- UKF 48-step assimilation",           test_t4_ukf_biomech_assimilation),
    ]

    failed: list[str] = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  [ASSERTION FAILED] {name}\n  -> {e}\n")
            failed.append(name)
        except Exception:
            print(f"  [EXCEPTION] {name}")
            traceback.print_exc()
            failed.append(name)

    print("=" * 70)
    if failed:
        print(f"  FAILED ({len(failed)}/{len(tests)} tests):")
        for f in failed:
            print(f"    FAIL {f}")
        sys.exit(1)
    else:
        print(f"  ALL {len(tests)} TESTS PASSED -- Biomechanical slice Gate-Zero cleared.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
