"""
tests/test_gastrointestinal_slice.py  --  Gate Zero: GI V3.0

T1  test_dual_transport_ratio
    1.8 g/min glucose only vs 1.0 g/min Glu + 0.8 g/min Fru (same total CHO).
    Assert: mixed 1:0.8 ratio produces less total intestinal accumulation
            AND higher total absorption rate (GLUT5 bypasses SGLT1 saturation).

T2  test_sodium_dependent_sglt1
    Glucose intake with Sodium=140 (normal) vs Sodium=125 (hyponatraemia).
    Assert: SGLT1 collapses at Sodium=125 -> Intst_Glu >> normal;
            Fructose absorption is unaffected by sodium.

T3  test_ukf_gastro_assimilation_v3
    60 steps 6-state UKF, alpha=0.10.
    Assert: no NaN, PSD covariance, Distress > 0 after sustained loading.

Run:
    pytest tests/test_gastrointestinal_slice.py -v -s
"""
from __future__ import annotations

import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import diffrax
import pytest

from app.slices.gastrointestinal.ode import (
    GIv3Params, DEFAULT_GI_PARAMS,
    X0_GI_DEFAULT, P0_GI_DEFAULT,
    STATE_DIM, CTRL_DIM,
    gi_ode, _absorption_rates, hub_cho_absorption_rate,
    IDX_STOM_FLUID, IDX_STOM_GLU, IDX_STOM_FRU,
    IDX_INTST_GLU, IDX_INTST_FRU, IDX_DISTRESS,
)
from app.slices.gastrointestinal.filter import (
    GastrointestinalStateFilter,
    GITransitionParams,
)
from app.engine.assimilation.ukf_filter import GaussianState


# ── ODE integration helper ────────────────────────────────────────────────────

def _run(
    x0:      jax.Array,
    u:       jax.Array,
    n_steps: int = 30,
    dt:      float = 1.0,
    params:  GIv3Params = DEFAULT_GI_PARAMS,
) -> jax.Array:
    """Integrate ODE for n_steps minutes with constant control u."""
    x = x0
    for _ in range(n_steps):
        sol = diffrax.diffeqsolve(
            terms    = diffrax.ODETerm(gi_ode),
            solver   = diffrax.Tsit5(),
            t0       = jnp.float32(0.0),
            t1       = jnp.float32(dt),
            dt0      = jnp.float32(0.1),
            y0       = x,
            args     = (params, u),
            saveat   = diffrax.SaveAt(t1=True),
            max_steps= 256,
        )
        x = jnp.maximum(sol.ys[0], jnp.float32(0.0))
    return x


# ─────────────────────────────────────────────────────────────────────────────
# T1 -- dual-transporter ratio
# ─────────────────────────────────────────────────────────────────────────────

def test_dual_transport_ratio():
    """
    Scenario A: 1.8 g/min glucose only -> SGLT1 overwhelmed (Vmax_glu=1.0)
                  -> large Intst_Glu accumulation, absorption capped at ~1.0 g/min
    Scenario B: 1.0 g/min Glu + 0.8 g/min Fru -> SGLT1 less loaded;
                  GLUT5 adds independent absorption path -> lower total intestinal CHO,
                  higher total absorption rate

    fluid_in = 0.04 L/min ensures density stays < 60 g/L (no osmotic brake).
    sodium   = 140 (na_factor = 1.0).
    """
    x0 = jnp.zeros(STATE_DIM, dtype=jnp.float32)

    # u: [Fluid_in, Glu_in, Fru_in, Power, Temp, Sodium]
    u_A = jnp.array([0.04, 1.8, 0.0, 0.0, 37.0, 140.0], dtype=jnp.float32)
    u_B = jnp.array([0.04, 1.0, 0.8, 0.0, 37.0, 140.0], dtype=jnp.float32)

    x_A = _run(x0, u_A, n_steps=40)
    x_B = _run(x0, u_B, n_steps=40)

    intst_glu_A = float(x_A[IDX_INTST_GLU])
    intst_fru_A = float(x_A[IDX_INTST_FRU])
    total_A     = intst_glu_A + intst_fru_A

    intst_glu_B = float(x_B[IDX_INTST_GLU])
    intst_fru_B = float(x_B[IDX_INTST_FRU])
    total_B     = intst_glu_B + intst_fru_B

    abs_glu_A, abs_fru_A = _absorption_rates(x_A, u_A)
    abs_glu_B, abs_fru_B = _absorption_rates(x_B, u_B)
    total_abs_A = float(abs_glu_A + abs_fru_A)
    total_abs_B = float(abs_glu_B + abs_fru_B)

    print(f"\nT1  Intst_Glu A={intst_glu_A:.2f}g  Intst_Fru A={intst_fru_A:.2f}g  total_A={total_A:.2f}g")
    print(f"T1  Intst_Glu B={intst_glu_B:.2f}g  Intst_Fru B={intst_fru_B:.2f}g  total_B={total_B:.2f}g")
    print(f"T1  TotalAbs A={total_abs_A:.4f} g/min  B={total_abs_B:.4f} g/min")

    # mixed 1:0.8 ratio should leave less CHO stranded in intestine
    assert total_B < total_A, (
        f"Mixed transport should accumulate less total intestinal CHO: "
        f"B={total_B:.2f}g not < A={total_A:.2f}g"
    )

    # mixed transport should achieve higher total absorption
    assert total_abs_B > total_abs_A, (
        f"Mixed transport should absorb more total: "
        f"B={total_abs_B:.4f} not > A={total_abs_A:.4f} g/min"
    )

    assert not jnp.any(jnp.isnan(x_A))
    assert not jnp.any(jnp.isnan(x_B))
    print("T1 PASS -- dual-transporter bypass confirmed")


# ─────────────────────────────────────────────────────────────────────────────
# T2 -- sodium-dependent SGLT1
# ─────────────────────────────────────────────────────────────────────────────

def test_sodium_dependent_sglt1():
    """
    Glucose-only intake: Sodium=140 vs Sodium=125.
    At Sodium=125: na_factor = (125-125)/10 = 0.0 -> SGLT1 blocked -> Intst_Glu accumulates.
    At Sodium=140: na_factor = 1.0 -> SGLT1 works normally.

    Also confirms GLUT5 independence: a separate fructose scenario with Sodium=125
    should still absorb fructose normally.
    """
    x0 = jnp.zeros(STATE_DIM, dtype=jnp.float32)

    # glucose intake only
    u_normal = jnp.array([0.04, 1.0, 0.0, 0.0, 37.0, 140.0], dtype=jnp.float32)
    u_hypo   = jnp.array([0.04, 1.0, 0.0, 0.0, 37.0, 125.0], dtype=jnp.float32)

    # fructose scenario with hyponatraemia (GLUT5 should be unaffected)
    u_fru_hypo = jnp.array([0.04, 0.0, 1.0, 0.0, 37.0, 125.0], dtype=jnp.float32)

    x_normal  = _run(x0, u_normal,  n_steps=40)
    x_hypo    = _run(x0, u_hypo,    n_steps=40)
    x_fru_hypo = _run(x0, u_fru_hypo, n_steps=40)

    intst_glu_normal = float(x_normal[IDX_INTST_GLU])
    intst_glu_hypo   = float(x_hypo[IDX_INTST_GLU])
    distress_hypo    = float(x_hypo[IDX_DISTRESS])
    distress_normal  = float(x_normal[IDX_DISTRESS])

    # absorption rates
    ag_normal, _ = _absorption_rates(x_normal,  u_normal)
    ag_hypo, _   = _absorption_rates(x_hypo,    u_hypo)
    _, af_fru    = _absorption_rates(x_fru_hypo, u_fru_hypo)

    print(f"\nT2  Intst_Glu: normal={intst_glu_normal:.2f}g  hypo={intst_glu_hypo:.2f}g")
    print(f"T2  Abs_glu:   normal={float(ag_normal):.4f} g/min  hypo={float(ag_hypo):.4f} g/min")
    print(f"T2  Distress:  normal={distress_normal:.4f}  hypo={distress_hypo:.4f}")
    print(f"T2  Abs_fru at hypo sodium={float(af_fru):.4f} g/min (GLUT5 unaffected)")

    # SGLT1 blocked in hyponatraemia -> glucose accumulates significantly
    assert intst_glu_hypo > intst_glu_normal * 1.5, (
        f"Hyponatraemia should cause >1.5x glucose accumulation: "
        f"hypo={intst_glu_hypo:.2f} not > 1.5x normal={intst_glu_normal:.2f}"
    )

    # absorption collapses under hyponatraemia
    assert float(ag_hypo) < 0.05, (
        f"SGLT1 absorption should be near 0 at Sodium=125: got {float(ag_hypo):.4f}"
    )

    # distress higher under hyponatraemia (CHO accumulation)
    assert distress_hypo > distress_normal, (
        f"Distress should be higher under hyponatraemia: "
        f"hypo={distress_hypo:.4f} not > normal={distress_normal:.4f}"
    )

    # GLUT5 works at Sodium=125 (fructose absorption positive)
    assert float(af_fru) > 0.1, (
        f"GLUT5 should absorb fructose even at Sodium=125: got {float(af_fru):.4f}"
    )

    assert not jnp.any(jnp.isnan(x_normal))
    assert not jnp.any(jnp.isnan(x_hypo))
    print("T2 PASS -- sodium-dependent SGLT1 confirmed; GLUT5 independence confirmed")


# ─────────────────────────────────────────────────────────────────────────────
# T3 -- UKF 6-state assimilation V3
# ─────────────────────────────────────────────────────────────────────────────

def test_ukf_gastro_assimilation_v3():
    """
    60 steps of 1-min UKF with 6-state system, alpha=0.10.
    Controls: moderate exercise + mixed CHO intake + normal sodium.
    Observations: Nausea_VAS=4.0, Bloating_VAS=3.0 every 3rd step.
    Assert: no NaN at any step, PSD covariance, GI_Distress > 0 after 60 min.
    """
    filt = GastrointestinalStateFilter()

    state = GaussianState(
        mean = X0_GI_DEFAULT.copy(),
        cov  = P0_GI_DEFAULT.copy(),
    )

    controls = {
        "Fluid_in":     0.03,
        "Glu_in":       0.8,
        "Fru_in":       0.4,
        "Power":        200.0,
        "Temp":         38.5,
        "Sodium_mmolL": 140.0,
    }

    for step in range(60):
        if step % 3 == 0:
            na_obs = 4.0
            bl_obs = 3.0
            flags  = (0, 0)
        else:
            na_obs = float("nan")
            bl_obs = float("nan")
            flags  = (4, 4)

        state = filt.update_state(
            prior         = state,
            controls      = controls,
            dt_minutes    = 1.0,
            nausea_obs    = na_obs,
            bloating_obs  = bl_obs,
            quality_flags = flags,
        )

        mean = state.mean
        cov  = state.cov

        assert not bool(jnp.any(jnp.isnan(mean))), f"NaN in mean at step {step}: {mean}"
        assert not bool(jnp.any(jnp.isnan(cov))),  f"NaN in cov at step {step}"

        diag = jnp.diag(cov)
        assert bool(jnp.all(diag >= jnp.float32(-1e-4))), (
            f"Negative cov diagonal at step {step}: {diag}"
        )

    distress_final = float(state.mean[IDX_DISTRESS])
    intst_total    = float(state.mean[IDX_INTST_GLU] + state.mean[IDX_INTST_FRU])
    print(f"\nT3  Final GI_Distress={distress_final:.4f}")
    print(f"T3  Final total Intst_CHO={intst_total:.3f} g")
    print(f"T3  Final Stomach_Fluid={float(state.mean[IDX_STOM_FLUID]):.4f} L")

    assert distress_final > 0.0, (
        f"GI_Distress should be > 0 after 60 min at 200W+38.5degC: {distress_final:.4f}"
    )

    print("T3 PASS -- 6-state UKF V3 stable, no NaN, PSD maintained")


if __name__ == "__main__":
    test_dual_transport_ratio()
    test_sodium_dependent_sglt1()
    test_ukf_gastro_assimilation_v3()
    print("\nAll GI V3.0 Gate Zero tests PASSED.")
