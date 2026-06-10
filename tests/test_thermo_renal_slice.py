"""
tests/test_thermo_renal_slice.py  —  Gate Zero: Thermo-Renal V2

Four crash tests for the 5-state, minute-timescale ODE:
  T1  test_heat_stroke_dynamics        — power → T_core rises, PV falls
  T2  test_hyponatremia_eah_danger     — sweat + pure water → Na_plasma drops vs control
  T3  test_starling_forces_fluid_shift — PV deficit → fluid shifts from IV to PV
  T4  test_ukf_thermo_assimilation     — 60-step UKF: no NaN, covariance PSD

All tests use the V2 5-state ODE (time unit: MINUTES).
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import diffrax

from app.slices.thermo_renal.ode import (
    thermo_renal_ode,
    ThermoRenalParams,
    DEFAULT_TR_PARAMS,
    X0_TR_DEFAULT,
    P0_TR_DEFAULT,
    IDX_CORE_TEMP,
    IDX_SKIN_TEMP,
    IDX_PLASMA_VOL,
    IDX_INTERS_VOL,
    IDX_PLASMA_NA,
    STATE_DIM,
)
from app.slices.thermo_renal.filter import ThermoRenalStateFilter, TRTransitionParams
from app.engine.assimilation.ukf_filter import GaussianState


# ── Shared helper ─────────────────────────────────────────────────────────────

def _run_ode(
    u:          list[float],
    t1_minutes: float,
    x0:         jax.Array | None = None,
    params:     ThermoRenalParams = DEFAULT_TR_PARAMS,
) -> jax.Array:
    """Integrate TR V2 ODE for t1_minutes under constant control u. Returns terminal x."""
    x0_ = X0_TR_DEFAULT if x0 is None else x0
    # Pad to CTRL_DIM=4 (u[3] = hub_basal_temp_offset, default 0.0)
    u_padded = list(u) + [0.0] * (4 - len(u))
    u_  = jnp.array(u_padded, dtype=jnp.float32)
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(thermo_renal_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.float32(t1_minutes),
        dt0       = jnp.float32(0.05),
        y0        = x0_,
        args      = (params, u_),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 16384,
    )
    return sol.ys[0]


# ── T1: Heat-Stroke Dynamics ──────────────────────────────────────────────────

class TestHeatStrokeDynamics:
    """
    hub_power_watts=300, no fluid or sodium intake, 60 minutes.

    Expected:
      - Core_Temp_C rises above 37°C (heat exceeds initial cooling capacity)
      - Plasma_Volume_L falls (sweat removes fluid, no replacement)
    """

    U = [300.0, 0.0, 0.0]   # [power_W, fluid_L_min, Na_mmol_min]

    def test_core_temp_rises(self):
        x_final   = _run_ode(self.U, t1_minutes=60.0)
        t_core    = float(x_final[IDX_CORE_TEMP])
        t_core_i  = float(X0_TR_DEFAULT[IDX_CORE_TEMP])

        print(f"\n[T1] Core_Temp: {t_core_i:.3f} -> {t_core:.3f} degC (60 min, 300W)")
        assert t_core > t_core_i + 0.25, (
            f"Core_Temp did not rise: {t_core:.3f}°C vs initial {t_core_i:.3f}°C. "
            f"Expected ≥ {t_core_i + 0.25:.3f}°C."
        )

    def test_plasma_volume_falls(self):
        x_final  = _run_ode(self.U, t1_minutes=60.0)
        pv_final = float(x_final[IDX_PLASMA_VOL])
        pv_init  = float(X0_TR_DEFAULT[IDX_PLASMA_VOL])

        print(f"\n[T1] Plasma_Volume: {pv_init:.3f} -> {pv_final:.3f} L (60 min, 300W, no fluids)")
        assert pv_final < pv_init - 0.05, (
            f"Plasma_Volume did not fall: {pv_final:.3f} L vs initial {pv_init:.3f} L. "
            f"Expected < {pv_init - 0.05:.3f} L."
        )


# ── T2: EAH — Hyponatremia Danger ────────────────────────────────────────────

class TestHyponatremiaEAHDanger:
    """
    Athlete sweats heavily (power=500W, hot conditions) and drinks ONLY pure water.
    Compare to a control that replaces both fluid and sodium.

    Expected:
      - Plasma_Sodium_mmol in EAH scenario < Plasma_Sodium_mmol in control
      - EAH plasma Na drops by at least 5% relative to initial (dilution + sweat loss)
    """

    # Elevated sweat: salty sweater variant
    PARAMS_SALTY = DEFAULT_TR_PARAMS._replace(
        sweat_sensitivity = 0.007,   # higher than population mean
        sweat_na_conc     = 70.0,    # salty sweater [mmol/L]
        T_ambient         = 30.0,    # hot environment
    )

    U_EAH     = [500.0, 0.025, 0.0]    # overdrinking pure water (1.5 L/h), no Na
    U_CONTROL = [500.0, 0.025, 1.5]    # same fluid but with Na replacement ~90 mmol/h

    DURATION_MIN = 120.0

    def test_na_drops_vs_control(self):
        x_eah  = _run_ode(self.U_EAH,     self.DURATION_MIN, params=self.PARAMS_SALTY)
        x_ctrl = _run_ode(self.U_CONTROL, self.DURATION_MIN, params=self.PARAMS_SALTY)

        na_eah  = float(x_eah[IDX_PLASMA_NA])
        na_ctrl = float(x_ctrl[IDX_PLASMA_NA])
        na_init = float(X0_TR_DEFAULT[IDX_PLASMA_NA])

        pv_eah = float(x_eah[IDX_PLASMA_VOL])
        conc_eah  = na_eah  / max(pv_eah, 0.1)
        pv_ctrl = float(x_ctrl[IDX_PLASMA_VOL])
        conc_ctrl = na_ctrl / max(pv_ctrl, 0.1)

        print(
            f"\n[T2] Plasma_Na: init={na_init:.1f} mmol | "
            f"EAH={na_eah:.1f} mmol ([Na+]={conc_eah:.1f} mmol/L) | "
            f"ctrl={na_ctrl:.1f} mmol ([Na+]={conc_ctrl:.1f} mmol/L)"
        )

        assert na_eah < na_ctrl, (
            f"EAH Na ({na_eah:.1f}) not below control Na ({na_ctrl:.1f}) — "
            f"dilution mechanism not working."
        )

    def test_na_drops_from_initial(self):
        x_eah  = _run_ode(self.U_EAH, self.DURATION_MIN, params=self.PARAMS_SALTY)
        na_eah = float(x_eah[IDX_PLASMA_NA])
        na_init = float(X0_TR_DEFAULT[IDX_PLASMA_NA])

        print(f"\n[T2] Na drop: {na_init:.1f} -> {na_eah:.1f} mmol ({100*(na_init-na_eah)/na_init:.1f}% loss)")
        assert na_eah < na_init * 0.95, (
            f"Na did not drop enough: {na_eah:.1f} mmol vs initial {na_init:.1f}. "
            f"Expected < {na_init * 0.95:.1f} (5% drop)."
        )


# ── T3: Starling Forces — Fluid Shift ─────────────────────────────────────────

class TestStarlingForcesFluidShift:
    """
    Start with Plasma_Volume_L = 2.5 L (depleted, below ref 4.2 L).
    No exercise, no fluid intake.

    Expected (Starling mechanism):
      - IV_final < IV_initial  (interstitium donates fluid to plasma)
      - PV_final > PV_initial  (plasma gains fluid from interstitium)
    """

    # Low PV, extra IV, thermoneutral, no exercise
    X0_LOW_PV: jax.Array = jnp.array([
        37.0,   # T_core at thermoneutral → sweat = 0
        34.0,   # T_skin
        2.5,    # PV depleted [L]
        13.9,   # IV elevated [L]  (total = 16.4, close to ref 16.2)
        350.0,  # Na_plasma [mmol]  (depleted slightly)
    ], dtype=jnp.float32)

    U = [0.0, 0.0, 0.0]    # no power, no fluid, no Na
    DURATION_MIN = 30.0

    def test_iv_decreases_and_pv_increases(self):
        x_final = _run_ode(self.U, self.DURATION_MIN, x0=self.X0_LOW_PV)

        pv_init  = float(self.X0_LOW_PV[IDX_PLASMA_VOL])
        pv_final = float(x_final[IDX_PLASMA_VOL])
        iv_init  = float(self.X0_LOW_PV[IDX_INTERS_VOL])
        iv_final = float(x_final[IDX_INTERS_VOL])

        print(
            f"\n[T3] PV: {pv_init:.3f} -> {pv_final:.3f} L  |  "
            f"IV: {iv_init:.3f} -> {iv_final:.3f} L  ({self.DURATION_MIN:.0f} min)"
        )

        assert iv_final < iv_init, (
            f"Interstitial volume did not decrease: {iv_final:.3f} L vs {iv_init:.3f} L. "
            f"Starling shift from IV→PV expected."
        )
        assert pv_final > pv_init, (
            f"Plasma volume did not increase: {pv_final:.3f} L vs {pv_init:.3f} L. "
            f"Starling inflow from IV expected."
        )


# ── T4: UKF Assimilation — 60 Steps ──────────────────────────────────────────

class TestUKFThermoAssimilation:
    """
    60 predict-update steps at dt_minutes=1.0 (= 60 minutes).
    Each step assimilates a noisy Core_Temp observation; BW drop absent.

    Required:
      1. No NaN in posterior_mean at any step.
      2. Covariance diagonal > 0 at every step (PSD maintained).
    """

    CONTROLS = {
        "hub_power_watts":           200.0,
        "hub_fluid_intake_L_min":    0.008,   # ~500 mL/h
        "hub_sodium_intake_mmol_min": 0.4,
    }

    def test_no_nan_and_psd_over_60_steps(self):
        filt  = ThermoRenalStateFilter()
        state = GaussianState(mean=X0_TR_DEFAULT, cov=P0_TR_DEFAULT)
        rng   = np.random.default_rng(seed=7)

        for step in range(60):
            # Noisy Core_Temp observation (σ ≈ 0.3°C)
            ct_noisy = float(state.mean[IDX_CORE_TEMP]) + rng.normal(0.0, 0.3)

            state = filt.update_state(
                prior          = state,
                core_temp_obs  = ct_noisy,
                bw_drop_obs    = float("nan"),
                controls       = self.CONTROLS,
                dt_minutes     = 1.0,
                quality_flags  = (0, 4),
            )

            # 1. No NaN
            assert not bool(jnp.any(jnp.isnan(state.mean))), (
                f"Step {step+1}/60: posterior_mean contains NaN — UKF diverged."
            )

            # 2. PSD covariance
            diag = jnp.diag(state.cov)
            assert bool(jnp.all(diag > jnp.float32(0.0))), (
                f"Step {step+1}/60: covariance diagonal non-positive. diag={diag}"
            )

        t_core_final = float(state.mean[IDX_CORE_TEMP])
        pv_final     = float(state.mean[IDX_PLASMA_VOL])
        na_final     = float(state.mean[IDX_PLASMA_NA])
        diag_final   = jnp.diag(state.cov)

        print(
            f"\n[T4] 60 steps OK. T_core={t_core_final:.3f}°C, "
            f"PV={pv_final:.3f}L, Na={na_final:.1f}mmol"
        )
        print(f"[T4] Final cov diagonal: {diag_final}")
