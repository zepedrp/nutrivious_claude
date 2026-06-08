"""
tests/test_metabolic_reds_slice.py — Module 13 Gate Zero

Three mechanistic acceptance tests for the Metabolic RED-S / Thyroid / Fatmax slice.
All tests must pass before Module 13 can enter the integration pipeline.

T1 test_reds_t3_suppression
    Verify the SPINA-Dietrich deiodinase collapse: a 10-day caloric deficit
    (EA_net → 20 kcal/kg FFM) must suppress fT3, reduce RMR_Multiplier, and
    trigger the Phase3Envelope EA hard constraint (EA_Pool < 30).

T2 test_ukf_sparse_thyroid
    Verify UKF stability under sparse assimilation: daily RMR_Proxy + one
    episodic fT3 lab draw over 5 daily steps. Posterior must be NaN-free and
    covariance positive-semi-definite.

T3 test_fatmax_metabolic_protection
    Verify that a 21-day energy deficit drags Fatmax_State downward alongside
    RMR_Multiplier, confirming the mitochondrial biogenesis coupling (Block E).
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import diffrax

from app.slices.metabolic_reds.ode import (
    metabolic_reds_ode,
    DEFAULT_MR_PARAMS,
    X0_MR_DEFAULT,
    IDX_EA_POOL,
    IDX_FREE_T4,
    IDX_FREE_T3,
    IDX_RMR_MULT,
    IDX_FATMAX,
)
from app.slices.metabolic_reds.envelope import build_metabolic_reds_envelope
from app.slices.metabolic_reds.filter import (
    MetabolicRedsStateFilter,
    P0_MR_DEFAULT,
)
from app.engine.assimilation.ukf_filter import GaussianState


# ── Shared helpers ────────────────────────────────────────────────────────────

def _run_ode(
    u:       jax.Array,
    t1_days: float,
    x0:      jax.Array | None = None,
) -> jax.Array:
    """Integrate metabolic_reds_ode from 0 to t1_days; return final state."""
    y0 = x0 if x0 is not None else jnp.array(X0_MR_DEFAULT, dtype=jnp.float32)
    sol = diffrax.diffeqsolve(
        terms    = diffrax.ODETerm(metabolic_reds_ode),
        solver   = diffrax.Tsit5(),
        t0       = jnp.float32(0.0),
        t1       = jnp.float32(t1_days),
        dt0      = jnp.float32(0.2),
        y0       = y0,
        args     = (u, DEFAULT_MR_PARAMS),
        saveat   = diffrax.SaveAt(t1=True),
        max_steps= 512,
    )
    return sol.ys[0]


# ── T1 — RED-S T3 suppression and envelope alarm ─────────────────────────────

def test_reds_t3_suppression():
    """
    10-day caloric deficit (CI=1500, TEE=3000, FFM=60 kg) → EA_net = 20 kcal/kg FFM.
    Assertions:
      (a) Free_T3 falls significantly below 5.0 pmol/L baseline (T3 < 4.0).
      (b) RMR_Multiplier begins to adapt downward (RMR_Mult < 0.95).
      (c) EA_Pool drops below RED-S hard floor (EA_Pool < 30 kcal/kg FFM).
      (d) Phase3Envelope EA hard constraint is violated.
      (e) No NaN in final state.
    """
    u = jnp.array([
        1500.0,   # hub_caloric_intake       [kcal/day]  — severe deficit
        3000.0,   # hub_total_energy_expenditure [kcal/day]
        0.3,      # hub_training_stress      [au] — moderate training
        60.0,     # hub_fat_free_mass_kg     [kg]
    ], dtype=jnp.float32)

    # EA_net = (1500 - 3000) / 60 + 45 = -25 + 45 = 20 kcal/kg FFM
    # After 10 days (tau=7d): EA_Pool ~= 26  → below RED-S floor of 30
    x_final = _run_ode(u, t1_days=10.0)

    fT3_val  = float(x_final[IDX_FREE_T3])
    rmr_val  = float(x_final[IDX_RMR_MULT])
    ea_val   = float(x_final[IDX_EA_POOL])

    # (a) T3 suppressed: deiodinase half-active when EA < 30
    assert fT3_val < 4.0, (
        f"Free_T3 should collapse below 4.0 pmol/L under 10-day deficit; "
        f"got {fT3_val:.3f}"
    )

    # (b) RMR_Multiplier adapting (14-day lag — partial response expected)
    assert rmr_val < 0.95, (
        f"RMR_Multiplier should start adapting below 0.95; got {rmr_val:.4f}"
    )

    # (c) EA_Pool below RED-S floor
    assert ea_val < 30.0, (
        f"EA_Pool should be below RED-S floor 30 kcal/kg FFM; got {ea_val:.2f}"
    )

    # (d) Envelope hard constraint triggered
    envelope = build_metabolic_reds_envelope()
    ea_violated = any(
        ea_val < c.rhs
        for c in envelope.hard_constraints
        if c.state_key == "Hub_MR_EA_Pool"
    )
    assert ea_violated, (
        f"Phase3Envelope EA hard constraint should be violated at EA={ea_val:.2f}; "
        f"floor={30.0}"
    )

    # (e) NaN-free
    assert not bool(jnp.any(jnp.isnan(x_final))), "Final state must not contain NaN"

    print(f"\n[T1] EA_Pool={ea_val:.2f}  Free_T3={fT3_val:.3f}  RMR_Mult={rmr_val:.4f}")
    print(f"[T1] EA hard constraint violated={ea_violated}")


# ── T2 — UKF stability under sparse thyroid assimilation ─────────────────────

def test_ukf_sparse_thyroid():
    """
    5 daily UKF steps:
      - Days 1-5: RMR_Proxy available (flag=0), fT4 missing (flag=4).
      - Day 3 only: fT3 lab draw provided (flag=0).
    Assertions:
      (a) posterior_mean is NaN-free after all 5 steps.
      (b) posterior_cov diagonal is >= -1e-6 (positive-semi-definite).
    """
    filt = MetabolicRedsStateFilter()

    mean0 = jnp.array(X0_MR_DEFAULT, dtype=jnp.float32)
    prior = GaussianState(mean=mean0, cov=P0_MR_DEFAULT)

    controls = {
        "hub_caloric_intake":           2300.0,   # slight deficit
        "hub_total_energy_expenditure": 2600.0,
        "hub_training_stress":          0.2,
        "hub_fat_free_mass_kg":         58.0,
    }

    for day in range(1, 6):
        # fT3 lab only on day 3
        fT3_obs  = 4.9 if day == 3 else float("nan")
        fT4_obs  = float("nan")          # always episodic — not available
        rmr_obs  = 1680.0 + day * 5.0   # wearable daily

        flag_fT3 = 0 if (day == 3) else 4
        flag_fT4 = 4
        flag_rmr = 0

        prior = filt.update_state(
            prior         = prior,
            controls      = controls,
            dt_days       = 1.0,
            fT3_obs       = fT3_obs,
            fT4_obs       = fT4_obs,
            rmr_proxy     = rmr_obs,
            quality_flags = (flag_fT3, flag_fT4, flag_rmr),
        )

    # (a) posterior_mean NaN-free
    assert not bool(jnp.any(jnp.isnan(prior.mean))), (
        f"posterior_mean contains NaN after 5-step assimilation: {prior.mean}"
    )

    # (b) posterior_cov PSD
    diag = jnp.diag(prior.cov)
    assert bool(jnp.all(diag >= jnp.float32(-1e-6))), (
        f"posterior_cov diagonal must be >= -1e-6 (PSD); got {diag}"
    )

    print(f"\n[T2] posterior_mean = {prior.mean}")
    print(f"[T2] posterior_cov diag = {jnp.diag(prior.cov)}")
    print(f"[T2] NaN-free={not bool(jnp.any(jnp.isnan(prior.mean)))}  PSD={bool(jnp.all(diag >= -1e-6))}")


# ── T3 — Fatmax metabolic protection coupling ─────────────────────────────────

def test_fatmax_metabolic_protection():
    """
    21-day energy deficit (EA_net = 20 kcal/kg FFM):
    Assertions:
      (a) RMR_Multiplier falls below 0.85 (severe metabolic adaptation threshold).
      (b) Fatmax_State falls below 0.90 (mitochondrial biogenesis coupled to RMR).
      (c) Fatmax_State >= RMR_Multiplier − 0.01: Fatmax lags RMR due to slower
          time constant (tau_fatmax=21d > tau_rmr=14d); Fatmax decays slower.
      (d) No NaN in final state.
    """
    u = jnp.array([
        1500.0,   # hub_caloric_intake       [kcal/day]
        3000.0,   # hub_total_energy_expenditure [kcal/day]
        0.0,      # hub_training_stress      [au] — rest (isolate metabolic effect)
        60.0,     # hub_fat_free_mass_kg     [kg]
    ], dtype=jnp.float32)

    # After 21 days (~3×tau_rmr=14d), RMR should be well below 0.85
    x_final = _run_ode(u, t1_days=21.0)

    rmr_val   = float(x_final[IDX_RMR_MULT])
    fatmax_val = float(x_final[IDX_FATMAX])

    # (a) RMR severely adapted (21 days > 1.5 tau_rmr=14)
    assert rmr_val < 0.85, (
        f"RMR_Multiplier should be < 0.85 after 21-day deficit; got {rmr_val:.4f}"
    )

    # (b) Fatmax also fallen (mitochondrial biogenesis tracks RMR)
    assert fatmax_val < 0.90, (
        f"Fatmax_State should be < 0.90 after 21-day deficit; got {fatmax_val:.4f}"
    )

    # (c) Fatmax lags RMR — Fatmax decays slower (tau=21d > tau_rmr=14d),
    # so Fatmax >= RMR throughout the descent.
    assert fatmax_val >= rmr_val - 0.01, (
        f"Fatmax ({fatmax_val:.4f}) should lag behind RMR ({rmr_val:.4f}) "
        f"due to longer time constant; expected Fatmax >= RMR."
    )

    # (d) NaN-free
    assert not bool(jnp.any(jnp.isnan(x_final))), "Final state must not contain NaN"

    print(f"\n[T3] RMR_Multiplier={rmr_val:.4f}  Fatmax_State={fatmax_val:.4f}")
    print(f"[T3] Fatmax >= RMR lag check: {fatmax_val:.4f} >= {rmr_val:.4f}")
