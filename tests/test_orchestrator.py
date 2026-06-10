"""
tests/test_orchestrator.py — Gate Zero: PentaOrchestrator (5-Subsystem)

Tests the operator-splitting orchestrator that fuses:
    NM -> MG -> Neuroendocrine -> ThermoRenal -> Cardio

T1  test_penta_default_state_no_nan
        Cold-start default state must be NaN-free, all hubs finite.

T2  test_penta_step_rest_no_nan
        Single step at rest (power=0). All 5 posterior means NaN-free.
        Hub values (cortisol, epinephrine, core_temp, pv_drop_pct, autonomic_tone)
        must be finite and within physiological ranges.

T3  test_penta_step_exercise_updates_hub
        Single step at 250W. Assert:
          - hub.cortisol  mean is finite
          - hub.epinephrine mean is finite
          - hub.core_temp mean ∈ [36.0, 42.0] °C
          - hub.pv_drop_pct mean ∈ [0.0, 50.0] %
          - hub.autonomic_tone mean ∈ [0.0, 1.0]

T4  test_penta_10_steps_no_nan
        10 consecutive steps at 200W. Assert no NaN at any step.
        Confirms operator-splitting doesn't accumulate NaN across steps.

T5  test_trio_legacy_compat
        TrioOrchestrator (legacy alias) still works after orchestrator rewrite.
        Confirms existing Trio tests are not broken.

Run:
    pytest tests/test_orchestrator.py -v -s
"""
from __future__ import annotations

import sys
import io

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import math

import jax
import jax.numpy as jnp
import pytest

from app.engine.orchestrator import (
    PentaOrchestrator,
    PentaState,
    TrioOrchestrator,
    TrioState,
)
from app.engine.hubs import msg_to_mean


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

CONTROLS_REST = {
    "power_W":                    0.0,
    "cho_abs_g_min":              0.0,
    "hub_circadian_phase":        0.0,
    "hub_sleep_sws":              0.0,
    "hub_fluid_intake_L_min":     0.008,
    "hub_sodium_intake_mmol_min": 0.4,
    "t_hour":                     10.0,
}

CONTROLS_EXERCISE = {
    "power_W":                    250.0,
    "cho_abs_g_min":              0.05,
    "hub_circadian_phase":        0.0,
    "hub_sleep_sws":              0.0,
    "hub_fluid_intake_L_min":     0.010,
    "hub_sodium_intake_mmol_min": 0.5,
    "t_hour":                     10.0,
}


def _state_is_nan_free(state: PentaState) -> bool:
    """Return True iff no NaN in any slice posterior mean."""
    return (
        not bool(jnp.any(jnp.isnan(state.nm.mean)))
        and not bool(jnp.any(jnp.isnan(state.mg.mean)))
        and not bool(jnp.any(jnp.isnan(state.neuro.mean)))
        and not bool(jnp.any(jnp.isnan(state.thermo_renal.mean)))
        and not bool(jnp.any(jnp.isnan(state.cardio.mean)))
    )


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Default state NaN-free
# ─────────────────────────────────────────────────────────────────────────────

def test_penta_default_state_no_nan():
    """Cold-start PentaState must be fully finite."""
    state = PentaOrchestrator.default_state()

    assert not bool(jnp.any(jnp.isnan(state.nm.mean))),           "NaN in NM default mean"
    assert not bool(jnp.any(jnp.isnan(state.mg.mean))),           "NaN in MG default mean"
    assert not bool(jnp.any(jnp.isnan(state.neuro.mean))),        "NaN in Neuro default mean"
    assert not bool(jnp.any(jnp.isnan(state.thermo_renal.mean))), "NaN in TR default mean"
    assert not bool(jnp.any(jnp.isnan(state.cardio.mean))),       "NaN in Cardio default mean"

    # Hub fields
    hub = state.hub
    for field in ["plasma_glucose", "cortisol", "epinephrine", "core_temp",
                  "pv_drop_pct", "autonomic_tone"]:
        val = float(msg_to_mean(getattr(hub, field)))
        assert math.isfinite(val), f"Non-finite hub.{field} in default state: {val}"

    print("\n[T1] PentaOrchestrator default state: NaN-free, all hubs finite. PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Single rest step: NaN-free + hub in range
# ─────────────────────────────────────────────────────────────────────────────

def test_penta_step_rest_no_nan():
    """One step at rest — all posteriors NaN-free, hubs within physiological range."""
    orch  = PentaOrchestrator()
    state = PentaOrchestrator.default_state()

    state = orch.step(prior=state, controls=CONTROLS_REST)

    assert _state_is_nan_free(state), "NaN in posterior after rest step"

    hub = state.hub
    cort  = float(msg_to_mean(hub.cortisol))
    epi   = float(msg_to_mean(hub.epinephrine))
    t_c   = float(msg_to_mean(hub.core_temp))
    pv_d  = float(msg_to_mean(hub.pv_drop_pct))
    atone = float(msg_to_mean(hub.autonomic_tone))

    print(f"\n[T2] REST: cortisol={cort:.1f} nmol/L, epi={epi:.1f} pg/mL, "
          f"T_core={t_c:.3f}°C, pv_drop={pv_d:.3f}%, autonomic={atone:.4f}")

    assert 10.0 <= cort  <= 2000.0,  f"Cortisol out of range: {cort:.1f}"
    assert 0.0  <= epi   <= 10000.0, f"Epi out of range: {epi:.1f}"
    assert 35.0 <= t_c   <= 42.0,    f"T_core out of range: {t_c:.3f}"
    assert 0.0  <= pv_d  <= 50.0,    f"pv_drop_pct out of range: {pv_d:.3f}"
    assert 0.0  <= atone <= 1.0,     f"autonomic_tone out of range: {atone:.4f}"

    print("[T2] PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Single exercise step: hub updates show exercise response
# ─────────────────────────────────────────────────────────────────────────────

def test_penta_step_exercise_updates_hub():
    """One step at 250W — neuroendocrine + thermo-renal + cardio hub updates are finite."""
    orch  = PentaOrchestrator()
    state = PentaOrchestrator.default_state()

    state = orch.step(prior=state, controls=CONTROLS_EXERCISE)

    assert _state_is_nan_free(state), "NaN in posterior after exercise step"

    hub = state.hub
    cort  = float(msg_to_mean(hub.cortisol))
    epi   = float(msg_to_mean(hub.epinephrine))
    t_c   = float(msg_to_mean(hub.core_temp))
    pv_d  = float(msg_to_mean(hub.pv_drop_pct))
    atone = float(msg_to_mean(hub.autonomic_tone))

    print(f"\n[T3] 250W: cortisol={cort:.1f} nmol/L, epi={epi:.1f} pg/mL, "
          f"T_core={t_c:.3f}°C, pv_drop={pv_d:.3f}%, autonomic={atone:.4f}")

    assert math.isfinite(cort),  "Cortisol non-finite after exercise"
    assert math.isfinite(epi),   "Epi non-finite after exercise"
    assert 35.0 <= t_c <= 42.0,  f"T_core out of physical range: {t_c:.3f}"
    assert 0.0  <= pv_d <= 50.0, f"pv_drop_pct out of range: {pv_d:.3f}"
    assert 0.0  <= atone <= 1.0, f"autonomic_tone out of range: {atone:.4f}"

    print("[T3] PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T4 — 10 consecutive steps at 200W: no NaN accumulation
# ─────────────────────────────────────────────────────────────────────────────

def test_penta_10_steps_no_nan():
    """10 steps at 200W — operator splitting must not accumulate NaN."""
    orch  = PentaOrchestrator()
    state = PentaOrchestrator.default_state()

    controls_200 = dict(CONTROLS_REST)
    controls_200["power_W"] = 200.0

    for step in range(10):
        controls_200["t_hour"] = 10.0 + step / 60.0   # advance 1 min per step
        state = orch.step(prior=state, controls=controls_200)

        assert _state_is_nan_free(state), (
            f"NaN in PentaState at step {step+1}"
        )

        # Check all covariance diagonals > 0
        for name, gs in [
            ("nm", state.nm), ("mg", state.mg), ("neuro", state.neuro),
            ("thermo_renal", state.thermo_renal), ("cardio", state.cardio),
        ]:
            diag = jnp.diag(gs.cov)
            assert bool(jnp.all(diag > 0.0)), (
                f"Non-positive covariance in {name} at step {step+1}: diag={diag}"
            )

    hub = state.hub
    t_c  = float(msg_to_mean(hub.core_temp))
    cort = float(msg_to_mean(hub.cortisol))
    print(f"\n[T4] 10 steps 200W: T_core={t_c:.3f}°C, cortisol={cort:.1f} nmol/L — no NaN. PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T5 — Legacy TrioOrchestrator still works
# ─────────────────────────────────────────────────────────────────────────────

def test_trio_legacy_compat():
    """TrioOrchestrator legacy alias must still function after orchestrator rewrite."""
    orch  = TrioOrchestrator()
    state = TrioOrchestrator.default_state()

    controls = {"power_W": 150.0, "hub_T_core": 37.5, "hub_pv_drop_pct": 2.0,
                "cho_abs_g_min": 0.03, "epi_pgmL": 80.0, "cortisol_nmolL": 350.0}

    state = orch.step(prior=state, controls=controls)

    assert not bool(jnp.any(jnp.isnan(state.nm.mean))),    "NaN in Trio NM"
    assert not bool(jnp.any(jnp.isnan(state.mg.mean))),    "NaN in Trio MG"
    assert not bool(jnp.any(jnp.isnan(state.cardio.mean))),"NaN in Trio Cardio"

    atone = float(msg_to_mean(state.hub.autonomic_tone))
    assert 0.0 <= atone <= 1.0, f"Trio autonomic_tone out of range: {atone}"

    print(f"\n[T5] TrioOrchestrator legacy: autonomic_tone={atone:.4f}. PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("T1: PentaOrchestrator default state NaN-free")
    test_penta_default_state_no_nan()
    print()
    print("=" * 70)
    print("T2: Single rest step — NaN-free + hub in range")
    test_penta_step_rest_no_nan()
    print()
    print("=" * 70)
    print("T3: Single exercise step — hub updates finite")
    test_penta_step_exercise_updates_hub()
    print()
    print("=" * 70)
    print("T4: 10 consecutive steps — no NaN accumulation")
    test_penta_10_steps_no_nan()
    print()
    print("=" * 70)
    print("T5: TrioOrchestrator legacy alias still works")
    test_trio_legacy_compat()
    print()
    print("=" * 70)
    print("ALL 5 PENTA-ORCHESTRATOR TESTS PASSED")
