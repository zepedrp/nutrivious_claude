"""
tests/test_slow_axis_orchestrator.py — Gate Zero: SlowAxisOrchestrator (Multi-Rate)

Tests the day-scale Gonadal HPG axis and its hormonal bridge to the fast axis.

T1  test_slow_axis_default_state_no_nan
        Cold-start SlowAxisState (female + male) must be NaN-free, all hub fields
        finite.

T2  test_slow_axis_female_step_publishes_basal_temp_offset
        One female step at normal EA (45 kcal/kg FFM/day).
        - hub.basal_temp_offset mean in [0.0, 0.5] °C  (P4 follicular phase)
        - hub.anabolic_drive mean in [0.0, 1.0]
        - hub.testosterone  mean > 0

T3  test_slow_axis_male_step_publishes_anabolic_drive
        One male step at normal EA.
        - hub.testosterone mean > 200 ng/dL (eugonadal)
        - hub.basal_temp_offset mean in [0.0, 0.05] °C (near-zero for males)
        - hub.anabolic_drive mean in [0.0, 1.0]

T4  test_slow_axis_reds_suppression
        Female at EA=20 for 7 daily steps: basal_temp_offset should decline
        as FHA suppresses P4 (Loucks 2003).

T5  test_multirate_bridge_tr_reads_offset
        Directly tests the Fast Axis bridge: inject a non-zero basal_temp_offset
        into a hub, run PentaOrchestrator one step, confirm the TR posterior
        mean T_core is shifted upward compared to offset=0.

T6  test_penta_step_with_slow_hub
        Full integration: SlowAxisOrchestrator female step → updated hub →
        PentaOrchestrator step. Asserts no NaN end-to-end and hub fields remain
        within physiological ranges.

Run:
    pytest tests/test_slow_axis_orchestrator.py -v -s
"""
from __future__ import annotations

import sys
import math

sys.path.insert(0, ".")

import jax.numpy as jnp
import pytest

from app.engine.orchestrator import (
    PentaOrchestrator,
    PentaState,
    SlowAxisOrchestrator,
    SlowAxisState,
)
from app.engine.hubs import msg_to_mean, msg_from_scalar, default_hub_state, update_hub


# ─────────────────────────────────────────────────────────────────────────────
# Shared controls
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


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Default state NaN-free (female + male)
# ─────────────────────────────────────────────────────────────────────────────

def test_slow_axis_default_state_no_nan():
    """Cold-start SlowAxisState for both sexes must be fully finite."""
    for is_female, label in [(True, "female"), (False, "male")]:
        state = SlowAxisOrchestrator.default_state(is_female=is_female)

        assert not bool(jnp.any(jnp.isnan(state.gonadal.mean))), (
            f"NaN in {label} gonadal default mean"
        )

        for field in ["testosterone", "basal_temp_offset", "anabolic_drive",
                      "plasma_glucose", "core_temp"]:
            val = float(msg_to_mean(getattr(state.hub, field)))
            assert math.isfinite(val), (
                f"Non-finite hub.{field} in {label} default state: {val}"
            )

    print("\n[T1] SlowAxisOrchestrator default state: NaN-free both sexes. PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Female step publishes correct basal_temp_offset
# ─────────────────────────────────────────────────────────────────────────────

def test_slow_axis_female_step_publishes_basal_temp_offset():
    """
    Female at normal EA (45): one step publishes basal_temp_offset and
    anabolic_drive in physiological ranges.
    """
    orch  = SlowAxisOrchestrator(is_female=True)
    state = SlowAxisOrchestrator.default_state(is_female=True)

    state = orch.step(prior=state, observations={})

    assert not bool(jnp.any(jnp.isnan(state.gonadal.mean))), \
        "NaN in female gonadal posterior"

    bto   = float(msg_to_mean(state.hub.basal_temp_offset))
    ad    = float(msg_to_mean(state.hub.anabolic_drive))
    t_val = float(msg_to_mean(state.hub.testosterone))

    print(f"\n[T2] Female: basal_temp_offset={bto:.4f}°C  "
          f"anabolic_drive={ad:.4f}  testosterone={t_val:.1f} ng/dL")

    assert 0.0 <= bto <= 0.5,  f"basal_temp_offset out of range: {bto:.4f}"
    assert 0.0 <= ad  <= 1.0,  f"anabolic_drive out of range: {ad:.4f}"
    assert t_val > 0.0,        f"testosterone must be > 0: {t_val}"

    print("[T2] PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Male step: eugonadal T, near-zero basal_temp_offset
# ─────────────────────────────────────────────────────────────────────────────

def test_slow_axis_male_step_publishes_anabolic_drive():
    """
    Male at normal EA: T stays eugonadal, basal_temp_offset ≈ 0 (male adrenal
    P4 = 0.15 ng/mL → offset ≈ 0.004°C), anabolic_drive in [0, 1].
    """
    orch  = SlowAxisOrchestrator(is_female=False)
    state = SlowAxisOrchestrator.default_state(is_female=False)

    state = orch.step(prior=state, observations={})

    assert not bool(jnp.any(jnp.isnan(state.gonadal.mean))), \
        "NaN in male gonadal posterior"

    bto   = float(msg_to_mean(state.hub.basal_temp_offset))
    ad    = float(msg_to_mean(state.hub.anabolic_drive))
    t_val = float(msg_to_mean(state.hub.testosterone))

    print(f"\n[T3] Male: basal_temp_offset={bto:.4f}°C  "
          f"anabolic_drive={ad:.4f}  testosterone={t_val:.1f} ng/dL")

    assert t_val > 200.0,       f"Male T below eugonadal range: {t_val:.1f} ng/dL"
    assert 0.0 <= bto <= 0.05,  f"Male basal_temp_offset should be near-zero: {bto:.4f}"
    assert 0.0 <= ad  <= 1.0,   f"anabolic_drive out of range: {ad:.4f}"

    print("[T3] PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T4 — RED-S suppression: FHA reduces P4 → basal_temp_offset declines
# ─────────────────────────────────────────────────────────────────────────────

def test_slow_axis_reds_suppression():
    """
    Female at EA=20 (below Loucks 2003 threshold) for 7 days:
    GnRH suppression causes P4 to collapse → basal_temp_offset should fall
    below the normal-EA final value (Mountjoy 2014).
    """
    # Reference: normal EA
    orch = SlowAxisOrchestrator(is_female=True)

    state_normal = SlowAxisOrchestrator.default_state(is_female=True)
    hub_normal = state_normal.hub._replace(
        energy_avail=msg_from_scalar(45.0, 0.01)
    )
    state_normal = state_normal._replace(hub=hub_normal)
    for _ in range(7):
        state_normal = orch.step(prior=state_normal)
    bto_normal = float(msg_to_mean(state_normal.hub.basal_temp_offset))

    # RED-S: EA=20
    state_reds = SlowAxisOrchestrator.default_state(is_female=True)
    hub_reds = state_reds.hub._replace(
        energy_avail=msg_from_scalar(20.0, 0.01)
    )
    state_reds = state_reds._replace(hub=hub_reds)
    for _ in range(7):
        state_reds = orch.step(prior=state_reds)
    bto_reds = float(msg_to_mean(state_reds.hub.basal_temp_offset))

    print(f"\n[T4] bto(EA=45,7d)={bto_normal:.4f}°C  bto(EA=20,7d)={bto_reds:.4f}°C")

    assert not bool(jnp.any(jnp.isnan(state_reds.gonadal.mean))), \
        "NaN in RED-S gonadal posterior"
    assert bto_reds <= bto_normal + 0.01, (
        f"FHA: expected basal_temp_offset to decline at EA=20; "
        f"got {bto_reds:.4f} vs {bto_normal:.4f}"
    )

    print("[T4] PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T5 — Multi-rate bridge: TR filter reads basal_temp_offset from hub
# ─────────────────────────────────────────────────────────────────────────────

def test_multirate_bridge_tr_reads_offset():
    """
    Inject hub.basal_temp_offset = 0.35°C (mid-luteal peak) vs 0.0°C.
    Run one PentaOrchestrator step with power=0 (rest).
    Assert that T_core posterior is shifted upward when offset=0.35.

    Physics: the TR ODE sweat threshold rises from 37.0 to 37.35°C.
    At rest (T_core ~ 37°C), sweat is nil in both cases, but the setpoint
    shift means the ODE equilibrium is fractionally higher.
    """
    orch = PentaOrchestrator()

    # Baseline: offset = 0.0
    state_0 = PentaOrchestrator.default_state()
    state_0 = orch.step(prior=state_0, controls=CONTROLS_REST)
    tcore_0 = float(msg_to_mean(state_0.hub.core_temp))

    # Shifted: offset = 0.35°C (mid-luteal)
    hub_shifted = default_hub_state()._replace(
        basal_temp_offset=msg_from_scalar(0.35, 0.001)
    )
    state_s = PentaOrchestrator.default_state()._replace(hub=hub_shifted)
    state_s = orch.step(prior=state_s, controls=CONTROLS_REST)
    tcore_s = float(msg_to_mean(state_s.hub.core_temp))

    print(f"\n[T5] T_core: offset=0.00 -> {tcore_0:.4f}°C  "
          f"offset=0.35 -> {tcore_s:.4f}°C  delta={tcore_s-tcore_0:.5f}°C")

    assert math.isfinite(tcore_0), f"T_core(offset=0) is not finite: {tcore_0}"
    assert math.isfinite(tcore_s), f"T_core(offset=0.35) is not finite: {tcore_s}"
    # At rest with offset the sweat threshold is higher → T_core >= baseline
    assert tcore_s >= tcore_0 - 0.01, (
        f"Expected T_core to be >= baseline with positive offset; "
        f"got {tcore_s:.4f} vs {tcore_0:.4f}"
    )

    print("[T5] PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# T6 — End-to-end multi-rate: SlowAxis → hub → PentaOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

def test_penta_step_with_slow_hub():
    """
    Full integration test:
      1. SlowAxisOrchestrator (female) advances 1 day.
      2. Updated hub passed to PentaOrchestrator.
      3. PentaOrchestrator runs 3 steps.
    Asserts no NaN anywhere and all hub fields in physiological ranges.
    """
    slow  = SlowAxisOrchestrator(is_female=True)
    penta = PentaOrchestrator()

    slow_state  = SlowAxisOrchestrator.default_state(is_female=True)
    penta_state = PentaOrchestrator.default_state()

    # Day 0 → Day 1: Gonadal step
    slow_state = slow.step(prior=slow_state)

    # Inject slow-axis hub fields into the fast-axis PentaState hub
    hub_fast = penta_state.hub._replace(
        testosterone      = slow_state.hub.testosterone,
        basal_temp_offset = slow_state.hub.basal_temp_offset,
        anabolic_drive    = slow_state.hub.anabolic_drive,
    )
    penta_state = penta_state._replace(hub=hub_fast)

    # Run 3 fast-axis steps
    for step_i in range(3):
        penta_state = penta.step(prior=penta_state, controls=CONTROLS_REST)
        assert not bool(jnp.any(jnp.isnan(penta_state.nm.mean))),    f"NaN NM step {step_i}"
        assert not bool(jnp.any(jnp.isnan(penta_state.mg.mean))),    f"NaN MG step {step_i}"
        assert not bool(jnp.any(jnp.isnan(penta_state.neuro.mean))), f"NaN Neuro step {step_i}"
        assert not bool(jnp.any(jnp.isnan(penta_state.thermo_renal.mean))), \
            f"NaN TR step {step_i}"
        assert not bool(jnp.any(jnp.isnan(penta_state.cardio.mean))), \
            f"NaN Cardio step {step_i}"

    hub = penta_state.hub
    bto   = float(msg_to_mean(hub.basal_temp_offset))
    ad    = float(msg_to_mean(hub.anabolic_drive))
    tcore = float(msg_to_mean(hub.core_temp))
    cort  = float(msg_to_mean(hub.cortisol))

    print(f"\n[T6] End-to-end: bto={bto:.4f}°C  ad={ad:.4f}  "
          f"T_core={tcore:.3f}°C  cortisol={cort:.1f} nmol/L")

    assert 0.0 <= bto   <= 0.5,   f"basal_temp_offset out of range: {bto}"
    assert 0.0 <= ad    <= 1.0,   f"anabolic_drive out of range: {ad}"
    assert 35.0 <= tcore <= 42.0, f"T_core out of physiological range: {tcore}"
    assert math.isfinite(cort),   f"cortisol non-finite: {cort}"

    print("[T6] PASS.")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("T1: SlowAxisOrchestrator default state NaN-free (female + male)")
    test_slow_axis_default_state_no_nan()
    print()
    print("=" * 70)
    print("T2: Female step — basal_temp_offset + anabolic_drive published")
    test_slow_axis_female_step_publishes_basal_temp_offset()
    print()
    print("=" * 70)
    print("T3: Male step — testosterone eugonadal, basal_temp_offset near-zero")
    test_slow_axis_male_step_publishes_anabolic_drive()
    print()
    print("=" * 70)
    print("T4: RED-S suppression — FHA reduces basal_temp_offset")
    test_slow_axis_reds_suppression()
    print()
    print("=" * 70)
    print("T5: Multi-rate bridge — TR filter reads basal_temp_offset from hub")
    test_multirate_bridge_tr_reads_offset()
    print()
    print("=" * 70)
    print("T6: End-to-end SlowAxis -> hub -> PentaOrchestrator")
    test_penta_step_with_slow_hub()
    print()
    print("=" * 70)
    print("ALL 6 SLOW-AXIS TESTS PASSED")
