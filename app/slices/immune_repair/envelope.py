"""
app/slices/immune_repair/envelope.py -- L3 Phase3Envelope: Immune Repair

Path A language ONLY. Zero medical vocabulary, zero disease, zero diagnosis.
All constraints are expressed as performance / recovery readiness limits.

Hard constraints (MPC never crosses these):
    Muscle_Damage_au < 0.85      (structural integrity floor)

Chance constraints (probabilistic; alpha = target exceedance probability):
    Muscle_Damage_au < 0.65   alpha=5%    (readiness gate)

Refer-out rule (generated text -- Path A, no medical language):
    "Os teus biomarcadores de prontidao sistemica indicam supressao prolongada
     da capacidade de recuperacao. Risco elevado de dano estrutural acumulado
     (overreaching nao-funcional)."

References
----------
    CLAUDE.md Section 1 -- Path A language mandate
    IOC Consensus (2018) -- overtraining / overreaching definitions
    Meeusen et al. (2013) Eur J Sport Sci 13:1              -- NFO criteria
"""
from __future__ import annotations

from typing import NamedTuple


# ── Constraint containers ──────────────────────────────────────────────────────

class ImmuneHardConstraint(NamedTuple):
    name:       str
    state_idx:  int
    upper:      float      # state must be < upper (or > lower if lower is set)
    lower:      float = float("-inf")


class ImmuneChanceConstraint(NamedTuple):
    name:       str
    state_idx:  int
    upper:      float
    alpha:      float      # allowed exceedance probability


class ImmuneReferOutRule(NamedTuple):
    key:        str
    condition:  str        # plain description; evaluation logic in orchestrator
    message:    str        # Path A user-facing message (no medical vocabulary)


# ── Phase 3 envelope ──────────────────────────────────────────────────────────

class ImmunePhase3Envelope(NamedTuple):
    hard_constraints:   tuple
    chance_constraints: tuple
    refer_out_rules:    tuple


def build_immune_envelope() -> ImmunePhase3Envelope:
    hard = (
        ImmuneHardConstraint(
            name      = "damage_toxic_ceiling",
            state_idx = 0,     # Muscle_Damage_au
            upper     = 0.85,
        ),
    )

    chance = (
        ImmuneChanceConstraint(
            name      = "damage_readiness_gate",
            state_idx = 0,     # Muscle_Damage_au
            upper     = 0.65,
            alpha     = 0.05,
        ),
    )

    refer_out = (
        ImmuneReferOutRule(
            key       = "nfo_risk",
            condition = "Muscle_Damage_au > 0.65 AND Macrophage_M1 persistently elevated AND M2 suppressed",
            message   = (
                "Os teus biomarcadores de prontidao sistemica indicam supressao "
                "prolongada da capacidade de recuperacao. Risco elevado de dano "
                "estrutural acumulado (overreaching nao-funcional)."
            ),
        ),
    )

    return ImmunePhase3Envelope(
        hard_constraints   = hard,
        chance_constraints = chance,
        refer_out_rules    = refer_out,
    )


DEFAULT_IMMUNE_ENVELOPE = build_immune_envelope()


def check_hard_constraints(state_mean, envelope: ImmunePhase3Envelope = DEFAULT_IMMUNE_ENVELOPE) -> dict:
    """
    Returns dict of {constraint_name: bool (True = violated)}.
    Used by orchestrator Fail-Loud gate.
    """
    results = {}
    for c in envelope.hard_constraints:
        val = float(state_mean[c.state_idx])
        results[c.name] = val >= c.upper
    return results
