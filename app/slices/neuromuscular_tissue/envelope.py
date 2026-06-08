"""
app/slices/neuromuscular_tissue/envelope.py

Phase 3 Envelope V4.0 -- Neuromuscular Tissue Slice (L3)

Generates the Phase3Envelope (priors + constraints + refer-out rules) for
the intra-session neuromuscular V4.0 state.

CLAUDE.md golden rules enforced:
  * Path A language -- no medical vocabulary in any user-facing message string.
  * Genetics = weak prior shift only (<=0.5*sigma).
  * Phase 3 = priors + constraints, never a multiplicative model.

Hub variables monitored
------------------------
  Hub_Peripheral_Fatigue_au  in [0, 10]
    = alpha_ATP * max(0, ATP_rest - ATP) + alpha_RyR1 * RyR1_Damage

  Hub_Muscle_Glycogen_mmolkg  in [0+]
    = Muscle_Glycogen_mmol  (raw state x[5])

Safety thresholds -- Peripheral Fatigue
-----------------------------------------
Hard ceiling:  Hub_Peripheral_Fatigue <= 10.0
  Maximum possible value (ATP fully depleted and RyR1 saturated).
  At this level contractile output is negligible and joint stability is
  compromised by compensatory motor patterns.

Refer-out threshold: Fatigue > 8.0
  Severe peripheral fatigue state -- acute risk of compensatory injury via
  kinematic substitution (Falla & Farina 2007 Clin Neurophysiol).

Chance constraint:  Fatigue <= 7.0  (alpha=0.10)
  Comfortable safety margin for 90% of sessions.

Safety thresholds -- Muscle Glycogen
--------------------------------------
Chance constraint:  Hub_Muscle_Glycogen >= 15.0  (alpha=0.10)
  Bonk threshold: T2 gate closes at ~15 mmol/kg; below this level
  fast-twitch fiber function is substantially impaired.

Refer-out threshold: Glycogen < 10.0
  Deep bonk: near-complete T2 suppression via glycogen gate.
  Path A message: no medical vocabulary.

References
----------
  Falla D., Farina D. (2007) Clin Neurophysiol 118:1368   [compensatory patterns]
  Westerblad H. et al. (2002) J Physiol 540:111            [peripheral fatigue]
  Bergstrom J. et al. (1967) Acta Physiol Scand 71:140    [muscle glycogen bonk]
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
    build_engine_priors,
)

# -- Safety thresholds: Peripheral Fatigue -------------------------------------

_FATIGUE_HARD_CEILING:   float = 10.0  # absolute physical maximum
_FATIGUE_REFER_OUT:      float = 8.0   # refer-out trigger
_FATIGUE_CHANCE_CEILING: float = 7.0   # soft ceiling (alpha=0.10)

# -- Safety thresholds: Muscle Glycogen ----------------------------------------

_GLYCOGEN_CHANCE_FLOOR:  float = 15.0  # mmol/kg soft floor (alpha=0.10)
_GLYCOGEN_REFER_OUT:     float = 10.0  # mmol/kg refer-out trigger


# -- Constraint builders -------------------------------------------------------

def _hard_constraints_nm_v4() -> list[Constraint]:
    return [
        Constraint(
            name       = "peripheral_fatigue_hard_ceiling",
            state_key  = "Hub_Peripheral_Fatigue_au",
            op         = "<=",
            rhs        = _FATIGUE_HARD_CEILING,
            units      = "au [0-10]",
            source_doi = "10.1113/jphysiol.2001.013154",
        ),
    ]


def _chance_constraints_nm_v4() -> list[ChanceConstraint]:
    return [
        ChanceConstraint(
            name       = "peripheral_fatigue_soft_ceiling",
            state_key  = "Hub_Peripheral_Fatigue_au",
            op         = "<=",
            rhs        = _FATIGUE_CHANCE_CEILING,
            alpha      = 0.10,
            units      = "au [0-10]",
            source_doi = "10.1113/jphysiol.2001.013154",
        ),
        ChanceConstraint(
            name       = "muscle_glycogen_soft_floor",
            state_key  = "Hub_Muscle_Glycogen_mmolkg",
            op         = ">=",
            rhs        = _GLYCOGEN_CHANCE_FLOOR,
            alpha      = 0.10,
            units      = "mmol/kg dm",
            source_doi = "10.1111/j.1748-1716.1967.tb03581.x",
        ),
    ]


def _refer_out_rules_nm_v4() -> list[ReferOutRule]:
    return [
        ReferOutRule(
            name      = "severe_contractile_fatigue",
            state_key = "Hub_Peripheral_Fatigue_au",
            op        = ">=",
            threshold = _FATIGUE_REFER_OUT,
            message   = (
                "Incapacidade contractil neuromuscular severa detetada. "
                "Risco agudo de lesao articular por compensacao."
            ),
        ),
        ReferOutRule(
            name      = "deep_glycogen_depletion",
            state_key = "Hub_Muscle_Glycogen_mmolkg",
            op        = "<=",
            threshold = _GLYCOGEN_REFER_OUT,
            message   = (
                "Reservas de energia muscular locais em nivel critico. "
                "Capacidade de esforco intenso severamente comprometida. "
                "Recomenda-se paragem e reposicao de hidratos de carbono "
                "antes de retomar atividade de alta intensidade. "
                "(Path A: continuar apenas exercicio de baixa intensidade.)"
            ),
        ),
    ]


# -- Public API ----------------------------------------------------------------

def build_neuromuscular_v4_envelope(
    athlete_data:    dict,
    genotype:        dict[str, str],
    phase1_ceilings: dict | None = None,
) -> Phase3Envelope:
    """
    Build the Phase 3 envelope for the Neuromuscular Tissue V4.0 slice.

    Parameters
    ----------
    athlete_data    : dict -- Phase 2 biomarker dict.
    genotype        : dict -- SNP genotype dict (weak prior shifts only).
    phase1_ceilings : dict | None -- species-level ceilings from Phase 1.

    Returns
    -------
    Phase3Envelope -- priors + hard + chance + refer-out for MPC consumption.
    """
    bayesian_priors = build_engine_priors(athlete_data, genotype, phase1_ceilings)

    # ACTN3 -> P_th_2 weak prior shift (fast-twitch fiber threshold)
    # R allele: slightly lower P_th_2 (more T2 at same power) <= 0.5*sigma
    actn3 = genotype.get("actn3_rs1815739", "")
    if actn3 == "RR":
        bayesian_priors["P_th_2_prior_W"] = 237.5   # -0.5*sigma (sigma~25W)
    elif actn3 == "XX":
        bayesian_priors["P_th_2_prior_W"] = 262.5   # +0.5*sigma
    else:
        bayesian_priors["P_th_2_prior_W"] = 250.0

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = _hard_constraints_nm_v4(),
        chance_constraints = _chance_constraints_nm_v4(),
        refer_out_rules    = _refer_out_rules_nm_v4(),
    )
