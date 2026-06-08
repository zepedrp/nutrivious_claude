"""
app/slices/metabolic_glucose/envelope.py -- Phase 3 Envelope V2.0

CLAUDE.md §3: "Fase 3 e' priors + restricoes, nunca um modelo multiplicativo."

Safety floors
─────────────
Hypoglycaemia (plasma glucose):
    < 55 mg/dL  -> hard constraint + refer-out (severe)
    < 70 mg/dL  -> chance constraint  P(G < 70) <= 0.05

Muscle glycogen / bonking:
    MG < 15% MG_max -> chance constraint  P(MG < 60 g) <= 0.05

Postprandial hyperglycaemia:
    G > 180 mg/dL  -> chance constraint  P(G > 180) <= 0.10
    G > 250 mg/dL  -> hard ceiling

Refer-out rule (Path A language -- no medical vocabulary):
    "Os teus padroes de glicose indicam resistencia metabolica periferica.
     Possivel resposta a carga alostatica (stress cronico)."
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
    build_engine_priors,
)
from app.slices.metabolic_glucose.ode import DEFAULT_PARAMS


# -- Safety thresholds
_HYPO_HARD_MGDL:    float = 55.0
_HYPO_CHANCE_MGDL:  float = 70.0
_HYPER_CHANCE_MGDL: float = 180.0
_HYPER_HARD_MGDL:   float = 250.0
_MG_BONK_FLOOR_G:   float = DEFAULT_PARAMS.MG_max * 0.15   # 60 g


def _hard_constraints() -> list[Constraint]:
    return [
        Constraint(
            name       = "plasma_glucose_severe_hypo",
            state_key  = "Plasma_Glucose_mgdL",
            op         = ">=",
            rhs        = _HYPO_HARD_MGDL,
            units      = "mg/dL",
            source_doi = "10.1210/jc.2012-3931",
        ),
        Constraint(
            name       = "plasma_glucose_critical_hyper",
            state_key  = "Plasma_Glucose_mgdL",
            op         = "<=",
            rhs        = _HYPER_HARD_MGDL,
            units      = "mg/dL",
            source_doi = "10.2337/dc22-S006",
        ),
    ]


def _chance_constraints() -> list[ChanceConstraint]:
    return [
        ChanceConstraint(
            name       = "glucose_alert_hypo",
            state_key  = "Plasma_Glucose_mgdL",
            op         = ">=",
            rhs        = _HYPO_CHANCE_MGDL,
            alpha      = 0.05,
            units      = "mg/dL",
            source_doi = "10.1210/jc.2012-3931",
        ),
        ChanceConstraint(
            name       = "postprandial_hyper_target",
            state_key  = "Plasma_Glucose_mgdL",
            op         = "<=",
            rhs        = _HYPER_CHANCE_MGDL,
            alpha      = 0.10,
            units      = "mg/dL",
            source_doi = "10.2337/dc22-S006",
        ),
        ChanceConstraint(
            name       = "muscle_glycogen_bonk_floor",
            state_key  = "Muscle_Glycogen_g",
            op         = ">=",
            rhs        = _MG_BONK_FLOOR_G,
            alpha      = 0.05,
            units      = "g",
            source_doi = "10.1123/ijsnem.2011-0074",
        ),
    ]


def _refer_out_rules() -> list[ReferOutRule]:
    return [
        ReferOutRule(
            name      = "glucose_severe_hypo_referout",
            state_key = "Plasma_Glucose_mgdL",
            op        = "<=",
            threshold = _HYPO_HARD_MGDL,
            message   = (
                "A tua leitura de glicose esta abaixo do teu intervalo habitual. "
                "Estamos a pausar todas as recomendacoes de performance e nutricao. "
                "Recomendamos que confirmes com um profissional de saude qualificado "
                "antes de retomares a atividade."
            ),
        ),
        ReferOutRule(
            name      = "peripheral_metabolic_resistance_pattern",
            state_key = "Plasma_Glucose_mgdL",
            op        = ">=",
            threshold = _HYPER_CHANCE_MGDL,
            message   = (
                "Os teus padroes de glicose indicam resistencia metabolica periferica. "
                "Possivel resposta a carga alostatica (stress cronico). "
                "Estamos a ajustar as recomendacoes de nutricao e treino. "
                "Uma consulta com um profissional de saude e recomendada."
            ),
        ),
    ]


def build_metabolic_glucose_envelope(
    athlete_data: dict,
    genotype:     dict[str, str],
    phase1_ceilings: dict | None = None,
) -> Phase3Envelope:
    """
    Build the Phase3Envelope for the Metabolic Glucose V2.0 slice.

    Parameters
    ----------
    athlete_data    : Phase 2 biomarker dict (glucose_fasting_mg_dL, homa_ir)
    genotype        : SNP dict (genetics as weak priors, CLAUDE.md §3.2)
    phase1_ceilings : species-level ceilings from Phase 1

    Returns
    -------
    Phase3Envelope -- priors + hard + chance + refer-out for MPC consumption
    """
    priors = build_engine_priors(athlete_data, genotype, phase1_ceilings)

    fg = athlete_data.get("glucose_fasting_mg_dL")
    if fg is not None:
        priors["G_fasting_prior_mgdL"] = float(fg)

    homa = athlete_data.get("homa_ir")
    if homa is not None:
        priors["IS_0_prior"] = DEFAULT_PARAMS.IS_0 / max(float(homa), 1.0)

    return Phase3Envelope(
        bayesian_priors    = priors,
        hard_constraints   = _hard_constraints(),
        chance_constraints = _chance_constraints(),
        refer_out_rules    = _refer_out_rules(),
    )
