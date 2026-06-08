"""
app/slices/neuroendocrine/envelope.py — Phase 3 Envelope: SAM × HPA × Somatotropic  V2.0

CLAUDE.md §3 — "Fase 3 é priors + restrições, nunca um modelo multiplicativo."

Safety constraints (Path A vocabulary — ZERO medical language)
--------------------------------------------------------------
Hard:
    Cortisol  ≤ 800  nmol/L  — allostatic load ceiling
    IGF-1     ≥  80  ng/mL   — anabolic floor

Chance (P(violation) ≤ α):
    P(Cortisol > 800 nmol/L) ≤ 0.01   — 99 % confidence below ceiling
    P(IGF-1  < 150 ng/mL)    ≤ 0.05   — anabolic readiness floor

Refer-out rules (Path A only):
    Sustained catabolic imbalance (Cortisol > 600 AND IGF-1 < 180):
        → pause performance recommendations, prompt professional consultation.

References
----------
    Duclos et al. (2003) Int J Sports Med 24:461–467  — cortisol allostatic ceiling
    Kraemer & Ratamess (2005) Sports Med 35:339–361   — IGF-1 performance floor
    CLAUDE.md §3 Golden Rule 5 — safety floors are inviolable hard constraints
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
    build_engine_priors,
)

# ── Thresholds ─────────────────────────────────────────────────────────────────

_CORTISOL_HARD_CEIL:   float = 800.0   # nmol/L
_IGF1_HARD_FLOOR:      float =  80.0   # ng/mL

_CORTISOL_CHANCE_CEIL: float = 800.0   # nmol/L  α=0.01
_IGF1_CHANCE_FLOOR:    float = 150.0   # ng/mL   α=0.05

_CORTISOL_REFER:       float = 600.0   # nmol/L  refer-out trigger
_IGF1_REFER:           float = 180.0   # ng/mL   refer-out trigger


# ── Hard constraints ───────────────────────────────────────────────────────────

def _hard_constraints() -> list[Constraint]:
    return [
        Constraint(
            name       = "cortisol_allostatic_ceiling",
            state_key  = "Hub_Cortisol_nmolL",
            op         = "<=",
            rhs        = _CORTISOL_HARD_CEIL,
            units      = "nmol/L",
            source_doi = "10.1055/s-2003-40498",
        ),
        Constraint(
            name       = "igf1_anabolic_floor",
            state_key  = "Hub_IGF1_ngmL",
            op         = ">=",
            rhs        = _IGF1_HARD_FLOOR,
            units      = "ng/mL",
            source_doi = "10.2165/00007256-200535050-00003",
        ),
    ]


# ── Chance constraints ─────────────────────────────────────────────────────────

def _chance_constraints() -> list[ChanceConstraint]:
    return [
        ChanceConstraint(
            name       = "cortisol_overload_chance",
            state_key  = "Hub_Cortisol_nmolL",
            op         = "<=",
            rhs        = _CORTISOL_CHANCE_CEIL,
            alpha      = 0.01,
            units      = "nmol/L",
            source_doi = "10.1055/s-2003-40498",
        ),
        ChanceConstraint(
            name       = "igf1_anabolic_readiness_chance",
            state_key  = "Hub_IGF1_ngmL",
            op         = ">=",
            rhs        = _IGF1_CHANCE_FLOOR,
            alpha      = 0.05,
            units      = "ng/mL",
            source_doi = "10.2165/00007256-200535050-00003",
        ),
    ]


# ── Refer-out rules (Path A only — no medical vocabulary) ─────────────────────

def _refer_out_rules() -> list[ReferOutRule]:
    _MSG = (
        "O teu racio de recuperacao sistemica (HPA/Somatotropico) indica estado "
        "catabolico sustentado. Risco elevado de overreaching nao-funcional. "
        "Recomendamos reduzir a carga de treino e priorizar recuperacao ativa "
        "antes de reiniciar prescricoes de performance."
    )
    return [
        ReferOutRule(
            name      = "sustained_catabolic_imbalance",
            state_key = "Hub_Cortisol_nmolL",
            op        = ">=",
            threshold = _CORTISOL_REFER,
            message   = _MSG,
        ),
    ]


# ── Envelope builder ───────────────────────────────────────────────────────────

def build_neuro_envelope(
    athlete_data: dict | None = None,
    posterior_theta: dict | None = None,
) -> Phase3Envelope:
    """
    Build the Phase3Envelope for the SAM × HPA × Somatotropic slice.

    Cold-start safe: uses population priors when athlete_data / posterior_theta
    are absent.  All thresholds are hard constraints derived from the HLD safety
    table; no value is computed by a multiplicative model.
    """
    athlete_data    = athlete_data    or {}
    posterior_theta = posterior_theta or {}

    neuro_priors: dict[str, float] = {
        "cortisol_morning_nmolL":              posterior_theta.get("Cortisol_peak",  420.0),
        "igf1_ngmL":                           posterior_theta.get("IGF1_baseline",  250.0),
        "GC_sensitivity":                      posterior_theta.get("GC_sensitivity",   1.0),
        "IGF1_conv_rate":                      posterior_theta.get("IGF1_conv_rate",   6.0),
        "Metabolic_Flexibility_Threshold_mgdL":posterior_theta.get("MFT",             70.0),
        "epinephrine_rest_pgmL":               posterior_theta.get("Epi_rest",         50.0),
        "norepinephrine_rest_pgmL":            posterior_theta.get("NE_rest",          300.0),
    }

    base_priors = build_engine_priors(athlete_data, genotype={})
    merged      = {**base_priors, **neuro_priors}

    return Phase3Envelope(
        bayesian_priors    = merged,
        hard_constraints   = _hard_constraints(),
        chance_constraints = _chance_constraints(),
        refer_out_rules    = _refer_out_rules(),
    )
