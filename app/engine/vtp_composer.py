"""
VTP Composer — Vector de Teto Pessoal (Personal Ceiling Vector).

Master orchestrator: integrates the genetic, biochemical, and strategic layers
into a single personal ceiling vector with ranked intervention priorities.

Pipeline per system:
    1. Genetic ceiling  → compute_all_genetic_ceilings(genotype)
    2. Expressed ceiling → calculator breakdown (TEA = T_espécie × G × ∏ M_k^B)
    3. Adaptation gap   → calculate_adaptation_gap(TGI, TEA)
    4. Priority ranking → rank_interventions(system_gaps, structural_breakdown)

CHO exception: G_CHO = 1.0 by document specification (Doc1 §3.2); the genetic
modifier is not passed to calculate_cho_breakdown.

Structural exception: T_SPECIES["structural_weekly_tonnage_ref"] = None (not
declared numerically in the reference documents). The structural ceiling is None
and the adaptation gap is skipped; the structural breakdown is still computed and
forwarded to the priority engine for the Liebig safety rule evaluation.

Reference: Doc1 §2.x, Doc1 §6.2, Doc2 §4–5, Doc2 §16
"""

from __future__ import annotations

import math

from app.engine.calculators.aerobic_calculator import calculate_vo2max_breakdown
from app.engine.calculators.anabolic_calculator import calculate_mps_breakdown
from app.engine.calculators.cognitive_calculator import calculate_cognitive_breakdown
from app.engine.calculators.glycolytic_calculator import calculate_cho_breakdown
from app.engine.calculators.lipolytic_calculator import calculate_fatmax_breakdown
from app.engine.calculators.power_calculator import calculate_power_breakdown
from app.engine.calculators.structural_calculator import calculate_structural_breakdown
from app.engine.calculators.thermoregulation_calculator import calculate_thermoregulation_breakdown
from app.engine.constants import (
    ACTN3_TAU_FATIGUE_DAYS,
    ACE_CARDIAC_EFFICIENCY,
    BAYESIAN_PRIOR_MAPPINGS,
    MTHFR_METHYLATION_EFFICIENCY,
    MSTN_INHIBITION_FACTOR,
    PPARGC1A_OXPHOS_FACTOR,
)
from app.engine.modifiers.genetic import compute_all_genetic_ceilings
from app.engine.priority_engine import rank_interventions
from app.engine.trainability import calculate_adaptation_gap


def _apply_prior_transform(
    transform: str,
    raw: float | str,
    reference_value: float | None,
) -> float | None:
    """Apply a named transform to a raw biomarker/allele value → physical prior."""
    if transform == "direct":
        return float(raw)
    if transform == "reciprocal_normalised":
        val = float(raw)
        if val == 0.0:
            return None
        ref = reference_value if reference_value is not None else 1.0
        return ref / val
    if transform == "direct_mg_dL_to_mmol_L":
        return float(raw) / 18.0182
    if transform == "actn3_allele_to_tau":
        return ACTN3_TAU_FATIGUE_DAYS.get(str(raw))
    if transform == "ace_allele_to_efficiency":
        return ACE_CARDIAC_EFFICIENCY.get(str(raw))
    if transform == "mstn_allele_to_inhibition":
        return MSTN_INHIBITION_FACTOR.get(str(raw))
    if transform == "mthfr_allele_to_efficiency":
        return MTHFR_METHYLATION_EFFICIENCY.get(str(raw))
    if transform == "msf_to_phi_rad":
        ref = reference_value if reference_value is not None else 3.5
        return (float(raw) - ref) * (math.pi / 12.0)
    if transform == "pvt_to_alertness_rate":
        val = float(raw)
        if val == 0.0:
            return None
        ref = reference_value if reference_value is not None else 250.0
        return ref / val
    if transform == "diversity_to_eta":
        ref = reference_value if reference_value is not None else 3.5
        return min(1.0, float(raw) / ref)
    if transform == "ppargc1a_allele_to_oxphos":
        return PPARGC1A_OXPHOS_FACTOR.get(str(raw))
    return None


def compose_bayesian_priors(
    athlete_data: dict,
    genotype: dict[str, str],
) -> dict[str, float]:
    """
    Extract the vector of physical model priors (SystemParameters) for MHDS.

    Phase 3 no longer outputs a static personal ceiling score. Instead it emits
    a BayesianPriors dict: each key is a canonical MHDS state parameter name;
    each value is the athlete-specific initial constant derived from Phase 2
    inputs (biomarkers + SNPs).

    Source priority: athlete_data first, genotype dict second (for SNP fields).
    float("nan") is used as sentinel for absent biomarkers — this is the only
    JAX-safe missing-data representation; None would cause TypeError in jnp.array.

    Reference: NOVO_ENGINE_NUTRIVIOUS_1.1.txt §4 (MHDS Parameter Table)
    """
    priors: dict[str, float] = {}

    for source_field, mapping in BAYESIAN_PRIOR_MAPPINGS.items():
        prior_name: str = mapping["prior_name"]
        transform: str = mapping["transform"]
        reference_value: float | None = mapping.get("reference_value")

        raw = athlete_data.get(source_field)
        if raw is None:
            raw = genotype.get(source_field)
        if raw is None:
            priors[prior_name] = float("nan")
            continue

        result = _apply_prior_transform(transform, raw, reference_value)
        priors[prior_name] = result if result is not None else float("nan")

    return priors


def compose_vtp(
    athlete_data: dict,
    genotype: dict[str, str],
    user_weights: dict[str, float] | None = None,
) -> dict:
    """
    Compute the full Personal Ceiling Vector (VTP) for an athlete.

    Args:
        athlete_data: flat dict of biomarkers; recognised keys are the union of
                      all keys accepted by the 6 system calculators.
        genotype:     dict mapping SNP profile field names to genotype strings,
                      e.g. {"ACTN3": "RR", "ACE": "II", "MTHFR_C677T": "CT"}.
        user_weights: optional dict mapping system name → importance weight for
                      the priority engine (default 1.0 per system if omitted).

    Returns dict with:
        "personal_ceilings"     → {system: float | None}
        "modifier_breakdown"    → {system: breakdown_dict}
        "genetic_ceilings"      → {system: float} (TGI layer modifier scalars)
        "adaptation_gaps"       → {system: gap_dict}   (skipped when TGI undefined)
        "priority_interventions"→ list[dict] sorted by priority score
        "bayesian_priors"       → {prior_name: float | None} — MHDS SystemParameters
                                  vector; the new canonical Phase 3 output consumed
                                  by the Phase 4 ODE solvers and RBPF filter.
    """
    # ── 1. Genetic ceiling modifiers (TGI layer) ──────────────────────────────
    genetic_ceilings: dict[str, float] = compute_all_genetic_ceilings(genotype)

    # ── 2. System breakdowns (TEA layer) ─────────────────────────────────────
    vo2max_bd        = calculate_vo2max_breakdown(athlete_data, genetic_ceilings["vo2max"])
    cho_bd           = calculate_cho_breakdown(athlete_data)  # G_CHO = 1.0 always
    fatmax_bd        = calculate_fatmax_breakdown(athlete_data, genetic_ceilings["fatmax"])
    mps_bd           = calculate_mps_breakdown(athlete_data, genetic_ceilings["mps"])
    power_bd         = calculate_power_breakdown(athlete_data, genetic_ceilings["peak_power"])
    structural_bd    = calculate_structural_breakdown(athlete_data, genetic_ceilings["structural"])
    cognitive_bd     = calculate_cognitive_breakdown(athlete_data, genetic_ceilings["cognitive"])
    thermoreg_bd     = calculate_thermoregulation_breakdown(
                           athlete_data, genetic_ceilings["thermoregulatory"]
                       )

    breakdowns = {
        "vo2max":          vo2max_bd,
        "cho":             cho_bd,
        "fatmax":          fatmax_bd,
        "mps":             mps_bd,
        "peak_power":      power_bd,
        "structural":      structural_bd,
        "cognitive":       cognitive_bd,
        "thermoregulatory": thermoreg_bd,
    }

    personal_ceilings: dict[str, float | None] = {
        system: bd.get("ceiling") for system, bd in breakdowns.items()
    }

    # ── 3. Adaptation gaps ───────────────────────────────────────────────────
    # TGI = T_espécie × genetic_modifier for systems with declared T_espécie.
    # CHO: TGI = T_espécie × 1.0 (g_cho field in cho_bd; genetic_modifier = 1.0).
    # Structural: T_espécie = None → TGI undefined → gap skipped.
    adaptation_gaps: dict[str, dict] = {}

    for system, bd in breakdowns.items():
        t_especie = bd.get("t_especie")
        if t_especie is None:
            continue

        if system == "cho":
            g = bd.get("g_cho", 1.0)
        else:
            g = bd.get("genetic_modifier", genetic_ceilings.get(system, 1.0))

        tgi = t_especie * g
        tea = bd.get("ceiling")
        if tea is None:
            continue

        adaptation_gaps[system] = calculate_adaptation_gap(tgi, tea)

    # ── 4. Priority ranking ──────────────────────────────────────────────────
    system_gaps: dict[str, float] = {
        system: gap_dict["absolute_gap"]
        for system, gap_dict in adaptation_gaps.items()
    }

    priority_interventions = rank_interventions(
        system_gaps=system_gaps,
        structural_breakdown=structural_bd,
        user_weights=user_weights,
        athlete_data=athlete_data,
    )

    # ── 5. Bayesian priors vector (MHDS SystemParameters) ────────────────────
    # New Phase 3 output: maps each MHDS physical parameter to its athlete-
    # specific initial value. Consumed by Phase 4 ODE solvers and RBPF filter.
    # Reference: NOVO_ENGINE_NUTRIVIOUS_1.1.txt §4
    bayesian_priors: dict[str, float | None] = compose_bayesian_priors(
        athlete_data=athlete_data,
        genotype=genotype,
    )

    return {
        "personal_ceilings":      personal_ceilings,
        "modifier_breakdown":     breakdowns,
        "genetic_ceilings":       genetic_ceilings,
        "adaptation_gaps":        adaptation_gaps,
        "priority_interventions": priority_interventions,
        "bayesian_priors":        bayesian_priors,
    }
