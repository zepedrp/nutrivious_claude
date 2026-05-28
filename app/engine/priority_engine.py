"""
Priority engine — ranks nutritional/clinical interventions by expected ceiling impact.

Priority score (Doc1 §6.2):
    score(j) = Σ_i [ w_i × D_ji × GA_i × (1 / T_weeks_j) × L_j ]

Where:
    j         = intervention key (row in ELASTICITY_MATRIX)
    i         = physiological system (column in ELASTICITY_MATRIX)
    w_i       = user-declared importance weight for system i (default 1.0)
    D_ji      = ELASTICITY_MATRIX[j][i] — fractional elasticity of ceiling i to intervention j
    GA_i      = absolute adaptation gap for system i (TGI − TEA)
    T_weeks_j = TIME_TO_EFFECT_WEEKS[j]
    L_j       = TRAINABILITY_LEVERAGE[j]

Triage order (applied before economic score):
    1. Clinical red-flag overrides (CLINICAL_RED_FLAGS): biomarkers that breach
       safety thresholds force their intervention to the absolute top with
       score=∞ and forced_first=True. Multiple red flags stack in detection order.
    2. Structural Liebig safety rule (Doc1 §3.6 + Doc2 §12.1): if any structural
       biochemical modifier < STRUCTURAL_LIEBIG_THRESHOLD (0.85), its corrective
       intervention is forced above the economic ranking (but below any red flags).
    3. Economic score ranking for all remaining interventions.

Reference: Doc1 §6.2, Doc2 §16
"""

from __future__ import annotations

from app.engine.constants import (
    CLINICAL_RED_FLAGS,
    ELASTICITY_MATRIX,
    LIEBIG_INTERVENTION_MAP,
    TIME_TO_EFFECT_WEEKS,
    TRAINABILITY_LEVERAGE,
)

STRUCTURAL_LIEBIG_THRESHOLD: float = 0.85

_DEFAULT_WEIGHT: float = 1.0


def calculate_priority_score(
    intervention: str,
    system_gaps: dict[str, float],
    user_weights: dict[str, float] | None = None,
) -> float:
    """
    Compute the priority score for one intervention across all systems.

    Args:
        intervention: key in ELASTICITY_MATRIX (e.g. "iron_repletion")
        system_gaps:  dict mapping system name → absolute adaptation gap (GA_i = TGI − TEA).
                      Systems absent from this dict contribute 0 to the score.
        user_weights: optional dict mapping system name → importance weight (default 1.0).

    Returns:
        Scalar priority score ≥ 0. Returns 0.0 for unknown intervention keys.
    """
    elasticity_row = ELASTICITY_MATRIX.get(intervention)
    if elasticity_row is None:
        return 0.0

    t_weeks: float = float(TIME_TO_EFFECT_WEEKS.get(intervention, 1))
    leverage: float = float(TRAINABILITY_LEVERAGE.get(intervention, 1.0))
    weights: dict[str, float] = user_weights or {}

    score: float = 0.0
    for system, gap in system_gaps.items():
        elasticity: float = elasticity_row.get(system, 0.0)
        weight: float = weights.get(system, _DEFAULT_WEIGHT)
        score += weight * elasticity * gap * (1.0 / t_weeks) * leverage

    return score


def rank_interventions(
    system_gaps: dict[str, float],
    structural_breakdown: dict | None = None,
    user_weights: dict[str, float] | None = None,
    athlete_data: dict | None = None,
) -> list[dict]:
    """
    Rank all known interventions by priority score, applying triage overrides.

    Triage order:
        1. Clinical red flags (athlete_data biomarkers breaching safety thresholds)
           → forced to the top with score = deviation × urgency_weight, sorted by
           severity; multiple red flags stack in descending severity order.
        2. Structural Liebig bottleneck (modifier below STRUCTURAL_LIEBIG_THRESHOLD)
           → forced above the economic ranking, below any red flags.
        3. Economic score ranking for all remaining interventions.

    Args:
        system_gaps:          dict mapping system → absolute adaptation gap (TGI − TEA).
        structural_breakdown: output dict from calculate_structural_breakdown (or None).
        user_weights:         optional system-level importance weights.
        athlete_data:         flat biomarker dict (SI units) used for clinical triage.

    Returns:
        List of dicts sorted by priority, each containing:
            "intervention" → intervention key string
            "score"        → numeric priority score (inf for red-flag items)
            "forced_first" → True if promoted by triage (red flag or Liebig rule)
    """
    scored: list[dict] = [
        {
            "intervention": iv,
            "score": calculate_priority_score(iv, system_gaps, user_weights),
            "forced_first": False,
        }
        for iv in ELASTICITY_MATRIX
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)

    # ── Liebig structural safety override ─────────────────────────────────────
    if structural_breakdown is not None:
        modifier_detail: dict = structural_breakdown.get("modifier_detail", {})
        active: dict[str, float] = {k: v for k, v in modifier_detail.items() if v is not None}
        if active:
            bottleneck_key = min(active, key=lambda k: active[k])
            if active[bottleneck_key] < STRUCTURAL_LIEBIG_THRESHOLD:
                forced_iv = LIEBIG_INTERVENTION_MAP.get(bottleneck_key)
                if forced_iv is not None:
                    scored = [e for e in scored if e["intervention"] != forced_iv]
                    scored.insert(0, {
                        "intervention": forced_iv,
                        "score": calculate_priority_score(forced_iv, system_gaps, user_weights),
                        "forced_first": True,
                    })

    # ── Clinical red-flag triage (absolute top priority) ──────────────────────
    if athlete_data is not None:
        red_flags: list[dict] = []
        for intervention, rule in CLINICAL_RED_FLAGS.items():
            value = athlete_data.get(rule["biomarker"])
            if value is None:
                continue
            op = rule["operator"]
            threshold = rule["threshold"]
            triggered = (op == "gt" and value > threshold) or (op == "lt" and value < threshold)
            if triggered:
                deviation = (value - threshold) if op == "gt" else (threshold - value)
                scored = [e for e in scored if e["intervention"] != intervention]
                red_flags.append({
                    "intervention": intervention,
                    "score": deviation * rule["urgency_weight"],
                    "forced_first": True,
                })
        red_flags.sort(key=lambda x: x["score"], reverse=True)
        scored = red_flags + scored

    return scored
