"""
Cognitive ceiling calculator — master function for the cognitive capacity system.

Implements the cascade (Doc2 §14.1):
    C_cognitiva_real = C_espécie × genetic_modifier × ∏ M_k^B

T_SPECIES["cognitive_capacity_index"] is None (not numerically declared in the
documents). The ceiling is accordingly None; the modifier breakdown is still
computed for diagnostic priority output and the Liebig safety rule.

Genetic modifier for this system includes:
    MTHFR_C677T_cognitive × COMT_Val158Met × ε(MTHFR×COMT) = 0.72 if TT+Met_Met

Biochemical modifiers:
    M_Hcy_cognitive              — Doc2 §14.1 (homocysteine, cerebrovascular)
    M_HbA1c_NCV                 — Doc2 §9.1  (HbA1c, myelin glycation)
    M_glycemic_variability_cognitive — Doc2 §14.1 (CGM CV%, executive function)
    M_B12_Neural                 — Doc2 §9.1  (MMA, myelin synthesis proxy)
    M_omega3_myelination         — Doc2 §9.1  (DHA, myelin membrane fluidity)
    M_sleep_PVT                  — Doc2 §9.2  (sleep deficit, psychomotor vigilance)

Reference: Doc2 §9.1–9.2 + Doc2 §14.1
"""

from __future__ import annotations

from app.engine.base import multiplicative_combine, sleep_pvt_modifier
from app.engine.constants import MODIFIER_PARAMS, SLEEP_PARAMS, T_SPECIES
from app.engine.modifiers.cognitive import (
    compute_M_glycemic_variability_cognitive,
    compute_M_HbA1c_NCV,
    compute_M_Hcy_cognitive,
)
from app.engine.modifiers.neuromuscular import compute_M_B12_Neural, compute_M_omega3_myelination

_NEUTRAL = 1.0


def calculate_cognitive_ceiling(
    athlete_data: dict,
    genetic_modifier: float,
) -> float | None:
    """
    Compute the cognitive capacity personal ceiling (TEA layer, biochemical + genetic).

    Args:
        athlete_data: dict of biomarkers. Recognised keys:
            "homocysteine_umol_l"              → float (Doc2 §14.1 cerebrovascular)
            "hba1c_pct"                        → float (Doc2 §9.1 myelin glycation)
            "glycemic_cv_pct"                  → float (CGM CV%; Doc2 §14.1)
            "mma_umol_mmol_creatinine"         → float (functional B12; Doc2 §9.1)
            "omega3_index_pct"                 → float (DHA myelination; Doc2 §9.1)
            "sleep_deficit_hours"              → float (PVT; Doc2 §9.2)

        genetic_modifier: float — ∏ Mj^G × ε(MTHFR×COMT) from genetic.py.

    Returns:
        None — T_espécie["cognitive_capacity_index"] is not numerically declared.
    """
    t_especie = T_SPECIES["cognitive_capacity_index"]
    modifiers: list[float] = []

    if (hcy := athlete_data.get("homocysteine_umol_l")) is not None:
        modifiers.append(compute_M_Hcy_cognitive(hcy))

    if (hba1c := athlete_data.get("hba1c_pct")) is not None:
        modifiers.append(compute_M_HbA1c_NCV(hba1c))

    if (cv := athlete_data.get("glycemic_cv_pct")) is not None:
        modifiers.append(compute_M_glycemic_variability_cognitive(cv))

    if (mma := athlete_data.get("mma_umol_mmol_creatinine")) is not None:
        modifiers.append(compute_M_B12_Neural(mma))

    if (o3 := athlete_data.get("omega3_index_pct")) is not None:
        modifiers.append(compute_M_omega3_myelination(o3))

    if (sleep_def := athlete_data.get("sleep_deficit_hours")) is not None:
        p = MODIFIER_PARAMS["M_sleep_PVT"]
        modifiers.append(sleep_pvt_modifier(sleep_def, p["rate"]))

    biochemical_modifier: float = multiplicative_combine(modifiers) if modifiers else _NEUTRAL

    if t_especie is None:
        return None

    return t_especie * genetic_modifier * biochemical_modifier


def calculate_cognitive_breakdown(
    athlete_data: dict,
    genetic_modifier: float,
) -> dict:
    """
    Return a full breakdown of the cognitive ceiling computation for reporting.

    Same inputs as calculate_cognitive_ceiling. Returns a dict with:
        "t_especie"             → None (not numerically declared)
        "genetic_modifier"      → input genetic modifier (TGI layer, includes ε_MTHFR×COMT)
        "biochemical_modifier"  → ∏ M_k^B combined
        "ceiling"               → None (T_espécie undefined)
        "modifier_detail"       → per-modifier scalar (None if biomarker absent)
    """
    p_pvt = MODIFIER_PARAMS["M_sleep_PVT"]

    detail: dict[str, float | None] = {
        "M_Hcy_cognitive":      compute_M_Hcy_cognitive(athlete_data["homocysteine_umol_l"])
                                if "homocysteine_umol_l" in athlete_data else None,
        "M_HbA1c_NCV":         compute_M_HbA1c_NCV(athlete_data["hba1c_pct"])
                                if "hba1c_pct" in athlete_data else None,
        "M_glycemic_var":       compute_M_glycemic_variability_cognitive(
                                    athlete_data["glycemic_cv_pct"]
                                )
                                if "glycemic_cv_pct" in athlete_data else None,
        "M_B12_Neural":         compute_M_B12_Neural(athlete_data["mma_umol_mmol_creatinine"])
                                if "mma_umol_mmol_creatinine" in athlete_data else None,
        "M_omega3_myelination": compute_M_omega3_myelination(athlete_data["omega3_index_pct"])
                                if "omega3_index_pct" in athlete_data else None,
        "M_sleep_PVT":          sleep_pvt_modifier(
                                    athlete_data["sleep_deficit_hours"], p_pvt["rate"]
                                )
                                if "sleep_deficit_hours" in athlete_data else None,
    }

    active_modifiers = [v for v in detail.values() if v is not None]
    biochemical_modifier = multiplicative_combine(active_modifiers) if active_modifiers else _NEUTRAL
    t_especie = T_SPECIES["cognitive_capacity_index"]
    ceiling = t_especie * genetic_modifier * biochemical_modifier if t_especie is not None else None

    return {
        "t_especie": t_especie,
        "genetic_modifier": genetic_modifier,
        "biochemical_modifier": biochemical_modifier,
        "ceiling": ceiling,
        "modifier_detail": detail,
    }
