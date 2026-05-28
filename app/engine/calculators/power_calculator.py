"""
Peak power ceiling calculator — master function for the neuromuscular system.

Implements the cascade:
    PC_Power = T_espécie × genetic_modifier × ∏ M_k^B (biochemical layer)

All biochemical modifiers are drawn exclusively from neuromuscular.py, which in
turn pulls every parameter from constants.py. No numeric literals here.

Reference: Doc1 §2.5 + Doc1 §3.5 formula + Doc2 §9.1–9.2
"""

from __future__ import annotations

from app.engine.base import multiplicative_combine
from app.engine.constants import T_SPECIES
from app.engine.modifiers.aerobic import compute_M_BF_aerobic
from app.engine.modifiers.neuromuscular import (
    compute_M_B12_Neural,
    compute_M_PCr_Power,
    compute_M_T_Power_male,
    compute_M_omega3_myelination,
    compute_M_sleep_PVT,
)

_NEUTRAL = 1.0


def calculate_power_ceiling(
    athlete_data: dict,
    genetic_modifier: float,
) -> float:
    """
    Compute the peak anaerobic power personal ceiling (TEA layer, biochemical + genetic).

    Args:
        athlete_data: dict of biomarkers. Recognised keys:
            "sex"                         → "male" | "female" (default "male")
            "pcr_mmol_kg"                 → float (intramuscular [PCr], mmol/kg dry mass)
            "testosterone_ng_dl"          → float (used only for males)
            "mma_umol_mmol_creatinine"    → float (methylmalonic acid, functional B12 proxy)
            "sleep_deficit_hours"         → float (Δ below genetic TST optimum)
            "omega3_index_pct"            → float (% omega-3 index)
            "body_fat_fraction"           → float (body fat as fraction 0–1; Doc1 §3.5)

        genetic_modifier: float — ∏ Mj^G × ε from compute_genetic_ceiling("peak_power").

    Returns:
        Peak 3s power ceiling in Watts (absolute).
        Formula: T_espécie × genetic_modifier × ∏ M_k^B
    """
    t_especie: float = T_SPECIES["peak_anaerobic_power_watts"]
    sex: str = athlete_data.get("sex", "male")
    modifiers: list[float] = []

    if (pcr := athlete_data.get("pcr_mmol_kg")) is not None:
        modifiers.append(compute_M_PCr_Power(pcr))

    if sex == "male" and (testo := athlete_data.get("testosterone_ng_dl")) is not None:
        modifiers.append(compute_M_T_Power_male(testo))

    if (mma := athlete_data.get("mma_umol_mmol_creatinine")) is not None:
        modifiers.append(compute_M_B12_Neural(mma))

    if (sleep_def := athlete_data.get("sleep_deficit_hours")) is not None:
        modifiers.append(compute_M_sleep_PVT(sleep_def))

    if (o3 := athlete_data.get("omega3_index_pct")) is not None:
        modifiers.append(compute_M_omega3_myelination(o3))

    if (bf := athlete_data.get("body_fat_fraction")) is not None:
        modifiers.append(compute_M_BF_aerobic(bf, sex))

    biochemical_modifier: float = multiplicative_combine(modifiers) if modifiers else _NEUTRAL
    return t_especie * genetic_modifier * biochemical_modifier


def calculate_power_breakdown(
    athlete_data: dict,
    genetic_modifier: float,
) -> dict:
    """
    Return a full breakdown of the peak power ceiling computation for reporting.

    Same inputs as calculate_power_ceiling. Returns a dict with:
        "t_especie"             → species ceiling
        "genetic_modifier"      → input genetic modifier (TGI layer)
        "biochemical_modifier"  → ∏ M_k^B combined
        "ceiling"               → final PC_Power value
        "modifier_detail"       → per-modifier scalar (None if biomarker absent)
    """
    sex: str = athlete_data.get("sex", "male")

    detail: dict[str, float | None] = {
        "M_PCr_Power":          compute_M_PCr_Power(athlete_data["pcr_mmol_kg"])
                                if "pcr_mmol_kg" in athlete_data else None,
        "M_T_Power_male":       compute_M_T_Power_male(athlete_data["testosterone_ng_dl"])
                                if sex == "male" and "testosterone_ng_dl" in athlete_data else None,
        "M_B12_Neural":         compute_M_B12_Neural(athlete_data["mma_umol_mmol_creatinine"])
                                if "mma_umol_mmol_creatinine" in athlete_data else None,
        "M_sleep_PVT":          compute_M_sleep_PVT(athlete_data["sleep_deficit_hours"])
                                if "sleep_deficit_hours" in athlete_data else None,
        "M_omega3_myelination": compute_M_omega3_myelination(athlete_data["omega3_index_pct"])
                                if "omega3_index_pct" in athlete_data else None,
        "M_BF_aerobic":         compute_M_BF_aerobic(athlete_data["body_fat_fraction"], sex)
                                if "body_fat_fraction" in athlete_data else None,
    }

    active_modifiers = [v for v in detail.values() if v is not None]
    biochemical_modifier = multiplicative_combine(active_modifiers) if active_modifiers else _NEUTRAL
    t_especie = T_SPECIES["peak_anaerobic_power_watts"]
    ceiling = t_especie * genetic_modifier * biochemical_modifier

    return {
        "t_especie": t_especie,
        "genetic_modifier": genetic_modifier,
        "biochemical_modifier": biochemical_modifier,
        "ceiling": ceiling,
        "modifier_detail": detail,
    }
