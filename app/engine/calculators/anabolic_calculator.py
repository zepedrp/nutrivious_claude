"""
MPS ceiling calculator — master function for the anabolic system.

Implements the cascade:
    PC_MPS = T_espécie × genetic_modifier × ∏ M_k^B (biochemical layer)

All biochemical modifiers are drawn exclusively from anabolic.py, which in turn
pulls every parameter from constants.py. No numeric literals here.

Reference: Doc1 §2.4 + Doc1 §3.4 formula
"""

from __future__ import annotations

from app.engine.base import multiplicative_combine, sleep_gh_modifier
from app.engine.constants import SLEEP_PARAMS, T_SPECIES
from app.engine.modifiers.anabolic import (
    compute_M_Cortisol_MPS,
    compute_M_Energy_MPS,
    compute_M_IGF1_MPS,
    compute_M_Inflam_MPS,
    compute_M_T_MPS,
    compute_M_VitD_MPS,
)
from app.engine.modifiers.neuromuscular import compute_M_B12_Neural

_NEUTRAL = 1.0


def calculate_mps_ceiling(
    athlete_data: dict,
    genetic_modifier: float,
) -> float:
    """
    Compute the MPS fractional rate personal ceiling (TEA layer, biochemical + genetic).

    Args:
        athlete_data: dict of biomarkers. Recognised keys:
            "sex"                              → "male" | "female" (default "male")
            "testosterone_ng_dl"               → float
            "cortisol_ug_dl"                   → float (morning peak)
            "vitamin_d_ng_ml"                  → float
            "igf1_ng_ml"                       → float
            "crp_mg_l"                         → float (hsCRP)
            "energy_availability_kcal_kg_ffm"  → float (kcal/kg FFM/day)
            "mma_umol_mmol_creatinine"         → float (MMA, functional B12 proxy; Doc2 §7.4)
            "sws_pct_actual"                   → float (SWS as fraction 0–1; Doc2 §7.3)
            "tst_actual_hours"                 → float (total sleep time; Doc2 §7.3)
            "tst_genetic_hours"                → float (genotype TST optimum; from SLEEP_PARAMS)

        genetic_modifier: float — ∏ Mj^G × ε from compute_genetic_ceiling("mps").

    Returns:
        MPS fractional rate ceiling in %/h.
        Formula: T_espécie × genetic_modifier × ∏ M_k^B
    """
    t_especie: float = T_SPECIES["mps_fractional_rate_pct_per_h"]
    sex: str = athlete_data.get("sex", "male")
    modifiers: list[float] = []

    if (testo := athlete_data.get("testosterone_ng_dl")) is not None:
        modifiers.append(compute_M_T_MPS(testo, sex))

    if (cortisol := athlete_data.get("cortisol_ug_dl")) is not None:
        modifiers.append(compute_M_Cortisol_MPS(cortisol))

    if (vd := athlete_data.get("vitamin_d_ng_ml")) is not None:
        modifiers.append(compute_M_VitD_MPS(vd))

    if (igf1 := athlete_data.get("igf1_ng_ml")) is not None:
        modifiers.append(compute_M_IGF1_MPS(igf1))

    if (crp := athlete_data.get("crp_mg_l")) is not None:
        modifiers.append(compute_M_Inflam_MPS(crp))

    if (ea := athlete_data.get("energy_availability_kcal_kg_ffm")) is not None:
        modifiers.append(compute_M_Energy_MPS(ea))

    if (mma := athlete_data.get("mma_umol_mmol_creatinine")) is not None:
        modifiers.append(compute_M_B12_Neural(mma))

    sws = athlete_data.get("sws_pct_actual")
    tst = athlete_data.get("tst_actual_hours")
    tst_gen = athlete_data.get("tst_genetic_hours")
    if sws is not None and tst is not None and tst_gen is not None:
        sws_opt: float = SLEEP_PARAMS["sws_pct_optimal"]
        modifiers.append(sleep_gh_modifier(sws, tst, tst_gen, sws_opt))

    biochemical_modifier: float = multiplicative_combine(modifiers) if modifiers else _NEUTRAL
    return t_especie * genetic_modifier * biochemical_modifier


def calculate_mps_breakdown(
    athlete_data: dict,
    genetic_modifier: float,
) -> dict:
    """
    Return a full breakdown of the MPS ceiling computation for reporting.

    Same inputs as calculate_mps_ceiling. Returns a dict with:
        "t_especie"             → species ceiling
        "genetic_modifier"      → input genetic modifier (TGI layer)
        "biochemical_modifier"  → ∏ M_k^B combined
        "ceiling"               → final PC_MPS value
        "modifier_detail"       → per-modifier scalar (None if biomarker absent)
    """
    sex: str = athlete_data.get("sex", "male")

    detail: dict[str, float | None] = {
        "M_T_MPS":        compute_M_T_MPS(athlete_data["testosterone_ng_dl"], sex)
                          if "testosterone_ng_dl" in athlete_data else None,
        "M_Cortisol_MPS": compute_M_Cortisol_MPS(athlete_data["cortisol_ug_dl"])
                          if "cortisol_ug_dl" in athlete_data else None,
        "M_VitD_MPS":     compute_M_VitD_MPS(athlete_data["vitamin_d_ng_ml"])
                          if "vitamin_d_ng_ml" in athlete_data else None,
        "M_IGF1_MPS":     compute_M_IGF1_MPS(athlete_data["igf1_ng_ml"])
                          if "igf1_ng_ml" in athlete_data else None,
        "M_Inflam_MPS":   compute_M_Inflam_MPS(athlete_data["crp_mg_l"])
                          if "crp_mg_l" in athlete_data else None,
        "M_Energy_MPS":   compute_M_Energy_MPS(athlete_data["energy_availability_kcal_kg_ffm"])
                          if "energy_availability_kcal_kg_ffm" in athlete_data else None,
        "M_B12_Neural":   compute_M_B12_Neural(athlete_data["mma_umol_mmol_creatinine"])
                          if "mma_umol_mmol_creatinine" in athlete_data else None,
        "M_sleep_GH":     sleep_gh_modifier(
                              athlete_data["sws_pct_actual"],
                              athlete_data["tst_actual_hours"],
                              athlete_data["tst_genetic_hours"],
                              SLEEP_PARAMS["sws_pct_optimal"],
                          )
                          if all(k in athlete_data for k in (
                              "sws_pct_actual", "tst_actual_hours", "tst_genetic_hours"
                          )) else None,
    }

    active_modifiers = [v for v in detail.values() if v is not None]
    biochemical_modifier = multiplicative_combine(active_modifiers) if active_modifiers else _NEUTRAL
    t_especie = T_SPECIES["mps_fractional_rate_pct_per_h"]
    ceiling = t_especie * genetic_modifier * biochemical_modifier

    return {
        "t_especie": t_especie,
        "genetic_modifier": genetic_modifier,
        "biochemical_modifier": biochemical_modifier,
        "ceiling": ceiling,
        "modifier_detail": detail,
    }
