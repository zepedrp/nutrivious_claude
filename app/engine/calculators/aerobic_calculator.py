"""
VO2max ceiling calculator — master function for the aerobic system.

Implements the cascade:
    PC_VO2 = T_espécie × genetic_modifier × ∏ M_k^B (biochemical layer)

All biochemical modifiers are drawn exclusively from aerobic.py, which in turn
pulls every parameter from constants.py. No numeric literals here.

Reference: Doc1 §2.1 + Doc2 §5.2 + Fase3 §3.1 formula
"""

from __future__ import annotations

from app.engine.base import multiplicative_combine
from app.engine.constants import T_SPECIES
from app.engine.modifiers.aerobic import (
    compute_E_PGC1A_methylation,
    compute_M_BF_aerobic,
    compute_M_CRP_aerobic,
    compute_M_DunedinPACE,
    compute_M_Fe_aerobic,
    compute_M_Hb,
    compute_M_HOMA_vo2,
    compute_M_O3_aerobic,
    compute_M_T3_vo2,
    compute_M_VitD_aerobic,
    compute_M_cortisol_dhea_ratio_vo2,
    compute_M_ferritin_vo2,
    compute_M_testosterone_vo2_male,
)

# Sentinel: missing biomarker → modifier defaults to 1.0 (neutral, no penalty)
_NEUTRAL = 1.0


def calculate_vo2max_ceiling(
    athlete_data: dict,
    genetic_modifier: float,
) -> float:
    """
    Compute the VO2max personal ceiling (TEA layer, biochemical + genetic).

    Args:
        athlete_data: dict of biomarkers. Recognised keys:
            "sex"                         → "male" | "female"  (default "male")
            "hemoglobin_g_dl"             → float, e.g. 14.8
            "body_fat_fraction"           → float [0, 1], e.g. 0.19 for 19%
            "omega3_index_pct"            → float %, e.g. 5.2
            "vitamin_d_ng_ml"             → float
            "crp_mg_l"                    → float (hsCRP)
            "stfr_logferritin_ratio"       → float (sTfR/log(Ferritin) — primary; Doc1 §2.1)
            "ferritin_ug_l"               → float (fallback when ratio absent; Doc2 §5.2)
            "homa_ir"                     → float
            "cortisol_dhea_ratio"         → float (morning cortisol µg/dL / DHEA-S µg/dL)
            "dunedin_pace"                → float (epigenetic aging clock output)
            "t3_free_pmol_l"              → float (free T3)
            "testosterone_nmol_l"         → float (total testosterone; used only for males)
            "ppargc1a_methylation_pct"    → float (% PPARGC1A promoter methylation; Doc1 §3.1 E layer)

        genetic_modifier: float — ∏ Mj^G × ε from genetic.py for the "vo2max" system.

    Returns:
        VO2max personal ceiling in mL/kg/min.
        Formula: T_espécie × genetic_modifier × ∏ M_k^B
    """
    t_especie: float = T_SPECIES["vo2max_ml_per_kg_min"]
    sex: str = athlete_data.get("sex", "male")

    modifiers: list[float] = []

    # --- Blood oxygen transport (Doc1 §2.1 — Fick equation) ---
    if (hb := athlete_data.get("hemoglobin_g_dl")) is not None:
        modifiers.append(compute_M_Hb(hb, sex))

    # --- Morphological cost (Doc1 §2.1 — inert mass) ---
    if (bf := athlete_data.get("body_fat_fraction")) is not None:
        modifiers.append(compute_M_BF_aerobic(bf, sex))

    # --- Mitochondrial membrane quality (Doc1 §2.1 — cardiolipin) ---
    if (o3 := athlete_data.get("omega3_index_pct")) is not None:
        modifiers.append(compute_M_O3_aerobic(o3))

    # --- Multi-target VDR axis (Doc1 §2.1 — EPO + cardiac + PGC-1α) ---
    if (vd := athlete_data.get("vitamin_d_ng_ml")) is not None:
        modifiers.append(compute_M_VitD_aerobic(vd))

    # --- Chronic inflammation (Doc1 §2.1 — hepcidina + eNOS) ---
    if (crp := athlete_data.get("crp_mg_l")) is not None:
        modifiers.append(compute_M_CRP_aerobic(crp))

    # --- Iron status: sTfR/log(Ferritin) primary (Doc1 §2.1), ferritin fallback (Doc2 §5.2) ---
    if (fe_ratio := athlete_data.get("stfr_logferritin_ratio")) is not None:
        modifiers.append(compute_M_Fe_aerobic(fe_ratio))
    elif (ferr := athlete_data.get("ferritin_ug_l")) is not None:
        modifiers.append(compute_M_ferritin_vo2(ferr))

    # --- Insulin resistance / cardiac output efficiency (Doc2 §5.2) ---
    if (homa := athlete_data.get("homa_ir")) is not None:
        modifiers.append(compute_M_HOMA_vo2(homa))

    # --- HPA axis balance (Doc2 §5.2 — catabolic dominance) ---
    if (ratio := athlete_data.get("cortisol_dhea_ratio")) is not None:
        modifiers.append(compute_M_cortisol_dhea_ratio_vo2(ratio))

    # --- Epigenetic aging rate (Doc2 §5.2 — DunedinPACE) ---
    if (pace := athlete_data.get("dunedin_pace")) is not None:
        modifiers.append(compute_M_DunedinPACE(pace))

    # --- Thyroid axis / mitochondrial biogenesis (Doc2 §5.2 — PGC-1α, PPAR-α) ---
    if (t3 := athlete_data.get("t3_free_pmol_l")) is not None:
        modifiers.append(compute_M_T3_vo2(t3))

    # --- Anabolic support for aerobic adaptation (Doc2 §5.2 — male only) ---
    if sex == "male" and (testo := athlete_data.get("testosterone_nmol_l")) is not None:
        modifiers.append(compute_M_testosterone_vo2_male(testo))

    biochemical_modifier_product: float = (
        multiplicative_combine(modifiers) if modifiers else _NEUTRAL
    )

    # --- Epigenetic layer E (Doc1 §3.1 — semi-annual; distinct from M layer) ---
    epigenetic_modifier: float = _NEUTRAL
    if (meth := athlete_data.get("ppargc1a_methylation_pct")) is not None:
        epigenetic_modifier = compute_E_PGC1A_methylation(meth)

    return t_especie * genetic_modifier * epigenetic_modifier * biochemical_modifier_product


def calculate_vo2max_breakdown(
    athlete_data: dict,
    genetic_modifier: float,
) -> dict:
    """
    Return a full breakdown of the VO2max ceiling computation for reporting.

    Same inputs as calculate_vo2max_ceiling. Returns a dict with:
        "t_especie"                  → species ceiling
        "genetic_modifier"           → input genetic modifier (TGI layer)
        "epigenetic_modifier"        → E_PGC1A (1.0 if ppargc1a_methylation_pct absent)
        "biochemical_modifier"       → ∏ M_k^B combined
        "ceiling"                    → final PC_VO2 value
        "modifier_detail"            → per-modifier scalar (None if biomarker absent)
    """
    sex: str = athlete_data.get("sex", "male")

    epigenetic_modifier: float = _NEUTRAL
    if (meth := athlete_data.get("ppargc1a_methylation_pct")) is not None:
        epigenetic_modifier = compute_E_PGC1A_methylation(meth)

    detail: dict[str, float | None] = {
        "M_Hb":                     compute_M_Hb(athlete_data["hemoglobin_g_dl"], sex)
                                    if "hemoglobin_g_dl" in athlete_data else None,
        "M_BF_aerobic":             compute_M_BF_aerobic(athlete_data["body_fat_fraction"], sex)
                                    if "body_fat_fraction" in athlete_data else None,
        "M_O3_aerobic":             compute_M_O3_aerobic(athlete_data["omega3_index_pct"])
                                    if "omega3_index_pct" in athlete_data else None,
        "M_VitD_aerobic":           compute_M_VitD_aerobic(athlete_data["vitamin_d_ng_ml"])
                                    if "vitamin_d_ng_ml" in athlete_data else None,
        "M_CRP_aerobic":            compute_M_CRP_aerobic(athlete_data["crp_mg_l"])
                                    if "crp_mg_l" in athlete_data else None,
        "M_Fe_aerobic":             compute_M_Fe_aerobic(athlete_data["stfr_logferritin_ratio"])
                                    if "stfr_logferritin_ratio" in athlete_data else None,
        "M_ferritin_vo2":           compute_M_ferritin_vo2(athlete_data["ferritin_ug_l"])
                                    if "ferritin_ug_l" in athlete_data
                                    and "stfr_logferritin_ratio" not in athlete_data
                                    else None,
        "M_HOMA_vo2":               compute_M_HOMA_vo2(athlete_data["homa_ir"])
                                    if "homa_ir" in athlete_data else None,
        "M_cortisol_dhea_ratio_vo2":compute_M_cortisol_dhea_ratio_vo2(athlete_data["cortisol_dhea_ratio"])
                                    if "cortisol_dhea_ratio" in athlete_data else None,
        "M_DunedinPACE":            compute_M_DunedinPACE(athlete_data["dunedin_pace"])
                                    if "dunedin_pace" in athlete_data else None,
        "M_T3_vo2":                 compute_M_T3_vo2(athlete_data["t3_free_pmol_l"])
                                    if "t3_free_pmol_l" in athlete_data else None,
        "M_testosterone_vo2_male":  compute_M_testosterone_vo2_male(athlete_data["testosterone_nmol_l"])
                                    if sex == "male" and "testosterone_nmol_l" in athlete_data else None,
    }

    active_modifiers = [v for v in detail.values() if v is not None]
    biochemical_modifier = multiplicative_combine(active_modifiers) if active_modifiers else _NEUTRAL
    t_especie = T_SPECIES["vo2max_ml_per_kg_min"]
    ceiling = t_especie * genetic_modifier * epigenetic_modifier * biochemical_modifier

    return {
        "t_especie": t_especie,
        "genetic_modifier": genetic_modifier,
        "epigenetic_modifier": epigenetic_modifier,
        "biochemical_modifier": biochemical_modifier,
        "ceiling": ceiling,
        "modifier_detail": detail,
    }
