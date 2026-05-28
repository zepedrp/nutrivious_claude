"""
Fatmax ceiling calculator — master function for the lipolytic system.

Implements the cascade:
    PC_Fatmax = T_espécie × G_Fat × ∏ M_k^B (biochemical layer)

G_Fat = genetic_modifier from compute_genetic_ceiling("fatmax"), which resolves
ADRB2 and ACTN3_fatmax SNPs (Doc1 §2.3 / Doc2 §8.2).

The carnitine modifier carries a 3rd-order cross-talk with HOMA-IR: elevated
insulin resistance shifts the apparent Km of CPT-1 for carnitine via malonyl-CoA
competitive inhibition (Doc1 §2.3 + Doc2 §8.1). When the athlete's homa_ir is
missing, an effective value of 1.0 (normoinsulinaemic) is passed to preserve
continuity in the carnitine modifier.

Reference: Doc1 §2.3 + Doc1 §3.3 + Doc2 §8.1–8.2
"""

from __future__ import annotations

from app.engine.base import multiplicative_combine
from app.engine.constants import T_SPECIES
from app.engine.modifiers.lipolytic import (
    compute_M_Carnitine_Fat,
    compute_M_HOMA_Fat,
    compute_M_Insulin_Lipolysis,
    compute_M_Mito_Fat,
    compute_M_T3_Fat,
)

_NEUTRAL = 1.0
_HOMA_IR_NORMOINSULINAEMIC = 1.0


def calculate_fatmax_ceiling(
    athlete_data: dict,
    genetic_modifier: float,
) -> float:
    """
    Compute the Fatmax personal ceiling (TEA layer, biochemical + genetic).

    Args:
        athlete_data: dict of biomarkers. Recognised keys:
            "homa_ir"                → float (insulin resistance index)
            "insulin_uiu_ml"         → float (fasting insulin, µU/mL; Doc2 §8.1 HSL inhibition)
            "ft3_rt3_ratio"          → float (free T3 / reverse T3)
            "carnitine_free_umol_l"  → float (free plasma carnitine, µmol/L)
            "cs_activity_umol_min_g" → float (citrate synthase activity, µmol/min/g)

        genetic_modifier: float — ∏ Mj^G × ε from compute_genetic_ceiling("fatmax").

    Returns:
        Fatmax ceiling in g/min.
        Formula: T_espécie × genetic_modifier × ∏ M_k^B
    """
    t_especie: float = T_SPECIES["fat_oxidation_max_g_per_min"]
    homa_ir: float | None = athlete_data.get("homa_ir")
    modifiers: list[float] = []

    if homa_ir is not None:
        modifiers.append(compute_M_HOMA_Fat(homa_ir))

    if (insulin := athlete_data.get("insulin_uiu_ml")) is not None:
        modifiers.append(compute_M_Insulin_Lipolysis(insulin))

    if (ft3_rt3 := athlete_data.get("ft3_rt3_ratio")) is not None:
        modifiers.append(compute_M_T3_Fat(ft3_rt3))

    if (carnitine := athlete_data.get("carnitine_free_umol_l")) is not None:
        effective_homa = homa_ir if homa_ir is not None else _HOMA_IR_NORMOINSULINAEMIC
        modifiers.append(compute_M_Carnitine_Fat(carnitine, effective_homa))

    if (cs := athlete_data.get("cs_activity_umol_min_g")) is not None:
        modifiers.append(compute_M_Mito_Fat(cs))

    biochemical_modifier: float = multiplicative_combine(modifiers) if modifiers else _NEUTRAL
    return t_especie * genetic_modifier * biochemical_modifier


def calculate_fatmax_breakdown(
    athlete_data: dict,
    genetic_modifier: float,
) -> dict:
    """
    Return a full breakdown of the Fatmax ceiling computation for reporting.

    Same inputs as calculate_fatmax_ceiling. Returns a dict with:
        "t_especie"             → species ceiling
        "genetic_modifier"      → input genetic modifier (TGI layer)
        "biochemical_modifier"  → ∏ M_k^B combined
        "ceiling"               → final PC_Fatmax value
        "modifier_detail"       → per-modifier scalar (None if biomarker absent)
    """
    homa_ir: float | None = athlete_data.get("homa_ir")
    effective_homa = homa_ir if homa_ir is not None else _HOMA_IR_NORMOINSULINAEMIC

    detail: dict[str, float | None] = {
        "M_HOMA_Fat":          compute_M_HOMA_Fat(homa_ir)
                               if homa_ir is not None else None,
        "M_Insulin_Lipolysis": compute_M_Insulin_Lipolysis(athlete_data["insulin_uiu_ml"])
                               if "insulin_uiu_ml" in athlete_data else None,
        "M_T3_Fat":            compute_M_T3_Fat(athlete_data["ft3_rt3_ratio"])
                               if "ft3_rt3_ratio" in athlete_data else None,
        "M_Carnitine_Fat": compute_M_Carnitine_Fat(
                               athlete_data["carnitine_free_umol_l"], effective_homa
                           )
                           if "carnitine_free_umol_l" in athlete_data else None,
        "M_Mito_Fat":      compute_M_Mito_Fat(athlete_data["cs_activity_umol_min_g"])
                           if "cs_activity_umol_min_g" in athlete_data else None,
    }

    active_modifiers = [v for v in detail.values() if v is not None]
    biochemical_modifier = multiplicative_combine(active_modifiers) if active_modifiers else _NEUTRAL
    t_especie = T_SPECIES["fat_oxidation_max_g_per_min"]
    ceiling = t_especie * genetic_modifier * biochemical_modifier

    return {
        "t_especie": t_especie,
        "genetic_modifier": genetic_modifier,
        "biochemical_modifier": biochemical_modifier,
        "ceiling": ceiling,
        "modifier_detail": detail,
    }
