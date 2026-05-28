"""
Structural ceiling calculator — master function for the bone/connective tissue system.

Implements the cascade:
    PC_Structural = T_espécie × genetic_modifier × M_structural^B (Liebig bottleneck)

CRITICAL BIOLOGICAL DISTINCTION:
Unlike aerobic/anabolic/power systems (multiplicative_combine), structural integrity
obeys Liebig's Law of the Minimum — a tendon tears at its weakest point, and no
other modifier can compensate for a structural deficiency.

Therefore: biochemical_modifier = min_combine_bottleneck(active_modifiers)

Six biochemical modifiers evaluated:
    M_BMD_structural        — bone mineral density (compressive capacity)
    M_VitC_Collagen         — vitamin C (collagen cross-link quality)
    M_PINP_CTX_structural   — bone turnover balance (PINP:CTX ratio)
    M_VitD_mineralization   — vitamin D (osteoid mineralisation, Doc2 §12.2)
    M_testosterone_mineral. — testosterone (type-I collagen synthesis, Doc2 §12.2)
    M_Inflam_Structural     — chronic inflammation (NF-κB collagen suppression)

T_SPECIES["structural_weekly_tonnage_ref"] is None (not declared numerically in
the documents). The ceiling is accordingly None; the bottleneck modifier and
its identity are still computed for diagnostic priority output.

Reference: Doc1 §2.6 + Doc1 §3.6 + Doc2 §12.1–12.2 + Doc2 §5.2
"""

from __future__ import annotations

from app.engine.base import min_combine_bottleneck
from app.engine.constants import T_SPECIES
from app.engine.modifiers.structural import (
    compute_M_BMD_structural,
    compute_M_Inflam_Structural,
    compute_M_PINP_CTX_structural,
    compute_M_VitC_Collagen,
    compute_M_VitD_mineralization,
    compute_M_testosterone_mineralization,
)

_NEUTRAL = 1.0


def calculate_structural_ceiling(
    athlete_data: dict,
    genetic_modifier: float,
) -> float | None:
    """
    Compute the structural integrity personal ceiling (TEA layer, biochemical + genetic).

    Uses min_combine_bottleneck (Liebig's Law): the weakest structural modifier
    gates the system regardless of other modifier values.

    Args:
        athlete_data: dict of biomarkers. Recognised keys:
            "t_score"               → float (bone mineral density T-score)
            "vitamin_c_umol_l"      → float (plasma vitamin C, µmol/L)
            "pinp_ug_l"             → float (P1NP bone formation marker, µg/L)
            "ctx_ng_ml"             → float (CTX bone resorption marker, ng/mL)
            "vitamin_d_ng_ml"       → float (25-OH-D, ng/mL) — Doc2 §12.2
            "testosterone_nmol_l"   → float (total testosterone, nmol/L) — Doc2 §12.2
            "crp_mg_l"              → float (hsCRP, mg/L) — Doc1 §3.6 Inflam

        genetic_modifier: float — ∏ Mj^G × ε from compute_genetic_ceiling("structural").

    Returns:
        Structural ceiling (weekly tonnage units) or None when T_espécie is undeclared.
        Formula: T_espécie × genetic_modifier × min(M_k^B)
    """
    t_especie = T_SPECIES["structural_weekly_tonnage_ref"]
    modifiers: list[float] = []

    if (t_score := athlete_data.get("t_score")) is not None:
        modifiers.append(compute_M_BMD_structural(t_score))

    if (vitc := athlete_data.get("vitamin_c_umol_l")) is not None:
        modifiers.append(compute_M_VitC_Collagen(vitc))

    if (pinp := athlete_data.get("pinp_ug_l")) is not None and \
       (ctx := athlete_data.get("ctx_ng_ml")) is not None:
        modifiers.append(compute_M_PINP_CTX_structural(pinp, ctx))

    if (vitd := athlete_data.get("vitamin_d_ng_ml")) is not None:
        modifiers.append(compute_M_VitD_mineralization(vitd))

    if (testo := athlete_data.get("testosterone_nmol_l")) is not None:
        modifiers.append(compute_M_testosterone_mineralization(testo))

    if (crp := athlete_data.get("crp_mg_l")) is not None:
        modifiers.append(compute_M_Inflam_Structural(crp))

    biochemical_modifier: float = min_combine_bottleneck(modifiers) if modifiers else _NEUTRAL

    if t_especie is None:
        return None

    return t_especie * genetic_modifier * biochemical_modifier


def calculate_structural_breakdown(
    athlete_data: dict,
    genetic_modifier: float,
) -> dict:
    """
    Return a full breakdown of the structural ceiling computation for reporting.

    Same inputs as calculate_structural_ceiling. Returns a dict with:
        "t_especie"             → species ceiling (None — not declared in documents)
        "genetic_modifier"      → input genetic modifier (TGI layer)
        "biochemical_modifier"  → min(M_k^B) — Liebig bottleneck value
        "bottleneck_key"        → name of the limiting modifier (or None if no data)
        "ceiling"               → final PC_Structural value (None if t_especie is None)
        "modifier_detail"       → per-modifier scalar (None if biomarker absent)
    """
    pinp = athlete_data.get("pinp_ug_l")
    ctx = athlete_data.get("ctx_ng_ml")

    detail: dict[str, float | None] = {
        "M_BMD_structural":    compute_M_BMD_structural(athlete_data["t_score"])
                               if "t_score" in athlete_data else None,
        "M_VitC_Collagen":     compute_M_VitC_Collagen(athlete_data["vitamin_c_umol_l"])
                               if "vitamin_c_umol_l" in athlete_data else None,
        "M_PINP_CTX_structural": compute_M_PINP_CTX_structural(pinp, ctx)
                                 if pinp is not None and ctx is not None else None,
        "M_VitD_mineralization": compute_M_VitD_mineralization(athlete_data["vitamin_d_ng_ml"])
                                 if "vitamin_d_ng_ml" in athlete_data else None,
        "M_testosterone_mineralization": compute_M_testosterone_mineralization(
                                            athlete_data["testosterone_nmol_l"]
                                         )
                                         if "testosterone_nmol_l" in athlete_data else None,
        "M_Inflam_Structural": compute_M_Inflam_Structural(athlete_data["crp_mg_l"])
                               if "crp_mg_l" in athlete_data else None,
    }

    active = {k: v for k, v in detail.items() if v is not None}
    biochemical_modifier = min_combine_bottleneck(list(active.values())) if active else _NEUTRAL
    bottleneck_key = min(active, key=lambda k: active[k]) if active else None

    t_especie = T_SPECIES["structural_weekly_tonnage_ref"]
    ceiling = t_especie * genetic_modifier * biochemical_modifier if t_especie is not None else None

    return {
        "t_especie": t_especie,
        "genetic_modifier": genetic_modifier,
        "biochemical_modifier": biochemical_modifier,
        "bottleneck_key": bottleneck_key,
        "ceiling": ceiling,
        "modifier_detail": detail,
    }
