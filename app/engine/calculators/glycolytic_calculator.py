"""
CHO absorption ceiling calculator — master function for the glycolytic system.

Implements the cascade:
    PC_CHO = T_espécie × G_CHO × ∏ M_k^B (biochemical layer)

G_CHO = 1.0 by document specification (Doc1 §3.2: no dominant nuclear genetic
modifier declared for CHO intestinal transport).

All biochemical modifiers are drawn exclusively from glycolytic.py, which in turn
pulls every parameter from constants.py. No numeric literals here.

Reference: Doc1 §2.2 + Doc1 §3.2 formula
"""

from __future__ import annotations

from app.engine.base import multiplicative_combine
from app.engine.constants import T_SPECIES
from app.engine.modifiers.glycolytic import (
    compute_C_micro_Veillonella,
    compute_M_Calprotectin_CHO,
    compute_M_GutTrain_CHO,
    compute_M_HOMA_CHO,
    compute_M_Microbiome_CHO,
    compute_M_Zonulin_CHO,
)

_NEUTRAL = 1.0
_G_CHO = 1.0


def calculate_cho_ceiling(athlete_data: dict) -> float:
    """
    Compute the CHO absorption personal ceiling (TEA layer, biochemical only).

    Args:
        athlete_data: dict of biomarkers. Recognised keys:
            "homa_ir"             → float (insulin resistance index)
            "zonulin_ng_ml"       → float (intestinal permeability marker)
            "gut_training_weeks"  → float (cumulative gut CHO training)
            "microbiome_shannon"  → float (Shannon diversity index)
            "calprotectin_ug_g"   → float (intestinal inflammation marker)
            "veillonella_pct"     → float (Veillonella atypica abundance %; Doc2 §11.1)

    Returns:
        CHO absorption ceiling in g/h.
        Formula: T_espécie × 1.0 × ∏ M_k^B × C_micro^Veillonella
    """
    t_especie: float = T_SPECIES["cho_absorption_g_per_h"]
    modifiers: list[float] = []

    if (homa := athlete_data.get("homa_ir")) is not None:
        modifiers.append(compute_M_HOMA_CHO(homa))

    if (zonulin := athlete_data.get("zonulin_ng_ml")) is not None:
        modifiers.append(compute_M_Zonulin_CHO(zonulin))

    if (gut_weeks := athlete_data.get("gut_training_weeks")) is not None:
        modifiers.append(compute_M_GutTrain_CHO(gut_weeks))

    if (shannon := athlete_data.get("microbiome_shannon")) is not None:
        modifiers.append(compute_M_Microbiome_CHO(shannon))

    if (calpro := athlete_data.get("calprotectin_ug_g")) is not None:
        modifiers.append(compute_M_Calprotectin_CHO(calpro))

    biochemical_modifier: float = multiplicative_combine(modifiers) if modifiers else _NEUTRAL

    # Microbiome co-factor C_micro (Doc2 §11.1 — Veillonella lactate→propionate pathway)
    c_micro: float = _NEUTRAL
    if (veillonella := athlete_data.get("veillonella_pct")) is not None:
        c_micro = compute_C_micro_Veillonella(veillonella)

    return t_especie * _G_CHO * biochemical_modifier * c_micro


def calculate_cho_breakdown(athlete_data: dict) -> dict:
    """
    Return a full breakdown of the CHO ceiling computation for reporting.

    Same inputs as calculate_cho_ceiling. Returns a dict with:
        "t_especie"             → species ceiling
        "g_cho"                 → genetic modifier (always 1.0 per Doc1 §3.2)
        "biochemical_modifier"  → ∏ M_k^B combined
        "c_micro"               → Veillonella co-factor (1.0 if absent)
        "ceiling"               → final PC_CHO value (includes C_micro)
        "modifier_detail"       → per-modifier scalar (None if biomarker absent)
    """
    detail: dict[str, float | None] = {
        "M_HOMA_CHO":         compute_M_HOMA_CHO(athlete_data["homa_ir"])
                              if "homa_ir" in athlete_data else None,
        "M_Zonulin_CHO":      compute_M_Zonulin_CHO(athlete_data["zonulin_ng_ml"])
                              if "zonulin_ng_ml" in athlete_data else None,
        "M_GutTrain_CHO":     compute_M_GutTrain_CHO(athlete_data["gut_training_weeks"])
                              if "gut_training_weeks" in athlete_data else None,
        "M_Microbiome_CHO":   compute_M_Microbiome_CHO(athlete_data["microbiome_shannon"])
                              if "microbiome_shannon" in athlete_data else None,
        "M_Calprotectin_CHO": compute_M_Calprotectin_CHO(athlete_data["calprotectin_ug_g"])
                              if "calprotectin_ug_g" in athlete_data else None,
        "C_micro_Veillonella": compute_C_micro_Veillonella(athlete_data["veillonella_pct"])
                               if "veillonella_pct" in athlete_data else None,
    }

    # C_micro is a separate layer (Doc2 §11.1) — not part of ∏ M_k^B product.
    c_micro_val: float | None = detail.pop("C_micro_Veillonella")
    active_modifiers = [v for v in detail.values() if v is not None]
    biochemical_modifier = multiplicative_combine(active_modifiers) if active_modifiers else _NEUTRAL
    c_micro: float = c_micro_val if c_micro_val is not None else _NEUTRAL

    t_especie = T_SPECIES["cho_absorption_g_per_h"]
    ceiling = t_especie * _G_CHO * biochemical_modifier * c_micro

    # Restore C_micro into detail for reporting.
    detail["C_micro_Veillonella"] = c_micro_val

    return {
        "t_especie": t_especie,
        "g_cho": _G_CHO,
        "biochemical_modifier": biochemical_modifier,
        "c_micro": c_micro,
        "ceiling": ceiling,
        "modifier_detail": detail,
    }
