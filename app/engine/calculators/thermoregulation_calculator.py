"""
Thermoregulation ceiling calculator — master function for the thermoregulatory system.

Implements the cascade (Doc2 §10.1–10.2):
    ṁ_suor_real = ṁ_suor_espécie × genetic_modifier × ∏ M_k^B

T_SPECIES["sweat_rate_l_per_h_max"] is None (not numerically declared in the
documents). The ceiling is accordingly None; the modifier breakdown is still
computed for diagnostic output.

No nuclear genetic modifiers are declared in the documents for thermoregulation;
the genetic_modifier input defaults to 1.0 from genetic.py for this system.

Biochemical modifiers:
    M_dehydration_thermoreg  — Doc2 §10.1 (% body weight loss → sudation↓)
    M_acclimation_thermoreg  — Doc2 §10.2 (heat acclimatisation, binary)

Reference: Doc2 §10.1–10.2
"""

from __future__ import annotations

from app.engine.base import multiplicative_combine
from app.engine.constants import T_SPECIES
from app.engine.modifiers.thermoregulatory import (
    compute_M_acclimation_thermoreg,
    compute_M_dehydration_thermoreg,
)

_NEUTRAL = 1.0


def calculate_thermoregulation_ceiling(
    athlete_data: dict,
    genetic_modifier: float,
) -> float | None:
    """
    Compute the thermoregulatory personal ceiling (TEA layer, biochemical + genetic).

    Args:
        athlete_data: dict of biomarkers. Recognised keys:
            "body_weight_loss_pct"  → float (% BW lost to dehydration; Doc2 §10.1)
            "is_heat_acclimatized"  → bool (heat acclimatisation status; Doc2 §10.2)

        genetic_modifier: float — always 1.0 for this system (no declared SNPs).

    Returns:
        None — T_espécie["sweat_rate_l_per_h_max"] is not numerically declared.
    """
    t_especie = T_SPECIES["sweat_rate_l_per_h_max"]
    modifiers: list[float] = []

    if (bw_loss := athlete_data.get("body_weight_loss_pct")) is not None:
        modifiers.append(compute_M_dehydration_thermoreg(bw_loss))

    if (accl := athlete_data.get("is_heat_acclimatized")) is not None:
        modifiers.append(compute_M_acclimation_thermoreg(bool(accl)))

    biochemical_modifier: float = multiplicative_combine(modifiers) if modifiers else _NEUTRAL

    if t_especie is None:
        return None

    return t_especie * genetic_modifier * biochemical_modifier


def calculate_thermoregulation_breakdown(
    athlete_data: dict,
    genetic_modifier: float,
) -> dict:
    """
    Return a full breakdown of the thermoregulation ceiling computation for reporting.

    Same inputs as calculate_thermoregulation_ceiling. Returns a dict with:
        "t_especie"             → None (not numerically declared)
        "genetic_modifier"      → input genetic modifier (1.0 for this system)
        "biochemical_modifier"  → ∏ M_k^B combined
        "ceiling"               → None (T_espécie undefined)
        "modifier_detail"       → per-modifier scalar (None if biomarker absent)
    """
    accl = athlete_data.get("is_heat_acclimatized")

    detail: dict[str, float | None] = {
        "M_dehydration_thermoreg": compute_M_dehydration_thermoreg(
                                       athlete_data["body_weight_loss_pct"]
                                   )
                                   if "body_weight_loss_pct" in athlete_data else None,
        "M_acclimation_thermoreg": compute_M_acclimation_thermoreg(bool(accl))
                                   if accl is not None else None,
    }

    active_modifiers = [v for v in detail.values() if v is not None]
    biochemical_modifier = multiplicative_combine(active_modifiers) if active_modifiers else _NEUTRAL
    t_especie = T_SPECIES["sweat_rate_l_per_h_max"]
    ceiling = t_especie * genetic_modifier * biochemical_modifier if t_especie is not None else None

    return {
        "t_especie": t_especie,
        "genetic_modifier": genetic_modifier,
        "biochemical_modifier": biochemical_modifier,
        "ceiling": ceiling,
        "modifier_detail": detail,
    }
