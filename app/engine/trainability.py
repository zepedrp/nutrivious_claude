"""
Trainability layer — adaptation gap analysis and RGC trajectory projection.

Gap de Adaptação: GA(S) = TGI(S) − TEA(S)
RGC trajectory:   TEA(t) = TGI − GA_0 × e^(−RGC × t)

RGC (Rate of Gap Closure) is genotype-aware: athletes with favorable SNPs
for a system express the higher RGC_RATES["with_favorable"] rate.

Reference: Doc1 §6.1, Doc2 §4.2, Doc2 §18.2
"""

from __future__ import annotations

import math

from app.engine.base import sleep_pvt_modifier
from app.engine.constants import (
    FAVORABLE_ALLELES_PER_SYSTEM,
    MODIFIER_PARAMS,
    RGC_RATES,
    TRAJECTORY_DEGRADATION_PARAMS,
)

# Maps each BOS system to its RGC_RATES key.
# cho: G_CHO = 1.0 → TGI = T_espécie; gap is biochemical only; no RGC model applies.
# structural, cognitive, thermoregulatory: no RGC model declared in the documents.
_SYSTEM_TO_RGC_KEY: dict[str, str | None] = {
    "vo2max":     "vo2max",
    "cho":        None,
    "fatmax":     "fatmax",
    "mps":        "mps_hypertrophy",
    "peak_power": "peak_power",
    "structural": None,
}


def calculate_adaptation_gap(tgi: float, current_tea: float) -> dict:
    """
    Compute the adaptation gap for one physiological system.

    Args:
        tgi:         genetic ceiling (T_espécie × genetic_modifier)
        current_tea: expressed actual ceiling (TEA — biochemical + genetic combined)

    Returns dict with:
        "tgi"          → genetic ceiling
        "current_tea"  → expressed ceiling
        "absolute_gap" → TGI − TEA (clamped to ≥ 0)
        "gap_pct"      → absolute_gap / tgi × 100  (0.0 when tgi == 0)
    """
    absolute_gap = max(tgi - current_tea, 0.0)
    gap_pct = (absolute_gap / tgi * 100.0) if tgi > 0.0 else 0.0
    return {
        "tgi":          tgi,
        "current_tea":  current_tea,
        "absolute_gap": absolute_gap,
        "gap_pct":      gap_pct,
    }


def gap_closure_trajectory(
    tgi: float,
    current_tea: float,
    system: str,
    genotype: dict[str, str],
    weeks: int,
    athlete_data: dict | None = None,
) -> list[float]:
    """
    Project TEA(t) over `weeks` weeks of optimal training.

    Formula (Doc2 §4.2 + Doc2 §4.3 T_modifier chain):
        RGC_effective = RGC_genetic × M_sleep_PVT × (2.0 − DunedinPACE)
        TEA(t) = TGI − GA_0 × e^(−RGC_effective × t)

    The base RGC rate is drawn from RGC_RATES (favorable vs. unfavorable genotype).
    Lifestyle blockers from athlete_data scale the effective rate down:
        M_sleep_PVT   — Doc2 §9.2: exp(−0.12 × sleep_deficit_hours)
        DunedinPACE   — Doc2 §4.2: (2.0 − pace); pace=1.0 → neutral, pace>1.0 → slower

    Systems without a defined RGC key return an empty list.

    Args:
        tgi:          genetic ceiling for this system
        current_tea:  expressed ceiling at t=0
        system:       BOS system name
        genotype:     athlete's SNP dict (field → genotype string)
        weeks:        projection horizon (returns weeks + 1 values: t=0 … t=weeks)
        athlete_data: optional biomarker dict for T_modifier computation;
                      keys used: "sleep_deficit_hours", "dunedin_pace".

    Returns:
        list[float] of length weeks + 1, or [] when the system has no RGC model.
    """
    rgc_key = _SYSTEM_TO_RGC_KEY.get(system)
    if rgc_key is None:
        return []

    rgc_table = RGC_RATES.get(rgc_key)
    if rgc_table is None:
        return []

    favorable = any(
        genotype.get(field) == value
        for field, value in FAVORABLE_ALLELES_PER_SYSTEM.get(system, ())
    )
    rgc: float = (
        rgc_table["with_favorable"] if favorable else rgc_table["without_favorable"]
    )

    # T_modifier chain: lifestyle blockers that compress the effective RGC.
    if athlete_data is not None:
        # M_sleep_PVT: psychomotor vigilance degradation by sleep deficit (Doc2 §9.2)
        p_pvt = MODIFIER_PARAMS["M_sleep_PVT"]
        sleep_deficit: float = athlete_data.get("sleep_deficit_hours", 0.0)
        m_sleep: float = sleep_pvt_modifier(sleep_deficit, p_pvt["rate"])

        # DunedinPACE factor: (base − pace) → 1.0 at pace=1.0; lower for fast agers.
        pace: float = athlete_data.get("dunedin_pace", 1.0)
        pace_base: float = TRAJECTORY_DEGRADATION_PARAMS["dunedin_pace_base"]
        pace_floor: float = TRAJECTORY_DEGRADATION_PARAMS["dunedin_pace_floor"]
        pace_factor: float = max(pace_floor, pace_base - pace)

        rgc = rgc * m_sleep * pace_factor

    ga0 = max(tgi - current_tea, 0.0)

    return [tgi - ga0 * math.exp(-rgc * t) for t in range(weeks + 1)]
