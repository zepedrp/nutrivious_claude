"""
Neuromuscular modifier layer — individual transfer functions for peak power and
neural conduction system.

Each function pulls its own parameter dict from MODIFIER_PARAMS and calls the
appropriate base.py primitive. Zero numeric literals live in this file.

Reference: Doc1 §2.5 + Doc2 §9.1–9.2
"""

from __future__ import annotations

from app.engine.base import (
    linear_ratio,
    piecewise_linear,
    power_law,
    sleep_pvt_modifier,
)
from app.engine.constants import MODIFIER_PARAMS


def compute_M_PCr_Power(pcr_mmol_kg: float) -> float:
    """M_PCr_Power — intramuscular phosphocreatine as immediate energy buffer modifier.

    Doc1 §2.5: M_PCr = clamp([PCr] / 80, 0.80, 1.25). ref=80 mmol/kg dry mass.
    Creatine loading ≥4 weeks raises [PCr] ~20–30% → M ≈ 1.22.
    """
    p = MODIFIER_PARAMS["M_PCr_Power"]
    return linear_ratio(pcr_mmol_kg, p["reference"], p["floor"], p["ceiling"])


def compute_M_T_Power_male(testosterone_ng_dl: float) -> float:
    """M_T_Power_male — testosterone as neuromuscular excitability modifier (male only).

    Doc1 §2.5: exponent=0.25 (vs 0.35 in MPS) because neural effect (motoneuron
    excitability + MHC-IIx expression) saturates faster than transcriptional anabolism.
    Female athletes: pass 1.0 (not applicable).
    """
    p = MODIFIER_PARAMS["M_T_Power_male"]
    return power_law(testosterone_ng_dl, p["reference"], p["exponent"], p["floor"], p["ceiling"])


def compute_M_B12_Neural(mma_umol_mmol_creatinine: float) -> float:
    """M_B12_Neural — methylmalonic acid as functional B12 deficiency modifier.

    Doc1 §2.5: MMA > 3 µmol/mmol_creatinine = functional intracellular B12 deficit.
    MMA is more sensitive than serum B12 for detecting neural deficit.
    """
    p = MODIFIER_PARAMS["M_B12_Neural"]
    return piecewise_linear(
        mma_umol_mmol_creatinine, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


def compute_M_sleep_PVT(sleep_deficit_hours: float) -> float:
    """M_sleep_PVT — sleep deficit as psychomotor vigilance and reaction time modifier.

    Doc2 §9.2: M_sono^PVT = exp(−0.12 × Δ_sono).
    Δ_sono = hours below the athlete's genetic TST optimum (PER3 VNTR).
    """
    p = MODIFIER_PARAMS["M_sleep_PVT"]
    return sleep_pvt_modifier(sleep_deficit_hours, p["rate"])


def compute_M_omega3_myelination(omega3_index_pct: float) -> float:
    """M_omega3_myelination — omega-3 index as myelin membrane fluidity modifier.

    Doc2 §9.1: DHA constitutes >35% of grey matter fatty acids; higher DHA
    content → membrane fluidity → nerve conduction velocity↑.
    Linear 0.80 at 0% to 1.00 at ≥8% omega-3 index.
    """
    p = MODIFIER_PARAMS["M_omega3_myelination"]
    return piecewise_linear(
        omega3_index_pct, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )
