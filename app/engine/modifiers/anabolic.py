"""
Anabolic modifier layer — individual transfer functions for MPS system.

Each function receives a single biomarker value (plus sex where needed), pulls
its own parameter dict from MODIFIER_PARAMS, and calls the appropriate base.py
primitive. Zero numeric literals live in this file.

Reference: Doc1 §2.4
"""

from __future__ import annotations

from app.engine.base import (
    exponential_decay,
    linear_with_threshold,
    piecewise_linear,
    power_law,
)
from app.engine.constants import MODIFIER_PARAMS


def compute_M_T_MPS(testosterone_ng_dl: float, sex: str) -> float:
    """M_T_MPS — testosterone as androgen receptor-mediated MPS modifier.

    Doc1 §2.4: power-law (exponent=0.35) models saturable AR kinetics (Kd ≈ 0.1–1 nM).
    ref=600 ng/dL (male) | 40 ng/dL (female). floor=0.75, ceiling=1.20.
    Bhasin et al. NEJM 2001 dose-response.
    """
    key = "M_T_MPS_male" if sex == "male" else "M_T_MPS_female"
    p = MODIFIER_PARAMS[key]
    return power_law(testosterone_ng_dl, p["reference"], p["exponent"], p["floor"], p["ceiling"])


def compute_M_Cortisol_MPS(cortisol_ug_dl: float) -> float:
    """M_Cortisol_MPS — morning cortisol as catabolic suppression modifier on MPS.

    Doc1 §2.4: REDD1 → TSC1/2 → mTORC1 inhibition; MuRF-1 + MAFbx → proteolysis.
    k=0.025; ref=14 µg/dL; floor=0.60 (severe hypercortisolism).
    """
    p = MODIFIER_PARAMS["M_Cortisol_MPS"]
    return linear_with_threshold(
        cortisol_ug_dl, p["reference"], p["slope"], p["floor"], p["ceiling"]
    )


def compute_M_VitD_MPS(vitamin_d_ng_ml: float) -> float:
    """M_VitD_MPS — vitamin D as myocyte VDR and mTORC1 co-activation modifier.

    Doc1 §2.4: VDR upregulates IGF-1R; activates myofibrillar genes; VDR-RXR
    recruits mTORC1 co-activators. Flat below 20 ng/mL (floor=0.72).
    """
    p = MODIFIER_PARAMS["M_VitD_MPS"]
    return piecewise_linear(
        vitamin_d_ng_ml, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


def compute_M_IGF1_MPS(igf1_ng_ml: float) -> float:
    """M_IGF1_MPS — IGF-1 as PI3K/Akt/mTORC1 anabolic signalling modifier.

    Doc1 §2.4: activates mTORC1 independently of insulin resistance.
    Power-law exponent=0.40; ref=180 ng/mL.
    """
    p = MODIFIER_PARAMS["M_IGF1_MPS"]
    return power_law(igf1_ng_ml, p["reference"], p["exponent"], p["floor"], p["ceiling"])


def compute_M_Inflam_MPS(crp_mg_l: float) -> float:
    """M_Inflam_MPS — hsCRP as IRS-1 Ser307 phosphorylation penalty on MPS.

    Doc1 §2.4: IKK-β and JNK phosphorylate IRS-1 at Ser307, independent and
    additive to cortisol's effect. k=0.20 (more aggressive than aerobic); floor=0.65.
    """
    p = MODIFIER_PARAMS["M_Inflam_MPS"]
    return exponential_decay(crp_mg_l, p["rate"], p["reference"], p["floor"], p["ceiling"])


def compute_M_Energy_MPS(energy_availability_kcal_kg_ffm: float) -> float:
    """M_Energy_MPS — energy availability as RED-S gating modifier on MPS.

    Doc1 §2.4: EA < 45 kcal/kg_FFM/day → organism sacrifices protein synthesis
    to conserve energy (RED-S cascade). Floor=0.50 at severe restriction.
    """
    p = MODIFIER_PARAMS["M_Energy_MPS"]
    return piecewise_linear(
        energy_availability_kcal_kg_ffm,
        p["x_points"], p["y_points"], p["floor"], p["ceiling"],
    )
