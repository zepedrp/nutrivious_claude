"""
Lipolytic modifier layer — individual transfer functions for fat oxidation system.

Each function receives biomarker value(s), pulls its own parameter dict
from MODIFIER_PARAMS or HILL_KINETICS, and calls the appropriate base.py primitive.
Zero numeric literals live in this file.

compute_M_Carnitine_Fat implements the 3rd-order CPT-1/malonyl-CoA cross-talk:
HOMA-IR → [malonyl-CoA]↑ → competitive CPT-1 inhibition → Km_apparent↑.
Falls back to piecewise_linear when Km_basal or homa_to_malonyl_k are None.

Reference: Doc1 §2.3 + Doc2 §8.1–8.2
"""

from __future__ import annotations

from app.engine.base import (
    clamp,
    hill_inhibition,
    hill_saturation,
    linear_ratio,
    linear_with_threshold,
    piecewise_linear,
)
from app.engine.constants import HILL_KINETICS, MODIFIER_PARAMS


def compute_M_HOMA_Fat(homa_ir: float) -> float:
    """M_HOMA_Fat — insulin resistance as CPT-1/HSL suppression modifier.

    Doc1 §2.3: k=0.08 (less aggressive than CHO) because malonyl-CoA inhibition
    of CPT-1 is competitive, not total. Floor=0.50 at severe IR.
    """
    p = MODIFIER_PARAMS["M_HOMA_Fat"]
    return linear_with_threshold(homa_ir, p["reference"], p["slope"], p["floor"], p["ceiling"])


def compute_M_T3_Fat(ft3_rt3_ratio: float) -> float:
    """M_T3_Fat — FT3/rT3 ratio as fat oxidation transcription modifier.

    Doc1 §2.3: T3 activates PGC-1α, PPAR-α, CPT-1, UCP1 transcription.
    rT3 competes for same receptors without activating transcription.
    Plateau at ratio ≥ 20; deficit floor at 0.75.
    """
    p = MODIFIER_PARAMS["M_T3_Fat"]
    return piecewise_linear(ft3_rt3_ratio, p["x_points"], p["y_points"], p["floor"], p["ceiling"])


def compute_M_Carnitine_Fat(carnitine_free_umol_l: float, homa_ir: float) -> float:
    """M_Carnitine_Fat — free carnitine as CPT-1 substrate with HOMA-IR cross-talk.

    3rd-order effect (Doc1 §2.3 + Doc2 §8.1): HOMA-IR → malonyl-CoA↑ →
    competitive CPT-1 inhibition → Km_apparent = Km_basal × (1 + [MalonylCoA]/Ki).
    km_shift = Km_basal × (homa_ir × homa_to_malonyl_k) / ki_umol_l.

    Falls back to piecewise_linear when CPT1_carnitine_basal_km or
    homa_to_malonyl_k are None (not declared in documents).
    """
    cpt1_inhibition = HILL_KINETICS["malonyl_coa_CPT1_inhibition"]
    km_basal = HILL_KINETICS["CPT1_carnitine_basal_km"]
    homa_to_malonyl_k = cpt1_inhibition["homa_to_malonyl_k"]
    ki = cpt1_inhibition["ki_umol_l"]

    p = MODIFIER_PARAMS["M_Carnitine_Fat"]

    if km_basal is not None and homa_to_malonyl_k is not None:
        malonyl_coa = homa_ir * homa_to_malonyl_k
        km_shift = km_basal * malonyl_coa / ki
        raw = hill_saturation(carnitine_free_umol_l, km_basal, km_shift=km_shift)
        return clamp(raw, p["floor"], p["ceiling"])

    return piecewise_linear(
        carnitine_free_umol_l, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


def compute_M_Mito_Fat(cs_activity_umol_min_g: float) -> float:
    """M_Mito_Fat — citrate synthase activity as mitochondrial density modifier.

    Doc1 §3.3: M_Mito_Fat = clamp(CS / 25, 0.40, 1.80).
    CS activity is the gold-standard proxy for mitochondrial density without biopsy.
    """
    p = MODIFIER_PARAMS["M_Mito_Fat"]
    return linear_ratio(cs_activity_umol_min_g, p["reference"], p["floor"], p["ceiling"])


def compute_M_Insulin_Lipolysis(insulin_uiu_ml: float) -> float:
    """M_insulina^lipólise — HSL inhibition by fasting insulin (Hill inhibition).

    Doc2 §8.1: M = 1 / (1 + (Insulin / IC50)^n), IC50 = 10 µU/mL, n = 2.
    Elevated fasting insulin suppresses hormone-sensitive lipase activity,
    directly limiting the rate of intracellular lipolysis.
    """
    p = HILL_KINETICS["insulin_HSL_inhibition"]
    return hill_inhibition(insulin_uiu_ml, p["ic50_uiu_ml"], p["n"])
