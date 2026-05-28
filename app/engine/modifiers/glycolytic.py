"""
Glycolytic modifier layer — individual transfer functions for CHO oxidation system.

Each function receives a single biomarker value, pulls its own parameter dict
from MODIFIER_PARAMS, and calls the appropriate base.py primitive.
Zero numeric literals live in this file.

Reference: Doc1 §2.2 + Fase3 §3.2 formula
"""

from __future__ import annotations

import math

from app.engine.base import (
    exponential_decay,
    linear_with_threshold,
    piecewise_linear,
)
from app.engine.constants import MODIFIER_PARAMS


def compute_M_HOMA_CHO(homa_ir: float) -> float:
    """M_HOMA_CHO — insulin resistance as exponential suppressor of CHO oxidation.

    Doc1 §2.2: HOMA-IR ↑ → GLUT4 translocation ↓ + SGLT1 induction ↓ (PI3K/Akt)
    Exponential: each unit above ref compounds the deficit multiplicatively.
    k=0.15: HOMA=3.0 → ~26% penalty; floor=0.40 (type-2 diabetic floor).
    """
    p = MODIFIER_PARAMS["M_HOMA_CHO"]
    return exponential_decay(homa_ir, p["rate"], p["reference"], p["floor"], p["ceiling"])


def compute_M_Zonulin_CHO(zonulin_ng_ml: float) -> float:
    """M_Zonulin_CHO — intestinal permeability as SGLT1 suppression modifier.

    Doc1 §2.2: Elevated zonulin → LPS translocation → TLR4 → NF-κB represses SGLT1.
    k=0.012 per ng/mL above 20; Z_ref=20 ng/mL (intact mucosa).
    """
    p = MODIFIER_PARAMS["M_Zonulin_CHO"]
    return linear_with_threshold(
        zonulin_ng_ml, p["reference"], p["slope"], p["floor"], p["ceiling"]
    )


def compute_M_GutTrain_CHO(gut_training_weeks: float) -> float:
    """M_GutTrain_CHO — cumulative gut CHO training as SGLT1 upregulation modifier.

    Doc1 §2.2: Chronic intraluminal CHO exposure upregulates SGLT1 in jejunal epithelium.
    80% of max upregulation achieved in ~8 weeks. Naïve athletes floor at 0.70.
    """
    p = MODIFIER_PARAMS["M_GutTrain_CHO"]
    return piecewise_linear(
        gut_training_weeks, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


def compute_M_Microbiome_CHO(shannon_diversity: float) -> float:
    """M_Microbiome_CHO — gut microbiome diversity as SCFA/barrier integrity modifier.

    Doc1 §2.2: Shannon index ↓ → butyrate ↓ → colonocyte energy deficit → tight-junction
    compromise → indirect SGLT1 suppression. Plateau at H ≥ 3.5.
    """
    p = MODIFIER_PARAMS["M_Microbiome_CHO"]
    return piecewise_linear(
        shannon_diversity, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


def compute_C_micro_Veillonella(veillonella_pct: float) -> float:
    """C_micro — Veillonella atypica as microbiome lactate-to-propionate co-factor.

    Doc2 §11.1: C_micro = 1.0 + 0.03 × ln(Veillonella% / 0.01%)
    Veillonella converts lactate → propionate → hepatic gluconeogenesis →
    additional energetic substrate captured by muscle during prolonged effort.
    Elite athletes with ~1.0% Veillonella: C_micro ≈ 1.138 (+13.8% on CHO ceiling).
    Floor at 1.00: co-factor adds capacity; abundances below 0.01% yield no bonus.
    """
    if veillonella_pct is None or veillonella_pct <= 0.0:
        return 1.0  # absent or zero abundance → neutral (no ln(0) domain error)
    p = MODIFIER_PARAMS["C_micro_Veillonella"]
    raw = 1.0 + p["coefficient"] * math.log(veillonella_pct / p["reference_pct"])
    return max(p["floor"], raw)


def compute_M_Calprotectin_CHO(calprotectin_ug_g: float) -> float:
    """M_Calprotectin_CHO — intestinal inflammation as mucosal absorptive capacity modifier.

    Doc1 §2.2: Calprotectin is a proxy for neutrophil/macrophage mucosal infiltration.
    Elevated levels signal active enterocyte damage that impairs absorptive surface area.
    k=0.002 per µg/g above 50; floor=0.80.
    """
    p = MODIFIER_PARAMS["M_Calprotectin_CHO"]
    return linear_with_threshold(
        calprotectin_ug_g, p["reference"], p["slope"], p["floor"], p["ceiling"]
    )
