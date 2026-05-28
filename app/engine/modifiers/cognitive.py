"""
Cognitive modifier layer — individual transfer functions for cognitive capacity system.

Implements the biochemical modifier cascade for Doc2 §14.1:
    C_cognitiva_real = C_espécie × G_cognitive × ∏ M_k^B (biochemical layer)

Three biochemical modifiers explicitly formulised in the documents:
    M_Hcy_cognitive              — homocysteine as cerebrovascular perfusion modifier
    M_HbA1c_NCV                 — HbA1c as myelin glycation / NCV modifier
    M_glycemic_variability_cognitive — CGM CV% as cognitive function modifier

Myelination modifiers (Doc2 §9.1) are shared with the neuromuscular system
and imported directly from neuromuscular.py in the calculator layer.

Reference: Doc2 §9.1 (NCV/myelination) + Doc2 §14.1 (cognitive integration)
"""

from __future__ import annotations

from app.engine.base import exponential_decay, linear_with_threshold
from app.engine.constants import MODIFIER_PARAMS


def compute_M_Hcy_cognitive(homocysteine_umol_l: float) -> float:
    """M_Hcy_vascular — homocysteine as cerebrovascular perfusion modifier.

    Doc2 §14.1: M = max(0.70, 1.0 − 0.025 × (Hcy − 10)).
    Hcy > 10 µmol/L inhibits eNOS → reduced cerebrovascular perfusion;
    also activates NMDA receptors pathologically → neuronal excitotoxicity.
    MTHFR TT athletes without 5-MTHF supplementation typically show Hcy 14–18.
    """
    p = MODIFIER_PARAMS["M_Hcy_cognitive"]
    return linear_with_threshold(
        homocysteine_umol_l, p["reference"], p["slope"], p["floor"], p["ceiling"]
    )


def compute_M_HbA1c_NCV(hba1c_pct: float) -> float:
    """M_HbA1c_NCV — HbA1c as myelin glycation and nerve conduction velocity modifier.

    Doc2 §9.1: M = 1.0 − 0.08 × (HbA1c − 5.0%).
    Chronic glycation of myelin basic protein (MBP) compromises myelin sheath
    integrity → slower saltatory conduction → increased EMD and reaction time.
    HbA1c 5.8%: M=0.936; HbA1c 6.5%: M=0.880.
    """
    p = MODIFIER_PARAMS["M_HbA1c_NCV"]
    return linear_with_threshold(
        hba1c_pct, p["reference"], p["slope"], p["floor"], p["ceiling"]
    )


def compute_M_glycemic_variability_cognitive(cv_pct: float) -> float:
    """M_glycemic_variability — CGM coefficient of variation as cognitive modifier.

    Doc2 §14.1: M = exp(−0.015 × CV%).
    Glycaemic variability impairs cognitive function more acutely than mean
    glycaemia; each excursion triggers transient cerebral hypoglycaemia or
    hyperglycaemia disrupting prefrontal executive function.
    CV = 40% → M = exp(−0.015 × 40) = 0.549.
    """
    p = MODIFIER_PARAMS["M_glycemic_variability_cognitive"]
    return exponential_decay(cv_pct, p["rate"], p["reference"], p["floor"], p["ceiling"])
