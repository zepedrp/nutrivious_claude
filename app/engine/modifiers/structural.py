"""
Structural modifier layer — individual transfer functions for bone and connective tissue.

compute_M_PINP_CTX_structural implements the anabolic balance ratio
(PINP/PINP_ref) / (CTX/CTX_ref) inline with clamp — no standard primitive covers
the ratio-of-ratios pattern. All reference values come from MODIFIER_PARAMS.
Zero numeric literals live in this file.

Reference: Doc1 §2.6 + Doc1 §3.6 + Doc2 §12.2 + Doc2 §5.2
"""

from __future__ import annotations

from app.engine.base import clamp, exponential_decay, piecewise_linear
from app.engine.constants import MODIFIER_PARAMS


def compute_M_BMD_structural(t_score: float) -> float:
    """M_BMD_structural — bone mineral density T-score as compressive load capacity modifier.

    Doc1 §2.6: compressive strength ∝ ρ² (Gibson relation) — justifies non-linearity.
    T=-4.0→0.60, T=-2.5→0.95, T=-1.0 plateau to 0.0→1.00, T=2.0→1.10.
    """
    p = MODIFIER_PARAMS["M_BMD_structural"]
    return piecewise_linear(t_score, p["x_points"], p["y_points"], p["floor"], p["ceiling"])


def compute_M_VitC_Collagen(vitamin_c_umol_l: float) -> float:
    """M_VitC_Collagen — vitamin C as obligatory cofactor for collagen hydroxylation.

    Doc1 §2.6: prolyl-4-hydroxylase requires vitamin C; insufficient hydroxyproline
    → no pyridinoline cross-links → ultimate tensile strength↓.
    """
    p = MODIFIER_PARAMS["M_VitC_Collagen"]
    return piecewise_linear(
        vitamin_c_umol_l, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


def compute_M_PINP_CTX_structural(pinp_ug_l: float, ctx_ng_ml: float) -> float:
    """M_PINP_CTX_structural — bone turnover balance as net bone formation modifier.

    Doc1 §3.6: M = clamp((PINP / PINP_ref) / (CTX / CTX_ref), 0.60, 1.40).
    ratio > 1.0 → formation > resorption (anabolic bone state).
    PINP_ref=60 µg/L; CTX_ref=0.4 ng/mL.
    """
    p = MODIFIER_PARAMS["M_PINP_CTX_structural"]
    pinp_ratio = pinp_ug_l / p["pinp_reference"]
    ctx_ratio = ctx_ng_ml / p["ctx_reference"]
    return clamp(pinp_ratio / ctx_ratio, p["floor"], p["ceiling"])


def compute_M_VitD_mineralization(vitamin_d_ng_ml: float) -> float:
    """M_VitD_mineralization — vitamin D as bone mineralisation modifier.

    Doc2 §12.2 + §5.2: M = min(1.0, 0.6 + 0.4 × VitD / 60).
    VDR on osteoblasts drives osteocalcin / osteopontin expression; deficiency
    impairs osteoid mineralisation. Floor=0.60 at VitD=0; ceiling=1.00 at ≥60 ng/mL.
    """
    p = MODIFIER_PARAMS["M_VitD_mineralization"]
    return piecewise_linear(
        vitamin_d_ng_ml, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


def compute_M_testosterone_mineralization(testosterone_nmol_l: float) -> float:
    """M_testosterone_mineralization — testosterone as bone mineralisation modifier.

    Doc2 §12.2 + §5.2: M = min(1.0, 0.7 + 0.3 × T / 22) — T in nmol/L.
    Androgen receptors on osteoblasts drive type-I collagen synthesis and
    mineralisation; deficiency shifts balance toward net resorption.
    Floor=0.70 at T=0; ceiling=1.00 at T≥22 nmol/L.
    """
    p = MODIFIER_PARAMS["M_testosterone_mineralization"]
    return piecewise_linear(
        testosterone_nmol_l, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


def compute_M_Inflam_Structural(crp_mg_l: float) -> float:
    """M_Inflam_Structural — chronic inflammation as collagen synthesis suppressor.

    Doc1 §3.6 (listed; no explicit formula). TNF-α / IL-6 via NF-κB suppress
    type-I and type-II collagen synthesis in fibroblasts and chondrocytes.
    Exponential decay with k=0.10 (same as aerobic CRP — structural collagen is
    more resilient than mTORC1/MPS pathway but follows the same kinetic shape).
    Floor=0.70: collagen synthesis never collapses to zero (fibroblast redundancy).
    """
    p = MODIFIER_PARAMS["M_Inflam_Structural"]
    return exponential_decay(crp_mg_l, p["rate"], p["reference"], p["floor"], p["ceiling"])
