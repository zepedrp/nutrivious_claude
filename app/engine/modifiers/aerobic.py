"""
Aerobic modifier layer — individual transfer functions for VO2max system.

Each function receives a single biomarker value, pulls its own parameter dict
from MODIFIER_PARAMS, and calls the appropriate base.py primitive.
Zero numeric literals live in this file.

Reference: Doc1 §2.1 (primary formulas) + Doc2 §5.2 (extended panel)
"""

from __future__ import annotations

from app.engine.base import (
    exponential_decay,
    lean_fraction_modifier,
    linear_deviation,
    linear_ratio,
    linear_with_threshold,
    piecewise_linear,
    power_law,
)

from app.engine.constants import MODIFIER_PARAMS


# ---------------------------------------------------------------------------
# Doc1 §2.1 — Primary aerobic modifier set
# ---------------------------------------------------------------------------

def compute_M_Hb(hemoglobin_g_dl: float, sex: str) -> float:
    """M_Hb — hemoglobin as O2 transport modifier (Fick equation, linear).

    Doc1 §2.1: M_Hb = clamp(Hb / Hb_ref, 0.75, 1.15)
    Hb_ref = 15.5 g/dL (male) | 13.5 g/dL (female)
    """
    key = "M_Hb_male" if sex == "male" else "M_Hb_female"
    p = MODIFIER_PARAMS[key]
    return linear_ratio(hemoglobin_g_dl, p["reference"], p["floor"], p["ceiling"])


def compute_M_BF_aerobic(body_fat_fraction: float, sex: str) -> float:
    """M_BF — body fat fraction as inert mass cost on VO2max relative.

    Doc1 §2.1: M_BF = (1 - BF%) / (1 - BF%_ref)
    body_fat_fraction in [0, 1], e.g. 0.19 for 19% body fat.
    ref = 0.12 (male) | 0.20 (female)
    """
    key = "M_BF_aerobic_male" if sex == "male" else "M_BF_aerobic_female"
    p = MODIFIER_PARAMS[key]
    return lean_fraction_modifier(body_fat_fraction, p["ref_bf_pct"], p["floor"], p["ceiling"])


def compute_M_O3_aerobic(omega3_index_pct: float) -> float:
    """M_O3 — omega-3 index as mitochondrial membrane efficiency modifier.

    Doc1 §2.1: cardiolipin DHA content → CTE electron transfer efficiency.
    Plateau at O3 ≥ 8%; marginal benefit 8–12%; ceiling 1.02 at ≥ 12%.
    """
    p = MODIFIER_PARAMS["M_O3_aerobic"]
    return piecewise_linear(omega3_index_pct, p["x_points"], p["y_points"], p["floor"], p["ceiling"])


def compute_M_VitD_aerobic(vitamin_d_ng_ml: float) -> float:
    """M_VitD — vitamin D as multi-target aerobic modifier.

    Doc1 §2.1: VDR → EPO sensitivity + cardiac VDR + PGC-1α promoter.
    Captures residual effect not covered by M_Hb.
    """
    p = MODIFIER_PARAMS["M_VitD_aerobic"]
    return piecewise_linear(vitamin_d_ng_ml, p["x_points"], p["y_points"], p["floor"], p["ceiling"])


def compute_M_CRP_aerobic(crp_mg_l: float) -> float:
    """M_CRP — hsCRP as chronic inflammation penalty on VO2max.

    Doc1 §2.1: log-linear suppression of erythropoiesis + endothelial eNOS.
    k = 0.10; reference = 0.3 mg/L (minimum baseline inflammation).
    """
    p = MODIFIER_PARAMS["M_CRP_aerobic"]
    return exponential_decay(crp_mg_l, p["rate"], p["reference"], p["floor"], p["ceiling"])


def compute_M_Fe_aerobic(stfr_logferritin_ratio: float) -> float:
    """M_Fe — sTfR/log(Ferritin) ratio as functional iron deficiency modifier.

    Doc1 §2.1: ratio immune to inflammation masking of ferritin.
    Penalises functional iron deficiency even when ferritin appears normal.
    """
    p = MODIFIER_PARAMS["M_Fe_aerobic"]
    return piecewise_linear(
        stfr_logferritin_ratio, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


# ---------------------------------------------------------------------------
# Doc2 §5.2 — Extended panel (alternative / complementary formulations)
# ---------------------------------------------------------------------------

def compute_M_ferritin_vo2(ferritin_ug_l: float) -> float:
    """M_ferritin — ferritin as hematopoiesis modifier (Doc2 §5.2 power-law).

    sqrt relationship: M = min(1.0, sqrt(ferritin / 80)).
    Easier to obtain than the sTfR/logFerritin ratio.
    """
    p = MODIFIER_PARAMS["M_ferritin_vo2_doc2"]
    return power_law(ferritin_ug_l, p["reference"], p["exponent"], p["floor"], p["ceiling"])


def compute_M_HOMA_vo2(homa_ir: float) -> float:
    """M_HOMA — insulin resistance as cardiac output efficiency modifier (Doc2 §5.2).

    Insulin resistance compromises ventricular stroke volume and mitochondrial
    substrate flexibility relevant to VO2max even before the Fatmax pathway.
    """
    p = MODIFIER_PARAMS["M_HOMA_vo2_doc2"]
    return linear_with_threshold(homa_ir, p["reference"], p["slope"], p["floor"], p["ceiling"])


def compute_M_cortisol_dhea_ratio_vo2(cortisol_dhea_ratio: float) -> float:
    """M_cortisol_DHEA — HPA axis balance as aerobic capacity modifier (Doc2 §5.2).

    Elevated ratio signals chronic catabolic dominance suppressing mitochondrial
    biogenesis and red cell production.
    """
    p = MODIFIER_PARAMS["M_cortisol_dhea_ratio_vo2_doc2"]
    return linear_with_threshold(
        cortisol_dhea_ratio, p["reference"], p["slope"], p["floor"], p["ceiling"]
    )


def compute_M_DunedinPACE(pace: float) -> float:
    """M_DunedinPACE — epigenetic aging rate as systemic performance modifier (Doc2 §5.2).

    pace = 1.0 → neutral; pace > 1.0 → accelerated aging → ceiling reduction.
    """
    p = MODIFIER_PARAMS["M_DunedinPACE"]
    return linear_deviation(pace, p["reference"], p["slope"], p["floor"], p["ceiling"])


def compute_M_T3_vo2(t3_free_pmol_l: float) -> float:
    """M_T3 — free T3 as mitochondrial biogenesis modifier via PGC-1α (Doc2 §5.2).

    T3 activates PGC-1α, PPAR-α and UCP1 transcription.
    rT3 competition captured indirectly via FT3 level.
    """
    p = MODIFIER_PARAMS["M_T3_vo2_doc2"]
    return power_law(t3_free_pmol_l, p["reference"], p["exponent"], p["floor"], p["ceiling"])


def compute_M_testosterone_vo2_male(testosterone_nmol_l: float) -> float:
    """M_testosterone — testosterone as VO2max anabolic support (Doc2 §5.2, male only).

    Acts via mitochondrial biogenesis and erythropoiesis augmentation.
    Male-specific modifier; female athletes use 1.0 (not applicable).
    """
    p = MODIFIER_PARAMS["M_testosterone_vo2_male_doc2"]
    return piecewise_linear(
        testosterone_nmol_l, p["x_points"], p["y_points"], p["floor"], p["ceiling"]
    )


# ---------------------------------------------------------------------------
# Doc1 §3.1 — Epigenetic layer E (semi-annual update frequency)
# ---------------------------------------------------------------------------

def compute_E_PGC1A_methylation(ppargc1a_methylation_pct: float) -> float:
    """E_PGC1A — PPARGC1A promoter methylation as epigenetic aerobic modifier.

    Doc1 §3.1: E_PGC1A = max(0.75, 1.0 − 0.005 × max(0, methylation% − 20)).
    High methylation silences PGC-1α → attenuated mitochondrial biogenesis
    response to training, compressing both VO2max ceiling and trainability.
    Camada E (epigenética): updated semi-annually, distinct from M-layer biomarkers.
    """
    p = MODIFIER_PARAMS["E_PGC1A_methylation"]
    return linear_with_threshold(
        ppargc1a_methylation_pct, p["reference"], p["slope"], p["floor"], p["ceiling"]
    )
