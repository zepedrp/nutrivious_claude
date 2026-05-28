"""
Unit converter — US lab report format → SI (Sistema Internacional).

Converts US-pattern keys to SI-normalized keys and injects them into the
output dict. Original keys are preserved; SI keys are added or overwritten.

Conversion factors (exact, per NIST / clinical chemistry references):
    glucose_mg_dl   ÷ 18.015  → glucose_mmol_l      (MW glucose = 180.15 g/mol)
    vitamin_d_ng_ml × 2.496   → vitamin_d_nmol_l     (MW 25-OH-D3 = 384.64 g/mol → factor 2.496)
    testosterone_ng_dl ÷ 28.84 → testosterone_nmol_l  (MW testosterone = 288.4 g/mol → ÷28.84)

Reference: Sprint 0.1A — SI normalization boundary layer.
"""

from __future__ import annotations

_GLUCOSE_MG_DL_TO_MMOL_L: float = 18.015
_VITAMIN_D_NG_ML_TO_NMOL_L: float = 2.496
_TESTOSTERONE_NG_DL_TO_NMOL_L: float = 28.84


def normalize_to_si(raw_data: dict) -> dict:
    """
    Inject SI-normalized keys into a copy of raw_data.

    Only keys present in raw_data are converted; missing keys are silently
    skipped (no KeyError, no invented values). The returned dict is always
    a shallow copy — the input dict is never mutated.

    Conversions applied:
        "glucose_mg_dl"      → "glucose_mmol_l"
        "vitamin_d_ng_ml"    → "vitamin_d_nmol_l"
        "testosterone_ng_dl" → "testosterone_nmol_l"

    Args:
        raw_data: dict of raw biomarker values (may contain US or SI keys).

    Returns:
        New dict with all original keys plus any SI-converted keys.
    """
    out = dict(raw_data)

    if (glucose_mg := raw_data.get("glucose_mg_dl")) is not None:
        out["glucose_mmol_l"] = glucose_mg / _GLUCOSE_MG_DL_TO_MMOL_L

    if (vd_ng := raw_data.get("vitamin_d_ng_ml")) is not None:
        out["vitamin_d_nmol_l"] = vd_ng * _VITAMIN_D_NG_ML_TO_NMOL_L

    if (testo_ng := raw_data.get("testosterone_ng_dl")) is not None:
        out["testosterone_nmol_l"] = testo_ng / _TESTOSTERONE_NG_DL_TO_NMOL_L

    return out
