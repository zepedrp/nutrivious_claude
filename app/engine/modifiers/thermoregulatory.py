"""
Thermoregulatory modifier layer — individual transfer functions for
the thermoregulatory / sudation capacity system.

Implements two biochemical modifiers from Doc2 §10.1–10.2:
    M_dehydration_thermoreg  — % body weight loss as sweat rate modifier
    M_acclimation_thermoreg  — heat acclimatisation status (binary lookup)

Reference: Doc2 §10.1–10.2
"""

from __future__ import annotations

from app.engine.base import linear_with_threshold
from app.engine.constants import MODIFIER_PARAMS


def compute_M_dehydration_thermoreg(body_weight_loss_pct: float) -> float:
    """M_dehydration — % body weight lost as thermoregulatory capacity modifier.

    Doc2 §10.1: M = 1.0 − 0.06 × (% weight loss).
    Dehydration reduces plasma volume and cutaneous perfusion; each 1% BW loss
    in water reduces sudation capacity by 6% and raises core temperature by ~0.5°C.
    3% dehydration (typical 90 min without fluid): M = 1.0 − 0.18 = 0.82.
    """
    p = MODIFIER_PARAMS["M_dehydration_thermoreg"]
    return linear_with_threshold(
        body_weight_loss_pct, p["reference"], p["slope"], p["floor"], p["ceiling"]
    )


def compute_M_acclimation_thermoreg(is_acclimatized: bool) -> float:
    """M_acclimation^HSP — heat acclimatisation status as thermoregulatory ceiling modifier.

    Doc2 §10.2: acclimatised (≥10–14 days at 35–40°C) → M=1.00;
    not acclimatised → M=0.95.
    Acclimatisation induces HSP70/HSP90 synthesis (raising protein thermal tolerance),
    plasma volume expansion (+15–20%), and earlier sweat onset (lower threshold temp).
    In hot-environment performance (38°C), this 5% gap separates completion from collapse.
    """
    p = MODIFIER_PARAMS["M_acclimation_thermoreg"]
    return float(p["acclimatized"] if is_acclimatized else p["not_acclimatized"])
