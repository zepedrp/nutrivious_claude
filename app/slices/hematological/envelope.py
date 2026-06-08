"""
app/slices/hematological/envelope.py  Phase 3 Envelope - Hematological Slice

Hard constraint:
  Hub_Hematocrit_pct <= 55.0  (thrombosis risk ceiling)

Refer-out rules:
  Hct >= 55.0 -> "Risco de hiperviscosidade sanguinea e trombose.
                  Suspender treino de altitude imediatamente."
  Ferritin < 15.0 -> "Esgotamento das reservas de ferro.
                       Anemia desportiva em curso."
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from app.slices.hematological.ode import (
    IDX_FERRITIN,
    HematologicalParams, DEFAULT_HEM_PARAMS,
    compute_hematocrit,
)

HCT_THROMBOSIS_CEILING = 55.0   # % hard ceiling
FERRITIN_DEPLETION_FLOOR = 15.0  # ug/L refer-out threshold


class HematologicalEnvelope(NamedTuple):
    """Evaluation result from check_hematological_envelope."""
    hard_constraint_violated: bool
    refer_out_messages:       list
    hematocrit_pct:           float
    ferritin_ug_L:            float


def check_hematological_envelope(
    x:      jnp.ndarray,
    params: HematologicalParams = DEFAULT_HEM_PARAMS,
) -> HematologicalEnvelope:
    """
    Evaluate hard constraint and refer-out rules from the current state.

    Hard constraint:  Hub_Hematocrit_pct <= 55.0
    Refer-out 1:      Hct >= 55.0 (hyperviscosity / thrombosis risk)
    Refer-out 2:      Ferritin < 15.0 (iron store depletion)
    """
    hct      = float(compute_hematocrit(x))
    ferritin = float(jnp.maximum(x[IDX_FERRITIN], 0.0))

    messages: list[str] = []
    hard_violated = False

    if hct >= HCT_THROMBOSIS_CEILING:
        hard_violated = True
        messages.append(
            "Risco de hiperviscosidade sanguinea e trombose. "
            "Suspender treino de altitude imediatamente."
        )

    if ferritin < FERRITIN_DEPLETION_FLOOR:
        messages.append(
            "Esgotamento das reservas de ferro. "
            "Anemia desportiva em curso."
        )

    return HematologicalEnvelope(
        hard_constraint_violated=hard_violated,
        refer_out_messages=messages,
        hematocrit_pct=hct,
        ferritin_ug_L=ferritin,
    )
