"""
app/slices/gastrointestinal/envelope.py  --  GI Slice V3.0

Phase3Envelope for the V3.0 dual-transporter GI system.
Path A language -- ZERO medical vocabulary in refer-out strings.

Hard constraints
  Hub_Stomach_Volume  <= 1.0 L
  Hub_GI_Distress     <= 8.0

Chance constraint
  Hub_GI_Distress     <= 5.0  (alpha=0.10)

Refer-out rule
  GI_Distress >= 5.0 ->
    "Bloqueio absortivo gastrointestinal detetado por stress isquemico/termico.
     Interromper ingestao."
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
)


def build_gastrointestinal_envelope(
    vmax_glu_prior:    float = 1.0,
    vmax_fru_prior:    float = 0.6,
    vol_tolerance_prior: float = 0.8,
    athlete_data:      dict | None = None,
) -> Phase3Envelope:
    """
    Build Phase3Envelope for GI V3.0 (dual-transporter SGLT1/GLUT5).

    Parameters
    ----------
    vmax_glu_prior     : float [g/min] -- SGLT1 glucose capacity prior
    vmax_fru_prior     : float [g/min] -- GLUT5 fructose capacity prior
    vol_tolerance_prior: float [L]     -- elastic stomach capacity prior
    athlete_data       : optional dict

    Returns
    -------
    Phase3Envelope
    """
    bayesian_priors = {
        "Vmax_glu":       vmax_glu_prior,
        "Vmax_fru":       vmax_fru_prior,
        "Vol_tolerance":  vol_tolerance_prior,
    }

    hard_constraints = [
        Constraint(
            name      = "gastric_volume_ceiling",
            state_key = "Hub_Stomach_Volume",
            op        = "<=",
            rhs       = 1.0,
            units     = "L",
            source_doi= "10.1007/BF00877708",
        ),
        Constraint(
            name      = "gi_distress_hard_ceiling",
            state_key = "Hub_GI_Distress",
            op        = "<=",
            rhs       = 8.0,
            units     = "au",
            source_doi= "10.1249/MSS.0000000000000584",
        ),
    ]

    chance_constraints = [
        ChanceConstraint(
            name      = "gi_distress_chance",
            state_key = "Hub_GI_Distress",
            op        = "<=",
            rhs       = 5.0,
            alpha     = 0.10,
            units     = "au",
            source_doi= "10.1249/MSS.0000000000000584",
        ),
    ]

    refer_out_rules = [
        ReferOutRule(
            name      = "gi_absorptive_block",
            state_key = "Hub_GI_Distress",
            op        = ">=",
            threshold = 5.0,
            message   = (
                "Bloqueio absortivo gastrointestinal detetado por stress "
                "isquemico/termico. Interromper ingestao."
            ),
        ),
    ]

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = hard_constraints,
        chance_constraints = chance_constraints,
        refer_out_rules    = refer_out_rules,
    )
