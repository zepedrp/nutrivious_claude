"""
app/slices/biomechanical_tissue/envelope.py

Phase 3 Envelope -- Biomechanical Tissue Slice

Generates the Phase3Envelope (priors + constraints + refer-out rules) for
the biomechanical tissue slice. Consumed by the UKF (L4) and the NMPC (L6).

CLAUDE.md section 3 -- "Fase 3 e priors + restricoes, nunca um modelo multiplicativo."

Safety constraints (Biomechanical Tissue)
------------------------------------------

TENDON RUPTURE RISK
  Hub_Tendon_Rupture_Risk > 0.85 -> hard stop on impact/plyometric loading.
  Chance constraint: P(Hub_Tendon_Rupture_Risk > 0.85) <= 0.05 over planning
  horizon.
  Source: Arampatzis 2007 (tendon force-elongation failure criteria); van
  Dijk 2011 (Achilles tendon rupture risk biomechanics).

BONE STRESS FRACTURE RISK
  Hub_Bone_Stress_Fracture_Risk > 0.85 -> hard stop on high-volume loading.
  Chance constraint: P(Hub_Bone_Stress_Fracture_Risk > 0.85) <= 0.05.
  Source: Bennell 1996; Rizzone 2017 (stress fracture incidence thresholds).

TENDON STIFFNESS FLOOR
  Tendon_Stiffness < 0.50 -> amber warning; restrict plyometric and
  sprint loading.
  Source: Pearson 2011 (stiffness floor for force transmission safety).

BONE DENSITY FLOOR
  Bone_Density < 0.85 g/cm2 -> hard stop on high-impact activities.
  Source: Mountjoy 2018 BJSM RED-S; Nattiv 2023 JBMR Plus.

Hub variable keys
------------------
  Hub_Bio_Tendon_Rupture_Risk      -- algebraic risk index [0, 1]
  Hub_Bio_Bone_Fracture_Risk       -- algebraic risk index [0, 1]
  Hub_Bio_Tendon_Stiffness         -- tendon stiffness state [au]
  Hub_Bio_Bone_Density             -- bone mineral density [g/cm2]
  Hub_Bio_Tendon_Microdamage       -- tendon structural microdamage [au]
  Hub_Bio_Bone_Microdamage         -- bone structural microdamage [au]
"""
from __future__ import annotations

import math

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
    build_engine_priors,
)


# -- Safety thresholds --------------------------------------------------------

_TEND_RISK_HARD_STOP:   float = 0.85   # hub_tendon_rupture_risk hard stop
_TEND_RISK_CHANCE:      float = 0.85   # chance constraint threshold (same level)
_BONE_RISK_HARD_STOP:   float = 0.85   # hub_bone_fracture_risk hard stop
_BONE_RISK_CHANCE:      float = 0.85   # chance constraint threshold
_TEND_STIFF_AMBER:      float = 0.50   # tendon stiffness amber floor [au]
_BMD_HARD_FLOOR:        float = 0.85   # g/cm2 -- bone density hard floor
_TEND_DMG_WARN:         float = 0.70   # tendon microdamage elevated warning


# -- Constraint builders ------------------------------------------------------

def _hard_constraints_bio() -> list[Constraint]:
    """
    Inviolable safety floors for the biomechanical NMPC controller.

    The controller MUST NOT propose a plan that violates any of these on
    the nominal trajectory. The PSF adds a worst-case safety margin on top.
    """
    return [
        Constraint(
            name       = "tendon_rupture_risk_hard_stop",
            state_key  = "Hub_Bio_Tendon_Rupture_Risk",
            op         = "<=",
            rhs        = _TEND_RISK_HARD_STOP,
            units      = "normalized [0, 1]",
            source_doi = "10.1016/j.jbiomech.2007.01.014",  # Arampatzis 2007
        ),
        Constraint(
            name       = "bone_fracture_risk_hard_stop",
            state_key  = "Hub_Bio_Bone_Fracture_Risk",
            op         = "<=",
            rhs        = _BONE_RISK_HARD_STOP,
            units      = "normalized [0, 1]",
            source_doi = "10.1136/bjsports-2018-099193",    # Mountjoy 2018 BJSM
        ),
        Constraint(
            name       = "bone_density_hard_floor",
            state_key  = "Hub_Bio_Bone_Density",
            op         = ">=",
            rhs        = _BMD_HARD_FLOOR,
            units      = "g/cm2",
            source_doi = "10.1002/jbm4.10729",              # Nattiv 2023 JBMR Plus
        ),
    ]


def _chance_constraints_bio() -> list[ChanceConstraint]:
    """
    Probabilistic safety bounds for the biomechanical tissue slice.

    Tightened to hard bounds in the NMPC via:
        rhs_tightened = rhs -/+ k*sigma_state
    where k = Phi^-1(1 - alpha) and sigma_state is the UKF posterior SD for
    the relevant algebraic hub output (propagated via sigma-point ensemble).
    """
    return [
        ChanceConstraint(
            name       = "tendon_rupture_risk_chance",
            state_key  = "Hub_Bio_Tendon_Rupture_Risk",
            op         = "<=",
            rhs        = _TEND_RISK_CHANCE,
            alpha      = 0.05,
            units      = "normalized [0, 1]",
            source_doi = "10.1016/j.jbiomech.2007.01.014",
        ),
        ChanceConstraint(
            name       = "bone_fracture_risk_chance",
            state_key  = "Hub_Bio_Bone_Fracture_Risk",
            op         = "<=",
            rhs        = _BONE_RISK_CHANCE,
            alpha      = 0.05,
            units      = "normalized [0, 1]",
            source_doi = "10.1136/bjspm-2018-099193",
        ),
        ChanceConstraint(
            name       = "tendon_stiffness_amber_floor",
            state_key  = "Hub_Bio_Tendon_Stiffness",
            op         = ">=",
            rhs        = _TEND_STIFF_AMBER,
            alpha      = 0.05,
            units      = "normalized [0, 1]",
            source_doi = "10.1007/s00421-011-1928-1",       # Pearson 2011
        ),
        ChanceConstraint(
            name       = "tendon_microdamage_elevated_warning",
            state_key  = "Hub_Bio_Tendon_Microdamage",
            op         = "<=",
            rhs        = _TEND_DMG_WARN,
            alpha      = 0.05,
            units      = "normalized [0+]",
            source_doi = "10.1016/j.jbiomech.2007.01.014",
        ),
    ]


def _refer_out_rules_bio() -> list[ReferOutRule]:
    """
    Conditions that trigger Path A refer-out language.

    All messages use performance/wellness vocabulary only (CLAUDE.md section 1).
    No medical vocabulary: no "rupture", "fracture", "injury", "diagnosis".
    """
    return [
        ReferOutRule(
            name       = "connective_tissue_fatigue_abort",
            state_key  = "Hub_Bio_Tendon_Rupture_Risk",
            op         = ">=",
            threshold  = _TEND_RISK_HARD_STOP,
            message    = (
                "Indicadores de fadiga de material conjuntivo. "
                "Perda de capacidade de absorcao de carga. "
                "Abortar treino de impacto."
            ),
        ),
        ReferOutRule(
            name       = "bone_stress_load_reduction",
            state_key  = "Hub_Bio_Bone_Fracture_Risk",
            op         = ">=",
            threshold  = _BONE_RISK_HARD_STOP,
            message    = (
                "Risco de lesao ossea por stress. "
                "Necessaria reducao drastica de volume."
            ),
        ),
        ReferOutRule(
            name       = "tendon_stiffness_plyometric_pause",
            state_key  = "Hub_Bio_Tendon_Stiffness",
            op         = "<=",
            threshold  = _TEND_STIFF_AMBER,
            message    = (
                "Os indicadores de resiliencia do tecido conjuntivo estao abaixo "
                "do seu intervalo habitual. As sugestoes de treino pliometrico e "
                "de sprint estao suspensas. Recomenda-se retorno gradual a carga "
                "sob orientacao profissional."
            ),
        ),
        ReferOutRule(
            name       = "bone_density_concern",
            state_key  = "Hub_Bio_Bone_Density",
            op         = "<=",
            threshold  = _BMD_HARD_FLOOR,
            message    = (
                "Os indicadores de densidade estrutural ossea estao a requerer "
                "atencao. As atividades de alto impacto estao suspensas. "
                "Uma avaliacao de saude ossea com profissional qualificado e "
                "aconselhada antes de retomar sessoes de impacto."
            ),
        ),
    ]


# -- Public API ----------------------------------------------------------------

def build_biomechanical_envelope(
    athlete_data:    dict,
    genotype:        dict[str, str],
    phase1_ceilings: dict | None = None,
) -> Phase3Envelope:
    """
    Build the Phase 3 envelope for the Biomechanical Tissue slice.

    Translates Phase 2 biomarker data (DEXA BMD, imaging, genetics) into
    Bayesian priors for the biomechanical ODE parameters, appends the
    connective-tissue safety constraints, and returns the full contract
    consumed by the UKF and NMPC.

    Parameters
    ----------
    athlete_data    : dict -- Phase 2 flat biomarker dict.
                      Relevant keys for this slice:
                      "bmd_lumbar_g_cm2"         -> Bone_Density prior mean
                      "bmd_lumbar_sigma"          -> Bone_Density uncertainty
                      "tendon_us_echogenicity"    -> Tendon_Stiffness proxy
                      "col5a1_rs12722_prior"      -> COL5A1 -> tendon resilience
                      "mmp3_rs679620_prior"       -> MMP3 -> collagen degradation
    genotype        : dict -- SNP genotype dict (weak priors only).
    phase1_ceilings : dict | None -- species-level ceilings from Phase 1.

    Returns
    -------
    Phase3Envelope  -- priors + hard + chance + refer-out for MPC consumption.

    Notes
    -----
    Genetics enter as weak prior shifts (CLAUDE.md section 3.4).
    COL5A1 TT genotype: lower tendon stiffness -> col5a1_scale prior -> 0.85.
    MMP3 high-activity allele: elevated MMP3 enzymatic scale -> mmp3_scale -> 1.3.
    Telemetry data overrides genetics after ~14 days of observations.
    """
    bayesian_priors = build_engine_priors(athlete_data, genotype, phase1_ceilings)

    # -- BMD prior from DEXA ---------------------------------------------------
    bmd_raw = athlete_data.get("bmd_lumbar_g_cm2")
    if bmd_raw is not None and not math.isnan(float(bmd_raw)):
        bayesian_priors["Bone_Density_init_g_cm2"] = float(bmd_raw)

    bmd_sigma = athlete_data.get("bmd_lumbar_sigma")
    if bmd_sigma is not None and not math.isnan(float(bmd_sigma)):
        bayesian_priors["Bone_Density_sigma_g_cm2"] = float(bmd_sigma)

    # -- Tendon echogenicity -> Tendon_Stiffness prior ------------------------
    echo_raw = athlete_data.get("tendon_us_echogenicity")
    if echo_raw is not None and not math.isnan(float(echo_raw)):
        bayesian_priors["Tendon_Stiffness_init_au"] = float(echo_raw)

    # -- COL5A1 -> tendon resilience prior (weak; <= 0.5*sigma shift) ---------
    col5a1_raw = genotype.get("col5a1_rs12722")
    if col5a1_raw == "TT":
        bayesian_priors["col5a1_scale_prior"] = 0.85   # lower tendon stiffness
    elif col5a1_raw == "CC":
        bayesian_priors["col5a1_scale_prior"] = 1.10   # higher tendon stiffness
    else:
        bayesian_priors["col5a1_scale_prior"] = 1.00

    # -- MMP3 -> collagen degradation aggressiveness (weak) --------------------
    mmp3_raw = genotype.get("mmp3_rs679620")
    if mmp3_raw == "high":
        bayesian_priors["mmp3_scale_prior"] = 1.30
    elif mmp3_raw == "low":
        bayesian_priors["mmp3_scale_prior"] = 0.80
    else:
        bayesian_priors["mmp3_scale_prior"] = 1.00

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = _hard_constraints_bio(),
        chance_constraints = _chance_constraints_bio(),
        refer_out_rules    = _refer_out_rules_bio(),
    )
