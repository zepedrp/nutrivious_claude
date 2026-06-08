"""
app/slices/neural_cognitive/envelope.py  — REMASTER v2.0

Phase 3 Envelope — Neural/Cognitive Slice

Generates the Phase3Envelope (priors + constraints + refer-out rules) for the
neural/cognitive fatigue axis.  Consumed by L4 (UKF) and L6 (MPC/PSF).

CLAUDE.md §3 — "Fase 3 é priors + restrições, nunca um modelo multiplicativo."
CLAUDE.md §1 — Path A: no medical vocabulary in any user-facing message string.

Safety Floors
─────────────
CENTRAL ACTIVATION RATIO (CAR) — BIOMECHANICAL INJURY GATE
  CAR < 0.75 signals neuromuscular coordination degraded to the point where
  biomechanical loading patterns become erratic → acute mechanical injury risk
  escalates non-linearly (Thomas et al. 2015; Gandevia 2001; Amann 2011).

  Hard floor: CAR ≥ 0.75 (any prescribed high-intensity activity paused).
  Chance constraint: P(CAR < 0.80) ≤ 0.10 on 24h planning horizon.

PVT COGNITIVE READINESS
  PVT_Lapses > 4/10 min indicates psychomotor fatigue incompatible with safe
  execution of technical or high-speed training (Van Dongen 2003; Lim & Dinges 2010).
  Chance constraint: P(PVT_Lapses > 4.0) ≤ 0.15.

Personalisation priors
─────────────────────
  dopamine_resilience: COMT Val158Met → prior shift ≤ 0.5·σ (CLAUDE.md §3.2)
  CYP1A2_clearance:   genetic metaboliser phenotype → prior on caffeine half-life
  car_baseline:       onboarding twitch-interpolation or voluntary activation test
  pvt_baseline:       onboarding morning PVT (Dinges & Powell 1985)

References
──────────
  Amann M. et al. (2011) J Physiol 589(10):2467–2476
  Gandevia S.C. (2001) Physiol Rev 81(4):1725–1789
  Lim J., Dinges D.F. (2010) Annu Rev Psychol 61:591–612
  Thomas K. et al. (2015) Eur J Appl Physiol 115(7):1499
  Van Dongen H.P.A. et al. (2003) Sleep 26(2):117–126
"""
from __future__ import annotations

import math

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
)


# ── Safety floor constants ────────────────────────────────────────────────────
# Thomas 2015; Gandevia 2001; Amann 2011
CAR_HARD_FLOOR:   float = 0.75   # [au] hard; CAR < 0.75 → biomechanical injury gate
CAR_CHANCE_FLOOR: float = 0.80   # [au] chance P(CAR < 0.80) ≤ 0.10
CAR_REFER_FLOOR:  float = 0.70   # [au] refer-out trigger

# Lim & Dinges 2010; Van Dongen 2003
PVT_CHANCE_CEILING: float = 4.0   # lapses/10 min; P ≤ 0.15
PVT_REFER_CEILING:  float = 6.0   # lapses/10 min; refer-out trigger

# COMT Val158Met → dopamine_resilience prior shift (CLAUDE.md §3.2: ≤ 0.5·σ)
# σ_prior ≈ 0.20 → max shift = 0.10 (absolute)
_COMT_SHIFTS: dict[str, float] = {
    "Val/Val":  0.90,   # high COMT activity → faster DA degradation → lower resilience
    "Val/Met":  1.00,   # population reference
    "Met/Met":  1.10,   # low COMT → slower DA degradation → higher resilience
}

# CYP1A2 phenotype → caffeine half-life → clearance rate (ln(2)/t½)
_CYP1A2_CLEARANCE: dict[str, float] = {
    "rapid":    math.log(2) / 2.5,    # 0.277 h⁻¹; CYP1A2*1F homozygous fast
    "normal":   math.log(2) / 5.0,    # 0.139 h⁻¹; population mean
    "slow":     math.log(2) / 10.0,   # 0.069 h⁻¹; CYP1A2*1C / *1D / inhibited
}


def build_neural_cognitive_envelope(
    dopamine_resilience_prior:    float = 1.0,
    cyp1a2_clearance_prior:       float = 0.139,
    car_baseline:                 float = 0.95,
    pvt_baseline:                 float = 0.0,
    comt_allele:                  str   = "Val/Met",
    cyp1a2_phenotype:             str   = "normal",
    athlete_data:                 dict  | None = None,
    genotype:                     dict  | None = None,
) -> Phase3Envelope:
    """
    Build the Phase3Envelope for the neural/cognitive slice.

    Parameters
    ----------
    dopamine_resilience_prior : float — prior mean for DA resilience [au]
    cyp1a2_clearance_prior    : float — prior mean for CYP1A2 clearance [h⁻¹]
    car_baseline              : float — onboarding CAR (twitch interpolation; default 0.95)
    pvt_baseline              : float — onboarding PVT pool (0 = fully rested)
    comt_allele               : str   — 'Val/Val' | 'Val/Met' | 'Met/Met'
    cyp1a2_phenotype          : str   — 'rapid' | 'normal' | 'slow'
    athlete_data              : optional onboarding measurements dict
    genotype                  : optional SNP calls dict

    Returns
    -------
    Phase3Envelope with bayesian_priors, hard_constraints, chance_constraints,
    and refer_out_rules.
    """
    athlete_data = athlete_data or {}
    genotype     = genotype     or {}

    # ── Bayesian priors ───────────────────────────────────────────────────────

    # COMT Val158Met → dopamine_resilience shift (CLAUDE.md §3.2; ≤ 0.5·σ)
    comt_key  = genotype.get("COMT_Val158Met", comt_allele)
    da_res    = dopamine_resilience_prior * _COMT_SHIFTS.get(comt_key, 1.00)

    # CYP1A2 phenotype → clearance rate prior
    cyp1a2_pheno = genotype.get("CYP1A2_phenotype", cyp1a2_phenotype)
    cyp1a2_rate  = _CYP1A2_CLEARANCE.get(cyp1a2_pheno, cyp1a2_clearance_prior)

    # Onboarding CAR and PVT from athlete_data if provided
    car_obs = float(athlete_data.get("car_baseline",     car_baseline))
    pvt_obs = float(athlete_data.get("pvt_pool_baseline", pvt_baseline))

    bayesian_priors = {
        "dopamine_resilience":    float(da_res),
        "CYP1A2_clearance_rate":  float(cyp1a2_rate),
        "car_baseline":           car_obs,
        "pvt_pool_baseline":      pvt_obs,
        "comt_allele_shift":      float(_COMT_SHIFTS.get(comt_key, 1.00)),
        "cyp1a2_half_life_h":     float(math.log(2) / (cyp1a2_rate + 1e-9)),
    }

    # ── Hard constraints ──────────────────────────────────────────────────────
    hard_constraints = [
        Constraint(
            name       = "car_hard_floor",
            state_key  = "Hub_NC_CAR",
            op         = "≥",
            rhs        = CAR_HARD_FLOOR,
            units      = "au",
            source_doi = "10.1007/s00421-015-3141-7",   # Thomas 2015
        ),
    ]

    # ── Chance constraints ────────────────────────────────────────────────────
    # k = Φ⁻¹(0.90) ≈ 1.28  (α=10% for CAR)
    # k = Φ⁻¹(0.85) ≈ 1.04  (α=15% for PVT)
    chance_constraints = [
        ChanceConstraint(
            name       = "car_chance_floor",
            state_key  = "Hub_NC_CAR",
            op         = "≥",
            rhs        = CAR_CHANCE_FLOOR,
            alpha      = 0.10,
            units      = "au",
            source_doi = "10.1113/jphysiol.2011.224022",   # Amann 2011
        ),
        ChanceConstraint(
            name       = "pvt_cognitive_ceiling",
            state_key  = "Hub_NC_PVT_Lapses",
            op         = "≤",
            rhs        = PVT_CHANCE_CEILING,
            alpha      = 0.15,
            units      = "count/10 min",
            source_doi = "10.1146/annurev.psych.093008.100416",   # Lim & Dinges 2010
        ),
    ]

    # ── Refer-out rules (Path A — zero medical vocabulary) ───────────────────
    refer_out_rules = [
        ReferOutRule(
            name      = "car_critical_inhibition",
            state_key = "Hub_NC_CAR",
            op        = "≤",
            threshold = CAR_REFER_FLOOR,
            message   = (
                "Your neuromuscular readiness is critically reduced. "
                "The level of central fatigue detected indicates a high risk "
                "of movement quality breaking down under load, which significantly "
                "increases the chance of a sudden mechanical strain. "
                "We have paused high-intensity recommendations and suggest rest "
                "or light movement only until your readiness recovers. "
                "If this persists beyond 24 hours at rest, please check in "
                "with a qualified health professional."
            ),
        ),
        ReferOutRule(
            name      = "pvt_sustained_cognitive_impairment",
            state_key = "Hub_NC_PVT_Lapses",
            op        = "≥",
            threshold = PVT_REFER_CEILING,
            message   = (
                "Your cognitive readiness score indicates significant fatigue "
                "accumulation. Reaction time and coordination under load are "
                "likely impaired at this level, increasing risk during technical "
                "or high-speed training. "
                "We recommend prioritising recovery before your next structured "
                "session. If sleep quality or quantity has been consistently "
                "poor, consider consulting a qualified health professional."
            ),
        ),
    ]

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = hard_constraints,
        chance_constraints = chance_constraints,
        refer_out_rules    = refer_out_rules,
    )
