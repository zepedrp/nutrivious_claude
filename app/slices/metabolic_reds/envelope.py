"""
app/slices/metabolic_reds/envelope.py — Module 13

Phase 3 Envelope — Metabolic RED-S / Thyroid / Fatmax Slice

Generates the Phase3Envelope (priors + constraints + refer-out rules) for the
metabolic energy availability and thyroid hormone axis.  Consumed by L4 (UKF)
and L6 (MPC/PSF).

CLAUDE.md §3 — "Fase 3 é priors + restrições, nunca um modelo multiplicativo."
CLAUDE.md §1 — Path A: no medical vocabulary in any user-facing message string.

Safety Floors
─────────────
ENERGY AVAILABILITY — RED-S GATE
  EA < 30 kcal/kg FFM/day is the Loucks threshold beyond which deiodinase
  activity is ≤50% of normal, fT3 drops, bone turnover is suppressed, and
  the neuroendocrine axis enters conservation mode (Loucks 2003;
  Mountjoy 2018 IOC RED-S consensus; De Souza 2014).

  Hard floor: EA_Pool ≥ 30 kcal/kg FFM/day
  Chance constraint: P(EA_Pool < 33) ≤ 0.10 on 7-day planning horizon.

RMR METABOLIC ADAPTATION GATE
  RMR_Multiplier < 0.85 indicates >15% suppression of resting metabolic rate,
  associated with hormonal adaptation, reduced training responsiveness, and
  increased risk of non-functional overreaching in the presence of high load
  (Prentice 1992; Müller 2013; Mountjoy 2018).

  Chance constraint: P(RMR_Multiplier < 0.85) ≤ 0.05.
  Refer-out trigger: RMR_Multiplier < 0.80 (≥20% suppression).

FATMAX MONITORING
  Fatmax_State < 0.80 indicates >20% reduction in fat oxidation capacity,
  constraining substrate availability at sub-maximal intensities.
  Chance constraint: P(Fatmax_State < 0.80) ≤ 0.10.

References
──────────
  Achten J., Jeukendrup A.E. (2003) Sports Med 33(8):559–591
  De Souza M.J. et al. (2014) Br J Sports Med 48(4):289–300
  Loucks A.B. (2003) J Sports Sci 21(10):879–883
  Mountjoy M. et al. (2018) Br J Sports Med 52(11):687–697 [IOC RED-S consensus]
  Müller M.J. et al. (2013) Obes Rev 14(11):908–921
  Prentice A.M. et al. (1992) Eur J Clin Nutr 46:S91–S98
  SPINA-Dietrich (2015) Front Endocrinol 6:57
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
)


# ── Safety floor constants ────────────────────────────────────────────────────
# Loucks 2003; Mountjoy 2018 IOC RED-S consensus
EA_HARD_FLOOR:   float = 30.0   # [kcal/kg FFM/day] hard; deiodinase collapse threshold
EA_CHANCE_FLOOR: float = 33.0   # [kcal/kg FFM/day] chance P(EA < 33) ≤ 0.10
EA_REFER_FLOOR:  float = 25.0   # [kcal/kg FFM/day] refer-out trigger

# RMR adaptation thresholds (Prentice 1992; Müller 2013)
RMR_CHANCE_FLOOR: float = 0.85  # [au] P(RMR_Mult < 0.85) ≤ 0.05 → severe adaptation
RMR_REFER_FLOOR:  float = 0.80  # [au] refer-out trigger (>=20% suppression)

# Fatmax monitoring (Achten & Jeukendrup 2003)
FATMAX_CHANCE_FLOOR: float = 0.80   # [au] P(Fatmax < 0.80) ≤ 0.10


def build_metabolic_reds_envelope(
    ea_baseline:          float = 45.0,
    fT3_baseline:         float = 5.0,
    fT4_baseline:         float = 16.0,
    rmr_multiplier_prior: float = 1.0,
    fatmax_prior:         float = 1.0,
    athlete_data:         dict  | None = None,
) -> Phase3Envelope:
    """
    Build the Phase3Envelope for the metabolic RED-S / thyroid / Fatmax slice.

    Parameters
    ----------
    ea_baseline          : float — onboarding/estimated habitual EA [kcal/kg FFM/day]
    fT3_baseline         : float — onboarding lab fT3 [pmol/L]
    fT4_baseline         : float — onboarding lab fT4 [pmol/L]
    rmr_multiplier_prior : float — prior mean for RMR Multiplier [au]
    fatmax_prior         : float — prior mean for Fatmax State [au]
    athlete_data         : optional dict with measured onboarding values

    Returns
    -------
    Phase3Envelope with bayesian_priors, hard_constraints, chance_constraints,
    and refer_out_rules.
    """
    athlete_data = athlete_data or {}

    # ── Bayesian priors ───────────────────────────────────────────────────────
    ea_obs  = float(athlete_data.get("ea_habitual",          ea_baseline))
    fT3_obs = float(athlete_data.get("fT3_baseline",         fT3_baseline))
    fT4_obs = float(athlete_data.get("fT4_baseline",         fT4_baseline))
    rmr_obs = float(athlete_data.get("rmr_multiplier_prior", rmr_multiplier_prior))
    fat_obs = float(athlete_data.get("fatmax_prior",         fatmax_prior))

    bayesian_priors = {
        "EA_Pool_baseline":        ea_obs,
        "Free_T3_baseline":        fT3_obs,
        "Free_T4_baseline":        fT4_obs,
        "RMR_Multiplier_prior":    rmr_obs,
        "Fatmax_State_prior":      fat_obs,
        # Fixed structural parameters (non-identifiable per-subject; locked to prior)
        "K_deio":                  30.0,     # [kcal/kg FFM] Loucks 2003
        "fT3_ref":                  5.0,     # [pmol/L] Brent & Larsen 2013
        "fT4_ref":                 16.0,     # [pmol/L] Brent & Larsen 2013
        "RMR_ref_kcal_day":      1700.0,     # [kcal/day] Harris-Benedict population mean
    }

    # ── Hard constraints ──────────────────────────────────────────────────────
    hard_constraints = [
        Constraint(
            name       = "ea_red_s_floor",
            state_key  = "Hub_MR_EA_Pool",
            op         = "≥",
            rhs        = EA_HARD_FLOOR,
            units      = "kcal/kg FFM/day",
            source_doi = "10.1080/02640410310001632169",    # Loucks 2003
        ),
    ]

    # ── Chance constraints ────────────────────────────────────────────────────
    # k = Φ⁻¹(0.90) ≈ 1.28  (α=10% for EA)
    # k = Φ⁻¹(0.95) ≈ 1.64  (α=5% for RMR)
    # k = Φ⁻¹(0.90) ≈ 1.28  (α=10% for Fatmax)
    chance_constraints = [
        ChanceConstraint(
            name       = "ea_chance_floor",
            state_key  = "Hub_MR_EA_Pool",
            op         = "≥",
            rhs        = EA_CHANCE_FLOOR,
            alpha      = 0.10,
            units      = "kcal/kg FFM/day",
            source_doi = "10.1136/bjsports-2018-099193",   # Mountjoy 2018
        ),
        ChanceConstraint(
            name       = "rmr_adaptation_floor",
            state_key  = "Hub_MR_RMR_Multiplier",
            op         = "≥",
            rhs        = RMR_CHANCE_FLOOR,
            alpha      = 0.05,
            units      = "au",
            source_doi = "10.1007/BF01232350",             # Prentice 1992
        ),
        ChanceConstraint(
            name       = "fatmax_chance_floor",
            state_key  = "Hub_MR_Fatmax_State",
            op         = "≥",
            rhs        = FATMAX_CHANCE_FLOOR,
            alpha      = 0.10,
            units      = "au",
            source_doi = "10.2165/00007256-200333080-00003",  # Achten & Jeukendrup 2003
        ),
    ]

    # ── Refer-out rules (Path A — zero medical vocabulary) ───────────────────
    refer_out_rules = [
        ReferOutRule(
            name      = "ea_fuel_deficit_critical",
            state_key = "Hub_MR_EA_Pool",
            op        = "≤",
            threshold = EA_REFER_FLOOR,
            message   = (
                "Your metabolic readiness score indicates a sustained state of "
                "fuel conservation. Your body's available energy is below the "
                "level needed to fully support training adaptation, performance, "
                "and hormonal regulation. "
                "We have paused high-intensity and high-volume session "
                "recommendations until your energy balance recovers. "
                "Prioritise consistent fuelling across the day, especially "
                "before and after sessions. "
                "If this pattern has persisted for more than a few days or is "
                "accompanied by fatigue, mood changes, or disrupted sleep, we "
                "recommend checking in with a qualified health professional."
            ),
        ),
        ReferOutRule(
            name      = "rmr_metabolic_conservation",
            state_key = "Hub_MR_RMR_Multiplier",
            op        = "≤",
            threshold = RMR_REFER_FLOOR,
            message   = (
                "Your metabolic rate indicator shows a significant conservation "
                "response compared to your personal baseline — your body is "
                "currently operating at a reduced energy output level. "
                "This level of metabolic adaptation is associated with reduced "
                "responsiveness to training stimulus and slower recovery between "
                "sessions. "
                "We recommend increasing your daily fuelling and reducing "
                "overall training load until your metabolic readiness score "
                "returns to its normal range. "
                "If fatigue, reduced motivation, or changes in body composition "
                "have been present alongside this pattern, consider speaking "
                "with a qualified health professional."
            ),
        ),
    ]

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = hard_constraints,
        chance_constraints = chance_constraints,
        refer_out_rules    = refer_out_rules,
    )
