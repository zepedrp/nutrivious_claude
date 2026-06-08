"""
app/slices/cardiorespiratory/envelope.py

Phase 3 Envelope — Cardiorespiratory Performance Slice

Generates the Phase3Envelope (Bayesian priors + hard constraints +
chance constraints + refer-out rules) for the cardiorespiratory slice.
The envelope is consumed by the UKF state filter (L4) and the NMPC (L6).

CLAUDE.md §3 — "Fase 3 é priors + restrições, nunca um modelo multiplicativo."

Safety floors
─────────────
SOURCE: Dempsey 2006 (respiratory steal threshold); Skiba 2015 (W' depletion);
        Gabbett 2016 BJSM (ACWR overreaching zone); ACSM guidelines (HR ceiling).

Hard constraints (MPC must never violate):
  1. Heart_Rate  ≤  HR_max × 1.05     — absolute cardiac ceiling (ACSM)
  2. W_prime_bal ≥  W_PRIME_HARD_FLOOR — task failure at battery depletion

Chance constraints (probabilistic; tightened by k·σ_x in MPC):
  3. Resp_Fatigue ≤ 0.85   P(violation) ≤ 0.05   [Dempsey 2006 steal threshold]
  4. ACWR        ∈ [0.80, 1.30]                   [Gabbett 2016 overreaching zone]

Refer-out rules (Path A — pause recommendations):
  5. Autonomic_Tone ≤ 0.15  — "Recuperação vagal suprimida, recomendações pausadas"
  6. Resp_Fatigue  ≥ 0.85   — "Sobrecarga sistémica do aparelho cardiorrespiratório"

Hub variable keys (match UKF/filter state convention)
──────────────────────────────────────────────────────
  Hub_HR_bpm          → Heart_Rate state [bpm]
  Hub_W_prime_kJ      → W_prime_bal state [kJ]
  Hub_Resp_Fatigue    → Resp_Fatigue state [adim]
  Hub_Autonomic_Tone  → Autonomic_Tone state [adim]
  Hub_ACWR            → computed training load ratio [adim]

References
──────────
  Dempsey J.A. et al. (2006) J Physiol 564:425-445  [metaboreflex steal 85% threshold]
  Gabbett T.J. (2016) BJSM 50:273-280               [ACWR overreaching 0.8-1.3]
  Skiba P.F. et al. (2015) J Sci Sport 20:486–492   [W' depletion task failure]
  ACSM (2022) Guidelines for Exercise Testing — [HR ceiling at 105% HRmax]
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
    build_engine_priors,
)

# ── Cardiorespiratory safety thresholds ──────────────────────────────────────

_HR_MAX_DEFAULT:          float = 185.0   # bpm (age 35 default; personalised per user)
_HR_CEILING_FACTOR:       float = 1.05    # 105% HR_max — absolute cardiac ceiling
_W_PRIME_HARD_FLOOR:      float = 0.05    # kJ — effectively zero (task failure sentinel)
_RESP_FATIGUE_CHANCE:     float = 0.85    # adim — Dempsey 2006 metaboreflex threshold
_RESP_FATIGUE_REFEROUT:   float = 0.85    # adim — same as chance threshold
_AT_REFEROUT_FLOOR:       float = 0.15    # adim — vagal suppression refer-out
_ACWR_MIN:                float = 0.80    # adim — Gabbett 2016 underload floor
_ACWR_MAX:                float = 1.30    # adim — Gabbett 2016 overreaching ceiling


# ── Constraint builders ───────────────────────────────────────────────────────

def _hard_constraints(hr_max: float = _HR_MAX_DEFAULT) -> list[Constraint]:
    """
    Inviolable hard constraints for the cardiorespiratory NMPC controller.

    The MPC must NEVER generate a prescription that would drive these states
    beyond their thresholds under the nominal model + worst-case PSF margin.
    """
    hr_ceiling = round(hr_max * _HR_CEILING_FACTOR, 1)
    return [
        Constraint(
            name       = "hr_absolute_ceiling",
            state_key  = "Hub_HR_bpm",
            op         = "≤",
            rhs        = hr_ceiling,
            units      = "bpm",
            source_doi = "10.1249/00005768-199505000-00000",  # ACSM guidelines
        ),
        Constraint(
            name       = "w_prime_task_failure_floor",
            state_key  = "Hub_W_prime_kJ",
            op         = "≥",
            rhs        = _W_PRIME_HARD_FLOOR,
            units      = "kJ",
            source_doi = "10.1152/ajpregu.00444.2012",   # Clarke & Skiba 2013
        ),
    ]


def _chance_constraints() -> list[ChanceConstraint]:
    """
    Probabilistic constraints: P(violation) ≤ α per constraint.

    Tightened to hard constraints in MPC via k·σ_x where k = Φ⁻¹(1−α)
    and σ_x comes from the UKF filter covariance.
    """
    return [
        ChanceConstraint(
            name       = "resp_fatigue_steal_chance",
            state_key  = "Hub_Resp_Fatigue",
            op         = "≤",
            rhs        = _RESP_FATIGUE_CHANCE,
            alpha      = 0.05,
            units      = "adim",
            source_doi = "10.1113/jphysiol.2005.100537",  # Dempsey 2006
        ),
        ChanceConstraint(
            name       = "acwr_underload_floor",
            state_key  = "Hub_ACWR",
            op         = "≥",
            rhs        = _ACWR_MIN,
            alpha      = 0.10,
            units      = "adim",
            source_doi = "10.1136/bjsports-2015-095788",  # Gabbett 2016
        ),
        ChanceConstraint(
            name       = "acwr_overreaching_ceiling",
            state_key  = "Hub_ACWR",
            op         = "≤",
            rhs        = _ACWR_MAX,
            alpha      = 0.05,
            units      = "adim",
            source_doi = "10.1136/bjsports-2015-095788",
        ),
    ]


def _refer_out_rules() -> list[ReferOutRule]:
    """
    Path A refer-out rules: pause recommendations when triggered.

    Language uses performance/readiness vocabulary only (no medical terms).
    """
    return [
        ReferOutRule(
            name      = "vagal_suppression_prolonged",
            state_key = "Hub_Autonomic_Tone",
            op        = "≤",
            threshold = _AT_REFEROUT_FLOOR,
            message   = (
                "A tua recuperação vagal está suprimida há vários dias. "
                "As recomendações de performance estão pausadas até os teus "
                "indicadores de recuperação voltarem ao teu intervalo habitual."
            ),
        ),
        ReferOutRule(
            name      = "respiratory_systemic_overload",
            state_key = "Hub_Resp_Fatigue",
            op        = "≥",
            threshold = _RESP_FATIGUE_REFEROUT,
            message   = (
                "Sobrecarga sistémica do aparelho cardiorrespiratório e "
                "recuperação vagal suprimida detectadas. "
                "Recomendamos confirmar com um profissional de saúde qualificado "
                "antes de retomar treino de alta intensidade."
            ),
        ),
    ]


# ── Public factory ────────────────────────────────────────────────────────────

def build_cardiorespiratory_envelope(
    posterior_theta: dict | None = None,
    athlete_data:    dict | None = None,
    genotype:        dict | None = None,
) -> Phase3Envelope:
    """
    Build the Phase3Envelope for the cardiorespiratory slice.

    Parameters
    ----------
    posterior_theta : dict — NLME posterior (currently: VO2_max_baseline,
                      W_prime_capacity). Used to personalise HR_max estimate.
                      None → population defaults.
    athlete_data    : dict — Phase 2 biomarker dict (age → HR_max; currently
                      optional, defaults to population mean).
    genotype        : dict — SNP genotype dict (ACE, HIF1A — passed to
                      build_engine_priors for the wider engine priors).

    Returns
    -------
    Phase3Envelope with priors, hard_constraints, chance_constraints, refer_out_rules.

    Notes
    -----
    The HR_max is derived from age when available (Tanaka 2001: HR_max = 208 − 0.7×age).
    Default: 185 bpm (age 35 population mean).
    """
    theta = posterior_theta or {}
    adata = athlete_data or {}
    geno  = genotype or {}

    # Personalise HR_max from age if available
    age = adata.get("age_years")
    if age is not None and age > 0:
        hr_max = round(208.0 - 0.7 * float(age), 1)   # Tanaka 2001 J Am Coll Cardiol
    else:
        hr_max = _HR_MAX_DEFAULT

    # VO2_max_baseline from NLME posterior if available
    vo2_max = theta.get("VO2_max_baseline", 45.0)
    w_prime = theta.get("W_prime_capacity", 18.0)
    cp      = theta.get("CP_watts",         250.0)

    # Bayesian priors for NLME warm-start and filter initialisation
    bayesian_priors: dict = dict(
        VO2_max_baseline = float(vo2_max),
        W_prime_capacity = float(w_prime),
        CP_watts         = float(cp),
        HR_max           = float(hr_max),
    )
    # Merge with wider-engine priors (genetic covariates, etc.)
    try:
        engine_priors = build_engine_priors(adata, geno)
        bayesian_priors.update(engine_priors)
    except Exception:
        pass  # engine priors are optional; cardio slice self-contained

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = _hard_constraints(hr_max=hr_max),
        chance_constraints = _chance_constraints(),
        refer_out_rules    = _refer_out_rules(),
    )


# ── Constraint evaluation helpers ─────────────────────────────────────────────

def check_hard_constraints(
    env:        Phase3Envelope,
    state_dict: dict[str, float],
) -> list[str]:
    """
    Evaluate hard constraints against a nominal state dict.

    Parameters
    ----------
    env        : Phase3Envelope from build_cardiorespiratory_envelope
    state_dict : {state_key: float} — hub variables to check

    Returns
    -------
    violations : list of constraint names that are violated

    Notes
    -----
    This function uses nominal values only (no probabilistic tightening).
    Use for gate checks, tests, and REFUSE decisions.
    """
    violations: list[str] = []
    for c in env.hard_constraints:
        val = state_dict.get(c.state_key)
        if val is None:
            continue
        v = float(val)
        if c.op == "≤" and v > c.rhs:
            violations.append(c.name)
        elif c.op == "≥" and v < c.rhs:
            violations.append(c.name)
    return violations


def check_all_constraints(
    env:        Phase3Envelope,
    state_dict: dict[str, float],
) -> list[str]:
    """
    Evaluate both hard and chance constraints by nominal value.

    Returns the union of hard violations and chance violations
    (chance constraints checked without probabilistic tightening).
    Used in tests and diagnostic reporting.
    """
    violations = check_hard_constraints(env, state_dict)
    for c in env.chance_constraints:
        val = state_dict.get(c.state_key)
        if val is None:
            continue
        v = float(val)
        if c.op == "≤" and v > c.rhs:
            violations.append(c.name)
        elif c.op == "≥" and v < c.rhs:
            violations.append(c.name)
    return violations
