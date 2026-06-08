"""
app/slices/gonadal_axis/envelope.py

Phase 3 Envelope — Gonadal Axis Slice (polymorphic: female / male)

CLAUDE.md §3 — "Fase 3 é priors + restrições, nunca um modelo multiplicativo."

No hard constraints: endocrine failure is not instantaneous; the ODE degrades
gracefully over days-to-weeks. Safety is expressed as chance constraints and
refer-out rules.

Chance constraints
──────────────────
Female: Hub_Estradiol ≥ 30 pg/mL (P(violation) ≤ 0.10)
  Below this threshold: bone remodelling shifts to net loss.
  Khosla 2001 (J Bone Miner Res): E2 < 30 pg/mL → accelerated resorption.
  Prior 2006 (Clin Rev Bone Miner Metab): FHA → 2.5–3% annual BMD loss.

Male: Hub_Testosterone ≥ 300 ng/dL (P(violation) ≤ 0.10)
  Below 300 ng/dL: catabolic state (Bhasin 2006 testosterone reference range).
  Hackney 2020: chronic low T → impaired MPS, fatigue, mood.

Refer-out rules (Path A — no medical language in user-facing message)
──────────────────────────────────────────────────────────────────────
  Condition: Hub_Estradiol < 25 pg/mL (female) or Hub_Testosterone < 250 ng/dL (male)
  Message: "O teu perfil de biomarcadores sugere supressão neuroendócrina
            (adaptação ao défice de energia). As recomendações de performance
            estão pausadas. Recomendamos revisão com um profissional de saúde."

References
──────────
  Khosla S. et al. (2001) J Bone Miner Res 16:552-559
    DOI 10.1359/jbmr.2001.16.3.552  [estradiol threshold for bone protection]
  Prior J.C. (2006) Clin Rev Bone Miner Metab 4:1-26
    [FHA → BMD loss ~2.5-3%/yr]
  Bhasin S. et al. (2006) J Clin Endocrinol Metab 91:4335-4343
    DOI 10.1210/jc.2006-0935  [T 300 ng/dL lower reference limit]
  Hackney A.C. (2020) Sports Med 50:971-976
    DOI 10.1007/s40279-020-01259-y  [exercise-induced hypogonadism]
  Mountjoy M. et al. (2014) Br J Sports Med 48:491-497
    DOI 10.1136/bjsports-2014-093502  [RED-S neuroendocrine suppression]
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
    build_engine_priors,
)

# ── Thresholds ────────────────────────────────────────────────────────────────

_E2_CHANCE_FLOOR_F:   float = 30.0    # pg/mL — bone protection floor (Khosla 2001)
_E2_REFEROUT_FLOOR_F: float = 25.0    # pg/mL — neuroendocrine suppression threshold
_T_CHANCE_FLOOR_M:    float = 300.0   # ng/dL — Bhasin 2006 lower reference limit
_T_REFEROUT_FLOOR_M:  float = 250.0   # ng/dL — severe hypogonadism threshold

_REFEROUT_MESSAGE = (
    "O teu perfil de biomarcadores sugere supressão neuroendócrina "
    "(adaptação ao défice de energia). "
    "As recomendações de performance estão pausadas. "
    "Recomendamos revisão com um profissional de saúde qualificado."
)


# ── Constraint builders ───────────────────────────────────────────────────────

def _female_chance_constraints() -> list[ChanceConstraint]:
    return [
        ChanceConstraint(
            name       = "estradiol_bone_protection_floor",
            state_key  = "Hub_Estradiol_pg_mL",
            op         = "≥",
            rhs        = _E2_CHANCE_FLOOR_F,
            alpha      = 0.10,
            units      = "pg/mL",
            source_doi = "10.1359/jbmr.2001.16.3.552",
        ),
    ]


def _male_chance_constraints() -> list[ChanceConstraint]:
    return [
        ChanceConstraint(
            name       = "testosterone_anabolic_floor",
            state_key  = "Hub_Testosterone_ng_dL",
            op         = "≥",
            rhs        = _T_CHANCE_FLOOR_M,
            alpha      = 0.10,
            units      = "ng/dL",
            source_doi = "10.1210/jc.2006-0935",
        ),
    ]


def _female_refer_out_rules() -> list[ReferOutRule]:
    return [
        ReferOutRule(
            name      = "female_neuroendocrine_suppression",
            state_key = "Hub_Estradiol_pg_mL",
            op        = "≤",
            threshold = _E2_REFEROUT_FLOOR_F,
            message   = _REFEROUT_MESSAGE,
        ),
    ]


def _male_refer_out_rules() -> list[ReferOutRule]:
    return [
        ReferOutRule(
            name      = "male_neuroendocrine_suppression",
            state_key = "Hub_Testosterone_ng_dL",
            op        = "≤",
            threshold = _T_REFEROUT_FLOOR_M,
            message   = _REFEROUT_MESSAGE,
        ),
    ]


# ── Shared Bayesian priors ────────────────────────────────────────────────────

def _gonadal_priors_female(posterior_theta: dict) -> dict:
    """
    Female-specific NLME priors.
    NLME D=2: E2_peak_baseline [pg/mL], tau_LM [days].
    """
    return dict(
        E2_peak_baseline = float(posterior_theta.get("E2_peak_baseline", 300.0)),
        tau_LM           = float(posterior_theta.get("tau_LM",           12.0)),
    )


def _gonadal_priors_male(posterior_theta: dict) -> dict:
    """
    Male-specific NLME priors.
    NLME D=2: T_baseline [ng/dL], Leydig_reserve [adim].
    """
    return dict(
        T_baseline     = float(posterior_theta.get("T_baseline",     600.0)),
        Leydig_reserve = float(posterior_theta.get("Leydig_reserve", 1.0)),
    )


# ── Public factories ──────────────────────────────────────────────────────────

def build_female_gonadal_envelope(
    posterior_theta: dict | None = None,
    athlete_data:    dict | None = None,
    genotype:        dict | None = None,
) -> Phase3Envelope:
    """
    Build Phase3Envelope for female gonadal slice.

    Parameters
    ----------
    posterior_theta : NLME posterior dict (E2_peak_baseline, tau_LM)
    athlete_data    : Phase 2 biomarker dict (age, DEXA, labs)
    genotype        : SNP dict (passed to build_engine_priors)

    Returns
    -------
    Phase3Envelope — no hard_constraints; chance + refer_out per sex.
    """
    theta = posterior_theta or {}
    adata = athlete_data or {}
    geno  = genotype or {}

    bayesian_priors: dict = _gonadal_priors_female(theta)
    try:
        engine_priors = build_engine_priors(adata, geno)
        bayesian_priors.update(engine_priors)
    except Exception:
        pass

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = [],
        chance_constraints = _female_chance_constraints(),
        refer_out_rules    = _female_refer_out_rules(),
    )


def build_male_gonadal_envelope(
    posterior_theta: dict | None = None,
    athlete_data:    dict | None = None,
    genotype:        dict | None = None,
) -> Phase3Envelope:
    """
    Build Phase3Envelope for male gonadal slice.
    """
    theta = posterior_theta or {}
    adata = athlete_data or {}
    geno  = genotype or {}

    bayesian_priors: dict = _gonadal_priors_male(theta)
    try:
        engine_priors = build_engine_priors(adata, geno)
        bayesian_priors.update(engine_priors)
    except Exception:
        pass

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = [],
        chance_constraints = _male_chance_constraints(),
        refer_out_rules    = _male_refer_out_rules(),
    )


# ── Constraint evaluation helpers ─────────────────────────────────────────────

def check_gonadal_constraints(
    env:        Phase3Envelope,
    state_dict: dict[str, float],
) -> list[str]:
    """
    Evaluate chance constraints by nominal value (no probabilistic tightening).
    Returns list of violated constraint names.

    Parameters
    ----------
    env        : Phase3Envelope from build_female/male_gonadal_envelope
    state_dict : {hub_key: float} — must include Hub_Estradiol_pg_mL (female)
                 or Hub_Testosterone_ng_dL (male)
    """
    violations: list[str] = []
    for c in env.chance_constraints:
        val = state_dict.get(c.state_key)
        if val is None:
            continue
        v = float(val)
        if c.op == "≥" and v < c.rhs:
            violations.append(c.name)
        elif c.op == "≤" and v > c.rhs:
            violations.append(c.name)
    return violations


def check_gonadal_refer_out(
    env:        Phase3Envelope,
    state_dict: dict[str, float],
) -> list[tuple[str, str]]:
    """
    Evaluate refer-out rules.
    Returns list of (rule_name, message) for any triggered rules.
    """
    triggered: list[tuple[str, str]] = []
    for r in env.refer_out_rules:
        val = state_dict.get(r.state_key)
        if val is None:
            continue
        v = float(val)
        if r.op == "≤" and v <= r.threshold:
            triggered.append((r.name, r.message))
        elif r.op == "≥" and v >= r.threshold:
            triggered.append((r.name, r.message))
    return triggered
