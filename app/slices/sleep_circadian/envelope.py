"""
app/slices/sleep_circadian/envelope.py  —  Phase 3 Envelope V2.0

CLAUDE.md §3: "Fase 3 é priors + restrições, nunca um modelo multiplicativo."

Safety floors enforced as hard constraints (MPC never violates):
  Sleep_Drive_SWS ≥ 0 (trivially enforced by physics)
  P(Adenosine > 1.8) ≤ 0.05  — cognitive-collapse chance constraint
  Refer-out when circadian desynchrony detected (Path A language only)

References
──────────
  Watson N.F. et al. (2015) Sleep 38(6):843–844
  Lim J. & Dinges D.F. (2010) Annu Rev Psychol 61:591–620
  CLAUDE.md §3 golden rules
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
    build_engine_priors,
)

# ── Thresholds ─────────────────────────────────────────────────────────────────

_ADEN_TOXIC:  float = 1.80   # a.u.  extreme adenosine → cognitive collapse
_ADEN_CHANCE: float = 1.80   # same threshold for chance constraint
_SWS_FLOOR:   float = 0.0    # trivial physical floor

_ADEN_REFER:  float = 1.60   # earlier refer-out before toxic threshold
_SWS_REFER:   float = 0.05   # near-zero SWS for 3+ successive steps → refer-out


# ── Hard constraints ───────────────────────────────────────────────────────────

def _hard_constraints() -> list[Constraint]:
    return [
        Constraint(
            name       = "sws_physical_floor",
            state_key  = "Hub_Sleep_SWS_Fraction",
            op         = "≥",
            rhs        = _SWS_FLOOR,
            units      = "dimensionless",
            source_doi = "10.1177/0748730406297512",  # Phillips-Robinson 2007
        ),
    ]


# ── Chance constraints ─────────────────────────────────────────────────────────

def _chance_constraints() -> list[ChanceConstraint]:
    """P(Adenosine > toxic_threshold) ≤ 0.05 — cognitive/motor collapse guard."""
    return [
        ChanceConstraint(
            name       = "adenosine_toxic_load",
            state_key  = "Hub_Adenosine",
            op         = "≥",
            rhs        = _ADEN_CHANCE,
            alpha      = 0.05,
            units      = "a.u.",
            source_doi = "10.1146/annurev.psych.60.110707.163612",  # Lim 2010
        ),
    ]


# ── Refer-out rules (Path A — zero medical vocabulary) ────────────────────────

def _refer_out_rules() -> list[ReferOutRule]:
    _MSG_DESYNC = (
        "O teu padrão de recuperação indica dessincronização circadiana severa. "
        "Prioriza higiene de luz e repouso absoluto no próximo microciclo. "
        "Considera consultar um profissional de saúde qualificado se o padrão persistir."
    )
    _MSG_ADEN = (
        "A tua carga de recuperação acumulada está muito elevada. "
        "Estamos a pausar as recomendações de treino de alta intensidade "
        "até o teu padrão de descanso estabilizar."
    )
    return [
        ReferOutRule(
            name      = "circadian_desynchrony_severe",
            state_key = "Hub_Sleep_SWS_Fraction",
            op        = "≤",
            threshold = _SWS_REFER,
            message   = _MSG_DESYNC,
        ),
        ReferOutRule(
            name      = "adenosine_overload",
            state_key = "Hub_Adenosine",
            op        = "≥",
            threshold = _ADEN_REFER,
            message   = _MSG_ADEN,
        ),
    ]


# ── Envelope builder ───────────────────────────────────────────────────────────

def build_sleep_envelope(
    athlete_data: dict | None = None,
    posterior_theta: dict | None = None,
) -> Phase3Envelope:
    """
    Build the Phase3Envelope for the sleep-circadian V2 slice.

    Parameters
    ──────────
    athlete_data     : onboarding dict; cold-start safe if None.
    posterior_theta  : NLME posterior {param_name: float}; NaN if absent.

    Returns Phase3Envelope with:
      - bayesian_priors  : engine priors + sleep NLME posteriors
      - hard_constraints : 1 inviolable floor
      - chance_constraints : 1 cognitive-load chance constraint
      - refer_out_rules : 2 Path A refer-out gates
    """
    athlete_data    = athlete_data or {}
    posterior_theta = posterior_theta or {}

    sleep_priors: dict[str, float] = {
        "sleep_tau_c":   posterior_theta.get("tau_c",   24.18),
        "sleep_k_clear": posterior_theta.get("k_clear",  0.13),
    }

    base_priors = build_engine_priors(athlete_data, genotype={})
    merged      = {**base_priors, **sleep_priors}

    return Phase3Envelope(
        bayesian_priors    = merged,
        hard_constraints   = _hard_constraints(),
        chance_constraints = _chance_constraints(),
        refer_out_rules    = _refer_out_rules(),
    )
