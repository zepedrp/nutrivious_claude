"""
app/slices/thermo_renal/envelope.py  —  Phase 3 Envelope, Thermo-Renal V2

Hard ceilings (life-threatening)
  Hub_Core_Temp    < 40.0 °C      (Exertional Heat Stroke abort)
  Hub_Plasma_Sodium > 132.0 mmol/L  (Severe EAH — cerebral oedema)

Hard alerts (recommend halt)
  Hub_Core_Temp    < 39.0 °C
  Hub_Plasma_Sodium > 135.0 mmol/L

Path A: all user-facing messages use performance/wellness vocabulary only.
No medical terms (hyponatremia, hyperthermia, disease, diagnose, treat).

References
  Hew-Butler T. et al. (2015) Clin J Sport Med 35   DOI 10.1249/MSS.0000000000000621
  Bouchama A., Knochel J.P. (2002) NEJM 346         DOI 10.1056/NEJMra011089
"""
from __future__ import annotations

from app.engine.phase3_envelope import (
    Constraint,
    ChanceConstraint,
    ReferOutRule,
    Phase3Envelope,
    build_engine_priors,
)

# ── Safety thresholds ─────────────────────────────────────────────────────────
_T_CORE_ABORT:   float = 40.0    # Heat Stroke — catastrophic
_T_CORE_ALERT:   float = 39.0    # Early hyperthermia — halt recommended
_T_CORE_CHANCE:  float = 39.5    # Chance constraint P(>39.5°C) ≤ 5%

_NA_ABORT:       float = 132.0   # Severe EAH — cerebral oedema risk
_NA_ALERT:       float = 135.0   # EAH alert — fluid pause
_NA_CHANCE:      float = 135.0   # Chance constraint P(<135) ≤ 1%


def _hard_constraints() -> list[Constraint]:
    return [
        Constraint(
            name="core_temp_abort",
            state_key="Hub_Core_Temp",
            op="<=",
            rhs=_T_CORE_ABORT,
            units="degC",
            source_doi="10.1056/NEJMra011089",
        ),
        Constraint(
            name="core_temp_alert",
            state_key="Hub_Core_Temp",
            op="<=",
            rhs=_T_CORE_ALERT,
            units="degC",
            source_doi="10.1056/NEJMra011089",
        ),
        Constraint(
            name="plasma_na_abort",
            state_key="Hub_Plasma_Sodium",
            op=">=",
            rhs=_NA_ABORT,
            units="mmol/L",
            source_doi="10.1249/MSS.0000000000000621",
        ),
        Constraint(
            name="plasma_na_alert",
            state_key="Hub_Plasma_Sodium",
            op=">=",
            rhs=_NA_ALERT,
            units="mmol/L",
            source_doi="10.1249/MSS.0000000000000621",
        ),
    ]


def _chance_constraints() -> list[ChanceConstraint]:
    return [
        ChanceConstraint(
            name="core_temp_thermal_injury",
            state_key="Hub_Core_Temp",
            op="<=",
            rhs=_T_CORE_CHANCE,
            alpha=0.05,
            units="degC",
            source_doi="10.1056/NEJMra011089",
        ),
        ChanceConstraint(
            name="plasma_na_eah_risk",
            state_key="Hub_Plasma_Sodium",
            op=">=",
            rhs=_NA_CHANCE,
            alpha=0.01,
            units="mmol/L",
            source_doi="10.1249/MSS.0000000000000621",
        ),
    ]


def _refer_out_rules() -> list[ReferOutRule]:
    return [
        ReferOutRule(
            name="thermal_ceiling_refer_out",
            state_key="Hub_Core_Temp",
            op=">",
            threshold=_T_CORE_ALERT,
            message=(
                "O teu indicador de temperatura central ultrapassou o limite de "
                "seguranca para a performance. Interrompe a atividade imediatamente, "
                "arrefece e hidrata. Retoma apenas quando a leitura estiver "
                "dentro da janela de treino segura."
            ),
        ),
        ReferOutRule(
            name="eah_fluid_refer_out",
            state_key="Hub_Plasma_Sodium",
            op="<",
            threshold=_NA_ALERT,
            message=(
                "Os teus marcadores de hidratacao e equilibrio de fluidos estao "
                "fora da janela de performance segura. Pausamos todas as "
                "recomendacoes de ingestao de fluidos ate revisao por um "
                "profissional de saude qualificado."
            ),
        ),
    ]


def build_thermo_renal_envelope(
    athlete_data: dict | None = None,
    genotype:     dict | None = None,
) -> Phase3Envelope:
    """
    Build Phase3Envelope for the Thermo-Renal V2 slice.

    Returns priors + hard/chance constraints + refer-out rules.
    The envelope never produces a prediction — it constrains the MPC.
    """
    _data = athlete_data or {}
    _geno = genotype     or {}

    priors = build_engine_priors(_data, _geno) if (_data or _geno) else {}

    return Phase3Envelope(
        bayesian_priors    = priors,
        hard_constraints   = _hard_constraints(),
        chance_constraints = _chance_constraints(),
        refer_out_rules    = _refer_out_rules(),
    )
