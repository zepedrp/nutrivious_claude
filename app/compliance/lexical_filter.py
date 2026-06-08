"""
app/compliance/lexical_filter.py

Path A Lexical Filter — MDR Compliance Gate

Ensures all user-facing output uses performance/wellness vocabulary only.
Never emits medical claims, diagnoses, treatment language, or disease-risk
terminology (CLAUDE.md §1, HLD §7).

Two mechanisms
──────────────
1. SemanticMap — maps internal physiological variable names to blind
   sporting/wellness labels used throughout the user-facing output.

2. TextGate — deterministic regex gate applied to every string destined for
   the user. Raises PathAViolationError on any forbidden term match.
   No LLM involved — the gate is a pure string operation.

PSF-BLOCKED masking
───────────────────
When the Predictive Safety Filter returns BLOCKED (autonomic overreach guard,
W'_BAL depletion, ACWR ceiling breach), the prescriber emits "Active Recovery
Protocol". Internal violation reasons (e.g. "V_vagal: worst_case=0.08 <
critical=0.10") are logged internally and never surfaced in the user payload.

Fail-Loud contract
──────────────────
PathAViolationError is always raised (never swallowed). The orchestrator must
catch it, set computation_status='failed', reason='path_a_violation', and
emit the safe fallback response. A violation here is a logic error in the
narrative builder, not a user error — it warrants immediate investigation.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

import numpy as np

try:
    from pydantic import BaseModel, Field
    _PYDANTIC_OK = True
except ImportError:          # pragma: no cover — optional in test environments
    _PYDANTIC_OK = False
    BaseModel = object       # type: ignore[assignment,misc]
    Field     = lambda **_: None  # type: ignore[assignment]

from app.control.safety_filter import SafetyVerdict, SafetyVerdictStatus
from app.control.nmpc_engine import NMPCAction
from app.engine.assimilation.ukf_filter import GaussianState
from app.engine.observation.aerobic_observer import IDX_V_VAGAL, IDX_W_PRIME

logger = logging.getLogger(__name__)


# ── Exception ──────────────────────────────────────────────────────────────────

class PathAViolationError(ValueError):
    """
    Raised when forbidden medical/diagnostic language is detected in output.

    This is a logic error in the narrative builder — the gate is deterministic
    and should never fire in production if templates are correct. Treat it as
    a defect requiring immediate fix.
    """


# ── Semantic map: internal physiological name → wellness label ─────────────────

SEMANTIC_MAP: dict[str, str] = {
    # ── ODE state variables (Mod 3 cardiorespiratory + W'_bal) ────────────
    "V_vagal":                           "Recovery Capacity",
    "W_prime_bal":                       "Anaerobic Battery",
    "NE":                                "Arousal Level",
    "E":                                 "Activation Level",
    "P_a":                               "Cardiovascular Tone",
    "PaCO2":                             "Breathing Drive",
    "PbCO2":                             "Breathing Drive",
    "SpO2":                              "Oxygen Status",
    "V_E":                               "Ventilation Effort",
    # ── Hub variables ──────────────────────────────────────────────────────
    "Hub_RMSSD_relative":                "Recovery Signal",
    "Hub_REDS_Suppression":              "Energy Readiness",
    "Hub_ACWR":                          "Training Load Balance",
    "Hub_Core_Temp":                     "Core Temperature Status",
    "Hub_Sodium_Concentration":          "Hydration Balance",
    "Hub_HR_Rest_bpm":                   "Morning Heart Rate",
    "Hub_Energy_Availability_kcal_kg_ffm": "Fuelling Status",
    "Hub_Sleep_Duration_h":              "Sleep Quality",
    "Hub_W_Prime_Balance_kJ":            "Anaerobic Battery",
    # ── NMPC planning state ────────────────────────────────────────────────
    "load_acute":                        "Recent Activity Level",
    "load_chronic":                      "Baseline Fitness Level",
    # ── Raw wearable signals ───────────────────────────────────────────────
    "RMSSD_ms":                          "Heart Rate Variability",
    "HR_bpm":                            "Heart Rate",
    # ── PSF internal violation names → neutral descriptions ────────────────
    "W_prime_bal: projected":            "Anaerobic Battery: projected",
    "V_vagal: worst_case":               "Recovery Capacity: projected",
    "ACWR_upper":                        "Training Load Balance",
    "ACWR_lower":                        "Training Load Balance",
    "load_day_exceeded":                 "Session Energy Limit",
    # ── Module names ───────────────────────────────────────────────────────
    "ingestion":                         "data pipeline",
    "twin":                              "performance model",
    "control":                           "planning engine",
}


# ── Forbidden term patterns ────────────────────────────────────────────────────
# Any match in user-facing text → PathAViolationError.
# List derived from CLAUDE.md §4 + HLD §7 + Compass Passo 3.

_FORBIDDEN: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bdiagnos(e|is|tic|tics|ed|ing)\b",
        r"\btreat(ment|ed|ing|s|ments)?\b",
        r"\btherap(y|ies|eutic|eutical|ist|ists)\b",
        r"\bcure[ds]?\b",
        r"\bprescri(be|ption|bed|bing|ptions)\b",
        r"\bmedic(ate|ation|ated|ating|ations|ament|ine|al)\b",
        r"\bdiseas(e|es|ed)\b",
        r"\bdisorder\b",
        r"\bsyndrome\b",
        r"\billness(es)?\b",
        r"\bpatholog(y|ical|ies)\b",
        r"\bpatient(s)?\b",
        r"\blongevity\b",
        r"\blifespan\b",
        r"\bbiological.?age\b",
        r"\bdeficien(t|cy|cies)\b",
        r"\babnormal\b",
        r"\bclinical(ly)?\b",
        r"\bdiabetes\b",
        r"\bhypertension\b",
        r"\bheart\s+disease\b",
    ]
]


# ── Output schemas ─────────────────────────────────────────────────────────────

if _PYDANTIC_OK:
    class ConfidenceInterval(BaseModel):
        """Point estimate with 95% credible interval — all metrics surfaced to user."""
        value: float
        ci_lo: float = Field(description="95% CI lower bound")
        ci_hi: float = Field(description="95% CI upper bound")
        unit:  str   = ""

    class FilteredPrescription(BaseModel):
        """
        User-facing training prescription. All fields use Path A vocabulary.
        FastAPI-ready (Pydantic v2 model_config json_schema_extra can be added
        by the endpoint layer for OpenAPI documentation).
        """
        session_label:         str
        target_duration_min:   float
        target_intensity_zone: str
        target_power_watts:    float | None
        training_load_balance: ConfidenceInterval
        recovery_capacity:     ConfidenceInterval
        anaerobic_battery_pct: ConfidenceInterval
        narrative_lines:       list[str]
        safety_gate_status:    Literal["APPROVED", "ACTIVE_RECOVERY"]
        refer_out_message:     str | None = None

else:
    from dataclasses import dataclass as _dc

    @_dc
    class ConfidenceInterval:           # type: ignore[no-redef]
        value: float
        ci_lo: float
        ci_hi: float
        unit:  str = ""

    @_dc
    class FilteredPrescription:         # type: ignore[no-redef]
        session_label:         str
        target_duration_min:   float
        target_intensity_zone: str
        target_power_watts:    "float | None"
        training_load_balance: "ConfidenceInterval"
        recovery_capacity:     "ConfidenceInterval"
        anaerobic_battery_pct: "ConfidenceInterval"
        narrative_lines:       list
        safety_gate_status:    str
        refer_out_message:     "str | None" = None


# ── PathAFilter ────────────────────────────────────────────────────────────────

class PathAFilter:
    """
    MDR Compliance Lexical Gate + Semantic Translator.

    Translates internal physiological state into user-facing wellness language
    and enforces the Path A vocabulary boundary on all output strings.

    Usage
    ─────
    filt = PathAFilter(CP_watts=250.0)

    prescription = filt.filter_prescription(
        verdict       = psf_verdict,
        state         = ukf_posterior,
        action        = emitted_action,
        load_balance  = {"value": 1.05, "ci_lo": 0.95, "ci_hi": 1.15},
        refer_out_msg = None,
        CP_watts      = 250.0,
    )

    The gate validates every narrative string. A PathAViolationError from here
    is a defect in the narrative builder — the orchestrator must treat it as
    computation_status='failed'.

    Static utilities
    ────────────────
    PathAFilter.assert_path_a(text)  — raises PathAViolationError on hit
    PathAFilter.translate(name)      — internal name → wellness label
    """

    # Population-level W' capacity (Skiba 2015; Clarke & Skiba 2013)
    _W_PRIME_MAX_KJ: float = 18.0

    def __init__(self, CP_watts: float = 250.0) -> None:
        self._default_cp = CP_watts

    # ── Primary public API ─────────────────────────────────────────────────

    def filter_prescription(
        self,
        verdict:       SafetyVerdict,
        state:         GaussianState,
        action:        NMPCAction,
        load_balance:  dict[str, float],
        refer_out_msg: str | None = None,
        CP_watts:      float | None = None,
    ) -> FilteredPrescription:
        """
        Translate PSF verdict + UKF state into a Path A FilteredPrescription.

        If PSF returned BLOCKED, the session is masked as "Active Recovery
        Protocol" regardless of the original NMPC action. Internal violation
        reasons are never exposed in the returned object.

        Parameters
        ----------
        verdict       : SafetyVerdict from PredictiveSafetyFilter.evaluate()
        state         : GaussianState — UKF posterior (mean, covariance)
        action        : NMPCAction — the action to emit (already PSF-filtered)
        load_balance  : {"value", "ci_lo", "ci_hi"} — ACWR + 95% CI
        refer_out_msg : str | None — Path A refer-out sentence (pre-built)
        CP_watts      : float | None — individual Critical Power (overrides default)

        Returns
        -------
        FilteredPrescription — fully Path A compliant, gate-validated

        Raises
        ------
        PathAViolationError if any output string fails the text gate.
        """
        cp         = CP_watts if CP_watts is not None else self._default_cp
        is_blocked = verdict.status == SafetyVerdictStatus.BLOCKED

        # ── UKF posterior → wellness CIs ──────────────────────────────────
        mean = np.array(state.mean, dtype=np.float64)
        std  = np.sqrt(np.maximum(0.0, np.diag(np.array(state.cov, dtype=np.float64))))

        v_vag  = float(mean[IDX_V_VAGAL])
        s_vvag = float(std[IDX_V_VAGAL])
        w_kJ   = float(mean[IDX_W_PRIME])
        s_wkJ  = float(std[IDX_W_PRIME])

        w_pct  = (w_kJ  / self._W_PRIME_MAX_KJ) * 100.0
        s_wpct = (s_wkJ / self._W_PRIME_MAX_KJ) * 100.0

        recovery_ci = ConfidenceInterval(
            value = round(v_vag, 3),
            ci_lo = round(max(0.0, v_vag - 1.96 * s_vvag), 3),
            ci_hi = round(min(1.0, v_vag + 1.96 * s_vvag), 3),
            unit  = "a.u.",
        )
        battery_ci = ConfidenceInterval(
            value = round(w_pct, 1),
            ci_lo = round(max(  0.0, w_pct - 1.96 * s_wpct), 1),
            ci_hi = round(min(100.0, w_pct + 1.96 * s_wpct), 1),
            unit  = "%",
        )
        lb_ci = ConfidenceInterval(
            value = round(load_balance.get("value", 1.0), 3),
            ci_lo = round(load_balance.get("ci_lo",  0.8), 3),
            ci_hi = round(load_balance.get("ci_hi",  1.2), 3),
            unit  = "a.u.",
        )

        # ── Session classification ─────────────────────────────────────────
        if is_blocked or action.power_watts < 5.0:
            session_label = "Active Recovery Protocol"
            zone          = "Zone 1 — Light Activity"
            duration_min: float = max(20.0, action.session_dur_min)
            power_out: float | None = None
        else:
            zone, session_label = self._classify_session(action.power_watts, cp)
            duration_min        = round(action.session_dur_min, 0)
            power_out           = round(action.power_watts, 0)

        # ── Narrative (2–3 Path A sentences) ─────────────────────────────
        narrative = self._build_narrative(
            is_blocked    = is_blocked,
            v_vag         = v_vag,
            w_pct         = w_pct,
            acwr          = lb_ci.value,
            session_label = session_label,
        )

        # ── Text gate: validate every user-facing string ──────────────────
        self.assert_path_a(session_label)
        self.assert_path_a(zone)
        for line in narrative:
            self.assert_path_a(line)
        if refer_out_msg:
            self.assert_path_a(refer_out_msg)

        logger.debug(
            "PathAFilter: gate=APPROVED session=%r dur=%.0f min PSF=%s",
            session_label, duration_min, verdict.status.value,
        )

        return FilteredPrescription(
            session_label         = session_label,
            target_duration_min   = float(duration_min),
            target_intensity_zone = zone,
            target_power_watts    = power_out,
            training_load_balance = lb_ci,
            recovery_capacity     = recovery_ci,
            anaerobic_battery_pct = battery_ci,
            narrative_lines       = narrative,
            safety_gate_status    = "ACTIVE_RECOVERY" if is_blocked else "APPROVED",
            refer_out_message     = refer_out_msg,
        )

    # ── Text gate (static — callable independently) ────────────────────────

    @staticmethod
    def assert_path_a(text: str) -> None:
        """
        Assert that `text` contains no forbidden medical/diagnostic term.

        Deterministic regex gate — no LLM involvement.

        Raises
        ------
        PathAViolationError with the matched term and regex pattern.
        """
        for pattern in _FORBIDDEN:
            m = pattern.search(text)
            if m:
                raise PathAViolationError(
                    f"Path A violation: forbidden term '{m.group()}' in output. "
                    f"Pattern: {pattern.pattern!r}. Emit refused."
                )

    @staticmethod
    def translate(variable_name: str) -> str:
        """
        Translate an internal physiological/state variable name to wellness vocabulary.
        Falls back to the original name if no mapping is registered.

        Examples
        --------
        PathAFilter.translate("V_vagal")           → "Recovery Capacity"
        PathAFilter.translate("W_prime_bal")        → "Anaerobic Battery"
        PathAFilter.translate("Hub_ACWR")           → "Training Load Balance"
        PathAFilter.translate("unknown_var")        → "unknown_var"
        """
        return SEMANTIC_MAP.get(variable_name, variable_name)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _classify_session(power_w: float, cp_w: float) -> tuple[str, str]:
        """
        Map power relative to CP → (training zone label, session name).

        Thresholds align with the 5-zone model anchored at CP:
            < 55% CP  → Zone 1 (recovery)
            55–75% CP → Zone 2 (aerobic base)
            75–88% CP → Zone 3 (tempo)
            88–100% CP→ Zone 4 (threshold)
            ≥ 100% CP → Zone 5 (high intensity)
        """
        r = power_w / max(cp_w, 1.0)
        if r < 0.55:
            return "Zone 1 — Light Activity",     "Easy Endurance Session"
        if r < 0.75:
            return "Zone 2 — Aerobic Base",       "Aerobic Base Session"
        if r < 0.88:
            return "Zone 3 — Moderate Aerobic",   "Moderate Aerobic Session"
        if r < 1.00:
            return "Zone 4 — Threshold",          "Threshold Session"
        return "Zone 5 — High Intensity",          "High Intensity Interval Session"

    @staticmethod
    def _build_narrative(
        is_blocked:    bool,
        v_vag:         float,
        w_pct:         float,
        acwr:          float,
        session_label: str,
    ) -> list[str]:
        """
        Build 2–3 user-facing sentences in Path A vocabulary.

        All strings are pre-validated at construction; any violation would be
        a bug in this method, not in the gate.
        """
        if is_blocked:
            return [
                "Your recovery metrics suggest your body would benefit from "
                "a lighter effort today.",
                "We have adjusted your plan to an Active Recovery Protocol "
                "to support your readiness for tomorrow.",
            ]

        lines: list[str] = []

        # Recovery Capacity narrative
        if v_vag >= 0.75:
            lines.append(
                "Your Recovery Capacity is excellent — "
                "your autonomic system is well primed for today's session."
            )
        elif v_vag >= 0.50:
            lines.append(
                f"Your Recovery Capacity is solid. "
                f"Today's {session_label} fits well within your current readiness."
            )
        else:
            lines.append(
                "Your Recovery Capacity is moderate today. "
                "We have kept the intensity conservative to support your adaptation."
            )

        # Anaerobic Battery narrative
        if w_pct >= 80.0:
            lines.append(
                f"Your Anaerobic Battery is well stocked ({w_pct:.0f}%) — "
                "higher intensity is fully supported."
            )
        elif w_pct >= 40.0:
            lines.append(
                f"Your Anaerobic Battery stands at {w_pct:.0f}%, "
                "which is factored into today's intensity selection."
            )

        # Training Load Balance narrative
        if 0.85 <= acwr <= 1.10:
            lines.append(
                "Your Training Load Balance is in the optimal window — "
                "a great sign for continued performance progression."
            )
        elif acwr > 1.10:
            lines.append(
                "Your Training Load Balance is elevated. "
                "Today's session has been moderated to maintain long-term progress."
            )

        return lines[:3]
