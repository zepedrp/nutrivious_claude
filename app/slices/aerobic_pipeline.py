"""
app/slices/aerobic_pipeline.py  [DEPRECATED]

DEPRECATED: This module is superseded by the cardiorespiratory slice
(app/slices/cardiorespiratory/) which provides a self-contained L2-L4
digital twin slice with 6-state ODE, NLME, UKF filter, Phase3Envelope,
and fail-loud gate architecture.

Migration path:
    from app.slices.cardiorespiratory import (
        CardioStateFilter, CardioTransitionParams,
        build_cardiorespiratory_envelope,
    )

This file is retained for backwards compatibility only and will be removed
in the next major version.

Aerobic Slice Orchestrator — Full L2-to-L8 Daily Cycle Pipeline

Aggregates the complete daily inference-control cycle for the Aerobic/HRV
Digital Twin slice, as specified in HLD §10 and Compass Passo 11–15:

    Step 1  Phase3Envelope    → Bayesian priors + hard constraints    (L2/L3)
    Step 2  Observer params   → personalised AerobicObserverParams    (L3 NLME)
    Step 3  UKF state update  → GaussianState posterior               (L4)
    Step 4  Gate Zero         → walk-forward validation decision       (L4)
    Step 5  NMPC              → optimal training action (IPOPT)        (L6)
    Step 6  PSF               → Wabersich-Zeilinger safety verdict     (L6)
    Step 7  PathAFilter       → MDR-compliant FilteredPrescription     (L8)

Public entry point
──────────────────
    orchestrator = AerobicSliceOrchestrator()
    output = orchestrator.run_daily_cycle(
        user_id      = "user-abc",
        athlete_data = {...},
        genotype     = {...},
        telemetry    = {"HR_bpm": 52.0, "RMSSD_ms": 78.0},
        controls     = {"power_watts": 220.0, "session_dur_min": 60.0},
        prior_state  = yesterday_posterior,   # GaussianState from DB
        obs_params   = nlme_obs_params,       # NLME posterior
    )
    # → AerobicSliceOutput  (Pydantic v2 / dataclass fallback)

Slice mode
──────────
    "production"   — Gate Zero PASS; prescriptions emitted.
    "experimental" — Gate Zero FAIL or SKIP; Active Recovery Protocol
                     emitted as conservative safe default.
    Gate Zero result takes precedence over the constructor default whenever
    `history_for_gate` is supplied.

Fail-Loud contract
──────────────────
    UKF divergence      → computation_status='failed', no prescription.
    NMPC infeasibility  → computation_status='failed', no prescription.
    PSF BLOCKED         → computation_status='ok',  Active Recovery emitted.
    PathAViolationError → computation_status='failed', reason logged.
    Missing prior_state → population default used, 'prior_state' added to
                          missing_inputs, computation_status='degraded'.
    Never returns silently invalid or zero-filled data.
"""
from __future__ import annotations

import logging
import math
import uuid
from datetime import date
from typing import Literal

import numpy as np

try:
    from pydantic import BaseModel
    _PYDANTIC_OK = True
except ImportError:          # pragma: no cover
    _PYDANTIC_OK = False
    BaseModel = object       # type: ignore[assignment,misc]

from app.engine.phase3_envelope import Phase3Envelope, build_phase3_envelope
from app.engine.assimilation.ukf_filter import (
    AerobicStateFilter,
    AerobicTransitionParams,
    GaussianState,
    X0_DEFAULT,
    P0_DEFAULT,
)
from app.engine.observation.aerobic_observer import (
    AerobicObserverParams,
    DEFAULT_OBSERVER_PARAMS,
    h_observer,
    IDX_V_VAGAL,
    IDX_W_PRIME,
)
from app.engine.solvers.cardiorespiratory_solver import (
    CardiorespiratorySolver,
)
from app.control.nmpc_engine import AerobicNMPC, NMPCConfig, NMPCAction
from app.control.safety_filter import (
    PredictiveSafetyFilter,
    SafetyConfig,
    SafetyVerdict,
    SafetyVerdictStatus,
)
from app.validation.backtest_engine import (
    ValidationHarness,
    GateZeroDecision,
)
from app.compliance.lexical_filter import (
    PathAFilter,
    FilteredPrescription,
    ConfidenceInterval,
    PathAViolationError,
)

logger = logging.getLogger(__name__)

# ── Physical constants (population-level defaults; overridden by NLME) ────────
_W_PRIME_KJ_DEFAULT: float = 18.0    # kJ  (Skiba 2015)
_TAU_W_REC_DEFAULT:  float = 240.0   # min (Clarke & Skiba 2013)


# ── Output schema ──────────────────────────────────────────────────────────────

if _PYDANTIC_OK:
    class AerobicSliceOutput(BaseModel):
        """
        FastAPI-ready response from the Aerobic/HRV daily cycle.
        All user-visible fields use Path A vocabulary.
        """
        user_id:            str
        date_iso:           str
        slice_mode:         Literal["production", "experimental"]
        prescription:       FilteredPrescription
        validation_gate:    Literal["PASS", "FAIL", "SKIP"]
        computation_status: Literal["ok", "degraded", "failed"]
        missing_inputs:     list[str]
        trace_id:           str

else:
    from dataclasses import dataclass as _dc

    @_dc
    class AerobicSliceOutput:          # type: ignore[no-redef]
        user_id:            str
        date_iso:           str
        slice_mode:         str
        prescription:       FilteredPrescription
        validation_gate:    str
        computation_status: str
        missing_inputs:     list
        trace_id:           str


# ── Orchestrator ───────────────────────────────────────────────────────────────

class AerobicSliceOrchestrator:
    """
    L2-to-L8 Daily Cycle Pipeline for the Aerobic/HRV Digital Twin Slice.

    Instantiate once per user session (or once per server process when a user
    pool handles separate state). The NMPC is JIT-compiled on first call
    (~5 s warm-up); subsequent calls are fast (<300 ms target p95).

    Parameters
    ----------
    nmpc_config   : NMPCConfig | None — NMPC hyperparameters (default: population)
    safety_config : SafetyConfig | None — PSF hard ceiling configuration
    slice_mode    : "production" | "experimental" — constructor default;
                    overridden by Gate Zero result when history is supplied
    CP_watts      : float — population Critical Power prior [W]
                    Overridden per call when NLME posterior supplies CP_watts

    Fail-Loud contract
    ──────────────────
    All module failures → AerobicSliceOutput(computation_status='failed').
    Never returns silently invalid or zero-filled data.
    """

    def __init__(
        self,
        nmpc_config:   NMPCConfig   | None = None,
        safety_config: SafetyConfig | None = None,
        slice_mode:    Literal["production", "experimental"] = "experimental",
        CP_watts:      float = 250.0,
    ) -> None:
        self._nmpc_cfg    = nmpc_config   or NMPCConfig()
        self._safety_cfg  = safety_config or SafetyConfig()
        self._slice_mode  = slice_mode
        self._CP          = CP_watts
        self._filter      = AerobicStateFilter()
        self._path_a      = PathAFilter(CP_watts=CP_watts)

        # Population-level default Mod 3 params (no genetic priors)
        self._default_cardio = CardiorespiratorySolver()._build_params({})

        # Lazy-init: NMPC and PSF are JIT-compiled on first use
        self._nmpc: AerobicNMPC            | None = None
        self._psf:  PredictiveSafetyFilter | None = None

        logger.info(
            "AerobicSliceOrchestrator ready — mode=%s, CP=%.0f W.",
            slice_mode, CP_watts,
        )

    # ── Primary public API ─────────────────────────────────────────────────

    def run_daily_cycle(
        self,
        user_id:           str,
        athlete_data:      dict,
        genotype:          dict[str, str],
        telemetry:         dict[str, float],
        controls:          dict[str, float],
        prior_state:       GaussianState         | None = None,
        obs_params:        AerobicObserverParams | None = None,
        history_for_gate:  dict                  | None = None,
        rmssd_baseline_ms: float                 | None = None,
    ) -> "AerobicSliceOutput":
        """
        Execute the full daily inference-control-filter cycle.

        Parameters
        ----------
        user_id           : str — user identifier (audit trail; not stored here)
        athlete_data      : dict — Phase 2 flat biomarker dict; absent keys → NaN
        genotype          : dict — SNP genotype strings {"ACTN3": "RX", ...}
        telemetry         : dict — morning wearable readings
                            Required keys: "HR_bpm", "RMSSD_ms"
                            NaN values are accepted (missing-observation handling)
        controls          : dict — yesterday's session + EWMA loads
                            Required: "power_watts", "session_dur_min"
                            Optional: "load_acute_Wh", "load_chronic_Wh"
        prior_state       : GaussianState | None — yesterday's UKF posterior
                            Uses population default (cold start) when None
        obs_params        : AerobicObserverParams | None — NLME posterior params
                            Uses population prior (DEFAULT_OBSERVER_PARAMS) if None
        history_for_gate  : dict | None — {"observations": (T, 2), "inputs": (T, 3)}
                            Enables Gate Zero walk-forward evaluation.
                            Skip if < 7 days of history.
        rmssd_baseline_ms : float | None — personal 7-day RMSSD mean [ms]
                            Enables the HRV-suppression refer-out rule.

        Returns
        -------
        AerobicSliceOutput — Pydantic model ready for FastAPI serialisation

        Notes
        -----
        Call this method once per user per day, after wearable sync completes.
        Persist the returned `trace_id` for audit and debugging.
        """
        trace_id = str(uuid.uuid4())
        today    = date.today().isoformat()
        missing: list[str] = []

        # ── Step 1: Phase3Envelope ─────────────────────────────────────────
        envelope   = build_phase3_envelope(athlete_data, genotype)
        obs_params = obs_params or DEFAULT_OBSERVER_PARAMS

        # ── Step 2: Resolve Critical Power ────────────────────────────────
        cp = self._resolve_cp(envelope, missing)

        transition_params = AerobicTransitionParams(
            cardio     = self._default_cardio,
            CP_watts   = cp,
            W_prime_kJ = _W_PRIME_KJ_DEFAULT,
            tau_W_rec  = _TAU_W_REC_DEFAULT,
        )

        # ── Step 3: UKF state update ───────────────────────────────────────
        if prior_state is None:
            prior_state = GaussianState(mean=X0_DEFAULT, cov=P0_DEFAULT)
            missing.append("prior_state")

        try:
            posterior = self._filter.update_state(
                prior     = prior_state,
                telemetry = telemetry,
                controls  = controls,
                params    = transition_params,
            )
        except RuntimeError as exc:
            logger.error("UKF divergence trace=%s: %s", trace_id, exc)
            return self._failed(user_id, today, trace_id, "ukf_diverged", missing)

        # ── Step 4: Gate Zero (optional) ──────────────────────────────────
        gate = GateZeroDecision.SKIP
        if history_for_gate is not None:
            gate = self._run_gate_zero(
                history_for_gate, transition_params, obs_params,
                prior_state, trace_id,
            )

        effective_mode: Literal["production", "experimental"] = (
            "production" if gate == GateZeroDecision.PASS else self._slice_mode
        )

        # ── Step 5: NMPC ───────────────────────────────────────────────────
        self._ensure_nmpc(cp)

        mean        = np.array(posterior.mean, dtype=np.float64)
        v_vagal     = float(mean[IDX_V_VAGAL])
        w_prime     = float(mean[IDX_W_PRIME])
        load_acute  = float(controls.get("load_acute_Wh",   180.0))
        load_chronic= float(controls.get("load_chronic_Wh", 160.0))

        try:
            nmpc_action = self._nmpc.compute_action(      # type: ignore[union-attr]
                v_vagal_morning  = v_vagal,
                w_prime_bal_kJ   = w_prime,
                load_acute_Wh    = load_acute,
                load_chronic_Wh  = load_chronic,
                obs_params       = obs_params,
            )
        except RuntimeError as exc:
            logger.error("NMPC infeasible trace=%s: %s", trace_id, exc)
            return self._failed(user_id, today, trace_id, "nmpc_infeasible", missing)

        # ── Step 6: Predictive Safety Filter ──────────────────────────────
        self._ensure_psf(transition_params)

        verdict = self._psf.evaluate(                      # type: ignore[union-attr]
            u_proposed      = nmpc_action,
            state           = posterior,
            load_acute_Wh   = load_acute,
            load_chronic_Wh = load_chronic,
        )

        # In experimental mode: force Active Recovery regardless of NMPC
        if effective_mode == "experimental":
            verdict = _force_recovery(verdict, nmpc_action)

        # ── Step 7: ACWR CI + refer-out + PathAFilter ─────────────────────
        load_balance = _compute_acwr_ci(verdict.action, load_acute, load_chronic)
        refer_out    = _check_refer_out(posterior, obs_params, envelope, rmssd_baseline_ms)

        try:
            prescription = self._path_a.filter_prescription(
                verdict       = verdict,
                state         = posterior,
                action        = verdict.action,
                load_balance  = load_balance,
                refer_out_msg = refer_out,
                CP_watts      = cp,
            )
        except PathAViolationError as exc:
            logger.error("Path A violation trace=%s: %s", trace_id, exc)
            return self._failed(user_id, today, trace_id, "path_a_violation", missing)

        status: Literal["ok", "degraded", "failed"] = "degraded" if missing else "ok"

        return AerobicSliceOutput(
            user_id            = user_id,
            date_iso           = today,
            slice_mode         = effective_mode,
            prescription       = prescription,
            validation_gate    = gate.value,
            computation_status = status,
            missing_inputs     = missing,
            trace_id           = trace_id,
        )

    # ── Lazy-init helpers ──────────────────────────────────────────────────

    def _ensure_nmpc(self, cp: float) -> None:
        """Initialise or rebuild NMPC when CP changes by more than 5%."""
        if self._nmpc is None:
            self._nmpc = AerobicNMPC(NMPCConfig(CP_watts=cp))
            logger.info("NMPC JIT compiled — CP=%.0f W.", cp)
            return

        drift = abs(cp - self._nmpc.config.CP_watts) / max(self._nmpc.config.CP_watts, 1.0)
        if drift > 0.05:
            old = self._nmpc.config.CP_watts
            old_cfg = self._nmpc.config
            self._nmpc = AerobicNMPC(NMPCConfig(
                horizon          = old_cfg.horizon,
                CP_watts         = cp,
                W_prime_kJ       = old_cfg.W_prime_kJ,
                tau_W_rec_min    = old_cfg.tau_W_rec_min,
                L_ref_Wh         = old_cfg.L_ref_Wh,
                power_max        = old_cfg.power_max,
                session_max_h    = old_cfg.session_max_h,
                V_vag_ref        = old_cfg.V_vag_ref,
                V_vag_min        = old_cfg.V_vag_min,
                k_rel            = old_cfg.k_rel,
                k_fatigue        = old_cfg.k_fatigue,
                w_vag            = old_cfg.w_vag,
                w_rmssd          = old_cfg.w_rmssd,
                w_load           = old_cfg.w_load,
                w_wprime         = old_cfg.w_wprime,
                target_load_norm = old_cfg.target_load_norm,
                acwr_min         = old_cfg.acwr_min,
                acwr_max         = old_cfg.acwr_max,
                k_chance         = old_cfg.k_chance,
                sigma_acwr       = old_cfg.sigma_acwr,
            ))
            logger.info("NMPC rebuilt — CP %.0f → %.0f W.", old, cp)

    def _ensure_psf(self, transition_params: AerobicTransitionParams) -> None:
        """Initialise PSF (JIT warm-up on first call)."""
        if self._psf is None:
            self._psf = PredictiveSafetyFilter(
                config            = self._safety_cfg,
                transition_params = transition_params,
            )
            logger.info("PSF initialised.")

    # ── Private helpers ────────────────────────────────────────────────────

    def _resolve_cp(self, envelope: Phase3Envelope, missing: list[str]) -> float:
        """Extract CP from Phase3Envelope priors or fall back to population prior."""
        cp_raw = envelope.bayesian_priors.get("CP_watts")
        if cp_raw is None or (isinstance(cp_raw, float) and math.isnan(cp_raw)):
            missing.append("CP_watts")
            return self._CP
        return float(cp_raw)

    def _run_gate_zero(
        self,
        history:           dict,
        transition_params: AerobicTransitionParams,
        obs_params:        AerobicObserverParams,
        prior_state:       GaussianState,
        trace_id:          str,
    ) -> GateZeroDecision:
        """Run walk-forward Gate Zero backtest. Returns SKIP on any error."""
        try:
            import jax.numpy as jnp
            harness = ValidationHarness(
                state_filter      = self._filter,
                transition_params = transition_params,
                obs_params        = obs_params,
            )
            obs_arr  = jnp.asarray(history["observations"], dtype=jnp.float32)
            ctrl_arr = jnp.asarray(history["inputs"],       dtype=jnp.float32)
            results  = harness.run_walk_forward(
                observations  = obs_arr,
                controls      = ctrl_arr,
                initial_state = prior_state,
            )
            logger.info(
                "Gate Zero trace=%s gate=%s MAE_twin=%s MAE_rolling=%s",
                trace_id, results.gate.value,
                results.mae_twin, results.mae_rolling,
            )
            return results.gate
        except Exception as exc:
            logger.warning("Gate Zero failed trace=%s: %s", trace_id, exc)
            return GateZeroDecision.SKIP

    def _failed(
        self,
        user_id:  str,
        today:    str,
        trace_id: str,
        reason:   str,
        missing:  list[str],
    ) -> "AerobicSliceOutput":
        """Return a safe failed output — no prescription emitted."""
        prescription = FilteredPrescription(
            session_label         = "Active Recovery Protocol",
            target_duration_min   = 20.0,
            target_intensity_zone = "Zone 1 — Light Activity",
            target_power_watts    = None,
            training_load_balance = ConfidenceInterval(value=1.0, ci_lo=0.8, ci_hi=1.2),
            recovery_capacity     = ConfidenceInterval(value=0.5, ci_lo=0.0, ci_hi=1.0),
            anaerobic_battery_pct = ConfidenceInterval(value=50.0, ci_lo=0.0, ci_hi=100.0),
            narrative_lines       = [
                "Our performance model encountered an issue today. "
                "We recommend a light active recovery session.",
                "Please ensure your device is synced and try again.",
            ],
            safety_gate_status    = "ACTIVE_RECOVERY",
            refer_out_message     = None,
        )
        logger.error("AerobicSlice FAILED trace=%s reason=%s", trace_id, reason)
        return AerobicSliceOutput(
            user_id            = user_id,
            date_iso           = today,
            slice_mode         = self._slice_mode,
            prescription       = prescription,
            validation_gate    = "SKIP",
            computation_status = "failed",
            missing_inputs     = missing + [reason],
            trace_id           = trace_id,
        )


# ── Module-level pure helpers ──────────────────────────────────────────────────

def _force_recovery(
    original:    SafetyVerdict,
    nmpc_action: NMPCAction,
) -> SafetyVerdict:
    """
    Force a BLOCKED verdict for experimental-mode output.

    The original NMPC action is preserved in original_action for internal
    audit; it is never surfaced in the user-facing FilteredPrescription.
    """
    return SafetyVerdict(
        status          = SafetyVerdictStatus.BLOCKED,
        action          = NMPCAction(
            power_watts       = 0.0,
            session_dur_min   = 0.0,
            is_optimal        = False,
            objective_value   = float("nan"),
            acwr_predicted    = float("nan"),
            w_prime_predicted = float("nan"),
        ),
        violations      = ["slice_mode=experimental"],
        x_worst_case    = original.x_worst_case,
        original_action = nmpc_action,
    )


def _compute_acwr_ci(
    action:       NMPCAction,
    load_acute:   float,
    load_chronic: float,
) -> dict[str, float]:
    """
    Compute projected ACWR point estimate + approximate 95% CI.

    Uses EWMA forward-step with population-level σ_ACWR ≈ 5% of ACWR value.
    (Gabbett 2016; Compass §6 population σ_ACWR = 0.05.)
    """
    load_day = action.power_watts * (action.session_dur_min / 60.0)
    la_next  = math.exp(-1.0 / 7.0)  * load_acute   + load_day
    lc_next  = math.exp(-1.0 / 28.0) * load_chronic + load_day
    acwr     = la_next / max(lc_next, 1.0)
    sigma    = 0.05 * max(acwr, 0.1)   # conservative 5% population σ_ACWR
    return {
        "value": round(acwr, 3),
        "ci_lo": round(max(0.0, acwr - 1.96 * sigma), 3),
        "ci_hi": round(acwr + 1.96 * sigma, 3),
    }


def _check_refer_out(
    state:             GaussianState,
    obs_params:        AerobicObserverParams,
    envelope:          Phase3Envelope,
    rmssd_baseline_ms: float | None,
) -> str | None:
    """
    Check Phase3Envelope refer-out rules against the UKF posterior.

    Only aerobic-slice-observable hub variables are evaluated here:
        Hub_HR_Rest_bpm    — from h_observer predicted HR
        Hub_RMSSD_relative — from h_observer RMSSD vs personal 7-day baseline
                             (σ assumed 15% of baseline; skip if baseline=None)

    Returns the Path A message of the first triggered rule, or None.
    """
    obs    = h_observer(state.mean, obs_params)
    hr     = float(obs[0])
    rmssd  = float(obs[1])

    rmssd_rel = float("nan")
    if rmssd_baseline_ms is not None and rmssd_baseline_ms > 0.0:
        sigma_rmssd = 0.15 * rmssd_baseline_ms
        rmssd_rel   = (rmssd - rmssd_baseline_ms) / max(sigma_rmssd, 1.0)

    hub: dict[str, float] = {
        "Hub_HR_Rest_bpm":    hr,
        "Hub_RMSSD_relative": rmssd_rel,
    }

    for rule in envelope.refer_out_rules:
        val = hub.get(rule.state_key)
        if val is None or math.isnan(val):
            continue
        if rule.op == "≥" and val >= rule.threshold:
            return rule.message
        if rule.op == "≤" and val <= rule.threshold:
            return rule.message

    return None
