"""
app/slices/cardiorespiratory/orchestrator.py

Canonical L2→L6 Orchestrator — Cardiorespiratory / HRV Slice

Wires the 6-state pipeline in execution order:
    L2  ODE transition        cardiorespiratory_slice_ode (via UKF predict)
    L3  NLME personalisation  CardioNLME (optional; degrades to population prior)
    L4  State filter          CardioStateFilter (UKF predict + update + clamp)
    L5  Phase3Envelope        build_cardiorespiratory_envelope
    L6  Safety filter         PredictiveSafetyFilter (PSF)
    L7  Compliance            PathAFilter
    L8  Output                FilteredPrescription

Fail-Loud contract (I1)
───────────────────────
Every module reports ok | degraded | failed.
The orchestrator REFUSES to emit guidance if any nuclear module fails:
    nuclear = {L4 (filter), L5 (envelope)}
If a nuclear module is failed → computation_status = "failed", no prescription.
Non-nuclear failures (L3, L7) degrade to population defaults and log WARNING.

Module status propagation
──────────────────────────
ok        — module ran without exception, output is trusted
degraded  — module raised a recoverable exception; fallback applied
failed    — module raised an unrecoverable exception; pipeline blocked

Usage
─────
orch = CardioOrchestrator()

result = orch.step(
    prior        = GaussianState(mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT),
    observations = {"HR_obs_bpm": 145.0, "VO2_obs": float("nan"),
                    "RMSSD_obs_ms": float("nan")},
    controls     = {"power_watts": 220.0, "hub_T_core": 38.0,
                    "hub_pv_drop_pct": 3.5},
    athlete_data = {"age_years": 32},
    genotype     = {"ACE": "II"},
    load_acute_Wh   = 180.0,
    load_chronic_Wh = 155.0,
)

if result.computation_status == "failed":
    # pipeline refused — do not emit prescription
    log.error("pipeline blocked: %s", result.module_statuses)
else:
    emit(result.prescription)
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import NamedTuple

from app.engine.assimilation.ukf_filter import GaussianState
from app.slices.cardiorespiratory.ode import (
    DEFAULT_CARDIO_SLICE_PARAMS,
    X0_CARDIO_DEFAULT,
    P0_CARDIO_DEFAULT,
)
from app.slices.cardiorespiratory.filter import (
    CardioStateFilter,
    CardioTransitionParams,
    DEFAULT_TRANSITION_PARAMS,
)
from app.slices.cardiorespiratory.envelope import build_cardiorespiratory_envelope
from app.engine.phase3_envelope import Phase3Envelope

logger = logging.getLogger(__name__)


# ── Module status ─────────────────────────────────────────────────────────────

class ModuleStatus(str, Enum):
    OK       = "ok"
    DEGRADED = "degraded"
    FAILED   = "failed"


# ── Nuclear modules — pipeline blocked if any of these fail ──────────────────
_NUCLEAR = frozenset({"L4_filter", "L5_envelope"})


# ── Orchestration result ──────────────────────────────────────────────────────

class OrchestrationResult(NamedTuple):
    """
    Output of one CardioOrchestrator.step() call.

    Attributes
    ----------
    posterior          : GaussianState — updated 6-state belief (always present)
    envelope           : Phase3Envelope | None — None only on nuclear L5 failure
    prescription       : object | None — FilteredPrescription, or None if blocked
    computation_status : str — "ok" | "degraded" | "failed"
    module_statuses    : dict[str, ModuleStatus] — per-module status map
    block_reasons      : list[str] — human-readable reasons why guidance was blocked
    """
    posterior:          GaussianState
    envelope:           Phase3Envelope | None
    prescription:       object | None
    computation_status: str
    module_statuses:    dict
    block_reasons:      list


# ── Orchestrator ──────────────────────────────────────────────────────────────

class CardioOrchestrator:
    """
    L2→L8 canonical orchestrator for the 6-state cardiorespiratory slice.

    Stateless between calls — all state is passed in via GaussianState prior.
    Thread-safe (no shared mutable state).
    """

    def __init__(
        self,
        filter_instance: CardioStateFilter | None = None,
    ) -> None:
        self._filter = filter_instance or CardioStateFilter()

    # ── Primary public API ────────────────────────────────────────────────

    def step(
        self,
        prior:           GaussianState,
        observations:    dict[str, float],
        controls:        dict[str, float],
        athlete_data:    dict | None = None,
        genotype:        dict | None = None,
        posterior_theta: dict | None = None,
        transition_params: CardioTransitionParams | None = None,
        load_acute_Wh:   float = 0.0,
        load_chronic_Wh: float = 1.0,
        quality_flag:    int   = 0,
    ) -> OrchestrationResult:
        """
        Execute one step of the 6-state cardiorespiratory pipeline.

        Parameters
        ----------
        prior             : GaussianState — belief at previous step
        observations      : wearable observation dict (HR_obs_bpm, VO2_obs, RMSSD_obs_ms)
        controls          : control dict (power_watts, hub_T_core, hub_pv_drop_pct)
        athlete_data      : Phase 2 biomarker dict (age_years, etc.)
        genotype          : SNP genotype dict (ACE, ACTN3, etc.)
        posterior_theta   : NLME posterior dict (VO2_max_baseline, W_prime_capacity)
        transition_params : ODE + filter params; defaults to population prior
        load_acute_Wh     : 7-day EWMA load for ACWR / PSF
        load_chronic_Wh   : 28-day EWMA load for ACWR / PSF
        quality_flag      : int ∈ [0,4] — overall data quality (inflates R)

        Returns
        -------
        OrchestrationResult
        """
        statuses: dict[str, ModuleStatus] = {}
        block_reasons: list[str]          = []
        trans = transition_params or DEFAULT_TRANSITION_PARAMS

        # ── L4: State filter (nuclear) ────────────────────────────────────
        posterior = prior   # carry prior forward on failure
        try:
            posterior = self._filter.update_state(
                prior        = prior,
                observations = observations,
                controls     = controls,
                params       = trans,
                quality_flag = quality_flag,
            )
            statuses["L4_filter"] = ModuleStatus.OK
        except Exception as exc:
            statuses["L4_filter"] = ModuleStatus.FAILED
            block_reasons.append(f"L4_filter failed: {exc}")
            logger.error("CardioOrchestrator L4 FAILED: %s", exc)

        # ── L5: Phase3Envelope (nuclear) ──────────────────────────────────
        envelope: Phase3Envelope | None = None
        try:
            envelope = build_cardiorespiratory_envelope(
                posterior_theta = posterior_theta,
                athlete_data    = athlete_data,
                genotype        = genotype,
            )
            statuses["L5_envelope"] = ModuleStatus.OK
        except Exception as exc:
            statuses["L5_envelope"] = ModuleStatus.FAILED
            block_reasons.append(f"L5_envelope failed: {exc}")
            logger.error("CardioOrchestrator L5 FAILED: %s", exc)

        # ── Nuclear check — refuse guidance if any nuclear module failed ──
        nuclear_failed = [m for m in _NUCLEAR if statuses.get(m) == ModuleStatus.FAILED]
        if nuclear_failed:
            comp_status = "failed"
            logger.error(
                "CardioOrchestrator: nuclear modules failed %s — "
                "prescription REFUSED (I1 fail-loud).",
                nuclear_failed,
            )
            return OrchestrationResult(
                posterior          = posterior,
                envelope           = envelope,
                prescription       = None,
                computation_status = comp_status,
                module_statuses    = statuses,
                block_reasons      = block_reasons,
            )

        # ── L6+L7: Safety filter + compliance (optional — degrades) ──────
        prescription = None
        try:
            from app.control.safety_filter import (
                PredictiveSafetyFilter,
                SafetyConfig,
                SafetyVerdictStatus,
            )
            from app.control.nmpc_engine import AerobicNMPC, NMPCConfig
            from app.compliance.lexical_filter import PathAFilter

            at_val = float(posterior.mean[5])    # IDX_AT = 5
            wp_val = float(posterior.mean[3])    # IDX_WPRIME = 3

            nmpc = AerobicNMPC(NMPCConfig())
            nmpc_action = nmpc.compute_action(
                v_vagal_morning = at_val,
                w_prime_bal_kJ  = wp_val,
                load_acute_Wh   = load_acute_Wh,
                load_chronic_Wh = load_chronic_Wh,
            )

            psf    = PredictiveSafetyFilter(SafetyConfig(), trans)
            verdict = psf.evaluate(
                u_proposed      = nmpc_action,
                state           = posterior,
                load_acute_Wh   = load_acute_Wh,
                load_chronic_Wh = load_chronic_Wh,
            )

            filt = PathAFilter()
            acwr_val = load_acute_Wh / max(load_chronic_Wh, 1.0)
            prescription = filt.filter_prescription(
                verdict      = verdict,
                state        = posterior,
                action       = verdict.action,
                load_balance = {
                    "value": acwr_val,
                    "ci_lo": max(0.0, acwr_val - 0.15),
                    "ci_hi": min(2.0, acwr_val + 0.15),
                },
            )
            statuses["L6_safety"] = ModuleStatus.OK
            statuses["L7_compliance"] = ModuleStatus.OK

        except ImportError:
            statuses["L6_safety"] = ModuleStatus.DEGRADED
            statuses["L7_compliance"] = ModuleStatus.DEGRADED
            logger.warning(
                "CardioOrchestrator: L6/L7 unavailable (do-mpc or pydantic missing). "
                "Returning state + envelope only."
            )
        except Exception as exc:
            statuses["L6_safety"] = ModuleStatus.DEGRADED
            statuses["L7_compliance"] = ModuleStatus.DEGRADED
            logger.warning("CardioOrchestrator L6/L7 degraded: %s", exc)

        # ── Determine aggregate status ─────────────────────────────────────
        all_statuses = list(statuses.values())
        if any(s == ModuleStatus.FAILED for s in all_statuses):
            comp_status = "failed"
        elif any(s == ModuleStatus.DEGRADED for s in all_statuses):
            comp_status = "degraded"
        else:
            comp_status = "ok"

        return OrchestrationResult(
            posterior          = posterior,
            envelope           = envelope,
            prescription       = prescription,
            computation_status = comp_status,
            module_statuses    = statuses,
            block_reasons      = block_reasons,
        )

    # ── Factory helpers ───────────────────────────────────────────────────

    @staticmethod
    def cold_start() -> tuple["CardioOrchestrator", GaussianState]:
        """
        Return a fresh orchestrator + population-level initial state.

        Use for the very first step when no UKF history is available.
        Replace the returned state with the posterior from each subsequent step.
        """
        orch = CardioOrchestrator()
        state = GaussianState(mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT)
        return orch, state
