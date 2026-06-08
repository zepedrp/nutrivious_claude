"""
app/control/safety_filter.py

Predictive Safety Filter (PSF) — Wabersich-Zeilinger Disjuntor

L6 Safety Layer — validates every NMPC action against worst-case
physiological projections before emitting a prescription.

Architecture (Wabersich & Zeilinger 2021, Automatica)
────────────────────────────────────────────────────────────────────────────
The PSF runs in PARALLEL with the NMPC:
  1. NMPC produces u_opt (the "greedy" optimal action)
  2. PSF projects state one step ahead with u_opt under WORST-CASE uncertainty
  3. If ANY hard ceiling is violated in the worst-case projection:
       → PSF replaces u_opt with u_safe (fallback: active recovery day)
       → Emits SafetyVerdict(passed=False, reason=..., action=u_safe)
  4. If all ceilings clear:
       → Emits SafetyVerdict(passed=True, action=u_opt)

Worst-case projection
────────────────────────────────────────────────────────────────────────────
Uses the UKF posterior covariance Σ to compute the worst-case state:
  For lower-bound constraint x[i] ≥ lb:   x_worst[i] = μ[i] - k · √Σ[i,i]
  For upper-bound constraint x[i] ≤ ub:   x_worst[i] = μ[i] + k · √Σ[i,i]

k_safety = 2.0 (default) → 97.7% of the Gaussian distribution is covered.

Hard ceilings checked (Phase3Envelope HARD_CEILINGS)
────────────────────────────────────────────────────────────────────────────
  W'_BAL ≥ 0           (anaerobic depletion; Skiba 2015)
  V_vagal ≥ V_critical  (autonomic overreach guard; Buchheit 2014)
  ACWR ≤ acwr_max       (overreach; Gabbett 2016)
  load_day ≤ load_max   (absolute session energy cap)

Fallback action (u_safe)
────────────────────────────────────────────────────────────────────────────
  power_watts     = 0 W          (complete rest)
  session_dur_min = 0 min        (no session)
  Interpreted as: rest day or light active recovery at athlete's discretion.

Fail-Loud contract
────────────────────────────────────────────────────────────────────────────
  RuntimeError if UKF propagation raises (propagated unmodified).
  SafetyVerdict.passed = False is NOT an error — it is the correct response.
  Violations are logged at WARNING level with full detail.
"""
from __future__ import annotations

import logging
import math
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from app.engine.assimilation.ukf_filter import (
    AerobicTransitionParams,
    GaussianState,
    _ukf_predict,
    Q_DEFAULT,
)
from app.engine.observation.aerobic_observer import (
    IDX_V_VAGAL,
    IDX_W_PRIME,
    STATE_DIM,
)
from app.control.nmpc_engine import NMPCAction

logger = logging.getLogger(__name__)

# ── State index aliases (clarity) ────────────────────────────────────────────
_IX_V_VAGAL   = IDX_V_VAGAL   # 2
_IX_W_PRIME   = IDX_W_PRIME   # 8

# ── Safety verdict ────────────────────────────────────────────────────────────

class SafetyVerdictStatus(Enum):
    """
    PSF verdict status.
    APPROVED  — action cleared; no hard ceiling violated.
    BLOCKED   — action blocked; fallback substituted.
    """
    APPROVED = "APPROVED"
    BLOCKED  = "BLOCKED"


class SafetyVerdict(NamedTuple):
    """
    Output of one PSF evaluation.

    Attributes
    ----------
    status         : SafetyVerdictStatus
    action         : NMPCAction — the emitted action (original or fallback)
    violations     : list[str]  — names of violated constraints (empty if APPROVED)
    x_worst_case   : tuple      — (mean, worst-case) projections [V_vagal, W'_bal]
    original_action: NMPCAction — the NMPC's original proposal (for auditing)
    """
    status:          SafetyVerdictStatus
    action:          NMPCAction
    violations:      list
    x_worst_case:    tuple   # (mean, worst) for [V_vagal, W'_bal]
    original_action: NMPCAction


# ── Safety configuration ──────────────────────────────────────────────────────

class SafetyConfig(NamedTuple):
    """
    PSF configuration. All parameters have physiologically-grounded defaults.

    k_safety      : σ-multiplier for worst-case bound (2.0 → 97.7% coverage)
    V_vag_critical: minimum allowed projected V_vagal (absolute floor)
    W_prime_min   : minimum allowed projected W'_bal [kJ] (hard: 0)
    acwr_max      : maximum allowed projected ACWR (Phase3Envelope ceiling)
    load_max_Wh   : maximum allowed session energy [W·h] (absolute)
    CP_watts      : Critical Power [W] (for ACWR computation fallback)
    W_prime_kJ    : W' capacity [kJ] (for W'_bal projection)
    tau_W_rec_min : W' recovery τ [min]
    """
    k_safety:       float = 2.0     # worst-case σ multiplier
    V_vag_critical: float = 0.10    # absolute vagal floor (Phase3Envelope)
    W_prime_min:    float = 0.0     # W'_BAL ≥ 0 (hard; Skiba 2015)
    acwr_max:       float = 1.3     # ACWR ceiling (Gabbett 2016; Compass §6)
    load_max_Wh:    float = 600.0   # max 400W × 1.5h = 600 W·h
    CP_watts:       float = 250.0   # Critical Power [W]
    W_prime_kJ:     float = 18.0    # W' capacity [kJ]
    tau_W_rec_min:  float = 240.0   # W' recovery τ [min]


# ── Fallback action ───────────────────────────────────────────────────────────

def _make_rest_action() -> NMPCAction:
    """
    Fallback action: complete rest day.
    Zero exercise prevents any further W'_BAL depletion or vagal suppression.
    """
    return NMPCAction(
        power_watts       = 0.0,
        session_dur_min   = 0.0,
        is_optimal        = False,
        objective_value   = float("nan"),
        acwr_predicted    = float("nan"),
        w_prime_predicted = float("nan"),
    )


# ── JAX-compiled one-step projection ─────────────────────────────────────────

@jax.jit
def _project_one_step(
    state_mean:        jax.Array,
    state_cov:         jax.Array,
    u:                 jax.Array,
    transition_params: AerobicTransitionParams,
) -> tuple[jax.Array, jax.Array]:
    """
    JIT-compiled UKF predict step for PSF worst-case propagation.

    Parameters
    ----------
    state_mean        : shape (STATE_DIM,)
    state_cov         : shape (STATE_DIM, STATE_DIM)
    u                 : shape (3,) — [power_w, sess_dur_min, 0]
    transition_params : AerobicTransitionParams

    Returns
    -------
    (mean_next, cov_next)
    """
    mean_next, cov_next = _ukf_predict(
        state_mean, state_cov, u, transition_params, Q_DEFAULT
    )
    return mean_next, cov_next


# ── Predictive Safety Filter ──────────────────────────────────────────────────

class PredictiveSafetyFilter:
    """
    Wabersich-Zeilinger Predictive Safety Filter for the Aerobic/HRV slice.

    Runs in parallel with AerobicNMPC and acts as a deterministic safety
    disjuntor (circuit-breaker): if the NMPC's proposed action would drive the
    physiological state into any hard-ceiling zone under worst-case uncertainty,
    the PSF blocks it and substitutes a safe fallback.

    Typical usage
    ─────────────
    psf = PredictiveSafetyFilter(SafetyConfig(), transition_params)

    verdict = psf.evaluate(
        u_proposed       = nmpc_action,
        state            = ukf_posterior,
        load_acute_Wh    = 180.0,
        load_chronic_Wh  = 160.0,
    )

    if verdict.status == SafetyVerdictStatus.APPROVED:
        emit_prescription(verdict.action)
    else:
        logger.warning("PSF blocked: %s", verdict.violations)
        emit_rest_day(verdict.action)

    Fail-Loud contract
    ──────────────────
    RuntimeError from UKF propagation propagates unmodified.
    PSF blocking is NOT an error — log at WARNING and emit fallback.
    All violations logged with full context for audit trail.
    """

    def __init__(
        self,
        config:            SafetyConfig,
        transition_params: AerobicTransitionParams,
    ) -> None:
        self.config            = config
        self.transition_params = transition_params
        logger.info(
            "PredictiveSafetyFilter initialised — k_safety=%.1f, "
            "V_vag_crit=%.2f, W'_min=%.1f kJ, ACWR_max=%.2f.",
            config.k_safety,
            config.V_vag_critical,
            config.W_prime_min,
            config.acwr_max,
        )

    # ── Primary public API ────────────────────────────────────────────────

    def evaluate(
        self,
        u_proposed:     NMPCAction,
        state:          GaussianState,
        load_acute_Wh:  float = 0.0,
        load_chronic_Wh: float = 1.0,
    ) -> SafetyVerdict:
        """
        Evaluate the proposed NMPC action against all hard ceilings.

        Parameters
        ----------
        u_proposed      : NMPCAction — the NMPC's recommended training action
        state           : GaussianState — UKF posterior (mean, covariance)
        load_acute_Wh   : float — current 7-day EWMA load for ACWR check
        load_chronic_Wh : float — current 28-day EWMA load for ACWR check

        Returns
        -------
        SafetyVerdict — either APPROVED (u_proposed) or BLOCKED (u_safe=rest)

        Raises
        ------
        RuntimeError if UKF propagation fails (propagated unmodified).
        """
        cfg = self.config

        # ── (1) Control bounds check (before ODE propagation) ─────────────
        ctrl_violations = self._check_control_bounds(u_proposed)

        # ── (2) Project one step ahead with proposed action ───────────────
        u_jax = jnp.array(
            [u_proposed.power_watts,
             u_proposed.session_dur_min,
             0.0],
            dtype=jnp.float32,
        )
        mean_next, cov_next = _project_one_step(
            state.mean, state.cov, u_jax, self.transition_params
        )

        # ── (3) Compute worst-case state bounds ───────────────────────────
        std_next    = jnp.sqrt(jnp.maximum(
            jnp.float32(0.0), jnp.diag(cov_next)
        ))
        x_mean      = np.array(mean_next, dtype=np.float64)
        x_std       = np.array(std_next,  dtype=np.float64)

        # Lower-bound worst case: mean - k·std  (for ≥ constraints)
        x_worst_lb  = x_mean - cfg.k_safety * x_std
        # Upper-bound worst case: mean + k·std  (for ≤ constraints)
        x_worst_ub  = x_mean + cfg.k_safety * x_std

        # ── (4) Check state hard ceilings ─────────────────────────────────
        state_violations = self._check_state_ceilings(
            x_worst_lb, x_worst_ub,
            u_proposed, load_acute_Wh, load_chronic_Wh,
        )

        all_violations = ctrl_violations + state_violations

        # ── (5) Verdict ───────────────────────────────────────────────────
        x_worst_report = (
            float(x_mean[_IX_V_VAGAL]),
            float(x_worst_lb[_IX_V_VAGAL]),
            float(x_mean[_IX_W_PRIME]),
            float(x_worst_lb[_IX_W_PRIME]),
        )

        if all_violations:
            logger.warning(
                "PSF BLOCKED — violations: %s | "
                "V_vag_worst=%.2f (≥%.2f), W'_worst=%.1f kJ (≥%.1f), "
                "proposed: %.0fW × %.0fmin",
                all_violations,
                x_worst_report[1], cfg.V_vag_critical,
                x_worst_report[3], cfg.W_prime_min,
                u_proposed.power_watts, u_proposed.session_dur_min,
            )
            fallback = _make_rest_action()
            return SafetyVerdict(
                status          = SafetyVerdictStatus.BLOCKED,
                action          = fallback,
                violations      = all_violations,
                x_worst_case    = x_worst_report,
                original_action = u_proposed,
            )

        logger.debug(
            "PSF APPROVED — V_vag_worst=%.2f (≥%.2f), W'_worst=%.1f kJ (≥%.1f)",
            x_worst_report[1], cfg.V_vag_critical,
            x_worst_report[3], cfg.W_prime_min,
        )
        return SafetyVerdict(
            status          = SafetyVerdictStatus.APPROVED,
            action          = u_proposed,
            violations      = [],
            x_worst_case    = x_worst_report,
            original_action = u_proposed,
        )

    # ── Constraint checkers ───────────────────────────────────────────────

    def _check_control_bounds(self, u: NMPCAction) -> list[str]:
        """
        Check absolute control bounds before propagation (fast pre-check).
        """
        cfg        = self.config
        violations = []

        # Session energy cap (absolute; prevents extreme single-session load)
        load_Wh = u.power_watts * (u.session_dur_min / 60.0)
        if load_Wh > cfg.load_max_Wh:
            violations.append(
                f"load_day_exceeded: {load_Wh:.0f} W·h > {cfg.load_max_Wh:.0f} W·h"
            )

        if u.power_watts < 0.0:
            violations.append(f"power_negative: {u.power_watts:.1f} W")

        if u.session_dur_min < 0.0:
            violations.append(f"duration_negative: {u.session_dur_min:.0f} min")

        return violations

    def _check_state_ceilings(
        self,
        x_worst_lb:     np.ndarray,
        x_worst_ub:     np.ndarray,
        u:              NMPCAction,
        load_acute_Wh:  float,
        load_chronic_Wh: float,
    ) -> list[str]:
        """
        Evaluate all physiological hard ceilings on the worst-case projected state.
        """
        cfg        = self.config
        violations = []

        # ── W'_BAL ≥ 0 (Skiba 2015) ──────────────────────────────────────
        # Use algebraic Skiba projection (consistent with ukf_filter.py)
        w_prime_mean   = float(x_worst_lb[_IX_W_PRIME])  # worst: min of projected W'
        excess         = max(0.0, u.power_watts - cfg.CP_watts)
        w_depletion    = excess * (u.session_dur_min * 0.060)
        t_rest_min     = max(0.0, 24.0 * 60.0 - u.session_dur_min)
        w_deficit      = max(0.0, cfg.W_prime_kJ - (w_prime_mean - w_depletion))
        w_recovery     = w_deficit * (1.0 - math.exp(-t_rest_min / cfg.tau_W_rec_min))
        w_prime_projected = w_prime_mean - w_depletion + max(0.0, w_recovery)

        if w_prime_projected < cfg.W_prime_min:
            violations.append(
                f"W_prime_bal: projected={w_prime_projected:.1f} kJ < min={cfg.W_prime_min:.1f} kJ"
            )

        # ── V_vagal ≥ V_vag_critical (Buchheit 2014) ─────────────────────
        v_vag_worst = float(x_worst_lb[_IX_V_VAGAL])
        if v_vag_worst < cfg.V_vag_critical:
            violations.append(
                f"V_vagal: worst_case={v_vag_worst:.3f} < critical={cfg.V_vag_critical:.2f}"
            )

        # ── ACWR ≤ acwr_max (Gabbett 2016) ───────────────────────────────
        load_day  = u.power_watts * (u.session_dur_min / 60.0)
        la_next   = math.exp(-1.0 / 7.0)  * load_acute_Wh  + load_day
        lc_next   = math.exp(-1.0 / 28.0) * load_chronic_Wh + load_day
        acwr_next = la_next / max(lc_next, 1.0)

        if acwr_next > cfg.acwr_max:
            violations.append(
                f"ACWR_upper: projected={acwr_next:.2f} > max={cfg.acwr_max:.2f}"
            )

        return violations

    # ── Utility: PSF pre-screening (lightweight, no ODE propagation) ──────

    def quick_screen(self, u: NMPCAction) -> list[str]:
        """
        Fast bounds check without ODE propagation — for pre-MPC screening.

        Returns a list of constraint names that are trivially violated
        (power < 0, duration < 0, session energy > cap). Empty = no issues.
        """
        return self._check_control_bounds(u)

    # ── PSF divergence monitor ────────────────────────────────────────────

    @staticmethod
    def compute_blocking_rate(verdicts: list[SafetyVerdict]) -> float:
        """
        Compute fraction of PSF-blocked recommendations over a window.

        Used for production monitoring (SLO: blocking_rate < 0.05).
        """
        if not verdicts:
            return 0.0
        n_blocked = sum(
            1 for v in verdicts if v.status == SafetyVerdictStatus.BLOCKED
        )
        return n_blocked / len(verdicts)
