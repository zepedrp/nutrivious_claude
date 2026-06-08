"""
app/control/nmpc_engine.py

L6 Aerobic NMPC — Non-Linear Model Predictive Control

Generates the optimal training stimulus for the next micro-cycle (H=2 days)
by solving a constrained nonlinear optimisation problem.

Architecture (HLD §4.5 + Compass Passo 15)
──────────────────────────────────────────────────────────────────────────────
Framework : do-mpc 4.6 (https://www.do-mpc.com)
Solver    : IPOPT (Interior Point Optimizer) via CasADi SX
Model     : 4-state discrete-time planning model (Δt = 1 day)
Horizon   : H = 2 days (configurable micro-cycle)
State dim : 4  [V_vagal, W_prime_bal, load_acute, load_chronic]
Control   : 2  [power_watts, session_dur_h]

Planning model (distinct from 9-state UKF — standard MPC practice)
──────────────────────────────────────────────────────────────────────────────
State x ∈ ℝ⁴:
    x[0]  V_vagal       [adim., ∈(0,1)]   morning vagal tone
    x[1]  W_prime_bal   [kJ]               anaerobic work capacity remaining
    x[2]  load_acute    [W·h]              7-day EWMA of session energy
    x[3]  load_chronic  [W·h]              28-day EWMA of session energy

Control u ∈ ℝ²:
    u[0]  power_watts   [W]    planned session power
    u[1]  session_dur_h [h]    planned session duration

Discrete dynamics (Δt = 1 day):
    load_day             = u[0] × u[1]                                         (1)
    load_acute_{t+1}     = α_a × x[2] + load_day,  α_a = e^{-1/7}  ≈ 0.867   (2)
    load_chronic_{t+1}   = α_c × x[3] + load_day,  α_c = e^{-1/28} ≈ 0.964   (3)
    W_dep                = max(0, u[0]-CP) × u[1] × 60 × 0.060                 (4)
    t_rest_min           = 24 × 60 - u[1] × 60                                 (5)
    W_rec                = (W_max - x[1] + W_dep) × (1 - e^{-t_rest/τ_rec})   (6)
    W_prime_bal_{t+1}    = clip(x[1] - W_dep + W_rec, 0, W_max)               (7)
    load_norm            = load_day / L_ref                                     (8)
    V_vagal_{t+1}        = (1-k_rel) × x[0] + k_rel × (V_ref - k_fat×load_norm) (9)

North Star objective:
    max J = Σ_t [ w_vag × V_vagal_t
                + w_rmssd × RMSSD_ref × exp(k_rmssd × (V_vagal_t - 0.5))
                - w_load × (power_norm_t - target)²
                - w_wprime × relu(1 - W_prime_norm_t) ]

Phase3Envelope injection
──────────────────────────────────────────────────────────────────────────────
Hard constraints (inviolable IPOPT bounds):
    x[0] ≥ V_vag_min = 0.15      (vagal floor — overtraining guard)
    x[1] ≥ 0                     (W'_BAL ≥ 0  — anaerobic safety)
    u[0] ∈ [0, power_max]
    u[1] ∈ [0, session_dur_max]

Chance constraints (tightened by k·σ, k = Φ⁻¹(1-α)):
    ACWR ≤ 1.3 - k × σ_ACWR     (overreach guard; α=0.01 → k≈2.33)
    ACWR ≥ 0.8 + k × σ_ACWR     (detraining guard)

Fail-Loud contract
──────────────────────────────────────────────────────────────────────────────
    IPOPT infeasibility raises RuntimeError. No silent fallback to zeros.
    Import failures of casadi/do-mpc raise RuntimeError with install guidance.
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np

try:
    import casadi
    import do_mpc
    _DOMPC_OK = True
except ImportError:  # pragma: no cover — optional dependency
    casadi  = None  # type: ignore[assignment]
    do_mpc  = None  # type: ignore[assignment]
    _DOMPC_OK = False

from app.engine.observation.aerobic_observer import (
    AerobicObserverParams,
    DEFAULT_OBSERVER_PARAMS,
)

logger = logging.getLogger(__name__)

# ── Planning model state/control indices ─────────────────────────────────────
_IX_VVAG  = 0   # V_vagal
_IX_WP    = 1   # W_prime_bal [kJ]
_IX_LA    = 2   # load_acute  [W·h]
_IX_LC    = 3   # load_chronic [W·h]

_IU_PWR   = 0   # power_watts [W]
_IU_DUR   = 1   # session_dur_h [h]

# ── IPOPT solver options ──────────────────────────────────────────────────────
_IPOPT_OPTS: dict = {
    "ipopt.max_iter":         300,
    "ipopt.print_level":      0,
    "ipopt.sb":               "yes",        # suppress banner
    "ipopt.warm_start_init_point": "yes",
    "ipopt.tol":              1e-6,
    "ipopt.acceptable_tol":   1e-4,
    "print_time":             0,
}


# ── Configuration ─────────────────────────────────────────────────────────────

class NMPCConfig(NamedTuple):
    """
    NMPC hyperparameters. All values have physiologically-grounded defaults.

    Phase3Envelope hard ceilings:
        V_vag_min    = 0.15   — absolute vagal floor (overtraining)
        W_prime_min  = 0.0    — W'_BAL ≥ 0 always
        acwr_min     = 0.8    — detraining guard (Gabbett 2016)
        acwr_max     = 1.3    — overreach guard (Gabbett 2016)
    """
    horizon:        int   = 2        # prediction horizon [days]
    CP_watts:       float = 250.0    # Critical Power [W]  (population prior)
    W_prime_kJ:     float = 18.0     # W' capacity [kJ]
    tau_W_rec_min:  float = 240.0    # W' recovery τ [min] = 4h
    L_ref_Wh:       float = 200.0    # reference daily load [W·h]
    power_max:      float = 400.0    # max allowed power [W]
    session_max_h:  float = 3.0      # max session duration [h]
    V_vag_ref:      float = 0.80     # reference resting vagal tone
    V_vag_min:      float = 0.15     # hard vagal floor (Phase3Envelope)
    k_rel:          float = 0.30     # daily vagal relaxation rate
    k_fatigue:      float = 0.30     # load→vagal fatigue coefficient
    w_vag:          float = 10.0     # objective: vagal tone weight
    w_rmssd:        float = 5.0      # objective: RMSSD weight
    w_load:         float = 1.0      # objective: load deviation weight
    w_wprime:       float = 3.0      # objective: W'_depletion penalty weight
    target_load_norm: float = 0.50   # target normalised load [0,1]
    acwr_min:       float = 0.80     # ACWR lower bound
    acwr_max:       float = 1.30     # ACWR upper bound
    k_chance:       float = 2.33     # tightening multiplier (α=0.01, k=Φ⁻¹(0.99))
    sigma_acwr:     float = 0.05     # ACWR uncertainty estimate


# ── NMPC action result ────────────────────────────────────────────────────────

class NMPCAction(NamedTuple):
    """
    Output of one NMPC solve.

    Attributes
    ----------
    power_watts     : float — recommended session power [W]
    session_dur_min : float — recommended session duration [min]
    is_optimal      : bool  — True if IPOPT converged (status 'S')
    objective_value : float — optimal J (lower is better for minimisation form)
    acwr_predicted  : float — projected ACWR after executing this action
    w_prime_predicted: float — projected W'_bal after session + overnight [kJ]
    """
    power_watts:      float
    session_dur_min:  float
    is_optimal:       bool
    objective_value:  float
    acwr_predicted:   float
    w_prime_predicted: float


# ── NMPC engine ───────────────────────────────────────────────────────────────

class AerobicNMPC:
    """
    Non-Linear Model Predictive Control — Aerobic/HRV Slice.

    Generates the optimal micro-cycle training stimulus using CasADi + do-mpc
    with IPOPT. The planning model is a 4-state discrete-time approximation of
    the daily cardiorespiratory dynamics, designed for fast trajectory planning
    (distinct from the full 9-state UKF model used for state estimation).

    Typical usage
    ─────────────
    nmpc = AerobicNMPC(NMPCConfig())

    action = nmpc.compute_action(
        v_vagal_morning    = 0.75,
        w_prime_bal_kJ     = 14.0,
        load_acute_Wh      = 180.0,
        load_chronic_Wh    = 160.0,
        obs_params         = my_observer_params,   # from NLME posterior
    )
    # → NMPCAction(power_watts=210.0, session_dur_min=55.0, is_optimal=True, ...)

    Fail-Loud contract
    ──────────────────
    IPOPT infeasibility raises RuntimeError (never returns zeros silently).
    Import failure raises RuntimeError with installation instructions.
    """

    def __init__(self, config: NMPCConfig | None = None) -> None:
        if not _DOMPC_OK:
            raise RuntimeError(
                "casadi and do-mpc>=4.6 required for AerobicNMPC. "
                "Install: pip install casadi do-mpc"
            )
        self.config = config or NMPCConfig()
        self._model  = self._build_model()
        self._mpc    = self._build_controller(self._model)
        self._x0     = self._default_x0()
        logger.info(
            "AerobicNMPC initialised — H=%d, IPOPT, CasADi model 4-state.",
            self.config.horizon,
        )

    # ── Internal model construction ───────────────────────────────────────

    def _build_model(self) -> "do_mpc.model.Model":
        """
        Build the 4-state discrete-time planning model in CasADi SX.

        Dynamics: Eqs (1)–(9) from module docstring.
        """
        cfg = self.config
        model = do_mpc.model.Model(model_type="discrete")

        # ── State variables ────────────────────────────────────────────────
        V_vagal      = model.set_variable("_x", "V_vagal",      shape=(1, 1))
        W_prime_bal  = model.set_variable("_x", "W_prime_bal",  shape=(1, 1))
        load_acute   = model.set_variable("_x", "load_acute",   shape=(1, 1))
        load_chronic = model.set_variable("_x", "load_chronic", shape=(1, 1))

        # ── Control variables ──────────────────────────────────────────────
        power_w      = model.set_variable("_u", "power_watts",   shape=(1, 1))
        session_h    = model.set_variable("_u", "session_dur_h", shape=(1, 1))

        # ── Derived quantities ─────────────────────────────────────────────
        # (1) Session energy
        load_day = power_w * session_h

        # (2-3) Load EWMA updates (Gabbett 2016; Cumulative Workload Model)
        alpha_a = casadi.DM(math.exp(-1.0 / 7.0))    # 7-day decay
        alpha_c = casadi.DM(math.exp(-1.0 / 28.0))   # 28-day decay
        load_acute_next   = alpha_a * load_acute   + load_day
        load_chronic_next = alpha_c * load_chronic + load_day

        # (4-7) W'_bal: Skiba algebraic model
        CP_dm    = casadi.DM(cfg.CP_watts)
        W_max_dm = casadi.DM(cfg.W_prime_kJ)
        tau_dm   = casadi.DM(cfg.tau_W_rec_min)

        W_dep   = casadi.fmax(casadi.DM(0.0), power_w - CP_dm) * session_h * casadi.DM(60.0 * 0.060)
        t_rest  = casadi.DM(24.0 * 60.0) - session_h * casadi.DM(60.0)
        t_rest  = casadi.fmax(casadi.DM(0.0), t_rest)
        W_rec   = (W_max_dm - W_prime_bal + W_dep) * (1.0 - casadi.exp(-t_rest / tau_dm))
        W_rec   = casadi.fmax(casadi.DM(0.0), W_rec)

        W_prime_next = casadi.fmin(
            W_max_dm, casadi.fmax(casadi.DM(0.0), W_prime_bal - W_dep + W_rec)
        )

        # (8-9) V_vagal daily model (phenomenological recovery + fatigue)
        L_ref_dm  = casadi.DM(cfg.L_ref_Wh)
        V_ref_dm  = casadi.DM(cfg.V_vag_ref)
        k_rel_dm  = casadi.DM(cfg.k_rel)
        k_fat_dm  = casadi.DM(cfg.k_fatigue)

        load_norm   = load_day / L_ref_dm
        V_vag_next  = (casadi.DM(1.0) - k_rel_dm) * V_vagal \
                    + k_rel_dm * (V_ref_dm - k_fat_dm * load_norm)
        V_vag_next  = casadi.fmax(casadi.DM(0.01), casadi.fmin(casadi.DM(1.0), V_vag_next))

        # ── Auxiliary expression: ACWR ─────────────────────────────────────
        acwr = load_acute / casadi.fmax(load_chronic, casadi.DM(1.0))
        model.set_expression("ACWR", acwr)

        # ── Register RHS ───────────────────────────────────────────────────
        model.set_rhs("V_vagal",      V_vag_next)
        model.set_rhs("W_prime_bal",  W_prime_next)
        model.set_rhs("load_acute",   load_acute_next)
        model.set_rhs("load_chronic", load_chronic_next)

        model.setup()
        return model

    def _build_controller(self, model: "do_mpc.model.Model") -> "do_mpc.controller.MPC":
        """
        Build the do-mpc MPC controller with IPOPT, objective, and constraints.
        """
        cfg = self.config
        mpc = do_mpc.controller.MPC(model)

        # ── Solver parameters ──────────────────────────────────────────────
        mpc.set_param(
            n_horizon         = cfg.horizon,
            t_step            = 1.0,
            n_robust          = 0,       # deterministic planning (PSF handles robustness)
            store_full_solution = True,
            nlpsol_opts       = _IPOPT_OPTS,
        )

        # ── Objective function (North Star) ───────────────────────────────
        # CasADi symbolic references to state / control:
        V_vag_sym   = model.x["V_vagal"]
        W_prime_sym = model.x["W_prime_bal"]
        power_sym   = model.u["power_watts"]

        RMSSD_ref = casadi.DM(DEFAULT_OBSERVER_PARAMS.RMSSD_ref)
        k_rmssd   = casadi.DM(DEFAULT_OBSERVER_PARAMS.k_rmssd)

        # RMSSD proxy from vagal tone (h_observer Eq. 2)
        rmssd_proxy = RMSSD_ref * casadi.exp(k_rmssd * (V_vag_sym - casadi.DM(0.5)))

        # Power normalised deviation from target
        power_target = cfg.target_load_norm * cfg.power_max
        power_dev    = (power_sym / casadi.DM(cfg.power_max) - casadi.DM(cfg.target_load_norm)) ** 2

        # W'_bal depletion penalty
        W_norm = casadi.fmax(casadi.DM(0.0), casadi.DM(1.0) - W_prime_sym / casadi.DM(cfg.W_prime_kJ))

        # Stage cost l(x, u)  — minimise this (IPOPT minimises)
        l_term = (
            - casadi.DM(cfg.w_vag)    * V_vag_sym
            - casadi.DM(cfg.w_rmssd)  * rmssd_proxy
            + casadi.DM(cfg.w_load)   * power_dev
            + casadi.DM(cfg.w_wprime) * W_norm
        )

        # Terminal cost m(x_H)  — 2× emphasis on terminal vagal recovery
        m_term = (
            - casadi.DM(cfg.w_vag * 2.0)    * V_vag_sym
            - casadi.DM(cfg.w_wprime * 1.5)  * (W_prime_sym / casadi.DM(cfg.W_prime_kJ))
        )

        mpc.set_objective(mterm=m_term, lterm=l_term)

        # Control input penalty (smooth control — prevents bang-bang solutions)
        mpc.set_rterm(power_watts=0.01, session_dur_h=0.10)

        # ── Hard bounds (Phase3Envelope HARD_CEILINGS) ──────────────────────
        # State lower bounds
        mpc.bounds["lower", "_x", "V_vagal"]     = cfg.V_vag_min   # vagal floor
        mpc.bounds["lower", "_x", "W_prime_bal"] = 0.0             # W'_BAL ≥ 0
        mpc.bounds["lower", "_x", "load_acute"]  = 0.0
        mpc.bounds["lower", "_x", "load_chronic"]= 0.0

        # Control bounds
        mpc.bounds["lower", "_u", "power_watts"]    = 0.0
        mpc.bounds["upper", "_u", "power_watts"]    = cfg.power_max
        mpc.bounds["lower", "_u", "session_dur_h"]  = 0.0
        mpc.bounds["upper", "_u", "session_dur_h"]  = cfg.session_max_h

        # ── ACWR chance constraints (tightened by k·σ) ──────────────────────
        # ACWR = load_acute / max(load_chronic, 1)
        # Tightening: k = Φ⁻¹(0.99) ≈ 2.33 (α=0.01)
        # Constraint ≤ ACWR_max - k·σ and ≥ ACWR_min + k·σ
        k_sig = cfg.k_chance * cfg.sigma_acwr
        acwr_upper_tight = cfg.acwr_max - k_sig
        acwr_lower_tight = cfg.acwr_min + k_sig

        acwr_expr = model.x["load_acute"] / casadi.fmax(
            model.x["load_chronic"], casadi.DM(1.0)
        )
        # ACWR ≤ acwr_upper (overreach guard)
        mpc.set_nl_cons("acwr_upper", acwr_expr,  ub=acwr_upper_tight)
        # ACWR ≥ acwr_lower (detraining guard) → -ACWR ≤ -acwr_lower
        mpc.set_nl_cons("acwr_lower", -acwr_expr, ub=-acwr_lower_tight)

        mpc.setup()
        return mpc

    # ── Public API ────────────────────────────────────────────────────────

    def set_phase3_constraints(
        self,
        v_vag_critical:   float = 0.10,
        acwr_upper:       float = 1.30,
        acwr_lower:       float = 0.80,
        sigma_acwr:       float = 0.05,
        k_chance:         float = 2.33,
    ) -> None:
        """
        Update NMPC constraints from Phase3Envelope at runtime.

        Call this after creating AerobicNMPC to inject personalised constraints
        derived from the individual's Phase3Envelope (e.g. tighter ACWR for
        athletes with prior RED-S history).

        Note: bounds and nl_cons cannot be changed after `mpc.setup()` in
        do-mpc 4.6 without rebuilding the controller. This method rebuilds the
        controller with updated constraint values.

        Parameters
        ----------
        v_vag_critical : float — minimum allowed morning V_vagal (hard)
        acwr_upper     : float — ACWR upper hard ceiling (default 1.30)
        acwr_lower     : float — ACWR lower hard floor (default 0.80)
        sigma_acwr     : float — ACWR uncertainty estimate for tightening
        k_chance       : float — tightening multiplier Φ⁻¹(1-α)
        """
        # Create updated config with new constraint values
        old = self.config
        new_config = NMPCConfig(
            horizon         = old.horizon,
            CP_watts        = old.CP_watts,
            W_prime_kJ      = old.W_prime_kJ,
            tau_W_rec_min   = old.tau_W_rec_min,
            L_ref_Wh        = old.L_ref_Wh,
            power_max       = old.power_max,
            session_max_h   = old.session_max_h,
            V_vag_ref       = old.V_vag_ref,
            V_vag_min       = v_vag_critical,
            k_rel           = old.k_rel,
            k_fatigue       = old.k_fatigue,
            w_vag           = old.w_vag,
            w_rmssd         = old.w_rmssd,
            w_load          = old.w_load,
            w_wprime        = old.w_wprime,
            target_load_norm= old.target_load_norm,
            acwr_min        = acwr_lower,
            acwr_max        = acwr_upper,
            k_chance        = k_chance,
            sigma_acwr      = sigma_acwr,
        )
        self.config = new_config
        self._model = self._build_model()
        self._mpc   = self._build_controller(self._model)
        logger.info(
            "Phase3 constraints updated — V_vag_min=%.2f, ACWR∈[%.2f, %.2f].",
            v_vag_critical, acwr_lower, acwr_upper,
        )

    def compute_action(
        self,
        v_vagal_morning:   float,
        w_prime_bal_kJ:    float,
        load_acute_Wh:     float,
        load_chronic_Wh:   float,
        obs_params:        AerobicObserverParams | None = None,
    ) -> NMPCAction:
        """
        Solve the NMPC problem and return the optimal training action.

        Parameters
        ----------
        v_vagal_morning  : float — current morning vagal tone from UKF x[IDX_V_VAGAL]
        w_prime_bal_kJ   : float — current W'_bal from UKF x[IDX_W_PRIME]
        load_acute_Wh    : float — 7-day EWMA load [W·h] from training log
        load_chronic_Wh  : float — 28-day EWMA load [W·h] from training log
        obs_params       : AerobicObserverParams | None — personalised observer params
                           (used to update RMSSD_ref / k_rmssd if provided)

        Returns
        -------
        NMPCAction

        Raises
        ------
        RuntimeError if IPOPT fails to find a feasible solution.
        """
        # ── Optional: update RMSSD params from NLME posterior ───────────
        if obs_params is not None:
            self._update_rmssd_params(obs_params)

        # ── Initial state vector (4-dim planning model) ──────────────────
        x0 = np.array([
            [float(v_vagal_morning)],
            [float(w_prime_bal_kJ)],
            [float(load_acute_Wh)],
            [float(load_chronic_Wh)],
        ], dtype=np.float64)

        # ── Set initial state ────────────────────────────────────────────
        self._mpc.x0 = x0
        self._mpc.set_initial_guess()

        # ── Solve the NMPC (one call to IPOPT) ───────────────────────────
        try:
            u_opt = self._mpc.make_step(x0)   # shape (2, 1)
        except Exception as exc:
            raise RuntimeError(
                f"AerobicNMPC.compute_action: IPOPT optimisation failed: {exc}. "
                "Check Phase3Envelope constraints and initial state feasibility."
            ) from exc

        # ── Extract solution ─────────────────────────────────────────────
        power_w  = float(np.clip(u_opt[0, 0], 0.0, self.config.power_max))
        dur_h    = float(np.clip(u_opt[1, 0], 0.0, self.config.session_max_h))

        # Predicted ACWR and W'_bal from first step
        load_day  = power_w * dur_h
        load_a1   = math.exp(-1.0/7.0)  * load_acute_Wh  + load_day
        load_c1   = math.exp(-1.0/28.0) * load_chronic_Wh + load_day
        acwr_pred = load_a1 / max(load_c1, 1.0)

        excess    = max(0.0, power_w - self.config.CP_watts)
        w_dep     = excess * dur_h * 60.0 * 0.060
        t_rest    = max(0.0, 24.0 * 60.0 - dur_h * 60.0)
        w_rec     = (self.config.W_prime_kJ - w_prime_bal_kJ + w_dep) * (
            1.0 - math.exp(-t_rest / self.config.tau_W_rec_min)
        )
        w_prime_pred = float(np.clip(w_prime_bal_kJ - w_dep + max(0.0, w_rec),
                                     0.0, self.config.W_prime_kJ))

        # ── IPOPT convergence check ───────────────────────────────────────
        sol_stats = self._mpc.get_stats()
        is_optimal = str(sol_stats.get("return_status", "")) in {
            "Solve_Succeeded", "Solved_To_Acceptable_Level"
        }

        if not is_optimal:
            raise RuntimeError(
                f"AerobicNMPC: IPOPT did not converge — status: "
                f"{sol_stats.get('return_status', 'unknown')}. "
                "Prescriptions withheld (Fail-Loud)."
            )

        obj_val = float(sol_stats.get("obj_val", float("nan")))

        logger.debug(
            "NMPC solve OK — power=%.1f W, dur=%.0f min, ACWR=%.2f, W'=%.1f kJ, J=%.2f",
            power_w, dur_h * 60, acwr_pred, w_prime_pred, obj_val,
        )

        return NMPCAction(
            power_watts      = power_w,
            session_dur_min  = dur_h * 60.0,
            is_optimal       = is_optimal,
            objective_value  = obj_val,
            acwr_predicted   = acwr_pred,
            w_prime_predicted= w_prime_pred,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _update_rmssd_params(self, obs_params: AerobicObserverParams) -> None:
        """
        Update RMSSD_ref and k_rmssd in the objective from NLME posterior.

        Requires rebuilding the controller (objective depends on these params).
        Only rebuilds if params differ significantly (>5%) to avoid overhead.
        """
        ref_change = abs(obs_params.RMSSD_ref - DEFAULT_OBSERVER_PARAMS.RMSSD_ref)
        if ref_change / DEFAULT_OBSERVER_PARAMS.RMSSD_ref > 0.05:
            logger.debug(
                "AerobicNMPC: RMSSD_ref updated %.1f→%.1f, rebuilding controller.",
                DEFAULT_OBSERVER_PARAMS.RMSSD_ref, obs_params.RMSSD_ref,
            )
            self._model = self._build_model()
            self._mpc   = self._build_controller(self._model)

    def _default_x0(self) -> np.ndarray:
        """Population-level resting planning state."""
        return np.array([
            [self.config.V_vag_ref],   # V_vagal at reference
            [self.config.W_prime_kJ],  # W'_bal fully replete
            [self.config.L_ref_Wh],    # load_acute at reference
            [self.config.L_ref_Wh],    # load_chronic at reference (steady-state)
        ], dtype=np.float64)
