"""
app/slices/cardiorespiratory/filter.py

L4 State Filter — Cardiorespiratory Slice
Unscented Kalman Filter for the 6-state cardiorespiratory performance system.

Architecture (HLD §4.3 — L4: State Estimation)
────────────────────────────────────────────────
State x ∈ ℝ⁶:
    [V_O2, Heart_Rate, Stroke_Volume, W_prime_bal, Resp_Fatigue, Autonomic_Tone]

Observation y ∈ ℝ³:
    [HR_obs_bpm, VO2_obs_mLkgmin, RMSSD_obs_ms]

Transition f(x, u) — 1-minute ODE advance
───────────────────────────────────────────
Integration window: Δt = 1 min (typical wearable HR sampling rate)
Integrator: diffrax.Tsit5() — 4th/5th order Runge-Kutta.
            Adequate for 1-min window (not severely stiff at this timescale).
Control u = (power_watts, hub_T_core, hub_pv_drop_pct)

Physical clamps (MANDATORY post-update)
────────────────────────────────────────
Applied immediately after every UKF measurement update to prevent the linear
Kalman gain from driving states into physiologically impossible territory:
    V_O2          ≥ 0 mL/kg/min   (oxygen consumption never negative)
    Heart_Rate    ≥ HR_floor       (absolute bradycardia sentinel, default 30 bpm)
    Stroke_Volume ≥ 0.020 L        (minimum viable cardiac output sentinel)
    W_prime_bal   ≥ 0 kJ           (battery cannot be < 0 by definition)
    Resp_Fatigue  ∈ [0, 1]         (bounded fraction)
    Autonomic_Tone ∈ [0, 1]        (bounded fraction)
These clamps use jnp.maximum / jnp.clip — no Python control flow.

Multi-signal observation (quality-aware)
─────────────────────────────────────────
HR is dense (wearable heartbeat); VO2 and RMSSD are sparse (metabolic cart,
morning-only HRV). Missing observations (NaN) inflate the corresponding
R diagonal entry by 1e8, making the innovation zero — a predict-only step
for that channel. This preserves state coherence without silent zero-filling.

UKF parametrisation (Merwe & Wan 2000) — N = STATE_DIM = 6
────────────────────────────────────────────────────────────
    α = 1e-3  (tight sigma-point spread; good for near-linear physiology)
    β = 2.0   (optimal for Gaussian distributions)
    κ = 0.0
    λ = α²(n+κ) − n = 1e-6 × 6 − 6 ≈ −5.999994

Fail-Loud contract
──────────────────
    • All exceptions re-raised — no silent substitution.
    • NaN in posterior_mean raises RuntimeError (filter divergence).
    • Posterior covariance symmetrised numerically after each update.
    • Physical clamps applied post-update (jnp.maximum, not if/else).

References
──────────
  Merwe R. & Wan E.A. (2000) Proc. ASSPCC — Unscented Kalman Filter (UKF)
  Merwe R. (2004) PhD thesis — σ-point weight parametrisation
  Clarke D.C. & Skiba P.F. (2013) Am J Physiol — W' balance model
  Coyle E.F. & Gonzalez-Alonso J. (2001) J Appl Physiol — CV drift
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

try:
    from dynamax.nonlinear_gaussian_ssm import UnscentedKalmanFilter, ParamsNLGSSM
    _DYNAMAX_OK = True
except ImportError:
    UnscentedKalmanFilter = None  # type: ignore[assignment,misc]
    ParamsNLGSSM          = None  # type: ignore[assignment,misc]
    _DYNAMAX_OK = False

from app.slices.cardiorespiratory.ode import (
    CardioSliceParams,
    DEFAULT_CARDIO_SLICE_PARAMS,
    X0_CARDIO_DEFAULT,
    P0_CARDIO_DEFAULT,
    STATE_DIM,
    OBS_DIM,
    IDX_VO2, IDX_HR, IDX_SV, IDX_WPRIME, IDX_RF, IDX_AT,
    cardiorespiratory_slice_ode,
)
from app.slices.cardiorespiratory.observation import (
    CardioObsParams,
    DEFAULT_OBS_PARAMS,
    h_cardio,
    h_cardio_sigma,
    R_DEFAULT,
    inflate_R_per_channel,
    obs_dict_to_array,
)
from app.engine.assimilation.ukf_filter import GaussianState

logger = logging.getLogger(__name__)


# ── UKF Merwe-Wan σ-point parameters (n = STATE_DIM = 6) ─────────────────────
#
# α=0.10 chosen over the conventional 1e-3 for numerical stability in float32:
# with α=1e-3, WM_0 ≈ −999999 and WI ≈ 83333 — the weighted mean requires
# cancellation of terms ~10⁶ apart, which exceeds float32 relative precision
# (~7 decimal digits) and causes NaN after a few sequential UKF steps.
# With α=0.10:  WM_0 = −99, WI = 8.333 — fully representable in float32.
# The cardio ODE is mildly nonlinear over a 1-min window, so larger σ-point
# spread (from larger α) is physicallyacceptable and numerically preferable.

_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM                            # 6

_LAM:   float = _ALPHA ** 2 * (_N + _KAPPA) - _N   # = 0.01×6 − 6 = −5.94
_WM_0:  float = _LAM  / (_N + _LAM)                 # = −5.94/0.06 = −99
_WC_0:  float = _WM_0 + (1.0 - _ALPHA ** 2 + _BETA) # = −99 + 2.99 = −96.01
_WI:    float = 0.5   / (_N + _LAM)                 # = 0.5/0.06 = 8.333

_WM: jax.Array = jnp.array([_WM_0] + [_WI] * (2 * _N), dtype=jnp.float32)
_WC: jax.Array = jnp.array([_WC_0] + [_WI] * (2 * _N), dtype=jnp.float32)


# ── Process noise Q (diagonal, Δt = 1 min) ───────────────────────────────────
# Each entry reflects expected state variability from unmodelled disturbances
# over a 1-minute integration window.

_Q_DIAG: jax.Array = jnp.array([
    4.0,     # V_O2   [(mL/kg/min)²] — VO2 kinetic variability over 1 min
    9.0,     # HR     [bpm²]         — HR noise + autonomic fluctuation
    2.5e-4,  # SV     [L²]           — SV variability (σ ≈ 0.016 L)
    1.0,     # W'_bal [kJ²]          — W' kinetic variability (σ ≈ 1 kJ/min)
    1.0e-3,  # RF     [adim²]        — respiratory fatigue variability
    1.0e-3,  # AT     [adim²]        — autonomic tone fluctuation (σ ≈ 0.032)
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


# ── Transition parameters ─────────────────────────────────────────────────────

class CardioTransitionParams(NamedTuple):
    """
    Full parameter set for the 1-min cardio ODE transition step.

    Carries CardioSliceParams (personalised by NLME) plus the integration
    window and default hub values for the predict step.
    """
    cardio:         CardioSliceParams   # ODE kinetic parameters (personalised)
    dt_min:         float = 1.0         # [min] — UKF integration window
    hub_T_core:     float = 37.0        # [°C]  — default (overridden by real-time)
    hub_pv_drop_pct: float = 0.0        # [%]   — default (overridden by real-time)


DEFAULT_TRANSITION_PARAMS: CardioTransitionParams = CardioTransitionParams(
    cardio = DEFAULT_CARDIO_SLICE_PARAMS,
)


# ── JIT-safe 1-min ODE step ───────────────────────────────────────────────────

def _integrate_1min(
    x:      jax.Array,
    u:      jax.Array,
    params: CardioTransitionParams,
) -> jax.Array:
    """
    Integrate the cardiorespiratory ODE for params.dt_min minutes.

    vmap-compatible: only x varies across sigma points; u and params are
    broadcast (non-mapped) within jax.vmap.

    Parameters
    ----------
    x      : (STATE_DIM,) — current state
    u      : (3,) — [power_watts, hub_T_core, hub_pv_drop_pct]
    params : CardioTransitionParams

    Returns
    -------
    x_next : (STATE_DIM,) — state after dt_min minutes
    """
    power   = u[0]
    t_core  = u[1]
    pv_drop = u[2]
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(cardiorespiratory_slice_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.asarray(params.dt_min, dtype=jnp.float32),
        dt0       = jnp.float32(0.1),   # 6-s initial step
        y0        = x,
        args      = (params.cardio, power, t_core, pv_drop),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 32,
    )
    return sol.ys[0]


# Small PSD jitter added before Cholesky to absorb float32 rounding drift
# that accumulates over many sequential UKF update steps. Value chosen so
# the added uncertainty (~1e-6 in each state dimension) is well below the
# minimum process noise floor in Q_DEFAULT.
_COV_JITTER: float = 1e-6

# ── Pure-JAX UKF kernels ──────────────────────────────────────────────────────

def _sigma_points(mean: jax.Array, cov: jax.Array) -> jax.Array:
    """
    Merwe-Wan 2n+1 sigma points.
    x_0 = μ; x_i = μ + L.T[i-1]; x_{n+i} = μ − L.T[i-1]
    L = chol((n+λ)·Σ_reg)  where Σ_reg = Σ + ε·I ensures PSD after float32 drift.
    Returns (2n+1, STATE_DIM).
    """
    n = mean.shape[0]
    # Symmetrise + tiny jitter before Cholesky to absorb float32 rounding drift
    cov_sym = 0.5 * (cov + cov.T) + _COV_JITTER * jnp.eye(n, dtype=mean.dtype)
    L = jnp.linalg.cholesky((n + _LAM) * cov_sym)
    pos = mean[None, :] + L.T
    neg = mean[None, :] - L.T
    return jnp.concatenate([mean[None, :], pos, neg], axis=0)


def _recover_mean_cov(
    pts:       jax.Array,
    noise_cov: jax.Array,
    Wm:        jax.Array,
    Wc:        jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Recover (mean, cov) from propagated sigma points + additive noise."""
    mean = jnp.einsum("i,ij->j", Wm, pts)
    diff = pts - mean[None, :]
    cov  = jnp.einsum("i,ij,ik->jk", Wc, diff, diff) + noise_cov
    return mean, cov


@jax.jit
def _ukf_predict_cardio(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: CardioTransitionParams,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF predict step: unscented transform through the 1-min ODE.

    Propagates 2n+1 = 13 sigma points through _integrate_1min via vmap.

    Parameters
    ----------
    mean   : (STATE_DIM,)
    cov    : (STATE_DIM, STATE_DIM)
    u      : (3,) — [power_watts, hub_T_core, hub_pv_drop_pct]
    params : CardioTransitionParams
    Q      : (STATE_DIM, STATE_DIM) — process noise

    Returns
    -------
    (mean_pred, cov_pred)
    """
    sigma      = _sigma_points(mean, cov)                     # (2n+1, STATE_DIM)
    sigma_next = jax.vmap(
        _integrate_1min, in_axes=(0, None, None)
    )(sigma, u, params)                                        # (2n+1, STATE_DIM)
    mean_pred, cov_pred = _recover_mean_cov(sigma_next, Q, _WM, _WC)
    # Symmetrise predicted covariance to prevent eigenvalue drift
    cov_pred = 0.5 * (cov_pred + cov_pred.T)
    return mean_pred, cov_pred


@jax.jit
def _ukf_update_cardio(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: CardioObsParams,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF measurement update for the 3-channel observation.

    Uses h_cardio_sigma (sigma-point propagation through h).

    Parameters
    ----------
    mean_pred  : (STATE_DIM,)
    cov_pred   : (STATE_DIM, STATE_DIM)
    y_obs      : (OBS_DIM,) = (3,) — [HR, VO2, RMSSD]; NaN-safe (inflation handled upstream)
    obs_params : CardioObsParams
    R          : (OBS_DIM, OBS_DIM) — potentially per-channel inflated noise

    Returns
    -------
    (posterior_mean, posterior_cov)
    """
    sigma    = _sigma_points(mean_pred, cov_pred)          # (2n+1, STATE_DIM)
    y_sigma  = h_cardio_sigma(sigma, obs_params)           # (2n+1, OBS_DIM)

    y_mean   = jnp.einsum("i,ij->j", _WM, y_sigma)        # (OBS_DIM,)
    dy_s     = y_sigma - y_mean[None, :]                   # (2n+1, OBS_DIM)
    dx_s     = sigma   - mean_pred[None, :]                # (2n+1, STATE_DIM)

    S_yy     = jnp.einsum("i,ij,ik->jk", _WC, dy_s, dy_s) + R  # (OBS_DIM, OBS_DIM)
    P_xy     = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)      # (STATE_DIM, OBS_DIM)

    K                = P_xy @ jnp.linalg.inv(S_yy)               # (STATE_DIM, OBS_DIM)
    innovation       = y_obs - y_mean                             # (OBS_DIM,)
    posterior_mean   = mean_pred + K @ innovation
    posterior_cov    = cov_pred  - K @ S_yy @ K.T

    return posterior_mean, posterior_cov


# ── Physical clamps (mandatory post-update) ───────────────────────────────────

def _apply_physical_clamps(
    mean:   jax.Array,
    params: CardioSliceParams,
) -> jax.Array:
    """
    Apply physiological lower bounds via jnp.maximum / jnp.clip.

    MANDATORY: called after every UKF update step to prevent the linear
    Kalman gain from pushing states into physiologically impossible territory.
    Uses only jnp.maximum / jnp.clip — no Python control flow.

    Bounds
    ------
    V_O2          ≥ 0           — oxygen consumption never negative
    Heart_Rate    ≥ HR_floor    — absolute bradycardia sentinel
    Stroke_Volume ≥ 0.020 L     — minimal viable stroke volume
    W_prime_bal   ≥ 0           — battery cannot drop below zero
    Resp_Fatigue  ∈ [0, 1]      — bounded probability-like variable
    Autonomic_Tone ∈ [0, 1]     — bounded tone variable
    """
    mean = mean.at[IDX_VO2].set(
        jnp.maximum(jnp.float32(0.0), mean[IDX_VO2])
    )
    mean = mean.at[IDX_HR].set(
        jnp.maximum(jnp.float32(params.HR_floor), mean[IDX_HR])
    )
    mean = mean.at[IDX_SV].set(
        jnp.maximum(jnp.float32(0.020), mean[IDX_SV])
    )
    mean = mean.at[IDX_WPRIME].set(
        jnp.maximum(jnp.float32(0.0), mean[IDX_WPRIME])
    )
    mean = mean.at[IDX_RF].set(
        jnp.clip(mean[IDX_RF], jnp.float32(0.0), jnp.float32(1.0))
    )
    mean = mean.at[IDX_AT].set(
        jnp.clip(mean[IDX_AT], jnp.float32(0.0), jnp.float32(1.0))
    )
    return mean


# ── Public filter class ───────────────────────────────────────────────────────

class CardioStateFilter:
    """
    L4 Unscented Kalman Filter for the 6-state cardiorespiratory performance system.

    Infers V_O2, HR, SV, W'_bal, Resp_Fatigue, and Autonomic_Tone from
    the wearable HR signal (dense) plus sparse VO2 / RMSSD measurements.

    Typical usage (single 1-min HR update)
    ──────────────────────────────────────
    filt  = CardioStateFilter()
    state = GaussianState(mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT)
    trans = CardioTransitionParams(cardio=my_nlme_params, dt_min=1.0)

    # Each 1-min wearable sample:
    state = filt.update_state(
        prior        = state,
        observations = {"HR_obs_bpm": 145.0, "VO2_obs": float("nan"),
                        "RMSSD_obs_ms": float("nan")},
        controls     = {"power_watts": 220.0, "hub_T_core": 38.0,
                        "hub_pv_drop_pct": 3.5},
        params       = trans,
    )

    Fail-Loud contract
    ──────────────────
    RuntimeError if posterior_mean contains NaN (filter divergence).
    Physical clamps applied before divergence check (jnp.maximum, not Python if).
    All other exceptions propagate unmodified.
    """

    def __init__(
        self,
        Q:          jax.Array | None = None,
        R:          jax.Array | None = None,
        obs_params: CardioObsParams  | None = None,
    ) -> None:
        self.Q          = Q          if Q          is not None else Q_DEFAULT
        self.R          = R          if R          is not None else R_DEFAULT
        self.obs_params = obs_params if obs_params is not None else DEFAULT_OBS_PARAMS

        if _DYNAMAX_OK:
            self._dyn_ukf = UnscentedKalmanFilter(STATE_DIM, OBS_DIM)
            logger.info(
                "CardioStateFilter — dynamax UKF backend registered "
                "(batch filtering available)."
            )
        else:
            self._dyn_ukf = None
            logger.warning(
                "CardioStateFilter — dynamax not installed; "
                "single-step update_state available only."
            )

    # ── Primary public API: single 1-min update ───────────────────────────

    def update_state(
        self,
        prior:        GaussianState,
        observations: dict[str, float],
        controls:     dict[str, float],
        params:       CardioTransitionParams,
        quality_flag: int = 0,
    ) -> GaussianState:
        """
        Assimilate one 1-min observation bundle into the cardio state estimate.

        Two-step UKF:
            1. Predict: propagate prior through the 1-min ODE (f_transition)
            2. Update:  correct with available observations (h_cardio + inflated R)
            3. Clamp:   apply physical bounds via jnp.maximum (mandatory)

        Parameters
        ----------
        prior        : GaussianState(mean, cov) — state at previous 1-min step
        observations : dict with zero or more of:
                         "HR_obs_bpm"   float [bpm]         — HR wearable
                         "VO2_obs"      float [mL/kg/min]   — metabolic cart
                         "RMSSD_obs_ms" float [ms]          — HRV morning
                       Missing keys or NaN values → predict-only for that channel
        controls     : dict with:
                         "power_watts"      float [W] — session power demand
                         "hub_T_core"       float [°C] (optional, NaN OK)
                         "hub_pv_drop_pct"  float [%]  (optional, NaN OK)
        params       : CardioTransitionParams — ODE parameters + dt
        quality_flag : int ∈ [0, 4] — overall data quality; inflates all R

        Returns
        -------
        GaussianState — updated (mean, cov) for this 1-min step

        Raises
        ------
        RuntimeError if posterior_mean contains NaN (filter divergence).
        """
        # ── Build control vector ──────────────────────────────────────────
        power   = float(controls.get("power_watts",     0.0))
        t_core  = float(controls.get("hub_T_core",      math.nan))
        pv_drop = float(controls.get("hub_pv_drop_pct", 0.0))
        u = jnp.array([power, t_core, pv_drop], dtype=jnp.float32)

        # ── Predict step ──────────────────────────────────────────────────
        mean_pred, cov_pred = _ukf_predict_cardio(
            prior.mean, prior.cov, u, params, self.Q
        )

        # ── Observation assembly + per-channel R inflation ────────────────
        y_raw          = obs_dict_to_array(observations)
        R_step         = self.R
        if quality_flag > 0:
            from app.slices.cardiorespiratory.observation import inflate_R_for_quality
            R_step = inflate_R_for_quality(quality_flag, R_step)
        R_inflated, y_safe = inflate_R_per_channel(R_step, y_raw)

        # Replace NaN y_safe entries with the predicted observation
        # (zero innovation = predict-only, already handled by 1e8 R inflation;
        # but we must also remove NaN from y_safe to avoid NaN propagation)
        y_predicted = h_cardio(mean_pred, self.obs_params)
        y_final     = jnp.where(jnp.isnan(y_raw), y_predicted, y_safe)

        # ── Update step ───────────────────────────────────────────────────
        posterior_mean, posterior_cov = _ukf_update_cardio(
            mean_pred, cov_pred, y_final, self.obs_params, R_inflated
        )

        # ── MANDATORY physical clamps (jnp.maximum — no Python branching) ──
        posterior_mean = _apply_physical_clamps(posterior_mean, params.cardio)

        # ── Fail-Loud: divergence detection ───────────────────────────────
        if bool(jnp.any(jnp.isnan(posterior_mean))):
            raise RuntimeError(
                "CardioStateFilter.update_state: posterior_mean contains NaN. "
                f"HR obs: {observations.get('HR_obs_bpm')}, "
                f"quality_flag: {quality_flag}. "
                "Verify Q/R matrices and CardioTransitionParams."
            )

        # Numerically symmetrise covariance (prevent eigenvalue drift from
        # accumulated floating-point asymmetry over many update steps)
        posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)

        return GaussianState(mean=posterior_mean, cov=posterior_cov)

    # ── Batch filtering API (dynamax backend) ─────────────────────────────

    def build_dynamax_params(
        self,
        transition_params: CardioTransitionParams,
        initial_mean:      jax.Array | None = None,
        initial_cov:       jax.Array | None = None,
    ) -> "ParamsNLGSSM":
        """
        Build a dynamax ParamsNLGSSM for batch time-series filtering.

        Raises RuntimeError if dynamax is not installed.
        """
        if not _DYNAMAX_OK:
            raise RuntimeError(
                "dynamax>=0.1 is required for build_dynamax_params. "
                "Install: pip install dynamax"
            )

        _tp         = transition_params
        _obs_params = self.obs_params

        def _dynamics_fn(x: jax.Array, u: jax.Array) -> jax.Array:
            return _integrate_1min(x, u, _tp)

        def _emission_fn(x: jax.Array, _input: jax.Array) -> jax.Array:
            return h_cardio(x, _obs_params)

        return ParamsNLGSSM(
            initial_mean       = initial_mean if initial_mean is not None
                                 else X0_CARDIO_DEFAULT,
            initial_covariance = initial_cov  if initial_cov  is not None
                                 else P0_CARDIO_DEFAULT,
            dynamics_function  = _dynamics_fn,
            dynamics_covariance= self.Q,
            emission_function  = _emission_fn,
            emission_covariance= self.R,
        )
