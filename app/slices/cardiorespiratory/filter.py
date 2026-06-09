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
from app.engine.assimilation.ukf_filter import (
    GaussianState,
    nearest_psd,
    variance_floor,
    ukf_weights,
    sigma_points,
    unscented_transform,
    scale_Q,
    lower_clamp_moments,
    range_clamp_moments,
    clamp_dim,
)

logger = logging.getLogger(__name__)


# ── UKF Merwe-Wan sigma-point parameters (n = STATE_DIM = 6) ─────────────────
#
# alpha=0.10: float32-safe for n=6 (Wm[0] = -99.0; no catastrophic cancellation).
# See ukf_filter.py module docstring for the full float32 safety analysis.

_N:     int   = STATE_DIM   # 6
_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0

_WM, _WC, _LAM = ukf_weights(_N, _ALPHA, _BETA, _KAPPA)

# Simon 2010 variance floor -- minimum diagonal variance after truncated-normal
# clamping.  Values chosen so the floor is well below sensor noise (R diagonal)
# but prevents the filter from claiming zero uncertainty on any state.
_VAR_FLOOR: jax.Array = jnp.array([
    1.0e-2,   # VO2   (0.1 mL/kg/min)^2
    2.5e-1,   # HR    (0.5 bpm)^2
    1.0e-6,   # SV    (0.001 L)^2
    1.0e-4,   # W'    (0.01 kJ)^2
    1.0e-6,   # RF    (0.001 adim)^2
    1.0e-6,   # AT    (0.001 adim)^2
], dtype=jnp.float32)


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


# ── Pure-JAX UKF kernels (use shared primitives from ukf_filter.py) ───────────

@jax.jit
def _ukf_predict_cardio(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: CardioTransitionParams,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF predict step: unscented transform through the dt_min ODE.

    Uses shared sigma_points() (eigh-based, PSD-robust) instead of the old
    Cholesky path which could fail on float32 non-PSD covariances.
    Q must already be scaled to dt_real before calling (via scale_Q).

    Parameters
    ----------
    mean   : (STATE_DIM,)
    cov    : (STATE_DIM, STATE_DIM)
    u      : (3,) -- [power_watts, hub_T_core, hub_pv_drop_pct]
    params : CardioTransitionParams
    Q      : (STATE_DIM, STATE_DIM) -- dt-scaled process noise

    Returns
    -------
    (mean_pred, cov_pred)
    """
    pts        = sigma_points(mean, cov, _LAM)               # (2n+1, STATE_DIM)
    pts_next   = jax.vmap(
        _integrate_1min, in_axes=(0, None, None)
    )(pts, u, params)                                         # (2n+1, STATE_DIM)
    mean_pred, cov_pred = unscented_transform(pts_next, Q, _WM, _WC)
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
    UKF measurement update for the 3-channel cardiorespiratory observation.

    Cross-covariance P_xy and innovation covariance S_yy computed via
    sigma-point propagation through h_cardio. Innovation covariance inverted
    with jnp.linalg.inv (3x3 -- negligible cost).

    Parameters
    ----------
    mean_pred  : (STATE_DIM,)
    cov_pred   : (STATE_DIM, STATE_DIM)
    y_obs      : (OBS_DIM,) = (3,) -- [HR, VO2, RMSSD]; NaN replaced upstream
    obs_params : CardioObsParams
    R          : (OBS_DIM, OBS_DIM) -- per-channel inflated noise

    Returns
    -------
    (posterior_mean, posterior_cov)
    """
    pts    = sigma_points(mean_pred, cov_pred, _LAM)   # (2n+1, STATE_DIM)
    y_pts  = h_cardio_sigma(pts, obs_params)           # (2n+1, OBS_DIM)

    y_mean = jnp.einsum("i,ij->j", _WM, y_pts)        # (OBS_DIM,)
    dy     = y_pts - y_mean[None, :]                   # (2n+1, OBS_DIM)
    dx     = pts   - mean_pred[None, :]                # (2n+1, STATE_DIM)

    S_yy   = jnp.einsum("i,ij,ik->jk", _WC, dy, dy) + R   # (OBS_DIM, OBS_DIM)
    P_xy   = jnp.einsum("i,ij,ik->jk", _WC, dx, dy)        # (STATE_DIM, OBS_DIM)

    K            = P_xy @ jnp.linalg.inv(S_yy)              # (STATE_DIM, OBS_DIM)
    innovation   = y_obs - y_mean                            # (OBS_DIM,)
    post_mean    = mean_pred + K @ innovation
    post_cov     = cov_pred  - K @ S_yy @ K.T

    return post_mean, post_cov


# ── Physical clamps — shared Simon 2010 truncated-normal moment matching ───────
#
# lower_clamp_moments, range_clamp_moments, clamp_dim, variance_floor are all
# imported from app.engine.assimilation.ukf_filter (shared module).
# This slice only defines _apply_physical_clamps: the policy (which bounds)
# not the math (which is canonical).

def _apply_physical_clamps(
    mean:   jax.Array,
    cov:    jax.Array,
    params: CardioSliceParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Gaussian-coherent physical clamping via truncated-normal moment matching
    (Simon 2010) followed by Simon variance floor enforcement.

    Step 1 -- truncated-normal moment matching per constrained dimension:
        V_O2          >= 0             mL/kg/min
        Heart_Rate    >= HR_floor      bpm
        Stroke_Volume >= 0.020         L
        W_prime_bal   >= 0             kJ
        Resp_Fatigue  in [0, 1]        adim
        Autonomic_Tone in [0, 1]       adim

    Step 2 -- variance floor (Simon 2010 Section 5.4):
        diag(cov) >= _VAR_FLOOR  (prevents overconfidence after tight clamping)

    Step 3 -- nearest_psd repair:
        Ensures the final cov is strictly PSD for the next sigma-point generation.

    All ops via jnp -- JIT-compatible, no Python control flow on array values.
    """
    # Step 1: truncated-normal moment matching
    m, v = lower_clamp_moments(mean[IDX_VO2],    cov[IDX_VO2, IDX_VO2],       0.0)
    mean, cov = clamp_dim(mean, cov, IDX_VO2, m, v)

    m, v = lower_clamp_moments(mean[IDX_HR],     cov[IDX_HR, IDX_HR],         float(params.HR_floor))
    mean, cov = clamp_dim(mean, cov, IDX_HR, m, v)

    m, v = lower_clamp_moments(mean[IDX_SV],     cov[IDX_SV, IDX_SV],         0.020)
    mean, cov = clamp_dim(mean, cov, IDX_SV, m, v)

    m, v = lower_clamp_moments(mean[IDX_WPRIME], cov[IDX_WPRIME, IDX_WPRIME], 0.0)
    mean, cov = clamp_dim(mean, cov, IDX_WPRIME, m, v)

    m, v = range_clamp_moments(mean[IDX_RF],     cov[IDX_RF, IDX_RF],         0.0, 1.0)
    mean, cov = clamp_dim(mean, cov, IDX_RF, m, v)

    m, v = range_clamp_moments(mean[IDX_AT],     cov[IDX_AT, IDX_AT],         0.0, 1.0)
    mean, cov = clamp_dim(mean, cov, IDX_AT, m, v)

    # Step 2: variance floor
    cov = variance_floor(cov, _VAR_FLOOR)

    # Step 3: nearest PSD repair (absorbs any residual float32 drift)
    cov = nearest_psd(cov)

    return mean, cov


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
    Physical clamps (truncated-normal moment matching) applied before divergence
    check — updates both mean and covariance for Gaussian coherence.
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
        quality_flag: int   = 0,
        dt_real:      float = 1.0,
    ) -> GaussianState:
        """
        Assimilate one observation bundle into the cardio state estimate.

        Three-step UKF (Simon 2010 truncated UKF):
            1. Predict: sigma points through dt_real-min ODE; Q scaled by dt_real
            2. Update:  sigma points through h_cardio; per-channel R inflation
            3. Clamp:   truncated-normal moment matching + variance floor + nearest_psd

        Parameters
        ----------
        prior        : GaussianState(mean, cov)
        observations : dict with zero or more of:
                         "HR_obs_bpm"   float [bpm]         -- HR wearable
                         "VO2_obs"      float [mL/kg/min]   -- metabolic cart
                         "RMSSD_obs_ms" float [ms]          -- HRV morning
                       Missing keys or NaN values -> predict-only for that channel.
        controls     : dict with:
                         "power_watts"      float [W]
                         "hub_T_core"       float [deg C]  (optional, NaN OK)
                         "hub_pv_drop_pct"  float [%]      (optional, NaN OK)
        params       : CardioTransitionParams
        quality_flag : int in [0, 4] -- inflates all R channels
        dt_real      : float [min] -- actual integration window.
                       Default 1.0 min.  Supply the true inter-sample interval
                       for variable-rate wearable data.
                       Q_step = Q_per_min * dt_real  (linear Wiener scaling).

        Returns
        -------
        GaussianState -- updated (mean, cov)

        Raises
        ------
        RuntimeError if posterior_mean contains NaN (filter divergence).
        """
        # ── Build control vector ──────────────────────────────────────────
        power   = float(controls.get("power_watts",     0.0))
        t_core  = float(controls.get("hub_T_core",      math.nan))
        pv_drop = float(controls.get("hub_pv_drop_pct", 0.0))
        u = jnp.array([power, t_core, pv_drop], dtype=jnp.float32)

        # ── dt-scaled process noise (Wiener scaling: Q grows with dt) ─────
        Q_step = scale_Q(self.Q, dt_real)

        # ── Predict step ──────────────────────────────────────────────────
        mean_pred, cov_pred = _ukf_predict_cardio(
            prior.mean, prior.cov, u, params, Q_step
        )

        # ── Observation assembly + per-channel R inflation ────────────────
        y_raw = obs_dict_to_array(observations)
        R_step = self.R
        if quality_flag > 0:
            from app.slices.cardiorespiratory.observation import inflate_R_for_quality
            R_step = inflate_R_for_quality(quality_flag, R_step)
        R_inflated, y_safe = inflate_R_per_channel(R_step, y_raw)

        # NaN channels -> use predicted observation (1e8 R inflation ensures
        # near-zero Kalman gain; replacing NaN with y_pred prevents NaN math)
        y_predicted = h_cardio(mean_pred, self.obs_params)
        y_final     = jnp.where(jnp.isnan(y_raw), y_predicted, y_safe)

        # ── Update step ───────────────────────────────────────────────────
        posterior_mean, posterior_cov = _ukf_update_cardio(
            mean_pred, cov_pred, y_final, self.obs_params, R_inflated
        )

        # ── Truncated UKF clamps (Simon 2010): moment matching + floor + PSD
        posterior_mean, posterior_cov = _apply_physical_clamps(
            posterior_mean, posterior_cov, params.cardio
        )

        # ── Fail-Loud: divergence detection ───────────────────────────────
        if bool(jnp.any(jnp.isnan(posterior_mean))):
            raise RuntimeError(
                "CardioStateFilter.update_state: posterior_mean contains NaN. "
                f"HR obs: {observations.get('HR_obs_bpm')}, "
                f"dt_real: {dt_real}, quality_flag: {quality_flag}. "
                "Check Q/R matrices and CardioTransitionParams."
            )

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
