"""
app/slices/neural_cognitive/filter.py  — REMASTER v2.0

L4 State Filter — Neural/Cognitive Slice
Unscented Kalman Filter for the 7-state neural/cognitive system.

Architecture (HLD §4.3 — L4: State Estimation)
────────────────────────────────────────────────
State x ∈ ℝ⁷ (hours):
    [Brain_5HT, Brain_DA, Brain_Ammonia, Cerebral_O2_Sat,
     Adenosine_Pool, Caffeine_Plasma, CAR]

Observations y ∈ ℝ²:
    [RPE_Proxy [1–10], PVT_Lapses [count/10 min]]

Transition f(x, u) — ODE advance over dt_hours (diffrax.Tsit5)
────────────────────────────────────────────────────────────────
Time constants (system NOT stiff → Tsit5 appropriate):
    5HT/DA        t½ ≈ 3.5 h
    NH3           t½ ≈ 1.4 h
    Cerebral O2   τ  ≈ 0.5 h (k_o2_relax = 2 h⁻¹)
    Adenosine     t½ ≈ 7 h
    Caffeine      t½ ≈ 5 h (population mean CYP1A2)
    CAR           τ  ≈ 0.5 h (k_car_rec = 2 h⁻¹)

Sparse Assimilation
───────────────────
RPE_Proxy:  available only during/after training sessions.
PVT_Lapses: available only when a morning cognitive test is performed.
flag = 4 → R_channel × 1e8 (predict-only for that channel).

UKF parametrisation (Merwe & Wan 2000)
────────────────────────────────────────
    α = 1e-3, β = 2, κ = 0
    n = 7 → 2n+1 = 15 sigma points

Process noise Q (diagonal, calibrated to 1-hour biological variability):
    Brain_5HT         : (0.10 au)²   — hourly exercise-dependent variation
    Brain_DA          : (0.10 au)²   — COMT-driven variability
    Brain_NH3         : (0.03 au)²   — slow AMP deamination dynamics
    Cerebral_O2_Sat   : (0.02)²      — tightly CBF-regulated
    Adenosine_Pool    : (0.05 au)²   — slow adenosine dynamics
    Caffeine_Plasma   : (0.50 mg/L)² — depends on intake variability
    CAR               : (0.05)²      — fast recovery; well constrained

Fail-Loud contract
──────────────────
RuntimeError if posterior_mean contains NaN.
RuntimeError if posterior_cov diagonal contains negative values.

References
──────────
    Merwe & Wan (2000) Proc ASSPCC
    Nehlig A. (2010) Neurosci Biobehav Rev 35(2):430
    Van Dongen H.P.A. et al. (2003) Sleep 26(2):117
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

try:
    from dynamax.nonlinear_gaussian_ssm import UnscentedKalmanFilter, ParamsNLGSSM
    _DYNAMAX_OK = True
except ImportError:
    UnscentedKalmanFilter = None   # type: ignore[assignment,misc]
    ParamsNLGSSM          = None   # type: ignore[assignment,misc]
    _DYNAMAX_OK = False

from app.slices.neural_cognitive.ode import (
    NeuralCognitiveParams,
    DEFAULT_NC_PARAMS,
    X0_NC_DEFAULT,
    P0_NC_DEFAULT,
    STATE_DIM,
    CTRL_DIM,
    neural_cognitive_ode,
    IDX_5HT,
    IDX_DA,
    IDX_NH3,
    IDX_O2SAT,
    IDX_ADEN,
    IDX_CAF,
    IDX_CAR,
)
from app.slices.neural_cognitive.observation import (
    NeuralCogObsParams,
    DEFAULT_NC_OBS_PARAMS,
    OBS_DIM,
    h_nc,
    h_nc_sigma,
    R_NC_DEFAULT,
    inflate_R_nc,
)
from app.engine.assimilation.ukf_filter import GaussianState

logger = logging.getLogger(__name__)


# ── UKF Merwe-Wan σ-point parameters (n = STATE_DIM = 7) ─────────────────────
#
# alpha = 0.5 (NOT 1e-3): with n=7, alpha=1e-3 gives n+lambda ≈ 7e-6 and
# weights of ±1e6, causing float32 catastrophic cancellation (WM_0 ≈ -999999,
# WM_i ≈ 71429; sum OK in exact arithmetic, but not in float32).
# alpha = 0.5 → n+lambda = 1.75, WM_0 = -3.0, WM_i = 0.286 — no cancellation.

_ALPHA: float = 0.5
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM                           # 7

_LAM:   float = _ALPHA**2 * (_N + _KAPPA) - _N    # = 0.25×7 − 7 = −5.25
_WM_0:  float = _LAM  / (_N + _LAM)               # = −5.25/1.75 = −3.0
_WC_0:  float = _WM_0 + (1.0 - _ALPHA**2 + _BETA) # = −3.0 + 2.75 = −0.25
_WI:    float = 0.5   / (_N + _LAM)               # = 0.5/1.75 ≈ 0.2857

_WM: jax.Array = jnp.array([_WM_0] + [_WI] * (2 * _N), dtype=jnp.float32)
_WC: jax.Array = jnp.array([_WC_0] + [_WI] * (2 * _N), dtype=jnp.float32)


# ── Process noise Q (diagonal, Δt = 1 hour) ──────────────────────────────────

_Q_DIAG: jax.Array = jnp.array([
    0.0100,   # Brain_5HT        [au]²  = (0.10)²  — exercise variation
    0.0100,   # Brain_DA         [au]²  = (0.10)²  — COMT-driven variability
    0.0009,   # Brain_NH3        [au]²  = (0.03)²  — slow AMP deamination
    0.0004,   # Cerebral_O2_Sat  [.]²   = (0.02)²  — CBF-regulated; tight
    0.0025,   # Adenosine_Pool   [au]²  = (0.05)²  — slow adenosine dynamics
    0.2500,   # Caffeine_Plasma  [mg/L]²= (0.50)²  — intake variability
    0.0025,   # CAR              [.]²   = (0.05)²  — fast recovery; tight
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


# ── Transition parameters ─────────────────────────────────────────────────────

class NeuralCogTransitionParams(NamedTuple):
    """Full parameter set for the neural/cognitive ODE transition step."""
    nc:       NeuralCognitiveParams = DEFAULT_NC_PARAMS
    dt_hours: float                 = 1.0


DEFAULT_NC_TRANSITION_PARAMS: NeuralCogTransitionParams = NeuralCogTransitionParams()


# ── JIT-safe ODE step ─────────────────────────────────────────────────────────

def _integrate_step(
    x:      jax.Array,
    u:      jax.Array,
    params: NeuralCogTransitionParams,
) -> jax.Array:
    """
    Integrate the 7-state neural/cognitive ODE for params.dt_hours hours.

    vmap-compatible: only x varies across sigma points.
    dt0 = 0.05 h (fixed; JIT-safe — not derived from params.dt_hours).
    """
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(neural_cognitive_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.asarray(params.dt_hours, dtype=jnp.float32),
        dt0       = jnp.float32(0.05),   # fixed 3-min step; JIT-safe
        y0        = x,
        args      = (params.nc, u),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 512,
    )
    return sol.ys[0]


# ── Pure-JAX UKF kernels ──────────────────────────────────────────────────────

def _sigma_points(mean: jax.Array, cov: jax.Array) -> jax.Array:
    """Merwe-Wan 2n+1 sigma points. Returns shape (15, 7)."""
    n = mean.shape[0]
    L = jnp.linalg.cholesky((_N + _LAM) * cov)
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
def _ukf_predict(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: NeuralCogTransitionParams,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """UKF predict: unscented transform through 7-state ODE. 15 sigma points."""
    Q_scaled  = Q * jnp.float32(params.dt_hours)
    sigma     = _sigma_points(mean, cov)
    sigma_nxt = jax.vmap(
        _integrate_step, in_axes=(0, None, None)
    )(sigma, u, params)
    return _recover_mean_cov(sigma_nxt, Q_scaled, _WM, _WC)


@jax.jit
def _ukf_update(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: NeuralCogObsParams,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """UKF measurement update ([RPE_Proxy, PVT_Lapses])."""
    sigma   = _sigma_points(mean_pred, cov_pred)       # (15, 7)
    y_sigma = h_nc_sigma(sigma, obs_params)             # (15, 2)

    y_mean = jnp.einsum("i,ij->j", _WM, y_sigma)       # (2,)
    dy_s   = y_sigma  - y_mean[None, :]                 # (15, 2)
    dx_s   = sigma    - mean_pred[None, :]               # (15, 7)

    S_yy   = jnp.einsum("i,ij,ik->jk", _WC, dy_s, dy_s) + R   # (2,2)
    P_xy   = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)        # (7,2)

    K              = P_xy @ jnp.linalg.inv(S_yy)
    innovation     = y_obs - y_mean
    posterior_mean = mean_pred + K @ innovation
    posterior_cov  = cov_pred  - K @ S_yy @ K.T

    return posterior_mean, posterior_cov


# ── Public filter class ───────────────────────────────────────────────────────

class NeuralCognitiveStateFilter:
    """
    L4 Unscented Kalman Filter for the 7-state Neural/Cognitive system.

    Sparse assimilation: RPE (post-session) + PVT_Lapses (morning test).
    Most steps are predict-only (quality_flag = 4).

    Dynamic dt_hours: intra-session (≤ 1 h) to between-session (≤ 24 h).
    Q is scaled by dt_hours per call.

    Fail-Loud contract
    ──────────────────
    RuntimeError on NaN posterior_mean.
    RuntimeError on non-PSD covariance.
    """

    def __init__(
        self,
        Q:          jax.Array | None = None,
        R:          jax.Array | None = None,
        obs_params: NeuralCogObsParams | None = None,
    ) -> None:
        self.Q          = Q          if Q          is not None else Q_DEFAULT
        self.R          = R          if R          is not None else R_NC_DEFAULT
        self.obs_params = obs_params if obs_params is not None else DEFAULT_NC_OBS_PARAMS

        if _DYNAMAX_OK:
            self._dyn_ukf = UnscentedKalmanFilter(STATE_DIM, OBS_DIM)
            logger.info("NeuralCognitiveStateFilter — dynamax backend registered.")
        else:
            self._dyn_ukf = None
            logger.warning(
                "NeuralCognitiveStateFilter — dynamax not installed; "
                "single-step only."
            )

    # ── Primary API: single-step update ──────────────────────────────────────

    def update_state(
        self,
        prior:         GaussianState,
        controls:      dict[str, float],
        dt_hours:      float = 1.0,
        rpe_proxy:     float = float("nan"),
        pvt_lapses:    float = float("nan"),
        quality_flags: tuple[int, int] = (4, 4),
        params:        NeuralCogTransitionParams | None = None,
    ) -> GaussianState:
        """
        Assimilate one time step.

        Parameters
        ----------
        prior         : GaussianState(mean ∈ ℝ⁷, cov ∈ ℝ⁷ˣ⁷)
        controls      : dict with keys:
                        'hub_training_stress', 'hub_muscle_damage', 'hub_T_core',
                        'hub_sleep_debt', 'hub_IL6', 'hub_metabolic_stress',
                        'caffeine_intake_plasma', 'hub_hypoglycemia'
        dt_hours      : float — integration window [h]
        rpe_proxy     : float [1–10]; NaN = unavailable
        pvt_lapses    : float [count/10 min]; NaN = unavailable
        quality_flags : (flag_RPE, flag_PVT); 4 = predict-only
        params        : NeuralCogTransitionParams (defaults to population)

        Returns
        -------
        GaussianState — posterior (mean, cov)

        Raises
        ------
        RuntimeError on NaN posterior or non-PSD covariance.
        """
        if params is None:
            params = NeuralCogTransitionParams(nc=DEFAULT_NC_PARAMS, dt_hours=dt_hours)
        else:
            params = params._replace(dt_hours=dt_hours)

        u = jnp.array([
            float(controls.get("hub_training_stress",   0.0)),
            float(controls.get("hub_muscle_damage",     0.0)),
            float(controls.get("hub_T_core",            37.0)),
            float(controls.get("hub_sleep_debt",        0.0)),
            float(controls.get("hub_IL6",               0.0)),
            float(controls.get("hub_metabolic_stress",  0.0)),
            float(controls.get("caffeine_intake_plasma", 0.0)),
            float(controls.get("hub_hypoglycemia",      0.0)),
        ], dtype=jnp.float32)

        # ── Predict ───────────────────────────────────────────────────────────
        mean_pred, cov_pred = _ukf_predict(prior.mean, prior.cov, u, params, self.Q)

        # ── Inflate R for unavailable channels ───────────────────────────────
        obs_vals = [rpe_proxy, pvt_lapses]
        flags    = list(quality_flags)
        for ch_idx, val in enumerate(obs_vals):
            if val != val:   # IEEE NaN check
                flags[ch_idx] = 4
        R_step = inflate_R_nc((flags[0], flags[1]), self.R)

        # Replace NaN obs with predicted observation (pure predict for that channel)
        y_predicted = h_nc(mean_pred, self.obs_params)
        y_obs = jnp.array([
            y_predicted[0] if obs_vals[0] != obs_vals[0] else float(obs_vals[0]),
            y_predicted[1] if obs_vals[1] != obs_vals[1] else float(obs_vals[1]),
        ], dtype=jnp.float32)

        # ── Update ────────────────────────────────────────────────────────────
        posterior_mean, posterior_cov = _ukf_update(
            mean_pred, cov_pred, y_obs, self.obs_params, R_step
        )

        # ── Physics-law bounds enforcement ────────────────────────────────────
        _EPS_CONC = jnp.float32(1e-4)
        posterior_mean = posterior_mean.at[IDX_5HT].set(
            jnp.maximum(posterior_mean[IDX_5HT], _EPS_CONC)
        )
        posterior_mean = posterior_mean.at[IDX_DA].set(
            jnp.maximum(posterior_mean[IDX_DA], _EPS_CONC)
        )
        posterior_mean = posterior_mean.at[IDX_NH3].set(
            jnp.clip(posterior_mean[IDX_NH3], jnp.float32(0.0), jnp.float32(1.0))
        )
        posterior_mean = posterior_mean.at[IDX_O2SAT].set(
            jnp.clip(posterior_mean[IDX_O2SAT], jnp.float32(0.0), jnp.float32(1.0))
        )
        posterior_mean = posterior_mean.at[IDX_ADEN].set(
            jnp.maximum(posterior_mean[IDX_ADEN], jnp.float32(0.0))
        )
        posterior_mean = posterior_mean.at[IDX_CAF].set(
            jnp.maximum(posterior_mean[IDX_CAF], jnp.float32(0.0))
        )
        posterior_mean = posterior_mean.at[IDX_CAR].set(
            jnp.clip(posterior_mean[IDX_CAR], jnp.float32(0.0), jnp.float32(1.0))
        )

        # ── Fail-Loud checks ──────────────────────────────────────────────────
        if bool(jnp.any(jnp.isnan(posterior_mean))):
            raise RuntimeError(
                "NeuralCognitiveStateFilter.update_state: posterior_mean contains NaN. "
                f"rpe_proxy={rpe_proxy}, pvt_lapses={pvt_lapses}, "
                f"quality_flags={quality_flags}, dt_hours={dt_hours}."
            )
        diag = jnp.diag(posterior_cov)
        if bool(jnp.any(diag < jnp.float32(-1e-6))):
            raise RuntimeError(
                "NeuralCognitiveStateFilter.update_state: posterior_cov has negative "
                "diagonal — filter diverged. Increase Q or R."
            )

        posterior_cov = jnp.float32(0.5) * (posterior_cov + posterior_cov.T)
        return GaussianState(mean=posterior_mean, cov=posterior_cov)

    # ── Batch filtering (dynamax backend) ─────────────────────────────────────

    def build_dynamax_params(
        self,
        transition_params: NeuralCogTransitionParams,
        initial_mean:      jax.Array | None = None,
        initial_cov:       jax.Array | None = None,
    ) -> "ParamsNLGSSM":
        if not _DYNAMAX_OK:
            raise RuntimeError("dynamax>=0.1 required for build_dynamax_params.")
        _tp         = transition_params
        _obs_params = self.obs_params

        def _dynamics_fn(x: jax.Array, u: jax.Array) -> jax.Array:
            return _integrate_step(x, u, _tp)

        def _emission_fn(x: jax.Array, _: jax.Array) -> jax.Array:
            return h_nc(x, _obs_params)

        return ParamsNLGSSM(
            initial_mean       = initial_mean if initial_mean is not None else X0_NC_DEFAULT,
            initial_covariance = initial_cov  if initial_cov  is not None else P0_NC_DEFAULT,
            dynamics_function  = _dynamics_fn,
            dynamics_covariance= self.Q,
            emission_function  = _emission_fn,
            emission_covariance= self.R,
        )

    def filter_time_series(
        self,
        emissions:         jax.Array,
        inputs:            jax.Array,
        transition_params: NeuralCogTransitionParams,
        initial_mean:      jax.Array | None = None,
        initial_cov:       jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Batch UKF filter over a time series.

        Parameters
        ----------
        emissions  : (T, OBS_DIM) = (T, 2); NaN accepted
        inputs     : (T, CTRL_DIM) = (T, 8)

        Returns
        -------
        (filtered_means, filtered_covs) : (T,7), (T,7,7)
        """
        if not _DYNAMAX_OK:
            raise RuntimeError("dynamax>=0.1 required for filter_time_series.")
        dyn_params = self.build_dynamax_params(transition_params, initial_mean, initial_cov)
        posterior  = self._dyn_ukf.filter(dyn_params, emissions, inputs)
        return posterior.filtered_means, posterior.filtered_covariances
