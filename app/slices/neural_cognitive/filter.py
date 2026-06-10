"""
app/slices/neural_cognitive/filter.py  -- NC Slice V3.0 (TUKF Gold Standard)

L4 State Filter -- Neural/Cognitive Slice
Unscented Kalman Filter for the 7-state neural/cognitive system.

Architecture (HLD ss4.3 -- L4: State Estimation)
--------------------------------------------------
State x in R^7 (hours):
    [Brain_5HT, Brain_DA, Brain_Ammonia, Cerebral_O2_Sat,
     Adenosine_Pool, Caffeine_Plasma, CAR]

Observations y in R^2:
    [RPE_Proxy [1-10], PVT_Lapses [count/10 min]]

UKF parametrisation (Merwe & Wan 2000):
    alpha = 0.5 (float32-safe for n=7; Wm[0] = -3.0, no cancellation)
    beta  = 2.0, kappa = 0.0

TUKF Blindage (Simon 2010):
    sigma_points : eigh-based (PSD-robust), shared from ukf_filter.py.
    Post-predict : Q scaled by dt_hours (Wiener scaling).
    Post-update  : _apply_physical_clamps_nc:
                     lower_clamp_moments(lb=0) for 5HT, DA, Aden, Caf.
                     range_clamp_moments([0,1]) for NH3, O2Sat, CAR.
                     variance_floor + nearest_psd.
    Fail-Loud    : RuntimeError on NaN posterior_mean or non-PSD diagonal.

Process noise Q (diagonal, calibrated to 1-hour biological variability):
    Brain_5HT         : (0.10 au)^2
    Brain_DA          : (0.10 au)^2
    Brain_NH3         : (0.03 au)^2
    Cerebral_O2_Sat   : (0.02)^2
    Adenosine_Pool    : (0.05 au)^2
    Caffeine_Plasma   : (0.50 mg/L)^2
    CAR               : (0.05)^2

References
----------
    Merwe & Wan (2000) Proc ASSPCC
    Simon D. (2010) Optimal State Estimation, Wiley. Sections 5.3-5.4.
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


# ── UKF Merwe-Wan sigma-point parameters (n = STATE_DIM = 7) ─────────────────
#
# alpha = 0.5: float32-safe for n=7.
# With alpha=1e-3, n=7: Wm[0] ~ -1e6, Wi ~ 71429; catastrophic cancellation in
# float32. alpha=0.5 gives Wm[0] = -3.0, Wi = 0.286 — no cancellation.
# See ukf_filter.py module docstring for the full float32 safety analysis.

_ALPHA: float = 0.5
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM  # 7

_WM, _WC, _LAM = ukf_weights(_N, _ALPHA, _BETA, _KAPPA)


# ── Simon 2010 variance floor (Section 5.4) ───────────────────────────────────

_VAR_FLOOR: jax.Array = jnp.array([
    1e-6,   # Brain_5HT        (0.001 au)^2
    1e-6,   # Brain_DA         (0.001 au)^2
    1e-8,   # Brain_NH3        (0.0001 au)^2
    1e-8,   # Cerebral_O2_Sat  (0.0001 frac)^2
    1e-6,   # Adenosine_Pool   (0.001 au)^2
    1e-4,   # Caffeine_Plasma  (0.01 mg/L)^2
    1e-8,   # CAR              (0.0001)^2
], dtype=jnp.float32)


# ── Process noise Q (diagonal, per hour) ─────────────────────────────────────

_Q_DIAG: jax.Array = jnp.array([
    0.0100,   # Brain_5HT        [au]^2   = (0.10)^2  -- exercise variation
    0.0100,   # Brain_DA         [au]^2   = (0.10)^2  -- COMT-driven variability
    0.0009,   # Brain_NH3        [au]^2   = (0.03)^2  -- slow AMP deamination
    0.0004,   # Cerebral_O2_Sat  [.]^2    = (0.02)^2  -- CBF-regulated; tight
    0.0025,   # Adenosine_Pool   [au]^2   = (0.05)^2  -- slow adenosine dynamics
    0.2500,   # Caffeine_Plasma  [mg/L]^2 = (0.50)^2  -- intake variability
    0.0025,   # CAR              [.]^2    = (0.05)^2  -- fast recovery; tight
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


# ── Transition parameters ─────────────────────────────────────────────────────

class NeuralCogTransitionParams(NamedTuple):
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
    Integrate the 7-state NC ODE for params.dt_hours hours.

    vmap-compatible: only x varies across sigma points.
    dt0 = 0.05 h (fixed; JIT-safe -- not derived from params.dt_hours).
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


# ── Physical clamps (Simon 2010 truncated-normal) ─────────────────────────────

def _apply_physical_clamps_nc(
    mean: jax.Array,
    cov:  jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Gaussian-coherent physical clamping for all 7 NC states.

    Step 1 -- truncated-normal moment matching:
        Brain_5HT       >= 0       (positive concentration)
        Brain_DA        >= 0       (positive concentration)
        Brain_NH3       in [0, 1]  (normalised toxicity index)
        Cerebral_O2_Sat in [0, 1]  (fractional oxygenation)
        Adenosine_Pool  >= 0       (accumulation pool)
        Caffeine_Plasma >= 0       (non-negative plasma concentration)
        CAR             in [0, 1]  (voluntary drive fraction)

    Step 2 -- variance floor (Simon 2010 Section 5.4).
    Step 3 -- nearest_psd repair (Higham 1988).
    """
    m, v = lower_clamp_moments(mean[IDX_5HT], cov[IDX_5HT, IDX_5HT], 0.0)
    mean, cov = clamp_dim(mean, cov, IDX_5HT, m, v)

    m, v = lower_clamp_moments(mean[IDX_DA], cov[IDX_DA, IDX_DA], 0.0)
    mean, cov = clamp_dim(mean, cov, IDX_DA, m, v)

    m, v = range_clamp_moments(mean[IDX_NH3], cov[IDX_NH3, IDX_NH3], 0.0, 1.0)
    mean, cov = clamp_dim(mean, cov, IDX_NH3, m, v)

    m, v = range_clamp_moments(mean[IDX_O2SAT], cov[IDX_O2SAT, IDX_O2SAT], 0.0, 1.0)
    mean, cov = clamp_dim(mean, cov, IDX_O2SAT, m, v)

    m, v = lower_clamp_moments(mean[IDX_ADEN], cov[IDX_ADEN, IDX_ADEN], 0.0)
    mean, cov = clamp_dim(mean, cov, IDX_ADEN, m, v)

    m, v = lower_clamp_moments(mean[IDX_CAF], cov[IDX_CAF, IDX_CAF], 0.0)
    mean, cov = clamp_dim(mean, cov, IDX_CAF, m, v)

    m, v = range_clamp_moments(mean[IDX_CAR], cov[IDX_CAR, IDX_CAR], 0.0, 1.0)
    mean, cov = clamp_dim(mean, cov, IDX_CAR, m, v)

    cov = variance_floor(cov, _VAR_FLOOR)
    cov = nearest_psd(cov)
    return mean, cov


# ── Pure-JAX UKF kernels ───────────────────────────────────────────────────────

@jax.jit
def _ukf_predict_nc(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: NeuralCogTransitionParams,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF predict: eigh-based sigma points through the dt_hours NC ODE.

    Q must be pre-scaled to dt_hours (via scale_Q) before calling.

    Parameters
    ----------
    mean   : (STATE_DIM,)
    cov    : (STATE_DIM, STATE_DIM)
    u      : (CTRL_DIM,)
    params : NeuralCogTransitionParams
    Q      : (STATE_DIM, STATE_DIM) -- dt-scaled process noise

    Returns
    -------
    (mean_pred, cov_pred)
    """
    pts      = sigma_points(mean, cov, _LAM)
    pts_next = jax.vmap(_integrate_step, in_axes=(0, None, None))(pts, u, params)
    return unscented_transform(pts_next, Q, _WM, _WC)


@jax.jit
def _ukf_update_nc(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: NeuralCogObsParams,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF measurement update ([RPE_Proxy, PVT_Lapses]).

    Parameters
    ----------
    mean_pred  : (STATE_DIM,)
    cov_pred   : (STATE_DIM, STATE_DIM)
    y_obs      : (OBS_DIM,) = (2,)
    obs_params : NeuralCogObsParams
    R          : (OBS_DIM, OBS_DIM) -- per-channel inflated noise

    Returns
    -------
    (posterior_mean, posterior_cov)
    """
    pts    = sigma_points(mean_pred, cov_pred, _LAM)
    y_pts  = h_nc_sigma(pts, obs_params)

    y_mean = jnp.einsum("i,ij->j", _WM, y_pts)
    dy     = y_pts - y_mean[None, :]
    dx     = pts   - mean_pred[None, :]

    S_yy   = jnp.einsum("i,ij,ik->jk", _WC, dy, dy) + R
    P_xy   = jnp.einsum("i,ij,ik->jk", _WC, dx, dy)

    K          = P_xy @ jnp.linalg.inv(S_yy)
    innovation = y_obs - y_mean
    post_mean  = mean_pred + K @ innovation
    post_cov   = cov_pred  - K @ S_yy @ K.T

    return post_mean, post_cov


# ── Public filter class ───────────────────────────────────────────────────────

class NeuralCognitiveStateFilter:
    """
    L4 TUKF for the 7-state Neural/Cognitive system.

    Sparse assimilation: RPE (post-session) + PVT_Lapses (morning test).
    Most steps are predict-only (quality_flag = 4).
    Dynamic dt_hours: intra-session (<=1 h) to between-session (<=24 h).

    TUKF blindage (Simon 2010):
        sigma_points : eigh-based (robust to float32 non-PSD drift).
        lower_clamp_moments for: 5HT, DA, Aden, Caf.
        range_clamp_moments for: NH3 [0,1], O2Sat [0,1], CAR [0,1].
        variance_floor + nearest_psd after every update.
        Q scaled by dt_hours (Wiener scaling).

    Fail-Loud contract
    ------------------
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
            logger.info("NeuralCognitiveStateFilter -- dynamax backend registered.")
        else:
            self._dyn_ukf = None
            logger.warning(
                "NeuralCognitiveStateFilter -- dynamax not installed; "
                "single-step only."
            )

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
        prior         : GaussianState(mean in R^7, cov in R^7x7)
        controls      : dict with keys:
                        'hub_training_stress'    [0-1]      aerobic intensity
                        'hub_muscle_damage'      [0-1]      structural damage
                        'hub_T_core'             [degC]     core temperature
                        'hub_sleep_debt'         [au]       sleep deficit
                        'hub_IL6'                [au]       systemic IL-6
                        'hub_metabolic_stress'   [0-1]      metabolic acidosis
                        'caffeine_intake_plasma' [mg/L/h]   caffeine absorption
                        'hub_hypoglycemia'       [0-1]      glucose deficit
        dt_hours      : float -- integration window [h]
        rpe_proxy     : float [1-10]; NaN = unavailable
        pvt_lapses    : float [count/10 min]; NaN = unavailable
        quality_flags : (flag_RPE, flag_PVT); 4 = predict-only
        params        : NeuralCogTransitionParams (defaults to population)

        Returns
        -------
        GaussianState -- posterior (mean, cov)

        Raises
        ------
        RuntimeError on NaN posterior or non-PSD covariance.
        """
        if params is None:
            params = NeuralCogTransitionParams(nc=DEFAULT_NC_PARAMS, dt_hours=dt_hours)
        else:
            params = params._replace(dt_hours=dt_hours)

        u = jnp.array([
            float(controls.get("hub_training_stress",    0.0)),
            float(controls.get("hub_muscle_damage",      0.0)),
            float(controls.get("hub_T_core",             37.0)),
            float(controls.get("hub_sleep_debt",         0.0)),
            float(controls.get("hub_IL6",                0.0)),
            float(controls.get("hub_metabolic_stress",   0.0)),
            float(controls.get("caffeine_intake_plasma", 0.0)),
            float(controls.get("hub_hypoglycemia",       0.0)),
        ], dtype=jnp.float32)

        # ── dt-scaled Q (Wiener scaling) ──────────────────────────────────────
        Q_step = scale_Q(self.Q, dt_hours)

        # ── Predict ───────────────────────────────────────────────────────────
        mean_pred, cov_pred = _ukf_predict_nc(prior.mean, prior.cov, u, params, Q_step)

        # ── Inflate R for unavailable channels ────────────────────────────────
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
        post_mean, post_cov = _ukf_update_nc(
            mean_pred, cov_pred, y_obs, self.obs_params, R_step
        )

        # ── TUKF physical clamps (Simon 2010): moment matching + floor + PSD ──
        post_mean, post_cov = _apply_physical_clamps_nc(post_mean, post_cov)

        # ── Fail-Loud: divergence detection ───────────────────────────────────
        if bool(jnp.any(jnp.isnan(post_mean))):
            raise RuntimeError(
                "NeuralCognitiveStateFilter.update_state: posterior_mean contains NaN. "
                f"rpe_proxy={rpe_proxy}, pvt_lapses={pvt_lapses}, "
                f"quality_flags={quality_flags}, dt_hours={dt_hours}."
            )
        diag = jnp.diag(post_cov)
        if bool(jnp.any(diag < jnp.float32(-1e-6))):
            raise RuntimeError(
                "NeuralCognitiveStateFilter.update_state: posterior_cov has negative "
                "diagonal -- filter diverged. Increase Q or R."
            )

        return GaussianState(mean=post_mean, cov=post_cov)

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
