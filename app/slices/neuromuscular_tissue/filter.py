"""
app/slices/neuromuscular_tissue/filter.py

L4 State Filter V4.0 -- Neuromuscular Tissue Slice
Unscented Kalman Filter for the 6-state intra-session system (minutes timescale).

State x in R^6:
    [ATP_Muscle_mmol, Calcium_Cytosolic_uM,
     Recruitment_Type1, Recruitment_Type2, RyR1_Damage_au,
     Muscle_Glycogen_mmol]

Observations y in R^2:
    [EMG_Amplitude_mV, SmO2_pct]

UKF parametrisation (van der Merwe & Wan 2000)
-----------------------------------------------
    alpha = 0.10  MANDATORY for float32 stability with n=6.
                  W0_mean = lambda/(n+lambda) = 1 - 1/alpha^2 = -99.0
                  INDEPENDENT of n -- safe for float32.
                  Wi_mean = 0.5 / (n+lambda) = 0.5 / 0.06 = 8.333...
                  With alpha=0.001: n+lambda=6e-6 -> Wi~83333 -> catastrophic.
    beta  = 2.0   Gaussian kurtosis prior
    kappa = 0.0
    n     = 6  ->  2n+1 = 13 sigma points
    lambda = alpha^2*(n+kappa) - n = 0.01*6 - 6 = -5.94
    n+lambda = 0.06

Numerical stabilisation (mandatory)
--------------------------------------
    Jitter 1e-3:   before each Cholesky: chol((n+lambda)*P + 1e-3*I)
    Clamp post-predict: physical bounds applied to all sigma points after ODE
    Floor diagonal 1e-3: after update: P_ii = max(P_ii, 1e-3)

Transition: 1-minute Tsit5 ODE integration.
Sigma-point propagation via jax.vmap over 13 ODE solves.

Fail-Loud contract
------------------
    NaN in posterior_mean   -> RuntimeError("NMv4Filter: divergence")
    Negative diagonal cov   -> RuntimeError("NMv4Filter: negative variance")

References
----------
    van der Merwe & Wan (2000) Proc. ASSPCC
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import diffrax

from app.slices.neuromuscular_tissue.ode import (
    NMv4Params,
    DEFAULT_V4_PARAMS,
    X0_NM_V4,
    P0_NM_V4,
    STATE_DIM,
    CTRL_DIM,
    IDX_ATP, IDX_CA, IDX_R1, IDX_R2, IDX_RYR1, IDX_GLYCOGEN,
    nm_v4_ode,
)
from app.slices.neuromuscular_tissue.observation import (
    NMv4ObsParams,
    DEFAULT_V4_OBS_PARAMS,
    OBS_DIM,
    h_nm_v4,
    inflate_R_nm_v4,
    R_NM_V4_DEFAULT,
)
from app.engine.assimilation.ukf_filter import GaussianState

logger = logging.getLogger(__name__)

# -- UKF constants (alpha=0.10, n=6) -------------------------------------------

_ALPHA  = 0.10
_BETA   = 2.0
_KAPPA  = 0.0
_N      = STATE_DIM   # 6

_LAMBDA   = _ALPHA ** 2 * (_N + _KAPPA) - _N   # = 0.06 - 6 = -5.94
_N_LAMBDA = _N + _LAMBDA                         # = 0.06
_JITTER   = 1e-3                                 # added to (n+lambda)*P before Cholesky

_W0_MEAN: float = _LAMBDA / _N_LAMBDA                          # = -99.0  (same as n=5 case)
_WI_MEAN: float = 0.5 / _N_LAMBDA                              # = 8.333...
_W0_COV:  float = _W0_MEAN + (1.0 - _ALPHA ** 2 + _BETA)      # = -96.01 (same as n=5 case)
_WI_COV:  float = _WI_MEAN                                     # = 8.333...

_WM: jax.Array = jnp.array(
    [_W0_MEAN] + [_WI_MEAN] * (2 * _N), dtype=jnp.float32
)   # shape (13,)

_WC: jax.Array = jnp.array(
    [_W0_COV] + [_WI_COV] * (2 * _N), dtype=jnp.float32
)   # shape (13,)

# -- Process noise (per 1-minute step) -----------------------------------------

_Q_DIAG: jax.Array = jnp.array([
    0.010,   # ATP      (mmol/kg)^2 -- driven by deterministic exercise biochemistry
    0.010,   # Ca       (uM)^2      -- SERCA dynamics well-characterised
    0.001,   # R1       -- recruitment nearly deterministic given power input
    0.001,   # R2       -- idem
    0.001,   # RyR1     -- slow; damage nearly deterministic per Ca excess
    1.000,   # Glycogen (mmol/kg)^2 -- more uncertainty per minute (meal timing, etc.)
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


# -- Physical clamp (JIT/vmap safe) -------------------------------------------

def _clamp_physical_v4(x: jax.Array) -> jax.Array:
    """Clamp V4.0 state to physiological bounds. Used post-predict on sigma points."""
    return jnp.stack([
        jnp.maximum(x[IDX_ATP],      jnp.float32(0.0)),            # ATP >= 0
        jnp.maximum(x[IDX_CA],       jnp.float32(0.0)),            # Ca >= 0
        jnp.clip(x[IDX_R1], jnp.float32(0.0), jnp.float32(1.0)),  # R1 in [0,1]
        jnp.clip(x[IDX_R2], jnp.float32(0.0), jnp.float32(1.0)),  # R2 in [0,1]
        jnp.maximum(x[IDX_RYR1],     jnp.float32(0.0)),            # RyR1 >= 0
        jnp.maximum(x[IDX_GLYCOGEN], jnp.float32(0.0)),            # Glycogen >= 0
    ])


# -- ODE transition (1 minute, Tsit5) ------------------------------------------

def _integrate_1min(
    x:      jax.Array,
    u:      jax.Array,
    params: NMv4Params,
) -> jax.Array:
    """
    Advance one sigma point by 1 minute via Tsit5.

    vmap-safe: only x varies across sigma points.
    """
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(nm_v4_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.float32(1.0),
        dt0       = jnp.float32(0.1),
        y0        = x,
        args      = (params, u),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 32,
    )
    return _clamp_physical_v4(sol.ys[0])


# -- UKF predict step ----------------------------------------------------------

def _ukf_predict_v4(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: NMv4Params,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF predict step: unscented transform through the 1-minute ODE.

    Jitter: chol(_N_LAMBDA * P + _JITTER * I) for numerical stability.
    Clamp: each propagated sigma point is clamped to physical bounds.

    Parameters
    ----------
    mean   : (STATE_DIM,) = (6,)
    cov    : (STATE_DIM, STATE_DIM) = (6, 6)
    u      : (CTRL_DIM,)
    params : NMv4Params
    Q      : (STATE_DIM, STATE_DIM) = (6, 6)

    Returns
    -------
    (mean_pred, cov_pred)
    """
    # Cholesky with jitter for numerical safety
    P_scaled = jnp.float32(_N_LAMBDA) * cov \
             + jnp.float32(_JITTER) * jnp.eye(_N, dtype=jnp.float32)
    L        = jnp.linalg.cholesky(P_scaled)   # (N, N) lower triangular

    # 2n+1 sigma points: x0, x0+L.T[i], x0-L.T[i]
    sp_pos  = mean[None, :] + L.T   # (N, N) = (6, 6)
    sp_neg  = mean[None, :] - L.T   # (N, N) = (6, 6)
    sigmas  = jnp.concatenate([mean[None, :], sp_pos, sp_neg], axis=0)  # (13, 6)

    # Propagate all 13 sigma points through ODE (vmap)
    sigmas_f = jax.vmap(lambda x_i: _integrate_1min(x_i, u, params))(sigmas)  # (13, 6)

    # Weighted mean and covariance
    mean_pred = jnp.einsum("i,is->s", _WM, sigmas_f)
    diff      = sigmas_f - mean_pred[None, :]
    P_pred    = jnp.einsum("i,is,it->st", _WC, diff, diff) + Q
    P_pred    = jnp.float32(0.5) * (P_pred + P_pred.T)

    return mean_pred, P_pred


# -- UKF update step -----------------------------------------------------------

def _ukf_update_v4(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: NMv4ObsParams,
    nm_params:  NMv4Params,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF measurement update: [EMG_mV, SmO2_pct].

    Floor diagonal at 1e-3 post-update for variance stability.
    Physical clamp applied to posterior mean.

    Parameters
    ----------
    mean_pred  : (STATE_DIM,) = (6,)
    cov_pred   : (STATE_DIM, STATE_DIM) = (6, 6)
    y_obs      : (OBS_DIM,) = (2,)
    obs_params : NMv4ObsParams
    nm_params  : NMv4Params
    R          : (OBS_DIM, OBS_DIM) = (2, 2)

    Returns
    -------
    (posterior_mean, posterior_cov)
    """
    # Sigma points from predicted distribution
    P_scaled = jnp.float32(_N_LAMBDA) * cov_pred \
             + jnp.float32(_JITTER) * jnp.eye(_N, dtype=jnp.float32)
    L        = jnp.linalg.cholesky(P_scaled)
    sp_pos   = mean_pred[None, :] + L.T
    sp_neg   = mean_pred[None, :] - L.T
    sigmas   = jnp.concatenate([mean_pred[None, :], sp_pos, sp_neg], axis=0)   # (13, 6)

    # Observation predictions for each sigma point
    y_sigma = jax.vmap(lambda x_i: h_nm_v4(x_i, obs_params, nm_params))(sigmas)  # (13, 2)

    y_mean = jnp.einsum("i,io->o", _WM, y_sigma)
    dy_s   = y_sigma - y_mean[None, :]
    dx_s   = sigmas  - mean_pred[None, :]

    S_yy = jnp.einsum("i,io,ij->oj", _WC, dy_s, dy_s) + R   # (OBS_DIM, OBS_DIM)
    P_xy = jnp.einsum("i,is,io->so", _WC, dx_s, dy_s)       # (STATE_DIM, OBS_DIM)

    K         = P_xy @ jnp.linalg.inv(S_yy)
    innov     = y_obs - y_mean
    post_mean = mean_pred + K @ innov
    post_cov  = cov_pred  - K @ S_yy @ K.T
    post_cov  = jnp.float32(0.5) * (post_cov + post_cov.T)

    # Floor diagonal at 1e-3 (prevents variance collapse)
    idx      = jnp.arange(_N)
    floored  = jnp.maximum(jnp.diag(post_cov), jnp.float32(1e-3))
    post_cov = post_cov.at[idx, idx].set(floored)

    # Physical clamp on posterior mean
    post_mean = _clamp_physical_v4(post_mean)

    return post_mean, post_cov


# -- Public filter class -------------------------------------------------------

class NMv4Filter:
    """
    L4 Unscented Kalman Filter for the Neuromuscular Tissue V4.0 intra-session state.

    Assimilates [EMG_Amplitude_mV, SmO2_pct] at 1-minute intervals to infer the
    6 hidden physiological states [ATP, Ca, R1, R2, RyR1_Damage, Muscle_Glycogen].

    UKF parameters: alpha=0.10 (MANDATORY), n=6 -> 13 sigma points.
    Jitter=1e-3, clamp post-predict, floor diagonal 1e-3 post-update.

    Fail-Loud contract
    ------------------
    RuntimeError if posterior_mean contains NaN (filter divergence).
    RuntimeError if posterior_cov has negative diagonal (non-PSD).
    """

    def __init__(
        self,
        Q:          jax.Array | None     = None,
        R:          jax.Array | None     = None,
        obs_params: NMv4ObsParams | None = None,
        nm_params:  NMv4Params   | None  = None,
    ) -> None:
        self.Q          = Q          if Q          is not None else Q_DEFAULT
        self.R          = R          if R          is not None else R_NM_V4_DEFAULT
        self.obs_params = obs_params if obs_params is not None else DEFAULT_V4_OBS_PARAMS
        self.nm_params  = nm_params  if nm_params  is not None else DEFAULT_V4_PARAMS

    def update_state(
        self,
        prior:         GaussianState,
        emg_mV:        float,
        smo2_pct:      float,
        u:             jax.Array,
        quality_flags: tuple[int, int] = (0, 0),
    ) -> GaussianState:
        """
        Full UKF cycle: 1-min Tsit5 ODE predict -> 2-channel observation update.

        Parameters
        ----------
        prior         : GaussianState(mean=(6,), cov=(6,6))
        emg_mV        : float [mV] -- EMG amplitude; NaN -> missing (inflate R)
        smo2_pct      : float [%]  -- SmO2; NaN -> missing (inflate R)
        u             : shape (CTRL_DIM,) -- [hub_Power_W, hub_Lactate, hub_Glucose]
        quality_flags : (flag_EMG, flag_SmO2) in [0,4]; 4 -> R*1e8 (predict-only)

        Returns
        -------
        GaussianState -- updated posterior (6-state)

        Raises
        ------
        RuntimeError on NaN posterior mean.
        RuntimeError on negative diagonal covariance.
        """
        # -- Predict step ------------------------------------------------------
        mean_pred, cov_pred = _ukf_predict_v4(
            prior.mean, prior.cov, u, self.nm_params, self.Q
        )

        # -- NaN channels -> inflate R to 1e8 (predict-only) ------------------
        flags = list(quality_flags)
        if emg_mV   != emg_mV:    flags[0] = 4
        if smo2_pct != smo2_pct:  flags[1] = 4

        R_step = inflate_R_nm_v4((flags[0], flags[1]), self.R)

        # Substitute NaN observations with predicted values (no information added)
        y_hat = h_nm_v4(mean_pred, self.obs_params, self.nm_params)
        y_obs = jnp.array([
            y_hat[0] if flags[0] == 4 else float(emg_mV),
            y_hat[1] if flags[1] == 4 else float(smo2_pct),
        ], dtype=jnp.float32)

        # -- Update step -------------------------------------------------------
        post_mean, post_cov = _ukf_update_v4(
            mean_pred, cov_pred, y_obs, self.obs_params, self.nm_params, R_step
        )

        # -- Fail-Loud checks --------------------------------------------------
        if bool(jnp.any(jnp.isnan(post_mean))):
            raise RuntimeError(
                "NMv4Filter.update_state: NaN in posterior mean. "
                f"EMG={emg_mV}, SmO2={smo2_pct}, flags={quality_flags}. "
                "Inspect Q/R matrices and ODE stability."
            )

        diag = jnp.diag(post_cov)
        if bool(jnp.any(diag < jnp.float32(-1e-6))):
            raise RuntimeError(
                "NMv4Filter.update_state: negative diagonal in posterior cov. "
                f"diag(P)={diag}. Increase Q or R."
            )

        return GaussianState(mean=post_mean, cov=post_cov)

    def filter_session(
        self,
        emg_sequence:  jax.Array,
        smo2_sequence: jax.Array,
        u_sequence:    jax.Array,
        x0:            jax.Array | None = None,
        P0:            jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Run UKF for a full intra-session sequence.

        Parameters
        ----------
        emg_sequence  : (T,) float32 -- EMG [mV] per minute
        smo2_sequence : (T,) float32 -- SmO2 [%] per minute
        u_sequence    : (T, CTRL_DIM) float32 -- control inputs per minute
        x0, P0        : initial state/covariance (defaults to X0_NM_V4, P0_NM_V4)

        Returns
        -------
        means : (T, STATE_DIM) = (T, 6)
        covs  : (T, STATE_DIM, STATE_DIM) = (T, 6, 6)
        """
        state = GaussianState(
            mean = x0 if x0 is not None else X0_NM_V4,
            cov  = P0 if P0 is not None else P0_NM_V4,
        )
        T = emg_sequence.shape[0]
        means, covs = [], []

        for step in range(T):
            state = self.update_state(
                prior         = state,
                emg_mV        = float(emg_sequence[step]),
                smo2_pct      = float(smo2_sequence[step]),
                u             = u_sequence[step],
                quality_flags = (0, 0),
            )
            means.append(state.mean)
            covs.append(state.cov)

        return jnp.stack(means), jnp.stack(covs)
