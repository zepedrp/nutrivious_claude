"""
app/slices/neuromuscular_tissue/filter.py

L4 State Filter V4.1 -- Neuromuscular Tissue Slice
Unscented Kalman Filter for the 6-state intra-session system (minutes timescale).

Refactored to use canonical UKF primitives from app.engine.assimilation.ukf_filter,
eliminating the local Cholesky-fragile sigma-point path in V4.0. Same gold
standard as the Cardiorespiratory slice.

State x in R^6:
    [ATP_Muscle_mmol, Calcium_Cytosolic_uM,
     Recruitment_Type1, Recruitment_Type2, RyR1_Damage_au,
     Muscle_Glycogen_mmol]

Observations y in R^2:
    [EMG_Amplitude_mV, SmO2_pct]

UKF parametrisation (van der Merwe & Wan 2000)
-----------------------------------------------
    alpha = 0.10  MANDATORY for float32 stability with n=6.
                  Wm[0] = 1 - 1/alpha^2 = -99.0  (n-independent).
    beta  = 2.0   Gaussian kurtosis prior
    kappa = 0.0
    n     = 6  ->  2n+1 = 13 sigma points

Truncated UKF (Simon 2010):
    1. Predict: sigma_points(eigh, PSD-robust) -> vmap ODE -> unscented_transform
    2. Update:  sigma_points(eigh) -> h_nm_v4 -> Kalman gain
    3. Post:    lower_clamp_moments + range_clamp_moments + variance_floor + nearest_psd

Fail-Loud contract
------------------
    NaN in posterior_mean   -> RuntimeError("NMv4Filter: divergence")
    Negative diagonal cov   -> RuntimeError("NMv4Filter: negative variance")

References
----------
    van der Merwe & Wan (2000) Proc. ASSPCC
    Simon D. (2010) Optimal State Estimation, Wiley
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
from app.engine.assimilation.ukf_filter import (
    GaussianState,
    ukf_weights,
    sigma_points,
    unscented_transform,
    nearest_psd,
    variance_floor,
    lower_clamp_moments,
    range_clamp_moments,
    clamp_dim,
)

logger = logging.getLogger(__name__)

# -- UKF constants (alpha=0.10, n=6) -------------------------------------------

_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM   # 6

_WM, _WC, _LAM = ukf_weights(_N, _ALPHA, _BETA, _KAPPA)
# lambda = 0.01*6 - 6 = -5.94; (n+lambda) = 0.06
# Wm[0] = -99.0; Wi = 8.333...

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

# -- Simon 2010 variance floor (post-update minimum variance per dimension) -----

_VAR_FLOOR: jax.Array = jnp.array([
    1.0e-4,   # ATP      (0.01 mmol/kg)^2
    1.0e-4,   # Ca       (0.01 uM)^2
    1.0e-6,   # R1       (0.001)^2
    1.0e-6,   # R2       (0.001)^2
    1.0e-6,   # RyR1     (0.001)^2
    1.0e-2,   # Glycogen (0.1 mmol/kg)^2
], dtype=jnp.float32)


# -- ODE transition (1 minute, Tsit5) ------------------------------------------

def _integrate_1min(
    x:      jax.Array,
    u:      jax.Array,
    params: NMv4Params,
) -> jax.Array:
    """Advance one sigma point by 1 minute via Tsit5. vmap-safe."""
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
    return sol.ys[0]


# -- Physical clamp (vmap safe) ------------------------------------------------

def _clamp_physical_v4(x: jax.Array) -> jax.Array:
    """Clamp V4 state to physiological bounds. Used on sigma points post-ODE."""
    return jnp.stack([
        jnp.maximum(x[IDX_ATP],      jnp.float32(0.0)),
        jnp.maximum(x[IDX_CA],       jnp.float32(0.0)),
        jnp.clip(x[IDX_R1], jnp.float32(0.0), jnp.float32(1.0)),
        jnp.clip(x[IDX_R2], jnp.float32(0.0), jnp.float32(1.0)),
        jnp.maximum(x[IDX_RYR1],     jnp.float32(0.0)),
        jnp.maximum(x[IDX_GLYCOGEN], jnp.float32(0.0)),
    ])


# -- Truncated-normal physical clamps (Simon 2010) ----------------------------

def _apply_physical_clamps(
    mean: jax.Array,
    cov:  jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Gaussian-coherent physical clamping (Simon 2010 truncated-normal).

    Step 1 -- truncated-normal moment matching per constrained dimension:
        ATP          >= 0  mmol/kg
        Ca           >= 0  uM
        R1           in [0, 1]
        R2           in [0, 1]
        RyR1_Damage  >= 0
        Glycogen     >= 0  mmol/kg

    Step 2 -- variance floor (Simon 2010 Section 5.4).
    Step 3 -- nearest_psd repair (Higham 1988).
    """
    m, v = lower_clamp_moments(mean[IDX_ATP],      cov[IDX_ATP,      IDX_ATP],      0.0)
    mean, cov = clamp_dim(mean, cov, IDX_ATP, m, v)

    m, v = lower_clamp_moments(mean[IDX_CA],       cov[IDX_CA,       IDX_CA],       0.0)
    mean, cov = clamp_dim(mean, cov, IDX_CA, m, v)

    m, v = range_clamp_moments(mean[IDX_R1],       cov[IDX_R1,       IDX_R1],       0.0, 1.0)
    mean, cov = clamp_dim(mean, cov, IDX_R1, m, v)

    m, v = range_clamp_moments(mean[IDX_R2],       cov[IDX_R2,       IDX_R2],       0.0, 1.0)
    mean, cov = clamp_dim(mean, cov, IDX_R2, m, v)

    m, v = lower_clamp_moments(mean[IDX_RYR1],     cov[IDX_RYR1,     IDX_RYR1],     0.0)
    mean, cov = clamp_dim(mean, cov, IDX_RYR1, m, v)

    m, v = lower_clamp_moments(mean[IDX_GLYCOGEN], cov[IDX_GLYCOGEN, IDX_GLYCOGEN], 0.0)
    mean, cov = clamp_dim(mean, cov, IDX_GLYCOGEN, m, v)

    cov = variance_floor(cov, _VAR_FLOOR)
    cov = nearest_psd(cov)

    return mean, cov


# -- UKF predict step ----------------------------------------------------------

@jax.jit
def _ukf_predict_nm_v4(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: NMv4Params,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF predict step: eigh-based sigma points through the 1-minute ODE.

    Uses shared sigma_points() (PSD-robust eigh) instead of the old Cholesky
    path. Physical clamp applied to each propagated sigma point post-ODE.

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
    pts        = sigma_points(mean, cov, _LAM)                         # (13, 6)
    pts_next   = jax.vmap(
        _integrate_1min, in_axes=(0, None, None)
    )(pts, u, params)                                                   # (13, 6)
    pts_next   = jax.vmap(_clamp_physical_v4)(pts_next)                # clamp sigma pts
    mean_pred, cov_pred = unscented_transform(pts_next, Q, _WM, _WC)
    return mean_pred, cov_pred


# -- UKF update step -----------------------------------------------------------

@jax.jit
def _ukf_update_nm_v4(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: NMv4ObsParams,
    nm_params:  NMv4Params,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF measurement update: [EMG_mV, SmO2_pct].

    Uses eigh-based sigma_points() for the update step.

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
    pts    = sigma_points(mean_pred, cov_pred, _LAM)                   # (13, 6)
    y_pts  = jax.vmap(
        lambda x_i: h_nm_v4(x_i, obs_params, nm_params)
    )(pts)                                                              # (13, 2)

    y_mean = jnp.einsum("i,ij->j", _WM, y_pts)                        # (2,)
    dy     = y_pts - y_mean[None, :]                                   # (13, 2)
    dx     = pts   - mean_pred[None, :]                                # (13, 6)

    S_yy   = jnp.einsum("i,ij,ik->jk", _WC, dy, dy) + R              # (2, 2)
    P_xy   = jnp.einsum("i,ij,ik->jk", _WC, dx, dy)                  # (6, 2)

    K          = P_xy @ jnp.linalg.inv(S_yy)                          # (6, 2)
    innov      = y_obs - y_mean                                        # (2,)
    post_mean  = mean_pred + K @ innov
    post_cov   = cov_pred  - K @ S_yy @ K.T

    return post_mean, post_cov


# -- Public filter class -------------------------------------------------------

class NMv4Filter:
    """
    L4 Unscented Kalman Filter for the Neuromuscular Tissue V4.0 intra-session state.

    Assimilates [EMG_Amplitude_mV, SmO2_pct] at 1-minute intervals to infer the
    6 hidden physiological states [ATP, Ca, R1, R2, RyR1_Damage, Muscle_Glycogen].

    UKF V4.1: uses canonical eigh-based sigma_points(), nearest_psd(),
    variance_floor(), and truncated-normal moment matching (Simon 2010).
    Same gold standard as the Cardiorespiratory filter.

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
        Full UKF cycle (Truncated UKF, Simon 2010):
            1. Predict: eigh sigma_points -> 1-min Tsit5 ODE -> unscented_transform
            2. Update:  eigh sigma_points -> h_nm_v4 -> Kalman gain
            3. Clamp:   truncated-normal moment matching + variance_floor + nearest_psd

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
        # -- Predict step (eigh sigma points) ----------------------------------
        mean_pred, cov_pred = _ukf_predict_nm_v4(
            prior.mean, prior.cov, u, self.nm_params, self.Q
        )

        # -- NaN channels -> inflate R to 1e8 (predict-only) ------------------
        flags = list(quality_flags)
        if emg_mV   != emg_mV:    flags[0] = 4
        if smo2_pct != smo2_pct:  flags[1] = 4

        R_step = inflate_R_nm_v4((flags[0], flags[1]), self.R)

        # Substitute NaN observations with predicted values
        y_hat = h_nm_v4(mean_pred, self.obs_params, self.nm_params)
        y_obs = jnp.array([
            y_hat[0] if flags[0] == 4 else float(emg_mV),
            y_hat[1] if flags[1] == 4 else float(smo2_pct),
        ], dtype=jnp.float32)

        # -- Update step -------------------------------------------------------
        post_mean, post_cov = _ukf_update_nm_v4(
            mean_pred, cov_pred, y_obs, self.obs_params, self.nm_params, R_step
        )

        # -- Truncated UKF clamps (Simon 2010) ---------------------------------
        post_mean, post_cov = _apply_physical_clamps(post_mean, post_cov)

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
