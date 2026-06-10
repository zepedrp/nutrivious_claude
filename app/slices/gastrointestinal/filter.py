"""
app/slices/gastrointestinal/filter.py  --  GI Slice V4.0 (TUKF Gold Standard)

L4 TUKF  --  6-state, dt_minutes=1.0, alpha=0.10.

n=6, alpha=0.10, beta=2, kappa=0:
  lambda = 0.01*6 - 6 = -5.94
  n+lambda = 0.06
  WM_0 = -5.94/0.06 = -99.0
  WC_0 = -99 + (1-0.01+2) = -96.01
  WI   = 0.5/0.06 = 8.3333

TUKF Blindage (Simon 2010):
  sigma_points  : eigh-based (PSD-robust), shared from ukf_filter.py.
  Post-predict  : Q scaled by dt_minutes (Wiener scaling).
  Post-update   : _apply_physical_clamps_gi:
                    lower_clamp_moments(lb=0.0) for all 5 mass/volume dims.
                    range_clamp_moments([0,1]) for GI_Distress.
                    variance_floor + nearest_psd.
  Fail-Loud     : RuntimeError on NaN posterior_mean or non-PSD diagonal.
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
    UnscentedKalmanFilter = None  # type: ignore[assignment,misc]
    ParamsNLGSSM          = None  # type: ignore[assignment,misc]
    _DYNAMAX_OK = False

from app.slices.gastrointestinal.ode import (
    GIv3Params, DEFAULT_GI_PARAMS,
    X0_GI_DEFAULT, P0_GI_DEFAULT,
    STATE_DIM, CTRL_DIM,
    gi_ode,
    IDX_STOM_FLUID, IDX_STOM_GLU, IDX_STOM_FRU,
    IDX_INTST_GLU, IDX_INTST_FRU, IDX_DISTRESS,
)
from app.slices.gastrointestinal.observation import (
    GIObsParams, DEFAULT_GI_OBS_PARAMS,
    OBS_DIM, h_gi, h_gi_sigma,
    R_GI_DEFAULT, inflate_R_gi,
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

# ── UKF Merwe-Wan weights (alpha=0.10, n=6) ───────────────────────────────────
# float32-safe: Wm[0] = -99.0, no catastrophic cancellation.
# See ukf_filter.py module docstring for the full float32 safety analysis.

_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM   # 6

_WM, _WC, _LAM = ukf_weights(_N, _ALPHA, _BETA, _KAPPA)

# ── Simon 2010 variance floor (Section 5.4) ───────────────────────────────────

_VAR_FLOOR: jax.Array = jnp.array([
    1e-6,   # Stomach_Fluid_L   (0.001 L)^2
    1e-4,   # Stom_Glu_g        (0.01 g)^2
    1e-4,   # Stom_Fru_g        (0.01 g)^2
    1e-4,   # Intst_Glu_g       (0.01 g)^2
    1e-4,   # Intst_Fru_g       (0.01 g)^2
    1e-6,   # GI_Distress_au    (0.001)^2
], dtype=jnp.float32)

# ── Process noise Q (diagonal, per minute) ────────────────────────────────────

_Q_DIAG: jax.Array = jnp.array([
    2.5e-3,   # Stomach_Fluid_L  (0.05 L/min)^2
    0.09,     # Stom_Glu_g       (0.3 g/min)^2
    0.09,     # Stom_Fru_g       (0.3 g/min)^2
    0.04,     # Intst_Glu_g      (0.2 g/min)^2
    0.04,     # Intst_Fru_g      (0.2 g/min)^2
    1e-4,     # GI_Distress_au   (0.01)^2
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


class GITransitionParams(NamedTuple):
    gi:   GIv3Params = DEFAULT_GI_PARAMS
    dt_m: float      = 1.0   # [min]


DEFAULT_TRANSITION_PARAMS: GITransitionParams = GITransitionParams()


# ── JIT-safe 1-min ODE step (vmap-compatible) ─────────────────────────────────

def _integrate_1min(
    x:      jax.Array,
    u:      jax.Array,
    params: GITransitionParams,
) -> jax.Array:
    sol = diffrax.diffeqsolve(
        terms    = diffrax.ODETerm(gi_ode),
        solver   = diffrax.Tsit5(),
        t0       = jnp.float32(0.0),
        t1       = jnp.asarray(params.dt_m, dtype=jnp.float32),
        dt0      = jnp.float32(0.1),
        y0       = x,
        args     = (params.gi, u),
        saveat   = diffrax.SaveAt(t1=True),
        max_steps= 256,
    )
    return sol.ys[0]


# ── Physical clamps (Simon 2010 truncated-normal) ─────────────────────────────

def _apply_physical_clamps_gi(
    mean: jax.Array,
    cov:  jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Gaussian-coherent physical clamping for all 6 GI states.

    Step 1 -- truncated-normal moment matching per constrained dimension:
        Stom_Fluid, Stom_Glu, Stom_Fru, Intst_Glu, Intst_Fru  >= 0
        GI_Distress                                              in [0, 1]

    Step 2 -- variance floor (Simon 2010 Section 5.4).
    Step 3 -- nearest_psd repair (Higham 1988).
    """
    for dim in [IDX_STOM_FLUID, IDX_STOM_GLU, IDX_STOM_FRU,
                IDX_INTST_GLU,  IDX_INTST_FRU]:
        m, v = lower_clamp_moments(mean[dim], cov[dim, dim], 0.0)
        mean, cov = clamp_dim(mean, cov, dim, m, v)

    m, v = range_clamp_moments(mean[IDX_DISTRESS], cov[IDX_DISTRESS, IDX_DISTRESS], 0.0, 1.0)
    mean, cov = clamp_dim(mean, cov, IDX_DISTRESS, m, v)

    cov = variance_floor(cov, _VAR_FLOOR)
    cov = nearest_psd(cov)
    return mean, cov


# ── Pure-JAX UKF kernels ───────────────────────────────────────────────────────

@jax.jit
def _ukf_predict_gi(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: GITransitionParams,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF predict: eigh-based sigma points through the 1-min GI ODE.

    Q must be pre-scaled to dt_real (via scale_Q) before calling.

    Parameters
    ----------
    mean   : (STATE_DIM,)
    cov    : (STATE_DIM, STATE_DIM)
    u      : (CTRL_DIM,)
    params : GITransitionParams
    Q      : (STATE_DIM, STATE_DIM) — dt-scaled process noise

    Returns
    -------
    (mean_pred, cov_pred)
    """
    pts      = sigma_points(mean, cov, _LAM)
    pts_next = jax.vmap(_integrate_1min, in_axes=(0, None, None))(pts, u, params)
    return unscented_transform(pts_next, Q, _WM, _WC)


@jax.jit
def _ukf_update_gi(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: GIObsParams,
    gi_params:  GIv3Params,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF measurement update for GI [Nausea_VAS, Bloating_VAS].

    Parameters
    ----------
    mean_pred  : (STATE_DIM,)
    cov_pred   : (STATE_DIM, STATE_DIM)
    y_obs      : (OBS_DIM,) = (2,)
    obs_params : GIObsParams
    gi_params  : GIv3Params
    R          : (OBS_DIM, OBS_DIM) — per-channel inflated noise

    Returns
    -------
    (posterior_mean, posterior_cov)
    """
    pts    = sigma_points(mean_pred, cov_pred, _LAM)
    y_pts  = h_gi_sigma(pts, obs_params, gi_params)

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


# ── Public filter ──────────────────────────────────────────────────────────────

class GastrointestinalStateFilter:
    """
    L4 TUKF for GI V4.0 -- 6-state, alpha=0.10, dt_minutes=1.0.

    Observations: [Nausea_VAS, Bloating_VAS] (VAS 0-10).

    TUKF blindage (Simon 2010):
        sigma_points : eigh-based (robust to float32 non-PSD drift).
        lower_clamp_moments(lb=0.0) for all 5 mass/volume dimensions.
        range_clamp_moments([0,1]) for GI_Distress_au.
        variance_floor + nearest_psd after every update.
        Q scaled by dt_minutes (Wiener scaling).
    """

    def __init__(
        self,
        Q:          jax.Array | None = None,
        R:          jax.Array | None = None,
        obs_params: GIObsParams | None = None,
    ) -> None:
        self.Q          = Q          if Q          is not None else Q_DEFAULT
        self.R          = R          if R          is not None else R_GI_DEFAULT
        self.obs_params = obs_params if obs_params is not None else DEFAULT_GI_OBS_PARAMS

        if _DYNAMAX_OK:
            self._dyn_ukf = UnscentedKalmanFilter(STATE_DIM, OBS_DIM)
        else:
            self._dyn_ukf = None

    def update_state(
        self,
        prior:         GaussianState,
        controls:      dict[str, float],
        dt_minutes:    float = 1.0,
        nausea_obs:    float = float("nan"),
        bloating_obs:  float = float("nan"),
        quality_flags: tuple[int, int] = (4, 4),
        params:        GITransitionParams | None = None,
    ) -> GaussianState:
        """
        One TUKF step (predict + update + physical clamps).

        Parameters
        ----------
        prior         : GaussianState(mean in R^6, cov in R^6x6)
        controls      : dict with keys:
                        'Fluid_in'      [L/min]   fluid ingestion rate
                        'Glu_in'        [g/min]   glucose intake
                        'Fru_in'        [g/min]   fructose intake
                        'Power'         [W]       mechanical output
                        'Temp'          [degC]    core temperature
                        'Sodium_mmolL'  [mmol/L]  plasma sodium (default 140)
        quality_flags : (flag_nausea, flag_bloating); 4 = predict-only channel.

        Returns
        -------
        GaussianState -- posterior (mean, cov)

        Raises
        ------
        RuntimeError on NaN posterior or non-PSD covariance.
        """
        if params is None:
            params = GITransitionParams(gi=DEFAULT_GI_PARAMS, dt_m=float(dt_minutes))
        else:
            params = params._replace(dt_m=float(dt_minutes))

        u = jnp.array([
            float(controls.get("Fluid_in",      0.0)),
            float(controls.get("Glu_in",         0.0)),
            float(controls.get("Fru_in",         0.0)),
            float(controls.get("Power",           0.0)),
            float(controls.get("Temp",           37.0)),
            float(controls.get("Sodium_mmolL",  140.0)),
        ], dtype=jnp.float32)

        # ── dt-scaled Q (Wiener scaling) ──────────────────────────────────────
        Q_step = scale_Q(self.Q, dt_minutes)

        # ── Predict ───────────────────────────────────────────────────────────
        mean_p, cov_p = _ukf_predict_gi(prior.mean, prior.cov, u, params, Q_step)

        # ── Per-channel R inflation ───────────────────────────────────────────
        fn, fb  = quality_flags
        _S      = {0: 1.0, 1: 2.0, 2: 5.0, 3: 20.0, 4: 1e8}
        scale_n = _S.get(fn, 1e8)
        scale_b = _S.get(fb, 1e8)
        R_step  = jnp.array(
            [[self.R[0, 0] * scale_n, 0.0], [0.0, self.R[1, 1] * scale_b]],
            dtype=jnp.float32,
        )

        # fill missing obs with predicted observation (1e8 R -> zero Kalman gain)
        y_hat = h_gi(mean_p, self.obs_params, params.gi)
        nv    = y_hat[0] if (nausea_obs  != nausea_obs  or fn == 4) else jnp.float32(nausea_obs)
        bv    = y_hat[1] if (bloating_obs != bloating_obs or fb == 4) else jnp.float32(bloating_obs)
        y_obs = jnp.array([nv, bv], dtype=jnp.float32)

        # ── Update ────────────────────────────────────────────────────────────
        post_mean, post_cov = _ukf_update_gi(
            mean_p, cov_p, y_obs, self.obs_params, params.gi, R_step
        )

        # ── TUKF physical clamps (Simon 2010): moment matching + floor + PSD ──
        post_mean, post_cov = _apply_physical_clamps_gi(post_mean, post_cov)

        # ── Fail-Loud: divergence detection ───────────────────────────────────
        if bool(jnp.any(jnp.isnan(post_mean))):
            raise RuntimeError(
                "GastrointestinalStateFilter.update_state: posterior_mean NaN. "
                f"flags={quality_flags}, dt={dt_minutes} min."
            )
        diag = jnp.diag(post_cov)
        if bool(jnp.any(diag < jnp.float32(-1e-5))):
            raise RuntimeError(
                "GastrointestinalStateFilter.update_state: posterior_cov non-PSD. "
                f"diag={diag}"
            )

        return GaussianState(mean=post_mean, cov=post_cov)

    def build_dynamax_params(
        self,
        tp:           GITransitionParams,
        initial_mean: jax.Array | None = None,
        initial_cov:  jax.Array | None = None,
    ) -> "ParamsNLGSSM":
        if not _DYNAMAX_OK:
            raise RuntimeError("dynamax required. pip install dynamax")
        _op = self.obs_params

        def _dyn(x, u):
            return _integrate_1min(x, u, tp)

        def _emit(x, _):
            return h_gi(x, _op, tp.gi)

        return ParamsNLGSSM(
            initial_mean       = initial_mean if initial_mean is not None else X0_GI_DEFAULT,
            initial_covariance = initial_cov  if initial_cov  is not None else P0_GI_DEFAULT,
            dynamics_function  = _dyn,
            dynamics_covariance= self.Q,
            emission_function  = _emit,
            emission_covariance= self.R,
        )

    def filter_time_series(
        self,
        emissions:    jax.Array,
        inputs:       jax.Array,
        tp:           GITransitionParams,
        initial_mean: jax.Array | None = None,
        initial_cov:  jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if not _DYNAMAX_OK:
            raise RuntimeError("dynamax required for filter_time_series.")
        dyn_p = self.build_dynamax_params(tp, initial_mean, initial_cov)
        post  = self._dyn_ukf.filter(dyn_p, emissions, inputs)
        return post.filtered_means, post.filtered_covariances
