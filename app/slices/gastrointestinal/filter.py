"""
app/slices/gastrointestinal/filter.py  --  GI Slice V3.0

L4 UKF  --  6-state, dt_minutes=1.0, alpha=0.10 MANDATORY.

n=6, alpha=0.10, beta=2, kappa=0:
  lambda = 0.01*6 - 6 = -5.94
  n+lambda = 0.06
  WM_0 = -5.94/0.06 = -99.0
  WC_0 = -99 + (1-0.01+2) = -96.01
  WI   = 0.5/0.06 = 8.3333

Post-predict  : clamp jnp.maximum(x, 0) + jitter _JITTER*I.
Post-update   : floor diagonal at _COV_FLOOR.
Fail-Loud     : RuntimeError on NaN posterior_mean or negative cov diagonal.
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
)
from app.slices.gastrointestinal.observation import (
    GIObsParams, DEFAULT_GI_OBS_PARAMS,
    OBS_DIM, h_gi, h_gi_sigma,
    R_GI_DEFAULT, inflate_R_gi,
)
from app.engine.assimilation.ukf_filter import GaussianState

logger = logging.getLogger(__name__)

# ── UKF weights (alpha=0.10, n=6) ────────────────────────────────────────────
_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM   # 6

_LAM:   float = _ALPHA**2 * (_N + _KAPPA) - _N   # -5.94
_NL:    float = _N + _LAM                          #  0.06

_WM_0:  float = _LAM / _NL           # -99.0
_WC_0:  float = _WM_0 + (1.0 - _ALPHA**2 + _BETA)  # -96.01
_WI:    float = 0.5 / _NL            #  8.3333

_WM: jax.Array = jnp.array([_WM_0] + [_WI] * (2 * _N), dtype=jnp.float32)
_WC: jax.Array = jnp.array([_WC_0] + [_WI] * (2 * _N), dtype=jnp.float32)

_JITTER:    jax.Array = jnp.float32(1e-3)
_COV_FLOOR: jax.Array = jnp.float32(1e-3)

# ── Process noise Q (diagonal, per minute) ───────────────────────────────────
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


# ── ODE step (vmap-compatible) ────────────────────────────────────────────────

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
    return jnp.maximum(sol.ys[0], jnp.float32(0.0))


# ── sigma-point helpers ───────────────────────────────────────────────────────

def _sigma_points(mean: jax.Array, cov: jax.Array) -> jax.Array:
    """Merwe-Wan 2n+1 sigma points.  Returns (2n+1, n)."""
    n = mean.shape[0]
    # add small regularisation before Cholesky
    cov_reg = jnp.float32(_NL) * (cov + jnp.float32(1e-7) * jnp.eye(n, dtype=jnp.float32))
    L = jnp.linalg.cholesky(cov_reg)
    return jnp.concatenate(
        [mean[None, :], mean[None, :] + L.T, mean[None, :] - L.T], axis=0
    )


def _recover_mean_cov(
    pts:       jax.Array,
    noise_cov: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean = jnp.einsum("i,ij->j", _WM, pts)
    diff = pts - mean[None, :]
    cov  = jnp.einsum("i,ij,ik->jk", _WC, diff, diff) + noise_cov
    return mean, cov


@jax.jit
def _ukf_predict(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: GITransitionParams,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    sigma      = _sigma_points(mean, cov)
    sigma_next = jax.vmap(_integrate_1min, in_axes=(0, None, None))(sigma, u, params)
    mean_p, cov_p = _recover_mean_cov(sigma_next, Q)
    cov_p = cov_p + _JITTER * jnp.eye(STATE_DIM, dtype=jnp.float32)
    return mean_p, cov_p


@jax.jit
def _ukf_update(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: GIObsParams,
    gi_params:  GIv3Params,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    sigma  = _sigma_points(mean_pred, cov_pred)
    y_sig  = h_gi_sigma(sigma, obs_params, gi_params)

    y_mean = jnp.einsum("i,ij->j", _WM, y_sig)
    dy_s   = y_sig  - y_mean[None, :]
    dx_s   = sigma  - mean_pred[None, :]

    S_yy = jnp.einsum("i,ij,ik->jk", _WC, dy_s, dy_s) + R
    P_xy = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)

    K          = P_xy @ jnp.linalg.inv(S_yy)
    innovation = y_obs - y_mean
    post_mean  = mean_pred + K @ innovation
    post_cov   = cov_pred - K @ S_yy @ K.T

    # floor diagonal at _COV_FLOOR
    diag_vals = jnp.diag(post_cov)
    deficit   = jnp.maximum(_COV_FLOOR - diag_vals, jnp.float32(0.0))
    post_cov  = post_cov + jnp.diag(deficit)

    return post_mean, post_cov


# ── Public filter ─────────────────────────────────────────────────────────────

class GastrointestinalStateFilter:
    """
    L4 UKF for GI V3.0 -- 6-state, alpha=0.10, dt_minutes=1.0.

    Observations: [Nausea_VAS, Bloating_VAS]  (VAS 0-10).
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
        One UKF step (predict + update).

        quality_flags: (flag_nausea, flag_bloating); 4 -> predict-only channel.
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

        # ── predict ───────────────────────────────────────────────────────────
        mean_p, cov_p = _ukf_predict(prior.mean, prior.cov, u, params, self.Q)

        # ── per-channel R inflation ───────────────────────────────────────────
        fn, fb  = quality_flags
        _S      = {0: 1.0, 1: 2.0, 2: 5.0, 3: 20.0, 4: 1e8}
        scale_n = _S.get(fn, 1e8)
        scale_b = _S.get(fb, 1e8)
        R_step  = jnp.array(
            [[self.R[0, 0] * scale_n, 0.0], [0.0, self.R[1, 1] * scale_b]],
            dtype=jnp.float32,
        )

        # fill missing obs with predicted value (predict-only semantics)
        y_hat = h_gi(mean_p, self.obs_params, params.gi)
        nv    = y_hat[0] if (nausea_obs  != nausea_obs  or fn == 4) else jnp.float32(nausea_obs)
        bv    = y_hat[1] if (bloating_obs != bloating_obs or fb == 4) else jnp.float32(bloating_obs)
        y_obs = jnp.array([nv, bv], dtype=jnp.float32)

        # ── update ────────────────────────────────────────────────────────────
        post_mean, post_cov = _ukf_update(
            mean_p, cov_p, y_obs, self.obs_params, params.gi, R_step
        )

        # ── Fail-Loud ─────────────────────────────────────────────────────────
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

        post_cov = jnp.float32(0.5) * (post_cov + post_cov.T)
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
