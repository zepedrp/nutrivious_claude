"""
app/slices/thermo_renal/filter.py  —  L4 UKF, Thermo-Renal V2

5-state Unscented Kalman Filter over the thermo-renal state:
  [Core_Temp_C, Skin_Temp_C, Plasma_Volume_L, Inters_Volume_L, Plasma_Sodium_mmol]

UKF parametrisation (Merwe & Wan 2000):
  alpha = 0.10  (MANDATORY — float32 stability; alpha=1e-3 underflows with n=5)
  beta  = 2.0
  kappa = 0.0
  n = 5  →  lambda = 0.01×5 − 5 = −4.95,  n+lambda = 0.05
  W_m_0 = −99,  W_c_0 = −96.01,  W_i = 10  (sum W_m = 1 ✓)

Time step: dt_minutes (default 1.0 min).
Integrator: diffrax.Tsit5, dt0=0.1 min, max_steps=512.

State clamp: jnp.maximum(x, 0) applied after each ODE step (volumes, sodium).

Fail-Loud contract:
  NaN in posterior_mean → RuntimeError.
  Negative covariance diagonal → RuntimeError.
  Covariance symmetrised each step.

References
  Merwe R., Wan E. (2000) Proc. ASSPCC
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

from app.slices.thermo_renal.ode import (
    ThermoRenalParams,
    DEFAULT_TR_PARAMS,
    X0_TR_DEFAULT,
    P0_TR_DEFAULT,
    STATE_DIM,
    CTRL_DIM,
    thermo_renal_ode,
)
from app.slices.thermo_renal.observation import (
    TRObsParams,
    DEFAULT_TR_OBS_PARAMS,
    OBS_DIM,
    R_TR_DEFAULT,
    h_tr,
    h_tr_sigma,
    inflate_R_tr,
)
from app.engine.assimilation.ukf_filter import GaussianState

logger = logging.getLogger(__name__)

# ── UKF weights (n = STATE_DIM = 5, alpha = 0.10) ─────────────────────────────
_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM   # 5

_LAM:  float = _ALPHA ** 2 * (_N + _KAPPA) - _N   # = 0.01×5 − 5 = −4.95
_NL:   float = _N + _LAM                           # = 0.05

_WM_0: float = _LAM / _NL                          # = −99.0
_WC_0: float = _WM_0 + (1.0 - _ALPHA ** 2 + _BETA) # = −99 + 2.99 = −96.01
_WI:   float = 0.5 / _NL                           # = 10.0

_WM: jax.Array = jnp.array([_WM_0] + [_WI] * (2 * _N), dtype=jnp.float32)
_WC: jax.Array = jnp.array([_WC_0] + [_WI] * (2 * _N), dtype=jnp.float32)


# ── Process noise Q (per MINUTE) ─────────────────────────────────────────────
_Q_DIAG_PER_MIN: jax.Array = jnp.array([
    3e-4,   # Core_Temp_C  [°C²/min]
    5e-3,   # Skin_Temp_C  [°C²/min]
    5e-5,   # Plasma_Volume_L  [L²/min]
    2e-4,   # Inters_Volume_L  [L²/min]
    0.10,   # Plasma_Sodium_mmol  [mmol²/min]
], dtype=jnp.float32)

Q_PER_MIN: jax.Array = jnp.diag(_Q_DIAG_PER_MIN)


# ── Transition parameters ─────────────────────────────────────────────────────

class TRTransitionParams(NamedTuple):
    tr:         ThermoRenalParams = DEFAULT_TR_PARAMS
    dt_minutes: float             = 1.0


DEFAULT_TR_TRANSITION_PARAMS: TRTransitionParams = TRTransitionParams()


# ── ODE integration (vmap-compatible) ────────────────────────────────────────

def _integrate_step(
    x:      jax.Array,
    u:      jax.Array,
    params: TRTransitionParams,
) -> jax.Array:
    """Integrate TR ODE for params.dt_minutes minutes. Returns x_next clamped ≥ 0."""
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(thermo_renal_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.asarray(params.dt_minutes, dtype=jnp.float32),
        dt0       = jnp.float32(0.1),
        y0        = x,
        args      = (params.tr, u),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 512,
    )
    x_next = sol.ys[0]
    return jnp.maximum(x_next, jnp.float32(0.0))   # clamp all states ≥ 0


# ── Pure-JAX UKF kernels ──────────────────────────────────────────────────────

def _sigma_points(mean: jax.Array, cov: jax.Array) -> jax.Array:
    """Generate 2n+1 sigma points via Cholesky decomposition."""
    n   = mean.shape[0]
    L   = jnp.linalg.cholesky(jnp.float32(_NL) * cov)     # (n, n)
    sp0 = mean[None, :]                                      # (1, n)
    sp_pos = mean[None, :] + L.T                             # (n, n)
    sp_neg = mean[None, :] - L.T                             # (n, n)
    return jnp.concatenate([sp0, sp_pos, sp_neg], axis=0)   # (2n+1, n)


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
    params: TRTransitionParams,
) -> tuple[jax.Array, jax.Array]:
    """UKF predict: propagate sigma points through the TR ODE."""
    Q_step      = Q_PER_MIN * jnp.float32(params.dt_minutes)
    sigma       = _sigma_points(mean, cov)                         # (11, 5)
    sigma_next  = jax.vmap(_integrate_step, in_axes=(0, None, None))(sigma, u, params)
    return _recover_mean_cov(sigma_next, Q_step)


@jax.jit
def _ukf_update(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: TRObsParams,
    tr_params:  ThermoRenalParams,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """UKF update: assimilate [Core_Temp_obs, Body_Mass_Drop_kg]."""
    sigma   = _sigma_points(mean_pred, cov_pred)                   # (11, 5)
    y_sigma = h_tr_sigma(sigma, obs_params, tr_params)             # (11, 2)

    y_mean = jnp.einsum("i,ij->j", _WM, y_sigma)
    dy_s   = y_sigma - y_mean[None, :]
    dx_s   = sigma   - mean_pred[None, :]

    S_yy = jnp.einsum("i,ij,ik->jk", _WC, dy_s, dy_s) + R        # (2, 2)
    P_xy = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)            # (5, 2)

    K              = P_xy @ jnp.linalg.inv(S_yy)
    innovation     = y_obs - y_mean
    posterior_mean = mean_pred + K @ innovation
    posterior_cov  = cov_pred  - K @ S_yy @ K.T

    return posterior_mean, posterior_cov


# ── Public filter class ───────────────────────────────────────────────────────

class ThermoRenalStateFilter:
    """
    L4 UKF for the 5-state Thermo-Renal V2 slice.

    Usage
    -----
    filt  = ThermoRenalStateFilter()
    state = GaussianState(mean=X0_TR_DEFAULT, cov=P0_TR_DEFAULT)

    state = filt.update_state(
        prior         = state,
        core_temp_obs = 37.6,          # °C from pill/patch
        bw_drop_obs   = float("nan"),  # no weigh-in
        controls      = {"hub_power_watts": 300.0, "hub_fluid_intake_L_min": 0.0,
                         "hub_sodium_intake_mmol_min": 0.0},
        dt_minutes    = 1.0,
    )

    Fail-Loud
    ---------
    RuntimeError if posterior_mean contains NaN.
    RuntimeError if covariance diagonal has negative entries.
    """

    def __init__(
        self,
        R:          jax.Array | None = None,
        obs_params: TRObsParams | None = None,
    ) -> None:
        self.R          = R          if R          is not None else R_TR_DEFAULT
        self.obs_params = obs_params if obs_params is not None else DEFAULT_TR_OBS_PARAMS

    def update_state(
        self,
        prior:          GaussianState,
        core_temp_obs:  float,
        bw_drop_obs:    float,
        controls:       dict[str, float],
        dt_minutes:     float = 1.0,
        quality_flags:  tuple[int, int] = (0, 4),
        params:         TRTransitionParams | None = None,
    ) -> GaussianState:
        """
        One-step UKF predict + update.

        Parameters
        ----------
        prior          : GaussianState(mean, cov)
        core_temp_obs  : float [°C] — NaN if pill not active
        bw_drop_obs    : float [kg] — NaN if no weigh-in
        controls       : dict with keys "hub_power_watts", "hub_fluid_intake_L_min",
                         "hub_sodium_intake_mmol_min"
        dt_minutes     : integration window [min]
        quality_flags  : (flag_core_temp, flag_bw_drop) in [0, 4]

        Returns
        -------
        GaussianState — posterior (mean, cov)
        """
        if params is None:
            params = TRTransitionParams(tr=DEFAULT_TR_PARAMS, dt_minutes=float(dt_minutes))
        else:
            params = params._replace(dt_minutes=float(dt_minutes))

        u = jnp.array([
            float(controls.get("hub_power_watts",           0.0)),
            float(controls.get("hub_fluid_intake_L_min",    0.0)),
            float(controls.get("hub_sodium_intake_mmol_min", 0.0)),
        ], dtype=jnp.float32)

        # Predict
        mean_pred, cov_pred = _ukf_predict(prior.mean, prior.cov, u, params)

        # NaN masking → inflate R and substitute predicted value
        ct_nan = core_temp_obs != core_temp_obs
        bw_nan = bw_drop_obs  != bw_drop_obs

        flags = list(quality_flags)
        if ct_nan: flags[0] = 4
        if bw_nan: flags[1] = 4

        R_step    = inflate_R_tr((flags[0], flags[1]), self.R)
        y_pred    = h_tr(mean_pred, self.obs_params, params.tr)
        y_obs_arr = jnp.array([
            y_pred[0] if ct_nan else float(core_temp_obs),
            y_pred[1] if bw_nan else float(bw_drop_obs),
        ], dtype=jnp.float32)

        # Update
        posterior_mean, posterior_cov = _ukf_update(
            mean_pred, cov_pred, y_obs_arr, self.obs_params, params.tr, R_step
        )

        # Fail-Loud checks
        if bool(jnp.any(jnp.isnan(posterior_mean))):
            raise RuntimeError(
                "ThermoRenalStateFilter: posterior_mean NaN — filter diverged. "
                f"core_temp_obs={core_temp_obs}, dt_minutes={dt_minutes}."
            )

        diag = jnp.diag(posterior_cov)
        if bool(jnp.any(diag < jnp.float32(-1e-6))):
            raise RuntimeError(
                "ThermoRenalStateFilter: posterior_cov has negative diagonal. "
                "Increase Q or R, or check ODE stability."
            )

        posterior_cov = jnp.float32(0.5) * (posterior_cov + posterior_cov.T)
        return GaussianState(mean=posterior_mean, cov=posterior_cov)
