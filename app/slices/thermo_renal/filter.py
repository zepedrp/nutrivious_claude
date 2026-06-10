"""
app/slices/thermo_renal/filter.py  —  L4 TUKF, Thermo-Renal V3.0

5-state Truncated Unscented Kalman Filter over the thermo-renal state:
  [Core_Temp_C, Skin_Temp_C, Plasma_Volume_L, Inters_Volume_L, Plasma_Sodium_mmol]

Uses the canonical UKF primitives from app.engine.assimilation.ukf_filter:
    sigma_points        — eigh-based, PSD-robust (replaces jnp.linalg.cholesky)
    unscented_transform — weighted mean + covariance from propagated sigma pts
    nearest_psd         — Higham (1988) PSD repair after update
    variance_floor      — Simon (2010) §5.4 minimum variance enforcement
    range_clamp_moments — truncated-normal [lb, ub] for physical hard bounds
    lower_clamp_moments — truncated-normal lb for non-negative constraints
    clamp_dim           — applies per-dimension truncated correction to (mean, cov)

UKF parametrisation (Merwe & Wan 2000):
  alpha = 0.10  (MANDATORY — float32 stability; alpha=1e-3 underflows with n=5)
  beta  = 2.0
  kappa = 0.0
  n = 5  →  lambda = 0.01×5 − 5 = −4.95,  n+lambda = 0.05
  W_m_0 = −99,  W_c_0 = −96.01,  W_i = 10  (sum W_m = 1 ✓)

Time step: dt_minutes (default 1.0 min).
Integrator: diffrax.Tsit5, dt0=0.1 min, max_steps=512.

Physical clamp bounds (Simon 2010 — range_clamp_moments):
  Core_Temp_C  : [35.0, 42.0]   °C  — hypothermia floor / hyperthermia ceiling
  Skin_Temp_C  : [15.0, 42.0]   °C  — physical skin temperature range
  Plasma_Volume_L   : lb=0.5    L   — physiological minimum
  Inters_Volume_L   : lb=0.5    L   — physiological minimum
  Plasma_Sodium_mmol: lb=0.0    mmol — non-negative

Fail-Loud contract:
  NaN in posterior_mean → RuntimeError.
  Negative covariance diagonal → RuntimeError.
  Covariance symmetrised and nearest_psd-repaired each step.

References
  Merwe R., Wan E. (2000) Proc. ASSPCC
  Simon D. (2010) Optimal State Estimation, Wiley §5.3-5.4
  Higham N.J. (1988) Linear Algebra Appl. 103:103-118
  Fiala D. et al. (1999) J Appl Physiol 87(5):1957-1972  [2-node thermo]
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
from app.engine.assimilation.ukf_filter import (
    GaussianState,
    sigma_points,
    unscented_transform,
    nearest_psd,
    variance_floor,
    ukf_weights,
    range_clamp_moments,
    lower_clamp_moments,
    clamp_dim,
)

logger = logging.getLogger(__name__)

# ── UKF weights (n = STATE_DIM = 5, alpha = 0.10) ─────────────────────────────

_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM   # 5

_WM, _WC, _LAM = ukf_weights(_N, _ALPHA, _BETA, _KAPPA)

# Variance floor per state dimension (prevents filter overconfidence)
_VAR_FLOOR_TR: jax.Array = jnp.array([
    1e-4,    # Core_Temp_C         (σ_min ≈ 0.01 °C)
    4e-4,    # Skin_Temp_C         (σ_min ≈ 0.02 °C)
    1e-4,    # Plasma_Volume_L     (σ_min ≈ 0.01 L)
    4e-4,    # Inters_Volume_L     (σ_min ≈ 0.02 L)
    1.0,     # Plasma_Sodium_mmol  (σ_min ≈ 1 mmol)
], dtype=jnp.float32)

# Physical bounds for range_clamp_moments (Simon 2010 eq. 5.31)
_CORE_TEMP_LB:   float = 35.0    # °C  — severe hypothermia floor
_CORE_TEMP_UB:   float = 42.0    # °C  — life-threatening hyperthermia ceiling
_SKIN_TEMP_LB:   float = 15.0    # °C  — physical lower bound
_SKIN_TEMP_UB:   float = 42.0    # °C  — same upper bound as core
_PV_LB:          float = 0.5     # L   — physiological minimum plasma volume
_IV_LB:          float = 0.5     # L   — physiological minimum interstitial volume
_NA_LB:          float = 0.0     # mmol — non-negative sodium


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
    return jnp.maximum(x_next, jnp.float32(0.0))   # coarse clamp before TUKF


# ── Physical clamp: truncated-normal moment matching (Simon 2010) ─────────────

def _apply_physical_clamps(
    mean: jax.Array,
    cov:  jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Apply per-dimension truncated-normal clamps to (mean, cov).

    Dimension layout:
      0: Core_Temp_C      → range_clamp [35, 42]
      1: Skin_Temp_C      → range_clamp [15, 42]
      2: Plasma_Volume_L  → lower_clamp lb=0.5
      3: Inters_Volume_L  → lower_clamp lb=0.5
      4: Plasma_Sodium_mmol → lower_clamp lb=0
    """
    # Core_Temp_C: physiologically bounded [35°C, 42°C]
    mu_new, s2_new = range_clamp_moments(mean[0], cov[0, 0], _CORE_TEMP_LB, _CORE_TEMP_UB)
    mean, cov = clamp_dim(mean, cov, 0, mu_new, s2_new)

    # Skin_Temp_C: physically bounded [15°C, 42°C]
    mu_new, s2_new = range_clamp_moments(mean[1], cov[1, 1], _SKIN_TEMP_LB, _SKIN_TEMP_UB)
    mean, cov = clamp_dim(mean, cov, 1, mu_new, s2_new)

    # Plasma_Volume_L: must be ≥ 0.5 L
    mu_new, s2_new = lower_clamp_moments(mean[2], cov[2, 2], _PV_LB)
    mean, cov = clamp_dim(mean, cov, 2, mu_new, s2_new)

    # Inters_Volume_L: must be ≥ 0.5 L
    mu_new, s2_new = lower_clamp_moments(mean[3], cov[3, 3], _IV_LB)
    mean, cov = clamp_dim(mean, cov, 3, mu_new, s2_new)

    # Plasma_Sodium_mmol: must be ≥ 0
    mu_new, s2_new = lower_clamp_moments(mean[4], cov[4, 4], _NA_LB)
    mean, cov = clamp_dim(mean, cov, 4, mu_new, s2_new)

    return mean, cov


# ── Pure-JAX UKF kernels (eigh-based) ─────────────────────────────────────────

@jax.jit
def _ukf_predict(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: TRTransitionParams,
) -> tuple[jax.Array, jax.Array]:
    """UKF predict: propagate sigma points through the TR ODE (eigh-based)."""
    Q_step = Q_PER_MIN * jnp.float32(params.dt_minutes)

    # eigh-based sigma points (PSD-robust; no Cholesky failure)
    spts      = sigma_points(mean, cov, _LAM)                              # (11, 5)
    spts_next = jax.vmap(_integrate_step, in_axes=(0, None, None))(spts, u, params)  # (11, 5)

    mu_pred, P_pred = unscented_transform(spts_next, Q_step, _WM, _WC)
    P_pred = nearest_psd(P_pred)
    return mu_pred, P_pred


@jax.jit
def _ukf_update(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: TRObsParams,
    tr_params:  ThermoRenalParams,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """UKF update: assimilate [Core_Temp_obs, Body_Mass_Drop_kg] (eigh-based)."""
    spts   = sigma_points(mean_pred, cov_pred, _LAM)               # (11, 5)
    y_sigma = h_tr_sigma(spts, obs_params, tr_params)               # (11, 2)

    y_mean, S = unscented_transform(y_sigma, R, _WM, _WC)          # (2,), (2, 2)

    dx_s = spts   - mean_pred[None, :]    # (11, 5)
    dy_s = y_sigma - y_mean[None, :]      # (11, 2)
    P_xy = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)             # (5, 2)

    K              = P_xy @ jnp.linalg.inv(S)
    innovation     = y_obs - y_mean
    posterior_mean = mean_pred + K @ innovation
    posterior_cov  = cov_pred  - K @ S @ K.T

    P_new = nearest_psd(posterior_cov)
    return posterior_mean, P_new


# ── Public filter class ───────────────────────────────────────────────────────

class ThermoRenalStateFilter:
    """
    L4 TUKF for the 5-state Thermo-Renal V2 slice (V3.0 filter — canonical primitives).

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
        One-step TUKF predict + update with physical clamps.

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
        GaussianState — posterior (mean, cov) with physical bounds enforced
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

        # Update (eigh-based, nearest_psd inside)
        posterior_mean, posterior_cov = _ukf_update(
            mean_pred, cov_pred, y_obs_arr, self.obs_params, params.tr, R_step
        )

        # Truncated-normal physical clamps (Simon 2010 §5.3)
        posterior_mean, posterior_cov = _apply_physical_clamps(posterior_mean, posterior_cov)

        # Variance floor (Simon 2010 §5.4)
        posterior_cov = variance_floor(posterior_cov, _VAR_FLOOR_TR)

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

        return GaussianState(mean=posterior_mean, cov=posterior_cov)
