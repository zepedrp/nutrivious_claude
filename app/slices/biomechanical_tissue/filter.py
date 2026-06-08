"""
app/slices/biomechanical_tissue/filter.py

L4 State Filter -- Biomechanical Tissue Slice
Unscented Kalman Filter for the 5-state hourly biomechanical system.

Architecture
------------
State x in R^5 (hourly timescale):
    [Tendon_Microdamage, Collagen_Synthesis_Rate, Tendon_Stiffness,
     Bone_Microdamage, Bone_Density]

Observations y in R^3:
    [Pain_VAS [0-10], Ultrasound_Echogenicity [au], DEXA_ZScore_proxy [au]]

Transition f(x, u) -- 1-hour ODE advance
------------------------------------------
Integration window: dt = 1 hour
Integrator: diffrax.Tsit5()
Control u = (hub_load, hub_IGF1, hub_T, hub_Cortisol, hub_Estrogen, hub_Nutrition)

UKF parametrisation (Merwe & Wan 2000) -- MANDATORY alpha = 0.10
------------------------------------------------------------------
    alpha = 0.10, beta = 2, kappa = 0
    n = 5 -> 2n+1 = 11 sigma points
    lambda = alpha^2 * (n + kappa) - n = 0.01 * 5 - 5 = -4.95

State positivity clamp: jnp.maximum(x, 0) applied after predict step.
Covariance jitter: 1e-3 * I added to predicted covariance to guarantee PSD
    before Cholesky decomposition in sigma-point generation.

Quality-flag-aware R inflation (asynchronous / sparse assimilation)
--------------------------------------------------------------------
Flags per channel in [0, 4]:
  0 -- excellent (direct measurement)
  1 -- good      (indirect proxy)
  2 -- moderate  (estimated)
  3 -- poor      (imputed / carry-forward)
  4 -- missing   -> R * 1e8 (predict-only for that channel)

DEXA_Z is sparse (measured ~twice per year); flag_DEXA defaults to 4 on
all non-scan hours. Pain_VAS and Echo are daily; sub-daily hours default
to flag 1-2 (carry-forward between daily app check-ins is expected).

Fail-Loud contract
------------------
  * NaN in posterior_mean -> RuntimeError (filter divergence).
  * Negative diagonal covariance entries -> RuntimeError (non-PSD).
  * All exceptions propagate unmodified.

References
----------
  Merwe & Wan (2000) Proc. ASSPCC
  Arampatzis A. et al. (2007) J Biomech 40:2704    [tendon damage]
  Shultz S.J. et al. (2012) JOSPT 42:640           [E2 laxity]
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

from app.slices.biomechanical_tissue.ode import (
    BiomechanicalParams,
    DEFAULT_BIO_PARAMS,
    X0_BIO_DEFAULT,
    P0_BIO_DEFAULT,
    STATE_DIM,
    CTRL_DIM,
    biomechanical_ode,
    hub_tendon_rupture_risk,
    hub_bone_stress_fracture_risk,
)
from app.slices.biomechanical_tissue.observation import (
    BioObsParams,
    DEFAULT_BIO_OBS_PARAMS,
    OBS_DIM,
    h_bio,
    h_bio_sigma,
    R_BIO_DEFAULT,
    inflate_R_bio,
)
from app.engine.assimilation.ukf_filter import GaussianState

logger = logging.getLogger(__name__)


# -- UKF Merwe-Wan sigma-point parameters (n = STATE_DIM = 5) -----------------
# MANDATORY: alpha = 0.10

_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM   # 5

_LAM:   float = _ALPHA**2 * (_N + _KAPPA) - _N    # 0.01 * 5 - 5 = -4.95
_WM_0:  float = _LAM / (_N + _LAM)                # -4.95 / 0.05 = -99.0
_WC_0:  float = _WM_0 + (1.0 - _ALPHA**2 + _BETA) # -99.0 + 2.99 = -96.01
_WI:    float = 0.5 / (_N + _LAM)                 # 0.5 / 0.05 = 10.0

_WM: jax.Array = jnp.array([_WM_0] + [_WI] * (2 * _N), dtype=jnp.float32)
_WC: jax.Array = jnp.array([_WC_0] + [_WI] * (2 * _N), dtype=jnp.float32)

# Jitter added to predicted covariance before Cholesky -- guarantees PSD
_JITTER: float = 1e-3


# -- Process noise Q (diagonal, dt = 1 hour) ----------------------------------
# Calibrated to expected hourly variability of each state under unmodelled
# disturbances (hydration, temperature, soft-tissue microvibration, etc.)

_Q_DIAG: jax.Array = jnp.array([
    0.0004,   # Tendon_Microdamage    -- small per-hour unmodelled load
    0.0001,   # Collagen_Synthesis_Rate -- slow endocrine-driven changes
    0.0004,   # Tendon_Stiffness      -- hourly viscoelastic variability
    0.0001,   # Bone_Microdamage      -- very slow per-hour
    1e-8,     # Bone_Density          -- BMD negligible per hour
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


# -- Transition parameters container ------------------------------------------

class BioTransitionParams(NamedTuple):
    """Full parameter set for the 1-hour biomechanical ODE transition step."""
    bio:       BiomechanicalParams = DEFAULT_BIO_PARAMS
    dt_hours:  float               = 1.0    # integration window [hours]


DEFAULT_TRANSITION_PARAMS: BioTransitionParams = BioTransitionParams()


# -- JIT-safe 1-hour ODE step --------------------------------------------------

def _integrate_1hour(
    x:      jax.Array,
    u:      jax.Array,
    params: BioTransitionParams,
) -> jax.Array:
    """
    Integrate the biomechanical ODE for params.dt_hours hours.

    vmap-compatible over x (sigma points); u and params are broadcast.

    Parameters
    ----------
    x      : (STATE_DIM,)
    u      : (CTRL_DIM,) -- [hub_load, hub_T, hub_Cortisol, hub_Estrogen]
    params : BioTransitionParams

    Returns
    -------
    x_next : (STATE_DIM,) with non-negative clamp applied
    """
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(biomechanical_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.asarray(params.dt_hours, dtype=jnp.float32),
        dt0       = jnp.float32(0.1),
        y0        = x,
        args      = (params.bio, u),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 64,
    )
    x_next = sol.ys[0]
    # State positivity clamp: physical states cannot go negative
    return jnp.maximum(x_next, jnp.float32(0.0))


# -- Pure-JAX UKF kernels ------------------------------------------------------

def _sigma_points(mean: jax.Array, cov: jax.Array) -> jax.Array:
    """
    Merwe-Wan 2n+1 sigma points.
    Jitter of _JITTER * I applied to cov before Cholesky to guarantee PSD.
    x_0 = mu; x_i = mu + L.T[i-1]; x_{n+i} = mu - L.T[i-1]
    L = chol((n + lambda) * (Sigma + jitter*I)).
    Returns (2n+1, STATE_DIM).
    """
    n          = mean.shape[0]
    cov_jitter = cov + _JITTER * jnp.eye(n, dtype=jnp.float32)
    L          = jnp.linalg.cholesky((n + _LAM) * cov_jitter)
    pos        = mean[None, :] + L.T
    neg        = mean[None, :] - L.T
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
    params: BioTransitionParams,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF predict step: unscented transform through the 1-hour ODE.

    Propagates 2n+1 = 11 sigma points via vmap over _integrate_1hour.
    State positivity clamp applied inside _integrate_1hour.

    Parameters
    ----------
    mean   : (STATE_DIM,)
    cov    : (STATE_DIM, STATE_DIM)
    u      : (CTRL_DIM,)
    params : BioTransitionParams
    Q      : (STATE_DIM, STATE_DIM)

    Returns
    -------
    (mean_pred, cov_pred)
    """
    sigma      = _sigma_points(mean, cov)           # (11, STATE_DIM)
    sigma_next = jax.vmap(
        _integrate_1hour, in_axes=(0, None, None)
    )(sigma, u, params)                              # (11, STATE_DIM)
    mean_p, cov_p = _recover_mean_cov(sigma_next, Q, _WM, _WC)
    # Apply clamp to predicted mean (sigma points already clamped individually)
    mean_p = jnp.maximum(mean_p, jnp.float32(0.0))
    return mean_p, cov_p


@jax.jit
def _ukf_update(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: BioObsParams,
    bio_params: BiomechanicalParams,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    UKF measurement update ([Pain_VAS, Echogenicity, DEXA_Z]).

    Parameters
    ----------
    mean_pred  : (STATE_DIM,)
    cov_pred   : (STATE_DIM, STATE_DIM)
    y_obs      : (OBS_DIM,) = (3,)
    obs_params : BioObsParams
    bio_params : BiomechanicalParams
    R          : (OBS_DIM, OBS_DIM) -- potentially inflated

    Returns
    -------
    (posterior_mean, posterior_cov)
    """
    sigma   = _sigma_points(mean_pred, cov_pred)              # (11, STATE_DIM)
    y_sigma = h_bio_sigma(sigma, obs_params, bio_params)      # (11, OBS_DIM)

    y_mean  = jnp.einsum("i,ij->j", _WM, y_sigma)            # (OBS_DIM,)
    dy_s    = y_sigma - y_mean[None, :]                       # (11, OBS_DIM)
    dx_s    = sigma   - mean_pred[None, :]                    # (11, STATE_DIM)

    S_yy    = jnp.einsum("i,ij,ik->jk", _WC, dy_s, dy_s) + R  # (OBS_DIM, OBS_DIM)
    P_xy    = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)      # (STATE_DIM, OBS_DIM)

    K              = P_xy @ jnp.linalg.inv(S_yy)              # (STATE_DIM, OBS_DIM)
    innovation     = y_obs - y_mean
    posterior_mean = mean_pred + K @ innovation
    posterior_cov  = cov_pred  - K @ S_yy @ K.T

    # State positivity clamp on posterior mean
    posterior_mean = jnp.maximum(posterior_mean, jnp.float32(0.0))
    return posterior_mean, posterior_cov


# -- Public filter class -------------------------------------------------------

class BiomechanicalStateFilter:
    """
    L4 Unscented Kalman Filter for the Biomechanical Tissue hourly state.

    Infers 5 hidden biomechanical states (tendon microdamage, collagen
    synthesis, tendon stiffness, bone microdamage, bone density) from hourly
    Pain_VAS and Echogenicity observations and sparse DEXA Z-scores.

    Typical usage (single hourly update)
    -------------------------------------
    filt  = BiomechanicalStateFilter()
    state = GaussianState(mean=X0_BIO_DEFAULT, cov=P0_BIO_DEFAULT)

    state = filt.update_state(
        prior        = state,
        pain_vas     = 2.5,
        echo         = 0.68,
        controls     = {
            "hub_load":       1.2,
            "hub_IGF1":     180.0,
            "hub_T":         18.0,
            "hub_Cortisol": 250.0,
            "hub_Estrogen": 120.0,
            "hub_Nutrition":  1.0,
        },
        quality_flags = (1, 1, 4),   # DEXA=4 -> inflated to 1e8 (no scan today)
    )

    Hub outputs (algebraic, available at any time)
    -----------------------------------------------
    risk_t = filt.compute_hub_risks(state, u)
    # -> {"Hub_Tendon_Rupture_Risk": float, "Hub_Bone_Stress_Fracture_Risk": float}

    Fail-Loud contract
    ------------------
    RuntimeError if posterior_mean contains NaN.
    RuntimeError if posterior_cov diagonal contains negative values.
    """

    def __init__(
        self,
        Q:          jax.Array | None = None,
        R:          jax.Array | None = None,
        obs_params: BioObsParams | None = None,
    ) -> None:
        self.Q          = Q          if Q          is not None else Q_DEFAULT
        self.R          = R          if R          is not None else R_BIO_DEFAULT
        self.obs_params = obs_params if obs_params is not None else DEFAULT_BIO_OBS_PARAMS

    def update_state(
        self,
        prior:         GaussianState,
        pain_vas:      float,
        echo:          float,
        controls:      dict[str, float],
        quality_flags: tuple[int, int, int] = (1, 1, 4),
        dexa_z:        float = float("nan"),
        params:        BioTransitionParams | None = None,
    ) -> GaussianState:
        """
        Assimilate one hour of observations into the biomechanical state estimate.

        Two-step UKF: Predict (1-hour ODE) -> Update ([VAS, Echo, DEXA_Z]).
        DEXA_Z is asynchronous: pass dexa_z=nan or set quality_flags[2]=4
        to perform predict-only for that channel.

        Parameters
        ----------
        prior         : GaussianState(mean, cov)
        pain_vas      : float [0-10] -- Pain VAS self-report; NaN = missing
        echo          : float [au]   -- Echogenicity; NaN = missing
        controls      : dict with keys:
                        "hub_load"      [au],     "hub_IGF1"     [ng/mL],
                        "hub_T"         [nmol/L], "hub_Cortisol" [nmol/L],
                        "hub_Estrogen"  [pg/mL],  "hub_Nutrition" [0-1]
        quality_flags : (flag_VAS, flag_Echo, flag_DEXA) in [0, 4]
        dexa_z        : float -- DEXA Z-score; NaN = no scan this hour
        params        : BioTransitionParams (defaults to population params)

        Returns
        -------
        GaussianState -- updated posterior (mean, cov)

        Raises
        ------
        RuntimeError if posterior_mean contains NaN.
        RuntimeError if posterior_cov has negative diagonal.
        """
        if params is None:
            params = DEFAULT_TRANSITION_PARAMS

        u = jnp.array([
            float(controls.get("hub_load",       0.0)),
            float(controls.get("hub_IGF1",     150.0)),
            float(controls.get("hub_T",         20.0)),
            float(controls.get("hub_Cortisol", 300.0)),
            float(controls.get("hub_Estrogen",  50.0)),
            float(controls.get("hub_Nutrition",  1.0)),
        ], dtype=jnp.float32)

        # -- Predict step -------------------------------------------------------
        mean_pred, cov_pred = _ukf_predict(prior.mean, prior.cov, u, params, self.Q)

        # -- Observation: NaN / missing -> inflate R * 1e8 ----------------------
        vas_nan  = pain_vas != pain_vas
        echo_nan = echo     != echo
        dexa_nan = dexa_z   != dexa_z

        flags = list(quality_flags)
        if vas_nan:  flags[0] = 4
        if echo_nan: flags[1] = 4
        if dexa_nan: flags[2] = 4

        R_step = inflate_R_bio((flags[0], flags[1], flags[2]), self.R)

        y_hat = h_bio(mean_pred, self.obs_params, params.bio)
        y_obs = jnp.array([
            y_hat[0] if vas_nan  else float(pain_vas),
            y_hat[1] if echo_nan else float(echo),
            y_hat[2] if dexa_nan else float(dexa_z),
        ], dtype=jnp.float32)

        # -- Update step --------------------------------------------------------
        posterior_mean, posterior_cov = _ukf_update(
            mean_pred, cov_pred, y_obs, self.obs_params, params.bio, R_step
        )

        # Symmetrise and add floor matching jitter magnitude to prevent float32
        # near-zero cancellation after high-information DEXA updates
        posterior_cov = jnp.float32(0.5) * (posterior_cov + posterior_cov.T)
        posterior_cov = posterior_cov + jnp.float32(_JITTER) * jnp.eye(_N, dtype=jnp.float32)

        # -- Fail-Loud: NaN divergence ------------------------------------------
        if bool(jnp.any(jnp.isnan(posterior_mean))):
            raise RuntimeError(
                "BiomechanicalStateFilter.update_state: posterior_mean contains NaN. "
                f"Pain_VAS={pain_vas}, Echo={echo}, DEXA_Z={dexa_z}, "
                f"quality_flags={quality_flags}. "
                "Verify Q/R matrices and BioTransitionParams."
            )

        # -- Fail-Loud: non-PSD covariance ---------------------------------------
        diag = jnp.diag(posterior_cov)
        if bool(jnp.any(diag < jnp.float32(-1e-6))):
            raise RuntimeError(
                "BiomechanicalStateFilter.update_state: posterior_cov has negative "
                "diagonal -- filter diverged. Increase Q or R."
            )

        return GaussianState(mean=posterior_mean, cov=posterior_cov)

    def compute_hub_risks(
        self,
        state:   GaussianState,
        u:       jax.Array,
        params:  BioTransitionParams | None = None,
    ) -> dict[str, float]:
        """
        Compute algebraic hub risk outputs from the current posterior mean.

        Parameters
        ----------
        state  : GaussianState
        u      : (CTRL_DIM,) -- current hub inputs
        params : BioTransitionParams (defaults to population)

        Returns
        -------
        dict with keys:
          "Hub_Tendon_Rupture_Risk"      : float in [0, 1]
          "Hub_Bone_Stress_Fracture_Risk": float in [0, 1]
        """
        if params is None:
            params = DEFAULT_TRANSITION_PARAMS

        rupture_risk = hub_tendon_rupture_risk(state.mean, u, params.bio)
        fracture_risk = hub_bone_stress_fracture_risk(state.mean, params.bio)

        return {
            "Hub_Tendon_Rupture_Risk":       float(rupture_risk),
            "Hub_Bone_Stress_Fracture_Risk": float(fracture_risk),
        }
