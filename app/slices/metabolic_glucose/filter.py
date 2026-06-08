"""
app/slices/metabolic_glucose/filter.py -- UKF State Filter V2.0

L4 Unscented Kalman Filter for the 6-state Metabolic Glucose slice.

UKF parametrisation (Merwe & Wan 2000)
───────────────────────────────────────
    alpha = 0.10   MANDATORY -- prevents float32 catastrophic cancellation.
                   At alpha=1e-3: (N+lambda) = N*alpha^2 = 6e-6 (near-zero
                   sigma spread, float32 unstable). At alpha=0.10:
                   lambda = 0.01*6 - 6 = -5.94; (N+lambda) = 0.06  STABLE.
    beta  = 2.0    optimal for Gaussian distributions
    kappa = 0.0
    lambda = alpha^2*(N+kappa) - N = -5.94
    (N+lambda) = 0.06  -> sigma spread = sqrt(0.06*P), above float32 floor

State N=6: [G, I, Gc, LG, MG, Lac]  -> 13 sigma points
Observation: y = [G_obs mg/dL]  (CGM, scalar)
Transition: 1-min ODE via diffrax.Tsit5

MANDATORY post-update:
    posterior_mean = jnp.maximum(posterior_mean, 0)
    cov = 0.5*(cov + cov.T) + 1e-3 * I   (symmetrise + jitter for PSD)

Fail-Loud:
    NaN in posterior_mean -> RuntimeError

References:
    Merwe & Wan (2000) Proc. ASSPCC
    Dalla Man et al. (2007) IEEE TBME 54:1740-1749
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import diffrax

from app.engine.assimilation.ukf_filter import GaussianState
from app.slices.metabolic_glucose.ode import (
    metabolic_glucose_ode,
    GlucoseMetabParams,
    DEFAULT_PARAMS,
    HUBS_DEFAULT,
    STATE_DIM,
    IDX_G,
    P0_DEFAULT,
)
from app.slices.metabolic_glucose.observation import (
    h_cgm,
    h_cgm_sigma,
    R_CGM_ONLY,
    inflate_R_for_quality,
)

logger = logging.getLogger(__name__)

# -- UKF weights (alpha=0.10, beta=2.0, kappa=0.0, N=6)
_ALPHA: float = 0.10   # MANDATORY: prevents float32 cancellation at N=6
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM   # 6

_LAM: float = _ALPHA ** 2 * (_N + _KAPPA) - _N   # = 0.01*6 - 6 = -5.94
_NL:  float = _N + _LAM                           # = 0.06

_WM_0: float = _LAM / _NL                         # = -99.0
_WC_0: float = _WM_0 + 1.0 - _ALPHA ** 2 + _BETA  # = -96.01
_WI:   float = 0.5 / _NL                          # = 8.333...

_WM: jax.Array = jnp.array([_WM_0] + [_WI] * (2 * _N), dtype=jnp.float32)
_WC: jax.Array = jnp.array([_WC_0] + [_WI] * (2 * _N), dtype=jnp.float32)

# -- Process noise Q (per 1-min step, dt_minutes=1.0)
_Q_DIAG: jax.Array = jnp.array([
    2.00,    # G   [mg/dL]^2    unmodelled EGP / meal fluctuations
    0.25,    # I   [pmol/L]^2   insulin pulse variability
    1.00,    # Gc  [pg/mL]^2    glucagon pulse variability
    0.010,   # LG  [g]^2        slow hepatic dynamics
    0.010,   # MG  [g]^2        slow muscle glycogen dynamics
    0.001,   # Lac [mmol/L]^2   slow lactate variability
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


def _integrate_1min(
    x:      jax.Array,
    hubs:   jax.Array,
    params: GlucoseMetabParams,
) -> jax.Array:
    """Advance the 6-state glucose ODE by dt_minutes=1.0 minute via Tsit5."""
    sol = diffrax.diffeqsolve(
        terms    = diffrax.ODETerm(metabolic_glucose_ode),
        solver   = diffrax.Tsit5(),
        t0       = jnp.float32(0.0),
        t1       = jnp.float32(1.0),
        dt0      = jnp.float32(0.5),
        y0       = x,
        args     = (params, hubs),
        saveat   = diffrax.SaveAt(t1=True),
        max_steps= 64,
    )
    return sol.ys[0]


def _sigma_points(mean: jax.Array, cov: jax.Array) -> jax.Array:
    """
    Merwe-Wan sigma points: sigma_0 = mu; sigma_i = mu +/- L.T[i-1]
    where L = chol((N+lambda) * Sigma). Returns (2N+1, STATE_DIM).
    """
    L   = jnp.linalg.cholesky(_NL * cov)
    pos = mean[None, :] + L.T
    neg = mean[None, :] - L.T
    return jnp.concatenate([mean[None, :], pos, neg], axis=0)


def _recover_mean_cov(
    pts: jax.Array,
    Q:   jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Recover (mean, cov) from propagated sigma points + process noise Q."""
    mean = jnp.einsum("i,ij->j", _WM, pts)
    diff = pts - mean[None, :]
    cov  = jnp.einsum("i,ij,ik->jk", _WC, diff, diff) + Q
    return mean, cov


class MetabolicGlucoseFilter:
    """
    L4 Unscented Kalman Filter -- 6-state Metabolic Glucose slice.

    Infers plasma glucose, insulin, glucagon, liver/muscle glycogen, and lactate
    from the scalar CGM signal at 1-minute (dt_minutes=1.0) resolution.

    Usage
    ─────
    filt  = MetabolicGlucoseFilter()
    state = GaussianState(mean=X0_DEFAULT, cov=P0_DEFAULT)

    for cgm_reading in cgm_stream:
        state = filt.update_state(state, cgm_reading)

    Fail-Loud:
        RuntimeError if posterior_mean contains NaN (filter divergence).
    """

    def __init__(
        self,
        Q: jax.Array | None = None,
        R: jax.Array | None = None,
    ) -> None:
        self.Q = Q if Q is not None else Q_DEFAULT
        self.R = R if R is not None else R_CGM_ONLY
        logger.info("MetabolicGlucoseFilter (N=%d, alpha=%.2f, dt_minutes=1.0).", _N, _ALPHA)

    def update_state(
        self,
        prior:        GaussianState,
        cgm_reading:  float,
        hubs:         jax.Array | None = None,
        params:       GlucoseMetabParams | None = None,
        quality_flag: int = 0,
    ) -> GaussianState:
        """
        Assimilate one 1-minute CGM reading into the 6-state estimate.

        Steps:
            1. Predict -- propagate prior through 1-min ODE (sigma points)
            2. Update  -- correct with CGM observation (quality-flag-aware R)
            3. Post    -- clamp jnp.maximum(x, 0); symmetrise cov; add 1e-3 jitter

        Parameters
        ----------
        prior        : GaussianState(mean, cov)
        cgm_reading  : float [mg/dL]; NaN -> predict-only step
        hubs         : (HUB_DIM,); defaults to HUBS_DEFAULT
        params       : GlucoseMetabParams; defaults to DEFAULT_PARAMS
        quality_flag : int in [0, 4]; 4 = missing -> R * 1e8

        Returns
        -------
        GaussianState -- updated (mean, cov)

        Raises
        ------
        RuntimeError if posterior_mean contains NaN.
        """
        if hubs is None:
            hubs = HUBS_DEFAULT
        if params is None:
            params = DEFAULT_PARAMS

        # Predict step: propagate prior sigma points through 1-min ODE
        sigma      = _sigma_points(prior.mean, prior.cov)        # (2N+1, N)
        sigma_next = jax.vmap(
            _integrate_1min, in_axes=(0, None, None)
        )(sigma, hubs, params)                                    # (2N+1, N)
        mean_pred, cov_pred = _recover_mean_cov(sigma_next, self.Q)

        # Update step: correct with CGM (quality-flag-aware R inflation)
        R_step = inflate_R_for_quality(quality_flag, self.R)

        cgm_is_nan = cgm_reading != cgm_reading   # NaN check
        if cgm_is_nan:
            posterior_mean = mean_pred
            posterior_cov  = cov_pred
        else:
            y_obs   = jnp.array([float(cgm_reading)], dtype=jnp.float32)
            sigma_u = _sigma_points(mean_pred, cov_pred)
            y_sigma = h_cgm_sigma(sigma_u)                        # (2N+1, 1)

            y_mean = jnp.einsum("i,ij->j", _WM, y_sigma)
            dy_s   = y_sigma - y_mean[None, :]
            dx_s   = sigma_u - mean_pred[None, :]

            S_yy = jnp.einsum("i,ij,ik->jk", _WC, dy_s, dy_s) + R_step
            P_xy = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)

            K              = P_xy @ jnp.linalg.inv(S_yy)
            innovation     = y_obs - y_mean
            posterior_mean = mean_pred + K @ innovation
            posterior_cov  = cov_pred - K @ S_yy @ K.T

        # MANDATORY: clamp concentrations to non-negative
        posterior_mean = jnp.maximum(posterior_mean, jnp.float32(0.0))

        # Symmetrise + 1e-3 jitter on diagonal (guarantees PSD)
        posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
        posterior_cov = posterior_cov + jnp.float32(1e-3) * jnp.eye(
            STATE_DIM, dtype=jnp.float32
        )

        # Fail-Loud: divergence detection
        if bool(jnp.any(jnp.isnan(posterior_mean))):
            raise RuntimeError(
                "MetabolicGlucoseFilter.update_state: posterior_mean contains NaN. "
                f"CGM reading={cgm_reading}, quality_flag={quality_flag}. "
                "Check Q/R matrices and GlucoseMetabParams."
            )

        return GaussianState(mean=posterior_mean, cov=posterior_cov)
