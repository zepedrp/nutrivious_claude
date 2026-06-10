"""
app/slices/metabolic_glucose/filter.py -- UKF State Filter V3.0

L4 Unscented Kalman Filter for the 5-state Metabolic Glucose slice (V3.0).
Refactored to use canonical UKF primitives from app.engine.assimilation.ukf_filter,
eliminating the local Cholesky-fragile sigma-point path.

State N=5: [G, I, Gc, LG, Lac]  -> 2*5+1 = 11 sigma points
Observation: y = [G_obs mg/dL]  (CGM, scalar)
Transition: 1-min ODE via diffrax.Tsit5

UKF parametrisation (Merwe & Wan 2000)
  alpha = 0.10  float32-safe (Wm[0] = 1 - 1/alpha^2 = -99.0, n-independent)
  beta  = 2.0
  kappa = 0.0
  lambda = alpha^2*(N+kappa) - N = 0.01*5 - 5 = -4.95
  (N+lambda) = 0.05 -> sigma spread = sqrt(0.05*P), above float32 floor

Post-update (Truncated UKF, Simon 2010):
    clamp non-negative  (jnp.maximum(x, 0))
    variance_floor      (prevents over-confidence)
    nearest_psd         (Higham 1988 eigh repair)

Fail-Loud:
    NaN in posterior_mean -> RuntimeError

References:
    Merwe & Wan (2000) Proc. ASSPCC
    Simon D. (2010) Optimal State Estimation, Wiley
    Higham N.J. (1988) Linear Algebra Appl. 103:103-118
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import diffrax

from app.engine.assimilation.ukf_filter import (
    GaussianState,
    ukf_weights,
    sigma_points,
    unscented_transform,
    nearest_psd,
    variance_floor,
)
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

# -- UKF weights (alpha=0.10, N=STATE_DIM=5)
_ALPHA: float = 0.10
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM   # 5 after V3.0

_WM, _WC, _LAM = ukf_weights(_N, _ALPHA, _BETA, _KAPPA)
# lambda = -4.95; (N+lambda) = 0.05; Wm[0] = -99.0; Wi = 0.5/0.05 = 10.0

# -- Process noise Q (per 1-min step, dt_minutes=1.0)
_Q_DIAG: jax.Array = jnp.array([
    2.00,    # G   [mg/dL]^2    unmodelled EGP / meal fluctuations
    0.25,    # I   [pmol/L]^2   insulin pulse variability
    1.00,    # Gc  [pg/mL]^2    glucagon pulse variability
    0.010,   # LG  [g]^2        slow hepatic dynamics
    0.001,   # Lac [mmol/L]^2   slow lactate variability
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)

# -- Simon 2010 variance floor (post-update minimum variance per dimension)
_VAR_FLOOR: jax.Array = jnp.array([
    1.0e-2,   # G   (0.1 mg/dL)^2
    1.0e-2,   # I   (0.1 pmol/L)^2
    1.0e-2,   # Gc  (0.1 pg/mL)^2
    1.0e-4,   # LG  (0.01 g)^2
    1.0e-4,   # Lac (0.01 mmol/L)^2
], dtype=jnp.float32)


def _integrate_1min(
    x:      jax.Array,
    hubs:   jax.Array,
    params: GlucoseMetabParams,
) -> jax.Array:
    """Advance the 5-state glucose ODE by dt_minutes=1.0 minute via Tsit5."""
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


class MetabolicGlucoseFilter:
    """
    L4 Unscented Kalman Filter -- 5-state Metabolic Glucose slice V3.0.

    Infers plasma glucose, insulin, glucagon, liver glycogen, and lactate
    from the scalar CGM signal at 1-minute (dt_minutes=1.0) resolution.

    Uses canonical UKF primitives (eigh-based sigma points, nearest_psd,
    variance_floor) from app.engine.assimilation.ukf_filter -- same gold
    standard as the Cardiorespiratory slice.

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
        Assimilate one 1-minute CGM reading into the 5-state estimate.

        Steps (Truncated UKF, Simon 2010):
            1. Predict -- sigma_points(eigh) -> vmap ODE -> unscented_transform
            2. Update  -- sigma_points(eigh) -> h_cgm -> Kalman gain
            3. Post    -- non-negative clamp; variance_floor; nearest_psd

        Parameters
        ----------
        prior        : GaussianState(mean, cov)
        cgm_reading  : float [mg/dL]; NaN -> predict-only step
        hubs         : (HUB_DIM=5,); defaults to HUBS_DEFAULT
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

        # -- Predict step (eigh-based sigma points) ----------------------------
        pts_prior  = sigma_points(prior.mean, prior.cov, _LAM)     # (11, 5)
        pts_next   = jax.vmap(
            _integrate_1min, in_axes=(0, None, None)
        )(pts_prior, hubs, params)                                  # (11, 5)
        mean_pred, cov_pred = unscented_transform(pts_next, self.Q, _WM, _WC)

        # -- Update step -------------------------------------------------------
        R_step     = inflate_R_for_quality(quality_flag, self.R)
        cgm_is_nan = cgm_reading != cgm_reading

        if cgm_is_nan:
            posterior_mean = mean_pred
            posterior_cov  = cov_pred
        else:
            y_obs   = jnp.array([float(cgm_reading)], dtype=jnp.float32)
            pts_upd = sigma_points(mean_pred, cov_pred, _LAM)      # (11, 5)
            y_sigma = h_cgm_sigma(pts_upd)                         # (11, 1)

            y_mean = jnp.einsum("i,ij->j", _WM, y_sigma)
            dy_s   = y_sigma - y_mean[None, :]
            dx_s   = pts_upd - mean_pred[None, :]

            S_yy = jnp.einsum("i,ij,ik->jk", _WC, dy_s, dy_s) + R_step
            P_xy = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)

            K              = P_xy @ jnp.linalg.inv(S_yy)
            innovation     = y_obs - y_mean
            posterior_mean = mean_pred + K @ innovation
            posterior_cov  = cov_pred - K @ S_yy @ K.T

        # -- Post-update: Truncated UKF (Simon 2010) ---------------------------
        # Clamp all concentrations to non-negative (physical lower bound)
        posterior_mean = jnp.maximum(posterior_mean, jnp.float32(0.0))

        # Variance floor: prevent over-confidence after truncated-normal contraction
        posterior_cov = variance_floor(posterior_cov, _VAR_FLOOR)

        # PSD repair via eigendecomposition (Higham 1988)
        posterior_cov = nearest_psd(posterior_cov)

        # -- Fail-Loud: divergence detection -----------------------------------
        if bool(jnp.any(jnp.isnan(posterior_mean))):
            raise RuntimeError(
                "MetabolicGlucoseFilter.update_state: posterior_mean contains NaN. "
                f"CGM reading={cgm_reading}, quality_flag={quality_flag}. "
                "Check Q/R matrices and GlucoseMetabParams."
            )

        return GaussianState(mean=posterior_mean, cov=posterior_cov)
