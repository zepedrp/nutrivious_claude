"""
app/slices/metabolic_glucose/observation.py -- Observation Model V2.0

Maps the 6-state metabolic glucose state to two observable signals:
    y[0]  Glucose_obs   [mg/dL]   CGM (Dexcom G7 / Libre 3)
    y[1]  Lactate_obs   [mmol/L]  capillary analyser (sparse)

h(x, theta):
    y[0] = x[IDX_G]    -- plasma glucose (direct; interstitial delay ~7 min
                          absorbed by CGM noise sigma=5 mg/dL per spec)
    y[1] = x[IDX_LAC]  -- blood lactate

Noise:
    CGM sigma = 5 mg/dL  -> R[0,0] = 25.0
    Lactate sigma = 0.2 mmol/L -> R[1,1] = 0.04

References:
    Rebrin et al. (1999) Am J Physiol 277:E561-E571
    Rauch et al. (2021) Sensors 21:6369
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from app.slices.metabolic_glucose.ode import IDX_G, IDX_LAC, STATE_DIM, OBS_DIM


# -- Observation noise covariance
# CGM: sigma = 5 mg/dL  -> sigma^2 = 25
# Lactate: capillary CV ~5% at 3 mmol/L -> sigma ~0.15-0.20 -> sigma^2 = 0.04
R_DEFAULT: jax.Array = jnp.diag(jnp.array([25.0, 0.04], dtype=jnp.float32))
R_CGM_ONLY: jax.Array = jnp.array([[25.0]], dtype=jnp.float32)


@jax.jit
def h_obs(x: jax.Array) -> jax.Array:
    """
    Noiseless observation function: h(x) -> y in R^2.

    Returns
    -------
    y : (OBS_DIM,) = (2,)  [Glucose_obs mg/dL, Lactate_obs mmol/L]
    """
    return jnp.array([x[IDX_G], x[IDX_LAC]], dtype=jnp.float32)


@jax.jit
def h_cgm(x: jax.Array) -> jax.Array:
    """
    CGM-only: h_cgm(x) -> y in R^1.

    Returns
    -------
    y : (1,)  [Glucose_obs mg/dL]
    """
    return jnp.array([x[IDX_G]], dtype=jnp.float32)


@jax.jit
def h_cgm_sigma(sigma_pts: jax.Array) -> jax.Array:
    """
    Vectorised CGM observation over UKF sigma points.

    Parameters
    ----------
    sigma_pts : (2*STATE_DIM+1, STATE_DIM)

    Returns
    -------
    y_sigma : (2*STATE_DIM+1, 1)
    """
    return jax.vmap(h_cgm)(sigma_pts)


def inflate_R_for_quality(
    quality_flag: int,
    R_nominal: jax.Array = R_CGM_ONLY,
) -> jax.Array:
    """
    Inflate observation covariance by quality flag.

    Flags (CanonicalObservation):
        0 -- excellent  -> R * 1
        1 -- good       -> R * 1.5
        2 -- moderate   -> R * 4
        3 -- poor       -> R * 16
        4 -- missing    -> R * 1e8 (predict-only)
    """
    scale_map = {0: 1.0, 1: 1.5, 2: 4.0, 3: 16.0, 4: 1e8}
    scale = float(scale_map.get(int(quality_flag), 1e8))
    return R_nominal * jnp.float32(scale)
