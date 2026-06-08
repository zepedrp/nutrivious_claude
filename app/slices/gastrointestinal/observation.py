"""
app/slices/gastrointestinal/observation.py  --  GI Slice V3.0

h(x, theta) -> y in R^2 (VAS 0-10 subjective scores).

y[0]  Nausea_VAS    [0-10]  gastric volume + distress
y[1]  Bloating_VAS  [0-10]  total intestinal CHO (Intst_Glu + Intst_Fru) + distress

R base = 1.0 for both channels.
inflate_R_gi: quality-flag inflation (flag=4 -> R * 1e8).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.gastrointestinal.ode import (
    IDX_STOM_FLUID, IDX_INTST_GLU, IDX_INTST_FRU, IDX_DISTRESS,
    STATE_DIM, OBS_DIM,
    GIv3Params, DEFAULT_GI_PARAMS,
)


class GIObsParams(NamedTuple):
    nausea_vol_thresh:  float = 0.70   # [L]   gastric sigmoid centre
    nausea_vol_scale:   float = 0.15   # [L]   sigmoid half-width
    w_nausea_vol:       float = 0.60   # volume component weight in Nausea_VAS
    w_nausea_dist:      float = 0.40   # distress component weight

    bloat_cho_thresh:   float = 15.0   # [g]   total intestinal CHO sigmoid centre
    bloat_cho_scale:    float = 7.0    # [g]   sigmoid half-width
    w_bloat_cho:        float = 0.70   # CHO component weight in Bloating_VAS
    w_bloat_dist:       float = 0.30   # distress component weight

    sigma_nausea:       float = 1.0    # base noise [VAS units]
    sigma_bloating:     float = 1.0


DEFAULT_GI_OBS_PARAMS: GIObsParams = GIObsParams()

R_GI_DEFAULT: jax.Array = jnp.diag(jnp.array([1.0, 1.0], dtype=jnp.float32))


@jax.jit
def h_gi(
    x:          jax.Array,
    obs_params: GIObsParams = DEFAULT_GI_OBS_PARAMS,
    gi_params:  GIv3Params  = DEFAULT_GI_PARAMS,
) -> jax.Array:
    """
    h(x, theta) -> y (2,).

    y[0]  Nausea_VAS   [0-10]
    y[1]  Bloating_VAS [0-10]
    """
    stom_fluid  = jnp.maximum(x[IDX_STOM_FLUID], jnp.float32(0.0))
    intst_glu   = jnp.maximum(x[IDX_INTST_GLU],  jnp.float32(0.0))
    intst_fru   = jnp.maximum(x[IDX_INTST_FRU],  jnp.float32(0.0))
    distress    = jnp.clip(x[IDX_DISTRESS], jnp.float32(0.0), jnp.float32(1.0))
    total_intst = intst_glu + intst_fru

    # volume sigmoid
    vol_sig = jnp.float32(0.5) * (
        jnp.float32(1.0) + jnp.tanh(
            (stom_fluid - jnp.float32(obs_params.nausea_vol_thresh))
            / jnp.float32(obs_params.nausea_vol_scale)
        )
    )

    nausea_raw = (
        jnp.float32(obs_params.w_nausea_vol)  * vol_sig
        + jnp.float32(obs_params.w_nausea_dist) * distress
    )
    Nausea_VAS = jnp.float32(10.0) * jnp.clip(nausea_raw, jnp.float32(0.0), jnp.float32(1.0))

    # total intestinal CHO sigmoid (Intst_Glu + Intst_Fru)
    cho_sig = jnp.float32(0.5) * (
        jnp.float32(1.0) + jnp.tanh(
            (total_intst - jnp.float32(obs_params.bloat_cho_thresh))
            / jnp.float32(obs_params.bloat_cho_scale)
        )
    )

    bloat_raw = (
        jnp.float32(obs_params.w_bloat_cho)  * cho_sig
        + jnp.float32(obs_params.w_bloat_dist) * distress
    )
    Bloating_VAS = jnp.float32(10.0) * jnp.clip(bloat_raw, jnp.float32(0.0), jnp.float32(1.0))

    return jnp.array([Nausea_VAS, Bloating_VAS], dtype=jnp.float32)


@jax.jit
def h_gi_sigma(
    sigma_pts:  jax.Array,
    obs_params: GIObsParams = DEFAULT_GI_OBS_PARAMS,
    gi_params:  GIv3Params  = DEFAULT_GI_PARAMS,
) -> jax.Array:
    """Vectorised h_gi over (2n+1) sigma points. Returns (2n+1, OBS_DIM)."""
    return jax.vmap(h_gi, in_axes=(0, None, None))(sigma_pts, obs_params, gi_params)


def inflate_R_gi(
    quality_flag: int,
    R_nominal:    jax.Array = R_GI_DEFAULT,
) -> jax.Array:
    """Inflate R by quality flag. flag=4 -> R * 1e8 (predict-only)."""
    _MAP = {0: 1.0, 1: 2.0, 2: 5.0, 3: 20.0, 4: 1e8}
    scale = jnp.float32(_MAP.get(quality_flag, 1e8))
    return R_nominal * scale
