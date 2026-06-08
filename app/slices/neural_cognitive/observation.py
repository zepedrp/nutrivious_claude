"""
app/slices/neural_cognitive/observation.py  — REMASTER v2.0

Observation Model — Neural/Cognitive Slice (L2/L3)

Maps the 7-state neural/cognitive state vector x to observable outputs:

    y[0]  RPE_Proxy    [1–10]         — Borg perceived exertion (log-inverse of CAR)
    y[1]  PVT_Lapses   [count/10 min] — psychomotor vigilance lapses (∝ Active_Adenosine)

OBS_DIM = 2.

Physiological mappings
──────────────────────
RPE_Proxy  (log-inverse of CAR):
    When voluntary motor drive is maximal (CAR → 1.0), perceived exertion is
    minimal (RPE → 1).  The mapping is LOGARITHMIC (concave) rather than linear,
    reflecting that small reductions from full activation are perceived more
    acutely than proportional reductions from low baseline (Borg 1982; RPE
    non-linearity from Foster et al. 2001 MSSE).

        RPE = rpe_max − rpe_range × log(1 + CAR_c × (e − 1))

    Verification:
        CAR = 1.0 → log(1 + 1.0×1.718) = log(e) = 1.0 → RPE = 10 − 9 = 1.0 ✓
        CAR = 0.0 → log(1 + 0.0) = 0.0  → RPE = 10 − 0 = 10.0 ✓
        CAR = 0.5 → RPE ≈ 10 − 9×0.620 ≈ 4.4  (concave; 50% CAR ≠ 55% RPE)

PVT_Lapses  (proportional to Active Adenosine):
    Psychomotor Vigilance Task lapses (Van Dongen 2003) are driven by EFFECTIVE
    adenosine receptor occupancy — not by raw pool size.  When caffeine is
    present, it competitively displaces adenosine → fewer lapses at the same
    sleep debt level (Nehlig 2010).

        Active_Adenosine = Adenosine_Pool / (1 + Caffeine_Plasma / K_caf)
        PVT_Lapses = k_pvt_lapse × Active_Adenosine

    Example: Adenosine_Pool = 0.73 (after 24h sleep debt), Caffeine = 4.8 mg/L:
        Active_Aden = 0.73 / (1 + 4.8/3.0) = 0.73/2.6 ≈ 0.28
        PVT ≈ 4.0 × 0.28 ≈ 1.1 lapses  (vs 2.9 without caffeine) → 62% reduction

References
──────────
    Borg G.A.V. (1982) Med Sci Sports Exerc 14(5):377–381
    Foster C. et al. (2001) Med Sci Sports Exerc 33(9):1576–1583
    Nehlig A. (2010) Neurosci Biobehav Rev 35(2):430–440
    Van Dongen H.P.A. et al. (2003) Sleep 26(2):117–126
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.neural_cognitive.ode import (
    IDX_CAR,
    IDX_ADEN,
    IDX_CAF,
    STATE_DIM,
    OBS_DIM,
    compute_active_adenosine,
)

# Euler's number as float32 constant
_E_MINUS_1: jax.Array = jnp.float32(math.e - 1.0)   # ≈ 1.7183


# ── Observation model parameters ─────────────────────────────────────────────

class NeuralCogObsParams(NamedTuple):
    """Observation model parameters for the neural/cognitive slice."""
    # RPE log-inverse mapping (Borg 1982; Foster 2001)
    rpe_min:      float = 1.0    # Borg floor (rest/passive)
    rpe_range:    float = 9.0    # Borg range [1, 10]; = rpe_max − rpe_min
    rpe_max:      float = 10.0   # Borg ceiling (maximal effort)

    # PVT lapse mapping (Van Dongen 2003; Nehlig 2010)
    k_pvt_lapse:  float = 4.0    # lapses/10 min per unit Active_Adenosine
                                  # at Aden=1.0, no caffeine: 4 lapses (moderate impairment)
    K_caf:        float = 3.0    # mg/L — A1/A2A competitive IC50 (must match ODE params)


DEFAULT_NC_OBS_PARAMS: NeuralCogObsParams = NeuralCogObsParams()


# ── Observation noise covariance (population default) ─────────────────────────
# R_NC_DEFAULT is a 2×2 diagonal matrix.
# σ_RPE = 1.0 (Borg ±1 point subjective variability; Day et al. 2004)
# σ_PVT = 1.0 (±1 lapse test-retest; Dinges & Powell 1985)

R_NC_DEFAULT: jax.Array = jnp.diag(jnp.array([1.0, 1.0], dtype=jnp.float32))


# ── Observation function ──────────────────────────────────────────────────────

def h_nc(
    x:          jax.Array,
    obs_params: NeuralCogObsParams = DEFAULT_NC_OBS_PARAMS,
) -> jax.Array:
    """
    Observation function h(x) → y ∈ ℝ² for the neural/cognitive slice.

    Pure JAX — JIT + vmap safe.

    Parameters
    ----------
    x          : shape (STATE_DIM,) = (7,)
    obs_params : NeuralCogObsParams

    Returns
    -------
    y : shape (OBS_DIM,) = (2,) — [RPE_Proxy [1–10], PVT_Lapses [count/10 min]]
    """
    CAR_c = jnp.clip(x[IDX_CAR], jnp.float32(0.0), jnp.float32(1.0))

    # RPE: logarithmic-inverse of CAR (Borg 1982; Foster 2001)
    # RPE = rpe_max − rpe_range × log(1 + CAR × (e − 1))
    # Monotone: RPE = 1 at CAR=1, RPE = 10 at CAR=0
    log_arg = jnp.float32(1.0) + CAR_c * _E_MINUS_1
    rpe = (
        jnp.float32(obs_params.rpe_max)
        - jnp.float32(obs_params.rpe_range) * jnp.log(log_arg)
    )
    rpe = jnp.clip(rpe, jnp.float32(obs_params.rpe_min), jnp.float32(obs_params.rpe_max))

    # PVT_Lapses: proportional to Active Adenosine (competitive antagonism)
    active_aden = compute_active_adenosine(x, K_caf=obs_params.K_caf)
    pvt_lapses  = jnp.float32(obs_params.k_pvt_lapse) * active_aden
    pvt_lapses  = jnp.maximum(pvt_lapses, jnp.float32(0.0))

    return jnp.stack([rpe, pvt_lapses])


def h_nc_sigma(
    sigma_pts:  jax.Array,
    obs_params: NeuralCogObsParams = DEFAULT_NC_OBS_PARAMS,
) -> jax.Array:
    """
    Apply h_nc over (2n+1) UKF sigma points.

    Parameters
    ----------
    sigma_pts  : shape (2*STATE_DIM+1, STATE_DIM) = (15, 7)
    obs_params : NeuralCogObsParams

    Returns
    -------
    y_sigma    : shape (2*STATE_DIM+1, OBS_DIM) = (15, 2)
    """
    return jax.vmap(h_nc, in_axes=(0, None))(sigma_pts, obs_params)


# ── R inflation (sparse assimilation) ─────────────────────────────────────────

def inflate_R_nc(
    quality_flags: tuple[int, int],
    R:             jax.Array = R_NC_DEFAULT,
) -> jax.Array:
    """
    Inflate observation noise R for unavailable channels.

    quality_flags : (flag_RPE, flag_PVT)
        flag = 4 → channel not available this time step → R_channel × 1e8

    Channels are available:
      RPE_Proxy  — only during/immediately after a training session
      PVT_Lapses — only when a morning cognitive test is performed

    Returns
    -------
    R_inflated : (OBS_DIM, OBS_DIM)
    """
    factor = jnp.ones(OBS_DIM, dtype=jnp.float32)
    for i, flag in enumerate(quality_flags):
        if flag == 4:
            factor = factor.at[i].set(jnp.float32(1e8))
    return R * jnp.diag(factor)
