"""
app/slices/neuroendocrine/observation.py — Observation Model h(x, θ) → y  V2.0

Observation vector  y ∈ ℝ⁴:
    y[0]  Epinephrine_pgmL     plasma epinephrine               [pg/mL]
    y[1]  Norepinephrine_pgmL  plasma norepinephrine            [pg/mL]
    y[2]  Cortisol_nmolL       serum cortisol                   [nmol/L]
    y[3]  IGF1_ngmL            serum IGF-1                      [ng/mL]

All four are direct state readouts with additive Gaussian noise.

Noise model (diagonal R)
------------------------
    σ_Epi       = 50  pg/mL   plasma catecholamine ELISA, CV ~10% at 500 pg/mL
    σ_NE        = 100 pg/mL   plasma catecholamine ELISA, CV ~33% at 300 pg/mL
    σ_Cortisol  = 28  nmol/L  immunoassay CV ~10% at 280 nmol/L
    σ_IGF1      = 25  ng/mL   ELISA CV ~10% at 250 ng/mL

quality_flag inflation convention (identical across all Nutrivious slices):
    0 → ×1    (validated lab draw)
    1 → ×2    (point-of-care / salivary assay)
    2 → ×4    (wearable proxy / indirect marker)
    3 → ×10   (proxy, no direct lab validation)
    4 → ×1e8  (missing — predict-only step)

References
----------
    Goldstein (2010) Cell Mol Neurobiol 30:1283      — plasma catecholamine CV
    Kirschbaum & Hellhammer (1989) Psychoneuroendo.  — cortisol assay precision
    Bidlingmaier & Freda (2010) Growth Horm IGF Res  — IGF-1 ELISA precision
    CLAUDE.md §3 prior table
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from app.slices.neuroendocrine.ode import (
    IDX_EPI, IDX_NE, IDX_CORT, IDX_IGF1,
    STATE_DIM, OBS_DIM,
    NeuroParams, DEFAULT_NEURO_PARAMS,
)


# ── Observation parameter container ───────────────────────────────────────────

class NeuroObsParams(NamedTuple):
    """
    Observation noise standard deviations for the 4-channel neuroendocrine model.
    """
    sigma_Epi:      float = 50.0    # pg/mL  plasma epinephrine
    sigma_NE:       float = 100.0   # pg/mL  plasma norepinephrine
    sigma_Cortisol: float = 28.0    # nmol/L immunoassay
    sigma_IGF1:     float = 25.0    # ng/mL  ELISA


DEFAULT_OBS_PARAMS = NeuroObsParams()


# ── Quality-flag inflation factors ────────────────────────────────────────────

_QUALITY_INFLATE = jnp.array([1.0, 2.0, 4.0, 10.0, 1e8, 1e8])


# ── Observation function ──────────────────────────────────────────────────────

def h_neuro(
    x: jnp.ndarray,
    obs_params: NeuroObsParams = DEFAULT_OBS_PARAMS,
    neuro_params: NeuroParams = DEFAULT_NEURO_PARAMS,
) -> jnp.ndarray:
    """
    h: state ℝ⁹ → observations ℝ⁴.

    y[0] = Epinephrine     [pg/mL]  — direct readout of x[IDX_EPI]
    y[1] = Norepinephrine  [pg/mL]  — direct readout of x[IDX_NE]
    y[2] = Cortisol        [nmol/L] — direct readout of x[IDX_CORT]
    y[3] = IGF-1           [ng/mL]  — direct readout of x[IDX_IGF1]

    JAX-traceable; no data-dependent branches.
    """
    return jnp.array([x[IDX_EPI], x[IDX_NE], x[IDX_CORT], x[IDX_IGF1]])


# ── Observation noise covariance ──────────────────────────────────────────────

def observation_noise_R(
    obs_params: NeuroObsParams = DEFAULT_OBS_PARAMS,
    quality_flag: int = 0,
) -> jnp.ndarray:
    """
    Diagonal R matrix (OBS_DIM × OBS_DIM) with quality-flag inflation.

    All four channels receive the same inflation factor (scalar quality_flag).
    For per-channel masking, set specific channels to quality_flag=4 externally
    by passing an inflated R directly to the UKF update step.

    Returns shape (4, 4).
    """
    inflate = _QUALITY_INFLATE[jnp.clip(quality_flag, 0, 5)]
    sigmas  = jnp.array([
        obs_params.sigma_Epi,
        obs_params.sigma_NE,
        obs_params.sigma_Cortisol,
        obs_params.sigma_IGF1,
    ])
    return jnp.diag((sigmas ** 2) * inflate)


def inflate_R_neuro(
    obs_params: NeuroObsParams = DEFAULT_OBS_PARAMS,
    flags: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> jnp.ndarray:
    """
    Per-channel R inflation for sparse assimilation.

    flags: (flag_Epi, flag_NE, flag_Cort, flag_IGF1)
    flag=4 → channel R×1e8 (predict-only for that channel).
    """
    inflate = jnp.array([
        _QUALITY_INFLATE[jnp.clip(flags[0], 0, 5)],
        _QUALITY_INFLATE[jnp.clip(flags[1], 0, 5)],
        _QUALITY_INFLATE[jnp.clip(flags[2], 0, 5)],
        _QUALITY_INFLATE[jnp.clip(flags[3], 0, 5)],
    ])
    sigmas = jnp.array([
        obs_params.sigma_Epi,
        obs_params.sigma_NE,
        obs_params.sigma_Cortisol,
        obs_params.sigma_IGF1,
    ])
    return jnp.diag((sigmas ** 2) * inflate)
