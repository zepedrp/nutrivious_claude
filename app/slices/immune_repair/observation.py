"""
app/slices/immune_repair/observation.py -- L2/L3 Observation Model

Observation  y in R^2:
    y[0]  hsCRP_obs   [mg/L]    high-sensitivity CRP (hepatic IL-6 proxy, ~6-24h lag)
    y[1]  CK_obs      [U/L]     creatine kinase (muscle damage proxy)

Both are noisy linear proxies of hidden states.

    hsCRP = hsCRP_basal + k_hsCRP * IL6   (IL-6 drives acute-phase response)
    CK    = CK_basal    + k_CK    * D     (damage releases CK into bloodstream)

References
----------
    Gabay & Kushner (1999) NEJM 340:448          -- IL-6 -> CRP hepatic synthesis
    Minetto et al. (2011) J Appl Physiol 111:687 -- CK as muscle damage biomarker
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from app.slices.immune_repair.ode import IDX_DMG, IDX_IL6, OBS_DIM

# ── Observation indices ────────────────────────────────────────────────────────
IDX_HSCRP = 0
IDX_CK    = 1


# ── Observation parameters ─────────────────────────────────────────────────────

class ImmuneObsParams(NamedTuple):
    hsCRP_basal:  float = 0.50     # [mg/L]  resting background CRP
    k_hsCRP:      float = 0.08     # [mg/L per pg/mL IL-6]  hepatic conversion
    CK_basal:     float = 80.0     # [U/L]   resting creatine kinase
    k_CK:         float = 400.0    # [U/L per au damage]
    sigma_hsCRP:  float = 0.50     # [mg/L]  assay noise
    sigma_CK:     float = 60.0     # [U/L]   assay noise


DEFAULT_OBS_PARAMS = ImmuneObsParams()


# ── Observation function ───────────────────────────────────────────────────────

def h_immune(x: jnp.ndarray, obs_params: ImmuneObsParams = DEFAULT_OBS_PARAMS) -> jnp.ndarray:
    """
    h(x, theta) -> y in R^2  [hsCRP_mg_L, CK_U_L]

    JIT-compatible; no Python branching.
    """
    D   = jnp.maximum(x[IDX_DMG], 0.0)
    IL6 = jnp.maximum(x[IDX_IL6], 0.0)

    hsCRP = obs_params.hsCRP_basal + obs_params.k_hsCRP * IL6
    CK    = obs_params.CK_basal    + obs_params.k_CK    * D

    return jnp.array([hsCRP, CK])


def h_immune_vmap(sigma_pts: jnp.ndarray, obs_params: ImmuneObsParams = DEFAULT_OBS_PARAMS) -> jnp.ndarray:
    """Map h_immune over (2n+1) sigma points."""
    import jax
    return jax.vmap(lambda x: h_immune(x, obs_params))(sigma_pts)


# ── Observation noise matrix ───────────────────────────────────────────────────

R_IMMUNE_DEFAULT = jnp.diag(jnp.array([
    DEFAULT_OBS_PARAMS.sigma_hsCRP ** 2,   # hsCRP variance
    DEFAULT_OBS_PARAMS.sigma_CK    ** 2,   # CK variance
]))


def observation_noise_R(
    obs_params: ImmuneObsParams = DEFAULT_OBS_PARAMS,
    quality_flags: tuple[int, int] = (0, 0),
) -> jnp.ndarray:
    """
    Build 2x2 diagonal R.
    quality_flag = 4  ->  R[i,i] *= 1e8  (predict-only channel).
    """
    r_hscrp = obs_params.sigma_hsCRP ** 2
    r_ck    = obs_params.sigma_CK    ** 2

    # flag==4 means channel is absent -> inflate R so UKF effectively ignores it
    r_hscrp = jnp.where(quality_flags[0] == 4, r_hscrp * 1e8, r_hscrp)
    r_ck    = jnp.where(quality_flags[1] == 4, r_ck    * 1e8, r_ck)

    return jnp.diag(jnp.array([r_hscrp, r_ck]))
