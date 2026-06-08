"""
app/slices/neuroendocrine/nlme.py — L3 NLME: SAM × HPA × Somatotropic  V2.0  (D=3)

Personalises 3 physiological parameters per individual (θ_i) from sparse
Epinephrine, Norepinephrine, Cortisol, and IGF-1 data via NLME (NumPyro).

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015):
    η_raw_i ~ N(0, I_D)
    η_i     = L_Ω · η_raw_i
    θ_i     = θ_pop · exp(η_i)   [log-normal; always positive]

Personalised parameters (D=3)
-------------------------------
    GC_sensitivity                  — glucocorticoid receptor density
                                      High → lower cortisol set-point
                                      Prior: LogN(log 1.0, 0.25²)

    IGF1_conv_rate                  — hepatic GH→IGF-1 conversion rate [h⁻¹]
                                      Proportional to functional GH receptor density
                                      Prior: LogN(log 6.0, 0.30²)

    Metabolic_Flexibility_Threshold — glucose threshold for CRH panic [mg/dL]
                                      Fat-adapted athletes: ~55 mg/dL
                                      CHO-dependent athletes: ~75 mg/dL
                                      Prior: LogN(log 70.0, 0.15²)

Identifiability:
    GC_sensitivity  ← morning cortisol peak height and diurnal amplitude
    IGF1_conv_rate  ← IGF-1 level relative to estimated GH exposure
    MFT             ← Cortisol response during hypoglycaemic episodes
                      (requires data with glucose drops below ~70 mg/dL)

Fail-Loud: RuntimeError on NaN ELBO, missing dependency.

References
----------
    Betancourt & Girolami (2015) arXiv:1312.0906   — non-centred NLME
    Keenan & Veldhuis (1997) Am J Physiol 273:E1182 — HPA pulsatility
    Cryer (2001) J Clin Invest 108:1533            — glucose counterregulation
    CLAUDE.md §3 prior table
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    _NUMPYRO_OK = True
except ImportError:
    _NUMPYRO_OK = False

try:
    import optax
    _OPTAX_OK = True
except ImportError:
    _OPTAX_OK = False

from app.slices.neuroendocrine.ode import (
    NeuroParams, DEFAULT_NEURO_PARAMS,
    STATE_DIM, OBS_DIM,
    initial_state, integrate_1h,
)
from app.slices.neuroendocrine.observation import (
    NeuroObsParams, DEFAULT_OBS_PARAMS,
    h_neuro,
)

_LOG = logging.getLogger(__name__)

# ── Dimension ─────────────────────────────────────────────────────────────────

PARAM_NAMES = ("GC_sensitivity", "IGF1_conv_rate", "Metabolic_Flexibility_Threshold")
D = 3


# ── Prior hyperparameters ─────────────────────────────────────────────────────

class NeuroNLMEPriors(NamedTuple):
    """
    Log-space population priors for D=3 neuroendocrine parameters.
    All parameters are positive → log-normal parametrisation.
    """
    # GC_sensitivity — log(1.0) = 0
    log_GC_mean:   float = 0.000     # log(1.0)
    log_GC_sd:     float = 0.250

    # IGF1_conv_rate — log(6.0) h⁻¹
    log_IGF1_mean: float = 1.792     # log(6.0)
    log_IGF1_sd:   float = 0.300

    # Metabolic_Flexibility_Threshold — log(70.0) mg/dL
    log_MFT_mean:  float = 4.248     # log(70.0)
    log_MFT_sd:    float = 0.150     # narrow: physiologically constrained


DEFAULT_NLME_PRIORS = NeuroNLMEPriors()


# ── Individual posterior container ────────────────────────────────────────────

class NeuroIndividualParams(NamedTuple):
    """Posterior kinetic parameters θ_i for one subject."""
    GC_sensitivity:                  float
    IGF1_conv_rate:                  float
    Metabolic_Flexibility_Threshold: float
    GC_sensitivity_sd:               float = float("nan")
    IGF1_conv_rate_sd:               float = float("nan")
    MFT_sd:                          float = float("nan")


def params_from_individual(ind: NeuroIndividualParams) -> NeuroParams:
    """Build NeuroParams with personalised NLME parameters; all others fixed."""
    base = DEFAULT_NEURO_PARAMS
    return NeuroParams(
        GC_sensitivity                  = ind.GC_sensitivity,
        IGF1_conv_rate                  = ind.IGF1_conv_rate,
        Metabolic_Flexibility_Threshold = ind.Metabolic_Flexibility_Threshold,
        # Fixed at population priors ──────────────────────────────────────────
        gamma_Epi    = base.gamma_Epi,    k_Epi_basal  = base.k_Epi_basal,
        k_Epi_stress = base.k_Epi_stress,
        gamma_NE     = base.gamma_NE,     k_NE_basal   = base.k_NE_basal,
        k_NE_stress  = base.k_NE_stress,
        gamma_CRH    = base.gamma_CRH,    s_CRH0       = base.s_CRH0,
        A_CRH        = base.A_CRH,        phi_CRH      = base.phi_CRH,
        k_CRH_IL6    = base.k_CRH_IL6,    k_CRH_hypo   = base.k_CRH_hypo,
        k_CRH_CAR    = base.k_CRH_CAR,    k_CRH_stress = base.k_CRH_stress,
        K_CRH_fb     = base.K_CRH_fb,     sigma_CRH_fb = base.sigma_CRH_fb,
        gamma_ACTH   = base.gamma_ACTH,   k_ACTH_CRH   = base.k_ACTH_CRH,
        K_ACTH_fb    = base.K_ACTH_fb,    sigma_ACTH_fb= base.sigma_ACTH_fb,
        gamma_Cort   = base.gamma_Cort,   k_Cort_ACTH  = base.k_Cort_ACTH,
        gamma_GHRH   = base.gamma_GHRH,   s_GHRH0      = base.s_GHRH0,
        A_GHRH       = base.A_GHRH,       phi_GHRH     = base.phi_GHRH,
        k_SWS_GHRH   = base.k_SWS_GHRH,
        gamma_SS     = base.gamma_SS,     k_SS_basal   = base.k_SS_basal,
        k_SS_Cort    = base.k_SS_Cort,    k_SS_SAM     = base.k_SS_SAM,
        gamma_GH     = base.gamma_GH,     k_GH_GHRH    = base.k_GH_GHRH,
        K_SS_inhib   = base.K_SS_inhib,
        gamma_IGF1   = base.gamma_IGF1,   k_IGF1_basal = base.k_IGF1_basal,
    )


# ── Differentiable forward pass ───────────────────────────────────────────────

def _predict_one_subject(
    theta_raw: jnp.ndarray,
    x0: jnp.ndarray,
    n_hours: int,
    priors: NeuroNLMEPriors,
    obs_params: NeuroObsParams,
    t0: float = 8.0,
) -> jnp.ndarray:
    """
    theta_raw ∈ ℝ³ (non-centred; 0 → population mean).
    Returns (n_hours, OBS_DIM=4) predictions.

    Forward control: resting euglycaemic (no stress, glucose=90).
    Identification of MFT requires hypoglycaemic episodes in the data;
    for NLME fitting on standard telemetry, GC_sensitivity and IGF1_conv_rate
    are the primary identifiable parameters.
    """
    GC_s   = jnp.exp(priors.log_GC_mean   + priors.log_GC_sd   * theta_raw[0])
    IGF1_r = jnp.exp(priors.log_IGF1_mean + priors.log_IGF1_sd * theta_raw[1])
    MFT    = jnp.exp(priors.log_MFT_mean  + priors.log_MFT_sd  * theta_raw[2])

    base = DEFAULT_NEURO_PARAMS
    params_i = NeuroParams(
        GC_sensitivity                  = GC_s,
        IGF1_conv_rate                  = IGF1_r,
        Metabolic_Flexibility_Threshold = MFT,
        gamma_Epi    = base.gamma_Epi,    k_Epi_basal  = base.k_Epi_basal,
        k_Epi_stress = base.k_Epi_stress,
        gamma_NE     = base.gamma_NE,     k_NE_basal   = base.k_NE_basal,
        k_NE_stress  = base.k_NE_stress,
        gamma_CRH    = base.gamma_CRH,    s_CRH0       = base.s_CRH0,
        A_CRH        = base.A_CRH,        phi_CRH      = base.phi_CRH,
        k_CRH_IL6    = base.k_CRH_IL6,    k_CRH_hypo   = base.k_CRH_hypo,
        k_CRH_CAR    = base.k_CRH_CAR,    k_CRH_stress = base.k_CRH_stress,
        K_CRH_fb     = base.K_CRH_fb,     sigma_CRH_fb = base.sigma_CRH_fb,
        gamma_ACTH   = base.gamma_ACTH,   k_ACTH_CRH   = base.k_ACTH_CRH,
        K_ACTH_fb    = base.K_ACTH_fb,    sigma_ACTH_fb= base.sigma_ACTH_fb,
        gamma_Cort   = base.gamma_Cort,   k_Cort_ACTH  = base.k_Cort_ACTH,
        gamma_GHRH   = base.gamma_GHRH,   s_GHRH0      = base.s_GHRH0,
        A_GHRH       = base.A_GHRH,       phi_GHRH     = base.phi_GHRH,
        k_SWS_GHRH   = base.k_SWS_GHRH,
        gamma_SS     = base.gamma_SS,     k_SS_basal   = base.k_SS_basal,
        k_SS_Cort    = base.k_SS_Cort,    k_SS_SAM     = base.k_SS_SAM,
        gamma_GH     = base.gamma_GH,     k_GH_GHRH    = base.k_GH_GHRH,
        K_SS_inhib   = base.K_SS_inhib,
        gamma_IGF1   = base.gamma_IGF1,   k_IGF1_basal = base.k_IGF1_basal,
    )

    def step(carry, step_idx):
        t_h  = t0 + step_idx.astype(float)
        ctrl = lambda _: jnp.array([0.0, 0.0, 0.0, 90.0, 0.0])
        x_next = integrate_1h(carry, params=params_i, control_fn=ctrl, t0=t_h)
        y_pred = h_neuro(x_next, obs_params=obs_params)
        return x_next, y_pred

    _, y_preds = jax.lax.scan(step, x0, jnp.arange(n_hours))
    return y_preds   # (n_hours, OBS_DIM=4)


# ── NumPyro model ─────────────────────────────────────────────────────────────

def build_neuro_nlme_model(
    n_hours: int,
    priors: NeuroNLMEPriors = DEFAULT_NLME_PRIORS,
    obs_params: NeuroObsParams = DEFAULT_OBS_PARAMS,
):
    """
    Factory returning a NumPyro model for the SAM × HPA × Somatotropic NLME.

    model(y_obs, subject_idx, x0s)
      y_obs       : (N_total_obs, OBS_DIM=4)
      subject_idx : (N_total_obs,) int
      x0s         : (N_subjects, STATE_DIM=9)
    """
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro required. pip install numpyro")

    sigma_y = jnp.array([
        obs_params.sigma_Epi,
        obs_params.sigma_NE,
        obs_params.sigma_Cortisol,
        obs_params.sigma_IGF1,
    ])

    def model(y_obs, subject_idx, x0s):
        N_subjects = x0s.shape[0]

        Omega_chol = numpyro.sample(
            "Omega_chol",
            dist.LKJCholesky(D, concentration=2.0),
        )

        with numpyro.plate("subjects", N_subjects):
            eta_raw = numpyro.sample(
                "eta_raw",
                dist.Normal(jnp.zeros(D), jnp.ones(D)),
            )

        eta_i = eta_raw @ Omega_chol.T   # (N_subjects, D)

        def _pred_s(args):
            eta_s, x0_s = args
            return _predict_one_subject(eta_s, x0_s, n_hours, priors, obs_params)

        y_pred_all  = jax.vmap(_pred_s)((eta_i, x0s))   # (N_subj, n_hours, OBS_DIM)
        y_pred_flat = y_pred_all[subject_idx]

        numpyro.sample(
            "y",
            dist.Normal(y_pred_flat, sigma_y[None, :]).to_event(1),
            obs=y_obs,
        )

    return model


# ── SVI training ──────────────────────────────────────────────────────────────

class NeuroNLMEResult(NamedTuple):
    svi_state:        object
    params:           dict
    elbo_history:     jnp.ndarray
    individual_means: jnp.ndarray   # (N_subjects, D)
    individual_sds:   jnp.ndarray


def fit_neuro_nlme(
    y_obs: jnp.ndarray,
    subject_idx: jnp.ndarray,
    x0s: jnp.ndarray,
    n_hours: int,
    priors: NeuroNLMEPriors = DEFAULT_NLME_PRIORS,
    obs_params: NeuroObsParams = DEFAULT_OBS_PARAMS,
    n_steps: int = 20_000,
    lr: float = 5e-4,
    seed: int = 42,
) -> NeuroNLMEResult:
    """Fit the SAM × HPA × Somatotropic NLME via SVI. Fail-Loud on NaN ELBO."""
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro required.")
    if not _OPTAX_OK:
        raise RuntimeError("optax required.")

    model = build_neuro_nlme_model(n_hours, priors, obs_params)
    guide = AutoLowRankMultivariateNormal(model, rank=3)
    optim = numpyro.optim.optax_to_numpyro(optax.adam(lr))
    svi   = SVI(model, guide, optim, loss=Trace_ELBO())

    rng       = jax.random.PRNGKey(seed)
    svi_state = svi.init(rng, y_obs, subject_idx, x0s)

    elbo_hist = []
    for step in range(n_steps):
        svi_state, loss = svi.update(svi_state, y_obs, subject_idx, x0s)
        elbo_hist.append(-float(loss))
        if jnp.isnan(loss):
            raise RuntimeError(f"NeuroNLME: ELBO NaN at step {step}.")
        if step % 5_000 == 0:
            _LOG.info("NeuroNLME SVI step %d  ELBO=%.1f", step, -float(loss))

    params   = svi.get_params(svi_state)
    site_loc = params.get("eta_raw_loc",   jnp.zeros((x0s.shape[0], D)))
    site_sd  = params.get("eta_raw_scale", jnp.ones((x0s.shape[0], D)) * 0.1)

    return NeuroNLMEResult(
        svi_state        = svi_state,
        params           = params,
        elbo_history     = jnp.array(elbo_hist),
        individual_means = site_loc,
        individual_sds   = site_sd,
    )


def individual_params_from_nlme(
    result: NeuroNLMEResult,
    subject_id: int,
    priors: NeuroNLMEPriors = DEFAULT_NLME_PRIORS,
) -> NeuroIndividualParams:
    eta   = result.individual_means[subject_id]
    eta_s = result.individual_sds[subject_id]

    GC_s   = float(jnp.exp(priors.log_GC_mean   + priors.log_GC_sd   * eta[0]))
    IGF1_r = float(jnp.exp(priors.log_IGF1_mean + priors.log_IGF1_sd * eta[1]))
    MFT    = float(jnp.exp(priors.log_MFT_mean  + priors.log_MFT_sd  * eta[2]))

    return NeuroIndividualParams(
        GC_sensitivity                  = GC_s,
        IGF1_conv_rate                  = IGF1_r,
        Metabolic_Flexibility_Threshold = MFT,
        GC_sensitivity_sd = float(GC_s   * priors.log_GC_sd   * eta_s[0]),
        IGF1_conv_rate_sd = float(IGF1_r * priors.log_IGF1_sd * eta_s[1]),
        MFT_sd            = float(MFT    * priors.log_MFT_sd  * eta_s[2]),
    )


def cold_start_params(
    athlete_data: dict | None = None,
    priors: NeuroNLMEPriors = DEFAULT_NLME_PRIORS,
) -> NeuroIndividualParams:
    """
    Population-prior individual parameters for a new user (cold start).

    Optional athlete_data keys:
        'fat_adapted' (bool) — shifts MFT prior toward 55 mg/dL (max ±0.5·σ)
    """
    athlete_data = athlete_data or {}
    fat_adapted  = bool(athlete_data.get("fat_adapted", False))

    MFT_shift = -0.5 * priors.log_MFT_sd if fat_adapted else 0.0
    MFT_val   = float(jnp.exp(jnp.array(priors.log_MFT_mean + MFT_shift)))

    return NeuroIndividualParams(
        GC_sensitivity                  = float(jnp.exp(jnp.array(priors.log_GC_mean))),
        IGF1_conv_rate                  = float(jnp.exp(jnp.array(priors.log_IGF1_mean))),
        Metabolic_Flexibility_Threshold = MFT_val,
    )
