"""
app/slices/immune_repair/nlme.py -- L3 NLME: Immune Repair  (D=2)

Personalises 2 physiological parameters per individual via NLME (NumPyro).
Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015).

Personalised parameters (D=2)
------------------------------
    M1_Activation_Rate        -- inter-individual variability in the inflammatory
                                  response magnitude (Tidball & Villalta 2010).
                                  Prior: LogN(log 0.40, 0.35^2)

    M1_M2_Polarization_Time   -- time constant (hours) for M1->M2 polarisation;
                                  partially driven by IL-4/IL-10 genetics.
                                  Shorter = faster repair phase.
                                  Prior: LogN(log 12.5, 0.30^2)

Identifiability:
    M1_Activation_Rate   <- peak M1 amplitude relative to damage stimulus
    Polarization_Time    <- lag between damage peak and M2 peak in longitudinal data

Fail-Loud: RuntimeError on NaN ELBO, missing numpyro dependency.

References
----------
    Betancourt & Girolami (2015) arXiv:1312.0906
    Tidball & Villalta (2010) Am J Physiol Cell Physiol 298:C1173
    Peake et al. (2017) J Physiol 595:5981
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

from app.slices.immune_repair.ode import (
    ImmuneParams, DEFAULT_IMMUNE_PARAMS,
    STATE_DIM, OBS_DIM,
    initial_state, integrate_1h,
)
from app.slices.immune_repair.observation import (
    ImmuneObsParams, DEFAULT_OBS_PARAMS,
    h_immune,
)

_LOG = logging.getLogger(__name__)

PARAM_NAMES = ("M1_Activation_Rate", "M1_M2_Polarization_Time")
D = 2


# ── Prior hyperparameters ─────────────────────────────────────────────────────

class ImmuneNLMEPriors(NamedTuple):
    log_M1_act_mean:   float = math.log(0.40)    # log(k_M1_recruit prior)
    log_M1_act_sd:     float = 0.35
    log_polar_mean:    float = math.log(12.5)    # log(polarisation t1/2 in hours)
    log_polar_sd:      float = 0.30


DEFAULT_NLME_PRIORS = ImmuneNLMEPriors()


# ── Individual params ─────────────────────────────────────────────────────────

class ImmuneIndividualParams(NamedTuple):
    M1_Activation_Rate:      float
    M1_M2_Polarization_Time: float
    M1_act_sd:               float = float("nan")
    polar_sd:                float = float("nan")


def params_from_individual(ind: ImmuneIndividualParams) -> ImmuneParams:
    base  = DEFAULT_IMMUNE_PARAMS
    k_pol = math.log(2) / ind.M1_M2_Polarization_Time   # t1/2 -> rate
    return ImmuneParams(
        k_M1_recruit = ind.M1_Activation_Rate,
        k_polar      = k_pol,
        # Fixed at population priors
        k_dmg          = base.k_dmg,
        k_M2_repair    = base.k_M2_repair,
        k_cort_supp    = base.k_cort_supp,
        Cortisol_threshold = base.Cortisol_threshold,
        k_M2_decay     = base.k_M2_decay,
        k_il6_myo      = base.k_il6_myo,
        k_il6_M1       = base.k_il6_M1,
        gamma_IL6      = base.gamma_IL6,
    )


# ── Differentiable forward pass ────────────────────────────────────────────────

def _predict_one_subject(
    theta_raw: jnp.ndarray,
    x0: jnp.ndarray,
    n_hours: int,
    priors: ImmuneNLMEPriors,
    obs_params: ImmuneObsParams,
    t0: float = 0.0,
) -> jnp.ndarray:
    """
    theta_raw in R^2 (non-centred; 0 -> population mean).
    Returns (n_hours, OBS_DIM=2).
    """
    M1_act = jnp.exp(priors.log_M1_act_mean + priors.log_M1_act_sd  * theta_raw[0])
    polar_t = jnp.exp(priors.log_polar_mean + priors.log_polar_sd   * theta_raw[1])
    k_pol   = math.log(2) / 12.5   # anchor; individual variation via theta
    k_pol   = jnp.log(2.0) / polar_t

    base = DEFAULT_IMMUNE_PARAMS
    params_i = ImmuneParams(
        k_M1_recruit       = M1_act,
        k_polar            = k_pol,
        k_dmg              = base.k_dmg,
        k_M2_repair        = base.k_M2_repair,
        k_cort_supp        = base.k_cort_supp,
        Cortisol_threshold = base.Cortisol_threshold,
        k_M2_decay         = base.k_M2_decay,
        k_il6_myo          = base.k_il6_myo,
        k_il6_M1           = base.k_il6_M1,
        gamma_IL6          = base.gamma_IL6,
    )

    def step(carry, step_idx):
        t_h    = t0 + step_idx.astype(float)
        ctrl   = lambda _: jnp.array([0.0, 0.0, 300.0])   # resting
        x_next = integrate_1h(carry, params=params_i, control_fn=ctrl, t0=t_h)
        y_pred = h_immune(x_next, obs_params=obs_params)
        return x_next, y_pred

    _, y_preds = jax.lax.scan(step, x0, jnp.arange(n_hours))
    return y_preds   # (n_hours, OBS_DIM=2)


# ── NumPyro model ─────────────────────────────────────────────────────────────

def build_immune_nlme_model(
    n_hours: int,
    priors: ImmuneNLMEPriors = DEFAULT_NLME_PRIORS,
    obs_params: ImmuneObsParams = DEFAULT_OBS_PARAMS,
):
    """Factory returning a NumPyro model for the immune repair NLME."""
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro required. pip install numpyro")

    sigma_y = jnp.array([obs_params.sigma_hsCRP, obs_params.sigma_CK])

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

class ImmuneNLMEResult(NamedTuple):
    svi_state:        object
    params:           dict
    elbo_history:     jnp.ndarray
    individual_means: jnp.ndarray
    individual_sds:   jnp.ndarray


def fit_immune_nlme(
    y_obs: jnp.ndarray,
    subject_idx: jnp.ndarray,
    x0s: jnp.ndarray,
    n_hours: int,
    priors: ImmuneNLMEPriors = DEFAULT_NLME_PRIORS,
    obs_params: ImmuneObsParams = DEFAULT_OBS_PARAMS,
    n_steps: int = 20_000,
    lr: float = 5e-4,
    seed: int = 42,
) -> ImmuneNLMEResult:
    """Fit immune NLME via SVI. Fail-Loud on NaN ELBO."""
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro required.")
    if not _OPTAX_OK:
        raise RuntimeError("optax required.")

    model = build_immune_nlme_model(n_hours, priors, obs_params)
    guide = AutoLowRankMultivariateNormal(model, rank=2)
    optim = numpyro.optim.optax_to_numpyro(optax.adam(lr))
    svi   = SVI(model, guide, optim, loss=Trace_ELBO())

    rng       = jax.random.PRNGKey(seed)
    svi_state = svi.init(rng, y_obs, subject_idx, x0s)

    elbo_hist = []
    for step in range(n_steps):
        svi_state, loss = svi.update(svi_state, y_obs, subject_idx, x0s)
        elbo_hist.append(-float(loss))
        if jnp.isnan(loss):
            raise RuntimeError(f"ImmuneNLME: ELBO NaN at step {step}.")
        if step % 5_000 == 0:
            _LOG.info("ImmuneNLME SVI step %d  ELBO=%.1f", step, -float(loss))

    params   = svi.get_params(svi_state)
    site_loc = params.get("eta_raw_loc",   jnp.zeros((x0s.shape[0], D)))
    site_sd  = params.get("eta_raw_scale", jnp.ones((x0s.shape[0], D)) * 0.1)

    return ImmuneNLMEResult(
        svi_state        = svi_state,
        params           = params,
        elbo_history     = jnp.array(elbo_hist),
        individual_means = site_loc,
        individual_sds   = site_sd,
    )


def cold_start_params(
    athlete_data: dict | None = None,
    priors: ImmuneNLMEPriors = DEFAULT_NLME_PRIORS,
) -> ImmuneIndividualParams:
    """Population-prior individual parameters for a new user (cold start)."""
    return ImmuneIndividualParams(
        M1_Activation_Rate      = float(jnp.exp(jnp.array(priors.log_M1_act_mean))),
        M1_M2_Polarization_Time = float(jnp.exp(jnp.array(priors.log_polar_mean))),
    )
