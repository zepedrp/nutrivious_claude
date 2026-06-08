"""
app/slices/hematological/nlme.py  L3 NLME Population Layer - Hematological V3.0

Personalises D=2 kinetic parameters per individual via NumPyro NLME.

Non-centred parametrisation (Matt trick):
    eta_raw_i ~ N(0, I_D)
    eta_i     = L_Omega . eta_raw_i
    theta_i   = theta_pop * exp(eta_i)   [log-normal; always positive]

Personalised parameters (D=2):
    k_epo_rbc  -- marrow sensitivity to EPO (erythropoiesis rate)
                  Prior: LogN(log(0.05), 0.35^2)
    k_pv_exp   -- plasma volume elasticity to mechanical load
                  Prior: LogN(log(0.01), 0.40^2)

Fail-Loud: RuntimeError on NaN ELBO or missing dependencies.
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

from app.slices.hematological.ode import (
    HematologicalParams, DEFAULT_HEM_PARAMS,
    STATE_DIM, OBS_DIM,
    X0_HEM_DEFAULT, integrate_1h,
)
from app.slices.hematological.observation import (
    HemObsParams, DEFAULT_HEM_OBS_PARAMS,
    h_hem,
)

_LOG = logging.getLogger(__name__)

D = 2
PARAM_NAMES = ("k_epo_rbc", "k_pv_exp")


# ── Prior hyperparameters ──────────────────────────────────────────────────────

class HemNLMEPriors(NamedTuple):
    log_epo_rbc_mean: float = math.log(0.05)   # log(0.05)
    log_epo_rbc_sd:   float = 0.35
    log_pv_exp_mean:  float = math.log(0.01)   # log(0.01)
    log_pv_exp_sd:    float = 0.40


DEFAULT_HEM_NLME_PRIORS = HemNLMEPriors()


# ── Individual posterior container ────────────────────────────────────────────

class HemIndividualParams(NamedTuple):
    k_epo_rbc:    float
    k_pv_exp:     float
    k_epo_rbc_sd: float = float("nan")
    k_pv_exp_sd:  float = float("nan")


def params_from_individual(ind: HemIndividualParams) -> HematologicalParams:
    return DEFAULT_HEM_PARAMS._replace(
        k_epo_rbc=ind.k_epo_rbc,
        k_pv_exp=ind.k_pv_exp,
    )


# ── Differentiable forward pass ───────────────────────────────────────────────

def _predict_one_subject(
    theta_raw:    jnp.ndarray,
    x0:           jnp.ndarray,
    u_sequence:   jnp.ndarray,
    priors:       HemNLMEPriors   = DEFAULT_HEM_NLME_PRIORS,
    obs_params:   HemObsParams    = DEFAULT_HEM_OBS_PARAMS,
) -> jnp.ndarray:
    """
    theta_raw in R^2 (non-centred; 0 -> population mean).
    u_sequence: (T, CTRL_DIM) constant or time-varying controls.
    Returns (T, OBS_DIM=3) predictions.
    """
    k_epo_rbc = jnp.exp(priors.log_epo_rbc_mean + priors.log_epo_rbc_sd * theta_raw[0])
    k_pv_exp  = jnp.exp(priors.log_pv_exp_mean  + priors.log_pv_exp_sd  * theta_raw[1])

    params_i = DEFAULT_HEM_PARAMS._replace(k_epo_rbc=k_epo_rbc, k_pv_exp=k_pv_exp)

    def step(x, u_t):
        x_next = integrate_1h(x, params=params_i, u=u_t, t0=0.0)
        y_pred = h_hem(x_next, obs_params=obs_params)
        return x_next, y_pred

    _, y_preds = jax.lax.scan(step, x0, u_sequence)
    return y_preds   # (T, OBS_DIM)


# ── NumPyro model ─────────────────────────────────────────────────────────────

def build_hem_nlme_model(
    n_hours:    int,
    u_sequence: jnp.ndarray,
    priors:     HemNLMEPriors = DEFAULT_HEM_NLME_PRIORS,
    obs_params: HemObsParams  = DEFAULT_HEM_OBS_PARAMS,
):
    """
    Factory: returns a NumPyro model for D=2 hematological NLME.

    model(y_obs, subject_idx, x0s)
      y_obs       : (N_total_obs, OBS_DIM=3)
      subject_idx : (N_total_obs,) int
      x0s         : (N_subjects, STATE_DIM=5)
    """
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro required. pip install numpyro")

    sigma_y = jnp.array([
        obs_params.sigma_Hgb,
        obs_params.sigma_Hct,
        obs_params.sigma_Ferritin,
    ], dtype=jnp.float32)

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
            return _predict_one_subject(
                eta_s, x0_s, u_sequence, priors, obs_params
            )

        y_pred_all  = jax.vmap(_pred_s)((eta_i, x0s))   # (N_subj, T, OBS_DIM)
        y_pred_flat = y_pred_all[subject_idx]             # (N_total_obs, T, OBS_DIM)

        numpyro.sample(
            "y",
            dist.Normal(y_pred_flat, sigma_y[None, None, :]).to_event(2),
            obs=y_obs,
        )

    return model


# ── SVI training ──────────────────────────────────────────────────────────────

class HemNLMEResult(NamedTuple):
    svi_state:        object
    params:           dict
    elbo_history:     jnp.ndarray
    individual_means: jnp.ndarray
    individual_sds:   jnp.ndarray


def fit_hem_nlme(
    y_obs:        jnp.ndarray,
    subject_idx:  jnp.ndarray,
    x0s:          jnp.ndarray,
    n_hours:      int,
    u_sequence:   jnp.ndarray,
    priors:       HemNLMEPriors = DEFAULT_HEM_NLME_PRIORS,
    obs_params:   HemObsParams  = DEFAULT_HEM_OBS_PARAMS,
    n_steps:      int           = 30_000,
    lr:           float         = 1e-3,
    seed:         int           = 42,
) -> HemNLMEResult:
    """
    Fit hematological NLME via SVI (AutoLowRankMultivariateNormal).
    Fail-Loud: RuntimeError on NaN ELBO.
    """
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro required.")
    if not _OPTAX_OK:
        raise RuntimeError("optax required.")

    model = build_hem_nlme_model(n_hours, u_sequence, priors, obs_params)
    guide = AutoLowRankMultivariateNormal(model, rank=5)
    optim = numpyro.optim.optax_to_numpyro(optax.adam(lr))
    svi   = SVI(model, guide, optim, loss=Trace_ELBO())

    rng   = jax.random.PRNGKey(seed)
    state = svi.init(rng, y_obs, subject_idx, x0s)

    elbos = []
    for step in range(n_steps):
        state, loss = svi.update(state, y_obs, subject_idx, x0s)
        if jnp.isnan(loss):
            raise RuntimeError(
                f"HemNLME: NaN ELBO at step {step}. "
                "Check priors, controls, or initial state."
            )
        if step % 5000 == 0:
            _LOG.info("HemNLME SVI step %d  ELBO=%.2f", step, -loss)
        elbos.append(float(-loss))

    params      = svi.get_params(state)
    median      = guide.median(params)
    eta_raw_all = median.get("eta_raw", jnp.zeros((x0s.shape[0], D)))

    return HemNLMEResult(
        svi_state        = state,
        params           = params,
        elbo_history     = jnp.array(elbos),
        individual_means = eta_raw_all,
        individual_sds   = jnp.zeros_like(eta_raw_all),
    )


def cold_start_params(
    priors: HemNLMEPriors = DEFAULT_HEM_NLME_PRIORS,
) -> HemIndividualParams:
    """Population-mean individual parameters (no subject data)."""
    return HemIndividualParams(
        k_epo_rbc = math.exp(priors.log_epo_rbc_mean),
        k_pv_exp  = math.exp(priors.log_pv_exp_mean),
    )
