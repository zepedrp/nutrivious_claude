"""
app/slices/sleep_circadian/nlme.py  —  L3 NLME Layer V2.0

Personalises D=2 chronobiological parameters via Non-centred NumPyro NLME.

Personalised parameters (D=2)
─────────────────────────────
  tau_c    [h]   intrinsic circadian period  (chronotype: 23.5–24.8 h)
                 Morning type ≈ 23.7 h; Evening type ≈ 24.5 h
  k_clear  [h⁻¹] adenosine clearance rate  (genetic sleep recovery speed)
                 Fast clearer (≈ 0.17 h⁻¹, t½ ≈ 4 h) vs slow (≈ 0.10 h⁻¹, t½ ≈ 7 h)

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015 arXiv:1312.0906):
  η_raw_i ~ N(0, I_D)
  θ_i     = exp(log_θ_pop + L_Ω · η_raw_i)   [log-normal; both params positive]

Population priors
─────────────────
  tau_c   ~ LogN(log 24.18, 0.008²)   h   (SD ≈ 0.20 h on natural scale)
  k_clear ~ LogN(log 0.13,  0.20²)   h⁻¹  (CV ≈ 20%; Achermann 2003)

Identifiability rationale
──────────────────────────
  tau_c   identifiable from ≥14 d of sleep-onset timing (Phillips 2017 Sci Adv)
  k_clear identifiable from adenosine-proxy signals under varying sleep debt

Fail-Loud
──────────
  NaN ELBO → RuntimeError. Missing deps → RuntimeError with install hint.

References
──────────
  Phillips A.J.K. et al. (2017) Sci Adv 3(12):e1700445
  Achermann P. & Borbély A.A. (2003) Front Biosci 8:s683–693
  Betancourt M. & Girolami M. (2015) arXiv:1312.0906
"""
from __future__ import annotations

import logging
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

from app.slices.sleep_circadian.ode import (
    SleepParams,
    DEFAULT_SLEEP_PARAMS,
    HubInputs,
    initial_state,
    simulate_nsteps,
    STATE_DIM,
)
from app.slices.sleep_circadian.observation import (
    SleepObsParams,
    DEFAULT_OBS_PARAMS,
    h_sleep,
    OBS_DIM,
)

_LOG = logging.getLogger(__name__)

# ── Population prior hyperparameters ──────────────────────────────────────────

PARAM_NAMES = ("tau_c", "k_clear")
D = 2

class SleepNLMEPriors(NamedTuple):
    """
    Log-space prior means and SDs for D=2 personalised sleep parameters.

    Source: CLAUDE.md §3 prior table; Forger 1999; Achermann 2003.
    """
    log_tau_c_mean:    float = float(jnp.log(24.18))
    log_tau_c_sd:      float = 0.008   # ≈ 0.20 h SD on natural scale
    log_k_clear_mean:  float = float(jnp.log(0.13))
    log_k_clear_sd:    float = 0.20    # 20% CV (Achermann 2003)


DEFAULT_NLME_PRIORS = SleepNLMEPriors()


class SleepIndividualParams(NamedTuple):
    """Individual NLME posterior for one subject."""
    tau_c:       float
    k_clear:     float
    tau_c_sd:    float = float("nan")
    k_clear_sd:  float = float("nan")


def params_from_individual(ind: SleepIndividualParams) -> SleepParams:
    """Build SleepParams from individual NLME posterior (fixes population params)."""
    base = DEFAULT_SLEEP_PARAMS
    return SleepParams(
        tau_c    = ind.tau_c,
        k_clear  = ind.k_clear,
        mu_c     = base.mu_c,
        alpha_c  = base.alpha_c,
        kappa_c  = base.kappa_c,
        G_phot   = base.G_phot,
        r_acc    = base.r_acc,
        k_stress = base.k_stress,
        k_sec    = base.k_sec,
        k_mel    = base.k_mel,
        K_cort   = base.K_cort,
        Aden_thr = base.Aden_thr,
        Mel_thr  = base.Mel_thr,
        k_Aden   = base.k_Aden,
        k_Mel    = base.k_Mel,
        tau_sws  = base.tau_sws,
    )


# ── Forward simulation for one subject ────────────────────────────────────────

def _predict_one_subject(
    eta_raw: jnp.ndarray,          # (D,) non-centred random effects
    x0: jnp.ndarray,               # (STATE_DIM,)
    hubs_seq: jnp.ndarray,         # (T, 3)
    priors: SleepNLMEPriors,
    obs_params: SleepObsParams,
) -> jnp.ndarray:                  # (T, OBS_DIM)
    """Decode η_raw → θ_i → simulate → predict observations."""
    tau_c   = jnp.exp(priors.log_tau_c_mean   + priors.log_tau_c_sd   * eta_raw[0])
    k_clear = jnp.exp(priors.log_k_clear_mean + priors.log_k_clear_sd * eta_raw[1])

    params_i = SleepParams(
        tau_c   = tau_c,
        k_clear = k_clear,
        mu_c     = DEFAULT_SLEEP_PARAMS.mu_c,
        alpha_c  = DEFAULT_SLEEP_PARAMS.alpha_c,
        kappa_c  = DEFAULT_SLEEP_PARAMS.kappa_c,
        G_phot   = DEFAULT_SLEEP_PARAMS.G_phot,
        r_acc    = DEFAULT_SLEEP_PARAMS.r_acc,
        k_stress = DEFAULT_SLEEP_PARAMS.k_stress,
        k_sec    = DEFAULT_SLEEP_PARAMS.k_sec,
        k_mel    = DEFAULT_SLEEP_PARAMS.k_mel,
        K_cort   = DEFAULT_SLEEP_PARAMS.K_cort,
        Aden_thr = DEFAULT_SLEEP_PARAMS.Aden_thr,
        Mel_thr  = DEFAULT_SLEEP_PARAMS.Mel_thr,
        k_Aden   = DEFAULT_SLEEP_PARAMS.k_Aden,
        k_Mel    = DEFAULT_SLEEP_PARAMS.k_Mel,
        tau_sws  = DEFAULT_SLEEP_PARAMS.tau_sws,
    )

    xs = simulate_nsteps(x0, hubs_seq, params=params_i)   # (T+1, STATE_DIM)
    xs_obs = xs[1:]                                         # (T, STATE_DIM)
    y_pred = jax.vmap(lambda x: h_sleep(x, obs_params=obs_params))(xs_obs)
    return y_pred   # (T, OBS_DIM)


# ── NumPyro model ──────────────────────────────────────────────────────────────

def build_sleep_nlme_model(
    n_steps: int,
    priors: SleepNLMEPriors = DEFAULT_NLME_PRIORS,
    obs_params: SleepObsParams = DEFAULT_OBS_PARAMS,
):
    """
    Factory returning a NumPyro model for D=2 sleep NLME.

    model(y_obs, subject_idx, x0s, hubs_seq) → None
      y_obs       : (N_total, OBS_DIM)
      subject_idx : (N_total,) int
      x0s         : (N_subjects, STATE_DIM)
      hubs_seq    : (N_subjects, n_steps, 3)
    """
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro required: pip install numpyro")

    sigma_y = jnp.array([obs_params.sigma_sws, obs_params.sigma_phase])

    def model(y_obs, subject_idx, x0s, hubs_seq):
        N_subjects = x0s.shape[0]

        Omega_chol = numpyro.sample(
            "Omega_chol",
            dist.LKJCholesky(D, concentration=2.0),
        )

        with numpyro.plate("subjects", N_subjects):
            eta_raw = numpyro.sample(
                "eta_raw",
                dist.Normal(jnp.zeros(D), jnp.ones(D)),
            )   # (N_subjects, D)
            eta_i = eta_raw @ Omega_chol.T   # (N_subjects, D)

        def _pred(args):
            eta_i_s, x0_s, hubs_s = args
            return _predict_one_subject(eta_i_s, x0_s, hubs_s, priors, obs_params)

        y_pred_all = jax.vmap(_pred)((eta_i, x0s, hubs_seq))  # (N_subj, T, OBS_DIM)
        y_pred_flat = y_pred_all[subject_idx]                   # (N_total, OBS_DIM)

        numpyro.sample(
            "y",
            dist.Normal(y_pred_flat, sigma_y[None, :]).to_event(1),
            obs=y_obs,
        )

    return model


# ── SVI entry point ────────────────────────────────────────────────────────────

class SleepNLMEResult(NamedTuple):
    svi_state:        object
    params:           dict
    elbo_history:     jnp.ndarray
    individual_means: jnp.ndarray   # (N_subjects, D)
    individual_sds:   jnp.ndarray


def fit_sleep_nlme(
    y_obs: jnp.ndarray,
    subject_idx: jnp.ndarray,
    x0s: jnp.ndarray,
    hubs_seq: jnp.ndarray,
    n_steps: int,
    priors: SleepNLMEPriors = DEFAULT_NLME_PRIORS,
    obs_params: SleepObsParams = DEFAULT_OBS_PARAMS,
    n_svi_steps: int = 20_000,
    lr: float = 5e-4,
    seed: int = 42,
) -> SleepNLMEResult:
    """
    Fit sleep NLME via SVI (AutoLowRankMultivariateNormal).
    Fail-Loud: RuntimeError on NaN ELBO or missing dependency.
    """
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro required: pip install numpyro")
    if not _OPTAX_OK:
        raise RuntimeError("optax required: pip install optax")

    model = build_sleep_nlme_model(n_steps, priors, obs_params)
    guide = AutoLowRankMultivariateNormal(model, rank=2)
    optim = numpyro.optim.optax_to_numpyro(optax.adam(lr))
    svi   = SVI(model, guide, optim, loss=Trace_ELBO())

    rng = jax.random.PRNGKey(seed)
    svi_state = svi.init(rng, y_obs, subject_idx, x0s, hubs_seq)

    elbo_hist = []
    for step in range(n_svi_steps):
        svi_state, loss = svi.update(svi_state, y_obs, subject_idx, x0s, hubs_seq)
        elbo_hist.append(-float(loss))
        if jnp.isnan(loss):
            raise RuntimeError(f"SleepNLME: ELBO NaN at step {step}.")
        if step % 5_000 == 0:
            _LOG.info("SleepNLME SVI step %d  ELBO=%.1f", step, -float(loss))

    params      = svi.get_params(svi_state)
    elbo_array  = jnp.array(elbo_hist)
    site_loc    = params.get("eta_raw_loc",   jnp.zeros((x0s.shape[0], D)))
    site_sd     = params.get("eta_raw_scale", jnp.ones((x0s.shape[0], D)) * 0.1)

    return SleepNLMEResult(
        svi_state        = svi_state,
        params           = params,
        elbo_history     = elbo_array,
        individual_means = site_loc,
        individual_sds   = site_sd,
    )


def individual_params_from_nlme(
    result: SleepNLMEResult,
    subject_id: int,
    priors: SleepNLMEPriors = DEFAULT_NLME_PRIORS,
) -> SleepIndividualParams:
    """Decode η_raw posterior for a subject → SleepIndividualParams."""
    eta   = result.individual_means[subject_id]
    eta_s = result.individual_sds[subject_id]

    tau_c   = float(jnp.exp(priors.log_tau_c_mean   + priors.log_tau_c_sd   * eta[0]))
    k_clear = float(jnp.exp(priors.log_k_clear_mean + priors.log_k_clear_sd * eta[1]))

    tau_c_sd   = float(tau_c   * priors.log_tau_c_sd   * eta_s[0])
    k_clear_sd = float(k_clear * priors.log_k_clear_sd * eta_s[1])

    return SleepIndividualParams(
        tau_c=tau_c, k_clear=k_clear,
        tau_c_sd=tau_c_sd, k_clear_sd=k_clear_sd,
    )
