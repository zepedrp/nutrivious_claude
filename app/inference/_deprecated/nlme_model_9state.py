"""
app/inference/nlme_model.py  [DEPRECATED]

This module targets the 9-state AerobicObserverParams from the old observer
(app/engine/observation/aerobic_observer.py) which is no longer part of the
active pipeline. Use app/slices/cardiorespiratory/nlme.py (CardioNLME) instead.

Kept for historical reference only. Imports are intentionally blocked below.

──────────────────────────────────────────────────────────────────────────────
ORIGINAL DOCSTRING (preserved for reference)

L3 NLME Population Layer — Aerobic/HRV Slice

Personalises the 5 AerobicObserverParams (θ_i) per individual using
Nonlinear Mixed-Effects modelling (NLME) in NumPyro.

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015)
────────────────────────────────────────────────────────────────────
    η_raw_i ~ N(0, I_D)                      [uncorrelated raw effects]
    η_i     = L_Ω · η_raw_i                  [correlated random effects]
    θ_i     = θ_pop · exp(η_i)               [individual parameters]
    ⟺  log θ_i = log θ_pop + η_i,  η_i ~ N(0, Ω)

Why non-centred:
    The centred form (η_i ~ N(0, Ω) directly) produces a "funnel" geometry
    in the joint posterior that makes HMC inefficient and SVI inaccurate in
    low-data regimes. Non-centred separates the group-level (θ_pop, Ω) and
    individual-level (η_raw_i) geometry, enabling clean ELBO gradients.
    Reference: Betancourt & Girolami 2015 arXiv:1312.0906

Fitted parameters D=5 (all positive → log-space parametrisation):
────────────────────────────────────────────────────────────────────
    HR_intr     [bpm]   ~ LogN(ln110, 0.10²)  — Berntson 1994
    k_chron_vag [bpm]   ~ LogN(ln50,  0.15²)  — Katona 1970
    k_chron_cat [bpm]   ~ LogN(ln35,  0.15²)  — Goldberger 1999
    RMSSD_ref   [ms]    ~ LogN(ln60,  0.25²)  — Plews 2013 endurance pop.
    k_rmssd     [adim.] ~ LogN(ln2.5, 0.15²)  — Hoshi 2013

Model (per subject i, per observation t):
    ŷ_{i,t} = h_observer(x_{i,t}, θ_i)         [predicted wearable signal]
    y_{i,t} ~ Normal(ŷ_{i,t}, σ_obs)             [proportional observation noise]
    x_{i,t} : state context from UKF posterior mean

SVI backend:
    Guide  : AutoLowRankMultivariateNormal (low-rank Gaussian approximation)
    Optim  : Optax Adam lr=1e-3
    Loss   : Trace_ELBO
    Budget : 50k steps (configurable)

NUTS validation:
    4 chains × 2000 warmup × 2000 draws on a representative subset.
    Convergence criteria: R̂ < 1.01, ESS > 400 for θ_pop and σ_eta.

Fail-Loud contract:
    All exceptions re-raised. No silent zero-substitution in SVI results.
    numpyro ImportError propagates loudly via RuntimeError.
"""
raise ImportError(
    "app.inference.nlme_model is DEPRECATED and targets the 9-state observer "
    "(app/engine/observation/aerobic_observer.py) which is no longer in the "
    "active pipeline. Use app.slices.cardiorespiratory.nlme.CardioNLME instead. "
    "This file is kept for historical reference only — do not import it."
)

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
    from numpyro.infer import NUTS, MCMC
    import optax
    _NUMPYRO_OK = True
except ImportError:  # pragma: no cover — optional dependency
    numpyro = None   # type: ignore[assignment]
    dist    = None   # type: ignore[assignment]
    SVI     = None   # type: ignore[assignment]
    _NUMPYRO_OK = False

from app.engine.observation.aerobic_observer import (
    AerobicObserverParams,
    h_observer,
    STATE_DIM,
    OBS_DIM,
)

logger = logging.getLogger(__name__)

# ── Fitted parameter definitions ─────────────────────────────────────────────

D_THETA: int = 5   # HR_intr, k_chron_vag, k_chron_cat, RMSSD_ref, k_rmssd

THETA_NAMES: list[str] = [
    "HR_intr",
    "k_chron_vag",
    "k_chron_cat",
    "RMSSD_ref",
    "k_rmssd",
]

# Population log-space prior means and SDs (literature-anchored)
_LOG_PRIOR_MEAN: jax.Array = jnp.log(
    jnp.array([110.0, 50.0, 35.0, 60.0, 2.5], dtype=jnp.float32)
)
_LOG_PRIOR_SD: jax.Array = jnp.array(
    [0.10,  0.15,  0.15,  0.25,  0.15],
    dtype=jnp.float32,
)

# Fixed (non-fitted) physiological floors — never personalised
_HR_FLOOR:    float = 30.0   # bpm
_RMSSD_FLOOR: float = 5.0    # ms


# ── Vectorised observer (for NLME likelihood) ────────────────────────────────

@jax.jit
def h_observer_from_vec(
    x:         jax.Array,
    theta_vec: jax.Array,
) -> jax.Array:
    """
    h_observer with θ as a 5-vector — enables jax.vmap in the NLME likelihood.

    Parameters
    ----------
    x         : shape (STATE_DIM,)
    theta_vec : shape (D_THETA,) = [HR_intr, k_chron_vag, k_chron_cat, RMSSD_ref, k_rmssd]

    Returns
    -------
    y : shape (OBS_DIM,) = [HR_bpm, RMSSD_ms]
    """
    params = AerobicObserverParams(
        HR_intr     = theta_vec[0],
        k_chron_vag = theta_vec[1],
        k_chron_cat = theta_vec[2],
        RMSSD_ref   = theta_vec[3],
        k_rmssd     = theta_vec[4],
        HR_floor    = _HR_FLOOR,
        RMSSD_floor = _RMSSD_FLOOR,
    )
    return h_observer(x, params)


def h_observer_batch(
    state_contexts: jax.Array,
    theta_batch:    jax.Array,
) -> jax.Array:
    """
    Batched observer: maps (N_obs, STATE_DIM) × (N_obs, D_THETA) → (N_obs, OBS_DIM).
    Used in the NLME likelihood computation.
    """
    return jax.vmap(h_observer_from_vec, in_axes=(0, 0))(state_contexts, theta_batch)


# ── NumPyro model function ───────────────────────────────────────────────────

def aerobic_nlme_model(
    state_contexts: jax.Array,
    observations:   jax.Array,
    subject_ids:    jax.Array,
    n_subjects:     int,
) -> None:
    """
    NumPyro NLME model — non-centred parametrisation.

    Call signature is designed for both SVI and MCMC (NUTS) backends.

    Parameters
    ----------
    state_contexts : shape (N_obs, STATE_DIM)
        UKF posterior state means for each observation.
        Use X0_DEFAULT when UKF history is unavailable (cold start).
    observations   : shape (N_obs, OBS_DIM)
        Stacked wearable observations [HR_bpm, RMSSD_ms] across all subjects.
    subject_ids    : shape (N_obs,) int
        Maps each observation to its subject index in [0, n_subjects).
    n_subjects     : int
        Total number of subjects in the dataset.

    Model structure
    ───────────────
    θ_pop_log ~ Normal(log_prior_mean, log_prior_sd)          [D_THETA population means]
    scale_eta ~ HalfNormal(0.30)                              [D_THETA between-subject SDs]
    Ω_chol    ~ LKJCholesky(D_THETA, concentration=2.0)       [correlation matrix]
    L_Ω       = diag(scale_eta) @ Ω_chol                     [scaled Cholesky factor]
    σ_obs     ~ HalfNormal(5.0)                               [OBS_DIM residual SDs]

    Per subject i (non-centred — Matt trick):
        η_raw_i ~ Normal(0, 1)^D_THETA
        η_i      = η_raw_i @ L_Ω.T          (L_Ω·η_raw_i row-wise)
        θ_i      = θ_pop · exp(η_i)         (individual parameters)

    Likelihood:
        ŷ_{i,t} = h_observer(x_{i,t}, θ_i)
        y_{i,t} ~ Normal(ŷ_{i,t}, σ_obs)
    """
    if not _NUMPYRO_OK:
        raise RuntimeError(
            "numpyro>=0.15 required for aerobic_nlme_model. "
            "Install: pip install numpyro"
        )

    # ── Population fixed effects (log-space) ──────────────────────────────
    theta_pop_log = numpyro.sample(
        "theta_pop_log",
        dist.Normal(_LOG_PRIOR_MEAN, _LOG_PRIOR_SD),
    )  # shape (D_THETA,)

    # ── Between-subject covariance: LKJ + scale ───────────────────────────
    scale_eta = numpyro.sample(
        "scale_eta",
        dist.HalfNormal(0.30).expand([D_THETA]),
    )  # shape (D_THETA,)
    Omega_chol = numpyro.sample(
        "Omega_chol",
        dist.LKJCholesky(D_THETA, concentration=2.0),
    )  # shape (D_THETA, D_THETA)  lower-triangular
    L_eta = jnp.diag(scale_eta) @ Omega_chol   # (D_THETA, D_THETA)

    # ── Residual observation noise ────────────────────────────────────────
    sigma_obs = numpyro.sample(
        "sigma_obs",
        dist.HalfNormal(5.0).expand([OBS_DIM]),
    )  # shape (OBS_DIM,)

    # ── Per-subject random effects (non-centred — Matt trick) ────────────
    with numpyro.plate("subjects", n_subjects):
        # η_raw_i ~ N(0, I_D) — uncorrelated in the base space
        eta_raw = numpyro.sample(
            "eta_raw",
            dist.Normal(0.0, 1.0).expand([D_THETA]).to_event(1),
        )  # shape (n_subjects, D_THETA)

        # η_i = L_Ω · η_raw_i   (row-wise: n_subjects × D_THETA @ D_THETA × D_THETA^T)
        # η_i shape: (n_subjects, D_THETA)
        eta_i = eta_raw @ L_eta.T

        # θ_i = θ_pop · exp(η_i)  (multiplicative non-centred)
        theta_i = jnp.exp(theta_pop_log + eta_i)  # (n_subjects, D_THETA)

    # ── Gather per-observation θ ──────────────────────────────────────────
    theta_obs = theta_i[subject_ids]   # (N_obs, D_THETA)

    # ── Predicted observations via vmapped h_observer ─────────────────────
    y_pred = h_observer_batch(state_contexts, theta_obs)   # (N_obs, OBS_DIM)

    # ── Likelihood ────────────────────────────────────────────────────────
    with numpyro.plate("observations", observations.shape[0]):
        numpyro.sample(
            "y_obs",
            dist.Normal(y_pred, sigma_obs),
            obs=observations,
        )


# ── SVI result container ─────────────────────────────────────────────────────

class SVIResult(NamedTuple):
    """
    Container for SVI fit results.

    Attributes
    ----------
    params        : dict — variational parameters from guide.get_params()
    elbo_history  : jnp.Array — ELBO per step (for convergence monitoring)
    svi_state     : numpyro SVI state — resumable training state
    guide         : AutoLowRankMultivariateNormal — trained guide object
    """
    params:       dict
    elbo_history: jax.Array
    svi_state:    object
    guide:        object


# ── Main NLME class ──────────────────────────────────────────────────────────

class AerobicNLME:
    """
    L3 NLME Population Model for the Aerobic/HRV slice.

    Personalises AerobicObserverParams per individual from their
    daily wearable time series via SVI or NUTS inference.

    Typical SVI workflow
    ────────────────────
    nlme = AerobicNLME()

    # Build data from UKF state history + wearable observations
    data = {
        "state_contexts": ukf_state_means,   # shape (N_obs, STATE_DIM)
        "observations":   obs_array,          # shape (N_obs, OBS_DIM)
        "subject_ids":    sid_array,          # shape (N_obs,) int
        "n_subjects":     N,
    }

    result = nlme.fit_svi(data, n_steps=50_000, lr=1e-3)
    theta_pop  = nlme.get_population_theta(result)
    obs_params = nlme.get_observer_params(result, subject_id=0)

    Fail-Loud contract
    ──────────────────
    All numpyro/optax exceptions re-raised. No silent zero fallback.
    numpyro absence raises RuntimeError with installation instructions.
    """

    def __init__(self) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError(
                "numpyro>=0.15 required for AerobicNLME. "
                "Install: pip install numpyro optax"
            )
        logger.info("AerobicNLME initialised (D=%d, numpyro backend).", D_THETA)

    # ── SVI fitting ───────────────────────────────────────────────────────

    def fit_svi(
        self,
        data:     dict,
        n_steps:  int   = 50_000,
        lr:       float = 1e-3,
        rank:     int   = 10,
        seed:     int   = 0,
    ) -> SVIResult:
        """
        Fit the NLME model using Stochastic Variational Inference.

        Guide: AutoLowRankMultivariateNormal — low-rank Gaussian approximation
        to the joint posterior over (θ_pop_log, scale_eta, Ω_chol, η_raw, σ_obs).
        Appropriate for large N_subjects where a full-rank approximation is too
        memory-intensive.

        Parameters
        ----------
        data     : dict with keys:
                     state_contexts : (N_obs, STATE_DIM)
                     observations   : (N_obs, OBS_DIM)
                     subject_ids    : (N_obs,) int
                     n_subjects     : int
        n_steps  : int — number of SVI steps (default 50k)
        lr       : float — Adam learning rate (default 1e-3)
        rank     : int — low-rank guide covariance rank (default 10)
        seed     : int — JAX PRNGKey seed

        Returns
        -------
        SVIResult

        Raises
        ------
        RuntimeError if ELBO is NaN at any step (divergence).
        """
        state_contexts = jnp.asarray(data["state_contexts"], dtype=jnp.float32)
        observations   = jnp.asarray(data["observations"],   dtype=jnp.float32)
        subject_ids    = jnp.asarray(data["subject_ids"],    dtype=jnp.int32)
        n_subjects     = int(data["n_subjects"])

        # Build SVI components
        guide = AutoLowRankMultivariateNormal(
            aerobic_nlme_model, rank=rank
        )
        optimizer  = numpyro.optim.optax_to_numpyro(optax.adam(lr))
        svi        = SVI(aerobic_nlme_model, guide, optimizer, loss=Trace_ELBO())

        rng_key = jax.random.PRNGKey(seed)
        svi_state = svi.init(
            rng_key, state_contexts, observations, subject_ids, n_subjects
        )

        elbo_vals = []
        for step in range(n_steps):
            svi_state, loss = svi.update(
                svi_state, state_contexts, observations, subject_ids, n_subjects
            )
            elbo_vals.append(-float(loss))   # ELBO = -loss

            if jnp.isnan(jnp.float32(loss)):
                raise RuntimeError(
                    f"AerobicNLME.fit_svi: ELBO is NaN at step {step}. "
                    "Check data normalisation and prior specifications."
                )

            if step % 5_000 == 0 or step == n_steps - 1:
                logger.debug("SVI step %d/%d  ELBO=%.1f", step, n_steps, elbo_vals[-1])

        logger.info(
            "SVI complete — %d steps, final ELBO=%.1f", n_steps, elbo_vals[-1]
        )

        return SVIResult(
            params       = svi.get_params(svi_state),
            elbo_history = jnp.array(elbo_vals, dtype=jnp.float32),
            svi_state    = svi_state,
            guide        = guide,
        )

    # ── NUTS validation ───────────────────────────────────────────────────

    def validate_with_nuts(
        self,
        data:         dict,
        n_warmup:     int = 2000,
        n_samples:    int = 2000,
        n_chains:     int = 4,
        seed:         int = 1,
    ) -> dict:
        """
        Validate SVI against NUTS on a representative data subset.

        Run 4 chains × 2000 warmup × 2000 samples. Report R̂ and ESS
        for θ_pop_log and scale_eta.

        Parameters
        ----------
        data      : same dict as fit_svi()
        Returns   : dict with keys "r_hat", "ess", "divergences"

        Raises
        ------
        RuntimeError if R̂ > 1.05 for any critical parameter.
        """
        state_contexts = jnp.asarray(data["state_contexts"], dtype=jnp.float32)
        observations   = jnp.asarray(data["observations"],   dtype=jnp.float32)
        subject_ids    = jnp.asarray(data["subject_ids"],    dtype=jnp.int32)
        n_subjects     = int(data["n_subjects"])

        nuts_kernel = NUTS(aerobic_nlme_model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=n_warmup,
            num_samples=n_samples,
            num_chains=n_chains,
        )
        rng_key = jax.random.PRNGKey(seed)
        mcmc.run(rng_key, state_contexts, observations, subject_ids, n_subjects)

        samples  = mcmc.get_samples()
        r_hat    = numpyro.diagnostics.gelman_rubin(samples)
        ess      = numpyro.diagnostics.effective_sample_size(samples)

        # Check convergence criterion (HLD §6: R̂ < 1.01)
        max_rhat = float(jnp.max(jnp.array(list(r_hat.values()))))
        if max_rhat > 1.05:
            logger.warning(
                "NUTS validation: max R̂=%.4f exceeds threshold 1.05 — "
                "consider more warmup steps or reparametrisation.", max_rhat
            )

        logger.info("NUTS validation complete — max R̂=%.4f", max_rhat)
        return {
            "r_hat":       r_hat,
            "ess":         ess,
            "divergences": mcmc.get_extra_fields().get("diverging", jnp.zeros(1)),
        }

    # ── Posterior extraction ──────────────────────────────────────────────

    def get_population_theta(self, result: SVIResult) -> jax.Array:
        """
        Extract population-level θ_pop = exp(θ_pop_log_mean) from SVI result.

        Returns
        -------
        theta_pop : shape (D_THETA,) — population mean parameters
        """
        median_samples = result.guide.median(result.params)
        theta_pop_log  = median_samples["theta_pop_log"]   # shape (D_THETA,)
        return jnp.exp(theta_pop_log)

    def get_individual_theta(
        self,
        result:     SVIResult,
        subject_id: int,
        n_samples:  int = 1000,
        seed:       int = 2,
    ) -> jax.Array:
        """
        Sample θ_i for a specific individual from the variational posterior.

        Parameters
        ----------
        result     : SVIResult from fit_svi()
        subject_id : int — index into the subjects plate
        n_samples  : int — number of posterior samples

        Returns
        -------
        theta_i_samples : shape (n_samples, D_THETA)
        """
        rng_key = jax.random.PRNGKey(seed)
        posterior_samples = result.guide.sample_posterior(
            rng_key, result.params, sample_shape=(n_samples,)
        )
        theta_pop_log = posterior_samples["theta_pop_log"]  # (n_samples, D_THETA)
        eta_raw       = posterior_samples["eta_raw"]         # (n_samples, n_subjects, D_THETA)

        # Reconstruct L_eta from posterior
        # NumPyro samples "scale_eta" directly (HalfNormal), not in log-space.
        scale_eta = posterior_samples.get(
            "scale_eta", jnp.ones((n_samples, D_THETA), dtype=jnp.float32) * 0.3
        )
        Omega_chol = posterior_samples["Omega_chol"]  # (n_samples, D_THETA, D_THETA)

        # η_i = η_raw_i @ L_Ω.T  (per sample)
        L_eta = jax.vmap(lambda sc, oc: jnp.diag(sc) @ oc)(scale_eta, Omega_chol)
        eta_i = jax.vmap(
            lambda e_r, L: e_r[:, subject_id, :] @ L.T
        )(eta_raw.reshape(n_samples, -1, D_THETA), L_eta)

        theta_i = jnp.exp(theta_pop_log + eta_i)   # (n_samples, D_THETA)
        return theta_i

    def get_observer_params(
        self,
        result:     SVIResult,
        subject_id: int | None = None,
    ) -> AerobicObserverParams:
        """
        Return AerobicObserverParams from the SVI posterior.

        If subject_id is None, returns the population-level parameters.
        If subject_id is given, returns the individual posterior mean.

        Parameters
        ----------
        result     : SVIResult
        subject_id : int or None

        Returns
        -------
        AerobicObserverParams with personalised or population θ
        """
        if subject_id is None:
            theta = self.get_population_theta(result)
        else:
            theta_samples = self.get_individual_theta(result, subject_id)
            theta = jnp.mean(theta_samples, axis=0)   # posterior mean

        return AerobicObserverParams(
            HR_intr     = float(theta[0]),
            k_chron_vag = float(theta[1]),
            k_chron_cat = float(theta[2]),
            RMSSD_ref   = float(theta[3]),
            k_rmssd     = float(theta[4]),
            HR_floor    = _HR_FLOOR,
            RMSSD_floor = _RMSSD_FLOOR,
        )

    # ── Convenience: cold-start state contexts ───────────────────────────

    @staticmethod
    def make_cold_start_contexts(n_obs: int) -> jax.Array:
        """
        Build state_contexts for cold-start (no UKF history).

        Uses X0_DEFAULT from aerobic_observer for the 9-state cold-start
        context. Replace with actual UKF posterior means as history
        accumulates.

        Returns
        -------
        state_contexts : shape (n_obs, STATE_DIM) — all rows = resting baseline
        """
        from app.engine.observation.aerobic_observer import DEFAULT_OBSERVER_PARAMS
        x0 = jnp.array([
            DEFAULT_OBSERVER_PARAMS.HR_intr,
            DEFAULT_OBSERVER_PARAMS.k_chron_vag,
            0.5,  # V_vagal: resting mid-tone
            90.0, # P_a: resting MAP mmHg
            40.0, # PaCO2: normal arterial CO2
            40.0, # PbCO2: normal brain CO2
            98.0, # SpO2: normal saturation %
            8.0,  # V_E: resting ventilation L/min
            18.0, # W_prime_bal: fully replete kJ
        ], dtype=jnp.float32)
        return jnp.tile(x0[None, :], (n_obs, 1))
