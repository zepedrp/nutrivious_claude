"""
app/slices/cardiorespiratory/nlme.py

L3 NLME Population Layer — Cardiorespiratory Slice

Personalises 2 kinetic parameters per individual (θ_i) from wearable HR
and/or VO2 time-series data using Nonlinear Mixed-Effects modelling (NLME)
in NumPyro.

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015)
────────────────────────────────────────────────────────────────────
    η_raw_i ~ N(0, I_D)                  [uncorrelated base effects]
    η_i     = L_Ω · η_raw_i             [correlated random effects]
    θ_i     = θ_pop · exp(η_i)          [individual parameters — log-space]

Personalised parameters (D = 2)
────────────────────────────────
    VO2_max_baseline [mL/kg/min] — individual peak aerobic capacity
    W_prime_capacity [kJ]        — individual anaerobic battery capacity

Identifiability rationale
──────────────────────────
These two parameters are the primary determinants of the HR response profile
to a graded exercise test and the depletion / recovery kinetics during
supramaximal efforts (Burnley & Jones 2018 review):
  • VO2_max_baseline — controls the HR-VO2 relationship plateau (Fick principle)
  • W_prime_capacity — controls how long above-CP effort can be sustained before
                       autonomic withdrawal and HR drift appear

All other ODE parameters remain fixed at population priors
(CLAUDE.md §3: "non-identifiable parameters are fixed at the prior").

Population priors (literature-anchored, log-space)
───────────────────────────────────────────────────
    VO2_max ~ LogN(log 45, 0.22²)  — mean 45 mL/kg/min, CV ≈ 22%
              (Bouchard 1999 HERITAGE; σ ≈ 10 mL/kg/min)
    W'      ~ LogN(log 18, 0.28²)  — mean 18 kJ, CV ≈ 28%
              (Skiba 2015 J Sci Sport; σ ≈ 5 kJ)

Forward simulation
──────────────────
The NLME likelihood requires simulating the HR response to a power profile.
Uses jax.lax.scan over T time steps × dt_min=1 min, calling diffrax.Tsit5
per step. Fully JAX-traceable — supports SVI gradient estimation and NUTS.

SVI backend
───────────
Guide  : AutoLowRankMultivariateNormal (low-rank Gaussian approximation)
Optim  : Optax Adam lr=1e-3
Loss   : Trace_ELBO
Budget : 20 000 steps (typical convergence for D=2 cardio NLME)

Fail-Loud contract
──────────────────
All numpyro/optax exceptions re-raised. ELBO NaN → RuntimeError (divergence).
numpyro absence → RuntimeError with installation instructions.

References
──────────
  Burnley M. & Jones A.M. (2018) Int J Sports Physiol Perform 13:279-289
    DOI 10.1123/ijspp.2017-0415  [VO2max and W' identifiability]
  Skiba P.F. et al. (2015) J Sci Sport 20:486-492
    DOI 10.1016/j.jsams.2014.07.004  [W' population distribution]
  Bouchard C. et al. (1999) Med Sci Sports Exerc 31:805-814 (HERITAGE)
    DOI 10.1097/00005768-199906000-00003  [VO2max heritability + population CV]
  Betancourt M. & Girolami M. (2015) arXiv:1312.0906 [non-centred parametrisation]
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    from numpyro.infer import NUTS, MCMC
    import optax
    _NUMPYRO_OK = True
except ImportError:
    numpyro = None      # type: ignore[assignment]
    dist    = None      # type: ignore[assignment]
    SVI     = None      # type: ignore[assignment]
    _NUMPYRO_OK = False

from app.slices.cardiorespiratory.ode import (
    DEFAULT_CARDIO_SLICE_PARAMS,
    X0_CARDIO_DEFAULT,
    STATE_DIM,
    IDX_HR,
    IDX_VO2,
    cardiorespiratory_slice_ode,
    CardioSliceParams,
)

logger = logging.getLogger(__name__)


# ── Fitted parameter definitions ─────────────────────────────────────────────

D_THETA:     int        = 2
THETA_NAMES: list[str]  = ["VO2_max_baseline", "W_prime_capacity"]

# Population log-space prior means (literature anchored)
_LOG_PRIOR_MEAN: jax.Array = jnp.log(
    jnp.array([45.0, 18.0], dtype=jnp.float32)   # [mL/kg/min, kJ]
)
# Population log-space prior SDs  (CV of 22% and 28% respectively)
_LOG_PRIOR_SD: jax.Array = jnp.array(
    [0.22, 0.28], dtype=jnp.float32
)


# ── Forward simulation (JIT-traceable, vmap-safe) ─────────────────────────────

def _build_params_from_log_theta(
    log_theta_i: jax.Array,
    base: CardioSliceParams = DEFAULT_CARDIO_SLICE_PARAMS,
) -> CardioSliceParams:
    """
    Build CardioSliceParams with VO2_max_baseline and W_prime_capacity
    replaced by exp(log_theta_i[0]) and exp(log_theta_i[1]).
    All other fields inherited from `base`.
    """
    theta_i = jnp.exp(log_theta_i)
    return base._replace(
        VO2_max_baseline = theta_i[0],
        W_prime_capacity = theta_i[1],
    )


@jax.jit
def simulate_hr_trajectory(
    log_theta_i:  jax.Array,   # (D_THETA,): log [VO2_max, W_prime]
    x0:           jax.Array,   # (STATE_DIM,): initial state
    power_rates:  jax.Array,   # (T,): power at each 1-min step [W]
    dt_min:       float = 1.0,
) -> jax.Array:                # (T,): predicted HR [bpm]
    """
    Simulate the HR response to a power time series for one subject.

    Integrates the cardiorespiratory ODE step-by-step over T × dt_min
    minutes using jax.lax.scan + diffrax.Tsit5. JAX-traceable for SVI.

    Parameters
    ----------
    log_theta_i : (D_THETA,) — log [VO2_max_baseline, W_prime_capacity]
    x0          : (STATE_DIM,) — initial state (e.g. X0_CARDIO_DEFAULT)
    power_rates : (T,) — power demand [W] at each 1-min step
    dt_min      : float — integration step [min]; default 1.0

    Returns
    -------
    HR_traj : (T,) — predicted Heart_Rate [bpm] at each step
    """
    params_i = _build_params_from_log_theta(log_theta_i)

    def _step(
        x: jax.Array,
        power_t: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        sol = diffrax.diffeqsolve(
            terms   = diffrax.ODETerm(cardiorespiratory_slice_ode),
            solver  = diffrax.Tsit5(),
            t0      = jnp.float32(0.0),
            t1      = jnp.float32(dt_min),
            dt0     = jnp.float32(0.1),
            y0      = x,
            args    = (params_i, power_t, jnp.float32(37.0), jnp.float32(0.0)),
            saveat  = diffrax.SaveAt(t1=True),
            max_steps = 32,
        )
        x_next = sol.ys[0]
        return x_next, x_next[IDX_HR]

    _, HR_traj = jax.lax.scan(_step, x0, power_rates)
    return HR_traj   # shape (T,)


# ── NumPyro NLME model ────────────────────────────────────────────────────────

def cardio_nlme_model(
    session_data: dict,
    observations: jax.Array,   # (N_obs,) — HR readings [bpm]
    subject_ids:  jax.Array,   # (N_obs,) int — maps obs to subject
    obs_step_ids: jax.Array,   # (N_obs,) int — maps obs to time step
    n_subjects:   int,
) -> None:
    """
    NumPyro NLME model for cardiorespiratory kinetics — non-centred parametrisation.

    Data contract (session_data dict)
    ──────────────────────────────────
    "power_rates" : (n_subjects, T) float32 — power [W] at each 1-min step
    "x0s"         : (n_subjects, STATE_DIM) — initial states per subject

    Parameters
    ----------
    session_data  : dict — see above
    observations  : (N_obs,) — HR readings [bpm]; NaN accepted (masked below)
    subject_ids   : (N_obs,) int — subject index for each observation
    obs_step_ids  : (N_obs,) int — time-step index (0..T-1) for each observation
    n_subjects    : int — fleet size

    Model
    ─────
    θ_pop_log   ~ N(log_prior_mean, log_prior_sd)         [D_THETA pop means]
    scale_eta   ~ HalfNormal(0.25)^D_THETA               [between-subject SDs]
    Ω_chol      ~ LKJCholesky(D_THETA, concentration=2.0) [correlation]
    L_Ω         = diag(scale_eta) @ Ω_chol
    σ_obs       ~ HalfNormal(5.0)                         [HR residual SD, bpm]

    Per subject i (non-centred Matt trick):
        η_raw_i ~ N(0, I_D)
        η_i      = η_raw_i @ L_Ω.T
        θ_i      = exp(θ_pop_log + η_i)
        HR_traj  = simulate_hr_trajectory(log(θ_i), x0_i, power_rates_i)

    Likelihood:
        y_t ~ N(HR_traj_i[t], σ_obs)
    """
    if not _NUMPYRO_OK:
        raise RuntimeError(
            "numpyro>=0.15 required for cardio_nlme_model. "
            "Install: pip install numpyro"
        )

    # ── Population fixed effects ──────────────────────────────────────────
    theta_pop_log = numpyro.sample(
        "theta_pop_log",
        dist.Normal(_LOG_PRIOR_MEAN, _LOG_PRIOR_SD),
    )   # (D_THETA,)

    # ── Between-subject covariance (LKJ + scale) ──────────────────────────
    scale_eta = numpyro.sample(
        "scale_eta",
        dist.HalfNormal(0.25).expand([D_THETA]),
    )   # (D_THETA,)
    Omega_chol = numpyro.sample(
        "Omega_chol",
        dist.LKJCholesky(D_THETA, concentration=2.0),
    )   # (D_THETA, D_THETA)
    L_eta = jnp.diag(scale_eta) @ Omega_chol   # (D_THETA, D_THETA)

    # ── Residual observation noise ─────────────────────────────────────────
    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(5.0))  # bpm

    # ── Per-subject random effects (non-centred) ──────────────────────────
    with numpyro.plate("subjects", n_subjects):
        eta_raw = numpyro.sample(
            "eta_raw",
            dist.Normal(0.0, 1.0).expand([D_THETA]).to_event(1),
        )   # (n_subjects, D_THETA)
        eta_i       = eta_raw @ L_eta.T                 # (n_subjects, D_THETA)
        log_theta_i = theta_pop_log + eta_i             # (n_subjects, D_THETA)

    # ── Forward simulation for every subject (vmapped) ────────────────────
    HR_trajs = jax.vmap(
        simulate_hr_trajectory,
        in_axes=(0, 0, 0, None),
    )(
        log_theta_i,
        jnp.asarray(session_data["x0s"],         dtype=jnp.float32),
        jnp.asarray(session_data["power_rates"], dtype=jnp.float32),
    )   # (n_subjects, T)

    # ── Gather per-observation predictions ───────────────────────────────
    y_pred = HR_trajs[subject_ids, obs_step_ids]    # (N_obs,)

    # ── NaN mask: skip observations where HR was missing ─────────────────
    obs_valid = ~jnp.isnan(observations)
    sigma_eff = jnp.where(obs_valid, sigma_obs, jnp.float32(1e4))
    y_safe    = jnp.where(obs_valid, observations, y_pred)

    # ── Likelihood ────────────────────────────────────────────────────────
    with numpyro.plate("observations", observations.shape[0]):
        numpyro.sample(
            "y_obs",
            dist.Normal(y_pred, sigma_eff),
            obs=y_safe,
        )


# ── SVI result container ─────────────────────────────────────────────────────

class CardioSVIResult(NamedTuple):
    """Container for cardio NLME SVI fit results."""
    params:       dict
    elbo_history: jax.Array
    svi_state:    object
    guide:        object


# ── Main NLME class ───────────────────────────────────────────────────────────

class CardioNLME:
    """
    L3 NLME Population Model for the Cardiorespiratory slice.

    Personalises VO2_max_baseline and W_prime_capacity per individual
    from HR (and optionally VO2) response data via SVI or NUTS.

    Typical SVI workflow
    ────────────────────
    nlme = CardioNLME()

    result = nlme.fit_svi(
        session_data  = {"power_rates": ..., "x0s": ...},
        observations  = hr_obs,       # (N_obs,) bpm; NaN for missing
        subject_ids   = sid_arr,      # (N_obs,) int
        obs_step_ids  = step_arr,     # (N_obs,) int
        n_subjects    = N,
    )

    theta_pop   = nlme.get_population_theta(result)   # (2,): [VO2_max, W_prime]
    params_i    = nlme.get_cardio_params(result, sid=0)  # personalised CardioSliceParams

    Fail-Loud contract
    ──────────────────
    All exceptions re-raised. ELBO NaN → RuntimeError.
    numpyro absence → RuntimeError with installation instructions.
    """

    def __init__(self) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError(
                "numpyro>=0.15 required for CardioNLME. "
                "Install: pip install numpyro optax"
            )
        logger.info("CardioNLME initialised (D=%d: %s).", D_THETA, THETA_NAMES)

    # ── SVI fitting ───────────────────────────────────────────────────────

    def fit_svi(
        self,
        session_data:  dict,
        observations:  jax.Array,
        subject_ids:   jax.Array,
        obs_step_ids:  jax.Array,
        n_subjects:    int,
        n_steps:       int   = 20_000,
        lr:            float = 1e-3,
        rank:          int   = 4,
        seed:          int   = 0,
    ) -> CardioSVIResult:
        """
        Fit the cardio NLME model using Stochastic Variational Inference.

        Guide: AutoLowRankMultivariateNormal (rank-4 approximation to the
        joint posterior over θ_pop_log, scale_eta, Ω_chol, η_raw, σ_obs).

        Parameters
        ----------
        session_data  : dict — {"power_rates": (N, T), "x0s": (N, STATE_DIM)}
        observations  : (N_obs,) — HR readings [bpm]; NaN for missing
        subject_ids   : (N_obs,) int
        obs_step_ids  : (N_obs,) int
        n_subjects    : int
        n_steps       : int — SVI iterations (default 20k)
        lr            : float — Adam learning rate
        rank          : int — guide covariance rank
        seed          : int — JAX PRNGKey seed

        Returns
        -------
        CardioSVIResult

        Raises
        ------
        RuntimeError if ELBO is NaN at any step.
        """
        obs  = jnp.asarray(observations, dtype=jnp.float32)
        sids = jnp.asarray(subject_ids,  dtype=jnp.int32)
        tids = jnp.asarray(obs_step_ids, dtype=jnp.int32)

        guide     = AutoLowRankMultivariateNormal(cardio_nlme_model, rank=rank)
        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(lr))
        svi       = SVI(cardio_nlme_model, guide, optimizer, loss=Trace_ELBO())

        rng_key   = jax.random.PRNGKey(seed)
        svi_state = svi.init(rng_key, session_data, obs, sids, tids, n_subjects)

        elbo_vals: list[float] = []
        for step in range(n_steps):
            svi_state, loss = svi.update(
                svi_state, session_data, obs, sids, tids, n_subjects
            )
            elbo_vals.append(-float(loss))
            if jnp.isnan(jnp.float32(loss)):
                raise RuntimeError(
                    f"CardioNLME.fit_svi: ELBO is NaN at step {step}. "
                    "Check power_rates normalisation and HR data quality."
                )
            if step % 5_000 == 0 or step == n_steps - 1:
                logger.debug("SVI step %d/%d  ELBO=%.1f", step, n_steps, elbo_vals[-1])

        logger.info(
            "CardioNLME SVI complete — %d steps, final ELBO=%.1f",
            n_steps, elbo_vals[-1],
        )
        return CardioSVIResult(
            params       = svi.get_params(svi_state),
            elbo_history = jnp.array(elbo_vals, dtype=jnp.float32),
            svi_state    = svi_state,
            guide        = guide,
        )

    # ── Posterior extraction ──────────────────────────────────────────────

    def get_population_theta(self, result: CardioSVIResult) -> jax.Array:
        """
        Extract population-level θ_pop = exp(θ_pop_log_median).

        Returns
        -------
        theta_pop : (D_THETA,) = [VO2_max_baseline_pop, W_prime_capacity_pop]
        """
        median_samples = result.guide.median(result.params)
        log_theta_pop  = median_samples["theta_pop_log"]
        return jnp.exp(log_theta_pop)

    def get_individual_log_theta(
        self,
        result:     CardioSVIResult,
        subject_id: int,
        n_samples:  int = 500,
        seed:       int = 1,
    ) -> jax.Array:
        """
        Sample log θ_i for subject from the variational posterior.

        Returns
        -------
        log_theta_samples : (n_samples, D_THETA)
        """
        rng_key  = jax.random.PRNGKey(seed)
        posterior = result.guide.sample_posterior(
            rng_key, result.params, sample_shape=(n_samples,)
        )
        theta_pop_log = posterior["theta_pop_log"]   # (n_samples, D_THETA)
        eta_raw       = posterior["eta_raw"]          # (n_samples, n_subjects, D_THETA)
        scale_eta     = posterior["scale_eta"]        # (n_samples, D_THETA)
        Omega_chol    = posterior["Omega_chol"]       # (n_samples, D_THETA, D_THETA)

        L_eta  = jax.vmap(lambda sc, oc: jnp.diag(sc) @ oc)(scale_eta, Omega_chol)
        eta_i  = jax.vmap(
            lambda e, L: e[:, subject_id, :] @ L.T
        )(eta_raw, L_eta)
        return theta_pop_log + eta_i   # (n_samples, D_THETA)

    def get_cardio_params(
        self,
        result:     CardioSVIResult,
        subject_id: int,
        base:       CardioSliceParams = DEFAULT_CARDIO_SLICE_PARAMS,
    ) -> CardioSliceParams:
        """
        Return CardioSliceParams with VO2_max_baseline and W_prime_capacity
        set to the posterior median for the given subject.
        All other parameters inherited from `base`.
        """
        log_theta_samples = self.get_individual_log_theta(result, subject_id)
        log_theta_median  = jnp.median(log_theta_samples, axis=0)   # (D_THETA,)
        theta_median      = jnp.exp(log_theta_median)
        return base._replace(
            VO2_max_baseline = float(theta_median[0]),
            W_prime_capacity = float(theta_median[1]),
        )
