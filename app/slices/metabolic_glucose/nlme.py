"""
app/slices/metabolic_glucose/nlme.py -- NLME Population Layer V2.0

Personalises D=2 kinetic parameters per individual from CGM meal-challenge data.

Identifiable parameters (D=2)
──────────────────────────────
    theta[0]  IS_0    -- insulin sensitivity baseline [mg/dL/min / (pmol/L)]
                         Prior: LogN(log 0.010, 0.30^2)
    theta[1]  LG_max  -- liver glycogen capacity [g]
                         Prior: LogN(log 80, 0.12^2)  (CV ~12%)

Non-centred parametrisation (Matt trick, Betancourt-Girolami 2015):
    eta_raw_i ~ N(0, I_D)
    eta_i      = L_Omega * eta_raw_i
    theta_i    = exp(log(theta_pop) + eta_i)

Identifiability rationale:
    IS_0   -- controls amplitude/duration of postprandial glucose excursion.
    LG_max -- controls EGP capacity and timing of fasting/exercise glucose peak.
    All other parameters fixed at population prior (CLAUDE.md §3: non-identifiable
    parameters are fixed at the prior, not estimated).

Backends:
    NumPyro SVI (AutoLowRankMultivariateNormal) for fleet-scale fitting.
    NUTS validation in subset for R-hat < 1.01 convergence check.

Fail-Loud contract:
    ELBO NaN       -> RuntimeError (divergence).
    numpyro absent -> RuntimeError with installation instructions.
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
    import optax
    _NUMPYRO_OK = True
except ImportError:
    numpyro = None  # type: ignore[assignment]
    dist    = None  # type: ignore[assignment]
    SVI     = None  # type: ignore[assignment]
    _NUMPYRO_OK = False

from app.slices.metabolic_glucose.ode import (
    metabolic_glucose_ode,
    GlucoseMetabParams,
    DEFAULT_PARAMS,
    X0_DEFAULT,
    HUBS_DEFAULT,
    STATE_DIM,
    IDX_G,
)

logger = logging.getLogger(__name__)

D_THETA: int = 2
THETA_NAMES: list[str] = ["IS_0", "LG_max"]

_LOG_PRIOR_MEAN = jnp.log(jnp.array([0.010, 80.0], dtype=jnp.float32))
_LOG_PRIOR_SD   = jnp.array([0.30, 0.12], dtype=jnp.float32)


def _params_from_log_theta(log_theta_i: jax.Array) -> GlucoseMetabParams:
    """Build GlucoseMetabParams with IS_0 and LG_max set to exp(log_theta_i)."""
    theta = jnp.exp(log_theta_i)
    return DEFAULT_PARAMS._replace(IS_0=theta[0], LG_max=theta[1])


@jax.jit
def simulate_glucose_trajectory(
    log_theta_i:  jax.Array,    # (D_THETA,)
    x0:           jax.Array,    # (STATE_DIM,)
    hub_schedule: jax.Array,    # (T, HUB_DIM) -- hub values per 1-min step
    dt_min:       float = 1.0,
) -> jax.Array:                 # (T,) -- predicted Plasma_Glucose [mg/dL]
    """
    Simulate the 6-state glucose trajectory for one subject.

    jax.lax.scan over T steps; fully JAX-traceable for SVI gradient estimation.
    """
    params_i = _params_from_log_theta(log_theta_i)

    def _step(x: jax.Array, hubs_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        sol = diffrax.diffeqsolve(
            terms    = diffrax.ODETerm(metabolic_glucose_ode),
            solver   = diffrax.Tsit5(),
            t0       = jnp.float32(0.0),
            t1       = jnp.float32(dt_min),
            dt0      = jnp.float32(0.5),
            y0       = x,
            args     = (params_i, hubs_t),
            saveat   = diffrax.SaveAt(t1=True),
            max_steps= 64,
        )
        x_next = sol.ys[0]
        return x_next, x_next[IDX_G]

    _, G_traj = jax.lax.scan(_step, x0, hub_schedule)
    return G_traj   # (T,)


def glucose_nlme_model(
    data:         dict,
    observations: jax.Array,   # (N_obs,) CGM readings [mg/dL]; NaN = missing
    subject_ids:  jax.Array,   # (N_obs,) int
    obs_step_ids: jax.Array,   # (N_obs,) int -- time-step index per observation
    n_subjects:   int,
) -> None:
    """
    NumPyro NLME model for glucose kinetics (D=2, non-centred / Matt trick).

    data dict:
        "hub_schedules" : (n_subjects, T, HUB_DIM)
        "x0s"           : (n_subjects, STATE_DIM)
    """
    if not _NUMPYRO_OK:
        raise RuntimeError(
            "numpyro>=0.15 required for glucose_nlme_model. "
            "Install: pip install numpyro optax"
        )

    # Population fixed effects
    theta_pop_log = numpyro.sample(
        "theta_pop_log",
        dist.Normal(_LOG_PRIOR_MEAN, _LOG_PRIOR_SD),
    )

    # Between-subject covariance (LKJ-Cholesky + scale)
    scale_eta = numpyro.sample(
        "scale_eta",
        dist.HalfNormal(0.20).expand([D_THETA]),
    )
    Omega_chol = numpyro.sample(
        "Omega_chol",
        dist.LKJCholesky(D_THETA, concentration=2.0),
    )
    L_eta = jnp.diag(scale_eta) @ Omega_chol

    sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(5.0))

    with numpyro.plate("subjects", n_subjects):
        eta_raw = numpyro.sample(
            "eta_raw",
            dist.Normal(0.0, 1.0).expand([D_THETA]).to_event(1),
        )
        # Matt trick: non-centred parametrisation
        eta_i       = eta_raw @ L_eta.T
        log_theta_i = theta_pop_log + eta_i

    G_trajs = jax.vmap(
        simulate_glucose_trajectory,
        in_axes=(0, 0, 0, None),
    )(
        log_theta_i,
        jnp.asarray(data["x0s"],          dtype=jnp.float32),
        jnp.asarray(data["hub_schedules"], dtype=jnp.float32),
    )   # (n_subjects, T)

    y_pred    = G_trajs[subject_ids, obs_step_ids]   # (N_obs,)
    obs_valid = ~jnp.isnan(observations)
    sigma_eff = jnp.where(obs_valid, sigma_obs, jnp.float32(1e4))
    y_safe    = jnp.where(obs_valid, observations, y_pred)

    with numpyro.plate("observations", observations.shape[0]):
        numpyro.sample("y_obs", dist.Normal(y_pred, sigma_eff), obs=y_safe)


class GlucoseNLMEResult(NamedTuple):
    params:       dict
    elbo_history: jax.Array
    svi_state:    object
    guide:        object


class GlucoseNLME:
    """
    L3 NLME Population Model -- Metabolic Glucose V2.0.

    Personalises IS_0 (insulin sensitivity) and LG_max (liver glycogen capacity)
    per individual from CGM meal-challenge data.

    Fail-Loud: ELBO NaN -> RuntimeError; numpyro absent -> RuntimeError.
    """

    def __init__(self) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError(
                "numpyro>=0.15 required for GlucoseNLME. "
                "Install: pip install numpyro optax"
            )
        logger.info("GlucoseNLME initialised (D=%d: %s).", D_THETA, THETA_NAMES)

    def fit_svi(
        self,
        data:         dict,
        observations: jax.Array,
        subject_ids:  jax.Array,
        obs_step_ids: jax.Array,
        n_subjects:   int,
        n_steps:      int   = 30_000,
        lr:           float = 1e-3,
        rank:         int   = 4,
        seed:         int   = 0,
    ) -> GlucoseNLMEResult:
        """Fit via SVI (AutoLowRankMultivariateNormal, Adam lr)."""
        obs  = jnp.asarray(observations, dtype=jnp.float32)
        sids = jnp.asarray(subject_ids,  dtype=jnp.int32)
        tids = jnp.asarray(obs_step_ids, dtype=jnp.int32)

        guide     = AutoLowRankMultivariateNormal(glucose_nlme_model, rank=rank)
        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(lr))
        svi       = SVI(glucose_nlme_model, guide, optimizer, loss=Trace_ELBO())

        rng_key   = jax.random.PRNGKey(seed)
        svi_state = svi.init(rng_key, data, obs, sids, tids, n_subjects)

        elbo_vals: list[float] = []
        for step in range(n_steps):
            svi_state, loss = svi.update(svi_state, data, obs, sids, tids, n_subjects)
            elbo_vals.append(-float(loss))
            if jnp.isnan(jnp.float32(loss)):
                raise RuntimeError(
                    f"GlucoseNLME.fit_svi: ELBO is NaN at step {step}. "
                    "Fail-Loud: divergence detected."
                )
            if step % 5_000 == 0 or step == n_steps - 1:
                logger.debug("SVI step %d/%d  ELBO=%.1f", step, n_steps, elbo_vals[-1])

        return GlucoseNLMEResult(
            params       = svi.get_params(svi_state),
            elbo_history = jnp.array(elbo_vals, dtype=jnp.float32),
            svi_state    = svi_state,
            guide        = guide,
        )

    def get_population_theta(self, result: GlucoseNLMEResult) -> jax.Array:
        """Return theta_pop = exp(theta_pop_log median), shape (D_THETA,)."""
        median = result.guide.median(result.params)
        return jnp.exp(median["theta_pop_log"])

    def get_personalised_params(
        self,
        result:     GlucoseNLMEResult,
        subject_id: int,
        n_samples:  int = 500,
        seed:       int = 1,
    ) -> GlucoseMetabParams:
        """Return GlucoseMetabParams with IS_0 and LG_max at posterior median."""
        rng = jax.random.PRNGKey(seed)
        posterior = result.guide.sample_posterior(rng, result.params, sample_shape=(n_samples,))
        theta_pop  = posterior["theta_pop_log"]
        eta_raw    = posterior["eta_raw"]
        scale_eta  = posterior["scale_eta"]
        Omega_chol = posterior["Omega_chol"]

        L_eta  = jax.vmap(lambda sc, oc: jnp.diag(sc) @ oc)(scale_eta, Omega_chol)
        eta_i  = jax.vmap(lambda e, L: e[:, subject_id, :] @ L.T)(eta_raw, L_eta)
        log_theta_samples = theta_pop + eta_i   # (n_samples, D_THETA)

        log_theta_med = jnp.median(log_theta_samples, axis=0)
        theta_med     = jnp.exp(log_theta_med)

        return DEFAULT_PARAMS._replace(IS_0=float(theta_med[0]), LG_max=float(theta_med[1]))
