"""
app/slices/gonadal_axis/nlme.py

L3 NLME Population Layer — Gonadal Axis Slice (polymorphic)

Personalised parameters
────────────────────────
Female (D=2):
  Follicular_Sensitivity  — k_FM_growth scale [adim]
    Controls time-to-ovulation: low → slow folliculogenesis (older / athletic
    adaptation). Prior: LogN(log 1.0, 0.30²). Unit: multiplicative on k_FM_growth.
    Röblitz 2013 / Dólleman 2013 (age-dependent follicle dynamics).

  Luteal_Lifespan         — tau_LM [days]
    Corpus luteum survival. Short → luteal phase defect; long → normal/hyperprogesterone.
    Prior: LogN(log 12.0, 0.20²). Csapo 1956 / Filicori 1984.

Male (D=2):
  Leydig_Sensitivity      — k_T_LH scale [adim]
    Steroidogenic efficiency per IU/L LH. Prior: LogN(log 1.0, 0.35²).
    Veldhuis 1994 (age/training-dependent Leydig efficiency).

  Aromatase_Activity      — k_aromatase scale [adim]
    Peripheral T→E2 conversion factor. Prior: LogN(log 1.0, 0.40²).
    Sinha-Hikim 1998; adiposity- and age-dependent.

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015)
  η_raw_i ~ N(0, I_D)
  η_i      = L_Ω · η_raw_i
  θ_i      = θ_pop · exp(η_i)

References
──────────
  Röblitz S. et al. (2013) Adv Comput Math 39:23-48  DOI 10.1007/s10444-012-9260-0
  Veldhuis J.D. et al. (1994) Recent Prog Horm Res 49:363-395
  Betancourt M. & Girolami M. (2015) arXiv:1312.0906
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
    numpyro = None      # type: ignore[assignment]
    dist    = None      # type: ignore[assignment]
    SVI     = None      # type: ignore[assignment]
    _NUMPYRO_OK = False

from app.slices.gonadal_axis.female_ode import (
    female_gonadal_ode,
    FemaleGonadalParams,
    DEFAULT_FEMALE_PARAMS,
    X0_FEMALE_DEFAULT,
    IDX_F_LH, IDX_F_E2,
    STATE_DIM_FEMALE,
)
from app.slices.gonadal_axis.male_ode import (
    male_gonadal_ode,
    MaleGonadalParams,
    DEFAULT_MALE_PARAMS,
    X0_MALE_DEFAULT,
    IDX_M_T,
    STATE_DIM_MALE,
)

logger = logging.getLogger(__name__)

# ── Parameter definitions ─────────────────────────────────────────────────────

D_THETA_FEMALE:     int       = 2
THETA_NAMES_FEMALE: list[str] = ["Follicular_Sensitivity", "Luteal_Lifespan"]

D_THETA_MALE:     int       = 2
THETA_NAMES_MALE: list[str] = ["Leydig_Sensitivity", "Aromatase_Activity"]

# Population log-space priors (log-mean, log-sd) — literature anchored
_FEMALE_LOG_PRIOR_MEAN = jnp.log(jnp.array([1.0,  12.0], dtype=jnp.float32))
_FEMALE_LOG_PRIOR_SD   = jnp.array([0.30, 0.20], dtype=jnp.float32)

_MALE_LOG_PRIOR_MEAN   = jnp.log(jnp.array([1.0, 1.0], dtype=jnp.float32))
_MALE_LOG_PRIOR_SD     = jnp.array([0.35, 0.40], dtype=jnp.float32)


# ── Parameter patching helpers ────────────────────────────────────────────────

def _patch_female_params(
    log_theta_i: jax.Array,
    base: FemaleGonadalParams = DEFAULT_FEMALE_PARAMS,
) -> FemaleGonadalParams:
    """
    Return FemaleGonadalParams with Follicular_Sensitivity and Luteal_Lifespan
    scaled by exp(log_theta_i).

    log_theta_i[0] → scale on k_FM_growth  (Follicular_Sensitivity)
    log_theta_i[1] → tau_LM                (Luteal_Lifespan, directly)
    """
    theta = jnp.exp(log_theta_i)
    return base._replace(
        k_FM_growth = base.k_FM_growth * theta[0],
        tau_LM      = theta[1],
    )


def _patch_male_params(
    log_theta_i: jax.Array,
    base: MaleGonadalParams = DEFAULT_MALE_PARAMS,
) -> MaleGonadalParams:
    """
    Return MaleGonadalParams with Leydig_Sensitivity and Aromatase_Activity
    scaled by exp(log_theta_i).

    log_theta_i[0] → scale on k_T_LH         (Leydig_Sensitivity)
    log_theta_i[1] → scale on k_aromatase     (Aromatase_Activity)
    """
    theta = jnp.exp(log_theta_i)
    return base._replace(
        k_T_LH      = base.k_T_LH      * theta[0],
        k_aromatase = base.k_aromatase  * theta[1],
    )


# ── Forward simulation functions (JIT-traceable for SVI) ─────────────────────

@jax.jit
def simulate_lh_trajectory_female(
    log_theta_i: jax.Array,   # (2,): log [Follicular_Sensitivity, Luteal_Lifespan]
    x0:          jax.Array,   # (STATE_DIM_FEMALE,)
    ea_series:   jax.Array,   # (T,): EA [kcal/kg FFM/day] per day
    dt_day:      float = 1.0,
) -> jax.Array:               # (T,): predicted LH [IU/L]
    """
    Simulate daily LH trajectory for one female subject.
    Uses jax.lax.scan + diffrax.Tsit5 over T days.
    """
    params_i = _patch_female_params(log_theta_i)

    def _step(x: jax.Array, ea_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(female_gonadal_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(dt_day),
            dt0       = jnp.float32(0.1),
            y0        = x,
            args      = (params_i, ea_t),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 64,
        )
        x_next = sol.ys[0]
        return x_next, x_next[IDX_F_LH]

    _, lh_traj = jax.lax.scan(_step, x0, ea_series)
    return lh_traj


@jax.jit
def simulate_t_trajectory_male(
    log_theta_i: jax.Array,   # (2,): log [Leydig_Sensitivity, Aromatase_Activity]
    x0:          jax.Array,   # (STATE_DIM_MALE,)
    ea_series:   jax.Array,   # (T,): EA [kcal/kg FFM/day]
    dt_day:      float = 1.0,
) -> jax.Array:               # (T,): predicted Testosterone [ng/dL]
    """
    Simulate daily testosterone trajectory for one male subject.
    """
    params_i = _patch_male_params(log_theta_i)

    def _step(x: jax.Array, ea_t: jax.Array) -> tuple[jax.Array, jax.Array]:
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(male_gonadal_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(dt_day),
            dt0       = jnp.float32(0.1),
            y0        = x,
            args      = (params_i, ea_t),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 64,
        )
        x_next = sol.ys[0]
        return x_next, x_next[IDX_M_T]

    _, t_traj = jax.lax.scan(_step, x0, ea_series)
    return t_traj


# ── NumPyro NLME models ───────────────────────────────────────────────────────

def female_gonadal_nlme_model(
    session_data: dict,
    observations: jax.Array,   # (N_obs,) — LH readings [IU/L]
    subject_ids:  jax.Array,
    obs_step_ids: jax.Array,
    n_subjects:   int,
) -> None:
    """
    Female HPG axis NLME model. Personalises Follicular_Sensitivity and Luteal_Lifespan.
    Non-centred parametrisation.
    """
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro>=0.15 required.")

    theta_pop_log = numpyro.sample(
        "theta_pop_log", dist.Normal(_FEMALE_LOG_PRIOR_MEAN, _FEMALE_LOG_PRIOR_SD)
    )
    scale_eta  = numpyro.sample("scale_eta",  dist.HalfNormal(0.25).expand([D_THETA_FEMALE]))
    Omega_chol = numpyro.sample("Omega_chol", dist.LKJCholesky(D_THETA_FEMALE, 2.0))
    L_eta      = jnp.diag(scale_eta) @ Omega_chol
    sigma_obs  = numpyro.sample("sigma_obs",  dist.HalfNormal(5.0))

    with numpyro.plate("subjects", n_subjects):
        eta_raw     = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([D_THETA_FEMALE]).to_event(1))
        eta_i       = eta_raw @ L_eta.T
        log_theta_i = theta_pop_log + eta_i

    lh_trajs = jax.vmap(simulate_lh_trajectory_female, in_axes=(0, 0, 0, None))(
        log_theta_i,
        jnp.asarray(session_data["x0s"],      dtype=jnp.float32),
        jnp.asarray(session_data["ea_series"], dtype=jnp.float32),
    )
    y_pred   = lh_trajs[subject_ids, obs_step_ids]
    obs_val  = ~jnp.isnan(observations)
    sigma_eff = jnp.where(obs_val, sigma_obs, jnp.float32(1e4))
    y_safe    = jnp.where(obs_val, observations, y_pred)

    with numpyro.plate("observations", observations.shape[0]):
        numpyro.sample("y_obs", dist.Normal(y_pred, sigma_eff), obs=y_safe)


def male_gonadal_nlme_model(
    session_data: dict,
    observations: jax.Array,   # (N_obs,) — T readings [ng/dL]
    subject_ids:  jax.Array,
    obs_step_ids: jax.Array,
    n_subjects:   int,
) -> None:
    """
    Male HPG axis NLME model. Personalises Leydig_Sensitivity and Aromatase_Activity.
    """
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro>=0.15 required.")

    theta_pop_log = numpyro.sample(
        "theta_pop_log", dist.Normal(_MALE_LOG_PRIOR_MEAN, _MALE_LOG_PRIOR_SD)
    )
    scale_eta  = numpyro.sample("scale_eta",  dist.HalfNormal(0.30).expand([D_THETA_MALE]))
    Omega_chol = numpyro.sample("Omega_chol", dist.LKJCholesky(D_THETA_MALE, 2.0))
    L_eta      = jnp.diag(scale_eta) @ Omega_chol
    sigma_obs  = numpyro.sample("sigma_obs",  dist.HalfNormal(30.0))

    with numpyro.plate("subjects", n_subjects):
        eta_raw     = numpyro.sample("eta_raw", dist.Normal(0.0, 1.0).expand([D_THETA_MALE]).to_event(1))
        eta_i       = eta_raw @ L_eta.T
        log_theta_i = theta_pop_log + eta_i

    t_trajs = jax.vmap(simulate_t_trajectory_male, in_axes=(0, 0, 0, None))(
        log_theta_i,
        jnp.asarray(session_data["x0s"],      dtype=jnp.float32),
        jnp.asarray(session_data["ea_series"], dtype=jnp.float32),
    )
    y_pred   = t_trajs[subject_ids, obs_step_ids]
    obs_val  = ~jnp.isnan(observations)
    sigma_eff = jnp.where(obs_val, sigma_obs, jnp.float32(1e4))
    y_safe    = jnp.where(obs_val, observations, y_pred)

    with numpyro.plate("observations", observations.shape[0]):
        numpyro.sample("y_obs", dist.Normal(y_pred, sigma_eff), obs=y_safe)


# ── SVI result container ─────────────────────────────────────────────────────

class GonadalSVIResult(NamedTuple):
    params:       dict
    elbo_history: jax.Array
    svi_state:    object
    guide:        object
    is_female:    bool


# ── Main NLME class ───────────────────────────────────────────────────────────

class GonadalNLME:
    """
    L3 NLME for the gonadal axis slice.

    Personalises:
      Female: Follicular_Sensitivity, Luteal_Lifespan
      Male:   Leydig_Sensitivity, Aromatase_Activity

    Usage
    -----
    nlme = GonadalNLME(is_female=True)
    result = nlme.fit_svi(session_data, observations, subject_ids, obs_step_ids, N)
    params_i = nlme.get_params_for_subject(result, subject_id=0)
    """

    def __init__(self, is_female: bool) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError("numpyro>=0.15 required. Install: pip install numpyro optax")
        self.is_female = is_female
        self._model    = female_gonadal_nlme_model if is_female else male_gonadal_nlme_model
        self._D        = D_THETA_FEMALE if is_female else D_THETA_MALE
        self._names    = THETA_NAMES_FEMALE if is_female else THETA_NAMES_MALE
        logger.info("GonadalNLME(is_female=%s) D=%d: %s", is_female, self._D, self._names)

    def fit_svi(
        self,
        session_data: dict,
        observations: jax.Array,
        subject_ids:  jax.Array,
        obs_step_ids: jax.Array,
        n_subjects:   int,
        n_steps:      int   = 20_000,
        lr:           float = 1e-3,
        rank:         int   = 4,
        seed:         int   = 0,
    ) -> GonadalSVIResult:
        obs  = jnp.asarray(observations, dtype=jnp.float32)
        sids = jnp.asarray(subject_ids,  dtype=jnp.int32)
        tids = jnp.asarray(obs_step_ids, dtype=jnp.int32)

        guide     = AutoLowRankMultivariateNormal(self._model, rank=rank)
        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(lr))
        svi       = SVI(self._model, guide, optimizer, loss=Trace_ELBO())
        rng_key   = jax.random.PRNGKey(seed)
        svi_state = svi.init(rng_key, session_data, obs, sids, tids, n_subjects)

        elbo_vals: list[float] = []
        for step in range(n_steps):
            svi_state, loss = svi.update(svi_state, session_data, obs, sids, tids, n_subjects)
            elbo_vals.append(-float(loss))
            if jnp.isnan(jnp.float32(loss)):
                raise RuntimeError(f"GonadalNLME.fit_svi: ELBO NaN at step {step}.")
            if step % 5_000 == 0 or step == n_steps - 1:
                logger.debug("SVI %d/%d  ELBO=%.1f", step, n_steps, elbo_vals[-1])

        return GonadalSVIResult(
            params       = svi.get_params(svi_state),
            elbo_history = jnp.array(elbo_vals, dtype=jnp.float32),
            svi_state    = svi_state,
            guide        = guide,
            is_female    = self.is_female,
        )

    def get_params_for_subject(
        self,
        result:     GonadalSVIResult,
        subject_id: int,
        n_samples:  int = 500,
        seed:       int = 1,
    ) -> FemaleGonadalParams | MaleGonadalParams:
        """
        Return personalised ODE params at posterior median for subject.
        """
        rng_key   = jax.random.PRNGKey(seed)
        posterior = result.guide.sample_posterior(
            rng_key, result.params, sample_shape=(n_samples,)
        )
        log_theta_pop = posterior["theta_pop_log"]   # (n_samples, D)
        eta_raw       = posterior["eta_raw"]          # (n_samples, n_subjects, D)
        scale_eta     = posterior["scale_eta"]
        Omega_chol    = posterior["Omega_chol"]
        L_eta  = jax.vmap(lambda sc, oc: jnp.diag(sc) @ oc)(scale_eta, Omega_chol)
        eta_i  = jax.vmap(lambda e, L: e[:, subject_id, :] @ L.T)(eta_raw, L_eta)
        log_theta_i   = log_theta_pop + eta_i                          # (n_samples, D)
        log_theta_med = jnp.median(log_theta_i, axis=0)

        if self.is_female:
            return _patch_female_params(log_theta_med)
        else:
            return _patch_male_params(log_theta_med)
