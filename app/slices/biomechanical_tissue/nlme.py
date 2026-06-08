"""
app/slices/biomechanical_tissue/nlme.py

L3 NLME Population Layer -- Biomechanical Tissue Slice

Personalises 2 kinetic parameters per individual (theta_i) from hourly
Pain_VAS and Ultrasound_Echogenicity data using Nonlinear Mixed-Effects
modelling (NumPyro).

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015)
--------------------------------------------------------------------
    eta_raw_i ~ N(0, I_D)
    eta_i     = L_Omega * eta_raw_i
    theta_i   = theta_pop * exp(eta_i)    [log-normal, positive]

Personalised parameters (D = 2)
---------------------------------
    Baseline_Tendon_Stiffness [au]   -- resting tendon stiffness prior
                                        Genetic basis: COL1A1 collagen
                                        scaffold quality determines
                                        baseline mechanical stiffness
                                        (Mokone 2005 Am J Hum Genet;
                                         Posthumus 2009 Int J Sports Med)
    Bone_Remodeling_Rate      [h^-1] -- Wolff's Law osteoblast anabolic
                                        rate; encodes individual adaptation
                                        velocity to mechanical loading
                                        (Turner 1998; Robling 2008 Nature)

Identifiability rationale
--------------------------
Only these two parameters are structurally identifiable from hourly
Pain_VAS + Echogenicity streams at reasonable SNR:
  Baseline_Tendon_Stiffness: drives baseline echogenicity and pain
    intercept; observable from resting (zero-load) windows.
  Bone_Remodeling_Rate: drives the rate of BMD accumulation under
    moderate loading; reflected in the DEXA_ZScore trend over weeks.

All other biomechanical ODE parameters remain fixed at population priors.

Population priors (log-space, literature-anchored)
----------------------------------------------------
    Baseline_Tendon_Stiffness ~ LogN(log 1.00, 0.25^2)  -- 25% pop. CV
    Bone_Remodeling_Rate      ~ LogN(log 5e-5, 0.35^2)  -- 35% pop. CV
        (k_wolff = 5e-5 h^-1; inter-individual range ~ 2.5e-5 to 1e-4)

Forward simulation
-------------------
diffrax.Tsit5 with dt_hours=1.0. Exact parity with filter.py
_integrate_1hour. jax.lax.scan for O(T) memory; fully JIT-traceable.
Supports SVI gradient estimation and NUTS validation.

Fail-Loud contract
-------------------
RuntimeError if ELBO contains NaN.
RuntimeError if numpyro is not installed (with install instructions).

References
----------
  Turner C.H. (1998) Bone 23:399-407                   [bone mechanostat]
  Robling A.G. et al. (2008) Nature 461:316-320         [sclerostin/Wnt]
  Mokone G.G. et al. (2005) Am J Hum Genet 77:163-169  [COL1A1 tendon]
  Posthumus M. et al. (2009) Int J Sports Med 30:590-5  [COL1A1 CV]
  Betancourt M., Girolami M. (2015) arXiv:1312.0906    [Matt trick]
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
    numpyro = None       # type: ignore[assignment]
    dist    = None       # type: ignore[assignment]
    SVI     = None       # type: ignore[assignment]
    _NUMPYRO_OK = False

from app.slices.biomechanical_tissue.ode import (
    BiomechanicalParams,
    DEFAULT_BIO_PARAMS,
    X0_BIO_DEFAULT,
    STATE_DIM,
    IDX_TEND_STIFF,
    IDX_BONE_DENS,
    biomechanical_ode,
)
from app.slices.biomechanical_tissue.observation import (
    BioObsParams,
    DEFAULT_BIO_OBS_PARAMS,
    h_bio,
)

logger = logging.getLogger(__name__)

# -- D = 2 personalised parameter names ----------------------------------------
PARAM_NAMES: tuple[str, ...] = ("Baseline_Tendon_Stiffness", "Bone_Remodeling_Rate")
D: int = len(PARAM_NAMES)

# -- Population prior means (log-space) ----------------------------------------
_THETA_POP_LOG_MEAN: dict[str, float] = {
    "Baseline_Tendon_Stiffness": float(jnp.log(jnp.float32(1.00))),   # au
    "Bone_Remodeling_Rate":      float(jnp.log(jnp.float32(5e-5))),   # h^-1
}

# -- Population prior SDs (log-space) ------------------------------------------
_THETA_POP_LOG_SD: dict[str, float] = {
    "Baseline_Tendon_Stiffness": 0.25,   # 25% pop. CV (Posthumus 2009)
    "Bone_Remodeling_Rate":      0.35,   # 35% pop. CV (Turner 1998)
}


# -- diffrax.Tsit5 forward simulation ------------------------------------------

def _forward_simulate(
    theta_i:    jax.Array,
    x0:         jax.Array,
    controls:   jax.Array,
    obs_params: BioObsParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulate the biomechanical ODE for T hourly steps with personalised theta_i.

    theta_i = [Baseline_Tendon_Stiffness, Bone_Remodeling_Rate] (natural space).

    Baseline_Tendon_Stiffness shifts the initial Tendon_Stiffness state x0[2].
    Bone_Remodeling_Rate overrides params.k_wolff.

    Parameters
    ----------
    theta_i   : (D,) in natural space
    x0        : (STATE_DIM,)
    controls  : (T, CTRL_DIM) -- hourly hub inputs
    obs_params: BioObsParams

    Returns
    -------
    y_pred  : (T, OBS_DIM)   -- [Pain_VAS, Echogenicity, DEXA_Z] per step
    x_traj  : (T, STATE_DIM) -- full state trajectory
    """
    stiffness_base   = theta_i[0]
    bone_rmodel_rate = theta_i[1]

    bio_params = DEFAULT_BIO_PARAMS._replace(
        k_wolff = bone_rmodel_rate,
    )

    # Shift initial stiffness by personalised baseline
    x0_i = x0.at[IDX_TEND_STIFF].set(
        jnp.maximum(stiffness_base, jnp.float32(1e-4))
    )

    def _step(x_carry, u_t):
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(biomechanical_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(1.0),    # dt = 1 hour
            dt0       = jnp.float32(0.1),
            y0        = x_carry,
            args      = (bio_params, u_t),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 64,
        )
        x_next = sol.ys[0]
        y_t    = h_bio(x_next, obs_params, bio_params)
        return x_next, (y_t, x_next)

    _, (y_pred, x_traj) = jax.lax.scan(_step, x0_i, controls)
    return y_pred, x_traj   # (T, OBS_DIM), (T, STATE_DIM)


# -- NumPyro NLME model --------------------------------------------------------

def biomechanical_nlme_model(
    observations:   "jax.Array | None",  # (N_total, 2) [Pain_VAS, Echo] or None
    controls_list:  jax.Array,           # (N_subjects, T, CTRL_DIM)
    subject_ids:    jax.Array,           # (N_total,) int
    obs_idx:        jax.Array,           # (N_total,) int
    n_subjects:     int,
    dexa_obs:       "jax.Array | None" = None,   # (N_dexa,) Z-scores
    dexa_sub_ids:   "jax.Array | None" = None,   # (N_dexa,) int
    dexa_obs_idx:   "jax.Array | None" = None,   # (N_dexa,) int
) -> None:
    """
    NumPyro hierarchical NLME model for biomechanical tissue personalisation.

    Non-centred (Matt trick) parametrisation:
        eta_raw_i ~ N(0, I_D)
        eta_i     = L_Omega * eta_raw_i
        theta_i   = theta_pop * exp(eta_i)

    Likelihood:
        [Pain_VAS, Echogenicity] ~ Normal(h_bio(ODE(theta_i)[t])[:2], sigma_obs)
        DEXA_Z                  ~ Normal(h_bio(x_t)[2],               sigma_dexa)

    Parameters
    ----------
    observations  : (N_total, 2) float32 -- [Pain_VAS, Echogenicity]; None for prior pred
    controls_list : (N_subjects, T, CTRL_DIM) float32
    subject_ids   : (N_total,) int32
    obs_idx       : (N_total,) int32
    n_subjects    : int
    dexa_obs      : (N_dexa,) float32 -- DEXA Z-scores (optional, sparse)
    dexa_sub_ids  : (N_dexa,) int32
    dexa_obs_idx  : (N_dexa,) int32
    """
    if not _NUMPYRO_OK:
        raise RuntimeError(
            "numpyro is required for BiomechanicalNLME. "
            "Install: pip install numpyro>=0.15"
        )

    # -- Population-level parameters (log-space) --------------------------------
    theta_pop_log = numpyro.sample(
        "theta_pop_log",
        dist.Normal(
            jnp.array([
                _THETA_POP_LOG_MEAN["Baseline_Tendon_Stiffness"],
                _THETA_POP_LOG_MEAN["Bone_Remodeling_Rate"],
            ], dtype=jnp.float32),
            jnp.array([
                _THETA_POP_LOG_SD["Baseline_Tendon_Stiffness"],
                _THETA_POP_LOG_SD["Bone_Remodeling_Rate"],
            ], dtype=jnp.float32),
        ),
    )   # (D,)

    # -- Between-subject covariance (LKJ-Cholesky; concentration=2.0) -----------
    Omega_chol = numpyro.sample(
        "Omega_chol",
        dist.LKJCholesky(D, concentration=2.0),
    )   # (D, D)

    # -- Between-subject scale ---------------------------------------------------
    scale_eta = numpyro.sample(
        "scale_eta",
        dist.HalfNormal(jnp.array([0.25, 0.35], dtype=jnp.float32)),
    )   # (D,)

    # -- Observation noise (Pain_VAS, Echogenicity) -----------------------------
    sigma_vas  = numpyro.sample("sigma_vas",  dist.HalfNormal(jnp.float32(1.5)))
    sigma_echo = numpyro.sample("sigma_echo", dist.HalfNormal(jnp.float32(0.10)))
    sigma_obs  = jnp.stack([sigma_vas, sigma_echo])   # (2,)

    # -- Subject-level parameters (non-centred) ---------------------------------
    with numpyro.plate("subjects", n_subjects):
        eta_raw = numpyro.sample(
            "eta_raw",
            dist.Normal(
                jnp.zeros(D, dtype=jnp.float32),
                jnp.ones(D,  dtype=jnp.float32),
            ),
        )   # (N_subjects, D)

        eta_i      = eta_raw * scale_eta[None, :]
        eta_i_corr = jnp.einsum("ij,sj->si", Omega_chol, eta_i)
        theta_i    = jnp.exp(theta_pop_log[None, :] + eta_i_corr)   # (N_subjects, D)

    # -- Forward simulation (vmap over subjects) --------------------------------
    obs_params_fixed = DEFAULT_BIO_OBS_PARAMS

    def _simulate_subject(theta_s, controls_s):
        return _forward_simulate(theta_s, X0_BIO_DEFAULT, controls_s, obs_params_fixed)

    y_pred_all, x_traj_all = jax.vmap(_simulate_subject)(theta_i, controls_list)
    # (N_subjects, T, OBS_DIM), (N_subjects, T, STATE_DIM)

    # -- Likelihood -- Pain_VAS + Echogenicity daily stream --------------------
    if observations is not None:
        y_pred_flat = y_pred_all[subject_ids, obs_idx, :2]   # (N_total, 2)
        with numpyro.plate("observations", observations.shape[0]):
            numpyro.sample(
                "y_obs",
                dist.Normal(y_pred_flat, sigma_obs[None, :]),
                obs=observations,
            )

    # -- Likelihood -- DEXA Z-score (sparse) -----------------------------------
    if dexa_obs is not None:
        sigma_dexa = numpyro.sample(
            "sigma_dexa",
            dist.HalfNormal(jnp.float32(obs_params_fixed.sigma_DEXA)),
        )
        dexa_pred = y_pred_all[dexa_sub_ids, dexa_obs_idx, 2]   # (N_dexa,)
        with numpyro.plate("dexa_observations", dexa_obs.shape[0]):
            numpyro.sample("y_dexa", dist.Normal(dexa_pred, sigma_dexa), obs=dexa_obs)


# -- Public class --------------------------------------------------------------

class BiomechanicalNLME:
    """
    L3 NLME personalisation wrapper for the Biomechanical Tissue slice.

    Personalises Baseline_Tendon_Stiffness and Bone_Remodeling_Rate from
    hourly Pain_VAS + Ultrasound_Echogenicity and optional sparse DEXA Z-scores
    via SVI (AutoLowRankMultivariateNormal). NUTS validation available.

    Typical usage
    -------------
    nlme   = BiomechanicalNLME()
    result = nlme.fit_svi(data, n_steps=20_000)
    params = nlme.get_bio_params(result, subject_id=0)
    # -> BiomechanicalParams with k_wolff personalised;
    #    x0 shift applied separately via result["x0_shifts"]

    Fail-Loud contract
    ------------------
    RuntimeError on NaN ELBO.
    RuntimeError if numpyro not installed.
    """

    def __init__(self) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError(
                "numpyro is required for BiomechanicalNLME. "
                "Install: pip install numpyro>=0.15"
            )

    def fit_svi(
        self,
        data:     dict,
        n_steps:  int   = 20_000,
        lr:       float = 1e-3,
        rank:     int   = 4,
        rng_key:  int   = 42,
    ) -> dict:
        """
        Fit NLME model via SVI (AutoLowRankMultivariateNormal).

        Parameters
        ----------
        data    : dict with keys:
                  "observations"  : (N_total, 2) float32 -- [Pain_VAS, Echo]
                  "controls_list" : (N_subjects, T, CTRL_DIM) float32
                  "subject_ids"   : (N_total,) int32
                  "obs_idx"       : (N_total,) int32
                  "n_subjects"    : int
                  "dexa_obs"      : (N_dexa,) float32 (optional)
                  "dexa_sub_ids"  : (N_dexa,) int32   (optional)
                  "dexa_obs_idx"  : (N_dexa,) int32   (optional)
        n_steps : int
        lr      : float -- Adam learning rate
        rank    : int   -- low-rank covariance rank

        Returns
        -------
        dict: "svi_result", "elbo_history", "params", "guide"

        Raises
        ------
        RuntimeError on NaN ELBO.
        """
        key   = jax.random.PRNGKey(rng_key)
        guide = AutoLowRankMultivariateNormal(biomechanical_nlme_model, rank=rank)
        svi   = SVI(biomechanical_nlme_model, guide, optax.adam(lr), loss=Trace_ELBO())

        _dexa_kwargs = dict(
            dexa_obs     = data.get("dexa_obs"),
            dexa_sub_ids = data.get("dexa_sub_ids"),
            dexa_obs_idx = data.get("dexa_obs_idx"),
        )

        svi_state = svi.init(
            key,
            observations  = data["observations"],
            controls_list = data["controls_list"],
            subject_ids   = data["subject_ids"],
            obs_idx       = data["obs_idx"],
            n_subjects    = data["n_subjects"],
            **_dexa_kwargs,
        )

        elbo_history: list[float] = []

        for step in range(n_steps):
            svi_state, loss = svi.update(
                svi_state,
                observations  = data["observations"],
                controls_list = data["controls_list"],
                subject_ids   = data["subject_ids"],
                obs_idx       = data["obs_idx"],
                n_subjects    = data["n_subjects"],
                **_dexa_kwargs,
            )
            elbo_history.append(float(loss))

            if jnp.isnan(jnp.float32(loss)):
                raise RuntimeError(
                    f"BiomechanicalNLME.fit_svi: NaN ELBO at step {step}. "
                    "Check data scaling, prior means, and learning rate."
                )

            if step % 5_000 == 0:
                logger.info(
                    "BiomechanicalNLME SVI step %d/%d -- ELBO: %.2f",
                    step, n_steps, loss,
                )

        params = svi.get_params(svi_state)
        logger.info("BiomechanicalNLME SVI complete -- final ELBO: %.2f", elbo_history[-1])

        return {
            "svi_result":   svi_state,
            "elbo_history": elbo_history,
            "params":       params,
            "guide":        guide,
        }

    def get_bio_params(
        self,
        svi_result: dict,
        subject_id: int,
        n_samples:  int = 200,
        rng_key:    int = 1,
    ) -> tuple[BiomechanicalParams, jax.Array]:
        """
        Draw posterior mean personalised BiomechanicalParams for subject_id.

        Returns
        -------
        (BiomechanicalParams, x0_personalised)
            BiomechanicalParams: k_wolff replaced by Bone_Remodeling_Rate posterior mean.
            x0_personalised: X0_BIO_DEFAULT with Tendon_Stiffness = Baseline_Tendon_Stiffness.
        """
        key   = jax.random.PRNGKey(rng_key)
        guide = svi_result["guide"]

        posterior_samples = guide.sample_posterior(
            key, svi_result["params"], sample_shape=(n_samples,)
        )

        eta_raw_s    = posterior_samples.get("eta_raw",        jnp.zeros((n_samples, 1, D)))
        scale_eta_s  = posterior_samples.get("scale_eta",      jnp.ones((n_samples, D)) * 0.30)
        Omega_chol_s = posterior_samples.get("Omega_chol",     jnp.tile(jnp.eye(D)[None], (n_samples, 1, 1)))
        theta_pop_s  = posterior_samples.get(
            "theta_pop_log",
            jnp.array([[
                _THETA_POP_LOG_MEAN["Baseline_Tendon_Stiffness"],
                _THETA_POP_LOG_MEAN["Bone_Remodeling_Rate"],
            ]]),
        )

        def _theta_i(eta_r, sc, om, tp):
            eta   = eta_r[subject_id] * sc
            eta_c = om @ eta
            return jnp.exp(tp + eta_c)

        theta_samples = jax.vmap(_theta_i)(eta_raw_s, scale_eta_s, Omega_chol_s, theta_pop_s)
        theta_mean    = jnp.mean(theta_samples, axis=0)   # (D,)

        stiff_base       = float(theta_mean[0])
        bone_rmodel_rate = float(theta_mean[1])

        x0_i = X0_BIO_DEFAULT.at[IDX_TEND_STIFF].set(jnp.float32(max(stiff_base, 1e-4)))

        logger.info(
            "BiomechanicalNLME posterior mean (subject %d): "
            "Baseline_Tendon_Stiffness=%.4f, Bone_Remodeling_Rate=%.2e",
            subject_id, stiff_base, bone_rmodel_rate,
        )

        return (
            DEFAULT_BIO_PARAMS._replace(k_wolff=bone_rmodel_rate),
            x0_i,
        )

    @staticmethod
    def cold_start_params(
        col1a1_scale: float = 1.00,
    ) -> tuple[BiomechanicalParams, jax.Array]:
        """
        Population-mean BiomechanicalParams for new users (cold start).

        col1a1_scale: weak prior nudge from COL1A1 genotype [0.7, 1.3].
        Clamped to <=0.5*sigma shift per CLAUDE.md rule 3.4.

        Returns (BiomechanicalParams, x0_default).
        """
        prior_sd = 1.0 * 0.25    # sigma = 25% * 1.0
        max_shift = 0.5 * prior_sd
        scale_c = max(1.0 - max_shift, min(1.0 + max_shift, float(col1a1_scale)))

        x0_i = X0_BIO_DEFAULT.at[IDX_TEND_STIFF].set(
            jnp.float32(float(X0_BIO_DEFAULT[IDX_TEND_STIFF]) * scale_c)
        )
        return DEFAULT_BIO_PARAMS, x0_i
