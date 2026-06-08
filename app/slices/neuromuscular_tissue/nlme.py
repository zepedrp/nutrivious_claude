"""
app/slices/neuromuscular_tissue/nlme.py

L3 NLME Population Layer V4.0 -- Neuromuscular Tissue Slice

Personalises 2 kinetic parameters per individual from intra-session
[EMG_Amplitude_mV, SmO2_pct] time-series using Nonlinear Mixed-Effects
modelling (NumPyro).

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015)
--------------------------------------------------------------------
  eta_raw_i ~ N(0, I_D)                   D=2
  eta_i     = L_Omega * (scale_eta * eta_raw_i)
  theta_i   = theta_pop * exp(eta_i)      [log-space; both params positive]

Personalised parameters (D=2)
------------------------------
  P_th_2  [W]       -- Type 2 fiber recruitment threshold.
    Driven by ACTN3 R577X genotype (fast-twitch fiber proportion).
    Observable: SmO2 drop and EMG inflection at the power where T2 fibers
    activate (visible as a secondary rise in EMG slope and a steeper SmO2 fall).
    Population prior: LogN(log 250, 0.25^2)  -- 25% CV; range ~150-400W.

  k_SERCA_base  [min^-1] -- SERCA Ca2+ re-uptake rate.
    Governs calcium clearance speed and affects both Ca_SS level and LFF
    rate (via RyR1 oxidation).  Observable from EMG/SmO2 dynamics during
    the recovery phase after a high-intensity bout.
    Population prior: LogN(log 5.0, 0.20^2) -- 20% CV; range ~3-8 min^-1.

SVI backend
-----------
  Guide  : AutoLowRankMultivariateNormal (rank=5)
  Optim  : Optax Adam lr=1e-3
  Loss   : Trace_ELBO
  Budget : 30 000 steps

NUTS validation
---------------
  4 chains, 500 warmup, 500 samples. Diagnostics: R-hat, ESS.

Fail-Loud contract
------------------
  RuntimeError on NaN ELBO.
  RuntimeError if numpyro absent.

References
----------
  Betancourt M., Girolami M. (2015) arXiv:1312.0906   [non-centred param]
  Burke R.E. (1967) J Physiol 189:545                 [Henneman principle]
  MacIntosh B.R. et al. (2000) J Appl Physiol 89:1971 [SERCA rates]
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
    numpyro = None   # type: ignore[assignment]
    dist    = None   # type: ignore[assignment]
    SVI     = None   # type: ignore[assignment]
    _NUMPYRO_OK = False

from app.slices.neuromuscular_tissue.ode import (
    NMv4Params,
    DEFAULT_V4_PARAMS,
    X0_NM_V4,
    STATE_DIM,
    CTRL_DIM,
    nm_v4_ode,
)
from app.slices.neuromuscular_tissue.observation import (
    NMv4ObsParams,
    DEFAULT_V4_OBS_PARAMS,
    OBS_DIM,
    h_nm_v4,
)

logger = logging.getLogger(__name__)

# -- D=2 personalised parameter definitions ------------------------------------
PARAM_NAMES: tuple[str, ...] = ("P_th_2", "k_SERCA_base")
D: int = 2

# Population prior log-means
_LOG_MEAN_P_TH_2:    float = float(jnp.log(jnp.float32(250.0)))
_LOG_MEAN_K_SERCA:   float = float(jnp.log(jnp.float32(5.0)))

# Population prior log-SDs (CV)
_LOG_SD_P_TH_2:      float = 0.25   # 25% CV
_LOG_SD_K_SERCA:     float = 0.20   # 20% CV


# -- Forward simulation (parity with filter.py _integrate_1min) ----------------

def _forward_simulate_v4(
    theta_i:    jax.Array,
    x0:         jax.Array,
    controls:   jax.Array,
    obs_params: NMv4ObsParams,
) -> jax.Array:
    """
    Simulate the NM V4.0 ODE for T minutes with personalised parameters.

    theta_i = [P_th_2, k_SERCA_base] in natural (not log) space.
    Each 1-minute step uses diffrax.Tsit5 -- parity with filter.py.
    Uses jax.lax.scan for O(T) memory, fully JIT-traceable.

    Parameters
    ----------
    theta_i   : (D,) -- [P_th_2 [W], k_SERCA_base [min^-1]]
    x0        : (STATE_DIM,) = (6,) -- initial state
    controls  : (T, CTRL_DIM) -- [hub_Power_W, hub_Lactate_mmolL, hub_Glucose] per minute
    obs_params: NMv4ObsParams

    Returns
    -------
    y_pred : (T, OBS_DIM) -- [EMG_mV, SmO2_pct] predicted at each minute
    """
    nm_params = DEFAULT_V4_PARAMS._replace(
        P_th_2       = theta_i[0],
        k_SERCA_base = theta_i[1],
    )

    def _step(x_carry, u_t):
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(nm_v4_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(1.0),
            dt0       = jnp.float32(0.1),
            y0        = x_carry,
            args      = (nm_params, u_t),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 32,
        )
        x_next = sol.ys[0]
        y_t    = h_nm_v4(x_next, obs_params, nm_params)
        return x_next, y_t

    _, y_pred = jax.lax.scan(_step, x0, controls)
    return y_pred   # (T, OBS_DIM)


# -- NumPyro NLME model --------------------------------------------------------

def nm_v4_nlme_model(
    observations:  "jax.Array | None",   # (N_total, OBS_DIM) or None
    controls_list: jax.Array,            # (N_subjects, T, CTRL_DIM)
    subject_ids:   jax.Array,            # (N_total,) int
    obs_idx:       jax.Array,            # (N_total,) int
    n_subjects:    int,
) -> None:
    """
    NumPyro hierarchical NLME model for NM V4.0 personalisation.

    Non-centred (Matt trick):
      eta_raw_i ~ N(0, I_D)
      eta_i     = L_Omega * (scale_eta * eta_raw_i)
      theta_i   = theta_pop * exp(eta_i)

    Likelihood:
      y_obs ~ Normal(h_nm_v4(ODE(theta_i, controls_i)[t], obs_params), sigma_obs)
    """
    if not _NUMPYRO_OK:
        raise RuntimeError(
            "numpyro is required for NMv4NLME. "
            "Install: pip install numpyro>=0.15"
        )

    # -- Population-level log parameters --------------------------------------
    theta_pop_log = numpyro.sample(
        "theta_pop_log",
        dist.Normal(
            jnp.array([_LOG_MEAN_P_TH_2, _LOG_MEAN_K_SERCA], dtype=jnp.float32),
            jnp.array([_LOG_SD_P_TH_2,   _LOG_SD_K_SERCA],   dtype=jnp.float32),
        ),
    )   # (D,)

    # -- Between-subject covariance (LKJCholesky; concentration=2.0) ----------
    Omega_chol = numpyro.sample(
        "Omega_chol",
        dist.LKJCholesky(D, concentration=2.0),
    )   # (D, D)

    # -- Between-subject scale (half-normal priors on eta SDs) ----------------
    scale_eta = numpyro.sample(
        "scale_eta",
        dist.HalfNormal(jnp.array([_LOG_SD_P_TH_2, _LOG_SD_K_SERCA], dtype=jnp.float32)),
    )   # (D,)

    # -- Observation noise (EMG, SmO2) ----------------------------------------
    sigma_emg  = numpyro.sample("sigma_emg",  dist.HalfNormal(jnp.float32(1.0)))
    sigma_smo2 = numpyro.sample("sigma_smo2", dist.HalfNormal(jnp.float32(1.0)))
    sigma_obs  = jnp.stack([sigma_emg, sigma_smo2])   # (2,)

    # -- Subject-level parameters (Matt trick / non-centred) ------------------
    with numpyro.plate("subjects", n_subjects):
        eta_raw = numpyro.sample(
            "eta_raw",
            dist.Normal(
                jnp.zeros(D, dtype=jnp.float32),
                jnp.ones(D, dtype=jnp.float32),
            ),
        )   # (N_subjects, D)

        eta_scaled = eta_raw * scale_eta[None, :]               # (N_subjects, D)
        eta_corr   = jnp.einsum("ij,sj->si", Omega_chol, eta_scaled)  # (N_subjects, D)

        theta_i_log = theta_pop_log[None, :] + eta_corr         # (N_subjects, D)
        theta_i     = jnp.exp(theta_i_log)                      # positive

    # -- Forward simulation (vmap over subjects) --------------------------------
    obs_params = DEFAULT_V4_OBS_PARAMS

    def _sim_subject(theta_s, controls_s):
        return _forward_simulate_v4(theta_s, X0_NM_V4, controls_s, obs_params)

    y_pred_all = jax.vmap(_sim_subject)(theta_i, controls_list)  # (N_subj, T, OBS_DIM)

    # -- Likelihood -----------------------------------------------------------
    if observations is not None:
        y_pred_flat = y_pred_all[subject_ids, obs_idx, :]   # (N_total, OBS_DIM)

        with numpyro.plate("observations", observations.shape[0]):
            numpyro.sample(
                "y_obs",
                dist.Normal(y_pred_flat, sigma_obs[None, :]),
                obs=observations,
            )


# -- Public class --------------------------------------------------------------

class NMv4NLME:
    """
    L3 NLME personalisation wrapper for the Neuromuscular Tissue V4.0 slice.

    Personalises P_th_2 and k_SERCA_base from intra-session EMG + SmO2
    time-series via SVI (AutoLowRankMultivariateNormal). NUTS validation
    available for R-hat / ESS diagnostics.

    Fail-Loud contract
    ------------------
    RuntimeError on NaN ELBO.
    RuntimeError if numpyro absent.
    """

    def __init__(self) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError(
                "numpyro is required for NMv4NLME. "
                "Install: pip install numpyro>=0.15"
            )

    def fit_svi(
        self,
        data:     dict,
        n_steps:  int   = 30_000,
        lr:       float = 1e-3,
        rank:     int   = 5,
        rng_key:  int   = 42,
    ) -> dict:
        """
        Fit NLME model via SVI.

        Parameters
        ----------
        data : dict with keys:
               "observations"  : (N_total, OBS_DIM) float32 -- [EMG mV, SmO2 %]
               "controls_list" : (N_subjects, T, CTRL_DIM) float32
               "subject_ids"   : (N_total,) int32
               "obs_idx"       : (N_total,) int32
               "n_subjects"    : int

        Returns
        -------
        dict: "svi_result", "elbo_history", "params", "guide"
        """
        key   = jax.random.PRNGKey(rng_key)
        guide = AutoLowRankMultivariateNormal(nm_v4_nlme_model, rank=rank)
        optim = optax.adam(lr)
        svi   = SVI(nm_v4_nlme_model, guide, optim, loss=Trace_ELBO())

        svi_state = svi.init(
            key,
            observations  = data["observations"],
            controls_list = data["controls_list"],
            subject_ids   = data["subject_ids"],
            obs_idx       = data["obs_idx"],
            n_subjects    = data["n_subjects"],
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
            )
            elbo_history.append(float(loss))

            if jnp.isnan(jnp.float32(loss)):
                raise RuntimeError(
                    f"NMv4NLME.fit_svi: NaN ELBO at step {step}. "
                    "Check data scaling, prior means, and learning rate."
                )

            if step % 5_000 == 0:
                logger.info("NMv4NLME SVI step %d/%d -- ELBO: %.2f", step, n_steps, loss)

        params = svi.get_params(svi_state)
        logger.info("NMv4NLME SVI complete -- final ELBO: %.2f", elbo_history[-1])

        return {
            "svi_result":   svi_state,
            "elbo_history": elbo_history,
            "params":       params,
            "guide":        guide,
        }

    def validate_with_nuts(
        self,
        data:      dict,
        n_chains:  int = 4,
        n_warmup:  int = 500,
        n_samples: int = 500,
        rng_key:   int = 0,
    ) -> dict:
        """NUTS validation for R-hat / ESS diagnostics."""
        key = jax.random.PRNGKey(rng_key)
        nuts_kernel = NUTS(nm_v4_nlme_model)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup  = n_warmup,
            num_samples = n_samples,
            num_chains  = n_chains,
            progress_bar= False,
        )
        mcmc.run(
            key,
            observations  = data["observations"],
            controls_list = data["controls_list"],
            subject_ids   = data["subject_ids"],
            obs_idx       = data["obs_idx"],
            n_subjects    = data["n_subjects"],
        )

        samples     = mcmc.get_samples()
        extra       = mcmc.get_extra_fields()
        divergences = int(jnp.sum(extra.get("diverging", jnp.array(0))))

        from numpyro.diagnostics import summary
        summ      = summary(samples, prob=0.95)
        r_hats    = [float(summ[k]["r_hat"]) for k in summ]
        ess_vals  = [float(summ[k]["n_eff"]) for k in summ]

        r_hat_max = max(r_hats)  if r_hats  else float("nan")
        ess_min   = min(ess_vals) if ess_vals else float("nan")

        logger.info(
            "NMv4NLME NUTS -- max R-hat: %.4f, min ESS: %.1f, divergences: %d",
            r_hat_max, ess_min, divergences,
        )
        return {
            "r_hat":       r_hat_max,
            "ess":         ess_min,
            "divergences": divergences,
            "samples":     samples,
        }

    def get_nm_params(
        self,
        svi_result: dict,
        subject_id: int,
        n_samples:  int = 200,
        rng_key:    int = 1,
    ) -> NMv4Params:
        """
        Draw posterior mean personalised NMv4Params for subject_id.

        Returns NMv4Params with P_th_2 and k_SERCA_base replaced by
        posterior means for the given subject.
        """
        key   = jax.random.PRNGKey(rng_key)
        guide = svi_result["guide"]

        posterior = guide.sample_posterior(
            key, svi_result["params"], sample_shape=(n_samples,)
        )

        eta_raw_s   = posterior.get("eta_raw", jnp.zeros((n_samples, 1, D)))
        scale_s     = posterior.get("scale_eta", jnp.ones((n_samples, D)) * 0.20)
        Omega_s     = posterior.get("Omega_chol", jnp.tile(jnp.eye(D)[None], (n_samples, 1, 1)))
        theta_pop_s = posterior.get(
            "theta_pop_log",
            jnp.array([[_LOG_MEAN_P_TH_2, _LOG_MEAN_K_SERCA]])
        )

        def _compute_theta_i(eta_r, sc, Om, tp):
            eta   = eta_r[subject_id] * sc
            eta_c = Om @ eta
            return jnp.exp(tp + eta_c)

        theta_samples = jax.vmap(_compute_theta_i)(eta_raw_s, scale_s, Omega_s, theta_pop_s)
        theta_mean    = jnp.mean(theta_samples, axis=0)   # (D,)

        P_th_2_i  = float(theta_mean[0])
        k_SERCA_i = float(theta_mean[1])

        logger.info(
            "NMv4NLME posterior (subject %d): P_th_2=%.1f W, k_SERCA=%.3f min^-1",
            subject_id, P_th_2_i, k_SERCA_i,
        )

        return DEFAULT_V4_PARAMS._replace(P_th_2=P_th_2_i, k_SERCA_base=k_SERCA_i)

    @staticmethod
    def cold_start_params(actn3_genotype: str = "RX") -> NMv4Params:
        """
        Return population-mean NMv4Params with weak ACTN3 prior shift.

        ACTN3 R577X -> P_th_2 prior (genetics = weak prior, <=0.5*sigma).
          RR (power): P_th_2 = 237.5 W  (-0.5*sigma, sigma=25W)
          RX (mixed): P_th_2 = 250.0 W  (population mean)
          XX (endur): P_th_2 = 262.5 W  (+0.5*sigma)
        """
        if actn3_genotype == "RR":
            p_th_2 = 237.5
        elif actn3_genotype == "XX":
            p_th_2 = 262.5
        else:
            p_th_2 = 250.0

        return DEFAULT_V4_PARAMS._replace(P_th_2=p_th_2)
