"""
app/slices/metabolic_reds/nlme.py — Module 13

L3 NLME Population Layer — Metabolic RED-S / Thyroid / Fatmax Slice

Personalises 2 kinetic parameters per individual from fT3, fT4, and RMR_Proxy
time-series using Nonlinear Mixed-Effects modelling (NumPyro).

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015)
────────────────────────────────────────────────────────────────────
    η_raw_i ~ N(0, I_D)
    η_i     = L_Ω · η_raw_i               [correlated random effects]
    θ_i     = θ_pop · exp(η_i)            [log-space; both strictly positive]

Personalised parameters (D = 2)
────────────────────────────────

    k_t3_conv  [day⁻¹; population mean = 0.2806]  — Deiodinase_Capacity
        Type-2 deiodinase (D2) converts T4 → T3.  The DIO2 Thr92Ala
        polymorphism (rs225014) reduces D2 activity by ~8% (Thr/Ala) to ~15%
        (Ala/Ala) relative to the Thr/Thr reference (Bianco & Larsen 2005;
        Canani et al. 2005).  Observable from the fT3/fT4 ratio in response
        to caloric restriction protocols.
        Prior: LogN(log(0.2806), 0.30²) — 30% log-CV.

    k_fatmax_relax  [day⁻¹; population mean = 1/21]  — Fatmax MFO adaptation
        Rate of mitochondrial biogenesis adaptation (Holloszy 2011).
        Anchored to Achten-Jeukendrup MFO inter-individual variability
        (mean 0.50 g/min, CV ~35%; Achten & Jeukendrup 2003).
        Observable from long-term RMR_Proxy tracking across energy flux cycles.
        Prior: LogN(log(1/21), 0.35²) — 35% log-CV.

DIO2 Thr92Ala prior shift (rs225014)
──────────────────────────────────────
    Thr/Thr → k_t3_conv × 1.00  (full D2 activity — reference)
    Thr/Ala → k_t3_conv × 0.92  (~8% reduction; heterozygous)
    Ala/Ala → k_t3_conv × 0.85  (~15% reduction; homozygous)
    Max shift (Thr→Ala) = ×0.92; (Thr→Ala/Ala) = ×0.85.

SVI backend
───────────
Guide  : AutoLowRankMultivariateNormal (rank=5)
Optim  : Optax Adam lr=1e-3
Loss   : Trace_ELBO
Budget : 30 000 steps

Fail-Loud contract
──────────────────
RuntimeError on NaN ELBO.
RuntimeError if numpyro absent.

References
──────────
    Achten J., Jeukendrup A.E. (2003) Sports Med 33(8):559–591
    Betancourt M., Girolami M. (2015) arXiv:1312.0906
    Bianco A.C., Larsen P.R. (2005) Thyroid 15(7):655–670
    Canani L.H. et al. (2005) J Clin Endocrinol Metab 90(7):3799–3804
    Holloszy J.O. (2011) J Appl Physiol 110(3):694–701
"""
from __future__ import annotations

import logging
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import diffrax

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO, NUTS, MCMC
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    import optax
    _NUMPYRO_OK = True
except ImportError:
    _NUMPYRO_OK = False

from app.slices.metabolic_reds.ode import (
    MetabolicRedsParams,
    DEFAULT_MR_PARAMS,
    X0_MR_DEFAULT,
    STATE_DIM,
    OBS_DIM,
    CTRL_DIM,
    metabolic_reds_ode,
)
from app.slices.metabolic_reds.observation import (
    DEFAULT_MR_OBS_PARAMS,
    h_mr,
)

logger = logging.getLogger(__name__)

# NLME dimensions
D: int = 2   # [k_t3_conv (Deiodinase_Capacity), k_fatmax_relax (Fatmax MFO)]

# Population prior means (log-space: θ_pop = log of physical mean)
_LOG_K_DEIO_POP:   float = math.log(0.2806)         # log(k_t3_conv population mean)
_LOG_K_FATMAX_POP: float = math.log(1.0 / 21.0)     # log(k_fatmax_relax population mean)

# Between-subject SDs in log-space
_SD_DEIO:   float = 0.30   # 30% log-CV; DIO2 polymorphism + individual D2 variation
_SD_FATMAX: float = 0.35   # 35% log-CV; Achten-Jeukendrup MFO inter-individual variability

# DIO2 Thr92Ala shifts on k_t3_conv (Bianco & Larsen 2005; Canani 2005)
_DIO2_FACTORS: dict[str, float] = {
    "Thr/Thr": 1.00,    # reference — full D2 activity
    "Thr/Ala": 0.92,    # ~8% D2 reduction (heterozygous)
    "Ala/Ala": 0.85,    # ~15% D2 reduction (homozygous)
}


# ── Forward simulation ────────────────────────────────────────────────────────

def _forward_simulate(
    params:  MetabolicRedsParams,
    u_seq:   jax.Array,
    dt_days: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulate the 5-state metabolic_reds ODE over T daily time steps.

    Parameters
    ----------
    params   : MetabolicRedsParams (personalised θ_i applied)
    u_seq    : shape (T, CTRL_DIM) = (T, 4)
    dt_days  : float — integration window per step [days]

    Returns
    -------
    y_pred : shape (T, OBS_DIM) = (T, 3)
    x_traj : shape (T, STATE_DIM) = (T, 5)
    """
    x0 = jnp.array(X0_MR_DEFAULT, dtype=jnp.float32)

    def _step(x_carry, u_t):
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(metabolic_reds_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(dt_days),
            dt0       = jnp.float32(0.1),
            y0        = x_carry,
            args      = (u_t, params),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 128,
        )
        x_next = sol.ys[0]
        y_t    = h_mr(x_next, DEFAULT_MR_OBS_PARAMS)
        return x_next, (y_t, x_next)

    _, (y_pred, x_traj) = jax.lax.scan(_step, x0, u_seq)
    return y_pred, x_traj


# ── NumPyro NLME model ────────────────────────────────────────────────────────

def metabolic_reds_nlme_model(
    u_sequences:  jax.Array,   # (N_subjects, T, CTRL_DIM)
    observations: jax.Array,   # (N_subjects, T, OBS_DIM); NaN = missing
    subject_ids:  jax.Array,   # (N_subjects,) int
    n_subjects:   int,
    dt_days:      float = 1.0,
) -> None:
    """
    NumPyro NLME model — personalises k_t3_conv (Deiodinase_Capacity) and
    k_fatmax_relax (Fatmax MFO adaptation) per subject from fT3/fT4/RMR time-series.

    Non-centred parametrisation (Matt trick):
        θ_pop  = [log(k_t3_conv_pop), log(k_fatmax_relax_pop)]
        Ω_chol ~ LKJCholesky(D=2, concentration=2.0)
        η_raw  ~ N(0, I_D)
        η_i    = η_raw @ Ω_chol.T
        θ_i    = θ_pop + η_i   (log-space)
        k_deio_i     = exp(θ_i[0])
        k_fatmax_i   = exp(θ_i[1])

    Likelihood: Normal(h_mr(x_t), σ_resid) per channel.
    Missing (NaN) observations masked from likelihood.
    """
    if not _NUMPYRO_OK:
        raise RuntimeError(
            "numpyro>=0.15 required for metabolic_reds_nlme_model. "
            "Install: pip install numpyro"
        )

    θ_pop = numpyro.sample(
        "θ_pop",
        dist.Normal(
            jnp.array([_LOG_K_DEIO_POP,   _LOG_K_FATMAX_POP], dtype=jnp.float32),
            jnp.array([_SD_DEIO,           _SD_FATMAX],         dtype=jnp.float32),
        ),
    )

    Ω_chol = numpyro.sample("Ω_chol", dist.LKJCholesky(D, concentration=2.0))

    # σ_resid per channel: fT3 (pmol/L), fT4 (pmol/L), RMR_Proxy (kcal/day)
    σ_resid = numpyro.sample(
        "σ_resid",
        dist.HalfNormal(jnp.array([0.5, 1.5, 150.0], dtype=jnp.float32)),
    )

    with numpyro.plate("subjects", n_subjects):
        η_raw = numpyro.sample(
            "η_raw",
            dist.Normal(
                jnp.zeros(D, dtype=jnp.float32),
                jnp.ones(D,  dtype=jnp.float32),
            ).expand([n_subjects, D]),
        )
        η_i = jnp.einsum("sd,ed->se", η_raw, Ω_chol)   # (N, D)
        θ_i = θ_pop[None, :] + η_i                      # (N, D) log-space

    def _subject_likelihood(s: int) -> jax.Array:
        k_deio   = jnp.exp(θ_i[s, 0])
        k_fatmax = jnp.exp(θ_i[s, 1])
        params_s = DEFAULT_MR_PARAMS._replace(
            k_t3_conv      = k_deio,
            k_fatmax_relax = k_fatmax,
        )
        y_pred, _ = _forward_simulate(params_s, u_sequences[s], dt_days)
        return y_pred   # (T, OBS_DIM)

    for s in range(n_subjects):
        y_pred_s = _subject_likelihood(s)
        y_obs_s  = observations[s]
        numpyro.sample(
            f"y_obs_{s}",
            dist.Normal(y_pred_s, σ_resid[None, :]).mask(~jnp.isnan(y_obs_s)),
            obs=jnp.nan_to_num(y_obs_s, nan=0.0),
        )


# ── Public NLME class ─────────────────────────────────────────────────────────

class MetabolicRedsNLME:
    """
    L3 NLME Population Layer — Metabolic RED-S / Thyroid / Fatmax Slice.

    Personalises k_t3_conv (Deiodinase_Capacity) and k_fatmax_relax per
    individual from fT3 / fT4 / RMR_Proxy time-series
    (SVI production; NUTS validation).

    Fail-Loud contract
    ──────────────────
    RuntimeError on NaN ELBO.
    RuntimeError if numpyro is not installed.
    """

    def __init__(
        self,
        n_steps_svi: int   = 30_000,
        lr:          float = 1e-3,
        rank:        int   = 5,
    ) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError("numpyro>=0.15 required. Install: pip install numpyro")
        self.n_steps_svi = n_steps_svi
        self.lr          = lr
        self.rank        = rank

    def fit_svi(
        self,
        u_sequences:  jax.Array,
        observations: jax.Array,
        n_subjects:   int,
        dt_days:      float = 1.0,
        rng_key:      jax.Array | None = None,
    ) -> Any:
        """Fit NLME with SVI. Raises RuntimeError if ELBO is NaN."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        subject_ids = jnp.arange(n_subjects, dtype=jnp.int32)
        guide       = AutoLowRankMultivariateNormal(
            metabolic_reds_nlme_model, rank=self.rank
        )
        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(self.lr))
        svi       = SVI(metabolic_reds_nlme_model, guide, optimizer, loss=Trace_ELBO())

        svi_state = svi.init(
            rng_key, u_sequences, observations, subject_ids, n_subjects, dt_days,
        )
        for step in range(self.n_steps_svi):
            svi_state, loss = svi.update(
                svi_state, u_sequences, observations, subject_ids, n_subjects, dt_days,
            )
            if jnp.isnan(loss):
                raise RuntimeError(
                    f"MetabolicRedsNLME.fit_svi: ELBO NaN at step {step}. "
                    "Check u_sequences for NaN/Inf and ODE convergence."
                )
            if step % 5000 == 0:
                logger.info("SVI step %d/%d — ELBO = %.4f", step, self.n_steps_svi, -loss)

        return svi.get_params(svi_state)

    def validate_with_nuts(
        self,
        u_sequences:  jax.Array,
        observations: jax.Array,
        n_subjects:   int,
        dt_days:      float = 1.0,
        num_chains:   int   = 4,
        num_warmup:   int   = 500,
        num_samples:  int   = 500,
        rng_key:      jax.Array | None = None,
    ) -> dict[str, Any]:
        """NUTS validation — returns r_hat, ess, divergences per parameter."""
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)

        subject_ids = jnp.arange(n_subjects, dtype=jnp.int32)
        mcmc = MCMC(
            NUTS(metabolic_reds_nlme_model),
            num_warmup   = num_warmup,
            num_samples  = num_samples,
            num_chains   = num_chains,
            progress_bar = False,
        )
        mcmc.run(
            rng_key, u_sequences, observations, subject_ids, n_subjects, dt_days,
        )
        summary: dict[str, Any] = {}
        try:
            from numpyro.diagnostics import summary as numpyro_summary
            summ = numpyro_summary(mcmc.get_samples(group_by_chain=True))
            summary = {
                "r_hat":       {k: float(v["r_hat"]) for k, v in summ.items()},
                "ess":         {k: float(v["n_eff"])  for k, v in summ.items()},
                "divergences": int(
                    mcmc.get_extra_fields().get("diverging", jnp.array(0)).sum()
                ),
            }
        except Exception as exc:
            logger.warning("NUTS summary failed: %s", exc)
        return summary

    @staticmethod
    def cold_start_params(
        dio2_genotype: str  = "Thr/Thr",
        genotype:      dict | None = None,
    ) -> MetabolicRedsParams:
        """
        Return population-mean MetabolicRedsParams with weak DIO2 genetic shift.

        DIO2 Thr92Ala (rs225014): shifts k_t3_conv by ±8–15% (Bianco & Larsen 2005).
        k_fatmax_relax: population mean (no strong genetic prior available).
        """
        genotype   = genotype or {}
        allele     = genotype.get("DIO2_Thr92Ala", dio2_genotype)
        dio2_shift = _DIO2_FACTORS.get(allele, 1.00)

        return DEFAULT_MR_PARAMS._replace(
            k_t3_conv = DEFAULT_MR_PARAMS.k_t3_conv * dio2_shift,
        )

    @staticmethod
    def get_personalised_params(result: Any, subject_id: int) -> MetabolicRedsParams:
        """Extract personalised MetabolicRedsParams from SVIResult."""
        try:
            means = result.params.get("auto_loc", None)
            if means is None:
                logger.warning("SVIResult has no 'auto_loc' — cold-start returned.")
                return DEFAULT_MR_PARAMS
            k_deio   = float(jnp.exp(means[0]))
            k_fatmax = float(jnp.exp(means[1]))
            return DEFAULT_MR_PARAMS._replace(
                k_t3_conv      = max(0.05, min(1.0,  k_deio)),
                k_fatmax_relax = max(0.01, min(0.25, k_fatmax)),
            )
        except Exception as exc:
            logger.warning(
                "get_personalised_params failed for subject %d: %s — cold-start.",
                subject_id, exc,
            )
            return DEFAULT_MR_PARAMS
