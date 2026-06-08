"""
app/slices/neural_cognitive/nlme.py  — REMASTER v2.0

L3 NLME Population Layer — Neural/Cognitive Slice

Personalises 2 kinetic parameters per individual from RPE and PVT time-series
using Nonlinear Mixed-Effects modelling (NumPyro).

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015)
────────────────────────────────────────────────────────────────────
    η_raw_i ~ N(0, I_D)
    η_i     = L_Ω · η_raw_i               [correlated random effects]
    θ_i     = θ_pop · exp(η_i)            [log-space; both params strictly positive]

Personalised parameters (D = 2)
────────────────────────────────

    CYP1A2_clearance_rate  [h⁻¹; population mean = ln(2)/5 ≈ 0.139]
        — Caffeine half-life is STRICTLY determined by CYP1A2 genotype
          (rs762551, *1F vs *1C/*1D alleles).  Population range: 2.5 h – 10 h.
          Observable from the plasma caffeine decay curve after a known dose,
          or indirectly from PVT_Lapses time-to-recovery after caffeine intake.
          Prior: LogN(log(0.139), 0.40²) — 40% CV in log-space, covers 2.5–10 h.

    dopamine_resilience  [au; population mean = 1.0]
        — COMT Val158Met modulates DA degradation rate.  Observable as the
          slope of RPE rise vs work-rate over multiple sessions (low-resilience
          individuals show steeper RPE at the same absolute load across sessions
          with varying 5-HT/DA divergence).
          Prior: LogN(log 1.0, 0.20²) — 20% population CV.
          COMT Val158Met shifts prior mean ≤ 0.5·σ (CLAUDE.md §3.2).

Identifiability rationale
──────────────────────────
CYP1A2_clearance_rate: structurally identifiable from the caffeine arm —
    PVT_Lapses drop (via Active_Adenosine) encodes the rate at which caffeine
    clears and A1/A2A receptor occupancy recovers.  Two sessions with different
    caffeine doses and intervals constrain the rate.

dopamine_resilience: identifiable from repeated RPE measurements across
    sessions at the same external load but varying monoamine divergence
    (5-HT accumulates proportionally to duration; DA resilience shifts the slope).

COMT Val158Met prior shift (CLAUDE.md §3.2: ≤ 0.5·σ)
────────────────────────────────────────────────────
    Val/Val → mean × 0.90  (high COMT, faster DA catabolism, lower resilience)
    Val/Met → mean × 1.00  (population reference)
    Met/Met → mean × 1.10  (low COMT, slower catabolism, higher resilience)
    Max shift = 0.10 = 0.5 × σ_prior (0.20)  ✓

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
    Betancourt M., Girolami M. (2015) arXiv:1312.0906   [non-centred param]
    Meeusen R. et al. (2018) Exerc Sport Sci Rev 46(1)   [COMT & fatigue]
    Nehlig A. (2010) Neurosci Biobehav Rev 35(2):430     [CYP1A2 caffeine]
    Sachse C. et al. (2003) Pharmacogenet 13(8):481      [CYP1A2 genotypes]
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

from app.slices.neural_cognitive.ode import (
    NeuralCognitiveParams,
    DEFAULT_NC_PARAMS,
    X0_NC_DEFAULT,
    STATE_DIM,
    OBS_DIM,
    CTRL_DIM,
    neural_cognitive_ode,
)
from app.slices.neural_cognitive.observation import (
    NeuralCogObsParams,
    DEFAULT_NC_OBS_PARAMS,
    h_nc,
)

logger = logging.getLogger(__name__)

# NLME dimensions
D: int = 2   # [CYP1A2_clearance_rate, dopamine_resilience]

# Population prior means (log-space: θ_pop = log of physical mean)
_LOG_CYP1A2_POP:  float = math.log(0.139)   # log(ln(2)/5h)
_LOG_DA_RES_POP:  float = 0.0               # log(1.0)

# Between-subject SDs in log-space (CV of the physical parameter)
_SD_CYP1A2:   float = 0.40   # 40% log-CV; covers t½ ∈ [2.5, 10] h at ±2σ
_SD_DA_RES:   float = 0.20   # 20% log-CV; COMT-driven variability

# COMT allele prior shifts (cap at 0.5·σ_prior = 0.10)
_COMT_FACTORS: dict[str, float] = {
    "Val/Val": 0.90,
    "Val/Met": 1.00,
    "Met/Met": 1.10,
}


# ── Forward simulation ────────────────────────────────────────────────────────

def _forward_simulate(
    params:    NeuralCognitiveParams,
    u_seq:     jax.Array,
    dt_hours:  float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulate the 7-state neural/cognitive ODE over T time steps.

    Parameters
    ----------
    params    : NeuralCognitiveParams (personalised θ_i applied)
    u_seq     : shape (T, CTRL_DIM) = (T, 8)
    dt_hours  : float — integration window per step [h]

    Returns
    -------
    y_pred : shape (T, OBS_DIM) = (T, 2)
    x_traj : shape (T, STATE_DIM) = (T, 7)
    """
    def _step(x_carry, u_t):
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(neural_cognitive_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(dt_hours),
            dt0       = jnp.float32(0.05),
            y0        = x_carry,
            args      = (params, u_t),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 512,
        )
        x_next = sol.ys[0]
        y_t    = h_nc(x_next, DEFAULT_NC_OBS_PARAMS)
        return x_next, (y_t, x_next)

    _, (y_pred, x_traj) = jax.lax.scan(_step, X0_NC_DEFAULT, u_seq)
    return y_pred, x_traj


# ── NumPyro NLME model ────────────────────────────────────────────────────────

def neural_cognitive_nlme_model(
    u_sequences:  jax.Array,   # (N_subjects, T, CTRL_DIM)
    observations: jax.Array,   # (N_subjects, T, OBS_DIM); NaN = missing
    subject_ids:  jax.Array,   # (N_subjects,) int
    n_subjects:   int,
    dt_hours:     float = 1.0,
) -> None:
    """
    NumPyro NLME model — personalises CYP1A2_clearance_rate and
    dopamine_resilience per subject from RPE + PVT_Lapses time-series.

    Non-centred parametrisation (Matt trick):
        θ_pop  = [log(cyp1a2_pop), log(da_res_pop)]
        Ω_chol ~ LKJCholesky(D=2, concentration=2.0)
        η_raw  ~ N(0, I_D)
        η_i    = η_raw @ Ω_chol.T
        θ_i    = θ_pop + η_i   (log-space)
        cyp1a2_i = exp(θ_i[0])
        da_res_i = exp(θ_i[1])

    Likelihood: Normal(h_nc(x_t), σ_resid) per channel.
    Missing (NaN) observations masked from likelihood.
    """
    if not _NUMPYRO_OK:
        raise RuntimeError(
            "numpyro>=0.15 required for neural_cognitive_nlme_model. "
            "Install: pip install numpyro"
        )

    θ_pop = numpyro.sample(
        "θ_pop",
        dist.Normal(
            jnp.array([_LOG_CYP1A2_POP, _LOG_DA_RES_POP], dtype=jnp.float32),
            jnp.array([_SD_CYP1A2,      _SD_DA_RES],       dtype=jnp.float32),
        ),
    )

    Ω_chol = numpyro.sample("Ω_chol", dist.LKJCholesky(D, concentration=2.0))

    σ_resid = numpyro.sample(
        "σ_resid",
        dist.HalfNormal(jnp.array([1.0, 1.5], dtype=jnp.float32)),
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
        cyp1a2  = jnp.exp(θ_i[s, 0])
        da_res  = jnp.exp(θ_i[s, 1])
        params_s = DEFAULT_NC_PARAMS._replace(
            CYP1A2_clearance_rate = cyp1a2,
            dopamine_resilience   = da_res,
        )
        y_pred, _ = _forward_simulate(params_s, u_sequences[s], dt_hours)
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

class NeuralCognitiveNLME:
    """
    L3 NLME Population Layer — Neural/Cognitive Slice.

    Personalises CYP1A2_clearance_rate and dopamine_resilience per individual
    from RPE + PVT_Lapses time-series (SVI production; NUTS validation).

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
        dt_hours:     float = 1.0,
        rng_key:      jax.Array | None = None,
    ) -> Any:
        """
        Fit NLME with SVI (AutoLowRankMultivariateNormal guide).

        Raises RuntimeError if ELBO is NaN.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        subject_ids = jnp.arange(n_subjects, dtype=jnp.int32)
        guide       = AutoLowRankMultivariateNormal(
            neural_cognitive_nlme_model, rank=self.rank
        )
        optimizer = numpyro.optim.optax_to_numpyro(optax.adam(self.lr))
        svi       = SVI(neural_cognitive_nlme_model, guide, optimizer, loss=Trace_ELBO())

        svi_state = svi.init(
            rng_key, u_sequences, observations, subject_ids, n_subjects, dt_hours,
        )
        for step in range(self.n_steps_svi):
            svi_state, loss = svi.update(
                svi_state, u_sequences, observations, subject_ids, n_subjects, dt_hours,
            )
            if jnp.isnan(loss):
                raise RuntimeError(
                    f"NeuralCognitiveNLME.fit_svi: ELBO NaN at step {step}. "
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
        dt_hours:     float = 1.0,
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
            NUTS(neural_cognitive_nlme_model),
            num_warmup  = num_warmup,
            num_samples = num_samples,
            num_chains  = num_chains,
            progress_bar = False,
        )
        mcmc.run(
            rng_key, u_sequences, observations, subject_ids, n_subjects, dt_hours,
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
        comt_allele:      str  = "Val/Met",
        cyp1a2_phenotype: str  = "normal",
        genotype:         dict | None = None,
    ) -> NeuralCognitiveParams:
        """
        Return population-mean NeuralCognitiveParams with weak genetic prior shifts.

        COMT Val158Met → dopamine_resilience shift (≤ 0.5·σ; CLAUDE.md §3.2).
        CYP1A2 phenotype → CYP1A2_clearance_rate prior.
        """
        genotype = genotype or {}

        allele   = genotype.get("COMT_Val158Met",   comt_allele)
        da_res   = _COMT_FACTORS.get(allele, 1.00)

        pheno    = genotype.get("CYP1A2_phenotype", cyp1a2_phenotype)
        from app.slices.neural_cognitive.envelope import _CYP1A2_CLEARANCE
        cyp_rate = _CYP1A2_CLEARANCE.get(pheno, 0.139)

        return DEFAULT_NC_PARAMS._replace(
            dopamine_resilience   = da_res,
            CYP1A2_clearance_rate = cyp_rate,
        )

    @staticmethod
    def get_personalised_params(result: Any, subject_id: int) -> NeuralCognitiveParams:
        """Extract personalised NeuralCognitiveParams from SVIResult."""
        try:
            means   = result.params.get("auto_loc", None)
            if means is None:
                logger.warning("SVIResult has no 'auto_loc' — cold-start returned.")
                return DEFAULT_NC_PARAMS
            cyp_rate = float(jnp.exp(means[0]))
            da_res   = float(jnp.exp(means[1]))
            return DEFAULT_NC_PARAMS._replace(
                CYP1A2_clearance_rate = max(0.05, min(0.50, cyp_rate)),
                dopamine_resilience   = max(0.50, min(2.00, da_res)),
            )
        except Exception as exc:
            logger.warning(
                "get_personalised_params failed for subject %d: %s — cold-start.",
                subject_id, exc,
            )
            return DEFAULT_NC_PARAMS
