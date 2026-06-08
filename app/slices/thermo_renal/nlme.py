"""
app/slices/thermo_renal/nlme.py  —  L3 NLME, Thermo-Renal V2

Personalises D=2 parameters per individual from wearable Core_Temp and
body-weight drop trajectories.

Personalised parameters
  sweat_sensitivity [L/min per exp(°C) above 37]  — thermoregulatory gain
  sweat_na_conc     [mmol/L]                       — sweat sodium (20–80 range)

Non-centred parametrisation (Matt trick; Betancourt & Girolami 2015)
  eta_raw_i ~ N(0, I_D)
  eta_i     = L_Omega . eta_raw_i
  theta_i   = theta_pop . exp(eta_i)   [positive, log-normal]

Population priors (log-space)
  sweat_sensitivity ~ LogN(log 0.004, 0.35²)   — Sawka 2007; ~40% population CV
  sweat_na_conc     ~ LogN(log 50.0,  0.35²)   — Montain 1992; salty vs light

Fail-Loud: ELBO NaN → RuntimeError. numpyro absent → RuntimeError.

References
  Sawka M.N. et al. (2007) Med Sci Sports Exerc 39(2):377–390
  Montain S.J., Coyle E.F. (1992) J Appl Physiol
  Betancourt M., Girolami M. (2015) arXiv:1312.0906
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
    import optax as _optax
    _NUMPYRO_OK = True
except ImportError:
    numpyro = None      # type: ignore[assignment]
    dist    = None      # type: ignore[assignment]
    SVI     = None      # type: ignore[assignment]
    _NUMPYRO_OK = False

from app.slices.thermo_renal.ode import (
    ThermoRenalParams,
    DEFAULT_TR_PARAMS,
    X0_TR_DEFAULT,
    STATE_DIM,
    CTRL_DIM,
    thermo_renal_ode,
)
from app.slices.thermo_renal.observation import (
    TRObsParams,
    DEFAULT_TR_OBS_PARAMS,
    h_tr,
)

logger = logging.getLogger(__name__)

# ── D=2 parameter names ───────────────────────────────────────────────────────
PARAM_NAMES: tuple[str, ...] = ("sweat_sensitivity", "sweat_na_conc")
D: int = len(PARAM_NAMES)

_LOG_MEAN: dict[str, float] = {
    "sweat_sensitivity": float(jnp.log(jnp.float32(0.004))),
    "sweat_na_conc":     float(jnp.log(jnp.float32(50.0))),
}
_LOG_SD: dict[str, float] = {
    "sweat_sensitivity": 0.35,
    "sweat_na_conc":     0.35,
}


# ── Forward simulation (1-minute steps, lax.scan) ────────────────────────────

def _forward_simulate(
    theta_i:    jax.Array,
    x0:         jax.Array,
    controls:   jax.Array,
    obs_params: TRObsParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Simulate T 1-minute steps with personalised theta_i=[sweat_sensitivity, sweat_na_conc].

    Returns
    -------
    y_pred  : (T, OBS_DIM=2)  — [Core_Temp_obs, BW_drop]
    x_traj  : (T, STATE_DIM=5)
    """
    tr_params = DEFAULT_TR_PARAMS._replace(
        sweat_sensitivity = theta_i[0],
        sweat_na_conc     = theta_i[1],
    )

    def _step(x_carry, u_t):
        sol = diffrax.diffeqsolve(
            terms     = diffrax.ODETerm(thermo_renal_ode),
            solver    = diffrax.Tsit5(),
            t0        = jnp.float32(0.0),
            t1        = jnp.float32(1.0),
            dt0       = jnp.float32(0.1),
            y0        = x_carry,
            args      = (tr_params, u_t),
            saveat    = diffrax.SaveAt(t1=True),
            max_steps = 512,
        )
        x_next = jnp.maximum(sol.ys[0], jnp.float32(0.0))
        y_t    = h_tr(x_next, obs_params, tr_params)
        return x_next, (y_t, x_next)

    _, (y_pred, x_traj) = jax.lax.scan(_step, x0, controls)
    return y_pred, x_traj


# ── NumPyro NLME model ────────────────────────────────────────────────────────

def thermo_renal_nlme_model(
    core_temp_obs:   "jax.Array | None",   # (N_total,) [°C]
    controls_list:   jax.Array,            # (N_subjects, T, CTRL_DIM)
    subject_ids:     jax.Array,            # (N_total,) int
    obs_idx:         jax.Array,            # (N_total,) int
    n_subjects:      int,
    bw_obs:          "jax.Array | None" = None,    # (N_bw,) [kg]
    bw_obs_idx:      "jax.Array | None" = None,
    bw_subject_ids:  "jax.Array | None" = None,
) -> None:
    """NumPyro NLME model — non-centred, trimodal likelihood."""
    if not _NUMPYRO_OK:
        raise RuntimeError("numpyro is required. Install: pip install numpyro>=0.15")

    theta_pop_log = numpyro.sample(
        "theta_pop_log",
        dist.Normal(
            jnp.array([_LOG_MEAN["sweat_sensitivity"], _LOG_MEAN["sweat_na_conc"]], dtype=jnp.float32),
            jnp.array([_LOG_SD["sweat_sensitivity"],   _LOG_SD["sweat_na_conc"]],   dtype=jnp.float32),
        ),
    )
    Omega_chol = numpyro.sample("Omega_chol", dist.LKJCholesky(D, concentration=2.0))
    scale_eta  = numpyro.sample("scale_eta",  dist.HalfNormal(jnp.array([0.35, 0.35], dtype=jnp.float32)))
    sigma_ct   = numpyro.sample("sigma_ct",   dist.HalfNormal(jnp.float32(0.3)))
    sigma_bw   = numpyro.sample("sigma_bw",   dist.HalfNormal(jnp.float32(0.1)))

    with numpyro.plate("subjects", n_subjects):
        eta_raw    = numpyro.sample("eta_raw", dist.Normal(jnp.zeros(D, dtype=jnp.float32), jnp.ones(D, dtype=jnp.float32)))
        eta_i      = eta_raw * scale_eta[None, :]
        eta_i_corr = jnp.einsum("ij,sj->si", Omega_chol, eta_i)
        theta_i    = jnp.exp(theta_pop_log[None, :] + eta_i_corr)

    def _sim(theta_s, controls_s):
        return _forward_simulate(theta_s, X0_TR_DEFAULT, controls_s, DEFAULT_TR_OBS_PARAMS)

    y_pred_all, _ = jax.vmap(_sim)(theta_i, controls_list)   # (N, T, 2)

    if core_temp_obs is not None:
        ct_pred = y_pred_all[subject_ids, obs_idx, 0]
        with numpyro.plate("ct_obs", core_temp_obs.shape[0]):
            numpyro.sample("y_ct", dist.Normal(ct_pred, sigma_ct), obs=core_temp_obs)

    if bw_obs is not None:
        bw_pred = y_pred_all[bw_subject_ids, bw_obs_idx, 1]
        with numpyro.plate("bw_obs_plate", bw_obs.shape[0]):
            numpyro.sample("y_bw", dist.Normal(bw_pred, sigma_bw), obs=bw_obs)


# ── Public class ──────────────────────────────────────────────────────────────

class ThermoRenalNLME:
    """
    L3 NLME for sweat_sensitivity and sweat_na_conc personalisation.

    Requires numpyro; raises RuntimeError on import failure or NaN ELBO.
    """

    def __init__(self) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError("numpyro is required. Install: pip install numpyro>=0.15")

    def fit_svi(
        self,
        data:    dict,
        n_steps: int   = 20_000,
        lr:      float = 1e-3,
        rank:    int   = 3,
        rng_key: int   = 42,
    ) -> dict:
        """SVI fit — AutoLowRankMultivariateNormal guide."""
        import optax
        key   = jax.random.PRNGKey(rng_key)
        guide = AutoLowRankMultivariateNormal(thermo_renal_nlme_model, rank=rank)
        svi   = SVI(thermo_renal_nlme_model, guide, optax.adam(lr), loss=Trace_ELBO())
        kw    = {k: data[k] for k in data}

        svi_state    = svi.init(key, **kw)
        elbo_history: list[float] = []

        for step in range(n_steps):
            svi_state, loss = svi.update(svi_state, **kw)
            elbo_history.append(float(loss))
            if jnp.isnan(jnp.float32(loss)):
                raise RuntimeError(f"ThermoRenalNLME: NaN ELBO at step {step}.")
            if step % 5_000 == 0:
                logger.info("TR-NLME SVI step %d/%d  ELBO=%.2f", step, n_steps, loss)

        return {"svi_result": svi_state, "elbo_history": elbo_history,
                "params": svi.get_params(svi_state), "guide": guide}

    def get_population_priors(self) -> dict[str, tuple[float, float]]:
        """Return {param_name: (log_mean, log_sd)} for the two NLME parameters."""
        return {k: (_LOG_MEAN[k], _LOG_SD[k]) for k in PARAM_NAMES}
