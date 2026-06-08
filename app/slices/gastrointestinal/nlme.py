"""
app/slices/gastrointestinal/nlme.py  --  GI Slice V3.0

L3 NLME (NumPyro, Fail-Loud).  D=2 personalised parameters:

  Vmax_glu  [g/min] -- SGLT1 glucose transporter capacity
                       prior: LogNormal(log(1.0), 0.35^2)
                       SLC5A1 genetics -> weak prior shift

  Vmax_fru  [g/min] -- GLUT5 fructose transporter capacity
                       prior: LogNormal(log(0.6), 0.35^2)
                       SLC2A5 genetics -> weak prior shift

Non-centred parametrisation (Matt trick).  LKJCholesky(D=2, concentration=2.0).
SVI: AutoLowRankMultivariateNormal rank=5, Adam 1e-3, 20k steps, Trace_ELBO.
Fail-Loud: RuntimeError on NaN ELBO or numpyro absence.
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
    numpyro = None  # type: ignore[assignment]
    dist    = None  # type: ignore[assignment]
    SVI     = None  # type: ignore[assignment]
    _NUMPYRO_OK = False

from app.slices.gastrointestinal.ode import (
    GIv3Params, DEFAULT_GI_PARAMS,
    X0_GI_DEFAULT, STATE_DIM, CTRL_DIM,
    gi_ode,
)
from app.slices.gastrointestinal.observation import (
    GIObsParams, DEFAULT_GI_OBS_PARAMS,
    OBS_DIM, h_gi,
)

logger = logging.getLogger(__name__)

D: int = 2

_LOG_MU_VMAX_GLU: float = float(jnp.log(jnp.float32(1.0)))
_LOG_MU_VMAX_FRU: float = float(jnp.log(jnp.float32(0.6)))
_SD_GLU:  float = 0.35
_SD_FRU:  float = 0.35


class SVIResult(NamedTuple):
    params:     dict
    losses:     jax.Array
    guide:      object
    svi_object: object


class NUTSResult(NamedTuple):
    r_hat:       dict[str, float]
    ess:         dict[str, float]
    divergences: int
    samples:     dict


def _integrate_dt(
    x: jax.Array, u: jax.Array, params: GIv3Params, dt_min: float
) -> jax.Array:
    sol = diffrax.diffeqsolve(
        terms    = diffrax.ODETerm(gi_ode),
        solver   = diffrax.Tsit5(),
        t0       = jnp.float32(0.0),
        t1       = jnp.float32(dt_min),
        dt0      = jnp.float32(0.1),
        y0       = x,
        args     = (params, u),
        saveat   = diffrax.SaveAt(t1=True),
        max_steps= 512,
    )
    return sol.ys[0]


def _forward_simulate(
    x0:         jax.Array,
    us:         jax.Array,
    dt_arr:     jax.Array,
    params:     GIv3Params,
    obs_params: GIObsParams,
) -> tuple[jax.Array, jax.Array]:
    def _step(carry, inp):
        u_t, dt_t = inp
        x_next = _integrate_dt(carry, u_t, params, float(dt_t))
        y_t    = h_gi(x_next, obs_params, params)
        return x_next, (y_t, x_next)

    _, (y_pred, x_traj) = jax.lax.scan(_step, x0, (us, dt_arr))
    return y_pred, x_traj


class GastrointestinalNLME:
    """
    L3 NLME for GI V3.0.  Personalises Vmax_glu (SGLT1) and Vmax_fru (GLUT5).
    Raises RuntimeError if numpyro absent.
    """

    def __init__(
        self,
        base_params: GIv3Params  | None = None,
        obs_params:  GIObsParams | None = None,
    ) -> None:
        if not _NUMPYRO_OK:
            raise RuntimeError(
                "GastrointestinalNLME requires numpyro. pip install numpyro"
            )
        self.base_params = base_params or DEFAULT_GI_PARAMS
        self.obs_params  = obs_params  or DEFAULT_GI_OBS_PARAMS

    def _model(
        self,
        subject_ids:  list[str],
        observations: list[jax.Array],
        controls:     list[jax.Array],
        dt_lists:     list[jax.Array],
    ) -> None:
        N = len(subject_ids)

        log_vglu_pop = numpyro.sample("log_Vmax_glu_pop",
                                      dist.Normal(_LOG_MU_VMAX_GLU, _SD_GLU))
        log_vfru_pop = numpyro.sample("log_Vmax_fru_pop",
                                      dist.Normal(_LOG_MU_VMAX_FRU, _SD_FRU))
        theta_pop = jnp.stack([log_vglu_pop, log_vfru_pop])

        scale_eta = numpyro.sample(
            "scale_eta",
            dist.HalfNormal(jnp.array([_SD_GLU, _SD_FRU])),
        )
        Omega_chol = numpyro.sample(
            "Omega_chol", dist.LKJCholesky(D, concentration=2.0)
        )
        L_Omega = Omega_chol * scale_eta[:, None]

        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(jnp.float32(1.0)))

        with numpyro.plate("subjects", N):
            eta_raw = numpyro.sample(
                "eta_raw", dist.Normal(jnp.zeros(D), jnp.ones(D))
            )
        eta_i = eta_raw @ L_Omega.T

        for i in range(N):
            log_th_i = theta_pop + eta_i[i]
            vg_i = jnp.exp(log_th_i[0])
            vf_i = jnp.exp(log_th_i[1])
            p_i = self.base_params._replace(
                Vmax_glu = float(vg_i),
                Vmax_fru = float(vf_i),
            )
            y_pred, _ = _forward_simulate(
                X0_GI_DEFAULT, controls[i], dt_lists[i], p_i, self.obs_params
            )
            y_obs = observations[i]
            mask  = jnp.isfinite(y_obs)
            with numpyro.plate(f"obs_{i}", y_obs.shape[0]):
                numpyro.sample(
                    f"y_{i}",
                    dist.Normal(y_pred, sigma_obs).mask(mask),
                    obs=y_obs,
                )

    def fit_svi(
        self,
        data:     list[dict],
        n_steps:  int   = 20_000,
        lr:       float = 1e-3,
        rank:     int   = 5,
        rng_seed: int   = 0,
    ) -> SVIResult:
        subject_ids  = [d["subject_id"] for d in data]
        observations = [jnp.asarray(d["observations"], dtype=jnp.float32) for d in data]
        controls     = [jnp.asarray(d["controls"],     dtype=jnp.float32) for d in data]
        dt_lists     = [jnp.asarray(
            d.get("dt_minutes", jnp.ones(d["observations"].shape[0])),
            dtype=jnp.float32,
        ) for d in data]

        args    = (subject_ids, observations, controls, dt_lists)
        guide   = AutoLowRankMultivariateNormal(self._model, rank=rank)
        svi_obj = SVI(self._model, guide, optax.adam(lr), loss=Trace_ELBO())
        rng     = jax.random.PRNGKey(rng_seed)
        state   = svi_obj.init(rng, *args)
        losses  = []

        for step in range(n_steps):
            state, loss = svi_obj.update(state, *args)
            lv = float(loss)
            if jnp.isnan(jnp.float32(lv)):
                raise RuntimeError(
                    f"GastrointestinalNLME.fit_svi: ELBO NaN at step {step}."
                )
            losses.append(lv)
            if step % 5_000 == 0:
                logger.info("GI NLME SVI step %d | ELBO: %.2f", step, -lv)

        return SVIResult(
            params=svi_obj.get_params(state),
            losses=jnp.array(losses),
            guide=guide,
            svi_object=svi_obj,
        )

    def cold_start_params(self) -> GIv3Params:
        return self.base_params._replace(Vmax_glu=1.0, Vmax_fru=0.6)
