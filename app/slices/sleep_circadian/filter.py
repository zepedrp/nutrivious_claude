"""
app/slices/sleep_circadian/filter.py  —  L4 UKF State Filter V2.0

Unscented Kalman Filter for the 5-state sleep-circadian system.
dt_hours = 0.5 h per step (half-hour resolution to capture intraday dynamics).

UKF parametrisation (Merwe & Wan 2000)
────────────────────────────────────────
  α = 0.10   MANDATORY (wider spread — circadian oscillator is strongly nonlinear)
  β = 2.0
  κ = 0.0
  λ = α²(n+κ) − n   with n = STATE_DIM = 5
  2n+1 = 11 sigma points

Fail-Loud contract
──────────────────
  • NaN in posterior_mean → RuntimeError("SleepUKFv2: filter divergence")
  • jnp.maximum(x, 0) applied to posterior_mean (concentrations non-negative)
  • Negative diagonal covariance → RuntimeError("SleepUKFv2: negative variance")
  • min_eigval check: raises if min(eigval(P)) < −1e-4

References
──────────
  Merwe R. van der & Wan E.A. (2000) Proc. ASSPCC
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.sleep_circadian.ode import (
    IDX_SCN_X, IDX_SCN_Y, IDX_ADENOSINE, IDX_MELATONIN, IDX_SWS,
    STATE_DIM, OBS_DIM,
    SleepParams, DEFAULT_SLEEP_PARAMS,
    HubInputs, DEFAULT_HUBS,
    integrate_hours, initial_state,
)
from app.slices.sleep_circadian.observation import (
    SleepObsParams, DEFAULT_OBS_PARAMS,
    h_sleep, observation_noise_R,
)

_LOG = logging.getLogger(__name__)

# ── UKF constants ──────────────────────────────────────────────────────────────

_ALPHA  = 0.10     # MANDATORY per directive
_BETA   = 2.0
_KAPPA  = 0.0
_N      = STATE_DIM
_LAMBDA = _ALPHA ** 2 * (_N + _KAPPA) - _N

_W0_MEAN = _LAMBDA / (_N + _LAMBDA)
_WI_MEAN = 0.5 / (_N + _LAMBDA)
_W0_COV  = _W0_MEAN + (1.0 - _ALPHA ** 2 + _BETA)
_WI_COV  = _WI_MEAN

_WM = jnp.array([_W0_MEAN] + [_WI_MEAN] * (2 * _N))
_WC = jnp.array([_W0_COV]  + [_WI_COV]  * (2 * _N))

# dt per UKF step [hours]
DT_HOURS: float = 0.5


# ── Filter state container ─────────────────────────────────────────────────────

class SleepFilterState(NamedTuple):
    mean: jnp.ndarray   # (STATE_DIM,)
    cov:  jnp.ndarray   # (STATE_DIM, STATE_DIM)


# ── Process noise ──────────────────────────────────────────────────────────────

def default_process_noise_Q() -> jnp.ndarray:
    """
    Diagonal Q for a 0.5-h step.

    Scaled down vs daily filter — half-hour ODE step is accurate;
    noise reflects only residual unmodelled variability per half hour.
    """
    q_diag = jnp.array([
        5e-4,    # SCN_x   — circadian drift per 0.5 h (slow)
        5e-4,    # SCN_y
        2e-4,    # Adenosine — small intra-step noise
        2.0,     # Melatonin — 2 (pg/mL)² per 0.5 h (light/stress fluctuation)
        1e-4,    # SWS       — very small (fast ODE dynamics dominate)
    ])
    return jnp.diag(q_diag)


# ── Sigma-point ODE transition ─────────────────────────────────────────────────

def _transition_sigma(
    x: jnp.ndarray,
    params: SleepParams,
    hubs: HubInputs,
) -> jnp.ndarray:
    """Advance one sigma point by DT_HOURS via ODE."""
    x_next = integrate_hours(x, hubs=hubs, params=params,
                              t_hours=DT_HOURS, t0=0.0)
    # Non-negativity: concentrations cannot be negative
    x_next = jnp.maximum(x_next, 0.0)
    return x_next


# ── UKF predict step ──────────────────────────────────────────────────────────

def _ukf_predict(
    state: SleepFilterState,
    params: SleepParams,
    hubs: HubInputs,
    Q: jnp.ndarray,
) -> SleepFilterState:
    mu = state.mean
    P  = state.cov

    sqrt_P  = jnp.linalg.cholesky((_N + _LAMBDA) * P)
    sp_pos  = mu[None, :] + sqrt_P.T
    sp_neg  = mu[None, :] - sqrt_P.T
    sigmas  = jnp.concatenate([mu[None, :], sp_pos, sp_neg], axis=0)  # (11, 5)

    prop = jax.vmap(lambda x: _transition_sigma(x, params, hubs))
    sigmas_f = prop(sigmas)   # (11, 5)

    mu_pred = jnp.einsum("i,is->s", _WM, sigmas_f)
    diff    = sigmas_f - mu_pred[None, :]
    P_pred  = jnp.einsum("i,is,it->st", _WC, diff, diff) + Q
    P_pred  = 0.5 * (P_pred + P_pred.T)

    return SleepFilterState(mean=mu_pred, cov=P_pred)


# ── UKF update step ───────────────────────────────────────────────────────────

def _ukf_update(
    state: SleepFilterState,
    y_obs: jnp.ndarray,
    R: jnp.ndarray,
    obs_params: SleepObsParams,
) -> SleepFilterState:
    mu = state.mean
    P  = state.cov

    sqrt_P  = jnp.linalg.cholesky((_N + _LAMBDA) * P)
    sp_pos  = mu[None, :] + sqrt_P.T
    sp_neg  = mu[None, :] - sqrt_P.T
    sigmas  = jnp.concatenate([mu[None, :], sp_pos, sp_neg], axis=0)

    obs_fn   = jax.vmap(lambda x: h_sleep(x, obs_params=obs_params))
    y_sigmas = obs_fn(sigmas)   # (11, OBS_DIM)

    y_mean = jnp.einsum("i,io->o", _WM, y_sigmas)
    dy     = y_sigmas - y_mean[None, :]
    dx     = sigmas   - mu[None, :]

    S   = jnp.einsum("i,io,ij->oj", _WC, dy, dy) + R
    Pxy = jnp.einsum("i,is,io->so", _WC, dx, dy)

    K      = Pxy @ jnp.linalg.inv(S)
    innov  = y_obs - y_mean
    mu_new = mu + K @ innov
    P_new  = P - K @ S @ K.T
    P_new  = 0.5 * (P_new + P_new.T)

    return SleepFilterState(mean=mu_new, cov=P_new)


# ── Public single-step update (Fail-Loud) ─────────────────────────────────────

def update_state(
    state: SleepFilterState,
    y_obs: jnp.ndarray,
    params: SleepParams = DEFAULT_SLEEP_PARAMS,
    obs_params: SleepObsParams = DEFAULT_OBS_PARAMS,
    hubs: HubInputs = DEFAULT_HUBS,
    Q: jnp.ndarray | None = None,
    quality_flag: int = 0,
) -> SleepFilterState:
    """
    One full UKF cycle: predict (0.5-h ODE advance) + update (wearable obs).

    Fail-Loud:
      • NaN in posterior mean → RuntimeError
      • Negative diagonal variance → RuntimeError
      • min_eigval(P) < −1e-4 → RuntimeError
      • jnp.maximum(mean, 0) applied after update (concentrations ≥ 0)
    """
    if Q is None:
        Q = default_process_noise_Q()

    R = observation_noise_R(obs_params=obs_params, quality_flag=quality_flag)

    state_pred = _ukf_predict(state, params=params, hubs=hubs, Q=Q)
    state_post = _ukf_update(state_pred, y_obs=y_obs, R=R, obs_params=obs_params)

    # Non-negativity constraint on concentrations
    mu_clean = jnp.maximum(state_post.mean, 0.0)
    state_post = SleepFilterState(mean=mu_clean, cov=state_post.cov)

    mu = state_post.mean
    P  = state_post.cov

    if jnp.any(jnp.isnan(mu)):
        raise RuntimeError(
            f"SleepUKFv2: filter divergence — NaN in posterior mean. "
            f"prior_mean={state.mean}, y_obs={y_obs}."
        )

    if jnp.any(jnp.diag(P) < -1e-8):
        raise RuntimeError(
            f"SleepUKFv2: negative variance in posterior covariance. "
            f"diag(P)={jnp.diag(P)}."
        )

    eigvals = jnp.linalg.eigvalsh(P)
    if jnp.min(eigvals) < -1e-4:
        raise RuntimeError(
            f"SleepUKFv2: covariance not PSD. min_eigval={float(jnp.min(eigvals)):.2e}."
        )

    return state_post


# ── Initialisation ─────────────────────────────────────────────────────────────

def initial_filter_state(
    x0: jnp.ndarray | None = None,
    params: SleepParams = DEFAULT_SLEEP_PARAMS,
) -> SleepFilterState:
    """
    Build the initial SleepFilterState.

    P0 encodes prior uncertainty at first sync:
      SCN_x/y: ±0.30 (circadian phase unknown)
      Adenosine: ±0.15 (moderate uncertainty)
      Melatonin: ±20 pg/mL
      SWS: ±0.20
    """
    x_init = x0 if x0 is not None else initial_state(params)
    P0_diag = jnp.array([0.09, 0.09, 0.0225, 400.0, 0.04])
    return SleepFilterState(mean=x_init, cov=jnp.diag(P0_diag))


# ── Multi-step convenience wrapper ────────────────────────────────────────────

def filter_nsteps(
    y_obs_sequence: jnp.ndarray,         # (T, OBS_DIM)
    hubs_sequence: jnp.ndarray,          # (T, 3)  [lux, stress, cort]
    params: SleepParams = DEFAULT_SLEEP_PARAMS,
    obs_params: SleepObsParams = DEFAULT_OBS_PARAMS,
    x0: jnp.ndarray | None = None,
    quality_flags: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run UKF over T half-hour steps.

    Returns:
      means : (T, STATE_DIM) — posterior means
      covs  : (T, STATE_DIM, STATE_DIM) — posterior covariances

    Fail-Loud: raises immediately on divergence.
    """
    T = y_obs_sequence.shape[0]
    quality_flags = quality_flags if quality_flags is not None else jnp.zeros(T, dtype=int)

    state  = initial_filter_state(x0=x0, params=params)
    means, covs = [], []

    for t in range(T):
        hub_t = HubInputs(
            hub_light_lux       = float(hubs_sequence[t, 0]),
            hub_training_stress = float(hubs_sequence[t, 1]),
            hub_Cortisol_nmolL  = float(hubs_sequence[t, 2]),
        )
        state = update_state(
            state       = state,
            y_obs       = y_obs_sequence[t],
            params      = params,
            obs_params  = obs_params,
            hubs        = hub_t,
            quality_flag = int(quality_flags[t]),
        )
        means.append(state.mean)
        covs.append(state.cov)

    return jnp.stack(means), jnp.stack(covs)
