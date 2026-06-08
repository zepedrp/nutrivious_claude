"""
app/slices/immune_repair/filter.py -- L4 UKF: Immune Repair  (4-state)

Unscented Kalman Filter for the 4-state:
    [Muscle_Damage_au, Macrophage_M1, Macrophage_M2, Interleukin_6_pgmL]

UKF parametrisation (van der Merwe & Wan 2000)
----------------------------------------------
    alpha = 0.10  MANDATORY for float32 stability with n=4.
                  n+lambda = alpha^2*(n+kappa) = 0.01*4 = 0.04
                  W0_mean  = lambda/(n+lambda) = (0.04-4)/0.04 = -99
                  With alpha=0.001: W0=-9999 -> catastrophic cancellation.
                  With alpha=0.10:  |W0|=99 << 1/eps_float32 -> safe.
    beta  = 2.0   Gaussian kurtosis prior
    kappa = 0.0
    n     = 4  ->  2n+1 = 9 sigma points

Prediction: 1-hour Tsit5 ODE integration.
Sigma-point propagation via jax.vmap over 9 ODE solves.

Observation: y in R^2 = [hsCRP_mg_L, CK_U_L]

Physical clamp: jnp.maximum(x, 0) after every update.
Jitter: 1e-3 added to diagonal of P_pred (guarantees PSD before Cholesky).

Fail-Loud:
    NaN in posterior_mean -> RuntimeError
    Negative diagonal cov -> RuntimeError

References
----------
    van der Merwe & Wan (2000) Proc. ASSPCC
    Tidball & Villalta (2010) Am J Physiol 298:C1173
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.immune_repair.ode import (
    IDX_DMG, IDX_M1, IDX_M2, IDX_IL6,
    STATE_DIM,
    ImmuneParams, DEFAULT_IMMUNE_PARAMS,
    zero_control, integrate_1h, initial_state,
)
from app.slices.immune_repair.observation import (
    ImmuneObsParams, DEFAULT_OBS_PARAMS,
    h_immune, observation_noise_R,
)

# ── UKF constants (alpha=0.10, n=4) ──────────────────────────────────────────

_ALPHA  = 0.10
_BETA   = 2.0
_KAPPA  = 0.0
_N      = STATE_DIM   # 4

_LAMBDA   = _ALPHA ** 2 * (_N + _KAPPA) - _N   # = 0.04 - 4 = -3.96
_N_LAMBDA = _N + _LAMBDA                         # = 0.04

_W0_MEAN = _LAMBDA / _N_LAMBDA                   # = -99.0
_WI_MEAN = 0.5 / _N_LAMBDA                       # = 12.5
_W0_COV  = _W0_MEAN + (1.0 - _ALPHA ** 2 + _BETA)  # = -99 + 2.99 = -96.01
_WI_COV  = _WI_MEAN

_WM = jnp.array([_W0_MEAN] + [_WI_MEAN] * (2 * _N))   # (9,)
_WC = jnp.array([_W0_COV]  + [_WI_COV]  * (2 * _N))


# ── Filter state ───────────────────────────────────────────────────────────────

class ImmuneFilterState(NamedTuple):
    mean: jnp.ndarray   # (4,)
    cov:  jnp.ndarray   # (4, 4)


# ── Process noise ──────────────────────────────────────────────────────────────

def default_process_noise_Q() -> jnp.ndarray:
    q_diag = jnp.array([
        1e-4,    # Muscle_Damage_au   [au^2/h]   slow structural change
        1e-4,    # Macrophage_M1      [au^2/h]   moderate dynamics
        1e-4,    # Macrophage_M2      [au^2/h]   slower than M1
        1.0,     # IL-6 pgmL         [pg^2/mL^2/h]  pulsatile, noisy
    ])
    return jnp.diag(q_diag)


# ── Physical clamp ─────────────────────────────────────────────────────────────

def _clamp_physical(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(x, 0.0)


# ── Sigma-point ODE transition ─────────────────────────────────────────────────

def _transition_sigma(
    x: jnp.ndarray,
    params: ImmuneParams,
    control_fn,
    t0: float,
) -> jnp.ndarray:
    x_next = integrate_1h(x, params=params, control_fn=control_fn, t0=t0)
    return _clamp_physical(x_next)


# ── UKF predict ───────────────────────────────────────────────────────────────

def _ukf_predict(
    state: ImmuneFilterState,
    params: ImmuneParams,
    Q: jnp.ndarray,
    control_fn,
    t0: float,
) -> ImmuneFilterState:
    mu = state.mean
    P  = state.cov

    # Jitter for PSD guarantee before Cholesky
    P_jit   = P + 1e-3 * jnp.eye(_N)
    sqrt_P  = jnp.linalg.cholesky(_N_LAMBDA * P_jit)
    sp_pos  = mu[None, :] + sqrt_P.T
    sp_neg  = mu[None, :] - sqrt_P.T
    sigmas  = jnp.concatenate([mu[None, :], sp_pos, sp_neg], axis=0)   # (9, 4)

    propagate  = jax.vmap(lambda x: _transition_sigma(x, params, control_fn, t0))
    sigmas_f   = propagate(sigmas)

    mu_pred = jnp.einsum("i,is->s", _WM, sigmas_f)
    diff    = sigmas_f - mu_pred[None, :]
    P_pred  = jnp.einsum("i,is,it->st", _WC, diff, diff) + Q
    P_pred  = 0.5 * (P_pred + P_pred.T)

    return ImmuneFilterState(mean=mu_pred, cov=P_pred)


# ── UKF update ────────────────────────────────────────────────────────────────

def _ukf_update(
    state: ImmuneFilterState,
    y_obs: jnp.ndarray,
    R: jnp.ndarray,
    obs_params: ImmuneObsParams,
) -> ImmuneFilterState:
    mu = state.mean
    P  = state.cov

    P_jit   = P + 1e-3 * jnp.eye(_N)
    sqrt_P  = jnp.linalg.cholesky(_N_LAMBDA * P_jit)
    sp_pos  = mu[None, :] + sqrt_P.T
    sp_neg  = mu[None, :] - sqrt_P.T
    sigmas  = jnp.concatenate([mu[None, :], sp_pos, sp_neg], axis=0)

    obs_fn   = jax.vmap(lambda x: h_immune(x, obs_params))
    y_sigmas = obs_fn(sigmas)   # (9, 2)

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

    mu_new = _clamp_physical(mu_new)

    return ImmuneFilterState(mean=mu_new, cov=P_new)


# ── Public API (Fail-Loud) ────────────────────────────────────────────────────

def update_state(
    state: ImmuneFilterState,
    y_obs: jnp.ndarray,
    params: ImmuneParams = DEFAULT_IMMUNE_PARAMS,
    obs_params: ImmuneObsParams = DEFAULT_OBS_PARAMS,
    Q: jnp.ndarray | None = None,
    quality_flags: tuple[int, int] = (0, 0),
    control_fn=None,
    t0: float = 0.0,
) -> ImmuneFilterState:
    """
    Full UKF cycle: 1-h Tsit5 ODE predict -> 2-channel observation update.
    Fail-Loud: NaN or negative diagonal cov -> RuntimeError.
    """
    if Q is None:
        Q = default_process_noise_Q()
    if control_fn is None:
        control_fn = zero_control

    R    = observation_noise_R(obs_params=obs_params, quality_flags=quality_flags)
    pred = _ukf_predict(state, params=params, Q=Q, control_fn=control_fn, t0=t0)
    post = _ukf_update(pred, y_obs=y_obs, R=R, obs_params=obs_params)

    mu = post.mean
    P  = post.cov

    if jnp.any(jnp.isnan(mu)):
        raise RuntimeError(
            "ImmuneUKF: divergence -- NaN in posterior mean. "
            f"Prior mean={state.mean}, y_obs={y_obs}. "
            "Check Q diagonal or integration stability."
        )

    if jnp.any(jnp.diag(P) < 0.0):
        raise RuntimeError(
            "ImmuneUKF: negative variance in posterior covariance. "
            f"diag(P)={jnp.diag(P)}."
        )

    return post


# ── Initialisation ────────────────────────────────────────────────────────────

def initial_filter_state(
    x0: jnp.ndarray | None = None,
    P0_diag: jnp.ndarray | None = None,
) -> ImmuneFilterState:
    x_init = x0 if x0 is not None else initial_state()

    if P0_diag is None:
        P0_diag = jnp.array([
            0.04,    # Damage  sigma=0.20 au
            0.01,    # M1      sigma=0.10 au
            0.01,    # M2      sigma=0.10 au
            25.0,    # IL-6    sigma=5 pg/mL
        ])

    return ImmuneFilterState(mean=x_init, cov=jnp.diag(P0_diag))


# ── Multi-step filter ─────────────────────────────────────────────────────────

def filter_history(
    y_obs_sequence: jnp.ndarray,
    params: ImmuneParams = DEFAULT_IMMUNE_PARAMS,
    obs_params: ImmuneObsParams = DEFAULT_OBS_PARAMS,
    x0: jnp.ndarray | None = None,
    quality_flags_seq: list | None = None,
    control_sequence: jnp.ndarray | None = None,
    t_start: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run UKF for T hourly steps.

    Parameters
    ----------
    y_obs_sequence   : (T, 2)
    quality_flags_seq: list of (int, int) tuples, default all (0, 0)

    Returns
    -------
    means : (T, 4)
    covs  : (T, 4, 4)
    """
    T = y_obs_sequence.shape[0]
    if quality_flags_seq is None:
        quality_flags_seq = [(0, 0)] * T
    if control_sequence is None:
        control_sequence = jnp.zeros((T, 3))

    state = initial_filter_state(x0=x0)
    means, covs = [], []

    for step in range(T):
        u_t  = control_sequence[step]
        ctrl = lambda _: u_t
        state = update_state(
            state          = state,
            y_obs          = y_obs_sequence[step],
            params         = params,
            obs_params     = obs_params,
            quality_flags  = quality_flags_seq[step],
            control_fn     = ctrl,
            t0             = t_start + float(step),
        )
        means.append(state.mean)
        covs.append(state.cov)

    return jnp.stack(means), jnp.stack(covs)
