"""
app/slices/hematological/filter.py  L4 UKF - Hematological Slice V3.0

Unscented Kalman Filter for the 5-state hourly hematological system:
  [RBC_Mass_g, Plasma_Vol_L, EPO_mIU_mL, Ferritin_ug_L, Hemolysis_Tox_au]

Observations y in R^3: [Hgb_g_dL, Hematocrit_pct, Ferritin_lab]

UKF parametrisation (van der Merwe & Wan 2000)
----------------------------------------------
  alpha = 0.10  MANDATORY for float32 stability with n=5.
    lambda = alpha^2*(n+kappa)-n = 0.01*5-5 = -4.95
    n+lambda = 0.05
    W0_mean = -4.95/0.05 = -99  (same magnitude as n=9 case; float32 safe)
    Wi_mean = 0.5/0.05   = 10.0
    W0_cov  = -99 + (1-0.01+2) = -96.01
    Wi_cov  = 10.0
  beta  = 2.0
  kappa = 0.0
  n=5 -> 2n+1 = 11 sigma points

Numerical safeguards:
  jitter  : add 1e-3*I before cholesky (ensures P is numerically PD)
  clamp   : jnp.maximum(sigma_propagated, 0) after ODE step (physical bounds)
  floor   : ensure diag(P_posterior) >= 1e-3 after update

Fail-Loud:
  NaN in posterior mean   -> RuntimeError
  Negative cov diagonal   -> RuntimeError
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.hematological.ode import (
    STATE_DIM, OBS_DIM,
    HematologicalParams, DEFAULT_HEM_PARAMS,
    X0_HEM_DEFAULT, P0_HEM_DEFAULT,
    integrate_1h,
)
from app.slices.hematological.observation import (
    HemObsParams, DEFAULT_HEM_OBS_PARAMS,
    h_hem, inflate_R_hem,
)

# ── UKF weights (alpha=0.10, n=5) ─────────────────────────────────────────────

_ALPHA  = 0.10
_BETA   = 2.0
_KAPPA  = 0.0
_N      = STATE_DIM   # 5

_LAMBDA   = _ALPHA ** 2 * (_N + _KAPPA) - _N   # = 0.01*5 - 5 = -4.95
_NL       = _N + _LAMBDA                         # = 0.05

_W0_MEAN  = _LAMBDA / _NL                        # = -99.0
_WI_MEAN  = 0.5     / _NL                        # = 10.0
_W0_COV   = _W0_MEAN + (1.0 - _ALPHA**2 + _BETA) # = -96.01
_WI_COV   = _WI_MEAN                             # = 10.0

_WM: jax.Array = jnp.array(
    [_W0_MEAN] + [_WI_MEAN] * (2 * _N), dtype=jnp.float32
)
_WC: jax.Array = jnp.array(
    [_W0_COV]  + [_WI_COV]  * (2 * _N), dtype=jnp.float32
)

_JITTER = 1e-3   # added to cov diagonal before Cholesky
_COV_FLOOR = 1e-3  # minimum diagonal element after update


# ── Process noise Q ────────────────────────────────────────────────────────────

_Q_DIAG: jax.Array = jnp.array([
    100.0,   # RBC_Mass_g [g]^2    slow; 10g/h variability
    0.001,   # Plasma_Vol_L [L]^2  ~0.03L/h variability
    1.0,     # EPO_mIU_mL          ~1 unit/h pulsatile
    4.0,     # Ferritin_ug_L       ~2 ug/L/h stores flux
    0.01,    # Hemolysis_Tox_au    slow clearing
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


# ── Filter state ───────────────────────────────────────────────────────────────

class HemFilterState(NamedTuple):
    mean: jax.Array   # (STATE_DIM=5,)
    cov:  jax.Array   # (STATE_DIM=5, STATE_DIM=5)


# ── Sigma-point helpers ────────────────────────────────────────────────────────

def _sigma_points(mean: jax.Array, cov: jax.Array) -> jax.Array:
    """
    Merwe-Wan 2n+1 sigma points with jitter for numerical stability.
    Returns shape (11, STATE_DIM).
    """
    P_jit = cov + _JITTER * jnp.eye(_N)
    L     = jnp.linalg.cholesky(_NL * P_jit)
    pos   = mean[None, :] + L.T    # (n, STATE_DIM)
    neg   = mean[None, :] - L.T    # (n, STATE_DIM)
    return jnp.concatenate([mean[None, :], pos, neg], axis=0)   # (2n+1, STATE_DIM)


def _recover_mean_cov(
    pts:    jax.Array,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean = jnp.einsum("i,ij->j", _WM, pts)
    diff = pts - mean[None, :]
    cov  = jnp.einsum("i,ij,ik->jk", _WC, diff, diff) + Q
    cov  = 0.5 * (cov + cov.T)
    return mean, cov


# ── Transition function (1-hour ODE + physical clamp) ─────────────────────────

def _integrate_clamp(
    x:      jax.Array,
    u:      jax.Array,
    params: HematologicalParams,
) -> jax.Array:
    """1-hour ODE step; clamp all states >= 0 after propagation."""
    x_next = integrate_1h(x, params=params, u=u, t0=0.0)
    return jnp.maximum(x_next, 0.0)


# ── UKF predict ───────────────────────────────────────────────────────────────

def _ukf_predict(
    state:  HemFilterState,
    u:      jax.Array,
    params: HematologicalParams,
    Q:      jax.Array,
) -> HemFilterState:
    sigma      = _sigma_points(state.mean, state.cov)
    sigma_next = jax.vmap(
        _integrate_clamp, in_axes=(0, None, None)
    )(sigma, u, params)
    mean_pred, cov_pred = _recover_mean_cov(sigma_next, Q)
    return HemFilterState(mean=mean_pred, cov=cov_pred)


# ── UKF update ────────────────────────────────────────────────────────────────

def _ukf_update(
    state:      HemFilterState,
    y_obs:      jax.Array,
    R:          jax.Array,
    obs_params: HemObsParams,
) -> HemFilterState:
    sigma   = _sigma_points(state.mean, state.cov)
    y_sigma = jax.vmap(lambda x: h_hem(x, obs_params))(sigma)   # (11, OBS_DIM)

    y_mean = jnp.einsum("i,io->o", _WM, y_sigma)
    dy_s   = y_sigma - y_mean[None, :]
    dx_s   = sigma   - state.mean[None, :]

    S_yy = jnp.einsum("i,io,ij->oj", _WC, dy_s, dy_s) + R   # (OBS_DIM, OBS_DIM)
    P_xy = jnp.einsum("i,is,io->so", _WC, dx_s, dy_s)        # (STATE_DIM, OBS_DIM)

    K          = P_xy @ jnp.linalg.inv(S_yy)
    innov      = y_obs - y_mean
    mu_new     = state.mean + K @ innov
    P_new      = state.cov  - K @ S_yy @ K.T
    P_new      = 0.5 * (P_new + P_new.T)

    # Physical clamp on posterior mean
    mu_new = jnp.maximum(mu_new, 0.0)

    # Floor diagonal >= COV_FLOOR (numerical stability)
    diag_idx = jnp.arange(_N)
    diag_new = jnp.maximum(jnp.diag(P_new), _COV_FLOOR)
    P_new    = P_new.at[diag_idx, diag_idx].set(diag_new)

    return HemFilterState(mean=mu_new, cov=P_new)


# ── Public API (Fail-Loud) ────────────────────────────────────────────────────

def update_state(
    state:         HemFilterState,
    y_obs:         jax.Array,
    params:        HematologicalParams = DEFAULT_HEM_PARAMS,
    obs_params:    HemObsParams        = DEFAULT_HEM_OBS_PARAMS,
    Q:             jax.Array | None    = None,
    quality_flags: tuple[int, int, int] = (4, 4, 4),
    u:             jax.Array | None    = None,
) -> HemFilterState:
    """
    Full UKF cycle: 1-h ODE predict -> 3-channel observation update.

    quality_flags = (flag_Hgb, flag_Hct, flag_Ferritin)
    flag=4 -> R*1e8 for that channel (predict-only assimilation).

    Fail-Loud:
      NaN posterior mean  -> RuntimeError
      Negative cov diag   -> RuntimeError
    """
    if Q is None:
        Q = Q_DEFAULT
    if u is None:
        u = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)

    R = inflate_R_hem(obs_params=obs_params, flags=quality_flags)

    pred = _ukf_predict(state, u=u, params=params, Q=Q)
    post = _ukf_update(pred,  y_obs=y_obs, R=R, obs_params=obs_params)

    if jnp.any(jnp.isnan(post.mean)):
        raise RuntimeError(
            "HemUKF: NaN in posterior mean. "
            f"Prior mean={state.mean}, y_obs={y_obs}. "
            "Increase Q diagonal or check ODE stability."
        )
    if jnp.any(jnp.diag(post.cov) < 0.0):
        raise RuntimeError(
            f"HemUKF: negative diagonal in posterior cov. diag={jnp.diag(post.cov)}"
        )
    return post


# ── Initialisation ────────────────────────────────────────────────────────────

def initial_filter_state(
    x0:      jax.Array | None = None,
    P0_diag: jax.Array | None = None,
) -> HemFilterState:
    """Build the initial HemFilterState with population priors."""
    x_init = x0 if x0 is not None else X0_HEM_DEFAULT
    if P0_diag is None:
        P_init = P0_HEM_DEFAULT
    else:
        P_init = jnp.diag(P0_diag)
    return HemFilterState(mean=x_init, cov=P_init)


# ── Multi-step filter ─────────────────────────────────────────────────────────

def filter_history(
    y_obs_sequence:  jax.Array,
    params:          HematologicalParams             = DEFAULT_HEM_PARAMS,
    obs_params:      HemObsParams                   = DEFAULT_HEM_OBS_PARAMS,
    x0:              jax.Array | None               = None,
    quality_flags:   list[tuple[int, int, int]] | None = None,
    u_sequence:      jax.Array | None               = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Run UKF for T hourly steps.

    y_obs_sequence  : (T, OBS_DIM=3)
    quality_flags   : list of T tuples, default all (4,4,4)
    u_sequence      : (T, CTRL_DIM=4), default all zeros

    Returns (means: (T, STATE_DIM), covs: (T, STATE_DIM, STATE_DIM))
    """
    T = y_obs_sequence.shape[0]
    if quality_flags is None:
        quality_flags = [(4, 4, 4)] * T
    if u_sequence is None:
        u_sequence = jnp.zeros((T, 4), dtype=jnp.float32)

    state  = initial_filter_state(x0=x0)
    means, covs = [], []

    for step in range(T):
        state = update_state(
            state         = state,
            y_obs         = y_obs_sequence[step],
            params        = params,
            obs_params    = obs_params,
            quality_flags = quality_flags[step],
            u             = u_sequence[step],
        )
        means.append(state.mean)
        covs.append(state.cov)

    return jnp.stack(means), jnp.stack(covs)
