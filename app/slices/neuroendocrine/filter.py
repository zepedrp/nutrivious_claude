"""
app/slices/neuroendocrine/filter.py — L4 UKF: SAM × HPA × Somatotropic  V2.0

Unscented Kalman Filter for the 9-state:
    [Epinephrine, Norepinephrine, CRH, ACTH, Cortisol,
     GHRH, Somatostatin, GH, IGF-1]

UKF parametrisation (van der Merwe & Wan 2000)
-----------------------------------------------
    α = 0.10  — MANDATORY for float32 stability with n=9.
                W0_mean = 1 − 1/α² = −99 (same as n=7 case; α determines weight
                magnitude, not n). With α=0.001 we'd get W0=−9999 → catastrophic
                cancellation. With α=0.10 absolute cancellation ≈ 100× the result
                value, well within float32's 7-digit precision.
    β = 2.0   — Gaussian kurtosis prior
    κ = 0.0
    n = 9  →  2n+1 = 19 sigma points
    λ = α²(n+κ) − n = 0.01×9 − 9 = −8.91
    n+λ = 0.09

Prediction: 1-hour Kvaerno5 ODE integration (stiff system).
Sigma-point propagation via jax.vmap over 19 ODE solves.

Observation: y ∈ ℝ⁴ = [Epi_pgmL, NE_pgmL, Cortisol_nmolL, IGF1_ngmL]

Thermodynamic constraint (MANDATORY)
--------------------------------------
    After each update, applies jnp.maximum(posterior_mean, 0.0).
    Hormone concentrations cannot be negative.

Fail-Loud contract
------------------
    NaN in posterior_mean     → RuntimeError("NeuroUKF: divergence")
    Negative diagonal cov     → RuntimeError("NeuroUKF: negative variance")
    All exceptions propagate — no silent substitution.

References
----------
    van der Merwe & Wan (2000) Proc. ASSPCC
    Vinther et al. (2011) J Math Biol 63:663–690
    Goldstein (2010) Cell Mol Neurobiol 30:1283–1295
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.neuroendocrine.ode import (
    IDX_CORT, IDX_IGF1, IDX_EPI, IDX_NE,
    STATE_DIM,
    NeuroParams, DEFAULT_NEURO_PARAMS,
    zero_control, integrate_1h, initial_state,
)
from app.slices.neuroendocrine.observation import (
    NeuroObsParams, DEFAULT_OBS_PARAMS,
    h_neuro, observation_noise_R,
)

# ── UKF constants (α=0.10, n=9) ───────────────────────────────────────────────

_ALPHA  = 0.10
_BETA   = 2.0
_KAPPA  = 0.0
_N      = STATE_DIM   # 9

_LAMBDA   = _ALPHA ** 2 * (_N + _KAPPA) - _N   # = 0.09 - 9 = -8.91
_N_LAMBDA = _N + _LAMBDA                         # = 0.09

_W0_MEAN = _LAMBDA / _N_LAMBDA                   # = -99.0
_WI_MEAN = 0.5 / _N_LAMBDA                       # = 5.5556
_W0_COV  = _W0_MEAN + (1.0 - _ALPHA ** 2 + _BETA)  # = -99 + 2.99 = -96.01
_WI_COV  = _WI_MEAN

_WM = jnp.array([_W0_MEAN] + [_WI_MEAN] * (2 * _N))   # (2n+1 = 19,)
_WC = jnp.array([_W0_COV]  + [_WI_COV]  * (2 * _N))


# ── Filter state ───────────────────────────────────────────────────────────────

class NeuroFilterState(NamedTuple):
    mean: jnp.ndarray   # (STATE_DIM=9,)
    cov:  jnp.ndarray   # (STATE_DIM=9, STATE_DIM=9)


# ── Process noise ──────────────────────────────────────────────────────────────

def default_process_noise_Q() -> jnp.ndarray:
    """
    Diagonal Q per 1-hour step.

    Epi/NE: large Q because catecholamines are highly pulsatile and
    change dramatically within minutes (stress events not in control input).
    All other states: calibrated to hourly physiological variability.
    """
    q_diag = jnp.array([
        2500.0,   # Epinephrine    [pg/mL]²  σ_proc≈50 pg/mL/h pulsatile
        10000.0,  # Norepinephrine [pg/mL]²  σ_proc≈100 pg/mL/h pulsatile
        0.50,     # CRH            [pg/mL]²
        4.00,     # ACTH           [pg/mL]²
        400.0,    # Cortisol       [nmol/L]² pulsatile ~20 nmol/L/h
        25.0,     # GHRH           [pg/mL]²
        9.00,     # Somatostatin   [pg/mL]²
        1.00,     # GH             [ng/mL]²
        4.00,     # IGF-1          [ng/mL]²  slow but small hourly drift
    ])
    return jnp.diag(q_diag)


# ── Physical clamp (thermodynamic invariant) ───────────────────────────────────

def _clamp_physical(x: jnp.ndarray) -> jnp.ndarray:
    """All hormone concentrations must be ≥ 0."""
    return jnp.maximum(x, 0.0)


# ── Sigma-point ODE transition ─────────────────────────────────────────────────

def _transition_sigma(
    x: jnp.ndarray,
    params: NeuroParams,
    control_fn: object,
    t0: float,
) -> jnp.ndarray:
    """Advance one sigma point by 1 h via Kvaerno5; enforce physical bounds."""
    x_next = integrate_1h(x, params=params, control_fn=control_fn, t0=t0)
    return _clamp_physical(x_next)


# ── UKF predict step ──────────────────────────────────────────────────────────

def _ukf_predict(
    state: NeuroFilterState,
    params: NeuroParams,
    Q: jnp.ndarray,
    control_fn: object,
    t0: float,
) -> NeuroFilterState:
    mu = state.mean
    P  = state.cov

    # Cholesky decomposition of scaled covariance
    sqrt_P  = jnp.linalg.cholesky(_N_LAMBDA * P)
    sp_pos  = mu[None, :] + sqrt_P.T    # (n, STATE_DIM) — n positive sigma pts
    sp_neg  = mu[None, :] - sqrt_P.T    # (n, STATE_DIM) — n negative sigma pts
    sigmas  = jnp.concatenate([mu[None, :], sp_pos, sp_neg], axis=0)  # (2n+1, STATE_DIM)

    # Propagate all 19 sigma points through the ODE simultaneously
    propagate = jax.vmap(
        lambda x: _transition_sigma(x, params, control_fn, t0)
    )
    sigmas_f = propagate(sigmas)

    mu_pred = jnp.einsum("i,is->s", _WM, sigmas_f)
    diff    = sigmas_f - mu_pred[None, :]
    P_pred  = jnp.einsum("i,is,it->st", _WC, diff, diff) + Q
    P_pred  = 0.5 * (P_pred + P_pred.T)   # enforce symmetry

    return NeuroFilterState(mean=mu_pred, cov=P_pred)


# ── UKF update step ───────────────────────────────────────────────────────────

def _ukf_update(
    state: NeuroFilterState,
    y_obs: jnp.ndarray,
    R: jnp.ndarray,
    obs_params: NeuroObsParams,
) -> NeuroFilterState:
    mu = state.mean
    P  = state.cov

    sqrt_P   = jnp.linalg.cholesky(_N_LAMBDA * P)
    sp_pos   = mu[None, :] + sqrt_P.T
    sp_neg   = mu[None, :] - sqrt_P.T
    sigmas   = jnp.concatenate([mu[None, :], sp_pos, sp_neg], axis=0)

    obs_fn   = jax.vmap(lambda x: h_neuro(x, obs_params=obs_params))
    y_sigmas = obs_fn(sigmas)   # (2n+1, OBS_DIM=4)

    y_mean = jnp.einsum("i,io->o", _WM, y_sigmas)
    dy     = y_sigmas - y_mean[None, :]
    dx     = sigmas   - mu[None, :]

    S   = jnp.einsum("i,io,ij->oj", _WC, dy, dy) + R     # (OBS_DIM, OBS_DIM)
    Pxy = jnp.einsum("i,is,io->so", _WC, dx, dy)         # (STATE_DIM, OBS_DIM)

    K      = Pxy @ jnp.linalg.inv(S)
    innov  = y_obs - y_mean
    mu_new = mu + K @ innov
    P_new  = P - K @ S @ K.T
    P_new  = 0.5 * (P_new + P_new.T)

    # Thermodynamic clamp — concentrations cannot be negative
    mu_new = _clamp_physical(mu_new)

    return NeuroFilterState(mean=mu_new, cov=P_new)


# ── Public API (Fail-Loud) ────────────────────────────────────────────────────

def update_state(
    state: NeuroFilterState,
    y_obs: jnp.ndarray,
    params: NeuroParams = DEFAULT_NEURO_PARAMS,
    obs_params: NeuroObsParams = DEFAULT_OBS_PARAMS,
    Q: jnp.ndarray | None = None,
    quality_flag: int = 0,
    control_fn: object = zero_control,
    t0: float = 8.0,
) -> NeuroFilterState:
    """
    Full UKF cycle: 1-h Kvaerno5 ODE predict → 4-channel observation update.

    Fail-Loud:
      • NaN in posterior mean   → RuntimeError("NeuroUKF: divergence …")
      • Negative diagonal cov   → RuntimeError("NeuroUKF: negative variance …")
    """
    if Q is None:
        Q = default_process_noise_Q()

    R = observation_noise_R(obs_params=obs_params, quality_flag=quality_flag)

    pred = _ukf_predict(state, params=params, Q=Q, control_fn=control_fn, t0=t0)
    post = _ukf_update(pred, y_obs=y_obs, R=R, obs_params=obs_params)

    mu = post.mean
    P  = post.cov

    if jnp.any(jnp.isnan(mu)):
        raise RuntimeError(
            "NeuroUKF: divergence — NaN in posterior mean. "
            f"Prior mean={state.mean}, y_obs={y_obs}. "
            "Inspect Kvaerno5 convergence or increase Q diagonal."
        )

    if jnp.any(jnp.diag(P) < 0.0):
        raise RuntimeError(
            "NeuroUKF: negative variance in posterior covariance. "
            f"diag(P)={jnp.diag(P)}."
        )

    return post


# ── Initialisation ────────────────────────────────────────────────────────────

def initial_filter_state(
    x0: jnp.ndarray | None = None,
    P0_diag: jnp.ndarray | None = None,
) -> NeuroFilterState:
    """
    Build the initial NeuroFilterState.

    P0 diagonal (population uncertainty at onboarding):
        Epi:        σ = 50   pg/mL  → 2500
        NE:         σ = 100  pg/mL  → 10000
        CRH:        σ = 2    pg/mL  → 4
        ACTH:       σ = 10   pg/mL  → 100
        Cortisol:   σ = 70   nmol/L → 4900
        GHRH:       σ = 30   pg/mL  → 900
        SS:         σ = 15   pg/mL  → 225
        GH:         σ = 3    ng/mL  → 9
        IGF-1:      σ = 60   ng/mL  → 3600
    """
    x_init = x0 if x0 is not None else initial_state()

    if P0_diag is None:
        P0_diag = jnp.array([
            2500.0,   # Epi
            10000.0,  # NE
            4.0,      # CRH
            100.0,    # ACTH
            4900.0,   # Cortisol
            900.0,    # GHRH
            225.0,    # SS
            9.0,      # GH
            3600.0,   # IGF-1
        ])

    return NeuroFilterState(mean=x_init, cov=jnp.diag(P0_diag))


# ── Multi-step filter ─────────────────────────────────────────────────────────

def filter_history(
    y_obs_sequence: jnp.ndarray,
    params: NeuroParams = DEFAULT_NEURO_PARAMS,
    obs_params: NeuroObsParams = DEFAULT_OBS_PARAMS,
    x0: jnp.ndarray | None = None,
    quality_flags: jnp.ndarray | None = None,
    control_sequence: jnp.ndarray | None = None,
    t_start: float = 8.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run UKF for T hourly steps.

    Parameters
    ----------
    y_obs_sequence   : (T, OBS_DIM=4)
    quality_flags    : (T,) int, default all 0
    control_sequence : (T, CTRL_DIM=5) float, default zero_control

    Returns
    -------
    means : (T, STATE_DIM=9)
    covs  : (T, STATE_DIM, STATE_DIM)
    """
    T = y_obs_sequence.shape[0]
    quality_flags    = quality_flags    if quality_flags    is not None else jnp.zeros(T, dtype=int)
    control_sequence = control_sequence if control_sequence is not None else jnp.zeros((T, 5))

    state  = initial_filter_state(x0=x0)
    means, covs = [], []

    for step in range(T):
        u_t  = control_sequence[step]
        ctrl = lambda _: u_t
        state = update_state(
            state        = state,
            y_obs        = y_obs_sequence[step],
            params       = params,
            obs_params   = obs_params,
            quality_flag = int(quality_flags[step]),
            control_fn   = ctrl,
            t0           = t_start + float(step),
        )
        means.append(state.mean)
        covs.append(state.cov)

    return jnp.stack(means), jnp.stack(covs)
