"""
app/engine/assimilation/ukf_filter.py

Shared UKF Primitives for all L4 Slice Filters.

Each slice (cardiorespiratory, neuromuscular, neuroendocrine, gonadal, ...) has
its own filter.py with slice-specific state dimensions, ODE kernels, and
observation models. This module provides the stateless mathematical building
blocks used by ALL slice filters, eliminating code duplication across ~13 files.

PUBLIC API
----------
GaussianState               -- filter belief container (mean, cov)
nearest_psd                 -- PSD repair via eigenvalue clipping (Higham 1988)
variance_floor              -- minimum variance enforcement (Simon 2010 S5.4)
ukf_weights                 -- Merwe-Wan sigma-point weights
sigma_points                -- 2n+1 sigma points (eigh-based, PSD-robust)
unscented_transform         -- moment recovery from propagated sigma points
scale_Q                     -- dt-proportional process noise Q_step = Q/min * dt
lower_clamp_moments         -- truncated-normal lower bound (Simon 2010 eq.5.28)
range_clamp_moments         -- truncated-normal [lb, ub] bounds (Simon 2010 eq.5.31)
clamp_dim                   -- apply per-dimension truncated-normal correction

DESIGN INVARIANTS
-----------------
  - Pure JAX: all ops via jnp (JIT + vmap safe).
  - No Python-level conditionals on array values.
  - Float32 throughout (hardware inference; avoids float64 on GPU/TPU).
  - Fail-Loud: NaN propagates -- no silent substitution anywhere here.

PSD STRATEGY (eigendecomposition vs Cholesky)
---------------------------------------------
Sigma-point generation requires sqrt((n+lam)*P). The standard implementation
uses Cholesky, which fails if P has any negative eigenvalue. Over many sequential
UKF update steps in float32, accumulated rounding errors and large Kalman gains
can drive diagonal elements of P slightly negative.

We use jnp.linalg.eigh (symmetric eigendecomposition) instead of Cholesky:
  - eigh returns all real eigenvalues for symmetric input
  - Negative eigenvalues are clipped to `psd_floor` before sqrt
  - No Python try/except needed -- entirely JIT-traceable
  - Cost: ~3x slower than Cholesky for small n (<= 10), acceptable for inference

SIMON 2010 TRUNCATED UKF
-------------------------
Simon D. (2010) "Optimal State Estimation", Wiley, Sections 5.3-5.4.
Simon & Chia (2002) IEEE Trans. Signal Process. 50(2):345-357.

The truncated UKF adds two corrections after each standard UKF update:
  1. Truncated-normal moment matching: update (mean, variance) per constrained
     dimension to reflect the truncated distribution, not the original Gaussian.
  2. Variance floor: after truncated-normal contraction, ensure no diagonal
     element of P falls below a user-specified minimum. Prevents the filter
     from becoming overconfident and refusing future measurement updates.

FLOAT32 SAFETY NOTES FOR SIGMA-POINT WEIGHTS
---------------------------------------------
With alpha=1e-3 (the textbook default) and n >= 5:
    Wm[0] = lam/(n+lam) ~ -1e6       (large negative weight)
    Wi    = 0.5/(n+lam) ~  8.3e4     (large positive weights)
Weighted mean requires cancellation of terms ~10^6 apart, which exceeds
float32 relative precision (~1.2e-7 * 10^6 ~ 0.1 per term). NaN in ~3 steps.

Validated float32-safe alpha values (Wm[0] = 1 - 1/alpha^2, n-independent):
    alpha = 0.10  --> Wm[0] = -99.0    (good for n = 5, 6, 9)
    alpha = 0.50  --> Wm[0] = -3.0     (good for n = 7)
    alpha = 0.30  --> Wm[0] = -10.1    (good for n = 8)

Each slice filter.py specifies its own alpha appropriate for its n.

References
----------
Merwe R. & Wan E.A. (2000) Proc. ASSPCC -- Unscented Kalman Filter.
Merwe R. (2004) PhD thesis, Oregon Health & Science University.
Simon D. (2010) Optimal State Estimation, Wiley. Sections 5.3-5.4.
Simon D. & Chia T.L. (2002) IEEE Trans. Signal Process. 50(2):345-357.
Higham N.J. (1988) Linear Algebra Appl. 103:103-118.
    [Nearest PSD matrix via eigenvalue clipping]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as _jnorm


# ── Filter belief container ───────────────────────────────────────────────────

class GaussianState(NamedTuple):
    """
    First two moments of the UKF state belief at a single timestep.

    Attributes
    ----------
    mean : shape (n,)    -- posterior mean mu
    cov  : shape (n, n)  -- posterior covariance Sigma (symmetric, PSD)

    All slice filters produce and consume GaussianState. The orchestrator
    threads it between timesteps as the carry state.
    """
    mean: jax.Array
    cov:  jax.Array


# ── PSD repair (Higham 1988) ──────────────────────────────────────────────────

def nearest_psd(cov: jax.Array, floor: float = 1e-8) -> jax.Array:
    """
    Project `cov` onto the cone of positive-semidefinite matrices.

    Algorithm:
        1. Symmetrize: cov_sym = 0.5*(cov + cov.T)
        2. Eigendecompose: cov_sym = V diag(d) V.T   (eigh; real, sorted)
        3. Clip: d_safe = max(d, floor)
        4. Reconstruct: P = V diag(d_safe) V.T

    JIT + vmap safe (no Python branching on array values).
    Cost: O(n^3) -- same as Cholesky but avoids failure on non-PSD input.

    Parameters
    ----------
    cov   : (n, n) symmetric matrix (may have small negative eigenvalues)
    floor : minimum eigenvalue (default 1e-8; must be > 0 for strict PD)

    Returns
    -------
    (n, n) symmetric positive-semidefinite matrix nearest to `cov`.

    Reference: Higham (1988) Linear Algebra Appl. 103:103-118.
    """
    cov_sym   = jnp.float32(0.5) * (cov + cov.T)
    vals, vecs = jnp.linalg.eigh(cov_sym)
    vals_safe  = jnp.maximum(vals, jnp.asarray(floor, dtype=cov.dtype))
    return (vecs * vals_safe[None, :]) @ vecs.T


# ── Variance floor (Simon 2010 Section 5.4) ───────────────────────────────────

def variance_floor(cov: jax.Array, floor_diag: jax.Array) -> jax.Array:
    """
    Enforce a minimum variance on each diagonal element of `cov`.

    After truncated-normal moment matching, variances can contract below
    the measurement noise floor, making the filter overconfident and resistant
    to future updates. This function adds the minimum deficit to each diagonal
    element. Off-diagonal elements are not modified (conservative).

    Parameters
    ----------
    cov       : (n, n) covariance matrix
    floor_diag: (n,) minimum variance per state dimension

    Returns
    -------
    (n, n) cov with diag(cov) >= floor_diag, off-diagonal unchanged.

    Reference: Simon (2010) Optimal State Estimation, Section 5.4, p. 208.
    """
    diag_now = jnp.diag(cov)
    deficit  = jnp.maximum(floor_diag - diag_now, jnp.zeros_like(diag_now))
    return cov + jnp.diag(deficit)


# ── UKF sigma-point weights ───────────────────────────────────────────────────

def ukf_weights(
    n:     int,
    alpha: float,
    beta:  float = 2.0,
    kappa: float = 0.0,
) -> tuple[jax.Array, jax.Array, float]:
    """
    Merwe-Wan (2000) sigma-point weights for an n-dimensional state.

    Parameters
    ----------
    n     : state dimension
    alpha : spread parameter; use alpha >= 0.05 for n >= 5 in float32
            (see module-level docstring for float32 safety note)
    beta  : distribution parameter (2.0 optimal for Gaussian)
    kappa : secondary scaling (0.0 recommended)

    Returns
    -------
    Wm  : (2n+1,) float32 -- mean weights
    Wc  : (2n+1,) float32 -- covariance weights
    lam : float           -- lambda = alpha^2*(n+kappa) - n
                             used as the denominator scale for sigma offsets
    """
    lam    = alpha ** 2 * (n + kappa) - n
    Wm_0   = lam / (n + lam)
    Wc_0   = Wm_0 + (1.0 - alpha ** 2 + beta)
    Wi     = 0.5 / (n + lam)
    Wm = jnp.array([Wm_0] + [Wi] * (2 * n), dtype=jnp.float32)
    Wc = jnp.array([Wc_0] + [Wi] * (2 * n), dtype=jnp.float32)
    return Wm, Wc, float(lam)


# ── Sigma points (eigh-based, PSD-robust) ─────────────────────────────────────

def sigma_points(
    mean:      jax.Array,
    cov:       jax.Array,
    lam:       float,
    psd_floor: float = 1e-8,
) -> jax.Array:
    """
    Generate 2n+1 Merwe-Wan sigma points via symmetric eigendecomposition.

    Uses eigh instead of Cholesky to tolerate covariance matrices with
    small negative eigenvalues (float32 accumulation drift after many steps).
    Negative eigenvalues are clipped to `psd_floor` before taking the sqrt.

    Math:
        cov_sym = V diag(d) V.T       [eigh decomposition]
        d_safe  = max(d, psd_floor)
        offsets = V * sqrt((n+lam)*d_safe)  [shape (n, n)]
        x_0     = mean
        x_i     = mean + offsets[:, i-1]    for i in 1..n
        x_{n+i} = mean - offsets[:, i-1]    for i in 1..n

    Parameters
    ----------
    mean      : (n,)
    cov       : (n, n)
    lam       : scaling parameter from ukf_weights()
    psd_floor : minimum eigenvalue clip (default 1e-8)

    Returns
    -------
    sigma_pts : (2n+1, n)
    """
    n         = mean.shape[0]
    cov_sym   = jnp.float32(0.5) * (cov + cov.T)
    vals, vecs = jnp.linalg.eigh(cov_sym)
    vals_safe  = jnp.maximum(vals, jnp.asarray(psd_floor, dtype=mean.dtype))
    scale      = jnp.sqrt(jnp.asarray(n + lam, dtype=mean.dtype) * vals_safe)
    offsets    = vecs * scale[None, :]          # (n, n): col i = i-th offset direction
    pos        = mean[None, :] + offsets.T      # (n, n)
    neg        = mean[None, :] - offsets.T      # (n, n)
    return jnp.concatenate([mean[None, :], pos, neg], axis=0)   # (2n+1, n)


# ── Unscented transform ───────────────────────────────────────────────────────

def unscented_transform(
    sigma_pts: jax.Array,
    noise_cov: jax.Array,
    Wm:        jax.Array,
    Wc:        jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Recover weighted mean and covariance from propagated sigma points.

    Parameters
    ----------
    sigma_pts : (2n+1, m) -- sigma points after propagation through f or h
    noise_cov : (m, m)    -- additive noise covariance (Q for predict, R for update)
    Wm        : (2n+1,)   -- mean weights
    Wc        : (2n+1,)   -- covariance weights

    Returns
    -------
    mean : (m,)
    cov  : (m, m)  -- symmetrised before returning
    """
    mean = jnp.einsum("i,ij->j", Wm, sigma_pts)
    diff = sigma_pts - mean[None, :]
    cov  = jnp.einsum("i,ij,ik->jk", Wc, diff, diff) + noise_cov
    cov  = jnp.float32(0.5) * (cov + cov.T)
    return mean, cov


# ── Process noise scaling ─────────────────────────────────────────────────────

def scale_Q(
    Q_per_unit_time: jax.Array,
    dt_real:         float | jax.Array,
) -> jax.Array:
    """
    Scale process noise proportionally to the actual integration window.

    For a Wiener-process model of unmodelled disturbances, Q grows linearly
    with dt. Applying a fixed 1-min Q budget to a 10-second step inflates
    the predicted uncertainty by 6x.

    Q_step = Q_per_unit_time * dt_real

    Parameters
    ----------
    Q_per_unit_time : (n, n) process noise per unit time (e.g., per minute)
    dt_real         : actual time step in the same units (scalar)

    Returns
    -------
    Q_step : (n, n) -- Q scaled to `dt_real`
    """
    return Q_per_unit_time * jnp.asarray(dt_real, dtype=Q_per_unit_time.dtype)


# ── Truncated-normal moment matching (Simon 2010 Sections 5.3-5.4) ────────────

def lower_clamp_moments(
    mu:     jax.Array,
    sigma2: jax.Array,
    lb:     float,
) -> tuple[jax.Array, jax.Array]:
    """
    Truncated-normal (mean, variance) for the constraint x >= lb.

    For a Gaussian N(mu, sigma^2) truncated to [lb, inf):
        alpha  = (lb - mu) / sigma
        phi(a) = standard normal PDF at alpha
        Phi(a) = standard normal CDF at alpha
        Z      = 1 - Phi(alpha)              [probability mass in [lb, inf)]

        mu_trunc     = mu + sigma * phi(alpha) / Z
        sigma2_trunc = sigma^2 * [1 + alpha*phi(alpha)/Z - (phi(alpha)/Z)^2]

    When the constraint is inactive (mu >> lb): truncated moments ~= original.
    JIT-safe: all ops via jnp, no Python branching on array values.

    Parameters
    ----------
    mu     : current mean (scalar or batched)
    sigma2 : current variance (scalar or batched, >= 0)
    lb     : lower bound (Python float, not a traced value)

    Returns
    -------
    (mu_new, sigma2_new) -- truncated-normal moments (both non-negative)

    Reference: Simon (2010) eq. 5.28-5.30; Simon & Chia (2002) eq. 7-8.
    """
    s      = jnp.sqrt(jnp.maximum(sigma2, jnp.float32(1e-12)))
    alpha  = (jnp.float32(lb) - mu) / s
    phi_a  = _jnorm.pdf(alpha)
    Z      = jnp.maximum(jnp.float32(1.0) - _jnorm.cdf(alpha), jnp.float32(1e-12))
    mu_new     = mu + s * phi_a / Z
    sigma2_new = jnp.maximum(
        sigma2 * (jnp.float32(1.0) + alpha * phi_a / Z - (phi_a / Z) ** 2),
        jnp.float32(1e-12),
    )
    return mu_new, sigma2_new


def range_clamp_moments(
    mu:     jax.Array,
    sigma2: jax.Array,
    lb:     float,
    ub:     float,
) -> tuple[jax.Array, jax.Array]:
    """
    Truncated-normal (mean, variance) for the constraint lb <= x <= ub.

    Extends lower_clamp_moments to a two-sided constraint [lb, ub]:
        a_lo = (lb - mu) / sigma
        a_hi = (ub - mu) / sigma
        Z    = Phi(a_hi) - Phi(a_lo)       [probability mass in [lb, ub]]

        mu_trunc     = mu + sigma*(phi(a_lo) - phi(a_hi)) / Z
        sigma2_trunc = sigma^2 * [1 + (a_lo*phi(a_lo) - a_hi*phi(a_hi))/Z
                                    - ((phi(a_lo)-phi(a_hi))/Z)^2]

    JIT-safe: all ops via jnp.

    Parameters
    ----------
    mu     : current mean
    sigma2 : current variance
    lb, ub : lower and upper bounds (Python floats)

    Returns
    -------
    (mu_new, sigma2_new)

    Reference: Simon (2010) eq. 5.31-5.33; Simon & Chia (2002) eq. 9-10.
    """
    s    = jnp.sqrt(jnp.maximum(sigma2, jnp.float32(1e-12)))
    a_lo = (jnp.float32(lb) - mu) / s
    a_hi = (jnp.float32(ub) - mu) / s
    p_lo = _jnorm.pdf(a_lo)
    p_hi = _jnorm.pdf(a_hi)
    Z    = jnp.maximum(
        _jnorm.cdf(a_hi) - _jnorm.cdf(a_lo),
        jnp.float32(1e-12),
    )
    mu_new     = mu + s * (p_lo - p_hi) / Z
    sigma2_new = jnp.maximum(
        sigma2 * (
            jnp.float32(1.0)
            + (a_lo * p_lo - a_hi * p_hi) / Z
            - ((p_lo - p_hi) / Z) ** 2
        ),
        jnp.float32(1e-12),
    )
    return mu_new, sigma2_new


def clamp_dim(
    mean:      jax.Array,
    cov:       jax.Array,
    i:         int,
    mu_new:    jax.Array,
    sigma2_new: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Apply truncated-normal moment correction to dimension `i` of (mean, cov).

    Updates:
        mean[i]  <- mu_new
        cov[i,i] <- sigma2_new
        cov[i,:] *= sqrt(sigma2_new / sigma2_old)   [scale cross-covariances]
        cov[:,i] *= sqrt(sigma2_new / sigma2_old)

    The cross-covariance scaling preserves the correlation structure
    (Pearson rho_{ij} is unchanged) while reflecting the reduced variance
    in dimension i. This is a first-order approximation -- exact for
    independent dimensions, conservative for correlated ones.

    Parameters
    ----------
    mean       : (n,) state mean vector
    cov        : (n, n) covariance matrix
    i          : dimension index to update (Python int, not traced)
    mu_new     : truncated-normal mean for dimension i
    sigma2_new : truncated-normal variance for dimension i

    Returns
    -------
    (mean_updated, cov_updated) -- both float32 arrays

    Reference: Simon & Chia (2002) eq. 11; Simon (2010) p. 209.
    """
    sigma2_old = jnp.maximum(cov[i, i], jnp.float32(1e-12))
    scale      = jnp.sqrt(sigma2_new / sigma2_old)
    cov        = cov.at[i, :].mul(scale)
    cov        = cov.at[:, i].mul(scale)
    cov        = cov.at[i, i].set(sigma2_new)   # exact restore (was double-scaled)
    mean       = mean.at[i].set(mu_new)
    return mean, cov
