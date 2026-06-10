"""
app/engine/oed/identifiability.py

PILAR 1 -- Dynamic Sensitivity and Fisher Information Matrix (FIM) Evaluator.

Mathematical Foundation
-----------------------
For an ODE system with output y_t = h(x(t; theta)) and additive Gaussian noise
y_t_obs ~ N(y_t, R), the Fisher Information Matrix (FIM) at parameter point theta is:

    FIM(theta) = sum_{t=1}^{T} J_t^T @ R^{-1} @ J_t

where J_t = dy_t/d(theta) is the sensitivity matrix (Jacobian of the observation
map with respect to the parameters at time t).

Sensitivities are computed via jax.jacfwd (forward-mode automatic differentiation),
which is exact (not finite-difference) and JIT-traceable. Forward-mode AD is preferred
over reverse-mode here because D << T (few parameters, many time steps), making
JVP-based (forward) computation more efficient than VJP-based (reverse).

Identifiability Criterion
--------------------------
Eigendecompose: FIM = V @ diag(lambda) @ V^T (via jnp.linalg.eigh, ascending order).

A parameter direction v_i is UNIDENTIFIABLE if:
    lambda_i < threshold_rel * max(lambda)

i.e., the condition number of the FIM exceeds 1/threshold_rel in that direction.
This is a relative criterion robust to scaling differences across parameters.

D-Optimality
-------------
The D-optimal design maximises log det(FIM) = sum_i log(lambda_i), which minimises
the volume of the ellipsoidal confidence region for all parameters simultaneously.
This is the scalar metric used by OEDProtocolGenerator to rank candidate protocols.

Compatibility
-------------
- Pure JAX: all array operations are JIT-traceable and vmap-safe.
- jax.jacfwd traces through jax.lax.scan and diffrax.diffeqsolve (which uses
  jax.lax.while_loop). Both support forward-mode JVP in JAX >= 0.4.
- If a slice uses a non-differentiable op, provide a simplified forward_fn
  (e.g., fixed-step RK4) instead of the production diffrax solver.

References
----------
  Balsa-Canto E. et al. (2010) Biotechnol Prog 26:326-333
    DOI 10.1002/btpr.311  [OED for dynamic biological models]
  Walter E. & Pronzato L. (1997) Identification of Parametric Models.
    Masson, Paris.  [FIM theory for ODE identification]
  Cobelli C. & DiStefano J. (1980) Am J Physiol 239:R7-R24
    DOI 10.1152/ajpregu.1980.239.1.R7  [identifiability of physiological models]
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class FIMResult(NamedTuple):
    """
    Complete identifiability report for one (theta, protocol) pair.

    Fields
    ------
    fim                 : (D, D) Fisher Information Matrix
    eigenvalues         : (D,) ascending eigenvalues of FIM
    eigenvectors        : (D, D) corresponding eigenvectors (columns)
    log_det             : float -- D-optimality score = log det(FIM)
    unidentifiable_mask : (D,) bool -- True if eigenvalue below threshold
    unidentifiable_params : list[str] -- names of unidentifiable parameters
    condition_number    : float -- max(|lambda|) / min(|lambda|)
    """
    fim:                   jax.Array
    eigenvalues:           jax.Array
    eigenvectors:          jax.Array
    log_det:               float
    unidentifiable_mask:   jax.Array
    unidentifiable_params: list
    condition_number:      float


class SliceFIMConfig(NamedTuple):
    """
    Configuration bundle for FIM evaluation of a specific physiological slice.

    Fields
    ------
    param_names   : tuple[str] -- ordered parameter names (matches theta dimension)
    theta_prior   : (D,) -- linearisation point in unconstrained (log) space
    R_inv         : scalar or (obs_dim, obs_dim) -- inverse observation noise covariance
    threshold_rel : float -- identifiability threshold: lambda_i unidentifiable if
                             lambda_i < threshold_rel * max(lambda)
    """
    param_names:   tuple
    theta_prior:   jax.Array
    R_inv:         object
    threshold_rel: float = 1e-4


# ---------------------------------------------------------------------------
# Core FIM computation (JIT-able)
# ---------------------------------------------------------------------------

def compute_fim(
    forward_fn: Callable,
    theta:      jax.Array,
    R_inv:      object,
) -> jax.Array:
    """
    Compute FIM = sum_t J_t^T @ R^{-1} @ J_t via jax.jacfwd.

    Parameters
    ----------
    forward_fn : callable -- theta: (D,) -> y: (T,) or (T, obs_dim)
                 Must be differentiable (no Python-level conditionals on traced values).
    theta      : (D,) -- parameter vector at which to linearise.
    R_inv      : scalar or (obs_dim, obs_dim) -- inverse observation noise covariance.
                 For scalar obs (y: (T,)), pass a scalar float or 0-d array.
                 For vector obs (y: (T, obs_dim)), pass (obs_dim, obs_dim) array.

    Returns
    -------
    fim : (D, D) -- Fisher Information Matrix.

    Notes
    -----
    J = jax.jacfwd(forward_fn)(theta) has shape:
        (T, D)          if forward_fn returns (T,)
        (T, obs_dim, D) if forward_fn returns (T, obs_dim)

    FIM for scalar obs: R_inv * J.T @ J
    FIM for vector obs: einsum("tij,il,tlk->jk", J, R_inv, J)
        which equals sum_t J_t.T @ R_inv @ J_t for each t.
    """
    J = jax.jacfwd(forward_fn)(theta)

    if J.ndim == 2:
        # J: (T, D) -- scalar observation channel
        # FIM[j,k] = R_inv * sum_t J[t,j] * J[t,k]
        fim = jnp.asarray(R_inv, dtype=jnp.float32) * jnp.einsum("ti,tj->ij", J, J)
    else:
        # J: (T, obs_dim, D) -- vector observation
        # FIM[j,k] = sum_{t,i,l} J[t,i,j] * R_inv[i,l] * J[t,l,k]
        fim = jnp.einsum(
            "tij,il,tlk->jk",
            J,
            jnp.asarray(R_inv, dtype=jnp.float32),
            J,
        )

    return fim   # (D, D)


# ---------------------------------------------------------------------------
# Eigendecomposition and identifiability labelling
# ---------------------------------------------------------------------------

def analyze_fim(
    fim:          jax.Array,
    param_names:  list,
    threshold_rel: float = 1e-4,
) -> FIMResult:
    """
    Eigendecompose FIM and produce a full identifiability report.

    Parameters
    ----------
    fim           : (D, D) -- Fisher Information Matrix (symmetric PSD).
    param_names   : list[str] of length D -- parameter names for labelling.
    threshold_rel : float -- relative identifiability threshold.
                    Direction i unidentifiable if lambda_i < threshold_rel * max(|lambda|).

    Returns
    -------
    FIMResult
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(fim)   # ascending order

    max_ev     = jnp.max(jnp.abs(eigenvalues))
    threshold  = threshold_rel * jnp.maximum(max_ev, jnp.float32(1e-30))
    unid_mask  = eigenvalues < threshold

    # D-optimality: log det(FIM) = sum log(lambda_i), clamped to avoid -inf
    log_det = float(
        jnp.sum(jnp.log(jnp.maximum(eigenvalues, jnp.float32(1e-30))))
    )

    min_ev_abs = jnp.maximum(jnp.min(jnp.abs(eigenvalues)), jnp.float32(1e-30))
    cond       = float(max_ev / min_ev_abs)

    # Project each unidentifiable eigenvector onto the parameter axes.
    # Eigenvalue index i does NOT correspond to parameter i -- eigh returns
    # eigenvectors in an arbitrary rotated basis. We find the parameter axis
    # each unidentifiable eigenvector most aligns with (|v_ij| largest).
    unid_params: list = []
    for i in range(len(param_names)):
        if bool(unid_mask[i]):
            evec        = eigenvectors[:, i]      # i-th eigenvector (column)
            dom_idx     = int(jnp.argmax(jnp.abs(evec)))
            dom_name    = param_names[dom_idx]
            if dom_name not in unid_params:       # deduplicate for D=2 edge cases
                unid_params.append(dom_name)

    return FIMResult(
        fim                   = fim,
        eigenvalues           = eigenvalues,
        eigenvectors          = eigenvectors,
        log_det               = log_det,
        unidentifiable_mask   = unid_mask,
        unidentifiable_params = unid_params,
        condition_number      = cond,
    )


# ---------------------------------------------------------------------------
# High-level convenience entry point
# ---------------------------------------------------------------------------

def evaluate_identifiability(
    forward_fn:   Callable,
    theta:        jax.Array,
    R_inv:        object,
    param_names:  list,
    threshold_rel: float = 1e-4,
) -> FIMResult:
    """
    One-call FIM computation + identifiability analysis.

    Parameters
    ----------
    forward_fn    : callable -- theta (D,) -> y (T,) or (T, obs_dim).
    theta         : (D,) -- linearisation point (unconstrained / log space).
    R_inv         : scalar or (obs_dim, obs_dim).
    param_names   : list[str] of length D.
    threshold_rel : float.

    Returns
    -------
    FIMResult

    Example
    -------
    >>> from app.slices.cardiorespiratory.nlme import (
    ...     simulate_hr_trajectory, _LOG_PRIOR_MEAN, THETA_NAMES
    ... )
    >>> from app.slices.cardiorespiratory.ode import X0_CARDIO_DEFAULT
    >>> import jax.numpy as jnp
    >>>
    >>> power_rates = jnp.full((20,), 300.0)   # 20-min at 300 W
    >>> sigma_obs   = 5.0                       # HR noise [bpm]
    >>>
    >>> fwd = lambda log_th: simulate_hr_trajectory(log_th, X0_CARDIO_DEFAULT, power_rates)
    >>> result = evaluate_identifiability(
    ...     fwd, _LOG_PRIOR_MEAN, 1.0 / sigma_obs**2, THETA_NAMES
    ... )
    >>> print("Unidentifiable:", result.unidentifiable_params)
    >>> print("log det(FIM):", result.log_det)
    """
    fim = compute_fim(forward_fn, theta, R_inv)
    return analyze_fim(fim, list(param_names), threshold_rel)
