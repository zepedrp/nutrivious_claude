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
which is exact (not finite-difference) and JIT-traceable. For FIM computation where
D (parameters) << T (time steps), forward-mode AD requires O(D) JVP passes vs
O(T) VJP passes for reverse-mode -- D=2..10 vs T=100..10000 is a clear win.

CRITICAL: forward_fn MUST be built with make_scan_rk4_forward (lax.scan + RK4),
NOT by wrapping diffrax.diffeqsolve. The adaptive solver uses jax.lax.while_loop
whose JVP tape accumulates O(T) memory, destroying the speed/memory advantage.
A fixed-step scan has O(1) memory per JVP step and compiles to a static loop.

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
- Use make_scan_rk4_forward to build forward_fn. lax.scan + fixed-step RK4 gives
  O(1) JVP memory per step and compiles as a static loop (no while_loop tape).
- Do NOT wrap diffrax.diffeqsolve in forward_fn for jacfwd: the adaptive solver's
  while_loop accumulates a JVP tape of length proportional to actual solver steps,
  causing O(T * n_solver_steps) memory blow-up.

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

    Build forward_fn with make_scan_rk4_forward to guarantee O(1) memory per step.
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


# ---------------------------------------------------------------------------
# Fixed-step RK4 forward simulator for memory-safe jacfwd
# ---------------------------------------------------------------------------

def make_scan_rk4_forward(
    ode_fn:    Callable,
    h_fn:      Callable,
    x0:        jax.Array,
    make_args: Callable,
    T:         int,
    dt:        float = 1.0,
) -> Callable:
    """
    Build a forward_fn(theta) -> (T, obs_dim) using fixed-step RK4 + jax.lax.scan.

    This is the required wrapper for compute_fim / jacfwd.  It replaces the
    anti-pattern of wrapping diffrax.diffeqsolve (whose while_loop accumulates a
    JVP tape of length O(T * n_solver_steps) under jacfwd).

    Fixed-step lax.scan gives O(1) JVP memory per step because:
    - The scan body is a *static* function compiled once.
    - JAX differentiates through lax.scan via the efficient associative-scan rule.
    - No dynamic loop tape is accumulated during forward-mode differentiation.

    Complexity vs alternatives
    --------------------------
    D = n_params,  T = n_time_steps,  S = n_adaptive_solver_steps

      method                      | CPU          | RAM
      jacfwd + lax.scan RK4       | O(D * T)     | O(1) per step   <-- this fn
      jacfwd + diffrax while_loop | O(D * T * S) | O(T * S)        <-- BAD
      jacrev + RecursiveAdjoint   | O(T)         | O(sqrt(T))      <-- good for D>>T

    Parameters
    ----------
    ode_fn    : (t, x, args) -> dx/dt.  Must be JAX-differentiable.
    h_fn      : (x) -> (obs_dim,).  Observation function.
    x0        : (state_dim,) initial state -- constant (not differentiated).
    make_args : theta (D,) -> args accepted by ode_fn.  Must be JAX-differentiable.
    T         : number of RK4 steps.
    dt        : step size in same units as the ODE's independent variable.

    Returns
    -------
    forward_fn : theta (D,) -> y (T, obs_dim)
        JIT-compilable and jacfwd-safe.

    Example
    -------
    >>> from app.slices.cardiorespiratory.ode import (
    ...     cardiorespiratory_slice_ode, X0_CARDIO_DEFAULT,
    ...     DEFAULT_CARDIO_SLICE_PARAMS,
    ... )
    >>> import jax.numpy as jnp
    >>>
    >>> # Linearise around VO2max and W' capacity (log-space)
    >>> THETA_NAMES = ["log_VO2max", "log_W_prime"]
    >>> log_theta0  = jnp.log(jnp.array([45.0, 18.0]))
    >>>
    >>> def make_args(log_th):
    ...     p = DEFAULT_CARDIO_SLICE_PARAMS._replace(
    ...         VO2_max_baseline = jnp.exp(log_th[0]),
    ...         W_prime_capacity = jnp.exp(log_th[1]),
    ...     )
    ...     return (p, 300.0, 37.0, 0.0)   # 300 W effort
    >>>
    >>> def h_fn(x):
    ...     return x[jnp.array([1, 0])]    # observe HR and VO2
    >>>
    >>> fwd = make_scan_rk4_forward(
    ...     cardiorespiratory_slice_ode, h_fn,
    ...     X0_CARDIO_DEFAULT, make_args, T=60, dt=1.0,
    ... )
    >>> result = evaluate_identifiability(fwd, log_theta0, jnp.eye(2), THETA_NAMES)
    """
    dt_f32 = jnp.float32(dt)
    half   = jnp.float32(0.5) * dt_f32

    def _rk4_step(x: jax.Array, args: tuple, t: jax.Array) -> jax.Array:
        k1 = ode_fn(t,        x,              args)
        k2 = ode_fn(t + half, x + half * k1,  args)
        k3 = ode_fn(t + half, x + half * k2,  args)
        k4 = ode_fn(t + dt_f32, x + dt_f32 * k3, args)
        return x + (dt_f32 / jnp.float32(6.0)) * (k1 + jnp.float32(2.0)*k2
                                                       + jnp.float32(2.0)*k3 + k4)

    def forward_fn(theta: jax.Array) -> jax.Array:
        args  = make_args(theta)
        t_seq = jnp.arange(T, dtype=jnp.float32) * dt_f32

        def scan_body(x_carry: jax.Array, t_i: jax.Array):
            x_next = _rk4_step(x_carry, args, t_i)
            y_t    = h_fn(x_next)
            return x_next, y_t

        _, ys = jax.lax.scan(scan_body, x0, t_seq)
        return ys   # (T, obs_dim)

    return forward_fn
