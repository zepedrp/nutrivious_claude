"""
app/engine/oed/fisher_design.py

PILAR 3 -- OED Protocol Generator (D-Optimality via Fisher Information).

When PILAR 1 (identifiability.py) flags unidentifiable parameters, the Digital
Twin must inject a targeted stimulus ("active inference") to resolve the ambiguity.
This module implements the OED (Optimal Experimental Design) selector that picks
the exercise protocol maximising information gain about the hidden parameters.

D-Optimality Criterion
-----------------------
Given a set of candidate protocols {u_1, ..., u_K} (exercise profiles), each
induces a different Fisher Information Matrix:

    FIM_k = sum_t J_t(theta_0, u_k)^T @ R^{-1} @ J_t(theta_0, u_k)

The D-optimal protocol maximises:

    phi_D(FIM_k) = log det(FIM_k) = sum_i log(lambda_i^k)

This criterion minimises the volume of the joint confidence ellipsoid for all
parameters simultaneously. It is the standard criterion in clinical trial design
and systems biology OED (Balsa-Canto 2010, Pronzato 2008).

Usage pattern
-------------
1. Build a forward_factory for your slice:
       factory = make_cardio_fim_factory()
   or supply any callable: (x0, protocol) -> (theta -> y_trajectory).

2. Build the OEDProtocolGenerator with a SliceFIMConfig.

3. Call select_optimal(candidates, x0) -- gets the D-optimal protocol + an OEDAction.

4. Feed OEDAction back to the Twin's prescription engine (NMPC / Phase3Envelope).

References
----------
  Balsa-Canto E. et al. (2010) Biotechnol Prog 26:326-333  [OED for ODE models]
  Pronzato L. (2008) Automatica 44:303-325                  [D-optimal design review]
  Fedorov V.V. (1972) Theory of Optimal Experiments. Academic Press.
"""
from __future__ import annotations

import logging
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from app.engine.oed.identifiability import (
    FIMResult,
    SliceFIMConfig,
    compute_fim,
    analyze_fim,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OEDAction -- output artefact
# ---------------------------------------------------------------------------

class OEDAction(NamedTuple):
    """
    Result of OED protocol selection.

    Fields
    ------
    protocol_name         : str -- key of the winning candidate protocol.
    power_profile         : (T,) -- power time-series [W] of the winning protocol.
    d_optimality          : float -- log det(FIM) for winning protocol.
    fim_eigenvalues       : (D,) -- eigenvalues of FIM for winning protocol (ascending).
    unidentifiable_params : list[str] -- params still unidentifiable under winning protocol.
    all_fim_results       : dict[str, FIMResult] -- full FIM results per candidate.
    """
    protocol_name:         str
    power_profile:         jax.Array
    d_optimality:          float
    fim_eigenvalues:       jax.Array
    unidentifiable_params: list
    all_fim_results:       dict


# ---------------------------------------------------------------------------
# OEDProtocolGenerator
# ---------------------------------------------------------------------------

class OEDProtocolGenerator:
    """
    Select the D-optimal exercise protocol from a candidate pool.

    The generator evaluates the Fisher Information Matrix for each candidate
    protocol using jax.jacfwd sensitivities, then picks the protocol that
    maximises log det(FIM) (D-optimality).

    Parameters
    ----------
    forward_factory : callable
        Signature: (x0: jax.Array, protocol: jax.Array) -> forward_fn
        where forward_fn: (theta: jax.Array) -> y_traj: jax.Array.
        Both x0 and protocol are concrete arrays (not JAX-traced) at factory
        call time; forward_fn must be differentiable w.r.t. theta.

    config : SliceFIMConfig
        Contains param_names, theta_prior (linearisation point), R_inv, threshold_rel.

    Example
    -------
    >>> gen = make_cardio_oed_generator()
    >>> candidates = {
    ...     "zone2_20min":   jnp.full((20,), 220.0),
    ...     "sprint_3x30s":  _build_sprint_profile(),
    ...     "ramp_8min":     jnp.linspace(100.0, 350.0, 8),
    ... }
    >>> action = gen.select_optimal(candidates, x0=X0_CARDIO_DEFAULT)
    >>> print(action.protocol_name, action.d_optimality)
    >>> print("Still blind to:", action.unidentifiable_params)
    """

    def __init__(
        self,
        forward_factory: Callable,
        config:          SliceFIMConfig,
    ) -> None:
        self.forward_factory = forward_factory
        self.config          = config

    def evaluate_protocol(
        self,
        protocol: jax.Array,
        x0:       jax.Array,
    ) -> FIMResult:
        """
        Evaluate FIM for a single protocol at the config's theta_prior.

        Parameters
        ----------
        protocol : (T,) -- power time-series [W] for this candidate.
        x0       : (STATE_DIM,) -- initial physiological state.

        Returns
        -------
        FIMResult
        """
        forward_fn = self.forward_factory(x0, protocol)
        fim = compute_fim(forward_fn, self.config.theta_prior, self.config.R_inv)
        return analyze_fim(
            fim,
            list(self.config.param_names),
            self.config.threshold_rel,
        )

    def select_optimal(
        self,
        candidates: dict,
        x0:         jax.Array,
    ) -> OEDAction:
        """
        Evaluate all candidates and return the D-optimal OEDAction.

        Parameters
        ----------
        candidates : dict[str, jax.Array]
            Candidate protocols: name -> power_profile (T,).
        x0         : (STATE_DIM,) -- initial state.

        Returns
        -------
        OEDAction -- winning protocol + all FIM results.

        Raises
        ------
        ValueError if candidates is empty.
        """
        if not candidates:
            raise ValueError("OEDProtocolGenerator: candidates dict is empty.")

        all_results: dict[str, FIMResult] = {}
        for name, protocol in candidates.items():
            result = self.evaluate_protocol(jnp.asarray(protocol, dtype=jnp.float32), x0)
            all_results[name] = result
            logger.debug(
                "OED candidate %-20s | log det(FIM)=%+.2f | cond=%6.1e | unid=%s",
                name, result.log_det, result.condition_number,
                result.unidentifiable_params or "none",
            )

        best_name  = max(all_results, key=lambda n: all_results[n].log_det)
        best       = all_results[best_name]

        logger.info(
            "OED winner: %s  log_det=%.2f  (D=%d params, %d unidentifiable)",
            best_name, best.log_det,
            len(self.config.param_names), len(best.unidentifiable_params),
        )

        return OEDAction(
            protocol_name         = best_name,
            power_profile         = jnp.asarray(candidates[best_name], dtype=jnp.float32),
            d_optimality          = best.log_det,
            fim_eigenvalues       = best.eigenvalues,
            unidentifiable_params = best.unidentifiable_params,
            all_fim_results       = all_results,
        )


# ---------------------------------------------------------------------------
# Slice-specific forward factories
# ---------------------------------------------------------------------------

def make_cardio_fim_factory() -> Callable:
    """
    Return a forward_factory for the Cardiorespiratory slice (8-state ODE).

    The factory wraps simulate_hr_trajectory so that:
        forward_fn(log_theta) -> HR_traj (T,)
    where log_theta = [log(VO2_max_baseline), log(W_prime_capacity)].

    jax.jacfwd applied to forward_fn yields the sensitivity J (T, 2):
        J[t, 0] = d HR_t / d log(VO2_max_baseline)
        J[t, 1] = d HR_t / d log(W_prime_capacity)

    Returns
    -------
    factory : (x0: jax.Array, power_rates: jax.Array) -> forward_fn
    """
    from app.slices.cardiorespiratory.nlme import simulate_hr_trajectory

    def factory(x0: jax.Array, power_rates: jax.Array) -> Callable:
        def forward_fn(log_theta: jax.Array) -> jax.Array:
            return simulate_hr_trajectory(log_theta, x0, power_rates)
        return forward_fn

    return factory


def make_neuromuscular_fim_factory() -> Callable:
    """
    Return a forward_factory for the Neuromuscular Tissue slice (6-state ODE).

    The factory wraps _forward_simulate_v4 so that:
        forward_fn(log_theta) -> y_traj (T, 2)   [EMG_mV, SmO2_pct]
    where log_theta = [log(P_th_2), log(k_SERCA_base)].

    jax.jacfwd applied to forward_fn yields the sensitivity J (T, 2, 2):
        J[t, obs, p] = d y_t^obs / d log(theta^p)

    Returns
    -------
    factory : (x0: jax.Array, controls: jax.Array) -> forward_fn
    """
    from app.slices.neuromuscular_tissue.nlme import _forward_simulate_v4
    from app.slices.neuromuscular_tissue.ode import DEFAULT_V4_PARAMS
    from app.slices.neuromuscular_tissue.observation import DEFAULT_V4_OBS_PARAMS

    obs_params = DEFAULT_V4_OBS_PARAMS

    def factory(x0: jax.Array, controls: jax.Array) -> Callable:
        def forward_fn(log_theta: jax.Array) -> jax.Array:
            theta = jnp.exp(log_theta)
            return _forward_simulate_v4(theta, x0, controls, obs_params)
        return forward_fn

    return factory


# ---------------------------------------------------------------------------
# Convenience constructors for common slice generators
# ---------------------------------------------------------------------------

def make_cardio_oed_generator(
    x0:        jax.Array | None = None,
    sigma_obs: float = 5.0,
) -> OEDProtocolGenerator:
    """
    Build a ready-to-use OEDProtocolGenerator for the Cardiorespiratory slice.

    Parameters
    ----------
    x0        : (8,) initial state. Defaults to X0_CARDIO_DEFAULT (resting).
    sigma_obs : float -- HR observation noise [bpm]. Default 5.0.

    Returns
    -------
    OEDProtocolGenerator

    Example
    -------
    >>> from app.engine.oed.fisher_design import make_cardio_oed_generator
    >>> import jax.numpy as jnp
    >>>
    >>> gen = make_cardio_oed_generator()
    >>> candidates = {
    ...     "zone2_20min":  jnp.full((20,), 200.0),
    ...     "ramp_to_max":  jnp.linspace(100.0, 380.0, 20),
    ...     "sprint_5x1":   _make_sprint(n=5, power=400.0, T=20),
    ... }
    >>> action = gen.select_optimal(candidates, x0=gen._default_x0)
    """
    from app.slices.cardiorespiratory.nlme import _LOG_PRIOR_MEAN, THETA_NAMES
    from app.slices.cardiorespiratory.ode import X0_CARDIO_DEFAULT

    if x0 is None:
        x0 = X0_CARDIO_DEFAULT

    config = SliceFIMConfig(
        param_names   = tuple(THETA_NAMES),
        theta_prior   = _LOG_PRIOR_MEAN,
        R_inv         = jnp.float32(1.0 / (sigma_obs ** 2)),
        threshold_rel = 1e-4,
    )
    gen = OEDProtocolGenerator(make_cardio_fim_factory(), config)
    gen._default_x0 = x0   # attach for convenience (not used internally)
    return gen


def make_neuromuscular_oed_generator(
    x0:         jax.Array | None = None,
    sigma_emg:  float = 1.0,
    sigma_smo2: float = 1.0,
) -> OEDProtocolGenerator:
    """
    Build a ready-to-use OEDProtocolGenerator for the Neuromuscular Tissue slice.

    Parameters
    ----------
    x0         : (6,) initial NM state. Defaults to X0_NM_V4.
    sigma_emg  : float -- EMG noise [mV]. Default 1.0.
    sigma_smo2 : float -- SmO2 noise [%]. Default 1.0.

    Returns
    -------
    OEDProtocolGenerator
    """
    from app.slices.neuromuscular_tissue.nlme import PARAM_NAMES
    from app.slices.neuromuscular_tissue.ode import X0_NM_V4, DEFAULT_V4_PARAMS
    import math

    if x0 is None:
        x0 = X0_NM_V4

    # Log-prior means: log(population mean)
    log_P_th_2  = float(jnp.log(jnp.float32(250.0)))
    log_k_SERCA = float(jnp.log(jnp.float32(5.0)))
    theta_prior = jnp.array([log_P_th_2, log_k_SERCA], dtype=jnp.float32)

    R_inv = jnp.diag(jnp.array(
        [1.0 / sigma_emg**2, 1.0 / sigma_smo2**2],
        dtype=jnp.float32,
    ))

    config = SliceFIMConfig(
        param_names   = tuple(PARAM_NAMES),
        theta_prior   = theta_prior,
        R_inv         = R_inv,
        threshold_rel = 1e-4,
    )
    gen = OEDProtocolGenerator(make_neuromuscular_fim_factory(), config)
    gen._default_x0 = x0
    return gen
