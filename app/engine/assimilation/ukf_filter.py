"""
app/engine/assimilation/ukf_filter.py

UKF primitives shared by all L4 slice filters.

Each slice (cardiorespiratory, metabolic_reds, neural_cognitive, gonadal_axis,
etc.) implements its own UKF class with slice-specific state dimensions, ODE
kernels, and observation models.  This module provides only the stateless
container GaussianState used by every slice filter.

References
----------
Merwe & Wan (2000) Proc. ASSPCC — Unscented Kalman Filter
Wan & Merwe (2000) Kalman Filtering Ch. 7
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── Gaussian state container ──────────────────────────────────────────────────

class GaussianState(NamedTuple):
    """
    First two moments of the filter's state belief.

    Attributes
    ----------
    mean : shape (n,)      — posterior mean μ
    cov  : shape (n, n)    — posterior covariance Σ
    """
    mean: jax.Array
    cov:  jax.Array
