"""
app/slices/metabolic_reds/filter.py — Module 13

L4 State Filter — Metabolic RED-S / Thyroid / Fatmax Slice
Unscented Kalman Filter for the 5-state metabolic system (daily timescale).

Architecture (HLD §4.3 — L4: State Estimation)
────────────────────────────────────────────────
State x ∈ ℝ⁵ (days):
    [EA_Pool, Free_T4, Free_T3, RMR_Multiplier, Fatmax_State]

Observations y ∈ ℝ³:
    [fT3_obs [pmol/L], fT4_obs [pmol/L], RMR_Proxy [kcal/day]]

Transition f(x, u) — ODE advance over dt_days (diffrax.Tsit5)
──────────────────────────────────────────────────────────────
Time constants (system NOT stiff → Tsit5 appropriate):
    EA_Pool          τ ≈  7 days  (k_ea_relax = 1/7)
    Free_T4          t½ ≈10 days  (k_t4_clear = 0.069)
    Free_T3          t½ ≈ 1 day   (k_t3_clear = 0.693)
    RMR_Multiplier   τ ≈14 days   (k_rmr_relax = 1/14)
    Fatmax_State     τ ≈21 days   (k_fatmax_relax = 1/21)

Sparse Assimilation
───────────────────
fT3_obs / fT4_obs: episodic lab draws (weeks to months between draws).
RMR_Proxy:         daily from wearable.
flag = 4 → R_channel × 1e8 (predict-only for that channel).

UKF parametrisation (Merwe & Wan 2000)
────────────────────────────────────────
    n = 5 → 2n+1 = 11 sigma points
    α = 0.5  (NOT 1e-3 — with n=5, α=1e-3 causes float32 cancellation)
        α=0.5: n+λ = 0.25×5−5+5 = 1.25, WM_0 = −3.0, WM_i = 0.40 — safe.
    β = 2.0, κ = 0.0

Process noise Q (diagonal, per day):
    EA_Pool          : (3.0 kcal/kg FFM)² — daily caloric variability
    Free_T4          : (0.5 pmol/L)²      — slow; tight
    Free_T3          : (0.3 pmol/L)²      — moderate daily variation
    RMR_Multiplier   : (0.02)²            — slow adaptation; tight
    Fatmax_State     : (0.02)²            — very slow mitochondrial change

Thermodynamic floor (MANDATORY on every update)
────────────────────────────────────────────────
After every posterior update, jnp.maximum(posterior_mean, 0.0) is applied
to all states to prevent thermodynamically impossible negative concentrations
or fractions. Additional physics bounds clip Free_T4 [0.1, 40], Free_T3 [0, 20],
RMR_Multiplier [0, 1.5], Fatmax_State [0, 1.2].

Fail-Loud contract
──────────────────
RuntimeError if posterior_mean contains NaN.
RuntimeError if posterior_cov diagonal contains negative values.

References
──────────
    Merwe & Wan (2000) Proc ASSPCC
    Achten J., Jeukendrup A.E. (2003) Sports Med 33(8):559–591
    Loucks A.B. (2003) J Sports Sci 21(10):879–883
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

try:
    from dynamax.nonlinear_gaussian_ssm import UnscentedKalmanFilter, ParamsNLGSSM
    _DYNAMAX_OK = True
except ImportError:
    UnscentedKalmanFilter = None   # type: ignore[assignment,misc]
    ParamsNLGSSM          = None   # type: ignore[assignment,misc]
    _DYNAMAX_OK = False

from app.slices.metabolic_reds.ode import (
    MetabolicRedsParams,
    DEFAULT_MR_PARAMS,
    X0_MR_DEFAULT,
    STATE_DIM,
    CTRL_DIM,
    metabolic_reds_ode,
    IDX_EA_POOL,
    IDX_FREE_T4,
    IDX_FREE_T3,
    IDX_RMR_MULT,
    IDX_FATMAX,
)
from app.slices.metabolic_reds.observation import (
    DEFAULT_MR_OBS_PARAMS,
    MetabolicRedsObsParams,
    OBS_DIM,
    h_mr,
    h_mr_sigma,
    R_MR_DEFAULT,
    inflate_R_mr,
)
from app.engine.assimilation.ukf_filter import GaussianState

logger = logging.getLogger(__name__)


# ── Initial covariance ────────────────────────────────────────────────────────
# Onboarding uncertainty before any personalisation.

P0_MR_DEFAULT: jax.Array = jnp.diag(jnp.array([
    100.0000,   # EA_Pool        σ² = (10.0 kcal/kg FFM)²  — uncertain at onboarding
      9.0000,   # Free_T4        σ² = (3.0 pmol/L)²         — population SD
      2.2500,   # Free_T3        σ² = (1.5 pmol/L)²         — population SD
      0.0100,   # RMR_Multiplier σ² = (0.10)²               — tight at onboarding
      0.0225,   # Fatmax_State   σ² = (0.15)²               — moderate uncertainty
], dtype=jnp.float32))


# ── UKF Merwe-Wan σ-point parameters (n = STATE_DIM = 5) ─────────────────────
#
# α = 0.5 (NOT 1e-3): with n=5, α=1e-3 gives n+λ ≈ 2.5e-6 → weights ±1e6 →
# float32 catastrophic cancellation.  α=0.5: n+λ=1.25, WM_0=-3.0, WM_i=0.40.
# Check: WM_0 + 2×5×WM_i = −3.0 + 4.0 = 1.0 ✓

_ALPHA: float = 0.5
_BETA:  float = 2.0
_KAPPA: float = 0.0
_N:     int   = STATE_DIM                            # 5

_LAM:   float = _ALPHA**2 * (_N + _KAPPA) - _N      # = 0.25×5 − 5 = −3.75
_WM_0:  float = _LAM  / (_N + _LAM)                 # = −3.75/1.25 = −3.0
_WC_0:  float = _WM_0 + (1.0 - _ALPHA**2 + _BETA)  # = −3.0 + 2.75 = −0.25
_WI:    float = 0.5   / (_N + _LAM)                 # = 0.5/1.25 = 0.40

_WM: jax.Array = jnp.array([_WM_0] + [_WI] * (2 * _N), dtype=jnp.float32)
_WC: jax.Array = jnp.array([_WC_0] + [_WI] * (2 * _N), dtype=jnp.float32)


# ── Process noise Q (diagonal, per day) ──────────────────────────────────────

_Q_DIAG: jax.Array = jnp.array([
    9.0000,   # EA_Pool         [kcal/kg FFM]² = (3.0)²   — daily caloric flux
    0.2500,   # Free_T4         [pmol/L]²      = (0.5)²   — slow; tight
    0.0900,   # Free_T3         [pmol/L]²      = (0.3)²   — moderate variation
    0.0004,   # RMR_Multiplier  [au]²          = (0.02)²  — slow adaptation
    0.0004,   # Fatmax_State    [au]²          = (0.02)²  — very slow; tight
], dtype=jnp.float32)

Q_DEFAULT: jax.Array = jnp.diag(_Q_DIAG)


# ── Transition parameters ─────────────────────────────────────────────────────

class MetabolicRedsTransitionParams(NamedTuple):
    """Full parameter set for the metabolic_reds ODE transition step."""
    ode:     MetabolicRedsParams = DEFAULT_MR_PARAMS
    dt_days: float               = 1.0


DEFAULT_MR_TRANSITION_PARAMS: MetabolicRedsTransitionParams = MetabolicRedsTransitionParams()


# ── JIT-safe ODE step ─────────────────────────────────────────────────────────

def _integrate_step(
    x:      jax.Array,
    u:      jax.Array,
    params: MetabolicRedsTransitionParams,
) -> jax.Array:
    """
    Integrate the 5-state metabolic_reds ODE for params.dt_days days.

    vmap-compatible: only x varies across sigma points.
    dt0 = 0.1 day (fixed; JIT-safe).
    """
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(metabolic_reds_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.asarray(params.dt_days, dtype=jnp.float32),
        dt0       = jnp.float32(0.1),
        y0        = x,
        args      = (u, params.ode),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 128,
    )
    return sol.ys[0]


# ── Pure-JAX UKF kernels ──────────────────────────────────────────────────────

def _sigma_points(mean: jax.Array, cov: jax.Array) -> jax.Array:
    """Merwe-Wan 2n+1 sigma points. Returns shape (11, 5)."""
    L   = jnp.linalg.cholesky((_N + _LAM) * cov)
    pos = mean[None, :] + L.T
    neg = mean[None, :] - L.T
    return jnp.concatenate([mean[None, :], pos, neg], axis=0)


def _recover_mean_cov(
    pts:       jax.Array,
    noise_cov: jax.Array,
    Wm:        jax.Array,
    Wc:        jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Recover (mean, cov) from propagated sigma points + additive noise."""
    mean = jnp.einsum("i,ij->j", Wm, pts)
    diff = pts - mean[None, :]
    cov  = jnp.einsum("i,ij,ik->jk", Wc, diff, diff) + noise_cov
    return mean, cov


@jax.jit
def _ukf_predict(
    mean:   jax.Array,
    cov:    jax.Array,
    u:      jax.Array,
    params: MetabolicRedsTransitionParams,
    Q:      jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """UKF predict: unscented transform through 5-state metabolic ODE. 11 sigma points."""
    Q_scaled  = Q * jnp.float32(params.dt_days)
    sigma     = _sigma_points(mean, cov)
    sigma_nxt = jax.vmap(
        _integrate_step, in_axes=(0, None, None)
    )(sigma, u, params)
    return _recover_mean_cov(sigma_nxt, Q_scaled, _WM, _WC)


@jax.jit
def _ukf_update(
    mean_pred:  jax.Array,
    cov_pred:   jax.Array,
    y_obs:      jax.Array,
    obs_params: MetabolicRedsObsParams,
    R:          jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """UKF measurement update ([fT3_obs, fT4_obs, RMR_Proxy])."""
    sigma   = _sigma_points(mean_pred, cov_pred)       # (11, 5)
    y_sigma = h_mr_sigma(sigma, obs_params)             # (11, 3)

    y_mean = jnp.einsum("i,ij->j", _WM, y_sigma)       # (3,)
    dy_s   = y_sigma  - y_mean[None, :]                 # (11, 3)
    dx_s   = sigma    - mean_pred[None, :]               # (11, 5)

    S_yy   = jnp.einsum("i,ij,ik->jk", _WC, dy_s, dy_s) + R   # (3,3)
    P_xy   = jnp.einsum("i,ij,ik->jk", _WC, dx_s, dy_s)        # (5,3)

    K              = P_xy @ jnp.linalg.inv(S_yy)
    innovation     = y_obs - y_mean
    posterior_mean = mean_pred + K @ innovation
    posterior_cov  = cov_pred  - K @ S_yy @ K.T

    return posterior_mean, posterior_cov


# ── Public filter class ───────────────────────────────────────────────────────

class MetabolicRedsStateFilter:
    """
    L4 Unscented Kalman Filter for the 5-state Metabolic RED-S system.

    Sparse assimilation:
      - fT3 / fT4: episodic lab draws (flag=4 on non-draw days).
      - RMR_Proxy: daily from wearable (flag 0–3 by coverage).

    Thermodynamic floor enforced after every update (jnp.maximum, all states >= 0).

    Fail-Loud contract
    ──────────────────
    RuntimeError on NaN posterior_mean.
    RuntimeError on non-PSD covariance.
    """

    def __init__(
        self,
        Q:          jax.Array | None = None,
        R:          jax.Array | None = None,
        obs_params: MetabolicRedsObsParams | None = None,
    ) -> None:
        self.Q          = Q          if Q          is not None else Q_DEFAULT
        self.R          = R          if R          is not None else R_MR_DEFAULT
        self.obs_params = obs_params if obs_params is not None else DEFAULT_MR_OBS_PARAMS

        if _DYNAMAX_OK:
            self._dyn_ukf = UnscentedKalmanFilter(STATE_DIM, OBS_DIM)
            logger.info("MetabolicRedsStateFilter — dynamax backend registered.")
        else:
            self._dyn_ukf = None
            logger.warning(
                "MetabolicRedsStateFilter — dynamax not installed; single-step only."
            )

    # ── Primary API: single-step update ──────────────────────────────────────

    def update_state(
        self,
        prior:         GaussianState,
        controls:      dict[str, float],
        dt_days:       float = 1.0,
        fT3_obs:       float = float("nan"),
        fT4_obs:       float = float("nan"),
        rmr_proxy:     float = float("nan"),
        quality_flags: tuple[int, int, int] = (4, 4, 4),
        params:        MetabolicRedsTransitionParams | None = None,
    ) -> GaussianState:
        """
        Assimilate one daily time step.

        Parameters
        ----------
        prior         : GaussianState(mean ∈ ℝ⁵, cov ∈ ℝ⁵ˣ⁵)
        controls      : dict with keys:
                        'hub_caloric_intake', 'hub_total_energy_expenditure',
                        'hub_training_stress', 'hub_fat_free_mass_kg'
        dt_days       : float — integration window [days]
        fT3_obs       : float [pmol/L]; NaN = not available (lab only)
        fT4_obs       : float [pmol/L]; NaN = not available (lab only)
        rmr_proxy     : float [kcal/day]; NaN = not available (wearable)
        quality_flags : (flag_fT3, flag_fT4, flag_RMR); 4 = predict-only
        params        : MetabolicRedsTransitionParams (defaults to population)

        Returns
        -------
        GaussianState — posterior (mean, cov)

        Raises
        ------
        RuntimeError on NaN posterior or non-PSD covariance.
        """
        if params is None:
            params = MetabolicRedsTransitionParams(ode=DEFAULT_MR_PARAMS, dt_days=dt_days)
        else:
            params = params._replace(dt_days=dt_days)

        u = jnp.array([
            float(controls.get("hub_caloric_intake",           2500.0)),
            float(controls.get("hub_total_energy_expenditure", 2500.0)),
            float(controls.get("hub_training_stress",          0.0)),
            float(controls.get("hub_fat_free_mass_kg",         60.0)),
        ], dtype=jnp.float32)

        # ── Predict ───────────────────────────────────────────────────────────
        mean_pred, cov_pred = _ukf_predict(prior.mean, prior.cov, u, params, self.Q)

        # ── Inflate R for unavailable channels ───────────────────────────────
        obs_vals = [fT3_obs, fT4_obs, rmr_proxy]
        flags    = list(quality_flags)
        for ch_idx, val in enumerate(obs_vals):
            if val != val:   # IEEE NaN check
                flags[ch_idx] = 4
        R_step = inflate_R_mr((flags[0], flags[1], flags[2]), self.R)

        # Replace NaN obs with predicted observation (pure predict for that channel)
        y_predicted = h_mr(mean_pred, self.obs_params)
        y_obs = jnp.array([
            y_predicted[0] if obs_vals[0] != obs_vals[0] else float(obs_vals[0]),
            y_predicted[1] if obs_vals[1] != obs_vals[1] else float(obs_vals[1]),
            y_predicted[2] if obs_vals[2] != obs_vals[2] else float(obs_vals[2]),
        ], dtype=jnp.float32)

        # ── Update ────────────────────────────────────────────────────────────
        posterior_mean, posterior_cov = _ukf_update(
            mean_pred, cov_pred, y_obs, self.obs_params, R_step
        )

        # ── Thermodynamic floor (MANDATORY) ──────────────────────────────────
        # Apply jnp.maximum to all states: concentrations and fractions >= 0.
        posterior_mean = jnp.maximum(posterior_mean, jnp.float32(0.0))

        # Additional physics bounds (upper limits and refined floors)
        posterior_mean = posterior_mean.at[IDX_FREE_T4].set(
            jnp.clip(posterior_mean[IDX_FREE_T4],
                     jnp.float32(0.1), jnp.float32(40.0))
        )
        posterior_mean = posterior_mean.at[IDX_FREE_T3].set(
            jnp.minimum(posterior_mean[IDX_FREE_T3], jnp.float32(20.0))
        )
        posterior_mean = posterior_mean.at[IDX_RMR_MULT].set(
            jnp.minimum(posterior_mean[IDX_RMR_MULT], jnp.float32(1.5))
        )
        posterior_mean = posterior_mean.at[IDX_FATMAX].set(
            jnp.minimum(posterior_mean[IDX_FATMAX], jnp.float32(1.2))
        )
        posterior_mean = posterior_mean.at[IDX_EA_POOL].set(
            jnp.minimum(posterior_mean[IDX_EA_POOL], jnp.float32(80.0))
        )

        # ── Fail-Loud checks ──────────────────────────────────────────────────
        if bool(jnp.any(jnp.isnan(posterior_mean))):
            raise RuntimeError(
                "MetabolicRedsStateFilter.update_state: posterior_mean contains NaN. "
                f"fT3_obs={fT3_obs}, fT4_obs={fT4_obs}, rmr_proxy={rmr_proxy}, "
                f"quality_flags={quality_flags}, dt_days={dt_days}."
            )
        diag = jnp.diag(posterior_cov)
        if bool(jnp.any(diag < jnp.float32(-1e-6))):
            raise RuntimeError(
                "MetabolicRedsStateFilter.update_state: posterior_cov has negative "
                "diagonal — filter diverged. Increase Q or R."
            )

        posterior_cov = jnp.float32(0.5) * (posterior_cov + posterior_cov.T)
        return GaussianState(mean=posterior_mean, cov=posterior_cov)

    # ── Convenience: build initial GaussianState ──────────────────────────────

    @staticmethod
    def make_initial_state(
        mean: jax.Array | None = None,
        cov:  jax.Array | None = None,
    ) -> GaussianState:
        """Return GaussianState from defaults or provided arrays."""
        m = mean if mean is not None else jnp.array(X0_MR_DEFAULT, dtype=jnp.float32)
        c = cov  if cov  is not None else P0_MR_DEFAULT
        return GaussianState(mean=m, cov=c)

    # ── Batch filtering (dynamax backend) ─────────────────────────────────────

    def build_dynamax_params(
        self,
        transition_params: MetabolicRedsTransitionParams,
        initial_mean:      jax.Array | None = None,
        initial_cov:       jax.Array | None = None,
    ) -> "ParamsNLGSSM":
        if not _DYNAMAX_OK:
            raise RuntimeError("dynamax>=0.1 required for build_dynamax_params.")
        _tp         = transition_params
        _obs_params = self.obs_params

        def _dynamics_fn(x: jax.Array, u: jax.Array) -> jax.Array:
            return _integrate_step(x, u, _tp)

        def _emission_fn(x: jax.Array, _: jax.Array) -> jax.Array:
            return h_mr(x, _obs_params)

        _m0 = initial_mean if initial_mean is not None else jnp.array(
            X0_MR_DEFAULT, dtype=jnp.float32
        )
        return ParamsNLGSSM(
            initial_mean       = _m0,
            initial_covariance = initial_cov if initial_cov is not None else P0_MR_DEFAULT,
            dynamics_function  = _dynamics_fn,
            dynamics_covariance= self.Q,
            emission_function  = _emission_fn,
            emission_covariance= self.R,
        )
