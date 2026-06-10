"""
app/slices/gonadal_axis/filter.py

L4 State Filter — Gonadal Axis Slice (polymorphic: female n=7, male n=5)

GonadalStateFilter(is_female=True/False) instantiates a sex-specific UKF:
  • Female: 7-state vector, female_gonadal_ode transition, 1-day integration step
  • Male:   5-state vector, male_gonadal_ode transition, 1-day integration step

UKF parametrisation (Merwe-Wan α=0.10 for float32 stability)
  Female (n=7): λ=−6.93, WM_0=−99, WI=7.14
  Male   (n=5): λ=−4.95, WM_0=−99, WI=10.0
  α=0.10 chosen for float32 safety (same argument as cardiorespiratory filter).

MANDATORY physical clamps (jnp.maximum — no Python branching)
  All hormonal concentrations and masses ≥ 0 (cannot be negative by definition).
  Mass fractions (FM, LM, LC) clipped to [0, 1].

Hub variable exports (identical keys for both sexes)
  Hub_Testosterone  [ng/dL] — state (male) or adrenal baseline constant (female)
  Hub_Estradiol     [pg/mL] — state (female) or algebraic aromatization (male)
  Hub_Progesterone  [ng/mL] — state (female) or adrenal basal constant (male)

Fail-Loud contract
  RuntimeError if posterior_mean contains NaN.

References
──────────
  Merwe R. & Wan E.A. (2000) Proc. ASSPCC — UKF σ-point parametrisation
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

from app.slices.gonadal_axis.female_ode import (
    female_gonadal_ode,
    FemaleGonadalParams,
    DEFAULT_FEMALE_PARAMS,
    X0_FEMALE_DEFAULT,
    P0_FEMALE_DEFAULT,
    STATE_DIM_FEMALE,
    IDX_F_GNRH, IDX_F_LH, IDX_F_FSH, IDX_F_E2, IDX_F_P4, IDX_F_FM, IDX_F_LM,
)
from app.slices.gonadal_axis.male_ode import (
    male_gonadal_ode,
    MaleGonadalParams,
    DEFAULT_MALE_PARAMS,
    X0_MALE_DEFAULT,
    P0_MALE_DEFAULT,
    STATE_DIM_MALE,
    IDX_M_GNRH, IDX_M_LH, IDX_M_FSH, IDX_M_T, IDX_M_LC,
    male_algebraic_outputs,
    P4_ADRENAL_BASAL_MALE,
)
from app.slices.gonadal_axis.observation import (
    GonadalObsParams,
    DEFAULT_OBS_PARAMS_GONADAL,
    R_DEFAULT_FEMALE,
    R_DEFAULT_MALE,
    h_gonadal_female,
    h_gonadal_male,
    h_gonadal_female_sigma,
    h_gonadal_male_sigma,
    inflate_R_per_channel_gonadal,
    obs_dict_to_array_gonadal,
)
from app.engine.assimilation.ukf_filter import GaussianState
from app.engine.hubs import GaussianMsg, msg_from_scalar

logger = logging.getLogger(__name__)

# Female adrenal baseline T (not in 7-state model; used for hub export)
_T_FEMALE_ADRENAL_ng_dL: float = 30.0   # ng/dL (mid-range female reference)

_COV_JITTER: float = 1e-6


# ── Transition parameter containers ──────────────────────────────────────────

class FemaleGonadalTransition(NamedTuple):
    params:      FemaleGonadalParams = DEFAULT_FEMALE_PARAMS
    dt_days:     float = 1.0
    hub_EA_Pool: float = 45.0


class MaleGonadalTransition(NamedTuple):
    params:      MaleGonadalParams = DEFAULT_MALE_PARAMS
    dt_days:     float = 1.0
    hub_EA_Pool: float = 45.0


DEFAULT_FEMALE_TRANSITION = FemaleGonadalTransition()
DEFAULT_MALE_TRANSITION   = MaleGonadalTransition()


# ── Default process noise Q (day scale) ──────────────────────────────────────

_Q_DIAG_FEMALE: jax.Array = jnp.array([
    0.01,    # GnRH   [pM²]       σ ≈ 0.1 pM/day
    1.0,     # LH     [(IU/L)²]   σ ≈ 1 IU/L/day
    0.25,    # FSH    [(IU/L)²]   σ ≈ 0.5 IU/L/day
    400.0,   # E2     [(pg/mL)²]  σ ≈ 20 pg/mL/day
    0.04,    # P4     [(ng/mL)²]  σ ≈ 0.2 ng/mL/day
    0.01,    # FM     [adim²]     σ ≈ 0.1/day
    0.01,    # LM     [adim²]     σ ≈ 0.1/day
], dtype=jnp.float32)

_Q_DIAG_MALE: jax.Array = jnp.array([
    0.01,    # GnRH   [pM²]
    0.25,    # LH     [(IU/L)²]
    0.25,    # FSH    [(IU/L)²]
    2500.0,  # T      [(ng/dL)²]  σ ≈ 50 ng/dL/day
    0.0025,  # LC     [adim²]     σ ≈ 0.05/day
], dtype=jnp.float32)

Q_DEFAULT_FEMALE: jax.Array = jnp.diag(_Q_DIAG_FEMALE)
Q_DEFAULT_MALE:   jax.Array = jnp.diag(_Q_DIAG_MALE)


# ── UKF σ-point utilities ─────────────────────────────────────────────────────

def _ukf_weights(n: int, alpha: float = 0.10, beta: float = 2.0, kappa: float = 0.0):
    lam   = alpha**2 * (n + kappa) - n
    wm_0  = lam / (n + lam)
    wc_0  = wm_0 + 1.0 - alpha**2 + beta
    wi    = 0.5 / (n + lam)
    WM    = jnp.array([wm_0] + [wi]*(2*n), dtype=jnp.float32)
    WC    = jnp.array([wc_0] + [wi]*(2*n), dtype=jnp.float32)
    return lam, WM, WC


def _sigma_points(mean: jax.Array, cov: jax.Array, lam: float) -> jax.Array:
    n       = mean.shape[0]
    cov_reg = 0.5*(cov + cov.T) + _COV_JITTER * jnp.eye(n, dtype=jnp.float32)
    L       = jnp.linalg.cholesky((n + lam) * cov_reg)
    pos     = mean[None, :] + L.T
    neg     = mean[None, :] - L.T
    return jnp.concatenate([mean[None, :], pos, neg], axis=0)  # (2n+1, n)


def _recover_mean_cov(pts, noise_cov, WM, WC):
    mean = jnp.einsum("i,ij->j", WM, pts)
    diff = pts - mean[None, :]
    cov  = jnp.einsum("i,ij,ik->jk", WC, diff, diff) + noise_cov
    return mean, cov


# ── Sex-specific 1-day ODE step ───────────────────────────────────────────────

def _integrate_female(x: jax.Array, trans: FemaleGonadalTransition) -> jax.Array:
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(female_gonadal_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.float32(trans.dt_days),
        dt0       = jnp.float32(0.1),
        y0        = x,
        args      = (trans.params, trans.hub_EA_Pool),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 64,
    )
    return sol.ys[0]


def _integrate_male(x: jax.Array, trans: MaleGonadalTransition) -> jax.Array:
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(male_gonadal_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(0.0),
        t1        = jnp.float32(trans.dt_days),
        dt0       = jnp.float32(0.1),
        y0        = x,
        args      = (trans.params, trans.hub_EA_Pool),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = 64,
    )
    return sol.ys[0]


# ── Physical clamps (MANDATORY — jnp.maximum only) ───────────────────────────

def _clamp_female(mean: jax.Array) -> jax.Array:
    """All states ≥ 0; mass fractions FM, LM clipped to [0, 1]."""
    mean = jnp.maximum(mean, jnp.float32(0.0))
    mean = mean.at[IDX_F_FM].set(jnp.clip(mean[IDX_F_FM], jnp.float32(0.0), jnp.float32(1.0)))
    mean = mean.at[IDX_F_LM].set(jnp.clip(mean[IDX_F_LM], jnp.float32(0.0), jnp.float32(1.0)))
    return mean


def _clamp_male(mean: jax.Array) -> jax.Array:
    """All states ≥ 0; Leydig_Capacity clipped to [0, 1]."""
    mean = jnp.maximum(mean, jnp.float32(0.0))
    mean = mean.at[IDX_M_LC].set(jnp.clip(mean[IDX_M_LC], jnp.float32(0.0), jnp.float32(1.0)))
    return mean


# ── Main filter class ─────────────────────────────────────────────────────────

class GonadalStateFilter:
    """
    L4 Unscented Kalman Filter for the gonadal axis slice.

    Polymorphic: is_female=True → 7-state female HPG; is_female=False → 5-state male HPG.

    update_state() assimilates one daily observation bundle.
    compute_hub_exports() returns identical hub keys for both sexes.

    Physical clamps (MANDATORY per CLAUDE.md §3)
    ─────────────────────────────────────────────
    Applied post-update via jnp.maximum(x, 0):
    • All hormonal concentrations ≥ 0 (GnRH, LH, FSH, E2, P4, T — cannot be negative)
    • Mass fractions FM, LM, LC ∈ [0, 1]

    Fail-Loud contract
    ──────────────────
    RuntimeError if posterior_mean contains NaN.
    """

    def __init__(
        self,
        is_female:  bool,
        Q:          jax.Array | None = None,
        R:          jax.Array | None = None,
        obs_params: GonadalObsParams | None = None,
    ) -> None:
        self.is_female = is_female
        self.n         = STATE_DIM_FEMALE if is_female else STATE_DIM_MALE

        # UKF weights
        self._lam, self._WM, self._WC = _ukf_weights(self.n)

        # Sex-specific defaults
        if is_female:
            self.Q          = Q          if Q          is not None else Q_DEFAULT_FEMALE
            self.R          = R          if R          is not None else R_DEFAULT_FEMALE
            self._integrate = _integrate_female
            self._clamp     = _clamp_female
            self._h_sigma   = h_gonadal_female_sigma
            self._h         = h_gonadal_female
        else:
            self.Q          = Q          if Q          is not None else Q_DEFAULT_MALE
            self.R          = R          if R          is not None else R_DEFAULT_MALE
            self._integrate = _integrate_male
            self._clamp     = _clamp_male
            self._h_sigma   = h_gonadal_male_sigma
            self._h         = h_gonadal_male

        self.obs_params = obs_params if obs_params is not None else DEFAULT_OBS_PARAMS_GONADAL
        logger.info(
            "GonadalStateFilter(is_female=%s) n=%d  α=0.10",
            is_female, self.n
        )

    def _predict(
        self,
        mean: jax.Array,
        cov:  jax.Array,
        trans,          # FemaleGonadalTransition | MaleGonadalTransition
    ) -> tuple[jax.Array, jax.Array]:
        sigma      = _sigma_points(mean, cov, self._lam)
        sigma_next = jax.vmap(self._integrate, in_axes=(0, None))(sigma, trans)
        mean_pred, cov_pred = _recover_mean_cov(sigma_next, self.Q, self._WM, self._WC)
        cov_pred = 0.5 * (cov_pred + cov_pred.T)
        return mean_pred, cov_pred

    def _update(
        self,
        mean_pred: jax.Array,
        cov_pred:  jax.Array,
        y_obs:     jax.Array,
        R:         jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        sigma    = _sigma_points(mean_pred, cov_pred, self._lam)
        y_sigma  = self._h_sigma(sigma, self.obs_params)
        y_mean   = jnp.einsum("i,ij->j", self._WM, y_sigma)
        dy_s     = y_sigma - y_mean[None, :]
        dx_s     = sigma   - mean_pred[None, :]
        S_yy     = jnp.einsum("i,ij,ik->jk", self._WC, dy_s, dy_s) + R
        P_xy     = jnp.einsum("i,ij,ik->jk", self._WC, dx_s, dy_s)
        K              = P_xy @ jnp.linalg.inv(S_yy)
        innovation     = y_obs - y_mean
        post_mean      = mean_pred + K @ innovation
        post_cov       = cov_pred  - K @ S_yy @ K.T
        return post_mean, post_cov

    # ── Primary public API ────────────────────────────────────────────────

    def update_state(
        self,
        prior:        GaussianState,
        observations: dict[str, float],
        transition:   FemaleGonadalTransition | MaleGonadalTransition | None = None,
        quality_flag: int = 0,
    ) -> GaussianState:
        """
        Assimilate one daily observation bundle.

        Parameters
        ----------
        prior        : GaussianState — state at previous day
        observations : dict — keys from obs_dict_to_array_gonadal
                       (E2_obs_pg_mL, P4_obs_ng_mL, BBT_obs_C, Total_T_obs_ng_dL)
                       Missing / non-applicable → NaN → R×1e8 inflation
        transition   : FemaleGonadalTransition or MaleGonadalTransition
                       If None, default is used.
        quality_flag : int [0-4]

        Returns
        -------
        GaussianState — posterior (mean, cov)

        Raises
        ------
        RuntimeError if posterior_mean contains NaN.
        """
        if transition is None:
            transition = (
                DEFAULT_FEMALE_TRANSITION if self.is_female
                else DEFAULT_MALE_TRANSITION
            )

        # ── Predict ───────────────────────────────────────────────────────
        mean_pred, cov_pred = self._predict(prior.mean, prior.cov, transition)

        # ── Observation + per-channel R inflation ─────────────────────────
        y_raw            = obs_dict_to_array_gonadal(observations)
        R_step           = self.R * float(
            (1.0, 2.0, 10.0, 100.0, 1e8)[max(0, min(4, quality_flag))]
        )
        R_inf, y_safe    = inflate_R_per_channel_gonadal(R_step, y_raw)

        # Replace NaN y with predicted (zero-innovation predict-only).
        # Safety nan_to_num: h functions return 0.0 for sex-inapplicable channels,
        # but guard here against any residual NaN before passing to _update.
        y_pred_obs = self._h(mean_pred, self.obs_params)
        y_pred_obs = jnp.nan_to_num(y_pred_obs, nan=0.0)
        y_final    = jnp.where(jnp.isnan(y_raw), y_pred_obs, y_safe)

        # ── Update ────────────────────────────────────────────────────────
        post_mean, post_cov = self._update(mean_pred, cov_pred, y_final, R_inf)

        # ── MANDATORY physical clamps (jnp.maximum — no Python branching) ─
        post_mean = self._clamp(post_mean)

        # ── Fail-Loud divergence check ────────────────────────────────────
        if bool(jnp.any(jnp.isnan(post_mean))):
            raise RuntimeError(
                f"GonadalStateFilter(is_female={self.is_female}).update_state: "
                f"posterior_mean contains NaN. obs={observations}"
            )

        post_cov = 0.5 * (post_cov + post_cov.T)
        return GaussianState(mean=post_mean, cov=post_cov)

    # ── Slow-axis hub publications (GaussianMsg — for SlowAxisOrchestrator) ─

    def compute_slow_hub_publications(
        self,
        state: GaussianState,
    ) -> dict[str, GaussianMsg]:
        """
        Compute GaussianMsg publications for the slow-axis hub boundary conditions.

        Published fields (HubState keys)
        ---------------------------------
        testosterone      [ng/dL] — anabolic drive source
        basal_temp_offset [°C]    — progesterone-driven TR setpoint elevation
        anabolic_drive    [0-1]   — normalised T/E2 anabolic index

        Physics
        -------
        basal_temp_offset:
            Linear proxy validated by Baker & Jeukendrup (2001) J Physiol and
            Forsyth et al. (2007): P4 at mid-luteal peak (~15 ng/mL) elevates
            basal body temperature by ~0.3–0.4°C.
            offset = clamp(0.4 × P4 / 15, 0, 0.5)  [°C]
            Uncertainty propagation: sigma2_offset = (0.4/15)^2 * sigma2_P4

        anabolic_drive:
            Female: E2-driven anabolism modulated by P4 (De Crée 1998).
                    drive = clamp(E2/200 × (1 – 0.2 × P4/15), 0, 1)
            Male:   Testosterone-driven anabolism (Bhasin 2001).
                    drive = clamp(T/700, 0, 1)
        """
        x   = state.mean
        cov = state.cov

        _P4_LUTEAL_PEAK = 15.0   # ng/mL — mid-luteal reference (Baker 2001)

        if self.is_female:
            p4      = float(x[IDX_F_P4])
            p4_var  = float(cov[IDX_F_P4, IDX_F_P4])
            e2      = float(x[IDX_F_E2])
            e2_var  = float(cov[IDX_F_E2, IDX_F_E2])
            t_val   = _T_FEMALE_ADRENAL_ng_dL
            t_var   = 25.0   # adrenal baseline T nearly constant in females

            # anabolic_drive: E2-driven with mild P4 inhibition (De Crée 1998)
            p4_norm = min(p4 / _P4_LUTEAL_PEAK, 1.0)
            ad_mean = float(max(0.0, min(1.0, (e2 / 200.0) * (1.0 - 0.2 * p4_norm))))
            ad_var  = e2_var / (200.0 ** 2) + 1e-6
        else:
            E2_arr, _ = male_algebraic_outputs(x, DEFAULT_MALE_PARAMS)
            p4        = float(P4_ADRENAL_BASAL_MALE)
            p4_var    = 1e-4   # male adrenal P4 nearly constant
            e2_var    = float(cov[IDX_M_T, IDX_M_T]) * (DEFAULT_MALE_PARAMS.k_aromatase ** 2)
            t_val     = float(x[IDX_M_T])
            t_var     = float(cov[IDX_M_T, IDX_M_T])

            # anabolic_drive: T-normalised by population-mean eugonadal T (Bhasin 2001)
            ad_mean = float(max(0.0, min(1.0, t_val / 700.0)))
            ad_var  = t_var / (700.0 ** 2) + 1e-6

        # basal_temp_offset [°C]: progesterone-driven setpoint elevation
        bto_mean = float(max(0.0, min(0.5, 0.4 * p4 / _P4_LUTEAL_PEAK)))
        bto_var  = (0.4 / _P4_LUTEAL_PEAK) ** 2 * p4_var + 1e-6

        return {
            "testosterone":      msg_from_scalar(t_val,    t_var    + 1e-8),
            "basal_temp_offset": msg_from_scalar(bto_mean, bto_var  + 1e-8),
            "anabolic_drive":    msg_from_scalar(ad_mean,  ad_var   + 1e-8),
        }

    # ── Hub variable export (identical keys for both sexes) ───────────────

    def compute_hub_exports(self, state: GaussianState) -> dict[str, float]:
        """
        Export hub variables with identical keys for female and male.

        Returns
        -------
        {
          "Hub_Testosterone":  float [ng/dL]
          "Hub_Estradiol":     float [pg/mL]
          "Hub_Progesterone":  float [ng/mL]
        }

        Female: Hub_Testosterone = adrenal baseline (constant, ~30 ng/dL);
                Hub_Estradiol from state; Hub_Progesterone from state.
        Male:   Hub_Testosterone from state; Hub_Estradiol via aromatization;
                Hub_Progesterone = adrenal baseline constant (0.15 ng/mL).
        """
        x = state.mean
        if self.is_female:
            return {
                "Hub_Testosterone": _T_FEMALE_ADRENAL_ng_dL,
                "Hub_Estradiol":    float(x[IDX_F_E2]),
                "Hub_Progesterone": float(x[IDX_F_P4]),
            }
        else:
            E2, P4 = male_algebraic_outputs(x, DEFAULT_MALE_PARAMS)
            return {
                "Hub_Testosterone": float(x[IDX_M_T]),
                "Hub_Estradiol":    float(E2),
                "Hub_Progesterone": float(P4),
            }
