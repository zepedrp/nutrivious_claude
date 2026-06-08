"""
app/slices/gonadal_axis/observation.py

L2 Observation Model — Gonadal Axis Slice (polymorphic: female / male)

Observation array y ∈ ℝ⁴ (shared layout; NaN where metric does not apply)
──────────────────────────────────────────────────────────────────────────
  y[0]  E2_pg_mL      plasma estradiol          [pg/mL]    — both sexes
  y[1]  P4_ng_mL      plasma progesterone       [ng/mL]    — female only (NaN males)
  y[2]  BBT_C         basal body temperature    [°C]       — female only (NaN males)
  y[3]  Total_T_ng_dL plasma total testosterone [ng/dL]    — male only  (NaN females)

Observation equations
─────────────────────
Female:
  y[0] = Estradiol_state                                        + ε_E2
  y[1] = Progesterone_state                                     + ε_P4
  y[2] = BBT_base + ΔBBT × sigmoid(k_BBT × (P4 − P4_BBT_thresh)) + ε_BBT
  y[3] = NaN

Male:
  y[0] = k_aromatase × Testosterone                            + ε_E2
  y[1] = NaN
  y[2] = NaN
  y[3] = Testosterone                                           + ε_T

Basal Body Temperature (BBT)
─────────────────────────────
BBT rises ~0.3–0.5°C after ovulation due to progesterone (Guermandi 2001).
Model: BBT = BBT_base + ΔBBT_max × sigmoid(k_BBT × (P4 − P4_BBT_threshold))
Where P4_BBT_threshold ≈ 3 ng/mL, ΔBBT_max ≈ 0.4°C.

Missing-observation protocol
─────────────────────────────
NaN channels (sex-inappropriate metrics or missing wearable data) trigger
R × 1e8 inflation in the UKF update — predict-only for those dimensions.

References
──────────
  Guermandi E. et al. (2001) Fertil Steril 75:1052-1058
    DOI 10.1016/S0015-0282(01)01790-X  [BBT ovulation detection]
  Sinha-Hikim I. et al. (1998) J Clin Endocrinol Metab 83:1313-8
    [male T → E2 aromatization fraction]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from app.slices.gonadal_axis.female_ode import (
    IDX_F_E2, IDX_F_P4,
    DEFAULT_FEMALE_PARAMS,
)
from app.slices.gonadal_axis.male_ode import (
    IDX_M_T,
    DEFAULT_MALE_PARAMS,
    male_algebraic_outputs,
    P4_ADRENAL_BASAL_MALE,
)

# ── Observation channel indices ───────────────────────────────────────────────
IDX_OBS_E2   = 0   # E2       [pg/mL]   — both sexes
IDX_OBS_P4   = 1   # P4       [ng/mL]   — female only
IDX_OBS_BBT  = 2   # BBT      [°C]      — female only
IDX_OBS_T    = 3   # Total_T  [ng/dL]   — male only

OBS_DIM_GONADAL: int = 4

# Sentinel — channels not applicable to a sex
_NAN = float("nan")


# ── Quality-flag R inflation (shared with cardiorespiratory convention) ───────

_QUALITY_INFLATE: tuple[float, ...] = (1.0, 2.0, 10.0, 100.0, 1.0e8)


def inflate_R_for_quality(quality_flag: int, R_base: jax.Array) -> jax.Array:
    flag  = max(0, min(4, quality_flag))
    return R_base * jnp.float32(_QUALITY_INFLATE[flag])


# ── Observation parameter container ──────────────────────────────────────────

class GonadalObsParams(NamedTuple):
    """
    Observation noise + kinetic parameters for the gonadal slice.

    Fields
    ------
    sigma_E2_pg_mL    : E2 assay noise [pg/mL]
                        Immunoassay CV ≈ 10% at 100 pg/mL → σ ≈ 10 pg/mL
    sigma_P4_ng_mL    : P4 assay noise [ng/mL]
                        Typical σ ≈ 0.5 ng/mL (Guermandi 2001)
    sigma_BBT_C       : wearable thermometry noise [°C]
                        Oral thermometer SD ≈ 0.05°C; wearable ≈ 0.10°C
    sigma_T_ng_dL     : T assay noise [ng/dL]
                        LC-MS/MS σ ≈ 20 ng/dL (Bhasin 2021)
    BBT_base_C        : basal body temperature at low P4 [°C]
    BBT_delta_C       : maximum BBT rise in luteal phase [°C] (Guermandi 2001)
    BBT_P4_threshold  : P4 level at sigmoid midpoint [ng/mL]
    BBT_k             : sigmoid steepness [ng/mL⁻¹]
    k_aromatase       : male T → E2 conversion [pg/mL per ng/dL]
    """
    sigma_E2_pg_mL:   float = 10.0    # pg/mL
    sigma_P4_ng_mL:   float = 0.5     # ng/mL
    sigma_BBT_C:      float = 0.10    # °C
    sigma_T_ng_dL:    float = 20.0    # ng/dL
    BBT_base_C:       float = 36.5    # °C
    BBT_delta_C:      float = 0.4     # °C (Guermandi 2001)
    BBT_P4_threshold: float = 3.0     # ng/mL
    BBT_k:            float = 2.0     # ng/mL⁻¹
    k_aromatase:      float = 0.05    # pg/mL per ng/dL (Sinha-Hikim 1998)


DEFAULT_OBS_PARAMS_GONADAL = GonadalObsParams()

# Default R matrices per sex (diagonal)
_p = DEFAULT_OBS_PARAMS_GONADAL

R_DEFAULT_FEMALE: jax.Array = jnp.diag(jnp.array([
    _p.sigma_E2_pg_mL ** 2,    # E2    [pg/mL]²
    _p.sigma_P4_ng_mL ** 2,    # P4    [ng/mL]²
    _p.sigma_BBT_C    ** 2,    # BBT   [°C]²
    1.0e8,                     # T     — not observed in females (predict-only)
], dtype=jnp.float32))

R_DEFAULT_MALE: jax.Array = jnp.diag(jnp.array([
    _p.sigma_E2_pg_mL ** 2,    # E2    — observed via assay
    1.0e8,                     # P4    — not observed in males
    1.0e8,                     # BBT   — not applicable
    _p.sigma_T_ng_dL  ** 2,    # T     [ng/dL]²
], dtype=jnp.float32))


# ── Sex-specific observation functions ───────────────────────────────────────

@jax.jit
def h_gonadal_female(
    x:          jax.Array,
    obs_params: GonadalObsParams,
) -> jax.Array:
    """
    h(x) for female 7-state vector → y ∈ ℝ⁴.

    x : (7,) — [GnRH, LH, FSH, E2, P4, FM, LM]
    Returns [E2_pg_mL, P4_ng_mL, BBT_C, 0.0]

    Channel 3 (Total_T) returns 0.0 (not NaN) so that σ-point propagation
    inside the UKF does not produce NaN in y_mean / S_yy.
    The corresponding R diagonal entry is 1e8 (R_DEFAULT_FEMALE), making
    the Kalman gain for this channel negligible (predict-only).
    """
    E2 = x[IDX_F_E2]
    P4 = x[IDX_F_P4]

    # Basal body temperature: sigmoid step at luteal-phase P4 rise
    bbt = (
        jnp.float32(obs_params.BBT_base_C)
        + jnp.float32(obs_params.BBT_delta_C) * jax.nn.sigmoid(
            jnp.float32(obs_params.BBT_k) * (P4 - jnp.float32(obs_params.BBT_P4_threshold))
        )
    )

    # Channel 3 = 0.0 (not NaN): R[3,3]=1e8 makes it predict-only; NaN would
    # propagate through UKF sigma-point weighted mean → S_yy → K → NaN state.
    return jnp.array([E2, P4, bbt, jnp.float32(0.0)], dtype=jnp.float32)


@jax.jit
def h_gonadal_male(
    x:          jax.Array,
    obs_params: GonadalObsParams,
) -> jax.Array:
    """
    h(x) for male 5-state vector → y ∈ ℝ⁴.

    x : (5,) — [GnRH, LH, FSH, T, LC]
    Returns [E2_pg_mL, 0.0, 0.0, Total_T_ng_dL]

    Channels 1 (P4) and 2 (BBT) return 0.0 (not NaN) for UKF numerical safety.
    Corresponding R diagonals are 1e8, making them predict-only.
    """
    T   = x[IDX_M_T]
    E2  = jnp.float32(obs_params.k_aromatase) * T

    return jnp.array([E2, jnp.float32(0.0), jnp.float32(0.0), T], dtype=jnp.float32)


def h_gonadal(
    x:           jax.Array,
    is_female:   float,
    obs_params:  GonadalObsParams,
) -> jax.Array:
    """
    Polymorphic observation function.

    Parameters
    ----------
    x          : shape (7,) if female, (5,) if male — caller must pass
                 the correct sex-specific state vector.
    is_female  : 1.0 = female, 0.0 = male  (compile-time constant for JIT)
    obs_params : GonadalObsParams

    Returns
    -------
    y : shape (OBS_DIM_GONADAL=4,)
        Channels not applicable to the sex are NaN.

    Notes
    -----
    For UKF sigma-point propagation (vmap), the state dim differs by sex.
    Use h_gonadal_female or h_gonadal_male directly inside sex-specific
    UKF instances. h_gonadal is a convenience dispatcher for single calls.
    """
    if bool(is_female):
        return h_gonadal_female(x, obs_params)
    else:
        return h_gonadal_male(x, obs_params)


# ── Sigma-point vmapped versions ─────────────────────────────────────────────

def h_gonadal_female_sigma(
    sigma_pts:  jax.Array,
    obs_params: GonadalObsParams,
) -> jax.Array:
    """shape (2n+1, 7) → (2n+1, 4)"""
    return jax.vmap(h_gonadal_female, in_axes=(0, None))(sigma_pts, obs_params)


def h_gonadal_male_sigma(
    sigma_pts:  jax.Array,
    obs_params: GonadalObsParams,
) -> jax.Array:
    """shape (2n+1, 5) → (2n+1, 4)"""
    return jax.vmap(h_gonadal_male, in_axes=(0, None))(sigma_pts, obs_params)


# ── Observation dict → array ──────────────────────────────────────────────────

def obs_dict_to_array_gonadal(obs_dict: dict[str, float]) -> jax.Array:
    """
    Convert observation dict to (OBS_DIM_GONADAL,) JAX array.

    Keys
    ----
    "E2_obs_pg_mL"    : float or NaN
    "P4_obs_ng_mL"    : float or NaN
    "BBT_obs_C"       : float or NaN
    "Total_T_obs_ng_dL" : float or NaN
    """
    e2  = float(obs_dict.get("E2_obs_pg_mL",       _NAN))
    p4  = float(obs_dict.get("P4_obs_ng_mL",        _NAN))
    bbt = float(obs_dict.get("BBT_obs_C",            _NAN))
    t   = float(obs_dict.get("Total_T_obs_ng_dL",    _NAN))
    return jnp.array([e2, p4, bbt, t], dtype=jnp.float32)


def inflate_R_per_channel_gonadal(
    R_base: jax.Array,
    y_obs:  jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Inflate R per-channel for NaN observations (missing or sex-N/A).
    NaN channel → R × 1e8; y_obs channel → 0 (pure predict-only).
    """
    nan_mask   = jnp.isnan(y_obs)
    scale      = jnp.where(nan_mask, jnp.float32(1e8), jnp.float32(1.0))
    R_inflated = R_base * jnp.diag(scale)
    y_safe     = jnp.where(nan_mask, jnp.float32(0.0), y_obs)
    return R_inflated, y_safe
