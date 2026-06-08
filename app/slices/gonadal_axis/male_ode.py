"""
app/slices/gonadal_axis/male_ode.py

L2 Backbone ODE — Gonadal Axis, Male Polymorphism (5 states, day scale)

══════════════════════════════════════════════════════════════════════
STATE VECTOR  x ∈ ℝ⁵   (time unit: DAYS)
══════════════════════════════════════════════════════════════════════
  x[0]  GnRH             hypothalamic GnRH effective mean  [pM]
  x[1]  LH               pituitary LH                      [IU/L]
  x[2]  FSH              pituitary FSH                     [IU/L]
  x[3]  Testosterone     plasma total T                    [ng/dL]
  x[4]  Leydig_Capacity  Leydig cell functional reserve    [adim, 0–1]

CONTROL INPUTS
  hub_EA_Pool  [kcal/kg FFM/day]  — energy availability (Mod 13 hub); NaN → 45.0

ALGEBRAIC OUTPUTS (not states — derived at observation step)
  Estradiol  = k_aromatase × Testosterone           [pg/mL]
  Progesterone = P4_adrenal_basal (constant)        [ng/mL] — 0.15 ng/mL

PHYSICS
  Veldhuis 1994 / Keenan 2003 simplified:
    • Strict negative feedback of Testosterone on GnRH (hypothalamus) and
      LH/FSH (pituitary). No positive-feedback arc in males.
    • Leydig_Capacity encodes chronic HPG axis reserve — declines with RED-S.
    • Testosterone production ∝ LH × Leydig_Capacity (Leydig cell response).

RED-S COUPLING (Mod 13)
  EA < 30 kcal/kg FFM/day → ea_gate → 0 → GnRH synthesis collapses →
  LH/FSH drop → Testosterone drops (Hypogonadal Male Condition).
  Leydig_Capacity tracks EA over 30-day window (slow recovery).

References
──────────
  Veldhuis J.D. et al. (1994) Recent Prog Horm Res 49:363-395
    [male HPG pulsatile model framework]
  Keenan D.M. & Veldhuis J.D. (2003) Am J Physiol 285:E1039-E1050
    DOI 10.1152/ajpendo.00099.2003  [dose-response feedback model]
  Hackney A.C. (2020) Sports Med 50:971-976
    DOI 10.1007/s40279-020-01259-y  [exercise-induced testosterone suppression]
  Sinha-Hikim I. et al. (1998) J Clin Endocrinol Metab 83:1313-8
    [aromatization fraction: ~0.3-0.5% of T → E2]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── State indices ─────────────────────────────────────────────────────────────
IDX_M_GNRH = 0   # GnRH              [pM]
IDX_M_LH   = 1   # LH                [IU/L]
IDX_M_FSH  = 2   # FSH               [IU/L]
IDX_M_T    = 3   # Testosterone      [ng/dL]
IDX_M_LC   = 4   # Leydig_Capacity   [adim, 0–1]

STATE_DIM_MALE: int = 5

# Algebraic constant — adrenal baseline progesterone in males
P4_ADRENAL_BASAL_MALE: float = 0.15   # ng/mL (De Ronde 2006)


# ── Parameter container ───────────────────────────────────────────────────────

class MaleGonadalParams(NamedTuple):
    """
    Parameter set for 5-state male HPG axis ODE.

    All rates in day⁻¹ or compatible day-scale units.
    NLME D=2: personalised via (T_baseline, Leydig_reserve).
    """
    # ── I. GnRH (hypothalamus) ───────────────────────────────────────────
    k_GnRH_M:     float   # pM/day — synthesis rate
    tau_GnRH_M:   float   # days
    K_T_GnRH:     float   # ng/dL — T negative-feedback half-sat
    n_T_GnRH:     float   # Hill exponent on T feedback
    EA_threshold: float   # kcal/kg FFM/day — RED-S gate threshold
    EA_steepness: float   # kcal/kg FFM/day

    # ── II. LH (pituitary) ───────────────────────────────────────────────
    LH_basal_M:   float   # IU/L
    k_LH_GnRH_M:  float   # IU/L per pM·day
    tau_LH_M:     float   # days
    K_T_LH:       float   # ng/dL — T negative-feedback half-sat on LH
    n_T_LH:       float   # Hill exponent

    # ── III. FSH (pituitary) ─────────────────────────────────────────────
    FSH_basal_M:  float   # IU/L
    k_FSH_GnRH_M: float   # IU/L per pM·day
    tau_FSH_M:    float   # days
    K_T_FSH:      float   # ng/dL — T (+ inhibin-B proxy) feedback

    # ── IV. Testosterone ─────────────────────────────────────────────────
    k_T_LH:       float   # ng·dL⁻¹·IU⁻¹·L·day — Leydig steroidogenesis
    tau_T:        float   # days — effective T clearance (0.1 d ≈ pulsatile mean)
    T_ref:        float   # ng/dL — reference T (population mean)

    # ── V. Leydig capacity ───────────────────────────────────────────────
    tau_LC:       float   # days — Leydig adaptation time constant (~30 days)

    # ── Aromatization ────────────────────────────────────────────────────
    k_aromatase:  float   # pg/mL per ng/dL — peripheral T→E2 conversion

    # ── Hub NaN default ──────────────────────────────────────────────────
    EA_default_M: float   # kcal/kg FFM/day


DEFAULT_MALE_PARAMS = MaleGonadalParams(
    # I. GnRH
    k_GnRH_M     = 8.0,    # pM/day — calibrated: GnRH_ss ≈ 2 pM at T=600
    tau_GnRH_M   = 0.5,    # days
    K_T_GnRH     = 600.0,  # ng/dL — half-sat (Keenan 2003)
    n_T_GnRH     = 2.0,    # Hill exponent
    EA_threshold  = 30.0,  # Loucks 2003
    EA_steepness  = 5.0,

    # II. LH
    LH_basal_M   = 1.0,    # IU/L
    k_LH_GnRH_M  = 6.5,    # calibrated: LH_ss ≈ 5 IU/L at GnRH=2, T=600
    tau_LH_M     = 0.05,   # days (~1.2 h — fast pituitary response)
    K_T_LH       = 400.0,  # ng/dL (Veldhuis 1994)
    n_T_LH       = 2.0,

    # III. FSH
    FSH_basal_M   = 1.5,
    k_FSH_GnRH_M  = 5.0,
    tau_FSH_M     = 0.5,   # days (slower than LH)
    K_T_FSH       = 350.0, # ng/dL

    # IV. Testosterone
    k_T_LH   = 1200.0,  # ng·dL⁻¹·IU⁻¹·L·day — calibrated: T_ss ≈ 600 ng/dL
    tau_T    = 0.1,     # days (~2.4 h effective; Bhasin 2001)
    T_ref    = 600.0,   # ng/dL — adult male population mean

    # V. Leydig capacity
    tau_LC   = 30.0,    # days — chronic adaptation (Hackney 2020)

    # Aromatization: T=600 ng/dL → E2 ≈ 30 pg/mL (Sinha-Hikim 1998)
    k_aromatase  = 0.05,   # pg/mL per ng/dL

    EA_default_M = 45.0,
)


# ── Default initial state (adult eugonadal male) ──────────────────────────────

X0_MALE_DEFAULT: jax.Array = jnp.array([
    2.0,    # GnRH              [pM]
    5.0,    # LH                [IU/L]
    5.0,    # FSH               [IU/L]
    600.0,  # Testosterone      [ng/dL]
    1.0,    # Leydig_Capacity   [adim]
], dtype=jnp.float32)

P0_MALE_DEFAULT: jax.Array = jnp.diag(jnp.array([
    0.25,     # GnRH  σ² = (0.5 pM)²
    4.0,      # LH    σ² = (2 IU/L)²
    4.0,      # FSH   σ² = (2 IU/L)²
    40000.0,  # T     σ² = (200 ng/dL)² — high onboarding uncertainty
    0.04,     # LC    σ² = (0.2)²
], dtype=jnp.float32))


# ── ODE (JIT + vmap safe) ─────────────────────────────────────────────────────

def male_gonadal_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    5-state male HPG axis ODE (day scale).

    Parameters
    ----------
    t     : scalar [days]
    x     : shape (5,) — [GnRH, LH, FSH, Testosterone, Leydig_Capacity]
    args  : tuple(MaleGonadalParams, hub_EA_Pool)

    Returns
    -------
    dx/dt : shape (5,)

    Algebraic outputs (not in state vector)
    ----------------------------------------
    Estradiol  = k_aromatase × x[IDX_M_T]   [pg/mL]
    Progesterone = P4_ADRENAL_BASAL_MALE      [ng/mL] (constant)

    Use male_algebraic_outputs(x, params) to compute these for the
    observation model and hub variable export.
    """
    params, hub_ea = args

    GnRH           = x[IDX_M_GNRH]
    LH             = x[IDX_M_LH]
    FSH            = x[IDX_M_FSH]
    Testosterone   = x[IDX_M_T]
    Leydig_Cap     = x[IDX_M_LC]

    # ── Hub NaN guard ─────────────────────────────────────────────────────
    ea_raw = jnp.asarray(hub_ea, dtype=jnp.float32)
    EA     = jnp.where(jnp.isnan(ea_raw), jnp.float32(params.EA_default_M), ea_raw)
    EA     = jnp.maximum(jnp.float32(0.0), EA)

    # ── RED-S gate ────────────────────────────────────────────────────────
    ea_gate = jax.nn.sigmoid(
        (EA - jnp.float32(params.EA_threshold)) / jnp.float32(params.EA_steepness)
    )

    # ══════════════════════════════════════════════════════════════════════
    # I. GnRH — strict negative feedback of T (Veldhuis 1994)
    # ══════════════════════════════════════════════════════════════════════
    T_norm_GnRH = Testosterone / jnp.float32(params.K_T_GnRH)
    T_neg_GnRH  = jnp.float32(1.0) / (
        jnp.float32(1.0) + T_norm_GnRH ** jnp.float32(params.n_T_GnRH)
    )
    dGnRH_dt = (
        jnp.float32(params.k_GnRH_M) * ea_gate * T_neg_GnRH
        - GnRH / jnp.float32(params.tau_GnRH_M)
    )

    # ══════════════════════════════════════════════════════════════════════
    # II. LH — GnRH stimulated; T negative feedback (Keenan 2003)
    # ══════════════════════════════════════════════════════════════════════
    T_norm_LH = Testosterone / jnp.float32(params.K_T_LH)
    T_neg_LH  = jnp.float32(1.0) / (
        jnp.float32(1.0) + T_norm_LH ** jnp.float32(params.n_T_LH)
    )
    LH_target = (
        jnp.float32(params.LH_basal_M)
        + jnp.float32(params.k_LH_GnRH_M) * GnRH * T_neg_LH
    )
    dLH_dt = (LH_target - LH) / jnp.float32(params.tau_LH_M)

    # ══════════════════════════════════════════════════════════════════════
    # III. FSH — GnRH stimulated; T+inhibin negative feedback
    # ══════════════════════════════════════════════════════════════════════
    T_norm_FSH = Testosterone / jnp.float32(params.K_T_FSH)
    T_neg_FSH  = jnp.float32(1.0) / (
        jnp.float32(1.0) + T_norm_FSH ** 2
    )
    FSH_target = (
        jnp.float32(params.FSH_basal_M)
        + jnp.float32(params.k_FSH_GnRH_M) * GnRH * T_neg_FSH
    )
    dFSH_dt = (FSH_target - FSH) / jnp.float32(params.tau_FSH_M)

    # ══════════════════════════════════════════════════════════════════════
    # IV. Testosterone — LH × Leydig_Capacity steroidogenesis
    # ══════════════════════════════════════════════════════════════════════
    T_production = jnp.float32(params.k_T_LH) * LH * Leydig_Cap
    dT_dt = T_production - Testosterone / jnp.float32(params.tau_T)

    # ══════════════════════════════════════════════════════════════════════
    # V. Leydig capacity — EA-dependent reserve (chronic 30-day adaptation)
    #    At normal EA: LC_target = 1.0 (full reserve).
    #    Under RED-S:  LC_target = ea_gate → 0 over ~30 days (Hackney 2020).
    # ══════════════════════════════════════════════════════════════════════
    dLC_dt = (ea_gate - Leydig_Cap) / jnp.float32(params.tau_LC)

    return jnp.stack([dGnRH_dt, dLH_dt, dFSH_dt, dT_dt, dLC_dt])


# ── Algebraic outputs (for observation model and hub export) ──────────────────

def male_algebraic_outputs(
    x:      jax.Array,
    params: MaleGonadalParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute algebraic (non-state) male gonadal outputs.

    Returns
    -------
    E2_pg_mL  : scalar — plasma estradiol via peripheral aromatization
    P4_ng_mL  : scalar — adrenal baseline progesterone (constant)
    """
    T         = x[IDX_M_T]
    E2_pg_mL  = jnp.float32(params.k_aromatase) * T
    P4_ng_mL  = jnp.float32(P4_ADRENAL_BASAL_MALE)
    return E2_pg_mL, P4_ng_mL
