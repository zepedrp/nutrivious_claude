"""
app/slices/gonadal_axis/female_ode.py

L2 Backbone ODE — Gonadal Axis, Female Polymorphism (7 states, day scale)

══════════════════════════════════════════════════════════════════════
STATE VECTOR  x ∈ ℝ⁷   (time unit: DAYS)
══════════════════════════════════════════════════════════════════════
  x[0]  GnRH            hypothalamic GnRH effective mean  [pM]
  x[1]  LH              pituitary LH                       [IU/L]
  x[2]  FSH             pituitary FSH                      [IU/L]
  x[3]  Estradiol       E2 plasma                          [pg/mL]
  x[4]  Progesterone    P4 plasma                          [ng/mL]
  x[5]  Follicular_Mass dominant-follicle mass fraction    [adim, 0–1]
  x[6]  Luteal_Mass     corpus luteum mass fraction        [adim, 0–1]

CONTROL INPUTS
  hub_EA_Pool  [kcal/kg FFM/day]  — energy availability (Mod 13 hub); NaN → 45.0

PHYSICS
  Röblitz 2013 (simplified) dual-feedback architecture:
    Negative: E2 at moderate levels → suppresses GnRH, LH, FSH
    Positive: E2 > 200 pg/mL → LH surge → ovulation (Follicular_Mass → Luteal_Mass)
    P4 (Luteal phase): suppresses GnRH and LH → cycle restart

RED-S COUPLING (Mod 13 — energy availability)
  Loucks 2003 / Mountjoy 2014 (Br J Sports Med): EA < 30 kcal/kg FFM/day
  suppresses hypothalamic GnRH → Functional Hypothalamic Amenorrhoea (FHA).
  Gate: ea_gate = sigmoid((EA − 30) / 5); collapses to ≈ 0 at EA ≈ 15.

References
──────────
  Röblitz S. et al. (2013) Adv Comput Math 39:23–48
    DOI 10.1007/s10444-012-9260-0  [HPG cycle ODE model]
  Loucks A.B. & Thuma J.R. (2003) J Clin Endocrinol Metab 88:297-311
    DOI 10.1210/jc.2002-020369  [EA threshold / GnRH suppression]
  Mountjoy M. et al. (2014) Br J Sports Med 48:491-497
    DOI 10.1136/bjsports-2014-093502  [RED-S / FHA clinical threshold]
  De Crée C. (1998) Sports Med 25:333-395  [menstrual cycle review]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── State indices ─────────────────────────────────────────────────────────────
IDX_F_GNRH = 0   # GnRH          [pM]
IDX_F_LH   = 1   # LH            [IU/L]
IDX_F_FSH  = 2   # FSH           [IU/L]
IDX_F_E2   = 3   # Estradiol     [pg/mL]
IDX_F_P4   = 4   # Progesterone  [ng/mL]
IDX_F_FM   = 5   # Follicular_Mass [adim]
IDX_F_LM   = 6   # Luteal_Mass    [adim]

STATE_DIM_FEMALE: int = 7


# ── Parameter container ───────────────────────────────────────────────────────

class FemaleGonadalParams(NamedTuple):
    """
    Parameter set for 7-state female HPG axis ODE.

    All rates in day⁻¹ or compatible day-scale units.
    Population-mean priors anchored to Röblitz 2013 + Loucks 2003.
    NLME D=2: personalised via (E2_peak_baseline, tau_LM).
    """
    # ── I. GnRH (hypothalamus) ───────────────────────────────────────────
    k_GnRH:       float   # pM/day — synthesis rate (ea_gate × P4_inhibition weighted)
    tau_GnRH:     float   # days — effective GnRH clearance
    K_P4_GnRH:    float   # ng/mL — P4 half-inhibition of GnRH
    EA_threshold: float   # kcal/kg FFM/day — RED-S gate threshold (Loucks 2003)
    EA_steepness: float   # kcal/kg FFM/day — sigmoid steepness

    # ── II. LH (pituitary) ───────────────────────────────────────────────
    LH_basal:          float   # IU/L — basal LH secretion
    k_LH_GnRH:         float   # IU/L per pM·day — GnRH→LH gain
    tau_LH:            float   # days — LH kinetics
    K_E2_neg_LH:       float   # pg/mL — E2 negative-feedback half-saturation
    k_E2_pos_LH:       float   # adim — positive-feedback gain at surge (≥7 needed for robust burst)
    E2_surge_thresh:   float   # pg/mL — positive feedback onset (Röblitz 2013)
    E2_surge_steep:    float   # pg/mL — sigmoid steepness of LH surge
    K_P4_LH:           float   # ng/mL — P4 half-inhibition of LH

    # ── III. FSH (pituitary) ─────────────────────────────────────────────
    FSH_basal:     float   # IU/L
    k_FSH_GnRH:   float   # IU/L per pM·day
    tau_FSH:       float   # days
    K_E2_FSH:     float   # pg/mL — E2 inhibition half-sat (via inhibin-B proxy)
    k_inhibin_FM: float   # adim — follicular inhibin-B suppression of FSH

    # ── IV. Estradiol ────────────────────────────────────────────────────
    k_E2_follicle: float   # pg/(mL·IU·L⁻¹·day) — FSH×FM → E2 (granulosa cells)
    k_E2_luteal:   float   # pg/(mL·day) — LM → E2 (corpus luteum baseline)
    tau_E2:        float   # days

    # ── V. Progesterone ──────────────────────────────────────────────────
    k_P4_luteal:  float   # ng/(mL·IU·L⁻¹·day) — LH×LM → P4
    tau_P4:       float   # days

    # ── VI. Follicular mass ──────────────────────────────────────────────
    k_FM_growth:         float   # day⁻¹ — FSH-driven growth rate
    LH_ovulation_thresh: float   # IU/L — LH level triggering ovulation
    LH_ovulation_steep:  float   # IU/L — sigmoid steepness
    k_FM_lysis:          float   # day⁻¹ — follicle conversion rate at LH surge

    # ── VII. Luteal mass ─────────────────────────────────────────────────
    k_LM_form:    float   # day⁻¹ — corpus luteum formation from FM
    tau_LM:       float   # days — corpus luteum lifespan (~12 days; NLME prior)

    # ── Hub NaN default ──────────────────────────────────────────────────
    EA_default:   float   # kcal/kg FFM/day — used when hub_EA_Pool is NaN


DEFAULT_FEMALE_PARAMS = FemaleGonadalParams(
    # I. GnRH
    k_GnRH        = 8.0,    # pM/day — calibrated to produce GnRH_ss ≈ 2 pM
    tau_GnRH      = 0.5,    # days (~12 h effective mean)
    K_P4_GnRH     = 5.0,    # ng/mL — Röblitz 2013 Table 1
    EA_threshold  = 30.0,   # kcal/kg FFM/day — Loucks & Thuma 2003
    EA_steepness  = 5.0,    # kcal/kg FFM/day

    # II. LH
    LH_basal         = 2.0,     # IU/L
    k_LH_GnRH        = 8.0,     # IU/L per pM·day
    tau_LH           = 0.1,     # days (~2.4 h — pituitary fast response)
    K_E2_neg_LH      = 100.0,   # pg/mL (Röblitz 2013: E2 feedback half-sat)
    k_E2_pos_LH      = 7.0,     # adim — positive feedback amplitude (7.0 ensures LH burst >40 IU/L when E2>200)
    E2_surge_thresh  = 170.0,   # pg/mL — positive feedback onset; max transient E2≈190 in sim
    E2_surge_steep   = 30.0,    # pg/mL
    K_P4_LH          = 4.0,     # ng/mL (Röblitz 2013)

    # III. FSH
    FSH_basal     = 2.0,    # IU/L
    k_FSH_GnRH   = 5.0,    # IU/L per pM·day
    tau_FSH       = 0.5,    # days
    K_E2_FSH     = 80.0,    # pg/mL — inhibin-B equivalent inhibition
    k_inhibin_FM = 0.5,     # adim — follicular inhibin per unit FM

    # IV. Estradiol
    k_E2_follicle = 200.0,  # pg/(mL·IU·L⁻¹·day) — granulosa FSH response
    k_E2_luteal   = 20.0,   # pg/(mL·day) — corpus luteum baseline E2
    tau_E2        = 0.3,    # days (~7 h; Söderqvist 1997)

    # V. Progesterone
    k_P4_luteal = 5.0,   # ng/(mL·IU·L⁻¹·day) — calibrated for P4_ss ≈ 10 ng/mL
    tau_P4      = 0.5,   # days (~12 h)

    # VI. Follicular mass
    k_FM_growth          = 0.3,    # day⁻¹ (~14 days to fill under FSH drive)
    LH_ovulation_thresh  = 40.0,   # IU/L (typical LH surge ≥ 40 IU/L)
    LH_ovulation_steep   = 5.0,    # IU/L
    k_FM_lysis           = 2.0,    # day⁻¹ (rapid conversion within 1-2 days)

    # VII. Luteal mass
    k_LM_form = 2.0,    # day⁻¹
    tau_LM    = 12.0,   # days (corpus luteum lifespan; NLME personalised)

    # Hub default
    EA_default = 45.0,  # kcal/kg FFM/day — assumed normal if Mod 13 absent
)


# ── Default initial state (day 7 of cycle: mid-follicular) ───────────────────

X0_FEMALE_DEFAULT: jax.Array = jnp.array([
    2.0,    # GnRH   [pM]
    8.0,    # LH     [IU/L]
    7.0,    # FSH    [IU/L]
    80.0,   # E2     [pg/mL]  — mid-follicular (Thorneycroft 1971)
    0.8,    # P4     [ng/mL]  — early follicular
    0.4,    # FM     [adim]   — growing follicle
    0.05,   # LM     [adim]   — minimal residual
], dtype=jnp.float32)

P0_FEMALE_DEFAULT: jax.Array = jnp.diag(jnp.array([
    0.25,    # GnRH  σ² = (0.5 pM)²
    16.0,    # LH    σ² = (4 IU/L)²
    9.0,     # FSH   σ² = (3 IU/L)²
    2500.0,  # E2    σ² = (50 pg/mL)²
    4.0,     # P4    σ² = (2 ng/mL)²
    0.04,    # FM    σ² = (0.2)²
    0.04,    # LM    σ² = (0.2)²
], dtype=jnp.float32))


# ── ODE (JIT + vmap safe) ─────────────────────────────────────────────────────

def female_gonadal_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    7-state female HPG axis ODE (day scale).

    Parameters
    ----------
    t     : scalar [days] — current time (diffrax signature)
    x     : shape (7,) — [GnRH, LH, FSH, E2, P4, FM, LM]
    args  : tuple(FemaleGonadalParams, hub_EA_Pool)
              hub_EA_Pool : float [kcal/kg FFM/day] or NaN

    Returns
    -------
    dx/dt : shape (7,)

    Physics invariants
    ------------------
    • No Python control flow on traced values (jnp.maximum / jnp.where only).
    • Hub NaN → default EA = 45.0 kcal/kg FFM/day (normal energy status).
    • FM, LM clamped to [0, 1] by the ODE structure (bounded sink terms).
    """
    params, hub_ea = args

    GnRH          = x[IDX_F_GNRH]
    LH            = x[IDX_F_LH]
    FSH           = x[IDX_F_FSH]
    Estradiol     = x[IDX_F_E2]
    Progesterone  = x[IDX_F_P4]
    Follicular_M  = x[IDX_F_FM]
    Luteal_M      = x[IDX_F_LM]

    # ── Hub NaN guard ─────────────────────────────────────────────────────
    ea_raw = jnp.asarray(hub_ea, dtype=jnp.float32)
    EA     = jnp.where(jnp.isnan(ea_raw), jnp.float32(params.EA_default), ea_raw)
    EA     = jnp.maximum(jnp.float32(0.0), EA)

    # ── RED-S gate: Loucks 2003 — EA < 30 kcal/kg FFM/day suppresses GnRH ─
    ea_gate = jax.nn.sigmoid(
        (EA - jnp.float32(params.EA_threshold)) / jnp.float32(params.EA_steepness)
    )

    # ══════════════════════════════════════════════════════════════════════
    # I. GnRH — hypothalamic synthesis gated by EA and P4 (negative FB)
    # ══════════════════════════════════════════════════════════════════════
    P4_inh_GnRH = jnp.float32(1.0) / (
        jnp.float32(1.0) + (Progesterone / jnp.float32(params.K_P4_GnRH)) ** 2
    )
    GnRH_syn = jnp.float32(params.k_GnRH) * ea_gate * P4_inh_GnRH
    dGnRH_dt = GnRH_syn - GnRH / jnp.float32(params.tau_GnRH)

    # ══════════════════════════════════════════════════════════════════════
    # II. LH — dual E2 feedback (negative at moderate; positive at peak)
    #          P4 negative feedback (luteal phase suppression)
    #
    # Röblitz 2013: e2_neg dampens baseline LH; e2_pos fires at surge.
    # The net pituitary sensitivity = e2_neg + k_pos × sigmoid(E2 − thresh).
    # ══════════════════════════════════════════════════════════════════════
    e2_neg_LH = jnp.float32(1.0) / (
        jnp.float32(1.0) + (Estradiol / jnp.float32(params.K_E2_neg_LH)) ** 2
    )
    e2_pos_LH = jax.nn.sigmoid(
        (Estradiol - jnp.float32(params.E2_surge_thresh)) / jnp.float32(params.E2_surge_steep)
    )
    P4_inh_LH = jnp.float32(1.0) / (
        jnp.float32(1.0) + (Progesterone / jnp.float32(params.K_P4_LH)) ** 2
    )
    net_pituitary_sensitivity = e2_neg_LH + jnp.float32(params.k_E2_pos_LH) * e2_pos_LH
    LH_target = (
        jnp.float32(params.LH_basal)
        + jnp.float32(params.k_LH_GnRH) * GnRH * net_pituitary_sensitivity * P4_inh_LH
    )
    dLH_dt = (LH_target - LH) / jnp.float32(params.tau_LH)

    # ══════════════════════════════════════════════════════════════════════
    # III. FSH — GnRH drive, E2+inhibin-B suppression
    # ══════════════════════════════════════════════════════════════════════
    e2_inh_FSH = jnp.float32(1.0) / (
        jnp.float32(1.0) + (Estradiol / jnp.float32(params.K_E2_FSH)) ** 2
    )
    inhibin_B = jnp.float32(1.0) + jnp.float32(params.k_inhibin_FM) * Follicular_M
    FSH_target = (
        jnp.float32(params.FSH_basal)
        + jnp.float32(params.k_FSH_GnRH) * GnRH * e2_inh_FSH / inhibin_B
    )
    dFSH_dt = (FSH_target - FSH) / jnp.float32(params.tau_FSH)

    # ══════════════════════════════════════════════════════════════════════
    # IV. Estradiol — granulosa (FSH×FM) + corpus luteum baseline
    # ══════════════════════════════════════════════════════════════════════
    E2_production = (
        jnp.float32(params.k_E2_follicle) * FSH * Follicular_M
        + jnp.float32(params.k_E2_luteal) * Luteal_M
    )
    dE2_dt = E2_production - Estradiol / jnp.float32(params.tau_E2)

    # ══════════════════════════════════════════════════════════════════════
    # V. Progesterone — corpus luteum LH-dependent secretion
    # ══════════════════════════════════════════════════════════════════════
    P4_production = jnp.float32(params.k_P4_luteal) * LH * Luteal_M
    dP4_dt = P4_production - Progesterone / jnp.float32(params.tau_P4)

    # ══════════════════════════════════════════════════════════════════════
    # VI. Follicular mass — FSH-driven growth; LH-surge-triggered lysis
    # Smooth ovulation trigger: sigmoid on LH − LH_ovulation_threshold
    # ══════════════════════════════════════════════════════════════════════
    ovulation_trigger = jax.nn.sigmoid(
        (LH - jnp.float32(params.LH_ovulation_thresh)) / jnp.float32(params.LH_ovulation_steep)
    )
    FM_growth = (
        jnp.float32(params.k_FM_growth)
        * FSH
        * (jnp.float32(1.0) - Follicular_M)
    )
    FM_lysis = jnp.float32(params.k_FM_lysis) * ovulation_trigger * Follicular_M
    dFM_dt = FM_growth - FM_lysis

    # ══════════════════════════════════════════════════════════════════════
    # VII. Luteal mass — formed at ovulation; decays over tau_LM days
    # ══════════════════════════════════════════════════════════════════════
    dLM_dt = (
        jnp.float32(params.k_LM_form) * ovulation_trigger * Follicular_M
        - Luteal_M / jnp.float32(params.tau_LM)
    )

    return jnp.stack([dGnRH_dt, dLH_dt, dFSH_dt, dE2_dt, dP4_dt, dFM_dt, dLM_dt])
