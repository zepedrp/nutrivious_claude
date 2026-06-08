"""
app/slices/hematological/ode.py  Hematological Slice V3.0
5-state hourly ODE: RBC homeostasis, plasma expansion, hepcidin-iron axis.

State x in R^5  (time unit: HOURS):
  x[0] RBC_Mass_g       Total RBC wet mass        [g]
  x[1] Plasma_Vol_L     Plasma volume             [L]
  x[2] EPO_mIU_mL       Erythropoietin            [mIU/mL]
  x[3] Ferritin_ug_L    Ferritin (iron stores)    [ug/L]
  x[4] Hemolysis_Tox_au Hemolysis byproduct       [au]

Hub inputs u in R^4  (NaN-guarded, defaults: 0, 0, 1.0, 0.0):
  u[0] hub_Hypoxia_pct    Hypoxic stimulus      [%]
  u[1] hub_Load_au        Mechanical load       [0-1]
  u[2] hub_IL6_pgmL       Systemic IL-6         [pg/mL]
  u[3] hub_Iron_intake_mg Iron available        [mg/h equivalent]

Physics (hourly):
  a) dEPO      = k_hypoxia * Hypoxia - k_epo_clear * EPO
  b) fe_abs    = Iron / (1 + k_hepcidin * IL6)
  c) E         = k_epo_rbc * EPO * Ferritin / (Km_Fe + Ferritin)  (erythropoiesis)
  d) dFerritin = fe_abs - k_iron_use * E
  e) Hemolysis = (k_base + k_footstrike * Load) * RBC_Mass
  f) dRBC      = E - Hemolysis
  g) dPlasma   = k_pv_exp * Load - k_pv_decay * (Plasma - PV_base)
  h) dHem_Tox  = k_tox * Hemolysis - k_clear_tox * Hem_Tox

Algebraic:
  Hub_Hematocrit_pct = RBC_Vol_mL / (RBC_Vol_mL + PV_mL) * 100
    where RBC_Vol_mL = RBC_Mass_g * 0.88  (1g ~ 0.88 mL; density 1.09 g/mL)
  Hub_O2_Capacity = Hgb_g_dL * 1.34
"""
from __future__ import annotations

from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp

# ── State indices ──────────────────────────────────────────────────────────────
IDX_RBC_MASS   = 0   # RBC wet mass [g]
IDX_PLASMA_VOL = 1   # Plasma volume [L]
IDX_EPO        = 2   # Erythropoietin [mIU/mL]
IDX_FERRITIN   = 3   # Ferritin [ug/L]
IDX_HEM_TOX    = 4   # Hemolysis toxin [au]

STATE_DIM = 5
OBS_DIM   = 3   # [Hgb_g_dL, Hematocrit_pct, Ferritin_lab]
CTRL_DIM  = 4

# ── Hub control indices ────────────────────────────────────────────────────────
UIDX_HYPOXIA = 0
UIDX_LOAD    = 1
UIDX_IL6     = 2
UIDX_IRON    = 3

_RBC_ML_PER_G  = 0.88     # 1g RBC ~ 0.88 mL (density ~1.09 g/mL)
_MCHC_FRAC_100 = 29.04    # (MCHC 33% wet mass) * 100; gives Hgb_g_dL units


# ── Parameter container ────────────────────────────────────────────────────────

class HematologicalParams(NamedTuple):
    """
    5-state hourly hematological ODE parameters.
    NLME-identifiable (D=2):
        k_epo_rbc  marrow EPO sensitivity  Prior: LogN(log 0.05, 0.35^2)
        k_pv_exp   plasma elasticity       Prior: LogN(log 0.01, 0.40^2)
    """
    # EPO axis
    k_hypoxia:        float = 0.10    # mIU/mL/h per % hypoxia (HIF-1a drive)
    k_epo_clear:      float = 0.1155  # h^-1  t1/2=6h  (ln2/6)
    # Hepcidin-iron axis (IL-6 proxy)
    k_hepcidin:       float = 0.10    # 1/(pg/mL IL6)
    # Erythropoiesis
    k_epo_rbc:        float = 0.05    # NLME-1  g/h per (mIU/mL x Ferritin sat)
    Km_Fe:            float = 30.0    # ug/L  Michaelis constant
    k_iron_use:       float = 25.0    # ug/L used per (g/h erythropoiesis)
    # Hemolysis
    k_base_hemolysis: float = 3.0e-4  # h^-1  RBC senescence (t1/2 ~2400h ~ 100d)
    k_footstrike:     float = 2.0e-3  # h^-1/au  footstrike impact hemolysis
    # Plasma volume
    k_pv_exp:         float = 0.01    # NLME-2  L/h per au Load (hypervolemia)
    k_pv_decay:       float = 0.05    # h^-1  PV return to baseline (t1/2~14h)
    PV_base:          float = 2.8     # L  resting plasma volume (70kg male)
    # Hemolysis toxin
    k_tox:            float = 0.5     # au/(g/h) toxin production
    k_clear_tox:      float = 0.2     # h^-1  toxin clearance (t1/2~3.5h)


DEFAULT_HEM_PARAMS = HematologicalParams()


def build_hem_params(**overrides: float) -> HematologicalParams:
    return DEFAULT_HEM_PARAMS._replace(**overrides)


# ── Default state and covariance ───────────────────────────────────────────────

X0_HEM_DEFAULT: jax.Array = jnp.array(
    [2500.0, 2.8, 15.0, 80.0, 0.0], dtype=jnp.float32
)

P0_HEM_DEFAULT: jax.Array = jnp.diag(jnp.array(
    [10000.0, 0.04, 25.0, 400.0, 0.01], dtype=jnp.float32
))


def initial_state() -> jax.Array:
    """Population reference state: male endurance athlete at rest."""
    return X0_HEM_DEFAULT


# ── Vector field ───────────────────────────────────────────────────────────────

def hematological_v3_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    5-state hematological ODE (hourly).
    args = (HematologicalParams, u: shape(CTRL_DIM,))
    Pure JAX -- no Python conditionals on traced values. JIT + vmap safe.
    """
    params, u = args
    p = params

    # Physical positivity clamps
    RBC_Mass   = jnp.maximum(x[IDX_RBC_MASS],   0.0)
    Plasma_Vol = jnp.maximum(x[IDX_PLASMA_VOL], 0.0)
    EPO        = jnp.maximum(x[IDX_EPO],        0.0)
    Ferritin   = jnp.maximum(x[IDX_FERRITIN],   0.0)
    Hem_Tox    = jnp.maximum(x[IDX_HEM_TOX],    0.0)

    # Hub NaN guards (defaults per spec: 0, 0, 1.0, 0.0)
    hub_Hypoxia = jnp.where(jnp.isnan(u[UIDX_HYPOXIA]), 0.0,
                            jnp.maximum(u[UIDX_HYPOXIA], 0.0))
    hub_Load    = jnp.where(jnp.isnan(u[UIDX_LOAD]),    0.0,
                            jnp.clip(u[UIDX_LOAD], 0.0, 1.0))
    hub_IL6     = jnp.where(jnp.isnan(u[UIDX_IL6]),     1.0,
                            jnp.maximum(u[UIDX_IL6], 0.0))
    hub_Iron    = jnp.where(jnp.isnan(u[UIDX_IRON]),    0.0,
                            jnp.maximum(u[UIDX_IRON], 0.0))

    # a) EPO kinetics
    dEPO = p.k_hypoxia * hub_Hypoxia - p.k_epo_clear * EPO

    # b) Iron absorption (hepcidin proxy via IL-6)
    fe_abs = hub_Iron / (1.0 + p.k_hepcidin * hub_IL6)

    # d) Erythropoiesis (EPO x Michaelis-Menten iron limitation)
    Erythropoiesis = p.k_epo_rbc * EPO * (Ferritin / (p.Km_Fe + Ferritin))

    # c) Ferritin dynamics
    dFerritin = fe_abs - p.k_iron_use * Erythropoiesis

    # e) Total hemolysis (senescence + footstrike)
    Hemolysis = (p.k_base_hemolysis + p.k_footstrike * hub_Load) * RBC_Mass

    # f) RBC mass
    dRBC = Erythropoiesis - Hemolysis

    # g) Plasma volume (hypervolemia with load; decay to PV_base)
    dPlasma = p.k_pv_exp * hub_Load - p.k_pv_decay * (Plasma_Vol - p.PV_base)

    # h) Hemolysis toxin
    dHem_Tox = p.k_tox * Hemolysis - p.k_clear_tox * Hem_Tox

    return jnp.array([dRBC, dPlasma, dEPO, dFerritin, dHem_Tox])


# ── Algebraic hub outputs ──────────────────────────────────────────────────────

def compute_hematocrit(x: jax.Array) -> jax.Array:
    """Hub_Hematocrit_pct [%]  (1g RBC ~ 0.88 mL)."""
    rbc_vol_mL  = jnp.maximum(x[IDX_RBC_MASS],   0.0) * _RBC_ML_PER_G
    pv_mL       = jnp.maximum(x[IDX_PLASMA_VOL], 0.0) * 1000.0
    return rbc_vol_mL / (rbc_vol_mL + pv_mL + 1e-6) * 100.0


def compute_o2_capacity(x: jax.Array) -> jax.Array:
    """Hub_O2_Capacity [mL O2/dL blood] = Hgb_g_dL * 1.34."""
    RBC_Mass = jnp.maximum(x[IDX_RBC_MASS],   0.0)
    PV_L     = jnp.maximum(x[IDX_PLASMA_VOL], 0.0)
    total_mL = RBC_Mass * _RBC_ML_PER_G + PV_L * 1000.0
    hgb_g_dL = (RBC_Mass * _MCHC_FRAC_100) / (total_mL + 1e-6)
    return hgb_g_dL * 1.34


# ── Integration helper ─────────────────────────────────────────────────────────

def integrate_1h(
    x0:        jax.Array,
    params:    HematologicalParams = DEFAULT_HEM_PARAMS,
    u:         jax.Array | None   = None,
    t0:        float               = 0.0,
    max_steps: int                 = 256,
) -> jax.Array:
    """
    Advance the 5-state hematological ODE by exactly 1 hour (Tsit5).
    Returns shape (STATE_DIM,).
    vmap-compatible: vary x0 over sigma points; params and u are broadcast.
    """
    if u is None:
        u = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    sol = diffrax.diffeqsolve(
        terms     = diffrax.ODETerm(hematological_v3_ode),
        solver    = diffrax.Tsit5(),
        t0        = jnp.float32(t0),
        t1        = jnp.float32(t0 + 1.0),
        dt0       = jnp.float32(0.1),
        y0        = x0,
        args      = (params, u),
        saveat    = diffrax.SaveAt(t1=True),
        max_steps = max_steps,
    )
    return sol.ys[0]
