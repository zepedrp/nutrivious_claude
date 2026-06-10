"""
app/slices/thermo_renal/ode.py  —  Thermo-Renal ODE V2

STATE VECTOR  x ∈ ℝ⁵   (time unit: MINUTES)
  x[0]  Core_Temp_C      Core temperature          [°C]
  x[1]  Skin_Temp_C      Mean skin temperature     [°C]
  x[2]  Plasma_Volume_L  Plasma volume             [L]
  x[3]  Inters_Volume_L  Interstitial volume       [L]
  x[4]  Plasma_Sodium_mmol  Total Na in plasma     [mmol]

CONTROL INPUTS  u ∈ ℝ⁴  (hub variables)
  u[0]  hub_power_watts          Metabolic power          [W]
  u[1]  hub_fluid_intake_L_min   Fluid intake rate        [L/min]
  u[2]  hub_sodium_intake_mmol_min  Sodium intake rate    [mmol/min]
  u[3]  hub_basal_temp_offset    P4-driven setpoint shift [°C] (SlowAxis hub)

PHYSICS
  Thermoreg : 80% of hub_power_watts → heat; linear drive on Core_Temp_C.
              Core cools by conduction to Skin and by sweat evaporation.
  Sweat     : rate ∝ exp(Core_Temp_C − 37) − 1 (zero at 37°C).
              Drains Plasma_Volume_L and Plasma_Sodium_mmol.
  Starling  : fluid shifts from interstitium → plasma when PV below reference
              (oncotic-pressure proxy: proportional to normalised PV deficit).
  Kidney    : basal urine suppressed quadratically when PV is low (AVP/Aldo proxy).
  NaN guards: jnp.where(jnp.isnan) on all hub inputs and state components.

References
  Fiala D. et al. (1999) J Appl Physiol 87(5):1957–1972   [2-node thermo]
  Karaaslan F., Hester R.L. (2005) Am J Physiol Regul     [renal fluid model]
  Hew-Butler T. et al. (2015) Clin J Sport Med 35         [EAH consensus]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── State indices ─────────────────────────────────────────────────────────────
IDX_CORE_TEMP  = 0
IDX_SKIN_TEMP  = 1
IDX_PLASMA_VOL = 2
IDX_INTERS_VOL = 3
IDX_PLASMA_NA  = 4

STATE_DIM: int = 5
OBS_DIM:   int = 2   # [Core_Temp_obs °C, Body_Mass_Drop_kg]

# ── Control indices ───────────────────────────────────────────────────────────
UIDX_POWER_W     = 0
UIDX_FLUID_L     = 1
UIDX_NA_MMOL     = 2
UIDX_BASAL_TEMP  = 3   # hub_basal_temp_offset [°C] from SlowAxisOrchestrator

CTRL_DIM:  int = 4
TIME_UNIT: str = "minutes"


# ── Parameter container ───────────────────────────────────────────────────────

class ThermoRenalParams(NamedTuple):
    """
    Population-prior parameters for the Thermo-Renal V2 ODE.

    Thermal capacities use Wh/°C (Fiala 1999).
    With time in minutes: dT/dt [°C/min] = Q[W] / (C[Wh/°C] × 60).
    Sweat uses exponential gain above 37°C.
    NLME personalises sweat_sensitivity and sweat_na_conc (D=2).
    """
    # ── Thermal ───────────────────────────────────────────────────────────────
    C_core_Wh: float = 47.0       # core thermal mass  [Wh/°C]  (Fiala 1999)
    C_skin_Wh: float = 4.5        # skin thermal mass  [Wh/°C]
    k_core_skin: float = 40.0     # core→skin conductance at rest [W/°C]
    h_skin_env:  float = 15.0     # skin→environment  [W/°C]
    T_ambient:   float = 22.0     # ambient temperature [°C]
    efficiency:  float = 0.20     # mechanical efficiency (rest → heat)
    L_evap_Wh_L: float = 680.0   # latent heat of sweat evaporation [Wh/L]

    # ── Sweat  (NLME personalised D=2) ────────────────────────────────────────
    sweat_sensitivity: float = 0.004   # [L/min] base rate at T_core=37+1°C
    sweat_na_conc:     float = 50.0    # sweat Na concentration [mmol/L]

    # ── Fluid balance ─────────────────────────────────────────────────────────
    PV_ref:     float = 4.2    # reference plasma volume [L]
    IV_ref:     float = 12.0   # reference interstitial volume [L]
    K_starling: float = 0.005  # Starling rate [L/min per unit normalised PV deficit]
    U_basal:    float = 0.001  # basal urine output [L/min] (~1.4 L/day)
    U_Na_frac:  float = 0.15   # urinary Na as fraction of filtered load (base)

    # ── Plasma sodium ─────────────────────────────────────────────────────────
    Na_ref_conc: float = 140.0   # reference [Na+]_plasma [mmol/L]


DEFAULT_TR_PARAMS: ThermoRenalParams = ThermoRenalParams()

_p = DEFAULT_TR_PARAMS

# ── Default initial state (resting, euhydrated, thermoneutral) ────────────────
X0_TR_DEFAULT: jax.Array = jnp.array([
    37.0,                              # Core_Temp_C
    34.0,                              # Skin_Temp_C
    _p.PV_ref,                         # Plasma_Volume_L
    _p.IV_ref,                         # Inters_Volume_L
    _p.PV_ref * _p.Na_ref_conc,        # Plasma_Sodium_mmol = 4.2×140 = 588
], dtype=jnp.float32)

# ── Default initial covariance ────────────────────────────────────────────────
P0_TR_DEFAULT: jax.Array = jnp.diag(jnp.array([
    0.04,    # Core_Temp_C  [°C²]
    0.25,    # Skin_Temp_C  [°C²]
    0.25,    # Plasma_Volume_L  [L²]
    1.00,    # Inters_Volume_L  [L²]
    100.0,   # Plasma_Sodium_mmol  [mmol²]
], dtype=jnp.float32))


# ── Pure-JAX ODE (JIT + vmap safe) ───────────────────────────────────────────

def thermo_renal_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    Thermo-Renal V2 ODE — time in MINUTES, pure JAX.

    Parameters
    ----------
    t    : scalar time [min]
    x    : (STATE_DIM=5,)
    args : (ThermoRenalParams, u) where u : (CTRL_DIM=3,)

    Returns
    -------
    dx/dt : (5,) in [°C/min, °C/min, L/min, L/min, mmol/min]
    """
    params, u = args

    # ── NaN guards on hub inputs ──────────────────────────────────────────────
    power_W  = jnp.where(jnp.isnan(u[UIDX_POWER_W]),    jnp.float32(0.0), u[UIDX_POWER_W])
    fluid    = jnp.where(jnp.isnan(u[UIDX_FLUID_L]),    jnp.float32(0.0), u[UIDX_FLUID_L])
    na_in    = jnp.where(jnp.isnan(u[UIDX_NA_MMOL]),    jnp.float32(0.0), u[UIDX_NA_MMOL])
    # Progesterone-driven setpoint shift from SlowAxis hub (Baker & Jeukendrup 2001).
    # Follicular/male default = 0.0 °C; mid-luteal peak P4 ≈ +0.35 °C.
    bto      = jnp.where(jnp.isnan(u[UIDX_BASAL_TEMP]), jnp.float32(0.0), u[UIDX_BASAL_TEMP])
    bto      = jnp.clip(bto, jnp.float32(-0.1), jnp.float32(0.5))

    # ── NaN guards on states ──────────────────────────────────────────────────
    T_core = jnp.where(jnp.isnan(x[IDX_CORE_TEMP]),  jnp.float32(37.0),             x[IDX_CORE_TEMP])
    T_skin = jnp.where(jnp.isnan(x[IDX_SKIN_TEMP]),  jnp.float32(34.0),             x[IDX_SKIN_TEMP])
    PV     = jnp.where(jnp.isnan(x[IDX_PLASMA_VOL]), jnp.float32(params.PV_ref),    x[IDX_PLASMA_VOL])
    IV     = jnp.where(jnp.isnan(x[IDX_INTERS_VOL]), jnp.float32(params.IV_ref),    x[IDX_INTERS_VOL])
    Na     = jnp.where(jnp.isnan(x[IDX_PLASMA_NA]),  jnp.float32(params.PV_ref * params.Na_ref_conc), x[IDX_PLASMA_NA])

    PV = jnp.maximum(PV, jnp.float32(0.1))
    IV = jnp.maximum(IV, jnp.float32(0.1))
    Na = jnp.maximum(Na, jnp.float32(0.0))

    # ── BLOCK A: Thermoregulation ─────────────────────────────────────────────
    # 80% of power is heat
    heat_W = power_W * jnp.float32(1.0 - params.efficiency)

    # Core-to-skin conduction [W]
    Q_cs  = jnp.float32(params.k_core_skin) * (T_core - T_skin)

    # Skin-to-environment convection+radiation [W]
    Q_env = jnp.float32(params.h_skin_env)  * (T_skin - jnp.float32(params.T_ambient))

    # Sweat rate: exponential above setpoint [L/min].
    # Setpoint shifts by hub_basal_temp_offset (progesterone; Baker & Jeukendrup 2001).
    T_sweat_setpoint = jnp.float32(37.0) + bto
    excess = jnp.maximum(T_core - T_sweat_setpoint, jnp.float32(0.0))
    sweat  = jnp.float32(params.sweat_sensitivity) * (jnp.exp(excess) - jnp.float32(1.0))

    # Evaporative cooling of skin [W]  =  sweat[L/min] × L_evap[Wh/L] × 60[min/h]
    E_evap = sweat * jnp.float32(params.L_evap_Wh_L) * jnp.float32(60.0)

    # dT/dt [°C/min] = Q[W] / (C[Wh/°C] × 60[min/h])
    dT_core = (heat_W - Q_cs)          / (jnp.float32(params.C_core_Wh) * jnp.float32(60.0))
    dT_skin = (Q_cs - Q_env - E_evap)  / (jnp.float32(params.C_skin_Wh) * jnp.float32(60.0))

    # ── BLOCK B: Fluid balance [L/min] ───────────────────────────────────────
    # Kidney: basal urine suppressed when PV is low (AVP/Aldo proxy)
    pv_frac = jnp.minimum(PV / jnp.float32(params.PV_ref), jnp.float32(1.0))
    urine   = jnp.float32(params.U_basal) * pv_frac * pv_frac      # [L/min]

    # Starling: net fluid flow from interstitium → plasma when PV is depleted
    pv_deficit_norm = (jnp.float32(params.PV_ref) - PV) / jnp.float32(params.PV_ref)
    J_star = jnp.float32(params.K_starling) * pv_deficit_norm       # [L/min]

    dPV = fluid - sweat - urine + J_star
    dIV = -J_star

    # ── BLOCK C: Plasma sodium [mmol/min] ─────────────────────────────────────
    Na_conc = Na / PV                                                # [mmol/L]

    # Sweat removes sodium proportional to sweat Na concentration
    dNa_sweat = -sweat * jnp.float32(params.sweat_na_conc)

    # Urine Na: Aldo reduces excretion when PV is low
    dNa_urine = -urine * Na_conc * jnp.float32(params.U_Na_frac) * pv_frac

    dNa = na_in + dNa_sweat + dNa_urine

    return jnp.stack([dT_core, dT_skin, dPV, dIV, dNa])
