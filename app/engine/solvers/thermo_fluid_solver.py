"""
app/engine/solvers/thermo_fluid_solver.py

Módulo 10 × Módulo 12 — Motor Termo-Fluido (MHDS Subsistemas 10 e 12) — v3.0

════════════════════════════════════════════════════════════════════════
ARCHITECTURAL CORRECTIONS v2→v3
════════════════════════════════════════════════════════════════════════

1. REAL HUB COUPLING (Mod 1 + Mod 3):
   - Hub_Energy_Expenditure (Mod 1) → ee_interp [kcal/h] mandatory
   - Hub_Minute_Ventilation_L (Mod 3) → ve_interp [L/min] mandatory
   - Algebraic P_norm shortcuts removed from ODE body.
   - Q_met_total = ee_interp.evaluate(t)  [total EE from Mod 1]
   - W_mech      = P_t × 0.860421        [mechanical output, kcal/h]
   - Q_heat      = Q_met_total − W_mech  [1st Law; heat to dissipate]
   - Ve_L_min    = ve_interp.evaluate(t) [from Mod 3 Grodins model]

2. ACTN3 REDUNDANCY REMOVED:
   - actn3_r577x_prior and eta_gross deleted from this module.
   - Genetic efficiency already embedded in ee_interp (Mod 1 owns η).
   - Retains only nos3_prior (vasodilation) and hspa1b_prior (sweating).

3. SkBF BIOLOGICAL CAP:
   - jnp.clip(SkBF, 0.0, 420.0) — 420 L/h ≈ 7 L/min lethal ceiling.
   - Above this threshold the heart fails (Rowell 1974 absolute limit).

════════════════════════════════════════════════════════════════════════
STEP 1 — EQUAÇÕES DO CALOR RESPIRATÓRIO E SÓDIO NO SUOR
════════════════════════════════════════════════════════════════════════

CALOR RESPIRATÓRIO (Hub_Minute_Ventilation_L do Módulo 3)
──────────────────────────────────────────────────────────

Pressão de Vapor de Saturação [Magnus/Tetens 1930]:
    P_sat(T) [kPa] = 0.6112 × exp(17.67 × T / (T + 243.5))

Ração de humidade do ar inspirado:
    P_vap_amb  = RH × P_sat(T_amb)
    w_insp [kg/kg] = 0.622 × P_vap_amb / (101.325 − P_vap_amb)

Perda evaporativa respiratória (ar expirado a ~35 °C, 100 % RH):
    w_exp = 0.0313 kg_H₂O/kg_ar   (Fanger 1970)
    Δw    = max(0, w_exp − w_insp)
    Eres [kcal/h] = K_Eres × Ve_L_h × Δw
    K_Eres = ρ_ar × λ = 1.292 g/L × 0.5808 kcal/g = 0.7504 kcal/(L·unit_Δw)

Perda convectiva respiratória:
    Cres [kcal/h] = K_Cres × Ve_L_h × (T_core − T_amb)
    K_Cres = ρ_ar × c_p / 4184 / 1000 = 3.107×10⁻⁴ kcal/(L·K)

Ve = ve_interp.evaluate(t) [L/min] — real array from Hub_Minute_Ventilation_L (Mod 3)

EMAX — LIMITE EVAPORATIVO CUTÂNEO (Lewis + Antoine)
────────────────────────────────────────────────────

    P_sat_skin [kPa] = 0.6112 × exp(17.67 × T_skin / (T_skin + 243.5))
    ΔP = max(0, P_sat_skin − P_vap_amb)
    h_conv_eff [kcal/(h·m²·°C)] = h_conv_base + k_wind × √v_wind
    Emax [kcal/h] = 16.7 × h_conv_eff × A_body × ΔP
        (factor 16.7 da Lei de Lewis; conversão kcal/h cancela exactamente)

    Esw_actual = min(SR × h_evap, Emax)   ← humidade BLOQUEIA arrefecimento

CONCENTRAÇÃO DE SÓDIO NO SUOR — DEPENDENTE DO CAUDAL
──────────────────────────────────────────────────────

Cinética das glândulas sudoríparas: caudal alto → tempo insuficiente para
reabsorção ductal de Na⁺ → [Na⁺] aumenta (Adams & Best 1994; Montain 2007):

    [Na]_sw [mmol/L] = Na_min + (Na_max − Na_min) × (1 − exp(−SR / SR_Na_sat))
    Na_min = 20 mmol/L,  Na_max = 70 mmol/L,  SR_Na_sat = 1.0 L/h

════════════════════════════════════════════════════════════════════════
SISTEMA DE 4 ODEs — y = [T_core, T_skin, V_p, Na_plasma_mmol]
════════════════════════════════════════════════════════════════════════

  y[0]  T_core [°C]        τ_core  ≈ 30–60 min  (massa térmica grande)
  y[1]  T_skin [°C]        τ_skin  ≈  5–10 min  (inércia cutânea pequena)
  y[2]  V_p    [L]         τ_plasma ≈ 20–40 min
  y[3]  Na_plasma [mmol]   τ_Na   ≈ 60–120 min

  Stiffness τ_core/τ_skin ≈ 6–12 → Kvaerno5 (DIRK 5ª ordem) obrigatório.

dT_core/dt = (Q_heat − Q_blood − Q_resp) / C_core
               Q_heat = ee_interp(t) − W_mech = 1st Law net heat

dT_skin/dt = (Q_blood − Esw_actual − Q_rad_conv) / C_skin

dV_p/dt    = fluid_abs(t) − f_plasma_eff × SR − resp_H₂O_loss
  f_plasma_eff = f_plasma_base × (V_p/V_p0)^k_osm   [osmotic defence, Costill 1977]

dNa_plasma/dt = na_abs(t) − [Na]_sw × SR

════════════════════════════════════════════════════════════════════════
HUB VARIABLES — Bond Graph Interface
════════════════════════════════════════════════════════════════════════

HUB INBOUND (REAL INTERPOLATORS — no algebraic shortcuts):
    Hub_Energy_Expenditure_Kcal_H (Mod 1) → ee_interp  → Q_heat = EE − W_mech
    Hub_Minute_Ventilation_L      (Mod 3) → ve_interp  → Eres, Cres
    Hub_Fluid_Absorption_Rate     (Mod 8) → fluid_abs_interp [L/h]
    Hub_Sodium_Absorption_Rate    (Mod 8) → na_abs_interp   [mmol/h]
    Hub_Basal_Temperature_Morning (Mod 6) → T_core_set shift [°C]

HUB OUTBOUND (6 arrays — «placa-mãe»):
    A) Hub_Core_Temp               [°C]      → Mod 4, Mod 5, NMPC
    B) Hub_Skin_Temp               [°C]      → Mod 12, Mod 11
    C) Hub_Sweat_Rate_L_h          [L/h]     → NMPC hidratação
    D) Hub_Plasma_Volume_Drop_Pct  [%]       → Mod 3 cardiovascular drift
    E) Hub_Sodium_Concentration    [mmol/L]  → Mod 12 osmolaridade
    F) Hub_Skin_Blood_Flow         [L/min]   → shunts GI, Mod 3

════════════════════════════════════════════════════════════════════════
MODIFICADORES GENÉTICOS (Fase 3)
════════════════════════════════════════════════════════════════════════

    nos3_prior    → nos3_scale   (NOS3 T786C → mais NO → SkBF↑; capped 420 L/h)
    hspa1b_prior  → hspa1b_scale (Hsp70 → aclimatação → SR onset↑)
    hub_basal_temp_offset → T_core_set (Mod 6 thyroid shift)

    NOTE: actn3_r577x_prior REMOVED — efficiency lives in Mod 1 (Hub_EE).
          Duplicating it here violates 1st Law conservation.

════════════════════════════════════════════════════════════════════════
REFERÊNCIAS
════════════════════════════════════════════════════════════════════════
Gagge (1971) J Appl Physiol 31:309–316      [2-node bioheat model]
Pennes (1948) J Appl Physiol 1:93–122       [bioheat equation]
Nadel (1971) J Appl Physiol 31:80–87        [sweat rate proportional control]
Tetens (1930) Meteor Z 66:329              [Magnus saturation vapour pressure]
Fanger (1970) Thermal Comfort              [Lewis relation, Eres, Cres, w_exp]
Adams & Best (1994) J Appl Physiol 77:1827 [sweat [Na+] kinetics]
Montain et al. (2007) J Appl Physiol 103:990 [sweat electrolytes vs flow]
Rowell (1974) Circ Res 34 Suppl:I-105      [cutaneous blood flow; 7 L/min cap]
Nishi & Gagge (1970) J Appl Physiol 29:830 [convective coefficient]
Hunt & Stubbs (1975) J Physiol 245:209     [gastric emptying fallback]
Maughan & Leiper (1985) Eur J Appl Physiol 54:439 [plasma fraction sweat]
Costill (1977) Int J Sports Med            [osmotic plasma defence]
Armstrong (2007) J Athl Train 42:333        [39.5 °C heat stroke threshold]
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

# ── Module-level physical constants ───────────────────────────────────────────
_W_TO_KCAL_H: float = 0.860421    # 1 W = 3600 J/h / 4184 J/kcal
_T_ALARM: float = 39.5            # °C — exertional heat-stroke threshold (Armstrong 2007)
_K_ERES: float = 0.7504           # kcal/(L·Δω): ρ_air 1.292 g/L × λ 0.5808 kcal/g
_K_CRES: float = 3.107e-4         # kcal/(L·K):  ρ_air × c_p / 4184 / 1000
_W_EXP: float = 0.0313            # kg_H₂O/kg_dry_air expired at ~35 °C, 100 % RH (Fanger 1970)
_P_ATM: float = 101.325           # kPa — standard atmospheric pressure
_NA_REF_MMOL_L: float = 140.0    # mmol/L — normal plasma [Na⁺]
_SKBF_MAX_L_H: float = 420.0     # L/h — absolute SkBF ceiling (Rowell 1974; ≈7 L/min)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter container (JAX-native pytree via NamedTuple)
# ─────────────────────────────────────────────────────────────────────────────

class ThermoFluidParams(NamedTuple):
    """
    Physical parameter vector for the 4-state Thermo-Fluid ODE — v3.0.

    v3.0 changes vs v2.0:
        - Removed: eta_gross (ACTN3 — lives in Mod 1 Hub_EE)
        - Removed: Q_met_base, Ve_basal_L_min, k_Ve_power, P_ref_W
          (these are now inside the ee_interp / ve_interp fallback builders,
           not inside the ODE parameter vector)
        - Added:   SkBF biological cap enforced via _SKBF_MAX_L_H constant

    Genetic priors injected via _build_params:
        nos3_prior    → nos3_scale   (eNOS vasodilation amplitude)
        hspa1b_prior  → hspa1b_scale (Hsp70 sweat acclimatisation)
        Hub_Basal_Temperature_Morning (Mod 6) → T_core_set shift
    """
    # ── I. Thermal masses ────────────────────────────────────────────────────
    C_core: float        # Core thermal capacity [kcal/°C]
    C_skin: float        # Skin thermal capacity [kcal/°C]

    # ── II. Cutaneous blood flow (SkBF) ──────────────────────────────────────
    k_blood: float       # Blood-to-skin convective coeff [kcal/(L·°C)]
    BF_skin_base: float  # Basal SkBF [L/h]
    k_BF: float          # SkBF gain above setpoint [L/(h·°C)]
    nos3_scale: float    # NOS3 eNOS vasodilation multiplier [adim]
    T_core_set: float    # Thermoregulatory setpoint [°C] — thyroid-shifted

    # ── III. Sweating ─────────────────────────────────────────────────────────
    k_sweat: float       # Sweat rate gain [L/(h·°C)]
    hspa1b_scale: float  # Hsp70 acclimatisation multiplier [adim]
    h_evap: float        # Latent heat of vaporisation [kcal/L] at 37 °C

    # ── IV. Surface heat exchange ─────────────────────────────────────────────
    A_body: float        # Body surface area [m²]
    h_conv_base: float   # Basal convective coefficient [kcal/(h·m²·°C)]
    h_rad: float         # Radiative coefficient [kcal/(h·m²·°C)]
    k_wind: float        # Wind enhancement: Δh = k_wind × √v_wind [kcal/(h·m²·°C·√(m/s))]

    # ── V. Plasma volume dynamics ─────────────────────────────────────────────
    f_plasma_sweat: float    # Base fraction of sweat from plasma [adim]
    k_osmotic_shift: float   # Osmotic defence exponent [adim]
    V_p0: float              # Reference plasma volume [L]
    tau_gastric: float       # Fallback gastric emptying τ [h]

    # ── VI. Sodium sweat kinetics ─────────────────────────────────────────────
    Na_sweat_min: float  # Min sweat [Na⁺] at low SR [mmol/L]
    Na_sweat_max: float  # Max sweat [Na⁺] at high SR [mmol/L]
    SR_Na_sat: float     # SR saturation constant for [Na⁺] kinetics [L/h]

    # ── VII. Melatonin peripheral vasodilation (Mod 4 → Mod 10) ──────────────
    k_Mel_BF: float      # melatonin → SkBF additive gain [L/(h·unit)] (Cagnacci 1992)


# ─────────────────────────────────────────────────────────────────────────────
# Antoine / Magnus helper — smooth, differentiable, JIT-safe
# ─────────────────────────────────────────────────────────────────────────────

def _p_sat_kpa(T_celsius: jax.Array) -> jax.Array:
    """Saturation vapour pressure [kPa] — Magnus formula (Tetens 1930)."""
    return 0.6112 * jnp.exp(17.67 * T_celsius / (T_celsius + 243.5))


# ─────────────────────────────────────────────────────────────────────────────
# Pure ODE — JIT-compilable
# ─────────────────────────────────────────────────────────────────────────────

def thermo_fluid_ode(
    t: jax.Array,
    y: jax.Array,
    args: tuple,
) -> jax.Array:
    """
    4-state JIT-compilable ODE for MHDS Modules 10 & 12 — v3.0.

    y = [T_core °C, T_skin °C, V_p L, Na_plasma mmol]   time t in hours.

    args = (params, power_w, t_sess_start_h, sess_dur_h,
            fluid_abs_interp, na_abs_interp,
            ee_interp, ve_interp,
            T_amb, RH_frac, v_wind_m_s)

    Hub inbound (real interpolators — no algebraic shortcuts):
        ee_interp        (Mod 1) → Q_met_total = ee_interp(t) [kcal/h]
        ve_interp        (Mod 3) → Ve_L_min    = ve_interp(t) [L/min]
        fluid_abs_interp (Mod 8) → dV_p influx [L/h]
        na_abs_interp    (Mod 8) → dNa influx  [mmol/h]
        T_core_set (in params)   (Mod 6) → baked in from Hub_Basal_Temp
    """
    (params, power_w, t_sess_start_h, sess_dur_h,
     fluid_abs_interp, na_abs_interp,
     ee_interp, ve_interp,
     T_amb, RH_frac, v_wind_m_s,
     mel_interp) = args

    T_core    = y[0]
    T_skin    = y[1]
    V_p       = y[2]
    Na_plasma = y[3]

    # ── Smooth session gate (differentiable Heaviside) ────────────────────
    def _H(t_on: jax.Array) -> jax.Array:
        return 0.5 * (1.0 + jnp.tanh(20.0 * (t - t_on)))

    gate = _H(t_sess_start_h) - _H(t_sess_start_h + sess_dur_h)
    P_t  = power_w * gate   # mechanical power output at time t [W]

    # ══════════════════════════════════════════════════════════════════════
    # I. METABOLIC HEAT — 1st Law: Q_heat = EE − W_mech
    # ee_interp carries total energy expenditure from Mod 1 (η already in).
    # W_mech is the mechanical power that leaves as useful work.
    # ══════════════════════════════════════════════════════════════════════
    Q_met_total = jnp.maximum(0.0, ee_interp.evaluate(t))   # [kcal/h] from Mod 1
    W_mech      = P_t * _W_TO_KCAL_H                        # [kcal/h] mechanical output
    Q_heat      = Q_met_total - W_mech                       # [kcal/h] net heat to dissipate

    # ══════════════════════════════════════════════════════════════════════
    # II. RESPIRATORY HEAT — Eres + Cres removed from Core (Fanger 1970)
    # Ve = ve_interp.evaluate(t) — real array from Hub_Minute_Ventilation_L
    # ══════════════════════════════════════════════════════════════════════
    Ve_L_min = jnp.maximum(0.0, ve_interp.evaluate(t))  # [L/min] from Mod 3
    Ve_L_h   = Ve_L_min * 60.0                           # [L/h]

    P_sat_amb = _p_sat_kpa(T_amb)
    P_vap_amb = RH_frac * P_sat_amb                               # [kPa]
    w_insp    = 0.622 * P_vap_amb / (_P_ATM - P_vap_amb)         # [kg_H₂O/kg_air]
    delta_w   = jnp.maximum(0.0, _W_EXP - w_insp)               # [kg/kg]

    Eres = _K_ERES * Ve_L_h * delta_w                 # evaporative respiratory [kcal/h]
    Cres = _K_CRES * Ve_L_h * (T_core - T_amb)        # convective respiratory  [kcal/h]
    Q_resp = Eres + Cres                               # total respiratory loss from Core

    # ══════════════════════════════════════════════════════════════════════
    # III. SKIN BLOOD FLOW (SkBF) — Pennes 1948 + biological cap 420 L/h
    # nos3_prior amplifies vasodilation gain; clipped at cardiac failure limit.
    # ══════════════════════════════════════════════════════════════════════
    dT_err = T_core - params.T_core_set
    Mel    = jnp.clip(mel_interp.evaluate(t), 0.0, None)
    # Melatonin (Cagnacci 1992): nocturnal peripheral vasodilation → SkBF↑ → T_core↓
    SkBF = jnp.clip(
        params.BF_skin_base
        + params.k_BF * params.nos3_scale * jnp.maximum(0.0, dT_err)
        + params.k_Mel_BF * Mel,
        0.0,
        _SKBF_MAX_L_H,
    )  # [L/h] — capped at 420 L/h (Rowell 1974)

    Q_blood = params.k_blood * SkBF * (T_core - T_skin)  # [kcal/h]

    # ══════════════════════════════════════════════════════════════════════
    # IV. SWEATING — Nadel 1971 proportional control
    # hspa1b_prior → Hsp70 → earlier onset + higher peak SR
    # ══════════════════════════════════════════════════════════════════════
    SR = params.k_sweat * params.hspa1b_scale * jnp.maximum(0.0, dT_err)  # [L/h]

    # ══════════════════════════════════════════════════════════════════════
    # V. EVAPORATIVE CEILING — Antoine equation + Lewis relation
    # High RH: P_sat_skin ≈ P_vap_amb → ΔP→0 → Emax→0 → sweat drips unproductively
    # ══════════════════════════════════════════════════════════════════════
    h_conv_eff = (
        params.h_conv_base
        + params.k_wind * jnp.sqrt(jnp.maximum(0.0, v_wind_m_s))
    )  # [kcal/(h·m²·°C)]
    P_sat_skin = _p_sat_kpa(T_skin)
    Emax = (
        16.7 * h_conv_eff * params.A_body
        * jnp.maximum(0.0, P_sat_skin - P_vap_amb)
    )  # [kcal/h] — Lewis relation: unit conversion cancels exactly

    Esw_potential = params.h_evap * SR                  # [kcal/h] if all evaporates
    Esw_actual    = jnp.minimum(Esw_potential, Emax)   # humidity ceiling

    # Dry heat exchange: convection + radiation pele → ambiente
    Q_rad_conv = (params.h_rad + h_conv_eff) * params.A_body * (T_skin - T_amb)

    # ══════════════════════════════════════════════════════════════════════
    # VI. ODEs
    # ══════════════════════════════════════════════════════════════════════

    # Core: gains net metabolic heat (1st Law), loses via blood + respiration
    dT_core_dt = (Q_heat - Q_blood - Q_resp) / params.C_core

    # Skin: gains from blood, loses via evaporation + dry exchange
    dT_skin_dt = (Q_blood - Esw_actual - Q_rad_conv) / params.C_skin

    # Plasma volume: Mod 8 influx − sweat plasma loss − respiratory water
    v_ratio      = jnp.maximum(V_p / params.V_p0, 0.1)
    f_plasma_eff = params.f_plasma_sweat * (v_ratio ** params.k_osmotic_shift)
    plasma_loss  = f_plasma_eff * SR                               # [L/h]
    resp_H2O     = 1.292e-3 * delta_w * Ve_L_h                    # [L/h] minor
    fluid_in     = jnp.maximum(0.0, fluid_abs_interp.evaluate(t)) # [L/h] Mod 8
    dV_p_dt      = fluid_in - plasma_loss - resp_H2O

    # Plasma sodium: Mod 8 Na influx − flow-dependent sweat Na loss
    Na_sw_conc = (
        params.Na_sweat_min
        + (params.Na_sweat_max - params.Na_sweat_min)
        * (1.0 - jnp.exp(-SR / params.SR_Na_sat))
    )  # [mmol/L] — Adams & Best 1994; Montain 2007
    Na_sweat_loss = Na_sw_conc * SR                                # [mmol/h]
    na_in         = jnp.maximum(0.0, na_abs_interp.evaluate(t))   # [mmol/h] Mod 8
    dNa_plasma_dt = na_in - Na_sweat_loss

    return jnp.stack([dT_core_dt, dT_skin_dt, dV_p_dt, dNa_plasma_dt])


# ─────────────────────────────────────────────────────────────────────────────
# JIT kernel — single compilation boundary
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _solve_thermo_fluid(
    y0: jax.Array,
    t0: jax.Array,
    t1: jax.Array,
    dt0: jax.Array,
    ts: jax.Array,
    args: tuple,
) -> diffrax.Solution:
    """
    JIT-compiled Kvaerno5 integrator — 4-state Thermo-Fluid system v3.0.
    DIRK 5th-order stiff-safe for τ_core/τ_skin ≈ 6–12.
    max_steps=32_768: handles up to 8 h at fine adaptive steps.
    """
    return diffrax.diffeqsolve(
        terms=diffrax.ODETerm(thermo_fluid_ode),
        solver=diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=32_768,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ThermoFluidSolver:
    """
    Orchestrator for MHDS Subsistemas 10 & 12 — Thermo-Fluid v3.0.

    v3.0 architectural corrections over v2.0:
        ✓ ee_interp from Hub_Energy_Expenditure (Mod 1) — real array, not algebraic
        ✓ ve_interp from Hub_Minute_Ventilation_L (Mod 3) — real array, not algebraic
        ✓ ACTN3/eta_gross removed (1st Law violation fixed; η lives in Mod 1)
        ✓ SkBF capped at 420 L/h biological ceiling (≈7 L/min cardiac limit)

    v2.0 physics preserved:
        ✓ Antoine/Magnus equation → Emax; humidity physically blocks evaporative cooling
        ✓ Eres + Cres from real Ve hub signal
        ✓ Sodium ODE: [Na]_sw depends on sweat rate (gland reabsorption kinetics)
        ✓ nos3_prior         → SkBF vasodilation amplitude
        ✓ hspa1b_prior        → sweat acclimatisation onset and magnitude
        ✓ Hub_Basal_Temperature_Morning (Mod 6) → thermoregulatory setpoint shift
        ✓ Hub_Fluid_Absorption_Rate + Hub_Na_Absorption_Rate (Mod 8) → interpolators
        ✓ Wind speed modulates h_conv_eff (convection and evaporation ceiling)
        ✓ 6 Hub outbound: Core_Temp, Skin_Temp, Sweat_Rate, PV_Drop_Pct,
                          Sodium_Concentration, Skin_Blood_Flow

    Phase-3 prior injections (v3.0)
    ─────────────────────────────────
    nos3_prior (NOS3 T786C):
        TT → 1.20 (more eNOS → more NO → vasodilation↑)
        TC → 1.00 (reference)
        CC → 0.85 (less vasodilation)
        Multiplies k_BF directly. SkBF hard-capped at 420 L/h.

    hspa1b_prior (HSPA1B Hsp70 polymorphism):
        Favourable → 1.15 (earlier and higher sweat response)
        Reference  → 1.00
        Unfavourable → 0.88
        Multiplies k_sweat directly.
    """

    # ── Reference constants (used in fallback interp builders only) ───────────
    _C_CORE_REF: float       = 61.0   # kcal/°C — 70 kg × 0.9 × 0.97 (Gagge 1971)
    _C_SKIN_REF: float       = 6.8    # kcal/°C — 70 kg × 0.1 × 0.97
    _Q_MET_BASE_REF: float   = 72.0   # kcal/h  — BMR ≈ 1700 kcal/day; fallback only
    _ETA_GROSS_REF: float    = 0.22   # adim    — fallback gross efficiency (Coyle 1992)
    _K_BLOOD_REF: float      = 0.92   # kcal/(L·°C) — c_p_blood × ρ_blood (Pennes 1948)
    _BF_SKIN_BASE_REF: float = 3.0   # L/h     — basal SkBF (Rowell 1974)
    _K_BF_REF: float         = 6.0   # L/(h·°C)— vasodilation gain (Rowell 1974)
    _T_CORE_SET_REF: float   = 37.0  # °C      — thermoregulatory setpoint
    _K_SWEAT_REF: float      = 1.5   # L/(h·°C)— sweat gain (Nadel 1971)
    _H_EVAP_REF: float       = 581.0 # kcal/L  — latent heat at 37 °C
    _A_BODY_REF: float       = 1.8   # m²      — DuBois BSA (70 kg, 175 cm)
    _H_CONV_BASE_REF: float  = 9.0   # kcal/(h·m²·°C) ≈ 10.5 W/(m²·K) at 0.5 m/s
    _H_RAD_REF: float        = 4.2   # kcal/(h·m²·°C) ≈ 4.9 W/(m²·K) at 27 °C env
    _K_WIND_REF: float       = 2.58  # kcal/(h·m²·°C·√(m/s)) — McArdle wind scaling
    _VE_BASAL_REF: float     = 8.0   # L/min   — resting Ve (Grodins 1967); fallback only
    _K_VE_POWER_REF: float   = 90.0  # L/min per P_ref; fallback only
    _P_REF_W: float          = 200.0 # W; fallback only
    _F_PLASMA_SWEAT_REF: float   = 0.65  # adim — Maughan & Leiper 1985
    _K_OSMOTIC_SHIFT_REF: float  = 4.0   # adim — Costill 1977
    _V_P0_REF: float             = 3.0   # L
    _TAU_GASTRIC_REF: float      = 0.20  # h — Hunt & Stubbs 1975
    _NA_SWEAT_MIN_REF: float     = 20.0  # mmol/L
    _NA_SWEAT_MAX_REF: float     = 70.0  # mmol/L
    _SR_NA_SAT_REF: float        = 1.0   # L/h
    _K_MEL_BF_REF: float         = 15.0  # L/(h·unit) — Cagnacci 1992: ~2 L/min vasodilation peak

    # ─────────────────────────────────────────────────────────────────────────

    def _build_params(
        self,
        bayesian_priors: dict[str, float],
        hub_basal_temp_offset_C: float = 0.0,
    ) -> ThermoFluidParams:
        """
        Map Phase-3 Bayesian priors and Mod 6 hub → ThermoFluidParams.

        Keys consumed from bayesian_priors:
            "nos3_prior"        — NOS3 vasodilation scaling  [adim]
            "hspa1b_prior"      — Hsp70 sweat acclimatisation [adim]

        hub_basal_temp_offset_C:
            Hub_Basal_Temperature_Morning (Mod 6) − 37.0 °C
            Shifts T_core_set to reflect thyroid-driven thermoregulatory setpoint.

        NOTE: "actn3_r577x_prior" intentionally not consumed here.
              Mechanical efficiency belongs to Mod 1 (Hub_Energy_Expenditure).
        """
        # ① NOS3 T786C → SkBF vasodilation amplitude
        nos3_raw = bayesian_priors.get("nos3_prior", float("nan"))
        nos3_scale = 1.0 if math.isnan(float(nos3_raw)) else max(0.5, float(nos3_raw))

        # ② HSPA1B Hsp70 → sweat acclimatisation
        hspa1b_raw = bayesian_priors.get("hspa1b_prior", float("nan"))
        hspa1b_scale = 1.0 if math.isnan(float(hspa1b_raw)) else max(0.5, float(hspa1b_raw))

        # ③ Thyroid setpoint shift (Hub_Basal_Temperature_Morning − 37.0 °C)
        T_core_set = max(36.0, min(38.5, self._T_CORE_SET_REF + float(hub_basal_temp_offset_C)))

        return ThermoFluidParams(
            C_core          = self._C_CORE_REF,
            C_skin          = self._C_SKIN_REF,
            k_blood         = self._K_BLOOD_REF,
            BF_skin_base    = self._BF_SKIN_BASE_REF,
            k_BF            = self._K_BF_REF,
            nos3_scale      = float(nos3_scale),
            T_core_set      = float(T_core_set),
            k_sweat         = self._K_SWEAT_REF,
            hspa1b_scale    = float(hspa1b_scale),
            h_evap          = self._H_EVAP_REF,
            A_body          = self._A_BODY_REF,
            h_conv_base     = self._H_CONV_BASE_REF,
            h_rad           = self._H_RAD_REF,
            k_wind          = self._K_WIND_REF,
            f_plasma_sweat  = self._F_PLASMA_SWEAT_REF,
            k_osmotic_shift = self._K_OSMOTIC_SHIFT_REF,
            V_p0            = self._V_P0_REF,
            tau_gastric     = self._TAU_GASTRIC_REF,
            Na_sweat_min    = self._NA_SWEAT_MIN_REF,
            Na_sweat_max    = self._NA_SWEAT_MAX_REF,
            SR_Na_sat       = self._SR_NA_SAT_REF,
            k_Mel_BF        = self._K_MEL_BF_REF,
        )

    def _build_fluid_abs_interp(
        self,
        t0_h: float,
        t1_h: float,
        water_L: float,
        t_drink_h: float,
        tau_gastric: float,
        hub_t_h: jax.Array | None,
        hub_rate_L_h: jax.Array | None,
    ) -> diffrax.LinearInterpolation:
        """
        Build fluid absorption interpolator.
        If Mod 8 hub arrays provided → use them directly.
        Otherwise → gastric emptying mono-exponential fallback (Hunt & Stubbs 1975).
        """
        if hub_t_h is not None and hub_rate_L_h is not None:
            ts = jnp.asarray(hub_t_h, dtype=jnp.float32)
            ys = jnp.asarray(hub_rate_L_h, dtype=jnp.float32)
        else:
            ts   = jnp.linspace(t0_h, t1_h, 512, dtype=jnp.float32)
            gate = 0.5 * (1.0 + jnp.tanh(20.0 * (ts - float(t_drink_h))))
            dt   = jnp.maximum(0.0, ts - float(t_drink_h))
            ys   = gate * (float(water_L) / float(tau_gastric)) * jnp.exp(-dt / float(tau_gastric))
        return diffrax.LinearInterpolation(ts=ts, ys=ys)

    def _build_na_abs_interp(
        self,
        t0_h: float,
        t1_h: float,
        na_dietary_mg: float,
        t_meal_h: float,
        hub_t_h: jax.Array | None,
        hub_rate_mmol_h: jax.Array | None,
    ) -> diffrax.LinearInterpolation:
        """
        Build sodium absorption interpolator.
        If Mod 8 hub arrays provided → use them directly.
        Otherwise → exponential absorption from dietary Na intake (fallback).
        """
        if hub_t_h is not None and hub_rate_mmol_h is not None:
            ts = jnp.asarray(hub_t_h, dtype=jnp.float32)
            ys = jnp.asarray(hub_rate_mmol_h, dtype=jnp.float32)
        else:
            ts        = jnp.linspace(t0_h, t1_h, 512, dtype=jnp.float32)
            Na_mmol   = float(na_dietary_mg) / 22.99   # mg → mmol (Na MW = 22.99 g/mol)
            tau_Na    = 1.0                              # h — intestinal Na absorption τ
            gate      = 0.5 * (1.0 + jnp.tanh(20.0 * (ts - float(t_meal_h))))
            dt        = jnp.maximum(0.0, ts - float(t_meal_h))
            ys        = gate * (Na_mmol / tau_Na) * jnp.exp(-dt / tau_Na)
        return diffrax.LinearInterpolation(ts=ts, ys=ys)

    def _build_ee_interp(
        self,
        t0_h: float,
        t1_h: float,
        power_w: float,
        sess_start_h: float,
        sess_dur_h: float,
        hub_t_h: jax.Array | None,
        hub_ee_kcal_h: jax.Array | None,
    ) -> diffrax.LinearInterpolation:
        """
        Build energy expenditure interpolator (Hub_Energy_Expenditure from Mod 1).
        If Mod 1 hub arrays provided → use them directly.
        Fallback: EE = Q_met_base + P_t / eta_ref × W_TO_KCAL_H (standard 22% efficiency).
        """
        if hub_t_h is not None and hub_ee_kcal_h is not None:
            ts = jnp.asarray(hub_t_h, dtype=jnp.float32)
            ys = jnp.asarray(hub_ee_kcal_h, dtype=jnp.float32)
        else:
            ts    = jnp.linspace(t0_h, t1_h, 512, dtype=jnp.float32)
            gate  = (
                0.5 * (1.0 + jnp.tanh(20.0 * (ts - float(sess_start_h))))
                - 0.5 * (1.0 + jnp.tanh(20.0 * (ts - float(sess_start_h + sess_dur_h))))
            )
            P_t_fb = float(power_w) * gate
            # Total EE = resting BMR + exercise metabolic rate (power / efficiency)
            ys = (
                jnp.float32(self._Q_MET_BASE_REF)
                + P_t_fb * jnp.float32(_W_TO_KCAL_H / self._ETA_GROSS_REF)
            )
        return diffrax.LinearInterpolation(ts=ts, ys=ys)

    def _build_ve_interp(
        self,
        t0_h: float,
        t1_h: float,
        power_w: float,
        sess_start_h: float,
        sess_dur_h: float,
        hub_t_h: jax.Array | None,
        hub_ve_L_min: jax.Array | None,
    ) -> diffrax.LinearInterpolation:
        """
        Build minute ventilation interpolator (Hub_Minute_Ventilation_L from Mod 3).
        If Mod 3 hub arrays provided → use them directly.
        Fallback: Ve = Ve_basal + k_Ve_power × (P_t / P_ref) — Grodins model (1967).
        """
        if hub_t_h is not None and hub_ve_L_min is not None:
            ts = jnp.asarray(hub_t_h, dtype=jnp.float32)
            ys = jnp.asarray(hub_ve_L_min, dtype=jnp.float32)
        else:
            ts    = jnp.linspace(t0_h, t1_h, 512, dtype=jnp.float32)
            gate  = (
                0.5 * (1.0 + jnp.tanh(20.0 * (ts - float(sess_start_h))))
                - 0.5 * (1.0 + jnp.tanh(20.0 * (ts - float(sess_start_h + sess_dur_h))))
            )
            P_t_fb  = float(power_w) * gate
            P_norm  = P_t_fb / jnp.float32(self._P_REF_W)
            ys = jnp.float32(self._VE_BASAL_REF) + jnp.float32(self._K_VE_POWER_REF) * P_norm
        return diffrax.LinearInterpolation(ts=ts, ys=ys)

    def simulate_thermo_fluid_response(
        self,
        bayesian_priors: dict[str, float],
        session_record: dict,
        fluid_intake: dict,
        env_conditions: dict,
        hub_inbound: dict | None = None,
        t_span_hours: float = 2.0,
        n_save_points: int = 240,
    ) -> dict:
        """
        Simulate thermoregulation, fluid balance, and plasma sodium — biophysics v3.0.

        Parameters
        ──────────
        bayesian_priors : dict
            "nos3_prior"         [adim]  NOS3 vasodilation scaling
            "hspa1b_prior"       [adim]  Hsp70 sweat acclimatisation
            NOTE: "actn3_r577x_prior" ignored — η lives in Hub_Energy_Expenditure (Mod 1).

        session_record : dict
            "power_output_watts"    [W]   mean session power
            "session_start_h"       [h]   session onset from window t=0
            "session_duration_secs" [s]   session length

        fluid_intake : dict
            "water_intake_L"        [L]   total water ingested
            "drink_timestamp_h"     [h]   when drinking occurs (fallback)
            "na_dietary_mg"         [mg]  total dietary Na (fallback)
            "na_meal_timestamp_h"   [h]   meal time (fallback)

        env_conditions : dict
            "T_env_celsius"         [°C]  ambient temperature
            "RH_fraction"           [0-1] relative humidity (default 0.5)
            "wind_speed_m_s"        [m/s] wind speed (default 0.5 m/s)
            "T_core_init_celsius"   [°C]  initial T_core (default 37.0)
            "T_skin_init_celsius"   [°C]  initial T_skin (default 34.0)
            "V_p_init_L"            [L]   initial plasma volume (default 3.0)

        hub_inbound : dict | None
            Hub signals from coupled modules (MANDATORY for full modular coupling):
                — Mod 1 (Energy Expenditure) —
                "ee_t_h"              [h]      time axis
                "ee_arr_kcal_h"       [kcal/h] total energy expenditure array
                — Mod 3 (Ventilation) —
                "ve_t_h"              [h]      time axis
                "ve_arr_L_min"        [L/min]  minute ventilation array
                — Mod 8 (GI absorption) —
                "fluid_abs_t_h"       [h]      time axis
                "fluid_abs_rate_L_h"  [L/h]    fluid absorption rate
                "na_abs_t_h"          [h]      time axis
                "na_abs_rate_mmol_h"  [mmol/h] Na absorption rate
                — Mod 6 (Thyroid) —
                "basal_temp_offset_C" [°C]     thermoregulatory setpoint shift

        Returns
        ───────
        dict with JAX arrays, shape (n_save_points,):

        Raw trajectories:
            "t_h"                           time axis [h]
            "T_core_C"                      core temperature [°C]
            "T_skin_C"                      skin temperature [°C]
            "V_p_L"                         plasma volume [L]
            "Na_plasma_mmol"                total plasma sodium [mmol]

        Reconstructed observables:
            "Sweat_Rate_L_h"                sweat rate [L/h]
            "SkBF_L_h"                      skin blood flow [L/h]
            "Emax_kcal_h"                   max evaporative capacity [kcal/h]
            "Esw_actual_kcal_h"             actual evaporative cooling [kcal/h]

        Hub Outbound (6 arrays — Bond Graph «placa-mãe»):
            "Hub_Core_Temp"                 [°C]     A) → Mod 4, Mod 5, NMPC
            "Hub_Skin_Temp"                 [°C]     B) → Mod 12, Mod 11
            "Hub_Sweat_Rate_L_h"            [L/h]    C) → NMPC hydration prescription
            "Hub_Plasma_Volume_Drop_Pct"    [%]      D) → Mod 3 cardiovascular drift
            "Hub_Sodium_Concentration"      [mmol/L] E) → Mod 12 osmolarity
            "Hub_Skin_Blood_Flow"           [L/min]  F) → GI shunts, Mod 3 preload
        """
        hub = hub_inbound or {}
        basal_temp_offset = float(hub.get("basal_temp_offset_C", 0.0))
        # Melatonin hub (Mod 4 → Mod 10): nocturnal vasodilation
        mel_raw = hub.get("mel_arr")
        mel_t   = hub.get("mel_t_h")
        params = self._build_params(bayesian_priors, basal_temp_offset)

        # ── Session inputs ─────────────────────────────────────────────────
        power_w      = float(session_record.get("power_output_watts", 0.0) or 0.0)
        sess_start_h = float(session_record.get("session_start_h", 0.0) or 0.0)
        sess_dur_h   = float(session_record.get("session_duration_secs", 0.0) or 0.0) / 3600.0

        # ── Fluid and sodium intake ────────────────────────────────────────
        water_L   = float(fluid_intake.get("water_intake_L", 0.0) or 0.0)
        t_drink_h = float(fluid_intake.get("drink_timestamp_h", 0.0) or 0.0)
        na_mg     = float(fluid_intake.get("na_dietary_mg", 0.0) or 0.0)
        t_meal_h  = float(fluid_intake.get("na_meal_timestamp_h", 0.0) or 0.0)

        # ── Environmental inputs ───────────────────────────────────────────
        T_amb   = jnp.float32(float(env_conditions.get("T_env_celsius", 22.0)))
        RH_frac = jnp.float32(float(env_conditions.get("RH_fraction", 0.5)))
        v_wind  = jnp.float32(float(env_conditions.get("wind_speed_m_s", 0.5)))

        # ── Initial conditions ─────────────────────────────────────────────
        T_core_init = float(env_conditions.get("T_core_init_celsius", 37.0))
        T_skin_init = float(env_conditions.get("T_skin_init_celsius", 34.0))
        V_p_init    = float(env_conditions.get("V_p_init_L", self._V_P0_REF))
        Na_init     = _NA_REF_MMOL_L * V_p_init  # 140 mmol/L × V_p_init [mmol]
        y0 = jnp.array([T_core_init, T_skin_init, V_p_init, Na_init], dtype=jnp.float32)

        # ── Time axis ──────────────────────────────────────────────────────
        t0 = jnp.float32(0.0)
        t1 = jnp.float32(t_span_hours)
        ts = jnp.linspace(t0, t1, n_save_points, dtype=jnp.float32)

        # ── Build hub inbound interpolators ───────────────────────────────
        # Mod 8: fluid and Na absorption
        fluid_abs_interp = self._build_fluid_abs_interp(
            t0_h=0.0, t1_h=float(t_span_hours),
            water_L=water_L, t_drink_h=t_drink_h, tau_gastric=params.tau_gastric,
            hub_t_h=hub.get("fluid_abs_t_h"),
            hub_rate_L_h=hub.get("fluid_abs_rate_L_h"),
        )
        na_abs_interp = self._build_na_abs_interp(
            t0_h=0.0, t1_h=float(t_span_hours),
            na_dietary_mg=na_mg, t_meal_h=t_meal_h,
            hub_t_h=hub.get("na_abs_t_h"),
            hub_rate_mmol_h=hub.get("na_abs_rate_mmol_h"),
        )
        # Mod 1: total energy expenditure
        ee_interp = self._build_ee_interp(
            t0_h=0.0, t1_h=float(t_span_hours),
            power_w=power_w, sess_start_h=sess_start_h, sess_dur_h=sess_dur_h,
            hub_t_h=hub.get("ee_t_h"),
            hub_ee_kcal_h=hub.get("ee_arr_kcal_h"),
        )
        # Mod 3: minute ventilation
        ve_interp = self._build_ve_interp(
            t0_h=0.0, t1_h=float(t_span_hours),
            power_w=power_w, sess_start_h=sess_start_h, sess_dur_h=sess_dur_h,
            hub_t_h=hub.get("ve_t_h"),
            hub_ve_L_min=hub.get("ve_arr_L_min"),
        )

        # ── Melatonin interpolator ─────────────────────────────────────────
        if mel_raw is not None and mel_t is not None:
            mel_ts = jnp.asarray(mel_t,   dtype=jnp.float32)
            mel_ys = jnp.asarray(mel_raw, dtype=jnp.float32)
        else:
            mel_ts = jnp.array([0.0, jnp.float32(t_span_hours)])
            mel_ys = jnp.array([0.0, 0.0], dtype=jnp.float32)
        mel_interp = diffrax.LinearInterpolation(ts=mel_ts, ys=mel_ys)

        # ── Assemble ODE args and solve ────────────────────────────────────
        args = (
            params,
            jnp.float32(power_w),
            jnp.float32(sess_start_h),
            jnp.float32(sess_dur_h),
            fluid_abs_interp,
            na_abs_interp,
            ee_interp,
            ve_interp,
            T_amb,
            RH_frac,
            v_wind,
            mel_interp,
        )

        sol = _solve_thermo_fluid(
            y0=y0, t0=t0, t1=t1,
            dt0=jnp.float32(1e-3),
            ts=ts, args=args,
        )

        ys         = sol.ys
        T_core_arr = ys[:, 0]
        T_skin_arr = ys[:, 1]
        V_p_arr    = ys[:, 2]
        Na_arr     = ys[:, 3]

        # ── Reconstruct algebraic observables from saved states ────────────
        dT_err   = jnp.maximum(0.0, T_core_arr - jnp.float32(params.T_core_set))
        SR_arr   = jnp.float32(params.k_sweat * params.hspa1b_scale) * dT_err
        SkBF_arr = jnp.clip(
            jnp.float32(params.BF_skin_base)
            + jnp.float32(params.k_BF * params.nos3_scale) * dT_err,
            0.0,
            jnp.float32(_SKBF_MAX_L_H),
        )  # [L/h] — capped at 420 L/h biological ceiling

        # Antoine-based Emax at each saved skin temperature
        P_vap_amb_sc   = RH_frac * _p_sat_kpa(T_amb)
        h_conv_eff_sc  = (
            jnp.float32(params.h_conv_base)
            + jnp.float32(params.k_wind) * jnp.sqrt(jnp.maximum(0.0, v_wind))
        )
        P_sat_skin_arr = _p_sat_kpa(T_skin_arr)
        Emax_arr = (
            16.7 * h_conv_eff_sc * jnp.float32(params.A_body)
            * jnp.maximum(0.0, P_sat_skin_arr - P_vap_amb_sc)
        )
        Esw_arr = jnp.minimum(SR_arr * jnp.float32(params.h_evap), Emax_arr)

        # ── Hub Outbound — 6 arrays (Bond Graph «placa-mãe») ──────────────
        Hub_A = T_core_arr                                              # Core_Temp [°C]
        Hub_B = T_skin_arr                                              # Skin_Temp [°C]
        Hub_C = SR_arr                                                  # Sweat_Rate_L_h [L/h]
        Hub_D = (1.0 - V_p_arr / jnp.float32(V_p_init)) * 100.0       # PV_Drop_Pct [%]
        Hub_E = Na_arr / jnp.maximum(V_p_arr, 0.1)                    # Sodium_Conc [mmol/L]
        Hub_F = SkBF_arr / 60.0                                        # Skin_Blood_Flow [L/min]

        return {
            # raw trajectories
            "t_h":              ts,
            "T_core_C":         T_core_arr,
            "T_skin_C":         T_skin_arr,
            "V_p_L":            V_p_arr,
            "Na_plasma_mmol":   Na_arr,
            # observables
            "Sweat_Rate_L_h":   SR_arr,
            "SkBF_L_h":         SkBF_arr,
            "Emax_kcal_h":      Emax_arr,
            "Esw_actual_kcal_h": Esw_arr,
            # Hub Outbound — Bond Graph placa-mãe (6 arrays)
            "Hub_Core_Temp":               Hub_A,
            "Hub_Skin_Temp":               Hub_B,
            "Hub_Sweat_Rate_L_h":          Hub_C,
            "Hub_Plasma_Volume_Drop_Pct":  Hub_D,
            "Hub_Sodium_Concentration":    Hub_E,
            "Hub_Skin_Blood_Flow":         Hub_F,
            # v1.0 backward-compatible aliases
            "Hub_Core_Temperature_Alarm":  Hub_A,
            "Hub_Plasma_Volume_Drift":     V_p_arr / jnp.float32(V_p_init),
        }
