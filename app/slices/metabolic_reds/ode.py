"""
app/slices/metabolic_reds/ode.py — Module 13

Thyroid / RED-S / Fatmax ODE — Daily Timescale (L2 Backbone)

Architecture
────────────
Models the suppression of thyroid hormone conversion and metabolic rate
via the SPINA-Dietrich RED-S mechanism: deiodinase (D2) activity collapses
non-linearly when Energy Availability (EA) falls below 30 kcal/kg FFM,
reducing T4 → T3 conversion and driving downward metabolic adaptation.

  1. ENERGY AVAILABILITY (Loucks 2003; Mountjoy 2018 IOC RED-S consensus)
     EA = (caloric intake − exercise EE) / FFM.  Pool relaxes toward the
     running 7-day average.  Below 30 kcal/kg FFM, neuroendocrine function
     is suppressed; below 20, bone loss accelerates.

  2. FREE T4 PRODUCTION (Hackney 2020; Schussler 2013)
     Thyroid gland maintains basal T4 secretion.  Prolonged training stress
     mildly suppresses hypothalamic TRH → reduced TSH pulsatility → reduced
     T4 output (~5% at maximal training stress; Hackney 2020 review).

  3. DEIODINASE T4 → T3 CONVERSION (SPINA-Dietrich 2015; Loucks 2003)
     Type-2 deiodinase activity follows a Hill function of EA_Pool with
     half-saturation at K_deio = 30 kcal/kg FFM (the RED-S threshold).
     Hill coefficient n=3 gives the sharp logistic-like collapse observed
     in prospective studies (Loucks & Thuma 2003; De Souza 2014).

  4. RMR MULTIPLIER ADAPTATION (Prentice 1992; Müller 2013)
     Resting metabolic rate tracks fT3 with a 14-day adaptation lag.
     Severe deficit → RMR_Multiplier < 0.85 signals metabolic conservation.

  5. FATMAX STATE (Achten & Jeukendrup 2003; Holloszy 2011)
     Maximal fat oxidation capacity tracks RMR_Multiplier with a slower
     21-day lag (mitochondrial biogenesis time constant; Holloszy 2011).

TIME UNIT: DAYS.

═══════════════════════════════════════════════════════════════════════════════
STATE VECTOR  x ∈ ℝ⁵   (time unit = days)
═══════════════════════════════════════════════════════════════════════════════

  x[0]  EA_Pool         Energy Availability (7-day avg)  [kcal/kg FFM/day; ref=45]
  x[1]  Free_T4         Free thyroxine                   [pmol/L; ref=16.0]
  x[2]  Free_T3         Free triiodothyronine             [pmol/L; ref=5.0]
  x[3]  RMR_Multiplier  RMR relative to baseline         [au; 1.0=normal, 0.85=severe]
  x[4]  Fatmax_State    Max fat oxidation capacity        [au; 1.0=normal]

CONTROL INPUTS  u ∈ ℝ⁴  (hub variables from orchestrator)
  u[0]  hub_caloric_intake             Daily caloric intake          [kcal/day]
  u[1]  hub_total_energy_expenditure   Total daily EE               [kcal/day]
  u[2]  hub_training_stress            Cumulative daily load         [au; 0=rest, 1=maximal]
  u[3]  hub_fat_free_mass_kg           Fat-free mass                 [kg]

ODE EQUATIONS  (time unit: days)
────────────────────────────────────────────────────────────────────────────────

BLOCK A — Energy Availability Pool
  EA_net = (caloric_intake − TEE) / max(FFM_kg, 0.5) + EA_setpoint
  dEA_Pool/dt = k_ea_relax × (EA_net − EA_Pool)
  k_ea_relax = 1/7 day⁻¹  (7-day exponential moving average)
  EA_setpoint = 45.0  (energy balance → EA converges to 45)

BLOCK B — Free T4
  k_t4_prod = k_t4_clear × fT4_ref   (ensures SS = fT4_ref at rest)
  dFree_T4/dt = k_t4_prod × (1 − k_t4_stress_sup × hub_training_stress)
              − k_t4_clear × Free_T4
  t½(T4) ≈ 10 days → k_t4_clear = ln(2)/10 = 0.0693 day⁻¹

BLOCK C — Free T3 (deiodinase gated by EA)
  deio_activity = EA_Pool³ / (K_deio³ + EA_Pool³)    [Hill, n=3]
    At EA=45:  deio ≈ 0.772  (normal T3 production)
    At EA=30:  deio = 0.500  (50% — RED-S threshold)
    At EA=15:  deio ≈ 0.111  (severe — fT3 crashes)
  dFree_T3/dt = k_t3_conv × deio_activity × Free_T4 − k_t3_clear × Free_T3
  k_t3_clear = ln(2)/1.0 = 0.693 day⁻¹  (t½ ≈ 1 day)
  k_t3_conv = 0.2806  (calibrated: SS(T3)=5.0 at EA=45, T4=16)

BLOCK D — RMR Multiplier  (follows fT3/fT3_ref; 14-day lag)
  fT3_norm = Free_T3 / fT3_ref
  dRMR_Multiplier/dt = k_rmr_relax × (fT3_norm − RMR_Multiplier)
  k_rmr_relax = 1/14 day⁻¹

BLOCK E — Fatmax State  (follows RMR_Multiplier; 21-day lag)
  dFatmax_State/dt = k_fatmax_relax × (RMR_Multiplier − Fatmax_State)
  k_fatmax_relax = 1/21 day⁻¹

HUB EXPORTS (algebraic)
  Hub_MR_EA_Pool          [kcal/kg FFM/day] → Phase3Envelope (RED-S gate)
  Hub_MR_RMR_Multiplier   [au]              → Phase3Envelope (metabolic adaptation)
  Hub_MR_Fatmax_State     [au]              → Mod 1/aerobic (fat oxidation ceiling)
  Hub_MR_Free_T3          [pmol/L]          → observation layer (lab assay)
  Hub_MR_Free_T4          [pmol/L]          → observation layer (lab assay)

Steady-state verification
  Rest (EA=45, stress=0): T4→16.0, T3→5.0, RMR→1.0, Fatmax→1.0 ✓
  Chronic RED-S (EA=28, stress=0.5, FFM=55, CI=1800, TEE=2500):
    EA_net = (1800-2500)/55 + 45 = -12.73 + 45 = 32.3
    EA_Pool → 32.3 (above hard floor 30, chance constraint 33 triggered)

References
──────────
  Achten J., Jeukendrup A.E. (2003) Sports Med 33(8):559–591
  De Souza M.J. et al. (2014) Br J Sports Med 48(4):289–300
  Hackney A.C. (2020) J Funct Morphol Kinesiol 5(1):5
  Holloszy J.O. (2011) J Appl Physiol 110(3):694–701
  Loucks A.B. (2003) J Sports Sci 21(10):879–883
  Loucks A.B., Thuma J.R. (2003) J Clin Endocrinol Metab 88(1):297–311
  Mountjoy M. et al. (2018) Br J Sports Med 52(11):687–697
  Müller M.J. et al. (2013) Obes Rev 14(11):908–921
  Prentice A.M. et al. (1992) Eur J Clin Nutr 46:S91–S98
  Schussler G.C. (2013) Thyroid 23(7):823–843
  SPINA-Dietrich (2015) Front Endocrinol 6:57
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── State vector indices ──────────────────────────────────────────────────────
IDX_EA_POOL  = 0   # Energy Availability (7-day avg)  [kcal/kg FFM/day; ref=45]
IDX_FREE_T4  = 1   # Free thyroxine                   [pmol/L; ref=16.0]
IDX_FREE_T3  = 2   # Free triiodothyronine             [pmol/L; ref=5.0]
IDX_RMR_MULT = 3   # RMR Multiplier                   [au; 1.0=normal]
IDX_FATMAX   = 4   # Fatmax State (max fat oxidation)  [au; 1.0=normal]

STATE_DIM: int = 5
OBS_DIM:   int = 3   # [fT3_obs [pmol/L], fT4_obs [pmol/L], RMR_Proxy [kcal/day]]

# ── Control input indices ─────────────────────────────────────────────────────
CIDX_CALORIC_INTAKE  = 0   # hub_caloric_intake            [kcal/day]
CIDX_TEE             = 1   # hub_total_energy_expenditure  [kcal/day]
CIDX_TRAINING_STRESS = 2   # hub_training_stress           [au; 0-1]
CIDX_FFM_KG          = 3   # hub_fat_free_mass_kg          [kg]

CTRL_DIM: int = 4

EPS: float = 1e-8

# ── Default initial state ─────────────────────────────────────────────────────
# Healthy reference: energy-balanced athlete, normal thyroid, normal metabolism.
X0_MR_DEFAULT: list[float] = [
    45.0,   # EA_Pool         [kcal/kg FFM/day]: Loucks (2003) normal reference
    16.0,   # Free_T4         [pmol/L]: population midpoint (Brent 2013)
     5.0,   # Free_T3         [pmol/L]: population midpoint (Brent 2013)
     1.0,   # RMR_Multiplier  [au]: fully normal metabolic rate
     1.0,   # Fatmax_State    [au]: fully normal fat oxidation capacity
]


# ── ODE parameters ────────────────────────────────────────────────────────────
class MetabolicRedsParams(NamedTuple):
    # ── Block A: Energy Availability Pool ────────────────────────────────────
    EA_setpoint:       float = 45.0          # [kcal/kg FFM/day] energy-balance reference
    k_ea_relax:        float = 1.0 / 7.0     # day⁻¹; 7-day exponential smoothing

    # ── Block B: Free T4 kinetics ─────────────────────────────────────────────
    # Schussler 2013: t½(T4) ≈ 10 days → k_clear = ln(2)/10
    # Hackney 2020: ~5% T4 suppression under sustained training stress
    fT4_ref:           float = 16.0          # [pmol/L] normal midpoint
    k_t4_clear:        float = 0.0693        # day⁻¹ (t½ = 10 days)
    k_t4_stress_sup:   float = 0.05          # [au] fractional T4 suppression at peak stress

    # ── Block C: Free T3 / Deiodinase (SPINA-Dietrich RED-S model) ───────────
    # Hill function: K_deio = RED-S threshold (Loucks 2003)
    K_deio:            float = 30.0          # [kcal/kg FFM] Hill half-saturation
    n_deio:            float = 3.0           # Hill coefficient (sharp collapse)
    # k_t3_conv calibration: at EA=45, T4=16 → SS T3=5.0
    #   deio(45) = 45³/(30³+45³) = 91125/118125 ≈ 0.7716
    #   k_t3_conv = k_t3_clear × fT3_ref / (deio(45) × fT4_ref)
    #             = 0.693 × 5.0 / (0.7716 × 16.0) ≈ 0.2806
    fT3_ref:           float = 5.0           # [pmol/L] normal midpoint
    k_t3_clear:        float = 0.693         # day⁻¹ (t½ = 1 day; Brent 2013)
    k_t3_conv:         float = 0.2806        # conversion factor (T4 → T3 via deiodinase)

    # ── Block D: RMR Multiplier adaptation ───────────────────────────────────
    # Prentice 1992: ~14-day lag for full RMR adaptation to thyroid change
    k_rmr_relax:       float = 1.0 / 14.0   # day⁻¹ (tau = 14 days)

    # ── Block E: Fatmax State (mitochondrial biogenesis) ─────────────────────
    # Holloszy 2011: mitochondrial adaptation ~3 weeks lag
    k_fatmax_relax:    float = 1.0 / 21.0   # day⁻¹ (tau = 21 days)


DEFAULT_MR_PARAMS: MetabolicRedsParams = MetabolicRedsParams()


# ── ODE vector field ──────────────────────────────────────────────────────────
def metabolic_reds_ode(
    t:    float,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    dx/dt for the metabolic_reds 5-state daily ODE.

    Parameters
    ----------
    t    : current time [days] (unused; autonomous system)
    x    : state (STATE_DIM=5,)
    args : (u, params) where
           u      is (CTRL_DIM=4,) control vector
           params is MetabolicRedsParams

    Returns
    -------
    dxdt : (STATE_DIM=5,) [mixed units per day]
    """
    u, params = args

    # ── NaN guards on hub inputs ──────────────────────────────────────────────
    hub_ci  = jnp.where(
        jnp.isfinite(u[CIDX_CALORIC_INTAKE]),
        u[CIDX_CALORIC_INTAKE], jnp.float32(2500.0)
    )
    hub_tee = jnp.where(
        jnp.isfinite(u[CIDX_TEE]),
        u[CIDX_TEE], jnp.float32(2500.0)
    )
    hub_ts = jnp.clip(
        jnp.where(
            jnp.isfinite(u[CIDX_TRAINING_STRESS]),
            u[CIDX_TRAINING_STRESS], jnp.float32(0.0)
        ),
        jnp.float32(0.0), jnp.float32(1.0)
    )
    hub_ffm = jnp.where(
        jnp.isfinite(u[CIDX_FFM_KG]) & (u[CIDX_FFM_KG] > jnp.float32(0.5)),
        u[CIDX_FFM_KG], jnp.float32(60.0)
    )

    # ── State clipping (physics bounds) ───────────────────────────────────────
    EA_c  = jnp.clip(x[IDX_EA_POOL],  jnp.float32(0.0),  jnp.float32(80.0))
    fT4_c = jnp.clip(x[IDX_FREE_T4],  jnp.float32(0.1),  jnp.float32(40.0))
    fT3_c = jnp.clip(x[IDX_FREE_T3],  jnp.float32(0.0),  jnp.float32(20.0))
    rmr_c = jnp.clip(x[IDX_RMR_MULT], jnp.float32(0.5),  jnp.float32(1.2))
    fat_c = jnp.clip(x[IDX_FATMAX],   jnp.float32(0.3),  jnp.float32(1.1))

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK A — Energy Availability Pool
    # EA_net [kcal/kg FFM/day] = (CI - TEE) / FFM + EA_setpoint
    # dEA_Pool/dt = k_ea_relax × (EA_net - EA_Pool)  [7-day moving avg]
    # ─────────────────────────────────────────────────────────────────────────
    ea_net = (hub_ci - hub_tee) / hub_ffm + jnp.float32(params.EA_setpoint)
    ea_net = jnp.clip(ea_net, jnp.float32(0.0), jnp.float32(80.0))
    dEA_Pool_dt = jnp.float32(params.k_ea_relax) * (ea_net - EA_c)

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK B — Free T4
    # k_t4_prod = k_t4_clear × fT4_ref  →  SS at rest = fT4_ref = 16.0 pmol/L
    # Mild suppression under training stress (Hackney 2020)
    # ─────────────────────────────────────────────────────────────────────────
    k_t4_prod  = jnp.float32(params.k_t4_clear * params.fT4_ref)
    t4_suppress = jnp.float32(1.0) - jnp.float32(params.k_t4_stress_sup) * hub_ts
    dFree_T4_dt = k_t4_prod * t4_suppress - jnp.float32(params.k_t4_clear) * fT4_c

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK C — Free T3 via Deiodinase (SPINA-Dietrich RED-S collapse)
    # Hill: deio_activity = EA^n / (K_deio^n + EA^n)
    #   EA=45 → deio ≈ 0.772  (normal)
    #   EA=30 → deio = 0.500  (RED-S threshold: half-activity)
    #   EA=15 → deio ≈ 0.111  (severe suppression)
    # ─────────────────────────────────────────────────────────────────────────
    K_n        = jnp.float32(params.K_deio ** params.n_deio)
    EA_n       = EA_c ** jnp.float32(params.n_deio)
    deio_act   = EA_n / (K_n + EA_n + jnp.float32(EPS))
    t3_conv    = jnp.float32(params.k_t3_conv) * deio_act * fT4_c
    dFree_T3_dt = t3_conv - jnp.float32(params.k_t3_clear) * fT3_c

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK D — RMR Multiplier  (tracks fT3; 14-day adaptation lag)
    # fT3_norm = fT3 / fT3_ref  (1.0 at normal, <1 under RED-S)
    # ─────────────────────────────────────────────────────────────────────────
    fT3_norm    = fT3_c / jnp.float32(params.fT3_ref + EPS)
    fT3_norm    = jnp.clip(fT3_norm, jnp.float32(0.5), jnp.float32(1.2))
    dRMR_Mult_dt = jnp.float32(params.k_rmr_relax) * (fT3_norm - rmr_c)

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK E — Fatmax State  (tracks RMR Multiplier; 21-day mitochondrial lag)
    # ─────────────────────────────────────────────────────────────────────────
    dFatmax_dt = jnp.float32(params.k_fatmax_relax) * (rmr_c - fat_c)

    return jnp.stack([
        dEA_Pool_dt,
        dFree_T4_dt,
        dFree_T3_dt,
        dRMR_Mult_dt,
        dFatmax_dt,
    ])
