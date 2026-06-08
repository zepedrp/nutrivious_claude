"""
app/slices/neuromuscular_tissue/ode.py

Neuromuscular Tissue ODE V4.0 -- Intra-Session MINUTES Timescale (L2 Backbone)

Physiology modelled:
  1. Fiber recruitment -- Henneman size principle (Type 1 / Type 2 split)
  2. Calcium transient -- SR release proportional to recruitment;
     re-uptake via SERCA pump inhibited by lactate
  3. ATP pool -- drained by active fibers AND SERCA pump; re-synthesis
     glycogen-primary (Block C proactive thermodynamic fix)
  4. Low-Frequency Fatigue (LFF) -- RyR1 channel oxidative damage from
     sustained cytosolic Ca overload (ROS cascade; Westerblad & Allen 2002)
  5. Muscle Glycogen -- local glycogen pool; depleted by T2 (anaerobic
     glycolysis) and T1 (oxidative); resynthesized slowly at rest.
     Glycogen depletion gates T2 max (Bonking / Hitting the Wall).

STATE VECTOR  x in R^6  (minutes timescale)
-------------------------------------------
  x[0]  ATP_Muscle_mmol       Intra-muscular ATP pool              [mmol/kg dm]
  x[1]  Calcium_Cytosolic_uM  Cytosolic free Ca2+                  [uM]
  x[2]  Recruitment_Type1     Slow-oxidative fiber active fraction  [0, 1]
  x[3]  Recruitment_Type2     Fast-glycolytic fiber active fraction [0, 1]
  x[4]  RyR1_Damage_au        Ryanodine receptor oxidative damage   [au, 0+]
  x[5]  Muscle_Glycogen_mmol  Local glycogen pool                   [mmol/kg dm]

HUB INPUTS (Dim=3) -- NaN guards applied
-----------------------------------------
  u[0]  hub_Power_W              Mechanical power output       [W]      NaN->0.0
  u[1]  hub_Lactate_mmolL        Blood lactate concentration   [mmol/L] NaN->1.0
  u[2]  hub_Plasma_Glucose_mgdL  Plasma glucose                [mg/dL]  NaN->90.0

HUB OUTPUTS (algebraic)
------------------------
  Hub_Peripheral_Fatigue_au
    = alpha_ATP * max(0, ATP_rest - ATP) + alpha_RyR1 * RyR1_Damage
    Clamped to [0, 10]
  Hub_Muscle_Glycogen_mmolkg  (monitored separately by envelope V4.0)

BLOCK A -- Henneman Recruitment (size principle; Burke 1967; Enoka 1994)
  R1_target       = sigmoid(k_sig * (Power - P_th_1))
  effective       = Power + k_fat_comp * RyR1_Damage
  hypoglycemia_gate = max(0, 1 - k_hypo_T2 * max(0, glu_hypo_thr - Glucose))
  glycogen_gate   = sigmoid(k_glycogen_gate * (Glycogen - Glycogen_bonk_thr))
  T2_max          = glycogen_gate * hypoglycemia_gate
  R2_target       = sigmoid(k_sig * (effective - P_th_2)) * T2_max
  dR/dt           = k_rec * (R_target - R)   tau = 0.5 min

BLOCK B -- Calcium Transient (Berchtold 2000; Allen 2008)
  Ca_release = k_Ca_release * (R1 + 2*R2) + Ca_basal_leak
  SERCA_rate = k_SERCA_base / (1 + k_lac_SERCA * Lactate)
  dCa/dt     = Ca_release - SERCA_rate * Ca

BLOCK C -- ATP Pool (Sahlin 1998) [PROACTIVE V4.0 THERMODYNAMIC FIX]
  fuel_avail = k_gly_ATP * (Glycogen / Glycogen_ref)
             + k_glc_ATP * (Glucose / glucose_ref)
  resyn      = k_ATP_resyn * fuel_avail * max(0, ATP_rest - ATP)
  drain_fib  = k_drain_T1 * R1 + k_drain_T2 * R2
  drain_serca= k_SERCA_ATP * SERCA_rate * Ca  [SERCA consumes ATP]
  dATP/dt    = resyn - drain_fib - drain_serca

BLOCK D -- LFF: RyR1 Damage (Westerblad & Allen 2002; Durham 2008)
  Ca_excess  = max(0, Ca - Ca_ROS_thresh)
  dRyR1/dt   = k_ROS * Ca_excess - k_RyR1_repair * RyR1

BLOCK E -- Muscle Glycogen (Bergstrom 1967; Gollnick 1974)
  Gly_pos    = max(0, Glycogen)
  Gly_sat    = Gly_pos / (Gly_pos + 1.0)          [saturating depletion floor]
  depletion  = (k_gly_T1 * R1 + k_gly_T2 * R2) * Gly_sat
  glc_excess = max(0, Glucose - glucose_ref) / glucose_ref
  rest_gate  = max(0, 1 - (R1 + R2))              [resynthesis only at rest]
  resyn_gly  = k_gly_resyn * glc_excess * rest_gate
  dGly/dt    = resyn_gly - depletion

References
----------
  Burke R.E. et al. (1967) J Physiol 189:545-556       [Henneman size principle]
  Enoka R.M. (1994) Neuromechanics of Human Movement    [recruitment]
  Berchtold M.W. et al. (2000) Physiol Rev 80:1215      [Ca cycling]
  Allen D.G. et al. (2008) Physiol Rev 88:287-332       [Ca & fatigue]
  Sahlin K. (1998) Can J Appl Physiol 23:87             [ATP limits]
  Westerblad H., Allen D.G. (2002) J Physiol 540:111    [ROS-RyR1 LFF]
  Durham W.J. et al. (2008) Cell 133:53-65              [RyR1 oxidation]
  Bergstrom J. et al. (1967) Acta Physiol Scand 71:140  [muscle glycogen]
  Gollnick P.D. et al. (1974) J Appl Physiol 37:614     [glycogen depletion]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# -- State vector indices -------------------------------------------------------
IDX_ATP      = 0   # ATP_Muscle_mmol       [mmol/kg dm]
IDX_CA       = 1   # Calcium_Cytosolic_uM  [uM]
IDX_R1       = 2   # Recruitment_Type1     [0, 1]
IDX_R2       = 3   # Recruitment_Type2     [0, 1]
IDX_RYR1     = 4   # RyR1_Damage_au        [au, 0+]
IDX_GLYCOGEN = 5   # Muscle_Glycogen_mmol  [mmol/kg dm]

STATE_DIM: int = 6
OBS_DIM:   int = 2   # [EMG_Amplitude_mV, SmO2_pct]

# -- Hub input indices ----------------------------------------------------------
UIDX_POWER   = 0   # hub_Power_W              [W]
UIDX_LACTATE = 1   # hub_Lactate_mmolL        [mmol/L]
UIDX_GLUCOSE = 2   # hub_Plasma_Glucose_mgdL  [mg/dL]

CTRL_DIM: int = 3

# NaN-substitution defaults: [power=0W, lactate=1 mmol/L, glucose=90 mg/dL]
HUB_NAN_DEFAULTS: jax.Array = jnp.array([0.0, 1.0, 90.0], dtype=jnp.float32)


# -- Parameter container --------------------------------------------------------

class NMv4Params(NamedTuple):
    """
    Neuromuscular Tissue V4.0 ODE parameters -- minutes timescale.

    All rates in min^-1 unless otherwise specified.

    NLME personalises: P_th_2 (ACTN3, mu=250 W) and k_SERCA_base (mu=5.0 min^-1).
    All other parameters remain fixed at population priors.
    """
    # -- Block A: Henneman recruitment ------------------------------------------
    P_th_1:           float = 50.0    # W      Type 1 recruitment threshold
    P_th_2:           float = 250.0   # W      Type 2 threshold (NLME: ACTN3 locus)
    k_sig_rec:        float = 0.02    # W^-1   sigmoid slope
    k_fat_comp:       float = 20.0    # W/au   RyR1 -> effective power (fatigue compensation)
    k_rec:            float = 2.0     # min^-1 recruitment time constant (tau=0.5 min)

    # Hypoglycemia gate: suppresses T2 max when plasma glucose < 60 mg/dL
    glucose_hypo_thr: float = 60.0    # mg/dL
    k_hypo_T2:        float = 0.015   # (mg/dL)^-1 smooth suppression slope

    # Glycogen bonk gate: suppresses T2 max when local glycogen depleted
    Glycogen_bonk_thr: float = 15.0   # mmol/kg dm bonk threshold
    k_glycogen_gate:   float = 0.5    # (mmol/kg)^-1 sigmoid steepness

    # -- Block B: Calcium transient ---------------------------------------------
    k_Ca_release:     float = 5.0     # uM/min per unit (R1 + 2*R2)
    k_SERCA_base:     float = 5.0     # min^-1 SERCA max uptake rate (NLME param)
    k_lac_SERCA:      float = 0.10    # (mmol/L)^-1 lactate inhibition coefficient
    Ca_basal_leak:    float = 0.05    # uM/min basal SR Ca2+ leak (resting Ca ~0.1 uM)

    # -- Block C: ATP pool [V4.0 thermodynamic fix] ----------------------------
    ATP_rest:         float = 8.0     # mmol/kg dm resting ATP reference
    k_ATP_resyn:      float = 2.0     # min^-1  re-synthesis rate constant
    glucose_ref:      float = 90.0    # mg/dL   normoglycemic reference
    k_drain_T1:       float = 0.5     # mmol/(kg*min) per unit T1
    k_drain_T2:       float = 1.5     # mmol/(kg*min) per unit T2
    k_gly_ATP:        float = 0.70    # glycogen fraction of ATP resynthesis capacity
    k_glc_ATP:        float = 0.30    # plasma glucose fraction of ATP resynthesis capacity
    k_SERCA_ATP:      float = 0.01    # mmol/(kg*uM*min^-1) SERCA ATP cost coefficient

    # -- Block D: LFF / RyR1 damage --------------------------------------------
    Ca_ROS_thresh:    float = 1.0     # uM     Ca threshold for ROS generation
    k_ROS:            float = 0.05    # au/(uM*min) ROS-mediated RyR1 damage rate
    k_RyR1_repair:    float = 3e-4    # min^-1 very slow repair (tau ~55 h)

    # -- Block E: Muscle Glycogen ----------------------------------------------
    Glycogen_ref:     float = 100.0   # mmol/kg dm nominal full glycogen store
    k_gly_T1:         float = 0.5     # mmol/(kg*min) per unit T1 (oxidative; slow drain)
    k_gly_T2:         float = 2.0     # mmol/(kg*min) per unit T2 (glycolytic; fast drain)
    k_gly_resyn:      float = 0.05    # mmol/(kg*min) glycogen resynthesis at rest

    # -- Hub output ------------------------------------------------------------
    alpha_ATP:        float = 1.0     # au/(mmol/kg) fatigue per unit ATP deficit
    alpha_RyR1:       float = 2.0     # au/au        fatigue per unit RyR1 damage


DEFAULT_V4_PARAMS: NMv4Params = NMv4Params()


def build_nm_v4_params(**overrides: float) -> NMv4Params:
    """Build NMv4Params with optional keyword overrides."""
    return NMv4Params()._replace(**overrides)


_p = DEFAULT_V4_PARAMS

X0_NM_V4: jax.Array = jnp.array([
    _p.ATP_rest,       # ATP at rest [mmol/kg]
    0.10,              # Ca cytosolic resting ~0.1 uM
    0.0,               # no T1 recruitment at rest
    0.0,               # no T2 recruitment at rest
    0.0,               # no RyR1 damage at rest
    _p.Glycogen_ref,   # full glycogen store at rest [mmol/kg]
], dtype=jnp.float32)

P0_NM_V4: jax.Array = jnp.diag(jnp.array([
    1.00,    # ATP       (mmol/kg)^2  -- onboarding uncertainty
    0.25,    # Ca        (uM)^2
    0.01,    # R1        -- near zero at rest
    0.01,    # R2        -- near zero at rest
    0.25,    # RyR1      -- uncertain at onboarding
    100.0,   # Glycogen  (mmol/kg)^2 -- wide prior (pre-session nutrition varies)
], dtype=jnp.float32))


# -- Internal sigmoid ----------------------------------------------------------

def _sigmoid_rec(x: jax.Array, k: float) -> jax.Array:
    """Smooth sigmoid via tanh: 0.5*(1 + tanh(k*x*0.5)). JIT/vmap safe."""
    return jnp.float32(0.5) * (jnp.float32(1.0) + jnp.tanh(jnp.float32(k * 0.5) * x))


# -- Hub outputs ---------------------------------------------------------------

def hub_peripheral_fatigue(
    x:      jax.Array,
    params: NMv4Params = DEFAULT_V4_PARAMS,
) -> jax.Array:
    """
    Hub_Peripheral_Fatigue_au in [0, 10].

    Combines ATP deficit (acute energy failure) and RyR1 structural damage
    (chronic low-frequency fatigue).
    """
    ATP  = jnp.maximum(x[IDX_ATP],  jnp.float32(0.0))
    RyR1 = jnp.maximum(x[IDX_RYR1], jnp.float32(0.0))
    atp_def = jnp.maximum(jnp.float32(params.ATP_rest) - ATP, jnp.float32(0.0))
    fatigue  = jnp.float32(params.alpha_ATP)  * atp_def \
             + jnp.float32(params.alpha_RyR1) * RyR1
    return jnp.clip(fatigue, jnp.float32(0.0), jnp.float32(10.0))


def hub_muscle_glycogen(x: jax.Array) -> jax.Array:
    """Hub_Muscle_Glycogen_mmolkg -- floored at 0."""
    return jnp.maximum(x[IDX_GLYCOGEN], jnp.float32(0.0))


# -- Pure ODE (JIT + vmap safe) ------------------------------------------------

def nm_v4_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    Neuromuscular Tissue V4.0 ODE -- minutes timescale.

    Pure JAX: no Python conditionals on traced values.
    JIT + vmap safe. Sigma-point propagation compatible.

    Parameters
    ----------
    t    : scalar -- current time [min] (unused; autonomous ODE)
    x    : shape (STATE_DIM,) = (6,) -- state vector
    args : tuple(NMv4Params, u)
           u : shape (CTRL_DIM,) -- [hub_Power_W, hub_Lactate_mmolL, hub_Plasma_Glucose]

    Returns
    -------
    dx/dt : shape (STATE_DIM,) = (6,)
    """
    params, u = args

    # -- NaN-guard hub inputs; apply physical floors ---------------------------
    power   = jnp.maximum(jnp.nan_to_num(u[UIDX_POWER],   nan=0.0),  jnp.float32(0.0))
    lactate = jnp.maximum(jnp.nan_to_num(u[UIDX_LACTATE], nan=1.0),  jnp.float32(0.0))
    glucose = jnp.maximum(jnp.nan_to_num(u[UIDX_GLUCOSE], nan=90.0), jnp.float32(0.0))

    # -- Unpack state with physical clamping -----------------------------------
    ATP      = jnp.maximum(x[IDX_ATP],      jnp.float32(0.0))
    Ca       = jnp.maximum(x[IDX_CA],       jnp.float32(0.0))
    R1       = jnp.clip(x[IDX_R1],  jnp.float32(0.0), jnp.float32(1.0))
    R2       = jnp.clip(x[IDX_R2],  jnp.float32(0.0), jnp.float32(1.0))
    RyR1     = jnp.maximum(x[IDX_RYR1],     jnp.float32(0.0))
    Glycogen = jnp.maximum(x[IDX_GLYCOGEN], jnp.float32(0.0))

    # -- BLOCK A: Henneman Recruitment -----------------------------------------
    # Type 1: low-threshold slow fibers
    R1_target = _sigmoid_rec(power - jnp.float32(params.P_th_1), k=params.k_sig_rec)

    # Type 2: high-threshold fast fibers; RyR1_Damage increases effective power
    effective_power = power + jnp.float32(params.k_fat_comp) * RyR1
    R2_raw = _sigmoid_rec(effective_power - jnp.float32(params.P_th_2), k=params.k_sig_rec)

    # Hypoglycemia gate (plasma glucose)
    hypo_supp = jnp.float32(params.k_hypo_T2) * jnp.maximum(
        jnp.float32(params.glucose_hypo_thr) - glucose, jnp.float32(0.0)
    )
    hypoglycemia_gate = jnp.maximum(jnp.float32(1.0) - hypo_supp, jnp.float32(0.0))

    # Glycogen bonk gate -- suppresses T2 when local glycogen depletes
    glycogen_gate = _sigmoid_rec(
        Glycogen - jnp.float32(params.Glycogen_bonk_thr),
        k=params.k_glycogen_gate,
    )

    T2_max    = glycogen_gate * hypoglycemia_gate
    R2_target = R2_raw * T2_max

    # First-order recruitment dynamics (tau = 1/k_rec = 0.5 min)
    dR1_dt = jnp.float32(params.k_rec) * (R1_target - R1)
    dR2_dt = jnp.float32(params.k_rec) * (R2_target - R2)

    # -- BLOCK B: Calcium Transient --------------------------------------------
    total_rec  = R1 + jnp.float32(2.0) * R2
    Ca_release = jnp.float32(params.k_Ca_release) * total_rec \
               + jnp.float32(params.Ca_basal_leak)
    SERCA_rate = jnp.float32(params.k_SERCA_base) / (
        jnp.float32(1.0) + jnp.float32(params.k_lac_SERCA) * lactate
    )
    dCa_dt = Ca_release - SERCA_rate * Ca

    # -- BLOCK C: ATP Pool [V4.0 fix: glycogen is primary fuel] ---------------
    Gly_norm  = Glycogen / jnp.maximum(jnp.float32(params.Glycogen_ref), jnp.float32(1.0))
    glc_norm  = glucose  / jnp.maximum(jnp.float32(params.glucose_ref),  jnp.float32(1.0))
    fuel_avail = jnp.float32(params.k_gly_ATP) * Gly_norm \
               + jnp.float32(params.k_glc_ATP) * glc_norm

    atp_deficit   = jnp.maximum(jnp.float32(params.ATP_rest) - ATP, jnp.float32(0.0))
    resyn         = jnp.float32(params.k_ATP_resyn) * fuel_avail * atp_deficit
    drain_fibers  = jnp.float32(params.k_drain_T1) * R1 + jnp.float32(params.k_drain_T2) * R2
    drain_SERCA   = jnp.float32(params.k_SERCA_ATP) * SERCA_rate * Ca  # SERCA uses ATP
    dATP_dt       = resyn - drain_fibers - drain_SERCA

    # -- BLOCK D: LFF -- RyR1 damage from ROS via Ca overload -----------------
    Ca_excess = jnp.maximum(Ca - jnp.float32(params.Ca_ROS_thresh), jnp.float32(0.0))
    dRyR1_dt  = jnp.float32(params.k_ROS) * Ca_excess \
              - jnp.float32(params.k_RyR1_repair) * RyR1

    # -- BLOCK E: Muscle Glycogen (Bonking dynamics) ---------------------------
    # Saturating depletion: slows naturally as Glycogen -> 0 (no negative crossings)
    Gly_sat   = Glycogen / (Glycogen + jnp.float32(1.0))
    depletion = (jnp.float32(params.k_gly_T1) * R1 + jnp.float32(params.k_gly_T2) * R2) \
              * Gly_sat

    # Resynthesis: slow, only at rest (R1+R2 low) and glucose above reference
    glc_excess = jnp.maximum(glucose - jnp.float32(params.glucose_ref), jnp.float32(0.0)) \
               / jnp.maximum(jnp.float32(params.glucose_ref), jnp.float32(1.0))
    rest_gate  = jnp.maximum(jnp.float32(1.0) - (R1 + R2), jnp.float32(0.0))
    resyn_gly  = jnp.float32(params.k_gly_resyn) * glc_excess * rest_gate
    dGlycogen_dt = resyn_gly - depletion

    return jnp.stack([
        dATP_dt,
        dCa_dt,
        dR1_dt,
        dR2_dt,
        dRyR1_dt,
        dGlycogen_dt,
    ])
