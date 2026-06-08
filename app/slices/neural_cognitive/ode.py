"""
app/slices/neural_cognitive/ode.py  — REMASTER v2.0

Neural/Cognitive Fatigue ODE — Intra-Session Timescale (L2 Backbone)

Architecture
────────────
Models central motor drive collapse through FOUR mechanistic pathways
integrated into a 7-state hybrid system:

  1. MONOAMINERGIC HYPOTHESIS (Davis & Bailey 1997; Meeusen 2018)
     Prolonged exercise raises brain 5-HT faster than DA (BCAA/tryptophan
     competition ratio). Rising 5-HT/DA suppresses CAR.

  2. CEREBRAL HYPOXIA (Subudhi et al. 2009 Exp Physiol 94(3):330-339)
     At near-maximal intensity, hyperventilation-induced hypocapnia causes
     cerebral vasoconstriction → reduced CBF → frontal O2 saturation falls
     NON-LINEARLY (quadratic in intensity; ~15% relative drop at VO2max).
     Suppresses CAR via premotor cortex hypoxia.

  3. AMMONIA TOXICITY (Nybo et al. 2005 J Physiol 566(2):533-541;
                       Mutch & Banister 1983 Eur J Appl Physiol 51:401)
     High-intensity exercise → AMP deamination → plasma NH3 rises and crosses
     the BBB. Cerebral ammonia impairs synaptic transmission and directly
     suppresses central voluntary drive (Michaelis-Menten inhibition of CAR).

  4. ADENOSINE / CAFFEINE (Van Dongen 2003; Nehlig 2010 Neurosci Biobehav Rev)
     Sleep debt and systemic inflammation accumulate adenosine (basal forebrain).
     Caffeine is a competitive antagonist at A1/A2A receptors — Active_Adenosine
     = Adenosine_Pool / (1 + Caffeine_Plasma/K_caf). Active adenosine drives
     psychomotor vigilance fatigue (PVT lapses); it does NOT suppress CAR
     directly but degrades coordination and reaction time through separate paths.

  5. AFFERENT INHIBITION (algebraic from hub; Amann 2011) + THERMAL (Nybo 2008)
     Group III/IV afferents and T_core still contribute to CAR suppression.

TIME UNIT: HOURS (intra-session to multi-day window).

═══════════════════════════════════════════════════════════════════════════════
STATE VECTOR  x ∈ ℝ⁷   (time unit = hours)
═══════════════════════════════════════════════════════════════════════════════

  x[0]  Brain_5HT          Serotonin tone (raphe projections)             [au; 1.0=rest]
  x[1]  Brain_DA            Dopaminergic tone (mesolimbic/motor)           [au; 1.0=rest]
  x[2]  Brain_Ammonia       Cerebral NH₃ level (AMP deamination pathway)  [au; 0=rest, 1=max]
  x[3]  Cerebral_O2_Sat     Frontal cerebral oxygenation (NIRS proxy)     [fraction; 0.70=rest]
  x[4]  Adenosine_Pool      Basal forebrain adenosine accumulation         [au; 0=rested]
  x[5]  Caffeine_Plasma     Plasma caffeine concentration                  [mg/L; 0=none]
  x[6]  CAR                 Central Activation Ratio (voluntary drive)     [0, 1]

CONTROL INPUTS  u ∈ ℝ⁸  (hub variables from orchestrator)
  u[0]  hub_training_stress   Aerobic intensity [au; 0=rest, 1=maximal; Mod 1/3]
  u[1]  hub_muscle_damage     Structural muscle damage [0, 1; Mod 2 D_mus]
  u[2]  hub_T_core            Core temperature [°C; Mod 10/12 sovereign]
  u[3]  hub_sleep_debt        Accumulated sleep deficit [au; Mod 4]
  u[4]  hub_IL6               Systemic IL-6 signal [au; Mod 8]
  u[5]  hub_metabolic_stress  Metabolic acidosis load [au; 0-1; H⁺/Pi/La from Mod 1]
  u[6]  caffeine_intake_plasma  Caffeine absorption rate [mg/L/h = dose_mg/(h · V_dist_L)]
                                e.g. 200 mg over 0.5 h in 40 L → 10.0 mg/L/h
  u[7]  hub_hypoglycemia      Blood glucose deficit signal [0, 1; Mod 1]

ODE EQUATIONS  (time unit: hours)
────────────────────────────────────────────────────────────────────────────────

BLOCK A — Brain Serotonin (Davis & Bailey 1997; Chaouloff 1997)

  dBrain_5HT/dt = k_5ht_prod_basal + k_5ht_stress × hub_stress − k_5ht_clear × 5HT
  Rest SS = 1.0; max stress SS = 2.5 (k_5ht_stress=0.30 / k_5ht_clear=0.20 + 1)

BLOCK B — Brain Dopamine (Meeusen 2018; COMT Val158Met via dopamine_resilience)

  dBrain_DA/dt = k_da_prod_basal + k_da_stress × hub_stress × dopamine_resilience − k_da_clear × DA
  Rest SS = 1.0; max stress SS = 1.75 → ratio 5HT/DA diverges to 1.43 at max load.

BLOCK C — Brain Ammonia (Nybo 2005; Mutch & Banister 1983)

  dBrain_NH3/dt = k_nh3_prod × hub_metabolic_stress × (1 − NH3_c)   [logistic ceiling]
                − k_nh3_clear × NH3_c
  SS at max metabolic stress: NH3_ss = k_nh3_prod / (k_nh3_prod + k_nh3_clear) = 0.50
  Logistic ceiling (1 − NH3_c) prevents unbounded accumulation.
  τ_NH3 ≈ 1.0 h.

BLOCK D — Cerebral O₂ Saturation (Subudhi et al. 2009 — frontal NIRS)

  O2_target = O2_rest × (1 − delta_O2_max × hub_stress²)
  dO2Sat/dt = k_o2_relax × (O2_target − O2Sat_c)
  Rest SS = O2_rest = 0.70; max intensity SS = 0.70 × 0.85 = 0.595 (15% relative drop)
  Quadratic in hub_stress: drop is small at VT1, large near VO2max.
  τ_O2 = 1/k_o2_relax = 0.5 h (rapid equilibration with CBF).

BLOCK E — Adenosine Pool (Van Dongen 2003; Mullington 2010)

  dAdenosine_Pool/dt = k_aden_sleep × hub_sleep_debt + k_aden_il6 × hub_IL6
                      − k_aden_rec × Adenosine_Pool
  Adenosine accumulates under sleep debt and inflammation.
  τ_adenosine = 1/k_aden_rec = 10 h (adenosine washout matching sleep homeostat).

BLOCK F — Caffeine Plasma (Nehlig 2010; CYP1A2 kinetics)

  dCaffeine_Plasma/dt = caffeine_intake_plasma − CYP1A2_clearance_rate × Caffeine_Plasma
  CYP1A2_clearance_rate = ln(2)/t½ where t½ = 5 h (population mean).
  NLME-personalised: t½ ranges 2.5 h (fast metaboliser) to 10 h (slow metaboliser).
  caffeine_intake_plasma [mg/L/h] already accounts for volume of distribution (~40 L).

ALGEBRAIC — Active Adenosine (competitive A1/A2A receptor antagonism; Nehlig 2010)

  Active_Adenosine = Adenosine_Pool / (1.0 + Caffeine_Plasma / K_caf)
  At K_caf = 3.0 mg/L: caffeine plasma of 5 mg/L reduces adenosine signalling by 63%.

ALGEBRAIC — Afferent Inhibition (Amann 2011 — Group III/IV afferents)

  aff_from_dmg  = k_aff_dmg × hub_muscle_damage / (K_aff_dmg + hub_muscle_damage)
  aff_from_glc  = k_aff_glc × hub_hypoglycemia
  Afferent_total = clamp(aff_from_dmg + aff_from_glc, 0, 1)

BLOCK G — Central Activation Ratio (5-pathway suppression; Nybo 2008; Amann 2011)

  ratio_5HT_DA   = Brain_5HT / max(Brain_DA, ε)
  thermal_inhib  = max(0, T_core − 38.5)
  hypoxia_inhib  = max(0, (O2_rest − O2Sat) / O2_rest)
  ammonia_inhib  = NH3 / (K_nh3 + NH3)
  afferent_inhib = Afferent_total

  dCAR/dt = k_car_rec × (1 − CAR)
           − k_car_5htda   × ratio_5HT_DA   × CAR    [monoaminergic]
           − k_car_aff     × afferent_inhib × CAR    [afferent]
           − k_car_thermal × thermal_inhib  × CAR    [thermal; Nybo 2008]
           − k_car_hypoxia × hypoxia_inhib  × CAR    [cerebral hypoxia; Subudhi 2009]
           − k_car_ammonia × ammonia_inhib  × CAR    [ammonia toxicity; Nybo 2005]

  Rest SS (all suppressors zero, ratio=1.0):
    CAR_ss = k_car_rec / (k_car_rec + k_car_5htda × 1.0) = 2.0/2.105 ≈ 0.950

  Extreme stress (max stress, max metabolic, T=37 °C, no muscle damage):
    After 6 h: ratio≈1.34, NH3≈0.499, hypoxia≈0.15, thermal=0, afferent=0
    CAR_ss ≈ 2.0 / (2.0 + 0.141 + 0.300 + 0.749) = 2.0/3.19 ≈ 0.627 < 0.75 hard floor

HUB EXPORTS (algebraic)
  Hub_NC_CAR            [0,1]     → Mod 2 (neuromuscular recruitment gate)
  Hub_NC_RPE            [1–10]    → orchestrator training load
  Hub_NC_Active_Adenosine [au]    → observation layer (PVT lapses)
  Hub_NC_PVT            [au]      → raw adenosine pool (monitoring only)

References
──────────
  Amann M. et al. (2011) J Physiol 589(10):2467–2476       [afferent inhibition]
  Chaouloff F. (1997) Acta Physiol Scand 161:1–14          [monoamines exercise]
  Davis J.M., Bailey S.P. (1997) Med Sci Sports Exerc 29(1):45–57 [5HT fatigue]
  Meeusen R. et al. (2018) Exerc Sport Sci Rev 46(1):1–5   [central fatigue review]
  Mullington J.M. et al. (2010) Ann NY Acad Sci 1213:48–56 [IL-6 adenosine crosstalk]
  Mutch B.J., Banister E.W. (1983) Eur J Appl Physiol 51:401–412 [NH3 fatigue]
  Nehlig A. (2010) Neurosci Biobehav Rev 35(2):430–440     [caffeine A1/A2A antagonism]
  Nybo L. (2008) J Physiol 586(1):83–95                    [thermal CAR suppression]
  Nybo L. et al. (2005) J Physiol 566(2):533–541           [NH3 & cerebral fatigue]
  Subudhi A.W. et al. (2009) Exp Physiol 94(3):330–339     [cerebral O2 & fatigue]
  Van Dongen H.P.A. et al. (2003) Sleep 26(2):117–126      [PVT & adenosine]
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── State vector indices ──────────────────────────────────────────────────────
IDX_5HT   = 0   # Brain serotonin tone                  [au; 1.0=rest]
IDX_DA    = 1   # Brain dopamine tone                   [au; 1.0=rest]
IDX_NH3   = 2   # Brain ammonia (AMP deamination)       [au; 0=rest, 1=max]
IDX_O2SAT = 3   # Cerebral O2 saturation (NIRS proxy)  [fraction; 0.70=rest]
IDX_ADEN  = 4   # Adenosine pool (sleep homeostat)      [au; 0=rested]
IDX_CAF   = 5   # Caffeine plasma (CYP1A2 clearance)    [mg/L; 0=none]
IDX_CAR   = 6   # Central Activation Ratio              [0, 1]

STATE_DIM: int = 7
OBS_DIM:   int = 2   # [RPE_Proxy [1–10], PVT_Lapses [count/10 min]]

# ── Control input indices ─────────────────────────────────────────────────────
UIDX_TRAINING_STRESS  = 0   # aerobic intensity [au; 0–1]
UIDX_MUSCLE_DAMAGE    = 1   # structural muscle damage [0, 1]
UIDX_T_CORE           = 2   # core temperature [°C]
UIDX_SLEEP_DEBT       = 3   # accumulated sleep deficit [au]
UIDX_IL6              = 4   # systemic IL-6 [au]
UIDX_METABOLIC_STRESS = 5   # metabolic acidosis stress [au; 0–1; H⁺/Pi/La]
UIDX_CAFFEINE_INTAKE  = 6   # caffeine absorption rate [mg/L/h]
UIDX_HYPOGLYCEMIA     = 7   # blood glucose deficit signal [0, 1]

CTRL_DIM: int = 8

# Numerical guard
_EPS: jax.Array = jnp.float32(1e-4)
_O2_REST_CONST: float = 0.70   # canonical resting frontal O2 saturation


# ── Parameter container ───────────────────────────────────────────────────────

class NeuralCognitiveParams(NamedTuple):
    """
    Neural/Cognitive ODE parameters (intra-session, hours).

    Calibration sources (see module docstring):
    ─ Davis & Bailey 1997  — 5-HT kinetics
    ─ Meeusen 2018         — monoamine ratio & fatigue
    ─ Nybo 2005            — NH3 & cerebral fatigue
    ─ Subudhi 2009         — cerebral O2 & fatigue
    ─ Van Dongen 2003      — PVT & adenosine
    ─ Nehlig 2010          — caffeine A1/A2A antagonism
    ─ Amann 2011           — afferent inhibition
    ─ Nybo 2008            — thermal CAR suppression
    """

    # ── BLOCK A: Brain Serotonin ──────────────────────────────────────────────
    k_5ht_prod_basal: float = 0.20   # h⁻¹ — basal synthesis; SS_rest = basal/clear = 1.0
    k_5ht_clear:      float = 0.20   # h⁻¹ — MAO-A/SERT clearance; t½ ≈ 3.5 h
    k_5ht_stress:     float = 0.30   # au·au⁻¹ — exercise-driven production gain

    # ── BLOCK B: Brain Dopamine ───────────────────────────────────────────────
    k_da_prod_basal:     float = 0.20   # h⁻¹ — basal TH pathway synthesis
    k_da_clear:          float = 0.20   # h⁻¹ — COMT/MAO-B clearance; t½ ≈ 3.5 h
    k_da_stress:         float = 0.15   # au·au⁻¹ — DA gain (< k_5ht_stress → ratio diverges)
    dopamine_resilience: float = 1.0    # au — NLME-personalised; COMT Val158Met prior

    # ── BLOCK C: Brain Ammonia ────────────────────────────────────────────────
    # AMP deamination: AMP → IMP + NH₃ during high-intensity exercise.
    # NH₃ crosses BBB and impairs glutamate/GABA neurotransmission.
    k_nh3_prod:  float = 0.50   # h⁻¹ — NH3 production rate under metabolic stress
                                 # logistic gate (1-NH3) caps SS at 0.5 at max stress
    k_nh3_clear: float = 0.50   # h⁻¹ — hepatic/renal clearance; t½ ≈ 1.4 h
    K_nh3:       float = 0.50   # au — Michaelis-Menten half-saturation for CAR inhibition

    # ── BLOCK D: Cerebral O₂ Saturation ──────────────────────────────────────
    # Subudhi 2009: frontal oxygenation falls ~15% relative from rest to VO2max
    # via hyperventilation → hypocapnia → cerebral vasoconstriction.
    O2_rest:      float = 0.70   # fraction — resting frontal cerebral oxygenation
    delta_O2_max: float = 0.15   # fraction — max relative O2 drop at hub_stress = 1.0
    k_o2_relax:   float = 2.0    # h⁻¹ — O2 equilibration rate; τ ≈ 30 min

    # ── BLOCK E: Adenosine Pool ───────────────────────────────────────────────
    k_aden_sleep: float = 0.10   # h⁻¹ — sleep debt → adenosine production
    k_aden_il6:   float = 0.05   # h⁻¹ — IL-6 → adenosine (cytokine crosstalk)
    k_aden_rec:   float = 0.10   # h⁻¹ — adenosine washout; t½ ≈ 7 h

    # ── BLOCK F: Caffeine Plasma ──────────────────────────────────────────────
    # CYP1A2_clearance_rate = ln(2) / t½_caffeine; population mean t½ = 5 h.
    # NLME-personalised: fast metabolisers (CYP1A2*1F): t½ ≈ 2.5 h (k=0.277)
    #                     slow metabolisers (CYP1A2*1C): t½ ≈ 10 h (k=0.069)
    CYP1A2_clearance_rate: float = 0.139   # h⁻¹ — population mean ln(2)/5h
    K_caf:                 float = 3.0     # mg/L — A1/A2A competitive IC50 (Nehlig 2010)
                                           # ~10–30 μmol/L caffeine → ~1.9–5.7 mg/L (MW=194)

    # ── Algebraic: Afferent Inhibition ───────────────────────────────────────
    k_aff_dmg:  float = 2.0    # h⁻¹ — Michaelis-Menten damage→afferent gain
    K_aff_dmg:  float = 0.5    # au — half-saturation for muscle damage signal
    k_aff_glc:  float = 0.30   # h⁻¹ — hypoglycemia→afferent gain

    # ── BLOCK G: Central Activation Ratio ────────────────────────────────────
    k_car_rec:        float = 2.0    # h⁻¹ — CAR recovery; τ ≈ 30 min
    k_car_5htda:      float = 0.105  # au — monoaminergic suppression coefficient
    k_car_aff:        float = 1.50   # au — afferent suppression coefficient
    k_car_thermal:    float = 0.50   # au·°C⁻¹ — thermal suppression per °C above 38.5
    T_core_threshold: float = 38.5   # °C — thermal inhibition onset (Nybo 2008)
    k_car_hypoxia:    float = 2.00   # au — cerebral hypoxia suppression (Subudhi 2009)
    k_car_ammonia:    float = 1.50   # au — ammonia toxicity suppression (Nybo 2005)


def build_nc_params(**overrides: float) -> NeuralCognitiveParams:
    """Build NeuralCognitiveParams with keyword overrides applied."""
    return NeuralCognitiveParams()._replace(**overrides)


DEFAULT_NC_PARAMS: NeuralCognitiveParams = build_nc_params()


# ── Steady-state initial conditions (well-rested athlete) ────────────────────
# Brain_5HT = 1.0   (k_5ht_prod_basal / k_5ht_clear = 1.0)
# Brain_DA  = 1.0   (same)
# Brain_NH3 = 0.0   (no metabolic stress at rest)
# O2_Sat    = 0.70  (O2_rest; rest level)
# Adenosine = 0.0   (fully rested; no sleep debt)
# Caffeine  = 0.0   (no caffeine)
# CAR       = 0.95  (k_car_rec / (k_car_rec + k_car_5htda × 1.0) ≈ 0.950)

X0_NC_DEFAULT: jax.Array = jnp.array([
    1.00,   # Brain_5HT        [au]
    1.00,   # Brain_DA         [au]
    0.00,   # Brain_NH3        [au]
    0.70,   # Cerebral_O2_Sat  [fraction]
    0.00,   # Adenosine_Pool   [au]
    0.00,   # Caffeine_Plasma  [mg/L]
    0.95,   # CAR              [0,1]
], dtype=jnp.float32)

# Initial covariance (diagonal; onboarding uncertainty pre-session)
P0_NC_DEFAULT: jax.Array = jnp.diag(jnp.array([
    0.0900,   # 5HT   σ² = (0.30)²  — individual tone variability
    0.0900,   # DA    σ² = (0.30)²  — COMT-driven variability
    0.0010,   # NH3   σ² = (0.032)² — tight near 0 at rest
    0.0009,   # O2    σ² = (0.030)² — tight near 0.70
    0.0400,   # Aden  σ² = (0.20)²  — uncertain at onboarding (unknown sleep history)
    0.0100,   # Caf   σ² = (0.10)²  — low at onboarding
    0.0025,   # CAR   σ² = (0.05)²  — tight near 0.95 at rest
], dtype=jnp.float32))


# ── Pure ODE (JIT + vmap safe) ────────────────────────────────────────────────

def neural_cognitive_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    Neural/Cognitive ODE — intra-session timescale (hours).

    Pure JAX — no Python conditionals on traced values.  JIT + vmap safe.

    Parameters
    ----------
    t    : scalar — current time [h]
    x    : shape (STATE_DIM,) = (7,)
    args : tuple(NeuralCognitiveParams, u)
           u : shape (CTRL_DIM,) = (8,)
    Returns
    -------
    dx/dt : shape (STATE_DIM,)
    """
    params, u = args

    # ── Unpack state ──────────────────────────────────────────────────────────
    Brain_5HT = x[IDX_5HT]
    Brain_DA  = x[IDX_DA]
    Brain_NH3 = x[IDX_NH3]
    O2_Sat    = x[IDX_O2SAT]
    Adenosine = x[IDX_ADEN]
    Caffeine  = x[IDX_CAF]
    CAR       = x[IDX_CAR]

    # ── Hub guards: NaN → safe physiological defaults ────────────────────────
    hub_stress = jnp.where(
        jnp.isnan(u[UIDX_TRAINING_STRESS]),
        jnp.float32(0.0),
        jnp.clip(u[UIDX_TRAINING_STRESS], jnp.float32(0.0), jnp.float32(1.0)),
    )
    hub_dmg = jnp.where(
        jnp.isnan(u[UIDX_MUSCLE_DAMAGE]),
        jnp.float32(0.0),
        jnp.clip(u[UIDX_MUSCLE_DAMAGE], jnp.float32(0.0), jnp.float32(1.0)),
    )
    hub_Tc = jnp.where(
        jnp.isnan(u[UIDX_T_CORE]),
        jnp.float32(37.0),
        u[UIDX_T_CORE],
    )
    hub_sleep = jnp.where(
        jnp.isnan(u[UIDX_SLEEP_DEBT]),
        jnp.float32(0.0),
        jnp.maximum(u[UIDX_SLEEP_DEBT], jnp.float32(0.0)),
    )
    hub_il6 = jnp.where(
        jnp.isnan(u[UIDX_IL6]),
        jnp.float32(0.0),
        jnp.maximum(u[UIDX_IL6], jnp.float32(0.0)),
    )
    hub_metab = jnp.where(
        jnp.isnan(u[UIDX_METABOLIC_STRESS]),
        jnp.float32(0.0),
        jnp.clip(u[UIDX_METABOLIC_STRESS], jnp.float32(0.0), jnp.float32(1.0)),
    )
    hub_caf = jnp.where(
        jnp.isnan(u[UIDX_CAFFEINE_INTAKE]),
        jnp.float32(0.0),
        jnp.maximum(u[UIDX_CAFFEINE_INTAKE], jnp.float32(0.0)),
    )
    hub_glc = jnp.where(
        jnp.isnan(u[UIDX_HYPOGLYCEMIA]),
        jnp.float32(0.0),
        jnp.clip(u[UIDX_HYPOGLYCEMIA], jnp.float32(0.0), jnp.float32(1.0)),
    )

    # ── State clamping (thermodynamic / biological floors) ────────────────────
    B5HT_pos  = jnp.maximum(Brain_5HT, _EPS)        # positive concentration
    BDA_pos   = jnp.maximum(Brain_DA,  _EPS)        # avoids 0/0 in ratio
    NH3_c     = jnp.clip(Brain_NH3,  jnp.float32(0.0), jnp.float32(1.0))
    O2_c      = jnp.clip(O2_Sat,     jnp.float32(0.0), jnp.float32(1.0))
    Aden_pos  = jnp.maximum(Adenosine,  jnp.float32(0.0))
    Caf_pos   = jnp.maximum(Caffeine,   jnp.float32(0.0))
    CAR_c     = jnp.clip(CAR, jnp.float32(0.0), jnp.float32(1.0))

    # ── BLOCK A: Brain Serotonin ──────────────────────────────────────────────
    # Free-tryptophan crosses BBB ↑ during exercise (BCAA competition).
    d5HT_dt = (
        jnp.float32(params.k_5ht_prod_basal)
        + jnp.float32(params.k_5ht_stress) * hub_stress
        - jnp.float32(params.k_5ht_clear)  * B5HT_pos
    )

    # ── BLOCK B: Brain Dopamine ───────────────────────────────────────────────
    # dopamine_resilience is NLME-personalised via COMT Val158Met.
    dDA_dt = (
        jnp.float32(params.k_da_prod_basal)
        + jnp.float32(params.k_da_stress) * hub_stress * jnp.float32(params.dopamine_resilience)
        - jnp.float32(params.k_da_clear)  * BDA_pos
    )

    # ── BLOCK C: Brain Ammonia ────────────────────────────────────────────────
    # Logistic ceiling (1 − NH3_c): prevents NH3 exceeding [0,1] bound.
    # SS at max metabolic stress: 0.5 × (k_nh3_prod / (k_nh3_prod + k_nh3_clear))
    dNH3_dt = (
        jnp.float32(params.k_nh3_prod)
        * hub_metab
        * (jnp.float32(1.0) - NH3_c)        # logistic ceiling
        - jnp.float32(params.k_nh3_clear) * NH3_c
    )

    # ── BLOCK D: Cerebral O₂ Saturation ──────────────────────────────────────
    # O2_target falls quadratically with intensity (hyperventilation non-linearity).
    O2_target = jnp.float32(params.O2_rest) * (
        jnp.float32(1.0)
        - jnp.float32(params.delta_O2_max) * hub_stress * hub_stress
    )
    dO2Sat_dt = jnp.float32(params.k_o2_relax) * (O2_target - O2_c)

    # ── BLOCK E: Adenosine Pool ───────────────────────────────────────────────
    dAden_dt = (
        jnp.float32(params.k_aden_sleep) * hub_sleep
        + jnp.float32(params.k_aden_il6)  * hub_il6
        - jnp.float32(params.k_aden_rec)  * Aden_pos
    )

    # ── BLOCK F: Caffeine Plasma (first-order CYP1A2 clearance) ──────────────
    # caffeine_intake_plasma already in [mg/L/h] (dose/V_dist).
    dCaf_dt = hub_caf - jnp.float32(params.CYP1A2_clearance_rate) * Caf_pos

    # ── ALGEBRAIC: Active Adenosine (competitive A1/A2A antagonism) ──────────
    # Caffeine displaces adenosine at A1/A2A receptors (Nehlig 2010).
    Active_Aden = Aden_pos / (
        jnp.float32(1.0) + Caf_pos / (jnp.float32(params.K_caf) + _EPS)
    )

    # ── ALGEBRAIC: Afferent Inhibition (Group III/IV; Amann 2011) ────────────
    # Michaelis-Menten for muscle damage → metabolite-sensitive afferents.
    aff_from_dmg = (
        jnp.float32(params.k_aff_dmg)
        * hub_dmg
        / (jnp.float32(params.K_aff_dmg) + hub_dmg + _EPS)
    )
    aff_from_glc  = jnp.float32(params.k_aff_glc) * hub_glc
    Afferent_total = jnp.clip(
        aff_from_dmg + aff_from_glc,
        jnp.float32(0.0),
        jnp.float32(1.0),
    )

    # ── BLOCK G: Central Activation Ratio (5-pathway suppression) ────────────
    ratio_5HT_DA  = B5HT_pos / BDA_pos                       # monoamine ratio
    thermal_inhib = jnp.maximum(
        hub_Tc - jnp.float32(params.T_core_threshold),
        jnp.float32(0.0),
    )
    # Fractional O2 loss relative to resting level (Subudhi 2009)
    hypoxia_inhib = jnp.maximum(
        (jnp.float32(params.O2_rest) - O2_c) / (jnp.float32(params.O2_rest) + _EPS),
        jnp.float32(0.0),
    )
    # Michaelis-Menten NH3 → CAR inhibition (Nybo 2005)
    ammonia_inhib = NH3_c / (jnp.float32(params.K_nh3) + NH3_c + _EPS)

    car_recovery = jnp.float32(params.k_car_rec) * (jnp.float32(1.0) - CAR_c)
    car_suppress  = (
        jnp.float32(params.k_car_5htda)   * ratio_5HT_DA   * CAR_c
        + jnp.float32(params.k_car_aff)   * Afferent_total * CAR_c
        + jnp.float32(params.k_car_thermal) * thermal_inhib * CAR_c
        + jnp.float32(params.k_car_hypoxia) * hypoxia_inhib * CAR_c
        + jnp.float32(params.k_car_ammonia) * ammonia_inhib * CAR_c
    )
    dCAR_dt = car_recovery - car_suppress

    return jnp.stack([
        d5HT_dt, dDA_dt, dNH3_dt, dO2Sat_dt,
        dAden_dt, dCaf_dt, dCAR_dt,
    ])


# ── Algebraic Active Adenosine (exported separately for observation model) ────

def compute_active_adenosine(x: jax.Array, K_caf: float = 3.0) -> jax.Array:
    """
    Competitive antagonism: Active_Adenosine = Aden / (1 + Caf / K_caf).

    JIT + vmap safe.

    Parameters
    ----------
    x     : shape (STATE_DIM,) = (7,)
    K_caf : float [mg/L] — caffeine-adenosine IC50 (population: 3.0 mg/L)

    Returns
    -------
    active_aden : scalar — effective adenosine receptor occupancy [au]
    """
    Aden_pos = jnp.maximum(x[IDX_ADEN], jnp.float32(0.0))
    Caf_pos  = jnp.maximum(x[IDX_CAF],  jnp.float32(0.0))
    return Aden_pos / (jnp.float32(1.0) + Caf_pos / (jnp.float32(K_caf) + _EPS))


# ── Hub export (algebraic, for orchestrator) ──────────────────────────────────

@jax.jit
def compute_nc_hub_exports(
    x:      jax.Array,
    params: NeuralCognitiveParams = DEFAULT_NC_PARAMS,
) -> dict[str, jax.Array]:
    """
    Compute neural/cognitive hub export variables from current state.

    Returns
    -------
    dict:
        Hub_NC_CAR                [0,1]     — voluntary motor drive
        Hub_NC_RPE                [1–10]    — perceived exertion proxy
        Hub_NC_Active_Adenosine   [au]      — effective adenosine signalling
        Hub_NC_PVT                [au]      — raw adenosine pool (monitoring)
    """
    CAR_out    = jnp.clip(x[IDX_CAR], jnp.float32(0.0), jnp.float32(1.0))
    RPE_out    = jnp.clip(
        jnp.float32(1.0) + jnp.float32(9.0) * (jnp.float32(1.0) - CAR_out),
        jnp.float32(1.0), jnp.float32(10.0),
    )
    active_aden = compute_active_adenosine(x, params.K_caf)
    return {
        "Hub_NC_CAR":               CAR_out,
        "Hub_NC_RPE":               RPE_out,
        "Hub_NC_Active_Adenosine":  active_aden,
        "Hub_NC_PVT":               jnp.maximum(x[IDX_ADEN], jnp.float32(0.0)),
    }
