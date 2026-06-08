"""
app/slices/cardiorespiratory/ode.py

L2 Backbone ODE — Cardiorespiratory Performance Slice

Models the 6 state variables most relevant for training prescription:
VO2 kinetics, cardiac response, anaerobic battery, respiratory steal,
and autonomic tone. Operates at the prescription timescale (1-min UKF
steps, daily NLME updates), complementing the session-level hemodynamic
detail in engine/solvers/cardiorespiratory_solver.py.

══════════════════════════════════════════════════════════════════════
STATE VECTOR  x ∈ ℝ⁶
══════════════════════════════════════════════════════════════════════

  x[0]  V_O2          oxygen uptake                   [mL/kg/min]
  x[1]  Heart_Rate    effective heart rate             [bpm]
  x[2]  Stroke_Volume stroke volume (Frank-Starling)   [L]
  x[3]  W_prime_bal   anaerobic battery balance        [kJ]
  x[4]  Resp_Fatigue  respiratory muscle fatigue       [adim, 0-1]
  x[5]  Autonomic_Tone vagal / parasympathetic tone    [adim, 0-1]

CONTROL INPUTS  (ODE args)
  power_watts    [W]   — session power demand
  hub_T_core     [°C]  — core temperature (Mod 10 hub); NaN if unavailable
  hub_pv_drop_pct [%]  — plasma volume drop (Mod 12 hub); NaN if unavailable

ODE EQUATIONS
─────────────

I. VO2 kinetics (Boillet, Messonnier & Cohen 2024, Sci Rep):
   VO2_ss  = VO2_rest + P_norm × (VO2_max − VO2_rest) × gain_VO2
   SC      = k_slow × max(0, P − CP) / P_ref          [slow component above CP]
   dVO2/dt = (VO2_ss − VO2) / τ_VO2 + SC

II. Heart rate — Fick coupling + CV drift (Coyle & Gonzalez-Alonso 2001):
   VO2_frac  = clip((VO2 − VO2_rest) / (VO2_max − VO2_rest), 0, 1.2)
   HR_Fick   = HR_basal + VO2_frac × (HR_max − HR_basal)
   HR_drift  = k_HR_drift × (1 − SV / SV_ref)          [Frank-Starling compensation]
   HR_Tcore  = k_HR_Tcore × max(0, T_core − 37)        [thermal tachycardia]
   dHR/dt    = (max(HR_floor, HR_Fick + HR_drift + HR_Tcore) − HR) / τ_HR

III. Stroke volume — Frank-Starling + CV drift + respiratory steal (Dempsey 2006):
   drift_factor = max(drift_floor, 1 − k_PV_SV × pv_drop / 100)
   resp_steal   = max(0.5, 1 − k_resp_SV × Resp_Fatigue)
   dSV/dt       = (SV_ref × drift_factor × resp_steal − SV) / τ_SV

IV. W' balance — Clarke-Skiba 2013 smooth form:
   drain    = max(0, P − CP) × 0.06          [kJ/min; 0.06 = 60 s/min × 1e-3 kJ/J]
   rec      = (CP − P) / P_ref × (W'_cap − W'_bal) / τ_W_rec
   dW'/dt   = max(0, rec) − drain

V. Respiratory fatigue — Dempsey metaboreflex 2006 (J Physiol 564:425-445):
   RF_target = sigmoid(k_RF_gain × (P/P_ref − RF_threshold))
   dRF/dt    = (RF_target − RF) / τ_RF

VI. Autonomic tone — HRV proxy (Arai 1989 + Kontro 2026):
   w_dep  = max(0, 1 − W'_bal / W'_cap)      [W' depletion fraction]
   dAT/dt = k_AT_rec × (1 − AT) − k_AT_sup × P_norm × AT − k_AT_W × w_dep × AT

HUB COUPLINGS (NaN-guarded)
────────────────────────────
  Hub Inbound:
    hub_T_core     (Mod 10 / thermo_renal) → HR drift [Coyle 2001]
    hub_pv_drop_pct (Mod 12 / renal)       → SV drift  [Frank-Starling]
  Hub Outbound:
    Hub_VO2_mLkgmin   → Mod 1 (metabolic energy flux)
    Hub_HR_bpm        → Mod 4 (sleep/circadian load)
    Hub_AT            → Mod 9 (cognitive readiness via RMSSD proxy)

DESIGN INVARIANTS
─────────────────
  • Pure JAX: no Python conditionals on traced values.
  • jnp.maximum / jnp.where for all smooth gates.
  • JIT + vmap safe (sigma-point propagation).
  • Fail-Loud: NaN propagates — no silent substitution.
  • Hub NaN guards via jnp.where(jnp.isnan(.)) before any use.

References
──────────
  Boillet L., Messonnier L.A., Cohen J. (2024) Sci Rep 14:5050
    DOI 10.1038/s41598-024-56042-0  [bioenergetic ODE model]
  Clarke D.C. & Skiba P.F. (2013) Am J Physiol Regul 305:R100-R111
    DOI 10.1152/ajpregu.00444.2012  [W' balance model]
  Coyle E.F. & Gonzalez-Alonso J. (2001) J Appl Physiol 90:1389-1395
    DOI 10.1152/jappl.2001.90.4.1389  [cardiovascular drift]
  Dempsey J.A. et al. (2006) J Physiol 564:425-445
    DOI 10.1113/jphysiol.2005.100537  [respiratory muscle steal]
  Arai Y. et al. (1989) Circulation 79:86-93
    DOI 10.1161/01.cir.79.1.86  [vagal recovery kinetics]
  Kontro H. et al. (2026) PLOS ONE DOI 10.1371/journal.pone.0341721
    [W' depletion → autonomic withdrawal channel]
  Malik M. et al. (1996) Eur Heart J 17:354-381 [RMSSD population reference]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── State vector indices ──────────────────────────────────────────────────────
IDX_VO2    = 0   # V_O2          [mL/kg/min]
IDX_HR     = 1   # Heart_Rate    [bpm]
IDX_SV     = 2   # Stroke_Volume [L]
IDX_WPRIME = 3   # W_prime_bal   [kJ]
IDX_RF     = 4   # Resp_Fatigue  [adim, 0-1]
IDX_AT     = 5   # Autonomic_Tone [adim, 0-1]

STATE_DIM: int = 6
OBS_DIM:   int = 3   # [HR_obs_bpm, VO2_obs_mLkgmin, RMSSD_obs_ms]


# ── Parameter container ───────────────────────────────────────────────────────

class CardioSliceParams(NamedTuple):
    """
    Parameter set for the 6-state cardiorespiratory performance ODE.

    All population-mean values anchored to cited literature.
    Personalised per-subject via NLME (VO2_max_baseline, W_prime_capacity).

    Time unit: minutes throughout (rates in min⁻¹, time constants in min).
    """
    # ── I. VO2 kinetics (Boillet 2024) ───────────────────────────────────
    VO2_max_baseline: float  # mL/kg/min — peak aerobic capacity (NLME prior)
    VO2_rest:         float  # mL/kg/min — resting metabolic rate (~3.5 MET×1)
    tau_VO2:          float  # min — VO2 primary component time constant
    k_slow:           float  # adim — slow component amplitude (fraction of VO2max)
    gain_VO2:         float  # adim — VO2/VO2max achieved at P_ref

    # ── II. Cardiac dynamics (Coyle 2001 CV drift) ───────────────────────
    HR_basal:      float  # bpm — resting HR at full vagal tone
    HR_max:        float  # bpm — age-predicted maximal HR
    tau_HR:        float  # min — HR response time constant
    k_HR_drift:    float  # bpm — max HR compensation for full SV loss
    k_HR_Tcore:    float  # bpm/°C — HR rise per degree above 37°C [Coyle 2001]

    # ── III. Stroke volume (Frank-Starling + CV drift) ────────────────────
    SV_ref:        float  # L — reference stroke volume at rest (~83 mL)
    tau_SV:        float  # min — SV adjustment time constant
    k_PV_SV:       float  # adim — PV drop fraction → SV fractional reduction
    drift_floor:   float  # adim — minimum SV as fraction of SV_ref
    k_resp_SV:     float  # adim — Resp_Fatigue → SV redistribution (steal)
    PV_drop_ceil:  float  # % — physical cap on plasma-volume-drop signal

    # ── IV. W' balance (Clarke-Skiba 2013) ───────────────────────────────
    W_prime_capacity: float  # kJ — anaerobic battery capacity (NLME prior)
    tau_W_rec:        float  # min — W' recovery time constant
    CP_watts:         float  # W — critical power
    P_ref:            float  # W — normalisation power (~power at VO2max)

    # ── V. Respiratory fatigue (Dempsey 2006 metaboreflex) ────────────────
    tau_RF:        float  # min — resp. fatigue response time constant
    k_RF_gain:     float  # adim — sigmoid steepness for metaboreflex onset
    RF_threshold:  float  # adim — P/P_ref threshold for metaboreflex activation

    # ── VI. Autonomic tone (HRV proxy) ────────────────────────────────────
    k_AT_rec:      float  # min⁻¹ — vagal tone recovery rate (Arai 1989)
    k_AT_sup:      float  # adim — exercise suppression of vagal tone
    k_AT_W:        float  # adim — W' depletion → vagal withdrawal (Kontro 2026)
    RMSSD_ref_ms:  float  # ms — RMSSD at full vagal tone (Malik 1996)

    # ── Numerical guards ─────────────────────────────────────────────────
    HR_floor:      float  # bpm — absolute physiological HR floor


# ── Population-mean default parameters (literature anchored) ─────────────────

DEFAULT_CARDIO_SLICE_PARAMS = CardioSliceParams(
    # I. VO2 kinetics
    VO2_max_baseline = 45.0,   # mL/kg/min — mean male endurance athlete
    VO2_rest         = 3.5,    # mL/kg/min — 1 MET (standard resting)
    tau_VO2          = 0.75,   # min = 45 s (Whipp & Wasserman 1972; Boillet 2024)
    k_slow           = 0.12,   # adim — slow component ~12% of VO2max (Gaesser 1984)
    gain_VO2         = 0.90,   # adim — VO2 at P_ref = 90% of VO2max

    # II. Cardiac (Coyle 2001)
    HR_basal     = 60.0,    # bpm — resting HR (Berntson 1994 population)
    HR_max       = 185.0,   # bpm — age 35 estimate: 220 − 35 = 185
    tau_HR       = 1.5,     # min — HR response τ (Ekblom 1968)
    k_HR_drift   = 20.0,    # bpm — drift amplitude (Coyle 2001: 15–25 bpm over 60 min)
    k_HR_Tcore   = 8.0,     # bpm/°C — cardiac thermal response (Coyle 2001)

    # III. Stroke volume
    SV_ref       = 0.083,   # L = 83 mL — resting SV (Scharhag 2002 SportsMed)
    tau_SV       = 2.0,     # min — SV adjustment kinetics
    k_PV_SV      = 1.5,     # adim — 8% PV drop → 12% SV reduction (Coyle 1986)
    drift_floor  = 0.50,    # adim — minimum SV fraction (extreme drift)
    k_resp_SV    = 0.15,    # adim — steal fraction at RF=1 (Dempsey 2006)
    PV_drop_ceil = 20.0,    # % — physical cap for safety

    # IV. W' balance (Clarke-Skiba 2013)
    W_prime_capacity = 18.0,   # kJ — population mean (Skiba 2015 J Sci Sport)
    tau_W_rec        = 6.3,    # min = 377 s (Clarke & Skiba 2013)
    CP_watts         = 250.0,  # W — population prior (Burnley & Jones 2018)
    P_ref            = 350.0,  # W — power at VO2max (45 mL/kg/min × 70 kg / Δa-vO2)

    # V. Respiratory fatigue (Dempsey 2006)
    tau_RF       = 3.0,    # min — respiratory muscle fatigue kinetics
    k_RF_gain    = 8.0,    # adim — sigmoid steepness
    RF_threshold = 0.85,   # adim — ~85% P_ref onset (Dempsey 2006 threshold ~80-90%)

    # VI. Autonomic tone
    k_AT_rec    = 0.050,   # min⁻¹ → τ = 20 min (Arai 1989)
    k_AT_sup    = 0.20,    # adim — exercise suppression coefficient
    k_AT_W      = 0.10,    # adim — W' depletion → vagal withdrawal (Kontro 2026)
    RMSSD_ref_ms = 35.0,   # ms — population RMSSD at full AT (Malik 1996)

    # Numerical guard
    HR_floor = 30.0,       # bpm — absolute floor (pathological bradycardia sentinel)
)


# ── Default initial conditions (resting state) ───────────────────────────────

_p = DEFAULT_CARDIO_SLICE_PARAMS

X0_CARDIO_DEFAULT: jax.Array = jnp.array([
    _p.VO2_rest,          # V_O2   — fasting metabolic rate
    _p.HR_basal,          # HR     — resting heart rate
    _p.SV_ref,            # SV     — resting stroke volume
    _p.W_prime_capacity,  # W'_bal — fully charged battery
    0.01,                 # RF     — near-zero at rest (sigmoid at P=0 ≈ 0.001)
    1.00,                 # AT     — full vagal tone at rest
], dtype=jnp.float32)

# Initial covariance — diagonal onboarding uncertainty
P0_CARDIO_DEFAULT: jax.Array = jnp.diag(jnp.array([
    25.0,    # V_O2   σ² = (5 mL/kg/min)²   — uncertain without ramp test
    100.0,   # HR     σ² = (10 bpm)²          — moderate HR variability
    2.25e-4, # SV     σ² = (0.015 L)²         — modest SV uncertainty
    25.0,    # W'_bal σ² = (5 kJ)²            — W' unknown without supra-max test
    1.0e-2,  # RF     σ² = (0.10)²            — small RF uncertainty at rest
    2.25e-2, # AT     σ² = (0.15)²            — HRV inter-individual variation
], dtype=jnp.float32))


# ── Pure ODE (JIT + vmap safe) ────────────────────────────────────────────────

def cardiorespiratory_slice_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    6-state cardiorespiratory performance ODE.

    Parameters
    ----------
    t     : scalar — current time [min] (diffrax signature)
    x     : shape (STATE_DIM,) — current state
    args  : tuple(CardioSliceParams, power_watts, hub_T_core, hub_pv_drop_pct)
              power_watts    : float [W]
              hub_T_core     : float [°C] or NaN (guarded internally)
              hub_pv_drop_pct: float [%]  or NaN (guarded internally)

    Returns
    -------
    dx/dt : shape (STATE_DIM,)

    Design
    ------
    No Python control flow on traced values; uses jnp.maximum / jnp.where.
    Hub NaN guards via jnp.where(jnp.isnan(.)) before any arithmetic use.
    JIT + vmap safe for sigma-point propagation in the UKF filter.
    """
    params, power_watts, hub_T_core, hub_pv_drop_pct = args

    V_O2          = x[IDX_VO2]
    Heart_Rate    = x[IDX_HR]
    Stroke_Volume = x[IDX_SV]
    W_prime_bal   = x[IDX_WPRIME]
    Resp_Fatigue  = x[IDX_RF]
    Autonomic_Tone = x[IDX_AT]

    P = jnp.asarray(power_watts, dtype=jnp.float32)

    # ── Hub NaN guards (replace missing hubs with neutral physiological defaults)
    hub_T   = jnp.asarray(hub_T_core,       dtype=jnp.float32)
    hub_pv  = jnp.asarray(hub_pv_drop_pct,  dtype=jnp.float32)
    T_core  = jnp.where(jnp.isnan(hub_T),  jnp.float32(37.0), hub_T)
    pv_drop = jnp.where(jnp.isnan(hub_pv), jnp.float32(0.0),  hub_pv)
    pv_drop = jnp.clip(pv_drop, jnp.float32(0.0), jnp.float32(params.PV_drop_ceil))

    P_norm = P / jnp.float32(params.P_ref)
    W_cap  = jnp.float32(params.W_prime_capacity)

    # ══════════════════════════════════════════════════════════════════════
    # I. VO2 KINETICS (Boillet 2024 bioenergetic ODE)
    # ══════════════════════════════════════════════════════════════════════
    VO2_max  = jnp.float32(params.VO2_max_baseline)
    VO2_rest = jnp.float32(params.VO2_rest)
    VO2_ss   = VO2_rest + P_norm * (VO2_max - VO2_rest) * jnp.float32(params.gain_VO2)
    VO2_ss   = jnp.minimum(VO2_ss, VO2_max)
    # Slow component: activates above CP (Gaesser & Poole 1996)
    slow_component = jnp.float32(params.k_slow) * jnp.maximum(
        jnp.float32(0.0), P - jnp.float32(params.CP_watts)
    ) / jnp.float32(params.P_ref)
    dVO2_dt = (VO2_ss - V_O2) / jnp.float32(params.tau_VO2) + slow_component

    # ══════════════════════════════════════════════════════════════════════
    # II. HEART RATE — Fick coupling + CV drift (Coyle 2001)
    # ══════════════════════════════════════════════════════════════════════
    VO2_range  = jnp.maximum(VO2_max - VO2_rest, jnp.float32(1.0))
    VO2_frac   = jnp.clip((V_O2 - VO2_rest) / VO2_range, jnp.float32(0.0), jnp.float32(1.2))
    HR_Fick    = jnp.float32(params.HR_basal) + VO2_frac * (
        jnp.float32(params.HR_max) - jnp.float32(params.HR_basal)
    )
    # CV drift: Frank-Starling compensation when SV drops below reference
    SV_safe    = jnp.maximum(jnp.float32(params.SV_ref) * jnp.float32(0.1), Stroke_Volume)
    HR_drift   = jnp.float32(params.k_HR_drift) * (
        jnp.float32(1.0) - SV_safe / jnp.float32(params.SV_ref)
    )
    # Thermal tachycardia (Coyle 2001): core temp above 37°C drives HR up
    T_delta    = jnp.maximum(jnp.float32(0.0), T_core - jnp.float32(37.0))
    HR_Tcore   = jnp.float32(params.k_HR_Tcore) * T_delta
    HR_target  = jnp.maximum(
        jnp.float32(params.HR_floor),
        HR_Fick + HR_drift + HR_Tcore,
    )
    dHR_dt = (HR_target - Heart_Rate) / jnp.float32(params.tau_HR)

    # ══════════════════════════════════════════════════════════════════════
    # III. STROKE VOLUME — Frank-Starling + PV drift + respiratory steal
    # ══════════════════════════════════════════════════════════════════════
    # PV drop → reduced ventricular filling (Frank-Starling; Coyle 1986)
    drift_factor = jnp.maximum(
        jnp.float32(params.drift_floor),
        jnp.float32(1.0) - jnp.float32(params.k_PV_SV) * pv_drop / jnp.float32(100.0),
    )
    # Respiratory steal: at extreme intensities, blood re-routes to respiratory muscles
    # This reduces locomotor SV effective filling (Dempsey 2006, J Physiol)
    resp_steal = jnp.maximum(
        jnp.float32(0.5),
        jnp.float32(1.0) - jnp.float32(params.k_resp_SV) * Resp_Fatigue,
    )
    SV_target = jnp.float32(params.SV_ref) * drift_factor * resp_steal
    dSV_dt    = (SV_target - Stroke_Volume) / jnp.float32(params.tau_SV)

    # ══════════════════════════════════════════════════════════════════════
    # IV. W' BALANCE — Clarke-Skiba 2013 smooth ODE form
    # ══════════════════════════════════════════════════════════════════════
    excess_power  = jnp.maximum(jnp.float32(0.0), P - jnp.float32(params.CP_watts))
    deficit_power = jnp.maximum(jnp.float32(0.0), jnp.float32(params.CP_watts) - P)
    # Drain: above CP, lose (P−CP) × 0.06 kJ/min (60 s/min × 10⁻³ kJ/J)
    drain_kJ_min = excess_power * jnp.float32(0.06)
    # Recovery: below CP, recover toward W'_capacity at rate ∝ deficit / P_ref
    rec_drive    = deficit_power / jnp.float32(params.P_ref)
    W_deficit    = jnp.maximum(jnp.float32(0.0), W_cap - W_prime_bal)
    rec_kJ_min   = rec_drive * W_deficit / jnp.float32(params.tau_W_rec)
    dW_dt = rec_kJ_min - drain_kJ_min

    # ══════════════════════════════════════════════════════════════════════
    # V. RESPIRATORY FATIGUE — Dempsey 2006 metaboreflex
    # ══════════════════════════════════════════════════════════════════════
    # Sigmoid activation above RF_threshold (≈85% P_ref), representing
    # diaphragm/accessory-muscle fatigue → group III/IV afferent firing
    RF_target = jnp.float32(1.0) / (
        jnp.float32(1.0) + jnp.exp(
            -jnp.float32(params.k_RF_gain) * (P_norm - jnp.float32(params.RF_threshold))
        )
    )
    dRF_dt = (RF_target - Resp_Fatigue) / jnp.float32(params.tau_RF)

    # ══════════════════════════════════════════════════════════════════════
    # VI. AUTONOMIC TONE — HRV proxy (Arai 1989 + Kontro 2026)
    # ══════════════════════════════════════════════════════════════════════
    # Recovery toward 1.0 at rest (τ ≈ 20 min, Arai 1989)
    # Suppressed by: (a) exercise intensity, (b) W' depletion metabolites (Kontro 2026)
    w_depletion = jnp.maximum(
        jnp.float32(0.0),
        jnp.float32(1.0) - W_prime_bal / jnp.maximum(W_cap, jnp.float32(0.1)),
    )
    dAT_dt = (
        jnp.float32(params.k_AT_rec) * (jnp.float32(1.0) - Autonomic_Tone)
        - jnp.float32(params.k_AT_sup)  * P_norm * Autonomic_Tone
        - jnp.float32(params.k_AT_W)    * w_depletion * Autonomic_Tone
    )

    return jnp.stack([dVO2_dt, dHR_dt, dSV_dt, dW_dt, dRF_dt, dAT_dt])
