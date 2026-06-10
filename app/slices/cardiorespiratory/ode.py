"""
app/slices/cardiorespiratory/ode.py

L2 Backbone ODE -- Cardiorespiratory Performance Slice (8-state)

State x in R^8
══════════════
  x[0]  V_O2           oxygen uptake                        [mL/kg/min]
  x[1]  Heart_Rate     effective heart rate                  [bpm]
  x[2]  Stroke_Volume  stroke volume (Frank-Starling)        [L]
  x[3]  W_fast         anaerobic fast pool (PCr-linked)      [kJ]   phi=0.40, tau~2 min
  x[4]  W_slow         anaerobic slow pool (metabolic)       [kJ]   phi=0.60, tau~30 min
  x[5]  Resp_Fatigue   respiratory muscle fatigue            [adim, 0-1]
  x[6]  Autonomic_Tone vagal / parasympathetic tone          [adim, 0-1]
  x[7]  RMSSD_load_7d  7-day EWMA of instantaneous RMSSD    [ms]

W' bi-exponential model (Caen et al. 2021, Med Sci Sports Exerc)
-----------------------------------------------------------------
W'_total = W_fast + W_slow
  Fast pool:  phi = 0.40;  W_fast_cap = phi * W'_capacity;  tau_fast ~ 2 min   (PCr)
  Slow pool:  1-phi = 0.60; W_slow_cap = (1-phi)*W'_capacity; tau_slow ~ 30 min (metabolic)

Depletion (P > CP):
  drain_total = (P - CP) * 0.06 kJ/min
  Each pool depletes proportional to its current fractional content:
    drain_i = (W_i / W_total) * drain_total

Recovery (P < CP, independent pool kinetics -- Caen 2021 Eq. 2):
  rec_i = (CP - P) / P_ref * max(0, W_i_cap - W_i) / tau_i

RMSSD_load_7d (continuous EWMA at prescription timescale)
----------------------------------------------------------
  RMSSD_inst = Autonomic_Tone * RMSSD_ref_ms   (instantaneous proxy)
  dRMSSD_7d/dt = (RMSSD_inst - RMSSD_load_7d) / tau_RMSSD_7d_min
  tau = 7 * 24 * 60 = 10080 min

Control inputs  (ODE args)
  power_watts      [W]   -- session power demand
  hub_T_core       [C]   -- core temperature (Mod 10 hub); NaN if unavailable
  hub_pv_drop_pct  [%]   -- plasma volume drop (Mod 12 hub); NaN if unavailable

ODE equations
─────────────
I.   VO2 kinetics     -- Boillet, Messonnier & Cohen (2024) Sci Rep 14:5050
II.  Heart rate       -- CV drift (Coyle & Gonzalez-Alonso 2001)
III. Stroke volume    -- Frank-Starling + respiratory steal (Dempsey 2006)
IV.  W' bi-exponential-- Caen et al. (2021) Med Sci Sports Exerc 53:2349-2360
V.   Resp. fatigue    -- Dempsey (2006) J Physiol 564:425-445
VI.  Autonomic tone   -- Arai (1989) + Kontro (2026)
VII. RMSSD_load_7d    -- continuous EWMA (Malik 1996)

Design invariants
─────────────────
  Pure JAX: no Python conditionals on traced values.
  jnp.maximum / jnp.where for all smooth gates.
  JIT + vmap safe (sigma-point propagation in UKF).
  Fail-Loud: NaN propagates -- no silent substitution.
  Hub NaN guards via jnp.where(jnp.isnan(.)) before any use.

References
──────────
  Boillet L. et al. (2024) Sci Rep 14:5050
    DOI 10.1038/s41598-024-56042-0
  Caen K. et al. (2021) Med Sci Sports Exerc 53:2349-2360
    DOI 10.1249/MSS.0000000000002673   [W' bi-exponential recovery]
  Clarke D.C. & Skiba P.F. (2013) Am J Physiol Regul 305:R100-R111
    DOI 10.1152/ajpregu.00444.2012
  Coyle E.F. & Gonzalez-Alonso J. (2001) J Appl Physiol 90:1389-1395
  Dempsey J.A. et al. (2006) J Physiol 564:425-445
  Arai Y. et al. (1989) Circulation 79:86-93
  Kontro H. et al. (2026) PLOS ONE DOI 10.1371/journal.pone.0341721
  Malik M. et al. (1996) Eur Heart J 17:354-381
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── State vector indices ──────────────────────────────────────────────────────
IDX_VO2     = 0   # V_O2          [mL/kg/min]
IDX_HR      = 1   # Heart_Rate    [bpm]
IDX_SV      = 2   # Stroke_Volume [L]
IDX_WFAST   = 3   # W_fast        [kJ]  fast pool (PCr, tau~2 min)
IDX_WSLOW   = 4   # W_slow        [kJ]  slow pool (metabolic, tau~30 min)
IDX_RF      = 5   # Resp_Fatigue  [adim, 0-1]
IDX_AT      = 6   # Autonomic_Tone [adim, 0-1]
IDX_RMSSD7D = 7   # RMSSD_load_7d [ms]  7-day EWMA

STATE_DIM: int = 8
OBS_DIM:   int = 3   # [HR_obs_bpm, VO2_obs_mLkgmin, RMSSD_obs_ms]


# ── Parameter container ───────────────────────────────────────────────────────

class CardioSliceParams(NamedTuple):
    """
    Parameters for the 8-state cardiorespiratory performance ODE.

    Time unit: minutes throughout (rates min^-1, time constants min).
    Personalised per-subject via NLME: VO2_max_baseline, W_prime_capacity.
    """
    # ── I. VO2 kinetics (Boillet 2024) ───────────────────────────────────
    VO2_max_baseline: float   # mL/kg/min
    VO2_rest:         float   # mL/kg/min
    tau_VO2:          float   # min
    k_slow:           float   # adim
    gain_VO2:         float   # adim

    # ── II. Cardiac (Coyle 2001 CV drift) ────────────────────────────────
    HR_basal:      float   # bpm
    HR_max:        float   # bpm
    tau_HR:        float   # min
    k_HR_drift:    float   # bpm
    k_HR_Tcore:    float   # bpm/C

    # ── III. Stroke volume (Frank-Starling + CV drift) ────────────────────
    SV_ref:        float   # L
    tau_SV:        float   # min
    k_PV_SV:       float   # adim
    drift_floor:   float   # adim
    k_resp_SV:     float   # adim
    PV_drop_ceil:  float   # %

    # ── IV. W' bi-exponential (Caen 2021) ────────────────────────────────
    W_prime_capacity: float   # kJ  total anaerobic battery
    W_prime_phi:      float   # adim  fraction in fast pool (0.40)
    tau_W_fast:       float   # min  fast recovery tau (~2 min, PCr)
    tau_W_slow:       float   # min  slow recovery tau (~30 min, metabolic)
    CP_watts:         float   # W   critical power
    P_ref:            float   # W   normalisation power

    # ── V. Respiratory fatigue (Dempsey 2006) ────────────────────────────
    tau_RF:        float   # min
    k_RF_gain:     float   # adim
    RF_threshold:  float   # adim

    # ── VI. Autonomic tone (Arai 1989 + Kontro 2026) ─────────────────────
    k_AT_rec:      float   # min^-1
    k_AT_sup:      float   # adim
    k_AT_W:        float   # adim
    RMSSD_ref_ms:  float   # ms

    # ── VII. RMSSD_load_7d EWMA ───────────────────────────────────────────
    tau_RMSSD_7d_min: float   # min  (7 * 24 * 60 = 10080)

    # ── Numerical guards ─────────────────────────────────────────────────
    HR_floor: float   # bpm

    # ── Circadian modulator (Borbely 1982 two-process model) ─────────────
    # AT equilibrium shifts to (1.0 + circ_amp) during Phase 2 sleep.
    # Set to 0.0 for all daytime operation; injected by _blind_overnight_predict.
    # Propagated correctly through UKF sigma points inside the ODE -- no external
    # mean-hacks needed.
    circ_amp: float = 0.0   # adim; AT_target = 1.0 + circ_amp


# ── Population-mean default parameters ───────────────────────────────────────

DEFAULT_CARDIO_SLICE_PARAMS = CardioSliceParams(
    # I. VO2 kinetics
    VO2_max_baseline = 45.0,
    VO2_rest         = 3.5,
    tau_VO2          = 0.75,
    k_slow           = 0.12,
    gain_VO2         = 0.90,

    # II. Cardiac
    HR_basal     = 60.0,
    HR_max       = 185.0,
    tau_HR       = 1.5,
    k_HR_drift   = 20.0,
    k_HR_Tcore   = 8.0,

    # III. Stroke volume
    SV_ref       = 0.083,
    tau_SV       = 2.0,
    k_PV_SV      = 1.5,
    drift_floor  = 0.50,
    k_resp_SV    = 0.15,
    PV_drop_ceil = 20.0,

    # IV. W' bi-exponential (Caen 2021)
    W_prime_capacity = 18.0,   # kJ   total capacity
    W_prime_phi      = 0.40,   # fast pool fraction (Caen 2021: phi ~0.40)
    tau_W_fast       = 2.0,    # min  ~120 s PCr replenishment (Caen 2021 Table 1)
    tau_W_slow       = 30.0,   # min  ~30 min metabolic recovery (Caen 2021 Table 1)
    CP_watts         = 250.0,
    P_ref            = 350.0,

    # V. Respiratory fatigue
    tau_RF       = 3.0,
    k_RF_gain    = 8.0,
    RF_threshold = 0.85,

    # VI. Autonomic tone
    k_AT_rec     = 0.050,
    k_AT_sup     = 0.20,
    k_AT_W       = 0.10,
    RMSSD_ref_ms = 35.0,

    # VII. RMSSD_load_7d
    tau_RMSSD_7d_min = 10080.0,   # 7 * 24 * 60

    # Numerical guard
    HR_floor = 30.0,

    # Circadian modulator (always off for daytime / default params)
    circ_amp = 0.0,
)


# ── Default initial conditions (resting state) ───────────────────────────────

_p = DEFAULT_CARDIO_SLICE_PARAMS

X0_CARDIO_DEFAULT: jax.Array = jnp.array([
    _p.VO2_rest,                                         # V_O2
    _p.HR_basal,                                         # HR
    _p.SV_ref,                                           # SV
    _p.W_prime_phi * _p.W_prime_capacity,               # W_fast (fully charged)
    (1.0 - _p.W_prime_phi) * _p.W_prime_capacity,       # W_slow (fully charged)
    0.01,                                                # RF (near-zero at rest)
    1.00,                                                # AT (full vagal tone)
    _p.RMSSD_ref_ms,                                    # RMSSD_7d (full recovery)
], dtype=jnp.float32)

# Initial covariance -- diagonal onboarding uncertainty
P0_CARDIO_DEFAULT: jax.Array = jnp.diag(jnp.array([
    25.0,    # V_O2    (5 mL/kg/min)^2
    100.0,   # HR      (10 bpm)^2
    2.25e-4, # SV      (0.015 L)^2
    4.0,     # W_fast  (2 kJ)^2    uncertain split between pools
    9.0,     # W_slow  (3 kJ)^2    uncertain split between pools
    1.0e-2,  # RF      (0.10)^2
    2.25e-2, # AT      (0.15)^2
    50.0,    # RMSSD_7d (7 ms)^2  broad HRV population prior
], dtype=jnp.float32))


# ── Pure ODE (JIT + vmap safe) ────────────────────────────────────────────────

def cardiorespiratory_slice_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    8-state cardiorespiratory performance ODE.

    Parameters
    ----------
    t     : scalar -- current time [min] (diffrax signature)
    x     : (STATE_DIM,) -- current state
    args  : (CardioSliceParams, power_watts, hub_T_core, hub_pv_drop_pct)

    Returns
    -------
    dx/dt : (STATE_DIM,)
    """
    params, power_watts, hub_T_core, hub_pv_drop_pct = args

    V_O2           = x[IDX_VO2]
    Heart_Rate     = x[IDX_HR]
    Stroke_Volume  = x[IDX_SV]
    W_fast         = x[IDX_WFAST]
    W_slow         = x[IDX_WSLOW]
    Resp_Fatigue   = x[IDX_RF]
    Autonomic_Tone = x[IDX_AT]
    RMSSD_load_7d  = x[IDX_RMSSD7D]

    P = jnp.asarray(power_watts, dtype=jnp.float32)

    # ── Hub NaN guards (replace missing hubs with neutral physiological defaults)
    hub_T   = jnp.asarray(hub_T_core,      dtype=jnp.float32)
    hub_pv  = jnp.asarray(hub_pv_drop_pct, dtype=jnp.float32)
    T_core  = jnp.where(jnp.isnan(hub_T),  jnp.float32(37.0), hub_T)
    pv_drop = jnp.where(jnp.isnan(hub_pv), jnp.float32(0.0),  hub_pv)
    pv_drop = jnp.clip(pv_drop, jnp.float32(0.0), jnp.float32(params.PV_drop_ceil))

    P_norm = P / jnp.float32(params.P_ref)
    W_cap  = jnp.float32(params.W_prime_capacity)

    # ══════════════════════════════════════════════════════════════════════
    # I. VO2 KINETICS (Boillet 2024)
    # ══════════════════════════════════════════════════════════════════════
    VO2_max  = jnp.float32(params.VO2_max_baseline)
    VO2_rest = jnp.float32(params.VO2_rest)
    VO2_ss   = VO2_rest + P_norm * (VO2_max - VO2_rest) * jnp.float32(params.gain_VO2)
    VO2_ss   = jnp.minimum(VO2_ss, VO2_max)
    slow_comp = jnp.float32(params.k_slow) * jnp.maximum(
        jnp.float32(0.0), P - jnp.float32(params.CP_watts)
    ) / jnp.float32(params.P_ref)
    dVO2_dt = (VO2_ss - V_O2) / jnp.float32(params.tau_VO2) + slow_comp

    # ══════════════════════════════════════════════════════════════════════
    # II. HEART RATE -- Fick coupling + CV drift (Coyle 2001)
    # ══════════════════════════════════════════════════════════════════════
    VO2_range = jnp.maximum(VO2_max - VO2_rest, jnp.float32(1.0))
    VO2_frac  = jnp.clip((V_O2 - VO2_rest) / VO2_range, jnp.float32(0.0), jnp.float32(1.2))
    HR_Fick   = jnp.float32(params.HR_basal) + VO2_frac * (
        jnp.float32(params.HR_max) - jnp.float32(params.HR_basal)
    )
    SV_safe   = jnp.maximum(jnp.float32(params.SV_ref) * jnp.float32(0.1), Stroke_Volume)
    HR_drift  = jnp.float32(params.k_HR_drift) * (
        jnp.float32(1.0) - SV_safe / jnp.float32(params.SV_ref)
    )
    T_delta   = jnp.maximum(jnp.float32(0.0), T_core - jnp.float32(37.0))
    HR_Tcore  = jnp.float32(params.k_HR_Tcore) * T_delta
    HR_target = jnp.maximum(
        jnp.float32(params.HR_floor),
        HR_Fick + HR_drift + HR_Tcore,
    )
    dHR_dt = (HR_target - Heart_Rate) / jnp.float32(params.tau_HR)

    # ══════════════════════════════════════════════════════════════════════
    # III. STROKE VOLUME -- Frank-Starling + PV drift + respiratory steal
    # ══════════════════════════════════════════════════════════════════════
    drift_factor = jnp.maximum(
        jnp.float32(params.drift_floor),
        jnp.float32(1.0) - jnp.float32(params.k_PV_SV) * pv_drop / jnp.float32(100.0),
    )
    resp_steal = jnp.maximum(
        jnp.float32(0.5),
        jnp.float32(1.0) - jnp.float32(params.k_resp_SV) * Resp_Fatigue,
    )
    SV_target = jnp.float32(params.SV_ref) * drift_factor * resp_steal
    dSV_dt    = (SV_target - Stroke_Volume) / jnp.float32(params.tau_SV)

    # ══════════════════════════════════════════════════════════════════════
    # IV. W' BI-EXPONENTIAL (Caen et al. 2021)
    #
    # Fast pool (phi=0.40, tau~2 min): PCr-linked, rapid reconstitution.
    # Slow pool (1-phi=0.60, tau~30 min): metabolic, glycolytic/oxidative.
    #
    # Depletion (P > CP): proportional drain per pool based on current fraction.
    # Recovery (P < CP): independent Caen 2021 kinetics per pool.
    # ══════════════════════════════════════════════════════════════════════
    phi         = jnp.float32(params.W_prime_phi)
    W_fast_cap  = phi * W_cap
    W_slow_cap  = (jnp.float32(1.0) - phi) * W_cap

    excess_power  = jnp.maximum(jnp.float32(0.0), P - jnp.float32(params.CP_watts))
    deficit_power = jnp.maximum(jnp.float32(0.0), jnp.float32(params.CP_watts) - P)

    # Drain: proportional to each pool's current content fraction
    drain_total = excess_power * jnp.float32(0.06)             # kJ/min
    W_total_eps = W_fast + W_slow + jnp.float32(1e-6)          # prevent div-by-zero
    drain_fast  = (W_fast / W_total_eps) * drain_total
    drain_slow  = (W_slow / W_total_eps) * drain_total

    # Recovery: independent Caen 2021 kinetics (Eq. 2), gated by deficit below CP
    rec_drive   = deficit_power / jnp.float32(params.P_ref)
    rec_fast    = rec_drive * jnp.maximum(
        jnp.float32(0.0), W_fast_cap - W_fast
    ) / jnp.float32(params.tau_W_fast)
    rec_slow    = rec_drive * jnp.maximum(
        jnp.float32(0.0), W_slow_cap - W_slow
    ) / jnp.float32(params.tau_W_slow)

    dW_fast_dt = rec_fast - drain_fast
    dW_slow_dt = rec_slow - drain_slow

    # ══════════════════════════════════════════════════════════════════════
    # V. RESPIRATORY FATIGUE -- Dempsey 2006 metaboreflex
    # ══════════════════════════════════════════════════════════════════════
    RF_target = jnp.float32(1.0) / (
        jnp.float32(1.0) + jnp.exp(
            -jnp.float32(params.k_RF_gain) * (P_norm - jnp.float32(params.RF_threshold))
        )
    )
    dRF_dt = (RF_target - Resp_Fatigue) / jnp.float32(params.tau_RF)

    # ══════════════════════════════════════════════════════════════════════
    # VI. AUTONOMIC TONE -- HRV proxy (Arai 1989 + Kontro 2026)
    #
    # Depletion signal uses TOTAL W' (fast + slow pools combined).
    # ══════════════════════════════════════════════════════════════════════
    W_total_real = W_fast + W_slow
    w_depletion  = jnp.maximum(
        jnp.float32(0.0),
        jnp.float32(1.0) - W_total_real / jnp.maximum(W_cap, jnp.float32(0.1)),
    )
    # AT_target = 1.0 during daytime (circ_amp=0); elevated during Phase 2 sleep.
    # sigma points all carry the same circ_amp so covariance propagates correctly.
    AT_target = jnp.float32(1.0) + jnp.float32(params.circ_amp)
    dAT_dt = (
        jnp.float32(params.k_AT_rec) * (AT_target - Autonomic_Tone)
        - jnp.float32(params.k_AT_sup) * P_norm * Autonomic_Tone
        - jnp.float32(params.k_AT_W)   * w_depletion * Autonomic_Tone
    )

    # ══════════════════════════════════════════════════════════════════════
    # VII. RMSSD_load_7d -- continuous EWMA (Malik 1996 proxy)
    #
    # Tracks the 7-day rolling RMSSD load for inter-day HRV prediction.
    # Feeds into NMPC and Phase3Envelope as a slow-timescale recovery index.
    # Instantaneous RMSSD proxy = AT * RMSSD_ref_ms (same as observation model).
    # tau = 10080 min (7 days) makes this state nearly constant at 1-min scale.
    # ══════════════════════════════════════════════════════════════════════
    RMSSD_inst   = jnp.float32(params.RMSSD_ref_ms) * Autonomic_Tone
    dRMSSD7D_dt  = (RMSSD_inst - RMSSD_load_7d) / jnp.float32(params.tau_RMSSD_7d_min)

    return jnp.stack([
        dVO2_dt,
        dHR_dt,
        dSV_dt,
        dW_fast_dt,
        dW_slow_dt,
        dRF_dt,
        dAT_dt,
        dRMSSD7D_dt,
    ])
