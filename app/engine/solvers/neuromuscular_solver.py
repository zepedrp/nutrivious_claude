"""
app/engine/solvers/neuromuscular_solver.py

Módulo 2 — Motor Neuromuscular e Fadiga (MHDS Subsistema 2) — Patch v1.2

Arquitectura Multi-Escala em dois estágios acoplados em cascata:

PARTE 1 — Motor Agudo (Xia-Frey-Law + Hill Recovery + Dano Estrutural, escala de minutos):
    Modela a dinâmica intra-sessão das fibras motoras segundo o formalismo de
    três compartimentos de Xia, Frey & Pham (2008), acoplado à transição
    activa ↔ fatigada de Law & Shields (1997).
    v1.2 eleva o vetor de estado de 2 → 3 ODEs, adicionando:
        — Recuperação Hill-type (inversamente proporcional à fadiga)
        — Dano Estrutural D_muscle (acumulação tensão mecânica × fadiga)

PARTE 2 — Motor Crónico (Busso 2003 Variable-Dose Response, escala de dias):
    3 ODEs não-lineares: G_fit, F_chr, k2_dynamic (ganho dinâmico).

══════════════════════════════════════════════════════════════════════════════
EQUAÇÕES DO MOTOR AGUDO v1.2 (M ∈ [0,1], D_muscle ∈ [0,+∞))
══════════════════════════════════════════════════════════════════════════════

Vetor de estado (escala aguda):  y_acute = [M_act, M_fat, D_muscle]

  M_act    : fracção de unidades motoras activas       (adim., ∈ [0,1])
  M_fat    : fracção de unidades motoras fatigadas     (adim., ∈ [0,1])
  D_muscle : dano estrutural muscular acumulado        (adim., ≥ 0)
  M_rest = 1 − M_act − M_fat  (deduzida por conservação)

──────────────────────────────────────────────────────────────────────────────
ODE I — Recrutamento / Deactivação
──────────────────────────────────────────────────────────────────────────────

  dM_act/dt = F_stim(P_t) · M_rest  −  k_fat · M_act
                                    −  k_rec · M_act · (1 − M_fat)

  onde  F_stim = f_max · tanh(k_stim · P_t / P_ref)

──────────────────────────────────────────────────────────────────────────────
ODE II — Acumulação de Fadiga com Hill-type Recovery (v1.2)
──────────────────────────────────────────────────────────────────────────────

  dM_fat/dt = k_fat · M_act  −  k_rec_fat_eff · M_fat

  onde:
    k_rec_fat_eff = k_rec_fat / (1 + k_hill_fat · M_fat)

  Interpretação fisiológica (Westerblad et al. 2002 / Allen et al. 2008):
  A recuperação da fadiga não é linear — abranda à medida que M_fat sobe
  porque a reacumulação de Ca²⁺ e a desfosforilação das cabeças de miosina
  são processos de saturação. k_hill_fat ≈ 5 → a 20% de fadiga, a taxa de
  recuperação já reduziu para 1/(1+1) = 50% do valor linear.

──────────────────────────────────────────────────────────────────────────────
ODE III — Dano Estrutural Muscular (v1.2, novo)
──────────────────────────────────────────────────────────────────────────────

  dD_muscle/dt = k_dmg · (P_t / P_ref) · M_fat  −  k_repair · D_muscle

  Física:
    k_dmg · (P_t / P_ref) · M_fat : acumulação — tensão mecânica normalisada
        ponderada pela fracção de fibras fatigadas (fibras fatigadas têm
        limiar de dano mais baixo — Morgan 1990, excentric exercise damage).
    k_repair · D_muscle            : reparação espontânea com τ ≈ 5.5h
        (k_repair ≈ 0.003 min⁻¹ → síntese de proteína estrutural / HSP70).

  Hub Variable downstream:
    D_muscle → Hub_CK_StructuralDamage (fuga de Creatina Quinase → Módulo 7)

══════════════════════════════════════════════════════════════════════════════
EQUAÇÕES DO MOTOR CRÓNICO (Busso 2003, dias) — inalteradas em v1.2
══════════════════════════════════════════════════════════════════════════════

Vetor crónico:  y_chr = [G_fit, F_chr, k2]

  dG/dt  = k1 · w_dose(t)  −  G / τ₁
  dF/dt  = k2(t) · w_dose(t)  −  F / τ₂
  dk2/dt = k3 · w_dose(t)  −  (k2 − k2_basal) / τ₃

══════════════════════════════════════════════════════════════════════════════
VARIÁVEIS HUB EXPORTADAS (v1.2, Bond Graph — acoplamento cross-módulo)
══════════════════════════════════════════════════════════════════════════════

  Hub_CK_StructuralDamage  ∝ D_muscle(t)   → Módulo 7 (Inflamatório)
      CK sérica aumenta proporcionalmente ao dano estrutural (Brancaccio 2007)

  Hub_FGF21_MetabolicStress ∝ M_fat(t)     → Módulo 6 (Tiroide) + Módulo 1
      FGF21 é libertada em resposta ao stress metabólico muscular (Xu 2009)

  Hub_FGF23_BoneCoupling ∝ max(0, −ΔM_act/Δt) → Módulo 11 (Biomecânico)
      A depleção de M_act liberta fosfato inorgânico → sinal FGF23 ósseo
      (Shimada 2004; fosfato acoplado ao remodelamento mineral ósseo)

══════════════════════════════════════════════════════════════════════════════
REFERÊNCIAS v1.2 (adicionadas)
══════════════════════════════════════════════════════════════════════════════
  Westerblad H. et al. (2002) News Physiol Sci 17:17–21  [Hill recovery]
  Allen D.G. et al. (2008) Physiol Rev 88:287–332        [Ca²⁺ saturation]
  Morgan D.L. (1990) J Physiol 415:351–361               [eccentric damage]
  Brancaccio P. et al. (2007) Clin Chem Lab Med 45:1235  [CK structural]
  Xu J. et al. (2009) Cell Metab 10:219–227              [FGF21 muscle]
  Shimada T. et al. (2004) J Clin Invest 113:561–568     [FGF23 phosphate]

Referências base:
  Xia T., Frey Law L.A., Pham L. (2008) J Biomech 41(5):1072–8
  Law L.A., Shields R.K. (1997) Clin Biomech
  Busso S. (2003) Med Sci Sports Exerc 35(7):1188–95
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

# ── Reference constants: acute (never hardcoded in business logic) ────────────
_TAU_FAT_REF_MIN: float  = 11.0 * 1440.0  # 11 days → minutes (τ₂ ref)
_K_FAT_REF: float        = 0.045           # active→fatigued [min⁻¹]
_K_REC_REF: float        = 0.010           # active→rest recovery [min⁻¹]
_K_REC_FAT_REF: float    = 0.008          # fatigued→rest recovery [min⁻¹]
_F_MAX_REF: float        = 0.60            # max recruitment rate [min⁻¹]
_K_STIM_REF: float       = 3.5            # stimulation gain [adim.]
_P_REF_W: float          = 200.0           # normalisation power [W]

# ── Reference constants: v1.2 additions (Hill recovery + structural damage) ───
_K_HILL_FAT_REF: float   = 5.0    # Hill-type recovery attenuation [adim.]
                                   # k_rec_fat_eff = k_rec_fat / (1 + k_hill_fat·M_fat)
                                   # @ M_fat=0.20 → rate × 0.50; @ M_fat=0.40 → × 0.33
_K_DMG_REF: float        = 0.020  # damage accumulation [min⁻¹, P normalized]
                                   # dD/dt = k_dmg·(P/P_ref)·M_fat − k_repair·D
_K_REPAIR_REF: float     = 0.003  # spontaneous repair [min⁻¹]; τ ≈ 333 min ≈ 5.5 h
                                   # HSP70 / protein synthesis timescale (Morgan 1990)

# ── Reference constants: chronic (Busso 2003, Table 1) ───────────────────────
_TAU1_REF_DAYS: float    = 45.0   # fitness decay [days]
_TAU2_REF_DAYS: float    = 11.0   # fatigue decay [days] (ACTN3-modulated)
_TAU3_REF_DAYS: float    = 5.0    # k2 return-to-basal [days]
_K1_REF: float           = 0.069  # fitness gain [u.a.·min⁻¹·day⁻¹]
_K2_BASAL_REF: float     = 0.27   # basal fatigue gain [u.a.·min⁻¹·day⁻¹]
_K3_REF: float           = 0.012  # k2 up-regulation sensitivity [day⁻²]
_P_BASAL_REF: float      = 100.0  # baseline performance [u.a.]
_DOSE_GATE_SIGMA: float  = 20.0   # tanh sharpness for daily dose gate [day⁻¹]
_DOSE_GATE_DELTA: float  = 1.0 / 24.0  # dose injection width [days] (1 hour)


# ── Parameter containers ─────────────────────────────────────────────────────

class AcuteXiaParams(NamedTuple):
    """
    Parameters for the Xia-Frey-Law acute motor unit ODE — v1.2 (3-state).

    All rates in [min⁻¹]; powers in [W]; Hill coefficient dimensionless.

    Fields (original)
    -----------------
    f_max      : maximum motor-unit recruitment rate
    k_stim     : dimensionless gain of the tanh stimulation function
    P_ref      : reference power for stimulation normalisation [W]
    k_fat      : M_act → M_fat transition rate (fatigue accumulation)
    k_rec      : M_act → M_rest recovery during low-load intervals
    k_rec_fat  : M_fat → M_rest base recovery rate (Hill-modified in ODE)

    Fields added in v1.2
    --------------------
    k_hill_fat : Hill-type attenuation of recovery as M_fat rises   [adim.]
                 k_rec_fat_eff = k_rec_fat / (1 + k_hill_fat · M_fat)
    k_dmg      : structural damage accumulation rate (P-normalised)  [min⁻¹]
                 dD/dt += k_dmg · (P_t / P_ref) · M_fat
    k_repair   : spontaneous damage repair rate                      [min⁻¹]
                 τ_repair = 1/k_repair ≈ 5.5 h (protein synthesis)
    """
    f_max:     float
    k_stim:    float
    P_ref:     float
    k_fat:     float
    k_rec:     float
    k_rec_fat: float
    # v1.2
    k_hill_fat: float
    k_dmg:      float
    k_repair:   float


# ── Pure ODE (JIT-compilable, no Python branches on JAX values) ──────────────

def acute_motor_unit_ode(t, y, args):
    """
    Xia-Frey-Law 3-state ODE for intra-session motor unit dynamics — v1.2.

    State vector
    ------------
    y[0]  M_act     : fraction of active motor units         ∈ [0, 1]
    y[1]  M_fat     : fraction of fatigued motor units       ∈ [0, 1]
    y[2]  D_muscle  : structural muscle damage accumulation  ∈ [0, +∞)
    (M_rest = 1 − M_act − M_fat, deduced from conservation)

    ODEs
    ----
    dM_act/dt    = F_stim·M_rest − k_fat·M_act − k_rec·M_act·(1−M_fat)
    dM_fat/dt    = k_fat·M_act − (k_rec_fat / (1 + k_hill_fat·M_fat))·M_fat
    dD_muscle/dt = k_dmg·(P_t/P_ref)·M_fat − k_repair·D_muscle

    args
    ----
    (AcuteXiaParams, power_w, t_sess_start, sess_dur_min)
    """
    params, power_w, t_sess_start, sess_dur_min = args

    M_act    = y[0]
    M_fat    = y[1]
    D_muscle = y[2]
    M_rest   = jnp.maximum(0.0, 1.0 - M_act - M_fat)

    # ── Smooth exercise gate ──────────────────────────────────────────────────
    def _smooth_on(t0):
        return 0.5 * (1.0 + jnp.tanh(20.0 * (t - t0)))

    sess_gate = _smooth_on(t_sess_start) - _smooth_on(t_sess_start + sess_dur_min)
    P_t = power_w * sess_gate

    # ── Stimulation function ──────────────────────────────────────────────────
    F_stim = params.f_max * jnp.tanh(params.k_stim * P_t / params.P_ref)

    # ── ODE I: recruitment / deactivation ────────────────────────────────────
    dM_act_dt = (
        F_stim * M_rest
        - params.k_fat * M_act
        - params.k_rec * M_act * (1.0 - M_fat)
    )

    # ── ODE II: fatigue accumulation with Hill-type recovery (v1.2) ───────────
    # k_rec_fat_eff = k_rec_fat / (1 + k_hill_fat · M_fat)
    # As M_fat rises, recovery rate falls — Ca²⁺ re-uptake and myosin
    # dephosphorylation saturate (Westerblad 2002, Allen 2008).
    k_rec_fat_eff = params.k_rec_fat / (1.0 + params.k_hill_fat * M_fat)
    dM_fat_dt = params.k_fat * M_act - k_rec_fat_eff * M_fat

    # ── ODE III: structural damage (v1.2, new state) ─────────────────────────
    # dD_muscle/dt = k_dmg · (P_t / P_ref) · M_fat − k_repair · D_muscle
    # Fatigued fibres have lower damage threshold (Morgan 1990 eccentric model):
    # damage flux scales with M_fat (fatigued fraction) × normalized power.
    # Repair follows first-order kinetics: τ_repair ≈ 5.5 h (HSP70 synthesis).
    dD_muscle_dt = (
        params.k_dmg * (P_t / params.P_ref) * M_fat
        - params.k_repair * D_muscle
    )

    return jnp.stack([dM_act_dt, dM_fat_dt, dD_muscle_dt])


# ── JIT kernel: acute session integrator ─────────────────────────────────────

@jax.jit
def _solve_acute_session(
    params: AcuteXiaParams,
    power_w: float,
    t_sess_start: float,
    sess_dur_min: float,
    n_save: int = 120,
) -> diffrax.Solution:
    """
    JIT-compiled integrator for the acute Xia-Frey-Law ODE (v1.2, 3-state).

    Returns
    -------
    diffrax.Solution with .ys of shape (n_save, 3): [M_act, M_fat, D_muscle]
    """
    t0 = 0.0
    t1 = t_sess_start + sess_dur_min + 10.0   # +10-min recovery tail
    ts = jnp.linspace(t0, t1, n_save)

    # All states at zero initially: no active, no fatigued, no structural damage
    y0 = jnp.array([0.0, 0.0, 0.0])

    args = (params, power_w, t_sess_start, sess_dur_min)

    return diffrax.diffeqsolve(
        diffrax.ODETerm(acute_motor_unit_ode),
        diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=0.5,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=16_384,
    )


# ── Public helper: compute mechanical impulse w from acute solution ───────────

def compute_mechanical_impulse(sol: diffrax.Solution) -> float:
    """
    Compute scalar mechanical impulse w = ∫ M_fat(t) dt [min].

    Cross-scale bridge: w summarises intra-session neuromuscular stress
    and is injected as dose into the chronic Busso ODE.

    Works with both 2-state (legacy) and 3-state (v1.2) solutions
    because M_fat is always at column index 1.
    """
    ts    = sol.ts            # shape (n_save,)
    M_fat = sol.ys[:, 1]     # shape (n_save,)
    return float(jnp.trapezoid(M_fat, x=ts))


# ══════════════════════════════════════════════════════════════════════════════
# PARTE 2 — Motor Crónico (Busso 2003 Variable-Dose Response, escala de dias)
# ══════════════════════════════════════════════════════════════════════════════

class ChronicBussoParams(NamedTuple):
    """
    Parameters for the Busso 2003 variable-dose chronic adaptation ODE.

    Time constants in [days]; gains in [u.a.·min⁻¹·day⁻¹].

    Fields
    ------
    k1        : fitness gain coefficient
    k2_basal  : resting value of the dynamic fatigue gain k2
    k3        : up-regulation sensitivity of k2 to training load
    tau_1     : fitness decay time-constant                  [days]
    tau_2     : fatigue decay time-constant (ACTN3-scaled)   [days]
    tau_3     : k2 return-to-basal time-constant             [days]
    p_basal   : baseline performance level                   [u.a.]
    """
    k1:       float
    k2_basal: float
    k3:       float
    tau_1:    float
    tau_2:    float
    tau_3:    float
    p_basal:  float


# ── Pure chronic ODE (JIT-compilable) ────────────────────────────────────────

def chronic_busso_ode(t, y, args):
    """
    Busso 2003 non-linear 3-state chronic adaptation ODE.

    State vector
    ------------
    y[0]  G_fit  : fitness / structural adaptation          [u.a.]
    y[1]  F_chr  : chronic fatigue accumulation             [u.a.]
    y[2]  k2     : dynamic fatigue gain (Busso non-linear)  [u.a.·min⁻¹·day⁻¹]

    ODEs
    ----
    dG/dt  = k1 · w_dose(t) − G / τ₁
    dF/dt  = k2(t) · w_dose(t) − F / τ₂
    dk2/dt = k3 · w_dose(t) − (k2 − k2_basal) / τ₃

    w_dose(t) is a smooth gate centred on t_sess_day with σ=20 day⁻¹ and
    width δ=1/24 day (~1 hour), JIT-safe via double-tanh.
    """
    params, w_dose_scalar, t_sess_day = args

    G_fit = y[0]
    F_chr = y[1]
    k2    = y[2]

    gate_on  = 0.5 * (1.0 + jnp.tanh(_DOSE_GATE_SIGMA * (t - t_sess_day)))
    gate_off = 0.5 * (1.0 - jnp.tanh(_DOSE_GATE_SIGMA * (t - t_sess_day - _DOSE_GATE_DELTA)))
    w_dose   = w_dose_scalar * gate_on * gate_off

    dG_dt  = params.k1 * w_dose - G_fit / params.tau_1
    dF_dt  = k2 * w_dose - F_chr / params.tau_2
    dk2_dt = params.k3 * w_dose - (k2 - params.k2_basal) / params.tau_3

    return jnp.stack([dG_dt, dF_dt, dk2_dt])


# ── JIT kernel: chronic adaptation integrator ─────────────────────────────────

@jax.jit
def _solve_chronic_adaptation(
    params: ChronicBussoParams,
    w_dose_scalar: float,
    t_sess_day: float,
    t_span_days: float,
    n_save: int = 336,
) -> diffrax.Solution:
    """
    JIT-compiled integrator for the Busso 2003 chronic ODE.

    Default n_save=336 → half-day resolution over 14 days.

    Returns
    -------
    diffrax.Solution with .ys of shape (n_save, 3): [G_fit, F_chr, k2]
    """
    t0 = 0.0
    t1 = t_span_days
    ts = jnp.linspace(t0, t1, n_save)

    y0 = jnp.array([0.0, 0.0, params.k2_basal])

    return diffrax.diffeqsolve(
        diffrax.ODETerm(chronic_busso_ode),
        diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        args=(params, w_dose_scalar, t_sess_day),
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=32_768,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTRADOR MULTI-ESCALA — NeuromuscularSolver (v1.2)
# ══════════════════════════════════════════════════════════════════════════════

class NeuromuscularSolver:
    """
    Multi-scale neuromuscular orchestrator (MHDS Subsistema 2) — v1.2.

    Cross-scale cascade
    -------------------
    1. Acute engine (Xia-Frey-Law, minutes):
       3-state ODE [M_act, M_fat, D_muscle].
       → w = ∫ M_fat dt  [min] (mechanical impulse for chronic dose).
       → D_muscle, M_fat, ΔM_act trajectories → Hub Variables.

    2. Cross-scale bridge: w injected as session dose into Busso engine.

    3. Chronic engine (Busso 2003, days):
       3-state ODE [G_fit, F_chr, k2].
       → G_fit, F_chr, k2_dynamic, P_net.

    Hub Variables exported (v1.2)
    ------------------------------
    Hub_CK_StructuralDamage   ∝ D_muscle(t)    → Módulo 7 (Inflamatório)
    Hub_FGF21_MetabolicStress ∝ M_fat(t)       → Módulo 6 (Tiroide) / Módulo 1
    Hub_FGF23_BoneCoupling    ∝ max(0,−ΔM_act) → Módulo 11 (Biomecânico)

    Phase-3 prior injections
    ------------------------
    tau_fatigue_decay_prior (ACTN3 → τ₂ Busso):
        Encodes athlete-specific fatigue decay in days.
        NaN → fallback τ₂ = 11 days (Busso 2003 population mean).

    myh7_fiber_type_prior (MYH7 → k_fat Xia):
        Multiplicative scale on k_fat_ref.
        NaN → fallback k_fat = 0.045 min⁻¹.
    """

    def simulate_neuromuscular_response(
        self,
        bayesian_priors: dict,
        session_record: dict,
        t_span_days: float = 14.0,
    ) -> dict:
        """
        Orchestrate the full acute → chronic cascade and export Hub Variables.

        Parameters
        ----------
        bayesian_priors : dict[str, float]
            Phase-3 output. Keys consumed:
              "tau_fatigue_decay_prior" — τ₂ scaling [days] (ACTN3)
              "myh7_fiber_type_prior"   — k_fat scale [a.u.] (MYH7)
        session_record : dict
            Phase-4 telemetry. Keys consumed:
              "power_output_watts"    — mean power [W]
              "session_duration_secs" — duration [s]
            Optional:
              "t_sess_start_min"      — session onset in acute window [min] (def. 0)
              "t_sess_day"            — day index in chronic window (def. 0)
        t_span_days : float
            Chronic simulation horizon [days] (default 14).

        Returns
        -------
        dict with JAX arrays:
            "t_days"                  : shape (n_save_chr,)  chronic time axis [days]
            "G_fitness"               : shape (n_save_chr,)
            "F_chronic"               : shape (n_save_chr,)
            "k2_dynamic"              : shape (n_save_chr,)
            "Performance_Net"         : shape (n_save_chr,)
            "t_acute_min"             : shape (n_save_acu,)  acute time axis [min]
            "Hub_CK_StructuralDamage" : shape (n_save_acu,)  D_muscle trajectory
            "Hub_FGF21_MetabolicStress": shape (n_save_acu,) M_fat trajectory
            "Hub_FGF23_BoneCoupling"  : shape (n_save_acu,)  -ΔM_act/Δt (≥0)
        """
        # ── 1. Extract Phase-4 inputs ─────────────────────────────────────────
        power_w      = float(session_record.get("power_output_watts", 150.0))
        dur_secs     = float(session_record.get("session_duration_secs", 3600.0))
        sess_dur_min = dur_secs / 60.0
        t_sess_start = float(session_record.get("t_sess_start_min", 0.0))
        t_sess_day   = float(session_record.get("t_sess_day", 0.0))

        # ── 2. Phase-3 prior injection — ACTN3 → τ₂ ─────────────────────────
        raw_tau2 = bayesian_priors.get("tau_fatigue_decay_prior", float("nan"))
        if math.isnan(raw_tau2):
            tau_2 = _TAU2_REF_DAYS
        else:
            tau_2 = max(1.0, float(raw_tau2))

        # ── 3. Phase-3 prior injection — MYH7 → k_fat ────────────────────────
        raw_kfat = bayesian_priors.get("myh7_fiber_type_prior", float("nan"))
        if math.isnan(raw_kfat):
            k_fat_athlete = _K_FAT_REF
        else:
            k_fat_athlete = max(0.005, _K_FAT_REF * float(raw_kfat))

        # ── 4. Build parameter structs ────────────────────────────────────────
        xia_params = AcuteXiaParams(
            f_max      = _F_MAX_REF,
            k_stim     = _K_STIM_REF,
            P_ref      = _P_REF_W,
            k_fat      = k_fat_athlete,
            k_rec      = _K_REC_REF,
            k_rec_fat  = _K_REC_FAT_REF,
            k_hill_fat = _K_HILL_FAT_REF,
            k_dmg      = _K_DMG_REF,
            k_repair   = _K_REPAIR_REF,
        )

        busso_params = ChronicBussoParams(
            k1       = _K1_REF,
            k2_basal = _K2_BASAL_REF,
            k3       = _K3_REF,
            tau_1    = _TAU1_REF_DAYS,
            tau_2    = tau_2,
            tau_3    = _TAU3_REF_DAYS,
            p_basal  = _P_BASAL_REF,
        )

        # ── 5. ACUTE ENGINE — Xia-Frey-Law 3-state ────────────────────────────
        acute_sol = _solve_acute_session(
            params       = xia_params,
            power_w      = power_w,
            t_sess_start = t_sess_start,
            sess_dur_min = sess_dur_min,
        )

        t_acute    = acute_sol.ts          # shape (n_save_acu,)
        M_act_traj = acute_sol.ys[:, 0]   # shape (n_save_acu,)
        M_fat_traj = acute_sol.ys[:, 1]   # shape (n_save_acu,)
        D_muscle   = acute_sol.ys[:, 2]   # shape (n_save_acu,)

        # ── 6. CROSS-SCALE BRIDGE — mechanical impulse w ──────────────────────
        w_dose = compute_mechanical_impulse(acute_sol)

        # ── 7. CHRONIC ENGINE — Busso 2003 ────────────────────────────────────
        chronic_sol = _solve_chronic_adaptation(
            params        = busso_params,
            w_dose_scalar = w_dose,
            t_sess_day    = t_sess_day,
            t_span_days   = t_span_days,
        )

        t_days = chronic_sol.ts
        G_fit  = chronic_sol.ys[:, 0]
        F_chr  = chronic_sol.ys[:, 1]
        k2_dyn = chronic_sol.ys[:, 2]
        P_net  = busso_params.p_basal + G_fit - F_chr

        # ── 8. Hub Variables (v1.2) ───────────────────────────────────────────
        # Hub_CK_StructuralDamage: D_muscle trajectory (fuga de CK → Módulo 7)
        hub_ck = D_muscle

        # Hub_FGF21_MetabolicStress: M_fat trajectory (stress metabólico → FGF21)
        hub_fgf21 = M_fat_traj

        # Hub_FGF23_BoneCoupling: positive rate of M_act DEPLETION
        # Finite-difference approximation of -dM_act/dt clamped to ≥0:
        # Phosphate is released when active fraction drops (cross-bridge detachment).
        M_act_diff = jnp.diff(M_act_traj)                          # shape (n_save-1,)
        depletion  = jnp.maximum(0.0, -M_act_diff)
        hub_fgf23  = jnp.concatenate([depletion, jnp.array([0.0])])  # shape (n_save_acu,)

        return {
            "t_days":                   t_days,
            "G_fitness":                G_fit,
            "F_chronic":                F_chr,
            "k2_dynamic":               k2_dyn,
            "Performance_Net":          P_net,
            "t_acute_min":              t_acute,
            "Hub_CK_StructuralDamage":  hub_ck,
            "Hub_FGF21_MetabolicStress": hub_fgf21,
            "Hub_FGF23_BoneCoupling":   hub_fgf23,
        }
