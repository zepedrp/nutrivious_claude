"""
app/engine/solvers/cardiorespiratory_solver.py

Módulo 3 — Sistema Cardiorrespiratório e Autonómico (MHDS Subsistema 3).

Integra 3 modelos físicos num sistema híbrido de 6 ODEs:
    I.  Dinâmica Autonómica Explícita (NE, E, V_vagal)
        Catecolaminas e Tónus Vagal como variáveis de estado explícitas
        (rejeição do "implicit autonomic tone"; Berntson et al. 1994)
    II. Modelo Windkessel de 4 Elementos (P_a — pressão aórtica)
        HR e débito cardíaco governados pelo balanço autonómico
    III.Controlo Quimiotáctico Ventilatório de Grodins (PaCO2, SpO2)
        V_E responde a hipercapnia e hipóxia (HIF-1α-modulada)

══════════════════════════════════════════════════════════════════════════════
SISTEMA DE 6 ODEs (unidades SI-biológicas mistas)
══════════════════════════════════════════════════════════════════════════════

Vetor de estado:  y = [NE, E, V_vagal, P_a, PaCO2, SpO2]

Estado    Variável              Unidade     Escala temporal
───────   ───────────────────   ──────────  ────────────────────────────────
y[0]      NE(t)   norepinefrina [adim.]     τ_NE  ≈ 7 min   (Christensen 1980)
y[1]      E(t)    epinefrina    [adim.]     τ_E   ≈ 8 min   (Clutter 1980)
y[2]      V_vagal tónus vagal   [adim., ∈(0,1)]   τ_vag ≈ 20 min (Arai 1989)
y[3]      P_a     pressão aórt. [mmHg]      τ_P   ≈ 0.3 min (Windkessel)
y[4]      PaCO2   arterial CO2  [mmHg]      τ_CO2 ≈ 5 min   (Grodins 1967)
y[5]      SpO2    sat. O2       [frac, ∈(0,1)]     τ_SpO2 ≈ 10 min

Rácio de rigidez: τ_P / τ_vag ≈ 0.3/20 = 0.015 → Kvaerno5 necessário.

──────────────────────────────────────────────────────────────────────────────
I. DINÂMICA AUTONÓMICA EXPLÍCITA (Berntson et al. 1994, J Auton Nerv Syst)
──────────────────────────────────────────────────────────────────────────────

ODE I — Norepinefrina (simpática pós-gangliônica):
  dNE/dt = k_NE_rel · (P_t / P_ref)  −  k_NE_clear · NE

  NE normalizada (adim.): NE_ss = k_NE_rel / k_NE_clear = 1.0 a P = P_ref.
  Estímulo linear em intensidade (activação simpática progressiva).

ODE II — Epinefrina (medula adrenal):
  dE/dt = k_E_rel · (P_t / P_ref)²  −  k_E_clear · E

  Estímulo quadrático: E só é libertada significativamente acima de ~50% P_ref
  (Clutter et al. 1980 NEJM: limiar de libertação de E ≈ 60% VO2max).

ODE III — Tónus Vagal (parassimpático):
  dV_vagal/dt = k_vagal_rec · (1.0 − V_vagal)  −  k_vagal_sup · (NE + E) · V_vagal

  Recuperação para 1.0 ao repouso (τ_rec = 1/k_vagal_rec ≈ 20 min).
  Supressão recíproca pelas catecolaminas (Berntson 1994 reciprocal model).
  Regime de exercício máximo: V_vagal → k_vagal_rec / (k_vagal_rec + k_vagal_sup·(NE+E)).

──────────────────────────────────────────────────────────────────────────────
II. WINDKESSEL 4 ELEMENTOS (Stergiopulos et al. 1999, AJP Heart)
──────────────────────────────────────────────────────────────────────────────

ODE IV — Pressão Arterial Média (P_a):
  dP_a/dt = (Q_heart(t) − P_a / R_total) / C_art

  Derivadas algébricas (sem estado adicional):
    HR_eff = HR_intr + k_chron_cat · (NE + k_E_chron · E) − k_chron_vag · V_vagal
        HR_intr = 110 bpm (nó sinoauricular sem tonus autonómico; Opthof 1988)
        Ao repouso (NE=0, E=0, V_vagal=1): HR = 110 + 0 − 50 = 60 bpm ✓

    SV_eff = SV_ref · (1 + k_inot · (NE + E)) · (1 + k_La_cardiac · La_hub)
        La_hub (Hub Inbound Módulo 1): lactato como combustível cardíaco atenua
        fadiga metabólica do miocárdio sob esforço (Gertz 1988 J Clin Invest).

    Q_heart = HR_eff · SV_eff                [L·min⁻¹]

    R_total = R_basal · ace_scale · (1 + k_vc · NE) / (1 + k_vd · P_t / P_ref)
        ace_scale ← ACE-INDEL prior (Fase 3): II → 0.85; DD → 1.15 (Rigat 1990)
        Vasoconstrição NE vs. vasodilatação metabólica durante exercício.

  Interpretação física (Bond Graph):
    Q_heart é o fluxo de potência mecânica cardíaca; P_a é o potencial de pressão;
    C_art armazena energia elástica nas paredes aórticas; R_total dissipa energia.
    dP_a/dt = 0 em repouso: Q_rest = P_a_rest / R_basal (equilíbrio MAP).

──────────────────────────────────────────────────────────────────────────────
III. CONTROLO QUIMIOTÁCTICO VENTILATÓRIO DE GRODINS (Grodins et al. 1967)
     Simplificação mono-compartimento de CO2 + proxy de SpO2
──────────────────────────────────────────────────────────────────────────────

ODE V — CO2 Arterial:
  dPaCO2/dt = k_CO2_prod · (1 + k_CO2_exer · P_t/P_ref)  −  k_CO2_elim · (V_E / V_E_basal) · PaCO2

  Produção: proporcional ao VCO2 metabólico (base + componente de exercício).
  Eliminação: proporcional a V_E × PaCO2 (lei de difusão alveolar, Grodins 1967).

  V_E (derivada algébrica, resposta quimiorreflexa):
    V_E = V_E_basal
          + k_VE_CO2 · max(0, PaCO2 − PaCO2_set)          [resposta hipercápnica]
          + k_VE_hyp · hif1a_scale · max(0, SpO2_set − SpO2)  [resposta hipóxica]
    hif1a_scale ← HIF1A-Pro582Ser prior (Fase 3): Ser allele → HIF-1α mais estável
                   → maior VHR (ventilatory hypoxic response); Bruick & McKnight 2001.

ODE VI — Saturação de O2:
  dSpO2/dt = k_SpO2_decay · (SpO2_basal − k_SpO2_exer · P_t/P_ref − SpO2)
             + k_SpO2_VE · max(0, V_E / V_E_basal − 1.0)

  Setpoint de SpO2 decresce com P_t (VO2 sobe, O2 extracção muscular aumenta).
  V_E acima do basal melhora SpO2 (ventilação compensa extracção periférica).

══════════════════════════════════════════════════════════════════════════════
VARIÁVEIS HUB — Bond Graph de Acoplamento
══════════════════════════════════════════════════════════════════════════════

  HUB INBOUND:
    Hub_Lactate_Signalling (Módulo 1 → Módulo 3):
        La_hub atenua fadiga cardíaca: SV_eff × (1 + k_La_cardiac × La_hub)
        Sustenta Q_heart durante hipoglicémia de esforço (Gertz 1988).

  HUB OUTBOUND:
    Hub_Catecholamines_Tone (→ Módulo 4 Sono, → Módulo 5 HPA):
        NE + E: activa LOCUS COERULEUS → fragmenta SWS (Módulo 4)
    Hub_Vagal_Tone (→ Módulo 4 Sono, → Módulo 9 Cognitivo):
        V_vagal: correlacionado com RMSSD; suprimido → HRV↓ → risco parasónia

══════════════════════════════════════════════════════════════════════════════
SOLVER E STACK
══════════════════════════════════════════════════════════════════════════════

Solver:    diffrax.Kvaerno5 (DIRK 5ª ordem, stiff-safe)
Step ctrl: diffrax.PIDController (rtol=1e-4, atol=1e-6)
dt0:       0.05 min (3 s) — captura transiente rápido de P_a (τ ≈ 0.3 min)
max_steps: 32 768

Referências:
    Berntson G.G. et al. (1994) J Auton Nerv Syst 49:S121–S132  [autonomic model]
    Grodins F.S. et al. (1967) J Appl Physiol 22:260–276       [CO2 ventilatory]
    Opthof T. (1988) Cardiovasc Drugs Ther 1:573–597           [SA node intrinsic]
    Stergiopulos N. et al. (1999) AJP Heart 276:H81–H88        [4-element Windkessel]
    Rigat B. et al. (1990) J Clin Invest 86:1343–1346          [ACE-INDEL]
    Bruick R.K. & McKnight S.L. (2001) Science 294:1337         [HIF1A-Pro582Ser]
    Gertz E.W. et al. (1988) J Clin Invest 81:1924–1929        [cardiac lactate fuel]
    Christensen N.J. (1980) Acta Physiol Scand 109:325–333     [NE τ clearance]
    Clutter W.E. et al. (1980) NEJM 303:893–898                [E threshold]
    Arai Y. et al. (1989) Circulation 79:86–93                 [vagal recovery]
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

# ── Reference constants ───────────────────────────────────────────────────────
# Autonomic / Catecholamines
_K_NE_REL_REF: float   = 0.15   # NE release rate per P_norm [min⁻¹]
_K_NE_CLEAR_REF: float = 0.15   # NE clearance [min⁻¹]; τ ≈ 7 min (Christensen 1980)
_K_E_REL_REF: float    = 0.05   # E release rate per P_norm² [min⁻¹]
_K_E_CLEAR_REF: float  = 0.12   # E clearance [min⁻¹]; τ ≈ 8 min (Clutter 1980)
_K_VAGAL_REC_REF: float = 0.050 # Vagal recovery [min⁻¹]; τ ≈ 20 min (Arai 1989)
_K_VAGAL_SUP_REF: float = 0.30  # Vagal suppression by catecholamines [adim.]

# Windkessel / Hemodynamics
_HR_INTR_REF: float    = 110.0  # SA node intrinsic rate [bpm] (Opthof 1988)
_K_CHRON_CAT_REF: float = 30.0  # Chronotropy per unit (NE+E) [bpm]
_K_E_CHRON_REF: float  = 1.5   # E:NE chronotropy ratio [adim.]; E > NE on β1
_K_CHRON_VAG_REF: float = 50.0  # Vagal chronotropy reduction [bpm]
                                  # Resting: 110 + 0 − 50×1 = 60 bpm ✓
_SV_REF: float         = 0.070  # Reference stroke volume [L]  (70 mL)
_K_INOT_REF: float     = 0.40   # Inotropic catecholamine scaling [adim.]
_K_LA_CARDIAC_REF: float = 0.10 # Lactate hub cardiac protection [adim.]
_C_ART_REF: float      = 0.012  # Arterial compliance [L·mmHg⁻¹]
_R_BASAL_REF: float    = 18.0   # Peripheral resistance [mmHg·min·L⁻¹]
                                  # At rest: MAP = 5 L/min × 18 = 90 mmHg ✓
_K_VASOCONSTRICT_REF: float = 0.25  # NE vasoconstriction coefficient [adim.]
_K_VASODILATE_REF: float    = 0.45  # Metabolic vasodilation by exercise [adim.]

# Grodins / Ventilatory
_V_E_BASAL_REF: float  = 8.0    # Resting minute ventilation [L·min⁻¹]
_K_VE_CO2_REF: float   = 2.0    # Hypercapnic VR [L·min⁻¹·mmHg⁻¹] (Grodins 1967)
_K_VE_HYP_REF: float   = 40.0   # Hypoxic VR [L·min⁻¹·SpO2_unit⁻¹]
_PACO2_BASAL_REF: float = 40.0  # Resting arterial PCO2 [mmHg]
_PACO2_SET_REF: float  = 40.0   # CO2 chemostat setpoint [mmHg]
_K_CO2_PROD_REF: float = 8.0    # Resting CO2 production rate [mmHg·min⁻¹]
_K_CO2_EXER_REF: float = 0.50   # Exercise VCO2 multiplier [adim.]
_K_CO2_ELIM_REF: float = 0.20   # CO2 elimination [min⁻¹]; τ_CO2 = 1/0.2 = 5 min
_SPO2_BASAL_REF: float = 0.98   # Resting O2 saturation [frac]
_SPO2_SET_REF: float   = 0.95   # Hypoxic drive threshold [frac]
_K_SPO2_DECAY_REF: float = 0.10 # SpO2 relaxation rate [min⁻¹]; τ = 10 min
_K_SPO2_EXER_REF: float = 0.030 # Exercise SpO2 depression at P_ref [frac]
_K_SPO2_VE_REF: float  = 0.010  # V_E excess → SpO2 improvement [frac·min]
_P_REF_W: float        = 200.0  # Power normalisation [W]

# RMSSD reference (derived Hub output)
_RMSSD_REF_MS: float   = 35.0   # Population mean RMSSD [ms] (Malik et al. 1996)


# ── Parameter container ───────────────────────────────────────────────────────

class CardiorespiratoryParams(NamedTuple):
    """
    Parameter vector for the Cardiorespiratory and Autonomic ODE system — Módulo 3.

    Covers:
        — Autonomic catecholamine dynamics (NE, E, V_vagal)
        — Windkessel 4-element haemodynamics (P_a, Q_heart, HR_eff)
        — Grodins chemoreflex ventilatory control (PaCO2, SpO2, V_E)
        — Bayesian prior injection fields (ace_scale, hif1a_scale)
        — Hub inbound coupling (k_La_cardiac)

    All rates in [min⁻¹]; pressures in [mmHg]; flows in [L·min⁻¹]; volumes in [L].
    """
    # ── I. Autonomic catecholamines ───────────────────────────────────────────
    k_NE_rel:    float   # NE release per normalised power [min⁻¹]
    k_NE_clear:  float   # NE plasma clearance [min⁻¹]
    k_E_rel:     float   # E release per normalised power² [min⁻¹]
    k_E_clear:   float   # E plasma clearance [min⁻¹]
    k_vagal_rec: float   # Vagal tone recovery toward 1.0 [min⁻¹]
    k_vagal_sup: float   # Catecholamine suppression of vagal tone [adim.]

    # ── II. Windkessel / Haemodynamics ───────────────────────────────────────
    HR_intr:     float   # Intrinsic SA-node rate [bpm] (without autonomic input)
    k_chron_cat: float   # Catecholamine chronotropy [bpm per unit (NE+E)]
    k_E_chron:   float   # E:NE potency ratio for chronotropy [adim.]
    k_chron_vag: float   # Vagal chronotropy reduction [bpm per unit V_vagal]
    SV_ref:      float   # Reference stroke volume [L]
    k_inot:      float   # Catecholamine inotropic scaling [adim.]
    k_La_cardiac: float  # Hub_Lactate cardiac fuel protection [adim.]
    C_art:       float   # Arterial compliance [L·mmHg⁻¹]
    R_basal:     float   # Basal peripheral resistance [mmHg·min·L⁻¹]
    ace_scale:   float   # ACE-INDEL vascular resistance multiplier (Fase 3)
    k_vasoconstrict: float  # NE-driven vasoconstriction [adim.]
    k_vasodilate:    float  # Metabolic exercise vasodilation [adim.]

    # ── III. Grodins ventilatory control ─────────────────────────────────────
    V_E_basal:   float   # Resting minute ventilation [L·min⁻¹]
    k_VE_CO2:    float   # Hypercapnic ventilatory response [L·min⁻¹·mmHg⁻¹]
    k_VE_hyp:    float   # Hypoxic ventilatory response [L·min⁻¹ per SpO2 unit]
    hif1a_scale: float   # HIF-1α Pro582Ser scaling on hypoxic VR (Fase 3)
    PaCO2_basal: float   # Resting arterial PCO2 [mmHg]
    PaCO2_set:   float   # CO2 chemostat setpoint [mmHg]
    k_CO2_prod:  float   # Resting VCO2 rate [mmHg·min⁻¹]
    k_CO2_exer:  float   # Exercise VCO2 multiplier [adim.]
    k_CO2_elim:  float   # CO2 elimination rate coefficient [min⁻¹]
    SpO2_basal:  float   # Resting SpO2 [frac]
    SpO2_set:    float   # Hypoxic drive threshold [frac]
    k_SpO2_decay: float  # SpO2 relaxation rate [min⁻¹]
    k_SpO2_exer:  float  # Exercise SpO2 depression at P_ref [frac]
    k_SpO2_VE:    float  # V_E excess → SpO2 benefit [frac·min]
    P_ref:        float  # Power normalisation [W]
    k_PVdrop_SV:  float  # Cardiovascular drift: PV drop [%] → SV reduction (Coyle 1986)


# ── Pure ODE (JIT-compilable) ─────────────────────────────────────────────────

def cardiorespiratory_ode(t, y, args):
    """
    Cardiorespiratory 6-state ODE — Módulo 3.

    State vector
    ------------
    y[0]  NE       : normalised norepinephrine              [adim.]
    y[1]  E        : normalised epinephrine                 [adim.]
    y[2]  V_vagal  : vagal (parasympathetic) tone           [adim., ∈ (0,1)]
    y[3]  P_a      : mean arterial pressure                 [mmHg]
    y[4]  PaCO2    : arterial CO2 partial pressure          [mmHg]
    y[5]  SpO2     : arterial O2 saturation                 [frac]

    args
    ----
    (CardiorespiratoryParams, power_w, t_sess_start, sess_dur_min, La_hub)
        La_hub : scalar float — Hub_Lactate from Module 1 at the relevant
                 time point (mean over session, passed as a constant scalar)
    """
    params, power_w, t_sess_start, sess_dur_min, La_hub, pv_drop_interp = args

    NE      = y[0]
    E       = y[1]
    V_vagal = y[2]
    P_a     = y[3]
    PaCO2   = y[4]
    SpO2    = y[5]

    # ── Smooth exercise gate ──────────────────────────────────────────────────
    def _smooth_on(t0):
        return 0.5 * (1.0 + jnp.tanh(20.0 * (t - t0)))

    sess_gate = _smooth_on(t_sess_start) - _smooth_on(t_sess_start + sess_dur_min)
    P_t = power_w * sess_gate
    P_norm = P_t / params.P_ref

    # ══════════════════════════════════════════════════════════════════════════
    # I. CATECHOLAMINE DYNAMICS
    # ══════════════════════════════════════════════════════════════════════════

    # dNE/dt = k_NE_rel·P_norm − k_NE_clear·NE   [sympathetic drive, linear]
    dNE_dt = params.k_NE_rel * P_norm - params.k_NE_clear * NE

    # dE/dt = k_E_rel·P_norm² − k_E_clear·E  [adrenal medulla; quadratic — high-threshold]
    dE_dt = params.k_E_rel * P_norm * P_norm - params.k_E_clear * E

    # dV_vagal/dt = k_vagal_rec·(1−V_vagal) − k_vagal_sup·(NE+E)·V_vagal
    # Reciprocal autonomic model (Berntson 1994): catecholamines suppress vagal tone.
    dV_vagal_dt = (
        params.k_vagal_rec * (1.0 - V_vagal)
        - params.k_vagal_sup * (NE + E) * V_vagal
    )

    # ══════════════════════════════════════════════════════════════════════════
    # II. WINDKESSEL HAEMODYNAMICS
    # ══════════════════════════════════════════════════════════════════════════

    # Effective HR: intrinsic + catecholamine chronotropy − vagal brake
    HR_eff = (
        params.HR_intr
        + params.k_chron_cat * (NE + params.k_E_chron * E)
        - params.k_chron_vag * V_vagal
    )
    HR_eff = jnp.maximum(30.0, HR_eff)  # physiological floor [bpm]

    # Stroke volume: inotropy by catecholamines + lactate hub cardiac protection
    # La_hub > 0 → heart uses lactate as fuel → SV sustained under fatigue
    # Cardiovascular drift (Coyle & Gonzalez-Alonso 2001): plasma volume drop
    # reduces ventricular filling (Frank-Starling) → SV falls progressively
    PV_drop_pct = jnp.clip(pv_drop_interp.evaluate(t), 0.0, 30.0)
    drift_factor = jnp.maximum(1.0 - params.k_PVdrop_SV * PV_drop_pct / 100.0, 0.5)
    SV_eff = (
        params.SV_ref
        * (1.0 + params.k_inot * (NE + E))
        * (1.0 + params.k_La_cardiac * La_hub)
        * drift_factor
    )

    # Cardiac output [L·min⁻¹]
    Q_heart = HR_eff * SV_eff

    # Total peripheral resistance: ACE-modulated basal × NE vasoconstriction / metabolic dilation
    R_total = (
        params.R_basal
        * params.ace_scale
        * (1.0 + params.k_vasoconstrict * NE)
        / (1.0 + params.k_vasodilate * P_norm)
    )
    R_total = jnp.maximum(5.0, R_total)   # floor [mmHg·min·L⁻¹]

    # dP_a/dt = (Q_heart − P_a/R_total) / C_art
    dP_a_dt = (Q_heart - P_a / R_total) / params.C_art

    # ══════════════════════════════════════════════════════════════════════════
    # III. GRODINS CHEMOREFLEX — CO2 and SpO2
    # ══════════════════════════════════════════════════════════════════════════

    # Minute ventilation (algebraic chemoreflex response):
    #   V_E = V_E_basal + k_VE_CO2 · max(0, PaCO2−set) + k_VE_hyp·hif1a · max(0, SpO2_set−SpO2)
    # hif1a_scale: HIF-1α Pro582Ser allele → greater hypoxic ventilatory response
    V_E = (
        params.V_E_basal
        + params.k_VE_CO2 * jnp.maximum(0.0, PaCO2 - params.PaCO2_set)
        + params.k_VE_hyp * params.hif1a_scale * jnp.maximum(0.0, params.SpO2_set - SpO2)
    )
    V_E = jnp.maximum(params.V_E_basal * 0.5, V_E)  # min ventilation floor

    # dPaCO2/dt = VCO2_rate − CO2 elimination via V_E
    # VCO2 = k_CO2_prod × (1 + k_CO2_exer × P_norm)
    # Elimination = k_CO2_elim × (V_E / V_E_basal) × PaCO2
    VCO2_rate = params.k_CO2_prod * (1.0 + params.k_CO2_exer * P_norm)
    V_E_norm  = V_E / params.V_E_basal
    dPaCO2_dt = VCO2_rate - params.k_CO2_elim * V_E_norm * PaCO2

    # dSpO2/dt = decay toward exercise-adapted setpoint + V_E benefit
    # Setpoint falls with exercise (increasing O2 extraction by active muscles)
    SpO2_target = params.SpO2_basal - params.k_SpO2_exer * P_norm
    SpO2_target = jnp.maximum(0.88, SpO2_target)  # physiological floor
    dSpO2_dt = (
        params.k_SpO2_decay * (SpO2_target - SpO2)
        + params.k_SpO2_VE * jnp.maximum(0.0, V_E_norm - 1.0)
    )

    return jnp.stack([dNE_dt, dE_dt, dV_vagal_dt, dP_a_dt, dPaCO2_dt, dSpO2_dt])


# ── JIT kernel ────────────────────────────────────────────────────────────────

@jax.jit
def _solve_cardiorespiratory(
    y0: jax.Array,
    t0: jax.Array,
    t1: jax.Array,
    dt0: jax.Array,
    ts: jax.Array,
    args: tuple,
) -> diffrax.Solution:
    """
    JIT-compiled integrator for the cardiorespiratory 6-state ODE.

    dt0=0.05 min (3 s) — mandatory for P_a transient (τ_P ≈ 0.3 min).
    PIDController handles wide stiffness range: τ_P=0.3 min vs τ_vag=20 min.
    """
    return diffrax.diffeqsolve(
        terms=diffrax.ODETerm(cardiorespiratory_ode),
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


# ── Orchestrator ──────────────────────────────────────────────────────────────

class CardiorespiratorySolver:
    """
    Orchestrator for MHDS Subsistema 3 — Cardiorespiratory and Autonomic.

    Phase-3 prior injections
    ------------------------
    ace_indel_prior   (ACE-INDEL → ace_scale):
        II genotype → ace_scale ≈ 0.85 (lower ACE → lower Ang-II → lower R)
        DD genotype → ace_scale ≈ 1.15 (higher ACE → higher Ang-II → higher R)
        ID genotype → ace_scale ≈ 1.00 (reference)
        Rigat et al. (1990): ACE-INDEL explains ~47% of serum ACE variance.

    hif1a_prior (HIF1A-Pro582Ser → hif1a_scale):
        Pro/Pro (normal) → hif1a_scale = 1.00 (reference hypoxic VR)
        Ser allele → hif1a_scale > 1.0 (HIF-1α more stable under normoxia →
        greater hypoxic ventilatory response; Bruick & McKnight 2001)

    Hub Inbound
    -----------
    Hub_Lactate_Signalling (Módulo 1): La_hub (scalar, session-mean mmol/L above basal)
        → SV_eff × (1 + k_La_cardiac × La_hub) — cardiac substrate protection.

    Hub Outbound
    ------------
    Hub_Catecholamines_Tone → Módulo 4 (Sono): NE+E activates locus coeruleus
    Hub_Vagal_Tone           → Módulo 4 (Sono), Módulo 9 (Cognitivo): RMSSD proxy
    """

    # ── I. Autonomic references ───────────────────────────────────────────────
    _K_NE_REL_REF: float    = _K_NE_REL_REF
    _K_NE_CLEAR_REF: float  = _K_NE_CLEAR_REF
    _K_E_REL_REF: float     = _K_E_REL_REF
    _K_E_CLEAR_REF: float   = _K_E_CLEAR_REF
    _K_VAGAL_REC_REF: float = _K_VAGAL_REC_REF
    _K_VAGAL_SUP_REF: float = _K_VAGAL_SUP_REF

    # ── II. Windkessel references ─────────────────────────────────────────────
    _HR_INTR_REF: float        = _HR_INTR_REF
    _K_CHRON_CAT_REF: float    = _K_CHRON_CAT_REF
    _K_E_CHRON_REF: float      = _K_E_CHRON_REF
    _K_CHRON_VAG_REF: float    = _K_CHRON_VAG_REF
    _SV_REF: float             = _SV_REF
    _K_INOT_REF: float         = _K_INOT_REF
    _K_LA_CARDIAC_REF: float   = _K_LA_CARDIAC_REF
    _C_ART_REF: float          = _C_ART_REF
    _R_BASAL_REF: float        = _R_BASAL_REF
    _K_VASOCONSTRICT_REF: float = _K_VASOCONSTRICT_REF
    _K_VASODILATE_REF: float   = _K_VASODILATE_REF

    # ── III. Grodins references ───────────────────────────────────────────────
    _V_E_BASAL_REF: float    = _V_E_BASAL_REF
    _K_VE_CO2_REF: float     = _K_VE_CO2_REF
    _K_VE_HYP_REF: float     = _K_VE_HYP_REF
    _PACO2_BASAL_REF: float  = _PACO2_BASAL_REF
    _PACO2_SET_REF: float    = _PACO2_SET_REF
    _K_CO2_PROD_REF: float   = _K_CO2_PROD_REF
    _K_CO2_EXER_REF: float   = _K_CO2_EXER_REF
    _K_CO2_ELIM_REF: float   = _K_CO2_ELIM_REF
    _SPO2_BASAL_REF: float   = _SPO2_BASAL_REF
    _SPO2_SET_REF: float     = _SPO2_SET_REF
    _K_SPO2_DECAY_REF: float = _K_SPO2_DECAY_REF
    _K_SPO2_EXER_REF: float  = _K_SPO2_EXER_REF
    _K_SPO2_VE_REF: float    = _K_SPO2_VE_REF
    _P_REF_W: float          = _P_REF_W
    _K_PVDROP_SV_REF: float  = 0.010  # per % PV drop → SV fraction reduction (Coyle 1986)

    def _build_params(
        self,
        bayesian_priors: dict[str, float],
    ) -> CardiorespiratoryParams:
        """
        Map Phase-3 Bayesian priors → CardiorespiratoryParams physical constants.

        Prior mappings
        --------------
        ace_indel_prior   → ace_scale (R_total multiplier; NaN → 1.0)
        hif1a_prior       → hif1a_scale (hypoxic VR scaling; NaN → 1.0)
        """
        # ACE-INDEL → peripheral resistance scaling
        ace_prior = bayesian_priors.get("ace_indel_prior", float("nan"))
        ace_scale = 1.0 if math.isnan(ace_prior) else max(0.5, float(ace_prior))

        # HIF1A-Pro582Ser → hypoxic ventilatory response
        hif1a_prior = bayesian_priors.get("hif1a_prior", float("nan"))
        hif1a_scale = 1.0 if math.isnan(hif1a_prior) else max(0.5, float(hif1a_prior))

        return CardiorespiratoryParams(
            # I. Autonomic
            k_NE_rel    = self._K_NE_REL_REF,
            k_NE_clear  = self._K_NE_CLEAR_REF,
            k_E_rel     = self._K_E_REL_REF,
            k_E_clear   = self._K_E_CLEAR_REF,
            k_vagal_rec = self._K_VAGAL_REC_REF,
            k_vagal_sup = self._K_VAGAL_SUP_REF,
            # II. Windkessel
            HR_intr         = self._HR_INTR_REF,
            k_chron_cat     = self._K_CHRON_CAT_REF,
            k_E_chron       = self._K_E_CHRON_REF,
            k_chron_vag     = self._K_CHRON_VAG_REF,
            SV_ref          = self._SV_REF,
            k_inot          = self._K_INOT_REF,
            k_La_cardiac    = self._K_LA_CARDIAC_REF,
            C_art           = self._C_ART_REF,
            R_basal         = self._R_BASAL_REF,
            ace_scale       = ace_scale,
            k_vasoconstrict = self._K_VASOCONSTRICT_REF,
            k_vasodilate    = self._K_VASODILATE_REF,
            # III. Grodins
            V_E_basal   = self._V_E_BASAL_REF,
            k_VE_CO2    = self._K_VE_CO2_REF,
            k_VE_hyp    = self._K_VE_HYP_REF,
            hif1a_scale = hif1a_scale,
            PaCO2_basal = self._PACO2_BASAL_REF,
            PaCO2_set   = self._PACO2_SET_REF,
            k_CO2_prod  = self._K_CO2_PROD_REF,
            k_CO2_exer  = self._K_CO2_EXER_REF,
            k_CO2_elim  = self._K_CO2_ELIM_REF,
            SpO2_basal  = self._SPO2_BASAL_REF,
            SpO2_set    = self._SPO2_SET_REF,
            k_SpO2_decay = self._K_SPO2_DECAY_REF,
            k_SpO2_exer  = self._K_SPO2_EXER_REF,
            k_SpO2_VE    = self._K_SPO2_VE_REF,
            P_ref        = self._P_REF_W,
            k_PVdrop_SV  = self._K_PVDROP_SV_REF,
        )

    def simulate_cardiorespiratory_response(
        self,
        bayesian_priors: dict[str, float],
        session_record: dict,
        hub_lactate_mmol_L: float = 0.0,
        hub_plasma_volume_drop_pct_arr=None,  # Mod 10: Hub_Plasma_Volume_Drop_Pct [%]
        hub_pv_t_min=None,                    # time axis for PV_drop [min]
        t_span_hours: float = 4.0,
        n_save_points: int = 240,
    ) -> dict[str, jax.Array]:
        """
        Simulate cardiorespiratory and autonomic dynamics for a training session.

        Parameters
        ----------
        bayesian_priors : dict[str, float]
            Phase-3 output. Keys consumed:
              "ace_indel_prior"  — ACE-INDEL R_total multiplier [adim.]
              "hif1a_prior"      — HIF1A hypoxic VR scaling [adim.]
        session_record : dict
            Phase-4 telemetry. Keys consumed:
              "power_output_watts"    — mean session power [W]
              "session_duration_secs" — duration [s]
            Optional:
              "session_start_min"     — session onset in window [min] (def. 30)
        hub_lactate_mmol_L : float
            Hub_Lactate_Signalling from Module 1 (session-mean La above basal [mmol·L⁻¹]).
            Default 0.0 (no lactate coupling). Positive values protect cardiac SV.
        t_span_hours : float
            Simulation horizon [hours] (default 4h).
        n_save_points : int
            Output time points (default 240 → 1 min resolution over 4h).

        Returns
        -------
        dict with JAX arrays shape (n_save_points,):
            "t_min"                     : time axis [min]
            "NE_tone"                   : normalised norepinephrine
            "E_tone"                    : normalised epinephrine
            "Vagal_tone"                : parasympathetic tone [adim.]
            "P_a_mmHg"                  : mean arterial pressure [mmHg]
            "PaCO2_mmHg"                : arterial CO2 [mmHg]
            "SpO2_frac"                 : oxygen saturation [frac]
            "HR_bpm"                    : derived effective heart rate [bpm]
            "V_E_L_min"                 : derived minute ventilation [L·min⁻¹]
            "RMSSD_proxy_ms"            : RMSSD approximation [ms]
            "Hub_Catecholamines_Tone"   : NE + E (→ Módulo 4, Módulo 5)
            "Hub_Vagal_Tone"            : V_vagal (→ Módulo 4, Módulo 9)
        """
        params = self._build_params(bayesian_priors)

        # ── Extract session inputs ────────────────────────────────────────────
        power_w      = float(session_record.get("power_output_watts", 0.0))
        dur_secs     = float(session_record.get("session_duration_secs", 0.0))
        sess_dur_min = dur_secs / 60.0
        t_sess_start = float(session_record.get("session_start_min", 30.0))
        La_hub       = float(hub_lactate_mmol_L)

        # ── Plasma volume drop interpolator (Cardiovascular Drift) ─────────────
        t1_min = t_span_hours * 60.0
        if hub_plasma_volume_drop_pct_arr is not None and hub_pv_t_min is not None:
            pv_t   = jnp.asarray(hub_pv_t_min,                dtype=jnp.float32)
            pv_arr = jnp.asarray(hub_plasma_volume_drop_pct_arr, dtype=jnp.float32)
        else:
            pv_t   = jnp.array([0.0, jnp.float32(t1_min)])
            pv_arr = jnp.array([0.0, 0.0], dtype=jnp.float32)
        pv_drop_interp = diffrax.LinearInterpolation(ts=pv_t, ys=pv_arr)

        # ── Basal initial conditions ──────────────────────────────────────────
        # NE=0, E=0 (no sympathetic at rest)
        # V_vagal=1.0 (full vagal tone at rest)
        # P_a = MAP at rest: Q_rest × R_basal = (HR_rest × SV_ref) × R_basal
        HR_rest  = params.HR_intr - params.k_chron_vag * 1.0   # ≈ 60 bpm
        Q_rest   = max(HR_rest, 30.0) * params.SV_ref           # ≈ 4.2 L/min
        P_a_rest = Q_rest * params.R_basal * params.ace_scale    # ≈ 90 mmHg
        y0 = jnp.array([
            0.0,                   # NE
            0.0,                   # E
            1.0,                   # V_vagal
            P_a_rest,              # P_a [mmHg]
            params.PaCO2_basal,    # PaCO2
            params.SpO2_basal,     # SpO2
        ], dtype=jnp.float32)

        # ── Time axis ─────────────────────────────────────────────────────────
        t0 = jnp.float32(0.0)
        t1 = jnp.float32(t_span_hours * 60.0)
        ts = jnp.linspace(t0, t1, n_save_points, dtype=jnp.float32)

        args = (
            params,
            jnp.float32(power_w),
            jnp.float32(t_sess_start),
            jnp.float32(sess_dur_min),
            jnp.float32(La_hub),
            pv_drop_interp,
        )

        # ── Solve ──────────────────────────────────────────────────────────────
        solution = _solve_cardiorespiratory(
            y0=y0,
            t0=t0,
            t1=t1,
            dt0=jnp.float32(0.05),   # 3-second initial step for fast P_a transient
            ts=ts,
            args=args,
        )

        ys      = solution.ys
        NE_arr  = ys[:, 0]
        E_arr   = ys[:, 1]
        V_vag   = ys[:, 2]
        P_a_arr = ys[:, 3]
        PaCO2   = ys[:, 4]
        SpO2    = ys[:, 5]

        # ── Derived outputs ───────────────────────────────────────────────────
        # HR_eff from saved NE, E, V_vagal trajectories
        HR_bpm = jnp.maximum(
            30.0,
            params.HR_intr
            + params.k_chron_cat * (NE_arr + params.k_E_chron * E_arr)
            - params.k_chron_vag * V_vag,
        )

        # V_E chemoreflex at each saved point
        V_E = (
            params.V_E_basal
            + params.k_VE_CO2 * jnp.maximum(0.0, PaCO2 - params.PaCO2_set)
            + params.k_VE_hyp * params.hif1a_scale * jnp.maximum(0.0, params.SpO2_set - SpO2)
        )

        # RMSSD proxy: V_vagal correlates with HRV (Arai 1989)
        RMSSD_proxy = _RMSSD_REF_MS * V_vag   # linear approximation [ms]

        # Hub Variables
        hub_cat  = NE_arr + E_arr   # catecholamine tone → Módulo 4/5
        hub_vag  = V_vag            # vagal tone → Módulo 4/9

        return {
            "t_min":                   ts,
            "NE_tone":                 NE_arr,
            "E_tone":                  E_arr,
            "Vagal_tone":              V_vag,
            "P_a_mmHg":               P_a_arr,
            "PaCO2_mmHg":             PaCO2,
            "SpO2_frac":               SpO2,
            "HR_bpm":                  HR_bpm,
            "V_E_L_min":               V_E,
            "RMSSD_proxy_ms":          RMSSD_proxy,
            "Hub_Catecholamines_Tone": hub_cat,
            "Hub_Vagal_Tone":          hub_vag,
        }
