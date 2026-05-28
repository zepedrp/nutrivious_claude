"""
Módulo 5 — Eixo Neuroendócrino HPA/HPG + Estado RED-S  (Completo)
Keenan-Veldhuis-inspired HPA kinetics + IOC 2023 EA ODE + Churilov HPG.
5-state ODE: [CRH, ACTH, Cort, EA_norm, Testo]
Hub Inbound : SCN_phase_x (Mod 4), Hub_Catecholamines_Tone (Mod 3),
              Hub_Lactate_Signalling (Mod 1), Hub_Sleep_Debt_Metabolic (Mod 4, scalar),
              Hub_GH_Repair_Signalling (Mod 4, post-process only)
Hub Outbound: Hub_Cortisol_Catabolic, Hub_Testosterone_Anabolic, Hub_REDS_State
NOTE: HPG suppression is purely central GnRH inhibition by Cortisol + RED-S.
      Pregnenolone steal is biochemically false and is NOT implemented.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

# ---------------------------------------------------------------------------
# Reference constants — HPA
# ---------------------------------------------------------------------------
_K_CRH_BAS_REF: float = 2.5          # h⁻¹ — baseline CRH production (normalized SS ≈ 1.0)
_K_CAR_REF: float = 0.3              # adim — circadian CAR drive amplitude (Clow 2004)
_K_CAT_CRH_REF: float = 0.4         # adim — catecholamine→CRH stress drive (Chrousos 1992)
_K_SD_CRH_REF: float = 0.3          # adim — sleep debt→CRH basal elevation (Spiegel 1999)
_K_LA_BLUNT_REF: float = 0.5        # [La]⁻¹ — lactate HPA blunting factor (Barron 2001)
_K_CRH_NEG_FB_REF: float = 0.3     # [Cort]⁻¹ — Cortisol→CRH negative feedback
_K_CRH_CLEAR_REF: float = 2.0       # h⁻¹ — CRH clearance (t½ ≈ 20 min; Veldhuis 1999)
_K_ACTH_SYNTH_REF: float = 3.12     # h⁻¹ — ACTH synthesis rate per unit CRH
_K_ACTH_NEG_FB_REF: float = 0.3    # [Cort]⁻¹ — Cortisol→ACTH negative feedback
_K_ACTH_CLEAR_REF: float = 2.4      # h⁻¹ — ACTH clearance (t½ ≈ 17 min; Veldhuis 1999)
_K_CORT_SYNTH_REF: float = 0.693    # h⁻¹ — Cortisol synthesis per unit ACTH
_K_CORT_CLEAR_REF: float = 0.693    # h⁻¹ — Cortisol clearance (t½ ≈ 1 h; Veldhuis 1999)
_K_REDS_CORT_REF: float = 0.5       # adim — RED-S energy deficit→CRH amplification

# ---------------------------------------------------------------------------
# Reference constants — EA / RED-S
# ---------------------------------------------------------------------------
_TAU_EA_REF: float = 48.0            # h — EA homeostatic smoothing (2-day time constant)
_EA_OPTIMAL_REF: float = 45.0        # kcal/kg FFM/day — optimal EA setpoint (IOC 2023)
_EA_THRESHOLD_REF: float = 30.0      # kcal/kg FFM/day — RED-S onset threshold (IOC 2023)

# ---------------------------------------------------------------------------
# Reference constants — HPG
# ---------------------------------------------------------------------------
_K_T_SYNTH_REF: float = 0.14         # h⁻¹ — Testosterone synthesis (calibrated SS=1.0 at rest)
_K_T_CLEAR_REF: float = 0.07         # h⁻¹ — Testosterone clearance (t½ ≈ 10 h; Vermeulen 1996)
_K_GNRH_CORT_REF: float = 1.0       # [Cort]⁻¹ — Cortisol→GnRH central inhibition (Kalra 1993)
_K_GNRH_REDS_REF: float = 3.0       # adim — RED-S→GnRH suppression factor (Loucks 2003)
_K_GH_SYNERGY_REF: float = 0.5      # adim — GH×Testosterone anabolic synergy (Cuneo 1992)


# ---------------------------------------------------------------------------
# Parameter container — HPA + EA + HPG (complete)
# ---------------------------------------------------------------------------
class NeuroendocrineParams(NamedTuple):
    # HPA axis
    k_CRH_bas: float      # baseline CRH production rate [h⁻¹]
    k_CAR: float          # circadian CAR drive amplitude [adim.]
    k_Cat_CRH: float      # catecholamine→CRH stress drive [adim.]
    k_SD_CRH: float       # sleep debt→CRH basal elevation [adim.]
    k_La_blunt: float     # lactate HPA blunting coefficient [[La]⁻¹]
    k_CRH_neg_fb: float   # Cortisol→CRH negative feedback [[Cort]⁻¹]
    k_CRH_clear: float    # CRH clearance rate [h⁻¹]
    k_ACTH_synth: float   # ACTH synthesis rate per unit CRH [h⁻¹]
    k_ACTH_neg_fb: float  # Cortisol→ACTH negative feedback [[Cort]⁻¹]
    k_ACTH_clear: float   # ACTH clearance rate [h⁻¹]
    k_Cort_synth: float   # Cortisol synthesis per unit ACTH [h⁻¹]
    k_Cort_clear: float   # Cortisol clearance rate [h⁻¹]
    k_REDS_Cort: float    # RED-S energy deficit→CRH amplification [adim.]
    # Energy Availability / RED-S
    tau_EA: float         # EA homeostatic smoothing time constant [h]
    EA_optimal: float     # optimal EA setpoint [kcal/kg FFM/day]
    EA_threshold: float   # RED-S onset threshold [kcal/kg FFM/day]
    # HPG axis
    k_T_synth: float      # Testosterone synthesis rate [h⁻¹]
    k_T_clear: float      # Testosterone clearance rate [h⁻¹]
    k_GnRH_Cort: float   # Cortisol→GnRH central inhibition [[Cort]⁻¹]
    k_GnRH_REDS: float   # RED-S→GnRH suppression factor [adim.]
    k_GH_synergy: float  # GH×Testosterone anabolic synergy [adim.]
    # Bayesian priors (Fase 3)
    nr3c1_scale: float    # NR3C1 GR sensitivity → neg-feedback strength [adim.; Bamberger 1996]
    ar_scale: float       # AR CAG length → tissue androgen efficacy [adim.; Chamberlain 1994]


# ---------------------------------------------------------------------------
# Pure ODE — 5-state HPA + EA + HPG (JAX JIT-compilable)
# ---------------------------------------------------------------------------
def neuroendocrine_ode(
    t: float,
    y: jnp.ndarray,
    args: tuple,
) -> jnp.ndarray:
    """
    y    = [CRH, ACTH, Cort, EA_norm, Testo]
    args = (NeuroendocrineParams,
            t_scn, scn_arr,            # SCN phase from Módulo 4
            t_cat, Cat_arr,            # Catecholamines from Módulo 3
            t_La,  La_arr,             # Lactate from Módulo 1
            sleep_debt,                # scalar JAX array from Módulo 4
            t_EA,  EA_norm_target_arr) # daily energy balance target [normalized]
    time unit: hours
    """
    (params, t_scn, scn_arr, t_cat, Cat_arr,
     t_La, La_arr, sleep_debt, t_EA, EA_norm_target_arr) = args

    CRH, ACTH, Cort, EA_norm, Testo = y

    # Interpolate hub inbounds
    x_scn     = jnp.interp(t, t_scn, scn_arr)
    Cat_tone  = jnp.interp(t, t_cat, Cat_arr)
    La_hub    = jnp.interp(t, t_La,  La_arr)
    EA_target = jnp.interp(t, t_EA,  EA_norm_target_arr)

    # RED-S gate: 1 = energy deficit, 0 = replete
    EA_crit   = params.EA_threshold / params.EA_optimal
    REDS_gate = 0.5 * (1.0 - jnp.tanh(15.0 * (EA_norm - EA_crit)))

    # Circadian CAR drive: peaks when x_scn → +1 (SCN morning activation)
    CAR_drive = jnp.maximum(0.0, (1.0 + x_scn) / 2.0)

    # Lactate HPA blunting — divides CRH production rate (Barron 2001)
    La_blunt = 1.0 / (1.0 + params.k_La_blunt * La_hub)

    # --- CRH ODE ---
    # Baseline + CAR + sympathoadrenal stress + sleep debt + RED-S energy alarm
    # Blunted by exercise lactate; inhibited by Cortisol neg. feedback (nr3c1-scaled)
    S_CRH = (
        (params.k_CRH_bas
         + params.k_CAR       * CAR_drive
         + params.k_Cat_CRH   * Cat_tone
         + params.k_SD_CRH    * sleep_debt
         + params.k_REDS_Cort * REDS_gate)
        * La_blunt
        / (1.0 + params.k_CRH_neg_fb * params.nr3c1_scale * Cort)
    )
    dCRH_dt = S_CRH - params.k_CRH_clear * CRH

    # --- ACTH ODE ---
    S_ACTH = (
        params.k_ACTH_synth * CRH
        / (1.0 + params.k_ACTH_neg_fb * params.nr3c1_scale * Cort)
    )
    dACTH_dt = S_ACTH - params.k_ACTH_clear * ACTH

    # --- Cortisol ODE ---
    dCort_dt = params.k_Cort_synth * ACTH - params.k_Cort_clear * Cort

    # --- Energy Availability (slow, 48h homeostatic smoothing) ---
    dEA_norm_dt = (1.0 / params.tau_EA) * (EA_target - EA_norm)

    # --- HPG: Testosterone — central GnRH inhibition (NOT pregnenolone steal) ---
    # Cortisol suppresses GnRH/LH pulsatility at the hypothalamo-pituitary level
    # (Kalra 1993; Breen & Karsch 2006). RED-S independently suppresses via
    # kisspeptin/GnRH neuron energy sensing (Loucks 2003; Tena-Sempere 2013).
    GnRH_eff = 1.0 / (
        (1.0 + params.k_GnRH_Cort * Cort)
        * (1.0 + params.k_GnRH_REDS * REDS_gate)
    )
    dTesto_dt = params.k_T_synth * GnRH_eff - params.k_T_clear * Testo

    return jnp.stack([dCRH_dt, dACTH_dt, dCort_dt, dEA_norm_dt, dTesto_dt])


# ---------------------------------------------------------------------------
# Part-1 compatibility shim — retained for unit tests; delegates to full ODE
# ---------------------------------------------------------------------------
def hpa_reds_ode(
    t: float,
    y: jnp.ndarray,
    args: tuple,
) -> jnp.ndarray:
    """4-state shim: pads Testo=1 and calls neuroendocrine_ode, returns first 4 components."""
    y5 = jnp.concatenate([y, jnp.array([1.0])])
    dy5 = neuroendocrine_ode(t, y5, args)
    return dy5[:4]


# ---------------------------------------------------------------------------
# JIT solve kernels
# ---------------------------------------------------------------------------
@jax.jit
def _solve_hpa_reds(
    params: NeuroendocrineParams,
    t_scn: jnp.ndarray,
    scn_arr: jnp.ndarray,
    t_cat: jnp.ndarray,
    Cat_arr: jnp.ndarray,
    t_La: jnp.ndarray,
    La_arr: jnp.ndarray,
    sleep_debt: jnp.ndarray,
    t_EA: jnp.ndarray,
    EA_norm_target_arr: jnp.ndarray,
    t_eval: jnp.ndarray,
    y0: jnp.ndarray,
) -> jnp.ndarray:
    """4-state kernel (Part 1 compatibility)."""
    term = diffrax.ODETerm(hpa_reds_ode)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=t_eval[0], t1=t_eval[-1], dt0=0.05,
        y0=y0,
        args=(params, t_scn, scn_arr, t_cat, Cat_arr,
              t_La, La_arr, sleep_debt, t_EA, EA_norm_target_arr),
        saveat=diffrax.SaveAt(ts=t_eval),
        stepsize_controller=controller,
        max_steps=100_000,
    )
    return sol.ys


@jax.jit
def _solve_neuroendocrine(
    params: NeuroendocrineParams,
    t_scn: jnp.ndarray,
    scn_arr: jnp.ndarray,
    t_cat: jnp.ndarray,
    Cat_arr: jnp.ndarray,
    t_La: jnp.ndarray,
    La_arr: jnp.ndarray,
    sleep_debt: jnp.ndarray,
    t_EA: jnp.ndarray,
    EA_norm_target_arr: jnp.ndarray,
    t_eval: jnp.ndarray,
    y0: jnp.ndarray,
) -> jnp.ndarray:
    """5-state full HPA + EA + HPG kernel."""
    term = diffrax.ODETerm(neuroendocrine_ode)
    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=t_eval[0], t1=t_eval[-1],
        dt0=0.05,         # 3-min initial step for fast CRH/ACTH kinetics
        y0=y0,
        args=(params, t_scn, scn_arr, t_cat, Cat_arr,
              t_La, La_arr, sleep_debt, t_EA, EA_norm_target_arr),
        saveat=diffrax.SaveAt(ts=t_eval),
        stepsize_controller=controller,
        max_steps=100_000,
    )
    return sol.ys


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------
class NeuroendocrineSolver:
    def __init__(self, genetic_profile: dict | None = None) -> None:
        self._genetic = genetic_profile or {}

    def _build_params(self) -> NeuroendocrineParams:
        g = self._genetic

        # NR3C1: glucocorticoid receptor sensitivity [0 = low, 1 = high activity]
        # Higher sensitivity → stronger neg. feedback → lower chronic cortisol
        nr3c1_raw = g.get("nr3c1_prior", float("nan"))
        nr3c1_scale = 1.0 if math.isnan(nr3c1_raw) else 1.0 + 0.3 * nr3c1_raw

        # AR CAG repeat: shorter repeat → higher transcriptional activity
        # Prior convention: 0.0 = short/active, 1.0 = long/attenuated
        # ar_scale modulates Hub_Testosterone_Anabolic (tissue efficacy, not clearance)
        ar_raw = g.get("ar_prior", float("nan"))
        if math.isnan(ar_raw):
            ar_scale = 1.0
        else:
            ar_scale = 1.0 + 0.4 * (0.5 - ar_raw)  # short CAG → ar_scale > 1

        return NeuroendocrineParams(
            k_CRH_bas=_K_CRH_BAS_REF,
            k_CAR=_K_CAR_REF,
            k_Cat_CRH=_K_CAT_CRH_REF,
            k_SD_CRH=_K_SD_CRH_REF,
            k_La_blunt=_K_LA_BLUNT_REF,
            k_CRH_neg_fb=_K_CRH_NEG_FB_REF,
            k_CRH_clear=_K_CRH_CLEAR_REF,
            k_ACTH_synth=_K_ACTH_SYNTH_REF,
            k_ACTH_neg_fb=_K_ACTH_NEG_FB_REF,
            k_ACTH_clear=_K_ACTH_CLEAR_REF,
            k_Cort_synth=_K_CORT_SYNTH_REF,
            k_Cort_clear=_K_CORT_CLEAR_REF,
            k_REDS_Cort=_K_REDS_CORT_REF,
            tau_EA=_TAU_EA_REF,
            EA_optimal=_EA_OPTIMAL_REF,
            EA_threshold=_EA_THRESHOLD_REF,
            k_T_synth=_K_T_SYNTH_REF,
            k_T_clear=_K_T_CLEAR_REF,
            k_GnRH_Cort=_K_GNRH_CORT_REF,
            k_GnRH_REDS=_K_GNRH_REDS_REF,
            k_GH_synergy=_K_GH_SYNERGY_REF,
            nr3c1_scale=float(nr3c1_scale),
            ar_scale=float(ar_scale),
        )

    def simulate_neuroendocrine_response(
        self,
        duration_h: float = 24.0,
        hub_scn_phase: jnp.ndarray | None = None,
        t_scn_h: jnp.ndarray | None = None,
        hub_catecholamines_tone: jnp.ndarray | None = None,
        t_cat_h: jnp.ndarray | None = None,
        hub_lactate_signalling: jnp.ndarray | None = None,
        t_La_h: jnp.ndarray | None = None,
        hub_sleep_debt: float = 0.0,
        hub_gh_repair: jnp.ndarray | None = None,
        t_gh_h: jnp.ndarray | None = None,
        energy_intake_kcal_per_kg_ffm: float = 45.0,
        exercise_kcal_per_kg_ffm: float = 0.0,
        EA_init_norm: float = 1.0,
        n_save: int = 288,
    ) -> dict:
        """
        Simulate full HPA/HPG neuroendocrine axis for one day.

        Parameters
        ----------
        hub_scn_phase : SCN_phase_x array from Módulo 4.
        hub_catecholamines_tone : NE+E tone from Módulo 3.
        hub_lactate_signalling : La_hub array from Módulo 1.
        hub_sleep_debt : scalar sleep-debt from Módulo 4.
        hub_gh_repair : GH_Repair array from Módulo 4 (anabolic synergy, post-process).
        energy_intake_kcal_per_kg_ffm : daily intake [kcal/kg FFM].
        exercise_kcal_per_kg_ffm : exercise expenditure [kcal/kg FFM].
        EA_init_norm : initial EA state (1.0 = optimal, <0.667 = RED-S).

        Returns
        -------
        dict: CRH, ACTH, Cortisol, EA, Testosterone arrays + Hub outbound keys.
        """
        params = self._build_params()
        t_eval = jnp.linspace(0.0, duration_h, n_save)

        # SCN phase (Módulo 4)
        if hub_scn_phase is None or t_scn_h is None:
            t_scn   = jnp.array([0.0, duration_h])
            scn_arr = jnp.zeros(2)
        else:
            t_scn   = t_scn_h
            scn_arr = hub_scn_phase

        # Catecholamines (Módulo 3)
        if hub_catecholamines_tone is None or t_cat_h is None:
            t_cat   = jnp.array([0.0, duration_h])
            Cat_arr = jnp.zeros(2)
        else:
            t_cat   = t_cat_h
            Cat_arr = hub_catecholamines_tone

        # Lactate (Módulo 1)
        if hub_lactate_signalling is None or t_La_h is None:
            t_La   = jnp.array([0.0, duration_h])
            La_arr = jnp.zeros(2)
        else:
            t_La   = t_La_h
            La_arr = hub_lactate_signalling

        # EA target normalized to EA_optimal
        EA_target_norm = (
            (energy_intake_kcal_per_kg_ffm - exercise_kcal_per_kg_ffm)
            / params.EA_optimal
        )
        t_EA               = jnp.array([0.0, duration_h])
        EA_norm_target_arr = jnp.array([EA_target_norm, EA_target_norm])

        sleep_debt_jnp = jnp.array(float(hub_sleep_debt))

        # Initial conditions: normalized basal (all axes at rest = 1.0)
        y0 = jnp.array([1.0, 1.0, 1.0, float(EA_init_norm), 1.0])

        ys = _solve_neuroendocrine(
            params,
            t_scn, scn_arr,
            t_cat, Cat_arr,
            t_La,  La_arr,
            sleep_debt_jnp,
            t_EA, EA_norm_target_arr,
            t_eval, y0,
        )

        CRH_arr     = ys[:, 0]
        ACTH_arr    = ys[:, 1]
        Cort_arr    = ys[:, 2]
        EA_norm_arr = ys[:, 3]
        Testo_arr   = ys[:, 4]

        # RED-S gate (derived)
        EA_crit       = params.EA_threshold / params.EA_optimal
        REDS_gate_arr = 0.5 * (1.0 - jnp.tanh(15.0 * (EA_norm_arr - EA_crit)))

        # GH anabolic synergy (post-process; Cuneo 1992)
        if hub_gh_repair is not None and t_gh_h is not None:
            GH_interp      = jnp.interp(t_eval, t_gh_h, hub_gh_repair)
            GH_synergy_arr = 1.0 + params.k_GH_synergy * GH_interp
        else:
            GH_synergy_arr = jnp.ones(n_save)

        return {
            "t_h":               t_eval,
            "CRH_norm":          CRH_arr,
            "ACTH_norm":         ACTH_arr,
            "Cortisol_norm":     Cort_arr,
            "EA_norm":           EA_norm_arr,
            "Testosterone_norm": Testo_arr,
            "REDS_gate":         REDS_gate_arr,
            # Hub Outbound
            "Hub_Cortisol_Catabolic":    Cort_arr,                                    # → Módulo 7 (immunosuppression)
            "Hub_Testosterone_Anabolic": Testo_arr * params.ar_scale * GH_synergy_arr, # → Módulos 7, 11 (structural repair)
            "Hub_REDS_State":            REDS_gate_arr,                                # → Módulo 6 (thyroid suppression)
        }

    def simulate_hpa_reds(
        self,
        duration_h: float = 24.0,
        hub_scn_phase: jnp.ndarray | None = None,
        t_scn_h: jnp.ndarray | None = None,
        hub_catecholamines_tone: jnp.ndarray | None = None,
        t_cat_h: jnp.ndarray | None = None,
        hub_lactate_signalling: jnp.ndarray | None = None,
        t_La_h: jnp.ndarray | None = None,
        hub_sleep_debt: float = 0.0,
        energy_intake_kcal_per_kg_ffm: float = 45.0,
        exercise_kcal_per_kg_ffm: float = 0.0,
        EA_init_norm: float = 1.0,
        n_save: int = 288,
    ) -> dict:
        """Part 1 compatibility method — delegates to simulate_neuroendocrine_response."""
        result = self.simulate_neuroendocrine_response(
            duration_h=duration_h,
            hub_scn_phase=hub_scn_phase,
            t_scn_h=t_scn_h,
            hub_catecholamines_tone=hub_catecholamines_tone,
            t_cat_h=t_cat_h,
            hub_lactate_signalling=hub_lactate_signalling,
            t_La_h=t_La_h,
            hub_sleep_debt=hub_sleep_debt,
            energy_intake_kcal_per_kg_ffm=energy_intake_kcal_per_kg_ffm,
            exercise_kcal_per_kg_ffm=exercise_kcal_per_kg_ffm,
            EA_init_norm=EA_init_norm,
            n_save=n_save,
        )
        return {k: result[k] for k in
                ("t_h", "CRH_norm", "ACTH_norm", "Cortisol_norm", "EA_norm", "REDS_gate")}
