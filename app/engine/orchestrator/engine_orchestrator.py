"""
app/engine/orchestrator/engine_orchestrator.py

NutriviousEngine — Placa-Mãe do Motor BOS (MHDS v2.0 — Zero Proxies)

Ordem de Precedência Termodinâmica e Causal:
    BLOCO A  Mod 6 (Thyroid) → Mod 8 (GI)
    BLOCO B  Mod 1 (Metabolic) → Mod 2 (Neuromuscular) → Mod 3 (Cardiorespiratory)
    [conversão t_min → t_h]
    BLOCO C  Mod 10 (ThermoFluid) → Mod 5 (Neuroendocrine) → Mod 7 (ImmuneRepair)
    BLOCO D  Mod 4 (SleepCircadian) → Mod 11 (BiomechanicalTissue) → Mod 9 (CentralFatigue)

Conectoma Nativo (Zero Proxies — v2.0):
    • Hub_Core_Temp (Mod 10) → hub_core_temp_arr nativo em Mod 9 ODE (Nybo & Nielsen 2001)
    • Hub_FGF23_BoneCoupling (Mod 2) → hub_fgf23_arr nativo em Mod 11 ODE (Wnt/RANKL)
    • Hub_Vagal_Tone (Mod 3) + Hub_Skin_Blood_Flow (Mod 10) → hub_vagal_tone_arr +
      hub_skin_blood_flow_arr nativos em Mod 8 ODE (Enteric NS + Splanchnic Ischemia)
    • Hub_Plasma_Volume_Drop_Pct (Mod 10 dia anterior) → hub_plasma_volume_drop_pct_arr
      nativo em Mod 3 ODE (Frank-Starling Cardiovascular Drift; Coyle 1986)
    • Hub_Melatonin_Tone (Mod 4 dia anterior) → mel_interp nativo em Mod 7 ODE (NF-κB;
      Carrillo-Vico 2013) e Mod 10 ODE (peripheral vasodilation; Cagnacci 1992)

Dependências Circulares Resolvidas por Bootstrap:
    - Mod 8 precisa de SkBF do Mod 10 → usa previous-day SkBF do morning_state
    - Mod 3 precisa de PV_drop do Mod 10 → usa previous-day PV_drop do morning_state
    - Mod 7 precisa de Melatonin do Mod 4 → usa previous-day Mel do morning_state
    - Mod 10 precisa de Melatonin do Mod 4 → usa previous-day Mel do morning_state
"""
from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
import jax.numpy as jnp

from app.engine.solvers import (
    MetabolicSolver,
    NeuromuscularSolver,
    CardiorespiratorySolver,
    SleepCircadianSolver,
    NeuroendocrineSolver,
    ThermoFluidSolver,
    ImmuneRepairSolver,
    ThyroidBaselineSolver,
    GastrointestinalSolver,
    CentralFatigueSolver,
    BiomechanicalTissueSolver,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Type alias for the mega-dict returned by simulate_daily_cycle
# ─────────────────────────────────────────────────────────────────────────────
DigitalTwinState = dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zeros_h(t_h: jnp.ndarray) -> jnp.ndarray:
    """Return a zero array with the same shape as t_h."""
    return jnp.zeros_like(t_h)


def _min_to_h(arr_min: jnp.ndarray, t_min: jnp.ndarray, t_h: jnp.ndarray) -> jnp.ndarray:
    """
    Resample an array defined on t_min (minutes) onto t_h (hours).

    Uses JAX linear interpolation. Both axes must be monotone increasing.
    t_min values are converted to hours before interpolation.
    """
    t_min_as_h = t_min / 60.0
    return jnp.interp(t_h, t_min_as_h, arr_min)


def _scalar_to_arr(scalar: float, t_h: jnp.ndarray) -> jnp.ndarray:
    """Broadcast a scalar to a constant array on t_h."""
    return jnp.full_like(t_h, float(scalar))


def _safe_mean(arr) -> float:
    try:
        return float(jnp.mean(jnp.asarray(arr, dtype=jnp.float32)))
    except Exception:
        return 0.0


def _safe_last(arr) -> float:
    try:
        a = jnp.asarray(arr, dtype=jnp.float32)
        return float(a[-1])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class NutriviousEngine:
    """
    Orchestrates all 11 physiological solvers in strict causal precedence order.

    Usage
    -----
    engine = NutriviousEngine()
    state  = engine.simulate_daily_cycle(
        bayesian_priors       = priors_dict,
        morning_state         = yesterday_state_or_defaults,
        planned_session       = session_record_dict,
        meal_event            = meal_dict,
        fluid_intake          = fluid_dict,
        environmental_conditions = env_dict,
    )
    """

    def __init__(self) -> None:
        self.metabolic      = MetabolicSolver()
        self.neuromuscular  = NeuromuscularSolver()
        self.cardio         = CardiorespiratorySolver()
        self.sleep_circ     = SleepCircadianSolver()
        self.neuroendo      = NeuroendocrineSolver()
        self.thermo_fluid   = ThermoFluidSolver()
        self.immune         = ImmuneRepairSolver()
        self.thyroid        = ThyroidBaselineSolver()
        self.gi             = GastrointestinalSolver()
        self.central_fat    = CentralFatigueSolver()
        self.biomech        = BiomechanicalTissueSolver()

    # ─────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────

    def simulate_daily_cycle(
        self,
        bayesian_priors: dict,
        morning_state: dict,
        planned_session: dict,
        meal_event: dict | None = None,
        fluid_intake: dict | None = None,
        environmental_conditions: dict | None = None,
        t_span_exercise_h: float = 4.0,
        t_span_recovery_h: float = 24.0,
        n_save_exercise: int = 240,
        n_save_recovery: int = 512,
    ) -> DigitalTwinState:
        """
        Full daily simulation cycle. Returns DigitalTwinState containing all
        Hub Variables from all 11 modules plus meta-arrays.

        Parameters
        ----------
        bayesian_priors
            Phase-3 dict produced by compose_bayesian_priors().
        morning_state
            Previous-day terminal state (or default baseline). Used to break
            circular temporal dependencies (bootstrap values). Keys:
                "cortisol_norm"     : float  [normalized 0–3]
                "reds_state"        : float  [0–1]
                "scn_phase"         : float  [0–1]
                "sleep_debt"        : float  [normalized, ≥0]
                "gh_repair"         : float  [normalized ≥0]
                "ea_init_norm"      : float  [1.0 = optimal]
                "testosterone_norm" : float  [normalized]
        planned_session
            SessionRecord dict. Required keys:
                "power_output_watts"    : float
                "session_duration_secs" : float
            Optional:
                "session_start_min"     : float  (default 30 min after t0)
                "session_start_h"       : float  (for thermo solver)
                "t_sess_start_min"      : float
                "t_sess_day"            : float
        meal_event
            HydrationAndNutritionRecord subset:
                "carbohydrate_grams"   : float
                "meal_timestamp_min"   : float
                "fructose_grams"       : float  (optional)
                "protein_grams"        : float  (optional)
                "fat_grams"            : float  (optional)
                "water_ml"             : float  (optional)
                "sodium_mg"            : float  (optional)
        fluid_intake
            Dict consumed by ThermoFluidSolver (fallback values if Mod 8 rates absent):
                "water_intake_L"       : float
                "drink_timestamp_h"    : float
                "na_dietary_mg"        : float
                "na_meal_timestamp_h"  : float
        environmental_conditions
            Dict consumed by ThermoFluidSolver:
                "T_env_celsius"        : float
                "RH_fraction"          : float
                "wind_speed_m_s"       : float
                "T_core_init_celsius"  : float  (optional)
                "T_skin_init_celsius"  : float  (optional)
                "V_p_init_L"           : float  (optional)
                "light_lux"            : float  (optional, circadian photic drive)

        Returns
        -------
        DigitalTwinState — flat dict of all Hub arrays and meta information.
        """
        meal_event   = meal_event   or {}
        fluid_intake = fluid_intake or {}
        env_cond     = environmental_conditions or {}

        # ── Bootstrap values (break circular dependencies) ─────────────────
        boot_cortisol    = float(morning_state.get("cortisol_norm",     1.0))
        boot_reds        = float(morning_state.get("reds_state",         0.0))
        boot_scn         = float(morning_state.get("scn_phase",          0.5))
        boot_sleep_debt  = float(morning_state.get("sleep_debt",         0.0))
        boot_gh          = float(morning_state.get("gh_repair",          0.5))
        boot_ea_norm     = float(morning_state.get("ea_init_norm",       1.0))
        boot_testo       = float(morning_state.get("testosterone_norm",  1.0))
        # New bootstrap keys for native biological connections
        boot_pv_drop     = float(morning_state.get("pv_drop_pct",        0.0))
        boot_melatonin   = float(morning_state.get("melatonin_tone",     0.0))
        boot_skbf        = float(morning_state.get("skin_blood_flow_L_h", 3.0))

        # ── Common time axes ───────────────────────────────────────────────
        t_ex_h   = jnp.linspace(0.0, t_span_exercise_h,  n_save_exercise,  dtype=jnp.float32)
        t_rec_h  = jnp.linspace(0.0, t_span_recovery_h,  n_save_recovery,  dtype=jnp.float32)

        # Session energy for GI bootstrap (kcal/kg FFM estimado)
        power_w   = float(planned_session.get("power_output_watts",    150.0))
        dur_secs  = float(planned_session.get("session_duration_secs", 3600.0))
        ffm_kg    = float(bayesian_priors.get("fat_free_mass_kg",       70.0)) or 70.0
        gross_eff = 0.22
        sess_kcal = (power_w * dur_secs / gross_eff) / 4184.0
        exercise_kcal_per_kg_ffm = sess_kcal / ffm_kg

        state: DigitalTwinState = {}

        # ══════════════════════════════════════════════════════════════════
        # BLOCO A — Sistemas de Base e Absorção
        # ══════════════════════════════════════════════════════════════════

        # ── A1: Módulo 6 — Thyroid Baseline ───────────────────────────────
        # Needs: bootstrap Cortisol, bootstrap REDS (from yesterday)
        # Produces: Hub_Basal_Temperature_Morning, Hub_Thyroid_Anabolic_Tone,
        #           Hub_Basal_Metabolic_Rate_Kcal_Min, Hub_Resting_RQ
        try:
            boot_cort_arr = jnp.array([boot_cortisol, boot_cortisol], dtype=jnp.float32)
            boot_reds_arr = jnp.array([boot_reds,     boot_reds],     dtype=jnp.float32)
            boot_t_h      = jnp.array([0.0, t_span_recovery_h],       dtype=jnp.float32)

            thyroid_out = self.thyroid.simulate_thyroid_baseline(
                bayesian_priors      = bayesian_priors,
                hub_cortisol_arr     = boot_cort_arr,
                hub_reds_arr         = boot_reds_arr,
                t_hub_h              = boot_t_h,
                hub_catecholamines_arr = None,
                t_span_h             = (0.0, t_span_recovery_h),
            )
            state.update({k: v for k, v in thyroid_out.items()
                          if k.startswith("Hub_") or k == "t_h"})
            state["_thyroid_t_h"]       = thyroid_out["t_h"]
            state["_thyroid_t_tone"]    = thyroid_out["Hub_Thyroid_Anabolic_Tone"]
            state["_thyroid_basal_temp"]= float(_safe_mean(thyroid_out["Hub_Basal_Temperature_Morning"]))
            logger.debug("Mod 6 (Thyroid) OK — T_basal=%.2f°C", state["_thyroid_basal_temp"])
        except Exception as exc:
            logger.warning("Mod 6 (Thyroid) FAILED: %s — using bootstrap zeros", exc)
            state["Hub_Basal_Temperature_Morning"]     = _scalar_to_arr(36.6, t_rec_h)
            state["Hub_Thyroid_Anabolic_Tone"]         = _scalar_to_arr(1.0,  t_rec_h)
            state["Hub_Basal_Metabolic_Rate_Kcal_Min"] = _scalar_to_arr(1.2,  t_rec_h)
            state["Hub_Resting_RQ"]                    = _scalar_to_arr(0.85, t_rec_h)
            state["_thyroid_t_h"]       = t_rec_h
            state["_thyroid_t_tone"]    = _scalar_to_arr(1.0, t_rec_h)
            state["_thyroid_basal_temp"]= 36.6

        # ── A2: Módulo 8 — Gastrointestinal ───────────────────────────────
        # Needs: meal glucose/fructose/fluid/sodium rates, EE estimate, bootstrap
        #        Cortisol, REDS, Vagal_Tone (Mod 3), Skin_Blood_Flow (Mod 10)
        # Produces: Hub_Glucose_Absorption_Rate, Hub_Fructose_Absorption_Rate,
        #           Hub_Fluid_Absorption_Rate, Hub_Sodium_Absorption_Rate,
        #           Hub_GI_Distress_Index
        #
        # Vagal/SkBF bootstrap: Mod 3 and Mod 10 run AFTER Mod 8 in the same day.
        # We use previous-day terminal values from morning_state to break the cycle.
        # This is physiologically valid: yesterday's exercise ischemia informs today's
        # resting GI motility recovery.
        try:
            t_gi_h = t_rec_h  # GI opera na escala de horas (6h window interna)
            n_gi   = len(t_gi_h)

            # Meal rates from meal_event: distribute glucose, fructose, fluid, Na
            carb_g   = float(meal_event.get("carbohydrate_grams", 60.0))
            fruc_g   = float(meal_event.get("fructose_grams",     20.0))
            water_ml = float(meal_event.get("water_ml",           500.0))
            na_mg    = float(meal_event.get("sodium_mg",          600.0))

            # Bolus at t=0 — decaying exponential delivery [g·h⁻¹]
            t_decay  = 1.5   # hours half-time
            gluc_arr = jnp.asarray(
                (carb_g / t_decay) * np.exp(-np.array(t_gi_h, dtype=float) / t_decay),
                dtype=jnp.float32)
            fruc_arr = jnp.asarray(
                (fruc_g / t_decay) * np.exp(-np.array(t_gi_h, dtype=float) / t_decay),
                dtype=jnp.float32)
            fluid_ml_arr = jnp.asarray(
                (water_ml / t_decay) * np.exp(-np.array(t_gi_h, dtype=float) / t_decay),
                dtype=jnp.float32)
            na_arr = jnp.asarray(
                (na_mg / t_decay) * np.exp(-np.array(t_gi_h, dtype=float) / t_decay),
                dtype=jnp.float32)

            # EE estimate: scalar EE in kcal/h on exercise hours, 0 otherwise
            ee_est_kcal_h = sess_kcal / t_span_exercise_h
            ee_gi_arr = jnp.asarray(
                [ee_est_kcal_h if t <= t_span_exercise_h else 0.0
                 for t in np.array(t_gi_h, dtype=float)],
                dtype=jnp.float32)

            # Vagal tone bootstrap: resting state = 1.0 (full parasympathetic)
            # SkBF bootstrap: previous-day exercise SkBF (L/h) — default 3 L/h (rest)
            gi_vagal_arr = _scalar_to_arr(1.0,           t_gi_h)  # resting default
            gi_skbf_arr  = _scalar_to_arr(boot_skbf,     t_gi_h)  # previous-day SkBF

            gi_out = self.gi.simulate_gastrointestinal(
                bayesian_priors            = bayesian_priors,
                meal_glucose_rate_arr      = gluc_arr,
                meal_fructose_rate_arr     = fruc_arr,
                meal_fluid_rate_arr        = fluid_ml_arr,
                meal_sodium_rate_arr       = na_arr,
                hub_energy_expenditure_arr = ee_gi_arr,
                hub_cortisol_arr           = _scalar_to_arr(boot_cortisol, t_gi_h),
                hub_reds_arr               = _scalar_to_arr(boot_reds,     t_gi_h),
                t_hub_h                    = t_gi_h,
                hub_vagal_tone_arr         = gi_vagal_arr,
                hub_skin_blood_flow_arr    = gi_skbf_arr,
                t_span_h                   = (0.0, min(t_span_recovery_h, 6.0)),
                n_save                     = min(n_save_recovery, 512),
            )
            state.update({k: v for k, v in gi_out.items() if k.startswith("Hub_")})
            state["_gi_t_h"] = gi_out["t_h"]
            logger.debug("Mod 8 (GI) OK")
        except Exception as exc:
            logger.warning("Mod 8 (GI) FAILED: %s — using bootstrap zeros", exc)
            state["Hub_Glucose_Absorption_Rate"]  = _zeros_h(t_rec_h)
            state["Hub_Fructose_Absorption_Rate"] = _zeros_h(t_rec_h)
            state["Hub_Fluid_Absorption_Rate"]    = _zeros_h(t_rec_h)
            state["Hub_Sodium_Absorption_Rate"]   = _zeros_h(t_rec_h)
            state["Hub_GI_Distress_Index"]        = _zeros_h(t_rec_h)
            state["_gi_t_h"] = t_rec_h

        # ══════════════════════════════════════════════════════════════════
        # BLOCO B — Motor de Exercício (escala de minutos)
        # ══════════════════════════════════════════════════════════════════

        # ── B1: Módulo 1 — Metabolic ──────────────────────────────────────
        # Needs: bayesian_priors, planned_session, meal_event
        # Produces: Hub_Lactate_Signalling, Hub_Energy_Expenditure_Kcal
        try:
            met_out = self.metabolic.simulate_metabolic_response(
                bayesian_priors = bayesian_priors,
                meal_event      = meal_event,
                session_record  = planned_session,
                t_span_hours    = t_span_exercise_h,
                n_save_points   = n_save_exercise,
            )
            state["Hub_Lactate_Signalling"]   = met_out["Hub_Lactate_Signalling"]
            state["Hub_Energy_Expenditure_Kcal"] = met_out["Hub_Energy_Expenditure_Kcal"]
            state["_met_t_min"]               = met_out["t_min"]
            state["_met_La_arr"]              = met_out["La_mmol_L"]
            state["_met_EE_kcal_scalar"]      = met_out["Hub_Energy_Expenditure_Kcal"]
            logger.debug("Mod 1 (Metabolic) OK — La_peak=%.2f mmol/L",
                         float(jnp.max(met_out["La_mmol_L"])))
        except Exception as exc:
            logger.warning("Mod 1 (Metabolic) FAILED: %s", exc)
            t_min_mock = jnp.linspace(0.0, t_span_exercise_h * 60.0,
                                      n_save_exercise, dtype=jnp.float32)
            state["Hub_Lactate_Signalling"]      = _zeros_h(t_min_mock)
            state["Hub_Energy_Expenditure_Kcal"] = 0.0
            state["_met_t_min"]                  = t_min_mock
            state["_met_La_arr"]                 = _zeros_h(t_min_mock)
            state["_met_EE_kcal_scalar"]         = 0.0

        # ── B2: Módulo 2 — Neuromuscular ──────────────────────────────────
        # Needs: bayesian_priors, planned_session
        # Produces: Hub_CK_StructuralDamage, Hub_FGF21_MetabolicStress,
        #           Hub_FGF23_BoneCoupling  (all on t_acute_min axis)
        # Auditoria: Hub_FGF23_BoneCoupling injected into Mod 11 mechanical load
        try:
            nm_out = self.neuromuscular.simulate_neuromuscular_response(
                bayesian_priors = bayesian_priors,
                session_record  = planned_session,
                t_span_days     = 14.0,
            )
            state["Hub_CK_StructuralDamage"]  = nm_out["Hub_CK_StructuralDamage"]
            state["Hub_FGF21_MetabolicStress"]= nm_out["Hub_FGF21_MetabolicStress"]
            state["Hub_FGF23_BoneCoupling"]   = nm_out["Hub_FGF23_BoneCoupling"]
            state["_nm_t_acute_min"]          = nm_out["t_acute_min"]
            logger.debug("Mod 2 (Neuromuscular) OK")
        except Exception as exc:
            logger.warning("Mod 2 (Neuromuscular) FAILED: %s", exc)
            t_min_mock = jnp.linspace(0.0, t_span_exercise_h * 60.0,
                                      n_save_exercise, dtype=jnp.float32)
            state["Hub_CK_StructuralDamage"]  = _zeros_h(t_min_mock)
            state["Hub_FGF21_MetabolicStress"]= _zeros_h(t_min_mock)
            state["Hub_FGF23_BoneCoupling"]   = _zeros_h(t_min_mock)
            state["_nm_t_acute_min"]          = t_min_mock

        # ── B3: Módulo 3 — Cardiorespiratory ─────────────────────────────
        # Needs: bayesian_priors, planned_session, Lactate (Mod 1),
        #        PV_drop (Mod 10 previous-day bootstrap — Coyle 1986 Cardiovascular Drift)
        # Produces: Hub_Catecholamines_Tone, Hub_Vagal_Tone, V_E_L_min
        try:
            la_session_mean = float(jnp.mean(state["Hub_Lactate_Signalling"]))
            # PV_drop bootstrap: previous-day plasma volume depletion informs Frank-Starling
            t_min_ex = jnp.linspace(0.0, t_span_exercise_h * 60.0,
                                    n_save_exercise, dtype=jnp.float32)
            pv_boot_arr = _scalar_to_arr(boot_pv_drop, t_min_ex)

            cardio_out = self.cardio.simulate_cardiorespiratory_response(
                bayesian_priors                  = bayesian_priors,
                session_record                   = planned_session,
                hub_lactate_mmol_L               = la_session_mean,
                hub_plasma_volume_drop_pct_arr   = pv_boot_arr,
                hub_pv_t_min                     = t_min_ex,
                t_span_hours                     = t_span_exercise_h,
                n_save_points                    = n_save_exercise,
            )
            state["Hub_Catecholamines_Tone"] = cardio_out["Hub_Catecholamines_Tone"]
            state["Hub_Vagal_Tone"]          = cardio_out["Hub_Vagal_Tone"]
            state["_cardio_VE_arr"]          = cardio_out["V_E_L_min"]
            state["_cardio_t_min"]           = cardio_out["t_min"]
            logger.debug("Mod 3 (Cardiorespiratory) OK")
        except Exception as exc:
            logger.warning("Mod 3 (Cardiorespiratory) FAILED: %s", exc)
            t_min_mock = jnp.linspace(0.0, t_span_exercise_h * 60.0,
                                      n_save_exercise, dtype=jnp.float32)
            state["Hub_Catecholamines_Tone"] = _zeros_h(t_min_mock)
            state["Hub_Vagal_Tone"]          = jnp.ones_like(t_min_mock)
            state["_cardio_VE_arr"]          = _zeros_h(t_min_mock)
            state["_cardio_t_min"]           = t_min_mock

        # ══════════════════════════════════════════════════════════════════
        # CONVERSÃO t_min → t_h
        # Todos os arrays do Bloco B existem em minutos e devem ser
        # interpolados para o eixo horário de recuperação antes do Bloco C.
        # ══════════════════════════════════════════════════════════════════

        t_min_b1 = state["_met_t_min"]
        t_min_b2 = state["_nm_t_acute_min"]
        t_min_b3 = state["_cardio_t_min"]

        la_h     = _min_to_h(state["Hub_Lactate_Signalling"],   t_min_b1, t_rec_h)
        ck_h     = _min_to_h(state["Hub_CK_StructuralDamage"],  t_min_b2, t_rec_h)
        fgf21_h  = _min_to_h(state["Hub_FGF21_MetabolicStress"],t_min_b2, t_rec_h)
        fgf23_h  = _min_to_h(state["Hub_FGF23_BoneCoupling"],   t_min_b2, t_rec_h)
        cat_h    = _min_to_h(state["Hub_Catecholamines_Tone"],  t_min_b3, t_rec_h)
        vagal_h  = _min_to_h(state["Hub_Vagal_Tone"],           t_min_b3, t_rec_h)
        ve_h     = _min_to_h(state["_cardio_VE_arr"],           t_min_b3, t_rec_h)

        # EE array in kcal/h (scalar → constant array)
        ee_scalar = float(state.get("_met_EE_kcal_scalar") or 0.0)
        ee_h = jnp.asarray(
            [ee_scalar / t_span_exercise_h if t <= t_span_exercise_h else 0.0
             for t in np.array(t_rec_h, dtype=float)],
            dtype=jnp.float32,
        )

        # Mechanical load proxy [normalized]: power × session gate (pure physics)
        # FGF23 is now injected natively into Mod 11 ODE — no proxy needed here
        power_ref = 200.0
        mech_load_h = jnp.asarray(
            [(power_w / power_ref)
             if float(t_rec_h[i]) <= t_span_exercise_h else 0.0
             for i, _ in enumerate(t_rec_h)],
            dtype=jnp.float32,
        )

        # ══════════════════════════════════════════════════════════════════
        # BLOCO C — Circulação e Defesa Sistémica (escala de horas)
        # ══════════════════════════════════════════════════════════════════

        # ── C1: Módulo 10 — Thermo-Fluid ─────────────────────────────────
        # Needs: EE (Mod1), VE (Mod3), Fluid/Na (Mod8), Basal_Temp (Mod6)
        # Produces: Hub_Core_Temp, Hub_Skin_Temp, Hub_Sweat_Rate_L_h,
        #           Hub_Plasma_Volume_Drop_Pct, Hub_Sodium_Concentration,
        #           Hub_Skin_Blood_Flow
        try:
            gi_t   = state["_gi_t_h"]
            fl_abs = state["Hub_Fluid_Absorption_Rate"]
            na_abs = state["Hub_Sodium_Absorption_Rate"]
            fl_abs_h = jnp.interp(t_rec_h, gi_t, jnp.asarray(fl_abs, dtype=jnp.float32))
            na_abs_h = jnp.interp(t_rec_h, gi_t, jnp.asarray(na_abs, dtype=jnp.float32))

            hub_inbound_thermo = {
                "ee_t_h":             t_rec_h,
                "ee_arr_kcal_h":      ee_h,
                "ve_t_h":             t_rec_h,
                "ve_arr_L_min":       ve_h,
                "fluid_abs_t_h":      t_rec_h,
                "fluid_abs_rate_L_h": fl_abs_h / 1000.0,   # ml→L
                "na_abs_t_h":         t_rec_h,
                "na_abs_rate_mmol_h": na_abs_h / 23.0,     # mg→mmol
                "basal_temp_offset_C": state["_thyroid_basal_temp"] - 36.6,
                # Melatonin (Cagnacci 1992): nocturnal SkBF vasodilation — previous-day bootstrap
                "mel_t_h":  t_rec_h,
                "mel_arr":  _scalar_to_arr(boot_melatonin, t_rec_h),
            }

            thermo_out = self.thermo_fluid.simulate_thermo_fluid_response(
                bayesian_priors = bayesian_priors,
                session_record  = planned_session,
                fluid_intake    = fluid_intake,
                env_conditions  = env_cond,
                hub_inbound     = hub_inbound_thermo,
                t_span_hours    = t_span_exercise_h,
                n_save_points   = n_save_exercise,
            )
            state.update({k: v for k, v in thermo_out.items() if k.startswith("Hub_")})
            state["_thermo_t_h"] = thermo_out["t_h"]

            # Resample Hub_Core_Temp onto t_rec_h for downstream modules
            thermo_t  = jnp.asarray(thermo_out["t_h"], dtype=jnp.float32)
            core_h    = jnp.interp(t_rec_h, thermo_t,
                                   jnp.asarray(thermo_out["Hub_Core_Temp"], dtype=jnp.float32))
            state["_core_temp_on_rec_h"] = core_h
            logger.debug("Mod 10 (ThermoFluid) OK — T_core_peak=%.2f°C",
                         float(jnp.max(core_h)))
        except Exception as exc:
            logger.warning("Mod 10 (ThermoFluid) FAILED: %s", exc)
            state["Hub_Core_Temp"]              = _scalar_to_arr(37.0, t_rec_h)
            state["Hub_Skin_Temp"]              = _scalar_to_arr(34.0, t_rec_h)
            state["Hub_Sweat_Rate_L_h"]         = _zeros_h(t_rec_h)
            state["Hub_Plasma_Volume_Drop_Pct"] = _zeros_h(t_rec_h)
            state["Hub_Sodium_Concentration"]   = _scalar_to_arr(140.0, t_rec_h)
            state["Hub_Skin_Blood_Flow"]        = _zeros_h(t_rec_h)
            state["_thermo_t_h"]               = t_rec_h
            state["_core_temp_on_rec_h"]       = _scalar_to_arr(37.0, t_rec_h)

        # ── C2: Módulo 5 — Neuroendocrine (HPA/HPG) ──────────────────────
        # NOTA: Não recebe bayesian_priors — chama self._build_params() internamente.
        # Needs: bootstrap SCN, bootstrap GH, bootstrap sleep_debt,
        #        Catecholamines (Mod3), Lactate (Mod1)
        # Produces: Hub_Cortisol_Catabolic, Hub_Testosterone_Anabolic,
        #           Hub_REDS_State
        try:
            boot_scn_arr = jnp.array([boot_scn, boot_scn], dtype=jnp.float32)
            boot_scn_t   = jnp.array([0.0, float(t_span_recovery_h)], dtype=jnp.float32)
            boot_gh_arr  = jnp.array([boot_gh, boot_gh], dtype=jnp.float32)
            boot_gh_t    = jnp.array([0.0, float(t_span_recovery_h)], dtype=jnp.float32)

            energy_intake_est = float(
                bayesian_priors.get("energy_intake_kcal_per_day", 2500.0)) / max(ffm_kg, 1.0)

            ne_out = self.neuroendo.simulate_neuroendocrine_response(
                duration_h                   = t_span_recovery_h,
                hub_scn_phase                = boot_scn_arr,
                t_scn_h                      = boot_scn_t,
                hub_catecholamines_tone       = cat_h,
                t_cat_h                      = t_rec_h,
                hub_lactate_signalling        = la_h,
                t_La_h                       = t_rec_h,
                hub_sleep_debt               = boot_sleep_debt,
                hub_gh_repair                = boot_gh_arr,
                t_gh_h                       = boot_gh_t,
                energy_intake_kcal_per_kg_ffm= energy_intake_est,
                exercise_kcal_per_kg_ffm     = exercise_kcal_per_kg_ffm,
                EA_init_norm                 = boot_ea_norm,
                n_save                       = n_save_recovery,
            )
            state["Hub_Cortisol_Catabolic"]    = ne_out["Hub_Cortisol_Catabolic"]
            state["Hub_Testosterone_Anabolic"] = ne_out["Hub_Testosterone_Anabolic"]
            state["Hub_REDS_State"]            = ne_out["Hub_REDS_State"]
            state["_ne_t_h"]                   = ne_out["t_h"]
            state["_ne_cortisol"]              = ne_out["Hub_Cortisol_Catabolic"]
            state["_ne_testo"]                 = ne_out["Hub_Testosterone_Anabolic"]
            state["_ne_reds"]                  = ne_out["Hub_REDS_State"]
            logger.debug("Mod 5 (Neuroendocrine) OK — Cort_mean=%.2f",
                         float(jnp.mean(ne_out["Hub_Cortisol_Catabolic"])))
        except Exception as exc:
            logger.warning("Mod 5 (Neuroendocrine) FAILED: %s", exc)
            state["Hub_Cortisol_Catabolic"]    = _scalar_to_arr(boot_cortisol, t_rec_h)
            state["Hub_Testosterone_Anabolic"] = _scalar_to_arr(boot_testo,    t_rec_h)
            state["Hub_REDS_State"]            = _scalar_to_arr(boot_reds,     t_rec_h)
            state["_ne_t_h"]                   = t_rec_h
            state["_ne_cortisol"]              = _scalar_to_arr(boot_cortisol, t_rec_h)
            state["_ne_testo"]                 = _scalar_to_arr(boot_testo,    t_rec_h)
            state["_ne_reds"]                  = _scalar_to_arr(boot_reds,     t_rec_h)

        # ── C3: Módulo 7 — Immune & Repair ───────────────────────────────
        # Needs: CK (Mod2), Cortisol (Mod5), Testosterone (Mod5), REDS (Mod5),
        #        Thyroid_Tone (Mod6), GH bootstrap,
        #        Melatonin (Mod4 previous-day bootstrap — Carrillo-Vico 2013 NF-κB)
        try:
            thyroid_t_h    = state["_thyroid_t_h"]
            thyroid_t_tone = state["_thyroid_t_tone"]
            ne_t_h         = state["_ne_t_h"]

            # Resample onto common t_rec_h
            t_tone_on_rec = jnp.interp(t_rec_h,
                                        jnp.asarray(thyroid_t_h, dtype=jnp.float32),
                                        jnp.asarray(thyroid_t_tone, dtype=jnp.float32))
            cort_on_rec   = jnp.interp(t_rec_h,
                                        jnp.asarray(ne_t_h, dtype=jnp.float32),
                                        jnp.asarray(state["_ne_cortisol"], dtype=jnp.float32))
            testo_on_rec  = jnp.interp(t_rec_h,
                                        jnp.asarray(ne_t_h, dtype=jnp.float32),
                                        jnp.asarray(state["_ne_testo"], dtype=jnp.float32))
            reds_on_rec   = jnp.interp(t_rec_h,
                                        jnp.asarray(ne_t_h, dtype=jnp.float32),
                                        jnp.asarray(state["_ne_reds"], dtype=jnp.float32))

            gh_boot_arr_rec     = _scalar_to_arr(boot_gh,        t_rec_h)
            mel_boot_arr_rec    = _scalar_to_arr(boot_melatonin, t_rec_h)

            imm_out = self.immune.simulate_immune_repair(
                bayesian_priors      = bayesian_priors,
                hub_ck_arr           = ck_h,
                hub_cortisol_arr     = cort_on_rec,
                hub_testosterone_arr = testo_on_rec,
                hub_reds_arr         = reds_on_rec,
                hub_t_tone_arr       = t_tone_on_rec,
                hub_gh_arr           = gh_boot_arr_rec,
                t_hub_h              = t_rec_h,
                hub_melatonin_arr    = mel_boot_arr_rec,
                t_span_h             = (0.0, t_span_recovery_h),
            )
            state.update({k: v for k, v in imm_out.items() if k.startswith("Hub_")})
            state["_imm_t_h"]  = imm_out["t_h"]
            state["_imm_il6"]  = imm_out["Hub_Cytokine_IL6_Systemic"]
            logger.debug("Mod 7 (ImmuneRepair) OK")
        except Exception as exc:
            logger.warning("Mod 7 (ImmuneRepair) FAILED: %s", exc)
            state["Hub_Cytokine_IL6_Systemic"]    = _zeros_h(t_rec_h)
            state["Hub_Cytokine_TNFa"]            = _zeros_h(t_rec_h)
            state["Hub_Biomarker_CRP_mg_L"]       = _scalar_to_arr(1.0, t_rec_h)
            state["Hub_Tissue_Repair_Completion"] = _scalar_to_arr(1.0, t_rec_h)
            state["_imm_t_h"]                     = t_rec_h
            state["_imm_il6"]                     = _zeros_h(t_rec_h)

        # ══════════════════════════════════════════════════════════════════
        # BLOCO D — Cérebro, Chassis e Recuperação
        # ══════════════════════════════════════════════════════════════════

        # ── D1: Módulo 4 — Sleep & Circadian ─────────────────────────────
        # Needs: EE (Mod1), Catecholamines (Mod3), Cortisol (Mod5)
        # Produces: Hub_SCN_Phase, Hub_Sleep_Debt_Metabolic,
        #           Hub_GH_Repair_Signalling, Hub_Melatonin_Tone
        # Auditoria: GH do Mod4 supera GH bootstrap → volta a Mod7 e Mod11.
        #   Melatonina do Mod4 tem efeito antiinflamatório direto em Mod7
        #   (Carrillo-Vico 2013). Atualmente não existe canal explícito
        #   Mel→Mod7; registado no DigitalTwinState para uso futuro.
        try:
            light_lux = float(env_cond.get("light_lux", 0.0))
            if light_lux > 0:
                light_arr_rec = _scalar_to_arr(light_lux, t_rec_h)
            else:
                light_arr_rec = None

            cort_on_rec_sleep = jnp.interp(
                t_rec_h,
                jnp.asarray(state["_ne_t_h"], dtype=jnp.float32),
                jnp.asarray(state["_ne_cortisol"], dtype=jnp.float32),
            )

            sleep_out = self.sleep_circ.simulate_sleep_circadian(
                bayesian_priors             = bayesian_priors,
                hub_energy_expenditure_arr  = ee_h,
                hub_catecholamines_arr      = cat_h,
                hub_cortisol_arr            = cort_on_rec_sleep,
                t_hub_h                     = t_rec_h,
                hub_light_lux_arr           = light_arr_rec,
                t_span_h                    = (0.0, t_span_recovery_h),
                H_init                      = boot_sleep_debt,
                n_save                      = n_save_recovery,
            )
            state.update({k: v for k, v in sleep_out.items() if k.startswith("Hub_")})
            state["_sleep_t_h"] = sleep_out["t_h"]
            state["_sleep_gh"]  = sleep_out["Hub_GH_Repair_Signalling"]
            state["_sleep_debt"]= sleep_out["Hub_Sleep_Debt_Metabolic"]
            logger.debug("Mod 4 (SleepCircadian) OK")
        except Exception as exc:
            logger.warning("Mod 4 (SleepCircadian) FAILED: %s", exc)
            state["Hub_SCN_Phase"]              = _scalar_to_arr(boot_scn,        t_rec_h)
            state["Hub_Sleep_Debt_Metabolic"]   = _scalar_to_arr(boot_sleep_debt, t_rec_h)
            state["Hub_GH_Repair_Signalling"]   = _scalar_to_arr(boot_gh,         t_rec_h)
            state["Hub_Melatonin_Tone"]         = _zeros_h(t_rec_h)
            state["_sleep_t_h"] = t_rec_h
            state["_sleep_gh"]  = _scalar_to_arr(boot_gh,         t_rec_h)
            state["_sleep_debt"]= _scalar_to_arr(boot_sleep_debt, t_rec_h)

        # ── D2: Módulo 11 — Biomechanical Tissue ─────────────────────────
        # Needs: Mechanical_Load (Mod2), GH (Mod4), Cortisol (Mod5),
        #        Sex_Hormones (Mod5), IL6 (Mod7), FGF23 (Mod2 — nativo na ODE)
        # FGF23 from Mod 2 is now injected natively in the Mod 11 ODE for
        # Wnt/LRP5 OBL amplification (Jones et al. 2006; Robling 2008).
        try:
            sleep_t_h  = jnp.asarray(state["_sleep_t_h"], dtype=jnp.float32)
            ne_t_h_arr = jnp.asarray(state["_ne_t_h"],    dtype=jnp.float32)
            imm_t_h    = jnp.asarray(state["_imm_t_h"],   dtype=jnp.float32)

            gh_on_rec   = jnp.interp(t_rec_h, sleep_t_h,
                                      jnp.asarray(state["_sleep_gh"],    dtype=jnp.float32))
            cort_on_rec2= jnp.interp(t_rec_h, ne_t_h_arr,
                                      jnp.asarray(state["_ne_cortisol"], dtype=jnp.float32))
            testo_on_rec2=jnp.interp(t_rec_h, ne_t_h_arr,
                                      jnp.asarray(state["_ne_testo"],    dtype=jnp.float32))
            il6_on_rec  = jnp.interp(t_rec_h, imm_t_h,
                                      jnp.asarray(state["_imm_il6"],     dtype=jnp.float32))

            bio_out = self.biomech.simulate_biomechanical_tissue(
                bayesian_priors         = bayesian_priors,
                hub_mechanical_load_arr = mech_load_h,
                hub_gh_repair_arr       = gh_on_rec,
                hub_cortisol_arr        = cort_on_rec2,
                hub_sex_hormones_arr    = testo_on_rec2,
                hub_il6_arr             = il6_on_rec,
                hub_fgf23_arr           = fgf23_h,
                t_hub_h                 = t_rec_h,
                t_span_h                = (0.0, t_span_recovery_h),
                n_save                  = n_save_recovery,
            )
            state.update({k: v for k, v in bio_out.items() if k.startswith("Hub_")})
            state["_bio_t_h"]        = bio_out["t_h"]
            state["_bio_microdamage"]= bio_out["Hub_Tissue_Microdamage_z"]
            logger.debug("Mod 11 (Biomechanical) OK")
        except Exception as exc:
            logger.warning("Mod 11 (Biomechanical) FAILED: %s", exc)
            state["Hub_Tissue_Microdamage_z"]      = _zeros_h(t_rec_h)
            state["Hub_Bone_Mass_Density"]         = _scalar_to_arr(1.0, t_rec_h)
            state["Hub_Tendon_Stiffness_Capacity"] = _scalar_to_arr(1.0, t_rec_h)
            state["_bio_t_h"]        = t_rec_h
            state["_bio_microdamage"]= _zeros_h(t_rec_h)

        # ── D3: Módulo 9 — Central Fatigue / Governador ──────────────────
        # Needs: Lactate (Mod1), Catecholamines (Mod3), Sleep_Debt (Mod4),
        #        IL6 (Mod7), GI_Distress (Mod8),
        #        Hub_Core_Temp (Mod10) → hub_core_temp_arr NATIVO na ODE (Nybo & Nielsen 2001)
        try:
            gi_t_h   = jnp.asarray(state["_gi_t_h"], dtype=jnp.float32)
            gi_dist  = jnp.asarray(state["Hub_GI_Distress_Index"], dtype=jnp.float32)
            gi_on_rec= jnp.interp(t_rec_h, gi_t_h, gi_dist)

            sleep_debt_rec = jnp.interp(
                t_rec_h,
                jnp.asarray(state["_sleep_t_h"], dtype=jnp.float32),
                jnp.asarray(state["_sleep_debt"], dtype=jnp.float32),
            )
            il6_on_rec2 = jnp.interp(
                t_rec_h,
                jnp.asarray(state["_imm_t_h"], dtype=jnp.float32),
                jnp.asarray(state["_imm_il6"], dtype=jnp.float32),
            )

            # Hub_Core_Temp → Mod 9: injected natively in ODE (Nybo & Nielsen 2001)
            core_temp_rec = state["_core_temp_on_rec_h"]

            cf_out = self.central_fat.simulate_central_fatigue(
                bayesian_priors          = bayesian_priors,
                hub_lactate_arr          = la_h,
                hub_catecholamines_arr   = cat_h,
                hub_sleep_debt_arr       = sleep_debt_rec,
                hub_il6_arr              = il6_on_rec2,
                hub_gi_distress_arr      = gi_on_rec,
                hub_core_temp_arr        = core_temp_rec,
                t_hub_h                  = t_rec_h,
                t_span_h                 = (0.0, t_span_exercise_h),
                n_save                   = n_save_exercise,
            )
            state.update({k: v for k, v in cf_out.items() if k.startswith("Hub_")})
            state["_cf_t_h"] = cf_out["t_h"]
            logger.debug("Mod 9 (CentralFatigue) OK — CFI_max=%.3f",
                         float(jnp.max(cf_out["Hub_Central_Fatigue_Index"])))
        except Exception as exc:
            logger.warning("Mod 9 (CentralFatigue) FAILED: %s", exc)
            t_ex_mock = jnp.linspace(0.0, t_span_exercise_h, n_save_exercise, dtype=jnp.float32)
            state["Hub_RPE_Borg"]              = _scalar_to_arr(6.0, t_ex_mock)
            state["Hub_Central_Fatigue_Index"] = _zeros_h(t_ex_mock)
            state["Hub_Motor_Recruitment_Cap"] = jnp.ones_like(t_ex_mock)
            state["_cf_t_h"]                  = t_ex_mock

        # ══════════════════════════════════════════════════════════════════
        # COMPILAÇÃO DO DigitalTwinState
        # ══════════════════════════════════════════════════════════════════

        # Meta-axes
        state["_t_exercise_h"]  = t_ex_h
        state["_t_recovery_h"]  = t_rec_h

        # Summary scalars — next day's morning_state
        state["next_morning_state"] = {
            "cortisol_norm":       _safe_last(state.get("Hub_Cortisol_Catabolic",
                                                         [boot_cortisol])),
            "reds_state":          _safe_last(state.get("Hub_REDS_State",
                                                         [boot_reds])),
            "scn_phase":           _safe_last(state.get("Hub_SCN_Phase",
                                                         [boot_scn])),
            "sleep_debt":          _safe_last(state.get("Hub_Sleep_Debt_Metabolic",
                                                         [boot_sleep_debt])),
            "gh_repair":           _safe_last(state.get("Hub_GH_Repair_Signalling",
                                                         [boot_gh])),
            "ea_init_norm":        _safe_last(state.get("_ne_reds",
                                                         [boot_ea_norm])),
            "testosterone_norm":   _safe_last(state.get("Hub_Testosterone_Anabolic",
                                                         [boot_testo])),
            # New keys for native biological connections (forward-propagated)
            "pv_drop_pct":         _safe_last(state.get("Hub_Plasma_Volume_Drop_Pct",
                                                         [boot_pv_drop])),
            "melatonin_tone":      _safe_last(state.get("Hub_Melatonin_Tone",
                                                         [boot_melatonin])),
            # Hub_Skin_Blood_Flow is in L/min (thermo output); convert to L/h for GI ischemia gate
            "skin_blood_flow_L_h": _safe_last(state.get("Hub_Skin_Blood_Flow",
                                                         [boot_skbf / 60.0])) * 60.0,
        }

        logger.info(
            "NutriviousEngine.simulate_daily_cycle COMPLETE — %d Hub variables exported.",
            sum(1 for k in state if k.startswith("Hub_")),
        )
        return state
