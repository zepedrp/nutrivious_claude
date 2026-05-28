from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
import diffrax


_CHO_OSM: float = 5555.0          # mOsm·ml·g⁻¹·L⁻¹  (MW≈180 g/mol)
_NA_MW_MG_MMOL: float = 22.99     # mg per mmol Na
# SGLT1 stoichiometry: 2 Na⁺ per glucose molecule
_NA_SGLT1_RATIO: float = 2.0 * 22990.0 / 180.0   # ≈ 255.4 mg Na per g glucose


class GastrointestinalParams(NamedTuple):
    # Gastric emptying (Hunt & Stubbs √V kinetics)
    k_empty: float        # ml^0.5·h⁻¹
    EE_thresh: float      # kcal·h⁻¹ — exercise inhibition threshold
    n_EE: float           # logistic steepness
    k_Osm_empty: float    # (mOsm/L)⁻¹ — exponential osmolarity inhibition
    Osm_ref: float        # mOsm/L — threshold above which emptying slows
    # SGLT1 absorption (glucose, sodium-coupled)
    Vmax_SGLT1: float     # g·h⁻¹
    Km_SGLT1: float       # g·L⁻¹
    # GLUT5 absorption (fructose, facilitated)
    Vmax_GLUT5: float     # g·h⁻¹
    Km_GLUT5: float       # g·L⁻¹
    # 2:1 SGLT1/GLUT5 synergy (fructose co-transport upregulation)
    k_synergy: float
    Km_syn: float         # g·L⁻¹
    # Catabolic suppression of absorption
    k_Cort_abs: float
    k_REDS_abs: float
    # Water dynamics
    k_water_abs: float    # h⁻¹ — isotonic water absorption rate
    k_osm: float          # ml·h⁻¹·(mOsm/L)⁻¹ — osmotic flux coefficient
    Osm_blood: float      # mOsm/L — plasma osmolarity reference
    # Sodium dynamics
    k_Na_passive: float   # h⁻¹·(mg/ml)⁻¹ — passive gradient-driven Na absorption
    Na_blood_mg_ml: float # mg/ml — plasma Na concentration (~150 mmol/L × 22.99)
    # GI distress scoring
    W_distress_ref: float    # ml — intestinal volume at distress=5
    Osm_distress_ref: float  # mOsm/L — osmolarity at distress onset
    # Genetic priors (resolved to scalars)
    shannon_scale: float     # microbiome diversity → transporter expression
    zonulin_scale: float     # tight-junction permeability → osmotic flux gain
    # Autonomic and haemodynamic coupling
    k_vagal_empty: float     # vagal tone → gastric emptying amplification (enteric NS)
    k_SkBF_abs: float        # splanchnic ischemia: SkBF above threshold → absorption suppression


_SKBF_ISCH_THRESH: float = 60.0   # L/h above which ischemia starts (Rehrer 2001)


def gastrointestinal_ode(t, y, args):
    params, ee_interp, cort_interp, reds_interp, \
        gluc_rate_interp, fruc_rate_interp, fluid_rate_interp, na_rate_interp, \
        vagal_interp, skbf_interp = args

    V_s  = y[0]   # stomach volume [ml]
    G_s  = y[1]   # stomach glucose [g]
    F_s  = y[2]   # stomach fructose [g]
    Na_s = y[3]   # stomach sodium [mg]
    G_i  = y[4]   # intestine glucose [g]
    F_i  = y[5]   # intestine fructose [g]
    W_i  = y[6]   # intestine water [ml]
    Na_i = y[7]   # intestine sodium [mg]

    E_exp  = jnp.clip(ee_interp.evaluate(t),    0.0, None)
    Cort   = jnp.clip(cort_interp.evaluate(t),  0.0, None)
    REDS   = jnp.clip(reds_interp.evaluate(t),  0.0, None)
    Vagal  = jnp.clip(vagal_interp.evaluate(t), 0.0, 1.0)
    SkBF   = jnp.clip(skbf_interp.evaluate(t),  0.0, None)

    meal_gluc  = jnp.clip(gluc_rate_interp.evaluate(t),  0.0, None)
    meal_fruc  = jnp.clip(fruc_rate_interp.evaluate(t),  0.0, None)
    meal_fluid = jnp.clip(fluid_rate_interp.evaluate(t), 0.0, None)
    meal_na    = jnp.clip(na_rate_interp.evaluate(t),    0.0, None)

    V_s_pos  = jnp.maximum(V_s,  1e-6)
    G_s_pos  = jnp.maximum(G_s,  0.0)
    F_s_pos  = jnp.maximum(F_s,  0.0)
    Na_s_pos = jnp.maximum(Na_s, 0.0)
    G_i_pos  = jnp.maximum(G_i,  0.0)
    F_i_pos  = jnp.maximum(F_i,  0.0)
    W_i_pos  = jnp.maximum(W_i,  1e-6)
    Na_i_pos = jnp.maximum(Na_i, 0.0)

    # Stomach osmolarity (CHO only; Na too dilute at physiological concentrations to inhibit significantly)
    Osm_s = (G_s_pos + F_s_pos) / V_s_pos * _CHO_OSM

    # Exercise inhibition (logistic: →0 above EE_thresh)
    EE_gate = 1.0 / (1.0 + (E_exp / params.EE_thresh) ** params.n_EE)

    # Osmolarity inhibition of gastric emptying (exponential decay above Osm_ref)
    Osm_gate_s = jnp.exp(-params.k_Osm_empty * jnp.maximum(Osm_s - params.Osm_ref, 0.0))

    # Vagal tone amplifies gastric emptying via enteric nervous system (Powley & Berthoud 1985):
    # V_vagal=1.0 (rest) → 1+k factor; V_vagal→0 (intense exercise) → no amplification
    vagal_empty_factor = 1.0 + params.k_vagal_empty * Vagal

    # Hunt & Stubbs gastric emptying rate [ml·h⁻¹]
    k_empty_eff   = params.k_empty * EE_gate * Osm_gate_s * vagal_empty_factor
    emptying_flow = k_empty_eff * jnp.sqrt(jnp.maximum(V_s, 0.0))

    # Fractional CHO and Na delivery from stomach to intestine
    G_delivery  = (G_s_pos  / V_s_pos) * emptying_flow
    F_delivery  = (F_s_pos  / V_s_pos) * emptying_flow
    Na_delivery = (Na_s_pos / V_s_pos) * emptying_flow

    # Catabolic + ischemic suppression of transporter expression
    # Splanchnic ischemia (Rehrer 2001): high SkBF = blood redistribution to skin,
    # reducing mesenteric flow → SGLT1/GLUT5 expression falls
    ischemia_factor = params.k_SkBF_abs * jnp.maximum(SkBF - _SKBF_ISCH_THRESH, 0.0)
    Cort_REDS_gate = 1.0 / (1.0 + params.k_Cort_abs * Cort + params.k_REDS_abs * REDS + ischemia_factor)

    # Intestinal osmolarity including Na⁺ + Cl⁻ (equimolar anion assumed → factor 2)
    # Osm_i = CHO term + 2 × [Na_i (mg) / 22.99 (mg/mmol)] / [W_i (ml) / 1000 (ml/L)]
    Na_osm_i = 2.0 * (Na_i_pos / _NA_MW_MG_MMOL) / (W_i_pos / 1000.0)
    Osm_i    = (G_i_pos + F_i_pos) / W_i_pos * _CHO_OSM + Na_osm_i

    # SGLT1 glucose absorption with 2:1 fructose synergy
    synergy  = 1.0 + params.k_synergy * F_i_pos / (F_i_pos + params.Km_syn)
    Gluc_abs = (params.Vmax_SGLT1 * params.shannon_scale
                * G_i_pos / (params.Km_SGLT1 + G_i_pos)
                * synergy * Cort_REDS_gate)

    # GLUT5 fructose absorption
    Fruc_abs = (params.Vmax_GLUT5 * params.shannon_scale
                * F_i_pos / (params.Km_GLUT5 + F_i_pos)
                * Cort_REDS_gate)

    # SGLT1 Na⁺ co-transport: 2 Na⁺ per glucose molecule (stoichiometric, luminal depletion)
    Na_SGLT1_abs = _NA_SGLT1_RATIO * Gluc_abs  # mg·h⁻¹

    # Passive Na⁺ absorption (gradient-driven above plasma concentration)
    Na_i_conc_mg_ml = Na_i_pos / W_i_pos  # mg/ml
    Na_passive_abs  = (params.k_Na_passive
                       * jnp.maximum(Na_i_conc_mg_ml - params.Na_blood_mg_ml, 0.0)
                       * W_i_pos)

    Na_abs_total = Na_SGLT1_abs + Na_passive_abs

    # Osmotic water flux (bidirectional: <0 = secretion into lumen if hypertonic)
    k_osm_eff = params.k_osm * params.zonulin_scale
    J_osm     = k_osm_eff * (params.Osm_blood - Osm_i)

    # Isotonic water absorption (co-transport driven)
    Osm_abs_gate = params.Osm_blood ** 2 / (params.Osm_blood ** 2 + Osm_i ** 2)
    J_water_abs  = params.k_water_abs * W_i_pos * Osm_abs_gate

    dV_s  = meal_fluid - emptying_flow
    dG_s  = meal_gluc  - G_delivery
    dF_s  = meal_fruc  - F_delivery
    dNa_s = meal_na    - Na_delivery
    dG_i  = G_delivery - Gluc_abs
    dF_i  = F_delivery - Fruc_abs
    dW_i  = emptying_flow + J_osm - J_water_abs
    dNa_i = Na_delivery   - Na_abs_total

    return jnp.array([dV_s, dG_s, dF_s, dNa_s, dG_i, dF_i, dW_i, dNa_i])


def _solve_gastrointestinal(
    params: GastrointestinalParams,
    t_span_h: tuple,
    ee_interp,
    cort_interp,
    reds_interp,
    gluc_rate_interp,
    fruc_rate_interp,
    fluid_rate_interp,
    na_rate_interp,
    vagal_interp,
    skbf_interp,
    n_save: int = 512,
) -> diffrax.Solution:
    term       = diffrax.ODETerm(gastrointestinal_ode)
    solver     = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    t0, t1     = float(t_span_h[0]), float(t_span_h[1])
    saveat     = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save))
    # Fasting initial: W_i=50ml residual, Na_i=50×3.45mg/ml (plasma-isotonic)
    y0 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 172.5], dtype=jnp.float32)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.001,
        y0=y0,
        args=(params, ee_interp, cort_interp, reds_interp,
              gluc_rate_interp, fruc_rate_interp, fluid_rate_interp, na_rate_interp,
              vagal_interp, skbf_interp),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=65536,
    )


class GastrointestinalSolver:
    # Gastric emptying
    _K_EMPTY_REF:    float = 12.0    # ml^0.5·h⁻¹ → 400ml empties ≈3.3h; half-empty ≈58min
    _EE_THRESH_REF:  float = 700.0   # kcal·h⁻¹
    _N_EE_REF:       float = 4.0
    _K_OSM_EMPTY_REF: float = 0.003  # (mOsm/L)⁻¹
    _OSM_REF_REF:    float = 300.0   # mOsm/L

    # SGLT1
    _VMAX_SGLT1_REF: float = 60.0    # g·h⁻¹; with 2:1 synergy → up to 90 g·h⁻¹
    _KM_SGLT1_REF:   float = 4.0     # g·L⁻¹

    # GLUT5
    _VMAX_GLUT5_REF: float = 30.0    # g·h⁻¹
    _KM_GLUT5_REF:   float = 6.0     # g·L⁻¹

    # Synergy (fructose → SGLT1 upregulation): ceiling factor = 1.5×
    _K_SYNERGY_REF:  float = 0.50
    _KM_SYN_REF:     float = 3.0     # g·L⁻¹

    # Catabolic suppression
    _K_CORT_ABS_REF: float = 0.30
    _K_REDS_ABS_REF: float = 0.40

    # Water dynamics
    _K_WATER_ABS_REF:  float = 3.0   # h⁻¹
    _K_OSM_REF:        float = 1.0   # ml·h⁻¹·(mOsm/L)⁻¹
    _OSM_BLOOD_REF:    float = 290.0 # mOsm/L

    # Sodium dynamics
    _K_NA_PASSIVE_REF:  float = 0.50  # h⁻¹·(mg/ml)⁻¹
    _NA_BLOOD_MG_ML_REF: float = 3.45 # mg/ml ≈ 150 mmol/L × 22.99 mg/mmol / 1000

    # GI distress thresholds
    _W_DISTRESS_REF:   float = 300.0  # ml → distress score 5
    _OSM_DISTRESS_REF: float = 350.0  # mOsm/L → distress onset

    # Autonomic and haemodynamic coupling
    _K_VAGAL_EMPTY_REF: float = 0.40  # per unit V_vagal — enteric NS gastric motility
    _K_SKBF_ABS_REF:    float = 0.008 # per L/h above ischemia threshold — transporter inhibition

    def _build_params(self, bayesian_priors: dict) -> GastrointestinalParams:
        shannon_raw  = bayesian_priors.get("shannon_diversity_prior", float("nan"))
        zonulin_raw  = bayesian_priors.get("zonulin_prior",           float("nan"))
        shannon_scale = (1.0 if math.isnan(float(shannon_raw))
                         else float(np.clip(float(shannon_raw), 0.5, 1.5)))
        zonulin_scale = (1.0 if math.isnan(float(zonulin_raw))
                         else float(np.clip(float(zonulin_raw), 0.5, 2.0)))
        return GastrointestinalParams(
            k_empty          = self._K_EMPTY_REF,
            EE_thresh        = self._EE_THRESH_REF,
            n_EE             = self._N_EE_REF,
            k_Osm_empty      = self._K_OSM_EMPTY_REF,
            Osm_ref          = self._OSM_REF_REF,
            Vmax_SGLT1       = self._VMAX_SGLT1_REF,
            Km_SGLT1         = self._KM_SGLT1_REF,
            Vmax_GLUT5       = self._VMAX_GLUT5_REF,
            Km_GLUT5         = self._KM_GLUT5_REF,
            k_synergy        = self._K_SYNERGY_REF,
            Km_syn           = self._KM_SYN_REF,
            k_Cort_abs       = self._K_CORT_ABS_REF,
            k_REDS_abs       = self._K_REDS_ABS_REF,
            k_water_abs      = self._K_WATER_ABS_REF,
            k_osm            = self._K_OSM_REF,
            Osm_blood        = self._OSM_BLOOD_REF,
            k_Na_passive     = self._K_NA_PASSIVE_REF,
            Na_blood_mg_ml   = self._NA_BLOOD_MG_ML_REF,
            W_distress_ref   = self._W_DISTRESS_REF,
            Osm_distress_ref = self._OSM_DISTRESS_REF,
            shannon_scale    = shannon_scale,
            zonulin_scale    = zonulin_scale,
            k_vagal_empty    = self._K_VAGAL_EMPTY_REF,
            k_SkBF_abs       = self._K_SKBF_ABS_REF,
        )

    @staticmethod
    def bolus_to_rate_arrays(
        meal_events: list,
        t_hub_h,
        bolus_width_h: float = 0.083,
    ):
        """Convert discrete meal events to smooth Gaussian rate arrays.

        meal_events: [(t_meal_h, glucose_g, fructose_g, fluid_ml, sodium_mg), ...]
        Returns (gluc_rate, fruc_rate, fluid_rate, na_rate) in g·h⁻¹ / ml·h⁻¹ / mg·h⁻¹.
        """
        t_arr      = np.asarray(t_hub_h, dtype=np.float64)
        sigma      = float(bolus_width_h)
        norm       = sigma * np.sqrt(2.0 * np.pi)
        gluc_rate  = np.zeros_like(t_arr)
        fruc_rate  = np.zeros_like(t_arr)
        fluid_rate = np.zeros_like(t_arr)
        na_rate    = np.zeros_like(t_arr)
        for event in meal_events:
            t_meal = float(event[0])
            g_g    = float(event[1])
            f_g    = float(event[2])
            fl_ml  = float(event[3])
            na_mg  = float(event[4]) if len(event) > 4 else 0.0
            gauss      = np.exp(-0.5 * ((t_arr - t_meal) / sigma) ** 2) / norm
            gluc_rate  += g_g   * gauss
            fruc_rate  += f_g   * gauss
            fluid_rate += fl_ml * gauss
            na_rate    += na_mg * gauss
        return gluc_rate, fruc_rate, fluid_rate, na_rate

    def simulate_gastrointestinal(
        self,
        bayesian_priors: dict,
        meal_glucose_rate_arr,
        meal_fructose_rate_arr,
        meal_fluid_rate_arr,
        meal_sodium_rate_arr,
        hub_energy_expenditure_arr,
        hub_cortisol_arr,
        hub_reds_arr,
        t_hub_h,
        hub_vagal_tone_arr=None,       # Mod 3: Hub_Vagal_Tone [0–1] — enteric motility
        hub_skin_blood_flow_arr=None,  # Mod 10: Hub_Skin_Blood_Flow [L/h] — ischemia gate
        t_span_h: tuple = (0.0, 6.0),
        n_save: int = 512,
    ) -> dict:
        params = self._build_params(bayesian_priors)

        t_hub     = jnp.asarray(t_hub_h,                   dtype=jnp.float32)
        ee_arr    = jnp.asarray(hub_energy_expenditure_arr, dtype=jnp.float32)
        cort_arr  = jnp.asarray(hub_cortisol_arr,           dtype=jnp.float32)
        reds_arr  = jnp.asarray(hub_reds_arr,               dtype=jnp.float32)
        gluc_arr  = jnp.asarray(meal_glucose_rate_arr,      dtype=jnp.float32)
        fruc_arr  = jnp.asarray(meal_fructose_rate_arr,     dtype=jnp.float32)
        fluid_arr = jnp.asarray(meal_fluid_rate_arr,        dtype=jnp.float32)
        na_arr    = jnp.asarray(meal_sodium_rate_arr,       dtype=jnp.float32)

        # Vagal tone: default 1.0 (resting) if not provided
        vagal_arr = (jnp.ones_like(t_hub) if hub_vagal_tone_arr is None
                     else jnp.asarray(hub_vagal_tone_arr, dtype=jnp.float32))
        # SkBF: default 0.0 (no ischemia) if not provided
        skbf_arr  = (jnp.zeros_like(t_hub) if hub_skin_blood_flow_arr is None
                     else jnp.asarray(hub_skin_blood_flow_arr, dtype=jnp.float32))

        ee_interp         = diffrax.LinearInterpolation(ts=t_hub, ys=ee_arr)
        cort_interp       = diffrax.LinearInterpolation(ts=t_hub, ys=cort_arr)
        reds_interp       = diffrax.LinearInterpolation(ts=t_hub, ys=reds_arr)
        gluc_rate_interp  = diffrax.LinearInterpolation(ts=t_hub, ys=gluc_arr)
        fruc_rate_interp  = diffrax.LinearInterpolation(ts=t_hub, ys=fruc_arr)
        fluid_rate_interp = diffrax.LinearInterpolation(ts=t_hub, ys=fluid_arr)
        na_rate_interp    = diffrax.LinearInterpolation(ts=t_hub, ys=na_arr)
        vagal_interp      = diffrax.LinearInterpolation(ts=t_hub, ys=vagal_arr)
        skbf_interp       = diffrax.LinearInterpolation(ts=t_hub, ys=skbf_arr)

        sol = _solve_gastrointestinal(
            params, t_span_h,
            ee_interp, cort_interp, reds_interp,
            gluc_rate_interp, fruc_rate_interp, fluid_rate_interp, na_rate_interp,
            vagal_interp, skbf_interp,
            n_save=n_save,
        )

        ts      = sol.ts
        V_s_arr  = sol.ys[:, 0]
        G_s_arr  = sol.ys[:, 1]
        F_s_arr  = sol.ys[:, 2]
        Na_s_arr = sol.ys[:, 3]
        G_i_arr  = sol.ys[:, 4]
        F_i_arr  = sol.ys[:, 5]
        W_i_arr  = sol.ys[:, 6]
        Na_i_arr = sol.ys[:, 7]

        # Instantaneous absorption rates at saved time points
        G_i_pos  = jnp.maximum(G_i_arr, 0.0)
        F_i_pos  = jnp.maximum(F_i_arr, 0.0)
        W_i_pos  = jnp.maximum(W_i_arr, 1e-6)
        Na_i_pos = jnp.maximum(Na_i_arr, 0.0)

        cort_on_ts = jnp.clip(jnp.interp(ts, t_hub, cort_arr), 0.0, None)
        reds_on_ts = jnp.clip(jnp.interp(ts, t_hub, reds_arr), 0.0, None)

        Cort_REDS_gate = 1.0 / (1.0 + params.k_Cort_abs * cort_on_ts + params.k_REDS_abs * reds_on_ts)

        synergy_arr  = 1.0 + params.k_synergy * F_i_pos / (F_i_pos + params.Km_syn)
        Gluc_abs_arr = (params.Vmax_SGLT1 * params.shannon_scale
                        * G_i_pos / (params.Km_SGLT1 + G_i_pos)
                        * synergy_arr * Cort_REDS_gate)
        Fruc_abs_arr = (params.Vmax_GLUT5 * params.shannon_scale
                        * F_i_pos / (params.Km_GLUT5 + F_i_pos)
                        * Cort_REDS_gate)

        # Sodium absorption rate (SGLT1 co-transport + passive)
        Na_SGLT1_arr   = _NA_SGLT1_RATIO * Gluc_abs_arr
        Na_i_conc_arr  = Na_i_pos / W_i_pos
        Na_passive_arr = (params.k_Na_passive
                          * jnp.maximum(Na_i_conc_arr - params.Na_blood_mg_ml, 0.0)
                          * W_i_pos)
        Na_abs_arr     = jnp.maximum(Na_SGLT1_arr + Na_passive_arr, 0.0)

        # Isotonic water absorption
        Na_osm_i_arr  = 2.0 * (Na_i_pos / _NA_MW_MG_MMOL) / (W_i_pos / 1000.0)
        Osm_i_arr     = (G_i_pos + F_i_pos) / W_i_pos * _CHO_OSM + Na_osm_i_arr
        Osm_abs_gate  = params.Osm_blood ** 2 / (params.Osm_blood ** 2 + Osm_i_arr ** 2)
        J_water_arr   = params.k_water_abs * W_i_pos * Osm_abs_gate

        # Convert to per-minute rates for hub outbounds
        gluc_rate_gmin    = Gluc_abs_arr / 60.0    # g·min⁻¹
        fruc_rate_gmin    = Fruc_abs_arr / 60.0    # g·min⁻¹
        fluid_rate_mlmin  = J_water_arr  / 60.0    # ml·min⁻¹
        na_rate_mgmin     = Na_abs_arr   / 60.0    # mg·min⁻¹

        # GI Distress Index [0–10]
        vol_distress = jnp.clip(W_i_arr / params.W_distress_ref * 5.0, 0.0, 5.0)
        osm_distress = jnp.clip(
            jnp.maximum(Osm_i_arr - params.Osm_distress_ref, 0.0) / 100.0, 0.0, 5.0
        )
        GI_distress = jnp.clip(vol_distress + osm_distress, 0.0, 10.0)

        return {
            "t_h":                          ts,
            "Stomach_Volume_ml":            V_s_arr,
            "Stomach_Glucose_g":            G_s_arr,
            "Stomach_Fructose_g":           F_s_arr,
            "Stomach_Na_mg":                Na_s_arr,
            "Intestine_Glucose_g":          G_i_arr,
            "Intestine_Fructose_g":         F_i_arr,
            "Intestine_Water_ml":           W_i_arr,
            "Intestine_Na_mg":              Na_i_arr,
            "Hub_Glucose_Absorption_Rate":  gluc_rate_gmin,
            "Hub_Fructose_Absorption_Rate": fruc_rate_gmin,
            "Hub_Fluid_Absorption_Rate":    fluid_rate_mlmin,
            "Hub_Sodium_Absorption_Rate":   na_rate_mgmin,
            "Hub_GI_Distress_Index":        GI_distress,
        }
