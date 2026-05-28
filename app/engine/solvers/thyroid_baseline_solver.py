from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import diffrax


class ThyroidBaselineParams(NamedTuple):
    # TSH kinetics (half-life ~1 h)
    k_TSH_prod: float
    k_TSH_decay: float
    k_Cort_TSH: float
    k_REDS_TSH: float
    K_fb_T3: float
    # T4 kinetics (half-life ~7 days = 168 h); SPINA G_T secretion
    G_T: float
    k_T4_decay: float
    # Deiodinase kinetics
    k_D12_base: float
    k_D3_base: float
    k_REDS_D12: float
    k_REDS_D3: float
    # T3 & rT3 clearance
    k_T3_decay: float
    k_rT3_decay: float
    # Competitive inhibition constant (T_tone denominator)
    K_i_rT3: float
    # Genetic priors (resolved to scalars before construction)
    dio2_scale: float
    ppargc1a_scale: float
    # BMR thermogenesis
    BMR_lean_kcal_min: float
    k_NE_BMR: float
    # Temperature asymptote (tanh, survival floor)
    T_basal_base: float   # euthyroid ceiling (°C)
    T_floor: float        # hypothyroid survival floor (°C)
    k_T_asym: float       # tanh sharpness
    # RQ stoichiometry (tanh transition, lipid → glycolytic)
    RQ_fat: float         # pure lipid RQ at T_tone_norm=1.0
    RQ_carb: float        # glycolytic RQ at T_tone_norm→0
    k_RQ: float           # transition sharpness
    theta_RQ: float       # T_tone_norm midpoint of RQ transition


def thyroid_hpt_ode(t, y, args):
    params, cort_interp, reds_interp = args
    TSH = y[0]
    T4  = y[1]
    T3  = y[2]
    rT3 = y[3]

    Cort = jnp.clip(cort_interp.evaluate(t), 0.0, None)
    REDS = jnp.clip(reds_interp.evaluate(t), 0.0, None)

    # D3 shunt: REDS inhibits D1/D2 (less T3), amplifies D3 (more rT3)
    k_D12_eff = params.k_D12_base * params.dio2_scale / (1.0 + params.k_REDS_D12 * REDS)
    k_D3_eff  = params.k_D3_base * (1.0 + params.k_REDS_D3 * REDS)

    # TSH: hypothalamic drive inhibited by Cortisol and REDS; T3 negative feedback (SPINA)
    I_Cort = 1.0 / (1.0 + params.k_Cort_TSH * Cort)
    I_REDS = 1.0 / (1.0 + params.k_REDS_TSH * REDS)
    f_fb   = params.K_fb_T3 / (params.K_fb_T3 + jnp.maximum(T3, 1e-6))
    dTSH   = params.k_TSH_prod * I_Cort * I_REDS * f_fb - params.k_TSH_decay * TSH

    # T4: secreted by G_T × TSH; consumed by D12 (→T3), D3 (→rT3), direct clearance
    T4s  = jnp.maximum(T4, 0.0)
    TSHs = jnp.maximum(TSH, 0.0)
    dT4  = params.G_T * TSHs - (k_D12_eff + k_D3_eff + params.k_T4_decay) * T4s

    # T3: produced from T4 via D12; cleared
    dT3  = k_D12_eff * T4s - params.k_T3_decay * jnp.maximum(T3, 0.0)

    # rT3: produced from T4 via D3; cleared rapidly (t½ = 4 h)
    drT3 = k_D3_eff * T4s - params.k_rT3_decay * jnp.maximum(rT3, 0.0)

    return jnp.array([dTSH, dT4, dT3, drT3])


def _solve_thyroid_hpt(
    params: ThyroidBaselineParams,
    t_span_h: tuple,
    cort_interp,
    reds_interp,
    n_save: int = 512,
) -> diffrax.Solution:
    term       = diffrax.ODETerm(thyroid_hpt_ode)
    solver     = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    t0, t1     = float(t_span_h[0]), float(t_span_h[1])
    saveat     = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save))
    y0         = jnp.ones(4, dtype=jnp.float32)  # all states normalized to 1.0
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y0,
        args=(params, cort_interp, reds_interp),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=65536,
    )


class ThyroidBaselineSolver:
    # Half-life-derived rate constants: k = ln(2) / t_half_h
    _K_TSH_DECAY_REF:  float = 0.6931 / 1.0    # TSH   t½ = 1 h
    _K_T4_DECAY_REF:   float = 0.6931 / 168.0  # T4    t½ = 7 days
    _K_D12_BASE_REF:   float = 0.6931 / 24.0   # D12 → T3  (t½ T3 = 24 h)
    _K_D3_BASE_REF:    float = 0.6931 / 4.0    # D3  → rT3 (t½ rT3 = 4 h)
    _K_T3_DECAY_REF:   float = 0.6931 / 24.0
    _K_RTT3_DECAY_REF: float = 0.6931 / 4.0

    # G_T: SPINA steady-state balance → G_T = k_D12 + k_D3 + k_T4_decay (with TSH=T4=1)
    _G_T_REF: float = (0.6931 / 24.0) + (0.6931 / 4.0) + (0.6931 / 168.0)

    # TSH_prod: with K_fb=1 → f_fb_SS=0.5; k_prod = 2 × k_decay for unit SS
    _K_TSH_PROD_REF: float = 2.0 * (0.6931 / 1.0)

    # Inhibition / feedback strengths
    _K_CORT_TSH_REF: float = 0.30
    _K_REDS_TSH_REF: float = 0.50
    _K_FB_T3_REF:    float = 1.0
    _K_REDS_D12_REF: float = 0.80
    _K_REDS_D3_REF:  float = 2.0
    _K_I_RTT3_REF:   float = 1.0  # competitive inhibition in T_tone formula

    # BMR thermogenesis
    _BMR_LEAN_KCAL_MIN_REF: float = 1.2    # kcal·min⁻¹ for a 70 kg lean athlete
    _K_NE_BMR_REF:          float = 0.10   # catecholamine-driven UCP thermogenic gain

    # Temperature asymptote (tanh: smooth survival floor, never breached)
    _T_BASAL_BASE_REF: float = 36.6  # euthyroid ceiling (°C)
    _T_FLOOR_REF:      float = 35.5  # hypothyroid survival floor (°C)
    _K_T_ASYM_REF:     float = 4.0   # tanh sharpness

    # RQ stoichiometry (tanh transition between lipid and glycolytic metabolism)
    _RQ_FAT_REF:   float = 0.72  # pure lipid oxidation (T_tone_norm = 1.0)
    _RQ_CARB_REF:  float = 0.85  # glycolytic dominance (T_tone_norm → 0)
    _K_RQ_REF:     float = 3.0   # transition sharpness
    _THETA_RQ_REF: float = 0.5   # T_tone_norm midpoint

    def _build_params(self, bayesian_priors: dict) -> ThyroidBaselineParams:
        dio2_raw     = bayesian_priors.get("dio2_rs225014_prior",       float("nan"))
        ppargc1a_raw = bayesian_priors.get("ppargc1a_gly482ser_prior",  float("nan"))
        dio2_scale    = 1.0 if math.isnan(float(dio2_raw))     else float(dio2_raw)
        ppargc1a_scale = 1.0 if math.isnan(float(ppargc1a_raw)) else float(ppargc1a_raw)
        return ThyroidBaselineParams(
            k_TSH_prod        = self._K_TSH_PROD_REF,
            k_TSH_decay       = self._K_TSH_DECAY_REF,
            k_Cort_TSH        = self._K_CORT_TSH_REF,
            k_REDS_TSH        = self._K_REDS_TSH_REF,
            K_fb_T3           = self._K_FB_T3_REF,
            G_T               = self._G_T_REF,
            k_T4_decay        = self._K_T4_DECAY_REF,
            k_D12_base        = self._K_D12_BASE_REF,
            k_D3_base         = self._K_D3_BASE_REF,
            k_REDS_D12        = self._K_REDS_D12_REF,
            k_REDS_D3         = self._K_REDS_D3_REF,
            k_T3_decay        = self._K_T3_DECAY_REF,
            k_rT3_decay       = self._K_RTT3_DECAY_REF,
            K_i_rT3           = self._K_I_RTT3_REF,
            dio2_scale        = dio2_scale,
            ppargc1a_scale    = ppargc1a_scale,
            BMR_lean_kcal_min = self._BMR_LEAN_KCAL_MIN_REF,
            k_NE_BMR          = self._K_NE_BMR_REF,
            T_basal_base      = self._T_BASAL_BASE_REF,
            T_floor           = self._T_FLOOR_REF,
            k_T_asym          = self._K_T_ASYM_REF,
            RQ_fat            = self._RQ_FAT_REF,
            RQ_carb           = self._RQ_CARB_REF,
            k_RQ              = self._K_RQ_REF,
            theta_RQ          = self._THETA_RQ_REF,
        )

    def simulate_thyroid_baseline(
        self,
        bayesian_priors: dict,
        hub_cortisol_arr,
        hub_reds_arr,
        t_hub_h,
        hub_catecholamines_arr=None,
        t_span_h: tuple = (0.0, 168.0),
    ) -> dict:
        params = self._build_params(bayesian_priors)

        t_hub    = jnp.asarray(t_hub_h,          dtype=jnp.float32)
        cort_arr = jnp.asarray(hub_cortisol_arr, dtype=jnp.float32)
        reds_arr = jnp.asarray(hub_reds_arr,     dtype=jnp.float32)
        cat_arr  = (
            jnp.zeros_like(t_hub)
            if hub_catecholamines_arr is None
            else jnp.asarray(hub_catecholamines_arr, dtype=jnp.float32)
        )

        cort_interp = diffrax.LinearInterpolation(ts=t_hub, ys=cort_arr)
        reds_interp = diffrax.LinearInterpolation(ts=t_hub, ys=reds_arr)

        sol = _solve_thyroid_hpt(params, t_span_h, cort_interp, reds_interp)

        ts      = sol.ts
        TSH_arr = sol.ys[:, 0]
        T4_arr  = sol.ys[:, 1]
        T3_arr  = sol.ys[:, 2]
        rT3_arr = sol.ys[:, 3]

        # T_tone: competitive inhibition of T3 by rT3 at receptor level
        # Normalized: T_tone_norm = 1.0 at baseline (T3=rT3=1 → T_tone_ref = 1/(1+K_i))
        T_tone_ref  = 1.0 / (1.0 + params.K_i_rT3)
        T3_pos      = jnp.maximum(T3_arr,  0.0)
        rT3_pos     = jnp.maximum(rT3_arr, 0.0)
        T_tone      = T3_pos / (T3_pos + params.K_i_rT3 * rT3_pos + 1e-6)
        T_tone_norm = T_tone / T_tone_ref

        # Catecholamines interpolated onto solution time axis (UCP thermogenesis)
        cat_on_ts = jnp.clip(jnp.interp(ts, t_hub, cat_arr), 0.0, None)

        # BMR: lean baseline × ppargc1a (mitochondrial ceiling) × T_tone × catecholamine-UCP gain
        BMR_arr = (
            params.BMR_lean_kcal_min
            * params.ppargc1a_scale
            * T_tone_norm
            * (1.0 + params.k_NE_BMR * cat_on_ts)
        )

        # RQ: tanh sigmoid inverse — lipid-dominant at euthyroid, glycolytic at rT3 dominance
        # RQ = RQ_fat + (RQ_carb − RQ_fat) × 0.5 × (1 − tanh(k_RQ × (T_tone_norm − θ_RQ)))
        RQ_arr = params.RQ_fat + (params.RQ_carb - params.RQ_fat) * 0.5 * (
            1.0 - jnp.tanh(params.k_RQ * (T_tone_norm - params.theta_RQ))
        )

        # Temperature: tanh asymptote — smooth approach to T_floor, never breached
        # T_basal = T_floor + (T_base − T_floor) × (0.5 + 0.5 × tanh(k_T × (T_tone_norm − 0.5)))
        T_basal_arr = params.T_floor + (params.T_basal_base - params.T_floor) * (
            0.5 + 0.5 * jnp.tanh(params.k_T_asym * (T_tone_norm - 0.5))
        )

        return {
            "t_h":                               ts,
            "TSH":                               TSH_arr,
            "T4":                                T4_arr,
            "T3":                                T3_arr,
            "rT3":                               rT3_arr,
            "Hub_Basal_Metabolic_Rate_Kcal_Min": BMR_arr,
            "Hub_Thyroid_Anabolic_Tone":         T_tone_norm,
            "Hub_Resting_RQ":                    RQ_arr,
            "Hub_Basal_Temperature_Morning":     T_basal_arr,
        }
