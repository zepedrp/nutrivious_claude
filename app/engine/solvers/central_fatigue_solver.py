from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
import diffrax


class CentralFatigueParams(NamedTuple):
    # Serotonin Load kinetics (tryptophan-BCAA competition + brain influx)
    k_S_prod: float       # h⁻¹ — basal tryptophan influx rate
    k_IL6_S: float        # IL6 amplification of serotonin load (sickness behavior)
    k_debt_S: float       # sleep debt amplification of serotonin production
    k_Lact_S: float       # lactate → albumin-bound tryptophan release → brain entry
    k_S_clear: float      # h⁻¹ — baseline serotonin reuptake (SERT-mediated)
    # Dopamine Drive kinetics (prefrontal cortex motivational tone)
    k_D_prod: float       # catecholamine → cortical dopamine synthesis
    k_D_clear: float      # h⁻¹ — COMT-mediated prefrontal clearance
    k_Lact_D: float       # lactate pain → dopamine decay amplifier (per mmol/L above threshold)
    Lact_thresh: float    # mmol/L — pain threshold for dopamine decay amplification
    k_GI_D: float         # GI distress (visceral panic) → dopamine decay amplifier
    # RPE Borg mapping [6–20] via S/D ratio sigmoid (Marcora psychobiological model)
    k_RPE: float          # sigmoid sharpness of S/D ratio → RPE
    theta_RPE: float      # S/D ratio at RPE midpoint (Borg ≈ 13)
    # Governor survival thresholds (Noakes catastrophe model)
    Lact_crit: float      # mmol/L — lactate crisis boundary
    IL6_crit: float       # normalized IL6 crisis boundary
    GI_crit: float        # GI distress crisis (0–10 scale)
    Debt_crit: float      # sleep debt crisis (normalized)
    k_Lact_gov: float     # threat sigmoid sharpness — lactate
    k_IL6_gov: float      # threat sigmoid sharpness — IL6
    k_GI_gov: float       # threat sigmoid sharpness — GI
    k_debt_gov: float     # threat sigmoid sharpness — sleep debt
    k_RPE_gov: float      # threat sigmoid sharpness — RPE normalized
    # Thermal fatigue (Nybo & Nielsen 2001 — T_core > 38.5°C degrades dopamine drive)
    k_Temp_D: float       # °C⁻¹ above 38.5°C — thermal decay amplifier on dopamine
    Temp_crit: float      # °C — core temperature governor crisis boundary
    k_Temp_gov: float     # threat sigmoid sharpness — T_core
    # Genetic priors (resolved to scalars)
    comt_scale: float     # COMT enzyme activity [0.5, 1.5]: higher = faster dopamine clearance
    sertpr_scale: float   # SERT expression [0.5, 1.5]: higher = less reuptake = more serotonin accumulation


def central_fatigue_ode(t, y, args):
    params, lact_interp, cat_interp, debt_interp, il6_interp, gi_interp, core_temp_interp = args

    S = y[0]   # Serotonin Load (normalized)
    D = y[1]   # Dopamine Drive (normalized)

    Lact = jnp.clip(lact_interp.evaluate(t), 0.0, None)
    Cat  = jnp.clip(cat_interp.evaluate(t),  0.0, None)
    Debt = jnp.clip(debt_interp.evaluate(t), 0.0, None)
    IL6    = jnp.clip(il6_interp.evaluate(t),       0.0, None)
    GI     = jnp.clip(gi_interp.evaluate(t),        0.0, None)
    T_core = jnp.clip(core_temp_interp.evaluate(t), 36.0, 42.0)

    S_pos = jnp.maximum(S, 0.0)
    D_pos = jnp.maximum(D, 0.0)

    # Serotonin Load: rises with exercise duration, exponentially amplified by
    # IL6 (sickness behavior) and Sleep Debt (basal apathy); lactate drives
    # tryptophan release from albumin → accelerated brain entry
    dS = (
        params.k_S_prod
        * (1.0 + params.k_IL6_S * IL6 + params.k_debt_S * Debt)
        * (1.0 + params.k_Lact_S * Lact)
        - (params.k_S_clear / params.sertpr_scale) * S_pos
    )

    # Dopamine Drive: synthesised from catecholamines (fight/flight substrate);
    # COMT-mediated clearance (comt_scale) is brutally accelerated by pain
    # (lactate above threshold), visceral panic (GI distress), and hyperthermia
    # (Nybo & Nielsen 2001, J Physiol: T_core > 38.5°C → dopaminergic motor drive collapses)
    decay_amp = (
        1.0
        + params.k_Lact_D * jnp.maximum(Lact - params.Lact_thresh, 0.0)
        + params.k_GI_D * GI
        + params.k_Temp_D * jnp.maximum(T_core - 38.5, 0.0)
    )
    dD = (
        params.k_D_prod * Cat
        - params.k_D_clear * params.comt_scale * D_pos * decay_amp
    )

    return jnp.array([dS, dD])


def _solve_central_fatigue(
    params: CentralFatigueParams,
    t_span_h: tuple,
    lact_interp,
    cat_interp,
    debt_interp,
    il6_interp,
    gi_interp,
    core_temp_interp,
    n_save: int = 512,
) -> diffrax.Solution:
    term       = diffrax.ODETerm(central_fatigue_ode)
    solver     = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    t0, t1     = float(t_span_h[0]), float(t_span_h[1])
    saveat     = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save))
    y0         = jnp.array([0.0, 1.0], dtype=jnp.float32)  # S=0 (rested), D=1 (baseline drive)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.001,
        y0=y0,
        args=(params, lact_interp, cat_interp, debt_interp, il6_interp, gi_interp, core_temp_interp),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=65536,
    )


class CentralFatigueSolver:
    # Serotonin Load
    _K_S_PROD_REF:  float = 0.10   # h⁻¹ — basal tryptophan influx
    _K_IL6_S_REF:   float = 0.50   # per unit IL6
    _K_DEBT_S_REF:  float = 0.30   # per unit sleep debt
    _K_LACT_S_REF:  float = 0.40   # per mmol/L lactate
    _K_S_CLEAR_REF: float = 0.50   # h⁻¹ baseline SERT-mediated reuptake

    # Dopamine Drive
    _K_D_PROD_REF:    float = 0.80  # catecholamine → dopamine gain
    _K_D_CLEAR_REF:   float = 0.60  # h⁻¹ × comt_scale
    _K_LACT_D_REF:    float = 0.30  # per mmol/L above pain threshold
    _LACT_THRESH_REF: float = 2.0   # mmol/L — pain onset for dopamine decay
    _K_GI_D_REF:      float = 0.20  # per GI distress unit

    # RPE Borg mapping (Marcora S/D ratio → perceived exertion)
    _K_RPE_REF:     float = 3.0    # sigmoid sharpness
    _THETA_RPE_REF: float = 1.5    # S/D ratio at Borg 13 (effort midpoint)

    # Governor critical thresholds (Noakes catastrophe)
    _LACT_CRIT_REF:  float = 6.0   # mmol/L — acidosis crisis
    _IL6_CRIT_REF:   float = 3.0   # normalized IL6 crisis
    _GI_CRIT_REF:    float = 7.0   # GI distress crisis (out of 10)
    _DEBT_CRIT_REF:  float = 2.0   # sleep debt crisis (normalized)
    _K_LACT_GOV_REF: float = 2.0   # threat sharpness
    _K_IL6_GOV_REF:  float = 1.5
    _K_GI_GOV_REF:   float = 1.0
    _K_DEBT_GOV_REF: float = 1.5
    _K_RPE_GOV_REF:  float = 5.0   # high sharpness: Borg 17 (RPE_norm=0.80) triggers rapidly

    # Thermal fatigue (Nybo & Nielsen 2001)
    _K_TEMP_D_REF:   float = 0.40  # per °C above 38.5°C — dopamine decay amplifier
    _TEMP_CRIT_REF:  float = 40.0  # °C — heat-stroke governor boundary
    _K_TEMP_GOV_REF: float = 2.0   # threat sigmoid sharpness for T_core

    def _build_params(self, bayesian_priors: dict) -> CentralFatigueParams:
        comt_raw   = bayesian_priors.get("comt_val158met_prior",  float("nan"))
        sertpr_raw = bayesian_priors.get("slc6a4_5httlpr_prior",  float("nan"))
        comt_scale   = (1.0 if math.isnan(float(comt_raw))
                        else float(np.clip(float(comt_raw), 0.5, 1.5)))
        sertpr_scale = (1.0 if math.isnan(float(sertpr_raw))
                        else float(np.clip(float(sertpr_raw), 0.5, 1.5)))
        return CentralFatigueParams(
            k_S_prod     = self._K_S_PROD_REF,
            k_IL6_S      = self._K_IL6_S_REF,
            k_debt_S     = self._K_DEBT_S_REF,
            k_Lact_S     = self._K_LACT_S_REF,
            k_S_clear    = self._K_S_CLEAR_REF,
            k_D_prod     = self._K_D_PROD_REF,
            k_D_clear    = self._K_D_CLEAR_REF,
            k_Lact_D     = self._K_LACT_D_REF,
            Lact_thresh  = self._LACT_THRESH_REF,
            k_GI_D       = self._K_GI_D_REF,
            k_RPE        = self._K_RPE_REF,
            theta_RPE    = self._THETA_RPE_REF,
            Lact_crit    = self._LACT_CRIT_REF,
            IL6_crit     = self._IL6_CRIT_REF,
            GI_crit      = self._GI_CRIT_REF,
            Debt_crit    = self._DEBT_CRIT_REF,
            k_Lact_gov   = self._K_LACT_GOV_REF,
            k_IL6_gov    = self._K_IL6_GOV_REF,
            k_GI_gov     = self._K_GI_GOV_REF,
            k_debt_gov   = self._K_DEBT_GOV_REF,
            k_RPE_gov    = self._K_RPE_GOV_REF,
            k_Temp_D     = self._K_TEMP_D_REF,
            Temp_crit    = self._TEMP_CRIT_REF,
            k_Temp_gov   = self._K_TEMP_GOV_REF,
            comt_scale   = comt_scale,
            sertpr_scale = sertpr_scale,
        )

    def simulate_central_fatigue(
        self,
        bayesian_priors: dict,
        hub_lactate_arr,           # Mod 1: Hub_Lactate_Signalling [mmol/L]
        hub_catecholamines_arr,    # Mod 3: Hub_Catecholamines_Tone [normalized]
        hub_sleep_debt_arr,        # Mod 4: Hub_Sleep_Debt_Metabolic [normalized]
        hub_il6_arr,               # Mod 7: Hub_Cytokine_IL6_Systemic [normalized]
        hub_gi_distress_arr,       # Mod 8: Hub_GI_Distress_Index [0–10]
        hub_core_temp_arr,         # Mod 10: Hub_Core_Temp [°C] — Nybo & Nielsen 2001
        t_hub_h,
        t_span_h: tuple = (0.0, 4.0),
        n_save: int = 512,
    ) -> dict:
        params = self._build_params(bayesian_priors)

        t_hub      = jnp.asarray(t_hub_h,               dtype=jnp.float32)
        lact_arr   = jnp.asarray(hub_lactate_arr,        dtype=jnp.float32)
        cat_arr    = jnp.asarray(hub_catecholamines_arr, dtype=jnp.float32)
        debt_arr   = jnp.asarray(hub_sleep_debt_arr,     dtype=jnp.float32)
        il6_arr    = jnp.asarray(hub_il6_arr,            dtype=jnp.float32)
        gi_arr     = jnp.asarray(hub_gi_distress_arr,    dtype=jnp.float32)
        temp_arr   = jnp.asarray(hub_core_temp_arr,      dtype=jnp.float32)

        lact_interp      = diffrax.LinearInterpolation(ts=t_hub, ys=lact_arr)
        cat_interp       = diffrax.LinearInterpolation(ts=t_hub, ys=cat_arr)
        debt_interp      = diffrax.LinearInterpolation(ts=t_hub, ys=debt_arr)
        il6_interp       = diffrax.LinearInterpolation(ts=t_hub, ys=il6_arr)
        gi_interp        = diffrax.LinearInterpolation(ts=t_hub, ys=gi_arr)
        core_temp_interp = diffrax.LinearInterpolation(ts=t_hub, ys=temp_arr)

        sol = _solve_central_fatigue(
            params, t_span_h,
            lact_interp, cat_interp, debt_interp, il6_interp, gi_interp,
            core_temp_interp,
            n_save=n_save,
        )

        ts    = sol.ts
        S_arr = sol.ys[:, 0]
        D_arr = sol.ys[:, 1]

        S_pos = jnp.maximum(S_arr, 0.0)
        D_pos = jnp.maximum(D_arr, 1e-6)

        # RPE Borg [6, 20] — Marcora: perceived effort = f(S/D ratio)
        SD_ratio = S_pos / (D_pos + 1e-6)
        RPE_norm = 0.5 * (1.0 + jnp.tanh(params.k_RPE * (SD_ratio - params.theta_RPE)))
        RPE_Borg = jnp.clip(6.0 + 14.0 * RPE_norm, 6.0, 20.0)

        # Reconstruct hub signals on solution time axis for Governor computation
        lact_on_ts = jnp.clip(jnp.interp(ts, t_hub, lact_arr), 0.0, None)
        il6_on_ts  = jnp.clip(jnp.interp(ts, t_hub, il6_arr),  0.0, None)
        gi_on_ts   = jnp.clip(jnp.interp(ts, t_hub, gi_arr),   0.0, None)
        debt_on_ts = jnp.clip(jnp.interp(ts, t_hub, debt_arr), 0.0, None)
        temp_on_ts = jnp.clip(jnp.interp(ts, t_hub, temp_arr), 36.0, 42.0)

        # Governor of Survival (Noakes): multiplicative product of sigmoidal threats
        # CFI = 1 − ∏ₖ(1−Tₖ)
        # If ANY single Tₖ → 1.0 (isolated system failure), product → 0, CFI → 1.0
        def _threat(x):
            return 0.5 * (1.0 + jnp.tanh(x))

        T_Lact  = _threat(params.k_Lact_gov * (lact_on_ts - params.Lact_crit))
        T_IL6   = _threat(params.k_IL6_gov  * (il6_on_ts  - params.IL6_crit))
        T_GI    = _threat(params.k_GI_gov   * (gi_on_ts   - params.GI_crit))
        T_Sleep = _threat(params.k_debt_gov * (debt_on_ts - params.Debt_crit))
        T_RPE   = _threat(params.k_RPE_gov  * (RPE_norm   - 0.80))
        T_Temp  = _threat(params.k_Temp_gov * (temp_on_ts - params.Temp_crit))

        survival_product = (
            (1.0 - T_Lact)
            * (1.0 - T_IL6)
            * (1.0 - T_GI)
            * (1.0 - T_Sleep)
            * (1.0 - T_RPE)
            * (1.0 - T_Temp)
        )
        CFI       = jnp.clip(1.0 - survival_product, 0.0, 1.0)
        Motor_Cap = jnp.clip(1.0 - CFI,              0.0, 1.0)

        return {
            "t_h":                       ts,
            "Serotonin_Load":            S_arr,
            "Dopamine_Drive":            D_arr,
            "Hub_RPE_Borg":              RPE_Borg,
            "Hub_Central_Fatigue_Index": CFI,
            "Hub_Motor_Recruitment_Cap": Motor_Cap,
        }
