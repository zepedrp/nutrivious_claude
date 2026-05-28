from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import diffrax


class ImmuneRepairParams(NamedTuple):
    # Neutrophil kinetics (acute phase responder — Clermont 2004)
    k_Neut_act: float        # CK-driven neutrophil activation rate
    k_Neut_rep_inh: float    # repair-driven resolution: inhibits neutrophil activation
    k_Neut_decay: float      # neutrophil clearance (t½ ≈ 8h in tissue)
    # TNF-alpha kinetics (very short t½ ≈ 30min; primary acute mediator)
    k_TNFa_prod: float       # TNF-a production from neutrophils
    k_TNFa_clear: float      # TNF-a base clearance (ApoE-modulated at runtime)
    # IL-6 kinetics (cascade from TNF-a, Zak 2010 fractional Hill ^0.6)
    k_IL6_prod: float        # IL-6 production from TNF-a
    k_IL6_clear: float       # IL-6 base clearance (ApoE-modulated, slow systemic)
    k_IL6_rep_inh: float     # IL-6 active inhibition by repair (resolution)
    # CRP kinetics (hepatic synthesis with lag; first-order approach to IL-6-driven SS)
    k_CRP_prod: float        # CRP approach rate (t½ ≈ 19h)
    k_CRP_IL6: float         # IL-6 → CRP amplification factor
    CRP_baseline: float      # healthy resting CRP (mg/L)
    # Hormonal gate (M→Resolution transition; tanh sigmoid)
    k_gate: float            # gate sharpness
    theta_gate: float        # gate threshold (Anabolic − Catabolic offset)
    # mTOR proxy: anabolic synergy amplifiers for repair rate
    k_Testo_mTOR: float
    k_T3_mTOR: float
    k_GH_mTOR: float
    # Repair synthesis and damage kinetics
    k_T_synth: float         # repair synthesis rate (GATE × mTOR)
    k_damage_influx: float   # structural damage accumulation rate from CK
    # NLR dynamics
    k_Cort_lymph: float      # cortisol-driven lymphopenia coefficient
    L0_ref: float            # normalized reference lymphocyte count
    # Genetic priors (resolved to scalars before construction)
    tnfa_scale: float        # tnfa_rs308_prior → TNF-a production amplitude
    il6_scale: float         # il6_rs174_prior → IL-6 production amplitude
    col1a1_scale: float      # col1a1_rs1800012_prior → repair efficiency
    apoe_clearance: float    # apoe_genotype_prior → inflammatory clearance rate
    # Melatonin anti-inflammatory coupling (Mod 4 → Mod 7)
    k_Mel_inflam: float      # melatonin → enhanced TNF-a and IL-6 clearance (Carrillo-Vico 2013)


def immune_repair_ode(t, y, args):
    params, ck_interp, cort_interp, testo_interp, reds_interp, t_tone_interp, gh_interp, mel_interp = args
    Neut   = y[0]
    TNFa   = y[1]
    IL6    = y[2]
    CRP    = y[3]
    T_rep  = y[4]  # cumulative repair integral (monotonically increasing)
    R_debt = y[5]  # cumulative net debt = ΣDamage − ΣRepair

    CK_excess = jnp.clip(ck_interp.evaluate(t),      0.0, None)
    Cort      = jnp.clip(cort_interp.evaluate(t),    0.0, None)
    Testo     = jnp.clip(testo_interp.evaluate(t),   0.0, None)
    REDS      = jnp.clip(reds_interp.evaluate(t),    0.0, None)
    T_tone    = jnp.clip(t_tone_interp.evaluate(t),  0.0, None)
    GH        = jnp.clip(gh_interp.evaluate(t),      0.0, None)
    Mel       = jnp.clip(mel_interp.evaluate(t),     0.0, None)

    Neut_pos = jnp.maximum(Neut, 0.0)
    TNFa_pos = jnp.maximum(TNFa, 0.0)
    IL6_pos  = jnp.maximum(IL6,  0.0)
    CRP_pos  = jnp.maximum(CRP,  0.0)

    # Hormonal gate: resolution window opens when Anabolic > Catabolic + threshold
    Anabolic  = Testo * T_tone * GH
    Catabolic = Cort + REDS
    GATE      = 0.5 * (1.0 + jnp.tanh(params.k_gate * (Anabolic - Catabolic - params.theta_gate)))

    # mTORC1 proxy: multiplicative anabolic synergy amplifies repair rate
    mTOR_synergy = (
        (1.0 + params.k_Testo_mTOR * Testo)
        * (1.0 + params.k_T3_mTOR  * T_tone)
        * (1.0 + params.k_GH_mTOR  * GH)
    )

    repair_rate = params.k_T_synth * params.col1a1_scale * GATE * mTOR_synergy
    damage_rate = params.k_damage_influx * CK_excess

    # Neutrophils: activated by CK; inhibited by repair (active resolution feedback)
    dNeut = (
        params.k_Neut_act * CK_excess / (1.0 + params.k_Neut_rep_inh * repair_rate)
        - params.k_Neut_decay * Neut_pos
    )

    # TNF-alpha: produced by neutrophils; clearance rate modulated by ApoE genotype
    # Melatonin (Carrillo-Vico 2013): amplifies TNF-a and IL-6 clearance at night
    # via NF-κB suppression and enhanced macrophage resolution signalling
    k_TNFa_clear_eff = params.k_TNFa_clear * params.apoe_clearance * (1.0 + params.k_Mel_inflam * Mel)
    dTNFa = (
        params.tnfa_scale * params.k_TNFa_prod * Neut_pos
        - k_TNFa_clear_eff * TNFa_pos
    )

    # IL-6: cascade from TNF-a (Zak 2010 ^0.6); ApoE-modulated clearance;
    #        melatonin further amplifies resolution; actively resolved by repair
    k_IL6_clear_eff = params.k_IL6_clear * params.apoe_clearance * (1.0 + params.k_Mel_inflam * Mel)
    dIL6 = (
        params.il6_scale * params.k_IL6_prod * (jnp.maximum(TNFa_pos, 1e-6) ** jnp.float32(0.6))
        - k_IL6_clear_eff * IL6_pos
        - params.k_IL6_rep_inh * repair_rate * IL6_pos
    )

    # CRP: hepatic first-order lag approach to IL-6-driven steady state
    CRP_target = params.CRP_baseline + params.k_CRP_IL6 * IL6_pos
    dCRP = params.k_CRP_prod * (CRP_target - CRP_pos)

    # T_rep: cumulative repair integral (monotonically increasing)
    dT_rep  = repair_rate

    # R_debt: running net structural debt; positive = unresolved damage
    dR_debt = damage_rate - repair_rate

    return jnp.array([dNeut, dTNFa, dIL6, dCRP, dT_rep, dR_debt])


def _solve_immune_repair(
    params: ImmuneRepairParams,
    t_span_h: tuple,
    ck_interp,
    cort_interp,
    testo_interp,
    reds_interp,
    t_tone_interp,
    gh_interp,
    mel_interp,
    y0,
    n_save: int = 512,
) -> diffrax.Solution:
    term       = diffrax.ODETerm(immune_repair_ode)
    solver     = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    t0, t1     = float(t_span_h[0]), float(t_span_h[1])
    saveat     = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save))
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.05,
        y0=y0,
        args=(params, ck_interp, cort_interp, testo_interp, reds_interp, t_tone_interp, gh_interp, mel_interp),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=65536,
    )


class ImmuneRepairSolver:
    # Neutrophil kinetics
    _K_NEUT_ACT_REF:      float = 0.60
    _K_NEUT_REP_INH_REF:  float = 1.50
    _K_NEUT_DECAY_REF:    float = 0.6931 / 8.0    # t½ ≈ 8h in tissue

    # TNF-alpha kinetics (stiff: very short t½)
    _K_TNFA_PROD_REF:     float = 2.00
    _K_TNFA_CLEAR_REF:    float = 0.6931 / 0.5    # t½ ≈ 30min systemic

    # IL-6 kinetics
    _K_IL6_PROD_REF:      float = 1.50
    _K_IL6_CLEAR_REF:     float = 0.6931 / 3.0    # t½ ≈ 3h slow systemic
    _K_IL6_REP_INH_REF:   float = 0.80

    # CRP kinetics (hepatic, slow approach)
    _K_CRP_PROD_REF:      float = 0.6931 / 19.0   # t½ ≈ 19h
    _K_CRP_IL6_REF:       float = 0.35             # IL6 → CRP amplification
    _CRP_BASELINE_REF:    float = 0.50             # mg/L healthy resting

    # Hormonal gate
    _K_GATE_REF:          float = 3.0
    _THETA_GATE_REF:      float = 0.20

    # mTOR anabolic amplifiers
    _K_TESTO_MTOR_REF:    float = 0.40
    _K_T3_MTOR_REF:       float = 0.35
    _K_GH_MTOR_REF:       float = 0.45

    # Repair and damage kinetics
    _K_T_SYNTH_REF:       float = 0.15
    _K_DAMAGE_INFLUX_REF: float = 0.50

    # NLR dynamics
    _K_CORT_LYMPH_REF:    float = 0.40
    _L0_REF_REF:          float = 1.00

    # Melatonin anti-inflammatory (Carrillo-Vico 2013)
    _K_MEL_INFLAM_REF:    float = 0.50  # per normalized melatonin unit

    def _build_params(self, bayesian_priors: dict) -> ImmuneRepairParams:
        tnfa_raw   = bayesian_priors.get("tnfa_rs308_prior",        float("nan"))
        il6_raw    = bayesian_priors.get("il6_rs174_prior",         float("nan"))
        col1a1_raw = bayesian_priors.get("col1a1_rs1800012_prior",  float("nan"))
        apoe_raw   = bayesian_priors.get("apoe_genotype_prior",     float("nan"))

        tnfa_scale     = 1.0 if math.isnan(float(tnfa_raw))   else max(0.5, min(2.0, float(tnfa_raw)))
        il6_scale      = 1.0 if math.isnan(float(il6_raw))    else max(0.5, min(2.0, float(il6_raw)))
        col1a1_scale   = 1.0 if math.isnan(float(col1a1_raw)) else max(0.5, min(1.5, float(col1a1_raw)))
        apoe_clearance = 1.0 if math.isnan(float(apoe_raw))   else max(0.5, min(1.5, float(apoe_raw)))

        return ImmuneRepairParams(
            k_Neut_act      = self._K_NEUT_ACT_REF,
            k_Neut_rep_inh  = self._K_NEUT_REP_INH_REF,
            k_Neut_decay    = self._K_NEUT_DECAY_REF,
            k_TNFa_prod     = self._K_TNFA_PROD_REF,
            k_TNFa_clear    = self._K_TNFA_CLEAR_REF,
            k_IL6_prod      = self._K_IL6_PROD_REF,
            k_IL6_clear     = self._K_IL6_CLEAR_REF,
            k_IL6_rep_inh   = self._K_IL6_REP_INH_REF,
            k_CRP_prod      = self._K_CRP_PROD_REF,
            k_CRP_IL6       = self._K_CRP_IL6_REF,
            CRP_baseline    = self._CRP_BASELINE_REF,
            k_gate          = self._K_GATE_REF,
            theta_gate      = self._THETA_GATE_REF,
            k_Testo_mTOR    = self._K_TESTO_MTOR_REF,
            k_T3_mTOR       = self._K_T3_MTOR_REF,
            k_GH_mTOR       = self._K_GH_MTOR_REF,
            k_T_synth       = self._K_T_SYNTH_REF,
            k_damage_influx = self._K_DAMAGE_INFLUX_REF,
            k_Cort_lymph    = self._K_CORT_LYMPH_REF,
            L0_ref          = self._L0_REF_REF,
            tnfa_scale      = tnfa_scale,
            il6_scale       = il6_scale,
            col1a1_scale    = col1a1_scale,
            apoe_clearance  = apoe_clearance,
            k_Mel_inflam    = self._K_MEL_INFLAM_REF,
        )

    def simulate_immune_repair(
        self,
        bayesian_priors: dict,
        hub_ck_arr,             # Mod 2: Hub_CK_StructuralDamage (D_muscle, [0,1])
        hub_cortisol_arr,       # Mod 5: Hub_Cortisol_Catabolic
        hub_testosterone_arr,   # Mod 5: Hub_Testosterone_Anabolic
        hub_reds_arr,           # Mod 5: Hub_REDS_State
        hub_t_tone_arr,         # Mod 6: Hub_Thyroid_Anabolic_Tone
        hub_gh_arr,             # Mod 4: Hub_GH_Repair_Signalling
        t_hub_h,
        hub_melatonin_arr=None, # Mod 4: Hub_Melatonin_Tone — anti-inflammatory (Carrillo-Vico 2013)
        t_span_h: tuple = (0.0, 96.0),
    ) -> dict:
        params = self._build_params(bayesian_priors)

        t_hub      = jnp.asarray(t_hub_h,               dtype=jnp.float32)
        ck_arr     = jnp.asarray(hub_ck_arr,            dtype=jnp.float32)
        cort_arr   = jnp.asarray(hub_cortisol_arr,      dtype=jnp.float32)
        testo_arr  = jnp.asarray(hub_testosterone_arr,  dtype=jnp.float32)
        reds_arr   = jnp.asarray(hub_reds_arr,          dtype=jnp.float32)
        t_tone_arr = jnp.asarray(hub_t_tone_arr,        dtype=jnp.float32)
        gh_arr     = jnp.asarray(hub_gh_arr,            dtype=jnp.float32)
        mel_arr    = (jnp.zeros_like(t_hub) if hub_melatonin_arr is None
                      else jnp.asarray(hub_melatonin_arr, dtype=jnp.float32))

        ck_interp     = diffrax.LinearInterpolation(ts=t_hub, ys=ck_arr)
        cort_interp   = diffrax.LinearInterpolation(ts=t_hub, ys=cort_arr)
        testo_interp  = diffrax.LinearInterpolation(ts=t_hub, ys=testo_arr)
        reds_interp   = diffrax.LinearInterpolation(ts=t_hub, ys=reds_arr)
        t_tone_interp = diffrax.LinearInterpolation(ts=t_hub, ys=t_tone_arr)
        gh_interp     = diffrax.LinearInterpolation(ts=t_hub, ys=gh_arr)
        mel_interp    = diffrax.LinearInterpolation(ts=t_hub, ys=mel_arr)

        # CRP initialized at healthy baseline; all inflammatory states at zero
        y0 = jnp.array(
            [0.0, 0.0, 0.0, params.CRP_baseline, 0.0, 0.0],
            dtype=jnp.float32,
        )

        sol = _solve_immune_repair(
            params, t_span_h,
            ck_interp, cort_interp, testo_interp,
            reds_interp, t_tone_interp, gh_interp,
            mel_interp,
            y0,
        )

        ts         = sol.ts
        Neut_arr   = sol.ys[:, 0]
        TNFa_arr   = sol.ys[:, 1]
        IL6_arr    = sol.ys[:, 2]
        CRP_arr    = sol.ys[:, 3]
        T_rep_arr  = sol.ys[:, 4]
        R_debt_arr = sol.ys[:, 5]

        # Repair completion: fraction of total damage load resolved
        R_debt_pos        = jnp.maximum(R_debt_arr, 0.0)
        repair_completion = T_rep_arr / (T_rep_arr + R_debt_pos + 1e-4)

        # Salivary NLR: cortisol-driven lymphopenia elevates neutrophil-to-lymphocyte ratio
        cort_on_ts = jnp.clip(jnp.interp(ts, t_hub, cort_arr), 0.0, None)
        Lymph_eff  = params.L0_ref / (1.0 + params.k_Cort_lymph * cort_on_ts)
        NLR_arr    = jnp.maximum(Neut_arr, 0.0) / (Lymph_eff + 1e-6)

        return {
            "t_h":                          ts,
            "Neutrophils_act":              Neut_arr,
            "Hub_Cytokine_TNFa":            jnp.maximum(TNFa_arr, 0.0),
            "Hub_Cytokine_IL6_Systemic":    jnp.maximum(IL6_arr,  0.0),
            "Hub_Biomarker_CRP_mg_L":       jnp.maximum(CRP_arr,  params.CRP_baseline),
            "Hub_Biomarker_Salivary_NLR":   NLR_arr,
            "Hub_Tissue_Repair_Completion": repair_completion,
            "Hub_Repair_Debt":              R_debt_pos,
        }
