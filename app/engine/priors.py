"""
app/engine/priors.py — Bayesian Prior Generator (NLME Layer)

Zero imports outside the standard library.

Exports
-------
build_engine_priors(athlete_data, genotype) -> dict[str, float]
    Isolated SciML scale-multiplier generator.

SNP_LOOKUPS          — allele → physical parameter value (for phase3_envelope)
BAYESIAN_PRIOR_MAPPINGS — Phase 2 field → canonical MHDS prior name + transform
HARD_CEILINGS        — inviolable physiological limits (MPC hard constraints)
"""
from __future__ import annotations

from types import MappingProxyType

# ─────────────────────────────────────────────────────────────────────────────
# 1. SNP LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────────────

SNP_LOOKUPS: MappingProxyType = MappingProxyType({

    "ACTN3_tau_fatigue_days": MappingProxyType({
        "RR": 28.0,
        "RX": 35.0,
        "XX": 42.0,
    }),
    "ACE_cardiac_efficiency": MappingProxyType({
        "II": 1.10,
        "ID": 1.00,
        "DD": 0.90,
    }),
    "MSTN_inhibition": MappingProxyType({
        "KK": 0.85,
        "KA": 1.00,
        "AA": 1.15,
    }),
    "MTHFR_C677T_methylation": MappingProxyType({
        "CC": 1.00,
        "CT": 0.93,
        "TT": 0.82,
    }),
    "PPARGC1A_oxphos": MappingProxyType({
        "GG": 1.15,
        "GA": 1.00,
        "AA": 0.85,
    }),
    "HIF1A_hypoxia": MappingProxyType({
        "CC": 1.00,
        "CT": 1.02,
        "TT": 1.04,
    }),
    "ACTN3_vo2max": MappingProxyType({
        "RR": 0.99,
        "RX": 1.00,
        "XX": 1.02,
    }),
    "ACTN3_power": MappingProxyType({
        "RR": 1.05,
        "RX": 1.00,
        "XX": 0.86,
    }),
})


# ─────────────────────────────────────────────────────────────────────────────
# 2. PHASE 2 → MHDS PRIOR MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────

BAYESIAN_PRIOR_MAPPINGS: MappingProxyType = MappingProxyType({

    "homa_ir": MappingProxyType({
        "prior_name":      "p3_insulin_sensitivity_prior",
        "subsystem":       "bioenergetic_metabolic",
        "transform":       "reciprocal_normalised",
        "reference_value": 1.0,
        "units":           "a.u.",
    }),
    "glucose_fasting_mg_dL": MappingProxyType({
        "prior_name":      "p3_basal_glucose_prior",
        "subsystem":       "bioenergetic_metabolic",
        "transform":       "direct_mg_dL_to_mmol_L",
        "units":           "mmol·L⁻¹",
    }),
    "PPARGC1A": MappingProxyType({
        "prior_name":      "p3_mitochondrial_efficiency_prior",
        "subsystem":       "bioenergetic_metabolic",
        "transform":       "ppargc1a_allele_to_oxphos",
        "units":           "a.u.",
    }),
    "ACTN3": MappingProxyType({
        "prior_name":      "tau_fatigue_decay_prior",
        "subsystem":       "neuromuscular_fatigue",
        "transform":       "actn3_allele_to_tau",
        "units":           "days",
    }),
    "myh7_fiber_type_fraction": MappingProxyType({
        "prior_name":      "myh7_fiber_type_prior",
        "subsystem":       "neuromuscular_fatigue",
        "transform":       "direct",
        "units":           "fraction",
    }),
    "MSTN": MappingProxyType({
        "prior_name":      "p3_myostatin_inhibition_prior",
        "subsystem":       "neuromuscular_fatigue",
        "transform":       "mstn_allele_to_inhibition",
        "units":           "a.u.",
    }),
    "ACE": MappingProxyType({
        "prior_name":      "ace_indel_prior",
        "subsystem":       "cardiorespiratory_autonomic",
        "transform":       "ace_allele_to_efficiency",
        "units":           "a.u.",
    }),
    "hif1a_prior": MappingProxyType({
        "prior_name":      "hif1a_prior",
        "subsystem":       "cardiorespiratory_autonomic",
        "transform":       "direct",
        "units":           "a.u.",
    }),
    "rmssd_baseline_ms": MappingProxyType({
        "prior_name":      "p3_vagal_tone_prior",
        "subsystem":       "cardiorespiratory_autonomic",
        "transform":       "direct",
        "reference_value": 50.0,
        "units":           "ms",
    }),
    "mctq_msf_sc": MappingProxyType({
        "prior_name":      "p3_circadian_phase_prior",
        "subsystem":       "sleep_circadian",
        "transform":       "msf_to_phi_rad",
        "reference_value": 3.5,
        "units":           "radians",
    }),
    "pvt_rt_baseline_ms": MappingProxyType({
        "prior_name":      "p3_alertness_decay_prior",
        "subsystem":       "sleep_circadian",
        "transform":       "pvt_to_alertness_rate",
        "reference_value": 250.0,
        "units":           "a.u.",
    }),
    "testosterone_pg_mL": MappingProxyType({
        "prior_name":      "p3_androgen_setpoint_prior",
        "subsystem":       "hpa_hpg_neuroendocrine",
        "transform":       "direct",
        "reference_value": 500.0,
        "units":           "pg·mL⁻¹",
    }),
    "crp_mg_L": MappingProxyType({
        "prior_name":      "p3_inflammatory_baseline_prior",
        "subsystem":       "immunologic_inflammatory",
        "transform":       "direct",
        "reference_value": 0.5,
        "units":           "mg·L⁻¹",
    }),
    "ck_u_L": MappingProxyType({
        "prior_name":      "p3_muscle_damage_baseline_prior",
        "subsystem":       "immunologic_inflammatory",
        "transform":       "direct",
        "reference_value": 150.0,
        "units":           "U·L⁻¹",
    }),
    "shannon_diversity": MappingProxyType({
        "prior_name":      "p3_absorption_efficiency_prior",
        "subsystem":       "gi_absorption",
        "transform":       "diversity_to_eta",
        "reference_value": 3.5,
        "units":           "a.u.",
    }),
    "MTHFR_C677T": MappingProxyType({
        "prior_name":      "p3_folate_methylation_prior",
        "subsystem":       "gi_absorption",
        "transform":       "mthfr_allele_to_efficiency",
        "units":           "a.u.",
    }),
    "pvt_rt_ms": MappingProxyType({
        "prior_name":      "p3_central_fatigue_tau_prior",
        "subsystem":       "cognitive_central_fatigue",
        "transform":       "pvt_to_alertness_rate",
        "reference_value": 250.0,
        "units":           "a.u.",
    }),
    "tendon_stiffness_N_mm": MappingProxyType({
        "prior_name":      "p3_tendon_elasticity_prior",
        "subsystem":       "biomechanical_tissue",
        "transform":       "direct",
        "reference_value": 200.0,
        "units":           "N·mm⁻¹",
    }),
    "ferritin_ng_mL": MappingProxyType({
        "prior_name":      "p3_iron_stores_prior",
        "subsystem":       "hydroelectrolytic_renal",
        "transform":       "direct",
        "reference_value": 80.0,
        "units":           "ng·mL⁻¹",
    }),
    "hemoglobin_g_dL": MappingProxyType({
        "prior_name":      "p3_oxygen_transport_prior",
        "subsystem":       "hydroelectrolytic_renal",
        "transform":       "direct",
        "reference_value": 15.0,
        "units":           "g·dL⁻¹",
    }),
    "epigenetic_pace_index": MappingProxyType({
        "prior_name":      "p3_epigenetic_pace_prior",
        "subsystem":       "reds_lea",
        "transform":       "direct",
        "reference_value": 1.0,
        "units":           "yr_epigenetic_per_yr_calendar",
    }),
    "measured_rmr": MappingProxyType({
        "prior_name":      "p3_energy_availability_prior",
        "subsystem":       "reds_lea",
        "transform":       "direct",
        "reference_value": None,
        "units":           "kcal·day⁻¹",
    }),
})


# ─────────────────────────────────────────────────────────────────────────────
# 3. HARD CEILINGS — inviolable physiological limits (MPC hard constraints)
# ─────────────────────────────────────────────────────────────────────────────

HARD_CEILINGS: MappingProxyType = MappingProxyType({
    "t_core_prescribe_max_C":                   39.0,
    "t_core_abort_C":                           40.0,
    "plasma_na_alert_mmol_L":                   135.0,
    "plasma_na_abort_mmol_L":                   132.0,
    "energy_availability_floor_kcal_kg_ffm":    30.0,
    "cho_absorption_normal_ceiling_g_h":        90.0,
    "cho_absorption_gut_trained_ceiling_g_h":   120.0,
    "protein_per_meal_min_g_kg":                0.30,
    "protein_per_meal_max_g_kg":                0.55,
    "sleep_minimum_h":                          7.0,
    "caffeine_max_mg_kg_per_bout":              6.0,
    "acwr_floor":                               0.8,
    "acwr_ceiling":                             1.3,
    "w_prime_bal_floor_kJ":                     0.0,
    "hr_prescription_max_fraction":             0.95,
})


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD ENGINE PRIORS — isolated SciML scale-multiplier generator
# ─────────────────────────────────────────────────────────────────────────────

_SCALE_SNP_MAP: dict[str, dict[str, dict[str, float]]] = {
    "ACTN3": {
        "RR": {"tau_VO2_scale": 0.95, "W_prime_scale": 1.05},
        "RX": {"tau_VO2_scale": 1.00, "W_prime_scale": 1.00},
        "XX": {"tau_VO2_scale": 1.05, "W_prime_scale": 0.94},
    },
    "ACE": {
        "II": {"cardiac_eff_scale": 1.05},
        "ID": {"cardiac_eff_scale": 1.00},
        "DD": {"cardiac_eff_scale": 0.95},
    },
    "DIO2_Thr92Ala": {
        "Thr/Thr": {"k_t3_conv_scale": 1.00},
        "Thr/Ala": {"k_t3_conv_scale": 0.94},
        "Ala/Ala": {"k_t3_conv_scale": 0.88},
    },
    "COMT_Val158Met": {
        "Val/Val": {"dopamine_clear_scale": 1.00},
        "Val/Met": {"dopamine_clear_scale": 0.95},
        "Met/Met": {"dopamine_clear_scale": 0.88},
    },
}

_NEUTRAL: dict[str, float] = {
    "tau_VO2_scale":        1.0,
    "W_prime_scale":        1.0,
    "cardiac_eff_scale":    1.0,
    "k_t3_conv_scale":      1.0,
    "dopamine_clear_scale": 1.0,
    "HR_max_prior":         185.0,
}


def build_engine_priors(
    athlete_data: dict,
    genotype: dict,
) -> dict[str, float]:
    """
    Build SciML scale multipliers from athlete onboarding data and genotype.

    Genetics enters as a weak prior shift: Δ = 0.5 × (allele_shift − 1.0).

    Parameters
    ----------
    athlete_data : dict  — expected key: "age_years" (optional)
    genotype     : dict  — expected keys: "ACTN3", "ACE",
                           "DIO2_Thr92Ala", "COMT_Val158Met" (all optional)

    Returns
    -------
    dict[str, float]
        Keys: tau_VO2_scale, W_prime_scale, cardiac_eff_scale,
              k_t3_conv_scale, dopamine_clear_scale, HR_max_prior.
        All values are 1.0 (neutral) if the corresponding input is absent.
    """
    out = dict(_NEUTRAL)

    for snp_key, allele_map in _SCALE_SNP_MAP.items():
        allele = genotype.get(snp_key)
        if allele is None:
            continue
        shifts = allele_map.get(allele, {})
        for param, shift in shifts.items():
            if param in out:
                out[param] = out[param] + 0.5 * (shift - 1.0)

    age = athlete_data.get("age_years")
    if age is not None:
        try:
            age_f = float(age)
            if age_f > 30:
                out["HR_max_prior"] = max(140.0, 208.0 - 0.7 * age_f)
        except (TypeError, ValueError):
            pass

    return out
