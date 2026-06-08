"""
app/engine/phase3_envelope.py — Phase 3 Envelope (Priors + Constraints)

The Phase 3 envelope is NOT a multiplicative model. It is a contract:
  - bayesian_priors  : flat dict {param_name: float} — NaN for absent biomarkers.
                       Consumed by L3 (NLME) and L4 (state filter).
  - hard_constraints : inviolable safety floors/ceilings for the MPC.
  - chance_constraints: probabilistic safety bounds P(violation) ≤ α.
  - refer_out_rules  : conditions that trigger Path A refer-out language.

Public API
----------
    build_engine_priors(athlete_data, genotype, phase1_ceilings=None)
        → dict[str, float]   — flat bayesian_priors ready for NutriviousEngine

    build_phase3_envelope(athlete_data, genotype, phase1_ceilings=None)
        → Phase3Envelope     — full contract for L6 MPC

Rule: this module never performs performance calculations. It translates raw
measurements into probability-distribution parameters and safety constraints.

Reference: Nutrivious_HLD_Sistema_Hibrido.md §3; CLAUDE.md §3
"""
from __future__ import annotations

import math
from typing import NamedTuple

from app.engine.priors import (
    BAYESIAN_PRIOR_MAPPINGS,
    HARD_CEILINGS,
    SNP_LOOKUPS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTRAINT TYPES
# ─────────────────────────────────────────────────────────────────────────────

class Constraint(NamedTuple):
    """
    A hard constraint on a state variable: lhs op rhs always holds.

    Fields
    ------
    name       : human-readable identifier
    state_key  : the hub/state variable key this constraint acts on
    op         : '≤' | '≥'
    rhs        : threshold value (same units as state_key)
    units      : physical units string
    source_doi : citation
    """
    name:       str
    state_key:  str
    op:         str   # '≤' or '≥'
    rhs:        float
    units:      str
    source_doi: str


class ChanceConstraint(NamedTuple):
    """
    A probabilistic constraint: P(state_key op rhs) ≤ α.

    Tightened to a hard constraint in the MPC via: rhs_tightened = rhs ∓ k·σ_x
    where k = Φ⁻¹(1 - α) and σ_x comes from the state filter covariance.

    Fields
    ------
    name       : human-readable identifier
    state_key  : the hub/state variable key
    op         : '≤' | '≥'
    rhs        : threshold value
    alpha      : maximum allowed probability of violation (e.g. 0.01 = 1%)
    units      : physical units
    source_doi : citation
    """
    name:       str
    state_key:  str
    op:         str
    rhs:        float
    alpha:      float
    units:      str
    source_doi: str


class ReferOutRule(NamedTuple):
    """
    A condition under which Path A refer-out language must be emitted.
    The system pauses performance recommendations until acknowledged.

    Fields
    ------
    name       : rule identifier
    state_key  : the observable triggering the rule
    op         : '≤' | '≥'
    threshold  : value that triggers referral
    message    : Path A-compliant user-facing message (no medical language)
    """
    name:      str
    state_key: str
    op:        str
    threshold: float
    message:   str


class Phase3Envelope(NamedTuple):
    """
    The full Phase 3 contract. Consumed by NLME (L3), state filter (L4), MPC (L6).

    Fields
    ------
    bayesian_priors     : flat {prior_name: float} — NaN for absent inputs;
                          used to initialise θ_i in NLME and warm-start filter
    hard_constraints    : list of inviolable Constraint objects for MPC
    chance_constraints  : list of ChanceConstraint for MPC tightening
    refer_out_rules     : list of ReferOutRule for Path A safety gate
    """
    bayesian_priors:    dict
    hard_constraints:   list
    chance_constraints: list
    refer_out_rules:    list


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORM ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _apply_transform(transform: str, raw: float | str,
                     reference_value: float | None) -> float | None:
    """
    Apply a named transform to a raw biomarker/allele value → physical prior.
    Returns None if the transform cannot be applied (unknown allele, etc.).
    Callers replace None with float("nan").
    """
    if transform == "direct":
        return float(raw)

    if transform == "reciprocal_normalised":
        val = float(raw)
        if val == 0.0:
            return None
        ref = reference_value if reference_value is not None else 1.0
        return ref / val

    if transform == "direct_mg_dL_to_mmol_L":
        return float(raw) / 18.0182

    if transform == "actn3_allele_to_tau":
        return SNP_LOOKUPS["ACTN3_tau_fatigue_days"].get(str(raw))

    if transform == "ace_allele_to_efficiency":
        return SNP_LOOKUPS["ACE_cardiac_efficiency"].get(str(raw))

    if transform == "mstn_allele_to_inhibition":
        return SNP_LOOKUPS["MSTN_inhibition"].get(str(raw))

    if transform == "mthfr_allele_to_efficiency":
        return SNP_LOOKUPS["MTHFR_C677T_methylation"].get(str(raw))

    if transform == "ppargc1a_allele_to_oxphos":
        return SNP_LOOKUPS["PPARGC1A_oxphos"].get(str(raw))

    if transform == "msf_to_phi_rad":
        ref = reference_value if reference_value is not None else 3.5
        return (float(raw) - ref) * (math.pi / 12.0)

    if transform == "pvt_to_alertness_rate":
        val = float(raw)
        if val == 0.0:
            return None
        ref = reference_value if reference_value is not None else 250.0
        return ref / val

    if transform == "diversity_to_eta":
        ref = reference_value if reference_value is not None else 3.5
        return min(1.0, float(raw) / ref)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRIOR BUILDER — Phase 2 → MHDS parameter dict
# ─────────────────────────────────────────────────────────────────────────────

def build_engine_priors(
    athlete_data: dict,
    genotype: dict[str, str],
    phase1_ceilings: dict | None = None,
) -> dict[str, float]:
    """
    Translate Phase 2 athlete data + genotype into a flat bayesian_priors dict
    consumed by NutriviousEngine.simulate_daily_cycle().

    Absent biomarkers produce float("nan") — the only JAX-safe sentinel for
    missing data (None would TypeError inside jnp.array).

    Parameters
    ----------
    athlete_data
        Flat dict of biomarker values keyed by clinical field names.
        Example keys: "homa_ir", "glucose_fasting_mg_dL", "rmssd_baseline_ms",
        "crp_mg_L", "ferritin_ng_mL", "hemoglobin_g_dL".
    genotype
        Flat dict of SNP genotype strings.
        Example keys: "ACTN3" → "RX", "ACE" → "II", "PPARGC1A" → "GG".
    phase1_ceilings
        Optional flat dict from Phase 1 species tables. When provided, hard
        ceiling values are included as constraint scalars in the priors dict
        under the "ceiling_*" namespace.

    Returns
    -------
    dict[str, float]
        Flat prior dict. NaN signals an absent/untransformable input.
    """
    priors: dict[str, float] = {}

    for source_field, mapping in BAYESIAN_PRIOR_MAPPINGS.items():
        prior_name: str     = mapping["prior_name"]
        transform: str      = mapping["transform"]
        reference_value     = mapping.get("reference_value")

        # Source priority: athlete_data > genotype
        raw = athlete_data.get(source_field)
        if raw is None:
            raw = genotype.get(source_field)
        if raw is None:
            priors[prior_name] = float("nan")
            continue

        result = _apply_transform(transform, raw, reference_value)
        priors[prior_name] = result if result is not None else float("nan")

    # Pass-through metadata fields the engine uses directly
    for key in ("fat_free_mass_kg", "body_weight_kg", "energy_intake_kcal_per_day"):
        if key in athlete_data:
            priors[key] = float(athlete_data[key])

    # Expose Phase 1 hard ceilings as scalars in the prior dict (read-only)
    if phase1_ceilings:
        for k, v in phase1_ceilings.items():
            if v is not None:
                priors[f"ceiling_{k}"] = float(v)

    return priors


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSTRAINT CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────

def _build_hard_constraints() -> list[Constraint]:
    """Return the master list of inviolable MPC constraints."""
    return [
        Constraint(
            name="t_core_prescribe_ceiling",
            state_key="Hub_Core_Temp",
            op="≤",
            rhs=HARD_CEILINGS["t_core_prescribe_max_C"],
            units="°C",
            source_doi="10.1249/MSS.0b013e318149f22c",
        ),
        Constraint(
            name="t_core_abort_ceiling",
            state_key="Hub_Core_Temp",
            op="≤",
            rhs=HARD_CEILINGS["t_core_abort_C"],
            units="°C",
            source_doi="10.1249/MSS.0b013e318149f22c",
        ),
        Constraint(
            name="plasma_na_abort_floor",
            state_key="Hub_Sodium_Concentration",
            op="≥",
            rhs=HARD_CEILINGS["plasma_na_abort_mmol_L"],
            units="mmol·L⁻¹",
            source_doi="10.1136/bjsports-2014-093953",
        ),
        Constraint(
            name="energy_availability_floor",
            state_key="Hub_Energy_Availability_kcal_kg_ffm",
            op="≥",
            rhs=HARD_CEILINGS["energy_availability_floor_kcal_kg_ffm"],
            units="kcal·kg⁻¹·FFM·day⁻¹",
            source_doi="10.1136/bjsports-2018-099193",
        ),
        Constraint(
            name="acwr_floor",
            state_key="Hub_ACWR",
            op="≥",
            rhs=HARD_CEILINGS["acwr_floor"],
            units="a.u.",
            source_doi="10.1136/bjsports-2015-095445",
        ),
        Constraint(
            name="acwr_ceiling",
            state_key="Hub_ACWR",
            op="≤",
            rhs=HARD_CEILINGS["acwr_ceiling"],
            units="a.u.",
            source_doi="10.1136/bjsports-2015-095445",
        ),
        Constraint(
            name="w_prime_balance_floor",
            state_key="Hub_W_Prime_Balance_kJ",
            op="≥",
            rhs=HARD_CEILINGS["w_prime_bal_floor_kJ"],
            units="kJ",
            source_doi="10.1371/journal.pone.0030956",
        ),
        Constraint(
            name="sleep_floor",
            state_key="Hub_Sleep_Duration_h",
            op="≥",
            rhs=HARD_CEILINGS["sleep_minimum_h"],
            units="h",
            source_doi="10.1126/science.1149441",
        ),
    ]


def _build_chance_constraints() -> list[ChanceConstraint]:
    """
    Return probabilistic safety bounds for the MPC.
    P(state_key op rhs) ≤ alpha under the current filter uncertainty.
    """
    return [
        ChanceConstraint(
            name="eah_hyponatraemia_alert",
            state_key="Hub_Sodium_Concentration",
            op="≤",
            rhs=HARD_CEILINGS["plasma_na_alert_mmol_L"],
            alpha=0.01,
            units="mmol·L⁻¹",
            source_doi="10.1136/bjsports-2014-093953",
        ),
        ChanceConstraint(
            name="hyperthermia_chance_constraint",
            state_key="Hub_Core_Temp",
            op="≥",
            rhs=39.5,
            alpha=0.05,
            units="°C",
            source_doi="10.1249/MSS.0b013e318149f22c",
        ),
        ChanceConstraint(
            name="rmssd_suppression_alert",
            state_key="Hub_RMSSD_relative",
            op="≤",
            rhs=-1.0,   # < -1σ below personal baseline
            alpha=0.10,
            units="σ",
            source_doi="10.1007/s10484-013-9235-5",
        ),
    ]


def _build_refer_out_rules() -> list[ReferOutRule]:
    """
    Return conditions that trigger Path A refer-out language.
    Messages use performance/wellness vocabulary only (no medical claims).
    """
    return [
        ReferOutRule(
            name="resting_hr_elevated",
            state_key="Hub_HR_Rest_bpm",
            op="≥",
            threshold=100.0,
            message=(
                "Your resting heart rate has been consistently elevated. "
                "We're pausing performance recommendations until you check in "
                "with a qualified health professional."
            ),
        ),
        ReferOutRule(
            name="eah_risk_flag",
            state_key="Hub_Sodium_Concentration",
            op="≤",
            threshold=135.0,
            message=(
                "Some of your recent readings are outside your usual range. "
                "We're pausing hydration recommendations until you confirm "
                "with a qualified health professional."
            ),
        ),
        ReferOutRule(
            name="reds_energy_deficit_flag",
            state_key="Hub_Energy_Availability_kcal_kg_ffm",
            op="≤",
            threshold=30.0,
            message=(
                "Your estimated energy availability is below the recommended "
                "training floor. We're pausing load recommendations until "
                "you review your fuelling with a sports nutrition professional."
            ),
        ),
        ReferOutRule(
            name="hrv_sustained_suppression",
            state_key="Hub_RMSSD_relative",
            op="≤",
            threshold=-2.0,
            message=(
                "Your HRV has been consistently below your personal baseline "
                "for several days. We recommend a recovery-focused week and "
                "consultation with a sports health professional."
            ),
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOP-LEVEL PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_phase3_envelope(
    athlete_data: dict,
    genotype: dict[str, str],
    phase1_ceilings: dict | None = None,
) -> Phase3Envelope:
    """
    Build the complete Phase 3 contract (priors + constraints).

    This is the authoritative entry point for the Phase 3 layer.
    The orchestrator and MPC consume the returned Phase3Envelope.

    Parameters
    ----------
    athlete_data
        Flat dict of Phase 2 biomarker values.
    genotype
        Flat dict of SNP genotype strings.
    phase1_ceilings
        Optional flat dict from Phase 1 species tables.

    Returns
    -------
    Phase3Envelope
        Complete contract with priors, hard constraints, chance constraints,
        and refer-out rules. No performance calculations performed.
    """
    bayesian_priors     = build_engine_priors(athlete_data, genotype, phase1_ceilings)
    hard_constraints    = _build_hard_constraints()
    chance_constraints  = _build_chance_constraints()
    refer_out_rules     = _build_refer_out_rules()

    return Phase3Envelope(
        bayesian_priors    = bayesian_priors,
        hard_constraints   = hard_constraints,
        chance_constraints = chance_constraints,
        refer_out_rules    = refer_out_rules,
    )
