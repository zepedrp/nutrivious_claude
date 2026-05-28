from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesFluidElectrolyte(Base):
    """
    Phase 1 — Fluid compartment, electrolyte, and sweat ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 9 / Matrices 9.1–9.5 (TBW, sweat rate, electrolytes, osmolality, Starling forces);
      Domain VI.1 (body fluid compartments); Domain VI.2 (sweat gland kinetics);
      Domain VI.3 (plasma/interstitial Starling forces); Domain VI.4 (RAAS / aldosterone / AVP);
      Domain VI.5 (dehydration thresholds and hyponatremia limits).

    Key equations encoded:
      TBW = ICF + ECF = 0.60 × BM (male) / 0.50 × BM (female)
      J_v = K_f × [(Pc − Pi) − σ × (πc − πi)]        [Starling microvascular filtration]
      Osmolality_p ≈ 2 × [Na⁺] + [Glucose]/18 + [BUN]/2.8  [serum osmolality estimate]
      Sweat_heat_loss = Q_sweat × 2.43 kJ/mL           [evaporative heat dissipation]

    Units:
      _l          = litres (absolute)
      _ml_per_kg  = mL / kg body mass
      _l_per_h    = L / hour
      _meq_l      = mEq / L
      _mm         = mM (millimolar)
      _mosmol_kg  = mOsm / kg H₂O
      _mmhg       = mmHg
      _fraction   = dimensionless 0–1
      _kj_per_ml  = kJ / mL
      _meq_h      = mEq / hour
    """

    __tablename__ = "species_fluid_electrolyte"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="fluid_electrolyte")

    # ── Total Body Water (TBW) and Fluid Compartment Distribution ────────────
    # TBW = ICF (67%) + ECF (33%); ECF = plasma (25%) + interstitial (75%)
    tbw_fraction_body_mass_male: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.60; TBW as fraction of body mass in males (60% of BM = ~42 L in 70 kg); from Domain VI.1 / Matrix 9.1"
    )
    tbw_fraction_body_mass_female: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; TBW as fraction of body mass in females (higher adipose fraction reduces TBW%); from Domain VI.1"
    )
    tbw_elite_endurance_male_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.65–0.70; TBW fraction in elite male endurance athletes (lower adiposity → higher TBW%); from Domain VI.1"
    )
    icf_fraction_tbw: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.67; intracellular fluid (ICF) as fraction of TBW; ~28 L in 70 kg male; from Domain VI.1"
    )
    ecf_fraction_tbw: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.33; extracellular fluid (ECF) as fraction of TBW; ~14 L in 70 kg male; from Domain VI.1"
    )
    plasma_volume_fraction_ecf: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.25; plasma volume as fraction of ECF (~3.5 L in 70 kg male); from Domain VI.1"
    )
    interstitial_volume_fraction_ecf: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.75; interstitial fluid as fraction of ECF (~10.5 L in 70 kg male); from Domain VI.1"
    )
    lymphatic_return_l_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 L/day; lymphatic return to circulation lower bound; from Domain VI.1"
    )
    lymphatic_return_l_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0 L/day; lymphatic return to circulation upper bound; from Domain VI.1"
    )

    # ── Plasma Electrolyte Reference Ranges ────────────────────────────────────
    # ECF is the regulated compartment; plasma reflects ECF electrolyte status
    na_plasma_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="135 mEq/L; plasma [Na⁺] lower bound of normal (hyponatremia below this); from Domain VI.1"
    )
    na_plasma_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="145 mEq/L; plasma [Na⁺] upper bound of normal (hypernatremia above this); from Domain VI.1"
    )
    k_plasma_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.5 mEq/L; plasma [K⁺] lower bound of normal (hypokalemia below this); from Domain VI.1"
    )
    k_plasma_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 mEq/L; plasma [K⁺] upper bound of normal; from Domain VI.1"
    )
    cl_plasma_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="96 mEq/L; plasma [Cl⁻] lower bound of normal; from Domain VI.1"
    )
    cl_plasma_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="106 mEq/L; plasma [Cl⁻] upper bound of normal; from Domain VI.1"
    )
    hco3_plasma_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="22 mEq/L; plasma [HCO₃⁻] lower bound of normal; from Domain VI.1"
    )
    hco3_plasma_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="28 mEq/L; plasma [HCO₃⁻] upper bound of normal; from Domain VI.1"
    )
    mg_plasma_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.5 mEq/L; plasma [Mg²⁺] lower bound of normal (0.75 mM); from Domain VI.1"
    )
    mg_plasma_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.5 mEq/L; plasma [Mg²⁺] upper bound of normal (1.25 mM); from Domain VI.1"
    )
    ca_plasma_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.5 mEq/L; total plasma [Ca²⁺] lower bound of normal (2.25 mM); from Domain VI.1"
    )
    ca_plasma_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.5 mEq/L; total plasma [Ca²⁺] upper bound of normal (2.75 mM); from Domain VI.1"
    )
    albumin_plasma_g_per_dl_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.5 g/dL; plasma albumin lower bound; primary determinant of oncotic pressure; from Domain VI.3"
    )
    albumin_plasma_g_per_dl_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 g/dL; plasma albumin upper bound; from Domain VI.3"
    )

    # ── Intracellular Electrolyte Concentrations ───────────────────────────────
    # ICF maintained by Na⁺/K⁺-ATPase: 3 Na⁺ out / 2 K⁺ in / ATP; creates −70 mV resting potential
    na_icf_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 mEq/L; intracellular [Na⁺] lower bound; from Domain VI.1"
    )
    na_icf_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 mEq/L; intracellular [Na⁺] upper bound; from Domain VI.1"
    )
    k_icf_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="140 mEq/L; intracellular [K⁺] lower bound; dominant ICF cation; from Domain VI.1"
    )
    k_icf_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="155 mEq/L; intracellular [K⁺] upper bound; from Domain VI.1"
    )
    mg_icf_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 mEq/L; intracellular [Mg²⁺] lower bound (mostly protein-bound); from Domain VI.1"
    )
    mg_icf_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 mEq/L; intracellular [Mg²⁺] upper bound; from Domain VI.1"
    )
    nka_stoichiometry_na_per_atp: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0; Na⁺/K⁺-ATPase exports 3 Na⁺ per ATP hydrolysed (electrogenic); from Domain VI.1"
    )
    nka_stoichiometry_k_per_atp: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0; Na⁺/K⁺-ATPase imports 2 K⁺ per ATP hydrolysed; from Domain VI.1"
    )
    nka_fraction_basal_atp_consumption: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.20–0.40; fraction of basal cellular ATP consumed by Na⁺/K⁺-ATPase; from Domain VI.1"
    )

    # ── Plasma Osmolality Limits ───────────────────────────────────────────────
    # Osmolality_p ≈ 2×[Na⁺] + [Glc]/18 + [BUN]/2.8; [Na⁺] accounts for ~95% of ECF osmolality
    plasma_osmolality_normal_mosmol_kg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="275 mOsm/kg H₂O; plasma osmolality lower bound of normal; from Domain VI.1 / Matrix 9.2"
    )
    plasma_osmolality_normal_mosmol_kg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="295 mOsm/kg H₂O; plasma osmolality upper bound of normal; from Domain VI.1"
    )
    plasma_osmolality_adh_threshold_mosmol_kg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~295 mOsm/kg; osmolality threshold at which ADH (AVP) secretion begins; from Domain VI.4"
    )
    plasma_osmolality_thirst_threshold_mosmol_kg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~296 mOsm/kg; osmolality threshold for thirst sensation; slightly above ADH threshold; from Domain VI.4"
    )
    plasma_osmolality_max_exercise_dehydration_mosmol_kg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~310–320 mOsm/kg; maximum plasma osmolality during hypertonic exercise dehydration; from Domain VI.5"
    )
    icf_osmolality_normal_mosmol_kg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="285 mOsm/kg; intracellular osmolality lower bound (iso-osmotic with ECF at rest); from Domain VI.1"
    )
    icf_osmolality_normal_mosmol_kg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="295 mOsm/kg; intracellular osmolality upper bound; from Domain VI.1"
    )

    # ── Sweat Rate Ceilings ────────────────────────────────────────────────────
    # Eccrine sweat gland secretion; rate limited by gland density × secretory rate × precursor delivery
    # Evaporative efficiency depends on humidity: 100% efficient in dry heat, near 0% in saturated air
    sweat_rate_moderate_exercise_l_per_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 L/h; sweat rate during moderate exercise lower bound; from Domain VI.2 / Matrix 9.3"
    )
    sweat_rate_moderate_exercise_l_per_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 L/h; sweat rate during moderate exercise upper bound; from Domain VI.2"
    )
    sweat_rate_max_exercise_heat_l_per_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 L/h; maximum sweat rate during intense exercise + heat lower bound; from Domain VI.2"
    )
    sweat_rate_max_exercise_heat_l_per_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.5 L/h; maximum sweat rate during intense exercise + heat upper bound (acclimatised); from Domain VI.2"
    )
    sweat_rate_absolute_record_l_per_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~3.5 L/h; absolute species maximum sweat rate (extreme heat + maximal exercise); from Domain VI.2"
    )
    sweat_heat_dissipation_kj_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.43 kJ/mL; heat of vaporisation of water at 37°C (580 cal/g); 1 mL sweat evaporated = 2.43 kJ dissipated; from Domain VI.2"
    )
    eccrine_gland_density_per_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 glands/cm²; eccrine sweat gland density lower bound (body average); from Domain VI.2"
    )
    eccrine_gland_density_per_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="600 glands/cm²; eccrine sweat gland density upper bound (palms/soles); from Domain VI.2"
    )
    total_eccrine_glands_count_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2e6; total eccrine gland count lower bound; from Domain VI.2"
    )
    total_eccrine_glands_count_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4e6; total eccrine gland count upper bound; from Domain VI.2"
    )

    # ── Sweat Electrolyte Composition ─────────────────────────────────────────
    # [Na⁺] in sweat is primary driver of hyponatremia risk; aldosterone reduces it with acclimatisation
    # Sweat is always hypotonic relative to plasma (electrolyte-free water loss dominates)
    sweat_na_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 mEq/L; sweat [Na⁺] lower bound (heat-acclimatised / aldosterone-adapted); from Domain VI.2 / Matrix 9.3"
    )
    sweat_na_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="90 mEq/L; sweat [Na⁺] upper bound (unacclimatised / high individual variation); from Domain VI.2"
    )
    sweat_na_typical_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 mEq/L; sweat [Na⁺] typical lower bound; from Domain VI.2"
    )
    sweat_na_typical_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 mEq/L; sweat [Na⁺] typical upper bound; from Domain VI.2"
    )
    sweat_cl_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 mEq/L; sweat [Cl⁻] lower bound; follows Na⁺ closely (co-secreted); from Domain VI.2"
    )
    sweat_cl_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 mEq/L; sweat [Cl⁻] upper bound; from Domain VI.2"
    )
    sweat_k_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 mEq/L; sweat [K⁺] lower bound; from Domain VI.2"
    )
    sweat_k_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8 mEq/L; sweat [K⁺] upper bound; from Domain VI.2"
    )
    sweat_mg_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.2 mEq/L; sweat [Mg²⁺] lower bound; from Domain VI.2"
    )
    sweat_mg_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.5 mEq/L; sweat [Mg²⁺] upper bound; from Domain VI.2"
    )
    max_na_loss_sweat_meq_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~270 mEq/h; maximum Na⁺ loss rate via sweat (90 mEq/L × 3 L/h); from Domain VI.2"
    )
    sweat_osmolality_mosmol_kg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 mOsm/kg; sweat osmolality lower bound (always hypotonic vs plasma); from Domain VI.2"
    )
    sweat_osmolality_mosmol_kg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="180 mOsm/kg; sweat osmolality upper bound (still hypotonic vs plasma ~290); from Domain VI.2"
    )

    # ── Heat Acclimatisation Effects on Sweat ─────────────────────────────────
    # 7-14 days → increased sweat rate ceiling + aldosterone-mediated Na⁺ conservation
    heat_acclimatisation_sweat_rate_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; sweat rate increase fraction after 7-14 days heat acclimatisation (~50% above baseline); from Domain VI.2"
    )
    heat_acclimatisation_sweat_na_reduction_meq_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 mEq/L; reduction in sweat [Na⁺] after acclimatisation (e.g. 60 → 40 mEq/L); from Domain VI.2"
    )
    heat_acclimatisation_period_days_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="7 days; minimum heat acclimatisation duration for significant adaptation; from Domain VI.2"
    )
    heat_acclimatisation_period_days_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="14 days; optimal heat acclimatisation duration for maximal adaptation; from Domain VI.2"
    )

    # ── Dehydration Thresholds ────────────────────────────────────────────────
    # Progressive thresholds expressed as % body mass loss
    dehydration_performance_impairment_bm_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.02; 2% body mass loss → onset of aerobic performance impairment and plasma volume reduction; from Domain VI.5 / Matrix 9.5"
    )
    dehydration_thermoregulation_impairment_bm_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.03–0.04; 3-4% body mass loss → significant thermoregulatory strain (↑ core temp); from Domain VI.5"
    )
    dehydration_cardiovascular_strain_bm_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.04–0.05; 4-5% body mass loss → cardiovascular drift (↑ HR, ↓ SV, ↓ Q); from Domain VI.5"
    )
    dehydration_collapse_risk_bm_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.06–0.08; 6-8% body mass loss → exertional collapse / heat stroke risk; from Domain VI.5"
    )
    dehydration_lethal_threshold_bm_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.15–0.20; 15-20% body mass loss → lethal dehydration without intervention; from Domain VI.5"
    )

    # ── Hyponatremia Limits ───────────────────────────────────────────────────
    # Dilutional: excess hypotonic intake during prolonged exercise > sweat Na⁺ loss
    hyponatremia_threshold_na_meq_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="135 mEq/L; plasma [Na⁺] below which hyponatremia is defined; from Domain VI.5"
    )
    hyponatremia_symptomatic_na_meq_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="130 mEq/L; plasma [Na⁺] below which symptomatic hyponatremia occurs (nausea/headache); from Domain VI.5"
    )
    hyponatremia_severe_na_meq_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="125 mEq/L; plasma [Na⁺] below which severe exercise-associated hyponatremia (EAH) → cerebral edema; from Domain VI.5"
    )
    hyponatremia_lethal_na_meq_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~115 mEq/L; plasma [Na⁺] below which encephalopathy and herniation risk without treatment; from Domain VI.5"
    )

    # ── Starling Forces — Microvascular Fluid Exchange ────────────────────────
    # J_v = K_f × [(Pc − Pi) − σ × (πc − πi)]; K_f = filtration coefficient
    capillary_hydrostatic_pressure_arterial_end_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~35 mmHg; capillary hydrostatic pressure (Pc) at arterial end; net filtration outward; from Domain VI.3"
    )
    capillary_hydrostatic_pressure_venous_end_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~15 mmHg; capillary hydrostatic pressure (Pc) at venous end; net reabsorption; from Domain VI.3"
    )
    interstitial_hydrostatic_pressure_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="-3 mmHg; interstitial hydrostatic pressure (Pi) lower bound (slightly sub-atmospheric); from Domain VI.3"
    )
    interstitial_hydrostatic_pressure_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2 mmHg; interstitial hydrostatic pressure (Pi) upper bound; from Domain VI.3"
    )
    capillary_oncotic_pressure_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 mmHg; capillary oncotic pressure (πc) lower bound (albumin 3.5 g/dL); from Domain VI.3"
    )
    capillary_oncotic_pressure_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="28 mmHg; capillary oncotic pressure (πc) upper bound (albumin 5.0 g/dL); from Domain VI.3"
    )
    interstitial_oncotic_pressure_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 mmHg; interstitial oncotic pressure (πi) lower bound; from Domain VI.3"
    )
    interstitial_oncotic_pressure_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 mmHg; interstitial oncotic pressure (πi) upper bound; from Domain VI.3"
    )
    starling_reflection_coefficient_sigma: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.90–0.95; Starling reflection coefficient (σ) for albumin across capillary wall; from Domain VI.3"
    )
    net_filtration_pressure_arterial_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10 mmHg; net outward filtration pressure at arterial end (Pc − Pi − πc + πi); from Domain VI.3"
    )

    # ── RAAS Regulation — Renin-Angiotensin-Aldosterone System ───────────────
    # Renin: JGA cells respond to ↓ MAP, ↓ Na⁺ delivery, β₁ stimulation
    # AT-II: vasoconstriction + aldosterone secretion + AVP release + thirst
    aldosterone_na_reabsorption_fold_max: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0; maximum fold-increase in renal Na⁺ reabsorption under maximal aldosterone; from Domain VI.4"
    )
    aldosterone_sweat_na_reduction_meq_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 mEq/L; reduction in sweat gland [Na⁺] secretion under maximal aldosterone (acclimatisation); from Domain VI.4"
    )
    angiotensin_ii_pressor_dose_ng_kg_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–3 ng/kg/min; AT-II infusion dose causing significant blood pressure rise (vasoconstrictor potency); from Domain VI.4"
    )
    anp_release_threshold_atrial_stretch_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5–10 mmHg; atrial wall tension increase above which ANP secretion rises (opposes RAAS); from Domain VI.4"
    )
    raas_suppression_na_intake_meq_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~200 mEq/day; dietary Na⁺ intake above which RAAS is suppressed (low renin state); from Domain VI.4"
    )

    # ── Osmoreceptor and AVP Kinetics ─────────────────────────────────────────
    avp_half_life_plasma_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~15–20 min; AVP (vasopressin) plasma half-life; from Domain VI.4"
    )
    avp_max_plasma_concentration_pg_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20–30 pg/mL; maximum plasma AVP concentration during maximal osmotic or volume stimulus; from Domain VI.4"
    )
    osmoreceptor_sensitivity_mosmol_kg_per_pg_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5 mOsm/kg per pg/mL; osmoreceptor gain (slope of AVP response to osmolality); from Domain VI.4"
    )

    # ── Electrolyte Losses at Exercise ────────────────────────────────────────
    # Daily exercise Na⁺ replacement requirement at maximum sweat rate
    daily_na_loss_max_sweat_meq_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~500 mEq/day; Na⁺ loss via sweat lower bound at high sweat rate; from Domain VI.2"
    )
    daily_na_loss_max_sweat_meq_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1000 mEq/day; Na⁺ loss via sweat upper bound (3.5 L/h × 8 h × 35 mEq/L); from Domain VI.2"
    )
    daily_k_loss_sweat_meq_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 mEq/day; K⁺ loss via sweat lower bound; from Domain VI.2"
    )
    daily_k_loss_sweat_meq_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~60 mEq/day; K⁺ loss via sweat upper bound; from Domain VI.2"
    )
