from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesThermoregulation(Base):
    """
    Phase 1 — Thermoregulation ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 15 / Matrices 15.1–15.5 (core temperature limits, HSPs, enzyme denaturation,
      heat dissipation, cutaneous circulation);
      Domain XII.1 (core temperature: normal range, hyperthermic and hypothermic thresholds);
      Domain XII.2 (heat shock response: HSF1 activation, HSP70/HSP90/HSP27 induction);
      Domain XII.3 (enzymatic thermal denaturation: Tm values, pH × temperature interaction);
      Domain XII.4 (heat dissipation: evaporation, radiation, convection, cutaneous blood flow);
      Domain XII.5 (heat acclimation: sweat onset threshold, plasma volume, HSP upregulation).

    Key equations encoded:
      Q_evap = m_sweat × L_v  (L_v = 2.43 kJ/mL at 37°C)        [evaporative heat loss]
      Q_rad  = ε × σ × A_skin × (T_skin⁴ − T_env⁴)              [radiation; ε_skin ≈ 0.97]
      Q_conv = h_c × A_skin × (T_skin − T_air)                    [convection; h_c forced ≈ 30 W/m²°C]
      Q_metabolic = VO2 × ΔH_O2 × (1 − η_mechanical)            [metabolic heat; η ≈ 0.25]

    Units:
      _deg_c       = °C
      _w           = watts
      _w_m2        = W / m² (heat flux density)
      _w_m2_deg_c  = W / m² / °C (heat transfer coefficient)
      _kj_ml       = kJ / mL
      _ml_per_min  = mL / min
      _l_per_h     = L / h
      _mg_cm2_min  = mg / cm² / min (local sweat rate)
      _h           = hours
      _min         = minutes
      _fraction    = dimensionless 0–1
      _fold        = fold-change
    """

    __tablename__ = "species_thermoregulation"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="thermoregulation")

    # ── Core Body Temperature — Normal Range and Set Point ────────────────────
    # Hypothalamic set point: ~37°C; regulated by warm- and cold-sensitive neurones in preoptic area
    # Circadian variation: ±0.5°C (see chronobiology.py CBT columns)
    core_temp_normal_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="36.5°C; core body temperature lower bound of normal (oral/rectal); from Domain XII.1 / Matrix 15.1"
    )
    core_temp_normal_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="37.5°C; core body temperature upper bound of normal; from Domain XII.1"
    )
    hypothalamic_set_point_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~37.0°C; hypothalamic thermoregulatory set point (preoptic area warm-sensitive neurones); from Domain XII.1"
    )

    # ── Hyperthermic Critical Thresholds ─────────────────────────────────────
    # Exercise-induced hyperthermia: elite athletes briefly sustain 41-42°C at VO2max
    # Exertional heat stroke (EHS): Tc >40°C + CNS dysfunction (confusion, ataxia, collapse)
    # Enzyme denaturation accelerates above 42°C (see HSP and enzyme sections)
    core_temp_heat_exhaustion_onset_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~38.5–39.0°C; core temperature at heat exhaustion onset (cardiovascular strain, weakness); from Domain XII.1"
    )
    core_temp_heat_stroke_threshold_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40.0°C; core temperature above which exertional heat stroke (EHS) is defined (+ CNS dysfunction); from Domain XII.1"
    )
    core_temp_max_tolerated_exercise_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="41.0°C; maximum core temperature transiently tolerated by elite endurance athletes at VO2max lower bound; from Domain XII.1"
    )
    core_temp_max_tolerated_exercise_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="42.0°C; maximum core temperature transiently tolerated by elite athletes upper bound (Gonzalez-Alonso et al. 1999); from Domain XII.1"
    )
    core_temp_enzyme_denaturation_onset_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~42.0°C; core temperature above which significant metabolic enzyme denaturation begins (PFK-1 threshold); from Domain XII.1"
    )
    core_temp_cns_failure_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~42.0–44.0°C; core temperature at which irreversible CNS failure and death risk without intervention; from Domain XII.1"
    )
    core_temp_absolute_lethal_ceiling_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~44.0°C; absolute lethal core temperature ceiling (protein denaturation widespread; multi-organ failure); from Domain XII.1"
    )

    # ── Hypothermic Critical Thresholds ───────────────────────────────────────
    # Mild: 32-35°C; Moderate: 28-32°C (cardiac arrhythmia risk); Severe: <28°C (VF)
    # Lowest survived: 13.7°C (Anna Bågenholm, 1999; accidental hypothermia + ECMO)
    core_temp_mild_hypothermia_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="32.0°C; mild hypothermia lower bound (shivering ceases; metabolic depression); from Domain XII.1"
    )
    core_temp_mild_hypothermia_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35.0°C; mild hypothermia upper bound (onset of impaired cognition, coordination); from Domain XII.1"
    )
    core_temp_moderate_hypothermia_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="28.0°C; moderate hypothermia lower bound; from Domain XII.1"
    )
    core_temp_moderate_hypothermia_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="32.0°C; moderate hypothermia upper bound (cardiac arrhythmia risk); from Domain XII.1"
    )
    core_temp_severe_hypothermia_threshold_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="<28.0°C; severe hypothermia threshold (ventricular fibrillation risk); from Domain XII.1"
    )
    core_temp_lowest_survived_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="13.7°C; lowest core temperature survived in humans (Bågenholm 1999; accidental hypothermia with ECMO); from Domain XII.1"
    )
    shivering_onset_threshold_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~35.5–36.0°C; core temperature at which involuntary shivering thermogenesis is activated; from Domain XII.1"
    )
    shivering_max_heat_production_w: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~300–600 W; maximum heat production by shivering (up to 5× resting metabolic rate); from Domain XII.1"
    )

    # ── Heat Shock Proteins (HSPs) — Induction Thresholds and Kinetics ────────
    # HSP70 (HSPA1A): major inducible chaperone; prevents aggregation and refolds denatured proteins
    # HSF1 trimerisation model: heat displaces HSP70/HSP90 from HSF1 → trimer binds HSE → transcription
    # Heat acclimation increases basal HSP70 content → earlier protection at lower temperatures
    hsf1_activation_temperature_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~39.0–40.0°C; HSF1 (heat shock factor 1) trimerisation and nuclear translocation threshold; from Domain XII.2 / Matrix 15.2"
    )
    hsf1_strong_activation_temperature_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~41.0–42.0°C; core temperature for maximal HSF1 activation and full HSP gene induction; from Domain XII.2"
    )
    hsp70_mrna_induction_fold_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0; HSP70 mRNA induction fold lower bound at 41°C (vs unstressed baseline); from Domain XII.2"
    )
    hsp70_mrna_induction_fold_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50.0; HSP70 mRNA induction fold upper bound at 42-43°C; from Domain XII.2"
    )
    hsp70_mrna_peak_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–2 h; time to peak HSP70 mRNA after heat stress onset; from Domain XII.2"
    )
    hsp70_protein_peak_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~4–6 h; time to peak HSP70 protein accumulation after heat stress; from Domain XII.2"
    )
    hsp70_baseline_fraction_total_protein: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.001–0.005; HSP70 as fraction of total cellular protein under unstressed conditions; from Domain XII.2"
    )
    hsp70_max_induced_fraction_total_protein: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.02–0.05; HSP70 fraction of total cellular protein at maximum induction after severe heat stress; from Domain XII.2"
    )
    hsp90_baseline_fraction_total_protein: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.01–0.02; HSP90 fraction of total cellular protein at baseline (constitutively expressed chaperone); from Domain XII.2"
    )
    hsp90_induction_fold_heat_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.5; HSP90 induction fold at 41°C lower bound (modest; more constitutive than HSP70); from Domain XII.2"
    )
    hsp90_induction_fold_heat_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0; HSP90 induction fold upper bound; from Domain XII.2"
    )
    hsp27_activation_temperature_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~39.0–41.0°C; HSP27 (HSPB1) induction threshold; cytoprotective; inhibits apoptosis; from Domain XII.2"
    )
    hsp27_induction_fold_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0; HSP27 induction fold at moderate heat stress lower bound; from Domain XII.2"
    )
    hsp27_induction_fold_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20.0; HSP27 induction fold at severe heat stress upper bound; from Domain XII.2"
    )
    heat_acclimation_hsp70_baseline_increase_fold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1.5–2.5; fold increase in basal HSP70 content after 10-14 days heat acclimation; from Domain XII.2"
    )
    hsp_induction_exercise_temperature_threshold_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~38.5–39.5°C; core temperature during exercise above which significant HSP70 mRNA induction occurs in working muscle; from Domain XII.2"
    )

    # ── Enzymatic Thermal Denaturation ────────────────────────────────────────
    # Denaturation rate increases exponentially with temperature (Arrhenius)
    # pH-temperature interaction: low pH + elevated temperature = accelerated denaturation
    pfk1_thermal_denaturation_onset_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~42.0°C; PFK-1 (phosphofructokinase-1) thermal denaturation onset temperature; from Domain XII.3 / Matrix 15.3"
    )
    pfk1_tm_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~44.0°C; PFK-1 thermal melting temperature (Tm; 50% activity loss); from Domain XII.3"
    )
    pfk1_ph_temperature_synergy_threshold_ph: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6.2; intramuscular pH below which PFK-1 denaturation accelerates dramatically at temperatures >40°C (pH × T double-hit); from Domain XII.3"
    )
    creatine_kinase_tm_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="42.0°C; creatine kinase (CK) thermal denaturation onset lower bound; from Domain XII.3"
    )
    creatine_kinase_tm_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="44.0°C; CK Tm upper bound (similar to PFK-1; key indicator of heat injury); from Domain XII.3"
    )
    ldh_tm_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~55.0°C; lactate dehydrogenase (LDH) thermal melting temperature (more thermostable than PFK-1); from Domain XII.3"
    )
    myosin_atpase_tm_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~48–50°C; myosin ATPase Tm (safe range for contractile machinery at physiological temperatures); from Domain XII.3"
    )
    membrane_enzyme_denaturation_onset_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40.0–41.0°C; membrane-bound enzyme denaturation onset (more heat-sensitive than soluble enzymes due to lipid fluidity changes); from Domain XII.3"
    )
    protein_denaturation_arrhenius_q10: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2.0–3.0; Q10 for protein denaturation rate (rate doubles/triples per 10°C increase above threshold); from Domain XII.3"
    )

    # ── Evaporative Heat Dissipation ──────────────────────────────────────────
    # Primary heat loss mechanism during exercise; dependent on skin-to-air vapour pressure gradient
    # L_v = 2.43 kJ/mL at 37°C (heat of vaporisation of water)
    latent_heat_vaporisation_kj_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.43 kJ/mL; heat of vaporisation of water at 37°C skin temperature (580 cal/g); from Domain XII.4 / Matrix 15.4"
    )
    max_evaporative_capacity_w_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1200 W; maximum evaporative cooling capacity lower bound (3 L/h × 100% evaporation × 2.43 kJ/mL × 1h); from Domain XII.4"
    )
    max_evaporative_capacity_w_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2400 W; maximum evaporative cooling capacity upper bound (3.5 L/h × 100% evaporation); from Domain XII.4"
    )
    sweat_evaporation_efficiency_dry_air_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.90–1.00; fraction of sweat that evaporates in dry air (RH <30%); from Domain XII.4"
    )
    sweat_evaporation_efficiency_humid_air_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30–0.50; fraction of sweat that evaporates in humid air (RH >80%); from Domain XII.4"
    )
    local_sweat_rate_forehead_mg_cm2_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 mg/cm²/min; local sweat rate on forehead lower bound; from Domain XII.4"
    )
    local_sweat_rate_forehead_mg_cm2_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0 mg/cm²/min; local sweat rate on forehead upper bound (highest regional density); from Domain XII.4"
    )
    local_sweat_rate_forearm_mg_cm2_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 mg/cm²/min; local sweat rate on forearm lower bound; from Domain XII.4"
    )
    local_sweat_rate_forearm_mg_cm2_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 mg/cm²/min; local sweat rate on forearm upper bound; from Domain XII.4"
    )
    eccrine_gland_output_nl_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2 nL/min/gland; eccrine gland individual secretory rate lower bound; from Domain XII.4"
    )
    eccrine_gland_output_nl_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 nL/min/gland; eccrine gland individual secretory rate upper bound (maximally stimulated); from Domain XII.4"
    )
    sweating_onset_threshold_deg_c_untrained: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~37.5–38.0°C; core temperature at which sweating begins (untrained; higher threshold); from Domain XII.4"
    )
    sweating_onset_threshold_deg_c_heat_acclimatised: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~37.0–37.5°C; sweating onset threshold after heat acclimation (~0.5°C lower; earlier activation); from Domain XII.4"
    )

    # ── Radiative Heat Exchange ────────────────────────────────────────────────
    # Stefan-Boltzmann: Q_rad = ε × σ × A_skin × (T_skin⁴ − T_env⁴)
    # Skin emissivity ≈ 0.97 (near perfect blackbody for infrared)
    skin_emissivity: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.97; human skin infrared emissivity (nearly perfect blackbody; independent of melanin content); from Domain XII.4"
    )
    stefan_boltzmann_constant_w_m2_k4: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.67e-8 W/m²/K⁴; Stefan-Boltzmann constant (σ); from Domain XII.4"
    )
    body_surface_area_m2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.5 m²; body surface area lower bound (Dubois formula; 50 kg, 160 cm); from Domain XII.4"
    )
    body_surface_area_m2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.2 m²; body surface area upper bound (90 kg, 185 cm); from Domain XII.4"
    )
    radiation_heat_loss_neutral_env_w_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~50 W; radiative heat loss at T_skin 35°C vs T_env 25°C lower bound; from Domain XII.4"
    )
    radiation_heat_loss_neutral_env_w_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~80 W; radiative heat loss at neutral environment upper bound; from Domain XII.4"
    )

    # ── Convective Heat Transfer ───────────────────────────────────────────────
    # Q_conv = h_c × A_skin × (T_skin − T_air)
    # h_c forced (running ~3 m/s): ~30 W/m²°C — dramatically enhances cooling
    convective_coefficient_natural_w_m2_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~3–5 W/m²°C; natural convection heat transfer coefficient (still air); from Domain XII.4"
    )
    convective_coefficient_forced_running_w_m2_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~25–35 W/m²°C; forced convection coefficient at running velocity ~3 m/s; from Domain XII.4"
    )
    convective_heat_loss_running_hot_env_w_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~100 W; convective heat loss running in hot environment (T_skin 35°C, T_air 30°C, h_c 30) lower bound; from Domain XII.4"
    )
    convective_heat_loss_running_hot_env_w_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~300 W; convective heat loss upper bound (larger BSA + faster running); from Domain XII.4"
    )

    # ── Cutaneous Blood Flow — Thermoregulatory Vasodilation ─────────────────
    # Active cutaneous vasodilation: cholinergic + VIP (vasoactive intestinal peptide) + NO-mediated
    # Competes with skeletal muscle for Q at VO2max in heat (cardiovascular strain)
    cutaneous_blood_flow_resting_ml_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 mL/min; resting cutaneous blood flow lower bound (~5% of cardiac output); from Domain XII.4"
    )
    cutaneous_blood_flow_resting_ml_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="400 mL/min; resting cutaneous blood flow upper bound; from Domain XII.4"
    )
    cutaneous_blood_flow_max_heat_stress_l_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0 L/min; maximum cutaneous blood flow during severe heat stress lower bound; from Domain XII.4"
    )
    cutaneous_blood_flow_max_heat_stress_l_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8.0 L/min; maximum cutaneous blood flow upper bound (up to 60-70% of resting cardiac output); from Domain XII.4"
    )
    cutaneous_blood_flow_fraction_q_max_heat: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.60–0.70; cutaneous blood flow as fraction of cardiac output during maximal heat stress (resting Q ~6 L/min); from Domain XII.4"
    )

    # ── Metabolic Heat Production ──────────────────────────────────────────────
    # At VO2max: ~75% of metabolic rate → heat (mechanical efficiency ~25%)
    # Running: ~1 kcal/kg/km generated as heat (speed-dependent)
    metabolic_heat_production_vo2max_w_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~750 W; heat produced at VO2max lower bound (0.75 × metabolic rate of ~1000 W); from Domain XII.4"
    )
    metabolic_heat_production_vo2max_w_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1125 W; heat produced at VO2max upper bound (0.75 × 1500 W metabolic rate); from Domain XII.4"
    )
    mechanical_efficiency_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.25; gross mechanical efficiency of skeletal muscle (25% → work; 75% → heat); from Domain XII.4"
    )
    resting_metabolic_rate_w_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 W; basal resting metabolic rate lower bound (basal heat production); from Domain XII.4"
    )
    resting_metabolic_rate_w_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 W; basal resting metabolic rate upper bound; from Domain XII.4"
    )
    running_heat_generation_kcal_kg_km: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1 kcal/kg/km; heat generated during running (~1 kcal per kg body mass per km; speed-independent for aerobic running); from Domain XII.4"
    )

    # ── Heat Acclimation Adaptations ─────────────────────────────────────────
    # 10-14 days protocol; key adaptations: ↑ plasma volume, ↓ sweat onset threshold,
    # ↑ sweat rate, ↓ sweat [Na⁺], ↑ HSP70 content, ↓ resting core temperature
    heat_acclimation_duration_days_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 days; minimum heat acclimation duration for significant physiological adaptation; from Domain XII.5 / Matrix 15.5"
    )
    heat_acclimation_duration_days_optimal: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="14 days; optimal heat acclimation duration for maximal adaptation; from Domain XII.5"
    )
    heat_acclimation_plasma_volume_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10–0.15; plasma volume expansion fraction after heat acclimation (10-15%); from Domain XII.5"
    )
    heat_acclimation_sweat_rate_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; sweat rate ceiling increase fraction after heat acclimation (50% above baseline); from Domain XII.5"
    )
    heat_acclimation_sweat_na_reduction_meq_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 mEq/L; reduction in sweat [Na⁺] after acclimation (aldosterone-mediated); from Domain XII.5"
    )
    heat_acclimation_core_temp_reduction_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.3–0.5°C; reduction in resting and exercise core temperature after heat acclimation; from Domain XII.5"
    )
    heat_acclimation_decay_days_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="14 days; heat acclimation benefit decay lower bound (half-adaptation lost); from Domain XII.5"
    )
    heat_acclimation_decay_days_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="28 days; heat acclimation benefit decay upper bound (full washout); from Domain XII.5"
    )

    # ── Non-Shivering Thermogenesis (BAT / UCP1) ─────────────────────────────
    # Brown adipose tissue (BAT): activated by sympathetic NE → β₃-AR → UCP1 uncoupling
    # Adult human BAT: cervical, supraclavicular, paravertebral, perirenal depots
    ucp1_activation_temperature_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~35.0–36.0°C; core temperature below which UCP1-mediated NST is significantly activated; from Domain XII.1"
    )
    bat_nst_max_heat_production_w_adult_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 W; maximum non-shivering thermogenesis heat production (adult BAT) lower bound; from Domain XII.1"
    )
    bat_nst_max_heat_production_w_adult_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 W; maximum NST heat production adult upper bound (cold-acclimatised individuals); from Domain XII.1"
    )
    cold_acclimatisation_bat_activity_increase_fold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0–5.0; BAT metabolic activity fold-increase after 4-6 weeks cold acclimation (FDG-PET measured); from Domain XII.1"
    )
    norepinephrine_bat_beta3ar_ec50_nmol_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10 nM; NE EC50 for β3-adrenergic receptor-mediated UCP1 activation in BAT (sympathetic thermogenic drive); from Domain XII.1"
    )
    bat_glucose_uptake_fold_cold_vs_rest: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10; fold-increase in BAT 18F-FDG glucose uptake during cold exposure vs thermoneutral rest (PET imaging); from Domain XII.1"
    )
    bat_thermogenesis_fatty_acid_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.85; fraction of BAT thermogenic substrate from fatty acid oxidation (remainder glucose); from Domain XII.1"
    )
    bat_ucp1_protein_content_cold_fold_increase: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0–4.0; fold increase in UCP1 protein content per gram BAT after prolonged cold acclimation; from Domain XII.1"
    )
    sympathetic_drive_bat_activation_threshold_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~35.5°C; core temperature at which skin-cooling reflex maximally recruits sympathetic NE drive to BAT; from Domain XII.1"
    )

    # ── Whole-Body Heat Balance Summary Constants ─────────────────────────────
    # Used by MPC to compute real-time heat storage rate: dQ/dt = Q_metabolic − Q_evap − Q_rad − Q_conv
    body_heat_capacity_kj_per_kg_per_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~3.47 kJ/kg/°C; specific heat capacity of the human body (weighted mean of tissue + water); from Domain XII.4"
    )
    core_to_shell_thermal_conductance_w_per_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5 W/°C; core-to-shell thermal conductance at rest (vasoconstricted shell); from Domain XII.4"
    )
    core_to_shell_thermal_conductance_w_per_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~75 W/°C; core-to-shell thermal conductance at maximum cutaneous vasodilation; from Domain XII.4"
    )
    wet_bulb_globe_temperature_heat_stroke_threshold_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~28°C WBGT; wet-bulb globe temperature threshold above which exertional heat stroke risk rises sharply in unacclimatised athletes; from Domain XII.5"
    )
    heat_index_extreme_danger_threshold_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~54°C apparent temperature; NOAA 'extreme danger' heat index (heat stroke imminent without cooling); from Domain XII.5"
    )
