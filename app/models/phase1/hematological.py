from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesHematological(Base):
    __tablename__ = "species_hematological"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="hematological")

    # ── Total Blood Volume ────────────────────────────────────────────────────
    blood_volume_male_ml_per_kg_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 mL/kg — lower total blood volume in adult males (Nadler formula reference)"
    )
    blood_volume_male_ml_per_kg_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="80 mL/kg — upper total blood volume in adult males"
    )
    blood_volume_female_ml_per_kg_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="60 mL/kg — lower total blood volume in adult females (lower RBC mass)"
    )
    blood_volume_female_ml_per_kg_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 mL/kg — upper total blood volume in adult females"
    )
    blood_volume_elite_endurance_ml_per_kg_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="90 mL/kg — lower total blood volume in elite endurance athletes (training-expanded)"
    )
    blood_volume_elite_endurance_ml_per_kg_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="110 mL/kg — upper blood volume in elite athletes; positively correlates with VO₂max"
    )

    # ── Plasma Volume ─────────────────────────────────────────────────────────
    plasma_volume_normal_ml_per_kg_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="40 mL/kg — lower normal plasma volume fraction (~55% of blood volume)"
    )
    plasma_volume_normal_ml_per_kg_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="45 mL/kg — upper normal plasma volume"
    )
    plasma_volume_trained_ml_per_kg_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 mL/kg — lower plasma volume in endurance-trained athletes"
    )
    plasma_volume_trained_ml_per_kg_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="55 mL/kg — upper plasma volume in elite endurance athletes"
    )
    plasma_volume_training_expansion_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 mL — lower plasma volume expansion with 8–12 weeks endurance training"
    )
    plasma_volume_training_expansion_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 mL — upper plasma volume expansion with training (aldosterone + albumin synthesis)"
    )
    plasma_volume_acute_dehydration_2pct_bw_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.04 — ~4% plasma volume reduction per 1% body mass lost by dehydration"
    )

    # ── Erythrocyte Mass and Counts ───────────────────────────────────────────
    rbc_mass_male_ml_per_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="27.5 mL/kg — midpoint 25–30 mL/kg; total RBC mass in males (⁵¹Cr dilution)"
    )
    rbc_mass_female_ml_per_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="22.5 mL/kg — midpoint 20–25 mL/kg; total RBC mass in females"
    )
    rbc_count_male_per_ul_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.5e6 per µL — lower RBC count in adult males (4.5–5.9 × 10⁶/µL)"
    )
    rbc_count_male_per_ul_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.9e6 per µL — upper normal RBC count in adult males"
    )
    rbc_count_female_per_ul_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.0e6 per µL — lower RBC count in adult females"
    )
    rbc_count_female_per_ul_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.2e6 per µL — upper normal RBC count in adult females"
    )
    rbc_lifespan_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="120 days — normal erythrocyte lifespan (⁵¹Cr half-life ~28 days; destroyed by spleen/liver)"
    )
    rbc_production_per_second: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5e6 per second — ~2.5 million RBCs produced per second to replace ~200 billion/day"
    )
    reticulocyte_fraction_rbc: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.015 — midpoint 0.5–2.5%; reticulocyte fraction of circulating RBCs (maturation index)"
    )
    rbc_diameter_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="7 µm — midpoint 6–8 µm; normal RBC diameter (biconcave disc)"
    )
    rbc_volume_mcv_fl: Mapped[Optional[float]] = mapped_column(
        Float, comment="90 fL — midpoint 80–100 fL; mean corpuscular volume (MCV); macrocytosis >100, microcytosis <80"
    )
    rbc_thickness_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 µm — midpoint 1.5–2.5 µm; RBC thickness at rim; central pallor 1.0 µm"
    )
    rbc_surface_area_um2: Mapped[Optional[float]] = mapped_column(
        Float, comment="140 µm² — normal RBC surface area; 40% excess vs sphere of same volume → deformability"
    )
    rbc_minimum_capillary_diameter_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="3 µm — minimum capillary diameter RBC can deform through (spectrin cytoskeleton)"
    )
    rbc_mcch_g_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="32 g/dL — lower MCHC (mean corpuscular Hb concentration)"
    )
    rbc_mchc_g_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="36 g/dL — upper MCHC; >36 = spherocytosis / dehydration artifact"
    )
    rbc_mch_pg_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="27 pg — lower MCH (mean corpuscular Hb); <27 = microcytic hypochromic"
    )
    rbc_mch_pg_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="33 pg — upper MCH"
    )

    # ── Hemoglobin — Concentration and Hüfner Constant ───────────────────────
    hemoglobin_male_g_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="13.5 g/dL — lower normal hemoglobin in adult males (WHO anaemia threshold)"
    )
    hemoglobin_male_g_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="17.5 g/dL — upper normal hemoglobin in adult males"
    )
    hemoglobin_female_g_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="12.0 g/dL — lower normal hemoglobin in adult females"
    )
    hemoglobin_female_g_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="16.0 g/dL — upper normal hemoglobin in adult females"
    )
    hemoglobin_elite_endurance_male_g_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, comment="17.0 g/dL — midpoint 16–18 g/dL; elite male endurance athlete Hb (high-altitude adapted)"
    )
    hemoglobin_anaemia_threshold_male_g_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, comment="13.0 g/dL — WHO anaemia definition in males; impairs VO₂max and endurance"
    )
    hemoglobin_anaemia_threshold_female_g_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, comment="12.0 g/dL — WHO anaemia definition in females"
    )
    hufner_constant_ml_o2_per_g_hb: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.34 mL O₂/g Hb — Hüfner constant (empirical); maximum O₂ carried per gram of Hb when fully saturated"
    )
    hufner_constant_theoretical_ml_o2_per_g_hb: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.39 mL O₂/g Hb — theoretical Hüfner constant (64,450 g/mol × 4 O₂/Hb × 22,400 mL/mol O₂)"
    )
    cao2_max_ml_per_dl_blood: Mapped[Optional[float]] = mapped_column(
        Float, comment="20.5 mL O₂/dL — midpoint 20–21 mL O₂/dL; maximal arterial O₂ content (Hb 15 g/dL × 1.34 × 0.99 + dissolved)"
    )
    o2_dissolved_plasma_ml_per_dl_per_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.003 mL O₂/dL/mmHg — O₂ physical solubility in plasma at 37°C (Henry's law)"
    )
    cao2_dissolved_fraction_at_normal_pao2: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.015 — ~1.5% of total CaO2 is dissolved O₂ at normal PaO2 95 mmHg (0.003 × 95 / 20)"
    )

    # ── Hematocrit — Normal Range and Critical Thresholds ────────────────────
    hematocrit_male_normal_lower_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.42 — lower normal hematocrit in adult males (42%)"
    )
    hematocrit_male_normal_upper_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.52 — upper normal hematocrit in adult males (52%)"
    )
    hematocrit_female_normal_lower_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.37 — lower normal hematocrit in adult females (37%)"
    )
    hematocrit_female_normal_upper_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.47 — upper normal hematocrit in adult females (47%)"
    )
    hematocrit_elite_endurance_lower_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.45 — lower Hct in elite endurance athletes (altitude-adapted; high RBC mass)"
    )
    hematocrit_elite_endurance_upper_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.52 — upper Hct in elite endurance athletes (high end; legal natural range)"
    )
    hematocrit_hyperviscosity_threshold_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.515 — midpoint 50–53%; Hct above which hyperviscosity + thrombosis risk rises sharply"
    )
    hematocrit_thrombosis_risk_critical_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.55 — Hct ≥55%: exponential thrombosis risk; stroke/PE risk zone; polycythemia vera"
    )
    hematocrit_anti_doping_uci_limit_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.50 — UCI biological passport Hct limit for male cyclists (anti-doping bright line)"
    )
    hematocrit_optimal_o2_delivery_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.44 — midpoint 40–48%; optimal Hct for maximal O₂ delivery (DO₂ = CO × CaO2; viscosity tradeoff)"
    )
    blood_viscosity_normal_mpa_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.5 mPa·s — midpoint 3–4 mPa·s; whole blood viscosity at 37°C, Hct 45% (Poiseuille flow)"
    )
    blood_viscosity_hct50_mpa_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.0 mPa·s — approximate blood viscosity at Hct 50%; 40% increase above normal"
    )
    blood_viscosity_hct60_mpa_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="9.0 mPa·s — midpoint 8–10 mPa·s; blood viscosity at Hct 60%; exponential rise"
    )
    plasma_viscosity_mpa_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.2 mPa·s — midpoint 1.1–1.3 mPa·s; plasma viscosity (albumin + fibrinogen dependent)"
    )

    # ── Oxyhemoglobin Dissociation Curve — P50 and Bohr Effect ───────────────
    p50_standard_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="26.5 mmHg — standard P50 (O₂ partial pressure at 50% SaO₂; pH 7.40, 37°C, PCO₂ 40 mmHg)"
    )
    p50_range_lower_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="24 mmHg — lower physiological P50 (alkalosis / hypothermia / low 2,3-DPG; left shift)"
    )
    p50_range_upper_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="31 mmHg — upper physiological P50 (acidosis / hyperthermia / high 2,3-DPG; right shift)"
    )
    p50_exercise_max_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="35 mmHg — estimated P50 in working muscle at VO₂max (pH 6.8 + T 41°C + DPG; maximal Bohr)"
    )
    hill_coefficient_n: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.7 — Hill coefficient n for cooperative O₂ binding by Hb (n=1 = no cooperativity; n=4 = max)"
    )
    bohr_effect_dp50_per_ph_unit: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 mmHg/0.01 pH — P50 shift per 0.01 pH unit decrease (Bohr coefficient; acidosis → right shift)"
    )
    bohr_effect_dp50_per_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.8 mmHg/°C — P50 increase per 1°C temperature rise (thermodynamic effect on Hb-O₂ affinity)"
    )
    bohr_effect_dp50_per_pco2_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.3 mmHg P50/mmHg PCO₂ — direct CO₂ effect on P50 (carbamino + pH-mediated Bohr)"
    )
    sao2_normal_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.97 — midpoint 0.95–0.99; normal arterial O₂ saturation at rest (PaO₂ ~95 mmHg)"
    )
    sao2_eiah_threshold_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.91 — SaO₂ below which exercise-induced arterial hypoxaemia (EIAH) is defined in athletes"
    )
    sao2_altitude_4000m_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.85 — midpoint 0.82–0.88; resting SaO₂ at 4,000 m altitude (PaO₂ ~50–55 mmHg)"
    )
    svo2_mixed_venous_rest_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.75 — normal mixed venous O₂ saturation at rest (PvO₂ ~40 mmHg; right heart/PA catheter)"
    )
    svo2_mixed_venous_max_exercise_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — midpoint 0.15–0.25; mixed venous SvO₂ at VO₂max (maximal O₂ extraction)"
    )
    a_vo2_difference_rest_ml_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.5 mL O₂/dL — midpoint 4–5 mL O₂/dL; arteriovenous O₂ difference at rest"
    )
    a_vo2_difference_max_exercise_ml_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, comment="16.5 mL O₂/dL — midpoint 15–18 mL O₂/dL; a-vO₂ difference at VO₂max (elite athletes)"
    )
    pao2_normal_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="95 mmHg — midpoint 90–100 mmHg; normal arterial PO₂ (alveolar-arterial gradient <15 mmHg)"
    )
    pao2_eiah_threshold_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="80 mmHg — midpoint 75–85 mmHg; PaO₂ at VO₂max in EIAH athletes (diffusion limitation)"
    )

    # ── 2,3-Bisphosphoglycerate (2,3-DPG) ────────────────────────────────────
    dpg_23_normal_mmol_per_l_rbc: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.75 mmol/L RBC — midpoint 4.5–5.0 mmol/L; intracellular 2,3-DPG concentration at rest"
    )
    dpg_23_p50_sensitivity_mmhg_per_mmol_l: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.0 mmHg/0.1 mmol/L — P50 shift per 0.1 mmol/L rise in 2,3-DPG"
    )
    dpg_23_altitude_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — midpoint 20–30%; 2,3-DPG increase after 24–48 h at altitude (compensatory right shift)"
    )
    dpg_23_exercise_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.15 — midpoint 10–20%; 2,3-DPG increase after prolonged exercise (acidosis-induced)"
    )
    dpg_23_storage_depletion_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="24 h — 2,3-DPG falls by ~50% in stored blood within 24 h (transfusion consideration)"
    )
    bpg_mutase_km_13dpg_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM — bisphosphoglycerate mutase Km for 1,3-DPG (converts 1,3-DPG → 2,3-DPG in glycolysis)"
    )

    # ── CO2 Transport ────────────────────────────────────────────────────────
    co2_transport_bicarbonate_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.70 — ~70% of CO₂ transported as HCO₃⁻ (CA-catalysed in RBC; Hamburger shift)"
    )
    co2_transport_dissolved_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.07 — ~7% of CO₂ transported dissolved in plasma (Henry: 0.023 mL/dL/mmHg)"
    )
    co2_transport_carbamino_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.23 — ~23% of CO₂ as carbamino compounds (mainly HbNHCOO⁻; more at venous PCO₂)"
    )
    paco2_normal_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="40 mmHg — normal arterial PCO₂ (35–45 mmHg); primary respiratory acid-base variable"
    )
    pvco2_mixed_venous_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="46 mmHg — midpoint 44–48 mmHg; normal mixed venous PCO₂ at rest"
    )
    pvco2_max_exercise_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="90 mmHg — midpoint 80–100 mmHg; working muscle venous PCO₂ at VO₂max (Haldane effect)"
    )
    rbc_carbonic_anhydrase_ii_kcat_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1e6 per s — CA II kcat in RBC; converts CO₂ + H₂O ↔ H₂CO₃ in microseconds"
    )
    chloride_shift_band3_turnover_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="5e4 per s — Band 3 (AE1/SLC4A1) HCO₃⁻/Cl⁻ exchange rate; Hamburger shift"
    )
    haldane_effect_co2_capacity_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.15 — ~15% increase in blood CO₂ capacity when O₂ is released (deoxygenated Hb binds more CO₂)"
    )

    # ── EPO — Erythropoiesis Regulation ──────────────────────────────────────
    epo_plasma_normal_miu_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 mIU/mL — lower normal plasma EPO in healthy adults (ELISA)"
    )
    epo_plasma_normal_miu_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 mIU/mL — upper normal plasma EPO; rises 10–1000× with severe anaemia"
    )
    epo_half_life_iv_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.5 h — midpoint 5–8 h; IV EPO plasma t½ (biexponential; alpha t½ ~1 h, beta ~6–8 h)"
    )
    epo_half_life_sc_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="27 h — midpoint 24–30 h; subcutaneous EPO t½ (slow absorption; prolonged exposure)"
    )
    epo_altitude_induction_fold_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 — lower EPO fold increase at altitude 2500–3500 m within 24–48 h"
    )
    epo_altitude_induction_fold_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="10.0 — upper EPO fold induction at extreme altitude (>5000 m) or severe hypoxia"
    )
    epo_optimal_altitude_m_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="2500 m — lower altitude for maximal EPO + Hb adaptation response (Live High Train Low)"
    )
    epo_optimal_altitude_m_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="3500 m — upper altitude for optimal EPO stimulus without excessive performance impairment"
    )
    epo_time_to_hb_increase_weeks: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.5 weeks — midpoint 3–4 weeks; time for measurable Hb rise after altitude EPO induction"
    )
    epo_hif1a_stabilisation_threshold_po2_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="40 mmHg — PO₂ below which PHD2 is inhibited → HIF-1α stabilised → EPO gene transcription"
    )
    phd2_km_o2_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="175 µM — midpoint 100–250 µM; PHD2 Km for O₂; operates near atmospheric O₂ (oxygen sensor)"
    )
    epo_receptor_jak2_stat5_ec50_miu_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 mIU/mL — approximate EpoR/JAK2 activation EC50; very sensitive to EPO"
    )

    # ── O2 Delivery (DO2) — Fick Principle ───────────────────────────────────
    do2_resting_l_o2_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.0 L O₂/min — midpoint 0.9–1.1 L O₂/min; resting DO₂ (CO 5 L/min × CaO₂ 20 mL/dL)"
    )
    do2_max_elite_l_o2_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="8.0 L O₂/min — midpoint 6–8 L O₂/min; maximal DO₂ in elite endurance athletes"
    )
    o2_extraction_ratio_rest: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — midpoint 20–30%; resting O₂ extraction ratio (VO₂/DO₂ = 0.25)"
    )
    o2_extraction_ratio_max_exercise: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.80 — midpoint 75–85%; O₂ extraction ratio at VO₂max (DO₂/VO₂ critically matched)"
    )
    critical_do2_threshold_ml_o2_per_min_per_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 mL O₂/min/kg — critical DO₂ below which anaerobic metabolism begins (DO₂crit; ICU)"
    )

    # ── Plasma Proteins ────────────────────────────────────────────────────────
    plasma_albumin_g_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.5 g/dL — lower normal plasma albumin; main oncotic pressure provider (2.5 mmHg/g)"
    )
    plasma_albumin_g_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.0 g/dL — upper normal plasma albumin; synthesised by liver at ~12 g/day"
    )
    plasma_total_protein_g_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.0 g/dL — lower normal total plasma protein"
    )
    plasma_total_protein_g_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="8.0 g/dL — upper normal total plasma protein (albumin + globulins + fibrinogen)"
    )
    plasma_fibrinogen_mg_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 mg/dL — lower normal fibrinogen; acute-phase reactant; clotting substrate"
    )
    plasma_fibrinogen_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="400 mg/dL — upper normal fibrinogen; t½ ~4 days; rises with inflammation"
    )
    plasma_oncotic_pressure_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="26 mmHg — midpoint 25–28 mmHg; plasma colloid oncotic pressure (mainly albumin; Starling force)"
    )

    # ── Platelets ─────────────────────────────────────────────────────────────
    platelet_count_per_ul_lower: Mapped[Optional[int]] = mapped_column(
        Integer, comment="150000 per µL — lower normal platelet count (thrombocytopenia below this)"
    )
    platelet_count_per_ul_upper: Mapped[Optional[int]] = mapped_column(
        Integer, comment="400000 per µL — upper normal platelet count (thrombocytosis above this)"
    )
    platelet_lifespan_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 days — midpoint 8–12 days; platelet lifespan (removed by spleen/liver)"
    )
    platelet_activation_arachidonic_acid_threshold_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 µM — approximate arachidonic acid threshold for TXA₂-mediated platelet aggregation"
    )
    bleeding_time_min_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 min — lower normal bleeding time (Ivy template method)"
    )
    bleeding_time_min_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="7 min — upper normal bleeding time; >10 min = platelet dysfunction"
    )
    pt_normal_s_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 s — lower prothrombin time (PT; extrinsic pathway); INR 0.8–1.2"
    )
    pt_normal_s_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="14 s — upper PT normal range"
    )
    aptt_normal_s_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 s — lower activated partial thromboplastin time (aPTT; intrinsic pathway)"
    )
    aptt_normal_s_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="35 s — upper aPTT normal; >70 s = therapeutic anticoagulation (heparin)"
    )

    # ── Leukocytes — Total Blood Count ────────────────────────────────────────
    wbc_count_per_ul_lower: Mapped[Optional[int]] = mapped_column(
        Integer, comment="4000 per µL — lower normal WBC count (leukopenia below this)"
    )
    wbc_count_per_ul_upper: Mapped[Optional[int]] = mapped_column(
        Integer, comment="11000 per µL — upper normal WBC count (leukocytosis above; infection/exercise)"
    )
    wbc_post_exercise_leukocytosis_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 — midpoint 2–3×; WBC fold increase immediately post-maximal exercise (demargination)"
    )
    neutrophil_fraction_wbc: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — midpoint 50–70%; neutrophil fraction of circulating WBC at rest"
    )
    lymphocyte_fraction_wbc: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.30 — midpoint 20–40%; lymphocyte fraction of WBC at rest"
    )

    # ── Carbon Monoxide — Affinity and Toxicity ───────────────────────────────
    co_hb_relative_affinity_vs_o2: Mapped[Optional[float]] = mapped_column(
        Float, comment="240 — CO-Hb affinity 240× greater than O₂-Hb (Haldane constant; explains CO toxicity)"
    )
    cohb_normal_fraction_nonsmoker: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.005 — <0.5% COHb in non-smokers (endogenous CO from haem catabolism)"
    )
    cohb_smoker_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.07 — midpoint 5–10% COHb in active smokers; impairs CaO₂ and P50"
    )
    cohb_headache_threshold_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.05 — 5% COHb → headache and mild cognitive impairment"
    )
    cohb_lethal_threshold_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.50 — 50% COHb → death (hypoxic hypoxia + impaired mitochondrial function)"
    )
    co_p50_left_shift_per_10pct_cohb_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="6 mmHg — P50 left shift per 10% COHb (remaining Hb subunits increase O₂ affinity; dual toxicity)"
    )
