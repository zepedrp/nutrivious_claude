from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesOsseousSystem(Base):
    __tablename__ = "species_osseous_system"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="osseous_system")

    # ── Peak Bone Mineral Density (BMD) — DXA Reference Values ───────────────
    bmd_lumbar_spine_peak_lower_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.0 g/cm² — lower peak BMD lumbar spine L1–L4 (DXA; healthy young adult 25–35 y)"
    )
    bmd_lumbar_spine_peak_upper_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.4 g/cm² — upper peak BMD lumbar spine; T-score 0 reference ~1.0 g/cm²"
    )
    bmd_femoral_neck_peak_lower_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.9 g/cm² — lower peak BMD femoral neck (DXA; WHO fracture risk site)"
    )
    bmd_femoral_neck_peak_upper_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.2 g/cm² — upper peak BMD femoral neck in healthy young adults"
    )
    bmd_total_hip_peak_lower_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.9 g/cm² — lower peak BMD total hip; clinical fracture risk surrogate"
    )
    bmd_total_hip_peak_upper_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.2 g/cm² — upper peak BMD total hip in young healthy adults"
    )
    bmd_whole_body_peak_lower_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.0 g/cm² — lower whole-body BMD peak (DXA)"
    )
    bmd_whole_body_peak_upper_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.3 g/cm² — upper whole-body BMD peak"
    )
    bmd_peak_age_lower_years: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 years — lower age at peak bone mass achievement"
    )
    bmd_peak_age_upper_years: Mapped[Optional[float]] = mapped_column(
        Float, comment="35 years — upper age at peak bone mass (plateau phase)"
    )
    bmd_peak_fraction_achieved_by_age_20: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.95 — ~95% of peak bone mass is achieved by age 20; adolescence critical window"
    )
    bmd_annual_loss_post_peak_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.0075 — 0.5–1.0%/year BMD loss after peak in adults (both sexes)"
    )
    bmd_annual_loss_postmenopausal_first5y_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.025 — 2–3%/year accelerated BMD loss in first 5 years postmenopause (oestrogen withdrawal)"
    )
    bmd_osteopenia_t_score_threshold: Mapped[Optional[float]] = mapped_column(
        Float, comment="-1.0 — T-score upper cutoff for osteopenia (WHO: -1.0 to -2.5)"
    )
    bmd_osteoporosis_t_score_threshold: Mapped[Optional[float]] = mapped_column(
        Float, comment="-2.5 — T-score threshold for osteoporosis diagnosis (WHO 1994)"
    )
    bmd_exercise_annual_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.02 — midpoint 1–3%/year BMD increase at loaded sites with impact exercise training"
    )

    # ── Bone Composition ──────────────────────────────────────────────────────
    cortical_bone_skeleton_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.80 — ~80% of total skeletal mass is cortical (compact) bone"
    )
    trabecular_bone_skeleton_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — ~20% of skeletal mass is trabecular (cancellous) bone; 80% metabolic activity"
    )
    cortical_bone_density_g_per_cm3: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.9 g/cm³ — midpoint 1.8–2.0 g/cm³; cortical bone true material density"
    )
    trabecular_bone_apparent_density_g_per_cm3_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.1 g/cm³ — lower apparent density of trabecular bone (low BV/TV sites)"
    )
    trabecular_bone_apparent_density_g_per_cm3_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 g/cm³ — upper apparent density of trabecular bone"
    )
    hydroxyapatite_fraction_dry_weight: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.70 — ~70% of bone dry weight is hydroxyapatite Ca₁₀(PO₄)₆(OH)₂"
    )
    collagen_type1_fraction_dry_weight: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.22 — ~22% midpoint 20–25%; Type I collagen dry weight fraction; tensile scaffold"
    )
    bone_water_fraction_wet_weight: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.12 — ~12% midpoint 10–15%; water fraction of wet bone weight"
    )
    skeletal_calcium_total_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.0 kg — total calcium stored in adult human skeleton (99% of body calcium)"
    )
    skeletal_phosphorus_total_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.6 kg — midpoint 0.5–0.7 kg; skeletal phosphorus (85% of body total)"
    )
    bone_mineral_crystal_width_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 nm — midpoint 25–75 nm; width of hydroxyapatite crystals in bone matrix"
    )
    bone_mineral_crystal_thickness_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 nm — midpoint 2–6 nm; thickness of hydroxyapatite platelets"
    )
    mineral_to_collagen_ratio_mass: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 — mineral:collagen mass ratio in mature human cortical bone"
    )

    # ── Cortical Bone Mechanical Properties ───────────────────────────────────
    cortical_elastic_modulus_longitudinal_gpa_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 GPa — lower Young's modulus of cortical bone in longitudinal direction"
    )
    cortical_elastic_modulus_longitudinal_gpa_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 GPa — upper Young's modulus cortical bone longitudinal; anisotropic material"
    )
    cortical_elastic_modulus_transverse_gpa_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 GPa — lower transverse (radial) elastic modulus cortical bone"
    )
    cortical_elastic_modulus_transverse_gpa_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="13 GPa — upper transverse elastic modulus"
    )
    cortical_tensile_strength_longitudinal_mpa_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 MPa — lower tensile strength cortical bone longitudinal axis"
    )
    cortical_tensile_strength_longitudinal_mpa_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="180 MPa — upper tensile strength cortical bone longitudinal"
    )
    cortical_compressive_strength_longitudinal_mpa_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="130 MPa — lower compressive strength cortical bone longitudinal"
    )
    cortical_compressive_strength_longitudinal_mpa_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 MPa — upper compressive strength cortical bone longitudinal"
    )
    cortical_shear_strength_mpa: Mapped[Optional[float]] = mapped_column(
        Float, comment="68 MPa — midpoint 65–71 MPa; shear strength cortical bone"
    )
    cortical_ultimate_strain_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.025 — midpoint 1.5–3%; cortical bone ultimate strain before failure"
    )
    cortical_fracture_toughness_mpa_m05_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 MPa·m⁰·⁵ — lower fracture toughness cortical bone (KIc; crack propagation resistance)"
    )
    cortical_fracture_toughness_mpa_m05_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="7 MPa·m⁰·⁵ — upper fracture toughness cortical bone"
    )
    cortical_fatigue_limit_mpa: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 MPa — cyclic fatigue limit below which cortical bone has effectively infinite life"
    )
    femur_bone_safety_factor_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="6 — lower safety factor of femoral shaft (ratio fracture load:body weight)"
    )
    femur_bone_safety_factor_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 — upper safety factor femur; decreases with osteoporosis and fatigue"
    )

    # ── Trabecular Bone Mechanical Properties ────────────────────────────────
    trabecular_elastic_modulus_lower_gpa: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.1 GPa — lower trabecular bone elastic modulus (low-density cancellous)"
    )
    trabecular_elastic_modulus_upper_gpa: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.0 GPa — upper trabecular elastic modulus (dense vertebral trabeculae)"
    )
    trabecular_compressive_strength_lower_mpa: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 MPa — lower trabecular compressive strength (highly density-dependent)"
    )
    trabecular_compressive_strength_upper_mpa: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 MPa — upper trabecular compressive strength at high BV/TV"
    )
    trabecular_bv_tv_ratio_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.10 — lower bone volume/total volume (BV/TV) in human trabeculae"
    )
    trabecular_bv_tv_ratio_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.40 — upper BV/TV in dense trabecular regions (vertebral endplate)"
    )
    trabecular_thickness_um_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 µm — lower trabecular thickness (Tb.Th) in human bone"
    )
    trabecular_thickness_um_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 µm — upper Tb.Th; thicker with loading and anabolic states"
    )
    trabecular_spacing_um_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 µm — lower inter-trabecular spacing (Tb.Sp)"
    )
    trabecular_spacing_um_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="1500 µm — upper Tb.Sp; increases with osteoporosis"
    )
    trabecular_number_per_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 per mm — midpoint 1–2/mm; trabecular number (Tb.N) in healthy adults"
    )
    trabecular_modulus_density_power_law_exponent: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 — power law exponent: modulus ∝ apparent density^2 (Gibson 1985; Carter-Hayes)"
    )

    # ── Calcium Metabolism — Plasma and Absorption ───────────────────────────
    plasma_calcium_total_mg_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="8.5 mg/dL — lower normal total plasma calcium (2.12 mmol/L)"
    )
    plasma_calcium_total_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="10.5 mg/dL — upper normal total plasma calcium (2.62 mmol/L)"
    )
    plasma_calcium_ionised_mg_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.5 mg/dL — lower ionised (free) plasma calcium (1.12 mmol/L); physiologically active"
    )
    plasma_calcium_ionised_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.5 mg/dL — upper ionised plasma calcium (1.37 mmol/L)"
    )
    plasma_calcium_albumin_bound_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.40 — ~40% of total plasma calcium bound to albumin (decreases with hypoalbuminaemia)"
    )
    calcium_intestinal_absorption_fraction_adult: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.35 — midpoint 30–40%; net intestinal calcium absorption fraction in adults"
    )
    calcium_intestinal_absorption_fraction_child: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.65 — midpoint 60–70%; intestinal Ca absorption in children/adolescents (high demand)"
    )
    trpv6_km_calcium_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.3 mM — midpoint 0.1–0.5 mM; TRPV6 (apical Ca²⁺ channel) Km for luminal Ca²⁺"
    )
    trpv6_vitamin_d_induction_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.5 — midpoint 5–10×; TRPV6 + calbindin-D9k expression fold increase by 1,25(OH)₂D₃"
    )
    trpv5_renal_km_calcium_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.2 mM — TRPV5 (renal DCT apical channel) Km for Ca²⁺; primary renal reabsorption gate"
    )
    renal_calcium_reabsorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.985 — ~98.5% midpoint 98–99%; total renal Ca²⁺ reabsorption fraction"
    )
    urinary_calcium_excretion_normal_mg_per_day_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 mg/day — lower urinary calcium excretion in healthy adults"
    )
    urinary_calcium_excretion_normal_mg_per_day_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 mg/day — upper normal; >300 mg/day = hypercalciuria"
    )
    daily_calcium_intake_rda_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="1000 mg/day — RDA for calcium in adults 19–70 y; 1200 mg/day >70 y and postmenopausal"
    )
    calcium_pool_exchangeable_g: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 g — rapidly exchangeable calcium pool between plasma and bone surface"
    )

    # ── Phosphorus Metabolism ─────────────────────────────────────────────────
    plasma_phosphate_mg_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 mg/dL — lower plasma inorganic phosphate (0.81 mmol/L)"
    )
    plasma_phosphate_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.5 mg/dL — upper normal plasma phosphate (1.45 mmol/L)"
    )
    phosphorus_intestinal_absorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.70 — midpoint 60–80%; intestinal phosphorus absorption (mainly passive NaPi-IIb)"
    )
    renal_phosphate_reabsorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.88 — ~85–97% renal phosphate reabsorption (tubular maximum ~6.5 mg/min)"
    )
    ca_p_ratio_hydroxyapatite_mass: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.15 — midpoint ~2.2:1 calcium:phosphorus mass ratio in stoichiometric hydroxyapatite"
    )
    fgf23_plasma_normal_ru_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="44 RU/mL — lower normal plasma FGF-23 (osteocyte-secreted phosphaturic hormone)"
    )
    fgf23_plasma_normal_ru_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 RU/mL — upper normal FGF-23; inhibits 1-hydroxylase; drives phosphaturia"
    )

    # ── Hormonal Regulation — PTH / Vitamin D / Calcitonin ───────────────────
    pth_plasma_pg_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 pg/mL — lower normal intact PTH (parathyroid hormone 1-84)"
    )
    pth_plasma_pg_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="65 pg/mL — upper normal intact PTH"
    )
    pth_calcium_sensing_receptor_ec50_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.25 mM — CaSR EC50 for Ca²⁺ sensing on parathyroid chief cells; controls PTH secretion"
    )
    pth_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 min — intact PTH plasma t½; rapidly cleaved to inactive fragments"
    )
    calcidiol_25ohd_optimal_ng_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 ng/mL — lower optimal serum 25-OH-D₃ (calcidiol) for bone health (75 nmol/L)"
    )
    calcidiol_25ohd_optimal_ng_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 ng/mL — upper optimal 25-OH-D₃; toxicity threshold >100 ng/mL"
    )
    calcidiol_deficiency_threshold_ng_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 ng/mL — 25-OH-D₃ deficiency threshold; PTH rises; Ca absorption falls"
    )
    calcitriol_1alpha25ohd_pg_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 pg/mL — lower plasma 1,25(OH)₂D₃ (calcitriol; active hormonal form)"
    )
    calcitriol_1alpha25ohd_pg_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="60 pg/mL — upper plasma calcitriol in healthy adults"
    )
    cyp27b1_1hydroxylase_km_25ohd_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 nM — CYP27B1 (renal 1α-hydroxylase) Km for 25-OH-D₃; PTH-upregulated"
    )
    vdr_kd_calcitriol_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.2 nM — midpoint 0.1–0.3 nM; VDR nuclear receptor Kd for 1,25(OH)₂D₃"
    )
    calcitonin_osteoclast_inhibition_threshold_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 pg/mL — calcitonin concentration inhibiting osteoclast activity; basal <10 pg/mL"
    )
    vitamin_d_skin_synthesis_iu_per_day_full_sun: Mapped[Optional[float]] = mapped_column(
        Float, comment="15000 IU/day — midpoint 10,000–20,000 IU; 25-OH-D₃ precursor synthesis with full-body UVB"
    )

    # ── Bone Remodeling Cycle — Cellular Kinetics ────────────────────────────
    remodeling_cycle_activation_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 days — midpoint 0–3 days; osteoclast activation phase of BMU (basic multicellular unit)"
    )
    remodeling_cycle_resorption_weeks: Mapped[Optional[float]] = mapped_column(
        Float, comment="3 weeks — midpoint 2–4 weeks; osteoclast resorption phase per BMU"
    )
    remodeling_cycle_reversal_weeks: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 weeks — midpoint 1–2 weeks; reversal phase (mononuclear cells prepare resorption lacuna)"
    )
    remodeling_cycle_formation_months: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 months — midpoint 2–3 months; osteoblast matrix deposition + mineralisation phase"
    )
    remodeling_cycle_total_months: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.5 months — midpoint 3–6 months; total BMU remodeling cycle duration"
    )
    cortical_bone_annual_turnover_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.06 — midpoint 2–10%/year; cortical bone annual turnover fraction"
    )
    trabecular_bone_annual_turnover_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — midpoint 20–30%/year; trabecular bone annual turnover fraction"
    )
    osteoblast_active_lifespan_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 days — midpoint 10–30 days; active osteoblast lifespan before apoptosis/embedding"
    )
    osteoclast_lifespan_weeks: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 weeks — midpoint 2–3 weeks; multinucleated osteoclast lifespan"
    )
    osteocyte_lifespan_years: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 years — midpoint 20–30 years; osteocyte lifespan embedded in bone matrix"
    )
    osteocyte_density_per_mm3: Mapped[Optional[float]] = mapped_column(
        Float, comment="20000 per mm³ — midpoint 14,000–26,000; osteocyte lacunar density in human bone"
    )
    osteocyte_canalicular_network_length_per_osteocyte_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 mm — ~100 mm of canalicular channels per osteocyte for nutrient/signal transport"
    )
    bone_mineralisation_rate_um_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.75 µm/day — midpoint 0.5–1.0 µm/day; mineralisation appositional rate (tetracycline labelling)"
    )
    rankl_rank_kd_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 nM — midpoint 0.1–1 nM; RANKL binding affinity for RANK receptor; drives osteoclastogenesis"
    )
    opg_rankl_kd_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 nM — OPG (osteoprotegerin) Kd for RANKL; decoy receptor inhibiting osteoclast formation"
    )
    cathepsin_k_km_collagen_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM — cathepsin K Km for collagen substrate (osteoclast resorption lacuna pH 4.5)"
    )
    cathepsin_k_vmax_nmol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 nmol/min/mg — cathepsin K Vmax for collagen degradation in resorption lacuna"
    )

    # ── Bone Turnover Biomarkers ──────────────────────────────────────────────
    p1np_formation_marker_ug_per_l_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 µg/L — lower P1NP (procollagen type I N-terminal propeptide); bone formation marker"
    )
    p1np_formation_marker_ug_per_l_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="75 µg/L — upper P1NP in healthy adults; elevates 2–5× during puberty/exercise"
    )
    ctx_resorption_marker_pg_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="600 pg/mL — upper normal serum β-CTX (C-terminal telopeptide); bone resorption marker"
    )
    osteocalcin_plasma_ng_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 ng/mL — lower serum osteocalcin (bone Gla protein); secreted exclusively by osteoblasts"
    )
    osteocalcin_plasma_ng_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 ng/mL — upper serum osteocalcin; also a metabolic hormone (insulin sensitiser)"
    )
    alp_bone_specific_u_per_l_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 U/L — lower bone-specific alkaline phosphatase (BALP); osteoblast mineralisation enzyme"
    )
    alp_bone_specific_u_per_l_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="65 U/L — upper BALP in healthy adults"
    )
    dpd_deoxypyridinoline_nmol_per_mmol_creatinine_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="8 nmol/mmol creatinine — upper normal urinary DPD; collagen crosslink resorption marker"
    )

    # ── Wolff's Law — Mechanostat and Bone Strain Thresholds ─────────────────
    bone_strain_habitual_walking_microstrain_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="1000 µε — lower in vivo bone strain during walking (telemetry/strain gauge data)"
    )
    bone_strain_habitual_walking_microstrain_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="3000 µε — upper in vivo strain during vigorous walking; cortical mid-diaphysis"
    )
    bone_strain_running_femur_microstrain_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="2000 µε — lower peak bone strain femur during running at 4 m/s"
    )
    bone_strain_running_femur_microstrain_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="3500 µε — upper peak bone strain during sprinting/high-impact running"
    )
    mechanostat_minimum_effective_strain_modeling_microstrain: Mapped[Optional[float]] = mapped_column(
        Float, comment="1250 µε — midpoint 1000–1500 µε; MESm: threshold above which bone modelling begins (Frost)"
    )
    mechanostat_remodeling_threshold_microstrain: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 µε — midpoint 100–300 µε; MESr: below which disuse resorption is triggered (Frost)"
    )
    mechanostat_fracture_threshold_microstrain: Mapped[Optional[float]] = mapped_column(
        Float, comment="25000 µε — MESfx: bone failure strain (~2.5% elongation); ultimate strain threshold"
    )
    bone_adaptation_onset_novel_loading_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="84 h — midpoint 72–96 h; time from novel mechanical loading to measurable bone formation response"
    )
    strain_rate_osteogenic_threshold_microstrain_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1000 µε/s — strain rate above which osteogenic stimulus is potentiated (dynamic > static loading)"
    )
    osteogenic_loading_cycles_per_session: Mapped[Optional[float]] = mapped_column(
        Float, comment="68 — midpoint 36–100; loading cycles per session yielding maximal adaptive signal (diminishing returns beyond)"
    )
    bone_fluid_flow_shear_stress_threshold_pa: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 Pa — midpoint 2–3 Pa; canalicular fluid flow shear stress activating osteocyte mechanosensing"
    )
    piezoelectric_voltage_per_mpa_mv: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 mV/MPa — midpoint 1–10 mV/MPa; collagen piezoelectric voltage generated per MPa compression"
    )
    impact_loading_body_weight_running_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 — lower peak ground reaction force during running (~5× body weight at heel strike)"
    )
    impact_loading_body_weight_running_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="12 — upper peak GRF during running/jumping (~12× BW; site-specific loading)"
    )
    bone_stress_injury_threshold_cyclic_microstrain: Mapped[Optional[float]] = mapped_column(
        Float, comment="6000 µε — cyclic strain amplitude above which fatigue microdamage accumulates faster than repair"
    )
