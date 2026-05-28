from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesBioenergetics(Base):
    """
    Phase 1 — Bioenergetic ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 — Sections 1.1 (PCr), 1.2 (Glycolysis),
    1.3 (OXPHOS), Domain II.4–6 (Power-Duration framework).

    Units encoded in column names:
      _mmol_kg_dm   = mmol / kg dry mass
      _mmol_kg_min  = mmol / kg / min
      _umol_min_g   = µmol / min / g wet tissue
      _kj_mol       = kJ / mol
      _ml_kg_min    = mL / kg / min
      _ml_per_l     = mL / L blood
      _mm           = mM (millimolar)
      _um           = µM (micromolar)
      _cm2_s        = cm² / s
      _fraction     = dimensionless ratio 0–1
      _w            = Watts
      _kj           = kilojoules
      _s            = seconds
      _min          = minutes
    """

    __tablename__ = "species_bioenergetics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="bioenergetics")

    # ── Power-Duration Relationship (Monod & Scherrer 1965; Whipp & Wasserman 1972) ──
    # P × t = W' + CP × t  →  P = CP + W'/t
    power_duration_equation_note: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    critical_power_elite_60min_w_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 380 W (WorldTour cyclists)
    critical_power_elite_60min_w_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 480 W
    critical_power_fraction_of_map_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 0.75
    critical_power_fraction_of_map_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 0.85
    w_prime_anaerobic_capacity_kj_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 20 kJ (trained cyclists)
    w_prime_anaerobic_capacity_kj_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 25 kJ
    mechanical_efficiency_cycling_eta_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.24
    mechanical_efficiency_cycling_eta_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 0.26

    # ── MATRIX 1.1 — Phosphagenic System (PCr) ────────────────────────────────
    # CK: PCr + ADP + H⁺ ⇌ Cr + ATP  (ΔG°' = -12.5 kJ/mol; Keq ≈ 160)
    pcr_resting_concentration_mmol_kg_dm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 75–80 mmol/kg dm
    pcr_max_concentration_with_loading_mmol_kg_dm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 155 mmol/kg dm (SLC6A8 saturation)
    slc6a8_transporter_km_creatine_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # ~30 µM
    atp_stock_resting_mmol_kg_dm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # 25–30 mmol/kg dm
    ck_atp_resynthesis_rate_max_mmol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # ~120 mmol ATP/kg dm/min
    ck_vmax_umol_min_g_wet_tissue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # ~1000 µmol/min/g wet
    ck_kcat_per_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                                # 4×10³ s⁻¹
    ck_equilibrium_constant_keq: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # ~160
    ck_reaction_delta_g_kj_mol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # -12.5 kJ/mol
    atp_delta_g_hydrolysis_in_vivo_kj_mol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # -54 kJ/mol in vivo
    pcr_system_atp_flux_max_mmol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # ~73 mmol ATP/kg/min
    pcr_duration_at_max_power_s_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # 8 s
    pcr_duration_at_max_power_s_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # 12 s
    pcr_depletion_fraction_at_10s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # 0.95 (95% depleted at 10s)
    pcr_50pct_depletion_time_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # 3–4 s
    pcr_90pct_depletion_time_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # 8–12 s
    pcr_recovery_half_life_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                     # 26–30 s
    pcr_recovery_tau_first_order_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # 30–45 s (τ in [PCr](t)=[PCr]₀×(1-e^(-t/τ)))
    pcr_95pct_recovery_time_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # 3–5 min
    peak_power_3s_burst_elite_w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # ~2800 W (cycling, 3s)
    peak_power_10s_sprint_low_w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # 1600 W (30 kg active muscle)
    peak_power_10s_sprint_high_w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # 2000 W
    peak_power_0_1s_absolute_w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # ~3500 W (elite sprinter estimated by cinematography)
    peak_power_per_kg_active_muscle_low_w_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 150 W/kg
    peak_power_per_kg_active_muscle_high_w_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 200 W/kg

    # ── MATRIX 1.2 — Glycolytic System ───────────────────────────────────────
    # Rate-limiting enzyme: PFK-1  (v = Vmax × [S] / (Km + [S]))
    # PFK activity: f([AMP][F26BP] / [ATP][Citrate][H⁺])
    pfk1_vmax_umol_f6p_min_g_wet_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # 300 µmol/min/g wet
    pfk1_vmax_umol_f6p_min_g_wet_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # 400 µmol/min/g wet
    pfk1_km_f6p_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                             # ~0.1 mM
    pfk1_active_site_pka_h_inhibition: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # 6.8 (pKa of active site His residues)
    pfk1_activity_residual_at_ph_6_5_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.20 (20% of Vmax at pH 6.5)
    pfk1_activity_loss_per_0_1_ph_unit_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 0.25 (25% loss per 0.1 pH below 6.8)
    ldh_vmax_umol_min_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                        # ~1200 µmol/min/g (not rate-limiting)
    lactate_production_rate_max_mmol_kg_wet_min_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 3 mmol/kg wet/min
    lactate_production_rate_max_mmol_kg_wet_min_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 4 mmol/kg wet/min
    glycolytic_atp_flux_max_mmol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # ~53 mmol ATP/kg muscle/min
    glucose_to_atp_anaerobic_net_mol_per_mol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 2 (net; 1 glucose → 2 ATP + 2 lactate)
    lactate_per_glucose_anaerobic_mol_per_mol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 2
    muscle_ph_minimum_functional: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # 6.3
    muscle_ph_collapse_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # 6.2 (below: functional collapse)
    muscle_ph_minimum_documented_exercise: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 6.3–6.5
    muscle_lactate_max_mmol_kg_dm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # 25 mmol/kg dm
    muscle_lactate_max_mmol_kg_dm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 30 mmol/kg dm
    blood_lactate_max_elite_mmol_l_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 22 mmol/L
    blood_lactate_max_elite_mmol_l_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 26 mmol/L
    mlss_blood_lactate_mmol_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # 4 mmol/L (Maximum Lactate Steady State)
    mlss_hepatic_gng_clearance_mmol_kg_min_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 0.6 mmol/kg/min (Cori cycle)
    mlss_hepatic_gng_clearance_mmol_kg_min_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 0.8 mmol/kg/min
    pepck_hepatic_vmax_umol_min_g_liver: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # ~40 µmol/min/g liver (PEPCK; rate-limiting for GNG from lactate)
    anaerobic_atp_whole_body_max_mmol_min_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 200 mmol ATP/min (20-25 kg active muscle × 4 × 2)
    anaerobic_atp_whole_body_max_mmol_min_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 240 mmol ATP/min
    anaerobic_power_max_sustainable_w_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 600 W mechanical (~150% VO2max)
    anaerobic_power_max_sustainable_w_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 900 W mechanical (~170% VO2max)
    anaerobic_max_duration_s_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # 60 s
    anaerobic_max_duration_s_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # 90 s
    lt2_anaerobic_threshold_vo2max_fraction_elite_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.85
    lt2_anaerobic_threshold_vo2max_fraction_elite_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 0.92
    henderson_hasselbalch_pka_bicarbonate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 6.1 (pH = 6.1 + log([HCO₃⁻]/[0.0307×pCO₂]))

    # ── MATRIX 1.3 — Oxidative Phosphorylation (OXPHOS) ──────────────────────
    # VO2max = Q̇_max × (CaO2 - CvO2)_max   [Fick equation]
    # Q̇ = FC × VS;  CaO2 = [Hb] × 1.34 × SaO2 + 0.003 × PaO2
    vo2max_record_documented_ml_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 97.5 mL/kg/min (Oskar Svendsen 2012)
    vo2max_absolute_record_l_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # ~8.0 L/min
    vo2max_practical_species_ceiling_ml_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # ~100–105 mL/kg/min (attainable)
    vo2max_theoretical_absolute_ceiling_ml_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # ~128 mL/kg/min (unattainable; all components simultaneously maximized)
    fc_max_species_ceiling_bpm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # 220 bpm
    fc_max_species_ceiling_bpm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 230 bpm (documented in prepubertal children, extreme athletes)
    fc_max_tanaka_formula_slope: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # 0.7 (FC_max ≈ 208 - 0.7 × age; Tanaka 2001)
    fc_max_tanaka_formula_intercept: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 208
    vs_max_ml_sedentary_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                    # 70 mL
    vs_max_ml_sedentary_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # 90 mL
    vs_max_ml_elite_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                        # 170 mL (eccentric hypertrophy)
    vs_max_ml_elite_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                       # 220 mL
    vs_max_ml_theoretical_ceiling: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # ~230 mL
    edv_max_ml_elite: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                           # 270–320 mL (end-diastolic volume)
    esv_min_ml_elite: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                           # 60–80 mL (end-systolic volume)
    q_max_l_min_sedentary_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # 14 L/min
    q_max_l_min_sedentary_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # 18 L/min
    q_max_l_min_elite_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                      # 40 L/min
    q_max_l_min_elite_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                     # 45 L/min
    q_max_l_min_theoretical_ceiling: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # ~48 L/min
    a_vo2_diff_max_ml_o2_per_l_blood_elite_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 170 mL/L (15 mL/100mL)
    a_vo2_diff_max_ml_o2_per_l_blood_elite_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 180 mL/L
    a_vo2_diff_max_ml_o2_per_l_blood_sedentary_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 100 mL/L
    a_vo2_diff_max_ml_o2_per_l_blood_sedentary_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 140 mL/L
    cao2_max_ml_per_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                          # ~200 mL/L
    cvo2_min_ml_per_l_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                      # 20 mL/L
    cvo2_min_ml_per_l_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                     # 30 mL/L
    o2_extraction_fraction_max_type_i_fibers: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # ~0.90 (90% of transported O2)
    svo2_min_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                          # 0.15–0.20 (vs 0.25 at rest)
    p_o_ratio_nadh_linked_mol_atp_per_mol_o_atom_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 2.3
    p_o_ratio_nadh_linked_mol_atp_per_mol_o_atom_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 2.5
    f0f1_atp_synthase_h_per_atp_stoichiometry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # ~2.7 H⁺/ATP (n_H⁺/ATP in F₀F₁)
    cox_km_o2_um_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                           # 0.1 µM
    cox_km_o2_um_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                          # 0.3 µM (≡ ~0.07 mmHg pO2)
    cox_km_o2_mmhg_equivalent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # ~0.07 mmHg (COX saturated when pO2 > 0.5 mmHg)
    mitochondrial_critical_po2_mmhg_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 0.5 mmHg (Krogh cylinder model)
    mitochondrial_critical_po2_mmhg_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 1.0 mmHg
    o2_diffusion_coefficient_muscle_cm2_per_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 1.7×10⁻⁵ cm²/s (J = -D × dC/dx)
    myoglobin_d_facilitation_fold_increase_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 2×
    myoglobin_d_facilitation_fold_increase_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 4× (D_Mb = D_free × [Mb] × ΔO2_sat / ΔC_O2)
    myoglobin_concentration_type_i_mm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 0.3 mM
    myoglobin_concentration_type_i_mm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 0.5 mM
    myoglobin_concentration_type_iia_mm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 0.1 mM
    myoglobin_concentration_type_iia_mm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 0.2 mM
    citrate_synthase_athletes_umol_min_g_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 25 µmol/min/g (proxy for mitochondrial density)
    citrate_synthase_athletes_umol_min_g_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 45 µmol/min/g
    citrate_synthase_sedentary_umol_min_g_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 5 µmol/min/g
    citrate_synthase_sedentary_umol_min_g_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 8 µmol/min/g
    mitochondrial_volume_fraction_type_i_ceiling: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 0.35 (35%; above: contractile protein compromised)
    capillary_transit_time_min_exercise_s_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.25 s (below: EIAH onset)
    capillary_transit_time_min_exercise_s_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 0.50 s
    capillary_transit_time_rest_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # 0.75 s
    capillary_density_trained_per_fiber_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 4 capillaries/fiber
    capillary_density_trained_per_fiber_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 6 capillaries/fiber
    capillary_density_sedentary_per_fiber_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 2 capillaries/fiber
    capillary_density_sedentary_per_fiber_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 3 capillaries/fiber
    eiah_prevalence_elite_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # 0.40–0.50 (40–50% of athletes with VO2max >68 mL/kg/min)
    eiah_sao2_floor_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # 0.92 (SaO2 drops to 92–95% in EIAH)

    # ── Glycogen Stores ───────────────────────────────────────────────────────
    glycogen_max_supercompensation_mmol_glucosyl_kg_dm: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # ~900 mmol/kg dm
    glycogen_normal_trained_mmol_glucosyl_kg_dm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 500 mmol/kg dm
    glycogen_normal_trained_mmol_glucosyl_kg_dm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 600 mmol/kg dm
    glycogen_muscle_total_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                            # ~700 g
    glycogen_liver_total_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                             # ~100 g
    glycogen_total_energy_kcal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                         # ~3200 kcal

    # ── O2 Reserves ───────────────────────────────────────────────────────────
    # O2_Hb = [Hb] × 1.34 × SaO2 × Vol_blood + O2_Mb + O2_lung + O2_dissolved
    o2_reserves_total_organism_ml_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # 1800 mL
    o2_reserves_total_organism_ml_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 2000 mL
    o2_reserves_pulmonary_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # ~250 mL
    o2_reserves_dissolved_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # ~150 mL
    o2_reserves_duration_at_vo2max_s_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 16 s
    o2_reserves_duration_at_vo2max_s_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 18 s
    fatmax_elite_fat_adapted_g_min_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 1.5 g/min (CPT-1 Ki malonyl-CoA ≈ 0.02 µM)
    fatmax_elite_fat_adapted_g_min_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 1.7 g/min
