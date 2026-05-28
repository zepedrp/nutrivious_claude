from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesMitochondrial(Base):
    """
    Phase 1 — Mitochondrial ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 — Section 1.3 (OXPHOS); Domain II.6
    (Mitochondrial density, P/O ratio, COX kinetics, myoglobin diffusion);
    Domain II.7 Bottleneck 3 (beta-oxidation / Krebs capacity limit).

    Covers: ETC Complexes I–V, PMF, proton leak, electron carriers,
    organelle ultrastructure, biogenesis (PGC-1α/TFAM), dynamics
    (DRP1/MFN2/mitophagy), ROS production, shuttle systems.

    Units:
      _nmol_mg_min   = nmol / mg protein / min
      _mv            = millivolts
      _um            = µM
      _nm            = nM
      _pmol_mg_min   = pmol / mg protein / min
      _kj_mol        = kJ / mol
      _umol_min_g    = µmol / min / g
    """

    __tablename__ = "species_mitochondrial"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="mitochondrial")

    # ── Complex I — NADH:Ubiquinone Oxidoreductase ────────────────────────────
    complex_i_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_i_km_nadh_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_i_km_ubiquinone_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_i_protons_translocated_per_2e: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 4 H⁺/2e⁻
    complex_i_gibbs_energy_kj_mol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_i_superoxide_electron_leak_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # fraction of electron flow → O₂•⁻
    complex_i_rotenone_ki_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_i_fmn_reduction_rate_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_i_fe_s_cluster_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)                 # 8 Fe-S clusters

    # ── Complex II — Succinate:Ubiquinone Oxidoreductase (SDH) ───────────────
    complex_ii_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_ii_km_succinate_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_ii_km_ubiquinone_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_ii_ki_oxaloacetate_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # potent product inhibitor
    complex_ii_fad_redox_midpoint_potential_mv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Complex III — Ubiquinol:Cytochrome c Oxidoreductase (bc1) ────────────
    complex_iii_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iii_km_ubiquinol_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iii_km_cytochrome_c_oxidized_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iii_protons_translocated_per_2e: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 4 H⁺/2e⁻ (Q-cycle mechanism)
    complex_iii_antimycin_ki_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iii_q_cycle_semiquinone_lifetime_ns: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iii_rieske_fe_s_midpoint_potential_mv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# +300 mV

    # ── Complex IV — Cytochrome c Oxidase (CcO / COX) ────────────────────────
    # From document: Km for O2 = 0.1–0.3 µM (≡ 0.07 mmHg); saturated when pO2 > 0.5 mmHg
    # Critical mitochondrial pO2 = 0.5–1.0 mmHg (measured by EPR in contracting muscle)
    # Krogh cylinder: pO2_mito = pO2_cap - (VO2 × r²) / (4 × D_O2)
    complex_iv_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iv_km_cytochrome_c_reduced_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iv_km_o2_um_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                      # 0.1 µM
    complex_iv_km_o2_um_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                     # 0.3 µM
    complex_iv_km_o2_mmhg_equivalent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # ~0.07 mmHg
    complex_iv_critical_po2_mitochondria_mmhg_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 0.5 mmHg (EPR-measured)
    complex_iv_critical_po2_mitochondria_mmhg_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 1.0 mmHg
    complex_iv_protons_translocated_per_2e: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 2 H⁺/2e⁻
    complex_iv_cyanide_ki_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iv_turnover_number_per_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_iv_cox4i1_to_cox4i2_isoform_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # COX4I2 upregulated at altitude/low O2
    complex_iv_copper_a_midpoint_potential_mv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # +245 mV

    # ── Complex V — F₀F₁-ATP Synthase ────────────────────────────────────────
    # From document: n_H⁺/ATP ≈ 2.7; P/O ratio (NADH-linked) = 2.3–2.5
    # ΔG_pmf / ΔG_ATP_synt = 200 mV × F / (ΔG_ATP / n_H⁺)
    complex_v_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_v_km_adp_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_v_km_pi_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_v_h_per_atp_stoichiometry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # ~2.7 (from document)
    complex_v_p_o_ratio_nadh_substrates_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 2.3 mol ATP / mol O atom (from document)
    complex_v_p_o_ratio_nadh_substrates_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 2.5 mol ATP / mol O atom (from document)
    complex_v_p_o_ratio_fadh2_substrates: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # ~1.5–1.6
    complex_v_proton_conductance_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_v_oligomycin_ki_ng_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_v_c_ring_rotation_rate_per_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_v_gamma_subunit_torque_pn_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complex_v_delta_g_atp_synthesis_kj_mol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Proton Motive Force (PMF) & Membrane Potential ───────────────────────
    pmf_resting_mv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pmf_maximal_uncoupled_mv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    membrane_potential_delta_psi_resting_mv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # ~-180 mV
    delta_ph_matrix_to_ims: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                       # ~0.5–1.0 pH units
    inner_membrane_capacitance_uf_per_mg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    proton_motive_force_phosphorylation_potential_kj_mol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    inner_membrane_area_per_mg_protein_um2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Proton Leak & Uncoupling Proteins ─────────────────────────────────────
    basal_proton_leak_rate_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    leak_fraction_of_basal_o2_consumption: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # ~15–25% at rest
    ucp2_proton_conductance_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ucp3_proton_conductance_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # skeletal muscle
    ant_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                         # Adenine Nucleotide Translocator
    ant_km_adp_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    phosphate_carrier_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Electron Carrier Pools ────────────────────────────────────────────────
    coenzyme_q10_total_pool_nmol_mg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coenzyme_q10_reduced_fraction_resting: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cytochrome_c_total_pool_nmol_mg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cytochrome_c_reduced_fraction_resting: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nad_total_pool_matrix_nmol_mg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nadh_nad_ratio_matrix_resting: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nadh_nad_ratio_matrix_state3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # state 3 = max ADP-stimulated
    fad_total_pool_nmol_mg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Shuttle Systems ───────────────────────────────────────────────────────
    malate_aspartate_shuttle_flux_max_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    glycerol_3_phosphate_shuttle_flux_max_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nad_kinase_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── O2 Transport — Myoglobin (from document Section 1.3) ─────────────────
    # D_effective = D_free × [Mb] × (ΔO2_sat) / ΔC_O2; facilitates by 2–4×
    # D_O2 muscle = 1.7 × 10⁻⁵ cm²/s  (J = -D × dC/dx)
    o2_diffusion_coefficient_muscle_cm2_per_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 1.7×10⁻⁵ cm²/s (from document)
    myoglobin_d_facilitation_fold_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 2×
    myoglobin_d_facilitation_fold_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # 4×
    myoglobin_concentration_type_i_mm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 0.3 mM (from document)
    myoglobin_concentration_type_i_mm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 0.5 mM
    myoglobin_concentration_type_iia_mm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 0.1 mM (from document)
    myoglobin_concentration_type_iia_mm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 0.2 mM
    myoglobin_p50_o2_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                          # ~1.5 µM (high O2 affinity)

    # ── Mitochondrial Density & Ultrastructure ────────────────────────────────
    # From document: MitoVD conflict with myofibrillar volume (key trade-off)
    # Type I ceiling ~35%; Type IIx ~10%
    mitochondrial_volume_fraction_type_i_resting_sedentary: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.20–0.25
    mitochondrial_volume_fraction_type_i_trained: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # 0.30–0.35
    mitochondrial_volume_fraction_type_i_ceiling: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # ~0.35–0.40 (from document)
    mitochondrial_volume_fraction_type_iia_trained: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # 0.15–0.20
    mitochondrial_volume_fraction_type_iia_ceiling: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # ~0.25
    mitochondrial_volume_fraction_type_iix_trained: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # 0.04–0.07
    mitochondrial_volume_fraction_type_iix_ceiling: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # ~0.10
    cristae_membrane_area_per_mitochondrion_um2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mitochondria_count_per_cell_type_i: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    matrix_volume_per_mitochondrion_fl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outer_membrane_vdac_conductance_ns: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    subsarcolemmal_fraction_of_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # ~0.20 (20%)
    intermyofibrillar_fraction_of_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # ~0.80 (80%)

    # ── O2 Consumption Kinetics ───────────────────────────────────────────────
    # From document: CS activity = proxy for mitochondrial density
    citrate_synthase_athletes_umol_min_g_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 25 µmol/min/g
    citrate_synthase_athletes_umol_min_g_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 45 µmol/min/g
    citrate_synthase_sedentary_umol_min_g_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 5 µmol/min/g
    citrate_synthase_sedentary_umol_min_g_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 8 µmol/min/g
    state3_respiration_rate_max_nmol_o2_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # max ADP-stimulated
    state4_respiration_rate_nmol_o2_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # resting (leak-driven)
    respiratory_control_ratio_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # state3/state4
    p50_o2_mitochondria_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                       # COX Km range: 0.1–0.3 µM (from document)

    # ── Mitochondrial Biogenesis (PGC-1α / TFAM axis) ────────────────────────
    pgc1a_mrna_baseline_arbitrary_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tfam_binding_affinity_km_dna_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mtdna_copy_number_per_cell_type_i: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nrf1_transcription_factor_activity_baseline_au: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mitochondrial_rrna_transcription_rate_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tom40_protein_import_capacity_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Mitochondrial Dynamics (Fission / Fusion / Mitophagy) ────────────────
    drp1_fission_rate_constant_per_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mfn2_fusion_rate_constant_per_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mitophagy_flux_basal_arbitrary_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pink1_parkin_activation_threshold_delta_psi_mv: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # depolarization threshold for mitophagy

    # ── ROS Production & Antioxidant Defence ─────────────────────────────────
    superoxide_production_rate_complex_i_pmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    superoxide_production_rate_complex_iii_pmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    h2o2_production_rate_baseline_pmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mnsod_activity_umol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                    # SOD2
    gpx1_mitochondrial_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thioredoxin_reductase_2_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
