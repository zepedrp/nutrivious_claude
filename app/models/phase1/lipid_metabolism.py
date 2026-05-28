from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesLipidMetabolism(Base):
    """
    Phase 1 — Lipid metabolism ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 1.3 (Fatmax, CPT-1 Ki);
      Section 4.3 (GI lipid absorption, bile, chylomicrons);
      Domain II.7 (Three-bottleneck model: HSL → CPT-I → β-oxidation/Krebs);
      Domain VIII.28 (adipokine signaling via leptin, adiponectin).

    Three serial bottlenecks (from document):
      Bottleneck 1 — Adipocyte mobilisation (HSL/ATGL): rate limited by
                     β₁-adrenoreceptor density (~2000/adipocyte) and
                     albumin transport capacity (~800–1200 µmol AGL/L plasma).
      Bottleneck 2 — Mitochondrial import (CPT-I):
                     Ki(malonyl-CoA) ≈ 0.02 µM — near-total inhibition during
                     active glycolysis (malonyl-CoA ↑ when insulin/citrate ↑).
      Bottleneck 3 — β-oxidation + Krebs capacity:
                     Rate limited by oxaloacetate availability for Acetyl-CoA
                     condensation; HADH (3-HAD): 5–25 µmol/min/g muscle.

    Units:
      _g_min         = g / min
      _umol_kg_min   = µmol / kg / min
      _nmol_mg_min   = nmol / mg protein / min
      _umol_min_g    = µmol / min / g wet tissue
      _mm            = mM
      _um            = µM
      _nm            = nM
    """

    __tablename__ = "species_lipid_metabolism"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="lipid_metabolism")

    # ── Fatmax — Global Ceiling (from document Section 1.3 and Domain II.7) ───
    # Three serial bottlenecks determine Fatmax.
    # CPT-1 Ki(malonyl-CoA) ≈ 0.02 µM → nearly fully inhibited when glycolysis active.
    fatmax_sedentary_g_min_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # 0.3 g/min
    fatmax_sedentary_g_min_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # 0.5 g/min
    fatmax_sedentary_vo2max_fraction_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 0.40
    fatmax_sedentary_vo2max_fraction_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 0.50
    fatmax_trained_g_min_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # 0.7 g/min
    fatmax_trained_g_min_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 1.0 g/min
    fatmax_trained_vo2max_fraction_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 0.50
    fatmax_trained_vo2max_fraction_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0.65
    fatmax_elite_fat_adapted_g_min_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 1.2 g/min
    fatmax_elite_fat_adapted_g_min_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 1.7 g/min
    fatmax_elite_vo2max_fraction_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 0.60
    fatmax_elite_vo2max_fraction_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 0.70
    fatmax_record_documented_g_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 1.73 g/min (Volek et al.)
    fatmax_theoretical_species_ceiling_g_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # ~2.0 g/min

    # ── BOTTLENECK 1 — Adipocyte Mobilisation (HSL / ATGL) ───────────────────
    # HSL: DAG + H₂O → MAG + FA;  activated via β-AR → cAMP → PKA → Ser-660
    # ATGL: TG → DAG (first hydrolysis);  MGL: MAG → glycerol + FA (completion)
    beta1_adrenoreceptor_density_per_adipocyte: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # ~2000 receptors/adipocyte (from document)
    hsl_activation_pka_phosphorylation_site_ser: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 660.0 → Ser-660 residue number
    hsl_km_dag_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                         # ~0.1 mM (from document; DAG as substrate)
    hsl_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    atgl_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mgl_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    adipose_lipolysis_rate_max_umol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plasma_nefa_max_concentration_mmol_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    glycerol_release_rate_max_umol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    imtg_concentration_type_i_mmol_kg_dm: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # intramuscular TG, type I fibers
    imtg_concentration_type_ii_mmol_kg_dm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    imtg_utilisation_rate_max_mmol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Albumin-AGL Transport System ──────────────────────────────────────────
    # From document: 3–5 high-affinity binding sites/albumin; [albumin] ~40 g/L
    # → transport capacity ~800–1200 µmol AGL/L plasma
    albumin_plasma_concentration_g_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # ~40 g/L
    albumin_high_affinity_binding_sites_per_molecule_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 3
    albumin_high_affinity_binding_sites_per_molecule_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 5
    albumin_nefa_transport_capacity_umol_per_l_plasma_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 800 µmol/L
    albumin_nefa_transport_capacity_umol_per_l_plasma_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 1200 µmol/L

    # ── FFA Membrane Transport ────────────────────────────────────────────────
    cd36_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # FAT/CD36 (fatty acid translocase)
    cd36_km_fatty_acid_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fabp_cytoplasmic_concentration_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # FABP intracellular
    fabp_km_fatty_acid_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fatp1_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # FATP1/SLC27A1
    fatp4_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # FATP4/SLC27A4 (intestinal)

    # ── BOTTLENECK 2 — Mitochondrial Import (CPT-I System) ───────────────────
    # CPT-I is the gatekeeper: inhibited by malonyl-CoA (product of ACC)
    # When insulin/citrate ↑ → ACC active → malonyl-CoA ↑ → CPT-I OFF → no β-ox
    # At >65–70% VO2max: malonyl-CoA ↑ → Fatmax suppressed.
    # d[Malonyl-CoA]/dt = k_ACC × [Acetyl-CoA] × [ATP] - k_MCD × [Malonyl-CoA]
    cpt1a_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # CPT1A (liver isoform)
    cpt1b_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # CPT1B (muscle isoform; dominant in skeletal muscle)
    cpt1_km_palmitoyl_coa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cpt1_km_l_carnitine_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cpt1_ki_malonyl_coa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # 0.02 µM (from document — near-total inhibition)
    cpt2_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cpt2_km_palmitoyl_carnitine_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cact_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # Carnitine-Acylcarnitine Translocase
    total_carnitine_muscle_mmol_kg_dm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    free_carnitine_muscle_resting_mmol_kg_dm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    acylcarnitine_fraction_resting: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    malonyl_coa_resting_nmol_per_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    acc2_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # Acetyl-CoA Carboxylase 2 (mitochondrial gate)
    acc2_km_acetyl_coa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mcd_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # Malonyl-CoA Decarboxylase (reverses ACC2)

    # ── BOTTLENECK 3 — β-Oxidation Enzymes ───────────────────────────────────
    # From document: HADH (3-HAD) = 5–25 µmol/min/g muscle (rate-limiting within β-ox)
    # β-oxidation cycle: 7 cycles per palmitoyl-CoA → 8 acetyl-CoA + 7 FADH2 + 7 NADH
    # ATP yield: 1 palmitoyl-CoA → 131 ATP net
    vlcad_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # VLCAD (C14–C20)
    vlcad_km_palmitoyl_coa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lcad_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # LCAD (C12–C18)
    mcad_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # MCAD (C6–C12)
    mcad_km_octanoyl_coa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scad_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # SCAD (C4–C6)
    enoyl_coa_hydratase_vmax_umol_min_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hadh_vmax_umol_min_g_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # 5 µmol/min/g (from document — 3-HAD / HADH)
    hadh_vmax_umol_min_g_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 25 µmol/min/g (from document)
    hadh_km_hydroxyacyl_coa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hadh_km_nad_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thiolase_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # Ketoacyl-CoA thiolase
    beta_oxidation_cycles_per_palmitoyl_coa: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 7 cycles
    beta_oxidation_acetyl_coa_per_palmitoyl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 8 acetyl-CoA
    beta_oxidation_fadh2_per_cycle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # 1 FADH2/cycle
    beta_oxidation_nadh_per_cycle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 1 NADH/cycle
    atp_yield_per_palmitoyl_coa_net_mol: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # ~131 ATP net

    # ── Ketone Body Metabolism ─────────────────────────────────────────────────
    hmgcs2_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # HMG-CoA Synthase 2 (mitochondrial; ketogenesis)
    hmgcl_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # HMG-CoA Lyase
    bdh1_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # β-hydroxybutyrate dehydrogenase
    plasma_ketone_max_concentration_mmol_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plasma_ketone_resting_fasting_mmol_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ketone_oxidation_rate_brain_max_umol_g_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    oxct1_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # Succinyl-CoA:3-ketoacid transferase (peripheral utilisation)

    # ── GI Lipid Absorption (from document Section 4.3 / Domain I.3) ─────────
    # Bile acid pool: ~2–4 g; recycled 6–10×/day → 12–40 g/day effective
    # Chylomicrons synthesised by enterocyte (ApoB-48); → lymph → thoracic duct
    bile_acid_pool_g_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # 2 g
    bile_acid_pool_g_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # 4 g
    bile_acid_recycling_cycles_per_day_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 6×/day
    bile_acid_recycling_cycles_per_day_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 10×/day
    bile_acid_effective_daily_flux_g_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 12 g/day
    bile_acid_effective_daily_flux_g_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 40 g/day
    pancreatic_lipase_concentration_g_per_l_juice: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 2–3 g/L
    intestinal_fat_absorption_rate_max_g_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # ~100–150 g/h (optimised)
    intestinal_fat_absorption_daily_max_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # ~400–600 g/day
    micelle_diameter_nm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # 3 nm
    micelle_diameter_nm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # 6 nm

    # ── De Novo Lipogenesis & Lipogenesis Control ──────────────────────────────
    fas_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # Fatty Acid Synthase
    acc1_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # ACC1 (cytosolic; lipogenesis)
    acc1_km_acetyl_coa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    malic_enzyme_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # NADPH supplier for FAS

    # ── Cholesterol & Lipoprotein System ──────────────────────────────────────
    hmgcr_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # HMG-CoA Reductase (rate-limiting for cholesterol synthesis)
    hmgcr_km_hmg_coa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cholesterol_synthesis_rate_liver_mg_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # ~800–1000 mg/day
    ldlr_density_hepatocyte_receptors_per_cell: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vldl_secretion_rate_mg_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lpl_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # Lipoprotein Lipase
    lpl_km_triacylglycerol_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lcat_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # Lecithin-Cholesterol Acyltransferase (HDL maturation)

    # ── Adipokine Signalling (from document Domain VIII) ──────────────────────
    # Leptin: produced by WAT; receptor ObRb in hypothalamus (Kd ~0.1–1 nM)
    # Suppresses NPY/AgRP (hunger) → activates POMC (satiety)
    leptin_plasma_range_ng_ml_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 5 ng/mL (lean)
    leptin_plasma_range_ng_ml_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 20 ng/mL (proportional to fat mass)
    leptin_obrb_receptor_kd_nm_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 0.1 nM
    leptin_obrb_receptor_kd_nm_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 1.0 nM
    adiponectin_plasma_concentration_ug_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 5–30 µg/mL (inversely proportional to adiposity)
    resistin_plasma_concentration_ng_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
