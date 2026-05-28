from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesHepatic(Base):
    """
    Phase 1 — Hepatic processing ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 5 / Matrices 5.1–5.4 (biotransformation, glutathione, urea cycle, Cori cycle);
      Domain II.1 (CYP450 Phase I kinetics); Domain II.2 (Phase II conjugation);
      Domain II.3 (urea cycle / ammonia clearance); Domain II.4 (hepatic gluconeogenesis / Cori);
      Domain II.5 (ketogenesis and bile acid synthesis).

    Key equations encoded:
      CL_H = Q_H × (f_u × CL_int) / (Q_H + f_u × CL_int)   [hepatic clearance, well-stirred]
      NH3_clearance = CPS1_Vmax × [NH3] / (Km_NH3 + [NH3])   [Michaelis-Menten, CPS1 rate-limiting]
      GNG_flux = f(PEPCK, FBPase1, G6Pase)                    [gluconeogenesis bottleneck chain]

    Units:
      _pmol_mg      = pmol / mg microsomal protein
      _nmol_min_mg  = nmol / min / mg microsomal protein
      _um           = µM
      _mm           = mM
      _umol_g_h     = µmol / g tissue / h
      _umol_min_g   = µmol / min / g tissue
      _meq_day      = mEq / day
      _g_per_day    = g / day
      _g_per_h      = g / hour
      _ml_per_min   = mL / min
      _fraction     = dimensionless 0–1
    """

    __tablename__ = "species_hepatic"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="hepatic")

    # ── CYP450 — Phase I Biotransformation (Oxidation/Reduction/Hydrolysis) ──
    # CL_H = Q_H × (f_u × CL_int) / (Q_H + f_u × CL_int); ceiling = hepatic blood flow Q_H
    # NADPH + O₂ consumed per oxidation cycle; CYP-reductase stoichiometry 1:1
    cyp450_total_content_pmol_per_g_liver: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1000 pmol total CYP450/g liver; sets Vmax ceiling for all Phase I oxidations; from Domain II.1"
    )
    cyp450_microsomal_protein_mg_per_g_liver: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40 mg microsomal protein/g liver; denominator for all per-mg Vmax values; from Domain II.1"
    )
    cyp450_hepatocytes_per_g_liver: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1.5×10⁸ hepatocytes/g liver; cellular density context for per-cell CYP content; from Domain II.1"
    )
    cyp3a4_fraction_total_cyp: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30–0.40; CYP3A4 = 30-40% of total hepatic CYP450; metabolises ~50% of drugs; from Domain II.1"
    )
    cyp3a4_km_midazolam_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 µM; K_m^CYP3A4 lower bound (midazolam probe substrate); from Domain II.1"
    )
    cyp3a4_km_midazolam_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0 µM; K_m^CYP3A4 upper bound; from Domain II.1"
    )
    cyp3a4_vmax_nmol_min_mg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 nmol/min/mg microsomal protein; Vmax_CYP3A4 lower bound; from Domain II.1"
    )
    cyp3a4_vmax_nmol_min_mg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 nmol/min/mg microsomal protein; Vmax_CYP3A4 upper bound; from Domain II.1"
    )
    cyp2d6_fraction_total_cyp: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.02–0.04; CYP2D6 = 2-4% total CYP; metabolises alkaloids, β-blockers, opioids; from Domain II.1"
    )
    cyp1a2_fraction_total_cyp: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.13–0.15; CYP1A2 = 13-15% total CYP; caffeine Km ~350 µM; from Domain II.1"
    )
    cyp2c9_fraction_total_cyp: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.18–0.20; CYP2C9 = 18-20% total CYP; warfarin, NSAIDs, phenytoin; from Domain II.1"
    )
    cyp_nadph_o2_stoichiometry: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0; 1 NADPH + 1 O₂ consumed per CYP oxidation cycle; from Domain II.1"
    )
    hepatic_blood_flow_ml_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1200 mL/min; hepatic blood flow lower bound (~20-25% cardiac output); theoretical CL_H ceiling; from Domain II.1"
    )
    hepatic_blood_flow_ml_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1500 mL/min; hepatic blood flow upper bound; CL_H → Q_H when CL_int >> Q_H; from Domain II.1"
    )
    first_pass_extraction_ratio_ceiling: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0; theoretical maximum first-pass extraction (100%); e.g. lidocaine ~0.70, nitroglycerin ~0.99; from Domain II.1"
    )

    # ── Phase II — Glucuronidation (UGT Superfamily) ──────────────────────────
    # UDPGA cofactor transfers glucuronate onto hydroxyl/carboxyl/amino groups
    ugt1a1_km_bilirubin_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 µM; K_m^UGT1A1 lower bound for bilirubin; from Domain II.2"
    )
    ugt1a1_km_bilirubin_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 µM; K_m^UGT1A1 upper bound; from Domain II.2"
    )
    ugt_vmax_nmol_min_mg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 nmol/min/mg microsomal protein; UGT Vmax lower bound; from Domain II.2"
    )
    ugt_vmax_nmol_min_mg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 nmol/min/mg microsomal protein; UGT Vmax upper bound; from Domain II.2"
    )
    udpga_cofactor_pool_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 µM; hepatocytic UDPGA pool lower bound; limiting when substrate flux >> cofactor; from Domain II.2"
    )
    udpga_cofactor_pool_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="500 µM; hepatocytic UDPGA pool upper bound; from Domain II.2"
    )
    bilirubin_conjugation_capacity_g_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 g/day; hepatic bilirubin conjugation capacity lower bound; normal turnover 250-350 mg/day; from Domain II.2"
    )
    bilirubin_conjugation_capacity_g_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0 g/day; hepatic bilirubin conjugation capacity upper bound; from Domain II.2"
    )

    # ── Phase II — Glutathione System (GSH Synthesis, Conjugation, Recycling) ─
    # Rate-limiting: GCL (γ-glutamylcysteine ligase); recycled by GR using NADPH
    # Toxicological threshold: GSH < 30% of normal → hepatocyte necrosis zone
    hepatic_gsh_pool_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 mM; intracellular hepatic GSH lower bound (basal, cytosolic); from Domain II.2"
    )
    hepatic_gsh_pool_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10.0 mM; intracellular hepatic GSH upper bound; from Domain II.2"
    )
    mitochondrial_gsh_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 mM; mitochondrial GSH pool lower bound (~10-15% of total); from Domain II.2"
    )
    mitochondrial_gsh_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 mM; mitochondrial GSH pool upper bound; from Domain II.2"
    )
    gcl_km_glutamate_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.8 mM; K_m^GCL for glutamate; GCL = γ-glutamylcysteine ligase; from Domain II.2"
    )
    gcl_km_cysteine_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.3 mM; K_m^GCL for cysteine; rate-limiting substrate under oxidative stress; from Domain II.2"
    )
    gcl_vmax_basal_umol_g_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 µmol GSH/g liver/h; Vmax_GCL basal lower bound; from Domain II.2"
    )
    gcl_vmax_basal_umol_g_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 µmol GSH/g liver/h; Vmax_GCL basal upper bound; from Domain II.2"
    )
    gcl_induction_fold_max: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0; maximum GCL induction fold under Nrf2 activation (e.g. sulforaphane); from Domain II.2"
    )
    gst_vmax_nmol_min_mg_cytosol: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~500 nmol/min/mg cytosolic protein; Vmax_GST for electrophile conjugation; from Domain II.2"
    )
    gpx1_km_h2o2_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1.0 µM; K_m^GPx1 for H₂O₂; near-diffusion-limited peroxidase; from Domain II.2"
    )
    gssg_reductase_km_gssg_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 µM; K_m^GR for GSSG lower bound; recycles GSSG → 2 GSH (NADPH-dependent); from Domain II.2"
    )
    gssg_reductase_km_gssg_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 µM; K_m^GR for GSSG upper bound; from Domain II.2"
    )
    daily_gsh_conjugation_ceiling_umol_g_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 µmol GSH conjugates/g liver/day; GSH conjugation ceiling lower bound; from Domain II.2"
    )
    daily_gsh_conjugation_ceiling_umol_g_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 µmol GSH conjugates/g liver/day; GSH conjugation ceiling upper bound; from Domain II.2"
    )
    gsh_depletion_hepatotoxicity_threshold_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30; GSH < 30% of baseline → hepatocyte necrosis risk (APAP model); from Domain II.2"
    )

    # ── Urea Cycle — Ammonia Clearance Ceiling ────────────────────────────────
    # 5-enzyme cycle: CPS1 (rate-limiting, mitochondria) → OCT → ASS → ASL → ARG1
    # Overflow NH3 scavenged by GS in perivenous hepatocytes (low-capacity, high-affinity)
    # Ceiling: 6 g N/day ≡ 40 g dietary protein/day urea-cycle capacity
    cps1_vmax_umol_min_g_liver_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 µmol/min/g liver; CPS1 Vmax lower bound; rate-limiting urea cycle step; from Domain II.3 / Matrix 5.3"
    )
    cps1_vmax_umol_min_g_liver_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 µmol/min/g liver; CPS1 Vmax upper bound; from Domain II.3"
    )
    cps1_km_nh3_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 mM; K_m^CPS1 for NH₃ lower bound; from Domain II.3"
    )
    cps1_km_nh3_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mM; K_m^CPS1 for NH₃ upper bound; from Domain II.3"
    )
    cps1_km_hco3_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mM; K_m^CPS1 for HCO₃⁻ lower bound; from Domain II.3"
    )
    cps1_km_hco3_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 mM; K_m^CPS1 for HCO₃⁻ upper bound; from Domain II.3"
    )
    cps1_atp_per_cycle: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0; 2 ATP consumed per CPS1 catalytic cycle (NH₃ + HCO₃⁻ + 2 ATP → carbamoyl phosphate); from Domain II.3"
    )
    oct_km_carbamoyl_phosphate_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.3 mM; K_m^OCT for carbamoyl phosphate; mitochondrial; from Domain II.3"
    )
    oct_km_ornithine_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 mM; K_m^OCT for ornithine lower bound; from Domain II.3"
    )
    oct_km_ornithine_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mM; K_m^OCT for ornithine upper bound; from Domain II.3"
    )
    ass_km_citrulline_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05 mM; K_m^ASS for citrulline lower bound; cytoplasmic; from Domain II.3"
    )
    ass_km_citrulline_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10 mM; K_m^ASS for citrulline upper bound; from Domain II.3"
    )
    arg1_km_arginine_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mM; K_m^ARG1 for arginine lower bound; low-affinity/high-capacity cytoplasmic; from Domain II.3"
    )
    arg1_km_arginine_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10.0 mM; K_m^ARG1 for arginine upper bound; from Domain II.3"
    )
    urea_cycle_nitrogen_ceiling_g_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6.0 g N/day; urea cycle nitrogen clearance ceiling ≡ 40 g dietary protein/day; from Domain II.3"
    )
    urea_synthesis_ceiling_g_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~35 g urea/day; maximum urinary urea production at ceiling protein flux; from Domain II.3"
    )
    max_hepatic_ammonia_clearance_mmol_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.5 mmol NH₃/min; total hepatic ammonia clearance lower bound (urea + GS combined); from Domain II.3"
    )
    max_hepatic_ammonia_clearance_mmol_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 mmol NH₃/min; total hepatic ammonia clearance upper bound; from Domain II.3"
    )
    glutamine_synthetase_km_nh3_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.3 mM; K_m^GS for NH₃; perivenous hepatocytes scavenge overflow NH₃ post-CPS1; from Domain II.3"
    )
    glutamine_synthetase_vmax_umol_min_g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 µmol/min/g liver; Vmax_GS lower bound; from Domain II.3"
    )
    glutamine_synthetase_vmax_umol_min_g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 µmol/min/g liver; Vmax_GS upper bound; from Domain II.3"
    )

    # ── Cori Cycle — Hepatic Lactate Clearance & Gluconeogenesis ─────────────
    # Flux chain: Lactate → Pyruvate (LDH) → OAA (PC) → PEP (PEPCK) →
    #             F1,6BP (FBPase1) → G6P (G6Pase) → Glucose (export)
    # Rate-governing trio: PEPCK → FBPase1 → G6Pase
    basal_hepatic_lactate_uptake_mmol_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 mmol/min; hepatic lactate uptake at rest lower bound; from Domain II.4"
    )
    basal_hepatic_lactate_uptake_mmol_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mmol/min; hepatic lactate uptake at rest upper bound; from Domain II.4"
    )
    arterial_lactate_max_exercise_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15.0 mM; arterial lactate at VO2max lower bound; Cori cycle substrate surge; from Domain II.4"
    )
    arterial_lactate_max_exercise_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25.0 mM; arterial lactate at VO2max upper bound; from Domain II.4"
    )
    hepatic_lactate_extraction_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; hepatic extraction fraction for lactate lower bound (50-80%); from Domain II.4"
    )
    hepatic_lactate_extraction_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.80; hepatic extraction fraction for lactate upper bound; from Domain II.4"
    )
    max_gluconeogenesis_from_lactate_umol_min_g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0 µmol/min/g liver; GNG flux from lactate lower bound; from Domain II.4"
    )
    max_gluconeogenesis_from_lactate_umol_min_g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 µmol/min/g liver; GNG flux from lactate upper bound; from Domain II.4"
    )
    pepck_km_oaa_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.07 mM; K_m^PEPCK for OAA lower bound; rate-governing GNG step; from Domain II.4"
    )
    pepck_km_oaa_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10 mM; K_m^PEPCK for OAA upper bound; from Domain II.4"
    )
    pepck_vmax_umol_min_g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 µmol/min/g liver; Vmax_PEPCK lower bound; from Domain II.4"
    )
    pepck_vmax_umol_min_g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 µmol/min/g liver; Vmax_PEPCK upper bound; from Domain II.4"
    )
    fbpase1_km_f16bp_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.010 mM; K_m^FBPase1 for F1,6BP lower bound; allosteric inhibition by AMP; from Domain II.4"
    )
    fbpase1_km_f16bp_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.050 mM; K_m^FBPase1 for F1,6BP upper bound; from Domain II.4"
    )
    fbpase1_vmax_umol_min_g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 µmol/min/g liver; Vmax_FBPase1 lower bound; from Domain II.4"
    )
    fbpase1_vmax_umol_min_g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0 µmol/min/g liver; Vmax_FBPase1 upper bound; from Domain II.4"
    )
    g6pase_km_g6p_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mM; K_m^G6Pase for G6P lower bound; liver+kidney only; from Domain II.4"
    )
    g6pase_km_g6p_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0 mM; K_m^G6Pase for G6P upper bound; from Domain II.4"
    )
    g6pase_vmax_umol_min_g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 µmol/min/g liver; Vmax_G6Pase lower bound; from Domain II.4"
    )
    g6pase_vmax_umol_min_g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 µmol/min/g liver; Vmax_G6Pase upper bound; from Domain II.4"
    )
    ldh_km_pyruvate_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05 mM; K_m^LDH for pyruvate lower bound (hepatic isoform); from Domain II.4"
    )
    ldh_km_pyruvate_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.15 mM; K_m^LDH for pyruvate upper bound; from Domain II.4"
    )
    ldh_km_lactate_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="7.0 mM; K_m^LDH for lactate lower bound; from Domain II.4"
    )
    ldh_km_lactate_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10.0 mM; K_m^LDH for lactate upper bound; from Domain II.4"
    )
    max_cori_cycle_glucose_output_g_per_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 g glucose/h; Cori cycle maximum hepatic glucose output lower bound; from Domain II.4"
    )
    max_cori_cycle_glucose_output_g_per_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 g glucose/h; Cori cycle maximum hepatic glucose output upper bound; from Domain II.4"
    )
    hepatic_glucose_output_exercise_max_mg_kg_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~8 mg/kg/min; total hepatic glucose output at VO2max (GNG + glycogenolysis combined); from Domain II.4"
    )

    # ── Bile Acid Synthesis ────────────────────────────────────────────────────
    # CYP7A1 is the committed, rate-limiting step; primary → secondary (gut bacteria)
    cyp7a1_km_cholesterol_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~30 µM; K_m^CYP7A1 for cholesterol; rate-limiting step in primary bile acid synthesis; from Domain II.5"
    )
    bile_acid_synthesis_rate_g_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5 g/day; net bile acid synthesis replacing fecal loss; pool maintained at 2-4 g; from Domain II.5"
    )
    bile_flow_l_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.6 L/day; bile flow lower bound; carries bile acids, bilirubin, GSH conjugates; from Domain II.5"
    )
    bile_flow_l_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.2 L/day; bile flow upper bound; from Domain II.5"
    )

    # ── Ketogenesis Ceiling ────────────────────────────────────────────────────
    # HMGCS2 (mitochondrial) is rate-limiting; activated by SIRT3 deacetylation during fasting
    # Ketone bodies: acetoacetate + β-OHB; brain shifts from 5% → 60-70% utilisation at 3-4 days
    hmgcs2_km_acetyl_coa_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05 mM; K_m^HMGCS2 for acetyl-CoA lower bound; rate-limiting for ketogenesis; from Domain II.5"
    )
    hmgcs2_km_acetyl_coa_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10 mM; K_m^HMGCS2 for acetyl-CoA upper bound; from Domain II.5"
    )
    max_ketone_production_mmol_min_fasting: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1.0 mmol/min; maximum hepatic ketone production during prolonged fasting; from Domain II.5"
    )
    blood_ketone_ceiling_physiological_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6.0 mM; blood ketone ceiling during physiological ketosis (fasting/VLCD) lower bound; from Domain II.5"
    )
    blood_ketone_ceiling_physiological_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8.0 mM; blood ketone ceiling physiological upper bound; above this → pathological DKA; from Domain II.5"
    )
    insulin_ketogenesis_suppression_threshold_iu_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10 µU/mL; plasma insulin above which ketogenesis is largely suppressed via malonyl-CoA; from Domain II.5"
    )

    # ── De Novo Cholesterol Synthesis ─────────────────────────────────────────
    # HMGCR: rate-limiting; statin target; feedback inhibited by oxysterols and PCSK9
    hmgcr_km_hmg_coa_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.1 mM; K_m^HMGCR for HMG-CoA; rate-limiting step in de novo cholesterol synthesis; from Domain II.5"
    )
    hepatic_cholesterol_synthesis_g_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 g/day; hepatic de novo cholesterol synthesis lower bound; from Domain II.5"
    )
    hepatic_cholesterol_synthesis_g_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 g/day; hepatic de novo cholesterol synthesis upper bound; from Domain II.5"
    )
