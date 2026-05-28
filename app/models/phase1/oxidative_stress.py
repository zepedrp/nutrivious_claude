from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesOxidativeStress(Base):
    __tablename__ = "species_oxidative_stress"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="oxidative_stress")

    # ── Mitochondrial Electron Leakage — Superoxide Production ───────────────
    electron_leak_fraction_o2_consumed_basal: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.001 — revised estimate ~0.1–0.2% of O₂ consumed leaks to O₂•⁻ at rest (older texts cite 1–2%; newer isotope tracing: 0.15%)"
    )
    electron_leak_fraction_o2_consumed_max_exercise: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.005 — ~0.5% electron leak fraction during maximal exercise (absolute flux rises ~20× despite lower % due to higher VO₂)"
    )
    mitochondrial_superoxide_production_basal_nmol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.3 nmol/min/mg protein — midpoint 0.1–0.5 nmol/min/mg; basal O₂•⁻ production isolated mitochondria (Complex I+III)"
    )
    mitochondrial_superoxide_production_max_nmol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.0 nmol/min/mg protein — upper O₂•⁻ production at maximal uncoupled respiration"
    )
    complex_i_ros_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.55 — ~55% of mitochondrial ROS originates at Complex I (site IQ + IF) under normal conditions"
    )
    complex_iii_ros_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.35 — ~35% of mitochondrial ROS from Complex III site IIIQo (outer ubiquinol site)"
    )
    delta_psi_ros_threshold_mv: Mapped[Optional[float]] = mapped_column(
        Float, comment="-140 mV — ΔΨm threshold above which ROS production rises exponentially; physiological ΔΨm ~-180 mV"
    )
    delta_psi_resting_mv: Mapped[Optional[float]] = mapped_column(
        Float, comment="-180 mV — typical resting mitochondrial membrane potential in coupled state"
    )
    ros_production_exercise_fold_increase: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.0 — midpoint 3–10× fold increase in total cellular ROS flux at VO₂max vs rest"
    )
    mitochondrial_h2o2_basal_nmol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 nmol/min/mg — midpoint 0.1–0.5; H₂O₂ efflux from isolated mitochondria at rest after SOD-mediated dismutation"
    )

    # ── Reactive Oxygen / Nitrogen Species — Half-Lives ───────────────────────
    superoxide_half_life_us: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µs — midpoint 1–10 µs; O₂•⁻ half-life in aqueous solution (pH 7.4, 37°C)"
    )
    hydrogen_peroxide_half_life_ms: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 ms — approximate cellular H₂O₂ t½ with active catalase/peroxiredoxin; can reach minutes without enzymes"
    )
    hydroxyl_radical_half_life_ns: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 ns — •OH half-life; most reactive ROS; reacts within ~2 nm of formation site"
    )
    singlet_oxygen_half_life_us: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 µs — ¹O₂ half-life in aqueous biological matrix (longer in lipid ~50 µs)"
    )
    nitric_oxide_half_life_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 s — midpoint 0.5–2 s; NO• half-life in tissue; limited by O₂ and superoxide quenching"
    )
    peroxynitrite_half_life_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.01 s — ONOO⁻ half-life at physiological pH; formed from NO• + O₂•⁻ (k=1.9×10¹⁰ M⁻¹s⁻¹)"
    )
    no_superoxide_reaction_rate_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.9e10 M⁻¹s⁻¹ — NO• + O₂•⁻ → ONOO⁻ rate constant; near-diffusion limit"
    )
    fenton_reaction_rate_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.6e7 M⁻¹s⁻¹ — Fenton: Fe²⁺ + H₂O₂ → Fe³⁺ + OH⁻ + •OH; generates most damaging ROS"
    )

    # ── Intracellular H₂O₂ Steady-State ──────────────────────────────────────
    h2o2_steady_state_cytosol_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 nM — midpoint 0.5–5 nM; cytosolic H₂O₂ steady-state concentration at rest"
    )
    h2o2_steady_state_mitochondria_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 nM — midpoint 10–100 nM; mitochondrial matrix H₂O₂ steady-state (higher than cytosol)"
    )
    h2o2_signalling_threshold_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 nM — threshold for redox-signalling activation (Nrf2, NF-κB, MAPK pathways)"
    )
    h2o2_cytotoxicity_threshold_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 µM — midpoint 10–100 µM; H₂O₂ concentration causing significant cell death"
    )
    h2o2_peroxiredoxin_floodgate_threshold_um_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM/s — H₂O₂ flux threshold for Prx hyperoxidation (Prx-SO₂H); floodgate model"
    )

    # ── Glutathione — Pool Concentrations ────────────────────────────────────
    gsh_cytosol_muscle_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="3 mM — midpoint 2–4 mM; intracellular [GSH] in resting skeletal muscle"
    )
    gsh_cytosol_liver_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="8 mM — midpoint 5–10 mM; hepatocyte cytosolic [GSH]; highest in body"
    )
    gsh_plasma_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 µM — midpoint 2–20 µM; plasma total GSH (mainly from hepatic export)"
    )
    gsh_erythrocyte_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 mM — midpoint 2–3 mM; erythrocyte [GSH]; primary RBC antioxidant"
    )
    gsh_total_body_g: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 g — midpoint 3–5 g; estimated total body GSH (muscle ~70%, liver ~15%)"
    )
    gsh_gssg_ratio_resting: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 — midpoint 100–400; [GSH]/[GSSG] ratio in healthy resting cells (highly reduced)"
    )
    gsh_gssg_ratio_moderate_exercise: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 — midpoint 20–100; GSH:GSSG ratio during moderate exercise (mild oxidative shift)"
    )
    gsh_gssg_ratio_max_exercise: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 — midpoint 10–20; GSH:GSSG ratio at VO₂max; significant but reversible"
    )
    gsh_gssg_ratio_oxidative_stress_threshold: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 — GSH:GSSG ratio below which frank oxidative stress is defined (clinical threshold)"
    )
    gsh_depletion_cytotoxicity_threshold_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — <25% residual GSH triggers mitochondrial permeability transition and apoptosis"
    )
    gssg_fraction_total_glutathione_resting: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.005 — ~0.5% of total glutathione pool is GSSG at rest in healthy cells"
    )

    # ── Glutathione Peroxidase (GPx) ─────────────────────────────────────────
    gpx1_km_h2o2_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM — midpoint 1–10 µM; GPx1 Km for H₂O₂ (selenoprotein; cytosolic)"
    )
    gpx1_km_gsh_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 mM — midpoint 0.2–1 mM; GPx1 Km for GSH co-substrate"
    )
    gpx1_vmax_nmol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 nmol/min/mg — midpoint 200–1000; GPx1 Vmax in human tissue"
    )
    gpx4_km_phospholipid_hydroperoxide_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 µM — GPx4 Km for phospholipid hydroperoxide (PLOOH); sole enzyme reducing membrane PUFA-OOH"
    )
    gpx4_activity_minimum_fraction_for_ferroptosis_threshold: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — <20% residual GPx4 activity triggers ferroptotic lipid peroxidation cascade"
    )
    gpx_selenium_requirement_ug_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="55 µg/day — RDA for selenium; required for selenocysteine in GPx1/4/3 active site"
    )

    # ── Glutathione Reductase (GR) ────────────────────────────────────────────
    gr_km_gssg_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="80 µM — midpoint 60–100 µM; glutathione reductase Km for GSSG substrate"
    )
    gr_km_nadph_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="7 µM — midpoint 5–10 µM; GR Km for NADPH co-substrate"
    )
    gr_vmax_nmol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 nmol/min/mg — GR Vmax in human erythrocytes; restores GSH pool"
    )
    gr_exercise_induction_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.4 — midpoint 1.2–1.6×; GR activity increase after chronic aerobic training"
    )

    # ── Glutamate-Cysteine Ligase (GCL) — GSH Synthesis ──────────────────────
    gcl_km_cysteine_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.3 mM — GCL Km for L-cysteine; rate-limiting substrate for GSH synthesis"
    )
    gcl_km_glutamate_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.8 mM — midpoint 1.6–2.0 mM; GCL Km for L-glutamate co-substrate"
    )
    gcl_km_atp_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 mM — GCL Km for ATP; reaction requires 1 ATP per γ-glutamylcysteine formed"
    )
    gcl_vmax_umol_per_g_liver_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.2 µmol/g liver/h — midpoint 0.5–2 µmol/g/h; GCL Vmax in hepatic tissue"
    )
    gsh_synthesis_rate_liver_umol_per_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 µmol/g/h — midpoint 5–15; total GSH synthesis rate in liver under demand"
    )
    gcl_feedback_inhibition_ki_gsh_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.3 mM — GCL product inhibition Ki for GSH (feedback control of own synthesis)"
    )

    # ── Superoxide Dismutase — SOD1 / SOD2 / SOD3 ────────────────────────────
    sod1_reaction_rate_constant_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.4e9 M⁻¹s⁻¹ — SOD1 (Cu/Zn-SOD) dismutation rate constant; near diffusion limit"
    )
    sod1_km_superoxide_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 µM — midpoint 10–100 µM; SOD1 apparent Km for O₂•⁻; low (high-affinity)"
    )
    sod1_kcat_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="2e6 per s — SOD1 kcat (turnover number); operates at diffusion-limited regime"
    )
    sod1_activity_muscle_u_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="350 U/mg — midpoint 200–500 U/mg; SOD1 activity in human skeletal muscle"
    )
    sod2_kcat_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.4e6 per s — SOD2 (Mn-SOD, mitochondrial) kcat; slightly lower than SOD1"
    )
    sod2_activity_mitochondria_u_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="600 U/mg — midpoint 400–1000 U/mg; SOD2 activity in mitochondrial fraction"
    )
    sod2_exercise_training_induction_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.35 — midpoint 1.2–1.5×; SOD2 protein induction after endurance training (PGC-1α)"
    )
    sod3_plasma_activity_u_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 U/mL — midpoint 1–5 U/mL; extracellular SOD3 activity in human plasma"
    )
    sod_total_liver_u_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="2000 U/mg — midpoint 1000–3000 U/mg; total SOD activity in human liver"
    )

    # ── Catalase ──────────────────────────────────────────────────────────────
    catalase_kcat_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="4e7 per s — catalase kcat; one of the highest known; ~40 million H₂O₂/molecule/s"
    )
    catalase_km_h2o2_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="60 mM — midpoint 25–100 mM; catalase Km for H₂O₂ (never saturated physiologically)"
    )
    catalase_vmax_umol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="17000 µmol/min/mg — catalase Vmax at saturating H₂O₂; highest enzymatic rate known"
    )
    catalase_activity_erythrocyte_ku_per_g_hb: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 kU/g Hb — midpoint 150–250 kU/g Hb; catalase activity in human erythrocytes"
    )
    catalase_activity_liver_u_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="350 U/mg — midpoint 300–400 U/mg; catalase activity in human liver"
    )
    catalase_activity_muscle_u_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 U/mg — catalase activity in skeletal muscle (low; muscle relies on GPx/Prx)"
    )
    catalase_exercise_induction_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.2 — midpoint 1.1–1.3×; catalase induction after chronic aerobic training"
    )

    # ── Peroxiredoxin / Thioredoxin System ───────────────────────────────────
    prx_km_h2o2_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM — midpoint 1–50 µM; typical 2-Cys Prx Km for H₂O₂ (lower than catalase)"
    )
    prx_kcat_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="2e7 M⁻¹s⁻¹ — Prx second-order rate constant for H₂O₂; major cytosolic scavenger"
    )
    prx_fraction_h2o2_scavenged_cytosol: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.90 — ~90% of cytosolic H₂O₂ scavenged by Prx1/2 under physiological flux"
    )
    prx_hyperoxidation_inactivation_threshold_um_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM/s — H₂O₂ flux causing Prx-SO₂H hyperoxidation (floodgate); allows H₂O₂ signalling"
    )
    prx_reactivation_by_sulfiredoxin_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 h — midpoint 1–4 h; time for sulfiredoxin (Srxn1) to reduce Prx-SO₂H back to active form"
    )
    trx1_km_nadph_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM — thioredoxin reductase 1 (TrxR1) Km for NADPH; cytosolic Trx recycling"
    )
    trxr1_vmax_nmol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="150 nmol/min/mg — midpoint 100–200 nmol/min/mg; TrxR1 Vmax in human cells"
    )
    trx_total_cellular_concentration_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 µM — midpoint 5–15 µM; total thioredoxin concentration in human cells"
    )

    # ── Nrf2 / Antioxidant Response Element (ARE) ────────────────────────────
    nrf2_keap1_kd_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 nM — Keap1 binding affinity for Nrf2 ETGE motif under reducing conditions"
    )
    nrf2_nuclear_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 min — midpoint 20–30 min; Nrf2 t½ in nucleus under induction conditions"
    )
    nrf2_activation_threshold_h2o2_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 µM — threshold [H₂O₂] for significant Nrf2 nuclear translocation"
    )
    nrf2_target_genes_count: Mapped[Optional[int]] = mapped_column(
        Integer, comment="200 — ~200 ARE-driven genes activated by Nrf2 (NQO1, HO-1, GCL, GPx, SOD, Trx)"
    )
    nrf2_ho1_induction_fold_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.0 — lower fold induction of HO-1 (heme oxygenase-1) by Nrf2 activation"
    )
    nrf2_ho1_induction_fold_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="20.0 — upper fold induction of HO-1; most sensitive Nrf2 target gene"
    )
    nrf2_exercise_activation_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 — midpoint 1.5–3×; Nrf2 nuclear accumulation fold after acute exercise"
    )
    nrf2_training_chronic_antioxidant_increase_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 — midpoint 1.3–2×; sustained increase in Nrf2 target protein levels after training"
    )

    # ── NADPH Oxidase (NOX) — Immune / Vascular ROS ─────────────────────────
    nox2_superoxide_production_neutrophil_nmol_per_min_per_10e6_cells: Mapped[Optional[float]] = mapped_column(
        Float, comment="1500 nmol/min/10⁶ cells — peak O₂•⁻ during neutrophil respiratory burst"
    )
    nox2_respiratory_burst_duration_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 min — midpoint 5–20 min; duration of peak NOX2 respiratory burst in phagocytes"
    )
    nox4_basal_h2o2_production_vascular_nmol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 nmol/min/mg — NOX4 constitutive H₂O₂ production in vascular smooth muscle (tonic ROS)"
    )
    nox2_km_o2_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 µM — NOX2 Km for O₂; operational even at low pO₂ in tissue"
    )
    nox2_km_nadph_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 µM — NOX2 Km for NADPH; cytosolic NADPH pool controls burst amplitude"
    )

    # ── Lipid Peroxidation ────────────────────────────────────────────────────
    lipid_peroxidation_initiation_rate_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1e7 M⁻¹s⁻¹ — rate constant for •OH abstraction of bis-allylic H from PUFA (initiation)"
    )
    lipid_peroxidation_propagation_rate_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="65 M⁻¹s⁻¹ — midpoint 60–70 M⁻¹s⁻¹; lipid peroxyl radical propagation constant in PUFA"
    )
    alpha_tocopherol_chain_break_rate_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.5e6 M⁻¹s⁻¹ — midpoint 2–5×10⁶ M⁻¹s⁻¹; α-tocopherol chain-breaking rate constant"
    )
    membrane_pufa_fraction_critical_threshold: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — >20% membrane PUFA content significantly raises lipid peroxidation susceptibility"
    )
    mda_plasma_basal_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.25 µM — midpoint 0.5–2.0 µM; plasma MDA (malondialdehyde) at rest (TBARS assay)"
    )
    mda_plasma_post_exercise_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM — midpoint 3–8 µM; plasma MDA peak after exhaustive exercise"
    )
    hne_cytotoxicity_threshold_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 µM — midpoint 10–50 µM; 4-hydroxynonenal (4-HNE) cytotoxic threshold; adducts proteins"
    )
    hne_signalling_threshold_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 µM — midpoint 0.1–5 µM; 4-HNE concentration activating Nrf2 and stress kinases"
    )
    isoprostane_f2_urine_basal_ng_per_mg_creatinine: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 ng/mg creatinine — midpoint 0.5–2.5; basal urinary F2-isoprostane (gold standard lipid peroxidation marker)"
    )
    isoprostane_exercise_increase_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.0 — midpoint 3–10×; urinary F2-isoprostane fold increase after ultramarathon/VO₂max test"
    )
    tbars_plasma_basal_nmol_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 nmol MDA/mL — midpoint 1–3 nmol/mL plasma TBARS at rest"
    )
    vitamin_e_membrane_concentration_nmol_per_g_lipid: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 nmol/g lipid — midpoint 30–100; membrane α-tocopherol concentration; 1 per ~1000 PUFA"
    )

    # ── DNA Oxidative Damage ──────────────────────────────────────────────────
    dna_strand_breaks_per_cell_basal: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 per cell — midpoint 10–30 SSB + DSB per cell per day (comet assay; basal endogenous damage)"
    )
    dna_strand_breaks_exercise_fold_increase: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 — midpoint 1.5–4×; DNA strand breaks fold increase in leukocytes post-exhaustive exercise"
    )
    oxidative_dna_lesions_per_cell_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="10000 — ~10,000 oxidative base lesions generated per cell per day (endogenous O₂ flux)"
    )
    ohhdg_urine_basal_ng_per_mg_creatinine: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 ng/mg creatinine — midpoint 1–10; urinary 8-OHdG at rest (8-hydroxy-2'-deoxyguanosine)"
    )
    ohhdg_exercise_increase_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 — midpoint 2–3×; urinary 8-OHdG fold increase post-marathon/exhaustive exercise"
    )
    base_excision_repair_rate_lesions_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 lesion/s per cell — approximate OGG1-initiated BER capacity for 8-oxoG removal"
    )
    nucleotide_excision_repair_rate_lesions_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="1000 per h per cell — NER capacity for bulky adducts/UV lesions in human cells"
    )

    # ── Antioxidant Vitamins — Plasma Reference Levels ───────────────────────
    vitamin_c_plasma_optimal_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 µM — midpoint 40–80 µM; optimal plasma ascorbate concentration (saturation ~70 µM)"
    )
    vitamin_c_plasma_deficiency_threshold_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="11 µM — WHO deficiency threshold for plasma ascorbate"
    )
    vitamin_c_rate_constant_superoxide_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.7e5 M⁻¹s⁻¹ — ascorbate rate constant with O₂•⁻; recycles tocopheroxyl radical"
    )
    vitamin_e_plasma_optimal_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 µM — midpoint 20–40 µM; optimal plasma α-tocopherol concentration"
    )
    uric_acid_plasma_upper_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="360 µM (female) / 420 µM (male) — upper normal plasma urate; major aqueous antioxidant (~60% plasma AOC)"
    )
    uric_acid_rate_constant_hydroxyl_m_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.2e9 M⁻¹s⁻¹ — uric acid rate constant with •OH; near-diffusion-limited scavenging"
    )
    total_antioxidant_capacity_plasma_mm_trolox: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 mM Trolox-eq — midpoint 1.2–2.0; plasma total antioxidant capacity (ORAC/FRAP assay)"
    )
