from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesEpigenetic(Base):
    __tablename__ = "species_epigenetic"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="epigenetic")

    # ── RNA Polymerase II — mRNA Transcription ────────────────────────────────
    rnapol2_elongation_rate_lower_nt_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 nt/s — lower RNA Pol II elongation rate during active transcription"
    )
    rnapol2_elongation_rate_upper_nt_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="80 nt/s — upper RNA Pol II elongation rate; gene/chromatin context dependent"
    )
    rnapol2_pausing_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — ~60% of RNA Pol II is paused 25–50 nt downstream of TSS in human cells"
    )
    rnapol2_initiation_rate_lower_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 per h — lower transcription initiation rate per active gene promoter"
    )
    rnapol2_initiation_rate_upper_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 per h — upper initiation rate at highly active promoters (burst firing)"
    )
    transcription_burst_size_lower_mrna: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 — lower mRNA molecules per transcriptional burst (binary on/off)"
    )
    transcription_burst_size_upper_mrna: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 — upper mRNA molecules per burst (highly expressed genes)"
    )
    transcription_burst_frequency_lower_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.1 per h — lower burst frequency; rarely active genes"
    )
    transcription_burst_frequency_upper_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 per h — upper burst frequency; constitutively active housekeeping genes"
    )

    # ── RNA Polymerase I/III — rRNA / tRNA ────────────────────────────────────
    rnapol1_elongation_rate_nt_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 nt/s — midpoint 40–100 nt/s; RNA Pol I rate for 47S pre-rRNA synthesis"
    )
    rnapol1_copies_per_nucleolus: Mapped[Optional[float]] = mapped_column(
        Float, comment="150 — midpoint 100–200 Pol I molecules active per nucleolus at max rDNA"
    )
    rrna_genes_total_copies: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 — ~200–500 rRNA gene copies across 5 acrocentric chromosomes (13,14,15,21,22)"
    )
    rrna_genes_active_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.50 — ~50% of rDNA copies transcriptionally active; remainder silenced by CpG methylation"
    )
    rnapol3_trna_transcription_rate_nt_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 nt/s — RNA Pol III elongation rate for tRNA genes (~75 nt per tRNA)"
    )

    # ── mRNA Half-Life ────────────────────────────────────────────────────────
    mrna_half_life_median_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="600 min (~10 h) — median mRNA half-life across human transcriptome"
    )
    mrna_half_life_unstable_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 min — half-life of labile mRNAs (e.g., c-Fos, c-Myc; ARE-mediated decay)"
    )
    mrna_half_life_stable_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="120 h — half-life of highly stable mRNAs (e.g., beta-globin; erythrocyte life)"
    )
    mrna_pool_whole_body_g: Mapped[Optional[float]] = mapped_column(
        Float, comment="~1 g — estimated total mRNA mass in human body (~3 pg/cell × 3.7×10¹³ cells)"
    )
    pre_mrna_splicing_rate_nt_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 nt/s — approximate intron removal rate by spliceosome machinery"
    )
    active_genes_per_cell: Mapped[Optional[float]] = mapped_column(
        Float, comment="8000 — typical number of genes actively transcribed per differentiated cell type"
    )

    # ── Protein Synthesis — Ribosome / Translation ────────────────────────────
    ribosome_elongation_rate_aa_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 aa/s — midpoint 3–6 aa/s; human ribosome peptide elongation rate at 37°C"
    )
    ribosome_initiation_rate_per_mrna_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 per min — average 80S ribosome initiation rate per mRNA; ~1–10 per min"
    )
    polysome_max_ribosomes_per_mrna: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 — maximum ribosomes per mRNA in polysome (1 per ~100 nt, average mRNA ~3000 nt)"
    )
    whole_body_protein_synthesis_g_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="280 g/day — midpoint 250–300 g; total whole-body protein synthesised daily"
    )
    muscle_protein_synthesis_fraction_per_day_basal: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.015 — 1.5%/day basal myofibrillar protein synthesis rate (FSR mixed muscle)"
    )
    muscle_protein_synthesis_fraction_per_day_exercise: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.03 — 3%/day myofibrillar FSR post resistance exercise (24–48 h elevation)"
    )
    liver_protein_synthesis_fraction_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.50 — ~50%/day liver protein synthesis rate (high turnover secretory proteins)"
    )
    albumin_half_life_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 days — albumin plasma half-life; main index of hepatic synthetic capacity"
    )
    fibrinogen_half_life_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 days — fibrinogen plasma half-life; acute-phase reactant"
    )
    mtor_protein_synthesis_induction_fold_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 — lower fold stimulation of MPS by maximal mTORC1 activation"
    )
    mtor_protein_synthesis_induction_fold_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.0 — upper fold MPS stimulation; leucine + insulin + mechanical load"
    )
    leucine_mps_threshold_g_per_meal: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 g/meal — midpoint 2–3 g leucine to maximally activate mTORC1/MPS"
    )

    # ── Protein Degradation — UPS / Autophagy ────────────────────────────────
    ubiquitin_proteasome_fraction_degradation: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.75 — ~75% of intracellular protein degradation via 26S UPS pathway"
    )
    autophagy_lysosomal_fraction_degradation: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — ~25% degradation via lysosomal autophagy (macro + micro + CMA)"
    )
    proteasome_substrate_processing_rate_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 per min — ~5 ubiquitinated substrate molecules degraded per 26S proteasome per min"
    )
    ubiquitin_chain_min_length_for_degradation: Mapped[Optional[int]] = mapped_column(
        Integer, comment="4 — minimum K48-linked polyubiquitin chain length recognised by 26S proteasome"
    )
    autophagy_flux_basal_fraction_protein_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.015 — ~1.5%/h basal autophagic flux in liver (fraction of protein mass)"
    )
    autophagy_flux_fasting_fold_increase: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.0 — midpoint 2–4× fold increase in autophagy flux during 24 h fasting"
    )
    protein_half_life_short_lived_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 min — half-life of short-lived regulatory proteins (e.g., p53, cyclins)"
    )
    protein_half_life_long_lived_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="14 days — median half-life of long-lived structural proteins (e.g., collagen)"
    )
    protein_half_life_lens_crystallin_years: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 years — t½ of human lens crystallins; synthesised embryonically, not turned over"
    )

    # ── Telomere Biology — Length, Shortening, Hayflick ──────────────────────
    telomere_length_birth_bp: Mapped[Optional[float]] = mapped_column(
        Float, comment="12000 bp — midpoint 10,000–15,000 bp; leukocyte telomere length at birth (TRF)"
    )
    telomere_length_adult_lower_bp: Mapped[Optional[float]] = mapped_column(
        Float, comment="5000 bp — lower leukocyte telomere length in healthy adults"
    )
    telomere_length_adult_upper_bp: Mapped[Optional[float]] = mapped_column(
        Float, comment="10000 bp — upper leukocyte telomere length in healthy adults"
    )
    telomere_shortening_rate_bp_per_division: Mapped[Optional[float]] = mapped_column(
        Float, comment="37 bp/division — midpoint 25–50 bp lost per cell division (end-replication problem)"
    )
    telomere_shortening_rate_bp_per_year_adult: Mapped[Optional[float]] = mapped_column(
        Float, comment="60 bp/year — midpoint 50–100 bp/year shortening in adult somatic tissues"
    )
    telomere_critical_length_bp: Mapped[Optional[float]] = mapped_column(
        Float, comment="3000 bp — midpoint 2,000–4,000 bp; critical length triggering p53/p21 senescence"
    )
    hayflick_limit_divisions_fibroblast_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 divisions — lower Hayflick limit in WI-38 human fibroblasts (Hayflick & Moorhead 1961)"
    )
    hayflick_limit_divisions_fibroblast_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 divisions — upper Hayflick limit; replicative senescence in vitro"
    )
    hayflick_limit_divisions_lymphocyte: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 divisions — estimated in vivo replicative limit for peripheral T-lymphocytes"
    )
    telomerase_activity_stem_cells_fold_vs_somatic: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 — ~100× higher telomerase (hTERT) activity in stem/germline vs somatic cells"
    )
    telomerase_extension_rate_nt_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 nt/s — midpoint 40–60 nt/s; hTERT reverse transcriptase extension rate"
    )
    telomere_repeat_unit_bp: Mapped[Optional[int]] = mapped_column(
        Integer, comment="6 — TTAGGG hexanucleotide repeat unit; added by telomerase"
    )
    oxidative_stress_telomere_shortening_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.5 — midpoint 5–10× acceleration of telomere attrition under oxidative stress"
    )
    shelterin_trf2_kd_telomere_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 nM — TRF2 binding affinity for double-stranded TTAGGG repeats; shelterin core"
    )
    pot1_kd_ssdna_telomere_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 nM — POT1 binding affinity for 3' G-overhang ssDNA (TTAGGG); prevents ATR"
    )

    # ── DNA Methylation — DNMT Kinetics ──────────────────────────────────────
    cpg_sites_total_genome: Mapped[Optional[float]] = mapped_column(
        Float, comment="28e6 — ~28 million CpG dinucleotides in haploid human genome"
    )
    cpg_methylation_fraction_global: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.045 — ~4.5% of cytosines are 5-methylcytosine (5mC) globally"
    )
    cpg_island_fraction_methylated: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.05 — ~5% of CpG islands are methylated in normal somatic cells"
    )
    dnmt1_km_hemi_methylated_dna_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.3 µM — midpoint 0.1–0.5 µM; DNMT1 Km for hemi-methylated substrate (maintenance)"
    )
    dnmt1_vmax_cpg_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 CpG/s — midpoint 1–10 CpG/s; DNMT1 maintenance methylation velocity at Vmax"
    )
    dnmt3a_km_sam_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 µM — midpoint 5–15 µM; DNMT3A Km for S-adenosylmethionine (methyl donor)"
    )
    dnmt3b_vmax_cpg_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 CpG/s — DNMT3B de novo methylation velocity; lower than DNMT1 maintenance"
    )
    sam_intracellular_concentration_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="80 µM — midpoint 50–150 µM; intracellular S-adenosylmethionine in liver"
    )
    methylation_clock_rate_cpg_fraction_per_decade: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.005 — ~0.5% change in CpG methylation per decade post age 40 (Horvath clock)"
    )
    tet_enzyme_km_5mc_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 µM — approximate TET1/2 Km for 5-methylcytosine oxidation (→5hmC demethylation)"
    )
    active_demethylation_5hmc_intermediate_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.006 — ~0.6% of cytosines are 5-hydroxymethylcytosine (5hmC) in brain/ES cells"
    )
    exercise_dnmt_expression_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.30 — ~30% reduction in DNMT3A/3B expression after acute aerobic exercise bout"
    )
    exercise_pgc1a_promoter_demethylation_cpg_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.15 — ~15% reduction in CpG methylation at PGC-1α promoter after endurance exercise"
    )

    # ── Histone Modification — HAT / HDAC Kinetics ───────────────────────────
    hat_p300_km_acetyl_coa_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 µM — midpoint 5–20 µM; p300/CBP HAT Km for acetyl-CoA co-substrate"
    )
    hat_p300_vmax_pmol_per_min_per_ug: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.5 pmol/min/µg — midpoint 5–10 pmol/min/µg; p300 acetyltransferase Vmax"
    )
    hdac1_km_acetyl_histone_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="1000 µM — midpoint 0.2–2 mM; HDAC1 Km for acetylated histone substrate"
    )
    hdac_class_i_vmax_pmol_per_min_per_ug: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 pmol/min/µg — approximate class I HDAC (1/2/3/8) deacetylation Vmax"
    )
    sirt1_km_nad_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 µM — midpoint 50–200 µM; SIRT1 (class III HDAC) Km for NAD⁺ co-substrate"
    )
    sirt1_km_acetyl_substrate_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µM — midpoint 2–10 µM; SIRT1 Km for acetylated histone/p53 substrate"
    )
    h3k4me3_active_promoter_fraction_histones: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.015 — ~1.5% of H3 histones carry K4me3; marks active transcription start sites"
    )
    h3k27me3_repressed_fraction_histones: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.08 — ~8% of H3 histones carry K27me3; Polycomb repressive complex mark"
    )
    h3k9me3_heterochromatin_fraction_histones: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.12 — ~12% of H3 carry K9me3; constitutive heterochromatin/pericentromeric"
    )
    histone_h3_turnover_active_promoter_t_half_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 h — H3 histone t½ at active promoters (rapid exchange near TSS)"
    )
    histone_h3_turnover_heterochromatin_t_half_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 days — H3 t½ in heterochromatin (slow exchange; epigenetic memory)"
    )
    exercise_hat_activity_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.75 — ~75% increase in muscle HAT activity (p300/CBP) post acute exercise"
    )

    # ── Chromatin Structure — Nucleosome / Compaction ─────────────────────────
    nucleosome_dna_wraps_bp: Mapped[Optional[int]] = mapped_column(
        Integer, comment="147 — base pairs of DNA wrapped 1.65 turns around histone octamer core"
    )
    nucleosome_repeat_length_bp: Mapped[Optional[float]] = mapped_column(
        Float, comment="192 bp — midpoint 185–200 bp; nucleosome repeat length (147 core + ~45 linker)"
    )
    nucleosome_density_per_kb: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.2 per kb — average nucleosome density in human genome (~1 per 192 bp)"
    )
    chromatin_compaction_nucleosome_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.5 — midpoint 6–7×; linear compaction at nucleosome level vs naked DNA"
    )
    chromatin_compaction_30nm_fiber_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="40 — ~40× total compaction at 30 nm chromatin fibre level"
    )
    chromatin_compaction_loop_domain_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="1000 — ~1000× compaction at loop domain level (cohesin/CTCF-anchored)"
    )
    chromatin_total_compaction_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="10000 — ~10,000× total DNA compaction in metaphase chromosome"
    )
    tad_size_lower_kb: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 kb — lower bound Topologically Associating Domain (TAD) size"
    )
    tad_size_upper_mb: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 Mb — upper bound TAD size; self-interacting chromatin domains by Hi-C"
    )
    loop_size_lower_kb: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 kb — lower chromatin loop size (cohesin-mediated; CTCF anchor points)"
    )
    loop_size_upper_kb: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 kb — upper chromatin loop size; forms gene regulatory hubs"
    )
    open_chromatin_fraction_genome: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.03 — ~3% of genome is in open/accessible chromatin (ATAC-seq peaks) in any cell type"
    )

    # ── Non-Coding RNA ────────────────────────────────────────────────────────
    mirna_total_species_human: Mapped[Optional[int]] = mapped_column(
        Integer, comment="2600 — ~2,600 mature miRNA species annotated in human miRBase (v22)"
    )
    mirna_half_life_lower_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="24 h — lower miRNA half-life; most miRNAs are stable (t½ 1–5 days)"
    )
    mirna_half_life_upper_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 days — upper miRNA half-life; exceptional stability relative to mRNA"
    )
    mirna_targets_per_species: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 — midpoint estimate of mRNA targets per miRNA (seed-match algorithms)"
    )
    ago2_risc_km_mirna_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 nM — approximate AGO2-RISC Kd for mature miRNA loading"
    )
    lncrna_total_species_human: Mapped[Optional[int]] = mapped_column(
        Integer, comment="16000 — ~16,000 annotated long non-coding RNA genes in human genome (GENCODE)"
    )
    circrna_half_life_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="48 h — median circular RNA (circRNA) half-life; ~10× more stable than linear RNA"
    )

    # ── Genome-Wide Constants ─────────────────────────────────────────────────
    human_genome_size_bp: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.2e9 — ~3.2 × 10⁹ bp haploid human genome (GRCh38)"
    )
    protein_coding_genes: Mapped[Optional[int]] = mapped_column(
        Integer, comment="20000 — ~20,000 protein-coding genes (GENCODE v42; range 19,000–21,000)"
    )
    gene_density_per_mb: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.5 genes/Mb — average protein-coding gene density (20,000 genes / 3,100 Mb)"
    )
    exon_fraction_genome: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.015 — ~1.5% of haploid genome is protein-coding exonic sequence"
    )
    intron_fraction_genome: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — ~25% of genome is intronic sequence"
    )
    repetitive_element_fraction_genome: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.45 — ~45% of human genome is repetitive elements (SINEs, LINEs, LTRs)"
    )
    dna_replication_fork_speed_kb_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 kb/min — midpoint 1–2 kb/min; human replication fork velocity"
    )
    replication_origins_total: Mapped[Optional[int]] = mapped_column(
        Integer, comment="30000 — ~30,000–50,000 potential replication origins; ~10,000–30,000 fired per S phase"
    )
    s_phase_duration_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="8 h — midpoint 6–10 h; S-phase duration in typical dividing human cells"
    )
    dna_polymerase_delta_fidelity_error_rate: Mapped[Optional[float]] = mapped_column(
        Float, comment="1e-7 — intrinsic DNA Pol δ error rate before mismatch repair (~10⁻⁷/bp/division)"
    )
    post_replication_mismatch_repair_fidelity: Mapped[Optional[float]] = mapped_column(
        Float, comment="1e-10 — post-MMR net error rate (~10⁻¹⁰/bp/division; ~0.64 mutations per cell division)"
    )
    spontaneous_mutation_rate_per_cell_division: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.64 — ~0.64 de novo mutations per cell division in human somatic cells (Kong et al.)"
    )
