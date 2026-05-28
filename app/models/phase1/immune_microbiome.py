from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesImmuneMicrobiome(Base):
    __tablename__ = "species_immune_microbiome"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="immune_microbiome")

    # ── Intestinal Epithelial Barrier — Permeability Baseline ─────────────────
    teer_small_intestine_ohm_cm2_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 Ω·cm² — lower TEER (transepithelial electrical resistance) healthy small intestine"
    )
    teer_small_intestine_ohm_cm2_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 Ω·cm² — upper TEER healthy human small intestine; integrity index"
    )
    teer_colon_ohm_cm2_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 Ω·cm² — lower TEER healthy colon epithelium"
    )
    teer_colon_ohm_cm2_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="600 Ω·cm² — upper TEER healthy colon; higher than small intestine"
    )
    teer_impaired_threshold_ohm_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 Ω·cm² — TEER below which barrier impairment is functionally significant (Caco-2 model)"
    )
    lactulose_mannitol_ratio_normal_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.03 — upper normal lactulose/mannitol (L/M) urinary ratio; gold standard intestinal permeability"
    )
    lactulose_mannitol_ratio_leaky_gut_threshold: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.10 — L/M ratio ≥0.10 defines clinically significant intestinal hyperpermeability"
    )
    lactulose_mannitol_ratio_exercise_70pct_vo2max: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.07 — midpoint; L/M ratio during exercise ≥70% VO₂max (2–3× basal increase)"
    )
    lactulose_mannitol_ratio_heat_stress_fold_increase: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.0 — midpoint 3–5×; L/M ratio fold increase during combined exercise + heat stress"
    )

    # ── Tight Junction Proteins — Zonulin / Occludin / Claudin ───────────────
    zonulin_serum_basal_ng_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 ng/mL — midpoint 30–80 ng/mL; serum zonulin at rest in healthy adults"
    )
    zonulin_leaky_gut_threshold_ng_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 ng/mL — serum zonulin above which intestinal hyperpermeability is diagnosed"
    )
    zonulin_exercise_heat_increase_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 — midpoint 2–3×; serum zonulin fold increase after ultramarathon in heat"
    )
    occludin_expression_reduction_exercise_heat_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — ~60% midpoint 50–70% reduction in occludin protein after endurance exercise in heat"
    )
    claudin_3_expression_reduction_exercise_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.40 — ~40% reduction in claudin-3 after 2 h exercise at 60% VO₂max in heat"
    )
    zo1_phosphorylation_increase_tnf_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.0 — midpoint 2–4×; ZO-1 phosphorylation fold increase by TNF-α (1 ng/mL); opens TJ"
    )
    mlck_activity_increase_permeability_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 — MLCK (myosin light chain kinase) activity fold increase causing TJ disassembly"
    )
    lps_plasma_basal_eu_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 EU/mL — midpoint 0.1–1.0 EU/mL; fasting plasma LPS (endotoxin units) in healthy adults"
    )
    lps_metabolic_endotoxemia_threshold_eu_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 EU/mL — plasma LPS threshold defining metabolic endotoxemia (chronic low-grade)"
    )
    lps_sepsis_threshold_eu_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 EU/mL — plasma LPS level associated with frank sepsis and systemic shock"
    )
    lps_post_exercise_increase_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 — midpoint 1.5–3×; plasma LPS fold rise immediately after exhaustive exercise"
    )

    # ── Mucus Layer / Goblet Cells ────────────────────────────────────────────
    mucus_outer_layer_thickness_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="125 µM — midpoint 100–150 µm; outer loose mucus layer (colonised by bacteria)"
    )
    mucus_inner_layer_thickness_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="75 µM — midpoint 50–100 µm; inner firmly attached sterile mucus layer"
    )
    goblet_cell_fraction_epithelium: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.12 — midpoint 10–15%; goblet cell fraction of intestinal epithelium"
    )
    muc2_secretion_rate_per_goblet_pg_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 pg/h/cell — approximate MUC2 mucin secretion rate per goblet cell"
    )
    enterocyte_turnover_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 days — midpoint 3–5 days; intestinal epithelial cell turnover (crypt-to-villus migration)"
    )
    crypt_cell_production_per_day_per_crypt: Mapped[Optional[int]] = mapped_column(
        Integer, comment="1400 per day — cells produced per intestinal crypt; Lgr5+ stem cell driven"
    )
    intestinal_alkaline_phosphatase_activity_u_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 U/mg — approximate IAP activity in intestinal brush border; detoxifies LPS lipid A"
    )

    # ── Gut Microbiome — Composition and Diversity ────────────────────────────
    total_gut_bacteria_count: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.8e13 — ~3.8×10¹³ bacteria in human GI tract (Sender et al. 2016; 1:1 ratio with human cells)"
    )
    microbial_gene_count: Mapped[Optional[int]] = mapped_column(
        Integer, comment="3300000 — ~3.3 million microbial genes in human gut microbiome vs ~20,000 human genes"
    )
    microbiome_species_diversity_per_individual: Mapped[Optional[int]] = mapped_column(
        Integer, comment="160 — midpoint 100–200 species per individual (MetaHIT; operational taxonomic units)"
    )
    shannon_diversity_index_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.0 — midpoint 3–5; Shannon entropy index H' for healthy adult gut microbiome"
    )
    firmicutes_fraction_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.70 — midpoint 60–80%; Firmicutes phylum relative abundance in healthy adults"
    )
    bacteroidetes_fraction_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — midpoint 20–30%; Bacteroidetes phylum relative abundance healthy adults"
    )
    firmicutes_bacteroidetes_ratio_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 — midpoint 1–2; F/B ratio in healthy adult; higher in obesity (>3); lower in IBD (<0.5)"
    )
    actinobacteria_fraction_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.06 — midpoint 3–10%; Actinobacteria fraction (mainly Bifidobacterium)"
    )
    proteobacteria_fraction_healthy_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.03 — upper normal Proteobacteria fraction; >10% signals dysbiosis"
    )
    akkermansia_muciniphila_fraction_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.03 — midpoint 0.1–5%; Akkermansia muciniphila abundance; inversely correlated with metabolic disease"
    )
    lactobacillus_fraction_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.01 — midpoint 0.01–1%; Lactobacillus fraction (highly variable; diet-dependent)"
    )
    bifidobacterium_fraction_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.05 — midpoint 0.01–10%; Bifidobacterium fraction; declines with age and antibiotic use"
    )
    microbiome_weight_wet_g: Mapped[Optional[float]] = mapped_column(
        Float, comment="850 g — midpoint 200–1500 g wet weight; total gut microbiome biomass"
    )
    antibiotic_diversity_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.70 — ~70% midpoint 50–90% reduction in microbiome diversity after broad-spectrum antibiotics"
    )
    antibiome_recovery_weeks: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 weeks — midpoint 4–6 weeks for partial microbiome recovery post-antibiotics (full: months–years)"
    )
    exercise_microbiome_diversity_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — ~20% increase in alpha diversity with chronic aerobic exercise training"
    )

    # ── Short-Chain Fatty Acids (SCFAs) ──────────────────────────────────────
    scfa_total_production_mmol_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="350 mmol/day — midpoint 300–400 mmol/day; total colonic SCFA production"
    )
    acetate_colonic_concentration_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="67 mM — midpoint 60–75 mM; proximal colonic acetate concentration; absorbed + systemic"
    )
    propionate_colonic_concentration_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 mM — midpoint 10–20 mM; propionate; gluconeogenic substrate for liver"
    )
    butyrate_colonic_concentration_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 mM — midpoint 5–15 mM; butyrate; primary colonocyte fuel (~70% of energy)"
    )
    butyrate_colonocyte_energy_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.70 — ~70% of colonocyte ATP derived from butyrate β-oxidation"
    )
    butyrate_hdac_inhibition_ki_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 mM — midpoint 0.5–2 mM; butyrate Ki for HDAC inhibition (epigenetic regulation)"
    )
    scfa_acetate_fraction_total: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — ~60% of total SCFA pool is acetate (dominant product of fermentation)"
    )
    colon_ph_proximal: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.0 — midpoint 5.5–6.5; proximal colon pH; lower due to SCFA accumulation"
    )
    colon_ph_distal: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.0 — midpoint 6.5–7.5; distal colon pH; higher as SCFAs are absorbed"
    )

    # ── Secretory IgA (sIgA) ─────────────────────────────────────────────────
    siga_total_daily_production_g: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.5 g/day — midpoint 2–5 g/day; total sIgA production; most abundantly secreted antibody class"
    )
    siga_serum_iga_concentration_g_per_l_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.7 g/L — lower serum IgA normal range"
    )
    siga_serum_iga_concentration_g_per_l_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.0 g/L — upper serum IgA normal range"
    )
    siga_salivary_concentration_ug_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 µg/mL — lower salivary sIgA concentration in healthy adults"
    )
    siga_salivary_concentration_ug_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 µg/mL — upper salivary sIgA; reflects mucosal immune status"
    )
    siga_salivary_daily_secretion_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 mg/day — midpoint 200–400 mg; daily salivary sIgA output"
    )
    siga_gut_fraction_total: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — ~60% midpoint 50–70% of total sIgA secreted into gut lumen"
    )
    siga_half_life_serum_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="6 days — midpoint 5–7 days; serum IgA t½"
    )
    siga_half_life_gut_lumen_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 h — midpoint 2–6 h; sIgA t½ in gut lumen (protected by secretory component)"
    )
    siga_lamina_propria_plasma_cells_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.80 — ~80% of lamina propria plasma cells produce IgA (dominant mucosal isotype)"
    )
    siga_overtraining_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.55 — ~55% midpoint 50–60% reduction in salivary sIgA with overtraining syndrome"
    )
    siga_moderate_exercise_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — ~25% increase in salivary sIgA secretion rate with regular moderate exercise"
    )

    # ── Innate Immunity — Cellular Counts and Kinetics ────────────────────────
    neutrophil_blood_count_lower_per_ul: Mapped[Optional[float]] = mapped_column(
        Float, comment="2500 per µL — lower neutrophil count in healthy adults (50–70% of leukocytes)"
    )
    neutrophil_blood_count_upper_per_ul: Mapped[Optional[float]] = mapped_column(
        Float, comment="7500 per µL — upper normal neutrophil count"
    )
    neutrophil_half_life_blood_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="7 h — midpoint 6–8 h; circulating neutrophil t½ before margination/tissue migration"
    )
    neutrophil_half_life_tissue_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="36 h — midpoint 24–48 h; tissue neutrophil lifespan (extended by IL-8, GM-CSF)"
    )
    neutrophil_phagocytosis_bacteria_per_cell: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 — midpoint 10–20 bacteria per neutrophil before exhaustion"
    )
    nk_cell_fraction_lymphocytes: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.08 — midpoint 2–13%; NK cell fraction of peripheral blood lymphocytes (CD16+CD56+)"
    )
    nk_cell_cytotoxicity_reduction_post_marathon_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — ~60% midpoint 50–70% reduction in NK cell cytotoxic activity post-marathon"
    )
    complement_c3_plasma_g_per_l: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 g/L — midpoint 0.9–1.8 g/L; plasma C3 concentration; most abundant complement protein"
    )
    complement_c4_plasma_g_per_l: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.35 g/L — midpoint 0.2–0.5 g/L; plasma C4"
    )
    monocyte_blood_count_lower_per_ul: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 per µL — lower monocyte count in healthy adults"
    )
    monocyte_blood_count_upper_per_ul: Mapped[Optional[float]] = mapped_column(
        Float, comment="800 per µL — upper monocyte count; differentiate to macrophages/DCs in tissue"
    )

    # ── Adaptive Immunity — T Cell / B Cell ───────────────────────────────────
    cd4_t_cell_count_lower_per_ul: Mapped[Optional[float]] = mapped_column(
        Float, comment="700 per µL — lower normal CD4+ T helper cell count"
    )
    cd4_t_cell_count_upper_per_ul: Mapped[Optional[float]] = mapped_column(
        Float, comment="2100 per µL — upper normal CD4+ T helper cell count"
    )
    cd8_t_cell_count_lower_per_ul: Mapped[Optional[float]] = mapped_column(
        Float, comment="400 per µL — lower normal CD8+ cytotoxic T cell count"
    )
    cd8_t_cell_count_upper_per_ul: Mapped[Optional[float]] = mapped_column(
        Float, comment="1300 per µL — upper normal CD8+ T cell count"
    )
    cd4_cd8_ratio_normal_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 — lower normal CD4:CD8 ratio; <1.0 signals immune deficiency"
    )
    cd4_cd8_ratio_normal_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.0 — upper normal CD4:CD8 ratio"
    )
    b_cell_count_per_ul_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 per µL — lower B cell count in peripheral blood"
    )
    b_cell_count_per_ul_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 per µL — upper B cell count"
    )
    t_cell_activation_il2_production_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 h — midpoint 4–6 h; time to IL-2 production onset after TCR activation"
    )
    t_cell_proliferation_onset_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="48 h — midpoint 24–72 h; onset of clonal T cell proliferation after antigen stimulation"
    )
    antibody_peak_production_days_post_antigen: Mapped[Optional[float]] = mapped_column(
        Float, comment="12 days — midpoint 10–14 days; peak IgG antibody titre after primary immunisation"
    )
    treg_fraction_cd4_healthy: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.08 — ~8% midpoint 5–10%; Foxp3+ Treg fraction of CD4+ T cells (immune homeostasis)"
    )
    treg_exercise_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.25 — ~25% increase in Treg proportion with regular aerobic exercise training"
    )
    tcr_diversity_clonotypes: Mapped[Optional[float]] = mapped_column(
        Float, comment="1e7 — ~10⁷–10⁸ unique TCR clonotypes in healthy adult (deep sequencing estimates)"
    )

    # ── Cytokines — Pro-Inflammatory (IL-6, TNF-α, IL-1β, IL-8) ─────────────
    il6_basal_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 pg/mL — midpoint <2–10 pg/mL; baseline serum IL-6 in healthy resting adults"
    )
    il6_peak_exercise_pg_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 pg/mL — lower peak serum IL-6 during prolonged exercise (myokine release)"
    )
    il6_peak_exercise_pg_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="7000 pg/mL — upper peak IL-6 during ultramarathon/Ironman; up to 1000-fold above rest"
    )
    il6_exercise_fold_increase: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 — midpoint 100–1000×; IL-6 fold increase during prolonged exercise (mainly from muscle)"
    )
    il6_half_life_plasma_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="6 min — midpoint 5–8 min; plasma IL-6 t½ (rapid clearance)"
    )
    il6_peak_time_post_exercise_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 h — midpoint 1–3 h; time to peak serum IL-6 after endurance exercise"
    )
    il6_cytokine_storm_threshold_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 pg/mL — IL-6 threshold used in cytokine storm diagnosis (+ ferritin + CRP criteria)"
    )
    tnf_alpha_basal_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 pg/mL — midpoint <5–15 pg/mL; basal serum TNF-α in healthy adults"
    )
    tnf_alpha_sepsis_pg_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="1000 pg/mL — lower TNF-α in septic shock"
    )
    tnf_alpha_sepsis_pg_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="10000 pg/mL — upper TNF-α in fulminant sepsis; associated with shock and mortality"
    )
    tnf_alpha_half_life_plasma_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="17 min — midpoint 15–20 min; plasma TNF-α t½; rapidly cleared by receptor shedding"
    )
    tnf_alpha_nfkb_activation_threshold_ng_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 ng/mL — TNF-α concentration for 50% NF-κB activation in target cells"
    )
    il1b_basal_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="3 pg/mL — midpoint <2–5 pg/mL; basal serum IL-1β in healthy adults"
    )
    il1b_nlrp3_inflammasome_activation_atp_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 µM — extracellular ATP threshold activating NLRP3 inflammasome → IL-1β maturation"
    )
    il8_basal_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 pg/mL — midpoint 5–25 pg/mL; basal IL-8 (CXCL8); key neutrophil chemokine"
    )
    il8_peak_exercise_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 pg/mL — midpoint 50–200 pg/mL; peak IL-8 post-exhaustive exercise (muscle/neutrophil)"
    )

    # ── Cytokines — Anti-Inflammatory (IL-10, TGF-β, IL-4) ───────────────────
    il10_basal_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="7 pg/mL — midpoint <2–15 pg/mL; basal serum IL-10 (main anti-inflammatory cytokine)"
    )
    il10_peak_post_exercise_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 pg/mL — midpoint 20–100 pg/mL; IL-10 peak post-exercise (counter-regulatory surge)"
    )
    il10_il6_ratio_resolution_threshold: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.10 — IL-10:IL-6 ratio below which inflammatory resolution is impaired"
    )
    tgf_beta_plasma_pg_per_ml_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 pg/mL — lower plasma TGF-β1 in healthy adults (mostly platelet-stored)"
    )
    tgf_beta_plasma_pg_per_ml_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 pg/mL — upper plasma active TGF-β1; key for sIgA class switch and Treg induction"
    )
    ifn_gamma_basal_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 pg/mL — midpoint <10–50 pg/mL; basal IFN-γ; Th1 signature cytokine (NK/CD8)"
    )

    # ── Acute Phase Proteins — CRP / Ferritin / Fibrinogen ───────────────────
    crp_normal_upper_mg_per_l: Mapped[Optional[float]] = mapped_column(
        Float, comment="1 mg/L — CRP upper normal; <1 mg/L = low cardiovascular risk; hsCRP threshold"
    )
    crp_significant_inflammation_mg_per_l: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 mg/L — CRP above which significant acute inflammation is defined"
    )
    crp_sepsis_mg_per_l: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 mg/L — CRP threshold associated with bacterial sepsis"
    )
    crp_half_life_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="19 h — CRP plasma t½; hepatic synthesis IL-6–stimulated; peaks 48–72 h post-insult"
    )
    crp_post_marathon_mg_per_l_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 mg/L — lower CRP after marathon; peaks 24–48 h post-race"
    )
    crp_post_marathon_mg_per_l_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 mg/L — upper CRP after marathon (exercise-induced inflammation)"
    )
    ferritin_cytokine_storm_threshold_ng_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 ng/mL — serum ferritin threshold used in cytokine storm / MAS diagnosis"
    )

    # ── Open Window — Post-Exercise Immune Depression ────────────────────────
    open_window_duration_lower_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="3 h — lower estimate of post-exhaustive exercise immune depression window (infection risk)"
    )
    open_window_duration_upper_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="72 h — upper estimate of open window (prolonged after ultramarathon)"
    )
    neutrophil_burst_reduction_post_marathon_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.45 — ~45% reduction in neutrophil oxidative burst capacity immediately post-marathon"
    )
    lymphocyte_count_nadir_fraction_post_exercise: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.40 — ~40% reduction in circulating lymphocytes 1–2 h post-marathon (redistribution)"
    )
    upper_respiratory_infection_risk_overtraining_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 — midpoint 2–3×; URTI incidence fold increase in overtrained athletes vs controls"
    )

    # ── Mucosal Immune Architecture ───────────────────────────────────────────
    peyers_patches_count_small_intestine: Mapped[Optional[int]] = mapped_column(
        Integer, comment="35 — midpoint 30–40; Peyer's patches in human small intestine; antigen sampling sites"
    )
    iel_ratio_per_enterocytes: Mapped[Optional[float]] = mapped_column(
        Integer, comment="1 — 1 IEL per 10–25 enterocytes; intraepithelial lymphocytes (CD8+ αβ + γδ TCR)"
    )
    lamina_propria_plasma_cells_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — ~60% of lamina propria immune cells are IgA-secreting plasma cells"
    )
    lamina_propria_t_cells_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — ~20% lamina propria cells are T cells (mainly CD4+ Th17 + Tregs)"
    )
    lamina_propria_macrophages_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.15 — ~15% lamina propria cells are macrophages (tolerogenic M2 phenotype)"
    )
    pigr_transcytosis_rate_molecules_per_s_per_cell: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 per s — approximate pIgR-mediated transcytosis rate per epithelial cell (IgA → lumen)"
    )
