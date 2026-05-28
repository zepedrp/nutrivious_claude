from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesGastrointestinal(Base):
    __tablename__ = "species_gastrointestinal"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="gastrointestinal")

    # ── SGLT1 — Sodium-Glucose Cotransporter 1 (Glucose) ─────────────────────
    sglt1_km_glucose_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.75 mM — midpoint 0.5–1.0 mM; SGLT1 Km for luminal D-glucose in human jejunum"
    )
    sglt1_km_na_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 mM — midpoint 10–50 mM; SGLT1 Km for luminal Na⁺; 2 Na⁺ per glucose stoichiometry"
    )
    sglt1_na_glucose_stoichiometry: Mapped[Optional[int]] = mapped_column(
        Integer, comment="2 — 2 Na⁺ ions co-transported per glucose molecule; electrogenic; drives apical uptake"
    )
    sglt1_vmax_umol_per_min_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 µmol/min/cm² — midpoint 80–120; SGLT1 Vmax in human jejunal brush border (Ussing chamber)"
    )
    sglt1_max_glucose_absorption_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="60 g/h — SGLT1 intestinal saturation ceiling for glucose; Jeukendrup & Moseley 2010"
    )
    sglt1_glucose_gi_distress_threshold_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 g/h — glucose ingestion rate above which osmotic GI distress occurs (unabsorbed load)"
    )
    sglt1_expression_training_increase_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 — midpoint 1.2–2.0×; SGLT1 protein upregulation after 4 weeks high-CHO + training"
    )

    # ── GLUT5 — Facilitative Fructose Transporter ────────────────────────────
    glut5_km_fructose_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 mM — midpoint 5–15 mM; GLUT5 Km for D-fructose; facilitative (no Na⁺ dependency)"
    )
    glut5_vmax_umol_per_min_per_cm2: Mapped[Optional[float]] = mapped_column(
        Float, comment="45 µmol/min/cm² — midpoint 40–50; GLUT5 Vmax in human small intestinal brush border"
    )
    glut5_max_fructose_absorption_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 g/h — midpoint 20–30 g/h; GLUT5 saturation ceiling for fructose absorption"
    )
    glut5_fructose_tolerance_threshold_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 g/h — fructose load above which colonic delivery and fermentation cause GI distress"
    )
    glut5_expression_fructose_diet_induction_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 — midpoint 1.5–3×; GLUT5 upregulation with chronic high-fructose diet"
    )

    # ── GLUT2 — Basolateral Exit Transporter ─────────────────────────────────
    glut2_km_glucose_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="17 mM — midpoint 15–20 mM; GLUT2 Km for glucose at basolateral membrane; high-capacity exit"
    )
    glut2_km_fructose_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="65 mM — GLUT2 Km for fructose (lower affinity); not rate-limiting for fructose exit"
    )

    # ── Multiple Transportable Carbohydrates (MTC) — Co-Ingestion Ceiling ───
    mtc_glucose_fructose_2to1_max_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="90 g/h — maximum exogenous CHO oxidation: glucose (60 g/h SGLT1) + fructose (30 g/h GLUT5); Currell & Jeukendrup 2008"
    )
    mtc_glucose_fructose_sucrose_max_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="108 g/h — midpoint 90–120 g/h; max with glucose + fructose + sucrose cocktail (dual-source MTC ceiling)"
    )
    mtc_single_source_glucose_only_max_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="60 g/h — single-transporter ceiling: glucose-only ingestion saturates SGLT1 at ~60 g/h"
    )
    mtc_exogenous_cho_oxidation_rate_max_g_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 g/min — midpoint 1.3–1.8 g/min; peak exogenous CHO oxidation at VO₂max (indirect calorimetry)"
    )
    mtc_optimal_glucose_fructose_ratio: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.0 — optimal glucose:fructose mass ratio for maximal exogenous CHO oxidation (2:1)"
    )
    mtc_cho_oxidation_gut_training_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — ~20% increase in peak exogenous CHO oxidation after gut training (4–6 weeks)"
    )
    mtc_intestinal_cho_absorption_efficiency: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.95 — ~95% of ingested CHO absorbed in small intestine; 5% reaches colon (fermented)"
    )
    cho_energy_density_kj_per_g: Mapped[Optional[float]] = mapped_column(
        Float, comment="17 kJ/g — carbohydrate energy density (4 kcal/g); used for MTC energetic calculations"
    )

    # ── Gastric Emptying — Liquid Phase ──────────────────────────────────────
    gastric_emptying_rate_isotonic_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 mL/min — midpoint 20–30 mL/min; gastric emptying rate for isotonic solution (290 mOsm/kg)"
    )
    gastric_emptying_rate_4pct_cho_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="27 mL/min — midpoint 25–30 mL/min; emptying rate with 4% CHO solution (optimal sports drink)"
    )
    gastric_emptying_rate_8pct_cho_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="22 mL/min — midpoint 18–25 mL/min; emptying rate with 8% CHO solution"
    )
    gastric_emptying_rate_12pct_cho_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="17 mL/min — midpoint 14–20 mL/min; emptying rate with 12% CHO solution"
    )
    gastric_emptying_rate_600mosmol_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="13 mL/min — midpoint 10–15 mL/min; emptying rate at 600 mOsm/kg (hypertonic gel)"
    )
    gastric_emptying_rate_900mosmol_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="8 mL/min — midpoint 6–10 mL/min; emptying at 900 mOsm/kg (highly concentrated solution)"
    )
    gastric_emptying_caloric_rate_kcal_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="3 kcal/min — midpoint 2–4 kcal/min; regulated caloric emptying rate (CCK/GLP-1 feedback)"
    )
    gastric_emptying_half_time_liquid_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 min — midpoint 20–30 min; liquid gastric emptying t½ (scintigraphy; isotonic)"
    )
    gastric_emptying_half_time_solid_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="75 min — midpoint 60–90 min; solid food gastric emptying t½ (lag phase 20–40 min)"
    )
    gastric_emptying_osmolality_sensitivity_ml_per_min_per_100mosmol: Mapped[Optional[float]] = mapped_column(
        Float, comment="3 mL/min per 100 mOsm/kg — emptying rate reduction per 100 mOsm above isotonic threshold"
    )
    gastric_emptying_exercise_70pct_vo2max_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.50 — ~50% slowing of gastric emptying at ≥70% VO₂max (sympathetic inhibition)"
    )
    gastric_emptying_exercise_50pct_vo2max_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.15 — ~15% slowing at 50% VO₂max; emptying largely preserved at moderate intensity"
    )
    gastric_emptying_heat_stress_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.35 — midpoint 25–45% slowing of gastric emptying during exercise in heat (35°C)"
    )

    # ── Gastric Anatomy and Capacity ──────────────────────────────────────────
    gastric_volume_fasting_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="75 mL — midpoint 50–100 mL; fasting gastric volume (smooth muscle tone)"
    )
    gastric_volume_max_comfortable_exercise_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="2000 mL — midpoint 1500–2500 mL; maximum comfortable gastric volume during exercise"
    )
    gastric_volume_absolute_max_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="4000 mL — midpoint 1000–4000 mL; absolute distension capacity (pain/vomiting threshold)"
    )
    gastric_acid_secretion_l_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 L/day — midpoint 1–3 L/day; total HCl secretion by parietal cells (proton pump)"
    )
    gastric_acid_peak_concentration_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="150 mM — peak gastric HCl concentration (pH ~0.82); parietal cell H⁺/K⁺-ATPase"
    )
    gastric_ph_fasting: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 — midpoint 1.0–2.0; fasting intragastric pH (high acid; protein denaturation + sterilisation)"
    )
    gastric_ph_postprandial: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.5 — midpoint 3.0–6.0; postprandial intragastric pH (buffered by food proteins)"
    )
    hk_atpase_parietal_vmax_umol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 µmol H⁺/min/mg — gastric H⁺/K⁺-ATPase Vmax at maximal stimulation (omeprazole target)"
    )

    # ── Intestinal Surface Area ───────────────────────────────────────────────
    small_intestine_length_m: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.5 m — midpoint 6–7 m; total small intestine length in vivo (living adult)"
    )
    intestinal_mucosal_surface_area_m2: Mapped[Optional[float]] = mapped_column(
        Float, comment="32 m² — midpoint 30–40 m²; effective absorptive surface (villi + microvilli amplification)"
    )
    villous_height_mm_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 mm — lower villous height in human jejunum"
    )
    villous_height_mm_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.6 mm — upper villous height; tallest in jejunum; shortest in ileum"
    )
    microvillus_height_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 µm — midpoint 1–2 µm; microvillus (brush border) height per enterocyte"
    )
    villi_surface_amplification_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 — ~10× surface area amplification from intestinal villi over flat epithelium"
    )
    microvilli_surface_amplification_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 — ~20× additional surface area amplification from brush border microvilli"
    )
    enterocyte_turnover_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 days — midpoint 3–5 days; villous enterocyte lifespan (crypt → villous tip migration)"
    )

    # ── Protein Absorption Transporters ───────────────────────────────────────
    pept1_km_dipeptide_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 mM — midpoint 1–5 mM; PEPT1 (SLC15A1) Km for di/tripeptides; H⁺-dependent; main peptide transporter"
    )
    pept1_capacity_g_protein_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="250 g/day — midpoint 200–300 g; PEPT1 theoretical daily capacity for peptide absorption"
    )
    b0at1_km_neutral_aa_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 mM — midpoint 1–2 mM; B⁰AT1 (SLC6A19) Km for neutral amino acids (Ile, Leu, Val)"
    )
    protein_absorption_max_rate_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="8 g/h — midpoint 5–10 g/h; practical peak protein absorption rate (source-dependent)"
    )
    whey_protein_absorption_rate_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 g/h — whey protein (fast): rapid gastric emptying + hydrolysis; fastest source"
    )
    casein_protein_absorption_rate_g_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.5 g/h — casein (slow): clots in stomach; sustained low-level aminoacidaemia 6–8 h"
    )
    max_protein_per_meal_for_mps_g: Mapped[Optional[float]] = mapped_column(
        Float, comment="40 g/meal — upper protein dose maximising acute MPS; beyond this: oxidised, no extra anabolism"
    )
    leucine_mps_threshold_g_per_meal: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 g/meal — midpoint 2–3 g leucine per meal to maximally activate mTORC1 → MPS"
    )

    # ── Fat Absorption ────────────────────────────────────────────────────────
    pancreatic_lipase_km_tg_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 mM — midpoint 0.1–1 mM; pancreatic lipase Km for triglyceride (requires colipase)"
    )
    bile_acid_critical_micellar_concentration_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 mM — midpoint 1–2 mM; bile acid CMC for mixed micelle formation (solubilises LCFA)"
    )
    bile_acid_secretion_max_g_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 g/day — midpoint 20–30 g/day; maximum bile acid secretion into duodenum"
    )
    fat_absorption_max_g_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 g/day — approximate maximum dietary fat absorption capacity (steatorrhoea >7 g/day)"
    )
    npc1l1_km_cholesterol_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 µM — midpoint 10–50 µM; NPC1L1 Km for cholesterol; apical absorptive transporter"
    )
    cholesterol_absorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.45 — midpoint 30–60%; fractional dietary cholesterol absorption in adults"
    )

    # ── Water and Electrolyte Absorption ─────────────────────────────────────
    intestinal_water_absorption_total_l_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="9.5 L/day — midpoint 9–10 L/day; total intestinal water absorption (2 L intake + 7–8 L secretions)"
    )
    max_fluid_absorption_rate_small_intestine_ml_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="700 mL/h — midpoint 600–800 mL/h; maximum water absorption rate in proximal small intestine"
    )
    ors_glucose_concentration_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="75 mM — WHO ORS glucose concentration (SGLT1-driven Na⁺ + water co-absorption)"
    )
    ors_sodium_concentration_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="75 mM — WHO ORS Na⁺ concentration; equimolar with glucose for optimal SGLT1 coupling"
    )
    ors_osmolality_mosmol_per_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="245 mOsm/kg — WHO reduced-osmolarity ORS; hypo-osmolar for maximal net water absorption"
    )
    intestinal_secretion_total_l_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.5 L/day — midpoint 7–8 L/day; total daily GI secretions (saliva + gastric + biliary + pancreatic + intestinal)"
    )

    # ── Gut Hormones — Kinetics ───────────────────────────────────────────────
    glp1_postprandial_peak_pm: Mapped[Optional[float]] = mapped_column(
        Float, comment="75 pM — midpoint 50–100 pM; peak plasma GLP-1 within 30 min of carbohydrate meal"
    )
    glp1_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 min — midpoint 1–2 min; GLP-1 plasma t½ (DPP-4/NEP24.11 degradation)"
    )
    glp1_gastric_emptying_ec50_pm: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 pM — midpoint 10–50 pM; GLP-1 EC50 for ileal brake inhibition of gastric emptying"
    )
    cck_postprandial_peak_pm: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 pM — midpoint 10–30 pM; peak plasma CCK in response to fat + protein in duodenum"
    )
    cck_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 min — midpoint 3–7 min; CCK plasma t½; inhibits gastric emptying + stimulates gallbladder"
    )
    gip_postprandial_peak_pm: Mapped[Optional[float]] = mapped_column(
        Float, comment="400 pM — midpoint 200–600 pM; peak plasma GIP (glucose-dependent insulinotropic polypeptide)"
    )
    gip_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="7 min — midpoint 5–10 min; GIP plasma t½ (DPP-4 sensitive)"
    )
    secretin_duodenal_ph_threshold: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.5 — duodenal pH below which secretin release from S-cells is maximally triggered"
    )
    ghrelin_fasting_acylated_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="75 pg/mL — midpoint 50–100 pg/mL; fasting acylated ghrelin (orexigenic; peaks preprandially)"
    )
    ghrelin_postprandial_suppression_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — midpoint 50–70%; postprandial suppression of acylated ghrelin within 30–60 min"
    )

    # ── Pancreatic Digestive Enzymes ──────────────────────────────────────────
    pancreatic_amylase_vmax_umol_per_min_per_mg: Mapped[Optional[float]] = mapped_column(
        Float, comment="150 µmol/min/mg — pancreatic α-amylase Vmax; hydrolysis of α-1,4 glycosidic bonds at pH 7.0"
    )
    pancreatic_bicarbonate_concentration_meq_per_l: Mapped[Optional[float]] = mapped_column(
        Float, comment="120 mEq/L — peak pancreatic juice [HCO₃⁻]; secretin-stimulated; neutralises acid chyme"
    )
    pancreatic_juice_volume_l_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 L/day — midpoint 1–2 L/day; total daily pancreatic secretion volume"
    )

    # ── Splanchnic Blood Flow — Ischemia Thresholds ──────────────────────────
    splanchnic_blood_flow_resting_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="1500 mL/min — midpoint 1400–1600 mL/min; resting splanchnic BF (25–30% of resting CO)"
    )
    splanchnic_blood_flow_postprandial_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.75 — midpoint 50–100% increase in splanchnic BF within 30–60 min of meal (hyperaemia)"
    )
    splanchnic_blood_flow_exercise_70pct_vo2max_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — midpoint 50–70% reduction in splanchnic BF at 70% VO₂max (sympathetic vasoconstriction)"
    )
    splanchnic_blood_flow_vo2max_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="400 mL/min — midpoint 300–500 mL/min; residual splanchnic BF at VO₂max (severe reduction)"
    )
    splanchnic_ischemia_threshold_fraction_resting_flow: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.40 — below 40% of resting splanchnic BF → mucosal ischemia and barrier dysfunction"
    )
    mesenteric_o2_extraction_fraction_resting: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — midpoint 15–25%; splanchnic O₂ extraction fraction at rest"
    )
    mesenteric_o2_extraction_fraction_max_exercise: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.55 — midpoint 50–60%; splanchnic O₂ extraction at VO₂max (compensates ↓ BF)"
    )
    gut_ischemia_lps_release_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="3.0 — midpoint 2–4×; plasma LPS fold increase during maximal exercise (mucosal ischemia)"
    )
    ifabp_plasma_basal_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 pg/mL — midpoint <50–150 pg/mL; basal plasma I-FABP (enterocyte damage marker)"
    )
    ifabp_plasma_post_marathon_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="750 pg/mL — midpoint 500–1000 pg/mL; plasma I-FABP post-marathon (5–10× rise)"
    )
    splanchnic_vascular_resistance_fold_increase_vo2max: Mapped[Optional[float]] = mapped_column(
        Float, comment="5.0 — midpoint 4–6×; mesenteric vascular resistance fold increase at VO₂max (α-adrenergic)"
    )

    # ── Intestinal Transit Times ───────────────────────────────────────────────
    small_intestinal_transit_h_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 h — lower small intestine transit time (lactulose breath test)"
    )
    small_intestinal_transit_h_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="4 h — upper small intestine transit time"
    )
    colonic_transit_h_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="16 h — lower colonic transit time (radio-opaque markers; healthy adults)"
    )
    colonic_transit_h_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="48 h — upper normal colonic transit time"
    )
    total_gi_transit_h_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="24 h — lower total mouth-to-anus transit time (Bristol stool type 3–4)"
    )
    total_gi_transit_h_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="72 h — upper normal total GI transit time"
    )
