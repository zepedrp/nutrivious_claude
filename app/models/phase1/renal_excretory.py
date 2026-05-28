from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesRenalExcretory(Base):
    __tablename__ = "species_renal_excretory"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="renal_excretory")

    # ── Glomerular Filtration Rate (GFR) — Absolute Ceiling ──────────────────
    gfr_normal_lower_ml_per_min_per_1_73m2: Mapped[Optional[float]] = mapped_column(
        Float, comment="90 mL/min/1.73m² — lower CKD-EPI/MDRD normal GFR in healthy adults"
    )
    gfr_normal_upper_ml_per_min_per_1_73m2: Mapped[Optional[float]] = mapped_column(
        Float, comment="125 mL/min/1.73m² — upper normal GFR (inulin clearance gold standard)"
    )
    gfr_peak_trained_athlete_ml_per_min_per_1_73m2: Mapped[Optional[float]] = mapped_column(
        Float, comment="135 mL/min/1.73m² — midpoint 130–140; peak GFR in endurance-trained athletes"
    )
    gfr_age_decline_ml_per_min_per_year: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.0 mL/min/year — GFR decline rate after age 40 (physiological ageing; Berlin Ageing Study)"
    )
    gfr_ckd_stage2_threshold_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="60 mL/min/1.73m² — GFR below which CKD stage G3a begins; renal reserve reduced"
    )
    gfr_ckd_renal_failure_threshold_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 mL/min/1.73m² — GFR defining CKD G5 (kidney failure; replacement therapy needed)"
    )
    gfr_exercise_reduction_fraction_vo2max: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.45 — midpoint 40–50% GFR reduction at VO₂max (renal vasoconstriction; sympathetic)"
    )
    gfr_autoregulation_map_lower_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 mmHg — lower MAP limit of renal autoregulation (myogenic + TGF; GFR maintained)"
    )
    gfr_autoregulation_map_upper_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="180 mmHg — upper MAP limit of autoregulation before pressure-dependent GFR rise"
    )
    single_nephron_gfr_nl_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="60 nL/min — midpoint 50–70 nL/min; single nephron GFR (SNGFR) by micropuncture"
    )
    nephron_count_per_kidney_lower: Mapped[Optional[int]] = mapped_column(
        Integer, comment="700000 — lower nephron count per kidney (autopsy/stereology; Bertram et al.)"
    )
    nephron_count_per_kidney_upper: Mapped[Optional[int]] = mapped_column(
        Integer, comment="1200000 — upper nephron count per kidney; lower counts predict hypertension risk"
    )
    plasma_filtered_per_day_l: Mapped[Optional[float]] = mapped_column(
        Float, comment="180 L/day — total plasma filtered per day (120 mL/min × 1440 min); raw filtrate volume"
    )
    glomerular_kf_nl_per_min_per_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, comment="6 nL/min/mmHg — midpoint 4–8; glomerular ultrafiltration coefficient Kf"
    )
    protein_normal_excretion_upper_mg_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="150 mg/day — upper normal urinary protein excretion; >300 mg/day = proteinuria"
    )
    albumin_excretion_normal_upper_mg_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="30 mg/day — upper normal microalbuminuria threshold (AER); below = normoalbuminuria"
    )

    # ── Renal Blood Flow and Filtration Fraction ──────────────────────────────
    renal_blood_flow_ml_per_min_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="1100 mL/min — lower normal RBF (~25% of resting cardiac output)"
    )
    renal_blood_flow_ml_per_min_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="1300 mL/min — upper normal RBF; measured by PAH clearance / Fick principle"
    )
    renal_plasma_flow_ml_per_min_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="625 mL/min — lower effective RPF (measured by PAH clearance at saturation)"
    )
    renal_plasma_flow_ml_per_min_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="750 mL/min — upper effective RPF in healthy adults"
    )
    filtration_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.20 — midpoint 0.18–0.22; filtration fraction = GFR/RPF; rises with dehydration"
    )
    pah_tubular_extraction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.92 — ~92% PAH (para-aminohippurate) extraction per renal pass; RPF surrogate"
    )
    pah_tm_mg_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="80 mg/min — tubular maximum for PAH secretion (OAT1/OAT3; saturates above ~0.6 mM)"
    )

    # ── Urinary Concentration — Osmolarity Ceiling ───────────────────────────
    urine_max_osmolality_mosmol_per_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="1300 mOsm/kg — midpoint 1200–1400; maximum urinary osmolality (inner medullary gradient)"
    )
    urine_min_osmolality_mosmol_per_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 mOsm/kg — minimum urine osmolality (maximum dilution; no AVP; TAL diluting segment)"
    )
    urine_isotonic_osmolality_mosmol_per_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="290 mOsm/kg — iso-osmotic urine (no concentrating or diluting; loop outflow)"
    )
    inner_medullary_osmolality_gradient_mosmol_per_kg: Mapped[Optional[float]] = mapped_column(
        Float, comment="1300 mOsm/kg — peak inner medullary interstitial osmolality at papillary tip (urea + NaCl)"
    )
    avp_v2r_ec50_pm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.75 pM — midpoint 0.5–1.0 pM; AVP EC50 for V2R-mediated AQP2 insertion in collecting duct"
    )
    avp_plasma_threshold_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.0 pg/mL — plasma AVP threshold for detectable urinary concentration (1 pg/mL ≈ antidiuresis)"
    )
    avp_plasma_max_antidiuresis_pg_per_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 pg/mL — AVP level achieving maximal antidiuresis; plateau beyond this"
    )
    aqp2_collecting_duct_insertion_time_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="15 min — time for AQP2 vesicle exocytosis to apical membrane post-AVP (within 15 min)"
    )
    urea_inner_medullary_concentration_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="600 mM — peak urea concentration in inner medullary interstitium (UT-A1 dependent)"
    )
    urea_recycling_fraction_filtered: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.45 — midpoint 40–50% filtered urea reabsorbed and recycled into medullary gradient"
    )
    minimum_urine_volume_obligatory_ml_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="450 mL/day — midpoint 400–500; minimum urine volume to excrete daily solute load at max concentration"
    )
    maximum_urine_output_l_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="18 L/day — maximum free water clearance capacity (no AVP; maximum diluting ability)"
    )

    # ── Urea Excretion and Handling ───────────────────────────────────────────
    urea_excretion_normal_g_per_day_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 g/day — lower daily urea excretion (moderate protein intake ~70 g/day)"
    )
    urea_excretion_normal_g_per_day_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="35 g/day — upper daily urea excretion (high protein ~150 g/day)"
    )
    urea_excretion_high_protein_diet_g_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="45 g/day — urea excretion ceiling on very high protein diet (>200 g/day protein)"
    )
    bun_normal_mg_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="7 mg/dL — lower blood urea nitrogen (BUN) in fasting healthy adults"
    )
    bun_normal_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="20 mg/dL — upper BUN; BUN:creatinine ratio 10–20:1 distinguishes pre-renal vs renal"
    )
    serum_urea_mmol_per_l_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 mmol/L — lower serum urea (= BUN × 2.14; urea molecular weight 60 g/mol)"
    )
    serum_urea_mmol_per_l_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.1 mmol/L — upper normal serum urea"
    )
    urea_fractional_excretion_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.55 — midpoint 40–70%; fractional excretion of urea (FEurea); passive + UT-A mediated"
    )
    urea_clearance_fraction_gfr: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.60 — midpoint 54–65%; urea clearance as fraction of GFR (net tubular reabsorption)"
    )
    urea_liver_synthesis_rate_max_g_n_per_h: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 g N/h — midpoint 1–2 g N/h; maximum urea synthesis rate by hepatic urea cycle"
    )
    uta1_urea_transporter_km_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="100 mM — UT-A1 Km for urea in inner medullary collecting duct (AVP-stimulated)"
    )

    # ── Creatinine Excretion and Clearance ───────────────────────────────────
    serum_creatinine_male_mg_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.6 mg/dL — lower serum creatinine in adult males"
    )
    serum_creatinine_male_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.2 mg/dL — upper serum creatinine in adult males"
    )
    serum_creatinine_female_mg_per_dl_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.5 mg/dL — lower serum creatinine in adult females (lower muscle mass)"
    )
    serum_creatinine_female_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.1 mg/dL — upper serum creatinine in adult females"
    )
    creatinine_production_g_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.5 g/day — midpoint 1–2 g/day; daily creatinine production (non-enzymatic from muscle PCr)"
    )
    creatinine_clearance_male_ml_per_min_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="97 mL/min — lower creatinine clearance in males (Cockroft-Gault; slightly > GFR)"
    )
    creatinine_clearance_male_ml_per_min_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="137 mL/min — upper creatinine clearance males; OCT2/MATE1 tubular secretion adds ~10–20%"
    )
    creatinine_tubular_secretion_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.15 — ~15% midpoint 10–20%; creatinine excretion fraction from tubular secretion (OCT2)"
    )
    urinary_creatinine_excretion_g_per_day_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="1.0 g/day — lower urinary creatinine excretion (female, low muscle mass)"
    )
    urinary_creatinine_excretion_g_per_day_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="2.5 g/day — upper urinary creatinine excretion (male, high muscle mass)"
    )
    creatinine_excretion_per_kg_muscle_g_per_kg_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.023 g/kg muscle/day — constant creatinine production rate per kg skeletal muscle mass"
    )

    # ── Uric Acid Excretion ───────────────────────────────────────────────────
    serum_urate_male_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.2 mg/dL — upper normal serum urate in males; solubility limit 6.8 mg/dL at 37°C"
    )
    serum_urate_female_mg_per_dl_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.0 mg/dL — upper normal serum urate in females (oestrogen uricosuric effect)"
    )
    urate_saturation_threshold_mg_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.8 mg/dL — monosodium urate saturation at 37°C body temperature; crystal deposition threshold"
    )
    urinary_uric_acid_excretion_mg_per_day_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="250 mg/day — lower urinary uric acid excretion in healthy adults"
    )
    urinary_uric_acid_excretion_mg_per_day_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="750 mg/day — upper urinary uric acid excretion (normal diet)"
    )
    urate_fractional_excretion_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.10 — ~10% net fractional excretion of urate (filtered − reabsorbed + secreted); URAT1 dominant"
    )
    urat1_km_urate_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="200 µM — URAT1 (SLC22A12) Km for urate; main renal urate reabsorber; target of probenecid"
    )

    # ── Acid-Base — Bicarbonate Reabsorption ─────────────────────────────────
    plasma_bicarbonate_normal_meq_per_l_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="22 mEq/L — lower normal plasma [HCO₃⁻] (arterial; normal 22–26 mEq/L)"
    )
    plasma_bicarbonate_normal_meq_per_l_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="26 mEq/L — upper normal plasma [HCO₃⁻]"
    )
    bicarbonate_filtered_per_day_meq: Mapped[Optional[float]] = mapped_column(
        Float, comment="4320 mEq/day — filtered HCO₃⁻ load (120 mL/min × 24 mEq/L × 1440 min)"
    )
    bicarbonate_pct_reabsorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.83 — midpoint 80–85%; fraction of filtered HCO₃⁻ reabsorbed in proximal convoluted tubule"
    )
    bicarbonate_tm_meq_per_l_plasma_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 mEq/L — lower plasma [HCO₃⁻] threshold above which HCO₃⁻ spills into urine (Tm)"
    )
    bicarbonate_tm_meq_per_l_plasma_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="28 mEq/L — upper Tm for bicarbonate; PTH lowers Tm → bicarbonaturia"
    )
    carbonic_anhydrase_ii_kcat_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1e6 per s — CA II kcat (~10⁶/s); cytosolic; critical for PCT H⁺ secretion and HCO₃⁻ generation"
    )
    carbonic_anhydrase_iv_luminal_kcat_per_s: Mapped[Optional[float]] = mapped_column(
        Float, comment="1e5 per s — CA IV kcat (membrane-bound luminal); converts H₂CO₃ → CO₂ for reabsorption"
    )
    nhe3_km_h_intracellular_nm: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 nM — NHE3 (Na⁺/H⁺ exchanger 3) Km for intracellular H⁺; main PCT H⁺ secretor"
    )
    nhe3_km_na_intraluminal_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 mM — NHE3 Km for luminal Na⁺; Na⁺ gradient from basolateral Na⁺/K⁺-ATPase drives"
    )

    # ── Acid-Base — Net Acid Excretion ────────────────────────────────────────
    net_acid_excretion_basal_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="70 mEq/day — midpoint 60–80 mEq/day; normal NAE = TA + NH₄⁺ − HCO₃⁻ (urine)"
    )
    net_acid_excretion_max_acidosis_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="500 mEq/day — maximum NAE during severe metabolic acidosis (7–10 days adaptation)"
    )
    urine_ph_minimum: Mapped[Optional[float]] = mapped_column(
        Float, comment="4.5 — minimum achievable urine pH (H⁺-ATPase limited in α-intercalated cells of collecting duct)"
    )
    urine_ph_maximum: Mapped[Optional[float]] = mapped_column(
        Float, comment="8.2 — midpoint 8.0–8.5; maximum urine pH (β-intercalated cells secreting HCO₃⁻)"
    )
    titratable_acid_excretion_basal_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="35 mEq/day — midpoint 30–40 mEq/day; titratable acid excretion (mainly H₂PO₄⁻; pKa 6.8)"
    )
    titratable_acid_excretion_max_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="150 mEq/day — midpoint 100–200 mEq/day; maximum titratable acid under severe acidosis"
    )
    phosphate_buffer_pka: Mapped[Optional[float]] = mapped_column(
        Float, comment="6.8 — pKa HPO₄²⁻/H₂PO₄⁻; optimal urinary buffer (between plasma 7.4 and min urine pH 4.5)"
    )
    urinary_phosphate_excretion_mmol_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="25 mmol/day — midpoint 20–30 mmol/day; urinary phosphate; main titratable acid buffer"
    )

    # ── Renal Ammonia — Production and Excretion ─────────────────────────────
    renal_ammonia_excretion_basal_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="35 mEq/day — midpoint 30–40 mEq/day; basal urinary NH₄⁺ excretion"
    )
    renal_ammonia_excretion_acidosis_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="250 mEq/day — midpoint 200–300 mEq/day; NH₄⁺ excretion in compensated metabolic acidosis"
    )
    renal_ammonia_excretion_ceiling_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="350 mEq/day — midpoint 300–400 mEq/day; absolute NH₄⁺ excretion ceiling"
    )
    renal_ammonia_induction_days: Mapped[Optional[float]] = mapped_column(
        Float, comment="6 days — midpoint 5–7 days; adaptation period to reach maximal NH₄⁺ excretion during acidosis"
    )
    gls1_glutaminase_km_glutamine_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 mM — midpoint 1–3 mM; renal GLS1 Km for glutamine (primary NH₃ substrate)"
    )
    gls1_acidosis_induction_fold: Mapped[Optional[float]] = mapped_column(
        Float, comment="7.5 — midpoint 5–10×; GLS1 mRNA induction fold in proximal tubule during chronic acidosis"
    )
    nh3_pka_aqueous: Mapped[Optional[float]] = mapped_column(
        Float, comment="9.2 — pKa of NH₄⁺/NH₃; at urine pH 4.5 → 99.98% as NH₄⁺ (ionic trap; non-diffusible)"
    )
    nkcc2_ammonium_km_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 mM — midpoint NKCC2 Km for NH₄⁺ (substitutes K⁺ on K⁺ site); TAL NH₄⁺ reabsorption"
    )

    # ── Water Balance — Obligatory Losses ────────────────────────────────────
    insensible_water_loss_respiratory_ml_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="350 mL/day — midpoint 300–400 mL/day; respiratory insensible water loss at rest"
    )
    insensible_water_loss_skin_ml_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="450 mL/day — midpoint 400–500 mL/day; cutaneous transepidermal water loss at rest"
    )
    fecal_water_loss_ml_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="150 mL/day — midpoint 100–200 mL/day; fecal water loss in healthy adults"
    )
    minimum_daily_water_intake_ml: Mapped[Optional[float]] = mapped_column(
        Float, comment="1300 mL/day — midpoint 1200–1500 mL/day; minimum water intake to cover all obligatory losses"
    )
    water_reabsorption_pct_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.67 — ~67% of filtered water reabsorbed in proximal tubule (iso-osmotic; AQP1)"
    )
    water_reabsorption_loop_henle_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.15 — ~15% filtered water reabsorbed in descending limb of loop of Henle"
    )
    water_reabsorption_collecting_duct_max_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.19 — up to 19% additional filtered water reabsorbed in collecting duct (AVP-dependent)"
    )
    tal_water_permeability: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.0 — thick ascending limb (TAL) is water-impermeable; diluting segment; NaCl reabsorbed without water"
    )

    # ── Tubular Transport — Key Organic Solutes ───────────────────────────────
    sglt2_km_glucose_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="2 mM — SGLT2 Km for glucose (low-affinity high-capacity; proximal S1/S2 segment; 1Na:1Glc)"
    )
    sglt1_renal_km_glucose_mm: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.35 mM — midpoint 0.2–0.5 mM; renal SGLT1 Km (high-affinity; S3 segment; 2Na:1Glc)"
    )
    glucose_tm_mg_per_min: Mapped[Optional[float]] = mapped_column(
        Float, comment="375 mg/min — midpoint 260–525; renal glucose Tm; glycosuria above this threshold"
    )
    glucose_plasma_threshold_splay_mg_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, comment="180 mg/dL — plasma glucose threshold for glycosuria (splay phenomenon; individual variation)"
    )
    oat1_km_organic_anion_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 µM — midpoint OAT1 (SLC22A6) Km for model organic anion (p-aminohippurate); basolateral uptake"
    )
    oct2_km_creatinine_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="5000 µM — OCT2 (SLC22A2) Km for creatinine; basolateral organic cation secretion"
    )
    mate1_km_creatinine_um: Mapped[Optional[float]] = mapped_column(
        Float, comment="1500 µM — MATE1 (SLC47A1) Km for creatinine at apical membrane; rate-limiting secretion step"
    )
    amino_acid_reabsorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.99 — >99% fractional reabsorption of filtered amino acids; <150 mg/day lost (generalized aminoaciduria threshold)"
    )

    # ── Electrolyte Excretion Ceilings ───────────────────────────────────────
    sodium_excretion_minimum_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="5 mEq/day — minimum urinary Na⁺ excretion during maximal conservation (aldosterone + ANP suppressed)"
    )
    sodium_excretion_maximum_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="1000 mEq/day — upper Na⁺ excretion capacity (~23 g NaCl/day diet); linearly filtered/excreted"
    )
    potassium_excretion_minimum_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="10 mEq/day — minimum K⁺ excretion (maximal conservation; aldosterone suppressed)"
    )
    potassium_excretion_maximum_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, comment="700 mEq/day — midpoint 600–800 mEq/day; maximum K⁺ excretion ceiling (high-K diet)"
    )
    chloride_excretion_fraction_filtered: Mapped[Optional[float]] = mapped_column(
        Float, comment="0.005 — ~0.5% filtered Cl⁻ excreted normally; follows Na⁺ reabsorption passively"
    )
    magnesium_excretion_mg_per_day_lower: Mapped[Optional[float]] = mapped_column(
        Float, comment="50 mg/day — lower urinary Mg²⁺ excretion (serum Mg²⁺ maintained by renal threshold)"
    )
    magnesium_excretion_mg_per_day_upper: Mapped[Optional[float]] = mapped_column(
        Float, comment="300 mg/day — upper urinary Mg²⁺ excretion; 60% of filtered Mg²⁺ reabsorbed in TAL"
    )
