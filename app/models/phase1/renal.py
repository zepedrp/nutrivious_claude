from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesRenal(Base):
    """
    Phase 1 — Renal filtration and regulation ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 6 / Matrices 6.1–6.4 (GFR, tubular reabsorption, ammonia excretion, exercise);
      Domain III.1 (glomerular filtration kinetics); Domain III.2 (Na⁺/HCO₃⁻ reabsorption);
      Domain III.3 (glucose tubular maximum, SGLT2/SGLT1); Domain III.4 (ammonia excretion);
      Domain III.5 (EPO/HIF axis); Domain III.6 (water reabsorption / vasopressin axis).

    Key equations encoded:
      GFR = K_f × (P_GC - P_BS - π_GC)            [Starling ultrafiltration]
      CL_cr = (U_cr × V_urine) / P_cr              [creatinine clearance ≈ GFR]
      NH4⁺_excretion = GLS_Vmax × [Gln] / (Km + [Gln])  [proximal tubule glutaminase]
      Tm_glucose = SGLT2_Vmax + SGLT1_Vmax         [renal glucose tubular maximum]

    Units:
      _ml_min_m2   = mL / min / 1.73 m² body surface
      _ml_per_min  = mL / min (absolute)
      _meq_day     = mEq / day
      _mm          = mM (millimolar)
      _um          = µM
      _mosmol_kg   = mOsm / kg H₂O
      _mmhg        = mmHg
      _fraction    = dimensionless 0–1
      _mg_per_min  = mg / min
      _umol_g_min  = µmol / g tissue / min
    """

    __tablename__ = "species_renal"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="renal")

    # ── GFR — Glomerular Filtration Rate ──────────────────────────────────────
    # GFR = K_f × (P_GC − P_BS − π_GC); net ultrafiltration pressure ~15-20 mmHg
    # Gold standard: inulin clearance; clinical proxy: creatinine clearance (CL_cr)
    gfr_normal_ml_min_m2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="90 mL/min/1.73 m²; GFR normal lower bound (young adult); from Domain III.1 / Matrix 6.1"
    )
    gfr_normal_ml_min_m2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="125 mL/min/1.73 m²; GFR normal upper bound; from Domain III.1"
    )
    gfr_max_trained_athlete_ml_min_m2: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~130 mL/min/1.73 m²; GFR ceiling in highly trained endurance athletes (renal hypertrophy); from Domain III.1"
    )
    renal_blood_flow_ml_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1100 mL/min; renal blood flow lower bound (~22% cardiac output); from Domain III.1"
    )
    renal_blood_flow_ml_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1300 mL/min; renal blood flow upper bound (~25% cardiac output); from Domain III.1"
    )
    renal_plasma_flow_ml_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="650 mL/min; renal plasma flow lower bound; from Domain III.1"
    )
    renal_plasma_flow_ml_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="750 mL/min; renal plasma flow upper bound; from Domain III.1"
    )
    filtration_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.18; filtration fraction lower bound (GFR / RPF); from Domain III.1"
    )
    filtration_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.22; filtration fraction upper bound; from Domain III.1"
    )
    net_ultrafiltration_pressure_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 mmHg; net Starling ultrafiltration pressure lower bound (P_GC − P_BS − π_GC); from Domain III.1"
    )
    net_ultrafiltration_pressure_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 mmHg; net Starling ultrafiltration pressure upper bound; from Domain III.1"
    )
    glomerular_hydrostatic_pressure_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~55 mmHg; glomerular capillary hydrostatic pressure (P_GC); from Domain III.1"
    )
    bowman_capsule_pressure_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~15 mmHg; Bowman's capsule hydrostatic pressure (P_BS); from Domain III.1"
    )
    glomerular_oncotic_pressure_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~30 mmHg; glomerular capillary oncotic pressure (π_GC); from Domain III.1"
    )

    # ── Exercise-Induced Renal Hemodynamics ────────────────────────────────────
    # Sympathetic α₁ vasoconstriction diverts renal flow to active muscle at VO2max
    # Autoregulation: myogenic reflex maintains GFR when MAP = 80–180 mmHg
    rbf_reduction_at_vo2max_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50–0.75; RBF reduction fraction at VO2max (50-75% below resting); from Domain III.1 / Section 6.4"
    )
    gfr_reduction_at_vo2max_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.40–0.50; GFR reduction fraction at maximal exercise (GFR → 60-75 mL/min); from Domain III.1"
    )
    vo2max_threshold_renal_vasoconstriction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.60; >60% VO2max triggers significant renal sympathetic vasoconstriction onset; from Domain III.1"
    )
    autoregulation_map_ceiling_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="180 mmHg; MAP ceiling for renal autoregulation; above this GFR rises uncontrolled; from Domain III.1"
    )
    autoregulation_map_floor_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 mmHg; MAP floor for renal autoregulation; below this GFR falls with pressure; from Domain III.1"
    )

    # ── Sodium (Na⁺) Reabsorption Ceilings ────────────────────────────────────
    # PCT: 67% via NHE3 + Na-coupled transporters
    # TAL: 25% via NKCC2 (furosemide-sensitive)
    # DCT: 5% via NCC (thiazide-sensitive)
    # Collecting duct: 2-5% via ENaC (aldosterone-regulated)
    filtered_na_meq_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="24000 mEq/day; filtered Na⁺ lower bound (GFR 120 × 140 mEq/L × 1440 min); from Domain III.2"
    )
    filtered_na_meq_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="26000 mEq/day; filtered Na⁺ upper bound; from Domain III.2"
    )
    fena_normal_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.004; fractional excretion of Na⁺ (FENa) lower bound (0.4%) normal; from Domain III.2"
    )
    fena_normal_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.012; FENa upper bound (1.2%) normal; from Domain III.2"
    )
    pct_na_reabsorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.67; proximal convoluted tubule reabsorbs 67% of filtered Na⁺ via NHE3 + Na-cotransporters; from Domain III.2"
    )
    tal_na_reabsorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.25; thick ascending limb reabsorbs 25% of filtered Na⁺ via NKCC2; from Domain III.2"
    )
    dct_na_reabsorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05; distal convoluted tubule reabsorbs 5% of filtered Na⁺ via NCC; from Domain III.2"
    )
    collecting_duct_na_reabsorption_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.02; collecting duct ENaC-mediated Na⁺ reabsorption fraction lower bound; from Domain III.2"
    )
    collecting_duct_na_reabsorption_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05; collecting duct Na⁺ reabsorption fraction upper bound (aldosterone-stimulated); from Domain III.2"
    )
    nhe3_km_na_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~50 mM; K_m^NHE3 for Na⁺; driven by apical Na⁺ gradient; PCT rate-governing exchanger; from Domain III.2"
    )
    nkcc2_km_na_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 mM; K_m^NKCC2 for Na⁺ lower bound; thick ascending limb; furosemide-sensitive; from Domain III.2"
    )
    nkcc2_km_na_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 mM; K_m^NKCC2 for Na⁺ upper bound; from Domain III.2"
    )
    nkcc2_km_cl_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 mM; K_m^NKCC2 for Cl⁻ lower bound; from Domain III.2"
    )
    nkcc2_km_cl_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 mM; K_m^NKCC2 for Cl⁻ upper bound; from Domain III.2"
    )
    enac_open_probability_basal: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30–0.50; ENaC open probability at baseline (no aldosterone); from Domain III.2"
    )
    enac_open_probability_aldosterone_max: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.70–0.90; ENaC open probability at maximal aldosterone stimulation; from Domain III.2"
    )
    enac_aldosterone_fold_increase: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0; maximum ENaC Na⁺ transport fold-increase under aldosterone (RAAS ceiling); from Domain III.2"
    )
    minimum_urinary_na_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="<10 mEq/day; minimum urinary Na⁺ (maximum aldosterone + extreme restriction); from Domain III.2"
    )
    maximum_urinary_na_meq_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="500 mEq/day; maximum urinary Na⁺ excretion lower bound (high dietary Na⁺ load); from Domain III.2"
    )
    maximum_urinary_na_meq_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="800 mEq/day; maximum urinary Na⁺ excretion upper bound; from Domain III.2"
    )

    # ── Bicarbonate (HCO₃⁻) Reabsorption & Acid-Base Regulation ─────────────
    # PCT: 80-85% via CA II (cytosolic) + CA IV (apical luminal)
    # α-intercalated cells in collecting duct: generate new HCO₃⁻ + secrete H⁺
    # Tm HCO₃: plasma threshold; above this HCO₃⁻ spills into urine
    filtered_hco3_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~4320 mEq/day; filtered HCO₃⁻ (GFR 125 × 24 mEq/L × 1440 min); from Domain III.2"
    )
    pct_hco3_reabsorption_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.80; PCT reabsorbs 80% of filtered HCO₃⁻ via CA II + CA IV + NHE3; from Domain III.2"
    )
    pct_hco3_reabsorption_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85; PCT HCO₃⁻ reabsorption fraction upper bound; from Domain III.2"
    )
    tm_hco3_plasma_threshold_meq_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 mEq/L; plasma HCO₃⁻ tubular maximum (Tm) lower bound; above this → bicarbonate spills; from Domain III.2"
    )
    tm_hco3_plasma_threshold_meq_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="28 mEq/L; Tm HCO₃⁻ upper bound; from Domain III.2"
    )
    minimum_urine_ph: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.5; minimum achievable urine pH (H⁺ gradient 1000:1 vs plasma); maximum acidification; from Domain III.2"
    )
    maximum_urine_ph: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8.0; maximum achievable urine pH (metabolic alkalosis, all H⁺ secretion suppressed); from Domain III.2"
    )
    ca2_kcat_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1e6 s⁻¹; carbonic anhydrase II kcat (one of the fastest enzymes known); from Domain III.2"
    )
    ca2_km_co2_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~9 mM; K_m^CA II for CO₂ (cytoplasmic; catalyses CO₂ + H₂O → H⁺ + HCO₃⁻); from Domain III.2"
    )
    ca4_km_hco3_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="7 mM; K_m^CA IV for HCO₃⁻ lower bound (luminal; HCO₃⁻ → CO₂ + OH⁻ in lumen); from Domain III.2"
    )
    ca4_km_hco3_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 mM; K_m^CA IV for HCO₃⁻ upper bound; from Domain III.2"
    )
    max_net_acid_excretion_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~500 mEq/day; maximum net acid excretion in severe metabolic acidosis (NH₄⁺ + titratable acid − HCO₃⁻); from Domain III.2"
    )

    # ── Glucose Tubular Maximum (Tm_glucose) ──────────────────────────────────
    # SGLT2 (S1/S2, low-affinity/high-capacity): 90% of glucose reabsorption
    # SGLT1 (S3, high-affinity/low-capacity): 10%; same transporter as intestinal SGLT1
    # Glycosuria onset when plasma glucose exceeds renal threshold (~180 mg/dL)
    tm_glucose_mg_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~375 mg/min; mean tubular maximum for glucose reabsorption; from Domain III.3"
    )
    tm_glucose_mg_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="260 mg/min; Tm_glucose lower bound (inter-individual variation); from Domain III.3"
    )
    tm_glucose_mg_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="525 mg/min; Tm_glucose upper bound; from Domain III.3"
    )
    renal_glucose_threshold_mg_dl_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="180 mg/dL; plasma glucose threshold for glycosuria onset lower bound; from Domain III.3"
    )
    renal_glucose_threshold_mg_dl_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 mg/dL; plasma glucose threshold for glycosuria onset upper bound; from Domain III.3"
    )
    sglt2_km_glucose_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2 mM; K_m^SGLT2 for glucose (S1/S2 segment; low-affinity, high-capacity); from Domain III.3"
    )
    sglt2_na_glucose_stoichiometry: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0; SGLT2 cotransport stoichiometry: 1 Na⁺ per 1 glucose; from Domain III.3"
    )
    sglt2_fraction_glucose_reabsorption: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.90; SGLT2 accounts for 90% of total renal glucose reabsorption; from Domain III.3"
    )
    sglt1_renal_km_glucose_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.2 mM; K_m^SGLT1 renal for glucose lower bound (S3 segment; high-affinity); from Domain III.3"
    )
    sglt1_renal_km_glucose_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 mM; K_m^SGLT1 renal for glucose upper bound; from Domain III.3"
    )
    sglt1_renal_na_glucose_stoichiometry: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0; SGLT1 renal cotransport: 2 Na⁺ per 1 glucose (same as intestinal SGLT1); from Domain III.3"
    )
    sglt1_renal_fraction_glucose_reabsorption: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10; SGLT1 accounts for 10% of total renal glucose reabsorption (residual S3 capacity); from Domain III.3"
    )

    # ── Ammonia Excretion — Metabolic Acidosis Buffering Ceiling ─────────────
    # Proximal tubule GLS1 generates NH₃ from glutamine (rate-limited by GLS1 + SNAT3 uptake)
    # NKCC2 carries NH₄⁺ (substituting K⁺) → concentrates in medullary interstitium
    # Collecting duct: NH₃ diffuses from interstitium into acid urine → trapped as NH₄⁺
    # RhCG/RhBG glycoproteins facilitate NH₃ transport across collecting duct epithelium
    basal_urinary_nh4_meq_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 mEq/day; basal urinary NH₄⁺ excretion lower bound; from Domain III.4"
    )
    basal_urinary_nh4_meq_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 mEq/day; basal urinary NH₄⁺ excretion upper bound; from Domain III.4"
    )
    compensated_acidosis_nh4_meq_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 mEq/day; NH₄⁺ excretion lower bound during compensated metabolic acidosis (5-7 day adaptation); from Domain III.4"
    )
    compensated_acidosis_nh4_meq_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="300 mEq/day; NH₄⁺ excretion upper bound during compensated metabolic acidosis; from Domain III.4"
    )
    max_nh4_excretion_ceiling_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~300–400 mEq/day; absolute maximum renal NH₄⁺ excretion ceiling after full adaptation; from Domain III.4"
    )
    nh4_adaptation_period_days: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5–7 days; time required to reach maximum NH₄⁺ excretion after onset of metabolic acidosis; from Domain III.4"
    )
    gls1_km_glutamine_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mM; K_m^GLS1 for glutamine lower bound; proximal tubule mitochondrial glutaminase; from Domain III.4"
    )
    gls1_km_glutamine_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0 mM; K_m^GLS1 for glutamine upper bound; from Domain III.4"
    )
    gls1_vmax_basal_umol_g_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 µmol/g kidney/min; Vmax_GLS1 basal lower bound; from Domain III.4"
    )
    gls1_vmax_basal_umol_g_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 µmol/g kidney/min; Vmax_GLS1 basal upper bound; from Domain III.4"
    )
    gls1_acidosis_induction_fold_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0; GLS1 induction fold lower bound in chronic acidosis (low pH + low HCO₃⁻ → upregulates GLS1 mRNA); from Domain III.4"
    )
    gls1_acidosis_induction_fold_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10.0; GLS1 induction fold upper bound after 5-7 days acidosis; from Domain III.4"
    )
    snat3_km_glutamine_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 mM; K_m^SNAT3 (SLC38A3) for glutamine lower bound; basolateral uptake into proximal tubule; from Domain III.4"
    )
    snat3_km_glutamine_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mM; K_m^SNAT3 for glutamine upper bound; from Domain III.4"
    )
    max_metabolic_acid_buffered_by_nh4_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~300–400 mEq/day; metabolic acid equivalents buffered by renal NH₄⁺ excretion ceiling; from Domain III.4"
    )

    # ── Potassium (K⁺) Regulation ─────────────────────────────────────────────
    # PCT + TAL reabsorb 80-90% passively; ROMK (Kir1.1) and BK (maxi-K) secrete K⁺
    # Aldosterone: upregulates ENaC + ROMK → K⁺ secretion + Na⁺ retention coupling
    filtered_k_meq_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="700 mEq/day; filtered K⁺ lower bound (GFR 125 × 4 mEq/L × 1440 min); from Domain III.2"
    )
    filtered_k_meq_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="800 mEq/day; filtered K⁺ upper bound; from Domain III.2"
    )
    max_k_excretion_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~600–800 mEq/day; maximum K⁺ excretion (exceeds filtered load; net tubular secretion); from Domain III.2"
    )
    min_k_excretion_meq_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 mEq/day; minimum K⁺ excretion in severe K⁺ depletion (ROMK + BK suppressed); from Domain III.2"
    )

    # ── EPO / HIF-2α Oxygen-Sensing Axis ─────────────────────────────────────
    # Peritubular fibroblasts (cortex/outer medulla) sense O₂ via PHD2 → HIF-2α → EPO gene
    # PHD2 (prolyl hydroxylase 2): K_m(O₂) ~100-250 µM → exquisitely sensitive at physiological pO₂
    # Normal pO₂ keeps PHD2 active → HIF-2α hydroxylated → VHL-mediated proteasomal degradation
    # Hypoxia: PHD2 inactive → HIF-2α stable → EPO transcription
    phd2_km_o2_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 µM; K_m^PHD2 for O₂ lower bound; HIF-2α prolyl hydroxylase; from Domain III.5"
    )
    phd2_km_o2_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="250 µM; K_m^PHD2 for O₂ upper bound; sets O₂-sensitivity range for EPO regulation; from Domain III.5"
    )
    hypoxia_threshold_po2_mmhg_hif_stabilization: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40 mmHg; renal cortex pO₂ below which HIF-2α stabilises → EPO gene induction; from Domain III.5"
    )
    epo_induction_fold_altitude_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0; EPO plasma fold-increase at altitude >2500 m lower bound (within 24-48 h); from Domain III.5"
    )
    epo_induction_fold_altitude_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10.0; EPO plasma fold-increase upper bound during acclimatisation; from Domain III.5"
    )
    epo_basal_plasma_mu_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 mU/mL; basal plasma EPO lower bound (sea level, healthy); from Domain III.5"
    )
    epo_basal_plasma_mu_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 mU/mL; basal plasma EPO upper bound; from Domain III.5"
    )
    epo_half_life_iv_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~8–12 h; endogenous/exogenous EPO plasma half-life intravenous; from Domain III.5"
    )
    epo_half_life_sc_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~24 h; EPO half-life subcutaneous (slower absorption → longer effective t½); from Domain III.5"
    )
    altitude_threshold_epo_response_m: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2500 m; altitude threshold for significant EPO induction (hypoxic stimulus); from Domain III.5"
    )
    optimal_altitude_training_m_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2500 m; optimal altitude training lower bound (Live High Train Low); from Domain III.5"
    )
    optimal_altitude_training_m_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3500 m; optimal altitude training upper bound; above 4000 m hypoxia burden outweighs erythropoietic gain; from Domain III.5"
    )

    # ── Water Reabsorption / Vasopressin (AVP / ADH) Axis ────────────────────
    # AVP binds V2-R on collecting duct → cAMP → PKA → AQP2 Ser256 phosphorylation → apical insertion
    # Medullary countercurrent: papillary tip ~1200 mOsm vs cortex ~300 mOsm
    daily_filtered_water_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~178 L/day; total water filtered at GFR 125 mL/min; >99.3% reabsorbed; from Domain III.6"
    )
    pct_water_reabsorption_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.67; PCT reabsorbs 67% of filtered water (osmotic; AQP1-driven); from Domain III.6"
    )
    max_urinary_concentration_mosmol_kg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1200–1400 mOsm/kg H₂O; maximum urinary concentration (maximal AVP + medullary gradient); from Domain III.6"
    )
    min_urinary_concentration_mosmol_kg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50–100 mOsm/kg H₂O; minimum urinary concentration (maximal water diuresis, AVP suppressed); from Domain III.6"
    )
    avp_v2r_ec50_pm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 pM; AVP V2-receptor EC50 lower bound (collecting duct; extremely high affinity); from Domain III.6"
    )
    avp_v2r_ec50_pm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 pM; AVP V2-receptor EC50 upper bound; from Domain III.6"
    )
    medullary_osmolality_papilla_mosmol_kg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1200 mOsm/kg; papillary tip osmolality (countercurrent multiplier maximum); from Domain III.6"
    )
    max_free_water_clearance_ml_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="18 mL/min; maximum free water clearance lower bound (during acute water load); from Domain III.6"
    )
    max_free_water_clearance_ml_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="22 mL/min; maximum free water clearance upper bound; from Domain III.6"
    )
    urine_volume_daily_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 L/day; minimum urine volume (maximal AVP + antidiuresis; below this → oliguria); from Domain III.6"
    )
    urine_volume_daily_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20.0 L/day; maximum urine volume (maximal water diuresis, complete AVP suppression); from Domain III.6"
    )
