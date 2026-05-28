from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesProteinMetabolism(Base):
    """
    Phase 1 — Protein metabolism ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 4.2 (GI protein/AA absorption: whey 10 g/h, casein 6.1 g/h);
      Section 4.4 (Hepatic capacity: urea cycle 6 g N/day ≡ ~40 g protein/day);
      Domain I.1 (Hepatic glycogen & nitrogen processing);
      Domain III.13 (MPS/FSR: 0.03–0.25 %/h; mTORC1; ribosomes);
      Domain X.31 (Structural recovery: collagen synthesis, MPS timelines);
      Domain XII (Repair & Regeneration: satellite cells, collagen, DNA repair);
      Domain VIII.28 (GH/IGF-1 axis: GH pulse ~20–60 ng/mL SWS; IGF-1 ~400–600 ng/mL).

    mTORC1 activation (from document):
      p70S6K1 (Ser389) → S6 ribosomal phosphorylation → 5'TOP mRNA translation
      4E-BP1 (Thr37/46) → dissociates from eIF4E → cap-dependent translation
      Leucine sensor: Sestrin2/GATOR2/RAG GTPase axis
      Elongation rate: ~5 amino acids/second/ribosome (mammalian cells)
      Ribosome count: ~10⁸–10⁹ per muscle cell

    Units:
      _fsr_pct_h     = fractional synthetic rate (%/hour)
      _g_per_h       = g / hour
      _g_per_kg      = g / kg body mass
      _um            = µM
      _ng_ml         = ng / mL
      _nmol_mg_min   = nmol / mg protein / min
      _umol_min_g    = µmol / min / g
    """

    __tablename__ = "species_protein_metabolism"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="protein_metabolism")

    # ── GI Protein Absorption (from document Section 4.2 / Domain I.2) ────────
    # Rate-limiting phase: luminal proteolysis (trypsin, chymotrypsin, elastase + brush-border peptidases)
    # PepT1 (SLC15A1) is H⁺-coupled; higher flux than free AA transporters → hydrolysates absorbed faster
    whey_absorption_rate_max_g_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # ~10 g/h (from document)
    casein_absorption_rate_max_g_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # ~6.1 g/h (gastric coagulation retards emptying)
    casein_hydrolysate_absorption_rate_max_g_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# ~8–9 g/h
    egg_albumin_absorption_rate_max_g_per_h_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 1.5 g/h
    egg_albumin_absorption_rate_max_g_per_h_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 3.0 g/h
    soy_isolate_absorption_rate_max_g_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # ~3.9 g/h
    whey_absorption_completeness_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 0.97 (97% absorbed in 5h; from document)
    whey_absorption_half_time_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # 2–3 h
    casein_absorption_half_time_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                # 5–7 h

    # ── Plasma Leucine Kinetics (from document) ───────────────────────────────
    # Leucine threshold for mTORC1 activation: intracellular ≥ 150–200 µM
    # Peak plasma leucine after 40g whey in 90 min: 350–500 µM
    leucine_plasma_resting_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                    # ~120–150 µM
    leucine_plasma_peak_postprandial_whey_40g_um_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 350 µM (from document)
    leucine_plasma_peak_postprandial_whey_40g_um_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 500 µM
    leucine_intracellular_mtorc1_activation_threshold_um_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 150 µM
    leucine_intracellular_mtorc1_activation_threshold_um_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 200 µM
    leucine_dose_per_meal_for_mps_activation_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # ~2.5–3.0 g (from document)

    # ── Amino Acid Transporters ────────────────────────────────────────────────
    # From document: sensor Sestrina2/GATOR2/RAG GTPase
    lat1_vmax_pmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                        # LAT1/SLC7A5 (large neutral AA incl. Leu, Phe)
    lat1_km_leucine_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lat1_km_phenylalanine_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    b0at1_vmax_pmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                       # B0AT1/SLC6A19 (broad neutral AA)
    b0at1_km_glutamine_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snat2_vmax_pmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                       # SNAT2/SLC38A2 (glutamine)
    snat2_km_glutamine_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rBAT_b0atAT_km_basic_aa_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                   # rBAT/b0,+AT for Lys, Arg (from document)
    pept1_h_coupled_km_dipeptide_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # PepT1/SLC15A1

    # ── Muscle Protein Synthesis (MPS / FSR) — from document Domain III.13 ────
    # FSR regulated by mTORC1 → p70S6K1 + 4E-BP1 → ribosome capacity
    # Elongation: ~5 aa/s/ribosome; cells: ~10⁸–10⁹ ribosomes
    # "Muscle full" saturation: ~20–40 g high-quality protein/meal
    mps_fsr_basal_pct_per_h_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # 0.03 %/h (from document)
    mps_fsr_basal_pct_per_h_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                 # 0.05 %/h
    mps_fsr_post_exercise_protein_type_iia_pct_per_h_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 0.10 %/h (from document)
    mps_fsr_post_exercise_protein_type_iia_pct_per_h_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 0.20 %/h
    mps_fsr_max_absolute_documented_pct_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 0.25 %/h (type I; endurance + protein; from document)
    mps_fsr_theoretical_ceiling_pct_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # ~0.35 %/h
    protein_per_meal_muscle_full_saturation_g_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 20 g (from document)
    protein_per_meal_muscle_full_saturation_g_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 40 g high-quality protein
    protein_dose_max_mps_g_per_kg_per_meal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # ~0.4 g/kg body mass (from document)
    eaa_dose_max_mps_stimulation_g: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # ~20–40 g EAA (above → oxidised)
    whole_body_protein_turnover_g_per_kg_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # from document
    muscle_mass_gain_rate_max_kg_per_week_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 0.2 kg/week (from document)
    muscle_mass_gain_rate_max_kg_per_week_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 0.5 kg/week (young man, ideal conditions)
    muscle_mass_gain_annual_novice_kg_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 10 kg/year (from document)
    muscle_mass_gain_annual_novice_kg_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 15 kg/year
    muscle_mass_gain_annual_advanced_kg_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 1 kg/year (from document)
    muscle_mass_gain_annual_advanced_kg_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)      # 2 kg/year

    # ── mTORC1 Signalling (from document) ────────────────────────────────────
    # Activated by: leucine (via Sestrina2/GATOR2/RAG GTPase) + insulin (PI3K/Akt/TSC2)
    # Targets: p70S6K1 (Ser389) → 5'TOP mRNA; 4E-BP1 (Thr37/46) → eIF4E release
    mtorc1_s6k1_phosphorylation_rate_max_fold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mtorc1_4ebp1_phosphorylation_rate_max_fold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raptor_rapamycin_ki_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sestrin2_leucine_binding_kd_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # Sestrina2: leucine sensor
    tsc2_gap_activity_baseline_arbitrary_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    akt_phosphorylation_rate_max_fold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ribosome_count_per_cell_basal_millions_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 100 M (10⁸; from document)
    ribosome_count_per_cell_basal_millions_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 1000 M (10⁹; from document)
    ribosome_translation_rate_aa_per_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # ~5 aa/s/ribosome (from document)
    eif4e_availability_fraction_resting: Mapped[Optional[float]] = mapped_column(Float, nullable=True)          # low when 4E-BP1 not phosphorylated
    polypeptide_elongation_rate_aa_per_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # ~5 aa/s (from document)

    # ── Muscle Protein Breakdown ───────────────────────────────────────────────
    mpb_fsr_basal_pct_per_h_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mpb_fsr_basal_pct_per_h_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ubiquitin_proteasome_flux_basal_nmol_per_g_per_h: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    autophagy_flux_basal_arbitrary_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calpain_1_activity_basal_u_per_mg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calpain_2_activity_basal_u_per_mg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    atrogin1_mrna_resting_arbitrary_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # MAFbx; E3 ligase
    murf1_mrna_resting_arbitrary_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # MuRF1; E3 ligase

    # ── Nitrogen Metabolism & Urea Cycle (from document Section 4.2 / 4.4) ────
    # Hepatic urea cycle ceiling: ~6 g N/day ≡ ~40 g protein/day (from document)
    # CPS1 is rate-limiting: Vmax ~1–2 µmol/min/g liver
    nitrogen_excretion_resting_g_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    urea_cycle_ceiling_g_nitrogen_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # ~6 g N/day (from document)
    urea_cycle_ceiling_protein_equivalent_g_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # ~40 g protein/day (from document)
    protein_intake_daily_max_practical_g_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # ~300–350 g protein/day (3–4 g/kg)
    ammonia_plasma_max_tolerated_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # hyperammonaemia threshold
    cps1_vmax_umol_min_g_liver_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)               # 1 µmol/min/g liver (from document; rate-limiting)
    cps1_vmax_umol_min_g_liver_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # 2 µmol/min/g liver
    arginase_vmax_umol_min_mg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    glutamine_synthetase_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── BCAA Metabolism ────────────────────────────────────────────────────────
    leucine_oxidation_rate_max_umol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    isoleucine_oxidation_rate_max_umol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    valine_oxidation_rate_max_umol_kg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bcat2_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                       # BCAA transaminase (mitochondrial)
    bcat2_km_leucine_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bckdh_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                       # BCKDH complex (rate-limiting BCAA catabolism)
    bckdh_km_kic_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                              # Km for α-ketoisocaproate (KIC)
    bckdk_inhibitory_ki_thiamine_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)              # BCKDK inhibitor (activates BCKDH)

    # ── GH / IGF-1 Axis (from document Domain VIII, Section 9) ───────────────
    # From document: GH peak SWS = 20–60 ng/mL; GH half-life = ~20 min
    # IGF-1 max (adolescent/acromegaly borderline) = 400–600 ng/mL
    # GH → JAK2/STAT5b → IGF-1 gene transcription (hepatic)
    gh_pulse_amplitude_sws_ng_ml_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # 20 ng/mL (SWS first half of night; from document)
    gh_pulse_amplitude_sws_ng_ml_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 60 ng/mL
    gh_half_life_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                              # ~20 min (from document)
    gh_receptor_jak2_activation_kd_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    igf1_plasma_max_natural_ng_ml_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 400 ng/mL (from document)
    igf1_plasma_max_natural_ng_ml_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)           # 600 ng/mL
    igf1_receptor_km_igf1_ng_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    igf1_half_life_h_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                         # 12 h (bound to IGFBP-3; from document)
    igf1_half_life_h_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                        # 15 h
    igfbp3_plasma_concentration_ug_ml: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Collagen & Structural Protein Synthesis (from document Domain X.31 / XII) ─
    # From document: peak collagen synthesis 6–24h post optimal mechanical load; +40–50% FSR
    # Cross-linking: pyridinoline, deoxypyridinoline (LOX-dependent; cofactor Cu²⁺ + vitamin C)
    # Biomarker: PINP (formation) + CTX-I (resorption)
    collagen_synthesis_rate_tendon_fsr_pct_per_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # %/day FSR
    collagen_synthesis_peak_post_load_h_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 6 h (from document)
    collagen_synthesis_peak_post_load_h_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)       # 24 h
    collagen_synthesis_increase_vs_rest_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 0.40–0.50 (from document: +40–50%)
    p4h_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                           # Prolyl-4-hydroxylase (vitamin C dependent)
    plod2_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                         # Lysyl hydroxylase 2
    lysyl_oxidase_vmax_nmol_mg_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                  # LOX (cross-linking; cofactor Cu²⁺)
    type_i_collagen_half_life_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Satellite Cells & Myogenesis (from document Domain XII) ──────────────
    # From document: quiescent: PAX7⁺/MYOD⁻; activated: PAX7⁺/MYOD⁺
    # Hayflick limit: 50–70 divisions/lineage; telomere shortening ~50–200 bp/division
    satellite_cell_activation_fraction_after_injury: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 0.05–0.10 (5–10% activated; from document)
    satellite_cell_hayflick_limit_divisions_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # 50
    satellite_cell_hayflick_limit_divisions_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)    # 70
    telomere_shortening_per_division_bp_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # 50 bp/division (from document)
    telomere_shortening_per_division_bp_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)        # 200 bp/division
    telomere_shortening_oxidative_stress_bp_per_div_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # 200 bp (ROS damage; from document)
    telomere_shortening_oxidative_stress_bp_per_div_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)# 400 bp
    myoblast_fusion_rate_arbitrary_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── DNA Repair Capacity (from document Domain XII) ───────────────────────
    # From document: ~10⁶ repairs/cell/day; 80% via BER (Base Excision Repair)
    # OGG1 (removes 8-OHdG): Vmax ~1–2 lesions/min/enzyme; total ~10,000–50,000 copies/cell
    dna_repair_rate_max_per_cell_per_day_millions: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # 1 M (10⁶; from document)
    ber_fraction_of_total_repair: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                    # 0.80 (80% via BER; from document)
    ogg1_vmax_lesions_per_min_per_enzyme: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # 1–2 lesions/min/enzyme (from document)
    ogg1_copies_per_cell_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                        # 10,000 (from document)
    ogg1_copies_per_cell_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)                       # 50,000 (from document)
