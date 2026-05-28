from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesNeuralCognitive(Base):
    """
    Phase 1 — Neural processing and central fatigue ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 12 / Matrices 12.1–12.5 (cognitive speed, central fatigue, BBB, glymphatic, synaptic);
      Domain IX.1 (cognitive processing: reaction time, synaptic delay, conduction velocity);
      Domain IX.2 (central fatigue: adenosine accumulation, dopamine/serotonin saturation);
      Domain IX.3 (blood-brain barrier permeability: GLUT1, LAT1, P-gp efflux, tight junctions);
      Domain IX.4 (glymphatic system: AQP4, CSF-ISF exchange, sleep-dependent clearance);
      Domain IX.5 (central synaptic transmission: glutamate/GABA kinetics, LTP, vesicle pools).

    Key equations encoded:
      [Adenosine]_BF ≈ f(wakefulness_duration)            [basal forebrain accumulation; caffeine antagonism]
      BBB_flux = P × A × (C_plasma − C_brain)             [Fick permeability law; P = permeability coefficient]
      J_glymphatic = AQP4_conductance × Δosmotic_pressure  [convective ISF-CSF exchange]
      5-HT/DA ratio → central fatigue threshold            [Meeusen et al. 2006 central fatigue model]

    Units:
      _ms         = milliseconds
      _m_per_s    = metres / second
      _nm         = nM (nanomolar)
      _um         = µM (micromolar)
      _mm         = mM (millimolar)
      _hz         = Hz (cycles per second)
      _ml_per_min = mL / min
      _fraction   = dimensionless 0–1
      _umol_g     = µmol / g tissue
      _ml_min_100g= mL / min / 100 g brain tissue
    """

    __tablename__ = "species_neural_cognitive"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="neural_cognitive")

    # ── Cognitive Processing Speed ─────────────────────────────────────────────
    # Simple reaction time: sensory → primary cortex → motor cortex → effector
    # Choice reaction time adds discrimination time (Hick's law: RT ∝ log₂ n_choices)
    simple_reaction_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="150 ms; simple reaction time lower bound (visual stimulus → key press); from Domain IX.1 / Matrix 12.1"
    )
    simple_reaction_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="250 ms; simple reaction time upper bound; from Domain IX.1"
    )
    choice_reaction_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="300 ms; choice reaction time lower bound (2-choice discrimination); from Domain IX.1"
    )
    choice_reaction_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="500 ms; choice reaction time upper bound; from Domain IX.1"
    )
    cortical_processing_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 ms; cortical processing time lower bound (simple sensory–motor loop); from Domain IX.1"
    )
    cortical_processing_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="150 ms; cortical processing time upper bound (complex executive decision); from Domain IX.1"
    )
    synaptic_delay_chemical_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 ms; chemical synaptic delay lower bound (vesicle fusion → postsynaptic AP); from Domain IX.1"
    )
    synaptic_delay_chemical_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 ms; chemical synaptic delay upper bound (modulatory synapses); from Domain IX.1"
    )
    myelinated_axon_conduction_cortical_m_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 m/s; myelinated cortical axon conduction velocity lower bound (smaller diameter fibres); from Domain IX.1"
    )
    myelinated_axon_conduction_cortical_m_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 m/s; myelinated cortical axon conduction velocity upper bound; from Domain IX.1"
    )
    hicks_law_slope_ms_per_bit: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~150 ms/bit; Hick's law slope (RT increase per additional bit of information); from Domain IX.1"
    )

    # ── Central Fatigue — Adenosine Accumulation ──────────────────────────────
    # ATP catabolism: ATP → ADP → AMP → adenosine (IMP pathway also contributes)
    # Adenosine accumulates in basal forebrain during wakefulness → inhibits arousal centres
    # Caffeine competitively antagonises A1 and A2A adenosine receptors
    adenosine_basal_forebrain_resting_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.1 µM; basal forebrain adenosine concentration at wakefulness onset (rested); from Domain IX.2 / Matrix 12.2"
    )
    adenosine_basal_forebrain_16h_wakefulness_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.3–0.5 µM; basal forebrain adenosine after 16h continuous wakefulness (sleep pressure peak); from Domain IX.2"
    )
    adenosine_accumulation_rate_um_per_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.02 µM/h; adenosine accumulation rate in basal forebrain during sustained wakefulness; from Domain IX.2"
    )
    adenosine_a1_receptor_km_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~300 nM; K_m^A1R for adenosine; A1 receptor inhibits arousal neurones (locus coeruleus, raphe); from Domain IX.2"
    )
    adenosine_a2a_receptor_km_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~150 nM; K_m^A2AR for adenosine; A2A disinhibits GABAergic sleep-promoting neurones; from Domain IX.2"
    )
    caffeine_ki_a1_receptor_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 µM; caffeine competitive Ki at A1 adenosine receptor lower bound; from Domain IX.2"
    )
    caffeine_ki_a1_receptor_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 µM; caffeine Ki at A1 receptor upper bound; from Domain IX.2"
    )
    caffeine_ki_a2a_receptor_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2 µM; caffeine competitive Ki at A2A receptor lower bound (more potent than A1); from Domain IX.2"
    )
    caffeine_ki_a2a_receptor_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 µM; caffeine Ki at A2A receptor upper bound; from Domain IX.2"
    )
    plasma_caffeine_ergogenic_threshold_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10–20 µM; plasma caffeine concentration for significant adenosine receptor occupancy and ergogenic effect; from Domain IX.2"
    )

    # ── Central Fatigue — Dopamine and Serotonin Limits ──────────────────────
    # Central fatigue hypothesis (Meeusen et al. 2006): ↑ 5-HT/DA ratio → central fatigue
    # DA depletion in striatum/prefrontal → ↓ motivation, ↓ motor drive
    # 5-HT from ↑ brain tryptophan uptake (BCAA compete at LAT1 for BBB entry)
    dopamine_striatum_basal_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 nM; extracellular striatal dopamine at baseline lower bound; from Domain IX.2"
    )
    dopamine_striatum_basal_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 nM; extracellular striatal dopamine at baseline upper bound; from Domain IX.2"
    )
    dopamine_d1_receptor_km_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1000 nM; K_m^D1R for dopamine lower bound (low affinity; activated at high concentrations); from Domain IX.2"
    )
    dopamine_d1_receptor_km_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5000 nM; K_m^D1R for dopamine upper bound; from Domain IX.2"
    )
    dopamine_d2_receptor_km_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 nM; K_m^D2R for dopamine lower bound (high affinity; autoreceptor on dopaminergic terminals); from Domain IX.2"
    )
    dopamine_d2_receptor_km_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 nM; K_m^D2R for dopamine upper bound; from Domain IX.2"
    )
    serotonin_frontal_cortex_basal_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 nM; extracellular serotonin (5-HT) in frontal cortex at baseline lower bound; from Domain IX.2"
    )
    serotonin_frontal_cortex_basal_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 nM; extracellular serotonin in frontal cortex at baseline upper bound; from Domain IX.2"
    )
    serotonin_5ht1a_receptor_km_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 nM; K_m^5-HT1A for serotonin lower bound; inhibitory autoreceptor; fatigue-linked; from Domain IX.2"
    )
    serotonin_5ht1a_receptor_km_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 nM; K_m^5-HT1A for serotonin upper bound; from Domain IX.2"
    )
    central_fatigue_serotonin_da_ratio_threshold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2.0–5.0; 5-HT/DA ratio in striatum/cortex above which central fatigue behavioural markers appear (Meeusen et al. 2006); from Domain IX.2"
    )
    central_fatigue_onset_duration_min_moderate: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~90 min; exercise duration at moderate-high intensity above which central fatigue mechanisms become rate-limiting; from Domain IX.2"
    )
    brain_tryptophan_lat1_km_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.1 mM; K_m^LAT1 for tryptophan at BBB (competes with BCAA for brain entry → ↑ brain 5-HT); from Domain IX.2"
    )
    bcaa_lat1_km_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 mM; K_m^LAT1 for BCAA (leucine) lower bound; competitive with tryptophan at same transporter; from Domain IX.2"
    )
    bcaa_lat1_km_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.3 mM; K_m^LAT1 for BCAA upper bound; from Domain IX.2"
    )
    norepinephrine_locus_coeruleus_basal_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–10 nM; extracellular norepinephrine in forebrain at baseline; locus coeruleus origin; from Domain IX.2"
    )

    # ── Blood-Brain Barrier (BBB) Permeability ────────────────────────────────
    # Tight junctions (claudin-5, occludin, ZO-1): near-zero paracellular flux for polar molecules
    # Transcellular: gases (O2, CO2) freely permeable; glucose via GLUT1; amino acids via LAT1
    # P-gp (ABCB1): luminal efflux pump; extrudes lipophilic drugs/toxins
    bbb_glut1_km_glucose_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 mM; K_m^GLUT1 for glucose at BBB lower bound; primary brain glucose transporter; from Domain IX.3 / Matrix 12.3"
    )
    bbb_glut1_km_glucose_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0 mM; K_m^GLUT1 for glucose at BBB upper bound; from Domain IX.3"
    )
    brain_glucose_uptake_umol_100g_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5 µmol/100g brain/min; basal cerebral glucose metabolic rate (CMRglc); from Domain IX.3"
    )
    critical_brain_glucose_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2 mM; critical brain extracellular glucose below which cognitive impairment occurs; from Domain IX.3"
    )
    critical_brain_po2_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 mmHg; critical brain pO2 below which loss of consciousness threshold; from Domain IX.3"
    )
    bbb_o2_permeability_freely_permeable: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0; O2 permeability coefficient relative flag (freely permeable; no transporter needed); from Domain IX.3"
    )
    bbb_co2_permeability_freely_permeable: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0; CO2 permeability coefficient relative flag (freely permeable; central chemoreception); from Domain IX.3"
    )
    bbb_albumin_permeability_fraction_per_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="<0.001 fraction/h; BBB albumin transcytosis rate (essentially impermeable to large proteins); from Domain IX.3"
    )
    bbb_tight_junction_paracellular_permeability_cm_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1e-7 cm/s; BBB paracellular permeability for polar solutes (claudin-5 / ZO-1 sealed junctions); from Domain IX.3"
    )
    pgp_abcb1_atp_per_molecule_exported: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0; P-glycoprotein (ABCB1) ATP hydrolysed per substrate molecule exported from brain; from Domain IX.3"
    )
    bbb_lat1_slc7a5_tryptophan_km_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.1 mM; K_m^LAT1/SLC7A5 for tryptophan at luminal BBB surface; from Domain IX.3"
    )
    cerebral_blood_flow_rest_ml_min_100g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 mL/min/100g; resting cerebral blood flow lower bound; from Domain IX.3"
    )
    cerebral_blood_flow_rest_ml_min_100g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 mL/min/100g; resting cerebral blood flow upper bound; from Domain IX.3"
    )
    cerebral_blood_flow_moderate_exercise_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.20–0.30; CBF increase fraction during moderate exercise (20-30% above rest); from Domain IX.3"
    )
    cerebral_blood_flow_max_exercise_reduction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.10–0.20; CBF reduction fraction at >70% VO2max in some subjects (hypocapnia-mediated vasoconstriction); from Domain IX.3"
    )
    neurovascular_coupling_delay_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 s; neurovascular coupling delay lower bound (neural activity → local CBF increase); from Domain IX.3"
    )
    neurovascular_coupling_delay_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 s; neurovascular coupling delay upper bound (basis of fMRI BOLD signal); from Domain IX.3"
    )
    neurovascular_cbf_increase_fold_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0; local CBF increase fold in activated cortical region lower bound (functional hyperemia); from Domain IX.3"
    )
    neurovascular_cbf_increase_fold_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0; local CBF increase fold in activated cortical region upper bound; from Domain IX.3"
    )

    # ── Glymphatic System — Sleep-Dependent Brain Clearance ───────────────────
    # Nedergaard (2012): CSF enters brain along periarterial spaces; AQP4 on astrocyte end-feet
    # drives convective exchange with ISF; clears solutes into perivenous spaces → cervical lymphatics
    # Interstitial space expands during SWS: 14% → 23% of brain volume (60% increase)
    aqp4_localization_perivascular_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.70–0.80; fraction of total AQP4 protein localised at perivascular astrocyte end-feet; from Domain IX.4 / Matrix 12.4"
    )
    glymphatic_isf_csf_exchange_awake_ml_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.01 mL/min/100g; glymphatic ISF-CSF exchange rate during wakefulness lower bound; from Domain IX.4"
    )
    glymphatic_isf_csf_exchange_awake_ml_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.03 mL/min/100g; glymphatic ISF-CSF exchange rate during wakefulness upper bound; from Domain IX.4"
    )
    glymphatic_isf_csf_exchange_sws_fold_increase: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0–10.0; fold increase in glymphatic convective flow during slow-wave sleep vs wakefulness (Nedergaard 2013); from Domain IX.4"
    )
    brain_interstitial_space_fraction_awake: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.14; brain interstitial volume fraction during wakefulness (14% of brain volume); from Domain IX.4"
    )
    brain_interstitial_space_fraction_sws: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.23; brain interstitial volume fraction during SWS (23%; ~60% expansion enables convective washout); from Domain IX.4"
    )
    beta_amyloid_clearance_fraction_per_h_sws: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.07; fraction of brain interstitial Aβ cleared per hour during SWS (~7%/h); from Domain IX.4"
    )
    csf_production_rate_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.35 mL/min; CSF production rate (choroid plexus); ~500 mL/day; from Domain IX.4"
    )
    csf_turnover_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3–4 × per day; CSF complete turnover rate (volume 150 mL / 0.35 mL/min ÷ 1440 × 1440); from Domain IX.4"
    )
    tau_glymphatic_clearance_fraction_per_h_sws: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.05–0.07; fraction of interstitial tau protein cleared per hour during SWS; from Domain IX.4"
    )
    lactate_glymphatic_clearance_fraction_per_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.20–0.40; fraction of post-exercise brain interstitial lactate cleared per hour via glymphatic; from Domain IX.4"
    )
    brain_glycogen_astrocyte_umol_per_g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 µmol/g brain; astrocytic glycogen concentration lower bound; from Domain IX.4"
    )
    brain_glycogen_astrocyte_umol_per_g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 µmol/g brain; astrocytic glycogen concentration upper bound; from Domain IX.4"
    )
    brain_glycogen_depletion_exercise_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30–0.50; fraction of brain glycogen depleted during intense prolonged exercise; from Domain IX.4"
    )

    # ── Central Synaptic Transmission — Glutamate and GABA Kinetics ───────────
    # Glutamate: primary excitatory NT; cleared by EAAT transporters within 1-3 ms
    # GABA: primary inhibitory NT; GABA-A (Cl⁻ channel) and GABA-B (K⁺/cAMP coupled)
    glutamate_synaptic_peak_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 mM; peak synaptic cleft glutamate during vesicle fusion lower bound; from Domain IX.5 / Matrix 12.5"
    )
    glutamate_synaptic_peak_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 mM; peak synaptic cleft glutamate upper bound; from Domain IX.5"
    )
    glutamate_extracellular_basal_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 µM; basal extracellular glutamate lower bound (EAAT-maintained); from Domain IX.5"
    )
    glutamate_extracellular_basal_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 µM; basal extracellular glutamate upper bound; from Domain IX.5"
    )
    ampa_km_glutamate_mm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5–1.0 mM; K_m^AMPAR for glutamate; fast ionotropic receptor (Na⁺/K⁺); from Domain IX.5"
    )
    nmda_km_glutamate_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 µM; K_m^NMDAR for glutamate lower bound (high-affinity coincidence detector; Mg²⁺ block); from Domain IX.5"
    )
    nmda_km_glutamate_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 µM; K_m^NMDAR for glutamate upper bound; from Domain IX.5"
    )
    nmda_glycine_coagonist_km_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5–50 µM; K_m^NMDAR for glycine co-agonist (GluN1 site); required for channel opening; from Domain IX.5"
    )
    eaat_glutamate_clearance_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 ms; EAAT astrocyte glutamate clearance time lower bound post-release; from Domain IX.5"
    )
    eaat_glutamate_clearance_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 ms; EAAT glutamate clearance time upper bound; from Domain IX.5"
    )
    gaba_extracellular_basal_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 µM; basal extracellular GABA lower bound (GAT-1 maintained); from Domain IX.5"
    )
    gaba_extracellular_basal_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 µM; basal extracellular GABA upper bound; from Domain IX.5"
    )
    gaba_a_km_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 µM; K_m^GABA-A for GABA lower bound (ionotropic Cl⁻; fast inhibition); from Domain IX.5"
    )
    gaba_a_km_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 µM; K_m^GABA-A for GABA upper bound; from Domain IX.5"
    )
    gaba_b_km_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 µM; K_m^GABA-B for GABA lower bound (metabotropic; K⁺ / ↓cAMP; slow inhibition); from Domain IX.5"
    )
    gaba_b_km_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 µM; K_m^GABA-B for GABA upper bound; from Domain IX.5"
    )

    # ── Synaptic Vesicle Pools ─────────────────────────────────────────────────
    # RRP (readily releasable pool): docked at active zone; released by single AP
    # Reserve pool: replenishes RRP; mobilised during high-frequency firing
    synaptic_vesicle_rrp_count_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 vesicles; readily releasable pool (RRP) per bouton lower bound; from Domain IX.5"
    )
    synaptic_vesicle_rrp_count_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 vesicles; readily releasable pool per bouton upper bound; from Domain IX.5"
    )
    synaptic_vesicle_reserve_pool_count_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 vesicles; reserve pool per bouton lower bound; from Domain IX.5"
    )
    synaptic_vesicle_reserve_pool_count_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 vesicles; reserve pool per bouton upper bound; from Domain IX.5"
    )
    vesicle_recycling_time_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 s; synaptic vesicle recycling time (compensatory endocytosis) lower bound; from Domain IX.5"
    )
    vesicle_recycling_time_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 s; synaptic vesicle recycling time upper bound; from Domain IX.5"
    )
    release_probability_per_ap_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10; vesicle release probability per action potential lower bound (p_release); from Domain IX.5"
    )
    release_probability_per_ap_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; vesicle release probability per AP upper bound; from Domain IX.5"
    )

    # ── Long-Term Potentiation (LTP) — Plasticity Thresholds ─────────────────
    # LTP: NMDAR coincidence detection → Ca²⁺ influx → CaMKII → AMPAR insertion
    # Threshold: requires both presynaptic glutamate and postsynaptic depolarisation (> −55 mV)
    ltp_nmda_ca2_influx_threshold_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–5 µM; postsynaptic [Ca²⁺] threshold for LTP induction (vs <0.5 µM → LTD); from Domain IX.5"
    )
    camkii_km_ca2_calmodulin_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5–2 µM; K_m^CaMKII for Ca²⁺-calmodulin complex; autophosphorylation → sustained kinase activity; from Domain IX.5"
    )
    ltp_induction_stimulation_frequency_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~100 Hz; minimum tetanic stimulation frequency for LTP induction in hippocampal CA1 (theta burst also effective); from Domain IX.5"
    )
    ltp_duration_hours_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 h; early-LTP (E-LTP) duration lower bound (protein-synthesis independent); from Domain IX.5"
    )
    ltp_duration_hours_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="24 h+; late-LTP (L-LTP) duration (requires BDNF, Arc, protein synthesis); from Domain IX.5"
    )

    # ── Neural Oscillations — Frequency Bands ─────────────────────────────────
    # Dominant oscillation reflects cognitive state; coupling between bands enables information transfer
    oscillation_delta_hz_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 Hz; delta band lower bound (0.5-4 Hz; SWS; memory consolidation; glymphatic activation); from Domain IX.1"
    )
    oscillation_delta_hz_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0 Hz; delta band upper bound; from Domain IX.1"
    )
    oscillation_theta_hz_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4 Hz; theta band lower bound (4-8 Hz; hippocampal; spatial navigation; working memory); from Domain IX.1"
    )
    oscillation_theta_hz_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8 Hz; theta band upper bound; from Domain IX.1"
    )
    oscillation_alpha_hz_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8 Hz; alpha band lower bound (8-12 Hz; relaxed wakefulness; cortical inhibition); from Domain IX.1"
    )
    oscillation_alpha_hz_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12 Hz; alpha band upper bound; from Domain IX.1"
    )
    oscillation_beta_hz_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12 Hz; beta band lower bound (12-30 Hz; active motor planning; cortical arousal); from Domain IX.1"
    )
    oscillation_beta_hz_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 Hz; beta band upper bound; from Domain IX.1"
    )
    oscillation_gamma_hz_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 Hz; gamma band lower bound (30-100+ Hz; feature binding; focused attention); from Domain IX.1"
    )
    oscillation_gamma_hz_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 Hz; gamma band upper bound (high-gamma extends to 200+ Hz in some cortical recordings); from Domain IX.1"
    )
    sws_slow_oscillation_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.75 Hz; SWS cortical slow oscillation frequency (up-state/down-state cycle; drives glymphatic pump); from Domain IX.4"
    )
