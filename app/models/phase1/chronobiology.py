from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesChronobiology(Base):
    """
    Phase 1 — Circadian rhythm and chronobiological ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 14 / Matrices 14.1–14.5 (circadian period, melatonin, CAR, CBT, clock genes);
      Domain XI.1 (SCN pacemaker: intrinsic τ, zeitgeber sensitivity, PRC);
      Domain XI.2 (melatonin kinetics: DLMO, peak, MT1/MT2 receptors, light suppression);
      Domain XI.3 (HPA circadian axis: cortisol rhythm, cortisol awakening response);
      Domain XI.4 (core body temperature oscillation: Tmin/Tmax, amplitude);
      Domain XI.5 (TTFL clock gene network: CLOCK/BMAL1/PER/CRY/Rev-Erbα kinetics);
      Domain XI.6 (phase desynchronisation: jet lag, shift work, social jetlag limits).

    Key equations encoded:
      TTFL: CLOCK:BMAL1 → E-box → Per/Cry → PER:CRY|CLOCK:BMAL1 inhibition  [~24h loop]
      CK1ε: PER phosphorylation → β-TrCP ubiquitination → 26S proteasome degradation
      DLMO + 14h ≈ habitual wake time                [melatonin-sleep phase relationship]
      CAR_AUCi = ∫(cortisol − C_baseline) dt  [0→45 min post-awakening]

    Units:
      _h          = hours (absolute time or duration)
      _min        = minutes
      _pg_ml      = pg / mL (melatonin, hormone concentrations)
      _nmol_l     = nmol / L
      _nm         = nM (receptor Kd)
      _lux        = lux (illuminance)
      _nm_light   = nanometres (wavelength)
      _deg_c      = °C
      _fraction   = dimensionless 0–1
      _fold       = fold-change
    """

    __tablename__ = "species_chronobiology"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="chronobiology")

    # ── Intrinsic Circadian Period (τ) — SCN Master Clock ─────────────────────
    # Measured in constant routine / forced desynchrony protocols (Czeisler et al.)
    # τ > 24h requires daily light-phase advance for entrainment to solar day
    tau_intrinsic_period_h_mean: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="24.18 h; mean intrinsic circadian period (τ) of the human SCN (Czeisler et al. 1999, Science 284:2177); from Domain XI.1 / Matrix 14.1"
    )
    tau_intrinsic_period_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="23.5 h; minimum τ recorded in human subjects (extreme morning chronotype); from Domain XI.1"
    )
    tau_intrinsic_period_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="24.7 h; maximum τ recorded in human subjects (extreme evening chronotype); from Domain XI.1"
    )
    supra_24h_daily_phase_advance_required_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.18 h; daily phase advance required for τ 24.18h to entrain to 24.00h solar day; from Domain XI.1"
    )
    entrainment_rate_per_day_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1 h/day; maximum sustainable circadian phase re-entrainment rate (jet lag recovery); from Domain XI.1"
    )
    westward_reentrainment_rate_h_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1.5 h/day; phase delay (westward travel) re-entrainment rate; faster than phase advance; from Domain XI.1"
    )
    eastward_reentrainment_rate_h_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.75 h/day; phase advance (eastward travel) re-entrainment rate; harder against natural τ; from Domain XI.1"
    )

    # ── Light Entrainment — Phase Response Curve (PRC) ────────────────────────
    # ipRGC (intrinsically photosensitive retinal ganglion cells) → RHT → SCN
    # Melanopsin (OPN4): peak sensitivity 480 nm (blue); drives non-visual light effects
    melanopsin_peak_sensitivity_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~480 nm; melanopsin (OPN4) peak spectral sensitivity wavelength (short-wave/blue light); from Domain XI.1"
    )
    light_suppression_melatonin_threshold_lux_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 lux; minimum illuminance for detectable melatonin suppression lower bound; from Domain XI.1 / Matrix 14.1"
    )
    light_suppression_melatonin_threshold_lux_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 lux; illuminance for ~50% melatonin suppression (EC50 range); from Domain XI.1"
    )
    light_melatonin_suppression_ec50_lux: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~200–400 lux; EC50 for melatonin suppression by polychromatic white light (at eye level); from Domain XI.1"
    )
    prc_phase_delay_window_clock_time_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="21.0 h (21:00); start of phase-delay zone in PRC (light exposure → phase delay); from Domain XI.1"
    )
    prc_phase_delay_window_clock_time_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="04.0 h (04:00); end of phase-delay zone in PRC; from Domain XI.1"
    )
    prc_phase_advance_window_clock_time_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="04.0 h (04:00); start of phase-advance zone in PRC (light exposure → phase advance); from Domain XI.1"
    )
    prc_phase_advance_window_clock_time_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12.0 h (12:00); end of phase-advance zone (midday light near dead zone); from Domain XI.1"
    )
    minimum_entraining_light_lux: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10–50 lux; minimum illuminance at eye level required to entrain SCN rhythm; from Domain XI.1"
    )

    # ── Melatonin — Pineal Secretion Kinetics ─────────────────────────────────
    # Synthesised from tryptophan → serotonin → NAS → melatonin (AANAT rate-limiting)
    # Dim Light Melatonin Onset (DLMO): robust phase marker at 10 pg/mL threshold
    # DLMO typically 2h before habitual sleep onset; melatonin suppressed by morning light
    dlmo_threshold_pg_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 pg/mL; standard DLMO (Dim Light Melatonin Onset) threshold (salivary or plasma); from Domain XI.2 / Matrix 14.2"
    )
    dlmo_clock_time_mean_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~21.5 h (21:30); mean DLMO clock time in intermediate chronotype on normal schedule; from Domain XI.2"
    )
    dlmo_before_sleep_onset_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2 h; DLMO precedes habitual sleep onset by ~2h (melatonin opens the sleep gate); from Domain XI.2"
    )
    melatonin_peak_pg_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 pg/mL; nocturnal melatonin peak lower bound (dim-light conditions); from Domain XI.2"
    )
    melatonin_peak_pg_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="400 pg/mL; nocturnal melatonin peak upper bound (high inter-individual variation); from Domain XI.2"
    )
    melatonin_peak_pg_ml_typical_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 pg/mL; typical nocturnal melatonin peak lower bound; from Domain XI.2"
    )
    melatonin_peak_pg_ml_typical_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="250 pg/mL; typical nocturnal melatonin peak upper bound; from Domain XI.2"
    )
    melatonin_peak_clock_time_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 h (02:00); melatonin peak clock time lower bound; from Domain XI.2"
    )
    melatonin_peak_clock_time_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0 h (04:00); melatonin peak clock time upper bound; from Domain XI.2"
    )
    melatonin_morning_suppression_pg_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="<10 pg/mL; plasma melatonin by 07:00-08:00 (light suppression complete); from Domain XI.2"
    )
    melatonin_daily_total_production_ug: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10–30 µg/night; total pineal melatonin production per night; from Domain XI.2"
    )
    melatonin_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40–60 min; plasma melatonin half-life (hepatic CYP1A2 O-demethylation → 6-OH-melatonin); from Domain XI.2"
    )
    mt1_receptor_kd_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 nM; MT1 melatonin receptor Kd lower bound (SCN, pars tuberalis); from Domain XI.2"
    )
    mt1_receptor_kd_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 nM; MT1 Kd upper bound; from Domain XI.2"
    )
    mt2_receptor_kd_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 nM; MT2 melatonin receptor Kd lower bound (retina, SCN; phase-shifting); from Domain XI.2"
    )
    mt2_receptor_kd_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 nM; MT2 Kd upper bound; from Domain XI.2"
    )
    aanat_activity_night_to_day_fold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~70–100×; night-to-day fold increase in AANAT (arylalkylamine N-acetyltransferase) activity (rate-limiting for melatonin synthesis); from Domain XI.2"
    )

    # ── Cortisol Awakening Response (CAR) ─────────────────────────────────────
    # CAR is a distinct ACTH-independent cortisol surge occurring within 30-45 min post-awakening
    # Driven by hypothalamic CRH and anticipatory stress; robust neuroendocrine biomarker
    # Blunted CAR: burnout, chronic stress, adrenal fatigue; enhanced: anticipatory anxiety
    car_onset_min_post_awakening: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0 min; CAR (cortisol awakening response) onset: begins immediately at wake; from Domain XI.3 / Matrix 14.3"
    )
    car_peak_min_post_awakening_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 min; CAR peak time post-awakening lower bound; from Domain XI.3"
    )
    car_peak_min_post_awakening_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="45 min; CAR peak time post-awakening upper bound; from Domain XI.3"
    )
    car_amplitude_percent_increase_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50%; CAR amplitude lower bound (% increase above pre-awakening cortisol); from Domain XI.3"
    )
    car_amplitude_percent_increase_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="160%; CAR amplitude upper bound; from Domain XI.3"
    )
    car_duration_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~45–60 min; total CAR duration from awakening until return to pre-wake trajectory; from Domain XI.3"
    )
    car_auci_nmol_l_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="150 nmol/L × min; CAR area under the curve (increase) lower bound; from Domain XI.3"
    )
    car_auci_nmol_l_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="600 nmol/L × min; CAR AUCi upper bound (high inter-individual variation); from Domain XI.3"
    )

    # ── Cortisol Circadian Rhythm ──────────────────────────────────────────────
    cortisol_circadian_amplitude_fold_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0; cortisol circadian amplitude (morning peak / midnight nadir) lower bound; from Domain XI.3"
    )
    cortisol_circadian_amplitude_fold_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0; cortisol circadian amplitude upper bound; from Domain XI.3"
    )
    cortisol_acrophase_clock_time_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="7.0 h (07:00); cortisol circadian acrophase (peak) lower bound clock time; from Domain XI.3"
    )
    cortisol_acrophase_clock_time_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="9.0 h (09:00); cortisol acrophase upper bound clock time; from Domain XI.3"
    )
    cortisol_nadir_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~00:00 h (midnight); cortisol circadian nadir time; from Domain XI.3"
    )

    # ── Core Body Temperature (CBT) Circadian Oscillation ────────────────────
    # CBT drives peripheral clock synchronisation; Tmin is gold-standard circadian phase marker
    # Tmin occurs ~1-2h before natural wake time; Tmax in late afternoon (~17:00)
    cbt_mean_deg_c: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="37.0°C; mean core body temperature (over 24h circadian cycle); from Domain XI.4 / Matrix 14.4"
    )
    cbt_tmin_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="36.3°C; circadian core body temperature minimum (Tmin) lower bound; from Domain XI.4"
    )
    cbt_tmin_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="36.7°C; Tmin upper bound; from Domain XI.4"
    )
    cbt_tmax_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="37.2°C; circadian core body temperature maximum (Tmax) lower bound; from Domain XI.4"
    )
    cbt_tmax_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="37.6°C; Tmax upper bound; from Domain XI.4"
    )
    cbt_circadian_amplitude_deg_c_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5°C; CBT circadian amplitude (Tmax − Tmin) lower bound; from Domain XI.4"
    )
    cbt_circadian_amplitude_deg_c_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0°C; CBT circadian amplitude upper bound; from Domain XI.4"
    )
    cbt_tmin_before_wake_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–2 h; Tmin precedes habitual wake time by 1-2h (robust phase marker for circadian timing); from Domain XI.4"
    )
    cbt_tmax_clock_time_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="16.0 h (16:00); Tmax clock time lower bound (late afternoon peak); from Domain XI.4"
    )
    cbt_tmax_clock_time_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="18.0 h (18:00); Tmax clock time upper bound; from Domain XI.4"
    )

    # ── Clock Gene TTFL — Transcription-Translation Feedback Loop ─────────────
    # Positive limb: CLOCK:BMAL1 heterodimer → E-box → Per1/2, Cry1/2, Rev-Erbα transcription
    # Negative limb: PER:CRY complex → translocates to nucleus → inhibits CLOCK:BMAL1
    # Stabilising limb: Rev-Erbα → represses BMAL1; RORα → activates BMAL1
    bmal1_mrna_acrophase_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~15–17 h (15:00-17:00); BMAL1 mRNA peak time (antiphase to PER); from Domain XI.5 / Matrix 14.5"
    )
    bmal1_protein_acrophase_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~22–24 h (22:00-00:00); BMAL1 protein peak time (delayed ~6h from mRNA); from Domain XI.5"
    )
    per1_mrna_acrophase_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~06–10 h (06:00-10:00); PER1 mRNA peak time (early morning); from Domain XI.5"
    )
    per2_mrna_acrophase_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~08–12 h (08:00-12:00); PER2 mRNA peak time; from Domain XI.5"
    )
    cry1_mrna_acrophase_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~14–18 h (14:00-18:00); CRY1 mRNA peak time (afternoon); CRY1 is a potent CLOCK:BMAL1 repressor; from Domain XI.5"
    )
    cry2_mrna_acrophase_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10–14 h (10:00-14:00); CRY2 mRNA peak time; from Domain XI.5"
    )
    rev_erba_mrna_acrophase_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~09–12 h (09:00-12:00); Rev-Erbα mRNA peak time; represses BMAL1 transcription via RORE; from Domain XI.5"
    )
    rora_mrna_acrophase_clock_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~16–20 h (16:00-20:00); RORα mRNA peak time (antiphase to Rev-Erbα); activates BMAL1; from Domain XI.5"
    )
    per_mrna_to_protein_delay_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~4–6 h; delay from PER mRNA peak to PER protein peak (translation + post-translational modification); from Domain XI.5"
    )
    per_protein_nuclear_entry_delay_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2–4 h; additional delay for PER:CRY complex nuclear translocation after protein peak; from Domain XI.5"
    )
    ttfl_total_loop_period_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~24 h; full TTFL period from CLOCK:BMAL1 activation → PER:CRY repression → BMAL1 derepression; from Domain XI.5"
    )

    # ── Clock Protein Degradation Kinetics ────────────────────────────────────
    # CK1δ/ε: phosphorylates PER → β-TrCP SCF ubiquitin ligase → 26S proteasomal degradation
    # FBXL3: E3 ligase targeting CRY for proteasomal degradation
    ck1_delta_epsilon_km_per_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2–10 µM; CK1δ/ε Km for PER substrate; phosphorylation at multiple Ser residues; from Domain XI.5"
    )
    per_protein_half_life_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4 h; PER protein half-life lower bound (with CK1δ/ε phosphorylation → proteasome); from Domain XI.5"
    )
    per_protein_half_life_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6 h; PER protein half-life upper bound; from Domain XI.5"
    )
    cry_protein_half_life_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 h; CRY protein half-life lower bound (FBXL3-mediated ubiquitination); from Domain XI.5"
    )
    cry_protein_half_life_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 h; CRY protein half-life upper bound; from Domain XI.5"
    )
    bmal1_protein_half_life_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~8–12 h; BMAL1 protein half-life (relatively stable; regulated by SIAH2 ubiquitin ligase); from Domain XI.5"
    )
    clock_protein_half_life_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~15–20 h; CLOCK protein half-life (constitutively expressed; relatively stable); from Domain XI.5"
    )

    # ── Sleep Architecture ─────────────────────────────────────────────────────
    # One complete sleep cycle ≈ 90 min; N3 (SWS) dominates first half; REM second half
    sleep_cycle_duration_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 min; one sleep cycle (N1→N2→N3→REM) duration lower bound; from Domain XI.6"
    )
    sleep_cycle_duration_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="110 min; one sleep cycle duration upper bound; from Domain XI.6"
    )
    sleep_cycles_per_night_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4 cycles/night; typical sleep cycle count lower bound (7h sleep); from Domain XI.6"
    )
    sleep_cycles_per_night_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6 cycles/night; typical sleep cycle count upper bound (9h sleep); from Domain XI.6"
    )
    n3_sws_fraction_total_sleep_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.13; N3 (slow-wave sleep) fraction of total sleep time lower bound; from Domain XI.6"
    )
    n3_sws_fraction_total_sleep_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.23; N3 fraction upper bound (predominantly first half of night); from Domain XI.6"
    )
    rem_fraction_total_sleep_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.20; REM sleep fraction of total sleep time lower bound; from Domain XI.6"
    )
    rem_fraction_total_sleep_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.25; REM fraction upper bound (predominantly second half of night); from Domain XI.6"
    )
    sleep_need_adult_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="7.0 h; minimum sleep need for full recovery in healthy adults lower bound; from Domain XI.6"
    )
    sleep_need_adult_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="9.0 h; sleep need upper bound; from Domain XI.6"
    )
    sleep_onset_latency_normal_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 min; normal sleep onset latency lower bound (time from lights-out to N1); from Domain XI.6"
    )
    sleep_onset_latency_normal_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 min; normal sleep onset latency upper bound; from Domain XI.6"
    )

    # ── Phase Desynchronisation Limits ────────────────────────────────────────
    # Shift work / jet lag: SCN entrains to light; peripheral clocks entrain to feeding/activity
    # Desynchrony between SCN and peripheral clocks → metabolic, immune and cognitive impairment
    max_phase_shift_jet_lag_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~12 h; maximum acute phase shift experienced during long-haul eastward flight; from Domain XI.6"
    )
    shift_work_internal_desynchrony_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4 h; peripheral clock (liver) phase delay vs SCN lower bound in rotating shift workers; from Domain XI.6"
    )
    shift_work_internal_desynchrony_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12 h; peripheral clock phase delay vs SCN upper bound (night shift chronic); from Domain XI.6"
    )
    social_jetlag_typical_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 h; social jetlag lower bound (discrepancy between biological and social wake time); from Domain XI.6"
    )
    social_jetlag_typical_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 h; social jetlag typical upper bound in modern populations; from Domain XI.6"
    )
    social_jetlag_metabolic_risk_threshold_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 h; social jetlag threshold above which metabolic syndrome risk is significantly elevated; from Domain XI.6"
    )
    free_running_disorder_tau_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~24.5–25.0 h; approximate τ in non-24h sleep-wake disorder (blind subjects; no light entrainment); from Domain XI.6"
    )
    clock_gene_disruption_metabolic_effect_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.30–0.50; fraction increase in metabolic syndrome markers (BMI, glucose, triglycerides) in chronic shift workers vs day workers; from Domain XI.6"
    )

    # ── Feeding-Fasting Zeitgeber for Peripheral Clocks ──────────────────────
    # Time-restricted eating (TRE): feeding window entrains liver, gut, adipose peripheral clocks
    # Overrides light-based SCN entrainment of peripheral clocks when feeding is mistimed
    feeding_window_liver_clock_entrainment_h_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4 h; minimum feeding window duration to entrain liver peripheral clock lower bound (TRE protocols); from Domain XI.6"
    )
    feeding_window_liver_clock_entrainment_h_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 h; maximum TRE window maintaining liver clock alignment with SCN upper bound; from Domain XI.6"
    )
    nocturnal_feeding_liver_phase_shift_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~8–12 h; liver peripheral clock phase shift when feeding is restricted to night (antiphase to SCN); from Domain XI.6"
    )
    exercise_peripheral_muscle_clock_phase_shift_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–3 h; phase shift of muscle peripheral clock achievable by acute exercise bout at specific circadian time; from Domain XI.6"
    )
