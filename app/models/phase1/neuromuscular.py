from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesNeuromuscular(Base):
    """
    Phase 1 — Neuromuscular system ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 10 / Matrices 10.1–10.5 (motor unit types, firing rates, NMJ, contraction velocity);
      Domain VII.1 (motor unit recruitment / Henneman size principle);
      Domain VII.2 (neuromuscular junction kinetics: ACh, nAChR, AChE);
      Domain VII.3 (Hill force-velocity / Vmax); Domain VII.4 (Ca²⁺ cycling / cross-bridge);
      Domain VII.5 (muscle architecture and fiber type distribution).

    Key equations encoded:
      F = PCSA × σ_specific × cos(θ_pennation)       [muscle force, pennate architecture]
      P = P0 × (1 − V/Vmax) / (1 + V/(Vmax × a/P0)) [Hill force-velocity hyperbola]
      P_max = P0 × Vmax / 4   (at V ≈ Vmax/3)        [peak mechanical power]
      EPP ≈ n_quanta × q_size × conductance            [end-plate potential summation]

    Units:
      _hz         = Hz (action potentials / s)
      _ms         = milliseconds
      _m_per_s    = metres / second
      _nm         = nanometres
      _um         = µM
      _n_per_cm2  = N / cm² (specific tension)
      _fl_per_s   = fibre lengths / second (Vmax)
      _mv         = millivolts
      _fraction   = dimensionless 0–1
      _per_um2    = per µm² (receptor density)
    """

    __tablename__ = "species_neuromuscular"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="neuromuscular")

    # ── Motor Unit Classification and Count ────────────────────────────────────
    # Henneman (1965) size principle: S (slow) → FR (fast-resistant) → FF (fast-fatigable)
    # Recruitment is graded by axon diameter (rheobase current threshold)
    motor_unit_type_i_contraction_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 ms; Type I (slow-oxidative) twitch contraction time lower bound; from Domain VII.1 / Matrix 10.1"
    )
    motor_unit_type_i_contraction_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="120 ms; Type I twitch contraction time upper bound; from Domain VII.1"
    )
    motor_unit_type_iia_contraction_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 ms; Type IIa (fast-resistant) twitch contraction time lower bound; from Domain VII.1"
    )
    motor_unit_type_iia_contraction_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 ms; Type IIa twitch contraction time upper bound; from Domain VII.1"
    )
    motor_unit_type_iix_contraction_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 ms; Type IIx (fast-fatigable) twitch contraction time lower bound; from Domain VII.1"
    )
    motor_unit_type_iix_contraction_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 ms; Type IIx twitch contraction time upper bound; from Domain VII.1"
    )
    motor_unit_type_i_fatigue_index_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.75; Type I fatigue index lower bound (Burke 1973: force at 2 min / initial force); from Domain VII.1"
    )
    motor_unit_type_i_fatigue_index_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.00; Type I fatigue index upper bound (virtually non-fatigable); from Domain VII.1"
    )
    motor_unit_type_iia_fatigue_index_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.25; Type IIa fatigue index lower bound; from Domain VII.1"
    )
    motor_unit_type_iia_fatigue_index_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.75; Type IIa fatigue index upper bound; from Domain VII.1"
    )
    motor_unit_type_iix_fatigue_index_ceiling: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="<0.25; Type IIx fatigue index ceiling (rapidly fatigable); from Domain VII.1"
    )

    # ── Motor Unit Recruitment Thresholds (Henneman Size Principle) ───────────
    # % MVC at which each pool begins recruitment; complete at ~85-95% MVC
    recruitment_threshold_type_i_mvc_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05; Type I MU recruitment onset lower bound (5% MVC); smallest MUs first; from Domain VII.1"
    )
    recruitment_threshold_type_i_mvc_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10; Type I MU recruitment onset upper bound (10% MVC); from Domain VII.1"
    )
    recruitment_threshold_type_iia_mvc_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30; Type IIa MU recruitment onset lower bound (30% MVC); from Domain VII.1"
    )
    recruitment_threshold_type_iia_mvc_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; Type IIa MU recruitment onset upper bound (50% MVC); from Domain VII.1"
    )
    recruitment_threshold_type_iix_mvc_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.60; Type IIx MU recruitment onset lower bound (60% MVC); from Domain VII.1"
    )
    recruitment_threshold_type_iix_mvc_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85; Type IIx MU recruitment onset upper bound (85% MVC); from Domain VII.1"
    )
    recruitment_complete_mvc_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85–0.95; MVC fraction at which full motor unit pool is recruited in large limb muscles; from Domain VII.1"
    )

    # ── Firing Rates by Fiber Type ─────────────────────────────────────────────
    # Firing rate determines force gradation after full recruitment (rate coding)
    firing_rate_type_i_onset_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~8 Hz; Type I MU onset (minimum) firing rate; from Domain VII.1 / Matrix 10.2"
    )
    firing_rate_type_i_max_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~25 Hz; Type I MU maximum sustained firing rate; from Domain VII.1"
    )
    firing_rate_type_i_optimal_tetanus_hz_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 Hz; Type I optimal tetanic fusion frequency lower bound; from Domain VII.1"
    )
    firing_rate_type_i_optimal_tetanus_hz_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 Hz; Type I optimal tetanic fusion frequency upper bound; from Domain VII.1"
    )
    firing_rate_type_iia_onset_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 Hz; Type IIa MU onset firing rate; from Domain VII.1"
    )
    firing_rate_type_iia_max_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~50 Hz; Type IIa MU maximum sustained firing rate; from Domain VII.1"
    )
    firing_rate_type_iia_optimal_tetanus_hz_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 Hz; Type IIa optimal tetanic fusion frequency lower bound; from Domain VII.1"
    )
    firing_rate_type_iia_optimal_tetanus_hz_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 Hz; Type IIa optimal tetanic fusion frequency upper bound; from Domain VII.1"
    )
    firing_rate_type_iix_onset_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40 Hz; Type IIx MU onset firing rate; from Domain VII.1"
    )
    firing_rate_type_iix_max_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~100 Hz; Type IIx MU maximum sustained firing rate; from Domain VII.1"
    )
    firing_rate_type_iix_optimal_tetanus_hz_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 Hz; Type IIx optimal tetanic fusion frequency lower bound; from Domain VII.1"
    )
    firing_rate_type_iix_optimal_tetanus_hz_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="120 Hz; Type IIx optimal tetanic fusion frequency upper bound; from Domain VII.1"
    )
    firing_rate_ballistic_peak_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~200 Hz; peak firing rate during ballistic (rapid) voluntary contractions (brief doublets); from Domain VII.1"
    )

    # ── Axonal Conduction Velocity ─────────────────────────────────────────────
    # Velocity ∝ axon diameter; myelination increases velocity ~100× vs unmyelinated
    alpha_motor_neuron_conduction_m_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 m/s; alpha motor neuron (Aα) axonal conduction velocity lower bound; from Domain VII.1"
    )
    alpha_motor_neuron_conduction_m_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="120 m/s; alpha motor neuron (Aα) conduction velocity upper bound (large-diameter, heavily myelinated); from Domain VII.1"
    )
    gamma_motor_neuron_conduction_m_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 m/s; gamma motor neuron (Aγ) conduction velocity lower bound (spindle fusimotor); from Domain VII.1"
    )
    gamma_motor_neuron_conduction_m_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 m/s; gamma motor neuron conduction velocity upper bound; from Domain VII.1"
    )
    ia_afferent_conduction_m_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 m/s; Ia afferent (muscle spindle primary) conduction velocity lower bound; from Domain VII.1"
    )
    ia_afferent_conduction_m_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="120 m/s; Ia afferent conduction velocity upper bound; from Domain VII.1"
    )
    ib_afferent_conduction_m_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 m/s; Ib afferent (Golgi tendon organ) conduction velocity lower bound; from Domain VII.1"
    )
    ib_afferent_conduction_m_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="120 m/s; Ib afferent conduction velocity upper bound; from Domain VII.1"
    )
    c_fiber_conduction_m_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 m/s; unmyelinated C fiber conduction velocity lower bound (pain, thermoception); from Domain VII.1"
    )
    c_fiber_conduction_m_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 m/s; unmyelinated C fiber conduction velocity upper bound; from Domain VII.1"
    )

    # ── Neuromuscular Junction (NMJ) — ACh Release and Reception ─────────────
    # Safety factor: EPP amplitude >> AP threshold; EPP ~40-60 mV, threshold ~-55 mV
    # AChE degrades ACh in synaptic cleft within ~1 ms → terminates signal
    nmj_ach_vesicles_per_ap: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~100–200 vesicles; ACh vesicles exocytosed per presynaptic action potential; from Domain VII.2 / Matrix 10.3"
    )
    nmj_ach_molecules_per_vesicle_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5000 molecules; ACh molecules per synaptic vesicle lower bound; from Domain VII.2"
    )
    nmj_ach_molecules_per_vesicle_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10000 molecules; ACh molecules per synaptic vesicle upper bound; from Domain VII.2"
    )
    nmj_synaptic_cleft_width_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~50 nm; NMJ synaptic cleft width; ACh diffuses across in <0.1 ms; from Domain VII.2"
    )
    nmj_nachr_density_per_um2: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10000 nAChR/µm² at motor end-plate (highest receptor density in body); from Domain VII.2"
    )
    nmj_nachr_opening_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 ms; nAChR channel mean open time lower bound; from Domain VII.2"
    )
    nmj_nachr_opening_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 ms; nAChR channel mean open time upper bound; from Domain VII.2"
    )
    nmj_ache_kcat_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1e4 s⁻¹; acetylcholinesterase (AChE) kcat; degrades ~10⁴ ACh molecules/s/enzyme; from Domain VII.2"
    )
    nmj_ache_km_ach_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 µM; K_m^AChE for ACh lower bound; from Domain VII.2"
    )
    nmj_ache_km_ach_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 µM; K_m^AChE for ACh upper bound; from Domain VII.2"
    )
    nmj_mepp_amplitude_mv: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5–1.0 mV; miniature end-plate potential (MEPP) amplitude (spontaneous single-vesicle release); from Domain VII.2"
    )
    nmj_epp_amplitude_mv_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 mV; end-plate potential (EPP) amplitude lower bound (suprathreshold by ~40 mV); from Domain VII.2"
    )
    nmj_epp_amplitude_mv_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 mV; EPP amplitude upper bound; from Domain VII.2"
    )
    nmj_safety_factor_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0; NMJ safety factor lower bound (EPP / threshold ratio); ~4-10× excess ensures reliable transmission; from Domain VII.2"
    )
    nmj_safety_factor_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10.0; NMJ safety factor upper bound; from Domain VII.2"
    )
    nmj_ach_resynthesis_rate_molecules_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5000 molecules/s/terminal; ACh resynthesis rate (ChAT enzyme); matches moderate-frequency firing; from Domain VII.2"
    )

    # ── Refractory Periods ─────────────────────────────────────────────────────
    # Absolute: Na⁺ channels inactivated → no AP possible regardless of stimulus strength
    # Relative: Na⁺ channels partially recovered → AP requires larger stimulus
    absolute_refractory_period_motor_neuron_ms: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1 ms; absolute refractory period of alpha motor neuron; ceiling firing rate = 1/ARP = ~1000 Hz; from Domain VII.1"
    )
    relative_refractory_period_motor_neuron_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 ms; relative refractory period lower bound (motor neuron); from Domain VII.1"
    )
    relative_refractory_period_motor_neuron_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 ms; relative refractory period upper bound; from Domain VII.1"
    )
    max_theoretical_firing_rate_motor_neuron_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~200–250 Hz; maximum sustained motor neuron firing rate (limited by ARP + afterhyperpolarization); from Domain VII.1"
    )
    absolute_refractory_period_muscle_fiber_ms: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–2 ms; absolute refractory period of muscle fiber action potential; from Domain VII.2"
    )

    # ── Muscle Contraction Velocity (Hill Vmax) ────────────────────────────────
    # P(V + b) = (P0 + a) × b  [Hill 1938 hyperbola]; Vmax = P0×b/a (at P=0)
    # Vmax ∝ myosin ATPase rate; type IIx ≫ type I (cross-bridge detachment rate)
    vmax_type_i_fiber_lengths_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 FL/s; Vmax Type I fibre lower bound (slow myosin MHC-I ATPase); from Domain VII.3 / Matrix 10.4"
    )
    vmax_type_i_fiber_lengths_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 FL/s; Vmax Type I fibre upper bound; from Domain VII.3"
    )
    vmax_type_iia_fiber_lengths_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0 FL/s; Vmax Type IIa fibre lower bound; from Domain VII.3"
    )
    vmax_type_iia_fiber_lengths_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6.0 FL/s; Vmax Type IIa fibre upper bound; from Domain VII.3"
    )
    vmax_type_iix_fiber_lengths_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6.0 FL/s; Vmax Type IIx fibre lower bound; from Domain VII.3"
    )
    vmax_type_iix_fiber_lengths_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12.0 FL/s; Vmax Type IIx fibre upper bound (fastest human skeletal myosin); from Domain VII.3"
    )
    hill_a_p0_ratio_type_i: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.16; Hill's a/P0 ratio for Type I fibre (dimensionless curvature parameter); from Domain VII.3"
    )
    hill_a_p0_ratio_type_iix: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.25; Hill's a/P0 ratio for Type IIx fibre; higher curvature → more power-oriented; from Domain VII.3"
    )
    peak_mechanical_power_velocity_fraction_vmax: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.30; velocity at peak mechanical power ≈ 30% Vmax (Hill equation); from Domain VII.3"
    )
    peak_mechanical_power_force_fraction_p0: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.30; force at peak mechanical power ≈ 30% P0 (Hill equation); from Domain VII.3"
    )

    # ── Specific Tension and Isometric Force ──────────────────────────────────
    # Specific tension = P0 / PCSA; depends on fiber type, sarcomere length optimum
    specific_tension_type_i_n_per_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="17 N/cm²; specific tension (P0/PCSA) Type I fibre lower bound; from Domain VII.3"
    )
    specific_tension_type_i_n_per_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 N/cm²; specific tension Type I fibre upper bound; from Domain VII.3"
    )
    specific_tension_type_iix_n_per_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 N/cm²; specific tension Type IIx fibre lower bound; from Domain VII.3"
    )
    specific_tension_type_iix_n_per_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35 N/cm²; specific tension Type IIx fibre upper bound; from Domain VII.3"
    )
    optimal_sarcomere_length_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2.2–2.5 µm; optimal sarcomere length for peak isometric force (maximum actin-myosin overlap); from Domain VII.3"
    )

    # ── Calcium Cycling — SR Release, SERCA, Troponin C ─────────────────────
    # SR Ca²⁺ release: RyR1 opens → [Ca²⁺]_cyto rises from ~0.1 µM to 10-100 µM
    # SERCA (SERCA1 in type II; SERCA2 in type I): pumps Ca²⁺ back to SR
    sr_ca_release_peak_concentration_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 µM; peak cytoplasmic [Ca²⁺] during SR release lower bound; from Domain VII.4 / Matrix 10.5"
    )
    sr_ca_release_peak_concentration_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 µM; peak cytoplasmic [Ca²⁺] during SR release upper bound (type IIx fast twitch); from Domain VII.4"
    )
    resting_cytoplasmic_ca_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.1 µM; resting cytoplasmic [Ca²⁺] (maintained by SERCA and plasma membrane ATPase); from Domain VII.4"
    )
    serca_km_ca_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 µM; K_m^SERCA for Ca²⁺ lower bound (high affinity; begins reuptake immediately post-release); from Domain VII.4"
    )
    serca_km_ca_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.3 µM; K_m^SERCA for Ca²⁺ upper bound; from Domain VII.4"
    )
    serca_reuptake_time_ms_type_i_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 ms; SERCA Ca²⁺ reuptake half-time Type I fibre lower bound; from Domain VII.4"
    )
    serca_reuptake_time_ms_type_i_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 ms; SERCA Ca²⁺ reuptake half-time Type I fibre upper bound; from Domain VII.4"
    )
    serca_reuptake_time_ms_type_iix_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 ms; SERCA Ca²⁺ reuptake half-time Type IIx fibre lower bound (SERCA1 fast isoform); from Domain VII.4"
    )
    serca_reuptake_time_ms_type_iix_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 ms; SERCA Ca²⁺ reuptake half-time Type IIx fibre upper bound; from Domain VII.4"
    )
    troponin_c_km_ca_um_type_i: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5–1.0 µM; K_m^TnC for Ca²⁺ in Type I fibre (higher sensitivity → sustained force at lower [Ca²⁺]); from Domain VII.4"
    )
    troponin_c_km_ca_um_type_iix: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.3–0.5 µM; K_m^TnC for Ca²⁺ in Type IIx fibre (paradoxically lower Km); from Domain VII.4"
    )

    # ── Cross-Bridge Cycling Kinetics ─────────────────────────────────────────
    # f = attachment rate; g = detachment rate; duty ratio = f/(f+g)
    # Type I: slow attachment + slow detachment → high duty ratio (economical)
    # Type IIx: fast attachment + fast detachment → low duty ratio (powerful)
    cross_bridge_cycle_rate_type_i_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5–10 cycles/s; cross-bridge cycling rate Type I (slow myosin MHC-I kcat); from Domain VII.4"
    )
    cross_bridge_cycle_rate_type_iix_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~60–80 cycles/s; cross-bridge cycling rate Type IIx (fast myosin MHC-IIx kcat); from Domain VII.4"
    )
    cross_bridge_attachment_rate_f_type_i_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~30–50 s⁻¹; cross-bridge attachment rate constant (f) Type I; from Domain VII.4"
    )
    cross_bridge_attachment_rate_f_type_iix_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~100–200 s⁻¹; cross-bridge attachment rate constant (f) Type IIx; from Domain VII.4"
    )
    cross_bridge_duty_ratio_type_i: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.10–0.20; duty ratio (fraction of time cross-bridge is attached) Type I; from Domain VII.4"
    )
    cross_bridge_duty_ratio_type_iix: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.02–0.05; duty ratio Type IIx (low → fast but brief force production); from Domain VII.4"
    )
    cross_bridge_power_stroke_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10 nm; myosin power stroke displacement per cross-bridge cycle; from Domain VII.4"
    )
    cross_bridge_force_per_head_pn: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~3–4 pN; force per individual myosin cross-bridge head (optical trap measurements); from Domain VII.4"
    )

    # ── Fiber Type Distribution (Vastus Lateralis Reference Muscle) ───────────
    fiber_type_i_fraction_endurance_athlete_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; Type I fibre fraction in vastus lateralis lower bound (elite endurance athlete); from Domain VII.5"
    )
    fiber_type_i_fraction_endurance_athlete_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.70; Type I fibre fraction in vastus lateralis upper bound (elite endurance athlete); from Domain VII.5"
    )
    fiber_type_i_fraction_sprinter_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.20; Type I fibre fraction in sprinters lower bound; from Domain VII.5"
    )
    fiber_type_i_fraction_sprinter_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.35; Type I fibre fraction in sprinters upper bound; from Domain VII.5"
    )
    fiber_type_iix_fraction_sprinter_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.40; Type IIx fibre fraction in elite sprinters lower bound; from Domain VII.5"
    )
    fiber_type_iix_fraction_sprinter_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.80; Type IIx fibre fraction in elite sprinters upper bound; from Domain VII.5"
    )

    # ── Muscle Architecture ────────────────────────────────────────────────────
    # Pennation angle reduces effective force transmission (cos θ) but increases PCSA
    pennation_angle_parallel_fiber_deg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.0°; pennation angle in parallel-fibred muscles (e.g. sartorius); force = PCSA × σ × 1.0; from Domain VII.5"
    )
    pennation_angle_max_deg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~30–35°; maximum pennation angle in highly pennate muscles (e.g. gastrocnemius); from Domain VII.5"
    )
    pennation_force_reduction_max_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.13; maximum force reduction due to pennation cos(30°) − 1 ≈ 0.13 (13% loss vs parallel); from Domain VII.5"
    )

    # ── Sensory Feedback — Muscle Spindle and Golgi Tendon Organ ─────────────
    # Spindle Ia: velocity-sensitive (dynamic response) + length-sensitive (static)
    # GTO Ib: force-sensitive; inhibits agonist MN (autogenic inhibition) via Ib interneurons
    spindle_ia_max_firing_rate_hz: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~300 Hz; maximum Ia afferent firing rate during maximal velocity stretch; from Domain VII.5"
    )
    gto_force_threshold_g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 g; Golgi tendon organ activation threshold lower bound; from Domain VII.5"
    )
    gto_force_threshold_g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.2 g; Golgi tendon organ activation threshold upper bound; from Domain VII.5"
    )
    stretch_reflex_loop_time_ms_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 ms; monosynaptic stretch reflex loop time lower bound (Ia → spinal cord → MN → muscle); from Domain VII.5"
    )
    stretch_reflex_loop_time_ms_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35 ms; stretch reflex loop time upper bound; from Domain VII.5"
    )
