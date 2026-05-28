from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesMusculoskeletal(Base):
    """
    Phase 1 — Musculoskeletal biomechanical ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 11 / Matrices 11.1–11.5 (specific tension, PCSA, tendon mechanics, SSC, bone);
      Domain VIII.1 (muscle architecture: PCSA, fascicle length, pennation);
      Domain VIII.2 (sarcomere ultrastructure: titin, thick/thin filaments);
      Domain VIII.3 (tendon mechanics: Young's modulus, stiffness, failure limits);
      Domain VIII.4 (stretch-shortening cycle: elastic energy, SSC potentiation);
      Domain VIII.5 (bone material properties and lever arm biomechanics).

    Key equations encoded:
      PCSA = (V_muscle × cos θ) / L_fascicle              [physiological cross-sectional area]
      F_muscle = PCSA × σ_specific × cos θ_pennation       [muscle force output]
      E_elastic = ½ × k_tendon × Δx²                      [tendon elastic energy storage]
      σ_failure = F_failure / CSA_tendon                   [tendon ultimate tensile stress]
      ε = ΔL / L_0                                         [tendon strain at failure]

    Units:
      _n_per_cm2  = N / cm² (specific tension / stress)
      _mpa        = MPa (megapascals = N/mm²)
      _gpa        = GPa (gigapascals)
      _n_per_mm   = N / mm (stiffness)
      _mm2        = mm² (cross-sectional area)
      _cm         = centimetres (length)
      _um         = µm (micrometres; sarcomere scale)
      _fraction   = dimensionless 0–1
      _j          = joules (energy)
      _percent    = % (strain, composition)
      _g_per_cm2  = g/cm² (bone mineral density, DXA)
    """

    __tablename__ = "species_musculoskeletal"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="musculoskeletal")

    # ── Whole-Muscle Specific Tension and PCSA ────────────────────────────────
    # In vivo specific tension is lower than in vitro due to angle effects and activation limits
    # Voluntary activation ceiling: ~95-99% in highly trained athletes
    specific_tension_in_vitro_n_per_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 N/cm²; maximum voluntary specific tension in vitro lower bound (isolated fibre); from Domain VIII.1 / Matrix 11.1"
    )
    specific_tension_in_vitro_n_per_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35 N/cm²; maximum specific tension in vitro upper bound; from Domain VIII.1"
    )
    specific_tension_in_vivo_n_per_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 N/cm²; in vivo whole-muscle specific tension lower bound (ultrasound + dynamometry); from Domain VIII.1"
    )
    specific_tension_in_vivo_n_per_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 N/cm²; in vivo whole-muscle specific tension upper bound; from Domain VIII.1"
    )
    voluntary_activation_level_trained_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.95–0.99; voluntary activation fraction in highly trained athletes (interpolated twitch technique); from Domain VIII.1"
    )
    voluntary_activation_level_untrained_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85–0.95; voluntary activation fraction in untrained subjects; from Domain VIII.1"
    )
    pcsa_quadriceps_combined_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="130 cm²; combined quadriceps PCSA lower bound (sum VL + VM + VI + RF); from Domain VIII.1"
    )
    pcsa_quadriceps_combined_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 cm²; combined quadriceps PCSA upper bound; from Domain VIII.1"
    )
    pcsa_vastus_lateralis_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 cm²; vastus lateralis PCSA lower bound; from Domain VIII.1"
    )
    pcsa_vastus_lateralis_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 cm²; vastus lateralis PCSA upper bound; from Domain VIII.1"
    )
    pcsa_gluteus_maximus_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 cm²; gluteus maximus PCSA lower bound (largest single muscle mass); from Domain VIII.1"
    )
    pcsa_gluteus_maximus_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 cm²; gluteus maximus PCSA upper bound; from Domain VIII.1"
    )

    # ── Fascicle Geometry ──────────────────────────────────────────────────────
    # Fascicle length determines velocity potential (sarcomeres in series × Vmax/sarcomere)
    # Shorter fascicles = slower velocity but higher PCSA for same volume
    fascicle_length_vastus_lateralis_cm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8 cm; VL fascicle length lower bound; from Domain VIII.1"
    )
    fascicle_length_vastus_lateralis_cm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12 cm; VL fascicle length upper bound; from Domain VIII.1"
    )
    fascicle_length_soleus_cm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 cm; soleus fascicle length lower bound (short = high PCSA, low velocity); from Domain VIII.1"
    )
    fascicle_length_soleus_cm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 cm; soleus fascicle length upper bound; from Domain VIII.1"
    )
    fascicle_length_gastrocnemius_cm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 cm; gastrocnemius fascicle length lower bound; from Domain VIII.1"
    )
    fascicle_length_gastrocnemius_cm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8 cm; gastrocnemius fascicle length upper bound; from Domain VIII.1"
    )
    sarcomeres_in_series_per_fiber_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10000; sarcomeres in series per muscle fibre lower bound (10 cm fascicle / 2.5 µm); from Domain VIII.2"
    )
    sarcomeres_in_series_per_fiber_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~50000; sarcomeres in series per muscle fibre upper bound (long fibres); from Domain VIII.2"
    )

    # ── Sarcomere Ultrastructure ───────────────────────────────────────────────
    # Thick filament (myosin): 1.65 µm; thin filament (actin): 1.00 µm
    # Optimal overlap (peak force) at sarcomere length 2.2-2.5 µm
    sarcomere_length_optimal_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.2 µm; optimal sarcomere length lower bound (maximum actin-myosin overlap → peak force); from Domain VIII.2"
    )
    sarcomere_length_optimal_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.5 µm; optimal sarcomere length upper bound; from Domain VIII.2"
    )
    thick_filament_length_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.65 µm; myosin thick filament length (A-band width); from Domain VIII.2"
    )
    thin_filament_length_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.00 µm; actin thin filament length (half I-band + half A-band overlap zone); from Domain VIII.2"
    )
    titin_molecular_weight_mda_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0 MDa; titin (connectin) molecular weight lower bound; largest known protein; from Domain VIII.2"
    )
    titin_molecular_weight_mda_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.2 MDa; titin molecular weight upper bound; spans Z-disk to M-line; from Domain VIII.2"
    )
    titin_passive_stiffness_n2ba_kn_per_m: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5–2 kN/m; titin passive stiffness (N2BA isoform, skeletal); stiffer isoforms resist overstretching; from Domain VIII.2"
    )
    sarcomere_length_descending_limb_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.5–3.5 µm; sarcomere length range on descending limb of force-length (partial overlap); from Domain VIII.2"
    )
    sarcomere_length_force_zero_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~3.65 µm; sarcomere length at which active force = 0 (no actin-myosin overlap); from Domain VIII.2"
    )

    # ── Tendon Mechanical Properties ──────────────────────────────────────────
    # Tendons act as biological springs; energy storage in SSC reduces metabolic cost
    # Collagen crimp straightens in toe region (0-3% strain) before linear elastic region
    achilles_tendon_young_modulus_gpa_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 GPa; Achilles tendon Young's elastic modulus lower bound; from Domain VIII.3 / Matrix 11.3"
    )
    achilles_tendon_young_modulus_gpa_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.8 GPa; Achilles tendon Young's elastic modulus upper bound; from Domain VIII.3"
    )
    achilles_tendon_stiffness_n_per_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 N/mm; Achilles tendon stiffness lower bound; from Domain VIII.3"
    )
    achilles_tendon_stiffness_n_per_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="400 N/mm; Achilles tendon stiffness upper bound (trained > untrained); from Domain VIII.3"
    )
    patellar_tendon_stiffness_n_per_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 N/mm; patellar tendon stiffness lower bound; from Domain VIII.3"
    )
    patellar_tendon_stiffness_n_per_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="600 N/mm; patellar tendon stiffness upper bound; from Domain VIII.3"
    )
    tendon_ultimate_tensile_stress_mpa_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 MPa; tendon ultimate tensile stress lower bound (failure); from Domain VIII.3"
    )
    tendon_ultimate_tensile_stress_mpa_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 MPa; tendon ultimate tensile stress upper bound; from Domain VIII.3"
    )
    tendon_strain_at_failure_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10; tendon strain at failure lower bound (10% elongation); from Domain VIII.3"
    )
    tendon_strain_at_failure_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.15; tendon strain at failure upper bound (15% elongation); from Domain VIII.3"
    )
    tendon_toe_region_strain_limit_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.03; toe region upper limit (0-3% strain: collagen crimps straighten, nonlinear); from Domain VIII.3"
    )
    tendon_linear_region_strain_limit_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.08; linear elastic region upper limit (3-8% strain: Hookean behaviour); from Domain VIII.3"
    )
    achilles_tendon_csa_mm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 mm²; Achilles tendon cross-sectional area lower bound; from Domain VIII.3"
    )
    achilles_tendon_csa_mm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 mm²; Achilles tendon cross-sectional area upper bound; from Domain VIII.3"
    )
    patellar_tendon_csa_mm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 mm²; patellar tendon cross-sectional area lower bound; from Domain VIII.3"
    )
    patellar_tendon_csa_mm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="90 mm²; patellar tendon cross-sectional area upper bound; from Domain VIII.3"
    )
    tendon_collagen_content_dry_weight_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.65; tendon Type I collagen content as fraction of dry weight lower bound; from Domain VIII.3"
    )
    tendon_collagen_content_dry_weight_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85; tendon Type I collagen content as fraction of dry weight upper bound; from Domain VIII.3"
    )
    collagen_fibril_diameter_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 nm; collagen fibril diameter lower bound; from Domain VIII.3"
    )
    collagen_fibril_diameter_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="300 nm; collagen fibril diameter upper bound (larger = stronger); from Domain VIII.3"
    )
    collagen_crimp_period_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~60–200 µm; collagen crimp period (helical wave wavelength in resting tendon); from Domain VIII.3"
    )
    tendon_collagen_turnover_fraction_per_day: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.01–0.02; basal tendon collagen turnover rate per day (1-2%); from Domain VIII.3"
    )

    # ── Stretch-Shortening Cycle (SSC) — Elastic Energy ──────────────────────
    # SSC: eccentric loading stores elastic energy in tendons → released during concentric phase
    # Energy return reduces metabolic cost; SSC potentiation: higher force than pure concentric
    ssc_energy_storage_achilles_per_step_j_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35 J; Achilles tendon elastic energy stored per running step lower bound; from Domain VIII.4 / Matrix 11.4"
    )
    ssc_energy_storage_achilles_per_step_j_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="55 J; Achilles tendon elastic energy stored per running step upper bound; from Domain VIII.4"
    )
    ssc_achilles_elongation_running_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 mm; Achilles tendon elongation during running lower bound (stance phase); from Domain VIII.4"
    )
    ssc_achilles_elongation_running_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6 mm; Achilles tendon elongation during running upper bound; from Domain VIII.4"
    )
    ssc_achilles_strain_running_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05; Achilles tendon strain during running lower bound (5%); from Domain VIII.4"
    )
    ssc_achilles_strain_running_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.08; Achilles tendon strain during running upper bound (8%; near linear-to-failure transition); from Domain VIII.4"
    )
    ssc_elastic_energy_return_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.35; fraction of stored elastic energy returned during concentric phase lower bound; from Domain VIII.4"
    )
    ssc_elastic_energy_return_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.60; fraction of stored elastic energy returned upper bound (hysteresis ~10-20%); from Domain VIII.4"
    )
    ssc_force_potentiation_vs_concentric_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.20; SSC force potentiation above pure concentric lower bound (+20%); from Domain VIII.4"
    )
    ssc_force_potentiation_vs_concentric_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; SSC force potentiation upper bound (+50%); from Domain VIII.4"
    )
    ssc_critical_stretch_velocity_cm_per_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~3 cm/s; minimum stretch velocity above which SSC elastic recoil is mechanically significant; from Domain VIII.4"
    )
    plantar_fascia_energy_storage_per_step_j_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 J; plantar fascia elastic energy storage per running step lower bound; from Domain VIII.4"
    )
    plantar_fascia_energy_storage_per_step_j_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 J; plantar fascia elastic energy storage per running step upper bound; from Domain VIII.4"
    )
    tendon_hysteresis_energy_loss_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10–0.15; tendon energy loss (heat) per loading-unloading cycle (hysteresis); from Domain VIII.4"
    )

    # ── Lever Arm Biomechanics ─────────────────────────────────────────────────
    # Mechanical advantage = muscle moment arm / external load moment arm
    # Small muscle moment arms → force amplification at cost of speed/excursion
    achilles_tendon_moment_arm_cm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4 cm; Achilles tendon moment arm at ankle lower bound; from Domain VIII.5 / Matrix 11.5"
    )
    achilles_tendon_moment_arm_cm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6 cm; Achilles tendon moment arm at ankle upper bound; from Domain VIII.5"
    )
    patellar_tendon_moment_arm_cm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4 cm; patellar tendon moment arm at knee lower bound; from Domain VIII.5"
    )
    patellar_tendon_moment_arm_cm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 cm; patellar tendon moment arm at knee upper bound; from Domain VIII.5"
    )
    mechanical_advantage_ankle_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.15; ankle mechanical advantage (Achilles MA / foot length) lower bound; from Domain VIII.5"
    )
    mechanical_advantage_ankle_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.25; ankle mechanical advantage upper bound; from Domain VIII.5"
    )
    gear_ratio_limb_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05; muscle-to-joint gear ratio lower bound (small MA → high speed amplification); from Domain VIII.5"
    )
    gear_ratio_limb_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30; muscle-to-joint gear ratio upper bound; from Domain VIII.5"
    )

    # ── Bone Material Properties ───────────────────────────────────────────────
    # Wolff's law: bone architecture adapts to habitual mechanical loading
    cortical_bone_young_modulus_gpa_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 GPa; cortical bone Young's elastic modulus lower bound; from Domain VIII.5"
    )
    cortical_bone_young_modulus_gpa_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 GPa; cortical bone Young's elastic modulus upper bound; from Domain VIII.5"
    )
    cortical_bone_tensile_strength_mpa_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 MPa; cortical bone ultimate tensile strength lower bound; from Domain VIII.5"
    )
    cortical_bone_tensile_strength_mpa_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="180 MPa; cortical bone ultimate tensile strength upper bound; from Domain VIII.5"
    )
    cortical_bone_compressive_strength_mpa_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="130 MPa; cortical bone compressive strength lower bound; from Domain VIII.5"
    )
    cortical_bone_compressive_strength_mpa_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 MPa; cortical bone compressive strength upper bound; from Domain VIII.5"
    )
    trabecular_bone_young_modulus_gpa_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 GPa; trabecular bone Young's modulus lower bound (highly variable with density); from Domain VIII.5"
    )
    trabecular_bone_young_modulus_gpa_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 GPa; trabecular bone Young's modulus upper bound; from Domain VIII.5"
    )
    bmd_lumbar_spine_g_per_cm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.9 g/cm²; bone mineral density lumbar spine lower bound of normal (DXA); from Domain VIII.5"
    )
    bmd_lumbar_spine_g_per_cm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.4 g/cm²; bone mineral density lumbar spine upper bound; from Domain VIII.5"
    )
    cortical_bone_turnover_fraction_per_year: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.02–0.10; cortical bone turnover rate (2-10% per year; coupled remodelling via BMUs); from Domain VIII.5"
    )
    trabecular_bone_turnover_fraction_per_year: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.20–0.30; trabecular bone turnover rate (20-30% per year; higher surface-to-volume); from Domain VIII.5"
    )
    bone_safety_factor_femoral_shaft: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~6–10; bone safety factor (failure load / habitual load) for femoral shaft in walking; from Domain VIII.5"
    )

    # ── Cartilage Properties ───────────────────────────────────────────────────
    # Articular cartilage: viscoelastic, avascular; compressive stiffness via proteoglycan swelling
    cartilage_compressive_modulus_mpa_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 MPa; articular cartilage aggregate compressive modulus lower bound; from Domain VIII.5"
    )
    cartilage_compressive_modulus_mpa_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 MPa; articular cartilage aggregate compressive modulus upper bound; from Domain VIII.5"
    )
    cartilage_water_content_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.65–0.80; articular cartilage water content fraction (hydrostatic pressurisation resists compression); from Domain VIII.5"
    )
    cartilage_thickness_knee_mm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2 mm; articular cartilage thickness at knee lower bound; from Domain VIII.5"
    )
    cartilage_thickness_knee_mm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6 mm; articular cartilage thickness at knee upper bound (weight-bearing areas); from Domain VIII.5"
    )
