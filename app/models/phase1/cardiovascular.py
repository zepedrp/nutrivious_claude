from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesCardiovascular(Base):
    """
    Phase 1 — Cardiovascular transport ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 7 / Matrices 7.1–7.5 (cardiac output, stroke volume, HR, O2 delivery, blood volume);
      Domain IV.1 (Fick equation decomposition); Domain IV.2 (Frank-Starling / ventricular mechanics);
      Domain IV.3 (vascular resistance / pressure limits); Domain IV.4 (capillary microcirculation);
      Domain IV.5 (hemoglobin / O2 carrying capacity); Domain IV.6 (coronary reserve).

    Key equations encoded:
      Q = HR × SV                                   [cardiac output identity]
      VO2max = Q × (CaO2 − CvO2)                   [Fick equation]
      CaO2 = 1.34 × [Hb] × SaO2 + 0.003 × PaO2   [arterial O2 content, Hüfner constant]
      DO2 = Q × CaO2                                [systemic O2 delivery]
      SVR = (MAP − CVP) / Q × 80                   [systemic vascular resistance, dyne·s·cm⁻⁵]

    Units:
      _l_per_min   = L / min
      _ml_per_beat = mL / beat
      _bpm         = beats per minute
      _mmhg        = mmHg
      _ml_per_kg   = mL / kg body mass
      _ml_per_dl   = mL / 100 mL blood
      _g_per_dl    = g / 100 mL blood
      _ml_per_min  = mL / min
      _dyne_s_cm5  = dyne · s · cm⁻⁵
      _per_mm2     = per mm²
      _fraction    = dimensionless 0–1
    """

    __tablename__ = "species_cardiovascular"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="cardiovascular")

    # ── Cardiac Output (Q = HR × SV) ──────────────────────────────────────────
    # Q ceiling is the single most important cardiovascular determinant of VO2max
    # Elite endurance athletes (cross-country skiers, cyclists) hold the absolute record
    cardiac_output_basal_l_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5.0 L/min; resting cardiac output (seated); from Domain IV.1 / Matrix 7.1"
    )
    cardiac_output_max_untrained_l_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 L/min; maximum cardiac output in untrained adults; from Domain IV.1"
    )
    cardiac_output_max_trained_l_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40 L/min; maximum cardiac output in elite endurance athletes; from Domain IV.1"
    )
    cardiac_output_absolute_record_l_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~42–45 L/min; absolute species record cardiac output (elite cross-country skiers/cyclists); from Domain IV.1"
    )
    cardiac_output_theoretical_ceiling_l_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~45 L/min; theoretical Q ceiling constrained by ventricular filling time and myocardial O2; from Domain IV.1"
    )

    # ── Heart Rate (HR) ────────────────────────────────────────────────────────
    # HR_max = 220 − age (Haskell & Fox 1970); refined: 208 − 0.7×age (Tanaka 2001)
    # Trained sinus bradycardia: autonomic remodelling + enhanced parasympathetic tone
    hr_max_absolute_bpm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~220 bpm; absolute maximum HR recorded in young athletes; from Domain IV.1"
    )
    hr_max_formula_age_coefficient_haskell: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="220.0; Haskell & Fox (1970) intercept: HR_max = 220 − age; from Domain IV.1"
    )
    hr_max_formula_age_slope_tanaka: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.7; Tanaka et al. (2001) slope: HR_max = 208 − 0.7 × age; more accurate ±3 bpm; from Domain IV.1"
    )
    hr_max_formula_intercept_tanaka: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="208.0; Tanaka et al. (2001) intercept; from Domain IV.1"
    )
    hr_resting_untrained_bpm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 bpm; resting HR lower bound in untrained adults; from Domain IV.1"
    )
    hr_resting_untrained_bpm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 bpm; resting HR upper bound in untrained adults; from Domain IV.1"
    )
    hr_resting_trained_bpm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="28 bpm; resting HR minimum in elite endurance athletes (pathological if symptoms); from Domain IV.1"
    )
    hr_resting_trained_bpm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="45 bpm; resting HR upper bound in trained athletes; from Domain IV.1"
    )

    # ── Stroke Volume (SV) — Frank-Starling Mechanism ─────────────────────────
    # SV = EDV − ESV; governed by preload (EDV), afterload (MAP), contractility (inotropy)
    # Frank-Starling: ↑ EDV → ↑ sarcomere length → ↑ Ca²⁺ sensitivity → ↑ force
    sv_resting_untrained_ml_per_beat: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~70 mL/beat; resting stroke volume untrained adult; from Domain IV.2"
    )
    sv_resting_trained_ml_per_beat_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 mL/beat; resting SV trained endurance athlete lower bound; from Domain IV.2"
    )
    sv_resting_trained_ml_per_beat_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="120 mL/beat; resting SV trained endurance athlete upper bound; from Domain IV.2"
    )
    sv_max_exercise_elite_ml_per_beat_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 mL/beat; maximum SV during exercise in elite athletes lower bound; from Domain IV.2"
    )
    sv_max_exercise_elite_ml_per_beat_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="220 mL/beat; maximum SV during exercise in elite athletes upper bound; from Domain IV.2"
    )
    sv_absolute_record_ml_per_beat: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~227 mL/beat; absolute species record SV (elite endurance athlete); from Domain IV.2"
    )
    edv_resting_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="120 mL; end-diastolic volume at rest lower bound; from Domain IV.2"
    )
    edv_resting_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="150 mL; end-diastolic volume at rest upper bound; from Domain IV.2"
    )
    edv_max_exercise_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 mL; end-diastolic volume at peak exercise lower bound (Frank-Starling stretch); from Domain IV.2"
    )
    edv_max_exercise_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="220 mL; end-diastolic volume at peak exercise upper bound; from Domain IV.2"
    )
    esv_resting_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 mL; end-systolic volume at rest lower bound; from Domain IV.2"
    )
    esv_resting_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 mL; end-systolic volume at rest upper bound; from Domain IV.2"
    )
    esv_min_exercise_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~30 mL; minimum ESV at peak exercise (maximal inotropy + sympathetic drive); from Domain IV.2"
    )

    # ── Ejection Fraction ──────────────────────────────────────────────────────
    ejection_fraction_normal_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.55; ejection fraction lower bound of normal range (EF = SV/EDV); from Domain IV.2"
    )
    ejection_fraction_normal_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.70; ejection fraction upper bound of normal range; from Domain IV.2"
    )
    ejection_fraction_elite_max: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.80; ejection fraction ceiling in elite athletes at peak exercise (increased inotropy); from Domain IV.2"
    )
    ejection_fraction_hf_threshold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.40; ejection fraction below which heart failure with reduced EF (HFrEF) is defined; from Domain IV.2"
    )

    # ── Fick Equation — O2 Delivery and Extraction ────────────────────────────
    # VO2max = Q × (CaO2 − CvO2); CaO2 = 1.34 × [Hb] × SaO2 + 0.003 × PaO2
    # Hüfner constant: 1.34 mL O2/g Hb; theoretical max = 1.39 mL O2/g Hb
    hufner_constant_ml_o2_per_g_hb: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.34 mL O2/g Hb; Hüfner constant (empirical); theoretical max = 1.39; from Domain IV.1"
    )
    cao2_max_ml_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20–21 mL O2/100 mL blood; maximum arterial O2 content (Hb 15 g/dL × 1.34 × SaO2 0.99); from Domain IV.1"
    )
    cvo2_min_exercise_ml_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2–4 mL O2/100 mL blood; minimum mixed venous O2 content at VO2max; from Domain IV.1"
    )
    av_o2_diff_resting_ml_per_dl: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5–6 mL O2/100 mL blood; arteriovenous O2 difference at rest; from Domain IV.1"
    )
    av_o2_diff_max_exercise_ml_per_dl_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 mL O2/100 mL blood; a-vO2 difference at VO2max lower bound; from Domain IV.1"
    )
    av_o2_diff_max_exercise_ml_per_dl_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="18 mL O2/100 mL blood; a-vO2 difference at VO2max upper bound (trained > untrained); from Domain IV.1"
    )
    do2_max_l_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~8 L O2/min; maximum systemic O2 delivery (DO2 = Q × CaO2 = 40 L/min × 200 mL/L); from Domain IV.1"
    )

    # ── Blood Volume and Plasma Volume ─────────────────────────────────────────
    # Blood volume expansion is an independent adaptation to endurance training
    # Plasma volume expands faster (days) than red cell mass (weeks/months)
    total_blood_volume_ml_per_kg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 mL/kg; total blood volume lower bound (untrained male); from Domain IV.5"
    )
    total_blood_volume_ml_per_kg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 mL/kg; total blood volume upper bound (untrained male); from Domain IV.5"
    )
    total_blood_volume_elite_ml_per_kg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="90 mL/kg; total blood volume lower bound in elite endurance athletes; from Domain IV.5"
    )
    total_blood_volume_elite_ml_per_kg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="110 mL/kg; total blood volume upper bound in elite endurance athletes; from Domain IV.5"
    )
    plasma_volume_fraction_of_blood: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.55–0.60; plasma fraction of total blood volume (1 − hematocrit); from Domain IV.5"
    )
    plasma_volume_expansion_training_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="300 mL; chronic training-induced plasma volume expansion lower bound; from Domain IV.5"
    )
    plasma_volume_expansion_training_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="500 mL; chronic training-induced plasma volume expansion upper bound; from Domain IV.5"
    )
    plasma_volume_acute_expansion_exercise_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.15–0.20; acute plasma volume expansion fraction immediately post-exercise (hemoconcentration reversal); from Domain IV.5"
    )

    # ── Hemoglobin and O2 Carrying Capacity ───────────────────────────────────
    # Hematocrit and [Hb] set the ceiling on CaO2 and thus VO2max via Fick
    hematocrit_male_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.40; male hematocrit lower bound of normal range; from Domain IV.5"
    )
    hematocrit_male_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; male hematocrit upper bound of normal range; from Domain IV.5"
    )
    hemoglobin_male_g_per_dl_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="13.5 g/dL; hemoglobin concentration male lower bound of normal; from Domain IV.5"
    )
    hemoglobin_male_g_per_dl_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="17.5 g/dL; hemoglobin concentration male upper bound of normal; from Domain IV.5"
    )
    hemoglobin_female_g_per_dl_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12.0 g/dL; hemoglobin concentration female lower bound; from Domain IV.5"
    )
    hemoglobin_female_g_per_dl_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="16.0 g/dL; hemoglobin concentration female upper bound; from Domain IV.5"
    )
    sao2_normal_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.95; arterial O2 saturation lower bound at rest (normal); from Domain IV.1"
    )
    sao2_normal_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.99; arterial O2 saturation upper bound at rest; from Domain IV.1"
    )
    sao2_exercise_induced_desaturation_threshold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.91; SaO2 below which exercise-induced arterial hypoxaemia (EIAH) is defined (>60% VO2max); from Domain IV.1"
    )

    # ── Blood Pressure — Systemic Arterial ────────────────────────────────────
    # Systolic: LV ejection pressure; diastolic: arterial wall recoil / resistance
    # Exercise systolic rises with Q; diastolic may fall (peripheral vasodilation)
    bp_systolic_resting_mmhg_normal_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="110 mmHg; resting systolic BP lower bound of normal; from Domain IV.3"
    )
    bp_systolic_resting_mmhg_normal_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="130 mmHg; resting systolic BP upper bound of normal; from Domain IV.3"
    )
    bp_diastolic_resting_mmhg_normal_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 mmHg; resting diastolic BP lower bound of normal; from Domain IV.3"
    )
    bp_diastolic_resting_mmhg_normal_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="85 mmHg; resting diastolic BP upper bound of normal; from Domain IV.3"
    )
    bp_systolic_max_exercise_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 mmHg; maximum systolic BP during maximal exercise lower bound (healthy); from Domain IV.3"
    )
    bp_systolic_max_exercise_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="240 mmHg; maximum systolic BP during maximal exercise upper bound; from Domain IV.3"
    )
    bp_systolic_hypertensive_response_threshold_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="250 mmHg; systolic exercise BP above which hypertensive response is pathological; from Domain IV.3"
    )
    bp_diastolic_min_exercise_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~60–70 mmHg; minimum diastolic BP during maximal aerobic exercise (peripheral vasodilation); from Domain IV.3"
    )
    map_resting_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 mmHg; mean arterial pressure at rest lower bound (MAP = DBP + 1/3×PP); from Domain IV.3"
    )
    map_resting_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 mmHg; mean arterial pressure at rest upper bound; from Domain IV.3"
    )

    # ── Pulmonary Arterial Pressure ────────────────────────────────────────────
    mpap_resting_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="14 mmHg; mean pulmonary artery pressure (MPAP) at rest lower bound; from Domain IV.3"
    )
    mpap_resting_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="18 mmHg; MPAP at rest upper bound; from Domain IV.3"
    )
    mpap_max_exercise_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 mmHg; MPAP at maximal exercise lower bound (driven by high Q); from Domain IV.3"
    )
    mpap_max_exercise_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 mmHg; MPAP at maximal exercise upper bound (elite athletes); from Domain IV.3"
    )
    pulmonary_hypertension_resting_threshold_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 mmHg; MPAP threshold for pulmonary hypertension diagnosis at rest; from Domain IV.3"
    )

    # ── Systemic Vascular Resistance (SVR) ────────────────────────────────────
    # SVR = (MAP − CVP) / Q × 80 [dyne·s·cm⁻⁵]
    # Exercise: skeletal muscle vasodilation dramatically reduces SVR
    svr_resting_dyne_s_cm5_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="800 dyne·s·cm⁻⁵; SVR at rest lower bound; from Domain IV.3"
    )
    svr_resting_dyne_s_cm5_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1200 dyne·s·cm⁻⁵; SVR at rest upper bound; from Domain IV.3"
    )
    svr_max_exercise_dyne_s_cm5_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 dyne·s·cm⁻⁵; SVR at VO2max lower bound (massive skeletal muscle vasodilation); from Domain IV.3"
    )
    svr_max_exercise_dyne_s_cm5_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="300 dyne·s·cm⁻⁵; SVR at VO2max upper bound; from Domain IV.3"
    )

    # ── Capillary Density — Skeletal Muscle Microcirculation ─────────────────
    # Capillary density sets the diffusion distance for O2 from RBC to mitochondria
    # Training increases capillary-to-fiber ratio (angiogenesis via VEGF)
    capillary_density_untrained_per_mm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="300 capillaries/mm²; skeletal muscle capillary density untrained lower bound; from Domain IV.4"
    )
    capillary_density_untrained_per_mm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="600 capillaries/mm²; skeletal muscle capillary density untrained upper bound; from Domain IV.4"
    )
    capillary_density_trained_per_mm2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="700 capillaries/mm²; skeletal muscle capillary density trained lower bound; from Domain IV.4"
    )
    capillary_density_trained_per_mm2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="900 capillaries/mm²; skeletal muscle capillary density trained upper bound; from Domain IV.4"
    )
    capillary_to_fiber_ratio_untrained_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.5; capillary-to-fiber ratio untrained lower bound; from Domain IV.4"
    )
    capillary_to_fiber_ratio_untrained_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.5; capillary-to-fiber ratio untrained upper bound; from Domain IV.4"
    )
    capillary_to_fiber_ratio_trained_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.5; capillary-to-fiber ratio trained endurance athlete lower bound; from Domain IV.4"
    )
    capillary_to_fiber_ratio_trained_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0; capillary-to-fiber ratio trained endurance athlete upper bound; from Domain IV.4"
    )
    o2_diffusion_distance_capillary_to_mito_um_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~20 µm; O2 diffusion distance capillary wall to nearest mitochondrion lower bound; from Domain IV.4"
    )
    o2_diffusion_distance_capillary_to_mito_um_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40 µm; O2 diffusion distance upper bound (untrained, further mitochondria); from Domain IV.4"
    )
    capillary_transit_time_resting_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 s; red blood cell capillary transit time at rest lower bound; from Domain IV.4"
    )
    capillary_transit_time_resting_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 s; RBC capillary transit time at rest upper bound; from Domain IV.4"
    )
    capillary_transit_time_min_exercise_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.3–0.5 s; minimum RBC transit time at VO2max (risks O2 equilibration failure if < 0.25 s); from Domain IV.4"
    )

    # ── Coronary Reserve ──────────────────────────────────────────────────────
    # Coronary blood flow: 5% of Q at rest; vasodilates up to 4-5× (coronary flow reserve)
    coronary_flow_basal_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~250–300 mL/min; basal coronary blood flow (~5% cardiac output); from Domain IV.6"
    )
    coronary_flow_max_exercise_ml_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1000 mL/min; maximum coronary blood flow at VO2max lower bound; from Domain IV.6"
    )
    coronary_flow_max_exercise_ml_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1200 mL/min; maximum coronary blood flow at VO2max upper bound; from Domain IV.6"
    )
    coronary_flow_reserve_fold_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.0; coronary flow reserve (max/basal) lower bound in healthy subjects; from Domain IV.6"
    )
    coronary_flow_reserve_fold_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0; coronary flow reserve upper bound; reduced in CAD (<2.0 = diagnostic threshold); from Domain IV.6"
    )
    coronary_o2_extraction_fraction_max: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.75; maximum O2 extraction by myocardium (~75%); heart is near-maximum extractor at rest; from Domain IV.6"
    )

    # ── Venous Return / Central Venous Pressure ───────────────────────────────
    cvp_resting_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2 mmHg; central venous pressure at rest lower bound; from Domain IV.3"
    )
    cvp_resting_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8 mmHg; central venous pressure at rest upper bound; from Domain IV.3"
    )
    venous_capacitance_fraction_total_blood_volume: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.64; fraction of total blood volume in venous capacitance vessels at rest (64%); from Domain IV.3"
    )

    # ── Aortic Elastic Modulus / Wall Mechanics ────────────────────────────────
    # Windkessel effect: aortic compliance buffers LV pulsatile ejection
    # Arterial stiffness increases with age/atherosclerosis; training preserves compliance
    aortic_compliance_ml_per_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.2 mL/mmHg; aortic compliance lower bound (young healthy); from Domain IV.3"
    )
    aortic_compliance_ml_per_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 mL/mmHg; aortic compliance upper bound; from Domain IV.3"
    )
    pulse_wave_velocity_young_m_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 m/s; aortic pulse wave velocity (PWV) lower bound in young healthy adults; from Domain IV.3"
    )
    pulse_wave_velocity_young_m_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="7 m/s; PWV upper bound young healthy (marker of arterial stiffness; >10 m/s = elevated risk); from Domain IV.3"
    )

    # ── Endothelial Function ───────────────────────────────────────────────────
    # Flow-mediated dilation (FMD): NO-dependent vasodilation of brachial artery
    # eNOS activation: shear stress → Ca²⁺/calmodulin → eNOS → NO → cGMP → vasodilation
    fmd_brachial_artery_normal_fraction_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05; flow-mediated dilation (FMD) of brachial artery lower bound of normal (5%); from Domain IV.4"
    )
    fmd_brachial_artery_normal_fraction_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.15; FMD upper bound of normal (15%); trained athletes > sedentary; from Domain IV.4"
    )
    fmd_endothelial_dysfunction_threshold_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.04; FMD below which endothelial dysfunction is defined (<4%); from Domain IV.4"
    )
    no_half_life_endothelial_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–2 s; nitric oxide (NO) half-life at endothelial surface (rapidly scavenged by Hb); from Domain IV.4"
    )
    enos_km_arginine_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~3 µM; K_m^eNOS for L-arginine; near-saturated at physiological [arginine] ~100 µM; from Domain IV.4"
    )
