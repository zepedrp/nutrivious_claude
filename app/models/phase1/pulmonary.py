from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesPulmonary(Base):
    """
    Phase 1 — Pulmonary mechanics and gas exchange ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 8 / Matrices 8.1–8.5 (ventilation, lung volumes, diffusion, thresholds, O2 cost);
      Domain V.1 (alveolar gas equation / PAO2-PACO2); Domain V.2 (DLCO/DLO2 diffusion);
      Domain V.3 (ventilatory thresholds VT1/VT2); Domain V.4 (work of breathing / diaphragm);
      Domain V.5 (pulmonary circulation / transit time).

    Key equations encoded:
      PAO2 = FIO2 × (PB − PH2O) − PACO2 / R         [alveolar gas equation; R = respiratory quotient]
      DLCO = VO_CO / (PACO − PcCO)                    [Fick law for CO diffusion]
      1/DLO2 = 1/DM + 1/(θO2 × Vc)                   [Roughton-Forster: membrane + RBC conductance]
      VE = VT × RR                                     [minute ventilation identity]
      VD/VT = (PaCO2 − PECO2) / PaCO2                 [Bohr dead-space equation]

    Units:
      _l          = litres
      _l_per_min  = L / min
      _ml_min_mmhg= mL / min / mmHg (diffusing capacity)
      _mmhg       = mmHg
      _bpm        = breaths per minute
      _fraction   = dimensionless 0–1
      _ml_per_min = mL / min
      _w          = watts
    """

    __tablename__ = "species_pulmonary"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="pulmonary")

    # ── Static Lung Volumes ────────────────────────────────────────────────────
    # TLC = VC + RV; FRC = ERV + RV; spirometry-defined ceilings
    tlc_male_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 L; total lung capacity male lower bound; from Domain V.1 / Matrix 8.1"
    )
    tlc_male_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8.0 L; total lung capacity male upper bound (tall endurance athletes); from Domain V.1"
    )
    tlc_female_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.5 L; total lung capacity female lower bound; from Domain V.1"
    )
    tlc_female_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6.5 L; total lung capacity female upper bound; from Domain V.1"
    )
    fvc_male_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.5 L; forced vital capacity male lower bound (predicted); from Domain V.1"
    )
    fvc_male_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6.5 L; FVC male upper bound (tall endurance athletes may exceed 7 L); from Domain V.1"
    )
    fvc_female_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.5 L; forced vital capacity female lower bound; from Domain V.1"
    )
    fvc_female_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5.0 L; FVC female upper bound; from Domain V.1"
    )
    fev1_fvc_ratio_normal_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.70; FEV1/FVC ratio lower bound of normal (below = obstructive pattern); from Domain V.1"
    )
    fev1_fvc_ratio_normal_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85; FEV1/FVC ratio upper bound of normal; from Domain V.1"
    )
    rv_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 L; residual volume lower bound; from Domain V.1"
    )
    rv_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.5 L; residual volume upper bound; from Domain V.1"
    )
    frc_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.5 L; functional residual capacity lower bound; from Domain V.1"
    )
    frc_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.5 L; functional residual capacity upper bound; from Domain V.1"
    )
    anatomical_dead_space_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~150 mL; anatomical dead space (conducting airways; no gas exchange); from Domain V.1"
    )
    vd_vt_ratio_rest: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.30; physiological VD/VT ratio at rest (Bohr equation); from Domain V.1"
    )
    vd_vt_ratio_max_exercise: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.10–0.15; VD/VT ratio at VO2max (increased VT reduces dead-space fraction); from Domain V.1"
    )

    # ── Dynamic Ventilation — VE, VT, RR ──────────────────────────────────────
    # VE = VT × RR; at VO2max, VE is the primary constraint only in elite athletes
    ve_resting_l_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 L/min; minute ventilation at rest lower bound; from Domain V.1"
    )
    ve_resting_l_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8 L/min; minute ventilation at rest upper bound; from Domain V.1"
    )
    ve_max_untrained_l_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 L/min; VE max untrained adults lower bound; from Domain V.1 / Matrix 8.2"
    )
    ve_max_untrained_l_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="150 L/min; VE max untrained adults upper bound; from Domain V.1"
    )
    ve_max_elite_l_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="180 L/min; VE max elite endurance athletes lower bound; from Domain V.1"
    )
    ve_max_elite_l_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="220 L/min; VE max elite endurance athletes upper bound; from Domain V.1"
    )
    ve_max_theoretical_ceiling_l_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~240 L/min; theoretical VE ceiling constrained by respiratory muscle power; from Domain V.1"
    )
    vt_resting_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5 L; tidal volume at rest; from Domain V.1"
    )
    vt_max_exercise_elite_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0 L; maximum tidal volume during exercise elite athletes lower bound; from Domain V.1"
    )
    vt_max_exercise_elite_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.5 L; maximum tidal volume during exercise elite athletes upper bound; from Domain V.1"
    )
    rr_resting_bpm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12 breaths/min; respiratory rate at rest lower bound; from Domain V.1"
    )
    rr_resting_bpm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="16 breaths/min; respiratory rate at rest upper bound; from Domain V.1"
    )
    rr_max_exercise_bpm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 breaths/min; respiratory rate at VO2max lower bound; from Domain V.1"
    )
    rr_max_exercise_bpm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 breaths/min; respiratory rate at VO2max upper bound; from Domain V.1"
    )

    # ── Alveolar Gas Partial Pressures ────────────────────────────────────────
    # PAO2 = FIO2 × (PB − PH2O) − PACO2/R; PB = 760 mmHg at sea level
    # PACO2 ≈ PaCO2 under normal conditions (alveolar CO2 assumption)
    pb_sea_level_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="760 mmHg; barometric pressure at sea level; from Domain V.1"
    )
    ph2o_body_temp_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="47 mmHg; water vapour pressure at 37°C body temperature; from Domain V.1"
    )
    fio2_atmospheric: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.2093; inspired O2 fraction in atmospheric air; from Domain V.1"
    )
    respiratory_quotient_mixed_diet: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85; respiratory quotient (R = VCO2/VO2) on mixed diet; from Domain V.1"
    )
    respiratory_quotient_pure_fat: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.71; respiratory quotient oxidising pure fat (palmitate); from Domain V.1"
    )
    respiratory_quotient_pure_cho: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.00; respiratory quotient oxidising pure carbohydrate; from Domain V.1"
    )
    pao2_rest_sea_level_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~100 mmHg; alveolar PO2 at rest sea level (FIO2 0.209 × (760-47) − 40/0.85); from Domain V.1"
    )
    paco2_rest_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40 mmHg; alveolar PCO2 at rest (= PaCO2 under normal conditions); from Domain V.1"
    )
    paco2_max_exercise_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35 mmHg; PACO2 at VO2max lower bound (hyperventilation-induced hypocapnia); from Domain V.1"
    )
    paco2_max_exercise_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="38 mmHg; PACO2 at VO2max upper bound; from Domain V.1"
    )
    pao2_max_exercise_sea_level_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~105 mmHg; PAO2 at VO2max sea level lower bound (hyperventilation increases PAO2); from Domain V.1"
    )
    pao2_max_exercise_sea_level_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~115 mmHg; PAO2 at VO2max sea level upper bound; from Domain V.1"
    )
    pa_a_gradient_rest_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 mmHg; alveolar-arterial O2 gradient (A-a DO2) at rest lower bound; from Domain V.1"
    )
    pa_a_gradient_rest_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 mmHg; A-a DO2 at rest upper bound (young healthy); from Domain V.1"
    )
    pa_a_gradient_max_exercise_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 mmHg; A-a DO2 at VO2max lower bound (normal widening); from Domain V.1"
    )
    pa_a_gradient_max_exercise_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35 mmHg; A-a DO2 at VO2max upper bound (EIAH threshold in elite); from Domain V.1"
    )
    pao2_eiah_threshold_mmhg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~75–85 mmHg; PaO2 at VO2max in elite athletes with exercise-induced arterial hypoxaemia (EIAH); from Domain V.1"
    )

    # ── Pulmonary Diffusing Capacity (DLCO / DLO2) ────────────────────────────
    # DLCO = total conductance = 1/(1/DM + 1/(θO2 × Vc)); Roughton-Forster model
    # Exercise increases DLCO by recruiting alveolar capillaries and dilating existing ones
    dlco_rest_ml_min_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 mL/min/mmHg; DLCO at rest lower bound (male); from Domain V.2 / Matrix 8.3"
    )
    dlco_rest_ml_min_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35 mL/min/mmHg; DLCO at rest upper bound (male); from Domain V.2"
    )
    dlco_max_exercise_ml_min_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 mL/min/mmHg; DLCO at VO2max lower bound (3× increase from rest); from Domain V.2"
    )
    dlco_max_exercise_ml_min_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 mL/min/mmHg; DLCO at VO2max upper bound; from Domain V.2"
    )
    dlo2_to_dlco_ratio: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.23; DLO2/DLCO conversion factor (ratio of O2:CO diffusion coefficients); from Domain V.2"
    )
    dm_fraction_of_dlco: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.60–0.80; membrane conductance (DM) as fraction of total DLCO; from Domain V.2"
    )
    pulmonary_capillary_blood_volume_rest_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 mL; pulmonary capillary blood volume (Vc) at rest lower bound; from Domain V.2"
    )
    pulmonary_capillary_blood_volume_rest_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 mL; Vc at rest upper bound; from Domain V.2"
    )
    pulmonary_capillary_blood_volume_exercise_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 mL; Vc at VO2max lower bound (recruitment + dilation of alveolar capillaries); from Domain V.2"
    )
    pulmonary_capillary_blood_volume_exercise_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="250 mL; Vc at VO2max upper bound; from Domain V.2"
    )
    pulmonary_capillary_transit_time_rest_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.75 s; RBC transit time through pulmonary capillary at rest; O2 equilibration needs ~0.25 s; from Domain V.2"
    )
    pulmonary_capillary_transit_time_max_exercise_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.25 s; minimum RBC transit time at VO2max lower bound; borderline O2 equilibration failure; from Domain V.2"
    )
    pulmonary_capillary_transit_time_max_exercise_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.40 s; minimum RBC transit time at VO2max upper bound; from Domain V.2"
    )
    o2_equilibration_time_required_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.25 s; time required for complete O2 equilibration between alveolus and RBC; below this → diffusion limitation; from Domain V.2"
    )

    # ── Ventilatory Thresholds (VT1 / VT2) ────────────────────────────────────
    # VT1 (first threshold): onset of disproportionate VCO2 rise; ≈ lactate threshold 1
    # VT2 (second threshold / respiratory compensation point): isocapnic buffering ends
    # Identified by V-slope method, VE/VO2 and VE/VCO2 crossovers, PETCO2 plateau
    vt1_fraction_vo2max_untrained_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.45; VT1 as fraction of VO2max lower bound (untrained); from Domain V.3 / Matrix 8.4"
    )
    vt1_fraction_vo2max_untrained_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.60; VT1 as fraction of VO2max upper bound (untrained); from Domain V.3"
    )
    vt1_fraction_vo2max_trained_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.65; VT1 as fraction of VO2max lower bound (trained endurance athlete); from Domain V.3"
    )
    vt1_fraction_vo2max_trained_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.80; VT1 as fraction of VO2max upper bound (trained); from Domain V.3"
    )
    vt2_fraction_vo2max_untrained_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.65; VT2 (respiratory compensation point) as fraction of VO2max lower bound (untrained); from Domain V.3"
    )
    vt2_fraction_vo2max_untrained_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.75; VT2 as fraction of VO2max upper bound (untrained); from Domain V.3"
    )
    vt2_fraction_vo2max_elite_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85; VT2 as fraction of VO2max lower bound (elite); from Domain V.3"
    )
    vt2_fraction_vo2max_elite_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.93; VT2 as fraction of VO2max upper bound (elite); from Domain V.3"
    )
    rer_at_vt2_threshold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="≥1.10; respiratory exchange ratio (VCO2/VO2) at VT2; confirmed by ≥1.10 criterion; from Domain V.3"
    )
    petco2_plateau_at_vt1_mmhg_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 mmHg; end-tidal PCO2 plateau at VT1 lower bound (isocapnic buffering zone onset); from Domain V.3"
    )
    petco2_plateau_at_vt1_mmhg_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="45 mmHg; PETCO2 plateau at VT1 upper bound; from Domain V.3"
    )
    ve_vo2_at_vt1_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="22; ventilatory equivalent for O2 (VE/VO2) at VT1 lower bound; from Domain V.3"
    )
    ve_vo2_at_vt1_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="28; VE/VO2 at VT1 upper bound; onset of VE/VO2 rise marks VT1; from Domain V.3"
    )
    ve_vco2_nadir_at_vt2_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25; VE/VCO2 nadir value lower bound (minimum = maximum ventilatory efficiency); from Domain V.3"
    )
    ve_vco2_nadir_at_vt2_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="32; VE/VCO2 nadir value upper bound; from Domain V.3"
    )

    # ── Oxygen Cost of Breathing (Work of Breathing) ──────────────────────────
    # O2 cost of ventilation competes with locomotor muscles for Q at VO2max
    # Diaphragm has exclusive priority via vasoconstriction of locomotor beds if fatigued
    o2_cost_breathing_rest_ml_per_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–2 mL O2/min; O2 cost of breathing at rest (1-2% of total VO2); from Domain V.4 / Matrix 8.5"
    )
    o2_cost_breathing_max_exercise_ml_per_min_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="80 mL O2/min; O2 cost of breathing at VO2max lower bound; from Domain V.4"
    )
    o2_cost_breathing_max_exercise_ml_per_min_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 mL O2/min; O2 cost of breathing at VO2max upper bound; from Domain V.4"
    )
    o2_cost_breathing_fraction_vo2max: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10–0.15; O2 cost of ventilatory muscles as fraction of VO2max at maximum exercise; from Domain V.4"
    )
    diaphragm_power_output_rest_w: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–2 W; diaphragm mechanical power output at rest; from Domain V.4"
    )
    diaphragm_power_output_max_exercise_w_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 W; diaphragm mechanical power output at VO2max lower bound; from Domain V.4"
    )
    diaphragm_power_output_max_exercise_w_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 W; diaphragm mechanical power output at VO2max upper bound; from Domain V.4"
    )
    diaphragm_fatigue_threshold_ve_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.85; fraction of maximum VE above which sustained breathing induces diaphragm fatigue (>10-15 min); from Domain V.4"
    )
    diaphragm_blood_flow_fraction_q_rest: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.005–0.010; diaphragm blood flow as fraction of Q at rest (0.5-1.0%); from Domain V.4"
    )
    diaphragm_blood_flow_fraction_q_max_exercise: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.03–0.04; diaphragm blood flow as fraction of Q at VO2max (3-4%); from Domain V.4"
    )

    # ── Expiratory Flow Limitation ────────────────────────────────────────────
    # Elite athletes may approach their maximum expiratory flow envelope at VO2max
    # (flow limitation): breathing on the descending limb of maximal flow-volume curve
    mef75_l_per_s_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6 L/s; maximum expiratory flow at 75% FVC (MEF75) lower bound; from Domain V.1"
    )
    mef75_l_per_s_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 L/s; MEF75 upper bound; from Domain V.1"
    )
    expiratory_flow_limitation_fraction_ve_max: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.90–1.00; fraction of VE_max at which elite athletes reach expiratory flow limitation boundary; from Domain V.1"
    )

    # ── Pulmonary Vascular Resistance ─────────────────────────────────────────
    pvr_rest_dyne_s_cm5_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 dyne·s·cm⁻⁵; pulmonary vascular resistance at rest lower bound; from Domain V.5"
    )
    pvr_rest_dyne_s_cm5_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 dyne·s·cm⁻⁵; PVR at rest upper bound (much lower than SVR ~1000); from Domain V.5"
    )
    pvr_max_exercise_dyne_s_cm5_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 dyne·s·cm⁻⁵; PVR at VO2max lower bound (capillary recruitment reduces resistance); from Domain V.5"
    )
    pvr_max_exercise_dyne_s_cm5_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 dyne·s·cm⁻⁵; PVR at VO2max upper bound; from Domain V.5"
    )

    # ── Bronchodilation and Airway Mechanics ──────────────────────────────────
    # Sympathetic β₂ activation during exercise dilates airways; reduces resistance
    airway_resistance_rest_cmh2o_l_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–2 cmH2O/L/s; total airway resistance at rest; from Domain V.1"
    )
    airway_resistance_max_exercise_cmh2o_l_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5–1.0 cmH2O/L/s; airway resistance at VO2max (β₂-mediated bronchodilation); from Domain V.1"
    )
    specific_airway_conductance_normal_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.10 L/s/cmH2O/L; specific airway conductance (sGaw) lower bound of normal; from Domain V.1"
    )
    specific_airway_conductance_normal_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30 L/s/cmH2O/L; sGaw upper bound of normal; from Domain V.1"
    )

    # ── Closing Volume and Alveolar Stability ─────────────────────────────────
    # CV: lung volume at which dependent airways begin to close during expiration
    # Increased CV (or CV/VC > 0.30) → air trapping / V/Q mismatch risk
    closing_volume_vc_ratio_normal_threshold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30; closing volume / VC ratio threshold above which air trapping is significant; from Domain V.1"
    )
    surfactant_surface_tension_mn_per_m: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–5 mN/m; minimum alveolar surface tension with surfactant (prevents collapse); LaPlace: P = 2T/r; from Domain V.1"
    )
