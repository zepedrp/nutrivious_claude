from typing import Optional
from pydantic import Field
from .base_metric import AgnosticMetricBase


class NeuromuscularTest(AgnosticMetricBase):
    cmj_height_cm: Optional[float] = Field(None, ge=0)
    handgrip_kg: Optional[float] = Field(None, ge=0)
    bar_velocity_m_s: Optional[float] = Field(None, ge=0)
    rfd_n_s: Optional[float] = Field(None, ge=0)
    squat_jump_height_cm: Optional[float] = Field(None, ge=0)
    reactive_strength_index: Optional[float] = Field(None, ge=0)


class NeuroendocrineSample(AgnosticMetricBase):
    cortisol_nmol_L: Optional[dict[str, float]] = None  # keys: "t0","t15","t30","t45"
    testosterone_pg_mL: Optional[float] = Field(None, ge=0)
    siga_ug_mL: Optional[float] = Field(None, ge=0)
    siga_flow_mL_min: Optional[float] = Field(None, ge=0)
    crp_mg_L: Optional[float] = Field(None, ge=0)
    ck_u_L: Optional[float] = Field(None, ge=0)
    hemoglobin_g_dL: Optional[float] = Field(None, ge=0)
    ferritin_ng_mL: Optional[float] = Field(None, ge=0)
    ferritina_afectada_por_inflamacao: Optional[bool] = None
    nlr: Optional[float] = Field(None, ge=0)


class CognitiveTest(AgnosticMetricBase):
    pvt_rt_ms: Optional[float] = Field(None, ge=0)
    pvt_lapses: Optional[int] = Field(None, ge=0)
    pvt_rt_cv_pct: Optional[float] = Field(None, ge=0)
    iaf_hz: Optional[float] = Field(None, ge=0)


class DailySubjective(AgnosticMetricBase):
    doms_score: Optional[dict[str, float]] = None  # keys: anatomical regions
    wellbeing_score: Optional[dict[str, float]] = None  # keys: "energia","humor","qualidade_sono","dor_muscular","motivacao","stress_percebido"
    rpe_score: Optional[float] = Field(None, ge=0, le=10)
    alcohol_g_etanol: float = Field(0.0, ge=0)
    perceived_stress_nrs: Optional[float] = Field(None, ge=0, le=10)
    tqr_score: Optional[float] = Field(None, ge=1, le=10)


class DailySleepSummary(AgnosticMetricBase):
    tst_hours: Optional[float] = Field(None, ge=0)
    sws_pct: Optional[float] = Field(None, ge=0, le=100)
    rem_pct: Optional[float] = Field(None, ge=0, le=100)
    efficiency_pct: Optional[float] = Field(None, ge=0, le=100)
    rmssd_first_4h_ms: Optional[float] = Field(None, ge=0)
    hrv_by_phase: Optional[dict[str, float]] = None  # keys: "sws","rem","n1_n2"
    nocturnal_hypoglycemia: Optional[bool] = None
    sleep_onset_time: Optional[str] = None
    spo2_nadir_pct: Optional[float] = Field(None, ge=0, le=100)


class SessionRecord(AgnosticMetricBase):
    power_output_watts: Optional[float] = Field(None, ge=0)
    cadence_rpm: Optional[float] = Field(None, ge=0)
    session_duration_secs: Optional[float] = Field(None, ge=0)
    session_type: Optional[str] = None  # "endurance"|"strength"|"hiit"|"recovery"
    distance_meters: Optional[float] = Field(None, ge=0)
    tss_estimate: Optional[float] = Field(None, ge=0)


class HydrationAndNutritionRecord(AgnosticMetricBase):
    meal_timestamp: Optional[str] = None
    carbohydrate_grams: Optional[float] = Field(None, ge=0)
    protein_grams: Optional[float] = Field(None, ge=0)
    fat_grams: Optional[float] = Field(None, ge=0)
    fiber_grams: Optional[float] = Field(None, ge=0)
    water_intake_ml: Optional[float] = Field(None, ge=0)
    sodium_intake_mg: Optional[float] = Field(None, ge=0)
    potassium_intake_mg: Optional[float] = Field(None, ge=0)
    fasting_window_hours: Optional[float] = Field(None, ge=0)


class BiomechanicalLoad(AgnosticMetricBase):
    step_count_int: Optional[int] = Field(None, ge=0)
    g_force_peak: Optional[float] = Field(None, ge=0)
    ground_contact_time_ms: Optional[float] = Field(None, ge=0)
    vertical_oscillation_cm: Optional[float] = Field(None, ge=0)


class MetabolicBaseline(AgnosticMetricBase):
    respiratory_quotient_rest: Optional[float] = Field(None, ge=0.6, le=1.1)
    measured_rmr: Optional[float] = Field(None, ge=0)
