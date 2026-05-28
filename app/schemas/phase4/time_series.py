from typing import Optional
from pydantic import Field
from .base_metric import AgnosticMetricBase


class CardiacPoint(AgnosticMetricBase):
    hr_bpm: Optional[float] = Field(None, ge=0)
    rmssd_ms: Optional[float] = Field(None, ge=0)


class GlucosePoint(AgnosticMetricBase):
    glucose_mg_dL: float = Field(..., ge=0)


class ThermalPoint(AgnosticMetricBase):
    core_temp_celsius: Optional[float] = None
    skin_temp_celsius: Optional[float] = None


class EnvironmentalTelemetry(AgnosticMetricBase):
    light_exposure_lux: Optional[float] = Field(None, ge=0)
    ambient_temperature_c: Optional[float] = None
    ambient_humidity_pct: Optional[float] = Field(None, ge=0, le=100)
    altitude_m: Optional[float] = None
