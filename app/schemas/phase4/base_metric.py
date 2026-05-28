from pydantic import BaseModel
from datetime import datetime


class AgnosticMetricBase(BaseModel):
    timestamp: datetime
    source_method: str  # "wearable_api" | "app_slider" | "manual_input"
