"""
AthleteData — Pydantic input schema for the BOS engine (SI units throughout).

All fields are optional (float | None = None): the engine applies modifiers only
for biomarkers that are present. Fields that cannot be physiologically negative
carry ge=0 constraints. No mathematical logic lives here — validation only.

SI nomenclature:
    *_mmol_l    → millimoles per litre  (glucose, lactate, …)
    *_nmol_l    → nanomoles per litre   (vitamin D, testosterone, …)
    *_pmol_l    → picomoles per litre   (free T3, …)
    *_g_dl      → grams per decilitre   (haemoglobin)
    *_mg_l      → milligrams per litre  (CRP, …)
    *_ng_ml     → nanograms per mL      (zonulin, ferritin as ng/mL, …)
    *_ug_l      → micrograms per litre  (ferritin as µg/L — preferred SI form)
    *_ug_g      → micrograms per gram   (calprotectin)
    *_uiu_ml    → µIU per mL            (insulin — clinical convention)
    *_pct       → percentage            (HbA1c, Veillonella abundance, …)
    *_fraction  → dimensionless [0, 1]  (body fat)
    *_index     → dimensionless index   (HOMA-IR, omega-3 index, …)

Reference: Sprint 0.1A
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AthleteData(BaseModel):
    # ── Identity ─────────────────────────────────────────────────────────────
    sex: str | None = None  # "male" | "female"

    # ── Glycaemic / insulin axis ──────────────────────────────────────────────
    glucose_mmol_l: float | None = Field(default=None, ge=0)
    insulin_uiu_ml: float | None = Field(default=None, ge=0)
    hba1c_pct: float | None = Field(default=None, ge=0)
    homa_ir: float | None = Field(default=None, ge=0)

    # ── Lipid / fat oxidation axis ────────────────────────────────────────────
    triglycerides_mmol_l: float | None = Field(default=None, ge=0)
    hdl_mmol_l: float | None = Field(default=None, ge=0)
    ldl_mmol_l: float | None = Field(default=None, ge=0)
    body_fat_fraction: float | None = Field(default=None, ge=0)

    # ── Oxygen transport / haematology ────────────────────────────────────────
    hemoglobin_g_dl: float | None = Field(default=None, ge=0)
    ferritin_ug_l: float | None = Field(default=None, ge=0)
    stfr_logferritin_ratio: float | None = Field(default=None, ge=0)

    # ── Inflammation ──────────────────────────────────────────────────────────
    crp_mg_l: float | None = Field(default=None, ge=0)
    calprotectin_ug_g: float | None = Field(default=None, ge=0)

    # ── Micronutrients / hormones ─────────────────────────────────────────────
    vitamin_d_nmol_l: float | None = Field(default=None, ge=0)
    omega3_index_pct: float | None = Field(default=None, ge=0)
    testosterone_nmol_l: float | None = Field(default=None, ge=0)
    t3_free_pmol_l: float | None = Field(default=None, ge=0)
    cortisol_dhea_ratio: float | None = Field(default=None, ge=0)

    # ── Gut / microbiome axis ─────────────────────────────────────────────────
    zonulin_ng_ml: float | None = Field(default=None, ge=0)
    microbiome_shannon: float | None = Field(default=None, ge=0)
    veillonella_pct: float | None = Field(default=None, ge=0)
    gut_training_weeks: float | None = Field(default=None, ge=0)

    # ── Epigenetic / aging clocks ─────────────────────────────────────────────
    dunedin_pace: float | None = Field(default=None, ge=0)
    ppargc1a_methylation_pct: float | None = Field(default=None, ge=0)

    # ── Structural / bone turnover ────────────────────────────────────────────
    pinp_ug_l: float | None = Field(default=None, ge=0)
    ctx_ng_ml: float | None = Field(default=None, ge=0)

    # ── Lifestyle inputs (for RGC trajectory modifiers) ───────────────────────
    sleep_deficit_hours: float | None = Field(default=None, ge=0)

    model_config = {"extra": "allow"}
