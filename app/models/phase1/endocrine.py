from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase1.core import SpeciesCore


class SpeciesEndocrine(Base):
    """
    Phase 1 — Endocrine system ceiling constants for Homo sapiens.

    Source: Nutrivious BOS Phase 1 —
      Section 13 / Matrices 13.1–13.7 (testosterone, cortisol, insulin, thyroid, GH/IGF-1,
      catecholamines, leptin/ghrelin);
      Domain X.1 (androgens: AR kinetics, SHBG, DHT/testosterone ratio);
      Domain X.2 (HPA axis: cortisol diurnal ceiling, GR density, CBG);
      Domain X.3 (pancreatic axis: insulin/glucagon, IR kinetics, beta-cell capacity);
      Domain X.4 (HPT axis: T3/T4/TSH, thyroid receptor, peripheral conversion);
      Domain X.5 (somatotropic axis: GH pulse architecture, IGF-1/IGFBP-3, JAK2/STAT5);
      Domain X.6 (catecholamines: Epi/NE exercise surge, adrenoceptor subtypes);
      Domain X.7 (adipokines/gut hormones: leptin, ghrelin, GLP-1).

    Key equations encoded:
      BT = TT − (SHBG × TT / (Kd_SHBG + TT))        [bioavailable testosterone]
      Hormone_clearance = MCR × plasma_concentration   [metabolic clearance rate]
      GH_IGF1 = f(GH_pulse_amplitude, liver_GHR)       [somatotropic axis gain]
      T3 = T4_secretion × f_conversion + T3_direct     [thyroid peripheral conversion]

    Units:
      _nmol_l      = nmol / L
      _pmol_l      = pmol / L
      _ng_ml       = ng / mL
      _pg_ml       = pg / mL
      _ug_dl       = µg / dL
      _iu_ml       = µU / mL (insulin units)
      _miu_l       = mIU / L (TSH)
      _nm          = nM (receptor Kd)
      _min         = minutes (half-life)
      _h           = hours
      _mg_day      = mg / day (production rate)
      _ug_day      = µg / day
      _fraction    = dimensionless 0–1
    """

    __tablename__ = "species_endocrine"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_core_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )

    species_core: Mapped["SpeciesCore"] = relationship(back_populates="endocrine")

    # ── Testosterone — Gonadal Androgen Axis ─────────────────────────────────
    # Diurnal rhythm: peak 07:00-09:00; nadir 20:00-22:00; amplitude ~30-40% variation
    # SHBG binds ~40-50% (inactive); albumin ~50% (bioavailable); free ~1-3%
    testosterone_male_total_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 nmol/L; total plasma testosterone male lower bound of normal (~290 ng/dL); from Domain X.1 / Matrix 13.1"
    )
    testosterone_male_total_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="35 nmol/L; total plasma testosterone male upper bound of normal (~1010 ng/dL); from Domain X.1"
    )
    testosterone_female_total_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 nmol/L; total plasma testosterone female lower bound (~14 ng/dL); from Domain X.1"
    )
    testosterone_female_total_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.5 nmol/L; total plasma testosterone female upper bound (~72 ng/dL); from Domain X.1"
    )
    testosterone_free_male_pmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 pmol/L; free testosterone male lower bound (~1.4 ng/dL); biologically active fraction; from Domain X.1"
    )
    testosterone_free_male_pmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="250 pmol/L; free testosterone male upper bound; from Domain X.1"
    )
    testosterone_daily_production_male_mg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~6 mg/day; male daily testosterone production (Leydig cells); from Domain X.1"
    )
    testosterone_daily_production_female_mg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.25 mg/day; female daily testosterone production (ovarian + adrenal); from Domain X.1"
    )
    testosterone_half_life_plasma_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~70–100 min; unesterified testosterone plasma half-life (IV); from Domain X.1"
    )
    testosterone_diurnal_amplitude_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30–0.40; diurnal amplitude (morning peak / evening nadir − 1); from Domain X.1"
    )
    testosterone_exercise_acute_increase_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.15–0.25; acute testosterone increase fraction at VO2max vs resting (15-25%); from Domain X.1"
    )
    shbg_testosterone_kd_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1 nM; SHBG dissociation constant for testosterone; from Domain X.1"
    )
    shbg_bound_testosterone_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.40–0.50; fraction of plasma testosterone bound to SHBG (inactive); from Domain X.1"
    )
    androgen_receptor_kd_testosterone_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 nM; androgen receptor (AR) Kd for testosterone lower bound; from Domain X.1"
    )
    androgen_receptor_kd_testosterone_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 nM; AR Kd for testosterone upper bound; from Domain X.1"
    )
    androgen_receptor_kd_dht_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 nM; AR Kd for DHT (dihydrotestosterone) lower bound; ~5× higher affinity than testosterone; from Domain X.1"
    )
    androgen_receptor_kd_dht_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.3 nM; AR Kd for DHT upper bound; from Domain X.1"
    )
    srd5a2_t_to_dht_conversion_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.05–0.10; fraction of testosterone converted to DHT via 5α-reductase type II in androgen-sensitive tissues; from Domain X.1"
    )

    # ── Cortisol — HPA Axis ────────────────────────────────────────────────────
    # ACTH from pituitary (CRH-driven); cortisol suppresses ACTH (negative feedback)
    # CBG (cortisol-binding globulin) binds ~80-90%; free cortisol is biologically active
    cortisol_morning_peak_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="300 nmol/L; morning cortisol peak lower bound (~11 µg/dL; 08:00-09:00); from Domain X.2 / Matrix 13.2"
    )
    cortisol_morning_peak_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="700 nmol/L; morning cortisol peak upper bound (~25 µg/dL); from Domain X.2"
    )
    cortisol_midnight_nadir_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 nmol/L; cortisol midnight nadir lower bound (~1.8 µg/dL); from Domain X.2"
    )
    cortisol_midnight_nadir_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="150 nmol/L; cortisol midnight nadir upper bound; from Domain X.2"
    )
    cortisol_max_stress_exercise_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="800 nmol/L; cortisol at VO2max / maximal stress lower bound; from Domain X.2"
    )
    cortisol_max_stress_exercise_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1000 nmol/L; cortisol at VO2max / maximal stress upper bound; from Domain X.2"
    )
    cortisol_absolute_max_stimulated_nmol_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2000–3000 nmol/L; absolute maximum plasma cortisol (ACTH stimulation / severe illness); from Domain X.2"
    )
    cortisol_daily_production_mg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~7.5 mg/day; basal cortisol daily production (range 5-10 mg); from Domain X.2"
    )
    cortisol_half_life_plasma_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~70–100 min; total plasma cortisol half-life; from Domain X.2"
    )
    cortisol_free_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~16–20 min; free (unbound) cortisol half-life; from Domain X.2"
    )
    cortisol_diurnal_amplitude_fold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3–5; fold variation between morning peak and midnight nadir; from Domain X.2"
    )
    glucocorticoid_receptor_kd_cortisol_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 nM; glucocorticoid receptor (GR) Kd for cortisol lower bound; from Domain X.2"
    )
    glucocorticoid_receptor_kd_cortisol_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 nM; GR Kd for cortisol upper bound; from Domain X.2"
    )
    cbg_cortisol_kd_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 nM; cortisol-binding globulin (CBG) Kd for cortisol lower bound; from Domain X.2"
    )
    cbg_cortisol_kd_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="30 nM; CBG Kd for cortisol upper bound; from Domain X.2"
    )
    cbg_bound_cortisol_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.80–0.90; fraction of plasma cortisol bound to CBG at normal concentrations; from Domain X.2"
    )

    # ── Insulin — Pancreatic Beta-Cell Axis ───────────────────────────────────
    # Biphasic secretion: first phase (0-10 min; pre-formed granules) + second phase (10-120 min)
    # Liver extracts ~50% first-pass; kidney clears remainder; t½ = 5-10 min
    insulin_fasting_iu_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 µU/mL; fasting insulin lower bound (~30 pmol/L); from Domain X.3 / Matrix 13.3"
    )
    insulin_fasting_iu_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 µU/mL; fasting insulin upper bound (~90 pmol/L); from Domain X.3"
    )
    insulin_peak_postprandial_iu_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 µU/mL; peak postprandial insulin lower bound (~300 pmol/L; 60 g CHO meal); from Domain X.3"
    )
    insulin_peak_postprandial_iu_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 µU/mL; peak postprandial insulin upper bound (~600 pmol/L); from Domain X.3"
    )
    insulin_max_secretory_capacity_iu_ml: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~200–300 µU/mL; maximum plasma insulin under sustained hyperglycaemia + beta-cell stimulation; from Domain X.3"
    )
    insulin_daily_production_units: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~40–50 units/day; basal + prandial insulin production by pancreatic beta cells; from Domain X.3"
    )
    insulin_half_life_plasma_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5–10 min; plasma insulin half-life (rapid enzymatic degradation by insulinase); from Domain X.3"
    )
    insulin_receptor_kd_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 nM; insulin receptor (IR) Kd for insulin lower bound (high-affinity site); from Domain X.3"
    )
    insulin_receptor_kd_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 nM; IR Kd for insulin upper bound (low-affinity site; negative cooperativity); from Domain X.3"
    )
    insulin_liver_first_pass_extraction_fraction: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.50; fraction of portal insulin extracted by liver on first pass; from Domain X.3"
    )
    beta_cell_mass_g_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 g; total pancreatic beta-cell mass lower bound; from Domain X.3"
    )
    beta_cell_mass_g_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.5 g; total pancreatic beta-cell mass upper bound; from Domain X.3"
    )

    # ── Glucagon — Alpha-Cell Axis ─────────────────────────────────────────────
    glucagon_fasting_pg_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="70 pg/mL; fasting plasma glucagon lower bound (~20 pmol/L); from Domain X.3"
    )
    glucagon_fasting_pg_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="150 pg/mL; fasting plasma glucagon upper bound (~43 pmol/L); from Domain X.3"
    )
    glucagon_peak_hypoglycemia_pg_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="500 pg/mL; peak glucagon during hypoglycaemia lower bound; from Domain X.3"
    )
    glucagon_peak_hypoglycemia_pg_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1000 pg/mL; peak glucagon during hypoglycaemia upper bound; from Domain X.3"
    )
    glucagon_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~5–10 min; plasma glucagon half-life; from Domain X.3"
    )
    glucagon_receptor_kd_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 nM; glucagon receptor (GCGR) Kd lower bound; from Domain X.3"
    )
    glucagon_receptor_kd_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 nM; GCGR Kd upper bound; from Domain X.3"
    )
    glucagon_exercise_increase_fold_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0; glucagon fold-increase at high-intensity exercise lower bound; from Domain X.3"
    )
    glucagon_exercise_increase_fold_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.0; glucagon fold-increase at high-intensity exercise upper bound; from Domain X.3"
    )

    # ── Thyroid Hormones — HPT Axis ───────────────────────────────────────────
    # T4 is the prohormone; D2 (deiodinase type 2) converts T4 → T3 peripherally
    # 80% of circulating T3 comes from peripheral T4 → T3 conversion
    tsh_normal_miu_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 mIU/L; TSH (thyroid-stimulating hormone) lower bound of normal; from Domain X.4 / Matrix 13.4"
    )
    tsh_normal_miu_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="4.5 mIU/L; TSH upper bound of normal; from Domain X.4"
    )
    t4_total_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 nmol/L; total T4 (thyroxine) lower bound of normal; from Domain X.4"
    )
    t4_total_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="140 nmol/L; total T4 upper bound of normal; from Domain X.4"
    )
    t4_free_pmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="12 pmol/L; free T4 lower bound of normal (~0.93 ng/dL); from Domain X.4"
    )
    t4_free_pmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="22 pmol/L; free T4 upper bound of normal (~1.7 ng/dL); from Domain X.4"
    )
    t3_total_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.2 nmol/L; total T3 (triiodothyronine) lower bound of normal; from Domain X.4"
    )
    t3_total_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.5 nmol/L; total T3 upper bound of normal; from Domain X.4"
    )
    t3_free_pmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3.5 pmol/L; free T3 lower bound of normal; from Domain X.4"
    )
    t3_free_pmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8.0 pmol/L; free T3 upper bound of normal; from Domain X.4"
    )
    t4_half_life_days: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="6–7 days; T4 plasma half-life (slow turnover; large distribution volume); from Domain X.4"
    )
    t3_half_life_days: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1–2 days; T3 plasma half-life (biologically active; faster turnover than T4); from Domain X.4"
    )
    t4_daily_production_ug: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~90 µg/day; daily T4 secretion by thyroid gland (range 80-100 µg); from Domain X.4"
    )
    t3_daily_production_ug_thyroid: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~6 µg/day; T3 directly secreted by thyroid (~20% of total T3); from Domain X.4"
    )
    t3_daily_production_ug_peripheral_conversion: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~24 µg/day; T3 produced from peripheral T4 → T3 conversion via D2 deiodinase (~80% of total T3); from Domain X.4"
    )
    thyroid_receptor_kd_t3_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.05–0.10 nM; thyroid hormone receptor (TR-β/TR-α) Kd for T3 (very high affinity); from Domain X.4"
    )
    tbg_fraction_t4_bound: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.75; fraction of plasma T4 bound to TBG (thyroxine-binding globulin); from Domain X.4"
    )

    # ── Growth Hormone — Somatotropic Axis ────────────────────────────────────
    # Pulsatile secretion; largest pulse during SWS (N3 sleep); GHRH stimulates; somatostatin inhibits
    # GH → JAK2/STAT5 → IGF-1 production (liver primary source ~75%)
    gh_basal_daytime_ng_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 ng/mL; basal daytime GH lower bound (inter-pulse nadir); from Domain X.5 / Matrix 13.5"
    )
    gh_basal_daytime_ng_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 ng/mL; basal daytime GH upper bound; from Domain X.5"
    )
    gh_peak_sws_ng_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 ng/mL; GH peak during slow-wave sleep lower bound; from Domain X.5"
    )
    gh_peak_sws_ng_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="45 ng/mL; GH peak during slow-wave sleep upper bound; from Domain X.5"
    )
    gh_peak_exercise_stress_ng_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="20 ng/mL; GH peak during maximal exercise/stress lower bound (documented in Phase 1 protein section); from Domain X.5"
    )
    gh_peak_exercise_stress_ng_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="60 ng/mL; GH peak during maximal exercise/stress upper bound; from Domain X.5"
    )
    gh_pulses_per_day_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 pulses/day; GH secretory pulses per day lower bound; from Domain X.5"
    )
    gh_pulses_per_day_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="9 pulses/day; GH secretory pulses per day upper bound; from Domain X.5"
    )
    gh_daily_secretion_mg_male: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.7 mg/day; male daily GH secretion (range 0.4-1.0 mg); from Domain X.5"
    )
    gh_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~15–25 min; GH plasma half-life; from Domain X.5"
    )
    gh_receptor_kd_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 nM; growth hormone receptor (GHR) Kd for GH lower bound; from Domain X.5"
    )
    gh_receptor_kd_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="2.0 nM; GHR Kd for GH upper bound; dimerisation activates JAK2/STAT5; from Domain X.5"
    )
    igf1_basal_adult_ng_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 ng/mL; basal serum IGF-1 lower bound (adult, age-dependent); from Domain X.5"
    )
    igf1_basal_adult_ng_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="300 ng/mL; basal serum IGF-1 upper bound (adult); from Domain X.5"
    )
    igf1_max_trained_athlete_ng_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="400 ng/mL; maximum IGF-1 in elite trained athletes lower bound (Phase 1 protein section); from Domain X.5"
    )
    igf1_max_trained_athlete_ng_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="600 ng/mL; maximum IGF-1 in elite trained athletes upper bound; from Domain X.5"
    )
    igf1_half_life_ternary_complex_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~12–15 h; IGF-1 half-life as IGFBP-3/ALS ternary complex (long-acting reservoir); from Domain X.5"
    )
    igf1_free_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~10–15 min; free IGF-1 half-life (rapidly cleared); from Domain X.5"
    )
    igf1_receptor_kd_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 nM; IGF-1 receptor (IGF-1R) Kd for IGF-1 lower bound; from Domain X.5"
    )
    igf1_receptor_kd_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1.0 nM; IGF-1R Kd for IGF-1 upper bound; from Domain X.5"
    )
    igfbp3_fraction_igf1_bound: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.75; fraction of circulating IGF-1 bound in IGFBP-3/ALS ternary complex; from Domain X.5"
    )

    # ── Catecholamines — Adrenomedullary/Sympathetic Axis ────────────────────
    # Epi from adrenal medulla; NE from sympathetic nerve terminals + adrenal
    # Half-lives: 1-3 min (enzymatic degradation by MAO and COMT)
    epinephrine_basal_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.1 nmol/L; basal plasma epinephrine lower bound (~20 pg/mL); from Domain X.6 / Matrix 13.6"
    )
    epinephrine_basal_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.5 nmol/L; basal plasma epinephrine upper bound (~100 pg/mL); from Domain X.6"
    )
    epinephrine_max_exercise_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 nmol/L; plasma epinephrine at VO2max lower bound (~600 pg/mL); from Domain X.6"
    )
    epinephrine_max_exercise_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 nmol/L; plasma epinephrine at VO2max upper bound (~2000 pg/mL); from Domain X.6"
    )
    epinephrine_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–3 min; epinephrine plasma half-life (MAO + COMT metabolism); from Domain X.6"
    )
    norepinephrine_basal_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 nmol/L; basal plasma norepinephrine lower bound (~170 pg/mL); from Domain X.6"
    )
    norepinephrine_basal_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 nmol/L; basal plasma norepinephrine upper bound (~850 pg/mL); from Domain X.6"
    )
    norepinephrine_max_exercise_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="10 nmol/L; plasma norepinephrine at VO2max lower bound; from Domain X.6"
    )
    norepinephrine_max_exercise_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="40 nmol/L; plasma norepinephrine at VO2max upper bound; from Domain X.6"
    )
    norepinephrine_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~2–3 min; norepinephrine plasma half-life; from Domain X.6"
    )
    beta1_ar_kd_norepinephrine_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~1–10 µM; β₁-adrenoceptor Kd for norepinephrine; cardiac chronotropy/inotropy; from Domain X.6"
    )
    beta2_ar_kd_epinephrine_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.1–1 µM; β₂-adrenoceptor Kd for epinephrine; skeletal muscle vasodilation + bronchodilation; from Domain X.6"
    )
    alpha1_ar_kd_norepinephrine_um: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.1–1 µM; α₁-adrenoceptor Kd for norepinephrine; vasoconstriction of splanchnic/renal beds; from Domain X.6"
    )

    # ── Leptin — Adipokine Satiety Signal ─────────────────────────────────────
    # Proportional to fat mass; signals hypothalamus (ARC) to suppress appetite + increase EE
    # Falls rapidly within 12-24h of caloric restriction (before fat mass loss)
    leptin_male_ng_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="3 ng/mL; fasting plasma leptin male lower bound (lean); from Domain X.7 / Matrix 13.7"
    )
    leptin_male_ng_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="15 ng/mL; fasting plasma leptin male upper bound; from Domain X.7"
    )
    leptin_female_ng_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="8 ng/mL; fasting plasma leptin female lower bound (higher than male due to adiposity + estrogen); from Domain X.7"
    )
    leptin_female_ng_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="25 ng/mL; fasting plasma leptin female upper bound; from Domain X.7"
    )
    leptin_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~25 min; plasma leptin half-life; from Domain X.7"
    )
    leptin_receptor_kd_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 nM; leptin receptor (LepRb) Kd lower bound; from Domain X.7"
    )
    leptin_receptor_kd_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 nM; LepRb Kd upper bound; from Domain X.7"
    )
    leptin_fasting_suppression_time_h: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~12–24 h; time for leptin to fall significantly during caloric restriction (precedes fat mass loss); from Domain X.7"
    )

    # ── Ghrelin — Orexigenic Gut Hormone ──────────────────────────────────────
    # Acylated (active) ghrelin: octanoyl modification at Ser3 by GOAT enzyme
    # Stimulates GH release + appetite; rises pre-meal; suppressed post-meal
    ghrelin_total_fasting_pg_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="200 pg/mL; total fasting plasma ghrelin lower bound; from Domain X.7"
    )
    ghrelin_total_fasting_pg_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="400 pg/mL; total fasting plasma ghrelin upper bound; from Domain X.7"
    )
    ghrelin_acylated_fasting_pg_ml_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="50 pg/mL; acylated (active) ghrelin lower bound (biologically potent form); from Domain X.7"
    )
    ghrelin_acylated_fasting_pg_ml_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="100 pg/mL; acylated ghrelin upper bound; from Domain X.7"
    )
    ghrelin_acylated_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~25–30 min; acylated ghrelin plasma half-life; from Domain X.7"
    )
    ghsr1a_kd_ghrelin_nm_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="1 nM; ghrelin receptor (GHSR-1a) Kd for acylated ghrelin lower bound; from Domain X.7"
    )
    ghsr1a_kd_ghrelin_nm_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="5 nM; GHSR-1a Kd upper bound; from Domain X.7"
    )

    # ── Aldosterone — Mineralocorticoid ───────────────────────────────────────
    # Produced in adrenal zona glomerulosa; regulated by AT-II + plasma [K⁺] + ACTH
    aldosterone_basal_nmol_l_low: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.05 nmol/L; basal plasma aldosterone lower bound (supine; 2 ng/dL); from Domain X.2"
    )
    aldosterone_basal_nmol_l_high: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="0.30 nmol/L; basal plasma aldosterone upper bound (supine; 10 ng/dL); from Domain X.2"
    )
    aldosterone_max_stimulated_nmol_l: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.80 nmol/L; maximum plasma aldosterone (upright + volume-depleted + RAAS activated); from Domain X.2"
    )
    aldosterone_half_life_min: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~15–30 min; aldosterone plasma half-life; from Domain X.2"
    )
    mineralocorticoid_receptor_kd_aldosterone_nm: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.5–1.0 nM; mineralocorticoid receptor (MR) Kd for aldosterone (high affinity); from Domain X.2"
    )
    aldosterone_daily_production_mg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="~0.15–0.25 mg/day; daily aldosterone production (range 0.1-0.5 mg); from Domain X.2"
    )
