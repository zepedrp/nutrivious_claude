from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
import diffrax


class BiomechanicalTissueParams(NamedTuple):
    # --- Tendon: micro-damage and collagen turnover ---
    k_damage: float             # h⁻¹ — micro-damage rate per unit supra-threshold load^n
    n_damage: float             # power-law exponent (load-damage non-linearity)
    load_threshold: float       # normalized yield load (col5a1_scale shifts this)
    k_repair_tendon: float      # h⁻¹ — GH × Col driven damage resolution
    k_collagen_synth: float     # h⁻¹ — GH-driven collagen synthesis (Magnusson model)
    k_collagen_basal_deg: float # h⁻¹ — basal MMP collagen degradation
    k_Cort_MMP3: float          # Cortisol → MMP3 enzymatic amplifier
    k_IL6_MMP3: float           # IL6 → MMP3 enzymatic amplifier
    k_z_sat: float              # Michaelis-Menten half-saturation: damage at which MMP3 effect = 50 %
    # --- Bone: Frost Mechanostat (non-linear U-curve) ---
    lazy_thresh: float          # normalized load below which no osteoblast formation signal
    pathology_thresh: float     # normalized load above which OBL proliferation collapses and apoptosis fires
    k_load_sclero: float        # sclerostin inhibition gain within the adaptive window
    k_obl_apoptosis: float      # h⁻¹ — OBL apoptosis rate per unit overload above pathology threshold
    # --- RANK-RANKL-OPG axis ---
    k_Cort_RANKL: float         # Cortisol → RANKL (osteoclast recruitment)
    k_IL6_RANKL: float          # IL6 → RANKL
    k_sex_hormone_OPG: float    # Sex hormone (Testosterone/Estradiol) → OPG (RANKL competitive block)
    k_ocl_fracture: float       # extra OCL activation per unit overload² (microfracture-driven remodeling)
    k_OBL_prolif: float         # h⁻¹ — baseline osteoblast proliferation
    k_OBL_decay: float          # h⁻¹ — osteoblast apoptosis (basal)
    k_OCL_recruit: float        # h⁻¹ — osteoclast recruitment by free RANKL
    k_OCL_decay: float          # h⁻¹ — osteoclast apoptosis
    k_form: float               # BMD formation rate per unit OBL [g/cm²·h⁻¹]
    k_resorp: float             # BMD resorption rate per unit OCL  [g/cm²·h⁻¹]
    bmd_init: float             # g/cm² — baseline BMD from Bayesian prior
    # --- Genetic priors (resolved to scalars) ---
    col5a1_scale: float         # tendon resilience [0.5,1.5]: higher → greater yield threshold
    col1a1_scale: float         # collagen synthesis capacity [0.5,1.5]
    mmp3_scale: float           # MMP3 enzymatic aggressiveness [0.5,2.0]
    # FGF23 bone-muscle coupling (Mod 2 → Mod 11)
    k_FGF23_Wnt: float          # FGF23 from exercising muscle → Wnt/LRP5 OBL amplification
    k_FGF23_OPG: float          # FGF23 → mild OPG reduction (phosphaturic endocrine effect)


def biomechanical_tissue_ode(t, y, args):
    params, load_interp, gh_interp, cort_interp, sex_hormone_interp, il6_interp, fgf23_interp = args

    z   = y[0]   # Tendon micro-damage (unbounded; normalized to [0,1] in post-processing)
    Col = y[1]   # Collagen matrix quality [0,1] — 1.0 = fully intact
    OBL = y[2]   # Osteoblast normalized activity
    OCL = y[3]   # Osteoclast normalized activity
    BMD = y[4]   # Bone mineral density [g/cm²]

    M_load = jnp.clip(load_interp.evaluate(t),        0.0, None)
    GH     = jnp.clip(gh_interp.evaluate(t),          0.0, None)
    Cort   = jnp.clip(cort_interp.evaluate(t),        0.0, None)
    SexH   = jnp.clip(sex_hormone_interp.evaluate(t), 0.0, None)
    IL6    = jnp.clip(il6_interp.evaluate(t),         0.0, None)
    FGF23  = jnp.clip(fgf23_interp.evaluate(t),       0.0, None)

    z_pos   = jnp.maximum(z,   0.0)
    Col_pos = jnp.maximum(Col, 0.0)
    OBL_pos = jnp.maximum(OBL, 0.0)
    OCL_pos = jnp.maximum(OCL, 0.0)
    BMD_pos = jnp.maximum(BMD, 0.01)

    # ── TENDON: Micro-damage accumulation (col5a1 raises yield threshold) ─────
    effective_threshold = params.col5a1_scale * params.load_threshold
    load_excess = jnp.maximum(M_load - effective_threshold, 0.0) ** params.n_damage
    dz = params.k_damage * load_excess - params.k_repair_tendon * GH * Col_pos * z_pos

    # ── COLLAGEN MATRIX: Magnusson turnover + Michaelis-Menten damage cap ─────
    # Synthesis: GH-driven, col1a1 capacity, logistic ceiling at Col=1
    k_synth_eff = params.k_collagen_synth * params.col1a1_scale * GH * (1.0 - Col_pos)

    # MMP3 enzymatic degradation: Cortisol (Mod 5) + IL6 (Mod 7) drive catabolism
    MMP3_activity = params.mmp3_scale * (params.k_Cort_MMP3 * Cort + params.k_IL6_MMP3 * IL6)

    # Michaelis-Menten: z_pos amplifies degradation but saturates at k_z_sat
    # Without saturation z→∞ drives k_degrad→∞ (non-physical rupture singularity)
    z_sat_factor = z_pos / (params.k_z_sat + z_pos)   # ∈ [0, 1) — never diverges
    k_degrad_eff = params.k_collagen_basal_deg * (1.0 + MMP3_activity) * (1.0 + z_sat_factor)

    dCol = k_synth_eff - k_degrad_eff * Col_pos

    # ── BONE: Non-linear Frost Mechanostat (U-curve) — JAX-differentiable ─────
    #
    # Three zones (all computed via jnp.maximum — no if/else, JIT-safe):
    #   [0, lazy_thresh]          → no formation signal (bone remodeling dormant)
    #   [lazy_thresh, path_thresh]→ adaptive window: OBL proliferates via Wnt/sclerostin
    #   [path_thresh, ∞)          → pathological overload: OBL collapses + fracture OCL
    #
    above_lazy = jnp.maximum(M_load - params.lazy_thresh,      0.0)
    above_path = jnp.maximum(M_load - params.pathology_thresh,  0.0)

    # Net osteoblast mechanical signal: rises in adaptive window, self-limits above pathology
    # above_lazy - above_path → positive only between thresholds; net → 0 at extreme overload
    obl_mech_signal = above_lazy - above_path   # ∈ [0, path_thresh - lazy_thresh]; no if/else

    # Apoptosis: continuous, rate-proportional suppression above pathology threshold
    # When above_path >> 0 → apoptosis_force >> k_OBL_decay (OBL decimated rapidly)
    apoptosis_force = params.k_obl_apoptosis * above_path

    # FGF23 (Mod 2 → Mod 11): exercising muscle secretes FGF21 (myokine) and
    # bone FGF23 is amplified by mechanical loading. FGF23 acts via FGF receptor/
    # klotho on osteocytes → Wnt/LRP5 amplification of OBL proliferation in the
    # adaptive window (Jones et al. 2006; Robling 2008).
    wnt_amplification = 1.0 + params.k_FGF23_Wnt * FGF23

    dOBL = (
        params.k_OBL_prolif * wnt_amplification * (1.0 + params.k_load_sclero * obl_mech_signal)
        - params.k_OBL_decay * OBL_pos
        - apoptosis_force * OBL_pos
    )

    # ── RANK-RANKL-OPG axis ───────────────────────────────────────────────────
    # RANKL: catabolic — Cortisol (HPA) + IL6 (inflammatory) drive osteoclastogenesis
    RANKL = params.k_Cort_RANKL * Cort + params.k_IL6_RANKL * IL6

    # OPG: anabolic — Sex hormones (T AND E2 unified) competitively block RANKL
    # RED-S: when SexH → 0 (e.g. amenorrhoea), OPG → 0, RANKL acts unopposed
    # FGF23: phosphaturic endocrine action mildly reduces OPG at high levels
    OPG = params.k_sex_hormone_OPG * SexH * jnp.maximum(1.0 - params.k_FGF23_OPG * FGF23, 0.1)

    # Free RANKL: competitive binding with OPG for RANK receptor
    RANKL_free = RANKL / (RANKL + OPG + 1e-6)

    # Microfracture-driven extra OCL: above pathology threshold, necrotic debris activates
    # macrophages independently of RANKL (Verborgt 2000 — osteocyte apoptosis pathway)
    ocl_fracture_signal = params.k_ocl_fracture * above_path ** 2

    dOCL = (
        params.k_OCL_recruit * RANKL_free
        + ocl_fracture_signal
        - params.k_OCL_decay * OCL_pos
    )

    # BMD: net balance — formation by OBL vs resorption by OCL × current density
    dBMD = params.k_form * OBL_pos - params.k_resorp * OCL_pos * BMD_pos

    return jnp.array([dz, dCol, dOBL, dOCL, dBMD])


def _solve_biomechanical_tissue(
    params: BiomechanicalTissueParams,
    t_span_h: tuple,
    load_interp,
    gh_interp,
    cort_interp,
    sex_hormone_interp,
    il6_interp,
    fgf23_interp,
    n_save: int = 512,
) -> diffrax.Solution:
    term       = diffrax.ODETerm(biomechanical_tissue_ode)
    solver     = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    t0, t1     = float(t_span_h[0]), float(t_span_h[1])
    saveat     = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save))
    y0 = jnp.array([
        0.0,               # z: no micro-damage at baseline
        1.0,               # Col: healthy collagen matrix
        1.0,               # OBL: baseline osteoblast activity
        1.0,               # OCL: baseline osteoclast activity
        params.bmd_init,   # BMD: from Bayesian prior
    ], dtype=jnp.float32)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.001,
        y0=y0,
        args=(params, load_interp, gh_interp, cort_interp, sex_hormone_interp, il6_interp, fgf23_interp),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=65536,
    )


class BiomechanicalTissueSolver:
    # Tendon damage
    _K_DAMAGE_REF:        float = 0.30
    _N_DAMAGE_REF:        float = 2.0
    _LOAD_THRESHOLD_REF:  float = 1.0
    _K_REPAIR_TENDON_REF: float = 0.15

    # Collagen matrix (Magnusson model)
    _K_COLLAGEN_SYNTH_REF:     float = 0.04
    _K_COLLAGEN_BASAL_DEG_REF: float = 0.008
    _K_CORT_MMP3_REF:          float = 0.50
    _K_IL6_MMP3_REF:           float = 0.35
    _K_Z_SAT_REF:              float = 0.50   # Michaelis-Menten half-saturation (normalized z units)

    # Frost Mechanostat U-curve
    _LAZY_THRESH_REF:      float = 0.20   # load below this → bone dormant
    _PATHOLOGY_THRESH_REF: float = 3.00   # load above this → OBL collapses, fractures
    _K_LOAD_SCLERO_REF:    float = 0.40   # sclerostin inhibition gain within adaptive window
    _K_OBL_APOPTOSIS_REF:  float = 0.50   # h⁻¹ OBL apoptosis per unit overload

    # RANK-RANKL-OPG
    _K_CORT_RANKL_REF:       float = 0.60
    _K_IL6_RANKL_REF:        float = 0.45
    _K_SEX_HORMONE_OPG_REF:  float = 0.55   # unified Testosterone + Estradiol OPG drive
    _K_OCL_FRACTURE_REF:     float = 0.25   # extra OCL per unit overload²

    # Osteoblast / Osteoclast dynamics
    _K_OBL_PROLIF_REF:  float = 0.20
    _K_OBL_DECAY_REF:   float = 0.10
    _K_OCL_RECRUIT_REF: float = 0.35
    _K_OCL_DECAY_REF:   float = 0.12

    # BMD (slow — measurable change over weeks, not hours)
    _K_FORM_REF:   float = 0.001
    _K_RESORP_REF: float = 0.001
    _BMD_INIT_REF: float = 1.15   # g/cm² — lumbar spine population reference

    # FGF23 bone-muscle coupling
    _K_FGF23_WNT_REF: float = 0.30  # Wnt/LRP5 OBL amplification by exercise-derived FGF23
    _K_FGF23_OPG_REF: float = 0.15  # mild OPG reduction at high FGF23 (phosphaturic effect)

    def _build_params(self, bayesian_priors: dict) -> BiomechanicalTissueParams:
        col5a1_raw = bayesian_priors.get("col5a1_rs12722_prior",   float("nan"))
        col1a1_raw = bayesian_priors.get("col1a1_rs1800012_prior", float("nan"))
        mmp3_raw   = bayesian_priors.get("mmp3_rs679620_prior",    float("nan"))
        bmd_raw    = bayesian_priors.get("bmd_total_prior",        float("nan"))

        col5a1_scale = (1.0 if math.isnan(float(col5a1_raw))
                        else float(np.clip(float(col5a1_raw), 0.5, 1.5)))
        col1a1_scale = (1.0 if math.isnan(float(col1a1_raw))
                        else float(np.clip(float(col1a1_raw), 0.5, 1.5)))
        mmp3_scale   = (1.0 if math.isnan(float(mmp3_raw))
                        else float(np.clip(float(mmp3_raw), 0.5, 2.0)))
        bmd_init     = (self._BMD_INIT_REF if math.isnan(float(bmd_raw))
                        else float(np.clip(float(bmd_raw), 0.5, 2.0)))

        return BiomechanicalTissueParams(
            k_damage              = self._K_DAMAGE_REF,
            n_damage              = self._N_DAMAGE_REF,
            load_threshold        = self._LOAD_THRESHOLD_REF,
            k_repair_tendon       = self._K_REPAIR_TENDON_REF,
            k_collagen_synth      = self._K_COLLAGEN_SYNTH_REF,
            k_collagen_basal_deg  = self._K_COLLAGEN_BASAL_DEG_REF,
            k_Cort_MMP3           = self._K_CORT_MMP3_REF,
            k_IL6_MMP3            = self._K_IL6_MMP3_REF,
            k_z_sat               = self._K_Z_SAT_REF,
            lazy_thresh           = self._LAZY_THRESH_REF,
            pathology_thresh      = self._PATHOLOGY_THRESH_REF,
            k_load_sclero         = self._K_LOAD_SCLERO_REF,
            k_obl_apoptosis       = self._K_OBL_APOPTOSIS_REF,
            k_Cort_RANKL          = self._K_CORT_RANKL_REF,
            k_IL6_RANKL           = self._K_IL6_RANKL_REF,
            k_sex_hormone_OPG     = self._K_SEX_HORMONE_OPG_REF,
            k_ocl_fracture        = self._K_OCL_FRACTURE_REF,
            k_OBL_prolif          = self._K_OBL_PROLIF_REF,
            k_OBL_decay           = self._K_OBL_DECAY_REF,
            k_OCL_recruit         = self._K_OCL_RECRUIT_REF,
            k_OCL_decay           = self._K_OCL_DECAY_REF,
            k_form                = self._K_FORM_REF,
            k_resorp              = self._K_RESORP_REF,
            bmd_init              = bmd_init,
            col5a1_scale          = col5a1_scale,
            col1a1_scale          = col1a1_scale,
            mmp3_scale            = mmp3_scale,
            k_FGF23_Wnt           = self._K_FGF23_WNT_REF,
            k_FGF23_OPG           = self._K_FGF23_OPG_REF,
        )

    def simulate_biomechanical_tissue(
        self,
        bayesian_priors: dict,
        hub_mechanical_load_arr,    # normalized load (G × N_cycles proxy; 1.0 = reference)
        hub_gh_repair_arr,          # Mod 4: Hub_GH_Repair_Signalling [normalized]
        hub_cortisol_arr,           # Mod 5: Hub_Cortisol [normalized]
        hub_sex_hormones_arr,       # Mod 5: Hub_Sex_Hormone_Tone [T+E2 unified, normalized]
        hub_il6_arr,                # Mod 7: Hub_Cytokine_IL6_Systemic [normalized]
        hub_fgf23_arr,              # Mod 2: Hub_FGF23_BoneCoupling [normalized]
        t_hub_h,
        t_span_h: tuple = (0.0, 168.0),
        n_save: int = 512,
    ) -> dict:
        params = self._build_params(bayesian_priors)

        t_hub     = jnp.asarray(t_hub_h,                  dtype=jnp.float32)
        load_arr  = jnp.asarray(hub_mechanical_load_arr,   dtype=jnp.float32)
        gh_arr    = jnp.asarray(hub_gh_repair_arr,         dtype=jnp.float32)
        cort_arr  = jnp.asarray(hub_cortisol_arr,          dtype=jnp.float32)
        sexh_arr  = jnp.asarray(hub_sex_hormones_arr,      dtype=jnp.float32)
        il6_arr   = jnp.asarray(hub_il6_arr,               dtype=jnp.float32)
        fgf23_arr = jnp.asarray(hub_fgf23_arr,             dtype=jnp.float32)

        load_interp        = diffrax.LinearInterpolation(ts=t_hub, ys=load_arr)
        gh_interp          = diffrax.LinearInterpolation(ts=t_hub, ys=gh_arr)
        cort_interp        = diffrax.LinearInterpolation(ts=t_hub, ys=cort_arr)
        sex_hormone_interp = diffrax.LinearInterpolation(ts=t_hub, ys=sexh_arr)
        il6_interp         = diffrax.LinearInterpolation(ts=t_hub, ys=il6_arr)
        fgf23_interp       = diffrax.LinearInterpolation(ts=t_hub, ys=fgf23_arr)

        sol = _solve_biomechanical_tissue(
            params, t_span_h,
            load_interp, gh_interp, cort_interp, sex_hormone_interp, il6_interp,
            fgf23_interp,
            n_save=n_save,
        )

        ts      = sol.ts
        z_arr   = sol.ys[:, 0]
        Col_arr = sol.ys[:, 1]
        OBL_arr = sol.ys[:, 2]
        OCL_arr = sol.ys[:, 3]
        BMD_arr = sol.ys[:, 4]

        Col_pos = jnp.maximum(Col_arr, 0.0)
        OBL_pos = jnp.maximum(OBL_arr, 0.0)
        OCL_pos = jnp.maximum(OCL_arr, 0.0)
        BMD_pos = jnp.maximum(BMD_arr, 0.0)

        # Hub_Tissue_Microdamage_z [0,1]: read by Mod 7 (macrophage IL6) and Mod 9 (pain)
        z_norm    = jnp.clip(z_arr, 0.0, 1.0)

        # Hub_Tendon_Stiffness_Capacity [0,1]: collagen quality × (1 - normalized damage)
        stiffness = jnp.clip(Col_pos * (1.0 - z_norm), 0.0, 1.0)

        return {
            "t_h":                           ts,
            "Tendon_Damage_z_raw":           jnp.maximum(z_arr, 0.0),
            "Collagen_Matrix":               Col_pos,
            "Osteoblast_Activity":           OBL_pos,
            "Osteoclast_Activity":           OCL_pos,
            "Hub_Tissue_Microdamage_z":      z_norm,
            "Hub_Bone_Mass_Density":         BMD_pos,
            "Hub_Tendon_Stiffness_Capacity": stiffness,
        }
