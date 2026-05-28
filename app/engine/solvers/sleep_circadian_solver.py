from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import diffrax


# ---------------------------------------------------------------------------
# Reference constants
# ---------------------------------------------------------------------------
# Phillips-Robinson (2007) flip-flop nuclei
_TAU_V_REF:    float = 1.0     # h    — VLPO soma relaxation time
_TAU_M_REF:    float = 1.0     # h    — MA soma relaxation time
_G_VM_REF:     float = 2.0     # mV·h — VLPO→MA inhibitory coupling
_G_MV_REF:     float = 2.0     # mV·h — MA→VLPO inhibitory coupling
_Q_MAX_V_REF:  float = 5.0     # mV   — VLPO max output proxy
_Q_MAX_M_REF:  float = 5.0     # mV   — MA max output proxy
_THETA_V_REF:  float = 10.0    # mV   — VLPO sigmoid threshold
_THETA_M_REF:  float = 10.0    # mV   — MA sigmoid threshold
_SIGMA_V_REF:  float = 3.0     # mV   — VLPO sigmoid width
_SIGMA_M_REF:  float = 3.0     # mV   — MA sigmoid width
_V_V0_REF:     float = 2.0     # mV   — VLPO baseline drive
_V_M0_REF:     float = 15.0    # mV   — MA baseline wake drive
_V_VH_REF:     float = 30.0    # mV/[H] — adenosine→VLPO coupling (Porkka-Heiskanen 1997)
_V_VC_REF:     float = 5.0     # mV/[Mel] — melatonin→VLPO coupling
_K_CAT_WAKE_REF: float = 3.0   # mV/[Cat] — catecholamines→MA wake drive (Berntson 1994)
_K_SWS_REF:    float = 0.5     # mV⁻¹  — SWS gate sharpness on (Q_v − Q_m)

# Adenosine homeostasis (Borbély two-process; Xie 2013 glymphatic clearance)
_CHI_WAKE_REF:  float = 0.050  # h⁻¹  — base adenosine accumulation rate during wake
_RHO_CLEAR_REF: float = 0.65   # h⁻¹  — SWS glymphatic clearance (AQP4 channels)
_RHO_BASE_REF:  float = 0.005  # h⁻¹  — residual non-SWS decay
_H_TARGET_REF:  float = 0.15   # [H]  — adequate-sleep residual adenosine target
# Metabolic amplification of adenosine accumulation
_K_EE_H_REF:    float = 0.0005 # per kcal·h⁻¹ — energy expenditure → adenosine amplification
_K_CAT_H_REF:   float = 0.20   # per [Cat]    — catecholamine direct adenosine amplification

# SWS cortisol gating
_K_CORT_SWS_REF: float = 1.50  # — cortisol inhibition of SWS (Mod 5 coupling)

# Kronauer Van der Pol SCN oscillator (Czeisler 1999; Kronauer 1999)
_TAU_SCN_REF:  float = 24.2    # h    — intrinsic SCN free-running period
_MU_SCN_REF:   float = 0.23    # adim — Van der Pol nonlinearity
_K_LIGHT_REF:  float = 0.05    # adim — photic drive gain
_L_HALF_REF:   float = 200.0   # lux  — SCN half-saturation

# Melatonin (Lewy 1999; Brainard 2001)
_K_MEL_PROD_REF:  float = 0.8  # h⁻¹  — pineal synthesis rate at peak
_K_MEL_CLEAR_REF: float = 0.5  # h⁻¹  — plasma clearance (τ ≈ 2h)
_L_MEL_HALF_REF:  float = 200.0 # lux  — melatonin suppression half-saturation

# GH pulse ODE (Van Cauter 2000; Iovino 1997)
_K_GH_PROD_REF:  float = 1.50              # h⁻¹  — GH secretion rate during SWS
_K_GH_DECAY_REF: float = 0.6931 / 0.33    # h⁻¹  — GH clearance (t½ ≈ 20min)


class SleepCircadianParams(NamedTuple):
    # Phillips-Robinson flip-flop nuclei
    tau_v: float
    tau_m: float
    g_vm: float
    g_mv: float
    Q_max_v: float
    Q_max_m: float
    theta_v: float
    theta_m: float
    sigma_v: float
    sigma_m: float
    v_v0: float
    v_m0: float
    v_vh: float         # adenosine→VLPO coupling (modulated by adora2a_scale)
    v_vc: float
    k_cat_wake: float   # catecholamines→MA wake drive
    k_sws: float        # SWS gate sharpness
    # Adenosine (Process S)
    chi_wake: float     # base adenosine accumulation rate [h⁻¹]
    rho_clear: float    # SWS glymphatic clearance [h⁻¹]
    rho_base: float     # residual non-SWS adenosine decay [h⁻¹]
    H_target: float     # adequate-sleep adenosine setpoint
    k_EE_H: float       # energy expenditure → adenosine amplification
    k_Cat_H: float      # catecholamines → direct adenosine amplification
    # SWS cortisol gating
    k_Cort_SWS: float   # cortisol → SWS suppression strength
    # Kronauer SCN Van der Pol oscillator
    tau_scn: float
    mu_scn: float
    k_light: float
    L_half: float
    # Melatonin
    k_mel_prod: float
    k_mel_clear: float
    L_mel_half: float
    # GH pulse ODE
    k_GH_prod: float    # GH secretion rate during SWS
    k_GH_decay: float   # GH clearance rate
    # Genetic priors (resolved to scalars)
    per3_scale: float       # per3_vntr_prior → adenosine accumulation rate modifier
    clock_scale: float      # clock_rs1801260_prior → SCN period modifier
    adora2a_scale: float    # adora2a_prior → adenosine receptor sensitivity (v_vh gain)


def sleep_circadian_ode(t, y, args):
    """
    7-state sleep/circadian ODE.
    y    = [V_v, V_m, H, x_scn, xd_scn, Mel, GH]
    args = (SleepCircadianParams, ee_interp, cat_interp, cort_interp, light_interp)
    Time unit: hours.
    """
    params, ee_interp, cat_interp, cort_interp, light_interp = args
    V_v    = y[0]   # VLPO membrane potential proxy [mV]
    V_m    = y[1]   # MA membrane potential proxy [mV]
    H      = y[2]   # Adenosine load (Process S)
    x_scn  = y[3]   # SCN Van der Pol state x
    xd_scn = y[4]   # SCN Van der Pol state ẋ
    Mel    = y[5]   # Melatonin [normalized]
    GH     = y[6]   # Growth hormone [normalized]

    E_exp  = jnp.clip(ee_interp.evaluate(t),    0.0, None)  # kcal·h⁻¹
    Cat    = jnp.clip(cat_interp.evaluate(t),   0.0, None)
    Cort   = jnp.clip(cort_interp.evaluate(t),  0.0, None)
    L_lux  = jnp.clip(light_interp.evaluate(t), 0.0, None)

    H_pos   = jnp.maximum(H,   0.0)
    Mel_pos = jnp.maximum(Mel, 0.0)
    GH_pos  = jnp.maximum(GH,  0.0)

    # Firing-rate proxies (sigmoid, Phillips-Robinson 2007)
    Q_v = params.Q_max_v / (1.0 + jnp.exp(-(V_v - params.theta_v) / params.sigma_v))
    Q_m = params.Q_max_m / (1.0 + jnp.exp(-(V_m - params.theta_m) / params.sigma_m))

    # SWS gate: Phillips-Robinson flip-flop × cortisol suppression (Mod 5)
    SWS_PR   = 0.5 * (1.0 + jnp.tanh(params.k_sws * (Q_v - Q_m)))
    Cort_SWS = 0.5 * (1.0 - jnp.tanh(params.k_Cort_SWS * Cort))
    SWS_gate = SWS_PR * Cort_SWS

    # Normalised light [0, 1]
    L_norm = L_lux / (L_lux + params.L_half)

    # Kronauer Van der Pol SCN oscillator
    omega_scn   = 2.0 * jnp.pi / (params.tau_scn * params.clock_scale)
    photic_drive = (
        params.k_light * L_norm
        * (1.0 - 0.4 * x_scn)
        * (1.0 - 0.4 * xd_scn)
    )
    dx_scn  = xd_scn
    dxd_scn = (
        params.mu_scn * (1.0 - x_scn ** 2) * xd_scn
        - omega_scn ** 2 * x_scn
        + photic_drive
    )

    # Melatonin: pineal synthesis during SCN night, suppressed by light
    SCN_night = 0.5 * (1.0 - jnp.tanh(5.0 * x_scn))
    L_mel_inh = params.L_mel_half ** 2 / (params.L_mel_half ** 2 + L_lux ** 2)
    dMel = (
        params.k_mel_prod * SCN_night * L_mel_inh
        - params.k_mel_clear * Mel_pos
    )

    # Phillips-Robinson drives
    # adora2a_scale modulates v_vh: high A2A receptor density → stronger adenosine→VLPO coupling
    v_vh_eff = params.v_vh * params.adora2a_scale
    D_v = params.v_v0 + v_vh_eff * H_pos + params.v_vc * Mel_pos
    D_m = params.v_m0 + params.k_cat_wake * Cat

    dV_v = (1.0 / params.tau_v) * (-V_v - params.g_vm * Q_m + D_v)
    dV_m = (1.0 / params.tau_m) * (-V_m - params.g_mv * Q_v + D_m)

    # Adenosine (Process S): Borbély + energy expenditure + catecholamine amplification
    # Mod 1 (E_exp) and Mod 3 (Cat) directly amplify central adenosine accumulation
    chi_eff   = params.chi_wake * params.per3_scale
    wake_gate = 1.0 - SWS_gate
    dH = (
        chi_eff * wake_gate * (1.0 + params.k_EE_H * E_exp + params.k_Cat_H * Cat)
        - (params.rho_base + params.rho_clear * SWS_gate) * H_pos
    )

    # GH pulse ODE: fires exclusively during SWS; cortisol blocks via SWS_gate
    dGH = params.k_GH_prod * SWS_gate - params.k_GH_decay * GH_pos

    return jnp.stack([dV_v, dV_m, dH, dx_scn, dxd_scn, dMel, dGH])


def _solve_sleep_circadian(
    params: SleepCircadianParams,
    t_span_h: tuple,
    ee_interp,
    cat_interp,
    cort_interp,
    light_interp,
    y0,
    n_save: int = 512,
) -> diffrax.Solution:
    term       = diffrax.ODETerm(sleep_circadian_ode)
    solver     = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    t0, t1     = float(t_span_h[0]), float(t_span_h[1])
    saveat     = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save))
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0,
        args=(params, ee_interp, cat_interp, cort_interp, light_interp),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=100_000,
    )


class SleepCircadianSolver:
    # Phillips-Robinson flip-flop
    _TAU_V_REF:    float = _TAU_V_REF
    _TAU_M_REF:    float = _TAU_M_REF
    _G_VM_REF:     float = _G_VM_REF
    _G_MV_REF:     float = _G_MV_REF
    _Q_MAX_V_REF:  float = _Q_MAX_V_REF
    _Q_MAX_M_REF:  float = _Q_MAX_M_REF
    _THETA_V_REF:  float = _THETA_V_REF
    _THETA_M_REF:  float = _THETA_M_REF
    _SIGMA_V_REF:  float = _SIGMA_V_REF
    _SIGMA_M_REF:  float = _SIGMA_M_REF
    _V_V0_REF:     float = _V_V0_REF
    _V_M0_REF:     float = _V_M0_REF
    _V_VH_REF:     float = _V_VH_REF
    _V_VC_REF:     float = _V_VC_REF
    _K_CAT_WAKE_REF: float = _K_CAT_WAKE_REF
    _K_SWS_REF:    float = _K_SWS_REF

    # Adenosine
    _CHI_WAKE_REF:  float = _CHI_WAKE_REF
    _RHO_CLEAR_REF: float = _RHO_CLEAR_REF
    _RHO_BASE_REF:  float = _RHO_BASE_REF
    _H_TARGET_REF:  float = _H_TARGET_REF
    _K_EE_H_REF:    float = _K_EE_H_REF
    _K_CAT_H_REF:   float = _K_CAT_H_REF

    # SWS cortisol gate
    _K_CORT_SWS_REF: float = _K_CORT_SWS_REF

    # SCN oscillator
    _TAU_SCN_REF:  float = _TAU_SCN_REF
    _MU_SCN_REF:   float = _MU_SCN_REF
    _K_LIGHT_REF:  float = _K_LIGHT_REF
    _L_HALF_REF:   float = _L_HALF_REF

    # Melatonin
    _K_MEL_PROD_REF:  float = _K_MEL_PROD_REF
    _K_MEL_CLEAR_REF: float = _K_MEL_CLEAR_REF
    _L_MEL_HALF_REF:  float = _L_MEL_HALF_REF

    # GH pulse ODE
    _K_GH_PROD_REF:  float = _K_GH_PROD_REF
    _K_GH_DECAY_REF: float = _K_GH_DECAY_REF

    def _build_params(self, bayesian_priors: dict) -> SleepCircadianParams:
        # PER3-VNTR: prior = fraction of 5-repeat alleles [0=4/4, 1=5/5]
        # 4/4 → morning type → faster adenosine accumulation (+15%)
        per3_raw = bayesian_priors.get("per3_vntr_prior", float("nan"))
        if math.isnan(float(per3_raw)):
            per3_scale = 1.0
        elif float(per3_raw) < 0.25:
            per3_scale = 1.15
        elif float(per3_raw) > 0.75:
            per3_scale = 0.95
        else:
            per3_scale = 1.0 + 0.15 * (0.5 - float(per3_raw)) / 0.5

        # CLOCK-rs1801260: T-allele dose [0=C/C, 1=T/T]; T/T → period +1.5%
        clock_raw   = bayesian_priors.get("clock_rs1801260_prior", float("nan"))
        clock_scale = 1.0 if math.isnan(float(clock_raw)) else 1.0 + 0.015 * float(clock_raw)

        # ADORA2A: A2A receptor density/sensitivity [0.5=low, 1.0=baseline, 2.0=high]
        adora2a_raw   = bayesian_priors.get("adora2a_prior", float("nan"))
        adora2a_scale = 1.0 if math.isnan(float(adora2a_raw)) else max(0.5, min(2.0, float(adora2a_raw)))

        return SleepCircadianParams(
            tau_v        = self._TAU_V_REF,
            tau_m        = self._TAU_M_REF,
            g_vm         = self._G_VM_REF,
            g_mv         = self._G_MV_REF,
            Q_max_v      = self._Q_MAX_V_REF,
            Q_max_m      = self._Q_MAX_M_REF,
            theta_v      = self._THETA_V_REF,
            theta_m      = self._THETA_M_REF,
            sigma_v      = self._SIGMA_V_REF,
            sigma_m      = self._SIGMA_M_REF,
            v_v0         = self._V_V0_REF,
            v_m0         = self._V_M0_REF,
            v_vh         = self._V_VH_REF,
            v_vc         = self._V_VC_REF,
            k_cat_wake   = self._K_CAT_WAKE_REF,
            k_sws        = self._K_SWS_REF,
            chi_wake     = self._CHI_WAKE_REF,
            rho_clear    = self._RHO_CLEAR_REF,
            rho_base     = self._RHO_BASE_REF,
            H_target     = self._H_TARGET_REF,
            k_EE_H       = self._K_EE_H_REF,
            k_Cat_H      = self._K_CAT_H_REF,
            k_Cort_SWS   = self._K_CORT_SWS_REF,
            tau_scn      = self._TAU_SCN_REF,
            mu_scn       = self._MU_SCN_REF,
            k_light      = self._K_LIGHT_REF,
            L_half       = self._L_HALF_REF,
            k_mel_prod   = self._K_MEL_PROD_REF,
            k_mel_clear  = self._K_MEL_CLEAR_REF,
            L_mel_half   = self._L_MEL_HALF_REF,
            k_GH_prod    = self._K_GH_PROD_REF,
            k_GH_decay   = self._K_GH_DECAY_REF,
            per3_scale   = float(per3_scale),
            clock_scale  = float(clock_scale),
            adora2a_scale = float(adora2a_scale),
        )

    def simulate_sleep_circadian(
        self,
        bayesian_priors: dict,
        hub_energy_expenditure_arr,   # Mod 1: Hub_Energy_Expenditure_Kcal (kcal·h⁻¹)
        hub_catecholamines_arr,       # Mod 3: Hub_Catecholamines_Tone
        hub_cortisol_arr,             # Mod 5: Hub_Cortisol_Catabolic
        t_hub_h,
        hub_light_lux_arr=None,       # optional: ambient light telemetry [lux]
        t_span_h: tuple = (0.0, 168.0),
        H_init: float = 0.0,
        n_save: int = 512,
    ) -> dict:
        params = self._build_params(bayesian_priors)

        t_hub   = jnp.asarray(t_hub_h,                   dtype=jnp.float32)
        ee_arr  = jnp.asarray(hub_energy_expenditure_arr, dtype=jnp.float32)
        cat_arr = jnp.asarray(hub_catecholamines_arr,     dtype=jnp.float32)
        cort_arr = jnp.asarray(hub_cortisol_arr,          dtype=jnp.float32)

        if hub_light_lux_arr is None:
            light_arr = jnp.zeros_like(t_hub)
        else:
            light_arr = jnp.asarray(hub_light_lux_arr, dtype=jnp.float32)

        ee_interp    = diffrax.LinearInterpolation(ts=t_hub, ys=ee_arr)
        cat_interp   = diffrax.LinearInterpolation(ts=t_hub, ys=cat_arr)
        cort_interp  = diffrax.LinearInterpolation(ts=t_hub, ys=cort_arr)
        light_interp = diffrax.LinearInterpolation(ts=t_hub, ys=light_arr)

        # Initial conditions: wake state at t=0 (morning)
        # V_v=-5 (VLPO suppressed), V_m=15 (MA active), H=H_init,
        # x_scn=1 (morning peak), xd_scn=0, Mel=0 (daylight), GH=0
        y0 = jnp.array([-5.0, 15.0, float(H_init), 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

        sol = _solve_sleep_circadian(
            params, t_span_h,
            ee_interp, cat_interp, cort_interp, light_interp,
            y0, n_save,
        )

        ts      = sol.ts
        V_v_arr  = sol.ys[:, 0]
        V_m_arr  = sol.ys[:, 1]
        H_arr    = sol.ys[:, 2]
        x_scn_arr = sol.ys[:, 3]
        xd_scn_arr = sol.ys[:, 4]
        Mel_arr  = sol.ys[:, 5]
        GH_arr   = sol.ys[:, 6]

        # Recompute derived gate arrays from solution for Hub outbounds
        Q_v_arr = params.Q_max_v / (1.0 + jnp.exp(-(V_v_arr - params.theta_v) / params.sigma_v))
        Q_m_arr = params.Q_max_m / (1.0 + jnp.exp(-(V_m_arr - params.theta_m) / params.sigma_m))
        cort_on_ts = jnp.clip(jnp.interp(ts, t_hub, cort_arr), 0.0, None)
        SWS_PR_arr  = 0.5 * (1.0 + jnp.tanh(params.k_sws * (Q_v_arr - Q_m_arr)))
        Cort_SWS_arr = 0.5 * (1.0 - jnp.tanh(params.k_Cort_SWS * cort_on_ts))
        SWS_arr = SWS_PR_arr * Cort_SWS_arr

        # SCN phase normalized [0, 1]: 0 = subjective morning, 0.5 = subjective night
        SCN_phase_norm = (jnp.arctan2(xd_scn_arr, x_scn_arr) + jnp.pi) / (2.0 * jnp.pi)

        return {
            "t_h":                          ts,
            "V_VLPO_mV":                    V_v_arr,
            "V_MA_mV":                      V_m_arr,
            "Adenosine_H":                  jnp.maximum(H_arr,   0.0),
            "SCN_phase_x":                  x_scn_arr,
            "SWS_gate":                     SWS_arr,
            "Hub_SCN_Phase":                SCN_phase_norm,              # → Mod 5 endocrine sync
            "Hub_Sleep_Debt_Metabolic":     jnp.maximum(H_arr,   0.0),  # → Mod 9 central fatigue
            "Hub_GH_Repair_Signalling":     jnp.maximum(GH_arr,  0.0),  # → Mod 7 tissue repair
            "Hub_Melatonin_Tone":           jnp.maximum(Mel_arr, 0.0),  # → Mod 5 + thermoreg
        }
