"""
app/slices/neuroendocrine/ode.py — SAM × HPA × Somatotropic Axis ODE  V2.0
                                   9-state, time unit: HOURS

State  x ∈ ℝ⁹:
    x[0]  Epinephrine    [pg/mL]   t½ ≈ 2 min   (SAM — ultra-fast)
    x[1]  Norepinephrine [pg/mL]   t½ ≈ 3 min   (SAM — ultra-fast)
    x[2]  CRH            [pg/mL]   t½ ≈ 83 min  (HPA)
    x[3]  ACTH           [pg/mL]   t½ ≈ 70 min  (HPA)
    x[4]  Cortisol       [nmol/L]  t½ ≈ 92 min  (HPA)
    x[5]  GHRH           [pg/mL]   t½ ≈ 35 min  (Somatotropic)
    x[6]  Somatostatin   [pg/mL]                 (Somatotropic)
    x[7]  GH             [ng/mL]   t½ ≈ 100 min (Somatotropic)
    x[8]  IGF_1          [ng/mL]   t½ ≈ 15 h    (Somatotropic — ultra-slow)

Control (hub variables)  u ∈ ℝ⁵  (NaN-guarded before use):
    u[0]  hub_training_stress     [0, 1]    acute exercise / psychological load
    u[1]  hub_circadian_phase     [0, 1]    awakening drive (1.0 = full CAR trigger)
    u[2]  hub_IL6_pgmL            [pg/mL]   inflammatory cytokine
    u[3]  hub_glucose_mg_dL       [mg/dL]   plasma glucose
    u[4]  hub_sleep_sws_fraction  [0, 1]    slow-wave sleep fraction

STIFFNESS
---------
The ratio γ_Epi / γ_IGF1 ≈ 20.8 / 0.046 ≈ 450 creates a genuinely stiff ODE.
All integration helpers MUST use diffrax.Kvaerno5() + PIDController.

Physics
-------
SAM (ultra-fast catecholamines):
    dEpi/dt = k_Epi_basal + k_Epi_stress·stress − γ_Epi·Epi
    dNE/dt  = k_NE_basal  + k_NE_stress·stress  − γ_NE·NE

HPA:
    dCRH/dt   = (s_CRH(t) + k_CRH_CAR·circ + k_CRH_stress·stress
                 + k_CRH_IL6·IL6 + glucose_panic) · fb_CRH(Cortisol)
                 − γ_CRH·CRH
    dACTH/dt  = k_ACTH·CRH · fb_ACTH(Cortisol) − γ_ACTH·ACTH
    dCortisol/dt = k_Cort·ACTH − γ_Cort·Cortisol

    glucose_panic = k_CRH_hypo · (exp(max(0, MFT−glucose)/10) − 1)
    MFT = Metabolic_Flexibility_Threshold  [NLME-identifiable, prior mean 70 mg/dL]

Somatotropic:
    dGHRH/dt  = s_GHRH(t)·(1+k_SWS·SWS) − γ_GHRH·GHRH
    dSS/dt    = k_SS_basal + k_SS_Cort·Cortisol
                + k_SS_SAM·(Epi+NE)/1000 − γ_SS·SS
    dGH/dt    = k_GH·GHRH/(1+SS/K_SS) − γ_GH·GH
    dIGF1/dt  = k_IGF1_basal + IGF1_conv_rate·GH − γ_IGF1·IGF1

GC_sensitivity scales the negative-feedback sigmoids on CRH and ACTH.
IGF1_conv_rate (hepatic GH→IGF-1 conversion) and Metabolic_Flexibility_Threshold
are identifiable by NLME (D=3).

NaN guards on all hub variables — always returns finite derivatives.

References
----------
  Keenan & Veldhuis (1997) Am J Physiol 273:E1182       — cortisol pulsatility
  Vinther et al. (2011) J Math Biol 63:663–690           — HPA minimal model
  Veldhuis et al. (1995) J Clin Endocrinol Metab         — GH pulsatility
  Goldstein (2010) Cell Mol Neurobiol 30:1283–1295       — SAM axis kinetics
  Cryer (2001) J Clin Invest 108:1533–1535               — glucose counterregulation
  CLAUDE.md §3 prior table
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

# ── State indices ──────────────────────────────────────────────────────────────

IDX_EPI  = 0
IDX_NE   = 1
IDX_CRH  = 2
IDX_ACTH = 3
IDX_CORT = 4
IDX_GHRH = 5
IDX_SS   = 6
IDX_GH   = 7
IDX_IGF1 = 8

STATE_DIM = 9
OBS_DIM   = 4   # [Epinephrine_pgmL, Norepinephrine_pgmL, Cortisol_nmolL, IGF1_ngmL]
CTRL_DIM  = 5   # [training_stress, circadian_phase, IL6, glucose, SWS]

_TWO_PI_OVER_24 = 2.0 * math.pi / 24.0


# ── Parameter container ────────────────────────────────────────────────────────

class NeuroParams(NamedTuple):
    """
    Physiological parameters for the SAM × HPA × Somatotropic 9-state ODE.

    NLME-identifiable (D=3):
        GC_sensitivity                — GR density scaling for HPA feedback
                                        Prior: LogN(log 1.0, 0.25²)
        IGF1_conv_rate                — hepatic GH→IGF-1 conversion [h⁻¹]
                                        Prior: LogN(log 6.0, 0.30²)
        Metabolic_Flexibility_Threshold — glucose threshold for CRH panic [mg/dL]
                                        Prior: LogN(log 70.0, 0.15²)
    """
    # ── NLME-identifiable ─────────────────────────────────────────────────────
    GC_sensitivity:                   float = 1.0    # GR density scaling
    IGF1_conv_rate:                   float = 6.0    # h⁻¹  GH→IGF-1 hepatic conversion
    Metabolic_Flexibility_Threshold:  float = 70.0   # mg/dL  glucose panic threshold

    # ── SAM axis (ultra-fast catecholamines) ──────────────────────────────────
    # Epinephrine: t½ = 2 min → γ = ln2/(2/60) ≈ 20.79 h⁻¹
    gamma_Epi:       float = 20.79    # h⁻¹
    k_Epi_basal:     float = 1039.5   # pg/mL/h  resting SS ≈ 50 pg/mL
    k_Epi_stress:    float = 18750.0  # pg/mL/h  max-stress increment (SS_max ≈ 950 extra)

    # Norepinephrine: t½ = 3 min → γ = ln2/(3/60) ≈ 13.86 h⁻¹
    gamma_NE:        float = 13.86    # h⁻¹
    k_NE_basal:      float = 4158.0   # pg/mL/h  resting SS ≈ 300 pg/mL
    k_NE_stress:     float = 37422.0  # pg/mL/h  max-stress increment (SS_max ≈ 2700 extra)

    # ── CRH (hypothalamus) ────────────────────────────────────────────────────
    gamma_CRH:       float = 0.50     # h⁻¹  t½ ≈ 83 min
    s_CRH0:          float = 2.60     # pg/mL/h  basal circadian drive
    A_CRH:           float = 0.60     # circadian amplitude
    phi_CRH:         float = 3.0      # h  drive peak at 03:00
    k_CRH_IL6:       float = 0.10     # (pg/mL CRH/h) / (pg/mL IL6)
    k_CRH_hypo:      float = 2.5      # pg/mL/h  metabolic panic amplitude
    k_CRH_CAR:       float = 3.0      # pg/mL/h  Cortisol Awakening Response boost
    k_CRH_stress:    float = 0.50     # pg/mL/h  acute stress direct → CRH drive
    K_CRH_fb:        float = 250.0    # nmol/L  GR feedback half-max
    sigma_CRH_fb:    float = 90.0     # nmol/L  sigmoid width

    # ── ACTH (pituitary) ─────────────────────────────────────────────────────
    gamma_ACTH:      float = 0.60     # h⁻¹  t½ ≈ 70 min
    k_ACTH_CRH:      float = 20.0     # h⁻¹  CRH→ACTH stimulation
    K_ACTH_fb:       float = 250.0    # nmol/L  GR feedback half-max
    sigma_ACTH_fb:   float = 90.0     # nmol/L  sigmoid width

    # ── Cortisol (adrenal) ────────────────────────────────────────────────────
    gamma_Cort:      float = 0.45     # h⁻¹  t½ ≈ 92 min
    k_Cort_ACTH:     float = 5.5      # (nmol/L·h⁻¹) / (pg/mL ACTH)

    # ── GHRH (hypothalamus) ───────────────────────────────────────────────────
    gamma_GHRH:      float = 1.20     # h⁻¹  t½ ≈ 35 min (lumped)
    s_GHRH0:         float = 85.0     # pg/mL/h  basal
    A_GHRH:          float = 0.80     # nocturnal amplitude
    phi_GHRH:        float = 22.0     # h  nocturnal peak at 22:00
    k_SWS_GHRH:      float = 1.50     # SWS amplitude multiplier

    # ── Somatostatin (D-cells) ─────────────────────────────────────────────────
    gamma_SS:        float = 0.50     # h⁻¹
    k_SS_basal:      float = 10.0     # pg/mL/h  basal SS secretion
    k_SS_Cort:       float = 0.040    # (pg/mL/h) / (nmol/L cortisol)
    k_SS_SAM:        float = 0.001    # (pg/mL/h) / (pg/mL catecholamine)

    # ── GH (pituitary) ────────────────────────────────────────────────────────
    gamma_GH:        float = 0.40     # h⁻¹  t½ ≈ 100 min
    k_GH_GHRH:       float = 0.025    # (ng/mL·h⁻¹) / (pg/mL GHRH)
    K_SS_inhib:      float = 40.0     # pg/mL  somatostatin half-max inhibition

    # ── IGF-1 (hepatic) ───────────────────────────────────────────────────────
    gamma_IGF1:      float = 0.0462   # h⁻¹  t½ ≈ 15 h  (ultra-slow)
    k_IGF1_basal:    float = 10.0     # ng/mL/h  liver baseline IGF-1 secretion


DEFAULT_NEURO_PARAMS = NeuroParams()


# ── Default control (resting euglycaemic) ─────────────────────────────────────

def zero_control(t: jnp.ndarray) -> jnp.ndarray:
    """Resting state: no stress, no awakening, IL6=0, glucose=90, SWS=0."""
    return jnp.array([0.0, 0.0, 0.0, 90.0, 0.0])


# ── Physiologically plausible initial state ───────────────────────────────────

def initial_state() -> jnp.ndarray:
    """
    Resting state at 08:00 — post-awakening.

    SAM near basal (stress has subsided); HPA at morning Cortisol peak;
    Somatotropic at daytime low; IGF-1 at mid-range.
    """
    return jnp.array([
        50.0,    # Epinephrine    [pg/mL]   resting basal
        300.0,   # Norepinephrine [pg/mL]   resting basal
        2.5,     # CRH            [pg/mL]   post-peak morning
        28.0,    # ACTH           [pg/mL]
        420.0,   # Cortisol       [nmol/L]  morning peak
        22.0,    # GHRH           [pg/mL]   daytime valley
        42.0,    # Somatostatin   [pg/mL]   elevated by morning cortisol
        0.7,     # GH             [ng/mL]   near basal
        250.0,   # IGF-1          [ng/mL]
    ])


# ── Vector field ───────────────────────────────────────────────────────────────

class NeuroVectorField(diffrax.AbstractTerm):
    """
    diffrax-compatible vector field for the SAM × HPA × Somatotropic 9-state ODE.

    control_fn(t) → jnp.ndarray([stress, circ_phase, IL6, glucose, SWS])
    All hub inputs NaN-guarded before use.
    STIFF: γ_Epi/γ_IGF1 ≈ 450.  Use Kvaerno5 + PIDController.
    """
    params:     NeuroParams
    control_fn: object   # callable: float → jnp.ndarray(5,)

    def vf(
        self,
        t: jnp.ndarray,
        x: jnp.ndarray,
        args: object,
    ) -> jnp.ndarray:
        p = self.params
        u = self.control_fn(t)

        # ── NaN guards on all hub variables ───────────────────────────────────
        hub_stress  = jnp.where(jnp.isnan(u[0]), 0.0,  u[0])
        hub_circ    = jnp.where(jnp.isnan(u[1]), 0.0,  u[1])
        hub_IL6     = jnp.where(jnp.isnan(u[2]), 0.0,  u[2])
        hub_glucose = jnp.where(jnp.isnan(u[3]), 90.0, u[3])
        hub_sws     = jnp.where(jnp.isnan(u[4]), 0.0,  u[4])

        hub_stress  = jnp.clip(hub_stress,  0.0,   1.0)
        hub_circ    = jnp.clip(hub_circ,    0.0,   1.0)
        hub_IL6     = jnp.maximum(hub_IL6,  0.0)
        hub_glucose = jnp.clip(hub_glucose, 30.0,  400.0)
        hub_sws     = jnp.clip(hub_sws,     0.0,   1.0)

        Epi  = x[IDX_EPI]
        NE   = x[IDX_NE]
        CRH  = x[IDX_CRH]
        ACTH = x[IDX_ACTH]
        Cort = x[IDX_CORT]
        GHRH = x[IDX_GHRH]
        SS   = x[IDX_SS]
        GH   = x[IDX_GH]
        IGF1 = x[IDX_IGF1]

        # ── SAM axis (ultra-fast) ─────────────────────────────────────────────
        dEpi = (p.k_Epi_basal + p.k_Epi_stress * hub_stress) - p.gamma_Epi * Epi
        dNE  = (p.k_NE_basal  + p.k_NE_stress  * hub_stress) - p.gamma_NE  * NE

        # ── Circadian drives ──────────────────────────────────────────────────
        s_CRH  = p.s_CRH0  * (1.0 + p.A_CRH  * jnp.cos(_TWO_PI_OVER_24 * (t - p.phi_CRH)))
        s_CRH  = jnp.maximum(s_CRH, 0.0)

        s_GHRH = p.s_GHRH0 * (1.0 + p.A_GHRH * jnp.cos(_TWO_PI_OVER_24 * (t - p.phi_GHRH)))
        s_GHRH = jnp.maximum(s_GHRH, 0.0)

        # ── Metabolic panic (personalised threshold) ──────────────────────────
        hypo_excess   = jnp.maximum(0.0, p.Metabolic_Flexibility_Threshold - hub_glucose)
        glucose_panic = p.k_CRH_hypo * (jnp.exp(hypo_excess / 10.0) - 1.0)

        # ── GR negative-feedback sigmoids (GC_sensitivity scales strength) ────
        fb_CRH  = jax.nn.sigmoid(-p.GC_sensitivity * (Cort - p.K_CRH_fb)  / p.sigma_CRH_fb)
        fb_ACTH = jax.nn.sigmoid(-p.GC_sensitivity * (Cort - p.K_ACTH_fb) / p.sigma_ACTH_fb)

        # ── HPA axis ──────────────────────────────────────────────────────────
        total_CRH_drive = (
            s_CRH
            + p.k_CRH_CAR    * hub_circ
            + p.k_CRH_stress * hub_stress
            + p.k_CRH_IL6    * hub_IL6
            + glucose_panic
        )
        dCRH  = total_CRH_drive * fb_CRH - p.gamma_CRH * CRH
        dACTH = p.k_ACTH_CRH * CRH * fb_ACTH - p.gamma_ACTH * ACTH
        dCort = p.k_Cort_ACTH * ACTH - p.gamma_Cort * Cort

        # ── Somatotropic axis ─────────────────────────────────────────────────
        sws_boost = 1.0 + p.k_SWS_GHRH * hub_sws
        dGHRH = s_GHRH * sws_boost - p.gamma_GHRH * GHRH

        # Somatostatin: elevated by cortisol AND by sustained catecholamine load
        dSS = (
            p.k_SS_basal
            + p.k_SS_Cort * Cort
            + p.k_SS_SAM  * (Epi + NE) / 1000.0
            - p.gamma_SS  * SS
        )

        # GH: GHRH-stimulated, somatostatin-inhibited (Michaelis-Menten)
        gh_production = p.k_GH_GHRH * GHRH / (1.0 + SS / p.K_SS_inhib)
        dGH = gh_production - p.gamma_GH * GH

        # IGF-1: basal hepatic secretion + GH-driven hepatic synthesis
        dIGF1 = p.k_IGF1_basal + p.IGF1_conv_rate * GH - p.gamma_IGF1 * IGF1

        return jnp.array([dEpi, dNE, dCRH, dACTH, dCort, dGHRH, dSS, dGH, dIGF1])

    def contr(self, t0: jnp.ndarray, t1: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return t1 - t0

    def prod(self, vf: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        return vf * control


# ── Integration helpers (MANDATORY: Kvaerno5 + PIDController) ─────────────────

def integrate_1h(
    x0: jnp.ndarray,
    params: NeuroParams = DEFAULT_NEURO_PARAMS,
    control_fn: object = zero_control,
    t0: float = 8.0,
    max_steps: int = 4_000,
) -> jnp.ndarray:
    """
    Advance the neuroendocrine state by exactly 1 h.

    Uses Kvaerno5 (L-stable implicit RK5) + PIDController to handle the
    stiffness ratio γ_Epi/γ_IGF1 ≈ 450.
    """
    term   = NeuroVectorField(params=params, control_fn=control_fn)
    solver = diffrax.Kvaerno5()
    ctrl   = diffrax.PIDController(rtol=1e-4, atol=1e-4, dtmax=0.25)

    sol = diffrax.diffeqsolve(
        terms              = term,
        solver             = solver,
        t0                 = t0,
        t1                 = t0 + 1.0,
        dt0                = 0.01,
        y0                 = x0,
        stepsize_controller = ctrl,
        max_steps          = max_steps,
        adjoint            = diffrax.DirectAdjoint(),
    )
    return sol.ys[-1]


def integrate_Nh(
    x0: jnp.ndarray,
    n_hours: float,
    params: NeuroParams = DEFAULT_NEURO_PARAMS,
    control_fn: object = zero_control,
    t0: float = 8.0,
    max_steps: int = 16_000,
) -> jnp.ndarray:
    """
    Advance by n_hours (float) using Kvaerno5.

    Accepts fractional hours (e.g. 0.1 h = 6 min) for timescale-separation tests.
    """
    term   = NeuroVectorField(params=params, control_fn=control_fn)
    solver = diffrax.Kvaerno5()
    ctrl   = diffrax.PIDController(rtol=1e-4, atol=1e-4, dtmax=0.25)

    sol = diffrax.diffeqsolve(
        terms              = term,
        solver             = solver,
        t0                 = t0,
        t1                 = t0 + float(n_hours),
        dt0                = 0.01,
        y0                 = x0,
        stepsize_controller = ctrl,
        max_steps          = max_steps,
        adjoint            = diffrax.DirectAdjoint(),
    )
    return sol.ys[-1]
