"""
app/slices/neuroendocrine/ode.py -- SAM x HPA x Somatotropic Axis ODE  V3.0
                                    9-state, time unit: HOURS
                                    + pulsatile Cortisol via ControlTerm SDE

State  x in R^9:
    x[0]  Epinephrine    [pg/mL]   t1/2 ~2 min   (SAM -- ultra-fast)
    x[1]  Norepinephrine [pg/mL]   t1/2 ~3 min   (SAM -- ultra-fast)
    x[2]  CRH            [pg/mL]   t1/2 ~83 min  (HPA)
    x[3]  ACTH           [pg/mL]   t1/2 ~70 min  (HPA)
    x[4]  Cortisol       [nmol/L]  t1/2 ~92 min  (HPA)
    x[5]  GHRH           [pg/mL]   t1/2 ~35 min  (Somatotropic)
    x[6]  Somatostatin   [pg/mL]                 (Somatotropic)
    x[7]  GH             [ng/mL]   t1/2 ~100 min (Somatotropic)
    x[8]  IGF_1          [ng/mL]   t1/2 ~15 h    (Somatotropic -- ultra-slow)

Control (hub variables)  u in R^5  (NaN-guarded before use):
    u[0]  hub_training_stress     [0, 1]    acute exercise / psychological load
    u[1]  hub_circadian_phase     [0, 1]    awakening drive (1.0 = full CAR trigger)
    u[2]  hub_IL6_pgmL            [pg/mL]   inflammatory cytokine
    u[3]  hub_glucose_mg_dL       [mg/dL]   plasma glucose
    u[4]  hub_sleep_sws_fraction  [0, 1]    slow-wave sleep fraction

PULSATILE CORTISOL ARCHITECTURE (V3.0)
---------------------------------------
HPA pulsatility operates at hourly timescales (Veldhuis 1995: ~12-15 pulses/24h).
Architecture:
    1. PRE-SAMPLE outside JIT (Python/NumPy):
         ts_pulses, amp_pulses = sample_cortisol_pulses(rng_key, t0, t1)
         ts_pulses:  Poisson inter-arrival times (exponential; lambda ~0.8/h)
         amp_pulses: LogNormal amplitudes (mu_ln = log(8) - 0.5*0.25, sigma=0.5)
         Physiological basis: CRH bolus amplitudes 5-15 pg/mL (Keenan 1997)

    2. BUILD control path (Python, outside JIT):
         pulse_control = _build_pulse_control(t0, t1, ts_pulses, amp_pulses)
         Returns diffrax.LinearInterpolation of the cumulative pulse sum.
         Over [t, t+dt], control.evaluate(t, t+dt) = sum of pulses landing in [t, t+dt].

    3. ASSEMBLE terms:
         continuous_term = diffrax.ODETerm(_neuro_continuous_vf)
         pulse_term      = diffrax.ControlTerm(_pulse_injection_vf, pulse_control)
         terms           = diffrax.MultiTerm(continuous_term, pulse_term)
         solver          = diffrax.Kvaerno5()  (L-stable; handles stiffness ratio ~450)

    4. INJECT: _pulse_injection_vf returns (STATE_DIM,) with 1.0 at IDX_CRH.
       Each pulse bolus enters as a direct CRH increment. The HPA axis then
       propagates CRH -> ACTH -> Cortisol through the continuous ODE kinetics.

STIFFNESS
---------
gamma_Epi / gamma_IGF1 ~ 20.8 / 0.046 ~ 450.
Mandatory: diffrax.Kvaerno5() + PIDController for all integration helpers.

References
----------
  Keenan & Veldhuis (1997) Am J Physiol 273:E1182       -- cortisol pulsatility
  Vinther et al. (2011) J Math Biol 63:663-690           -- HPA minimal model
  Veldhuis et al. (1995) J Clin Endocrinol Metab         -- GH pulsatility
  Goldstein (2010) Cell Mol Neurobiol 30:1283-1295       -- SAM axis kinetics
  Cryer (2001) J Clin Invest 108:1533-1535               -- glucose counterregulation
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
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
    Physiological parameters for the SAM x HPA x Somatotropic 9-state ODE.

    NLME-identifiable (D=3):
        GC_sensitivity                -- GR density scaling for HPA feedback
        IGF1_conv_rate                -- hepatic GH->IGF-1 conversion [h^-1]
        Metabolic_Flexibility_Threshold -- glucose threshold for CRH panic [mg/dL]
    """
    # NLME-identifiable
    GC_sensitivity:                  float = 1.0
    IGF1_conv_rate:                  float = 6.0
    Metabolic_Flexibility_Threshold: float = 70.0

    # SAM axis
    gamma_Epi:       float = 20.79
    k_Epi_basal:     float = 1039.5
    k_Epi_stress:    float = 18750.0
    gamma_NE:        float = 13.86
    k_NE_basal:      float = 4158.0
    k_NE_stress:     float = 37422.0

    # CRH
    gamma_CRH:       float = 0.50
    s_CRH0:          float = 2.60
    A_CRH:           float = 0.60
    phi_CRH:         float = 3.0
    k_CRH_IL6:       float = 0.10
    k_CRH_hypo:      float = 2.5
    k_CRH_CAR:       float = 3.0
    k_CRH_stress:    float = 0.50
    K_CRH_fb:        float = 250.0
    sigma_CRH_fb:    float = 90.0

    # ACTH
    gamma_ACTH:      float = 0.60
    k_ACTH_CRH:      float = 20.0
    K_ACTH_fb:       float = 250.0
    sigma_ACTH_fb:   float = 90.0

    # Cortisol
    gamma_Cort:      float = 0.45
    k_Cort_ACTH:     float = 5.5

    # GHRH
    gamma_GHRH:      float = 1.20
    s_GHRH0:         float = 85.0
    A_GHRH:          float = 0.80
    phi_GHRH:        float = 22.0
    k_SWS_GHRH:      float = 1.50

    # Somatostatin
    gamma_SS:        float = 0.50
    k_SS_basal:      float = 10.0
    k_SS_Cort:       float = 0.040
    k_SS_SAM:        float = 0.001

    # GH
    gamma_GH:        float = 0.40
    k_GH_GHRH:       float = 0.025
    K_SS_inhib:      float = 40.0

    # IGF-1
    gamma_IGF1:      float = 0.0462
    k_IGF1_basal:    float = 10.0


DEFAULT_NEURO_PARAMS = NeuroParams()


# ── Default hub control (resting euglycaemic) ─────────────────────────────────

def zero_control(t: jnp.ndarray) -> jnp.ndarray:
    """Resting: no stress, no awakening, IL6=0, glucose=90, SWS=0."""
    return jnp.array([0.0, 0.0, 0.0, 90.0, 0.0])


# ── Physiologically plausible initial state ───────────────────────────────────

def initial_state() -> jnp.ndarray:
    """Resting state at 08:00 -- post-awakening."""
    return jnp.array([
        50.0,    # Epinephrine    [pg/mL]
        300.0,   # Norepinephrine [pg/mL]
        2.5,     # CRH            [pg/mL]
        28.0,    # ACTH           [pg/mL]
        420.0,   # Cortisol       [nmol/L] morning peak
        22.0,    # GHRH           [pg/mL]  daytime valley
        42.0,    # Somatostatin   [pg/mL]
        0.7,     # GH             [ng/mL]
        250.0,   # IGF-1          [ng/mL]
    ])


# ── Continuous vector field (standalone, ODETerm-compatible) ──────────────────

def _neuro_continuous_vf(
    t:    jnp.ndarray,
    y:    jnp.ndarray,
    args: tuple,
) -> jnp.ndarray:
    """
    Continuous 9-state neuroendocrine ODE.

    args = (NeuroParams, hub_control_fn)
    hub_control_fn(t) -> jnp.ndarray([stress, circ_phase, IL6, glucose, SWS])

    NaN guards on all hub variables.
    Stiffness ratio gamma_Epi/gamma_IGF1 ~ 450 -- use Kvaerno5 always.
    """
    p, control_fn = args
    u = control_fn(t)

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

    Epi  = y[IDX_EPI]
    NE   = y[IDX_NE]
    CRH  = y[IDX_CRH]
    ACTH = y[IDX_ACTH]
    Cort = y[IDX_CORT]
    GHRH = y[IDX_GHRH]
    SS   = y[IDX_SS]
    GH   = y[IDX_GH]
    IGF1 = y[IDX_IGF1]

    # SAM axis (ultra-fast catecholamines)
    dEpi = (p.k_Epi_basal + p.k_Epi_stress * hub_stress) - p.gamma_Epi * Epi
    dNE  = (p.k_NE_basal  + p.k_NE_stress  * hub_stress) - p.gamma_NE  * NE

    # Circadian drives
    s_CRH  = p.s_CRH0  * (1.0 + p.A_CRH  * jnp.cos(_TWO_PI_OVER_24 * (t - p.phi_CRH)))
    s_CRH  = jnp.maximum(s_CRH, 0.0)
    s_GHRH = p.s_GHRH0 * (1.0 + p.A_GHRH * jnp.cos(_TWO_PI_OVER_24 * (t - p.phi_GHRH)))
    s_GHRH = jnp.maximum(s_GHRH, 0.0)

    # Metabolic panic
    hypo_excess   = jnp.maximum(0.0, p.Metabolic_Flexibility_Threshold - hub_glucose)
    glucose_panic = p.k_CRH_hypo * (jnp.exp(hypo_excess / 10.0) - 1.0)

    # GR negative-feedback sigmoids
    fb_CRH  = jax.nn.sigmoid(-p.GC_sensitivity * (Cort - p.K_CRH_fb)  / p.sigma_CRH_fb)
    fb_ACTH = jax.nn.sigmoid(-p.GC_sensitivity * (Cort - p.K_ACTH_fb) / p.sigma_ACTH_fb)

    # HPA axis
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

    # Somatotropic axis
    sws_boost = 1.0 + p.k_SWS_GHRH * hub_sws
    dGHRH = s_GHRH * sws_boost - p.gamma_GHRH * GHRH

    dSS = (
        p.k_SS_basal
        + p.k_SS_Cort * Cort
        + p.k_SS_SAM  * (Epi + NE) / 1000.0
        - p.gamma_SS  * SS
    )

    gh_production = p.k_GH_GHRH * GHRH / (1.0 + SS / p.K_SS_inhib)
    dGH = gh_production - p.gamma_GH * GH

    dIGF1 = p.k_IGF1_basal + p.IGF1_conv_rate * GH - p.gamma_IGF1 * IGF1

    return jnp.array([dEpi, dNE, dCRH, dACTH, dCort, dGHRH, dSS, dGH, dIGF1])


# ── Pulsatile ControlTerm vector field ────────────────────────────────────────

def _pulse_injection_vf(
    t:    jnp.ndarray,
    y:    jnp.ndarray,
    args: Any,
) -> jnp.ndarray:
    """
    CRH pulse injection vector field for diffrax.ControlTerm.

    Returns (STATE_DIM,) with 1.0 at IDX_CRH; all other states receive 0.
    The ControlTerm multiplies this by the control increment over [t, t+dt]:
        delta_CRH = 1.0 * (cumulative_pulses(t+dt) - cumulative_pulses(t))
                  = sum of pulse amplitudes arriving in [t, t+dt].

    This is the physiologically correct mechanism: each cortisol pulse is
    a bolus CRH stimulation that then propagates via:
        CRH -> ACTH -> Cortisol (continuous ODE kinetics; Keenan 1997).
    """
    return jnp.zeros(STATE_DIM, dtype=jnp.float32).at[IDX_CRH].set(jnp.float32(1.0))


# ── Pre-sampling utilities (outside JIT -- Python/NumPy) ─────────────────────

def sample_cortisol_pulses(
    rng_key:        jax.Array,
    t_start_h:      float,
    t_end_h:        float,
    lambda_per_h:   float = 0.8,     # ~12-15 pulses/24h (Veldhuis 1995)
    amp_mean_pgml:  float = 8.0,     # mean CRH bolus amplitude [pg/mL]
    amp_sigma_ln:   float = 0.5,     # LogNormal sigma
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-sample a Poisson CRH pulse train (CPU, outside JIT).

    Poisson process: inter-arrival times ~ Exponential(1/lambda).
    Amplitudes: LogNormal(mu_ln, sigma_ln) where mu_ln = log(mean) - sigma^2/2.

    Parameters
    ----------
    rng_key       : JAX PRNGKey (used to seed NumPy RNG)
    t_start_h     : integration start time [h]
    t_end_h       : integration end time [h]
    lambda_per_h  : pulse rate [pulses/h]
    amp_mean_pgml : mean pulse amplitude [pg/mL]
    amp_sigma_ln  : LogNormal sigma of amplitudes

    Returns
    -------
    ts_pulses  : np.ndarray float32 -- pulse arrival times [h], sorted
    amp_pulses : np.ndarray float32 -- CRH bolus amplitudes [pg/mL]
    """
    seed = int(jax.random.fold_in(rng_key, 0)[0]) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)

    duration_h = float(t_end_h - t_start_h)
    n_buffer = max(int(lambda_per_h * duration_h * 2) + 20, 30)

    inter = rng.exponential(1.0 / lambda_per_h, size=n_buffer)
    ts    = float(t_start_h) + np.cumsum(inter)
    ts    = ts[ts < float(t_end_h)]

    if len(ts) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    mu_ln = np.log(amp_mean_pgml) - 0.5 * amp_sigma_ln ** 2
    amps  = rng.lognormal(mu_ln, amp_sigma_ln, size=len(ts)).astype(np.float32)

    return ts.astype(np.float32), amps


def _build_pulse_control(
    t0:         float,
    t1:         float,
    ts_pulses:  np.ndarray,
    amp_pulses: np.ndarray,
    n_base:     int = 201,
) -> diffrax.LinearInterpolation:
    """
    Build a cumulative-sum LinearInterpolation of the pre-sampled pulse train.

    The control is a staircase function; slopes are zero between pulses.
    ControlTerm computes delta_control over [t, t+dt] via
        control.evaluate(t, t+dt) = ys(t+dt) - ys(t)
                                   = sum of pulse amplitudes in (t, t+dt].

    To create near-exact steps, a phantom grid point is inserted just before
    each pulse arrival (distance eps). Over [t_pulse - eps, t_pulse], the
    LinearInterpolation ramps up by amp_i, approximating the Dirac delta.

    Parameters
    ----------
    t0, t1     : integration window endpoints [h]
    ts_pulses  : np.ndarray -- pulse arrival times [h]
    amp_pulses : np.ndarray -- CRH bolus amplitudes [pg/mL]
    n_base     : number of base grid points
    """
    ts_base = np.linspace(float(t0), float(t1), n_base, dtype=np.float32)

    if len(ts_pulses) > 0:
        eps     = np.float32(1e-5)
        ts_pre  = (ts_pulses - eps).astype(np.float32)
        ts_all  = np.sort(np.unique(np.concatenate([ts_base, ts_pre, ts_pulses])))
    else:
        ts_all = ts_base

    # Build cumulative sum forward pass
    ys_all = np.zeros(len(ts_all), dtype=np.float32)
    p_idx  = 0
    cumul  = np.float32(0.0)
    for i, t in enumerate(ts_all):
        while p_idx < len(ts_pulses) and ts_pulses[p_idx] <= t:
            cumul  += amp_pulses[p_idx]
            p_idx  += 1
        ys_all[i] = cumul

    return diffrax.LinearInterpolation(
        ts=jnp.array(ts_all, dtype=jnp.float32),
        ys=jnp.array(ys_all, dtype=jnp.float32),
    )


# ── Legacy wrapper (for backward compat with old AbstractTerm usage) ──────────

class NeuroVectorField(diffrax.AbstractTerm):
    """
    Legacy diffrax AbstractTerm wrapper. Used by integrate_1h / integrate_Nh.
    New code should use _neuro_continuous_vf with ODETerm directly.
    """
    params:     NeuroParams
    control_fn: object

    def vf(self, t, x, args):
        return _neuro_continuous_vf(t, x, (self.params, self.control_fn))

    def contr(self, t0, t1, **kwargs):
        return t1 - t0

    def prod(self, vf, control):
        return vf * control


# ── Integration helpers ────────────────────────────────────────────────────────

def integrate_1h(
    x0:         jnp.ndarray,
    params:     NeuroParams = DEFAULT_NEURO_PARAMS,
    control_fn: object = zero_control,
    t0:         float = 8.0,
    max_steps:  int = 4_000,
) -> jnp.ndarray:
    """Advance neuroendocrine state by 1 h (Kvaerno5, no pulses)."""
    term   = diffrax.ODETerm(_neuro_continuous_vf)
    solver = diffrax.Kvaerno5()
    ctrl   = diffrax.PIDController(rtol=1e-4, atol=1e-4, dtmax=0.25)

    sol = diffrax.diffeqsolve(
        terms               = term,
        solver              = solver,
        t0                  = t0,
        t1                  = t0 + 1.0,
        dt0                 = 0.01,
        y0                  = x0,
        args                = (params, control_fn),
        stepsize_controller = ctrl,
        max_steps           = max_steps,
        adjoint             = diffrax.DirectAdjoint(),
    )
    return sol.ys[-1]


def integrate_Nh(
    x0:         jnp.ndarray,
    n_hours:    float,
    params:     NeuroParams = DEFAULT_NEURO_PARAMS,
    control_fn: object = zero_control,
    t0:         float = 8.0,
    max_steps:  int = 16_000,
) -> jnp.ndarray:
    """Advance by n_hours using Kvaerno5 (no pulses)."""
    term   = diffrax.ODETerm(_neuro_continuous_vf)
    solver = diffrax.Kvaerno5()
    ctrl   = diffrax.PIDController(rtol=1e-4, atol=1e-4, dtmax=0.25)

    sol = diffrax.diffeqsolve(
        terms               = term,
        solver              = solver,
        t0                  = t0,
        t1                  = t0 + float(n_hours),
        dt0                 = 0.01,
        y0                  = x0,
        args                = (params, control_fn),
        stepsize_controller = ctrl,
        max_steps           = max_steps,
        adjoint             = diffrax.DirectAdjoint(),
    )
    return sol.ys[-1]


def integrate_Nh_pulsatile(
    x0:         jnp.ndarray,
    n_hours:    float,
    params:     NeuroParams = DEFAULT_NEURO_PARAMS,
    control_fn: object = zero_control,
    rng_key:    jax.Array | None = None,
    t0:         float = 8.0,
    max_steps:  int = 16_000,
) -> jnp.ndarray:
    """
    Advance by n_hours with pulsatile CRH stimulation (Keenan 1997 architecture).

    Pulse pre-sampling (Poisson arrivals + LogNormal amplitudes) occurs entirely
    in Python/NumPy, outside the JAX computation graph. The resulting cumulative
    staircase function is passed as a diffrax.LinearInterpolation ControlTerm.

    diffrax.MultiTerm(ODETerm, ControlTerm) solved by Kvaerno5 (L-stable, vital
    for the stiffness ratio gamma_Epi/gamma_IGF1 ~ 450).

    Parameters
    ----------
    x0         : initial state (STATE_DIM,)
    n_hours    : integration duration [h]
    params     : NeuroParams
    control_fn : hub control callable -- (t) -> (CTRL_DIM,)
    rng_key    : JAX PRNGKey for pulse sampling; defaults to key(0)
    t0         : start time [h]
    max_steps  : diffrax max integration steps

    Returns
    -------
    x_final : (STATE_DIM,) -- state at t0 + n_hours
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Pre-sample pulses (CPU, outside JIT)
    ts_pulses, amp_pulses = sample_cortisol_pulses(rng_key, t0, t0 + n_hours)

    # Build cumulative control path
    pulse_control = _build_pulse_control(float(t0), float(t0 + n_hours), ts_pulses, amp_pulses)

    # Assemble MultiTerm
    continuous_term = diffrax.ODETerm(_neuro_continuous_vf)
    pulse_term      = diffrax.ControlTerm(_pulse_injection_vf, pulse_control)
    terms           = diffrax.MultiTerm(continuous_term, pulse_term)

    solver = diffrax.Kvaerno5()
    ctrl   = diffrax.PIDController(rtol=1e-4, atol=1e-4, dtmax=0.25)

    sol = diffrax.diffeqsolve(
        terms               = terms,
        solver              = solver,
        t0                  = t0,
        t1                  = t0 + float(n_hours),
        dt0                 = 0.01,
        y0                  = x0,
        args                = (params, control_fn),
        stepsize_controller = ctrl,
        max_steps           = max_steps,
        adjoint             = diffrax.DirectAdjoint(),
    )
    return sol.ys[-1]
