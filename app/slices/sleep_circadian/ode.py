"""
app/slices/sleep_circadian/ode.py  —  Sleep-Circadian ODE V2.0

State vector x ∈ ℝ⁵  (time unit = HOURS)
══════════════════════════════════════════
  x[0]  SCN_x           circadian pacemaker x-component   [dimensionless]
  x[1]  SCN_y           circadian pacemaker velocity       [dimensionless]
  x[2]  Adenosine       homeostatic sleep pressure         [a.u., 0–2]
  x[3]  Melatonin       plasma melatonin                   [pg/mL]
  x[4]  Sleep_Drive_SWS slow-wave sleep drive              [0=wake, 1=SWS]

Hub INPUTS (passed as HubInputs NamedTuple via diffrax args):
  hub_light_lux       ambient illuminance [lux]   (0 = dark; 10 000 = bright office)
  hub_training_stress normalised training stress  [0–1]
  hub_Cortisol_nmolL  plasma cortisol             [nmol/L]  (normal night ≈ 50–150)

Hub OUTPUTS (algebraic, exported for downstream slices):
  Hub_Circadian_Phase    = arctan2(SCN_y, SCN_x)   [rad, −π..π]
  Hub_Sleep_SWS_Fraction = x[4]

Physics
───────
SCN (Forger-Jewett-Kronauer 1999):
  light_frac = clip(hub_light_lux / 10 000, 0, 1)
  B = G_phot · light_frac                          [photic drive]
  dSCN_x/dt = (π/12)·(SCN_y + B)
  dSCN_y/dt = (π/12)·[μ·(SCN_x − 4/3·SCN_x³) − ω²·(SCN_x + α·SCN_y) − κ·B]
  ω² = (24/(0.99729·τ_c))²

Adenosine (Borbély Process S — modified):
  wake_frac = 1 − Sleep_Drive_SWS
  dAden/dt  = wake_frac · r_acc · (1 + k_stress · hub_training_stress) − SWS · Aden · k_clear
  (k_clear is the NLME-personalised parameter; r_acc fixed at population mean)

Melatonin (Lewy 1999; Cajochen 2000):
  mel_light_gate = max(0, 1 − hub_light_lux/100)        ← 0 when lux ≥ 100
  cort_inhib     = 1/(1 + (hub_Cortisol_nmolL/K_cort)²) ← cortisol stress insomnia
  M_drive = max(0, −SCN_x) · mel_light_gate · cort_inhib
  dMel/dt  = k_sec · M_drive − k_mel · Mel

Sleep_Drive_SWS (AND of adenosine-gate × melatonin-gate):
  aden_gate  = σ(k_Aden · (Aden − Aden_thr))
  mel_gate   = σ(k_Mel  · (Mel  − Mel_thr ))
  SWS_eq     = aden_gate · mel_gate
  dSWS/dt    = (SWS_eq − SWS) / τ_sws

NaN guards on all hub inputs: jnp.where(jnp.isnan(hub_x), fallback, hub_x)

References
──────────
  Forger D.B. et al. (1999) J Biol Rhythms 14(6):532–537
  Borbély A.A. (1982) Hum Neurobiol 1(3):195–204
  Lewy A.J. et al. (1999) J Biol Rhythms 14(4):315–321
  Cajochen C. et al. (2000) J Biol Rhythms 15(2):86–95  (cortisol × melatonin)
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

# ── Index constants ────────────────────────────────────────────────────────────
IDX_SCN_X    = 0
IDX_SCN_Y    = 1
IDX_ADENOSINE = 2
IDX_MELATONIN = 3
IDX_SWS      = 4

STATE_DIM = 5
OBS_DIM   = 2   # [SWS_proxy, CBT_nadir_phase_rad]

_PI_OVER_12 = math.pi / 12.0


# ── Hub inputs container ───────────────────────────────────────────────────────

class HubInputs(NamedTuple):
    """Hub variables entering the ODE from other slices / environment."""
    hub_light_lux:       float = 0.0    # lux  (dark night)
    hub_training_stress: float = 0.0    # [0–1]
    hub_Cortisol_nmolL:  float = 100.0  # nmol/L (normal nighttime baseline)


DEFAULT_HUBS = HubInputs()


# ── ODE parameters ─────────────────────────────────────────────────────────────

class SleepParams(NamedTuple):
    """
    ODE parameters for the sleep-circadian V2 system.

    NLME-personalised (D=2):
        tau_c    ~ N(24.18, 0.20²)  h    Forger 1999; Czeisler 1999 Science 284:2177
        k_clear  ~ LogN(log 0.13, 0.20²) h⁻¹  individual adenosine clearance rate
                   (t½ = ln2/k_clear; faster = genetically efficient glymphatic clearance)

    Population-fixed (not identifiable from daily wearables):
        μ_c, α_c, κ_c, G_phot  — FJK oscillator
        r_acc, k_stress         — adenosine accumulation
        k_sec, k_mel            — melatonin kinetics
        K_cort                  — cortisol inhibition EC50
        Aden_thr, Mel_thr       — SWS AND-gate thresholds
        k_Aden, k_Mel           — SWS AND-gate steepness
        τ_sws                   — SWS relaxation time constant
    """
    # ── NLME-personalised ─────────────────────────────────────────────────────
    tau_c:    float = 24.18   # h   intrinsic circadian period
    k_clear:  float = 0.13    # h⁻¹ adenosine clearance rate (t½ ≈ 5.3 h)

    # ── FJK oscillator (fixed population) ────────────────────────────────────
    mu_c:     float = 0.23    # VDP nonlinearity
    alpha_c:  float = 0.16    # cross-coupling
    kappa_c:  float = 0.55    # photic coupling (Jewett 1999)
    G_phot:   float = 0.097   # photic drive gain (Kronauer 1999)

    # ── Adenosine (fixed population) ──────────────────────────────────────────
    r_acc:    float = 0.055   # a.u./h  baseline wake accumulation rate
    k_stress: float = 0.50    # dimensionless  training stress multiplier

    # ── Melatonin (fixed population) ──────────────────────────────────────────
    k_sec:    float = 7.0     # pg/mL/h  secretion rate
    k_mel:    float = 0.77    # h⁻¹      clearance (t½ ≈ 54 min)
    K_cort:   float = 500.0   # nmol/L   cortisol inhibition EC50 (squared sigmoid)

    # ── SWS drive gate (fixed population) ────────────────────────────────────
    Aden_thr: float = 0.50    # a.u.     adenosine threshold for SWS
    Mel_thr:  float = 25.0    # pg/mL    melatonin threshold for SWS
    k_Aden:   float = 10.0    # steepness of adenosine sigmoid
    k_Mel:    float = 0.10    # steepness of melatonin sigmoid
    tau_sws:  float = 0.40    # h        SWS relaxation time constant


DEFAULT_SLEEP_PARAMS = SleepParams()


# ── Initial conditions ─────────────────────────────────────────────────────────

def initial_state(params: SleepParams = DEFAULT_SLEEP_PARAMS) -> jnp.ndarray:
    """
    Physiologically plausible state at midnight for an average chronotype.

    SCN at midnight: approaching temperature nadir (~4 AM → C_x minimum).
    SCN_x ≈ −0.60 (subjective night), SCN_y ≈ +0.20 (advancing toward dawn).
    Adenosine ≈ 0.55 (after ~15 h wake since 7 AM; ~2 h into sleep).
    Melatonin ≈ 50 pg/mL (onset ~2 h prior).
    SWS ≈ 0.45 (transitioning into first NREM cycle).
    """
    return jnp.array([
        -0.60,   # SCN_x
         0.20,   # SCN_y
         0.55,   # Adenosine
        50.0,    # Melatonin [pg/mL]
         0.45,   # Sleep_Drive_SWS
    ])


# ── Hub output functions (algebraic) ──────────────────────────────────────────

def hub_circadian_phase(x: jnp.ndarray) -> jnp.ndarray:
    """Hub_Circadian_Phase = arctan2(SCN_y, SCN_x) in [-π, π] rad."""
    return jnp.arctan2(x[IDX_SCN_Y], x[IDX_SCN_X])


def hub_sws_fraction(x: jnp.ndarray) -> jnp.ndarray:
    """Hub_Sleep_SWS_Fraction = Sleep_Drive_SWS state."""
    return x[IDX_SWS]


# ── Vector field ───────────────────────────────────────────────────────────────

class SleepVectorField(diffrax.AbstractTerm):
    """
    diffrax-compatible vector field for the Sleep-Circadian V2 ODE.

    Usage:
        term = SleepVectorField(params, hubs)
        sol  = diffrax.diffeqsolve(term, solver, t0, t1, dt0, x0, args=None)

    `hubs` is a HubInputs NamedTuple with constant values for the integration
    window (time-varying inputs can be passed via linear interpolation externally).
    """
    params: SleepParams
    hubs:   HubInputs

    def vf(self, t: jnp.ndarray, x: jnp.ndarray, args: object) -> jnp.ndarray:
        p = self.params
        h = self.hubs

        SCN_x  = x[IDX_SCN_X]
        SCN_y  = x[IDX_SCN_Y]
        Aden   = x[IDX_ADENOSINE]
        Mel    = x[IDX_MELATONIN]
        SWS    = x[IDX_SWS]

        # ── NaN guards on all hub inputs ──────────────────────────────────────
        lux    = jnp.where(jnp.isnan(h.hub_light_lux),       0.0,   h.hub_light_lux)
        stress = jnp.where(jnp.isnan(h.hub_training_stress),  0.0,   h.hub_training_stress)
        cort   = jnp.where(jnp.isnan(h.hub_Cortisol_nmolL),  100.0, h.hub_Cortisol_nmolL)

        # ── SCN: Forger-Jewett-Kronauer photic drive ──────────────────────────
        light_frac = jnp.clip(lux / 10_000.0, 0.0, 1.0)
        B = p.G_phot * light_frac

        omega_sq = (24.0 / (0.99729 * p.tau_c)) ** 2
        dSCN_x = _PI_OVER_12 * (SCN_y + B)
        dSCN_y = _PI_OVER_12 * (
            p.mu_c * (SCN_x - (4.0 / 3.0) * SCN_x ** 3)
            - omega_sq * (SCN_x + p.alpha_c * SCN_y)
            - p.kappa_c * B
        )

        # ── Adenosine: accumulates in wake, clears in SWS ────────────────────
        wake_frac = 1.0 - SWS
        stress_mult = 1.0 + p.k_stress * jnp.clip(stress, 0.0, 1.0)
        dAden = wake_frac * p.r_acc * stress_mult - SWS * Aden * p.k_clear

        # ── Melatonin: light-suppressed + cortisol-inhibited ─────────────────
        mel_light_gate = jnp.maximum(0.0, 1.0 - lux / 100.0)
        cort_inhib = 1.0 / (1.0 + (cort / p.K_cort) ** 2)
        M_drive = jnp.maximum(0.0, -SCN_x) * mel_light_gate * cort_inhib
        dMel = p.k_sec * M_drive - p.k_mel * Mel

        # ── Sleep_Drive_SWS: AND-gate (adenosine HIGH and melatonin HIGH) ─────
        aden_gate = jax.nn.sigmoid(p.k_Aden * (Aden - p.Aden_thr))
        mel_gate  = jax.nn.sigmoid(p.k_Mel  * (Mel  - p.Mel_thr))
        SWS_eq    = aden_gate * mel_gate
        dSWS      = (SWS_eq - SWS) / p.tau_sws

        return jnp.array([dSCN_x, dSCN_y, dAden, dMel, dSWS])

    def contr(self, t0: jnp.ndarray, t1: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return t1 - t0

    def prod(self, vf: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        return vf * control


# ── Integration helpers ────────────────────────────────────────────────────────

def integrate_hours(
    x0: jnp.ndarray,
    hubs: HubInputs = DEFAULT_HUBS,
    params: SleepParams = DEFAULT_SLEEP_PARAMS,
    t_hours: float = 0.5,
    t0: float = 0.0,
    dt0: float = 0.05,
    max_steps: int = 4_000,
) -> jnp.ndarray:
    """
    Advance the sleep-circadian state by `t_hours` hours.

    Tsit5 (non-stiff RK45) with PIDController; slowest mode ≈ τ_sws = 0.4 h,
    fastest adenosine clearance ≈ 0.13 h⁻¹ → stiffness ratio ≈ 3×, well within Tsit5.
    """
    term   = SleepVectorField(params=params, hubs=hubs)
    solver = diffrax.Tsit5()
    ctrl   = diffrax.PIDController(rtol=1e-5, atol=1e-7, dtmax=0.1)

    sol = diffrax.diffeqsolve(
        terms   = term,
        solver  = solver,
        t0      = t0,
        t1      = t0 + t_hours,
        dt0     = dt0,
        y0      = x0,
        stepsize_controller = ctrl,
        max_steps           = max_steps,
        adjoint             = diffrax.DirectAdjoint(),
    )
    return sol.ys[-1]


def simulate_nsteps(
    x0: jnp.ndarray,
    hubs_sequence: jnp.ndarray,   # (T, 3) — [lux, stress, cortisol] per step
    params: SleepParams = DEFAULT_SLEEP_PARAMS,
    dt_hours: float = 0.5,
) -> jnp.ndarray:
    """
    Simulate T half-hour steps. Returns trajectory (T+1, STATE_DIM).

    hubs_sequence shape: (T, 3) where columns are [lux, stress, cort].
    Uses jax.lax.scan for JIT-compilable multi-step rollout.
    """
    def step(carry: jnp.ndarray, hub_row: jnp.ndarray) -> tuple:
        h = HubInputs(
            hub_light_lux       = hub_row[0],
            hub_training_stress = hub_row[1],
            hub_Cortisol_nmolL  = hub_row[2],
        )
        x_next = integrate_hours(carry, hubs=h, params=params,
                                  t_hours=dt_hours, t0=0.0)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, hubs_sequence)
    return jnp.concatenate([x0[None], xs], axis=0)
