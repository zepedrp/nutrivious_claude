"""
app/slices/immune_repair/ode.py -- Immune Repair ODE  (4-state, HOURS)

State  x in R^4:
    x[0]  Muscle_Damage_au      [0, 1]     arbitrary normalised damage units
    x[1]  Macrophage_M1         [0, 1]     pro-inflammatory macrophage density
    x[2]  Macrophage_M2         [0, 1]     anti-inflammatory / repair macrophage density
    x[3]  Interleukin_6_pgmL    [pg/mL]    circulating IL-6 (t1/2 ~ 1 h)

Control (hub variables) u in R^3  (NaN-guarded before use):
    u[0]  hub_training_stress    [0, 1]    eccentric / mechanical loading stress
    u[1]  hub_power_watts        [W]       contractile power (myokine IL-6 source)
    u[2]  hub_Cortisol_nmolL     [nmol/L]  circulating cortisol

Physics
-------
DAMAGE:
    dD/dt = k_dmg * hub_training_stress - k_M2_repair * M2 * D

MACROPHAGE M1  (pro-inflammatory, destructive):
    Recruited proportional to damage.
    Suppressed if Cortisol > Cortisol_immune_threshold (exponential suppression).
    Polarises into M2 at rate k_polar.

    dM1/dt = k_M1_recruit * D * (1 - M1)
             - k_polar * M1
             - k_cort_supp * max(0, Cort - C_threshold) * M1

MACROPHAGE M2  (anti-inflammatory, repair):
    Born from M1 polarisation.  Decays naturally.

    dM2/dt = k_polar * M1 - k_M2_decay * M2

IL-6  (dual source, fast decay t1/2 ~ 1 h):
    Source 1 -- Myokine: proportional to hub_power_watts (contractile, instantaneous).
    Source 2 -- Cytokine: sustained production by M1 (inflammatory).
    gamma_IL6 = ln(2) / t1/2 ~ 0.693 h^-1

    dIL6/dt = k_il6_myo * hub_power_watts
              + k_il6_M1 * M1
              - gamma_IL6 * IL6

NaN Guards
----------
All hub inputs are guarded with jnp.where(jnp.isnan(u), 0.0, u) before use.
All rate terms are wrapped in jnp.where(state < 0, 0.0, term) to prevent
negative-state instability.

References
----------
    Tidball & Villalta (2010) Am J Physiol Cell Physiol 298:C1173
    Peake et al. (2017) J Physiol 595:5981
    Pedersen & Febbraio (2008) Nat Rev Endocrinol 4:71  -- myokine IL-6
    Minetto et al. (2011) J Appl Physiol 111:687        -- CK as damage proxy
    Vanhorebeek et al. (2012) J Clin Endocrinol Metab   -- cortisol immunosuppression
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import diffrax

# ── State indices ──────────────────────────────────────────────────────────────
IDX_DMG  = 0   # Muscle_Damage_au
IDX_M1   = 1   # Macrophage_M1
IDX_M2   = 2   # Macrophage_M2
IDX_IL6  = 3   # Interleukin_6_pgmL

STATE_DIM = 4
OBS_DIM   = 2   # [hsCRP_obs, CK_obs]

# ── Hub indices ────────────────────────────────────────────────────────────────
HUB_STRESS   = 0
HUB_POWER    = 1
HUB_CORTISOL = 2


# ── Parameters ─────────────────────────────────────────────────────────────────

class ImmuneParams(NamedTuple):
    # Damage
    k_dmg:          float = 0.20    # damage rate per unit stress [au/h]
    k_M2_repair:    float = 0.15    # M2-driven repair rate [h^-1]

    # M1 (pro-inflammatory)
    k_M1_recruit:   float = 0.40    # recruitment rate [h^-1]  (Tidball 2010)
    k_polar:        float = 0.08    # M1->M2 polarisation rate [h^-1] (~12h half-life)
    k_cort_supp:    float = 0.004   # cortisol M1 suppression rate [h^-1 per nmol/L excess]
    Cortisol_threshold: float = 550.0  # nmol/L above which suppression fires (Vanhorebeek 2012)

    # M2 (repair)
    k_M2_decay:     float = 0.05    # natural decay [h^-1]

    # IL-6
    k_il6_myo:      float = 0.006   # myokine source [pg/mL/W/h]  (Pedersen 2008)
    k_il6_M1:       float = 5.0     # M1 cytokine source [pg/mL per M1-unit/h]
    gamma_IL6:      float = 0.693   # decay [h^-1]  (t1/2 = 1 h)


DEFAULT_IMMUNE_PARAMS = ImmuneParams()


# ── ODE vector field ───────────────────────────────────────────────────────────

def immune_ode(t: float, x: jnp.ndarray, args) -> jnp.ndarray:
    """
    Right-hand side of the 4-state immune repair ODE.

    args = (params: ImmuneParams, control_fn: callable)
    control_fn(t) -> u in R^3  [hub_training_stress, hub_power_watts, hub_Cortisol_nmolL]
    """
    params, control_fn = args

    u_raw = control_fn(t)
    u     = jnp.where(jnp.isnan(u_raw), 0.0, u_raw)

    stress   = jnp.clip(u[HUB_STRESS],   0.0, 1.0)
    power_w  = jnp.maximum(u[HUB_POWER], 0.0)
    cortisol = jnp.maximum(u[HUB_CORTISOL], 0.0)

    D   = jnp.maximum(x[IDX_DMG], 0.0)
    M1  = jnp.maximum(x[IDX_M1],  0.0)
    M2  = jnp.maximum(x[IDX_M2],  0.0)
    IL6 = jnp.maximum(x[IDX_IL6], 0.0)

    # Cortisol excess above immunosuppression threshold
    cort_excess = jnp.maximum(cortisol - params.Cortisol_threshold, 0.0)

    # BLOCK A: Muscle Damage
    # damage source = training stress; sink = M2-mediated repair
    dD_dt = (params.k_dmg * stress
             - params.k_M2_repair * M2 * D)

    # BLOCK B: Macrophage M1
    # recruited by damage; polarised into M2; suppressed by cortisol excess
    dM1_dt = (params.k_M1_recruit * D * jnp.maximum(1.0 - M1, 0.0)
              - params.k_polar * M1
              - params.k_cort_supp * cort_excess * M1)

    # BLOCK C: Macrophage M2
    # born from M1 polarisation; natural decay
    dM2_dt = (params.k_polar * M1
              - params.k_M2_decay * M2)

    # BLOCK D: IL-6 (dual source)
    # myokine (power-dependent, contractile) + cytokine (M1, inflammatory)
    dIL6_dt = (params.k_il6_myo * power_w
               + params.k_il6_M1 * M1
               - params.gamma_IL6 * IL6)

    return jnp.array([dD_dt, dM1_dt, dM2_dt, dIL6_dt])


# ── Initial state (resting, fully recovered) ──────────────────────────────────

def initial_state() -> jnp.ndarray:
    """Resting baseline: minimal damage, M1 quiescent, no elevated IL-6."""
    return jnp.array([
        0.05,    # Muscle_Damage_au   small background
        0.02,    # Macrophage_M1      quiescent
        0.05,    # Macrophage_M2      low homeostatic
        1.0,     # IL-6 pgmL          basal
    ])


def zero_control(_t):
    return jnp.array([0.0, 0.0, 300.0])   # rest, no power, normal cortisol


# ── Integration helpers ────────────────────────────────────────────────────────

def integrate_Nh(
    x0: jnp.ndarray,
    n_hours: float,
    params: ImmuneParams = DEFAULT_IMMUNE_PARAMS,
    control_fn=None,
    t0: float = 0.0,
) -> jnp.ndarray:
    """Integrate ODE for n_hours; returns final state vector."""
    if control_fn is None:
        control_fn = zero_control

    term   = diffrax.ODETerm(immune_ode)
    solver = diffrax.Tsit5()
    ctrl   = diffrax.PIDController(rtol=1e-4, atol=1e-6)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0    = t0,
        t1    = t0 + n_hours,
        dt0   = 0.01,
        y0    = x0,
        args  = (params, control_fn),
        stepsize_controller = ctrl,
        max_steps = 4096,
    )
    return sol.ys[-1]


def integrate_1h(
    x0: jnp.ndarray,
    params: ImmuneParams = DEFAULT_IMMUNE_PARAMS,
    control_fn=None,
    t0: float = 0.0,
) -> jnp.ndarray:
    return integrate_Nh(x0, n_hours=1.0, params=params, control_fn=control_fn, t0=t0)
