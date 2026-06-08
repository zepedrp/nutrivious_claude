"""
app/slices/metabolic_glucose/ode.py -- Metabolic Glucose V2.0

6-state ODE coupled to Neuroendocrine hub variables.
Time unit: MINUTES.

State x in R^6:
    x[0]  Plasma_Glucose_mgdL  [mg/dL]    fasting ~90
    x[1]  Insulin_pmolL        [pmol/L]   fasting ~50
    x[2]  Glucagon_pgmL        [pg/mL]    fasting ~80
    x[3]  Liver_Glycogen_g     [g]        fasting ~70 (LG_max=80)
    x[4]  Muscle_Glycogen_g    [g]        fasting ~350 (MG_max=400)
    x[5]  Lactate_mmolL        [mmol/L]   fasting ~1.0

Hub inputs hubs in R^4 (NaN-guarded via jnp.where(jnp.isnan)):
    hubs[0]  hub_cho_absorption_g_min  [g/min]   from GI slice
    hubs[1]  hub_Epinephrine_pgmL      [pg/mL]   from neuroendocrine SAM axis
    hubs[2]  hub_Cortisol_nmolL        [nmol/L]  from neuroendocrine HPA axis
    hubs[3]  hub_power_watts           [W]        from exercise / training

Physics:
    Plasma_Glucose: Ra_gut (CHO absorption) + EGP (gluconeogenesis + Epi/Gc-
                    stimulated glycogenolysis via Michaelis-Menten) - Rd_insulin
                    (IS suppressed exponentially by chronic Cortisol) - Rd_exercise
    Insulin:        beta-cell glucose-stimulated secretion - clearance
    Glucagon:       basal + hypoglycemia drive + Epi counterregulation - clearance
    Liver_Glycogen: post-meal synthesis (insulin x glucose) - glycogenolysis drain
    Muscle_Glycogen:exercise depletion (power x MG_frac) + insulin recovery
    Lactate:        quadratic anaerobic production (power^2 x MG_frac) - Cori clearance

Steady-state verification (fasting, Epi=50 pg/mL, Cort=300 nmol/L, power=0):
    EGP = 0.312 + 0.764 * 0.875 * 0.70 = 0.78 mg/dL/min
    Rd  = 0.010 * 50 * 90/180 + 0.53   = 0.25 + 0.53 = 0.78  CHECK

References:
    Dalla Man et al. (2007) IEEE TBME 54:1740-1749
    Cryer (2001) J Clin Invest 108:1533-1535
    Coyle et al. (1986) J Appl Physiol 61:165-172
"""
from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

# -- State indices
IDX_G   = 0   # Plasma_Glucose_mgdL
IDX_I   = 1   # Insulin_pmolL
IDX_GC  = 2   # Glucagon_pgmL
IDX_LG  = 3   # Liver_Glycogen_g
IDX_MG  = 4   # Muscle_Glycogen_g
IDX_LAC = 5   # Lactate_mmolL

STATE_DIM: int = 6
OBS_DIM:   int = 2   # [CGM glucose mg/dL, capillary lactate mmol/L]

# -- Hub indices
HUB_CHO  = 0
HUB_EPI  = 1
HUB_CORT = 2
HUB_POW  = 3
HUB_DIM  = 4


class GlucoseMetabParams(NamedTuple):
    """
    Parameters for the 6-state metabolic glucose ODE.

    NLME-identifiable (D=2):
        IS_0   -- insulin sensitivity baseline  Prior: LogN(log 0.010, 0.30^2)
        LG_max -- liver glycogen capacity [g]   Prior: N(80, 10^2)
    """
    # Body composition
    BW:  float = 70.0     # kg
    Vg:  float = 1.88     # dL/kg  glucose distribution volume

    # Basal reference values
    Gb:     float = 90.0    # mg/dL
    Ib:     float = 50.0    # pmol/L
    Gc_b:   float = 80.0    # pg/mL
    Epi_b:  float = 50.0    # pg/mL
    Cort_b: float = 300.0   # nmol/L

    # Glycogen capacities
    LG_max: float = 80.0    # g   [NLME-identifiable D=1]
    MG_max: float = 400.0   # g

    # Basal EGP components (sum = 0.78 mg/dL/min at fasting SS)
    EGP_gng:   float = 0.312   # mg/dL/min  gluconeogenesis background
    k_EGP_gly: float = 0.764   # mg/dL/min  max glycogenolysis rate constant

    # Michaelis-Menten half-saturation for EGP hormonal stimulation
    K_Epi_gly: float = 200.0   # pg/mL
    K_Gc_gly:  float = 80.0    # pg/mL

    # Insulin kinetics
    k_I:    float = 0.126   # min^-1  t1/2 ~5.5 min
    Sb:     float = 6.3     # pmol/L/min  basal secretion (= k_I * Ib)
    K_beta: float = 1.5     # pmol/L/min per mg/dL above Gb

    # Insulin-mediated glucose disposal [NLME-identifiable D=0]
    IS_0:  float = 0.010    # mg/dL/min / (pmol/L)  [NLME D=0]
    Km_G:  float = 90.0     # mg/dL  M-M half-saturation
    Fcns:  float = 0.53     # mg/dL/min  brain + RBC (constant)

    # Cortisol-induced insulin resistance (exponential suppression)
    k_IS_Cort: float = 0.004   # per nmol/L above Cort_b

    # Exercise glucose uptake (AMPK, non-insulin-dependent)
    k_ex: float = 0.003   # mg/dL/min per 100 W

    # Glucagon kinetics
    k_Gc_clear: float = 0.14    # min^-1  t1/2 ~5 min
    k_Gc_sec_b: float = 11.2    # pg/mL/min  basal secretion
    k_Gc_hypo:  float = 0.5     # pg/mL/min per mg/dL below Gb
    k_Gc_Epi:   float = 0.020   # pg/mL/min per pg/mL Epi above Epi_b

    # Liver glycogen synthesis
    k_LG_syn: float = 0.0010    # g/min per pmol/L I per mg/dL above Gb

    # Muscle glycogen kinetics
    k_MG_ex:  float = 0.0104    # g/min per W  (400 W -> ~4 g/min)
    k_MG_syn: float = 0.0003    # g/min per pmol/L I

    # Lactate kinetics
    k_Lac_basal: float = 0.040  # mmol/L/min  resting Cori cycle steady-state
    k_Lac_ex:    float = 0.080  # mmol/L/min  quadratic anaerobic scaling
    k_Lac_clear: float = 0.040  # min^-1  Cori cycle clearance


DEFAULT_PARAMS: GlucoseMetabParams = GlucoseMetabParams()

# -- Default state, hubs, and initial covariance
X0_DEFAULT: jax.Array = jnp.array([
    90.0,    # G    [mg/dL]
    50.0,    # I    [pmol/L]
    80.0,    # Gc   [pg/mL]
    70.0,    # LG   [g]
    350.0,   # MG   [g]
    1.0,     # Lac  [mmol/L]
], dtype=jnp.float32)

HUBS_DEFAULT: jax.Array = jnp.array([
    0.0,    # cho_absorption  [g/min]
    50.0,   # Epinephrine     [pg/mL]
    300.0,  # Cortisol        [nmol/L]
    0.0,    # power           [W]
], dtype=jnp.float32)

P0_DEFAULT: jax.Array = jnp.diag(jnp.array([
    100.0,   # G   sigma^2 = 10^2
    25.0,    # I   sigma^2 = 5^2
    100.0,   # Gc  sigma^2 = 10^2
    25.0,    # LG  sigma^2 = 5^2
    400.0,   # MG  sigma^2 = 20^2
    1.0,     # Lac sigma^2 = 1^2
], dtype=jnp.float32))


# -- Pure ODE: JIT-safe and vmap-safe
def metabolic_glucose_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    6-state metabolic glucose ODE with neuroendocrine hub coupling.

    Parameters
    ----------
    t    : scalar [min]
    x    : (STATE_DIM,)
    args : (GlucoseMetabParams, hubs[HUB_DIM])

    Returns
    -------
    dx/dt : (STATE_DIM,)
    """
    params, hubs = args
    Vd_dL = params.BW * params.Vg   # ~131.6 dL

    # Non-negative state clamps
    G_safe   = jnp.maximum(x[IDX_G],   0.0)
    I_safe   = jnp.maximum(x[IDX_I],   0.0)
    Gc_safe  = jnp.maximum(x[IDX_GC],  0.0)
    LG_safe  = jnp.maximum(x[IDX_LG],  0.0)
    MG_safe  = jnp.maximum(x[IDX_MG],  0.0)
    Lac_safe = jnp.maximum(x[IDX_LAC], 0.0)

    # -- NaN guards on all hub variables (Physics Boundary)
    hub_cho  = jnp.where(jnp.isnan(hubs[HUB_CHO]),  jnp.float32(0.0),   hubs[HUB_CHO])
    hub_Epi  = jnp.where(jnp.isnan(hubs[HUB_EPI]),  jnp.float32(50.0),  hubs[HUB_EPI])
    hub_Cort = jnp.where(jnp.isnan(hubs[HUB_CORT]), jnp.float32(300.0), hubs[HUB_CORT])
    hub_pow  = jnp.where(jnp.isnan(hubs[HUB_POW]),  jnp.float32(0.0),   hubs[HUB_POW])

    hub_cho  = jnp.maximum(hub_cho, 0.0)
    hub_Epi  = jnp.clip(hub_Epi,  0.0, 5000.0)
    hub_Cort = jnp.clip(hub_Cort, 0.0, 2000.0)
    hub_pow  = jnp.clip(hub_pow,  0.0, 1000.0)

    # -- Gut CHO absorption -> plasma glucose rate
    Ra_gut = hub_cho * 1000.0 / Vd_dL   # mg/dL/min

    # -- Hepatic EGP: gluconeogenesis + Michaelis-Menten glycogenolysis
    #    Epi stimulates glycogenolysis (fight-or-flight response)
    #    Glucagon stimulates glycogenolysis (hypoglycemia response)
    Epi_mm  = hub_Epi  / (hub_Epi  + params.K_Epi_gly)   # in [0, 1]
    Gc_mm   = Gc_safe  / (Gc_safe  + params.K_Gc_gly)     # in [0, 1]
    stim    = Epi_mm + Gc_mm                               # in [0, 2]
    LG_frac = LG_safe / params.LG_max                      # in [0, 1]
    Ra_gly   = params.k_EGP_gly * LG_frac * stim
    Ra_liver = params.EGP_gng + Ra_gly

    # -- Insulin-mediated disposal; IS suppressed exponentially by chronic cortisol
    cort_excess = jnp.maximum(hub_Cort - params.Cort_b, 0.0)
    IS_eff  = params.IS_0 * jnp.exp(-params.k_IS_Cort * cort_excess)
    Rd_ins  = IS_eff * I_safe * G_safe / (params.Km_G + G_safe)

    # -- Exercise uptake (AMPK, non-insulin-dependent)
    Rd_ex = params.k_ex * (hub_pow / 100.0) * G_safe / (params.Km_G + G_safe)

    # Block a: dG/dt
    dG = Ra_gut + Ra_liver - Rd_ins - Rd_ex - params.Fcns

    # Block b: dI/dt (beta-cell secretion)
    G_above = jnp.maximum(x[IDX_G] - params.Gb, 0.0)
    S_ins   = params.Sb + params.K_beta * G_above
    dI      = S_ins - params.k_I * I_safe

    # dGc/dt (glucagon)
    hypo_drive   = params.k_Gc_hypo * jnp.maximum(params.Gb - x[IDX_G], 0.0)
    epi_Gc_drive = params.k_Gc_Epi  * jnp.maximum(hub_Epi - params.Epi_b, 0.0)
    Gc_sec       = params.k_Gc_sec_b + hypo_drive + epi_Gc_drive
    dGc          = Gc_sec - params.k_Gc_clear * Gc_safe

    # dLG/dt (liver glycogen: synthesis - glycogenolysis drain)
    gly_rate_g = params.k_EGP_gly * LG_frac * stim * Vd_dL / 1000.0   # g/min
    LG_room    = jnp.maximum(params.LG_max - LG_safe, 0.0) / params.LG_max
    LG_syn     = params.k_LG_syn * I_safe * G_above * LG_room
    dLG        = LG_syn - gly_rate_g

    # Block c: dMG/dt (muscle glycogen depleted mechanically by hub_power_watts)
    MG_frac = MG_safe / params.MG_max
    MG_depl = params.k_MG_ex * hub_pow * MG_frac
    MG_room = jnp.maximum(params.MG_max - MG_safe, 0.0) / params.MG_max
    MG_syn  = params.k_MG_syn * I_safe * G_above * MG_room
    dMG     = MG_syn - MG_depl

    # Block d: dLac/dt (quadratic anaerobic production at threshold + Cori clearance)
    lac_prod_ex = params.k_Lac_ex * (hub_pow / 200.0) ** 2.0 * MG_frac
    dLac = params.k_Lac_basal + lac_prod_ex - params.k_Lac_clear * Lac_safe

    return jnp.stack([dG, dI, dGc, dLG, dMG, dLac])
