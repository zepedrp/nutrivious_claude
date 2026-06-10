"""
app/slices/metabolic_glucose/ode.py -- Metabolic Glucose V3.0

5-state ODE. Muscle_Glycogen_g (old x[4]) REMOVED: the Neuromuscular slice is
the sole authority on local glycogen (via hub_local_gly_norm). This eliminates
the thermodynamic double-accounting of a shared metabolite.

Mass-conservation protocol (operator-splitting):
  [1] NM predict step advances Muscle_Glycogen_mmolkg -> publishes hub_local_gly_norm
  [2] MG predict step reads hub_local_gly_norm to scale anaerobic lactate flux
  [3] Cori Cycle: elevated plasma lactate -> hepatic gluconeogenesis (Ra_cori)

State x in R^5:
    x[0]  Plasma_Glucose_mgdL  [mg/dL]    fasting ~90
    x[1]  Insulin_pmolL        [pmol/L]   fasting ~50
    x[2]  Glucagon_pgmL        [pg/mL]    fasting ~80
    x[3]  Liver_Glycogen_g     [g]        fasting ~70 (LG_max=80)
    x[4]  Lactate_mmolL        [mmol/L]   fasting ~1.0

Hub inputs hubs in R^5 (NaN-guarded):
    hubs[0]  hub_cho_absorption_g_min  [g/min]   from GI slice
    hubs[1]  hub_Epinephrine_pgmL      [pg/mL]   from neuroendocrine SAM axis
    hubs[2]  hub_Cortisol_nmolL        [nmol/L]  from neuroendocrine HPA axis
    hubs[3]  hub_power_watts           [W]        from exercise / training
    hubs[4]  hub_local_gly_norm        [0-1]      from NM slice (sole glycogen owner)

Physics:
    Plasma_Glucose: Ra_gut + EGP (gluconeogenesis + Epi/Gc glycogenolysis)
                    + Ra_cori (Cori cycle: plasma lactate -> hepatic gluconeogenesis)
                    - Rd_insulin (IS suppressed exponentially by chronic Cortisol)
                    - Rd_exercise
    Insulin:        beta-cell glucose-stimulated secretion - clearance
    Glucagon:       basal + hypoglycemia drive + Epi counterregulation - clearance
    Liver_Glycogen: post-meal synthesis (insulin x glucose) - glycogenolysis drain
    Lactate:        quadratic anaerobic production (power^2 x local_gly_norm)
                    + basal Cori - Cori clearance

Cori Cycle connection (NM -> MG):
    hub_local_gly_norm gates anaerobic glycolytic flux (lac_prod_ex).
    Elevated plasma lactate accelerates hepatic gluconeogenesis (Ra_cori),
    which quantifies the Cori cycle as a direct NM->MG substrate bridge.

Steady-state verification (fasting, Epi=50 pg/mL, Cort=300 nmol/L, power=0,
                           local_gly_norm=1.0, Lac=1.0):
    Ra_cori = k_Lac_cori * max(0, 1.0 - 1.0) = 0.0   (inactive at rest)
    EGP = 0.312 + 0.764 * 0.875 * 0.70 = 0.78 mg/dL/min
    Rd  = 0.010 * 50 * 90/180 + 0.53   = 0.25 + 0.53 = 0.78  CHECK

References:
    Dalla Man et al. (2007) IEEE TBME 54:1740-1749
    Cryer (2001) J Clin Invest 108:1533-1535
    Coyle et al. (1986) J Appl Physiol 61:165-172
    Cori C.F. (1929) J Biol Chem 81:389  [Cori cycle: lactate -> liver glucose]
    Brooks G.A. (2018) Cell Metab 27:757-785  [lactate shuttle]
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
IDX_LAC = 4   # Lactate_mmolL   (was x[5] in V2.0)

STATE_DIM: int = 5
OBS_DIM:   int = 2   # [CGM glucose mg/dL, capillary lactate mmol/L]

# -- Hub indices
HUB_CHO     = 0
HUB_EPI     = 1
HUB_CORT    = 2
HUB_POW     = 3
HUB_GLY_NORM = 4   # hub_local_gly_norm from NM slice [0-1]
HUB_DIM: int = 5


class GlucoseMetabParams(NamedTuple):
    """
    Parameters for the 5-state metabolic glucose ODE V3.0.

    NLME-identifiable (D=2):
        IS_0   -- insulin sensitivity baseline  Prior: LogN(log 0.010, 0.30^2)
        LG_max -- liver glycogen capacity [g]   Prior: N(80, 10^2)

    Muscle_Glycogen removed: NM is sole owner (V3.0).
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

    # Glycogen capacities (liver only; muscle belongs to NM)
    LG_max: float = 80.0    # g   [NLME-identifiable D=1]

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

    # Lactate kinetics
    k_Lac_basal: float = 0.040  # mmol/L/min  resting Cori cycle steady-state
    k_Lac_ex:    float = 0.080  # mmol/L/min  quadratic anaerobic scaling
    k_Lac_clear: float = 0.040  # min^-1  Cori cycle clearance

    # Cori Cycle: plasma lactate -> hepatic gluconeogenesis
    # Ra_cori = k_Lac_cori * max(0, Lac - 1.0) [mg/dL/min per mmol/L excess]
    # Inactive at fasting (Lac=1.0); contributes ~0.14 mg/dL/min at Lac=8.
    # Calibrated: ~18% of resting EGP per 7 mmol/L excess lactate (Brooks 2018).
    k_Lac_cori: float = 0.020   # mg/dL/min per mmol/L above basal


DEFAULT_PARAMS: GlucoseMetabParams = GlucoseMetabParams()

# -- Default state and hubs (5-state V3.0)
X0_DEFAULT: jax.Array = jnp.array([
    90.0,    # G    [mg/dL]
    50.0,    # I    [pmol/L]
    80.0,    # Gc   [pg/mL]
    70.0,    # LG   [g]
    1.0,     # Lac  [mmol/L]
], dtype=jnp.float32)

HUBS_DEFAULT: jax.Array = jnp.array([
    0.0,    # cho_absorption    [g/min]
    50.0,   # Epinephrine       [pg/mL]
    300.0,  # Cortisol          [nmol/L]
    0.0,    # power             [W]
    1.0,    # local_gly_norm    [0-1]  full glycogen at rest
], dtype=jnp.float32)

P0_DEFAULT: jax.Array = jnp.diag(jnp.array([
    100.0,   # G   sigma^2 = 10^2
    25.0,    # I   sigma^2 = 5^2
    100.0,   # Gc  sigma^2 = 10^2
    25.0,    # LG  sigma^2 = 5^2
    1.0,     # Lac sigma^2 = 1^2
], dtype=jnp.float32))


# -- Pure ODE: JIT-safe and vmap-safe
def metabolic_glucose_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    5-state metabolic glucose ODE V3.0 with hub coupling.

    Parameters
    ----------
    t    : scalar [min]
    x    : (STATE_DIM,) = (5,)
    args : (GlucoseMetabParams, hubs[HUB_DIM])

    Returns
    -------
    dx/dt : (STATE_DIM,) = (5,)
    """
    params, hubs = args
    Vd_dL = params.BW * params.Vg   # ~131.6 dL

    # Non-negative state clamps
    G_safe   = jnp.maximum(x[IDX_G],   0.0)
    I_safe   = jnp.maximum(x[IDX_I],   0.0)
    Gc_safe  = jnp.maximum(x[IDX_GC],  0.0)
    LG_safe  = jnp.maximum(x[IDX_LG],  0.0)
    Lac_safe = jnp.maximum(x[IDX_LAC], 0.0)

    # -- NaN guards on all hub variables
    hub_cho      = jnp.where(jnp.isnan(hubs[HUB_CHO]),      jnp.float32(0.0),   hubs[HUB_CHO])
    hub_Epi      = jnp.where(jnp.isnan(hubs[HUB_EPI]),      jnp.float32(50.0),  hubs[HUB_EPI])
    hub_Cort     = jnp.where(jnp.isnan(hubs[HUB_CORT]),     jnp.float32(300.0), hubs[HUB_CORT])
    hub_pow      = jnp.where(jnp.isnan(hubs[HUB_POW]),      jnp.float32(0.0),   hubs[HUB_POW])
    hub_gly_norm = jnp.where(jnp.isnan(hubs[HUB_GLY_NORM]), jnp.float32(1.0),   hubs[HUB_GLY_NORM])

    hub_cho      = jnp.maximum(hub_cho, 0.0)
    hub_Epi      = jnp.clip(hub_Epi,  0.0, 5000.0)
    hub_Cort     = jnp.clip(hub_Cort, 0.0, 2000.0)
    hub_pow      = jnp.clip(hub_pow,  0.0, 1000.0)
    hub_gly_norm = jnp.clip(hub_gly_norm, 0.0, 2.0)   # guard: NM glycogen norm

    # -- Gut CHO absorption -> plasma glucose rate
    Ra_gut = hub_cho * 1000.0 / Vd_dL   # mg/dL/min

    # -- Hepatic EGP: gluconeogenesis + Michaelis-Menten glycogenolysis
    Epi_mm  = hub_Epi  / (hub_Epi  + params.K_Epi_gly)
    Gc_mm   = Gc_safe  / (Gc_safe  + params.K_Gc_gly)
    stim    = Epi_mm + Gc_mm
    LG_frac = LG_safe / params.LG_max
    Ra_gly  = params.k_EGP_gly * LG_frac * stim
    Ra_liver = params.EGP_gng + Ra_gly

    # -- Cori Cycle: plasma lactate excess -> hepatic gluconeogenesis
    # Inactive at basal lactate (1.0 mmol/L); activates during exercise.
    # Connects NM anaerobic glycolysis (via elevated plasma Lac) to hepatic EGP.
    lac_excess = jnp.maximum(Lac_safe - jnp.float32(1.0), jnp.float32(0.0))
    Ra_cori    = jnp.float32(params.k_Lac_cori) * lac_excess

    # -- Insulin-mediated disposal; IS suppressed exponentially by chronic cortisol
    cort_excess = jnp.maximum(hub_Cort - params.Cort_b, 0.0)
    IS_eff  = params.IS_0 * jnp.exp(-params.k_IS_Cort * cort_excess)
    Rd_ins  = IS_eff * I_safe * G_safe / (params.Km_G + G_safe)

    # -- Exercise uptake (AMPK, non-insulin-dependent)
    Rd_ex = params.k_ex * (hub_pow / 100.0) * G_safe / (params.Km_G + G_safe)

    # dG/dt: includes Cori cycle gluconeogenesis
    dG = Ra_gut + Ra_liver + Ra_cori - Rd_ins - Rd_ex - params.Fcns

    # dI/dt (beta-cell secretion)
    G_above = jnp.maximum(x[IDX_G] - params.Gb, 0.0)
    S_ins   = params.Sb + params.K_beta * G_above
    dI      = S_ins - params.k_I * I_safe

    # dGc/dt (glucagon)
    hypo_drive   = params.k_Gc_hypo * jnp.maximum(params.Gb - x[IDX_G], 0.0)
    epi_Gc_drive = params.k_Gc_Epi  * jnp.maximum(hub_Epi - params.Epi_b, 0.0)
    Gc_sec       = params.k_Gc_sec_b + hypo_drive + epi_Gc_drive
    dGc          = Gc_sec - params.k_Gc_clear * Gc_safe

    # dLG/dt (liver glycogen: synthesis - glycogenolysis drain)
    gly_rate_g = params.k_EGP_gly * LG_frac * stim * Vd_dL / 1000.0
    LG_room    = jnp.maximum(params.LG_max - LG_safe, 0.0) / params.LG_max
    LG_syn     = params.k_LG_syn * I_safe * G_above * LG_room
    dLG        = LG_syn - gly_rate_g

    # dLac/dt: anaerobic production scaled by NM local_gly_norm (hub coupling)
    # Cori clearance term represents lactate uptake by liver for gluconeogenesis.
    lac_prod_ex = params.k_Lac_ex * (hub_pow / 200.0) ** 2.0 * hub_gly_norm
    dLac = params.k_Lac_basal + lac_prod_ex - params.k_Lac_clear * Lac_safe

    return jnp.stack([dG, dI, dGc, dLG, dLac])
