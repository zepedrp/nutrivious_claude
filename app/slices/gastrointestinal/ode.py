"""
app/slices/gastrointestinal/ode.py  --  GI Slice V3.0

6-state intra-session ODE (time unit: MINUTES).
Dual-transporter absorption: SGLT1 (glucose, Na-dependent) + GLUT5 (fructose, Na-independent).

STATE VECTOR  x (6,)  [MINUTES]
  x[0]  Stomach_Fluid_L   gastric fluid volume        [L]
  x[1]  Stom_Glu_g        glucose in stomach          [g]
  x[2]  Stom_Fru_g        fructose in stomach         [g]
  x[3]  Intst_Glu_g       glucose in small intestine  [g]
  x[4]  Intst_Fru_g       fructose in small intestine [g]
  x[5]  GI_Distress_au    distress accumulator        [0,1]

HUB INPUTS  u (6,)
  u[0]  Fluid_in      fluid ingestion rate    [L/min]
  u[1]  Glu_in        glucose intake rate     [g/min]
  u[2]  Fru_in        fructose intake rate    [g/min]
  u[3]  Power         mechanical output       [W]
  u[4]  Temp          core temperature        [degC]
  u[5]  Sodium_mmolL  plasma sodium           [mmol/L]  default 140.0

NaN guards: fluid/glu/fru/power -> 0.0; temp -> 37.0; sodium -> 140.0.

PHYSICS V3.0
  a) Osmotic brake:
       density = (Stom_Glu + Stom_Fru) / max(Stom_Fluid, 1e-4)  [g/L]
       osmotic_brake = exp(-k_osm * max(0, density - 60.0))
  b) Ischaemia factor:
       power_norm = max(0, Power - 200) / 300
       temp_norm  = max(0, Temp  - 38.5) / 2.0
       isch_factor = exp(-k_isch * (power_norm + temp_norm))
  c) Sodium factor (SGLT1 only):
       na_factor = clip((Sodium_mmolL - 125.0) / 10.0, 0.0, 1.0)
  d) Gastric emptying:
       GER = k_ge * osmotic_brake * isch_factor  [min^-1]
       dStom_Fluid = Fluid_in - GER * Stom_Fluid
       dStom_Glu   = Glu_in   - GER * Stom_Glu
       dStom_Fru   = Fru_in   - GER * Stom_Fru
  e) Intestinal absorption:
       abs_glu = Vmax_glu * Intst_Glu / (Km + Intst_Glu) * isch_factor * na_factor  [SGLT1]
       abs_fru = Vmax_fru * Intst_Fru / (Km + Intst_Fru) * isch_factor              [GLUT5, no Na]
       dIntst_Glu = GER * Stom_Glu - abs_glu
       dIntst_Fru = GER * Stom_Fru - abs_fru
  f) Distress:
       total_intst = Intst_Glu + Intst_Fru
       vol_stim = max(0, Stom_Fluid - Vol_tolerance) * 5.0
       cho_stim = max(0, total_intst - 15.0) / 15.0
       stim = clip(vol_stim + cho_stim, 0, 1)
       dDistress = k_rise * stim * (1 - Distress) - k_decay * Distress

HUB EXPORTS (algebraic):
  hub_glu_absorption_rate  [g/min]  -> Mod 1
  hub_fru_absorption_rate  [g/min]  -> Mod 1
  hub_cho_absorption_rate  [g/min]  = glu + fru
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ── state indices ─────────────────────────────────────────────────────────────
IDX_STOM_FLUID = 0
IDX_STOM_GLU   = 1
IDX_STOM_FRU   = 2
IDX_INTST_GLU  = 3
IDX_INTST_FRU  = 4
IDX_DISTRESS   = 5

# ── control indices ───────────────────────────────────────────────────────────
UIDX_FLUID  = 0
UIDX_GLU_IN = 1
UIDX_FRU_IN = 2
UIDX_POWER  = 3
UIDX_TEMP   = 4
UIDX_SODIUM = 5

STATE_DIM: int = 6
OBS_DIM:   int = 2   # [Nausea_VAS, Bloating_VAS]
CTRL_DIM:  int = 6

_EPS = jnp.float32(1e-4)


class GIv3Params(NamedTuple):
    # osmotic brake
    k_osm:          float = 0.05    # [L/g]  -- exponential decay rate above 60 g/L
    osm_threshold:  float = 60.0    # [g/L]

    # ischaemia
    k_isch:         float = 6.0     # sensitivity to normalised power+temp excess
    power_thresh:   float = 200.0   # [W]
    power_range:    float = 300.0   # [W]
    temp_thresh:    float = 38.5    # [degC]
    temp_range:     float = 2.0     # [degC]

    # gastric emptying
    k_ge:           float = 0.04    # [min^-1]  t1/2 ~ 17 min

    # absorption -- NLME-personalised
    Vmax_glu:       float = 1.0     # [g/min]  SGLT1 glucose capacity
    Vmax_fru:       float = 0.6     # [g/min]  GLUT5 fructose capacity
    Km:             float = 10.0    # [g]      shared half-saturation constant

    # distress dynamics
    Vol_tolerance:  float = 0.8     # [L]  elastic stomach threshold
    k_rise:         float = 0.05    # [min^-1]
    k_decay:        float = 0.01    # [min^-1]


DEFAULT_GI_PARAMS: GIv3Params = GIv3Params()

X0_GI_DEFAULT: jax.Array = jnp.array(
    [0.1, 5.0, 2.0, 3.0, 1.0, 0.0], dtype=jnp.float32
)

P0_GI_DEFAULT: jax.Array = jnp.diag(jnp.array(
    [0.01, 4.0, 1.0, 1.0, 0.25, 1e-4], dtype=jnp.float32
))


# ── helpers ───────────────────────────────────────────────────────────────────

def _ng(v: jax.Array, default: float) -> jax.Array:
    """NaN guard."""
    return jnp.where(jnp.isnan(v), jnp.float32(default), v)


@jax.jit
def gi_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    GI V3.0 ODE -- pure JAX, JIT+vmap safe.

    Parameters
    ----------
    t    : scalar [min]
    x    : (STATE_DIM,) = (6,)
    args : (GIv3Params, u)  u shape (CTRL_DIM,) = (6,)
    """
    params, u = args

    # positivity clamp
    stom_fluid = jnp.maximum(x[IDX_STOM_FLUID], jnp.float32(0.0))
    stom_glu   = jnp.maximum(x[IDX_STOM_GLU],   jnp.float32(0.0))
    stom_fru   = jnp.maximum(x[IDX_STOM_FRU],   jnp.float32(0.0))
    intst_glu  = jnp.maximum(x[IDX_INTST_GLU],  jnp.float32(0.0))
    intst_fru  = jnp.maximum(x[IDX_INTST_FRU],  jnp.float32(0.0))
    distress   = jnp.clip(x[IDX_DISTRESS], jnp.float32(0.0), jnp.float32(1.0))

    # hub inputs with NaN guards
    fluid_in = jnp.maximum(_ng(u[UIDX_FLUID],  0.0), jnp.float32(0.0))
    glu_in   = jnp.maximum(_ng(u[UIDX_GLU_IN], 0.0), jnp.float32(0.0))
    fru_in   = jnp.maximum(_ng(u[UIDX_FRU_IN], 0.0), jnp.float32(0.0))
    power    = jnp.maximum(_ng(u[UIDX_POWER],  0.0), jnp.float32(0.0))
    temp     = _ng(u[UIDX_TEMP],   37.0)
    sodium   = _ng(u[UIDX_SODIUM], 140.0)

    # a) osmotic brake
    density = (stom_glu + stom_fru) / (stom_fluid + _EPS)
    density_excess = jnp.maximum(density - jnp.float32(params.osm_threshold), jnp.float32(0.0))
    osmotic_brake  = jnp.exp(-jnp.float32(params.k_osm) * density_excess)

    # b) ischaemia factor
    pn = jnp.maximum(power - jnp.float32(params.power_thresh), jnp.float32(0.0)) / jnp.float32(params.power_range)
    tn = jnp.maximum(temp  - jnp.float32(params.temp_thresh),  jnp.float32(0.0)) / jnp.float32(params.temp_range)
    isch_factor = jnp.exp(-jnp.float32(params.k_isch) * (pn + tn))

    # c) sodium factor (SGLT1 only)
    na_factor = jnp.clip(
        (sodium - jnp.float32(125.0)) / jnp.float32(10.0),
        jnp.float32(0.0), jnp.float32(1.0),
    )

    # d) gastric emptying
    GER        = jnp.float32(params.k_ge) * osmotic_brake * isch_factor
    flux_fluid = GER * stom_fluid
    flux_glu   = GER * stom_glu
    flux_fru   = GER * stom_fru

    dStom_Fluid = fluid_in - flux_fluid
    dStom_Glu   = glu_in   - flux_glu
    dStom_Fru   = fru_in   - flux_fru

    # e) intestinal absorption
    abs_glu = (
        jnp.float32(params.Vmax_glu)
        * intst_glu / (jnp.float32(params.Km) + intst_glu)
        * isch_factor * na_factor
    )
    abs_fru = (
        jnp.float32(params.Vmax_fru)
        * intst_fru / (jnp.float32(params.Km) + intst_fru)
        * isch_factor
        # GLUT5 does NOT require sodium
    )

    dIntst_Glu = flux_glu - abs_glu
    dIntst_Fru = flux_fru - abs_fru

    # f) distress
    total_intst = intst_glu + intst_fru
    vol_stim = jnp.maximum(stom_fluid - jnp.float32(params.Vol_tolerance), jnp.float32(0.0)) * jnp.float32(5.0)
    cho_stim = jnp.maximum(total_intst - jnp.float32(15.0), jnp.float32(0.0)) / jnp.float32(15.0)
    stim     = jnp.clip(vol_stim + cho_stim, jnp.float32(0.0), jnp.float32(1.0))
    dDistress = (
        jnp.float32(params.k_rise) * stim * (jnp.float32(1.0) - distress)
        - jnp.float32(params.k_decay) * distress
    )

    return jnp.stack([
        dStom_Fluid, dStom_Glu, dStom_Fru,
        dIntst_Glu,  dIntst_Fru, dDistress,
    ])


# ── algebraic hub exports ─────────────────────────────────────────────────────

@jax.jit
def _absorption_rates(
    x:      jax.Array,
    u:      jax.Array,
    params: GIv3Params = DEFAULT_GI_PARAMS,
) -> tuple[jax.Array, jax.Array]:
    """Return (abs_glu, abs_fru) [g/min]."""
    intst_glu = jnp.maximum(x[IDX_INTST_GLU], jnp.float32(0.0))
    intst_fru = jnp.maximum(x[IDX_INTST_FRU], jnp.float32(0.0))
    power  = jnp.maximum(_ng(u[UIDX_POWER],  0.0), jnp.float32(0.0))
    temp   = _ng(u[UIDX_TEMP],   37.0)
    sodium = _ng(u[UIDX_SODIUM], 140.0)
    pn     = jnp.maximum(power - jnp.float32(params.power_thresh), jnp.float32(0.0)) / jnp.float32(params.power_range)
    tn     = jnp.maximum(temp  - jnp.float32(params.temp_thresh),  jnp.float32(0.0)) / jnp.float32(params.temp_range)
    isch   = jnp.exp(-jnp.float32(params.k_isch) * (pn + tn))
    na_f   = jnp.clip((sodium - 125.0) / 10.0, jnp.float32(0.0), jnp.float32(1.0))
    ag = jnp.float32(params.Vmax_glu) * intst_glu / (jnp.float32(params.Km) + intst_glu) * isch * na_f
    af = jnp.float32(params.Vmax_fru) * intst_fru / (jnp.float32(params.Km) + intst_fru) * isch
    return ag, af


@jax.jit
def hub_cho_absorption_rate(
    x:      jax.Array,
    u:      jax.Array,
    params: GIv3Params = DEFAULT_GI_PARAMS,
) -> jax.Array:
    """Total CHO absorption rate [g/min]."""
    ag, af = _absorption_rates(x, u, params)
    return ag + af
