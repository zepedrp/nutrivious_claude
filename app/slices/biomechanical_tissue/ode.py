"""
app/slices/biomechanical_tissue/ode.py

Biomechanical Tissue ODE -- Hourly Timescale (L2 Backbone)

Architecture
------------
Unified sub-hourly physics for two connective-tissue subsystems:

  1. Tendon adaptation (Magnusson 1998; Arampatzis 2007; Pearson 2011)
     Microdamage accumulation, collagen synthesis dynamics, stiffness
     modulation, and acute estrogen-mediated laxity (ACL-risk model
     for females; Shultz 2012 JOSPT).

  2. Bone remodelling at hourly resolution (Frost Mechanostat; Turner 1998)
     Impact-driven microdamage, osteoblast/osteoclast coupling modulated by
     Testosterone (anabolic) and load magnitude (Wolff's Law; Carter 1987).

Time unit: HOURS.

====================================================================
STATE VECTOR  x in R^5  (hourly timescale)
====================================================================

  x[0]  Tendon_Microdamage       Tendon structural microdamage    [au, 0+]
  x[1]  Collagen_Synthesis_Rate  Active collagen turnover rate    [au/h]
  x[2]  Tendon_Stiffness         Tendon mechanical stiffness      [au, 0+]
  x[3]  Bone_Microdamage         Bone structural microdamage      [au, 0+]
  x[4]  Bone_Density             Bone mineral density proxy       [g/cm2]

HUB INPUTS (algebraic, read each step -- NaN Guards applied)  Dim=6
  hub_mechanical_load_au    Normalised mechanical load           [au; 1.0 = ref]
  hub_IGF1_ngmL             Circulating IGF-1                   [ng/mL]
  hub_Testosterone_nmolL    Circulating testosterone             [nmol/L]
  hub_Cortisol_nmolL        Circulating cortisol                 [nmol/L]
  hub_Estrogen_pgmL         Circulating estradiol E2             [pg/mL]
  hub_Nutrition             Nutritional sufficiency gate         [0-1; 1=replete]
                            (Glycine/Proline/VitC; LOX substrate availability)
                            Gate=0 blocks collagen synthesis entirely.

ODE EQUATIONS (time unit: hours)
---------------------------------

BLOCK A -- Tendon Microdamage (Arampatzis 2007; Magnusson 1998)

  dTendon_Microdamage/dt = k_load_tend * hub_load
                         - k_repair_tend * Tendon_Microdamage

BLOCK B -- Collagen Synthesis Rate (Magnusson 1998; Paxton 2010 IGF-1 tendon)

  dCollagen_Synthesis_Rate/dt = (k_IGF1_csr * hub_IGF1) * hub_Nutrition
                               - k_Cort_csr * hub_Cort
                               - k_csr_decay * (CSR - CSR_basal)

  hub_Nutrition is an absolute multiplicative gate: if hub_Nutrition = 0
  (glycine/proline/VitC deficit), synthesis term = 0 regardless of IGF-1.
  Physiological basis: LOX enzyme (lysyl oxidase) requires copper and
  ascorbate; fibrillogenesis requires proline/glycine as substrates.
  Source: Paxton J.Z. et al. (2010) Tissue Eng Part A 16:1387.

  Bone repair continues to use Testosterone (hub_T) via OPG/RANKL pathway
  (Block E, unaffected by Nutrition gate).

BLOCK C -- Tendon Stiffness (Pearson 2011; Shultz 2012 ACL-laxity)

  E2_excess  = max(0, hub_Estrogen - E2_laxity_threshold)
  laxity_fac = k_E2_laxity * E2_excess

  dTendon_Stiffness/dt = k_stiff_synth * CSR
                       - k_stiff_dmg   * Tendon_Microdamage * Tendon_Stiffness
                       - laxity_fac    * Tendon_Stiffness

  Physiological basis: high peri-ovulatory E2 (>200 pg/mL) activates matrix
  metalloproteinases and reduces proteoglycan content, causing acute tendon
  and ligament laxity -- the dominant biomechanical factor in female ACL
  rupture risk (Shultz 2012; Dragoo 2011 Am J Sports Med).

BLOCK D -- Bone Microdamage (Turner 1998; Carter 1987)

  dBone_Microdamage/dt = k_impact * hub_load
                       - k_repair_bone * Bone_Microdamage

BLOCK E -- Bone Density (Frost Mechanostat; Carter 1987 Wolff's Law;
           RANK-RANKL modulated by Testosterone)

  load_moderate = hub_load * (1 - hub_load / load_pathology_thresh)
  load_anabolic = max(0, load_moderate)

  dBone_Density/dt = k_wolff * hub_T_norm * load_anabolic
                   - k_resorb * Bone_Microdamage * Bone_Density

  hub_T_norm = hub_T / T_reference_nmolL  (unit-normalised)

  Physiological basis: Testosterone upregulates OPG, suppressing RANKL-driven
  osteoclastogenesis (Smith 2001; Vanderschueren 2004). Moderate cyclical load
  triggers sclerostin inhibition -> Wnt/beta-catenin -> osteoblast anabolic
  response (Robling 2008). Excessive load accumulates microdamage faster than
  repair, causing net resorption (stress fracture pathway; Bennell 1996).

HUB OUTPUTS (algebraic -- no new state, computed on demand)
------------------------------------------------------------

  Hub_Tendon_Rupture_Risk:
    load_stress    = hub_load / max(Tendon_Stiffness, eps)
    risk_signal    = alpha_tend * load_stress + beta_tend * Tendon_Microdamage
    = sigmoid(risk_signal, k=k_sig_tend, x0=x0_sig_tend)

  Hub_Bone_Stress_Fracture_Risk:
    dmg_density    = Bone_Microdamage / max(Bone_Density, eps)
    = sigmoid(dmg_density, k=k_sig_bone, x0=x0_sig_bone)

Fail-Loud / Design invariants
------------------------------
  * Pure JAX: no Python-level conditionals on traced values.
  * jnp.maximum for all smooth gates; jnp.tanh for smooth sigmoids.
  * JIT + vmap safe (sigma-point propagation for UKF).
  * Fail-Loud: NaN propagates -- no silent substitution.
  * Hub inputs guarded: NaN -> 0.0 via jnp.nan_to_num before jnp.maximum.
  * All genetic scale factors arrive pre-resolved in params.

References
----------
  Magnusson S.P. et al. (1998) J Physiol 513:899-907        [collagen turnover]
  Arampatzis A. et al. (2007) J Biomech 40:2704-2710        [tendon microdamage]
  Pearson S.J. et al. (2011) Eur J Appl Physiol 111:1523    [stiffness adaptation]
  Shultz S.J. et al. (2012) JOSPT 42:640-649                [E2 ACL laxity]
  Dragoo J.L. et al. (2011) Am J Sports Med 39:2081-2086    [E2 ligament laxity]
  Frost H.M. (2003) Anat Rec 275A:1081-1101                 [mechanostat]
  Carter D.R. (1987) Calcif Tissue Int 41:76                [Wolff's Law]
  Turner C.H. (1998) Bone 23:399-407                        [bone mechanostat]
  Smith M.R. et al. (2001) J Clin Endocrinol Metab 86:2787  [T -> OPG]
  Robling A.G. et al. (2008) Nature 461:316-320             [sclerostin Wnt]
  Bennell K.L. et al. (1996) Am J Sports Med 24:810-818     [stress fracture]
  Hansen M. et al. (2009) J Appl Physiol 106:1281-1287      [E2 collagen synth]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# -- State vector indices ------------------------------------------------------
IDX_TEND_DMG   = 0   # Tendon_Microdamage       [au, 0+]
IDX_CSR        = 1   # Collagen_Synthesis_Rate  [au/h]
IDX_TEND_STIFF = 2   # Tendon_Stiffness         [au, 0+]
IDX_BONE_DMG   = 3   # Bone_Microdamage         [au, 0+]
IDX_BONE_DENS  = 4   # Bone_Density             [g/cm2]

STATE_DIM: int = 5
OBS_DIM:   int = 3   # [Pain_VAS, Ultrasound_Echogenicity, DEXA_ZScore_proxy]

# -- Hub input indices (control vector u)  Dim=6 --------------------------------
UIDX_LOAD      = 0   # hub_mechanical_load_au    [au; 1.0 = ref]
UIDX_IGF1      = 1   # hub_IGF1_ngmL             [ng/mL]
UIDX_TESTO     = 2   # hub_Testosterone_nmolL    [nmol/L]
UIDX_CORTISOL  = 3   # hub_Cortisol_nmolL        [nmol/L]
UIDX_ESTROGEN  = 4   # hub_Estrogen_pgmL         [pg/mL]
UIDX_NUTRITION = 5   # hub_Nutrition             [0-1]

CTRL_DIM: int = 6

# Default hub values at population resting state
# [load, IGF1, T, Cortisol, E2, Nutrition]
HUBS_DEFAULT: list[float] = [0.0, 150.0, 20.0, 300.0, 50.0, 1.0]


# -- Parameter container -------------------------------------------------------

class BiomechanicalParams(NamedTuple):
    """
    Biomechanical Tissue ODE parameters -- hourly timescale.

    Prior sources
    -------------
    Tendon repair    : Magnusson 1998 (collagen half-life ~60-90 days -> k_repair_tend
                       ~0.008-0.012 h^-1); Arampatzis 2007 (damage accumulation)
    Collagen synth   : Hansen 2009 (T -> collagen; Cortisol -> MMP3 suppression)
    Stiffness        : Pearson 2011 (adaptation rates)
    E2 laxity        : Shultz 2012 (peri-ovulatory E2 > 200 pg/mL -> acute laxity)
    Bone repair      : Turner 1998 (microdamage repair ~weeks -> k_repair_bone ~0.001 h^-1)
    Wolff / OPG      : Carter 1987; Smith 2001 (Testosterone -> OPG -> bone anabolic)
    Sigmoids         : calibrated to produce risk -> 0.85 at load/stiffness = 2.0
                       and Bone_Microdamage/Bone_Density > 0.8
    """
    # -- Tendon microdamage (Block A) ------------------------------------------
    k_load_tend:    float = 0.030   # h^-1  -- damage per unit load per hour
    k_repair_tend:  float = 0.010   # h^-1  -- basal tendon repair (tau ~ 100 h)

    # -- Collagen synthesis rate (Block B) -------------------------------------
    k_IGF1_csr:     float = 0.001   # h^-1 per ng/mL -- IGF-1 -> CSR synthesis
    k_Cort_csr:     float = 0.0005  # h^-1 per nmol/L -- Cortisol -> CSR suppression
    k_csr_decay:    float = 0.05    # h^-1  -- CSR return to basal (tau ~ 20 h)
    CSR_basal:      float = 0.10    # au/h  -- resting collagen synthesis rate
    # Calibrated so that at HUBS_DEFAULT (IGF1=150, Cortisol=300, Nutrition=1.0):
    # synthesis = 0.001*150*1.0 = 0.15; suppression = 0.0005*300 = 0.15
    # -> dCSR/dt = 0 at CSR_basal (neutral resting equilibrium)

    # -- Tendon stiffness (Block C) --------------------------------------------
    k_stiff_synth:  float = 0.004   # h^-1  -- CSR -> stiffness gain
    k_stiff_dmg:    float = 0.020   # h^-1  -- damage-driven stiffness loss
    k_E2_laxity:    float = 0.0003  # h^-1 per pg/mL excess -- acute E2 laxity
    E2_laxity_thr:  float = 200.0   # pg/mL -- peri-ovulatory laxity threshold

    # -- Bone microdamage (Block D) --------------------------------------------
    k_impact:       float = 0.003   # h^-1  -- bone damage per unit load per hour
    k_repair_bone:  float = 0.010   # h^-1  -- bone repair (tau ~ 100 h ~ 4 days)
    # Calibrated so bone_dmg equilibrium = k_impact/k_repair_bone = 0.3 au at load=1.0
    # Resorption at eq: k_resorb * 0.3 * BMD ~ 3.5e-5 g/cm2/h
    # Wolff formation (T=25, load=1.0): ~5.6e-5 g/cm2/h -> net positive ✓

    # -- Bone density (Block E) ------------------------------------------------
    k_wolff:        float = 0.00005 # g/cm2/h -- Wolff's Law anabolic rate
    k_resorb:       float = 0.0001  # g/cm2/h per unit DMG*BMD -- resorption
    T_ref_nmolL:    float = 15.0    # nmol/L  -- reference T for normalisation
    load_path_thr:  float = 3.0     # au      -- pathological overload threshold

    # -- Hub output sigmoid parameters -----------------------------------------
    # Tendon rupture risk: sigmoid on (alpha*load_stress + beta*tend_dmg)
    alpha_tend:     float = 0.60    # au^-1  -- load/stiffness weight
    beta_tend:      float = 0.40    # au^-1  -- microdamage weight
    k_sig_tend:     float = 10.0    # au^-1  -- sigmoid sharpness
    x0_sig_tend:    float = 0.70    # au     -- sigmoid midpoint (risk ~ 0.5 here)

    # Bone stress fracture risk: sigmoid on damage/density ratio
    k_sig_bone:     float = 15.0    # au^-1  -- sigmoid sharpness
    x0_sig_bone:    float = 0.50    # au     -- sigmoid midpoint

    # -- Genetic scales (resolved from weak priors; 1.0 = population mean) -----
    col5a1_scale:   float = 1.0     # [0.7, 1.3] -- COL5A1 -> tendon stiffness
    mmp3_scale:     float = 1.0     # [0.5, 2.0] -- MMP3 enzymatic aggressiveness


def build_bio_params(**overrides: float) -> BiomechanicalParams:
    """Build BiomechanicalParams with optional keyword overrides."""
    p = BiomechanicalParams()
    return p._replace(**overrides)


DEFAULT_BIO_PARAMS: BiomechanicalParams = build_bio_params()


# -- Default initial conditions ------------------------------------------------
# Rest state: no active microdamage; CSR at basal; stiffness intact; BMD nominal.

_p = DEFAULT_BIO_PARAMS

X0_BIO_DEFAULT: jax.Array = jnp.array([
    0.0,    # Tendon_Microdamage   -- no damage at rest
    _p.CSR_basal,  # Collagen_Synthesis_Rate -- basal turnover
    1.0,    # Tendon_Stiffness     -- normalized (1.0 = healthy)
    0.0,    # Bone_Microdamage     -- no damage at rest
    1.15,   # Bone_Density         -- lumbar spine population reference [g/cm2]
], dtype=jnp.float32)

P0_BIO_DEFAULT: jax.Array = jnp.diag(jnp.array([
    0.04,   # Tendon_Microdamage   sigma^2
    0.01,   # Collagen_Synthesis_Rate sigma^2
    0.04,   # Tendon_Stiffness     sigma^2
    0.04,   # Bone_Microdamage     sigma^2
    0.04,   # Bone_Density         sigma^2 ~ (0.2 g/cm2)^2 DEXA precision
], dtype=jnp.float32))


# -- Hub output functions (algebraic) -----------------------------------------

def hub_tendon_rupture_risk(
    x:      jax.Array,
    u:      jax.Array,
    params: BiomechanicalParams,
) -> jax.Array:
    """
    Hub_Tendon_Rupture_Risk in [0, 1] -- sigmoid on load/stiffness + damage.

    Returns scalar float32.
    """
    hub_load     = jnp.maximum(jnp.nan_to_num(u[UIDX_LOAD], nan=0.0), jnp.float32(0.0))
    tend_dmg     = jnp.maximum(x[IDX_TEND_DMG],   jnp.float32(0.0))
    tend_stiff   = jnp.maximum(x[IDX_TEND_STIFF], jnp.float32(1e-4))

    load_stress  = hub_load / tend_stiff
    risk_signal  = params.alpha_tend * load_stress + params.beta_tend * tend_dmg
    logit        = params.k_sig_tend * (risk_signal - params.x0_sig_tend)
    return jnp.float32(0.5) * (jnp.float32(1.0) + jnp.tanh(logit * jnp.float32(0.5)))


def hub_bone_stress_fracture_risk(
    x:      jax.Array,
    params: BiomechanicalParams,
) -> jax.Array:
    """
    Hub_Bone_Stress_Fracture_Risk in [0, 1] -- sigmoid on damage/density ratio.

    Returns scalar float32.
    """
    bone_dmg   = jnp.maximum(x[IDX_BONE_DMG],  jnp.float32(0.0))
    bone_dens  = jnp.maximum(x[IDX_BONE_DENS], jnp.float32(0.1))

    dmg_ratio  = bone_dmg / bone_dens
    logit      = params.k_sig_bone * (dmg_ratio - params.x0_sig_bone)
    return jnp.float32(0.5) * (jnp.float32(1.0) + jnp.tanh(logit * jnp.float32(0.5)))


# -- Pure ODE (JIT + vmap safe) ------------------------------------------------

def biomechanical_ode(
    t:    jax.Array,
    x:    jax.Array,
    args: tuple,
) -> jax.Array:
    """
    Biomechanical Tissue ODE -- hourly timescale.

    Pure JAX -- no Python conditionals on traced values. JIT + vmap safe.

    Parameters
    ----------
    t    : scalar -- current time [hours]
    x    : shape (STATE_DIM,) -- current state
    args : tuple(BiomechanicalParams, u)
           u : shape (CTRL_DIM,) -- hub inputs at this step (NaN-guarded)

    Returns
    -------
    dx/dt : shape (STATE_DIM,)
    """
    params, u = args

    # -- Unpack state ----------------------------------------------------------
    tend_dmg   = x[IDX_TEND_DMG]
    csr        = x[IDX_CSR]
    tend_stiff = x[IDX_TEND_STIFF]
    bone_dmg   = x[IDX_BONE_DMG]
    bone_dens  = x[IDX_BONE_DENS]

    # -- NaN guards on hub inputs (NaN -> 0) then physical floor ---------------
    hub_load  = jnp.maximum(jnp.nan_to_num(u[UIDX_LOAD],      nan=0.0), jnp.float32(0.0))
    hub_IGF1  = jnp.maximum(jnp.nan_to_num(u[UIDX_IGF1],      nan=0.0), jnp.float32(0.0))
    hub_T     = jnp.maximum(jnp.nan_to_num(u[UIDX_TESTO],     nan=0.0), jnp.float32(0.0))
    hub_Cort  = jnp.maximum(jnp.nan_to_num(u[UIDX_CORTISOL],  nan=0.0), jnp.float32(0.0))
    hub_E2    = jnp.maximum(jnp.nan_to_num(u[UIDX_ESTROGEN],  nan=0.0), jnp.float32(0.0))
    # Nutrition gate clamped to [0, 1]: 0 = full substrate deficit, 1 = replete
    hub_Nutr  = jnp.clip(jnp.nan_to_num(u[UIDX_NUTRITION], nan=1.0),
                         jnp.float32(0.0), jnp.float32(1.0))

    # -- Positive-definite clamps on state -------------------------------------
    tend_dmg_pos   = jnp.maximum(tend_dmg,   jnp.float32(0.0))
    csr_pos        = jnp.maximum(csr,         jnp.float32(0.0))
    tend_stiff_pos = jnp.maximum(tend_stiff,  jnp.float32(1e-4))
    bone_dmg_pos   = jnp.maximum(bone_dmg,    jnp.float32(0.0))
    bone_dens_pos  = jnp.maximum(bone_dens,   jnp.float32(0.1))

    # -- BLOCK A: Tendon microdamage -------------------------------------------
    # Repair is amplified by CSR: high collagen synthesis = faster structural repair
    k_repair_eff = params.k_repair_tend * (jnp.float32(1.0) + csr_pos)
    dTendDmg_dt = (
        params.k_load_tend * hub_load
        - k_repair_eff * tend_dmg_pos
    )

    # -- BLOCK B: Collagen synthesis rate (IGF-1 + Nutrition gate) -----------
    # IGF-1 drives CSR; Nutrition (0-1) is an absolute multiplicative gate
    # (LOX substrate availability: Glycine/Proline/VitC).
    # Cortisol suppresses CSR linearly; first-order return to basal.
    dCSR_dt = (
        (params.k_IGF1_csr * hub_IGF1) * hub_Nutr
        - params.k_Cort_csr * hub_Cort
        - params.k_csr_decay * (csr_pos - params.CSR_basal)
    )

    # -- BLOCK C: Tendon stiffness (Pearson 2011 + Shultz 2012 E2 laxity) -----
    # E2_excess: amount above peri-ovulatory threshold; drives acute laxity
    E2_excess    = jnp.maximum(hub_E2 - params.E2_laxity_thr, jnp.float32(0.0))
    laxity_rate  = params.k_E2_laxity * params.col5a1_scale * E2_excess

    dTendStiff_dt = (
        params.k_stiff_synth * csr_pos
        - params.k_stiff_dmg * tend_dmg_pos * tend_stiff_pos
        - laxity_rate        * tend_stiff_pos
    )

    # -- BLOCK D: Bone microdamage --------------------------------------------
    dBoneDmg_dt = (
        params.k_impact     * hub_load
        - params.k_repair_bone * bone_dmg_pos
    )

    # -- BLOCK E: Bone density (Wolff's Law + RANKL/OPG balance) --------------
    # load_moderate: anabolic window -- diminishes above pathological threshold
    # Uses a smooth quadratic suppression: positive only below load_path_thr
    hub_load_norm  = hub_load / jnp.maximum(params.load_path_thr, jnp.float32(1e-4))
    load_anabolic  = jnp.maximum(
        hub_load * (jnp.float32(1.0) - hub_load_norm),
        jnp.float32(0.0),
    )
    T_norm         = hub_T / jnp.maximum(params.T_ref_nmolL, jnp.float32(1e-4))

    dBoneDens_dt = (
        params.k_wolff * T_norm * load_anabolic
        - params.k_resorb * bone_dmg_pos * bone_dens_pos
    )

    return jnp.stack([
        dTendDmg_dt,
        dCSR_dt,
        dTendStiff_dt,
        dBoneDmg_dt,
        dBoneDens_dt,
    ])
