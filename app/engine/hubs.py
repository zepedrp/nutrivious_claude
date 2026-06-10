"""
app/engine/hubs.py

Gaussian Message Passing Hub -- Inter-Slice Mass Conservation Layer.

PURPOSE
-------
Replace scalar hub pass-through with probabilistic (mean, variance) tuples
in information form. Enables:
    1. Uncertainty propagation across slice boundaries.
    2. Bayesian fusion when multiple slices contribute to the same quantity.
    3. Explicit mass-conservation protocol for the NM <-> Metabolic glycogen
       coupling (Local_Glycogen_mmolkg vs. Systemic_Glycogen_g).

INFORMATION FORM
----------------
We use the canonical (information) form of a Gaussian:
    eta       = J * mu        (information vector)
    precision = J = 1/sigma^2 (precision; scalar for 1D, matrix for nD)

Fusion of N independent sources is O(N) addition:
    J_fused   = sum(J_i)
    eta_fused = sum(eta_i)
    mu_fused  = eta_fused / J_fused

This avoids the product-of-Gaussians formula (which requires inversion) and
is numerically stable when precisions differ by many orders of magnitude.

NM <-> METABOLIC OPERATOR-SPLITTING PROTOCOL
--------------------------------------------
The NM and Metabolic slices have a circular dependency:
    NM    reads  hub_plasma_glucose  from Metabolic ODE state
    Metab reads  hub_local_gly_norm  from NM ODE state

Breaking the circle with one-step staggered operator splitting:
    Step t:
        [1] Metabolic predict  uses local_gly_norm from step t-1 (stale by 1 dt)
        [2] NM predict         uses plasma_glucose  from step [1] (fresh)
        [3] NM publishes       hub_local_gly_norm(t) via nm_glycogen_to_hub()
        [4] Metabolic update   corrects lactate term with fresh gly hub

    Stale error bound (1-min dt, tau_gly ~ 5-30 min):
        |error| < dt/tau_gly ~ 0.03-0.20  --> acceptable at 1-min resolution.

DESIGN INVARIANTS
-----------------
    - GaussianMsg is a JAX-compatible NamedTuple (valid pytree leaf).
    - All arithmetic is via jnp (JIT/vmap safe).
    - No Python-level branching on array values.
    - HubState is immutable (NamedTuple); orchestrator creates new instances.

References
----------
Koller D. & Friedman N. (2009) Probabilistic Graphical Models, MIT Press.
    [Gaussian belief propagation; information-form message passing]
Loeliger H.-A. (2004) IEEE Signal Process. Mag. 21(1):28-41.
    [Factor graph belief propagation; canonical form fusion]
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── Information-form Gaussian message ─────────────────────────────────────────

class GaussianMsg(NamedTuple):
    """
    Scalar Gaussian message in information (canonical) form.

    Fields
    ------
    eta       : information value  eta = precision * mean
    precision : information        precision = 1 / variance  (>= 0)

    Recover mean    : mu    = eta / precision
    Recover variance: sigma2 = 1 / precision

    JAX pytree: NamedTuple leaves are valid JAX arrays -- JIT/vmap safe.
    """
    eta:       jax.Array   # scalar float32
    precision: jax.Array   # scalar float32 >= 0


# ── Constructors ──────────────────────────────────────────────────────────────

def msg_from_scalar(
    mu:     float | jax.Array,
    sigma2: float | jax.Array,
) -> GaussianMsg:
    """
    Build a GaussianMsg from N(mu, sigma^2).

    Parameters
    ----------
    mu     : mean (scalar)
    sigma2 : variance (scalar, > 0)

    Returns
    -------
    GaussianMsg in information form.
    """
    J   = jnp.asarray(1.0, dtype=jnp.float32) / jnp.maximum(
        jnp.asarray(sigma2, dtype=jnp.float32), jnp.float32(1e-30)
    )
    eta = jnp.asarray(mu, dtype=jnp.float32) * J
    return GaussianMsg(eta=eta, precision=J)


def msg_uninformative() -> GaussianMsg:
    """
    Completely uninformative (flat prior) message.
    Fusing with this message leaves the result unchanged.
    """
    return GaussianMsg(
        eta=jnp.float32(0.0),
        precision=jnp.float32(0.0),
    )


# ── Accessors ─────────────────────────────────────────────────────────────────

def msg_to_mean(msg: GaussianMsg) -> jax.Array:
    """Extract posterior mean from information-form message."""
    return msg.eta / jnp.maximum(msg.precision, jnp.float32(1e-30))


def msg_to_variance(msg: GaussianMsg) -> jax.Array:
    """Extract posterior variance from information-form message."""
    return jnp.float32(1.0) / jnp.maximum(msg.precision, jnp.float32(1e-30))


def msg_to_std(msg: GaussianMsg) -> jax.Array:
    """Extract posterior standard deviation from information-form message."""
    return jnp.sqrt(msg_to_variance(msg))


# ── Bayesian fusion ───────────────────────────────────────────────────────────

def fuse_gaussian(*msgs: GaussianMsg) -> GaussianMsg:
    """
    Bayesian fusion of N independent Gaussian sources (information form).

    Equivalent to the product of N Gaussian PDFs:
        p(x) = prod_i N(x | mu_i, sigma_i^2)
             proportional to N(x | mu_fused, sigma_fused^2)

    O(N) in the number of sources -- no matrix inversion.

    Parameters
    ----------
    *msgs : one or more GaussianMsg to fuse.

    Returns
    -------
    GaussianMsg with fused eta and precision.

    Raises
    ------
    ValueError if no messages provided.
    """
    if not msgs:
        raise ValueError("fuse_gaussian requires at least one message.")
    J_total   = sum(m.precision for m in msgs)
    eta_total = sum(m.eta       for m in msgs)
    return GaussianMsg(eta=eta_total, precision=J_total)


# ── Hub variable name constants ───────────────────────────────────────────────

class HubName:
    """
    Canonical string keys for all inter-slice hub variables.

    Each slice ODE reads and/or writes these names via HubState fields.
    Names use snake_case with physical units embedded to prevent silent
    unit mismatches across slice boundaries.
    """
    # Glycolytic coupling (NM <-> Metabolic)
    PLASMA_GLUCOSE_MGDL   = "hub_plasma_glucose_mgdL"
    PLASMA_LACTATE_MMOLL  = "hub_plasma_lactate_mmolL"
    LOCAL_GLY_NORM        = "hub_local_glycogen_norm"    # NM -> Metabolic  [0-1]
    SYSTEMIC_GLY_G        = "hub_systemic_glycogen_g"    # Metabolic -> GI, others

    # Neuroendocrine -> downstream
    CORTISOL_NMOLL        = "hub_cortisol_nmolL"
    EPINEPHRINE_PGML      = "hub_epinephrine_pgmL"
    TESTOSTERONE_NGDL     = "hub_testosterone_ngdL"
    IGF1_NGML             = "hub_igf1_ngmL"

    # Cardiorespiratory -> downstream
    HR_BPM                = "hub_hr_bpm"
    VO2_MLKGMIN           = "hub_vo2_mLkgmin"
    AUTONOMIC_TONE        = "hub_autonomic_tone"
    W_PRIME_BAL_KJ        = "hub_w_prime_bal_kJ"

    # Thermo / Renal (Mod 12 sovereign)
    CORE_TEMP_C           = "hub_core_temp_C"
    PV_DROP_PCT           = "hub_pv_drop_pct"
    PLASMA_NA_MMOLL       = "hub_plasma_Na_mmolL"

    # Energy availability (RED-S Mod 13)
    ENERGY_AVAIL_KCAL_KG  = "hub_energy_avail_kcal_kgFFM"

    # Immune / repair
    IL6_PGML              = "hub_IL6_pgmL"
    GH_NGML               = "hub_GH_ngmL"

    # GI absorption
    CHO_ABS_G_MIN         = "hub_cho_absorption_g_min"
    FRU_ABS_G_MIN         = "hub_fru_absorption_g_min"


# ── NM <-> Metabolic glycogen hub functions ───────────────────────────────────

_NM_GLY_REF: float = 100.0   # NMv4 Glycogen_ref [mmol/kg dm]


def nm_glycogen_to_hub(
    gly_mmolkg:  float | jax.Array,
    sigma2_gly:  float | jax.Array = 25.0,
) -> GaussianMsg:
    """
    Convert NM Local_Glycogen_mmol UKF posterior to a normalised hub message.

    Called by the NM slice after each UKF update to publish:
        Hub_Local_Glycogen_norm = Glycogen_mmolkg / _NM_GLY_REF   [0-1 range]

    The normalisation propagates uncertainty correctly:
        sigma2_norm = sigma2_gly / _NM_GLY_REF^2

    Parameters
    ----------
    gly_mmolkg  : UKF posterior mean of Local_Glycogen_mmol [mmol/kg dm]
    sigma2_gly  : UKF posterior variance of Glycogen state  [(mmol/kg)^2]
                  Default 25.0 = (5 mmol/kg)^2 -- typical post-update uncertainty.

    Returns
    -------
    GaussianMsg for Hub_Local_Glycogen_norm (information form, dimensionless).
    """
    norm_mean   = jnp.asarray(gly_mmolkg, dtype=jnp.float32) / jnp.float32(_NM_GLY_REF)
    norm_sigma2 = jnp.asarray(sigma2_gly, dtype=jnp.float32) / jnp.float32(_NM_GLY_REF ** 2)
    return msg_from_scalar(norm_mean, norm_sigma2 + jnp.float32(1e-8))


def hub_to_metabolic_gly_frac(
    msg: GaussianMsg,
) -> tuple[jax.Array, jax.Array]:
    """
    Extract glycogen fraction (mean, variance) for the Metabolic ODE lactate term.

    The Metabolic ODE replaces its own MG_frac with the hub value:
        lac_prod_ex = k_Lac_ex * (hub_pow / 200)^2 * hub_gly_norm

    Returns
    -------
    (gly_norm_mean, gly_norm_var) -- both scalar float32, mean clipped to [0, 2].
    """
    mean = jnp.clip(msg_to_mean(msg), jnp.float32(0.0), jnp.float32(2.0))
    var  = msg_to_variance(msg)
    return mean, var


# ── Full hub state container ──────────────────────────────────────────────────

class HubState(NamedTuple):
    """
    Immutable snapshot of all inter-slice hub variables (information form).

    The orchestrator maintains one HubState per timestep and passes it
    to each slice in operator-splitting order. Each slice returns a new
    HubState with its owned fields updated.

    Parallel Federated Filtering topology (intra-session, 1-min steps):
        All organs receive the same frozen hub_t snapshot and publish to hub_{t+1}.
        No organ reads another organ's same-step output.

        hub_t  ->  [NM || MG || Neuroendocrine || ThermoRenal || Cardio]  ->  hub_{t+1}

    Daily-scale slices (Gonadal, Hematological, Bone) run once per day
    using the session-end HubState as their control input.

    Use msg_to_mean(field) to extract the scalar mean for ODE args[].
    Use msg_to_variance(field) for uncertainty-aware couplings.
    """
    plasma_glucose:   GaussianMsg   # [mg/dL]          Metabolic -> NM, Neuroendocrine
    plasma_lactate:   GaussianMsg   # [mmol/L]          Metabolic -> NM, Neural
    local_gly_norm:   GaussianMsg   # [0-1, norm]       NM -> Metabolic (lactate term)
    systemic_gly_g:   GaussianMsg   # [g]               Metabolic -> GI, Hematological
    cortisol:         GaussianMsg   # [nmol/L]          Neuroendocrine -> Metabolic, NM, Immune
    epinephrine:      GaussianMsg   # [pg/mL]           Neuroendocrine -> Metabolic, Cardio
    testosterone:     GaussianMsg   # [ng/dL]           Gonadal -> NM repair, Bone
    igf1:             GaussianMsg   # [ng/mL]           Neuroendocrine -> NM repair, Bone
    autonomic_tone:   GaussianMsg   # [0-1]             Cardio -> Neural, NMPC
    w_prime_bal_kJ:   GaussianMsg   # [kJ]              Cardio -> NMPC, PSF
    core_temp:        GaussianMsg   # [deg C]           Thermo -> Cardio, GI, Neural
    pv_drop_pct:      GaussianMsg   # [%]               Renal (sovereign) -> Cardio, Hemato
    energy_avail:     GaussianMsg   # [kcal/kgFFM/day]  REDS -> Gonadal, Neuroendocrine
    il6:              GaussianMsg   # [pg/mL]           Immune -> Hemato, Neuroendocrine
    cho_absorption:   GaussianMsg   # [g/min]           GI -> Metabolic
    basal_temp_offset: GaussianMsg  # [°C]              Gonadal/P4 -> ThermoRenal setpoint
    anabolic_drive:   GaussianMsg   # [0-1]             T/E2 -> NM repair, MG insulin sensitivity


def default_hub_state() -> HubState:
    """
    Population-prior hub state for cold-start (no session data yet).

    All precision values are low (wide priors) to reflect maximum uncertainty
    before any slice UKF has run. The orchestrator replaces each field with
    the actual UKF posterior at the end of each timestep.
    """
    def _w(mu: float, sigma2: float) -> GaussianMsg:
        return msg_from_scalar(mu, sigma2)

    return HubState(
        plasma_glucose  = _w(90.0,   100.0),   # 90 +/- 10 mg/dL
        plasma_lactate  = _w(1.0,    0.25),     # 1 +/- 0.5 mmol/L
        local_gly_norm  = _w(1.0,    0.04),     # 1.0 +/- 0.2 (full tank)
        systemic_gly_g  = _w(350.0,  1600.0),   # 350 +/- 40 g
        cortisol        = _w(300.0,  2500.0),   # 300 +/- 50 nmol/L
        epinephrine     = _w(50.0,   625.0),    # 50 +/- 25 pg/mL
        testosterone    = _w(600.0,  40000.0),  # 600 +/- 200 ng/dL
        igf1            = _w(200.0,  3600.0),   # 200 +/- 60 ng/mL
        autonomic_tone  = _w(0.5,    0.04),     # 0.5 +/- 0.2
        w_prime_bal_kJ  = _w(18.0,   16.0),     # 18 +/- 4 kJ (full)
        core_temp       = _w(37.0,   0.25),     # 37 +/- 0.5 deg C
        pv_drop_pct     = _w(0.0,    4.0),      # 0 +/- 2 %
        energy_avail       = _w(45.0,   100.0),    # 45 +/- 10 kcal/kgFFM
        il6                = _w(1.0,    1.0),      # 1 +/- 1 pg/mL
        cho_absorption     = _w(0.0,    0.01),     # 0 g/min (rest)
        basal_temp_offset  = _w(0.0,    0.01),     # 0 +/- 0.1 °C (follicular/male default)
        anabolic_drive     = _w(0.5,    0.04),     # 0.5 +/- 0.2 (moderate default)
    )


def update_hub(
    hub:   HubState,
    field: str,
    msg:   GaussianMsg,
) -> HubState:
    """
    Return a new HubState with one field replaced by `msg`.

    Preferred over direct _replace() because it validates field names.

    Parameters
    ----------
    hub   : current HubState
    field : field name (must be a valid HubState attribute)
    msg   : new GaussianMsg for that field

    Returns
    -------
    New HubState with hub.<field> = msg.

    Raises
    ------
    AttributeError if field is not a valid HubState attribute.
    """
    if not hasattr(hub, field):
        raise AttributeError(
            f"HubState has no field '{field}'. "
            f"Valid fields: {list(HubState._fields)}"
        )
    return hub._replace(**{field: msg})
