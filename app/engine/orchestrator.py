"""
app/engine/orchestrator.py

Trio Orchestrator -- Operator-Splitting Engine (dt = 1 min)

Fuses the three intra-session subsystems (Neuromuscular, Metabolic Glucose,
Cardiorespiratory) into a single coherent step, enforcing:

    1. Thermodynamic mass conservation: Muscle Glycogen is owned exclusively by
       the NM slice. MG reads hub_local_gly_norm (not its own internal state).

    2. Cori Cycle: NM glycolysis produces lactate (scaled by hub_local_gly_norm
       in the MG ODE), which feeds MG's hepatic gluconeogenesis term (Ra_cori).

    3. Probabilistic hub coupling: hub variables are Gaussian messages
       (mean, variance) propagated via information-form addition (hubs.py).

Operator-Splitting Topology (dt = 1 min):
─────────────────────────────────────────
  Step 1 — NM Predict + Update
      NM ODE reads: hub_plasma_glucose(t-1), hub_plasma_lactate(t-1) from hub
      NM UKF update: assimilates EMG_mV and SmO2_pct
      NM publishes: hub_local_gly_norm(t) [Gaussian msg]

  Step 2 — MG Predict + Update
      MG ODE reads: hub_local_gly_norm(t) [from NM, just published]
                    + hub_cho_absorption, hub_Epi, hub_Cort, hub_power
      MG UKF update: assimilates CGM reading
      MG publishes: hub_plasma_glucose(t), hub_plasma_lactate(t)

  Step 3 — Cardio Predict + Update
      Cardio ODE reads: hub_T_core, hub_pv_drop_pct, power_watts
      Cardio UKF update: assimilates HR_obs, VO2_obs, RMSSD_obs
      Cardio publishes: hub_autonomic_tone(t)

Stale-error analysis (1-min dt):
    NM reads plasma_glucose(t-1): stale by 1 min.
    tau_G in plasma >> 1 min -> |error| = dG/dt * dt < 1 mg/dL. Acceptable.
    Tight coupling available via sub-minute stepping if needed.

Design invariants:
    Fail-Loud: every filter raises RuntimeError on NaN. The orchestrator
    propagates these -- no silent substitution.
    Operator-splitting: each step sees the latest hub from the preceding step,
    not the stale t-1 value for within-step predecessors.

References:
    Koller D. & Friedman N. (2009) Probabilistic Graphical Models, MIT Press.
    Strang G. (1968) SIAM J Numer Anal 5:506-517  [operator splitting]
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

from app.engine.hubs import (
    HubState,
    GaussianMsg,
    msg_from_scalar,
    msg_to_mean,
    nm_glycogen_to_hub,
    update_hub,
    default_hub_state,
)
from app.engine.assimilation.ukf_filter import GaussianState
from app.slices.neuromuscular_tissue.filter import NMv4Filter
from app.slices.neuromuscular_tissue.ode import (
    X0_NM_V4,
    P0_NM_V4,
    IDX_GLYCOGEN as NM_IDX_GLYCOGEN,
)
from app.slices.metabolic_glucose.filter import MetabolicGlucoseFilter
from app.slices.metabolic_glucose.ode import (
    X0_DEFAULT as X0_MG,
    P0_DEFAULT as P0_MG,
    IDX_G      as MG_IDX_G,
    IDX_LAC    as MG_IDX_LAC,
    HUB_DIM    as MG_HUB_DIM,
)
from app.slices.cardiorespiratory.filter import (
    CardioStateFilter,
    CardioTransitionParams,
    DEFAULT_TRANSITION_PARAMS,
)
from app.slices.cardiorespiratory.ode import (
    X0_CARDIO_DEFAULT,
    P0_CARDIO_DEFAULT,
    IDX_AT as CARDIO_IDX_AT,
)


# ── Trio state container ──────────────────────────────────────────────────────

class TrioState(NamedTuple):
    """
    Immutable snapshot of the three-slice state at a single timestep.

    Fields
    ------
    nm     : GaussianState -- NM Tissue (6-state, minutes timescale)
    mg     : GaussianState -- Metabolic Glucose (5-state, minutes timescale)
    cardio : GaussianState -- Cardiorespiratory (8-state, minutes timescale)
    hub    : HubState      -- All inter-slice Gaussian messages
    """
    nm:     GaussianState
    mg:     GaussianState
    cardio: GaussianState
    hub:    HubState


# ── Orchestrator ──────────────────────────────────────────────────────────────

class TrioOrchestrator:
    """
    Operator-splitting orchestrator for the NM + MG + Cardio trio.

    Advances the three subsystems by dt=1 minute in strict topological order:
    NM -> MG -> Cardio, with Gaussian hub messages connecting them.

    Typical usage
    ─────────────
    orch  = TrioOrchestrator()
    state = TrioOrchestrator.default_state()

    # Per 1-minute wearable sample:
    state = orch.step(
        prior    = state,
        controls = {"power_W": 200.0, "hub_T_core": 37.5, ...},
        obs_nm   = {"emg_mV": 1.2, "smo2_pct": 68.0},
        obs_mg   = {"cgm_reading": 85.0},
        obs_cardio = {"HR_obs_bpm": 145.0},
    )

    Fail-Loud contract
    ──────────────────
    Any filter NaN -> RuntimeError propagated unmodified (no silent fallback).
    """

    def __init__(
        self,
        nm_filter:     NMv4Filter     | None = None,
        mg_filter:     MetabolicGlucoseFilter | None = None,
        cardio_filter: CardioStateFilter | None = None,
        cardio_params: CardioTransitionParams | None = None,
    ) -> None:
        self.nm_filt     = nm_filter     or NMv4Filter()
        self.mg_filt     = mg_filter     or MetabolicGlucoseFilter()
        self.cardio_flt  = cardio_filter or CardioStateFilter()
        self.cardio_prms = cardio_params or DEFAULT_TRANSITION_PARAMS

    @staticmethod
    def default_state() -> TrioState:
        """Population-prior cold-start state for all three subsystems."""
        return TrioState(
            nm     = GaussianState(mean=X0_NM_V4,          cov=P0_NM_V4),
            mg     = GaussianState(mean=X0_MG,              cov=P0_MG),
            cardio = GaussianState(mean=X0_CARDIO_DEFAULT,  cov=P0_CARDIO_DEFAULT),
            hub    = default_hub_state(),
        )

    def step(
        self,
        prior:        TrioState,
        controls:     dict[str, float],
        obs_nm:       dict[str, object]  | None = None,
        obs_mg:       dict[str, object]  | None = None,
        obs_cardio:   dict[str, float]   | None = None,
        cardio_params: CardioTransitionParams | None = None,
        dt_real:      float = 1.0,
    ) -> TrioState:
        """
        Advance the trio by dt_real minutes using operator splitting.

        Parameters
        ----------
        prior       : TrioState from the previous timestep
        controls    : dict with:
                        "power_W"          float [W]      exercise power
                        "hub_T_core"       float [deg C]  (optional, NaN ok)
                        "hub_pv_drop_pct"  float [%]      (optional, NaN ok)
                        "cho_abs_g_min"    float [g/min]  gut CHO absorption
                        "epi_pgmL"         float [pg/mL]  epinephrine
                        "cortisol_nmolL"   float [nmol/L] cortisol
        obs_nm      : dict with "emg_mV", "smo2_pct", "quality_flags"
                      Defaults: all NaN (predict-only for NM)
        obs_mg      : dict with "cgm_reading", "quality_flag"
                      Defaults: NaN cgm (predict-only for MG)
        obs_cardio  : dict with "HR_obs_bpm", "VO2_obs", "RMSSD_obs_ms"
                      Defaults: all NaN (predict-only for Cardio)
        cardio_params : override CardioTransitionParams for this step
        dt_real     : actual integration window [min]; default 1.0

        Returns
        -------
        TrioState with updated (nm, mg, cardio, hub)

        Raises
        ------
        RuntimeError if any filter produces NaN posterior (Fail-Loud).
        """
        if obs_nm    is None: obs_nm    = {}
        if obs_mg    is None: obs_mg    = {}
        if obs_cardio is None: obs_cardio = {}
        c_params = cardio_params or self.cardio_prms

        hub = prior.hub

        power_W         = float(controls.get("power_W",          0.0))
        hub_T_core      = float(controls.get("hub_T_core",       math.nan))
        hub_pv_drop_pct = float(controls.get("hub_pv_drop_pct",  0.0))
        cho_abs         = float(controls.get("cho_abs_g_min",    0.0))
        epi             = float(controls.get("epi_pgmL",         50.0))
        cortisol        = float(controls.get("cortisol_nmolL",   300.0))

        # ── Step 1: NM Predict + Update ───────────────────────────────────────
        # NM reads plasma_glucose and plasma_lactate from hub (stale t-1).
        nm_u = jnp.array([
            power_W,
            float(msg_to_mean(hub.plasma_lactate)),
            float(msg_to_mean(hub.plasma_glucose)),
        ], dtype=jnp.float32)

        nm_state = self.nm_filt.update_state(
            prior         = prior.nm,
            emg_mV        = float(obs_nm.get("emg_mV",   math.nan)),
            smo2_pct      = float(obs_nm.get("smo2_pct", math.nan)),
            u             = nm_u,
            quality_flags = obs_nm.get("quality_flags", (4, 4)),
        )

        # NM publishes hub_local_gly_norm (Gaussian message)
        gly_mean = float(nm_state.mean[NM_IDX_GLYCOGEN])
        gly_var  = float(nm_state.cov[NM_IDX_GLYCOGEN, NM_IDX_GLYCOGEN])
        hub = update_hub(hub, "local_gly_norm", nm_glycogen_to_hub(gly_mean, gly_var))

        # ── Step 2: MG Predict + Update ───────────────────────────────────────
        # MG reads hub_local_gly_norm (fresh from NM Step 1)
        local_gly_norm = float(msg_to_mean(hub.local_gly_norm))

        mg_hubs = jnp.array([
            cho_abs,
            epi,
            cortisol,
            power_W,
            local_gly_norm,
        ], dtype=jnp.float32)

        mg_state = self.mg_filt.update_state(
            prior        = prior.mg,
            cgm_reading  = float(obs_mg.get("cgm_reading", math.nan)),
            hubs         = mg_hubs,
            quality_flag = int(obs_mg.get("quality_flag", 4)),
        )

        # MG publishes plasma_glucose and plasma_lactate
        hub = update_hub(
            hub, "plasma_glucose",
            msg_from_scalar(
                float(mg_state.mean[MG_IDX_G]),
                float(mg_state.cov[MG_IDX_G, MG_IDX_G]),
            ),
        )
        hub = update_hub(
            hub, "plasma_lactate",
            msg_from_scalar(
                float(mg_state.mean[MG_IDX_LAC]),
                float(mg_state.cov[MG_IDX_LAC, MG_IDX_LAC]),
            ),
        )

        # ── Step 3: Cardio Predict + Update ───────────────────────────────────
        cardio_state = self.cardio_flt.update_state(
            prior        = prior.cardio,
            observations = obs_cardio,
            controls     = {
                "power_watts":     power_W,
                "hub_T_core":      hub_T_core,
                "hub_pv_drop_pct": hub_pv_drop_pct,
            },
            params   = c_params,
            dt_real  = dt_real,
        )

        # Cardio publishes autonomic_tone
        at_mean = float(cardio_state.mean[CARDIO_IDX_AT])
        at_var  = float(cardio_state.cov[CARDIO_IDX_AT, CARDIO_IDX_AT])
        hub = update_hub(
            hub, "autonomic_tone",
            msg_from_scalar(at_mean, at_var + 1e-8),
        )

        return TrioState(nm=nm_state, mg=mg_state, cardio=cardio_state, hub=hub)
