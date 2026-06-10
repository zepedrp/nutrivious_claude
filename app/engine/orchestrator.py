"""
app/engine/orchestrator.py

Penta Orchestrator — Operator-Splitting Engine (dt = 1 min)

Fuses FIVE intra-session subsystems (Neuromuscular, Metabolic Glucose,
Neuroendocrine, Thermo-Renal, Cardiorespiratory) into a single coherent step:

    1. Thermodynamic mass conservation: Muscle Glycogen owned exclusively by NM.
    2. Cori Cycle: NM glycolysis → lactate → MG hepatic gluconeogenesis.
    3. Endocrine coupling: NM/MG drive → Cortisol/Epi from Neuroendocrine step.
    4. Thermal drive: power → Core_Temp + PV_drop from Thermo-Renal step.
    5. Cardio integrates T_core + catecholamines from hub → autonomic_tone.

Operator-Splitting Topology (dt = 1 min):
─────────────────────────────────────────
  Step 1 — NM Predict + Update
      NM ODE reads : hub_plasma_glucose(t-1), hub_plasma_lactate(t-1)
      NM publishes : hub_local_gly_norm(t)

  Step 2 — MG Predict + Update
      MG ODE reads : hub_local_gly_norm(t) [from NM, fresh]
                     hub_cho_absorption, hub_epinephrine, hub_cortisol, power
      MG publishes : hub_plasma_glucose(t), hub_plasma_lactate(t)

  Step 3 — Neuroendocrine Predict + Update (1-hour resolution via dt=1-min steps)
      Neuro ODE reads : training_stress = f(power_W), hub_plasma_glucose(t),
                        hub_IL6(t-1), hub_circadian_phase, hub_sleep_sws
      Neuro publishes : hub_cortisol(t), hub_epinephrine(t)

  Step 4 — Thermo-Renal Predict + Update
      TR ODE reads  : hub_power_watts, hub_fluid_intake, hub_sodium_intake
      TR publishes  : hub_core_temp(t), hub_pv_drop_pct(t)

  Step 5 — Cardio Predict + Update
      Cardio ODE reads : power_W, hub_core_temp(t) [from Step 4, fresh],
                         hub_pv_drop_pct(t) [from Step 4, fresh]
      Cardio publishes : hub_autonomic_tone(t)

Stale-error analysis (1-min dt):
    Steps 1-2: same as TrioOrchestrator (|ΔG| < 1 mg/dL). Acceptable.
    Step 3 (Neuro) reads MG glucose fresh (same-step) — no stale error on glucose.
    Step 3 reads hub_cortisol(t-1) for MG (Steps 1-2 use stale Neuro) — tau_Cort ~2h >> 1 min.
    Steps 4-5: TR and Cardio see fresh Neuro output.

Design invariants:
    Fail-Loud: every filter raises RuntimeError on NaN. Orchestrator propagates.
    Operator-splitting: each step sees latest hub from the preceding step.

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
from app.slices.neuroendocrine.filter import (
    NeuroFilterState,
    initial_filter_state as neuro_initial_filter_state,
    update_state as neuro_update_state,
    default_process_noise_Q as neuro_default_Q,
)
from app.slices.neuroendocrine.ode import (
    IDX_CORT as NEURO_IDX_CORT,
    IDX_EPI  as NEURO_IDX_EPI,
    NeuroParams, DEFAULT_NEURO_PARAMS,
    initial_state as neuro_initial_state,
    zero_control as neuro_zero_control,
)
from app.slices.neuroendocrine.observation import DEFAULT_OBS_PARAMS as NEURO_DEFAULT_OBS_PARAMS
from app.slices.thermo_renal.filter import (
    ThermoRenalStateFilter,
    TRTransitionParams,
    DEFAULT_TR_TRANSITION_PARAMS,
)
from app.slices.thermo_renal.ode import (
    X0_TR_DEFAULT,
    P0_TR_DEFAULT,
    IDX_CORE_TEMP as TR_IDX_CORE_TEMP,
    IDX_PLASMA_VOL as TR_IDX_PLASMA_VOL,
    DEFAULT_TR_PARAMS,
)

# Reference plasma volume for pv_drop_pct computation
_TR_PV_REF: float = DEFAULT_TR_PARAMS.PV_ref   # 4.2 L


# ── Penta state container ─────────────────────────────────────────────────────

class PentaState(NamedTuple):
    """
    Immutable snapshot of the five-slice state at a single timestep.

    Fields
    ------
    nm          : GaussianState — NM Tissue (6-state, minutes timescale)
    mg          : GaussianState — Metabolic Glucose (5-state, minutes timescale)
    neuro       : GaussianState — Neuroendocrine (9-state, hourly ODE; 1-min filter steps)
    thermo_renal: GaussianState — Thermo-Renal (5-state, minutes timescale)
    cardio      : GaussianState — Cardiorespiratory (8-state, minutes timescale)
    hub         : HubState      — All inter-slice Gaussian messages
    """
    nm:           GaussianState
    mg:           GaussianState
    neuro:        GaussianState
    thermo_renal: GaussianState
    cardio:       GaussianState
    hub:          HubState


# ── Orchestrator ──────────────────────────────────────────────────────────────

class PentaOrchestrator:
    """
    Operator-splitting orchestrator for NM + MG + Neuroendocrine + ThermoRenal + Cardio.

    Advances five subsystems by dt=1 minute in strict topological order:
        NM -> MG -> Neuroendocrine -> ThermoRenal -> Cardio

    The resulting hub after each step is threaded to the next step, so
    each subsystem reads the freshest available estimate from its predecessors.

    Typical usage
    ─────────────
    orch  = PentaOrchestrator()
    state = PentaOrchestrator.default_state()

    # Per 1-minute wearable sample:
    state = orch.step(
        prior    = state,
        controls = {"power_W": 200.0, "hub_circadian_phase": 0.0,
                    "hub_sleep_sws": 0.0, "hub_fluid_intake_L_min": 0.008,
                    "hub_sodium_intake_mmol_min": 0.4},
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
        nm_filter:     NMv4Filter                  | None = None,
        mg_filter:     MetabolicGlucoseFilter       | None = None,
        neuro_params:  NeuroParams                  | None = None,
        tr_filter:     ThermoRenalStateFilter       | None = None,
        cardio_filter: CardioStateFilter            | None = None,
        cardio_params: CardioTransitionParams       | None = None,
    ) -> None:
        self.nm_filt      = nm_filter     or NMv4Filter()
        self.mg_filt      = mg_filter     or MetabolicGlucoseFilter()
        self.neuro_params = neuro_params  or DEFAULT_NEURO_PARAMS
        self.neuro_Q      = neuro_default_Q()
        self.tr_filt      = tr_filter     or ThermoRenalStateFilter()
        self.cardio_flt   = cardio_filter or CardioStateFilter()
        self.cardio_prms  = cardio_params or DEFAULT_TRANSITION_PARAMS

    @staticmethod
    def default_state() -> PentaState:
        """Population-prior cold-start state for all five subsystems."""
        neuro_fs = neuro_initial_filter_state()
        return PentaState(
            nm           = GaussianState(mean=X0_NM_V4,         cov=P0_NM_V4),
            mg           = GaussianState(mean=X0_MG,             cov=P0_MG),
            neuro        = GaussianState(mean=neuro_fs.mean,     cov=neuro_fs.cov),
            thermo_renal = GaussianState(mean=X0_TR_DEFAULT,     cov=P0_TR_DEFAULT),
            cardio       = GaussianState(mean=X0_CARDIO_DEFAULT, cov=P0_CARDIO_DEFAULT),
            hub          = default_hub_state(),
        )

    def step(
        self,
        prior:         PentaState,
        controls:      dict[str, float],
        obs_nm:        dict[str, object] | None = None,
        obs_mg:        dict[str, object] | None = None,
        obs_neuro:     dict[str, object] | None = None,
        obs_thermo:    dict[str, float]  | None = None,
        obs_cardio:    dict[str, float]  | None = None,
        cardio_params: CardioTransitionParams | None = None,
        dt_real:       float = 1.0,
    ) -> PentaState:
        """
        Advance the penta-system by dt_real minutes using operator splitting.

        Parameters
        ----------
        prior       : PentaState from the previous timestep
        controls    : dict with:
            "power_W"                   float [W]      exercise power
            "cho_abs_g_min"             float [g/min]  gut CHO absorption
            "hub_circadian_phase"       float [0-1]    awakening drive (CAR)
            "hub_sleep_sws"             float [0-1]    SWS fraction (for GH pulse)
            "hub_fluid_intake_L_min"    float [L/min]  fluid intake
            "hub_sodium_intake_mmol_min" float [mmol/min] sodium intake
        obs_nm      : {"emg_mV", "smo2_pct", "quality_flags"} — defaults: NaN (predict-only)
        obs_mg      : {"cgm_reading", "quality_flag"}          — defaults: NaN (predict-only)
        obs_neuro   : {"quality_flag"} int [0-4]               — defaults: 4 (predict-only)
        obs_thermo  : {"core_temp_obs", "bw_drop_obs",
                       "quality_flags"}                         — defaults: NaN (predict-only)
        obs_cardio  : {"HR_obs_bpm", "VO2_obs", "RMSSD_obs_ms"} — defaults: NaN (predict-only)
        cardio_params : override CardioTransitionParams
        dt_real     : actual integration window [min]; default 1.0

        Returns
        -------
        PentaState with updated (nm, mg, neuro, thermo_renal, cardio, hub)

        Raises
        ------
        RuntimeError if any filter produces NaN posterior (Fail-Loud).
        """
        if obs_nm     is None: obs_nm     = {}
        if obs_mg     is None: obs_mg     = {}
        if obs_neuro  is None: obs_neuro  = {}
        if obs_thermo is None: obs_thermo = {}
        if obs_cardio is None: obs_cardio = {}
        c_params = cardio_params or self.cardio_prms

        hub = prior.hub

        power_W        = float(controls.get("power_W",                   0.0))
        cho_abs        = float(controls.get("cho_abs_g_min",              0.0))
        hub_circ_phase = float(controls.get("hub_circadian_phase",        0.0))
        hub_sws        = float(controls.get("hub_sleep_sws",              0.0))
        hub_fluid      = float(controls.get("hub_fluid_intake_L_min",     0.0))
        hub_na_in      = float(controls.get("hub_sodium_intake_mmol_min", 0.0))

        # Read stale-by-1-step cortisol and epi from hub (for MG step below)
        epi_for_mg     = float(msg_to_mean(hub.epinephrine))
        cortisol_for_mg = float(msg_to_mean(hub.cortisol))

        # ── Step 1: NM Predict + Update ───────────────────────────────────────
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
        local_gly_norm = float(msg_to_mean(hub.local_gly_norm))

        mg_hubs = jnp.array([
            cho_abs,
            epi_for_mg,
            cortisol_for_mg,
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

        # ── Step 3: Neuroendocrine Predict + Update ───────────────────────────
        # Training stress = normalized power [0, 1]; MG glucose is fresh (Step 2).
        training_stress = float(jnp.clip(
            jnp.float32(power_W) / jnp.float32(400.0), 0.0, 1.0
        ))
        hub_IL6_pgmL = float(msg_to_mean(hub.il6))
        hub_glucose  = float(msg_to_mean(hub.plasma_glucose))

        neuro_ctrl = lambda t: jnp.array([    # noqa: E731
            training_stress,
            hub_circ_phase,
            hub_IL6_pgmL,
            hub_glucose,
            hub_sws,
        ], dtype=jnp.float32)

        # Predict-only y_obs using prior mean (quality_flag=4 → R×1e8)
        neuro_state_in = NeuroFilterState(
            mean=prior.neuro.mean,
            cov=prior.neuro.cov,
        )
        neuro_qflag = int(obs_neuro.get("quality_flag", 4))
        # Build predict-only observation from prior state (no real sensor data)
        from app.slices.neuroendocrine.observation import h_neuro as _h_neuro
        y_neuro = _h_neuro(prior.neuro.mean, obs_params=NEURO_DEFAULT_OBS_PARAMS)
        # Replace with real obs if provided
        if "y_obs" in obs_neuro:
            y_neuro = jnp.array(obs_neuro["y_obs"], dtype=jnp.float32)

        neuro_result = neuro_update_state(
            state        = neuro_state_in,
            y_obs        = y_neuro,
            params       = self.neuro_params,
            obs_params   = NEURO_DEFAULT_OBS_PARAMS,
            Q            = self.neuro_Q,
            quality_flag = neuro_qflag,
            control_fn   = neuro_ctrl,
            t0           = float(controls.get("t_hour", 8.0)),
        )
        neuro_state = GaussianState(mean=neuro_result.mean, cov=neuro_result.cov)

        # Neuroendocrine publishes cortisol and epinephrine to hub
        cort_mean = float(neuro_state.mean[NEURO_IDX_CORT])
        cort_var  = float(neuro_state.cov[NEURO_IDX_CORT, NEURO_IDX_CORT])
        hub = update_hub(
            hub, "cortisol",
            msg_from_scalar(cort_mean, cort_var + 1e-8),
        )
        epi_mean = float(neuro_state.mean[NEURO_IDX_EPI])
        epi_var  = float(neuro_state.cov[NEURO_IDX_EPI, NEURO_IDX_EPI])
        hub = update_hub(
            hub, "epinephrine",
            msg_from_scalar(epi_mean, epi_var + 1e-8),
        )

        # ── Step 4: Thermo-Renal Predict + Update ─────────────────────────────
        tr_controls = {
            "hub_power_watts":           power_W,
            "hub_fluid_intake_L_min":    hub_fluid,
            "hub_sodium_intake_mmol_min": hub_na_in,
        }
        tr_state = self.tr_filt.update_state(
            prior         = prior.thermo_renal,
            core_temp_obs = float(obs_thermo.get("core_temp_obs", math.nan)),
            bw_drop_obs   = float(obs_thermo.get("bw_drop_obs",   math.nan)),
            controls      = tr_controls,
            dt_minutes    = dt_real,
            quality_flags = obs_thermo.get("quality_flags", (4, 4)),
        )

        # ThermoRenal publishes core_temp and pv_drop_pct to hub
        t_core_mean = float(tr_state.mean[TR_IDX_CORE_TEMP])
        t_core_var  = float(tr_state.cov[TR_IDX_CORE_TEMP, TR_IDX_CORE_TEMP])
        hub = update_hub(
            hub, "core_temp",
            msg_from_scalar(t_core_mean, t_core_var + 1e-8),
        )
        pv_mean    = float(tr_state.mean[TR_IDX_PLASMA_VOL])
        pv_var     = float(tr_state.cov[TR_IDX_PLASMA_VOL, TR_IDX_PLASMA_VOL])
        pv_drop    = float(jnp.clip(
            jnp.float32(100.0) * (jnp.float32(_TR_PV_REF) - jnp.float32(pv_mean)) / jnp.float32(_TR_PV_REF),
            jnp.float32(0.0),
            jnp.float32(50.0),
        ))
        # Variance of pv_drop_pct = (100/PV_ref)^2 * pv_var
        pv_drop_var = (100.0 / _TR_PV_REF) ** 2 * pv_var
        hub = update_hub(
            hub, "pv_drop_pct",
            msg_from_scalar(pv_drop, pv_drop_var + 1e-8),
        )

        # ── Step 5: Cardio Predict + Update ───────────────────────────────────
        # Cardio reads T_core and pv_drop from hub (fresh from Step 4)
        cardio_state = self.cardio_flt.update_state(
            prior        = prior.cardio,
            observations = obs_cardio,
            controls     = {
                "power_watts":     power_W,
                "hub_T_core":      float(msg_to_mean(hub.core_temp)),
                "hub_pv_drop_pct": float(msg_to_mean(hub.pv_drop_pct)),
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

        return PentaState(
            nm           = nm_state,
            mg           = mg_state,
            neuro        = neuro_state,
            thermo_renal = tr_state,
            cardio       = cardio_state,
            hub          = hub,
        )


# ── Legacy alias ──────────────────────────────────────────────────────────────

# Keep TrioOrchestrator / TrioState available for backward-compat with existing tests.

class TrioState(NamedTuple):
    """Legacy 3-slice state (NM, MG, Cardio). Use PentaState for new code."""
    nm:     GaussianState
    mg:     GaussianState
    cardio: GaussianState
    hub:    HubState


class TrioOrchestrator:
    """
    Legacy Trio orchestrator (NM + MG + Cardio only).
    Kept for backward compatibility with existing tests.
    New code should use PentaOrchestrator.
    """

    def __init__(
        self,
        nm_filter:     NMv4Filter               | None = None,
        mg_filter:     MetabolicGlucoseFilter    | None = None,
        cardio_filter: CardioStateFilter         | None = None,
        cardio_params: CardioTransitionParams    | None = None,
    ) -> None:
        self.nm_filt     = nm_filter     or NMv4Filter()
        self.mg_filt     = mg_filter     or MetabolicGlucoseFilter()
        self.cardio_flt  = cardio_filter or CardioStateFilter()
        self.cardio_prms = cardio_params or DEFAULT_TRANSITION_PARAMS

    @staticmethod
    def default_state() -> TrioState:
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

        # Step 1: NM
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
        gly_mean = float(nm_state.mean[NM_IDX_GLYCOGEN])
        gly_var  = float(nm_state.cov[NM_IDX_GLYCOGEN, NM_IDX_GLYCOGEN])
        hub = update_hub(hub, "local_gly_norm", nm_glycogen_to_hub(gly_mean, gly_var))

        # Step 2: MG
        local_gly_norm = float(msg_to_mean(hub.local_gly_norm))
        mg_hubs = jnp.array([cho_abs, epi, cortisol, power_W, local_gly_norm], dtype=jnp.float32)
        mg_state = self.mg_filt.update_state(
            prior        = prior.mg,
            cgm_reading  = float(obs_mg.get("cgm_reading", math.nan)),
            hubs         = mg_hubs,
            quality_flag = int(obs_mg.get("quality_flag", 4)),
        )
        hub = update_hub(
            hub, "plasma_glucose",
            msg_from_scalar(float(mg_state.mean[MG_IDX_G]),   float(mg_state.cov[MG_IDX_G,   MG_IDX_G])),
        )
        hub = update_hub(
            hub, "plasma_lactate",
            msg_from_scalar(float(mg_state.mean[MG_IDX_LAC]), float(mg_state.cov[MG_IDX_LAC, MG_IDX_LAC])),
        )

        # Step 3: Cardio
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
        at_mean = float(cardio_state.mean[CARDIO_IDX_AT])
        at_var  = float(cardio_state.cov[CARDIO_IDX_AT, CARDIO_IDX_AT])
        hub = update_hub(hub, "autonomic_tone", msg_from_scalar(at_mean, at_var + 1e-8))

        return TrioState(nm=nm_state, mg=mg_state, cardio=cardio_state, hub=hub)
