"""
app/slices/cardiorespiratory/__init__.py

Cardiorespiratory L2-L4 Digital Twin Slice.

Public API
──────────
  ode        : cardiorespiratory_slice_ode, DEFAULT_CARDIO_SLICE_PARAMS,
               X0_CARDIO_DEFAULT, P0_CARDIO_DEFAULT, IDX_*
  observation: h_cardio, CardioObsParams, DEFAULT_OBS_PARAMS, R_DEFAULT
  envelope   : build_cardiorespiratory_envelope, check_hard_constraints,
               check_all_constraints
  nlme       : CardioNLME, simulate_hr_trajectory
  filter     : CardioStateFilter, CardioTransitionParams, DEFAULT_TRANSITION_PARAMS

Supersedes: app/slices/aerobic_pipeline.py (deprecated — see that module).
"""
from app.slices.cardiorespiratory.ode import (
    cardiorespiratory_slice_ode,
    DEFAULT_CARDIO_SLICE_PARAMS,
    CardioSliceParams,
    X0_CARDIO_DEFAULT,
    P0_CARDIO_DEFAULT,
    IDX_VO2, IDX_HR, IDX_SV,
    IDX_WFAST, IDX_WSLOW,
    IDX_RF, IDX_AT, IDX_RMSSD7D,
    STATE_DIM, OBS_DIM,
)
from app.slices.cardiorespiratory.observation import (
    h_cardio,
    h_cardio_sigma,
    CardioObsParams,
    DEFAULT_OBS_PARAMS,
    R_DEFAULT,
)
from app.slices.cardiorespiratory.envelope import (
    build_cardiorespiratory_envelope,
    check_hard_constraints,
    check_all_constraints,
)
from app.slices.cardiorespiratory.filter import (
    CardioStateFilter,
    CardioTransitionParams,
    DEFAULT_TRANSITION_PARAMS,
)

__all__ = [
    "cardiorespiratory_slice_ode",
    "DEFAULT_CARDIO_SLICE_PARAMS",
    "CardioSliceParams",
    "X0_CARDIO_DEFAULT",
    "P0_CARDIO_DEFAULT",
    "IDX_VO2", "IDX_HR", "IDX_SV",
    "IDX_WFAST", "IDX_WSLOW",
    "IDX_RF", "IDX_AT", "IDX_RMSSD7D",
    "STATE_DIM", "OBS_DIM",
    "h_cardio",
    "h_cardio_sigma",
    "CardioObsParams",
    "DEFAULT_OBS_PARAMS",
    "R_DEFAULT",
    "build_cardiorespiratory_envelope",
    "check_hard_constraints",
    "check_all_constraints",
    "CardioStateFilter",
    "CardioTransitionParams",
    "DEFAULT_TRANSITION_PARAMS",
]
