"""
app/slices/hematological/__init__.py  Hematological Slice V3.0

5-state hourly ODE: RBC homeostasis, plasma expansion, hepcidin-iron axis.
States: [RBC_Mass_g, Plasma_Vol_L, EPO_mIU_mL, Ferritin_ug_L, Hemolysis_Tox_au]
"""
from app.slices.hematological.ode import (
    HematologicalParams,
    DEFAULT_HEM_PARAMS,
    X0_HEM_DEFAULT,
    P0_HEM_DEFAULT,
    STATE_DIM,
    OBS_DIM,
    hematological_v3_ode,
    build_hem_params,
    integrate_1h,
    compute_hematocrit,
    compute_o2_capacity,
)
from app.slices.hematological.observation import (
    HemObsParams,
    DEFAULT_HEM_OBS_PARAMS,
    h_hem,
    inflate_R_hem,
    observation_noise_R,
)
from app.slices.hematological.envelope import (
    HematologicalEnvelope,
    check_hematological_envelope,
)
from app.slices.hematological.filter import (
    HemFilterState,
    initial_filter_state,
    update_state,
    filter_history,
    Q_DEFAULT,
)

__all__ = [
    "HematologicalParams", "DEFAULT_HEM_PARAMS",
    "X0_HEM_DEFAULT", "P0_HEM_DEFAULT",
    "STATE_DIM", "OBS_DIM",
    "hematological_v3_ode", "build_hem_params",
    "integrate_1h", "compute_hematocrit", "compute_o2_capacity",
    "HemObsParams", "DEFAULT_HEM_OBS_PARAMS",
    "h_hem", "inflate_R_hem", "observation_noise_R",
    "HematologicalEnvelope", "check_hematological_envelope",
    "HemFilterState", "initial_filter_state",
    "update_state", "filter_history", "Q_DEFAULT",
]
