"""
app/slices/neuromuscular_tissue/__init__.py

Neuromuscular Tissue Slice V4.0 -- L2-L4 Components (minutes timescale)

Modules
-------
ode         -- 6-state intra-session ODE (Henneman + SERCA + ATP + LFF/RyR1 + Glycogen)
observation -- h(x, theta) -> [EMG_Amplitude_mV, SmO2_pct]
envelope    -- Phase3Envelope: Fatigue + Glycogen constraints (Path A)
nlme        -- L3 NLME in NumPyro (D=2: P_th_2, k_SERCA; Matt trick)
filter      -- L4 UKF (6-state, alpha=0.10, dt=1 min, 13 sigma points)
"""
from app.slices.neuromuscular_tissue.ode import (
    NMv4Params,
    DEFAULT_V4_PARAMS,
    X0_NM_V4,
    P0_NM_V4,
    STATE_DIM,
    OBS_DIM,
    CTRL_DIM,
    IDX_ATP, IDX_CA, IDX_R1, IDX_R2, IDX_RYR1, IDX_GLYCOGEN,
    nm_v4_ode,
    hub_peripheral_fatigue,
    hub_muscle_glycogen,
)
from app.slices.neuromuscular_tissue.observation import (
    NMv4ObsParams,
    DEFAULT_V4_OBS_PARAMS,
    R_NM_V4_DEFAULT,
    h_nm_v4,
    h_nm_v4_sigma,
    inflate_R_nm_v4,
)
from app.slices.neuromuscular_tissue.envelope import build_neuromuscular_v4_envelope
from app.slices.neuromuscular_tissue.nlme import NMv4NLME
from app.slices.neuromuscular_tissue.filter import NMv4Filter

__all__ = [
    "NMv4Params",
    "DEFAULT_V4_PARAMS",
    "X0_NM_V4",
    "P0_NM_V4",
    "STATE_DIM",
    "OBS_DIM",
    "CTRL_DIM",
    "IDX_ATP", "IDX_CA", "IDX_R1", "IDX_R2", "IDX_RYR1", "IDX_GLYCOGEN",
    "nm_v4_ode",
    "hub_peripheral_fatigue",
    "hub_muscle_glycogen",
    "NMv4ObsParams",
    "DEFAULT_V4_OBS_PARAMS",
    "R_NM_V4_DEFAULT",
    "h_nm_v4",
    "h_nm_v4_sigma",
    "inflate_R_nm_v4",
    "build_neuromuscular_v4_envelope",
    "NMv4NLME",
    "NMv4Filter",
]
