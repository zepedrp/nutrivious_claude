"""
tests/test_metabolic_glucose_slice.py -- Metabolic Glucose V2.0 Gate Zero

T1  test_epinephrine_glucose_spike
    hub_Epinephrine = 2000 pg/mL (fight-or-flight, no CHO).
    Assertion: Plasma_Glucose rises >= 5 mg/dL vs rest control at t=30 min
    (hepatic glycogenolysis via Michaelis-Menten Epi stimulation).

T2  test_cortisol_insulin_resistance
    hub_Cortisol chronic = 700 nmol/L + constant CHO 0.5 g/min
    vs control at 300 nmol/L + same CHO.
    Assertion: Plasma_Glucose at t=30 min >= 10 mg/dL higher in stress scenario
    (cortisol exponentially suppresses IS_eff -> impaired glucose clearance).

T3  test_muscle_glycogen_depletion
    hub_power_watts = 400 W for 30 min.
    Assertions: (a) Muscle_Glycogen drops > 15% of initial (substantial depletion)
                (b) Lactate rises above 2.0 mmol/L (anaerobic threshold exceeded).

T4  test_ukf_cgm_assimilation
    60 steps of 1-min CGM assimilation (90 +/- 5 mg/dL synthetic trace).
    Assertions: (a) posterior_mean NaN-free after 60 updates
                (b) covariance diagonal all positive (PSD maintained).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import diffrax

from app.slices.metabolic_glucose.ode import (
    metabolic_glucose_ode,
    DEFAULT_PARAMS,
    X0_DEFAULT,
    HUBS_DEFAULT,
    IDX_G,
    IDX_MG,
    IDX_LAC,
    HUB_CHO,
    HUB_EPI,
    HUB_CORT,
    HUB_POW,
)
from app.slices.metabolic_glucose.filter import MetabolicGlucoseFilter, P0_DEFAULT
from app.engine.assimilation.ukf_filter import GaussianState


def _run_ode(
    x0:     jax.Array,
    hubs:   jax.Array,
    dt_min: float = 30.0,
    params=DEFAULT_PARAMS,
) -> jax.Array:
    """Integrate the 6-state glucose ODE for dt_min minutes; return final state."""
    sol = diffrax.diffeqsolve(
        terms    = diffrax.ODETerm(metabolic_glucose_ode),
        solver   = diffrax.Tsit5(),
        t0       = jnp.float32(0.0),
        t1       = jnp.float32(dt_min),
        dt0      = jnp.float32(0.5),
        y0       = x0,
        args     = (params, hubs),
        saveat   = diffrax.SaveAt(t1=True),
        max_steps= 512,
    )
    return sol.ys[0]


# -- T1: Epinephrine spike (fight-or-flight -> hepatic glycogenolysis)
def test_epinephrine_glucose_spike():
    """
    Inject hub_Epinephrine = 2000 pg/mL (40x resting baseline).
    Assertion: Plasma_Glucose >= 5 mg/dL above no-Epi control at t=30 min.
    """
    x_ctrl = _run_ode(X0_DEFAULT, HUBS_DEFAULT, dt_min=30.0)

    hubs_epi = HUBS_DEFAULT.at[HUB_EPI].set(jnp.float32(2000.0))
    x_epi    = _run_ode(X0_DEFAULT, hubs_epi, dt_min=30.0)

    G_ctrl = float(x_ctrl[IDX_G])
    G_epi  = float(x_epi[IDX_G])

    print(f"\nT1: G_ctrl={G_ctrl:.1f} mg/dL  G_epi={G_epi:.1f} mg/dL  delta={G_epi - G_ctrl:.1f}")
    assert G_epi > G_ctrl + 5.0, (
        f"Epinephrine should trigger hepatic glucose release: "
        f"G_epi={G_epi:.1f} vs G_ctrl={G_ctrl:.1f} (expected delta >= 5 mg/dL)"
    )


# -- T2: Cortisol-induced insulin resistance
def test_cortisol_insulin_resistance():
    """
    Chronic hub_Cortisol = 700 nmol/L + CHO 0.5 g/min.
    IS_eff = IS_0 * exp(-k_IS_Cort * 400) -> ~80% suppression.
    Assertion: Plasma_Glucose >= 10 mg/dL higher than basal-cortisol control.
    """
    hubs_cho_base = HUBS_DEFAULT.at[HUB_CHO].set(jnp.float32(0.5))

    x_normal  = _run_ode(X0_DEFAULT, hubs_cho_base, dt_min=30.0)

    hubs_hicort = hubs_cho_base.at[HUB_CORT].set(jnp.float32(700.0))
    x_hicort    = _run_ode(X0_DEFAULT, hubs_hicort, dt_min=30.0)

    G_normal = float(x_normal[IDX_G])
    G_hicort = float(x_hicort[IDX_G])

    print(f"\nT2: G_normal={G_normal:.1f} mg/dL  G_hicort={G_hicort:.1f} mg/dL  delta={G_hicort - G_normal:.1f}")
    assert G_hicort > G_normal + 10.0, (
        f"Chronic cortisol should impair glucose clearance: "
        f"G_hicort={G_hicort:.1f} vs G_normal={G_normal:.1f} (expected delta >= 10 mg/dL)"
    )


# -- T3: Muscle glycogen depletion + lactate rise
def test_muscle_glycogen_depletion():
    """
    hub_power_watts = 400 W for 30 min (hard VO2max effort).
    Assertions:
        (a) Muscle_Glycogen drops > 15% of initial (strong depletion).
        (b) Lactate rises above 2.0 mmol/L (anaerobic flux).
    """
    hubs_ex = HUBS_DEFAULT.at[HUB_POW].set(jnp.float32(400.0))
    x_final = _run_ode(X0_DEFAULT, hubs_ex, dt_min=30.0)

    MG_init  = float(X0_DEFAULT[IDX_MG])
    MG_final = float(x_final[IDX_MG])
    Lac_final = float(x_final[IDX_LAC])

    print(f"\nT3: MG_init={MG_init:.0f}g  MG_final={MG_final:.0f}g  drop={100*(MG_init-MG_final)/MG_init:.1f}%  Lac={Lac_final:.2f} mmol/L")
    assert MG_final < MG_init * 0.85, (
        f"Muscle glycogen should deplete strongly at 400 W: "
        f"MG_final={MG_final:.1f} g (expected < {MG_init * 0.85:.1f} g)"
    )
    assert Lac_final > 2.0, (
        f"Lactate should rise with high-power exercise: "
        f"Lac_final={Lac_final:.2f} mmol/L (expected > 2.0)"
    )


# -- T4: UKF CGM assimilation stability
def test_ukf_cgm_assimilation():
    """
    Assimilate 60 steps of synthetic CGM (1-min, 90 +/- 5 mg/dL).
    Assertions:
        (a) posterior_mean NaN-free after 60 updates (alpha=0.10 mandatory).
        (b) covariance diagonal all positive (jitter 1e-3 guarantees PSD).
    """
    filt  = MetabolicGlucoseFilter()
    state = GaussianState(mean=X0_DEFAULT, cov=P0_DEFAULT)

    rng = jax.random.PRNGKey(0)
    for step in range(60):
        rng, key = jax.random.split(rng)
        cgm_reading = 90.0 + float(jax.random.normal(key) * 5.0)
        state = filt.update_state(state, cgm_reading=cgm_reading)

    print(f"\nT4: posterior_mean={state.mean[:3]}  cov_diag={jnp.diag(state.cov)[:3]}")
    assert not bool(jnp.any(jnp.isnan(state.mean))), (
        f"UKF posterior_mean contains NaN after 60 CGM steps: {state.mean}"
    )
    diag = jnp.diag(state.cov)
    assert bool(jnp.all(diag > 0.0)), (
        f"UKF covariance diagonal must be positive: {diag}"
    )
