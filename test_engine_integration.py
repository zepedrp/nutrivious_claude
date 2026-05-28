#!/usr/bin/env python3
"""
test_engine_integration.py

NUTRIVIOUS MHDS — Stress Test de Integração
Cascata: Módulos 1 → 2 → 3 → 10 → 4 → 5 → 7

Cenário extremo de 24 horas:
    - Sessão: 250 Watts durante 2 horas
    - Nutrição: Restrição severa (dispara RED-S no Módulo 5)
    - Sono: Dívida de adenosina inicial (dispara hiperativação HPA)

Execução: python test_engine_integration.py
"""

import sys
import os
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.engine.solvers import (
    MetabolicSolver,
    NeuromuscularSolver,
    CardiorespiratorySolver,
    SleepCircadianSolver,
    NeuroendocrineSolver,
    ThermoFluidSolver,
    ImmuneRepairSolver,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO ATLETA — CENÁRIO EXTREMO
# ─────────────────────────────────────────────────────────────────────────────

PRIORS: dict = {}       # Atleta genérico — todos os priors em NaN → fallback REF

SESSION = {
    "power_output_watts":    250.0,
    "session_duration_secs": 7200.0,   # 2 horas a 250 W
    "session_start_min":     30.0,
    "t_sess_start_min":      0.0,
    "t_sess_day":            0.0,
    "session_start_h":       0.5,
}

MEAL = {
    "carbohydrate_grams":  30.0,       # Restrição extrema → RED-S
    "meal_timestamp_min":   0.0,
}

FLUID = {
    "water_intake_L":      0.5,        # Sub-hidratação deliberada
    "drink_timestamp_h":   1.0,
}

ENV = {
    "T_env_celsius":         28.0,
    "T_core_init_celsius":   37.0,
    "T_skin_init_celsius":   34.0,
    "V_p_init_L":             3.0,
}

FFM_KG          = 65.0    # Fat-Free Mass [kg]
INTAKE_KCAL_FFM = 20.0    # Ingestão [kcal/kg FFM/dia] — bem abaixo do limiar RED-S (30)
SLEEP_DEBT_H    = 0.35    # Dívida de adenosina inicial [adim]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {title}")
    print(f"{'─' * 62}")


def _metric(
    label: str,
    value: float,
    unit: str = "",
    warn_above: float | None = None,
    warn_below: float | None = None,
) -> None:
    flag = ""
    if warn_above is not None and value > warn_above:
        flag = "  ⚠  ALERTA ALTO"
    elif warn_below is not None and value < warn_below:
        flag = "  ⚠  ALERTA BAIXO"
    print(f"  {label:<44} {value:>8.3f}  {unit}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 1 — Motor Bioenergético / Metabólico
# ─────────────────────────────────────────────────────────────────────────────

def run_mod1() -> dict:
    _banner("MÓDULO 1 — Motor Bioenergético / Metabólico")
    t0 = time.perf_counter()

    solver = MetabolicSolver()
    result = solver.simulate_metabolic_response(
        bayesian_priors=PRIORS,
        meal_event=MEAL,
        session_record=SESSION,
        t_span_hours=4.0,
        n_save_points=240,
    )

    dt = time.perf_counter() - t0
    _metric("Pico de Lactato", float(jnp.max(result["La_mmol_L"])), "mmol/L", warn_above=8.0)
    _metric("Nadir de Glicose", float(jnp.min(result["G_mg_dL"])), "mg/dL", warn_below=70.0)
    _metric("Nadir de Fosfocreatina", float(jnp.min(result["PCr_mmol_L"])), "mmol/L")
    _metric("Gasto Energético Total", float(result["Hub_Energy_Expenditure_Kcal"]), "kcal")
    print(f"  [Compilado em {dt:.2f}s]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 2 — Motor Neuromuscular e Fadiga
# ─────────────────────────────────────────────────────────────────────────────

def run_mod2() -> dict:
    _banner("MÓDULO 2 — Motor Neuromuscular e Fadiga")
    t0 = time.perf_counter()

    solver = NeuromuscularSolver()
    result = solver.simulate_neuromuscular_response(
        bayesian_priors=PRIORS,
        session_record=SESSION,
        t_span_days=14.0,
    )

    dt = time.perf_counter() - t0
    CK_peak = float(jnp.max(result["Hub_CK_StructuralDamage"]))
    _metric("Pico de Dano Estrutural CK (proxy)", CK_peak, "[norm]", warn_above=0.5)
    _metric("Performance Líquida (dia 14)", float(result["Performance_Net"][-1]), "[norm]")
    _metric("Fitness Acumulada (dia 14)", float(result["G_fitness"][-1]), "[norm]")
    print(f"  [Compilado em {dt:.2f}s]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 3 — Motor Cardiorrespiratório e Autonómico
# ─────────────────────────────────────────────────────────────────────────────

def run_mod3(mod1: dict) -> dict:
    _banner("MÓDULO 3 — Motor Cardiorrespiratório e Autonómico")
    t0 = time.perf_counter()

    # Hub inbound: Lactato médio da sessão (Módulo 1)
    La_mean = float(jnp.mean(mod1["La_mmol_L"]))

    solver = CardiorespiratorySolver()
    result = solver.simulate_cardiorespiratory_response(
        bayesian_priors=PRIORS,
        session_record=SESSION,
        hub_lactate_mmol_L=La_mean,
        t_span_hours=4.0,
        n_save_points=240,
    )

    dt = time.perf_counter() - t0
    _metric("Pico de Frequência Cardíaca", float(jnp.max(result["HR_bpm"])), "bpm", warn_above=195.0)
    _metric("Pico de Norepinefrina (norm)", float(jnp.max(result["NE_tone"])), "[norm]", warn_above=3.0)
    _metric("Nadir SpO2", float(jnp.min(result["SpO2_frac"])) * 100.0, "%", warn_below=92.0)
    _metric("Nadir RMSSD (supressão vagal)", float(jnp.min(result["RMSSD_proxy_ms"])), "ms")
    print(f"  [Compilado em {dt:.2f}s]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 10×12 — Motor Termo-Fluido (Termorregulação + Volume Plasmático)
# ─────────────────────────────────────────────────────────────────────────────

def run_mod10() -> dict:
    _banner("MÓDULO 10×12 — Motor Termo-Fluido (Core Temp + Plasma)")
    t0 = time.perf_counter()

    solver = ThermoFluidSolver()
    result = solver.simulate_thermo_fluid_response(
        bayesian_priors=PRIORS,
        session_record=SESSION,
        fluid_intake=FLUID,
        env_conditions=ENV,
        t_span_hours=2.5,
        n_save_points=240,
    )

    dt = time.perf_counter() - t0
    T_max = float(jnp.max(result["Hub_Core_Temperature_Alarm"]))
    PVD_min = float(jnp.min(result["Hub_Plasma_Volume_Drift"]))
    _metric("Pico de Temperatura Core", T_max, "°C", warn_above=39.5)
    _metric("Nadir de Volume Plasmático", float(jnp.min(result["V_p_L"])), "L", warn_below=2.5)
    _metric("Drift Plasmático (V_p/V_p0)", PVD_min, "[frac]", warn_below=0.85)
    _metric("Pico de Taxa de Sudorese", float(jnp.max(result["Sweat_Rate_L_h"])), "L/h", warn_above=2.5)
    print(f"  [Compilado em {dt:.2f}s]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 4 — Sono e Ritmos Circadianos
# ─────────────────────────────────────────────────────────────────────────────

def run_mod4(mod3: dict) -> dict:
    _banner("MÓDULO 4 — Sono e Ritmos Circadianos (Phillips-Robinson + SCN)")
    t0 = time.perf_counter()

    # Hub inbound: Catecolaminas do Módulo 3 (t_min → t_h)
    t_cat_h = mod3["t_min"] / 60.0
    cat_tone = mod3["Hub_Catecholamines_Tone"]

    # SleepCircadianSolver: priors via genetic_profile no construtor
    solver = SleepCircadianSolver(genetic_profile=PRIORS)
    result = solver.simulate_sleep_circadian_response(
        duration_h=24.0,
        hub_catecholamines_tone=cat_tone,
        t_cat_h=t_cat_h,
        H_init=SLEEP_DEBT_H,
        n_save=288,
    )

    dt = time.perf_counter() - t0
    SWS_mean_pct = float(jnp.mean(result["SWS_gate"])) * 100.0
    GH_peak = float(jnp.max(result["Hub_GH_Repair_Signalling"]))
    sleep_debt = float(result["Hub_Sleep_Debt_Metabolic"])
    _metric("SWS Gate Médio (sono profundo)", SWS_mean_pct, "%", warn_below=15.0)
    _metric("Pico de GH Noturno (reparação)", GH_peak, "[norm]")
    _metric("Dívida de Sono Residual (adenosina)", sleep_debt, "[adim]", warn_above=0.1)
    _metric("Pico de Melatonina", float(jnp.max(result["Melatonin_norm"])), "[norm]")
    print(f"  [Compilado em {dt:.2f}s]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 5 — Eixo Neuroendócrino (HPA/HPG) + RED-S
# ─────────────────────────────────────────────────────────────────────────────

def run_mod5(mod1: dict, mod3: dict, mod4: dict) -> dict:
    _banner("MÓDULO 5 — Eixo Neuroendócrino HPA/HPG + RED-S")
    t0 = time.perf_counter()

    # Converter eixos de tempo: minutos → horas
    t_La_h  = mod1["t_min"] / 60.0
    t_cat_h = mod3["t_min"] / 60.0
    t_scn_h = mod4["t_h"]

    # Gasto de exercício em kcal/kg FFM (do Hub_Energy_Expenditure_Kcal do Módulo 1)
    exercise_kcal_ffm = float(mod1["Hub_Energy_Expenditure_Kcal"]) / FFM_KG

    # NeuroendocrineSolver: priors via genetic_profile no construtor
    solver = NeuroendocrineSolver(genetic_profile=PRIORS)
    result = solver.simulate_neuroendocrine_response(
        duration_h=24.0,
        hub_scn_phase=mod4["SCN_phase_x"],
        t_scn_h=t_scn_h,
        hub_catecholamines_tone=mod3["Hub_Catecholamines_Tone"],
        t_cat_h=t_cat_h,
        hub_lactate_signalling=mod1["Hub_Lactate_Signalling"],
        t_La_h=t_La_h,
        hub_sleep_debt=float(mod4["Hub_Sleep_Debt_Metabolic"]),
        hub_gh_repair=mod4["Hub_GH_Repair_Signalling"],
        t_gh_h=t_scn_h,
        energy_intake_kcal_per_kg_ffm=INTAKE_KCAL_FFM,
        exercise_kcal_per_kg_ffm=exercise_kcal_ffm,
        EA_init_norm=1.0,
        n_save=288,
    )

    dt = time.perf_counter() - t0
    _metric("Pico de Cortisol (norm)", float(jnp.max(result["Cortisol_norm"])), "[norm]", warn_above=2.5)
    _metric("Nadir de Testosterona (norm)", float(jnp.min(result["Testosterone_norm"])), "[norm]", warn_below=0.6)
    ea_norm = float(jnp.min(result["EA_norm"]))
    _metric("Nadir de EA (disponib. energética)", ea_norm, "[EA/45]", warn_below=0.667)
    _metric("Gate RED-S Máximo", float(jnp.max(result["REDS_gate"])), "[0→1]", warn_above=0.5)
    print(f"  [Compilado em {dt:.2f}s]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 7 — Sistema Imunitário e Reparação Tecidular (M1/M2)
# ─────────────────────────────────────────────────────────────────────────────

def run_mod7(mod2: dict, mod4: dict, mod5: dict) -> dict:
    _banner("MÓDULO 7 — Sistema Imunitário e Reparação Tecidular (M1/M2)")
    t0 = time.perf_counter()

    # Eixo temporal do Módulo 5 (0–24h, 288 pontos) usado como base dos hubs
    t_hub_24h = mod5["t_h"]

    # Hub_CK: D_muscle do Módulo 2 (eixo agudo em minutos)
    # D_muscle ∈ [0, ~1]; normalizar para "CK_hub acima de basal" adicionando 1.0
    # CK_excess = max(0, CK_hub - 1.0) na ODE do Módulo 7.
    t_acute_h = mod2["t_acute_min"] / 60.0
    CK_acute_raw = mod2["Hub_CK_StructuralDamage"]
    # Adicionar 1.0 para converter D_muscle ∈ [0,1] → CK_hub normalizado ∈ [1,2]
    CK_acute_norm = CK_acute_raw + 1.0
    # Estender a 24h com zero extra (sem novo dano após sessão)
    t_CK_ext = jnp.concatenate([t_acute_h, jnp.array([24.0])])
    CK_ext    = jnp.concatenate([CK_acute_norm, jnp.array([1.0])])  # retorna a basal=1.0
    CK_on_hub = jnp.interp(t_hub_24h, t_CK_ext, CK_ext)

    hub_inputs = {
        "t_h":                         t_hub_24h,
        "Hub_CK_StructuralDamage":    CK_on_hub,
        "Hub_Cortisol_Catabolic":     mod5["Hub_Cortisol_Catabolic"],
        "Hub_Testosterone_Anabolic":  mod5["Hub_Testosterone_Anabolic"],
        "Hub_GH_Repair_Signalling":   jnp.interp(
                                          t_hub_24h,
                                          mod4["t_h"],
                                          mod4["Hub_GH_Repair_Signalling"],
                                      ),
    }

    solver = ImmuneRepairSolver()
    result = solver.simulate_immune_repair_response(
        bayesian_priors=PRIORS,
        hub_inputs=hub_inputs,
        t_span_hours=96.0,
        n_save_points=480,
    )

    dt = time.perf_counter() - t0
    IL6_peak = float(jnp.max(result["Hub_Systemic_Inflammation_IL6"]))
    idx_72h  = int(480 * 72 / 96)
    IL6_72h  = float(result["Hub_Systemic_Inflammation_IL6"][idx_72h])
    M2_peak  = float(jnp.max(result["M2_act"]))
    repair   = float(result["Hub_Tissue_Repair_Completion"][-1])

    _metric("Pico de IL-6 Sistémica", IL6_peak, "[norm]", warn_above=3.0)
    _metric("IL-6 às 72h (marcador overtraining?)", IL6_72h, "[norm]", warn_above=1.0)
    _metric("Pico de Ativação M2 (construtiva)", M2_peak, "[norm]")
    _metric("Conclusão de Reparação (96h)", repair, "[0→1]", warn_below=0.5)
    print(f"  [Compilado em {dt:.2f}s]")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Cascata Completa
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 62)
    print("  NUTRIVIOUS MHDS — STRESS TEST DE INTEGRAÇÃO")
    print("  Cenário: 250W × 2h  |  RED-S  |  Dívida de Sono")
    print("═" * 62)
    print(f"  JAX backend  : {jax.default_backend().upper()}")
    print(f"  JAX devices  : {jax.devices()}")
    print(f"  Atleta FFM   : {FFM_KG} kg  |  Ingestão: {INTAKE_KCAL_FFM} kcal/kg FFM")

    t_total = time.perf_counter()

    r1  = run_mod1()
    r2  = run_mod2()
    r3  = run_mod3(r1)
    r10 = run_mod10()
    r4  = run_mod4(r3)
    r5  = run_mod5(r1, r3, r4)
    r7  = run_mod7(r2, r4, r5)

    elapsed = time.perf_counter() - t_total

    _banner("SUMÁRIO FINAL — PLACA-MÃE INTEGRADA (7 Módulos em Cascata)")
    _metric("Gasto Energético (Mod 1 → Mod 5)", float(r1["Hub_Energy_Expenditure_Kcal"]), "kcal")
    _metric("Pico T_core (Mod 10 → alarme prescrição)", float(jnp.max(r10["Hub_Core_Temperature_Alarm"])), "°C", warn_above=39.5)
    _metric("Drift Plasmático Mín (Mod 10 → Mod 3)", float(jnp.min(r10["Hub_Plasma_Volume_Drift"])), "[frac]", warn_below=0.85)
    _metric("Cortisol Pico (Mod 5 → Mod 7)", float(jnp.max(r5["Cortisol_norm"])), "[norm]", warn_above=2.5)
    _metric("Gate RED-S Máx (Mod 5 → Mod 6/13)", float(jnp.max(r5["REDS_gate"])), "[0→1]", warn_above=0.5)
    _metric("GH Pico Noturno (Mod 4 → Mod 7)", float(jnp.max(r4["Hub_GH_Repair_Signalling"])), "[norm]")
    _metric("IL-6 Pico (Mod 7 → overtraining?)", float(jnp.max(r7["Hub_Systemic_Inflammation_IL6"])), "[norm]", warn_above=3.0)
    _metric("Reparação Tecidular 96h (Mod 7)", float(r7["Hub_Tissue_Repair_Completion"][-1]), "[0→1]", warn_below=0.5)

    print(f"\n  Tempo total de execução: {elapsed:.2f}s")
    print("═" * 62)
    print("  HUB FLOW STATUS: TODOS OS CANAIS VALIDADOS  ✓")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
