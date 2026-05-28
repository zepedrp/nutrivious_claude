"""
app/engine/solvers/metabolic_solver.py

Módulo 1 — Motor Bioenergético / Metabólico (MHDS Subsistema 1) — Patch v1.2

Unifica quatro contribuições físicas num único sistema de 4 ODEs acopladas:
    I.  Bergman-Cobelli Expandido + Roy-Parker 2007 + Holloszy GLUT4 post-exercício
        Glucosa plasmática G(t) + Insulina remota X(t)
    II. Cinética de Fosfatos 3-Componentes (Boillet, Messonnier & Cohen 2024)
        Fosfocreatina muscular PCr(t) — aeróbio + anaeróbio láctico + aláctico
    III.Modelo de Transporte de Lactato de Brooks (1986, 2018) — Lactate Shuttle
        Lactato sanguíneo La(t) — variável Hub exportada para acoplamento multi-módulo

══════════════════════════════════════════════════════════════════════════════
SISTEMA DE EQUAÇÕES DIFERENCIAIS (unidades SI-biológicas) — v1.2
══════════════════════════════════════════════════════════════════════════════

Vetor de estado:  y = [G,  X,  PCr,  La]

Estado   Variável             Unidade       Escala temporal
──────   ─────────────────    ──────────    ─────────────────────────────────
y[0]     G(t)  glucosa        mmol·L⁻¹      τ_G  ≈ 26 min   (lento)
y[1]     X(t)  insulina rem.  min⁻¹         τ_X  ≈ 40 min   (lento)
y[2]     PCr(t) fosfocreatina mmol·L⁻¹      τ_PCr ≈ 1–3 min  (rápido)
y[3]     La(t) lactato        mmol·L⁻¹      τ_La ≈ 20 min   (moderado)

Rácio de rigidez (stiffness): τ_X / τ_PCr ≈ 40 → sistema moderadamente stiff.
Solver Kvaerno5 (DIRK implícito 5ª ordem) + PIDController.

──────────────────────────────────────────────────────────────────────────────
I. BERGMAN-COBELLI EXPANDIDO v1.2
   (Bergman 1979 + Richter 1992 + Roy & Parker 2007 + Holloszy 2005)
──────────────────────────────────────────────────────────────────────────────

dG/dt = −(p₁ + X)·G + p₁·Gb + Ra(t) − u_mec·P(t) − G_uptake_post(t)

    −(p₁ + X)·G   : clearance dependente de insulina (GLUT4 via sinalização IR-IRS1-AKT)
    p₁·Gb         : produção hepática basal de glucosa (glicogenólise + neoglicogénese)
    Ra(t)         : aparecimento exógeno de glucosa (Dalla Man 2006, mono-exponencial)
    u_mec·P(t)    : captação muscular contráctil GLUT4 via AMPK (Richter 1992) — intra-sessão
    G_uptake_post : GLUT4 pós-exercício persistente (Holloszy 2005):
                    G_uptake_post = u_mec_post × power_w × Θ(t−t_sess_end) × exp(−k_GLUT4_decay·Δt_post)
                    τ_GLUT4_post = 1/k_GLUT4_decay ≈ 45 min (Richter & Hargreaves 2013)

dX/dt = −p₂·X + p3_eff·(I(t) − Ib)                              [min⁻²]

    p3_eff(t) = p3_base × (1 + k_SI_exercise × P(t) / P_SI_ref)

    Sensibilidade à insulina dinâmica durante o exercício (Roy & Parker 2007 Annals BME):
    A sinalização AMPK aumenta a densidade de GLUT4 no sarcolema → S_I efetiva cresce com
    a intensidade. k_SI_exercise ≈ 0.35 calibrado de Roy & Parker (2007) + Resalat et al.
    (2019) (Bayesian-validated T1D-exercise extension): 35% de aumento a 200W de referência.
    Ao repouso (P=0): p3_eff = p3_base (sem alteração); a P=P_ref: +35%; a P=2×P_ref: +70%.

I(t) [µU·mL⁻¹]: insulina bifásica pós-prandial (simplificada Dalla Man 2006):
    I(t) = Ib + meal_gate · I_peak · k_I · max(t−t_meal,0) · exp(−k_I·max(t−t_meal,0))
    Pico em t_meal + 1/k_I ≈ 20 min com k_I = 0.05 min⁻¹.

Ra(t) [mmol·L⁻¹·min⁻¹]: aparecimento mono-exponencial de glucosa da refeição:
    Ra(t) = (C_meal/V_g) · k_abs · exp(−k_abs · max(t−t_meal, 0))
    onde C_meal = carb_mmol = carbohydrate_grams / 180.16 × 1000 [mmol]

──────────────────────────────────────────────────────────────────────────────
II. CINÉTICA DE FOSFATOS 3-COMPONENTES
    (Boillet, Messonnier & Cohen 2024 + Meyer 1988 + Wallimann 1992)
──────────────────────────────────────────────────────────────────────────────

Balanço de potência Bond Graph (3 sistemas energéticos):
    P_demand = P_aeróbio + P_glicólítico + P_aláctico(PCr)

    V_demand       = k_CK · P(t)                         [mmol·L⁻¹·min⁻¹]  demanda total
    V_OXPHOS       = Vmax_OXPHOS · (1 − PCr/PCr_max)     [mmol·L⁻¹·min⁻¹]  supply aeróbio
    V_glyc_gap     = max(0, V_demand − V_OXPHOS)          [mmol·L⁻¹·min⁻¹]  gap → glicólise
    V_glyc_rephos  = k_glyc_pcr_frac · V_glyc_gap         [mmol·L⁻¹·min⁻¹]  fração → PCr

dPCr/dt = V_OXPHOS + V_glyc_rephos − V_demand             [mmol·L⁻¹·min⁻¹]

Interpretação física da rede de fosfotransferência (Wallimann et al. 1992):
Quando P(t) excede a capacidade aeróbia (V_glyc_gap > 0), a glicólise anaeróbia produz ATP
via fosforilação a nível de substrato. Uma fração k_glyc_pcr_frac ≈ 0.15 deste ATP glicólítico
re-fosforila PCr via CK em sentido inverso, ABRANDANDO a depleção de PCr em ≈15% durante
sprints. O restante ATP glicólítico vai directamente para trabalho mecânico e produção de La.

Forma simplificada (substituindo e expandindo):
    P > threshold: dPCr/dt = (1−k_glyc_pcr_frac)·(V_OXPHOS−V_demand)   [depleção mais lenta]
    P ≤ threshold: dPCr/dt = V_OXPHOS − V_demand                        [idêntico a Meyer 1988]

Calibração: Casey et al. (1996) biopsy: PCr_max ≈ 20 mmol·L⁻¹;
            Meyer (1988) Tabela 1: Vmax_OXPHOS ≈ 18 mmol·L⁻¹·min⁻¹;
            k_CK=0.085 → P=200W → ~17 mmol·L⁻¹·min⁻¹ demanda total.

──────────────────────────────────────────────────────────────────────────────
III. MODELO DE TRANSPORTE DE LACTATO DE BROOKS — LACTORMONA (Hub Variable)
     (Brooks 1986, 2018; Pedersen et al. 2003 — BDNF induction)
──────────────────────────────────────────────────────────────────────────────

dLa/dt = k_La_prod·max(0, P(t)−P_LT)                      [mmol·L⁻¹·min⁻¹]
         + k_La_glyc·max(0, PCr_max−PCr)·sess_gate(t)
         − k_La_clear·(La − La_b)

O lactato NÃO é lixo metabólico: é uma "Lactormona" — molécula de sinalização sistémica:
    • Combustível cardíaco e de fibras tipo I (Cell-Cell Lactate Shuttle, Brooks 2018)
    • Indutor de BDNF (Brain-Derived Neurotrophic Factor) via MCT2 cerebral (Pedersen 2003)
      → acoplamento Hub para Módulo 9 (Cognitivo e Fadiga Central)
    • Blunting do eixo HPA via cortisol supressor em exercício sustentado
      → acoplamento Hub para Módulo 5 (Neuroendócrino)
    • Substrato alternativo ao glucose cardíaco durante hipoglicémia de exercício
      → acoplamento Hub para Módulo 3 (Cardiorrespiratório)

A chave `Hub_Lactate_Signalling` no dicionário de retorno é o canal formal de saída
desta variável Hub para o RBPF da Fase 4 e o orquestrador multi-módulo.

══════════════════════════════════════════════════════════════════════════════
ACOPLAMENTOS CROSS-SUBSISTEMA (Hub Variables — Bond Graphs) — v1.2
══════════════════════════════════════════════════════════════════════════════

    PCr → La  : depleção de PCr activa glicólise anaeróbia → La ↑
    La  → PCr : glicólise alimenta re-fosforilação de PCr (V_glyc_rephos) [NOVO v1.2]
    G   → X   : glucosa estimula secreção de insulina → X sobe → G clearance ↑
    P(t)→ G   : GLUT4 contráctil remove glucosa independentemente de insulina
    P(t)→ X   : p3_eff sobe com P(t) → S_I dinâmica [NOVO v1.2]
    P(t)→ G   : efeito GLUT4 pós-exercício persistente [NOVO v1.2]
    P(t)→ PCr : demanda mecânica esgota PCr pela via CK
    P(t)→ La  : potência acima de P_LT gera overflow glicolítico
    La  → [Hub]: BDNF / HPA / cardíaco (export Hub_Lactate_Signalling) [NOVO v1.2]

══════════════════════════════════════════════════════════════════════════════
SOLVER E STACK
══════════════════════════════════════════════════════════════════════════════

Solver:     diffrax.Kvaerno5 (DIRK 5ª ordem, stiff-safe)
Step ctrl:  diffrax.PIDController (rtol=1e-4, atol=1e-6)
dt0:        0.1 min (6 s) — captura transiente rápido de PCr no início do sprint
max_steps:  32 768 — permite simulações de 4h com passos adaptativos finos
Stack:      JAX (jax.numpy) + Diffrax; dtype float32; JIT-compilável

Referências v1.2 (adicionadas):
    Roy & Parker (2007) Ann Biomed Eng 35:643–655           [S_I dinâmica exercício]
    Resalat et al. (2019) J Diabetes Sci Technol 13:1091    [T1D-exercise Bayesian ext.]
    Boillet, Messonnier & Cohen (2024) Nature Sci Rep        [3-component digital twin]
    Holloszy (2005) J Appl Physiol 99:338–343               [GLUT4 post-exercise]
    Wallimann et al. (1992) Biochem J 281:21–40             [phosphotransfer network]
    Pedersen et al. (2003) J Physiol 549:243–252            [lactate → BDNF induction]

Referências base (anteriores):
    Bergman et al. (1979) J Clin Invest 63:1456–1467
    Cobelli et al. (1984) Am J Physiol 246:E667–E677
    Dalla Man et al. (2006) IEEE Trans Biomed Eng 53:1740–1749
    Meyer (1988) Am J Physiol 254:C548–C553
    Brooks (1986) J Appl Physiol 60:2083–2092
    Brooks (2018) Cell Metab 27:757–785
    Richter & Hargreaves (2013) Physiol Rev 93:993–1017
    Baar et al. (2004) FASEB J 18:1175–1182
    Coyle (1992) J Appl Physiol 72:467–475
    Casey et al. (1996) J Physiol 492:887–895
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax

# ── Constantes físicas globais ─────────────────────────────────────────────────
_GLUCOSE_MW_G_PER_MOL: float = 180.16    # g·mol⁻¹
_MIN_PER_HOUR: float = 60.0
_MG_DL_PER_MMOL_L: float = 18.0182      # mmol·L⁻¹ → mg·dL⁻¹


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura de Parâmetros Unificada
# ─────────────────────────────────────────────────────────────────────────────

class MetabolicParams(NamedTuple):
    """
    Vector de parâmetros físicos do Motor Metabólico Unificado (Módulo 1) — v1.2.

    Cobre os quatro modelos base: Bergman-Cobelli + Roy-Parker (G/X),
    Holloszy GLUT4 post-exercise (G), Boillet-Meyer 3-component (PCr), Brooks (La).
    Implementado como NamedTuple para ser pytree JAX nativo.
    """
    # ── I. Bergman-Cobelli (Glucosa / Insulina) ───────────────────────────────
    p1: float           # clearance de glucosa independente de insulina  [min⁻¹]
    p2: float           # taxa de perda do compartimento remoto           [min⁻¹]
    p3: float           # sensibilidade basal à insulina (Fase 3 ← HOMA-IR) [min⁻² per µU·mL⁻¹]
    Gb: float           # glucosa plasmática basal                        [mmol·L⁻¹]
    Ib: float           # insulina plasmática basal                       [µU·mL⁻¹]
    V_g: float          # volume de distribuição de glucosa               [L]
    k_abs: float        # taxa de absorção mono-exponencial da refeição    [min⁻¹]
    I_peak: float       # pico pós-prandial de insulina acima de Ib       [µU·mL⁻¹]
    k_I: float          # taxa de decaimento do surto bifásico de insulina [min⁻¹]
    u_mec: float        # GLUT4-contráctil intra-sessão                   [mmol·W⁻¹·min⁻¹·L⁻¹]

    # ── I-ext. Roy & Parker 2007 — S_I dinâmica + Holloszy GLUT4 ────────────
    k_SI_exercise: float  # coeficiente de aumento de S_I por potência norm. [adim.]
                          # p3_eff = p3 × (1 + k_SI_exercise × P/P_SI_ref)
                          # Calibrado Roy & Parker (2007): 35% boost @ P_SI_ref
    P_SI_ref: float       # potência de referência para normalização S_I     [W]
    u_mec_post: float     # GLUT4 pós-exercício (Holloszy 2005)              [mmol·W⁻¹·min⁻¹·L⁻¹]
    k_GLUT4_decay: float  # taxa de decaimento GLUT4 post-exercício          [min⁻¹]
                          # τ = 1/k_GLUT4_decay ≈ 45 min (Richter 2013)

    # ── II. Boillet-Meyer — Cinética de Fosfatos 3-Componentes (PCr) ─────────
    PCr_max: float          # concentração de PCr em repouso completo       [mmol·L⁻¹]
    Vmax_OXPHOS: float      # taxa máx. de resíntese oxidativa de PCr       [mmol·L⁻¹·min⁻¹]
    k_CK: float             # coupling W → hidrólise de PCr via CK          [mmol·W⁻¹·min⁻¹·L⁻¹]
    k_glyc_pcr_frac: float  # fração do ATP glicólítico → re-fosforilação PCr [adim., 0–1]
                            # Wallimann (1992) phosphotransfer network; Boillet (2024): ≈0.15

    # ── III. Brooks — Transporte de Lactato (Hub Variable) ───────────────────
    La_b: float         # lactato basal em repouso                          [mmol·L⁻¹]
    P_LT: float         # potência no limiar de lactato (LT1)               [W]
    k_La_prod: float    # overflow glicolítico acima de P_LT                [mmol·W⁻¹·min⁻¹·L⁻¹]
    k_La_glyc: float    # glicólise anaeróbia acoplada à depleção de PCr    [min⁻¹]
    k_La_clear: float   # clearance periférica de lactato (MCT)             [min⁻¹]


# ─────────────────────────────────────────────────────────────────────────────
# Campo Vectorial ODE — JIT-compilável puro
# ─────────────────────────────────────────────────────────────────────────────

def bioenergetic_ode(
    t: jax.Array,
    y: jax.Array,
    args: tuple,
) -> jax.Array:
    """
    Campo vectorial JIT-compilável do sistema bioenergético unificado v1.2 (4 estados).

    Adições v1.2 vs v1.1:
        • p3_eff(t): S_I dinâmica durante exercício (Roy & Parker 2007)
        • G_uptake_post: GLUT4 persistente pós-exercício (Holloszy 2005)
        • V_glyc_rephos: re-fosforilação de PCr via ATP glicólítico (Boillet 2024)
        • Hub_Lactate_Signalling: La exportado com chave Hub (return do orquestrador)

    Mapeamento y = [G, X, PCr, La]:
        y[0] = G(t)   — glucosa plasmática      [mmol·L⁻¹]
        y[1] = X(t)   — ação remota da insulina [min⁻¹]
        y[2] = PCr(t) — fosfocreatina muscular  [mmol·L⁻¹]
        y[3] = La(t)  — lactato sanguíneo       [mmol·L⁻¹]
    """
    params, carb_mmol, power_w, t_meal, t_sess_start, sess_dur_min = args

    G   = y[0]
    X   = y[1]
    PCr = y[2]
    La  = y[3]

    # Porta suave (Heaviside diferenciável via tanh).
    # Slope=20 → transição em ~0.1 min; JIT-safe sem branches Python em valores JAX.
    def _smooth_on(t_start: jax.Array) -> jax.Array:
        return 0.5 * (1.0 + jnp.tanh(20.0 * (t - t_start)))

    # ── Porta intra-sessão: activa entre [t_sess_start, t_sess_start + sess_dur_min] ──
    sess_gate = _smooth_on(t_sess_start) - _smooth_on(t_sess_start + sess_dur_min)
    P_t = power_w * sess_gate

    # ── Porta pós-sessão + decaimento GLUT4 (Holloszy 2005) ───────────────────
    post_gate = _smooth_on(t_sess_start + sess_dur_min)
    dt_post = jnp.maximum(0.0, t - (t_sess_start + sess_dur_min))
    G_uptake_post = (
        params.u_mec_post
        * power_w
        * post_gate
        * jnp.exp(-params.k_GLUT4_decay * dt_post)
    )

    # ── Ra(t): aparecimento mono-exponencial de glucosa da refeição ────────────
    dt_meal = t - t_meal
    meal_gate = _smooth_on(t_meal)
    Ra = (
        meal_gate
        * (carb_mmol / params.V_g)
        * params.k_abs
        * jnp.exp(-params.k_abs * jnp.maximum(dt_meal, 0.0))
    )

    # ── I(t): surto bifásico pós-prandial de insulina (Dalla Man 2006) ─────────
    dt_meal_c = jnp.maximum(dt_meal, 0.0)
    I_t = (
        params.Ib
        + meal_gate
        * params.I_peak
        * params.k_I
        * dt_meal_c
        * jnp.exp(-params.k_I * dt_meal_c)
    )

    # ══════════════════════════════════════════════════════════════════════════
    # I. BERGMAN-COBELLI v1.2: dG/dt e dX/dt
    # ══════════════════════════════════════════════════════════════════════════

    # S_I dinâmica: Roy & Parker (2007) — AMPK aumenta translocalização GLUT4,
    # amplificando a sensibilidade à insulina proporcionalmente à intensidade.
    # Ao repouso (P_t=0): p3_eff=p3_base; @ P_SI_ref: +k_SI_exercise×100%.
    p3_eff = params.p3 * (1.0 + params.k_SI_exercise * P_t / params.P_SI_ref)

    # dG/dt: glucosa basal + aparecimento refeição − clearance insulin-dependente
    #        − captação contráctil intra-sessão − captação GLUT4 pós-exercício
    dG_dt = (
        -(params.p1 + X) * G
        + params.p1 * params.Gb
        + Ra
        - params.u_mec * P_t
        - G_uptake_post
    )

    # dX/dt: compartimento remoto de insulina com S_I dinâmica
    dX_dt = -params.p2 * X + p3_eff * (I_t - params.Ib)

    # ══════════════════════════════════════════════════════════════════════════
    # II. BOILLET-MEYER 3-COMPONENTES: dPCr/dt
    # ══════════════════════════════════════════════════════════════════════════

    # Balanço de potência Bond Graph (3 sistemas):
    # V_OXPHOS:      supply aeróbio (OXPHOS mitocondrial)
    # V_glyc_gap:    gap de potência acima da capacidade aeróbia → activa glicólise
    # V_glyc_rephos: fração do ATP glicólítico que re-fosforila PCr (Wallimann 1992)
    V_OXPHOS = params.Vmax_OXPHOS * (1.0 - PCr / params.PCr_max)
    V_demand = params.k_CK * P_t
    V_glyc_gap = jnp.maximum(0.0, V_demand - V_OXPHOS)
    V_glyc_rephos = params.k_glyc_pcr_frac * V_glyc_gap

    # dPCr/dt = resíntese aeróbia + re-fosforilação glicólítica − hidrólise CK
    # À alta intensidade: dPCr/dt = (1−k_glyc_pcr_frac)·(V_OXPHOS−V_demand) → depleção mais lenta
    dPCr_dt = V_OXPHOS + V_glyc_rephos - V_demand

    # ══════════════════════════════════════════════════════════════════════════
    # III. BROOKS — TRANSPORTE DE LACTATO (Lactormona / Hub Variable): dLa/dt
    # ══════════════════════════════════════════════════════════════════════════

    # Termo 1: overflow glicolítico aeróbio (acima do limiar de lactato LT1)
    La_aerobic_overflow = params.k_La_prod * jnp.maximum(0.0, P_t - params.P_LT)

    # Termo 2: glicólise anaeróbia acoplada à depleção de PCr (Cell-Cell Shuttle)
    # Fibras tipo IIx recorrem à glicólise quando PCr depleta; lactato produzido
    # é consumido por fibras tipo I e coração via MCT (Brooks 2018).
    La_anaerobic_glyc = (
        params.k_La_glyc
        * jnp.maximum(0.0, params.PCr_max - PCr)
        * sess_gate
    )

    # Termo 3: clearance periférica (fígado, coração, fibras tipo I via MCT1/MCT4)
    La_clearance = params.k_La_clear * (La - params.La_b)

    dLa_dt = La_aerobic_overflow + La_anaerobic_glyc - La_clearance

    return jnp.stack([dG_dt, dX_dt, dPCr_dt, dLa_dt])


# ─────────────────────────────────────────────────────────────────────────────
# Kernel JIT interno — fronteira de compilação JAX
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _solve_bioenergetic(
    y0: jax.Array,
    t0: jax.Array,
    t1: jax.Array,
    dt0: jax.Array,
    ts: jax.Array,
    args: tuple,
) -> diffrax.Solution:
    """
    Kernel JIT-compilado da integração numérica do sistema bioenergético (4 estados).

    Solver — diffrax.Kvaerno5:
        DIRK de 5ª ordem — adequado para sistema multi-escala:
        PCr (τ≈1 min), insulina (τ≈40 min), GLUT4 post (τ≈45 min).

    Step control — diffrax.PIDController(rtol=1e-4, atol=1e-6):
        Passo pequeno no transiente rápido de PCr (sprint onset);
        passo largo nas fases basais lentas de glucosa e lactato.

    max_steps=32_768: dimensionado para 4h com step médio ~0.5 min.
    """
    return diffrax.diffeqsolve(
        terms=diffrax.ODETerm(bioenergetic_ode),
        solver=diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=32_768,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orquestrador Python
# ─────────────────────────────────────────────────────────────────────────────

class MetabolicSolver:
    """
    Orquestrador Python para o Subsistema 1 (Bioenergético/Metabólico) — v1.2.

    Responsabilidades:
        1. Mapear BayesianPriors da Fase 3 → MetabolicParams físicos.
        2. Converter dicts Pydantic-derivados (meal_event, session_record) → JAX float32.
        3. Invocar o kernel JIT-compilado _solve_bioenergetic.
        4. Devolver trajectórias rotuladas incluindo Hub_Lactate_Signalling para o RBPF.

    Parâmetros de referência populacional:
        Bergman (1979) Tabela 2    — p1, p2, p3, Gb, Ib
        Meyer (1988) Tabela 1      — PCr_max, Vmax_OXPHOS, k_CK
        Brooks (1986, 2018)        — La_b, k_La_prod, k_La_clear
        Roy & Parker (2007)        — k_SI_exercise, P_SI_ref
        Holloszy (2005)            — u_mec_post, k_GLUT4_decay
        Boillet et al. (2024)      — k_glyc_pcr_frac
        Coyle (1992)               — u_mec
        Casey et al. (1996)        — PCr_max
    """

    # ── I. Bergman-Cobelli: referências populacionais ─────────────────────────
    _P1_REF: float = 0.028       # [min⁻¹]  — Bergman 1979 Tab.2 média
    _P2_REF: float = 0.025       # [min⁻¹]
    _P3_REF: float = 5.35e-5     # [min⁻² per µU·mL⁻¹]
    _GB_REF: float = 5.0         # [mmol·L⁻¹]  ≈ 90 mg·dL⁻¹
    _IB_REF: float = 10.0        # [µU·mL⁻¹]
    _VG_REF: float = 13.0        # [L]  ≈ 0.18 L·kg⁻¹ × 72 kg
    _K_ABS_REF: float = 0.025    # [min⁻¹]; sólido ~0.01, líquido ~0.05
    _I_PEAK_REF: float = 60.0    # [µU·mL⁻¹] acima de Ib
    _K_I_REF: float = 0.05       # [min⁻¹]; pico em 1/k_I ≈ 20 min
    _U_MEC_REF: float = 3.0e-4   # [mmol·W⁻¹·min⁻¹·L⁻¹]; calibrado Coyle (1992)

    # ── I-ext. Roy & Parker 2007 + Holloszy 2005 ─────────────────────────────
    _K_SI_EXERCISE_REF: float = 0.35   # [adim.]; 35% S_I boost @ P_SI_ref
                                       # Roy & Parker (2007) Ann Biomed Eng 35:643
    _P_SI_REF_W: float = 200.0         # [W]; potência de normalização S_I
    _U_MEC_POST_REF: float = 1.5e-4    # [mmol·W⁻¹·min⁻¹·L⁻¹]; ≈50% u_mec, pós-exercício
                                       # Holloszy (2005) J Appl Physiol 99:338
    _K_GLUT4_DECAY_REF: float = 0.022  # [min⁻¹]; τ = 1/0.022 ≈ 45 min
                                       # Richter & Hargreaves (2013) Physiol Rev 93:993

    # ── II. Boillet-Meyer PCr 3-componentes ──────────────────────────────────
    _PCR_MAX_REF: float = 20.0         # [mmol·L⁻¹]; Casey et al. (1996) biopsy
    _VMAX_OXPHOS_REF: float = 18.0     # [mmol·L⁻¹·min⁻¹]; Meyer (1988) média
    _K_CK_REF: float = 0.085           # [mmol·W⁻¹·min⁻¹·L⁻¹]
    _K_GLYC_PCR_FRAC_REF: float = 0.15 # [adim.]; Boillet et al. (2024) + Wallimann (1992)
                                        # 15% do ATP glicólítico → re-fosforilação PCr

    # ── III. Brooks Lactato: referências populacionais ────────────────────────
    _LA_B_REF: float = 1.0             # [mmol·L⁻¹]; lactato basal em repouso
    _P_LT_REF: float = 150.0           # [W]; LT1 para atleta de moderada aptidão
    _K_LA_PROD_REF: float = 0.020      # [mmol·W⁻¹·min⁻¹·L⁻¹]
    _K_LA_GLYC_REF: float = 0.025      # [min⁻¹]
    _K_LA_CLEAR_REF: float = 0.050     # [min⁻¹]; τ_clear ≈ 20 min (Brooks 1986)

    # ── IV. Eficiência Bruta Humana (Gross Efficiency) ────────────────────────
    # Coyle (1992) + Lucia et al. (2002 Med Sci Sports Exerc): 18–25%, mediana 22%
    # Ajustável por prior mitocondrial (PPARGC1A): GG→+15%, AA→-15%
    _GROSS_EFF_BASE: float = 0.22      # [adim.]; eficiência bruta padrão
    _J_PER_KCAL: float = 4184.0        # [J·kcal⁻¹]

    def _build_params(self, bayesian_priors: dict[str, float]) -> MetabolicParams:
        """
        Constrói MetabolicParams a partir dos BayesianPriors da Fase 3.

        Mapeamentos de priors → parâmetros físicos:

        ① p3_insulin_sensitivity_prior (← HOMA-IR, reciprocal_normalised):
              HOMA-IR = 1.0 → prior = 1.0 → p3 = P3_REF
              HOMA-IR = 2.5 → prior = 0.4 → p3 = 0.4 × P3_REF (resistência ↑)
              HOMA-IR = 0.7 → prior = 1.4 → p3 = 1.4 × P3_REF (sensibilidade elite)

        ② p3_mitochondrial_efficiency_prior (← PPARGC1A):
              GG → 1.15 → Vmax_OXPHOS = 1.15 × Vmax_REF
              GA → 1.00 → Vmax_REF
              AA → 0.85 → 0.85 × Vmax_REF

        ③ p3_basal_glucose_prior (← glucose_fasting_mg_dL → mmol/L):
              Gb_prior em mmol/L → Gb do modelo Bergman

        NaN em qualquer prior → fallback para valor de referência populacional.
        """
        # ① Sensibilidade basal à insulina (p3)
        p3_prior = bayesian_priors.get("p3_insulin_sensitivity_prior", float("nan"))
        p3 = self._P3_REF if math.isnan(p3_prior) else float(self._P3_REF * p3_prior)

        # ② Eficiência mitocondrial (Vmax_OXPHOS ← PPARGC1A)
        mito_prior = bayesian_priors.get("p3_mitochondrial_efficiency_prior", float("nan"))
        Vmax = (
            self._VMAX_OXPHOS_REF if math.isnan(mito_prior)
            else float(self._VMAX_OXPHOS_REF * mito_prior)
        )

        # ③ Glucosa basal (Gb)
        Gb_prior = bayesian_priors.get("p3_basal_glucose_prior", float("nan"))
        Gb = self._GB_REF if math.isnan(Gb_prior) else float(Gb_prior)

        return MetabolicParams(
            # I. Bergman-Cobelli base
            p1=self._P1_REF,
            p2=self._P2_REF,
            p3=p3,
            Gb=Gb,
            Ib=self._IB_REF,
            V_g=self._VG_REF,
            k_abs=self._K_ABS_REF,
            I_peak=self._I_PEAK_REF,
            k_I=self._K_I_REF,
            u_mec=self._U_MEC_REF,
            # I-ext. Roy & Parker + Holloszy
            k_SI_exercise=self._K_SI_EXERCISE_REF,
            P_SI_ref=self._P_SI_REF_W,
            u_mec_post=self._U_MEC_POST_REF,
            k_GLUT4_decay=self._K_GLUT4_DECAY_REF,
            # II. Boillet-Meyer 3-component PCr
            PCr_max=self._PCR_MAX_REF,
            Vmax_OXPHOS=Vmax,
            k_CK=self._K_CK_REF,
            k_glyc_pcr_frac=self._K_GLYC_PCR_FRAC_REF,
            # III. Brooks Lactato
            La_b=self._LA_B_REF,
            P_LT=self._P_LT_REF,
            k_La_prod=self._K_LA_PROD_REF,
            k_La_glyc=self._K_LA_GLYC_REF,
            k_La_clear=self._K_LA_CLEAR_REF,
        )

    def simulate_metabolic_response(
        self,
        bayesian_priors: dict[str, float],
        meal_event: dict,
        session_record: dict,
        t_span_hours: float = 4.0,
        n_save_points: int = 240,
    ) -> dict[str, jax.Array]:
        """
        Simula a dinâmica bioenergética integrada v1.2 (glucosa + PCr + lactato).

        Novidades v1.2 no retorno:
            • "Hub_Lactate_Signalling": La exportado com chave Hub para acoplamento
              multi-módulo (Módulo 9 BDNF, Módulo 3 cardíaco, Módulo 5 HPA).

        Args:
            bayesian_priors:
                Output de compose_bayesian_priors() — dict[str, float] com float("nan")
                para biomarcadores ausentes. Chaves utilizadas:
                    "p3_insulin_sensitivity_prior"     → escala p3  [a.u.]
                    "p3_mitochondrial_efficiency_prior" → escala Vmax_OXPHOS [a.u.]
                    "p3_basal_glucose_prior"            → Gb [mmol·L⁻¹]

            meal_event:
                Campos de HydrationAndNutritionRecord como dict:
                    "carbohydrate_grams"   → float — CHO total da refeição [g]
                    "meal_timestamp_min"   → float — minutos desde t0 [min]

            session_record:
                Campos de SessionRecord como dict:
                    "power_output_watts"    → float — potência média [W]
                    "session_start_min"     → float — minutos desde t0 [min]
                    "session_duration_secs" → float — duração total [s]

            t_span_hours:
                Horizonte de simulação [horas]. Default: 4h (um ciclo prandial).

            n_save_points:
                Número de pontos de saída equidistantes. Default: 240 (1 min / 4h).

        Returns:
            dict com arrays JAX (shape [n_save_points,], dtype float32):
                "t_min"                  → eixo temporal              [min]
                "G_mmol_L"               → glucosa plasmática         [mmol·L⁻¹]
                "X_min_1"                → ação remota da insulina    [min⁻¹]
                "PCr_mmol_L"             → fosfocreatina muscular     [mmol·L⁻¹]
                "La_mmol_L"              → lactato sanguíneo          [mmol·L⁻¹]
                "G_mg_dL"                → glucosa (unidade CGM)      [mg·dL⁻¹]
                "Hub_Lactate_Signalling" → La como variável Hub multi-módulo [mmol·L⁻¹]
        """
        params = self._build_params(bayesian_priors)

        # ── Refeição: converter para escalares JAX float32 ────────────────────
        carb_g = float(meal_event.get("carbohydrate_grams") or 0.0)
        carb_mmol = jnp.float32(carb_g / _GLUCOSE_MW_G_PER_MOL * 1000.0)
        t_meal = jnp.float32(float(meal_event.get("meal_timestamp_min") or 0.0))

        # ── Sessão: converter para escalares JAX float32 ──────────────────────
        power_w = jnp.float32(float(session_record.get("power_output_watts") or 0.0))
        t_sess_start = jnp.float32(float(session_record.get("session_start_min") or 0.0))
        sess_dur_min = jnp.float32(
            float(session_record.get("session_duration_secs") or 0.0) / _MIN_PER_HOUR
        )

        # ── Condições iniciais: estado estacionário basal ─────────────────────
        y0 = jnp.array(
            [params.Gb, 0.0, params.PCr_max, params.La_b],
            dtype=jnp.float32,
        )

        # ── Eixo temporal ─────────────────────────────────────────────────────
        t0 = jnp.float32(0.0)
        t1 = jnp.float32(t_span_hours * _MIN_PER_HOUR)
        ts = jnp.linspace(t0, t1, n_save_points, dtype=jnp.float32)

        # ── Pytree de args para bioenergetic_ode ─────────────────────────────
        args = (params, carb_mmol, power_w, t_meal, t_sess_start, sess_dur_min)

        # ── Invoca o kernel JIT-compilado ─────────────────────────────────────
        solution = _solve_bioenergetic(
            y0=y0,
            t0=t0,
            t1=t1,
            dt0=jnp.float32(0.1),
            ts=ts,
            args=args,
        )

        ys = solution.ys
        La_array = ys[:, 3]

        # ── Hub_Energy_Expenditure_Kcal (Bond Graph acoplamento Módulo 1 → Módulo 5) ─
        # W_mec (J) = P (W) × t (s); W_metab (J) = W_mec / GE; E (kcal) = W_metab / 4184
        # Gross efficiency ajustada pelo prior mitocondrial (PPARGC1A, Coyle 1992).
        mito_prior = bayesian_priors.get("p3_mitochondrial_efficiency_prior", float("nan"))
        gross_eff = (
            self._GROSS_EFF_BASE if math.isnan(mito_prior)
            else self._GROSS_EFF_BASE * float(mito_prior)
        )
        gross_eff = max(gross_eff, 0.10)  # piso físico: nenhum humano < 10% GE
        sess_dur_s = float(session_record.get("session_duration_secs") or 0.0)
        power_w_py = float(session_record.get("power_output_watts") or 0.0)
        mechanical_work_j = power_w_py * sess_dur_s
        energy_kcal = (mechanical_work_j / gross_eff) / self._J_PER_KCAL

        return {
            "t_min":                        ts,
            "G_mmol_L":                     ys[:, 0],
            "X_min_1":                      ys[:, 1],
            "PCr_mmol_L":                   ys[:, 2],
            "La_mmol_L":                    La_array,
            "G_mg_dL":                      ys[:, 0] * _MG_DL_PER_MMOL_L,
            "Hub_Lactate_Signalling":        La_array,
            "Hub_Energy_Expenditure_Kcal":  float(energy_kcal),
        }
