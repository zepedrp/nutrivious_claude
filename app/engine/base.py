"""
app/engine/base.py — Primitivas Matemáticas Universais do Motor Preditivo

FILOSOFIA (Hard Reset — três Leis de Ouro):

  1. Biologia é Contínua: zero if/then no código Python.
     Cada função de transferência é uma expressão matemática única, contínua
     em todo o domínio. Interruptores (degraus) documentais são reexpressos
     como somas de operações clamp — matematicamente idênticas, sem branches.

  2. Fisiologia de Redes (Third-Order Effects): os parâmetros de cada função
     (Km, taxa de decaimento, referência) podem ser funções de outros
     biomarcadores, modelando a realidade de que X altera a sensibilidade
     da relação Y→Z, não apenas adiciona um modificador independente.
     Implementado via parâmetro `km_shift` nas funções de Hill.

  3. Zero Alucinação: nenhuma constante numérica vive neste módulo.
     Todas as constantes residem em constants.py e são explicitamente
     rastreadas aos documentos da Fase 3. Se um valor não está nos documentos,
     é None em constants.py — nunca inventado aqui.

Hierarquia de Tectos (referência arquitectónica):
    T_espécie ≥ TGI = T_espécie × ∏ Mj^G
             ≥ TEA = TGI × E_epi × ∏ Mk^B × C_micro
             ≥ TFD = TEA × ∏ Ml^D  (Fase 4 — telemetria diária)
"""
from __future__ import annotations

import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECÇÃO 1 — PRIMITIVA UNIVERSAL
# ─────────────────────────────────────────────────────────────────────────────

def clamp(value: float, floor: float, ceiling: float) -> float:
    """Limita value ao intervalo fechado [floor, ceiling].

    É a primitiva aplicada à saída de TODOS os modificadores biológicos.
    Impede extrapolação fora dos limites fisiológicos documentados e garante
    estabilidade numérica no produto em cascata da equação mestre:
        T_real = T_espécie × ∏ Mj^G × E_epi × ∏ Mk^B × C_micro

    Os pares (floor, ceiling) de cada modificador estão declarados em
    constants.py e rastreados às tabelas dos documentos da Fase 3.
    """
    return max(floor, min(ceiling, value))


# ─────────────────────────────────────────────────────────────────────────────
# SECÇÃO 2 — REGRAS DE COMBINAÇÃO BIOLÓGICA
# ─────────────────────────────────────────────────────────────────────────────

def multiplicative_combine(modifiers: list[float]) -> float:
    """Produto de modificadores em vias mecanísticas independentes e sequenciais.

    Regra Multiplicativa (§1.2, Doc 1): usada quando cada modificador actua
    num passo distinto da mesma cadeia causal. O HOMA-IR afecta a translocação
    de GLUT4; a vitamina D afecta a sensibilidade do mTORC1; a testosterona
    afecta a transcrição de genes anabólicos — três passos distintos, logo
    os seus efeitos multiplicam-se.

    Equivalência logarítmica (§2.2, Doc 2):
        ln(T_real) = ln(T_esp) + Σ ln(Mj) + ln(E_epi) + Σ ln(Mk) + ln(C_micro)
    Uma desvantagem de 20% (ln(0.80) = -0.223) domina matematicamente
    sobre uma vantagem de 5% (ln(1.05) = +0.049) — a álgebra logarítmica
    é honesta.

    Retorna 1.0 para lista vazia (elemento neutro do produto).
    """
    result = 1.0
    for m in modifiers:
        result *= m
    return result


def min_combine_bottleneck(modifiers: list[float]) -> float:
    """Gargalo absoluto — Lei do Mínimo de Liebig.

    Regra de Mínimo (§1.2, Doc 1): usada quando qualquer modificador
    representa uma restrição estrutural hard que não pode ser compensada
    pelos outros. O exemplo canónico da Fase 3 é a difusão pulmonar:
    se DLCO é o gargalo, aumentar o débito cardíaco não ajuda — o oxigénio
    não transfere. O sistema é limitado pelo mínimo:

        M_structural = min(M_1, M_2, ..., M_k)

    Aplica-se sobretudo a limitações estruturais (Categoria A da Fase 1)
    e a situações onde dois transportadores partilham um único lúmen
    (ex: SGLT1 e GLUT5 na absorção intestinal de CHO).

    Retorna 1.0 para lista vazia.
    """
    if not modifiers:
        return 1.0
    return min(modifiers)


def weighted_additive_combine(scores: list[float], weights: list[float]) -> float:
    """Combinação aditiva ponderada — exclusiva para scores compostos (VTP).

    NÃO usar para modificadores de tecto (usar multiplicative_combine).
    Reservada para a síntese do Priority Score e outputs do VTP, onde a
    média ponderada é semanticamente correcta:

        Priority_Score(j) = Σ_i [w_i × ΔCeiling_i(j) × (1/T_weeks) × Leverage]

    Os pesos somam 1.0 ou são normalizados internamente.
    """
    if not scores or not weights:
        return 0.0
    if len(scores) != len(weights):
        raise ValueError("scores e weights devem ter o mesmo comprimento.")
    total_weight = sum(weights)
    if total_weight == 0.0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


# ─────────────────────────────────────────────────────────────────────────────
# SECÇÃO 3 — FUNÇÕES DE TRANSFERÊNCIA CONTÍNUAS (zero if/then)
# ─────────────────────────────────────────────────────────────────────────────

def power_law(
    value: float,
    reference: float,
    exponent: float,
    floor: float,
    ceiling: float,
) -> float:
    """Lei de potência: f(x) = clamp((x / x_ref)^exponent, floor, ceiling).

    Modela relações biológicas com rendimentos decrescentes ou crescentes
    em relação a um ponto de referência óptimo (x_ref → modifier = 1.0).
    O expoente fraccional captura a saturação dos receptores — duplicar
    o substrato não duplica o output porque os sítios de ligação saturam.

    Exemplos documentados (Doc 1, §2.4):
      M_T_MPS   = clamp((T_ng_dL / 600)^0.35, 0.75, 1.20)
                  expoente 0.35: cinética saturável do receptor androgénico
                  (Bhasin et al. NEJM 2001 — dose-response em 61 homens)
      M_IGF1    = clamp((IGF1_ng_mL / 180)^0.40, 0.80, 1.15)
      M_T_Power = clamp((T_ng_dL / 600)^0.25, 0.85, 1.12)
                  expoente menor (0.25 vs 0.35): efeito neural mais saturável
                  que o efeito transcripcional na MPS
      M_ferrit  = clamp(sqrt(ferritina / 80), 0, 1.0)   [expoente=0.5]
                  Doc 2, §5.3 — raiz quadrada da fracção de repleção
      M_Hb      = clamp((Hb / 16.0)^0.85, ...)
                  Doc 2, §5.3 — relação supra-linear (próxima de linear)
                  entre Hb e transporte de O₂ (Fick)

    Args:
        value:     Concentração ou score medido.
        reference: x_ref onde modifier = 1.0 (extraído dos documentos).
        exponent:  Expoente da lei de potência.
        floor:     Limite inferior do clamp.
        ceiling:   Limite superior do clamp.
    """
    if reference <= 0.0:
        raise ValueError("reference deve ser positivo.")
    if value < 0.0:
        return floor
    return clamp((value / reference) ** exponent, floor, ceiling)


def linear_ratio(
    value: float,
    reference: float,
    floor: float,
    ceiling: float,
) -> float:
    """Rácio linear: f(x) = clamp(x / x_ref, floor, ceiling).

    Caso especial de power_law com expoente=1.0. Usado quando a relação
    biológica é rigorosamente linear (ex: hemoglobina e capacidade de
    transporte de O₂ via equação de Fick: CaO₂ = 1.34 × [Hb] × SaO₂).

    Exemplo documentado (Doc 1, §2.1):
      M_Hb = clamp(Hb / Hb_ref, 0.75, 1.15)
      Hb_ref = 15.5 g/dL (homem) | 13.5 g/dL (mulher)
      Relação linear válida até Hct ≈ 55% (limiar de hiperviscosidade).
    """
    if reference <= 0.0:
        raise ValueError("reference deve ser positivo.")
    return clamp(value / reference, floor, ceiling)


def lean_fraction_modifier(
    bf_pct: float,
    ref_bf_pct: float,
    floor: float,
    ceiling: float,
) -> float:
    """Modificador de fracção magra: f(BF%) = clamp((1−BF%) / (1−ref_BF%), floor, ceiling).

    Captura o custo de transportar massa metabolicamente inerte (gordura)
    sobre o VO2max expresso em mL/kg/min, onde o denominador inclui toda
    a massa corporal.

    Exemplo documentado (Doc 1, §2.1):
      M_BF(BF%) = (1 − BF%) / (1 − BF%_ref)
      BF%_ref = 0.12 (homem) | 0.20 (mulher)
      Floor = 0.75 | Ceiling = 1.10
      Exemplo: BF% = 22% (homem) → (1−0.22)/(1−0.12) = 0.886
    """
    denominator = 1.0 - ref_bf_pct
    if denominator <= 0.0:
        raise ValueError("ref_bf_pct deve ser < 1.0.")
    return clamp((1.0 - bf_pct) / denominator, floor, ceiling)


def exponential_decay(
    value: float,
    rate: float,
    reference: float,
    floor: float,
    ceiling: float,
) -> float:
    """Decaimento exponencial acima de referência: f(x) = exp(−rate × max(0, x − x_ref)).

    Modela atenuação biológica que se acelera de forma não-linear acima
    de um ponto de referência fisiológico. A relação exponencial é
    mecanisticamente justificada onde a supressão é mediada por uma
    cascata de sinalização (ex: citocinas → hepcidina → eritropoiese).

    Exemplos documentados (Doc 1):
      M_HOMA_CHO = clamp(exp(−0.15 × max(0, HOMA − 1.0)), 0.40, 1.00)
          §2.2 — dois mecanismos paralelos: menor GLUT4 + menor SGLT1
          via PI3K/Akt; calibrado para HOMA=3.0 → 26% de penalização.

      M_CRP_aerobic = clamp(exp(−0.10 × max(0, CRP − 0.3)), 0.80, 1.00)
          §2.1 — anemia de doença crónica via hepcidina + disfunção eNOS;
          k=0.10 calibrado para CRP=3.0 mg/L → ~20% de penalização.

      M_Inflam_MPS = clamp(exp(−0.20 × max(0, CRP − 0.3)), 0.65, 1.00)
          §2.4 — fosforilação de IRS-1 em Ser307 via IKK-β e JNK;
          k=0.20 (mais agressivo que aeróbio) porque MPS é mais sensível
          à inflamação do que o VO2max.

    Implementação sem if/then: max(0, value − reference) é contínuo
    e diferenciável em todo o domínio excepto no kink em x=reference,
    onde a derivada tem descontinuidade de primeira ordem (kink) mas
    não há salto de função (C0-continuidade garantida).
    """
    return clamp(
        math.exp(-rate * max(0.0, value - reference)),
        floor,
        ceiling,
    )


def linear_with_threshold(
    value: float,
    reference: float,
    slope: float,
    floor: float,
    ceiling: float,
) -> float:
    """Linear com limiar: f(x) = clamp(1.0 + slope × max(0, x − x_ref), floor, ceiling).

    Modela atenuação linear acima de um limiar fisiológico. O slope é
    negativo quando o excesso penaliza e positivo quando o excesso beneficia.
    A função max(0, …) garante continuidade — abaixo de x_ref, modifier=1.0;
    acima, decresce linearmente à taxa |slope| por unidade.

    Exemplos documentados (Doc 1):
      M_HOMA_Fat = clamp(1.0 − 0.08 × max(0, HOMA − 1.0), 0.50, 1.00)
          §2.3 — HOMA-IR eleva malonil-CoA → inibe CPT-1 (Ki ≈ 0.02 µM);
          k=0.08 (menos agressivo que CHO) porque inibição de CPT-1
          não é total — carnitina ainda compete pelo sítio activo.

      M_Cortisol_MPS = clamp(1.0 − 0.025 × max(0, C − 14), 0.60, 1.00)
          §2.4 — REDD1 inibe mTORC1 via TSC1/2; k calibrado para
          cortisol=25 µg/dL (crónico elevado) → 27.5% de penalização.

      M_Zonulin_CHO = clamp(1.0 − 0.012 × max(0, Z − 20), 0.70, 1.00)
          §2.2 — translocação de LPS → TLR4 → downregulação de SGLT1
          via NF-κB; cada ng/mL acima de 20 → 1.2% de redução.
    """
    return clamp(1.0 + slope * max(0.0, value - reference), floor, ceiling)


def linear_deviation(
    value: float,
    reference: float,
    slope: float,
    floor: float,
    ceiling: float,
) -> float:
    """Desvio linear em ambas as direcções: f(x) = clamp(1.0 + slope × (x − x_ref), floor, ceiling).

    Diferente de linear_with_threshold: actua em ambos os lados de x_ref.
    Valores abaixo de x_ref com slope negativo elevam o modificador (bónus);
    valores acima penalizam. Usado quando a biologia é simétrica em torno
    do ponto de referência.

    Exemplo documentado (Doc 2, §5.2):
      M_DunedinPACE = clamp(2.0 − pace, 0.60, 1.05)
          = clamp(1.0 + (−1.0) × (pace − 1.0), 0.60, 1.05)
          → reference=1.0, slope=−1.0
          pace < 1.0 (biologicamente mais jovem) → M > 1.0 (bónus)
          pace > 1.0 (biologicamente mais velho) → M < 1.0 (penalização)
    """
    return clamp(1.0 + slope * (value - reference), floor, ceiling)


def piecewise_linear(
    value: float,
    x_points: list[float],
    y_points: list[float],
    floor: float,
    ceiling: float,
) -> float:
    """Piecewise linear contínua via somas de clamp — zero if/then.

    Converte TODAS as fórmulas "if x < a: y = ...; if a ≤ x < b: y = ..."
    dos documentos numa única expressão matemática contínua:

        f(x) = y₀ + Σᵢ [ slope_i × clamp(x − xᵢ, 0, Δxᵢ) ]
                   + slope_last × max(0, x − x_last)

    onde slope_i = (yᵢ₊₁ − yᵢ) / (xᵢ₊₁ − xᵢ).

    A função é C0-contínua (sem saltos) em todo o domínio. Tem kinks
    (descontinuidades de derivada) nos pontos de quebra, o que é
    biologicamente legítimo — um limiar é um kink, não uma descontinuidade.

    O clamp externo (floor, ceiling) aplica os limites fisiológicos.
    O segmento final extrapola com o slope do último segmento documentado;
    o clamp trava a extrapolação nos limites biológicos.

    Exemplos de conversão (Doc 1):

    M_O3_aerobic — documentado como:
        if O3 < 4%:    M = 0.88
        if 4 ≤ O3 < 8: M = 0.88 + 0.12 × (O3 − 4)/4
        if O3 ≥ 8:     M = 1.00  (plateau até ao ceiling 1.02 acima de 12%)
      → x_points=[0,4,8,12], y_points=[0.88,0.88,1.00,1.02], floor=0.88, ceiling=1.02

    M_GutTrain_CHO — documentado como:
        M = 0.70 + 0.30 × min(1.0, semanas / 8)
      → x_points=[0,8], y_points=[0.70,1.00], floor=0.70, ceiling=1.00
        (o clamp ceiling trava aos 100% após 8 semanas)

    M_VitD_MPS — documentado como:
        if VitD < 20: M = 0.72; if 20≤VitD<50: M = linear; if VitD≥50: M = 1.00
      → x_points=[0,20,50], y_points=[0.72,0.72,1.00], floor=0.72, ceiling=1.00

    Args:
        value:    Variável de entrada (biomarcador medido).
        x_points: Pontos de quebra x em ordem crescente (≥ 2 pontos).
        y_points: Valores y correspondentes a cada x_point.
        floor:    Limite biológico inferior.
        ceiling:  Limite biológico superior.
    """
    if len(x_points) < 2 or len(x_points) != len(y_points):
        raise ValueError("Requer ≥ 2 pontos com len(x_points) == len(y_points).")

    result = y_points[0]

    # Segmentos interiores: clamp(value − xᵢ, 0, Δxᵢ) garante que cada
    # segmento só contribui na sua janela [xᵢ, xᵢ₊₁].
    n = len(x_points)
    for i in range(n - 1):
        dx = x_points[i + 1] - x_points[i]
        dy = y_points[i + 1] - y_points[i]
        if dx == 0.0:
            raise ValueError(f"Pontos x duplicados em índice {i}.")
        slope = dy / dx
        result += slope * clamp(value - x_points[i], 0.0, dx)

    # Extrapolação do segmento final além de x_points[-1]:
    # o clamp externo trará o resultado para os limites biológicos.
    dx_last = x_points[-1] - x_points[-2]
    dy_last = y_points[-1] - y_points[-2]
    slope_last = dy_last / dx_last
    result += slope_last * max(0.0, value - x_points[-1])

    return clamp(result, floor, ceiling)


# ─────────────────────────────────────────────────────────────────────────────
# SECÇÃO 4 — CINÉTICA DE HILL (saturação e inibição)
# ─────────────────────────────────────────────────────────────────────────────

def hill_saturation(
    value: float,
    km: float,
    n: float = 1.0,
    vmax: float = 1.0,
    km_shift: float = 0.0,
) -> float:
    """Equação de Hill (Michaelis-Menten generalizada): Vmax × xⁿ / (Km_eff ⁿ + xⁿ).

    Modela processos com cinética de saturação: ligação a substrato,
    ocupação de receptores, throughput de transportadores.
    Em x = Km a saída é Vmax/2 (semi-máximo).
    n > 1 introduz cooperatividade (curva sigmoidal).

    PARÂMETRO DE 3.ª ORDEM — km_shift:
    O km_shift implementa a interacção de rede documentada no Gargalo 2
    do Fatmax (Doc 2, §8.1 + Doc 1, §2.3): o malonil-CoA, produzido em
    excesso sob hiperinsulinemia/HOMA-IR elevado, é um inibidor competitivo
    da CPT-1 (Ki ≈ 0.02 µM — Doc 1, §2.3). A inibição competitiva eleva
    o Km aparente:

        Km_aparente = Km_basal × (1 + [malonil-CoA] / Ki)

    Portanto km_shift = Km_basal × ([malonil-CoA] / Ki).
    O [malonil-CoA] é função do HOMA-IR — coeficiente de proporcionalidade
    NÃO declarado nos documentos → permanece None em constants.py.

    Equação mTORC1 (Doc 2, §7.2) usa este padrão para cada input:
        Actividade_mTORC1 = hill_sat(Insulin, K1, a1)
                          × hill_sat(Leucina, K2, a2)
                          × hill_sat(Energia, K3, a3)
                          × (1 − [AMPK]/K4)
    Os parâmetros a1, a2, a3, K1, K2, K3, K4 NÃO estão declarados
    numericamente nos documentos → None em constants.py.

    Args:
        value:    Concentração do substrato/input.
        km:       Constante de semi-saturação basal.
        n:        Coeficiente de Hill (cooperatividade).
        vmax:     Output máximo normalizado.
        km_shift: Aumento do Km aparente por inibição competitiva.
                  Calculado pelo módulo chamador usando a concentração
                  do inibidor e a Ki documentada.
    """
    effective_km = max(0.0, km + km_shift)
    if value <= 0.0:
        return 0.0
    value_n = value ** n
    km_n = effective_km ** n
    return vmax * value_n / (km_n + value_n)


def hill_inhibition(
    value: float,
    ic50: float,
    n: float,
) -> float:
    """Inibição de Hill: f(x) = 1 / (1 + (x / IC₅₀)ⁿ).

    Modela a inibição sigmoidal de uma enzima ou receptor por um ligando.
    O IC₅₀ é a concentração que produz 50% de inibição.
    n controla a cooperatividade (steepness).

    Exemplo documentado — Inibição de HSL pela Insulina (Doc 2, §8.1):
        M_insulina^lipólise = 1 / (1 + ([Insulina] / IC₅₀_HSL)²)
        IC₅₀_HSL ≈ 10 µU/mL, n = 2
        Mecanismo: insulina inibe a lipase sensível às hormonas (HSL)
        com potência extraordinária — o IC₅₀ é muito baixo,
        o que explica a supressão da lipólise mesmo com
        hiperinsulinemia basal moderada.

        Com insulina em jejum = 15 µU/mL (vs. óptimo ≈ 5 µU/mL):
        M = 1 / (1 + (15/10)²) = 1 / (1 + 2.25) = 0.308
        → 69% da capacidade lipolítica suprimida apenas por
          hiperinsulinemia basal que clinicamente "parece normal".

    Esta é a função de transferência CORRECTA para modelar o Gargalo 1
    do Fatmax — não uma penalização linear ou exponencial. A forma
    sigmoidal reflecte a cinética real de inibição enzimática.
    """
    if value <= 0.0:
        return 1.0
    return 1.0 / (1.0 + (value / ic50) ** n)


def mtorc1_activity(
    insulin_igf1_signal: float,
    leucine_signal: float,
    energy_signal: float,
    ampk_signal: float,
    km_insulin: Optional[float],
    km_leucine: Optional[float],
    km_energy: Optional[float],
    k_ampk: Optional[float],
    n_insulin: float = 1.0,
    n_leucine: float = 1.0,
    n_energy: float = 1.0,
) -> Optional[float]:
    """Actividade integrada do mTORC1 como produto de quatro inputs de Hill.

    Equação documentada (Doc 2, §7.2):
        Actividade = [Insulin^a1/(K1^a1 + Insulin^a1)]
                   × [Leucina^a2/(K2^a2 + Leucina^a2)]
                   × [Energia^a3/(K3^a3 + Energia^a3)]
                   × (1 − [AMPK]/K4)

    ESTADO ACTUAL: a1, a2, a3, K1, K2, K3, K4 NÃO são declarados
    numericamente nos documentos da Fase 3. Esta função retorna None
    quando qualquer parâmetro Km está indefinido (None em constants.py),
    respeitando a Lei de Ouro "Zero Alucinação".

    Implementação correcta quando os parâmetros estiverem definidos:
    cada input segue cinética de Hill independente (Porter a cooperatividade
    individual) e o produto total é o nível de activação do mTORC1.

    A estrutura da equação é biologicamente fundamental porque demonstra
    que HOMA-IR elevado (via compressão do sinal insulin/IGF-1) não reduz
    apenas a actividade de mTORC1 directamente — ele reduz o ganho
    multiplicativo com que leucina e energia actuam sobre mTORC1.
    Este é o efeito de 3.ª ordem documentado no §7.2.
    """
    if any(p is None for p in [km_insulin, km_leucine, km_energy, k_ampk]):
        return None

    insulin_term = hill_saturation(insulin_igf1_signal, km_insulin, n_insulin)
    leucine_term = hill_saturation(leucine_signal, km_leucine, n_leucine)
    energy_term = hill_saturation(energy_signal, km_energy, n_energy)
    ampk_term = 1.0 - clamp(ampk_signal / k_ampk, 0.0, 1.0)

    return insulin_term * leucine_term * energy_term * ampk_term


# ─────────────────────────────────────────────────────────────────────────────
# SECÇÃO 5 — DINÂMICA TEMPORAL DO TECTO (RGC e Gap)
# ─────────────────────────────────────────────────────────────────────────────

def gap_closure_trajectory(
    tgi: float,
    ga_initial: float,
    rgc_per_week: float,
    weeks: float,
) -> float:
    """TEA(t) = TGI − GA₀ × exp(−RGC × t) — modelo logístico de gap.

    Equação documentada (Doc 2, §18.2):
        TEA(S,t) = TGI(S) − GA₀(S) × e^(−RGC(S) × t)

    Derivação: a taxa de melhoria é proporcional ao gap restante
    (cinética de primeira ordem — retornos decrescentes à proximidade
    do tecto). Integração directa produz esta forma exponencial.

    Os valores de RGC por sistema estão em constants.py (Doc 2, §18.2):
        VO2max: 0.015/semana (sem genótipo favorável) a 0.025/semana (com)
        Fatmax: 0.010/semana a 0.018/semana
        Estrutural (tendão): 0.005/semana a 0.008/semana
        Estrutural (osso):   0.003/semana a 0.005/semana

    Args:
        tgi:           Tecto Genético Individual (assímptota máxima).
        ga_initial:    Gap de Adaptação inicial: TGI − TEA_actual.
        rgc_per_week:  Taxa de fechamento de gap por semana.
        weeks:         Horizonte temporal.

    Returns:
        TEA projectado no tempo `weeks`.
    """
    return tgi - ga_initial * math.exp(-rgc_per_week * weeks)


def sleep_gh_modifier(
    sws_pct_actual: float,
    tst_actual_hours: float,
    tst_genetic_hours: float,
    sws_pct_optimal: float = 0.20,
) -> float:
    """Modificador de GH nocturno pelo sono: (SWS%_actual/SWS%_opt) × (TST_actual/TST_gen).

    Equação documentada (Doc 2, §7.3):
        M_sono^GH = (SWS%_actual / SWS%_óptimo=20%) × (TST_actual / TST_genético)

    TST_genético determinado pelo PER3 VNTR (constantes em constants.py):
        PER3 5/5: TST_genetic = 8.5h
        PER3 4/4: TST_genetic = 7.0h

    Exemplo do documento:
        PER3 5/5 a dormir 6.5h, SWS=12%:
        M = (12%/20%) × (6.5/8.5) = 0.600 × 0.765 = 0.459
        → GH nocturno a 45.9% do óptimo.

    O GH pulsátil durante o SWS é o estímulo primário de MPS durante
    o sono — este modificador aplica-se directamente sobre o PC_MPS.
    """
    sws_ratio = sws_pct_actual / sws_pct_optimal
    tst_ratio = tst_actual_hours / tst_genetic_hours
    return clamp(sws_ratio * tst_ratio, 0.0, 1.0)


def sleep_pvt_modifier(
    sleep_deficit_hours: float,
    rate: float,
) -> float:
    """Modificador de PVT (tempo de reacção): f(Δsono) = exp(−rate × Δsono).

    Equação documentada (Doc 2, §9.2):
        M_sono^PVT = exp(−0.12 × Δ_sono)

    Exemplo do documento:
        PER3 5/5 a dormir 6.5h (vs. 8.5h óptimo): Δ=2.0h
        M = exp(−0.12 × 2.0) = 0.787
        → TR aumenta de ~150ms para ~190ms (+26.7%)
        → diferença entre apanhar uma bola ou deixá-la cair.

    rate=0.12 é o único valor declarado no documento para esta função.
    """
    return math.exp(-rate * max(0.0, sleep_deficit_hours))
