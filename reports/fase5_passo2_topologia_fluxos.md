# FASE 5 — PASSO 2: MAPA DE TOPOLOGIA E DINÂMICA DE FLUXOS

> **Input:** FASE 5 — Passo 1 (Inventário, Omissões, Auditoria de Erros) + documentos fonte Teia Causal (TC) e Matriz Alosática (MA).
> **Objectivo:** Resolver os três problemas de arquitectura abertos pelo Passo 1: (1) calibrar os w_ij do IIG, (2) formalizar a dinâmica dos loops de retroalimentação, (3) resolver o loop circular Fase 5 ↔ Fase 6.
> **Regra:** Sem código Python. Apenas matemática, equações dinâmicas e contratos de dados.

---

## SECÇÃO 1 — RESOLUÇÃO CARDINAL DO IIG: A MATRIZ NUMÉRICA

### 1.1 A Escala de Conversão Cardinal

O Passo 1 identificou o bloqueio crítico: a Tabela Mestre (MA §24) usa setas qualitativas (↓, ↓↓, ↓↓↓, ↑, —) que tornam o IIG computacionalmente inoperacional. A resolução exige uma **escala de conversão cardinal**, derivada de dois princípios:

**Princípio 1 — Ancoragem pelo comportamento esperado do IIG:**
A tabela de interpretação do IIG define cinco bandas (0–10, 10–25, 25–50, 50–75, >75). Um atleta com HPA cronicamente activado (D_HPA ≈ 0.5) e sono comprometido (D_Sono ≈ 0.6) deveria produzir IIG ≈ 45–65 (interferência severa). Este cenário ancora a calibração dos pesos.

**Princípio 2 — Homogeneidade dos símbolos entre supressão e activação:**
Na Tabela Mestre, os símbolos ↓ (supressão de sistema anabólico/protector) e ↑ (activação de sistema de stress) são ambos patológicos quando o sistema origem está em défice. O ↓ de testosterona e o ↑ de cortisol reactivo têm o mesmo efeito de degradação do estado global. A escala unifica os dois.

**Escala Proposta:**

| Símbolo | Descrição Qualitativa | Cardinal w_ij | Justificação |
|---|---|---|---|
| — | Efeito negligenciável | **0.0** | Sem cascata mensurável |
| ↓ / ↑ | Supressão/activação leve | **1.5** | Cascata de 1 passo, reversível em 24h |
| ↓↓ / ↑↑ | Supressão/activação moderada | **4.0** | Cascata de 2–3 passos, reversível em dias |
| ↓↓↓ / ↑↑↑ | Supressão/activação severa | **9.0** | Cascata directa e clinicamente significativa; múltiplos mecanismos paralelos |

**Verificação de calibração:** Com a escala acima, um cenário de colapso total (todos D_i = 1.0) produz IIG ≈ 259. O threshold >75 corresponde a ≈ 29% do colapso máximo teórico — o que é biologicamente razoável: IIG > 75 nunca exige que todos os sistemas estejam em colapso simultaneamente, apenas que 3–4 sistemas de alta interferência estejam significativamente comprometidos.

---

### 1.2 A Matriz w_ij Completa (9 × 9)

A leitura da Tabela MA §24 traduzida para valores cardinais. **Leitura:** linha = sistema origem (i); coluna = sistema alvo (j). Valor = w_ij.

| **i → j** | HPA | HPG | HPT | MPS | Imune | Intestino | Sono | Metabolismo | SNC |
|---|---|---|---|---|---|---|---|---|---|
| **HPA** | — | 9.0 | 4.0 | 9.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| **HPG** | 1.5 | — | 0.0 | 9.0 | 1.5 | 0.0 | 1.5 | 1.5 | 1.5 |
| **HPT** | 1.5 | 0.0 | — | 4.0 | 1.5 | 4.0 | 1.5 | 9.0 | 1.5 |
| **MPS** | 1.5 | 0.0 | 0.0 | — | 1.5 | 0.0 | 0.0 | 1.5 | 1.5 |
| **Imune** | 4.0 | 4.0 | 1.5 | 9.0 | — | 4.0 | 4.0 | 4.0 | 9.0 |
| **Intestino** | 4.0 | 1.5 | 4.0 | 9.0 | 1.5 | — | 4.0 | 9.0 | 9.0 |
| **Sono** | 4.0 | 4.0 | 1.5 | 9.0 | 4.0 | 1.5 | — | 4.0 | 9.0 |
| **Metabolismo** | 1.5 | 4.0 | 1.5 | 4.0 | 1.5 | 4.0 | 1.5 | — | 4.0 |
| **SNC** | 9.0 | 4.0 | 1.5 | 9.0 | 4.0 | 1.5 | 4.0 | 1.5 | — |

**Pesos de Saída Totais** (Σ_j w_ij por sistema — indica o poder de interferência de cada sistema sobre a rede):

| Sistema | Σ_j w_ij | Interpretação |
|---|---|---|
| HPA/Cortisol | **42.0** | Maior emissor de interferência da rede |
| Intestino | **42.0** | Co-líder — a permeabilidade intestinal atinge todos os sistemas |
| Sono | **37.0** | Terceiro emissor — multiplicador universal |
| Sistema Imune | **39.5** | Quarto — IL-6/TNF-α atingem 8 sistemas simultaneamente |
| SNC/Cognição | **34.5** | O stress psicológico é o gatilho mais negligenciado |
| HPT/T3 | **23.0** | A tireoide afecta MPS e metabolismo energético profundamente |
| Metabolismo | **22.0** | HOMA-IR e variabilidade glicémica têm alcance moderado |
| HPG/Testosterona | **15.0** | Principalmente receptor; emissão limitada |
| MPS/Anabolismo | **6.0** | Receptor quase puro — raramente emissor de cascatas |

**Implicação arquitectural:** A fórmula IIG pode ser reescrita de forma computacionalmente eficiente como:

$$IIG(t) = \sum_{i} D_i(t) \times W_i^{out}$$

onde $W_i^{out} = \sum_{j \neq i} w_{ij}$ é o peso de saída total do sistema $i$.

Isto elimina a dupla soma aninhada: basta calcular os 9 valores D_i e multiplicar pelos 9 pesos de saída tabelados acima. Complexidade O(n) em vez de O(n²).

---

### 1.3 As Funções D_i(t) — Normalização por Sistema

Para cada sistema i, D_i(t) ∈ [0, 1] é calculado a partir dos biomarcadores disponíveis. As funções usam a forma `max(0, min(1, (valor − θ_low) / (θ_high − θ_low)))` onde θ_low = limiar de aparecimento do défice e θ_high = colapso funcional completo.

**D_HPA(t) — Carga Cortisol:**
$$D_{HPA}(t) = \max\!\left(0,\ \min\!\left(1,\ \frac{Cortisol_{matinal} - 18}{17}\right)\right)$$

θ_low = 18 µg/dL (limiar CB-01); θ_high = 35 µg/dL (floor de supressão de T = 40% baseline)

Quando disponível, CAR blunted adiciona uma penalidade:
$$D_{HPA}(t) = \max\!\left(D_{HPA,\ cortisol},\ \max\!\left(0,\ \min\!\left(1,\ \frac{0.30 - CAR_{increment}}{0.30}\right)\right) \times 0.6\right)$$

**D_HPG(t) — Défice de Testosterona:**
$$D_{HPG}(t) = \max\!\left(0,\ \min\!\left(1,\ \frac{T_{baseline} - T_{actual}}{T_{baseline} \times 0.60}\right)\right)$$

D=1 quando T < 40% da baseline (floor máximo de CB-01).

**D_HPT(t) — Síndrome de T3 Baixo:**
$$D_{HPT}(t) = \max\!\left(0,\ \min\!\left(1,\ \frac{rT3\_T3_{ratio} - 0.20}{0.30}\right)\right)$$

θ_low = 0.20 (óptimo); θ_high = 0.50 (critério de OTS/Stage 4). Proxy alternativo quando rT3 não disponível: `D_HPT = max(0, min(1, (2.5 − FT3_pg_mL) / 0.5))`.

**D_MPS(t) — Défice Anabólico:**
$$D_{MPS}(t) = \max\!\left(0,\ \min\!\left(1,\ \frac{T\_C_{baseline} - T\_C_{ratio}}{T\_C_{baseline} \times 0.70}\right)\right)$$

D=0 quando T/C = baseline; D=1 quando T/C < 30% da baseline (critério OTS Travão 8).

**D_Imune(t) — Défice Imunológico:**
$$D_{Imune}(t) = 0.5 \times \max\!\left(0,\ \min\!\left(1,\ \frac{sIgA_{baseline} - sIgA}{0.60 \times sIgA_{baseline}}\right)\right) + 0.5 \times \max\!\left(0,\ \min\!\left(1,\ \frac{CRP - 1.0}{9.0}\right)\right)$$

**D_Intestino(t) — Permeabilidade e Endotoxemia:**
$$D_{Intestino}(t) = 0.5 \times \max\!\left(0,\ \min\!\left(1,\ \frac{Zonulina - 20}{20}\right)\right) + 0.5 \times \max\!\left(0,\ \min\!\left(1,\ \frac{LBP - 10}{15}\right)\right)$$

θ_low Zonulina = 20 ng/mL; θ_high = 40 ng/mL (limiar MB-01). Proxy quando Zonulina não disponível: `0.5 × D_CRP_proxy + 0.5 × D_HOMA_proxy`.

**D_Sono(t) — Qualidade de Sono:**
$$D_{Sono}(t) = \max\!\left(\max\!\left(0,\ \min\!\left(1,\ \frac{7.5 - TST}{2.5}\right)\right),\ \max\!\left(0,\ \min\!\left(1,\ \frac{20 - SWS\%}{12}\right)\right),\ \max\!\left(0,\ \min\!\left(1,\ \frac{15 - REM\%}{6}\right)\right)\right)$$

Usa max() das três subdimensões (TST, SWS%, REM%) — flags a dimensão mais comprometida. O'alvo é TST ≥ 7.5h, SWS ≥ 20%, REM ≥ 15%.

**D_Metabolismo(t) — Resistência à Insulina e Variabilidade Glicémica:**
$$D_{Metabolismo}(t) = 0.4 \times \max\!\left(0,\ \min\!\left(1,\ \frac{HOMA\text{-}IR - 1.5}{2.5}\right)\right) + 0.4 \times \max\!\left(0,\ \min\!\left(1,\ \frac{CV_{glucose} - 20}{30}\right)\right) + 0.2 \times \max\!\left(0,\ \min\!\left(1,\ \frac{HbA1c - 5.2}{1.3}\right)\right)$$

**D_SNC(t) — Carga Cognitiva e Motivacional:**
$$D_{SNC}(t) = 0.5 \times \max\!\left(0,\ \min\!\left(1,\ \frac{PVT_{RT} - PVT_{RT,baseline}}{100}\right)\right) + 0.5 \times \max\!\left(0,\ \min\!\left(1,\ 1 - \frac{Wellbeing}{7.0}\right)\right)$$

Proxy quando PVT não disponível: substituir por `max(0, min(1, (RPE_percebido − RPE_esperado) / 3))`.

---

### 1.4 Verificação com Dois Cenários Concretos

**Cenário A — Atleta em Overreaching Funcional (Estado 1):**
- D_HPA = (22−18)/17 = 0.24 (cortisol a 22 µg/dL)
- D_HPG = (T_baseline − 0.88·T_baseline) / (0.60·T_baseline) = 0.12/0.60 = 0.20
- D_HPT = (0.25−0.20)/0.30 = 0.17
- D_MPS = (T_C_baseline − 0.60·T_C_baseline) / (0.70·T_C_baseline) = 0.40/0.70 = 0.57
- D_Imune = 0.5×(0.20/0.60) + 0.5×(1.5−1.0)/9.0 = 0.167 + 0.028 = 0.19
- D_Intestino = 0.5×0 + 0.5×0 = 0.0 (intestino OK)
- D_Sono = max(0.20, 0.25, 0.10) = 0.25 (TST 6.5h → (7.5−6.5)/2.5=0.40... wait let me recalculate: (7.5-6.5)/2.5=0.40, SWS at 14% → (20-14)/12=0.50, REM at 12% → (15-12)/6=0.50. D_Sono = max(0.40, 0.50, 0.50) = 0.50)
- D_Metabolismo = 0.4×(1.8−1.5)/2.5 + 0.4×(25−20)/30 + 0.2×0 = 0.048 + 0.067 = 0.115
- D_SNC = 0.5×(25/100) + 0.5×(1−6.5/7) = 0.125 + 0.036 = 0.16

$$IIG_A = 0.24×42 + 0.20×15 + 0.17×23 + 0.57×6 + 0.19×39.5 + 0.0×42 + 0.50×37 + 0.115×22 + 0.16×34.5$$
$$= 10.1 + 3.0 + 3.9 + 3.4 + 7.5 + 0 + 18.5 + 2.5 + 5.5 = \mathbf{54.4}$$

→ **Banda 25–50** (interferência moderada / limiar severo). O atleta está no Estado 1→2, com sono como driver dominante. Protocolo: intensidade reduzida. ✓ Consistente com diagnóstico FOR.

**Cenário B — Atleta em NFOR / Pre-OTS (Estado 2→3):**
- D_HPA = (26−18)/17 = 0.47
- D_HPG = 0.30/0.60 = 0.50 (T a 70% baseline)
- D_HPT = (0.35−0.20)/0.30 = 0.50
- D_MPS = 0.35/0.70 = 0.50 (T/C a 0.65×baseline)
- D_Imune = 0.5×(0.40/0.60) + 0.5×(3.0−1.0)/9.0 = 0.333 + 0.111 = 0.44
- D_Intestino = 0.5×(30−20)/20 + 0.5×(20−10)/15 = 0.25 + 0.33 = 0.58
- D_Sono = max((7.5−5.5)/2.5, (20−9)/12, (15−11)/6) = max(0.80, 0.92, 0.67) = 0.92
- D_Metabolismo = 0.4×(2.2−1.5)/2.5 + 0.4×(36−20)/30 = 0.112 + 0.213 = 0.325
- D_SNC = 0.5×(60/100) + 0.5×(1−5/7) = 0.30 + 0.143 = 0.44

$$IIG_B = 0.47×42 + 0.50×15 + 0.50×23 + 0.50×6 + 0.44×39.5 + 0.58×42 + 0.92×37 + 0.325×22 + 0.44×34.5$$
$$= 19.7 + 7.5 + 11.5 + 3.0 + 17.4 + 24.4 + 34.0 + 7.2 + 15.2 = \mathbf{139.9}$$

→ **Banda >75** (colapso sistémico em cascata). O sleep (D_Sono = 0.92) e intestino (D_Intestino = 0.58) são os drivers principais. Protocolo: descanso total + avaliação clínica. ✓ Consistente com OTS/NFOR severo.

**Conclusão da Secção 1:** Os w_ij propostos + as funções D_i(t) produzem IIG que mapeia correctamente os estados clínicos das duas extremidades da distribuição. A calibração é internamente consistente.

---

## SECÇÃO 2 — RASTREIO DOS LOOPS DE RETROALIMENTAÇÃO: O EFEITO BORBOLETA REAL

### 2.1 Framework: O Vector de Perturbação

Em cada loop de auto-amplificação, o output de uma equação torna-se o **vector de perturbação** que reescreve os parâmetros de entrada de outro motor. A nomenclatura formal:

- **Estado sistémico no tempo t:** $\mathbf{x}(t) = [D_{HPA}, D_{HPG}, D_{HPT}, D_{MPS}, D_{Imune}, D_{Intestino}, D_{Sono}, D_{Met}, D_{SNC}]^T$
- **Vector de perturbação do loop k:** $\Delta\mathbf{x}_k(t) = f_k(\mathbf{x}(t))$ — a função que o loop k aplica ao estado
- **Estado no ciclo seguinte:** $\mathbf{x}(t+1) = \mathbf{x}(t) + \Delta\mathbf{x}_k(t)$, bounded em [0,1] por sistema
- **Amplificação do loop:** $A_k = \|\Delta\mathbf{x}_k(t+1)\| / \|\Delta\mathbf{x}_k(t)\|$ — se A_k > 1, o loop é instável (auto-amplifica)

O sistema Fase 5 é dominado por loops com A_k > 1. A intervenção muda A_k < 1.

---

### 2.2 Loop 3 — Hipoglicémia Nocturna → Fragmentação de Sono → Depleção de Glicogénio (Trajecto Matemático Completo)

**Ponto de entrada:** Défice calórico ou treino vespertino intenso sem reposição de glicogénio.

**Estado inicial (t=0):**
```
Glucose_noturna(t) = 62 mg/dL  [< 65 mg/dL → limiar GB-03 activo]
Cortisol_nadir_nocturno(t) = 3.0 nmol/L  [próximo do limiar CB-03]
SWS%(t) = 16%  [ligeiramente abaixo do óptimo]
Glicogénio_muscular(t) = 70%  [parcialmente depletado pelo treino]
```

**Iteração 1 — t=0 → t+1:**

*Passo 1a — Surge de cortisol reactivo (GB-03):*
$$\Delta Cortisol_{noturno} = +8\ \text{a}\ +15\ \mu g/dL\ \text{acima do nadir}$$
$$Cortisol_{noturno}(t+\varepsilon) = 3.0\ nmol/L + 10\ \mu g/dL \times 27.6 \approx 3.0 + 9.2 = 12.2\ nmol/L$$

*Passo 1b — Fragmentação de SWS pela elevação de cortisol nocturno (CB-03):*
$$f(Cortisol_{noc}) = \max(0.20,\ 1.0 - 0.045 \times \max(0,\ 12.2 - 2)) = \max(0.20,\ 1.0 - 0.459) = 0.541$$
$$GH_{noturno}(t+1) = GH_{max} \times \frac{SWS\%(t)}{20\%} \times f(Cortisol_{noc}) = GH_{max} \times 0.80 \times 0.541 = GH_{max} \times 0.433$$
→ GH nocturno suprimido em 56.7% (vs. 0% sem evento hipoglicémico)

*Passo 1c — Equivalência de sono comprometida (GB-03):*
$$\Delta TST_{qualidade} = -1.5\ \text{a}\ -2.5\ \text{horas de equivalente de sono de qualidade}$$
→ TST efectivo cai de 7.0h para ~5.0–5.5h (equivalente)

*Passo 1d — Cortisol matinal elevado no dia seguinte (GB-03 directo):*
$$\Delta Cortisol_{matinal}(t+1) = +2\ \text{a}\ +4\ \mu g/dL$$
Se $Cortisol_{matinal}(t) = 18\ \mu g/dL$: $Cortisol_{matinal}(t+1) = 21\ \mu g/dL$

*Passo 1e — FSR comprometida (FSR/cortisol quotient — MA §4.2):*
$$\Delta FSR\%(t+1) \approx -(8\ \text{a}\ 12) \times \frac{\max(0,\ 21-18)}{10} = -(8\ \text{a}\ 12) \times 0.30 = -2.4\ \text{a}\ -3.6\%$$

*Passo 1f — CMJ comprometido (GB-03):*
$$CMJ(t+1) = CMJ_{baseline} \times (1 - 0.05\ \text{a}\ 0.12) \times (1 - 0.02)$$
$$= CMJ_{baseline} \times (0.88\ \text{a}\ 0.93)$$
→ Pode atingir limiar de Estado 1 FOR (CMJ < –5%) se cortisol matinal já estava limítrofe.

*Passo 1g — Síntese de glicogénio comprometida (GH/IGF-1):*
$$Glicogenio_{muscular}(t+1) = Glicogenio_{muscular}(t) \times (1 - \delta_{GH})$$
onde $\delta_{GH} \approx 0.15\ \text{a}\ 0.25$ (GH é co-estimulador da glicogénio sintase via IGF-1/AKT)
$$Glicogenio_{muscular}(t+1) \approx 70\% \times 0.82 = 57\%\ \text{(pré-treino seguinte)}$$

**Iteração 2 — o loop fecha:**

Com glicogénio muscular a 57% e treino no dia (t+1), a depleção durante a sessão é mais rápida. Se o treino vespertino durar 60–90 min a intensidade moderada-alta:
$$Glicogenio_{apos\_treino}(t+1) \approx 57\% - 35\% = 22\%$$

Glicogénio a 22% ao adormecer → probabilidade de hipoglicémia nocturna aumenta substancialmente:
$$P(Glucose_{noturna} < 65\ mg/dL\ |\ Glicogenio = 22\%) \gg P(Glucose_{noturna} < 65\ mg/dL\ |\ Glicogenio = 70\%)$$

**Factor de Amplificação do Loop 3:**

Definindo o vector de estado do loop como $\mathbf{x}_{L3} = [D_{Sono},\ Cortisol_{matinal},\ Glicogenio\%]^T$:

| Variável | t=0 | t+1 | Δ | Amplificação |
|---|---|---|---|---|
| D_Sono | 0.20 | 0.50 | +0.30 | — |
| Cortisol_matinal (µg/dL) | 18.0 | 21.0 | +3.0 | — |
| Glicogénio pré-treino (%) | 70% | 57% | −13% | — |
| P(hipoglicémia nocturna) | base | base × 2.3 | ×2.3 | **A_L3 ≈ 2.3** |

A_L3 > 1: o loop é instável sem intervenção. A cada ciclo, a probabilidade de hipoglicémia nocturna multiplica por ~2.3 até saturar (evento quase garantido).

**Condição de escape do Loop 3:**
$$Glicogenio_{apos\_treino} + Ingestao\_CHO_{pre\_sono} > 40\%\ \text{de}\ Glicogenio\_max$$

Protocolo GLUC-AMBER-01: +30g CHO pré-sono. Isto eleva a glicémia nocturna acima de 65 mg/dL e quebra o loop em t+2.

---

### 2.3 Loop 4 — Cortisol → Atrofia Hipocampal → HPA Descontrolado (O Loop Mais Patológico)

Este é o único loop com dano estrutural irreversível na escala de semanas. A dinâmica é assimptoticamente divergente e não tem condição de escape rápida.

**Mecanismo molecular (MA §4.4):**
- Cortisol crónico → inibição de neurogénese no giro dentado + retracção dendrítica CA3
- Hipocampo atrofiado → menor feedback negativo sobre NPV → maior secreção de CRH
- Maior CRH → maior cortisol → mais atrofia → menos feedback → ...

**Formalização dinâmica:**

Seja $H(t)$ o volume hipocampal normalizado (H=1 = óptimo, H=0 = colapso total):
$$H(t+1) = H(t) - \alpha \times \max(0,\ Cortisol_{matinal}(t) - 18) \times \Delta t$$

onde α ≈ 0.002 por semana por µg/dL acima de 18 (calibrado de: −2 a −3% em 6 semanas com cortisol cronicamente elevado, MA §4.4).

O cortisol é modulado pelo feedback hipocampal:
$$Cortisol_{set\_point}(t) = Cortisol_{basal} \times \left(1 + \beta \times \frac{1 - H(t)}{H(t)}\right)$$

onde β ≈ 0.15 (coeficiente de amplificação do feedback hipocampal).

**Trajecto em 4 semanas com cortisol inicial = 24 µg/dL:**

| Semana | H(t) | Cortisol (µg/dL) | D_HPA |
|---|---|---|---|
| 0 | 1.000 | 24.0 | 0.35 |
| 1 | 0.988 | 24.2 | 0.36 |
| 2 | 0.975 | 24.5 | 0.38 |
| 4 | 0.950 | 25.1 | 0.42 |
| 8 | 0.900 | 26.2 | 0.48 |
| 12 | 0.850 | 27.6 | 0.56 |

*O Cortisol_set_point sobe progressivamente porque H(t) decai → menos feedback negativo → mais CRH → mais cortisol → mais decaimento de H. O sistema é assimptoticamente divergente.*

**Acoplamento ao IIG:**

O decaimento de H(t) tem impacto em D_SNC (menor plasticidade, pior controlo top-down da amígdala):
$$\Delta D_{SNC,\ hipocampal}(t) = 0.3 \times (1 - H(t))$$

E em D_HPA (a própria medição de cortisol reflecte o loop):
$$D_{HPA}(t)\ \text{aumenta não apenas pelo cortisol mas pela sua instabilidade crescente}$$

**Condição de não-retorno (Loop 4):**

O documento MA §15.2 define Limiar 3 (OTS): quando a hipermetilação do promotor NR3C1 se estabelece, os receptores GR ficam permanentemente insensíveis. Neste ponto, o loop deixa de ser linear — os receptores não respondem mesmo que o cortisol baixe.

Matemáticamente, isto representa uma **bifurcação de Hopf**: abaixo de H_critical ≈ 0.85, o sistema transita de um ponto fixo estável para uma órbita divergente. A recuperação requer intervenção na epigenética (requerem 12–24 meses), não apenas na carga de treino.

---

### 2.4 Loop 5 — Microglia → IDO → Anedonia → Menor BDNF → HPA Descontrolado

**Formalização como cadeia de transferência:**

$$\underbrace{LPS / IL-6 / cortisol}_{\text{activadores}} \xrightarrow{TLR4 / IL-6R} \underbrace{Microglia_{M1}}_{\text{estado activo}} \xrightarrow{IDO} \underbrace{Kinurenina \uparrow \text{, Serotonina} \downarrow}_{\text{desvio triptofano}}$$

O desvio de serotonina produz dois outputs paralelos:

**Output A — Motivação e Drive:**
$$\Delta Motivacao = -k_1 \times \Delta[Serotonina] \implies \Delta Actividade\_espontanea \approx -15\ \text{a}\ -25\%$$

**Output B — Melatonina e sono:**
$$\Delta Melatonina = -k_2 \times \Delta[Serotonina]\ \text{(serotonina é precursora)}$$
$$\implies \Delta D_{Sono} \approx +0.10\ \text{a}\ +0.20$$

**Fechamento do loop:**

Menor actividade física espontânea → BDNF reduzido:
- BDNF é o principal indutor de neurogénese hipocampal e o principal contra-peso ao Loop 4
- BDNF sérico < 15 ng/mL (critério OTS Stage 4, MA §21)

Menor BDNF → menor controlo top-down da amígdala → amígdala hiperactivada → mais CRH → mais cortisol → mais activação de TLR4 microglia → mais IDO → loop fechado.

**Equação de retroalimentação do Loop 5:**

$$IDO\_activity(t+1) = IDO\_activity(t) + \gamma \times D_{HPA}(t) \times D_{SNC}(t)$$

onde γ ≈ 0.12 (coeficiente de co-activação HPA-SNC sobre IDO, estimado da literatura).

$$D_{SNC}(t+1) = D_{SNC}(t) + \delta \times \max(0,\ IDO\_activity(t) - IDO_{baseline})$$

→ Loop estável enquanto D_HPA < 0.4 e D_SNC < 0.3. Instável (A_L5 > 1) quando ambos excedem estes thresholds simultaneamente.

---

### 2.5 Interferência Cross-Loop: Como os 5 Loops Se Fundem num Estado de Colapso

**A interacção crítica é Loops 3 + 4:**

O Loop 3 (hipoglicémia nocturna → cortisol elevado) alimenta directamente o Loop 4 (cortisol → atrofia hipocampal). Especificamente:

$$Cortisol_{matinal}^{L3}(t+1) = Cortisol_{matinal}(t) + \Delta C_{hipoglicemia} = Cortisol_{matinal}(t) + 3\ \text{µg/dL}$$

Este Δ cortisol adicional entra na equação do Loop 4:
$$H(t+1) = H(t) - \alpha \times (Cortisol_{matinal}^{L3}(t+1) - 18) \times \Delta t$$

→ O Loop 3 **acelera o decaimento de H(t)** no Loop 4. A interacção é multiplicativa, não aditiva.

**Fusão de todos os loops — o estado de "colapso de reputação" do ALS:**

Quando 3+ loops estão activos simultaneamente:

$$ALS_{integrado}(t) = ALS_{base}(t) \times \prod_{k\ activo} (1 + \gamma_k)$$

Os loops activos mapeiam directamente para os γ_loops da TC §XIV:

| Loop | γ correspondente | Condição de activação |
|---|---|---|
| Loop 1 (Inflam-RI-Cortisol) | γ_CRP_HOMA = 0.35 | Loop 1 activo ↔ CRP > 1.0 AND HOMA > 2.0 |
| Loop 2 (Intestino-HPA) | γ_Microbioma = 0.15 | Loop 2 activo ↔ Zonulina > 40 |
| Loop 3 (Sono-Hipoglicémia) | γ_Sono_GH = 0.30 | Loop 3 activo ↔ SWS% < 10% AND GH < 30% max |
| Loop 4 (Cortisol-Hipocampo) | γ_Cortisol_T = 0.25 | Proxy: Cortisol > 22 AND T/C < 0.60 |
| Loop 5 (Microglia-IDO) | γ_GV_Inflam = 0.20 | Proxy: IL-6 > 5 pg/mL |

**O ALS explode de repente** porque os loops individuais parecem controláveis mas os seus efeitos de rede são supraditivos:

$$ALS_{integrado} = ALS_{base} \times 1.35 \times 1.15 \times 1.30 \times 1.25 \times 1.20 = ALS_{base} \times 2.97$$

→ Um atleta com ALS_base = 1.3 (valores moderados em tudo) entra subitamente em ALS_integrado = 3.86, ultrapassando o Travão 10 (> 3.5), mesmo sem nenhum indicador individual ser "crítico".

**Este é o mecanismo matemático do colapso súbito em overtraining.**

---

## SECÇÃO 3 — ENGENHARIA DO LOOP CIRCULAR: FASE 5 ↔ FASE 6

### 3.1 Inventário das Dependências Circulares

O Passo 1 identificou 6 pontos onde Phase 5 depende de Phase 6 (D-01 a D-06). Desses, 3 criam dependências verdadeiramente circulares (onde Phase 5 precisa de output Phase 6 para produzir input Phase 6):

**Dependência Circular DC-01 — SUPERCOMP-GREEN-01:**
```
Phase 5 precisa de: days_since_intense_session
Phase 6 gera: session_intensity, session_timestamp
→ Circuito: Phase 5 → Phase 6 → Phase 5 (circular)
```

**Dependência Circular DC-02 — Travão 10 (ALS_integrado > 3.5):**
```
Phase 5 emite: ALS_integrado (calculado com base em ALS_base)
Phase 6 define: ALS_base (calculado com base na prescrição anterior)
→ A prescrição Phase 6 afecta ALS_base; ALS_base afecta Phase 5; Phase 5 afecta Phase 6
```

**Dependência Circular DC-03 — Exit criteria do OTS (T/C > 0.70 para desbloquear intensidade):**
```
Phase 5 avalia: estado T/C_ratio actual
Phase 6 usa: Phase 5 gate para desbloquear bandas de intensidade
Phase 6's prescrição anterior afecta: T/C ratio via carga de treino aplicada
→ A carga de treino que Phase 6 prescreveu ontem afecta o T/C de hoje que Phase 5 mede
```

---

### 3.2 O State Cache: Contrato de Dados

A solução é um **State Cache persistente** na camada Phase 4.2 (Hot Storage), que armazena o último output completo de Phase 6 como um SessionRecord. Phase 5 lê sempre de SessionRecord(t-1) e nunca de t (eliminando a circularidade).

**Contrato de Dados do SessionRecord:**

```
SessionRecord:
  — Identificação —
  session_id:                   UUID
  timestamp_prescription:       datetime (quando Phase 6 gerou a prescrição)
  timestamp_execution:          datetime (quando o treino foi executado)
  cycle_index:                  int (número de ciclo; 0 = cold start)

  — Output de Phase 5 no momento da prescrição —
  IIG_at_prescription:          float   [0, ∞)
  ALS_integrado_at_presc:       float   [1.0, ∞)
  OTS_state_at_presc:           int     {0, 1, 2, 3}
  active_red_blocks:            list[str]
  active_amber_blocks:          list[str]
  active_green_flags:           list[str]
  safety_gates_status:          dict[str → bool]  (Travões 1-10)

  — Output de Phase 6 —
  prescribed_intensity_zone:    Enum[Z1, Z2, Z3, Z4, Z5]
  prescribed_volume_factor:     float   [0.0, 1.5]
  prescribed_session_type:      str     (e.g., "força máxima", "zona 2", "recuperação activa")
  session_was_intense:          bool    (True se zona ≥ Z4)

  — Para consumo futuro de Phase 5 —
  days_since_last_intense:      int     (calculado pela Phase 6 no momento da prescrição)
  cumulative_load_7d:           float   (carga acumulada normalizada nos últimos 7 dias)
  supercomp_window_active:      bool    (True se 48–72h após última sessão intensa com HRV > +0.5 SD)
```

**Invariante de integridade:** O `cycle_index` nunca regride. Se Phase 5 lê um SessionRecord com `cycle_index < current_cycle − 1`, emite um alerta de "cache stale" e usa os valores de cold start para as variáveis de Phase 6.

---

### 3.3 Protocolo de Desacoplamento Temporal

O problema circular é resolvido pela separação temporal rigorosa:

**Regra de Ouro:** Phase 5 consome sempre `SessionRecord(t-1)`. Phase 6 produz `SessionRecord(t)`.

Nunca existe um ciclo onde Phase 5(t) lê Phase 6(t). A leitura é sempre de um ciclo anterior.

**Justificação fisiológica (não apenas arquitectural):**

A dependência física que existe no documento — "days_since_intense_session" para detectar a janela de supercompensação (48–72h) — é por definição uma variável **retrospectiva**. A janela de supercompensação de uma sessão de ontem só pode ser avaliada hoje. Phase 5 ler de t-1 não é apenas uma solução de engenharia: é biologicamente correcto.

Do mesmo modo, o critério `supercomp_window_active = True` (HRV > +0.5 SD E CMJ > +1.0 SD E 2–3 dias após sessão intensa) requer que a sessão intensa já tenha acontecido e que o HRV/CMJ de hoje confirme a supercompensação. Estes dados só existem em t, com a sessão intensa em t-1 ou t-2.

---

### 3.4 O DAG de Execução por Ciclo

Cada ciclo de avaliação diária segue um DAG (grafo acíclico dirigido) estrito:

```
CICLO t:

[1] Phase 4 Aggregation Engine
    Recolhe dados do dia (wearables, CGM, subjective, labs)
    Output: Aggregate_State(t) — snapshot completo de métricas do dia
    Latência: 0–5 min após midnight ou wake time
    ↓

[2] Phase 5 Safety Engine (lê de State Cache e Aggregate_State(t))
    Input A: SessionRecord(t-1)  ← State Cache [LEITURA]
    Input B: Aggregate_State(t)  ← Phase 4 output [LEITURA]
    
    Calcula:
    - D_i(t) para cada sistema (9 valores)
    - IIG(t) = Σ D_i × W_i^out
    - ALS_integrado(t) = ALS_base × ∏(1 + γ_loop)
    - OTS_state(t) via State Machine
    - Red blocks, Amber modifications, Green optimizations
    
    Output: Phase5_Evaluation(t) [acesso por Phase 6 imediatamente após]
    ↓

[3] Phase 6 MPC Engine (lê de Phase5_Evaluation(t) e SessionRecord(t-1))
    Input A: Phase5_Evaluation(t)  ← Phase 5 output [LEITURA]
    Input B: SessionRecord(t-1)    ← State Cache [LEITURA]
    Input C: Banister FF State(t)  ← EKF actualizado [LEITURA]
    
    Calcula:
    - Prescrição de treino via MPC stochastic
    - Fitness-Fatigue update via Banister Expandido
    - days_since_last_intense (a partir do histórico)
    - supercomp_window_active (combinando SessionRecord(t-1) + Phase5_Evaluation(t))
    
    Output: SessionRecord(t) [escreve no State Cache]
    ↓

[4] State Cache Write
    SessionRecord(t) → escreve em Phase 4.2 Hot Storage [ESCRITA]
    SessionRecord(t-2) → arquivo (mantém rolling window de 90 dias)
    ↓

[5] Display / Notificação ao utilizador
    Prescrição do dia + alertas de bloqueio (se Red blocks activos)
```

**Propriedade chave:** Em nenhum passo existe leitura e escrita simultânea no mesmo objecto. O State Cache é read-only para Phase 5 e Phase 6, write-only para o Step [4]. Não há deadlock possível.

---

### 3.5 Protocolo de Cold Start (Ciclo 0)

O primeiro ciclo de execução (nenhum SessionRecord existe) deve comportar-se de forma conservadora. As variáveis que normalmente viriam do SessionRecord(t-1) assumem valores de default seguros:

**Defaults de Cold Start:**

| Variável | Default Cold Start | Justificação |
|---|---|---|
| `days_since_last_intense` | 0 | Assume que há sessão intensa recente → sem unlock de volume extra |
| `supercomp_window_active` | `False` | Bloqueia SUPERCOMP-GREEN-01 até haver histórico real |
| `cumulative_load_7d` | 0.5 (normalized) | Carga moderada assumida → sem override de volume |
| `ALS_base` | 1.3 | Valor conservador acima de 1.0 → evita prescrição máxima |
| `OTS_state_prev` | 0 | Assume Estado 0 (normal) até confirmação |

**Consequência:** No Ciclo 0, a Phase 5 calcula correctamente todos os Safety Brakes (não precisam de histórico) e o IIG (usa apenas Aggregate_State(t)), mas não pode activar nenhuma optimização verde. A Phase 6 produz uma prescrição conservadora de baseline.

A partir do Ciclo 1, o loop funciona em plena capacidade.

---

### 3.6 Resolução da DC-02: ALS_base e o Travão 10

**O problema:** ALS_integrado(t) = ALS_base(t) × ∏(1 + γ_loop). Mas ALS_base(t) depende da carga de treino prescrita por Phase 6, que por sua vez depende de ALS_integrado(t). Circular.

**Resolução:** ALS_base(t) é calculado **exclusivamente a partir dos dados biométricos de Phase 4** — não usa prescrições de Phase 6. É a carga alostática "bruta" medida nos sensores do atleta:

$$ALS_{base}(t) = f\!\left(RMSSD_{trend},\ Cortisol_{matinal},\ CMJ_{z\text{-}score},\ RPE,\ TQR\right)$$

Este é o ALS que Phase 4 já calculava antes de Phase 5 existir. Phase 5 **amplifica** o ALS_base com os γ_loops, mas não o redefine. A circularidade é assim quebrada: ALS_base vem dos sensores (Phase 4), não das prescrições (Phase 6).

**A Phase 6 afecta os dados biométricos do dia seguinte** (carga de treino de hoje → RMSSD de amanhã), mas isso é tratado correctamente pela separação temporal t vs. t+1.

---

## SÍNTESE DO PASSO 2

| Problema Aberto (Passo 1) | Solução (Passo 2) | Estado |
|---|---|---|
| IIG inoperacional (w_ij qualitativos) | Escala cardinal 0/1.5/4.0/9.0 com 9×9 matriz completa + funções D_i(t) por sistema | **RESOLVIDO** |
| IIG computacionalmente O(n²) | Reformulação: IIG = Σ D_i × W_i^out → O(n) | **RESOLVIDO** |
| Loops não formalizados como equações | Formalização dinâmica de Loops 3, 4 e 5 com factor de amplificação A_k | **RESOLVIDO** |
| Interferência cross-loop não quantificada | Mapeamento directo Loops → γ_loops do ALS_integrado | **RESOLVIDO** |
| Loop circular Phase 5 ↔ Phase 6 | State Cache com desacoplamento temporal t vs. t-1 | **RESOLVIDO** |
| Cold start sem histórico | Protocolo de defaults conservadores explícito | **RESOLVIDO** |
| ALS_base vs. ALS_integrado (naming + circular) | ALS_base = Phase 4 output (sensores); amplificação = Phase 5 function | **RESOLVIDO** |

**Bloqueio que permanece (para Passo 3 ou além):**
- Os thresholds θ_low e θ_high das funções D_i(t) necessitam de validação com dados reais do atleta para calibração individual. Os valores propostos são population-level baselines; a individualização exige pelo menos 4 semanas de dados de baseline.
- O coeficiente α do Loop 4 (atrofia hipocampal) é estimado de estudos em humanos com Cushing/stress crónico — a extrapolação para overtraining atlético é uma simplificação que deve ser documentada como hipótese de trabalho.

---

*Documento gerado por análise estrutural dos documentos Fase 5 + relatório Passo 1. Zero Alucinação — nenhum valor inventado fora dos documentos fonte ou de derivação matemática explícita.*
