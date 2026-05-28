# FASE 5 — PASSO 3: CORREÇÕES E EXIGÊNCIAS CLÍNICAS

> **Input:** Passos 1 e 2 de Fase 5 + dois System Overrides do Founder.
> **Objectivo:** Corrigir erro arquitectural de tempo (t vs t-1), auditar limiares sem base documental sólida, mapear coeficientes sem fórmula clínica real, e identificar variáveis biológicas críticas ignoradas pelos algoritmos actuais.
> **Regra:** Sem código Python. Markdown crítico e técnico. Pontas abertas para Fase 6 onde necessário.

---

## SECÇÃO 0 — RECONHECIMENTO DOS OVERRIDES

**OVERRIDE 1 aceite e integrado.** O Passo 2 cometeu um erro conceptual ao descrever o desacoplamento temporal de forma genérica. A frase "Phase 5 lê de SessionRecord(t-1)" foi aplicada demasiado amplamente, criando a ilusão de que o estado biológico poderia vir de ontem. Está errado. O estado biológico é **sempre** de hoje.

**OVERRIDE 2 aceite e integrado.** A Fase 5 é um motor de avaliação, não de prescrição. Os seus outputs são tensores de estado — D_i(t), IIG(t), ALS_integrado(t), OTS_state(t), flags de bloqueio — que alimentam a Fase 6 (MPC + Banister). As "pontas" que se ligam à Fase 6 não devem ser fechadas prematuramente.

---

## SECÇÃO 1 — ARQUITECTURA CORRIGIDA DO TEMPO (t vs t-1)

### 1.1 O Modelo Correcto em 3 Linhas

**Linha 1 — Biologia de Hoje (Input A):** A Fase 5 lê **Aggregate_State(t)** da Fase 4: a totalidade do estado biológico medido hoje — cortisol matinal de hoje, HRV da noite passada, CMJ testado esta manhã, glicemia das últimas 24h via CGM, TST e SWS% da noite passada, temperatura hoje, CRP do último exame, etc. **Nenhuma variável biológica vem de t-1.**

**Linha 2 — Histórico de Treino de Ontem (Input B):** A Fase 5 lê **SessionRecord(t-1)** do State Cache da Fase 4.2: exclusivamente o log da prescrição e execução de treino da Fase 6 do dia anterior — `days_since_last_intense`, `cumulative_load_7d`, `supercomp_window_active`, `session_was_intense`. **Nenhum parâmetro biológico vem desta fonte.**

**Linha 3 — Tensores de Estado para a Fase 6 (Output):** A Fase 5 emite **Phase5_Evaluation(t)**: o vector D_i(t), os escalares IIG(t) e ALS_integrado(t), o estado da máquina OTS_state(t), e os flags de bloqueio (Red/Amber/Green). Estes tensores são os inputs da Fase 6 — a Fase 5 não prescreve nada.

### 1.2 Diagrama de Fusão Corrigido

```
AGGREGATE_STATE(t)          SESSION_RECORD(t-1)
[Fase 4 — hoje]             [State Cache — ontem]
  Cortisol_matinal(t)         days_since_last_intense
  HRV_RMSSD(t)                cumulative_load_7d
  CMJ_z(t)                    supercomp_window_active
  TST(t), SWS%(t)             session_was_intense
  Glucose_min_nocturna(t)     ──────────────────────
  CRP(t), IL-6(t)            [APENAS histórico de treino
  Temp_central(t)             da Fase 6 — ZERO biologia]
  ...
        │                              │
        └──────────┬───────────────────┘
                   ▼
         FASE 5 — Motor de Avaliação
                   │
                   ▼
         PHASE5_EVALUATION(t)
           D_i(t)  [vector 9 dimensões]
           IIG(t)  [escalar]
           ALS_integrado(t)  [escalar]
           OTS_state(t)  [0/1/2/3]
           Red_blocks[], Amber_mods[], Green_opts[]
                   │
                   ▼
              FASE 6 (MPC + Banister)
              [recebe tensores; prescreve]
```

### 1.3 O que muda face ao Passo 2

No DAG de 5 passos do Passo 2, o Step [2] já estava correcto na sua descrição operacional ("Input A: SessionRecord(t-1) ← State Cache; Input B: Aggregate_State(t) ← Phase 4"). O erro estava na **narrativa conceptual** que enquadrava o desacoplamento temporal como se toda a leitura da Fase 5 fosse retrospectiva. A correcção clarifica que:

- O eixo temporal da biologia é **t** (hoje, imperativo, sempre actual).
- O eixo temporal do treino é **t-1** (ontem, retrospectivo, único campo do State Cache lido).
- A "justificação fisiológica" do Passo 2 (supercompensação é retrospectiva) é **correcta** e mantém-se, mas aplica-se apenas à variável `days_since_last_intense`, não à biologia em geral.

---

## SECÇÃO 2 — INVENTÁRIO DE LIMIARES ARBITRÁRIOS

Classificação: **[DOC]** = explicitamente suportado nos documentos TC ou MA; **[INFERIDO]** = extrapolado de dados adjacentes nos documentos; **[INVENÇÃO]** = sem base documental — requer investigação de literatura médica pelo Founder.

### 2.1 Função D_HPA(t)

`D_HPA(t) = max(0, min(1, (Cortisol_matinal − θ_low) / (θ_high − θ_low))) + penalidade_CAR`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| θ_low = 18 µg/dL | Limiar de início de supressão de T e MPS | **[DOC]** TC §2.1 CB-01 e CB-02 | Confirmado |
| θ_high = 35 µg/dL | Nível de supressão máxima (T_suprimida = 0.40 × T_baseline) | **[INFERIDO]** CB-01 formula: floor 0.40 → Cortisol = (1−0.40)/0.030+18 ≈ 38 µg/dL. 35 é conservador. | Founder deve confirmar: é 35 ou 38 µg/dL o tecto clínico operacionalizável? |
| Penalidade CAR blunted (+0.10) | Adição manual quando CAR < 30% de incremento | **[INVENÇÃO]** | Founder deve determinar: qual é o limiar numérico de CAR blunted? < 20% de incremento? < 30%? Qual é o tamanho do efeito em D_HPA? |
| CAR limiar "plana" < 30% incremento | Definição de CAR blunted usada no OTS State 1 criteria | **[DOC]** TC §9.1 NFOR: "CAR: plana ou negativa (< 30% de incremento)" | Confirmado, mas só para diagnóstico OTS — não calibrado como penalidade contínua em D_HPA |

### 2.2 Função D_HPG(t)

`D_HPG(t) = max(0, min(1, (T_baseline − T_actual) / (T_baseline × 0.60)))`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| Ceiling de supressão = 60% de T_baseline | Implica floor de T_actual = 0.40 × T_baseline | **[DOC]** TC CB-01 formula: `T_suprimida = T_baseline × max(0.40, ...)` → floor = 0.40 | Confirmado |
| Uso de T_total (não T_livre) | D_HPG usa T_total como proxy | **[INVENÇÃO]** | Founder deve decidir: usar T_livre (biologicamente mais relevante) ou T_total (mais comum em labs)? O IB-01b mostra que SHBG eleva com CRP, tornando T_total enganosa. |

### 2.3 Função D_HPT(t)

`D_HPT(t) = max(0, min(1, (rT3_T3_ratio − θ_low) / (θ_high − θ_low)))`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| θ_low = 0.20 (rT3/T3) | Início de perturbação tiroideia | **[INFERIDO]** MA §15.2 OTS Stage 3: "rT3/T3 ratio > 0.5" implica 0.20 como margem de preocupação | Founder deve validar: qual é o rT3/T3 ratio em indivíduos saudáveis (= θ_low)? Literatura cita ~0.10-0.15 como normal. |
| θ_high = 0.50 (rT3/T3) | Colapso tiroideo máximo | **[DOC]** MA §15.2: "rT3/T3 ratio > 0.5" = OTS Stage 3 marker | Confirmado como threshold de OTS. Mas a função D_HPT normaliza linearmente entre 0.20 e 0.50 — esta linearidade precisa de validação. |
| rT3/T3 vs FT3/rT3 | TC §9.1 usa "FT3/rT3 < 6" (= rT3/FT3 > 0.167); MA usa "rT3/T3 > 0.5" | **[INCONSISTÊNCIA INTERNA]** | Founder deve clarificar: os documentos usam dois ratios diferentes. O código deve usar T3 total ou FT3 como denominador? Requer unificação antes de implementação. |

### 2.4 Função D_MPS(t)

`D_MPS(t) = max(0, min(1, (T_C_baseline − T_C_ratio) / (T_C_baseline × 0.70)))`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| Ceiling de supressão = 70% de T/C baseline | T/C = 0.30 × baseline = D_MPS = 1.0 | **[DOC]** TC §9.1: T/C < 0.30 × baseline = OTS (Estado 3) | Confirmado |
| Linearidade entre T/C 0.70 e 0.30 | MPS declina linearmente com T/C | **[INVENÇÃO]** | A supressão de MPS é linear com T/C ou exponencial? Cuthbertson 2005 (referenciado no TC) pode ter dados de dose-resposta. |
| T/C ratio = proxy suficiente de MPS | Uso de ratio hormonal como proxy de output funcional | **[INFERIDO]** | D_MPS seria mais preciso com FSR directa (TC MA §4.2 tem a fórmula FSR/Cortisol Quotient). Considerar adicionar FSR como sub-componente. |

### 2.5 Função D_Imune(t)

`D_Imune(t) = 0.5 × D_sIgA + 0.5 × D_CRP`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| sIgA θ_high = 60% de supressão (sIgA = 0.40 × baseline) | D_sIgA = 1.0 em OTS | **[DOC]** TC §9.1: sIgA < 40% = OTS Stage 3 | Confirmado |
| sIgA θ_low = 20% de supressão (sIgA = 0.80 × baseline) | Início de perturbação imune | **[DOC]** TC §9.1: sIgA 60-80% = FOR (Estado 1) | Confirmado |
| CRP θ_low = 1.0 mg/L | Início de inflamação relevante | **[DOC]** TC IB-03: "CRP > 1.0 mg/L E HOMA > 2.0" como activador de Loop 1 | Confirmado como threshold de loop, mas 1.0 como θ_low de D_CRP é extensão minha |
| CRP θ_high = 10.0 mg/L | Inflamação aguda severa = D_CRP = 1.0 | **[DOC]** TC Part XIII: Red Block INFLAM-RED-01 activa a CRP > 10.0 mg/L | Confirmado |
| Peso 50/50 entre sIgA e CRP | Ponderação igual das duas componentes | **[INVENÇÃO]** | Founder deve determinar: imunidade humoral (sIgA) e sistémica (CRP) têm o mesmo peso em D_Imune? Argumentável que sIgA é mais sensível ao overtraining que CRP. |

### 2.6 Função D_Intestino(t)

`D_Intestino(t) = 0.5 × D_Zonulina + 0.5 × D_LBP`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| Zonulina θ_high = 40 ng/mL | Permeabilidade intestinal activada | **[DOC]** TC MB-01: "Zonulina > 40 ng/mL E LBP > 25 µg/mL" | Confirmado |
| Zonulina θ_low = 20 ng/mL | Início de preocupação com permeabilidade | **[INVENÇÃO]** | Founder deve investigar: qual é o valor normal de zonulina sérica em adultos saudáveis? Literatura cita 17-40 ng/mL como variação normal. O θ_low = 20 pode estar dentro do range normal. |
| LBP θ_high = 25 µg/mL | Endotoxemia metabólica activa | **[DOC]** TC MB-01: "LBP > 25 µg/mL" | Confirmado |
| LBP θ_low = 10 µg/mL | Início de activação imune intestinal | **[INVENÇÃO]** | Founder deve investigar: qual é o LBP normal em adultos sedentários saudáveis? Literatura cita 5-15 µg/mL como range normal em saudáveis. O θ_low = 10 pode já estar no limite superior do normal. |

### 2.7 Função D_Sono(t)

`D_Sono(t) = max(D_TST, D_SWS, D_REM)`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| TST θ_low = 7.5h | Sono óptimo | **[INFERIDO]** TC SB-01: < 6h como problema, < 5h como Red Block. 7.5h como óptimo é medicina do sono geral (NSF: 7-9h) | Founder deve validar: 7.0h ou 7.5h como θ_low para atletas? A literatura de overtraining usa 8-9h como mínimo para elite. |
| TST θ_high = 5.0h | Privação severa = D_TST = 1.0 | **[DOC]** TC SB-01: "TTS < 5.0 horas (privação severa)" → protolcolo de emergência | Confirmado |
| SWS% θ_low = 20% do TST | SWS óptimo | **[DOC]** TC CB-03: "SWS_target% = 20% do TTS" | Confirmado |
| SWS% θ_high = 8% do TST | SWS comprometido = D_SWS = 1.0 | **[INFERIDO]** TC SB-02: SWS < 10% como limiar de bloqueio. Usei 8% como tecto conservador. | Founder deve confirmar: usar 10% (document value) como θ_high, não 8%. A escolha de 8% não tem base documental explícita. |
| REM% θ_low = 15% do TST | REM óptimo | **[INVENÇÃO]** | Founder deve investigar: qual é o REM% óptimo? TC usa 12% como limiar patológico. A meta deve ser > 20%? > 25%? A escolha de 15% é arbitrária. |
| REM% θ_high = 9% do TST | REM comprometido = D_REM = 1.0 | **[INFERIDO]** TC SB-02: "REM < 12%" como limiar de bloqueio. Usei 9% como tecto — sem justificação explícita. | Usar 12% como θ_high para consistência com documento. 9% é mais restritivo sem base. |

### 2.8 Função D_Metabolismo(t)

`D_Metabolismo(t) = 0.4 × D_HOMA + 0.4 × D_CV + 0.2 × D_HbA1c`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| HOMA-IR θ_low = 1.5 | Início de resistência à insulina subcínica | **[INVENÇÃO]** | Founder deve investigar: limiar de HOMA-IR normal em atletas? Em atletas de força e resistência, HOMA-IR de 1.0-1.8 é comum. O TC usa > 2.0 como threshold de loop. 1.5 como θ_low pode estar dentro do normal para atletas. |
| HOMA-IR θ_high = 4.0 | Resistência à insulina severa = D_HOMA = 1.0 | **[INVENÇÃO]** | TC não define ceiling de HOMA para atletas. Diabéticos tipo 2 têm HOMA > 4. Para atletas, 3.0 pode já ser severo. Requer validação. |
| CV_glucose θ_low = 20% | Variabilidade glicémica aceitável | **[INFERIDO]** | TC usa 36% como threshold. 20% como "normal" é da literatura CGM (CV < 36% = gestão aceitável; CV < 20% = óptima). Mas não está nos documentos da Fase 5. |
| CV_glucose θ_high = 50% | Variabilidade severa = D_CV = 1.0 | **[INFERIDO]** TC §3.2 GB-02: "CV > 50% (variabilidade glicémica severa)" | Confirmado como limiar de severidade mas TC não o usa como tecto de normalização. |
| HbA1c θ_low = 5.2% | HbA1c óptima | **[INVENÇÃO]** | TC GB-04 usa 5.6% como threshold. 5.2% é arbitrário. Founder deve definir: qual é o HbA1c alvo para atletas de performance? American Diabetes Association usa < 5.7% como normal. |
| HbA1c θ_high = 6.5% | HbA1c de diabético tipo 2 = D_HbA1c = 1.0 | **[INVENÇÃO]** | TC GB-04 não define ceiling. 6.5% é o critério diagnóstico de DM2 (ADA) — não específico para overtraining. Para atletas, 5.9% pode já ser problemático. |
| Pesos 0.4/0.4/0.2 | HOMA e CV pesam 2× mais que HbA1c | **[INVENÇÃO]** | HbA1c é marcador crónico (6-8 semanas), HOMA e CV são agudos. A ponderação 2:2:1 favorece o estado agudo. Clinicamente justificável mas não documentado. |

### 2.9 Função D_SNC(t)

`D_SNC(t) = 0.5 × D_PVT + 0.5 × D_Wellbeing`

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| PVT_RT +100ms como D_PVT = 1.0 | Degradação cognitiva máxima | **[INFERIDO]** TC SB-01: "+50 a +100 ms" para 3 noites < 6h. Usei o limite superior do intervalo. | Founder deve definir: +50ms ou +100ms como tecto? Usar o midpoint (+75ms)? |
| Wellbeing 7.0/10 como D_Wellbeing = 0.0 | Wellbeing óptimo | **[INFERIDO]** TC §9.1: Wellbeing > 6/10 = Estado 0 normal. Usei 7.0 como patamar óptimo. | Diferença entre 6 e 7 como threshold é clinicamente relevante? POMS não usa esta escala directamente. |
| Wellbeing 0/10 como D_Wellbeing = 1.0 | Colapso total de wellbeing | **[INVENÇÃO]** | TC usa "Vigor < 10/100; Fadiga > 60/100" do POMS para OTS. Wellbeing em escala 0-10 é mapeamento diferente. Requer harmonização de escalas. |

### 2.10 Matriz w_ij e Escala Cardinal

| Parâmetro | Valor Usado | Classificação | O que Validar |
|---|---|---|---|
| Escala 0.0 / 1.5 / 4.0 / 9.0 | Tradução de —/↓/↓↓/↓↓↓ em pesos cardinais | **[INVENÇÃO CALIBRADA]** Ancoragem ao comportamento esperado de IIG em bandas | Founder deve validar: a calibração IIG_A ≈ 54 para FOR e IIG_B ≈ 140 para NFOR reproduz estados clínicos reais? Se não, a escala inteira deve ser recalibrada. |
| Valores específicos do 9×9 w_ij | Leitura da Tabela MA §24 traduzida para cardinais | **[INTERPRETAÇÃO]** | A Tabela MA §24 usa ↓/↑/—/↓↓↓ mas não todos os símbolos são unívocos. Células ambíguas (ex: HPA→Intestino = ↑ ou ↑↑?) podem ter sido lidas erroneamente. Founder deve fazer leitura independente da Tabela §24. |
| W_i^out (totais por sistema) | Calculados da soma de colunas do 9×9 | **[DERIVADO]** — correctos se a matriz for correcta | Erro na matriz propaga para W_i^out. Revisão da matriz = revisão dos W_i^out. |

### 2.11 Condições de Activação dos γ_loops

| Loop | Condição de Activação Usada | Classificação | O que Validar |
|---|---|---|---|
| Loop 1 (γ = 0.35) | CRP > 1.0 AND HOMA > 2.0 | **[DOC]** TC IB-03 explicitamente | Confirmado |
| Loop 2 (γ = 0.15) | Zonulina > 40 | **[DOC]** TC MB-01 | Confirmado |
| Loop 3 (γ = 0.30) | SWS% < 10% AND GH_estimado < 30% máximo | SWS% **[DOC]**; GH < 30% **[INVENÇÃO]** | GH nocturno não é medido directamente. Founder deve definir: como estimar GH_noturno_real sem medição directa? Via fórmula CB-03 com SWS% e Cortisol_nocturno? |
| Loop 4 (γ = 0.25) | Proxy: Cortisol > 22 AND T/C < 0.60 | **[INVENÇÃO]** — H(t) (volume hipocampal) não é medível | O Loop 4 real requer neuroimagem. O proxy Cortisol+T/C detecta o estado que CAUSARIA atrofia hipocampal, não a atrofia em si. Founder deve confirmar: este proxy é suficiente para activar γ = 0.25? |
| Loop 5 (γ = 0.20) | IL-6 > 5 pg/mL como proxy de IDO activado | **[INFERIDO]** — IDO não é medida rotineiramente | MA §5.3: "IDO activada por IL-6/IFN-γ/LPS". IL-6 > 5 pg/mL é o mesmo limiar de IB-01. Razoável como proxy mas não confirmado directamente. |

---

## SECÇÃO 3 — OS COEFICIENTES "MÁGICOS"

Lista de coeficientes de amplificação, taxas de degradação, e factores de escala usados nos Passos 1 e 2 que não têm fórmula clínica validada na literatura. Ordenados por impacto arquitectural.

### 3.1 A_L3 ≈ 2.3 — Factor de Amplificação do Loop 3

**Usado em:** Secção 2.2, Loop 3 (Sono-Hipoglicémia).
**Como foi obtido:** Análise de cenário: glicogénio de 70% → 57% após Loop 3 → P(hipoglicémia nocturna) escalada por ~2.3×. Este factor é uma estimativa de plausibilidade biológica, não uma constante de literatura.
**O que falta:** Uma função de probabilidade calibrada que relacione `Glicogenio_pré_sono%` com `P(Glucose_nocturna < 65 mg/dL)`. Esta função requereria dados de CGM contínuo de uma coorte de atletas com diferentes níveis de glicogénio ao adormecer. Sem ela, A_L3 = 2.3 é um chute intuitivo.
**Impacto de erro:** Se A_L3 real = 1.4 (loop sub-estável) vs. 3.5 (loop super-instável), a recomendação de intervenção urgente muda completamente.

### 3.2 α ≈ 0.002/semana/µg_dL — Taxa de Decaimento Hipocampal (Loop 4)

**Usado em:** `H(t+1) = H(t) − α × max(0, Cortisol(t) − 18) × Δt`
**Como foi obtido:** Extrapolação de estudos de Doença de Cushing e stress crónico (McEwen 2007, Lupien 2009): cortisol cronicamente elevado a ~35 µg/dL → volume hipocampal −2 a −3% em 6 semanas via RM. Derivei α por divisão: (0.025 / (35−18) / 6) ≈ 0.0002. Usei 0.002 — um factor 10× maior sem justificação explícita.
**O que falta:** Estudos específicos em sobretreinamento atlético (não Cushing). A Doença de Cushing implica cortisol > 50 µg/dL; em NFOR/OTS, cortisol é 22-30 µg/dL. O mecanismo pode ser qualitativamente diferente. α para atletas pode ser 5-20× menor.
**Impacto de erro:** H_critical = 0.85 sendo atingido em 12 semanas com α = 0.002. Se α = 0.0002, o mesmo declínio leva 120 semanas — o Loop 4 passa de urgente para semi-crónico.

### 3.3 H_critical ≈ 0.85 — Ponto de Bifurcação de Hopf (Loop 4)

**Usado em:** Limiar abaixo do qual o feedback negativo hipocampal falha e o eixo HPA entra em divergência.
**Como foi obtido:** Extrapolação qualitativa: estudos de RM mostram que indivíduos com volume hipocampal < 85% do normal têm feedback HPA prejudicado (Pruessner 2010 citado indirectamente). O valor de 0.85 é uma leitura de 15% de atrofia como ponto de não-retorno.
**O que falta:** Nenhum estudo define um threshold numérico de volume hipocampal normalizado como ponto de bifurcação no eixo HPA em humanos atletas. O H_critical = 0.85 é um conceito de plausibilidade física, não um parâmetro clínico medido.
**Nota crítica:** O Loop 4 pode ser matematicamente elegante mas empiricamente não calibrável com dados actuais de overtraining. Proposta para o Founder: tratar H(t) como variável latente não observável e usar T/C ratio + CAR + HRV trend como proxy observável, sem tentar modelar o volume hipocampal explicitamente.

### 3.4 γ_IDO ≈ 0.12 — Co-activação HPA-SNC sobre IDO (Loop 5)

**Usado em:** `IDO_activity(t+1) = IDO_activity(t) + γ × D_HPA(t) × D_SNC(t)`
**Como foi obtido:** Estimativa pura. Calibrado de forma a que o loop seja sub-estável (A_L5 < 1) quando D_HPA < 0.4 e D_SNC < 0.3, e instável quando ambos excedem esses thresholds.
**O que falta:** Estudos de cinética de IDO em humanos saudáveis e em overtraining. A activação de IDO por IL-6 tem dados (Takikawa 1988, Myint 2012) mas não em contexto atlético. γ = 0.12 é um grau de liberdade não calibrado.

### 3.5 δ — Taxa de Crescimento de D_SNC por IDO (Loop 5)

**Usado em:** `D_SNC(t+1) = D_SNC(t) + δ × max(0, IDO_activity(t) − IDO_baseline)`
**Como foi obtido:** Não foi quantificado no Passo 2. O valor de δ foi deixado implícito ("calibrado de forma a que o loop feche"). É um coeficiente completamente indefinido.
**O que falta:** Dados de correlação entre actividade de IDO (ou ratio kynurenina/triptofano) e scores de função cognitiva/motivação em atletas. Este é provavelmente o coeficiente mais difícil de calibrar sem dados primários.

### 3.6 k₁ e k₂ — Serotonina → Motivação e Serotonina → Melatonina (Loop 5)

**Usados em:** `ΔMotivacao = −k₁ × Δ[Serotonina]` e `ΔMelatonina = −k₂ × Δ[Serotonina]`
**Como foram obtidos:** Descritos qualitativamente nos documentos (MA §5.2, §5.3) mas nunca quantificados.
**O que falta:** k₁ e k₂ são constantes de proporcionalidade entre serotonina periférica (mensurável) e efeitos centrais (não directamente mensuráveis). A serotonina sistémica não cruza a BHE — o que cruza é o triptofano. O modelo simplificado é útil como mapa causal mas não é programável sem proxies adicionais.

### 3.7 ALS_base Função de Composição

**Usado em:** `ALS_base(t) = f(RMSSD_trend, Cortisol_matinal, CMJ_z-score, RPE, TQR)`
**Como foi obtida:** Mencionada na Fase 4 como já existente antes da Fase 5. Mas a função f(·) nunca foi definida matematicamente no contexto da Fase 5.
**O que falta:** A fórmula exacta de composição do ALS_base a partir das 5 variáveis da Fase 4. Sem esta fórmula, o ALS_integrado = ALS_base × ∏(1 + γ) não é computável. **Este é um bloqueio de implementação de Nível 1** — o ALS_base deve estar definido na Fase 4; confirmar com o Founder se existe ou se precisa de ser desenhado.

### 3.8 Coeficientes da Fórmula T_suprimida (CB-01)

**Usados em:** `T_suprimida = T_baseline × max(0.40, 1.0 − 0.030 × max(0, Cortisol − 18))`
**Origem:** **[DOC]** TC §2.1 — calibrado de Cumming et al. 1983 e Brownlee et al. 2005.
**Status:** Documentalmente suportado. No entanto, os documentos originais (Cumming 1983) são de 1983 e a amostra era pequena (n=10 ciclistas). Para uma implementação robusta, o Founder deve verificar se há meta-análises mais recentes com coeficientes actualizados.

### 3.9 Coeficiente 0.025 da Leucina Efectiva (CB-02)

**Usado em:** `Leucina_efectiva = Leucina_ingerida × max(0.45, 1.0 − 0.025 × max(0, Cortisol − 18))`
**Origem:** **[DOC]** TC §2.2.
**Status:** Suportado no documento mas com a mesma ressalva de CB-01 (dados primários antigos). O limiar de leucina para activar mTORC1 (> 0.40 g/kg em vez de > 0.25 g/kg a Cortisol > 25 µg/dL) está referenciado no TC mas a fonte primária não é citada explicitamente.

### 3.10 Factor JNK ×1.35 (IB-02)

**Usado em:** Bloqueio de mTORC1 amplificado por IL-1β > 15 pg/mL.
**Origem:** **[DOC]** TC §4.2 — "efeito combinado TNF-α + IL-1β é supraditivo (não apenas aditivo); factor de amplificação JNK: ×1.35".
**Status:** Documentalmente suportado no TC, mas sem referência primária citada. O Founder deve verificar qual estudo fornece o factor 1.35. Hotamisligil 1994 ou Bennett 1997 são candidatos prováveis (activação de JNK por TNF-α/IL-1β).

---

## SECÇÃO 4 — VARIÁVEIS EXTERNAS NÃO-MAPEADAS

Factores biologicamente críticos que os loops e funções D_i(t) ignoram por ausência de mapeamento formal. Ordenados por impacto estimado no IIG.

### 4.1 Glicogénio Muscular em Tempo Real — CRÍTICO para Loop 3

**O que é:** Conteúdo intramuscular de glicogénio (mmol/kg músculo húmido). Faixa funcional: 20-120 mmol/kg.
**Porquê importa:** O Loop 3 (hipoglicémia nocturna) é mecanisticamente dependente do glicogénio ao adormecer, não da glicemia ao adormecer. Um atleta com glicemia de 80 mg/dL mas glicogénio de 18% tem risco alto de hipoglicémia nocturna. Um atleta com glicemia de 70 mg/dL mas glicogénio de 75% pode completar a noite sem evento.
**O que os algoritmos fazem:** Usam `Glucose_nocturna < 65 mg/dL` como trigger. Este é um proxy grosseiro — detecta o evento mas não a probabilidade do evento.
**Gap:** D_Metabolismo não tem componente de glicogénio. Loop 3 não tem função de probabilidade baseada em glicogénio.
**Solução possível (para Fase 6):** Estimação de glicogénio a partir de carga de treino + ingestão de CHO + tempo desde última sessão — variáveis que a Fase 6 controla. Circular mas resolvível pela mesma lógica do State Cache.

### 4.2 Fadiga do Sistema Nervoso Simpático — IMPORTANTE para D_HPA e D_SNC

**O que é:** Tónus simpático basal elevado cronicamente, distinto do cortisol. Medido por: variabilidade de HRV de alta frequência (HF-HRV) vs. baixa frequência (LF-HRV); resposta ortostática (Δ FC deitado→em pé); FC de repouso em trend crescente.
**Porquê importa:** Um atleta em NFOR pode ter RMSSD relativamente mantido (porque o SNC ainda compensa) mas ter LF/HF ratio elevado — indicando dominância simpática sem queda de tónus parassimpático. As D_i actuais usam RMSSD como proxy de HRV, não a componente simpática.
**Gap:** D_HPA é baseado em cortisol + CAR. A activação simpática autónoma não é capturada. O Loop 4 (cortisol → hipocampo) não inclui a contribuição do CRH e da activação simpática directa sobre a neuroplasticidade.

### 4.3 Estado de Hidratação (Δ BW%) — AUSENTE das Funções D_i

**O que é:** Percentagem de perda de massa corporal por desidratação. HB-01 e HB-02 têm thresholds de Red/Amber Block.
**Porquê importa:** Desidratação de 2% compromete VO2max (−4 a −8%), cognição (PVT +10-15 ms), síntese proteica, e termorregulação. Estes efeitos deveriam reflectir-se em D_SNC, D_MPS, e potencialmente D_HPA.
**Gap:** Δ BW% só activa bloqueios (Travão 2: > −3%), não entra em nenhuma D_i como variável contínua. Um atleta com −1.8% de desidratação crónica subtil tem IIG subestimado.
**Solução:** Adicionar D_Hidratação(t) como décima dimensão, ou incorporar Δ BW% como modificador em D_SNC e D_MPS.

### 4.4 Temperatura Corporal — AUSENTE das Funções D_i

**O que é:** Temperatura central (via cápsula ingestível, tímpano, ou proxy cutâneo).
**Porquê importa:** Temperatura > 38°C activa Travão 1 (bloqueio absoluto) mas não entra em nenhuma D_i. Uma temperatura de 37.6°C (sub-febril, clinicamente relevante) aumenta o custo metabólico de qualquer exercício, activa o eixo HPA, e perturba a qualidade do sono. O IIG não o captura.
**Gap:** Temperatura está completamente ausente do vector D_i(t). Só existe como trigger binário no motor de bloqueios.

### 4.5 Acidose Metabólica Residual — AUSENTE das Funções D_i

**O que é:** pH muscular pós-exercício e velocidade de recuperação para pH = 7.0. Proxy mensurável: lactato sérico de repouso (> 2 mmol/L em repouso = acidose residual).
**Porquê importa:** AB-01 mapeia completamente a cascata de colapso enzimático por acidose durante exercício. Mas D_MPS não inclui acidose residual. Um atleta com lactato de repouso de 3.5 mmol/L está num estado metabólico comprometido antes do treino — o IIG não detecta.
**Gap:** Lactato sérico de repouso não é variável de input em nenhuma D_i. Considerar como componente de D_Metabolismo.

### 4.6 Micronutrientes (Mg²⁺ e B6/PLP) — AUSENTES das Funções D_i

**O que é:** MA §14 dedica um capítulo completo às cascatas de Mg²⁺ (≥ 300 enzimas afectadas) e B6/PLP (≥ 100 enzimas afectadas). Ambos têm efeitos em MPS, HRV, neurotransmissores, e síntese de heme.
**Porquê importa:** Défice de Mg²⁺ → canais NMDA hiperactivos (insónia, ansiedade, HRV reduzido), ATP inactivo (toda a bioenergética comprometida), ribossomas instáveis (MPS reduzida). Estes efeitos afectam D_SNC, D_Sono, e D_MPS mas sem entrada directa.
**Gap:** Nenhuma D_i capta o estado de micronutrientes. O proxy mais próximo (HRV para Mg²⁺; PVT para B6) é demasiado coarse para detectar défices subtis.
**Limitação prática:** Mg²⁺ eritrocitário e PLP sérico não são medidas de rotina — são exames laboratoriais específicos. A inclusão requer que o sistema de inputs da Fase 4 os incorpore.

### 4.7 Alinhamento Circadiano (Cronotipo vs. Horário Real) — AUSENTE

**O que é:** O ângulo de fase entre o cronotipo do atleta (DLMO — Dim Light Melatonin Onset) e o horário actual de sono. MA §8 desenvolve o impacto epigenético do desalinhamento circadiano.
**Porquê importa:** Um atleta com cronotipo vespertino (DLMO às 23h) forçado a treinar às 6h tem desalinhamento circadiano que: reduz insulino-sensibilidade (↓ 15-20%), perturba a secreção de testosterona (pico desloca-se), e fragmenta o SWS mesmo que o TST seja adequado.
**Gap:** D_Sono usa TST/SWS%/REM% — métricas de quantidade e estrutura do sono, não de alinhamento circadiano. Dois atletas com o mesmo TST podem ter D_Sono idênticos mas cargas alostáticas muito diferentes se um está alinhado e outro desalinhado.

### 4.8 Stress Psicológico Exógeno — SUBREPRESENTADO em D_SNC

**O que é:** Score de stress psicossocial crónico (trabalho, relações, finanças, competição). MA §12 documenta que o stress psicológico activa o eixo HPA de forma indistinguível do stress físico.
**Porquê importa:** D_SNC usa PVT (proxy cognitivo) e Wellbeing subjectivo. Estas captam o EFEITO do stress mas não a CAUSA. Um atleta com alta carga psicológica externa pode ter D_HPA elevado sem qualquer razão de treino — e o IIG classificará correctamente mas sem saber porquê.
**Gap:** A Fase 5 não separa a contribuição de treino vs. vida para o D_HPA. Para a Fase 6 (que vai ajustar o treino), esta distinção é critica: se o cortisol é alto por stress do trabalho, reduzir volume de treino é incorreto; se é alto por overtraining, é correto. Ponta aberta para Fase 6: o MPC precisa desta distinção.

### 4.9 Paradoxo da Resistência Glucocorticoide — GAP DE CALIBRAÇÃO SISTÉMICO

**O que é:** MA §12.2 documenta que em OTS Stage 3+, os linfócitos desenvolvem resistência ao GR (downregulação). O cortisol perde o efeito anti-inflamatório mas mantém a imunossupressão.
**Porquê importa:** As fórmulas CB-01, CB-02, CB-03 são calibradas para sensibilidade normal ao cortisol. Em OTS Stage 3 (D_HPA ≈ 0.9), um atleta tem GR resistência — o cortisol de 28 µg/dL produz menos supressão de T e MPS do que as fórmulas predizem, mas mais inflamação (porque o efeito anti-inflamatório falhou). As D_i estarão sistematicamente erradas para atletas em OTS Stage 3+.
**Gap:** Nenhuma D_i tem um modificador de resistência glucocorticoide. Este é um erro de calibração que cresce com a severidade do estado — exactamente quando a precisão mais importa.
**Resolução possível:** Adicionar um flag `GC_resistance = True` quando OTS_state = 3 e usar fórmulas alternativas para CB-01/CB-02.

### 4.10 Estado de Lesão Activa e DOMS — AUSENTE de D_MPS

**O que é:** Presença de lesão musculoesquelética activa (tendão, músculo, articulação) ou DOMS severo (CK > 3× normal, dor > 7/10).
**Porquê importa:** Lesão activa activa IL-6, TNF-α localmente → D_Imune sobe via CRP proxy. Mas o impacto directo em D_MPS (o músculo lesionado não responde ao treino) não é capturado separadamente. Um atleta com tendinopatia do Aquiles tem D_MPS comprometido por razões estruturais, não hormonais.
**Gap:** D_MPS usa T/C ratio — hormonal. Não inclui estado estrutural musculoesquelético. CK elevada seria um input adicional relevante.

### 4.11 Medicação — CONFUNDIDOR NÃO CONTROLADO

**O que é:** Medicação crónica ou aguda que interfere directamente com os biomarcadores da Fase 5.
**Casos críticos documentados nos TC/MA:**
- AINEs (ibuprofeno, aspirina) → degradam TJ intestinais (MA §6.3) → D_Intestino subestimado
- Beta-bloqueadores → suprimem HRV baseline → D_HPA, D_SNC e D_Sono sistematicamente distorcidos
- Corticosteroides exógenos → sobrepõem-se ao cortisol endógeno → D_HPA inválido
- Contracetivos orais → alteram SHBG e testosterona basal → D_HPG inválido
**Gap:** Nenhum dos algoritmos tem um flag de medicação. Para implementação clínica, a Fase 4 deve ter um campo `medication_active[]` que desactive ou corrija os D_i afectados.

---

## SÍNTESE DO PASSO 3

| Categoria de Gap | Número de Itens | Impacto Máximo | Acção Necessária |
|---|---|---|---|
| Thresholds **[DOC]** confirmados | 28 | — | Nenhuma |
| Thresholds **[INFERIDO]** parcialmente suportados | 12 | Médio | Revisão pelo Founder vs. literatura |
| Thresholds **[INVENÇÃO]** sem base documental | 11 | Alto | Investigação de literatura médica obrigatória |
| Coeficientes mágicos sem fórmula clínica | 7 | Muito Alto | Calibração com dados primários ou revisão sistemática |
| Variáveis externas não mapeadas | 11 | Variado | Priorização pelo Founder antes de Fase 6 |

**Bloqueios de implementação identificados:**

| ID | Bloqueio | Severidade |
|---|---|---|
| B3-01 | ALS_base: função de composição da Fase 4 não definida matematicamente | **CRÍTICO** — sem isto, ALS_integrado não é computável |
| B3-02 | rT3/T3 vs FT3/rT3: inconsistência de ratio entre TC e MA | **CRÍTICO** — requer decisão do Founder antes de qualquer implementação |
| B3-03 | A_L3 = 2.3 sem função de probabilidade de hipoglicémia por glicogénio | **ALTO** — Loop 3 não é computável de forma robusta |
| B3-04 | α = 0.002 extrapolado de Cushing, não de overtraining atlético | **ALTO** — Loop 4 pode ter timescale errada por factor 10× |
| B3-05 | δ (Loop 5) completamente indefinido | **MÉDIO** — Loop 5 não é computável até δ ser estimado |
| B3-06 | GC_resistance não modelada para OTS State 3+ | **MÉDIO** — erros sistemáticos de D_i para atletas mais graves |

**Pontas abertas para Fase 6 (mandatórias por Override 2):**

1. D_i(t) são tensores de input para o MPC da Fase 6 — a estrutura de input da Fase 6 deve ser compatível com o vector de 9 dimensões produzido pela Fase 5.
2. OTS_state(t) é um estado discreto que a Fase 6 usa para seleccionar entre modelos Banister distintos (cada estado tem curvas de resposta dose-treino diferentes).
3. O flag `GC_resistance` (B3-06) deve ser transmitido à Fase 6 para que o MPC não prescreva redução de cortisol por via de treino quando o problema é resistência ao GR.
4. A distinção stress-de-treino vs. stress-de-vida (gap 4.8) é informação que a Fase 6 precisa para calibrar correctamente os seus parâmetros de ajuste de volume.
5. Glicogénio estimado (gap 4.1) é uma variável que a Fase 6 naturalmente controla — o loop pode fechar-se correctamente se a Fase 6 passar a estimativa de glicogénio para o State Cache para a Fase 5 consumir no ciclo seguinte.
