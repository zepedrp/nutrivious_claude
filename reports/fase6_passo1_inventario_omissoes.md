# FASE 6 — PASSO 1: INVENTÁRIO, OMISSÕES E ERROS
**Motor Preditivo / MPC / Banister Expandido**
Data: 2026-05-21 | Versão: 1.0 | Status: RAIO-X COMPLETO

---

## PREÂMBULO

Este documento audita os dois ficheiros-fonte da Fase 6:
- **Doc 1**: `nutrivious_bos_fase6_mpc_prescricao.md`
- **Doc 2**: `Nutrivious_BOS_Phase6_Motor_Preditivo.md`

Inputs recebidos da Fase 5 (contexto estabelecido):
`Phase5_Evaluation(t)` = { D_i(t) ∈ [0,1]⁹, IIG(t) ∈ ℝ, ALS_integrado(t) ∈ [0,1], OTS_state(t) ∈ {0,1,2,3}, flags_bloqueio ∈ {Red, Amber, Green}⁹ }

---

## SECÇÃO 1 — INVENTÁRIO MATEMÁTICO REAL

### 1.1 Função de Custo MPC — TRÊS FORMULAÇÕES ENCONTRADAS

**Formulação A (Doc 1, Parte I.1) — Lagrangiana + Custo Terminal:**

$$\min_{u(t)} J = \sum_{k=0}^{N-1} L(x(t+k), u(t+k)) + \Phi(x(t+N))$$

sujeito a:
- $x(t+k+1) = f(x(t+k), u(t+k))$ — dinâmica do sistema
- $u_{\min} \leq u(t+k) \leq u_{\max}$ — restrições de actução
- $g(x(t+k), u(t+k)) \leq 0$ — restrições clínicas

Onde $L(\cdot)$ = lagrangiana (não explicitada), $\Phi(\cdot)$ = custo terminal (não explicitado).

---

**Formulação B (Doc 2, Parte III.3) — Multi-Objectivo:**

$$J(u) = \sum_{k=0}^{N} \left[ -w_1 P(t+k|t) + w_2 \cdot \text{ALS}(t+k|t) + w_3 \cdot R_{\text{lesão}}(t+k|t) + w_4 \Delta u(t+k)^T R \Delta u(t+k) \right]$$

Componentes definidas no mesmo documento:

$$P(t+k|t) = \sum_i \omega_i^{\text{evento}} \times T_{\text{real}}(S_i, t+k|t)$$

$$\text{ALS}(t+k|t) = \sum_{\text{sistema}} (1 - \text{RI}_{\text{sistema}}) \times \text{peso} \times k_{\text{histórico}}$$

$$R_{\text{lesão}}(t+k|t) = \prod_{\text{tecido}} P_{\text{lesão,tecido}}(\text{carga}, \text{estado}, t+k)$$

---

**Formulação C (Doc 2, Parte V.18) — Quadrática de Rastreamento:**

$$\min_u J = \sum_{k=1}^{N} \left[ \|x(t+k|t) - x^{\text{ref}}(t+k)\|^2_Q + \|\Delta u(t+k)\|^2_R + \lambda_{\text{ALS}} \cdot \text{ALS}(t+k|t) \right]$$

Onde $Q \in \mathbb{R}^{18 \times 18}$ = matriz de peso dos estados, $R \in \mathbb{R}^{m \times m}$ = matriz de peso das acções, $x^{\text{ref}}$ = trajectória de referência.

---

### 1.2 Vector de Estado

**Doc 2, Parte II.2:**

$$x(t) \in \mathbb{R}^{18} = [F_{NM}, F_{ad,NM}, F_{Met}, F_{ad,Met}, F_{Neural}, F_{ad,Neural}, F_{Horm}, F_{ad,Horm}, F_{Estrutural}, F_{ad,Estrutural}, F_{Imune}, F_{ad,Imune}, G_{Glic}, G_{PCr}, G_{Líp}, T_{central}, \text{ALS}_{acum}, E_{epi}]$$

Interpretação por bloco:
- Posições 1-12: pares (Fitness, Fadiga) dos 6 compartimentos Banister
- Posições 13-15: substratos energéticos (glicogénio, PCr, lípidos)
- Posição 16: temperatura central
- Posição 17: ALS acumulado
- Posição 18: estado epigenético

---

### 1.3 Arquitectura de Dados MPC

**Doc 2:**

$$D_{MPC}(t) = \{G, E_{epi}(t)\} \cup \{VSF(t)\} \cup \{IIG(t)\} \cup \{\hat{x}_{FF}(t)\}$$

Onde:
- $G$ = perfil genético (constante)
- $E_{epi}(t)$ = estado epigenético dinâmico
- $VSF(t)$ = Vector de Saúde Funcional (fonte não definida)
- $IIG(t)$ = Índice de Integridade Geral (da Fase 5)
- $\hat{x}_{FF}(t)$ = estimativa do estado pelo filtro de Kalman

---

### 1.4 Modelo Banister Expandido — DUAS FORMULAÇÕES

**Formulação Discreta (Doc 1):**

$$F_i(t+1) = F_i(t) \times e^{-1/\tau_{f,i}} + k_{f,i} \times \text{Dose}_i(t) \times G_i \times E_i(t)$$

$$\text{Fad}_i(t+1) = \text{Fad}_i(t) \times e^{-1/\tau_{F,i}} + k_{F,i} \times \text{Dose}_i(t) \times G_i \times E_i(t)$$

Onde $G_i$ = modificador genético (constante), $E_i(t)$ = modificador epigenético (dinâmico).

**Performance por compartimento:**

$$\text{Perf}_i(t) = F_i(t) - \text{Fad}_i(t)$$

---

**Formulação ODE Contínua (Doc 2):**

$$\frac{dF_i}{dt} = -\frac{F_i}{\tau_{f,i}} + k_{f,i} \times \text{TRIMP}_i(t)$$

$$\frac{d\text{Fad}_i}{dt} = -\frac{\text{Fad}_i}{\tau_{F,i}} + k_{F,i} \times \text{TRIMP}_i(t)$$

**Nota:** Esta formulação omite $G_i$ e $E_i(t)$ presentes na Formulação Discreta.

---

### 1.5 Parâmetros Temporais por Compartimento

| Compartimento | τ_F (Fadiga) | τ_f (Fitness) |
|---|---|---|
| Neuromuscular | 1–3 dias | 14–21 dias |
| Neural Central | 2–5 dias | 21–42 dias |
| Metabólico | 1–4 dias | 28–56 dias |
| Hormonal | 3–7 dias | 42–84 dias |
| Estrutural | 5–15 dias | 60–180 dias |
| Imune | 0.5–2 dias | 7–14 dias |

---

### 1.6 Timing de Supercompensação

$$t^*_{\text{super},i} = \frac{\tau_{F,i} \times \tau_{f,i}}{\tau_{F,i} - \tau_{f,i}} \times \ln\!\left(\frac{k_{F,i} \times \tau_{F,i}}{k_{f,i} \times \tau_{f,i}}\right)$$

**Conflito de evento:** quando os picos de vários compartimentos são incompatíveis, o documento propõe:

$$t_{\text{treino}}^{\text{óptimo}} = \arg\min_t \sum_i w_i^{\text{evento}} \times |t - t^*_{\text{super},i}|$$

---

### 1.7 Curva Hormética e Dose Mínima Efectiva

**Curva geral:**

$$\Delta\text{Adaptação}(d) = A \times d \times e^{-\beta d} - C_{\text{alostático}}(d)$$

Pico em $d^* = 1/\beta$; DME ≈ 0.30 × d*; DMT ≈ 1.5 × d*

---

**DME Formulação 1 (Doc 1):**

$$\text{DME}_i(t) = \text{DME}_{\text{base},i} \times M_i(t) \times \text{ALS\_factor}(t) \times \text{Gap\_factor}_i(t)$$

$$\text{ALS\_factor}(t) = \max\!\left(0.40,\ 1.0 - 0.15 \times \text{ALS}_{\text{normalizado}}(t)\right)$$

$$\text{Gap\_factor}_i = \min\!\left(1.30,\ 1.0 + 0.10 \times \text{Gap}_{\text{normalizado},i}\right)$$

---

**DME Formulação 2 (Doc 2):**

$$\text{DME}(S,t) = \text{DME}_{\text{espécie}}(S) \times \frac{\text{TEA}(S,t)}{\text{TGI}(S)} \times \prod_k \text{RI}_k(t)^{\alpha_k}$$

Onde:
- $S$ = espécie de estímulo (força, potência, VO₂, etc.)
- $\text{TEA}$ = Tolerância ao Estímulo Actual
- $\text{TGI}$ = Tecto Genético Individual
- $\text{RI}_k(t)$ = Índice de Recuperação do sistema $k$
- $\alpha_k$ = expoente de sensibilidade ao sistema $k$

---

### 1.8 TRIMP por Compartimento

$$\text{TRIMP}_{NM} = \sum_{\text{séries}} \left(\frac{\text{Carga}}{1\text{RM}}\right)^2 \times N_{\text{reps}} \times \text{RIR\_factor}$$

$$\text{TRIMP}_{Met} = \int \dot{V}O_2(t) \times IF_{\text{intensidade}}(t)\, dt$$

$$\text{TRIMP}_{Neural} = \sum \left(\frac{v_{\text{barra}}}{v_{\text{barra,max}}}\right)^3 \times N_{\text{acções}} \times \text{Complexidade\_motora}$$

$$\text{TRIMP}_{Estrutural} = \sum F_{\text{impacto}}^{1.8} \times N_{\text{ciclos}}$$

**TRIMP_Hormonal** = **NÃO DEFINIDO**

**TRIMP_Imune** = **NÃO DEFINIDO**

---

### 1.9 Filtro de Kalman (Doc 2, Parte V.21)

**Etapa de Predição:**

$$\hat{x}(t|t-1) = f(\hat{x}(t-1|t-1),\, u(t-1))$$

$$P(t|t-1) = A(t-1)\,P(t-1)\,A(t-1)^T + Q$$

**Etapa de Actualização:**

$$K(t) = P(t|t-1)\,C^T \left[C\,P(t|t-1)\,C^T + R\right]^{-1}$$

$$\hat{x}(t|t) = \hat{x}(t|t-1) + K(t)\left[y(t) - C\,\hat{x}(t|t-1)\right]$$

$$P(t|t) = (I - K(t)\,C)\,P(t|t-1)$$

---

### 1.10 Actualização de Parâmetros (Doc 1, Parte XI)

Descrita como "Bayesiana" mas implementada como RLS (Recursive Least Squares):

$$\theta_{\text{novo}} = \theta_{\text{velho}} + K_{\text{param}} \times \varepsilon$$

Onde $\varepsilon = y_{\text{observado}} - y_{\text{previsto}}$ e $K_{\text{param}}$ = ganho de aprendizagem (escalar, não definido).

---

### 1.11 Prescrição Prática — Fórmulas Numéricas (Doc 1)

**Força (séries):**

$$\text{Sets}_{DME} = \text{round}\!\left(3 + 2 \times \min(1, \text{Gap}_{\text{força,norm}}) \times M_{NM}(t)\right)$$

**Intensidade (RIR alvo):**

$$\text{RIR}_{\text{alvo}} = \max\!\left(1,\ 4 - \text{round}\!\left(2 \times \text{Prontidão}_{NM,\text{norm}}\right)\right)$$

**Cardio (minutos Z2/semana):**

$$\text{Min}_{Z2} = \text{round}\!\left(150 \times (1 + 0.5 \times \text{Gap}_{VO_2,\text{norm}}) \times \text{ALS\_factor}\right)$$

**Nutrição:**

$$\text{TDEE} = \text{RMR} \times \text{PAL\_factor} + \text{TEE}_{\text{treino}}$$

$$\text{RMR} \leftarrow \text{Mifflin-St Jeor}$$

$$\text{EA}_{\text{alvo}} = 45\ \text{kcal/kg\_FFM}$$

**Dívida de sono (14 dias):**

$$\text{Dívida}_{\text{sono}}(t) = \sum_{i=t-14}^{t} \max\!\left(0,\ TST_{\text{genético}} - TST_{\text{real}}(i)\right)$$

---

### 1.12 Cinética de Glicogénio

ODE bilinear (forma funcional descrita, coeficientes não totalmente explicitados):
- $k_1 \approx 2 \times k_2$ (Doc 1)
- Janela óptima CHO pós-treino: 1.0–1.2 g/kg/h nas primeiras 4 horas

---

### 1.13 MPS — Função de Hill

$$\text{MPS}_{\text{refeição}} = \text{MPS}_{\max} \times \frac{d_{PRO}^n}{d_{PRO}^n + K_m^n}$$

Com $K_m \approx 0.24\ \text{g/kg de músculo}$, $n \approx 2$.

**Decaimento de sensibilidade mTORC1:**

$$\text{Sensibilidade}(t) = \text{Sensibilidade}_{\max} \times e^{-\lambda \cdot t_{\text{pós-ex}}}$$

---

### 1.14 Horizonte Duplo — Hierarquia Temporal

$$u^*_{\text{táctico}}(t) = \arg\min_u J_{24h}\!\left(u \;\big|\; x(t),\, u^*_{\text{estratégico}}(t \to t+28)\right)$$

Onde $u^*_{\text{estratégico}}$ é calculado numa iteração exterior de horizonte 28 dias.

---

## SECÇÃO 2 — AUDITORIA DE ERROS E OMISSÕES

### ERR-01 [CRÍTICO] — Três Funções de Custo Incompatíveis

As Formulações A, B e C (§1.1) não são equivalentes. São estruturalmente diferentes:

| | Forma | Tracking | ALS | R_lesão | ΔuᵀRΔu |
|---|---|---|---|---|---|
| A | Lagrangiana abstracta | implícito em L | implícito | implícito | implícito |
| B | Multi-objectivo explícito | via P(t+k) | directo | directo | directo |
| C | Quadrática de rastreamento | ‖x−x^ref‖²_Q | via λ_ALS | ausente | ‖Δu‖²_R |

**Consequência:** implementar qualquer uma sem decidir qual é a canónica produz optimizador errado. A Formulação B é a mais completa mas usa pesos $w_1\ldots w_4$ sem valores ou critério de tuning. A Formulação C é mais tratável numericamente mas perde o termo de risco de lesão.

---

### ERR-02 [CRÍTICO] — Filtro de Kalman Linear Denominado EKF

O documento (Doc 2, Parte V.21) rotula o filtro como "Extended Kalman Filter", mas as equações apresentadas são as do **Kalman Filter linear clássico**:
- Predição usa $A(t-1)$, a jacobiana linearizada — não a função $f$ não-linear aplicada à distribuição
- Não há step de linearização explícito

Um EKF verdadeiro requereria:
$$A(t) = \left.\frac{\partial f}{\partial x}\right|_{\hat{x}(t|t),\, u(t)}$$

e a propagação não-linear $\hat{x}(t|t-1) = f(\hat{x}(t-1|t-1), u(t-1))$ seguida da linearização apenas para o passo de covariância. O documento apresenta o KF linear com notação EKF. Com um estado 18-dimensional não-linear, o KF linear produzirá estimativas enviesadas.

---

### ERR-03 [CRÍTICO] — Duas Formulações DME Incompatíveis

Doc 1 (§1.7, Formulação 1): DME escala com $M_i(t)$ (modificador de prontidão neuromuscular), ALS\_factor e Gap\_factor — três factores multiplicativos sobre um valor base fixo.

Doc 2 (§1.7, Formulação 2): DME escala com TEA/TGI (ratio de capacidade actual vs genética) e um produto de índices de recuperação com expoentes $\alpha_k$.

As duas formulações têm dimensões conceptuais diferentes e não convergem para o mesmo resultado numérico. Não existe nenhuma indicação de qual usar para cada tipo de estímulo.

---

### ERR-04 [CRÍTICO] — TRIMP Hormonal e Imune Ausentes

O modelo Banister requer $\text{TRIMP}_i$ para cada um dos 6 compartimentos. O documento define TRIMP para 4: Neuromuscular, Metabólico, Neural, Estrutural. **Os compartimentos Hormonal e Imune não têm TRIMP definido.**

Sem $\text{TRIMP}_{Horm}$ e $\text{TRIMP}_{Imune}$, os dois compartimentos não podem ser actualizados pelo modelo de Banister. Toda a dinâmica hormonal e imune do estado de 18 dimensões fica sem driver de input.

---

### ERR-05 [CRÍTICO] — ALS_normalizado Nunca Definido

A Formulação DME 1 usa $\text{ALS}_{\text{normalizado}}(t)$ como argumento de ALS\_factor. Este valor nunca é definido:
- Qual é o domínio? [0,1]? [0,∞)?
- Normalizado relativamente a quê? Valor de referência? Histórico pessoal? Percentil populacional?

A Fase 5 emite `ALS_integrado(t) ∈ [0,1]`. Provavelmente $\text{ALS}_{\text{normalizado}} = \text{ALS}_{\text{integrado}}$, mas isto não está escrito em nenhum dos dois documentos.

---

### ERR-06 [ALTO] — Duas Formulações Banister Incompatíveis

Formulação Discreta (Doc 1) inclui $G_i$ (genético) e $E_i(t)$ (epigenético) como multiplicadores.
Formulação ODE (Doc 2) omite ambos.

Se o vector de estado $x(t) \in \mathbb{R}^{18}$ inclui $E_{epi}$ na posição 18, então $E_i(t)$ é trackado — mas a equação ODE que o deveria usar não o inclui. Inconsistência interna ao Doc 2.

---

### ERR-07 [ALTO] — Matrizes Q e R da Formulação C Não Definidas

A Formulação C (§1.1) requer:
- $Q \in \mathbb{R}^{18 \times 18}$: 324 parâmetros (ou pelo menos 18 diagonais se $Q = \text{diag}$)
- $R \in \mathbb{R}^{m \times m}$: dimensão $m$ (número de acções) não especificada

Não há critério de tuning, valores sugeridos, ou referência a literatura de controlo ótimo para inicializar estas matrizes.

---

### ERR-08 [ALTO] — Matriz de Observação C Não Definida

O filtro de Kalman requer $C \in \mathbb{R}^{p \times 18}$ onde $p$ = dimensão do vector de observações $y(t)$.

Não está definido:
- Quais variáveis do estado de 18 dimensões são directamente observáveis (wearables, biomarkers)?
- Quais são latentes (Fitness compartimento Estrutural, estado epigenético, etc.)?
- Qual é o mapeamento $y(t) = Cx(t) + \text{ruído}$?

Sem $C$, o filtro de Kalman não pode ser implementado.

---

### ERR-09 [ALTO] — Trajectória de Referência x^ref Não Definida

A Formulação C minimiza $\|x - x^{\text{ref}}\|^2_Q$. Mas $x^{\text{ref}}(t+k)$ — a trajectória de estado desejada ao longo do horizonte — nunca é especificada:
- É definida pelo utilizador (objectivos de performance do atleta)?
- É calculada pela supercompensação óptima (§1.6)?
- É um vector constante (estado de saúde de referência)?

A resposta altera fundamentalmente o comportamento do optimizador.

---

### ERR-10 [ALTO] — Expoentes $\alpha_k$ da DME Formulação 2 Não Definidos

$$\text{DME}(S,t) = \text{DME}_{\text{espécie}}(S) \times \frac{\text{TEA}(S,t)}{\text{TGI}(S)} \times \prod_k \text{RI}_k(t)^{\alpha_k}$$

Os $\alpha_k$ determinam a sensibilidade da DME ao estado de recuperação de cada sistema $k$. Sem valores ou critério de definição, o produto $\prod_k \text{RI}_k(t)^{\alpha_k}$ é incalculável. O documento mostra apenas um exemplo numérico sem explicitar os 6 valores de $\alpha_k$ correspondentes aos 6 compartimentos.

---

### ERR-11 [MÉDIO] — Performance Global Não Agregada

O modelo Banister produz $\text{Perf}_i(t) = F_i(t) - \text{Fad}_i(t)$ para cada um dos 6 compartimentos. A Formulação B da função de custo requer $P(t+k|t) = \sum_i \omega_i^{\text{evento}} \times T_{\text{real}}(S_i, t+k|t)$.

Não existe nenhuma função que mapeie os 6 valores $\text{Perf}_i(t)$ para um escalar de performance global observável. $T_{\text{real}}(S_i, t+k|t)$ não está definido em termos das variáveis de estado.

---

### ERR-12 [MÉDIO] — R_lesão Como Produto Mascara Riscos

$$R_{\text{lesão}} = \prod_{\text{tecido}} P_{\text{lesão,tecido}}$$

Um produto de probabilidades colapsa para zero quando qualquer factor individual é próximo de zero — o risco de lesão global pode ser dominado por um único tecido de baixo risco, mascarando outros tecidos em risco real. A formulação correcta para risco máximo seria:

$$R_{\text{lesão}} = 1 - \prod_{\text{tecido}} (1 - P_{\text{lesão,tecido}})$$

ou, mais conservadoramente:

$$R_{\text{lesão}} = \max_{\text{tecido}} P_{\text{lesão,tecido}}$$

---

### ERR-13 [MÉDIO] — Abordagem Estocástica do MPC Não Especificada

O documento menciona incerteza paramétrica e ruído de processo/observação (matrizes $Q$, $R$ do filtro), mas não especifica se o MPC é:
- **Determinístico** com estados estimados (use $\hat{x}$ como estado certo)
- **Stochastic MPC** com propagação de incerteza explícita no horizonte
- **Tube MPC** com tubos de robustez

Esta escolha afecta drasticamente a complexidade computacional e a garantia de satisfação de restrições.

---

### ERR-14 [MÉDIO] — Contradição na Cinética de PCr

Doc 1 afirma que a PCr repõe em ~90 segundos entre séries. Esta é a cinética de reposição PARCIAL (aproximadamente 50–70%). A reposição COMPLETA requer 3–5 minutos. O valor de 90s é usado para justificar intervalos de descanso curtos, mas a equação do vector de estado (posição $G_{PCr}$) usa um compartimento de recuperação que, se calibrado para 90s, subestimará o défice cumulativo em treinos de alto volume.

---

### ERR-15 [MÉDIO] — Interface Horizonte Duplo: Restrição Hard vs Soft Não Especificada

$$u^*_{\text{táctico}} = \arg\min_u J_{24h}(u \mid x(t), u^*_{\text{estratégico}})$$

Não está especificado se $u^*_{\text{estratégico}}$ entra como:
- **Restrição hard**: $u_{\text{táctico}} \subseteq \mathcal{U}(u^*_{\text{estratégico}})$ — o táctico não pode desviar do estratégico
- **Restrição soft**: $u^*_{\text{estratégico}}$ aparece como termo de penalização em $J_{24h}$
- **Ponto inicial**: $u^*_{\text{estratégico}}$ apenas inicializa o solver táctico

A escolha determina o comportamento quando as condições agudas (febre, lesão) contradizem o plano estratégico.

---

### ERR-16 [MÉDIO] — Actualização "Bayesiana" É RLS Simples

Doc 1, Parte XI descreve "actualização Bayesiana dos parâmetros $\tau_f$, $k_f$" mas a equação apresentada:

$$\theta_{\text{novo}} = \theta_{\text{velho}} + K_{\text{param}} \times \varepsilon$$

é o algoritmo de **Recursive Least Squares** (ou equivalentemente, filtro de Kalman para parâmetros estáticos). Não há distribuição posterior, não há prior conjugado, não há inferência de incerteza paramétrica. A diferença importa: Bayesiano verdadeiro quantifica a incerteza sobre $\theta$; RLS minimiza erro quadrático com ganho fixo e pode divergir com dados não-estacionários.

---

## SECÇÃO 3 — O CONTRATO DE INPUT

### 3.1 O Que a Fase 6 Realmente Usa (vs. o Que o Documento Declara)

A fórmula de dados MPC (§1.3) é:

$$D_{MPC}(t) = \{G, E_{epi}(t)\} \cup \{VSF(t)\} \cup \{IIG(t)\} \cup \{\hat{x}_{FF}(t)\}$$

Esta fórmula referencia explicitamente apenas **IIG(t)** como output da Fase 5. Mas a análise das fórmulas de prescrição (§1.11) e da função de custo (§1.1, Formulação B) revela que a Fase 6 **implicitamente depende** de inputs adicionais da Fase 5 que nunca são nomeados no contrato formal:

| Output Fase 5 | Usado em Fase 6? | Onde aparece | Declarado em D_MPC? |
|---|---|---|---|
| `IIG(t)` | Sim | D_MPC explícito | ✅ Sim |
| `D_i(t) ∈ [0,1]⁹` | Sim | M_i(t) nas fórmulas de prescrição; RI_k(t) na DME Formulação 2 | ❌ Não |
| `ALS_integrado(t)` | Sim | ALS_factor em DME Formulação 1; λ_ALS na Formulação C | ❌ Não |
| `OTS_state(t) ∈ {0,1,2,3}` | Implicitamente | Restrições clínicas da Formulação A | ❌ Não |
| `flags_bloqueio` | Implicitamente | Restrições $g(x,u) \leq 0$ da Formulação A | ❌ Não |

---

### 3.2 Contrato de Input Mínimo Necessário para Operação

Para que os motores da Fase 6 possam funcionar, a Fase 5 deve fornecer:

```
Phase5_Evaluation(t):
  ├── D_i(t)           ∈ [0,1]⁹   # 9 dimensões: HPA, HPT, HPG, SNC, Sono,
  │                                  #   Metabolismo, Intestino, Imune, MPS
  ├── IIG(t)           ∈ [0,1]    # Índice de Integridade Geral
  ├── ALS_integrado(t) ∈ [0,1]    # Alostatic Load Score
  ├── OTS_state(t)     ∈ {0,1,2,3} # 0=saudável, 1=funcional, 2=não-func, 3=OTS
  └── flags_bloqueio   ∈ {R,A,G}⁹  # por sistema: Red=bloqueio, Amber=cautela, Green=ok
```

---

### 3.3 Mapeamento de Inputs para Motores da Fase 6

| Motor Fase 6 | Inputs Fase 5 Necessários | Inputs Fase 6 Próprios |
|---|---|---|
| Filtro de Kalman — Actualização | `IIG(t)` como componente de `y(t)` | `u(t-1)`, wearable biomarkers |
| DME Formulação 1 | `ALS_integrado(t)` → ALS\_factor; `D_i(t)` → M_i(t) | DME_base_i, Gap_i |
| DME Formulação 2 | `D_i(t)` → RI_k(t) | DME_espécie, TEA, TGI, α_k |
| Função Custo B | `ALS_integrado(t)` → ALS(t+k\|t) | ω_i^evento, R_lesão |
| Função Custo C | `ALS_integrado(t)` → λ_ALS×ALS | Q, R, x^ref |
| Banister Update | Nenhum directo | TRIMP_i(t) calculado da sessão |
| Prescrição Séries | `D_NM(t)` → M_NM(t), Prontidão_NM | Gap_força |
| Prescrição Cardio | `ALS_integrado(t)` → ALS\_factor | Gap_VO₂ |
| Restrições Clínicas | `OTS_state(t)`, `flags_bloqueio` | limites u_max/u_min |

---

### 3.4 Variáveis Implícitas Nunca Declaradas no Contrato

As seguintes variáveis são usadas nas fórmulas da Fase 6 mas a sua origem não está declarada em nenhum dos dois documentos:

1. **VSF(t)** — Vector de Saúde Funcional: referenciado em D_MPC mas nunca definido; presumivelmente subset de D_i(t)
2. **$\hat{x}_{FF}(t)$** — estimativa do filtro: output do próprio filtro de Kalman; entrada circular (usa output anterior como input)
3. **Gap_força_normalizado** — diferença entre força actual e objectivo; não declarado como input da Fase 5 nem calculado internamente
4. **Gap_VO₂_normalizado** — idem para VO₂máx
5. **Prontidão_NM_normalizada** — presumivelmente $D_{NM}(t)$ mas não confirmado explicitamente
6. **M_i(t)** — modificador de prontidão: referenciado na DME Formulação 1 mas não mapeado para $D_i(t)$; são a mesma variável?
7. **RI_k(t)** — Índice de Recuperação: referenciado na DME Formulação 2; relação com $D_i(t)$ não declarada

---

### 3.5 Bloqueadores de Implementação Identificados

| ID | Severidade | Descrição |
|---|---|---|
| B6-01 | CRÍTICO | Qual das 3 funções de custo é a canónica? Decisão de Founder obrigatória. |
| B6-02 | CRÍTICO | KF linear vs EKF: escolher e implementar a linearização correcta para estado ℝ¹⁸ não-linear. |
| B6-03 | CRÍTICO | TRIMP_Hormonal e TRIMP_Imune não definidos; 2/6 compartimentos Banister inoperacionais. |
| B6-04 | CRÍTICO | ALS_normalizado não definido; DME Formulação 1 incalculável até clarificação. |
| B6-05 | CRÍTICO | D_MPC não inclui D_i(t) nem ALS_integrado(t); contrato de interface Fase 5→6 incompleto. |
| B6-06 | ALTO | Q, R da Formulação C: 324+ parâmetros sem critério de tuning. |
| B6-07 | ALTO | Matriz de observação C: quais das 18 dimensões do estado são observáveis? |
| B6-08 | ALTO | x^ref: como é calculada a trajectória de referência de 18 dimensões? |
| B6-09 | ALTO | α_k (DME Formulação 2): 6 expoentes de sensibilidade sem valores ou critério. |
| B6-10 | ALTO | Banister Formulação ODE omite G_i e E_i(t); inconsistente com Doc 1. Qual é a canónica? |
| B6-11 | MÉDIO | Interface horizonte duplo: restrição hard vs soft vs ponto inicial — não especificado. |
| B6-12 | MÉDIO | R_lesão como produto mascara riscos individuais; fórmula a rever. |
| B6-13 | MÉDIO | Actualização de parâmetros é RLS, não Bayesiana; ganho K_param não definido. |
| B6-14 | MÉDIO | Performance global de 6 compartimentos → escalar: função de agregação não definida. |
| B6-15 | MÉDIO | VSF(t), M_i(t), RI_k(t): relação com D_i(t) não declarada; risco de duplicação de cálculo. |
| B6-16 | BAIXO | PCr cinética: 90s é reposição parcial; calibrar τ_F para 3-5 min de reposição completa. |

---

## SUMÁRIO EXECUTIVO

**16 erros/omissões identificados:** 5 CRÍTICOS, 5 ALTOS, 5 MÉDIOS, 1 BAIXO.

**Problema central:** os dois documentos da Fase 6 foram escritos em momentos diferentes com modelos mentais ligeiramente diferentes. O resultado é três funções de custo incompatíveis, dois modelos Banister incompatíveis, e dois modelos DME incompatíveis. Antes de qualquer implementação, o Founder deve designar **uma** formulação canónica para cada.

**Problema de interface:** o contrato formal Fase 5 → Fase 6 (a fórmula D_MPC) está incompleto. Referencia apenas IIG(t). Para funcionar, a Fase 6 necessita de D_i(t), ALS_integrado(t), OTS_state(t) e flags_bloqueio. Este gap é silencioso — não causa erro explícito, apenas produz prescrições incorrectas quando os motores caem-back para valores default.

**Pontas abertas (per OVERRIDE 2):** as interfaces Fase 6 → Fase 7 (execução/log da sessão → State Cache) não foram auditadas aqui. O documento fase6 menciona SessionRecord mas o schema não está definido neste passo.

---

*Raio-X completo. Próximo passo: FASE 6 — PASSO 2 (Resolução das Incompatibilidades e Definição do Modelo Canónico).*
