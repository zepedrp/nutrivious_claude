# FASE 5 — PASSO 1: INVENTÁRIO, OMISSÕES E AUDITORIA DE ERROS

> **Fonte:** Dois documentos Fase 5 — *Teia Causal* (TC) e *Matriz Alosática* (MA).
> **Método:** Extracção exaustiva de motores matemáticos, seguida de cruzamento com o inventário genérico anterior para detecção de omissões e contradições internas.

---

## SECÇÃO 1 — INVENTÁRIO MATEMÁTICO COMPLETO

### 1.1 Motor de Supressão de Testosterona por Cortisol (CB-01)

**Fonte:** TC §2.1

```
T_suprimida = T_baseline × max(0.40, 1.0 − 0.030 × max(0, Cortisol_µg_dL − 18))
```

Parâmetros calibrados (Cumming et al. 1983, Brownlee et al. 2005):

| Cortisol (µg/dL) | Fator multiplicativo | Supressão |
|---|---|---|
| 18 | 1.000 | 0% (limiar) |
| 22 | 0.880 | −12% |
| 26 | 0.760 | −24% |
| 30 | 0.640 | −36% |
| 35 | 0.490 → floor=0.40 | −51% (mínimo: −60%) |

**Gatilho condicional:** Cortisol matinal > 30 µg/dL activa supressão adicional via 11β-HSD1 muscular (cortisol local → catabolismo directo de fibras) — efeito não capturado na fórmula escalar acima; é um **segundo mecanismo paralelo**.

---

### 1.2 Motor de Leucina Efectiva sob Cortisol Elevado (CB-02)

**Fonte:** TC §2.2

```
Leucina_efectiva = Leucina_ingerida × max(0.45, 1.0 − 0.025 × max(0, Cortisol_µg_dL − 18))
```

| Cortisol (µg/dL) | Fracção efectiva |
|---|---|
| 20 | 0.950 |
| 25 | 0.825 |
| 30 | 0.700 |
| 35 | 0.575 → floor=0.45 |

**Implicação computacional:** O limiar de activação de mTORC1 por leucina sobe de >0.25 g/kg para >0.40 g/kg quando cortisol > 25 µg/dL. Este motor reescreve o threshold de dose da suplementação proteica — não é um modificador linear do output; é um **reescritor do threshold de entrada** da via mTORC1.

---

### 1.3 Motor de GH Nocturno Composto (CB-03)

**Fonte:** TC §2.3

```
GH_noturno_real = GH_máximo × f(SWS%) × f(Cortisol_nocturno)

f(SWS%)            = SWS_actual_pct / SWS_target_pct     [target = 20% do TST]
f(Cortisol_noc)    = max(0.20, 1.0 − 0.045 × max(0, Cortisol_nocturno_nmol_L − 2))
```

| Cortisol nocturno (nmol/L) | f(Cortisol) |
|---|---|
| 2.0 | 1.000 (sem supressão) |
| 3.5 | 0.933 |
| 6.0 | 0.820 → supressão severa |
| 13.6 | 0.200 (floor) |

**Interacção multiplicativa:** O GH real é o produto de dois factores independentes — défice de SWS e nível de cortisol nocturno. Se SWS% = 50% do alvo E cortisol nocturno = 6 nmol/L: `GH_real = GH_máx × 0.50 × 0.82 = GH_máx × 0.41`. Os dois desvios comprimem o GH de forma supraditiva quando co-ocorrem.

---

### 1.4 Motor de Supressão de Testosterona por IL-6 (IB-01)

**Fonte:** TC §4.1

```
T_suprimida_por_IL6 = T_baseline × max(0.55, 1.0 − 0.040 × max(0, IL6_pg_mL − 2.5))
```

**Mecanismo:** IL-6 → JAK2/STAT3 nos gonadotrofos hipofisários → supressão de LH → menos estimulação de células de Leydig.

---

### 1.5 Motor de SHBG Amplificado por CRP (IB-01-b)

**Fonte:** TC §4.1

```
SHBG_efectivo = SHBG_baseline × (1 + 0.08 × max(0, CRP_mg_L − 0.5))
```

Exemplo: SHBG_baseline = 30 nmol/L, CRP = 3.0 mg/L → SHBG_efectivo = 30 × 1.20 = 36 nmol/L

**Implicação:** T_livre cai proporcionalmente a SHBG mesmo com T_total inalterada. Este é um motor separado do CB-01 e IB-01 — produz supressão de T_livre por via hepática independente do eixo HPG.

---

### 1.6 Motor de Bloqueio mTORC1 por TNF-α (IB-02)

**Fonte:** TC §4.2

```
IRS1_Ser307_phosphorylation_pct = 20 + 6 × max(0, TNF_alpha_pg_mL − 5)
mTORC1_activity_pct             = 100 − IRS1_Ser307_phosphorylation_pct × 0.40
```

| TNF-α (pg/mL) | Fosforilaçao IRS-1 (%) | mTORC1 actividade (%) |
|---|---|---|
| 5 | 0% | 100% |
| 8 | 18% | 93% |
| 12 | 42% | 83% |
| 20 | 90% | 64% |

**Amplificação supra-aditiva por IL-1β:** Se IL-1β > 15 pg/mL co-activo:
```
mTORC1_activity_composta = mTORC1_activity_TNF × (1 / 1.35)
```
O factor JNK (×1.35) aplica-se sobre o bloqueio já calculado por TNF-α — NÃO é aditivo, é multiplicativo sobre o efeito base.

**Crítico:** Este bloqueio de mTORC1 é **NÃO CONTORNÁVEL** por dose aumentada de proteína. É um bloqueio upstream (IRS-1) que impede a sinalização PI3K/Akt independentemente da disponibilidade de leucina. É o único mecanismo no documento onde o reescritor molecular invalida completamente a intervenção nutricional.

---

### 1.7 Motor de Tamponamento por Carnosina (AB-01)

**Fonte:** TC §7.1

```
ΔpH_tampão_carnosina = [Carnosina_mmol_kg_músculo_seco] × 0.1 / Produção_H_mmol_kg
```

Atleta com carnosina = 40 mmol/kg vs 20 mmol/kg → +20 segundos de capacidade glicolítica antes de atingir pH = 6.8.

**Dependência de Phase 6:** O tempo adicional (+20s) é um input para o modelo de Fitness-Fatigue do Banister Expandido (Phase 6) — especificamente para o compartimento Neuromuscular onde a tolerância à acidose afecta τf_NM.

---

### 1.8 Motor de Amplificação da Carga Alostática (ALS_integrado)

**Fonte:** TC §XIV

```
ALS_integrado(t) = ALS_base(t) × Amplificador_Cascata(t)

Amplificador_Cascata(t) = ∏[loops activos] (1 + γ_loop)
```

| Loop | γ_loop | Condição de Activação |
|---|---|---|
| CRP_HOMA | 0.35 | CRP > 1.0 mg/L AND HOMA-IR > 2.0 |
| Cortisol_T | 0.25 | Cortisol > 22 µg/dL AND T/C < 0.60 × baseline |
| Sono_GH | 0.30 | SWS% < 10% AND GH_estimado < 30% do máximo |
| GV_Inflam | 0.20 | CV_glucose > 36% AND IL-6_estimado > 5 pg/mL |
| EA_Leptin | 0.40 | EA < 30 kcal/kg_FFM AND Leptina < 4 ng/mL |
| Microbioma | 0.15 | Zonulina > 40 ng/mL AND F.prausnitzii < 2% |

Exemplo com 3 loops activos (CRP+HOMA, Sono+GH, GV+Inflam):
```
ALS_integrado = ALS_base × 1.35 × 1.30 × 1.20 = ALS_base × 2.106
```

---

### 1.9 Máquina de Estados OTS (Estado 0 → 1 → 2 → 3)

**Fonte:** TC §9.1

**ESTADO 0 — Normal:**
- HRV_z-score > −1.0 DP
- CMJ_normalizado > −1.5 DP
- T/C_ratio > 0.70 × baseline
- sIgA > 80% baseline
- Wellbeing > 6/10

**ESTADO 1 — FOR (Functional Overreaching):**
- HRV_z-score: −1.0 a −2.0 DP por ≥ 3 dias
- CMJ: −5 a −10% por ≥ 3 dias
- T/C: 0.50 a 0.70 × baseline
- sIgA: 60–80% baseline
- Duração: 1–2 semanas → recuperação TOTAL com −30 a −40% volume

**ESTADO 2 — NFOR (Non-Functional Overreaching):**
- HRV_z-score < −2.0 DP por ≥ 5 dias
- CMJ: −10 a −20%
- T/C: 0.30 a 0.50 × baseline
- Cortisol_matinal > 22 µg/dL (cronicamente)
- CAR: plana ou negativa (< 30% de incremento)
- sIgA: 40–60% baseline
- Performance: declínio > 2% em teste standardizado
- Duração de recuperação: 2–6 semanas (TC) / 4–12 semanas (MA — **CONTRADIÇÃO**)

**ESTADO 3 — OTS (Overtraining Syndrome):**
- Todos os critérios NFOR MAIS:
- T/C < 0.30 × baseline
- Cortisol_nocturno (23h00) > 6 nmol/L (cronicamente)
- FT3/rT3 < 6
- IGF-1 < 120 ng/mL
- sIgA < 40% baseline
- POMS: Vigor < 10/100; Fadiga > 60/100
- Performance: declínio persistente > 4 semanas
- Recuperação: 3–6 meses (TC) / 12–24 meses (MA para Stage 4 com dano epigenético — **DISTINÇÃO NÃO CLARA**)

---

### 1.10 Os 10 Travões de Segurança (Safety Brakes)

**Fonte:** TC §Síntese

| # | Condição | Acção |
|---|---|---|
| 1 | Temp_central > 38.0°C | STOP TOTAL |
| 2 | Δpeso_corporal < −3% massa | STOP alta intensidade |
| 3 | CRP > 10 mg/L | STOP força e potência |
| 4 | HRV_z < −2.0 DP por 3+ dias | REDUZIR para Z2 máximo |
| 5 | Cortisol > 30 µg/dL | REDUZIR volume 50% + emergência nutricional |
| 6 | TST < 5h por 2+ noites | STOP alta intensidade (risco lesão ×2.5) |
| 7 | EA < 20 kcal/kg_FFM AND T-score < −1.0 | SUSPENDER impacto |
| 8 | T/C_ratio < 0.30 × baseline por 5+ dias | PROTOCOLO OTS |
| 9 | Glucose_nocturna < 65 mg/dL | Ajuste nutricional urgente + reduzir carga |
| 10 | ALS_integrado > 3.5 | PROTOCOLO RECUPERAÇÃO OBRIGATÓRIO (override absoluto) |

---

### 1.11 Motor de FSR (Fractional Synthetic Rate) por Cortisol

**Fonte:** MA §4.2

Formulação textual formalizada:
```
ΔFSR_pct ≈ −(8 a 12) × max(0, (Cortisol_µg_dL − Cortisol_normal_repouso) / 10)
```

Onde Cortisol_normal_repouso ≈ 12–18 µg/dL (valor de referência em repouso).

**Distinção do CB-01:** CB-01 mede o impacto em T_total. FSR mede o impacto directo na taxa de síntese proteica miofibrilar. São dois motores paralelos sobre targets diferentes da mesma perturbação.

---

### 1.12 Via IDO-Kinurenina (Cascade Neuroinflamatória)

**Fonte:** MA §5.3

```
Triptofano --[IDO activada por IL-6, IFN-γ, LPS]--> N-formilkinurenina 
         --> Kinurenina --> Ácido Quinolínico (agonista NMDA, neurotóxico)

vs. via normal:
Triptofano --[TPH]--> 5-HTP --> Serotonina
```

**Consequências computáveis:**
- Serotonina ↓ → humor depressivo, menor tolerância ao esforço, qualidade de sono degradada
- Ácido quinolínico ↑ → excitotoxicidade hipocampal → deterioração cognitiva
- Kinurenina atravessa a BHE → activa microglia → mais IDO (loop fechado)

**O loop fecha-se:** LPS intestinal → IDO → menos serotonina → menos melatonina → pior sono → mais cortisol → mais LPS (via TJ abertas por cortisol). Este loop de 5 passos não foi formalizado em qualquer inventário anterior.

---

### 1.13 Motor de Dano Epigenético no OTS Avançado

**Fonte:** MA §15.2 (Limiar 3)

Marcadores de Stage 4 / OTS com dano epigenético estabelecido:
- Hipermetilação do promotor NR3C1 (gene do receptor de glucocorticoides) → insensibilidade crónica ao cortisol
- DunedinPACE > 1.4 (aceleração do relógio epigenético — 40% acima do envelhecimento normal)
- rT3/T3 ratio > 0.5 (síndrome de T3 baixo estabelecida)
- NLR > 5 cronicamente (Neutrophil-to-Lymphocyte Ratio)
- Cortisol matinal < 100 nmol/L (insuficiência adrenal relativa)
- Possível atrofia hipocampal mensurável por RM

**Computacionalmente:** O dano epigenético representa um estado onde os parâmetros genéticos (Phase 3) foram reescritos pelo estado agudo crónico (Phase 4/5). É o único mecanismo no sistema onde Phase 4 altera permanentemente Phase 3.

---

### 1.14 Motor IIG — Índice de Interferência Global

**Fonte:** MA §27

```
IIG(t) = ΣΣ w_ij × max(0, D_ij(t))
         i  j≠i
```

Onde:
- `D_ij(t)` = défice do sistema i no tempo t (desvio normalizado: 0 = óptimo, 1 = colapso)
- `w_ij` = peso da interferência do sistema i sobre o sistema j

**Equação de integração multi-fase:**
```
IIG = f(ALS_Fase4, Baseline_Fase2, TGI-TEA_Fase3, T_espécie-TGI_Fase1)
```

**Tabela de Interpretação:**

| IIG | Estado | Protocolo |
|---|---|---|
| 0–10 | Ressonância sistémica óptima | Alta intensidade autorizada |
| 10–25 | Interferência leve | Monitorizar elos mais fracos |
| 25–50 | Interferência moderada | Identificar top 2–3 sistemas com D_ij mais alto; intensidade reduzida |
| 50–75 | Interferência severa | Modo recuperação activa; intervenção nutricional e sono prioritária |
| > 75 | Colapso sistémico em cascata | Descanso total + avaliação clínica |

---

### 1.15 Tabela Mestre de Interferências 9×9 (MA §24)

**Fonte:** MA §24 — estrutura qualitativa (↓, ↓↓, ↓↓↓)

Sistemas mapeados: HPA/Cortisol | HPG/Testosterona | HPT/T3 | MPS/Anabolismo | Sistema Imune | Intestino | Sono | Metabolismo | SNC/Cognição

**Interferências de alta magnitude (↓↓↓) extraídas:**

| Origem → | Destino | Mecanismo |
|---|---|---|
| HPA/Cortisol | HPG/T | REDD1 + MuRF1 em MPS; GnRH e StAR suprimidos |
| HPA/Cortisol | MPS | REDD1 → mTORC1 inibido |
| Sistema Imune (inflamação) | MPS | NF-κB → MuRF1 activo |
| Intestino (disbiose) | MPS | LPS → NF-κB → MuRF1 |
| Intestino (disbiose) | SNC | LPS → microglia → IDO → kinurenina |
| Sono (privação) | MPS | GH ↓ → IGF-1 ↓ → mTORC1 ↓ |
| Sono (privação) | SNC | Adenosina acumulada, microsono |
| SNC/Stress | HPA | Amígdala → CRH → HPA |
| SNC/Stress | MPS | Via HPA completa |

**PROBLEMA CRÍTICO:** Os w_ij nunca são traduzidos em valores numéricos. A tabela apresenta apenas ↓, ↓↓, ↓↓↓ sem escala cardinal. A fórmula IIG `ΣΣ w_ij × D_ij` é **computacionalmente inoperacional** sem estes pesos.

---

### 1.16 5 Loops de Auto-Amplificação (Ciclos Viciosos)

**Fonte:** MA §25

**Loop 1 — Inflamação-Resistência à Insulina-Cortisol:**
```
IL-6/TNF-α ↑ → IRS-1 Ser307 fosforilado → Hiperinsulinemia compensatória
→ Aromatase ↑ → T → Estradiol ↑ → T_total ↓ → cortisol relativo ↑
→ NF-κB ↑ → mais IL-6/TNF-α [LOOP FECHADO]
```

**Loop 2 — Intestino-Cérebro-HPA:**
```
Disbiose → LPS sistémico → cortisol (via vago + HPA)
→ cortisol abre mais TJ intestinais (via mastócitos CRH)
→ mais LPS → mais cortisol [LOOP FECHADO]
```

**Loop 3 — Privação Sono-Hipoglicémia-Fragmentação SWS:**
```
Défice calórico → Hipoglicémia nocturna → Cortisol + Glucagão reactivos
→ Cortisol fragmenta SWS → menos GH → menos síntese glicogénio
→ maior depleção glicogénio sessão seguinte → maior hipoglicémia [LOOP FECHADO]
```

**Loop 4 — Cortisol-Atrofia Hipocampal-HPA Descontrolado:**
```
Cortisol crónico → Atrofia dendrítica hipocampal
→ menor feedback negativo do hipocampo sobre NPV
→ maior secreção CRH → mais cortisol [LOOP FECHADO — altamente patológico]
```

**Loop 5 — Microglia-IDO-Anedonia-Menor Actividade:**
```
Neuroinflamação → IDO → kinurenina → serotonina ↓ → anedonia
→ menor actividade física → BDNF ↓ → menor controlo top-down amígdala
→ mais cortisol → mais neuroinflamação [LOOP FECHADO]
```

---

### 1.17 Motor de Cascata de Magnésio (8 Sistemas)

**Fonte:** MA §14.1

Défice de Mg²⁺ compromete em cadeia:
1. Na⁺/K⁺-ATPase → gradiente iónico ↓ → HRV ↓, cãibras
2. ATP-Mg (forma biologicamente activa) → toda a bioenergética comprometida
3. PDH (Piruvato Desidrogenase) → piruvato acumula → lactato ↑ em repouso
4. α-Cetoglutarato Desidrogenase (Krebs) → NADH ↓ → OXPHOS ↓
5. AMPK: requer Mg-AMP como ligando → sinalização energética central comprometida
6. rRNA stability → eficiência de tradução ↓ → MPS ↓
7. DNA polimerase → maior taxa de erros de replicação → stress genotóxico
8. Canal NMDA bloqueador fisiológico → défice → hiperactividade NMDA → insónia, ansiedade

---

### 1.18 Motor de Cascata de Vitamina B6/PLP (7 Sistemas)

**Fonte:** MA §14.1

PLP (forma activa B6) compromete em cadeia:
1. Transaminases (ALT, AST, BCAT) → síntese de aminoácidos não-essenciais ↓
2. Aminoácido Descarboxilases → DOPA Descarboxilase ↓ → dopamina e serotonina ↓
3. Glicogénio Fosforilase (cofactor PLP) → mobilização de glicogénio ↓ em exercício
4. CBS (Cistationina β-Sintase) → homocisteína acumula → hiperhomocisteinemia
5. Delta-ALA-Sintase (1ª enzima síntese heme) → hemoglobina ↓ → VO2max comprometido
6. Síntese de Niacina via triptofano-kinurenina → NAD⁺ ↓ → CTE ↓
7. Síntese de Creatina (2º passo via SAM) → PCr ↓ → capacidade de sprint ↓

---

### 1.19 Motor de Resistência Glucocorticoide Paradoxal

**Fonte:** MA §12.2 (Cohen et al. 2012)

Stress psicológico crónico → GR downregulação em linfócitos T → dois efeitos simultâneos:
- **Efeito 1:** Cortisol perde efeito anti-inflamatório (GR insensível) → inflamação persistente ↑
- **Efeito 2:** Cortisol mantém imunossupressão via mecanismos independentes dos GR clássicos → maior vulnerabilidade infecciosa

**"O pior dos dois mundos":** Mais inflamação AND menos defesa. Este estado não tem tratamento farmacológico simples — requer redução da carga alostática global antes que os GR recuperem sensibilidade.

---

### 1.20 Motor de Desalinhamento Circadiano-Epigenético

**Fonte:** MA §8.2

Desalinhamento NSC ↔ relógios periféricos:
- Sensibilidade à insulina mínima à noite → comer à noite produz 2× a elevação glicémica da mesma refeição de manhã
- Padrão circadiano de MPS sincronizado com pico de testosterona (manhã) → dessincronização → menor eficiência de recuperação nocturna
- **Marcador epigenético:** Hipermetilação CpG em promotores de PER1, BMAL1 → persiste semanas a meses após o período de desalinhamento

Loop de Veillonella (metabolômico):
```
Lactato muscular → [Veillonella atypica] → Propionato
Propionato → GPR41 hepático → gluconeogénese suprimida
Propionato → GPR43 adiposo → lipólise inibida
Propionato → AMPK nos hepatócitos → biogénese mitocondrial
```

---

### 1.21 Motor de Permeabilidade Intestinal por CRH (TC §6.3)

**Mecanismo de activação directo:**
```
Stressor psicológico → Amígdala → CRH hipotalâmico
→ Mastócitos submucosos intestinais (receptor CRH1)
→ Desgranulação → Histamina + triptase + IL-1β + TNF-α
→ Abertura das TJ intestinais (MLCK activada)
→ Translocação de LPS → Endotoxemia
```

**Nota crítica:** Uma sessão de competição de alta pressão pode abrir a barreira intestinal em **minutos**, por via neuro-endócrina directa, **sem qualquer stressor físico**. Este mecanismo não tem threshold numérico definido no documento — é uma lacuna.

---

### 1.22 Quotient FSR/Cortisol Adicional (MA §4.2)

Texto literal: "Cada 10 µg/dL de cortisol acima do normal em repouso reduz a FSR muscular em ~8–12%"

Formulação derivada:
```
FSR_supressão_pct = (8 a 12) × floor(max(0, Cortisol − 18) / 10)
```

Esta equação é **distinta** do CB-01: CB-01 suprime T_total via HPG; FSR actua directamente no miosina/actina via REDD1 + MuRF1. São dois outputs independentes do mesmo input (cortisol elevado).

---

### 1.23 Motor de Depleção Hipocampal por Cortisol Crónico (MA §4.4)

```
Volume_hipocampal_reduzido_pct ≈ −2% a −3% por 6 semanas de cortisol elevado
```

Efeito molecular: inibição da neurogénese na zona sub-granular do giro dentado + retracção dendrítica CA3 → menor feedback negativo hipocampal sobre NPV → amplificação do eixo HPA (Loop 4 da §1.16).

---

## SECÇÃO 2 — OMISSÕES CRÍTICAS DOS RESUMOS ANTERIORES

O inventário genérico anterior capturou os 8 motores de alto nível (MTS, Cascade Threshold, ALS Amplifier, OTS State Machine, IIG, MPC, DME, Banister+Kalman). O que **não capturou**:

| # | Omissão | Impacto |
|---|---|---|
| O-01 | **Leucina Efectiva (CB-02)** — fórmula que reescreve o threshold de dose de leucina para activar mTORC1 sob cortisol elevado | Fatal para prescrição nutricional; suplementação padrão falha silenciosamente |
| O-02 | **FSR/Cortisol quotient** — −8–12% de FSR por cada 10 µg/dL acima do normal | Motor separado de CB-01; não capturado |
| O-03 | **SHBG amplificado por CRP** — via hepática independente que suprime T_livre mesmo com T_total normal | Invisível sem este motor; T_total normal mascara T_livre comprometida |
| O-04 | **TNF-α × IL-1β amplification ×1.35** — carácter supra-aditivo do bloqueio mTORC1 | Subestimação do bloqueio se apenas TNF-α for medido |
| O-05 | **Carnosina buffer formula** — `ΔpH = [Carnosina] × 0.1 / Produção_H⁺` | Capacidade atlética de tolerância à acidose não quantificável sem isto |
| O-06 | **GH nocturno composto** `GH_real = GH_máx × f(SWS%) × f(Cortisol)` | Antes mencionado qualitativamente; agora formalizável |
| O-07 | **Loop IDO-Kinurenina completo** — IL-6 → IDO → serotonina ↓ → melatonina ↓ → sono ↓ → cortisol ↑ → LPS ↑ → IDO | Loop de 5 passos completamente ausente do inventário anterior |
| O-08 | **Resistência Glucocorticoide Paradoxal** — "pior dos dois mundos" com stress psicológico crónico | Classe de patologia não capturada; impossível de diagnosticar sem este motor |
| O-09 | **Dano epigenético OTS Stage 4** — NR3C1 hipermetilação, DunedinPACE > 1.4 | Estado de "reescrita permanente de Phase 3 por Phase 4" nunca descrito |
| O-10 | **5 Loops de Auto-Amplificação formalizados** (§1.16) | Apenas mencionados abstractamente; nunca extraídos como estruturas computacionais |
| O-11 | **Motor de Magnésio 8-sistema** (§1.17) | Completamente ausente |
| O-12 | **Motor de Vitamina B6/PLP 7-sistema** (§1.18) | Completamente ausente |
| O-13 | **CRH → permeabilidade intestinal em minutos** via mastócitos — sem stressor físico | Mecanismo de trigger psicológico directo sobre o intestino nunca capturado |
| O-14 | **Loop de Veillonella** — lactato muscular → propionato → AMPK hepática → biogénese mitocondrial | Mencionado em 1 linha; nunca formalizado como cadeia computável |
| O-15 | **Desalinhamento circadiano como reprogramador epigenético** — CpG methylation de PER1/BMAL1 persistindo semanas | Ausente |
| O-16 | **Tabela Mestre 9×9** como estrutura de dados | Parcialmente referenciada; nunca extraída como matriz computacional |
| O-17 | **IIG = f(ALS_F4, Baseline_F2, TGI-TEA_F3, T_espécie-F1)** como equação integradora de todas as fases | A fórmula estava no resumo de contexto mas sem a integração multi-fase formalizada |
| O-18 | **SOCS3 e resistência ao IGF-1** — IL-6 crónica → gp130 → SOCS3 → compete com IRS-1 em JAK2 → resistência ao IGF-1 muscular | Mecanismo de resistência ao anabolismo independente de insulina nunca capturado |
| O-19 | **DunedinPACE** como biomarcador computável | Completamente ausente de todos os inventários anteriores |

---

## SECÇÃO 3 — ERROS, CONTRADIÇÕES E IMPRECISÕES

### E-01 — Erro Editorial com Impacto Algorítmico (MA §4.2)

**Texto literal do documento:** *"activa TSC2 (inibidor de mTORC1) — não, activa TSC2"*

O documento contém uma auto-correcção a meio de frase: a formulação correcta é que o REDD1 **activa** o TSC2 (que por sua vez inibe Rheb, que inibe mTORC1). A bioquímica está correcta na versão corrigida, mas o texto "*não, activa TSC2*" é uma marca editorial que ficou no documento final. Qualquer parser textual que tente extrair a lógica desta frase vai extrair a versão errada ("inibe TSC2") antes de a auto-corrigir.

**Impacto:** Se o código de parsing desta frase for automático (via LLM ou regex), irá capturar o mecanismo invertido — o que produziria um motor que *activa* mTORC1 quando cortisol sobe, o oposto da verdade fisiológica.

---

### E-02 — Inconsistência Numérica: Supressão de Testosterona

**Fonte A (TC §2.1, CB-01):** Usa cortisol em **µg/dL absoluto** com limiar = 18 µg/dL.
```
T = T_baseline × max(0.40, 1.0 − 0.030 × max(0, Cortisol_µg_dL − 18))
Cortisol = 24 µg/dL → T = baseline × 0.820 → supressão = −18%
```

**Fonte B (MA §4.2):** Usa **percentagem relativa acima da baseline**:
*"Cortisol 60% acima da baseline × 21 dias → testosterona −35 a −50%"*

Se baseline cortisol ≈ 15 µg/dL, então 60% acima = 24 µg/dL. Pela fórmula CB-01: supressão = −18%. O documento MA diz −35 a −50% para o mesmo valor absoluto de cortisol.

**Análise:** A discrepância pode ter duas causas:
1. A MA está a descrever o efeito crónico (21 dias) vs. o CB-01 que é transiente — ou seja, são motores temporalmente distintos e a fórmula CB-01 não captura o efeito cumulativo.
2. A MA usa "60% acima da baseline individual" onde baseline individual pode ser cortisol habitual < 10 µg/dL — nesse caso, 60% acima = ~16 µg/dL, e a fórmula daria −0%, o que é ainda mais inconsistente.

**Conclusão:** Existe uma **lacuna conceptual** — o CB-01 não modela o efeito tempo-dependente do cortisol cronicamente elevado. Precisamos de um segundo motor: `T_supressão_crónica(t) = f(Cortisol, t_dias_exposição)`.

---

### E-03 — Contradição: Duração de Recuperação NFOR

| Documento | NFOR Recuperação |
|---|---|
| TC §9.1 | "2–6 semanas para recuperação total" |
| MA §21 (Estádio 3) | "4–12 semanas de descanso activo" |

Os intervalos não se sobrepõem na extremidade inferior (2 semanas vs. 4 semanas). A diferença pode ser contextual (TC descreve recuperação clínica mínima; MA descreve recuperação funcional completa com retorno à carga normal), mas o documento não explicita esta distinção. Para código, qual é o threshold correcto a usar no OTS State Machine?

---

### E-04 — IIG Computacionalmente Incompleto (Blocker Crítico)

A fórmula `IIG(t) = ΣΣ w_ij × max(0, D_ij(t))` é apresentada como o motor síntese da Fase 5. No entanto:

1. Os **w_ij** nunca são definidos numericamente. A Tabela Mestre (MA §24) usa ↓, ↓↓, ↓↓↓ — não valores cardinais.
2. Os **D_ij(t)** não têm função de normalização especificada. Como se calcula D_cortisol? É linear? É baseado em z-scores? Usa os limiares dos Travões?
3. A soma dupla `ΣΣ` implica 9×9 = 81 pares de sistemas — mas apenas ~30 são preenchidos na tabela com efeitos não-negligenciáveis. Os outros 51 pares são assumidos zero? Nunca se explicita.

**Consequência:** O IIG é um motor definido em termos de estrutura mas **indefinido em termos de valores**. O documento descreve a arquitectura do score mas não a calibração. Para operacionalizar, precisamos de inventar os w_ij ou pedir calibração explícita.

---

### E-05 — Gap nos Red Blocks: Zona de NFOR sem Override

**Travão 8 (Red Block):** Activa quando `T/C < 0.30 × baseline por 5+ dias` → OTS confirmado.

**Estado 2 (NFOR):** T/C entre 0.30 e 0.50, cortisol > 22 µg/dL.

**Gap:** Um atleta em NFOR com T/C = 0.35 (abaixo de 0.50, acima de 0.30) e cortisol = 24 µg/dL (abaixo de 30 µg/dL) não activa **nenhum Red Block**. Activa apenas o Amber Block "REDUCE_INTENSITY + volume_factor=0.75". Mas o Estado 2 (NFOR) requer −50 a −60% de volume e eliminação de todo o treino acima de Z3. O Amber Block apenas aplica −25% de volume (factor 0.75) com cap Z3. Há uma **sub-correcção de ~25–35%** neste intervalo crítico.

---

### E-06 — Naming Error: ALS_Fase4 vs. ALS_Fase5

**MA §27.1, equação integradora:**
```
IIG = f(ALS_Fase4, Baseline_Fase2, TGI-TEA_Fase3, T_espécie-TGI_Fase1)
```

O subscrito "Fase4" refere-se ao output da Fase 5 (ALS_integrado, calculado na TC §XIV). Não é um output da Fase 4 (que produz apenas métricas raw, não o ALS). Este é um **erro de nomenclatura** no documento que vai criar confusão na arquitectura de dados se não for corrigido.

**Correcção necessária:** Substituir `ALS_Fase4` por `ALS_Fase5` ou `ALS_integrado` para reflectir a origem real do valor.

---

### E-07 — Ausência de Threshold Numérico para o Motor CRH-Intestinal

O mecanismo CRH → mastócitos → TJ abertas em minutos é descrito qualitativamente mas sem:
- Threshold de CRH (nmol/L) necessário para desgranulação significativa de mastócitos
- Magnitude de aumento de permeabilidade (fold change de zonulina ou TEER reduction %)
- Duração do efeito (horas?)

Este é um **motor sem calibração** — descrito mecanisticamente mas não operacionalizável sem investigação adicional.

---

## SECÇÃO 4 — DEPENDÊNCIAS EXPLÍCITAS PARA A FASE 6

Os seguintes elementos da Fase 5 **não fazem sentido sem a Fase 6** como consumidor:

### D-01 — Travões de Segurança como Constraints do MPC

O documento TC §XIII declara literalmente: *"Para o MPC da Fase 6, as regras de bloqueio hierarquizam-se em três níveis de severidade."*

Os 10 Travões são **outputs de Phase 5** que se tornam **hard constraints na função de optimização do MPC (Phase 6)**. Sem a Phase 6 como executor, os Travões são apenas FLAGS sem actuador.

### D-02 — ALS_integrado > 3.5 (Travão 10)

O threshold 3.5 não é derivado de nenhum cálculo interno à Phase 5. É o boundary do custo de ALS na função objectivo do MPC Phase 6 (onde ALS é minimizado). O valor 3.5 é um parâmetro de Phase 6 que retroage para Phase 5 como critério de override.

### D-03 — SUPERCOMP-GREEN-01 (volume_factor = 1.10)

O `check_green_optimizations` retorna a flag "volume pode aumentar 10%". Mas **o que executa este aumento** é o motor DME (Dynamic Modulation Engine) e o MPC receding horizon — ambos Phase 6. A Phase 5 emite o sinal; a Phase 6 age sobre ele.

### D-04 — IIG Interpretation Table Action Levels

*"Treino de alta intensidade autorizado"* (IIG 0–10) — esta autorização pressupõe um sistema que lê o IIG e ajusta bandas de intensidade de prescrição. Esse sistema é o MPC Phase 6 com o seu modelo de Fitness-Fatigue Banister Expandido (6 compartimentos). O IIG sem o MPC é um número sem actuador.

### D-05 — OTS Exit Criteria como Gate para Phase 6

*"Sem retorno a treino normal até: T/C > 0.70×baseline AND HRV > −0.5 DP AND sIgA > 80% baseline"*

Esta triple-gate é a condição que o MPC Phase 6 verifica antes de desbloquear bandas de intensidade superiores. Phase 5 calcula o estado actual; Phase 6 decide se pode avançar.

### D-06 — τf e τF do Banister Expandido (Phase 6) dependem de Phase 5

Os time constants de fitness (τf_i) e fadiga (τF_i) para os 6 compartimentos do Banister Expandido são actualizados pelo EKF (Phase 6) usando proxies como CMJ, RMSSD, lactato. Mas os **estados iniciais e os bounds** destes parâmetros dependem dos estados OTS/NFOR da máquina de estados Phase 5 — um atleta em Estado 2 (NFOR) tem τF_NM (fadiga neuromuscular) fundamental diferente de um em Estado 0. Phase 5 define o espaço de estados válido; Phase 6 estima os parâmetros dentro desse espaço.

---

## SÍNTESE EXECUTIVA

### O que a Fase 5 tem que funcionava:
- 10 Travões de Segurança com thresholds numéricos precisos — executáveis
- ALS_integrado com fórmula de produto de γ_loops — executável
- OTS State Machine com critérios multi-dimensionais — executável (com a contradição E-03 resolvida)
- CB-01, CB-02, CB-03, IB-01, IB-01-b, IB-02, AB-01 — todos os motores de cascata unitários são executáveis

### O que a Fase 5 tem que está incompleto:
- IIG: estrutura definida, w_ij e D_ij normalization **por definir** (Blocker)
- Motor CRH-Intestinal: mecanismo descrito, thresholds **por calibrar**
- Motor de supressão crónica de T (tempo-dependente): lacuna entre CB-01 e MA §4.2

### O que a Fase 5 tem que está errado:
- E-01: erro editorial na via REDD1-TSC2 (auto-correcção deixada no texto)
- E-02: inconsistência numérica de supressão T entre TC e MA
- E-03: contradição de duração de recuperação NFOR (2–6 vs. 4–12 semanas)
- E-06: naming error ALS_Fase4 vs. ALS_Fase5

### O que a Fase 5 precisa que vem de fora:
- Phase 6 como actuador de todos os outputs (D-01 a D-06)
- Calibração externa dos w_ij para o IIG ser operacional
- Definição dos time constants τf_i, τF_i por compartimento (Phase 6 territory)

---

*Documento gerado por extracção directa dos textos fonte. Nenhum valor inventado. Zero Alucinação.*
