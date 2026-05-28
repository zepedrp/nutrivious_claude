"""
app/engine/constants.py — Constantes Imutáveis do Motor Preditivo

REGRA ABSOLUTA: cada valor numérico neste ficheiro tem uma referência
explícita ao documento e secção de origem. Valores NÃO declarados nos
documentos estão definidos como None — NUNCA como estimativas.

Fontes primárias:
  Doc1 = nutrivious_bos_fase3_intersecao_matematica.md
  Doc2 = Nutrivious_BOS_Phase3_Interseccao_Teto_Pessoal.md

DISCREPÂNCIAS DOCUMENTADAS (valores em conflito entre Doc1 e Doc2):
  D1: T_espécie Fatmax — Doc1 §3.3: 1.7 g/min; Doc2 §8.2: 2.0 g/min
      → Adoptado: 1.7 g/min (Doc1 §3.3 — secção de fórmulas formais)
  D2: T_espécie MPS — Doc1 §3.4: 0.12 %/h; Doc2 §22.2: 0.35 %/h
      → Adoptado: 0.12 %/h (Doc1 §3.4 — secção de fórmulas formais)
  D3: PPARGC1A Gly/Ser (VO2max) — Doc1 §2.7: 0.95; Doc2 §5.2: 0.94
      → Adoptado: 0.94 (Doc2 §5.2 — tabela genética detalhada)
  D4: ACE I/D — Doc1 §2.7: II=1.05, ID=1.02, DD=0.98
                Doc2 §5.2: II=1.04, ID=1.00, DD=0.96
      → Adoptado: Doc2 §5.2
  D5: ACTN3 Power — Doc1 §2.7: RR=1.00, RX=0.95, XX=0.85 (usa RR como ref)
                    Doc2 §6.2: RR=1.05, RX=1.00, XX=0.86 (usa RX como ref)
      → Adoptado: Doc2 §6.2 (RX como referência da população)
  D6: MTHFR TT (MPS) — Doc1 §2.7: TT=0.92; Doc2 §7.4: CT=0.93, TT=0.82
      → Adoptado: Doc2 §7.4 (inclui heterozigótico e valor mais calibrado)
"""
from __future__ import annotations

from types import MappingProxyType


# ─────────────────────────────────────────────────────────────────────────────
# 1. TECTOS ABSOLUTOS DA ESPÉCIE — T_espécie
# ─────────────────────────────────────────────────────────────────────────────
# Valores máximos para Homo sapiens. Numerador de CPI(S) = Baseline / T_esp.
# Nenhum TEA pode exceder T_espécie por definição biológica (§1.2, Doc2).

T_SPECIES: MappingProxyType = MappingProxyType({
    # VO2max [mL/kg/min] — Doc2 §5.1 e Doc1 §3.1: "97.5 mL/kg/min"
    # Referência empírica: Oskar Svendsen (2012, ciclismo). TGI pode exceder
    # este valor em indivíduos com múltiplos alelos de vantagem (Doc2 §21.1).
    "vo2max_ml_per_kg_min": 97.5,

    # CHO oxidação intestinal [g/h] — Doc1 §3.2: "PC_CHO = 120 × G_CHO × M_CHO"
    # Protocolo multi-transportador SGLT1+GLUT5 (Jeukendrup 2010).
    "cho_absorption_g_per_h": 120.0,

    # Fatmax [g/min] — Doc1 §3.3: "PC_Fatmax(t) = 1.7 × G_Fat × M_Fat(t)"
    # DISCREPÂNCIA D1: Doc2 §8.2 usa 2.0 g/min no exemplo numérico.
    # Fonte adoptada: Doc1 §3.3 (secção formal de fórmulas).
    "fat_oxidation_max_g_per_min": 1.7,

    # MPS fraccional [%/h] — Doc1 §3.4: "PC_MPS(t) = 0.12 × G_MPS × ..."
    # DISCREPÂNCIA D2: Doc2 §22.2 Tabela Mestre usa 0.35 %/h.
    # Fonte adoptada: Doc1 §3.4 (secção formal de fórmulas).
    "mps_fractional_rate_pct_per_h": 0.12,

    # Potência pico 3s [W absolutos] — Doc1 §3.5: "PC_Power_3s(t) = 3500 × ..."
    # Watts absolutos (não W/kg). O modificador de BF% ajusta para relativo.
    "peak_anaerobic_power_watts": 3500.0,

    # Tonnage estrutural semanal — Doc1 §3.6: "Base_Tonnage_Ref"
    # Valor numérico não declarado nos documentos. Normalizado a 1.0
    # (Índice Base de Normalização Humana): o tecto pessoal é expresso
    # como fracção do tecto da espécie em vez de tonnage absoluta.
    "structural_weekly_tonnage_ref": 1.0,

    # Capacidade cognitiva integrada — Doc2 §14.1: "C_espécie"
    # Valor numérico não declarado nos documentos. Normalizado a 1.0
    # (Índice Base de Normalização Humana): o tecto é expresso como
    # índice adimensional de capacidade relativa à espécie.
    "cognitive_capacity_index": 1.0,

    # Taxa de sudação máxima [L/h] — Doc2 §10.1: "ṁ_suor,espécie"
    # Valor numérico não declarado nos documentos. Normalizado a 1.0
    # (Índice Base de Normalização Humana): modificadores expressos
    # como fracção da capacidade termorreguladora de espécie.
    "sweat_rate_l_per_h_max": 1.0,
})


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARÂMETROS DOS MODIFICADORES BIOQUÍMICOS — Camada M (trimestral)
# ─────────────────────────────────────────────────────────────────────────────
# Cada entrada define os parâmetros da função de base.py correspondente.
# "function": nome da primitiva em base.py a usar.
# Todos os valores são rastreados a Doc1 §2.x (biblioteca de modificadores).

MODIFIER_PARAMS: MappingProxyType = MappingProxyType({

    # ── Sistema Aeróbio (VO2max) ──────────────────────────────────────────────

    # Doc1 §2.1: M_Hb = clamp(Hb / Hb_ref, 0.75, 1.15)
    # Relação linear via equação de Fick: CaO₂ = 1.34 × [Hb] × SaO₂.
    # Válida até Hct ≈ 55% (limiar de hiperviscosidade).
    "M_Hb_male": {
        "function": "linear_ratio",
        "reference": 15.5,   # g/dL — Doc1 §2.1
        "floor": 0.75,
        "ceiling": 1.15,
    },
    "M_Hb_female": {
        "function": "linear_ratio",
        "reference": 13.5,   # g/dL — Doc1 §2.1
        "floor": 0.75,
        "ceiling": 1.15,
    },

    # Doc1 §2.1: M_BF = (1 − BF%) / (1 − BF%_ref)
    # Custo de transportar massa metabolicamente inerte sobre o VO2max/kg.
    "M_BF_aerobic_male": {
        "function": "lean_fraction",
        "ref_bf_pct": 0.12,  # Doc1 §2.1
        "floor": 0.75,
        "ceiling": 1.10,
    },
    "M_BF_aerobic_female": {
        "function": "lean_fraction",
        "ref_bf_pct": 0.20,  # Doc1 §2.1
        "floor": 0.75,
        "ceiling": 1.10,
    },

    # Doc1 §2.1: M_O3 piecewise — três segmentos documentados.
    # Mecanismo: DHA → cardiolipina da membrana mitocondrial interna →
    # eficiência de empacotamento dos Complexos I/III da CTE e F₀F₁-ATPase.
    # ~3-4% de eficiência oxidativa por unidade de défice abaixo de 8%.
    "M_O3_aerobic": {
        "function": "piecewise_linear",
        "x_points": [0.0, 4.0, 8.0, 12.0],   # % índice ómega-3 — Doc1 §2.1
        "y_points": [0.88, 0.88, 1.00, 1.02],
        "floor": 0.88,
        "ceiling": 1.02,
    },

    # Doc1 §2.1: M_VitD_aerobic piecewise.
    # Três vias paralelas: (1) EPO sensitivity → Hb; (2) VDR cardiomiócitos
    # → função sistólica; (3) VDR em promoter PGC-1α → biogénese mito.
    # SI: 20 ng/mL × 2.5 = 50 nmol/L; 50 ng/mL × 2.5 = 125 nmol/L.
    "M_VitD_aerobic": {
        "function": "piecewise_linear",
        "x_points": [0.0, 50.0, 125.0],   # nmol/L — Doc1 §2.1 (SI recalibrado)
        "y_points": [0.88, 0.88, 1.00],
        "floor": 0.88,
        "ceiling": 1.00,
    },

    # Doc1 §2.1: M_CRP_aerobic = clamp(exp(−0.10 × max(0, CRP − 0.3)), 0.80, 1.00)
    # k=0.10 calibrado para CRP=3.0 mg/L → ~20% de penalização.
    # Mecanismo: anemia de doença crónica via hepcidina + disfunção eNOS.
    "M_CRP_aerobic": {
        "function": "exponential_decay",
        "rate": 0.10,         # Doc1 §2.1
        "reference": 0.3,    # mg/L — Doc1 §2.1 (inflamação de fundo mínima)
        "floor": 0.80,
        "ceiling": 1.00,
    },

    # Doc1 §2.1: M_Fe piecewise via ratio sTfR/log(Ferritina).
    # Imune ao efeito mascarador da inflamação sobre a ferritina isolada.
    "M_Fe_aerobic": {
        "function": "piecewise_linear",
        "x_points": [0.0, 1.0, 2.0, 4.0],   # ratio sTfR/log(Ferritina) — Doc1 §2.1
        "y_points": [1.00, 1.00, 0.95, 0.75],
        "floor": 0.75,
        "ceiling": 1.00,
    },

    # Doc2 §5.2: M_ferritin = min(1.0, sqrt(v_ferr / 80))
    # Equivalente a power_law com expoente=0.5, ref=80, ceiling=1.0.
    "M_ferritin_vo2_doc2": {
        "function": "power_law",
        "reference": 80.0,   # ng/mL — Doc2 §5.2
        "exponent": 0.5,
        "floor": 0.0,
        "ceiling": 1.0,
    },

    # Doc2 §5.2: M_Hb = (Hb/16.0)^0.85 (homem)
    # Versão Doc2 — expoente sub-linear captura não-linearidade real.
    "M_Hb_male_doc2": {
        "function": "power_law",
        "reference": 16.0,   # g/dL — Doc2 §5.2
        "exponent": 0.85,
        "floor": 0.0,
        "ceiling": 1.5,      # sem ceiling explícito no Doc2 para este caso
    },

    # Doc2 §5.2: M_VitD_vo2 = min(1.0, 0.6 + 0.4 × VitD/150)
    # Piecewise linear: de 0 a 150 nmol/L, de 0.60 a 1.00.
    "M_VitD_vo2_doc2": {
        "function": "piecewise_linear",
        "x_points": [0.0, 150.0],   # nmol/L — Doc2 §5.2 (SI recalibrado)
        "y_points": [0.60, 1.00],
        "floor": 0.60,
        "ceiling": 1.00,
    },

    # Doc2 §5.2: M_T3_vo2 = min(1.0, (T3/5.5)^0.6)
    "M_T3_vo2_doc2": {
        "function": "power_law",
        "reference": 5.5,    # pmol/L — Doc2 §5.2
        "exponent": 0.60,
        "floor": 0.0,
        "ceiling": 1.00,
    },

    # Doc2 §5.2: M_testosterone_vo2 = min(1.0, 0.7 + 0.3 × T/22)
    # Piecewise linear: de 0 a 22 nmol/L, de 0.70 a 1.00.
    "M_testosterone_vo2_male_doc2": {
        "function": "piecewise_linear",
        "x_points": [0.0, 22.0],   # nmol/L — Doc2 §5.2
        "y_points": [0.70, 1.00],
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # Doc2 §5.2: M_omega3_vo2 = min(1.0, 0.85 + 0.15 × omega3/8)
    "M_omega3_vo2_doc2": {
        "function": "piecewise_linear",
        "x_points": [0.0, 8.0],   # % — Doc2 §5.2
        "y_points": [0.85, 1.00],
        "floor": 0.85,
        "ceiling": 1.00,
    },

    # Doc2 §5.2: M_HOMA_vo2 = max(0.7, 1.0 − 0.08 × (HOMA − 1))
    "M_HOMA_vo2_doc2": {
        "function": "linear_with_threshold",
        "reference": 1.0,    # Doc2 §5.2
        "slope": -0.08,
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # Doc2 §5.2: M_Cortisol_DHEA_vo2 = max(0.75, 1.0 − 0.025 × (ratio − 8))
    "M_cortisol_dhea_ratio_vo2_doc2": {
        "function": "linear_with_threshold",
        "reference": 8.0,    # Doc2 §5.2
        "slope": -0.025,
        "floor": 0.75,
        "ceiling": 1.00,
    },

    # Doc2 §5.2: M_DunedinPACE = clamp(2.0 − pace, 0.60, 1.05)
    # = linear_deviation com slope=−1.0, reference=1.0.
    "M_DunedinPACE": {
        "function": "linear_deviation",
        "reference": 1.0,    # Doc2 §5.2 (pace=1.0: envelhecimento cronológico)
        "slope": -1.0,
        "floor": 0.60,
        "ceiling": 1.05,
    },

    # ── Sistema Glicolítico (CHO) ─────────────────────────────────────────────

    # Doc1 §2.2: M_HOMA_CHO = clamp(exp(−0.15 × max(0, HOMA − 1.0)), 0.40, 1.00)
    # k=0.15 calibrado para HOMA=3.0 → 26% de penalização.
    # Mecanismos: (1) menor GLUT4 translocation; (2) menor SGLT1 via PI3K/Akt.
    "M_HOMA_CHO": {
        "function": "exponential_decay",
        "rate": 0.15,        # Doc1 §2.2
        "reference": 1.0,   # HOMA-IR — Doc1 §2.2
        "floor": 0.40,
        "ceiling": 1.00,
    },

    # Doc1 §2.2: M_Zonulin_CHO = clamp(1.0 − 0.012 × max(0, Z − 20), 0.70, 1.00)
    # Mecanismo: zonulina → TLR4 → NF-κB → downregulação de SGLT1.
    "M_Zonulin_CHO": {
        "function": "linear_with_threshold",
        "reference": 20.0,   # ng/mL — Doc1 §2.2
        "slope": -0.012,     # Doc1 §2.2: cada ng/mL acima de 20 → −1.2%
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # Doc1 §2.2: M_GutTrain = 0.70 + 0.30 × min(1.0, semanas / 8)
    # = piecewise_linear de 0 a 8 semanas, de 0.70 a 1.00.
    # SGLT1 upregulation por exposição crónica a CHO intraluminal.
    "M_GutTrain_CHO": {
        "function": "piecewise_linear",
        "x_points": [0.0, 8.0],   # semanas — Doc1 §2.2
        "y_points": [0.70, 1.00],
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # Doc1 §2.2: M_Microbiome_CHO piecewise via Shannon index.
    # Shannon < 3.5: M = 0.85 + 0.15 × (H/3.5); Shannon ≥ 3.5: M = 1.00.
    "M_Microbiome_CHO": {
        "function": "piecewise_linear",
        "x_points": [0.0, 3.5],   # índice de Shannon — Doc1 §2.2
        "y_points": [0.85, 1.00],
        "floor": 0.85,
        "ceiling": 1.00,
    },

    # Doc1 §3.2: M_Calprotectin = max(0.80, 1.0 − 0.002 × max(0, Calp − 50))
    "M_Calprotectin_CHO": {
        "function": "linear_with_threshold",
        "reference": 50.0,   # µg/g — Doc1 §3.2
        "slope": -0.002,
        "floor": 0.80,
        "ceiling": 1.00,
    },

    # ── Sistema Lipolítico (Fatmax) ───────────────────────────────────────────

    # Doc1 §2.3: M_HOMA_Fat = clamp(1.0 − 0.08 × max(0, HOMA − 1.0), 0.50, 1.00)
    # k=0.08 (menos agressivo que CHO) porque inibição de CPT-1 não é total.
    # Mecanismo: insulina → malonil-CoA → inibe CPT-1 (Ki ≈ 0.02 µM).
    "M_HOMA_Fat": {
        "function": "linear_with_threshold",
        "reference": 1.0,    # HOMA-IR — Doc1 §2.3
        "slope": -0.08,
        "floor": 0.50,
        "ceiling": 1.00,
    },

    # Doc1 §2.3: M_T3_Fat piecewise (versão corrigida no documento).
    # ratio FT3/rT3: T3 activa PGC-1α, PPAR-α, CPT-1, UCP1.
    # rT3 compete pelos mesmos receptores sem activar transcrição.
    "M_T3_Fat": {
        "function": "piecewise_linear",
        "x_points": [0.0, 10.0, 20.0],   # ratio FT3/rT3 — Doc1 §2.3
        "y_points": [0.75, 1.00, 1.04],
        "floor": 0.75,
        "ceiling": 1.04,
    },

    # Doc1 §2.3: M_Carnitine_Fat piecewise.
    # Carnitina é substrato obrigatório da CPT-1 para transporte de LCFA.
    "M_Carnitine_Fat": {
        "function": "piecewise_linear",
        "x_points": [0.0, 15.0, 25.0, 40.0],   # µmol/L — Doc1 §2.3
        "y_points": [0.75, 0.85, 0.95, 1.00],
        "floor": 0.75,
        "ceiling": 1.00,
    },

    # Doc1 §3.3: M_Mito_Fat = clamp(CS_activity / 25, 0.40, 1.80)
    # CS (citrato sintase) = melhor proxy de densidade mitocondrial sem biópsia.
    "M_Mito_Fat": {
        "function": "linear_ratio",
        "reference": 25.0,   # µmol/min/g — Doc1 §3.3
        "floor": 0.40,
        "ceiling": 1.80,
    },

    # ── Sistema Anabólico (MPS e Hipertrofia) ─────────────────────────────────

    # Doc1 §2.4: M_T_MPS = clamp((T / 20.8)^0.35, 0.75, 1.20) [homem]
    # Bhasin et al. NEJM 2001 — dose-response em 61 homens.
    # Expoente 0.35: cinética saturável do receptor androgénico (Kd ≈ 0.1-1 nM).
    "M_T_MPS_male": {
        "function": "power_law",
        "reference": 20.8,   # nmol/L — Doc1 §2.4 (SI recalibrado; 600 ng/dL ÷ 28.84)
        "exponent": 0.35,
        "floor": 0.75,
        "ceiling": 1.20,
    },
    "M_T_MPS_female": {
        "function": "power_law",
        "reference": 1.39,   # nmol/L — Doc1 §2.4 (SI recalibrado; 40 ng/dL ÷ 28.84)
        "exponent": 0.35,
        "floor": 0.75,
        "ceiling": 1.20,
    },

    # Doc1 §2.4: M_Cortisol_MPS = clamp(1.0 − 0.025 × max(0, C − 14), 0.60, 1.00)
    # Mecanismo: REDD1 → TSC1/2 → inibe mTORC1; MuRF-1 + MAFbx → proteólise.
    # k=0.025 calibrado para cortisol=25 µg/dL → 27.5% de penalização.
    "M_Cortisol_MPS": {
        "function": "linear_with_threshold",
        "reference": 14.0,   # µg/dL pico matinal — Doc1 §2.4
        "slope": -0.025,
        "floor": 0.60,
        "ceiling": 1.00,
    },

    # Doc1 §2.4: M_VitD_MPS piecewise.
    # VDR nos miócitos: (1) upregula IGF-1R; (2) activa genes miofibrilares;
    # (3) VDR-RXR recruta co-activadores do mTORC1.
    "M_VitD_MPS": {
        "function": "piecewise_linear",
        "x_points": [0.0, 50.0, 125.0],   # nmol/L — Doc1 §2.4 (SI recalibrado)
        "y_points": [0.72, 0.72, 1.00],
        "floor": 0.72,
        "ceiling": 1.00,
    },

    # Doc1 §2.4: M_IGF1_MPS = clamp((IGF1 / 180)^0.40, 0.80, 1.15)
    # Activa PI3K → Akt → mTORC1 independentemente da resistência insulínica.
    "M_IGF1_MPS": {
        "function": "power_law",
        "reference": 180.0,  # ng/mL — Doc1 §2.4
        "exponent": 0.40,
        "floor": 0.80,
        "ceiling": 1.15,
    },

    # Doc1 §2.4: M_Inflam_MPS = clamp(exp(−0.20 × max(0, CRP − 0.3)), 0.65, 1.00)
    # k=0.20 (mais agressivo que aeróbio): fosforilação de IRS-1 em Ser307
    # via IKK-β e JNK — independente e aditivo ao efeito do cortisol.
    "M_Inflam_MPS": {
        "function": "exponential_decay",
        "rate": 0.20,        # Doc1 §2.4
        "reference": 0.3,   # mg/L — Doc1 §2.4
        "floor": 0.65,
        "ceiling": 1.00,
    },

    # Doc1 §2.4: M_Energy_MPS piecewise via EA (kcal/kg_FFM/dia).
    # Limiar RED-S: EA < 45 → organismo sacrifica MPS para conservar energia.
    # Fórmula exacta: if EA < 25: M = clamp(0.75 − 0.020 × (25 − EA), 0.50, 0.75)
    # O slope é −0.020/unidade de EA=25 para baixo; o floor 0.50 activa a EA=12.5.
    # Breakpoint extra em 12.5 garante que o slope 0.020 é preservado (não 0.010).
    "M_Energy_MPS": {
        "function": "piecewise_linear",
        "x_points": [0.0, 12.5, 25.0, 35.0, 45.0],   # kcal/kg_FFM/dia — Doc1 §2.4
        "y_points": [0.50, 0.50, 0.75, 0.90, 1.00],
        "floor": 0.50,
        "ceiling": 1.00,
    },

    # Doc1 §2.4: M_Energy_proxy via RMR_ratio = RMR_medida / RMR_predita.
    # Proxy quando pesagem de alimentos não está disponível.
    "M_Energy_MPS_RMR_proxy": {
        "function": "linear_ratio",
        "reference": 1.0,    # Doc1 §2.4: RMR_ratio = 1.0 → sem supressão
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # ── Sistema Neuromuscular (Potência e RFD) ────────────────────────────────

    # Doc1 §2.5: M_PCr = clamp([PCr] / 80, 0.80, 1.25)
    # [PCr]_ref = 80 mmol/kg de massa muscular seca.
    # Com creatina loading (≥4 semanas): [PCr] sobe ~20-30% → M ≈ 1.22.
    "M_PCr_Power": {
        "function": "linear_ratio",
        "reference": 80.0,   # mmol/kg massa seca — Doc1 §2.5
        "floor": 0.80,
        "ceiling": 1.25,
    },

    # Doc1 §2.5: M_T_Power = clamp((T / 20.8)^0.25, 0.85, 1.12) [homem]
    # Expoente 0.25 (vs 0.35 da MPS): efeito neural (excitabilidade de
    # motoneurónio + expressão MHC-IIx) mais saturável que efeito transcripcional.
    "M_T_Power_male": {
        "function": "power_law",
        "reference": 20.8,   # nmol/L — Doc1 §2.5 (SI recalibrado; 600 ng/dL ÷ 28.84)
        "exponent": 0.25,
        "floor": 0.85,
        "ceiling": 1.12,
    },

    # Doc1 §2.5: M_B12_Neural piecewise via MMA (metilmalónico).
    # MMA > 3 µmol/mmol_creatinina = défice funcional de B12.
    # MMA é mais sensível que B12 sérico para défice intracelular.
    # Fórmula exacta: if MMA ≥ 10: M = clamp(0.895 − 0.020 × (MMA − 10), 0.80, 0.895)
    # Slope = −0.020/unidade; floor 0.80 activa a MMA = 14.75 (não 20).
    # Breakpoint em 14.75 preserva o slope correcto de −0.020 (não −0.0095).
    "M_B12_Neural": {
        "function": "piecewise_linear",
        "x_points": [0.0, 3.0, 10.0, 14.75],  # µmol/mmol creatinina — Doc1 §2.5
        "y_points": [1.00, 1.00, 0.895, 0.80],
        "floor": 0.80,
        "ceiling": 1.00,
    },

    # Doc2 §9.1: M_B12_myelination — via concentração sérica de B12.
    # Vitamina B12 → síntese de SAM → lecitina (fosfatidilcolina) → mielina.
    # Tabela discreta no Doc2; interpolação linear entre pontos.
    "M_B12_myelination_doc2": {
        "function": "piecewise_linear",
        "x_points": [0.0, 150.0, 250.0, 400.0],   # pg/mL — Doc2 §9.1
        "y_points": [0.70, 0.70, 0.85, 0.95],
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # Doc2 §9.1: M_omega3_myelination = 0.80 + 0.20 × (omega3_index / 8%)
    # DHA: >35% dos ácidos gordos da substância cinzenta → fluidez membranar.
    "M_omega3_myelination": {
        "function": "piecewise_linear",
        "x_points": [0.0, 8.0],   # % — Doc2 §9.1
        "y_points": [0.80, 1.00],
        "floor": 0.80,
        "ceiling": 1.00,
    },

    # Doc2 §9.1: M_glicemia_NCV = 1.0 − 0.08 × (HbA1c − 5.0%)
    # Glicação das proteínas da mielina (MBP) → velocidade de condução reduzida.
    "M_HbA1c_NCV": {
        "function": "linear_with_threshold",
        "reference": 5.0,    # % HbA1c — Doc2 §9.1
        "slope": -0.08,
        "floor": 0.70,       # Doc2 §9.1 implícito
        "ceiling": 1.00,
    },

    # Doc2 §9.2: M_sono^PVT = exp(−0.12 × Δ_sono)
    # rate=0.12 é o único valor declarado nos documentos para esta função.
    "M_sleep_PVT": {
        "function": "sleep_pvt_modifier",
        "rate": 0.12,        # Doc2 §9.2
    },

    # ── Sistema Estrutural (Tendão e Osso) ────────────────────────────────────

    # Doc1 §2.6: M_BMD piecewise via T-score.
    # Resistência à compressão ∝ ρ² (relação de Gibson) — justifica
    # a não-linearidade em T-score.
    "M_BMD_structural": {
        "function": "piecewise_linear",
        "x_points": [-4.0, -2.5, -1.0, 0.0, 2.0],   # T-score — Doc1 §2.6
        "y_points": [0.60, 0.95, 1.00, 1.00, 1.10],
        "floor": 0.60,
        "ceiling": 1.10,
    },

    # Doc1 §2.6: M_VitC_Collagen piecewise.
    # Prolil-4-hidroxilase requer vitamina C como cofactor obrigatório.
    # Sem hidroxiprolina adequada → sem cross-links de piridinolina → UTS↓.
    "M_VitC_Collagen": {
        "function": "piecewise_linear",
        "x_points": [0.0, 25.0, 60.0, 100.0],   # µmol/L — Doc1 §2.6
        "y_points": [0.80, 0.95, 1.00, 1.03],
        "floor": 0.80,
        "ceiling": 1.03,
    },

    # Doc1 §3.6: M_PINP_CTX = clamp((PINP/PINP_ref) / (CTX/CTX_ref), 0.60, 1.40)
    # ratio > 1.0: formação > reabsorção = estado anabólico ósseo.
    "M_PINP_CTX_structural": {
        "function": "ratio_of_ratios",   # implementado no calculador específico
        "pinp_reference": 60.0,          # µg/L — Doc1 §3.6
        "ctx_reference": 0.4,            # ng/mL — Doc1 §3.6
        "floor": 0.60,
        "ceiling": 1.40,
    },

    # Doc2 §12.2 + §5.2: M_VitD mineralizaçao óssea.
    # Fórmula §5.2: M = min(1.0, 0.6 + 0.4 × VitD / 150).
    # VDR nos osteoblastos → expressão de osteocalcina e osteopontina;
    # sem VitD adequada → mineralização da matriz osteóide comprometida.
    # Piecewise equivalente: x=[0,150], y=[0.60,1.00] → slope constante 0.4/150.
    "M_VitD_mineralization": {
        "function": "piecewise_linear",
        "x_points": [0.0, 150.0],   # nmol/L — Doc2 §12.2 / §5.2 (SI recalibrado)
        "y_points": [0.60, 1.00],
        "floor": 0.60,
        "ceiling": 1.00,
    },

    # Doc2 §12.2 + §5.2: M_testosterona mineralizaçao óssea.
    # Fórmula §5.2: M = min(1.0, 0.7 + 0.3 × T / 22) — T em nmol/L.
    # Testosterona activa receptores androgénicos nos osteoblastos →
    # síntese de colagénio tipo I e mineralização; défice → reabsorção dominante.
    "M_testosterone_mineralization": {
        "function": "piecewise_linear",
        "x_points": [0.0, 22.0],    # nmol/L — Doc2 §12.2 / §5.2
        "y_points": [0.70, 1.00],
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # Doc1 §3.6 — M_Inflam_Structural (sem fórmula explícita nos documentos).
    # Inflamação crónica (TNF-α / IL-6 via NF-κB) suprime a síntese de colagénio
    # tipo I e II nos fibroblastos e condrócitos — independente de M_Inflam_MPS.
    # Decaimento exponencial igual ao aeróbio (k=0.10): o tecido estrutural é
    # mais resiliente que a via de MPS (k=0.20) mas mais sensível que não tratado.
    # Floor=0.70: síntese de colagénio nunca colapsa a zero (redundância fibroblástica).
    "M_Inflam_Structural": {
        "function": "exponential_decay",
        "rate": 0.10,        # mesmo k de M_CRP_aerobic — Doc1 §2.1; calibrado
        "reference": 0.3,   # mg/L — estado inflamatório mínimo de fundo
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # Doc1 §3.1: E_PGC1A_methylation = max(0.75, 1.0 − 0.005 × max(0, methylation% − 20))
    # PPARGC1A promoter methylation suppresses mitochondrial biogenesis response to training.
    # Camada E (epigenética, semestral) — distinta dos modificadores M bioquímicos.
    "E_PGC1A_methylation": {
        "function": "linear_with_threshold",
        "reference": 20.0,   # % methylation — Doc1 §3.1
        "slope": -0.005,     # −0.5% de E por cada % de metilação acima de 20%
        "floor": 0.75,       # Doc1 §3.1: "max(0.75, ...)"
        "ceiling": 1.00,
    },

    # Doc2 §12.1: M_hidrataçao = 1.0 − 0.06 × (% perda de peso)
    "M_dehydration_thermoreg": {
        "function": "linear_with_threshold",
        "reference": 0.0,    # % perda de peso — Doc2 §12.1
        "slope": -0.06,
        "floor": 0.60,
        "ceiling": 1.00,
    },

    # Doc2 §14.1: M_Hcy_vascular = max(0.70, 1.0 − 0.025 × (Hcy − 10))
    # Homocisteína > 10 µmol/L inibe eNOS → perfusão cerebrovascular↓.
    "M_Hcy_cognitive": {
        "function": "linear_with_threshold",
        "reference": 10.0,   # µmol/L — Doc2 §14.1
        "slope": -0.025,
        "floor": 0.70,
        "ceiling": 1.00,
    },

    # Doc2 §14.1: M_glicemia_variabilidade = exp(−0.015 × CV%)
    # CV glicémico (CGM) — variabilidade é mais prejudicial que a média.
    "M_glycemic_variability_cognitive": {
        "function": "exponential_decay",
        "rate": 0.015,       # Doc2 §14.1
        "reference": 0.0,   # % CV (já é o excesso, sem limiar)
        "floor": 0.40,
        "ceiling": 1.00,
    },

    # Doc2 §10.2: M_aclimatação^HSP — heat shock protein induction (binary modifier).
    # Acclimatised (10–14 days 35–40°C exposure): M=1.00; not acclimatised: M=0.95.
    # Stored as discrete lookup; the modifier function returns the appropriate value.
    "M_acclimation_thermoreg": {
        "not_acclimatized": 0.95,   # Doc2 §10.2
        "acclimatized": 1.00,       # Doc2 §10.2 (reference)
    },

    # Doc2 §11.1: M_barreira^integridade (Zonulina → absorção CHO)
    # = max(0.75, 1.0 − 0.01 × (Zonulina − 20))
    # Mecanismo inverso ao de SGLT1: permeabilidade não aumenta absorção activa
    # mas aumenta LPS → TLR4 → NF-κB → suprime SGLT1.
    "M_Zonulin_absorption_doc2": {
        "function": "linear_with_threshold",
        "reference": 20.0,   # ng/mL — Doc2 §11.1
        "slope": -0.01,
        "floor": 0.75,
        "ceiling": 1.00,
    },

    # Doc2 §11.1: C_micro^Veillonella — co-factor microbiótico de 3ª ordem.
    # C_micro = 1.0 + 0.03 × ln(Veillonella% / 0.01%)
    # Veillonella atypica converte lactato em propionato → gluconeogénese hepática
    # → substrato energético alternativo absorvido pelo músculo em esforço prolongado.
    # Floor = 1.0: contribuição apenas positiva; abaixo de 0.01% o bónus é nulo.
    "C_micro_Veillonella": {
        "coefficient": 0.03,    # Doc2 §11.1
        "reference_pct": 0.01,  # Doc2 §11.1: Veillonella% / 0.01%
        "floor": 1.00,          # contribuição aditiva acima de 1.0 apenas
    },
})


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODIFICADORES GENÉTICOS — Camada G (imutável)
# ─────────────────────────────────────────────────────────────────────────────
# Mj^G(S) — constantes calculadas uma vez no onboarding.
# Convenção: 1.0 = genótipo de referência da população geral.
# Fonte primária: Doc2 (tabelas §5.2, §6.2, §7.x); Doc1 §2.7 como secundária.
# Discrepâncias D3-D6 resolvidas em favor do Doc2 (ver cabeçalho).

GENETIC_MODIFIERS: MappingProxyType = MappingProxyType({

    # ── PPARGC1A Gly482Ser — biogénese mitocondrial ───────────────────────────
    # Afecta: VO2max trainability (Doc2 §5.2), MPS via ATP disponível (Doc1 §3.4)
    # Doc2 §5.2 (DISCREPÂNCIA D3: Doc1 §2.7 dá Gly/Ser=0.95)
    "PPARGC1A": MappingProxyType({
        "Gly_Gly": 1.00,    # Doc2 §5.2
        "Gly_Ser": 0.94,    # Doc2 §5.2 — DISCREPÂNCIA D3 (Doc1: 0.95)
        "Ser_Ser": 0.88,    # Doc2 §5.2 e Doc1 §2.7 (concordam)
    }),

    # ── ACE I/D — débito cardíaco e trainability aeróbia ─────────────────────
    # DISCREPÂNCIA D4: Doc1 §2.7 dá II=1.05, ID=1.02, DD=0.98
    "ACE": MappingProxyType({
        "II": 1.04,    # Doc2 §5.2
        "ID": 1.00,    # Doc2 §5.2 (referência)
        "DD": 0.96,    # Doc2 §5.2
    }),

    # ── HIF1A Pro582Ser — resposta hipóxica e EPO ────────────────────────────
    # HIF-1α mais estável → maior expressão de EPO/VEGF.
    "HIF1A": MappingProxyType({
        "Pro_Pro": 1.00,    # Doc2 §5.2
        "Pro_Ser": 1.02,    # Doc2 §5.2
        "Ser_Ser": 1.04,    # Doc2 §5.2
    }),

    # ── VEGF rs2010963 — angiogénese muscular ───────────────────────────────
    "VEGF": MappingProxyType({
        "GG": 1.02,    # Doc2 §5.2
        "GC": 1.00,    # Doc2 §5.2 (referência)
        "CC": 0.96,    # Doc2 §5.2
    }),

    # ── Haplotipo mitocondrial — eficiência OXPHOS ────────────────────────────
    # Afecta: VO2max (Doc2 §5.2).
    "MTDNA_HAPLOGROUP": MappingProxyType({
        "H": 1.00,    # Doc2 §5.2 (referência)
        "J": 0.95,    # Doc2 §5.2 — menor eficiência OXPHOS
        "K": 0.97,    # Doc2 §5.2
    }),

    # ── ACTN3 R577X — fibras tipo IIx ────────────────────────────────────────
    # Afecta VO2max e Power de forma OPOSTA (tradeoff documentado):
    # VO2max: XX ligeiramente favorável (mais características aeróbias nas IIx)
    # Power/RFD: XX desfavorável (ausência de α-actinina-3 reduz tensão específica)
    # Fatmax: XX favorável (tradeoff de potência para endurance — Doc1 §3.3)
    "ACTN3_vo2max": MappingProxyType({
        "RR": 0.99,    # Doc2 §5.2
        "RX": 1.00,    # Doc2 §5.2 (referência)
        "XX": 1.02,    # Doc2 §5.2
    }),

    # DISCREPÂNCIA D5: Doc1 §2.7 usa RR=1.00 como referência.
    "ACTN3_power": MappingProxyType({
        "RR": 1.05,    # Doc2 §6.2
        "RX": 1.00,    # Doc2 §6.2 (referência)
        "XX": 0.86,    # Doc2 §6.2 — Doc1 §2.7 dá 0.85
    }),

    # Doc1 §3.3: G_ACTN3_Fat: RR=0.97, RX=1.00, XX=1.03
    "ACTN3_fatmax": MappingProxyType({
        "RR": 0.97,    # Doc1 §3.3
        "RX": 1.00,    # Doc1 §3.3 (referência)
        "XX": 1.03,    # Doc1 §3.3
    }),

    # ── MSTN — miostatina, inibição miogénica ────────────────────────────────
    # Doc2 §6.2. Range: 1.08-1.15 para baixa expressão (não ponto único).
    "MSTN": MappingProxyType({
        "low_expression": 1.10,    # Doc2 §6.2 — midpoint do range [1.08, 1.15]
        "normal": 1.00,            # referência
        "high_expression": 0.92,   # Doc2 §6.2
    }),

    # ── IGF1 promoter — sinalização anabólica ────────────────────────────────
    # Doc2 §6.2.
    "IGF1_promoter": MappingProxyType({
        "high_expression": 1.04,    # Doc2 §6.2
        "normal": 1.00,             # referência
        "low_expression": 0.94,     # Doc2 §6.2
    }),

    # ── MYH7 variant — miosina de cadeia pesada tipo I ───────────────────────
    # Referenciado em Doc1 §3.5: G_Power = G_ACTN3 × G_MYH.
    # Valores específicos NÃO declarados nos documentos → None.
    "MYH7": MappingProxyType({
        "normal": 1.00,
        "variant": None,    # NÃO declarado nos documentos
    }),

    # ── COL5A1 rs12722 — rigidez e UTS tendinosa ─────────────────────────────
    # Doc1 §2.7. TT = maior rigidez → +3-5% SSC (ciclo estiramento-encurtamento).
    "COL5A1": MappingProxyType({
        "TT": 1.05,    # Doc1 §2.7 e Doc2 §4.3
        "TC": 1.00,    # referência
        "CC": 0.93,    # Doc1 §2.7
    }),

    # ── COL1A1 Sp1 — densidade de colagénio ósseo ────────────────────────────
    # Doc2 §4.3 (contexto de epistasia).
    "COL1A1": MappingProxyType({
        "ss": None,    # valor individual não declarado; só ε em pares
        "Ss": 1.00,    # referência
        "SS": None,
    }),

    # ── MMP3 rs679620 — degradação da matriz extracelular ───────────────────
    # Doc2 §4.3 (contexto de epistasia). Valores individuais: None.
    "MMP3": MappingProxyType({
        "6A_6A": None,    # desvantagem estrutural (maior degradação MEC)
        "5A_6A": 1.00,    # referência
        "5A_5A": None,    # referenciado em epistasia (Doc2 §4.3) mas valor individual None
    }),

    # ── MTHFR C677T — ciclo de metionina e SAM ───────────────────────────────
    # Afecta MPS via SAM → síntese de creatina e carnitina.
    # DISCREPÂNCIA D6: Doc1 §2.7 dá TT=0.92; Doc2 §7.4 dá CT=0.93, TT=0.82.
    "MTHFR_C677T": MappingProxyType({
        "CC": 1.00,    # Doc1 §2.7 e Doc2 §7.4 (concordam)
        "CT": 0.93,    # Doc2 §7.4 — DISCREPÂNCIA D6 (Doc1 não dá CT isolado)
        "TT": 0.82,    # Doc2 §7.4 — DISCREPÂNCIA D6 (Doc1: 0.92)
    }),

    # ── ADRB2 Arg16Gly — receptor β2-adrenérgico (lipólise) ─────────────────
    # Doc1 §2.7 e Doc2 §8.1 (contexto Gargalo 1 Fatmax).
    "ADRB2": MappingProxyType({
        "Arg_Arg": 1.00,    # Doc1 §2.7 (referência)
        "Arg_Gly": 0.96,    # estimativa linear — valor individual não declarado
        "Gly_Gly": 0.92,    # Doc1 §2.7
    }),

    # ── DRD2 TaqIA — densidade de receptor D2 (drive dopaminérgico) ──────────
    # Doc2 §6.2.
    "DRD2": MappingProxyType({
        "A2_A2": 1.02,    # Doc2 §6.2
        "A1_A2": 1.00,    # referência
        "A1_A1": 0.97,    # estimativa — valor não declarado explicitamente
    }),

    # ── OPRM1 A118G — limiar de dor (tolerância a contracção máxima) ─────────
    # Doc2 §6.2.
    "OPRM1": MappingProxyType({
        "A_A": 1.00,    # referência
        "A_G": 1.01,    # estimativa linear
        "G_G": 1.03,    # Doc2 §6.2
    }),

    # ── MTHFR C677T — modificador cognitivo (distinto do MPS) ───────────────
    # Doc2 §4.3: valor individual para o sistema SNC cognitivo.
    # SAM deficit → menor metilaçao de fosfolipidos da mielina e clearance de DA.
    # TT=0.75 no sistema cognitivo (vs. 0.82 no sistema MPS — mecanismos distintos).
    "MTHFR_C677T_cognitive": MappingProxyType({
        "CC": 1.00,    # Doc2 §4.3 (referência)
        "CT": None,    # não declarado individualmente para cognição
        "TT": 0.75,    # Doc2 §4.3 — SAM deficit → mielina e DA clearance comprometidos
    }),

    # ── COMT Val158Met — clearance dopaminérgico pré-frontal ────────────────
    # Doc2 §4.3: COMT Met/Met → menor actividade enzimática → DA elevada no PFC.
    # Met/Met: M=0.82 no contexto do tecto cognitivo integrado (Doc2 §4.3).
    "COMT_Val158Met": MappingProxyType({
        "Val_Val": 1.00,    # Doc2 §4.3 (referência: clearance rápido)
        "Val_Met": None,    # não declarado individualmente
        "Met_Met": 0.82,    # Doc2 §4.3 — clearance lento; pior em stress extremo
    }),

    # ── PER3 VNTR — necessidade de sono e recovery rate ─────────────────────
    # Doc1 §2.7 (recovery rate) e Doc2 §7.3 (TST_genético).
    "PER3_VNTR": MappingProxyType({
        "4_4": 1.00,    # Doc1 §2.7 (referência)
        "4_5": 0.95,    # interpolação — não declarado
        "5_5": 0.90,    # Doc1 §2.7: "requer ≥8h sono; sem isso: −10% recovery"
    }),
})


# ─────────────────────────────────────────────────────────────────────────────
# 4. COEFICIENTES DE EPISTASIA — ε (interacções entre pares de SNPs)
# ─────────────────────────────────────────────────────────────────────────────
# Fonte exclusiva: Doc2 §4.3. Cinco pares documentados.
# M_composto = M_snp1 × M_snp2 × ε
# ε < 1.0: epistasia negativa (antagonismo — produto real < produto simples)
# ε > 1.0: epistasia positiva (sinergia)

EPISTASIS_COEFFICIENTS: MappingProxyType = MappingProxyType({
    # ε = 0.72 — MTHFR TT priva a COMT Met/Met do seu substrato SAM.
    # MTHFR TT: M=0.75, COMT Met/Met: M=0.82
    # Produto simples: 0.75 × 0.82 = 0.615
    # Com epistasia:   0.75 × 0.82 × 0.72 = 0.443 (−28% adicional vs. produto)
    ("MTHFR_C677T_TT", "COMT_Val158Met_MM"): 0.72,   # Doc2 §4.3

    # ε = 0.88 — convergência na direcção de menor capacidade aeróbia.
    ("PPARGC1A_Gly482Ser_SS", "ACE_DD"): 0.88,        # Doc2 §4.3

    # ε = 0.61 — "triângulo de fragilidade" do tecido conjuntivo.
    # Tripla epistasia negativa: tendões mais complacentes + colagénio mais
    # fraco + degradação excessiva da matriz = risco de lesão multiplicado.
    ("COL5A1_CC", "COL1A1_ss", "MMP3_5A_5A"): 0.61,  # Doc2 §4.3

    # ε = 0.79 — dupla limitação no metabolismo purínico durante sprints.
    ("ACTN3_R577X_XX", "AMPD1_Q12X_XX"): 0.79,        # Doc2 §4.3

    # ε = 0.83 — três genes convergindo em maior adiposidade e menor
    # sensibilidade insulínica.
    ("VDR_low_function", "FTO_AA", "PPARG_Pro12_low"): 0.83,  # Doc2 §4.3
})


# ─────────────────────────────────────────────────────────────────────────────
# 5. CINÉTICA DE HILL — parâmetros explícitos dos documentos
# ─────────────────────────────────────────────────────────────────────────────

HILL_KINETICS: MappingProxyType = MappingProxyType({
    # Inibição de HSL pela insulina — Gargalo 1 do Fatmax.
    # Doc2 §8.1: "M_insulina^lipólise = 1/(1 + ([Insulina]/IC₅₀,HSL)^n)
    # onde IC₅₀,HSL ≈ 10 µU/mL e n ≈ 2"
    "insulin_HSL_inhibition": {
        "ic50": 10.0,    # µU/mL — Doc2 §8.1
        "n": 2.0,        # Doc2 §8.1
    },

    # Inibição competitiva da CPT-1 pelo malonil-CoA — Gargalo 2 do Fatmax.
    # Doc1 §2.3: "malonil-CoA inibe a CPT-1 com Ki ≈ 0.02 µM"
    # O coeficiente de proporcionalidade entre HOMA-IR e [malonil-CoA]
    # NÃO está declarado → None.
    "malonyl_coa_CPT1_inhibition": {
        "ki_umol_l": 0.02,         # µM — Doc1 §2.3
        "homa_to_malonyl_k": None, # NÃO declarado nos documentos
    },

    # Equação de mTORC1 (Doc2 §7.2) — estrutura documentada, parâmetros: None.
    # "Actividade = [Insulin^a1/(K1^a1 + ...)] × [Leucina^a2/...] × ..."
    "mTORC1_insulin_igf1": {
        "km": None,    # K1 — NÃO declarado
        "n": None,     # a1 — NÃO declarado
    },
    "mTORC1_leucine": {
        "km": None,    # K2 — NÃO declarado
        "n": None,     # a2 — NÃO declarado
    },
    "mTORC1_energy": {
        "km": None,    # K3 — NÃO declarado
        "n": None,     # a3 — NÃO declarado
    },
    "mTORC1_AMPK_k4": None,        # K4 — NÃO declarado

    # Inibição de CPT-I pelo malonil-CoA (Km para carnitina, sem inibidor).
    # Doc1 §2.3 menciona "Km da CPT-1 para carnitina ≈ 500 µM (mM range)
    # mas a concentração intracelular mitocondrial é o factor limitante" —
    # valor explícito de Km basal: None (não declarado com precisão).
    "CPT1_carnitine_basal_km": None,  # NÃO declarado com precisão nos documentos
})


# ─────────────────────────────────────────────────────────────────────────────
# 6. TAXAS DE FECHAMENTO DE GAP (RGC) — por sistema
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc2 §18.2 — Tabela de Taxas Base de Adaptação.
# Unidade: fracção do gap fechada por semana de treino óptimo (T_modifier=1.0).
# Dois valores por sistema: sem genótipo favorável / com genótipo favorável.

RGC_RATES: MappingProxyType = MappingProxyType({
    # Doc2 §18.2: "0.015-0.025%/semana"
    # Constante biológica limitante: biogénese mitocondrial τ ≈ 6-8 semanas.
    "vo2max": {"without_favorable": 0.015, "with_favorable": 0.025},

    # Doc2 §18.2: "0.020-0.035%/semana"
    "force_max": {"without_favorable": 0.020, "with_favorable": 0.035},

    # Doc2 §18.2: "0.010-0.018%/semana"
    # Constante: enzimas de β-oxidação τ ≈ 8-12 semanas.
    "fatmax": {"without_favorable": 0.010, "with_favorable": 0.018},

    # Doc2 §18.2: "0.018-0.030%/semana"
    "mps_hypertrophy": {"without_favorable": 0.018, "with_favorable": 0.030},

    # Doc2 §18.2: "1.0-2.0%/semana"
    "peak_power": {"without_favorable": 0.010, "with_favorable": 0.020},

    # Doc2 §18.2: "1.5-3.0%/semana"
    "rfd": {"without_favorable": 0.015, "with_favorable": 0.030},

    # Doc2 §18.2: "0.2-0.4%/semana"
    # Constante: turnover de colagénio τ ≈ 90-180 dias.
    "tendon_structural": {"without_favorable": 0.002, "with_favorable": 0.004},

    # Doc2 §18.2: "0.1-0.3%/semana"
    # Constante: ciclo de remodelação óssea τ ≈ 90-120 dias.
    "bone_structural": {"without_favorable": 0.001, "with_favorable": 0.003},

    # Doc2 §18.2: "0.05-0.15 unidades/semana" (HOMA-IR absoluto, não %)
    "insulin_resistance_homa": {"without_favorable": 0.05, "with_favorable": 0.15},
})


# ─────────────────────────────────────────────────────────────────────────────
# 7. PARÂMETROS DE SONO — PER3 VNTR e modificadores temporais
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc2 §7.3 (GH nocturno) e Doc2 §9.2 (PVT).

SLEEP_PARAMS: MappingProxyType = MappingProxyType({
    # TST_genético (Total Sleep Time mínimo) por genótipo PER3 VNTR.
    # Doc2 §7.3.
    "tst_genetic_hours": MappingProxyType({
        "PER3_4_4": 7.0,    # Doc2 §7.3
        "PER3_4_5": 7.75,   # interpolação — não declarado explicitamente
        "PER3_5_5": 8.5,    # Doc2 §7.3
    }),

    # SWS% óptimo para secreção pulsátil de GH — Doc2 §7.3.
    "sws_pct_optimal": 0.20,   # 20% — Doc2 §7.3

    # Taxa de decaimento de PVT por hora de défice de sono — Doc2 §9.2.
    "pvt_decay_rate_per_hour": 0.12,   # Doc2 §9.2

    # Multiplicadores da τ de recuperação tendinosa por factor de sono.
    # Doc2 §12.1: "Sono SWS < 10%: τ × 1.4".
    "tau_tendon_sleep_multiplier_sws_below_10pct": 1.4,   # Doc2 §12.1
})


# ─────────────────────────────────────────────────────────────────────────────
# 8. REFERÊNCIAS ÓPTIMAS DE BIOMARCADORES (x_ref onde modifier = 1.0)
# ─────────────────────────────────────────────────────────────────────────────
# Compilação dos x_ref declarados nos documentos para consulta rápida.
# Fonte: Doc1 §2.x e Doc2 §5.2.

BIOMARKER_REFERENCE_OPTIMAL: MappingProxyType = MappingProxyType({
    "hemoglobin_male_g_dl": 15.5,         # Doc1 §2.1
    "hemoglobin_female_g_dl": 13.5,       # Doc1 §2.1
    "hemoglobin_male_high_perf_g_dl": 16.0,  # Doc2 §5.2
    "bf_pct_male": 0.12,                  # Doc1 §2.1
    "bf_pct_female": 0.20,                # Doc1 §2.1
    "omega3_index_pct": 8.0,              # Doc1 §2.1
    "vitamin_d_25oh_ng_ml": 50.0,         # Doc1 §2.1 (aeróbio); Doc1 §2.4 (MPS)
    "vitamin_d_25oh_high_perf_ng_ml": 60.0,  # Doc2 §5.2
    "crp_mg_l": 0.3,                      # Doc1 §2.1
    "homa_ir": 1.0,                       # Doc1 §2.2 e §2.3
    "testosterone_male_ng_dl": 600.0,     # Doc1 §2.4 e §2.5
    "testosterone_female_ng_dl": 40.0,    # Doc1 §2.4
    "testosterone_male_nmol_l": 22.0,     # Doc2 §5.2 (unidades diferentes)
    "cortisol_am_ug_dl": 14.0,           # Doc1 §2.4
    "igf1_ng_ml": 180.0,                  # Doc1 §2.4
    "ft3_rt3_ratio": 10.0,               # Doc1 §2.3 (M=1.00 a partir de ratio≥10)
    "carnitine_free_umol_l": 40.0,       # Doc1 §2.3
    "zonulin_ng_ml": 20.0,               # Doc1 §2.2
    "shannon_diversity_index": 3.5,       # Doc1 §2.2
    "pcr_mmol_kg_dry": 80.0,             # Doc1 §2.5 (fosfocreatina)
    "cs_activity_umol_min_g": 25.0,      # Doc1 §3.3 (citrato sintase)
    "ferritin_ng_ml_doc2": 80.0,         # Doc2 §5.2
    "t3_libre_pmol_l": 5.5,              # Doc2 §5.2
    "t_score_bmd_neutral": 0.0,          # Doc1 §2.6
    "pinp_ug_l": 60.0,                   # Doc1 §3.6
    "ctx_ng_ml": 0.4,                    # Doc1 §3.6
    "cortisol_dhea_s_ratio_optimal": 8.0,  # Doc2 §5.2
    "dunedinpace_neutral": 1.0,          # Doc2 §5.2
    "hba1c_pct_optimal": 5.0,           # Doc2 §9.1 (para NCV)
    "homocysteine_umol_l": 10.0,         # Doc2 §14.1
    "vitamin_c_umol_l": 60.0,            # Doc1 §2.6 (colágeno)
    "mma_umol_per_mmol_creatinine": 3.0,  # Doc1 §2.5 (B12 neural)
})


# ─────────────────────────────────────────────────────────────────────────────
# 9. MULTIPLICADORES DE τ DE RECUPERAÇÃO TENDINOSA
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc2 §12.1 — tabela de prolongamento de τ por factor.
# τ_base = 72h (Homo sapiens, condições óptimas — Doc2 §12.1).
# τ_real = τ_base × ∏ multiplicadores activos.

TENDON_RECOVERY_TAU_BASE_HOURS: float = 72.0  # Doc2 §12.1

TENDON_TAU_MULTIPLIERS: MappingProxyType = MappingProxyType({
    "COL5A1_CC": 1.4,             # Doc2 §12.1
    "COL1A1_ss": 1.5,             # Doc2 §12.1
    "MMP3_5A_5A": 1.3,            # Doc2 §12.1
    "vitamin_c_below_40_umol_l": 1.3,   # Doc2 §12.1 (cofactor prolil-hidroxilase)
    "testosterone_below_10_nmol_l": 1.2,  # Doc2 §12.1
    "sws_below_10_pct": 1.4,      # Doc2 §12.1
})


# ─────────────────────────────────────────────────────────────────────────────
# 10. MATRIZ DE ELASTICIDADE — IMPACTO FRACCIONAL DAS INTERVENÇÕES POR SISTEMA
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc2 §16 — Tabela de Elasticidades Críticas (derivada logarítmica).
# D_ki = ∂ln(T_real(Si)) / ∂ln(Mk)
# Interpretação: 1% de melhoria na intervenção k melhora o sistema i em D_ki%.
# Colunas: vo2max | cho | fatmax | mps | peak_power | structural
# peak_power agrega Força e TR Neural (média aritmética da tabela §16).
# cho: derivado da arquitectura dos modificadores glicométricos (§2.2, §3.2).

ELASTICITY_MATRIX: MappingProxyType = MappingProxyType({

    # Ferritina / reposição de ferro — Doc2 §16 + Doc2 §15 (grafo de propagação).
    # Maior alavancagem sistémica sobre VO2max (0.50) via transporte de O2 e CTE.
    "iron_repletion": MappingProxyType({
        "vo2max":      0.50,   # Doc2 §16 — dominante (Fick: CaO2 ∝ Hb)
        "cho":         0.05,   # indirecto via disponibilidade energética celular
        "fatmax":      0.25,   # Doc2 §16 — CTE mitocondrial
        "mps":         0.30,   # Doc2 §16 — síntese proteica energo-dependente
        "peak_power":  0.18,   # média (Força 0.20 + TR Neural 0.15) / 2 — Doc2 §16
        "structural":  0.10,   # Doc2 §16 — tendão (indirecto via testosterona)
    }),

    # Vitamina D — Doc2 §16 + Doc2 §12.2 + §5.2.
    # Alta elasticidade sobre structural (VDR osteoblastos, mineralização).
    "vitamin_d_correction": MappingProxyType({
        "vo2max":      0.25,   # Doc2 §16 — VDR cardiomiócitos + PGC-1α
        "cho":         0.08,   # indirecto via sensibilidade insulínica (VDR GLUT4)
        "fatmax":      0.15,   # Doc2 §16 — PGC-1α + biogénese mitocondrial
        "mps":         0.25,   # Doc2 §16 — VDR miócitos → IGF-1R + mTORC1
        "peak_power":  0.28,   # média (Força 0.35 + TR Neural 0.20) / 2 — Doc2 §16
        "structural":  0.40,   # Doc2 §16 — mineralização óssea (dominante)
    }),

    # Testosterona — Doc2 §16 + Doc1 §2.4 + §2.5.
    # Maior elasticidade sobre MPS (0.50) e structural (COL1A1 síntese).
    "testosterone_optimization": MappingProxyType({
        "vo2max":      0.15,   # Doc2 §16 — eritropoiese + função cardíaca
        "cho":         0.05,   # indirecto via translocação GLUT4
        "fatmax":      0.10,   # Doc2 §16 — lipólise adipocitária
        "mps":         0.50,   # Doc2 §16 — AR muscular dominante
        "peak_power":  0.28,   # média (Força 0.45 + TR Neural 0.10) / 2 — Doc2 §16
        "structural":  0.30,   # Doc2 §16 — mineralização óssea + síntese colagénio
    }),

    # Sensibilidade à insulina (redução HOMA-IR) — Doc2 §16 + Doc1 §2.2 + §2.3.
    # Maior elasticidade sobre Fatmax (0.60) via desinibição da CPT-1.
    "homa_intervention": MappingProxyType({
        "vo2max":      0.20,   # Doc2 §16 — função endotelial + eNOS
        "cho":         0.55,   # Doc1 §2.2: HOMA k=0.15 (mais agressivo que Fatmax)
        "fatmax":      0.60,   # Doc2 §16 — CPT-1 desinibição (dominante)
        "mps":         0.30,   # Doc2 §16 — sinalizaçào insulina/IGF-1 → mTORC1
        "peak_power":  0.13,   # média (Força 0.15 + TR Neural 0.10) / 2 — Doc2 §16
        "structural":  0.05,   # Doc2 §16 — efeito mínimo directo
    }),

    # Eixo tiroideo (FT3/rT3 ratio) — Doc2 §16 + Doc1 §2.3.
    # Maior elasticidade sobre Fatmax (0.35) via PPARα/CPT-1.
    "thyroid_optimization": MappingProxyType({
        "vo2max":      0.30,   # Doc2 §16 — débito cardíaco + mitocôndrias
        "cho":         0.08,   # indirecto via metabolismo basal
        "fatmax":      0.35,   # Doc2 §16 — PPARα + UCP1 (dominante)
        "mps":         0.25,   # Doc2 §16 — síntese proteica energo-dependente
        "peak_power":  0.18,   # média (Força 0.20 + TR Neural 0.15) / 2 — Doc2 §16
        "structural":  0.15,   # Doc2 §16 — remodelação óssea
    }),

    # Redução do ritmo de envelhecimento biológico (DunedinPACE) — Doc2 §16.
    # Elasticidade uniforme de 0.30 em todos os sistemas — Doc2 §16.
    "aging_pace_reduction": MappingProxyType({
        "vo2max":      0.30,   # Doc2 §16
        "cho":         0.20,   # Doc2 §16 (indirecto; não declarado explicitamente)
        "fatmax":      0.30,   # Doc2 §16
        "mps":         0.30,   # Doc2 §16
        "peak_power":  0.30,   # Doc2 §16
        "structural":  0.30,   # Doc2 §16
    }),

    # Reequilíbrio cortisol/DHEA-S — Doc2 §16.
    # Maior elasticidade sobre MPS (0.40) e eixo hormonal (0.55).
    "cortisol_dhea_rebalance": MappingProxyType({
        "vo2max":      0.25,   # Doc2 §16 — eritropoiese + eNOS
        "cho":         0.15,   # indirecto via resistência insulínica (Cortisol→REDD1)
        "fatmax":      0.25,   # Doc2 §16 — lipólise (Cortisol inibe HSL)
        "mps":         0.40,   # Doc2 §16 — REDD1 + MuRF-1 supressão (dominante)
        "peak_power":  0.28,   # média (Força 0.30 + TR Neural 0.25) / 2 — Doc2 §16
        "structural":  0.20,   # Doc2 §16 — sintese colagénio + remodelação óssea
    }),

    # Suplementação de ómega-3 — Doc2 §16 + Doc1 §2.1.
    # Maior elasticidade sobre TR Neural (0.30) via mielinização DHA.
    "omega3_supplementation": MappingProxyType({
        "vo2max":      0.15,   # Doc2 §16 — cardiolipina mitocondrial
        "cho":         0.10,   # anti-inflamatório → integridade barreira intestinal
        "fatmax":      0.20,   # Doc2 §16 — eficiência mitocondrial
        "mps":         0.15,   # Doc2 §16 — anti-inflamatório → IRS-1 protecção
        "peak_power":  0.20,   # média (Força 0.10 + TR Neural 0.30) / 2 — Doc2 §16
        "structural":  0.25,   # Doc2 §16 — DHA na bainha de mielina tendinosa
    }),

    # Restauração do microbioma (diversidade Shannon) — Doc2 §16 + Doc1 §2.2.
    # Maior impacto na absorção intestinal de CHO via SCFA/butirato.
    "microbiome_restoration": MappingProxyType({
        "vo2max":      0.10,   # Doc2 §16 — Veillonella propionato → gluconeogénese
        "cho":         0.25,   # Doc1 §2.2: butirato → SGLT1 expressão
        "fatmax":      0.15,   # Doc2 §16 — SCFA como substrato alternativo
        "mps":         0.10,   # Doc2 §16 — absorção de EAA
        "peak_power":  0.13,   # média (Força 0.05 + TR Neural 0.20) / 2 — Doc2 §16
        "structural":  0.05,   # Doc2 §16 — efeito mínimo directo
    }),

    # Correcção de permeabilidade intestinal (Zonulina) — Doc2 §16 + Doc1 §2.2.
    # Dominante sobre absorção de CHO e MPS via leucina.
    "gut_permeability_correction": MappingProxyType({
        "vo2max":      0.10,   # Doc2 §16 — LPS → inflamação → eritropoiese
        "cho":         0.30,   # Doc1 §2.2: NF-κB reprime SGLT1 (dominante CHO)
        "fatmax":      0.10,   # Doc2 §16 — indirecto via inflamação
        "mps":         0.15,   # Doc2 §16 — biodisponibilidade leucina → mTORC1
        "peak_power":  0.08,   # média (Força 0.05 + TR Neural 0.10) / 2 — Doc2 §16
        "structural":  0.05,   # Doc2 §16 — mínimo
    }),

    # Correcção de homocisteína (vitaminas B) — Doc2 §16 + Doc2 §14.1.
    # Maior elasticidade sobre TR Neural (0.35) via perfusão cerebrovascular.
    "homocysteine_correction": MappingProxyType({
        "vo2max":      0.05,   # Doc2 §16 — eNOS endotelial
        "cho":         0.05,   # indirecto via MTHFR → SAM → carnitina
        "fatmax":      0.05,   # Doc2 §16 — mínimo
        "mps":         0.10,   # Doc2 §16 — SAM → creatina síntese (GAMT)
        "peak_power":  0.20,   # média (Força 0.05 + TR Neural 0.35) / 2 — Doc2 §16
        "structural":  0.05,   # Doc2 §16 — mínimo
    }),

    # Reposição de magnésio eritrocitário — Doc2 §16.
    "magnesium_repletion": MappingProxyType({
        "vo2max":      0.20,   # Doc2 §16 — Na+/K+-ATPase + função cardíaca
        "cho":         0.08,   # cofactor de hexocinase/PFK
        "fatmax":      0.30,   # Doc2 §16 — Krebs (cofactor ATP-Mg) — dominante
        "mps":         0.20,   # Doc2 §16 — síntese proteica ribossomal
        "peak_power":  0.23,   # média (Força 0.25 + TR Neural 0.20) / 2 — Doc2 §16
        "structural":  0.15,   # Doc2 §16 — mineralização óssea (cofactor)
    }),

    # Gut training (CHO exposure protocol) — Doc1 §2.2 + §3.2 + §4.3.
    # Exclusivamente CHO (SGLT1 upregulation): 3-5%/semana nas primeiras 8 semanas.
    "gut_training_cho": MappingProxyType({
        "vo2max":      0.00,   # sem impacto directo
        "cho":         0.60,   # Doc1 §3.2: GutTrain 0.70→1.00 = +43% do tecto CHO
        "fatmax":      0.05,   # indirecto via disponibilidade energética
        "mps":         0.05,   # mínimo
        "peak_power":  0.00,
        "structural":  0.00,
    }),

    # Optimização do sono (SWS + TST) — Doc2 §7.3 + §9.2 + §12.1.
    # Maior leverage geral: GH pulsátil + PVT neural + tau tendinoso.
    "sleep_optimization": MappingProxyType({
        "vo2max":      0.20,   # Doc2 §7.3: GH nocturno → biogénese mitocondrial
        "cho":         0.05,   # indirecto via sensibilidade insulínica
        "fatmax":      0.10,   # indirecto via cortisol reduzido
        "mps":         0.30,   # Doc2 §7.3: GH pulsátil → MPS (SWS% crítico)
        "peak_power":  0.25,   # Doc2 §9.2: PVT — M_sleep_PVT dominante
        "structural":  0.20,   # Doc2 §12.1: tau_tendon_sleep_multiplier SWS<10%
    }),

    # Correcção de vitamina C — Doc1 §2.6 + Doc2 §12.1.
    # Exclusivamente structural: prolil-4-hidroxilase → cross-links colagénio.
    "vitamin_c_correction": MappingProxyType({
        "vo2max":      0.03,   # eNOS endotelial (antioxidante) — mínimo
        "cho":         0.00,
        "fatmax":      0.00,
        "mps":         0.05,   # indirecto via síntese de carnitina (requer VitC)
        "peak_power":  0.03,   # mínimo
        "structural":  0.35,   # Doc1 §2.6: UTS tendão — dominante
    }),

    # Creatina loading (PCr pool) — Doc1 §2.5 + §4.2.
    # Exclusivamente peak_power via M_PCr_Power; τ_loading ≈ 2-4 semanas.
    "creatine_loading": MappingProxyType({
        "vo2max":      0.00,
        "cho":         0.05,   # PCr reposiçõa anaeróbia → poupar glicogénio
        "fatmax":      0.00,
        "mps":         0.05,   # PCr → ATP para MPS post-treino
        "peak_power":  0.55,   # Doc1 §2.5: M_PCr [PCr]/80 — dominante
        "structural":  0.00,
    }),

    # Correcção do turnover ósseo (PINP/CTX ratio) — Doc1 §3.6.
    # Exclusivamente structural: balanço formação/reabsorção.
    "bone_turnover_correction": MappingProxyType({
        "vo2max":      0.00,
        "cho":         0.00,
        "fatmax":      0.00,
        "mps":         0.00,
        "peak_power":  0.00,
        "structural":  0.30,   # Doc1 §3.6: M_PINP_CTX [0.60, 1.40]
    }),
})


# ─────────────────────────────────────────────────────────────────────────────
# 11. TEMPO ATÉ AO EFEITO MENSURÁVEL (semanas)
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc2 §4.3 taxas de adaptação + Doc1 §6.1 exemplos + constantes τ §12.1.
# Valor = tempo mínimo em semanas para o modificador atingir o novo estado estável.

TIME_TO_EFFECT_WEEKS: MappingProxyType = MappingProxyType({
    "iron_repletion":             6,    # eritropoiese: τ ≈ 4-8 semanas — Doc2 §15
    "vitamin_d_correction":       8,    # saturação tecidual: τ ≈ 6-12 semanas — Doc1 §6.1
    "testosterone_optimization":  12,   # eixo HPG: τ ≈ 8-16 semanas — Doc2 §13.2
    "homa_intervention":          10,   # sensibilidade insulínica: 10-20 semanas — Doc2 §4.3
    "thyroid_optimization":       6,    # eixo tiroideo: τ ≈ 4-8 semanas
    "aging_pace_reduction":       24,   # remodelação epigenética: meses — Doc2 §3.2
    "cortisol_dhea_rebalance":    6,    # eixo HPA: τ ≈ 4-8 semanas — Doc2 §13.2
    "omega3_supplementation":     12,   # incorporação membranar: τ ≈ 6-12 semanas — Doc1 §6.1
    "microbiome_restoration":     6,    # ecologia intestinal: τ ≈ 4-8 semanas
    "gut_permeability_correction": 8,   # reparação barreira: τ ≈ 6-12 semanas
    "homocysteine_correction":    4,    # metilação B12/folato: τ ≈ 2-6 semanas — Doc2 §14.1
    "magnesium_repletion":        4,    # Mg eritrocitário: τ ≈ 3-6 semanas
    "gut_training_cho":           8,    # SGLT1 upregulation: τ ≈ 4-8 semanas — Doc2 §4.3
    "sleep_optimization":         2,    # adaptação comportamental: τ ≈ 1-4 semanas
    "vitamin_c_correction":       3,    # repleção cofactor: τ ≈ 2-4 semanas — Doc1 §2.6
    "creatine_loading":           3,    # loading PCr: τ ≈ 2-4 semanas — Doc1 §4.2
    "bone_turnover_correction":   16,   # ciclo remodelação óssea: τ ≈ 12-24 semanas — Doc2 §12.1
})


# ─────────────────────────────────────────────────────────────────────────────
# 12. LEVERAGE DE TREINABILIDADE (multiplicador de resposta ao treino)
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc2 §4.2 — Modificadores de Treinabilidade por Sistema.
# Interpretação: se a intervenção aumenta também a trainability (T_modifier),
# o impacto no priority_score é multiplicado por este factor.
# 1.0 = sem efeito adicional sobre trainability; >1.0 = desbloqueia trainability.
# T_VitD: 0.70→1.00 (+43%) | T_Sleep: 0.60→1.00 (+67%) | T_Creatine: 1.0→1.20 (+20%)

TRAINABILITY_LEVERAGE: MappingProxyType = MappingProxyType({
    "iron_repletion":             2.5,  # gargalo O2 — desbloqueia trainability total
    "vitamin_d_correction":       2.0,  # T_VitD 0.70→1.00 — Doc2 §4.2
    "testosterone_optimization":  1.8,  # eixo anabólico HPG
    "homa_intervention":          1.5,  # metabolismo sistémico
    "thyroid_optimization":       1.6,  # taxa metabólica basal
    "aging_pace_reduction":       1.3,  # epigenético — resposta lenta
    "cortisol_dhea_rebalance":    1.8,  # ratio anabólico:catabólico
    "omega3_supplementation":     1.4,  # membrana + anti-inflamatório
    "microbiome_restoration":     1.3,  # absorção indirecta
    "gut_permeability_correction": 1.5, # eficiência absorção
    "homocysteine_correction":    1.5,  # vascular + neural
    "magnesium_repletion":        1.6,  # cofactor enzimático ubíquo
    "gut_training_cho":           2.0,  # SGLT1 — desbloqueia tecto CHO directamente
    "sleep_optimization":         2.5,  # GH pulsátil + PVT — maior leverage — Doc2 §4.2
    "vitamin_c_correction":       1.8,  # cofactor essencial colagénio — Doc1 §2.6
    "creatine_loading":           2.2,  # substrato directo sistema PCr — Doc2 §4.2
    "bone_turnover_correction":   1.2,  # remodelação lenta
})


# ─────────────────────────────────────────────────────────────────────────────
# 13. ALELOS FAVORÁVEIS POR SISTEMA (treinabilidade genética)
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc2 §18.2 — Taxas de Adaptação Diferenciadas por Genótipo.
# Estrutura: sistema → tuplo de pares (campo_genótipo, valor_favorável).
# Interpretação: ANY match no tuplo → atleta tem genótipo favorável para o sistema.
# cho: G_CHO = 1.0 (Doc1 §3.2) — sem modificador genético declarado.

FAVORABLE_ALLELES_PER_SYSTEM: MappingProxyType = MappingProxyType({
    "vo2max": (
        ("ACE",   "II"),        # alelo endurance dominante — Doc2 §5.2
        ("HIF1A", "Ser/Ser"),   # resposta hipóxia óptima — Doc2 §5.2
    ),
    "cho": (),                  # nenhum SNP declarado — Doc1 §3.2
    "fatmax": (
        ("ACTN3", "XX"),        # alelo nulo α-actinina-3: favorece fatmax — Doc2 §8.2
    ),
    "mps": (
        ("MSTN",          "low"),   # miostatina reduzida → hipertrofia — Doc2 §7.x
        ("IGF1_promoter", "high"),  # IGF-1 aumentado → mTORC1 — Doc2 §7.x
    ),
    "peak_power": (
        ("ACTN3", "RR"),        # fast-twitch dominante — Doc2 §9.x
    ),
    "structural": (
        ("COL5A1", "TT"),       # rigidez tendinosa óptima — Doc2 §12.1
    ),
})


# ─────────────────────────────────────────────────────────────────────────────
# 14. MAPA DE INTERVENÇÃO LIEBIG (gargalo estrutural → intervenção correctora)
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc1 §3.6, Doc2 §12.1–12.2.
# Mapeamento: nome do modificador estrutural → chave de intervenção em ELASTICITY_MATRIX.
# NOTA CLÍNICA: M_Inflam_Structural reflecte supressão de colagénio via NF-κB;
# a intervenção correcta é anti-inflamatória de base (omega3), não reequilíbrio HPA.

LIEBIG_INTERVENTION_MAP: MappingProxyType = MappingProxyType({
    "M_BMD_structural":              "vitamin_d_correction",      # mineralização óssea — VDR osteoblastos
    "M_VitC_Collagen":               "vitamin_c_correction",      # cofactor prolil-hidroxilase
    "M_PINP_CTX_structural":         "bone_turnover_correction",  # balanço formação/reabsorção
    "M_VitD_mineralization":         "vitamin_d_correction",      # Doc2 §12.2 — osteóide
    "M_testosterone_mineralization": "testosterone_optimization", # Doc2 §12.2 — COL1A1
    "M_Inflam_Structural":           "omega3_supplementation",    # NF-κB → collagénio; omega3 ≠ HPA
})


# ─────────────────────────────────────────────────────────────────────────────
# 15. RED FLAGS CLÍNICAS (triagem sistémica pré-prioritização)
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc1 §3.6 (lei de Liebig sistémica), Doc2 §5.2, §9.2.
# Cada entrada define um limiar clínico que, se violado, força a intervenção
# correspondente para o topo absoluto das prioridades (forced_first: True),
# independentemente do score económico calculado pela matriz de elasticidades.
#
# Campos:
#   biomarker  — chave em athlete_data (após normalização SI)
#   operator   — "gt" (greater than) ou "lt" (less than)
#   threshold  — valor limiar clínico em unidades SI
#   note       — fundamento clínico resumido
#
# Múltiplas red flags podem ser acionadas simultaneamente.

CLINICAL_RED_FLAGS: MappingProxyType = MappingProxyType({
    "glycemic_rehabilitation": MappingProxyType({
        "biomarker":       "hba1c_pct",
        "operator":        "gt",
        "threshold":       6.0,    # HbA1c > 6.0% — limiar de pré-diabetes (ADA 2023)
        # urgency_weight: normaliza desvio em % HbA1c → score de severidade contínuo.
        # Calibrado para que HbA1c = 10.0 (desvio 4.0%) produza score ≈ 48.
        # Mecanismo: glicação crónica de MBP, supressão de mTORC1, disfunção mitocondrial.
        "urgency_weight":  12.0,
        "note":            "HbA1c > 6.0%: glicação crónica suprime mTORC1, MPS e NCV",
    }),
    "acute_inflammation_protocol": MappingProxyType({
        "biomarker":       "crp_mg_l",
        "operator":        "gt",
        "threshold":       5.0,    # CRP > 5.0 mg/L — inflamação sistémica activa
        # urgency_weight: normaliza desvio em mg/L → score contínuo.
        # Calibrado para que CRP = 50 mg/L (sépsis) produza score ≈ 67.5.
        # Mecanismo: NF-κB → TNF-α/IL-6 → supressão HPG, proteólise muscular, CTE.
        "urgency_weight":  1.5,
        "note":            "CRP > 5.0 mg/L: estado inflamatório agudo/subagudo — treino potencia dano",
    }),
    "clinical_anemia_correction": MappingProxyType({
        "biomarker":       "hemoglobin_g_dl",
        "operator":        "lt",
        "threshold":       13.0,   # Hb < 13.0 g/dL — anemia clínica (WHO, homem/mulher)
        # urgency_weight: normaliza desvio em g/dL → score contínuo.
        # Calibrado para que Hb = 7.0 g/dL (desvio 6.0) produza score ≈ 48.
        # Mecanismo: CaO2 ↓ → VO2max colapsado; AMPK ↑ → mTORC1 ↓ → MPS inibida.
        "urgency_weight":  8.0,
        "note":            "Hb < 13.0 g/dL: colapso de transporte de O2 — VO2max e MPS colapsam",
    }),
})


# ─────────────────────────────────────────────────────────────────────────────
# 16. PARÂMETROS DE DEGRADAÇÃO DA TRAJETÓRIA RGC (T_modifiers de lifestyle)
# ─────────────────────────────────────────────────────────────────────────────
# Fonte: Doc2 §4.2 — fórmula RGC_efectivo = RGC_genético × M_sleep × (2.0 − DunedinPACE).
# Extrai os literais numéricos do código de trainability para cumprir a regra
# "zero numeric literals em business logic".

TRAJECTORY_DEGRADATION_PARAMS: MappingProxyType = MappingProxyType({
    # Base do factor DunedinPACE: (base − pace) → 1.0 quando pace = 1.0 (neutro).
    # pace > 1.0 → factor < 1.0 (envelhecimento acelerado comprime a adaptação).
    # Doc2 §4.2: "RGC_efectivo = RGC_genético × M_sleep × (2.0 − DunedinPACE)"
    "dunedin_pace_base": 2.0,

    # Floor do factor de pace — impede colapso total do RGC em envelhecedores extremos.
    # Doc2 §4.2 (implícito na tabela §18.2): floor garante RGC mínimo de 30% do genético.
    "dunedin_pace_floor": 0.30,
})


# ─────────────────────────────────────────────────────────────────────────────
# 17. MAPEAMENTO PRIORS BAYESIANOS — Fase 3 → MHDS (TDD §4)
# ─────────────────────────────────────────────────────────────────────────────
# Tabela de correspondência entre campos de entrada (biomarkers/SNPs da Fase 2)
# e as constantes físicas iniciais consumidas pelos ODEs do MHDS.
# Fonte: NOVO_ENGINE_NUTRIVIOUS_1.1.txt §4 (MHDS Subsystem Parameter Table).
#
# Campos obrigatórios por entrada:
#   prior_name   — nome canónico do parâmetro no vector de estado do MHDS
#   subsystem    — um dos 13 subsistemas MHDS
#   model        — referência ao modelo matemático que consome este prior
#   transform    — tipo de transformação a aplicar ao valor bruto de entrada
#   units        — unidades do prior após transformação
#
# Transforms disponíveis (implementados em vtp_composer.compose_bayesian_priors):
#   "direct"                  → prior = valor bruto (sem conversão)
#   "reciprocal_normalised"   → prior = reference_value / valor (ex: HOMA-IR)
#   "direct_mg_dL_to_mmol_L" → prior = valor / 18.0182
#   "actn3_allele_to_tau"     → lookup ACTN3_TAU_FATIGUE_DAYS
#   "ace_allele_to_efficiency"→ lookup ACE_CARDIAC_EFFICIENCY
#   "mstn_allele_to_inhibition"→lookup MSTN_INHIBITION_FACTOR
#   "msf_to_phi_rad"          → prior = (valor − reference_value) × (π/12)
#   "pvt_to_alertness_rate"   → prior = reference_value / valor
#   "diversity_to_eta"        → prior = min(1.0, valor / reference_value)

BAYESIAN_PRIOR_MAPPINGS: MappingProxyType = MappingProxyType({
    # ── Subsistema 1: Bioenergético / Metabólico ──────────────────────────────
    # Modelo: Bergman Minimal Model (Bergman et al. 1979)
    # Parâmetros físicos: SI (insulin sensitivity index), Gb (basal glucose)
    "homa_ir": MappingProxyType({
        "prior_name":      "p3_insulin_sensitivity_prior",
        "subsystem":       "bioenergetic_metabolic",
        "model":           "Bergman_Minimal_Model_1979",
        "transform":       "reciprocal_normalised",
        "reference_value": 1.0,
        "units":           "a.u.",
    }),
    "glucose_fasting_mg_dL": MappingProxyType({
        "prior_name":      "p3_basal_glucose_prior",
        "subsystem":       "bioenergetic_metabolic",
        "model":           "Bergman_Minimal_Model_1979",
        "transform":       "direct_mg_dL_to_mmol_L",
        "reference_value": 85.0,
        "units":           "mmol·L⁻¹",
    }),
    "respiratory_quotient_rest": MappingProxyType({
        "prior_name":      "p3_substrate_oxidation_prior",
        "subsystem":       "bioenergetic_metabolic",
        "model":           "Frayn_Substrate_Oxidation_1983",
        "transform":       "direct",
        "reference_value": 0.85,
        "units":           "a.u.",
    }),
    "PPARGC1A": MappingProxyType({
        "prior_name":  "p3_mitochondrial_efficiency_prior",
        "subsystem":   "bioenergetic_metabolic",
        "model":       "Baar_PGC1alpha_2004",
        "transform":   "ppargc1a_allele_to_oxphos",
        "units":       "a.u.",
    }),

    # ── Subsistema 2: Neuromuscular / Fadiga ──────────────────────────────────
    # Modelo: Busso Fatigue-Fitness Model (Busso et al. 1994); Haff mTOR (MPS)
    "ACTN3": MappingProxyType({
        "prior_name":  "tau_fatigue_decay_prior",
        "subsystem":   "neuromuscular_fatigue",
        "model":       "Busso_1994",
        "transform":   "actn3_allele_to_tau",
        "units":       "days",
    }),
    "MSTN": MappingProxyType({
        "prior_name":  "p3_myostatin_inhibition_prior",
        "subsystem":   "neuromuscular_fatigue",
        "model":       "Haff_mTOR",
        "transform":   "mstn_allele_to_inhibition",
        "units":       "a.u.",
    }),

    # ── Subsistema 3: Cardiorrespiratório / Autonómico ────────────────────────
    # Modelo: Puthucheary ACE-cardiac efficiency (2011); Goldberger HRV
    "ACE": MappingProxyType({
        "prior_name":  "p3_cardiac_efficiency_prior",
        "subsystem":   "cardiorespiratory_autonomic",
        "model":       "Puthucheary_ACE_2011",
        "transform":   "ace_allele_to_efficiency",
        "units":       "a.u.",
    }),
    "rmssd_baseline_ms": MappingProxyType({
        "prior_name":      "p3_vagal_tone_prior",
        "subsystem":       "cardiorespiratory_autonomic",
        "model":           "Goldberger_HRV",
        "transform":       "direct",
        "reference_value": 50.0,
        "units":           "ms",
    }),

    # ── Subsistema 4: Sono / Circadiano ───────────────────────────────────────
    # Modelo: Kronauer Circadian Pacemaker; Jewett-Kronauer 1999
    "mctq_msf_sc": MappingProxyType({
        "prior_name":      "p3_circadian_phase_prior",
        "subsystem":       "sleep_circadian",
        "model":           "Kronauer_Circadian",
        "transform":       "msf_to_phi_rad",
        "reference_value": 3.5,   # population median MSFsc ≈ 03:30
        "units":           "radians",
    }),
    "pvt_rt_baseline_ms": MappingProxyType({
        "prior_name":      "p3_alertness_decay_prior",
        "subsystem":       "sleep_circadian",
        "model":           "Jewett_Kronauer_1999",
        "transform":       "pvt_to_alertness_rate",
        "reference_value": 250.0,
        "units":           "ms",
    }),

    # ── Subsistema 5: HPA / HPG Neuroendócrino ────────────────────────────────
    # Modelo: Veldhuis GnRH pulsatile model
    "testosterone_pg_mL": MappingProxyType({
        "prior_name":      "p3_androgen_setpoint_prior",
        "subsystem":       "hpa_hpg_neuroendocrine",
        "model":           "Veldhuis_GnRH_pulsatile",
        "transform":       "direct",
        "reference_value": 500.0,
        "units":           "pg·mL⁻¹",
    }),

    # ── Subsistema 7: Imunológico / Inflamatório ──────────────────────────────
    # Modelo: Schindler IL-6 cascade model (2006)
    "crp_mg_L": MappingProxyType({
        "prior_name":      "p3_inflammatory_baseline_prior",
        "subsystem":       "immunologic_inflammatory",
        "model":           "Schindler_IL6_2006",
        "transform":       "direct",
        "reference_value": 0.5,
        "units":           "mg·L⁻¹",
    }),
    "nlr": MappingProxyType({
        "prior_name":      "p3_immune_activation_prior",
        "subsystem":       "immunologic_inflammatory",
        "model":           "Schindler_IL6_2006",
        "transform":       "direct",
        "reference_value": 2.0,
        "units":           "a.u.",
    }),
    "ck_u_L": MappingProxyType({
        "prior_name":      "p3_muscle_damage_baseline_prior",
        "subsystem":       "immunologic_inflammatory",
        "model":           "Paulsen_CK_Kinetics_2012",
        "transform":       "direct",
        "reference_value": 150.0,
        "units":           "U·L⁻¹",
    }),

    # ── Subsistema 8: Gastrointestinal / Absorção ─────────────────────────────
    # Modelo: Zmora gut microbiome absorption efficiency (2018)
    "shannon_diversity": MappingProxyType({
        "prior_name":      "p3_absorption_efficiency_prior",
        "subsystem":       "gi_absorption",
        "model":           "Zmora_Gut_2018",
        "transform":       "diversity_to_eta",
        "reference_value": 3.5,
        "units":           "a.u.",
    }),
    "MTHFR_C677T": MappingProxyType({
        "prior_name":  "p3_folate_methylation_prior",
        "subsystem":   "gi_absorption",
        "model":       "Trimmer_MTHFR_2013",
        "transform":   "mthfr_allele_to_efficiency",
        "units":       "a.u.",
    }),

    # ── Subsistema 9: Cognitivo / Fadiga Central ──────────────────────────────
    # Modelo: Beelen Central Fatigue model (2010)
    "pvt_rt_ms": MappingProxyType({
        "prior_name":      "p3_central_fatigue_tau_prior",
        "subsystem":       "cognitive_central_fatigue",
        "model":           "Beelen_Central_Fatigue_2010",
        "transform":       "pvt_to_alertness_rate",
        "reference_value": 250.0,
        "units":           "ms",
    }),

    # ── Subsistema 11: Biomecânico / Tecidual ─────────────────────────────────
    # Modelo: Roberts tendon spring model (2002)
    "tendon_stiffness_N_mm": MappingProxyType({
        "prior_name":      "p3_tendon_elasticity_prior",
        "subsystem":       "biomechanical_tissue",
        "model":           "Roberts_Tendons_2002",
        "transform":       "direct",
        "reference_value": 200.0,
        "units":           "N·mm⁻¹",
    }),

    # ── Subsistema 12: Hidroelectrolítico / Renal ─────────────────────────────
    # Modelo: Cook iron stores kinetics (2003)
    "ferritin_ng_mL": MappingProxyType({
        "prior_name":      "p3_iron_stores_prior",
        "subsystem":       "hydroelectrolytic_renal",
        "model":           "Cook_Iron_Stores_2003",
        "transform":       "direct",
        "reference_value": 80.0,
        "units":           "ng·mL⁻¹",
    }),
    "hemoglobin_g_dL": MappingProxyType({
        "prior_name":      "p3_oxygen_transport_prior",
        "subsystem":       "hydroelectrolytic_renal",
        "model":           "Fick_Oxygen_Transport",
        "transform":       "direct",
        "reference_value": 15.0,
        "units":           "g·dL⁻¹",
    }),

    # ── Subsistema 13: RED-S / LEA ────────────────────────────────────────────
    # Modelo: Belsky DunedinPACE biological ageing rate (2022)
    "DunedinPACE": MappingProxyType({
        "prior_name":      "p3_biological_age_rate_prior",
        "subsystem":       "reds_lea",
        "model":           "Belsky_DunedinPACE_2022",
        "transform":       "direct",
        "reference_value": 1.0,
        "units":           "years_epigenetic·year_calendar⁻¹",
    }),
    "measured_rmr": MappingProxyType({
        "prior_name":      "p3_energy_availability_prior",
        "subsystem":       "reds_lea",
        "model":           "Mountjoy_REDS_2018",
        "transform":       "direct",
        "reference_value": None,   # athlete-specific; no population reference
        "units":           "kcal·day⁻¹",
    }),
})

# Tabelas de lookup para transforms alelo → constante física
# Fonte: NOVO_ENGINE_NUTRIVIOUS_1.1.txt §4; Doc2 §5.2, §6.2, §7.4
ACTN3_TAU_FATIGUE_DAYS: MappingProxyType = MappingProxyType({
    "RR": 28.0,   # fast-twitch dominant — fatigue decays faster
    "RX": 35.0,   # heterozygous — intermediate
    "XX": 42.0,   # alpha-actinin-3 absent — fatigue lingers longer
})

ACE_CARDIAC_EFFICIENCY: MappingProxyType = MappingProxyType({
    "II": 1.10,   # insertion/insertion — higher efficiency, endurance advantage
    "ID": 1.00,   # reference heterozygous
    "DD": 0.90,   # deletion/deletion — lower cardiac efficiency
})

MSTN_INHIBITION_FACTOR: MappingProxyType = MappingProxyType({
    "KK": 0.85,   # myostatin loss-of-function — reduced inhibition → MPS ↑
    "KA": 1.00,   # heterozygous reference
    "AA": 1.15,   # wild-type — full myostatin activity
})

MTHFR_METHYLATION_EFFICIENCY: MappingProxyType = MappingProxyType({
    "CC": 1.00,   # wild-type — full enzyme activity
    "CT": 0.93,   # heterozygous — Doc2 §7.4
    "TT": 0.82,   # homozygous variant — Doc2 §7.4
})

# Tabela PPARGC1A Gly482Ser (rs8192678) → factor de eficiência mitocondrial
# PGC-1α regula a biogénese mitocondrial; alelo Ser reduz a transcrição de TFAM.
# Fonte: Baar et al. (2004) FASEB J 18:1175; Eynon et al. (2011) Eur J Appl Physiol.
# Usado pelo MetabolicSolver para escalonar Vmax_OXPHOS no dPCr/dt de Meyer.
PPARGC1A_OXPHOS_FACTOR: MappingProxyType = MappingProxyType({
    "GG": 1.15,   # Gly/Gly — expressão máxima de PGC-1α → Vmax_OXPHOS ↑ 15%
    "GA": 1.00,   # Gly/Ser — referência heterozigótico
    "AA": 0.85,   # Ser/Ser — biogénese mitocondrial reduzida → Vmax_OXPHOS ↓ 15%
})
