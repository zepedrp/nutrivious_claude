from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase2.athlete_core import AthleteCore


class AthleteDexa(Base):
    """
    Radiografia individual de composição corporal (DEXA 3-compartimento +
    bioimpedância multi-frequência + antropometria). Cada coluna é um calibrador
    da Fase 2 que escala ou condiciona uma constante da Fase 1.

    Fonte documental: Nutrivious BOS Fase 2 — Eixo 3 (Tabela 3.1) e Módulo D1.
    Frequência de medição base: trimestral (DEXA, BIA) e mensal (antropometria).
    Padronização: 07h00-09h00, jejum 10-12h, 48h sem exercício intenso.
    """

    __tablename__ = "athlete_dexa"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    athlete_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("athlete_core.id", ondelete="CASCADE"),
        nullable=False,
    )
    assessment_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment=(
            "Data/hora do scan DEXA ou BIA. Múltiplas linhas por atleta permitem "
            "monitorizar a trajectória de recomposição corporal ao longo do tempo. "
            "Frequência mínima: trimestral (DEXA completo); mensal (antropometria)."
        ),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 1 — MASSA CORPORAL E DIMENSÕES BASE
    # ═══════════════════════════════════════════════════════════════════════

    total_weight_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Massa corporal total (kg) medida na balança DEXA. "
            "Denominador primário para todas as constantes massa-relativas da Fase 1: "
            "volume sanguíneo (species_hematological: 70-80 mL/kg H, 60-70 mL/kg M), "
            "taxa de sudação máxima (species_thermoregulation: 3 L/h), "
            "capacidade de armazenamento de calor (Q = m × c × ΔT, c = 3.47 kJ/kg/°C). "
            "Frequência: trimestral (DEXA) / mensal (pesagem)."
        ),
    )
    height_cm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Estatura em pé (cm). Divisor para IMC (kg/m²), IAMM (kg/m²), "
            "Ratio Cintura/Altura, e normalização do VO2max (mL/min/kg). "
            "Único na vida após crescimento completo."
        ),
    )
    bmi_kg_per_m2: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Índice de Massa Corporal = peso(kg) / altura(m)². "
            "Fonte: Tabela 3.1 (Eixo 3). Limitado como marcador isolado — "
            "'IMC > 30 com gordura visceral alta é mais informativo que IMC sozinho'. "
            "Thresholds WHO: < 18.5 magreza, 18.5-24.9 normal, 25-29.9 excesso, ≥ 30 obesidade. "
            "Frequência: mensal (calculado em cada pesagem)."
        ),
    )
    body_surface_area_m2: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Área de superfície corporal (m²) pela fórmula DuBois: "
            "0.007184 × altura_cm^0.725 × peso_kg^0.425. "
            "Escala a perda de calor radiativa + convectiva (species_thermoregulation) "
            "e a evaporação máxima de suor."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2 — COMPOSIÇÃO CORPORAL TOTAL (DEXA 3-COMPARTIMENTO)
    # Fonte: Tabela 3.1 — Massa Gorda Total, Massa Magra Total
    # ═══════════════════════════════════════════════════════════════════════

    fat_mass_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Massa Gorda Total (kg) por DEXA — Tabela 3.1 (Eixo 3). "
            "Soma de depósitos subcutâneo + visceral + ectópico. "
            "Particionamento: fat_mass_kg + fat_free_mass_kg = total_weight_kg. "
            "Escala o RMR via equação de Cunningham (RMR = 500 + 22 × FFM_kg) "
            "e modifica a constante de condutância térmica corpo-casca "
            "(species_thermoregulation: 5-75 W/°C). "
            "Frequência: trimestral."
        ),
    )
    body_fat_percentage: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "% Massa Gorda = fat_mass_kg / total_weight_kg × 100 — Tabela 3.1 (Eixo 3). "
            "'% gordura elevada: maior carga inflamatória de base (adipocinas pró-inflamatórias), "
            "maior resistência à insulina, menor VO2max relativo, maior stress articular.' "
            "Adiposidade essencial mínima: ~3% homem, ~12% mulher. "
            "Performance óptima: 6-13% H, 14-20% M. "
            "Modifica: isolamento térmico (↑BF → ↑resistência térmica da casca), "
            "aromatização hormonal (↑BF → ↑conversão estrogénica), "
            "e coeficiente de sensibilidade à insulina aplicado sobre "
            "species_bioenergetics GLUT4 Km. "
            "Frequência: trimestral."
        ),
    )
    fat_free_mass_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Massa Livre de Gordura (kg) = lean_mass_kg + bone_mineral_content_kg. "
            "Multiplicador primário para: potencial máximo de entrega de O2, "
            "TBW estimada (TBW ≈ 0.732 × FFM), taxa de produção de creatinina "
            "(20 mg/kg FFM/dia), e RMR de Cunningham. "
            "Tecto da espécie: VO2max escala ~50-80 mL/min/kg FFM "
            "(species_bioenergetics)."
        ),
    )
    lean_mass_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Massa Magra Total (kg) — tecido mole magro, exclui osso — Tabela 3.1 (Eixo 3). "
            "'Define a capacidade metabólica total (mitocôndrias, enzimas glicolíticas, "
            "reservas de glicogénio).' "
            "Escala directamente: taxa de síntese proteica de corpo inteiro "
            "(species_epigenetic: 280 g/dia a ~70 kg massa magra), FSR muscular, "
            "produção de ureia, e força máxima via densidade de pontes cruzadas "
            "de miosina (species_neuromuscular)."
        ),
    )
    lean_mass_index_kg_per_m2: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Lean Mass Index (LMI) = lean_mass_kg / altura_m² — Tabela 3.1 (Eixo 3). "
            "Normaliza a massa magra total pela estatura. "
            "Equivalente ao BMI mas para a componente magra — complementa o IAMM "
            "(que considera apenas membros) com a massa magra do tronco e cabeça."
        ),
    )
    bone_mineral_content_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Conteúdo Mineral Ósseo total (kg) por DEXA — Tabela 3.1 (Eixo 3). "
            "Referência: adulto masculino ~2.8 kg, feminino ~2.2 kg. "
            "Comparado contra species_osseous_system: "
            "cálcio esquelético 1.0 kg (99% do total corporal), "
            "fósforo 0.6 kg (85% do total). "
            "BMC + lean_mass_kg = fat_free_mass_kg."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 3 — DENSIDADE MINERAL ÓSSEA
    # Fonte: Tabela 3.1 — DMO (T-score e Z-score — coluna, colo femoral, corpo total)
    # ═══════════════════════════════════════════════════════════════════════

    bmd_total_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "DMO corpo total (g/cm²) por DEXA — Tabela 3.1 (Eixo 3). "
            "Referência espécie (species_osseous_system): "
            "pico L1-L4: 1.0-1.4 g/cm², colo femoral: 0.9-1.2 g/cm². "
            "Usada para calcular T-score e escalar risco de fractura contra "
            "MESfx = 25.000 µε (lei de Wolff)."
        ),
    )
    bmd_spine_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "DMO coluna lombar L1-L4 (g/cm²) — Tabela 3.1 (Eixo 3). "
            "Referência espécie: pico 1.0-1.4 g/cm². "
            "Local primário de diagnóstico de osteoporose: "
            "T-score ≤ -2.5 = osteoporose. Frequência: anual (ou semestral em atletas de alto risco)."
        ),
    )
    bmd_femoral_neck_g_per_cm2: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "DMO colo femoral (g/cm²) — Tabela 3.1 (Eixo 3). "
            "Referência espécie: pico 0.9-1.2 g/cm². "
            "Local de fractura de risco. Utilizado com DMO da coluna para calcular "
            "score FRAX e condicionar cargas de impacto osteogénico "
            "(MESm = 1.250 µε threshold — species_osseous_system)."
        ),
    )
    t_score: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "T-score WHO = (DMO individual − DMO média adulto jovem pico) / DP — Tabela 3.1 (Eixo 3). "
            "'T-score < -1.0: osteopénia — limita a carga mecânica máxima aplicável "
            "sem risco de fractura de stress. T-score < -2.5: osteoporose — contra-indica "
            "cargas de impacto elevadas.' "
            "Em atletas jovens com T-score < -1.5 antes dos 30 anos: RED-S ou síndrome da tríade. "
            "Frequência: anual."
        ),
    )
    z_score: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Z-score = (DMO individual − DMO média de pares da mesma idade e sexo) / DP — Tabela 3.1. "
            "'Z-score (comparativo com pares de mesma idade) mais relevante em atletas jovens.' "
            "Z-score < -2.0 = abaixo do esperado para a idade — "
            "separa perda óssea por idade de perda patológica."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4 — MASSA MAGRA REGIONAL (por membro)
    # Fonte: Tabela 3.1 — Massa Magra Regional (por membro — DEXA)
    # ═══════════════════════════════════════════════════════════════════════

    lean_mass_arms_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Massa magra bilateral dos braços (kg) — Tabela 3.1 (Eixo 3). "
            "Componente do ASMM. Reflecte tecido contrátil disponível para força "
            "de pressão/puxada. Escala força máxima de preensão e output neuromuscular "
            "do membro superior."
        ),
    )
    lean_mass_legs_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Massa magra bilateral das pernas (kg) — Tabela 3.1 (Eixo 3). "
            "Maior componente do ASMM. Determinante primário da força de reacção ao solo, "
            "potência de sprint e Wmax em ciclismo. "
            "Escala força pico (Fmax ∝ PCSA ∝ massa magra dos membros inferiores — "
            "species_neuromuscular)."
        ),
    )
    lean_mass_trunk_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Massa magra do tronco (kg) — Tabela 3.1 (Eixo 3). "
            "Inclui erectores espinhais, parede abdominal e musculatura respiratória. "
            "Escala a resistência postural, capacidade de pressão intra-abdominal "
            "e contribuição muscular respiratória para o limiar ventilatório."
        ),
    )
    lean_mass_dominant_nondominant_ratio: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Ratio massa magra membro dominante / não-dominante — Tabela 3.1 (Eixo 3). "
            "'Assimetria > 10-15% entre membros: indica padrão de uso desequilibrado "
            "ou lesão antiga que compensou; preditor de risco de lesão futura; "
            "afecta eficiência biomecânica e transferência de potência.' "
            "Frequência: trimestral."
        ),
    )
    lean_mass_limb_asymmetry_percentage: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "% de assimetria bilateral de massa magra entre membros homólogos — Módulo D1. "
            "'Uma assimetria de > 10% entre membros homólogos correlaciona-se com risco "
            "aumentado de lesão e padrões compensatórios de movimento.' "
            "Calculado como: |membro_forte - membro_fraco| / membro_forte × 100. "
            "Limiar de risco: > 10% aceitável; > 15% = comprometimento."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 5 — MÚSCULO ESQUELÉTICO APENDICULAR (sarcopénia e índice de performance)
    # Fonte: Módulo D1 — IAMM; Tabela 3.1 — Massa Magra Apendicular
    # ═══════════════════════════════════════════════════════════════════════

    appendicular_lean_mass_kg: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "ASMM (kg) = lean_mass_arms_kg + lean_mass_legs_kg — Tabela 3.1 + Módulo D1. "
            "'Massa Magra Total e Apendicular (DEXA): Define a capacidade metabólica total.' "
            "Thresholds de sarcopénia EWGSOP2: H < 20 kg, M < 15 kg. "
            "Escala VO2max, output de potência funcional e classificação de risco de queda."
        ),
    )
    appendicular_lean_mass_index_kg_per_m2: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "IAMM (kg/m²) = ASMM_kg / altura_m² — Módulo D1: "
            "'Índice de Massa Magra Apendicular (IAMM) = Massa magra membros / Altura².' "
            "Target H: > 7.26 kg/m²; M: > 5.50 kg/m². "
            "'Abaixo destes valores: sarcopénia funcional diagnosticável.' "
            "Normaliza a massa muscular pela estatura — preditor de performance "
            "normalizado por altura superior ao ASMM absoluto. "
            "Frequência: trimestral."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 6 — GORDURA VISCERAL E DISTRIBUIÇÃO REGIONAL
    # Fonte: Tabela 3.1 — Massa Gorda Visceral (Android Fat Ratio ou VAT)
    # ═══════════════════════════════════════════════════════════════════════

    visceral_fat_android_percentage: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "% Gordura Visceral da região android — Tabela 3.1 (Eixo 3). "
            "'Massa Gorda Visceral (Android Fat Ratio ou VAT via DEXA/ressonância): "
            "% da gordura android total ou cm² (ressonância: > 100 cm² = elevado risco).' "
            "'Gordura visceral é metabolicamente activa: secreta IL-6, TNF-α, resistina → "
            "inflamação sistémica, resistência à insulina, supressão do eixo HPG; "
            "preditor independente de comprometimento da recuperação.'"
        ),
    )
    visceral_fat_area_cm2: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Área de gordura visceral (cm²) por DEXA ou TC — Tabela 3.1 (Eixo 3). "
            "Threshold de risco elevado: > 100 cm². "
            "Correlaciona com deposição ectópica de lípidos no fígado (esteatose hepática) "
            "e músculo (IMCL ↑), degradando a eficiência de oxidação mitocondrial de gordura "
            "(species_bioenergetics). "
            "Frequência: trimestral."
        ),
    )
    android_fat_percentage: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "% gordura da região android (abdominal/cintura) por DEXA — Módulo D1. "
            "Gordura android elevada → ↑IL-6 basal (species_immune_microbiome: basal 5 pg/mL), "
            "↑TNF-α, ↑PCR, ↓adiponectina. "
            "Threshold de risco metabólico: H > 25%, M > 33%."
        ),
    )
    gynoid_fat_percentage: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "% gordura da região gynoid (anca/coxa) por DEXA — Módulo D1. "
            "Depósito glúteo-femoral: fenótipo metabólico protector vs. obesidade android. "
            "Gordura gynoid elevada associada a melhor sensibilidade à insulina "
            "quando o depósito android é também baixo."
        ),
    )
    android_gynoid_ratio: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Ratio Android/Gynoid — Módulo D1: "
            "'Ratio Android/Gynoid: Ratio > 1.0 indica deposição preferencial de gordura "
            "na região abdominal visceral → maior IL-6 adiposa → maior resistência à insulina.' "
            "Índice de risco cardiovascular-metabólico. "
            "Escala o multiplicador de citocinas inflamatórias aplicado sobre "
            "species_immune_microbiome IL-6, TNF-α, e PCR basais."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 7 — ÁGUA CORPORAL TOTAL E COMPARTIMENTOS
    # Fonte: Tabela 3.1 — Percentagem de Água Corporal Total (TBW)
    # ═══════════════════════════════════════════════════════════════════════

    total_body_water_percentage: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "% Água Corporal Total da massa corporal — Tabela 3.1 (Eixo 3). "
            "'Percentagem de Água Corporal Total (TBW): % da massa total "
            "(homem jovem: 60-70%; mulher: 50-60%). "
            "Hidratação crónica: afecta viscosidade sanguínea, transporte de nutrientes, "
            "eficiência da bomba Na⁺/K⁺, termorregulação; "
            "TBW cronicamente baixa é sinal de défice de electrólitos ou dieta restritiva.' "
            "Frequência: trimestral (via impedância multi-frequência)."
        ),
    )
    total_body_water_liters: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Água Corporal Total (L) por bioimpedância ou diluição de deutério — Tabela 3.1. "
            "Estimativa: TBW ≈ 0.60 × peso_kg (H) ou 0.50 × peso_kg (M); "
            "mais precisamente TBW ≈ 0.732 × FFM_kg. "
            "Escala: volume plasmático (VP ≈ TBW × 0.21), "
            "reservatório de suor para termorregulação "
            "(species_thermoregulation: taxa máx de suor 3 L/h), "
            "e homeostase fluido-electrolítica (species_fluid_electrolyte: "
            "distribuição Na⁺/K⁺)."
        ),
    )
    intracellular_water_liters: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Água Intracelular (L) por BIA multi-frequência. "
            "Referência: ~60% da TBW. ICW correlaciona com massa celular magra e "
            "reflecte o estado de hidratação celular. "
            "Ratio ICW/TBW < 0.55: desidratação celular ou catabolismo — "
            "modifica o multiplicador de síntese proteica de species_epigenetic."
        ),
    )
    extracellular_water_liters: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Água Extracelular (L) por BIA multi-frequência. "
            "Referência: ~40% da TBW. Inclui água intersticial + plasmática. "
            "Ratio ECW/TBW > 0.40: edema, inflamação ou overreaching — "
            "aplica penalização inflamatória em species_immune_microbiome "
            "e nos cálculos de pré-carga cardiovascular (Frank-Starling)."
        ),
    )
    ecw_tbw_ratio: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Ratio ECW / TBW. Biomarcador de hidratação e inflamação. "
            "Intervalo saudável: 0.36-0.39. Valores > 0.40 = edema/inflamação. "
            "Valores < 0.35 = desidratação intracelular. "
            "Condiciona os algoritmos de prescrição de fluidos e a "
            "classificação do estado inflamatório."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 8 — RATIO CINTURA/ALTURA E DISTRIBUIÇÃO ADIPOSA CENTRAL
    # Fonte: Tabela 3.1 — Ratio Cintura/Altura
    # ═══════════════════════════════════════════════════════════════════════

    waist_height_ratio: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Ratio Cintura/Altura — Tabela 3.1 (Eixo 3). "
            "'< 0.50 = baixo risco; > 0.60 = elevado risco metabólico.' "
            "'Indicador simples de adiposidade central; proxy de gordura visceral '  "
            "'sem necessidade de DEXA; fortemente correlacionado com inflamação '  "
            "'sistémica e resistência à insulina.' "
            "Frequência: mensal."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 9 — BIOIMPEDÂNCIA — INTEGRIDADE CELULAR
    # ═══════════════════════════════════════════════════════════════════════

    phase_angle_degrees: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Ângulo de fase BIA a 50 kHz = arctan(Xc / R). "
            "Reflecte integridade da membrana celular e estado nutricional. "
            "Referência adulto saudável: 5-7°. Atleta: 7-9°. Crítico: < 4°. "
            "Ângulo de fase < 5°: reduz multiplicador de FSR (species_epigenetic) "
            "e sinaliza estado catabólico ao motor prescritivo."
        ),
    )
    reactance_ohm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Reactância BIA (Ω) a 50 kHz. Reflecte propriedades capacitivas "
            "das membranas celulares. Input para o cálculo do ângulo de fase. "
            "Referência adulto masculino: ~60-75 Ω."
        ),
    )
    resistance_ohm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Resistência BIA (Ω) a 50 kHz. Reflecte condutância de água/electrólitos. "
            "Input para ângulo de fase e estimativa de TBW. "
            "Referência adulto masculino: ~450-550 Ω."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 10 — ANTROPOMETRIA
    # ═══════════════════════════════════════════════════════════════════════

    waist_circumference_cm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Perímetro da cintura (cm) ao nível do umbigo. "
            "Obesidade abdominal WHO: H ≥ 102 cm, M ≥ 88 cm. "
            "IDF: H ≥ 94 cm, M ≥ 80 cm. "
            "Numerador do waist_height_ratio. Proxy de VAT (r ≈ 0.80 com visceral_fat_area_cm2). "
            "Frequência: mensal."
        ),
    )
    hip_circumference_cm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Perímetro da anca (cm) no ponto mais largo sobre as nádegas. "
            "Denominador do waist_hip_ratio. Frequência: mensal."
        ),
    )
    waist_hip_ratio: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Ratio cintura/anca. Risco cardiovascular: H ≥ 0.90, M ≥ 0.85. "
            "Proxy secundário de gordura visceral quando DEXA não está disponível. "
            "Frequência: mensal."
        ),
    )
    neck_circumference_cm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Perímetro do pescoço (cm). Rastreio de SAOS: H > 40 cm, M > 35 cm. "
            "Risco de apneia do sono → afecta arquitectura de sono "
            "(species_chronobiology: %SWS, %REM) e performance cognitiva."
        ),
    )
    mid_arm_circumference_cm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Perímetro médio do braço (cm). Proxy de massa magra do membro superior. "
            "Limiar de desnutrição: < 23 cm. Atleta masculino de elite: 35-40 cm. "
            "Combinado com prega tricipital dá área muscular do braço (cm²)."
        ),
    )
    thigh_circumference_cm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Perímetro médio da coxa (cm). Proxy do volume quadricípite + isquiotibiais. "
            "Correlaciona com lean_mass_legs_kg (r ≈ 0.85). "
            "Monitoriza resposta hipertrófica ao treino de força."
        ),
    )
    calf_circumference_cm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Perímetro máximo da barriga da perna (cm). "
            "Proxy de sarcopénia: < 31 cm = risco sarcopénico (EWGSOP). "
            "Escala força do gastrocnémio/sóleo e carga do tendão de Aquiles."
        ),
    )
    wrist_circumference_cm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Perímetro do pulso (cm) para classificação de tamanho de esqueleto: "
            "pequeno < 15 cm, médio 15-17 cm, grande > 17 cm (H). "
            "Modifica o tecto de massa magra potencial no motor prescritivo."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 11 — PREGAS CUTÂNEAS (Durnin-Womersley / Jackson-Pollock)
    # ═══════════════════════════════════════════════════════════════════════

    skinfold_triceps_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Prega cutânea tricipital (mm). Face posterior do braço, ponto médio. "
            "Componente das equações de 4 locais (Durnin-Womersley) e 7 locais (Jackson-Pollock)."
        ),
    )
    skinfold_biceps_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Prega cutânea bicipital (mm). Face anterior do braço, ponto médio. "
            "Componente da equação de Durnin-Womersley de 4 locais."
        ),
    )
    skinfold_subscapular_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Prega subescapular (mm). Abaixo do ângulo inferior da omoplata. "
            "Marcador de gordura subcutânea do tronco. "
            "Componente das equações de 4 e 7 locais."
        ),
    )
    skinfold_suprailiac_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Prega suprailíaca (mm). Acima da crista ilíaca, linha axilar anterior. "
            "Componente das equações de 4 e 7 locais."
        ),
    )
    skinfold_abdominal_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Prega abdominal (mm). 2 cm lateral ao umbigo. "
            "Componente da equação de Jackson-Pollock de 7 locais."
        ),
    )
    skinfold_thigh_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Prega anterior da coxa (mm). Ponto médio da coxa anterior. "
            "Componente de Jackson-Pollock 3 locais (H) e 7 locais."
        ),
    )
    skinfold_chest_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Prega peitoral (mm). Dobra diagonal entre linha axilar anterior e mamilo (H). "
            "Componente de Jackson-Pollock 3 e 7 locais."
        ),
    )
    skinfold_medial_calf_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Prega medial da barriga da perna (mm). "
            "Usada em equações de 8 locais e perfil completo ISAK."
        ),
    )
    sum_of_skinfolds_mm: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "Soma de todas as pregas cutâneas medidas (mm). "
            "Input para equação de Durnin-Womersley: D = C − M × log(ΣPF). "
            "Combinada com equação de Siri (BF% = (4.95/D − 4.50) × 100) "
            "para estimar % gordura como validação cruzada do DEXA."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 12 — TAXA METABÓLICA DE REPOUSO (medida)
    # ═══════════════════════════════════════════════════════════════════════

    resting_metabolic_rate_kcal_per_day: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "RMR medida (kcal/dia) por calorimetria indirecta (gold standard) ou "
            "equação de Cunningham: RMR = 500 + 22 × FFM_kg. "
            "Referência espécie: ~1.700 kcal/dia a 70 kg de massa magra. "
            "RMR individual calibra a taxa de turnover de ATP (species_bioenergetics) "
            "e conduz o cálculo do gasto energético total (TEE = RMR × PAL)."
        ),
    )
    respiratory_quotient_rest: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment=(
            "QR de repouso = VCO2 / VO2. Intervalo: 0.70 (oxidação pura de gordura) "
            "a 1.00 (oxidação pura de HC). Derivado de calorimetria indirecta. "
            "Calibra o ponto de cruzamento de substrato em species_bioenergetics "
            "e a taxa de β-oxidação em repouso (species_lipid_metabolism)."
        ),
    )

    # ── Relação ──────────────────────────────────────────────────────────────
    athlete: Mapped["AthleteCore"] = relationship(back_populates="dexa_assessments")
