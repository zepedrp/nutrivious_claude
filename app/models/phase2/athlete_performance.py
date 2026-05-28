from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Float, String, func
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase2.athlete_core import AthleteCore


class AthletePerformance(Base):
    """
    Avaliação de performance funcional do atleta — Módulos D2 + D3.

    Relação 1:many com AthleteCore: avaliações periódicas (pré-época,
    mid-season, pós-época). Combina o perfil cardiorrespiratório (VO2max,
    limiares metabólicos) com a avaliação biomecânica (potência, força,
    mobilidade). Cada variável é expressável como CPI = valor_atleta /
    tecto_espécie (species_cardiovascular, species_neuromuscular).
    """

    __tablename__ = "athlete_performance"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    athlete_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("athlete_core.id", ondelete="CASCADE"),
        nullable=False,
    )

    assessment_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        comment=(
            "Data da avaliação de performance. "
            "Protocolo: 72h de recuperação após última sessão de alta intensidade, "
            "refeição padronizada 3h antes, sem cafeína 12h antes. "
            "Sequência recomendada: biomecânica (manhã) → cardiorrespiratório (tarde) "
            "com 4h de intervalo para evitar fadiga residual."
        ),
    )

    assessment_context: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment=(
            "'pre_season' | 'mid_season' | 'post_season' | 'return_to_sport' | 'ad_hoc'. "
            "Contextualiza o resultado na periodização anual."
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MÓDULO D3 — PERFIL CARDIORRESPIRATÓRIO
    # Interage com: species_cardiovascular, species_pulmonary,
    #               species_bioenergetics, species_mitochondrial
    # ══════════════════════════════════════════════════════════════════════════

    vo2max_ml_per_kg_min: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "VO2max — Consumo máximo de oxigénio relativo ao peso corporal. "
            "Unidade: mL/kg/min. "
            "Referência elite H endurance: 75-90 mL/kg/min; F: 65-80 mL/kg/min. "
            "Recordes absolutos: H ~97 mL/kg/min (Oskar Svendsen); F ~78 mL/kg/min. "
            "CPI_VO2max = vo2max_atleta / species_cardiovascular.vo2max_ceiling. "
            "VO2max = Débito Cardíaco × diferença arteriovenosa de O₂ "
            "(Fick: VO2 = DC × (CaO₂ − CvO₂)). "
            "Limitar superiormente pelo: (1) transporte de O₂ (Hb × SaO₂ × DC); "
            "(2) difusão alveolar (DLCO); (3) densidade capilar muscular (VEGF genótipo); "
            "(4) densidade mitocondrial (PPARGC1A genótipo). "
            "Modifica species_cardiovascular.vo2max_ceiling como denominador do CPI."
        ),
    )

    vo2max_absolute_l_per_min: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "VO2max absoluto. Unidade: L/min. "
            "Relevante para desportos com peso fixo (remo, ciclismo de pista, natação). "
            "VO2max_absoluto = vo2max_ml_per_kg_min × peso_kg / 1000. "
            "Elite remo H: 6.5-7.5 L/min."
        ),
    )

    ventilatory_threshold_1_pct_vo2max: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "VT1 — Primeiro Limiar Ventilatório (Aerobic Threshold / LT1). "
            "Unidade: % do VO2max. "
            "O VT1 corresponde ao início do aumento não-linear do VE/VCO₂ e à "
            "deflexão do RER — ponto onde a lactato sanguíneo começa a acumular "
            "acima da baseline (tipicamente 1.5-2.0 mmol/L). "
            "Atleta de endurance elite: VT1 a 70-80% VO2max. "
            "Zona 1/2 de treino: intensidades abaixo do VT1 (oxidação lipídica máxima). "
            "Interage com species_bioenergetics.aerobic_threshold_reference e "
            "species_lipid_metabolism.fat_oxidation_peak_intensity."
        ),
    )

    lactate_threshold_1_mmol: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "LT1 — Concentração de lactato no primeiro limiar. Unidade: mmol/L. "
            "Referência: ~1.5-2.0 mmol/L (ponto de deflexão da curva lactato-intensidade). "
            "Atletas de ultra-endurance: LT1 pode ser tão baixo como 1.0-1.2 mmol/L "
            "por maior capacidade de clearance e oxidação de lactato. "
            "Interacção com Veillonella spp. (microbioma): maior abundância → maior "
            "conversão de lactato em propionato → clearance mais eficiente "
            "(species_bioenergetics.lactate_clearance_rate)."
        ),
    )

    lactate_threshold_2_mmol: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "LT2/MLSS — Segundo Limiar de Lactato / Máximo Estado Estável de Lactato. "
            "Unidade: mmol/L. "
            "Referência: 3.5-5.5 mmol/L (convencionalmente 4.0 mmol/L — OnForm/Mader). "
            "LT2 = limiar anaeróbio individual; potência/pace acima do MLSS não é "
            "sustentável por > 30-60 min (lactato acumula progressivamente). "
            "Zona 4/5 de treino: intervalo entre LT2 e VO2max. "
            "species_bioenergetics.lactate_threshold_2_reference."
        ),
    )

    ventilatory_threshold_2_pct_vo2max: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "VT2 — Segundo Limiar Ventilatório (Respiratory Compensation Point). "
            "Unidade: % do VO2max. "
            "O VT2 corresponde ao ponto onde o VE aumenta mais steeply que o VCO₂ "
            "(hiperventilação de compensação da acidose metabólica). "
            "Atleta elite endurance: VT2 a 85-92% VO2max. "
            "Distância VT1-VT2 (zona aeróbia-anaeróbia) → espaço para treino tempo. "
            "species_cardiovascular.ventilatory_threshold_2_reference."
        ),
    )

    fatmax_pct_vo2max: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "FATmax — Intensidade de máxima oxidação lipídica. Unidade: % do VO2max. "
            "Calculado a partir do teste de oxidação de substratos (RER → VCO₂/VO₂). "
            "FATmax = intensidade onde (1.695 × VO₂ − 1.701 × VCO₂) é máximo. "
            "Referência: 45-65% VO2max em atletas de endurance. "
            "FATmax abaixo de 45% → baixa flexibilidade metabólica → dependência "
            "precoce de glicose → 'hitting the wall' em ultra. "
            "Interacção com FADS1 genótipo (conversão ALA→DHA) e omega_3_index "
            "(fluidez de membrana mitocondrial → eficiência de beta-oxidação). "
            "species_lipid_metabolism.fat_oxidation_max_rate e "
            "species_mitochondrial.beta_oxidation_capacity."
        ),
    )

    max_fat_oxidation_g_per_min: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "MFO — Máxima Oxidação de Gordura (g/min). "
            "Referência atletas endurance: 0.5-1.5 g/min. Elite 'fat-adapted': >1.2 g/min. "
            "Calculado: MFO = 1.695 × VO₂(FATmax) − 1.701 × VCO₂(FATmax) [g/min]. "
            "MFO > 1.2 g/min → 'metabolic efficiency' superior → menor depleção de "
            "glicogénio nas primeiras horas de esforço → vantagem em provas >3h."
        ),
    )

    max_heart_rate_bpm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "FC máxima medida em esforço máximo incremental. Unidade: bpm. "
            "NUNCA usar fórmulas (220−idade; Tanaka); sempre medida. "
            "Referência: altamente individual; decresce ~1 bpm/ano após os 25. "
            "FC máxima é o tecto do débito cardíaco por frequência: "
            "DC_max = FCmax × VS_max. "
            "species_cardiovascular.max_heart_rate_reference."
        ),
    )

    heart_rate_recovery_1min_bpm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Recuperação da FC ao 1º minuto pós-esforço máximo (HRR1). Unidade: bpm de queda. "
            "Referência: > 12 bpm (atleta saudável); elite endurance: 25-40 bpm/min. "
            "HRR1 < 12 bpm → risco cardiovascular aumentado (Nishime et al., NEJM 1999). "
            "HRR1 reflecte o tónus vagal e a reactivação parassimpática pós-esforço — "
            "correlaciona-se com HRV RMSSD em repouso "
            "(species_cardiovascular.parasympathetic_recovery_rate)."
        ),
    )

    respiratory_exchange_ratio_max: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "RER máximo no pico do teste incremental. Dimensionless (VCO₂/VO₂). "
            "RER ≥ 1.10 é critério de esforço máximo atingido. "
            "RER < 1.05 no pico → esforço submáximo → VO2max pode estar subestimado. "
            "Verifica que o protocolo foi realmente máximo antes de registar vo2max."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MÓDULO D2 — AVALIAÇÃO BIOMECÂNICA E FUNCIONAL
    # Interage com: species_neuromuscular, species_musculoskeletal
    # ══════════════════════════════════════════════════════════════════════════

    cmj_height_cm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "CMJ — Counter Movement Jump height. Unidade: cm. "
            "Medição: plataforma de força (gold-standard) ou tapete de contacto / "
            "acelerómetro (Myotest, PUSH Band). "
            "Referência H atletas desporto colectivo: 35-55 cm; F: 28-42 cm. "
            "CMJ mede a potência de pico dos extensores do membro inferior "
            "(cadeia posterior: glúteos, isquiotibiais, gémeos) em regime "
            "pliométrico (ciclo alongamento-encurtamento, SSC). "
            "CPI_CMJ = cmj_height / species_neuromuscular.ssc_jump_height_ceiling. "
            "Monitorização: queda de CMJ > 10% vs. baseline → fadiga neuromuscular "
            "significativa → reduzir carga de treino."
        ),
    )

    cmj_rsi_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "RSI — Reactive Strength Index (CMJ). Dimensionless. "
            "RSI = Altura de salto (m) / Tempo de contacto (s). "
            "Referência elite: RSI > 2.5; força explosiva alto nível: >3.0. "
            "O RSI mede a capacidade de produção rápida de força no SSC rápido "
            "(tempo de contacto < 250ms) — qualidade de mola do tendão e reflexo miotático. "
            "RSI < 1.5 → défice de stiffness tendinosa e/ou co-activação reflexa. "
            "Interage com COL5A1 genótipo (rigidez tendinosa genética) "
            "e tendon_stiffness_n_per_mm medida por ultrassom "
            "(species_musculoskeletal.tendon_stiffness_ceiling)."
        ),
    )

    imtp_peak_force_n: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "IMTP — Isometric Mid-Thigh Pull peak force. Unidade: N. "
            "Protocolo: plataforma de força, ângulo joelho 125-145°, 3s de pull máximo. "
            "Referência H atletas força/colectivo: 2000-3500 N. "
            "IMTP mede a força isométrica máxima do sistema triplo de extensão "
            "(tornozelo-joelho-anca); correlaciona-se com pico de produção de força "
            "em movimentos dinâmicos (sprint, mudança de direcção). "
            "IMTP/BW (força relativa ao peso) ≥ 3.0 → excelente para desportos de "
            "potência (species_neuromuscular.isometric_peak_force_ceiling). "
            "Assimetria IMTP esquerdo/direito > 10% → risco de lesão do membro "
            "inferior dominante."
        ),
    )

    handgrip_dominant_kg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Força de preensão manual — mão dominante. Unidade: kg (dinamómetro Jamar). "
            "Protocolo: 3 tentativas com 30s de intervalo; registar o maior valor. "
            "Referência H 20-30 anos: 50-70 kg; F: 30-45 kg. "
            "A força de preensão é um marcador proxy de força total e preditor "
            "independente de mortalidade all-cause (Leong et al., Lancet 2015: "
            "cada 5 kg de redução → +16% mortalidade cardiovascular). "
            "CPI_grip = handgrip_dominant / species_neuromuscular.handgrip_force_ceiling."
        ),
    )

    handgrip_nondominant_kg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Força de preensão manual — mão não-dominante. Unidade: kg. "
            "Assimetria normal: 5-10% (dominante > não-dominante). "
            "Assimetria > 15% → disfunção neuromuscular unilateral ou lesão. "
            "Rácio dominante/não-dominante: registar como indicador de lateralidade funcional."
        ),
    )

    ankle_dorsiflexion_dominant_deg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Amplitude de dorsiflexão do tornozelo — membro dominante. Unidade: graus. "
            "Teste: Weight-bearing lunge test (WBLT) — distância pé-parede em cm "
            "convertida em graus (tan⁻¹(d/h)). "
            "Referência funcional: ≥ 35-38° (ou ≥ 10 cm no WBLT). "
            "Dorsiflexão < 30° → compensação biomecânica: valgo dinâmico do joelho, "
            "aumento de carga patelar e Aquiliana, risco de fascite plantar. "
            "Interage com COL5A1 e TNXB genótipos (laxidez articular). "
            "species_musculoskeletal.ankle_mobility_reference."
        ),
    )

    ankle_dorsiflexion_nondominant_deg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Amplitude de dorsiflexão — membro não-dominante. Unidade: graus. "
            "Assimetria > 5° entre membros → investigar restrição unilateral "
            "(encurtamento do sóleo/gémeo, aderências cicatriciais de lesões anteriores). "
            "Assimetria bilateral correlaciona-se com padrão de lesão de tornozelo recorrente."
        ),
    )

    thoracic_rotation_dominant_deg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Rotação torácica — direcção dominante. Unidade: graus. "
            "Teste: seated thoracic rotation com pelvis fixada. "
            "Referência funcional: ≥ 50° bilateral. "
            "Rotação torácica restrita → compensação lombar → risco de lombalgia "
            "em desportos de rotação (golfe, ténis, basebol, remo). "
            "species_musculoskeletal.spinal_mobility_reference."
        ),
    )

    thoracic_rotation_nondominant_deg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Rotação torácica — direcção não-dominante. Unidade: graus. "
            "Assimetria > 10° → investigar disfunção costovertebral unilateral ou "
            "padrão de sobrecarga rotacional crónica (overuse de desportos assimétricos)."
        ),
    )

    tendon_cross_sectional_area_mm2: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "CSA do tendão-alvo (patelar ou Aquiles, conforme protocolo). Unidade: mm². "
            "Medição: ultrassom B-mode em corte transversal a 2-3 cm da inserção. "
            "Referência tendão patelar H: 50-100 mm²; Aquiles H: 45-80 mm². "
            "CSA aumentada (>120 mm² patelar) → tendinopatia crónica com espessamento. "
            "CSA reduzida para o nível de treino → hipotrofia tendinosa → risco de rotura. "
            "species_musculoskeletal.tendon_morphology_reference."
        ),
    )

    tendon_stiffness_n_per_mm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Rigidez tendinosa (stiffness). Unidade: N/mm. "
            "Medição: ultrassom sonotransdutor + dinamómetro isocinético (método de "
            "alongamento controlado com imaging simultâneo de junção miotendínosa). "
            "Referência Aquiles H treinado: 100-250 N/mm. "
            "Stiffness elevada → transmissão de força mais eficiente → RSI mais alto. "
            "Stiffness baixa → maior armazenamento de energia elástica mas menor "
            "eficiência de retorno → 'energia desperdiçada' no SSC. "
            "Interage com COL5A1 genótipo (architecture intrínseca das fibrilas): "
            "TT homozigoto → maior stiffness basal "
            "(species_musculoskeletal.tendon_stiffness_ceiling)."
        ),
    )

    # ── Relações ─────────────────────────────────────────────────────────────
    athlete: Mapped["AthleteCore"] = relationship(
        back_populates="performance_assessments"
    )
