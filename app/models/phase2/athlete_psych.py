from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Float, Integer, String, func
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase2.athlete_core import AthleteCore


class AthletePsych(Base):
    """
    Perfil psicológico, cronobiológico e de bem-estar do atleta — Eixo 9 / Módulo F.

    Relação 1:many com AthleteCore: reavaliação periódica (mensal ou por bloco
    de treino). Integra questionários validados de stress, recuperação, humor
    e cronotipo com o baseline de HRV e arquitectura do sono.
    Interage com: species_chronobiology, species_neural_cognitive, species_endocrine.
    """

    __tablename__ = "athlete_psych"

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
            "Data da avaliação psicológica. "
            "Protocolo: administrar RESTQ-Sport e POMS na mesma semana, "
            "preferencialmente ao mesmo dia da semana para controlar variabilidade "
            "intra-semanal. MCTQ: completar com dados de 4 semanas de sono. "
            "HRV: média de 7 dias consecutivos de medição matinal em repouso supino."
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # CRONÓTIPO E RITMO CIRCADIANO
    # Interage com: species_chronobiology
    # ══════════════════════════════════════════════════════════════════════════

    mctq_msf_sc: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "MCTQ MSFsc — Munich Chronotype Questionnaire, Sleep-corrected Midpoint "
            "of Sleep on Free Days. Unidade: horas decimais (ex: 3.5 = 03:30). "
            "Desenvolvido por Roenneberg et al.; é o gold-standard de avaliação "
            "de cronotipo sem viés de comportamento social. "
            "MSFsc < 2.0h → cronotipo extremamente matutino ('cegonha extrema'). "
            "MSFsc 2.0-4.0h → cronotipo intermédio (maioria da população). "
            "MSFsc > 5.0h → cronotipo vespertino ('coruja'); correlaciona-se com "
            "CLOCK rs1801260 CC e PER2 variantes. "
            "species_chronobiology.chronotype_phase_reference: "
            "pico de performance ≈ MSFsc + 7-8h (máximo de temperatura corporal). "
            "Informa horário óptimo de competição, treino e suplementação de cafeína."
        ),
    )

    social_jetlag_hours: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Social Jetlag (SJL). Unidade: horas. "
            "Fórmula: SJL = |MSF − MSW| onde MSF = midpoint sleep dias livres; "
            "MSW = midpoint sleep dias de treino/trabalho. "
            "SJL representa o 'jetlag interno' causado pelo conflito entre o relógio "
            "biológico e os horários sociais/de treino impostos. "
            "SJL > 2h → disrupção circadiana significativa → elevação de cortisol "
            "nocturno, supressão de melatonina, comprometimento da fase SWS do sono "
            "(species_chronobiology.social_jetlag_threshold). "
            "SJL > 2h associado a aumento de 33% do risco de obesidade e 1.5× "
            "risco de síndrome metabólica (Roenneberg et al.). "
            "Objectivo: SJL < 1h através de sincronização de horários de treino "
            "com o cronotipo natural."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # STRESS PERCEBIDO
    # Interage com: species_endocrine (eixo HPA), species_neural_cognitive
    # ══════════════════════════════════════════════════════════════════════════

    pss_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment=(
            "PSS — Perceived Stress Scale (Cohen et al., 1983). Escala: 0-40. "
            "10 itens, Likert 0-4; itens positivos invertidos. "
            "Classificação: 0-13 = stress baixo; 14-26 = stress moderado; "
            "27-40 = stress elevado. "
            "PSS > 26 → activação crónica do eixo HPA → cortisol_am elevado → "
            "catabolismo muscular, supressão imunológica, disrupção do sono "
            "(species_endocrine.hpa_activation_stress_threshold). "
            "PSS correlaciona-se com HRV RMSSD invertidamente: "
            "PSS 27+ → RMSSD tipicamente < 40ms em atletas. "
            "Monitorizar conjuntamente com RESTQ-Sport para distinção entre "
            "stress de treino (recuperável) e stress psicossocial (multifactorial)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RECUPERAÇÃO E STRESS DE TREINO
    # Interage com: species_immune_microbiome, species_endocrine,
    #               species_neuromuscular
    # ══════════════════════════════════════════════════════════════════════════

    restq_sport_stress_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "RESTQ-Sport — Recovery-Stress Questionnaire for Athletes (Kellmann & Kallus). "
            "Sub-escala de Stress total (média das 7 escalas de stress). "
            "Escala Likert 0-6 por item; maior = mais frequente. "
            "Stress total < 2.0 → carga de treino bem tolerada. "
            "Stress total > 3.5 → sobretreinamento iminente → protocolo de tapering. "
            "species_immune_microbiome.training_stress_threshold."
        ),
    )

    restq_sport_recovery_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "RESTQ-Sport — Sub-escala de Recuperação total (média das 5 escalas de recuperação). "
            "Recuperação > 4.0 → recuperação adequada para carga progressiva. "
            "Recuperação < 2.5 → défice de recuperação → reduzir volume 20-30%, "
            "aumentar sono, protocolo nutricional de recuperação. "
            "Rácio Recuperação/Stress > 1.2 → estado funcional positivo. "
            "Rácio < 0.8 → overreaching não-funcional (species_endocrine.recovery_deficit_threshold)."
        ),
    )

    poms_tmd_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "POMS — Profile of Mood States, Total Mood Disturbance (TMD). "
            "Fórmula: TMD = (Tensão + Depressão + Hostilidade + Fadiga + Confusão) − Vigor. "
            "Escala normalizada: TMD negativo → 'iceberg profile' (Vigor elevado, "
            "outros estados baixos) — perfil clássico de atleta bem-treinado. "
            "TMD positivo → desequilíbrio emocional → risco de overtraining síndrome. "
            "TMD > +20 → indicador de overtreino severo (Morgan et al., 1987). "
            "Monitorização semanal rápida (6 escalas, 65 itens ou versão curta 30 itens). "
            "species_neural_cognitive.mood_state_performance_threshold."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SAÚDE MENTAL — DASS-21
    # Interage com: species_neural_cognitive
    # ══════════════════════════════════════════════════════════════════════════

    dass21_depression_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment=(
            "DASS-21 — Depression Anxiety Stress Scales, sub-escala Depressão. "
            "Escala: 0-42 (7 itens × máx 6, × 2 para normalizar ao DASS-42). "
            "Classificação: 0-9 normal; 10-13 leve; 14-20 moderada; 21-27 severa; >28 extrema. "
            "Depressão no contexto desportivo: monitorizar em pós-lesão grave, "
            "desempenhamento prolongado, ou atletas com BDNF Val66Met AA "
            "(menor neuroplasticidade induzida pelo exercício). "
            "Score > 13 → referenciação para psicólogo do desporto "
            "(species_neural_cognitive.depression_risk_threshold)."
        ),
    )

    dass21_anxiety_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment=(
            "DASS-21 — Sub-escala Ansiedade. Escala: 0-42. "
            "Classificação: 0-7 normal; 8-9 leve; 10-14 moderada; 15-19 severa; >20 extrema. "
            "Ansiedade de pré-competição e ansiedade social de performance são frequentes "
            "em atletas — distinguir ansiedade facilitadora (leve, activadora) de "
            "ansiedade debilitante (moderada-severa). "
            "SLC6A4 SS homozigoto → maior hiperreactividade a stressores → "
            "scores de ansiedade mais elevados (species_neural_cognitive.anxiety_serotonin_modifier)."
        ),
    )

    dass21_stress_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment=(
            "DASS-21 — Sub-escala Stress (tensão e agitação). Escala: 0-42. "
            "Classificação: 0-14 normal; 15-18 leve; 19-25 moderada; 26-33 severa; >34 extrema. "
            "A sub-escala stress do DASS-21 é mais sensível a stress agudo que o PSS "
            "(que mede stress percebido crónico). "
            "Stress DASS21 > 18 persistente → activação HPA crónica → perfil de "
            "cortisol aplanado, DHEA-S baixo (species_endocrine.hpa_chronic_stress_axis)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PERSONALIDADE — HEXACO
    # Interage com: species_neural_cognitive
    # ══════════════════════════════════════════════════════════════════════════

    hexaco_honesty_humility: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HEXACO — Dimensão Honesty-Humility. Escala: 1.0-5.0 (média dos itens). "
            "Avalia sinceridade, equanimidade, não-gananciosidade e modéstia. "
            "Alto H-H → menor propensão para comportamentos anti-desportivos e doping. "
            "Relevante para ética desportiva e adesão a protocolos "
            "(species_neural_cognitive.integrity_behavioral_trait)."
        ),
    )

    hexaco_emotionality: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HEXACO — Dimensão Emotionality. Escala: 1.0-5.0. "
            "Avalia medo, ansiedade, dependência e sentimentalismo. "
            "Alto E → maior sensibilidade emocional → perfil de ansiedade de "
            "pré-competição elevada. Interacção com SLC6A4 SS e COMT Met/Met. "
            "Em lesão: alto E → pior regulação emocional no processo de reabilitação."
        ),
    )

    hexaco_extraversion: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HEXACO — Dimensão Extraversion. Escala: 1.0-5.0. "
            "Avalia auto-estima social, atrevimento, sociabilidade e animação. "
            "Alto X → maior motivação extrínseca (audiência, equipa) → "
            "vantagem em desportos colectivos e de alto espectáculo. "
            "DRD2 A2A2 → maior recompensa intrínseca → pode compensar baixo X "
            "em desportos individuais de resistência."
        ),
    )

    hexaco_agreeableness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HEXACO — Dimensão Agreeableness. Escala: 1.0-5.0. "
            "Avalia perdão, gentileza, flexibilidade e paciência. "
            "Baixo A → maior agressividade competitiva → vantagem em desportos de "
            "contacto; maior risco de conflito com equipa técnica. "
            "Monitorizar em atletas de alto rendimento com COMT Val/Val (baixa "
            "ruminação, alta tolerância ao stress agudo) e baixo A "
            "(species_neural_cognitive.competitive_aggressiveness_trait)."
        ),
    )

    hexaco_conscientiousness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HEXACO — Dimensão Conscientiousness. Escala: 1.0-5.0. "
            "Avalia organização, diligência, perfecionismo e prudência. "
            "Alto C → maior adesão a protocolos de treino, nutrição e recuperação → "
            "é o preditor de personalidade mais forte de performance de longo-prazo. "
            "DRD2 A2A2 (maior recompensa com progressão gradual) + alto C → "
            "perfil ideal para periodização estruturada de longo-prazo. "
            "species_neural_cognitive.adherence_personality_trait."
        ),
    )

    hexaco_openness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HEXACO — Dimensão Openness to Experience. Escala: 1.0-5.0. "
            "Avalia curiosidade estética, criatividade e inquisitividade. "
            "Alto O → maior abertura a novos métodos de treino, biohacking, "
            "tecnologias de monitorização. "
            "Relevante para adesão a intervenções nutricionais e de estilo de vida "
            "não-convencionais (crioterapia, jejum intermitente, altitude simulada)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # HRV E SONO — BASELINE WEARABLE CRÓNICO
    # Interage com: species_cardiovascular, species_chronobiology,
    #               species_neural_cognitive
    # ══════════════════════════════════════════════════════════════════════════

    hrv_rmssd_baseline_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HRV RMSSD basal crónico (média de 7 dias em repouso supino, matinal). "
            "Unidade: ms. "
            "RMSSD = Root Mean Square of Successive Differences dos intervalos RR — "
            "reflecte o tónus parassimpático (nervo vago → nodo SA). "
            "Referência atleta elite: 70-120 ms; atleta recreativo: 40-80 ms; "
            "sedentário saudável: 25-50 ms. "
            "RMSSD basal é um dos mais robustos marcadores de estado de recuperação: "
            "queda >15% vs. baseline de 7 dias → reduzir carga de treino. "
            "Correlaciona-se com: tónus vagal (species_cardiovascular.vagal_tone_index), "
            "cortisol_am (HRV ↑ ↔ cortisol ↓), e PER3 VNTR (5/5 → RMSSD mais baixo "
            "após privação de sono)."
        ),
    )

    resting_heart_rate_bpm: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "FC em repouso basal crónica (média de 7 dias, medição matinal supino). "
            "Unidade: bpm. "
            "Referência atleta elite endurance: 35-50 bpm (bradicardia de treino = "
            "adaptação ao volume de treino aeróbio: aumento do VS, regulação vagal). "
            "FC repouso cronicamente > 60 bpm em atleta de endurance de alto volume → "
            "sinal de overtraining, infecção subclínica ou défice de sono. "
            "Aumento súbito > 5 bpm vs. baseline → intervenção de recuperação. "
            "species_cardiovascular.resting_heart_rate_reference."
        ),
    )

    sleep_efficiency_percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Eficiência do sono basal crónica (wearable — Oura, Garmin, Whoop). "
            "Unidade: %. "
            "Fórmula: (Tempo total de sono / Tempo na cama) × 100. "
            "Referência: ≥ 85% (PSG gold-standard); wearable ≥ 80% como proxy. "
            "Eficiência < 80% → fragmentação do sono → redução de SWS e REM → "
            "comprometimento da consolidação de memória motora, secreção de GH "
            "(70% GH secretada em SWS pulsátil) e clearance glinfática cerebral "
            "(species_chronobiology.sleep_efficiency_threshold e "
            "species_endocrine.growth_hormone_sleep_dependency)."
        ),
    )

    sleep_duration_hours: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Duração média do sono (wearable ou diário). Unidade: horas decimais. "
            "Referência atletas: 8-10h (recomendação ACSM/NSCA para atletas de "
            "alto rendimento em fase de carga). "
            "Sono < 7h cronicamente → redução de testosterone_total em ~10-15% "
            "(Leproult & Van Cauter, JAMA 2011), aumento de cortisol vespertino, "
            "comprometimento de BDNF e neuroplasticidade "
            "(species_endocrine.sleep_testosterone_dependency)."
        ),
    )

    sleep_sws_percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Percentagem de sono de ondas lentas (SWS / N3) no sono total. "
            "Unidade: %. Referência adulto jovem: 15-25% do sono total. "
            "SWS é a fase de maior secreção de GH, maior síntese proteica muscular "
            "e maior actividade de reparação celular. "
            "SWS < 10% → défice de GH nocturna → comprometimento de recuperação "
            "muscular após treino de força "
            "(species_endocrine.growth_hormone_sleep_dependency). "
            "SWS reduz com a idade (~2% por década após os 30); treino de alta "
            "intensidade no dia anterior aumenta SWS transitoriamente."
        ),
    )

    sleep_rem_percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Percentagem de sono REM no sono total. Unidade: %. "
            "Referência: 20-25% do sono total. "
            "REM é a fase de: (1) consolidação de memória motora procedural (sequências "
            "técnicas, habilidades desportivas novas); (2) regulação emocional (via "
            "amígdala e córtex pré-frontal); (3) síntese de ACh e modulação "
            "serotoninérgica e noradrenérgica. "
            "REM < 15% → aprendizagem técnica comprometida e maior labilidade emocional "
            "(species_neural_cognitive.procedural_memory_consolidation_rem_dependency). "
            "REM suprimido por: álcool, cannabis, benzodiazepinas, beta-bloqueantes."
        ),
    )

    # ── Relações ─────────────────────────────────────────────────────────────
    athlete: Mapped["AthleteCore"] = relationship(
        back_populates="psych_assessments"
    )
