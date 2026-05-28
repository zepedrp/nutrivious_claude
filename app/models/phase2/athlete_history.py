from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, Text, func
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase2.athlete_core import AthleteCore


class AthleteHistory(Base):
    """
    Historial de treino, contexto desportivo e factores de estilo de vida — Módulo A.

    Relação 1:1 com AthleteCore (unique=True em athlete_id): é o contexto
    permanente que informa toda a interpretação dos outros clusters.
    A 'idade de treino' é uma das variáveis mais preditivas da taxa de
    adaptação residual ao estímulo de treino — atletas com >10 anos de treino
    sistemático têm margens de progressão menores mas tectos epigenéticos mais
    elevados do que iniciantes.
    """

    __tablename__ = "athlete_history"
    __table_args__ = (
        UniqueConstraint("athlete_id", name="uq_athlete_history_athlete"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    athlete_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("athlete_core.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    last_updated: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        comment="Data da última actualização do historial (rever a cada 3-6 meses).",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # BACKGROUND DESPORTIVO
    # Interage com: todos os clusters — contextualiza a interpretação dos CPI
    # ══════════════════════════════════════════════════════════════════════════

    primary_sport: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "Modalidade desportiva principal actual. Exemplos: 'Ciclismo de Estrada', "
            "'Triatlo', 'Futebol', 'Levantamento de Pesos', 'Natação'. "
            "Determina quais os sistemas fisiológicos prioritários e quais os tectos "
            "species_* mais relevantes para o CPI primário do atleta."
        ),
    )

    secondary_sport: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "Modalidade desportiva secundária ou histórica relevante. "
            "Ex: ex-atleta de natação que transicionou para triatlo — "
            "o background de natação informa a reserva cardiovascular e técnica aquática."
        ),
    )

    training_age_years: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Idade de treino — anos de treino sistemático e estruturado na modalidade "
            "principal (ou modalidade base). Unidade: anos decimais. "
            "A idade de treino é o modificador mais importante do Gradiente de "
            "adaptação: Gradiente_adaptação = (Tecto_Individual − Baseline) / Tecto_Individual. "
            "Iniciante (< 2 anos): gradiente alto → grandes ganhos com pequenos estímulos. "
            "Intermédio (2-5 anos): ganhos moderados; periodização necessária. "
            "Avançado (5-10 anos): ganhos requerem sobrecarga progressiva e variação. "
            "Elite (> 10 anos): margens de progressão estreitas; pequenos ganhos "
            "requerem intervenções altamente específicas e individualizadas. "
            "Calibra a taxa de progressão esperada em todos os sistemas da Fase 2."
        ),
    )

    competition_level: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment=(
            "Nível competitivo actual. Valores: 'recreational' | 'regional' | "
            "'national' | 'international' | 'elite_professional' | 'olympic'. "
            "Informa o contexto de comparação para os CPIs e os thresholds de "
            "intervenção (ex: um VO2max de 55 mL/kg/min é excelente para regional "
            "mas insuficiente para elite internacional em endurance)."
        ),
    )

    sport_start_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
        comment=(
            "Data de início do treino sistemático na modalidade actual. "
            "Permite calcular training_age_years de forma automática se não fornecido: "
            "training_age_years = (assessment_date − sport_start_date).years."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # CARGA DE TREINO ACTUAL
    # Contexto para interpretar biomarkers e performance
    # ══════════════════════════════════════════════════════════════════════════

    weekly_training_hours: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Volume semanal médio de treino (últimas 4-8 semanas). Unidade: horas. "
            "Referência: recreativo 5-10h; alto rendimento 12-20h; elite 20-35h. "
            "Contextualiza os biomarkers de stress (CK, urea, cortisol) e o "
            "RMSSD: um RMSSD de 45ms em atleta de 20h/sem é diferente de "
            "45ms em atleta de 8h/sem."
        ),
    )

    weekly_training_sessions: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment=(
            "Número médio de sessões de treino por semana. "
            "Referência: recreativo 3-5; alto rendimento 6-10; elite 10-16 (bi-diário). "
            "Número de sessões determina a janela de recuperação inter-sessão: "
            "2 sessões/dia → <12h de recuperação → biomarkers de dano crónico "
            "(CK crónico >500 U/L) são mais toleráveis."
        ),
    )

    periodization_phase: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment=(
            "Fase de periodização actual. Valores: 'base' | 'build' | 'peak' | "
            "'competition' | 'transition' | 'rehabilitation'. "
            "Crítico para interpretar todos os clusters: na fase de 'build' "
            "esperam-se marcadores de fadiga mais elevados (CK, urea, RMSSD baixo) "
            "que na fase 'peak' ou 'competition'."
        ),
    )

    altitude_training_history: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment=(
            "True se o atleta tem histórico de treino em altitude (>2000m) "
            "ou em tenda hipóxica nas últimas 12 semanas. "
            "Altitude → eritropoiese estimulada por HIF1α/EPO → Hb e ferritina "
            "transitoriamente elevadas pós-campo (até 3-4 semanas). "
            "Interpreta Hb e Hct com cautela se altitude_training_history = True "
            "(species_hematological.altitude_erythropoiesis_modifier)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORIAL DE LESÕES
    # Interage com: species_musculoskeletal, species_osseous_system
    # ══════════════════════════════════════════════════════════════════════════

    injury_history_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Resumo narrativo de lesões musculoesqueléticas relevantes. "
            "Formato sugerido: 'Rotura parcial LCA joelho D (2021, cirurgia), "
            "tendinopatia patelar bilateral (2023, conservador)'. "
            "Informa: (1) risco de recidiva e thresholds de volume seguros; "
            "(2) interpretação de assimetrias biomecânicas (CMJ, IMTP, dorsiflexão); "
            "(3) selecção de exercícios de prevenção prioritários. "
            "Para lesões complexas, criar registo estruturado separado (fora do BOS)."
        ),
    )

    fractures_stress_history: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment=(
            "True se o atleta tem histórico de fractura(s) de stress. "
            "Activa: (1) monitorização de bmd_total_g_per_cm2 semestral; "
            "(2) rastreio de RED-S (energy availability < 30 kcal/kg FFM/dia); "
            "(3) avaliação de vitamin_d_25oh e rbc_calcium. "
            "Interacção com COL1A1 TT genótipo (menor BMD basal) e com "
            "estradiol_baixo em F (Tríade da atleta feminina). "
            "species_osseous_system.stress_fracture_risk_modifier."
        ),
    )

    surgeries_relevant: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment=(
            "Cirurgias relevantes para o programa de treino. "
            "Exemplos: 'Reconstrução LCA joelho D com tendão patelar (2021)', "
            "'Artroscopia ombro E por SLAP (2020)'. "
            "Informa thresholds de carga seguros e exercícios contra-indicados."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FACTORES DE ESTILO DE VIDA E CONTEXTO
    # Contexto geral para interpretação multi-eixo
    # ══════════════════════════════════════════════════════════════════════════

    dietary_pattern: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment=(
            "Padrão alimentar predominante. Valores: 'omnivore' | 'mediterranean' | "
            "'vegetarian' | 'vegan' | 'ketogenic' | 'low_carb' | 'plant_based_high_protein'. "
            "Informa: (1) risco de défices específicos (B12 em vegan; ferro heme em plant-based; "
            "DHA em plant-based + FADS1 TT); "
            "(2) composição do microbioma (F/B ratio, Bifidobacterium em alta fibra); "
            "(3) disponibilidade de creatina endógena (ausente em vegan)."
        ),
    )

    sleep_aids_used: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        comment=(
            "Auxiliares de sono utilizados regularmente. "
            "Exemplos: 'melatonina 0.5mg', 'magnésio glicinato 400mg', "
            "'nenhum', 'alprazolam 0.5mg'. "
            "Benzodiazepinas e Z-drugs suprimem SWS e REM → interpretar "
            "sleep_sws_percentage e rem_percentage com cautela. "
            "Álcool ao jantar (> 2 unidades) → supressão de REM na primeira metade "
            "do sono → deficit de memória motora."
        ),
    )

    supplements_current: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Lista de suplementos em uso no momento da avaliação (formato livre). "
            "Exemplos: 'creatina monohidratada 5g/dia, beta-alanina 4g/dia, "
            "cafeína 200mg pré-treino, vitamina D 4000 UI/dia, ómega-3 3g EPA+DHA/dia'. "
            "Crítico para interpretar biomarkers: creatina → creatinina sérica elevada "
            "(não confundir com insuficiência renal); proteína em pó → ureia elevada; "
            "ferro suplementar → ferritina elevada (pode mascarar sobrecarga em HFE). "
            "Permite calcular gap entre ingestão total e necessidades prescritas."
        ),
    )

    medications_chronic: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Medicação crónica relevante para o programa fisiológico. "
            "Exemplos: 'levotiroxina 50μg/dia (hipotiroidismo)', 'metformina 500mg'. "
            "Impactos conhecidos: beta-bloqueantes → FCmax artificial < real → "
            "NUNCA usar para calcular zonas; metformina → supressão de complexo I "
            "mitocondrial → reduz produção de lactato → LT1 artificialmente baixo "
            "no teste incremental; SSRI → supressão REM. "
            "Lista não-exhaustiva: documentar para interpretar anomalias nos testes."
        ),
    )

    smoking_status: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment=(
            "'never' | 'ex_smoker' | 'current' | 'occasional'. "
            "Tabagismo activo: redução de VO2max (HbCO↑), aceleração de "
            "DunedinPACE e GrimAge, telómeros mais curtos. "
            "Ex-fumador: GrimAge permanentemente modificado pelo histórico de tabagismo "
            "(inclui 'pack-years' como input do algoritmo GrimAge)."
        ),
    )

    alcohol_units_per_week: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Consumo médio de álcool. Unidade: unidades padrão/semana (1 unidade = 10g etanol). "
            "Referência de baixo risco: H < 14 unidades/sem; F < 11 unidades/sem. "
            "Álcool > 3 unidades em noite de treino pesado → supressão de síntese "
            "proteica pós-exercício em ~24% (Parr et al., 2014), supressão de "
            "testosterone_total, comprometimento de SWS e REM. "
            "species_endocrine.alcohol_androgen_suppression_modifier."
        ),
    )

    # ── Relações ─────────────────────────────────────────────────────────────
    athlete: Mapped["AthleteCore"] = relationship(
        back_populates="history"
    )
