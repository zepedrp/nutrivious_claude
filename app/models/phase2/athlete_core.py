from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Date, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase2.athlete_biomarkers import AthleteBiomarkers
    from app.models.phase2.athlete_dexa import AthleteDexa
    from app.models.phase2.athlete_epigenetics import AthleteEpigenetics
    from app.models.phase2.athlete_genetics import AthleteGenetics
    from app.models.phase2.athlete_history import AthleteHistory
    from app.models.phase2.athlete_microbiome import AthleteMicrobiome
    from app.models.phase2.athlete_performance import AthletePerformance
    from app.models.phase2.athlete_psych import AthletePsych


class AthleteCore(Base):
    __tablename__ = "athlete_core"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # ── Herança Crítica — ancora o atleta ao tecto teórico da espécie ────────
    species_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("species_core.id", ondelete="RESTRICT"),
        nullable=False,
        comment=(
            "FK → species_core.id. Liga o atleta à estrutura da Fase 1 (Motor de Espécie). "
            "Todos os cálculos de performance individual são expressos como fracção ou "
            "multiplicador dos tectos absolutos da Homo sapiens definidos aí. "
            "CPI(sistema_i) = Baseline_individual(i) / Tecto_Espécie(i). "
            "ondelete=RESTRICT: um registo de espécie não pode ser apagado enquanto "
            "existirem atletas ligados a ele."
        ),
    )

    name: Mapped[str] = mapped_column(String(256), nullable=False)

    date_of_birth: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        comment=(
            "Data de nascimento. Usada para calcular a idade cronológica e seleccionar "
            "os intervalos de referência estratificados por idade da Fase 1 "
            "(VO2max, DMO, hormonas, hematologia). "
            "Permite calcular o DunedinPACE delta (idade biológica - cronológica)."
        ),
    )

    biological_sex: Mapped[str] = mapped_column(
        String(1),
        nullable=False,
        comment=(
            "'M' (Masculino) | 'F' (Feminino). "
            "Interruptor dimórfico aplicado em todos os módulos da Fase 1: "
            "Hb referência (H 13.5-17.5 g/dL vs M 12.0-16.0 g/dL), tecto VO2max, "
            "trajectória BMD mediada por estrogénios, limites de hematócrito "
            "(H 42-52%, M 37-47%), baselines hormonais e adiposidade essencial "
            "(mínimo ~3% H, ~12% M). "
            "Também determina os thresholds IAMM (H > 7.26 kg/m², M > 5.50 kg/m²) "
            "e TBW% (H 60-70%, M 50-60%)."
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ── Relações ─────────────────────────────────────────────────────────────
    dexa_assessments: Mapped[List["AthleteDexa"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan"
    )

    genetics_profile: Mapped[Optional["AthleteGenetics"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan", uselist=False
    )

    biomarker_panels: Mapped[List["AthleteBiomarkers"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan"
    )

    epigenetics_assessments: Mapped[List["AthleteEpigenetics"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan"
    )

    microbiome_assessments: Mapped[List["AthleteMicrobiome"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan"
    )

    performance_assessments: Mapped[List["AthletePerformance"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan"
    )

    psych_assessments: Mapped[List["AthletePsych"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan"
    )

    history: Mapped[Optional["AthleteHistory"]] = relationship(
        back_populates="athlete", cascade="all, delete-orphan", uselist=False
    )
