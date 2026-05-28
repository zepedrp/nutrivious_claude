from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from app.models.phase1.bioenergetics import SpeciesBioenergetics
    from app.models.phase1.mitochondrial import SpeciesMitochondrial
    from app.models.phase1.lipid_metabolism import SpeciesLipidMetabolism
    from app.models.phase1.protein_metabolism import SpeciesProteinMetabolism
    # Lote 2
    from app.models.phase1.gastrointestinal import SpeciesGastrointestinal
    from app.models.phase1.hepatic import SpeciesHepatic
    from app.models.phase1.renal import SpeciesRenal
    # Lote 3
    from app.models.phase1.cardiovascular import SpeciesCardiovascular
    from app.models.phase1.pulmonary import SpeciesPulmonary
    from app.models.phase1.fluid_electrolyte import SpeciesFluidElectrolyte
    # Lote 4
    from app.models.phase1.neuromuscular import SpeciesNeuromuscular
    from app.models.phase1.musculoskeletal import SpeciesMusculoskeletal
    from app.models.phase1.neural_cognitive import SpeciesNeuralCognitive
    # Lote 5
    from app.models.phase1.endocrine import SpeciesEndocrine
    from app.models.phase1.chronobiology import SpeciesChronobiology
    from app.models.phase1.thermoregulation import SpeciesThermoregulation
    # Lote 6
    from app.models.phase1.epigenetic import SpeciesEpigenetic
    from app.models.phase1.oxidative_stress import SpeciesOxidativeStress
    from app.models.phase1.immune_microbiome import SpeciesImmuneMicrobiome
    # Lote 7
    from app.models.phase1.osseous_system import SpeciesOsseousSystem
    # Lote 8
    from app.models.phase1.renal_excretory import SpeciesRenalExcretory
    from app.models.phase1.hematological import SpeciesHematological


class Base(DeclarativeBase):
    pass


class SpeciesCore(Base):
    __tablename__ = "species_core"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    species_name: Mapped[str] = mapped_column(
        String(128), nullable=False, default="Homo sapiens"
    )
    common_name: Mapped[str] = mapped_column(
        String(128), nullable=False, default="Human"
    )
    taxonomic_class: Mapped[str] = mapped_column(
        String(64), nullable=False, default="Mammalia"
    )
    reference_genome_assembly: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    schema_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="1.0.0"
    )
    data_source_citation: Mapped[Optional[str]] = mapped_column(
        String(1024), nullable=True
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

    # ── Lote 1 — Energy Production & Substrate ──────────────────────────────
    bioenergetics: Mapped[Optional["SpeciesBioenergetics"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    mitochondrial: Mapped[Optional["SpeciesMitochondrial"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    lipid_metabolism: Mapped[Optional["SpeciesLipidMetabolism"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    protein_metabolism: Mapped[Optional["SpeciesProteinMetabolism"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )

    # ── Lote 2 — Absorption, Detox & Filtration ─────────────────────────────
    gastrointestinal: Mapped[Optional["SpeciesGastrointestinal"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    hepatic: Mapped[Optional["SpeciesHepatic"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    renal: Mapped[Optional["SpeciesRenal"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )

    # ── Lote 3 — Transport & Fluid Homeostasis ───────────────────────────────
    cardiovascular: Mapped[Optional["SpeciesCardiovascular"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    pulmonary: Mapped[Optional["SpeciesPulmonary"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    fluid_electrolyte: Mapped[Optional["SpeciesFluidElectrolyte"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )

    # ── Lote 4 — Neuromuscular Output & Motor Command ────────────────────────
    neuromuscular: Mapped[Optional["SpeciesNeuromuscular"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    musculoskeletal: Mapped[Optional["SpeciesMusculoskeletal"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    neural_cognitive: Mapped[Optional["SpeciesNeuralCognitive"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )

    # ── Lote 5 — Systemic & Temporal Regulation ─────────────────────────────
    endocrine: Mapped[Optional["SpeciesEndocrine"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    chronobiology: Mapped[Optional["SpeciesChronobiology"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    thermoregulation: Mapped[Optional["SpeciesThermoregulation"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )

    # ── Lote 6 — Molecular Control & Defense ────────────────────────────────
    epigenetic: Mapped[Optional["SpeciesEpigenetic"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    oxidative_stress: Mapped[Optional["SpeciesOxidativeStress"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    immune_microbiome: Mapped[Optional["SpeciesImmuneMicrobiome"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )

    # ── Lote 7 — Structural Support ─────────────────────────────────────────
    osseous_system: Mapped[Optional["SpeciesOsseousSystem"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )

    # ── Lote 8 — Excretion & Haematology ────────────────────────────────────
    renal_excretory: Mapped[Optional["SpeciesRenalExcretory"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
    hematological: Mapped[Optional["SpeciesHematological"]] = relationship(
        back_populates="species_core", uselist=False, cascade="all, delete-orphan"
    )
