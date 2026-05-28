from app.models.phase1.core import Base, SpeciesCore

# Lote 1 — Energy & Substrate
from app.models.phase1.bioenergetics import SpeciesBioenergetics
from app.models.phase1.mitochondrial import SpeciesMitochondrial
from app.models.phase1.lipid_metabolism import SpeciesLipidMetabolism
from app.models.phase1.protein_metabolism import SpeciesProteinMetabolism

# Lote 2 — Absorption, Detox & Filtration
from app.models.phase1.gastrointestinal import SpeciesGastrointestinal
from app.models.phase1.hepatic import SpeciesHepatic
from app.models.phase1.renal import SpeciesRenal

# Lote 3 — Transport & Fluid Homeostasis
from app.models.phase1.cardiovascular import SpeciesCardiovascular
from app.models.phase1.pulmonary import SpeciesPulmonary
from app.models.phase1.fluid_electrolyte import SpeciesFluidElectrolyte

# Lote 4 — Neuromuscular Output & Motor Command
from app.models.phase1.neuromuscular import SpeciesNeuromuscular
from app.models.phase1.musculoskeletal import SpeciesMusculoskeletal
from app.models.phase1.neural_cognitive import SpeciesNeuralCognitive

# Lote 5 — Systemic & Temporal Regulation
from app.models.phase1.endocrine import SpeciesEndocrine
from app.models.phase1.chronobiology import SpeciesChronobiology
from app.models.phase1.thermoregulation import SpeciesThermoregulation

# Lote 6 — Molecular Control & Defense
from app.models.phase1.epigenetic import SpeciesEpigenetic
from app.models.phase1.oxidative_stress import SpeciesOxidativeStress
from app.models.phase1.immune_microbiome import SpeciesImmuneMicrobiome

# Lote 7 — Structural Support
from app.models.phase1.osseous_system import SpeciesOsseousSystem

# Lote 8 — Excretion & Haematology
from app.models.phase1.renal_excretory import SpeciesRenalExcretory
from app.models.phase1.hematological import SpeciesHematological

__all__ = [
    "Base",
    "SpeciesCore",
    # Lote 1
    "SpeciesBioenergetics",
    "SpeciesMitochondrial",
    "SpeciesLipidMetabolism",
    "SpeciesProteinMetabolism",
    # Lote 2
    "SpeciesGastrointestinal",
    "SpeciesHepatic",
    "SpeciesRenal",
    # Lote 3
    "SpeciesCardiovascular",
    "SpeciesPulmonary",
    "SpeciesFluidElectrolyte",
    # Lote 4
    "SpeciesNeuromuscular",
    "SpeciesMusculoskeletal",
    "SpeciesNeuralCognitive",
    # Lote 5
    "SpeciesEndocrine",
    "SpeciesChronobiology",
    "SpeciesThermoregulation",
    # Lote 6
    "SpeciesEpigenetic",
    "SpeciesOxidativeStress",
    "SpeciesImmuneMicrobiome",
    # Lote 7
    "SpeciesOsseousSystem",
    # Lote 8
    "SpeciesRenalExcretory",
    "SpeciesHematological",
]
