import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

load_dotenv()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

# Import Base and ALL models so autogenerate sees every table
from app.models.phase1 import (  # noqa: E402
    Base,
    SpeciesCore,
    SpeciesBioenergetics,
    SpeciesMitochondrial,
    SpeciesLipidMetabolism,
    SpeciesProteinMetabolism,
    SpeciesGastrointestinal,
    SpeciesHepatic,
    SpeciesRenal,
    SpeciesCardiovascular,
    SpeciesPulmonary,
    SpeciesFluidElectrolyte,
    SpeciesNeuromuscular,
    SpeciesMusculoskeletal,
    SpeciesNeuralCognitive,
    SpeciesEndocrine,
    SpeciesChronobiology,
    SpeciesThermoregulation,
    SpeciesEpigenetic,
    SpeciesOxidativeStress,
    SpeciesImmuneMicrobiome,
    SpeciesOsseousSystem,
    SpeciesRenalExcretory,
    SpeciesHematological,
)

from app.models.phase2 import (  # noqa: E402
    AthleteBiomarkers,
    AthleteCore,
    AthleteDexa,
    AthleteEpigenetics,
    AthleteGenetics,
    AthleteHistory,
    AthleteMicrobiome,
    AthletePerformance,
    AthletePsych,
)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
