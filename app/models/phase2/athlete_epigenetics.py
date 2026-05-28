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


class AthleteEpigenetics(Base):
    """
    Perfil epigenético do atleta — Eixo 2 / Módulo B.

    Relação 1:many com AthleteCore: o epigenoma é dinâmico e muda com treino,
    nutrição e stress. Séries temporais (baseline → mid-season → pós-época)
    permitem calcular a taxa de envelhecimento biológico (DunedinPACE).
    Cada coluna interage com species_epigenetic (Fase 1).
    """

    __tablename__ = "athlete_epigenetics"

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
            "Data da análise epigenética (colheita de sangue para arrays de metilação "
            "como Illumina EPIC 850K ou sequenciação de bisulfito WGBS). "
            "Permite comparar relógio epigenético ao longo do tempo: "
            "DunedinPACE_t2 − DunedinPACE_t1 → velocidade de envelhecimento no período."
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RELÓGIOS EPIGENÉTICOS (BIOLOGICAL AGE CLOCKS)
    # Interage com: species_epigenetic.biological_age_ceiling,
    #               species_epigenetic.methylation_clock_reference
    # ══════════════════════════════════════════════════════════════════════════

    horvath_clock_age: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Relógio de Horvath (2013) — primeira geração. Unidade: anos. "
            "Calculado a partir de 353 CpGs pan-tecidulares (Illumina 450K/EPIC). "
            "Delta_Horvath = horvath_clock_age − idade_cronológica: "
            "negativo → envelhecimento biológico mais lento que o calendário; "
            "positivo → envelhecimento acelerado. "
            "Atletas de endurance de elite apresentam delta médio de −3 a −5 anos. "
            "Interacção: species_epigenetic.dna_methylation_clock_reference "
            "define a trajectória esperada para a idade cronológica da Fase 1. "
            "Limitação: insensível a alterações agudas de treino (integra >2 anos)."
        ),
    )

    grimage_biological_age: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Relógio GrimAge (Lu et al., 2019) — segunda geração. Unidade: anos. "
            "Treinado em mortalidade all-cause; incorpora 8 proteínas plasmáticas "
            "derivadas de metilação (GDF15, PAI1, leptin, TIMP1, ADM, B2M, cystatin-C, "
            "pack-years de tabaco proxy). "
            "GrimAge é o relógio de mortalidade mais validado: cada ano de "
            "grimage_biological_age > idade_cronológica → +15% risco de mortalidade "
            "all-cause em 10 anos. "
            "species_epigenetic.mortality_risk_methylation_index. "
            "Sensível a obesidade, tabagismo, stress crónico e sedentarismo."
        ),
    )

    dunedinpace_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "DunedinPACE — Pace of Aging Calculated from the Epigenome. Escala: anos/ano. "
            "Calculado a partir de 173 CpGs calibrados no estudo de coorte Dunedin (NZ). "
            "Valor médio populaconal = 1.0 (1 ano biológico por 1 ano calendário). "
            "DunedinPACE = 0.8 → o atleta envelhece 0.8 anos biológicos por cada ano "
            "calendário (envelhecimento lento — perfil desejável). "
            "DunedinPACE = 1.2 → aceleração de envelhecimento. "
            "Delta_DunedinPACE = score_t2 − score_t1 / Δt → velocidade de mudança. "
            "Sensível a intervenções de curto prazo (6-12 meses): exercício aeróbio, "
            "restrição calórica, sono, redução de stress. "
            "Calcular DunedinPACE_delta = DunedinPACE − 1.0 para o CPI epigenético: "
            "species_epigenetic.aging_pace_reference (1.0 = tecto espécie médio)."
        ),
    )

    phenoage_biological_age: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "PhenoAge (Levine et al., 2018) — relógio de segunda geração. Unidade: anos. "
            "513 CpGs; treinado em marcadores bioquímicos de envelhecimento fenotípico "
            "(albumina, creatinina, glucose, CRP, linfócitos%, volume eritrocitário médio, "
            "fosfatase alcalina, glóbulos brancos, idade cronológica). "
            "PhenoAge captura a componente inflamatória do envelhecimento epigenético "
            "('inflammaging'). "
            "Correlação directa com species_immune_microbiome.chronic_inflammation_threshold: "
            "atletas com crp_high_sensitivity cronicamente >2 mg/L tendem a ter "
            "PhenoAge 2-4 anos acima da cronológica."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TELÓMEROS
    # Interage com: species_epigenetic.telomere_length_reference
    # ══════════════════════════════════════════════════════════════════════════

    telomere_length_kb: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Comprimento médio dos telómeros em leucócitos. Unidade: kb (kilobases). "
            "Medição: qPCR (rácio T/S) ou FISH fluxo. Referência 30 anos: ~7-8 kb. "
            "Os telómeros encurtam ~50 pb/ano em indivíduos sedentários; "
            "atletas de endurance apresentam telómeros 5-10% mais longos que controlos "
            "sedentários de mesma idade (espécie: telomerase activada pelo exercício). "
            "Telómeros <5 kb → risco de senescência celular prematura → disfunção "
            "imunológica e comprometimento de regeneração de células satélite musculares. "
            "species_epigenetic.telomere_length_reference define o intervalo esperado "
            "para a idade cronológica."
        ),
    )

    telomere_length_percentile: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Percentil do comprimento telomérico relativo a coorte de mesma idade e sexo. "
            "Escala: 0-100. "
            "Percentil ≥ 75 → comprimento telomérico preservado para a idade. "
            "Percentil < 25 → encurtamento acelerado → investigar causas: "
            "stress oxidativo crónico, tabagismo, obesidade, overtreino sem recuperação "
            "(interacção com species_oxidative_stress.reactive_oxygen_species_threshold)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # METILAÇÃO DO ADN — MARCADORES FUNCIONAIS
    # Interage com: species_epigenetic.sam_sah_ratio_reference,
    #               species_epigenetic.global_methylation_reference
    # ══════════════════════════════════════════════════════════════════════════

    global_dna_methylation_percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Metilação global do ADN (5-mC como % do total de citosinas). "
            "Medição: ELISA colorimétrico ou LUMA (Luminometric Methylation Assay). "
            "Referência: 70-80% de citosinas em contexto CpG estão metiladas no genoma adulto. "
            "Hipometilação global (<65%) → activação de elementos transponíveis, "
            "instabilidade genómica, risco oncológico. "
            "Associada a défice de SAM (S-adenosilmetionina) → défice de folato/B12/B6 "
            "(MTHFR C677T TT agrava este risco: species_epigenetic.sam_sah_ratio_reference). "
            "Monitorizar em atletas com MTHFR TT + défice vitamínico (ver biomarkers)."
        ),
    )

    sam_sah_ratio: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Rácio SAM/SAH (S-adenosilmetionina / S-adenosilhomocisteína). "
            "Medição: HPLC-MS/MS em eritrócitos ou plasma. "
            "SAM é o dador universal de grupos metilo para metilação do ADN, ARN e proteínas. "
            "SAH é o produto da reacção e inibidor competitivo das metiltransferases. "
            "Rácio óptimo: > 4.5 (H: 4-8; F: 4-8 como guia geral). "
            "Rácio < 3.0 → compromisso das reacções de metilação → hipermetilação de "
            "promotores de genes supressores e hipometilação global simultânea "
            "(paradoxo epigenético) → impacta species_epigenetic.sam_sah_ratio_reference. "
            "Causas: défice de folato ativo (5-MTHF), B12, B6, betaína, colina; "
            "agravado por MTHFR C677T TT e CBS A360A variantes."
        ),
    )

    elovl2_methylation_percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Metilação do gene ELOVL2 (Elongation of Very Long Chain Fatty Acids 2). "
            "Unidade: % metilação em CpGs específicos do promotor. "
            "ELOVL2 é um dos CpGs mais altamente correlacionados com a idade cronológica "
            "(r > 0.96 em múltiplos estudos). "
            "Utilizado como âncora de calibração para os relógios epigenéticos: "
            "taxa de metilação ~40% aos 20 anos → ~70% aos 60 anos. "
            "Desvio positivo (metilação mais alta que esperado para a idade) → "
            "envelhecimento acelerado neste locus específico."
        ),
    )

    fhl2_methylation_percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Metilação do gene FHL2 (Four and a Half LIM Domains 2). "
            "Unidade: % metilação. "
            "FHL2 regula a sinalização de androgénios e integrinas no músculo cardíaco "
            "e esquelético; a sua metilação aumenta progressivamente com a idade. "
            "Em atletas de força: menor metilação de FHL2 correlaciona-se com maior "
            "resposta hipertrófica ao treino de resistência "
            "(interacção com species_musculoskeletal.hypertrophy_epigenetic_modifier)."
        ),
    )

    # ── Relações ─────────────────────────────────────────────────────────────
    athlete: Mapped["AthleteCore"] = relationship(
        back_populates="epigenetics_assessments"
    )
