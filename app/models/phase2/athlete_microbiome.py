from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Float, func
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase2.athlete_core import AthleteCore


class AthleteMicrobiome(Base):
    """
    Perfil do microbioma intestinal do atleta — Eixo 6 / Módulo E.

    Relação 1:many com AthleteCore: o microbioma é dinâmico (responde em dias
    a mudanças de dieta e carga de treino). Análise de amostras fecais por
    16S rRNA sequencing (V3-V4) ou shotgun metagenomics.
    Interage com species_gastrointestinal, species_immune_microbiome,
    species_bioenergetics (via SCFAs e metabolismo energético).
    """

    __tablename__ = "athlete_microbiome"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    athlete_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("athlete_core.id", ondelete="CASCADE"),
        nullable=False,
    )

    sample_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        comment=(
            "Data de colheita da amostra fecal. "
            "Protocolo: amostra de manhã, antes de defecação após 12h de jejum nocturno, "
            "sem antibióticos nas 4 semanas anteriores, sem probióticos nas 2 semanas "
            "anteriores. Preservar em tampão RNAlater a −20°C até extracção. "
            "Para série temporal: baseline → 4 semanas de intervenção nutricional → "
            "re-avaliação (taxa de resposta do microbioma: ~2-4 semanas)."
        ),
    )

    sequencing_method: Mapped[str | None] = mapped_column(
        __import__('sqlalchemy').String(64),
        nullable=True,
        comment=(
            "'16S_V3V4' | '16S_V4' | 'shotgun_WGS' | 'ITS_fungi'. "
            "O método determina a resolução taxonómica: 16S identifica género; "
            "WGS identifica espécie e permite inferência funcional (KEGG pathways, "
            "CAZymes, SCFA gene clusters). Impacta a interpretação dos rácios."
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # DIVERSIDADE ALFA (DENTRO DA AMOSTRA)
    # Interage com: species_gastrointestinal.microbiome_diversity_reference,
    #               species_immune_microbiome.gut_diversity_threshold
    # ══════════════════════════════════════════════════════════════════════════

    shannon_diversity_index: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Índice de diversidade de Shannon (H'). Escala típica em humanos: 2.5-4.5. "
            "H' = −Σ(pi × ln(pi)) onde pi = proporção relativa do taxon i. "
            "Combina riqueza (número de espécies) e equitabilidade (distribuição). "
            "H' > 3.5 → microbioma diverso e resiliente → melhor resposta imunológica "
            "adaptativa e inata, maior produção de SCFAs, menor permeabilidade intestinal "
            "(species_gastrointestinal.microbiome_diversity_reference). "
            "H' < 2.5 → disbiose → risco de infecções respiratórias peri-competição, "
            "síntese reduzida de neurotransmissores (serotonina; ~90% produzida no intestino), "
            "menor absorção de micronutrientes. "
            "Atletas de ultra-endurance tendem a ter H' mais baixo por dietas restritivas "
            "e stress oxidativo GI durante provas longas."
        ),
    )

    species_richness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Riqueza de espécies observadas (OTUs/ASVs identificadas). "
            "Escala típica: 100-500 OTUs em sequenciação 16S com rarefacção a 10,000 reads. "
            "Complemento do Shannon: um microbioma pode ter alta riqueza mas baixa "
            "equitabilidade (uma espécie domina) → Shannon baixo. "
            "Riqueza > 300 OTUs → reservatório funcional amplo → maior redundância "
            "metabólica (se uma via é comprometida, outra espécie supre a função). "
            "Interacção com species_gastrointestinal.microbiome_richness_reference."
        ),
    )

    chao1_estimate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Estimador Chao1 de riqueza total (inclui espécies raras não observadas). "
            "Sempre ≥ species_richness observada. "
            "Razão Chao1/richness_observada → proporção do microbioma ainda não "
            "capturada pela profundidade de sequenciação actual. "
            "Chao1 > 500 → microbioma rico em espécies raras → reserva funcional "
            "metabólica elevada para adaptação a mudanças de dieta e stress fisiológico."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # RÁCIOS DE FILO
    # Interage com: species_gastrointestinal.firmicutes_bacteroidetes_ratio_reference
    # ══════════════════════════════════════════════════════════════════════════

    firmicutes_bacteroidetes_ratio: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Rácio Firmicutes / Bacteroidetes (F/B ratio). "
            "Referência saudável: 0.5-2.5. "
            "Firmicutes (Lachnospiraceae, Ruminococcaceae, Clostridia) → produtores "
            "de butirato; eficiência de extracção calórica dos carbohidratos. "
            "Bacteroidetes (Bacteroides, Prevotella) → produtores de propionato e "
            "acetato; regulação do metabolismo lipídico hepático. "
            "F/B > 3.0 → associado a obesidade, maior extracção calórica, disbiose "
            "metabólica; compromisso de species_bioenergetics.caloric_extraction_efficiency. "
            "F/B < 0.3 → dominância de Bacteroidetes; pode indicar dieta muito alta "
            "em fibra mas baixa em proteína animal. "
            "Em atletas de endurance: F/B tende para 1.0-1.5 com dieta mista equilibrada."
        ),
    )

    proteobacteria_percentage: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Abundância relativa de Proteobacteria (%). "
            "Referência saudável: < 5%. "
            "Proteobacteria incluem E. coli, Helicobacter, Salmonella e outros "
            "patóbiontes gram-negativos com LPS na parede celular. "
            "Proteobacteria > 10% → marcador de disbiose inflamatória → aumento de "
            "LPS circulante (endotoxemia metabólica) → activação de TLR4 → "
            "NF-κB → IL-6, TNF-α → resistência à insulina e inflamação sistémica "
            "(species_immune_microbiome.lps_endotoxin_threshold)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ESPÉCIES-CHAVE DE PERFORMANCE
    # Interage com: species_bioenergetics, species_cardiovascular,
    #               species_immune_microbiome
    # ══════════════════════════════════════════════════════════════════════════

    akkermansia_muciniphila_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Abundância relativa de Akkermansia muciniphila (%). "
            "Referência desejável em atletas: 1-5% (0.5-8% em adultos saudáveis). "
            "A. muciniphila coloniza a camada de muco intestinal (produce mucinases). "
            "Funções: reforço da barreira intestinal (aumenta expressão de claudina-3 "
            "e occludina), redução de permeabilidade, activação do receptor GLP-1 → "
            "maior sensibilidade à insulina (HOMA-IR ↓). "
            "A. muciniphila < 0.1% → barreira intestinal comprometida → 'leaky gut' → "
            "endotoxemia → inflamação sistémica (species_gastrointestinal.mucosal_barrier_integrity). "
            "Exercício aeróbio moderado (>150 min/sem) aumenta A. muciniphila; "
            "ultra-endurance de alta intensidade pode reduzi-la transitoriamente."
        ),
    )

    faecalibacterium_prausnitzii_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Abundância relativa de Faecalibacterium prausnitzii (%). "
            "Referência desejável: 5-15% (é uma das bactérias mais abundantes em "
            "humanos saudáveis). "
            "F. prausnitzii é o principal produtor de butirato intestinal via via "
            "de fermentação de acetato e lactato. "
            "Butirato: (1) fuel primário dos colonócitos (>70% energia); "
            "(2) inibidor de HDAC → supressão epigenética de genes pró-inflamatórios; "
            "(3) activação de receptores GPR41/GPR43 → secreção de PYY e GLP-1 → "
            "saciedade e controlo glicémico. "
            "F. prausnitzii < 2% → risco de doença inflamatória intestinal, "
            "maior permeabilidade → impacta species_gastrointestinal.butyrate_production "
            "e species_immune_microbiome.regulatory_t_cell_induction."
        ),
    )

    veillonella_spp_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Abundância relativa de Veillonella spp. (%). "
            "Referência em atletas de elite (dados Scheiman et al., Nature Medicine 2019): "
            "2-8% (significativamente superior à população sedentária). "
            "Veillonella converte lactato muscular (transportado para o lúmen intestinal) "
            "em propionato → absorvido pelo fígado → gluconeogénese e síntese de "
            "corpos cetónicos → substrato energético alternativo durante esforço prolongado. "
            "Ciclo lactato-propionato: músculo → Veillonella → propionato → fígado → "
            "glucose/cetona → músculo. "
            "Atletas com Veillonella > 5% apresentam melhor tempo em provas de "
            "resistência de 60 min+ (species_bioenergetics.lactate_recycling_efficiency). "
            "Aumenta com treino de endurance de alto volume; é um biomarcador emergente "
            "de adaptação ao treino de resistência."
        ),
    )

    lactobacillus_spp_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Abundância relativa de Lactobacillus spp. (%). "
            "Referência: 0.1-5% no cólon (maior no intestino delgado). "
            "Lactobacillus produz ácido láctico (L e D-lactato), bacteriocinas e "
            "H₂O₂ → inibição de patóbiontes. Estimula células dendríticas intestinais "
            "→ diferenciação Treg (tolerância imunológica) e produção de IgA secretora. "
            "Lactobacillus > 1% → menor incidência de URTI (upper respiratory tract "
            "infections) em atletas durante picos de carga de treino "
            "(species_immune_microbiome.mucosal_immunity_lactobacillus_support). "
            "Suplementação com L. acidophilus + L. rhamnosus GG → reduz URTI em 50% "
            "em atletas de endurance (dados de Haywood et al., 2014)."
        ),
    )

    bifidobacterium_spp_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Abundância relativa de Bifidobacterium spp. (%). "
            "Referência adulto saudável: 3-8% (declina com a idade). "
            "Bifidobacterium fermenta oligossacarídeos não-digestíveis (FOS, GOS, inulina) "
            "→ acetato e lactato → cross-feeding para produtores de butirato como "
            "F. prausnitzii. "
            "Produz vitaminas B (B1, B2, B6, B9, B12 numa fracção menor), aminoácidos "
            "essenciais, e modula a resposta imunológica mucosal via TLR2. "
            "Bifidobacterium < 1% → défice de fermentação de fibra prebiótica → "
            "redução do cross-feeding de butirato → intestino mais permeável "
            "(species_gastrointestinal.prebiotic_fermentation_capacity)."
        ),
    )

    ruminococcus_spp_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Abundância relativa de Ruminococcus spp. (%). "
            "Referência: 2-8%. "
            "Ruminococcus champanellensis e R. bromii são os principais degradadores de "
            "amido resistente e celulose → produção de butirato e acetato. "
            "Críticos para atletas com dieta alta em carbohidratos complexos "
            "(batata-doce, aveia, leguminosas). "
            "Ruminococcus > 5% associado a maior eficiência de síntese de glicogénio "
            "via butirato → estimulação de AMPK → GLUT4 → captação muscular de glicose "
            "(species_bioenergetics.glycogen_synthesis_microbiome_modifier)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # ÁCIDOS GORDOS DE CADEIA CURTA (SCFAs)
    # Interage com: species_bioenergetics, species_immune_microbiome,
    #               species_neural_cognitive (eixo intestino-cérebro)
    # ══════════════════════════════════════════════════════════════════════════

    butyrate_mmol_per_kg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Butirato fecal (ácido butírico). Unidade: mmol/kg de fezes húmidas. "
            "Referência: 5-20 mmol/kg. "
            "O butirato é o SCFA mais clinicamente relevante: "
            "(1) fuel primário colonócitos (oxidado em beta-oxidação mitocondrial); "
            "(2) inibidor de HDAC classes I/II → desacetilação de histonas → supressão "
            "NF-κB → menor expressão de IL-6, IL-1β, TNF-α; "
            "(3) activa GPR109a (receptor de niacina) em macrófagos → polarização M2; "
            "(4) cruza a BHE em pequenas quantidades → suporte de astrocitos. "
            "Butirato < 5 mmol/kg → inflamação intestinal crónica de baixo grau → "
            "impacta species_immune_microbiome.regulatory_t_cell_induction e "
            "species_gastrointestinal.colonocyte_fuel_efficiency."
        ),
    )

    propionate_mmol_per_kg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Propionato fecal (ácido propiónico). Unidade: mmol/kg de fezes húmidas. "
            "Referência: 5-25 mmol/kg. "
            "O propionato é produzido por Bacteroidetes e Veillonella spp. → "
            "absorvido pelo fígado via veia porta → substrato gluconeogénico "
            "(ciclo de Krebs via succinato) e regulador do colesterol hepático "
            "(inibição da HMG-CoA redutase). "
            "Em atletas: propionato elevado → gluconeogénese hepática mais eficiente "
            "em esforços prolongados → menor catabolismo de aminoácidos muscular "
            "(species_bioenergetics.hepatic_gluconeogenesis_scfa_modifier e "
            "species_hepatic.gluconeogenesis_substrate_availability)."
        ),
    )

    acetate_mmol_per_kg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Acetato fecal (ácido acético). Unidade: mmol/kg de fezes húmidas. "
            "Referência: 20-70 mmol/kg (é o SCFA mais abundante, ~60% do total). "
            "O acetato entra na circulação sistémica e é utilizado como: "
            "(1) substrato de acetil-CoA no ciclo de Krebs muscular e hepático; "
            "(2) precursor de corpos cetónicos nos hepatócitos; "
            "(3) sinal de saciedade via activação de GPR43 em adipócitos e GPR41 "
            "no sistema nervoso entérico. "
            "Acetato elevado correlaciona-se com maior diversidade de Firmicutes e "
            "alimenta o cross-feeding para produtores de butirato "
            "(species_bioenergetics.acetate_oxidation_rate)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PERMEABILIDADE INTESTINAL E IMUNIDADE MUCOSAL
    # Interage com: species_gastrointestinal, species_immune_microbiome
    # ══════════════════════════════════════════════════════════════════════════

    zonulin_ng_per_ml: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Zonulina sérica. Unidade: ng/mL. "
            "Referência saudável: < 45 ng/mL (método ELISA Immundiagnostik). "
            "A zonulina é uma proteína endógena (pre-haptoglobina 2) que regula a "
            "abertura das junções estreitas epiteliais (claudinas, occludina, ZO-1/2). "
            "Zonulina elevada → junções estreitas abertas → 'leaky gut' → translocação "
            "de LPS, péptidos alimentares e fragmentos bacterianos para a circulação "
            "→ endotoxemia → activação TLR4/NF-κB → inflamação sistémica crónica "
            "(species_gastrointestinal.tight_junction_integrity_threshold). "
            "Em atletas: esforços >75% VO2max por >60 min → hipertermia intestinal → "
            "zonulina transitoriamente elevada até 2h pós-exercício (fisiológico); "
            "cronicamente elevada → RED-S ou disbiose activa."
        ),
    )

    lps_eu_per_ml: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "LPS (Lipopolissacarídeo) sérico — endotoxina de gram-negativos. "
            "Unidade: EU/mL (Endotoxin Units). "
            "Referência: < 1.0 EU/mL em circulação sistémica. "
            "LPS é o principal activador do receptor TLR4/MD-2 → MyD88 → NF-κB → "
            "cascata inflamatória (IL-6, TNF-α, IL-1β, PCR). "
            "Endotoxemia metabólica crónica (LPS 2-3 EU/mL) → resistência à insulina, "
            "adipogénese visceral, aterogénese. "
            "Correlaciona-se com Proteobacteria elevada e zonulina elevada "
            "(species_immune_microbiome.lps_endotoxin_threshold e "
            "species_cardiovascular.endotoxin_atherogenic_modifier)."
        ),
    )

    siga_mg_per_dl: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "IgA secretora fecal (sIgA). Unidade: mg/dL ou mg/g de fezes. "
            "Referência fecal: 5-20 mg/dL (variabilidade alta entre métodos). "
            "A sIgA é o principal anticorpo da imunidade mucosal intestinal: "
            "neutraliza patógenos e toxinas no lúmen sem ativar complemento "
            "(resposta inflamatória mínima). Produzida por células plasmáticas na "
            "lâmina própria sob regulação de IL-10, TGF-β e células Treg. "
            "sIgA fecal baixa (<3 mg/dL) → imunidade mucosal comprometida → "
            "maior susceptibilidade a gastroenterites de treino e competição "
            "(species_immune_microbiome.mucosal_siga_threshold). "
            "sIgA salivar (colhida em repouso matinal, medida separadamente) é "
            "marcador do estado imunológico sistémico (ver athlete_psych para HRV)."
        ),
    )

    # ── Relações ─────────────────────────────────────────────────────────────
    athlete: Mapped["AthleteCore"] = relationship(
        back_populates="microbiome_assessments"
    )
