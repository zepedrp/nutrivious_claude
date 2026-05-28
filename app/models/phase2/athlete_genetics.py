from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Float, String, func
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase2.athlete_core import AthleteCore


class AthleteGenetics(Base):
    """
    Perfil genómico permanente do atleta — Eixo 1 / Módulo A.

    Relação 1:1 com AthleteCore (unique=True em athlete_id).
    O genótipo é imutável; a data de sequenciação regista a versão do painel.
    Cada SNP modula ou capeia os tectos absolutos da Fase 1 (species_*).
    """

    __tablename__ = "athlete_genetics"
    __table_args__ = (
        UniqueConstraint("athlete_id", name="uq_athlete_genetics_athlete"),
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

    sequencing_date: Mapped[date] = mapped_column(
        Date,
        nullable=True,
        comment=(
            "Data de recolha da amostra / emissão do relatório de sequenciação. "
            "Permite versionar o painel genómico caso o atleta repita o teste com "
            "tecnologia mais abrangente (WGS vs SNP array vs painéis direcionados)."
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 1 — COMPOSIÇÃO MUSCULAR E POTÊNCIA MECÂNICA
    # Interage com: species_neuromuscular, species_musculoskeletal
    # ══════════════════════════════════════════════════════════════════════════

    actn3_r577x: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "ACTN3 R577X (rs1815739). Genótipos: 'RR' | 'RX' | 'XX'. "
            "Alpha-actinina-3, proteína estrutural exclusiva das fibras de contracção "
            "rápida (Tipo IIx). "
            "RR → expressão máxima de ACTN3 → predomínio Tipo IIx → tecto de força "
            "e potência elevado (species_neuromuscular.fast_twitch_fibre_ceiling). "
            "RX → heterozigoto; capacidade mista. "
            "XX (stop-codon; ~18% população europeia) → ausência de ACTN3 → shift "
            "para Tipo I → maior economia metabólica mas tecto de potência absoluta "
            "reduzido. Modifica Gradiente_adaptação(força) da Fase 2: "
            "XX raramente excedem o percentil 80 em provas explosivas puras."
        ),
    )

    myh7_variant: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment=(
            "MYH7 (cadeia pesada de miosina beta — Tipo I). Variantes raras notáveis: "
            "'WT' (wild-type) | 'Arg403Gln' | 'Glu848Gly' | outro código HGVS. "
            "MYH7 determina a cinética de ciclagem das pontes cruzadas nas fibras lentas. "
            "Mutações de ganho-de-função aumentam força isométrica mas reduzem a "
            "velocidade de encurtamento → impacto em species_neuromuscular.isometric_force_max. "
            "Clinicamente relevante: variantes patogénicas associadas a miocardiopatia "
            "hipertrófica (HCM) — se detectadas deve ser activado o protocolo cardíaco."
        ),
    )

    mstn_variant: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment=(
            "MSTN — Miostatina (GDF-8). Variantes: 'WT' | 'K153R' | 'E164K' | outro. "
            "A miostatina é o principal inibidor do crescimento muscular; actua sobre "
            "Akt/mTOR e inibe a proliferação de células satélite. "
            "Variantes LOF (loss-of-function) ou hipomórficas → desinibição de mTORC1 "
            "→ tecto de hipertrofia elevado (species_musculoskeletal.lean_mass_ceiling). "
            "Interacção directa com IGF1 e espera-se sinergismo em K153R + IGF1-192."
        ),
    )

    igf1_promoter_192: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "IGF1 promotor — polimorfismo CA-repeat (rs35767 proxy: '192/192' | "
            "'192/non192' | 'non192/non192'). "
            "O alelo 192-bp associa-se a níveis basais de IGF-1 sérico ~20% superiores "
            "e maior resposta anabólica ao treino de resistência. "
            "Modifica species_endocrine.igf1_baseline e amplia o tecto de resposta de "
            "síntese proteica pós-treino. Importante para calibrar a janela anabólica "
            "na prescrição nutricional (proteína/leucina)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 2 — CAPACIDADE AERÓBIA E MITOCONDRIAL
    # Interage com: species_bioenergetics, species_mitochondrial,
    #               species_cardiovascular, species_pulmonary
    # ══════════════════════════════════════════════════════════════════════════

    ace_indel: Mapped[str | None] = mapped_column(
        String(4),
        nullable=True,
        comment=(
            "ACE I/D (rs4646994). Genótipos: 'II' | 'ID' | 'DD'. "
            "Polimorfismo de inserção/deleção de 287 pb no intrão 16 do gene da "
            "Enzima Conversora de Angiotensina. "
            "DD → actividade ACE elevada → maior Angiotensina II → vasoconstrição "
            "eficiente, hipertrofia cardíaca favorável para esforços curtos/intensos. "
            "II → actividade ACE reduzida → vasodilatação periférica melhorada → "
            "superior eficiência em provas longas (correlação com VO2max elite em "
            "endurance; species_cardiovascular.vo2max_ceiling). "
            "ID → fenotípico intermédio; maior plasticidade de adaptação."
        ),
    )

    ppargc1a_gly482ser: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "PPARGC1A Gly482Ser (rs8192678). Genótipos: 'GG' | 'GA' | 'AA'. "
            "PGC-1α é o co-activador transcricional mestre da biogénese mitocondrial. "
            "GG (Gly/Gly) → PGC-1α plenamente funcional → maior densidade mitocondrial "
            "induzível por treino → tecto de VO2max e MLSS mais elevado "
            "(species_mitochondrial.mitochondrial_density_ceiling). "
            "AA (Ser/Ser) → eficiência reduzida de biogénese → resposta ao treino "
            "aeróbio atenuada; estes atletas precisam de volumes maiores para atingir "
            "a mesma adaptação mitocondrial. "
            "Interage sinergicamente com ACE II: II+GG → perfil endurance de elite."
        ),
    )

    vegf_rs2010963: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "VEGF -634G>C (rs2010963). Genótipos: 'GG' | 'GC' | 'CC'. "
            "VEGF (Vascular Endothelial Growth Factor) regula a angiogénese induzida "
            "por exercício — capilarização muscular e densidade de capilares/fibra. "
            "GG → maior expressão de VEGF → superior capilarização pós-treino → "
            "melhor extracção periférica de O₂ (a(v-a)O₂ diff) → amplifica o impacto "
            "do VO2max em potência aeróbia sustentada "
            "(species_cardiovascular.capillary_density_ceiling)."
        ),
    )

    hif1a_pro582ser: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "HIF1A Pro582Ser (rs11549465). Genótipos: 'CC' (Pro/Pro) | 'CT' | 'TT' (Ser/Ser). "
            "HIF-1α é o factor de transcrição central da resposta hipóxica. "
            "Ser582 → degradação proteossómica de HIF-1α mais lenta em normóxia → "
            "activação constitutiva parcial da via hipóxica → maior EPO endógena, "
            "maior densidade mitocondrial basal, melhor tolerância a altitude. "
            "Modifica species_hematological.epo_sensitivity e "
            "species_mitochondrial.hypoxia_tolerance_ceiling."
        ),
    )

    ampd1_q12x: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "AMPD1 Q12X (rs17602729). Genótipos: 'CC' | 'CA' | 'AA'. "
            "AMP Deaminase 1 — enzima que converte AMP → IMP no ciclo dos nucleótidos "
            "de purina em músculo. "
            "AA (homozigoto stop-codon; ~2% população) → deficiência de AMPD1 → "
            "acumulação de AMP → maior sinalização AMPK → melhor economia metabólica "
            "em esforços prolongados mas risco de cãibras e fadiga prematura em "
            "esforços anaeróbios. "
            "Interage com species_bioenergetics.atp_resynthesis_rate."
        ),
    )

    mtdna_haplogroup: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "Haplogrupo do ADN mitocondrial matrilinear. Exemplos: 'H' | 'J' | 'T' | "
            "'U' | 'K' | 'L0' | etc. "
            "Haplogrupos J e T associados a maior actividade do complexo I mitocondrial "
            "e termogénese desacoplada (UCPs) → vantagem em endurance de altitude e "
            "climas frios; ligeira redução de eficiência calórica. "
            "Haplogrupo H (mais comum na Europa) → maior eficiência de acoplamento "
            "oxidativo → superior rendimento mecânico por mol de O₂. "
            "Modifica species_mitochondrial.oxidative_coupling_efficiency."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 3 — TECIDO CONJUNTIVO E RISCO DE LESÃO
    # Interage com: species_musculoskeletal, species_osseous_system
    # ══════════════════════════════════════════════════════════════════════════

    col5a1_rs12722: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "COL5A1 3'UTR BstUI RFLP (rs12722). Genótipos: 'CC' | 'CT' | 'TT'. "
            "Colagénio tipo V — componente estrutural dos tendões e ligamentos; "
            "regula o diâmetro das fibrilas de colagénio tipo I. "
            "TT → maior rigidez tendinosa → melhor transmissão de força e menor risco "
            "de tendinopatia patellar/Aquiles. "
            "CC → fibrilas mais finas → maior laxidez ligamentar → risco aumentado de "
            "entorses e rotura do LCA (species_musculoskeletal.connective_tissue_integrity). "
            "Essencial para estratificar o volume de treino pliométrico e a progressão "
            "de cargas excêntricas."
        ),
    )

    col1a1_sp1: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "COL1A1 Sp1 binding site (rs1800012). Genótipos: 'GG' | 'GT' | 'TT'. "
            "Colagénio tipo I — principal proteína estrutural do osso e tendão. "
            "TT → menor expressão de COL1A1 → densidade mineral óssea reduzida e "
            "maior fragilidade tendinosa → risco de fractura de stress "
            "(species_osseous_system.bmd_ceiling). "
            "GG → arquitectura óssea e tendinosa óptima. "
            "Em atletas TT, a suplementação de colágeno + vit C tem maior impacto "
            "relativo na síntese de colagénio tendinoso."
        ),
    )

    mmp3_rs679620: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "MMP3 5A/6A (rs679620 proxy). Genótipos: '5A5A' | '5A6A' | '6A6A'. "
            "Metaloproteinase-3 — enzima de remodelação da matriz extracelular do tendão. "
            "5A5A → maior actividade MMP3 → degradação mais rápida de colagénio "
            "danificado → maior taxa de remodelação tendinosa pós-treino excêntrico; "
            "mas também maior susceptibilidade à tendinopatia crónica se recuperação "
            "insuficiente. "
            "Modifica o período mínimo de recuperação entre sessões de carga tendinosa "
            "máxima (species_musculoskeletal.tendon_remodeling_rate)."
        ),
    )

    gdf5_rs143384: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "GDF5 (Growth Differentiation Factor 5) rs143384. Genótipos: 'AA' | 'AG' | 'GG'. "
            "GDF5 regula a diferenciação condrogénica e a morfogénese articular. "
            "AA → expressão reduzida → risco aumentado de osteoartrite, em particular "
            "do joelho; menor capacidade regenerativa da cartilagem articular. "
            "GG → expressão plena → articulações mais resilientes. "
            "Informa a prescrição do volume de impacto acumulado semanal e a "
            "monitorização de biomarcadores de cartilagem (COMP, s-GAG)."
        ),
    )

    tnxb_rs1061496: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "TNXB (Tenascina-X) rs1061496. Genótipos: 'AA' | 'AG' | 'GG'. "
            "Tenascina-X é uma glicoproteína da matriz extracelular que regula a "
            "organização das fibrilas de colagénio e a viscoelasticidade dos tecidos moles. "
            "Variantes LOF associadas à síndrome de Ehlers-Danlos hipermóvel (hEDS). "
            "Alelo A → hiperlaxidez ligamentar, instabilidade articular, maior risco "
            "de entorses recorrentes. "
            "Crucial para definir o programa de estabilização neuromuscular e os limites "
            "de amplitude em exercícios de mobilidade (species_musculoskeletal.joint_laxity)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 4 — METABOLISMO NUTRICIONAL E FARMACOGENÓMICA
    # Interage com: species_lipid_metabolism, species_protein_metabolism,
    #               species_gastrointestinal, species_hepatic, species_endocrine
    # ══════════════════════════════════════════════════════════════════════════

    mthfr_c677t: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "MTHFR C677T (rs1801133). Genótipos: 'CC' | 'CT' | 'TT'. "
            "5,10-metilenotetra-hidrofolato redutase — enzima central no ciclo do folato "
            "e na remetilação da homocisteína em metionina. "
            "TT → actividade enzimática reduzida ~70% → risco de hiper-homocisteinémia "
            "→ stress oxidativo vascular e disfunção mitocondrial. "
            "Impacta species_oxidative_stress.homocysteine_ceiling e "
            "species_epigenetic.sam_sah_ratio (disponibilidade de SAM para metilação do ADN). "
            "Prescrição: 5-MTHF (metilfolato activo) em vez de ácido fólico sintético; "
            "metilcobalamina B12; monitorização de homocisteína sérica (alvo <8 μmol/L)."
        ),
    )

    mthfr_a1298c: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "MTHFR A1298C (rs1801131). Genótipos: 'AA' | 'AC' | 'CC'. "
            "Segundo polimorfismo funcional do MTHFR; afecta a conversão de "
            "5-MTHF em THF (tetrahidrofolato) e a síntese de BH4 (tetrahidrobiopterina). "
            "CC → redução na biossíntese de BH4 → menor síntese de neurotransmissores "
            "dopaminérgicos e serotoninérgicos → impacta species_neural_cognitive.neurotransmitter_baseline. "
            "Composto heterozigoto C677T/A1298C → efeito cumulativo significativo; "
            "suplementação prioritária de 5-MTHF + B6 (P5P) + B12 (metil)."
        ),
    )

    comt_val158met: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "COMT Val158Met (rs4680). Genótipos: 'GG' (Val/Val) | 'GA' | 'AA' (Met/Met). "
            "Catecol-O-Metiltransferase — enzima que degrada dopamina, adrenalina e "
            "noradrenalina no córtex pré-frontal. "
            "AA (Met/Met) → actividade COMT ~40% inferior → maior disponibilidade de "
            "dopamina pré-frontal → melhor working memory e tomada de decisão sob carga "
            "cognitiva moderada (species_neural_cognitive.prefrontal_dopamine_clearance). "
            "GG (Val/Val) → clearance rápida → maior resiliência em stress agudo, "
            "'guerreiro' vs 'preocupado': melhor performance sob pressão competitiva extrema. "
            "AA sob stress intenso → hiperdopaminergia prefrontal → ansiedade e "
            "degradação de performance (paradoxo). "
            "Informa estratégias de gestão de arousal e uso de adaptógenos."
        ),
    )

    cyp1a2_rs762551: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "CYP1A2 -163C>A (rs762551). Genótipos: 'AA' | 'AC' | 'CC'. "
            "CYP1A2 é a isoenzima hepática responsável por ~95% do metabolismo da cafeína "
            "(1,3,7-trimetilxantina → paraxantina). "
            "AA → metabolizador rápido → t½ cafeína ~3-4h → ergogenic effect limpo, "
            "janela óptima: 60 min pré-treino, sem interferência no sono se tomada "
            "antes das 14h (species_chronobiology.caffeine_half_life = curto). "
            "CC → metabolizador lento → t½ cafeína ~7-12h → risco de acumulação, "
            "perturbação do sono, e — paradoxalmente — aumento do risco cardiovascular "
            "em doses >200 mg/d. Prescrição: dose máxima 100 mg em metabolizadores lentos."
        ),
    )

    vdr_foki: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "VDR FokI (rs2228570). Genótipos: 'FF' | 'Ff' | 'ff'. "
            "Polimorfismo no codão de iniciação do Receptor da Vitamina D. "
            "ff → proteína VDR 3 aa mais longa → menor eficiência de transactivação "
            "→ menor sensibilidade à vitamina D → necessidade de níveis séricos de "
            "25(OH)D mais elevados para atingir a mesma resposta imunológica, óssea "
            "e muscular (species_osseous_system.vitamin_d_sensitivity e "
            "species_immune_microbiome.vitamin_d_receptor_efficiency). "
            "ff requer 25(OH)D sérico alvo ≥50 ng/mL vs FF onde 40 ng/mL é suficiente."
        ),
    )

    fads1_rs174537: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "FADS1 rs174537. Genótipos: 'GG' | 'GT' | 'TT'. "
            "FADS1 (Delta-5-desaturase) e FADS2 (Delta-6-desaturase) controlam a "
            "conversão de ALA → EPA → DHA (ácidos gordos ómega-3 de cadeia longa). "
            "TT → actividade desaturase reduzida → conversão endógena ALA→DHA muito "
            "ineficiente (<5%) → dependência absoluta de EPA/DHA pré-formado "
            "(peixe gordo, suplemento de óleo de algas/peixe). "
            "Modifica species_lipid_metabolism.omega3_conversion_efficiency. "
            "Impacto directo na resolução de inflamação pós-treino e na fluidez "
            "das membranas mitocondriais (species_mitochondrial.membrane_composition)."
        ),
    )

    hfe_c282y: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "HFE C282Y (rs1800562). Genótipos: 'GG' (WT) | 'GA' | 'AA'. "
            "HFE — gene da hemocromatose hereditária. "
            "AA (homozigoto) → absorção intestinal de ferro desregulada → acumulação "
            "progressiva de ferro → hemossiderose hepática, cardíaca, articular. "
            "Em atletas de endurance: GA heterozigoto pode conferir vantagem de "
            "absorção de ferro e menor risco de anemia ferropénica, mas requer "
            "monitorização de ferritina (alvo 80-150 ng/mL; species_hematological.iron_stores_ceiling). "
            "AA exige restrição de suplementação de ferro e flebotomia terapêutica "
            "se ferritina >300 ng/mL."
        ),
    )

    hfe_h63d: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "HFE H63D (rs1799945). Genótipos: 'CC' (WT) | 'CG' | 'GG'. "
            "Segundo polimorfismo funcional do HFE; efeito mais moderado que C282Y. "
            "Composto heterozigoto C282Y/H63D → risco intermédio de sobrecarga de ferro. "
            "GG homozigoto → absorção de ferro aumentada em ~20-30%; monitorização "
            "trimestral de ferritina e saturação da transferrina em atletas de "
            "endurance de alto volume (>12h/sem)."
        ),
    )

    lct_rs4988235: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "LCT -13910C>T (rs4988235). Genótipos: 'CC' | 'CT' | 'TT'. "
            "Persistência da lactase — capacidade de digerir lactose na idade adulta. "
            "CC → não-persistência (hipolactasia do adulto) → intolerância à lactose → "
            "sintomas GI em contexto de ingestão peri-treino de whey/caseína; "
            "impacta species_gastrointestinal.lactase_persistence. "
            "TT → persistência plena → produtos lácteos bem tolerados → fonte proteica "
            "e de cálcio sem restrição. "
            "CT → persistência parcial; tolerância variável (limiar ~12 g lactose/refeição)."
        ),
    )

    apoe_genotype: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "ApoE genótipo combinado. Valores: 'E2E2' | 'E2E3' | 'E2E4' | "
            "'E3E3' | 'E3E4' | 'E4E4'. "
            "Apolipoproteína E — ligante de lipoproteínas; determina a clearance "
            "de lipoproteínas ricas em triglicéridos e LDL-C. "
            "E4 → clearance de LDL reduzida → LDL-C e ApoB mais elevados em dietas "
            "ricas em gordura saturada → risco cardiovascular elevado "
            "(species_lipid_metabolism.ldl_clearance_rate e "
            "species_cardiovascular.atherosclerosis_risk_modifier). "
            "E2 → clearance aumentada → LDL-C mais baixo; mas em E2E2 risco de "
            "hiperlipoproteinémia tipo III. "
            "E4E4 → requer dieta muito baixa em gordura saturada e monitorização "
            "lipídica semestral."
        ),
    )

    bcmo1_rs7501331: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "BCMO1 (BCO1) rs7501331. Genótipos: 'CC' | 'CA' | 'AA'. "
            "Beta-caroteno-15,15'-mono-oxigenase — enzima intestinal que converte "
            "β-caroteno (provitamina A) em retinal (vitamina A activa). "
            "AA → actividade enzimática reduzida ~69% → conversão insuficiente de "
            "β-caroteno em retinol → dependência de vitamina A pré-formada (fígado, "
            "óvos, suplemento de retinol) em dietas plant-based "
            "(species_gastrointestinal.carotenoid_conversion_efficiency). "
            "Relevante para função imunológica e integridade epitelial intestinal "
            "(species_immune_microbiome.mucosal_barrier_integrity)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 5 — CRONOBIOLOGIA GENÉTICA
    # Interage com: species_chronobiology
    # ══════════════════════════════════════════════════════════════════════════

    clock_rs1801260: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "CLOCK 3111T>C (rs1801260). Genótipos: 'TT' | 'TC' | 'CC'. "
            "CLOCK é a subunidade do heterodímero CLOCK:BMAL1 que activa a transcrição "
            "dos genes do relógio circadiano (Per, Cry, Rev-erb). "
            "CC → período circadiano endógeno mais longo (~24.5h) → tendência vespertina "
            "(cronotipo tardio, 'coruja'); atraso de fase de temperatura corporal e "
            "cortisol. Performance pico deslocada para tarde (16-20h). "
            "TT → período mais curto → cronotipo matutino; pico de performance ~9-12h. "
            "Informa o agendamento óptimo de treinos e competições "
            "(species_chronobiology.circadian_period e phase_shift_sensitivity)."
        ),
    )

    per2_rs2304672: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "PER2 rs2304672. Genótipos: 'CC' | 'CG' | 'GG'. "
            "Period-2 — gene efector da ansa negativa do relógio circadiano; "
            "regula a supressão de CLOCK:BMAL1 com período de ~24h. "
            "Variantes em PER2 associadas a FASPS (Familial Advanced Sleep Phase Syndrome) "
            "→ cronotipo extremamente matutino; sono iniciado 19-21h, despertar 3-5h AM. "
            "Em atletas: janela de treino óptima 06-10h; vigilância para overtreino "
            "vespertino que pode deslocar a fase circadiana."
        ),
    )

    per3_vntr: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "PER3 VNTR (rs57875989). Genótipos: '4/4' | '4/5' | '5/5'. "
            "Variable Number Tandem Repeat de 54 pb no exão 18 do gene PER3. "
            "5/5 → maior sensibilidade à privação de sono → declínio mais acentuado "
            "em vigilância psicomotora (PVT) após noite de privação parcial (<6h). "
            "Também associado a maior supressão de melatonina pela luz azul nocturna. "
            "(species_chronobiology.sleep_pressure_sensitivity). "
            "5/5 requer higiene de sono rigorosa: extinção de luz azul 2h antes do "
            "sono, blackout total, temperatura quarto 18-19°C."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 6 — NEUROPSICOLOGIA E COMPORTAMENTO
    # Interage com: species_neural_cognitive, species_endocrine
    # ══════════════════════════════════════════════════════════════════════════

    slc6a4_5httlpr: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "SLC6A4 5-HTTLPR (transportador de serotonina). "
            "Genótipos: 'LL' | 'LS' | 'SS' (alelos L=longo, S=curto). "
            "5-HTTLPR regula a expressão do transportador de recaptação de serotonina "
            "(SERT) no terminal pré-sináptico. "
            "SS → expressão SERT reduzida → maior disponibilidade sináptica de 5-HT "
            "basal mas menor tampão em situações de stress → hiperreactividade à "
            "adversidade; maior prevalência de ansiedade de pré-competição e PTSD "
            "pós-lesão grave (species_neural_cognitive.serotonin_reuptake_efficiency). "
            "LL → recaptação eficiente → maior homeostasia serotoninérgica; melhor "
            "resiliência a stressores crónicos (overtreino, viagens, jet lag)."
        ),
    )

    bdnf_val66met: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "BDNF Val66Met (rs6265). Genótipos: 'GG' (Val/Val) | 'GA' | 'AA' (Met/Met). "
            "Brain-Derived Neurotrophic Factor — neurotrofina essencial para "
            "neuroplasticidade, consolidação de memória motora e resposta ao exercício. "
            "GG → secreção actividade-dependente de BDNF plena → maior neuroplasticidade "
            "induzida por exercício → aprendizagem técnica mais rápida "
            "(species_neural_cognitive.bdnf_exercise_response). "
            "AA → secreção reduzida ~30% → menor resposta neurotrófica ao exercício aeróbio; "
            "risco aumentado de depressão pós-lesão prolongada. "
            "Exercício aeróbio (30 min, ≥65% VO2max) é o estímulo mais potente de BDNF "
            "para AA; prescrição deve incluir componente aeróbio diário mínimo."
        ),
    )

    oprm1_a118g: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "OPRM1 A118G (rs1799971). Genótipos: 'AA' | 'AG' | 'GG'. "
            "Receptor opióide mu-1 — principal receptor de β-endorfina e opioides "
            "endógenos; central na modulação da dor e no 'runner's high'. "
            "GG → receptor com maior afinidade para β-endorfina (3x) → maior analgesia "
            "endógena → tolerância à dor muscular mais elevada em treinos de alta "
            "intensidade; maior euforia pós-exercício (species_neural_cognitive.pain_threshold). "
            "AA → resposta opióide basal; sem alteração. "
            "Relevante para gestão de analgésicos em pós-cirúrgico (resposta reduzida "
            "a opioides exógenos em GG)."
        ),
    )

    drd2_taq1a: Mapped[str | None] = mapped_column(
        String(8),
        nullable=True,
        comment=(
            "DRD2 TaqIA (rs1800497). Genótipos: 'A1A1' | 'A1A2' | 'A2A2'. "
            "Polimorfismo no gene ANKK1 (flanco 3' de DRD2) que reduz a densidade "
            "de receptores D2 no estriado. "
            "A1A1 → ~30-40% menos receptores D2 → hipodopaminergia estriatal → "
            "menor sensação de recompensa intrínseca, maior drive para busca de "
            "novidade e estimulação externa → risco de comportamentos de risco e "
            "menor adesão a protocolos repetitivos de treino de base. "
            "(species_neural_cognitive.reward_circuit_sensitivity). "
            "A2A2 → densidade D2 normal → maior satisfação com progressão gradual; "
            "melhor tolerância a treinos de alto volume monótono (endurance base)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SCORES POLIGÉNICOS COMPOSTOS
    # Índices derivados — calculados externamente, armazenados como Float
    # ══════════════════════════════════════════════════════════════════════════

    polygenic_endurance_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Score poligénico de capacidade aeróbia/endurance. Escala 0.0-1.0. "
            "Calculado externamente combinando: ACE (II=+2, ID=+1, DD=0), "
            "PPARGC1A (GG=+2, GA=+1, AA=0), HIF1A (CT/TT=+1), VEGF (GG=+1), "
            "mtDNA haplogroup (J/T=+1). "
            "Score ≥ 0.7 → perfil genético favorável para endurance de elite. "
            "Modifica o tecto individual de VO2max (species_cardiovascular.vo2max_ceiling) "
            "como fracção do tecto da espécie: VO2max_tecto_individual = "
            "VO2max_espécie × (0.8 + 0.2 × polygenic_endurance_score)."
        ),
    )

    polygenic_power_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Score poligénico de potência/força explosiva. Escala 0.0-1.0. "
            "Calculado externamente combinando: ACTN3 (RR=+2, RX=+1, XX=0), "
            "ACE (DD=+2, ID=+1, II=0), IGF1-192 (192/192=+2), MSTN LOF (=+2), "
            "AMPD1 (CC=+1). "
            "Score ≥ 0.7 → perfil genético favorável para desportos de potência/força. "
            "Modifica species_neuromuscular.peak_power_ceiling como multiplicador."
        ),
    )

    polygenic_injury_risk_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Score poligénico de risco de lesão músculo-esquelética. Escala 0.0-1.0 "
            "(0 = menor risco, 1 = maior risco). "
            "Calculado combinando: COL5A1 (CC=+2, CT=+1, TT=0), COL1A1 (TT=+2, GT=+1), "
            "GDF5 (AA=+2, AG=+1), TNXB A (=+1), MMP3 5A5A (=+1). "
            "Score ≥ 0.6 → protocolo de prevenção de lesão obrigatório: excêntrico "
            "progressivo, colágeno + vit C 1h pré-treino, ultrassom tendinoso semestral. "
            "Informa o volume máximo de impacto semanal (species_musculoskeletal.injury_risk_threshold)."
        ),
    )

    # ── Relações ─────────────────────────────────────────────────────────────
    athlete: Mapped["AthleteCore"] = relationship(
        back_populates="genetics_profile"
    )
