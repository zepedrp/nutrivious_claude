from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.phase1.core import Base

if TYPE_CHECKING:
    from app.models.phase2.athlete_core import AthleteCore


class AthleteBiomarkers(Base):
    """
    Passaporte Clínico Basal — Eixo 4/5/7 / Módulos C e D.

    Relação 1:many com AthleteCore (blood_draw_date permite série temporal longitudinal).
    Cada coluna representa um biomarcador sérico que calibra os tectos da Fase 1:
    o genótipo (Fase 2 Cluster 2) define os limites estruturais;
    os biomarcadores definem o estado funcional real naquele momento.
    """

    __tablename__ = "athlete_biomarkers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    athlete_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("athlete_core.id", ondelete="CASCADE"),
        nullable=False,
    )

    blood_draw_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment=(
            "Data e hora exacta da colheita de sangue (com timezone). "
            "Crítico para o painel hormonal: cortisol_am exige colheita 07:00-09:00h "
            "(pico CAR — Cortisol Awakening Response). "
            "LH/FSH devem ser colhidos em jejum e em repouso (>24h sem exercício intenso). "
            "Testosterone total é mais estável mas idealmente matinal. "
            "Permite construção de série temporal: baseline pré-época → mid-season → "
            "pós-época, rastreando o vetor de adaptação ao longo do tempo."
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 1 — HORMONAL (EIXO HPG / HPA)
    # Interage com: species_endocrine
    # Eixo HPG: Hipotálamo → Pituitária → Gónadas (GnRH → LH/FSH → T/E2)
    # Eixo HPA: Hipotálamo → Pituitária → Adrenal (CRH → ACTH → Cortisol/DHEA)
    # ══════════════════════════════════════════════════════════════════════════

    testosterone_total: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Testosterona total sérica. Unidade: nmol/L (ou ng/dL; 1 ng/dL = 0.0347 nmol/L). "
            "Referência H: 10.4-34.7 nmol/L (300-1000 ng/dL); F: 0.5-2.6 nmol/L. "
            "A testosterona é o principal androgénio anabólico; activa o receptor AR → "
            "transcrição de MHC (miosina), IGF-1 local, e inibição da miostatina. "
            "Modifica directamente species_endocrine.testosterone_baseline: "
            "CPI_T = testosterone_total_atleta / testosterone_tecto_espécie. "
            "Valores <8 nmol/L em H → défice androgénico funcional → redução do "
            "estímulo anabólico pós-treino em ~30-40%; sinaliza overtreino ou RED-S. "
            "Valores >34.7 nmol/L sem explicação fisiológica → protocolo anti-doping."
        ),
    )

    testosterone_free: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Testosterona livre (fracção biologicamente activa). Unidade: pmol/L ou pg/mL. "
            "Referência H: 174-729 pmol/L; F: 3.5-43 pmol/L. "
            "Apenas ~2% da testosterona total circula livre; o restante liga-se a SHBG "
            "(alta afinidade) e albumina (baixa afinidade). "
            "testosterone_free é o modulador real da síntese proteica muscular — "
            "um atleta com testosterone_total normal mas SHBG elevada pode ter "
            "testosterone_free clinicamente baixa. "
            "Calculada por: T_free = T_total × (1 − (SHBG × Ka) / (1 + Ka × SHBG)), "
            "ou medida directamente por diálise em equilíbrio (método gold-standard). "
            "Interacção com species_endocrine.free_androgen_index."
        ),
    )

    shbg: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Sex Hormone Binding Globulin. Unidade: nmol/L. "
            "Referência H: 16-55 nmol/L; F (pré-menopausa): 25-100 nmol/L. "
            "A SHBG liga testosterona e estradiol com alta afinidade, regulando a "
            "biodisponibilidade hormonal. "
            "SHBG elevada (>70 nmol/L em H) → sequestro de testosterona → redução "
            "efectiva do sinal anabólico mesmo com testosterone_total normal. "
            "Causas: hipotiroidismo, anorexia, RED-S, estatinas. "
            "SHBG reduzida (<10 nmol/L em H) → excesso de androgénios livres; "
            "associada a resistência à insulina (HOMA-IR elevado). "
            "Modifica o denominador do cálculo de testosterone_free "
            "(species_endocrine.shbg_binding_capacity)."
        ),
    )

    estradiol: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Estradiol (E2, 17β-estradiol). Unidade: pmol/L (ou pg/mL; 1 pg/mL = 3.67 pmol/L). "
            "Referência H: 40-160 pmol/L; F (fase folicular): 150-750 pmol/L; "
            "F (pico ovulatório): até 1800 pmol/L. "
            "O estradiol tem papel anabólico e neuroprotetor: estimula IGF-1 hepático, "
            "protege a DMO (receptor ERα em osteoblastos), e modula a resposta "
            "inflamatória pós-exercício (species_osseous_system.bmd_estrogen_modifier). "
            "Em atletas femininas: E2 <100 pmol/L em fase folicular → Tríade (RED-S) → "
            "risco imediato de fractura de stress e amenorreia hipotalâmica. "
            "Em H: E2 >200 pmol/L → aromatização excessiva (excesso de gordura visceral "
            "→ aromatase adiposa) → supressão do eixo HPG via feedback negativo."
        ),
    )

    cortisol_am: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Cortisol matinal sérico (07:00-09:00h). Unidade: nmol/L (ou μg/dL; 1 μg/dL = 27.6 nmol/L). "
            "Referência: 138-690 nmol/L (5-25 μg/dL) em colheita matinal. "
            "O cortisol é o output final do eixo HPA: CRH → ACTH → cortisol adrenal. "
            "O pico matinal (CAR) representa a integridade do ritmo circadiano e a "
            "reserva adrenocortical (species_chronobiology.cortisol_awakening_response). "
            "CAR elevado (>690 nmol/L) → stress agudo, ansiedade de pré-competição, "
            "excesso de treino → estado catabólico: activação de ubiquitina-proteossoma "
            "muscular, gluconeogénese a partir de aminoácidos. "
            "CAR baixo (<138 nmol/L) → fadiga adrenal / overtreino crónico → perda de "
            "variabilidade circadiana → imunossupressão. "
            "Rácio testosterone/cortisol > 0.35 nmol/nmol → estado anabólico; <0.20 → "
            "estado catabólico de risco (species_endocrine.anabolic_catabolic_ratio)."
        ),
    )

    dhea_s: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "DHEA-S (Deidroepiandrosterona-Sulfato). Unidade: μmol/L (ou μg/dL). "
            "Referência H 20-30 anos: 5.7-13.4 μmol/L; F 20-30 anos: 2.8-11.0 μmol/L. "
            "O DHEA-S é o androgénio adrenal mais abundante; precursor de testosterona "
            "e estradiol periféricos (conversão em tecidos alvos via aromatase e 17β-HSD). "
            "Declínio fisiológico com a idade (~2%/ano após os 25 anos) — marcador de "
            "adrenopausa e reserva funcional adrenal "
            "(species_endocrine.dheas_age_trajectory). "
            "Em atletas: DHEA-S <3.5 μmol/L em H jovem → deficit adrenal funcional → "
            "comprometimento da resposta imunitária inata e da libido. "
            "Modifica a fracção de androgénios biodisponíveis para além do eixo HPG."
        ),
    )

    lh: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "LH — Hormona Luteinizante. Unidade: UI/L. "
            "Referência H (fase estável): 1.7-8.6 UI/L; F (fase folicular): 2.4-12.6 UI/L. "
            "O LH é secretado em pulsos pela hipófise anterior em resposta ao GnRH; "
            "em H estimula as células de Leydig → produção de testosterona. "
            "LH baixo com testosterone_total baixa → hipogonadismo hipogonadotrópico "
            "funcional (overtreino, RED-S, stress crónico) — o problema está no eixo "
            "central, não no testículo. "
            "LH elevado com testosterone_total baixa → falência testicular primária "
            "(espécie rara em atletas jovens). "
            "Diagnóstico diferencial essencial para estratégia de intervenção "
            "(species_endocrine.hpg_axis_integrity)."
        ),
    )

    fsh: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "FSH — Hormona Folículo-Estimulante. Unidade: UI/L. "
            "Referência H: 1.5-12.4 UI/L; F (fase folicular): 3.5-12.5 UI/L. "
            "Em H, o FSH estimula as células de Sertoli e a espermatogénese. "
            "Em F, o FSH regula o desenvolvimento folicular ovariano e a secreção "
            "de estradiol. "
            "FSH elevado isoladamente → reserva ovárica reduzida em F (marcador de "
            "envelhecimento reprodutivo); em H → disfunção tubular testicular. "
            "Rácio LH/FSH: em F, <1 na fase folicular precoce é normal; >2 sugere "
            "SOP (síndrome do ovário poliquístico). "
            "Combinado com LH e estradiol/testosterone → diagnóstico completo do "
            "eixo HPG (species_endocrine.hpg_axis_integrity)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 2 — HEMATOLÓGICO E METABOLISMO DO FERRO
    # Interage com: species_hematological, species_cardiovascular,
    #               species_renal_excretory
    # ══════════════════════════════════════════════════════════════════════════

    hemoglobin: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Hemoglobina sérica. Unidade: g/dL. "
            "Referência H: 13.5-17.5 g/dL; F: 12.0-16.0 g/dL. "
            "A hemoglobina é o transportador de O₂ no sangue; cada grama transporta "
            "1.34 mL O₂. A capacidade total de transporte de O₂ = Hb × 1.34 × SaO₂. "
            "VO2max = CaO₂ × DC = (Hb × 1.34 × SaO₂) × (FC × VS). "
            "Hb <13.0 H / <12.0 F → anemia → redução directa do tecto de VO2max "
            "(cada 1 g/dL de Hb perdida ≈ 3-4 mL/kg/min de VO2max). "
            "Modifica species_hematological.hemoglobin_reference como denominador do "
            "CPI hematológico: CPI_Hb = Hb_atleta / Hb_tecto_espécie."
        ),
    )

    hematocrit: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Hematócrito (fracção de volume de eritrócitos). Unidade: % ou fracção decimal. "
            "Referência H: 42-52%; F: 37-47%. "
            "O hematócrito determina a viscosidade sanguínea e a resistência vascular: "
            "Hct >55% → hiperviscosidade → maior risco trombótico e menor débito "
            "cardíaco eficiente (species_cardiovascular.blood_viscosity_threshold). "
            "Limiar UCI/WADA: Hct >50% H / >47% F em ciclismo → suspensão preventiva. "
            "Hct baixo com Hb normal → hemodiluição (estado de overhydration peri-treino "
            "ou expansão de volume plasmático — adaptação desejável ao treino aeróbio). "
            "Acompanhar com ferritin e sTfR para distinguir pseudo-anemia de anemia real."
        ),
    )

    ferritin: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Ferritina sérica. Unidade: μg/L (= ng/mL). "
            "Referência funcional para atletas H: 80-200 μg/L; F: 50-150 μg/L. "
            "A ferritina é a principal proteína de armazenamento de ferro intracelular; "
            "é também reagente de fase aguda (sobe em inflamação, independentemente "
            "das reservas de ferro). "
            "Ferritina <30 μg/L em atletas → depleção de ferro sem anemia → "
            "comprometimento da biogénese mitocondrial (ferro-enxofre clusters), "
            "da actividade da ribonucleotide redutase, e da resposta ao EPO "
            "(species_hematological.iron_stores_ceiling). "
            "Ferritina >300 μg/L → sobrecarga de ferro ou inflamação sistémica; "
            "em HFE C282Y AA → hemocromatose activa. "
            "Interpretar SEMPRE em conjunto com PCR-hs (se PCR >5 mg/L, ferritina "
            "pode estar falsamente elevada)."
        ),
    )

    serum_iron: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Ferro sérico (sideraemia). Unidade: μmol/L (ou μg/dL; 1 μg/dL = 0.179 μmol/L). "
            "Referência H: 10.6-28.3 μmol/L; F: 6.6-26.0 μmol/L. "
            "O ferro sérico reflecte o ferro ligado à transferrina em circulação — "
            "mais variável ao longo do dia (maior de manhã) e mais sensível a "
            "ingestão recente do que a ferritina. "
            "Baixo ferro sérico com ferritina normal → má absorção intestinal (doença "
            "celíaca subclínica, SIBO) ou hepcidin elevada (inflamação crónica). "
            "Combinar com transferrin_saturation para diagnóstico diferencial "
            "(species_gastrointestinal.iron_absorption_efficiency)."
        ),
    )

    transferrin_saturation: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Saturação da transferrina (TSAT). Unidade: %. "
            "Fórmula: TSAT = (Ferro sérico / TIBC) × 100. "
            "Referência: 20-45%. "
            "TSAT <16% → ferro insuficiente para a eritropoiese → anemia funcional "
            "por deficiência de ferro, mesmo com ferritina normal (ferro 'bloqueado' "
            "por hepcidin em inflamação) — 'anaemia of inflammation'. "
            "TSAT >60% → sobrecarga de ferro (suspeita de hemocromatose ou "
            "suplementação excessiva). "
            "Diagnóstico diferencial essencial em atletas com Hb baixa: "
            "se TSAT <16% → suplementar ferro IV/oral; se TSAT normal → "
            "investigar outras causas de anemia (B12, folato, hemólise por impacto "
            "do pé — 'footstrike hemolysis' em corredores). "
            "Interacção com species_hematological.transferrin_saturation_reference."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 3 — METABÓLICO E GLICÉMICO
    # Interage com: species_bioenergetics, species_hepatic, species_renal
    # ══════════════════════════════════════════════════════════════════════════

    hba1c: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HbA1c — Hemoglobina glicada. Unidade: % ou mmol/mol (IFCC). "
            "Conversão: % = (mmol/mol / 10.929) + 2.15. "
            "Referência ideal em atletas: 4.8-5.4% (29-36 mmol/mol). "
            "A HbA1c reflecte a glicemia média dos últimos 60-90 dias (vida média "
            "do eritrócito ~120 dias; contribuição ponderada: últimos 30d = ~50%). "
            "HbA1c <4.8% em atletas de endurance de alto volume → possível anemia "
            "hemolítica / turnover aumentado de eritrócitos → HbA1c subestimada. "
            "HbA1c >5.7% → pré-diabetes → resistência à insulina → compromisso da "
            "captação de glicose muscular mediada por GLUT4 "
            "(species_bioenergetics.glucose_transport_efficiency). "
            "Atletas com HbA1c >6.0% têm comprometimento significativo da glicogénese "
            "hepática e muscular pós-exercício."
        ),
    )

    fasting_glucose: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Glicemia em jejum (mínimo 8h de jejum). Unidade: mmol/L (ou mg/dL; 1 mg/dL = 0.0556 mmol/L). "
            "Referência óptima: 4.0-5.0 mmol/L (72-90 mg/dL). "
            "Glicemia em jejum 5.6-6.9 mmol/L → pré-diabetes (ADA 2024). "
            "Glicemia em jejum <3.5 mmol/L em repouso → hipoglicemia de jejum → "
            "suspeita de hiperinsulinémia, insulinoma, ou síndrome pós-exercício de "
            "hipoglicemia retardada (species_bioenergetics.glucoregulation_baseline). "
            "Em atletas de endurance: glicemia em jejum ligeiramente mais baixa "
            "(3.8-4.6 mmol/L) é fisiológica por maior sensibilidade à insulina e "
            "maior captação basal muscular. "
            "Interpretar SEMPRE em conjunto com fasting_insulin e HOMA-IR."
        ),
    )

    fasting_insulin: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Insulina em jejum. Unidade: pmol/L (ou mUI/L; 1 mUI/L = 6.0 pmol/L). "
            "Referência óptima em atletas: <60 pmol/L (<10 mUI/L). "
            "A insulina é o principal hormona anabólica do metabolismo glucídico: "
            "activa PI3K-Akt-mTORC1, GLUT4 translocação, síntese de glicogénio "
            "(glycogen synthase) e inibe lipólise. "
            "Insulina em jejum elevada (>120 pmol/L) → hiperinsulinémia compensatória "
            "→ resistência à insulina → captação de glicose GLUT4 comprometida "
            "(species_bioenergetics.insulin_sensitivity_index). "
            "Insulina muito baixa (<18 pmol/L) com glicemia normal → alta sensibilidade "
            "à insulina (fenotípico de atleta de endurance de elite)."
        ),
    )

    homa_ir: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HOMA-IR (Homeostatic Model Assessment of Insulin Resistance). "
            "Fórmula: HOMA-IR = (Glucose_jejum_mmol/L × Insulina_jejum_mUI/L) / 22.5. "
            "Referência óptima atletas: <1.0; limiar de resistência: >2.5; "
            "resistência moderada: >3.5. "
            "O HOMA-IR estima a resistência à insulina hepática (supressão da "
            "glicogenólise hepática) e muscular (captação de glicose via GLUT4). "
            "HOMA-IR >2.5 em atleta → redução da taxa de síntese de glicogénio "
            "muscular pós-exercício em ~25-40% → recuperação comprometida e menor "
            "disponibilidade de glicose para treinos consecutivos "
            "(species_bioenergetics.hepatic_insulin_sensitivity). "
            "Interacção directa com apoe_genotype (E4E4 → maior HOMA-IR basal) e "
            "com o painel inflamatório (IL-6 e CRP elevados → resistência à insulina "
            "mediada por inflamação)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 4 — INFLAMATÓRIO E DANOS MUSCULARES
    # Interage com: species_immune_microbiome, species_oxidative_stress,
    #               species_musculoskeletal
    # ══════════════════════════════════════════════════════════════════════════

    crp_high_sensitivity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "PCR de alta sensibilidade (hs-CRP). Unidade: mg/L. "
            "Referência de baixo risco cardiovascular: <1.0 mg/L. "
            "Referência óptima atletas: <0.5 mg/L (pré-treino, estado de recuperação). "
            "A PCR-hs é sintetizada pelo fígado em resposta a IL-6 e TNF-α; "
            "marcador de inflamação sistémica de baixo grau e stress oxidativo crónico. "
            "PCR-hs <1.0 mg/L → inflamação de baixo risco; 1-3 mg/L → risco intermédio; "
            ">3 mg/L → inflamação sistémica activa. "
            "Em atletas, PCR-hs >3 mg/L por >3 semanas consecutivas → sobrecarga "
            "imunológica → sobretreinamento inflamatório (species_immune_microbiome.systemic_inflammation_threshold). "
            "Falseia ferritina (para cima) e albumina (para baixo) em estados "
            "inflamatórios agudos — interpretar todo o painel bioquímico com PCR-hs "
            "como covariável de ajuste."
        ),
    )

    il6_basal: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "IL-6 basal (interleucina-6 em repouso, >24h pós-exercício). Unidade: pg/mL. "
            "Referência em repouso: <2.0 pg/mL. "
            "IMPORTANTE: IL-6 tem efeito bifásico no exercício — durante o exercício "
            "muscular actua como miocina pró-metabólica (estimula lipolise e uptake "
            "de glicose); em repouso elevada é marcador de inflamação sistémica "
            "crónica e activação de macrofagos M1. "
            "IL-6 em repouso >5 pg/mL → sinalização pro-inflamatória persistente → "
            "activação de SOCS3 → resistência à insulina e resistência ao IGF-1 "
            "(species_immune_microbiome.chronic_inflammation_cytokine_threshold e "
            "species_endocrine.igf1_signaling_efficiency). "
            "IL-6 crónica elevada → activação de HPA (cortisol↑) → catabolismo muscular "
            "por via ubiquitina-proteasoma → balanço azotado negativo."
        ),
    )

    creatine_kinase: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "CK — Creatina Quinase total sérica. Unidade: U/L. "
            "Referência em repouso (>48h pós-exercício): H <200 U/L; F <170 U/L. "
            "A CK é enzima intracelular libertada por dano da membrana miofibrilar; "
            "é o marcador mais sensível e específico de dano muscular agudo. "
            "CK 200-1000 U/L → dano muscular moderado → 48-72h de recuperação necessárias "
            "para restauração da função contráctil (species_musculoskeletal.muscle_damage_recovery_rate). "
            "CK >1000 U/L → dano muscular severo / risco de rabdomiólise. "
            "CK >10,000 U/L → rabdomiólise activa → risco de insuficiência renal aguda "
            "por mioglobinúria (activar protocolo de hidratação intensiva e monitorização renal). "
            "Cronicamente elevada em repouso (>500 U/L H / >400 U/L F) em múltiplas "
            "colheitas → resposta de dano crónico → overtreino mecânico."
        ),
    )

    urea: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Ureia sérica (BUN × 2.14 = ureia; ou reportada directamente). Unidade: mmol/L. "
            "Referência: 2.5-7.5 mmol/L. "
            "A ureia é o produto final do catabolismo de aminoácidos: "
            "NH₃ (desaminação) → ciclo da ureia hepático → ureia → excreção renal. "
            "Em atletas: ureia pré-treino elevada (>8 mmol/L) → balanço azotado "
            "negativo → catabolismo proteico excessivo → ingestão proteica insuficiente "
            "ou volume/intensidade de treino excessivo para a ingestão calórica actual "
            "(species_protein_metabolism.nitrogen_balance_threshold). "
            "Ureia pós-treino pode subir fisiologicamente 20-40% e normalizar em 24h. "
            "Ureia cronicamente elevada + creatinina elevada → avaliar função renal "
            "(species_renal_excretory.glomerular_filtration_rate). "
            "Ureia muito baixa (<2.0 mmol/L) → hiperidratação ou ingestão proteica "
            "insuficiente (em atletas plant-based com baixo turnover proteico)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 5 — TIROIDE
    # Interage com: species_endocrine, species_bioenergetics, species_chronobiology
    # ══════════════════════════════════════════════════════════════════════════

    tsh: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "TSH — Hormona Estimulante da Tiroide (Tireotrópica). Unidade: mUI/L. "
            "Referência laboratorial: 0.4-4.0 mUI/L; óptimo funcional atletas: 1.0-2.5 mUI/L. "
            "TSH > 2.5 mUI/L → suspeita de hipotiroidismo subclínico → redução do "
            "metabolismo basal (RMR), bradicardia, fadiga, aumento de LDL-C, "
            "perturbação do ritmo circadiano de temperatura corporal "
            "(species_endocrine.thyroid_function_reference). "
            "TSH < 0.4 mUI/L → hipertiroidismo → taquicardia, sudorese, perda de "
            "massa muscular, osteopenia. "
            "TSH é o melhor screening isolado mas deve ser confirmado com fT3/fT4. "
            "Interacção com VDR ff genótipo: défice de vitamina D pode agravar "
            "autoimunidade tiroideia (Hashimoto)."
        ),
    )

    ft3_free_triiodothyronine: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "fT3 — Tri-iodotironina livre (fracção activa). Unidade: pmol/L. "
            "Referência: 3.5-6.5 pmol/L. "
            "O T3 livre é a forma biologicamente activa da hormona tiroideia; "
            "liga-se ao receptor nuclear TRα (músculo cardíaco e esquelético) e TRβ "
            "(fígado, tecido adiposo) → aumento do consumo de O₂ mitocondrial, "
            "síntese de β-miosina e SERCA2a. "
            "fT3 baixo com TSH elevado → hipotiroidismo periférico. "
            "fT3 baixo com TSH normal/baixo → 'Síndrome do T3 baixo' (Low T3 Syndrome) "
            "em restrição calórica severa (RED-S) ou doença crónica → redução do RMR "
            "(species_bioenergetics.thyroid_metabolic_rate_modifier)."
        ),
    )

    ft4_free_thyroxine: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "fT4 — Tiroxina livre (precursor). Unidade: pmol/L. "
            "Referência: 10-23 pmol/L. "
            "O T4 é convertido perifericamente em T3 activo pela deiodinase tipo 1/2 "
            "(D1 hepática e renal; D2 muscular e cerebral). "
            "fT4 normal com fT3 baixo → défice de conversão periférica → comum em: "
            "défice de selénio (cofactor da deiodinase), stress crónico (cortisol "
            "inibe D2), restrição de carbohidratos extrema. "
            "Selénio (ver serum_selenium) é cofactor obrigatório das deiodinases "
            "(species_endocrine.t4_to_t3_conversion_efficiency)."
        ),
    )

    rt3_reverse_t3: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "rT3 — T3 reverso (metabolito inactivo). Unidade: pmol/L. "
            "Referência: 0.14-0.54 nmol/L (140-540 pmol/L; verificar unidades do lab). "
            "O rT3 é formado pela conversão alternativa de T4 pela deiodinase tipo 3 "
            "(inactivadora) em vez da D1/D2 (activadora). "
            "rT3 elevado → inibição competitiva do receptor TRα pelo rT3 → hipotiroidismo "
            "funcional apesar de TSH/fT4 normais. "
            "Causas de rT3 elevado: cortisol crónico elevado (stress/overtreino), "
            "défice de ferro (inibe D1), selénio insuficiente, inflamação sistémica. "
            "Rácio fT3/rT3 > 2.0 é considerado funcional; < 1.0 → tiroidização "
            "periférica comprometida (species_endocrine.rt3_functional_threshold)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 6 — LÍPIDOS COMPLETOS E RISCO CARDIOVASCULAR
    # Interage com: species_lipid_metabolism, species_cardiovascular,
    #               species_hepatic
    # ══════════════════════════════════════════════════════════════════════════

    ldl_cholesterol: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "LDL-C — Colesterol LDL calculado (Friedewald) ou medido directamente. "
            "Unidade: mmol/L (ou mg/dL; 1 mg/dL = 0.0259 mmol/L). "
            "Referência óptima atletas: < 2.6 mmol/L (<100 mg/dL). "
            "LDL-C é o principal marcador de risco aterosclerótico mas tem limitações "
            "em atletas com TG < 0.5 mmol/L (Friedewald subestima). "
            "Complementar SEMPRE com ApoB (conta partículas aterogénicas directamente). "
            "Interacção com APOE genótipo: E4/E4 → LDL-C elevado mesmo com dieta "
            "controlada (species_lipid_metabolism.ldl_clearance_rate)."
        ),
    )

    hdl_cholesterol: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "HDL-C — Colesterol HDL. Unidade: mmol/L. "
            "Referência desejável H: > 1.0 mmol/L; F: > 1.3 mmol/L. "
            "O treino aeróbio aumenta HDL-C em média 5-10% por regulação de ABCA1 "
            "e LCAT (Lecithin–Cholesterol Acyltransferase). "
            "HDL-C elevado em atletas de endurance: 1.8-2.5 mmol/L é comum. "
            "species_lipid_metabolism.hdl_reverse_cholesterol_transport."
        ),
    )

    triglycerides: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Triglicéridos em jejum. Unidade: mmol/L. "
            "Referência óptima: < 1.0 mmol/L; limite: < 1.7 mmol/L. "
            "TG > 1.7 → metabolismo lipídico comprometido; TG > 5.6 → risco de "
            "pancreatite aguda. "
            "Em atletas de endurance bem treinados: TG < 0.7 mmol/L é comum "
            "(alta actividade de LPL muscular → clearance de quilomicrons eficiente). "
            "TG elevados + HDL baixo = padrão de resistência à insulina (síndrome "
            "metabólica) — correlacionar com HOMA-IR "
            "(species_lipid_metabolism.triglyceride_clearance_rate)."
        ),
    )

    apob: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "ApoB — Apolipoproteína B100. Unidade: g/L. "
            "Referência desejável: < 0.90 g/L; óptimo: < 0.65 g/L. "
            "Cada partícula LDL, VLDL e IDL tem exactamente 1 molécula de ApoB → "
            "ApoB conta o número total de partículas aterogénicas (superior ao LDL-C). "
            "ApoB é o melhor preditor de risco cardiovascular em atletas com LDL-C "
            "pseudo-normal mas partículas LDL pequenas e densas. "
            "Interacção directa com APOE genótipo: E4/E4 → ApoB elevado resistente "
            "a intervenção dietética (species_cardiovascular.apob_atherogenic_threshold)."
        ),
    )

    lp_a: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Lp(a) — Lipoproteína (a). Unidade: nmol/L (ou mg/dL; preferir nmol/L). "
            "Referência de baixo risco: < 75 nmol/L (< 30 mg/dL). "
            "Alto risco: > 125 nmol/L (> 50 mg/dL). "
            "Lp(a) é determinada geneticamente (LPA gene); não é modificável "
            "por dieta ou exercício. "
            "Lp(a) elevada → risco aumentado de estenose aórtica calcificada e "
            "doença coronária prematura mesmo em atletas jovens. "
            "Se Lp(a) > 125 nmol/L → ecocardiograma anual e monitorização de "
            "ApoB (species_cardiovascular.lpa_genetic_cardiovascular_risk)."
        ),
    )

    non_hdl_cholesterol: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Não-HDL colesterol = Colesterol total − HDL-C. Unidade: mmol/L. "
            "Referência desejável: < 3.4 mmol/L. "
            "O não-HDL captura todas as partículas aterogénicas (LDL + VLDL + IDL + Lp(a)) "
            "numa única métrica. Mais informativo que LDL-C em jejum incompleto. "
            "species_lipid_metabolism.non_hdl_atherogenic_burden."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 7 — VITAMINAS E MICRONUTRIENTES CRÍTICOS
    # Interage com: species_osseous_system, species_immune_microbiome,
    #               species_endocrine, species_mitochondrial
    # ══════════════════════════════════════════════════════════════════════════

    vitamin_d_25oh: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "25-OH-Vitamina D (Calcidiol). Unidade: nmol/L (ou ng/mL; 1 ng/mL = 2.5 nmol/L). "
            "Referência óptima atletas: 100-150 nmol/L (40-60 ng/mL). "
            "Alvo mínimo: > 75 nmol/L (30 ng/mL). Deficiência: < 50 nmol/L (<20 ng/mL). "
            "A vitamina D activa (1,25-OH₂-D, calcitriol) via VDR regula >200 genes: "
            "síntese muscular (IGF-1 muscular, receptor androgénio), função imunológica "
            "(células Th1/Treg, catelicidina), mineralização óssea (osteocalcina, "
            "RANK/RANKL), e expressão de miocina (IL-6, irisin). "
            "Défice → redução de força e VO2max, maior risco de fracturas de stress e URTI. "
            "Prescrição ajustada por VDR FokI: ff → alvo 125-150 nmol/L; FF → 100-125 nmol/L "
            "(species_osseous_system.vitamin_d_bone_mineralization_threshold)."
        ),
    )

    vitamin_b12: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Vitamina B12 (Cobalamina) sérica. Unidade: pmol/L (ou pg/mL; 1 pg/mL = 0.738 pmol/L). "
            "Referência funcional: > 300 pmol/L (> 400 pg/mL). "
            "Sérico baixo (< 150 pmol/L) com sintomas → défice activo. "
            "B12 é cofactor da metionina sintase (remetilação de homocisteína → metionina) "
            "e da metilmalonyl-CoA mutase (metabolismo de ácidos gordos de cadeia ímpar). "
            "Atletas veganos + MTHFR C677T TT: risco máximo de défice funcional → "
            "preferir holotranscobalamina II (B12 activa) como marcador mais sensível. "
            "Défice → megaloblastose, hiperhomocisteinémia, desmielinização "
            "(species_epigenetic.sam_sah_ratio_reference — SAM depende de B12)."
        ),
    )

    rbc_folate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Folato eritrocitário (RBC folate). Unidade: nmol/L. "
            "Referência: > 340 nmol/L (> 150 ng/mL). "
            "O folato eritrocitário reflecte o status de folato dos últimos 120 dias "
            "(superior ao folato sérico que varia com ingestão recente). "
            "O 5-MTHF (metiltetrahidrofolato) é o substrato da metionina sintase; "
            "défice → hiperhomocisteinémia e compromisso de síntese de ADN (eritropoiese). "
            "Atletas MTHFR C677T TT necessitam de 5-MTHF (L-metilfolato) em vez de "
            "ácido fólico (não conseguem converter eficientemente) "
            "(species_epigenetic.sam_sah_ratio_reference)."
        ),
    )

    homocysteine: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Homocisteína total plasmática. Unidade: μmol/L. "
            "Referência óptima: < 8 μmol/L; normal: 5-15 μmol/L; "
            "hiperhomocisteinémia: > 15 μmol/L. "
            "A homocisteína é um aminoácido sulfurado formado a partir de metionina; "
            "é remetilada em metionina por B12+folato ou transulfurada para cisteína por B6. "
            "Homocisteína > 10 μmol/L → disfunção endotelial, stress oxidativo vascular, "
            "activação de NF-κB, oxidação de LDL → risco aterosclerótico "
            "(species_cardiovascular.endothelial_homocysteine_threshold). "
            "Em atletas: exercício intenso eleva homocisteína transitoriamente; "
            "crónica elevada → MTHFR TT + défice B12/folato + baixo SAM/SAH ratio "
            "(species_epigenetic.methylation_homocysteine_axis)."
        ),
    )

    omega_3_index: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Índice Ómega-3 (EPA + DHA como % dos ácidos gordos totais em eritrócitos). "
            "Unidade: %. "
            "Referência de baixo risco cardiovascular: > 8%. "
            "Zona de alto risco: < 4%. Zona intermédia: 4-8%. "
            "O índice ómega-3 reflecte os últimos 3-4 meses de ingestão de EPA+DHA "
            "(integração eritrocitária). "
            "EPA → precursor de eicosanóides anti-inflamatórios (PGE3, LTB5, resolvinas E). "
            "DHA → componente estrutural das membranas mitocondriais e neurais (fluência). "
            "Índice < 4% → membranas eritrocitárias menos fluidas → menor deformabilidade "
            "eritrocitária → prejuízo de extracção de O₂ em capilares estreitos "
            "(species_hematological.erythrocyte_deformability_modifier). "
            "Interacção com FADS1 TT genótipo: necessidade de EPA/DHA pré-formado "
            "(species_lipid_metabolism.omega3_membrane_composition_threshold)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 8 — MINERAIS E OLIGOELEMENTOS
    # Interage com: species_endocrine, species_mitochondrial,
    #               species_neural_cognitive, species_immune_microbiome
    # ══════════════════════════════════════════════════════════════════════════

    rbc_magnesium: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Magnésio intracelular (eritrocitário — RBC Mg). Unidade: mmol/L. "
            "Referência RBC: 1.65-2.65 mmol/L. "
            "IMPORTANTE: magnésio sérico (normal: 0.75-0.95 mmol/L) é homeostático "
            "e não reflecte o status intracelular — um atleta deficiente tem Mg sérico "
            "normal mas RBC Mg baixo. "
            "Magnésio é cofactor de >300 enzimas: ATP-sintetase (todo o ATP é Mg-ATP), "
            "creatina quinase, piruvato quinase, Na+/K+-ATPase, e adenilato ciclase. "
            "Défice → cãibras musculares, arritmias, insónia, maior produção de cortisol "
            "em resposta a stress (species_endocrine.magnesium_hpa_axis_buffer). "
            "Atletas perdem Mg no suor (~0.5-1.0 mmol/L) e na urina em volumes altos."
        ),
    )

    serum_zinc: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Zinco sérico. Unidade: μmol/L (ou μg/dL; 1 μg/dL = 0.153 μmol/L). "
            "Referência: 11-24 μmol/L (70-150 μg/dL). "
            "Zinco é cofactor da SOD-Cu/Zn (superóxido dismutase — principal "
            "antioxidante citosólico), da testosterona sintase (5α-redutase e "
            "aromatase), e da DNA polimerase. "
            "Zinco < 9 μmol/L → défice funcional → maior stress oxidativo pós-treino, "
            "supressão de testosterone (Mg+Zn juntos → ZMA → relevante em treino de "
            "força), comprometimento imunológico "
            "(species_endocrine.zinc_testosterone_cofactor e "
            "species_oxidative_stress.superoxide_dismutase_zinc_dependency)."
        ),
    )

    serum_selenium: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Selénio sérico. Unidade: μmol/L (ou μg/L; 1 μg/L ≈ 0.013 μmol/L). "
            "Referência: 1.0-2.0 μmol/L (80-160 μg/L). "
            "Selénio é cofactor obrigatório das selenoproteínas: "
            "(1) GPx1/GPx4 (glutationa peroxidase) — neutralização de H₂O₂ e "
            "hidroperóxidos lipídicos (species_oxidative_stress.glutathione_peroxidase_activity); "
            "(2) Deiodinases D1/D2/D3 — conversão T4→T3 activo; "
            "(3) Tioredoxina redutase (TrxR) — regeneração de tioredoxina oxidada. "
            "Défice → hipotiroidismo funcional (baixo fT3 por inibição de D2) + "
            "maior stress oxidativo mitocondrial pós-treino intenso "
            "(species_endocrine.selenium_thyroid_deiodinase_cofactor)."
        ),
    )

    coq10_plasma: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "CoQ10 (Coenzima Q10 / Ubiquinol) plasmático. Unidade: μmol/L. "
            "Referência jovem adulto: 0.8-1.5 μmol/L; "
            "em uso de estatinas: pode cair para < 0.4 μmol/L. "
            "CoQ10 é componente integral da cadeia de transporte de electrões "
            "mitocondrial (complexos I e II → CoQ10 → complexo III) e antioxidante "
            "lipossolúvel da membrana mitocondrial interna. "
            "CoQ10 < 0.6 μmol/L → comprometimento da produção de ATP mitocondrial → "
            "fadiga muscular prematura e comprometimento da performance aeróbia "
            "(species_mitochondrial.electron_transport_chain_efficiency). "
            "Síntese endógena requer: HMG-CoA reductase (inibida por estatinas → "
            "miopatia por estatinas), tirosina, e mevalonate pathway. "
            "Atletas em estatinas: CoQ10 + suplementação são obrigatórios."
        ),
    )

    nad_nadh_ratio: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Rácio NAD+/NADH em células mononucleares de sangue periférico (PBMC). "
            "Rácio óptimo: > 700 (plasma); referência em sangue total varia por método. "
            "NAD+ é cofactor obrigatório de: sirtuínas (SIRT1-7 → regulação epigenética "
            "e biogénese mitocondrial via PGC-1α), PARP1 (reparação do ADN), "
            "e complexos I/III da cadeia respiratória. "
            "NAD+ decresce com a idade, com exercício de alta intensidade sem recuperação, "
            "e com álcool (conversão para NADH pelo álcool desidrogenase → redução do "
            "rácio NAD+/NADH → inibição do ciclo de Krebs). "
            "Rácio baixo → menor actividade de SIRT1 → menor biogénese mitocondrial → "
            "comprometimento da adaptação ao treino aeróbio "
            "(species_mitochondrial.nad_sirtuin_axis_efficiency)."
        ),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PAINEL 9 — FUNÇÃO HEPÁTICA E RENAL
    # Interage com: species_hepatic, species_renal_excretory
    # ══════════════════════════════════════════════════════════════════════════

    alt: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "ALT — Alanina Aminotransferase. Unidade: U/L. "
            "Referência H: 7-56 U/L; F: 7-45 U/L. "
            "A ALT é enzima intracelular hepática; é o marcador mais específico "
            "de lesão hepatocelular (superior à AST em especificidade hepática). "
            "ALT elevada em atletas (x2-3 LSN): avaliar com CK — se CK muito elevada, "
            "a ALT pode ser de origem muscular (isoenzima muscular de ALT). "
            "ALT > 100 U/L sem explicação → biópsia hepática ou ecografia. "
            "Esteatose hepática não-alcoólica: ALT/AST > 1.0 + hiperinsulinemia "
            "(species_hepatic.hepatocellular_integrity_threshold)."
        ),
    )

    ast: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "AST — Aspartato Aminotransferase. Unidade: U/L. "
            "Referência H: 10-40 U/L; F: 10-35 U/L. "
            "AST existe no fígado, músculo cardíaco, músculo esquelético, rins e cérebro. "
            "Rácio AST/ALT (De Ritis): < 1 → hepatite viral/tóxica; > 2 → hepatite alcoólica. "
            "AST isolada elevada em atletas de força → origem muscular (CK também elevada). "
            "Monitorizar com CK para distinguir lesão hepática de lesão muscular "
            "(species_hepatic.liver_enzyme_muscle_context)."
        ),
    )

    ggt: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "GGT — Gama-Glutamiltransferase. Unidade: U/L. "
            "Referência H: 8-61 U/L; F: 5-36 U/L. "
            "GGT é marcador sensível de stress hepático e oxidativo: "
            "transfere γ-glutamil da glutationa (GSH) para outros peptídeos "
            "→ regeneração do precursor de GSH. "
            "GGT elevada → steatose hepática, consumo de álcool, medicação "
            "(estatinas, paracetamol), stress oxidativo crónico. "
            "GGT > 40 U/L + álcool > 7 unidades/semana → síndrome hepática "
            "induzida por álcool. "
            "Em atletas: GGT pode subir após cargas de treino muito elevadas "
            "(species_hepatic.ggt_oxidative_stress_marker)."
        ),
    )

    creatinine: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Creatinina sérica. Unidade: μmol/L (ou mg/dL; 1 mg/dL = 88.4 μmol/L). "
            "Referência H: 62-115 μmol/L; F: 44-97 μmol/L. "
            "ATENÇÃO em atletas: creatinina é produto da degradação de creatina muscular "
            "(1-2% da creatina total/dia → creatinina). "
            "Atletas de massa muscular elevada têm creatinina > 115 μmol/L sem "
            "comprometimento renal — contexto com lean_mass_kg é obrigatório. "
            "Suplementação de creatina monohidratada → creatinina elevada "
            "(falso alarme renal). "
            "Usar eGFR (calculada por CKD-EPI) em vez de creatinina isolada para "
            "avaliação da função renal (species_renal_excretory.glomerular_filtration_rate)."
        ),
    )

    egfr_ckd_epi: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "eGFR — Taxa de Filtração Glomerular Estimada (fórmula CKD-EPI 2021). "
            "Unidade: mL/min/1.73m². "
            "Classificação ERC: G1 ≥ 90 (normal/aumentada); G2 60-89 (ligeiramente diminuída); "
            "G3a 45-59; G3b 30-44; G4 15-29; G5 < 15 (falência). "
            "eGFR < 60 em atleta → investigar urgente (nefropatia por AINE, rabdomiólise "
            "prévia, proteinúria de esforço crónica). "
            "eGFR > 90 é esperado em atletas jovens bem hidratados "
            "(species_renal_excretory.glomerular_filtration_rate)."
        ),
    )

    # ── Relações ─────────────────────────────────────────────────────────────
    athlete: Mapped["AthleteCore"] = relationship(
        back_populates="biomarker_panels"
    )
