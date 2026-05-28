"""
Genetic ceiling layer — computes TGI (Teto Genético Individual) per system.

TGI = T_espécie × ∏ Mj^G × E_epi

All numeric values are sourced exclusively from app.engine.constants.
Untested SNPs (missing from the athlete's genotype dict) default to 1.0 (neutral).
Epistasis (ε) is applied after computing the individual gene product, implementing
the 3rd-order network interaction defined in Doc2 §4.3.

Reference: Doc1 §2.x and Doc2 §4–5 for system-specific SNP assignments.
"""

from __future__ import annotations

from app.engine.base import multiplicative_combine
from app.engine.constants import GENETIC_MODIFIERS
from app.engine.modifiers.epistasis import compute_epistasis_modifier

# ---------------------------------------------------------------------------
# SNP-to-system assignment
# Maps each physiological system to the list of (modifier_table_key, genotype_field)
# pairs that determine its genetic product.
#
# modifier_table_key  → key inside GENETIC_MODIFIERS
# genotype_field      → key in the athlete's genotype dict
# ---------------------------------------------------------------------------
_SYSTEM_SNP_MAP: dict[str, list[tuple[str, str]]] = {
    "vo2max": [
        # Doc1 §2.1 / Doc2 §5.2
        ("PPARGC1A",          "PPARGC1A"),
        ("ACE",               "ACE"),
        ("HIF1A",             "HIF1A"),
        ("VEGF",              "VEGF"),
        ("MTDNA_HAPLOGROUP",  "MTDNA_HAPLOGROUP"),
        ("ACTN3_vo2max",      "ACTN3"),
    ],
    "cho": [
        # Doc1 §3.2 — no dominant nuclear genetic modifier declared for CHO transport;
        # G_CHO = 1.0 by document specification (no SNPs listed for this system).
    ],
    "fatmax": [
        # Doc1 §2.3 / Doc2 §8.2
        ("ADRB2",         "ADRB2"),
        ("ACTN3_fatmax",  "ACTN3"),
    ],
    "mps": [
        # Doc1 §2.4 / Doc2 §7.x
        ("MTHFR_C677T",    "MTHFR_C677T"),
        ("IGF1_promoter",  "IGF1_promoter"),
        ("MSTN",           "MSTN"),
    ],
    "peak_power": [
        # Doc1 §2.5
        ("ACTN3_power",  "ACTN3"),
        ("MYH7",         "MYH7"),
    ],
    "structural": [
        # Doc1 §2.6 / Doc2 §12
        ("COL5A1",  "COL5A1"),
        ("COL1A1",  "COL1A1"),
        ("MMP3",    "MMP3"),
    ],
    "cognitive": [
        # Doc2 §4.3 — epistasis pair MTHFR × COMT defines this system's genetic layer
        ("MTHFR_C677T_cognitive",  "MTHFR_C677T"),
        ("COMT_Val158Met",         "COMT_Val158Met"),
    ],
    "thermoregulatory": [
        # No nuclear genetic modifiers declared in the documents for thermoregulation.
        # G = 1.0 by absence of declared SNPs.
    ],
}


def _lookup_modifier(modifier_table_key: str, genotype_value: str | None) -> float:
    """
    Retrieve the scalar modifier for a single SNP from GENETIC_MODIFIERS.

    Returns 1.0 (neutral) when:
      - The SNP was not tested (genotype_value is None)
      - The genotype string is not found in the modifier table
      - The stored modifier value for this genotype is None (undeclared in documents)
    """
    if genotype_value is None:
        return 1.0

    table = GENETIC_MODIFIERS.get(modifier_table_key)
    if table is None:
        return 1.0

    value = table.get(genotype_value)
    if value is None:
        return 1.0

    return float(value)


def compute_genetic_ceiling(
    system: str,
    genotype: dict[str, str],
) -> float:
    """
    Compute the genetic modifier product (∏ Mj^G) for one physiological system,
    then apply the epistatic network coefficient (ε) for that system.

    Formula: TGI_modifier = (∏ Mj^G) × ε_system

    Args:
        system:   one of "vo2max", "cho", "fatmax", "mps", "peak_power", "structural"
        genotype: dict mapping SNP profile field names to genotype strings,
                  e.g. {"ACTN3": "RR", "ACE": "II", "MTHFR_C677T": "CT", ...}

    Returns:
        Scalar ∈ (0, ∞) — the combined genetic ceiling modifier for this system.
        A value of 1.0 means this athlete's genome is neutral for this system.
    """
    snp_list = _SYSTEM_SNP_MAP.get(system, [])

    individual_modifiers: list[float] = [
        _lookup_modifier(modifier_table_key, genotype.get(genotype_field))
        for modifier_table_key, genotype_field in snp_list
    ]

    # ∏ Mj^G — multiplicative combination of all single-gene modifiers
    gene_product = multiplicative_combine(individual_modifiers) if individual_modifiers else 1.0

    # ε — 3rd-order epistatic network dampening (Doc2 §4.3)
    epsilon = compute_epistasis_modifier(genotype, system)

    return gene_product * epsilon


def compute_all_genetic_ceilings(
    genotype: dict[str, str],
) -> dict[str, float]:
    """
    Convenience wrapper: compute genetic ceiling modifiers for all 6 systems.

    Returns a dict keyed by system name, values are the TGI modifier scalars.
    """
    return {
        system: compute_genetic_ceiling(system, genotype)
        for system in _SYSTEM_SNP_MAP
    }
