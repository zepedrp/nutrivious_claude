"""
Epistasis layer — 3rd-order non-linear gene-gene interactions.

ε < 1.0 means the compound effect of two co-present SNPs is LESS than their
product would predict (negative epistasis / antagonism). This is not additive;
it is a network-level dampening of the individual genetic signals.

Reference: Doc2 §4.3 — EPISTASIS_COEFFICIENTS
"""

from __future__ import annotations

from app.engine.constants import EPISTASIS_COEFFICIENTS

# Maps each composite key string used in EPISTASIS_COEFFICIENTS to a
# (profile_field, required_genotype) pair so that we can look up whether
# the athlete carries that specific allelic variant.
#
# Convention: profile_field is the key in the athlete's genotype dict,
# required_genotype is the value that must match for the variant to be active.
_VARIANT_LOOKUP: dict[str, tuple[str, str]] = {
    # Pair 1 — SNC dopamine (Doc2 §4.3)
    "MTHFR_C677T_TT":           ("MTHFR_C677T",         "TT"),
    "COMT_Val158Met_MM":         ("COMT_Val158Met",       "Met_Met"),
    # Pair 2 — VO2max (Doc2 §4.3)
    "PPARGC1A_Gly482Ser_SS":    ("PPARGC1A",             "Ser_Ser"),
    "ACE_DD":                    ("ACE",                  "DD"),
    # Pair 3 — structural triangle (Doc2 §4.3)
    "COL5A1_CC":                 ("COL5A1",               "CC"),
    "COL1A1_ss":                 ("COL1A1",               "ss"),
    "MMP3_5A_5A":                ("MMP3",                 "5A_5A"),
    # Pair 4 — sprint (Doc2 §4.3)
    "ACTN3_R577X_XX":            ("ACTN3",                "XX"),
    "AMPD1_Q12X_XX":             ("AMPD1",                "XX"),
    # Pair 5 — body composition (Doc2 §4.3)
    "VDR_low_function":          ("VDR",                  "low_function"),
    "FTO_AA":                    ("FTO",                  "AA"),
    "PPARG_Pro12_low":           ("PPARG",                "Pro12_low"),
}

# Maps each epistatic pair (tuple key) to the physiological systems it modulates.
# Used by genetic.py to apply ε only to the relevant system modifier.
EPISTASIS_SYSTEM_MAP: dict[tuple, list[str]] = {
    ("MTHFR_C677T_TT", "COMT_Val158Met_MM"):                    ["cognitive"],
    ("PPARGC1A_Gly482Ser_SS", "ACE_DD"):                        ["vo2max"],
    ("COL5A1_CC", "COL1A1_ss", "MMP3_5A_5A"):                   ["structural"],
    ("ACTN3_R577X_XX", "AMPD1_Q12X_XX"):                        ["peak_power"],
    ("VDR_low_function", "FTO_AA", "PPARG_Pro12_low"):          ["fatmax", "mps"],
}


def _variant_present(variant_key: str, genotype: dict[str, str]) -> bool:
    """Return True iff the athlete carries the exact allelic variant required."""
    if variant_key not in _VARIANT_LOOKUP:
        return False
    profile_field, required_genotype = _VARIANT_LOOKUP[variant_key]
    athlete_genotype = genotype.get(profile_field)
    if athlete_genotype is None:
        return False
    return athlete_genotype == required_genotype


def compute_epistasis_modifier(
    genotype: dict[str, str],
    system: str,
) -> float:
    """
    Scan all epistatic pairs in EPISTASIS_COEFFICIENTS for the given system.

    For each pair that (a) affects this system and (b) has ALL variants present
    in the athlete's genotype, the corresponding ε is returned.  When multiple
    pairs are active for the same system, we take their product — each is an
    independent 3rd-order network constraint.

    If any required SNP in a pair is untested (missing from genotype), that pair
    is conservatively treated as inactive (ε = 1.0 for that pair).

    Args:
        genotype: dict mapping profile SNP fields to genotype strings,
                  e.g. {"ACTN3": "XX", "MTHFR_C677T": "TT", ...}
        system:   physiological system name, e.g. "vo2max", "structural"

    Returns:
        Composite ε ∈ (0, 1] — product of all active epistatic coefficients.
        Returns 1.0 when no active pairs are found (neutral, no dampening).
    """
    composite_epsilon = 1.0

    for pair_key, epsilon in EPISTASIS_COEFFICIENTS.items():
        # Check whether this pair affects the requested system
        affected_systems = EPISTASIS_SYSTEM_MAP.get(pair_key, [])
        if system not in affected_systems:
            continue

        # All variants in the pair must be simultaneously present
        if all(_variant_present(variant_key, genotype) for variant_key in pair_key):
            composite_epsilon *= epsilon

    return composite_epsilon
