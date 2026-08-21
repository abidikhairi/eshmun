"""
Pure logic for parsing UniProt's "Interacts with" (cc_interaction) field into
structured (entry, partner_accession) pairs.

No network calls, no file I/O.
"""

import re

_ISOFORM_SUFFIX_RE = re.compile(r"-\d+$")


def strip_isoform_suffix(accession: str) -> str:
    """UniProt isoform accessions look like 'O43251-6' -- our entry universe
    (human_protein_kg, sequence tables) is keyed by base accession only, so
    isoform-specific partners are folded down to their base protein."""
    return _ISOFORM_SUFFIX_RE.sub("", accession)


def parse_interactors(raw: str) -> list[str]:
    """'Q8N111; Q99653; O43251-6' -> ['Q8N111', 'Q99653', 'O43251']

    Empty/whitespace input (the common case -- most SwissProt entries have no
    annotated interactions) returns []. Self-interaction (a protein listed as
    its own interactor, i.e. homodimerization) is preserved, not filtered --
    it's a real, biologically meaningful annotation, not a data error.
    """
    if not raw or not raw.strip():
        return []
    tokens = [t.strip() for t in raw.split(";")]
    return [strip_isoform_suffix(t) for t in tokens if t]


def build_ppi_pairs(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """rows: list of (entry, raw_interacts_with_field) -> flat list of
    (entry, partner) pairs, one per interactor. Order-preserving, duplicates
    kept (caller's choice whether to dedupe)."""
    pairs = []
    for entry, raw in rows:
        for partner in parse_interactors(raw):
            pairs.append((entry, partner))
    return pairs
