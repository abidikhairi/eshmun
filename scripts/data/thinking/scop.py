"""
Pure logic for parsing SCOPe classification/description files and joining
them to UniProt accessions via SIFTS.

No network calls, no file I/O.
"""

from dataclasses import dataclass

# SCOPe node types we care about -- cl (class) is too coarse, dm/sp/px are
# per-domain/species/instance bookkeeping, not classification levels.
LEVEL_KEYS = ["cf", "sf", "fa"]  # fold, superfamily, family
LEVEL_TO_RELATION = {"cf": "scop_fold", "sf": "scop_superfamily", "fa": "scop_family"}

# SCOPe class "l" (Artifacts) has exactly one leaf, "Tags", covering
# expression/purification tags (His-tag, GST-tag, etc.) that got
# crystallized alongside the real protein -- not a genuine structural
# classification of the protein itself. Excluded at all three levels.
ARTIFACT_VALUES = {"Tags"}


@dataclass(frozen=True)
class ScopeDomain:
    sid: str
    pdbid: str
    chains: frozenset
    fold_sunid: str | None
    superfamily_sunid: str | None
    family_sunid: str | None


def parse_scope_chains(raw_chain_field: str) -> set[str]:
    """'D:8-147' -> {'D'}; 'A:1-13,A:355-392' -> {'A'};
    'C:488-585,D:603-875,E:885-1007' -> {'C','D','E'}. Residue ranges are
    discarded -- this project reuses each UniProt entry's own full sequence
    as the generation target (documented scope simplification, same as task
    family A), so only which chain(s) a domain touches matters, not the
    exact crystallized span."""
    chains = set()
    for segment in raw_chain_field.split(","):
        chain = segment.split(":")[0].strip()
        if chain:
            chains.add(chain)
    return chains


def parse_cla_line(line: str) -> ScopeDomain | None:
    """One line of SCOPe's dir.cla file -> ScopeDomain, or None for
    malformed/comment lines."""
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None
    parts = line.split("\t")
    if len(parts) != 6:
        return None
    sid, pdbid, chain_field, _sccs, _sunid, key_values = parts
    kv: dict[str, str] = {}
    for pair in key_values.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            kv[key] = value
    return ScopeDomain(
        sid=sid,
        pdbid=pdbid,
        chains=frozenset(parse_scope_chains(chain_field)),
        fold_sunid=kv.get("cf"),
        superfamily_sunid=kv.get("sf"),
        family_sunid=kv.get("fa"),
    )


def parse_des_lines(lines) -> dict[str, str]:
    """dir.des file -> {sunid: description}, restricted to fold/superfamily/
    family node types (cl/dm/sp/px rows are skipped -- not classification
    levels this project targets)."""
    result: dict[str, str] = {}
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        sunid, node_type, _sccs, _sid, description = parts
        if node_type in LEVEL_KEYS:
            result[sunid] = description
    return result


def build_sifts_map(rows: list[tuple[str, str, str]]) -> dict[tuple[str, str], set[str]]:
    """rows: (pdbid, chain, uniprot_accession) from SIFTS ->
    {(pdbid, chain): {accessions}}. A (pdbid, chain) can map to more than
    one accession across different residue ranges (fusion constructs); all
    are kept, not just the first."""
    result: dict[tuple[str, str], set[str]] = {}
    for pdbid, chain, accession in rows:
        result.setdefault((pdbid, chain), set()).add(accession)
    return result


def resolve_domain_to_entries(
    domain: ScopeDomain,
    sifts_map: dict[tuple[str, str], set[str]],
    reviewed_accessions: set[str],
) -> set[str]:
    """A domain's chains -> the set of reviewed UniProt accessions any of
    those chains map to, via SIFTS. Unreviewed (TrEMBL) hits are dropped --
    SIFTS covers both, this project is SwissProt-only."""
    entries: set[str] = set()
    for chain in domain.chains:
        for accession in sifts_map.get((domain.pdbid, chain), set()):
            if accession in reviewed_accessions:
                entries.add(accession)
    return entries


def build_scop_triples(
    domain: ScopeDomain,
    entries: set[str],
    descriptions: dict[str, str],
) -> list[tuple[str, str, str]]:
    """A resolved domain -> (entry, relation, value) triples for whichever
    of fold/superfamily/family sunids resolve to a known description."""
    triples: list[tuple[str, str, str]] = []
    level_sunids = {
        "cf": domain.fold_sunid,
        "sf": domain.superfamily_sunid,
        "fa": domain.family_sunid,
    }
    for level, sunid in level_sunids.items():
        if sunid is None or sunid not in descriptions:
            continue
        value = descriptions[sunid]
        if value in ARTIFACT_VALUES:
            continue
        relation = LEVEL_TO_RELATION[level]
        for entry in entries:
            triples.append((entry, relation, value))
    return triples
