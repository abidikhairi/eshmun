"""
Pure logic for parsing UniProt REST TSV field values (ft_motif, ft_domain,
ft_region, ec, go_p/go_f/go_c) into clean value lists for the KG.

No network calls, no file I/O.
"""

import re

_NOTE_RE = re.compile(r'/note="([^"]*)"')
_GO_ID_SUFFIX_RE = re.compile(r"\s*\[GO:\d+\]$")


def parse_note_field(raw: str) -> list[str]:
    """UniProt's ft_motif/ft_domain/ft_region fields all pack one or more
    features into a single string using the same layout, e.g.:
        'MOTIF 79..83; /note="SUMO-interacting motif"; /evidence="..."'
        'DOMAIN 44..89; /note="Gla"; /evidence="..."; DOMAIN 108..186; /note="Kringle 1"; ...'

    Only the /note text (the human-readable name) is kept -- position ranges
    and evidence codes aren't useful as a reasoning-trace value.
    """
    if not raw or not raw.strip():
        return []
    return _NOTE_RE.findall(raw)


def parse_motif_names(raw: str) -> list[str]:
    """Replaces the pilot's has_motif, which stored the raw amino-acid
    subsequence instead of a name. Thin, separately-named wrapper around
    parse_note_field for call-site clarity (motif vs. domain vs. region)."""
    return parse_note_field(raw)


def parse_ec_numbers(raw: str) -> list[str]:
    """UniProt's ec field is '; '-separated EC numbers, e.g.
    '3.1.3.16; 3.1.3.48'. Partial EC numbers (undetermined sub-subclass,
    e.g. '3.6.1.-') are kept as-is -- that dash is a real part of the EC
    notation, not missing data.
    """
    if not raw or not raw.strip():
        return []
    return [t.strip() for t in raw.split(";") if t.strip()]


def parse_go_terms(raw: str) -> list[str]:
    """UniProt's go_p/go_f/go_c fields are '; '-separated "term name
    [GO:0006953]" entries -- strips the trailing GO ID, keeping just the
    human-readable term name (the ID isn't useful as a reasoning-trace
    value, and would need cross-referencing to mean anything to the model)."""
    if not raw or not raw.strip():
        return []
    items = []
    for segment in raw.split(";"):
        name = _GO_ID_SUFFIX_RE.sub("", segment.strip()).strip()
        if name:
            items.append(name)
    return items
