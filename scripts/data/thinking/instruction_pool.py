"""
Pure logic for parsing DeepSeek's numbered-list instruction-pool responses.

No network calls, no file I/O.
"""

import re

_NUMBERED_ITEM_RE = re.compile(r"^\d+[.)]\s*(.+)$")


def parse_numbered_list(text: str) -> list[str]:
    """'1. What family does {protein} belong to?\n2. Tell me about {protein}.'
    -> ['What family does {protein} belong to?', 'Tell me about {protein}.']

    Lines that don't match the "N. " / "N) " numbering pattern are skipped
    (e.g. a stray blank line or preamble the model added despite
    instructions not to)."""
    items = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _NUMBERED_ITEM_RE.match(line)
        if match:
            items.append(match.group(1).strip())
    return items


_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,?!])")


def clean_phrasing(text: str) -> str:
    """Strips a stray space before trailing punctuation, e.g.
    'localize to the {joined} .' -> 'localize to the {joined}.' -- an
    observed DeepSeek formatting quirk (roughly a third of one relation's
    raw pool had it), not semantically meaningful."""
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)


def matches_any_keyword(text: str, keywords: list[str]) -> bool:
    """Case-insensitive substring check against a keyword list -- used to
    filter phrasings that drifted semantically off-relation (e.g. DeepSeek
    framing some "involved_in" [biological process] phrasings as
    catalysis/enzymatic-reaction requests, which is wrong for non-enzymatic
    processes like apoptosis or cell adhesion)."""
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)
