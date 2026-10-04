"""Tokenization for the Eshmun models.

`EshmunTokenizer` is the tokenizer itself; `load_tokenizer` is the one loading
path (decoder repair, merge probe, chat-template guard); the `spans` names are
the vocabulary of the models' responses -- tags, residue markers, and the
parsing helpers every consumer reads them back with.
"""

from eshmun.tokenization.tokenizer import EshmunTokenizer
from eshmun.tokenization.loading import load_tokenizer, tag_id
from eshmun.tokenization.spans import (
    MARKER,
    PROTEIN_END,
    PROTEIN_START,
    RESIDUES,
    TERMINATORS,
    THINK_END,
    THINK_START,
    extract_reasoning,
    extract_sequence,
    find_span,
    longest_marked_run,
    render_prompt,
    residue_runs,
    strip_terminators,
)

__all__ = [
    "EshmunTokenizer",
    "MARKER",
    "PROTEIN_END",
    "PROTEIN_START",
    "RESIDUES",
    "TERMINATORS",
    "THINK_END",
    "THINK_START",
    "extract_reasoning",
    "extract_sequence",
    "find_span",
    "load_tokenizer",
    "longest_marked_run",
    "render_prompt",
    "residue_runs",
    "strip_terminators",
    "tag_id",
]
