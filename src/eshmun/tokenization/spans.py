"""The vocabulary of an Eshmun response, and the helpers that read it back.

Generation, the CLI and the InstructProtein baseline all parse the same tags
and residue markers, so the constants and the parsing primitives live here and
the scripts import them rather than each carrying a copy. `find_span` and
`residue_runs` are the primitives; `extract_reasoning` and `extract_sequence`
are the two readings built on them; `render_prompt` is the other end of the
loop, the prompt a response is a continuation of.
"""

from __future__ import annotations

from typing import cast

from eshmun.tokenization.tokenizer import EshmunTokenizer

THINK_START = "<think>"
THINK_END = "</think>"
PROTEIN_START = "<protein>"
PROTEIN_END = "</protein>"

# The one-letter amino-acid alphabet. Residue tokens carry a `Ƥ` prefix that is
# not a residue, so filtering to this set removes it too.
RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")

# The marker on every residue token. It appears on residues and nowhere else,
# which is what makes it usable as a signal when the tags are missing.
MARKER = "Ƥ"

# What the decoder leaves on the end of a finished sequence.
TERMINATORS = ("</s>", "<pad>")


def find_span(text: str, start: str, end: str) -> str | None:
    """Text between `start` and `end`, or None when `start` never appears.

    An `end` that never arrives still yields the remainder: generation hits
    the token cap often enough that discarding those spans would throw away
    sequences that are otherwise perfectly usable.
    """
    at = text.find(start)
    if at == -1:
        return None
    at += len(start)
    stop = text.find(end, at)
    return text[at:] if stop == -1 else text[at:stop]


def strip_terminators(text: str) -> str:
    """Drop trailing EOS/pad markers, leaving the rest of the body alone."""
    while True:
        for token in TERMINATORS:
            if text.endswith(token):
                text = text[: -len(token)]
                break
        else:
            return text


def residue_runs(text: str) -> list[str]:
    """Contiguous runs of `Ƥ`-prefixed residues, reduced to their letters.

    A run ends at the first character that is not a marked residue, so prose
    between two stretches of sequence separates them rather than joining them.
    """
    runs: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(text):
        if (
            text[index] == MARKER
            and index + 1 < len(text)
            and text[index + 1] in RESIDUES
        ):
            current.append(text[index + 1])
            index += 2
            continue
        if current:
            runs.append("".join(current))
            current = []
        index += 1
    if current:
        runs.append("".join(current))
    return runs


def extract_reasoning(text: str) -> str | None:
    """The text between `<think>` and `</think>`, or None when it never opens."""
    span = find_span(text, THINK_START, THINK_END)
    return span.strip() if span is not None else None


def longest_marked_run(text: str) -> str | None:
    """The longest run of marked residues, or None when there is no run.

    This is the parse for a response with no `<protein>` span to take: an
    early checkpoint emits its residues bare, and InstructProtein's prompt owns
    the opening tag, so its responses never carry one. The longest run is taken
    rather than every run joined, so fragments separated by prose do not become
    one chimeric sequence.
    """
    runs = residue_runs(text)
    return max(runs, key=len) if runs else None


def extract_sequence(text: str) -> str | None:
    """Residues from the `<protein>` span, or from the bare marked runs.

    A span that is present but holds no residues yields "", which keeps
    "the model wrote nothing valid" distinct from "the model never got there".

    With no `<protein>` tag anywhere, the longest marked run stands in: an
    early checkpoint emits its residues bare, and the marker still identifies
    them exactly where filtering the whole response would sweep up prose. The
    longest run is taken rather than every run joined, so two fragments
    separated by prose do not become one chimeric sequence.
    """
    span = find_span(text, PROTEIN_START, PROTEIN_END)
    if span is not None:
        return "".join(character for character in span if character in RESIDUES)
    return longest_marked_run(text)


def render_prompt(tokenizer: EshmunTokenizer, prompt: str) -> str:
    """Render one prompt exactly as the training rows render it."""
    return cast(
        str,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
    )
