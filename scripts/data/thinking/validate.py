"""
Heuristic validation of DeepSeek-humanized instructions.

Deliberately cheap and deterministic (no second LLM call): these are the
gates that decide whether a generated instruction makes it into the dataset,
so they must be fast, free, and reproducible.

Ported verbatim from the InstructProtein pilot's validate.py (external repo)
-- this logic has no dependency on tokenizer/architecture, so it carries over
unchanged.
"""

import re
from dataclasses import dataclass, field

PLACEHOLDER = "{protein}"

MIN_WORDS = 3
MAX_WORDS = 60

REFUSAL_PATTERNS = [
    r"\bas an ai\b",
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi don'?t have access\b",
    r"\bi'?m sorry\b",
    r"\bi am sorry\b",
    r"\blanguage model\b",
    r"\bi'?m not able to\b",
    r"\bi am not able to\b",
]

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


@dataclass
class ValidationResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)


def contains_placeholder(text: str, placeholder: str = PLACEHOLDER) -> bool:
    return placeholder in text


def is_non_empty(text: str) -> bool:
    return bool(text and text.strip())


def within_word_bounds(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> bool:
    word_count = len(text.split())
    return min_words <= word_count <= max_words


def contains_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text))


def contains_leaked_value(text: str, leaked_values: list[str]) -> bool:
    """True if any of the (excluded-from-the-question) answer values leaked into the text."""
    lowered = text.lower()
    return any(v.lower() in lowered for v in leaked_values if v)


def validate_instruction(
    text: str, leaked_values: list[str] | None = None, placeholder: str = PLACEHOLDER
) -> ValidationResult:
    reasons = []

    if not is_non_empty(text):
        reasons.append("empty")
        return ValidationResult(valid=False, reasons=reasons)

    if not contains_placeholder(text, placeholder):
        reasons.append("missing_placeholder")
    if not within_word_bounds(text):
        reasons.append("out_of_word_bounds")
    if contains_refusal(text):
        reasons.append("refusal")
    if leaked_values and contains_leaked_value(text, leaked_values):
        reasons.append("leaked_answer_value")

    return ValidationResult(valid=len(reasons) == 0, reasons=reasons)
