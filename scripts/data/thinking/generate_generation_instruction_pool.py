"""
Generate a diverse pool of generation-direction instruction phrasings via
DeepSeek, one pool per target relation, mirroring
generate_instruction_pool.py's annotation-direction approach.

Unlike annotation, there's no leakage risk to design around here -- the
property being designed for is the *given input*, stated openly in the
instruction, not a hidden answer -- so DeepSeek is simply asked to phrase
diverse "design a sequence with property X" requests using the {joined}
placeholder (matching reasoning.build_generation_instruction_from_pool's
.format(joined=...) usage) instead of {protein}.

Usage:
    python3 scripts/data/thinking/generate_generation_instruction_pool.py
"""

import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instruction_pool import clean_phrasing, matches_any_keyword, parse_numbered_list
from validate import validate_instruction

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "generation_instruction_pool.json")

MODEL = "deepseek-v4-flash"
PHRASINGS_PER_CALL = 40
CALLS_PER_TASK = 3
PLACEHOLDER = "{joined}"

TARGET_TO_TASK = {
    "member_of": "family",
    "has_function": "function",
    "involved_in": "process",
    "located_in": "subcellular localization",
    "catalyzes": "catalytic activity (EC number)",
}

# involved_in (biological process) is broader than catalysis -- observed
# ~11% of a first raw pool drifting into catalysis/enzyme framing ("catalyze
# {joined}", "an enzyme that speeds up {joined}"), which is wrong for
# non-enzymatic processes like apoptosis or cell adhesion. catalyzes already
# has its own dedicated pool, so this framing is off-relation here, not just
# imprecise.
EXCLUDED_KEYWORDS_BY_RELATION = {
    "involved_in": ["catalyz", "enzyme", "enzymatic", "biocatalyst", "reaction"],
}

SYSTEM_PROMPT = """You are helping build an instruction-tuning dataset for a protein language model.
For the given TASK (a property of a protein), generate {n} distinct, natural,
human-sounding INSTRUCTIONS (requests) asking the model to design or
generate a protein sequence that has that property.

Rules:
- Output ONLY a numbered list, one instruction per line ("1. ...", "2. ...").
  No preamble, no explanation.
- Every instruction must include the literal placeholder {{joined}} exactly
  once, wherever the specific property value would be stated (e.g. a family
  name, a molecular function, a subcellular location, or an EC number).
- Make all {n} instructions genuinely different from each other: vary
  sentence structure (imperative, question, polite request), phrasing,
  formality, and length. Avoid minor rewordings of the same sentence.
- One sentence each, plain language, no jargon inflation."""


def call_pool(client: OpenAI, task: str, n: int) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
            {"role": "user", "content": f"TASK: {task}"},
        ],
        temperature=1.0,
        max_tokens=n * 40,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content
    return content or ""


def build_pool_for_task(client: OpenAI, relation: str, task: str) -> list[str]:
    excluded_keywords = EXCLUDED_KEYWORDS_BY_RELATION.get(relation, [])
    pool: list[str] = []
    for _ in range(CALLS_PER_TASK):
        raw = call_pool(client, task, PHRASINGS_PER_CALL)
        candidates = parse_numbered_list(raw)
        for candidate in candidates:
            cleaned = clean_phrasing(candidate)
            if not validate_instruction(cleaned, placeholder=PLACEHOLDER).valid:
                continue
            if excluded_keywords and matches_any_keyword(cleaned, excluded_keywords):
                continue
            pool.append(cleaned)
    seen = set()
    deduped = []
    for item in pool:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    pools: dict[str, list[str]] = {}
    for relation, task in TARGET_TO_TASK.items():
        pool = build_pool_for_task(client, relation, task)
        pools[relation] = pool
        print(f"{relation} ({task}): {len(pool)} unique valid phrasings")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(pools, f, indent=2)
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
