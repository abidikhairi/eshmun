"""
Generate a diverse pool of annotation-direction instruction phrasings via
DeepSeek, one pool per target relation (family/function/process/
localization/catalytic activity). Improves on the pilot's one-call-per-
example approach (documented diversity collapse: ~120 unique phrasings out
of 400 calls) by asking each call for many distinct phrasings at once, then
sampling from the resulting pool per training example -- far fewer calls,
genuinely more diversity per call (see cost estimate in conversation record).

DeepSeek is given only the task name, never the specific answer value --
structurally impossible to leak an answer into a question it was never
shown (same principle as the pilot's humanize_instructions.py).

Uses deepseek-v4-flash in non-thinking mode (deepseek-chat, used by the
pilot, is deprecated; verified deepseek-v4-flash is live and
extra_body={"thinking": {"type": "disabled"}} gives clean, fast completions
instead of burning the token budget on hidden reasoning).

Usage:
    python3 scripts/data/thinking/generate_instruction_pool.py
"""

import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instruction_pool import parse_numbered_list
from validate import validate_instruction

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "instruction_pool.json")

MODEL = "deepseek-v4-flash"
PHRASINGS_PER_CALL = 40
CALLS_PER_TASK = 3

TARGET_TO_TASK = {
    "member_of": "family",
    "has_function": "function",
    "involved_in": "process",
    "located_in": "subcellular localization",
    "catalyzes": "catalytic activity (EC number)",
}

SYSTEM_PROMPT = """You are helping build an instruction-tuning dataset for a protein language model.
For the given TASK, generate {n} distinct, natural, human-sounding INSTRUCTIONS
(requests or questions) that a person might type when asking about that
property of a protein.

Rules:
- Output ONLY a numbered list, one instruction per line ("1. ...", "2. ...").
  No preamble, no explanation.
- Every instruction must include the literal placeholder {{protein}} exactly
  once, wherever the protein sequence itself would be referenced.
- Do not state or guess any specific answer (e.g. a specific family name,
  function, location, or EC number) -- you are not given one, and must not
  invent one.
- Make all {n} instructions genuinely different from each other: vary
  sentence structure, phrasing, formality, and length. Avoid minor
  rewordings of the same sentence.
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


def build_pool_for_task(client: OpenAI, task: str) -> list[str]:
    pool: list[str] = []
    for _ in range(CALLS_PER_TASK):
        raw = call_pool(client, task, PHRASINGS_PER_CALL)
        candidates = parse_numbered_list(raw)
        for candidate in candidates:
            if validate_instruction(candidate).valid:
                pool.append(candidate)
    # dedupe while preserving order
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
        pool = build_pool_for_task(client, task)
        pools[relation] = pool
        print(f"{relation} ({task}): {len(pool)} unique valid phrasings")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(pools, f, indent=2)
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
