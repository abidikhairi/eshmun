"""
Build task family A's generation-direction examples: family/function/
process/localization/catalytic activity -> sequence, using the leak-fixed
generation_kg.parquet (category stats from train-split entries only) and
each entry's own real sequence as the generation target (documented scope
simplification, same as SCOP/PPI generation directions -- revisit per-
category exemplar aggregation at full scale per PLAN.md's original open
item).

Reasoning cites CATEGORY-level stats, not instance-specific facts -- the
causal-inversion fix (see conversation record). An entry's own sequence is
still a valid generation target regardless of which split it's in; only the
reasoning content is restricted to train-derived statistics.

Instructions are sampled from the DeepSeek generation_instruction_pool.json
(generate_generation_instruction_pool.py) rather than the ~3-variant
GENERATION_INSTRUCTION_TEMPLATES dict in reasoning.py -- see conversation
record for why generation direction didn't originally get the same
humanization treatment as annotation, and why that reasoning turned out to
be a weaker argument for stopping at 3 templates than it first looked.

Usage:
    python3 scripts/data/thinking/build_task_a_generation.py
"""

import hashlib
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning import (
    GENERATION_CONTEXT_SUBJECTS,
    Triple,
    build_generation_instruction_from_pool,
    build_reasoning_block,
    encode_sequence,
    with_subject,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ANNOTATION_KG_FILE = os.path.join(PROCESSED_DIR, "annotation_kg.parquet")
SPLIT_FILE = os.path.join(PROCESSED_DIR, "annotation_split.csv")
SEQ_FILE = os.path.join(RAW_DIR, "homosapiens-sequences.tsv")
GENERATION_KG_FILE = os.path.join(PROCESSED_DIR, "generation_kg.parquet")
POOL_FILE = os.path.join(PROCESSED_DIR, "generation_instruction_pool.json")

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "task_a_generation_pairs.jsonl")


def rng_for(*parts: str) -> random.Random:
    seed = int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def build_entries_by_category(annotation_kg: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Pre-groups the whole KG once: {(relation, value): {entries}}. Avoids
    re-filtering the full ~529k-row KG per category (4,575 categories x
    529,173 rows was the same O(n*m) bottleneck fixed in
    build_task_a_annotation.py's load_all_triples_by_entry)."""
    result: dict[tuple[str, str], set[str]] = {}
    for (relation, value), group in annotation_kg.groupby(["relation", "value"]):
        result[(relation, value)] = set(group["entry"])
    return result


def build_task_a_generation_example(
    entry: str, seq: str, target_relation: str, target_value: str,
    category_context_rows: pd.DataFrame, instruction_pool: list[str], rng: random.Random,
) -> dict:
    triples = with_subject(
        [
            Triple(relation=row["context_relation"], value=row["context_value"])
            for _, row in category_context_rows.iterrows()
        ],
        GENERATION_CONTEXT_SUBJECTS[target_relation],
    )
    reasoning = build_reasoning_block(triples, rng)
    instruction = build_generation_instruction_from_pool([target_value], instruction_pool, rng)
    answer = encode_sequence(seq)

    return {
        "entry": entry, "target_relation": target_relation, "target_value": target_value,
        "instruction": instruction, "reasoning": reasoning, "answer": answer,
    }


def main() -> None:
    annotation_kg = pd.read_parquet(ANNOTATION_KG_FILE)
    generation_kg = pd.read_parquet(GENERATION_KG_FILE)
    split_df = pd.read_csv(SPLIT_FILE)
    entry_to_split = dict(zip(split_df["entry"], split_df["split"]))

    seq_df = pd.read_csv(SEQ_FILE, sep="\t", dtype=str)
    seq_map = dict(zip(seq_df["Entry"], seq_df["Sequence"]))

    with open(POOL_FILE) as f:
        pools = json.load(f)

    valid_categories = generation_kg.groupby(["target_relation", "target_value"])
    print(f"categories: {valid_categories.ngroups}")
    entries_by_category = build_entries_by_category(annotation_kg)

    examples = []
    for (target_relation, target_value), context_rows in valid_categories:
        category_entries = entries_by_category.get((target_relation, target_value), set())
        for entry in category_entries:
            seq = seq_map.get(entry)
            if seq is None or entry not in entry_to_split:
                continue
            rng = rng_for("task_a_generation", entry, target_relation, target_value)
            example = build_task_a_generation_example(
                entry, seq, target_relation, target_value, context_rows, pools[target_relation], rng
            )
            example["split"] = entry_to_split[entry]
            examples.append(example)

    pd.DataFrame(examples).to_json(OUTPUT_FILE, orient="records", lines=True)

    print(f"total examples: {len(examples)}")
    df = pd.DataFrame(examples)
    print(df.groupby(["split", "target_relation"]).size())
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
