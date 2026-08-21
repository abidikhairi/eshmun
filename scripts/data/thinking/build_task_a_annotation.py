"""
Build task family A's annotation-direction examples: sequence -> family/
function/process/localization/catalytic activity, using the DeepSeek
instruction pool (instruction_pool.json) and the annotation KG.

Reasoning cites the entry's OTHER real facts (the remaining target relations
plus domain/region/motif/length) -- legitimate, since annotation's input is
a real, existing sequence (same reasoning as PPI/SCOP annotation).
interacts_with is excluded from context, same rationale as
build_ppi_dataset.py: a specific partner identity isn't a generalizable
single-protein trait, and citing it invites a shortcut ("already interacts
with lots of things") instead of content-based reasoning.

Usage:
    python3 scripts/data/thinking/build_task_a_annotation.py
"""

import hashlib
import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning import build_reasoning_block, build_response_text, encode_sequence, select_context_triples

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ANNOTATION_KG_FILE = os.path.join(PROCESSED_DIR, "annotation_kg.parquet")
SPLIT_FILE = os.path.join(PROCESSED_DIR, "annotation_split.csv")
SEQ_FILE = os.path.join(RAW_DIR, "homosapiens-sequences.tsv")
POOL_FILE = os.path.join(PROCESSED_DIR, "instruction_pool.json")

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "task_a_annotation_pairs.jsonl")

TARGET_RELATIONS = ["member_of", "has_function", "involved_in", "located_in", "catalyzes"]
CONTEXT_RELATIONS = TARGET_RELATIONS + ["has_domain", "has_region", "has_motif", "has_length"]
MAX_CONTEXT_PER_RELATION = 3


def rng_for(*parts: str) -> random.Random:
    seed = int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def triples_by_relation_for_entry(kg: pd.DataFrame, entry: str) -> dict[str, list[str]]:
    """Single-entry lookup, used directly in tests. main() below does NOT
    call this per-entry against the full KG (that's O(entries * KG size) --
    ~12,644 x 529,173 rows, which is what made the first full-scale run take
    >5 minutes without finishing); it pre-groups once via
    load_all_triples_by_entry instead and does O(1) dict lookups."""
    sub = kg[(kg["entry"] == entry) & (kg["relation"].isin(CONTEXT_RELATIONS))]
    result: dict[str, list[str]] = {}
    for relation, group in sub.groupby("relation"):
        result[relation] = group["value"].tolist()
    return result


def load_all_triples_by_entry(kg: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Pre-groups the whole KG once: {entry: {relation: [values]}}, restricted
    to CONTEXT_RELATIONS. O(KG size) total instead of O(entries * KG size)."""
    sub = kg[kg["relation"].isin(CONTEXT_RELATIONS)]
    result: dict[str, dict[str, list[str]]] = {}
    for entry, entry_group in sub.groupby("entry"):
        result[entry] = {
            relation: group["value"].tolist() for relation, group in entry_group.groupby("relation")
        }
    return result


def build_task_a_annotation_example(
    entry: str, seq: str, triples: dict[str, list[str]], target_relation: str,
    instruction_pool: list[str], rng: random.Random,
) -> dict | None:
    target_values = triples.get(target_relation)
    if not target_values:
        return None

    context = select_context_triples(triples, target_relation=target_relation, rng=rng, max_per_relation=MAX_CONTEXT_PER_RELATION)
    reasoning = build_reasoning_block(context, rng)
    answer = build_response_text(target_relation, target_values)
    instruction_template = rng.choice(instruction_pool)
    instruction = instruction_template.format(protein=encode_sequence(seq))

    return {
        "entry": entry, "target_relation": target_relation,
        "instruction": instruction, "reasoning": reasoning, "answer": answer,
    }


def main() -> None:
    kg = pd.read_parquet(ANNOTATION_KG_FILE)
    split_df = pd.read_csv(SPLIT_FILE)
    entry_to_split = dict(zip(split_df["entry"], split_df["split"]))

    seq_df = pd.read_csv(SEQ_FILE, sep="\t", dtype=str)
    seq_map = dict(zip(seq_df["Entry"], seq_df["Sequence"]))

    with open(POOL_FILE) as f:
        pools = json.load(f)

    triples_by_entry = load_all_triples_by_entry(kg)

    examples = []
    for entry, split_name in entry_to_split.items():
        seq = seq_map.get(entry)
        if seq is None:
            continue
        triples = triples_by_entry.get(entry, {})
        for target_relation in TARGET_RELATIONS:
            rng = rng_for("task_a_annotation", entry, target_relation)
            example = build_task_a_annotation_example(
                entry, seq, triples, target_relation, pools[target_relation], rng
            )
            if example is not None:
                example["split"] = split_name
                examples.append(example)

    pd.DataFrame(examples).to_json(OUTPUT_FILE, orient="records", lines=True)

    print(f"total examples: {len(examples)}")
    df = pd.DataFrame(examples)
    print(df.groupby(["split", "target_relation"]).size())
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
