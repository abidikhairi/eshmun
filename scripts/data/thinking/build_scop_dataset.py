"""
Build task family B's annotation (sequence -> scop_fold/scop_superfamily/
scop_family) and generation (fold/superfamily/family -> sequence) examples.

Annotation direction: reasoning cites the OTHER 2 SCOP levels for the same
entry -- real, known facts about an already-solved structure, no
causal-inversion issue (same reasoning as task family A's annotation
direction).

Generation direction: reasoning cites CATEGORY-level aggregate stats (the
same causal-inversion fix as task family A's generation direction), computed
from TRAIN-SPLIT ENTRIES ONLY -- unlike build_generation_kg.py's existing
task-family-A output (data/thinking/processed/generation_kg.parquet), which
was built from the full KG across all splits and therefore lets test-set
entries' own facts leak into "typical category" stats that could end up
informing train-split reasoning traces. Fixed here; task family A's
generation_kg.parquet needs the same fix before it's used to build actual
generation examples (not done yet, so no bad training data exists from it
so far -- just the intermediate aggregate-stats artifact needs rebuilding).

Usage:
    python3 scripts/data/thinking/build_scop_dataset.py
"""

import hashlib
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_generation_kg import build_generation_kg
from fasta import extract_sequences
from reasoning import (
    GENERATION_CONTEXT_SUBJECTS,
    Triple,
    build_generation_instruction,
    build_reasoning_block,
    build_response_text,
    encode_sequence,
    select_context_triples,
    with_subject,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SCOP_KG_FILE = os.path.join(PROCESSED_DIR, "scop_kg.parquet")
SPLIT_FILE = os.path.join(PROCESSED_DIR, "scop_split.csv")
FASTA_FILE = os.path.join(_HERE, "..", "..", "..", "data", "uniprot_sprot.fasta")
GENERATION_KG_FILE = os.path.join(PROCESSED_DIR, "scop_generation_kg.parquet")

ANNOTATION_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "scop_annotation_pairs.jsonl")
GENERATION_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "scop_generation_pairs.jsonl")

TARGET_RELATIONS = ["scop_fold", "scop_superfamily", "scop_family"]
MIN_GROUP_SIZE = 5
TOP_K_CONTEXT = 3

ANNOTATION_INSTRUCTION_TEMPLATES = {
    "scop_fold": [
        "What SCOP fold does {protein} adopt?",
        "Identify the structural fold of {protein}.",
        "Based on its structure, what fold does {protein} belong to?",
    ],
    "scop_superfamily": [
        "What SCOP superfamily does {protein} belong to?",
        "Identify the structural superfamily of {protein}.",
        "Which superfamily, per SCOP classification, does {protein} belong to?",
    ],
    "scop_family": [
        "What SCOP family does {protein} belong to?",
        "Identify the structural family (SCOP classification) of {protein}.",
        "Based on its structure, which family does {protein} belong to?",
    ],
}


def rng_for(*parts: str) -> random.Random:
    seed = int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def triples_by_relation_for_entry(kg: pd.DataFrame, entry: str) -> dict[str, list[str]]:
    sub = kg[kg["entry"] == entry]
    result: dict[str, list[str]] = {}
    for relation, group in sub.groupby("relation"):
        result[relation] = group["value"].tolist()
    return result


def build_scop_annotation_example(
    entry: str, seq: str, triples: dict[str, list[str]], target_relation: str, rng: random.Random
) -> dict | None:
    target_values = triples.get(target_relation)
    if not target_values:
        return None

    context = select_context_triples(triples, target_relation=target_relation, rng=rng, max_per_relation=TOP_K_CONTEXT)
    reasoning = build_reasoning_block(context, rng)
    answer = build_response_text(target_relation, target_values)
    instruction_template = rng.choice(ANNOTATION_INSTRUCTION_TEMPLATES[target_relation])
    instruction = instruction_template.format(protein=encode_sequence(seq))

    return {
        "entry": entry, "target_relation": target_relation,
        "instruction": instruction, "reasoning": reasoning, "answer": answer,
    }


def build_scop_generation_example(
    entry: str, seq: str, target_relation: str, target_value: str,
    category_context_rows: pd.DataFrame, rng: random.Random,
) -> dict:
    triples = with_subject(
        [
            Triple(relation=row["context_relation"], value=row["context_value"])
            for _, row in category_context_rows.iterrows()
        ],
        GENERATION_CONTEXT_SUBJECTS[target_relation],
    )
    reasoning = build_reasoning_block(triples, rng)
    instruction_template_values = [target_value]
    instruction = build_generation_instruction(target_relation, instruction_template_values, rng)
    answer = encode_sequence(seq)

    return {
        "entry": entry, "target_relation": target_relation, "target_value": target_value,
        "instruction": instruction, "reasoning": reasoning, "answer": answer,
    }


def main() -> None:
    scop_kg = pd.read_parquet(SCOP_KG_FILE)
    split_df = pd.read_csv(SPLIT_FILE)
    split_entries = set(split_df["entry"])

    with open(FASTA_FILE) as f:
        seq_map = extract_sequences(f, split_entries)
    print(f"sequences for {len(seq_map)}/{len(split_entries)} split entries")

    entry_to_split = dict(zip(split_df["entry"], split_df["split"]))

    # --- Annotation direction ---
    annotation_examples = []
    for entry in split_entries:
        seq = seq_map.get(entry)
        if seq is None:
            continue
        triples = triples_by_relation_for_entry(scop_kg, entry)
        for target_relation in TARGET_RELATIONS:
            rng = rng_for("scop_annotation", entry, target_relation)
            example = build_scop_annotation_example(entry, seq, triples, target_relation, rng)
            if example is not None:
                example["split"] = entry_to_split[entry]
                annotation_examples.append(example)
    print(f"annotation examples: {len(annotation_examples)}")

    # --- Generation direction: category stats from TRAIN entries only ---
    train_entries = {e for e, s in entry_to_split.items() if s == "train"}
    train_kg = scop_kg[scop_kg["entry"].isin(train_entries)]
    generation_kg = build_generation_kg(
        train_kg, target_relations=TARGET_RELATIONS, aggregable_relations=TARGET_RELATIONS,
        top_k=TOP_K_CONTEXT, min_group_size=MIN_GROUP_SIZE,
    )
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    generation_kg.to_parquet(GENERATION_KG_FILE, index=False)
    print(f"generation KG (train-only categories): {generation_kg.groupby(['target_relation', 'target_value']).ngroups} categories")

    valid_categories = generation_kg.groupby(["target_relation", "target_value"])
    generation_examples = []
    for (target_relation, target_value), context_rows in valid_categories:
        category_entries = set(scop_kg[(scop_kg["relation"] == target_relation) & (scop_kg["value"] == target_value)]["entry"])
        for entry in category_entries:
            seq = seq_map.get(entry)
            if seq is None or entry not in entry_to_split:
                continue
            rng = rng_for("scop_generation", entry, target_relation, target_value)
            example = build_scop_generation_example(entry, seq, target_relation, target_value, context_rows, rng)
            example["split"] = entry_to_split[entry]
            generation_examples.append(example)
    print(f"generation examples: {len(generation_examples)}")

    pd.DataFrame(annotation_examples).to_json(ANNOTATION_OUTPUT_FILE, orient="records", lines=True)
    pd.DataFrame(generation_examples).to_json(GENERATION_OUTPUT_FILE, orient="records", lines=True)
    print(f"saved to {ANNOTATION_OUTPUT_FILE}")
    print(f"saved to {GENERATION_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
