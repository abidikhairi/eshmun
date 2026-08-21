"""
Build the all-organism annotation-direction examples: sequence -> family/
function/process/localization/catalytic activity, for all reviewed SwissProt
entries. Reuses build_task_a_annotation.py's logic (rng_for,
load_all_triples_by_entry, build_task_a_annotation_example) unchanged --
only the data sources differ (all_organism_kg.parquet / all_organism_split.csv
/ the full FASTA instead of the human-only sequence table). The DeepSeek
instruction pool is also reused as-is: its phrasings ("What family does
{protein} belong to?") never reference organism, so there's nothing
organism-specific to regenerate.

Usage:
    python3 scripts/data/thinking/build_all_organism_annotation.py
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_task_a_annotation import (
    TARGET_RELATIONS,
    build_task_a_annotation_example,
    load_all_triples_by_entry,
    rng_for,
)
from fasta import extract_sequences

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ALL_ORGANISM_KG_FILE = os.path.join(PROCESSED_DIR, "all_organism_kg.parquet")
SPLIT_FILE = os.path.join(PROCESSED_DIR, "all_organism_split.csv")
FASTA_FILE = os.path.join(_HERE, "..", "..", "..", "data", "uniprot_sprot.fasta")
POOL_FILE = os.path.join(PROCESSED_DIR, "instruction_pool.json")

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "all_organism_annotation_pairs.jsonl")


def main() -> None:
    kg = pd.read_parquet(ALL_ORGANISM_KG_FILE)
    split_df = pd.read_csv(SPLIT_FILE)
    entry_to_split = dict(zip(split_df["entry"], split_df["split"]))
    split_entries = set(entry_to_split.keys())

    with open(FASTA_FILE) as f:
        seq_map = extract_sequences(f, split_entries)
    print(f"sequences for {len(seq_map)}/{len(split_entries)} split entries")

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
            rng = rng_for("all_organism_annotation", entry, target_relation)
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
