"""
Build the all-organism generation KG: category-level aggregate statistics
(same causal-inversion fix, same train-split-only aggregation to avoid the
test-leakage gap already fixed for task family A), for the all-organism KG
instead of the human-only one.

Usage:
    python3 scripts/data/thinking/build_all_organism_generation_kg.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_generation_kg import AGGREGABLE_RELATIONS, TARGET_RELATIONS, build_generation_kg

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ALL_ORGANISM_KG_FILE = os.path.join(PROCESSED_DIR, "all_organism_kg.parquet")
SPLIT_FILE = os.path.join(PROCESSED_DIR, "all_organism_split.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "all_organism_generation_kg.parquet")


def main() -> None:
    kg = pd.read_parquet(ALL_ORGANISM_KG_FILE)

    split_df = pd.read_csv(SPLIT_FILE)
    train_entries = set(split_df[split_df["split"] == "train"]["entry"])
    train_kg = kg[kg["entry"].isin(train_entries)]
    print(f"train entries: {len(train_entries)} (of {kg['entry'].nunique()} total in the KG)")

    # Same 9-relation schema as task family A's KG (minus interacts_with),
    # so TARGET_RELATIONS/AGGREGABLE_RELATIONS apply unchanged.
    generation_kg = build_generation_kg(train_kg, target_relations=TARGET_RELATIONS, aggregable_relations=AGGREGABLE_RELATIONS)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    generation_kg.to_parquet(OUTPUT_FILE, index=False)

    print(f"\ntotal (category, context) rows: {len(generation_kg)}")
    print(f"unique categories: {generation_kg.groupby(['target_relation', 'target_value']).ngroups}")
    print("categories per target_relation:")
    print(generation_kg.groupby("target_relation")["target_value"].nunique())
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
