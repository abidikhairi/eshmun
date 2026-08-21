"""
Build the all-organism generation-direction examples: family/function/
process/localization/catalytic activity -> sequence, for all reviewed
SwissProt entries. Reuses build_task_a_generation.py's rng_for and
build_task_a_generation_example unchanged, but NOT build_entries_by_category
-- that groups the *entire* KG by (relation, value), building an entry-set
for every distinct value (member_of/has_function/involved_in alone have
hundreds of thousands of distinct values at this scale, most singleton/tiny
and irrelevant), which OOM-killed the first attempt here. Restricted below
to only the ~24,584 categories that survived generation_kg's min-group-size
filter. Output is also streamed line-by-line instead of accumulated into one
big list + DataFrame, for the same reason (millions of examples expected).

Usage:
    python3 scripts/data/thinking/build_all_organism_generation.py
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_task_a_generation import build_task_a_generation_example, rng_for
from fasta import extract_sequences

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ALL_ORGANISM_KG_FILE = os.path.join(PROCESSED_DIR, "all_organism_kg.parquet")
SPLIT_FILE = os.path.join(PROCESSED_DIR, "all_organism_split.csv")
FASTA_FILE = os.path.join(_HERE, "..", "..", "..", "data", "uniprot_sprot.fasta")
GENERATION_KG_FILE = os.path.join(PROCESSED_DIR, "all_organism_generation_kg.parquet")
POOL_FILE = os.path.join(PROCESSED_DIR, "generation_instruction_pool.json")

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "all_organism_generation_pairs.jsonl")


def build_entries_for_needed_categories(
    annotation_kg: pd.DataFrame, needed_categories: set[tuple[str, str]]
) -> dict[tuple[str, str], set[str]]:
    """Like build_task_a_generation.build_entries_by_category, but only
    keeps entry-sets for the categories actually present in needed_categories
    instead of every distinct (relation, value) pair in the whole KG --
    restricting to the relevant relations up front bounds peak memory."""
    relations_needed = {relation for relation, _ in needed_categories}
    sub = annotation_kg[annotation_kg["relation"].isin(relations_needed)]
    result: dict[tuple[str, str], set[str]] = {}
    for (relation, value), group in sub.groupby(["relation", "value"]):
        key = (relation, value)
        if key in needed_categories:
            result[key] = set(group["entry"])
    return result


def main() -> None:
    annotation_kg = pd.read_parquet(ALL_ORGANISM_KG_FILE)
    generation_kg = pd.read_parquet(GENERATION_KG_FILE)
    split_df = pd.read_csv(SPLIT_FILE)
    entry_to_split = dict(zip(split_df["entry"], split_df["split"]))
    split_entries = set(entry_to_split.keys())

    with open(FASTA_FILE) as f:
        seq_map = extract_sequences(f, split_entries)
    print(f"sequences for {len(seq_map)}/{len(split_entries)} split entries")

    with open(POOL_FILE) as f:
        pools = json.load(f)

    valid_categories = generation_kg.groupby(["target_relation", "target_value"])
    print(f"categories: {valid_categories.ngroups}")
    needed_categories = set(valid_categories.groups.keys())
    entries_by_category = build_entries_for_needed_categories(annotation_kg, needed_categories)

    counts: dict[tuple[str, str], int] = {}
    total = 0
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as out_f:
        for (target_relation, target_value), context_rows in valid_categories:
            category_entries = entries_by_category.get((target_relation, target_value), set())
            for entry in category_entries:
                seq = seq_map.get(entry)
                split_name = entry_to_split.get(entry)
                if seq is None or split_name is None:
                    continue
                rng = rng_for("all_organism_generation", entry, target_relation, target_value)
                example = build_task_a_generation_example(
                    entry, seq, target_relation, target_value, context_rows, pools[target_relation], rng
                )
                example["split"] = split_name
                out_f.write(json.dumps(example) + "\n")
                total += 1
                counts[(split_name, target_relation)] = counts.get((split_name, target_relation), 0) + 1

    print(f"total examples: {total}")
    for (split_name, target_relation), n in sorted(counts.items()):
        print(f"  {split_name:<10} {target_relation:<15} {n}")
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
