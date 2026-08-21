"""
Build the generation knowledge graph (task family A, generation direction):
category-level aggregate statistics, keyed by (target_relation, target_value)
rather than by entry.

Why this exists (see conversation record / the "reasoning-trace attributes
that themselves need to be predicted" discussion): the generation direction
only gives the model ONE stated property as input (e.g. member_of =
"Cytochrome P450 family"). A reasoning trace that then cites *other*
instance-specific facts about a real training protein -- "(protein,
has_function, kinase activity)", "(protein, located_in, nucleus)" -- as if
already known asserts things the model has no way to know from the single
given input, and inverts the real causal order (sequence determines
function/localization, not the reverse). The fix: generation-direction
reasoning cites POPULATION-level statistics about the target category
("members of this family are typically membrane-associated, ~300-400
residues") instead of one instance's specific triples. This module computes
those statistics from the annotation knowledge graph.

For each (target_relation, target_value) with at least MIN_GROUP_SIZE member
entries, computes the top-K most common values of every other aggregable
relation among that group. Categories below MIN_GROUP_SIZE are dropped
entirely -- a "typical" claim about a 2-member category is exactly the
single-instance-overfitting problem this is meant to avoid.

Aggregation is computed from TRAIN-SPLIT ENTRIES ONLY (annotation_split.csv),
not the full KG. Using all splits would let a held-out test/validation
entry's own facts leak into "typical category" stats that later inform
train-split reasoning traces -- a test-set leak, even though no individual
test fact is directly copied. The resulting category stats are still applied
when building generation examples for every split (an entry's own real
sequence is still a valid generation *target* regardless of split -- only
the reasoning content must come from train-only statistics). Same fix as
build_scop_dataset.py's generation KG for task family B.

Output: data/thinking/processed/generation_kg.parquet, columns:
    target_relation, target_value, context_relation, context_value,
    count, group_size

Usage:
    python3 scripts/data/thinking/build_generation_kg.py
"""

import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ANNOTATION_KG_FILE = os.path.join(PROCESSED_DIR, "annotation_kg.parquet")
SPLIT_FILE = os.path.join(PROCESSED_DIR, "annotation_split.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "generation_kg.parquet")

# The relations task family A supports as generation-direction conditioning
# targets (locked scope: family, function, process, localization, catalytic
# activity). interacts_with is excluded on both sides of the aggregation --
# a PPI partner is a specific entity, not a generalizable category trait, so
# "typical interactor" isn't a meaningful population-level statement here.
TARGET_RELATIONS = ["member_of", "has_function", "involved_in", "located_in", "catalyzes"]

# Relations eligible to appear as aggregated context for a category (a
# subset of these, minus whichever is the current target).
AGGREGABLE_RELATIONS = [
    "member_of", "has_function", "involved_in", "located_in", "catalyzes",
    "has_domain", "has_region", "has_motif", "has_length",
]

TOP_K = 3
MIN_GROUP_SIZE = 5


def aggregate_context(
    group_triples: pd.DataFrame,
    target_relation: str,
    top_k: int = TOP_K,
    aggregable_relations: list[str] = AGGREGABLE_RELATIONS,
) -> list[dict]:
    """group_triples: all KG triples belonging to the entries in one category
    group (already filtered to that group). Returns the top_k most common
    values per aggregable relation (excluding target_relation itself),
    skipping any relation with no data in this group."""
    group_size = group_triples["entry"].nunique()
    results = []
    for relation in aggregable_relations:
        if relation == target_relation:
            continue
        sub = group_triples[group_triples["relation"] == relation]
        if sub.empty:
            continue
        for value, count in sub["value"].value_counts().head(top_k).items():
            results.append(
                {"context_relation": relation, "context_value": value, "count": int(count), "group_size": group_size}
            )
    return results


def build_generation_kg(
    annotation_kg: pd.DataFrame,
    target_relations: list[str] = TARGET_RELATIONS,
    aggregable_relations: list[str] = AGGREGABLE_RELATIONS,
    top_k: int = TOP_K,
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    # Pre-group once by entry (restricted to relations that can ever matter
    # here) instead of filtering the full KG with .isin() inside the
    # per-category loop -- at all-organism scale (millions of rows, tens of
    # thousands of categories) the naive O(categories * KG size) version
    # never finished in reasonable time. This makes total work O(KG size)
    # once, plus O(category size) per category, not O(KG size) per category.
    relevant_relations = set(target_relations) | set(aggregable_relations)
    relevant_kg = annotation_kg[annotation_kg["relation"].isin(relevant_relations)]
    triples_by_entry = {entry: group for entry, group in relevant_kg.groupby("entry")}

    rows = []
    dropped_categories = 0
    for target_relation in target_relations:
        target_df = annotation_kg[annotation_kg["relation"] == target_relation]
        for target_value, group in target_df.groupby("value"):
            entries_in_group = set(group["entry"])
            if len(entries_in_group) < min_group_size:
                dropped_categories += 1
                continue
            pieces = [triples_by_entry[e] for e in entries_in_group if e in triples_by_entry]
            group_triples = pd.concat(pieces, ignore_index=True) if pieces else annotation_kg.iloc[0:0]
            for agg in aggregate_context(group_triples, target_relation, top_k=top_k, aggregable_relations=aggregable_relations):
                rows.append({"target_relation": target_relation, "target_value": target_value, **agg})

    print(f"categories dropped (group_size < {min_group_size}): {dropped_categories}")
    columns = ["target_relation", "target_value", "context_relation", "context_value", "count", "group_size"]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    annotation_kg = pd.read_parquet(ANNOTATION_KG_FILE)

    split_df = pd.read_csv(SPLIT_FILE)
    train_entries = set(split_df[split_df["split"] == "train"]["entry"])
    train_kg = annotation_kg[annotation_kg["entry"].isin(train_entries)]
    print(f"train entries: {len(train_entries)} (of {annotation_kg['entry'].nunique()} total in the KG)")

    generation_kg = build_generation_kg(train_kg)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    generation_kg.to_parquet(OUTPUT_FILE, index=False)

    print(f"\ntotal (category, context) rows: {len(generation_kg)}")
    print(f"unique categories: {generation_kg.groupby(['target_relation', 'target_value']).ngroups}")
    print("categories per target_relation:")
    print(generation_kg.groupby("target_relation")["target_value"].nunique())
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
