"""
Build PPI annotation-direction (yes/no interaction classification) and
generation-direction (design a partner) examples from the annotation KG.

Annotation direction: both proteins are GIVEN inputs, so reasoning may cite
real KG facts about either one -- no causal-inversion issue (contrast with
task family A's other generation-direction fix). 1 positive : 3 negative
ratio; negatives are sampled by fixing the first protein of a positive pair
and swapping in a random non-partner, restricted to the SAME train/val/test
split as the positive pair (entry-level split integrity: an entry from test
must never appear inside a training example, including as a negative-sample
partner).

Generation direction: only protein_a (X) is given; protein_b (Y, the
interactor to design) is not yet generated, so reasoning cites only X's own
real facts (never Y's) -- the same fix applied to the rest of task family
A's generation direction, here following naturally from which side of the
pair is actually "given". Positive pairs only; no negative sampling (there's
no sensible "design a sequence that does NOT interact with X" framing
analogous to the annotation yes/no case).

Usage:
    python3 scripts/data/thinking/build_ppi_dataset.py
"""

import hashlib
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppi_pairs import build_partners_by_entry, build_positive_pair_set, sample_negatives_for_entry
from reasoning import Triple, build_reasoning_block, encode_sequence, with_subject

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ANNOTATION_KG_FILE = os.path.join(PROCESSED_DIR, "annotation_kg.parquet")
SPLIT_FILE = os.path.join(PROCESSED_DIR, "annotation_split.csv")
SEQ_FILE = os.path.join(RAW_DIR, "homosapiens-sequences.tsv")

ANNOTATION_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "ppi_annotation_pairs.jsonl")
GENERATION_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "ppi_generation_pairs.jsonl")

NEGATIVE_RATIO = 3
MAX_CONTEXT_PER_RELATION = 3
# Reasoning context for PPI never cites other interacts_with facts (would
# invite a "already interacts with lots of things" shortcut instead of
# content-based reasoning) or has_length (residue count isn't informative
# for whether two specific proteins bind each other).
CONTEXT_RELATIONS = ["member_of", "has_function", "involved_in", "located_in", "catalyzes", "has_domain", "has_region", "has_motif"]

ANNOTATION_INSTRUCTION_TEMPLATES = [
    "Do {protein_a} and {protein_b} interact with each other?",
    "Is there a known physical interaction between {protein_a} and {protein_b}?",
    "Determine whether {protein_a} and {protein_b} interact.",
    "Based on their sequences, do {protein_a} and {protein_b} bind each other?",
]

GENERATION_INSTRUCTION_TEMPLATES = [
    "Design a protein sequence that interacts with {protein_a}.",
    "Generate a protein sequence known to bind {protein_a}.",
    "Give me a protein sequence that physically interacts with {protein_a}.",
]


def rng_for(*parts: str) -> random.Random:
    seed = int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def triples_by_relation_for_entry(kg: pd.DataFrame, entry: str) -> dict[str, list[str]]:
    sub = kg[(kg["entry"] == entry) & (kg["relation"].isin(CONTEXT_RELATIONS))]
    result: dict[str, list[str]] = {}
    for relation, group in sub.groupby("relation"):
        result[relation] = group["value"].tolist()
    return result


def select_all_context_triples(
    triples_by_relation: dict[str, list[str]], rng: random.Random, max_per_relation: int = MAX_CONTEXT_PER_RELATION
) -> list[Triple]:
    """Like reasoning.select_context_triples, but with no target relation to
    exclude -- PPI annotation/generation don't have a "target relation"
    among member_of/has_function/etc., interacts_with itself is the target."""
    context: list[Triple] = []
    for relation, values in triples_by_relation.items():
        chosen = values if len(values) <= max_per_relation else rng.sample(values, max_per_relation)
        context.extend(Triple(relation=relation, value=v) for v in chosen)
    return context


def build_split_pairs(
    positive_pairs: set[frozenset],
    entries_in_split: list[str],
    ratio: int,
) -> list[tuple[str, str, bool]]:
    """For each positive pair fully contained in entries_in_split, emit the
    positive pair plus `ratio` negative pairs (same anchor entry, swapped
    partner, restricted to entries_in_split so split integrity holds).

    Negative sampling uses a per-pair deterministic RNG (rng_for) rather
    than a shared external one, so results don't depend on set iteration
    order (Python set ordering isn't guaranteed stable across runs) -- no
    external rng parameter needed."""
    entry_set = set(entries_in_split)
    partners_by_entry = build_partners_by_entry(positive_pairs)

    result: list[tuple[str, str, bool]] = []
    for pair in positive_pairs:
        # sorted(), not tuple(pair): frozenset iteration order depends on
        # Python's per-process hash randomization, so tuple(pair) would make
        # which entry becomes the negative-sampling anchor non-deterministic
        # across runs even with the same rng_for seeding.
        members = sorted(pair) if len(pair) == 2 else sorted(pair) * 2  # self-interaction: (e, e)
        a, b = members
        if a not in entry_set or b not in entry_set:
            continue
        result.append((a, b, True))
        try:
            negatives = sample_negatives_for_entry(
                entry=a, true_partners=partners_by_entry.get(a, set()),
                candidate_pool=entries_in_split, n=ratio, rng=rng_for("ppi_neg", a, b),
            )
        except ValueError:
            continue  # not enough eligible candidates in this split -- skip rather than under-sample
        for neg in negatives:
            result.append((a, neg, False))
    return result


def build_ppi_annotation_example(
    entry_a: str, seq_a: str, triples_a: dict[str, list[str]],
    entry_b: str, seq_b: str, triples_b: dict[str, list[str]],
    is_positive: bool,
    rng: random.Random,
) -> dict:
    context_a = with_subject(select_all_context_triples(triples_a, rng), "protein_a")
    context_b = with_subject(select_all_context_triples(triples_b, rng), "protein_b")
    reasoning = build_reasoning_block(context_a + context_b, rng)

    instruction_template = rng.choice(ANNOTATION_INSTRUCTION_TEMPLATES)
    instruction = instruction_template.format(protein_a=encode_sequence(seq_a), protein_b=encode_sequence(seq_b))
    answer = "Yes, these two proteins interact." if is_positive else "No, these two proteins do not interact."

    return {
        "entry_a": entry_a, "entry_b": entry_b,
        "instruction": instruction, "reasoning": reasoning, "answer": answer,
        "label": "positive" if is_positive else "negative",
    }


def build_ppi_generation_example(
    entry_a: str, seq_a: str, triples_a: dict[str, list[str]],
    entry_b: str, seq_b: str,
    rng: random.Random,
) -> dict:
    """entry_a (X) is given; entry_b (Y) is the interactor to design.
    Reasoning cites only X's real facts -- Y doesn't exist yet at
    "generation time", so nothing about Y is asserted as context."""
    context_a = with_subject(select_all_context_triples(triples_a, rng), "protein_a")
    reasoning = build_reasoning_block(context_a, rng)

    instruction_template = rng.choice(GENERATION_INSTRUCTION_TEMPLATES)
    instruction = instruction_template.format(protein_a=encode_sequence(seq_a))
    answer = encode_sequence(seq_b)

    return {
        "entry_a": entry_a, "entry_b": entry_b,
        "instruction": instruction, "reasoning": reasoning, "answer": answer,
    }


def main() -> None:
    kg = pd.read_parquet(ANNOTATION_KG_FILE)
    split_df = pd.read_csv(SPLIT_FILE)
    seq_df = pd.read_csv(SEQ_FILE, sep="\t", dtype=str)
    seq_map = dict(zip(seq_df["Entry"], seq_df["Sequence"]))

    ppi_rows = list(kg[kg["relation"] == "interacts_with"][["entry", "value"]].itertuples(index=False, name=None))
    positive_pairs = build_positive_pair_set(ppi_rows)
    print(f"positive pairs (undirected, human x human): {len(positive_pairs)}")

    triples_cache: dict[str, dict[str, list[str]]] = {}

    def get_triples(entry: str) -> dict[str, list[str]]:
        if entry not in triples_cache:
            triples_cache[entry] = triples_by_relation_for_entry(kg, entry)
        return triples_cache[entry]

    annotation_examples: list[dict] = []
    generation_examples: list[dict] = []

    for split_name in ["train", "validation", "test"]:
        entries_in_split = split_df[split_df["split"] == split_name]["entry"].tolist()
        pairs = build_split_pairs(positive_pairs, entries_in_split, ratio=NEGATIVE_RATIO)
        n_positive = sum(1 for p in pairs if p[2])
        print(f"[{split_name}] annotation pairs: {len(pairs)} ({n_positive} positive, {len(pairs) - n_positive} negative)")

        for a, b, is_positive in pairs:
            rng = rng_for("ppi_annotation", split_name, a, b, str(is_positive))
            example = build_ppi_annotation_example(
                a, seq_map[a], get_triples(a), b, seq_map[b], get_triples(b), is_positive, rng
            )
            example["split"] = split_name
            annotation_examples.append(example)

        # Generation direction: positive pairs only, both directions (each
        # side of a real interacting pair is a valid "given protein").
        positive_in_split = [(a, b) for a, b, is_pos in pairs if is_pos]
        for a, b in positive_in_split:
            for x, y in [(a, b), (b, a)]:
                rng = rng_for("ppi_generation", split_name, x, y)
                example = build_ppi_generation_example(x, seq_map[x], get_triples(x), y, seq_map[y], rng)
                example["split"] = split_name
                generation_examples.append(example)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pd.DataFrame(annotation_examples).to_json(ANNOTATION_OUTPUT_FILE, orient="records", lines=True)
    pd.DataFrame(generation_examples).to_json(GENERATION_OUTPUT_FILE, orient="records", lines=True)

    print(f"\ntotal annotation examples: {len(annotation_examples)}")
    print(f"total generation examples: {len(generation_examples)}")
    print(f"saved to {ANNOTATION_OUTPUT_FILE}")
    print(f"saved to {GENERATION_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
