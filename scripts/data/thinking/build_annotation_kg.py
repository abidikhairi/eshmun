"""
Build the annotation knowledge graph (task family A): per-entry KG triples
for the ~20,431 human SwissProt entries, extending human_protein_kg.parquet
(copied from the external pilot's KG build, data/thinking/raw/) with:

  - has_motif: named motifs (UniProt ft_motif /note text), REPLACING the raw
    amino-acid subsequence the pilot used as its value.
  - catalyzes: EC numbers, a new relation (data/thinking/raw/human_motif_catalytic.tsv,
    fetched via UniProt REST for this same human entry set).
  - has_length: bucketed range (e.g. "[300-400]"), REPLACING the raw residue
    count as the value -- see length_bucket.py.
  - interacts_with: PPI pairs (data/thinking/raw/ppi_interactions.tsv,
    fetched via UniProt REST for all reviewed/SwissProt entries), filtered
    down to human x human pairs only, since this KG is human-only.

All other relations (member_of, has_function, involved_in, located_in,
has_domain, has_region) are carried over from human_protein_kg.parquet
unchanged. Whether a relation is a query *target* vs. reasoning *context* is
not decided here -- that's a dataset-pair-building concern, not a KG-shape
concern, so all relations are stored uniformly, including for entries with
no eligible target relation at all (their domain/region/motif/etc. facts are
still valid context for entries that do have one).

Output: data/thinking/processed/annotation_kg.parquet, same 4-column shape
as the source (entry, relation, value, source).

Usage:
    python3 scripts/data/thinking/build_annotation_kg.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from length_bucket import length_bucket
from ppi import build_ppi_pairs
from uniprot_fields import parse_ec_numbers, parse_motif_names

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

KG_FILE = os.path.join(RAW_DIR, "human_protein_kg.parquet")
MOTIF_CATALYTIC_FILE = os.path.join(RAW_DIR, "human_motif_catalytic.tsv")
PPI_FILE = os.path.join(RAW_DIR, "ppi_interactions.tsv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "annotation_kg.parquet")

TRIPLE_COLUMNS = ["entry", "relation", "value", "source"]


def build_named_motif_triples(motif_df: pd.DataFrame) -> pd.DataFrame:
    """motif_df: columns Entry, Motif (raw ft_motif field) -> long-form
    has_motif triples, one row per parsed /note name (an entry with multiple
    motifs yields multiple rows)."""
    rows = []
    for entry, raw in zip(motif_df["Entry"], motif_df["Motif"]):
        for name in parse_motif_names(raw):
            rows.append(
                {"entry": entry, "relation": "has_motif", "value": name, "source": "UniProt ft_motif /note"}
            )
    return pd.DataFrame(rows, columns=TRIPLE_COLUMNS)


def build_catalyzes_triples(ec_df: pd.DataFrame) -> pd.DataFrame:
    """ec_df: columns Entry, EC number -> long-form catalyzes triples."""
    rows = []
    for entry, raw in zip(ec_df["Entry"], ec_df["EC number"]):
        for ec in parse_ec_numbers(raw):
            rows.append({"entry": entry, "relation": "catalyzes", "value": ec, "source": "UniProt EC"})
    return pd.DataFrame(rows, columns=TRIPLE_COLUMNS)


def build_bucketed_length_triples(length_df: pd.DataFrame) -> pd.DataFrame:
    """length_df: columns entry, value (raw residue count, as already present
    in human_protein_kg's has_length rows) -> bucketed has_length triples."""
    rows = []
    for entry, raw_length in zip(length_df["entry"], length_df["value"]):
        rows.append(
            {
                "entry": entry,
                "relation": "has_length",
                "value": length_bucket(int(raw_length)),
                "source": "computed (bucketed, bin_width=100)",
            }
        )
    return pd.DataFrame(rows, columns=TRIPLE_COLUMNS)


def build_human_ppi_triples(ppi_rows: list[tuple[str, str]], human_entries: set[str]) -> pd.DataFrame:
    """ppi_rows: list of (entry, raw_interacts_with_field) from the full
    all-organism PPI download -> interacts_with triples restricted to pairs
    where BOTH the entry and its partner are in the human entry set (this KG
    is human-only; cross-species interactions, e.g. viral-host, are dropped
    here, not because they're invalid data but because they'd introduce
    non-human entries into a KG whose scope is deliberately human-only)."""
    pairs = build_ppi_pairs(ppi_rows)
    rows = [
        {"entry": e, "relation": "interacts_with", "value": p, "source": "UniProt cc_interaction"}
        for e, p in pairs
        if e in human_entries and p in human_entries
    ]
    return pd.DataFrame(rows, columns=TRIPLE_COLUMNS)


def main() -> None:
    kg = pd.read_parquet(KG_FILE)
    human_entries = set(kg["entry"].unique())
    print(f"human entries in base KG: {len(human_entries)}")

    raw_length = kg[kg["relation"] == "has_length"][["entry", "value"]]
    kept = kg[~kg["relation"].isin(["has_motif", "has_length"])]
    print(f"carried over unchanged: {len(kept)} triples across relations {sorted(kept['relation'].unique())}")

    motif_df = pd.read_csv(MOTIF_CATALYTIC_FILE, sep="\t", dtype=str, keep_default_na=False)
    motif_triples = build_named_motif_triples(motif_df[["Entry", "Motif"]])
    catalyzes_triples = build_catalyzes_triples(motif_df[["Entry", "EC number"]])
    print(f"named motif triples: {len(motif_triples)} (entries: {motif_triples['entry'].nunique()})")
    print(f"catalyzes triples: {len(catalyzes_triples)} (entries: {catalyzes_triples['entry'].nunique()})")

    length_triples = build_bucketed_length_triples(raw_length)
    print(f"bucketed length triples: {len(length_triples)}")

    ppi_df = pd.read_csv(PPI_FILE, sep="\t", dtype=str, keep_default_na=False)
    ppi_rows = list(zip(ppi_df["Entry"], ppi_df["Interacts with"]))
    ppi_triples = build_human_ppi_triples(ppi_rows, human_entries)
    print(f"human x human PPI triples: {len(ppi_triples)} (entries: {ppi_triples['entry'].nunique()})")

    annotation_kg = pd.concat(
        [kept, motif_triples, catalyzes_triples, length_triples, ppi_triples],
        ignore_index=True,
    ).drop_duplicates()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    annotation_kg.to_parquet(OUTPUT_FILE, index=False)

    print(f"\ntotal triples: {len(annotation_kg)}")
    print(f"entries covered: {annotation_kg['entry'].nunique()}")
    print("relation counts:")
    print(annotation_kg["relation"].value_counts())
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
