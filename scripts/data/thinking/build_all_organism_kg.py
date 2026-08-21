"""
Build the all-organism annotation knowledge graph: same 9-relation schema as
the human-only annotation_kg.parquet (member_of, has_function, involved_in,
located_in, has_domain, has_region, has_motif, catalyzes, has_length), but
for all ~575,503 reviewed SwissProt entries, not just human.

Unlike the human KG (built from several pre-extracted, human-only tables
inherited from the pilot work), this is built from a single bulk UniProt
REST download (data/thinking/raw/all_swissprot_fields.tsv) covering every
reviewed entry. GO terms (function/process/location) come with a trailing
" [GO:0006953]" ID suffix that parse_go_terms strips -- the human KG got
term names from a separately-parsed GO ontology file, but the meaning is
identical (just a different source pipeline for the same information). No
interacts_with relation here -- PPI wasn't requested at all-organism scope;
task family A's own human x human restriction stays as-is.

Output: data/thinking/processed/all_organism_kg.parquet, same 4-column
shape as annotation_kg.parquet (entry, relation, value, source).

Usage:
    python3 scripts/data/thinking/build_all_organism_kg.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from length_bucket import length_bucket
from uniprot_fields import parse_ec_numbers, parse_go_terms, parse_note_field

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

FIELDS_FILE = os.path.join(RAW_DIR, "all_swissprot_fields.tsv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "all_organism_kg.parquet")

TRIPLE_COLUMNS = ["entry", "relation", "value", "source"]

# (source TSV column, relation name, parser, source label)
FIELD_SPECS = [
    ("Gene Ontology (molecular function)", "has_function", parse_go_terms, "UniProt GO (molecular_function)"),
    ("Gene Ontology (biological process)", "involved_in", parse_go_terms, "UniProt GO (biological_process)"),
    ("Gene Ontology (cellular component)", "located_in", parse_go_terms, "UniProt GO (cellular_component)"),
    ("Domain [FT]", "has_domain", parse_note_field, "UniProt ft_domain /note"),
    ("Region", "has_region", parse_note_field, "UniProt ft_region /note"),
    ("Motif", "has_motif", parse_note_field, "UniProt ft_motif /note"),
    ("EC number", "catalyzes", parse_ec_numbers, "UniProt EC"),
]


def build_family_triples(df: pd.DataFrame) -> pd.DataFrame:
    """Protein families is already a single clean string per entry (unlike
    the semicolon-list fields) -- no parsing needed beyond an emptiness check."""
    sub = df[df["Protein families"].str.strip() != ""]
    return pd.DataFrame(
        {
            "entry": sub["Entry"],
            "relation": "member_of",
            "value": sub["Protein families"],
            "source": "UniProt Protein families",
        },
        columns=TRIPLE_COLUMNS,
    )


def build_field_triples(df: pd.DataFrame, column: str, relation: str, parser, source: str) -> pd.DataFrame:
    rows = []
    for entry, raw in zip(df["Entry"], df[column]):
        for value in parser(raw):
            rows.append({"entry": entry, "relation": relation, "value": value, "source": source})
    return pd.DataFrame(rows, columns=TRIPLE_COLUMNS)


def build_length_triples(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entry, raw_length in zip(df["Entry"], df["Length"]):
        rows.append(
            {
                "entry": entry, "relation": "has_length",
                "value": length_bucket(int(raw_length)),
                "source": "computed (bucketed, bin_width=100)",
            }
        )
    return pd.DataFrame(rows, columns=TRIPLE_COLUMNS)


def main() -> None:
    df = pd.read_csv(FIELDS_FILE, sep="\t", dtype=str, keep_default_na=False)
    print(f"loaded {len(df)} entries")

    parts = [build_family_triples(df), build_length_triples(df)]
    for column, relation, parser, source in FIELD_SPECS:
        triples = build_field_triples(df, column, relation, parser, source)
        print(f"{relation}: {len(triples)} triples ({triples['entry'].nunique()} entries)")
        parts.append(triples)

    kg = pd.concat(parts, ignore_index=True).drop_duplicates()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    kg.to_parquet(OUTPUT_FILE, index=False)

    print(f"\ntotal triples: {len(kg)}")
    print(f"entries covered: {kg['entry'].nunique()}")
    print("relation counts:")
    print(kg["relation"].value_counts())
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
