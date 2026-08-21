"""
Split the SCOP KG's entries into train/validation/test by sequence identity,
reusing split_by_identity.py's core (same 70%/30% MMseqs2 thresholds as task
family A). Sequences come from the full, all-organism SwissProt FASTA
(data/uniprot_sprot.fasta) rather than the human-only sequence table, since
task family B is explicitly all-organisms.

Output: data/thinking/processed/scop_split.csv, columns (entry, split).

Usage:
    python3 scripts/data/thinking/split_scop_by_identity.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fasta import extract_sequences
from split_by_identity import MAX_SEQUENCE_LENGTH, run_identity_split

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SCOP_KG_FILE = os.path.join(PROCESSED_DIR, "scop_kg.parquet")
FASTA_FILE = os.path.join(_HERE, "..", "..", "..", "data", "uniprot_sprot.fasta")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "scop_split.csv")


def main() -> None:
    scop_kg = pd.read_parquet(SCOP_KG_FILE)
    entries = set(scop_kg["entry"].unique())
    print(f"{len(entries)} entries in the SCOP KG")

    with open(FASTA_FILE) as f:
        seq_map = extract_sequences(f, entries)
    print(f"{len(seq_map)}/{len(entries)} entries found in {FASTA_FILE}")

    df = pd.DataFrame({"Entry": list(seq_map.keys()), "Sequence": list(seq_map.values())})
    before_length_filter = len(df)
    df = df[df["Sequence"].str.len() < MAX_SEQUENCE_LENGTH]
    print(f"{len(df)}/{before_length_filter} entries with a sequence < {MAX_SEQUENCE_LENGTH} residues")

    run_identity_split(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
