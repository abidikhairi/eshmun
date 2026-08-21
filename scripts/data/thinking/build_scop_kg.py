"""
Build the SCOP knowledge graph (task family B): scop_fold / scop_superfamily
/ scop_family triples for reviewed UniProt entries, all organisms (unlike
task family A, which is human-only).

Pipeline: SCOPe classification (dir.cla, one row per structural domain) ->
SIFTS PDB-chain-to-UniProt mapping -> filtered to reviewed/SwissProt
accessions only (SIFTS covers both SwissProt and TrEMBL). Human-readable
names for each fold/superfamily/family come from SCOPe's dir.des file.

Coverage is necessarily much smaller than task family A's ~20k human
entries -- SCOPe only classifies proteins with a solved PDB structure, a
structurally-biased minority of SwissProt. Accepted scope, per the design
discussion (see conversation record / ROADMAP).

Output: data/thinking/processed/scop_kg.parquet, same 4-column shape as
annotation_kg.parquet (entry, relation, value, source).

Usage:
    python3 scripts/data/thinking/build_scop_kg.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scop import build_scop_triples, build_sifts_map, parse_cla_line, parse_des_lines, resolve_domain_to_entries

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

CLA_FILE = os.path.join(RAW_DIR, "scope_cla.txt")
DES_FILE = os.path.join(RAW_DIR, "scope_des.txt")
SIFTS_FILE = os.path.join(RAW_DIR, "pdb_chain_uniprot.csv")
REVIEWED_FILE = os.path.join(RAW_DIR, "reviewed_accessions.tsv")

OUTPUT_FILE = os.path.join(PROCESSED_DIR, "scop_kg.parquet")


def main() -> None:
    with open(CLA_FILE) as f:
        domains = [d for d in (parse_cla_line(line) for line in f) if d is not None]
    print(f"parsed {len(domains)} SCOPe domains")

    with open(DES_FILE) as f:
        descriptions = parse_des_lines(f)
    print(f"parsed {len(descriptions)} fold/superfamily/family descriptions")

    sifts_df = pd.read_csv(SIFTS_FILE, skiprows=1, dtype=str)
    sifts_rows = list(zip(sifts_df["PDB"], sifts_df["CHAIN"], sifts_df["SP_PRIMARY"]))
    sifts_map = build_sifts_map(sifts_rows)
    print(f"SIFTS: {len(sifts_rows)} rows -> {len(sifts_map)} (pdbid, chain) keys")

    reviewed = set(pd.read_csv(REVIEWED_FILE, sep="\t", dtype=str)["Entry"])
    print(f"reviewed accessions: {len(reviewed)}")

    all_triples: list[tuple[str, str, str]] = []
    domains_with_hits = 0
    for domain in domains:
        entries = resolve_domain_to_entries(domain, sifts_map, reviewed)
        if entries:
            domains_with_hits += 1
            all_triples.extend(build_scop_triples(domain, entries, descriptions))

    print(f"domains resolving to >=1 reviewed entry: {domains_with_hits}/{len(domains)}")

    scop_kg = pd.DataFrame(all_triples, columns=["entry", "relation", "value"])
    scop_kg["source"] = "SCOPe 2.08-stable + SIFTS"
    scop_kg = scop_kg.drop_duplicates()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    scop_kg.to_parquet(OUTPUT_FILE, index=False)

    print(f"\ntotal triples: {len(scop_kg)}")
    print(f"entries covered: {scop_kg['entry'].nunique()}")
    print("relation counts:")
    print(scop_kg["relation"].value_counts())
    print(f"\nsaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
