"""
Structural plausibility check for generated sequences (see
scripts/kothar/generate_sequences.py): fold each one with the public ESM
Atlas ESMFold API and report per-sequence mean pLDDT.

The API takes a raw sequence and returns a single-model PDB with per-residue
pLDDT stored in the B-factor column (see
https://esmatlas.com/about#api -- one sequence per request, no auth, capped
at 400 residues).

Usage:
    python3 scripts/eval/pretraining/plddt_esmfold.py --fasta data/generations/checkpoint-2500_gen30_prefixM.fasta
    python3 scripts/eval/pretraining/plddt_esmfold.py --fasta data/generations/checkpoint-2500_gen30_prefixM.fasta \\
        --output data/eval/pretraining/checkpoint-2500_gen30_prefixM_plddt.csv \\
        --save-pdb-dir data/eval/pretraining/pdb/checkpoint-2500_gen30_prefixM
"""

import argparse
import io
import os
import time

import pandas as pd
import requests
from Bio.PDB import PDBParser

ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
MAX_LENGTH = 400

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "..", "..", "..", "data", "eval", "pretraining")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="FASTA file of sequences to fold")
    parser.add_argument("--output", default=None, help="output CSV path (default: data/eval/pretraining/<fasta_stem>_plddt.csv)")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="ESMFold API per-sequence residue cap")
    parser.add_argument("--sleep", type=float, default=1.0, help="delay between API calls, seconds")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout, seconds")
    parser.add_argument("--save-pdb-dir", default=None, help="if set, save each folded structure's PDB here")
    return parser.parse_args()


def read_fasta(path: str) -> list[tuple[str, str]]:
    records = []
    header, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
        if header is not None:
            records.append((header, "".join(chunks)))
    return records


def fold_sequence(sequence: str, timeout: float) -> str:
    response = requests.post(ESMFOLD_API_URL, data=sequence, timeout=timeout)
    response.raise_for_status()
    return response.text


def mean_plddt(pdb_text: str) -> float:
    """ESMFold stores per-residue confidence in the B-factor field on a 0-1
    scale (verified against the raw API response -- not the 0-100 scale
    pLDDT is usually reported on); every atom in a residue carries the same
    value, so averaging over C-alpha atoms gives the per-residue mean, which
    we then rescale to the conventional 0-100 pLDDT range."""
    structure = PDBParser(QUIET=True).get_structure("prediction", io.StringIO(pdb_text))
    bfactors = [residue["CA"].get_bfactor() for residue in structure.get_residues() if "CA" in residue]
    return 100 * sum(bfactors) / len(bfactors)


def main() -> None:
    args = parse_args()
    if args.output:
        output = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.fasta))[0]
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output = os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_plddt.csv")
    if args.save_pdb_dir:
        os.makedirs(args.save_pdb_dir, exist_ok=True)

    records = read_fasta(args.fasta)
    print(f"loaded {len(records)} sequences from {args.fasta}")

    rows: list[dict[str, object]] = []
    for i, (header, sequence) in enumerate(records):
        if len(sequence) > args.max_length:
            print(f"[{i}] {header}: skipped, length {len(sequence)} > --max-length {args.max_length}")
            rows.append({"id": header, "length": len(sequence), "mean_plddt": None, "status": "skipped_too_long"})
            continue

        try:
            pdb_text = fold_sequence(sequence, args.timeout)
            plddt = mean_plddt(pdb_text)
            rows.append({"id": header, "length": len(sequence), "mean_plddt": plddt, "status": "ok"})
            print(f"[{i}] {header}: length={len(sequence)} mean_plddt={plddt:.2f}")
            if args.save_pdb_dir:
                with open(os.path.join(args.save_pdb_dir, f"{header}.pdb"), "w") as f:
                    f.write(pdb_text)
        except (requests.RequestException, KeyError, ZeroDivisionError) as e:
            print(f"[{i}] {header}: failed ({e})")
            rows.append({"id": header, "length": len(sequence), "mean_plddt": None, "status": f"failed: {e}"})

        if i < len(records) - 1:
            time.sleep(args.sleep)

    results = pd.DataFrame(rows)
    results.to_csv(output, index=False)
    print(f"\nwrote {len(results)} rows to {output}")

    ok = results.loc[results["status"] == "ok", "mean_plddt"]
    if len(ok):
        print(f"mean_plddt over {len(ok)} folded sequences: mean={ok.mean():.2f} median={ok.median():.2f} min={ok.min():.2f} max={ok.max():.2f}")
    n_other = len(results) - len(ok)
    if n_other:
        print(f"{n_other} sequence(s) skipped or failed")


if __name__ == "__main__":
    main()
