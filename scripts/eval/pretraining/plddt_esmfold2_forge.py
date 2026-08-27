"""
Structural plausibility check for generated sequences (see
scripts/kothar/generate_sequences.py): fold each one with EvolutionaryScale's
ESMFold2 via the Biohub Forge API and report per-sequence mean pLDDT.

Needs BIOHUB_TOKEN in the environment (source ~/.zshrc, which exports it,
before running this script).

Usage:
    python3 scripts/eval/pretraining/plddt_esmfold2_forge.py --fasta data/generations/checkpoint-2500_gen30_prefixM.fasta
    python3 scripts/eval/pretraining/plddt_esmfold2_forge.py --fasta data/generations/checkpoint-2500_gen30_prefixM.fasta \\
        --output data/eval/pretraining/checkpoint-2500_gen30_prefixM_plddt_esmfold2.csv \\
        --save-cif-dir data/eval/pretraining/cif/checkpoint-2500_gen30_prefixM
"""

import argparse
import os
import time

import pandas as pd
from esm.sdk.forge import ESMProteinError, FoldingConfig, ProteinInput, SequenceStructureForgeInferenceClient, StructurePredictionInput

FORGE_URL = "https://biohub.ai"
MODEL = "esmfold2-fast-2026-05"

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "..", "..", "..", "data", "eval", "pretraining")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="FASTA file of sequences to fold")
    parser.add_argument("--output", default=None, help="output CSV path (default: data/eval/pretraining/<fasta_stem>_plddt_esmfold2.csv)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--num-loops", type=int, default=3, help="trunk refinement loops; higher = slower, more accurate")
    parser.add_argument("--num-sampling-steps", type=int, default=10, help="diffusion ODE solver steps; higher = slower, more accurate")
    parser.add_argument("--sleep", type=float, default=0.5, help="delay between API calls, seconds")
    parser.add_argument("--save-cif-dir", default=None, help="if set, save each folded structure's mmCIF here")
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


def main() -> None:
    args = parse_args()
    if "BIOHUB_TOKEN" not in os.environ:
        raise SystemExit("BIOHUB_TOKEN not set -- source ~/.zshrc first")

    if args.output:
        output = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.fasta))[0]
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output = os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_plddt_esmfold2.csv")
    if args.save_cif_dir:
        os.makedirs(args.save_cif_dir, exist_ok=True)

    client = SequenceStructureForgeInferenceClient(model=args.model, url=FORGE_URL, token=os.environ["BIOHUB_TOKEN"])
    config = FoldingConfig(num_loops=args.num_loops, num_sampling_steps=args.num_sampling_steps, include_pae=False)

    records = read_fasta(args.fasta)
    print(f"loaded {len(records)} sequences from {args.fasta}")

    rows: list[dict[str, object]] = []
    for i, (header, sequence) in enumerate(records):
        inp = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
        result = client.fold_all_atom(inp, config=config)

        # The client's own retry_decorator gives up after a few seconds of
        # backoff, which isn't enough to clear a per-minute rate cap (hit in
        # practice: 20 req/min on esmfold2-fast-2026-05) -- wait it out and
        # retry once more before recording a failure.
        if isinstance(result, ESMProteinError) and "usage cap" in str(result):
            print(f"[{i}] {header}: rate-limited, waiting 65s before one more attempt")
            time.sleep(65)
            result = client.fold_all_atom(inp, config=config)

        if isinstance(result, ESMProteinError):
            print(f"[{i}] {header}: failed ({result})")
            rows.append({"id": header, "length": len(sequence), "mean_plddt": None, "status": f"failed: {result}"})
        else:
            assert not isinstance(result, list), "single-sequence input should not return a list of results"
            plddt = 100 * result.plddt.mean().item()
            rows.append({"id": header, "length": len(sequence), "mean_plddt": plddt, "status": "ok"})
            print(f"[{i}] {header}: length={len(sequence)} mean_plddt={plddt:.2f}")
            if args.save_cif_dir:
                with open(os.path.join(args.save_cif_dir, f"{header}.cif"), "w") as f:
                    f.write(result.complex.to_mmcif())

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
        print(f"{n_other} sequence(s) failed")


if __name__ == "__main__":
    main()
