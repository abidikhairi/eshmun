"""
Push the Kothar Stage-1 continued-pretraining mix (built by
build_pretrain_mix.py) to the HuggingFace Hub as a single-split dataset.

Usage:
    python3 scripts/kothar/push_pretrain_mix.py                # dry run: build + summarize only
    python3 scripts/kothar/push_pretrain_mix.py --push          # actually push
    python3 scripts/kothar/push_pretrain_mix.py --push --private
"""

import argparse
import os

from datasets import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_MIX_FILE = os.path.join(
    _HERE, "..", "..", "data", "kothar", "pretrain_mix.parquet"
)

REPO_ID = "khairi/kothar-pretrain-mix-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true", help="Push to the Hub. Without this, only builds and summarizes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ds = Dataset.from_parquet(PRETRAIN_MIX_FILE)
    print(ds)
    print("row counts by source:")
    counts = {}
    for row in ds["source"]:
        counts[row] = counts.get(row, 0) + 1
    for source, n in counts.items():
        print(f"  {source}: {n}")

    if not args.push:
        print("\n(dry run -- pass --push to actually push to the Hub)")
        return

    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"\ndone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
