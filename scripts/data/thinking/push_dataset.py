"""
Package all 8 built task-family JSONL files into a single multi-config
HuggingFace dataset repo, one config per task/direction, each with
train/validation/test splits derived from the existing `split` column
(dropped from the final schema, same as build_ablation_datasets.py's
pattern -- the DatasetDict split keys already encode this).

Uses datasets.load_dataset("json", ...) rather than pandas so the two
multi-GB files (all_organism_annotation/generation) go through Arrow's
memory-mapped backing instead of being materialized as Python lists --
this repo already hit two OOM kills this session (build_all_organism_kg's
category grouping, and generation_kg's per-category filtering) from
exactly that kind of full-materialization mistake.

Usage:
    python3 scripts/data/thinking/push_dataset.py --repo-id khairi/eshmun-thinking
    python3 scripts/data/thinking/push_dataset.py --repo-id khairi/eshmun-thinking --push
"""

import argparse
import os

from datasets import DatasetDict, load_dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking", "processed")

SPLIT_NAMES = ["train", "validation", "test"]

CONFIGS = {
    "task_a_annotation": "task_a_annotation_pairs.jsonl",
    "task_a_generation": "task_a_generation_pairs.jsonl",
    "all_organism_annotation": "all_organism_annotation_pairs.jsonl",
    "all_organism_generation": "all_organism_generation_pairs.jsonl",
    "ppi_annotation": "ppi_annotation_pairs.jsonl",
    "ppi_generation": "ppi_generation_pairs.jsonl",
    "scop_annotation": "scop_annotation_pairs.jsonl",
    "scop_generation": "scop_generation_pairs.jsonl",
}


def build_config_dataset_dict(jsonl_path: str) -> DatasetDict:
    raw = load_dataset("json", data_files=jsonl_path)["train"]
    splits = {}
    for split_name in SPLIT_NAMES:
        subset = raw.filter(lambda row: row["split"] == split_name, desc=f"filter:{split_name}")
        subset = subset.remove_columns("split")
        splits[split_name] = subset
    return DatasetDict(splits)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub. Without this, only builds and prints summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if args.push and not token:
        raise EnvironmentError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN to push to the Hub.")

    print("=== dataset config summary ===")
    for config_name, filename in CONFIGS.items():
        jsonl_path = os.path.join(PROCESSED_DIR, filename)
        dd = build_config_dataset_dict(jsonl_path)
        sizes = {split: len(dd[split]) for split in SPLIT_NAMES}
        print(f"{config_name}: {sizes}  columns={dd['train'].column_names}")

        if args.push:
            print(f"  pushing config '{config_name}' to {args.repo_id}...")
            dd.push_to_hub(args.repo_id, config_name=config_name, token=token, private=args.private)

    if args.push:
        print(f"\ndone: https://huggingface.co/datasets/{args.repo_id}")
    else:
        print("\n(dry run -- pass --push to actually push to the Hub)")


if __name__ == "__main__":
    main()
