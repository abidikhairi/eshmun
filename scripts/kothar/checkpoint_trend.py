"""
Walk every checkpoint-N directory under a Kothar pretraining output dir,
evaluate each on the held-out validation set (eval_checkpoint.py's logic),
and report loss/perplexity per source over training steps -- the trend to
watch for a plateau before moving to instruction tuning (Stage 2).

Works against either a local pull of checkpoints or, run directly on the
remote training machine, the live output-dir -- no need to copy every
checkpoint over the network just to evaluate it. Only needs the small
per-checkpoint files (config.json, generation_config.json,
model.safetensors), not optimizer.pt/rng_state.pth/etc.

Loads and evaluates in float32 (float16 has caused issues for this project
before). Frees the model between checkpoints to keep peak GPU memory to one
checkpoint at a time.

Usage:
    python3 scripts/kothar/checkpoint_trend.py --checkpoints-dir checkpoints/kothar-pretrain-409m
    python3 scripts/kothar/checkpoint_trend.py --checkpoints-dir checkpoints/kothar-pretrain-409m --output data/kothar/checkpoint_trend.csv
"""

import argparse
import glob
import os
import re

import pandas as pd
import torch
from transformers import AutoTokenizer

from eshmun.models.eshmun import EshmunForCausalLM

STUDENT_ID = "khairi/Kothar-student-seed-409M"

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(_HERE, "..", "..", "data", "kothar", "valid_holdout.parquet")
DEFAULT_OUTPUT = os.path.join(_HERE, "..", "..", "data", "kothar", "checkpoint_trend.csv")

CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True, help="Trainer output_dir containing checkpoint-N subdirs")
    parser.add_argument("--tokenizer", default=STUDENT_ID, help="tokenizer source; checkpoints don't carry their own")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV path for the per-checkpoint, per-source results")
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def find_checkpoints(checkpoints_dir: str) -> list[tuple[int, str]]:
    found = []
    for path in glob.glob(os.path.join(checkpoints_dir, "checkpoint-*")):
        match = CHECKPOINT_RE.search(path)
        if match and os.path.isfile(os.path.join(path, "model.safetensors")):
            found.append((int(match.group(1)), path))
    return sorted(found)


@torch.no_grad()
def sequence_loss(model, tokenizer, text: str, max_length: int, device) -> float | None:
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return None
    return model(input_ids=input_ids, labels=input_ids).loss.item()


def evaluate_checkpoint(checkpoint_path: str, tokenizer, df: pd.DataFrame, max_length: int, device) -> list[dict]:
    model = EshmunForCausalLM.from_pretrained(checkpoint_path, dtype=torch.float32).to(device)
    model.eval()

    rows = []
    for source in sorted(df["source"].unique()):
        for text in df.loc[df["source"] == source, "content"]:
            loss = sequence_loss(model, tokenizer, text, max_length, device)
            if loss is not None:
                rows.append({"source": source, "loss": loss})

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoints = find_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint-N dirs with model.safetensors under {args.checkpoints_dir}")
    print(f"found {len(checkpoints)} checkpoints: {[step for step, _ in checkpoints]}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    df = pd.read_parquet(args.dataset)

    all_rows = []
    for step, path in checkpoints:
        print(f"\nevaluating step {step} ({path})")
        rows = evaluate_checkpoint(path, tokenizer, df, args.max_length, device)
        for row in rows:
            row["step"] = step
        all_rows.extend(rows)

        step_df = pd.DataFrame(rows)
        for source, group in step_df.groupby("source"):
            mean_loss = group["loss"].mean()
            mean_ppl = torch.exp(torch.tensor(mean_loss)).item()
            print(f"  {source:<10s} n={len(group):>4d}  loss={mean_loss:>8.4f}  ppl={mean_ppl:>12.4f}")
        overall_loss = step_df["loss"].mean()
        print(f"  {'overall':<10s} n={len(step_df):>4d}  loss={overall_loss:>8.4f}  ppl={torch.exp(torch.tensor(overall_loss)).item():>12.4f}")

    results = pd.DataFrame(all_rows)
    summary = results.groupby(["step", "source"])["loss"].mean().reset_index()
    summary["ppl"] = summary["loss"].apply(lambda x: torch.exp(torch.tensor(x)).item())

    overall = results.groupby("step")["loss"].mean().reset_index()
    overall["source"] = "overall"
    overall["ppl"] = overall["loss"].apply(lambda x: torch.exp(torch.tensor(x)).item())

    trend = pd.concat([summary, overall], ignore_index=True).sort_values(["source", "step"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trend.to_csv(args.output, index=False)
    print(f"\nwrote {args.output}")

    print("\n=== trend (pivoted, ppl) ===")
    print(trend.pivot(index="step", columns="source", values="ppl").to_string())


if __name__ == "__main__":
    main()
