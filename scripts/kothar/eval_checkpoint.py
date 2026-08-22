"""
Evaluate a Kothar student checkpoint on a held-out parquet dataset
(entry, content, source columns -- e.g. build_valid_holdout.py's output):
per-example loss/perplexity, aggregated by source and overall.

Checkpoints saved mid-training (Trainer's periodic checkpoint-N dirs) don't
include tokenizer files, so the tokenizer is loaded separately from the Hub
student id by default.

Loads and evaluates in float32 (float16 has caused issues for this project
before).

Usage:
    python3 scripts/kothar/eval_checkpoint.py --checkpoint /tmp/kothar-checkpoint-500
    python3 scripts/kothar/eval_checkpoint.py --checkpoint /tmp/kothar-checkpoint-500 --dataset data/kothar/valid_holdout.parquet
"""

import argparse
import os

import pandas as pd
import torch

from eshmun.models.eshmun import EshmunForCausalLM

STUDENT_ID = "khairi/Kothar-student-seed-409M"

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(_HERE, "..", "..", "data", "kothar", "valid_holdout.parquet")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint-N directory (model weights + config, no tokenizer)")
    parser.add_argument("--tokenizer", default=STUDENT_ID, help="tokenizer source; checkpoints don't carry their own")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


@torch.no_grad()
def sequence_loss(model, tokenizer, text: str, max_length: int, device) -> float | None:
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return None
    return model(input_ids=input_ids, labels=input_ids).loss.item()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"loading checkpoint: {args.checkpoint}")
    model = EshmunForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32).to(device)
    model.eval()

    df = pd.read_parquet(args.dataset)

    rows = []
    for source in sorted(df["source"].unique()):
        for text in df.loc[df["source"] == source, "content"]:
            loss = sequence_loss(model, tokenizer, text, args.max_length, device)
            if loss is not None:
                rows.append({"source": source, "loss": loss})

    results = pd.DataFrame(rows)
    print(f"\n{'source':<10s} {'n':>4s} {'mean_loss':>10s} {'mean_ppl':>14s}")
    for source, group in results.groupby("source"):
        mean_loss = group["loss"].mean()
        mean_ppl = torch.exp(torch.tensor(mean_loss)).item()
        print(f"{source:<10s} {len(group):>4d} {mean_loss:>10.4f} {mean_ppl:>14.4f}")

    overall_loss = results["loss"].mean()
    overall_ppl = torch.exp(torch.tensor(overall_loss)).item()
    print(f"{'overall':<10s} {len(results):>4d} {overall_loss:>10.4f} {overall_ppl:>14.4f}")


if __name__ == "__main__":
    main()
