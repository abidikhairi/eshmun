"""
Sanity-check the Kothar pretraining mix against the teacher: sample 10 rows
per subset from data/kothar/pretrain_mix.parquet and report the teacher's
(InstructProtein's) average perplexity on each subset.

Loads and runs in float32 (float16 has caused issues for this project
before). Uses the local teacher checkpoint rather than the Hub copy.

Usage:
    python3 scripts/kothar/eval_pretrain_mix_perplexity.py
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
N_PER_SUBSET = 10
MAX_LENGTH = 150  # short samples -- the 1.3B teacher in float32 barely fits
                  # this machine's 6GB GPU already, no headroom for long
                  # sequences (OOM'd at the model's full 2048-token limit)

TEACHER_PATH = "/run/media/khairi/seagate/software/InstructProtein/model/InstructProtein"

_HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_MIX_FILE = os.path.join(_HERE, "..", "..", "data", "kothar", "pretrain_mix.parquet")


@torch.no_grad()
def sequence_perplexity(model, tokenizer, text: str, max_length: int, device: torch.device) -> float | None:
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return None
    loss = model(input_ids=input_ids, labels=input_ids).loss
    return torch.exp(loss).item()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading teacher: {TEACHER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    model = AutoModelForCausalLM.from_pretrained(TEACHER_PATH, dtype=torch.float32).to(device)
    model.eval()

    max_length = MAX_LENGTH

    df = pd.read_parquet(PRETRAIN_MIX_FILE)

    print(f"\n{'subset':<10s} {'n':>3s} {'mean_ppl':>12s}   sample perplexities")
    for source in sorted(df["source"].unique()):
        sample = df[df["source"] == source].sample(n=N_PER_SUBSET, random_state=SEED)
        ppls = []
        for text in sample["content"]:
            ppl = sequence_perplexity(model, tokenizer, text, max_length, device)
            if ppl is not None:
                ppls.append(ppl)
        mean_ppl = sum(ppls) / len(ppls)
        formatted = ", ".join(f"{p:.2f}" for p in ppls)
        print(f"{source:<10s} {len(ppls):>3d} {mean_ppl:>12.4f}   [{formatted}]")


if __name__ == "__main__":
    main()
