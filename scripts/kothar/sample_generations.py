"""
Qualitative check: for each source in the held-out validation set, sample one
example, feed the model only its first 5 tokens, and generate a continuation
-- a quick look at what the checkpoint has actually learned to produce per
source, alongside the loss/ppl numbers from eval_checkpoint.py.

Defaults to greedy decoding; pass --do-sample for nucleus/top-k sampling
with repetition penalty (greedy tends to collapse into repetition loops on
an early, undertrained checkpoint).

Loads and generates in float32 (float16 has caused issues for this project
before).

Usage:
    python3 scripts/kothar/sample_generations.py --checkpoint checkpoints/kothar-pretrain-409m/checkpoint-1000
    python3 scripts/kothar/sample_generations.py --checkpoint checkpoints/kothar-pretrain-409m/checkpoint-1000 \\
        --do-sample --top-p 0.95 --top-k 250 --repetition-penalty 1.3
"""

import argparse
import os

import pandas as pd
import torch
from transformers import AutoTokenizer

from eshmun.models.eshmun import EshmunForCausalLM

STUDENT_ID = "khairi/Kothar-student-seed-409M"

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(_HERE, "..", "..", "data", "kothar", "valid_holdout.parquet")

PROMPT_TOKENS = 5
MAX_NEW_TOKENS = 100
SEED = 4242


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint-N directory")
    parser.add_argument("--tokenizer", default=STUDENT_ID, help="tokenizer source; checkpoints don't carry their own")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"loading checkpoint: {args.checkpoint}")
    model = EshmunForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32).to(device)
    model.eval()

    torch.manual_seed(SEED)
    df = pd.read_parquet(args.dataset)

    for source in sorted(df["source"].unique()):
        example = df.loc[df["source"] == source].sample(n=1, random_state=SEED).iloc[0]
        full_ids = tokenizer(example["content"])["input_ids"]
        prompt_ids = full_ids[:PROMPT_TOKENS]

        input_ids = torch.tensor([prompt_ids]).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=args.do_sample,
                top_p=args.top_p if args.do_sample else None,
                top_k=args.top_k if args.do_sample else None,
                repetition_penalty=args.repetition_penalty if args.do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_text = tokenizer.decode(prompt_ids)
        generated_text = tokenizer.decode(output_ids[0][len(prompt_ids):], skip_special_tokens=False)

        print(f"\n=== source: {source} (entry: {example.get('entry', '?')}) ===")
        print(f"prompt ({PROMPT_TOKENS} tokens): {prompt_text!r}")
        print(f"generated continuation: {generated_text!r}")
        print(f"original (truncated to 300 chars): {example['content'][:300]!r}")


if __name__ == "__main__":
    main()
