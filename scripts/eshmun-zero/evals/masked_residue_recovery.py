import argparse
import os
import random

import pandas as pd
import torch
from tqdm import tqdm

from eshmun.models.zero.tokenization import EshmunZeroTokenizer

from eshmun.common.finetuning_commons import load_model, maskable_positions


TOP_K_VALUES = (1, 3, 5, 10)
NUM_PROTEINS = 100

def mask_sequence(input_ids: torch.Tensor, tokenizer: EshmunZeroTokenizer, mlm_probability: float, rng: random.Random):
    positions = maskable_positions(input_ids, tokenizer)
    n_mask = max(1, round(len(positions) * mlm_probability))
    masked_positions = rng.sample(positions, min(n_mask, len(positions)))

    mask_token_id = int(tokenizer.mask_token_id)  # type: ignore[arg-type]
    masked_input_ids = input_ids.clone()
    for pos in masked_positions:
        masked_input_ids[pos] = mask_token_id

    return masked_input_ids, masked_positions


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    tokenizer = EshmunZeroTokenizer()
    model = load_model(args.model, device)

    df = pd.read_csv(args.csv, sep='\t')
    sequences = df[args.sequence_column][:NUM_PROTEINS].astype(str).tolist()

    max_k = max(TOP_K_VALUES)
    total_correct = {k: 0 for k in TOP_K_VALUES}
    total_masked = 0

    per_sequence_rows = []
    for idx, seq in enumerate(tqdm(sequences, desc="Masked residue recovery")):
        encoding = tokenizer(seq, truncation=True, max_length=model.config.max_seq_len, return_tensors="pt")
        input_ids = encoding["input_ids"][0]

        masked_input_ids, masked_positions = mask_sequence(input_ids, tokenizer, args.mlm_probability, rng)

        attention_mask = torch.ones_like(masked_input_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids=masked_input_ids.unsqueeze(0).to(device), attention_mask=attention_mask).logits[0]

        topk = logits.topk(max_k, dim=-1).indices.tolist()

        seq_correct = {k: 0 for k in TOP_K_VALUES}
        for pos in masked_positions:
            true_token = input_ids[pos].item()
            preds = topk[pos]
            for k in TOP_K_VALUES:
                if true_token in preds[:k]:
                    seq_correct[k] += 1
                    total_correct[k] += 1

        total_masked += len(masked_positions)

        row: dict[str, float] = {"index": idx, "length": len(seq), "num_masked": len(masked_positions)}
        for k in TOP_K_VALUES:
            row[f"acc@{k}"] = seq_correct[k] / len(masked_positions)
        per_sequence_rows.append(row)

    per_sequence_df = pd.DataFrame(per_sequence_rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    per_sequence_df.to_csv(args.output, index=False)

    summary: dict[str, float] = {"num_sequences": len(sequences), "total_masked_residues": total_masked}
    for k in TOP_K_VALUES:
        summary[f"acc@{k}"] = total_correct[k] / total_masked

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(args.summary_output, index=False)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate masked residue recovery accuracy@{1,3,5,10}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with protein sequences.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained EshmunZero model.")
    parser.add_argument("--sequence-column", dest="sequence_column", type=str, default="sequence", help="CSV column containing sequences.")
    parser.add_argument("--mlm-probability", type=float, default=0.15, help="Fraction of residues to mask per sequence.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for masking.")
    parser.add_argument("--output", type=str, default="masked_residue_recovery_results.csv", help="Where to save per-sequence results.")
    parser.add_argument("--summary-output", type=str, default="masked_residue_recovery_summary.csv", help="Where to save the overall summary.")

    main(parser.parse_args())
