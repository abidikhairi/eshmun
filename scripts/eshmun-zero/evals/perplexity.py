import argparse
import math
import os

import pandas as pd
import torch
from tqdm import tqdm

from eshmun.models.zero.tokenization import EshmunZeroTokenizer
from eshmun.common.finetuning_commons import LENGTH_BUCKETS, length_bucket, load_model, maskable_positions


NUM_PROTEINS = 100


def pseudo_perplexity(model, tokenizer, input_ids: torch.Tensor, device, batch_size: int) -> float | None:
    """Pseudo-perplexity via mask-one-position-at-a-time (Salazar et al.)."""
    positions = maskable_positions(input_ids, tokenizer)
    if not positions:
        return None

    total_nll = 0.0
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        
        batch_input_ids = input_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
        for row, pos in enumerate(chunk):
            batch_input_ids[row, pos] = tokenizer.mask_token_id

        labels = batch_input_ids.clone().to(device)
        labels[labels != tokenizer.mask_token_id] = -100
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = torch.ones_like(batch_input_ids)

        with torch.no_grad():
            output = model(input_ids=batch_input_ids, attention_mask=attention_mask, labels=labels)
            logits = output.logits

        log_probs = torch.log_softmax(logits, dim=-1)
        for row, pos in enumerate(chunk):
            true_token = int(input_ids[pos].item())
            total_nll += -log_probs[row, pos, true_token].item()

    avg_nll = total_nll / len(positions)
    return math.exp(avg_nll)


def main(args):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    tokenizer = EshmunZeroTokenizer()
    model = load_model(args.model, device)

    df = pd.read_csv(args.csv, sep='\t')
    sequences = df[args.sequence_column][:NUM_PROTEINS].astype(str).tolist()

    rows = []
    for idx, seq in enumerate(tqdm(sequences, desc="Computing perplexity")):
        encoding = tokenizer(seq, truncation=True, max_length=model.config.max_seq_len, return_tensors="pt")
        input_ids = encoding["input_ids"][0]

        ppl = pseudo_perplexity(model, tokenizer, input_ids, device, args.batch_size)
        rows.append({"index": idx, "length": len(seq), "perplexity": ppl, "bucket": length_bucket(len(seq))})

    results_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results_df.to_csv(args.output, index=False)

    summary_rows = []
    for bucket in LENGTH_BUCKETS + ["all"]:
        subset = results_df if bucket == "all" else results_df[results_df["bucket"] == bucket]
        subset = subset.dropna(subset=["perplexity"])  # type: ignore[arg-type]
        if len(subset) == 0:
            summary_rows.append({"bucket": bucket, "count": 0, "mean_ppl": None, "std_ppl": None, "min_ppl": None, "max_ppl": None})
            continue

        summary_rows.append(
            # pyrefly: ignore [bad-argument-type]
            {
                "bucket": bucket,
                "count": len(subset),
                "mean_ppl": subset["perplexity"].mean(),
                "std_ppl": subset["perplexity"].std(),
                "min_ppl": subset["perplexity"].min(),
                "max_ppl": subset["perplexity"].max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary_output, index=False)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute pseudo-perplexity of protein sequences, bucketed by length.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with protein sequences.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained EshmunZero model.")
    parser.add_argument("--sequence-column", dest="sequence_column", type=str, default="sequence", help="CSV column containing sequences.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of masked copies scored per forward pass.")
    parser.add_argument("--output", type=str, default="perplexity_results.csv", help="Where to save per-sequence results.")
    parser.add_argument("--summary-output", type=str, default="perplexity_summary.csv", help="Where to save the bucketed summary.")

    main(parser.parse_args())
