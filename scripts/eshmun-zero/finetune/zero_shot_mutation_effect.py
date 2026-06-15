"""
Zero-shot mutation effect prediction on a ProteinGym-style DMS substitution benchmark.

For each unique mutated position in the wild-type sequence, the position is masked
and a single forward pass yields the model's per-token log-probabilities (the
"masked-marginal" scoring strategy). A mutant's score is the sum, over its mutated
positions, of log P(mutant_aa) - log P(wt_aa). Predicted scores are correlated
(Spearman) against the experimental DMS_score column.

Expected CSV columns (ProteinGym DMS substitution format):
  - mutant: e.g. "A123T" or "A123T:G56P" (1-indexed positions into the wild-type sequence)
  - DMS_score: experimental fitness score
"""

import argparse
import os

import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from Bio import SeqIO

from eshmun.models.zero.tokenization import EshmunZeroTokenizer

from common import load_model


def parse_mutant(mutant_str: str) -> list[tuple[str, int, str]]:
    """Parse 'A123T' or 'A123T:G56P' into [(wt_aa, position, mut_aa), ...] (1-indexed)."""
    mutations = []
    for token in mutant_str.split(":"):
        token = token.strip()
        wt_aa, pos, mut_aa = token[0], int(token[1:-1]), token[-1]
        mutations.append((wt_aa, pos, mut_aa))
    return mutations


def load_wt_sequence(args) -> str:
    if args.wt_sequence is not None:
        return args.wt_sequence.strip().upper()

    record = next(SeqIO.parse(args.wt_fasta, "fasta"))
    return str(record.seq).upper()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = EshmunZeroTokenizer()
    model = load_model(args.model, device)
    model.eval()

    wt_sequence = load_wt_sequence(args)

    df = pd.read_csv(args.csv)
    mutant_lists = df[args.mutant_column].astype(str).apply(parse_mutant).tolist()

    # 1-indexed wt_sequence positions -> 0-indexed token positions (offset by <bos>).
    unique_positions = sorted({pos - 1 for muts in mutant_lists for _, pos, _ in muts})

    encoding = tokenizer(wt_sequence, truncation=True, max_length=model.config.max_seq_len, return_tensors="pt")
    wt_input_ids = encoding["input_ids"][0]
    bos_offset = 1 if tokenizer.add_bos_token else 0

    log_probs_by_position: dict[int, torch.Tensor] = {}
    for start in tqdm(range(0, len(unique_positions), args.batch_size), desc="Scoring positions"):
        chunk = unique_positions[start : start + args.batch_size]

        batch_input_ids = wt_input_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
        for row, pos in enumerate(chunk):
            batch_input_ids[row, pos + bos_offset] = int(tokenizer.mask_token_id)  # type: ignore[arg-type]

        batch_input_ids = batch_input_ids.to(device)
        attention_mask = torch.ones_like(batch_input_ids)

        with torch.no_grad():
            logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits

        log_probs = torch.log_softmax(logits, dim=-1)
        for row, pos in enumerate(chunk):
            log_probs_by_position[pos] = log_probs[row, pos + bos_offset].cpu()

    scores = []
    for muts in tqdm(mutant_lists, desc="Scoring mutants"):
        score = 0.0
        for wt_aa, pos, mut_aa in muts:
            log_probs = log_probs_by_position[pos - 1]
            wt_id = tokenizer.aa_to_id(wt_aa)
            mut_id = tokenizer.aa_to_id(mut_aa)
            score += (log_probs[mut_id] - log_probs[wt_id]).item()
        scores.append(score)

    df["predicted_score"] = scores

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    if args.dms_column in df.columns:
        rho, pvalue = spearmanr(df["predicted_score"], df[args.dms_column])
        print(f"Spearman rho={rho:.4f}  p-value={pvalue:.4e}  n={len(df)}")
    else:
        print(f"Column '{args.dms_column}' not found, skipping correlation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot mutation effect prediction (ProteinGym-style masked-marginal scoring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to a ProteinGym DMS substitution CSV.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained EshmunZero model.")

    wt_group = parser.add_mutually_exclusive_group(required=True)
    wt_group.add_argument("--wt-sequence", type=str, default=None, help="Wild-type sequence as a string.")
    wt_group.add_argument("--wt-fasta", type=str, default=None, help="Path to a FASTA file containing the wild-type sequence.")

    parser.add_argument("--mutant-column", type=str, default="mutant", help="CSV column with mutation strings (e.g. 'A123T' or 'A123T:G56P').")
    parser.add_argument("--dms-column", type=str, default="DMS_score", help="CSV column with experimental fitness scores.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of masked positions scored per forward pass.")
    parser.add_argument("--output", type=str, default="mutation_effect_scores.csv", help="Where to save per-mutant predicted scores.")

    main(parser.parse_args())
