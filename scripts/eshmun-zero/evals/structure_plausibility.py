import argparse
import os
import random

import pandas as pd
import torch
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from eshmun.models.zero.tokenization import EshmunZeroTokenizer
from eshmun.common.finetuning_commons import load_model, maskable_positions


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
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    tokenizer = EshmunZeroTokenizer()
    model = load_model(args.model, device)
    model_name = os.path.basename(os.path.normpath(args.model))

    df = pd.read_csv(args.csv, sep='\t').take(indices=range(100))
    
    sequences = df[args.sequence_column][:NUM_PROTEINS].astype(str).tolist()

    records = []
    for idx, seq in enumerate(tqdm(sequences, desc="Structure plausibility")):
        encoding = tokenizer(seq, truncation=True, max_length=model.config.max_seq_len, return_tensors="pt")
        input_ids = encoding["input_ids"][0]

        masked_input_ids, masked_positions = mask_sequence(input_ids, tokenizer, args.mlm_probability, rng)

        attention_mask = torch.ones_like(masked_input_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_ids=masked_input_ids.unsqueeze(0).to(device), attention_mask=attention_mask).logits[0]

        top1 = logits.argmax(dim=-1).cpu()

        reconstructed = input_ids.clone()
        for pos in masked_positions:
            reconstructed[pos] = top1[pos]

        seq_str = str(tokenizer.decode(reconstructed, skip_special_tokens=True)).replace(" ", "")
        records.append(SeqRecord(Seq(seq_str), id=f"{model_name}_{idx}", description=""))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    SeqIO.write(records, args.output, "fasta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct sequences from 15% masked tokens (top-1 prediction) and save to FASTA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with protein sequences.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained EshmunZero model.")
    parser.add_argument("--sequence-column", dest="sequence_column", type=str, default="sequence", help="CSV column containing sequences.")
    parser.add_argument("--mlm-probability", type=float, default=0.15, help="Fraction of tokens to mask per sequence.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for masking.")
    parser.add_argument("--output", type=str, default="structure_plausibility.fasta", help="Where to save the reconstructed sequences.")

    main(parser.parse_args())
