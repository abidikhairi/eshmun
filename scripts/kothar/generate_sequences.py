"""
Protein sequence generation from a Kothar checkpoint: prompt the model with
`<bos><protein>` (optionally followed by a fixed amino-acid prefix) and let
it sample the rest of the residues until it either emits `</protein>` or
hits --max-new-tokens.

Sequences are tokenized with the InstructProtein-derived vocab's dedicated
per-residue tokens (`ƤA`..`ƤY`, see reasoning.py's encode_sequence) wrapped
in `<protein>...</protein>`; output sequences are the plain amino-acid
strings (prefix included) with the `Ƥ` marker stripped back out.

Loads and generates in float32 (float16 has caused issues for this project
before).

Usage:
    python3 scripts/kothar/generate_sequences.py --checkpoint checkpoints/kothar-pretrain-409m/checkpoint-2500
    python3 scripts/kothar/generate_sequences.py --checkpoint checkpoints/kothar-pretrain-409m/checkpoint-2500 \\
        --num-sequences 30 --output-fasta generations/checkpoint-2500.fasta
    python3 scripts/kothar/generate_sequences.py --checkpoint checkpoints/kothar-pretrain-409m/checkpoint-2500 \\
        --prefix MKT --num-sequences 30
"""

import argparse
import os

import torch
from transformers import AutoTokenizer

from eshmun.models.eshmun import EshmunForCausalLM

STUDENT_ID = "khairi/Kothar-student-seed-409M"

PROTEIN_START = "<protein>"
PROTEIN_END = "</protein>"
PROTEIN_PREFIX_TOKEN = "Ƥ"
RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")

SEED = 4242


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint-N directory")
    parser.add_argument("--tokenizer", default=STUDENT_ID, help="tokenizer source; checkpoints don't carry their own")
    parser.add_argument("--prefix", default="", help="fixed amino-acid prefix to seed generation with, e.g. MKT")
    parser.add_argument("--num-sequences", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--greedy", dest="do_sample", action="store_false", help="disable sampling (greedy decoding)")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-fasta", default=None, help="if set, write generated sequences here as FASTA")
    args = parser.parse_args()

    args.prefix = args.prefix.upper()
    invalid = set(args.prefix) - RESIDUES
    if invalid:
        parser.error(f"--prefix contains non-residue characters: {sorted(invalid)}")

    return args


def extract_sequence(decoded: str) -> tuple[str, bool]:
    """Pulls the residues out of a decoded `<protein>Ƥa Ƥb ...(</protein>)` span.

    Returns (sequence, complete) where complete indicates whether the model
    actually closed the tag within --max-new-tokens.
    """
    start = decoded.find(PROTEIN_START)
    if start == -1:
        return "", False
    start += len(PROTEIN_START)
    end = decoded.find(PROTEIN_END, start)
    complete = end != -1
    body = decoded[start:end if complete else None]
    residues = body.replace(PROTEIN_PREFIX_TOKEN, "")
    return residues, complete


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"loading checkpoint: {args.checkpoint}")
    model = EshmunForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32).to(device)
    model.eval()

    torch.manual_seed(args.seed)

    prompt_text = PROTEIN_START + "".join(f"{PROTEIN_PREFIX_TOKEN}{aa}" for aa in args.prefix)
    if args.prefix:
        print(f"prefix: {args.prefix!r} ({len(args.prefix)} residues)")
    prompt_ids = tokenizer(prompt_text)["input_ids"]
    input_ids = torch.tensor([prompt_ids]).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            num_return_sequences=args.num_sequences,
            top_p=args.top_p if args.do_sample else None,
            top_k=args.top_k if args.do_sample else None,
            repetition_penalty=args.repetition_penalty if args.do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    records = []
    for i, ids in enumerate(output_ids):
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        sequence, complete = extract_sequence(decoded)
        records.append((sequence, complete))
        status = "" if complete else "  [incomplete: hit --max-new-tokens before </protein>]"
        print(f">seq_{i} len={len(sequence)}{status}\n{sequence}")

    n_complete = sum(1 for _, complete in records if complete)
    print(f"\n{n_complete}/{len(records)} sequences closed within {args.max_new_tokens} new tokens")

    if args.output_fasta:
        os.makedirs(os.path.dirname(args.output_fasta) or ".", exist_ok=True)
        with open(args.output_fasta, "w") as f:
            for i, (sequence, complete) in enumerate(records):
                tag = "" if complete else " incomplete"
                f.write(f">seq_{i}{tag}\n{sequence}\n")
        print(f"wrote {len(records)} sequences to {args.output_fasta}")


if __name__ == "__main__":
    main()
