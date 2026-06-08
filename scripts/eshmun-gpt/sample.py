import argparse

import torch
from transformers import GenerationConfig
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord

from eshmun.tokenization import EshmunTokenizer
from eshmun.models.gpt import EshmunGPT


TOKENIZER_PATH = 'data/eshmun-gpt/tokenizer'


def main(args):
    tokenizer = EshmunTokenizer.from_pretrained(TOKENIZER_PATH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EshmunGPT.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(device)  # type: ignore[arg-type]
    model.eval()

    # Seed generation from the BOS token (unconditional sampling)
    input_ids = tokenizer(
        tokenizer.bos_token,
        return_tensors='pt',
        add_special_tokens=False,
    ).input_ids.to(device)

    generation_config = GenerationConfig(
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    sequences = []
    with torch.no_grad():
        for _ in tqdm(range(args.num_sequences), desc="Generating", unit="seq"):
            output_ids = model.generate(input_ids, generation_config=generation_config)  # type: ignore[attr-defined]
            sequence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            sequence = sequence.replace(' ', '')
            sequences.append(SeqRecord(id=f"Sequence{_:03d}", name=f"Sequence{_:03d}", description="", seq=Seq(sequence)))

    for i, rec in enumerate(sequences, 1):
        print(rec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample protein sequences from a pretrained EshmunGPT model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', required=True,
                        help='Path or HuggingFace Hub ID of the EshmunGPT checkpoint.')
    parser.add_argument('--num_sequences', type=int, default=10,
                        help='Number of sequences to generate.')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate per sequence.')
    parser.add_argument('--top_k', type=int, default=950,
                        help='Top-k sampling cutoff.')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus (top-p) sampling probability threshold.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature.')
    parser.add_argument('--repetition_penalty', type=float, default=1.3,
                        help='Penalty applied to already-seen tokens.')

    args = parser.parse_args()
    main(args)
