"""
Protein-space-only perplexity: isolate how well the teacher predicts the
NEXT RESIDUE given that it is emitting a residue, factoring out its (already
diagnosed, see eval_pretrain_mix_perplexity.py) confusion about whether to
emit a residue at all versus plain English.

For each UniRef50 sample: `Ƥ`-prefix every residue and wrap in
<protein>...</protein> (reasoning.py's encode_sequence), tokenize, shift
input_ids into labels (standard causal-LM next-token shift), then at each
position whose *label* is one of the 20 dedicated residue tokens (ƤA..ƤY),
restrict the softmax to just those 20 logits (renormalized -- not the full
~50K vocab) and score the true residue there. Positions whose label isn't a
residue token (<protein>, </protein>, </s>) are excluded, not scored against
an undefined protein-space label.

Usage:
    python3 scripts/kothar/eval_protein_space_perplexity.py
"""

import os

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
N_SAMPLES = 10
MAX_LENGTH = 150

TEACHER_PATH = "/run/media/khairi/seagate/software/InstructProtein/model/InstructProtein"

_HERE = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_MIX_FILE = os.path.join(_HERE, "..", "..", "data", "kothar", "pretrain_mix.parquet")

RESIDUES = list("ACDEFGHIKLMNPQRSTVWY")
PROTEIN_TOKENS = [f"Ƥ{aa}" for aa in RESIDUES]


@torch.no_grad()
def protein_space_loss(model, tokenizer, text: str, protein_token_ids: list[int], device) -> tuple[float, int]:
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].to(device)
    logits = model(input_ids=input_ids).logits[0]  # (seq, vocab)

    shift_logits = logits[:-1]              # predict position i+1 from logits at i
    shift_labels = input_ids[0, 1:]

    id_to_slot = {tid: i for i, tid in enumerate(protein_token_ids)}
    mask = torch.tensor([lbl.item() in id_to_slot for lbl in shift_labels], device=device)
    if mask.sum() == 0:
        return 0.0, 0

    residue_logits = shift_logits[mask][:, protein_token_ids]         # (n_residues, 20)
    residue_labels = torch.tensor(
        [id_to_slot[lbl.item()] for lbl in shift_labels[mask]], device=device
    )
    losses = F.cross_entropy(residue_logits, residue_labels, reduction="sum")
    return losses.item(), int(mask.sum().item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading teacher: {TEACHER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    model = AutoModelForCausalLM.from_pretrained(TEACHER_PATH, dtype=torch.float32).to(device)
    model.eval()

    protein_token_ids = tokenizer.convert_tokens_to_ids(PROTEIN_TOKENS)
    assert tokenizer.unk_token_id not in protein_token_ids, "not all 20 residue tokens found in tokenizer"

    df = pd.read_parquet(PRETRAIN_MIX_FILE)
    sample = df[df["source"] == "uniref50"].sample(n=N_SAMPLES, random_state=SEED)

    total_loss, total_residues = 0.0, 0
    print(f"\n{'sample':>6s} {'n_residues':>10s} {'mean_loss':>10s} {'ppl(<=20)':>10s}")
    for i, text in enumerate(sample["content"]):
        loss_sum, n = protein_space_loss(model, tokenizer, text, protein_token_ids, device)
        mean_loss = loss_sum / n if n else float("nan")
        ppl = torch.exp(torch.tensor(mean_loss)).item() if n else float("nan")
        print(f"{i:>6d} {n:>10d} {mean_loss:>10.4f} {ppl:>10.4f}")
        total_loss += loss_sum
        total_residues += n

    overall_mean_loss = total_loss / total_residues
    overall_ppl = torch.exp(torch.tensor(overall_mean_loss)).item()
    print(f"\noverall: {total_residues} residues, mean_loss={overall_mean_loss:.4f}, protein-space perplexity={overall_ppl:.4f} (max=20)")


if __name__ == "__main__":
    main()
