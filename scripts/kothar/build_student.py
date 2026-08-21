"""
Build the Kothar student model: a 6-layer subset of InstructProtein
(hicai-zju/InstructProtein, OPT-1.3B architecture, 24 layers).

Layer selection: 2 from the start, 2 from the middle, 2 from the end
(teacher indices [0, 1, 11, 12, 22, 23] for a 24-layer teacher), keeping
the teacher's embeddings, positional embeddings, final layer norm, and
(tied) LM head as-is. `<think>`/`</think>` are then added to the tokenizer
and the student's embedding/LM head are resized to match.

Loads and pushes in float32 (float16 has caused issues for this project
before). Requires `hf auth login` with write access to the `khairi` namespace.

Usage:
    python3 scripts/kothar/build_student.py            # build, verify, push
    python3 scripts/kothar/build_student.py --no-push   # build and verify only
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eshmun.models.eshmun import EshmunConfig, EshmunForCausalLM

TEACHER_ID = "hicai-zju/InstructProtein"
THINK_TOKENS = ["<think>", "</think>"]


def select_layers(num_teacher_layers: int) -> list[int]:
    """First two, middle two (straddling the center), last two."""
    mid = (num_teacher_layers - 1) / 2
    mid_lo, mid_hi = int(mid), int(mid) + 1
    return [0, 1, mid_lo, mid_hi, num_teacher_layers - 2, num_teacher_layers - 1]


def build_student(teacher, selected_layers: list[int]) -> EshmunForCausalLM:
    teacher_config = teacher.config
    student_config = EshmunConfig(
        vocab_size=teacher_config.vocab_size,
        hidden_size=teacher_config.hidden_size,
        num_hidden_layers=len(selected_layers),
        ffn_dim=teacher_config.ffn_dim,
        max_position_embeddings=teacher_config.max_position_embeddings,
        do_layer_norm_before=teacher_config.do_layer_norm_before,
        word_embed_proj_dim=teacher_config.word_embed_proj_dim,
        dropout=teacher_config.dropout,
        attention_dropout=teacher_config.attention_dropout,
        num_attention_heads=teacher_config.num_attention_heads,
        activation_function=teacher_config.activation_function,
        layerdrop=teacher_config.layerdrop,
        init_std=teacher_config.init_std,
        use_cache=teacher_config.use_cache,
        pad_token_id=teacher_config.pad_token_id,
        bos_token_id=teacher_config.bos_token_id,
        eos_token_id=teacher_config.eos_token_id,
        enable_bias=teacher_config.enable_bias,
        layer_norm_elementwise_affine=teacher_config.layer_norm_elementwise_affine,
        tie_word_embeddings=teacher_config.tie_word_embeddings,
        dtype=torch.float32,
    )
    student = EshmunForCausalLM(student_config)

    teacher_sd = teacher.state_dict()
    student_sd: dict[str, torch.Tensor] = {}
    for key, value in teacher_sd.items():
        prefix = "model.decoder.layers."
        if key.startswith(prefix):
            rest = key[len(prefix):]
            layer_idx_str, param_name = rest.split(".", 1)
            layer_idx = int(layer_idx_str)
            if layer_idx not in selected_layers:
                continue
            new_idx = selected_layers.index(layer_idx)
            student_sd[f"{prefix}{new_idx}.{param_name}"] = value.clone()
        else:
            student_sd[key] = value.clone()

    missing, unexpected = student.load_state_dict(student_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return student


def add_think_tokens(student: EshmunForCausalLM, tokenizer) -> None:
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": THINK_TOKENS})
    assert num_added == len(THINK_TOKENS), f"expected to add {len(THINK_TOKENS)} new tokens, got {num_added}"
    student.resize_token_embeddings(len(tokenizer))


def count_params_millions(model: torch.nn.Module) -> int:
    return round(sum(p.numel() for p in model.parameters()) / 1_000_000)


def sanity_forward(model: EshmunForCausalLM, tokenizer) -> None:
    model.eval()
    inputs = tokenizer("<think>", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    assert torch.isfinite(out.logits).all(), "student produced non-finite logits"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true", help="build and verify only, skip the Hub push")
    args = parser.parse_args()

    print(f"Loading teacher: {TEACHER_ID}", file=sys.stderr)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_ID, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)

    selected_layers = select_layers(teacher.config.num_hidden_layers)
    print(f"Selected teacher layers: {selected_layers}", file=sys.stderr)

    student = build_student(teacher, selected_layers)
    add_think_tokens(student, tokenizer)
    sanity_forward(student, tokenizer)

    n_params_m = count_params_millions(student)
    repo_id = f"khairi/Kothar-student-seed-{n_params_m}M"
    print(f"Student params: {n_params_m}M -> {repo_id}", file=sys.stderr)

    if args.no_push:
        print("--no-push set, skipping Hub upload.", file=sys.stderr)
        return

    student.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print(f"Pushed to https://huggingface.co/{repo_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
