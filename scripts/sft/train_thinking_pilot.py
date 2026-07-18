"""
LoRA SFT of the Eshmun-Thinking-Pilot base checkpoint on the thinking-aware pilot
dataset, using TRL. Drives both legs of the ablation study
(docs/thinking_pilot_evaluation_protocol.md): thinking (T1) vs. direct/non-thinking
(B1) SFT, same base model, same LoRA recipe, same entry-level train/val/test split
-- only the presence of the <think> block in the training target differs.

Base model: khairi/Eshmun-Thinking-Pilot   (InstructProtein + <think>/</think> tokens)
Dataset:    khairi/eshmun-thinking-pilot, one of four configs (annotation/generation
            x thinking/non_thinking), each with train/validation/test splits built by
            scripts/python/thinking/{split_annotation_by_identity,split_generation_random,
            build_ablation_datasets}.py in the external data repo.

Each dataset row (Entry, Instruction, Answer, [Reasoning]) is reshaped into a
TRL prompt-completion pair:
    prompt     = Instruction
    completion = "<think>\n{Reasoning}\n</think>\n{Answer}"   (thinking configs)
    completion = "{Answer}"                                    (non_thinking configs;
                                                                 no Reasoning column)

With this shape TRL's SFTTrainer defaults to completion-only loss (the model
isn't trained to predict the instruction, only the reasoning + answer) and
automatically appends the tokenizer's eos_token to the end of every
completion during dataset preparation.

Not run/validated here by design -- writing the pipeline only.

Usage:
    # T1 (thinking), both directions combined:
    python3 scripts/sft/train_thinking_pilot.py \\
        --dataset-configs annotation_thinking,generation_thinking \\
        --output-dir checkpoints/eshmun-thinking-pilot-t1

    # B1 (direct-SFT ablation), same split, no reasoning trace:
    python3 scripts/sft/train_thinking_pilot.py \\
        --dataset-configs annotation_non_thinking,generation_non_thinking \\
        --output-dir checkpoints/eshmun-thinking-pilot-b1
"""

import argparse

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "khairi/Eshmun-Thinking-Pilot"
DATASET = "khairi/eshmun-thinking-pilot"
DATASET_CONFIGS = ["annotation_thinking", "generation_thinking"]

# Only the attention projections get LoRA adapters, per the project's established
# recipe (see docs/paper/eshmun.md): rank 64, alpha 128, all attention projections.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

# embed_tokens/lm_head are an nn.Embedding/nn.Linear pair, not attention projections,
# so LoRA_TARGET_MODULES never touches them and they stay frozen by default. That
# would permanently strand the two newly added <think>/</think> embeddings at their
# (untrained, mean-init) starting value with zero gradient.
#
# trainable_token_indices (https://huggingface.co/docs/peft/package_reference/trainable_tokens,
# LoraConfig param: https://huggingface.co/docs/peft/package_reference/lora) trains only the
# specified embedding rows instead of the whole matrix -- unlike modules_to_save, which would
# make the full 50304x2048 embedding trainable (all pretrained rows included, not just the 2
# new ones). PEFT auto-detects that embed_tokens/lm_head are tied on this checkpoint
# (`model._tied_weights_keys`, `config.tie_word_embeddings=True`, verified directly against
# this checkpoint) and applies a linked adapter to lm_head too, so listing embed_tokens alone
# is enough -- no separate lm_head entry needed.
NEW_TOKENS = ["<think>", "</think>"]


def build_example(example: dict) -> dict:
    reasoning = example.get("Reasoning")
    completion = (
        f"<think>\n{reasoning}\n</think>\n{example['Answer']}"
        if reasoning is not None
        else example["Answer"]
    )
    return {"prompt": example["Instruction"], "completion": completion}


def load_split(dataset_name: str, configs: list[str], split: str):
    """Load and concatenate one split across multiple dataset configs (e.g. the
    annotation + generation halves of a single condition)."""
    parts = [load_dataset(dataset_name, config, split=split) for config in configs]
    # pyrefly: ignore [bad-specialization]
    return concatenate_datasets(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument(
        "--dataset-configs", default=",".join(DATASET_CONFIGS),
        help="Comma-separated dataset config names to concatenate for training, e.g. "
             "'annotation_thinking,generation_thinking' for T1 or "
             "'annotation_non_thinking,generation_non_thinking' for the B1 ablation.",
    )
    parser.add_argument(
        "--eval-split", default="validation",
        help="Split to use for periodic eval loss during training. Pass '' to disable.",
    )
    parser.add_argument("--output-dir", default="checkpoints/eshmun-thinking-pilot-lora")

    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Must not exceed the base model's max_position_embeddings (2048 here). "
             "Examples whose tokenized prompt+completion exceeds this are dropped, not "
             "truncated -- truncating from the right would silently cut off the answer.",
    )
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="QLoRA-style 4-bit base model loading, for GPUs with less VRAM than a "
             "Colab instance typically provides. Combined with --dtype float32 for the "
             "trainable adapters, only the frozen base weights are quantized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )

    # float32 per project convention -- run on Colab, which has the VRAM headroom this
    # 1.3B-parameter base model needs in full precision (unlike the local 6GB GPU this
    # was first scoped against).
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False  # required alongside gradient checkpointing
    model.enable_input_require_grads()  # keep gradients flowing into LoRA/embed adapters

    configs = args.dataset_configs.split(",")

    def fits_context(example: dict) -> bool:
        full_text = example["prompt"] + example["completion"]
        # pyrefly: ignore [not-callable]
        return len(tokenizer(full_text)["input_ids"]) <= args.max_length

    def prepare(split: str):
        ds = load_split(args.dataset, configs, split)
        ds = ds.map(build_example, remove_columns=ds.column_names)
        before = len(ds)
        ds = ds.filter(fits_context)
        print(f"[{split}] dropped {before - len(ds)}/{before} examples exceeding max_length={args.max_length}")
        return ds

    dataset = prepare("train")

    eval_dataset = None
    if args.eval_split:
        eval_dataset = prepare(args.eval_split)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        # pyrefly: ignore [missing-attribute]
        trainable_token_indices={"embed_tokens": tokenizer.convert_tokens_to_ids(NEW_TOKENS)},
        task_type="CAUSAL_LM",
        bias="none",
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
        seed=args.seed,
        # eos_token=None -> defaults to tokenizer.eos_token ("</s>"); TRL appends it to
        # the end of every `completion` when preparing a prompt-completion dataset.
        # completion_only_loss=None -> defaults to True for prompt-completion datasets:
        # loss is computed only on "<think>...</think>\n{answer}", not the instruction.
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        # pyrefly: ignore [bad-argument-type]
        train_dataset=dataset,
        # pyrefly: ignore [bad-argument-type]
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
