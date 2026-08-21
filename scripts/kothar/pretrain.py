"""
Kothar Stage 1: continued pretraining of the 409M student on the UniRef50 +
natural-text replay mix (khairi/kothar-pretrain-mix-v1).

Warm-starts from the layer-subsetted student checkpoint
(khairi/Kothar-student-seed-409M, built by build_student.py), not random
init. Plain causal-LM pretraining -- no teacher, no distillation loss (see
project decision: no distillation on the teacher).

Both the model and the dataset are loaded by their HF Hub ids, not a local
path, so this runs unchanged on any machine with the Hub reachable. Texts
are concatenated (each already carries its own leading </s>, acting as a
document separator) and packed into fixed-length blocks rather than padded
per-example, standard practice for pretraining.

Logs to Weights & Biases, project "Kothar".

Loads and trains in float32 (float16 has caused issues for this project
before).

Usage:
    python3 scripts/kothar/pretrain.py
    python3 scripts/kothar/pretrain.py --per-device-batch-size 8 --epochs 1
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from eshmun.models.eshmun import EshmunForCausalLM

STUDENT_ID = "khairi/Kothar-student-seed-409M"
DATASET_ID = "khairi/kothar-pretrain-mix-v1"

os.environ.setdefault("WANDB_PROJECT", "Kothar")


def tokenize(tokenizer):
    def _tokenize(batch):
        return tokenizer(batch["content"])

    return _tokenize


def group_texts(examples, block_size: int):
    """Concatenate all tokenized examples in the batch, then chop into
    contiguous `block_size` chunks -- standard causal-LM pretraining
    packing (see e.g. HF's run_clm.py). Drops the final partial block."""
    concatenated = {k: sum(examples[k], []) for k in examples}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/kothar-pretrain-409m")
    parser.add_argument("--block-size", type=int, default=2048, help="packed sequence length; matches the student's max_position_embeddings")
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--run-name", default="kothar-pretrain-409m")
    parser.add_argument("--num-proc", type=int, default=4, help="workers for tokenization/packing")
    parser.add_argument("--max-train-samples", type=int, default=None, help="cap on raw (pre-packing) dataset rows, for smoke tests")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps regardless of --epochs, for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    model = EshmunForCausalLM.from_pretrained(STUDENT_ID, dtype=torch.float32)

    dataset = load_dataset(DATASET_ID, split="train")
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    dataset = dataset.map(
        tokenize(tokenizer),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="tokenizing",
    )
    dataset = dataset.map(
        lambda examples: group_texts(examples, args.block_size),
        batched=True,
        num_proc=args.num_proc,
        desc=f"packing into {args.block_size}-token blocks",
    )

    # warmup_ratio is deprecated (transformers >=5.2) -- estimate total steps
    # ourselves for visibility; warmup_steps itself is a fixed CLI value.
    num_devices = max(torch.cuda.device_count(), 1)
    effective_batch_size = args.per_device_batch_size * args.gradient_accumulation_steps * num_devices
    steps_per_epoch = -(-len(dataset) // effective_batch_size)  # ceil division
    total_steps = args.max_steps if args.max_steps is not None else round(steps_per_epoch * args.epochs)
    print(
        f"{len(dataset)} packed blocks, effective batch size {effective_batch_size} "
        f"({args.per_device_batch_size} x {args.gradient_accumulation_steps} accum x {num_devices} device(s)) "
        f"-> {steps_per_epoch} steps/epoch, {total_steps} total steps, {args.warmup_steps} warmup steps"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        report_to=["wandb"],
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
