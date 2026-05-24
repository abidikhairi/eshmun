"""GRPO training entry point.

Usage
-----
Train from a YAML config:
    python scripts/grpo/train.py train --config scripts/grpo/config.yaml

List all registered reward functions:
    python scripts/grpo/train.py list-rewards
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eshmun.grpo.train")


# ---------------------------------------------------------------------------
# Registry bootstrap — importing the package triggers all @register decorators
# ---------------------------------------------------------------------------

import eshmun.trainer.grpo.reward_functions as _rf  # noqa: E402  (side-effect import)


# ---------------------------------------------------------------------------
# Subcommand: list-rewards
# ---------------------------------------------------------------------------

def cmd_list_rewards(_args: argparse.Namespace) -> None:
    entries = _rf.list_all()
    if not entries:
        print("No reward functions registered.")
        return

    col = max(len(n) for n in entries) + 2
    print(f"\n{'Name':<{col}}  Description")
    print("-" * (col + 60))
    for name, doc in entries.items():
        first_line = doc.splitlines()[0] if doc else "(no docstring)"
        print(f"{name:<{col}}  {first_line}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_reward_fns(reward_cfg: list[dict[str, Any]]) -> list[Any]:
    fns = []
    for entry in reward_cfg:
        name = entry["name"]
        kwargs: dict[str, Any] = entry.get("kwargs", {}) or {}
        cls = _rf.get(name)
        try:
            fns.append(cls(**kwargs))
        except ImportError as exc:
            logger.error(
                "Reward function %r requires an optional dependency: %s", name, exc
            )
            sys.exit(1)
    return fns


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_yaml(args.config)

    # --- lazy heavy imports (torch, transformers, datasets) ---
    import torch
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

    from eshmun.trainer.grpo.config import GRPOConfig
    from eshmun.trainer.grpo.trainer import GRPOTrainer

    # Model & tokenizer
    model_cfg = cfg["model"]
    tok_cfg = cfg.get("tokenizer") or model_cfg

    logger.info("Loading tokenizer from %s", tok_cfg["name_or_path"])
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tok_cfg["name_or_path"])  # pyrefly: ignore [bad-assignment]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model from %s", model_cfg["name_or_path"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        dtype=torch.bfloat16 if cfg["training"].get("bf16") else torch.float32,
    )

    # Dataset
    ds_cfg = cfg["dataset"]
    logger.info("Loading dataset %s (split=%s)", ds_cfg["name_or_path"], ds_cfg.get("split", "train"))
    raw: Dataset = load_dataset(ds_cfg["name_or_path"], split=ds_cfg.get("split", "train"))  # pyrefly: ignore [bad-argument-type, bad-assignment]

    prompt_col: str = ds_cfg["prompt_column"]
    max_prompt_len: int = ds_cfg.get("max_prompt_length", 512)

    chat_cfg: dict[str, Any] = ds_cfg.get("chat_template", {}) or {}
    use_chat_template: bool = chat_cfg.get("enabled", False)
    system_prompt: str | None = chat_cfg.get("system_prompt") or None

    if use_chat_template and tokenizer.chat_template is None:
        logger.warning(
            "chat_template.enabled=true but tokenizer has no chat_template; "
            "falling back to plain tokenization."
        )
        use_chat_template = False

    def _apply_chat_template(text: str) -> list[int]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        tokenized = tokenizer.apply_chat_template(  # pyrefly: ignore [bad-assignment]
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return tokenized["input_ids"][-max_prompt_len:]  # pyrefly: ignore [bad-index]

    def _plain_tokenize(text: str) -> list[int]:
        raw = tokenizer(
            text,
            truncation=True,
            max_length=max_prompt_len,
            padding=False,
        )["input_ids"]
        
        return [int(x) for x in raw]

    _encode = _apply_chat_template if use_chat_template else _plain_tokenize

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        input_ids = [_encode(t) for t in batch[prompt_col]]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    metadata_cols: list[str] = ds_cfg.get("metadata_columns", []) or []
    drop_cols = [c for c in raw.column_names if c not in metadata_cols]

    dataset: Dataset = raw.map(tokenize, batched=True, batch_size=8, remove_columns=drop_cols)  # pyrefly: ignore [bad-assignment]
    # dataset.set_format("torch", columns=["input_ids", "attention_mask"])
    dataset.set_format("torch")

    # Reward functions
    reward_fns = _build_reward_fns(cfg.get("rewards", []))
    if not reward_fns:
        logger.error("No reward functions specified in config.")
        sys.exit(1)

    # Training config
    training = cfg["training"]
    grpo_cfg = GRPOConfig(
        output_dir=training["output_dir"],
        num_epochs=training.get("num_epochs", 1),
        per_device_train_batch_size=training.get("per_device_train_batch_size", 4),
        learning_rate=training.get("learning_rate", 1e-5),
        weight_decay=training.get("weight_decay", 0.01),
        warmup_steps=training.get("warmup_steps", 0),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 1),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        logging_steps=training.get("logging_steps", 10),
        save_steps=training.get("save_steps", 500),
        eval_steps=training.get("eval_steps", 500),
        bf16=training.get("bf16", False),
        fp16=training.get("fp16", False),
        seed=training.get("seed", 42),
        group_size=training.get("group_size", 8),
        epsilon=training.get("epsilon", 0.2),
        beta=training.get("beta", 0.01),
        num_ppo_epochs=training.get("num_ppo_epochs", 1),
        temperature=training.get("temperature", 1.0),
        max_new_tokens=training.get("max_new_tokens", 256),
        reward_weights=cfg.get("reward_weights"),
        ref_model_sync_epochs=training.get("ref_model_sync_epochs", 1),
    )

    trainer = GRPOTrainer(
        model=model,
        config=grpo_cfg,
        tokenizer=tokenizer,  # pyrefly: ignore [bad-argument-type]
        reward_fns=reward_fns,
        train_dataset=dataset,  # pyrefly: ignore [bad-argument-type]
    )

    logger.info("Starting GRPO training — output_dir=%s", grpo_cfg.output_dir)
    metrics = trainer.train()
    logger.info("Training complete: %s", metrics)

    trainer.save_model()
    logger.info("Model saved to %s", grpo_cfg.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Eshmun GRPO training toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/grpo/train.py list-rewards
              python scripts/grpo/train.py train --config scripts/grpo/config.yaml
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "list-rewards",
        help="Print all registered reward functions and their descriptions.",
    )

    train_p = sub.add_parser("train", help="Run GRPO training from a YAML config.")
    train_p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML training config file.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-rewards":
        cmd_list_rewards(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
