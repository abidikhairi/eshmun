#!/usr/bin/env python
"""Recipe-driven supervised fine-tuning for Eshmun causal language models.

Accepts either of two dataset shapes:

  conversational -- a `messages` column, rendered with the tokenizer's chat
  template:

      {"messages": [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
      ]}

  pretokenized -- `input_ids`, plus `labels` when the loss mask is already
  built (`-100` on positions that should not contribute). The mask is then
  whatever the data says, so `assistant_only_loss` does not apply.

Hyperparameters live in a YAML recipe rather than on the command line, so runs
stay reproducible and diffable. See `recipes/example.yaml.example` for the full
schema; recipe files themselves are gitignored.

    python scripts/training/supervised_finetune.py --recipe recipes/sft/foo.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from eshmun.errors import fail as _fail
    from eshmun.tokenization import tag_id
    from eshmun.training.data import build_datasets, check_loss_masking
    from eshmun.training.model import (
        build_model,
        build_tokenizer,
        maybe_resize_embeddings,
    )
    from eshmun.training.trainer import SpanWeightedSFTTrainer
    from eshmun.training.validation import validate_keys, validate_section
except ModuleNotFoundError:
    # Running from a checkout rather than an installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from eshmun.errors import fail as _fail
    from eshmun.tokenization import tag_id
    from eshmun.training.data import build_datasets, check_loss_masking
    from eshmun.training.model import (
        build_model,
        build_tokenizer,
        maybe_resize_embeddings,
    )
    from eshmun.training.trainer import SpanWeightedSFTTrainer
    from eshmun.training.validation import validate_keys, validate_section

from trl import SFTConfig

logger = logging.getLogger("supervised_finetune")

# The recipe schema. Anything else is a typo and is reported as one.
MODEL_KEYS = {
    "name_or_path",
    "tokenizer_name_or_path",
    "torch_dtype",
    "attn_implementation",
    "attention_dropout",
    "trust_remote_code",
}

DATA_KEYS = {
    "name_or_path",
    "config",
    "split",
    "messages_column",
    "eval_split",
    "eval_name_or_path",
    "eval_fraction",
    "chat_template",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Path to the YAML recipe describing this run.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve the recipe, print it, and exit without loading anything.",
    )
    parser.add_argument("--output-dir", help="Override `training.output_dir`.")
    parser.add_argument("--model", help="Override `model.name_or_path`.")
    parser.add_argument("--dataset", help="Override `data.name_or_path`.")
    parser.add_argument("--seed", type=int, help="Override `training.seed`.")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override `training.max_steps` (useful for smoke tests).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume from a checkpoint directory, or 'true' to auto-resolve.",
    )
    return parser.parse_args()


def load_recipe(path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    """Read a recipe, apply CLI overrides, and fill in section defaults."""
    if not path.exists():
        _fail(f"recipe not found: {path}")

    with open(path) as f:
        recipe = yaml.safe_load(f) or {}

    if not isinstance(recipe, dict):
        _fail(f"recipe must be a YAML mapping, got {type(recipe).__name__}")

    recipe.setdefault("model", {})
    recipe.setdefault("data", {})
    recipe.setdefault("training", {})

    unknown = sorted(set(recipe) - {"model", "data", "training"})
    if unknown:
        _fail(
            "unknown recipe section(s): "
            + ", ".join(f"`{s}`" for s in unknown)
            + " (expected `model`, `data`, `training`)"
        )

    for section in ("model", "data", "training"):
        if not isinstance(recipe[section], dict):
            _fail(f"recipe section `{section}` must be a mapping")

    validate_keys("model", recipe["model"], MODEL_KEYS)
    validate_keys("data", recipe["data"], DATA_KEYS)

    # CLI overrides win over the recipe.
    override_paths = {
        "output_dir": ("training", "output_dir"),
        "model": ("model", "name_or_path"),
        "dataset": ("data", "name_or_path"),
        "seed": ("training", "seed"),
        "max_steps": ("training", "max_steps"),
    }
    for cli_key, (section, key) in override_paths.items():
        value = overrides.get(cli_key)
        if value is not None:
            recipe[section][key] = value

    if not recipe["model"].get("name_or_path"):
        _fail("`model.name_or_path` is required")
    if not recipe["data"].get("name_or_path"):
        _fail("`data.name_or_path` is required")
    if not recipe["training"].get("output_dir"):
        _fail("`training.output_dir` is required (or pass --output-dir)")

    return recipe


def build_sft_config(training_cfg: dict[str, Any], has_eval: bool) -> SFTConfig:
    cfg = dict(training_cfg)

    # Only evaluate on an interval if there is something to evaluate on.
    if "eval_strategy" not in cfg:
        cfg["eval_strategy"] = "steps" if has_eval else "no"
        if has_eval and "eval_steps" not in cfg:
            cfg["eval_steps"] = cfg.get("save_steps", 250)

    validate_section("training", cfg, SFTConfig)
    return SFTConfig(**cfg)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    recipe = load_recipe(args.recipe, vars(args))

    if args.print_config:
        print(yaml.safe_dump(recipe, sort_keys=False, default_flow_style=False))
        return

    model_cfg, data_cfg, training_cfg = recipe["model"], recipe["data"], recipe["training"]

    assistant_only_loss = bool(training_cfg.get("assistant_only_loss", False))

    # Everything that fails cheaply runs before the model is loaded.
    tokenizer = build_tokenizer(model_cfg, data_cfg)
    # The loss' span boundaries, resolved before anything expensive is loaded.
    think_close_id = tag_id(tokenizer, "</think>")
    protein_close_id = tag_id(tokenizer, "</protein>")
    logger.info(
        "loss spans: </think>=%d, </protein>=%d", think_close_id, protein_close_id
    )
    train_ds, eval_ds = build_datasets(data_cfg, default_seed=training_cfg.get("seed", 42))
    check_loss_masking(tokenizer, data_cfg, train_ds, assistant_only_loss)
    sft_config = build_sft_config(training_cfg, has_eval=eval_ds is not None)

    model = build_model(model_cfg)
    maybe_resize_embeddings(model, tokenizer)

    trainer = SpanWeightedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        think_close_id=think_close_id,
        protein_close_id=protein_close_id,
    )

    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True
    elif resume and resume.lower() in ("false", "none"):
        resume = None

    logger.info("training -> %s", sft_config.output_dir)
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)
    logger.info("saved model and tokenizer to %s", sft_config.output_dir)


if __name__ == "__main__":
    main()
