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
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, NoReturn, cast

import torch
import torch.nn.functional as F
import yaml

try:
    from eshmun.models.eshmun import EshmunConfig, EshmunForCausalLM
    from eshmun.tokenization import EshmunTokenizer
except ModuleNotFoundError:
    # Running from a checkout rather than an installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from eshmun.models.eshmun import EshmunConfig, EshmunForCausalLM
    from eshmun.tokenization import EshmunTokenizer

from datasets import Dataset, load_dataset, load_from_disk
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger("supervised_finetune")

# float32 is the default here on purpose: float16 has repeatedly produced
# unexpected errors on this project. Override via `model.torch_dtype` only if
# you have a specific reason.
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

FILE_BUILDERS = {
    ".jsonl": "json",
    ".json": "json",
    ".parquet": "parquet",
    ".csv": "csv",
}

# Markers left behind by `Dataset.save_to_disk`, which needs `load_from_disk`
# rather than `load_dataset`.
SAVED_DATASET_MARKERS = ("dataset_info.json", "state.json")
SAVED_DATASET_SUFFIXES = (".arrow",)

# Only the column TRL recognises as conversational. Anything else is renamed.
MESSAGES_COLUMN = "messages"

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

# Names that older transformers/TRL accepted, mapped to the name this
# environment wants. Only renames whose value carries over unchanged belong
# here, so that pointing at the new name is always safe.
RENAMED_KEYS = {
    "evaluation_strategy": "eval_strategy",
    # transformers 5.x dropped `warmup_ratio`; `warmup_steps` now takes a float
    # below 1 as a fraction of total steps, which is what a ratio was.
    "warmup_ratio": "warmup_steps",
}


def _fail(message: str) -> NoReturn:
    """Abort with a readable message instead of a traceback."""
    logger.error(message)
    raise SystemExit(1)


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


def validate_keys(section: str, given: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(given) - known)
    if not unknown:
        return

    hints = []
    for key in unknown:
        renamed = RENAMED_KEYS.get(key)
        close = [k for k in known if key.lower() in k.lower() or k.lower() in key.lower()]
        if renamed and renamed in known:
            hints.append(f"  - {key} (renamed: use `{renamed}`)")
        elif close:
            hints.append(f"  - {key} (did you mean: {', '.join(sorted(close))}?)")
        else:
            hints.append(f"  - {key}")
    _fail(f"unknown key(s) in recipe section `{section}`:\n" + "\n".join(hints))


def validate_section(section: str, given: dict[str, Any], target: type) -> None:
    """Reject keys that the destination dataclass would silently swallow.

    `TrainingArguments` accepts arbitrary extra kwargs, so a typo like
    `learning_rates` would otherwise be ignored without a word.
    """
    validate_keys(section, given, {f.name for f in dataclasses.fields(target)})


def build_tokenizer(model_cfg: dict[str, Any], data_cfg: dict[str, Any]):
    """Load the tokenizer and make sure it can render this dataset."""
    source = model_cfg.get("tokenizer_name_or_path") or model_cfg["name_or_path"]
    logger.info("loading tokenizer: %s", source)

    tokenizer = EshmunTokenizer.from_pretrained(
        source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    # A recipe may pin the template explicitly; otherwise trust the checkpoint.
    template_path = data_cfg.get("chat_template")
    if template_path:
        template_file = Path(template_path).expanduser()
        if not template_file.exists():
            _fail(f"chat template not found: {template_file}")
        tokenizer.chat_template = template_file.read_text()
        logger.info("chat template overridden from %s", template_file)

    return tokenizer


def tag_id(tokenizer, token: str) -> int:
    """The single token id `token` encodes to.

    `convert_tokens_to_ids` quietly hands back the unk id for a token the
    tokenizer does not know, which would put a loss span boundary on the wrong
    token without a word, so the encoding itself is checked instead.
    """
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        _fail(f"{token!r} is not a single token for this tokenizer (got {ids})")
    return ids[0]


def check_assistant_mask_support(tokenizer, assistant_only_loss: bool) -> None:
    """Fail early if `assistant_only_loss` cannot work with this template.

    TRL asks the chat template for `assistant_masks`, which transformers can
    only produce when the template wraps assistant turns in `{% generation %}`
    tags. Without them TRL raises part-way through dataset tokenization, so it
    is worth surfacing up front.
    """
    if not assistant_only_loss:
        return

    if "{% generation" not in (tokenizer.chat_template or ""):
        _fail(
            "`assistant_only_loss: true` requires the chat template to mark "
            "assistant turns with `{% generation %}...{% endgeneration %}`. "
            "This template has no such marker. Either add the markers to the "
            "template, or set `assistant_only_loss: false` to train on every token."
        )


def is_pretokenized(dataset: Dataset) -> bool:
    """True when the dataset arrives already tokenized, i.e. has `input_ids`.

    TRL uses the same test to decide whether to skip its chat-template path.
    """
    return "input_ids" in dataset.column_names


def validate_pretokenized(dataset: Dataset) -> Dataset:
    """Check a dataset that is already tokenized.

    TRL skips templating for these, and its collator takes the `labels` column
    as-is (padding with -100). Whatever mask the data carries is therefore the
    mask that trains -- there is nothing left for `assistant_only_loss` to do.
    """
    if "labels" not in dataset.column_names:
        logger.warning(
            "dataset has `input_ids` but no `labels`; every token will train. "
            "Add a `labels` column holding -100 where a position should not "
            "contribute, if that is not what you want."
        )
        return dataset

    sample = dataset[:5]
    for i, (ids, labels) in enumerate(zip(sample["input_ids"], sample["labels"])):
        if len(ids) != len(labels):
            _fail(
                f"`input_ids` and `labels` differ in length "
                f"(row {i}: {len(ids)} vs {len(labels)})"
            )

    logger.info("pretokenized dataset; the `labels` column supplies the loss mask")
    return dataset


def prepare_dataset(dataset: Dataset, data_cfg: dict[str, Any]) -> Dataset:
    """Validate whichever of the two supported shapes this dataset has."""
    if is_pretokenized(dataset):
        return validate_pretokenized(dataset)
    return prepare_messages_column(dataset, data_cfg)


def check_loss_masking(
    tokenizer, data_cfg: dict[str, Any], dataset: Dataset, assistant_only_loss: bool
) -> None:
    """Check that the loss mask for this dataset can actually be built.

    The two dataset shapes derive it differently, and each fails in its own
    way, so both are checked up front rather than part-way through training.
    """
    if is_pretokenized(dataset):
        if assistant_only_loss:
            _fail(
                "`assistant_only_loss: true` builds the mask by rendering the chat "
                "template, but this dataset is already tokenized (it has an "
                "`input_ids` column). Set `assistant_only_loss: false` and carry "
                "the mask in `labels` instead."
            )
        if data_cfg.get("chat_template"):
            logger.info(
                "`data.chat_template` is unused for a pretokenized dataset; the "
                "template only renders `messages`-format data."
            )
        return

    if not tokenizer.chat_template:
        _fail(
            "the dataset is in `messages` format, which needs a chat template, "
            "but the tokenizer has none. Set `data.chat_template` to a .jinja file."
        )

    check_assistant_mask_support(tokenizer, assistant_only_loss)


def build_model(model_cfg: dict[str, Any]):
    """Load Eshmun directly -- it is not registered with the Auto classes."""
    dtype_name = str(model_cfg.get("torch_dtype", "float32")).lower()
    if dtype_name not in DTYPES:
        _fail(f"unsupported `model.torch_dtype`: {dtype_name!r} (expected one of {sorted(DTYPES)})")

    source = model_cfg["name_or_path"]
    logger.info("loading model: %s (%s)", source, dtype_name)

    kwargs: dict[str, Any] = {
        "dtype": DTYPES[dtype_name],
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    # Each attention module copies `config.attention_dropout` into itself at
    # `__init__`, so the override has to happen on the config *before* the model
    # is built -- setting it on the loaded model afterwards would leave every
    # live module at the value it was born with.
    dropout = model_cfg.get("attention_dropout")
    if dropout is not None:
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            _fail(f"`model.attention_dropout` must be in [0, 1), got {dropout}")
        config = EshmunConfig.from_pretrained(source)
        config.attention_dropout = dropout
        kwargs["config"] = config
        logger.info("attention dropout overridden to %s", dropout)

    return EshmunForCausalLM.from_pretrained(source, **kwargs)


def maybe_resize_embeddings(model, tokenizer) -> None:
    """Grow the embedding table if the tokenizer knows more tokens than the model."""
    model_vocab = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab = len(tokenizer)
    if tokenizer_vocab > model_vocab:
        logger.warning(
            "tokenizer has %d tokens but the model embedding has %d; resizing",
            tokenizer_vocab,
            model_vocab,
        )
        model.resize_token_embeddings(tokenizer_vocab)
    elif tokenizer_vocab < model_vocab:
        logger.info(
            "model embedding (%d) is larger than the tokenizer (%d); leaving as-is",
            model_vocab,
            tokenizer_vocab,
        )


def looks_like_saved_dataset(path: Path) -> bool:
    """True for a directory produced by `Dataset.save_to_disk`."""
    if any((path / marker).exists() for marker in SAVED_DATASET_MARKERS):
        return True
    return any(p.suffix in SAVED_DATASET_SUFFIXES for p in path.iterdir() if p.is_file())


def from_saved_dataset(path: Path, split: str) -> Dataset:
    """Load a `save_to_disk` directory, which holds either one split or several."""
    loaded = load_from_disk(str(path))

    if isinstance(loaded, Dataset):
        # The directory itself is the split, so `data.split` does not apply.
        logger.info("loaded a single saved split (%d examples)", len(loaded))
        return loaded

    if split not in loaded:
        _fail(
            f"saved dataset at {path} has splits {sorted(loaded)}, "
            f"which does not include `{split}`"
        )
    return loaded[split]


def load_split(spec: str, config: str | None, split: str) -> Dataset:
    """Load one split from a saved dataset dir, a data file, or the Hub."""
    path = Path(spec).expanduser()

    if not path.exists():
        logger.info("loading dataset %s (split=%s) from the Hub", spec, split)
        # `split` is always given, so this is a single Dataset, not a DatasetDict.
        return cast(Dataset, load_dataset(spec, name=config, split=split))

    if path.is_dir():
        if looks_like_saved_dataset(path):
            logger.info("loading saved dataset: %s", path)
            return from_saved_dataset(path, split)

        files = [p for p in sorted(path.rglob("*")) if p.suffix in FILE_BUILDERS]
        if not files:
            _fail(
                f"{path} is a directory but looks like neither a `save_to_disk` "
                f"dataset nor a directory of {', '.join(FILE_BUILDERS)} files"
            )
        builder = FILE_BUILDERS[files[0].suffix]
        logger.info("loading %d file(s) from %s as %s", len(files), path, builder)
        data_files = [str(p) for p in files]
    else:
        builder = FILE_BUILDERS.get(path.suffix)
        if builder is None:
            _fail(f"unsupported dataset file {path} (expected one of {sorted(FILE_BUILDERS)})")
        logger.info("loading %s as %s", path, builder)
        data_files = str(path)

    return cast(Dataset, load_dataset(builder, data_files=data_files, split=split))


def prepare_messages_column(dataset: Dataset, data_cfg: dict[str, Any]) -> Dataset:
    """Rename the messages column if the recipe names it something else."""
    column = data_cfg.get("messages_column", MESSAGES_COLUMN)

    if column != MESSAGES_COLUMN:
        if column not in dataset.column_names:
            _fail(
                f"column {column!r} not found in dataset "
                f"(available: {', '.join(dataset.column_names)})"
            )
        dataset = dataset.rename_column(column, MESSAGES_COLUMN)

    if MESSAGES_COLUMN not in dataset.column_names:
        _fail(
            f"dataset has no `{MESSAGES_COLUMN}` column "
            f"(available: {', '.join(dataset.column_names)})"
        )

    sample = dataset[0][MESSAGES_COLUMN]
    if not isinstance(sample, list) or not sample:
        _fail(f"`{MESSAGES_COLUMN}` must be a non-empty list of role/content turns")
    if not {"role", "content"} <= set(sample[0]):
        _fail(
            f"each `{MESSAGES_COLUMN}` turn needs `role` and `content` keys; "
            f"got {sorted(sample[0])}"
        )

    return dataset


def build_datasets(data_cfg: dict[str, Any], default_seed: int = 42) -> tuple[Dataset, Dataset | None]:
    spec = data_cfg["name_or_path"]
    config = data_cfg.get("config")

    train = prepare_dataset(load_split(spec, config, data_cfg.get("split", "train")), data_cfg)

    eval_split = data_cfg.get("eval_split")
    eval_path = data_cfg.get("eval_name_or_path")
    eval_fraction = data_cfg.get("eval_fraction")

    if sum(bool(x) for x in (eval_split, eval_path, eval_fraction)) > 1:
        _fail(
            "set at most one of `data.eval_split`, `data.eval_name_or_path`, "
            "and `data.eval_fraction`"
        )

    if eval_split:
        eval_ds = prepare_dataset(load_split(spec, config, eval_split), data_cfg)
    elif eval_path:
        # A separate location, e.g. a sibling `save_to_disk` directory.
        logger.info("eval data comes from %s", eval_path)
        eval_ds = prepare_dataset(load_split(eval_path, config, "train"), data_cfg)
    elif eval_fraction:
        if not 0.0 < float(eval_fraction) < 1.0:
            _fail("`data.eval_fraction` must be strictly between 0 and 1")
        parts = train.train_test_split(test_size=float(eval_fraction), seed=default_seed)
        train, eval_ds = parts["train"], parts["test"]
    else:
        eval_ds = None

    logger.info("train examples: %d%s", len(train), f", eval examples: {len(eval_ds)}" if eval_ds else "")
    return train, eval_ds


def build_sft_config(training_cfg: dict[str, Any], has_eval: bool) -> SFTConfig:
    cfg = dict(training_cfg)

    # Only evaluate on an interval if there is something to evaluate on.
    if "eval_strategy" not in cfg:
        cfg["eval_strategy"] = "steps" if has_eval else "no"
        if has_eval and "eval_steps" not in cfg:
            cfg["eval_steps"] = cfg.get("save_steps", 250)

    validate_section("training", cfg, SFTConfig)
    return SFTConfig(**cfg)


# --- the span-weighted cross-entropy objective -------------------------------
#
# The objective averages cross-entropy inside two spans and mixes the two
# means, instead of taking a single mean over every supervised token:
#
#     L = 0.25 * mean(CE_think) + 0.75 * mean(CE_seq)
#
#   * thinking span -- the first supervised token through `</think>`
#   * sequence span -- everything after it: `<protein>`, the residues,
#     `</protein>`, and the closing `</s>`, so the end-of-turn token keeps
#     being trained
#
# The weights sit where the corpus' own loss mass already is: on the small
# split, 25.7% of labeled tokens fall in the thinking span and 74.3% in the
# sequence span (measured over 500 rows, closers included, which is exactly how
# this partition cuts them). So the mixture stays put and the split just
# becomes explicit -- and turnable -- instead of implicit in the token counts.
#
# Inside a span the tokens are not weighted equally: the token that *closes*
# the span -- `</think>` in the thinking span, `</protein>` in the sequence
# span -- carries STRUCTURAL_TOKEN_WEIGHT instead of 1. An early stop is one
# wrong token in a row of ~150, worth 0.7% of the sequence span under a plain
# mean and ~3% at 5. Each span is a weighted mean of its own tokens, so an
# upweighted token moves mass inside its span rather than rescaling the whole
# loss.
THINK_LOSS_WEIGHT = 0.25
SEQUENCE_LOSS_WEIGHT = 0.75
STRUCTURAL_TOKEN_WEIGHT = 5.0


def span_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    think_close_id: int,
    protein_close_id: int,
    think_weight: float = THINK_LOSS_WEIGHT,
    sequence_weight: float = SEQUENCE_LOSS_WEIGHT,
    structural_weight: float = STRUCTURAL_TOKEN_WEIGHT,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The objective above, plus each span's plain mean CE for logging.

    `logits` are the model's full, unsliced outputs. That is what the hook
    receives: with `compute_loss_func` set the trainer forwards without
    `labels`, so nothing slices them down to the supervised positions, and the
    two tensors line up as `logits[:, :-1]` / `labels[:, 1:]`.

    The returned loss is a mean over the batch; a caller under gradient
    accumulation scales it by its share of `num_items_in_batch` (see
    `SpanWeightedSFTTrainer`). The spans are pooled over the batch, not per
    row, and a batch with an empty span hands the other span the full weight,
    so neither case can divide by zero.
    """
    # `logits[:, t]` predicts `labels[:, t + 1]`. Spans are decided on the
    # predicted token, so a closer belongs to the span it closes.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    supervised = shift_labels != ignore_index

    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
        shift_labels.reshape(-1).long(),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape(shift_labels.shape)

    # The *first* `</think>` is the boundary; a repeat later in the row belongs
    # to the sequence span it sits in, not back to the thinking span. With no
    # `</think>` at all the whole row falls in the thinking span, which the
    # weights then renormalize into its plain mean.
    at_think_close = shift_labels == think_close_id
    at_protein_close = shift_labels == protein_close_id
    think_seen = at_think_close.cumsum(dim=1)
    protein_seen = at_protein_close.cumsum(dim=1)
    first_think_close = at_think_close & (think_seen == 1)
    first_protein_close = at_protein_close & (protein_seen == 1)
    in_think = (think_seen == 0) | first_think_close
    in_sequence = ~in_think

    weight = torch.ones_like(per_token)
    weight[first_think_close | first_protein_close] = structural_weight
    weight = weight * supervised

    think_mass = (weight * in_think).sum()
    sequence_mass = (weight * in_sequence).sum()
    think_terms = per_token * weight * in_think
    sequence_terms = per_token * weight * in_sequence
    think_mean = think_terms.sum() / think_mass.clamp(min=1)
    sequence_mean = sequence_terms.sum() / sequence_mass.clamp(min=1)

    think_on = think_weight if bool(think_mass > 0) else 0.0
    sequence_on = sequence_weight if bool(sequence_mass > 0) else 0.0
    total = think_on + sequence_on
    if total == 0.0:
        # Nothing supervised in this batch at all: a zero that still carries
        # the graph, so the step is a no-op rather than a crash.
        loss = per_token.sum() * 0.0
    else:
        loss = (think_on * think_mean + sequence_on * sequence_mean) / total

    with torch.no_grad():
        think_plain = (per_token * supervised * in_think).sum()
        sequence_plain = (per_token * supervised * in_sequence).sum()
        think_count = (supervised & in_think).sum().clamp(min=1)
        sequence_count = (supervised & in_sequence).sum().clamp(min=1)
        diagnostics = {
            "ce_think": (think_plain / think_count).item(),
            "ce_seq": (sequence_plain / sequence_count).item(),
        }
    return loss, diagnostics


class SpanWeightedSFTTrainer(SFTTrainer):
    """`SFTTrainer` whose objective is `span_weighted_cross_entropy`.

    The objective is installed through `compute_loss_func` rather than by
    overriding `compute_loss`. With that hook set the trainer sets the labels
    aside before the forward, so the plain cross-entropy is replaced rather
    than added to -- it is never computed -- while TRL's entropy and
    token-accuracy metrics still come out of the same forward.

    The hook also changes the gradient-accumulation convention: with it set,
    `Trainer.training_step` stops dividing a micro-batch's loss by the
    accumulation steps and expects the function to use `num_items_in_batch`
    itself. The loss returned here is scaled by the micro-batch's share of the
    accumulation window's supervised tokens -- the scaling the model's own
    summed loss carries -- which also keeps the logged `loss` on the scale of
    a plain mean.
    """

    def __init__(
        self,
        *args: Any,
        think_close_id: int,
        protein_close_id: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.think_close_id = think_close_id
        self.protein_close_id = protein_close_id
        # A bound method, so `Trainer.compute_loss` can call it exactly as
        # `compute_loss_func(outputs, labels, num_items_in_batch=...)`.
        self.compute_loss_func = self._span_weighted_loss

    def _span_weighted_loss(
        self,
        outputs: Any,
        labels: torch.Tensor,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        loss, diagnostics = span_weighted_cross_entropy(
            outputs.logits, labels, self.think_close_id, self.protein_close_id
        )
        mode = "train" if self.model.training else "eval"
        for name, value in diagnostics.items():
            self._metrics[mode][name].append(value)

        if num_items_in_batch is not None:
            # Counted the way `Trainer._get_num_items_in_batch` counts it, so
            # the two shares agree.
            counted = labels[..., 1:] if self._loss_shifts_labels else labels
            loss = loss * (counted != -100).sum() / num_items_in_batch
        return loss


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
