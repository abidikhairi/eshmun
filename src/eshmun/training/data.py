"""Loading a recipe's dataset and checking its loss mask can be built.

The training script accepts two dataset shapes -- conversational `messages`,
rendered with the tokenizer's chat template, and pretokenized `input_ids` with
`labels` carrying the mask -- and each fails in its own way. Every check here
runs before training starts, so a dataset that cannot train fails before the
model is loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset, load_from_disk

from eshmun.errors import fail

logger = logging.getLogger(__name__)

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
        fail(
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
            fail(
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
            fail(
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
        fail(
            "the dataset is in `messages` format, which needs a chat template, "
            "but the tokenizer has none. Set `data.chat_template` to a .jinja file."
        )

    check_assistant_mask_support(tokenizer, assistant_only_loss)


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
        fail(
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
            fail(
                f"{path} is a directory but looks like neither a `save_to_disk` "
                f"dataset nor a directory of {', '.join(FILE_BUILDERS)} files"
            )
        builder = FILE_BUILDERS[files[0].suffix]
        logger.info("loading %d file(s) from %s as %s", len(files), path, builder)
        data_files = [str(p) for p in files]
    else:
        builder = FILE_BUILDERS.get(path.suffix)
        if builder is None:
            fail(f"unsupported dataset file {path} (expected one of {sorted(FILE_BUILDERS)})")
        logger.info("loading %s as %s", path, builder)
        data_files = str(path)

    return cast(Dataset, load_dataset(builder, data_files=data_files, split=split))


def prepare_messages_column(dataset: Dataset, data_cfg: dict[str, Any]) -> Dataset:
    """Rename the messages column if the recipe names it something else."""
    column = data_cfg.get("messages_column", MESSAGES_COLUMN)

    if column != MESSAGES_COLUMN:
        if column not in dataset.column_names:
            fail(
                f"column {column!r} not found in dataset "
                f"(available: {', '.join(dataset.column_names)})"
            )
        dataset = dataset.rename_column(column, MESSAGES_COLUMN)

    if MESSAGES_COLUMN not in dataset.column_names:
        fail(
            f"dataset has no `{MESSAGES_COLUMN}` column "
            f"(available: {', '.join(dataset.column_names)})"
        )

    sample = dataset[0][MESSAGES_COLUMN]
    if not isinstance(sample, list) or not sample:
        fail(f"`{MESSAGES_COLUMN}` must be a non-empty list of role/content turns")
    if not {"role", "content"} <= set(sample[0]):
        fail(
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
        fail(
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
            fail("`data.eval_fraction` must be strictly between 0 and 1")
        parts = train.train_test_split(test_size=float(eval_fraction), seed=default_seed)
        train, eval_ds = parts["train"], parts["test"]
    else:
        eval_ds = None

    logger.info("train examples: %d%s", len(train), f", eval examples: {len(eval_ds)}" if eval_ds else "")
    return train, eval_ds
