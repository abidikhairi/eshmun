"""Building the tokenizer and the model from a recipe's `model` section."""

from __future__ import annotations

import logging
from typing import Any

import torch

from eshmun.errors import fail
from eshmun.models.eshmun import EshmunConfig, EshmunForCausalLM
from eshmun.tokenization import load_tokenizer

logger = logging.getLogger(__name__)

# float32 is the default here on purpose: float16 has repeatedly produced
# unexpected errors on this project. Override via `model.torch_dtype` only if
# you have a specific reason.
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_tokenizer(model_cfg: dict[str, Any], data_cfg: dict[str, Any]):
    """Load the tokenizer and make sure it can render this dataset."""
    source = model_cfg.get("tokenizer_name_or_path") or model_cfg["name_or_path"]
    return load_tokenizer(
        source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        template_path=data_cfg.get("chat_template"),
        # The run re-saves this tokenizer at the end, so nothing may be added
        # to it that the checkpoint lineage does not already carry.
        install_decoder=False,
    )


def build_model(model_cfg: dict[str, Any]):
    """Load Eshmun directly -- it is not registered with the Auto classes."""
    dtype_name = str(model_cfg.get("torch_dtype", "float32")).lower()
    if dtype_name not in DTYPES:
        fail(f"unsupported `model.torch_dtype`: {dtype_name!r} (expected one of {sorted(DTYPES)})")

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
            fail(f"`model.attention_dropout` must be in [0, 1), got {dropout}")
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
