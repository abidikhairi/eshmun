"""Loading an Eshmun tokenizer, with the repairs and checks callers need.

The checkpoints this project loads disagree about what their `tokenizer.json`
carries. Fine-tuning checkpoints save a file with no `decoder`; checkpoints
written under the tokenizer bug (commit e2962a5) carry no merges at all. The
repairs live here so generation, the CLI and training all get the same
treatment, and the keyword flags keep the callers' differences explicit.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tokenizers import decoders

from eshmun.errors import fail
from eshmun.tokenization.tokenizer import EshmunTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(
    source: str,
    *,
    trust_remote_code: bool = False,
    template_path: str | Path | None = None,
    install_decoder: bool = True,
    require_chat_template: bool = False,
    require_pad_token: bool = False,
    probe_merges: bool = False,
) -> EshmunTokenizer:
    """Load the tokenizer at `source`, with the optional repairs and checks.

    `install_decoder` restores the ByteLevel decoder a fine-tuning checkpoint
    omits -- without it `decode` cannot invert the byte-level encoding. Training
    turns it off: the tokenizer it loads is the one the run re-saves, and the
    checkpoints' decoder-less shape is part of the lineage it must not change.

    `probe_merges` warns when the BPE encodes one token per character, which
    means the merges are gone. The object exposes no `merges` to read, so the
    probe is behavioural.

    `template_path` overrides the chat template from a file. The two
    `require_*` flags turn the hard requirements -- a chat template, a pad
    token -- into failures here, where the reason is still visible.
    """
    logger.info("loading tokenizer: %s", source)
    tokenizer = EshmunTokenizer.from_pretrained(
        source, trust_remote_code=trust_remote_code
    )

    # The fine-tuning checkpoints save a `tokenizer.json` with no `decoder`, so
    # `decode` cannot invert the byte-level encoding: it falls back to joining
    # tokens with spaces and leaves `Ġ` (space) and `Ċ` (newline) literal in the
    # text. The Hub model of the same lineage ships the decoder; restore it
    # rather than shipping a response the model never wrote.
    if install_decoder and tokenizer.backend_tokenizer.decoder is None:
        tokenizer.backend_tokenizer.decoder = decoders.ByteLevel()
        logger.warning(
            "tokenizer at %s carries no decoder; installed ByteLevel", source
        )

    if probe_merges:
        # A merge-less BPE encodes one token per character, and the object
        # exposes no `merges` to read, so the probe is behavioural: a healthy
        # tokenizer turns these 16 characters into two or three tokens, a
        # damaged one into 16.
        probe = "protein sequence"
        if len(tokenizer(probe, add_special_tokens=False)["input_ids"]) >= len(probe):
            logger.warning(
                "tokenizer at %s encodes one token per character, so prompts are "
                "not encoded the way the model was trained; pass "
                "`--tokenizer khairi/kothar-it-409m`",
                source,
            )

    # A recipe may pin the template explicitly; otherwise trust the checkpoint.
    if template_path:
        template_file = Path(template_path).expanduser()
        if not template_file.exists():
            fail(f"chat template not found: {template_file}")
        tokenizer.chat_template = template_file.read_text()
        logger.info("chat template overridden from %s", template_file)

    if require_chat_template and not tokenizer.chat_template:
        fail(
            f"tokenizer at {source!r} has no chat template, which is how the "
            "prompt is built."
        )

    if require_pad_token and tokenizer.pad_token_id is None:
        fail("tokenizer has no pad token, which batched generation needs")

    return tokenizer


def tag_id(tokenizer: EshmunTokenizer, token: str) -> int:
    """The single token id `token` encodes to.

    `convert_tokens_to_ids` quietly hands back the unk id for a token the
    tokenizer does not know, which would put a loss span boundary on the wrong
    token without a word, so the encoding itself is checked instead.
    """
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        fail(f"{token!r} is not a single token for this tokenizer (got {ids})")
    return ids[0]
