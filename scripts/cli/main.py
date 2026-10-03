#!/usr/bin/env python
"""Interactive generation: type a family name, read back a sequence.

    python scripts/cli/main.py runs/sft/checkpoint-1500

The checkpoint is loaded once -- tokenizer, then model at float32 -- and the
session then reads family names from stdin, one per line. Each name is turned
into an instruction using one of the ten phrasings the training rows carry
(see `INSTRUCTIONS`), rendered through the chat template exactly as
`scripts/eval/generate.py` renders it, and continued with the same sampling
settings that script records -- so a response here is drawn the way a
generation there is. Each response is printed as the prompt that drew it,
the text between `<think>` and `</think>`, and the residues from `<protein>`
in the plain 20-letter alphabet, in that order -- cyan, yellow and green
when `rich` is installed, plain when it is not.

Nothing is seeded: each family name draws a fresh phrasing and a fresh
sample, so the same name typed twice gives two different proteins. Ctrl-D
(or Ctrl-C) ends the session.

`--tokenizer` defaults to the checkpoint. Point it at `khairi/kothar-it-409m`
for a checkpoint whose own `tokenizer.json` carries no merges (the ones saved
under the bug in commit e2962a5): such a checkpoint loads without complaint and
then encodes every prompt one character at a time.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, NoReturn, cast

import torch
from tokenizers import decoders

try:
    from rich.console import Console
    from rich.text import Text
except ModuleNotFoundError:
    # Colour is a nicety, not a dependency: without `rich` the session still
    # runs and prints the same three fields plain.
    Console = None  # pyrefly: ignore [bad-assignment]
    Text = None  # pyrefly: ignore [bad-assignment]

try:
    from eshmun.models.eshmun import EshmunForCausalLM
    from eshmun.tokenization import EshmunTokenizer
except ModuleNotFoundError:
    # Running from a checkout rather than an installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from eshmun.models.eshmun import EshmunForCausalLM
    from eshmun.tokenization import EshmunTokenizer

logger = logging.getLogger("cli")

THINK_START = "<think>"
THINK_END = "</think>"
PROTEIN_START = "<protein>"
PROTEIN_END = "</protein>"

# The one-letter amino-acid alphabet. Residue tokens carry a `Ƥ` prefix that is
# not a residue, so filtering to this set removes it too.
RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")

# The marker on every residue token. It appears on residues and nowhere else,
# which is what makes it usable as a signal when the tags are missing.
MARKER = "Ƥ"

# What the decoder leaves on the end of a finished sequence.
TERMINATORS = ("</s>", "<pad>")

# The ten instructions the training rows carry, with the family name where it
# goes. Each is 9-11% of the corpus, so one drawn uniformly reproduces the
# mix the model was trained on.
INSTRUCTIONS = (
    "Produce a protein sequence from {family}.",
    "What is a typical protein sequence for the family {family}?",
    "Generate a representative protein sequence of the {family} family.",
    "Give me a protein sequence belonging to the {family} family.",
    "Give an example sequence for the family {family}.",
    "Provide a protein sequence from the family: {family}.",
    "Show me a protein sequence from the {family} family.",
    "Can you generate a protein sequence for the {family} family?",
    "Write a protein sequence that belongs to {family}.",
    "I need an example protein sequence for the family {family}.",
)

# The sampling settings `scripts/eval/generate.py` records with.
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 250
# Without a penalty this model tends to emit long runs of a single residue.
REPETITION_PENALTY = 1.3


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
        "model",
        help="Checkpoint directory or Hub repo id to generate from.",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer source; defaults to the checkpoint. Use "
        "`khairi/kothar-it-409m` for a checkpoint whose own tokenizer.json "
        "has no merges.",
    )
    return parser.parse_args()


def _span(text: str, start: str, end: str) -> str | None:
    """Text between `start` and `end`, or None when `start` never appears.

    An `end` that never arrives still yields the remainder: generation hits
    the token cap often enough that discarding those spans would throw away
    sequences that are otherwise perfectly usable.
    """
    at = text.find(start)
    if at == -1:
        return None
    at += len(start)
    stop = text.find(end, at)
    return text[at:] if stop == -1 else text[at:stop]


def strip_terminators(text: str) -> str:
    """Drop trailing EOS/pad markers, leaving the rest of the body alone."""
    while True:
        for token in TERMINATORS:
            if text.endswith(token):
                text = text[: -len(token)]
                break
        else:
            return text


def _residue_runs(text: str) -> list[str]:
    """Contiguous runs of `Ƥ`-prefixed residues, reduced to their letters.

    A run ends at the first character that is not a marked residue, so prose
    between two stretches of sequence separates them rather than joining them.
    """
    runs: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(text):
        if (
            text[index] == MARKER
            and index + 1 < len(text)
            and text[index + 1] in RESIDUES
        ):
            current.append(text[index + 1])
            index += 2
            continue
        if current:
            runs.append("".join(current))
            current = []
        index += 1
    if current:
        runs.append("".join(current))
    return runs


def extract_reasoning(text: str) -> str | None:
    """The text between `<think>` and `</think>`, or None when it never opens."""
    span = _span(text, THINK_START, THINK_END)
    return span.strip() if span is not None else None


def extract_sequence(text: str) -> str | None:
    """Residues from the `<protein>` span, or from the bare marked runs.

    With no `<protein>` tag anywhere, the longest marked run stands in: an
    early checkpoint emits its residues bare, and the marker still identifies
    them exactly where filtering the whole response would sweep up prose.
    """
    span = _span(text, PROTEIN_START, PROTEIN_END)
    if span is not None:
        return "".join(character for character in span if character in RESIDUES)
    runs = _residue_runs(text)
    return max(runs, key=len) if runs else None


def load_tokenizer(source: str) -> EshmunTokenizer:
    logger.info("loading tokenizer: %s", source)
    tokenizer = EshmunTokenizer.from_pretrained(source)

    # A checkpoint saved under the tokenizer bug (commit e2962a5) carries a
    # `tokenizer.json` with no merges, which no decoder can repair -- this
    # guard only covers the missing decoder, so such a checkpoint needs
    # `--tokenizer`.
    if tokenizer.backend_tokenizer.decoder is None:
        tokenizer.backend_tokenizer.decoder = decoders.ByteLevel()
        logger.warning(
            "tokenizer at %s carries no decoder; installed ByteLevel", source
        )

    # A merge-less BPE encodes one token per character, and the object exposes
    # no `merges` to read, so the probe is behavioural: a healthy tokenizer
    # turns these 16 characters into two or three tokens, a damaged one into 16
    # (the encoding above found the same 16 ids for both damaged checkpoints).
    probe = "protein sequence"
    if len(tokenizer(probe, add_special_tokens=False)["input_ids"]) >= len(probe):
        logger.warning(
            "tokenizer at %s encodes one token per character, so prompts are "
            "not encoded the way the model was trained; pass "
            "`--tokenizer khairi/kothar-it-409m`",
            source,
        )

    if not tokenizer.chat_template:
        _fail(
            f"tokenizer at {source!r} has no chat template, which is how the "
            "prompt is built."
        )
    return tokenizer


def load_model(source: str, device: torch.device) -> EshmunForCausalLM:
    # float32 on purpose: float16 has repeatedly produced unexpected errors on
    # this project.
    logger.info("loading model: %s (float32)", source)
    model = EshmunForCausalLM.from_pretrained(source, dtype=torch.float32)
    model.to(device)
    model.eval()
    return model


def render_prompt(tokenizer: EshmunTokenizer, prompt: str) -> str:
    """Render one prompt exactly as the training rows render it."""
    return cast(
        str,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
    )


def generate_one(
    model: EshmunForCausalLM,
    tokenizer: EshmunTokenizer,
    prompt: str,
    device: torch.device,
) -> str | None:
    """One response for one prompt, terminators stripped.

    Returns None, with the reason logged, for a prompt that cannot fit.
    """
    rendered = render_prompt(tokenizer, prompt)
    # `add_special_tokens=False` is load-bearing: this tokenizer's BOS is
    # `</s>`, so the default would put one in front of the prompt. The training
    # rows carry no BOS, and a prompt the model never saw the shape of is not
    # the prompt being evaluated.
    encoded = tokenizer(
        rendered, return_tensors="pt", add_special_tokens=False
    ).to(device)

    # The positional table holds `max_position_embeddings + 2` entries and
    # position p lands at index p + 2, so the cap is a hard ceiling on prompt
    # plus continuation.
    ceiling = model.config.max_position_embeddings
    length = int(encoded["input_ids"].shape[1])
    if length + MAX_NEW_TOKENS > ceiling:
        logger.error(
            "prompt is %d tokens; with %d new tokens it overruns the "
            "%d-token positional limit. Shorten it.",
            length,
            MAX_NEW_TOKENS,
            ceiling,
        )
        return None

    with torch.no_grad():
        # `generate` annotates itself with a `GenerativePreTrainedModel` self
        # protocol which types `device` as a plain attribute, while transformers
        # 5.x exposes it as a property -- so no custom PreTrainedModel subclass
        # can satisfy it. Generation itself works.
        # pyrefly: ignore [bad-argument-type]
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # The training checkpoints record `use_cache: false` in
            # `config.json`, which is what `generate` would fall back on if a
            # checkpoint were copied without its `generation_config.json`.
            # Without a cache every step recomputes the whole prefix.
            use_cache=True,
        )

    continuation = output[:, encoded["input_ids"].shape[1] :]
    # Special tokens stay in the decode: `<think>` and `</think>` are
    # themselves special tokens, so `skip_special_tokens=True` would delete the
    # very delimiters the response is read by.
    return strip_terminators(
        cast(str, tokenizer.decode(continuation[0], skip_special_tokens=False))
    )


def print_field(console: Any, label: str, value: str, style: str) -> None:
    """Print `label value`, the label in bold, both in `style`.

    Without a console this is a plain `print`. With one the value goes through
    `Text.append` rather than a markup string, so a bracket in a family name or
    a response is printed instead of being read as markup.
    """
    if console is None or Text is None:
        print(f"{label:<10}{value}")
        return
    line = Text()
    line.append(f"{label:<10}", style=f"bold {style}")
    line.append(value, style=style)
    console.print(line)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # `soft_wrap` keeps a long sequence off rich's own line-wrapping, so what
    # is printed is what the model wrote whatever the terminal width.
    console = Console(soft_wrap=True) if Console is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("device: %s (%s)", device, torch.cuda.get_device_name(0))
    else:
        logger.warning("no CUDA device; generating on CPU will be very slow")

    tokenizer = load_tokenizer(args.tokenizer or args.model)
    model = load_model(args.model, device)

    logger.info(
        "ready: type a family name, one per line; Ctrl-D to quit "
        "(%d instructions in the mix)",
        len(INSTRUCTIONS),
    )

    while True:
        try:
            line = input("family> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        family = line.strip()
        if not family:
            continue

        instruction = random.choice(INSTRUCTIONS).format(family=family)
        logger.info("generating...")
        response = generate_one(model, tokenizer, instruction, device)
        if response is None:
            continue

        reasoning = extract_reasoning(response)
        sequence = extract_sequence(response)
        # The thinking span is one line per attribute; indent the continuations
        # under the label so the three fields stay apart.
        thinking = (
            "none found in the response"
            if reasoning is None
            else reasoning.replace("\n", "\n" + " " * 10)
        )
        print()
        print_field(console, "prompt:", instruction, "cyan")
        print_field(console, "thinking:", thinking, "yellow")
        print_field(
            console,
            "answer:",
            "none found in the response" if sequence is None else sequence,
            "green",
        )
        print()


if __name__ == "__main__":
    main()
