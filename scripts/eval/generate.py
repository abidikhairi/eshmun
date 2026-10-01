#!/usr/bin/env python
"""Generate a response for every prompt in a JSON file, and parse each one.

The input is a JSON array of objects, each carrying an `input` string. Every
`input` goes to the model as a single user turn, and the array is written back
with three fields added per object:

  output              the model's response, verbatim
  reasoning           the text between <think> and </think>, or null
  sanitized_sequence  the residues inside <protein>...</protein>, or null

`output` is decoded with the special tokens left in place, because <think> and
</think> are themselves special tokens -- `skip_special_tokens=True` would
delete the very delimiters `reasoning` is read from. Only the trailing EOS is
dropped.

`sanitized_sequence` keeps just the characters in the 20-residue alphabet. That
drops the `Ƥ` marker on every residue token, and with it whatever the model
drifts into on leaving the sequence -- prose, or literal `</s>`/`<pad>` text
leaking through the decoder. Filtering to the alphabet is deliberate: stripping
only `Ƥ` has been observed on this project to leave that garbage behind.

When no `<protein>` tag appears at all -- which an early checkpoint does,
emitting the residues bare and with no `Sequence:` prefix -- the longest run of
`Ƥ`-prefixed residues stands in for the span. `Ƥ` occurs on residue tokens and
nowhere else, so this stays exact where filtering the whole response would not:
`family eukaryota` is almost entirely valid residue letters.

    python scripts/eval/generate.py --model runs/sft/checkpoint-1000 \\
        --input prompts.json --output generations.json

Sampling is on by default, so a run only reproduces when the seed, the batch
size and the input order are all held fixed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, NoReturn, cast

import torch
from tokenizers import decoders

try:
    from eshmun.models.eshmun import EshmunForCausalLM
    from eshmun.tokenization import EshmunTokenizer
except ModuleNotFoundError:
    # Running from a checkout rather than an installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from eshmun.models.eshmun import EshmunForCausalLM
    from eshmun.tokenization import EshmunTokenizer

logger = logging.getLogger("generate")

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
        "--model",
        required=True,
        help="Checkpoint directory or Hub repo id to generate from.",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer source; defaults to `--model`.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON array of objects, each with an `input` string.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the array with the three fields added.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Prompts per forward pass. 8 is safe in 6 GB at float32.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Cap on the response length; generation stops early at EOS.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=250,
        help="Top-k sampling cutoff.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Penalty on already-emitted tokens; without one this model tends "
        "to emit long runs of a single residue.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4242,
        help="Sampling seed. Sampling only reproduces at a fixed seed, batch "
        "size and input order.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """Read the input array and check every object carries an `input` string."""
    if not path.exists():
        _fail(f"input file not found: {path}")

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        _fail(f"{path} is not valid JSON: {error}")

    if not isinstance(data, list):
        _fail(f"{path} must hold a JSON array, got {type(data).__name__}")

    for i, record in enumerate(data):
        if not isinstance(record, dict):
            _fail(f"entry {i} is {type(record).__name__}, expected an object")
        if not isinstance(record.get("input"), str):
            _fail(f"entry {i} has no `input` string")

    logger.info("%d prompts from %s", len(data), path)
    return data


def _span(text: str, start: str, end: str) -> str | None:
    """Text between `start` and `end`, or None when `start` never appears.

    An `end` that never arrives still yields the remainder: generation hits
    `--max-new-tokens` often enough that discarding those spans would throw
    away sequences that are otherwise perfectly usable.
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


def extract_reasoning(text: str) -> str | None:
    span = _span(text, THINK_START, THINK_END)
    return span.strip() if span is not None else None


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


def extract_sequence(text: str) -> str | None:
    """Residues from the `<protein>` span, or from the bare marked runs.

    A span that is present but holds no residues yields "", which keeps
    "the model wrote nothing valid" distinct from "the model never got there".

    With no `<protein>` tag anywhere, the longest marked run stands in: an
    early checkpoint emits its residues bare, and the marker still identifies
    them exactly where filtering the whole response would sweep up prose. The
    longest run is taken rather than every run joined, so two fragments
    separated by prose do not become one chimeric sequence.
    """
    span = _span(text, PROTEIN_START, PROTEIN_END)
    if span is not None:
        return "".join(character for character in span if character in RESIDUES)
    runs = _residue_runs(text)
    return max(runs, key=len) if runs else None


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


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    """Rewrite the whole array atomically, so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def generate_batch(
    model: EshmunForCausalLM,
    tokenizer: EshmunTokenizer,
    prompts: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    """Return one decoded response per prompt, terminators stripped."""
    rendered = [render_prompt(tokenizer, prompt) for prompt in prompts]
    # `add_special_tokens=False` is load-bearing: this tokenizer's BOS is
    # `</s>`, so the default would put one in front of every prompt. The
    # training rows carry no BOS, and a prompt the model never saw the shape of
    # is not the prompt being evaluated.
    encoded = tokenizer(
        rendered, return_tensors="pt", padding=True, add_special_tokens=False
    )
    encoded = encoded.to(device)

    with torch.no_grad():
        # `generate` annotates itself with a `GenerativePreTrainedModel` self
        # protocol which types `device` as a plain attribute, while transformers
        # 5.x exposes it as a property -- so no custom PreTrainedModel subclass
        # can satisfy it. Generation itself works.
        # pyrefly: ignore [bad-argument-type]
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # The training checkpoints record `use_cache: false` in
            # `config.json`, which is what `generate` would fall back on if a
            # checkpoint were copied without its `generation_config.json`.
            # Without a cache every step recomputes the whole prefix.
            use_cache=True,
        )

    # Left padding means every row's prompt ends at the same column, so the
    # continuation starts at one index for the whole batch.
    continuations = output[:, encoded["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(continuations, skip_special_tokens=False)
    return [strip_terminators(text) for text in decoded]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.batch_size < 1:
        _fail(f"`--batch-size` must be at least 1, got {args.batch_size}")

    records = load_prompts(args.input)
    if not records:
        logger.warning("no prompts in %s; nothing to do", args.input)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("device: %s (%s)", device, torch.cuda.get_device_name(0))
    else:
        logger.warning("no CUDA device; generating on CPU will be very slow")
        logger.info("device: %s", device)

    tokenizer_source = args.tokenizer or args.model
    logger.info("loading tokenizer: %s", tokenizer_source)
    tokenizer = EshmunTokenizer.from_pretrained(tokenizer_source)

    # The fine-tuning checkpoints save a `tokenizer.json` with no `decoder`, so
    # `decode` cannot invert the byte-level encoding: it falls back to joining
    # tokens with spaces and leaves `Ġ` (space) and `Ċ` (newline) literal in the
    # text. The Hub model of the same lineage ships the decoder; restore it
    # rather than shipping a response the model never wrote.
    if tokenizer.backend_tokenizer.decoder is None:
        tokenizer.backend_tokenizer.decoder = decoders.ByteLevel()
        logger.warning(
            "tokenizer at %s carries no decoder; installed ByteLevel",
            tokenizer_source,
        )

    if not tokenizer.chat_template:
        _fail(
            f"tokenizer at {tokenizer_source!r} has no chat template, which is "
            "how the prompt is built. Point `--tokenizer` at a checkpoint that "
            "ships one."
        )
    if tokenizer.pad_token_id is None:
        _fail("tokenizer has no pad token, which batched generation needs")

    # Left padding is what lets a batch be sliced at a single index below.
    tokenizer.padding_side = "left"

    # float32 on purpose: float16 has repeatedly produced unexpected errors on
    # this project.
    logger.info("loading model: %s (float32)", args.model)
    model = EshmunForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()

    # The positional table holds `max_position_embeddings + 2` entries and
    # position p lands at index p + 2, so `max_position_embeddings` is the hard
    # ceiling on prompt + continuation. `generate` counts the continuation from
    # the padded prompt width, which is the longest prompt in the batch, so
    # checking the longest prompt overall covers every batch. Catching this here
    # beats an IndexError partway through a long run.
    ceiling = model.config.max_position_embeddings
    over: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        rendered = render_prompt(tokenizer, record["input"])
        length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if length + args.max_new_tokens > ceiling:
            over.append((index, length))
    if over:
        shown = ", ".join(f"entry {i} ({n} tokens)" for i, n in over[:5])
        _fail(
            f"{len(over)} prompt(s) overrun the {ceiling}-token positional "
            f"limit at `--max-new-tokens {args.max_new_tokens}`: {shown}"
            + (", ..." if len(over) > 5 else "")
            + ". Lower `--max-new-tokens`, or drop those entries."
        )

    torch.manual_seed(args.seed)

    # Placeholders first, so an interrupted run still leaves a well-formed
    # array. `output` stays null on anything not yet generated.
    for record in records:
        record["output"] = None
        record["reasoning"] = None
        record["sanitized_sequence"] = None

    # Written before the first batch, so even a crash on batch one leaves a
    # valid file behind.
    write_json(args.output, records)

    total = len(records)
    for start in range(0, total, args.batch_size):
        batch = records[start : start + args.batch_size]
        responses = generate_batch(
            model, tokenizer, [record["input"] for record in batch], args, device
        )

        for record, response in zip(batch, responses):
            record["output"] = response
            record["reasoning"] = extract_reasoning(response)
            record["sanitized_sequence"] = extract_sequence(response)

        write_json(args.output, records)
        logger.info("%d/%d prompts done", min(start + args.batch_size, total), total)

    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
