#!/usr/bin/env python
"""Generate one response per prompt with InstructProtein, and parse each one.

InstructProtein (https://arxiv.org/abs/2310.03269) is the instruct-tuned
OPT-1.3B this project distills, so it is the baseline the fine-tuned
checkpoints are measured against. The prompts are the ones `generate.py`
takes -- a JSON array whose objects carry `input`, `family_name` and
`protein_accession` -- but the model is asked with its own template, the one
its repository's `generate_family.py` uses:

    Instruction: I would like a protein that is in {family}. Output: <protein>

`input` is rewritten to that rendered prompt: the prompt the model saw is
what the response is a response to, and the two models are not asked alike.

The response is the continuation alone. The prompt's tokens are sliced off
before decoding, so the text carries no echo of the instruction, and
`</protein>` is the stop token, as in the model's own scripts. `output`,
`reasoning` and `sanitized_sequence` are then added, matching the array
`generate.py` writes:

  output              the continuation, verbatim, terminators stripped
  reasoning           always null; this model was not trained to think
  sanitized_sequence  the residues the response carries, or null

The opening `<protein>` is part of the prompt, so it is absent from the
continuation and the span-based parse cannot anchor on it. The residues come
from the longest run of `Ƥ`-prefixed markers instead -- the same fallback
`generate.py` uses for a checkpoint that emits its response untagged, and it
is exact here because `Ƥ` occurs on residue tokens and nowhere else.

Sampling is on, so a run reproduces only when seed, model, and input order
are held fixed.

    python scripts/eval/generate_instructprotein.py \\
        --model ~/workspace/software/InstructProtein/model/InstructProtein \\
        --input runs/eval/test50/prompts.json \\
        --output runs/eval/instructprotein/generations-test50.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from eshmun.errors import fail as _fail
    from eshmun.tokenization import PROTEIN_END, longest_marked_run, strip_terminators
except ModuleNotFoundError:
    # Running from a checkout rather than an installed package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from eshmun.errors import fail as _fail
    from eshmun.tokenization import PROTEIN_END, longest_marked_run, strip_terminators

logger = logging.getLogger("generate_instructprotein")

# The model's own prompt template, from `generate_family.py` in the
# InstructProtein repository.
PROMPT_TEMPLATE = (
    "Instruction: I would like a protein that is in {family}. Output: <protein>"
)

# Generation settings from the same scripts.
MAX_NEW_TOKENS = 512
TOP_K = 40
TOP_P = 0.95

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Directory holding the InstructProtein checkpoint and tokenizer.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON array of objects, each with `family_name` and "
        "`protein_accession`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the array with the three fields added.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4242,
        help="Sampling seed. Sampling only reproduces at a fixed seed and "
        "input order.",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """Read the input array and check the fields the template and the fold need."""
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
        if not isinstance(record.get("family_name"), str):
            _fail(f"entry {i} has no `family_name` string")
        if not isinstance(record.get("protein_accession"), str):
            _fail(f"entry {i} has no `protein_accession` string")

    logger.info("%d prompts from %s", len(data), path)
    return data


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    """Rewrite the whole array atomically, so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def generate_one(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    eos_token_id: int,
) -> str:
    """Return the decoded continuation for one prompt, terminators stripped."""
    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **encoded,
            do_sample=True,
            max_new_tokens=MAX_NEW_TOKENS,
            top_k=TOP_K,
            top_p=TOP_P,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    continuation = output[0, encoded["input_ids"].shape[1] :]
    # Special tokens are left in place, as in `generate.py`: `</protein>` is
    # the stop marker and the response is only its own body.
    return strip_terminators(tokenizer.decode(continuation, skip_special_tokens=False))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    records = load_prompts(args.input)
    if not records:
        logger.warning("no prompts in %s; nothing to do", args.input)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("device: %s (%s)", device, torch.cuda.get_device_name(0))
    else:
        logger.warning("no CUDA device; generating on CPU will be very slow")

    logger.info("loading tokenizer: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer is None:
        _fail(f"no tokenizer at {args.model}")

    eos_token_id = tokenizer.convert_tokens_to_ids(PROTEIN_END)
    if eos_token_id is None or eos_token_id == tokenizer.unk_token_id:
        _fail(f"tokenizer at {args.model} has no {PROTEIN_END} token to stop at")

    # float32 on purpose: float16 has repeatedly produced unexpected errors on
    # this project, and the checkpoint is stored in float32 anyway.
    logger.info("loading model: %s (float32)", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()

    torch.manual_seed(args.seed)

    # Placeholders first, so an interrupted run still leaves a well-formed
    # array. Everything stays null until its own response is written, which
    # keeps "not generated yet" distinct from "generated nothing".
    for record in records:
        record["input"] = PROMPT_TEMPLATE.format(family=record["family_name"])
        record["output"] = None
        record["reasoning"] = None
        record["sanitized_sequence"] = None

    # Written before the first prompt, so even a crash on prompt one leaves a
    # valid file behind.
    write_json(args.output, records)

    total = len(records)
    for index, record in enumerate(records, start=1):
        response = generate_one(
            model, tokenizer, record["input"], device, eos_token_id
        )
        record["output"] = response
        record["reasoning"] = None
        record["sanitized_sequence"] = longest_marked_run(response)
        write_json(args.output, records)

        logger.info(
            "%d/%d %s %s -> %d residues",
            index,
            total,
            record["protein_accession"],
            record["family_name"],
            len(record["sanitized_sequence"] or ""),
        )

    empty = sum(1 for record in records if not record["sanitized_sequence"])
    logger.info("wrote %s (%d with no sequence)", args.output, empty)


if __name__ == "__main__":
    main()
