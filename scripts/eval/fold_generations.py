#!/usr/bin/env python
"""Fold every generated sequence in a JSON file with ESMFold2, via Biohub.

The input is a JSON array of objects of the shape `generate.py` writes, each
carrying a `sanitized_sequence` string and a `protein_accession`. Every
sequence goes to the Biohub ESMFold2 service as a single-chain protein and
the returned complex is written as mmCIF, under a directory named after the
input file with the `.json` extension stripped:

    runs/eval/generations-13750-20261002-001234.json
    runs/eval/generations-13750-20261002-001234/0000_A0A0H2ZFK2.cif

Files are named `<index>_<accession>.cif`: the index preserves the order of
the input array and disambiguates accessions, which repeat -- the full test
set holds 2,524 rows but only 2,223 unique accessions. mmCIF rather than PDB
because that is the only conversion ESMFold2's `MolecularComplex` offers
(`to_mmcif`). Alongside each mmCIF, a `<index>_<accession>.json` holds the
confidence ESMFold2 returns: `plddt` per residue on the 0-100 scale of the
B-factor column, `pae` in angstroms (residue x residue), and `ptm`.

The token comes from the `BIOHUB_TOKEN` environment variable. The model and
folding config are the ones the Biohub ESMFold2 tutorial notebook uses
(`cookbook/tutorials/esmfold2.ipynb` in the Biohub/esm GitHub repo):

    model="esmfold2-2026-05"
    FoldingConfig(num_loops=10, num_sampling_steps=100, include_pae=True)

Without `include_pae` the service returns PAE with the fold anyway; the flag
only decided whether we keep it. It is kept now, in the metrics sidecar.

Entries whose `sanitized_sequence` is null (a response that parsed to
nothing) are skipped, and so is any sequence whose mmCIF already exists.
That makes a re-run resume rather than re-fold, and it is also the retry
story: a call that fails is logged and folded on the next run. A sequence
whose mmCIF exists but whose metrics sidecar does not is re-folded, so a
run interrupted between the two writes comes back consistent.

Requires the `esm` SDK installed and network access:

    pip install esm
    python scripts/eval/fold_generations.py runs/eval/generations-13750-*.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, NoReturn

from esm.sdk import esmfold2_client
from esm.sdk.api import ESMProteinError, FoldingConfig
from esm.sdk.forge import SequenceStructureForgeInferenceClient
from esm.utils.structure import input_builder

logger = logging.getLogger("fold")

URL = "https://biohub.ai"
MODEL = "esmfold2-2026-05"
CONFIG = FoldingConfig(num_loops=10, num_sampling_steps=100, include_pae=True)

TOKEN_VARIABLE = "BIOHUB_TOKEN"


def _fail(message: str) -> NoReturn:
    """Abort with a readable message instead of a traceback."""
    logger.error(message)
    raise SystemExit(1)


def write_atomic(path: Path, text: str) -> None:
    """Write through a sibling `.tmp` so a killed run leaves no partial file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "generations",
        type=Path,
        help="JSON array of generations, each with `sanitized_sequence` and "
        "`protein_accession`; its `.json` is stripped to name the output "
        "directory beside it.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read the generations array and check every object carries an accession."""
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
        if not isinstance(record.get("protein_accession"), str):
            _fail(f"entry {i} has no `protein_accession` string")

    logger.info("%d entries from %s", len(data), path)
    return data


def prediction_input(sequence: str) -> input_builder.StructurePredictionInput:
    """One single-chain protein, the plain fold request."""
    return input_builder.StructurePredictionInput(
        sequences=[input_builder.ProteinInput(id="A", sequence=sequence)]
    )


def fold(
    client: SequenceStructureForgeInferenceClient, sequence: str, label: str
) -> tuple[str, dict[str, Any]] | None:
    """mmCIF text and confidence metrics for one sequence, or None when this fails."""
    try:
        result = client.fold_all_atom(prediction_input(sequence), config=CONFIG)
    except Exception as error:
        logger.warning("%s failed: %s", label, error)
        return None
    if isinstance(result, ESMProteinError):
        logger.warning("%s failed: %s", label, result)
        return None
    if isinstance(result, list):
        # One structure input answers with one result; the list is the batched
        # shape, and anything but a single element is unusable here.
        if len(result) != 1:
            logger.warning("%s failed: %d results for one input", label, len(result))
            return None
        result = result[0]

    plddt, pae = result.plddt, result.pae
    metrics: dict[str, Any] = {
        "ptm": result.ptm,
        "plddt": [],
        "pae": [],
    }
    if plddt is not None:
        # ESMFold2 answers on a 0-1 scale; 0-100 is the scale the mmCIF
        # B-factor column and the usual pLDDT thresholds use.
        metrics["plddt"] = [round(float(value) * 100, 2) for value in plddt]
    if pae is not None:
        # Angstroms, per residue pair. Rounded because the raw floats at two
        # decimals are still finer than the model's own resolution.
        metrics["pae"] = [[round(float(value), 2) for value in row] for row in pae]
    return result.complex.to_mmcif(), metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.generations.suffix != ".json":
        _fail(f"expected a .json file, got {args.generations}")

    token = os.environ.get(TOKEN_VARIABLE)
    if not token:
        _fail(
            f"{TOKEN_VARIABLE} is not set. Create an API key in the Biohub "
            "developer console (https://biohub.ai/developer-console/api-keys) "
            "and export it, e.g. in ~/.zshrc."
        )

    records = load_records(args.generations)
    output_dir = args.generations.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Everything with a sequence worth folding; nulls (a response that parsed
    # to nothing) are dropped here and counted below.
    todo = [
        (index, record["protein_accession"], record["sanitized_sequence"])
        for index, record in enumerate(records)
        if record.get("sanitized_sequence")
    ]
    empty = len(records) - len(todo)
    if not todo:
        logger.warning("no sequences to fold in %s", args.generations)
        return

    logger.info(
        "folding %d sequences (%d entries with no sequence skipped) -> %s",
        len(todo),
        empty,
        output_dir,
    )
    client = esmfold2_client(model=MODEL, url=URL, token=token)

    folded = skipped = failed = 0
    for position, (index, accession, sequence) in enumerate(todo, start=1):
        target = output_dir / f"{index:04d}_{accession}.cif"
        metrics_target = target.with_suffix(".json")
        if target.exists() and metrics_target.exists():
            skipped += 1
            continue

        started = time.monotonic()
        label = f"{position}/{len(todo)} {target.name}"
        result = fold(client, sequence, label)
        if result is None:
            failed += 1
            continue
        mmcif, metrics = result

        # A profile that dies mid-write must not leave a truncated file behind:
        # the skip-if-exists above would then take it for a finished fold. The
        # mmCIF goes first, so if only the metrics are missing the pair is
        # re-folded rather than left half written.
        write_atomic(target, mmcif)
        write_atomic(metrics_target, json.dumps(metrics))
        folded += 1
        logger.info(
            "%d/%d %s %d aa -> %s (%.1fs)",
            position,
            len(todo),
            accession,
            len(sequence),
            target.name,
            time.monotonic() - started,
        )

    logger.info(
        "folded %d, skipped %d existing, skipped %d without a sequence, failed %d",
        folded,
        skipped,
        empty,
        failed,
    )
    if failed:
        logger.warning("failed folds are retried by re-running this script")


if __name__ == "__main__":
    main()
