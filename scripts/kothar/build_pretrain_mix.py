"""
Build the Kothar Stage-1 continued-pretraining mix: UniRef50 protein
sequences plus a small amount of general-domain text, sampled from the
existing uniref-replay-mix raw sources.

Unlike that source mix (which also includes a StarCoder code split, for a
broader replay buffer), this one is protein + natural text only:

    - UniRef50 sequences:  2,000,000  (of ~10,000,000)
    - PubMed abstracts:       20,000  (of    283,302)
    - FineWeb-Edu:              5,000  (of    257,645)
    - FineMath:                  5,000  (of     54,615)

UniRef50 sequences are re-encoded with a `Ƥ` marker on every residue (via
scripts/data/thinking/reasoning.py's encode_sequence), so InstructProtein's
tokenizer lands on the dedicated per-residue `ƤX` tokens instead of
ambiguous plain-English BPE. Output is a single local parquet file (entry,
content, source columns), shuffled, seed 42 throughout for reproducibility.
Not pushed anywhere.

Usage:
    python3 scripts/kothar/build_pretrain_mix.py
"""

import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "data", "thinking"))
from reasoning import encode_sequence  # noqa: E402

SEED = 42

PROTEIN_TAG_RE = re.compile(r"^<protein>(.*)</protein>$")

SOURCE_DIR = "/run/media/khairi/seagate/data/uniref-replay-mix/data"
OUTPUT_DIR = "/run/media/khairi/seagate/data/eshmun/data/kothar"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pretrain_mix.parquet")

UNIREF_TARGET = 2_000_000
UNIREF_CHUNKSIZE = 1_000_000
UNIREF_OVERSAMPLE_FRAC = 0.25  # > 2M/10M, so one pass over the file is enough

PARQUET_TARGETS = {
    "pubmed": 20_000,
    "fineedu": 5_000,
    "finemath": 5_000,
}


def add_protein_markers(content: str) -> str:
    match = PROTEIN_TAG_RE.match(content)
    if not match:
        raise ValueError(f"unexpected uniref50 content format: {content[:50]!r}")
    return encode_sequence(match.group(1))


def sample_uniref50() -> pd.DataFrame:
    path = os.path.join(SOURCE_DIR, "uniref50", "train.csv")
    chunks = []
    total = 0
    reader = pd.read_csv(
        path, names=["entry", "content"], header=None, chunksize=UNIREF_CHUNKSIZE
    )
    for i, chunk in enumerate(reader):
        sampled = chunk.sample(frac=UNIREF_OVERSAMPLE_FRAC, random_state=SEED + i)
        chunks.append(sampled)
        total += len(sampled)
        if total >= UNIREF_TARGET:
            break
    df = pd.concat(chunks, ignore_index=True)
    assert len(df) >= UNIREF_TARGET, (
        f"only gathered {len(df)} UniRef50 rows, need {UNIREF_TARGET} "
        "-- raise UNIREF_OVERSAMPLE_FRAC or read more chunks"
    )
    df = df.sample(n=UNIREF_TARGET, random_state=SEED).reset_index(drop=True)
    df["content"] = df["content"].map(add_protein_markers)
    df["source"] = "uniref50"
    return df


def sample_parquet(name: str, n: int) -> pd.DataFrame:
    path = os.path.join(SOURCE_DIR, name, "train.parquet")
    df = pd.read_parquet(path, columns=["entry", "content"])
    df = df.sample(n=n, random_state=SEED).reset_index(drop=True)
    df["source"] = name
    return df


def main() -> None:
    frames = [sample_uniref50()]
    for name, n in PARQUET_TARGETS.items():
        frames.append(sample_parquet(name, n))

    mix = pd.concat(frames, ignore_index=True)
    mix = mix.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print("row counts by source:")
    print(mix["source"].value_counts())
    print("total rows:", len(mix))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mix.to_parquet(OUTPUT_FILE, index=False)
    print(f"wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
