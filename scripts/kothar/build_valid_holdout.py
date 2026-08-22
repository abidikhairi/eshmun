"""
Build a small held-out validation set, disjoint from
data/kothar/pretrain_mix.parquet, at the same 10:5:3:1
proteins:pubmed:finemath:fineweb-edu ratio used for training (anchored to
100 protein sequences this time instead of 500,000).

"Unseen" is enforced by excluding every `entry` id already present in
pretrain_mix.parquet (all of it, including finemath's with-replacement
duplicates) before sampling fresh rows from the raw uniref-replay-mix
sources. UniRef50 rows get the same Ƥ-marker treatment
(reasoning.py's encode_sequence) as the training mix.

Usage:
    python3 scripts/kothar/build_valid_holdout.py
"""

import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "data", "thinking"))
from reasoning import encode_sequence  # noqa: E402

SEED = 4242  # distinct from the training-mix seed (42)

SOURCE_DIR = "/run/media/khairi/seagate/data/uniref-replay-mix/data"
PRETRAIN_MIX_FILE = os.path.join(_HERE, "..", "..", "data", "kothar", "pretrain_mix.parquet")
OUTPUT_FILE = os.path.join(_HERE, "..", "..", "data", "kothar", "valid_holdout.parquet")

# 10:5:3:1 proteins:pubmed:finemath:fineweb-edu, anchored to 100 proteins
TARGETS = {
    "uniref50": 100,
    "pubmed": 50,
    "finemath": 30,
    "fineedu": 10,
}

PROTEIN_TAG_RE = re.compile(r"^<protein>(.*)</protein>$")


def add_protein_markers(content: str) -> str:
    match = PROTEIN_TAG_RE.match(content)
    if not match:
        raise ValueError(f"unexpected uniref50 content format: {content[:50]!r}")
    return encode_sequence(match.group(1))


def main() -> None:
    train_mix = pd.read_parquet(PRETRAIN_MIX_FILE, columns=["entry", "source"])
    used_entries = {
        source: set(train_mix.loc[train_mix.source == source, "entry"])
        for source in TARGETS
    }
    for source, used in used_entries.items():
        print(f"{source}: {len(used)} distinct entries already used in training")

    frames = []
    for source, n in TARGETS.items():
        if source == "uniref50":
            # 10M rows -- avoid loading the whole 4.3GB file (same concern as
            # build_pretrain_mix.py); scan in chunks, keep only unseen rows,
            # stop once we've gathered comfortably more than needed.
            path = os.path.join(SOURCE_DIR, "uniref50", "train.csv")
            unseen_chunks = []
            unseen_count = 0
            reader = pd.read_csv(path, names=["entry", "content"], header=None, chunksize=500_000)
            for chunk in reader:
                unseen_chunk = chunk[~chunk["entry"].isin(used_entries[source])]
                unseen_chunks.append(unseen_chunk)
                unseen_count += len(unseen_chunk)
                if unseen_count >= n * 20:  # comfortable margin, then sample down
                    break
            unseen = pd.concat(unseen_chunks, ignore_index=True)
        else:
            path = os.path.join(SOURCE_DIR, source, "train.parquet")
            df = pd.read_parquet(path, columns=["entry", "content"])
            unseen = df[~df["entry"].isin(used_entries[source])]

        print(f"{source}: {len(unseen)} unseen rows scanned, sampling {n}")
        sample = unseen.sample(n=n, random_state=SEED).reset_index(drop=True)

        if source == "uniref50":
            sample["content"] = sample["content"].map(add_protein_markers)

        sample["source"] = source
        frames.append(sample)

    valid = pd.concat(frames, ignore_index=True)
    valid = valid.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Safety check: no overlap with the training mix at all.
    train_entries = set(train_mix["entry"])
    overlap = set(valid["entry"]) & train_entries
    assert not overlap, f"{len(overlap)} validation rows overlap the training mix: {overlap}"

    print("\nrow counts by source:")
    print(valid["source"].value_counts())
    print("total rows:", len(valid))

    valid.to_parquet(OUTPUT_FILE, index=False)
    print(f"wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
