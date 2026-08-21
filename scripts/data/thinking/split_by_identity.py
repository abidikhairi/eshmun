"""
Split the annotation KG's ~20,431 human entries into train/validation/test by
sequence identity, using MMseqs2 -- same methodology and thresholds as the
InstructProtein pilot's split_annotation_by_identity.py, run here at full
scale instead of on a 400-entry pilot sample.

Validation entries may be up to 70% identical to a train sequence (a "near"
validation set); test entries must be below 30% identity to any train
sequence (a "far", remote-homology generalization test). This guards against
homology leakage: annotation's sequence is the model's *input*, so a plain
random split risks train/test sharing near-identical sequences, letting the
model "cheat" via a trivial sequence-similarity shortcut rather than genuine
generalization.

Output: data/thinking/processed/annotation_split.csv, columns (entry, split).

Usage:
    python3 scripts/data/thinking/split_by_identity.py
"""

import os
import subprocess
import tempfile

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "..", "data", "thinking")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ANNOTATION_KG_FILE = os.path.join(PROCESSED_DIR, "annotation_kg.parquet")
# NOTE: swissprot_sequence_features.parquet (also in RAW_DIR) looks like a
# plausible sequence source but is NOT one -- it's a leftover from the old,
# unrelated MLM-era SwissProtDataset pipeline and is silently pre-filtered to
# length <= 400 (its max length is exactly 400, no coincidence), which would
# silently drop >half of our human entries. homosapiens-sequences.tsv is the
# real, unfiltered source (one row per human entry, full length range).
SEQ_FEATURES_FILE = os.path.join(RAW_DIR, "homosapiens-sequences.tsv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "annotation_split.csv")

TRAIN_RATIO = 0.8
VALID_IDENTITY = 0.70  # valid: <70% identity to train (moderately similar allowed)
TEST_IDENTITY = 0.30  # test: <30% identity to train (remote homology)
SEED = 42
MAX_SEQUENCE_LENGTH = 512  # locked scope decision: dataset covers proteins < 512 residues


def partition_by_similarity(candidate_entries: list[str], too_similar: set[str]) -> tuple[list[str], list[str]]:
    """Splits candidate_entries into (moved_to_train, remaining), where
    moved_to_train are the ones flagged as too similar to an existing train
    sequence by an mmseqs search. Order-preserving, pure (no I/O)."""
    moved = [e for e in candidate_entries if e in too_similar]
    remaining = [e for e in candidate_entries if e not in too_similar]
    return moved, remaining


def tsv_to_fasta(df: pd.DataFrame, fasta_path: str) -> None:
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['Entry']}\n{row['Sequence']}\n")


def run_mmseqs_search(query_fasta: str, target_fasta: str, outdir: str, identity: float, tmpdir: str) -> set[str]:
    """Query entries with a hit >= identity against target -> 'too similar' set."""
    result_tsv = os.path.join(outdir, f"search_{int(identity * 100)}.tsv")
    cmd = [
        "mmseqs", "easy-search", query_fasta, target_fasta, result_tsv, tmpdir,
        "--min-seq-id", str(identity), "-c", "0.8", "--cov-mode", "0",
        "--format-output", "query,target,pident", "-v", "1",
    ]
    print(f"[MMseqs2] Searching at {int(identity * 100)}% identity...")
    subprocess.run(cmd, check=True)

    if os.path.getsize(result_tsv) == 0:
        return set()

    hits = pd.read_csv(result_tsv, sep="\t", header=None, names=["query", "target", "pident"])
    hits = hits[hits["query"] != hits["target"]]
    return set(hits["query"].unique())


def run_identity_split(
    df: pd.DataFrame,
    output_file: str,
    train_ratio: float = TRAIN_RATIO,
    valid_identity: float = VALID_IDENTITY,
    test_identity: float = TEST_IDENTITY,
    seed: int = SEED,
) -> None:
    """Shared core: df has columns (Entry, Sequence), already filtered to the
    caller's entry population and length cap. Reused by both task family A
    (human-only) and task family B (SCOP, all organisms) -- the split
    methodology doesn't depend on which KG the entries came from, only on
    having a sequence to run MMseqs2 against."""
    print(f"[1/5] Random candidate split (train_ratio={train_ratio})...")
    train_df = df.sample(frac=train_ratio, random_state=seed)
    holdout_df = df.drop(train_df.index)
    print(f"      candidate train: {len(train_df)}  candidate holdout: {len(holdout_df)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_fasta = os.path.join(tmpdir, "train.fasta")
        holdout_fasta = os.path.join(tmpdir, "holdout.fasta")
        tsv_to_fasta(train_df, train_fasta)
        tsv_to_fasta(holdout_df, holdout_fasta)

        print(f"[2/5] Filtering valid set (<{int(valid_identity * 100)}% id to train)...")
        too_similar_valid = run_mmseqs_search(
            query_fasta=holdout_fasta, target_fasta=train_fasta, outdir=tmpdir,
            identity=valid_identity, tmpdir=os.path.join(tmpdir, "search_tmp_valid"),
        )
        moved_to_train, valid_candidates = partition_by_similarity(list(holdout_df["Entry"]), too_similar_valid)
        valid_df = holdout_df[holdout_df["Entry"].isin(valid_candidates)]
        extra_train = holdout_df[holdout_df["Entry"].isin(moved_to_train)]
        train_df = pd.concat([train_df, extra_train])
        print(f"      moved back to train (>= {int(valid_identity * 100)}% id): {len(extra_train)}")
        print(f"      valid candidate size: {len(valid_df)}")

        print(f"[3/5] Filtering test set (<{int(test_identity * 100)}% id to train)...")
        valid_fasta = os.path.join(tmpdir, "valid.fasta")
        tsv_to_fasta(valid_df, valid_fasta)
        too_similar_test = run_mmseqs_search(
            query_fasta=valid_fasta, target_fasta=train_fasta, outdir=tmpdir,
            identity=test_identity, tmpdir=os.path.join(tmpdir, "search_tmp_test"),
        )
        # Entries with a >=30% identity hit against train aren't a fair
        # remote-homology test case, but they already cleared the <70% check,
        # so they land in the final validation set rather than being dropped.
        final_valid_entries, test_entries = partition_by_similarity(list(valid_df["Entry"]), too_similar_test)
        test_df = valid_df[valid_df["Entry"].isin(test_entries)]
        valid_df = valid_df[valid_df["Entry"].isin(final_valid_entries)]
        print(f"      test size: {len(test_df)}  valid size (final): {len(valid_df)}")

    print(f"[4/5] Saving split to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    rows = (
        [{"entry": e, "split": "train"} for e in train_df["Entry"]]
        + [{"entry": e, "split": "validation"} for e in valid_df["Entry"]]
        + [{"entry": e, "split": "test"} for e in test_df["Entry"]]
    )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_file, index=False)

    total = len(df)
    print("\n--- split summary ---")
    for split_name in ["train", "validation", "test"]:
        n = (out_df["split"] == split_name).sum()
        print(f"  {split_name:<10}: {n:>6}  ({100 * n / total:.1f}%)")
    print(f"  total     : {total}")


def main() -> None:
    annotation_kg = pd.read_parquet(ANNOTATION_KG_FILE)
    human_entries = set(annotation_kg["entry"].unique())
    print(f"{len(human_entries)} human entries from the annotation KG")

    seq_features = pd.read_csv(SEQ_FEATURES_FILE, sep="\t", dtype=str)
    df = seq_features[seq_features["Entry"].isin(human_entries)][["Entry", "Sequence"]].drop_duplicates("Entry")
    missing = human_entries - set(df["Entry"])
    if missing:
        print(f"WARNING: {len(missing)} entries have no sequence in {SEQ_FEATURES_FILE}, dropping them")

    before_length_filter = len(df)
    df = df[df["Sequence"].str.len() < MAX_SEQUENCE_LENGTH]
    print(f"{len(df)}/{before_length_filter} entries with a sequence < {MAX_SEQUENCE_LENGTH} residues")

    run_identity_split(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
