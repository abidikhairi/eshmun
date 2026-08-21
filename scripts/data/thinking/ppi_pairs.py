"""
Pure logic for PPI positive/negative pair sampling.

No network calls, no file I/O.
"""

import random


def build_positive_pair_set(rows: list[tuple[str, str]]) -> set[frozenset]:
    """rows: (entry, partner) interacts_with triples -> unordered pair set.

    UniProt's cc_interaction is recorded per-entry, so a real interaction
    between A and B may appear as (A, B), (B, A), or -- often -- both; a
    frozenset dedupes all of these to one undirected fact, so downstream
    negative sampling checks a single canonical positive set rather than
    having to check both directions everywhere.
    """
    return {frozenset((a, b)) for a, b in rows}


def build_partners_by_entry(positive_pairs: set[frozenset]) -> dict[str, set[str]]:
    """positive_pairs -> {entry: {known true partners}}, both directions."""
    partners: dict[str, set[str]] = {}
    for pair in positive_pairs:
        if len(pair) == 1:
            # self-interaction: {entry} as a 1-element frozenset
            (entry,) = tuple(pair)
            partners.setdefault(entry, set()).add(entry)
            continue
        a, b = tuple(pair)
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def sample_negatives_for_entry(
    entry: str,
    true_partners: set[str],
    candidate_pool: list[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Sample n negative partners for `entry` from candidate_pool, excluding
    entry itself and any of its true positive partners (in either
    direction). Raises if the pool doesn't have enough eligible candidates
    -- silently sampling fewer than requested would quietly break the 1:3
    ratio rather than surfacing the data problem."""
    eligible = [c for c in candidate_pool if c != entry and c not in true_partners]
    if len(eligible) < n:
        raise ValueError(
            f"not enough eligible negative candidates for {entry}: need {n}, have {len(eligible)}"
        )
    return rng.sample(eligible, n)
