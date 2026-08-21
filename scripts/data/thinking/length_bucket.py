"""
Pure logic for discretizing protein length into reasoning-trace buckets.

No network calls, no file I/O.
"""


def length_bucket(length: int, max_bucket_floor: int = 400, bin_width: int = 100) -> str:
    """'[300-400]' style bucket string, feeds directly into reasoning.py's
    Triple(relation="has_length", value=length_bucket(n)) -- format_triple
    appends " amino acids", giving "(protein, has_length, [300-400] amino
    acids)".

    Bins: "[<100]" below the first bin_width, "[k*w-(k+1)*w]" for interior
    bins, "[>max_bucket_floor]" for the open-ended top bin (today: <100,
    100-200, 200-300, 300-400, >400, matching the current <512-residue
    dataset cap).

    Scales by raising max_bucket_floor when the cap is raised later (e.g. to
    900 for a <1024 cap) -- the interior bins fill in automatically, no
    other code change needed.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if length < bin_width:
        return f"[<{bin_width}]"
    if length >= max_bucket_floor:
        return f"[>{max_bucket_floor}]"
    lower = (length // bin_width) * bin_width
    upper = lower + bin_width
    return f"[{lower}-{upper}]"
