"""Protein-sequence reward functions.

All classes are callable and satisfy the RewardFn signature:
    (prompts: Sequence[str], completions: Sequence[str]) -> list[float]
"""

import re
from collections.abc import Sequence

from eshmun.trainer.grpo.reward_functions.registry import register

# Standard 20 amino acids
_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
PROTEIN_PREFIX_TOKEN = "Ƥ"


def extract_sequence(completion: str):
    """
    Eshmun wraps the protein sequence between <protein>...</protein>
    """
    matches = re.search(r"<protein>(.*?)</protein>", completion, re.DOTALL)
    sequence = matches.group(1).strip() if matches else None

    if sequence is not None:
        return sequence.replace(PROTEIN_PREFIX_TOKEN, "")

    return sequence


@register
class SequenceValidityReward:
    """
    Returns 1.0 if every character in the completion is a valid amino-acid
    letter (case-insensitive), 0.0 otherwise.

    Useful as a hard filter before any downstream scoring.
    """

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], **kwargs
    ) -> list[float]:
        scores = []
        for completion in completions:
            stripped = extract_sequence(completion.strip())

            if stripped and all(c in _VALID_AA for c in stripped.upper()):
                scores.append(1.0)
            else:
                scores.append(0.0)
        return scores


@register
class SequenceSimilarityReward:
    """
    Returns the global pairwise sequence identity between the completion and a reference sequence.
    both sequences wrapped in ``<protein>...</protein>`` tags.

    Identity is computed as ``alignment_score / max(len(seq), len(ref))`` using
    BioPython's ``PairwiseAligner`` (global mode, match=1, mismatch/gap=0),
    giving a value in [0, 1]. Returns ``non_valid_seq_reward`` when either the
    completion or the reference cannot be parsed.

    Args:
        non_valid_seq_reward: Score returned for unparseable completions or
            references. Defaults to -1.
    """

    def __init__(self, non_valid_seq_reward: float = -1):
        from Bio.Align import PairwiseAligner

        self.not_valid_penalty = non_valid_seq_reward
        self._aligner = PairwiseAligner(
            mode="global",
            match_score=1,
            mismatch_score=0,
            open_gap_score=0,
            extend_gap_score=0,
        )

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], **kwargs
    ) -> list[float]:
        family_stats = kwargs.get("protein_family_stat", [])

        scores = []
        for completion, stat in zip(completions, family_stats):
            reference = stat["representative"]
            sequence = extract_sequence(completion.strip())
            if sequence is None:
                scores.append(self.not_valid_penalty)
                continue
            denom = max(len(sequence), len(reference))
            if denom == 0:
                scores.append(self.not_valid_penalty)
                continue
            identity = float(self._aligner.score(sequence, reference)) / denom

            scores.append(identity)
        return scores


@register
class LengthReward:
    """
    Rewards completions whose length falls within [min_length, max_length].

    Args:
        min_length: Minimum acceptable sequence length (inclusive).
        max_length: Maximum acceptable sequence length (inclusive).
        out_of_range_penalty: Score returned when length is outside the range.
            Defaults to 0.0 (hard constraint); set to a negative value for an
            explicit penalty.
    """

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 512,
        out_of_range_penalty: float = 0.0,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.out_of_range_penalty = out_of_range_penalty

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], **kwargs
    ) -> list[float]:
        family_stats = kwargs.get("protein_family_stat", [])
        scores = []
        for completion, stat in zip(completions, family_stats):
            sequence = extract_sequence(completion.strip())

            if sequence is not None:
                length = len(sequence)
                min_len = float(stat["min_len"])
                max_len = float(stat["max_len"])
                if min_len <= length <= max_len:
                    scores.append(1.0)
                else:
                    scores.append(self.out_of_range_penalty)
            else:
                scores.append(self.out_of_range_penalty)

        return scores


@register
class AminoAcidRepetitionPenalty:
    """
    Penalizes proteins with long consecutive runs of a single amino acid.

    Returns -1.0 if any amino acid repeats consecutively >= ``min_aa_repetition``
    times; 1.0 otherwise.  Unparseable completions also receive -1.0.

    Args:
        min_aa_repetition: Consecutive run length that triggers the penalty.
            Defaults to 10.
    """

    def __init__(self, min_aa_repetition: int = 10):
        self.min_aa_repetition = min_aa_repetition
        self._run_re = re.compile(r"(.)\1+")

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], **kwargs
    ) -> list[float]:
        scores = []
        for completion in completions:
            sequence = extract_sequence(completion.strip())

            if sequence is not None:
                has_long_run = any(
                    len(m.group(0)) >= self.min_aa_repetition
                    for m in self._run_re.finditer(sequence)
                )
                scores.append(-1.0 if has_long_run else 1.0)
            else:
                scores.append(-1.0)

        return scores


@register
class CompositeProteinReward:
    """
    Combines SequenceValidityReward and LengthReward: a completion must be
    both valid and within the length range to receive a positive score.

    A completion that fails validity always gets 0.0 regardless of length.

    Args:
        min_length: Passed to LengthReward.
        max_length: Passed to LengthReward.
    """

    def __init__(self, min_length: int = 20, max_length: int = 512):
        self._validity = SequenceValidityReward()
        self._length = LengthReward(min_length, max_length)

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], **kwargs
    ) -> list[float]:
        validity = self._validity(prompts, completions, **kwargs)
        length = self._length(prompts, completions, **kwargs)
        return [v * l for v, l in zip(validity, length)]
