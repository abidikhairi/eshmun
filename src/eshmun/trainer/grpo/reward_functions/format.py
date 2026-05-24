"""Format and structure reward functions.

All classes are callable and satisfy the RewardFn signature:
    (prompts: Sequence[str], completions: Sequence[str]) -> list[float]
"""

import re
from collections.abc import Callable, Sequence

from eshmun.trainer.grpo.reward_functions.registry import register


@register
class RegexReward:
    """
    Returns 1.0 if the completion matches a regular expression, 0.0 otherwise.

    Useful for enforcing structural constraints on generated text, e.g.
    checking that a response contains a required tag or follows a template.

    Args:
        pattern: Regular expression pattern to match against completions.
        flags: re module flags (e.g. re.IGNORECASE). Defaults to 0.
        full_match: If True, use re.fullmatch; otherwise use re.search.
    """

    def __init__(self, pattern: str, flags: int = 0, full_match: bool = False):
        self._re = re.compile(pattern, flags)
        self._match_fn = self._re.fullmatch if full_match else self._re.search

    def __call__(self, prompts: Sequence[str], completions: Sequence[str]) -> list[float]:
        return [1.0 if self._match_fn(c) else 0.0 for c in completions]


@register
class PredicateReward:
    """
    Applies an arbitrary boolean predicate to each completion.

    Returns 1.0 when the predicate returns True, 0.0 otherwise.

    Args:
        predicate: A callable that takes a single completion string and returns bool.
        reward_on_true: Score to return when predicate is True. Defaults to 1.0.
        reward_on_false: Score to return when predicate is False. Defaults to 0.0.

    Example::

        reward = PredicateReward(lambda c: c.startswith("<think>"))
    """

    def __init__(
        self,
        predicate: Callable[[str], bool],
        reward_on_true: float = 1.0,
        reward_on_false: float = 0.0,
    ):
        self._predicate = predicate
        self._on_true = reward_on_true
        self._on_false = reward_on_false

    def __call__(self, prompts: Sequence[str], completions: Sequence[str]) -> list[float]:
        return [
            self._on_true if self._predicate(c) else self._on_false
            for c in completions
        ]


@register
class PromptAdherenceReward:
    """
    Rewards completions that reference content from the prompt.

    Returns the fraction of prompt tokens (whitespace-split) that appear in
    the completion. Useful as a loose grounding signal to prevent the model
    from ignoring the prompt entirely.

    Args:
        min_overlap: Minimum fraction of prompt tokens that must appear in the
            completion to receive a non-zero reward.
    """

    def __init__(self, min_overlap: float = 0.1):
        self.min_overlap = min_overlap

    def __call__(self, prompts: Sequence[str], completions: Sequence[str]) -> list[float]:
        scores = []
        for prompt, completion in zip(prompts, completions):
            prompt_tokens = set(prompt.lower().split())
            if not prompt_tokens:
                scores.append(0.0)
                continue
            overlap = sum(1 for t in prompt_tokens if t in completion.lower())
            fraction = overlap / len(prompt_tokens)
            scores.append(fraction if fraction >= self.min_overlap else 0.0)
        return scores
