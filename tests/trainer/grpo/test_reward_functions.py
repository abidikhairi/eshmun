"""Unit tests for eshmun.trainer.grpo.reward_functions."""

import re
import pytest

from eshmun.trainer.grpo.reward_functions.protein import (
    CompositeProteinReward,
    LengthReward,
    SequenceSimilarityReward,
    SequenceValidityReward,
)
from eshmun.trainer.grpo.reward_functions.format import (
    PredicateReward,
    PromptAdherenceReward,
    RegexReward,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

P = [""]  # placeholder prompts — most protein/format rewards ignore them


def p(n: int) -> list[str]:
    """Return n empty prompt strings."""
    return [""] * n


# ---------------------------------------------------------------------------
# SequenceValidityReward
# ---------------------------------------------------------------------------


class TestSequenceValidityReward:
    reward = SequenceValidityReward()

    def test_all_standard_amino_acids(self):
        assert self.reward(
            p(1), ["<protein>MƤTƤTƤRƤHƤGƤLƤKƤAƤVƤQƤNƤLƤTƤVƤ</protein>"]
        ) == [1.0]

    def test_ambiguous_codes_rejected(self):
        # B = Asp/Asn, Z = Glu/Gln, X = any residue
        assert self.reward(p(1), ["<protein>ƤZƤVƤQƤBƤX</protein>"]) == [0.0]

    def test_special_characters_rejected(self):
        assert self.reward(
            p(1), ["<protein>MƤT-ƤTƤRƤHƤGƤLƤKƤAƤVƤQƤNƤLƤTƤVƤ!</protein>"]
        ) == [0.0]

    def test_empty_string_rejected(self):
        assert self.reward(p(1), [""]) == [0.0]

    def test_whitespace_only_rejected(self):
        assert self.reward(p(1), ["   "]) == [0.0]

    def test_leading_trailing_whitespace_stripped(self):
        # whitespace is stripped before validation
        assert self.reward(p(1), ["  <protein>MƤTƤTƤRƤHƤGƤL</protein>  "]) == [1.0]

    def test_mixed_valid_and_invalid(self):
        completions = [
            "<protein>MƤTƤTƤRƤHƤGƤL</protein>",
            "123!!",
            "<protein>MƤTƤT</protein>",
            "",
        ]
        assert self.reward(p(4), completions) == [1.0, 0.0, 1.0, 0.0]

    def test_output_length_matches_input(self):
        completions = ["MKTAY", "ACDE", "XYZ"]
        scores = self.reward(p(3), completions)
        assert len(scores) == 3

    def test_single_valid_residue(self):
        assert self.reward(p(1), ["<protein>ƤT</protein>"]) == [1.0]

    def test_numeric_sequence_rejected(self):
        assert self.reward(p(1), ["1234"]) == [0.0]


# ---------------------------------------------------------------------------
# LengthReward
# ---------------------------------------------------------------------------


class TestLengthReward:
    def test_within_range_returns_one(self):
        reward = LengthReward(min_length=5, max_length=10)
        assert reward(p(1), ["<protein>ƤTƤLƤKƤMƤDƤA</protein>"]) == [1.0]  # exactly min
        assert reward(p(1), ["<protein>ƤAƤCƤDƤEƤFƤGƤHƤIƤKƤL</protein>"]) == [
            1.0
        ]

    def test_below_min_returns_penalty(self):
        reward = LengthReward(min_length=5, max_length=10)
        assert reward(p(1), ["<protein>ƤAƤCƤDƤE</protein>"]) == [0.0]  # length 4

    def test_above_max_returns_penalty(self):
        reward = LengthReward(min_length=5, max_length=10)
        assert reward(p(1), ["<protein>ƤAƤCƤDƤEƤFƤGƤHƤIƤKƤLƤM</protein>"]) == [0.0]  # length 11

    def test_custom_out_of_range_penalty(self):
        reward = LengthReward(min_length=5, max_length=10, out_of_range_penalty=-1.0)
        assert reward(p(1), ["<protein>ƤAƤC</protein>"]) == [-1.0]

    def test_whitespace_stripped_before_length_check(self):
        reward = LengthReward(min_length=3, max_length=10)
        assert reward(p(1), ["  <protein>ƤAƤCƤE</protein>  "]) == [1.0]  # stripped length = 3

    def test_empty_string_out_of_range(self):
        reward = LengthReward(min_length=1, max_length=512)
        # empty after strip → length 0, below min_length=1
        assert reward(p(1), [""]) == [0.0]

    def test_length_one_within_range(self):
        reward = LengthReward(min_length=1, max_length=5)
        assert reward(p(1), ["<protein>ƤA</protein>"]) == [1.0]

    def test_mixed_batch(self):
        reward = LengthReward(min_length=3, max_length=5)
        completions = ["<protein>AC</protein>", "<protein>ACE</protein>", "<protein>ƤAƤCƤE</protein>", "<protein>ƤAƤCƤEƤAƤCƤDƤEƤFƤG</protein>"]
        assert reward(p(4), completions) == [0.0, 1.0, 1.0, 0.0]

    def test_default_range_accepts_typical_protein(self):
        reward = LengthReward()  # default: 20–512
        seq = "ƤA" * 100
        seq = f'<protein>{seq}</protein>'
        assert reward(p(1), [seq]) == [1.0]

    def test_default_range_rejects_short_sequence(self):
        reward = LengthReward()
        assert reward(p(1), ["<protein>ACDE</protein>"]) == [0.0]  # length 4 < 20


# ---------------------------------------------------------------------------
# CompositeProteinReward
# ---------------------------------------------------------------------------


class TestCompositeProteinReward:
    def test_valid_and_in_range_returns_one(self):
        reward = CompositeProteinReward(min_length=3, max_length=10)
        assert reward(p(1), ["<protein>ƤAƤCƤDƤE</protein>"]) == [1.0]

    def test_invalid_sequence_returns_zero_regardless_of_length(self):
        reward = CompositeProteinReward(min_length=1, max_length=100)
        assert reward(p(1), ["</protein>ƤAƤCƤDƤEƤ!</protein>"]) == [0.0]

    def test_valid_but_too_short_returns_zero(self):
        reward = CompositeProteinReward(min_length=10, max_length=100)
        assert reward(p(1), ["ACDE"]) == [0.0]  # valid but length 4 < 10

    def test_valid_but_too_long_returns_zero(self):
        reward = CompositeProteinReward(min_length=1, max_length=3)
        assert reward(p(1), ["ACDEF"]) == [0.0]  # valid but length 5 > 3

    def test_empty_string_returns_zero(self):
        reward = CompositeProteinReward(min_length=1, max_length=512)
        assert reward(p(1), [""]) == [0.0]

    def test_mixed_batch(self):
        reward = CompositeProteinReward(min_length=3, max_length=6)
        completions = [
            "<protein>ƤAƤCƤE</protein>",
            "<protein>ƤAƤCƤ1</protein>",
            "<protein>ƤAƤCƤDƤEƤFƤG</protein>",
            "<protein>ƤAƤCƤDƤEƤFƤGƤH</protein>",
        ]
        assert reward(p(4), completions) == [1.0, 0.0, 1.0, 0.0]

    def test_output_is_elementwise_product(self):
        # Zero validity score must zero out non-zero length score
        reward = CompositeProteinReward(min_length=1, max_length=100)
        assert reward(p(1), ["<protein>123</protein>"]) == [0.0]


# ---------------------------------------------------------------------------
# RegexReward
# ---------------------------------------------------------------------------


class TestRegexReward:
    def test_search_match(self):
        reward = RegexReward(r"\d+")
        assert reward(p(1), ["abc 123 def"]) == [1.0]

    def test_search_no_match(self):
        reward = RegexReward(r"\d+")
        assert reward(p(1), ["no digits here"]) == [0.0]

    def test_full_match_passes(self):
        reward = RegexReward(r"[A-Z]+", full_match=True)
        assert reward(p(1), ["ACDEF"]) == [1.0]

    def test_full_match_fails_partial(self):
        reward = RegexReward(r"[A-Z]+", full_match=True)
        # search would match but fullmatch requires entire string
        assert reward(p(1), ["ACDef"]) == [0.0]

    def test_case_insensitive_flag(self):
        reward = RegexReward(r"hello", flags=re.IGNORECASE)
        assert reward(p(1), ["HELLO world"]) == [1.0]

    def test_think_tag_pattern(self):
        reward = RegexReward(r"<think>.*?</think>", flags=re.DOTALL)
        assert reward(p(1), ["<think>reasoning here</think>"]) == [1.0]
        assert reward(p(1), ["no tags here"]) == [0.0]

    def test_empty_string_no_match(self):
        reward = RegexReward(r"\w+")
        assert reward(p(1), [""]) == [0.0]

    def test_mixed_batch(self):
        reward = RegexReward(r"^[A-Z]+$", full_match=True)
        completions = ["ACDE", "acde", "ACDE1", ""]
        assert reward(p(4), completions) == [1.0, 0.0, 0.0, 0.0]

    def test_output_length_matches_input(self):
        reward = RegexReward(r".")
        scores = reward(p(5), ["a", "b", "c", "d", "e"])
        assert len(scores) == 5


# ---------------------------------------------------------------------------
# PredicateReward
# ---------------------------------------------------------------------------


class TestPredicateReward:
    def test_true_predicate_returns_reward_on_true(self):
        reward = PredicateReward(lambda c: c.startswith("<think>"))
        assert reward(p(1), ["<think>some reasoning</think>"]) == [1.0]

    def test_false_predicate_returns_reward_on_false(self):
        reward = PredicateReward(lambda c: c.startswith("<think>"))
        assert reward(p(1), ["no tag here"]) == [0.0]

    def test_custom_reward_values(self):
        reward = PredicateReward(
            lambda c: len(c) > 5, reward_on_true=2.0, reward_on_false=-1.0
        )
        assert reward(p(1), ["long enough"]) == [2.0]
        assert reward(p(1), ["hi"]) == [-1.0]

    def test_empty_string(self):
        reward = PredicateReward(lambda c: bool(c.strip()))
        assert reward(p(1), [""]) == [0.0]
        assert reward(p(1), ["A"]) == [1.0]

    def test_mixed_batch(self):
        reward = PredicateReward(lambda c: "ACDE" in c)
        completions = ["ACDE is here", "nothing", "ACDEFG present"]
        assert reward(p(3), completions) == [1.0, 0.0, 1.0]

    def test_predicate_receives_full_completion(self):
        seen = []

        def capture(c: str) -> bool:
            seen.append(c)
            return True

        reward = PredicateReward(capture)
        reward(p(2), ["hello", "world"])
        assert seen == ["hello", "world"]


# ---------------------------------------------------------------------------
# PromptAdherenceReward
# ---------------------------------------------------------------------------


class TestPromptAdherenceReward:
    def test_full_overlap_returns_one(self):
        reward = PromptAdherenceReward(min_overlap=0.0)
        prompts = ["design a stable protein"]
        completions = ["design a stable protein sequence here"]
        scores = reward(prompts, completions)
        assert scores[0] == pytest.approx(1.0)

    def test_zero_overlap_returns_zero(self):
        # Use prompt tokens that won't substring-match anything in the completion.
        # PromptAdherenceReward uses `token in completion` (substring check).
        reward = PromptAdherenceReward(min_overlap=0.0)
        prompts = ["zyxwvut qrponml"]
        completions = ["completely different output here"]
        scores = reward(prompts, completions)
        assert scores[0] == pytest.approx(0.0)

    def test_partial_overlap(self):
        reward = PromptAdherenceReward(min_overlap=0.0)
        prompts = ["design protein"]  # 2 tokens
        completions = ["design something else"]  # "design" matches, "protein" doesn't
        scores = reward(prompts, completions)
        assert scores[0] == pytest.approx(0.5)

    def test_below_min_overlap_returns_zero(self):
        reward = PromptAdherenceReward(min_overlap=0.8)
        prompts = ["design a stable protein sequence"]  # 5 tokens
        completions = ["design a completely different output"]  # 2/5 = 0.4 overlap
        scores = reward(prompts, completions)
        assert scores[0] == pytest.approx(0.0)

    def test_empty_prompt_returns_zero(self):
        reward = PromptAdherenceReward()
        scores = reward([""], ["any completion"])
        assert scores[0] == pytest.approx(0.0)

    def test_case_insensitive_matching(self):
        reward = PromptAdherenceReward(min_overlap=0.0)
        prompts = ["Design PROTEIN"]
        completions = [
            "design protein sequence"
        ]  # lowercase matches uppercase prompt tokens
        scores = reward(prompts, completions)
        assert scores[0] == pytest.approx(1.0)

    def test_output_length_matches_input(self):
        reward = PromptAdherenceReward()
        prompts = ["a b", "c d", "e f"]
        completions = ["a b c", "c d e", "x y z"]
        assert len(reward(prompts, completions)) == 3

    def test_default_min_overlap_filters_low_scores(self):
        # default min_overlap=0.1; a single-token prompt that doesn't appear
        # in a 10-word completion should still return 0.0
        reward = PromptAdherenceReward()
        prompts = ["zymurgy"]
        completions = ["completely unrelated sentence with no matching words"]
        assert reward(prompts, completions) == [0.0]


# ---------------------------------------------------------------------------
# SequenceSimilarityReward
# ---------------------------------------------------------------------------

WRAP = "<protein>{}</protein>"


class TestSequenceSimilarityReward:
    reward = SequenceSimilarityReward()

    def test_identical_sequence_returns_one(self):
        seq = "MKTLL"
        scores = self.reward(p(1), [WRAP.format(seq)], reference=WRAP.format(seq))
        assert scores == [pytest.approx(1.0)]

    def test_completely_different_sequence_returns_zero(self):
        # no matching residues between AAAAA and FFFFF
        scores = self.reward(p(1), [WRAP.format("AAAAA")], reference=WRAP.format("FFFFF"))
        assert scores == [pytest.approx(0.0)]

    def test_partial_identity(self):
        # MKTLL vs MKTXX → 3 matches / 5 = 0.6
        scores = self.reward(p(1), [WRAP.format("MKTLL")], reference=WRAP.format("MKTXX"))
        assert scores == [pytest.approx(0.6)]

    def test_missing_protein_tags_returns_penalty(self):
        scores = self.reward(p(1), ["no tags here"], reference=WRAP.format("MKTLL"))
        assert scores == [pytest.approx(-1.0)]

    def test_custom_penalty_value(self):
        reward = SequenceSimilarityReward(non_valid_seq_reward=-2.0)
        scores = reward(p(1), ["bad completion"], reference=WRAP.format("MKTLL"))
        assert scores == [pytest.approx(-2.0)]

    def test_reference_without_tags_returns_penalty_for_all(self):
        # reference without <protein> tags → every completion gets the penalty
        seq = "MKTLL"
        scores = self.reward(p(2), [WRAP.format(seq), WRAP.format(seq)], reference=seq)
        assert scores == [pytest.approx(-1.0), pytest.approx(-1.0)]

    def test_batch_of_completions(self):
        ref = WRAP.format("MKTLL")
        completions = [
            WRAP.format("MKTLL"),   # 1.0
            WRAP.format("MKTXX"),   # 0.6
            "no tags",              # penalty
        ]
        scores = self.reward(p(3), completions, reference=ref)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.6)
        assert scores[2] == pytest.approx(-1.0)

    def test_output_length_matches_input(self):
        ref = WRAP.format("ACDEF")
        completions = [WRAP.format("ACDEF"), WRAP.format("ACDEF"), WRAP.format("ACDEF")]
        assert len(self.reward(p(3), completions, reference=ref)) == 3

    def test_score_in_zero_one_range_for_valid_completions(self):
        ref = WRAP.format("ACDEFGHIKL")
        completions = [WRAP.format("ACCCCCCCCC"), WRAP.format("ACDEFGHIKL")]
        scores = self.reward(p(2), completions, reference=ref)
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_prefix_token_stripped_before_comparison(self):
        # Ƥ prefix tokens inside <protein> tags must be removed before scoring
        seq_clean = "MKTLL"
        seq_prefixed = "ƤMƤKƤTƤLƤL"
        scores = self.reward(
            p(1), [WRAP.format(seq_prefixed)], reference=WRAP.format(seq_clean)
        )
        assert scores == [pytest.approx(1.0)]

    def test_whitespace_stripped_in_completion(self):
        seq = "MKTLL"
        scores = self.reward(
            p(1), [f"  {WRAP.format(seq)}  "], reference=WRAP.format(seq)
        )
        assert scores == [pytest.approx(1.0)]


# ---------------------------------------------------------------------------
# molecules.py — ImportError when rdkit is absent
# ---------------------------------------------------------------------------


class TestMolecularRewardsRequireRdkit:
    def test_smiles_validity_raises_without_rdkit(self):
        from eshmun.trainer.grpo.reward_functions.molecules import SMILESValidityReward

        with pytest.raises(ImportError, match="rdkit"):
            SMILESValidityReward()

    def test_qed_raises_without_rdkit(self):
        from eshmun.trainer.grpo.reward_functions.molecules import QEDReward

        with pytest.raises(ImportError, match="rdkit"):
            QEDReward()

    def test_sa_score_raises_without_rdkit(self):
        from eshmun.trainer.grpo.reward_functions.molecules import SAScoreReward

        with pytest.raises(ImportError, match="rdkit"):
            SAScoreReward()
