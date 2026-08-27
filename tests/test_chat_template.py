import os
import random
import sys
import unittest

from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(REPO_ROOT, "scripts", "kothar", "chat_template.jinja")
STUDENT_ID = "khairi/Kothar-student-seed-409M"

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "data", "thinking"))
# pyrefly: ignore [missing-import]
from reasoning import Triple, build_reasoning_block, build_response_text  # noqa: E402


def _load_tokenizer():
    with open(TEMPLATE_PATH) as f:
        template = f.read()
    tok = AutoTokenizer.from_pretrained(STUDENT_ID, local_files_only=True)
    tok.chat_template = template
    return tok


class KotharChatTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tok = _load_tokenizer()
        except Exception as e:  # not cached locally / no network
            raise unittest.SkipTest(f"tokenizer '{STUDENT_ID}' unavailable: {e}")

        cls.think_id = cls.tok.convert_tokens_to_ids("<think>")
        cls.unthink_id = cls.tok.convert_tokens_to_ids("</think>")
        cls.eos_id = cls.tok.eos_token_id

        rng = random.Random(0)
        context = [
            Triple(relation="has_function", value="ATP binding"),
            Triple(relation="located_in", value="cytoplasm"),
        ]
        cls.reasoning = build_reasoning_block(context, rng)
        cls.answer = build_response_text("member_of", ["Cytochrome P450 family"])
        cls.instruction = "What family does this protein PPPPPPPPPP belong to?"
        cls.assistant_content = cls.reasoning + "\n" + cls.answer

    def test_think_tokens_are_atomic_special_tokens(self):
        self.assertNotIn(self.think_id, (None, self.tok.unk_token_id))
        self.assertNotIn(self.unthink_id, (None, self.tok.unk_token_id))
        self.assertIn("<think>", self.tok.all_special_tokens)
        self.assertIn("</think>", self.tok.all_special_tokens)

    def test_user_only_prompt_ends_after_instruction(self):
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": self.instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertEqual(rendered, self.instruction + "\n")

    def test_full_conversation_matches_expected_concatenation(self):
        rendered = self.tok.apply_chat_template(
            [
                {"role": "user", "content": self.instruction},
                {"role": "assistant", "content": self.assistant_content},
            ],
            tokenize=False,
        )
        expected = self.instruction + "\n" + self.assistant_content + self.tok.eos_token
        self.assertEqual(rendered, expected)

    def test_tokenized_conversation_has_atomic_think_tokens_and_single_eos(self):
        text = self.tok.apply_chat_template(
            [
                {"role": "user", "content": self.instruction},
                {"role": "assistant", "content": self.assistant_content},
            ],
            tokenize=False,
        )
        ids = self.tok(text, add_special_tokens=False)["input_ids"]

        self.assertIn(self.think_id, ids)
        self.assertIn(self.unthink_id, ids)
        self.assertEqual(ids[-1], self.eos_id)
        self.assertEqual(ids.count(self.eos_id), 1)

    def test_round_trip_decode_matches_rendered_text(self):
        text = self.tok.apply_chat_template(
            [
                {"role": "user", "content": self.instruction},
                {"role": "assistant", "content": self.assistant_content},
            ],
            tokenize=False,
        )
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        decoded = self.tok.decode(ids, skip_special_tokens=False)
        self.assertEqual(decoded, text)

    def test_system_message_is_prepended_with_blank_line(self):
        rendered = self.tok.apply_chat_template(
            [
                {"role": "system", "content": "You are a protein annotation assistant."},
                {"role": "user", "content": self.instruction},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        expected = "You are a protein annotation assistant.\n\n" + self.instruction + "\n"
        self.assertEqual(rendered, expected)

    def test_invalid_role_raises(self):
        with self.assertRaises(Exception):
            self.tok.apply_chat_template(
                [{"role": "tool", "content": "x"}], tokenize=False
            )

    def test_multi_turn_conversation_renders_each_turn_in_order(self):
        rendered = self.tok.apply_chat_template(
            [
                {"role": "user", "content": "first question PPPP"},
                {"role": "assistant", "content": "<think>\nfoo\n</think>\nfirst answer"},
                {"role": "user", "content": "second question QQQQ"},
                {"role": "assistant", "content": "<think>\nbar\n</think>\nsecond answer"},
            ],
            tokenize=False,
        )
        expected = (
            "first question PPPP\n"
            "<think>\nfoo\n</think>\nfirst answer" + self.tok.eos_token +
            "second question QQQQ\n"
            "<think>\nbar\n</think>\nsecond answer" + self.tok.eos_token
        )
        self.assertEqual(rendered, expected)


if __name__ == "__main__":
    unittest.main()
