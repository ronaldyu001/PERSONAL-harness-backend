"""Tests for the Postgres conversation adapter."""

from __future__ import annotations

import unittest

from infrastructure.conversation.adapter_postgres import (
    UNTITLED_CONVERSATION,
    truncate_title,
)


class TitleDerivationTests(unittest.TestCase):
    def test_short_message_is_kept_whole(self) -> None:
        self.assertEqual(truncate_title("Explain checkpointing"), "Explain checkpointing")

    def test_long_message_is_cut_on_a_word_boundary(self) -> None:
        title = truncate_title(
            "Explain how LangGraph checkpointers persist conversation state"
        )
        self.assertEqual(title, "Explain how LangGraph checkpointers persist…")
        self.assertLessEqual(len(title.rstrip("…")), 44)

    def test_single_long_word_is_cut_at_the_limit(self) -> None:
        title = truncate_title("x" * 60)
        self.assertEqual(title, f"{'x' * 44}…")

    def test_blank_message_falls_back(self) -> None:
        self.assertEqual(truncate_title("   "), UNTITLED_CONVERSATION)

    def test_whitespace_is_collapsed_to_one_line(self) -> None:
        self.assertEqual(truncate_title("Explain\n  checkpointing"), "Explain checkpointing")


if __name__ == "__main__":
    unittest.main()
