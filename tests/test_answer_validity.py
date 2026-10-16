"""tests/test_answer_validity.py — Unit tests for unified answer validity and accuracy metrics.

Verifies:
  1. Canonical strict accuracy rule: correct only if NOT truncated AND parsed AND correct.
  2. With-fallback accuracy rule: raw stored correct field.
  3. Truncation and parse rate calculations on unpooled benchmark rows.
  4. Integration tests against frozen MedPsy-4B MedQA and PubMedQA prediction files.
"""

import os
import unittest
from pathlib import Path

from src.answer_validity import (
    fallback_accuracy,
    fallback_correct_list,
    is_fallback_correct,
    is_parsed,
    is_strict_correct,
    is_truncated,
    parse_rate,
    strict_accuracy,
    strict_correct_list,
    truncation_rate,
)
from src.common import read_jsonl


class TestAnswerValidityRules(unittest.TestCase):
    """Test unit rules for strict vs with-fallback answer validity."""

    def test_strict_correct_positive(self):
        """A row is strictly correct if not truncated, parsed, and correct."""
        row = {"truncated": False, "parsed": True, "correct": True}
        self.assertTrue(is_strict_correct(row))

    def test_strict_correct_truncated_is_wrong(self):
        """A truncated response must ALWAYS count as wrong (0) under strict evaluation."""
        # Even if a fallback letter happens to match the gold key:
        row = {"truncated": True, "parsed": True, "correct": True}
        self.assertFalse(is_strict_correct(row))
        # With-fallback still counts it as correct:
        self.assertTrue(is_fallback_correct(row))

    def test_strict_correct_unparsed_is_wrong(self):
        """An unparsed response must ALWAYS count as wrong (0) under strict evaluation."""
        row = {"truncated": False, "parsed": False, "correct": True}
        self.assertFalse(is_strict_correct(row))
        self.assertTrue(is_fallback_correct(row))

    def test_strict_correct_both_truncated_and_unparsed(self):
        """A row both truncated and unparsed is wrong."""
        row = {"truncated": True, "parsed": False, "correct": True}
        self.assertFalse(is_strict_correct(row))
        self.assertTrue(is_fallback_correct(row))

    def test_strict_correct_incorrect_gold_mismatch(self):
        """A row with correct=False is strictly wrong."""
        row = {"truncated": False, "parsed": True, "correct": False}
        self.assertFalse(is_strict_correct(row))
        self.assertFalse(is_fallback_correct(row))

    def test_missing_keys_defaults(self):
        """Gracefully handle missing keys using standard canonical defaults."""
        # Standard un-annotated row: truncated defaults to False, parsed to True, correct to False
        row_empty = {}
        self.assertFalse(is_strict_correct(row_empty))
        self.assertFalse(is_fallback_correct(row_empty))
        self.assertFalse(is_truncated(row_empty))
        self.assertTrue(is_parsed(row_empty))

        row_just_correct = {"correct": True}
        self.assertTrue(is_strict_correct(row_just_correct))
        self.assertTrue(is_fallback_correct(row_just_correct))

    def test_aggregate_metrics(self):
        """Test accuracy and rate computations over synthetic cohorts."""
        rows = [
            {"truncated": False, "parsed": True, "correct": True},   # Strict=1, FB=1, Trunc=0, Parsed=1
            {"truncated": True, "parsed": False, "correct": True},   # Strict=0, FB=1, Trunc=1, Parsed=0
            {"truncated": True, "parsed": True, "correct": False},   # Strict=0, FB=0, Trunc=1, Parsed=1
            {"truncated": False, "parsed": True, "correct": False},  # Strict=0, FB=0, Trunc=0, Parsed=1
        ]
        self.assertEqual(strict_correct_list(rows), [1, 0, 0, 0])
        self.assertEqual(fallback_correct_list(rows), [1, 1, 0, 0])
        self.assertEqual(strict_accuracy(rows), 0.25)
        self.assertEqual(fallback_accuracy(rows), 0.50)
        self.assertEqual(truncation_rate(rows), 0.50)
        self.assertEqual(parse_rate(rows), 0.75)

    def test_empty_rows(self):
        """Empty lists return None without crashing."""
        self.assertIsNone(strict_accuracy([]))
        self.assertIsNone(fallback_accuracy([]))
        self.assertIsNone(truncation_rate([]))
        self.assertIsNone(parse_rate([]))


class TestMedPsy4BFrozenEvaluation(unittest.TestCase):
    """Integration test asserting exact verified empirical figures for MedPsy-4B."""

    def test_medpsy4b_medqa_figures(self):
        """Verify MedPsy-4B MedQA test baseline metrics: 83.2% strict, 87.6% fallback, 10.6% truncated."""
        candidates = [
            "outputs/kaggle_qa/medpsy_final/work/phase5/medpsy-4b.jsonl",
            "outputs/kaggle_qa/full/medpsy-4b/work/phase5/medpsy-4b.jsonl",
        ]
        target_path = next((p for p in candidates if os.path.exists(p)), None)
        if not target_path:
            self.skipTest(f"MedPsy-4B Phase 5 output not found in {candidates}")

        all_rows = read_jsonl(target_path)
        # Answerable MedQA test rows only (strictly unpooled from unanswerables)
        medqa_rows = [
            r for r in all_rows
            if r.get("split") == "test" and r.get("dataset") == "medqa" and not r.get("should_abstain")
        ]
        self.assertEqual(len(medqa_rows), 500)

        trunc_count = sum(1 for r in medqa_rows if is_truncated(r))
        strict_corr = sum(1 for r in medqa_rows if is_strict_correct(r))
        fb_corr = sum(1 for r in medqa_rows if is_fallback_correct(r))

        # Assert exact verified values
        self.assertEqual(trunc_count, 53, "MedQA test truncation count must be exactly 53/500 (10.6%)")
        self.assertAlmostEqual(truncation_rate(medqa_rows), 0.1060, places=4)

        self.assertEqual(strict_corr, 416, "MedQA test strict correct count must be exactly 416/500 (83.2%)")
        self.assertAlmostEqual(strict_accuracy(medqa_rows), 0.8320, places=4)

        self.assertEqual(fb_corr, 438, "MedQA test with-fallback correct count must be exactly 438/500 (87.6%)")
        self.assertAlmostEqual(fallback_accuracy(medqa_rows), 0.8760, places=4)

    def test_medpsy4b_pubmedqa_figures(self):
        """Verify MedPsy-4B PubMedQA test +Abstract metrics: 78.0% strict, 78.0% fallback, 0.2% truncated."""
        candidates = [
            "outputs/kaggle_qa/medpsy_final/work/phase10/medpsy-4b.jsonl",
            "outputs/kaggle_qa/context/work/phase10/medpsy-4b.jsonl",
        ]
        target_path = next((p for p in candidates if os.path.exists(p)), None)
        if not target_path:
            self.skipTest(f"MedPsy-4B Phase 10 output not found in {candidates}")

        all_rows = read_jsonl(target_path)
        pubmed_rows = [
            r for r in all_rows
            if r.get("split") == "test" and r.get("dataset") == "pubmedqa" and not r.get("should_abstain")
        ]
        self.assertEqual(len(pubmed_rows), 500)

        trunc_count = sum(1 for r in pubmed_rows if is_truncated(r))
        strict_corr = sum(1 for r in pubmed_rows if is_strict_correct(r))
        fb_corr = sum(1 for r in pubmed_rows if is_fallback_correct(r))

        self.assertEqual(trunc_count, 1, "PubMedQA test truncation count must be exactly 1/500 (0.2%)")
        self.assertEqual(strict_corr, 390, "PubMedQA test strict correct count must be exactly 390/500 (78.0%)")
        self.assertEqual(fb_corr, 390, "PubMedQA test with-fallback correct count must be exactly 390/500 (78.0%)")


if __name__ == "__main__":
    unittest.main()
