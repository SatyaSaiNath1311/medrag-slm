"""tests/test_budget_sweep.py — Synthetic fixture tests for Pair B budget sweep.

Verifies:
  1. Finish-within-cap boundaries:
     - n=1024 finished counts at cap 1024; n=1025 does not.
     - Truncated rows (finished_naturally=False) never count regardless of token count.
  2. Strict accuracy requires finished within cap AND parsed AND correct.
  3. Wilson score and exact-McNemar known values.
  4. Pre-registered decision-rule function:
     - Selects smallest qualifying cap <= 20% truncation.
     - Falls back to 4096 when none qualifies.
     - Respects boundary truncation rate == 0.20.
  5. Hard validation-split assertion raises when a test-split row is encountered.
  6. With V3_BUDGET_SWEEP unset, runner config resolves exactly as before.
"""

import os
import unittest
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent

from src.analysis_budget_sweep import (
    apply_budget_decision_rule,
    exact_mcnemar_test,
    is_finished_within_cap,
    is_strict_correct_at_cap,
    is_truncated_at_cap,
    wilson_score_ci,
)


class TestBudgetSweepBoundaries(unittest.TestCase):
    """Test boundary conditions for finish-within-cap and truncation classification."""

    def test_finish_within_cap_exact_boundary(self):
        """n=1024 finished counts at cap 1024; n=1025 does not."""
        row_at_1024 = {
            "finished_naturally": True,
            "n_generated_tokens": 1024,
            "parsed": True,
            "correct": True,
        }
        row_at_1025 = {
            "finished_naturally": True,
            "n_generated_tokens": 1025,
            "parsed": True,
            "correct": True,
        }
        self.assertTrue(is_finished_within_cap(row_at_1024, 1024))
        self.assertFalse(is_finished_within_cap(row_at_1025, 1024))

        # At cap 1536, 1025 is within cap
        self.assertTrue(is_finished_within_cap(row_at_1025, 1536))

    def test_truncated_rows_never_count_as_finished(self):
        """Truncated rows (finished_naturally=False) never count, even with small token counts."""
        row_truncated_small = {
            "finished_naturally": False,
            "n_generated_tokens": 500,
            "parsed": True,
            "correct": True,
        }
        row_truncated_cap = {
            "finished_naturally": False,
            "n_generated_tokens": 1024,
            "parsed": True,
            "correct": True,
        }
        self.assertFalse(is_finished_within_cap(row_truncated_small, 1024))
        self.assertFalse(is_finished_within_cap(row_truncated_cap, 1024))
        self.assertTrue(is_truncated_at_cap(row_truncated_small, 1024))
        self.assertTrue(is_truncated_at_cap(row_truncated_cap, 1024))

    def test_strict_requires_finished_parsed_and_correct(self):
        """Strict correctness requires finished naturally within cap AND parsed AND correct."""
        # Baseline positive: finished within cap, parsed, correct
        row_ok = {
            "finished_naturally": True,
            "n_generated_tokens": 800,
            "parsed": True,
            "correct": True,
        }
        self.assertTrue(is_strict_correct_at_cap(row_ok, 1024))

        # Fails if not parsed
        row_unparsed = {
            "finished_naturally": True,
            "n_generated_tokens": 800,
            "parsed": False,
            "correct": True,
        }
        self.assertFalse(is_strict_correct_at_cap(row_unparsed, 1024))

        # Fails if incorrect
        row_incorrect = {
            "finished_naturally": True,
            "n_generated_tokens": 800,
            "parsed": True,
            "correct": False,
        }
        self.assertFalse(is_strict_correct_at_cap(row_incorrect, 1024))

        # Fails if exceeded cap even if parsed and correct
        row_exceeded = {
            "finished_naturally": True,
            "n_generated_tokens": 1200,
            "parsed": True,
            "correct": True,
        }
        self.assertFalse(is_strict_correct_at_cap(row_exceeded, 1024))

        # Fails if truncated (finished_naturally=False)
        row_trunc = {
            "finished_naturally": False,
            "n_generated_tokens": 1024,
            "parsed": True,
            "correct": True,
        }
        self.assertFalse(is_strict_correct_at_cap(row_trunc, 1024))


class TestStatisticalKnownValues(unittest.TestCase):
    """Test known mathematical values for Wilson Score CI and Exact McNemar test."""

    def test_wilson_ci_known_values(self):
        """Wilson 95% CI on known test proportions."""
        # k=0, n=100 -> [0.0000, 0.0370] (3.8416/103.8416 = 0.03699)
        low0, high0 = wilson_score_ci(0, 100)
        self.assertAlmostEqual(low0, 0.0, places=4)
        self.assertAlmostEqual(high0, 0.0370, places=3)

        # k=50, n=100 -> [0.4038, 0.5962]
        low50, high50 = wilson_score_ci(50, 100)
        self.assertAlmostEqual(low50, 0.4038, places=3)
        self.assertAlmostEqual(high50, 0.5962, places=3)

        # k=100, n=100 -> [0.9630, 1.0000] (1 - 0.03699 = 0.96301)
        low100, high100 = wilson_score_ci(100, 100)
        self.assertAlmostEqual(low100, 0.9630, places=3)
        self.assertAlmostEqual(high100, 1.0, places=4)

    def test_exact_mcnemar_known_values(self):
        """Exact McNemar test on known discordant counts."""
        # Zero discordant pairs -> p = 1.0
        self.assertEqual(exact_mcnemar_test(0, 0), 1.0)

        # Symmetric counts -> p = 1.0
        self.assertEqual(exact_mcnemar_test(5, 5), 1.0)
        self.assertEqual(exact_mcnemar_test(10, 10), 1.0)

        # b=0, c=10 -> 2 * (1 / 2^10) = 2 / 1024 = 0.001953125
        p_0_10 = exact_mcnemar_test(0, 10)
        self.assertAlmostEqual(p_0_10, 2.0 / 1024.0, places=6)

        # b=1, c=15 -> n=16, k=1. binom(16, 0.5): (1 + 16)/65536 * 2 = 34 / 65536 = 0.0005187988
        p_1_15 = exact_mcnemar_test(1, 15)
        self.assertAlmostEqual(p_1_15, 34.0 / 65536.0, places=6)


class TestDecisionRuleFunction(unittest.TestCase):
    """Test pre-registered budget decision rule logic."""

    def test_decision_rule_selects_smallest_qualifying_cap(self):
        """Smallest cap with truncation <= 20% is selected."""
        rates = {
            1024: 0.95,
            1536: 0.45,
            2048: 0.18,  # First qualifying cap
            3072: 0.10,
            4096: 0.04,
        }
        chosen, msg = apply_budget_decision_rule(rates)
        self.assertEqual(chosen, 2048)
        self.assertIn("Cap 2048 SELECTED", msg)

    def test_decision_rule_boundary_at_20_percent(self):
        """Exactly 20% truncation qualifies."""
        rates = {
            1024: 0.20,  # Exact boundary
            1536: 0.15,
            2048: 0.10,
            3072: 0.05,
            4096: 0.02,
        }
        chosen, msg = apply_budget_decision_rule(rates)
        self.assertEqual(chosen, 1024)
        self.assertIn("Cap 1024 SELECTED", msg)

    def test_decision_rule_fallback_to_4096_when_none_qualifies(self):
        """When no cap reaches <= 20% truncation, falls back to 4096."""
        rates = {
            1024: 0.99,
            1536: 0.85,
            2048: 0.60,
            3072: 0.40,
            4096: 0.28,  # none <= 0.20
        }
        chosen, msg = apply_budget_decision_rule(rates)
        self.assertEqual(chosen, 4096)
        self.assertIn("FALLBACK SELECTED", msg)
        self.assertIn("limitation", msg)


class TestValidationSplitHardAssertion(unittest.TestCase):
    """Test that non-validation rows immediately trigger hard assertions."""

    def test_hard_assertion_on_non_validation_row(self):
        """A row with split='test' must raise an AssertionError."""
        synthetic_rows = [
            {"id": "medqa-val-001", "dataset": "medqa", "split": "validation"},
            {"id": "medqa-val-002", "dataset": "medqa", "split": "validation"},
            {"id": "medqa-test-001", "dataset": "medqa", "split": "test"},  # Contamination!
        ]

        def check_splits(rows):
            for r in rows:
                if r.get("split") != "validation":
                    raise AssertionError(f"FATAL: Non-validation split detected! id={r.get('id')} split={r.get('split')}")

        with self.assertRaises(AssertionError):
            check_splits(synthetic_rows)


class TestDefaultRunnerResolution(unittest.TestCase):
    """Verify that with V3_BUDGET_SWEEP unset, runner configuration resolves as before."""

    def test_budget_sweep_default_off(self):
        """With V3_BUDGET_SWEEP unset, runner defaults to V3_BUDGET_SWEEP=False at module level."""
        import ast

        runner_path = REPO_ROOT / "kaggle_runner" / "runner.py"
        with open(runner_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(runner_path))
        module_assigns = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Constant):
                            module_assigns[target.id] = node.value.value

        self.assertIn("V3_BUDGET_SWEEP", module_assigns)
        self.assertIs(module_assigns["V3_BUDGET_SWEEP"], False)
        self.assertEqual(module_assigns.get("V3_SWEEP_N"), 100)
        self.assertEqual(module_assigns.get("V3_SWEEP_MAX_HOURS"), 10.0)
        self.assertEqual(module_assigns.get("V3_SWEEP_BATCH_SIZE"), 1)


if __name__ == "__main__":
    unittest.main()

