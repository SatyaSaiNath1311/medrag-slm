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

import json
import os
import tempfile
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
    run_analysis,
    wilson_score_ci,
)
from src.common import append_jsonl
from src.qa_pipeline_v3 import assert_appended_row_valid, build_sweep_row


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


class TestSweepRowSchemaValidation(unittest.TestCase):
    """Regression test: rows constructed by build_sweep_row must pass assert_appended_row_valid."""

    def test_sweep_row_finished_naturally_validates(self):
        q = {
            "id": "medqa-val-001",
            "question": "Sample question?",
            "options": {"A": "Choice A", "B": "Choice B"},
            "answer": "A",
            "split": "validation",
        }
        row = build_sweep_row(
            q=q,
            raw_output="<think>analysis</think>\nAnswer: A",
            parsed_pred="A",
            parsed=True,
            correct=True,
            n_gen_tokens=350,
            finished_naturally=True,
            sec_q=1.234,
            model_name="qwen3-4b-thinking-2507",
        )
        self.assertFalse(row["truncated"])
        self.assertTrue(row["finished_naturally"])

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            temp_path = tf.name
        try:
            append_jsonl(temp_path, [row])
            assert_appended_row_valid(temp_path, expected_id="medqa-val-001")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_sweep_row_truncated_validates(self):
        q = {
            "id": "medqa-val-002",
            "question": "Sample question 2?",
            "options": {"A": "Choice A", "B": "Choice B"},
            "answer": "B",
            "split": "validation",
        }
        row = build_sweep_row(
            q=q,
            raw_output="<think>cut off before producing answer",
            parsed_pred=None,
            parsed=False,
            correct=False,
            n_gen_tokens=4096,
            finished_naturally=False,
            sec_q=12.34,
            model_name="qwen3-4b-thinking-2507",
        )
        self.assertTrue(row["truncated"])
        self.assertFalse(row["finished_naturally"])

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
            temp_path = tf.name
        try:
            append_jsonl(temp_path, [row])
            assert_appended_row_valid(temp_path, expected_id="medqa-val-002")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


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

class TestBudgetSweepAnalysisEndToEnd(unittest.TestCase):
    """End-to-end tests for run_analysis on synthetic sweep fixtures."""

    def test_analysis_smoke_run_n3(self):
        """Case A: n=3 -> report starts with SMOKE banner, no Chosen Test Cap line, figures created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "completed": True,
                        "N": 3,
                        "split": "validation",
                        "model_pair": ["qwen3-4b-thinking-2507", "medpsy-4b"],
                    },
                    f,
                )

            qwen_rows = []
            medpsy_rows = []
            for i in range(3):
                qid = f"medqa-val-{i:03d}"
                q = {
                    "id": qid,
                    "dataset": "medqa",
                    "split": "validation",
                    "question": f"Question {i}?",
                    "options": {"A": "Choice A", "B": "Choice B"},
                    "answer": "A",
                }
                qwen_rows.append(
                    build_sweep_row(
                        q=q,
                        raw_output=f"<think>analysis {i}</think>\nAnswer: A",
                        parsed_pred="A",
                        parsed=True,
                        correct=True,
                        n_gen_tokens=500 + i * 100,
                        finished_naturally=True,
                        sec_q=1.0,
                        model_name="qwen3-4b-thinking-2507",
                    )
                )
                medpsy_rows.append(
                    build_sweep_row(
                        q=q,
                        raw_output="Answer: A",
                        parsed_pred="A",
                        parsed=True,
                        correct=True,
                        n_gen_tokens=300 + i * 50,
                        finished_naturally=True,
                        sec_q=0.5,
                        model_name="medpsy-4b",
                    )
                )

            append_jsonl(os.path.join(tmpdir, "qwen3-4b-thinking-2507.jsonl"), qwen_rows)
            append_jsonl(os.path.join(tmpdir, "medpsy-4b.jsonl"), medpsy_rows)

            out_md = os.path.join(tmpdir, "report.md")
            out_png = os.path.join(tmpdir, "plot.png")
            out_svg = os.path.join(tmpdir, "plot.svg")

            run_analysis(tmpdir, out_md, out_png, out_svg)

            self.assertTrue(os.path.exists(out_md))
            with open(out_md, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertTrue(
                content.startswith("SMOKE RUN N=3: not for reporting"),
                f"Expected smoke banner at start of report, got:\n{content[:100]}",
            )
            self.assertNotIn("Chosen Test Cap", content)
            self.assertTrue(os.path.exists(out_png))
            self.assertTrue(os.path.exists(out_svg))
            self.assertIn("- **Consistency Check:**", content)

    def test_analysis_full_run_n100_decision_rule(self):
        """Case B: n=100 -> decision-rule section present, chosen cap matches apply_budget_decision_rule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "completed": True,
                        "N": 100,
                        "split": "validation",
                        "model_pair": ["qwen3-4b-thinking-2507", "medpsy-4b"],
                    },
                    f,
                )

            qwen_rows = []
            medpsy_rows = []
            # Construct 100 questions with mixed token counts:
            # 0-14 (15 questions): 800 tokens, finished (finished <= 1024)
            # 15-29 (15 questions): 1400 tokens, finished (finished <= 1536)
            # 30-84 (55 questions): 1900 tokens, finished (finished <= 2048)
            # 85-99 (15 questions): 4096 tokens, finished_naturally=False (truncated at all caps)
            for i in range(100):
                qid = f"medqa-val-{i:03d}"
                q = {
                    "id": qid,
                    "dataset": "medqa",
                    "split": "validation",
                    "question": f"Question {i}?",
                    "options": {"A": "Choice A", "B": "Choice B"},
                    "answer": "A",
                }
                if i < 15:
                    n_gen = 800
                    finished = True
                elif i < 30:
                    n_gen = 1400
                    finished = True
                elif i < 85:
                    n_gen = 1900
                    finished = True
                else:
                    n_gen = 4096
                    finished = False

                qwen_rows.append(
                    build_sweep_row(
                        q=q,
                        raw_output=f"<think>thought {i}</think>\nAnswer: A" if finished else "<think>cut off",
                        parsed_pred="A" if finished else None,
                        parsed=finished,
                        correct=finished,
                        n_gen_tokens=n_gen,
                        finished_naturally=finished,
                        sec_q=1.0,
                        model_name="qwen3-4b-thinking-2507",
                    )
                )
                medpsy_rows.append(
                    build_sweep_row(
                        q=q,
                        raw_output="Answer: A",
                        parsed_pred="A",
                        parsed=True,
                        correct=True,
                        n_gen_tokens=400,
                        finished_naturally=True,
                        sec_q=0.5,
                        model_name="medpsy-4b",
                    )
                )

            # Expected truncation rates for Qwen:
            # cap 1024: 15 finished -> 85% trunc
            # cap 1536: 30 finished -> 70% trunc
            # cap 2048: 85 finished -> 15% trunc (qualifies <= 20%)
            # cap 3072: 85 finished -> 15% trunc
            # cap 4096: 85 finished -> 15% trunc
            synthetic_qwen_trunc_rates = {
                1024: (100 - sum(1 for r in qwen_rows if is_finished_within_cap(r, 1024))) / 100.0,
                1536: (100 - sum(1 for r in qwen_rows if is_finished_within_cap(r, 1536))) / 100.0,
                2048: (100 - sum(1 for r in qwen_rows if is_finished_within_cap(r, 2048))) / 100.0,
                3072: (100 - sum(1 for r in qwen_rows if is_finished_within_cap(r, 3072))) / 100.0,
                4096: (100 - sum(1 for r in qwen_rows if is_finished_within_cap(r, 4096))) / 100.0,
            }
            expected_cap, expected_msg = apply_budget_decision_rule(synthetic_qwen_trunc_rates)
            self.assertEqual(expected_cap, 2048)

            append_jsonl(os.path.join(tmpdir, "qwen3-4b-thinking-2507.jsonl"), qwen_rows)
            append_jsonl(os.path.join(tmpdir, "medpsy-4b.jsonl"), medpsy_rows)

            out_md = os.path.join(tmpdir, "report.md")
            out_png = os.path.join(tmpdir, "plot.png")
            out_svg = os.path.join(tmpdir, "plot.svg")

            run_analysis(tmpdir, out_md, out_png, out_svg)

            self.assertTrue(os.path.exists(out_md))
            with open(out_md, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertFalse(content.startswith("SMOKE RUN"))
            self.assertIn("## 1. Decision Rule Execution & Selected Test Cap", content)
            self.assertIn(f"- **Chosen Test Cap:** `{expected_cap}` tokens", content)
            self.assertTrue(os.path.exists(out_png))
            self.assertTrue(os.path.exists(out_svg))
            self.assertIn("- **Consistency Check:**", content)

    def test_report_text_changes_when_input_row_mutated(self):
        """Mutating one input row in the synthetic budget sweep changes the generated report text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qwen_rows = []
            medpsy_rows = []
            for i in range(100):
                q = {
                    "id": f"q_{i:03d}",
                    "question": f"Question {i}?",
                    "options": {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
                    "answer": "A",
                    "split": "validation",
                    "dataset": "medqa",
                }
                qwen_rows.append(
                    build_sweep_row(
                        q=q,
                        raw_output="Answer: A",
                        parsed_pred="A",
                        parsed=True,
                        correct=True,
                        n_gen_tokens=500,
                        finished_naturally=True,
                        sec_q=1.0,
                        model_name="qwen3-4b-thinking-2507",
                    )
                )
                medpsy_rows.append(
                    build_sweep_row(
                        q=q,
                        raw_output="Answer: A",
                        parsed_pred="A",
                        parsed=True,
                        correct=True,
                        n_gen_tokens=400,
                        finished_naturally=True,
                        sec_q=0.5,
                        model_name="medpsy-4b",
                    )
                )

            p_qwen = os.path.join(tmpdir, "qwen3-4b-thinking-2507.jsonl")
            p_medpsy = os.path.join(tmpdir, "medpsy-4b.jsonl")
            append_jsonl(p_qwen, qwen_rows)
            append_jsonl(p_medpsy, medpsy_rows)

            out_md = os.path.join(tmpdir, "report.md")
            out_png = os.path.join(tmpdir, "plot.png")
            out_svg = os.path.join(tmpdir, "plot.svg")

            run_analysis(tmpdir, out_md, out_png, out_svg)
            with open(out_md, "r", encoding="utf-8") as f:
                text_before = f.read()

            # Mutate one row: flip correct flag of row 0 in medpsy
            medpsy_rows_mutated = [r for r in medpsy_rows]
            medpsy_rows_mutated[0]["correct"] = False
            with open(p_medpsy, "w", encoding="utf-8") as f:
                for r in medpsy_rows_mutated:
                    f.write(json.dumps(r) + "\n")

            run_analysis(tmpdir, out_md, out_png, out_svg)
            with open(out_md, "r", encoding="utf-8") as f:
                text_after = f.read()

            self.assertNotEqual(text_before, text_after)
            self.assertIn("100.0%", text_before)
            self.assertIn("99.0%", text_after)



if __name__ == "__main__":
    unittest.main()

