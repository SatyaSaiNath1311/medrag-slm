"""tests/test_gemma_prompt_sens.py — Synthetic fixture tests for Gemma prompt sensitivity.

Verifies:
  1. build_prompt_sens_row output passes assert_appended_row_valid (both finished and truncated cases).
  2. Pre-registered stability verdict function:
     - Same sign + range <= 10 pp -> STABLE.
     - Sign flip -> PROMPT-SENSITIVE.
     - Range 10.1 pp -> PROMPT-SENSITIVE.
     - Exact boundary 10.0 pp -> STABLE.
  3. V1 template equals the existing builder's output (baseline_prompt) byte for byte.
  4. Hard assertion raises on a test-split row.
  5. The runner source (read as text with ast, never imported) has:
     - V3_GEMMA_PROMPT_SENS = False
     - V3_GEMMA_SENS_N = 200
     - V3_GEMMA_SENS_MAX_HOURS = 3.0
  6. End-to-end run_analysis on synthetic data:
     - N=3 (smoke run): report starts with 'SMOKE RUN N=3: not for reporting', stability verdict suppressed, figures created.
     - N=200 (full run): stability verdict present, figures created.
"""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from src.analysis_gemma_prompt_sensitivity import (
    check_format_confound,
    evaluate_gemma_stability,
    exact_mcnemar_test,
    format_p_val,
    holm_bonferroni,
    load_and_validate_gemma_sweeps,
    paired_bootstrap_ci,
    run_analysis,
    wilson_score_ci,
)
from src.common import append_jsonl
from src.llm import baseline_prompt
from src.qa_pipeline_v3 import (
    PROMPT_TEMPLATE_HASHES,
    assert_appended_row_valid,
    build_prompt_sens_row,
    prompt_v1,
    prompt_v2,
    prompt_v3,
)


class TestGemmaPromptSensRowValidity(unittest.TestCase):
    """Test that build_prompt_sens_row generates rows satisfying assert_appended_row_valid."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "test_row.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_row_finished_passes_assertion(self):
        """Finished row has truncated=False and passes assert_appended_row_valid."""
        q = {
            "id": "medqa_val_001",
            "question": "A 45-year-old male presents with acute chest pain...",
            "options": {"A": "Aspirin", "B": "Penicillin", "C": "Metformin", "D": "Warfarin"},
            "answer": "A",
            "split": "validation",
            "dataset": "medqa",
        }
        row = build_prompt_sens_row(
            q=q,
            variant="V1",
            raw_output="Answer: A",
            parsed_pred="A",
            parsed=True,
            correct=True,
            n_gen_tokens=4,
            finished_naturally=True,
            sec_q=0.25,
            model_name="gemma3-4b",
        )
        self.assertEqual(row["truncated"], False)
        self.assertEqual(row["finished_naturally"], True)
        self.assertEqual(row["variant"], "V1")
        self.assertEqual(row["condition"], "C1")

        append_jsonl(self.temp_file, [row])
        assert_appended_row_valid(self.temp_file, expected_id="medqa_val_001")

    def test_build_row_truncated_passes_assertion(self):
        """Truncated row has truncated=True and passes assert_appended_row_valid."""
        q = {
            "id": "medqa_val_002",
            "question": "A 60-year-old female presents with shortness of breath...",
            "options": {"A": "Asthma", "B": "COPD", "C": "Pneumonia", "D": "Embolism"},
            "answer": "B",
            "split": "validation",
            "dataset": "medqa",
        }
        row = build_prompt_sens_row(
            q=q,
            variant="V2",
            raw_output="The differential diagnosis is extensive and includes",
            parsed_pred=None,
            parsed=False,
            correct=False,
            n_gen_tokens=16,
            finished_naturally=False,
            sec_q=0.85,
            model_name="medgemma-1.5-4b-it",
        )
        self.assertEqual(row["truncated"], True)
        self.assertEqual(row["finished_naturally"], False)
        self.assertEqual(row["variant"], "V2")

        append_jsonl(self.temp_file, [row])
        assert_appended_row_valid(self.temp_file, expected_id="medqa_val_002")


class TestGemmaStabilityCriterion(unittest.TestCase):
    """Test the pre-registered stability verdict function under all boundary cases."""

    def test_stable_same_sign_within_10_pts(self):
        """Same negative sign and spread <= 10 pp -> STABLE."""
        # Negative gaps (favoring medgemma)
        is_stable, spread, msg = evaluate_gemma_stability(-0.06, -0.04, -0.08)
        self.assertTrue(is_stable)
        self.assertAlmostEqual(spread, 0.04, places=4)
        self.assertIn("STABLE", msg)
        self.assertIn("negative", msg)

        # Positive gaps (favoring gemma3)
        is_stable_pos, spread_pos, msg_pos = evaluate_gemma_stability(0.05, 0.08, 0.02)
        self.assertTrue(is_stable_pos)
        self.assertAlmostEqual(spread_pos, 0.06, places=4)
        self.assertIn("STABLE", msg_pos)
        self.assertIn("positive", msg_pos)

    def test_prompt_sensitive_sign_flip(self):
        """Sign flip across variants -> PROMPT-SENSITIVE regardless of small spread."""
        is_stable, spread, msg = evaluate_gemma_stability(0.02, -0.01, 0.01)
        self.assertFalse(is_stable)
        self.assertIn("PROMPT-SENSITIVE", msg)
        self.assertIn("sign flip", msg)

    def test_prompt_sensitive_range_10_point_1(self):
        """Range of 10.1 percentage points (0.101) -> PROMPT-SENSITIVE."""
        is_stable, spread, msg = evaluate_gemma_stability(-0.01, -0.111, -0.05)
        self.assertFalse(is_stable)
        self.assertAlmostEqual(spread, 0.101, places=4)
        self.assertIn("PROMPT-SENSITIVE", msg)
        self.assertIn("exceeds", msg)

    def test_stable_exact_boundary_10_pts(self):
        """Range of exactly 10.0 percentage points (0.100) -> STABLE."""
        is_stable, spread, msg = evaluate_gemma_stability(-0.02, -0.12, -0.07)
        self.assertTrue(is_stable)
        self.assertAlmostEqual(spread, 0.10, places=4)
        self.assertIn("STABLE", msg)


class TestGemmaPromptTemplates(unittest.TestCase):
    """Test prompt templates and byte-identical identity with baseline_prompt."""

    def test_v1_equals_baseline_prompt_byte_for_byte(self):
        """V1 template output is byte-identical to baseline_prompt(q)."""
        sample_q = {
            "id": "medqa_test_sample",
            "question": "What is the first-line medication for uncomplicated hypertension in non-black patients?",
            "options": {
                "A": "ACE inhibitor or ARB",
                "B": "Beta-blocker",
                "C": "Loop diuretic",
                "D": "Alpha-blocker",
            },
            "answer": "A",
        }
        v1_out = prompt_v1(sample_q)
        baseline_out = baseline_prompt(sample_q)
        self.assertEqual(v1_out, baseline_out)
        self.assertEqual(v1_out.encode("utf-8"), baseline_out.encode("utf-8"))

    def test_v2_and_v3_distinct_and_deterministic(self):
        """V2 and V3 format correctly and differ from V1."""
        sample_q = {
            "id": "medqa_sample_2",
            "question": "Which pathogen causes Lyme disease?",
            "options": {"A": "Borrelia burgdorferi", "B": "Treponema pallidum", "C": "Leptospira interrogans"},
            "answer": "A",
        }
        v1 = prompt_v1(sample_q)
        v2 = prompt_v2(sample_q)
        v3 = prompt_v3(sample_q)

        self.assertNotEqual(v1, v2)
        self.assertNotEqual(v1, v3)
        self.assertNotEqual(v2, v3)

        # Check option format in V2: '(A) Borrelia burgdorferi'
        self.assertIn("(A) Borrelia burgdorferi", v2)
        # Check option format in V3: '[A] Borrelia burgdorferi'
        self.assertIn("[A] Borrelia burgdorferi", v3)

        # Check hashes are registered
        self.assertIn("V1", PROMPT_TEMPLATE_HASHES)
        self.assertIn("V2", PROMPT_TEMPLATE_HASHES)
        self.assertIn("V3", PROMPT_TEMPLATE_HASHES)


class TestValidationSplitHardAssertion(unittest.TestCase):
    """Test that load_and_validate_gemma_sweeps raises when encountering a test-split row."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sweep_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hard_assertion_on_test_split_row(self):
        """A row with split='test' triggers a ValueError / Hard assertion failure."""
        gemma_file = self.sweep_dir / "gemma3-4b.jsonl"
        medgemma_file = self.sweep_dir / "medgemma-1.5-4b-it.jsonl"

        # Write one valid validation row and one illegal test row
        valid_row = {
            "id": "q1", "variant": "V1", "split": "validation", "dataset": "medqa",
            "pred": "A", "parsed": True, "correct": True, "truncated": False,
        }
        illegal_test_row = {
            "id": "q2", "variant": "V1", "split": "test", "dataset": "medqa",
            "pred": "B", "parsed": True, "correct": False, "truncated": False,
        }

        append_jsonl(gemma_file, [valid_row, illegal_test_row])
        append_jsonl(medgemma_file, [valid_row, valid_row])

        with self.assertRaises(ValueError) as ctx:
            load_and_validate_gemma_sweeps(self.sweep_dir)
        self.assertIn("Hard assertion failed", str(ctx.exception))
        self.assertIn("non-validation split detected", str(ctx.exception))


class TestRunnerDefaultsAST(unittest.TestCase):
    """Test runner.py default flags via AST inspection (NEVER importing runner.py)."""

    def test_runner_gemma_prompt_sens_defaults(self):
        runner_path = REPO_ROOT / "kaggle_runner" / "runner.py"
        self.assertTrue(runner_path.exists(), f"runner.py not found at {runner_path}")

        tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename="runner.py")

        assigned_constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Constant):
                            assigned_constants[target.id] = node.value.value

        self.assertIn("V3_GEMMA_PROMPT_SENS", assigned_constants)
        self.assertIs(assigned_constants["V3_GEMMA_PROMPT_SENS"], False)

        self.assertIn("V3_GEMMA_SENS_N", assigned_constants)
        self.assertEqual(assigned_constants["V3_GEMMA_SENS_N"], 200)

        self.assertIn("V3_GEMMA_SENS_MAX_HOURS", assigned_constants)
        self.assertEqual(assigned_constants["V3_GEMMA_SENS_MAX_HOURS"], 3.0)


class TestEndToEndAnalysis(unittest.TestCase):
    """Test end-to-end run_analysis on synthetic datasets for N=3 (smoke) and N=200 (full)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_dataset(self, n: int, gemma_correct_fn, medgemma_correct_fn) -> Path:
        sweep_dir = self.work_dir / f"sweep_n_{n}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        gemma_rows = []
        medgemma_rows = []

        variants = ["V1", "V2", "V3"]
        for v in variants:
            for i in range(n):
                qid = f"medqa_val_{i:04d}"
                g_corr = gemma_correct_fn(v, i)
                m_corr = medgemma_correct_fn(v, i)

                g_row = {
                    "id": qid,
                    "variant": v,
                    "model": "gemma3-4b",
                    "split": "validation",
                    "dataset": "medqa",
                    "condition": "C1",
                    "raw_output": f"Answer: {'A' if g_corr else 'B'}",
                    "pred": "A" if g_corr else "B",
                    "parsed": True,
                    "correct": g_corr,
                    "truncated": False,
                    "n_generated_tokens": 4,
                    "finished_naturally": True,
                    "seconds": 0.2,
                }
                m_row = {
                    "id": qid,
                    "variant": v,
                    "model": "medgemma-1.5-4b-it",
                    "split": "validation",
                    "dataset": "medqa",
                    "condition": "C1",
                    "raw_output": f"Answer: {'A' if m_corr else 'B'}",
                    "pred": "A" if m_corr else "B",
                    "parsed": True,
                    "correct": m_corr,
                    "truncated": False,
                    "n_generated_tokens": 4,
                    "finished_naturally": True,
                    "seconds": 0.2,
                }
                gemma_rows.append(g_row)
                medgemma_rows.append(m_row)

        append_jsonl(sweep_dir / "gemma3-4b.jsonl", gemma_rows)
        append_jsonl(sweep_dir / "medgemma-1.5-4b-it.jsonl", medgemma_rows)

        manifest = {
            "git_commit": "abcdef123456",
            "rule_hash": "dummyhash",
            "completed": True,
            "hours_used": 0.25,
            "prompt_template_hashes": {"V1": "h1", "V2": "h2", "V3": "h3"},
        }
        with open(sweep_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        return sweep_dir

    def test_run_analysis_smoke_n3(self):
        """N=3 generates report with SMOKE RUN banner and suppresses stability verdict."""
        # Synthetic data: 3 questions, medgemma wins 2/3 on all variants
        sweep_dir = self._create_synthetic_dataset(
            n=3,
            gemma_correct_fn=lambda v, i: (i == 0),      # 1/3 correct
            medgemma_correct_fn=lambda v, i: (i <= 1),   # 2/3 correct
        )
        out_report = self.work_dir / "smoke_report.md"
        out_png = self.work_dir / "smoke_plot.png"
        out_svg = self.work_dir / "smoke_plot.svg"

        res = run_analysis(
            sweep_dir=sweep_dir,
            out_report=out_report,
            out_png=out_png,
            out_svg=out_svg,
            expected_n=200,
        )

        self.assertEqual(res["n_rows"], 3)
        self.assertTrue(res["is_smoke"])
        self.assertIsNone(res["stability_info"])

        self.assertTrue(out_report.exists())
        self.assertTrue(out_png.exists())
        self.assertTrue(out_svg.exists())

        report_text = out_report.read_text(encoding="utf-8")
        self.assertTrue(report_text.startswith("SMOKE RUN N=3: not for reporting"))
        self.assertIn("Decision verdict is suppressed", report_text)
        self.assertNotIn("Pre-Registered Verdict: STABLE", report_text)
        self.assertNotIn("Pre-Registered Verdict: PROMPT-SENSITIVE", report_text)

    def test_run_analysis_full_n200(self):
        """N=200 generates full report with stability verdict and figures."""
        # Medgemma has ~70% acc (140/200), Gemma3 has ~65% acc (130/200) across all variants
        # -> Gap = -5 pp across all variants, spread = 0 pp -> STABLE
        sweep_dir = self._create_synthetic_dataset(
            n=200,
            gemma_correct_fn=lambda v, i: (i < 130),
            medgemma_correct_fn=lambda v, i: (i < 140),
        )
        out_report = self.work_dir / "full_report.md"
        out_png = self.work_dir / "full_plot.png"
        out_svg = self.work_dir / "full_plot.svg"

        res = run_analysis(
            sweep_dir=sweep_dir,
            out_report=out_report,
            out_png=out_png,
            out_svg=out_svg,
            expected_n=200,
        )

        self.assertEqual(res["n_rows"], 200)
        self.assertFalse(res["is_smoke"])
        self.assertIsNotNone(res["stability_info"])

        is_stable, spread, verdict_msg = res["stability_info"]
        self.assertTrue(is_stable)
        self.assertAlmostEqual(spread, 0.0, places=4)

        self.assertTrue(out_report.exists())
        self.assertTrue(out_png.exists())
        self.assertTrue(out_svg.exists())

        report_text = out_report.read_text(encoding="utf-8")
        self.assertFalse(report_text.startswith("SMOKE RUN"))
        self.assertIn("Pre-Registered Verdict:", report_text)
        self.assertIn("STABLE", report_text)

    def test_run_analysis_format_confound_reporting(self):
        """When a variant has parse-rate difference > 5.0 pp, report flags it and computes descriptive both-parsed gap."""
        # 10 questions: Variant V2 has Gemma parsed on 5/10 (50%) while MedGemma parsed on 10/10 (100%) -> diff = 50% > 5%
        sweep_dir = self.work_dir / "sweep_confound"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        gemma_rows = []
        medgemma_rows = []
        for v in ["V1", "V2", "V3"]:
            for i in range(10):
                qid = f"medqa_val_{i:04d}"
                # In V2, make Gemma parsed=False for odd i
                g_parsed = (v != "V2") or (i % 2 == 0)
                g_corr = (i < 6) and g_parsed
                m_parsed = True
                m_corr = (i < 7)

                g_row = {
                    "id": qid, "variant": v, "model": "gemma3-4b", "split": "validation",
                    "dataset": "medqa", "condition": "C1",
                    "raw_output": f"Answer: {'A' if g_corr else 'B'}",
                    "pred": "A" if g_parsed else None, "parsed": g_parsed,
                    "correct": g_corr, "truncated": False,
                    "n_generated_tokens": 4, "finished_naturally": True, "seconds": 0.2,
                }
                m_row = {
                    "id": qid, "variant": v, "model": "medgemma-1.5-4b-it", "split": "validation",
                    "dataset": "medqa", "condition": "C1",
                    "raw_output": f"Answer: {'A' if m_corr else 'B'}",
                    "pred": "A", "parsed": m_parsed,
                    "correct": m_corr, "truncated": False,
                    "n_generated_tokens": 4, "finished_naturally": True, "seconds": 0.2,
                }
                gemma_rows.append(g_row)
                medgemma_rows.append(m_row)

        append_jsonl(sweep_dir / "gemma3-4b.jsonl", gemma_rows)
        append_jsonl(sweep_dir / "medgemma-1.5-4b-it.jsonl", medgemma_rows)
        with open(sweep_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({"git_commit": "abc", "completed": True}, f)

        out_report = self.work_dir / "confound_report.md"
        out_png = self.work_dir / "confound_plot.png"
        out_svg = self.work_dir / "confound_plot.svg"

        res = run_analysis(sweep_dir, out_report, out_png, out_svg, expected_n=10)
        self.assertTrue(res["variants"]["V2"]["is_format_confounded"])
        self.assertFalse(res["variants"]["V1"]["is_format_confounded"])
        self.assertFalse(res["variants"]["V3"]["is_format_confounded"])

        report_text = out_report.read_text(encoding="utf-8")
        self.assertIn("FLAGGED: V2", report_text)
        self.assertIn("DESCRIPTIVE, selected subset", report_text)


class TestStatisticalHelpers(unittest.TestCase):
    """Test known statistical properties."""

    def test_wilson_ci_known_values(self):
        low, high = wilson_score_ci(0, 100)
        self.assertAlmostEqual(low, 0.0, places=9)
        self.assertAlmostEqual(high, 0.0370, places=3)

    def test_mcnemar_exact_symmetry_and_values(self):
        self.assertAlmostEqual(exact_mcnemar_test(5, 5), 1.0, places=9)
        self.assertAlmostEqual(exact_mcnemar_test(0, 10), 2.0 * (0.5**10), places=5)

    def test_holm_bonferroni(self):
        p_raw = [0.01, 0.04, 0.03]
        p_adj = holm_bonferroni(p_raw)
        # Sorted: 0.01 (x3=0.03), 0.03 (x2=0.06), 0.04 (x1=0.04 -> cummax 0.06)
        self.assertAlmostEqual(p_adj[0], 0.03, places=9)
        self.assertAlmostEqual(p_adj[1], 0.06, places=9)
        self.assertAlmostEqual(p_adj[2], 0.06, places=9)

    def test_format_p_val(self):
        self.assertEqual(format_p_val(0.00005), "<1e-4")
        self.assertEqual(format_p_val(0.000001), "<1e-4")
        self.assertEqual(format_p_val(0.01234), "0.0123")

    def test_format_confound_detection(self):
        """Test Section 4.1 format-confound flag boundary conditions."""
        # Under 5 pp diff: not confounded
        is_conf, diff = check_format_confound(0.95, 0.92)
        self.assertFalse(is_conf)
        self.assertAlmostEqual(diff, 0.03, places=4)

        # Exactly 5 pp diff: not confounded (> 5.0 pp required)
        is_conf_exact, diff_exact = check_format_confound(0.95, 0.90)
        self.assertFalse(is_conf_exact)
        self.assertAlmostEqual(diff_exact, 0.05, places=4)

        # Greater than 5 pp diff: confounded
        is_conf_gt, diff_gt = check_format_confound(0.98, 0.91)
        self.assertTrue(is_conf_gt)
        self.assertAlmostEqual(diff_gt, 0.07, places=4)


if __name__ == "__main__":
    unittest.main()
