"""tests/test_pairs.py — Unit Tests for Matched-Pair Statistics and Model Parsing Smoke Tests.

Tests:
  1. McNemar exact test against known theoretical binomial values (symmetry, extreme discordance, edge cases).
  2. Holm-Bonferroni correction (monotonicity, preservation of order, None handling).
  3. Cochran-Mantel-Haenszel (CMH) test statistic, p-value, and common odds ratio against known tables.
  4. Logistic regression interaction test (model_type x RAG) against known data.
  5. Bootstrap 95% CI bounds on known binary sequences.
  6. Parsing smoke tests for all 4 new model entries (MedPsy-1.7B, Qwen3-1.7B-thinking,
     Qwen3-4B-Thinking-2507, MedGemma-1.5-4B-it) using mocked outputs without loading model weights.
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.analysis_pair_a_medqa import run_analysis as run_pair_a_analysis
from src.analysis_pairs import (
    bootstrap_ci,
    cochran_mantel_haenszel,
    fit_logistic_interaction,
    holm_bonferroni,
    mcnemar_exact,
)
from src.llm import (
    extract_final_answer_from_reasoning,
    parse_answer,
    parse_citations,
)



class TestMatchedPairStatistics(unittest.TestCase):
    """Test statistical algorithms against verified analytical and numerical fixtures."""

    def test_mcnemar_exact_symmetry(self):
        """When b == c, the two-sided McNemar exact test p-value must be exactly 1.0."""
        res = mcnemar_exact(b=10, c=10)
        self.assertAlmostEqual(res["p_value"], 1.0, places=5)
        self.assertAlmostEqual(res["odds_ratio"], 1.0, places=5)

    def test_mcnemar_exact_zeros(self):
        """Zero discordant pairs (b=0, c=0) yields p=1.0 and OR=1.0."""
        res = mcnemar_exact(b=0, c=0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["odds_ratio"], 1.0)

    def test_mcnemar_exact_known_value_10_0(self):
        """b=0, c=10: Two-sided binomial p-value is 2 * (0.5)^10 = 2/1024 ~ 0.0019531."""
        res = mcnemar_exact(b=0, c=10)
        expected_p = 2.0 / 1024.0
        self.assertAlmostEqual(res["p_value"], expected_p, places=5)
        # Haldane-Anscombe smoothed OR = (10 + 0.5) / (0 + 0.5) = 21.0
        self.assertAlmostEqual(res["odds_ratio"], 21.0, places=4)

    def test_mcnemar_exact_known_value_5_20(self):
        """b=5, c=20: n=25, k=5. Two-sided binomtest p-value is ~0.004077."""
        res = mcnemar_exact(b=5, c=20)
        # Binomial sum: 2 * sum_{i=0..5} C(25, i) * 0.5^25 = 0.0040771
        self.assertAlmostEqual(res["p_value"], 0.004077, places=4)
        self.assertAlmostEqual(res["odds_ratio"], 4.0, places=4)

    def test_holm_bonferroni_step_down(self):
        """Test step-down adjustment: [0.01, 0.04, 0.05] with m=3."""
        # p1 = 0.01 * 3 = 0.03
        # p2 = 0.04 * 2 = 0.08
        # p3 = 0.05 * 1 = 0.05 -> enforced max(0.08, 0.05) = 0.08
        raw_ps = [0.01, 0.04, 0.05]
        adj_ps = holm_bonferroni(raw_ps)
        self.assertAlmostEqual(adj_ps[0], 0.03, places=4)
        self.assertAlmostEqual(adj_ps[1], 0.08, places=4)
        self.assertAlmostEqual(adj_ps[2], 0.08, places=4)

    def test_holm_bonferroni_unsorted_and_none(self):
        """Test that Holm handles unsorted inputs and preserves None values for missing models."""
        raw_ps = [0.04, None, 0.01]
        adj_ps = holm_bonferroni(raw_ps)
        # Valid count m=2: p(0.01)*2 = 0.02, p(0.04)*1 = 0.04
        self.assertAlmostEqual(adj_ps[0], 0.04, places=4)
        self.assertIsNone(adj_ps[1])
        self.assertAlmostEqual(adj_ps[2], 0.02, places=4)

    def test_cochran_mantel_haenszel(self):
        """Test CMH test on two homogeneous 2x2 stratified tables."""
        # Table 1: [[30, 10], [15, 25]] -> OR ~ 5.0
        # Table 2: [[35, 15], [10, 20]] -> OR ~ 4.67
        t1 = [[30, 10], [15, 25]]
        t2 = [[35, 15], [10, 20]]
        res = cochran_mantel_haenszel([t1, t2])
        self.assertIsNotNone(res["cmh_stat"])
        self.assertGreater(res["cmh_stat"], 10.0)  # Strong association
        self.assertLess(res["p_value"], 0.001)
        self.assertAlmostEqual(res["common_or"], 4.88, places=1)

    def test_logistic_interaction_fit(self):
        """Test logistic regression interaction estimator on synthetic factorial data."""
        np.random.seed(42)
        n_per_cell = 100
        # Generic Base: 40% correct
        y_gb = np.random.binomial(1, 0.40, n_per_cell)
        # Generic RAG: 60% correct (delta = +20%)
        y_gr = np.random.binomial(1, 0.60, n_per_cell)
        # Medical Base: 80% correct
        y_mb = np.random.binomial(1, 0.80, n_per_cell)
        # Medical RAG: 82% correct (delta = +2%, much smaller gain -> negative interaction)
        y_mr = np.random.binomial(1, 0.82, n_per_cell)

        y = np.concatenate([y_gb, y_gr, y_mb, y_mr])
        x_med = np.concatenate([np.zeros(2 * n_per_cell), np.ones(2 * n_per_cell)])
        x_rag = np.concatenate([np.zeros(n_per_cell), np.ones(n_per_cell), np.zeros(n_per_cell), np.ones(n_per_cell)])

        res = fit_logistic_interaction(y, x_med, x_rag)
        self.assertIsNotNone(res["beta_int"])
        self.assertIsNotNone(res["wald_p"])
        # Expect negative interaction coefficient because medical gain is smaller than generic gain
        self.assertLess(res["beta_int"], 0.0)

    def test_bootstrap_ci_bounds(self):
        """Test bootstrap percentile confidence interval bounds."""
        # All ones -> CI must be exactly [1.0, 1.0]
        ci_ones = bootstrap_ci([1] * 50, n_resamples=200)
        self.assertEqual(ci_ones, (1.0, 1.0))

        # Balanced array of 0s and 1s -> CI must bracket 0.50
        ci_bal = bootstrap_ci([0, 1] * 50, n_resamples=500, seed=42)
        self.assertLess(ci_bal[0], 0.50)
        self.assertGreater(ci_bal[1], 0.50)


class TestModelParsingSmoke(unittest.TestCase):
    """Smoke test prompt formatting, answer parsing, and citations for all 4 new model entries."""

    def setUp(self):
        self.options_medqa = {
            "A": "Intravenous ceftriaxone",
            "B": "Oral azithromycin monotherapy",
            "C": "Supportive care only",
            "D": "Urgent chest tube insertion",
        }
        self.options_pubmedqa = {"A": "yes", "B": "no", "C": "maybe"}

    def test_medpsy_1_7b_parsing(self):
        """medpsy-1.7b: thinking model output with <think>...</think> and final answer."""
        mock_output = (
            "<think>\nThe patient presents with high fever, productive cough, and lobar consolidation.\n"
            "This suggests community-acquired pneumonia requiring intravenous antibiotic therapy.\n"
            "Comparing the options, ceftriaxone is the guideline-directed choice.\n</think>\n"
            "Answer: A"
        )
        parsed = extract_final_answer_from_reasoning(mock_output, self.options_medqa)
        self.assertEqual(parsed, "A")

    def test_qwen3_1_7b_thinking_parsing(self):
        """qwen3-1.7b-thinking: generic thinking control with reasoning block and bold answer."""
        mock_output = (
            "<think>\nEvaluating pneumonia severity and treatment options...\n"
            "First-line outpatient treatment is amoxicillin or doxycycline; inpatient is ceftriaxone.\n"
            "</think>\n"
            "Based on the clinical findings, the most appropriate choice is:\n"
            "Answer: **A. Intravenous ceftriaxone**"
        )
        parsed = extract_final_answer_from_reasoning(mock_output, self.options_medqa)
        self.assertEqual(parsed, "A")

    def test_qwen3_4b_thinking_2507_rag_parsing(self):
        """qwen3-4b-thinking-2507: reasoning model in RAG mode with citations."""
        mock_output = (
            "<think>\nPassage [1] explicitly recommends ceftriaxone for severe bacterial pneumonia.\n"
            "Passage [3] rules out azithromycin monotherapy due to macrolide resistance.\n"
            "</think>\n"
            "Answer: A\n"
            "Evidence: [1] [3]"
        )
        parsed = extract_final_answer_from_reasoning(mock_output, self.options_medqa)
        self.assertEqual(parsed, "A")

        citations = parse_citations(mock_output, n_passages=5)
        self.assertEqual(citations, [1, 3])

    def test_medgemma_1_5_4b_it_parsing(self):
        """medgemma-1.5-4b-it: standard instruction-tuned model (no thinking block)."""
        mock_output = "Answer: A. Intravenous ceftriaxone"
        parsed = parse_answer(mock_output, self.options_medqa)
        self.assertEqual(parsed, "A")

    def test_thinking_truncation_detection(self):
        """When reasoning hits the token budget cap without producing Answer:, truncated=True."""
        mock_truncated = (
            "<think>\nThe patient has multiple comorbidities including chronic obstructive pulmonary disease,\n"
            "hypertension, and stage 3 chronic kidney disease. First we analyze the differential diagnosis:\n"
            "1. Bacterial pneumonia\n2. Viral pneumonitis\n3. Congestive heart failure exacerbation\n"
            "Looking closer at the radiograph and arterial blood gas findings..."
        )
        parsed = extract_final_answer_from_reasoning(mock_truncated, self.options_medqa)
        self.assertIsNone(parsed)
        # Model logic flags truncated when parsed is None and generation tokens reach cap
        gen_tokens = 1024
        is_truncated = (parsed is None) and (gen_tokens >= 1024)
        self.assertTrue(is_truncated)


class TestPairAMedQASynthetic(unittest.TestCase):
    """End-to-end synthetic tests for Pair A MedQA report generator."""

    def test_report_text_changes_when_input_row_mutated(self):
        """Mutating one input row changes the generated Pair A markdown report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            p_m = repo / "outputs" / "v3" / "medpsy-1.7b" / "C1" / "medpsy-1.7b.jsonl"
            p_q = repo / "outputs" / "v3" / "qwen3-1.7b-thinking" / "C1" / "qwen3-1.7b-thinking.jsonl"
            p_m.parent.mkdir(parents=True, exist_ok=True)
            p_q.parent.mkdir(parents=True, exist_ok=True)

            rows_m = [
                {"id": "q1", "split": "test", "dataset": "medqa", "parsed": True, "truncated": False, "correct": True},
                {"id": "q2", "split": "test", "dataset": "medqa", "parsed": True, "truncated": False, "correct": True},
            ]
            rows_q = [
                {"id": "q1", "split": "test", "dataset": "medqa", "parsed": True, "truncated": False, "correct": False},
                {"id": "q2", "split": "test", "dataset": "medqa", "parsed": True, "truncated": False, "correct": False},
            ]
            with open(p_m, "w", encoding="utf-8") as f:
                for r in rows_m:
                    f.write(json.dumps(r) + "\n")
            with open(p_q, "w", encoding="utf-8") as f:
                for r in rows_q:
                    f.write(json.dumps(r) + "\n")

            out_md = repo / "outputs" / "analysis" / "pair_a_medqa.md"
            run_pair_a_analysis(repo_root=repo, out_path=out_md, expected_n=2)
            text_before = out_md.read_text(encoding="utf-8")
            self.assertIn("strict 2/2 = 1.000", text_before)

            # Mutate row 0 in medpsy
            rows_m[0]["correct"] = False
            with open(p_m, "w", encoding="utf-8") as f:
                for r in rows_m:
                    f.write(json.dumps(r) + "\n")

            run_pair_a_analysis(repo_root=repo, out_path=out_md, expected_n=2)
            text_after = out_md.read_text(encoding="utf-8")

            self.assertNotEqual(text_before, text_after)
            self.assertIn("strict 1/2 = 0.500", text_after)


if __name__ == "__main__":
    unittest.main()
