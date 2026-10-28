"""tests/test_paper_tables.py — Synthetic fixture tests for V3 paper tables generation.

Covers:
  1. The MedQA filter on a 650-row file (500 medqa + 150 unanswerable -> 500 medqa).
  2. Shard files being ignored (check_not_shard raises ValueError on *_shard*).
  3. Macro-F1 computation on known tiny cases.
  4. Cross-check assertion failing loudly when a metric disagrees with analysis reports (Table 1).
  5. AUROC with tie handling and half credit on synthetic cases.
  6. Table 2 cross-check assertion failing loudly on AUROC or truncation disagreement.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.paper_tables import (
    auroc_with_ties,
    check_not_shard,
    compute_macro_f1,
    cross_check_results,
    cross_check_table2_results,
    filter_medqa_rows,
)


class TestPaperTablesSynthetic(unittest.TestCase):
    def test_medqa_filter_on_650_rows(self):
        # 500 MedQA rows and 150 non-MedQA rows (e.g. unanswerable set)
        rows = [
            {"id": f"medqa_{i}", "dataset": "medqa", "split": "test"}
            for i in range(500)
        ] + [
            {"id": f"unans_{i}", "dataset": "unanswerable", "split": "test"}
            for i in range(150)
        ]
        self.assertEqual(len(rows), 650)
        filtered = filter_medqa_rows(rows)
        self.assertEqual(len(filtered), 500)
        self.assertTrue(all(r["dataset"] == "medqa" for r in filtered))

    def test_shard_files_ignored(self):
        shard_path = Path("outputs/v3/medpsy-1.7b/C1/medpsy-1.7b_shard0.jsonl")
        with self.assertRaises(ValueError):
            check_not_shard(shard_path)

        shard_path_2 = Path("outputs/kaggle_qa/medpsy_final/work/phase5/medpsy-4b_shard2.jsonl")
        with self.assertRaises(ValueError):
            check_not_shard(shard_path_2)

        valid_path = Path("outputs/v3/medpsy-1.7b/C1/medpsy-1.7b.jsonl")
        # Should not raise
        check_not_shard(valid_path)

    def test_macro_f1_known_cases(self):
        # Case 1: Perfect match (1.0)
        golds_1 = ["A", "B", "C"]
        preds_1 = ["A", "B", "C"]
        self.assertAlmostEqual(compute_macro_f1(golds_1, preds_1, ["A", "B", "C"]), 1.0)

        # Case 2: Complete mismatch (0.0)
        golds_2 = ["A", "A"]
        preds_2 = ["B", "B"]
        self.assertAlmostEqual(compute_macro_f1(golds_2, preds_2, ["A", "B"]), 0.0)

        # Case 3: Symmetric 50% case
        # For A: TP=1, FP=1, FN=1 -> F1=0.5
        # For B: TP=1, FP=1, FN=1 -> F1=0.5 -> Macro-F1 = 0.5
        golds_3 = ["A", "B", "A", "B"]
        preds_3 = ["A", "A", "B", "B"]
        self.assertAlmostEqual(compute_macro_f1(golds_3, preds_3, ["A", "B"]), 0.5)

    def test_cross_check_fails_on_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            report_dir = tmp_root / "outputs" / "analysis"
            report_dir.mkdir(parents=True, exist_ok=True)

            report_file = report_dir / "pair_a_medqa.md"
            report_file.write_text(
                "# Pair A MedQA\n\n"
                "## C1\n"
                "- medpsy-1.7b: truncated 0/500 (0.0%); strict 250/500 = 0.500 [0.450, 0.550]; fallback 250/500 = 0.500\n",
                encoding="utf-8",
            )

            # Matching record
            matching_rec = {
                "pair": "Pair A",
                "model_key": "medpsy-1.7b",
                "condition": "C1 (closed-book)",
                "dataset": "MedQA",
                "strict_acc": 0.500,
                "fallback_acc": 0.500,
                "truncation_rate": 0.0,
            }
            # Should not raise
            cross_check_results([matching_rec], tmp_root)

            # Disagreeing strict accuracy
            mismatched_strict = dict(matching_rec)
            mismatched_strict["strict_acc"] = 0.400
            with self.assertRaises(AssertionError) as ctx:
                cross_check_results([mismatched_strict], tmp_root)
            self.assertIn("computed strict acc 0.400 != report 0.500", str(ctx.exception))

            # Disagreeing fallback accuracy
            mismatched_fallback = dict(matching_rec)
            mismatched_fallback["fallback_acc"] = 0.600
            with self.assertRaises(AssertionError) as ctx:
                cross_check_results([mismatched_fallback], tmp_root)
            self.assertIn("computed fallback acc 0.600 != report 0.500", str(ctx.exception))

    def test_auroc_with_ties_synthetic_fixtures(self):
        # 1. Complete tie (all scores identical) -> exactly 0.5
        pos_tie = np.array([0.5, 0.5])
        neg_tie = np.array([0.5, 0.5])
        self.assertAlmostEqual(auroc_with_ties(pos_tie, neg_tie), 0.5)

        # 2. Perfect separation (all pos > all neg) -> 1.0
        pos_perf = np.array([0.9, 0.8])
        neg_perf = np.array([0.2, 0.1])
        self.assertAlmostEqual(auroc_with_ties(pos_perf, neg_perf), 1.0)

        # 3. Complete inversion (all pos < all neg) -> 0.0
        pos_inv = np.array([0.1, 0.2])
        neg_inv = np.array([0.8, 0.9])
        self.assertAlmostEqual(auroc_with_ties(pos_inv, neg_inv), 0.0)

        # 4. Tie handling exact half credit:
        # Pairs: (1.0, 0.5) -> 1.0; (1.0, 0.0) -> 1.0; (0.5, 0.5) -> 0.5; (0.5, 0.0) -> 1.0
        # Expected AUROC = (1.0 + 1.0 + 0.5 + 1.0) / 4 = 3.5 / 4 = 0.875
        pos_mix = np.array([1.0, 0.5])
        neg_mix = np.array([0.5, 0.0])
        self.assertAlmostEqual(auroc_with_ties(pos_mix, neg_mix), 0.875)

        # 5. Empty inputs return None
        self.assertIsNone(auroc_with_ties(np.array([]), np.array([0.5])))
        self.assertIsNone(auroc_with_ties(np.array([0.5]), np.array([])))

    def test_table2_cross_check_fails_on_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            report_dir = tmp_root / "outputs" / "analysis"
            report_dir.mkdir(parents=True, exist_ok=True)

            report_file = report_dir / "pair_a_c45.md"
            report_file.write_text(
                "# Pair A C4/C5 Analysis\n\n"
                "## 3. Confidence Separation on All Rows (N=140)\n\n"
                "| Condition | Model | Mean Conf Chal (N=100) | Mean Conf Ctrl (N=40) | Chal Conf >= 0.90 (%) | AUROC (1 - conf) | 95% Bootstrap CI | MedPsy - Qwen Diff | 95% Diff CI |\n"
                "|---|---|---|---|---|---|---|---|---|\n"
                "| C4 | `qwen3-1.7b-thinking` | 0.9801 | 1.0000 | 94.0% | 0.6200 | [0.5800, 0.6632] | — | — |\n\n"
                "### Truncation Rate: Challenge Rows vs Control Rows\n\n"
                "| Condition | Model | Challenge Truncation (N=100, Wilson 95% CI) | Control Truncation (N=40, Wilson 95% CI) |\n"
                "|---|---|---|---|\n"
                "| C4 | `qwen3-1.7b-thinking` | 29/100 (29.0%, [21.0%, 38.5%]) | 0/40 (0.0%, [0.0%, 8.8%]) |\n",
                encoding="utf-8",
            )

            # Matching record
            matching_rec = {
                "model_key": "qwen3-1.7b-thinking",
                "condition": "C4 (closed-book unanswerable)",
                "auroc": 0.6200,
                "chal_trunc_pct": 29.0,
                "ctrl_trunc_pct": 0.0,
            }
            # Should not raise
            cross_check_table2_results([matching_rec], tmp_root)

            # Disagreeing AUROC
            mismatched_auc = dict(matching_rec)
            mismatched_auc["auroc"] = 0.7500
            with self.assertRaises(AssertionError) as ctx:
                cross_check_table2_results([mismatched_auc], tmp_root)
            self.assertIn("computed AUROC 0.7500 != report 0.6200", str(ctx.exception))

            # Disagreeing challenge truncation
            mismatched_chal = dict(matching_rec)
            mismatched_chal["chal_trunc_pct"] = 35.0
            with self.assertRaises(AssertionError) as ctx:
                cross_check_table2_results([mismatched_chal], tmp_root)
            self.assertIn("computed challenge trunc 35.0% != report 29.0%", str(ctx.exception))

            # Disagreeing control truncation
            mismatched_ctrl = dict(matching_rec)
            mismatched_ctrl["ctrl_trunc_pct"] = 10.0
            with self.assertRaises(AssertionError) as ctx:
                cross_check_table2_results([mismatched_ctrl], tmp_root)
            self.assertIn("computed control trunc 10.0% != report 0.0%", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
