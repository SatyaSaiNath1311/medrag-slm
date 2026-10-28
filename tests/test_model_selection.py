"""tests/test_model_selection.py — Unit and integration tests for deployment model selection.

Tests:
1. Pure rule function (select_deployment_model):
   - Eligibility on truncation rate threshold (<= 15%)
   - Eligibility on parse rate threshold (>= 95%)
   - Deployment cap selection for reasoning model (smallest cap with truncation <= 5%)
   - Winner selection when strict accuracy gap is statistically significant
   - Tie-breaking by operational latency when accuracy gap is not significant
   - Disqualification handling when no candidate is eligible or sole candidate is eligible
2. Shared-ID set assertion failure in load_and_validate_candidates
3. End-to-end execution and --check behavior in synthetic temporary environment
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.select_deployment_model import (
    load_and_validate_candidates,
    main,
    select_deployment_model,
)


class TestModelSelectionRuleFunction(unittest.TestCase):
    """Unit tests for the pure decision rule and cap selection."""

    def _make_dummy_medpsy(
        self,
        trunc_by_cap: Dict[int, float],
        strict_acc: float = 0.90,
        parse_rate: float = 1.0,
        seconds_mean: float = 30.0,
    ) -> Dict[str, Any]:
        chosen_cap = 4096
        for c in [1024, 1536, 2048, 3072, 4096]:
            if trunc_by_cap.get(c, 1.0) <= 0.05:
                chosen_cap = c
                break

        caps_dict = {}
        for c, t_rate in trunc_by_cap.items():
            caps_dict[c] = {
                "cap": c,
                "strict_acc": strict_acc,
                "strict_ci": (strict_acc - 0.05, strict_acc + 0.05),
                "trunc_rate": t_rate,
                "trunc_ci": (t_rate - 0.02, t_rate + 0.02),
                "parse_rate": parse_rate,
            }

        return {
            "model": "medpsy-4b",
            "chosen_cap": chosen_cap,
            "cap_rationale": f"Cap {chosen_cap}",
            "caps": caps_dict,
            "truncation_by_cap": trunc_by_cap,
            "at_chosen_cap": caps_dict[chosen_cap],
            "seconds_mean": seconds_mean,
            "seconds_median": seconds_mean - 2.0,
        }

    def _make_dummy_medgemma(
        self,
        strict_acc: float = 0.60,
        trunc_rate: float = 0.0,
        parse_rate: float = 1.0,
        seconds_mean: float = 1.0,
    ) -> Dict[str, Any]:
        return {
            "model": "medgemma-1.5-4b-it",
            "chosen_cap": 16,
            "cap_rationale": "Direct cap 16",
            "strict_acc": strict_acc,
            "strict_ci": (strict_acc - 0.05, strict_acc + 0.05),
            "trunc_rate": trunc_rate,
            "trunc_ci": (trunc_rate, trunc_rate + 0.02),
            "parse_rate": parse_rate,
            "parse_ci": (parse_rate - 0.02, parse_rate),
            "seconds_mean": seconds_mean,
            "seconds_median": seconds_mean,
        }

    def test_deployment_cap_choice_smallest_under_5_percent(self):
        # 1024 has 6% trunc, 1536 has 2% trunc -> 1536 chosen
        medpsy = self._make_dummy_medpsy({1024: 0.06, 1536: 0.02, 2048: 0.01, 3072: 0.0, 4096: 0.0})
        self.assertEqual(medpsy["chosen_cap"], 1536)

        # Fallback case: all caps > 5% -> 4096 chosen
        medpsy_fallback = self._make_dummy_medpsy({1024: 0.50, 1536: 0.40, 2048: 0.30, 3072: 0.20, 4096: 0.10})
        self.assertEqual(medpsy_fallback["chosen_cap"], 4096)

    def test_winner_with_significant_strict_accuracy_gap(self):
        # MedPsy has 92% acc, MedGemma has 61% acc, p < 0.001
        medpsy = self._make_dummy_medpsy({1024: 0.06, 1536: 0.02, 2048: 0.01, 3072: 0.0, 4096: 0.0}, strict_acc=0.92, seconds_mean=33.5)
        medgemma = self._make_dummy_medgemma(strict_acc=0.61, seconds_mean=0.8)

        cmp = {
            "diff": 0.31,
            "bootstrap_ci": (0.21, 0.41),
            "mcnemar_p": 0.0001,
            "is_significant": True,
            "b": 31,
            "c": 0,
        }

        decision = select_deployment_model(medpsy, medgemma, cmp)
        self.assertEqual(decision["winner"], "medpsy-4b")
        self.assertEqual(decision["deployment_cap"], 1536)
        self.assertEqual(decision["status"], "SELECTED")
        self.assertIn("Selected on strict accuracy", decision["rationale"])

    def test_tie_broken_by_latency_when_not_significant(self):
        # MedPsy has 70% acc, MedGemma has 70% acc, p = 1.0 (no significant difference)
        medpsy = self._make_dummy_medpsy({1024: 0.04, 1536: 0.02}, strict_acc=0.70, seconds_mean=30.0)
        medgemma = self._make_dummy_medgemma(strict_acc=0.70, seconds_mean=0.8)

        cmp = {
            "diff": 0.0,
            "bootstrap_ci": (-0.08, +0.08),
            "mcnemar_p": 1.0000,
            "is_significant": False,
            "b": 5,
            "c": 5,
        }

        decision = select_deployment_model(medpsy, medgemma, cmp)
        # Tie broken in favor of lower latency (MedGemma)
        self.assertEqual(decision["winner"], "medgemma-1.5-4b-it")
        self.assertEqual(decision["deployment_cap"], 16)
        self.assertIn("Tie-break by latency", decision["rationale"])

    def test_disqualification_on_truncation_threshold(self):
        # MedPsy has 20% truncation (> 15% threshold) -> disqualified
        medpsy = self._make_dummy_medpsy({1024: 0.20, 1536: 0.20, 4096: 0.20}, strict_acc=0.85)
        medgemma = self._make_dummy_medgemma(strict_acc=0.60, trunc_rate=0.0)

        cmp = {"diff": 0.25, "mcnemar_p": 0.01, "is_significant": True, "bootstrap_ci": (0.1, 0.4), "b": 25, "c": 0}

        decision = select_deployment_model(medpsy, medgemma, cmp)
        self.assertEqual(decision["winner"], "medgemma-1.5-4b-it")
        self.assertEqual(decision["status"], "SOLE_ELIGIBLE_CANDIDATE")
        self.assertFalse(decision["eligibility"]["medpsy-4b"]["eligible"])
        self.assertTrue(decision["eligibility"]["medgemma-1.5-4b-it"]["eligible"])

    def test_disqualification_on_parse_threshold(self):
        # MedGemma has 90% parse rate (< 95% threshold) -> disqualified
        medpsy = self._make_dummy_medpsy({1024: 0.02, 1536: 0.01}, strict_acc=0.85, parse_rate=1.0)
        medgemma = self._make_dummy_medgemma(strict_acc=0.60, parse_rate=0.90)

        cmp = {"diff": 0.25, "mcnemar_p": 0.01, "is_significant": True, "bootstrap_ci": (0.1, 0.4), "b": 25, "c": 0}

        decision = select_deployment_model(medpsy, medgemma, cmp)
        self.assertEqual(decision["winner"], "medpsy-4b")
        self.assertEqual(decision["status"], "SOLE_ELIGIBLE_CANDIDATE")
        self.assertTrue(decision["eligibility"]["medpsy-4b"]["eligible"])
        self.assertFalse(decision["eligibility"]["medgemma-1.5-4b-it"]["eligible"])

    def test_nobody_eligible(self):
        # Both fail eligibility
        medpsy = self._make_dummy_medpsy({1024: 0.25, 4096: 0.20}, parse_rate=0.80)
        medgemma = self._make_dummy_medgemma(trunc_rate=0.25, parse_rate=0.85)

        cmp = {"diff": 0.0, "mcnemar_p": 1.0, "is_significant": False, "bootstrap_ci": (0.0, 0.0), "b": 0, "c": 0}

        decision = select_deployment_model(medpsy, medgemma, cmp)
        self.assertIsNone(decision["winner"])
        self.assertEqual(decision["status"], "NO_ELIGIBLE_CANDIDATE")


class TestModelSelectionDataLoadingAndAssertions(unittest.TestCase):
    """Test data loading assertions and error handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_jsonl(self, path: Path, rows: List[Dict[str, Any]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_shared_id_assertion_failure(self):
        # Prepare mismatched IDs
        p_medpsy = self.repo / "outputs" / "sweeps" / "budget_pair_b" / "medpsy-4b.jsonl"
        p_medgemma = self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "medgemma-1.5-4b-it.jsonl"
        p_man_m = self.repo / "outputs" / "sweeps" / "budget_pair_b" / "manifest_medpsy-4b.json"
        p_man_g = self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "manifest_medgemma-1.5-4b-it.json"

        p_man_m.parent.mkdir(parents=True, exist_ok=True)
        p_man_g.parent.mkdir(parents=True, exist_ok=True)
        p_man_m.write_text(json.dumps({"hours_used": 0.93}), encoding="utf-8")
        p_man_g.write_text(json.dumps({"hours_used": 0.16}), encoding="utf-8")

        rows_m = [{"id": f"q_{i}", "split": "validation"} for i in range(100)]
        rows_g = [{"id": f"different_q_{i}", "split": "validation", "variant": "V1"} for i in range(100)]

        self._write_jsonl(p_medpsy, rows_m)
        self._write_jsonl(p_medgemma, rows_g)

        with self.assertRaises(AssertionError) as ctx:
            load_and_validate_candidates(self.repo)
        # New code checks for missing IDs before ordering; message is now "MedGemma V1 is missing"
        self.assertIn("MedGemma V1 is missing", str(ctx.exception))

    def test_v1_restricted_to_medpsy_id_order_not_slice(self):
        """Regression test for the first-100-slice bug.

        The gemma file has 3 variants (V1, V2, V3) × 200 questions. The 200 V1
        questions are a superset of the 100 medpsy questions. Accuracy over all 200
        V1 questions intentionally differs from accuracy over the 100 medpsy
        questions, and the correct value (57/100) must be returned, not the
        slice-based wrong value (61/200 = 61/200 from the first 100 of the 200 V1).
        """
        p_medpsy = self.repo / "outputs" / "sweeps" / "budget_pair_b" / "medpsy-4b.jsonl"
        p_medgemma = self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "medgemma-1.5-4b-it.jsonl"
        p_man_m = self.repo / "outputs" / "sweeps" / "budget_pair_b" / "manifest_medpsy-4b.json"
        p_man_g = self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "manifest_medgemma-1.5-4b-it.json"

        p_man_m.parent.mkdir(parents=True, exist_ok=True)
        p_man_g.parent.mkdir(parents=True, exist_ok=True)
        p_man_m.write_text(json.dumps({"hours_used": 0.93}), encoding="utf-8")
        p_man_g.write_text(json.dumps({"hours_used": 0.16}), encoding="utf-8")

        # medpsy has 100 questions: q_100 … q_199
        # (i.e. the second 100 of a hypothetical 200-question pool)
        medpsy_ids = [f"q_{i}" for i in range(100, 200)]
        rows_m = []
        for i, qid in enumerate(medpsy_ids):
            rows_m.append({
                "id": qid,
                "split": "validation",
                "parsed": True,
                "correct": (i < 57),        # 57/100 correct in the medpsy slice
                "finished_naturally": True,
                "n_generated_tokens": 500,
            })

        # gemma file has 200 questions × 3 variants = 600 rows.
        # The first 100 questions (q_0..q_99) are NOT in the medpsy sweep.
        # Questions q_100..q_199 ARE in the medpsy sweep.
        # Within V1, the first 100 rows are q_0..q_99 (all correct = 100% acc),
        # and q_100..q_199 rows have 57 correct; the old slice would have returned
        # the first-100-of-V1 (q_0..q_99, all correct) = 100%, not 57%.
        rows_g = []
        all_200_ids = [f"q_{i}" for i in range(200)]
        for variant in ["V1", "V2", "V3"]:
            for j, qid in enumerate(all_200_ids):
                # V1, q_0..q_99: all correct (these are NOT the medpsy questions)
                # V1, q_100..q_199: 57 correct (these ARE the medpsy questions)
                if variant == "V1" and j < 100:
                    correct = True   # high acc in non-medpsy slice
                elif variant == "V1":
                    correct = (j - 100 < 57)  # 57 correct in medpsy slice
                else:
                    correct = False
                rows_g.append({
                    "id": qid,
                    "split": "validation",
                    "variant": variant,
                    "parsed": True,
                    "correct": correct,
                    "finished_naturally": True,
                    "n_generated_tokens": 10,
                })

        self._write_jsonl(p_medpsy, rows_m)
        self._write_jsonl(p_medgemma, rows_g)

        mp_rows, mg_rows, _ = load_and_validate_candidates(self.repo)

        # Both sides must have exactly 100 rows
        self.assertEqual(len(mp_rows), 100)
        self.assertEqual(len(mg_rows), 100)

        # IDs must match medpsy file order (q_100..q_199)
        self.assertEqual([r["id"] for r in mp_rows], medpsy_ids)
        self.assertEqual([r["id"] for r in mg_rows], medpsy_ids)

        # Gemma strict accuracy on the correct 100 questions must be 57, NOT 100
        mg_strict = sum(
            1 for r in mg_rows if r["parsed"] and r["correct"]
        )
        self.assertEqual(mg_strict, 57,
            "load_and_validate_candidates returned gemma rows from the wrong slice "
            f"(got {mg_strict}/100, expected 57/100)")


class TestModelSelectionEndToEndAndCheck(unittest.TestCase):
    """Test end-to-end execution and --check consistency."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)

        # Build synthetic 100-question layout
        p_medpsy = self.repo / "outputs" / "sweeps" / "budget_pair_b" / "medpsy-4b.jsonl"
        p_medgemma = self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "medgemma-1.5-4b-it.jsonl"
        p_man_m = self.repo / "outputs" / "sweeps" / "budget_pair_b" / "manifest_medpsy-4b.json"
        p_man_g = self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "manifest_medgemma-1.5-4b-it.json"

        p_man_m.parent.mkdir(parents=True, exist_ok=True)
        p_man_g.parent.mkdir(parents=True, exist_ok=True)
        p_man_m.write_text(json.dumps({"hours_used": 0.9328, "timestamp": "2026-10-27T13:12:08Z"}), encoding="utf-8")
        p_man_g.write_text(json.dumps({"hours_used": 0.1600, "timestamp": "2026-10-28T07:26:08Z"}), encoding="utf-8")

        rows_m = []
        rows_g = []
        for i in range(100):
            qid = f"medqa-val-{i:03d}"
            # MedPsy has 90% accuracy, 6% truncation at 1024, 2% truncation at 1536
            m_corr = (i < 90)
            if i < 94:
                m_tokens = 500
            elif i < 98:
                m_tokens = 1200
            else:
                m_tokens = 1800

            rows_m.append({
                "id": qid,
                "split": "validation",
                "parsed": True,
                "correct": m_corr,
                "finished_naturally": True,
                "n_generated_tokens": m_tokens,
                "seconds": 25.0,
            })
            # MedGemma has 60% accuracy, 0% truncation
            g_corr = (i < 60)
            rows_g.append({
                "id": qid,
                "split": "validation",
                "variant": "V1",
                "parsed": True,
                "correct": g_corr,
                "truncated": False,
                "seconds": 0.8,
            })

        p_medpsy.parent.mkdir(parents=True, exist_ok=True)
        with open(p_medpsy, "w", encoding="utf-8") as f:
            for r in rows_m:
                f.write(json.dumps(r) + "\n")

        p_medgemma.parent.mkdir(parents=True, exist_ok=True)
        with open(p_medgemma, "w", encoding="utf-8") as f:
            for r in rows_g:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_main_generates_files_and_check_passes(self):
        out_md = self.repo / "docs" / "MODEL_SELECTION.md"
        out_json = self.repo / "docs" / "model_selection.json"

        # 1. Run generation
        main(["--repo-root", str(self.repo)])
        self.assertTrue(out_md.exists(), "MODEL_SELECTION.md was not created")
        self.assertTrue(out_json.exists(), "model_selection.json was not created")

        md_content = out_md.read_text(encoding="utf-8")
        self.assertIn("medpsy-4b", md_content)
        self.assertIn("1536", md_content)

        ledger = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(ledger["winner"], "medpsy-4b")
        self.assertEqual(ledger["deployment_cap"], 1536)

        # 2. Check --check succeeds
        try:
            main(["--repo-root", str(self.repo), "--check"])
        except SystemExit as exc:
            self.assertEqual(exc.code, 0, f"--check failed unexpectedly with code {exc.code}")

        # 3. Check --check fails if file is tampered
        out_md.write_text("TAMPERED", encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            main(["--repo-root", str(self.repo), "--check"])
        self.assertEqual(ctx.exception.code, 1)

    def test_report_text_changes_when_input_row_mutated(self):
        """Mutating one input row in the synthetic dataset must change the generated report text."""
        out_md = self.repo / "docs" / "MODEL_SELECTION.md"
        main(["--repo-root", str(self.repo)])
        text_before = out_md.read_text(encoding="utf-8")
        self.assertIn("90.0%", text_before)

        # Mutate row 0 of medpsy: flip correct from True to False
        p_medpsy = self.repo / "outputs" / "sweeps" / "budget_pair_b" / "medpsy-4b.jsonl"
        rows = [json.loads(line) for line in p_medpsy.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(rows[0]["correct"])
        rows[0]["correct"] = False
        with open(p_medpsy, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Re-run generation
        main(["--repo-root", str(self.repo)])
        text_after = out_md.read_text(encoding="utf-8")

        self.assertNotEqual(text_before, text_after)
        self.assertNotIn("90.0%", text_after)
        self.assertIn("89.0%", text_after)


if __name__ == "__main__":
    unittest.main()
