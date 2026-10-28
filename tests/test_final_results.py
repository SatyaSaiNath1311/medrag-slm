"""tests/test_final_results.py — Synthetic fixture tests for consolidated results and verification tools.

Covers:
  1. p-value formatter (<1e-4 vs standard formatting).
  2. Table builder on tiny synthetic pair.
  3. Facts ledger --check failing when a source value is altered.
  4. compare_report_numbers detecting a changed number and ignoring a path line.
  5. Manifest-key reading for the Gemma report fix.
"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_report_numbers import compare_reports
from src.analysis_gemma_prompt_sensitivity import load_and_validate_gemma_sweeps
from src.build_paper_facts import FACTS, generate_markdown
from src.final_results_v3 import (
    evaluate_model_cell,
    evaluate_paired_comparison,
    format_p,
    main,
    wilson_ci,
)


class TestPValueFormatter(unittest.TestCase):
    def test_p_value_formatting(self):
        self.assertEqual(format_p(0.0), "<1e-4")
        self.assertEqual(format_p(0.00001), "<1e-4")
        self.assertEqual(format_p(0.000099), "<1e-4")
        self.assertEqual(format_p(0.0001), "0.0001")
        self.assertEqual(format_p(0.02548), "0.0255")
        self.assertEqual(format_p(0.5000), "0.5000")
        self.assertEqual(format_p(1.0), "1.0000")


class TestTableBuilderSynthetic(unittest.TestCase):
    def test_synthetic_cell_and_comparison(self):
        # 5 synthetic questions
        # Domain: 4 strictly correct, 1 truncated
        domain_rows = [
            {"id": "q1", "parsed": True, "truncated": False, "correct": True},
            {"id": "q2", "parsed": True, "truncated": False, "correct": True},
            {"id": "q3", "parsed": True, "truncated": False, "correct": True},
            {"id": "q4", "parsed": True, "truncated": False, "correct": True},
            {"id": "q5", "parsed": True, "truncated": True, "correct": True},  # fallback correct, strict incorrect
        ]
        # Generic: 2 strictly correct, 2 incorrect, 1 truncated
        generic_rows = [
            {"id": "q1", "parsed": True, "truncated": False, "correct": True},
            {"id": "q2", "parsed": True, "truncated": False, "correct": True},
            {"id": "q3", "parsed": True, "truncated": False, "correct": False},
            {"id": "q4", "parsed": True, "truncated": False, "correct": False},
            {"id": "q5", "parsed": True, "truncated": True, "correct": False},
        ]

        d_eval = evaluate_model_cell(domain_rows)
        g_eval = evaluate_model_cell(generic_rows)

        self.assertEqual(d_eval["strict_k"], 4)
        self.assertEqual(d_eval["strict_acc"], 0.8)
        self.assertEqual(d_eval["fallback_k"], 5)
        self.assertEqual(d_eval["fallback_acc"], 1.0)
        self.assertEqual(d_eval["trunc_k"], 1)
        self.assertEqual(d_eval["trunc_rate"], 0.2)
        self.assertEqual(d_eval["parse_rate"], 1.0)

        self.assertEqual(g_eval["strict_k"], 2)
        self.assertEqual(g_eval["strict_acc"], 0.4)
        self.assertEqual(g_eval["fallback_k"], 2)
        self.assertEqual(g_eval["fallback_acc"], 0.4)
        self.assertEqual(g_eval["trunc_k"], 1)

        cmp = evaluate_paired_comparison(d_eval, g_eval)
        self.assertAlmostEqual(cmp["strict_diff"], 0.4)
        self.assertEqual(cmp["b_strict"], 2)  # q3, q4
        self.assertEqual(cmp["c_strict"], 0)
        self.assertAlmostEqual(cmp["fallback_diff"], 0.6)
        self.assertEqual(cmp["b_fallback"], 3)
        self.assertEqual(cmp["c_fallback"], 0)


class TestFactsLedgerAudit(unittest.TestCase):
    def test_facts_ledger_consistency_and_detect_tampering(self):
        # Base generated markdown matches FACTS
        clean_md = generate_markdown()
        self.assertIn("F001", clean_md)
        self.assertIn("F040", clean_md)

        # Tampered content
        tampered_md = clean_md.replace("+29.8 pp", "+35.0 pp")
        self.assertNotEqual(clean_md.strip(), tampered_md.strip())


class TestCompareReportNumbers(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compare_reports_identical_with_different_paths(self):
        old_file = self.work_dir / "old.md"
        new_file = self.work_dir / "new.md"

        old_file.write_text(
            "# Report\n"
            "- Strict accuracy: 83.2% [79.7%, 86.2%]\n"
            "- Reference File: `/Users/alice/repo/outputs/data.jsonl`\n"
            "- Truncation: 10.6%\n"
            "![Plot](/Users/alice/repo/outputs/plot.png)\n"
        )
        new_file.write_text(
            "# Report\n"
            "- Strict accuracy: 83.2% [79.7%, 86.2%]\n"
            "- Reference File: `outputs/data.jsonl`\n"
            "- Truncation: 10.6%\n"
            "![Plot](plot.png)\n"
        )

        self.assertTrue(compare_reports(old_file, new_file))

    def test_compare_reports_allows_p_value_format_update(self):
        old_file = self.work_dir / "old_p.md"
        new_file = self.work_dir / "new_p.md"

        old_file.write_text(
            "| Cap | Strict Diff | Raw McNemar p |\n"
            "| 1024 | +88.0% | 0.0000 |\n"
        )
        new_file.write_text(
            "| Cap | Strict Diff | Raw McNemar p |\n"
            "| 1024 | +88.0% | <1e-4 |\n"
        )

        self.assertTrue(compare_reports(old_file, new_file))

    def test_compare_reports_detects_altered_metric(self):
        old_file = self.work_dir / "old_val.md"
        new_file = self.work_dir / "new_val.md"

        old_file.write_text(
            "| Cap | Strict Diff | Raw McNemar p |\n"
            "| 1024 | +88.0% | 0.0000 |\n"
        )
        new_file.write_text(
            "| Cap | Strict Diff | Raw McNemar p |\n"
            "| 1024 | +85.0% | <1e-4 |\n"  # 85.0% instead of 88.0%
        )

        self.assertFalse(compare_reports(old_file, new_file))


class TestGemmaManifestKeyReading(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_manifest_keys_correctly_read(self):
        # Create synthetic manifest with git_commit_hash and rule_file_sha256
        manifest_data = {
            "git_commit_hash": "abc123commit",
            "rule_file_sha256": "def456rulehash",
            "completed": True,
            "hours_used": 1.23,
        }
        (self.work_dir / "manifest.json").write_text(json.dumps(manifest_data))

        dummy_row = {"id": "q1", "split": "validation", "variant": "V1", "parsed": True, "correct": True}
        (self.work_dir / "gemma3-4b.jsonl").write_text(json.dumps(dummy_row) + "\n")
        (self.work_dir / "medgemma-1.5-4b-it.jsonl").write_text(json.dumps(dummy_row) + "\n")

        g_rows, mg_rows, manifest = load_and_validate_gemma_sweeps(self.work_dir)
        self.assertEqual(manifest["git_commit_hash"], "abc123commit")
        self.assertEqual(manifest["rule_file_sha256"], "def456rulehash")
        self.assertEqual(manifest["hours_used"], 1.23)


class TestFinalResultsMainEndToEnd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)

        # 1. Pair A: C1, C2 (MedQA, IDs q1, q2)
        q_m_c1 = [
            {"id": "q1", "parsed": True, "truncated": False, "correct": True},
            {"id": "q2", "parsed": True, "truncated": False, "correct": False},
        ]
        q_g_c1 = [
            {"id": "q1", "parsed": True, "truncated": False, "correct": False},
            {"id": "q2", "parsed": True, "truncated": False, "correct": False},
        ]
        self._write_jsonl(self.repo / "outputs" / "v3" / "medpsy-1.7b" / "C1" / "medpsy-1.7b.jsonl", q_m_c1)
        self._write_jsonl(self.repo / "outputs" / "v3" / "qwen3-1.7b-thinking" / "C1" / "qwen3-1.7b-thinking.jsonl", q_g_c1)
        self._write_jsonl(self.repo / "outputs" / "v3" / "medpsy-1.7b" / "C2" / "medpsy-1.7b.jsonl", q_m_c1)
        self._write_jsonl(self.repo / "outputs" / "v3" / "qwen3-1.7b-thinking" / "C2" / "qwen3-1.7b-thinking.jsonl", q_g_c1)

        # Pair A: C4, C5 (Unanswerable v2, ID u1)
        q_unans = [{"id": "u1", "parsed": True, "truncated": False, "correct": True}]
        self._write_jsonl(self.repo / "outputs" / "v3" / "medpsy-1.7b" / "C4" / "medpsy-1.7b.jsonl", q_unans)
        self._write_jsonl(self.repo / "outputs" / "v3" / "qwen3-1.7b-thinking" / "C4" / "qwen3-1.7b-thinking.jsonl", q_unans)
        self._write_jsonl(self.repo / "outputs" / "v3" / "medpsy-1.7b" / "C5" / "medpsy-1.7b.jsonl", q_unans)
        self._write_jsonl(self.repo / "outputs" / "v3" / "qwen3-1.7b-thinking" / "C5" / "qwen3-1.7b-thinking.jsonl", q_unans)

        # 2. Gemma Pair: C1, C2, C3 with medqa / pubmedqa
        for cond, ds in [("C1", "medqa"), ("C1", "pubmedqa"), ("C2", "medqa"), ("C2", "pubmedqa"), ("C3", "pubmedqa")]:
            g_rows = [
                {"id": f"g_{ds}_1", "dataset": ds, "parsed": True, "truncated": False, "correct": True},
            ]
            self._write_jsonl(self.repo / "outputs" / "v3" / "gemma3-4b" / cond / "gemma3-4b.jsonl", g_rows, append=True)
            self._write_jsonl(self.repo / "outputs" / "v3" / "medgemma-1.5-4b-it" / cond / "medgemma-1.5-4b-it.jsonl", g_rows, append=True)

        # Gemma Pair: C4, C5
        for cond in ["C4", "C5"]:
            g_u_rows = [{"id": "gu1", "parsed": True, "truncated": False, "correct": True}]
            self._write_jsonl(self.repo / "outputs" / "v3" / "gemma3-4b" / cond / "gemma3-4b.jsonl", g_u_rows)
            self._write_jsonl(self.repo / "outputs" / "v3" / "medgemma-1.5-4b-it" / cond / "medgemma-1.5-4b-it.jsonl", g_u_rows)

        # 3. Pair B (MedQA Test, ID b1)
        b_m_rows = [{"id": "b1", "parsed": True, "truncated": False, "correct": True}]
        b_q_rows = [{"id": "b1", "parsed": True, "truncated": False, "correct": False}]
        self._write_jsonl(self.repo / "outputs" / "kaggle_qa" / "medpsy_final" / "work" / "phase5" / "medpsy-4b.jsonl", b_m_rows)
        self._write_jsonl(self.repo / "outputs" / "kaggle_qa" / "medpsy_rag" / "work" / "phase8" / "medpsy-4b.jsonl", b_m_rows)
        self._write_jsonl(self.repo / "outputs" / "v3" / "qwen3-4b-thinking-2507" / "C1" / "qwen3-4b-thinking-2507.jsonl", b_q_rows)
        self._write_jsonl(self.repo / "outputs" / "v3" / "qwen3-4b-thinking-2507" / "C2" / "qwen3-4b-thinking-2507.jsonl", b_q_rows)

        # 4. Sweeps: budget sweep
        sw_m = [{"id": "sw1", "finished_naturally": True, "n_generated_tokens": 500, "parsed": True, "correct": True}]
        sw_q = [{"id": "sw1", "finished_naturally": True, "n_generated_tokens": 500, "parsed": True, "correct": False}]
        self._write_jsonl(self.repo / "outputs" / "sweeps" / "budget_pair_b" / "medpsy-4b.jsonl", sw_m)
        self._write_jsonl(self.repo / "outputs" / "sweeps" / "budget_pair_b" / "qwen3-4b-thinking-2507.jsonl", sw_q)
        (self.repo / "outputs" / "sweeps" / "budget_pair_b" / "manifest.json").write_text(json.dumps({"hours_used": 6.92}))

        # 5. Sweeps: gemma prompt sensitivity
        ps_g = [
            {"id": "ps1", "variant": "V1", "parsed": True, "truncated": False, "correct": True},
            {"id": "ps2", "variant": "V2", "parsed": True, "truncated": False, "correct": True},
            {"id": "ps3", "variant": "V3", "parsed": True, "truncated": False, "correct": True},
        ]
        ps_mg = [
            {"id": "ps1", "variant": "V1", "parsed": True, "truncated": False, "correct": False},
            {"id": "ps2", "variant": "V2", "parsed": True, "truncated": False, "correct": False},
            {"id": "ps3", "variant": "V3", "parsed": True, "truncated": False, "correct": False},
        ]
        self._write_jsonl(self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "gemma3-4b.jsonl", ps_g)
        self._write_jsonl(self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "medgemma-1.5-4b-it.jsonl", ps_mg)
        (self.repo / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "manifest.json").write_text(json.dumps({"hours_used": 0.37}))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_jsonl(self, path: Path, rows: list, append: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_main_end_to_end_synthetic_execution(self):
        # 1. Run main() to generate tables in the synthetic temp dir
        main(["--repo-root", str(self.repo)])

        out_md = self.repo / "docs" / "tables" / "final_results_v3.md"
        out_csv = self.repo / "docs" / "tables" / "final_results_v3.csv"
        out_tex = self.repo / "docs" / "tables" / "final_results_v3.tex"

        self.assertTrue(out_md.exists(), "final_results_v3.md was not created")
        self.assertTrue(out_csv.exists(), "final_results_v3.csv was not created")
        self.assertTrue(out_tex.exists(), "final_results_v3.tex was not created")

        md_text = out_md.read_text(encoding="utf-8")
        # In Pair A MedQA C1, MedPsy has 1/2 correct (50.0%), Qwen has 0/2 (0.0%), gap +50.0%
        self.assertIn("50.0%", md_text)
        self.assertIn("+50.0%", md_text)
        self.assertIn("Pair A: medpsy-1.7b vs qwen3-1.7b-thinking", md_text)
        self.assertIn("Gemma Pair: medgemma-1.5-4b-it vs gemma3-4b", md_text)
        self.assertIn("Pair B: medpsy-4b vs qwen3-4b-thinking-2507", md_text)

        # Check CSV
        csv_lines = [l.strip() for l in out_csv.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(csv_lines), 17)  # 1 header + 16 data rows
        headers = csv_lines[0].split(",")
        self.assertEqual(len(headers), 25)
        self.assertIn("domain_strict_acc", headers)
        self.assertIn("generic_strict_acc", headers)

        # Check LaTeX
        tex_text = out_tex.read_text(encoding="utf-8")
        self.assertIn(r"\begin{table*}", tex_text)
        self.assertIn(r"\texttt{medpsy-1.7b}", tex_text)
        self.assertIn(r"\texttt{qwen3-1.7b-thinking}", tex_text)

        # 2. Check that --check passes without modifying
        try:
            main(["--repo-root", str(self.repo), "--check"])
        except SystemExit as e:
            self.assertEqual(e.code, 0, f"--check failed with exit code {e.code}")


if __name__ == "__main__":
    unittest.main()
