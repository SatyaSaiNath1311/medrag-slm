"""tests/test_v3_datasets.py — Unit tests for v3 dataset filtering and condition question sets.

Verifies:
  1. Default dataset setting ("medqa,pubmedqa") reproduces existing behavior exactly (1000 questions for C1/C2).
  2. MedQA-only gives exactly 500 questions for C1 and C2 on the test split.
  3. C4 and C5 remain unaffected (140 questions).
  4. Unknown dataset name raises ValueError.
  5. Requesting condition C3 without pubmedqa in --datasets raises ValueError.
  6. argparse default of --datasets in qa_pipeline_v3 is 'medqa,pubmedqa'.
"""

import os
import unittest

from src.qa_pipeline_v3 import (
    find_question_files,
    get_condition_questions,
    parse_and_validate_datasets,
    read_jsonl,
    sample_pilot_questions,
)


class TestV3Datasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate test question files
        p1_test, _, u2_test, _ = find_question_files("work")
        if not os.path.exists(p1_test) or not os.path.exists(u2_test):
            raise unittest.SkipTest("Test question files not found in workspace.")
        p1_questions = read_jsonl(p1_test)
        cls.medqa_all = [q for q in p1_questions if q.get("dataset") == "medqa"]
        cls.pubmedqa_all = [q for q in p1_questions if q.get("dataset") == "pubmedqa"]
        cls.u2_questions = read_jsonl(u2_test)

    def test_default_datasets_unchanged(self):
        """Default 'medqa,pubmedqa' gives 1000 questions for C1/C2 (500 MedQA + 500 PubMedQA)."""
        datasets = parse_and_validate_datasets("medqa,pubmedqa", ["C1", "C2", "C3", "C4", "C5"])
        self.assertEqual(datasets, ["medqa", "pubmedqa"])

        q_c1 = get_condition_questions("C1", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c2 = get_condition_questions("C2", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c3 = get_condition_questions("C3", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c4 = get_condition_questions("C4", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c5 = get_condition_questions("C5", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)

        self.assertEqual(len(q_c1), 1000)
        self.assertEqual(len(q_c2), 1000)
        self.assertEqual(len(q_c3), 500)
        self.assertEqual(len(q_c4), 140)
        self.assertEqual(len(q_c5), 140)

        # C1 and C2 are composed of medqa_all followed by pubmedqa_all
        self.assertEqual(q_c1, self.medqa_all + self.pubmedqa_all)
        self.assertEqual(q_c2, self.medqa_all + self.pubmedqa_all)

    def test_medqa_only_gives_500_for_c1_and_c2(self):
        """MedQA-only gives exactly 500 questions for C1 and C2; C4/C5 remain unaffected."""
        datasets = parse_and_validate_datasets("medqa", ["C1", "C2", "C4", "C5"])
        self.assertEqual(datasets, ["medqa"])

        q_c1 = get_condition_questions("C1", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c2 = get_condition_questions("C2", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c4 = get_condition_questions("C4", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c5 = get_condition_questions("C5", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)

        self.assertEqual(len(q_c1), 500)
        self.assertEqual(len(q_c2), 500)
        self.assertTrue(all(q.get("dataset") == "medqa" for q in q_c1))
        self.assertTrue(all(q.get("dataset") == "medqa" for q in q_c2))

        # C4/C5 unaffected
        self.assertEqual(len(q_c4), 140)
        self.assertEqual(len(q_c5), 140)
        self.assertEqual(q_c4, self.u2_questions)
        self.assertEqual(q_c5, self.u2_questions)

    def test_pubmedqa_only_gives_500_for_c1_and_c2(self):
        """PubMedQA-only gives 500 questions for C1/C2 and allows C3."""
        datasets = parse_and_validate_datasets("pubmedqa", ["C1", "C2", "C3"])
        self.assertEqual(datasets, ["pubmedqa"])

        q_c1 = get_condition_questions("C1", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c2 = get_condition_questions("C2", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)
        q_c3 = get_condition_questions("C3", datasets, self.medqa_all, self.pubmedqa_all, self.u2_questions)

        self.assertEqual(len(q_c1), 500)
        self.assertEqual(len(q_c2), 500)
        self.assertEqual(len(q_c3), 500)
        self.assertTrue(all(q.get("dataset") == "pubmedqa" for q in q_c1))

    def test_unknown_dataset_raises(self):
        """Specifying unknown datasets or empty string must raise ValueError."""
        with self.assertRaises(ValueError):
            parse_and_validate_datasets("foo", ["C1"])
        with self.assertRaises(ValueError):
            parse_and_validate_datasets("medqa,unknown", ["C1"])
        with self.assertRaises(ValueError):
            parse_and_validate_datasets("", ["C1"])
        with self.assertRaises(ValueError):
            parse_and_validate_datasets("   ", ["C1"])

    def test_c3_without_pubmedqa_raises(self):
        """Requesting C3 without pubmedqa must raise ValueError."""
        with self.assertRaises(ValueError):
            parse_and_validate_datasets("medqa", ["C3"])
        with self.assertRaises(ValueError):
            parse_and_validate_datasets("medqa", ["C1", "C2", "C3"])

    def test_sample_pilot_with_single_dataset(self):
        """Pilot question sampling works correctly when questions contain a single dataset."""
        q_medqa = list(self.medqa_all)
        pilot_sample = sample_pilot_questions(q_medqa, "C1", n=20, seed=42)
        self.assertEqual(len(pilot_sample), 20)
        self.assertTrue(all(q.get("dataset") == "medqa" for q in pilot_sample))

    def test_datasets_argparse_default(self):
        """Verify argparse default for --datasets in qa_pipeline_v3 is 'medqa,pubmedqa'."""
        import ast
        from pathlib import Path
        src = Path("src/qa_pipeline_v3.py").read_text()
        tree = ast.parse(src)
        default_val = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if any(isinstance(arg, ast.Constant) and arg.value == "--datasets" for arg in node.args):
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            default_val = kw.value.value
        self.assertEqual(default_val, "medqa,pubmedqa")


if __name__ == "__main__":
    unittest.main()
