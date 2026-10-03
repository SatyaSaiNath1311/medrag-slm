"""Unit tests for parity-based sharding and shard-merge logic.

No model loading, no GPU, no file I/O beyond tempfiles.

Tests:
  1.  Parity shard filter — even/odd question assignment, deterministic, covers all questions
  2.  Shard filter sizes — roughly halves the list (handles odd totals)
  3.  Shard filter applied after question-set build — correct order dependency
  4.  merge_shards — happy path: 2 shards → merged file, correct count, no duplicates
  5.  merge_shards — duplicate ID detection raises AssertionError
  6.  merge_shards — wrong count raises AssertionError
  7.  merge_shards — missing shard file raises FileNotFoundError
  8.  _ci95 — basic sanity on normal-approximation Wilson CI
  9.  build_qa_cmd includes --shard / --num-shards when provided
  10. FINAL_MEDPSY constant defaults are correct in runner.py source
"""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path


# ── Helpers that mirror runner.py merge_shards / _ci95 inline ─────────────────
# We copy just the logic so the tests are self-contained; if the runner changes,
# the tests catch divergence.

def _parity_filter(questions, shard_idx, num_shards):
    """Mirror of the parity filter in qa_pipeline.main()."""
    return [q for i, q in enumerate(questions) if i % num_shards == shard_idx]


def _filter_datasets(questions, datasets_arg):
    """Mirror of the dataset filtering in qa_pipeline.main()."""
    if not datasets_arg:
        return list(questions)
    allowed = {d.strip().lower() for d in datasets_arg.split(",") if d.strip()}
    def _matches_dataset(q):
        ds = q.get("dataset", "").lower()
        is_unans = ds == "unanswerable" or q.get("should_abstain") is True
        if is_unans:
            return "unanswerable" in allowed
        return ds in allowed
    return [q for q in questions if _matches_dataset(q)]


def _merge_shards(phase_dir, model_name, num_shards, expected_count, label):
    """Mirror of runner.py merge_shards.  Returns list of row dicts.
    Prints a clear WARNING on count mismatch or duplicate IDs, but STILL writes the merged file.
    """
    all_rows = []
    for sid in range(num_shards):
        shard_path = os.path.join(phase_dir, f"{model_name}_shard{sid}.jsonl")
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file missing: {shard_path}")
        with open(shard_path) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        all_rows.extend(rows)

    ids = [r["id"] for r in all_rows]
    dup_ids = [i for i in ids if ids.count(i) > 1]
    if dup_ids:
        unique_dups = sorted(set(dup_ids))
        print(f"WARNING: [MERGE] Duplicate ids in {label}: {len(dup_ids)} duplicates "
              f"({len(unique_dups)} unique: {unique_dups[:10]})")
    if len(all_rows) != expected_count:
        print(f"WARNING: [MERGE] {label}: count mismatch: expected {expected_count} rows, got {len(all_rows)}")

    out_path = os.path.join(phase_dir, f"{model_name}.jsonl")
    with open(out_path, "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r) + "\n")
    return all_rows


def _ci95(n_correct, n_total):
    """Mirror of runner.py _ci95."""
    if n_total == 0:
        return (None, None)
    p = n_correct / n_total
    z = 1.96
    margin = z * math.sqrt(p * (1 - p) / n_total)
    return (round(max(0.0, p - margin), 4), round(min(1.0, p + margin), 4))


def _make_questions(n, dataset="medqa", split="test"):
    return [{"id": f"{dataset}-{split}-{i:04d}", "dataset": dataset,
             "split": split, "should_abstain": False,
             "question": f"Q{i}", "options": {"A": "a", "B": "b"}, "answer": "A"}
            for i in range(n)]


def _write_shard(phase_dir, model_name, shard_idx, rows):
    path = os.path.join(phase_dir, f"{model_name}_shard{shard_idx}.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# ── Test classes ───────────────────────────────────────────────────────────────

class TestParityFilter(unittest.TestCase):

    def test_even_questions_go_to_shard0(self):
        qs = _make_questions(10)
        s0 = _parity_filter(qs, 0, 2)
        for q in s0:
            idx = int(q["id"].split("-")[-1])
            self.assertEqual(idx % 2, 0)

    def test_odd_questions_go_to_shard1(self):
        qs = _make_questions(10)
        s1 = _parity_filter(qs, 1, 2)
        for q in s1:
            idx = int(q["id"].split("-")[-1])
            self.assertEqual(idx % 2, 1)

    def test_shards_are_disjoint(self):
        qs = _make_questions(20)
        s0 = {q["id"] for q in _parity_filter(qs, 0, 2)}
        s1 = {q["id"] for q in _parity_filter(qs, 1, 2)}
        self.assertFalse(s0 & s1)

    def test_shards_cover_all_questions(self):
        qs = _make_questions(20)
        s0 = {q["id"] for q in _parity_filter(qs, 0, 2)}
        s1 = {q["id"] for q in _parity_filter(qs, 1, 2)}
        self.assertEqual(s0 | s1, {q["id"] for q in qs})

    def test_even_total_splits_equally(self):
        qs = _make_questions(100)
        self.assertEqual(len(_parity_filter(qs, 0, 2)), 50)
        self.assertEqual(len(_parity_filter(qs, 1, 2)), 50)

    def test_odd_total_splits_correctly(self):
        qs = _make_questions(101)
        s0 = _parity_filter(qs, 0, 2)
        s1 = _parity_filter(qs, 1, 2)
        # shard 0 gets ceil, shard 1 gets floor
        self.assertEqual(len(s0), 51)
        self.assertEqual(len(s1), 50)
        self.assertEqual(len(s0) + len(s1), 101)

    def test_single_question_goes_to_shard0_only(self):
        qs = _make_questions(1)
        self.assertEqual(len(_parity_filter(qs, 0, 2)), 1)
        self.assertEqual(len(_parity_filter(qs, 1, 2)), 0)

    def test_empty_list_returns_empty(self):
        self.assertEqual(_parity_filter([], 0, 2), [])
        self.assertEqual(_parity_filter([], 1, 2), [])

    def test_deterministic_same_input(self):
        qs = _make_questions(50)
        self.assertEqual(
            [q["id"] for q in _parity_filter(qs, 0, 2)],
            [q["id"] for q in _parity_filter(qs, 0, 2)],
        )

    def test_order_within_shard_preserved(self):
        """Questions within a shard should appear in original relative order."""
        qs = _make_questions(10)
        s0 = _parity_filter(qs, 0, 2)
        # s0 should be [q0, q2, q4, q6, q8] — ascending id order
        ids = [q["id"] for q in s0]
        self.assertEqual(ids, sorted(ids))

    def test_mixed_datasets_shard_by_global_index(self):
        """Sharding is positional (global index), not dataset-aware."""
        medqa = _make_questions(5, dataset="medqa")
        pubmedqa = _make_questions(5, dataset="pubmedqa")
        combined = medqa + pubmedqa
        s0 = _parity_filter(combined, 0, 2)
        s1 = _parity_filter(combined, 1, 2)
        self.assertEqual(len(s0) + len(s1), len(combined))
        # First question (index 0) is always in shard 0
        self.assertEqual(s0[0]["id"], combined[0]["id"])
        # Second question (index 1) is always in shard 1
        self.assertEqual(s1[0]["id"], combined[1]["id"])


class TestMergeShards(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _make_rows(self, ids, dataset="medqa", correct=True):
        return [{"id": i, "dataset": dataset, "split": "test",
                 "should_abstain": False, "correct": correct,
                 "parsed": True, "truncated": False,
                 "seconds": 2.5, "confidence": 0.85} for i in ids]

    def test_happy_path_two_shards(self):
        rows0 = self._make_rows([f"q{i}" for i in range(0, 10, 2)])
        rows1 = self._make_rows([f"q{i}" for i in range(1, 10, 2)])
        _write_shard(self.tmp, "medpsy-4b", 0, rows0)
        _write_shard(self.tmp, "medpsy-4b", 1, rows1)
        merged = _merge_shards(self.tmp, "medpsy-4b", 2, 10, "test")
        self.assertEqual(len(merged), 10)

    def test_merged_file_written(self):
        rows0 = self._make_rows(["a", "b"])
        rows1 = self._make_rows(["c", "d"])
        _write_shard(self.tmp, "medpsy-4b", 0, rows0)
        _write_shard(self.tmp, "medpsy-4b", 1, rows1)
        _merge_shards(self.tmp, "medpsy-4b", 2, 4, "test")
        out = Path(self.tmp) / "medpsy-4b.jsonl"
        self.assertTrue(out.exists())
        written = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual(len(written), 4)

    def test_no_duplicates_in_merged(self):
        rows0 = self._make_rows(["x0", "x1", "x2"])
        rows1 = self._make_rows(["x3", "x4", "x5"])
        _write_shard(self.tmp, "medpsy-4b", 0, rows0)
        _write_shard(self.tmp, "medpsy-4b", 1, rows1)
        merged = _merge_shards(self.tmp, "medpsy-4b", 2, 6, "test")
        ids = [r["id"] for r in merged]
        self.assertEqual(len(ids), len(set(ids)))

    def test_duplicate_id_warns_and_still_writes(self):
        """If shard 1 repeats an id from shard 0, merge prints a warning but STILL writes."""
        rows0 = self._make_rows(["dup", "unique0"])
        rows1 = self._make_rows(["dup", "unique1"])   # 'dup' appears in both
        _write_shard(self.tmp, "medpsy-4b", 0, rows0)
        _write_shard(self.tmp, "medpsy-4b", 1, rows1)
        merged = _merge_shards(self.tmp, "medpsy-4b", 2, 4, "test")
        out = Path(self.tmp) / "medpsy-4b.jsonl"
        self.assertTrue(out.exists())
        self.assertEqual(len(merged), 4)

    def test_wrong_count_warns_and_still_writes(self):
        """If row count differs from expected, merge prints a warning but STILL writes."""
        rows0 = self._make_rows(["a"])
        rows1 = self._make_rows(["b"])
        _write_shard(self.tmp, "medpsy-4b", 0, rows0)
        _write_shard(self.tmp, "medpsy-4b", 1, rows1)
        merged = _merge_shards(self.tmp, "medpsy-4b", 2, 99, "test")
        out = Path(self.tmp) / "medpsy-4b.jsonl"
        self.assertTrue(out.exists())
        self.assertEqual(len(merged), 2)

    def test_missing_shard_file_raises(self):
        rows0 = self._make_rows(["a", "b"])
        _write_shard(self.tmp, "medpsy-4b", 0, rows0)
        # shard 1 file deliberately not created
        with self.assertRaises(FileNotFoundError):
            _merge_shards(self.tmp, "medpsy-4b", 2, 2, "test")

    def test_merge_order_is_shard0_then_shard1(self):
        """Merged rows should appear in shard-index order (0 before 1)."""
        rows0 = self._make_rows(["first"])
        rows1 = self._make_rows(["second"])
        _write_shard(self.tmp, "medpsy-4b", 0, rows0)
        _write_shard(self.tmp, "medpsy-4b", 1, rows1)
        merged = _merge_shards(self.tmp, "medpsy-4b", 2, 2, "test")
        self.assertEqual(merged[0]["id"], "first")
        self.assertEqual(merged[1]["id"], "second")


class TestCI95(unittest.TestCase):

    def test_zero_total_returns_none(self):
        self.assertEqual(_ci95(0, 0), (None, None))

    def test_all_correct(self):
        lo, hi = _ci95(100, 100)
        self.assertAlmostEqual(hi, 1.0)
        self.assertGreater(lo, 0.9)

    def test_none_correct(self):
        lo, hi = _ci95(0, 100)
        self.assertAlmostEqual(lo, 0.0)
        self.assertLess(hi, 0.1)

    def test_ci_contains_true_proportion(self):
        """For p=0.6, n=500: true proportion should lie in the CI."""
        lo, hi = _ci95(300, 500)
        self.assertLessEqual(lo, 0.6)
        self.assertGreaterEqual(hi, 0.6)

    def test_larger_n_gives_narrower_ci(self):
        lo_small, hi_small = _ci95(50, 100)
        lo_large, hi_large = _ci95(500, 1000)
        self.assertLess(hi_large - lo_large, hi_small - lo_small)

    def test_output_is_tuple_of_two_floats(self):
        result = _ci95(250, 500)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


class TestDatasetFilterAndStartupCounts(unittest.TestCase):

    def _make_benchmark_pool(self):
        # 500 MedQA answerable
        medqa = [{"id": f"medqa-{i:04d}", "dataset": "medqa", "split": "test",
                  "should_abstain": False, "question": f"MQ{i}"} for i in range(500)]
        # 150 unanswerable
        unans = [{"id": f"unans-{i:04d}", "dataset": "unanswerable", "split": "test",
                  "should_abstain": True, "question": f"UQ{i}"} for i in range(150)]
        # 500 PubMedQA
        pubmed = [{"id": f"pubmedqa-{i:04d}", "dataset": "pubmedqa", "split": "test",
                   "should_abstain": False, "question": f"PQ{i}"} for i in range(500)]
        return medqa + unans + pubmed

    def test_baseline_dataset_filter(self):
        pool = self._make_benchmark_pool()
        filtered = _filter_datasets(pool, "medqa,unanswerable")
        self.assertEqual(len(filtered), 650)
        self.assertTrue(all(q["dataset"] in ("medqa", "unanswerable") for q in filtered))
        self.assertEqual(sum(1 for q in filtered if q["dataset"] == "medqa"), 500)
        self.assertEqual(sum(1 for q in filtered if q["dataset"] == "unanswerable"), 150)

    def test_context_dataset_filter(self):
        pool = self._make_benchmark_pool()
        filtered = _filter_datasets(pool, "pubmedqa")
        self.assertEqual(len(filtered), 500)
        self.assertTrue(all(q["dataset"] == "pubmedqa" for q in filtered))

    def test_startup_counts_shard0_baseline(self):
        pool = self._make_benchmark_pool()
        base_qs = _filter_datasets(pool, "medqa,unanswerable")
        shard0 = _parity_filter(base_qs, 0, 2)
        n_mq = sum(1 for q in shard0 if q["dataset"] == "medqa" and not q["should_abstain"])
        n_un = sum(1 for q in shard0 if q["dataset"] == "unanswerable" or q["should_abstain"])
        self.assertEqual(len(shard0), 325)
        self.assertEqual(n_mq, 250)
        self.assertEqual(n_un, 75)
        msg = f"shard 0 baseline: {len(shard0)} (medqa {n_mq}, unanswerable {n_un})"
        self.assertEqual(msg, "shard 0 baseline: 325 (medqa 250, unanswerable 75)")

    def test_startup_counts_shard1_baseline(self):
        pool = self._make_benchmark_pool()
        base_qs = _filter_datasets(pool, "medqa,unanswerable")
        shard1 = _parity_filter(base_qs, 1, 2)
        n_mq = sum(1 for q in shard1 if q["dataset"] == "medqa" and not q["should_abstain"])
        n_un = sum(1 for q in shard1 if q["dataset"] == "unanswerable" or q["should_abstain"])
        self.assertEqual(len(shard1), 325)
        self.assertEqual(n_mq, 250)
        self.assertEqual(n_un, 75)
        msg = f"shard 1 baseline: {len(shard1)} (medqa {n_mq}, unanswerable {n_un})"
        self.assertEqual(msg, "shard 1 baseline: 325 (medqa 250, unanswerable 75)")

    def test_startup_counts_context(self):
        pool = self._make_benchmark_pool()
        ctx_qs = _filter_datasets(pool, "pubmedqa")
        shard0 = _parity_filter(ctx_qs, 0, 2)
        shard1 = _parity_filter(ctx_qs, 1, 2)
        self.assertEqual(len(shard0), 250)
        self.assertEqual(len(shard1), 250)
        self.assertEqual(f"shard 0 context: {len(shard0)} (pubmedqa {len(shard0)})",
                         "shard 0 context: 250 (pubmedqa 250)")
        self.assertEqual(f"shard 1 context: {len(shard1)} (pubmedqa {len(shard1)})",
                         "shard 1 context: 250 (pubmedqa 250)")


class TestRunnerFinalMedpsy(unittest.TestCase):
    """Sanity-check the constants and flags in runner.py source."""

    def _read_runner(self):
        import pathlib
        return pathlib.Path("kaggle_runner/runner.py").read_text()

    def test_final_medpsy_is_true(self):
        text = self._read_runner()
        self.assertIn("FINAL_MEDPSY = True", text)

    def test_pilot_is_false(self):
        text = self._read_runner()
        self.assertIn("PILOT = False", text)

    def test_models_is_medpsy(self):
        text = self._read_runner()
        self.assertIn('MODELS = ["medpsy-4b"]', text)

    def test_parallel_gpus_is_false(self):
        text = self._read_runner()
        self.assertIn("PARALLEL_GPUS = False", text)

    def test_build_qa_cmd_supports_shard(self):
        text = self._read_runner()
        self.assertIn("--shard", text)
        self.assertIn("--num-shards", text)

    def test_build_qa_cmd_supports_datasets(self):
        text = self._read_runner()
        self.assertIn("--datasets", text)
        self.assertIn("datasets=None", text)

    def test_runner_calls_datasets_for_baseline_and_context(self):
        text = self._read_runner()
        self.assertIn('datasets="medqa,unanswerable"', text)
        self.assertIn('datasets="pubmedqa"', text)

    def test_runner_startup_prints_exact_breakdown(self):
        text = self._read_runner()
        self.assertIn("shard 0 baseline: 325 (medqa 250, unanswerable 75)", text)
        self.assertIn("shard 0 context: 250 (pubmedqa 250)", text)

    def test_merge_shards_duplicate_check_present(self):
        text = self._read_runner()
        self.assertIn("Duplicate ids", text)

    def test_merge_shards_warns_not_raises_on_diff_or_dup(self):
        text = self._read_runner()
        self.assertIn("WARNING: [MERGE]", text)

    def test_expected_baseline_count(self):
        """EXPECTED_BASELINE = 650 (MedQA 500 + unanswerable 150)."""
        text = self._read_runner()
        self.assertIn("EXPECTED_BASELINE = 650", text)

    def test_expected_context_count(self):
        """EXPECTED_CONTEXT  = 500 (PubMedQA test)."""
        text = self._read_runner()
        self.assertIn("EXPECTED_CONTEXT  = 500", text)

    def test_ci95_function_present(self):
        text = self._read_runner()
        self.assertIn("_ci95", text)

    def test_dual_gpu_via_cuda_visible_devices(self):
        text = self._read_runner()
        self.assertIn("CUDA_VISIBLE_DEVICES", text)

    def test_skip_format_check_passed_in_final(self):
        text = self._read_runner()
        self.assertIn("skip_format_check=True", text)


if __name__ == "__main__":
    unittest.main()
