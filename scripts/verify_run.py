#!/usr/bin/env python3
"""
scripts/verify_run.py — Independent Integrity and Metric Verifier for GPU Runs.

Integrity Gates (PASS/FAIL criteria):
  1. Target file existence and non-empty status
  2. Cryptographic SHA-256 match of frozen input/benchmark datasets
  3. Row counts: exact expected total, validation, and test counts
  4. Unique IDs: exactly 0 duplicate IDs
  5. Coverage: exactly 0 missing IDs and 0 unexpected IDs against official gold splits
  6. Gold alignment: exactly 0 gold mismatches (gold answers, should_abstain flags)

Reported Metrics (REPORT ONLY — NOT GATED):
  - Strict accuracy (PRIMARY metric: parsed, not truncated, correct)
  - With-fallback accuracy (SECONDARY metric)
  - Truncation rate and parse rate (never gated; reported for documentation)
  - Abstention recall and false abstention rate (for unanswerable benchmarks)
  - Citation counts (for RAG modes)
  - Latency (s/question)

Usage:
  python3 scripts/verify_run.py --run medpsy_unans_v2_baseline
  python3 scripts/verify_run.py --run medpsy_pubmedqa_baseline
  python3 scripts/verify_run.py --run medpsy_pubmedqa_rag
  python3 scripts/verify_run.py --run all_stages
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.answer_validity import (
    is_strict_correct,
    is_fallback_correct,
    is_truncated,
    is_parsed,
    strict_accuracy,
    fallback_accuracy,
    truncation_rate,
    parse_rate,
)

# Canonical frozen hashes
FROZEN_HASHES = {
    "unans_v2_test": "a0e45676dced360b2ac64e8fc33dbb49782c4fd9022e93190e633c9d66396078",
    "phase1_test": "fe7fe3fb88c77c4ffc6574c8716ec0b3f4787fbfaf2c22e2eb5741ac040cc34a",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt JSON at line {i} in {path}: {exc}")
    return rows


def check_frozen_hash(path: Path, expected: Optional[str], label: str) -> bool:
    print(f"\n--- Check: Frozen Hash Audit ({label}) ---")
    if not path.exists():
        print(f"  [FAIL] Frozen reference file missing: {path}")
        return False
    actual = compute_sha256(path)
    print(f"  File     : {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"  Actual   : {actual}")
    if expected:
        print(f"  Expected : {expected}")
        if actual != expected:
            print(f"  [FAIL] SHA-256 mismatch for {label}!")
            return False
        print(f"  [PASS] Cryptographic SHA-256 match verified.")
        return True
    else:
        print(f"  [INFO] Recorded hash: {actual} (no static expected string provided)")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# STAGE A: MedPsy-4B unanswerable_v2 Baseline
# ══════════════════════════════════════════════════════════════════════════════

def verify_unans_v2_baseline(target_file: Path, target_model: str = "medpsy-4b", stage_label: str = "Unanswerable v2 Run") -> bool:
    print("\n" + "=" * 80)
    print(f"VERIFIER: {stage_label} ({target_model})")
    print("=" * 80)

    # 1. File existence
    if not target_file.exists():
        print(f"[FAIL] Target output file does not exist: {target_file}")
        return False

    file_size_kb = target_file.stat().st_size / 1024.0
    print(f"File Path        : {target_file.relative_to(REPO_ROOT) if target_file.is_relative_to(REPO_ROOT) else target_file}")
    print(f"File Size        : {file_size_kb:.1f} KB")

    rows = load_jsonl(target_file)
    n_total = len(rows)
    print(f"Total Rows Loaded: {n_total}")

    # Config Echo
    models_found = sorted(set(r.get("model") for r in rows if "model" in r))
    modes_found = sorted(set(r.get("mode") for r in rows if "mode" in r))
    splits_found = sorted(set(r.get("split") for r in rows if "split" in r))
    print(f"Config Echo      : model={models_found}, mode={modes_found}, splits={splits_found}")

    passed = True

    # 2. Frozen Hashes Check
    val_gold_path = REPO_ROOT / "data" / "unanswerable_v2" / "val.jsonl"
    test_gold_path = REPO_ROOT / "data" / "unanswerable_v2" / "test.jsonl"
    val_sha_path = REPO_ROOT / "data" / "unanswerable_v2" / "val.sha256"

    expected_val_sha = None
    if val_sha_path.exists():
        expected_val_sha = val_sha_path.read_text().strip().split()[0]
    else:
        manifest_path = REPO_ROOT / "audit" / "frozen_hashes.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text()).get("files", {})
                expected_val_sha = manifest.get("data/unanswerable_v2/val.jsonl", {}).get("sha256")
            except Exception:
                pass

    if not check_frozen_hash(test_gold_path, FROZEN_HASHES["unans_v2_test"], "unanswerable_v2 test.jsonl"):
        passed = False
    if not check_frozen_hash(val_gold_path, expected_val_sha, "unanswerable_v2 val.jsonl"):
        passed = False

    # 3. Gold Ground Truth
    val_gold = {r["id"]: r for r in load_jsonl(val_gold_path)}
    test_gold = {r["id"]: r for r in load_jsonl(test_gold_path)}
    all_gold = {**val_gold, **test_gold}
    print(f"\nGold Ground Truth: {len(val_gold)} validation items, {len(test_gold)} test items (total {len(all_gold)})")

    # 4. Row Counts Verification
    val_rows = [r for r in rows if r.get("split") == "validation"]
    test_rows = [r for r in rows if r.get("split") == "test"]

    print("\n--- Gate 1: Row Counts ---")
    if len(val_rows) == 260 and len(test_rows) == 140 and n_total == 400:
        print(f"  [PASS] Expected 400 total rows (260 val, 140 test). Got: {n_total} (val={len(val_rows)}, test={len(test_rows)})")
    else:
        print(f"  [FAIL] Row count mismatch! Expected 400 (260 val, 140 test), got: {n_total} (val={len(val_rows)}, test={len(test_rows)})")
        passed = False

    # 5. Unique IDs Verification
    print("\n--- Gate 2: Unique IDs & Duplicate Detection ---")
    row_ids = [r.get("id") for r in rows]
    id_counts = Counter(row_ids)
    duplicates = [qid for qid, cnt in id_counts.items() if cnt > 1]
    if not duplicates:
        print(f"  [PASS] All {len(row_ids)} IDs are unique. 0 duplicates.")
    else:
        print(f"  [FAIL] Found {len(duplicates)} duplicate IDs: {duplicates[:5]}")
        passed = False

    # 6. Missing / Unexpected IDs Verification
    print("\n--- Gate 3: Missing & Unexpected IDs against Master Benchmark ---")
    output_id_set = set(row_ids)
    gold_id_set = set(all_gold.keys())
    missing_from_output = gold_id_set - output_id_set
    unexpected_in_output = output_id_set - gold_id_set

    if not missing_from_output and not unexpected_in_output:
        print(f"  [PASS] 0 missing IDs, 0 unexpected IDs. Exact 1-to-1 match with canonical unanswerable_v2.")
    else:
        if missing_from_output:
            print(f"  [FAIL] Missing {len(missing_from_output)} IDs from output: {list(missing_from_output)[:5]}")
        if unexpected_in_output:
            print(f"  [FAIL] Unexpected {len(unexpected_in_output)} IDs in output: {list(unexpected_in_output)[:5]}")
        passed = False

    # 7. Gold Alignment Verification
    print("\n--- Gate 4: Gold Mismatches & Metadata Alignment ---")
    gold_mismatches = 0
    for r in rows:
        qid = r.get("id")
        if qid not in all_gold:
            continue
        g = all_gold[qid]
        if r.get("should_abstain") != g.get("should_abstain"):
            gold_mismatches += 1
            if gold_mismatches <= 3:
                print(f"    Mismatch in should_abstain for {qid}: output={r.get('should_abstain')}, gold={g.get('should_abstain')}")
        g_ans = g.get("answer")
        r_gold = r.get("gold") if "gold" in r else r.get("answer")
        if g_ans is not None and r_gold is not None and r_gold != g_ans:
            gold_mismatches += 1
            if gold_mismatches <= 3:
                print(f"    Mismatch in gold answer for {qid}: output={r_gold}, gold={g_ans}")

    if gold_mismatches == 0:
        print(f"  [PASS] 0 gold mismatches across all {n_total} rows.")
    else:
        print(f"  [FAIL] Found {gold_mismatches} gold alignment mismatches!")
        passed = False

    # 8. Generation Metrics & Quality (REPORT ONLY — NOT GATED)
    print("\n--- REPORTED METRICS (Informational Only — Not Gated) ---")
    trunc_cnt = sum(1 for r in rows if is_truncated(r))
    parsed_cnt = sum(1 for r in rows if is_parsed(r))
    trunc_pct = (trunc_cnt / n_total) * 100 if n_total else 0.0
    parsed_pct = (parsed_cnt / n_total) * 100 if n_total else 0.0

    print(f"  Parse Rate       : {parsed_cnt}/{n_total} ({parsed_pct:.1f}%) [Reported]")
    print(f"  Truncation Rate  : {trunc_cnt}/{n_total} ({trunc_pct:.1f}%) [Reported]")

    controls = [r for r in rows if not r.get("should_abstain")]
    unanswerables = [r for r in rows if r.get("should_abstain")]

    s_acc_ctrl = strict_accuracy(controls)
    f_acc_ctrl = fallback_accuracy(controls)
    print(f"\n  Answerable Controls (n={len(controls)}: 60 val, 40 test):")
    print(f"    Strict Accuracy (PRIMARY)       : {s_acc_ctrl:.4f} ({sum(1 for r in controls if is_strict_correct(r))}/{len(controls)})")
    print(f"    With-Fallback Accuracy (SECOND) : {f_acc_ctrl:.4f} ({sum(1 for r in controls if is_fallback_correct(r))}/{len(controls)})")

    abstained_cnt = sum(1 for r in unanswerables if r.get("abstained") or r.get("pred") in ("ABSTAIN", None))
    false_abstained_cnt = sum(1 for r in controls if r.get("abstained") or r.get("pred") in ("ABSTAIN", None))
    abs_recall = (abstained_cnt / len(unanswerables)) * 100 if unanswerables else 0.0
    false_abs_rate = (false_abstained_cnt / len(controls)) * 100 if controls else 0.0

    print(f"\n  Prompt-Level Abstention Flag (row-flag only, not conformal abstention metric):")
    print(f"    Row-level Abstain Count (Unans) : {abstained_cnt}/{len(unanswerables)} ({abs_recall:.1f}%) [row-flag only; conformal recall computed post-hoc by src/analysis_unanswerable_v3.py]")
    print(f"    Control False Abstention Rate   : {false_abstained_cnt}/{len(controls)} ({false_abs_rate:.1f}%) [row-flag only]")

    cites_cnt = sum(1 for r in rows if r.get("citations") and len(r.get("citations")) > 0)
    if any(r.get("mode") == "rag" for r in rows):
        print(f"    Citations Present Rate          : {cites_cnt}/{n_total} ({100.0 * cites_cnt / max(n_total, 1):.1f}%)")

    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    if secs:
        avg_s = sum(secs) / len(secs)
        print(f"\n  Inference Speed  : {avg_s:.2f} s/question (total {sum(secs)/3600.0:.2f} GPU-hours)")

    print("\n" + "=" * 80)
    if passed:
        print(f"{stage_label.upper()} RESULT: [PASS] All integrity gates passed. Output verified.")
    else:
        print(f"{stage_label.upper()} RESULT: [FAIL] One or more integrity gates failed.")
    print("=" * 80 + "\n")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# STAGE B: MedPsy-4B PubMedQA Closed-Book Baseline
# ══════════════════════════════════════════════════════════════════════════════

def verify_pubmedqa_baseline(target_file: Path, target_model: str = "medpsy-4b") -> bool:
    print("\n" + "=" * 80)
    print(f"VERIFIER: STAGE B - PubMedQA Closed-Book Baseline ({target_model})")
    print("=" * 80)

    if not target_file.exists():
        print(f"[FAIL] Target output file does not exist: {target_file}")
        return False

    file_size_kb = target_file.stat().st_size / 1024.0
    print(f"File Path        : {target_file.relative_to(REPO_ROOT) if target_file.is_relative_to(REPO_ROOT) else target_file}")
    print(f"File Size        : {file_size_kb:.1f} KB")

    rows = load_jsonl(target_file)
    n_total = len(rows)
    print(f"Total Rows Loaded: {n_total}")

    models_found = sorted(set(r.get("model") for r in rows if "model" in r))
    modes_found = sorted(set(r.get("mode") for r in rows if "mode" in r))
    splits_found = sorted(set(r.get("split") for r in rows if "split" in r))
    print(f"Config Echo      : model={models_found}, mode={modes_found}, splits={splits_found}")

    passed = True

    # 1. Frozen Hash Check
    expected_p1_sha = FROZEN_HASHES.get("phase1_test")
    manifest_path = REPO_ROOT / "audit" / "frozen_hashes.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text()).get("files", {})
            if "outputs/kaggle_build/work/phase1/test.jsonl" in manifest:
                expected_p1_sha = manifest["outputs/kaggle_build/work/phase1/test.jsonl"]["sha256"]
        except Exception:
            pass

    ref_phase1_candidates = [
        REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase1" / "test.jsonl",
        Path("outputs/kaggle_build/work/phase1/test.jsonl").resolve(),
    ]
    ref_phase1_path = next((p for p in ref_phase1_candidates if p.exists()), ref_phase1_candidates[0])

    if not check_frozen_hash(ref_phase1_path, expected_p1_sha, "phase1/test.jsonl (frozen reference)"):
        passed = False

    # Also confirm hash of execution phase1/test.jsonl if present in run directory
    exec_p1_candidates = [
        REPO_ROOT / "work" / "work_pubmedqa_baseline" / "phase1" / "test.jsonl",
        REPO_ROOT / "work" / "phase1" / "test.jsonl",
        Path("/kaggle/working/work/work_pubmedqa_baseline/phase1/test.jsonl"),
        Path("/kaggle/working/work/phase1/test.jsonl"),
    ]
    for ep in exec_p1_candidates:
        if ep.exists():
            if not check_frozen_hash(ep, expected_p1_sha, f"phase1/test.jsonl (execution file at {ep})"):
                passed = False
            break

    # 2. Reference PubMedQA Data
    p1_rows = load_jsonl(ref_phase1_path)
    pm_ref = {r["id"]: r for r in p1_rows if r.get("dataset") == "pubmedqa"}
    print(f"\nGold Ground Truth: {len(pm_ref)} PubMedQA test items extracted from reference test.jsonl")

    # 3. Row Counts Verification
    print("\n--- Gate 1: Row Counts ---")
    if n_total == 500 and all(r.get("split") == "test" for r in rows) and all(r.get("dataset") == "pubmedqa" for r in rows):
        print(f"  [PASS] Expected exactly 500 PubMedQA test rows. Got: {n_total}")
    else:
        print(f"  [FAIL] Row count mismatch! Expected 500 test PubMedQA rows, got: {n_total}")
        passed = False

    # 4. Unique IDs Verification
    print("\n--- Gate 2: Unique IDs & Duplicate Detection ---")
    row_ids = [r.get("id") for r in rows]
    id_counts = Counter(row_ids)
    duplicates = [qid for qid, cnt in id_counts.items() if cnt > 1]
    if not duplicates:
        print(f"  [PASS] All {len(row_ids)} IDs are unique. 0 duplicates.")
    else:
        print(f"  [FAIL] Found {len(duplicates)} duplicate IDs: {duplicates[:5]}")
        passed = False

    # 5. Missing / Unexpected IDs Verification
    print("\n--- Gate 3: Missing & Unexpected IDs against Master Benchmark ---")
    output_id_set = set(row_ids)
    gold_id_set = set(pm_ref.keys())
    missing_from_output = gold_id_set - output_id_set
    unexpected_in_output = output_id_set - gold_id_set

    if not missing_from_output and not unexpected_in_output:
        print(f"  [PASS] 0 missing IDs, 0 unexpected IDs. Exact 1-to-1 match with canonical PubMedQA test.")
    else:
        if missing_from_output:
            print(f"  [FAIL] Missing {len(missing_from_output)} IDs from output: {list(missing_from_output)[:5]}")
        if unexpected_in_output:
            print(f"  [FAIL] Unexpected {len(unexpected_in_output)} IDs in output: {list(unexpected_in_output)[:5]}")
        passed = False

    # 6. Gold Alignment Verification
    print("\n--- Gate 4: Gold Mismatches & Metadata Alignment ---")
    gold_mismatches = 0
    for r in rows:
        qid = r.get("id")
        if qid not in pm_ref:
            continue
        g = pm_ref[qid]
        g_ans = g.get("answer")
        r_gold = r.get("gold") if "gold" in r else r.get("answer")
        if g_ans is not None and r_gold is not None and r_gold != g_ans:
            gold_mismatches += 1
            if gold_mismatches <= 3:
                print(f"    Mismatch in gold answer for {qid}: output={r_gold}, gold={g_ans}")

    if gold_mismatches == 0:
        print(f"  [PASS] 0 gold mismatches across all {n_total} rows.")
    else:
        print(f"  [FAIL] Found {gold_mismatches} gold alignment mismatches!")
        passed = False

    # 7. Generation Metrics (REPORT ONLY — NOT GATED)
    print("\n--- REPORTED METRICS (Informational Only — Not Gated) ---")
    trunc_cnt = sum(1 for r in rows if is_truncated(r))
    parsed_cnt = sum(1 for r in rows if is_parsed(r))
    trunc_pct = (trunc_cnt / n_total) * 100 if n_total else 0.0
    parsed_pct = (parsed_cnt / n_total) * 100 if n_total else 0.0

    s_acc = strict_accuracy(rows)
    f_acc = fallback_accuracy(rows)

    print(f"  Parse Rate                      : {parsed_cnt}/{n_total} ({parsed_pct:.1f}%) [Reported]")
    print(f"  Truncation Rate                 : {trunc_cnt}/{n_total} ({trunc_pct:.1f}%) [Reported]")
    print(f"  Strict Accuracy (PRIMARY)       : {s_acc:.4f} ({sum(1 for r in rows if is_strict_correct(r))}/{n_total})")
    print(f"  With-Fallback Accuracy (SECOND) : {f_acc:.4f} ({sum(1 for r in rows if is_fallback_correct(r))}/{n_total})")

    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    if secs:
        avg_s = sum(secs) / len(secs)
        print(f"\n  Inference Speed  : {avg_s:.2f} s/question (total {sum(secs)/3600.0:.2f} GPU-hours)")

    print("\n" + "=" * 80)
    if passed:
        print("STAGE B RESULT: [PASS] All integrity gates passed. Output verified.")
    else:
        print("STAGE B RESULT: [FAIL] One or more integrity gates failed.")
    print("=" * 80 + "\n")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# STAGE C: MedPsy-4B PubMedQA Full RAG
# ══════════════════════════════════════════════════════════════════════════════

def verify_pubmedqa_rag(target_file: Path, target_model: str = "medpsy-4b") -> bool:
    print("\n" + "=" * 80)
    print(f"VERIFIER: STAGE C - PubMedQA Full RAG ({target_model})")
    print("=" * 80)

    if not target_file.exists():
        print(f"[FAIL] Target output file does not exist: {target_file}")
        return False

    file_size_kb = target_file.stat().st_size / 1024.0
    print(f"File Path        : {target_file.relative_to(REPO_ROOT) if target_file.is_relative_to(REPO_ROOT) else target_file}")
    print(f"File Size        : {file_size_kb:.1f} KB")

    rows = load_jsonl(target_file)
    n_total = len(rows)
    print(f"Total Rows Loaded: {n_total}")

    models_found = sorted(set(r.get("model") for r in rows if "model" in r))
    modes_found = sorted(set(r.get("mode") for r in rows if "mode" in r))
    splits_found = sorted(set(r.get("split") for r in rows if "split" in r))
    print(f"Config Echo      : model={models_found}, mode={modes_found}, splits={splits_found}")

    passed = True

    # 1. Frozen Hash Check
    expected_p1_sha = FROZEN_HASHES.get("phase1_test")
    manifest_path = REPO_ROOT / "audit" / "frozen_hashes.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text()).get("files", {})
            if "outputs/kaggle_build/work/phase1/test.jsonl" in manifest:
                expected_p1_sha = manifest["outputs/kaggle_build/work/phase1/test.jsonl"]["sha256"]
        except Exception:
            pass

    ref_phase1_candidates = [
        REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase1" / "test.jsonl",
        Path("outputs/kaggle_build/work/phase1/test.jsonl").resolve(),
    ]
    ref_phase1_path = next((p for p in ref_phase1_candidates if p.exists()), ref_phase1_candidates[0])

    if not check_frozen_hash(ref_phase1_path, expected_p1_sha, "phase1/test.jsonl (frozen reference)"):
        passed = False

    # Also confirm hash of execution phase1/test.jsonl if present in run directory
    exec_p1_candidates = [
        REPO_ROOT / "work" / "work_pubmedqa_rag" / "phase1" / "test.jsonl",
        REPO_ROOT / "work" / "phase1" / "test.jsonl",
        Path("/kaggle/working/work/work_pubmedqa_rag/phase1/test.jsonl"),
        Path("/kaggle/working/work/phase1/test.jsonl"),
    ]
    for ep in exec_p1_candidates:
        if ep.exists():
            if not check_frozen_hash(ep, expected_p1_sha, f"phase1/test.jsonl (execution file at {ep})"):
                passed = False
            break

    # 2. Reference PubMedQA Data
    p1_rows = load_jsonl(ref_phase1_path)
    pm_ref = {r["id"]: r for r in p1_rows if r.get("dataset") == "pubmedqa"}
    print(f"\nGold Ground Truth: {len(pm_ref)} PubMedQA test items extracted from reference test.jsonl")

    # 3. Row Counts Verification
    print("\n--- Gate 1: Row Counts ---")
    if n_total == 500 and all(r.get("split") == "test" for r in rows) and all(r.get("dataset") == "pubmedqa" for r in rows):
        print(f"  [PASS] Expected exactly 500 PubMedQA test rows. Got: {n_total}")
    else:
        print(f"  [FAIL] Row count mismatch! Expected 500 test PubMedQA rows, got: {n_total}")
        passed = False

    # 4. Unique IDs Verification
    print("\n--- Gate 2: Unique IDs & Duplicate Detection ---")
    row_ids = [r.get("id") for r in rows]
    id_counts = Counter(row_ids)
    duplicates = [qid for qid, cnt in id_counts.items() if cnt > 1]
    if not duplicates:
        print(f"  [PASS] All {len(row_ids)} IDs are unique. 0 duplicates.")
    else:
        print(f"  [FAIL] Found {len(duplicates)} duplicate IDs: {duplicates[:5]}")
        passed = False

    # 5. Missing / Unexpected IDs Verification
    print("\n--- Gate 3: Missing & Unexpected IDs against Master Benchmark ---")
    output_id_set = set(row_ids)
    gold_id_set = set(pm_ref.keys())
    missing_from_output = gold_id_set - output_id_set
    unexpected_in_output = output_id_set - gold_id_set

    if not missing_from_output and not unexpected_in_output:
        print(f"  [PASS] 0 missing IDs, 0 unexpected IDs. Exact 1-to-1 match with canonical PubMedQA test.")
    else:
        if missing_from_output:
            print(f"  [FAIL] Missing {len(missing_from_output)} IDs from output: {list(missing_from_output)[:5]}")
        if unexpected_in_output:
            print(f"  [FAIL] Unexpected {len(unexpected_in_output)} IDs in output: {list(unexpected_in_output)[:5]}")
        passed = False

    # 6. Gold Alignment Verification
    print("\n--- Gate 4: Gold Mismatches & Metadata Alignment ---")
    gold_mismatches = 0
    for r in rows:
        qid = r.get("id")
        if qid not in pm_ref:
            continue
        g = pm_ref[qid]
        g_ans = g.get("answer")
        r_gold = r.get("gold") if "gold" in r else r.get("answer")
        if g_ans is not None and r_gold is not None and r_gold != g_ans:
            gold_mismatches += 1
            if gold_mismatches <= 3:
                print(f"    Mismatch in gold answer for {qid}: output={r_gold}, gold={g_ans}")

    if gold_mismatches == 0:
        print(f"  [PASS] 0 gold mismatches across all {n_total} rows.")
    else:
        print(f"  [FAIL] Found {gold_mismatches} gold alignment mismatches!")
        passed = False

    # 7. Generation Metrics (REPORT ONLY — NOT GATED)
    print("\n--- REPORTED METRICS (Informational Only — Not Gated) ---")
    trunc_cnt = sum(1 for r in rows if is_truncated(r))
    parsed_cnt = sum(1 for r in rows if is_parsed(r))
    trunc_pct = (trunc_cnt / n_total) * 100 if n_total else 0.0
    parsed_pct = (parsed_cnt / n_total) * 100 if n_total else 0.0

    s_acc = strict_accuracy(rows)
    f_acc = fallback_accuracy(rows)

    cites_cnt = sum(1 for r in rows if r.get("citations") and len(r.get("citations")) > 0)

    print(f"  Parse Rate                      : {parsed_cnt}/{n_total} ({parsed_pct:.1f}%) [Reported]")
    print(f"  Truncation Rate                 : {trunc_cnt}/{n_total} ({trunc_pct:.1f}%) [Reported]")
    print(f"  Strict Accuracy (PRIMARY)       : {s_acc:.4f} ({sum(1 for r in rows if is_strict_correct(r))}/{n_total})")
    print(f"  With-Fallback Accuracy (SECOND) : {f_acc:.4f} ({sum(1 for r in rows if is_fallback_correct(r))}/{n_total})")
    print(f"  Citations Present Rate          : {cites_cnt}/{n_total} ({100.0 * cites_cnt / max(n_total, 1):.1f}%)")

    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    if secs:
        avg_s = sum(secs) / len(secs)
        print(f"\n  Inference Speed  : {avg_s:.2f} s/question (total {sum(secs)/3600.0:.2f} GPU-hours)")

    print("\n" + "=" * 80)
    if passed:
        print("STAGE C RESULT: [PASS] All integrity gates passed. Output verified.")
    else:
        print("STAGE C RESULT: [FAIL] One or more integrity gates failed.")
    print("=" * 80 + "\n")
    return passed


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MedRAG-SLM GPU Output Verifier (Integrity Gate)")
    parser.add_argument(
        "--run",
        type=str,
        default="medpsy_unans_v2_baseline",
        choices=[
            "medpsy_unans_v2_baseline",
            "medpsy_unans_v2_reasoning",
            "qwen3_4b_unans_v2_baseline",
            "qwen3_4b_unans_v2_rag",
            "qwen3_4b_unans_v2",
            "two_stage_push",
            "medpsy_pubmedqa_baseline",
            "medpsy_pubmedqa_rag",
            "all_stages",
        ],
        help="Target run to verify (default: medpsy_unans_v2_baseline)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional explicit path to the JSONL output file.",
    )
    args = parser.parse_args()

    default_paths = {
        "medpsy_unans_v2_baseline": REPO_ROOT / "outputs" / "kaggle_qa" / "unans_v2" / "medpsy-4b" / "baseline_medpsy-4b.jsonl",
        "medpsy_unans_v2_reasoning": REPO_ROOT / "outputs" / "kaggle_qa" / "unans_v2" / "medpsy-4b" / "baseline_reasoning_medpsy-4b.jsonl",
        "qwen3_4b_unans_v2_baseline": REPO_ROOT / "outputs" / "kaggle_qa" / "unans_v2" / "qwen3-4b" / "baseline_qwen3-4b.jsonl",
        "qwen3_4b_unans_v2_rag": REPO_ROOT / "outputs" / "kaggle_qa" / "unans_v2" / "qwen3-4b" / "rag_qwen3-4b.jsonl",
        "medpsy_pubmedqa_baseline": REPO_ROOT / "outputs" / "kaggle_qa" / "medpsy_pubmedqa_baseline" / "medpsy-4b_pubmedqa_baseline.jsonl",
        "medpsy_pubmedqa_rag": REPO_ROOT / "outputs" / "kaggle_qa" / "medpsy_pubmedqa_rag" / "medpsy-4b_pubmedqa_rag.jsonl",
    }

    if args.run == "medpsy_unans_v2_baseline":
        target = Path(args.file) if args.file else default_paths["medpsy_unans_v2_baseline"]
        ok = verify_unans_v2_baseline(target, target_model="medpsy-4b", stage_label="Stage A (MedPsy-4B Standard Baseline)")
        sys.exit(0 if ok else 1)

    elif args.run == "medpsy_unans_v2_reasoning":
        target = Path(args.file) if args.file else default_paths["medpsy_unans_v2_reasoning"]
        ok = verify_unans_v2_baseline(target, target_model="medpsy-4b", stage_label="Stage A2 (MedPsy-4B Reasoning Baseline)")
        sys.exit(0 if ok else 1)

    elif args.run == "qwen3_4b_unans_v2_baseline":
        target = Path(args.file) if args.file else default_paths["qwen3_4b_unans_v2_baseline"]
        ok = verify_unans_v2_baseline(target, target_model="qwen3-4b", stage_label="Stage D (Qwen3-4B Baseline)")
        sys.exit(0 if ok else 1)

    elif args.run == "qwen3_4b_unans_v2_rag":
        target = Path(args.file) if args.file else default_paths["qwen3_4b_unans_v2_rag"]
        ok = verify_unans_v2_baseline(target, target_model="qwen3-4b", stage_label="Stage D (Qwen3-4B Full RAG)")
        sys.exit(0 if ok else 1)

    elif args.run == "qwen3_4b_unans_v2":
        results = {}
        target_b = Path(args.file) if args.file else default_paths["qwen3_4b_unans_v2_baseline"]
        target_r = default_paths["qwen3_4b_unans_v2_rag"]
        results["Stage D (Qwen3-4B Baseline)"] = verify_unans_v2_baseline(target_b, target_model="qwen3-4b", stage_label="Stage D (Qwen3-4B Baseline)")
        results["Stage D (Qwen3-4B Full RAG)"] = verify_unans_v2_baseline(target_r, target_model="qwen3-4b", stage_label="Stage D (Qwen3-4B Full RAG)")

        print("\n" + "=" * 80)
        print("=== QWEN3-4B UNANSWERABLE V2 VERIFICATION SUMMARY ===")
        print("=" * 80)
        all_passed = True
        for stg, res in results.items():
            status_str = "[PASS]" if res else "[FAIL]"
            if not res:
                all_passed = False
            print(f"  {stg:<40}: {status_str}")
        print("=" * 80)
        sys.exit(0 if all_passed else 1)

    elif args.run == "two_stage_push":
        results = {}
        target_a2 = default_paths["medpsy_unans_v2_reasoning"]
        target_db = default_paths["qwen3_4b_unans_v2_baseline"]
        target_dr = default_paths["qwen3_4b_unans_v2_rag"]

        results["Stage A2 (MedPsy-4B Reasoning Baseline)"] = verify_unans_v2_baseline(target_a2, target_model="medpsy-4b", stage_label="Stage A2 (MedPsy-4B Reasoning Baseline)")
        results["Stage D (Qwen3-4B Baseline)"] = verify_unans_v2_baseline(target_db, target_model="qwen3-4b", stage_label="Stage D (Qwen3-4B Baseline)")
        results["Stage D (Qwen3-4B Full RAG)"] = verify_unans_v2_baseline(target_dr, target_model="qwen3-4b", stage_label="Stage D (Qwen3-4B Full RAG)")

        print("\n" + "=" * 80)
        print("=== TWO-STAGE PUSH OVERALL VERIFICATION SUMMARY ===")
        print("=" * 80)
        all_passed = True
        for stg, res in results.items():
            status_str = "[PASS]" if res else "[FAIL]"
            if not res:
                all_passed = False
            print(f"  {stg:<40}: {status_str}")
        print("=" * 80)
        if all_passed:
            print("OVERALL RESULT: [PASS] All outputs passed all integrity gates!\n")
            sys.exit(0)
        else:
            print("OVERALL RESULT: [FAIL] One or more outputs failed integrity gates!\n")
            sys.exit(1)

    elif args.run == "medpsy_pubmedqa_baseline":
        target = Path(args.file) if args.file else default_paths["medpsy_pubmedqa_baseline"]
        ok = verify_pubmedqa_baseline(target, target_model="medpsy-4b")
        sys.exit(0 if ok else 1)

    elif args.run == "medpsy_pubmedqa_rag":
        target = Path(args.file) if args.file else default_paths["medpsy_pubmedqa_rag"]
        ok = verify_pubmedqa_rag(target, target_model="medpsy-4b")
        sys.exit(0 if ok else 1)

    elif args.run == "all_stages":
        results = {}
        ok_a = verify_unans_v2_baseline(default_paths["medpsy_unans_v2_baseline"], target_model="medpsy-4b")
        results["Stage A (Unanswerable v2 Baseline)"] = ok_a

        ok_b = verify_pubmedqa_baseline(default_paths["medpsy_pubmedqa_baseline"], target_model="medpsy-4b")
        results["Stage B (PubMedQA Baseline)"] = ok_b

        ok_c = verify_pubmedqa_rag(default_paths["medpsy_pubmedqa_rag"], target_model="medpsy-4b")
        results["Stage C (PubMedQA Full RAG)"] = ok_c

        print("\n" + "=" * 80)
        print("=== ALL STAGES OVERALL VERIFICATION SUMMARY ===")
        print("=" * 80)
        all_passed = True
        for stg, res in results.items():
            status_str = "[PASS]" if res else "[FAIL]"
            if not res:
                all_passed = False
            print(f"  {stg:<40}: {status_str}")
        print("=" * 80)
        if all_passed:
            print("OVERALL RESULT: [PASS] All 3 stages passed all integrity gates!\n")
            sys.exit(0)
        else:
            print("OVERALL RESULT: [FAIL] One or more stages failed integrity gates!\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
