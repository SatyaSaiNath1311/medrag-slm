#!/usr/bin/env python3
"""scripts/check_budget_sweep.py — Verification tool for Pair B validation token-budget sweep.

Performs automated PASS/FAIL checks:
  1. Manifest present with completed=true.
  2. Rule-file SHA-256 in manifest matches current docs/BUDGET_DECISION_RULE.md.
  3. Output files exist for both models (qwen3-4b-thinking-2507, medpsy-4b).
  4. Row count equals N per model.
  5. All row IDs are unique within each model.
  6. Identical ID sets across both models.
  7. Split is 'validation' on every row (hard rule).
  8. No missing or invalid n_generated_tokens.
  9. No Hugging Face token pattern (hf_...) in outputs.

Exit codes:
  0: All checks passed.
  1: One or more checks failed.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common import read_jsonl

MODELS = ["qwen3-4b-thinking-2507", "medpsy-4b"]
RULE_FILE_PATH = os.path.join(REPO_ROOT, "docs", "BUDGET_DECISION_RULE.md")


def compute_sha256(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_sweep(sweep_dir: str, expected_n: int = 100) -> bool:
    print("\n" + "=" * 70)
    print(f"=== PAIR B BUDGET SWEEP AUDIT: {sweep_dir} ===")
    print("=" * 70 + "\n")

    all_passed = True

    def report(name: str, passed: bool, detail: str = ""):
        nonlocal all_passed
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        msg = f"  {status} {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    # 1. Manifest present
    manifest_path = os.path.join(sweep_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        report("Manifest file exists", False, f"Not found at {manifest_path}")
        return False
    report("Manifest file exists", True, manifest_path)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        report("Manifest is valid JSON", False, str(exc))
        return False
    report("Manifest is valid JSON", True)

    # 2. Manifest completed == True
    is_completed = manifest.get("completed") is True
    report("Manifest completed == True", is_completed, f"completed={manifest.get('completed')}")

    # 3. Rule-file hash matches committed docs/BUDGET_DECISION_RULE.md
    expected_rule_hash = compute_sha256(RULE_FILE_PATH)
    manifest_rule_hash = manifest.get("rule_file_sha256", "")
    rule_hash_match = (expected_rule_hash != "") and (expected_rule_hash == manifest_rule_hash)
    report(
        "Rule-file SHA-256 matches docs/BUDGET_DECISION_RULE.md",
        rule_hash_match,
        f"manifest={manifest_rule_hash[:12]}... local={expected_rule_hash[:12]}...",
    )

    # Expected N: if manifest records N, use that; otherwise expected_n
    if "N" in manifest and manifest["N"] is not None:
        expected_n = manifest["N"]

    # 4. Check model files exist
    model_rows = {}
    for m in MODELS:
        mpath = os.path.join(sweep_dir, f"{m}.jsonl")
        exists = os.path.exists(mpath)
        report(f"Model output exists: {m}.jsonl", exists, mpath)
        if exists:
            try:
                rows = read_jsonl(mpath)
                model_rows[m] = rows
            except Exception as e:
                report(f"Parse JSONL for {m}", False, str(e))
                model_rows[m] = []
        else:
            model_rows[m] = []

    if any(len(model_rows[m]) == 0 for m in MODELS):
        print("\nAudit aborted: Missing or empty model output files.")
        return False

    # 5. Row count equals N per model
    for m in MODELS:
        cnt = len(model_rows[m])
        report(f"Row count equals N ({expected_n}) for {m}", cnt == expected_n, f"count={cnt}")

    # 6. All IDs unique within each model
    for m in MODELS:
        ids = [r.get("id") for r in model_rows[m]]
        unique_ids = set(ids)
        report(f"Unique row IDs in {m}", len(ids) == len(unique_ids), f"total={len(ids)}, unique={len(unique_ids)}")

    # 7. Identical ID sets across models
    id_sets = {m: set(r.get("id") for r in model_rows[m]) for m in MODELS}
    ids_identical = (id_sets[MODELS[0]] == id_sets[MODELS[1]])
    diff_count = len(id_sets[MODELS[0]] ^ id_sets[MODELS[1]])
    report("Identical ID sets across models", ids_identical, f"symmetric difference count={diff_count}")

    # 8. Split is validation only (HARD RULE)
    for m in MODELS:
        non_val = [r.get("id") for r in model_rows[m] if r.get("split") != "validation"]
        report(
            f"Split is 'validation' on all rows for {m}",
            len(non_val) == 0,
            f"non-validation rows={len(non_val)}",
        )

    # 9. No missing n_generated_tokens
    for m in MODELS:
        missing_tokens = [
            r.get("id") for r in model_rows[m]
            if r.get("n_generated_tokens") is None or not isinstance(r.get("n_generated_tokens"), int) or r.get("n_generated_tokens") < 0
        ]
        report(
            f"Valid n_generated_tokens for {m}",
            len(missing_tokens) == 0,
            f"invalid or missing count={len(missing_tokens)}",
        )

    # 10. No hf_ token pattern in outputs
    hf_token_pattern = re.compile(r"hf_[A-Za-z0-9]{20,}")
    for m in MODELS:
        leaks = [
            r.get("id") for r in model_rows[m]
            if hf_token_pattern.search(str(r.get("raw_output", "")))
        ]
        report(
            f"No HF token leaks in {m}",
            len(leaks) == 0,
            f"detected count={len(leaks)}",
        )

    print("\n" + "-" * 70)
    if all_passed:
        print("RESULT: ALL AUDIT CHECKS PASSED [OK]\n")
        return True
    else:
        print("RESULT: AUDIT FAILED [ERROR]\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check Pair B Token-Budget Sweep Outputs")
    parser.add_argument(
        "--sweep-dir",
        default="outputs/sweeps/budget_pair_b",
        help="Directory containing sweep JSONL and manifest.json (default: outputs/sweeps/budget_pair_b)",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=100,
        help="Expected row count per model (default: 100; overridden by manifest N if present)",
    )
    args = parser.parse_args()

    success = check_sweep(args.sweep_dir, expected_n=args.expected_n)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
