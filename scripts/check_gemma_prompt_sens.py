#!/usr/bin/env python3
"""scripts/check_gemma_prompt_sens.py — Verification tool for Gemma validation prompt-sensitivity sweep.

Performs automated PASS/FAIL checks:
  1. Manifest present with completed=true.
  2. Rule-file SHA-256 in manifest matches current docs/GEMMA_PROMPT_SENSITIVITY_RULE.md.
  3. Prompt-template hashes in manifest match prompt_template_hashes (V1, V2, V3).
  4. Output files exist for both models (gemma3-4b, medgemma-1.5-4b-it).
  5. Total row count equals N x 3 per model.
  6. Exactly N rows per variant (V1, V2, V3) for each model.
  7. All (id, variant) pairs are unique within each model.
  8. Identical question ID sets across both models for each variant and across variants.
  9. Split is 'validation' on every row (hard rule).
  10. Required schema keys present on every row (id, pred, parsed, correct, truncated, variant).
  11. No Hugging Face token pattern (hf_...) in outputs.

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
from src.qa_pipeline_v3 import PROMPT_TEMPLATE_HASHES

MODELS = ["gemma3-4b", "medgemma-1.5-4b-it"]
VARIANTS = ["V1", "V2", "V3"]
RULE_FILE_PATH = os.path.join(REPO_ROOT, "docs", "GEMMA_PROMPT_SENSITIVITY_RULE.md")


def compute_sha256(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_prompt_sens(sweep_dir: str, expected_n: int = 200) -> bool:
    print("\n" + "=" * 70)
    print(f"=== GEMMA PROMPT-SENSITIVITY AUDIT: {sweep_dir} ===")
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

    # 1. Manifest present and valid JSON
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

    # 3. Rule-file hash matches committed docs/GEMMA_PROMPT_SENSITIVITY_RULE.md
    expected_rule_hash = compute_sha256(RULE_FILE_PATH)
    manifest_rule_hash = manifest.get("rule_file_sha256", "")
    rule_hash_match = (expected_rule_hash != "") and (expected_rule_hash == manifest_rule_hash)
    report(
        "Rule-file SHA-256 matches docs/GEMMA_PROMPT_SENSITIVITY_RULE.md",
        rule_hash_match,
        f"manifest={manifest_rule_hash[:12]}... local={expected_rule_hash[:12]}...",
    )

    # 4. Prompt template hashes match
    manifest_template_hashes = manifest.get("prompt_template_hashes", {})
    hashes_match = (
        isinstance(manifest_template_hashes, dict)
        and all(manifest_template_hashes.get(v) == PROMPT_TEMPLATE_HASHES.get(v) for v in VARIANTS)
    )
    report(
        "Prompt-template hashes match manifest",
        hashes_match,
        f"verified variants: {VARIANTS}",
    )

    # Expected N resolution: if manifest records N, use that; otherwise expected_n
    if "N" in manifest and manifest["N"] is not None:
        expected_n = int(manifest["N"])
    total_expected_rows = expected_n * len(VARIANTS)

    # 5. Model files exist and load rows
    model_rows = {}
    for m in MODELS:
        mfile = os.path.join(sweep_dir, f"{m}.jsonl")
        exists = os.path.exists(mfile)
        report(f"Output file exists for {m}", exists, mfile)
        if not exists:
            return False
        try:
            rows = read_jsonl(mfile)
            model_rows[m] = rows
            report(f"Output file valid JSONL for {m}", True, f"loaded {len(rows)} rows")
        except Exception as exc:
            report(f"Output file valid JSONL for {m}", False, str(exc))
            return False

    # 6. Row counts equal N x 3 per model
    for m in MODELS:
        cnt = len(model_rows[m])
        report(
            f"Row count equals N x {len(VARIANTS)} ({total_expected_rows}) for {m}",
            cnt == total_expected_rows,
            f"count={cnt}",
        )

    # 7. Unique (id, variant) pairs within each model
    for m in MODELS:
        pairs = [(r.get("id"), r.get("variant")) for r in model_rows[m]]
        unique_pairs = set(pairs)
        none_pairs = [p for p in pairs if not p[0] or not p[1]]
        report(
            f"Unique (id, variant) pairs for {m}",
            len(pairs) == len(unique_pairs) and len(none_pairs) == 0,
            f"total={len(pairs)}, unique={len(unique_pairs)}, invalid={len(none_pairs)}",
        )

    # 8. Row count per variant equals N
    for m in MODELS:
        for v in VARIANTS:
            v_rows = [r for r in model_rows[m] if r.get("variant") == v]
            report(
                f"Variant {v} row count equals N ({expected_n}) for {m}",
                len(v_rows) == expected_n,
                f"count={len(v_rows)}",
            )

    # 9. Identical question ID sets across both models per variant and across variants
    base_id_set = None
    for v in VARIANTS:
        ids_gemma = {r.get("id") for r in model_rows[MODELS[0]] if r.get("variant") == v}
        ids_medgemma = {r.get("id") for r in model_rows[MODELS[1]] if r.get("variant") == v}
        identical_pair = (ids_gemma == ids_medgemma) and (len(ids_gemma) == expected_n)
        report(
            f"Identical ID sets across models for variant {v}",
            identical_pair,
            f"count={len(ids_gemma)}",
        )
        if base_id_set is None:
            base_id_set = ids_gemma
        else:
            consistent_across_v = (ids_gemma == base_id_set)
            report(
                f"Variant {v} IDs consistent with variant {VARIANTS[0]}",
                consistent_across_v,
                f"match={consistent_across_v}",
            )

    # 10. Split is 'validation' on every row (hard rule)
    for m in MODELS:
        non_val = [r.get("id") for r in model_rows[m] if r.get("split") != "validation"]
        report(
            f"Hard validation split assertion for {m}",
            len(non_val) == 0,
            f"non-validation rows={len(non_val)}",
        )

    # 11. Required schema keys present on every row
    required_keys = ["id", "pred", "parsed", "correct", "truncated", "variant"]
    for m in MODELS:
        missing_keys_count = 0
        for r in model_rows[m]:
            if any(k not in r for k in required_keys):
                missing_keys_count += 1
        report(
            f"Required evaluation schema keys present for {m}",
            missing_keys_count == 0,
            f"missing keys rows={missing_keys_count}",
        )

    # 12. Truncation consistency check: truncated == (not parsed) per Amendment 1
    for m in MODELS:
        inconsistent_trunc = [
            r.get("id") for r in model_rows[m]
            if r.get("truncated") != (not r.get("parsed"))
        ]
        report(
            f"Truncation consistency (truncated == not parsed) for {m}",
            len(inconsistent_trunc) == 0,
            f"inconsistent rows={len(inconsistent_trunc)}",
        )

    # 13. No HF token leaks in any output files
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
    parser = argparse.ArgumentParser(description="Audit Gemma Prompt-Sensitivity Outputs")
    parser.add_argument(
        "--sweep-dir",
        default="outputs/sweeps/gemma_prompt_sensitivity",
        help="Directory containing prompt-sensitivity JSONL and manifest.json (default: outputs/sweeps/gemma_prompt_sensitivity)",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=200,
        help="Expected question count per variant (default: 200; overridden by manifest N if present)",
    )
    args = parser.parse_args()

    success = check_prompt_sens(args.sweep_dir, expected_n=args.expected_n)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
