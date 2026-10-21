"""audit/make_frozen_hashes.py

Generates audit/frozen_hashes.json storing cryptographic SHA-256 checksums,
byte counts, and line counts for all canonical frozen test input and output files.

Usage:
    python audit/make_frozen_hashes.py
    python audit/make_frozen_hashes.py --out audit/frozen_hashes.json
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical list of frozen test input and output files
CANONICAL_TEST_FILES = [
    # Master test and validation split definitions
    "data/unanswerable_v2/val.jsonl",
    "data/unanswerable_v2/test.jsonl",
    "outputs/kaggle_build/work/phase1/test.jsonl",
    # MedPsy-4B frozen SC test outputs
    "outputs/sc_test/medpsy-4b/outputs/sc_test/medpsy-4b/medqa.jsonl",
    "outputs/sc_test_s2_pubmedqa/outputs/sc_test/medpsy-4b/pubmedqa.jsonl",
    "outputs/sc_test_s3_unanswerable/outputs/sc_test/medpsy-4b/unanswerable_v2.jsonl",
    # Final test evaluation metrics
    "outputs/analysis_test/reliability_test_results.json",
    "outputs/analysis_nli/medpsy_test_citation_metrics.json",
    # MedPsy-4B standard evaluation runs
    "outputs/kaggle_qa/medpsy_final/work/phase5/medpsy-4b.jsonl",
    "outputs/kaggle_qa/medpsy_final/work/phase10/medpsy-4b.jsonl",
    "outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl",
    # 5 standard SLMs: Phase 5 (Baseline)
    "outputs/kaggle_qa/full/qwen3-1.7b/work/phase5/qwen3-1.7b.jsonl",
    "outputs/kaggle_qa/full/phi4-mini/work/phase5/phi4-mini.jsonl",
    "outputs/kaggle_qa/full/smollm3-3b/work/phase5/smollm3-3b.jsonl",
    "outputs/kaggle_qa/full/qwen3-4b/work/phase5/qwen3-4b.jsonl",
    "outputs/kaggle_qa/full/gemma3-4b/work/phase5/gemma3-4b.jsonl",
    # 5 standard SLMs: Phase 8 (Full RAG)
    "outputs/kaggle_qa/full/qwen3-1.7b/work/phase8/qwen3-1.7b.jsonl",
    "outputs/kaggle_qa/full/phi4-mini/work/phase8/phi4-mini.jsonl",
    "outputs/kaggle_qa/full/smollm3-3b/work/phase8/smollm3-3b.jsonl",
    "outputs/kaggle_qa/full/qwen3-4b/work/phase8/qwen3-4b.jsonl",
    "outputs/kaggle_qa/full/gemma3-4b_rag/work/phase8/gemma3-4b.jsonl",
    # Phase 10: PubMedQA +Abstract Oracle Context
    "outputs/kaggle_qa/context/work/phase10/qwen3-1.7b.jsonl",
    "outputs/kaggle_qa/context/work/phase10/phi4-mini.jsonl",
    "outputs/kaggle_qa/context/work/phase10/smollm3-3b.jsonl",
    "outputs/kaggle_qa/context/work/phase10/qwen3-4b.jsonl",
    "outputs/kaggle_qa/context/work/phase10/gemma3-4b.jsonl",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    lines = 0
    with open(path, "rb") as fh:
        for _ in fh:
            lines += 1
    return lines


def main():
    parser = argparse.ArgumentParser(description="Create audit/frozen_hashes.json.")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "audit" / "frozen_hashes.json",
        help="Target output JSON path.",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    records = {}
    missing_files = []

    print(f"Computing SHA-256 for {len(CANONICAL_TEST_FILES)} canonical test files...")
    for rel_path in CANONICAL_TEST_FILES:
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            missing_files.append(rel_path)
            print(f"  [MISSING] {rel_path}")
            continue

        file_hash = sha256_file(full_path)
        file_bytes = full_path.stat().st_size
        n_lines = count_lines(full_path)

        records[rel_path] = {
            "sha256": file_hash,
            "bytes": file_bytes,
            "lines": n_lines,
            "last_modified": datetime.fromtimestamp(
                full_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
        print(f"  [OK] {rel_path} -> {file_hash[:12]}... ({file_bytes} bytes, {n_lines} lines)")

    # Cross-reference data/unanswerable_v2/test.sha256 and val.sha256 if present
    ref_sha_path = REPO_ROOT / "data" / "unanswerable_v2" / "test.sha256"
    if ref_sha_path.exists():
        expected_unans_sha = ref_sha_path.read_text().strip().split()[0]
        actual_unans_sha = records.get("data/unanswerable_v2/test.jsonl", {}).get("sha256")
        if actual_unans_sha:
            assert actual_unans_sha == expected_unans_sha, (
                f"FATAL: data/unanswerable_v2/test.jsonl ({actual_unans_sha}) does not match "
                f"test.sha256 ({expected_unans_sha})!"
            )
            print("  [VERIFIED] data/unanswerable_v2/test.sha256 matches test.jsonl hash exactly.")

    ref_val_sha_path = REPO_ROOT / "data" / "unanswerable_v2" / "val.sha256"
    if ref_val_sha_path.exists():
        expected_val_sha = ref_val_sha_path.read_text().strip().split()[0]
        actual_val_sha = records.get("data/unanswerable_v2/val.jsonl", {}).get("sha256")
        if actual_val_sha:
            assert actual_val_sha == expected_val_sha, (
                f"FATAL: data/unanswerable_v2/val.jsonl ({actual_val_sha}) does not match "
                f"val.sha256 ({expected_val_sha})!"
            )
            print("  [VERIFIED] data/unanswerable_v2/val.sha256 matches val.jsonl hash exactly.")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(records),
        "missing_count": len(missing_files),
        "missing_files": missing_files,
        "files": records,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nSaved {len(records)} file hashes to {args.out}")
    if missing_files:
        print(f"WARNING: {len(missing_files)} files were missing and not recorded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
