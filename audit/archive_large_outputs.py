#!/usr/bin/env python3
"""audit/archive_large_outputs.py

Archives remaining large raw JSONL files that are gitignored:
  - outputs/kaggle_qa/** (*.jsonl)
  - outputs/kaggle_build/work/phase2/corpus.jsonl
  - outputs/kaggle_build/work/phase7/evidence.jsonl
  - any other raw output JSONLs in kaggle/local/work/tiny

Steps performed:
1. Lists all target large raw files with sizes and calculates SHA-256 hashes.
2. Writes metadata and checksums to audit/archive_checksums.json.
3. Packages files into a compressed tarball at ~/Documents/MTECH/medrag_archive/medrag_raw_outputs_archive.tar.gz.
4. Computes and prints archive SHA-256 checksum and size.
5. Prints exact Kaggle CLI commands to upload as a private Kaggle Dataset.

Usage:
    python audit/archive_large_outputs.py
    python audit/archive_large_outputs.py --dry-run
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE_DIR = Path.home() / "Documents" / "MTECH" / "medrag_archive"
ARCHIVE_NAME = "medrag_raw_outputs_archive.tar.gz"
CHECKSUM_FILE = REPO_ROOT / "audit" / "archive_checksums.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def find_target_files() -> List[Path]:
    """Find all large raw JSONLs to archive."""
    targets = []

    # 1. corpus.jsonl and evidence.jsonl
    corpus = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase2" / "corpus.jsonl"
    if corpus.exists():
        targets.append(corpus)
    evidence = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase7" / "evidence.jsonl"
    if evidence.exists():
        targets.append(evidence)

    # 2. outputs/kaggle_qa/**/*.jsonl
    kq_dir = REPO_ROOT / "outputs" / "kaggle_qa"
    if kq_dir.exists():
        for p in sorted(kq_dir.glob("**/*.jsonl")):
            if p.is_file():
                targets.append(p)

    # 3. Any additional large jsonl files in outputs/ not covered by whitelist
    for extra_sub in ("kaggle", "local", "tiny", "work"):
        sub_dir = REPO_ROOT / "outputs" / extra_sub
        if sub_dir.exists():
            for p in sorted(sub_dir.glob("**/*.jsonl")):
                if p.is_file() and p not in targets:
                    targets.append(p)

    # Remove duplicates preserving order
    seen = set()
    deduped = []
    for p in targets:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Archive large raw JSONL files.")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Destination directory for tar.gz archive (default: ~/Documents/MTECH/medrag_archive/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and compute checksums without creating the tar.gz archive.",
    )
    args = parser.parse_args()

    dest_dir = args.dest_dir.expanduser().resolve()
    archive_path = dest_dir / ARCHIVE_NAME

    print("=" * 80)
    print("MedRAG-SLM Large Outputs Archival Utility")
    print("=" * 80)

    target_files = find_target_files()
    if not target_files:
        print("No matching large raw JSONL files found in outputs/.")
        sys.exit(0)

    total_bytes = sum(p.stat().st_size for p in target_files)
    print(f"Found {len(target_files)} target file(s) ({total_bytes / (1024 * 1024):.2f} MB total):\n")

    file_records = []
    for p in target_files:
        rel = p.relative_to(REPO_ROOT).as_posix()
        size = p.stat().st_size
        print(f"  Hashing: {rel} ({size / (1024 * 1024):.2f} MB)...", end="", flush=True)
        sha = sha256_file(p)
        print(" done")
        file_records.append({
            "rel_path": rel,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "sha256": sha,
        })

    # Write audit/archive_checksums.json
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(file_records),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "files": file_records,
    }

    CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKSUM_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[OK] Wrote file checksums to: {CHECKSUM_FILE.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print("\n[DRY-RUN] Skipping tar.gz packaging.")
        return

    # Build tar.gz
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPackaging {len(target_files)} files into {archive_path}...")
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in target_files:
            rel = p.relative_to(REPO_ROOT).as_posix()
            tar.add(p, arcname=rel)

    archive_size = archive_path.stat().st_size
    archive_sha = sha256_file(archive_path)

    print("\n" + "=" * 80)
    print("ARCHIVE BUILT SUCCESSFULLY")
    print("=" * 80)
    print(f"Location    : {archive_path}")
    print(f"Size        : {archive_size:,} bytes ({archive_size / (1024 * 1024):.2f} MB)")
    print(f"SHA-256     : {archive_sha}")
    print("=" * 80)

    # Kaggle Dataset Metadata & Upload Instructions
    meta_path = dest_dir / "dataset-metadata.json"
    dataset_slug = "medrag-slm-raw-outputs-archive"
    metadata = {
        "title": "MedRAG-SLM Raw Outputs Archive",
        "id": f"satyasainathteeparthi/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\nTo upload this archive to Kaggle as a private dataset, run:")
    print("    # 1. Verify Kaggle credentials in ~/.kaggle/kaggle.json")
    print(f"    # 2. Check metadata at: {meta_path} (isPrivate: true)")
    print("    # 3. Create private dataset:")
    print(f"    kaggle datasets create -p {dest_dir} -r tar\n")
    print("    # 4. (Optional) Check status or download to another machine:")
    print(f"    kaggle datasets status satyasainathteeparthi/{dataset_slug}")
    print("=" * 80)


if __name__ == "__main__":
    main()
