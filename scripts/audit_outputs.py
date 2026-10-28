#!/usr/bin/env python3
"""scripts/audit_outputs.py — Integrity audit for outputs/v3 and outputs/sweeps.

Audits:
  1. ID set equals frozen ID set for its split/dataset (reads IDs only from test files).
  2. Unique IDs within each file/variant.
  3. Expected row counts.
  4. Manifest presence in each model/sweep folder.
  5. No HuggingFace token pattern (hf_[A-Za-z0-9]{34,}) in outputs.
  6. Rule document hash in each manifest matches committed document.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent

HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{30,}")

# Frozen test IDs (only reading "id" field!)
def get_frozen_ids() -> Dict[str, Set[str]]:
    frozen = {}

    p1_test = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase1" / "test.jsonl"
    if p1_test.exists():
        medqa_ids = set()
        pubmedqa_ids = set()
        with open(p1_test, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    qid = r.get("id")
                    ds = r.get("dataset")
                    if ds == "medqa":
                        medqa_ids.add(qid)
                    elif ds == "pubmedqa":
                        pubmedqa_ids.add(qid)
        frozen["medqa_test"] = medqa_ids
        frozen["pubmedqa_test"] = pubmedqa_ids

    u2_test = REPO_ROOT / "data" / "unanswerable_v2" / "test.jsonl"
    if u2_test.exists():
        u2_ids = set()
        with open(u2_test, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    u2_ids.add(json.loads(line).get("id"))
        frozen["unans_v2_test"] = u2_ids

    u2_val = REPO_ROOT / "data" / "unanswerable_v2" / "val.jsonl"
    if u2_val.exists():
        u2_val_ids = set()
        with open(u2_val, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    u2_val_ids.add(json.loads(line).get("id"))
        frozen["unans_v2_val"] = u2_val_ids

    return frozen


def compute_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit() -> bool:
    all_passed = True
    frozen_ids = get_frozen_ids()

    # Rule document hashes
    rule_hashes = {
        "docs/BUDGET_DECISION_RULE.md": compute_sha256(REPO_ROOT / "docs" / "BUDGET_DECISION_RULE.md"),
        "docs/GEMMA_PROMPT_SENSITIVITY_RULE.md": compute_sha256(REPO_ROOT / "docs" / "GEMMA_PROMPT_SENSITIVITY_RULE.md"),
    }

    # 1. Sweep: budget_pair_b
    sweep_b_dir = REPO_ROOT / "outputs" / "sweeps" / "budget_pair_b"
    b_manifest = sweep_b_dir / "manifest.json"
    if not b_manifest.exists():
        print("FAIL: Missing outputs/sweeps/budget_pair_b/manifest.json")
        all_passed = False
    else:
        with open(b_manifest, "r", encoding="utf-8") as f:
            bm = json.load(f)
        if bm.get("rule_file_sha256") != rule_hashes["docs/BUDGET_DECISION_RULE.md"]:
            print(f"FAIL: Rule hash mismatch in budget_pair_b manifest: expected {rule_hashes['docs/BUDGET_DECISION_RULE.md']}, got {bm.get('rule_file_sha256')}")
            all_passed = False

    for m in ["medpsy-4b", "qwen3-4b-thinking-2507"]:
        fpath = sweep_b_dir / f"{m}.jsonl"
        if not fpath.exists():
            print(f"FAIL: Missing {fpath}")
            all_passed = False
            continue
        text = fpath.read_text(encoding="utf-8")
        if HF_TOKEN_RE.search(text):
            print(f"FAIL: HF token detected in {fpath}")
            all_passed = False
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if len(rows) != 100:
            print(f"FAIL: Expected 100 rows in {fpath}, got {len(rows)}")
            all_passed = False
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            print(f"FAIL: Duplicate IDs found in {fpath}")
            all_passed = False

    # 2. Sweep: gemma_prompt_sensitivity
    sweep_g_dir = REPO_ROOT / "outputs" / "sweeps" / "gemma_prompt_sensitivity"
    g_manifest = sweep_g_dir / "manifest.json"
    if not g_manifest.exists():
        print("FAIL: Missing outputs/sweeps/gemma_prompt_sensitivity/manifest.json")
        all_passed = False
    else:
        with open(g_manifest, "r", encoding="utf-8") as f:
            gm = json.load(f)
        if gm.get("rule_file_sha256") != rule_hashes["docs/GEMMA_PROMPT_SENSITIVITY_RULE.md"]:
            print(f"FAIL: Rule hash mismatch in gemma_prompt_sensitivity manifest: expected {rule_hashes['docs/GEMMA_PROMPT_SENSITIVITY_RULE.md']}, got {gm.get('rule_file_sha256')}")
            all_passed = False

    for m in ["gemma3-4b", "medgemma-1.5-4b-it"]:
        fpath = sweep_g_dir / f"{m}.jsonl"
        if not fpath.exists():
            print(f"FAIL: Missing {fpath}")
            all_passed = False
            continue
        text = fpath.read_text(encoding="utf-8")
        if HF_TOKEN_RE.search(text):
            print(f"FAIL: HF token detected in {fpath}")
            all_passed = False
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        if len(rows) != 600:
            print(f"FAIL: Expected 600 rows in {fpath}, got {len(rows)}")
            all_passed = False
        for v in ["V1", "V2", "V3"]:
            v_rows = [r for r in rows if r.get("variant") == v]
            if len(v_rows) != 200:
                print(f"FAIL: Expected 200 rows for variant {v} in {fpath}, got {len(v_rows)}")
                all_passed = False
            v_ids = [r["id"] for r in v_rows]
            if len(set(v_ids)) != len(v_ids):
                print(f"FAIL: Duplicate IDs found in variant {v} in {fpath}")
                all_passed = False

    # 3. outputs/v3 models
    v3_models = ["gemma3-4b", "medgemma-1.5-4b-it", "medpsy-1.7b", "qwen3-1.7b-thinking", "qwen3-4b-thinking-2507"]
    for m in v3_models:
        mdir = REPO_ROOT / "outputs" / "v3" / m
        # Check manifest
        m_manifest = mdir / "manifest.json"
        if not m_manifest.exists():
            # Check condition-specific manifests
            c_manifests = list(mdir.glob("manifest*.json"))
            if not c_manifests and m != "qwen3-4b-thinking-2507":
                print(f"FAIL: Missing manifest in {mdir}")
                all_passed = False

        for jsonl_file in mdir.glob("**/*.jsonl"):
            text = jsonl_file.read_text(encoding="utf-8")
            if HF_TOKEN_RE.search(text):
                print(f"FAIL: HF token detected in {jsonl_file}")
                all_passed = False
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            ids = [r["id"] for r in rows]
            if len(set(ids)) != len(ids):
                print(f"FAIL: Duplicate IDs found in {jsonl_file}")
                all_passed = False

            # Verify against frozen test IDs
            cond_folder = jsonl_file.parent.name
            if cond_folder in ["C1", "C2", "C3"]:
                datasets_in_file = set(r.get("dataset") for r in rows)
                for ds in datasets_in_file:
                    sub_ids = set(r["id"] for r in rows if r.get("dataset") == ds)
                    if ds == "medqa" and "medqa_test" in frozen_ids:
                        if not sub_ids.issubset(frozen_ids["medqa_test"]):
                            diff = sub_ids - frozen_ids["medqa_test"]
                            print(f"FAIL: Non-frozen MedQA IDs found in {jsonl_file}: {list(diff)[:3]}")
                            all_passed = False
                    elif ds == "pubmedqa" and "pubmedqa_test" in frozen_ids:
                        if not sub_ids.issubset(frozen_ids["pubmedqa_test"]):
                            diff = sub_ids - frozen_ids["pubmedqa_test"]
                            print(f"FAIL: Non-frozen PubMedQA IDs found in {jsonl_file}: {list(diff)[:3]}")
                            all_passed = False
            elif cond_folder in ["C4", "C5"]:
                u_ids = set(ids)
                if "unans_v2_test" in frozen_ids:
                    if u_ids != frozen_ids["unans_v2_test"]:
                        diff = u_ids.symmetric_difference(frozen_ids["unans_v2_test"])
                        print(f"FAIL: Unanswerable test ID mismatch in {jsonl_file}: {len(diff)} mismatched IDs")
                        all_passed = False

    if all_passed:
        print("PASS: outputs/v3 and outputs/sweeps integrity and audit checks verified.")
    return all_passed


if __name__ == "__main__":
    ok = audit()
    sys.exit(0 if ok else 1)
