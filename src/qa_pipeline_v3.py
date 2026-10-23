"""v3 Generic-vs-Medical Matched-Pair QA Pipeline.

Supports conditions C1 through C5:
  C1: Closed-book baseline on MedQA and PubMedQA (no context)
  C2: Full RAG on MedQA and PubMedQA (top-5 retrieved/reranked evidence)
  C3: PubMedQA + Abstract (oracle context)
  C4: unanswerable_v2 baseline (140 test [100 challenge + 40 control] or 260 val)
  C5: unanswerable_v2 RAG (top-5 evidence from medrag-unans-v2-evidence)

Features:
  - One model per push: downloads/loads model once, iterates requested conditions
  - Resumable: skips question IDs already present in the target JSONL file
  - Stratified PILOT mode: N=50 validation questions (never touches test set)
  - Full test guard: requires explicit confirmation before running on test split
  - Outputs self-documenting manifest.json with model ID, chat template, dtype, cap, seed
  - Saves to outputs/v3/<model>/<condition>/<model>.jsonl
  - Compatible with --check-only and --tiny flags
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.common import append_jsonl, banner, get_device, load_config, read_jsonl, set_seed
from src.llm import (
    LLM,
    baseline_prompt,
    context_prompt,
    rag_prompt,
    reasoning_baseline_prompt,
    reasoning_context_prompt,
    reasoning_rag_prompt,
)


def find_evidence_file(work_dir, target_type="general"):
    """Locate pre-computed Phase 7 evidence file across canonical and Kaggle input paths."""
    if target_type == "unans_v2":
        candidates = [
            os.path.join(work_dir, "work_unans_v2", "phase7", "evidence.jsonl"),
            os.path.join(work_dir, "unans_v2", "evidence.jsonl"),
            "outputs/kaggle_build/unans_v2/evidence.jsonl",
            "/kaggle/input/medrag-unans-v2-evidence/evidence.jsonl",
        ]
        candidates.extend(glob.glob("/kaggle/input/**/medrag-unans-v2-evidence/evidence.jsonl", recursive=True))
        candidates.extend([p for p in glob.glob("/kaggle/input/**/evidence.jsonl", recursive=True) if "unans-v2" in p])
    else:
        candidates = [
            os.path.join(work_dir, "phase7", "evidence.jsonl"),
            "outputs/kaggle_build/work/phase7/evidence.jsonl",
            "work/phase7/evidence.jsonl",
        ]
        candidates.extend(glob.glob("/kaggle/input/**/phase7/evidence.jsonl", recursive=True))
        candidates.extend([p for p in glob.glob("/kaggle/input/**/evidence.jsonl", recursive=True) if "unans" not in p])

    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


def find_question_files(work_dir):
    """Locate test and validation master question splits."""
    p1_test = os.path.join(work_dir, "phase1", "test.jsonl")
    p1_val = os.path.join(work_dir, "phase1", "validation.jsonl")
    if not os.path.exists(p1_test) or not os.path.exists(p1_val):
        for candidate in ["outputs/kaggle_build/work/phase1", "work/phase1"]:
            if os.path.exists(os.path.join(candidate, "test.jsonl")):
                p1_test = os.path.join(candidate, "test.jsonl")
                p1_val = os.path.join(candidate, "validation.jsonl")
                break
    if not os.path.exists(p1_test):
        hits = glob.glob("/kaggle/input/**/phase1/test.jsonl", recursive=True)
        if hits:
            p1_test = hits[0]
            p1_val = os.path.join(os.path.dirname(hits[0]), "validation.jsonl")

    # unanswerable_v2 paths
    u2_test = "data/unanswerable_v2/test.jsonl"
    u2_val = "data/unanswerable_v2/val.jsonl"
    if not os.path.exists(u2_test):
        u2_test_hits = glob.glob("**/data/unanswerable_v2/test.jsonl", recursive=True)
        if u2_test_hits:
            u2_test = u2_test_hits[0]
            u2_val = os.path.join(os.path.dirname(u2_test), "val.jsonl")

    return p1_test, p1_val, u2_test, u2_val


def load_evidence_dict(evidence_path):
    """Load JSONL evidence rows into a question_id -> row dictionary."""
    if not evidence_path or not os.path.exists(evidence_path):
        return {}
    ev_dict = {}
    with open(evidence_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                ev_dict[row["id"]] = row
    return ev_dict


def sample_pilot_questions(questions, condition, n=50, seed=42):
    """Stratified sampling of validation questions for PILOT mode (seed 42)."""
    rng = random.Random(seed)
    if condition in ("C1", "C2"):
        medqa = [q for q in questions if q.get("dataset") == "medqa"]
        pubmedqa = [q for q in questions if q.get("dataset") == "pubmedqa"]
        n_each = n // 2
        s_medqa = rng.sample(medqa, min(n_each, len(medqa)))
        s_pubmedqa = rng.sample(pubmedqa, min(n - len(s_medqa), len(pubmedqa)))
        sample = s_medqa + s_pubmedqa
    elif condition == "C3":
        pubmedqa = [q for q in questions if q.get("dataset") == "pubmedqa"]
        sample = rng.sample(pubmedqa, min(n, len(pubmedqa)))
    elif condition in ("C4", "C5"):
        # unanswerable_v2: 200 challenge + 60 controls in val
        challenge = [q for q in questions if q.get("should_abstain")]
        control = [q for q in questions if not q.get("should_abstain")]
        n_ctrl = int(n * 0.3)
        n_chal = n - n_ctrl
        s_chal = rng.sample(challenge, min(n_chal, len(challenge)))
        s_ctrl = rng.sample(control, min(n_ctrl, len(control)))
        sample = s_chal + s_ctrl
    else:
        sample = rng.sample(questions, min(n, len(questions)))
    return sorted(sample, key=lambda x: str(x["id"]))


def assert_appended_row_valid(file_path, expected_id=None):
    """Cheap guard: assert the last appended line in file_path parses as a JSON object with an 'id' field."""
    with open(file_path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        # Read the tail of the file (up to 64KB) to inspect the last line
        fh.seek(max(0, size - 65536))
        tail = fh.read()
    lines = [line.strip() for line in tail.split(b"\n") if line.strip()]
    if not lines:
        raise AssertionError(f"Write verification failed: {file_path} is empty after append")
    last_line = lines[-1].decode("utf-8", errors="replace")
    try:
        obj = json.loads(last_line)
    except Exception as exc:
        raise AssertionError(
            f"Write verification failed on {file_path}: invalid JSON: {exc}. Line snippet: {last_line[:120]!r}"
        ) from exc
    if not isinstance(obj, dict):
        raise AssertionError(
            f"Write verification failed on {file_path}: expected JSON object (dict), "
            f"got {type(obj).__name__}. Line snippet: {last_line[:120]!r}"
        )
    if "id" not in obj or not obj["id"]:
        raise AssertionError(
            f"Write verification failed on {file_path}: missing or empty 'id' field. Line snippet: {last_line[:120]!r}"
        )
    if expected_id is not None and obj["id"] != expected_id:
        raise AssertionError(
            f"Write verification failed on {file_path}: line 'id' ({obj['id']}) does not match expected ({expected_id})"
        )


def run_condition(
    llm,
    condition,
    model_name,
    questions,
    evidence,
    qcfg,
    out_dir,
    batch_size=8,
    check_only=False,
):
    """Run a single experimental condition C1-C5 with resume-skipping."""
    cond_dir = os.path.join(out_dir, condition)
    os.makedirs(cond_dir, exist_ok=True)
    out_file = Path(cond_dir) / f"{model_name}.jsonl"

    # Resume-skipping: load existing IDs from valid JSON object rows
    done_ids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line_str = line.strip()
                if line_str:
                    try:
                        row_obj = json.loads(line_str)
                        if isinstance(row_obj, dict) and "id" in row_obj and row_obj["id"]:
                            done_ids.add(row_obj["id"])
                    except Exception:
                        pass

    todo = [q for q in questions if q["id"] not in done_ids]
    total_q = len(questions)
    print(
        f"\n--- Condition {condition} ({model_name}): {total_q} questions ({len(done_ids)} done, {len(todo)} to do) ---",
        flush=True,
    )

    if not todo:
        print(f"Condition {condition} already complete ({len(done_ids)} rows in {out_file}).", flush=True)
        return

    if check_only:
        todo = todo[:min(5, len(todo))]
        print(f"CHECK_ONLY: Evaluating first {len(todo)} questions only.", flush=True)

    is_reasoning = getattr(llm, "is_reasoning", False)
    max_chars = qcfg.get("passage_max_chars", 1200)

    # Build prompts and options map
    prompts = {}
    options_map = {}
    letters_map = {}

    for q in todo:
        qid = q["id"]
        options_map[qid] = q["options"]
        letters_map[qid] = list(q["options"].keys())

        if condition in ("C1", "C4"):
            # Baseline (no context)
            if is_reasoning:
                prompts[qid] = reasoning_baseline_prompt(q)
            else:
                prompts[qid] = baseline_prompt(q)

        elif condition in ("C2", "C5"):
            # RAG (top-5 passages)
            if qid not in evidence or not evidence[qid].get("passages"):
                raise AssertionError(f"Condition {condition} error: Missing Phase 7 evidence for question {qid}")
            passages = evidence[qid]["passages"]
            if is_reasoning:
                prompts[qid] = reasoning_rag_prompt(q, passages, max_chars)
            else:
                prompts[qid] = rag_prompt(q, passages, max_chars)

        elif condition == "C3":
            # PubMedQA + Abstract (oracle context)
            if is_reasoning:
                prompts[qid] = reasoning_context_prompt(q, max_chars * 3)
            else:
                prompts[qid] = context_prompt(q, max_chars * 3)

    # Process in batches
    t0_all = time.time()
    n_processed = 0

    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        batch_prompts = [prompts[q["id"]] for q in batch]
        batch_letters = [letters_map[q["id"]] for q in batch]

        t0_batch = time.time()
        if is_reasoning:
            results = llm.generate_reasoning_safe(batch_prompts, batch_letters)
        else:
            mode_tag = "rag" if condition in ("C2", "C5") else ("context" if condition == "C3" else "baseline")
            results = llm.generate_safe(batch_prompts, batch_letters, mode_tag)
        sec_batch = time.time() - t0_batch
        sec_per_q = sec_batch / len(batch) if batch else 0.0

        for q, res in zip(batch, results):
            row = dict(q)
            for k, v in res.items():
                row[k] = v
            row["seconds"] = round(sec_per_q, 4)
            row["condition"] = condition
            row["model"] = model_name
            # Evaluate correctness if ground-truth answer present
            if "answer" in row and row.get("pred"):
                row["correct"] = bool(str(row["pred"]).strip().upper() == str(row["answer"]).strip().upper())
            elif "answer_idx" in row and row.get("pred"):
                # PubMedQA sometimes uses answer_idx
                row["correct"] = bool(str(row["pred"]).strip().upper() == str(row["answer_idx"]).strip().upper())

            append_jsonl(out_file, [row])
            assert_appended_row_valid(out_file, expected_id=row.get("id"))
            n_processed += 1

        if (i // batch_size + 1) % 5 == 0 or (i + batch_size >= len(todo)):
            elapsed = time.time() - t0_all
            rate = n_processed / elapsed if elapsed > 0 else 0
            eta_sec = (len(todo) - n_processed) / rate if rate > 0 else 0
            print(
                f"  [{condition}] Processed {n_processed}/{len(todo)} questions "
                f"({rate:.2f} q/s, ETA: {eta_sec/60:.1f}m)",
                flush=True,
            )

    print(f"Condition {condition} completed: {n_processed} new rows appended to {out_file}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="v3 Matched-Pair Generic vs Medical QA Pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base.yaml config")
    parser.add_argument("--work", default="work", help="Working directory containing phase1..7")
    parser.add_argument("--model", required=True, help="Model name from configs/base.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="test", help="Dataset split")
    parser.add_argument("--conditions", default="C1,C2,C3,C4,C5", help="Comma-separated conditions to run")
    parser.add_argument("--out", default=None, help="Output directory (defaults to outputs/v3/<model>)")
    parser.add_argument("--pilot", action="store_true", help="Run stratified N=50 validation pilot")
    parser.add_argument("--pilot-n", type=int, default=50, help="Pilot question count (default: 50)")
    parser.add_argument("--check-only", action="store_true", help="Format and parse check only")
    parser.add_argument("--tiny", action="store_true", help="Run tiny CPU check with tiny_model")
    parser.add_argument("--device", default=None, help="Torch device override (auto, cuda, cpu)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    qcfg = cfg.get("qa", {})
    v3_cfg = cfg.get("v3", {})
    set_seed(v3_cfg.get("seed", cfg.get("seed", 42)))

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    valid_conditions = {"C1", "C2", "C3", "C4", "C5"}
    bad_conds = [c for c in conditions if c not in valid_conditions]
    if bad_conds:
        sys.exit(f"ERROR: Unknown conditions: {bad_conds}. Allowed: {sorted(valid_conditions)}")

    # Resolve model config
    if args.tiny:
        mcfg = dict(cfg.get("tiny_model", {}))
        mcfg["name"] = args.model
        print(f"TINY MODE: using tiny model {mcfg.get('id')} for {args.model}")
    else:
        model_map = {m["name"]: m for m in cfg.get("models", [])}
        if args.model not in model_map:
            sys.exit(f"ERROR: Model '{args.model}' not found in {args.config}. Available: {list(model_map.keys())}")
        mcfg = model_map[args.model]

    # Resolve output directory
    if args.out:
        out_base = args.out
    else:
        out_base = f"/kaggle/working/outputs/v3/{args.model}" if os.path.exists("/kaggle/working") else f"outputs/v3/{args.model}"
    os.makedirs(out_base, exist_ok=True)

    # Locate question files
    p1_test, p1_val, u2_test, u2_val = find_question_files(args.work)
    print(f"Data sources:")
    print(f"  Phase 1 test       : {p1_test} (exists: {os.path.exists(p1_test)})")
    print(f"  Phase 1 val        : {p1_val} (exists: {os.path.exists(p1_val)})")
    print(f"  unanswerable_v2 test: {u2_test} (exists: {os.path.exists(u2_test)})")
    print(f"  unanswerable_v2 val : {u2_val} (exists: {os.path.exists(u2_val)})")

    # Locate evidence files
    ev_gen_path = find_evidence_file(args.work, target_type="general")
    ev_u2_path = find_evidence_file(args.work, target_type="unans_v2")
    print(f"Evidence sources:")
    print(f"  General RAG evidence (C2)     : {ev_gen_path}")
    print(f"  Unanswerable v2 evidence (C5) : {ev_u2_path}")

    # Build manifest
    manifest = {
        "v3_model": args.model,
        "hf_repo_id": mcfg.get("id"),
        "model_type": mcfg.get("loader", "causal_lm"),
        "dtype": mcfg.get("dtype"),
        "chat_template": mcfg.get("chat_template", "tokenizer_default"),
        "reasoning": bool(mcfg.get("reasoning", False)),
        "thinking_cap": mcfg.get("max_new_tokens_reasoning"),
        "seed": v3_cfg.get("seed", cfg.get("seed", 42)),
        "split": "validation" if args.pilot else args.split,
        "pilot": args.pilot,
        "pilot_n": args.pilot_n if args.pilot else None,
        "conditions": conditions,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = os.path.join(out_base, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest written to {manifest_path}", flush=True)

    # Load questions per condition
    split_name = "validation" if args.pilot else args.split

    # Load master Phase 1 questions
    if split_name == "validation":
        if not os.path.exists(p1_val):
            sys.exit(f"ERROR: Validation questions not found at {p1_val}")
        p1_questions = read_jsonl(p1_val)
        if not os.path.exists(u2_val):
            sys.exit(f"ERROR: unanswerable_v2 validation not found at {u2_val}")
        u2_questions = read_jsonl(u2_val)
    else:
        if not os.path.exists(p1_test):
            sys.exit(f"ERROR: Test questions not found at {p1_test}")
        p1_questions = read_jsonl(p1_test)
        if not os.path.exists(u2_test):
            sys.exit(f"ERROR: unanswerable_v2 test not found at {u2_test}")
        u2_questions = read_jsonl(u2_test)

    # Filter by dataset
    medqa_all = [q for q in p1_questions if q.get("dataset") == "medqa"]
    pubmedqa_all = [q for q in p1_questions if q.get("dataset") == "pubmedqa"]

    # Pre-load evidence dictionaries if C2 or C5 requested
    evidence_c2 = {}
    if "C2" in conditions:
        if not ev_gen_path and not args.check_only and not args.tiny:
            raise FileNotFoundError("ERROR: C2 requires general Phase 7 evidence file.")
        evidence_c2 = load_evidence_dict(ev_gen_path)

    evidence_c5 = {}
    if "C5" in conditions:
        if not ev_u2_path and not args.check_only and not args.tiny:
            raise FileNotFoundError("ERROR: C5 requires unanswerable_v2 evidence file.")
        evidence_c5 = load_evidence_dict(ev_u2_path)

    # Device resolution and LLM loading
    device = args.device or get_device()
    if args.tiny:
        device = "cpu"

    banner(f"LOADING MODEL: {args.model} ({mcfg.get('id')}) on {device}")
    t0_load = time.time()
    llm = LLM(mcfg, device=device)
    print(f"Model loaded in {time.time() - t0_load:.1f}s (load_source={getattr(llm, 'load_source', 'default')})", flush=True)

    # Determine batch size
    batch_size = mcfg.get("batch_size", qcfg.get("batch_size", 8))
    batch_size_rag = mcfg.get("batch_size_rag", batch_size)

    # Execute conditions
    for cond in conditions:
        if cond in ("C1", "C2"):
            q_list = medqa_all + pubmedqa_all
            ev_list = evidence_c2 if cond == "C2" else {}
            bs = batch_size_rag if cond == "C2" else batch_size
        elif cond == "C3":
            q_list = pubmedqa_all
            ev_list = {}
            bs = batch_size_rag
        elif cond in ("C4", "C5"):
            q_list = u2_questions
            ev_list = evidence_c5 if cond == "C5" else {}
            bs = batch_size_rag if cond == "C5" else batch_size

        if args.pilot:
            q_list = sample_pilot_questions(q_list, cond, n=args.pilot_n, seed=v3_cfg.get("seed", 42))

        run_condition(
            llm=llm,
            condition=cond,
            model_name=args.model,
            questions=q_list,
            evidence=ev_list,
            qcfg=qcfg,
            out_dir=out_base,
            batch_size=bs,
            check_only=args.check_only,
        )

    print(f"\n=== QA PIPELINE V3 COMPLETE for {args.model} (conditions: {conditions}) ===\n", flush=True)


if __name__ == "__main__":
    main()
