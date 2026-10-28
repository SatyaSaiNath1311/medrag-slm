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
import hashlib
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path
from typing import Any

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
    extract_final_answer_from_reasoning,
    parse_answer,
    parse_citations,
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
        if medqa and pubmedqa:
            n_each = n // 2
            s_medqa = rng.sample(medqa, min(n_each, len(medqa)))
            s_pubmedqa = rng.sample(pubmedqa, min(n - len(s_medqa), len(pubmedqa)))
            sample = s_medqa + s_pubmedqa
        elif medqa:
            sample = rng.sample(medqa, min(n, len(medqa)))
        elif pubmedqa:
            sample = rng.sample(pubmedqa, min(n, len(pubmedqa)))
        else:
            sample = []
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


def parse_and_validate_datasets(datasets_arg: str, conditions: list) -> list:
    """Parse and validate --datasets argument against requested conditions."""
    if not datasets_arg or not datasets_arg.strip():
        raise ValueError("ERROR: --datasets cannot be empty.")
    datasets = [d.strip().lower() for d in datasets_arg.split(",") if d.strip()]
    valid_datasets = {"medqa", "pubmedqa"}
    bad_datasets = [d for d in datasets if d not in valid_datasets]
    if bad_datasets:
        raise ValueError(f"ERROR: Unknown datasets: {bad_datasets}. Allowed: {sorted(valid_datasets)}")
    if "C3" in conditions and "pubmedqa" not in datasets:
        raise ValueError("ERROR: Condition C3 requires 'pubmedqa' dataset in --datasets.")
    return datasets


def get_condition_questions(cond: str, datasets: list, medqa_all: list, pubmedqa_all: list, u2_questions: list) -> list:
    """Resolve question list for a condition respecting dataset filter."""
    if cond in ("C1", "C2"):
        q_list = []
        if "medqa" in datasets:
            q_list += medqa_all
        if "pubmedqa" in datasets:
            q_list += pubmedqa_all
        return q_list
    elif cond == "C3":
        return list(pubmedqa_all)
    elif cond in ("C4", "C5"):
        return list(u2_questions)
    return []


def assert_appended_row_valid(file_path, expected_id=None):
    """Cheap guard: assert the last appended line in file_path parses as a JSON object with valid schema."""
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

    # Schema consistency check: ensure canonical evaluation fields are present
    required_keys = ["pred", "parsed", "correct", "truncated"]
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise AssertionError(
            f"Write verification failed on {file_path}: missing required schema keys {missing}. "
            f"Keys found: {sorted(obj.keys())}. Line snippet: {last_line[:120]!r}"
        )
    # Check for leaked private/generator keys
    leaked = [k for k in obj if k in ("_pred", "_parsed", "_pred_source")]
    if leaked:
        raise AssertionError(
            f"Write verification failed on {file_path}: unmapped generator keys found: {leaked}. "
            f"Line snippet: {last_line[:120]!r}"
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
            max_new = (
                qcfg.get("max_new_tokens_rag", 40)
                if condition in ("C2", "C5")
                else (qcfg.get("max_new_tokens_context", 40) if condition == "C3" else qcfg.get("max_new_tokens_baseline", 16))
            )
            results = llm.generate_safe(batch_prompts, batch_letters, max_new)
        sec_batch = time.time() - t0_batch
        sec_per_q = sec_batch / len(batch) if batch else 0.0

        mode_name = "rag" if condition in ("C2", "C5") else ("context" if condition == "C3" else "baseline")

        for q, res in zip(batch, results):
            row = dict(q)
            for k, v in res.items():
                row[k] = v

            # Map generator underscore keys to canonical names
            if "_pred" in row:
                row["pred"] = row.pop("_pred")
            if "_parsed" in row:
                row["parsed"] = row.pop("_parsed")
            if "_pred_source" in row:
                row["pred_source"] = row.pop("_pred_source")

            # Non-reasoning model parsing fallback
            if "pred" not in row:
                opts = options_map.get(q["id"], q.get("options", {}))
                parsed_pred = parse_answer(row.get("raw_output", ""), opts)
                parsed = parsed_pred is not None
                lp = row.get("letter_probs") or {}
                prob_pred_raw = max(lp, key=lp.get) if lp else None
                if parsed:
                    row["pred"] = parsed_pred
                    row["pred_source"] = "parsed"
                    row["parsed"] = True
                elif prob_pred_raw is not None:
                    row["pred"] = prob_pred_raw
                    row["pred_source"] = "logprob_fallback"
                    row["parsed"] = False
                else:
                    row["pred"] = None
                    row["pred_source"] = "none"
                    row["parsed"] = False

            if "truncated" not in row:
                row["truncated"] = False

            row["seconds"] = round(sec_per_q, 4)
            row["condition"] = condition
            row["mode"] = mode_name
            row["model"] = model_name
            row["load_source"] = getattr(llm, "load_source", None)

            lp = row.get("letter_probs") or {}
            row["confidence"] = max(lp.values()) if lp else None
            row["prob_pred"] = max(lp, key=lp.get) if lp else None

            # Canonical gold answer resolution
            gold = row.get("answer") if "answer" in row and row["answer"] is not None else row.get("answer_idx")
            row["gold"] = gold
            row["abstained"] = bool(row.get("pred") == "ABSTAIN")
            if "should_abstain" not in row:
                row["should_abstain"] = bool(q.get("should_abstain", False))

            # Evaluate correctness against canonical gold answer
            if gold is not None and row.get("pred") is not None:
                row["correct"] = bool(str(row["pred"]).strip().upper() == str(gold).strip().upper())
            else:
                row["correct"] = False

            # Citations for RAG and context conditions
            if condition in ("C2", "C5"):
                passages = evidence.get(q["id"], {}).get("passages", [])
                cites = parse_citations(row.get("raw_output", ""), len(passages))
                row["citations"] = cites
                row["cited_chunk_ids"] = [passages[c - 1]["chunk_id"] for c in cites if c - 1 < len(passages)]
                row["evidence_chunk_ids"] = [p["chunk_id"] for p in passages]
                row["top_rerank_score"] = passages[0]["rerank_score"] if passages else None
            elif condition == "C3":
                row["citations"] = parse_citations(row.get("raw_output", ""), 1)

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


def compute_loop_suspect(text: str) -> bool:
    """Descriptive heuristic: zlib compression ratio of last 2000 chars of output < 0.2."""
    if not text:
        return False
    tail = text[-2000:]
    if len(tail) < 200:
        return False
    compressed = zlib.compress(tail.encode("utf-8"))
    ratio = len(compressed) / len(tail)
    return ratio < 0.2


def build_sweep_row(
    q: dict,
    raw_output: str,
    parsed_pred: Any,
    parsed: bool,
    correct: bool,
    n_gen_tokens: int,
    finished_naturally: bool,
    sec_q: float,
    model_name: str,
) -> dict:
    """Pure function to construct a budget sweep row with all required evaluation schema keys.

    Required schema keys checked by assert_appended_row_valid:
      - 'id' (from q)
      - 'pred'
      - 'parsed'
      - 'correct'
      - 'truncated' (at the 4096 cap, truncated=True iff no EOS was emitted)
    """
    row = dict(q)
    row["raw_output"] = raw_output
    row["pred"] = parsed_pred
    row["parsed"] = parsed
    row["correct"] = correct
    row["truncated"] = not finished_naturally
    row["n_generated_tokens"] = n_gen_tokens
    row["finished_naturally"] = finished_naturally
    row["seconds"] = sec_q
    row["loop_suspect"] = compute_loop_suspect(raw_output)
    row["condition"] = "C1"
    row["model"] = model_name
    return row


def run_budget_sweep(args, cfg, mcfg, qcfg, v3_cfg, p1_val, out_base):
    """Run token-budget sweep for Pair B (greedy cap=4096 on validation split)."""
    if args.split != "validation":
        raise ValueError(f"ERROR: Budget sweep requires split='validation', got split='{args.split}'")

    if not os.path.exists(p1_val):
        sys.exit(f"ERROR: Validation questions not found at {p1_val}")
    p1_questions = read_jsonl(p1_val)

    # Hard assertion: Every single loaded row MUST have split == 'validation'
    for q in p1_questions:
        if q.get("split") != "validation":
            raise AssertionError(f"FATAL: Non-validation split detected! id={q.get('id')} split={q.get('split')}")

    # First N MedQA questions in file order
    medqa_val = [q for q in p1_questions if q.get("dataset") == "medqa"]
    sweep_n = args.sweep_n
    sweep_questions = medqa_val[:sweep_n]
    if len(sweep_questions) < sweep_n:
        raise ValueError(f"Requested sweep_n={sweep_n}, but only {len(sweep_questions)} MedQA val questions available.")
    assert all(q.get("split") == "validation" for q in sweep_questions), "FATAL: Question without split=='validation' in sweep subset"

    os.makedirs(out_base, exist_ok=True)
    out_file = os.path.join(out_base, f"{args.model}.jsonl")

    # Resume pattern: load existing IDs
    existing_rows = read_jsonl(out_file) if os.path.exists(out_file) else []
    existing_ids = {r["id"] for r in existing_rows}
    todo = [q for q in sweep_questions if q["id"] not in existing_ids]

    print(
        f"\n[{args.model}] BUDGET SWEEP START: N={len(sweep_questions)}, "
        f"already_done={len(existing_ids)}, remaining={len(todo)}, cap={args.sweep_cap}",
        flush=True,
    )

    device = args.device or get_device()
    if args.tiny:
        device = "cpu"

    llm = None
    if not (args.check_only or args.tiny):
        banner(f"LOADING MODEL FOR BUDGET SWEEP: {args.model} ({mcfg.get('id')}) on {device}")
        t0_load = time.time()
        llm = LLM(mcfg, device=device)
        print(f"Model loaded in {time.time() - t0_load:.1f}s (load_source={getattr(llm, 'load_source', 'default')})", flush=True)

    t0_sweep = time.time()
    max_sec = args.sweep_max_hours * 3600.0
    completed_all = True
    n_processed = 0

    for i, q in enumerate(todo):
        if (time.time() - t0_sweep) >= max_sec:
            print(f"[{args.model}] Reached max sweep hours ({args.sweep_max_hours}h). Stopping gracefully.", flush=True)
            completed_all = False
            break

        prompt = reasoning_baseline_prompt(q)
        letters = list(q.get("options", {}).keys())

        if args.check_only or args.tiny:
            # Synthetic / fast CPU mock
            raw_output = f"<think>\nReasoning trace for {q['id']}...\n</think>\nAnswer: {q.get('answer', 'A')}"
            finished_naturally = True
            n_gen_tokens = 250
            sec_q = 0.01
        else:
            wrapped = llm.wrap_reasoning(prompt)
            enc = llm.tok([wrapped], return_tensors="pt", padding=True, add_special_tokens=False)
            first_dev = next(llm.model.parameters()).device
            enc = {k: v.to(first_dev) for k, v in enc.items()}

            from transformers import GenerationConfig
            gcfg = GenerationConfig(
                max_new_tokens=args.sweep_cap,
                do_sample=False,
                eos_token_id=sorted(list(llm.eos)),
                pad_token_id=llm.pad_id,
                return_dict_in_generate=True,
            )
            if llm.device == "cuda" or (isinstance(llm.device, str) and llm.device.startswith("cuda")):
                torch.cuda.synchronize()
            t0_gen = time.perf_counter()
            with torch.inference_mode():
                out = llm.model.generate(**enc, generation_config=gcfg)
            if llm.device == "cuda" or (isinstance(llm.device, str) and llm.device.startswith("cuda")):
                torch.cuda.synchronize()
            sec_q = round(time.perf_counter() - t0_gen, 4)

            seqs = out.sequences if hasattr(out, "sequences") else out
            prompt_ids_len = enc["input_ids"].shape[1]
            all_gen_ids = seqs[0, prompt_ids_len:].tolist()

            eos_idx = next((idx for idx, t in enumerate(all_gen_ids) if t in llm.eos), None)
            if eos_idx is not None:
                finished_naturally = True
                n_gen_tokens = eos_idx + 1  # EOS counted
                gen_ids = all_gen_ids[:eos_idx]
                raw_output = llm.tok.decode(gen_ids, skip_special_tokens=True)
            else:
                finished_naturally = False
                n_gen_tokens = len(all_gen_ids)
                raw_output = llm.tok.decode(all_gen_ids, skip_special_tokens=True)

        parsed_pred = extract_final_answer_from_reasoning(raw_output, letters)
        if parsed_pred is None:
            parsed_pred = parse_answer(raw_output, q.get("options", {}))
        parsed = parsed_pred is not None

        gold = q.get("answer") if "answer" in q and q["answer"] is not None else q.get("answer_idx")
        correct = bool(parsed and str(parsed_pred).strip().upper() == str(gold).strip().upper())

        row = build_sweep_row(
            q=q,
            raw_output=raw_output,
            parsed_pred=parsed_pred,
            parsed=parsed,
            correct=correct,
            n_gen_tokens=n_gen_tokens,
            finished_naturally=finished_naturally,
            sec_q=sec_q,
            model_name=args.model,
        )

        append_jsonl(out_file, [row])
        assert_appended_row_valid(out_file, expected_id=row.get("id"))
        n_processed += 1

        if (n_processed % 10 == 0) or (i + 1 == len(todo)):
            print(f"[{args.model}] Processed {len(existing_ids) + n_processed}/{len(sweep_questions)}", flush=True)

    # Manifest creation
    total_done = len(read_jsonl(out_file)) if os.path.exists(out_file) else 0
    completed_model = completed_all and (total_done == len(sweep_questions))
    elapsed_hours = round((time.time() - t0_sweep) / 3600.0, 4)

    rule_file = "docs/BUDGET_DECISION_RULE.md"
    rule_hash = ""
    if os.path.exists(rule_file):
        with open(rule_file, "rb") as fh:
            rule_hash = hashlib.sha256(fh.read()).hexdigest()

    def _get_git_commit():
        try:
            import subprocess
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return os.environ.get("GIT_COMMIT", "unknown")

    manifest_data = {
        "git_commit_hash": _get_git_commit(),
        "rule_file_sha256": rule_hash,
        "generation_config": {
            "max_new_tokens": args.sweep_cap,
            "do_sample": False,
            "eos_token_id": sorted(list(llm.eos)) if llm is not None else [151645],
            "pad_token_id": llm.pad_id if llm is not None else 151643,
        },
        "torch_version": torch.__version__ if "torch" in sys.modules else "unknown",
        "transformers_version": sys.modules["transformers"].__version__ if "transformers" in sys.modules else "unknown",
        "dtype": str(mcfg.get("dtype", "float16")),
        "batch_size": args.batch_size or 1,
        "N": len(sweep_questions),
        "seed": 42,
        "completed": completed_model,
        "hours_used": elapsed_hours,
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = os.path.join(out_base, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2)
    with open(os.path.join(out_base, f"manifest_{args.model}.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2)
    print(f"[{args.model}] Manifest saved to {manifest_path} (completed={completed_model})", flush=True)
    print(f"\n=== BUDGET SWEEP COMPLETE for {args.model} (total_done={total_done}/{len(sweep_questions)}) ===\n", flush=True)


# ── Gemma Prompt Sensitivity Protocol & Helpers ───────────────────────────────

def prompt_v1(q: dict) -> str:
    """Variant 1: Baseline v3 C1 prompt (byte-identical to baseline_prompt)."""
    return baseline_prompt(q)


def prompt_v2(q: dict) -> str:
    """Variant 2: Clinical evaluation phrasing with parenthesized option labels."""
    opts_str = "\n".join(f"({k}) {v}" for k, v in q["options"].items())
    return (
        "The following is a multiple-choice medical examination question. "
        "Please evaluate the clinical scenario and select the single best option.\n\n"
        f"Question: {q['question']}\n"
        f"Options:\n{opts_str}\n\n"
        "Provide the letter corresponding to the correct answer."
    )


def prompt_v3(q: dict) -> str:
    """Variant 3: Board-style presentation with bracketed choice labels."""
    opts_str = "\n".join(f"[{k}] {v}" for k, v in q["options"].items())
    return (
        "Carefully read the patient presentation below and identify the correct diagnostic or management step.\n\n"
        f"Question: {q['question']}\n"
        f"Choices:\n{opts_str}\n\n"
        "State the letter of the single correct option."
    )


PROMPT_VARIANTS = {
    "V1": prompt_v1,
    "V2": prompt_v2,
    "V3": prompt_v3,
}

PROMPT_TEMPLATE_V1 = (
    "You are a medical expert answering a multiple-choice question.\n\n"
    "Question: {question}\nOptions:\n{options}\n\n"
    "Answer with the letter of the correct option."
)

PROMPT_TEMPLATE_V2 = (
    "The following is a multiple-choice medical examination question. "
    "Please evaluate the clinical scenario and select the single best option.\n\n"
    "Question: {question}\nOptions:\n{options}\n\n"
    "Provide the letter corresponding to the correct answer."
)

PROMPT_TEMPLATE_V3 = (
    "Carefully read the patient presentation below and identify the correct diagnostic or management step.\n\n"
    "Question: {question}\nChoices:\n{options}\n\n"
    "State the letter of the single correct option."
)

PROMPT_TEMPLATE_HASHES = {
    "V1": hashlib.sha256(PROMPT_TEMPLATE_V1.encode("utf-8")).hexdigest(),
    "V2": hashlib.sha256(PROMPT_TEMPLATE_V2.encode("utf-8")).hexdigest(),
    "V3": hashlib.sha256(PROMPT_TEMPLATE_V3.encode("utf-8")).hexdigest(),
}


def build_prompt_sens_row(
    q: dict,
    variant: str,
    raw_output: str,
    parsed_pred: Any,
    parsed: bool,
    correct: bool,
    n_gen_tokens: int,
    finished_naturally: bool,
    sec_q: float,
    model_name: str,
) -> dict:
    """Pure function to construct a Gemma prompt-sensitivity row.

    Required schema keys checked by assert_appended_row_valid:
      - 'id' (from q)
      - 'pred'
      - 'parsed'
      - 'correct'
      - 'truncated' (for direct-answer 16-token C1: truncated=True iff no parseable answer letter)

    Additional schema keys:
      - 'variant' ('V1', 'V2', 'V3')
      - 'n_generated_tokens'
      - 'finished_naturally'
      - 'hit_cap_without_eos'
      - 'seconds'
      - 'raw_output'
      - 'model'
      - 'condition' ('C1')
    """
    row = dict(q)
    row["variant"] = variant
    row["raw_output"] = raw_output
    row["pred"] = parsed_pred
    row["parsed"] = parsed
    row["correct"] = correct
    row["truncated"] = not parsed
    row["n_generated_tokens"] = n_gen_tokens
    row["finished_naturally"] = finished_naturally
    row["hit_cap_without_eos"] = not finished_naturally
    row["seconds"] = sec_q
    row["condition"] = "C1"
    row["model"] = model_name
    return row


def run_gemma_prompt_sens(args, cfg, mcfg, qcfg, v3_cfg, p1_val, out_base):
    """Run prompt-sensitivity check for Gemma pair on validation MedQA across V1, V2, V3."""
    if args.split != "validation":
        raise ValueError(f"ERROR: Gemma prompt-sensitivity check requires split='validation', got split='{args.split}'")

    if not os.path.exists(p1_val):
        sys.exit(f"ERROR: Validation questions not found at {p1_val}")
    p1_questions = read_jsonl(p1_val)

    # Hard assertion: Every single loaded row MUST have split == 'validation'
    for q in p1_questions:
        if q.get("split") != "validation":
            raise AssertionError(f"FATAL: Non-validation split detected! id={q.get('id')} split={q.get('split')}")

    # First N MedQA questions in file order
    medqa_val = [q for q in p1_questions if q.get("dataset") == "medqa"]
    sens_n = args.sens_n
    sweep_questions = medqa_val[:sens_n]
    if len(sweep_questions) < sens_n:
        raise ValueError(f"Requested sens_n={sens_n}, but only {len(sweep_questions)} MedQA val questions available.")
    assert all(q.get("split") == "validation" for q in sweep_questions), "FATAL: Question without split=='validation' in sweep subset"

    os.makedirs(out_base, exist_ok=True)
    out_file = os.path.join(out_base, f"{args.model}.jsonl")

    # Resume pattern: load existing (id, variant) pairs
    existing_rows = read_jsonl(out_file) if os.path.exists(out_file) else []
    done_pairs = {(r["id"], r.get("variant")) for r in existing_rows if "id" in r and "variant" in r}

    variants = ["V1", "V2", "V3"]
    total_expected = len(sweep_questions) * len(variants)
    print(
        f"\n=== GEMMA PROMPT-SENSITIVITY: {args.model} | N={len(sweep_questions)} | "
        f"Variants={variants} | Already done: {len(done_pairs)}/{total_expected} ===",
        flush=True,
    )

    if len(done_pairs) >= total_expected:
        print(f"[{args.model}] All {total_expected} (id, variant) pairs already completed in {out_file}.", flush=True)
        return

    # Device and model loading
    device = args.device or get_device()
    if args.tiny:
        device = "cpu"

    if args.check_only or args.tiny:
        llm = None
    else:
        banner(f"LOADING MODEL FOR PROMPT SENSITIVITY: {args.model} ({mcfg.get('id')}) on {device}")
        t0_load = time.time()
        llm = LLM(mcfg, device=device)
        print(f"Model loaded in {time.time() - t0_load:.1f}s (load_source={getattr(llm, 'load_source', 'default')})", flush=True)

    batch_size = args.batch_size or mcfg.get("batch_size", qcfg.get("batch_size", 8))
    t0_sens = time.time()
    max_sec = args.sens_max_hours * 3600.0
    completed_all = True
    n_processed = 0

    for variant in variants:
        if (time.time() - t0_sens) >= max_sec:
            print(f"[{args.model}] Reached max sensitivity hours ({args.sens_max_hours}h). Stopping gracefully.", flush=True)
            completed_all = False
            break

        todo = [q for q in sweep_questions if (q["id"], variant) not in done_pairs]
        if not todo:
            print(f"[{args.model}] Variant {variant} already complete ({len(sweep_questions)} rows). Skipping.", flush=True)
            continue

        print(f"[{args.model}] Starting Variant {variant}: {len(todo)} questions to process...", flush=True)
        builder = PROMPT_VARIANTS[variant]

        for i in range(0, len(todo), batch_size):
            if (time.time() - t0_sens) >= max_sec:
                print(f"[{args.model}] Reached max sensitivity hours ({args.sens_max_hours}h). Stopping gracefully.", flush=True)
                completed_all = False
                break

            batch = todo[i : i + batch_size]
            t0_batch = time.perf_counter()

            if args.check_only or args.tiny:
                # Fast mock for tests / check-only
                results = []
                for q in batch:
                    raw_out = f"Answer: {q.get('answer', 'A')}"
                    results.append({
                        "raw_output": raw_out,
                        "n_gen_tokens": 2,
                        "finished_naturally": True,
                        "seconds": 0.01,
                    })
            else:
                batch_prompts = [builder(q) for q in batch]
                batch_letters = [list(q.get("options", {}).keys()) for q in batch]

                # Use llm.generate_safe with max_new_tokens_baseline (16)
                gen_res = llm.generate_safe(batch_prompts, batch_letters, max_new_tokens=16)
                sec_per_q = round((time.perf_counter() - t0_batch) / len(batch), 4) if batch else 0.0

                results = []
                for res in gen_res:
                    raw_out = res.get("raw_output", "")
                    n_gen = res.get("gen_tokens", 0)
                    fin_nat = (n_gen < 16)
                    results.append({
                        "raw_output": raw_out,
                        "n_gen_tokens": n_gen,
                        "finished_naturally": fin_nat,
                        "seconds": sec_per_q,
                    })

            for q, r_info in zip(batch, results):
                raw_output = r_info["raw_output"]
                n_gen_tokens = r_info["n_gen_tokens"]
                finished_naturally = r_info["finished_naturally"]
                sec_q = r_info["seconds"]

                parsed_pred = parse_answer(raw_output, q.get("options", {}))
                parsed = (parsed_pred is not None)
                gold = q.get("answer") if "answer" in q and q["answer"] is not None else q.get("answer_idx")
                correct = bool(parsed and str(parsed_pred).strip().upper() == str(gold).strip().upper())

                row = build_prompt_sens_row(
                    q=q,
                    variant=variant,
                    raw_output=raw_output,
                    parsed_pred=parsed_pred,
                    parsed=parsed,
                    correct=correct,
                    n_gen_tokens=n_gen_tokens,
                    finished_naturally=finished_naturally,
                    sec_q=sec_q,
                    model_name=args.model,
                )

                append_jsonl(out_file, [row])
                assert_appended_row_valid(out_file, expected_id=row.get("id"))
                done_pairs.add((row["id"], variant))
                n_processed += 1

            print(f"  [{args.model} | {variant}] Done {len([p for p in done_pairs if p[1] == variant])}/{len(sweep_questions)}", flush=True)

    # Manifest creation
    total_done = len(read_jsonl(out_file)) if os.path.exists(out_file) else 0
    completed_model = completed_all and (total_done == total_expected)
    elapsed_hours = round((time.time() - t0_sens) / 3600.0, 4)

    rule_file = "docs/GEMMA_PROMPT_SENSITIVITY_RULE.md"
    rule_hash = ""
    if os.path.exists(rule_file):
        with open(rule_file, "rb") as fh:
            rule_hash = hashlib.sha256(fh.read()).hexdigest()

    def _get_git_commit():
        try:
            import subprocess
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return os.environ.get("GIT_COMMIT", "unknown")

    manifest_data = {
        "git_commit_hash": _get_git_commit(),
        "rule_file_sha256": rule_hash,
        "prompt_template_hashes": PROMPT_TEMPLATE_HASHES,
        "generation_config": {
            "max_new_tokens": 16,
            "do_sample": False,
        },
        "torch_version": torch.__version__ if "torch" in sys.modules else "unknown",
        "transformers_version": sys.modules["transformers"].__version__ if "transformers" in sys.modules else "unknown",
        "dtype": str(mcfg.get("dtype", "float32")),
        "batch_size": batch_size,
        "N": len(sweep_questions),
        "variants": variants,
        "seed": 42,
        "completed": completed_model,
        "hours_used": elapsed_hours,
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = os.path.join(out_base, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2)
    with open(os.path.join(out_base, f"manifest_{args.model}.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2)
    print(f"[{args.model}] Manifest saved to {manifest_path} (completed={completed_model})", flush=True)
    print(f"\n=== GEMMA PROMPT-SENSITIVITY COMPLETE for {args.model} (total_done={total_done}/{total_expected}) ===\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="v3 Matched-Pair Generic vs Medical QA Pipeline")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base.yaml config")
    parser.add_argument("--work", default="work", help="Working directory containing phase1..7")
    parser.add_argument("--model", required=True, help="Model name from configs/base.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="test", help="Dataset split")
    parser.add_argument("--conditions", default="C1,C2,C3,C4,C5", help="Comma-separated conditions to run")
    parser.add_argument("--datasets", default="medqa,pubmedqa", help="Comma-separated subset of medqa,pubmedqa (default: medqa,pubmedqa)")
    parser.add_argument("--out", default=None, help="Output directory (defaults to outputs/v3/<model>)")
    parser.add_argument("--pilot", action="store_true", help="Run stratified N=50 validation pilot")
    parser.add_argument("--pilot-n", type=int, default=50, help="Pilot question count (default: 50)")
    parser.add_argument("--check-only", action="store_true", help="Format and parse check only")
    parser.add_argument("--tiny", action="store_true", help="Run tiny CPU check with tiny_model")
    parser.add_argument("--device", default=None, help="Torch device override (auto, cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--budget-sweep", action="store_true", help="Run token-budget sweep for Pair B on validation")
    parser.add_argument("--sweep-n", type=int, default=100, help="Number of validation MedQA questions to sweep (default: 100)")
    parser.add_argument("--sweep-max-hours", type=float, default=10.0, help="Maximum execution time in hours before graceful stop (default: 10.0)")
    parser.add_argument("--sweep-cap", type=int, default=4096, help="Maximum new tokens for sweep generation (default: 4096)")
    parser.add_argument("--gemma-prompt-sens", action="store_true", help="Run prompt-sensitivity check for Gemma pair on validation")
    parser.add_argument("--sens-n", type=int, default=200, help="Number of validation MedQA questions (default: 200)")
    parser.add_argument("--sens-max-hours", type=float, default=3.0, help="Maximum execution time in hours (default: 3.0)")
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

    datasets = parse_and_validate_datasets(args.datasets, conditions)

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
        if args.budget_sweep:
            out_base = "/kaggle/working/outputs/sweeps/budget_pair_b" if os.path.exists("/kaggle/working") else "outputs/sweeps/budget_pair_b"
        elif args.gemma_prompt_sens:
            out_base = "/kaggle/working/outputs/sweeps/gemma_prompt_sensitivity" if os.path.exists("/kaggle/working") else "outputs/sweeps/gemma_prompt_sensitivity"
        else:
            out_base = f"/kaggle/working/outputs/v3/{args.model}" if os.path.exists("/kaggle/working") else f"outputs/v3/{args.model}"
    os.makedirs(out_base, exist_ok=True)

    # Locate question files
    p1_test, p1_val, u2_test, u2_val = find_question_files(args.work)

    # Branch immediately to budget sweep if requested (validation only)
    if args.budget_sweep:
        run_budget_sweep(args, cfg, mcfg, qcfg, v3_cfg, p1_val, out_base)
        return

    # Branch immediately to Gemma prompt-sensitivity sweep if requested (validation only)
    if args.gemma_prompt_sens:
        run_gemma_prompt_sens(args, cfg, mcfg, qcfg, v3_cfg, p1_val, out_base)
        return

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
        "datasets": datasets,
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
        q_list = get_condition_questions(cond, datasets, medqa_all, pubmedqa_all, u2_questions)
        if cond == "C2":
            ev_list = evidence_c2
            bs = batch_size_rag
        elif cond == "C3":
            ev_list = {}
            bs = batch_size_rag
        elif cond == "C5":
            ev_list = evidence_c5
            bs = batch_size_rag
        else:
            ev_list = {}
            bs = batch_size

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

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_mems = [
                f"GPU {i}: {torch.cuda.max_memory_allocated(i) / (1024**3):.2f} GiB"
                for i in range(torch.cuda.device_count())
            ]
            print(f"[{cond}] Peak VRAM: {', '.join(gpu_mems)}", flush=True)
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

    print(f"\n=== QA PIPELINE V3 COMPLETE for {args.model} (conditions: {conditions}) ===\n", flush=True)


if __name__ == "__main__":
    main()
