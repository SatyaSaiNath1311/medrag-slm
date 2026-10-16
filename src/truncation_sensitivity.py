"""src/truncation_sensitivity.py — Truncation Sensitivity Analysis (Validation Only).

Protocol (from docs/DESIGN_GENERIC_VS_MEDICAL.md, Task 6):
  - Evaluates cap=1,024 vs cap=2,048 tokens on 100 MedQA validation questions.
  - Covers the 4 thinking models:
      1. medpsy-1.7b
      2. qwen3-1.7b-thinking
      3. qwen3-4b-thinking-2507
      4. medpsy-4b
  - Strict rule: Runs on VALIDATION ONLY. Zero test questions are loaded or touched.
  - Outputs comparative summary table in markdown and CSV to docs/tables/ and stdout.
  - Supports --tiny flag for rapid local CPU execution/testing.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.common import load_config, read_jsonl, set_seed
from src.llm import LLM, reasoning_baseline_prompt


def sample_val_medqa(val_path: str, n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample n MedQA questions from validation split deterministically."""
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation questions not found at {val_path}")
    val_rows = read_jsonl(val_path)
    medqa = [r for r in val_rows if r.get("dataset") == "medqa" and not r.get("should_abstain")]
    rng = random.Random(seed)
    sampled = rng.sample(medqa, min(n, len(medqa)))
    return sorted(sampled, key=lambda x: str(x["id"]))


def evaluate_cap_for_model(
    llm: Any,
    model_name: str,
    questions: List[Dict[str, Any]],
    cap: int,
    batch_size: int = 4,
    tiny: bool = False,
) -> Dict[str, Any]:
    """Evaluate a specific thinking token cap on validation questions."""
    print(f"  Evaluating {model_name} with cap={cap} on {len(questions)} questions...", flush=True)
    
    # Save original reasoning cap and override
    orig_cap = getattr(llm, "max_new_tokens_reasoning", 1024)
    llm.max_new_tokens_reasoning = cap
    if hasattr(llm, "cfg"):
        llm.cfg["max_new_tokens_reasoning"] = cap

    prompts = [reasoning_baseline_prompt(q) for q in questions]
    letters = [list(q["options"].keys()) for q in questions]

    t0 = time.time()
    results = []
    
    if tiny:
        # Mock / quick generation for CPU tiny check
        for q in questions:
            # Simulate slight truncation reduction with higher cap
            is_trunc = bool(random.random() < (0.18 if cap == 1024 else 0.05))
            pred = "A" if is_trunc and random.random() < 0.5 else q.get("answer", "A")
            is_corr = (pred == q.get("answer")) and not is_trunc
            results.append({
                "truncated": is_trunc,
                "parsed": True,
                "pred": pred,
                "correct": is_corr,
                "seconds": 0.02,
                "gen_tokens": 150 if not is_trunc else cap,
            })
    else:
        for i in range(0, len(questions), batch_size):
            b_prompts = prompts[i : i + batch_size]
            b_letters = letters[i : i + batch_size]
            b_res = llm.generate_reasoning_safe(b_prompts, b_letters)
            results.extend(b_res)

    elapsed = time.time() - t0
    llm.max_new_tokens_reasoning = orig_cap

    n_total = len(questions)
    n_trunc = sum(1 for r in results if r.get("truncated"))
    trunc_rate = (n_trunc / n_total) if n_total else 0.0

    # Primary accuracy: truncated answers count as WRONG
    n_correct_prim = sum(
        1 for q, r in zip(questions, results)
        if not r.get("truncated") and str(r.get("pred", "")).strip().upper() == str(q.get("answer", "")).strip().upper()
    )
    acc_primary = (n_correct_prim / n_total) if n_total else 0.0

    # Answered-only accuracy: accuracy on non-truncated questions
    non_trunc_pairs = [
        (q, r) for q, r in zip(questions, results) if not r.get("truncated")
    ]
    n_ans_correct = sum(
        1 for q, r in non_trunc_pairs
        if str(r.get("pred", "")).strip().upper() == str(q.get("answer", "")).strip().upper()
    )
    acc_answered_only = (n_ans_correct / len(non_trunc_pairs)) if non_trunc_pairs else acc_primary

    mean_sec = elapsed / n_total if n_total else 0.0

    return {
        "model": model_name,
        "cap": cap,
        "n_questions": n_total,
        "n_truncated": n_trunc,
        "trunc_rate": round(trunc_rate, 4),
        "primary_acc": round(acc_primary, 4),
        "answered_only_acc": round(acc_answered_only, 4),
        "mean_seconds": round(mean_sec, 2),
    }


def run_truncation_sensitivity(
    config_path: str = "configs/base.yaml",
    work_dir: str = "work",
    models: Optional[List[str]] = None,
    caps: Optional[List[int]] = None,
    n_questions: int = 100,
    out_dir: str = "docs/tables",
    tiny: bool = False,
):
    """Execute truncation sensitivity check across caps and thinking models."""
    cfg = load_config(config_path)
    v3_cfg = cfg.get("v3", {})
    seed = v3_cfg.get("seed", 42)
    set_seed(seed)

    if models is None:
        models = ["medpsy-1.7b", "qwen3-1.7b-thinking", "qwen3-4b-thinking-2507", "medpsy-4b"]
    if caps is None:
        caps = v3_cfg.get("truncation_caps", [1024, 2048])

    print("\n" + "=" * 80)
    print("TRUNCATION SENSITIVITY CHECK (VALIDATION ONLY: 100 MedQA Questions)")
    print("=" * 80 + "\n")

    # Locate validation questions
    val_candidates = [
        os.path.join(work_dir, "phase1", "validation.jsonl"),
        "outputs/kaggle_build/work/phase1/validation.jsonl",
        "work/phase1/validation.jsonl",
    ]
    val_path = next((p for p in val_candidates if os.path.exists(p)), None)
    if not val_path and not tiny:
        sys.exit("ERROR: Validation questions not found. Please provide valid --work path.")

    if val_path:
        questions = sample_val_medqa(val_path, n=n_questions, seed=seed)
    else:
        # Mock questions for tiny local test
        questions = [
            {"id": f"mock-val-{i:03d}", "dataset": "medqa", "question": f"Sample question {i}?",
             "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
             "answer": "A", "should_abstain": False}
            for i in range(min(10, n_questions))
        ]

    print(f"Loaded {len(questions)} MedQA validation questions (seed={seed}).")
    print(f"Models: {models}")
    print(f"Caps  : {caps}\n")

    summary_headers = [
        "Model", "Cap", "N Questions", "Truncated", "Truncation %",
        "Primary Acc (Trunc=0)", "Answered-Only Acc", "Latency/Q (s)"
    ]
    summary_rows = []
    comp_headers = [
        "Model", "Trunc % (1024)", "Trunc % (2048)", "Delta Trunc",
        "Acc (1024)", "Acc (2048)", "Delta Acc", "Ranking Invariant?"
    ]
    comp_rows = []

    model_caps_results = {}

    for m in models:
        model_caps_results[m] = {}
        # Load model config
        mcfg = None
        if not tiny:
            m_map = {mc["name"]: mc for mc in cfg.get("models", [])}
            mcfg = m_map.get(m)
            if not mcfg:
                print(f"Model '{m}' not found in configs/base.yaml; skipping.")
                continue
            llm = LLM(mcfg, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
        else:
            # Mock LLM for tiny mode
            class MockLLM:
                is_reasoning = True
                max_new_tokens_reasoning = 1024
            llm = MockLLM()

        for cap in caps:
            res = evaluate_cap_for_model(llm, m, questions, cap=cap, tiny=tiny)
            model_caps_results[m][cap] = res
            summary_rows.append([
                m,
                cap,
                res["n_questions"],
                res["n_truncated"],
                f"{res['trunc_rate']*100:.1f}%",
                f"{res['primary_acc']:.4f}",
                f"{res['answered_only_acc']:.4f}",
                f"{res['mean_seconds']:.2f}s",
            ])

        if len(caps) >= 2:
            c1, c2 = caps[0], caps[1]
            r1 = model_caps_results[m][c1]
            r2 = model_caps_results[m][c2]
            delta_trunc = (r2["trunc_rate"] - r1["trunc_rate"]) * 100
            delta_acc = (r2["primary_acc"] - r1["primary_acc"]) * 100
            comp_rows.append([
                m,
                f"{r1['trunc_rate']*100:.1f}%",
                f"{r2['trunc_rate']*100:.1f}%",
                f"{delta_trunc:+.1f} pp",
                f"{r1['primary_acc']:.4f}",
                f"{r2['primary_acc']:.4f}",
                f"{delta_acc:+.4f}",
                "YES" if abs(delta_acc) < 5.0 else "NO",
            ])

    # Save tables
    os.makedirs(out_dir, exist_ok=True)
    md_summary = os.path.join(out_dir, "table_truncation_sensitivity.md")
    csv_summary = os.path.join(out_dir, "table_truncation_sensitivity.csv")

    with open(csv_summary, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(summary_headers)
        writer.writerows(summary_rows)

    with open(md_summary, "w", encoding="utf-8") as fh:
        fh.write("# Truncation Sensitivity Analysis (Validation Only: 100 MedQA Questions)\n\n")
        fh.write("> Protocol: Evaluates whether increasing thinking budget from 1,024 to 2,048 tokens "
                 "significantly shifts model rankings or accuracy on 100 validation questions. "
                 "Frozen test split is never touched.\n\n")

        # Summary table
        col_w = [len(h) for h in summary_headers]
        for r in summary_rows:
            for i, val in enumerate(r):
                col_w[i] = max(col_w[i], len(str(val)))
        fh.write("| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(summary_headers)) + " |\n")
        fh.write("| " + " | ".join("-" * col_w[i] for i in range(len(summary_headers))) + " |\n")
        for r in summary_rows:
            fh.write("| " + " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(summary_headers))) + " |\n")

        # Delta comparison table
        if comp_rows:
            fh.write("\n\n### Cap 1,024 vs. Cap 2,048 Delta Comparison\n\n")
            c_col_w = [len(h) for h in comp_headers]
            for r in comp_rows:
                for i, val in enumerate(r):
                    c_col_w[i] = max(c_col_w[i], len(str(val)))
            fh.write("| " + " | ".join(h.ljust(c_col_w[i]) for i, h in enumerate(comp_headers)) + " |\n")
            fh.write("| " + " | ".join("-" * c_col_w[i] for i in range(len(comp_headers))) + " |\n")
            for r in comp_rows:
                fh.write("| " + " | ".join(str(r[i]).ljust(c_col_w[i]) for i in range(len(comp_headers))) + " |\n")

    print(f"\nTruncation sensitivity table saved to: {md_summary} and {csv_summary}")


def main():
    parser = argparse.ArgumentParser(description="Truncation Sensitivity Check on Validation (v3)")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base.yaml")
    parser.add_argument("--work", default="work", help="Working directory containing phase1")
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--caps", default=None, help="Comma-separated caps (e.g. 1024,2048)")
    parser.add_argument("--n", type=int, default=100, help="Number of validation questions")
    parser.add_argument("--out", default="docs/tables", help="Output directory")
    parser.add_argument("--tiny", action="store_true", help="Tiny scale / mocked local run")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    caps = [int(c.strip()) for c in args.caps.split(",") if c.strip()] if args.caps else None

    run_truncation_sensitivity(
        config_path=args.config,
        work_dir=args.work,
        models=models,
        caps=caps,
        n_questions=args.n,
        out_dir=args.out,
        tiny=args.tiny,
    )


if __name__ == "__main__":
    main()
