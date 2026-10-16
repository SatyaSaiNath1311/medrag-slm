#!/usr/bin/env python3
"""
scripts/budget.py - GPU Quota Budget Calculator for MedRAG-SLM v3

Calculates GPU hours required for pending v2 completions and v3 matched-pair
evaluations based on empirical seconds-per-question (s/q) logged during prior runs.

Usage:
    python3 scripts/budget.py [--hours 30.0] [--pilot-only] [--json]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


# Default fallback empirical rates (seconds per question) on Kaggle T4 GPU
# Derived from logs in outputs/kaggle_qa/
EMPIRICAL_RATES: Dict[str, Dict[str, float]] = {
    # 1.7B non-thinking control
    "qwen3-1.7b": {
        "baseline": 0.12,
        "rag": 0.70,
        "pubmedqa_base": 0.12,
        "pubmedqa_context": 0.65,
        "unanswerable": 0.12,
    },
    # 1.7B thinking models (cap 1024 tokens)
    "medpsy-1.7b": {
        "baseline": 18.5,
        "rag": 32.0,
        "pubmedqa_base": 15.0,
        "pubmedqa_context": 18.0,
        "unanswerable": 16.0,
    },
    "qwen3-1.7b-thinking": {
        "baseline": 18.5,
        "rag": 32.0,
        "pubmedqa_base": 15.0,
        "pubmedqa_context": 18.0,
        "unanswerable": 16.0,
    },
    # 3B-4B non-thinking models
    "phi4-mini": {
        "baseline": 0.17,
        "rag": 1.14,
        "pubmedqa_base": 0.17,
        "pubmedqa_context": 1.05,
        "unanswerable": 0.17,
    },
    "smollm3-3b": {
        "baseline": 0.17,
        "rag": 0.95,
        "pubmedqa_base": 0.17,
        "pubmedqa_context": 0.90,
        "unanswerable": 0.17,
    },
    "qwen3-4b": {
        "baseline": 0.23,
        "rag": 1.47,
        "pubmedqa_base": 0.23,
        "pubmedqa_context": 1.35,
        "unanswerable": 0.23,
    },
    "gemma3-4b": {
        "baseline": 0.63,
        "rag": 3.40,
        "pubmedqa_base": 0.63,
        "pubmedqa_context": 3.10,
        "unanswerable": 0.63,
    },
    "medgemma-1.5-4b-it": {
        "baseline": 0.65,
        "rag": 3.50,
        "pubmedqa_base": 0.65,
        "pubmedqa_context": 3.20,
        "unanswerable": 0.65,
    },
    # 4B thinking models (cap 1024 tokens)
    "medpsy-4b": {
        "baseline": 37.58,
        "rag": 63.93,
        "pubmedqa_base": 30.28,
        "pubmedqa_context": 30.28,
        "unanswerable": 32.0,
    },
    "qwen3-4b-thinking-2507": {
        "baseline": 37.58,
        "rag": 63.93,
        "pubmedqa_base": 30.28,
        "pubmedqa_context": 30.28,
        "unanswerable": 32.0,
    },
}

# Question counts per dataset
QUESTION_COUNTS_FULL = {
    "C1": 500,  # MedQA Baseline
    "C2": 500,  # MedQA Full RAG
    "C3": 500,  # PubMedQA Baseline
    "C4": 500,  # PubMedQA Context
    "C5": 140,  # Unanswerable v2
}

QUESTION_COUNTS_PILOT = {
    "C1": 50,
    "C2": 50,
    "C3": 50,
    "C4": 50,
    "C5": 50,
}


def parse_log_rates(log_dir: Path) -> Dict[str, Dict[str, float]]:
    """Scan existing runner logs to extract observed s/q rates."""
    extracted = {}
    for log_path in log_dir.glob("**/medrag-slm-runner.log"):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            # Match baseline and context lines
            base_m = re.search(r"Baseline s/question:\s*([\d\.]+)", content)
            ctx_m = re.search(r"Context s/question\s*:\s*([\d\.]+)", content)
            rag_m = re.search(r"s/question\s*:\s*([\d\.]+)", content)
            # Match results summary line (e.g. "qwen3-1.7b    DONE ... 0.1192/0.7009")
            summary_m = re.search(
                r"(\b[\w\-]+)\s+(?:DONE|PROBLEM)\s+[\d\.]+\s+[\d\.]+\s+[\d\.\/]+\s+([\d\.]+)/([\d\.]+)",
                content,
            )
            if summary_m:
                model_name = summary_m.group(1)
                b_rate = float(summary_m.group(2))
                r_rate = float(summary_m.group(3))
                extracted[model_name] = {
                    "baseline": b_rate,
                    "rag": r_rate,
                }
        except Exception:
            pass
    return extracted


def calculate_task_hours(
    model: str,
    condition: str,
    n_questions: int,
    rates: Dict[str, Dict[str, float]],
) -> float:
    m_rates = rates.get(model, rates["qwen3-4b"])
    if condition == "C1":
        sq = m_rates.get("baseline", 1.0)
    elif condition == "C2":
        sq = m_rates.get("rag", 2.0)
    elif condition == "C3":
        sq = m_rates.get("pubmedqa_base", m_rates.get("baseline", 1.0))
    elif condition == "C4":
        sq = m_rates.get("pubmedqa_context", m_rates.get("rag", 2.0))
    elif condition == "C5":
        sq = m_rates.get("unanswerable", m_rates.get("baseline", 1.0))
    else:
        sq = 1.0
    total_seconds = n_questions * sq
    return total_seconds / 3600.0


def main():
    parser = argparse.ArgumentParser(description="MedRAG-SLM GPU Quota Budget Calculator")
    parser.add_argument(
        "--hours",
        type=float,
        default=30.0,
        help="Total GPU quota hours available (default: 30.0).",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Calculate budget for validation pilots (N=50 per dataset) only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary.",
    )
    parser.add_argument(
        "--include-v2-pending",
        action="store_true",
        default=True,
        help="Include pending v2 runs in the budget calculation (default: True).",
    )
    args = parser.parse_args()

    # Dynamic log rate updates if present
    log_rates = parse_log_rates(REPO_ROOT / "outputs" / "kaggle_qa")
    active_rates = dict(EMPIRICAL_RATES)
    for m, r in log_rates.items():
        if m in active_rates:
            active_rates[m].update(r)

    q_counts = QUESTION_COUNTS_PILOT if args.pilot_only else QUESTION_COUNTS_FULL

    plan: List[Dict[str, Any]] = []

    # 1. Pending v2 Runs
    if args.include_v2_pending and not args.pilot_only:
        v2_tasks = [
            ("smollm3-3b", "Unanswerable v2", 140, "unanswerable"),
            ("phi4-mini", "Unanswerable v2", 140, "unanswerable"),
            ("qwen3-4b", "Unanswerable v2", 140, "unanswerable"),
            ("gemma3-4b", "Unanswerable v2", 140, "unanswerable"),
            ("medpsy-4b", "Unanswerable v2 (Baseline)", 140, "unanswerable"),
            ("medpsy-4b", "PubMedQA Baseline", 500, "pubmedqa_base"),
            ("medpsy-4b", "PubMedQA Full RAG", 500, "rag"),
        ]
        for m, desc, nq, rate_key in v2_tasks:
            sq = active_rates[m].get(rate_key, 1.0)
            hrs = (nq * sq) / 3600.0
            plan.append({
                "phase": "v2_pending",
                "model": m,
                "description": desc,
                "n_questions": nq,
                "rate_sq": round(sq, 2),
                "gpu_hours": round(hrs, 3),
            })

    # 2. v3 Matched-Pair Models
    v3_models = [
        "medpsy-1.7b",
        "qwen3-1.7b-thinking",
        "qwen3-4b-thinking-2507",
        "medgemma-1.5-4b-it",
    ]

    for m in v3_models:
        for cond in ["C1", "C2", "C3", "C4", "C5"]:
            nq = q_counts[cond]
            hrs = calculate_task_hours(m, cond, nq, active_rates)
            desc_map = {
                "C1": "MedQA Baseline",
                "C2": "MedQA Full RAG",
                "C3": "PubMedQA Baseline",
                "C4": "PubMedQA +Context",
                "C5": "Unanswerable v2",
            }
            sq = (hrs * 3600.0) / nq
            plan.append({
                "phase": "v3_pilot" if args.pilot_only else "v3_full",
                "model": m,
                "condition": cond,
                "description": f"{cond}: {desc_map[cond]}",
                "n_questions": nq,
                "rate_sq": round(sq, 2),
                "gpu_hours": round(hrs, 3),
            })

    total_needed_hrs = sum(item["gpu_hours"] for item in plan)
    remaining_hrs = args.hours - total_needed_hrs
    fits = remaining_hrs >= 0.0

    if args.json:
        out = {
            "available_hours": args.hours,
            "total_needed_hours": round(total_needed_hrs, 2),
            "remaining_hours": round(remaining_hrs, 2),
            "fits": fits,
            "tasks": plan,
        }
        print(json.dumps(out, indent=2))
        return

    # Print human-readable report
    print("=" * 84)
    mode_str = "VALIDATION PILOT (N=50/dataset)" if args.pilot_only else "FULL EVALUATION"
    print(f"MEDRAG-SLM GPU QUOTA BUDGET ESTIMATE ({mode_str})")
    print(f"Available Quota: {args.hours:.1f} hours | Estimated Required: {total_needed_hrs:.2f} hours")
    print("=" * 84)
    print(f"{'Phase':<12} {'Model':<24} {'Task / Condition':<26} {'N':<5} {'s/q':<6} {'Hours':<6}")
    print("-" * 84)

    curr_phase = ""
    subtotal_hrs = 0.0
    for item in plan:
        phase = item["phase"]
        if phase != curr_phase:
            if curr_phase:
                print(f"  --> Subtotal {curr_phase:<20}: {subtotal_hrs:.2f} hours\n")
            curr_phase = phase
            subtotal_hrs = 0.0
        hrs = item["gpu_hours"]
        subtotal_hrs += hrs
        cond_desc = item.get("description", item.get("condition", ""))
        print(f"{phase:<12} {item['model']:<24} {cond_desc:<26} {item['n_questions']:<5} {item['rate_sq']:<6.1f} {hrs:<6.2f}")

    if curr_phase:
        print(f"  --> Subtotal {curr_phase:<20}: {subtotal_hrs:.2f} hours\n")

    print("=" * 84)
    print(f"TOTAL ESTIMATED GPU TIME : {total_needed_hrs:.2f} hours")
    print(f"REMAINING AFTER EXECUTION: {remaining_hrs:.2f} hours")
    if fits:
        print(f"STATUS                   : [OK] Fits within {args.hours:.1f}h quota! Margin: +{remaining_hrs:.2f}h")
    else:
        print(f"STATUS                   : [DEFICIT] Exceeds quota by {-remaining_hrs:.2f} hours!")
        print("  Recommendation: Execute pilots first (0.4h) or defer unanswerable v2 full runs.")
    print("=" * 84)


if __name__ == "__main__":
    main()
