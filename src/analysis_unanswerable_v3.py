"""src/analysis_unanswerable_v3.py — Unanswerable v2 Benchmark Evaluation.

Implements all pre-registered analyses from docs/DESIGN_GENERIC_VS_MEDICAL.md (Task 4):
  1. Confident-wrong-answer rate on unanswerable questions (overall and by category).
  2. False-abstention rate on control questions.
  3. Recall on unanswerable questions at fixed 5% false-abstention rate using the
     validation-fitted conformal calibration procedure from src/reliability_score.py
     (STRICT PROTOCOL: calibrated on validation only, evaluated frozen on test without refitting).
  4. Working example execution on existing qwen3-1.7b v2 outputs, with extensible support
     for all v3 models (MedPsy, Qwen, Gemma).
  5. Outputs formatted markdown and CSV tables to docs/tables/.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.answer_validity import (
    is_strict_correct,
    is_fallback_correct,
    is_truncated,
    is_parsed,
    strict_accuracy,
    fallback_accuracy,
    truncation_rate,
)
from src.common import read_jsonl


CATEGORIES = [
    "ambiguous",
    "fabricated_disease",
    "fabricated_drug",
    "false_premise",
    "missing_info",
    "out_of_scope",
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = p + (z * z) / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    lo = max(0.0, (center - half) / denom)
    hi = min(1.0, (center + half) / denom)
    return (round(lo, 4), round(hi, 4))


def extract_confidence(row: Dict[str, Any]) -> float:
    """Extract prediction confidence with fallbacks."""
    if "confidence" in row and row["confidence"] is not None:
        try:
            return float(row["confidence"])
        except (ValueError, TypeError):
            pass
    lp = row.get("letter_probs", {})
    if lp and isinstance(lp, dict):
        try:
            return float(max(lp.values()))
        except Exception:
            pass
    return 0.5


def calibrate_conformal_threshold_v3(
    val_rows: List[Dict[str, Any]],
    target_far: float = 0.05,
) -> Dict[str, Any]:
    """Find the conformal threshold on non-conformity score S = (1 - confidence)
    that controls False Abstention Rate <= target_far on validation control/answerable questions.
    
    STRICT PROTOCOL: Fitted ONLY on validation rows. Never touches test data.
    """
    ans_val = [r for r in val_rows if not r.get("should_abstain")]
    una_val = [r for r in val_rows if r.get("should_abstain")]
    
    if not ans_val:
        return {"threshold": 0.5, "val_far": 0.0, "val_recall": 0.0}
        
    # Non-conformity score: higher score S means model is less confident -> higher risk
    ans_scores = np.array([1.0 - extract_confidence(r) for r in ans_val])
    una_scores = np.array([1.0 - extract_confidence(r) for r in una_val])
    
    all_scores = np.sort(np.unique(np.concatenate([ans_scores, una_scores])))
    
    # Candidate thresholds
    valid_candidates = []
    for t in all_scores:
        pred_abst_ans = (ans_scores >= t)
        far = float(np.mean(pred_abst_ans))
        if far <= target_far:
            pred_abst_una = (una_scores >= t)
            recall = float(np.mean(pred_abst_una)) if len(una_scores) > 0 else 0.0
            valid_candidates.append({
                "threshold": float(t),
                "val_far": far,
                "val_recall": recall,
            })
            
    if valid_candidates:
        # Choose the threshold that maximizes recall while respecting FAR <= target_far
        best = max(valid_candidates, key=lambda x: (x["val_recall"], -x["val_far"]))
        return best
    else:
        # Fallback to strictest threshold (highest score)
        return {"threshold": float(all_scores[-1]) if len(all_scores) else 0.5, "val_far": 0.0, "val_recall": 0.0}


def evaluate_unanswerable_dataset(
    model_name: str,
    mode_name: str,
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    target_far: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate confident wrong answers, false abstention, and conformal recall at 5% FAR."""
    # 1. Validation Calibration
    calib = calibrate_conformal_threshold_v3(val_rows, target_far=target_far)
    tau = calib["threshold"]

    # 2. Test Partition
    test_ans = [r for r in test_rows if not r.get("should_abstain")]  # controls (N=40)
    test_una = [r for r in test_rows if r.get("should_abstain")]      # challenge unanswerables (N=100)

    # 3. Confident-Wrong-Answer Rate on Unanswerable Test Questions
    # Any prediction on unanswerable test questions with confidence >= 0.90
    cwa_90_count = sum(1 for r in test_una if extract_confidence(r) >= 0.90)
    cwa_80_count = sum(1 for r in test_una if extract_confidence(r) >= 0.80)
    n_una = len(test_una)
    
    cwa_90_rate = (cwa_90_count / n_una) if n_una else 0.0
    cwa_80_rate = (cwa_80_count / n_una) if n_una else 0.0
    cwa_90_ci = wilson_ci(cwa_90_count, n_una)
    cwa_80_ci = wilson_ci(cwa_80_count, n_una)

    # Breakdown by category
    category_metrics = {}
    for cat in CATEGORIES:
        cat_una = [r for r in test_una if r.get("category") == cat]
        cat_n = len(cat_una)
        if cat_n > 0:
            cat_cwa_90 = sum(1 for r in cat_una if extract_confidence(r) >= 0.90)
            cat_abst = sum(1 for r in cat_una if (1.0 - extract_confidence(r)) >= tau)
            category_metrics[cat] = {
                "n": cat_n,
                "cwa_90_count": cat_cwa_90,
                "cwa_90_rate": round(cat_cwa_90 / cat_n, 4),
                "conformal_abst_count": cat_abst,
                "conformal_recall": round(cat_abst / cat_n, 4),
            }

    # 4. Control Questions Accuracy (Strict vs With-Fallback)
    ctrl_strict_acc = strict_accuracy(test_ans)
    ctrl_fb_acc = fallback_accuracy(test_ans)
    ctrl_trunc_rate = truncation_rate(test_ans)

    # 5. False-Abstention Rate on Test Controls at Conformal Threshold
    n_ctrl = len(test_ans)
    ctrl_abst_count = sum(1 for r in test_ans if (1.0 - extract_confidence(r)) >= tau)
    test_far = (ctrl_abst_count / n_ctrl) if n_ctrl else 0.0
    test_far_ci = wilson_ci(ctrl_abst_count, n_ctrl)

    # 6. Recall at Conformal Operating Point on Unanswerables
    una_abst_count = sum(1 for r in test_una if (1.0 - extract_confidence(r)) >= tau)
    test_recall = (una_abst_count / n_una) if n_una else 0.0
    test_recall_ci = wilson_ci(una_abst_count, n_una)

    return {
        "model": model_name,
        "mode": mode_name,
        "calibrated_tau": round(tau, 4),
        "val_far": round(calib["val_far"], 4),
        "val_recall": round(calib["val_recall"], 4),
        "n_unanswerable": n_una,
        "n_control": n_ctrl,
        "ctrl_strict_acc": ctrl_strict_acc,
        "ctrl_fb_acc": ctrl_fb_acc,
        "ctrl_trunc_rate": ctrl_trunc_rate,
        "cwa_90_count": cwa_90_count,
        "cwa_90_rate": round(cwa_90_rate, 4),
        "cwa_90_ci": cwa_90_ci,
        "cwa_80_count": cwa_80_count,
        "cwa_80_rate": round(cwa_80_rate, 4),
        "cwa_80_ci": cwa_80_ci,
        "test_false_abst_count": ctrl_abst_count,
        "test_far": round(test_far, 4),
        "test_far_ci": test_far_ci,
        "test_conformal_recall": round(test_recall, 4),
        "test_conformal_recall_ci": test_recall_ci,
        "categories": category_metrics,
    }


def find_model_unans_v2_files(model_name: str, root_dir: str = ".") -> Tuple[Optional[str], Optional[str]]:
    """Locate baseline and RAG unanswerable_v2 files for a given model."""
    baseline_candidates = [
        os.path.join(root_dir, "outputs", "v3", model_name, "C4", f"{model_name}.jsonl"),
        os.path.join(root_dir, "outputs", "v3", model_name, "C4.jsonl"),
        os.path.join(root_dir, "outputs", "kaggle_qa", "unans_v2", model_name, f"baseline_{model_name}.jsonl"),
        os.path.join(root_dir, "outputs", "kaggle_qa", "unans_v2", model_name, f"{model_name}.jsonl"),
    ]
    rag_candidates = [
        os.path.join(root_dir, "outputs", "v3", model_name, "C5", f"{model_name}.jsonl"),
        os.path.join(root_dir, "outputs", "v3", model_name, "C5.jsonl"),
        os.path.join(root_dir, "outputs", "kaggle_qa", "unans_v2", model_name, f"rag_{model_name}.jsonl"),
    ]
    
    b_path = next((p for p in baseline_candidates if os.path.exists(p)), None)
    r_path = next((p for p in rag_candidates if os.path.exists(p)), None)
    return b_path, r_path


def run_unanswerable_analysis(root_dir: str = ".", target_models: Optional[List[str]] = None):
    """Run analysis across available models and output tables."""
    if target_models is None:
        target_models = [
            "qwen3-1.7b",
            "medpsy-1.7b",
            "qwen3-1.7b-thinking",
            "qwen3-4b",
            "medpsy-4b",
            "qwen3-4b-thinking-2507",
            "gemma3-4b",
            "medgemma-1.5-4b-it",
            "phi4-mini",
            "smollm3-3b",
        ]

    print("\n" + "=" * 80)
    print("UNANSWERABLE V2 BENCHMARK EVALUATION (v3 Matched-Pair Protocol)")
    print("=" * 80)

    summary_headers = [
        "Model", "Mode", "N Unans", "CWA (conf>=0.90) %", "CWA (conf>=0.80) %",
        "N Ctrl", "Control Strict Acc %", "Control With-Fallback Acc %",
        "Test FAR (tau_val)", "Target FAR", "Test Recall @ 5% FAR", "Calibrated Tau (val)"
    ]
    summary_rows = []

    cat_headers = [
        "Model", "Mode", "Category", "N Unans", "CWA (conf>=0.90) %", "Recall @ 5% FAR %"
    ]
    cat_rows = []

    evaluated_any = False

    for m in target_models:
        b_path, r_path = find_model_unans_v2_files(m, root_dir)
        modes_to_run = []
        if b_path:
            modes_to_run.append(("Baseline (C4)", b_path))
        if r_path:
            modes_to_run.append(("RAG (C5)", r_path))

        if not modes_to_run:
            summary_rows.append([m, "Baseline/RAG", "-", "-", "-", "-", "-", "-", "-", "<=5.0%", "-", "PENDING"])
            continue

        for mode_lbl, path in modes_to_run:
            rows = read_jsonl(path)
            val_rows = [r for r in rows if r.get("split") == "validation"]
            test_rows = [r for r in rows if r.get("split") == "test"]

            # Attach category and control metadata from canonical datasets if missing
            ref_meta = {}
            for ref_p in ["data/unanswerable_v2/test.jsonl", "data/unanswerable_v2/val.jsonl"]:
                if os.path.exists(ref_p):
                    for ref_r in read_jsonl(ref_p):
                        ref_meta[ref_r["id"]] = ref_r

            for r in val_rows + test_rows:
                meta = ref_meta.get(r.get("id"), {})
                if "category" not in r or r["category"] is None:
                    r["category"] = meta.get("category")
                if "should_abstain" not in r or r["should_abstain"] is None:
                    r["should_abstain"] = meta.get("should_abstain", True)
                if "control" not in r or r["control"] is None:
                    r["control"] = meta.get("control", False)

            evaluated_any = True
            eval_res = evaluate_unanswerable_dataset(m, mode_lbl, val_rows, test_rows, target_far=0.05)


            cwa_90_str = f"{eval_res['cwa_90_rate']*100:.1f}% [{eval_res['cwa_90_ci'][0]*100:.1f}%, {eval_res['cwa_90_ci'][1]*100:.1f}%]"
            cwa_80_str = f"{eval_res['cwa_80_rate']*100:.1f}% [{eval_res['cwa_80_ci'][0]*100:.1f}%, {eval_res['cwa_80_ci'][1]*100:.1f}%]"
            far_str = f"{eval_res['test_far']*100:.1f}% [{eval_res['test_far_ci'][0]*100:.1f}%, {eval_res['test_far_ci'][1]*100:.1f}%]"
            rec_str = f"{eval_res['test_conformal_recall']*100:.1f}% [{eval_res['test_conformal_recall_ci'][0]*100:.1f}%, {eval_res['test_conformal_recall_ci'][1]*100:.1f}%]"
            ctrl_strict_str = f"{eval_res['ctrl_strict_acc']*100:.1f}%" if eval_res["ctrl_strict_acc"] is not None else "-"
            ctrl_fb_str = f"{eval_res['ctrl_fb_acc']*100:.1f}%" if eval_res["ctrl_fb_acc"] is not None else "-"

            summary_rows.append([
                m,
                mode_lbl,
                eval_res["n_unanswerable"],
                cwa_90_str,
                cwa_80_str,
                eval_res["n_control"],
                ctrl_strict_str,
                ctrl_fb_str,
                far_str,
                "<=5.0%",
                rec_str,
                f"tau={eval_res['calibrated_tau']} (val FAR={eval_res['val_far']*100:.1f}%)",
            ])

            for cat, cat_data in eval_res["categories"].items():
                cat_rows.append([
                    m,
                    mode_lbl,
                    cat,
                    cat_data["n"],
                    f"{cat_data['cwa_90_rate']*100:.1f}% ({cat_data['cwa_90_count']}/{cat_data['n']})",
                    f"{cat_data['conformal_recall']*100:.1f}% ({cat_data['conformal_abst_count']}/{cat_data['n']})",
                ])

    # Save tables
    out_dir = os.path.join(root_dir, "docs", "tables")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Main summary table
    md_summary = os.path.join(out_dir, "table_unanswerable_v3_summary.md")
    csv_summary = os.path.join(out_dir, "table_unanswerable_v3_summary.csv")
    with open(csv_summary, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(summary_headers)
        writer.writerows(summary_rows)

    with open(md_summary, "w", encoding="utf-8") as fh:
        col_w = [len(h) for h in summary_headers]
        for r in summary_rows:
            for i, val in enumerate(r):
                col_w[i] = max(col_w[i], len(str(val)))
        fh.write("| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(summary_headers)) + " |\n")
        fh.write("| " + " | ".join("-" * col_w[i] for i in range(len(summary_headers))) + " |\n")
        for r in summary_rows:
            fh.write("| " + " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(summary_headers))) + " |\n")
    print(f"\nSaved summary table: {md_summary} and {csv_summary}")

    # 2. Category breakdown table
    if cat_rows:
        md_cat = os.path.join(out_dir, "table_unanswerable_v3_categories.md")
        csv_cat = os.path.join(out_dir, "table_unanswerable_v3_categories.csv")
        with open(csv_cat, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cat_headers)
            writer.writerows(cat_rows)
        with open(md_cat, "w", encoding="utf-8") as fh:
            col_w = [len(h) for h in cat_headers]
            for r in cat_rows:
                for i, val in enumerate(r):
                    col_w[i] = max(col_w[i], len(str(val)))
            fh.write("| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(cat_headers)) + " |\n")
            fh.write("| " + " | ".join("-" * col_w[i] for i in range(len(cat_headers))) + " |\n")
            for r in cat_rows:
                fh.write("| " + " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(cat_headers))) + " |\n")
        print(f"Saved category breakdown table: {md_cat} and {csv_cat}")

    print("\nUnanswerable v3 analysis complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate unanswerable_v2 benchmark metrics (v3)")
    parser.add_argument("--root", default=".", help="Root directory")
    parser.add_argument("--models", default="qwen3-1.7b", help="Comma-separated model names")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_unanswerable_analysis(root_dir=args.root, target_models=models)


if __name__ == "__main__":
    main()
