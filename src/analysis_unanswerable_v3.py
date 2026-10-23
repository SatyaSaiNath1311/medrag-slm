"""src/analysis_unanswerable_v3.py — Unanswerable v2 Benchmark Evaluation.

Implements all pre-registered analyses from docs/DESIGN_GENERIC_VS_MEDICAL.md (Task 4):
  1. Main comparison restricted to exactly three matched pairs:
       Pair A: qwen3-1.7b vs medpsy-1.7b
       Pair B: qwen3-4b vs medpsy-4b
       Pair C: gemma3-4b vs medgemma-1.5-4b-it
  2. Separate thinking-mode control table for qwen3-4b-thinking-2507, labeled budget-limited
     (99.6% truncation at the 1,024-token cap).
  3. Separate supplementary generic reference table for phi4-mini and smollm3-3b.
  4. Confident-wrong-answer (CWA) rate on unanswerable questions (overall and by category).
  5. False-abstention rate (FAR) on control questions.
  6. Recall on unanswerable questions at fixed 5% false-abstention rate using the
     validation-fitted conformal calibration procedure from src/reliability_score.py
     (STRICT PROTOCOL: calibrated on validation only, evaluated frozen on test without refitting).
  7. Outputs formatted markdown and CSV tables to docs/tables/.
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
        # Fallback to conservative threshold (never abstain)
        return {"threshold": 1.0, "val_far": 0.0, "val_recall": 0.0}


def evaluate_unanswerable_dataset(
    model_name: str,
    mode: str,
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    target_far: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate unanswerable_v2 metrics using validation-calibrated conformal threshold.
    
    Metrics:
      1. CWA Rate (conf >= 0.90 and conf >= 0.80) on unanswerable test questions.
      2. Test False Abstention Rate (FAR) on control test questions.
      3. Test Conformal Recall @ target FAR on unanswerable test questions.
      4. Category-level breakdowns.
    """
    # 1. Calibrate threshold on validation split ONLY
    calib = calibrate_conformal_threshold_v3(val_rows, target_far=target_far)
    tau = calib["threshold"]

    # 2. Partition test split
    unans_test = [r for r in test_rows if r.get("should_abstain")]
    ctrl_test = [r for r in test_rows if not r.get("should_abstain")]

    n_unans = len(unans_test)
    n_ctrl = len(ctrl_test)

    # 3. Confident-Wrong-Answer (CWA) rate on unanswerables
    cwa_90_count = sum(1 for r in unans_test if extract_confidence(r) >= 0.90)
    cwa_80_count = sum(1 for r in unans_test if extract_confidence(r) >= 0.80)

    cwa_90_rate = (cwa_90_count / n_unans) if n_unans > 0 else 0.0
    cwa_80_rate = (cwa_80_count / n_unans) if n_unans > 0 else 0.0
    cwa_90_ci = wilson_ci(cwa_90_count, n_unans)
    cwa_80_ci = wilson_ci(cwa_80_count, n_unans)

    # 4. False-Abstention-Rate on control questions
    ctrl_abst_count = sum(1 for r in ctrl_test if (1.0 - extract_confidence(r)) >= tau)
    test_far = (ctrl_abst_count / n_ctrl) if n_ctrl > 0 else 0.0
    test_far_ci = wilson_ci(ctrl_abst_count, n_ctrl)

    # Control questions accuracy
    ctrl_strict = [1 if is_strict_correct(r) else 0 for r in ctrl_test]
    ctrl_fb = [1 if is_fallback_correct(r) else 0 for r in ctrl_test]
    ctrl_strict_acc = (sum(ctrl_strict) / len(ctrl_strict)) if ctrl_strict else None
    ctrl_fb_acc = (sum(ctrl_fb) / len(ctrl_fb)) if ctrl_fb else None

    # 5. Conformal Recall on unanswerable test questions
    unans_abst_count = sum(1 for r in unans_test if (1.0 - extract_confidence(r)) >= tau)
    test_recall = (unans_abst_count / n_unans) if n_unans > 0 else 0.0
    test_recall_ci = wilson_ci(unans_abst_count, n_unans)

    # 6. Category-level breakdowns
    category_metrics = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in unans_test if r.get("category") == cat]
        n_cat = len(cat_rows)
        if n_cat == 0:
            category_metrics[cat] = {
                "n": 0, "cwa_90_count": 0, "cwa_90_rate": 0.0, "cwa_90_ci": (0.0, 0.0),
                "conformal_abst_count": 0, "conformal_recall": 0.0, "conformal_recall_ci": (0.0, 0.0)
            }
            continue
        cwa_90_c = sum(1 for r in cat_rows if extract_confidence(r) >= 0.90)
        abst_c = sum(1 for r in cat_rows if (1.0 - extract_confidence(r)) >= tau)
        category_metrics[cat] = {
            "n": n_cat,
            "cwa_90_count": cwa_90_c,
            "cwa_90_rate": round(cwa_90_c / n_cat, 4),
            "cwa_90_ci": wilson_ci(cwa_90_c, n_cat),
            "conformal_abst_count": abst_c,
            "conformal_recall": round(abst_c / n_cat, 4),
            "conformal_recall_ci": wilson_ci(abst_c, n_cat),
        }

    return {
        "model": model_name,
        "mode": mode,
        "calibrated_tau": round(tau, 4),
        "val_far": round(calib["val_far"], 4),
        "val_recall": round(calib["val_recall"], 4),
        "n_unanswerable": n_unans,
        "n_control": n_ctrl,
        "ctrl_strict_acc": round(ctrl_strict_acc, 4) if ctrl_strict_acc is not None else None,
        "ctrl_fb_acc": round(ctrl_fb_acc, 4) if ctrl_fb_acc is not None else None,
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

    # Handle model aliases like medgemma-4b <-> medgemma-1.5-4b-it
    if model_name in ("medgemma-4b", "medgemma-1.5-4b-it"):
        alt = "medgemma-1.5-4b-it" if model_name == "medgemma-4b" else "medgemma-4b"
        baseline_candidates.extend([
            os.path.join(root_dir, "outputs", "v3", alt, "C4", f"{alt}.jsonl"),
            os.path.join(root_dir, "outputs", "v3", alt, "C4.jsonl"),
        ])
        rag_candidates.extend([
            os.path.join(root_dir, "outputs", "v3", alt, "C5", f"{alt}.jsonl"),
            os.path.join(root_dir, "outputs", "v3", alt, "C5.jsonl"),
        ])
    
    b_path = next((p for p in baseline_candidates if os.path.exists(p) and os.path.getsize(p) > 0), None)
    r_path = next((p for p in rag_candidates if os.path.exists(p) and os.path.getsize(p) > 0), None)
    return b_path, r_path


def write_table_markdown_and_csv(table_name: str, headers: List[str], rows: List[List[Any]], out_dir: str = "docs/tables"):
    """Write tabular data to both Markdown and CSV files."""
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{table_name}.md")
    csv_path = os.path.join(out_dir, f"{table_name}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)

    col_widths = [len(h) for h in headers]
    str_rows = []
    for r in rows:
        row_strs = [str(val) if val is not None else "-" for val in r]
        str_rows.append(row_strs)
        for i, val_s in enumerate(row_strs):
            col_widths[i] = max(col_widths[i], len(val_s))

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |\n")
        fh.write("| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |\n")
        for r_s in str_rows:
            fh.write("| " + " | ".join(r_s[i].ljust(col_widths[i]) for i in range(len(headers))) + " |\n")

    print(f"Saved: {md_path} and {csv_path}")


def _evaluate_model_unans(m: str, label: str, root_dir: str, ref_meta: Dict[str, Any]):
    """Evaluate baseline and RAG unanswerable outputs for a model."""
    b_path, r_path = find_model_unans_v2_files(m, root_dir)
    modes_to_run = []
    if b_path:
        modes_to_run.append(("Baseline (C4)", b_path))
    if r_path:
        modes_to_run.append(("RAG (C5)", r_path))

    summary_entries = []
    cat_entries = []

    if not modes_to_run:
        summary_entries.append([label, m, "Baseline/RAG", "-", "-", "-", "-", "-", "-", "-", "<=5.0%", "-", "PENDING"])
        return summary_entries, cat_entries

    for mode_lbl, path in modes_to_run:
        rows = read_jsonl(path)
        val_rows = [r for r in rows if r.get("split") == "validation"]
        test_rows = [r for r in rows if r.get("split") == "test"]

        for r in val_rows + test_rows:
            meta = ref_meta.get(r.get("id"), {})
            if "category" not in r or r["category"] is None:
                r["category"] = meta.get("category")
            if "should_abstain" not in r or r["should_abstain"] is None:
                r["should_abstain"] = meta.get("should_abstain", True)
            if "control" not in r or r["control"] is None:
                r["control"] = meta.get("control", False)

        eval_res = evaluate_unanswerable_dataset(m, mode_lbl, val_rows, test_rows, target_far=0.05)

        cwa_90_str = f"{eval_res['cwa_90_rate']*100:.1f}% [{eval_res['cwa_90_ci'][0]*100:.1f}%, {eval_res['cwa_90_ci'][1]*100:.1f}%]"
        cwa_80_str = f"{eval_res['cwa_80_rate']*100:.1f}% [{eval_res['cwa_80_ci'][0]*100:.1f}%, {eval_res['cwa_80_ci'][1]*100:.1f}%]"
        far_str = f"{eval_res['test_far']*100:.1f}% [{eval_res['test_far_ci'][0]*100:.1f}%, {eval_res['test_far_ci'][1]*100:.1f}%]"
        rec_str = f"{eval_res['test_conformal_recall']*100:.1f}% [{eval_res['test_conformal_recall_ci'][0]*100:.1f}%, {eval_res['test_conformal_recall_ci'][1]*100:.1f}%]"
        ctrl_strict_str = f"{eval_res['ctrl_strict_acc']*100:.1f}%" if eval_res["ctrl_strict_acc"] is not None else "-"
        ctrl_fb_str = f"{eval_res['ctrl_fb_acc']*100:.1f}%" if eval_res["ctrl_fb_acc"] is not None else "-"

        summary_entries.append([
            label,
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
            cat_entries.append([
                label,
                m,
                mode_lbl,
                cat,
                cat_data["n"],
                f"{cat_data['cwa_90_rate']*100:.1f}% ({cat_data['cwa_90_count']}/{cat_data['n']})",
                f"{cat_data['conformal_recall']*100:.1f}% ({cat_data['conformal_abst_count']}/{cat_data['n']})",
            ])

    return summary_entries, cat_entries


def run_unanswerable_analysis(root_dir: str = ".", target_models: Optional[List[str]] = None):
    """Run unanswerable_v2 evaluation across matched pairs, thinking control, and supplementary references."""
    print("\n" + "=" * 80)
    print("UNANSWERABLE V2 BENCHMARK EVALUATION (v3 Matched-Pair Protocol)")
    print("Main comparison: Pair A (1.7B), Pair B (4B Qwen), Pair C (4B Gemma)")
    print("Thinking control: Qwen3-4B-Thinking-2507 (budget-limited control table)")
    print("Supplementary generic references: Phi-4-mini, SmolLM3-3B (separate reference table)")
    print("=" * 80 + "\n")

    # Load canonical reference metadata
    ref_meta = {}
    for ref_p in ["data/unanswerable_v2/test.jsonl", "data/unanswerable_v2/val.jsonl"]:
        p_full = os.path.join(root_dir, ref_p)
        if os.path.exists(p_full):
            for ref_r in read_jsonl(p_full):
                ref_meta[ref_r["id"]] = ref_r

    out_dir = os.path.join(root_dir, "docs", "tables")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Main Matched Pairs (Pair A, Pair B, Pair C)
    matched_specs = [
        ("qwen3-1.7b", "Pair A (Generic)"),
        ("medpsy-1.7b", "Pair A (Medical)"),
        ("qwen3-4b", "Pair B (Generic)"),
        ("medpsy-4b", "Pair B (Medical)"),
        ("gemma3-4b", "Pair C (Generic)"),
        ("medgemma-1.5-4b-it", "Pair C (Medical)"),
    ]
    if target_models is not None:
        matched_specs = [s for s in matched_specs if s[0] in target_models]

    main_headers = [
        "Matched Pair / Arm", "Model", "Mode", "N Unans", "CWA (conf>=0.90) %", "CWA (conf>=0.80) %",
        "N Ctrl", "Control Strict Acc %", "Control With-Fallback Acc %",
        "Test FAR (tau_val)", "Target FAR", "Test Recall @ 5% FAR", "Calibrated Tau (val)"
    ]
    cat_headers = [
        "Matched Pair / Arm", "Model", "Mode", "Category", "N Unans", "CWA (conf>=0.90) %", "Recall @ 5% FAR %"
    ]

    main_rows = []
    cat_rows = []
    for m, lbl in matched_specs:
        s_entries, c_entries = _evaluate_model_unans(m, lbl, root_dir, ref_meta)
        main_rows.extend(s_entries)
        cat_rows.extend(c_entries)

    write_table_markdown_and_csv("table_unanswerable_v3_summary", main_headers, main_rows, out_dir=out_dir)
    if cat_rows:
        write_table_markdown_and_csv("table_unanswerable_v3_categories", cat_headers, cat_rows, out_dir=out_dir)

    # 2. Thinking-Mode Control (Separate Table)
    control_specs = [
        ("qwen3-4b-thinking-2507", "Pair B Thinking Control"),
    ]
    if target_models is not None:
        control_specs = [s for s in control_specs if s[0] in target_models]

    if control_specs:
        ctrl_headers = [
            "Role", "Model", "Mode", "N Unans", "CWA (conf>=0.90) %", "CWA (conf>=0.80) %",
            "N Ctrl", "Control Strict Acc %", "Control With-Fallback Acc %",
            "Test FAR (tau_val)", "Target FAR", "Test Recall @ 5% FAR", "Calibrated Tau (val)", "Evaluation Note"
        ]
        ctrl_rows = []
        for m, lbl in control_specs:
            s_entries, _ = _evaluate_model_unans(m, lbl, root_dir, ref_meta)
            for r in s_entries:
                ctrl_rows.append(r + ["Budget-limited: 99.6% truncation at 1,024-token cap (abstention influenced by budget cap, not calibrated refusal)"])

        write_table_markdown_and_csv("table_unanswerable_v3_control", ctrl_headers, ctrl_rows, out_dir=out_dir)

    # 3. Supplementary Generic References (Separate Table)
    supp_specs = [
        ("phi4-mini", "Supplementary Generic Reference"),
        ("smollm3-3b", "Supplementary Generic Reference"),
    ]
    if target_models is not None:
        supp_specs = [s for s in supp_specs if s[0] in target_models]

    if supp_specs:
        supp_headers = [
            "Reference Role", "Model", "Mode", "N Unans", "CWA (conf>=0.90) %", "CWA (conf>=0.80) %",
            "N Ctrl", "Control Strict Acc %", "Control With-Fallback Acc %",
            "Test FAR (tau_val)", "Target FAR", "Test Recall @ 5% FAR", "Calibrated Tau (val)"
        ]
        supp_rows = []
        supp_cat_rows = []
        for m, lbl in supp_specs:
            s_entries, c_entries = _evaluate_model_unans(m, lbl, root_dir, ref_meta)
            supp_rows.extend(s_entries)
            supp_cat_rows.extend(c_entries)

        write_table_markdown_and_csv("table_unanswerable_v3_supplementary", supp_headers, supp_rows, out_dir=out_dir)
        if supp_cat_rows:
            write_table_markdown_and_csv("table_unanswerable_v3_supplementary_categories", cat_headers, supp_cat_rows, out_dir=out_dir)

    print("\nUnanswerable v3 analysis complete. All tables written to docs/tables/.\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate unanswerable_v2 benchmark metrics (v3)")
    parser.add_argument("--root", default=".", help="Root directory")
    parser.add_argument("--models", default=None, help="Optional comma-separated model names (defaults to all matched pairs + controls)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    run_unanswerable_analysis(root_dir=args.root, target_models=models)


if __name__ == "__main__":
    main()
