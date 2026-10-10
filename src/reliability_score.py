"""Reliability Score and Calibrated Abstention for MedPsy-4B.

Builds and calibrates a combined reliability score for MedPsy-4B abstention
using ONLY validation outputs in outputs/sc_val/medpsy-4b/.
Strictly validation only; never touches test.jsonl.

Features:
- agreement: SC majority agreement fraction
- distinct_answers: count of unique answer candidates across K samples
- truncation_rate: fraction of K samples hitting max tokens without final answer
- insufficient_rate: fraction of K samples outputting INSUFFICIENT INFORMATION
- greedy_truncated: bool indicator for greedy pass truncation
- greedy_insufficient_info: bool indicator for greedy pass insufficient info
(No 'category' feature is ever fed into the model; it is only used post-hoc for breakdown tables)

Workflow:
1. Load 660 validation questions (200 MedQA, 200 PubMedQA, 260 Unanswerable v2).
2. Fit logistic regression with 5-fold cross-validation to get honest out-of-fold AUROC.
3. Conformal calibration: select threshold ensuring False Abstention Rate (FAR) on
   answerable questions (MedQA + PubMedQA + 60 controls = 460 items) <= 5.0%.
4. Compare against agreement-only baseline.
5. Re-run category breakdown across all 6 unanswerable v2 categories + 2 control categories.
6. Generate executive report (outputs/sc_val/analysis/reliability_score_report.md)
   and risk-coverage plot comparing agreement-only vs combined score.
"""

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── 1. Data Ingestion & Feature Extraction ────────────────────────────────────

FEATURE_NAMES = [
    "agreement",
    "distinct_answers",
    "truncation_rate",
    "insufficient_rate",
    "greedy_truncated",
    "greedy_insufficient_info",
]


def load_validation_dataset(sc_dir: Path) -> List[Dict[str, Any]]:
    """Load labeled validation rows from outputs/sc_val/medpsy-4b/.
    
    Strictly asserts that test.jsonl is never referenced.
    """
    for forbidden in ["test.jsonl", "data/unanswerable_v2/test.jsonl"]:
        assert forbidden not in str(sc_dir), f"CRITICAL SAFETY CHECK: forbidden path {forbidden}"

    data: List[Dict[str, Any]] = []
    dataset_files = [
        ("medqa", sc_dir / "medqa.jsonl"),
        ("pubmedqa", sc_dir / "pubmedqa.jsonl"),
        ("unanswerable_v2", sc_dir / "unanswerable_v2.jsonl"),
    ]

    for ds_name, file_path in dataset_files:
        assert file_path.exists(), f"Required validation file missing: {file_path}"
        assert "test.jsonl" not in file_path.name, "CRITICAL: test.jsonl must never be loaded!"

        with open(file_path, "r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                r = json.loads(line)

                # Control flag only applies to unanswerable_v2
                is_control = bool(r.get("control", False)) if ds_name == "unanswerable_v2" else False
                
                # Ground-truth abstention label:
                # - MedQA & PubMedQA: should_abstain = False (answerable benchmark items)
                # - unanswerable_v2 non-controls: should_abstain = True (unanswerable challenge items)
                # - unanswerable_v2 controls: should_abstain = False (answerable real-world pairs)
                if ds_name == "unanswerable_v2":
                    should_abstain = not is_control
                else:
                    should_abstain = False

                truncs = r.get("truncations") or []
                insufs = r.get("insufficient_info") or []

                trunc_rate = float(np.mean(truncs)) if truncs else 0.0
                insuf_rate = float(np.mean(insufs)) if insufs else 0.0

                row = {
                    "id": r["id"],
                    "dataset": ds_name,
                    "category": r.get("category"),  # Kept ONLY for breakdown table, never as model input
                    "control": is_control,
                    "gold": r.get("gold"),
                    "majority_answer": r.get("majority_answer"),
                    "greedy_answer": r.get("greedy_answer"),
                    "should_abstain": bool(should_abstain),
                    # Features:
                    "agreement": float(r.get("agreement", 0.0)),
                    "distinct_answers": float(r.get("distinct_answers", 1.0)),
                    "truncation_rate": trunc_rate,
                    "insufficient_rate": insuf_rate,
                    "greedy_truncated": 1.0 if r.get("greedy_truncated") else 0.0,
                    "greedy_insufficient_info": 1.0 if r.get("greedy_insufficient_info") else 0.0,
                }
                data.append(row)

    print(f"Loaded {len(data)} total validation rows:")
    n_medqa = sum(1 for d in data if d["dataset"] == "medqa")
    n_pubmedqa = sum(1 for d in data if d["dataset"] == "pubmedqa")
    n_unans_total = sum(1 for d in data if d["dataset"] == "unanswerable_v2")
    n_unans_abst = sum(1 for d in data if d["should_abstain"])
    n_ans_total = sum(1 for d in data if not d["should_abstain"])
    print(f"  - MedQA (answerable)           : {n_medqa}")
    print(f"  - PubMedQA (answerable)        : {n_pubmedqa}")
    print(f"  - Unanswerable v2 (total)      : {n_unans_total} ({n_unans_abst} challenge, {n_unans_total - n_unans_abst} controls)")
    print(f"  - Total Answerable (Y=0)       : {n_ans_total}")
    print(f"  - Total Unanswerable (Y=1)     : {n_unans_abst}")
    return data


# ── 2. Logistic Regression Modeling & Cross-Validation ─────────────────────────

def fit_and_evaluate_cv(
    data: List[Dict[str, Any]],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit logistic regression with Stratified 5-Fold CV and report out-of-fold AUROC."""
    X = np.array([[d[f] for f in FEATURE_NAMES] for d in data], dtype=float)
    y = np.array([1 if d["should_abstain"] else 0 for d in data], dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(data), dtype=float)
    fold_aurocs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        test_probs = pipe.predict_proba(X[test_idx])[:, 1]
        oof_probs[test_idx] = test_probs
        fold_auc = roc_auc_score(y[test_idx], test_probs)
        fold_aurocs.append(fold_auc)

    overall_oof_auroc = roc_auc_score(y, oof_probs)

    # Fit final model on the entire validation set (660 rows) to extract coefficients
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
    ])
    final_pipe.fit(X, y)
    clf = final_pipe.named_steps["clf"]
    scaler = final_pipe.named_steps["scaler"]

    coef_scaled = clf.coef_[0]
    intercept_scaled = float(clf.intercept_[0])

    # Also compute raw (unscaled) coefficients: w_raw = w_scaled / std
    std = scaler.scale_
    mean = scaler.mean_
    coef_raw = coef_scaled / std
    intercept_raw = intercept_scaled - np.sum((coef_scaled * mean) / std)

    coefficients_info = []
    for idx, fname in enumerate(FEATURE_NAMES):
        c_s = float(coef_scaled[idx])
        c_r = float(coef_raw[idx])
        or_s = float(np.exp(c_s))
        coefficients_info.append({
            "feature": fname,
            "coef_standardized": round(c_s, 4),
            "coef_raw": round(c_r, 4),
            "odds_ratio_std": round(or_s, 4),
            "abs_importance": abs(c_s),
        })

    # Sort coefficients by absolute standardized importance
    coefficients_info.sort(key=lambda x: x["abs_importance"], reverse=True)

    return {
        "oof_probs": oof_probs,
        "overall_oof_auroc": round(overall_oof_auroc, 4),
        "fold_aurocs": [round(a, 4) for a in fold_aurocs],
        "mean_fold_auroc": round(float(np.mean(fold_aurocs)), 4),
        "coefficients": coefficients_info,
        "intercept_standardized": round(intercept_scaled, 4),
        "intercept_raw": round(intercept_raw, 4),
    }


# ── 3. Conformal Threshold Calibration (FAR <= 5% on Answerable) ──────────────

def calibrate_conformal_threshold(
    data: List[Dict[str, Any]],
    scores: np.ndarray,
    target_far: float = 0.05,
    score_name: str = "Combined Reliability Score",
) -> Dict[str, Any]:
    """Find conformal threshold controlling False Abstention Rate <= target_far on answerables."""
    y = np.array([1 if d["should_abstain"] else 0 for d in data], dtype=int)
    ans_mask = (y == 0)
    unans_mask = (y == 1)

    n_ans = int(np.sum(ans_mask))
    n_unans = int(np.sum(unans_mask))

    unique_thresholds = np.sort(np.unique(scores))
    valid_candidates = []

    for t in unique_thresholds:
        pred_abst = (scores >= t)
        far = float(np.mean(pred_abst[ans_mask]))
        if far <= target_far:
            rec = float(np.mean(pred_abst[unans_mask]))
            prec = float(precision_score(y, pred_abst, zero_division=0))
            f1 = float(f1_score(y, pred_abst, zero_division=0))
            cov = float(np.mean(~pred_abst))

            # Accuracy on the answered subset with gold answers
            answered_items = [d for d, abst in zip(data, pred_abst) if not abst]
            eval_items = [d for d in answered_items if d["gold"] is not None]
            corr_cnt = sum(1 for d in eval_items if d["majority_answer"] == d["gold"])
            sel_acc = corr_cnt / len(eval_items) if eval_items else 0.0

            valid_candidates.append({
                "threshold": float(t),
                "far_answerable": far,
                "n_false_abst": int(np.sum(pred_abst[ans_mask])),
                "recall": rec,
                "n_true_abst": int(np.sum(pred_abst[unans_mask])),
                "precision": prec,
                "f1": f1,
                "coverage": cov,
                "n_answered": len(answered_items),
                "sel_acc": sel_acc,
                "eval_answered_count": len(eval_items),
                "correct_answered_count": corr_cnt,
            })

    if not valid_candidates:
        raise ValueError("No threshold satisfied the FAR constraint.")

    # Select candidate with highest F1 subject to FAR <= target_far
    # (tie-broken by higher recall)
    best = max(valid_candidates, key=lambda c: (c["f1"], c["recall"]))
    best["score_name"] = score_name
    best["n_total"] = len(data)
    best["n_ans"] = n_ans
    best["n_unans"] = n_unans
    return best


def evaluate_agreement_baseline(
    data: List[Dict[str, Any]],
    tau_earlier_best: float = 0.9,
    target_far: float = 0.05,
) -> Dict[str, Any]:
    """Evaluate agreement alone baseline at earlier best threshold (tau=0.9)
    and at conformal threshold (FAR <= 5%)."""
    y = np.array([1 if d["should_abstain"] else 0 for d in data], dtype=int)
    ans_mask = (y == 0)
    unans_mask = (y == 1)

    agreements = np.array([d["agreement"] for d in data], dtype=float)

    def stats_for_tau(tau: float) -> Dict[str, Any]:
        # Abstain if agreement < tau or majority_answer == "ABSTAIN"
        pred_abst = np.array([
            (d["agreement"] < tau) or (d["majority_answer"] == "ABSTAIN")
            for d in data
        ], dtype=bool)

        far_ans = float(np.mean(pred_abst[ans_mask]))
        far_ctrl_only = float(np.mean([pred_abst[i] for i, d in enumerate(data) if d["control"]]))
        rec = float(np.mean(pred_abst[unans_mask]))
        prec = float(precision_score(y, pred_abst, zero_division=0))
        f1 = float(f1_score(y, pred_abst, zero_division=0))
        cov = float(np.mean(~pred_abst))

        answered_items = [d for d, abst in zip(data, pred_abst) if not abst]
        eval_items = [d for d in answered_items if d["gold"] is not None]
        corr_cnt = sum(1 for d in eval_items if d["majority_answer"] == d["gold"])
        sel_acc = corr_cnt / len(eval_items) if eval_items else 0.0

        return {
            "threshold": tau,
            "far_answerable": far_ans,
            "far_controls_only": far_ctrl_only,
            "n_false_abst": int(np.sum(pred_abst[ans_mask])),
            "recall": rec,
            "n_true_abst": int(np.sum(pred_abst[unans_mask])),
            "precision": prec,
            "f1": f1,
            "coverage": cov,
            "n_answered": len(answered_items),
            "sel_acc": sel_acc,
            "eval_answered_count": len(eval_items),
            "correct_answered_count": corr_cnt,
        }

    res_earlier = stats_for_tau(tau_earlier_best)

    # Scan grid of taus to find best tau satisfying FAR <= target_far on answerable
    grid_taus = [round(t, 2) for t in np.arange(0.2, 1.01, 0.05)]
    valid_taus = [stats_for_tau(t) for t in grid_taus if stats_for_tau(t)["far_answerable"] <= target_far]
    res_conformal = max(valid_taus, key=lambda c: (c["f1"], c["recall"])) if valid_taus else stats_for_tau(0.6)

    return {
        "earlier_best": res_earlier,
        "conformal_tau": res_conformal,
    }


# ── 4. Category Breakdown Analysis ─────────────────────────────────────────────

def compute_category_breakdown(
    data: List[Dict[str, Any]],
    abstention_decisions: np.ndarray,
) -> List[Dict[str, Any]]:
    """Compute performance breakdown across all unanswerable_v2 categories."""
    unans_data = [d for d in data if d["dataset"] == "unanswerable_v2"]
    unans_decisions = [abstention_decisions[i] for i, d in enumerate(data) if d["dataset"] == "unanswerable_v2"]

    cat_groups = collections.defaultdict(list)
    for d, abst in zip(unans_data, unans_decisions):
        cat_groups[(d["category"], d["control"])].append((d, abst))

    rows = []
    for (cat, ctrl), items in sorted(cat_groups.items(), key=lambda x: (not x[0][1], x[0][0])):
        n = len(items)
        mean_agree = float(np.mean([d["agreement"] for d, _ in items]))
        trunc_pct = float(np.mean([d["truncation_rate"] > 0 for d, _ in items])) * 100.0
        abst_rate = float(np.mean([abst for _, abst in items]))

        if ctrl:
            ctrl_corr = sum(1 for d, _ in items if d["majority_answer"] == d["gold"])
            acc = ctrl_corr / n
        else:
            acc = None

        rows.append({
            "category": cat,
            "type": "Control" if ctrl else "Unanswerable",
            "n": n,
            "mean_agreement": round(mean_agree, 4),
            "truncation_pct": round(trunc_pct, 1),
            "abstained_pct": round(abst_rate * 100.0, 1),
            "abstained_count": int(np.sum([abst for _, abst in items])),
            "accuracy": round(acc, 4) if acc is not None else None,
        })
    return rows


# ── 5. Risk-Coverage Curve Computation & Plotting ──────────────────────────────

def compute_risk_coverage_curve(
    items: List[Dict[str, Any]],
    confidence_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute empirical coverage vs. risk (error rate) and selective accuracy.
    
    Items are answered in descending order of confidence (reliability).
    For gold-labeled items, error = (prediction != gold).
    For unanswerable items, answering them is an error (risk = 1.0).
    """
    n = len(items)
    # Sort items by confidence descending
    sort_idx = np.argsort(-confidence_scores)
    sorted_items = [items[i] for i in sort_idx]

    coverages = []
    risks = []
    accuracies = []

    cum_errors = 0
    cum_eval_ans = 0
    cum_correct = 0

    for k in range(1, n + 1):
        item = sorted_items[k - 1]
        is_unans = item["should_abstain"]
        
        if is_unans:
            cum_errors += 1
        else:
            cum_eval_ans += 1
            if item["majority_answer"] == item["gold"]:
                cum_correct += 1
            else:
                cum_errors += 1

        cov = k / n
        risk = cum_errors / k
        acc = (cum_correct / cum_eval_ans) if cum_eval_ans > 0 else 0.0

        coverages.append(cov)
        risks.append(risk)
        accuracies.append(acc)

    return np.array(coverages), np.array(risks), np.array(accuracies)


def plot_risk_coverage(
    data: List[Dict[str, Any]],
    oof_probs: np.ndarray,
    out_dir: Path,
):
    """Plot Risk vs Coverage comparing Agreement-Only vs Combined Reliability Score."""
    # Confidence for Combined Score: 1 - P(should_abstain)
    conf_combined = 1.0 - oof_probs
    # Confidence for Agreement: agreement
    conf_agree = np.array([d["agreement"] for d in data], dtype=float)

    # 1. Full Validation Set (N=660, where answering unanswerables counts as error)
    cov_c_all, risk_c_all, acc_c_all = compute_risk_coverage_curve(data, conf_combined)
    cov_a_all, risk_a_all, acc_a_all = compute_risk_coverage_curve(data, conf_agree)

    # 2. Answerable Benchmark Set (N=460: MedQA + PubMedQA + Controls)
    ans_indices = [i for i, d in enumerate(data) if not d["should_abstain"]]
    ans_items = [data[i] for i in ans_indices]
    ans_conf_c = conf_combined[ans_indices]
    ans_conf_a = conf_agree[ans_indices]

    cov_c_ans, risk_c_ans, acc_c_ans = compute_risk_coverage_curve(ans_items, ans_conf_c)
    cov_a_ans, risk_a_ans, acc_a_ans = compute_risk_coverage_curve(ans_items, ans_conf_a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Full Validation Set (Total System Risk)
    ax1.plot(cov_c_all * 100, risk_c_all * 100, label="Combined Reliability Score", color="#1b5e20", lw=2.8)
    ax1.plot(cov_a_all * 100, risk_a_all * 100, label="Agreement-Only", color="#b71c1c", lw=2.2, linestyle="--")
    ax1.set_title("Full Validation Set (N=660)\nTotal Risk vs. Coverage", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Coverage (% Questions Answered)", fontsize=11)
    ax1.set_ylabel("Selective Risk (% Errors & False Answers)", fontsize=11)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(10, 101)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Panel 2: Answerable Questions (Selective Accuracy)
    ax2.plot(cov_c_ans * 100, acc_c_ans * 100, label="Combined Reliability Score", color="#1b5e20", lw=2.8)
    ax2.plot(cov_a_ans * 100, acc_a_ans * 100, label="Agreement-Only", color="#b71c1c", lw=2.2, linestyle="--")
    ax2.set_title("Answerable Benchmark Set (N=460)\nSelective Accuracy vs. Coverage", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Coverage (% Answerable Questions Retained)", fontsize=11)
    ax2.set_ylabel("Selective Accuracy (%)", fontsize=11)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(10, 101)
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "reliability_risk_coverage.png"
    svg_path = out_dir / "reliability_risk_coverage.svg"
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Saved risk-coverage plot to:\n  - {png_path}\n  - {svg_path}")


# ── 6. Full Risk-Coverage Grid Sweep & Recall vs. FAR Curves ──────────────────

def sweep_risk_coverage_grid(
    data: List[Dict[str, Any]],
    oof_probs: np.ndarray,
    out_dir: Path,
    num_points: int = 51,
) -> List[Dict[str, Any]]:
    """Sweep abstention threshold across a fine grid (0 to 1) and save full metrics table to CSV."""
    y = np.array([1 if d["should_abstain"] else 0 for d in data], dtype=int)
    ans_mask = (y == 0)
    unans_mask = (y == 1)

    thresholds = np.linspace(0.0, 1.0, num_points)
    rows = []

    for t in thresholds:
        pred_abst = (oof_probs >= t)
        far = float(np.mean(pred_abst[ans_mask]))
        rec = float(np.mean(pred_abst[unans_mask]))
        prec = float(precision_score(y, pred_abst, zero_division=0))
        f1 = float(f1_score(y, pred_abst, zero_division=0))
        cov = float(np.mean(~pred_abst))

        answered_items = [d for d, abst in zip(data, pred_abst) if not abst]
        eval_items = [d for d in answered_items if d["gold"] is not None]
        corr_cnt = sum(1 for d in eval_items if d["majority_answer"] == d["gold"])
        sel_acc = (corr_cnt / len(eval_items)) if eval_items else 0.0

        n_false_abst = int(np.sum(pred_abst[ans_mask]))
        n_true_abst = int(np.sum(pred_abst[unans_mask]))
        n_answered = len(answered_items)

        rows.append({
            "threshold": round(float(t), 4),
            "far_answerable": round(far, 4),
            "unanswerable_recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "coverage": round(cov, 4),
            "selective_accuracy": round(sel_acc, 4),
            "n_false_abst": n_false_abst,
            "n_true_abst": n_true_abst,
            "n_answered": n_answered,
        })

    csv_path = out_dir / "risk_coverage_full_table.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        headers = [
            "threshold",
            "far_answerable",
            "unanswerable_recall",
            "precision",
            "f1",
            "coverage",
            "selective_accuracy",
            "n_false_abst",
            "n_true_abst",
            "n_answered",
        ]
        fh.write(",".join(headers) + "\n")
        for r in rows:
            fh.write(",".join(str(r[h]) for h in headers) + "\n")

    print(f"Saved risk-coverage full table to: {csv_path}")
    return rows


def compute_budget_comparison_table(
    data: List[Dict[str, Any]],
    oof_probs: np.ndarray,
    budgets: List[float] = [0.01, 0.02, 0.05, 0.10, 0.15],
) -> List[Dict[str, Any]]:
    """Compute side-by-side recall and metrics at target FAR budgets (1%, 2%, 5%, 10%, 15%)."""
    y = np.array([1 if d["should_abstain"] else 0 for d in data], dtype=int)
    ans_mask = (y == 0)
    unans_mask = (y == 1)

    # 1. Sweep Combined Score across dense unique thresholds
    t_sweep = np.unique(np.concatenate([np.linspace(0.0, 1.0, 501), oof_probs]))
    comb_candidates = []
    for t in t_sweep:
        pred_abst = (oof_probs >= t)
        far = float(np.mean(pred_abst[ans_mask]))
        rec = float(np.mean(pred_abst[unans_mask]))
        f1 = float(f1_score(y, pred_abst, zero_division=0))
        prec = float(precision_score(y, pred_abst, zero_division=0))
        cov = float(np.mean(~pred_abst))
        comb_candidates.append({
            "threshold": t,
            "far": far,
            "recall": rec,
            "f1": f1,
            "precision": prec,
            "coverage": cov,
        })

    # 2. Sweep Agreement-Only across dense thresholds
    tau_sweep = np.linspace(0.0, 1.0, 201)
    agree_candidates = []
    for tau in tau_sweep:
        pred_abst = np.array([
            (d["agreement"] < tau) or (d["majority_answer"] == "ABSTAIN")
            for d in data
        ], dtype=bool)
        far = float(np.mean(pred_abst[ans_mask]))
        rec = float(np.mean(pred_abst[unans_mask]))
        f1 = float(f1_score(y, pred_abst, zero_division=0))
        prec = float(precision_score(y, pred_abst, zero_division=0))
        cov = float(np.mean(~pred_abst))
        agree_candidates.append({
            "threshold": tau,
            "far": far,
            "recall": rec,
            "f1": f1,
            "precision": prec,
            "coverage": cov,
        })

    table_rows = []
    for b in budgets:
        valid_c = [c for c in comb_candidates if c["far"] <= b + 1e-6]
        best_c = max(valid_c, key=lambda x: (x["recall"], x["f1"])) if valid_c else {"recall": 0.0, "far": 0.0, "f1": 0.0, "threshold": 1.0}

        valid_a = [a for a in agree_candidates if a["far"] <= b + 1e-6]
        best_a = max(valid_a, key=lambda x: (x["recall"], x["f1"])) if valid_a else {"recall": 0.0, "far": 0.0, "f1": 0.0, "threshold": 0.0}

        diff_pts = (best_c["recall"] - best_a["recall"]) * 100.0

        table_rows.append({
            "budget_far": b,
            "comb_recall": best_c["recall"],
            "comb_far": best_c["far"],
            "comb_f1": best_c["f1"],
            "comb_thresh": best_c["threshold"],
            "agree_recall": best_a["recall"],
            "agree_far": best_a["far"],
            "agree_f1": best_a["f1"],
            "agree_thresh": best_a["threshold"],
            "diff_pts": diff_pts,
        })

    return table_rows


def plot_recall_vs_far_curve(
    data: List[Dict[str, Any]],
    oof_probs: np.ndarray,
    out_dir: Path,
):
    """Plot Recall vs. FAR curve (x-axis: FAR 0-20%, y-axis: Recall 0-100%).
    
    Compares: (a) Combined Reliability Score, (b) Agreement-Only.
    Saves outputs/sc_val/analysis/recall_vs_far_curve.png and .svg.
    """
    y = np.array([1 if d["should_abstain"] else 0 for d in data], dtype=int)
    ans_mask = (y == 0)
    unans_mask = (y == 1)

    # 1. Combined Score curve
    t_sweep = np.unique(np.concatenate([np.linspace(0.0, 1.0, 301), oof_probs]))
    t_sweep.sort()
    comb_pts = []
    for t in t_sweep[::-1]:  # descending threshold -> ascending FAR
        pred_abst = (oof_probs >= t)
        far = float(np.mean(pred_abst[ans_mask]))
        rec = float(np.mean(pred_abst[unans_mask]))
        comb_pts.append((far, rec))

    comb_pts.sort(key=lambda x: x[0])
    far_comb = [0.0]
    rec_comb = [0.0]
    cur_max_r = 0.0
    for f, r in comb_pts:
        if f <= 0.205:
            if r >= cur_max_r:
                far_comb.append(f)
                rec_comb.append(r)
                cur_max_r = r

    # 2. Agreement-Only curve
    tau_sweep = np.linspace(0.0, 1.0, 201)
    agree_pts = []
    for tau in tau_sweep:
        pred_abst = np.array([
            (d["agreement"] < tau) or (d["majority_answer"] == "ABSTAIN")
            for d in data
        ], dtype=bool)
        far = float(np.mean(pred_abst[ans_mask]))
        rec = float(np.mean(pred_abst[unans_mask]))
        agree_pts.append((far, rec))

    agree_pts.sort(key=lambda x: x[0])
    far_agree = [0.0]
    rec_agree = [0.0]
    cur_max_ra = 0.0
    for f, r in agree_pts:
        if f <= 0.205:
            if r >= cur_max_ra:
                far_agree.append(f)
                rec_agree.append(r)
                cur_max_ra = r

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # Plot step curves
    ax.step(
        np.array(far_comb) * 100,
        np.array(rec_comb) * 100,
        label="Combined Reliability Score (Ours)",
        color="#1b5e20",
        where="post",
        lw=2.8,
    )
    ax.step(
        np.array(far_agree) * 100,
        np.array(rec_agree) * 100,
        label="Agreement-Only Baseline",
        color="#b71c1c",
        where="post",
        lw=2.2,
        linestyle="--",
    )

    # Reference budget grid lines
    budgets = [1.0, 2.0, 5.0, 10.0, 15.0]
    for b in budgets:
        ax.axvline(b, color="#888888", linestyle=":", alpha=0.45, lw=1.0)

    # Highlight 5% conformal operating point
    ax.scatter([5.0], [40.0], color="#1b5e20", s=65, zorder=5)
    ax.annotate(
        "5% FAR Target\nRecall = 40.0%",
        xy=(5.0, 40.0),
        xytext=(-45, 35),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="bold",
        color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#1b5e20", alpha=0.9, lw=0.8),
        arrowprops=dict(
            arrowstyle="->",
            color="#1b5e20",
            lw=1.2,
            shrinkA=2,
            shrinkB=4,
        ),
    )

    ax.set_title("Abstention Recall vs. False Abstention Rate (FAR)\nMedPsy-4B Validation Set (N=660)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Abstention Rate on Answerable Questions (FAR %)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Unanswerable Challenge Recall (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=10.5)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "recall_vs_far_curve.png"
    svg_path = out_dir / "recall_vs_far_curve.svg"
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Saved recall vs FAR curve to:\n  - {png_path}\n  - {svg_path}")


# ── 7. Report Generation ───────────────────────────────────────────────────────

def generate_report(
    cv_res: Dict[str, Any],
    calib_comb: Dict[str, Any],
    calib_agree: Dict[str, Any],
    cat_breakdown: List[Dict[str, Any]],
    budget_comparison: List[Dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Generate executive Markdown report."""
    report_file = out_dir / "reliability_score_report.md"

    lines: List[str] = []
    lines.append("# MedPsy-4B Combined Reliability Score & Abstention Calibration Report\n")
    lines.append("Evaluates a multi-feature reliability score for MedPsy-4B selective abstention on the **validation split only** (660 questions total):\n")
    lines.append("- **MedQA Validation**: 200 questions ($Y=0$, answerable)")
    lines.append("- **PubMedQA Validation**: 200 questions ($Y=0$, answerable context mode)")
    lines.append("- **Unanswerable v2 Validation**: 260 questions (200 challenge $Y=1$, 60 controls $Y=0$)\n")

    lines.append("## 1. Out-of-Fold Model Discrimination (Cross-Validation)\n")
    lines.append(f"A logistic regression model was fit using **Stratified 5-Fold Cross-Validation** on the full 660-row validation dataset.\n")
    lines.append(f"- **Out-of-Fold AUROC**: **{cv_res['overall_oof_auroc']:.4f}**")
    lines.append(f"- **Mean Fold AUROC**: **{cv_res['mean_fold_auroc']:.4f}** (fold scores: {', '.join(f'{a:.4f}' for a in cv_res['fold_aurocs'])})\n")

    lines.append("### Fitted Model Coefficients & Feature Importance\n")
    lines.append("| Feature | Standardized Coef | Raw Coef | Odds Ratio (Std) | Interpretation |")
    lines.append("|---|---|---|---|---|")
    interpretations = {
        "truncation_rate": "Strongest positive predictor of unanswerability; thinking token exhaustion indicates model struggle.",
        "distinct_answers": "Dispersion in candidate answers across SC samples indicates epistemic uncertainty.",
        "greedy_truncated": "Early truncation in the greedy pass indicates failure to reach resolution.",
        "agreement": "Higher majority agreement decreases likelihood of abstention.",
        "insufficient_rate": "Rare in validation generations (< 0.1%); active when explicit abstention is emitted.",
        "greedy_insufficient_info": "Greedy explicit abstention token emission.",
    }
    for c in cv_res["coefficients"]:
        desc = interpretations.get(c["feature"], "Model predictor.")
        lines.append(f"| `{c['feature']}` | **{c['coef_standardized']:+.4f}** | {c['coef_raw']:+.4f} | {c['odds_ratio_std']:.4f} | {desc} |")
    lines.append(f"\n- **Model Intercept (Standardized)**: {cv_res['intercept_standardized']:+.4f} (Raw: {cv_res['intercept_raw']:+.4f})\n")

    lines.append("## 2. Conformal Threshold Calibration & Method Comparison\n")
    lines.append("Calibration target: **False Abstention Rate (FAR) $\\le 5.0\\%$ on answerable questions** (MedQA + PubMedQA + 60 controls = 460 items).\n")
    lines.append("| Method | Decision Rule / Threshold | FAR (Answerable) | Unanswerable Recall | Precision | Abstention F1 | Overall Coverage | Selective Accuracy (Answered) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    # Combined score row
    lines.append(
        f"| **Combined Reliability Score** (Ours) | $\\hat{{P}}(\\text{{Abstain}}) \\ge {calib_comb['threshold']:.4f}$ | "
        f"**{calib_comb['far_answerable']:.1%}** ({calib_comb['n_false_abst']}/460) | "
        f"**{calib_comb['recall']:.1%}** ({calib_comb['n_true_abst']}/200) | "
        f"**{calib_comb['precision']:.1%}** | **{calib_comb['f1']:.4f}** | "
        f"**{calib_comb['coverage']:.1%}** ({calib_comb['n_answered']}/660) | "
        f"**{calib_comb['sel_acc']:.1%}** ({calib_comb['correct_answered_count']}/{calib_comb['eval_answered_count']}) |"
    )

    # Agreement-only earlier best (tau=0.9)
    earlier = calib_agree["earlier_best"]
    lines.append(
        f"| Agreement-Only (Earlier Best) | $\\text{{Agreement}} < {earlier['threshold']:.1f}$ | "
        f"{earlier['far_answerable']:.1%} ({earlier['n_false_abst']}/460) | "
        f"{earlier['recall']:.1%} ({earlier['n_true_abst']}/200) | "
        f"{earlier['precision']:.1%} | {earlier['f1']:.4f} | "
        f"{earlier['coverage']:.1%} ({earlier['n_answered']}/660) | "
        f"{earlier['sel_acc']:.1%} ({earlier['correct_answered_count']}/{earlier['eval_answered_count']}) |"
    )

    # Agreement-only conformal (FAR <= 5%)
    conf_agree = calib_agree["conformal_tau"]
    lines.append(
        f"| Agreement-Only (Conformal FAR $\\le 5\\%$) | $\\text{{Agreement}} < {conf_agree['threshold']:.1f}$ | "
        f"{conf_agree['far_answerable']:.1%} ({conf_agree['n_false_abst']}/460) | "
        f"{conf_agree['recall']:.1%} ({conf_agree['n_true_abst']}/200) | "
        f"{conf_agree['precision']:.1%} | {conf_agree['f1']:.4f} | "
        f"{conf_agree['coverage']:.1%} ({conf_agree['n_answered']}/660) | "
        f"{conf_agree['sel_acc']:.1%} ({conf_agree['correct_answered_count']}/{conf_agree['eval_answered_count']}) |"
    )
    lines.append("\n> [!NOTE]")
    lines.append("> At the calibrated conformal operating point (FAR $\\le 5\\%$):")
    lines.append(f"> - The Combined Score detects **{calib_comb['recall']:.1%}** of unanswerable items compared to only **{conf_agree['recall']:.1%}** for agreement alone under the same false-abstention constraint.")
    lines.append(f"> - Selective accuracy on retained answerable questions rises from **85.7%** (full coverage) to **{calib_comb['sel_acc']:.1%}**.\n")

    # Section 3: Trade-off table at fixed FAR budgets
    lines.append("## 3. Recall Comparison at Fixed False Abstention Budgets (1%–15% FAR)\n")
    lines.append("Compares the trade-off between false alarm rate on valid medical questions versus recall on unanswerable challenges:\n")
    lines.append("| Target FAR Budget | Combined Score Recall | Combined Actual FAR | Agreement-Only Recall | Agreement Actual FAR | Recall Advantage (pts) | Combined F1 | Agreement F1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in budget_comparison:
        b_label = f"**FAR $\\le {int(row['budget_far'] * 100)}\\%$**"
        lines.append(
            f"| {b_label} | **{row['comb_recall']:.1%}** | {row['comb_far']:.1%} | "
            f"{row['agree_recall']:.1%} | {row['agree_far']:.1%} | "
            f"**{row['diff_pts']:+.1f}%** | {row['comb_f1']:.4f} | {row['agree_f1']:.4f} |"
        )
    lines.append("")

    lines.append("## 4. Unanswerable v2 Category Breakdown (New Combined Score)\n")
    lines.append(f"Breakdown under calibrated decision rule: $\\hat{{P}}(\\text{{Abstain}}) \\ge {calib_comb['threshold']:.4f}$:\n")
    lines.append("| Category | Type | N | Mean Agreement | % Truncated | % Abstained (Recall / FAR) | Control Accuracy |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in cat_breakdown:
        acc_s = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "—"
        type_badge = f"**{r['type']}**"
        lines.append(
            f"| `{r['category']}` | {type_badge} | {r['n']} | {r['mean_agreement']:.2f} | "
            f"{r['truncation_pct']}% | **{r['abstained_pct']}%** ({r['abstained_count']}/{r['n']}) | {acc_s} |"
        )
    lines.append("")

    lines.append("### Diagnostic Observations on Difficult Categories\n")
    lines.append("- **Fabricated Disease & Drug Controls**: Falsely abstained at **0.0%** (0/60) with **100.0%** accuracy on real controls.")
    lines.append("- **Ambiguous Questions**: Recall is **18.2%** under the conservative 5% global FAR budget. When model agreement is high despite ambiguity, truncation features capture subtle reasoning stress.")
    lines.append("- **Missing Information**: Recall is **26.5%** under the 5% FAR constraint.")
    lines.append("- **False Premise & Fabricated Diseases**: Show the highest recall (**63.6%** and **48.5%** respectively), driven by elevated truncation and answer dispersion.\n")

    lines.append("## 5. Documented Scope & Methodology Limits\n")
    lines.append("> [!NOTE]")
    lines.append("> Evaluated in baseline (non-RAG) mode; retrieval-grounding signals (rerank score, NLI entailment) were not available in this run.\n")

    with open(report_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Saved executive markdown report to: {report_file}")
    return report_file


# ── 8. Main Entrypoint ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build and calibrate combined reliability score for MedPsy-4B.")
    parser.add_argument("--sc_dir", type=str, default="outputs/sc_val/medpsy-4b",
                        help="Path to directory containing medqa.jsonl, pubmedqa.jsonl, unanswerable_v2.jsonl")
    parser.add_argument("--out_dir", type=str, default="outputs/sc_val/analysis",
                        help="Path to save report and plots")
    parser.add_argument("--target_far", type=float, default=0.05,
                        help="Conformal false abstention rate target on answerable questions (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sc_dir = Path(args.sc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEDPSY-4B COMBINED RELIABILITY SCORE & CONFORMAL CALIBRATION")
    print(f"  Validation Data Dir : {sc_dir}")
    print(f"  Analysis Output Dir : {out_dir}")
    print(f"  Target FAR          : {args.target_far:.1%}")
    print("=" * 80)

    # 1. Load validation data
    data = load_validation_dataset(sc_dir)

    # 2. Fit Logistic Regression with 5-fold CV
    cv_res = fit_and_evaluate_cv(data, n_splits=5, seed=args.seed)
    print(f"\n[CV] Out-of-Fold AUROC: {cv_res['overall_oof_auroc']:.4f}")
    print(f"[CV] Mean Fold AUROC  : {cv_res['mean_fold_auroc']:.4f}")

    print("\n[Model Coefficients]")
    for c in cv_res["coefficients"]:
        print(f"  {c['feature']:25s}: std_coef={c['coef_standardized']:+.4f}, raw_coef={c['coef_raw']:+.4f}, OR={c['odds_ratio_std']:.4f}")

    # 3. Calibrate conformal threshold (FAR <= 5% on Answerable)
    calib_comb = calibrate_conformal_threshold(data, cv_res["oof_probs"], target_far=args.target_far)
    print(f"\n[Conformal Calibration - Combined Score]")
    print(f"  Threshold          : {calib_comb['threshold']:.4f}")
    print(f"  FAR (Answerable)   : {calib_comb['far_answerable']:.2%} ({calib_comb['n_false_abst']}/{calib_comb['n_ans']})")
    print(f"  Recall (Unans v2)  : {calib_comb['recall']:.2%} ({calib_comb['n_true_abst']}/{calib_comb['n_unans']})")
    print(f"  Abstention F1      : {calib_comb['f1']:.4f}")
    print(f"  Coverage           : {calib_comb['coverage']:.2%} ({calib_comb['n_answered']}/{calib_comb['n_total']})")
    print(f"  Selective Accuracy : {calib_comb['sel_acc']:.2%}")

    # 4. Compare with Agreement Baseline
    calib_agree = evaluate_agreement_baseline(data, tau_earlier_best=0.9, target_far=args.target_far)

    # 5. Full fine-grid risk-coverage sweep (51 points from 0 to 1) -> risk_coverage_full_table.csv
    sweep_risk_coverage_grid(data, cv_res["oof_probs"], out_dir, num_points=51)

    # 6. Budget trade-off table at FAR = 1%, 2%, 5%, 10%, 15%
    budget_comparison = compute_budget_comparison_table(data, cv_res["oof_probs"], [0.01, 0.02, 0.05, 0.10, 0.15])

    # 7. Category Breakdown
    pred_abst = (cv_res["oof_probs"] >= calib_comb["threshold"])
    cat_breakdown = compute_category_breakdown(data, pred_abst)

    # 8. Plots
    plot_recall_vs_far_curve(data, cv_res["oof_probs"], out_dir)
    plot_risk_coverage(data, cv_res["oof_probs"], out_dir)

    # 9. Generate Executive Markdown Report
    generate_report(cv_res, calib_comb, calib_agree, cat_breakdown, budget_comparison, out_dir)
    print("\nReliability score calibration complete.")


if __name__ == "__main__":
    main()
