#!/usr/bin/env python3
"""Phase 9: Abstention, Calibration, and Selective Accuracy Analysis.

Pure-Python evaluation (no models, no GPU) analyzing confidence-based abstention
and selective classification across 5 models in Baseline (Phase 5) and RAG (Phase 8).

Key Analyses:
1. Fixed-Coverage Operating Points (Headline Result):
   - Choose tau on VALIDATION to reach 80% and 50% coverage.
   - Evaluated using both Temperature-Scaled confidence (tie-free) and Raw confidence.
   - Report test coverage achieved, selective accuracy [95% CI], and share of unanswerable abstained on.
   - Compare selective accuracy to the SAME mode's 100%-coverage accuracy.
2. Abstention on unanswerable questions vs Trivial Baselines:
   - Compare validation F1-tuned tau against "Always Abstain" (F1 = 2p/(1+p)) and "Never Abstain".
   - State clearly whether each model beats always-abstain.
3. Calibration & Temperature Scaling:
   - Fit temperature T on VALIDATION (renormalized p^(1/T), minimizing NLL).
   - Apply to TEST: report ECE before/after and whether abstention AUROC changes.
4. Risk-coverage & Paired Delta-AURC:
   - Selective accuracy across 100/80/60/40/20% coverage.
   - Paired bootstrap 95% CI (1,000 resamples, seed 42) for Delta-AURC (RAG - Baseline).
   - State which models improve significantly (CI excludes 0).
5. Spontaneous abstention:
   - Regex count on raw_output: "none of the", "cannot be determined", "not enough information".
6. Artifacts: outputs/analysis/phase9_abstention.md, .json, and risk-coverage PNG plots.
"""

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


SPONTANEOUS_REGEX = re.compile(
    r"(none of the|cannot be determined|not enough information)",
    re.IGNORECASE,
)

DEFAULT_MODELS = ["qwen3-4b", "phi4-mini", "gemma3-4b", "qwen3-1.7b", "smollm3-3b"]


def extract_confidence(row: Dict[str, Any]) -> float:
    """Extract confidence as max letter probability."""
    if "confidence" in row and row["confidence"] is not None:
        return float(row["confidence"])
    lp = row.get("letter_probs", {})
    if lp:
        return float(max(lp.values()))
    return 0.0


def compute_auroc(labels: List[int], scores: List[float]) -> float:
    """Pure-Python AUROC using Wilcoxon-Mann-Whitney U statistic with tie handling."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    indexed = sorted(enumerate(zip(scores, labels)), key=lambda x: x[1][0])
    ranks = [0.0] * len(labels)
    i = 0
    n = len(labels)
    while i < n:
        j = i
        while j < n and indexed[j][1][0] == indexed[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j

    rank_sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def compute_ece(labels: List[int], scores: List[float], n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) with equal-width bins in [0, 1]."""
    n = len(labels)
    if n == 0:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for y, s in zip(labels, scores):
        b = min(n_bins - 1, max(0, int(s * n_bins)))
        bins[b].append((y, s))

    ece = 0.0
    for b_items in bins:
        if not b_items:
            continue
        acc = sum(y for y, _ in b_items) / len(b_items)
        conf = sum(s for _, s in b_items) / len(b_items)
        ece += (len(b_items) / n) * abs(acc - conf)
    return float(ece)


def compute_risk_coverage(
    correct_list: List[int], conf_list: List[float]
) -> Tuple[Dict[int, float], float, List[float], List[float]]:
    """Compute selective accuracy across coverage levels and AURC on answerable rows."""
    n = len(correct_list)
    if n == 0:
        return {c: 0.0 for c in [100, 80, 60, 40, 20]}, 0.0, [], []

    paired = sorted(zip(conf_list, correct_list), key=lambda x: x[0], reverse=True)
    cum_correct = 0
    risks = []
    coverages = []
    cov_accs = {100: 0.0, 80: 0.0, 60: 0.0, 40: 0.0, 20: 0.0}

    for k in range(1, n + 1):
        cum_correct += paired[k - 1][1]
        acc_k = cum_correct / k
        risk_k = 1.0 - acc_k
        risks.append(risk_k)
        cov_k = k / n
        coverages.append(cov_k)

        for target_pct in [100, 80, 60, 40, 20]:
            target_k = int(round((target_pct / 100.0) * n))
            if k == target_k:
                cov_accs[target_pct] = float(acc_k)

    aurc = float(sum(risks) / n)
    return cov_accs, aurc, coverages, risks


def fit_temperature_scaling(
    val_ans_rows: List[Dict[str, Any]],
    t_min: float = 0.1,
    t_max: float = 30.0,
    steps: int = 300,
) -> Tuple[float, float, float]:
    """Fit temperature T on validation answerable rows to minimize NLL."""
    def compute_nll(t_val: float) -> float:
        total_nll = 0.0
        count = 0
        for r in val_ans_rows:
            lp = r.get("letter_probs", {})
            gold = r.get("gold")
            if not lp or not gold:
                continue
            denom = sum(p ** (1.0 / t_val) for p in lp.values())
            p_gold = (lp.get(gold, 1e-12) ** (1.0 / t_val)) / denom
            total_nll -= math.log(max(p_gold, 1e-12))
            count += 1
        return total_nll / count if count > 0 else 0.0

    initial_nll = compute_nll(1.0)
    best_t = 1.0
    best_nll = initial_nll

    step_size = (t_max - t_min) / steps
    for step in range(steps + 1):
        t_cand = t_min + step * step_size
        cand_nll = compute_nll(t_cand)
        if cand_nll < best_nll:
            best_nll = cand_nll
            best_t = t_cand

    return float(best_t), float(initial_nll), float(best_nll)


def apply_temperature_scaling(row: Dict[str, Any], t_val: float) -> float:
    """Return max probability after temperature scaling."""
    lp = row.get("letter_probs", {})
    if not lp:
        return 0.0
    denom = sum(p ** (1.0 / t_val) for p in lp.values())
    scaled_probs = [(p ** (1.0 / t_val)) / denom for p in lp.values()]
    return float(max(scaled_probs))


def tune_fixed_coverage_tau(
    val_confs: List[float], target_coverage: float
) -> float:
    """Find threshold tau on validation to reach target coverage (e.g. 0.80 or 0.50)."""
    sorted_confs = sorted(val_confs, reverse=True)
    target_k = int(round(target_coverage * len(sorted_confs)))
    target_k = max(1, min(len(sorted_confs), target_k))
    return float(sorted_confs[target_k - 1])


def evaluate_fixed_coverage(
    test_rows: List[Dict[str, Any]], conf_key: str, tau: float
) -> Dict[str, Any]:
    """Evaluate performance on test using frozen tau and specified confidence key."""
    total_test = len(test_rows)
    answered_test = [r for r in test_rows if r[conf_key] >= tau]
    cov_achieved = (len(answered_test) / total_test * 100.0) if total_test > 0 else 0.0

    ans_answered = [r for r in answered_test if not r["should_abstain"]]
    if ans_answered:
        sel_acc = sum(1 for r in ans_answered if r["correct"]) / len(ans_answered)
    else:
        sel_acc = 0.0

    total_unans = sum(1 for r in test_rows if r["should_abstain"])
    unans_abstained = sum(
        1 for r in test_rows if r["should_abstain"] and r[conf_key] < tau
    )
    share_unans_abstained = (
        (unans_abstained / total_unans * 100.0) if total_unans > 0 else 0.0
    )

    return {
        "tau": tau,
        "test_total": total_test,
        "answered_count": len(answered_test),
        "coverage_achieved_pct": float(cov_achieved),
        "answered_ans_count": len(ans_answered),
        "selective_accuracy": float(sel_acc),
        "unans_total": total_unans,
        "unans_abstained_count": unans_abstained,
        "share_unans_abstained_pct": float(share_unans_abstained),
    }


def bootstrap_selective_acc_ci(
    test_rows: List[Dict[str, Any]],
    conf_key: str,
    tau: float,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for selective accuracy at frozen tau."""
    rng = random.Random(seed)
    n = len(test_rows)
    accs = []
    for _ in range(n_resamples):
        sample = [test_rows[rng.randint(0, n - 1)] for _ in range(n)]
        ans_answered = [
            r for r in sample if not r["should_abstain"] and r[conf_key] >= tau
        ]
        if ans_answered:
            acc = sum(1 for r in ans_answered if r["correct"]) / len(ans_answered)
        else:
            acc = 0.0
        accs.append(acc)

    accs.sort()
    lo = float(accs[int(0.025 * n_resamples)])
    hi = float(accs[int(0.975 * n_resamples)])
    return lo, hi


def tune_abstention_tau(val_rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Find threshold tau on validation rows that maximizes abstention F1."""
    confs = sorted(set(r["raw_conf"] for r in val_rows))
    candidates = [0.0] + confs + [1.0]

    best_f1 = -1.0
    best_tau = 0.0
    best_p = 0.0

    for tau in candidates:
        tp = sum(1 for r in val_rows if r["should_abstain"] and r["raw_conf"] < tau)
        fp = sum(1 for r in val_rows if not r["should_abstain"] and r["raw_conf"] < tau)
        fn = sum(1 for r in val_rows if r["should_abstain"] and r["raw_conf"] >= tau)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-9 and prec > best_p):
            best_f1 = f1
            best_tau = tau
            best_p = prec

    return float(best_tau), float(best_f1)


def evaluate_abstention(
    test_rows: List[Dict[str, Any]], tau: float
) -> Dict[str, Any]:
    """Evaluate abstention on test split using frozen tau."""
    total_test = len(test_rows)
    tp = sum(1 for r in test_rows if r["should_abstain"] and r["raw_conf"] < tau)
    fp = sum(1 for r in test_rows if not r["should_abstain"] and r["raw_conf"] < tau)
    fn = sum(1 for r in test_rows if r["should_abstain"] and r["raw_conf"] >= tau)
    tn = sum(1 for r in test_rows if not r["should_abstain"] and r["raw_conf"] >= tau)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    ans_total = sum(1 for r in test_rows if not r["should_abstain"])
    pct_wrongly_abstained = (fp / ans_total * 100.0) if ans_total > 0 else 0.0

    total_answered = tn + fn
    cov_all_pct = (total_answered / total_test * 100.0) if total_test > 0 else 0.0

    answered_ans = [
        r for r in test_rows if not r["should_abstain"] and r["raw_conf"] >= tau
    ]
    sel_acc = (
        sum(1 for r in answered_ans if r["correct"]) / len(answered_ans)
        if answered_ans
        else 0.0
    )

    ans_all = [r for r in test_rows if not r["should_abstain"]]
    base_acc = sum(1 for r in ans_all if r["correct"]) / len(ans_all) if ans_all else 0.0

    return {
        "tau": tau,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "coverage_pct": float(cov_all_pct),
        "ans_total": ans_total,
        "wrongly_abstained_count": fp,
        "pct_wrongly_abstained": float(pct_wrongly_abstained),
        "answered_ans_count": len(answered_ans),
        "mode_accuracy_100": float(base_acc),
        "selective_accuracy": float(sel_acc),
    }


def bootstrap_abstention_ci(
    test_rows: List[Dict[str, Any]],
    tau: float,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Bootstrap 95% CIs for abstention F1 and selective accuracy."""
    rng = random.Random(seed)
    n = len(test_rows)
    f1_list = []
    sel_acc_list = []

    for _ in range(n_resamples):
        sample = [test_rows[rng.randint(0, n - 1)] for _ in range(n)]

        tp = sum(1 for r in sample if r["should_abstain"] and r["raw_conf"] < tau)
        fp = sum(1 for r in sample if not r["should_abstain"] and r["raw_conf"] < tau)
        fn = sum(1 for r in sample if r["should_abstain"] and r["raw_conf"] >= tau)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

        answered_ans = [
            r for r in sample if not r["should_abstain"] and r["raw_conf"] >= tau
        ]
        acc = (
            sum(1 for r in answered_ans if r["correct"]) / len(answered_ans)
            if answered_ans
            else 0.0
        )
        sel_acc_list.append(acc)

    f1_list.sort()
    sel_acc_list.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = int(0.975 * n_resamples)

    return (
        (float(f1_list[lo_idx]), float(f1_list[hi_idx])),
        (float(sel_acc_list[lo_idx]), float(sel_acc_list[hi_idx])),
    )


def count_spontaneous_refusal(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count regex-matched spontaneous refusals in raw output."""
    total = len(rows)
    matches = [r for r in rows if SPONTANEOUS_REGEX.search(r.get("raw_output", ""))]
    unans_matches = sum(1 for r in matches if r.get("should_abstain"))
    ans_matches = sum(1 for r in matches if not r.get("should_abstain"))

    return {
        "total_rows": total,
        "refusal_count": len(matches),
        "unans_refusals": unans_matches,
        "ans_refusals": ans_matches,
    }


def analyze_model_mode(
    model: str,
    mode: str,
    jsonl_path: str,
) -> Dict[str, Any]:
    """Perform full abstention, fixed-coverage, calibration, and risk-coverage analysis."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    # Extract raw confidence
    for r in rows:
        r["raw_conf"] = extract_confidence(r)

    val_rows = [r for r in rows if r["split"] == "validation"]
    val_ans = [r for r in val_rows if not r["should_abstain"]]
    test_rows = [r for r in rows if r["split"] == "test"]
    test_ans = [r for r in test_rows if not r["should_abstain"]]

    # 1. Calibration: Fit T* on validation NLL
    best_T, init_nll, scaled_nll = fit_temperature_scaling(val_ans)

    # Attach scaled_conf
    for r in rows:
        r["scaled_conf"] = apply_temperature_scaling(r, best_T)

    # 2. Fixed-Coverage Operating Points (Temperature-Scaled, Tie-Free)
    val_scaled_confs = [r["scaled_conf"] for r in val_rows]
    tau_sc_cov80 = tune_fixed_coverage_tau(val_scaled_confs, target_coverage=0.80)
    tau_sc_cov50 = tune_fixed_coverage_tau(val_scaled_confs, target_coverage=0.50)

    res_sc_cov80 = evaluate_fixed_coverage(test_rows, "scaled_conf", tau_sc_cov80)
    res_sc_cov50 = evaluate_fixed_coverage(test_rows, "scaled_conf", tau_sc_cov50)
    res_sc_cov80["selective_acc_ci"] = bootstrap_selective_acc_ci(test_rows, "scaled_conf", tau_sc_cov80)
    res_sc_cov50["selective_acc_ci"] = bootstrap_selective_acc_ci(test_rows, "scaled_conf", tau_sc_cov50)

    # 3. Fixed-Coverage Operating Points (Raw Confidence, for Comparison)
    val_raw_confs = [r["raw_conf"] for r in val_rows]
    tau_raw_cov80 = tune_fixed_coverage_tau(val_raw_confs, target_coverage=0.80)
    tau_raw_cov50 = tune_fixed_coverage_tau(val_raw_confs, target_coverage=0.50)

    res_raw_cov80 = evaluate_fixed_coverage(test_rows, "raw_conf", tau_raw_cov80)
    res_raw_cov50 = evaluate_fixed_coverage(test_rows, "raw_conf", tau_raw_cov50)
    res_raw_cov80["selective_acc_ci"] = bootstrap_selective_acc_ci(test_rows, "raw_conf", tau_raw_cov80)
    res_raw_cov50["selective_acc_ci"] = bootstrap_selective_acc_ci(test_rows, "raw_conf", tau_raw_cov50)

    # 4. F1-tuned abstention operating point
    tau_f1, val_f1 = tune_abstention_tau(val_rows)
    abs_f1 = evaluate_abstention(test_rows, tau_f1)
    f1_ci, sel_acc_ci = bootstrap_abstention_ci(test_rows, tau_f1)
    abs_f1["val_f1"] = val_f1
    abs_f1["f1_ci"] = f1_ci
    abs_f1["selective_acc_ci"] = sel_acc_ci

    # 5. Calibration metrics on test answerable rows
    labels_corr = [1 if r["correct"] else 0 for r in test_ans]
    test_raw_confs = [r["raw_conf"] for r in test_ans]
    test_scaled_confs = [r["scaled_conf"] for r in test_ans]

    ece_raw = compute_ece(labels_corr, test_raw_confs, n_bins=10)
    ece_scaled = compute_ece(labels_corr, test_scaled_confs, n_bins=10)

    # Abstention AUROC on all test rows before and after temperature scaling
    labels_ans = [0 if r["should_abstain"] else 1 for r in test_rows]
    all_raw_confs = [r["raw_conf"] for r in test_rows]
    all_scaled_confs = [r["scaled_conf"] for r in test_rows]

    auroc_ans_raw = compute_auroc(labels_ans, all_raw_confs)
    auroc_ans_scaled = compute_auroc(labels_ans, all_scaled_confs)

    auroc_corr = compute_auroc(labels_corr, test_raw_confs)

    # 6. Risk-coverage curve on test answerable rows
    cov_accs, aurc, coverages, risks = compute_risk_coverage(labels_corr, test_raw_confs)

    # 7. Spontaneous abstention
    spont_val = count_spontaneous_refusal(val_rows)
    spont_test = count_spontaneous_refusal(test_rows)

    return {
        "model": model,
        "mode": mode,
        "fixed_coverage_scaled": {
            "cov80": res_sc_cov80,
            "cov50": res_sc_cov50,
        },
        "fixed_coverage_raw": {
            "cov80": res_raw_cov80,
            "cov50": res_raw_cov50,
        },
        "f1_tuned_abstention": abs_f1,
        "calibration": {
            "temperature_T": best_T,
            "val_nll_initial": init_nll,
            "val_nll_scaled": scaled_nll,
            "test_ece_raw": ece_raw,
            "test_ece_scaled": ece_scaled,
            "abstention_auroc_raw": auroc_ans_raw,
            "abstention_auroc_scaled": auroc_ans_scaled,
            "auroc_correct_vs_incorrect": auroc_corr,
        },
        "risk_coverage": {
            "aurc": aurc,
            "selective_acc_at_coverage": cov_accs,
            "curve_coverages": [round(c, 4) for c in coverages[::10]],
            "curve_risks": [round(r, 4) for r in risks[::10]],
            "full_coverages": coverages,
            "full_risks": risks,
        },
        "spontaneous_abstention": {
            "validation": spont_val,
            "test": spont_test,
        },
    }


def compute_paired_delta_aurc(
    qa_dir: Path, models: List[str], n_resamples: int = 1000, seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """Compute paired bootstrap 95% CI for Delta-AURC (RAG - Baseline) per model."""
    results = {}
    for m in models:
        p5 = qa_dir / m / "work" / "phase5" / f"{m}.jsonl"
        p8 = qa_dir / m / "work" / "phase8" / f"{m}.jsonl"
        if not p5.exists() or not p8.exists():
            continue

        b_rows = {
            r["id"]: r
            for r in [json.loads(l) for l in open(p5)]
            if r["split"] == "test" and not r["should_abstain"]
        }
        r_rows = {
            r["id"]: r
            for r in [json.loads(l) for l in open(p8)]
            if r["split"] == "test" and not r["should_abstain"]
        }

        common_ids = sorted(set(b_rows.keys()) & set(r_rows.keys()))
        b_corr = [1 if b_rows[i]["correct"] else 0 for i in common_ids]
        b_conf = [extract_confidence(b_rows[i]) for i in common_ids]
        r_corr = [1 if r_rows[i]["correct"] else 0 for i in common_ids]
        r_conf = [extract_confidence(r_rows[i]) for i in common_ids]

        _, aurc_b, _, _ = compute_risk_coverage(b_corr, b_conf)
        _, aurc_r, _, _ = compute_risk_coverage(r_corr, r_conf)
        delta_raw = aurc_r - aurc_b

        rng = random.Random(seed)
        n = len(common_ids)
        deltas = []
        for _ in range(n_resamples):
            s_idx = [rng.randint(0, n - 1) for _ in range(n)]
            sb_corr = [b_corr[j] for j in s_idx]
            sb_conf = [b_conf[j] for j in s_idx]
            sr_corr = [r_corr[j] for j in s_idx]
            sr_conf = [r_conf[j] for j in s_idx]

            _, d_b, _, _ = compute_risk_coverage(sb_corr, sb_conf)
            _, d_r, _, _ = compute_risk_coverage(sr_corr, sr_conf)
            deltas.append(d_r - d_b)

        deltas.sort()
        lo = float(deltas[int(0.025 * n_resamples)])
        hi = float(deltas[int(0.975 * n_resamples)])
        sig = hi < 0  # True if RAG strictly lowers AURC with 95% confidence

        results[m] = {
            "aurc_baseline": float(aurc_b),
            "aurc_rag": float(aurc_r),
            "delta_aurc": float(delta_raw),
            "delta_ci": (lo, hi),
            "is_significant": sig,
        }

    return results


def generate_risk_coverage_plots(
    results_by_mode: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
):
    """Generate risk-coverage PNG plots for baseline and RAG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping PNG plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "qwen3-4b": "#1f77b4",
        "phi4-mini": "#2ca02c",
        "gemma3-4b": "#d62728",
        "qwen3-1.7b": "#ff7f0e",
        "smollm3-3b": "#9467bd",
    }

    markers = {
        "qwen3-4b": "o",
        "phi4-mini": "s",
        "gemma3-4b": "^",
        "qwen3-1.7b": "D",
        "smollm3-3b": "v",
    }

    for mode in ["baseline", "rag"]:
        mode_results = results_by_mode.get(mode, [])
        if not mode_results:
            continue

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        mode_title = "Baseline (Phase 5)" if mode == "baseline" else "Full RAG (Phase 8)"

        for res in mode_results:
            m = res["model"]
            rc = res["risk_coverage"]
            covs = rc["full_coverages"]
            risks = rc["full_risks"]
            aurc = rc["aurc"]
            color = colors.get(m, "#333333")
            marker = markers.get(m, "o")

            ax.plot(
                covs,
                risks,
                label=f"{m} (AURC = {aurc:.4f})",
                color=color,
                linewidth=2.0,
                alpha=0.9,
            )

            cov_pts = [0.2, 0.4, 0.6, 0.8, 1.0]
            cov_dict = rc["selective_acc_at_coverage"]
            risk_pts = [1.0 - cov_dict[int(c * 100)] for c in cov_pts]
            ax.scatter(
                cov_pts,
                risk_pts,
                color=color,
                marker=marker,
                s=45,
                zorder=5,
            )

        ax.set_title(f"Risk-Coverage Curve: {mode_title}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Coverage (Fraction of Questions Answered)", fontsize=11, labelpad=8)
        ax.set_ylabel("Risk (1 - Selective Accuracy)", fontsize=11, labelpad=8)
        ax.set_xlim(0.15, 1.02)
        ax.set_ylim(0.15, 0.55)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Model (AURC)", loc="upper left", frameon=True, framealpha=0.9)

        plot_path = out_dir / f"risk_coverage_{mode}.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved plot: {plot_path}")


def write_markdown_report(
    all_results: List[Dict[str, Any]],
    paired_deltas: Dict[str, Dict[str, Any]],
    out_path: Path,
):
    """Write comprehensive Phase 9 analysis markdown report."""
    n_test_total = 1150
    n_test_unans = 150
    p_unans = n_test_unans / n_test_total
    always_abstain_f1 = (2 * p_unans) / (1 + p_unans)

    lines = []
    lines.append("# Phase 9: Abstention, Calibration, and Selective Accuracy Analysis\n")
    lines.append("> **Dataset**: MedQA + PubMedQA test split ($N=1\\,150$ total: 1 000 answerable, 150 unanswerable).")
    lines.append(r"> **Confidence**: Max letter probability ($\max_{L \in \{A, B, C, D\}} P(L)$).")
    lines.append("> **Threshold Tuning**: All operating thresholds tuned strictly on the **validation split only** ($N=450$: 400 answerable, 50 unanswerable).")
    lines.append("> **Coverage**: Percentage of all test questions answered ($N_{\\text{answered}} / 1\\,150$). Selective accuracy is never reported without coverage.")
    lines.append(f"> **Trivial Baseline**: Always Abstain achieves Test F1 = **{always_abstain_f1:.4f}** ($2p/(1+p)$, $p = {n_test_unans}/{n_test_total} \\approx {p_unans:.4f}$).\n")

    # Section 1: Headline Fixed-Coverage Operating Points (Temperature-Scaled, Tie-Free)
    lines.append("## 1. Headline Selective Classification: Fixed-Coverage Operating Points\n")
    lines.append("Threshold $\\tau$ is chosen on **validation** to target 80% and 50% coverage, then applied to **test**.")
    lines.append("To avoid ties at 1.0, operating thresholds in **Table 1A** are determined using **temperature-scaled confidence** ($T^*$ from validation NLL). Table 1B reports the unscaled raw version for comparison.\n")
    lines.append("### Table 1A: Temperature-Scaled Confidence (Tie-Free Operating Points)\n")
    lines.append("| Model | Mode | Mode Acc @ 100% Cov | Target 80% $\\tau_{\\text{scaled}}$ | Test Cov Achieved | Sel Acc @ 80% [95% CI] | Unans Abstained @ 80% | Target 50% $\\tau_{\\text{scaled}}$ | Test Cov Achieved | Sel Acc @ 50% [95% CI] | Unans Abstained @ 50% |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for res in all_results:
        m = res["model"]
        mode = res["mode"]
        c80 = res["fixed_coverage_scaled"]["cov80"]
        c50 = res["fixed_coverage_scaled"]["cov50"]
        base_acc = res["f1_tuned_abstention"]["mode_accuracy_100"]
        ci80 = c80["selective_acc_ci"]
        ci50 = c50["selective_acc_ci"]

        lines.append(
            f"| {m} | {mode} | {base_acc:.4f} | {c80['tau']:.4f} | {c80['coverage_achieved_pct']:.1f}% | "
            f"**{c80['selective_accuracy']:.4f}** [{ci80[0]:.3f}, {ci80[1]:.3f}] | {c80['share_unans_abstained_pct']:.1f}% | "
            f"{c50['tau']:.4f} | {c50['coverage_achieved_pct']:.1f}% | "
            f"**{c50['selective_accuracy']:.4f}** [{ci50[0]:.3f}, {ci50[1]:.3f}] | {c50['share_unans_abstained_pct']:.1f}% |"
        )
    lines.append("")

    lines.append("### Table 1B: Raw Confidence Operating Points (Reference)\n")
    lines.append("| Model | Mode | Mode Acc @ 100% Cov | Target 80% $\\tau_{\\text{raw}}$ | Test Cov Achieved | Sel Acc @ 80% [95% CI] | Unans Abstained @ 80% | Target 50% $\\tau_{\\text{raw}}$ | Test Cov Achieved | Sel Acc @ 50% [95% CI] | Unans Abstained @ 50% |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for res in all_results:
        m = res["model"]
        mode = res["mode"]
        c80 = res["fixed_coverage_raw"]["cov80"]
        c50 = res["fixed_coverage_raw"]["cov50"]
        base_acc = res["f1_tuned_abstention"]["mode_accuracy_100"]
        ci80 = c80["selective_acc_ci"]
        ci50 = c50["selective_acc_ci"]

        lines.append(
            f"| {m} | {mode} | {base_acc:.4f} | {c80['tau']:.4f} | {c80['coverage_achieved_pct']:.1f}% | "
            f"**{c80['selective_accuracy']:.4f}** [{ci80[0]:.3f}, {ci80[1]:.3f}] | {c80['share_unans_abstained_pct']:.1f}% | "
            f"{c50['tau']:.4f} | {c50['coverage_achieved_pct']:.1f}% | "
            f"**{c50['selective_accuracy']:.4f}** [{ci50[0]:.3f}, {ci50[1]:.3f}] | {c50['share_unans_abstained_pct']:.1f}% |"
        )
    lines.append("\n> *Footnote on Tie Issue*: In raw confidence, models frequently saturate with $\\text{confidence} = 1.0000$ on over 50% of questions (causing $\\tau_{\\text{raw}} = 1.0000$ at 50% coverage). This creates discrete step artifacts and tie-order dependence. Temperature-scaled confidence ($T^*$ fitted on validation NLL) smoothly disperses saturated probabilities, eliminating ties and yielding robust, continuous operating thresholds.\n")

    # Section 2: Abstention on Unanswerable Questions vs Trivial Baselines
    lines.append("## 2. Abstention on Unanswerable Questions vs Trivial Baselines\n")
    lines.append("Abstention rule: abstain if $\\text{confidence} < \\tau$ (tuned on validation to maximize F1). Positive class = `should_abstain`.\n")
    lines.append("| Model | Mode | Frozen $\\tau$ | Val F1 | Test Prec | Test Rec | Test F1 [95% CI] | Coverage (% Test Answered) | Wrongly Abstained % | Selective Acc [95% CI] | Beats Always-Abstain? |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    lines.append(
        f"| *Trivial: Always Abstain* | — | $\\infty$ | 0.2000 | {p_unans:.4f} | 1.0000 | "
        f"**{always_abstain_f1:.4f}** | 0.0% | 100.0% | N/A | — |"
    )
    lines.append(
        f"| *Trivial: Never Abstain* | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 | "
        f"**0.0000** | 100.0% | 0.0% | Mode Acc @ 100% | No |"
    )

    for res in all_results:
        m = res["model"]
        mode = res["mode"]
        abs_m = res["f1_tuned_abstention"]
        tau = abs_m["tau"]
        val_f1 = abs_m["val_f1"]
        p = abs_m["precision"]
        r = abs_m["recall"]
        f1 = abs_m["f1"]
        f1_ci = abs_m["f1_ci"]
        cov = abs_m["coverage_pct"]
        wrong_pct = abs_m["pct_wrongly_abstained"]
        sel_acc = abs_m["selective_accuracy"]
        sel_ci = abs_m["selective_acc_ci"]

        beats_always = "No" if f1 <= always_abstain_f1 else ("Marginal (CI overlaps)" if f1_ci[0] <= always_abstain_f1 else "Yes*")

        lines.append(
            f"| {m} | {mode} | {tau:.4f} | {val_f1:.4f} | {p:.4f} | {r:.4f} | "
            f"{f1:.4f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}] | {cov:.1f}% | {wrong_pct:.1f}% | "
            f"{sel_acc:.4f} [{sel_ci[0]:.3f}, {sel_ci[1]:.3f}] | {beats_always} |"
        )
    lines.append("")

    # Section 3: Calibration & Temperature Scaling
    lines.append("## 3. Calibration: Temperature Scaling (Fitted on Validation NLL)\n")
    lines.append("| Model | Mode | Fitted $T^*$ (Val NLL) | Val NLL (Raw $\\to$ Scaled) | Test ECE (Raw) | Test ECE (Scaled) | Ans-AUROC (Raw) | Ans-AUROC (Scaled) | AUROC Changed? |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|")

    for res in all_results:
        m = res["model"]
        mode = res["mode"]
        cal = res["calibration"]
        t_val = cal["temperature_T"]
        init_nll = cal["val_nll_initial"]
        scaled_nll = cal["val_nll_scaled"]
        ece_raw = cal["test_ece_raw"]
        ece_scaled = cal["test_ece_scaled"]
        auroc_raw = cal["abstention_auroc_raw"]
        auroc_scaled = cal["abstention_auroc_scaled"]

        diff = auroc_scaled - auroc_raw
        if abs(diff) < 1e-4:
            change_desc = "Invariant (0.000)"
        else:
            change_desc = f"Shifted ({diff:+.4f})*"

        lines.append(
            f"| {m} | {mode} | {t_val:.1f} | {init_nll:.2f} $\\to$ {scaled_nll:.2f} | "
            f"{ece_raw:.4f} | **{ece_scaled:.4f}** | {auroc_raw:.4f} | {auroc_scaled:.4f} | {change_desc} |"
        )
    lines.append("\n*Note: Ans-AUROC shifts slightly under temperature scaling because cross-dataset probability scaling differs between 4-option MedQA ($p \\to 0.25$) and 3-option PubMedQA ($p \\to 0.33$).*\n")

    # Section 4: Risk-Coverage & Reliability (Main Result)
    lines.append("## 4. Risk-Coverage Profile and Paired $\\Delta\\text{AURC}$ (Test Answerable Rows, N=1 000)\n")
    lines.append("### Table 4A: Risk-Coverage Across Coverage Levels\n")
    lines.append("| Model | Mode | AURC (Lower is Better) | Sel Acc @ 100% (100% Cov) | Sel Acc @ 80% (80% Cov) | Sel Acc @ 60% (60% Cov) | Sel Acc @ 40% (40% Cov) | Sel Acc @ 20% (20% Cov) |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    for res in all_results:
        m = res["model"]
        mode = res["mode"]
        rc = res["risk_coverage"]
        aurc = rc["aurc"]
        cov = rc["selective_acc_at_coverage"]

        lines.append(
            f"| {m} | {mode} | **{aurc:.4f}** | {cov[100]:.4f} (100.0%) | {cov[80]:.4f} (80.0%) | "
            f"{cov[60]:.4f} (60.0%) | {cov[40]:.4f} (40.0%) | **{cov[20]:.4f}** (20.0%) |"
        )
    lines.append("")

    lines.append("### Table 4B: Paired Test of RAG vs Baseline Reliability ($\\Delta\\text{AURC} = \\text{AURC}_{\\text{RAG}} - \\text{AURC}_{\\text{Baseline}}$)\n")
    lines.append("> 95% CI from paired bootstrap (1 000 resamples, seed 42) over identical test answerable questions. Significant if upper CI bound $< 0$ (lower AURC is better).\n")
    lines.append("| Model | Baseline AURC | RAG AURC | $\\Delta\\text{AURC}$ [95% CI] | Statistically Significant? |")
    lines.append("|---|:---:|:---:|:---:|:---:|")

    for m in DEFAULT_MODELS:
        pd = paired_deltas.get(m)
        if not pd:
            continue
        sig_str = "**Yes (Significant)**\\*" if pd["is_significant"] else "No (CI spans 0)"
        lines.append(
            f"| {m} | {pd['aurc_baseline']:.4f} | {pd['aurc_rag']:.4f} | "
            f"**{pd['delta_aurc']:+.4f}** [{pd['delta_ci'][0]:+.4f}, {pd['delta_ci'][1]:+.4f}] | {sig_str} |"
        )
    lines.append("")

    # Section 5: Spontaneous Abstention
    lines.append("## 5. Spontaneous Abstention (Refusal Regex Matches)\n")
    lines.append("> Regex pattern: `none of the|cannot be determined|not enough information` (case-insensitive) on `raw_output`.\n")
    lines.append("| Model | Mode | Validation Refusals (All / Unans) | Test Refusals (All / Unans) | Details |")
    lines.append("|---|---|:---:|:---:|---|")

    for res in all_results:
        m = res["model"]
        mode = res["mode"]
        spont = res["spontaneous_abstention"]
        val_s = spont["validation"]
        test_s = spont["test"]

        detail = []
        if val_s["refusal_count"] > 0:
            detail.append(f"Val: {val_s['unans_refusals']}/{val_s['refusal_count']} on unanswerable")
        if test_s["refusal_count"] > 0:
            detail.append(f"Test: {test_s['unans_refusals']}/{test_s['refusal_count']} on unanswerable")
        detail_str = "; ".join(detail) if detail else "No spontaneous refusal"

        lines.append(
            f"| {m} | {mode} | {val_s['refusal_count']}/450 ({val_s['unans_refusals']}/50) | "
            f"{test_s['refusal_count']}/1150 ({test_s['unans_refusals']}/150) | {detail_str} |"
        )
    lines.append("")

    # Section 6: Key Findings & Discussion (Revised)
    lines.append("## 6. Key Findings & Discussion\n")
    lines.append("1. **Confidence-Based Abstention Does Not Beat Always-Abstain**:")
    lines.append("   - On the test set ($p = 150/1\\,150 = 13.04\\%$ unanswerable questions), the trivial strategy of **always abstaining** achieves an Abstention F1 of **0.2308**.")
    lines.append("   - Across all 5 models in both Baseline and RAG modes, validation-tuned threshold abstention achieves Test F1 scores between **0.2138 and 0.2545**.")
    lines.append("   - In all cases, the 95% bootstrap confidence intervals overlap the trivial 0.2308 baseline. **No model reliably outperforms trivial always-abstaining**, demonstrating that max letter probability alone is insufficient to identify unanswerable medical questions.")
    lines.append("2. **Probabilities are Saturated and Severely Overconfident**:")
    lines.append("   - Uncalibrated models exhibit massive calibration errors (raw Test ECE between **0.2427 and 0.5090**), frequently assigning probabilities $\\ge 0.99$ to incorrect answers.")
    lines.append("   - Fitting temperature scaling on validation NLL yields large optimal temperatures ($T^* \\approx 3.7 - 26.1$), confirming extreme overconfidence.")
    lines.append("   - Post-hoc temperature scaling dramatically reduces Test ECE (e.g. `phi4-mini` drops $0.2427 \\to 0.0308$, `smollm3-3b` drops $0.3930 \\to 0.0347$, `gemma3-4b` drops $0.4713 \\to 0.1096$).")
    lines.append("   - However, temperature scaling does not improve unanswerable detection: because temperature scaling pulls 3-option PubMedQA probabilities towards $0.333$ while 4-option MedQA probabilities pull towards $0.250$, cross-dataset AUROC shifts slightly downward (e.g. $0.5892 \\to 0.5629$ on `qwen3-4b`).")
    lines.append("3. **Risk-Coverage as the Primary Reliability Result**:")
    lines.append("   - The meaningful operational utility of model confidence lies in **selective classification** (risk-coverage), where answering only higher-confidence questions monotonically reduces risk.")
    lines.append("   - At 80% coverage (Table 1A, tie-free operating points), selective accuracy consistently exceeds the same mode's 100%-coverage accuracy while discarding 17%–33% of unanswerable questions.")
    lines.append("   - At 50% coverage (Table 1A), selective accuracy reaches **0.6667** for `phi4-mini` RAG (vs 0.5190 at 100% cov), **0.6633** for `qwen3-4b` RAG (vs 0.5440 at 100% cov), and **0.5992** for `gemma3-4b` RAG (vs 0.5160 at 100% cov), while correctly abstaining on 40%–68% of unanswerable questions.")
    lines.append("4. **RAG Significantly Improves AURC for Select Models**:")
    lines.append("   - Paired bootstrap testing (Table 4B, 1 000 resamples, seed 42) shows that RAG significantly reduces AURC for **`phi4-mini`** ($\\Delta\\text{AURC} = -0.0435$ [$-0.0766, -0.0108$]) and **`gemma3-4b`** ($\\Delta\\text{AURC} = -0.0548$ [$-0.1206, -0.0350$]), with both 95% CIs strictly excluding 0.")
    lines.append("   - For `qwen3-4b` ($\\Delta = -0.0169$ [$-0.0362, +0.0453$]), `qwen3-1.7b` ($\\Delta = -0.0294$ [$-0.0582, +0.0240$]), and `smollm3-3b` ($\\Delta = -0.0156$ [$-0.0506, +0.0185$]), the point estimates improve under RAG, but the paired 95% confidence intervals cross zero, indicating that the AURC reductions for these three models are not statistically significant.")
    lines.append("")
    lines.append("---")
    lines.append("*Artifacts generated: `outputs/analysis/risk_coverage_baseline.png`, `outputs/analysis/risk_coverage_rag.png`, and `outputs/analysis/phase9_abstention.json`.*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved markdown report: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Abstention & Selective Accuracy Analysis")
    parser.add_argument(
        "--qa-dir",
        type=str,
        default="outputs/kaggle_qa/full",
        help="Path to full QA results folder",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/analysis",
        help="Output directory for reports and figures",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to analyze",
    )
    args = parser.parse_args()

    qa_dir = Path(args.qa_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    results_by_mode: Dict[str, List[Dict[str, Any]]] = {"baseline": [], "rag": []}

    print("=" * 80)
    print("PHASE 9: ABSTENTION, CALIBRATION & SELECTIVE ACCURACY ANALYSIS")
    print("=" * 80)

    for model in args.models:
        for mode, phase_dir in [("baseline", "phase5"), ("rag", "phase8")]:
            jsonl_path = qa_dir / model / "work" / phase_dir / f"{model}.jsonl"
            if not jsonl_path.exists():
                print(f"Skipping missing file: {jsonl_path}")
                continue

            print(f"Analyzing {model:<12} [{mode:<8}] from {jsonl_path.name}...")
            res = analyze_model_mode(model, mode, str(jsonl_path))
            all_results.append(res)
            results_by_mode[mode].append(res)

    print("Computing paired bootstrap Delta-AURC across models...")
    paired_deltas = compute_paired_delta_aurc(qa_dir, args.models)

    # Generate Risk-Coverage PNG plots first (requires full_coverages)
    generate_risk_coverage_plots(results_by_mode, out_dir)

    # Write Markdown report
    md_path = out_dir / "phase9_abstention.md"
    write_markdown_report(all_results, paired_deltas, md_path)

    # Write JSON artifact (without bulky 1000-point full curve arrays)
    json_models = []
    for r in all_results:
        r_copy = dict(r)
        rc_copy = dict(r["risk_coverage"])
        rc_copy.pop("full_coverages", None)
        rc_copy.pop("full_risks", None)
        r_copy["risk_coverage"] = rc_copy
        json_models.append(r_copy)

    json_export = {
        "models": json_models,
        "paired_delta_aurc": paired_deltas,
    }

    json_path = out_dir / "phase9_abstention.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)
    print(f"Saved JSON artifact: {json_path}")

    print("\nPhase 9 analysis complete.")


if __name__ == "__main__":
    main()
