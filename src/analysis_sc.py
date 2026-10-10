"""Analysis script for MedPsy-4B Self-Consistency (SC) Validation Outputs.

Evaluates downloaded validation outputs from kaggle_runner/runner.py (SC_MODE=True):
1. Greedy accuracy vs majority-vote accuracy with bootstrap 95% CIs (MedQA, PubMedQA).
2. AUROC of agreement for predicting correctness.
3. ECE (Expected Calibration Error) of agreement as a confidence score.
4. Abstention Precision, Recall, F1, and False-Abstention Rate on unanswerable_v2 val
   across agreement thresholds 0.2–1.0, combined with the truncation rule.
5. Breakdown across all 6 unanswerable_v2 categories + 2 control categories.
6. Paired comparison for fabricated_drug and fabricated_disease pairs (same stem: fake vs real).
7. Risk–coverage plots (selective classification) saved to outputs/sc_val/analysis/.

Usage:
  python -m src.analysis_sc
  python -m src.analysis_sc --sc_dir outputs/sc_val/medpsy-4b --out_dir outputs/sc_val/analysis
"""

import argparse
import collections
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Statistics Helpers (Self-Contained, Zero External Dependency) ──────────────

def bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute percentile bootstrap confidence interval for the sample mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    low_idx = int((alpha / 2.0) * n_boot)
    high_idx = int((1.0 - alpha / 2.0) * n_boot)
    return (round(means[low_idx], 4), round(means[min(high_idx, n_boot - 1)], 4))


def bootstrap_paired_diff_ci(
    vals_a: List[float],
    vals_b: List[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute paired bootstrap CI for mean(vals_a) - mean(vals_b)."""
    assert len(vals_a) == len(vals_b)
    diffs = [a - b for a, b in zip(vals_a, vals_b)]
    return bootstrap_ci(diffs, n_boot=n_boot, seed=seed)


def mcnemar_exact_p_value(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar test p-value given discordant counts b and c.
    
    b: model A correct, model B incorrect
    c: model A incorrect, model B correct
    """
    total = b + c
    if total == 0:
        return 1.0
    k = min(b, c)
    # Sum binomial probabilities P(X <= k) for X ~ Binomial(total, 0.5)
    cum = sum(math.comb(total, i) * (0.5 ** total) for i in range(k + 1))
    p = min(1.0, 2.0 * cum)
    return round(p, 6)


def wilcoxon_signed_rank_p(diffs: List[float]) -> float:
    """Approximate two-sided p-value for Wilcoxon signed-rank test on paired differences."""
    non_zero = [d for d in diffs if d != 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    if n < 5:
        # Simple sign test for tiny sample
        k = sum(1 for d in non_zero if d > 0)
        return mcnemar_exact_p_value(k, n - k)

    # Rank absolute differences with average ranks for ties
    abs_diffs = [abs(d) for d in non_zero]
    indices = sorted(range(n), key=lambda i: abs_diffs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[indices[j]] == abs_diffs[indices[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indices[k]] = avg_rank
        i = j

    w_pos = sum(ranks[i] for i in range(n) if non_zero[i] > 0)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return 1.0
    z = (w_pos - mean_w) / math.sqrt(var_w)
    # Standard normal CDF approximation
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return round(max(0.0, min(1.0, p)), 6)


def compute_auroc(y_true: List[int], scores: List[float]) -> float:
    """Calculate exact AUROC via Mann-Whitney U statistic with tie handling."""
    n1 = sum(y_true)
    n0 = len(y_true) - n1
    if n0 == 0 or n1 == 0:
        return 0.5

    n = len(scores)
    sorted_indices = sorted(range(n), key=lambda idx: scores[idx])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and scores[sorted_indices[j]] == scores[sorted_indices[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j

    rank_sum_pos = sum(ranks[idx] for idx in range(n) if y_true[idx] == 1)
    u = rank_sum_pos - (n1 * (n1 + 1)) / 2.0
    auroc = u / (n0 * n1)
    return round(float(auroc), 4)


def compute_ece(
    y_true: List[int],
    confidences: List[float],
    n_bins: int = 5,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Calculate Expected Calibration Error (ECE) and return per-bin diagnostics."""
    if not y_true:
        return (0.0, [])

    # Group into discrete bins based on unique confidence levels or equal width
    unique_confs = sorted(set(confidences))
    bin_details = []
    total_samples = len(y_true)
    ece = 0.0

    if len(unique_confs) <= n_bins:
        # Use discrete values directly
        for val in unique_confs:
            items = [(y, c) for y, c in zip(y_true, confidences) if c == val]
            if not items:
                continue
            cnt = len(items)
            acc = sum(y for y, _ in items) / cnt
            mean_conf = val
            weight = cnt / total_samples
            err = abs(acc - mean_conf)
            ece += weight * err
            bin_details.append({
                "bin": f"{val:.2f}",
                "count": cnt,
                "confidence": round(mean_conf, 4),
                "accuracy": round(acc, 4),
                "abs_error": round(err, 4),
            })
    else:
        # Equal width bins in [0, 1]
        bin_width = 1.0 / n_bins
        for b in range(n_bins):
            b_low = b * bin_width
            b_high = (b + 1) * bin_width
            if b == n_bins - 1:
                items = [(y, c) for y, c in zip(y_true, confidences) if b_low <= c <= b_high]
            else:
                items = [(y, c) for y, c in zip(y_true, confidences) if b_low <= c < b_high]
            if not items:
                continue
            cnt = len(items)
            acc = sum(y for y, _ in items) / cnt
            mean_conf = sum(c for _, c in items) / cnt
            weight = cnt / total_samples
            err = abs(acc - mean_conf)
            ece += weight * err
            bin_details.append({
                "bin": f"[{b_low:.2f}, {b_high:.2f}]",
                "count": cnt,
                "confidence": round(mean_conf, 4),
                "accuracy": round(acc, 4),
                "abs_error": round(err, 4),
            })

    return round(float(ece), 4), bin_details


# ── Data Loading Helpers ───────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Safely load a JSONL file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_greedy_map(path: Path, dataset_name: str) -> Dict[str, Dict[str, Any]]:
    """Load existing greedy MedPsy validation outputs keyed by id."""
    rows = load_jsonl(path)
    res = {}
    for r in rows:
        if r.get("dataset") == dataset_name or dataset_name in r.get("id", ""):
            res[r["id"]] = r
    return res


# ── Analysis Components ───────────────────────────────────────────────────────

def analyze_accuracy_comparison(
    sc_rows: List[Dict[str, Any]],
    greedy_map: Dict[str, Dict[str, Any]],
    dataset_name: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare greedy accuracy vs SC majority-vote accuracy with bootstrap 95% CIs.
    
    If rows contain 'greedy_answer' generated during SC mode, compares directly against that;
    otherwise falls back to greedy_map from prior runs.
    """
    matched = []
    n_own = 0
    n_fallback = 0
    n_missing = 0
    for r in sc_rows:
        qid = r["id"]
        gold = r.get("gold")
        if not gold:
            continue

        # Prefer paired greedy_answer inside row if present; else fallback to greedy_map
        if "greedy_answer" in r and r["greedy_answer"] is not None:
            g_pred = r["greedy_answer"]
            n_own += 1
        elif greedy_map and qid in greedy_map:
            g_row = greedy_map.get(qid)
            g_pred = g_row.get("pred") if g_row else None
            if g_pred is not None:
                n_fallback += 1
            else:
                n_missing += 1
        else:
            g_pred = None
            n_missing += 1

        sc_pred = r.get("majority_answer")

        g_corr = 1.0 if (g_pred is not None and g_pred == gold) else 0.0
        sc_corr = 1.0 if (sc_pred is not None and sc_pred == gold) else 0.0
        agree = float(r.get("agreement", 0.0))

        matched.append({
            "id": qid,
            "gold": gold,
            "greedy_pred": g_pred,
            "sc_pred": sc_pred,
            "greedy_corr": g_corr,
            "sc_corr": sc_corr,
            "agreement": agree,
        })

    n = len(matched)
    if n == 0:
        return {"n": 0}

    g_corrs = [m["greedy_corr"] for m in matched]
    sc_corrs = [m["sc_corr"] for m in matched]
    agreements = [m["agreement"] for m in matched]

    g_acc = sum(g_corrs) / n
    sc_acc = sum(sc_corrs) / n
    diff_acc = sc_acc - g_acc

    g_ci = bootstrap_ci(g_corrs, n_boot=n_boot, seed=seed)
    sc_ci = bootstrap_ci(sc_corrs, n_boot=n_boot, seed=seed)
    diff_ci = bootstrap_paired_diff_ci(sc_corrs, g_corrs, n_boot=n_boot, seed=seed)

    # Discordant pairs for McNemar test
    b = sum(1 for m in matched if m["sc_corr"] == 1.0 and m["greedy_corr"] == 0.0)
    c = sum(1 for m in matched if m["sc_corr"] == 0.0 and m["greedy_corr"] == 1.0)
    p_mcnemar = mcnemar_exact_p_value(b, c)

    # AUROC of agreement predicting correctness
    auroc = compute_auroc([int(c) for c in sc_corrs], agreements)
    rng = random.Random(seed)
    boot_aurocs = []
    for _ in range(n_boot):
        sample = rng.choices(matched, k=n)
        b_y = [int(m["sc_corr"]) for m in sample]
        b_s = [m["agreement"] for m in sample]
        boot_aurocs.append(compute_auroc(b_y, b_s))
    boot_aurocs.sort()
    auroc_ci = (round(boot_aurocs[int(0.025 * n_boot)], 4), round(boot_aurocs[int(0.975 * n_boot)], 4))

    # ECE of agreement
    ece, cal_bins = compute_ece([int(c) for c in sc_corrs], agreements)

    if n_own == n and n > 0:
        greedy_source = "row['greedy_answer'] (paired SC pass)"
    elif n_fallback == n and n > 0:
        greedy_source = "fallback reference file"
    elif n_own > 0 and n_fallback > 0:
        greedy_source = f"hybrid: row['greedy_answer'] ({n_own}/{n}) + fallback reference ({n_fallback}/{n})"
    elif n_own > 0:
        greedy_source = f"row['greedy_answer'] ({n_own}/{n}, {n_missing} missing)"
    else:
        greedy_source = "none (all missing)"

    return {
        "dataset": dataset_name,
        "n": n,
        "greedy_acc": round(g_acc, 4),
        "greedy_ci": g_ci,
        "greedy_source": greedy_source,
        "n_own": n_own,
        "n_fallback": n_fallback,
        "sc_acc": round(sc_acc, 4),
        "sc_ci": sc_ci,
        "diff_acc": round(diff_acc, 4),
        "diff_ci": diff_ci,
        "sc_wins_b": b,
        "greedy_wins_c": c,
        "mcnemar_p": p_mcnemar,
        "auroc": auroc,
        "auroc_ci": auroc_ci,
        "ece": ece,
        "cal_bins": cal_bins,
        "matched": matched,
    }


# ── Abstention Rule Evaluation (Side-by-Side across Tau 0.2–1.0) ───────────────

def compute_rule_metrics(
    unans_rows: List[Dict[str, Any]],
    pred_fn,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute TP, FP, FN, TN, Precision, Recall, F1, False-Abstention Rate on controls, and Coverage."""
    total = len(unans_rows)
    if total == 0:
        return {}

    tp = sum(1 for r in unans_rows if not r.get("control", False) and pred_fn(r))
    fp = sum(1 for r in unans_rows if r.get("control", False) and pred_fn(r))
    fn = sum(1 for r in unans_rows if not r.get("control", False) and not pred_fn(r))
    tn = sum(1 for r in unans_rows if r.get("control", False) and not pred_fn(r))

    n_controls = fp + tn
    n_unans = tp + fn

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    far = fp / n_controls if n_controls > 0 else 0.0
    cov = (tn + fn) / total

    # Bootstrap CIs for Precision, Recall, F1, and FAR
    rng = random.Random(seed)
    boot_p, boot_r, boot_f1, boot_far = [], [], [], []
    for _ in range(n_boot):
        sample = rng.choices(unans_rows, k=total)
        b_tp = sum(1 for r in sample if not r.get("control", False) and pred_fn(r))
        b_fp = sum(1 for r in sample if r.get("control", False) and pred_fn(r))
        b_fn = sum(1 for r in sample if not r.get("control", False) and not pred_fn(r))
        b_tn = sum(1 for r in sample if r.get("control", False) and not pred_fn(r))
        bp = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        br = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        bf1 = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0.0
        bfar = b_fp / (b_fp + b_tn) if (b_fp + b_tn) > 0 else 0.0
        boot_p.append(bp)
        boot_r.append(br)
        boot_f1.append(bf1)
        boot_far.append(bfar)

    boot_p.sort()
    boot_r.sort()
    boot_f1.sort()
    boot_far.sort()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "precision_ci": (round(boot_p[int(0.025 * n_boot)], 4), round(boot_p[int(0.975 * n_boot)], 4)),
        "recall": round(rec, 4),
        "recall_ci": (round(boot_r[int(0.025 * n_boot)], 4), round(boot_r[int(0.975 * n_boot)], 4)),
        "f1": round(f1, 4),
        "f1_ci": (round(boot_f1[int(0.025 * n_boot)], 4), round(boot_f1[int(0.975 * n_boot)], 4)),
        "far_controls": round(far, 4),
        "far_controls_ci": (round(boot_far[int(0.025 * n_boot)], 4), round(boot_far[int(0.975 * n_boot)], 4)),
        "coverage": round(cov, 4),
    }


def get_rule_pred_fn(rule_key: str, tau: float):
    """Factory creating the decision predicate for a given rule key and agreement threshold tau."""
    if rule_key == "current":
        # (a) current rule: agreement < tau OR any truncated OR any insufficient
        def _pred_a(r: Dict[str, Any]) -> bool:
            agree = float(r.get("agreement", 0.0))
            truncs = r.get("truncations", [])
            insufs = r.get("insufficient_info", [])
            any_trunc = any(truncs) if isinstance(truncs, list) else bool(truncs)
            any_insuf = any(insufs) if isinstance(insufs, list) else bool(insufs)
            return (agree < tau) or any_trunc or any_insuf
        return _pred_a

    elif rule_key == "fraction":
        # (b) fraction rule: agreement < tau OR truncation_rate >= 0.6 OR insufficient_rate >= 0.6
        def _pred_b(r: Dict[str, Any]) -> bool:
            agree = float(r.get("agreement", 0.0))
            truncs = r.get("truncations", [])
            insufs = r.get("insufficient_info", [])
            k_t = len(truncs) if isinstance(truncs, list) and len(truncs) > 0 else 5
            k_i = len(insufs) if isinstance(insufs, list) and len(insufs) > 0 else 5
            t_rate = (sum(truncs) / k_t) if isinstance(truncs, list) else (1.0 if truncs else 0.0)
            i_rate = (sum(insufs) / k_i) if isinstance(insufs, list) else (1.0 if insufs else 0.0)
            return (agree < tau) or (t_rate >= 0.6) or (i_rate >= 0.6)
        return _pred_b

    elif rule_key in ("agreement_only", "agree_only"):
        # (c) agreement-only: agreement < tau OR majority_answer == "ABSTAIN"
        def _pred_c(r: Dict[str, Any]) -> bool:
            agree = float(r.get("agreement", 0.0))
            return (agree < tau) or (r.get("majority_answer") == "ABSTAIN")
        return _pred_c

    else:
        raise ValueError(f"Unknown rule_key: {rule_key}")


def analyze_abstention_thresholds(
    unans_rows: List[Dict[str, Any]],
    thresholds: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate abstention rules side by side across agreement thresholds 0.2–1.0:
    (a) current rule: agreement < tau OR any truncated OR any insufficient
    (b) fraction rule: agreement < tau OR truncation_rate >= 0.6 OR insufficient_rate >= 0.6
    (c) agreement-only: agreement < tau OR majority_answer == "ABSTAIN"
    
    Reports F1, recall, false-abstention rate on controls, and coverage for each,
    and marks the best rule by F1 on validation.
    """
    total = len(unans_rows)
    if total == 0:
        return {"side_by_side": [], "all_evaluations": [], "best_rule": None, "by_rule": {}}

    rules_meta = [
        ("current", "(a) Current Rule", "agree < tau OR any trunc OR any insuf"),
        ("fraction", "(b) Fraction Rule", "agree < tau OR trunc_rate >= 0.6 OR insuf_rate >= 0.6"),
        ("agreement_only", "(c) Agreement-Only", "agree < tau OR majority == 'ABSTAIN'"),
    ]

    all_evaluations = []
    by_rule = collections.defaultdict(list)
    side_by_side = []

    for tau in thresholds:
        sbs_row = {"threshold": tau}
        for r_key, r_name, r_desc in rules_meta:
            fn = get_rule_pred_fn(r_key, tau)
            m = compute_rule_metrics(unans_rows, fn, n_boot=n_boot, seed=seed)
            m_eval = {
                "rule_key": r_key,
                "rule_name": r_name,
                "rule_desc": r_desc,
                "threshold": tau,
                **m,
            }
            all_evaluations.append(m_eval)
            by_rule[r_key].append(m_eval)
            sbs_row[r_key] = m_eval

        # Find best rule at this specific threshold
        best_at_tau = max(
            [sbs_row["current"], sbs_row["fraction"], sbs_row["agreement_only"]],
            key=lambda x: (x["f1"], -x["far_controls"], x["coverage"]),
        )
        sbs_row["best_at_tau"] = best_at_tau["rule_name"]
        side_by_side.append(sbs_row)

    # Globally best rule by F1 across all rules and thresholds on validation
    best_overall = max(
        all_evaluations,
        key=lambda x: (x["f1"], -x["far_controls"], x["coverage"]),
    )
    for ev in all_evaluations:
        ev["is_best"] = (ev is best_overall)

    return {
        "side_by_side": side_by_side,
        "by_rule": dict(by_rule),
        "all_evaluations": all_evaluations,
        "best_rule": best_overall,
    }


def analyze_unans_categories(
    unans_rows: List[Dict[str, Any]],
    best_rule_info: Optional[Dict[str, Any]] = None,
    rule_key_default: str = "fraction",
    tau_default: float = 0.8,
) -> List[Dict[str, Any]]:
    """Breakdown across all 6 unanswerable categories and 2 control categories
    evaluated under the best rule on validation (or defaults).
    """
    rule_key = best_rule_info["rule_key"] if best_rule_info else rule_key_default
    tau = best_rule_info["threshold"] if best_rule_info else tau_default
    pred_fn = get_rule_pred_fn(rule_key, tau)

    groups = collections.defaultdict(list)
    for r in unans_rows:
        cat = r.get("category", "unknown")
        is_ctrl = bool(r.get("control", False))
        groups[(cat, is_ctrl)].append(r)

    rows_summary = []
    for (cat, is_ctrl), items in sorted(groups.items(), key=lambda x: (not x[0][1], x[0][0])):
        n = len(items)
        agrees = [float(r.get("agreement", 0.0)) for r in items]
        mean_agree = sum(agrees) / n if n else 0.0
        trunc_cnt = sum(1 for r in items if any(r.get("truncations", [])))
        insuf_cnt = sum(1 for r in items if any(r.get("insufficient_info", [])))

        # Abstention under active rule
        abst_cnt = sum(1 for r in items if pred_fn(r))

        acc = None
        if is_ctrl:
            corr_cnt = sum(1 for r in items if r.get("majority_answer") is not None
                           and r.get("majority_answer") == r.get("gold"))
            acc = round(corr_cnt / n, 4)

        rows_summary.append({
            "category": cat,
            "is_control": is_ctrl,
            "type": "Control" if is_ctrl else "Unanswerable",
            "n": n,
            "mean_agreement": round(mean_agree, 4),
            "truncation_pct": round(100.0 * trunc_cnt / n, 1),
            "insufficient_info_pct": round(100.0 * insuf_cnt / n, 1),
            "abstained_pct": round(100.0 * abst_cnt / n, 1),
            "accuracy": acc,
        })

    return rows_summary


def analyze_paired_fabricated_comparison(
    unans_rows: List[Dict[str, Any]],
    best_rule_info: Optional[Dict[str, Any]] = None,
    rule_key_default: str = "fraction",
    tau_default: float = 0.8,
) -> Dict[str, Any]:
    """Perform paired comparison (same stem, fake vs real) for fabricated items
    evaluated under the best rule on validation (or defaults).
    """
    rule_key = best_rule_info["rule_key"] if best_rule_info else rule_key_default
    tau = best_rule_info["threshold"] if best_rule_info else tau_default
    pred_fn = get_rule_pred_fn(rule_key, tau)

    pair_dict = collections.defaultdict(dict)
    for r in unans_rows:
        sid = r.get("source_id", "")
        m = re.match(r"v2-(drug|disease)-(\d+)-(unans|ctrl)", sid)
        if m:
            domain, idx_str, kind = m.group(1), m.group(2), m.group(3)
            key = (domain, int(idx_str))
            pair_dict[key][kind] = r

    results_by_domain = {}
    for dom in ["drug", "disease", "combined"]:
        pairs = []
        for (d, idx), sub in pair_dict.items():
            if dom != "combined" and d != dom:
                continue
            if "unans" in sub and "ctrl" in sub:
                pairs.append((sub["unans"], sub["ctrl"]))

        if not pairs:
            continue

        n_pairs = len(pairs)
        fake_agrees = [float(p[0].get("agreement", 0.0)) for p in pairs]
        ctrl_agrees = [float(p[1].get("agreement", 0.0)) for p in pairs]
        diffs = [c - f for f, c in zip(fake_agrees, ctrl_agrees)]

        mean_fake_agree = sum(fake_agrees) / n_pairs
        mean_ctrl_agree = sum(ctrl_agrees) / n_pairs
        mean_diff = sum(diffs) / n_pairs
        p_wilcoxon = wilcoxon_signed_rank_p(diffs)

        fake_absts = [pred_fn(p[0]) for p in pairs]
        ctrl_absts = [pred_fn(p[1]) for p in pairs]

        fake_abst_rate = sum(fake_absts) / n_pairs
        ctrl_far = sum(ctrl_absts) / n_pairs

        # Contingency
        both_abst = sum(1 for f, c in zip(fake_absts, ctrl_absts) if f and c)
        selective_win = sum(1 for f, c in zip(fake_absts, ctrl_absts) if f and not c)
        both_ans = sum(1 for f, c in zip(fake_absts, ctrl_absts) if not f and not c)
        invert_err = sum(1 for f, c in zip(fake_absts, ctrl_absts) if not f and c)

        results_by_domain[dom] = {
            "n_pairs": n_pairs,
            "mean_fake_agree": round(mean_fake_agree, 4),
            "mean_ctrl_agree": round(mean_ctrl_agree, 4),
            "mean_diff_agree": round(mean_diff, 4),
            "wilcoxon_p": p_wilcoxon,
            "fake_abst_rate": round(fake_abst_rate, 4),
            "ctrl_false_abst_rate": round(ctrl_far, 4),
            "both_abstained": both_abst,
            "selective_win": selective_win,
            "both_answered": both_ans,
            "inverted_error": invert_err,
        }

    return results_by_domain


# ── Plotting Helpers ───────────────────────────────────────────────────────────

def plot_risk_coverage(
    eval_medqa: Optional[Dict[str, Any]],
    eval_pubmedqa: Optional[Dict[str, Any]],
    out_dir: Path,
):
    """Generate risk–coverage (selective classification) plot and save PNG/SVG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping risk-coverage plot.")
        return

    datasets_to_plot = []
    if eval_medqa and eval_medqa.get("n", 0) > 0:
        datasets_to_plot.append(("MedQA Validation", eval_medqa))
    if eval_pubmedqa and eval_pubmedqa.get("n", 0) > 0:
        datasets_to_plot.append(("PubMedQA Validation", eval_pubmedqa))

    if not datasets_to_plot:
        return

    fig, axes = plt.subplots(1, len(datasets_to_plot), figsize=(6.5 * len(datasets_to_plot), 5.5), squeeze=False)
    colors = ["#2b5c8f", "#107e3e"]

    for ax_idx, (ds_label, res) in enumerate(datasets_to_plot):
        ax = axes[0][ax_idx]
        matched = res["matched"]
        # Sort questions by agreement descending
        sorted_m = sorted(matched, key=lambda x: x["agreement"], reverse=True)
        n = len(sorted_m)

        coverages = []
        risks = []
        accuracies = []

        # Step through coverage percentages
        for k in range(1, n + 1):
            sub = sorted_m[:k]
            acc = sum(m["sc_corr"] for m in sub) / k
            risk = 1.0 - acc
            cov = k / n
            coverages.append(cov)
            risks.append(risk)
            accuracies.append(acc)

        c = colors[ax_idx % len(colors)]
        ax.plot(coverages, accuracies, label="Selective Accuracy", color=c, lw=2.5)
        ax.plot(coverages, risks, label="Selective Risk (Error)", color="#d9534f", lw=2.0, linestyle="--")

        # Baseline horizontal line at full coverage
        base_acc = res["sc_acc"]
        ax.axhline(base_acc, color=c, alpha=0.4, linestyle=":", label=f"Full Coverage Acc ({base_acc:.1%})")

        ax.set_title(f"{ds_label} (N={n})", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Coverage (Proportion Answered by Agreement)", fontsize=11)
        ax.set_ylabel("Rate", fontsize=11)
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "risk_coverage.png"
    svg_path = out_dir / "risk_coverage.svg"
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()
    print(f"Saved risk-coverage plots to:\n  - {png_path}\n  - {svg_path}")


# ── Markdown Report Generation ─────────────────────────────────────────────────

def generate_markdown_report(
    eval_medqa: Optional[Dict[str, Any]],
    eval_pubmedqa: Optional[Dict[str, Any]],
    abst_info: Dict[str, Any],
    cat_res: List[Dict[str, Any]],
    paired_res: Dict[str, Any],
    out_dir: Path,
):
    """Generate Markdown tables and executive summary report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# MedPsy-4B Self-Consistency Validation Report\n")
    lines.append("Evaluates self-consistency (SC, $K=5$, $T=0.7$, Top-$p=0.95$) on validation splits:\n")
    lines.append("- **MedQA Validation**: 200 questions, baseline reasoning mode.")
    lines.append("- **PubMedQA Validation**: 200 questions, +abstract context reasoning mode.")
    lines.append("- **Unanswerable Challenge v2 Validation**: 260 questions (200 unanswerable, 60 controls).\n")

    # ── Table 1: Accuracy comparison ──
    lines.append("## 1. Accuracy: Greedy vs. Self-Consistency Majority Vote\n")
    for ds_res in [eval_medqa, eval_pubmedqa]:
        if ds_res and ds_res.get("n", 0) > 0:
            lines.append(f"- **{ds_res['dataset']} Greedy Baseline Source**: {ds_res.get('greedy_source')}")
    lines.append("")
    lines.append("| Dataset | N | Greedy Acc [95% CI] | SC Maj-Vote Acc [95% CI] | Δ Acc (pts) [95% CI] | SC Wins / Greedy Wins | McNemar $p$-value | AUROC [95% CI] | ECE |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for ds_res in [eval_medqa, eval_pubmedqa]:
        if not ds_res or ds_res.get("n", 0) == 0:
            continue
        g_ci_str = f"[{ds_res['greedy_ci'][0]:.1%}, {ds_res['greedy_ci'][1]:.1%}]"
        sc_ci_str = f"[{ds_res['sc_ci'][0]:.1%}, {ds_res['sc_ci'][1]:.1%}]"
        diff_ci_str = f"[{ds_res['diff_ci'][0]:+.1%}, {ds_res['diff_ci'][1]:+.1%}]"
        auroc_str = f"{ds_res['auroc']:.4f} [{ds_res['auroc_ci'][0]:.3f}, {ds_res['auroc_ci'][1]:.3f}]"
        lines.append(
            f"| **{ds_res['dataset'].upper()}** | {ds_res['n']} | {ds_res['greedy_acc']:.1%} {g_ci_str} | "
            f"**{ds_res['sc_acc']:.1%}** {sc_ci_str} | **{ds_res['diff_acc']:+.1%}** {diff_ci_str} | "
            f"{ds_res['sc_wins_b']} / {ds_res['greedy_wins_c']} | $p={ds_res['mcnemar_p']:.4f}$ | "
            f"{auroc_str} | {ds_res['ece']:.4f} |"
        )
    lines.append("")

    # ── Table 2: Side-by-side Abstention rules comparison ──
    best_rule = abst_info.get("best_rule")
    side_by_side = abst_info.get("side_by_side", [])
    all_evals = abst_info.get("all_evaluations", [])

    if side_by_side:
        lines.append("## 2. Abstention Performance on Unanswerable v2 Validation Set\n")
        lines.append("Evaluates 3 decision rules side by side across agreement thresholds $\\tau \\in [0.2, 1.0]$:\n")
        lines.append("- **(a) Current Rule**: $\\text{Agreement} < \\tau$ OR any sample truncated OR any sample outputs `INSUFFICIENT INFORMATION`.")
        lines.append("- **(b) Fraction Rule**: $\\text{Agreement} < \\tau$ OR truncation rate $\\ge 0.6$ OR insufficient rate $\\ge 0.6$.")
        lines.append("- **(c) Agreement-Only Rule**: $\\text{Agreement} < \\tau$ OR $\\text{majority\\_answer} == \\text{'ABSTAIN'}$.\n")

        if best_rule:
            lines.append(
                f"> [!IMPORTANT]\n"
                f"> **★ BEST RULE ON VALIDATION (by F1)**: **{best_rule['rule_name']}** at **$\\tau = {best_rule['threshold']:.1f}$**  \n"
                f"> - **F1 Score**: **{best_rule['f1']:.3f}** [{best_rule['f1_ci'][0]:.3f}, {best_rule['f1_ci'][1]:.3f}]  \n"
                f"> - **Recall**: **{best_rule['recall']:.1%}** [{best_rule['recall_ci'][0]:.1%}, {best_rule['recall_ci'][1]:.1%}]  \n"
                f"> - **False Abstention Rate on Controls**: **{best_rule['far_controls']:.1%}** [{best_rule['far_controls_ci'][0]:.1%}, {best_rule['far_controls_ci'][1]:.1%}]  \n"
                f"> - **Coverage**: **{best_rule['coverage']:.1%}**  \n"
            )

        lines.append("### Side-by-Side Comparison Matrix\n")
        lines.append("| Agreement $\\tau$ | (a) Current Rule<br>F1 (Rec / FAR / Cov) | (b) Fraction Rule<br>F1 (Rec / FAR / Cov) | (c) Agreement-Only<br>F1 (Rec / FAR / Cov) | Best Rule at $\\tau$ |")
        lines.append("|---|---|---|---|---|")
        for s in side_by_side:
            ca = s["current"]
            cb = s["fraction"]
            cc = s["agreement_only"]
            str_a = f"**{ca['f1']:.3f}** ({ca['recall']:.1%} / {ca['far_controls']:.1%} / {ca['coverage']:.1%})"
            str_b = f"**{cb['f1']:.3f}** ({cb['recall']:.1%} / {cb['far_controls']:.1%} / {cb['coverage']:.1%})"
            str_c = f"**{cc['f1']:.3f}** ({cc['recall']:.1%} / {cc['far_controls']:.1%} / {cc['coverage']:.1%})"
            lines.append(f"| **$\\tau = {s['threshold']:.1f}$** | {str_a} | {str_b} | {str_c} | **{s['best_at_tau']}** |")
        lines.append("")

        lines.append("### Comprehensive Rule Breakdown across Thresholds\n")
        lines.append("| Rule | $\\tau$ | Precision [95% CI] | Recall [95% CI] | False Abstention (Controls) [95% CI] | F1 Score [95% CI] | Coverage | Status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for ev in all_evals:
            p_ci = f"[{ev['precision_ci'][0]:.1%}, {ev['precision_ci'][1]:.1%}]"
            r_ci = f"[{ev['recall_ci'][0]:.1%}, {ev['recall_ci'][1]:.1%}]"
            far_ci = f"[{ev['far_controls_ci'][0]:.1%}, {ev['far_controls_ci'][1]:.1%}]"
            f1_ci = f"[{ev['f1_ci'][0]:.3f}, {ev['f1_ci'][1]:.3f}]"
            status = "★ **BEST RULE**" if ev.get("is_best") else "—"
            lines.append(
                f"| {ev['rule_name']} | **{ev['threshold']:.1f}** | {ev['precision']:.1%} {p_ci} | "
                f"{ev['recall']:.1%} {r_ci} | {ev['far_controls']:.1%} {far_ci} | "
                f"**{ev['f1']:.3f}** {f1_ci} | {ev['coverage']:.1%} | {status} |"
            )
        lines.append("")

    # ── Table 3: Breakdown by Category ──
    if cat_res:
        lines.append("## 3. Unanswerable v2 Breakdown by Category\n")
        active_desc = f"{best_rule['rule_name']} at $\\tau={best_rule['threshold']:.1f}$" if best_rule else "$\\tau=0.8$"
        lines.append(f"> Evaluated under validation winning rule: **{active_desc}**.\n")
        lines.append("| Category | Type | N | Mean Agreement | % Truncated | % 'INSUFFICIENT INFO' | % Abstained | Majority-Vote Acc (Controls) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in cat_res:
            acc_str = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "—"
            lines.append(
                f"| `{r['category']}` | **{r['type']}** | {r['n']} | {r['mean_agreement']:.2f} | "
                f"{r['truncation_pct']}% | {r['insufficient_info_pct']}% | {r['abstained_pct']}% | {acc_str} |"
            )
        lines.append("")

    # ── Table 4: Paired Comparison ──
    if paired_res:
        lines.append("## 4. Paired Comparison of Fabricated vs. Real Controls (Identical Stems)\n")
        lines.append("| Domain | Matched Pairs | Fake Mean Agreement | Control Mean Agreement | Δ Agreement | Wilcoxon $p$-value | Fake Abstention (Recall) | Control False Abstention | Selective Abstention Success (Fake Abst & Ctrl Ans) | Hallucinated on Fake (Both Ans) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for dom, sub in paired_res.items():
            lines.append(
                f"| **{dom.capitalize()}** | {sub['n_pairs']} | {sub['mean_fake_agree']:.2f} | "
                f"{sub['mean_ctrl_agree']:.2f} | **{sub['mean_diff_agree']:+.2f}** | $p={sub['wilcoxon_p']:.4f}$ | "
                f"{sub['fake_abst_rate']:.1%} | {sub['ctrl_false_abst_rate']:.1%} | "
                f"**{sub['selective_win']}/{sub['n_pairs']} ({100*sub['selective_win']/sub['n_pairs']:.1f}%)** | "
                f"{sub['both_answered']}/{sub['n_pairs']} ({100*sub['both_answered']/sub['n_pairs']:.1f}%) |"
            )
        lines.append("")

    report_path = out_dir / "sc_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved executive markdown report to: {report_path}")

    # Also save separate tabular files
    if all_evals:
        t_path = out_dir / "table_abstention_thresholds.md"
        with open(t_path, "w", encoding="utf-8") as f:
            f.write("# Abstention Rules Comparison across Thresholds (Unanswerable v2 Validation)\n\n")
            if best_rule:
                f.write(f"**Best Rule on Validation**: {best_rule['rule_name']} at tau={best_rule['threshold']:.1f} (F1 = {best_rule['f1']:.4f}, Recall = {best_rule['recall']:.1%}, FAR = {best_rule['far_controls']:.1%}, Coverage = {best_rule['coverage']:.1%})\n\n")
            f.write("| Rule | Threshold | Precision | Recall | False Abstention Rate | F1 Score | Coverage | Status |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in all_evals:
                st = "★ BEST" if r.get("is_best") else ""
                f.write(f"| {r['rule_name']} | {r['threshold']:.1f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['far_controls']:.4f} | {r['f1']:.4f} | {r['coverage']:.4f} | {st} |\n")

    if cat_res:
        t_path = out_dir / "table_unanswerable_categories.md"
        with open(t_path, "w", encoding="utf-8") as f:
            f.write("| Category | Type | N | Mean Agreement | Abstained % | Control Acc |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in cat_res:
                acc_s = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "N/A"
                f.write(f"| {r['category']} | {r['type']} | {r['n']} | {r['mean_agreement']:.4f} | {r['abstained_pct']}% | {acc_s} |\n")


# ── Main Entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze MedPsy-4B Self-Consistency Validation Outputs.")
    parser.add_argument("--sc_dir", type=str, default="outputs/sc_val/medpsy-4b",
                        help="Directory containing medqa.jsonl, pubmedqa.jsonl, and unanswerable_v2.jsonl")
    parser.add_argument("--greedy_medqa", type=str,
                        default="outputs/kaggle_qa/medpsy_val/work/phase5/medpsy-4b.jsonl",
                        help="Path to greedy MedPsy validation outputs for MedQA (fallback if greedy_answer missing)")
    parser.add_argument("--greedy_pubmedqa", type=str,
                        default="outputs/kaggle_qa/medpsy_val/work/phase10/medpsy-4b.jsonl",
                        help="Path to greedy MedPsy validation outputs for PubMedQA (fallback if greedy_answer missing)")
    parser.add_argument("--out_dir", type=str, default="outputs/sc_val/analysis",
                        help="Directory to save summary tables and plots")
    parser.add_argument("--bootstrap_n", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sc_dir = Path(args.sc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEDPSY-4B SELF-CONSISTENCY VALIDATION ANALYSIS")
    print(f"  SC directory     : {sc_dir}")
    print(f"  Output directory : {out_dir}")
    print("=" * 80)

    # 1. Load SC files
    medqa_sc = load_jsonl(sc_dir / "medqa.jsonl")
    pubmedqa_sc = load_jsonl(sc_dir / "pubmedqa.jsonl")
    unans_sc = load_jsonl(sc_dir / "unanswerable_v2.jsonl")

    print(f"Loaded SC rows: MedQA={len(medqa_sc)}, PubMedQA={len(pubmedqa_sc)}, Unanswerable={len(unans_sc)}")

    # 2. Load Greedy files (fallback only for rows without greedy_answer)
    medqa_missing_greedy = sum(1 for r in medqa_sc if r.get("greedy_answer") is None) if medqa_sc else 0
    pubmedqa_missing_greedy = sum(1 for r in pubmedqa_sc if r.get("greedy_answer") is None) if pubmedqa_sc else 0

    g_medqa_map = {}
    if medqa_missing_greedy > 0:
        g_medqa_map = load_greedy_map(Path(args.greedy_medqa), "medqa")
        print(f"Loaded fallback greedy reference for MedQA: {len(g_medqa_map)} rows from {args.greedy_medqa} ({medqa_missing_greedy} rows missing greedy_answer)")
    elif medqa_sc:
        print("MedQA: all rows contain 'greedy_answer'; fallback reference file not needed.")

    g_pubmedqa_map = {}
    if pubmedqa_missing_greedy > 0:
        g_pubmedqa_map = load_greedy_map(Path(args.greedy_pubmedqa), "pubmedqa")
        print(f"Loaded fallback greedy reference for PubMedQA: {len(g_pubmedqa_map)} rows from {args.greedy_pubmedqa} ({pubmedqa_missing_greedy} rows missing greedy_answer)")
    elif pubmedqa_sc:
        print("PubMedQA: all rows contain 'greedy_answer'; fallback reference file not needed.")

    # 3. Accuracy evaluations
    eval_medqa = None
    if medqa_sc:
        eval_medqa = analyze_accuracy_comparison(
            medqa_sc, g_medqa_map, "MedQA", n_boot=args.bootstrap_n, seed=args.seed
        )
        print(f"MedQA greedy baseline source: {eval_medqa['greedy_source']}")
        print(f"MedQA: Greedy Acc = {eval_medqa['greedy_acc']:.1%}, SC Maj-Vote Acc = {eval_medqa['sc_acc']:.1%} (Δ = {eval_medqa['diff_acc']:+.1%}, AUROC = {eval_medqa['auroc']:.4f}, ECE = {eval_medqa['ece']:.4f})")

    eval_pubmedqa = None
    if pubmedqa_sc:
        eval_pubmedqa = analyze_accuracy_comparison(
            pubmedqa_sc, g_pubmedqa_map, "PubMedQA", n_boot=args.bootstrap_n, seed=args.seed
        )
        print(f"PubMedQA greedy baseline source: {eval_pubmedqa['greedy_source']}")
        print(f"PubMedQA: Greedy Acc = {eval_pubmedqa['greedy_acc']:.1%}, SC Maj-Vote Acc = {eval_pubmedqa['sc_acc']:.1%} (Δ = {eval_pubmedqa['diff_acc']:+.1%}, AUROC = {eval_pubmedqa['auroc']:.4f}, ECE = {eval_pubmedqa['ece']:.4f})")

    # 4. Unanswerable evaluations across rules (a), (b), (c)
    abst_info = {}
    cat_res = []
    paired_res = {}
    if unans_sc:
        abst_info = analyze_abstention_thresholds(unans_sc, n_boot=args.bootstrap_n, seed=args.seed)
        best_rule = abst_info.get("best_rule")
        cat_res = analyze_unans_categories(unans_sc, best_rule_info=best_rule)
        paired_res = analyze_paired_fabricated_comparison(unans_sc, best_rule_info=best_rule)

        print(f"\nEvaluated {len(abst_info.get('all_evaluations', []))} rule-threshold combinations on {len(unans_sc)} unanswerable v2 items.")
        if best_rule:
            print(f"★ BEST RULE BY F1 ON VALIDATION: {best_rule['rule_name']} at tau={best_rule['threshold']:.1f}")
            print(f"   F1 = {best_rule['f1']:.4f} {best_rule['f1_ci']}, Recall = {best_rule['recall']:.1%}, FAR Controls = {best_rule['far_controls']:.1%}, Coverage = {best_rule['coverage']:.1%}\n")

    # 5. Risk-Coverage Plot
    plot_risk_coverage(eval_medqa, eval_pubmedqa, out_dir)

    # 6. Generate Markdown Report and Tables
    generate_markdown_report(eval_medqa, eval_pubmedqa, abst_info, cat_res, paired_res, out_dir)

    print("\nAnalysis complete! Review report at:", out_dir / "sc_validation_report.md")


if __name__ == "__main__":
    main()
