#!/usr/bin/env python3
"""src/analysis_budget_sweep.py — Analysis of Pair B Validation Token-Budget Sweep.

Evaluates:
  - Model pair: medpsy-4b vs qwen3-4b-thinking-2507
  - Data: MedQA validation split (N=100)
  - Condition: C1 (closed-book multi-choice QA), greedy decoding
  - Derived cap grid: 1024, 1536, 2048, 3072, 4096

Derivation Validity:
  This single-run evaluation of multiple caps is mathematically exact ONLY under
  greedy decoding (do_sample=False), where the generated sequence prefix up to
  any cap c <= 4096 is strictly invariant to future generation steps.

Outputs:
  - outputs/analysis/budget_pair_b.md
  - outputs/analysis/budget_pair_b.png
  - outputs/analysis/budget_pair_b.svg
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.stats import binom

from src.common import read_jsonl

CAPS = [1024, 1536, 2048, 3072, 4096]
MODEL_GENERIC = "qwen3-4b-thinking-2507"
MODEL_MEDICAL = "medpsy-4b"


def wilson_score_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% confidence
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * (n**2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def exact_mcnemar_test(b: int, c: int) -> float:
    """Exact two-sided McNemar test based on the binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    # Two-sided p-value
    k = min(b, c)
    p_val = 2.0 * float(binom.cdf(k, n, 0.5))
    return min(1.0, p_val)


def paired_bootstrap_ci(
    arr1: np.ndarray,
    arr2: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% paired bootstrap CI for mean(arr1) - mean(arr2)."""
    assert len(arr1) == len(arr2), "Mismatched array lengths for paired bootstrap"
    n = len(arr1)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    diff = arr1 - arr2
    # Resample question indices
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = np.mean(diff[idx], axis=1)
    lower = float(np.percentile(boot_diffs, 2.5))
    upper = float(np.percentile(boot_diffs, 97.5))
    return lower, upper


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, orig_idx in enumerate(sorted_indices):
        multiplier = m - rank
        raw_p = p_values[orig_idx]
        adj_p = min(1.0, raw_p * multiplier)
        cum_max = max(cum_max, adj_p)
        adjusted[orig_idx] = cum_max
    return adjusted


def is_finished_within_cap(row: Dict[str, Any], cap: int) -> bool:
    """A row counts as finished within cap c iff finished_naturally and n_generated_tokens <= c."""
    return bool(row.get("finished_naturally") and (row.get("n_generated_tokens", 999999) <= cap))


def is_strict_correct_at_cap(row: Dict[str, Any], cap: int) -> bool:
    """Strict at cap c = finished within c AND parsed AND correct."""
    return bool(is_finished_within_cap(row, cap) and row.get("parsed") and row.get("correct"))


def is_truncated_at_cap(row: Dict[str, Any], cap: int) -> bool:
    """Truncated at cap c = not finished within c."""
    return not is_finished_within_cap(row, cap)


def apply_budget_decision_rule(
    truncation_rates_qwen: Dict[int, float],
    threshold: float = 0.20,
    caps: List[int] = CAPS,
) -> Tuple[int, str]:
    """Apply pre-registered decision rule to choose test cap for Pair B.

    The cap for the final Pair B test run is the smallest cap in the grid at which
    Qwen3-4B-Thinking-2507's validation truncation is <= 20%.
    If none qualifies, use 4096 and report the residual truncation as a limitation.
    """
    for c in caps:
        rate = truncation_rates_qwen.get(c, 1.0)
        if rate <= threshold:
            msg = (
                f"Cap {c} SELECTED: Smallest cap in grid where Qwen3-4B-Thinking-2507 validation "
                f"truncation <= {threshold*100:.1f}% (observed {rate*100:.1f}%)."
            )
            return c, msg

    fallback_rate = truncation_rates_qwen.get(4096, 1.0)
    msg = (
        f"Cap 4096 FALLBACK SELECTED: No cap achieved truncation <= {threshold*100:.1f}%. "
        f"Residual truncation at 4096 tokens is {fallback_rate*100:.1f}%, which is reported "
        "as a methodological budget limitation, not as parity."
    )
    return 4096, msg


def check_medpsy_validation_consistency(sweep_medpsy_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Consistency check against existing greedy validation outputs in outputs/sc_val/medpsy-4b/."""
    sc_val_path = "outputs/sc_val/medpsy-4b/medqa.jsonl"
    if not os.path.exists(sc_val_path):
        return {
            "status": "skipped",
            "reason": f"File not found at {sc_val_path}",
        }

    try:
        sc_val_rows = read_jsonl(sc_val_path)
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"Failed reading {sc_val_path}: {exc}",
        }

    sc_val_dict = {r["id"]: r for r in sc_val_rows if "id" in r}

    # Filter sweep rows that finished naturally within 1,024 tokens
    eligible = [
        r for r in sweep_medpsy_rows
        if is_finished_within_cap(r, 1024) and r.get("id") in sc_val_dict
    ]

    if not eligible:
        return {
            "status": "skipped",
            "reason": "No overlapping rows that finished within 1,024 tokens found.",
        }

    matches = 0
    comparisons = []
    for r in eligible:
        qid = r["id"]
        sweep_pred = r.get("pred")
        sc_val_pred = sc_val_dict[qid].get("greedy_answer")
        is_match = (sweep_pred is not None) and (str(sweep_pred).strip().upper() == str(sc_val_pred).strip().upper())
        if is_match:
            matches += 1
        comparisons.append({
            "id": qid,
            "sweep_pred": sweep_pred,
            "sc_val_pred": sc_val_pred,
            "match": is_match,
        })

    agreement_rate = matches / len(eligible)
    return {
        "status": "success",
        "n_eligible": len(eligible),
        "matches": matches,
        "agreement_rate": agreement_rate,
        "sc_val_file": sc_val_path,
    }


def plot_budget_sweep(
    results_by_cap: Dict[int, Dict[str, Any]],
    chosen_cap: int,
    out_png: str,
    out_svg: str,
):
    """Plot strict accuracy and truncation rate vs token cap with 95% Wilson CI bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caps = sorted(results_by_cap.keys())
    med_acc = [results_by_cap[c]["medpsy"]["strict_acc"] for c in caps]
    med_acc_low = [results_by_cap[c]["medpsy"]["strict_acc_ci"][0] for c in caps]
    med_acc_high = [results_by_cap[c]["medpsy"]["strict_acc_ci"][1] for c in caps]

    qwen_acc = [results_by_cap[c]["qwen"]["strict_acc"] for c in caps]
    qwen_acc_low = [results_by_cap[c]["qwen"]["strict_acc_ci"][0] for c in caps]
    qwen_acc_high = [results_by_cap[c]["qwen"]["strict_acc_ci"][1] for c in caps]

    med_trunc = [results_by_cap[c]["medpsy"]["trunc_rate"] for c in caps]
    med_trunc_low = [results_by_cap[c]["medpsy"]["trunc_rate_ci"][0] for c in caps]
    med_trunc_high = [results_by_cap[c]["medpsy"]["trunc_rate_ci"][1] for c in caps]

    qwen_trunc = [results_by_cap[c]["qwen"]["trunc_rate"] for c in caps]
    qwen_trunc_low = [results_by_cap[c]["qwen"]["trunc_rate_ci"][0] for c in caps]
    qwen_trunc_high = [results_by_cap[c]["qwen"]["trunc_rate_ci"][1] for c in caps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Strict Accuracy
    ax1.plot(caps, med_acc, marker="o", color="#1f77b4", label="MedPsy-4B (Medical)", linewidth=2)
    ax1.fill_between(caps, med_acc_low, med_acc_high, color="#1f77b4", alpha=0.15)
    ax1.plot(caps, qwen_acc, marker="s", color="#d62728", label="Qwen3-4B-Thinking-2507 (Generic)", linewidth=2)
    ax1.fill_between(caps, qwen_acc_low, qwen_acc_high, color="#d62728", alpha=0.15)
    ax1.axvline(chosen_cap, color="green", linestyle="--", linewidth=1.5, label=f"Chosen Test Cap ({chosen_cap})")
    ax1.set_xlabel("Token Budget Cap (tokens)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Strict Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Strict Accuracy vs Token Budget Cap (C1 Validation N=100)", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xticks(caps)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Panel 2: Truncation Rate
    ax2.plot(caps, med_trunc, marker="o", color="#1f77b4", label="MedPsy-4B", linewidth=2)
    ax2.fill_between(caps, med_trunc_low, med_trunc_high, color="#1f77b4", alpha=0.15)
    ax2.plot(caps, qwen_trunc, marker="s", color="#d62728", label="Qwen3-4B-Thinking-2507", linewidth=2)
    ax2.fill_between(caps, qwen_trunc_low, qwen_trunc_high, color="#d62728", alpha=0.15)
    ax2.axhline(0.20, color="gray", linestyle=":", linewidth=1.5, label="Target Truncation <= 20%")
    ax2.axvline(chosen_cap, color="green", linestyle="--", linewidth=1.5, label=f"Chosen Test Cap ({chosen_cap})")
    ax2.set_xlabel("Token Budget Cap (tokens)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Truncation Rate", fontsize=11, fontweight="bold")
    ax2.set_title("Truncation Rate vs Token Budget Cap (C1 Validation N=100)", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xticks(caps)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Figures saved to {out_png} and {out_svg}")


def run_analysis(sweep_dir: str, out_md: str, out_png: str, out_svg: str):
    qwen_file = os.path.join(sweep_dir, f"{MODEL_GENERIC}.jsonl")
    medpsy_file = os.path.join(sweep_dir, f"{MODEL_MEDICAL}.jsonl")

    if not os.path.exists(qwen_file) or not os.path.exists(medpsy_file):
        raise FileNotFoundError(f"Sweep files not found in {sweep_dir}: {qwen_file}, {medpsy_file}")

    qwen_rows = read_jsonl(qwen_file)
    medpsy_rows = read_jsonl(medpsy_file)

    # Hard validation split assertions
    assert all(r.get("split") == "validation" for r in qwen_rows), "FATAL: Non-validation row in Qwen sweep data"
    assert all(r.get("split") == "validation" for r in medpsy_rows), "FATAL: Non-validation row in MedPsy sweep data"

    # Match rows by ID
    qwen_dict = {r["id"]: r for r in qwen_rows}
    medpsy_dict = {r["id"]: r for r in medpsy_rows}
    common_ids = [r["id"] for r in qwen_rows if r["id"] in medpsy_dict]
    assert len(common_ids) == len(qwen_rows) == len(medpsy_rows), "Mismatched question IDs between models"

    N = len(common_ids)

    # Collect token statistics per model across the full 4096 run
    qwen_tokens = np.array([qwen_dict[qid].get("n_generated_tokens", 0) for qid in common_ids])
    medpsy_tokens = np.array([medpsy_dict[qid].get("n_generated_tokens", 0) for qid in common_ids])

    # Per cap evaluations
    results_by_cap = {}
    p_values_for_holm = []
    test_descriptors = []

    for cap in CAPS:
        # MedPsy indicators
        med_fin = np.array([is_finished_within_cap(medpsy_dict[qid], cap) for qid in common_ids], dtype=bool)
        med_strict = np.array([is_strict_correct_at_cap(medpsy_dict[qid], cap) for qid in common_ids], dtype=bool)
        med_trunc = ~med_fin

        # Qwen indicators
        qwen_fin = np.array([is_finished_within_cap(qwen_dict[qid], cap) for qid in common_ids], dtype=bool)
        qwen_strict = np.array([is_strict_correct_at_cap(qwen_dict[qid], cap) for qid in common_ids], dtype=bool)
        qwen_trunc = ~qwen_fin

        # Metrics for MedPsy
        med_strict_acc = float(np.mean(med_strict))
        med_strict_ci = wilson_score_ci(int(np.sum(med_strict)), N)
        med_trunc_rate = float(np.mean(med_trunc))
        med_trunc_ci = wilson_score_ci(int(np.sum(med_trunc)), N)
        med_trunc_rows = [medpsy_dict[qid] for qid in common_ids if is_truncated_at_cap(medpsy_dict[qid], cap)]
        med_loop_share = (
            sum(1 for r in med_trunc_rows if r.get("loop_suspect")) / len(med_trunc_rows)
            if med_trunc_rows else 0.0
        )

        # Metrics for Qwen
        qwen_strict_acc = float(np.mean(qwen_strict))
        qwen_strict_ci = wilson_score_ci(int(np.sum(qwen_strict)), N)
        qwen_trunc_rate = float(np.mean(qwen_trunc))
        qwen_trunc_ci = wilson_score_ci(int(np.sum(qwen_trunc)), N)
        qwen_trunc_rows = [qwen_dict[qid] for qid in common_ids if is_truncated_at_cap(qwen_dict[qid], cap)]
        qwen_loop_share = (
            sum(1 for r in qwen_trunc_rows if r.get("loop_suspect")) / len(qwen_trunc_rows)
            if qwen_trunc_rows else 0.0
        )

        # Paired strict comparison (MedPsy vs Qwen)
        b_strict = int(np.sum(med_strict & ~qwen_strict))
        c_strict = int(np.sum(~med_strict & qwen_strict))
        p_strict = exact_mcnemar_test(b_strict, c_strict)
        strict_diff_ci = paired_bootstrap_ci(med_strict.astype(float), qwen_strict.astype(float))

        # Paired truncation comparison (MedPsy vs Qwen)
        b_trunc = int(np.sum(med_trunc & ~qwen_trunc))
        c_trunc = int(np.sum(~med_trunc & qwen_trunc))
        p_trunc = exact_mcnemar_test(b_trunc, c_trunc)
        trunc_diff_ci = paired_bootstrap_ci(med_trunc.astype(float), qwen_trunc.astype(float))

        # Descriptive subset: both finished within cap
        both_fin = med_fin & qwen_fin
        n_both_fin = int(np.sum(both_fin))
        if n_both_fin > 0:
            med_both_acc = float(np.mean([medpsy_dict[qid].get("correct", False) for i, qid in enumerate(common_ids) if both_fin[i]]))
            qwen_both_acc = float(np.mean([qwen_dict[qid].get("correct", False) for i, qid in enumerate(common_ids) if both_fin[i]]))
        else:
            med_both_acc = 0.0
            qwen_both_acc = 0.0

        results_by_cap[cap] = {
            "medpsy": {
                "strict_acc": med_strict_acc,
                "strict_acc_ci": med_strict_ci,
                "trunc_rate": med_trunc_rate,
                "trunc_rate_ci": med_trunc_ci,
                "loop_share": med_loop_share,
                "n_trunc": len(med_trunc_rows),
            },
            "qwen": {
                "strict_acc": qwen_strict_acc,
                "strict_acc_ci": qwen_strict_ci,
                "trunc_rate": qwen_trunc_rate,
                "trunc_rate_ci": qwen_trunc_ci,
                "loop_share": qwen_loop_share,
                "n_trunc": len(qwen_trunc_rows),
            },
            "comparison": {
                "b_strict": b_strict,
                "c_strict": c_strict,
                "p_strict_raw": p_strict,
                "strict_diff": med_strict_acc - qwen_strict_acc,
                "strict_diff_ci": strict_diff_ci,
                "b_trunc": b_trunc,
                "c_trunc": c_trunc,
                "p_trunc_raw": p_trunc,
                "trunc_diff": med_trunc_rate - qwen_trunc_rate,
                "trunc_diff_ci": trunc_diff_ci,
            },
            "descriptive_both_finished": {
                "n": n_both_fin,
                "pct": (n_both_fin / N) * 100.0,
                "medpsy_acc": med_both_acc,
                "qwen_acc": qwen_both_acc,
            },
        }

        # Accumulate for the single Holm family (10 tests: 5 strict + 5 truncation)
        p_values_for_holm.append(p_strict)
        test_descriptors.append((cap, "strict"))
        p_values_for_holm.append(p_trunc)
        test_descriptors.append((cap, "trunc"))

    # Apply single Holm family across all 10 tests
    holm_adj_p = holm_bonferroni(p_values_for_holm)
    for idx, (cap, metric) in enumerate(test_descriptors):
        if metric == "strict":
            results_by_cap[cap]["comparison"]["p_strict_holm"] = holm_adj_p[idx]
        else:
            results_by_cap[cap]["comparison"]["p_trunc_holm"] = holm_adj_p[idx]

    # Apply pre-registered decision rule
    qwen_trunc_rates = {c: results_by_cap[c]["qwen"]["trunc_rate"] for c in CAPS}
    chosen_cap, decision_rationale = apply_budget_decision_rule(qwen_trunc_rates)
    print(f"\nDECISION RULE RESULT: {decision_rationale}\n", flush=True)

    # MedPsy consistency check against sc_val
    consistency_res = check_medpsy_validation_consistency(medpsy_rows)

    # Plot
    plot_budget_sweep(results_by_cap, chosen_cap, out_png, out_svg)

    # Write Markdown Report
    lines = []
    lines.append("# Pair B Validation Token-Budget Sweep Report")
    lines.append("")
    lines.append(f"**Models:** `medpsy-4b` (Medical SLM) vs `qwen3-4b-thinking-2507` (Generic Reasoning Control)")
    lines.append(f"**Dataset:** MedQA validation split (first {N} questions in file order)")
    lines.append(f"**Condition:** Condition C1 (closed-book baseline)")
    lines.append(f"**Decoding:** Strictly greedy (`do_sample=False`), one run at `max_new_tokens=4096`")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Mathematical Derivation Validity:**")
    lines.append("> This one-run-gives-all-caps derivation is mathematically valid **ONLY for greedy decoding** (`do_sample=False`).")
    lines.append("> Under greedy autoregressive decoding, token selection at step $t$ is deterministic and independent of the horizon cap.")
    lines.append(r"> Thus, the prefix sequence generated at any cap $c \in \{1024, 1536, 2048, 3072\}$ is identical to the first $c$ tokens of the 4096-token run.")
    lines.append("")

    lines.append("## 1. Decision Rule Execution & Selected Test Cap")
    lines.append("")
    lines.append(rf"**Pre-Registered Decision Rule:** Select the smallest cap in $[1024, 1536, 2048, 3072, 4096]$ at which `qwen3-4b-thinking-2507` validation truncation $\le 20.0\%$. If none qualifies, fall back to 4096.")
    lines.append("")
    lines.append(f"- **Chosen Test Cap:** `{chosen_cap}` tokens")
    lines.append(f"- **Rule Outcome:** {decision_rationale}")
    lines.append(f"- **Equal Budget Application:** Both `medpsy-4b` and `qwen3-4b-thinking-2507` will receive cap `{chosen_cap}` in the final Pair B test run.")
    lines.append("")

    lines.append("## 2. Token Generation Summary Across Full Generation (Cap=4096)")
    lines.append("")
    lines.append("| Model | Median Tokens | p90 Tokens | Min Tokens | Max Tokens | Finished Naturally (%) |")
    lines.append("|---|---|---|---|---|---|")
    med_fin_total = sum(1 for r in medpsy_rows if r.get("finished_naturally")) / N * 100.0
    qwen_fin_total = sum(1 for r in qwen_rows if r.get("finished_naturally")) / N * 100.0
    lines.append(f"| `medpsy-4b` | {np.median(medpsy_tokens):.1f} | {np.percentile(medpsy_tokens, 90):.1f} | {np.min(medpsy_tokens)} | {np.max(medpsy_tokens)} | {med_fin_total:.1f}% |")
    lines.append(f"| `qwen3-4b-thinking-2507` | {np.median(qwen_tokens):.1f} | {np.percentile(qwen_tokens, 90):.1f} | {np.min(qwen_tokens)} | {np.max(qwen_tokens)} | {qwen_fin_total:.1f}% |")
    lines.append("")

    lines.append("## 3. Cap Grid Evaluation: Strict Accuracy & Truncation Rate")
    lines.append("")
    lines.append("Strict accuracy requires: finished naturally within cap $c$, parsed successfully, and correct. Truncated or unparsed responses count as incorrect (0.0).")
    lines.append("")
    lines.append("| Cap | Model | Strict Accuracy (95% CI) | Truncation Rate (95% CI) | Loop Suspect Share (Truncated) |")
    lines.append("|---|---|---|---|---|")
    for cap in CAPS:
        m = results_by_cap[cap]["medpsy"]
        q = results_by_cap[cap]["qwen"]
        lines.append(
            f"| {cap} | `medpsy-4b` | {m['strict_acc']*100:.1f}% [{m['strict_acc_ci'][0]*100:.1f}%, {m['strict_acc_ci'][1]*100:.1f}%] | "
            f"{m['trunc_rate']*100:.1f}% [{m['trunc_rate_ci'][0]*100:.1f}%, {m['trunc_rate_ci'][1]*100:.1f}%] | "
            f"{m['loop_share']*100:.1f}% ({m['n_trunc']} trunc) |"
        )
        lines.append(
            f"| {cap} | `qwen3-4b-thinking-2507` | {q['strict_acc']*100:.1f}% [{q['strict_acc_ci'][0]*100:.1f}%, {q['strict_acc_ci'][1]*100:.1f}%] | "
            f"{q['trunc_rate']*100:.1f}% [{q['trunc_rate_ci'][0]*100:.1f}%, {q['trunc_rate_ci'][1]*100:.1f}%] | "
            f"{q['loop_share']*100:.1f}% ({q['n_trunc']} trunc) |"
        )
    lines.append("")

    lines.append("## 4. Paired Model Comparison (MedPsy-4B minus Qwen3-4B-Thinking)")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Holm Family Specification:**")
    lines.append("> A single family of 10 hypothesis tests is evaluated across the 5 caps (5 strict accuracy differences + 5 truncation rate differences).")
    lines.append("> Both raw two-sided McNemar exact p-values and Holm-adjusted p-values are reported.")
    lines.append("")
    lines.append("| Cap | Metric | MedPsy - Qwen Diff | Paired Bootstrap 95% CI | Discordant (b / c) | Raw McNemar p | Holm-Adjusted p |")
    lines.append("|---|---|---|---|---|---|---|")
    for cap in CAPS:
        cmp = results_by_cap[cap]["comparison"]
        lines.append(
            f"| {cap} | Strict Accuracy | {cmp['strict_diff']*100:+.1f}% | "
            f"[{cmp['strict_diff_ci'][0]*100:+.1f}%, {cmp['strict_diff_ci'][1]*100:+.1f}%] | "
            f"b={cmp['b_strict']}, c={cmp['c_strict']} | {cmp['p_strict_raw']:.4f} | {cmp['p_strict_holm']:.4f} |"
        )
        lines.append(
            f"| {cap} | Truncation Rate | {cmp['trunc_diff']*100:+.1f}% | "
            f"[{cmp['trunc_diff_ci'][0]*100:+.1f}%, {cmp['trunc_diff_ci'][1]*100:+.1f}%] | "
            f"b={cmp['b_trunc']}, c={cmp['c_trunc']} | {cmp['p_trunc_raw']:.4f} | {cmp['p_trunc_holm']:.4f} |"
        )
    lines.append("")

    lines.append("## 5. Descriptive Finished-Only Subset (Non-Comparative Diagnostic)")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **DESCRIPTIVE, selected subset:**")
    lines.append("> Accuracies reported below are computed strictly on the subset of questions where **both models finished naturally within cap $c$**.")
    lines.append("> This is conditioned on post-treatment survival and is subject to selection bias; it is presented purely as a diagnostic of reasoning quality when unconstrained.")
    lines.append("")
    lines.append("| Cap | Both Finished N (%) | MedPsy-4B Acc (Subset) | Qwen3-4B Acc (Subset) |")
    lines.append("|---|---|---|---|")
    for cap in CAPS:
        sub = results_by_cap[cap]["descriptive_both_finished"]
        lines.append(f"| {cap} | {sub['n']}/{N} ({sub['pct']:.1f}%) | {sub['medpsy_acc']*100:.1f}% | {sub['qwen_acc']*100:.1f}% |")
    lines.append("")

    lines.append("## 6. MedPsy-4B Validation Consistency Diagnostic")
    lines.append("")
    if consistency_res["status"] == "success":
        lines.append(f"- **Reference File:** `{consistency_res['sc_val_file']}` (greedy column)")
        lines.append(f"- **Evaluated Subset:** MedPsy-4B questions that finished within 1,024 tokens and overlap with `sc_val`")
        lines.append(f"- **Overlapping Rows:** {consistency_res['n_eligible']}")
        lines.append(f"- **Exact Predicted-Answer Agreement:** {consistency_res['matches']}/{consistency_res['n_eligible']} ({consistency_res['agreement_rate']*100:.1f}%)")
    else:
        lines.append(f"- **Consistency Check:** Skipped ({consistency_res.get('reason')})")
    lines.append("")

    lines.append("## 7. Visualisations")
    lines.append("")
    lines.append(f"![Pair B Budget Sweep Curves](budget_pair_b.png)")
    lines.append("")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {out_md}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Pair B Token-Budget Sweep")
    parser.add_argument(
        "--sweep-dir",
        default="outputs/sweeps/budget_pair_b",
        help="Directory with sweep JSONL outputs (default: outputs/sweeps/budget_pair_b)",
    )
    parser.add_argument(
        "--out-md",
        default="outputs/analysis/budget_pair_b.md",
        help="Path to output markdown report",
    )
    parser.add_argument(
        "--out-png",
        default="outputs/analysis/budget_pair_b.png",
        help="Path to output PNG plot",
    )
    parser.add_argument(
        "--out-svg",
        default="outputs/analysis/budget_pair_b.svg",
        help="Path to output SVG plot",
    )
    args = parser.parse_args()

    run_analysis(args.sweep_dir, args.out_md, args.out_png, args.out_svg)


if __name__ == "__main__":
    main()
