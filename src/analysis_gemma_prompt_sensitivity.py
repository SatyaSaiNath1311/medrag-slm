#!/usr/bin/env python3
"""src/analysis_gemma_prompt_sensitivity.py — Analysis of Gemma Prompt-Sensitivity Check.

Evaluates:
  - Model pair: gemma3-4b (generic baseline) vs medgemma-1.5-4b-it (medical adaptation)
  - Data: MedQA validation split (N=200, seeded sample seed 42)
  - Condition: C1 (closed-book multi-choice QA), greedy decoding
  - Three prompt variants:
      * V1: Standard C1 baseline template (imported from existing builder)
      * V2: Clinical evaluation framing with parenthesized options
      * V3: Board-style examination framing with bracketed choices

Pre-Registered Stability Rule (docs/GEMMA_PROMPT_SENSITIVITY_RULE.md):
  The gap (gemma3-4b minus medgemma-1.5-4b-it) is defined as "stable" iff:
    1. Its sign is the same across all three prompt variants (no sign flip).
    2. The spread (max(gap) - min(gap)) is <= 10 percentage points (0.10).
  Otherwise, it is reported as prompt-sensitive.

Outputs:
  - outputs/analysis/gemma_prompt_sensitivity.md
  - outputs/analysis/gemma_prompt_sensitivity.png
  - outputs/analysis/gemma_prompt_sensitivity.svg
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from src.common import read_jsonl

VARIANTS = ["V1", "V2", "V3"]
MODEL_GENERIC = "gemma3-4b"
MODEL_MEDICAL = "medgemma-1.5-4b-it"
REGISTERED_N = 200
STABILITY_SPREAD_THRESHOLD = 0.10  # 10 percentage points

VARIANT_LABELS = {
    "V1": "V1 (Standard Baseline)",
    "V2": "V2 (Clinical Scenario)",
    "V3": "V3 (Board Examination)",
}


def is_strict_correct(row: Dict[str, Any]) -> bool:
    """Strict accuracy: not truncated AND parsed AND correct."""
    is_trunc = bool(row.get("truncated", False))
    is_prsd = bool(row.get("parsed", True))
    is_corr = bool(row.get("correct", False))
    return (not is_trunc) and is_prsd and is_corr


# ── Statistical Helpers ─────────────────────────────────────────────────────────

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
    """Exact two-sided McNemar test on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_val = 2.0 * float(binom.cdf(k, n, 0.5))
    return min(1.0, p_val)


def paired_bootstrap_ci(
    arr1: np.ndarray,
    arr2: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% percentile bootstrap CI for mean(arr1) - mean(arr2)."""
    assert len(arr1) == len(arr2), "Mismatched array lengths for paired bootstrap"
    n = len(arr1)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    diff = arr1.astype(float) - arr2.astype(float)
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


def format_p_val(p: float) -> str:
    """Format p-value: if below 1e-4 print '<1e-4', otherwise 4 decimal places."""
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def evaluate_gemma_stability(
    gap_v1: float,
    gap_v2: float,
    gap_v3: float,
    threshold: float = STABILITY_SPREAD_THRESHOLD,
) -> Tuple[bool, float, str]:
    """Pre-registered stability criterion for the Gemma closed-book gap.

    The gap (gemma3-4b minus medgemma-1.5-4b-it) is defined as "stable" iff:
      1. Its sign is the same across all three variants (all > 0 or all < 0 or all == 0).
         (A sign flip occurs if at least one positive and one negative gap exist).
      2. The spread (max(gaps) - min(gaps)) is <= threshold (default 10 percentage points = 0.10).

    Returns:
      (is_stable, spread, verdict_message)
    """
    gaps = [gap_v1, gap_v2, gap_v3]
    spread = max(gaps) - min(gaps)
    has_pos = any(g > 0 for g in gaps)
    has_neg = any(g < 0 for g in gaps)
    same_sign = not (has_pos and has_neg)
    # Range check with tolerance for exact floating point representation
    range_ok = spread <= (threshold + 1e-7)
    is_stable = same_sign and range_ok

    if is_stable:
        if has_neg:
            direction_desc = "negative (favoring medgemma-1.5-4b-it)"
        elif has_pos:
            direction_desc = "positive (favoring gemma3-4b)"
        else:
            direction_desc = "zero"
        verdict = (
            f"STABLE: Gap sign is uniform across variants ({direction_desc}) "
            f"and gap spread ({spread*100:.2f} pp) is <= {threshold*100:.1f} pp."
        )
    else:
        reasons = []
        if not same_sign:
            reasons.append("sign flip observed across variants")
        if not range_ok:
            reasons.append(f"gap spread ({spread*100:.2f} pp) exceeds {threshold*100:.1f} pp")
        verdict = f"PROMPT-SENSITIVE: {'; '.join(reasons)}."

    return is_stable, spread, verdict


def check_format_confound(
    gemma_parse_rate: float,
    medgemma_parse_rate: float,
    threshold: float = 0.05,
) -> Tuple[bool, float]:
    """Check if variant is format-confounded (> 5.0 pp parse rate difference, Section 4.1)."""
    diff = abs(gemma_parse_rate - medgemma_parse_rate)
    is_confounded = bool(diff > (threshold + 1e-7))
    return is_confounded, diff


# ── Data Loading & Validation ──────────────────────────────────────────────────

def load_and_validate_gemma_sweeps(
    sweep_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load sweep outputs, asserting strict validation split and key integrity."""
    gemma_path = sweep_dir / f"{MODEL_GENERIC}.jsonl"
    medgemma_path = sweep_dir / f"{MODEL_MEDICAL}.jsonl"
    manifest_path = sweep_dir / "manifest.json"

    if not gemma_path.exists():
        raise FileNotFoundError(f"Missing generic sweep file: {gemma_path}")
    if not medgemma_path.exists():
        raise FileNotFoundError(f"Missing medical sweep file: {medgemma_path}")

    gemma_rows = read_jsonl(gemma_path)
    medgemma_rows = read_jsonl(medgemma_path)

    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    # Hard split assertion: every row must be validation split
    for r in gemma_rows:
        if r.get("split") != "validation":
            raise ValueError(
                f"Hard assertion failed: non-validation split detected in {gemma_path}: "
                f"id={r.get('id')} split={r.get('split')}"
            )
    for r in medgemma_rows:
        if r.get("split") != "validation":
            raise ValueError(
                f"Hard assertion failed: non-validation split detected in {medgemma_path}: "
                f"id={r.get('id')} split={r.get('split')}"
            )

    return gemma_rows, medgemma_rows, manifest


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_gemma_prompt_sensitivity(
    variant_results: Dict[str, Dict[str, Any]],
    out_png: Path,
    out_svg: Path,
    n_questions: int,
    is_smoke: bool = False,
    stability_info: Optional[Tuple[bool, float, str]] = None,
) -> None:
    """Generate publication-quality figure: Gap (Gemma3 - MedGemma) per variant."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    variants = VARIANTS
    x = np.arange(len(variants))
    labels = [VARIANT_LABELS.get(v, v) for v in variants]

    gaps = [variant_results[v]["gap"] * 100.0 for v in variants]
    ci_lows = [variant_results[v]["gap_ci"][0] * 100.0 for v in variants]
    ci_highs = [variant_results[v]["gap_ci"][1] * 100.0 for v in variants]

    yerr_lower = [gaps[i] - ci_lows[i] for i in range(len(variants))]
    yerr_upper = [ci_highs[i] - gaps[i] for i in range(len(variants))]
    yerr = [yerr_lower, yerr_upper]

    # Bar/point chart with error bars
    colors = ["#4C78A8", "#F58518", "#E45756"]
    bars = ax.bar(x, gaps, width=0.45, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2, zorder=3)
    ax.errorbar(x, gaps, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.8, capsize=6, capthick=1.8, zorder=4)

    # Reference line at y=0 (parity)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2, label="Parity (0 pp)")

    # Reference line for earlier test-set C1 gap (-6.20 pp)
    ax.axhline(-6.20, color="#E45756", linestyle=":", linewidth=1.5, alpha=0.8, zorder=2, label="Test C1 Gap (-6.20 pp)")

    # Data value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.5 if height >= 0 else -0.5
        ax.annotate(
            f"{height:+.1f} pp\n[{ci_lows[i]:+.1f}, {ci_highs[i]:+.1f}]",
            xy=(bar.get_x() + bar.get_width() / 2, height + offset),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="medium")
    ax.set_ylabel(r"Strict Accuracy Gap ($\Delta$ pp = Gemma3-4B $-$ MedGemma)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

    title = f"Gemma Closed-Book MedQA Gap Under Prompt Variations (Val N={n_questions})"
    if is_smoke:
        title = f"[SMOKE RUN N={n_questions}] " + title
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Informational subtitle/legend
    if stability_info is not None and not is_smoke:
        is_stable, spread, _ = stability_info
        status_text = f"Verdict: {'STABLE' if is_stable else 'PROMPT-SENSITIVE'} (Spread: {spread*100:.1f} pp, threshold <= 10 pp)"
        ax.text(
            0.5, -0.15, status_text,
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#2ca02c" if is_stable else "#d62728",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#ced4da"),
        )

    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)


# ── Markdown Report Generation ─────────────────────────────────────────────────

def generate_markdown_report(
    results: Dict[str, Any],
    n_rows: int,
    is_smoke: bool,
    stability_info: Optional[Tuple[bool, float, str]],
    manifest: Dict[str, Any],
    report_path: Path,
) -> None:
    """Write standardized markdown report to outputs/analysis/gemma_prompt_sensitivity.md."""
    lines: List[str] = []

    # 1. Smoke banner if applicable
    if is_smoke:
        lines.append(f"SMOKE RUN N={n_rows}: not for reporting")
        lines.append("")

    lines.append("# Gemma Prompt-Sensitivity Check: gemma3-4b vs medgemma-1.5-4b-it")
    lines.append("")
    lines.append(f"**Models Evaluated:** `gemma3-4b` (Generic Baseline) vs `medgemma-1.5-4b-it` (Medical Adaptation)")
    lines.append(f"**Dataset & Split:** MedQA validation split (first {n_rows} questions in seeded order, seed 42)")
    lines.append(f"**Condition & Decoding:** Condition C1 (closed-book), greedy decoding (`do_sample=False`, `max_new_tokens=16`)")
    lines.append(f"**Pre-Registered Protocol:** `docs/GEMMA_PROMPT_SENSITIVITY_RULE.md`")
    lines.append("")

    # 2. Pre-Registered Stability Verdict Section
    lines.append("## 1. Pre-Registered Stability Evaluation")
    lines.append("")
    if is_smoke:
        lines.append(f"> [!WARNING]")
        lines.append(f"> **SMOKE RUN (N={n_rows}):** Decision verdict is suppressed. Formal stability evaluation requires full registered N=200.")
        lines.append("")
    else:
        assert stability_info is not None
        is_stable, spread, verdict_msg = stability_info
        alert_type = "IMPORTANT" if is_stable else "WARNING"
        lines.append(f"> [!{alert_type}]")
        lines.append(f"> **Pre-Registered Verdict: {verdict_msg}**")
        lines.append("")
        lines.append(f"- **Criterion 1 (Sign Uniformity):** {'PASS (All variants share the same sign)' if (not results['has_sign_flip']) else 'FAIL (Sign flip detected across variants)'}.")
        lines.append(f"- **Criterion 2 (Spread Bound):** Observed spread across variants is **{spread*100:.2f} percentage points** (Pre-registered threshold: $\\le 10.0$ pp).")
        flagged_vars = [v for v in VARIANTS if results["variants"][v]["is_format_confounded"]]
        if flagged_vars:
            flag_str = f"FLAGGED: {', '.join(flagged_vars)} (parse rate diff > 5.0 pp)"
        else:
            flag_str = "None (all variants <= 5.0 pp parse-rate difference)"
        lines.append(f"- **Format-Confound Status (Section 4.1):** {flag_str}")
        lines.append("")

    # 3. Model x Variant Performance Table
    lines.append("## 2. Model x Variant Performance Summary")
    lines.append("")
    lines.append("| Variant | Template Description | Model | Strict Acc (95% CI) | Truncation Rate (95% CI) | Parse Rate (95% CI) |")
    lines.append("|---|---|---|---|---|---|")
    for v in VARIANTS:
        v_data = results["variants"][v]
        v_label = VARIANT_LABELS.get(v, v)
        for m_key, m_name in [("generic", MODEL_GENERIC), ("medical", MODEL_MEDICAL)]:
            m_stat = v_data[m_key]
            acc_str = f"{m_stat['strict_acc']*100:.2f}% [{m_stat['strict_ci'][0]*100:.1f}%, {m_stat['strict_ci'][1]*100:.1f}%]"
            trunc_str = f"{m_stat['trunc_rate']*100:.2f}% [{m_stat['trunc_ci'][0]*100:.1f}%, {m_stat['trunc_ci'][1]*100:.1f}%]"
            parse_str = f"{m_stat['parse_rate']*100:.2f}% [{m_stat['parse_ci'][0]*100:.1f}%, {m_stat['parse_ci'][1]*100:.1f}%]"
            lines.append(f"| {v} | {v_label} | `{m_name}` | {acc_str} | {trunc_str} | {parse_str} |")
    lines.append("")

    # 4. Paired Statistical Comparisons Table
    lines.append("## 3. Paired Statistical Comparisons (Gemma3 - MedGemma)")
    lines.append("")
    lines.append("Discordant counts: $b$ = Gemma3 correct & MedGemma incorrect; $c$ = Gemma3 incorrect & MedGemma correct.")
    lines.append("Differences are evaluated via exact two-sided McNemar tests and paired percentile bootstrap 95% CIs (10,000 resamples, seed 42).")
    lines.append("Familywise error rate across the three variants is controlled using a single Holm-Bonferroni family ($m=3$).")
    lines.append("")
    lines.append("| Variant | Gemma3 Acc | MedGemma Acc | Gap ($\\Delta$ pp) | 95% Bootstrap CI | Discordant ($b/c$) | Raw McNemar $p$ | Holm-Adjusted $p$ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        v_data = results["variants"][v]
        g_acc = v_data["generic"]["strict_acc"] * 100.0
        m_acc = v_data["medical"]["strict_acc"] * 100.0
        gap = v_data["gap"] * 100.0
        ci_low = v_data["gap_ci"][0] * 100.0
        ci_high = v_data["gap_ci"][1] * 100.0
        b = v_data["b"]
        c = v_data["c"]
        p_raw = format_p_val(v_data["p_raw"])
        p_holm = format_p_val(v_data["p_holm"])
        lines.append(
            f"| {v} | {g_acc:.2f}% | {m_acc:.2f}% | {gap:+.2f} pp | [{ci_low:+.2f}, {ci_high:+.2f}] | {b} / {c} | {p_raw} | {p_holm} |"
        )
    lines.append("")

    # 4.1 Format-Confound Check & Descriptive Sensitivity
    lines.append("### 3.1 Format-Confound Check & Descriptive Sensitivity (Section 4.1)")
    lines.append("")
    lines.append("A variant is flagged `FORMAT-CONFOUNDED` if the parse rates of the two models differ by more than 5.0 percentage points in that variant.")
    lines.append("For flagged variants, the gap computed only on questions where both models produced a parsed answer is shown, labelled `DESCRIPTIVE, selected subset`.")
    lines.append("")
    flagged_variants = [v for v in VARIANTS if results["variants"][v]["is_format_confounded"]]
    if flagged_variants:
        lines.append("| Variant | Status | Parse Rate Diff | Both-Parsed N (%) | Gemma3 Both-Parsed Acc | MedGemma Both-Parsed Acc | Descriptive Gap |")
        lines.append("|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            v_data = results["variants"][v]
            status = "**FORMAT-CONFOUNDED**" if v_data["is_format_confounded"] else "Not Confounded"
            bp = v_data["both_parsed"]
            bp_gap_label = f"{bp['gap']*100:+.2f} pp (DESCRIPTIVE, selected subset)" if v_data["is_format_confounded"] else f"{bp['gap']*100:+.2f} pp"
            lines.append(
                f"| {v} | {status} | {v_data['parse_diff']*100:.2f} pp | {bp['n']} ({bp['pct']:.1f}%) | "
                f"{bp['gemma_acc']*100:.2f}% | {bp['medgemma_acc']*100:.2f}% | {bp_gap_label} |"
            )
    else:
        lines.append("No variants were flagged `FORMAT-CONFOUNDED`. Pairwise parse-rate differences:")
        for v in VARIANTS:
            v_data = results["variants"][v]
            lines.append(f"- **{v}:** {v_data['parse_diff']*100:.2f} percentage points ($\\le 5.0$ pp threshold).")
    lines.append("")

    # 5. Visual Analysis Reference
    lines.append("## 4. Visual Analysis")
    lines.append("")
    lines.append(f"![Gemma Prompt Sensitivity Gap](gemma_prompt_sensitivity.png)")
    lines.append("")

    # 6. Scope Statement
    lines.append("## 5. Scope and Test-Set Integrity")
    lines.append("")
    lines.append("- **Validation Only:** All analyses and inferences in this report are conducted exclusively on the MedQA validation split ($N={0}$).".format(n_rows))
    lines.append("- **Test-Set Independence:** This check does not access or modify test data. The test-set results previously reported (e.g. MedGemma +6.20 pp in Condition C1) remain untouched as the final empirical benchmark.")
    lines.append("- **Purpose:** This sensitivity check quantifies the degree of prompt brittleness in the closed-book comparison, testing whether the adaptation advantage is robust to prompt rephrasing.")
    lines.append("")

    # 7. Metadata / Provenance
    lines.append("## 6. Execution Provenance & Manifest")
    lines.append("")
    lines.append(f"- **Git Commit:** `{manifest.get('git_commit_hash', manifest.get('git_commit', 'unknown'))}`")
    lines.append(f"- **Rule Document Hash (sha256):** `{manifest.get('rule_file_sha256', manifest.get('rule_hash', 'unknown'))}`")
    lines.append(f"- **Completed Flag:** `{manifest.get('completed', False)}`")
    lines.append(f"- **Hours Elapsed:** `{manifest.get('hours_used', 'N/A')}`")
    lines.append("")
    if "prompt_template_hashes" in manifest:
        lines.append("### Prompt Template Hashes (sha256)")
        for v, h in manifest["prompt_template_hashes"].items():
            lines.append(f"- `{v}`: `{h}`")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main Analysis Orchestrator ─────────────────────────────────────────────────

def run_analysis(
    sweep_dir: Path,
    out_report: Path,
    out_png: Path,
    out_svg: Path,
    expected_n: int = REGISTERED_N,
) -> Dict[str, Any]:
    """Run full prompt-sensitivity analysis and write outputs."""
    gemma_rows, medgemma_rows, manifest = load_and_validate_gemma_sweeps(sweep_dir)

    # Group by variant
    gemma_by_var: Dict[str, List[Dict[str, Any]]] = {v: [] for v in VARIANTS}
    medgemma_by_var: Dict[str, List[Dict[str, Any]]] = {v: [] for v in VARIANTS}

    for r in gemma_rows:
        v = r.get("variant")
        if v in gemma_by_var:
            gemma_by_var[v].append(r)
    for r in medgemma_rows:
        v = r.get("variant")
        if v in medgemma_by_var:
            medgemma_by_var[v].append(r)

    # Check question id integrity
    first_var = VARIANTS[0]
    base_ids = [r["id"] for r in gemma_by_var[first_var]]
    n_rows = len(base_ids)

    if n_rows == 0:
        raise ValueError("No rows found in sweep data for analysis.")

    # Assert integrity across variants and models
    base_id_set = set(base_ids)
    assert len(base_id_set) == n_rows, "Duplicate question IDs detected in V1"

    for v in VARIANTS:
        g_ids = {r["id"] for r in gemma_by_var[v]}
        m_ids = {r["id"] for r in medgemma_by_var[v]}
        assert g_ids == base_id_set, f"Mismatched question IDs in {MODEL_GENERIC} variant {v}"
        assert m_ids == base_id_set, f"Mismatched question IDs in {MODEL_MEDICAL} variant {v}"

    is_smoke = (n_rows != expected_n)

    # Build per-variant analysis
    variant_results: Dict[str, Dict[str, Any]] = {}
    p_values_for_holm: List[float] = []

    for v in VARIANTS:
        # Sort rows to align question ids
        g_dict = {r["id"]: r for r in gemma_by_var[v]}
        m_dict = {r["id"]: r for r in medgemma_by_var[v]}

        g_list = [g_dict[qid] for qid in base_ids]
        m_list = [m_dict[qid] for qid in base_ids]

        g_corr = np.array([1 if is_strict_correct(r) else 0 for r in g_list], dtype=int)
        m_corr = np.array([1 if is_strict_correct(r) else 0 for r in m_list], dtype=int)

        g_strict_k = int(np.sum(g_corr))
        m_strict_k = int(np.sum(m_corr))

        g_trunc_k = int(np.sum([1 if r.get("truncated") else 0 for r in g_list]))
        m_trunc_k = int(np.sum([1 if r.get("truncated") else 0 for r in m_list]))

        g_parsed_arr = np.array([1 if r.get("parsed") else 0 for r in g_list], dtype=int)
        m_parsed_arr = np.array([1 if r.get("parsed") else 0 for r in m_list], dtype=int)
        g_parse_k = int(np.sum(g_parsed_arr))
        m_parse_k = int(np.sum(m_parsed_arr))

        g_parse_rate = g_parse_k / n_rows
        m_parse_rate = m_parse_k / n_rows
        is_confounded, parse_diff = check_format_confound(g_parse_rate, m_parse_rate, threshold=0.05)

        both_parsed_mask = (g_parsed_arr == 1) & (m_parsed_arr == 1)
        n_both_parsed = int(np.sum(both_parsed_mask))
        if n_both_parsed > 0:
            g_both_parsed_acc = float(np.mean(g_corr[both_parsed_mask]))
            m_both_parsed_acc = float(np.mean(m_corr[both_parsed_mask]))
            both_parsed_gap = float(g_both_parsed_acc - m_both_parsed_acc)
        else:
            g_both_parsed_acc = 0.0
            m_both_parsed_acc = 0.0
            both_parsed_gap = 0.0

        b = int(np.sum((g_corr == 1) & (m_corr == 0)))
        c = int(np.sum((g_corr == 0) & (m_corr == 1)))
        p_raw = exact_mcnemar_test(b, c)
        p_values_for_holm.append(p_raw)

        gap = float(np.mean(g_corr) - np.mean(m_corr))
        gap_ci = paired_bootstrap_ci(g_corr, m_corr, n_boot=10000, seed=42)

        variant_results[v] = {
            "generic": {
                "strict_acc": g_strict_k / n_rows,
                "strict_ci": wilson_score_ci(g_strict_k, n_rows),
                "trunc_rate": g_trunc_k / n_rows,
                "trunc_ci": wilson_score_ci(g_trunc_k, n_rows),
                "parse_rate": g_parse_rate,
                "parse_ci": wilson_score_ci(g_parse_k, n_rows),
            },
            "medical": {
                "strict_acc": m_strict_k / n_rows,
                "strict_ci": wilson_score_ci(m_strict_k, n_rows),
                "trunc_rate": m_trunc_k / n_rows,
                "trunc_ci": wilson_score_ci(m_trunc_k, n_rows),
                "parse_rate": m_parse_rate,
                "parse_ci": wilson_score_ci(m_parse_k, n_rows),
            },
            "b": b,
            "c": c,
            "p_raw": p_raw,
            "gap": gap,
            "gap_ci": gap_ci,
            "parse_diff": parse_diff,
            "is_format_confounded": is_confounded,
            "both_parsed": {
                "n": n_both_parsed,
                "pct": (n_both_parsed / n_rows) * 100.0,
                "gemma_acc": g_both_parsed_acc,
                "medgemma_acc": m_both_parsed_acc,
                "gap": both_parsed_gap,
            },
        }

    # Holm-Bonferroni across the 3 variants
    holm_ps = holm_bonferroni(p_values_for_holm)
    for idx, v in enumerate(VARIANTS):
        variant_results[v]["p_holm"] = holm_ps[idx]

    # Evaluate pre-registered stability criterion
    gap_v1 = variant_results["V1"]["gap"]
    gap_v2 = variant_results["V2"]["gap"]
    gap_v3 = variant_results["V3"]["gap"]

    stability_info: Optional[Tuple[bool, float, str]] = None
    if not is_smoke:
        stability_info = evaluate_gemma_stability(gap_v1, gap_v2, gap_v3)

    gaps = [gap_v1, gap_v2, gap_v3]
    has_sign_flip = any(g > 0 for g in gaps) and any(g < 0 for g in gaps)

    results_bundle = {
        "n_rows": n_rows,
        "is_smoke": is_smoke,
        "variants": variant_results,
        "has_sign_flip": has_sign_flip,
        "stability_info": stability_info,
    }

    # Generate plot
    plot_gemma_prompt_sensitivity(
        variant_results,
        out_png,
        out_svg,
        n_questions=n_rows,
        is_smoke=is_smoke,
        stability_info=stability_info,
    )

    # Generate markdown report
    generate_markdown_report(
        results_bundle,
        n_rows=n_rows,
        is_smoke=is_smoke,
        stability_info=stability_info,
        manifest=manifest,
        report_path=out_report,
    )

    return results_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Gemma Prompt-Sensitivity Sweeps.")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "sweeps" / "gemma_prompt_sensitivity",
        help="Directory containing sweep .jsonl and manifest.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "gemma_prompt_sensitivity.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "gemma_prompt_sensitivity.png",
        help="Output PNG figure path",
    )
    parser.add_argument(
        "--out-svg",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "gemma_prompt_sensitivity.svg",
        help="Output SVG figure path",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=REGISTERED_N,
        help="Expected questions per variant (default 200). Non-matching N triggers smoke banner.",
    )
    args = parser.parse_args()

    results = run_analysis(
        sweep_dir=args.sweep_dir,
        out_report=args.out_report,
        out_png=args.out_png,
        out_svg=args.out_svg,
        expected_n=args.expected_n,
    )
    print(f"Analysis complete for N={results['n_rows']} questions per variant.")
    if results["is_smoke"]:
        print(f"SMOKE RUN: Decision verdict suppressed.")
    elif results["stability_info"]:
        _, _, verdict = results["stability_info"]
        print(f"VERDICT: {verdict}")
    print(f"Report written to: {args.out_report}")


if __name__ == "__main__":
    main()
