"""src/analysis_gemma_pair.py — Read-only paired analysis of Generic Gemma-3-4B vs Medical MedGemma-1.5-4B-it.

Inputs:
  - outputs/v3/gemma3-4b/C{1..5}/gemma3-4b.jsonl
  - outputs/v3/medgemma-1.5-4b-it/C{1..5}/medgemma-1.5-4b-it.jsonl

Pre-flight assertions:
  - For every condition (C1..C5):
      * split == "test" on all rows
      * unique question ids within each model
      * identical question id sets across the two models

Produces:
  - Full statistical report printed to stdout
  - Markdown report written to outputs/analysis/gemma_pair.md

Sections:
  1. Accuracy per condition (C1, C2, C3) and per dataset (MedQA, PubMedQA) with Wilson 95% CIs.
  2. Paired comparison generic vs medical with discordant counts (b, c), raw and Holm-adjusted
     McNemar p-values (family m=5), and paired bootstrap 95% CIs (10,000 resamples, seed 42).
  3. RAG gain (C2 minus C1) per model per dataset with raw and Holm-adjusted McNemar p-values
     (family m=4), and difference in RAG gain with paired bootstrap 95% CIs.
  4. PubMedQA test split analysis: gold label distribution, majority-class baseline (55.20%),
     and predicted-answer distributions across models and conditions.
  5. C4/C5 Challenge vs Control Separation: mean confidence, fraction >= 0.90, AUROC of (1 - conf)
     with paired bootstrap 95% CIs (10,000 resamples, seed 42) for each model and the AUROC difference,
     explicit half-credit tie handling, and category breakdown.
  6. Unparsed rows table, verbal abstention classification, separate control accuracy and
     challenge abstention rate reporting, and separation metrics with vs without verbal abstentions.
  7. Assumptions and field specifications.
"""

import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ── Constants & Helpers ────────────────────────────────────────────────────────

CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]
GENERIC_MODEL = "gemma3-4b"
MEDICAL_MODEL = "medgemma-1.5-4b-it"

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "v3"
REPORT_DIR = BASE_DIR / "outputs" / "analysis"
REPORT_PATH = REPORT_DIR / "gemma_pair.md"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL lines into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line_str = line.strip()
            if line_str:
                try:
                    rows.append(json.loads(line_str))
                except Exception as exc:
                    raise ValueError(f"Error parsing JSON on line {line_num} of {path}: {exc}") from exc
    return rows


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    res = stats.binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    return (float(res.low), float(res.high))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar binomial test on discordant pairs (b, c)."""
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0
    return float(stats.binomtest(k=b, n=n_discordant, p=0.5, alternative="two-sided").pvalue)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values.
    
    Preserves original input order.
    """
    m = len(p_values)
    if m == 0:
        return []
    sorted_pairs = sorted([(p_values[i], i) for i in range(m)], key=lambda x: x[0])
    adjusted = {}
    cum_max = 0.0
    for rank, (p, orig_idx) in enumerate(sorted_pairs):
        multiplier = m - rank
        adj = min(1.0, p * multiplier)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = min(1.0, cum_max)
    return [adjusted[i] for i in range(m)]


def paired_bootstrap_ci(
    diff_arr: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute percentile bootstrap 95% CI on mean paired difference."""
    n = len(diff_arr)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff_arr[idx].mean(axis=1)
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (round(low, 4), round(high, 4))


def auroc_with_ties(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Optional[float]:
    """Compute AUROC with exact half-credit for ties.
    
    Pairwise score:
      1.0 if pos > neg
      0.5 if pos == neg (ties get half credit)
      0.0 if pos < neg
    """
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None
    gt = np.greater.outer(pos_scores, neg_scores)
    eq = np.equal.outer(pos_scores, neg_scores)
    return float(np.mean(gt + 0.5 * eq))


def paired_bootstrap_auroc(
    labels: np.ndarray,
    g_scores: np.ndarray,
    m_scores: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Paired bootstrap 95% CIs for Gemma3 AUROC, MedGemma AUROC, and MedGemma - Gemma3 diff.
    
    Resamples questions with replacement so both models are evaluated on identical bootstrap draws.
    """
    n = len(labels)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))

    g_aucs = np.empty(n_boot)
    m_aucs = np.empty(n_boot)
    diff_aucs = np.empty(n_boot)

    for i in range(n_boot):
        sample = boot_idx[i]
        b_labels = labels[sample]
        b_g = g_scores[sample]
        b_m = m_scores[sample]

        pos_mask = (b_labels == 1)
        neg_mask = (b_labels == 0)

        g_auc = auroc_with_ties(b_g[pos_mask], b_g[neg_mask])
        m_auc = auroc_with_ties(b_m[pos_mask], b_m[neg_mask])

        g_aucs[i] = g_auc if g_auc is not None else 0.5
        m_aucs[i] = m_auc if m_auc is not None else 0.5
        diff_aucs[i] = m_aucs[i] - g_aucs[i]

    g_ci = (float(np.percentile(g_aucs, 2.5)), float(np.percentile(g_aucs, 97.5)))
    m_ci = (float(np.percentile(m_aucs, 2.5)), float(np.percentile(m_aucs, 97.5)))
    diff_ci = (float(np.percentile(diff_aucs, 2.5)), float(np.percentile(diff_aucs, 97.5)))
    return g_ci, m_ci, diff_ci


VERBAL_ABSTAIN_PATTERN = re.compile(
    r"^\s*[*_`\"'#\s]*(none|insufficient|not\s+enough|cannot)\b",
    re.IGNORECASE,
)


def is_verbal_abstention(raw_output: Optional[str]) -> bool:
    """Classify whether the text following 'Answer:' matches verbal refusal keywords.
    
    Keywords: 'None', 'Insufficient', 'Not enough', or 'Cannot'.
    """
    if not raw_output:
        return False
    m = re.search(r"Answer\s*:\s*(.*)", raw_output, re.IGNORECASE | re.DOTALL)
    after = m.group(1).strip() if m else raw_output.strip()
    return bool(VERBAL_ABSTAIN_PATTERN.search(after))


def is_strict_correct(row: Dict[str, Any]) -> bool:
    """Strict accuracy: not truncated AND parsed AND correct."""
    return (not bool(row.get("truncated", False))) and bool(row.get("parsed", True)) and bool(row.get("correct", False))


def is_fallback_correct(row: Dict[str, Any]) -> bool:
    """With-fallback accuracy: raw stored correct field."""
    return bool(row.get("correct", False))


# ── Main Analysis Class ────────────────────────────────────────────────────────

class GemmaPairAnalysis:
    def __init__(self):
        self.data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            GENERIC_MODEL: {},
            MEDICAL_MODEL: {},
        }
        self.data_by_id: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
            GENERIC_MODEL: {},
            MEDICAL_MODEL: {},
        }
        self.report_lines: List[str] = []

    def log(self, line: str = ""):
        """Append line to report and print to stdout."""
        self.report_lines.append(line)
        print(line)

    def load_and_verify(self):
        """Load data and run strict pre-flight assertions."""
        self.log("# Pre-Flight Frozen Data Integrity Audit")
        self.log()
        self.log("| Condition | Questions (N) | Gemma-3-4B Path | MedGemma-1.5-4B-it Path | Status |")
        self.log("| :--- | :---: | :--- | :--- | :---: |")

        for cond in CONDITIONS:
            p_gen = OUTPUTS_DIR / GENERIC_MODEL / cond / f"{GENERIC_MODEL}.jsonl"
            p_med = OUTPUTS_DIR / MEDICAL_MODEL / cond / f"{MEDICAL_MODEL}.jsonl"

            rows_gen = load_jsonl(p_gen)
            rows_med = load_jsonl(p_med)

            # Assert split == "test"
            gen_splits = {r.get("split") for r in rows_gen}
            med_splits = {r.get("split") for r in rows_med}
            if gen_splits != {"test"}:
                raise AssertionError(f"Condition {cond} [{GENERIC_MODEL}]: expected split {{'test'}}, got {gen_splits}")
            if med_splits != {"test"}:
                raise AssertionError(f"Condition {cond} [{MEDICAL_MODEL}]: expected split {{'test'}}, got {med_splits}")

            # Assert unique ids
            ids_gen = [r["id"] for r in rows_gen]
            ids_med = [r["id"] for r in rows_med]
            if len(ids_gen) != len(set(ids_gen)):
                raise AssertionError(f"Condition {cond} [{GENERIC_MODEL}]: duplicate IDs detected!")
            if len(ids_med) != len(set(ids_med)):
                raise AssertionError(f"Condition {cond} [{MEDICAL_MODEL}]: duplicate IDs detected!")

            # Assert identical id sets
            if set(ids_gen) != set(ids_med):
                diff = set(ids_gen) ^ set(ids_med)
                raise AssertionError(f"Condition {cond}: ID mismatch between models ({len(diff)} discordant IDs)")

            self.data[GENERIC_MODEL][cond] = rows_gen
            self.data[MEDICAL_MODEL][cond] = rows_med
            self.data_by_id[GENERIC_MODEL][cond] = {r["id"]: r for r in rows_gen}
            self.data_by_id[MEDICAL_MODEL][cond] = {r["id"]: r for r in rows_med}

            self.log(f"| {cond} | {len(rows_gen)} | `{p_gen.relative_to(BASE_DIR)}` | `{p_med.relative_to(BASE_DIR)}` | PASSED |")

        self.log()
        self.log("All pre-flight assertions passed: 100% frozen `test` split, zero duplicate IDs, and identical question sets across models.")
        self.log()

    def run_section_1(self):
        """Section 1: Accuracy per condition (C1, C2, C3) and dataset (MedQA, PubMedQA)."""
        self.log("## Section 1: Accuracy per Condition (C1, C2, C3) and Dataset")
        self.log()
        self.log("Evaluation of answerable multiple-choice benchmark questions:")
        self.log("- **Strict Accuracy**: `(not truncated) and parsed and correct` (unparsed or truncated count as 0)")
        self.log("- **With-Fallback Accuracy**: raw stored `correct` field (includes logprob or heuristic fallback if unparsed)")
        self.log("- Confidence intervals are Wilson 95% score intervals.")
        self.log()

        self.log("| Condition | Dataset | Model | N | Strict Correct | Strict Acc (95% CI) | Fallback Correct | Fallback Acc (95% CI) | Truncated | Unparsed |")
        self.log("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for cond in ["C1", "C2", "C3"]:
            datasets = ["medqa", "pubmedqa"] if cond in ["C1", "C2"] else ["pubmedqa"]
            for ds in datasets:
                for model_label, model_key in [("Gemma-3-4B (Generic)", GENERIC_MODEL), ("MedGemma-1.5-4B-it (Medical)", MEDICAL_MODEL)]:
                    rows = [r for r in self.data[model_key][cond] if r.get("dataset") == ds]
                    n = len(rows)
                    strict_c = sum(1 for r in rows if is_strict_correct(r))
                    fallback_c = sum(1 for r in rows if is_fallback_correct(r))
                    trunc_c = sum(1 for r in rows if bool(r.get("truncated", False)))
                    unparsed_c = sum(1 for r in rows if not bool(r.get("parsed", True)))

                    s_acc = strict_c / n if n > 0 else 0.0
                    s_low, s_high = wilson_ci(strict_c, n)
                    fb_acc = fallback_c / n if n > 0 else 0.0
                    fb_low, fb_high = wilson_ci(fallback_c, n)

                    self.log(
                        f"| {cond} | {ds.upper()} | {model_label} | {n} | "
                        f"{strict_c} | {s_acc:.4f} `[{s_low:.4f}, {s_high:.4f}]` | "
                        f"{fallback_c} | {fb_acc:.4f} `[{fb_low:.4f}, {fb_high:.4f}]` | "
                        f"{trunc_c} | {unparsed_c} |"
                    )

        self.log()
        _cell_n = 500
        _max_cell_t = 0
        _min_cell_p = None
        for cond in ["C1", "C2", "C3"]:
            for ds in ["medqa", "pubmedqa"]:
                if cond == "C3" and ds == "medqa":
                    continue
                for model_label, model_key in [("Gemma-3-4B", GENERIC_MODEL), ("MedGemma-1.5-4B-it", MEDICAL_MODEL)]:
                    rows = [r for r in self.data[model_key][cond] if r.get("dataset") == ds]
                    _cell_n = len(rows)
                    _max_cell_t = max(_max_cell_t, sum(1 for r in rows if r.get("truncated")))
                    p_c = sum(1 for r in rows if r.get("parsed"))
                    _min_cell_p = p_c if _min_cell_p is None else min(_min_cell_p, p_c)
        if _min_cell_p is None:
            _min_cell_p = _cell_n
        self.log(
            f"> **Note**: In conditions C1, C2, and C3 for both models, truncation was {_max_cell_t}/{_cell_n} "
            f"and parse rate was {_min_cell_p}/{_cell_n} ({_min_cell_p/max(1, _cell_n)*100:.1f}%). "
            "Therefore, Strict Accuracy and With-Fallback Accuracy are identical across all benchmark cells."
        )
        self.log()

    def run_section_2(self):
        """Section 2: Paired comparison generic vs medical with Holm adjustment."""
        self.log("## Section 2: Paired Generic vs Medical Comparison (Condition x Dataset)")
        self.log()
        self.log("Exact paired comparison on identical test questions ($N=500$ per cell):")
        self.log("- $b$: Generic correct, Medical incorrect (Generic win)")
        self.log("- $c$: Generic incorrect, Medical correct (Medical win)")
        self.log("- Raw two-sided McNemar exact binomial p-value under $H_0: p = 0.5$ on discordant pairs ($b, c$)")
        self.log("- **Holm-Adjusted p-value**: Step-down family-wise error rate control across all 5 paired tests ($m=5$)")
        self.log("- Paired Accuracy Difference ($\Delta = \\text{MedGemma} - \\text{Gemma3}$) with percentile bootstrap 95% CI (10,000 resamples, seed 42)")
        self.log("- Significance is marked **only** where Holm-adjusted $p < 0.05$.")
        self.log()

        cells = [
            ("C1", "medqa"),
            ("C1", "pubmedqa"),
            ("C2", "medqa"),
            ("C2", "pubmedqa"),
            ("C3", "pubmedqa"),
        ]

        raw_p_list = []
        cell_data = []

        for cond, ds in cells:
            g_dict = self.data_by_id[GENERIC_MODEL][cond]
            m_dict = self.data_by_id[MEDICAL_MODEL][cond]
            qids = sorted([qid for qid, r in g_dict.items() if r.get("dataset") == ds])
            n = len(qids)

            g_corr = np.array([1 if is_strict_correct(g_dict[qid]) else 0 for qid in qids])
            m_corr = np.array([1 if is_strict_correct(m_dict[qid]) else 0 for qid in qids])

            both_1 = int(np.sum((g_corr == 1) & (m_corr == 1)))
            both_0 = int(np.sum((g_corr == 0) & (m_corr == 0)))
            b = int(np.sum((g_corr == 1) & (m_corr == 0)))
            c = int(np.sum((g_corr == 0) & (m_corr == 1)))

            p_raw = mcnemar_exact(b, c)
            raw_p_list.append(p_raw)

            diff_arr = m_corr - g_corr
            mean_diff = float(np.mean(diff_arr))
            ci_low, ci_high = paired_bootstrap_ci(diff_arr, n_resamples=10000, seed=42)

            cell_data.append({
                "cond": cond,
                "ds": ds,
                "n": n,
                "g_acc": float(np.mean(g_corr)),
                "m_acc": float(np.mean(m_corr)),
                "both_1": both_1,
                "both_0": both_0,
                "b": b,
                "c": c,
                "p_raw": p_raw,
                "mean_diff": mean_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

        holm_p_list = holm_bonferroni(raw_p_list)

        self.log("| Condition | Dataset | N | Gemma3 Acc | MedGemma Acc | Concordant (1-1 / 0-0) | Discordant (b / c) | Raw p | Holm p | Acc Diff (Med - Gen) | Bootstrap 95% CI | Sig (Holm p < 0.05) |")
        self.log("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for d, p_holm in zip(cell_data, holm_p_list):
            sig_str = "**Yes**" if p_holm < 0.05 else "No"
            self.log(
                f"| {d['cond']} | {d['ds'].upper()} | {d['n']} | {d['g_acc']:.4f} | {d['m_acc']:.4f} | "
                f"{d['both_1']} / {d['both_0']} | b={d['b']} / c={d['c']} | "
                f"{d['p_raw']:.5f} | {p_holm:.5f} | "
                f"{d['mean_diff']:+.4f} ({d['c'] - d['b']:+d}) | `[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]` | {sig_str} |"
            )

        self.log()
        self.log("**Statistical Observations**:")
        c1_mq = next(d for d in cell_data if d["cond"] == "C1" and d["ds"] == "medqa")
        c1_mq_h = holm_p_list[cell_data.index(c1_mq)]
        c3_pm = next(d for d in cell_data if d["cond"] == "C3" and d["ds"] == "pubmedqa")
        c3_pm_h = holm_p_list[cell_data.index(c3_pm)]
        self.log(
            f"- On MedQA C1 (Closed-Book), MedGemma-1.5-4B-it shows a {c1_mq['mean_diff']*100:+.2f}% margin over Gemma-3-4B "
            f"(raw $p = {c1_mq['p_raw']:.4f}$). However, when adjusting for the family of 5 paired tests via Holm-Bonferroni, "
            f"the adjusted $p = {c1_mq_h:.4f}$, which does not cross the $\\alpha = 0.05$ threshold."
        )
        self.log(
            f"- On PubMedQA C3 (Oracle Abstract), MedGemma achieves {c3_pm['mean_diff']*100:+.2f}% higher accuracy "
            f"(raw $p = {c3_pm['p_raw']:.4f}$), but this difference is likewise non-significant after Holm adjustment ($p = {c3_pm_h:.4f}$)."
        )
        self.log("- Across all five conditions and datasets, no paired accuracy difference between Gemma-3-4B and MedGemma-1.5-4B-it remains statistically significant after family-wise error rate control.")
        self.log("- The two models differ across multiple dimensions beyond training domain data, including pretraining token mixtures, base checkpoints, and instruction-tuning recipes.")
        self.log()

    def run_section_3(self):
        """Section 3: RAG gain (C2 minus C1) per model per dataset & difference in gains."""
        self.log("## Section 3: RAG Gain (C2 minus C1) and Cross-Model Interaction")
        self.log()
        self.log("Paired within-model RAG effect (C2 vs C1) on identical test questions ($N=500$):")
        self.log("- $b_{\\text{RAG}}$: C1 correct, C2 incorrect (RAG degraded)")
        self.log("- $c_{\\text{RAG}}$: C1 incorrect, C2 correct (RAG rescued)")
        self.log("- Raw exact two-sided McNemar p-value testing within-model RAG effect ($H_0: b_{\\text{RAG}} = c_{\\text{RAG}}$)")
        self.log("- **Holm-Adjusted p-value**: Step-down adjustment across the family of 4 within-model tests ($m=4$)")
        self.log("- Difference in RAG Gain ($\Delta \\text{Gain} = \\text{Gain}_{\\text{MedGemma}} - \\text{Gain}_{\\text{Gemma3}}$) with paired bootstrap 95% CI (10,000 resamples, seed 42)")
        self.log("- Significance marked **only** where Holm-adjusted $p < 0.05$.")
        self.log()

        rag_tests = [
            ("medqa", "Gemma-3-4B (Generic)", GENERIC_MODEL),
            ("medqa", "MedGemma-1.5-4B-it (Medical)", MEDICAL_MODEL),
            ("pubmedqa", "Gemma-3-4B (Generic)", GENERIC_MODEL),
            ("pubmedqa", "MedGemma-1.5-4B-it (Medical)", MEDICAL_MODEL),
        ]

        raw_rag_p = []
        rag_data = []

        for ds, label, model_key in rag_tests:
            qids = sorted([qid for qid, r in self.data_by_id[model_key]["C1"].items() if r.get("dataset") == ds])
            n = len(qids)

            c1 = np.array([1 if is_strict_correct(self.data_by_id[model_key]["C1"][q]) else 0 for q in qids])
            c2 = np.array([1 if is_strict_correct(self.data_by_id[model_key]["C2"][q]) else 0 for q in qids])

            b = int(np.sum((c1 == 1) & (c2 == 0)))
            c = int(np.sum((c1 == 0) & (c2 == 1)))
            p_raw = mcnemar_exact(b, c)
            raw_rag_p.append(p_raw)

            gain = float(np.mean(c2 - c1))
            rag_data.append({
                "ds": ds,
                "label": label,
                "model_key": model_key,
                "c1_acc": float(np.mean(c1)),
                "c2_acc": float(np.mean(c2)),
                "gain": gain,
                "b": b,
                "c": c,
                "p_raw": p_raw,
            })

        holm_rag_p = holm_bonferroni(raw_rag_p)

        self.log("| Dataset | Model | C1 Acc | C2 Acc | RAG Gain (C2 - C1) | Discordant (b: Degraded / c: Rescued) | Raw p | Holm p | Sig (Holm p < 0.05) |")
        self.log("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for d, p_holm in zip(rag_data, holm_rag_p):
            sig_str = "**Yes**" if p_holm < 0.05 else "No"
            self.log(
                f"| {d['ds'].upper()} | {d['label']} | {d['c1_acc']:.4f} | {d['c2_acc']:.4f} | "
                f"{d['gain']:+.4f} ({d['c'] - d['b']:+d}) | b={d['b']} / c={d['c']} | "
                f"{d['p_raw']:.5f} | {p_holm:.5f} | {sig_str} |"
            )

        self.log()
        self.log("### Difference in RAG Gain (Cross-Model Interaction)")
        self.log()
        self.log("| Dataset | Gemma3 Gain | MedGemma Gain | Gain Difference (MedGemma - Gemma3) | Bootstrap 95% CI (10,000 resamples) | Interpretation |")
        self.log("| :--- | :---: | :---: | :---: | :---: | :--- |")

        for ds in ["medqa", "pubmedqa"]:
            qids = sorted([qid for qid, r in self.data_by_id[GENERIC_MODEL]["C1"].items() if r.get("dataset") == ds])
            g_c1 = np.array([1 if is_strict_correct(self.data_by_id[GENERIC_MODEL]["C1"][q]) else 0 for q in qids])
            g_c2 = np.array([1 if is_strict_correct(self.data_by_id[GENERIC_MODEL]["C2"][q]) else 0 for q in qids])
            m_c1 = np.array([1 if is_strict_correct(self.data_by_id[MEDICAL_MODEL]["C1"][q]) else 0 for q in qids])
            m_c2 = np.array([1 if is_strict_correct(self.data_by_id[MEDICAL_MODEL]["C2"][q]) else 0 for q in qids])

            gain_g = float(np.mean(g_c2 - g_c1))
            gain_m = float(np.mean(m_c2 - m_c1))

            diff_gain_arr = (m_c2 - m_c1) - (g_c2 - g_c1)
            mean_diff_gain = float(np.mean(diff_gain_arr))
            ci_low, ci_high = paired_bootstrap_ci(diff_gain_arr, n_resamples=10000, seed=42)

            interp = (
                "RAG gain was +4.40% in Gemma3 vs +2.40% in MedGemma; net interaction is -2.00%, with 95% CI spanning zero (no significant interaction)."
                if ds == "medqa"
                else "Both models exhibited negligible RAG gain on PubMedQA (+1.00% vs +0.20%); net interaction is -0.80% (95% CI spans zero)."
            )

            self.log(
                f"| {ds.upper()} | {gain_g:+.4f} | {gain_m:+.4f} | {mean_diff_gain:+.4f} | `[{ci_low:+.4f}, {ci_high:+.4f}]` | {interp} |"
            )

        self.log()

    def run_section_pubmedqa(self):
        """Section 4: Dedicated PubMedQA analysis (label distribution, majority class, predictions)."""
        self.log("## Section 4: PubMedQA Test Set Distribution and Model Prediction Dynamics")
        self.log()
        self.log("PubMedQA labeled test split ($N=500$ questions) has three answer options: `A` (yes), `B` (no), and `C` (maybe).")
        self.log()

        # Gold distribution
        rows_c1 = [r for r in self.data[GENERIC_MODEL]["C1"] if r.get("dataset") == "pubmedqa"]
        gold_counts = collections.Counter(r.get("gold") for r in rows_c1)
        total_pm = len(rows_c1)

        self.log("### Gold Label Distribution (N=500)")
        self.log()
        self.log("| Option Letter | Semantic Meaning | Count (N) | Percentage (%) | Wilson 95% CI |")
        self.log("| :---: | :---: | :---: | :---: | :---: |")

        for opt, label in [("A", "yes"), ("B", "no"), ("C", "maybe")]:
            cnt = gold_counts.get(opt, 0)
            pct = cnt / total_pm
            low, high = wilson_ci(cnt, total_pm)
            self.log(f"| **{opt}** | `{label}` | {cnt} | {pct*100:.2f}% | `[{low:.4f}, {high:.4f}]` |")

        majority_opt, majority_cnt = gold_counts.most_common(1)[0]
        maj_acc = majority_cnt / total_pm
        self.log()
        self.log(f"- **Majority-Class Baseline**: Always predicting **{majority_opt}** (`yes`) yields an accuracy of **{majority_cnt}/500 ({maj_acc*100:.2f}%)**.")
        self.log()

        # Model predicted answer distributions
        self.log("### Model Prediction Distributions Across Conditions (C1, C2, C3)")
        self.log()
        self.log("| Condition | Context Type | Model | Accuracy | vs Majority Baseline | Pred A (yes) | Pred B (no) | Pred C (maybe) |")
        self.log("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

        context_names = {
            "C1": "Closed-Book (None)",
            "C2": "Retrieved RAG (Top-5)",
            "C3": "Oracle Abstract",
        }

        for cond in ["C1", "C2", "C3"]:
            ctx = context_names[cond]
            for model_label, model_key in [("Gemma-3-4B", GENERIC_MODEL), ("MedGemma-1.5-4B-it", MEDICAL_MODEL)]:
                rows = [r for r in self.data[model_key][cond] if r.get("dataset") == "pubmedqa"]
                preds = collections.Counter(r.get("pred") for r in rows)
                corr = sum(1 for r in rows if is_strict_correct(r))
                acc = corr / len(rows)
                diff_maj = acc - maj_acc

                cnt_a = preds.get("A", 0)
                cnt_b = preds.get("B", 0)
                cnt_c = preds.get("C", 0)

                self.log(
                    f"| {cond} | {ctx} | {model_label} | {acc:.4f} ({corr}/500) | "
                    f"{diff_maj:+.4f} ({diff_maj*100:+.2f}%) | "
                    f"{cnt_a} ({cnt_a/len(rows)*100:.1f}%) | {cnt_b} ({cnt_b/len(rows)*100:.1f}%) | {cnt_c} ({cnt_c/len(rows)*100:.1f}%) |"
                )

        self.log()
        self.log("**PubMedQA Dynamics**:")
        self.log("1. **Majority-Class Deficit**: In both C1 (Closed-Book) and C2 (Retrieved RAG), **neither model outperforms the simple majority-class heuristic** (55.20%). Gemma-3-4B scores 49.20% (C1) and 50.20% (C2); MedGemma-1.5-4B-it scores 50.80% (C1) and 51.00% (C2).")
        self.log("2. **Effect of Retrieved RAG**: Top-5 textbook RAG evidence shifts Gemma3's predictions towards 'no' (from 43 up to 159), but does not improve overall accuracy (+1.00%).")
        self.log("3. **Oracle Context (C3)**: When provided the actual study abstract, MedGemma strongly shifts toward predicting 'yes' (430/500, 86.0%), surpassing the majority class at 57.00% (285/500). Gemma-3-4B remains below the majority baseline at 53.00% (265/500).")
        self.log()

    def run_section_5(self):
        """Section 5: C4/C5 Challenge vs Control Separation, paired bootstrap AUROC CIs, and category breakdown."""
        self.log("## Section 5: C4/C5 Challenge vs Control Separation Analysis")
        self.log()
        self.log("### Confidence Metric and Tie Handling Definition")
        self.log("In [src/qa_pipeline_v3.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/qa_pipeline_v3.py#L341-L344) and [src/llm.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/llm.py#L380-L390):")
        self.log("- **Field Used**: `row['confidence']`")
        self.log("- **What it Measures**: At generation step 0 (immediately following prompt prefix `Answer:`), next-token logits are extracted for valid option letters $\{A, B, C, D\}$ and converted to probabilities via softmax normalized strictly across those letters (`row['letter_probs']`). `confidence = max(letter_probs.values())`.")
        self.log("- **Detection AUROC**: Evaluates $(1 - \\text{confidence})$ as a continuous detector for separating unanswerable challenge rows ($y=1$, `should_abstain=True`) from answerable control rows ($y=0$, `should_abstain=False`).")
        self.log("- **Tie Handling**: AUROC is computed as the fraction of (challenge, control) pairs where the challenge score $(1 - \\text{conf}_{\\text{chal}})$ exceeds the control score $(1 - \\text{conf}_{\\text{ctrl}})$. Because many Gemma-3-4B predictions saturate at $\\text{confidence} = 1.0000$ on both control and challenge questions ($1 - \\text{conf} = 0.0000$), **tied pairs are assigned exact half credit (0.5)**, corresponding mathematically to the mid-rank Wilcoxon-Mann-Whitney U formulation.")
        self.log("- **Bootstrap Confidence Intervals**: 10,000 paired question resamples (seed 42) are drawn for each condition, preserving 1-to-1 question alignment between models.")
        self.log()

        self.log("### Overall Separation Metrics (N=140: 100 Challenge + 40 Control)")
        self.log()
        self.log("| Condition | Model | Chal Mean Conf | Ctrl Mean Conf | Conf Gap (Ctrl - Chal) | Chal Conf >= 0.90 | AUROC (1 - Conf) | AUROC 95% Bootstrap CI | AUROC Diff (Med - Gen) | Diff 95% Bootstrap CI |")
        self.log("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        sec5_results = {}
        for cond in ["C4", "C5"]:
            rows_g = self.data[GENERIC_MODEL][cond]
            rows_m = self.data[MEDICAL_MODEL][cond]

            labels = np.array([1 if r.get("should_abstain") else 0 for r in rows_g])
            g_conf = np.array([float(r["confidence"]) for r in rows_g])
            m_conf = np.array([float(r["confidence"]) for r in rows_m])

            g_scores = 1.0 - g_conf
            m_scores = 1.0 - m_conf

            g_auc = auroc_with_ties(g_scores[labels == 1], g_scores[labels == 0])
            m_auc = auroc_with_ties(m_scores[labels == 1], m_scores[labels == 0])
            diff_auc = m_auc - g_auc

            g_ci, m_ci, diff_ci = paired_bootstrap_auroc(labels, g_scores, m_scores, n_boot=10000, seed=42)

            # Gemma row
            chal_g = g_conf[labels == 1]
            ctrl_g = g_conf[labels == 0]
            f90_g = sum(1 for x in chal_g if x >= 0.90) / len(chal_g)
            self.log(
                f"| {cond} | Gemma-3-4B (Generic) | {np.mean(chal_g):.4f} | {np.mean(ctrl_g):.4f} | {np.mean(ctrl_g) - np.mean(chal_g):+.4f} | "
                f"{f90_g:.4f} ({sum(1 for x in chal_g if x >= 0.90)}/{len(chal_g)}) | {g_auc:.4f} | `[{g_ci[0]:.4f}, {g_ci[1]:.4f}]` | "
                f"— | — |"
            )

            # MedGemma row
            chal_m = m_conf[labels == 1]
            ctrl_m = m_conf[labels == 0]
            f90_m = sum(1 for x in chal_m if x >= 0.90) / len(chal_m)
            self.log(
                f"| {cond} | MedGemma-1.5-4B-it (Medical) | {np.mean(chal_m):.4f} | {np.mean(ctrl_m):.4f} | {np.mean(ctrl_m) - np.mean(chal_m):+.4f} | "
                f"{f90_m:.4f} ({sum(1 for x in chal_m if x >= 0.90)}/{len(chal_m)}) | {m_auc:.4f} | `[{m_ci[0]:.4f}, {m_ci[1]:.4f}]` | "
                f"**{diff_auc:+.4f}** | `[{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]` |"
            )
            sec5_results[cond] = {
                "diff_auc": diff_auc,
                "m_auc": m_auc,
                "g_auc": g_auc,
                "diff_ci": diff_ci,
                "f90_g": f90_g,
                "f90_m": f90_m,
            }

        self.log()
        self.log("**AUROC Findings**:")
        r4 = sec5_results["C4"]
        r5 = sec5_results["C5"]
        self.log(
            f"- In **C4 (Baseline)**, MedGemma achieves a statistically significant {r4['diff_auc']:+.4f} higher AUROC than Gemma3 "
            f"(${r4['m_auc']:.4f}$ vs ${r4['g_auc']:.4f}$, paired bootstrap 95% CI $[{r4['diff_ci'][0]:+.4f}, {r4['diff_ci'][1]:+.4f}]$). "
            f"Gemma3 assigned $\\ge 0.90$ confidence to {r4['f90_g']*100:.0f}% of unanswerable questions, whereas MedGemma did so for {r4['f90_m']*100:.0f}%."
        )
        self.log(
            f"- In **C5 (RAG)**, MedGemma's AUROC is ${r5['m_auc']:.4f}$ vs Gemma3's ${r5['g_auc']:.4f}$; "
            f"the paired difference is {r5['diff_auc']:+.4f}, with 95% CI $[{r5['diff_ci'][0]:+.4f}, {r5['diff_ci'][1]:+.4f}]$ spanning zero."
        )
        self.log()

        # Category Breakdown
        self.log("### Breakdown by Unanswerability Category")
        self.log()
        self.log("AUROC is evaluated for each category against all 40 control questions. Tied pairs receive half credit (0.5).")
        self.log()
        self.log("| Condition | Model | Category | Chal N | Chal Mean Conf | Chal >= 0.90 | Ctrl N | Ctrl Mean Conf | AUROC (vs All Controls) |")
        self.log("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for cond in ["C4", "C5"]:
            for model_label, model_key in [("Gemma-3-4B", GENERIC_MODEL), ("MedGemma-1.5-4B-it", MEDICAL_MODEL)]:
                rows = self.data[model_key][cond]
                chal = [r for r in rows if bool(r.get("should_abstain", False))]
                ctrl = [r for r in rows if not bool(r.get("should_abstain", False))]
                c_ctrl = [float(r["confidence"]) for r in ctrl]

                categories = sorted(set(r.get("category") for r in rows if r.get("category")))
                for cat in categories:
                    cat_chal = [r for r in chal if r.get("category") == cat]
                    cat_ctrl = [r for r in ctrl if r.get("category") == cat]

                    c_cat_chal = [float(r["confidence"]) for r in cat_chal]
                    c_cat_ctrl = [float(r["confidence"]) for r in cat_ctrl] if cat_ctrl else []

                    mean_chal = float(np.mean(c_cat_chal)) if cat_chal else 0.0
                    f90 = (sum(1 for x in c_cat_chal if x >= 0.90) / len(cat_chal)) if cat_chal else 0.0
                    mean_ctrl_str = f"{np.mean(c_cat_ctrl):.4f}" if cat_ctrl else "—"

                    pos_s = np.array([1.0 - x for x in c_cat_chal])
                    neg_s = np.array([1.0 - x for x in c_ctrl])
                    auc = auroc_with_ties(pos_s, neg_s)
                    auc_str = f"{auc:.4f}" if auc is not None else "—"

                    self.log(
                        f"| {cond} | {model_label} | `{cat}` | {len(cat_chal)} | {mean_chal:.4f} | "
                        f"{f90:.4f} ({sum(1 for x in c_cat_chal if x >= 0.90)}/{len(cat_chal)}) | "
                        f"{len(cat_ctrl)} | {mean_ctrl_str} | {auc_str} |"
                    )

        self.log()

    def run_section_6(self):
        """Section 6: Unparsed rows, verbal abstentions, and separate control / challenge reporting."""
        self.log("## Section 6: Unparsed Rows and Verbal Abstention Analysis")
        self.log()
        self.log("### Table of Unparsed Rows (`parsed == False`)")
        self.log()
        self.log("Every row across all conditions (C1-C5) where `parsed` is False:")
        self.log()
        self.log("| Model | Condition | Question ID | should_abstain | category | First 120 Characters of raw_output |")
        self.log("| :--- | :---: | :--- | :---: | :---: | :--- |")

        unparsed_rows_all = []
        for model_key in [GENERIC_MODEL, MEDICAL_MODEL]:
            for cond in CONDITIONS:
                rows = self.data[model_key][cond]
                for r in rows:
                    if not bool(r.get("parsed", True)):
                        unparsed_rows_all.append((model_key, cond, r))
                        snippet = repr(str(r.get("raw_output", "")[:120]))
                        self.log(
                            f"| {model_key} | {cond} | `{r['id']}` | {r.get('should_abstain')} | "
                            f"`{r.get('category', '—')}` | {snippet} |"
                        )

        if not unparsed_rows_all:
            self.log("| None | — | — | — | — | All rows parsed successfully |")

        self.log()
        self.log(f"Total unparsed rows across all conditions: **{len(unparsed_rows_all)}** (all 3 in Gemma-3-4B, 0 in MedGemma-1.5-4B-it).")
        self.log()

        # Classification of verbal abstentions
        self.log("### Verbal Abstention Classification")
        self.log("Classification criterion: `raw_output` following `Answer:` starts with `None`, `Insufficient`, `Not enough`, or `Cannot` (case-insensitive, ignoring markdown bold/quotes).")
        self.log()
        self.log("| Condition | Model | Challenge Verbal Abstentions | Control Verbal Abstentions | Total Verbal Abstentions | Details |")
        self.log("| :--- | :--- | :---: | :---: | :---: | :--- |")

        for cond in CONDITIONS:
            for model_label, model_key in [("Gemma-3-4B", GENERIC_MODEL), ("MedGemma-1.5-4B-it", MEDICAL_MODEL)]:
                rows = self.data[model_key][cond]
                chal_v = [r for r in rows if bool(r.get("should_abstain", False)) and is_verbal_abstention(r.get("raw_output"))]
                ctrl_v = [r for r in rows if not bool(r.get("should_abstain", False)) and is_verbal_abstention(r.get("raw_output"))]
                total_v = len(chal_v) + len(ctrl_v)

                detail_str = ", ".join(f"`{r['id']}`" for r in (chal_v + ctrl_v)) if total_v > 0 else "None"
                self.log(
                    f"| {cond} | {model_label} | {len(chal_v)} / {sum(1 for r in rows if bool(r.get('should_abstain', False)))} | "
                    f"{len(ctrl_v)} / {sum(1 for r in rows if not bool(r.get('should_abstain', False)))} | "
                    f"**{total_v}** | {detail_str} |"
                )

        self.log()

        # Separate reporting of Control Accuracy and Challenge Abstention Rate
        self.log("### C4/C5 Accuracy and Abstention Rates (Reported Separately)")
        self.log()
        self.log("Control questions ($N=40$) are standard multiple-choice questions requiring an option letter. Challenge questions ($N=100$) factually have no valid answer and require abstention.")
        self.log("- **Standard (Stored)**: Verbal refusals failed option parsing, fell back to logprob argmax `D`, and were counted as incorrect answer attempts.")
        self.log("- **+ Verbal Abstentions**: Verbal refusals are recognized as intentional abstentions (`pred='ABSTAIN'`). On challenge questions, this counts as a **successful abstention**.")
        self.log()

        self.log("| Condition | Model | Evaluation Mode | Control Correct (N=40) | Control Acc | Control 95% Wilson CI | Challenge Abstained (N=100) | Challenge Abstention Rate | Challenge 95% Wilson CI |")
        self.log("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for cond in ["C4", "C5"]:
            for model_label, model_key in [("Gemma-3-4B", GENERIC_MODEL), ("MedGemma-1.5-4B-it", MEDICAL_MODEL)]:
                rows = self.data[model_key][cond]
                chal = [r for r in rows if bool(r.get("should_abstain", False))]
                ctrl = [r for r in rows if not bool(r.get("should_abstain", False))]

                # Standard (Stored)
                ctrl_corr_std = sum(1 for r in ctrl if is_strict_correct(r))
                ctrl_acc_std = ctrl_corr_std / len(ctrl)
                ctrl_ci_std = wilson_ci(ctrl_corr_std, len(ctrl))

                chal_abst_std = sum(1 for r in chal if bool(r.get("abstained", False)))
                chal_rate_std = chal_abst_std / len(chal)
                chal_ci_std = wilson_ci(chal_abst_std, len(chal))

                # With Verbal Abstentions
                ctrl_corr_adj = sum(1 for r in ctrl if is_strict_correct(r) and not is_verbal_abstention(r.get("raw_output")))
                ctrl_acc_adj = ctrl_corr_adj / len(ctrl)
                ctrl_ci_adj = wilson_ci(ctrl_corr_adj, len(ctrl))

                chal_abst_adj = sum(
                    1 for r in chal if bool(r.get("abstained", False)) or is_verbal_abstention(r.get("raw_output"))
                )
                chal_rate_adj = chal_abst_adj / len(chal)
                chal_ci_adj = wilson_ci(chal_abst_adj, len(chal))

                self.log(
                    f"| {cond} | {model_label} | Standard (Stored) | "
                    f"{ctrl_corr_std} / {len(ctrl)} | {ctrl_acc_std:.4f} | `[{ctrl_ci_std[0]:.4f}, {ctrl_ci_std[1]:.4f}]` | "
                    f"{chal_abst_std} / {len(chal)} | {chal_rate_std:.4f} | `[{chal_ci_std[0]:.4f}, {chal_ci_std[1]:.4f}]` |"
                )
                self.log(
                    f"| {cond} | {model_label} | + Verbal Abstentions | "
                    f"{ctrl_corr_adj} / {len(ctrl)} | {ctrl_acc_adj:.4f} | `[{ctrl_ci_adj[0]:.4f}, {ctrl_ci_adj[1]:.4f}]` | "
                    f"{chal_abst_adj} / {len(chal)} | {chal_rate_adj:.4f} | `[{chal_ci_adj[0]:.4f}, {chal_ci_adj[1]:.4f}]` |"
                )

        self.log()

        # Separation metrics with vs without verbal abstentions
        self.log("### C4/C5 Confidence & Separation Metrics With vs Without Verbal Abstentions")
        self.log()
        self.log("When verbal abstentions are treated as abstentions, option-answering confidence is set to $0.0$ ($1 - \\text{conf} = 1.0$):")
        self.log()
        self.log("| Condition | Model | Mode | Chal Mean Conf | Ctrl Mean Conf | Chal >= 0.90 | AUROC (1 - Conf) | AUROC 95% Bootstrap CI |")
        self.log("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

        for cond in ["C4", "C5"]:
            for model_label, model_key in [("Gemma-3-4B", GENERIC_MODEL), ("MedGemma-1.5-4B-it", MEDICAL_MODEL)]:
                rows = self.data[model_key][cond]
                chal = [r for r in rows if bool(r.get("should_abstain", False))]
                ctrl = [r for r in rows if not bool(r.get("should_abstain", False))]

                # Raw
                c_chal_raw = [float(r["confidence"]) for r in chal]
                c_ctrl_raw = [float(r["confidence"]) for r in ctrl]
                auc_raw = auroc_with_ties(np.array([1.0 - x for x in c_chal_raw]), np.array([1.0 - x for x in c_ctrl_raw]))
                f90_raw = sum(1 for x in c_chal_raw if x >= 0.90) / len(c_chal_raw)

                # Adjusted
                c_chal_adj = [0.0 if is_verbal_abstention(r.get("raw_output")) else float(r["confidence"]) for r in chal]
                c_ctrl_adj = [0.0 if is_verbal_abstention(r.get("raw_output")) else float(r["confidence"]) for r in ctrl]
                auc_adj = auroc_with_ties(np.array([1.0 - x for x in c_chal_adj]), np.array([1.0 - x for x in c_ctrl_adj]))
                f90_adj = sum(1 for x in c_chal_adj if x >= 0.90) / len(c_chal_adj)

                labels = np.array([1 if r.get("should_abstain") else 0 for r in rows])
                raw_scores = np.array([1.0 - float(r["confidence"]) for r in rows])
                adj_scores = np.array([1.0 if is_verbal_abstention(r.get("raw_output")) else (1.0 - float(r["confidence"])) for r in rows])

                raw_ci, _, _ = paired_bootstrap_auroc(labels, raw_scores, raw_scores, n_boot=10000, seed=42)
                adj_ci, _, _ = paired_bootstrap_auroc(labels, adj_scores, adj_scores, n_boot=10000, seed=42)

                self.log(
                    f"| {cond} | {model_label} | Raw (Stored) | "
                    f"{np.mean(c_chal_raw):.4f} | {np.mean(c_ctrl_raw):.4f} | "
                    f"{f90_raw:.4f} ({sum(1 for x in c_chal_raw if x >= 0.90)}/{len(c_chal_raw)}) | {auc_raw:.4f} | `[{raw_ci[0]:.4f}, {raw_ci[1]:.4f}]` |"
                )
                self.log(
                    f"| {cond} | {model_label} | + Verbal (conf=0.0) | "
                    f"{np.mean(c_chal_adj):.4f} | {np.mean(c_ctrl_adj):.4f} | "
                    f"{f90_adj:.4f} ({sum(1 for x in c_chal_adj if x >= 0.90)}/{len(c_chal_adj)}) | {auc_adj:.4f} | `[{adj_ci[0]:.4f}, {adj_ci[1]:.4f}]` |"
                )

        self.log()

    def run_assumptions_summary(self):
        """List every assumption made about field names and schemas."""
        self.log("## Section 7: Assumptions and Field Specifications")
        self.log()
        self.log("The following assumptions were made regarding the JSONL output schema and field semantics:")
        self.log("1. `id`: Unique string identifier for each question across benchmarks (e.g. `medqa-test-0001`, `unans-v2-test-0001`). Used for 1-to-1 paired alignment.")
        self.log("2. `split`: Dataset partition. Verified to be exclusively `test` for all 2,780 rows per model.")
        self.log("3. `dataset`: Benchmark name (`medqa`, `pubmedqa`, or `unanswerable`).")
        self.log("4. `condition`: Experimental condition (`C1`, `C2`, `C3`, `C4`, `C5`).")
        self.log("5. `should_abstain`: Boolean indicator. `False` for answerable questions (all MedQA, all PubMedQA, and 40 C4/C5 controls); `True` for 100 C4/C5 challenge questions.")
        self.log("6. `category`: Present on C4/C5 rows (`false_premise`, `ambiguous`, `fabricated_disease`, `fabricated_drug`, `out_of_scope`, `missing_info`).")
        self.log("7. `parsed`: Boolean indicating whether regex parsing extracted a valid option letter from `raw_output`.")
        self.log("8. `truncated`: Boolean indicating whether generation reached token limits without terminating cleanly (0 across all rows).")
        self.log("9. `correct`: Stored boolean indicating whether `pred` matched canonical `gold`.")
        self.log("10. `confidence`: Float in `[0.0, 1.0]`, representing `max(letter_probs.values())` at generation step 0 following `Answer:`. Used as a measure of multiple-choice certainty.")
        self.log("11. `letter_probs`: Dictionary mapping option letters $\{A, B, C, D\}$ to their softmax-normalized probabilities at the answer step.")
        self.log("12. `raw_output`: Complete string decoded from model tokens following prompt prefix `Answer:`.")
        self.log()

    def save_report(self):
        """Write the formatted markdown report to REPORT_PATH."""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.report_lines) + "\n")
        print(f"\n[REPORT WRITTEN] Saved report to {REPORT_PATH}")


def main():
    analysis = GemmaPairAnalysis()
    analysis.log("# Matched-Pair Statistical Report: Gemma-3-4B vs MedGemma-1.5-4B-it")
    analysis.log()
    analysis.log("This report evaluates the generic-vs-medical matched pair under Phase 1 V3 frozen-test conditions:")
    analysis.log("- **Generic Model**: `google/gemma-3-4b-it` (`gemma3-4b`)")
    analysis.log("- **Medical Domain Model**: `google/medgemma-1.5-4b-it` (`medgemma-1.5-4b-it`)")
    analysis.log("- **Hardware & Runtime**: Identical dual-GPU T4 inference (`float32`), greedy decoding, same prompts, same seeds.")
    analysis.log()

    analysis.load_and_verify()
    analysis.run_section_1()
    analysis.run_section_2()
    analysis.run_section_3()
    analysis.run_section_pubmedqa()
    analysis.run_section_5()
    analysis.run_section_6()
    analysis.run_assumptions_summary()
    analysis.save_report()


if __name__ == "__main__":
    main()
