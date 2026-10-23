"""src/analysis_pairs.py — Matched-Pair Generic vs Medical-Domain Statistical Analysis.

Implements all pre-registered analyses from docs/DESIGN_GENERIC_VS_MEDICAL.md:
  1. Exactly three matched pairs in the main comparison:
       Pair A: qwen3-1.7b vs medpsy-1.7b
       Pair B: qwen3-4b vs medpsy-4b
       Pair C: gemma3-4b vs medgemma-1.5-4b-it
  2. Matched-pair McNemar exact tests (generic vs medical per pair and condition) with Holm correction.
  3. Bootstrap 95% CIs (1,000 resamples, seed 42) for accuracy and answered-only accuracy.
  4. RAG effect per model (McNemar baseline vs RAG, discordant counts, odds ratios) for matched pairs.
  5. Supplementary generic references (phi4-mini, smollm3-3b) isolated in their own reference table.
  6. Separate thinking-mode control table for qwen3-4b-thinking-2507, labeled budget-limited
     (99.6% truncation at the 1,024-token cap).
  7. Pooled Cochran-Mantel-Haenszel (CMH) tests and logistic interaction test for matched pairs.
  8. Primary strict accuracy (truncated = wrong) alongside answered-only accuracy.
  9. Outputs publication-ready markdown and CSV tables to docs/tables/.
  10. Resilient: skips models with missing outputs without crashing.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.stats as stats

from src.answer_validity import (
    is_strict_correct,
    is_fallback_correct,
    is_truncated,
    is_parsed,
    strict_correct_list,
    fallback_correct_list,
    strict_accuracy,
    fallback_accuracy,
    truncation_rate,
)
from src.common import load_config, read_jsonl


# Pre-registered condition-dataset mapping
CONDITION_DATASETS = {
    "C1": ["medqa", "pubmedqa"],
    "C2": ["medqa", "pubmedqa"],
    "C3": ["pubmedqa"],
}


# ── Statistical primitives ───────────────────────────────────────────────────

def mcnemar_exact(b, c):
    """Exact two-sided McNemar binomial test on discordant pairs (b, c).
    
    b: condition 1 correct, condition 2 wrong
    c: condition 1 wrong, condition 2 correct
    """
    n_discordant = b + c
    if n_discordant == 0:
        return {"p_value": 1.0, "odds_ratio": 1.0, "b": b, "c": c}
    # Two-sided binomial test under H0: p = 0.5
    res = stats.binomtest(k=b, n=n_discordant, p=0.5, alternative="two-sided")
    # Haldane-Anscombe smoothed odds ratio (c / b)
    odds_ratio = (c + 0.5) / (b + 0.5) if (b == 0 or c == 0) else (c / b)
    return {"p_value": float(res.pvalue), "odds_ratio": float(odds_ratio), "b": b, "c": c}


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni step-down correction to a list of p-values.
    
    Preserves input order and handles None values gracefully.
    """
    valid_indices = [i for i, p in enumerate(p_values) if p is not None]
    if not valid_indices:
        return [None] * len(p_values)
    
    m = len(valid_indices)
    sorted_pairs = sorted([(p_values[i], i) for i in valid_indices], key=lambda x: x[0])
    
    adjusted = {}
    cum_max = 0.0
    for rank, (p, orig_idx) in enumerate(sorted_pairs):
        multiplier = m - rank
        adj = min(1.0, p * multiplier)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = min(1.0, cum_max)
        
    return [adjusted.get(i, None) for i in range(len(p_values))]


def bootstrap_ci(binary_correct, n_resamples=1000, seed=42, alpha=0.05):
    """Compute percentile bootstrap 95% CI on binary accuracy."""
    arr = np.asarray(binary_correct, dtype=float)
    if len(arr) == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    boot_means = arr[indices].mean(axis=1)
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(round(low, 4)), float(round(high, 4)))


def cochran_mantel_haenszel(tables):
    """Compute Cochran-Mantel-Haenszel test across a list of 2x2 tables.
    
    Each table is [ [a, b], [c, d] ] where:
      a: RAG correct,   b: RAG wrong
      c: Base correct,  d: Base wrong
    """
    if not tables:
        return {"cmh_stat": None, "p_value": None, "common_or": None}
    
    sum_a = 0.0
    sum_e_a = 0.0
    sum_var_a = 0.0
    num_or = 0.0
    denom_or = 0.0
    
    for tbl in tables:
        a, b = float(tbl[0][0]), float(tbl[0][1])
        c, d = float(tbl[1][0]), float(tbl[1][1])
        n = a + b + c + d
        if n <= 1:
            continue
        e_a = (a + b) * (a + c) / n
        var_a = ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1))
        sum_a += a
        sum_e_a += e_a
        sum_var_a += var_a
        num_or += (a * d) / n
        denom_or += (b * c) / n
        
    if sum_var_a <= 0:
        return {"cmh_stat": None, "p_value": None, "common_or": None}
        
    # Continuity-corrected CMH statistic
    stat = (abs(sum_a - sum_e_a) - 0.5) ** 2 / sum_var_a
    p_val = 1.0 - stats.chi2.cdf(stat, df=1)
    common_or = (num_or / denom_or) if denom_or > 0 else None
    
    return {
        "cmh_stat": float(round(stat, 4)),
        "p_value": float(round(p_val, 6)),
        "common_or": float(round(common_or, 4)) if common_or is not None else None,
    }


def interpret_cmh(cmh_res):
    """Interpret Cochran-Mantel-Haenszel common odds ratio direction and significance."""
    if not cmh_res:
        return "No significant RAG effect"
    p_val = cmh_res.get("p_value")
    common_or = cmh_res.get("common_or")
    if p_val is None or common_or is None or p_val >= 0.05:
        return "No significant RAG effect"
    if common_or > 1.0:
        return "Significant RAG benefit"
    elif common_or < 1.0:
        return "Significant RAG degradation"
    return "No significant RAG effect"


def fit_logistic_interaction(y, x_med, x_rag):
    """Fit logistic regression y ~ 1 + x_med + x_rag + x_med*x_rag using Newton-Raphson.
    
    Returns Wald z, Wald p, interaction coefficient beta_3, and Likelihood Ratio p-value.
    """
    y = np.asarray(y, dtype=float)
    x_med = np.asarray(x_med, dtype=float)
    x_rag = np.asarray(x_rag, dtype=float)
    x_int = x_med * x_rag
    n = len(y)
    
    if n < 10 or y.sum() == 0 or y.sum() == n:
        return {"beta_int": None, "se_int": None, "wald_p": None, "lr_p": None}
        
    X_full = np.column_stack([np.ones(n), x_med, x_rag, x_int])
    X_red = np.column_stack([np.ones(n), x_med, x_rag])
    
    def _irls(X):
        p = X.shape[1]
        beta = np.zeros(p)
        for _ in range(50):
            eta = np.clip(X @ beta, -20, 20)
            pi = 1.0 / (1.0 + np.exp(-eta))
            pi = np.clip(pi, 1e-7, 1 - 1e-7)
            W = pi * (1.0 - pi)
            grad = X.T @ (y - pi)
            hess = X.T @ (X * W[:, None])
            try:
                delta = np.linalg.solve(hess + 1e-6 * np.eye(p), grad)
            except np.linalg.LinAlgError:
                return beta, None, None
            beta += delta
            if np.max(np.abs(delta)) < 1e-6:
                break
        eta = np.clip(X @ beta, -20, 20)
        pi = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-7, 1 - 1e-7)
        W = pi * (1.0 - pi)
        hess = X.T @ (X * W[:, None])
        try:
            cov = np.linalg.inv(hess + 1e-6 * np.eye(p))
        except np.linalg.LinAlgError:
            cov = None
        ll = np.sum(y * np.log(pi) + (1.0 - y) * np.log(1.0 - pi))
        return beta, cov, ll

    b_full, cov_full, ll_full = _irls(X_full)
    b_red, cov_red, ll_red = _irls(X_red)
    
    if cov_full is None or ll_full is None or ll_red is None:
        return {"beta_int": None, "se_int": None, "wald_p": None, "lr_p": None}
        
    se_int = np.sqrt(max(0.0, cov_full[3, 3]))
    beta_int = b_full[3]
    wald_z = beta_int / se_int if se_int > 0 else 0.0
    wald_p = 2.0 * (1.0 - stats.norm.cdf(abs(wald_z)))
    
    lr_stat = max(0.0, 2.0 * (ll_full - ll_red))
    lr_p = 1.0 - stats.chi2.cdf(lr_stat, df=1)
    
    return {
        "beta_int": float(round(beta_int, 4)),
        "se_int": float(round(se_int, 4)),
        "wald_p": float(round(wald_p, 6)),
        "lr_stat": float(round(lr_stat, 4)),
        "lr_p": float(round(lr_p, 6)),
    }


# ── File resolution ────────────────────────────────────────────────────────────

def resolve_model_rows(model_name, condition, root_dir="."):
    """Locate and read output JSONL rows for a model and condition.
    
    Checks v3 output paths first, falling back to legacy paths when available.
    For legacy paths, aggregates rows across matching dataset-specific files to ensure
    models evaluated across split runner runs (e.g. MedQA vs PubMedQA) are fully resolved.
    Verifies that every loaded row's model field matches the requested model_name.
    """
    allowed_models = {model_name}
    if model_name in ("medgemma-4b", "medgemma-1.5-4b-it"):
        allowed_models.update(["medgemma-4b", "medgemma-1.5-4b-it"])

    def _validate_row(r, file_path):
        row_model = r.get("model")
        if row_model is None and condition == "C4":
            return
        if row_model not in allowed_models:
            raise ValueError(
                f"Model mismatch in file '{file_path}': expected '{model_name}', "
                f"found '{row_model}'"
            )

    v3_candidates = [
        os.path.join(root_dir, "outputs", "v3", model_name, condition, f"{model_name}.jsonl"),
        os.path.join(root_dir, "outputs", "v3", model_name, f"{condition}.jsonl"),
        os.path.join("/kaggle/working/outputs/v3", model_name, condition, f"{model_name}.jsonl"),
    ]

    # Handle model aliases like medgemma-4b <-> medgemma-1.5-4b-it
    if model_name in ("medgemma-4b", "medgemma-1.5-4b-it"):
        alt_name = "medgemma-1.5-4b-it" if model_name == "medgemma-4b" else "medgemma-4b"
        v3_candidates.extend([
            os.path.join(root_dir, "outputs", "v3", alt_name, condition, f"{alt_name}.jsonl"),
            os.path.join(root_dir, "outputs", "v3", alt_name, f"{condition}.jsonl"),
        ])

    for p in v3_candidates:
        if os.path.exists(p):
            try:
                rows = read_jsonl(p)
            except Exception:
                continue
            if rows:
                for r in rows:
                    _validate_row(r, p)
                return rows

    # Legacy fallbacks: derived strictly from requested model_name
    legacy_candidates = []
    if condition == "C1":
        legacy_candidates.extend([
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", model_name, "work", "phase5", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_final", "work", "phase5", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_pubmedqa_baseline", f"{model_name}_pubmedqa_baseline.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_pubmedqa_baseline", f"{model_name}.jsonl"),
        ])
        if model_name in ("medgemma-4b", "medgemma-1.5-4b-it"):
            alt_name = "medgemma-1.5-4b-it" if model_name == "medgemma-4b" else "medgemma-4b"
            legacy_candidates.extend([
                os.path.join(root_dir, "outputs", "kaggle_qa", "full", alt_name, "work", "phase5", f"{alt_name}.jsonl"),
            ])
    elif condition == "C2":
        legacy_candidates.extend([
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", model_name, "work", "phase8", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", f"{model_name}_rag", "work", "phase8", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_rag", "work", "phase8", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_pubmedqa_rag", f"{model_name}_pubmedqa_rag.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_pubmedqa_rag", f"{model_name}.jsonl"),
        ])
        if model_name in ("medgemma-4b", "medgemma-1.5-4b-it"):
            alt_name = "medgemma-1.5-4b-it" if model_name == "medgemma-4b" else "medgemma-4b"
            legacy_candidates.extend([
                os.path.join(root_dir, "outputs", "kaggle_qa", "full", alt_name, "work", "phase8", f"{alt_name}.jsonl"),
                os.path.join(root_dir, "outputs", "kaggle_qa", "full", f"{alt_name}_rag", "work", "phase8", f"{alt_name}.jsonl"),
            ])
    elif condition == "C3":
        legacy_candidates.extend([
            os.path.join(root_dir, "outputs", "kaggle_qa", "context", "work", "phase10", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_final", "work", "phase10", f"{model_name}.jsonl"),
        ])
    elif condition == "C4":
        legacy_candidates.extend([
            os.path.join(root_dir, "outputs", "sc_test_s3_unanswerable", "outputs", "sc_test", model_name, "unanswerable_v2.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "unans_v2", model_name, f"baseline_{model_name}.jsonl"),
        ])

    seen_ids = set()
    combined_rows = []
    for p in legacy_candidates:
        if os.path.exists(p):
            try:
                file_rows = read_jsonl(p)
            except Exception:
                continue
            for r in file_rows:
                _validate_row(r, p)
                qid = r.get("id")
                if qid and qid not in seen_ids:
                    seen_ids.add(qid)
                    combined_rows.append(r)
                elif not qid:
                    combined_rows.append(r)

    if combined_rows:
        return combined_rows
    return None


def extract_evaluation_map(rows):
    """Index answerable rows by question ID, extracting correctness, truncation, and confidence.
    
    In accordance with the pre-registered protocol:
      - Primary Rule: Strict correctness (not truncated and parsed and correct).
      - Secondary Rule: With-fallback correctness (raw correct field).
      - Answered-only accuracy considers only non-truncated questions.
      - Split Rule: Filter to the frozen TEST split once, at load time. Validation
        rows are strictly excluded and never enter model_data or any output table.
    """
    res = {}
    for r in rows:
        qid = r.get("id")
        if not qid:
            continue
        # Answerable filter: ignore unanswerable questions when computing standard benchmark accuracy
        if r.get("should_abstain"):
            continue

        # Split filter: strictly exclude validation rows at load time
        split = str(r.get("split") or "test").strip().lower()
        if split != "test":
            continue
        
        is_trunc = is_truncated(r)
        is_prsd = is_parsed(r)
        strict_corr = int(is_strict_correct(r))
        fallback_corr = int(is_fallback_correct(r))
        
        res[qid] = {
            "correct_strict": strict_corr,
            "correct_primary": strict_corr,
            "correct_fallback": fallback_corr,
            "raw_correct": fallback_corr,
            "truncated": is_trunc,
            "parsed": is_prsd,
            "confidence": float(r.get("confidence", 0.0) or 0.0),
            "seconds": float(r.get("seconds", 0.0) or 0.0),
            "dataset": str(r.get("dataset") or "medqa").strip().lower(),
            "split": split,
        }
    return res


# ── Table writers ──────────────────────────────────────────────────────────────

def write_table_markdown_and_csv(table_name, headers, rows, out_dir="docs/tables"):
    """Write tabular data to both Markdown and CSV files."""
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{table_name}.md")
    csv_path = os.path.join(out_dir, f"{table_name}.csv")
    
    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
        
    # Write Markdown
    col_widths = [len(h) for h in headers]
    str_rows = []
    for r in rows:
        row_strs = [str(val) if val is not None else "-" for val in r]
        str_rows.append(row_strs)
        for i, val_s in enumerate(row_strs):
            col_widths[i] = max(col_widths[i], len(val_s))
            
    with open(md_path, "w", encoding="utf-8") as fh:
        # Header
        fh.write("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |\n")
        fh.write("| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |\n")
        # Rows
        for r_s in str_rows:
            fh.write("| " + " | ".join(r_s[i].ljust(col_widths[i]) for i in range(len(headers))) + " |\n")
            
    print(f"Saved: {md_path} and {csv_path}")


# ── Core Analysis Routine ──────────────────────────────────────────────────────

def run_analysis(config_path="configs/base.yaml", root_dir="."):
    """Execute complete matched-pair comparative statistical analysis."""
    # Canonical Scope: exactly three matched pairs
    pairs_cfg = {
        "Pair A": {"name": "1.7B (Qwen)", "generic": "qwen3-1.7b", "medical": "medpsy-1.7b"},
        "Pair B": {"name": "4B (Qwen)", "generic": "qwen3-4b", "medical": "medpsy-4b"},
        "Pair C": {"name": "4B (Gemma)", "generic": "gemma3-4b", "medical": "medgemma-1.5-4b-it"},
    }
    thinking_control = "qwen3-4b-thinking-2507"
    supp_generic_models = ["phi4-mini", "smollm3-3b"]
    
    print("\n" + "=" * 80)
    print("MEDRAG-SLM V3: MATCHED-PAIR GENERIC VS MEDICAL ANALYSIS")
    print("Main comparison: Pair A (1.7B), Pair B (4B Qwen), Pair C (4B Gemma)")
    print("Thinking control: Qwen3-4B-Thinking-2507 (separate control table)")
    print("Supplementary generic references: Phi-4-mini, SmolLM3-3B (separate reference table)")
    print("=" * 80 + "\n")
    
    # ── 1. Load outputs across models & conditions ──────────────────────────────
    matched_models = set()
    for p in pairs_cfg.values():
        matched_models.add(p["generic"])
        matched_models.add(p["medical"])

    all_models = set(matched_models)
    all_models.add(thinking_control)
    all_models.update(supp_generic_models)
            
    conditions = ["C1", "C2", "C3", "C4", "C5"]
    model_data = {}
    for m in sorted(all_models):
        model_data[m] = {}
        for c in conditions:
            rows = resolve_model_rows(m, c, root_dir)
            if rows:
                model_data[m][c] = extract_evaluation_map(rows)
            else:
                model_data[m][c] = None

    # Status summary
    print("Loaded model output availability:")
    for m in sorted(all_models):
        avail = [c for c in conditions if model_data[m].get(c) is not None]
        print(f"  {m:<24}: {', '.join(avail) if avail else 'NONE (pending)'}")
    print()

    # ── 2. RQ1: Matched Pairs Within-Pair McNemar + Holm (Main 3 Pairs Only) ───
    rq1_headers = [
        "Pair", "Arm Comparison", "Condition", "Dataset", "N Common",
        "Generic Strict Acc (95% CI)", "Generic With-Fallback Acc",
        "Medical Strict Acc (95% CI)", "Medical With-Fallback Acc",
        "Delta Strict Acc", "b/c Discordant (Strict)", "McNemar p (exact)", "Holm p-adj", "Supported (p<0.05)?"
    ]
    rq1_rows = []
    p_values_to_correct = []
    pending_row_indices = []

    for pid in ["Pair A", "Pair B", "Pair C"]:
        pinfo = pairs_cfg.get(pid)
        if not pinfo:
            continue
        gen_m = pinfo["generic"]
        med_m = pinfo["medical"]
        pname = pinfo.get("name", pid)
        
        for c in ["C1", "C2", "C3"]:
            expected_datasets = CONDITION_DATASETS.get(c, ["medqa", "pubmedqa"])
            for d in expected_datasets:
                gen_c = model_data.get(gen_m, {}).get(c)
                med_c = model_data.get(med_m, {}).get(c)
                
                gen_items = {q: v for q, v in gen_c.items() if v.get("dataset") == d} if gen_c else {}
                med_items = {q: v for q, v in med_c.items() if v.get("dataset") == d} if med_c else {}
                
                if not gen_items and not med_items:
                    rq1_rows.append([
                        f"{pid} ({pname})", f"{gen_m} vs {med_m}", c, d, "-",
                        "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({gen_m}, {med_m} {d} missing)"
                    ])
                    continue
                if not gen_items:
                    rq1_rows.append([
                        f"{pid} ({pname})", f"{gen_m} vs {med_m}", c, d, "-",
                        "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({gen_m} {d} missing)"
                    ])
                    continue
                if not med_items:
                    rq1_rows.append([
                        f"{pid} ({pname})", f"{gen_m} vs {med_m}", c, d, "-",
                        "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({med_m} {d} missing)"
                    ])
                    continue
                    
                common_qids = sorted(set(gen_items.keys()) & set(med_items.keys()))
                if not common_qids:
                    rq1_rows.append([
                        f"{pid} ({pname})", f"{gen_m} vs {med_m}", c, d, "-",
                        "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING (no common {d} questions)"
                    ])
                    continue
                    
                gen_correct = [gen_items[q]["correct_strict"] for q in common_qids]
                gen_fb = [gen_items[q]["correct_fallback"] for q in common_qids]
                med_correct = [med_items[q]["correct_strict"] for q in common_qids]
                med_fb = [med_items[q]["correct_fallback"] for q in common_qids]
                
                gen_acc = np.mean(gen_correct)
                gen_fb_acc = np.mean(gen_fb)
                med_acc = np.mean(med_correct)
                med_fb_acc = np.mean(med_fb)
                gen_ci = bootstrap_ci(gen_correct)
                med_ci = bootstrap_ci(med_correct)
                
                # Discordant counts: b = gen correct, med wrong; c = gen wrong, med correct
                b = sum(1 for g, m in zip(gen_correct, med_correct) if g == 1 and m == 0)
                c_disc = sum(1 for g, m in zip(gen_correct, med_correct) if g == 0 and m == 1)
                
                mcn = mcnemar_exact(b, c_disc)
                p_val = mcn["p_value"]
                
                idx = len(rq1_rows)
                pending_row_indices.append(idx)
                p_values_to_correct.append(p_val)
                
                rq1_rows.append([
                    f"{pid} ({pname})",
                    f"{gen_m} vs {med_m}",
                    c,
                    d,
                    len(common_qids),
                    f"{gen_acc:.4f} [{gen_ci[0]}, {gen_ci[1]}]",
                    f"{gen_fb_acc:.4f}",
                    f"{med_acc:.4f} [{med_ci[0]}, {med_ci[1]}]",
                    f"{med_fb_acc:.4f}",
                    f"{med_acc - gen_acc:+.4f}",
                    f"{b}/{c_disc}",
                    f"{p_val:.5f}",
                    None,  # placeholder for Holm adjustment
                    None,
                ])

    # Apply Holm correction strictly across the evaluated matched pair rows
    adj_ps = holm_bonferroni(p_values_to_correct)
    for row_idx, adj_p in zip(pending_row_indices, adj_ps):
        delta = float(rq1_rows[row_idx][9])
        rq1_rows[row_idx][12] = f"{adj_p:.5f}"
        supported = "YES" if (adj_p < 0.05 and delta > 0) else "NO"
        rq1_rows[row_idx][13] = supported

    write_table_markdown_and_csv("table_matched_pairs_rq1", rq1_headers, rq1_rows)

    # ── 2B. Thinking-Mode Control Table (Separate from Main Comparison) ─────────
    ctrl_headers = [
        "Control Arm", "Arm Comparison", "Condition", "Dataset", "N Common",
        "Thinking Control Strict Acc (95% CI)", "Thinking Control With-Fallback Acc",
        "Medical Strict Acc (95% CI)", "Medical With-Fallback Acc",
        "Delta Strict Acc", "b/c Discordant (Strict)", "McNemar p (exact)", "Evaluation Note"
    ]
    ctrl_rows = []
    med_m_b = pairs_cfg["Pair B"]["medical"]

    for c in ["C1", "C2", "C3"]:
        expected_datasets = CONDITION_DATASETS.get(c, ["medqa", "pubmedqa"])
        for d in expected_datasets:
            ctrl_c = model_data.get(thinking_control, {}).get(c)
            med_c = model_data.get(med_m_b, {}).get(c)

            ctrl_items = {q: v for q, v in ctrl_c.items() if v.get("dataset") == d} if ctrl_c else {}
            med_items = {q: v for q, v in med_c.items() if v.get("dataset") == d} if med_c else {}

            if not ctrl_items and not med_items:
                ctrl_rows.append([
                    "Pair B Control", f"{thinking_control} vs {med_m_b}", c, d, "-",
                    "-", "-", "-", "-", "-", "-", "-", f"PENDING ({thinking_control}, {med_m_b} {d} missing)"
                ])
                continue
            if not ctrl_items:
                ctrl_rows.append([
                    "Pair B Control", f"{thinking_control} vs {med_m_b}", c, d, "-",
                    "-", "-", "-", "-", "-", "-", "-", f"PENDING ({thinking_control} {d} missing)"
                ])
                continue
            if not med_items:
                ctrl_rows.append([
                    "Pair B Control", f"{thinking_control} vs {med_m_b}", c, d, "-",
                    "-", "-", "-", "-", "-", "-", "-", f"PENDING ({med_m_b} {d} missing)"
                ])
                continue

            ctrl_common = sorted(set(ctrl_items.keys()) & set(med_items.keys()))
            if not ctrl_common:
                ctrl_rows.append([
                    "Pair B Control", f"{thinking_control} vs {med_m_b}", c, d, "-",
                    "-", "-", "-", "-", "-", "-", "-", f"PENDING (no common {d} questions)"
                ])
                continue

            c_corr = [ctrl_items[q]["correct_strict"] for q in ctrl_common]
            c_fb = [ctrl_items[q]["correct_fallback"] for q in ctrl_common]
            m_corr = [med_items[q]["correct_strict"] for q in ctrl_common]
            m_fb = [med_items[q]["correct_fallback"] for q in ctrl_common]

            c_acc = np.mean(c_corr)
            c_fb_acc = np.mean(c_fb)
            m_acc = np.mean(m_corr)
            m_fb_acc = np.mean(m_fb)
            c_ci = bootstrap_ci(c_corr)
            m_ci = bootstrap_ci(m_corr)

            b_ctrl = sum(1 for g, m in zip(c_corr, m_corr) if g == 1 and m == 0)
            c_ctrl = sum(1 for g, m in zip(c_corr, m_corr) if g == 0 and m == 1)
            mcn_c = mcnemar_exact(b_ctrl, c_ctrl)

            ctrl_rows.append([
                "Pair B Control",
                f"{thinking_control} vs {med_m_b}",
                c,
                d,
                len(ctrl_common),
                f"{c_acc:.4f} [{c_ci[0]}, {c_ci[1]}]",
                f"{c_fb_acc:.4f}",
                f"{m_acc:.4f} [{m_ci[0]}, {m_ci[1]}]",
                f"{m_fb_acc:.4f}",
                f"{m_acc - c_acc:+.4f}",
                f"{b_ctrl}/{c_ctrl}",
                f"{mcn_c['p_value']:.5f}",
                "Budget-limited: 99.6% truncation at 1,024-token cap (not a knowledge deficit)",
            ])

    write_table_markdown_and_csv("table_thinking_control", ctrl_headers, ctrl_rows)

    # ── 3. RQ2: RAG Effect Per Model (Main Matched Pairs Only) ─────────────────
    rq2_headers = [
        "Model", "Matched Pair", "Group", "Dataset", "N Common",
        "Baseline Strict Acc (95% CI)", "Baseline With-Fallback Acc",
        "RAG Strict Acc (95% CI)", "RAG With-Fallback Acc",
        "Delta Strict Acc", "b (Base+ / RAG-)", "c (Base- / RAG+)", "Discordant OR", "McNemar p (exact)"
    ]
    rq2_rows = []
    
    generic_models = [pinfo["generic"] for pinfo in pairs_cfg.values()]
    medical_models = [pinfo["medical"] for pinfo in pairs_cfg.values()]

    model_to_pair = {}
    for pid, pinfo in pairs_cfg.items():
        model_to_pair[pinfo["generic"]] = pid
        model_to_pair[pinfo["medical"]] = pid

    for m in sorted(matched_models):
        group = "Medical" if m in medical_models else "Generic"
        pair_lbl = model_to_pair.get(m, "-")
        c1_map = model_data.get(m, {}).get("C1")
        c2_map = model_data.get(m, {}).get("C2")
        
        for d in ["medqa", "pubmedqa"]:
            c1_d = {q: v for q, v in c1_map.items() if v.get("dataset") == d} if c1_map else {}
            c2_d = {q: v for q, v in c2_map.items() if v.get("dataset") == d} if c2_map else {}
            
            if not c1_d and not c2_d:
                rq2_rows.append([m, pair_lbl, group, d, "-", "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({m} {d} C1/C2 missing)"])
                continue
            if not c1_d:
                rq2_rows.append([m, pair_lbl, group, d, "-", "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({m} {d} C1 missing)"])
                continue
            if not c2_d:
                rq2_rows.append([m, pair_lbl, group, d, "-", "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({m} {d} C2 missing)"])
                continue
                
            common_qids = sorted(set(c1_d.keys()) & set(c2_d.keys()))
            if not common_qids:
                rq2_rows.append([m, pair_lbl, group, d, "-", "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING (no common {d} C1/C2 questions)"])
                continue
                
            c1_corr = [c1_d[q]["correct_strict"] for q in common_qids]
            c1_fb = [c1_d[q]["correct_fallback"] for q in common_qids]
            c2_corr = [c2_d[q]["correct_strict"] for q in common_qids]
            c2_fb = [c2_d[q]["correct_fallback"] for q in common_qids]
            
            c1_acc = np.mean(c1_corr)
            c1_fb_acc = np.mean(c1_fb)
            c2_acc = np.mean(c2_corr)
            c2_fb_acc = np.mean(c2_fb)
            c1_ci = bootstrap_ci(c1_corr)
            c2_ci = bootstrap_ci(c2_corr)
            
            # b = Base correct, RAG wrong; c = Base wrong, RAG correct (Strict Rule)
            b = sum(1 for b_ans, r_ans in zip(c1_corr, c2_corr) if b_ans == 1 and r_ans == 0)
            c_disc = sum(1 for b_ans, r_ans in zip(c1_corr, c2_corr) if b_ans == 0 and r_ans == 1)
            mcn = mcnemar_exact(b, c_disc)
            
            rq2_rows.append([
                m,
                pair_lbl,
                group,
                d,
                len(common_qids),
                f"{c1_acc:.4f} [{c1_ci[0]}, {c1_ci[1]}]",
                f"{c1_fb_acc:.4f}",
                f"{c2_acc:.4f} [{c2_ci[0]}, {c2_ci[1]}]",
                f"{c2_fb_acc:.4f}",
                f"{c2_acc - c1_acc:+.4f}",
                b,
                c_disc,
                f"{mcn['odds_ratio']:.3f}",
                f"{mcn['p_value']:.5f}",
            ])

    write_table_markdown_and_csv("table_rag_effect_rq2", rq2_headers, rq2_rows)

    # ── 3B. Supplementary Generic Reference Table ─────────────────────────────
    supp_headers = [
        "Model", "Role", "Dataset", "N Common",
        "Baseline Strict Acc (95% CI)", "Baseline With-Fallback Acc",
        "RAG Strict Acc (95% CI)", "RAG With-Fallback Acc",
        "Delta Strict Acc", "b (Base+ / RAG-)", "c (Base- / RAG+)", "Discordant OR", "McNemar p (exact)"
    ]
    supp_rows = []

    for m in supp_generic_models:
        c1_map = model_data.get(m, {}).get("C1")
        c2_map = model_data.get(m, {}).get("C2")
        
        for d in ["medqa", "pubmedqa"]:
            c1_d = {q: v for q, v in c1_map.items() if v.get("dataset") == d} if c1_map else {}
            c2_d = {q: v for q, v in c2_map.items() if v.get("dataset") == d} if c2_map else {}
            
            if not c1_d and not c2_d:
                supp_rows.append([m, "Supplementary Generic Reference", d, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({m} {d} C1/C2 missing)"])
                continue
            if not c1_d:
                supp_rows.append([m, "Supplementary Generic Reference", d, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({m} {d} C1 missing)"])
                continue
            if not c2_d:
                supp_rows.append([m, "Supplementary Generic Reference", d, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING ({m} {d} C2 missing)"])
                continue

            common_qids = sorted(set(c1_d.keys()) & set(c2_d.keys()))
            if not common_qids:
                supp_rows.append([m, "Supplementary Generic Reference", d, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", f"PENDING (no common {d} C1/C2 questions)"])
                continue

            c1_corr = [c1_d[q]["correct_strict"] for q in common_qids]
            c1_fb = [c1_d[q]["correct_fallback"] for q in common_qids]
            c2_corr = [c2_d[q]["correct_strict"] for q in common_qids]
            c2_fb = [c2_d[q]["correct_fallback"] for q in common_qids]

            c1_acc = np.mean(c1_corr)
            c1_fb_acc = np.mean(c1_fb)
            c2_acc = np.mean(c2_corr)
            c2_fb_acc = np.mean(c2_fb)
            c1_ci = bootstrap_ci(c1_corr)
            c2_ci = bootstrap_ci(c2_corr)

            b = sum(1 for b_ans, r_ans in zip(c1_corr, c2_corr) if b_ans == 1 and r_ans == 0)
            c_disc = sum(1 for b_ans, r_ans in zip(c1_corr, c2_corr) if b_ans == 0 and r_ans == 1)
            mcn = mcnemar_exact(b, c_disc)

            supp_rows.append([
                m,
                "Supplementary Generic Reference",
                d,
                len(common_qids),
                f"{c1_acc:.4f} [{c1_ci[0]}, {c1_ci[1]}]",
                f"{c1_fb_acc:.4f}",
                f"{c2_acc:.4f} [{c2_ci[0]}, {c2_ci[1]}]",
                f"{c2_fb_acc:.4f}",
                f"{c2_acc - c1_acc:+.4f}",
                b,
                c_disc,
                f"{mcn['odds_ratio']:.3f}",
                f"{mcn['p_value']:.5f}",
            ])

    write_table_markdown_and_csv("table_supplementary_generic_references", supp_headers, supp_rows)

    # ── 4. Group CMH & Interaction Test (Strictly Matched Pairs, Per-Dataset) ──
    cmh_headers = [
        "Analysis Layer", "Dataset", "Stratification / Group", "N Generic", "N Medical",
        "Common Odds Ratio", "CMH / Stat", "p-value", "Interpretation"
    ]
    cmh_rows = []

    print("\n--- CMH & Logistic Interaction Strata Verification ---")
    for d in ["medqa", "pubmedqa"]:
        active_pairs = []
        cmh_generic_tables = []
        cmh_medical_tables = []
        
        inter_y = []
        inter_med = []
        inter_rag = []
        
        n_gen_stratum = 0
        n_med_stratum = 0

        for pid in ["Pair A", "Pair B", "Pair C"]:
            pinfo = pairs_cfg[pid]
            gen_m = pinfo["generic"]
            med_m = pinfo["medical"]
            
            gen_c1 = {q: v for q, v in model_data.get(gen_m, {}).get("C1", {}).items() if v.get("dataset") == d} if model_data.get(gen_m, {}).get("C1") else {}
            gen_c2 = {q: v for q, v in model_data.get(gen_m, {}).get("C2", {}).items() if v.get("dataset") == d} if model_data.get(gen_m, {}).get("C2") else {}
            med_c1 = {q: v for q, v in model_data.get(med_m, {}).get("C1", {}).items() if v.get("dataset") == d} if model_data.get(med_m, {}).get("C1") else {}
            med_c2 = {q: v for q, v in model_data.get(med_m, {}).get("C2", {}).items() if v.get("dataset") == d} if model_data.get(med_m, {}).get("C2") else {}
            
            # Matched pair requirement: both arms must have complete C1 and C2 on this dataset
            if not gen_c1 or not gen_c2 or not med_c1 or not med_c2:
                continue
                
            # Compute exact common question set for this pair across both models and both conditions
            pair_qids = sorted(set(gen_c1.keys()) & set(gen_c2.keys()) & set(med_c1.keys()) & set(med_c2.keys()))
            if not pair_qids:
                continue
                
            active_pairs.append(f"{pid} ({gen_m} vs {med_m})")
            n_pair = len(pair_qids)
            n_gen_stratum += n_pair
            n_med_stratum += n_pair
            
            # Generic 2x2 table: [[RAG correct, RAG wrong], [Base correct, Base wrong]]
            g_c1 = [gen_c1[q]["correct_strict"] for q in pair_qids]
            g_c2 = [gen_c2[q]["correct_strict"] for q in pair_qids]
            cmh_generic_tables.append([
                [sum(g_c2), n_pair - sum(g_c2)],
                [sum(g_c1), n_pair - sum(g_c1)],
            ])
            
            # Medical 2x2 table: [[RAG correct, RAG wrong], [Base correct, Base wrong]]
            m_c1 = [med_c1[q]["correct_strict"] for q in pair_qids]
            m_c2 = [med_c2[q]["correct_strict"] for q in pair_qids]
            cmh_medical_tables.append([
                [sum(m_c2), n_pair - sum(m_c2)],
                [sum(m_c1), n_pair - sum(m_c1)],
            ])
            
            # Logistic interaction data points (strictly matched questions)
            for y in g_c1:
                inter_y.append(y); inter_med.append(0.0); inter_rag.append(0.0)
            for y in g_c2:
                inter_y.append(y); inter_med.append(0.0); inter_rag.append(1.0)
            for y in m_c1:
                inter_y.append(y); inter_med.append(1.0); inter_rag.append(0.0)
            for y in m_c2:
                inter_y.append(y); inter_med.append(1.0); inter_rag.append(1.0)

        # Print stratum verification to stdout
        pairs_str = ", ".join(active_pairs) if active_pairs else "NONE (pending outputs)"
        print(f"  Dataset: {d:<10} | Stratum N (Generic): {n_gen_stratum} | Stratum N (Medical): {n_med_stratum} | Active Pairs: {pairs_str}")
        
        # Verify identical question counts between strata
        if n_gen_stratum != n_med_stratum:
            print(f"  WARNING: Strata N mismatch for {d}! Generic={n_gen_stratum}, Medical={n_med_stratum}")
            
        if not active_pairs or len(cmh_generic_tables) == 0 or len(cmh_medical_tables) == 0:
            cmh_rows.append([
                "Pooled CMH Test",
                d,
                "Matched Generic Group",
                "-", "-", "-", "-", "-",
                f"PENDING (no complete matched pairs for {d})"
            ])
            cmh_rows.append([
                "Pooled CMH Test",
                d,
                "Matched Medical Group",
                "-", "-", "-", "-", "-",
                f"PENDING (no complete matched pairs for {d})"
            ])
            cmh_rows.append([
                "Logistic Interaction (Model_Type x RAG)",
                d,
                "Generic vs Medical Differential RAG Gain",
                "-", "-", "-", "-", "-",
                f"PENDING (no complete matched pairs for {d})"
            ])
            continue

        cmh_gen = cochran_mantel_haenszel(cmh_generic_tables)
        cmh_med = cochran_mantel_haenszel(cmh_medical_tables)
        inter_res = fit_logistic_interaction(inter_y, inter_med, inter_rag)

        pairs_summary = f"({', '.join(active_pairs)})"
        cmh_rows.append([
            "Pooled CMH Test",
            d,
            f"Matched Generic Group {pairs_summary}",
            n_gen_stratum,
            n_med_stratum,
            f"{cmh_gen['common_or']}" if cmh_gen["common_or"] is not None else "-",
            f"Chi2={cmh_gen['cmh_stat']}" if cmh_gen["cmh_stat"] is not None else "-",
            f"{cmh_gen['p_value']}" if cmh_gen["p_value"] is not None else "-",
            interpret_cmh(cmh_gen),
        ])
        cmh_rows.append([
            "Pooled CMH Test",
            d,
            f"Matched Medical Group {pairs_summary}",
            n_gen_stratum,
            n_med_stratum,
            f"{cmh_med['common_or']}" if cmh_med["common_or"] is not None else "-",
            f"Chi2={cmh_med['cmh_stat']}" if cmh_med["cmh_stat"] is not None else "-",
            f"{cmh_med['p_value']}" if cmh_med["p_value"] is not None else "-",
            interpret_cmh(cmh_med),
        ])
        cmh_rows.append([
            "Logistic Interaction (Model_Type x RAG)",
            d,
            f"Generic vs Medical Differential RAG Gain {pairs_summary}",
            n_gen_stratum,
            n_med_stratum,
            f"Beta_int={inter_res['beta_int']}" if inter_res["beta_int"] is not None else "-",
            f"LR Chi2={inter_res.get('lr_stat')}" if inter_res.get("lr_stat") is not None else "-",
            f"Wald p={inter_res['wald_p']}, LR p={inter_res['lr_p']}" if inter_res["wald_p"] is not None else "-",
            "Differential RAG gain confirmed" if (inter_res.get("lr_p") is not None and inter_res["lr_p"] < 0.05) else "No differential gain / pending",
        ])

    print("------------------------------------------------------\n")
    write_table_markdown_and_csv("table_group_cmh_interaction", cmh_headers, cmh_rows)

    # ── 5. Truncation and Answered-Only Accuracy ───────────────────────────────
    trunc_headers = [
        "Model", "Role / Matched Arm", "Condition", "Dataset", "N Answerable", "Truncated Count", "Truncation %",
        "Strict Acc (Primary)", "With-Fallback Acc (Secondary)", "Answered-Only Acc", "Mean Seconds/Q"
    ]
    trunc_rows = []
    
    # Order: Matched pairs first, then thinking control, then supplementary references
    ordered_models = []
    for pid in ["Pair A", "Pair B", "Pair C"]:
        pinfo = pairs_cfg[pid]
        ordered_models.append((pinfo["generic"], f"{pid} (Generic)"))
        ordered_models.append((pinfo["medical"], f"{pid} (Medical)"))
    ordered_models.append((thinking_control, "Pair B Thinking Control (budget-limited: 99.6% trunc @ 1024 cap)"))
    for sm in supp_generic_models:
        ordered_models.append((sm, "Supplementary Generic Reference"))

    for m, role_lbl in ordered_models:
        for c in conditions:
            qmap = model_data.get(m, {}).get(c)
            if not qmap:
                continue
            for d in ["medqa", "pubmedqa"]:
                items = [
                    v for v in qmap.values()
                    if v.get("split") == "test" and v.get("dataset") == d
                ]
                if not items:
                    continue
                total_n = len(items)
                trunc_count = sum(1 for v in items if v["truncated"])
                trunc_pct = (trunc_count / total_n * 100) if total_n else 0.0
                
                prim_correct = [v["correct_strict"] for v in items]
                prim_acc = np.mean(prim_correct) if prim_correct else 0.0
                
                fb_correct = [v["correct_fallback"] for v in items]
                fb_acc = np.mean(fb_correct) if fb_correct else 0.0
                
                answered_only_correct = [v["correct_fallback"] for v in items if not v["truncated"]]
                ans_acc = np.mean(answered_only_correct) if answered_only_correct else prim_acc
                
                secs = [v["seconds"] for v in items if v.get("seconds", 0) > 0]
                mean_sec = np.mean(secs) if secs else 0.0

                trunc_rows.append([
                    m,
                    role_lbl,
                    c,
                    d,
                    total_n,
                    trunc_count,
                    f"{trunc_pct:.1f}%",
                    f"{prim_acc:.4f}",
                    f"{fb_acc:.4f}",
                    f"{ans_acc:.4f}",
                    f"{mean_sec:.2f}s",
                ])
            
    if trunc_rows:
        write_table_markdown_and_csv("table_truncation_answered_only", trunc_headers, trunc_rows)

    print("\nAnalysis complete. All markdown and CSV tables saved to docs/tables/.\n")


def main():
    parser = argparse.ArgumentParser(description="Run matched-pair statistical analyses (v3)")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base.yaml")
    parser.add_argument("--root", default=".", help="Workspace root directory")
    args = parser.parse_args()
    
    run_analysis(args.config, args.root)


if __name__ == "__main__":
    main()
