"""src/analysis_pairs.py — Matched-Pair Generic vs Medical-Domain Statistical Analysis.

Implements all pre-registered analyses from docs/DESIGN_GENERIC_VS_MEDICAL.md:
  1. Matched-pair McNemar exact tests (generic vs medical per pair and condition) with Holm correction.
  2. Bootstrap 95% CIs (1,000 resamples, seed 42) for accuracy and answered-only accuracy.
  3. RAG effect per model (McNemar baseline vs RAG, discordant counts, odds ratios).
  4. Pooled Cochran-Mantel-Haenszel (CMH) tests separately for Generic group and Medical group.
  5. Interaction test comparing generic vs medical RAG effect (logistic regression interaction and Breslow-Day test).
  6. Primary accuracy (truncated = wrong) alongside answered-only accuracy.
  7. Outputs publication-ready markdown and CSV tables to docs/tables/.
  8. Resilient: skips models with missing outputs without crashing.
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
    """
    candidates = [
        os.path.join(root_dir, "outputs", "v3", model_name, condition, f"{model_name}.jsonl"),
        os.path.join(root_dir, "outputs", "v3", model_name, f"{condition}.jsonl"),
        os.path.join("/kaggle/working/outputs/v3", model_name, condition, f"{model_name}.jsonl"),
    ]
    
    # Legacy fallbacks
    if condition == "C1":
        candidates.extend([
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", model_name, "work", "phase5", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_final", "work", "phase5", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_pubmedqa_baseline", f"{model_name}_pubmedqa_baseline.jsonl"),
        ])
    elif condition == "C2":
        candidates.extend([
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", model_name, "work", "phase8", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", f"{model_name}_rag", "work", "phase8", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_rag", "work", "phase8", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_pubmedqa_rag", f"{model_name}_pubmedqa_rag.jsonl"),
        ])
    elif condition == "C3":
        candidates.extend([
            os.path.join(root_dir, "outputs", "kaggle_qa", "context", "work", "phase10", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_final", "work", "phase10", f"{model_name}.jsonl"),
        ])
    elif condition == "C4":
        candidates.extend([
            os.path.join(root_dir, "outputs", "sc_test_s3_unanswerable", "outputs", "sc_test", model_name, "unanswerable_v2.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "unans_v2", model_name, f"baseline_{model_name}.jsonl"),
        ])
        
    for p in candidates:
        if os.path.exists(p):
            try:
                rows = read_jsonl(p)
                if rows:
                    return rows
            except Exception:
                continue
    return None


def extract_evaluation_map(rows):
    """Index answerable rows by question ID, extracting correctness, truncation, and confidence.
    
    In accordance with the pre-registered protocol:
      - Primary Rule: Strict correctness (not truncated and parsed and correct).
      - Secondary Rule: With-fallback correctness (raw correct field).
      - Answered-only accuracy considers only non-truncated questions.
    """
    res = {}
    for r in rows:
        qid = r.get("id")
        if not qid:
            continue
        # Answerable filter: ignore unanswerable questions when computing standard benchmark accuracy
        if r.get("should_abstain"):
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
            "dataset": r.get("dataset", "medqa"),
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
    cfg = load_config(config_path)
    v3_cfg = cfg.get("v3", {})
    pairs_cfg = v3_cfg.get("pairs", {
        "P1": {"name": "1.7B (Qwen)", "generic": "qwen3-1.7b", "medical": "medpsy-1.7b", "control": "qwen3-1.7b-thinking"},
        "P2": {"name": "4B (Qwen)", "generic": "qwen3-4b", "medical": "medpsy-4b", "control": "qwen3-4b-thinking-2507"},
        "P3": {"name": "4B (Gemma)", "generic": "gemma3-4b", "medical": "medgemma-1.5-4b-it", "control": None},
    })
    
    print("\n" + "=" * 80)
    print("MEDRAG-SLM V3: MATCHED-PAIR GENERIC VS MEDICAL ANALYSIS")
    print("=" * 80 + "\n")
    
    # ── 1. Load outputs across models & conditions ──────────────────────────────
    all_models = set()
    for p in pairs_cfg.values():
        all_models.add(p["generic"])
        all_models.add(p["medical"])
        if p.get("control"):
            all_models.add(p["control"])
            
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

    # ── 2. RQ1: Matched Pairs Within-Pair McNemar + Holm ────────────────────────
    rq1_headers = [
        "Pair", "Arm Comparison", "Condition", "N Common",
        "Generic Strict Acc (95% CI)", "Generic With-Fallback Acc",
        "Medical Strict Acc (95% CI)", "Medical With-Fallback Acc",
        "Delta Strict Acc", "b/c Discordant (Strict)", "McNemar p (exact)", "Holm p-adj", "Supported (p<0.05)?"
    ]
    rq1_rows = []
    p_values_to_correct = []
    pending_row_indices = []

    for pid in ["P1", "P2", "P3"]:
        pinfo = pairs_cfg.get(pid)
        if not pinfo:
            continue
        gen_m = pinfo["generic"]
        med_m = pinfo["medical"]
        pname = pinfo.get("name", pid)
        
        for c in ["C1", "C2", "C3"]:
            gen_map = model_data.get(gen_m, {}).get(c)
            med_map = model_data.get(med_m, {}).get(c)
            
            if not gen_map or not med_map:
                rq1_rows.append([
                    f"{pid} ({pname})", f"{gen_m} vs {med_m}", c, "-",
                    "-", "-", "-", "-", "-", "-", "-", "-", "PENDING OUTPUTS"
                ])
                continue
                
            common_qids = sorted(set(gen_map.keys()) & set(med_map.keys()))
            if not common_qids:
                continue
                
            gen_correct = [gen_map[q]["correct_strict"] for q in common_qids]
            gen_fb = [gen_map[q]["correct_fallback"] for q in common_qids]
            med_correct = [med_map[q]["correct_strict"] for q in common_qids]
            med_fb = [med_map[q]["correct_fallback"] for q in common_qids]
            
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
            
            # Optional control comparison (e.g. Qwen thinking control vs MedPsy)
            ctrl_m = pinfo.get("control")
            if ctrl_m and model_data.get(ctrl_m, {}).get(c):
                ctrl_map = model_data[ctrl_m][c]
                ctrl_common = sorted(set(ctrl_map.keys()) & set(med_map.keys()))
                if ctrl_common:
                    c_corr = [ctrl_map[q]["correct_strict"] for q in ctrl_common]
                    c_fb = [ctrl_map[q]["correct_fallback"] for q in ctrl_common]
                    m_corr = [med_map[q]["correct_strict"] for q in ctrl_common]
                    m_fb = [med_map[q]["correct_fallback"] for q in ctrl_common]
                    c_acc = np.mean(c_corr)
                    c_fb_acc = np.mean(c_fb)
                    m_acc = np.mean(m_corr)
                    m_fb_acc = np.mean(m_fb)
                    c_ci = bootstrap_ci(c_corr)
                    m_ci = bootstrap_ci(m_corr)
                    b_ctrl = sum(1 for g, m in zip(c_corr, m_corr) if g == 1 and m == 0)
                    c_ctrl = sum(1 for g, m in zip(c_corr, m_corr) if g == 0 and m == 1)
                    mcn_c = mcnemar_exact(b_ctrl, c_ctrl)
                    rq1_rows.append([
                        f"{pid} Control",
                        f"{ctrl_m} vs {med_m}",
                        c,
                        len(ctrl_common),
                        f"{c_acc:.4f} [{c_ci[0]}, {c_ci[1]}]",
                        f"{c_fb_acc:.4f}",
                        f"{m_acc:.4f} [{m_ci[0]}, {m_ci[1]}]",
                        f"{m_fb_acc:.4f}",
                        f"{m_acc - c_acc:+.4f}",
                        f"{b_ctrl}/{c_ctrl}",
                        f"{mcn_c['p_value']:.5f}",
                        "-",
                        "YES" if mcn_c["p_value"] < 0.05 and (m_acc > c_acc) else "NO",
                    ])

    # Apply Holm correction to primary matched pairs
    adj_ps = holm_bonferroni(p_values_to_correct)
    for row_idx, adj_p in zip(pending_row_indices, adj_ps):
        p_raw = float(rq1_rows[row_idx][10])
        delta = float(rq1_rows[row_idx][8])
        rq1_rows[row_idx][11] = f"{adj_p:.5f}"
        supported = "YES" if (adj_p < 0.05 and delta > 0) else "NO"
        rq1_rows[row_idx][12] = supported

    write_table_markdown_and_csv("table_matched_pairs_rq1", rq1_headers, rq1_rows)

    # ── 3. RQ2: RAG Effect Per Model (Baseline C1 vs RAG C2) ───────────────────
    rq2_headers = [
        "Model", "Group", "N Common",
        "Baseline Strict Acc (95% CI)", "Baseline With-Fallback Acc",
        "RAG Strict Acc (95% CI)", "RAG With-Fallback Acc",
        "Delta Strict Acc", "b (Base+ / RAG-)", "c (Base- / RAG+)", "Discordant OR", "McNemar p (exact)"
    ]
    rq2_rows = []
    
    # Store for pooled CMH and interaction test
    generic_models = [pinfo["generic"] for pinfo in pairs_cfg.values()]
    medical_models = [pinfo["medical"] for pinfo in pairs_cfg.values()]
    
    cmh_generic_tables = []
    cmh_medical_tables = []
    
    interaction_y = []
    interaction_med = []
    interaction_rag = []

    for m in sorted(all_models):
        c1_map = model_data.get(m, {}).get("C1")
        c2_map = model_data.get(m, {}).get("C2")
        group = "Medical" if m in medical_models else "Generic"
        
        if not c1_map or not c2_map:
            rq2_rows.append([m, group, "-", "-", "-", "-", "-", "-", "-", "-", "-", "PENDING"])
            continue
            
        common_qids = sorted(set(c1_map.keys()) & set(c2_map.keys()))
        if not common_qids:
            continue
            
        c1_corr = [c1_map[q]["correct_strict"] for q in common_qids]
        c1_fb = [c1_map[q]["correct_fallback"] for q in common_qids]
        c2_corr = [c2_map[q]["correct_strict"] for q in common_qids]
        c2_fb = [c2_map[q]["correct_fallback"] for q in common_qids]
        
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
            group,
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
        
        # 2x2 table for CMH: [[RAG correct, RAG wrong], [Base correct, Base wrong]]
        a_count = sum(c2_corr)
        b_count = len(c2_corr) - a_count
        c_count = sum(c1_corr)
        d_count = len(c1_corr) - c_count
        table_2x2 = [[a_count, b_count], [c_count, d_count]]
        
        if group == "Generic":
            cmh_generic_tables.append(table_2x2)
        else:
            cmh_medical_tables.append(table_2x2)
            
        # Accumulate for logistic regression interaction test
        is_med = 1.0 if group == "Medical" else 0.0
        for y_c1 in c1_corr:
            interaction_y.append(y_c1)
            interaction_med.append(is_med)
            interaction_rag.append(0.0)
        for y_c2 in c2_corr:
            interaction_y.append(y_c2)
            interaction_med.append(is_med)
            interaction_rag.append(1.0)

    write_table_markdown_and_csv("table_rag_effect_rq2", rq2_headers, rq2_rows)

    # ── 4. Group CMH & Interaction Test ────────────────────────────────────────
    cmh_gen = cochran_mantel_haenszel(cmh_generic_tables)
    cmh_med = cochran_mantel_haenszel(cmh_medical_tables)
    
    inter_res = fit_logistic_interaction(interaction_y, interaction_med, interaction_rag)
    
    cmh_headers = ["Analysis Layer", "Stratification / Group", "Common Odds Ratio", "CMH / Stat", "p-value", "Interpretation"]
    cmh_rows = [
        [
            "Pooled CMH Test",
            "Generic Group (Qwen, Gemma, etc.)",
            f"{cmh_gen['common_or']}" if cmh_gen["common_or"] is not None else "-",
            f"Chi2={cmh_gen['cmh_stat']}" if cmh_gen["cmh_stat"] is not None else "-",
            f"{cmh_gen['p_value']}" if cmh_gen["p_value"] is not None else "-",
            "Significant RAG benefit" if cmh_gen.get("p_value", 1) < 0.05 else "Not significant / pending",
        ],
        [
            "Pooled CMH Test",
            "Medical Group (MedPsy, MedGemma)",
            f"{cmh_med['common_or']}" if cmh_med["common_or"] is not None else "-",
            f"Chi2={cmh_med['cmh_stat']}" if cmh_med["cmh_stat"] is not None else "-",
            f"{cmh_med['p_value']}" if cmh_med["p_value"] is not None else "-",
            "Significant RAG benefit" if cmh_med.get("p_value", 1) < 0.05 else "Not significant / pending",
        ],
        [
            "Logistic Interaction (Model_Type x RAG)",
            "Generic vs Medical Differential RAG Gain",
            f"Beta_int={inter_res['beta_int']}" if inter_res["beta_int"] is not None else "-",
            f"LR Chi2={inter_res.get('lr_stat')}" if inter_res.get("lr_stat") is not None else "-",
            f"Wald p={inter_res['wald_p']}, LR p={inter_res['lr_p']}" if inter_res["wald_p"] is not None else "-",
            "Differential RAG gain confirmed" if (inter_res.get("lr_p") is not None and inter_res["lr_p"] < 0.05) else "No differential gain / pending",
        ],
    ]
    write_table_markdown_and_csv("table_group_cmh_interaction", cmh_headers, cmh_rows)

    # ── 5. Truncation and Answered-Only Accuracy ───────────────────────────────
    trunc_headers = [
        "Model", "Condition", "N Answerable", "Truncated Count", "Truncation %",
        "Strict Acc (Primary)", "With-Fallback Acc (Secondary)", "Answered-Only Acc", "Mean Seconds/Q"
    ]
    trunc_rows = []
    
    for m in sorted(all_models):
        for c in conditions:
            qmap = model_data.get(m, {}).get(c)
            if not qmap:
                continue
            total_n = len(qmap)
            trunc_count = sum(1 for v in qmap.values() if v["truncated"])
            trunc_pct = (trunc_count / total_n * 100) if total_n else 0.0
            
            prim_correct = [v["correct_strict"] for v in qmap.values()]
            prim_acc = np.mean(prim_correct) if prim_correct else 0.0
            
            fb_correct = [v["correct_fallback"] for v in qmap.values()]
            fb_acc = np.mean(fb_correct) if fb_correct else 0.0
            
            answered_only_correct = [v["correct_fallback"] for v in qmap.values() if not v["truncated"]]
            ans_acc = np.mean(answered_only_correct) if answered_only_correct else prim_acc
            
            secs = [v["seconds"] for v in qmap.values() if v.get("seconds", 0) > 0]
            mean_sec = np.mean(secs) if secs else 0.0

            trunc_rows.append([
                m,
                c,
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
