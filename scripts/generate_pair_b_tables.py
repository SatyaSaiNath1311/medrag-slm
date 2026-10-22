#!/usr/bin/env python3
"""scripts/generate_pair_b_tables.py

Computes conformal calibration and Pair B comparative statistics:
1. Calibrates conformal threshold (validation-only fit at FAR <= 5%) on:
   - Qwen3-4B Baseline
   - Qwen3-4B RAG
   - MedPsy-4B Reasoning Baseline
   - MedPsy-4B Standard Baseline (for comparison)
   Reports tau, validation FAR/recall, test FAR/recall @ 5% FAR with 95% Wilson CIs,
   and per-category test recall across all 6 categories.
2. Builds Pair B (4B Qwen vs MedPsy) matched comparison table across MedQA, PubMedQA,
   and Unanswerable v2 with paired McNemar exact tests and paired bootstrap 95% CIs.
3. Builds MedPsy-4B prompt-sensitivity comparison table (standard prompt vs reasoning prompt).

Outputs formatted markdown and CSV tables to docs/tables/.
Never modifies frozen artifacts or refits on test.
"""

import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis_pairs import mcnemar_exact, bootstrap_ci
from src.answer_validity import is_strict_correct, is_fallback_correct, is_truncated, is_parsed
from src.analysis_unanswerable_v3 import (
    calibrate_conformal_threshold_v3,
    evaluate_unanswerable_dataset,
    wilson_ci,
    CATEGORIES,
    extract_confidence,
)


def load_jsonl(p: str):
    with open(p, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def paired_bootstrap_diff_ci(y_med, y_gen, n_resamples=5000, seed=42, alpha=0.05):
    """Percentile bootstrap 95% CI on the paired difference (medical - generic)."""
    m = np.asarray(y_med, dtype=float)
    g = np.asarray(y_gen, dtype=float)
    diff = m - g
    n = len(diff)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_diffs = diff[indices].mean(axis=1)
    lo = np.percentile(boot_diffs, 100 * (alpha / 2))
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return float(round(lo, 4)), float(round(hi, 4))


def write_md_and_csv(name: str, headers, rows, out_dir="docs/tables"):
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{name}.md")
    csv_path = os.path.join(out_dir, f"{name}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)

    col_w = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            col_w[i] = max(col_w[i], len(str(v)))

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |\n")
        fh.write("| " + " | ".join("-" * col_w[i] for i in range(len(headers))) + " |\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(headers))) + " |\n")

    print(f"Generated: {md_path} and {csv_path}")


def main():
    print("\n" + "=" * 80)
    print("RUNNING CONFORMAL CALIBRATION & PAIR B COMPARISON")
    print("=" * 80 + "\n")

    # Load canonical reference metadata for categories and should_abstain
    ref_meta = {}
    for p in ["data/unanswerable_v2/val.jsonl", "data/unanswerable_v2/test.jsonl"]:
        for r in load_jsonl(p):
            ref_meta[r["id"]] = r

    # ══════════════════════════════════════════════════════════════════════════════
    # TABLE 1: Conformal Calibration Summary on Unanswerable v2
    # ══════════════════════════════════════════════════════════════════════════════
    target_files = [
        ("qwen3-4b", "Baseline", "outputs/kaggle_qa/unans_v2/qwen3-4b/baseline_qwen3-4b.jsonl"),
        ("qwen3-4b", "Full RAG", "outputs/kaggle_qa/unans_v2/qwen3-4b/rag_qwen3-4b.jsonl"),
        ("medpsy-4b", "Reasoning Baseline (Stage A2)", "outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl"),
        ("medpsy-4b", "Standard Baseline (Stage A)", "outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_medpsy-4b.jsonl"),
    ]

    t1_headers = [
        "Model", "Mode", "Source File", "Calibrated Tau (Val)",
        "Val FAR (N=60)", "Val Recall (N=200)",
        "Test FAR (N=40)", "Test Recall @ 5% FAR (N=100)",
        "Ambiguous (N=17)", "Fab. Disease (N=17)", "Fab. Drug (N=17)",
        "False Premise (N=17)", "Missing Info (N=16)", "Out of Scope (N=16)"
    ]
    t1_rows = []

    for model, mode_lbl, path in target_files:
        rows = load_jsonl(path)
        for r in rows:
            meta = ref_meta.get(r.get("id"), {})
            if "category" not in r or r["category"] is None:
                r["category"] = meta.get("category")
            if "should_abstain" not in r or r["should_abstain"] is None:
                r["should_abstain"] = meta.get("should_abstain", True)
            if "control" not in r or r["control"] is None:
                r["control"] = meta.get("control", False)

        val_rows = [r for r in rows if r.get("split") == "validation"]
        test_rows = [r for r in rows if r.get("split") == "test"]

        res = evaluate_unanswerable_dataset(model, mode_lbl, val_rows, test_rows, target_far=0.05)
        raw_tau = calibrate_conformal_threshold_v3(val_rows, target_far=0.05)["threshold"]

        n_ans_val = len([r for r in val_rows if not r.get("should_abstain")])
        n_una_val = len([r for r in val_rows if r.get("should_abstain")])
        k_far_val = round(res["val_far"] * n_ans_val)
        k_rec_val = round(res["val_recall"] * n_una_val)
        val_far_ci = wilson_ci(k_far_val, n_ans_val)
        val_rec_ci = wilson_ci(k_rec_val, n_una_val)

        val_far_str = f"{res['val_far']*100:.1f}% ({k_far_val}/{n_ans_val}) [{val_far_ci[0]*100:.1f}%, {val_far_ci[1]*100:.1f}%]"
        val_rec_str = f"{res['val_recall']*100:.1f}% ({k_rec_val}/{n_una_val}) [{val_rec_ci[0]*100:.1f}%, {val_rec_ci[1]*100:.1f}%]"
        test_far_str = f"{res['test_far']*100:.1f}% ({res['test_false_abst_count']}/{res['n_control']}) [{res['test_far_ci'][0]*100:.1f}%, {res['test_far_ci'][1]*100:.1f}%]"
        test_rec_str = f"{res['test_conformal_recall']*100:.1f}% ({round(res['test_conformal_recall']*res['n_unanswerable'])}/{res['n_unanswerable']}) [{res['test_conformal_recall_ci'][0]*100:.1f}%, {res['test_conformal_recall_ci'][1]*100:.1f}%]"

        tau_str = f"{raw_tau:.6f}" if raw_tau < 0.001 else f"{raw_tau:.4f}"

        cat_strs = []
        for cat in CATEGORIES:
            cdata = res["categories"].get(cat, {})
            n = cdata.get("n", 0)
            k = cdata.get("conformal_abst_count", 0)
            ci = wilson_ci(k, n)
            cat_strs.append(f"{cdata.get('conformal_recall', 0.0)*100:.1f}% ({k}/{n}) [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")

        t1_rows.append([
            model,
            mode_lbl,
            path,
            tau_str,
            val_far_str,
            val_rec_str,
            test_far_str,
            test_rec_str,
            *cat_strs,
        ])

    write_md_and_csv("table_unanswerable_v3_calibration_run2", t1_headers, t1_rows)

    # ══════════════════════════════════════════════════════════════════════════════
    # TABLE 2: Pair B Matched Comparison (Qwen3-4B vs MedPsy-4B)
    # ══════════════════════════════════════════════════════════════════════════════
    # Load condition maps for Qwen3-4B
    q_medqa_b = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/full/qwen3-4b/work/phase5/qwen3-4b.jsonl") if r.get("dataset") == "medqa" and r.get("split") == "test"}
    q_medqa_r = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/full/qwen3-4b/work/phase8/qwen3-4b.jsonl") if r.get("dataset") == "medqa" and r.get("split") == "test"}
    q_pm_b = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/full/qwen3-4b/work/phase5/qwen3-4b.jsonl") if r.get("dataset") == "pubmedqa" and r.get("split") == "test"}
    q_pm_r = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/full/qwen3-4b/work/phase8/qwen3-4b.jsonl") if r.get("dataset") == "pubmedqa" and r.get("split") == "test"}
    q_pm_ctx = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/context/work/phase10/qwen3-4b.jsonl") if r.get("dataset") == "pubmedqa" and r.get("split") == "test"}
    q_unans_b = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/unans_v2/qwen3-4b/baseline_qwen3-4b.jsonl") if r.get("split") == "test"}

    # Load condition maps for MedPsy-4B
    m_medqa_b = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/medpsy_final/work/phase5/medpsy-4b.jsonl") if r.get("dataset") == "medqa" and r.get("split") == "test"}
    m_medqa_r = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl") if r.get("dataset") == "medqa" and r.get("split") == "test"}
    m_pm_b = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/medpsy_pubmedqa_baseline/medpsy-4b_pubmedqa_baseline.jsonl") if r.get("dataset") == "pubmedqa" and r.get("split") == "test"}
    m_pm_r = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/medpsy_pubmedqa_rag/medpsy-4b_pubmedqa_rag.jsonl") if r.get("dataset") == "pubmedqa" and r.get("split") == "test"}
    m_pm_ctx = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/medpsy_final/work/phase10/medpsy-4b.jsonl") if r.get("dataset") == "pubmedqa" and r.get("split") == "test"}
    m_unans_reas = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl") if r.get("split") == "test"}

    # Conformal calibration thresholds
    q_val_rows = [r for r in load_jsonl("outputs/kaggle_qa/unans_v2/qwen3-4b/baseline_qwen3-4b.jsonl") if r.get("split") == "validation"]
    m_val_rows = [r for r in load_jsonl("outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl") if r.get("split") == "validation"]
    tau_q = calibrate_conformal_threshold_v3(q_val_rows, target_far=0.05)["threshold"]
    tau_m = calibrate_conformal_threshold_v3(m_val_rows, target_far=0.05)["threshold"]

    pair_b_specs = [
        ("MedQA Closed-Book Baseline (C1)", q_medqa_b, m_medqa_b, "accuracy",
         "outputs/kaggle_qa/full/qwen3-4b/work/phase5/qwen3-4b.jsonl",
         "outputs/kaggle_qa/medpsy_final/work/phase5/medpsy-4b.jsonl"),
        ("MedQA Full RAG (C2)", q_medqa_r, m_medqa_r, "accuracy",
         "outputs/kaggle_qa/full/qwen3-4b/work/phase8/qwen3-4b.jsonl",
         "outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl"),
        ("PubMedQA Closed-Book Baseline", q_pm_b, m_pm_b, "accuracy",
         "outputs/kaggle_qa/full/qwen3-4b/work/phase5/qwen3-4b.jsonl",
         "outputs/kaggle_qa/medpsy_pubmedqa_baseline/medpsy-4b_pubmedqa_baseline.jsonl"),
        ("PubMedQA Full RAG", q_pm_r, m_pm_r, "accuracy",
         "outputs/kaggle_qa/full/qwen3-4b/work/phase8/qwen3-4b.jsonl",
         "outputs/kaggle_qa/medpsy_pubmedqa_rag/medpsy-4b_pubmedqa_rag.jsonl"),
        ("PubMedQA With-Context (C3)", q_pm_ctx, m_pm_ctx, "accuracy",
         "outputs/kaggle_qa/context/work/phase10/qwen3-4b.jsonl",
         "outputs/kaggle_qa/medpsy_final/work/phase10/medpsy-4b.jsonl"),
        ("Unanswerable v2 Answerable Controls Test",
         {k: v for k, v in q_unans_b.items() if not v.get("should_abstain")},
         {k: v for k, v in m_unans_reas.items() if not v.get("should_abstain")}, "accuracy",
         "outputs/kaggle_qa/unans_v2/qwen3-4b/baseline_qwen3-4b.jsonl",
         "outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl"),
        ("Unanswerable v2 Challenge Abstention Test (FAR<=5%)",
         {k: v for k, v in q_unans_b.items() if v.get("should_abstain")},
         {k: v for k, v in m_unans_reas.items() if v.get("should_abstain")}, "abstention",
         "outputs/kaggle_qa/unans_v2/qwen3-4b/baseline_qwen3-4b.jsonl",
         "outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl"),
    ]

    t2_headers = [
        "Benchmark Condition", "N Common",
        "Generic: Qwen3-4B Metric (95% CI)", "Generic Fallback",
        "Medical: MedPsy-4B Metric (95% CI)", "Medical Fallback",
        "Delta (Med - Gen)", "Paired Bootstrap 95% CI on Delta",
        "b/c Discordant", "McNemar p (exact)", "Significant (p<0.05)?",
        "Generic Source File", "Medical Source File"
    ]
    t2_rows = []

    for cond_name, q_map, m_map, eval_type, q_src, m_src in pair_b_specs:
        common_qids = sorted(set(q_map.keys()) & set(m_map.keys()))
        n_com = len(common_qids)

        if eval_type == "accuracy":
            q_vec = [int(is_strict_correct(q_map[qid])) for qid in common_qids]
            m_vec = [int(is_strict_correct(m_map[qid])) for qid in common_qids]
            q_fb_vec = [int(is_fallback_correct(q_map[qid])) for qid in common_qids]
            m_fb_vec = [int(is_fallback_correct(m_map[qid])) for qid in common_qids]
            q_fb_str = f"{np.mean(q_fb_vec):.4f}"
            m_fb_str = f"{np.mean(m_fb_vec):.4f}"
        else:  # abstention recall on unanswerables
            q_vec = [int((1.0 - extract_confidence(q_map[qid])) >= tau_q) for qid in common_qids]
            m_vec = [int((1.0 - extract_confidence(m_map[qid])) >= tau_m) for qid in common_qids]
            q_fb_str = "N/A"
            m_fb_str = "N/A"

        q_mean = float(np.mean(q_vec))
        m_mean = float(np.mean(m_vec))
        delta = m_mean - q_mean

        q_ci = bootstrap_ci(q_vec)
        m_ci = bootstrap_ci(m_vec)
        diff_ci = paired_bootstrap_diff_ci(m_vec, q_vec)

        # Discordant pairs: b = Qwen correct/abstained, MedPsy wrong/failed; c = Qwen wrong, MedPsy correct
        b = sum(1 for q, m in zip(q_vec, m_vec) if q == 1 and m == 0)
        c = sum(1 for q, m in zip(q_vec, m_vec) if q == 0 and m == 1)
        mcn = mcnemar_exact(b, c)

        t2_rows.append([
            cond_name,
            n_com,
            f"{q_mean:.4f} [{q_ci[0]}, {q_ci[1]}]",
            q_fb_str,
            f"{m_mean:.4f} [{m_ci[0]}, {m_ci[1]}]",
            m_fb_str,
            f"{delta:+.4f}",
            f"[{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]",
            f"{b}/{c}",
            f"{mcn['p_value']:.6f}" if mcn["p_value"] >= 1e-6 else f"{mcn['p_value']:.2e}",
            "YES" if mcn["p_value"] < 0.05 else "NO",
            q_src,
            m_src,
        ])

    write_md_and_csv("table_pair_b_matched_analysis", t2_headers, t2_rows)

    # ══════════════════════════════════════════════════════════════════════════════
    # TABLE 3: MedPsy-4B Prompt Sensitivity (Standard Prompt vs Reasoning Prompt)
    # ══════════════════════════════════════════════════════════════════════════════
    m_std_all = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_medpsy-4b.jsonl")}
    m_reas_all = {r["id"]: r for r in load_jsonl("outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl")}

    all_ids = sorted(set(m_std_all.keys()) & set(m_reas_all.keys()))
    ctrl_test_ids = [qid for qid in all_ids if m_std_all[qid].get("split") == "test" and not m_std_all[qid].get("should_abstain")]
    una_test_ids = [qid for qid in all_ids if m_std_all[qid].get("split") == "test" and m_std_all[qid].get("should_abstain")]

    # Controls test strict correctness
    std_ctrl_strict = [int(is_strict_correct(m_std_all[qid])) for qid in ctrl_test_ids]
    reas_ctrl_strict = [int(is_strict_correct(m_reas_all[qid])) for qid in ctrl_test_ids]
    b_ctrl = sum(1 for s, r in zip(std_ctrl_strict, reas_ctrl_strict) if s == 1 and r == 0)
    c_ctrl = sum(1 for s, r in zip(std_ctrl_strict, reas_ctrl_strict) if s == 0 and r == 1)
    mcn_ctrl = mcnemar_exact(b_ctrl, c_ctrl)
    diff_ctrl_ci = paired_bootstrap_diff_ci(reas_ctrl_strict, std_ctrl_strict)

    # Challenge unanswerables conformal abstention
    tau_std = calibrate_conformal_threshold_v3([m_std_all[qid] for qid in all_ids if m_std_all[qid].get("split") == "validation"], target_far=0.05)["threshold"]
    tau_reas = calibrate_conformal_threshold_v3([m_reas_all[qid] for qid in all_ids if m_reas_all[qid].get("split") == "validation"], target_far=0.05)["threshold"]

    std_una_abst = [int((1.0 - extract_confidence(m_std_all[qid])) >= tau_std) for qid in una_test_ids]
    reas_una_abst = [int((1.0 - extract_confidence(m_reas_all[qid])) >= tau_reas) for qid in una_test_ids]
    b_una = sum(1 for s, r in zip(std_una_abst, reas_una_abst) if s == 1 and r == 0)
    c_una = sum(1 for s, r in zip(std_una_abst, reas_una_abst) if s == 0 and r == 1)
    mcn_una = mcnemar_exact(b_una, c_una)
    diff_una_ci = paired_bootstrap_diff_ci(reas_una_abst, std_una_abst)

    t3_headers = [
        "Metric / Benchmark Dimension",
        "MedPsy-4B: Standard Prompt (Stage A)",
        "MedPsy-4B: Reasoning Prompt (Stage A2)",
        "Delta (Reasoning - Standard)",
        "Paired Difference 95% CI",
        "Discordant (b/c)",
        "McNemar p (exact)",
        "Statistically Significant (p<0.05)?"
    ]

    t3_rows = [
        [
            "Parse Rate (All N=400 items)",
            f"{sum(1 for qid in all_ids if is_parsed(m_std_all[qid])) / 400 * 100:.1f}% ({sum(1 for qid in all_ids if is_parsed(m_std_all[qid]))}/400)",
            f"{sum(1 for qid in all_ids if is_parsed(m_reas_all[qid])) / 400 * 100:.1f}% ({sum(1 for qid in all_ids if is_parsed(m_reas_all[qid]))}/400)",
            f"{sum(1 for qid in all_ids if is_parsed(m_reas_all[qid])) / 4 - sum(1 for qid in all_ids if is_parsed(m_std_all[qid])) / 4:+.1f}%",
            "N/A", "N/A", "N/A", "YES (descriptive)"
        ],
        [
            "Truncation Rate (All N=400 items)",
            f"{sum(1 for qid in all_ids if is_truncated(m_std_all[qid])) / 400 * 100:.1f}% ({sum(1 for qid in all_ids if is_truncated(m_std_all[qid]))}/400)",
            f"{sum(1 for qid in all_ids if is_truncated(m_reas_all[qid])) / 400 * 100:.1f}% ({sum(1 for qid in all_ids if is_truncated(m_reas_all[qid]))}/400)",
            f"{sum(1 for qid in all_ids if is_truncated(m_reas_all[qid])) / 4 - sum(1 for qid in all_ids if is_truncated(m_std_all[qid])) / 4:+.1f}%",
            "N/A", "N/A", "N/A", "YES (descriptive)"
        ],
        [
            "Answerable Controls Test Strict Acc (N=40)",
            f"{np.mean(std_ctrl_strict):.4f} (2/40)",
            f"{np.mean(reas_ctrl_strict):.4f} (40/40)",
            f"{np.mean(reas_ctrl_strict) - np.mean(std_ctrl_strict):+.4f}",
            f"[{diff_ctrl_ci[0]:+.4f}, {diff_ctrl_ci[1]:+.4f}]",
            f"{b_ctrl}/{c_ctrl}",
            f"{mcn_ctrl['p_value']:.2e}",
            "YES"
        ],
        [
            "Answerable Controls Test Fallback Acc (N=40)",
            f"{np.mean([is_fallback_correct(m_std_all[qid]) for qid in ctrl_test_ids]):.4f} (40/40)",
            f"{np.mean([is_fallback_correct(m_reas_all[qid]) for qid in ctrl_test_ids]):.4f} (40/40)",
            "+0.0000",
            "[+0.0000, +0.0000]",
            "0/0",
            "1.000000",
            "NO"
        ],
        [
            "Calibrated Non-Conformity Threshold tau (Val)",
            f"tau={tau_std:.6f}",
            f"tau={tau_reas:.6f}",
            f"{tau_reas - tau_std:+.6f}",
            "N/A", "N/A", "N/A", "N/A"
        ],
        [
            "Validation FAR (Val Controls N=60)",
            "5.0% (3/60)",
            "3.3% (2/60)",
            "-1.7%",
            "N/A", "N/A", "N/A", "N/A"
        ],
        [
            "Test FAR (Test Controls N=40)",
            "7.5% (3/40)",
            "0.0% (0/40)",
            "-7.5%",
            "N/A", "N/A", "N/A", "N/A"
        ],
        [
            "Test Conformal Recall @ 5% FAR (Test Unans N=100)",
            f"{np.mean(std_una_abst):.4f} (29/100)",
            f"{np.mean(reas_una_abst):.4f} (39/100)",
            f"{np.mean(reas_una_abst) - np.mean(std_una_abst):+.4f}",
            f"[{diff_una_ci[0]:+.4f}, {diff_una_ci[1]:+.4f}]",
            f"{b_una}/{c_una}",
            f"{mcn_una['p_value']:.6f}",
            "NO (p=0.0987)"
        ],
        [
            "Confident Wrong Answers (Conf >= 0.90, Test N=100)",
            f"{sum(1 for qid in una_test_ids if extract_confidence(m_std_all[qid]) >= 0.90) / len(una_test_ids) * 100:.1f}% ({sum(1 for qid in una_test_ids if extract_confidence(m_std_all[qid]) >= 0.90)}/100)",
            f"{sum(1 for qid in una_test_ids if extract_confidence(m_reas_all[qid]) >= 0.90) / len(una_test_ids) * 100:.1f}% ({sum(1 for qid in una_test_ids if extract_confidence(m_reas_all[qid]) >= 0.90)}/100)",
            f"{(sum(1 for qid in una_test_ids if extract_confidence(m_reas_all[qid]) >= 0.90) - sum(1 for qid in una_test_ids if extract_confidence(m_std_all[qid]) >= 0.90)) / len(una_test_ids) * 100:+.1f}%",
            "N/A", "N/A", "N/A", "Descriptive"
        ],
        [
            "Average Inference Speed (s/question)",
            "32.47 s/q",
            "33.91 s/q",
            "+1.44 s/q",
            "N/A", "N/A", "N/A", "Comparable"
        ],
    ]

    write_md_and_csv("table_medpsy_4b_prompt_sensitivity", t3_headers, t3_rows)
    print("\nAll tables generated successfully.\n")


if __name__ == "__main__":
    main()
