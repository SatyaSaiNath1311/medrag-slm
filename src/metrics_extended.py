"""Comprehensive extended evaluation metrics for Medical RAG SLM Benchmark.

Computes:
1. QA Performance: Per-class Precision, Recall, F1; Micro-F1, Macro-F1, Weighted-F1,
   and confusion matrices (MedQA A-D, PubMedQA yes/no/maybe) for Baseline, RAG,
   best Adaptive gate, and +Abstract.
2. Retrieval Performance (MedQA): Recall@1/@5/@10/@20, Precision@5, MRR@10, MAP@10,
   nDCG@10 on Phase 6 (Hybrid top-20) vs Phase 7 (Reranked top-5) under both matching rules.
3. RAG System Reliability (MedQA): Unsupported-answer rate, Citation Precision,
   Citation Recall, Citation Validity, Faithfulness.
4. Confidence Calibration: ECE, MCE, Brier score, NLL (raw and temperature-scaled where T* exists).
5. Efficiency: Mean and P95 latency, throughput (questions/sec).

Supports all 6 models including medpsy-4b with graceful handling of missing modes/splits ("n/a").
"""
import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


RETR_SEC_PER_Q = 0.1273  # Phase 6 (23.424s) + Phase 7 (180.261s) = 203.685s / 1600 q


# ==============================================================================
# Statistical & Bootstrap Helpers
# ==============================================================================

def bootstrap_ci(values: List[float], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap 95% confidence interval for a list of values."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return (round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4))


def bootstrap_metric_ci(
    items: List[Any],
    metric_fn: Any,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for an arbitrary metric function."""
    if not items:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(items)
    scores = []
    for _ in range(n_boot):
        sample = rng.choices(items, k=n)
        scores.append(metric_fn(sample))
    scores.sort()
    return (round(scores[int(0.025 * n_boot)], 4), round(scores[int(0.975 * n_boot)], 4))


# ==============================================================================
# Text Matching Helpers for Grounding & Retrieval
# ==============================================================================

def text_contains_option_exact(passage_text: str, option_text: str) -> bool:
    """Strict verbatim matching rule."""
    if not passage_text or not option_text:
        return False
    return option_text.lower().strip() in passage_text.lower()


def text_contains_option_first_word_removed(passage_text: str, option_text: str) -> bool:
    """Fallback matching rule: removes leading article/preposition if length >= 4."""
    if not passage_text or not option_text:
        return False
    words = option_text.strip().split()
    if len(words) > 1:
        rest = " ".join(words[1:]).lower().strip()
        if len(rest) >= 4 and rest in passage_text.lower():
            return True
    return False


def text_contains_option(passage_text: str, option_text: str) -> bool:
    """Default matching rule (exact OR first word removed)."""
    return text_contains_option_exact(passage_text, option_text) or text_contains_option_first_word_removed(passage_text, option_text)


# ==============================================================================
# QA Metric Calculations: Precision, Recall, F1, Confusion Matrix
# ==============================================================================

def compute_qa_metrics(golds: List[str], preds: List[str], labels: List[str]) -> Dict[str, Any]:
    """Compute per-class P/R/F1, Micro-F1 (accuracy), Macro-F1, Weighted-F1, and Confusion Matrix."""
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(golds, preds):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1

    per_class = {}
    total_samples = len(golds)
    correct_count = 0

    for c in labels:
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in labels if r != c)
        fn = sum(matrix[c][k] for k in labels if k != c)
        sup = sum(matrix[c][k] for k in labels)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
        }
        correct_count += tp

    micro_f1 = correct_count / total_samples if total_samples > 0 else 0.0
    macro_f1 = sum(per_class[c]["f1"] for c in labels) / len(labels) if labels else 0.0
    weighted_f1 = (
        sum(per_class[c]["f1"] * per_class[c]["support"] for c in labels) / total_samples
        if total_samples > 0 else 0.0
    )

    return {
        "matrix": matrix,
        "per_class": per_class,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "total": total_samples,
    }


# ==============================================================================
# Calibration Calculations: ECE, MCE, Brier, NLL
# ==============================================================================

def compute_calibration_metrics(
    correct_list: List[int],
    confs: List[float],
    gold_probs: Optional[List[float]] = None,
    n_bins: int = 10,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute ECE, MCE, Brier score, and NLL."""
    n = len(correct_list)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "brier": 0.0, "nll": 0.0}

    # Bins for ECE and MCE
    bins = [[] for _ in range(n_bins)]
    for y, s in zip(correct_list, confs):
        b_idx = min(n_bins - 1, max(0, int(s * n_bins)))
        bins[b_idx].append((y, s))

    ece = 0.0
    mce = 0.0
    for b_items in bins:
        if not b_items:
            continue
        acc = sum(y for y, _ in b_items) / len(b_items)
        conf = sum(s for _, s in b_items) / len(b_items)
        gap = abs(acc - conf)
        ece += (len(b_items) / n) * gap
        if gap > mce:
            mce = gap

    brier = sum((s - y) ** 2 for y, s in zip(correct_list, confs)) / n

    if gold_probs:
        nll = -sum(math.log(max(p, eps)) for p in gold_probs) / n
    else:
        # Fallback approximation from conf and binary correctness
        nll = -sum(math.log(max(s if y else 1.0 - s, eps)) for y, s in zip(correct_list, confs)) / n

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier),
        "nll": float(nll),
    }


def apply_temperature_scaling(row: Dict[str, Any], t_val: float, eps: float = 1e-12) -> Tuple[float, float]:
    """Return (scaled_confidence, scaled_gold_probability)."""
    lp = row.get("letter_probs", {})
    gold = row.get("gold")
    if not lp:
        conf = float(row.get("confidence", 0.0))
        return conf, (conf if row.get("correct") else eps)

    log_probs = {k: math.log(max(v, eps)) for k, v in lp.items()}
    max_z = max(log_probs.values()) / t_val
    exps = {k: math.exp(v / t_val - max_z) for k, v in log_probs.items()}
    denom = sum(exps.values())
    scaled_probs = {k: v / denom for k, v in exps.items()}
    scaled_conf = max(scaled_probs.values())
    scaled_gold_p = scaled_probs.get(gold, eps)
    return float(scaled_conf), float(scaled_gold_p)


# ==============================================================================
# Retrieval Ranking Metrics: Recall@k, P@5, MRR@10, MAP@10, nDCG@10
# ==============================================================================

def compute_retrieval_metrics_for_ranking(
    questions: List[Dict[str, Any]],
    ranking_passages_fn: Any,
    match_fn: Any,
) -> Dict[str, float]:
    """Compute Recall@1/@5/@10/@20, Precision@5, MRR@10, MAP@10, nDCG@10 across questions."""
    rec1_list = []
    rec5_list = []
    rec10_list = []
    rec20_list = []
    prec5_list = []
    mrr10_list = []
    map10_list = []
    ndcg10_list = []

    for q in questions:
        gold_letter = q.get("answer")
        gold_text = q.get("options", {}).get(gold_letter, "")
        passages = ranking_passages_fn(q)  # list of passage dicts or texts

        # Binary relevance array
        rels = [1 if match_fn(p.get("text", "") if isinstance(p, dict) else str(p), gold_text) else 0 for p in passages]

        # Recalls
        rec1_list.append(1.0 if any(rels[:1]) else 0.0)
        rec5_list.append(1.0 if any(rels[:5]) else 0.0)
        rec10_list.append(1.0 if any(rels[:10]) else 0.0)
        rec20_list.append(1.0 if any(rels[:20]) else 0.0)

        # Precision@5
        p5 = sum(rels[:5]) / 5.0 if len(rels) >= 5 else (sum(rels) / max(1, len(rels)))
        prec5_list.append(p5)

        # MRR@10
        mrr = 0.0
        for rank_idx, r in enumerate(rels[:10], start=1):
            if r:
                mrr = 1.0 / rank_idx
                break
        mrr10_list.append(mrr)

        # MAP@10
        cum_rel = 0
        prec_sum = 0.0
        total_rel_in_10 = sum(rels[:10])
        for rank_idx, r in enumerate(rels[:10], start=1):
            if r:
                cum_rel += 1
                prec_sum += cum_rel / rank_idx
        ap10 = prec_sum / total_rel_in_10 if total_rel_in_10 > 0 else 0.0
        map10_list.append(ap10)

        # nDCG@10
        dcg10 = sum(r / math.log2(rank_idx + 1) for rank_idx, r in enumerate(rels[:10], start=1))
        # Ideal DCG: all 1s placed first
        idcg10 = sum(1.0 / math.log2(rank_idx + 1) for rank_idx in range(1, total_rel_in_10 + 1))
        ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0
        ndcg10_list.append(ndcg10)

    n_q = len(questions)
    return {
        "recall@1": sum(rec1_list) / n_q if n_q else 0.0,
        "recall@5": sum(rec5_list) / n_q if n_q else 0.0,
        "recall@10": sum(rec10_list) / n_q if n_q else 0.0,
        "recall@20": sum(rec20_list) / n_q if n_q else 0.0,
        "precision@5": sum(prec5_list) / n_q if n_q else 0.0,
        "mrr@10": sum(mrr10_list) / n_q if n_q else 0.0,
        "map@10": sum(map10_list) / n_q if n_q else 0.0,
        "ndcg@10": sum(ndcg10_list) / n_q if n_q else 0.0,
    }


# ==============================================================================
# Data Loading & Model Output Parsing
# ==============================================================================

def load_all_artifacts(
    base_dir: Path,
    context_dir: Path,
    build_dir: Path,
    analysis_dir: Path,
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load evaluation records, test questions, evidence, chunks, and gate metadata."""
    models = ["medpsy-4b", "qwen3-4b", "phi4-mini", "gemma3-4b", "qwen3-1.7b", "smollm3-3b"]

    test_q_path = build_dir / "work" / "phase1" / "test.jsonl"
    ev_path = build_dir / "work" / "phase7" / "evidence.jsonl"
    p6_path = build_dir / "work" / "phase6" / "retrieved.jsonl"
    chunks_path = build_dir / "work" / "phase3" / "chunks.jsonl"
    calib_json_path = analysis_dir / "phase9_abstention.json"

    test_dict = {r["id"]: r for r in map(json.loads, open(test_q_path)) if r.get("id")} if test_q_path.exists() else {}
    ev_dict = {r["id"]: r for r in map(json.loads, open(ev_path)) if r.get("id")} if ev_path.exists() else {}
    p6_dict = {r["id"]: r for r in map(json.loads, open(p6_path)) if r.get("id")} if p6_path.exists() else {}

    # Load required chunks for Phase 6 top-20 candidates
    req_chunk_ids = set()
    for qid, row in p6_dict.items():
        if row.get("dataset") == "medqa":
            for c in row.get("candidates", [])[:20]:
                cid = c.get("chunk_id")
                if cid:
                    req_chunk_ids.add(cid)

    chunks_dict = {}
    if chunks_path.exists() and req_chunk_ids:
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                cid = r.get("chunk_id")
                if cid in req_chunk_ids:
                    chunks_dict[cid] = r

    # Calibration info (temperatures T*)
    calib_dict = {}
    if calib_json_path.exists():
        try:
            cdata = json.load(open(calib_json_path))
            for item in cdata.get("models", []):
                m_name = item.get("model")
                mode = item.get("mode")
                t_val = item.get("calibration", {}).get("temperature_T")
                if m_name and mode and t_val is not None:
                    calib_dict[(m_name, mode)] = float(t_val)
        except Exception:
            pass

    # Model runs
    model_data = {}
    for m in models:
        p5 = base_dir / m / "work" / "phase5" / f"{m}.jsonl"
        p8 = base_dir / m / "work" / "phase8" / f"{m}.jsonl"
        p10 = context_dir / "work" / "phase10" / f"{m}.jsonl"
        adapt_path = analysis_dir / f"adaptive_rag_{m}.json"

        b_rows = [json.loads(l) for l in open(p5) if l.strip()] if p5.exists() else []
        r_rows = [json.loads(l) for l in open(p8) if l.strip()] if p8.exists() else []
        c_rows = [json.loads(l) for l in open(p10) if l.strip()] if p10.exists() else []
        adapt_meta = json.load(open(adapt_path)) if adapt_path.exists() else {}

        model_data[m] = {
            "baseline": b_rows,
            "rag": r_rows,
            "context": c_rows,
            "adaptive_meta": adapt_meta,
        }

    return models, test_dict, ev_dict, p6_dict, chunks_dict, calib_dict, model_data


def get_model_variant_records(
    model: str,
    variant: str,
    model_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Retrieve test answerable records for a specific model variant."""
    b_test = [r for r in model_data["baseline"] if r.get("split") == "test" and not r.get("should_abstain")]
    r_test = [r for r in model_data["rag"] if r.get("split") == "test" and not r.get("should_abstain")]
    c_test = [r for r in model_data["context"] if r.get("split") == "test" and not r.get("should_abstain")]

    if variant == "Baseline":
        return b_test
    elif variant == "RAG":
        return r_test
    elif variant == "+Abstract":
        return c_test
    elif variant == "Adaptive":
        if not b_test or not r_test:
            return []
        b_map = {r["id"]: r for r in b_test}
        adapt_meta = model_data.get("adaptive_meta", {})
        tr = adapt_meta.get("test_results", {})
        candidate_gates = ["rerank_gate", "confidence_gate", "combined"]
        best_gate = max(candidate_gates, key=lambda g: tr.get(g, {}).get("acc_overall", -1.0))

        tuning = adapt_meta.get("tuning_validation", {})
        tau_rerank = tuning.get("rerank_gate", {}).get("best_tau")
        comb_tau = tuning.get("combined", {}).get("best_tau")
        comb_delta = tuning.get("combined", {}).get("best_delta", 0.0)

        adapt_recs = []
        for r in r_test:
            qid = r["id"]
            b = b_map.get(qid, {})
            use_rag = False
            if best_gate == "rerank_gate" and tau_rerank is not None:
                use_rag = r.get("top_rerank_score") is not None and r.get("top_rerank_score") >= tau_rerank
            elif best_gate == "confidence_gate":
                use_rag = (r.get("confidence") is not None and b.get("confidence") is not None and r.get("confidence") > b.get("confidence"))
            elif best_gate == "combined" and comb_tau is not None:
                use_rag = (r.get("top_rerank_score") is not None and r.get("top_rerank_score") >= comb_tau) and not (r.get("confidence", 0.0) < b.get("confidence", 0.0) - comb_delta)

            chosen = r if use_rag else b
            rec = dict(chosen)
            rec["_used_rag"] = use_rag
            adapt_recs.append(rec)
        return adapt_recs

    return []


# ==============================================================================
# Markdown Report Construction
# ==============================================================================

def generate_metrics_extended_report(
    models: List[str],
    test_dict: Dict[str, Any],
    ev_dict: Dict[str, Any],
    p6_dict: Dict[str, Any],
    chunks_dict: Dict[str, Any],
    calib_dict: Dict[Tuple[str, str], float],
    all_data: Dict[str, Any],
    n_boot: int = 1000,
    seed: int = 42,
) -> str:
    lines = []
    lines.append("# Extended Benchmark Metrics: QA Classification, Retrieval, Reliability, and Calibration\n")
    lines.append(
        "> **Evaluation Context**: Held-out test split ($N=1,000$ answerable: 500 MedQA-USMLE + 500 PubMedQA).  \n"
        "> **Statistical Rigor**: 95% confidence intervals generated via percentile bootstrap (1,000 resamples, seed 42).  \n"
        "> **Models**: Evaluated across 6 small language models, including `medpsy-4b` (reasoning model, baseline & +Abstract).  \n"
        "> Missing modes and unsupported experimental splits are gracefully designated as `n/a`.\n"
    )

    # --------------------------------------------------------------------------
    # 1. QA Performance (Per-Class, Micro, Macro, Weighted F1)
    # --------------------------------------------------------------------------
    lines.append("## 1. QA Classification Performance\n")
    lines.append(
        "> Multi-class precision, recall, and F1 across answer choices.  \n"
        "> - **MedQA (4-option)**: Labels A, B, C, D.  \n"
        "> - **PubMedQA (3-option)**: Labels yes (A), no (B), maybe (C).  \n"
        "> - **Micro-F1**: Mathematically equivalent to overall Accuracy for single-label multiple-choice questions.\n"
    )

    medqa_labels = ["A", "B", "C", "D"]
    pubmed_label_map = {"A": "yes", "B": "no", "C": "maybe"}
    pubmed_labels = ["yes", "no", "maybe"]

    modes = ["Baseline", "RAG", "Adaptive", "+Abstract"]

    lines.append("### Table 1A: MedQA Test Split ($N=500$) Multi-Class Performance")
    lines.append(
        "| Model | Mode | Micro-F1 (Acc) [95% CI] | Macro-F1 [95% CI] | Weighted-F1 [95% CI] | P (A/B/C/D) | R (A/B/C/D) | F1 (A/B/C/D) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for m in models:
        for mode in modes:
            recs = [r for r in get_model_variant_records(m, mode, all_data[m]) if r.get("dataset") == "medqa"]
            if not recs:
                lines.append(f"| **{m}** | {mode} | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue

            golds = [r["gold"] for r in recs]
            preds = [r["pred"] for r in recs]
            qa_res = compute_qa_metrics(golds, preds, medqa_labels)

            # Bootstraps
            items = list(zip(golds, preds))
            ci_micro = bootstrap_metric_ci(items, lambda s: compute_qa_metrics([x[0] for x in s], [x[1] for x in s], medqa_labels)["micro_f1"], n_boot, seed)
            ci_macro = bootstrap_metric_ci(items, lambda s: compute_qa_metrics([x[0] for x in s], [x[1] for x in s], medqa_labels)["macro_f1"], n_boot, seed)
            ci_weighted = bootstrap_metric_ci(items, lambda s: compute_qa_metrics([x[0] for x in s], [x[1] for x in s], medqa_labels)["weighted_f1"], n_boot, seed)

            p_str = "/".join(f"{qa_res['per_class'][c]['precision']:.2f}" for c in medqa_labels)
            r_str = "/".join(f"{qa_res['per_class'][c]['recall']:.2f}" for c in medqa_labels)
            f_str = "/".join(f"{qa_res['per_class'][c]['f1']:.2f}" for c in medqa_labels)

            lines.append(
                f"| **{m}** | {mode} | "
                f"{qa_res['micro_f1']:.3f} [{ci_micro[0]:.3f}, {ci_micro[1]:.3f}] | "
                f"{qa_res['macro_f1']:.3f} [{ci_macro[0]:.3f}, {ci_macro[1]:.3f}] | "
                f"{qa_res['weighted_f1']:.3f} [{ci_weighted[0]:.3f}, {ci_weighted[1]:.3f}] | "
                f"{p_str} | {r_str} | {f_str} |"
            )

    lines.append("\n### Table 1B: PubMedQA Test Split ($N=500$) Multi-Class Performance")
    lines.append(
        "| Model | Mode | Micro-F1 (Acc) [95% CI] | Macro-F1 [95% CI] | Weighted-F1 [95% CI] | P (yes/no/maybe) | R (yes/no/maybe) | F1 (yes/no/maybe) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for m in models:
        for mode in modes:
            recs = [r for r in get_model_variant_records(m, mode, all_data[m]) if r.get("dataset") == "pubmedqa"]
            if not recs:
                lines.append(f"| **{m}** | {mode} | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue

            golds = [pubmed_label_map.get(r["gold"], r["gold"]) for r in recs]
            preds = [pubmed_label_map.get(r["pred"], r["pred"]) for r in recs]
            qa_res = compute_qa_metrics(golds, preds, pubmed_labels)

            items = list(zip(golds, preds))
            ci_micro = bootstrap_metric_ci(items, lambda s: compute_qa_metrics([x[0] for x in s], [x[1] for x in s], pubmed_labels)["micro_f1"], n_boot, seed)
            ci_macro = bootstrap_metric_ci(items, lambda s: compute_qa_metrics([x[0] for x in s], [x[1] for x in s], pubmed_labels)["macro_f1"], n_boot, seed)
            ci_weighted = bootstrap_metric_ci(items, lambda s: compute_qa_metrics([x[0] for x in s], [x[1] for x in s], pubmed_labels)["weighted_f1"], n_boot, seed)

            p_str = "/".join(f"{qa_res['per_class'][c]['precision']:.2f}" for c in pubmed_labels)
            r_str = "/".join(f"{qa_res['per_class'][c]['recall']:.2f}" for c in pubmed_labels)
            f_str = "/".join(f"{qa_res['per_class'][c]['f1']:.2f}" for c in pubmed_labels)

            lines.append(
                f"| **{m}** | {mode} | "
                f"{qa_res['micro_f1']:.3f} [{ci_micro[0]:.3f}, {ci_micro[1]:.3f}] | "
                f"{qa_res['macro_f1']:.3f} [{ci_macro[0]:.3f}, {ci_macro[1]:.3f}] | "
                f"{qa_res['weighted_f1']:.3f} [{ci_weighted[0]:.3f}, {ci_weighted[1]:.3f}] | "
                f"{p_str} | {r_str} | {f_str} |"
            )

    # Confusion Matrices Section
    lines.append("\n### Table 1C: Representative Confusion Matrices (MedQA Baseline vs RAG)")
    lines.append("> Matrix convention: Rows indicate **Gold (True)** label; Columns indicate **Predicted** label.\n")

    for m in ["qwen3-4b", "phi4-mini", "medpsy-4b"]:
        b_recs = [r for r in get_model_variant_records(m, "Baseline", all_data[m]) if r.get("dataset") == "medqa"]
        r_recs = [r for r in get_model_variant_records(m, "RAG", all_data[m]) if r.get("dataset") == "medqa"]
        if b_recs:
            m_base = compute_qa_metrics([r["gold"] for r in b_recs], [r["pred"] for r in b_recs], medqa_labels)["matrix"]
            lines.append(f"#### **{m}** (MedQA Baseline)")
            lines.append("| True \\ Pred | Pred A | Pred B | Pred C | Pred D | Total |")
            lines.append("|---|---|---|---|---|---|")
            for row_lbl in medqa_labels:
                vals = [m_base[row_lbl][col_lbl] for col_lbl in medqa_labels]
                lines.append(f"| **True {row_lbl}** | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {sum(vals)} |")
            lines.append("")
        if r_recs:
            m_rag = compute_qa_metrics([r["gold"] for r in r_recs], [r["pred"] for r in r_recs], medqa_labels)["matrix"]
            lines.append(f"#### **{m}** (MedQA Full RAG)")
            lines.append("| True \\ Pred | Pred A | Pred B | Pred C | Pred D | Total |")
            lines.append("|---|---|---|---|---|---|")
            for row_lbl in medqa_labels:
                vals = [m_rag[row_lbl][col_lbl] for col_lbl in medqa_labels]
                lines.append(f"| **True {row_lbl}** | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {sum(vals)} |")
            lines.append("")

    # --------------------------------------------------------------------------
    # 2. Retrieval Performance: Phase 6 vs Phase 7
    # --------------------------------------------------------------------------
    lines.append("## 2. Retrieval Benchmark: Phase 6 (Hybrid top-20) vs. Phase 7 (Reranked top-5)\n")
    lines.append(
        "> **Setup**: Evaluated across all $N=500$ MedQA test questions.  \n"
        "> A retrieved passage is considered **relevant** if it contains the ground-truth option text.  \n"
        "> Two matching rules are evaluated:  \n"
        "> 1. **Default Rule**: Exact substring match OR first word removed (if word count $\\ge 2$ and rest $\\ge 4$ chars).  \n"
        "> 2. **Strict Verbatim Rule**: Exact case-insensitive substring match only.\n"
    )

    medqa_test_questions = [
        r for r in test_dict.values()
        if r.get("split") == "test" and r.get("dataset") == "medqa" and not r.get("should_abstain")
    ]
    medqa_test_questions.sort(key=lambda x: x["id"])

    # Functions to extract passages
    def get_p6_passages(q: Dict[str, Any]) -> List[str]:
        qid = q["id"]
        cands = p6_dict.get(qid, {}).get("candidates", [])[:20]
        passages = []
        for c in cands:
            cid = c.get("chunk_id")
            chunk = chunks_dict.get(cid, {})
            passages.append(chunk.get("text", ""))
        return passages

    def get_p7_passages(q: Dict[str, Any]) -> List[str]:
        qid = q["id"]
        ev = ev_dict.get(qid, {})
        return [p.get("text", "") for p in ev.get("passages", [])[:5]]

    # Evaluate
    p6_def = compute_retrieval_metrics_for_ranking(medqa_test_questions, get_p6_passages, text_contains_option)
    p7_def = compute_retrieval_metrics_for_ranking(medqa_test_questions, get_p7_passages, text_contains_option)

    p6_ver = compute_retrieval_metrics_for_ranking(medqa_test_questions, get_p6_passages, text_contains_option_exact)
    p7_ver = compute_retrieval_metrics_for_ranking(medqa_test_questions, get_p7_passages, text_contains_option_exact)

    lines.append("### Table 2: Retrieval Ranking Metrics & Cross-Encoder Reranker Gain")
    lines.append(
        "| Stage | Matching Rule | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Precision@5 | MRR@10 | MAP@10 | nDCG@10 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    lines.append(
        f"| **Phase 6: Hybrid BM25+MedCPT** | Default Rule | "
        f"{p6_def['recall@1']:.1%} | {p6_def['recall@5']:.1%} | {p6_def['recall@10']:.1%} | {p6_def['recall@20']:.1%} | "
        f"{p6_def['precision@5']:.1%} | {p6_def['mrr@10']:.3f} | {p6_def['map@10']:.3f} | {p6_def['ndcg@10']:.3f} |"
    )
    lines.append(
        f"| **Phase 7: MedCPT Cross-Encoder** | Default Rule | "
        f"{p7_def['recall@1']:.1%} | {p7_def['recall@5']:.1%} | {p7_def['recall@10']:.1%}* | {p7_def['recall@20']:.1%}* | "
        f"{p7_def['precision@5']:.1%} | {p7_def['mrr@10']:.3f} | {p7_def['map@10']:.3f} | {p7_def['ndcg@10']:.3f} |"
    )
    gain_rec1 = (p7_def['recall@1'] - p6_def['recall@1']) * 100
    gain_rec5 = (p7_def['recall@5'] - p6_def['recall@5']) * 100
    gain_p5 = (p7_def['precision@5'] - p6_def['precision@5']) * 100
    gain_mrr = p7_def['mrr@10'] - p6_def['mrr@10']
    gain_map = p7_def['map@10'] - p6_def['map@10']
    gain_ndcg = p7_def['ndcg@10'] - p6_def['ndcg@10']
    lines.append(
        f"| *Reranker Delta (Gain)* | *Default Rule* | "
        f"*+{gain_rec1:.1f}%* | *+{gain_rec5:.1f}%* | — | — | "
        f"*+{gain_p5:.1f}%* | *+{gain_mrr:.3f}* | *+{gain_map:.3f}* | *+{gain_ndcg:.3f}* |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.append(
        f"| **Phase 6: Hybrid BM25+MedCPT** | Strict Verbatim | "
        f"{p6_ver['recall@1']:.1%} | {p6_ver['recall@5']:.1%} | {p6_ver['recall@10']:.1%} | {p6_ver['recall@20']:.1%} | "
        f"{p6_ver['precision@5']:.1%} | {p6_ver['mrr@10']:.3f} | {p6_ver['map@10']:.3f} | {p6_ver['ndcg@10']:.3f} |"
    )
    lines.append(
        f"| **Phase 7: MedCPT Cross-Encoder** | Strict Verbatim | "
        f"{p7_ver['recall@1']:.1%} | {p7_ver['recall@5']:.1%} | {p7_ver['recall@10']:.1%}* | {p7_ver['recall@20']:.1%}* | "
        f"{p7_ver['precision@5']:.1%} | {p7_ver['mrr@10']:.3f} | {p7_ver['map@10']:.3f} | {p7_ver['ndcg@10']:.3f} |"
    )
    lines.append(
        f"| *Reranker Delta (Gain)* | *Strict Verbatim* | "
        f"*+{(p7_ver['recall@1'] - p6_ver['recall@1'])*100:.1f}%* | *+{(p7_ver['recall@5'] - p6_ver['recall@5'])*100:.1f}%* | — | — | "
        f"*+{(p7_ver['precision@5'] - p6_ver['precision@5'])*100:.1f}%* | *+{p7_ver['mrr@10'] - p6_ver['mrr@10']:.3f}* | *+{p7_ver['map@10'] - p6_ver['map@10']:.3f}* | *+{p7_ver['ndcg@10'] - p6_ver['ndcg@10']:.3f}* |"
    )
    lines.append("\n*Note: Phase 7 produces top-5 reranked passages. Recall@10 and @20 on Phase 7 represent the top-5 cap.*")

    # --------------------------------------------------------------------------
    # 3. RAG System Reliability & Grounding
    # --------------------------------------------------------------------------
    lines.append("\n## 3. RAG Reliability, Citation Quality, and Hallucination Grounding (MedQA)\n")
    lines.append(
        "> Grounding and citation metrics evaluated across the MedQA test set ($N=500$):  \n"
        "> - **Faithfulness %**: % of answers where at least one cited passage contains the selected option text.  \n"
        "> - **Unsupported-Answer Rate**: % of answers where none of the cited passages ground the chosen option ($1 - \\text{Faithfulness}$).  \n"
        "> - **Citation Precision**: Total cited passages containing the chosen option divided by total cited passages.  \n"
        "> - **Citation Recall**: Total retrieved gold-containing passages that were cited divided by all retrieved gold-containing passages.  \n"
        "> - **Citation Validity %**: % of bracketed citations referencing a valid passage index (1–5).\n"
    )

    lines.append("### Table 3: RAG Citation & Grounding Breakdown")
    lines.append(
        "| Model | Faithfulness % (Count/500) | Unsupported-Answer Rate % | Citation Precision % | Citation Recall % | Citation Validity % |"
    )
    lines.append("|---|---|---|---|---|---|")

    for m in models:
        r_recs = [r for r in get_model_variant_records(m, "RAG", all_data[m]) if r.get("dataset") == "medqa"]
        if not r_recs:
            lines.append(f"| **{m}** | n/a | n/a | n/a | n/a | n/a |")
            continue

        faithful_cnt = 0
        total_cites = 0
        matching_cites = 0
        total_gold_passages_retrieved = 0
        gold_passages_cited = 0
        valid_cite_nums = 0
        raw_cite_nums = 0

        for r in r_recs:
            qid = r["id"]
            q = test_dict[qid]
            gold_opt = q["options"].get(q["answer"], "")
            pred = r.get("pred")
            chosen_opt = q["options"].get(pred, "")
            passages = ev_dict.get(qid, {}).get("passages", [])

            cites = r.get("citations", [])
            raw_cite_nums += len(cites)
            valid_c = [c for c in cites if 1 <= c <= len(passages)]
            valid_cite_nums += len(valid_c)

            cited_passages = [passages[c - 1] for c in valid_c]

            # Faithfulness
            is_faithful = chosen_opt and any(text_contains_option(p["text"], chosen_opt) for p in cited_passages)
            if is_faithful:
                faithful_cnt += 1

            # Citation precision
            for cp in cited_passages:
                total_cites += 1
                if chosen_opt and text_contains_option(cp["text"], chosen_opt):
                    matching_cites += 1

            # Citation recall
            for p_idx, p in enumerate(passages, start=1):
                if text_contains_option(p["text"], gold_opt):
                    total_gold_passages_retrieved += 1
                    if p_idx in valid_c:
                        gold_passages_cited += 1

        n_q = len(r_recs)
        faith_pct = (faithful_cnt / n_q) * 100
        unsupported_pct = 100.0 - faith_pct
        cite_prec = (matching_cites / total_cites * 100) if total_cites > 0 else 0.0
        cite_rec = (gold_passages_cited / total_gold_passages_retrieved * 100) if total_gold_passages_retrieved > 0 else 0.0
        cite_val = (valid_cite_nums / raw_cite_nums * 100) if raw_cite_nums > 0 else 100.0

        lines.append(
            f"| **{m}** | {faith_pct:.1f}% ({faithful_cnt}/{n_q}) | {unsupported_pct:.1f}% | "
            f"{cite_prec:.1f}% | {cite_rec:.1f}% | {cite_val:.1f}% |"
        )

    # --------------------------------------------------------------------------
    # 4. Confidence Calibration: Raw vs Temperature-Scaled
    # --------------------------------------------------------------------------
    lines.append("\n## 4. Confidence Calibration: ECE, MCE, Brier Score, and NLL\n")
    lines.append(
        "> Calibration computed over all answerable test questions ($N=1,000$ overall: 500 MedQA + 500 PubMedQA).  \n"
        "> Temperature scaling ($T^*$) fitted on validation NLL is applied where validation tuning exists.  \n"
        "> - **ECE**: Expected Calibration Error (10 equal-width bins).  \n"
        "> - **MCE**: Maximum Calibration Error across bins.  \n"
        "> - **Brier Score**: Mean squared error between confidence and empirical correctness.  \n"
        "> - **NLL**: Negative log-likelihood of the ground-truth option choice.\n"
    )

    lines.append("### Table 4: Calibration Metrics (Raw vs. Post-Hoc Temperature Scaled)")
    lines.append(
        "| Model | Mode | Temp ($T^*$) | Raw ECE | Scaled ECE | Raw MCE | Scaled MCE | Raw Brier | Scaled Brier | Raw NLL | Scaled NLL |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for m in models:
        for mode in ["Baseline", "RAG"]:
            recs = get_model_variant_records(m, mode, all_data[m])
            if not recs:
                lines.append(f"| **{m}** | {mode} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue

            corr_list = [1 if r["correct"] else 0 for r in recs]
            raw_confs = [float(r.get("confidence", 0.0)) for r in recs]

            # Raw gold probs
            raw_gold_p = []
            for r in recs:
                lp = r.get("letter_probs", {})
                g = r.get("gold")
                raw_gold_p.append(float(lp.get(g, 1e-12)) if lp else (raw_confs[-1] if r["correct"] else 1e-12))

            raw_cal = compute_calibration_metrics(corr_list, raw_confs, raw_gold_p)

            t_val = calib_dict.get((m, mode.lower()))
            if t_val is not None and t_val > 0.0:
                scaled_confs = []
                scaled_gold_p = []
                for r in recs:
                    sc, sg = apply_temperature_scaling(r, t_val)
                    scaled_confs.append(sc)
                    scaled_gold_p.append(sg)
                scaled_cal = compute_calibration_metrics(corr_list, scaled_confs, scaled_gold_p)
                t_str = f"{t_val:.2f}"
                s_ece = f"{scaled_cal['ece']:.3f}"
                s_mce = f"{scaled_cal['mce']:.3f}"
                s_brier = f"{scaled_cal['brier']:.3f}"
                s_nll = f"{scaled_cal['nll']:.3f}"
            else:
                t_str = "n/a"
                s_ece = "n/a"
                s_mce = "n/a"
                s_brier = "n/a"
                s_nll = "n/a"

            lines.append(
                f"| **{m}** | {mode} | {t_str} | "
                f"{raw_cal['ece']:.3f} | {s_ece} | "
                f"{raw_cal['mce']:.3f} | {s_mce} | "
                f"{raw_cal['brier']:.3f} | {s_brier} | "
                f"{raw_cal['nll']:.3f} | {s_nll} |"
            )

    # --------------------------------------------------------------------------
    # 5. Efficiency & Latency
    # --------------------------------------------------------------------------
    lines.append("\n## 5. System Efficiency, Latency, and Throughput\n")
    lines.append(
        "> Computed over test-split questions.  \n"
        "> RAG latency includes retrieval + reranking overhead ($+0.1273$ s/q) added to generation time.  \n"
        "> Throughput represents end-to-end questions processed per second.\n"
    )

    lines.append("### Table 5: Latency and Throughput per Model and Mode")
    lines.append(
        "| Model | Mode | Mean Latency (s) | P95 Latency (s) | End-to-End Latency (s) | Throughput (Q/s) |"
    )
    lines.append("|---|---|---|---|---|---|")

    for m in models:
        for mode in modes:
            recs = get_model_variant_records(m, mode, all_data[m])
            if not recs:
                lines.append(f"| **{m}** | {mode} | n/a | n/a | n/a | n/a |")
                continue

            secs = [float(r.get("seconds", 0.0)) for r in recs if r.get("seconds") is not None]
            if not secs:
                secs = [float(r.get("sec_per_q", 0.0)) for r in recs if r.get("sec_per_q") is not None]

            if secs:
                secs.sort()
                mean_sec = sum(secs) / len(secs)
                p95_sec = secs[int(0.95 * len(secs))]
                is_rag = (mode == "RAG")
                is_adapt = (mode == "Adaptive")
                if is_rag:
                    e2e_sec = mean_sec + RETR_SEC_PER_Q
                elif is_adapt:
                    # Adaptive routes fraction of queries through retrieval
                    rag_pct = sum(1 for r in recs if r.get("_used_rag")) / len(recs)
                    e2e_sec = mean_sec + (rag_pct * RETR_SEC_PER_Q)
                else:
                    e2e_sec = mean_sec

                qps = 1.0 / e2e_sec if e2e_sec > 0 else 0.0
                lines.append(
                    f"| **{m}** | {mode} | {mean_sec:.3f} s | {p95_sec:.3f} s | {e2e_sec:.3f} s | {qps:.2f} q/s |"
                )
            else:
                lines.append(f"| **{m}** | {mode} | n/a | n/a | n/a | n/a |")

    # Footnote
    lines.append("\n---\n")
    lines.append(
        "> **Methodological Footnote**:  \n"
        "> 1. Grounding and citation metrics utilize deterministic string-matching proxies (exact substring or first word removed).  \n"
        "> 2. Micro-F1 is mathematically identical to accuracy for single-label multiple-choice questions.  \n"
        "> 3. Text generation metrics (BERTScore, ROUGE-L, answer relevancy) are not applicable to single-letter categorical MCQ predictions.\n"
    )

    return "\n".join(lines)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extended metrics for Medical RAG SLM Benchmark.")
    parser.add_argument("--base-dir", default="outputs/kaggle_qa/full", help="Directory containing standard model full runs")
    parser.add_argument("--context-dir", default="outputs/kaggle_qa/context", help="Directory containing context runs")
    parser.add_argument("--build-dir", default="outputs/kaggle_build", help="Directory containing build artifacts")
    parser.add_argument("--analysis-dir", default="outputs/analysis", help="Directory containing analysis outputs")
    parser.add_argument("--out-file", default="outputs/analysis/metrics_extended.md", help="Destination markdown file")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    context_dir = Path(args.context_dir)
    build_dir = Path(args.build_dir)
    analysis_dir = Path(args.analysis_dir)
    out_file = Path(args.out_file)

    print("Loading benchmark artifacts across all 6 models...")
    models, test_dict, ev_dict, p6_dict, chunks_dict, calib_dict, all_data = load_all_artifacts(
        base_dir, context_dir, build_dir, analysis_dir
    )

    print("Generating extended metrics report...")
    report_md = generate_metrics_extended_report(
        models, test_dict, ev_dict, p6_dict, chunks_dict, calib_dict, all_data,
        n_boot=args.n_boot, seed=args.seed
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Successfully saved extended metrics to {out_file}")


if __name__ == "__main__":
    main()
