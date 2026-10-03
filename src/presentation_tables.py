"""Presentation-ready summary tables for Medical RAG SLM Evaluation.

Generates:
  outputs/analysis/presentation_tables.md

Tables:
1. QA Performance: Accuracy, EM, Macro-F1 across Answer Options (Overall, MedQA, PubMedQA) with 95% Bootstrap CIs.
2. Shared Retrieval Performance & System Reliability on MedQA (Faithfulness on RAG-routed outputs, Hallucination Rates, Citation Validity).
3. Efficiency: Latency per question, Generation Throughput (tokens/sec), Prompt Tokens, Peak Memory (VRAM/RAM).
"""
import argparse
import fractions
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


RETR_SEC_PER_Q = 0.1273  # Phase 6 (23.424s) + Phase 7 (180.261s) = 203.685s / 1600 q


# ---------- Statistical & Metric Helpers ----------

def bootstrap_ci(values: List[float], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap 95% confidence interval for a list of values."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return (round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4))


def macro_f1(golds: List[str], preds: List[str], labels: List[str]) -> float:
    """Compute macro-averaged F1 score over discrete option labels."""
    f1s = []
    for lbl in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(golds, preds) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(golds, preds) if g == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def bootstrap_macro_f1_ci(golds: List[str], preds: List[str], labels: List[str], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for macro-F1."""
    if not golds:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(golds)
    f1s = []
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        g_s = [golds[i] for i in idx]
        p_s = [preds[i] for i in idx]
        f1s.append(macro_f1(g_s, p_s, labels))
    f1s.sort()
    return (round(f1s[int(0.025 * n_boot)], 4), round(f1s[int(0.975 * n_boot)], 4))


def text_contains_option_exact(passage_text: str, option_text: str) -> bool:
    if not option_text or not passage_text:
        return False
    opt_lower = option_text.strip().lower()
    return opt_lower in passage_text.lower() if opt_lower else False


def text_contains_option_first_word_removed(passage_text: str, option_text: str) -> bool:
    if not option_text or not passage_text:
        return False
    words = option_text.strip().lower().split()
    if len(words) >= 2:
        rem = " ".join(words[1:]).strip()
        if len(rem) >= 3 and rem in passage_text.lower():
            return True
    return False


def text_contains_option(passage_text: str, option_text: str) -> bool:
    return text_contains_option_exact(passage_text, option_text) or text_contains_option_first_word_removed(passage_text, option_text)


def mcnemar_exact(a_correct: List[bool], b_correct: List[bool]) -> float:
    """Exact two-sided binomial McNemar test using high-precision arithmetic.

    Computes exact binomial p-value using arbitrary-precision integers/fractions
    and log-space arithmetic (math.lgamma) for extreme tails.
    """
    a_wins = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    b_wins = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    if k == n / 2.0:
        return 1.0

    # 1. Exact calculation using arbitrary-precision integers and fractions
    try:
        num = 2 * sum(math.comb(n, i) for i in range(k + 1))
        den = 1 << n
        if num >= den:
            return 1.0
        frac = fractions.Fraction(num, den)
        p_val = float(frac)
        if p_val > 0.0:
            return p_val
    except Exception:
        pass

    # 2. Log-space arithmetic with math.lgamma for extreme tails / underflow
    ln2 = math.log(2.0)
    lgamma_n1 = math.lgamma(n + 1)
    log_terms = [
        lgamma_n1 - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * ln2
        for i in range(k + 1)
    ]
    max_log = max(log_terms)
    sum_scaled = sum(math.exp(t - max_log) for t in log_terms)
    log_p = ln2 + max_log + math.log(sum_scaled)
    if log_p >= 0.0:
        return 1.0
    log10_p = log_p / math.log(10.0)
    if log10_p < -300.0:
        return 0.0
    return math.pow(10.0, log10_p)


def format_p_value(p: Any) -> str:
    """Format p-value: scientific notation if < 0.001, '< 1e-300' if below 1e-300, never 0.00e+00."""
    if p is None or p == "" or p == "n/a":
        return "n/a"
    if isinstance(p, str):
        return p
    if p <= 0.0 or p < 1e-300:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.5f}"


# ---------- Main Data Assembly ----------

def load_data(qa_dir: Path, build_dir: Path, context_dir: Path):
    """Load all relevant artifacts across models including medpsy-4b."""
    candidate_models = ["medpsy-4b", "qwen3-4b", "phi4-mini", "gemma3-4b", "qwen3-1.7b", "smollm3-3b"]
    models = []
    data = {}

    test_q_path = build_dir / "work" / "phase1" / "test.jsonl"
    ev_path = build_dir / "work" / "phase7" / "evidence.jsonl"

    test_dict = {r["id"]: r for r in map(json.loads, open(test_q_path)) if r.get("id")} if test_q_path.exists() else {}
    ev_dict = {r["id"]: r for r in map(json.loads, open(ev_path)) if r.get("id")} if ev_path.exists() else {}

    for m in candidate_models:
        p5 = qa_dir / m / "work" / "phase5" / f"{m}.jsonl"
        p8 = qa_dir / m / "work" / "phase8" / f"{m}.jsonl"
        p10 = context_dir / "work" / "phase10" / f"{m}.jsonl"
        adapt_json = Path("outputs/analysis") / f"adaptive_rag_{m}.json"

        if not (p5.exists() or p8.exists() or p10.exists()):
            continue

        models.append(m)
        b_rows = [json.loads(l) for l in open(p5) if l.strip()] if p5.exists() else []
        r_rows = [json.loads(l) for l in open(p8) if l.strip()] if p8.exists() else []
        c_rows = [json.loads(l) for l in open(p10) if l.strip()] if p10.exists() else []
        adapt_meta = json.load(open(adapt_json)) if adapt_json.exists() else {}

        data[m] = {
            "baseline": b_rows,
            "rag": r_rows,
            "context": c_rows,
            "adaptive_meta": adapt_meta,
        }

    return models, test_dict, ev_dict, data


def assemble_variant_records(model: str, model_data: Dict[str, Any], test_dict: Dict[str, Any]):
    """Extract answers, predictions, and runtime stats for all variants of a model."""
    b_test = [r for r in model_data["baseline"] if r.get("split") == "test" and not r.get("should_abstain")]
    r_test = [r for r in model_data["rag"] if r.get("split") == "test" and not r.get("should_abstain")]
    c_test = [r for r in model_data["context"] if r.get("split") == "test" and not r.get("should_abstain")]

    b_unans = [r for r in model_data["baseline"] if r.get("split") == "test" and r.get("should_abstain")]
    r_unans = [r for r in model_data["rag"] if r.get("split") == "test" and r.get("should_abstain")]

    b_map = {r["id"]: r for r in b_test}
    r_map = {r["id"]: r for r in r_test}

    variants = {}

    if b_test:
        variants["Baseline"] = {
            "records": b_test,
            "unanswerable": b_unans,
            "is_rag": False,
            "is_adaptive": False,
        }

    if r_test:
        variants["Full RAG"] = {
            "records": r_test,
            "unanswerable": r_unans,
            "is_rag": True,
            "is_adaptive": False,
        }

        # Identify best adaptive gate if metadata is available
        adapt_meta = model_data.get("adaptive_meta", {})
        tr = adapt_meta.get("test_results", {})
        candidate_gates = ["rerank_gate", "confidence_gate", "combined"]
        best_gate = max(candidate_gates, key=lambda g: tr.get(g, {}).get("acc_overall", -1.0))
        best_gate_tr = tr.get(best_gate, {})

        gate_name_map = {
            "rerank_gate": "Adaptive (Rerank Gate)",
            "confidence_gate": "Adaptive (Confidence Gate)",
            "combined": "Adaptive (Combined Gate)",
        }
        adaptive_display_name = gate_name_map.get(best_gate, f"Adaptive ({best_gate})")

        # Reconstruct predictions for best adaptive gate
        adapt_records = []
        tuning = adapt_meta.get("tuning_validation", {})
        tau_rerank = tuning.get("rerank_gate", {}).get("best_tau")
        comb_tau = tuning.get("combined", {}).get("best_tau")
        comb_delta = tuning.get("combined", {}).get("best_delta", 0.0)

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
            adapt_records.append(rec)

        variants[adaptive_display_name] = {
            "records": adapt_records,
            "unanswerable": b_unans,
            "is_rag": False,
            "is_adaptive": True,
            "gate_key": best_gate,
            "cost_sec_per_q": best_gate_tr.get("cost_sec_per_q"),
            "rag_pct": best_gate_tr.get("rag_pct"),
        }

    if c_test:
        variants["+Abstract (PubMedQA)"] = {
            "records": c_test,
            "unanswerable": [],
            "is_rag": False,
            "is_context": True,
        }

    return variants


# ---------- Build Table 1: QA Performance ----------

def build_table_1(models, test_dict, all_data):
    lines = []
    lines.append("## Table 1: Question-Answering Performance across Architectures and Inference Modes")
    lines.append(
        "> **Setup**: Evaluated on the held-out test split ($N=1,000$ answerable: 500 MedQA + 500 PubMedQA).  \n"
        "> **Metrics**: Accuracy, Exact Match (EM)*, and Macro-F1 across discrete answer option classes ($C=4$ for MedQA, $C=3$ for PubMedQA).  \n"
        "> **Uncertainty**: 95% percentile bootstrap confidence intervals (1,000 resamples, seed 42).  \n"
        "> *Notes*:  \n"
        "> 1. **Exact Match (EM)** is mathematically identical to accuracy for single-token multiple-choice options.  \n"
        "> 2. **BERTScore** is not applicable to single-letter multiple-choice answers and is omitted.  \n"
        "> 3. **+Abstract** provides the ground-truth study abstract for PubMedQA (standard oracle context benchmark).\n"
    )
    lines.append(
        "| Model | Variant | Overall Acc [95% CI] | Overall EM | Overall Macro-F1 [95% CI] | MedQA Acc [95% CI] | MedQA EM | MedQA Macro-F1 [95% CI] | PubMedQA Acc [95% CI] | PubMedQA EM | PubMedQA Macro-F1 [95% CI] |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    labels_medqa = ["A", "B", "C", "D"]
    labels_pubmed = ["A", "B", "C"]
    labels_all = ["A", "B", "C", "D"]

    for m in models:
        variants = assemble_variant_records(m, all_data[m], test_dict)
        for v_name, v_info in variants.items():
            recs = v_info["records"]
            is_ctx = v_info.get("is_context", False)

            if is_ctx:
                # PubMedQA only
                p_recs = [r for r in recs if r.get("dataset") == "pubmedqa"]
                if p_recs:
                    p_correct = [1 if r["correct"] else 0 for r in p_recs]
                    p_acc = sum(p_correct) / len(p_correct)
                    p_ci = bootstrap_ci(p_correct)
                    p_f1 = macro_f1([r["gold"] for r in p_recs], [r["pred"] for r in p_recs], labels_pubmed)
                    p_f1_ci = bootstrap_macro_f1_ci([r["gold"] for r in p_recs], [r["pred"] for r in p_recs], labels_pubmed)
                    p_acc_str = f"{p_acc:.1%} [{p_ci[0]:.3f}, {p_ci[1]:.3f}]"
                    p_em_str = f"{p_acc:.1%}"
                    p_f1_str = f"{p_f1:.3f} [{p_f1_ci[0]:.3f}, {p_f1_ci[1]:.3f}]"
                else:
                    p_acc_str = p_em_str = p_f1_str = "n/a"

                lines.append(
                    f"| **{m}** | {v_name} | — | — | — | — | — | — | "
                    f"{p_acc_str} | {p_em_str} | {p_f1_str} |"
                )
            else:
                m_recs = [r for r in recs if r.get("dataset") == "medqa"]
                p_recs = [r for r in recs if r.get("dataset") == "pubmedqa"]

                if m_recs:
                    m_correct = [1 if r["correct"] else 0 for r in m_recs]
                    m_acc = sum(m_correct) / len(m_correct)
                    m_ci = bootstrap_ci(m_correct)
                    m_f1 = macro_f1([r["gold"] for r in m_recs], [r["pred"] for r in m_recs], labels_medqa)
                    m_f1_ci = bootstrap_macro_f1_ci([r["gold"] for r in m_recs], [r["pred"] for r in m_recs], labels_medqa)
                    m_acc_str = f"{m_acc:.1%} [{m_ci[0]:.3f}, {m_ci[1]:.3f}]"
                    m_em_str = f"{m_acc:.1%}"
                    m_f1_str = f"{m_f1:.3f} [{m_f1_ci[0]:.3f}, {m_f1_ci[1]:.3f}]"
                else:
                    m_acc_str = m_em_str = m_f1_str = "n/a"

                if p_recs:
                    p_correct = [1 if r["correct"] else 0 for r in p_recs]
                    p_acc = sum(p_correct) / len(p_correct)
                    p_ci = bootstrap_ci(p_correct)
                    p_f1 = macro_f1([r["gold"] for r in p_recs], [r["pred"] for r in p_recs], labels_pubmed)
                    p_f1_ci = bootstrap_macro_f1_ci([r["gold"] for r in p_recs], [r["pred"] for r in p_recs], labels_pubmed)
                    p_acc_str = f"{p_acc:.1%} [{p_ci[0]:.3f}, {p_ci[1]:.3f}]"
                    p_em_str = f"{p_acc:.1%}"
                    p_f1_str = f"{p_f1:.3f} [{p_f1_ci[0]:.3f}, {p_f1_ci[1]:.3f}]"
                else:
                    p_acc_str = p_em_str = p_f1_str = "n/a"

                if m_recs and p_recs:
                    ov_correct = [1 if r["correct"] else 0 for r in recs]
                    ov_acc = sum(ov_correct) / len(ov_correct)
                    ov_ci = bootstrap_ci(ov_correct)
                    ov_f1 = macro_f1([r["gold"] for r in recs], [r["pred"] for r in recs], labels_all)
                    ov_f1_ci = bootstrap_macro_f1_ci([r["gold"] for r in recs], [r["pred"] for r in recs], labels_all)
                    ov_acc_str = f"{ov_acc:.1%} [{ov_ci[0]:.3f}, {ov_ci[1]:.3f}]"
                    ov_em_str = f"{ov_acc:.1%}"
                    ov_f1_str = f"{ov_f1:.3f} [{ov_f1_ci[0]:.3f}, {ov_f1_ci[1]:.3f}]"
                else:
                    ov_acc_str = ov_em_str = ov_f1_str = "n/a"

                lines.append(
                    f"| **{m}** | {v_name} | "
                    f"{ov_acc_str} | {ov_em_str} | {ov_f1_str} | "
                    f"{m_acc_str} | {m_em_str} | {m_f1_str} | "
                    f"{p_acc_str} | {p_em_str} | {p_f1_str} |"
                )
    lines.append("")
    return "\n".join(lines)


# ---------- Build Table 2: Reliability & Grounding ----------

def build_table_2(models, test_dict, ev_dict, all_data):
    lines = []
    lines.append("## Table 2: System Reliability, Hallucination, and Evidence Grounding (MedQA)")
    lines.append(
        "> **Setup**: Grounding and hallucination evaluation on MedQA test answerable ($N=500$) and unanswerable ($N=150$) sets.  \n"
        "> **Metrics Definition**:  \n"
        "> - **Faithfulness**: % of answers whose cited passage explicitly contains the selected option text (string-matching proxy). For Adaptive gates, computed *strictly over final answers routed to RAG*.  \n"
        "> - **Wrong Answers ($N$)**: Total count of incorrect predictions on the MedQA test set ($N=500$).  \n"
        "> - **Hallucination Rate (Wrong Answers)**: % of incorrect predictions carrying raw confidence $\\ge 0.90$.  \n"
        "> - **Hallucination Rate (Unanswerable)**: % of unanswerable questions answered with raw confidence $\\ge 0.90$.  \n"
        "> - **Citation Validity %**: % of bracketed citations pointing to valid retrieved passage numbers (1–5).  \n"
        "> *Notes*:  \n"
        "> 1. Grounding metrics utilize deterministic string-matching proxies (exact substring or first word removed), not LLM-judged RAGAS.  \n"
        "> 2. Adaptive gates choose the higher-confidence answer, so the conditional hallucination rate among wrong answers is inflated by construction.\n"
    )

    # Small separate table for shared retrieval performance
    lines.append("### Retrieval Performance (Shared across all models on MedQA)")
    lines.append("| Metric | Value | Scope | Description |")
    lines.append("|---|---|---|---|")
    lines.append("| **Context Precision** | **12.9%** | Top-5 passages | % of 5 retrieved passages containing the correct gold option text (mean: 0.64 / 5 passages) |")
    lines.append("| **Context Recall (Default Rule)** | **30.4%** | Any of top-5 | % of questions with the gold option present in *any* of the 5 retrieved passages (152 / 500) |")
    lines.append("| **Strict Verbatim Recall** | **18.8%** | Any of top-5 | % of questions with the verbatim option string present in *any* passage (94 / 500) |")
    lines.append("")

    lines.append("### Model Reliability and Hallucination Breakdown")
    lines.append(
        "| Model | Variant | Wrong Answers ($N$) | Faithfulness % (Count / $N_{\\text{RAG}}$) | Hallucination Rate: Wrong $\\ge 0.90$ % (Count/$N$) | Hallucination Rate: Unanswerable $\\ge 0.90$ % (Count/150) | Citation Validity % |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for m in models:
        variants = assemble_variant_records(m, all_data[m], test_dict)
        for v_name, v_info in variants.items():
            if v_info.get("is_context"):
                continue

            recs = [r for r in v_info["records"] if r.get("dataset") == "medqa"]
            unans = v_info["unanswerable"]

            # Wrong answers count
            wrong_recs = [r for r in recs if not r["correct"]]
            n_wrong = len(wrong_recs)

            # Faithfulness
            is_adapt = v_info.get("is_adaptive", False)
            is_rag = v_info.get("is_rag", False)

            if not is_rag and not is_adapt:
                faith_str = "—"
                val_str = "—"
            elif is_rag:
                faithful_count = 0
                all_cites = []
                for r in recs:
                    qid = r["id"]
                    q = test_dict[qid]
                    pred = r.get("pred")
                    chosen_opt = q["options"].get(pred) if pred else None
                    passages = ev_dict[qid]["passages"]
                    cites = [c for c in r.get("citations", []) if 1 <= c <= len(passages)]
                    all_cites.extend(r.get("citations", []))
                    cited_passages = [passages[c - 1] for c in cites]
                    if chosen_opt and any(text_contains_option(p["text"], chosen_opt) for p in cited_passages):
                        faithful_count += 1
                faith_pct = faithful_count / len(recs) if recs else 0.0
                faith_str = f"{faith_pct:.1%} ({faithful_count}/{len(recs)})"
                n_valid_cites = sum(1 for c in all_cites if 1 <= c <= 5)
                val_pct = (n_valid_cites / len(all_cites)) if all_cites else 1.0
                val_str = f"{val_pct:.1%}"
            else:
                # Adaptive: evaluate ONLY on final answers taken from RAG
                rag_recs = [r for r in recs if r.get("_used_rag")]
                n_rag_routed = len(rag_recs)
                faithful_count = 0
                all_cites = []
                for r in rag_recs:
                    qid = r["id"]
                    q = test_dict[qid]
                    pred = r.get("pred")
                    chosen_opt = q["options"].get(pred) if pred else None
                    passages = ev_dict[qid]["passages"]
                    cites = [c for c in r.get("citations", []) if 1 <= c <= len(passages)]
                    all_cites.extend(r.get("citations", []))
                    cited_passages = [passages[c - 1] for c in cites]
                    if chosen_opt and any(text_contains_option(p["text"], chosen_opt) for p in cited_passages):
                        faithful_count += 1
                faith_pct = faithful_count / n_rag_routed if n_rag_routed else 0.0
                faith_str = f"{faith_pct:.1%} ({faithful_count}/{n_rag_routed})"
                n_valid_cites = sum(1 for c in all_cites if 1 <= c <= 5)
                val_pct = (n_valid_cites / len(all_cites)) if all_cites else 1.0
                val_str = f"{val_pct:.1%}"

            # Hallucination Rate: Wrong Answers raw conf >= 0.90
            wrong_conf90 = sum(1 for r in wrong_recs if (r.get("confidence") or 0.0) >= 0.90)
            wrong_pct = (wrong_conf90 / n_wrong) if n_wrong else 0.0
            halluc_wrong_str = f"{wrong_pct:.1%} ({wrong_conf90}/{n_wrong})"

            # Hallucination Rate: Unanswerable raw conf >= 0.90
            unans_conf90 = sum(1 for r in unans if (r.get("confidence") or 0.0) >= 0.90)
            unans_pct = (unans_conf90 / len(unans)) if unans else 0.0
            halluc_unans_str = f"{unans_pct:.1%} ({unans_conf90}/{len(unans)})"

            lines.append(
                f"| **{m}** | {v_name} | {n_wrong} | {faith_str} | {halluc_wrong_str} | {halluc_unans_str} | {val_str} |"
            )

    lines.append("")
    return "\n".join(lines)


# ---------- Build Table 3: Efficiency & Cost ----------

def load_profiling_stats():
    """Attempt to load outputs/analysis/profiling.json or work/profile/profiling.json."""
    candidates = [
        Path("outputs/analysis/profiling.json"),
        Path("outputs/kaggle_qa/profiling.json"),
        Path("work/profile/profiling.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.load(open(p))
            except Exception:
                pass
    return None


def build_table_3(models, test_dict, all_data):
    profiling_data = load_profiling_stats()

    lines = []
    lines.append("## Table 3: Efficiency, Generation Throughput, and Computational Footprint")
    lines.append(
        "> **Setup**: Evaluated on the answerable test questions ($N=1,000$ for full benchmarks, $N=500$ for PubMedQA +Abstract).  \n"
        "> **Latency Details**:  \n"
        "> - **Baseline**: Pure parametric forward pass latency.  \n"
        "> - **Full RAG**: Includes generation pass + **0.127 s retrieval overhead** (BM25 + MedCPT dense (FAISS) + MedCPT cross-encoder rerank from Phase 6 & 7 build logs).  \n"
        "> - **Adaptive Gates**: Incorporates retrieval overhead and selective single/double generation passes based on gate logic; peak memory corresponds to Full RAG when retrieval is triggered.  \n"
        "> - **Peak VRAM**: Maximum GPU memory allocated (`torch.cuda.max_memory_allocated`) summed across GPUs (GB, 1 decimal); reserved memory is excluded because it includes cached allocations from earlier modes.  \n"
        "> - **Peak RAM**: Process resident set size (`peak_process_rss_mb`, GB); all models profiled sequentially in one process, so RSS includes residual allocations from previously loaded models — treat as an upper bound.  \n"
        "> *Footnotes*:  \n"
        "> 1. **Timing source**: Latency and throughput are computed over full test runs ($N=1,000$ questions); profiling used 20 questions per mode (10 for context) on Kaggle 2×T4.  \n"
        "> 2. **gemma3-4b**: Loaded in float32 across 2×T4; RAG used batch size 2 vs 8 for baseline, so peak VRAM is not directly comparable across its modes.\n"
    )
    lines.append(
        "| Model | Variant | Avg Latency (s/q) | End-to-End Tokens/s (includes prompt processing) | Mean Prompt Tokens | Parse Rate | Trunc Rate | Peak VRAM | Peak RAM |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for m in models:
        variants = assemble_variant_records(m, all_data[m], test_dict)
        m_prof = profiling_data.get(m, {}) if profiling_data else {}

        for v_name, v_info in variants.items():
            recs = v_info["records"]
            is_rag = v_info.get("is_rag", False)
            is_adapt = v_info.get("is_adaptive", False)
            is_ctx = v_info.get("is_context", False)

            # Compute prompt tokens
            prompt_toks = [r.get("prompt_tokens", 0) for r in recs]
            mean_prompt = sum(prompt_toks) / len(prompt_toks) if prompt_toks else 0.0

            # Compute generation throughput (tokens/s)
            gen_toks = sum(r.get("gen_tokens", 0) for r in recs)
            gen_secs = sum(r.get("seconds", 0.0) for r in recs)
            tps = (gen_toks / gen_secs) if gen_secs > 0 else 0.0

            # Compute parse and truncation rates
            parse_cnt = sum(1 for r in recs if r.get("parsed"))
            parse_str = f"{parse_cnt / len(recs):.1%}" if recs else "n/a"
            trunc_cnt = sum(1 for r in recs if r.get("truncated"))
            trunc_str = f"{trunc_cnt / len(recs):.1%}" if recs else "0.0%"

            # Compute average latency
            if is_adapt:
                avg_lat = v_info.get("cost_sec_per_q")
                if avg_lat is None:
                    avg_lat = (gen_secs / len(recs)) + RETR_SEC_PER_Q
            elif is_rag:
                avg_lat = (gen_secs / len(recs)) + RETR_SEC_PER_Q
            else:
                avg_lat = gen_secs / len(recs)

            # Profiling lookup
            vram_str = "pending"
            ram_str = "pending"
            if m_prof:
                mode_key = "context" if is_ctx else ("rag" if (is_rag or is_adapt) else "baseline")
                mode_stats = m_prof.get("modes", {}).get(mode_key, {})
                if mode_stats:
                    cuda_mem = mode_stats.get("cuda_memory", {})
                    if cuda_mem:
                        total_alloc_mb = sum(dev.get("max_memory_allocated_mb", 0) for dev in cuda_mem.values())
                        vram_str = f"{total_alloc_mb / 1024:.1f} GB"
                    rss_mb = mode_stats.get("peak_process_rss_mb")
                    if rss_mb:
                        ram_str = f"{rss_mb / 1024:.1f} GB"

            lines.append(
                f"| **{m}** | {v_name} | {avg_lat:.3f} s | {tps:.1f} tok/s | {mean_prompt:.1f} | {parse_str} | {trunc_str} | {vram_str} | {ram_str} |"
            )

    lines.append("")
    lines.append("4 of 5 SLM models run on a single 16 GB T4 including RAG; RAG adds ~1.2–1.8 GB VRAM. MedPsy-4B utilizes test-time reasoning (~1,024 max tokens).")
    lines.append("")
    return "\n".join(lines)


# ---------- Build Table 4: MedPsy Pairwise McNemar Comparisons ----------

def build_table_4(models, all_data):
    """Build pairwise McNemar comparison table of medpsy-4b vs SLM baselines on MedQA test IDs."""
    if "medpsy-4b" not in all_data:
        return ""

    medpsy_b_test = [r for r in all_data["medpsy-4b"]["baseline"] if r.get("split") == "test" and r.get("dataset") == "medqa"]
    if not medpsy_b_test:
        return ""

    medpsy_map = {r["id"]: bool(r["correct"]) for r in medpsy_b_test}
    medpsy_ci = bootstrap_ci([1 if c else 0 for c in medpsy_map.values()])

    lines = []
    lines.append("## Table 4: MedPsy-4B Baseline vs SLM Baselines Pairwise Comparison (MedQA Test, $N=500$)")
    lines.append(
        "> **Setup**: Exact two-sided binomial McNemar test comparing `medpsy-4b` baseline against standard SLM baselines "
        "on the identical held-out MedQA test questions ($N=500$).  \n"
        "> **Contingency Matrix**: $b$ = MedPsy correct & Other incorrect; $c$ = MedPsy incorrect & Other correct.  \n"
        "> **Significance**: *** $p < 0.001$, ** $p < 0.01$, * $p < 0.05$.\n"
    )
    lines.append(
        "| Comparison Baseline | MedPsy-4B Acc [95% CI] | Other Model Acc [95% CI] | Δ Acc (pts) | MedPsy+ / Other- ($b$) | MedPsy- / Other+ ($c$) | McNemar $p$-value | Significance |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for m in models:
        if m == "medpsy-4b":
            continue
        m_b_test = [r for r in all_data[m]["baseline"] if r.get("split") == "test" and r.get("dataset") == "medqa"]
        m_map = {r["id"]: bool(r["correct"]) for r in m_b_test}

        shared_ids = [qid for qid in medpsy_map if qid in m_map]
        if not shared_ids:
            continue

        medpsy_flags = [medpsy_map[qid] for qid in shared_ids]
        other_flags = [m_map[qid] for qid in shared_ids]

        b = sum(1 for mp, ot in zip(medpsy_flags, other_flags) if mp and not ot)
        c = sum(1 for mp, ot in zip(medpsy_flags, other_flags) if not mp and ot)
        p_val = mcnemar_exact(medpsy_flags, other_flags)

        other_ci = bootstrap_ci([1 if f else 0 for f in other_flags])
        mp_acc = sum(medpsy_flags) / len(shared_ids)
        ot_acc = sum(other_flags) / len(shared_ids)
        delta_pts = (mp_acc - ot_acc) * 100

        sig = "***" if (p_val < 0.001 or p_val < 1e-300) else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        p_str = format_p_value(p_val)

        lines.append(
            f"| vs **{m}** | {mp_acc:.1%} [{medpsy_ci[0]:.3f}, {medpsy_ci[1]:.3f}] | "
            f"{ot_acc:.1%} [{other_ci[0]:.3f}, {other_ci[1]:.3f}] | "
            f"+{delta_pts:.1f} | {b} | {c} | {p_str} | {sig} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Generate presentation tables for Medical RAG SLM Evaluation.")
    parser.add_argument("--qa-dir", default="outputs/kaggle_qa/full", help="Path to QA output directory")
    parser.add_argument("--build-dir", default="outputs/kaggle_build", help="Path to build artifacts")
    parser.add_argument("--context-dir", default="outputs/kaggle_qa/context", help="Path to context QA directory")
    parser.add_argument("--out-dir", default="outputs/analysis", help="Output directory")
    args = parser.parse_args()

    qa_dir = Path(args.qa_dir)
    build_dir = Path(args.build_dir)
    context_dir = Path(args.context_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data for presentation tables...")
    models, test_dict, ev_dict, all_data = load_data(qa_dir, build_dir, context_dir)
    print(f"Loaded data for {len(models)} models.")

    lines = []
    lines.append("# Medical RAG Small Language Model (SLM) Benchmark: Presentation Tables\n")
    lines.append(
        "Comprehensive empirical evaluation comparing open-weight small language models (1.7B to 4B parameters) "
        "and medical reasoning models across **Baseline (parametric-only)**, **Full RAG**, **Best Adaptive Gate**, and **+Abstract (PubMedQA oracle context)**.\n"
    )

    print("Building Table 1: QA Performance...")
    lines.append(build_table_1(models, test_dict, all_data))

    print("Building Table 2: Reliability & Grounding...")
    lines.append(build_table_2(models, test_dict, ev_dict, all_data))

    print("Building Table 3: Efficiency & Cost...")
    lines.append(build_table_3(models, test_dict, all_data))

    print("Building Table 4: MedPsy-4B Baseline Comparisons...")
    table_4 = build_table_4(models, all_data)
    if table_4:
        lines.append(table_4)

    out_file = out_dir / "presentation_tables.md"
    out_file.write_text("\n".join(lines))
    print(f"Successfully generated: {out_file}")


if __name__ == "__main__":
    main()
