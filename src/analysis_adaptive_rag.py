"""Adaptive RAG analysis over baseline (Phase 5) and RAG (Phase 8) outputs.

Pure Python analysis (no models, no GPU, no scipy).
Tunes gating thresholds on VALIDATION split only, then evaluates on TEST split.
Evaluates:
  - baseline
  - rag
  - rerank_gate (top_rerank_score >= tau)
  - confidence_gate (max letter probability)
  - combined (rerank_gate with confidence check)

Reports accuracy (overall, MedQA, PubMedQA) with 95% bootstrap CIs,
% RAG usage, and McNemar's exact test p-values against baseline.
"""
import argparse
import glob
import json
import math
import os
import random
from pathlib import Path


def mcnemar_exact(preds_base_correct, preds_strat_correct):
    """Exact two-sided McNemar binomial test on discordant pairs.

    b_wins: baseline correct, strategy wrong
    s_wins: strategy correct, baseline wrong
    """
    b_wins = sum(1 for b, s in zip(preds_base_correct, preds_strat_correct) if b and not s)
    s_wins = sum(1 for b, s in zip(preds_base_correct, preds_strat_correct) if not b and s)
    n = b_wins + s_wins
    if n == 0:
        return 1.0, {"base_wins": b_wins, "strat_wins": s_wins, "discordant_total": 0}
    k = min(b_wins, s_wins)
    # CDF of Binomial(n, 0.5) up to k
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_val = min(1.0, 2.0 * cdf)
    return round(p_val, 5), {"base_wins": b_wins, "strat_wins": s_wins, "discordant_total": n}


def bootstrap_ci(correct_list, n_boot=1000, seed=42):
    """Percentile bootstrap 95% CI for accuracy."""
    if not correct_list:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(correct_list)
    means = []
    for _ in range(n_boot):
        sample = rng.choices(correct_list, k=n)
        means.append(sum(sample) / n)
    means.sort()
    low = means[int(0.025 * n_boot)]
    high = means[int(0.975 * n_boot)]
    return (round(low, 4), round(high, 4))


def compute_metrics(pairs, strat_func, base_test_correct, seed=42):
    """Compute accuracy, sub-dataset accuracy, RAG %, and McNemar test for a strategy."""
    total = len(pairs)
    if total == 0:
        return {}

    preds = [strat_func(p) for p in pairs]
    correct_all = [pred == p["gold"] for pred, p in zip(preds, pairs)]
    used_rag_all = [p["_used_rag"] for p in pairs]

    # MedQA
    medqa_pairs = [(c, p) for c, p in zip(correct_all, pairs) if p["dataset"] == "medqa"]
    medqa_correct = [c for c, _ in medqa_pairs]

    # PubMedQA
    pubmed_pairs = [(c, p) for c, p in zip(correct_all, pairs) if p["dataset"] == "pubmedqa"]
    pubmed_correct = [c for c, _ in pubmed_pairs]

    acc_overall = round(sum(correct_all) / total, 4)
    acc_medqa = round(sum(medqa_correct) / len(medqa_correct), 4) if medqa_correct else None
    acc_pubmed = round(sum(pubmed_correct) / len(pubmed_correct), 4) if pubmed_correct else None
    rag_pct = round(100.0 * sum(used_rag_all) / total, 1)

    ci_overall = bootstrap_ci(correct_all, n_boot=1000, seed=seed)
    ci_medqa = bootstrap_ci(medqa_correct, n_boot=1000, seed=seed) if medqa_correct else (None, None)
    ci_pubmed = bootstrap_ci(pubmed_correct, n_boot=1000, seed=seed) if pubmed_correct else (None, None)

    if base_test_correct is not None:
        p_val, discordant = mcnemar_exact(base_test_correct, correct_all)
    else:
        p_val, discordant = None, None

    return {
        "acc_overall": acc_overall,
        "ci_overall": ci_overall,
        "acc_medqa": acc_medqa,
        "ci_medqa": ci_medqa,
        "acc_pubmedqa": acc_pubmed,
        "ci_pubmedqa": ci_pubmed,
        "rag_pct": rag_pct,
        "mcnemar_p_vs_baseline": p_val,
        "mcnemar_discordant": discordant,
        "correct_list": correct_all,
    }


def analyze_model(model_dir):
    model_name = os.path.basename(model_dir)
    p5_path = os.path.join(model_dir, "work", "phase5", f"{model_name}.jsonl")
    p8_path = os.path.join(model_dir, "work", "phase8", f"{model_name}.jsonl")

    if not os.path.exists(p5_path) or not os.path.exists(p8_path):
        print(f"Skipping {model_name}: missing phase5 or phase8 output files.")
        return None

    print(f"\n==================================================")
    print(f"Analyzing Model: {model_name}")
    print(f"==================================================")

    b_rows = {json.loads(l)["id"]: json.loads(l) for l in open(p5_path)}
    r_rows = {json.loads(l)["id"]: json.loads(l) for l in open(p8_path)}

    common_ids = sorted(list(set(b_rows.keys()) & set(r_rows.keys())))
    print(f"Total matching questions: {len(common_ids)}")

    val_pairs = []
    test_pairs = []

    for qid in common_ids:
        b = b_rows[qid]
        r = r_rows[qid]
        if b["should_abstain"]:
            continue  # Answerable rows only

        pair = {
            "id": qid,
            "split": b["split"],
            "dataset": b["dataset"],
            "gold": b["gold"],
            "b_pred": b["pred"],
            "r_pred": r["pred"],
            "b_correct": b["correct"],
            "r_correct": r["correct"],
            "b_conf": b.get("confidence"),
            "r_conf": r.get("confidence"),
            "top_rerank_score": r.get("top_rerank_score"),
            "_used_rag": False,
        }
        if b["split"] == "validation":
            val_pairs.append(pair)
        elif b["split"] == "test":
            test_pairs.append(pair)

    print(f"Answerable rows: {len(val_pairs)} validation, {len(test_pairs)} test")

    # 1. Field presence check
    has_b_conf = any(p["b_conf"] is not None for p in val_pairs + test_pairs)
    has_r_conf = any(p["r_conf"] is not None for p in val_pairs + test_pairs)
    has_rerank = any(p["top_rerank_score"] is not None for p in val_pairs + test_pairs)

    print(f"Field presence check:")
    print(f"  baseline confidence: {'yes' if has_b_conf else 'MISSING'}")
    print(f"  RAG confidence:      {'yes' if has_r_conf else 'MISSING'}")
    print(f"  top_rerank_score:    {'yes' if has_rerank else 'MISSING'}")

    can_rerank = has_rerank
    can_conf = has_b_conf and has_r_conf
    can_combined = can_rerank and can_conf

    if not can_rerank:
        print("  -> top_rerank_score missing: skipping rerank_gate & combined strategies")
    if not can_conf:
        print("  -> confidence missing: skipping confidence_gate & combined strategies")

    tuning_info = {}

    # Define baseline and RAG strat funcs
    def strat_baseline(p):
        p["_used_rag"] = False
        return p["b_pred"]

    def strat_rag(p):
        p["_used_rag"] = True
        return p["r_pred"]

    # 3. Tuning on validation split only
    # a. rerank_gate tuning
    best_tau = None
    if can_rerank:
        val_scores = sorted([p["top_rerank_score"] for p in val_pairs if p["top_rerank_score"] is not None])
        if val_scores:
            # 101 quantiles
            taus = sorted(list({val_scores[int(round(i / 100.0 * (len(val_scores) - 1)))] for i in range(101)}))
            taus.append(val_scores[-1] + 0.01)  # enables 100% baseline choice

            best_tau = None
            best_val_acc = -1.0
            for tau in taus:
                val_acc = sum(1 for p in val_pairs if (p["r_pred"] if p["top_rerank_score"] >= tau else p["b_pred"]) == p["gold"]) / len(val_pairs)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_tau = tau

            tuning_info["rerank_gate"] = {"best_tau": round(best_tau, 4), "val_acc": round(best_val_acc, 4)}
            print(f"Tuned rerank_gate on validation: tau={best_tau:.4f}, val_acc={best_val_acc:.4f}")

    # c. combined tuning
    best_comb_tau, best_comb_delta = None, None
    if can_combined and val_scores:
        deltas = [0.0, 0.05, 0.1, 0.2]
        best_comb_acc = -1.0
        for delta in deltas:
            for tau in taus:
                val_acc = sum(1 for p in val_pairs if (
                    p["r_pred"] if (p["top_rerank_score"] >= tau and not (p["r_conf"] < p["b_conf"] - delta)) else p["b_pred"]
                ) == p["gold"]) / len(val_pairs)
                if val_acc > best_comb_acc:
                    best_comb_acc = val_acc
                    best_comb_tau = tau
                    best_comb_delta = delta

        tuning_info["combined"] = {
            "best_tau": round(best_comb_tau, 4),
            "best_delta": best_comb_delta,
            "val_acc": round(best_comb_acc, 4),
        }
        print(f"Tuned combined on validation: tau={best_comb_tau:.4f}, delta={best_comb_delta}, val_acc={best_comb_acc:.4f}")

    # 4. Evaluation on TEST
    results = {}

    # Baseline on test
    res_base = compute_metrics(test_pairs, strat_baseline, base_test_correct=None, seed=42)
    base_correct_list = res_base.pop("correct_list")
    results["baseline"] = res_base

    # RAG on test
    res_rag = compute_metrics(test_pairs, strat_rag, base_test_correct=base_correct_list, seed=42)
    res_rag.pop("correct_list")
    results["rag"] = res_rag

    # rerank_gate on test
    if can_rerank and best_tau is not None:
        def strat_rerank(p):
            use_rag = p["top_rerank_score"] is not None and p["top_rerank_score"] >= best_tau
            p["_used_rag"] = use_rag
            return p["r_pred"] if use_rag else p["b_pred"]

        res_rerank = compute_metrics(test_pairs, strat_rerank, base_test_correct=base_correct_list, seed=42)
        res_rerank.pop("correct_list")
        results["rerank_gate"] = res_rerank
    else:
        results["rerank_gate"] = {"status": "skipped"}

    # confidence_gate on test
    if can_conf:
        def strat_conf(p):
            # whichever has the higher top letter probability (if tied, keep baseline)
            use_rag = (p["r_conf"] is not None and p["b_conf"] is not None and p["r_conf"] > p["b_conf"])
            p["_used_rag"] = use_rag
            return p["r_pred"] if use_rag else p["b_pred"]

        res_conf = compute_metrics(test_pairs, strat_conf, base_test_correct=base_correct_list, seed=42)
        res_conf.pop("correct_list")
        results["confidence_gate"] = res_conf
    else:
        results["confidence_gate"] = {"status": "skipped"}

    # combined on test
    if can_combined and best_comb_tau is not None:
        def strat_comb(p):
            use_rag = (
                p["top_rerank_score"] is not None
                and p["top_rerank_score"] >= best_comb_tau
                and not (p["r_conf"] < p["b_conf"] - best_comb_delta)
            )
            p["_used_rag"] = use_rag
            return p["r_pred"] if use_rag else p["b_pred"]

        res_comb = compute_metrics(test_pairs, strat_comb, base_test_correct=base_correct_list, seed=42)
        res_comb.pop("correct_list")
        results["combined"] = res_comb
    else:
        results["combined"] = {"status": "skipped"}

    # Save output json
    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"adaptive_rag_{model_name}.json"
    save_data = {
        "model": model_name,
        "n_validation": len(val_pairs),
        "n_test": len(test_pairs),
        "tuning_validation": tuning_info,
        "test_results": results,
    }
    out_json.write_text(json.dumps(save_data, indent=2))
    print(f"Saved: {out_json}")

    return model_name, save_data


def print_combined_table(all_model_data):
    """Print one clean combined markdown table across all analyzed models."""
    header = (
        "| Model | Strategy | % RAG | Test Acc [95% CI] | MedQA | PubMedQA | McNemar p (vs Base) |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = []

    strat_order = ["baseline", "rag", "rerank_gate", "confidence_gate", "combined"]
    strat_labels = {
        "baseline": "Baseline",
        "rag": "Full RAG",
        "rerank_gate": "Rerank Gate",
        "confidence_gate": "Confidence Gate",
        "combined": "Combined Gate",
    }

    for model_name, data in all_model_data:
        test_res = data["test_results"]
        for s in strat_order:
            info = test_res.get(s, {})
            if info.get("status") == "skipped":
                rows.append(f"| {model_name} | {strat_labels[s]} | - | skipped | - | - | - |")
                continue

            acc_str = f"{info['acc_overall']:.4f} [{info['ci_overall'][0]:.3f}, {info['ci_overall'][1]:.3f}]"
            med_str = f"{info['acc_medqa']:.4f}" if info.get("acc_medqa") is not None else "-"
            pub_str = f"{info['acc_pubmedqa']:.4f}" if info.get("acc_pubmedqa") is not None else "-"
            rag_pct = f"{info['rag_pct']:.1f}%"
            p_val = info.get("mcnemar_p_vs_baseline")
            p_str = f"{p_val:.4f}" if p_val is not None else "-"

            rows.append(f"| {model_name} | {strat_labels[s]} | {rag_pct} | {acc_str} | {med_str} | {pub_str} | {p_str} |")

    table = header + "\n" + "\n".join(rows)
    print("\n" + "=" * 80)
    print("COMBINED ADAPTIVE RAG TEST RESULTS")
    print("=" * 80)
    print(table)
    print("=" * 80 + "\n")
    return table


def main():
    parser = argparse.ArgumentParser(description="Adaptive RAG analysis over Phase 5 and Phase 8 outputs.")
    parser.add_argument("--base-dir", default="outputs/kaggle_qa/full", help="Base directory containing model folders")
    parser.add_argument("--models", nargs="*", help="Specific model folder names to analyze (default: all)")
    args = parser.parse_args()

    candidates = sorted([
        d for d in glob.glob(os.path.join(args.base_dir, "*"))
        if os.path.isdir(d)
    ])

    if args.models:
        candidates = [d for d in candidates if os.path.basename(d) in args.models]

    if not candidates:
        print(f"No model output folders found in {args.base_dir}.")
        return

    print(f"Found {len(candidates)} candidate model folder(s): {[os.path.basename(c) for c in candidates]}")

    all_data = []
    for c in candidates:
        res = analyze_model(c)
        if res:
            all_data.append(res)

    if all_data:
        print_combined_table(all_data)


if __name__ == "__main__":
    main()
