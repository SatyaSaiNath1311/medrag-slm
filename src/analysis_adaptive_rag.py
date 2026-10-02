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
share of answers taken from RAG, cost (generations/question + mean seconds/question),
McNemar exact tests vs baseline and vs Full RAG,
and Holm-Bonferroni corrected p-values for strategy-vs-baseline within each model.
"""
import argparse
import glob
import json
import math
import os
import random
from pathlib import Path


# ---------- Statistics ----------

def mcnemar_exact(preds_a_correct, preds_b_correct):
    """Exact two-sided McNemar binomial test.
    Returns p-value and dict of discordant counts.
    a_wins: A correct, B wrong; b_wins: B correct, A wrong.
    """
    a_wins = sum(1 for a, b in zip(preds_a_correct, preds_b_correct) if a and not b)
    b_wins = sum(1 for a, b in zip(preds_a_correct, preds_b_correct) if not a and b)
    n = a_wins + b_wins
    if n == 0:
        return 1.0, {"a_wins": a_wins, "b_wins": b_wins, "discordant_total": 0}
    k = min(a_wins, b_wins)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_val = min(1.0, 2.0 * cdf)
    return round(p_val, 5), {"a_wins": a_wins, "b_wins": b_wins, "discordant_total": n}


def holm_bonferroni(pvals_and_keys):
    """Holm-Bonferroni correction over a list of (key, raw_p) pairs.
    Returns dict: key -> {"raw_p": ..., "adj_p": ..., "significant": bool}.
    """
    m = len(pvals_and_keys)
    sorted_items = sorted(pvals_and_keys, key=lambda x: x[1])
    result = {}
    for rank, (key, raw_p) in enumerate(sorted_items, start=1):
        adj_p = min(1.0, raw_p * (m - rank + 1))
        # Monotonicity: adjusted p must be >= previous adjusted p
        if rank > 1:
            prev_key = sorted_items[rank - 2][0]
            adj_p = max(adj_p, result[prev_key]["adj_p"])
        result[key] = {"raw_p": round(raw_p, 5), "adj_p": round(adj_p, 5)}
    return result


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


def safe_mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else None


# ---------- Cost model ----------
# generations per question (retrieval cost is assumed equal for RAG-using strategies)
COST_GENS = {
    "baseline": 1,
    "rag": 2,             # 1 retrieval + 1 generation
    "rerank_gate": 2,     # 1 retrieval always + 1 generation
    "confidence_gate": 3, # 2 generations + 1 retrieval
    "combined": 3,        # 2 generations + 1 retrieval
}


def compute_metrics(pairs, strat_func, base_test_correct, rag_test_correct, seed=42):
    """Compute accuracy, sub-dataset accuracy, RAG share, cost, and McNemar tests."""
    total = len(pairs)
    if total == 0:
        return {}

    preds = [strat_func(p) for p in pairs]
    correct_all = [pred == p["gold"] for pred, p in zip(preds, pairs)]
    used_rag_all = [p["_used_rag"] for p in pairs]

    # Sub-datasets
    medqa_pairs = [(c, p) for c, p in zip(correct_all, pairs) if p["dataset"] == "medqa"]
    medqa_correct = [c for c, _ in medqa_pairs]
    pubmed_pairs = [(c, p) for c, p in zip(correct_all, pairs) if p["dataset"] == "pubmedqa"]
    pubmed_correct = [c for c, _ in pubmed_pairs]

    acc_overall = round(sum(correct_all) / total, 4)
    acc_medqa = round(sum(medqa_correct) / len(medqa_correct), 4) if medqa_correct else None
    acc_pubmed = round(sum(pubmed_correct) / len(pubmed_correct), 4) if pubmed_correct else None
    rag_pct = round(100.0 * sum(used_rag_all) / total, 1)

    ci_overall = bootstrap_ci(correct_all, n_boot=1000, seed=seed)
    ci_medqa = bootstrap_ci(medqa_correct, n_boot=1000, seed=seed) if medqa_correct else (None, None)
    ci_pubmed = bootstrap_ci(pubmed_correct, n_boot=1000, seed=seed) if pubmed_correct else (None, None)

    # McNemar vs baseline
    if base_test_correct is not None:
        p_vs_base, disc_vs_base = mcnemar_exact(base_test_correct, correct_all)
    else:
        p_vs_base, disc_vs_base = None, None

    # McNemar vs Full RAG
    if rag_test_correct is not None:
        p_vs_rag, disc_vs_rag = mcnemar_exact(rag_test_correct, correct_all)
    else:
        p_vs_rag, disc_vs_rag = None, None

    # Per-dataset McNemar vs baseline
    pd_mcnemar = {}
    if base_test_correct is not None:
        for ds, ds_pairs in [("medqa", medqa_pairs), ("pubmedqa", pubmed_pairs)]:
            if not ds_pairs:
                continue
            ds_correct = [c for c, _ in ds_pairs]
            ds_idx = [i for i, p in enumerate(pairs) if p["dataset"] == ds]
            base_ds = [base_test_correct[i] for i in ds_idx]
            p_ds, disc_ds = mcnemar_exact(base_ds, ds_correct)
            pd_mcnemar[ds] = {"p_vs_baseline": p_ds, "discordant": disc_ds}

    return {
        "acc_overall": acc_overall,
        "ci_overall": ci_overall,
        "acc_medqa": acc_medqa,
        "ci_medqa": ci_medqa,
        "acc_pubmedqa": acc_pubmed,
        "ci_pubmedqa": ci_pubmed,
        "rag_pct": rag_pct,
        "mcnemar_p_vs_baseline": p_vs_base,
        "mcnemar_discordant_vs_baseline": disc_vs_base,
        "mcnemar_p_vs_rag": p_vs_rag,
        "mcnemar_discordant_vs_rag": disc_vs_rag,
        "per_dataset_mcnemar_vs_baseline": pd_mcnemar,
        "correct_list": correct_all,
        "used_rag_list": used_rag_all,
    }


def compute_cost(pairs, strat_name):
    """Mean seconds/question for a strategy.

    Cost model:
      baseline:         mean b_seconds
      rag:              mean r_seconds
      rerank_gate:      b_seconds (always run) + r_seconds when RAG used
      confidence_gate:  b_seconds + r_seconds (both always run)
      combined:         b_seconds + r_seconds (both always run, we gate on scores)
    """
    if not pairs:
        return None
    if strat_name == "baseline":
        secs = [p["b_sec"] for p in pairs]
    elif strat_name == "rag":
        secs = [p["r_sec"] for p in pairs]
    elif strat_name == "rerank_gate":
        # RAG generation runs only when used; retrieval+rerank always runs (part of r_sec)
        secs = [p["b_sec"] + (p["r_sec"] if p["_used_rag"] else 0.0) for p in pairs]
    elif strat_name in ("confidence_gate", "combined"):
        # Both passes always run
        secs = [p["b_sec"] + p["r_sec"] for p in pairs]
    else:
        return None
    return round(sum(secs) / len(secs), 4)


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

    val_pairs, test_pairs = [], []

    for qid in common_ids:
        b = b_rows[qid]
        r = r_rows[qid]
        if b["should_abstain"]:
            continue
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
            "b_sec": b.get("seconds", 0.0) or 0.0,
            "r_sec": r.get("seconds", 0.0) or 0.0,
            "_used_rag": False,
        }
        if b["split"] == "validation":
            val_pairs.append(pair)
        elif b["split"] == "test":
            test_pairs.append(pair)

    print(f"Answerable rows: {len(val_pairs)} validation, {len(test_pairs)} test")

    # Field presence check
    has_b_conf = any(p["b_conf"] is not None for p in val_pairs + test_pairs)
    has_r_conf = any(p["r_conf"] is not None for p in val_pairs + test_pairs)
    has_rerank = any(p["top_rerank_score"] is not None for p in val_pairs + test_pairs)

    print("Field presence check:")
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

    # --- Strat functions ---
    def strat_baseline(p):
        p["_used_rag"] = False
        return p["b_pred"]

    def strat_rag(p):
        p["_used_rag"] = True
        return p["r_pred"]

    # --- Tune on validation only ---
    best_tau = None
    best_comb_tau, best_comb_delta = None, None
    val_scores = None
    taus = None

    if can_rerank:
        val_scores = sorted([p["top_rerank_score"] for p in val_pairs if p["top_rerank_score"] is not None])
        if val_scores:
            taus = sorted(list({val_scores[int(round(i / 100.0 * (len(val_scores) - 1)))] for i in range(101)}))
            taus.append(val_scores[-1] + 0.01)

            best_tau = None
            best_val_acc = -1.0
            for tau in taus:
                val_acc = sum(
                    1 for p in val_pairs
                    if (p["r_pred"] if p["top_rerank_score"] >= tau else p["b_pred"]) == p["gold"]
                ) / len(val_pairs)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_tau = tau

            tuning_info["rerank_gate"] = {"best_tau": round(best_tau, 4), "val_acc": round(best_val_acc, 4)}
            print(f"Tuned rerank_gate on validation: tau={best_tau:.4f}, val_acc={best_val_acc:.4f}")

    if can_combined and taus:
        deltas = [0.0, 0.05, 0.1, 0.2]
        best_comb_acc = -1.0
        for delta in deltas:
            for tau in taus:
                val_acc = sum(
                    1 for p in val_pairs
                    if (
                        p["r_pred"] if (p["top_rerank_score"] >= tau and not (p["r_conf"] < p["b_conf"] - delta))
                        else p["b_pred"]
                    ) == p["gold"]
                ) / len(val_pairs)
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

    # --- Evaluate on TEST ---
    results = {}

    res_base = compute_metrics(test_pairs, strat_baseline, base_test_correct=None, rag_test_correct=None, seed=42)
    base_correct_list = res_base.pop("correct_list")
    res_base.pop("used_rag_list")
    res_base["cost_gens_per_q"] = COST_GENS["baseline"]
    res_base["cost_sec_per_q"] = compute_cost(test_pairs, "baseline")
    results["baseline"] = res_base

    res_rag = compute_metrics(test_pairs, strat_rag, base_test_correct=base_correct_list, rag_test_correct=None, seed=42)
    rag_correct_list = res_rag.pop("correct_list")
    res_rag.pop("used_rag_list")
    res_rag["cost_gens_per_q"] = COST_GENS["rag"]
    res_rag["cost_sec_per_q"] = compute_cost(test_pairs, "rag")
    results["rag"] = res_rag

    # rerank_gate
    if can_rerank and best_tau is not None:
        def strat_rerank(p):
            use_rag = p["top_rerank_score"] is not None and p["top_rerank_score"] >= best_tau
            p["_used_rag"] = use_rag
            return p["r_pred"] if use_rag else p["b_pred"]

        res_rerank = compute_metrics(test_pairs, strat_rerank, base_test_correct=base_correct_list,
                                     rag_test_correct=rag_correct_list, seed=42)
        res_rerank.pop("correct_list")
        res_rerank.pop("used_rag_list")
        res_rerank["cost_gens_per_q"] = COST_GENS["rerank_gate"]
        res_rerank["cost_sec_per_q"] = compute_cost(test_pairs, "rerank_gate")
        results["rerank_gate"] = res_rerank
    else:
        results["rerank_gate"] = {"status": "skipped"}

    # confidence_gate
    if can_conf:
        def strat_conf(p):
            use_rag = (p["r_conf"] is not None and p["b_conf"] is not None and p["r_conf"] > p["b_conf"])
            p["_used_rag"] = use_rag
            return p["r_pred"] if use_rag else p["b_pred"]

        res_conf = compute_metrics(test_pairs, strat_conf, base_test_correct=base_correct_list,
                                   rag_test_correct=rag_correct_list, seed=42)
        res_conf.pop("correct_list")
        res_conf.pop("used_rag_list")
        res_conf["cost_gens_per_q"] = COST_GENS["confidence_gate"]
        res_conf["cost_sec_per_q"] = compute_cost(test_pairs, "confidence_gate")
        results["confidence_gate"] = res_conf
    else:
        results["confidence_gate"] = {"status": "skipped"}

    # combined
    if can_combined and best_comb_tau is not None:
        def strat_comb(p):
            use_rag = (
                p["top_rerank_score"] is not None
                and p["top_rerank_score"] >= best_comb_tau
                and not (p["r_conf"] < p["b_conf"] - best_comb_delta)
            )
            p["_used_rag"] = use_rag
            return p["r_pred"] if use_rag else p["b_pred"]

        res_comb = compute_metrics(test_pairs, strat_comb, base_test_correct=base_correct_list,
                                   rag_test_correct=rag_correct_list, seed=42)
        res_comb.pop("correct_list")
        res_comb.pop("used_rag_list")
        res_comb["cost_gens_per_q"] = COST_GENS["combined"]
        res_comb["cost_sec_per_q"] = compute_cost(test_pairs, "combined")
        results["combined"] = res_comb
    else:
        results["combined"] = {"status": "skipped"}

    # --- Holm-Bonferroni correction on strategy-vs-baseline p-values ---
    strat_keys = ["rag", "rerank_gate", "confidence_gate", "combined"]
    raw_p_pairs = [
        (s, results[s].get("mcnemar_p_vs_baseline"))
        for s in strat_keys
        if isinstance(results.get(s), dict) and results[s].get("mcnemar_p_vs_baseline") is not None
    ]
    if raw_p_pairs:
        hb = holm_bonferroni(raw_p_pairs)
        for key, hb_vals in hb.items():
            if key in results and isinstance(results[key], dict):
                results[key]["holm_adj_p_vs_baseline"] = hb_vals["adj_p"]
    else:
        hb = {}

    print("\nHolm-Bonferroni corrected p vs baseline:")
    for s in strat_keys:
        if isinstance(results.get(s), dict):
            raw = results[s].get("mcnemar_p_vs_baseline")
            adj = results[s].get("holm_adj_p_vs_baseline")
            if raw is not None:
                print(f"  {s}: raw={raw:.5f}  adj={adj:.5f}")

    # --- Save ---
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
    """Print one combined markdown table across all analyzed models."""
    strat_order = ["baseline", "rag", "rerank_gate", "confidence_gate", "combined"]
    strat_labels = {
        "baseline": "Baseline",
        "rag": "Full RAG",
        "rerank_gate": "Rerank Gate",
        "confidence_gate": "Confidence Gate",
        "combined": "Combined Gate",
    }

    # --- Tuning table ---
    print("\n" + "=" * 90)
    print("VALIDATION TUNING SUMMARY")
    print("=" * 90)
    print(f"{'Model':<16} {'Strategy':<18} {'tau':>8} {'delta':>6} {'Val Acc':>9}")
    print("-" * 60)
    for model_name, data in all_model_data:
        tuning = data.get("tuning_validation", {})
        for strat, info in tuning.items():
            tau_str = f"{info['best_tau']:.4f}" if "best_tau" in info else "-"
            delta_str = f"{info.get('best_delta', '-')}"
            val_acc_str = f"{info['val_acc']:.4f}" if "val_acc" in info else "-"
            print(f"{model_name:<16} {strat:<18} {tau_str:>8} {delta_str:>6} {val_acc_str:>9}")
    print("=" * 90)

    # --- Main results table ---
    header = (
        "| Model | Strategy | % RAG | Gens/Q | Sec/Q | "
        "Test Acc [95% CI] | MedQA | PubMedQA | "
        "p vs Base (raw/adj) | p vs RAG |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []

    for model_name, data in all_model_data:
        test_res = data["test_results"]
        for s in strat_order:
            info = test_res.get(s, {})
            if info.get("status") == "skipped":
                rows.append(f"| {model_name} | {strat_labels[s]} | - | - | - | skipped | - | - | - | - |")
                continue

            acc_str = (
                f"{info['acc_overall']:.4f} "
                f"[{info['ci_overall'][0]:.3f}, {info['ci_overall'][1]:.3f}]"
            )
            med_str = f"{info['acc_medqa']:.4f}" if info.get("acc_medqa") is not None else "-"
            pub_str = f"{info['acc_pubmedqa']:.4f}" if info.get("acc_pubmedqa") is not None else "-"
            rag_pct = f"{info['rag_pct']:.1f}%"
            gens = str(info.get("cost_gens_per_q", "-"))
            secs = f"{info['cost_sec_per_q']:.3f}" if info.get("cost_sec_per_q") is not None else "-"

            raw_p = info.get("mcnemar_p_vs_baseline")
            adj_p = info.get("holm_adj_p_vs_baseline")
            if raw_p is None:
                p_base_str = "-"
            elif adj_p is not None:
                p_base_str = f"{raw_p:.4f}/{adj_p:.4f}"
            else:
                p_base_str = f"{raw_p:.4f}"

            p_rag = info.get("mcnemar_p_vs_rag")
            p_rag_str = f"{p_rag:.4f}" if p_rag is not None else "-"

            rows.append(
                f"| {model_name} | {strat_labels[s]} | {rag_pct} | {gens} | {secs} "
                f"| {acc_str} | {med_str} | {pub_str} | {p_base_str} | {p_rag_str} |"
            )

    table = header + "\n" + "\n".join(rows)
    print("\n" + "=" * 110)
    print("COMBINED ADAPTIVE RAG TEST RESULTS")
    print("(% RAG = share of final answers taken from RAG)")
    print("=" * 110)
    print(table)
    print("=" * 110)

    # --- Per-dataset McNemar table for Full RAG and best strategy per model ---
    print("\n" + "=" * 80)
    print("PER-DATASET McNEMAR (vs Baseline)  — Full RAG and each gated strategy")
    print("=" * 80)
    print(f"{'Model':<16} {'Strategy':<18} {'Dataset':<12} {'Acc':>7} {'p_raw':>9} {'discordant':>12}")
    print("-" * 75)
    for model_name, data in all_model_data:
        test_res = data["test_results"]
        for s in ["rag", "rerank_gate", "confidence_gate", "combined"]:
            info = test_res.get(s, {})
            if info.get("status") == "skipped" or not isinstance(info, dict):
                continue
            pd = info.get("per_dataset_mcnemar_vs_baseline", {})
            for ds in ["medqa", "pubmedqa"]:
                ds_acc = info.get(f"acc_{ds}") if ds == "medqa" else info.get("acc_pubmedqa")
                ds_info = pd.get(ds, {})
                p_str = f"{ds_info['p_vs_baseline']:.5f}" if "p_vs_baseline" in ds_info else "-"
                disc = ds_info.get("discordant", {})
                disc_str = (
                    f"a={disc.get('a_wins','?')},b={disc.get('b_wins','?')}"
                    if disc else "-"
                )
                acc_str = f"{ds_acc:.4f}" if ds_acc is not None else "-"
                print(f"{model_name:<16} {strat_labels[s]:<18} {ds:<12} {acc_str:>7} {p_str:>9} {disc_str:>12}")
    print("=" * 80)

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive RAG analysis over Phase 5 and Phase 8 outputs."
    )
    parser.add_argument("--base-dir", default="outputs/kaggle_qa/full",
                        help="Base directory containing model folders")
    parser.add_argument("--models", nargs="*",
                        help="Specific model folder names to analyze (default: all)")
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
