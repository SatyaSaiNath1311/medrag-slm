"""Adaptive RAG analysis over baseline (Phase 5) and RAG (Phase 8) outputs.

Pure Python analysis (no models, no GPU, no scipy).
Tunes gating thresholds on VALIDATION split only, then evaluates on TEST split.
Evaluates:
  - baseline
  - rag
  - rerank_gate (top_rerank_score >= tau  -> uses RAG; gate decided from rerank score,
                 NOT requiring a second generation)
  - confidence_gate (pick whichever of baseline/RAG has higher confidence; 2 generations)
  - combined (rerank_gate + confidence check; 2 generations)

Cost model
----------
Strategy         Gens/Q  Retrieval  Seconds/Q
baseline           1       no       b_sec
Full RAG           1       yes      r_sec + retr_sec
rerank_gate        1       yes      (r_sec if RAG used else b_sec) + retr_sec
confidence_gate    2       yes      b_sec + r_sec + retr_sec
combined           2       yes      b_sec + r_sec + retr_sec

retr_sec is the mean per-question retrieval+rerank wall time derived from the
Runner A phase6/phase7 manifests in outputs/kaggle_build/medrag-build.log:
  Phase 6 (retrieval): 552.68 s -> 576.11 s  =>  23.42 s / 1600 q = 0.01464 s/q
  Phase 7 (rerank):    577.57 s -> 757.83 s  => 180.26 s / 1600 q = 0.11266 s/q
  Total  retr_sec = 0.01464 + 0.11266 = 0.12730 s/q

Reports accuracy (overall, MedQA, PubMedQA) with 95% bootstrap CIs,
share of answers taken from RAG, cost (gens/q, retrieval yes/no, sec/q),
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

# ---------------------------------------------------------------------------
# Retrieval+rerank cost derived from Runner A log (phase6 + phase7 timestamps)
# Build log: outputs/kaggle_build/medrag-build.log
#   Phase 6 start:  time=552.682 s  Phase 6 end:  time=576.105 s  => 23.424 s
#   Phase 7 start:  time=577.568 s  Phase 7 end:  time=757.829 s  => 180.261 s
#   Total for 1600 questions: 203.685 s  =>  0.12730 s/q
RETR_SEC_PER_Q = (23.424 + 180.261) / 1600   # = 0.12730 s/q
N_RETRIEVAL_QUESTIONS = 1600

# Cost metadata per strategy
COST_META = {
    # (gens_per_q, retrieval)
    "baseline":         (1, False),
    "rag":              (1, True),
    "rerank_gate":      (1, True),
    "confidence_gate":  (2, True),
    "combined":         (2, True),
}


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
    Returns dict: key -> {"raw_p": ..., "adj_p": ...}.
    """
    m = len(pvals_and_keys)
    sorted_items = sorted(pvals_and_keys, key=lambda x: x[1])
    result = {}
    for rank, (key, raw_p) in enumerate(sorted_items, start=1):
        adj_p = min(1.0, raw_p * (m - rank + 1))
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
    means = [sum(rng.choices(correct_list, k=n)) / n for _ in range(n_boot)]
    means.sort()
    return (round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4))


def compute_metrics(pairs, strat_func, base_test_correct, rag_test_correct, seed=42):
    """Compute accuracy, sub-dataset accuracy, RAG share, and McNemar tests."""
    total = len(pairs)
    if total == 0:
        return {}

    preds = [strat_func(p) for p in pairs]
    correct_all = [pred == p["gold"] for pred, p in zip(preds, pairs)]
    used_rag_all = [p["_used_rag"] for p in pairs]

    medqa_idx    = [i for i, p in enumerate(pairs) if p["dataset"] == "medqa"]
    pubmed_idx   = [i for i, p in enumerate(pairs) if p["dataset"] == "pubmedqa"]
    medqa_correct  = [correct_all[i] for i in medqa_idx]
    pubmed_correct = [correct_all[i] for i in pubmed_idx]

    acc_overall = round(sum(correct_all) / total, 4)
    acc_medqa   = round(sum(medqa_correct)  / len(medqa_correct),  4) if medqa_correct  else None
    acc_pubmed  = round(sum(pubmed_correct) / len(pubmed_correct), 4) if pubmed_correct else None
    rag_pct     = round(100.0 * sum(used_rag_all) / total, 1)

    ci_overall = bootstrap_ci(correct_all,  n_boot=1000, seed=seed)
    ci_medqa   = bootstrap_ci(medqa_correct,  n_boot=1000, seed=seed) if medqa_correct  else (None, None)
    ci_pubmed  = bootstrap_ci(pubmed_correct, n_boot=1000, seed=seed) if pubmed_correct else (None, None)

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
        for ds, ds_idx in [("medqa", medqa_idx), ("pubmedqa", pubmed_idx)]:
            if not ds_idx:
                continue
            base_ds = [base_test_correct[i] for i in ds_idx]
            strat_ds = [correct_all[i] for i in ds_idx]
            p_ds, disc_ds = mcnemar_exact(base_ds, strat_ds)
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


def compute_cost_sec(pairs, strat_name):
    """Mean seconds/question including retrieval where applicable.

    retr_sec = RETR_SEC_PER_Q (derived from build log; see module docstring).

    baseline:         b_sec  (no retrieval)
    Full RAG:         r_sec + retr_sec
    rerank_gate:      retr_sec always + (r_sec if RAG used else b_sec)
                      — gate is decided from stored rerank score; only ONE generation
    confidence_gate:  b_sec + r_sec + retr_sec  (both passes always run)
    combined:         b_sec + r_sec + retr_sec  (both passes always run)
    """
    if not pairs:
        return None
    if strat_name == "baseline":
        secs = [p["b_sec"] for p in pairs]
    elif strat_name == "rag":
        secs = [p["r_sec"] + RETR_SEC_PER_Q for p in pairs]
    elif strat_name == "rerank_gate":
        secs = [
            RETR_SEC_PER_Q + (p["r_sec"] if p["_used_rag"] else p["b_sec"])
            for p in pairs
        ]
    elif strat_name in ("confidence_gate", "combined"):
        secs = [p["b_sec"] + p["r_sec"] + RETR_SEC_PER_Q for p in pairs]
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

    common_ids = sorted(set(b_rows) & set(r_rows))
    print(f"Total matching questions: {len(common_ids)}")

    val_pairs, test_pairs = [], []
    for qid in common_ids:
        b, r = b_rows[qid], r_rows[qid]
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
            "b_sec": b.get("seconds") or 0.0,
            "r_sec": r.get("seconds") or 0.0,
            "_used_rag": False,
        }
        if b["split"] == "validation":
            val_pairs.append(pair)
        elif b["split"] == "test":
            test_pairs.append(pair)

    print(f"Answerable rows: {len(val_pairs)} validation, {len(test_pairs)} test")

    has_b_conf = any(p["b_conf"] is not None for p in val_pairs + test_pairs)
    has_r_conf = any(p["r_conf"] is not None for p in val_pairs + test_pairs)
    has_rerank = any(p["top_rerank_score"] is not None for p in val_pairs + test_pairs)

    print("Field presence check:")
    print(f"  baseline confidence: {'yes' if has_b_conf else 'MISSING'}")
    print(f"  RAG confidence:      {'yes' if has_r_conf else 'MISSING'}")
    print(f"  top_rerank_score:    {'yes' if has_rerank else 'MISSING'}")

    can_rerank   = has_rerank
    can_conf     = has_b_conf and has_r_conf
    can_combined = can_rerank and can_conf

    if not can_rerank:
        print("  -> top_rerank_score missing: skipping rerank_gate & combined")
    if not can_conf:
        print("  -> confidence missing: skipping confidence_gate & combined")

    tuning_info = {}

    # --- Strategy helpers ---
    def strat_baseline(p):
        p["_used_rag"] = False
        return p["b_pred"]

    def strat_rag(p):
        p["_used_rag"] = True
        return p["r_pred"]

    # --- Tune on validation only ---
    best_tau = None
    best_comb_tau, best_comb_delta = None, None
    taus = None

    if can_rerank:
        val_scores = sorted(p["top_rerank_score"] for p in val_pairs if p["top_rerank_score"] is not None)
        if val_scores:
            taus = sorted({
                val_scores[int(round(i / 100.0 * (len(val_scores) - 1)))]
                for i in range(101)
            })
            taus.append(val_scores[-1] + 0.01)

            best_tau, best_val_acc = None, -1.0
            for tau in taus:
                acc = sum(
                    1 for p in val_pairs
                    if (p["r_pred"] if p["top_rerank_score"] >= tau else p["b_pred"]) == p["gold"]
                ) / len(val_pairs)
                if acc > best_val_acc:
                    best_val_acc, best_tau = acc, tau

            tuning_info["rerank_gate"] = {"best_tau": round(best_tau, 4), "val_acc": round(best_val_acc, 4)}
            print(f"Tuned rerank_gate on validation: tau={best_tau:.4f}, val_acc={best_val_acc:.4f}")

    if can_combined and taus:
        best_comb_acc = -1.0
        for delta in [0.0, 0.05, 0.1, 0.2]:
            for tau in taus:
                acc = sum(
                    1 for p in val_pairs
                    if (
                        p["r_pred"] if (
                            p["top_rerank_score"] >= tau
                            and not (p["r_conf"] < p["b_conf"] - delta)
                        ) else p["b_pred"]
                    ) == p["gold"]
                ) / len(val_pairs)
                if acc > best_comb_acc:
                    best_comb_acc, best_comb_tau, best_comb_delta = acc, tau, delta

        tuning_info["combined"] = {
            "best_tau": round(best_comb_tau, 4),
            "best_delta": best_comb_delta,
            "val_acc": round(best_comb_acc, 4),
        }
        print(f"Tuned combined on validation: tau={best_comb_tau:.4f}, delta={best_comb_delta}, val_acc={best_comb_acc:.4f}")

    # --- Evaluate on TEST ---
    results = {}

    res_base = compute_metrics(test_pairs, strat_baseline, None, None, seed=42)
    base_correct_list = res_base.pop("correct_list")
    res_base.pop("used_rag_list")
    gens, retr = COST_META["baseline"]
    res_base.update({"cost_gens_per_q": gens, "cost_retrieval": retr,
                     "cost_sec_per_q": compute_cost_sec(test_pairs, "baseline")})
    results["baseline"] = res_base

    res_rag = compute_metrics(test_pairs, strat_rag, base_correct_list, None, seed=42)
    rag_correct_list = res_rag.pop("correct_list")
    res_rag.pop("used_rag_list")
    gens, retr = COST_META["rag"]
    res_rag.update({"cost_gens_per_q": gens, "cost_retrieval": retr,
                    "cost_sec_per_q": compute_cost_sec(test_pairs, "rag")})
    results["rag"] = res_rag

    # rerank_gate — 1 generation, retrieval always runs, gate from stored score
    if can_rerank and best_tau is not None:
        def strat_rerank(p):
            use = p["top_rerank_score"] is not None and p["top_rerank_score"] >= best_tau
            p["_used_rag"] = use
            return p["r_pred"] if use else p["b_pred"]

        res_rerank = compute_metrics(test_pairs, strat_rerank, base_correct_list, rag_correct_list, seed=42)
        res_rerank.pop("correct_list")
        res_rerank.pop("used_rag_list")
        gens, retr = COST_META["rerank_gate"]
        res_rerank.update({"cost_gens_per_q": gens, "cost_retrieval": retr,
                           "cost_sec_per_q": compute_cost_sec(test_pairs, "rerank_gate")})
        results["rerank_gate"] = res_rerank
    else:
        results["rerank_gate"] = {"status": "skipped"}

    # confidence_gate — 2 generations, retrieval always runs
    if can_conf:
        def strat_conf(p):
            use = (p["r_conf"] is not None and p["b_conf"] is not None
                   and p["r_conf"] > p["b_conf"])
            p["_used_rag"] = use
            return p["r_pred"] if use else p["b_pred"]

        res_conf = compute_metrics(test_pairs, strat_conf, base_correct_list, rag_correct_list, seed=42)
        res_conf.pop("correct_list")
        res_conf.pop("used_rag_list")
        gens, retr = COST_META["confidence_gate"]
        res_conf.update({"cost_gens_per_q": gens, "cost_retrieval": retr,
                         "cost_sec_per_q": compute_cost_sec(test_pairs, "confidence_gate")})
        results["confidence_gate"] = res_conf
    else:
        results["confidence_gate"] = {"status": "skipped"}

    # combined — 2 generations, retrieval always runs
    if can_combined and best_comb_tau is not None:
        def strat_comb(p):
            use = (
                p["top_rerank_score"] is not None
                and p["top_rerank_score"] >= best_comb_tau
                and not (p["r_conf"] < p["b_conf"] - best_comb_delta)
            )
            p["_used_rag"] = use
            return p["r_pred"] if use else p["b_pred"]

        res_comb = compute_metrics(test_pairs, strat_comb, base_correct_list, rag_correct_list, seed=42)
        res_comb.pop("correct_list")
        res_comb.pop("used_rag_list")
        gens, retr = COST_META["combined"]
        res_comb.update({"cost_gens_per_q": gens, "cost_retrieval": retr,
                         "cost_sec_per_q": compute_cost_sec(test_pairs, "combined")})
        results["combined"] = res_comb
    else:
        results["combined"] = {"status": "skipped"}

    # --- Holm-Bonferroni correction on strategy-vs-baseline p-values ---
    strat_keys = ["rag", "rerank_gate", "confidence_gate", "combined"]
    raw_p_pairs = [
        (s, results[s]["mcnemar_p_vs_baseline"])
        for s in strat_keys
        if isinstance(results.get(s), dict) and results[s].get("mcnemar_p_vs_baseline") is not None
    ]
    if raw_p_pairs:
        hb = holm_bonferroni(raw_p_pairs)
        for key, hb_vals in hb.items():
            if isinstance(results.get(key), dict):
                results[key]["holm_adj_p_vs_baseline"] = hb_vals["adj_p"]

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
        "retr_sec_per_q": round(RETR_SEC_PER_Q, 5),
        "retr_sec_derivation": (
            f"Phase6 ({23.424:.3f}s) + Phase7 ({180.261:.3f}s) "
            f"= {23.424+180.261:.3f}s / {N_RETRIEVAL_QUESTIONS} q "
            f"= {RETR_SEC_PER_Q:.5f} s/q  (from outputs/kaggle_build/medrag-build.log)"
        ),
        "tuning_validation": tuning_info,
        "test_results": results,
    }
    out_json.write_text(json.dumps(save_data, indent=2))
    print(f"Saved: {out_json}")

    return model_name, save_data


def print_combined_table(all_model_data):
    """Print combined markdown table across all analyzed models."""
    strat_order = ["baseline", "rag", "rerank_gate", "confidence_gate", "combined"]
    strat_labels = {
        "baseline":        "Baseline",
        "rag":             "Full RAG",
        "rerank_gate":     "Rerank Gate",
        "confidence_gate": "Confidence Gate",
        "combined":        "Combined Gate",
    }

    # Tuning summary
    print("\n" + "=" * 95)
    print("VALIDATION TUNING SUMMARY  (thresholds used on test set unchanged)")
    print("=" * 95)
    print(f"{'Model':<16} {'Strategy':<18} {'tau':>10} {'delta':>7} {'Val Acc':>9}")
    print("-" * 65)
    for model_name, data in all_model_data:
        for strat, info in data.get("tuning_validation", {}).items():
            tau_s   = f"{info['best_tau']:.4f}" if "best_tau" in info else "-"
            delta_s = str(info.get("best_delta", "-"))
            val_s   = f"{info['val_acc']:.4f}" if "val_acc" in info else "-"
            print(f"{model_name:<16} {strat:<18} {tau_s:>10} {delta_s:>7} {val_s:>9}")
    print("=" * 95)

    # Cost note
    print(f"\nRetrieval cost: retr_sec = {RETR_SEC_PER_Q:.5f} s/q  "
          f"(Phase6 23.424 s + Phase7 180.261 s = 203.685 s / 1600 q, "
          f"from outputs/kaggle_build/medrag-build.log)")

    # Main table
    header = (
        "| Model | Strategy | Gens/Q | Retrieval | Sec/Q | % RAG | "
        "Test Acc [95% CI] | MedQA | PubMedQA | "
        "p vs Base (raw/adj-HB) | p vs RAG |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []

    for model_name, data in all_model_data:
        test_res = data["test_results"]
        for s in strat_order:
            info = test_res.get(s, {})
            if info.get("status") == "skipped":
                rows.append(f"| {model_name} | {strat_labels[s]} | - | - | - | - | skipped | - | - | - | - |")
                continue

            gens  = str(info.get("cost_gens_per_q", "-"))
            retr  = "yes" if info.get("cost_retrieval") else "no"
            secs  = f"{info['cost_sec_per_q']:.3f}" if info.get("cost_sec_per_q") is not None else "-"
            rag   = f"{info['rag_pct']:.1f}%"
            acc   = (f"{info['acc_overall']:.4f} "
                     f"[{info['ci_overall'][0]:.3f}, {info['ci_overall'][1]:.3f}]")
            med   = f"{info['acc_medqa']:.4f}"   if info.get("acc_medqa")   is not None else "-"
            pub   = f"{info['acc_pubmedqa']:.4f}" if info.get("acc_pubmedqa") is not None else "-"

            raw_p = info.get("mcnemar_p_vs_baseline")
            adj_p = info.get("holm_adj_p_vs_baseline")
            if raw_p is None:
                p_base = "-"
            elif adj_p is not None:
                p_base = f"{raw_p:.4f} / {adj_p:.4f}"
            else:
                p_base = f"{raw_p:.4f}"

            p_rag = info.get("mcnemar_p_vs_rag")
            p_rag_s = f"{p_rag:.4f}" if p_rag is not None else "-"

            rows.append(
                f"| {model_name} | {strat_labels[s]} | {gens} | {retr} | {secs} | {rag} "
                f"| {acc} | {med} | {pub} | {p_base} | {p_rag_s} |"
            )

    table = header + "\n" + "\n".join(rows)
    print("\n" + "=" * 120)
    print("COMBINED ADAPTIVE RAG TEST RESULTS")
    print("(% RAG = share of final answers taken from RAG)")
    print("=" * 120)
    print(table)
    print("=" * 120)

    # Per-dataset McNemar
    print("\n" + "=" * 85)
    print("PER-DATASET McNEMAR vs Baseline  (Full RAG + gated strategies)")
    print("=" * 85)
    print(f"{'Model':<16} {'Strategy':<18} {'Dataset':<12} {'Acc':>7} "
          f"{'p_raw':>9} {'a_wins':>7} {'b_wins':>7}")
    print("-" * 80)
    for model_name, data in all_model_data:
        for s in ["rag", "rerank_gate", "confidence_gate", "combined"]:
            info = data["test_results"].get(s, {})
            if not isinstance(info, dict) or info.get("status") == "skipped":
                continue
            pd = info.get("per_dataset_mcnemar_vs_baseline", {})
            for ds in ["medqa", "pubmedqa"]:
                ds_acc = info.get("acc_medqa") if ds == "medqa" else info.get("acc_pubmedqa")
                ds_info = pd.get(ds, {})
                p_s  = f"{ds_info['p_vs_baseline']:.5f}" if "p_vs_baseline" in ds_info else "-"
                disc = ds_info.get("discordant", {})
                aw   = str(disc.get("a_wins", "-"))
                bw   = str(disc.get("b_wins", "-"))
                acc_s = f"{ds_acc:.4f}" if ds_acc is not None else "-"
                print(f"{model_name:<16} {strat_labels[s]:<18} {ds:<12} {acc_s:>7} {p_s:>9} {aw:>7} {bw:>7}")
    print("=" * 85)

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive RAG analysis over Phase 5 and Phase 8 outputs."
    )
    parser.add_argument("--base-dir", default="outputs/kaggle_qa/full")
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()

    candidates = sorted(d for d in glob.glob(os.path.join(args.base_dir, "*")) if os.path.isdir(d))
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
