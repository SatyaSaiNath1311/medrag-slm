"""Final 5-model results: baseline vs Full RAG accuracy comparison.

Reads Phase 5 (baseline) and Phase 8 (RAG) jsonl outputs for every model found
under --base-dir, computes accuracy (overall, MedQA, PubMedQA) with 95%
bootstrap CIs, McNemar p-values, parse rates, fallback counts, and mean
seconds per question. Writes outputs/analysis/final_results.md and .csv.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
from pathlib import Path


# ---------- statistics ----------

def bootstrap_ci(correct_list, n_boot=1000, seed=42):
    if not correct_list:
        return (None, None)
    rng = random.Random(seed)
    n = len(correct_list)
    means = sorted(sum(rng.choices(correct_list, k=n)) / n for _ in range(n_boot))
    return (round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4))


def mcnemar_exact(a_correct, b_correct):
    a_wins = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    b_wins = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2.0 * cdf), 5)


def acc(rows):
    return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None


def parse_rate(rows):
    return round(sum(bool(r.get("parsed")) for r in rows) / len(rows), 4) if rows else None


def fallback_count(rows):
    return sum(1 for r in rows if r.get("pred_source") == "logprob_fallback")


def mean_sec(rows):
    return round(sum(r["seconds"] for r in rows) / len(rows), 4) if rows else None


# ---------- per-model stats ----------

def model_stats(model_name, p5_path, p8_path):
    b_rows = [json.loads(l) for l in open(p5_path)]
    r_rows = [json.loads(l) for l in open(p8_path)]

    # Align by id
    b_by_id = {r["id"]: r for r in b_rows}
    r_by_id = {r["id"]: r for r in r_rows}
    common = sorted(set(b_by_id) & set(r_by_id))

    result = {"model": model_name, "n_common": len(common)}

    for mode, rows_by_id in [("baseline", b_by_id), ("rag", r_by_id)]:
        rows = list(rows_by_id.values())
        test_ans = [r for r in rows if r["split"] == "test" and not r["should_abstain"]]
        test_medqa = [r for r in test_ans if r["dataset"] == "medqa"]
        test_pubmed = [r for r in test_ans if r["dataset"] == "pubmedqa"]

        corr_all   = [r["correct"] for r in test_ans]
        corr_med   = [r["correct"] for r in test_medqa]
        corr_pub   = [r["correct"] for r in test_pubmed]

        result[mode] = {
            "n_test_answerable": len(test_ans),
            "acc_overall":  acc(test_ans),
            "ci_overall":   bootstrap_ci(corr_all),
            "acc_medqa":    acc(test_medqa),
            "ci_medqa":     bootstrap_ci(corr_med),
            "acc_pubmedqa": acc(test_pubmed),
            "ci_pubmedqa":  bootstrap_ci(corr_pub),
            "parse_rate":   parse_rate(rows),
            "fallback_count": fallback_count(rows),
            "mean_sec_per_q": mean_sec(rows),
            "correct_list": corr_all,
            "correct_medqa": corr_med,
            "correct_pubmedqa": corr_pub,
        }

    # McNemar overall
    b_ans_by_id = {r["id"]: r for r in b_by_id.values()
                   if r["split"] == "test" and not r["should_abstain"]}
    r_ans_by_id = {r["id"]: r for r in r_by_id.values()
                   if r["split"] == "test" and not r["should_abstain"]}
    shared_ans = sorted(set(b_ans_by_id) & set(r_ans_by_id))

    b_corr = [b_ans_by_id[i]["correct"] for i in shared_ans]
    r_corr = [r_ans_by_id[i]["correct"] for i in shared_ans]
    result["mcnemar_p_overall"] = mcnemar_exact(b_corr, r_corr)

    # McNemar per dataset
    for ds in ("medqa", "pubmedqa"):
        shared_ds = [i for i in shared_ans
                     if b_ans_by_id[i]["dataset"] == ds]
        if shared_ds:
            b_ds = [b_ans_by_id[i]["correct"] for i in shared_ds]
            r_ds = [r_ans_by_id[i]["correct"] for i in shared_ds]
            result[f"mcnemar_p_{ds}"] = mcnemar_exact(b_ds, r_ds)
        else:
            result[f"mcnemar_p_{ds}"] = None

    return result


# ---------- output ----------

def fmt_acc_ci(acc_val, ci):
    if acc_val is None:
        return "-"
    lo, hi = ci if ci else (None, None)
    if lo is None:
        return f"{acc_val:.4f}"
    return f"{acc_val:.4f} [{lo:.3f}, {hi:.3f}]"


def write_markdown(stats_list, out_path):
    lines = []
    lines.append("# Final Results: Baseline vs Full RAG (5 Models)\n")
    lines.append(
        "> Accuracy on the **test set, answerable questions only** (should_abstain==False).  \n"
        "> 95% CI: percentile bootstrap (1 000 resamples, seed 42).  \n"
        "> McNemar: exact two-sided binomial test on discordant pairs.\n"
    )
    lines.append("")

    # Main accuracy table
    lines.append("## Accuracy")
    lines.append(
        "| Model | Baseline Overall [95% CI] | Baseline MedQA | Baseline PubMedQA "
        "| RAG Overall [95% CI] | RAG MedQA | RAG PubMedQA "
        "| McNemar p (overall) | McNemar p (MedQA) | McNemar p (PubMedQA) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in stats_list:
        b, r = s["baseline"], s["rag"]
        lines.append(
            f"| {s['model']} "
            f"| {fmt_acc_ci(b['acc_overall'], b['ci_overall'])} "
            f"| {fmt_acc_ci(b['acc_medqa'], b['ci_medqa'])} "
            f"| {fmt_acc_ci(b['acc_pubmedqa'], b['ci_pubmedqa'])} "
            f"| {fmt_acc_ci(r['acc_overall'], r['ci_overall'])} "
            f"| {fmt_acc_ci(r['acc_medqa'], r['ci_medqa'])} "
            f"| {fmt_acc_ci(r['acc_pubmedqa'], r['ci_pubmedqa'])} "
            f"| {s['mcnemar_p_overall']:.5f} "
            f"| {s.get('mcnemar_p_medqa', '-')} "
            f"| {s.get('mcnemar_p_pubmedqa', '-')} |"
        )

    lines.append("")
    lines.append("## Parse Rate, Fallback Count, and Speed")
    lines.append(
        "| Model | Baseline Parse Rate | Baseline Fallback | Baseline Sec/Q "
        "| RAG Parse Rate | RAG Fallback | RAG Sec/Q |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for s in stats_list:
        b, r = s["baseline"], s["rag"]
        lines.append(
            f"| {s['model']} "
            f"| {b['parse_rate']:.4f} "
            f"| {b['fallback_count']} "
            f"| {b['mean_sec_per_q']:.3f} s "
            f"| {r['parse_rate']:.4f} "
            f"| {r['fallback_count']} "
            f"| {r['mean_sec_per_q']:.3f} s |"
        )

    lines.append("")
    lines.append("---")
    lines.append(
        f"*Retrieval+rerank cost (not in sec/q above): 0.12730 s/q "
        f"(Phase6 23.424 s + Phase7 180.261 s = 203.685 s / 1600 q, "
        f"from outputs/kaggle_build/medrag-build.log)*"
    )

    out_path.write_text("\n".join(lines))
    print(f"Saved markdown: {out_path}")


def write_csv(stats_list, out_path):
    fields = [
        "model",
        "baseline_acc_overall", "baseline_ci_lo", "baseline_ci_hi",
        "baseline_acc_medqa", "baseline_ci_medqa_lo", "baseline_ci_medqa_hi",
        "baseline_acc_pubmedqa", "baseline_ci_pubmedqa_lo", "baseline_ci_pubmedqa_hi",
        "baseline_parse_rate", "baseline_fallback_count", "baseline_sec_per_q",
        "rag_acc_overall", "rag_ci_lo", "rag_ci_hi",
        "rag_acc_medqa", "rag_ci_medqa_lo", "rag_ci_medqa_hi",
        "rag_acc_pubmedqa", "rag_ci_pubmedqa_lo", "rag_ci_pubmedqa_hi",
        "rag_parse_rate", "rag_fallback_count", "rag_sec_per_q",
        "mcnemar_p_overall", "mcnemar_p_medqa", "mcnemar_p_pubmedqa",
        "n_test_answerable",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in stats_list:
            b, r = s["baseline"], s["rag"]
            w.writerow({
                "model": s["model"],
                "baseline_acc_overall":      b["acc_overall"],
                "baseline_ci_lo":            b["ci_overall"][0],
                "baseline_ci_hi":            b["ci_overall"][1],
                "baseline_acc_medqa":        b["acc_medqa"],
                "baseline_ci_medqa_lo":      b["ci_medqa"][0],
                "baseline_ci_medqa_hi":      b["ci_medqa"][1],
                "baseline_acc_pubmedqa":     b["acc_pubmedqa"],
                "baseline_ci_pubmedqa_lo":   b["ci_pubmedqa"][0],
                "baseline_ci_pubmedqa_hi":   b["ci_pubmedqa"][1],
                "baseline_parse_rate":       b["parse_rate"],
                "baseline_fallback_count":   b["fallback_count"],
                "baseline_sec_per_q":        b["mean_sec_per_q"],
                "rag_acc_overall":           r["acc_overall"],
                "rag_ci_lo":                 r["ci_overall"][0],
                "rag_ci_hi":                 r["ci_overall"][1],
                "rag_acc_medqa":             r["acc_medqa"],
                "rag_ci_medqa_lo":           r["ci_medqa"][0],
                "rag_ci_medqa_hi":           r["ci_medqa"][1],
                "rag_acc_pubmedqa":          r["acc_pubmedqa"],
                "rag_ci_pubmedqa_lo":        r["ci_pubmedqa"][0],
                "rag_ci_pubmedqa_hi":        r["ci_pubmedqa"][1],
                "rag_parse_rate":            r["parse_rate"],
                "rag_fallback_count":        r["fallback_count"],
                "rag_sec_per_q":             r["mean_sec_per_q"],
                "mcnemar_p_overall":         s["mcnemar_p_overall"],
                "mcnemar_p_medqa":           s.get("mcnemar_p_medqa"),
                "mcnemar_p_pubmedqa":        s.get("mcnemar_p_pubmedqa"),
                "n_test_answerable":         b["n_test_answerable"],
            })
    print(f"Saved CSV: {out_path}")


def print_table(stats_list):
    print("\n" + "=" * 130)
    print("FINAL RESULTS: BASELINE vs FULL RAG  (test, answerable questions only)")
    print("=" * 130)
    hdr = (f"{'Model':<16} {'Mode':<9} {'Overall [95% CI]':<26} "
           f"{'MedQA [95% CI]':<24} {'PubMedQA [95% CI]':<24} "
           f"{'Parse':>6} {'Fallbk':>7} {'Sec/Q':>7} {'McNemar p':>12}")
    print(hdr)
    print("-" * 130)
    for s in stats_list:
        for mode_key, label in [("baseline", "Baseline"), ("rag", "RAG")]:
            m = s[mode_key]
            overall = f"{m['acc_overall']:.4f} [{m['ci_overall'][0]:.3f},{m['ci_overall'][1]:.3f}]"
            medqa   = f"{m['acc_medqa']:.4f} [{m['ci_medqa'][0]:.3f},{m['ci_medqa'][1]:.3f}]"
            pubmed  = f"{m['acc_pubmedqa']:.4f} [{m['ci_pubmedqa'][0]:.3f},{m['ci_pubmedqa'][1]:.3f}]"
            p = f"{s['mcnemar_p_overall']:.5f}" if label == "RAG" else "-"
            print(
                f"{s['model']:<16} {label:<9} {overall:<26} {medqa:<24} {pubmed:<24} "
                f"{m['parse_rate']:>6.4f} {m['fallback_count']:>7} {m['mean_sec_per_q']:>7.3f} {p:>12}"
            )
        print()
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description="Final 5-model results table.")
    parser.add_argument("--base-dir", default="outputs/kaggle_qa/full")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--out-dir", default="outputs/analysis")
    args = parser.parse_args()

    candidates = sorted(
        d for d in glob.glob(os.path.join(args.base_dir, "*")) if os.path.isdir(d)
    )
    # Exclude helper dirs like gemma3-4b_rag
    candidates = [d for d in candidates if not os.path.basename(d).endswith("_rag")]
    if args.models:
        candidates = [d for d in candidates if os.path.basename(d) in args.models]

    stats_list = []
    for d in candidates:
        name = os.path.basename(d)
        p5 = os.path.join(d, "work", "phase5", f"{name}.jsonl")
        p8 = os.path.join(d, "work", "phase8", f"{name}.jsonl")
        if not os.path.exists(p5) or not os.path.exists(p8):
            print(f"Skipping {name}: missing phase5 or phase8")
            continue
        print(f"Processing {name}...")
        stats_list.append(model_stats(name, p5, p8))

    if not stats_list:
        print("No models found.")
        return

    # Sort by MedQA baseline accuracy descending
    stats_list.sort(key=lambda s: s["baseline"]["acc_medqa"] or 0, reverse=True)

    print_table(stats_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(stats_list, out_dir / "final_results.md")
    write_csv(stats_list, out_dir / "final_results.csv")


if __name__ == "__main__":
    main()
