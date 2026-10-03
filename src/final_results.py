"""Final 5-model results: baseline vs Full RAG accuracy comparison.

Reads Phase 5 (baseline) and Phase 8 (RAG) jsonl outputs for every model found
under --base-dir, computes accuracy (overall, MedQA, PubMedQA) with 95%
bootstrap CIs, McNemar p-values, parse rates, fallback counts, and mean
seconds per question. Writes outputs/analysis/final_results.md and .csv.
"""
import argparse
import csv
import fractions
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


def mcnemar_exact(a_correct, b_correct) -> float:
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


def format_p_value(p) -> str:
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


def cochran_mantel_haenszel(strata):
    """Cochran-Mantel-Haenszel test for 2x2xK tables.
    
    strata: list of tuples (a, b, c, d) where:
      a: Group 1 (RAG) Success (correct)
      b: Group 1 (RAG) Failure (incorrect)
      c: Group 2 (Baseline) Success (correct)
      d: Group 2 (Baseline) Failure (incorrect)
    """
    sum_a = sum(a for a, b, c, d in strata)
    sum_e = 0.0
    sum_var = 0.0
    R = 0.0
    S = 0.0
    sum_PR = 0.0
    sum_PS_QR = 0.0
    sum_QS = 0.0

    for a, b, c, d in strata:
        n1 = a + b
        n2 = c + d
        m1 = a + c
        m2 = b + d
        N = n1 + n2
        if N <= 1 or n1 == 0 or n2 == 0 or m1 == 0 or m2 == 0:
            continue
        e = n1 * m1 / N
        v = (n1 * n2 * m1 * m2) / (N * N * (N - 1))
        sum_e += e
        sum_var += v

        R_k = (a * d) / N
        S_k = (b * c) / N
        R += R_k
        S += S_k
        P_k = (a + d) / N
        Q_k = (b + c) / N
        sum_PR += P_k * R_k
        sum_PS_QR += P_k * S_k + Q_k * R_k
        sum_QS += Q_k * S_k

    diff = abs(sum_a - sum_e)
    chi2_raw = (diff ** 2) / sum_var if sum_var > 0 else 0.0
    chi2_cc = (max(0.0, diff - 0.5) ** 2) / sum_var if sum_var > 0 else 0.0
    p_raw = math.erfc(math.sqrt(chi2_raw / 2.0))
    p_cc = math.erfc(math.sqrt(chi2_cc / 2.0))

    or_mh = R / S if S > 0 else float("inf")
    if R > 0 and S > 0:
        var_ln_or = (sum_PR / (2 * R * R)) + (sum_PS_QR / (2 * R * S)) + (sum_QS / (2 * S * S))
        se = math.sqrt(var_ln_or)
        ci_lo = round(math.exp(math.log(or_mh) - 1.96 * se), 4)
        ci_hi = round(math.exp(math.log(or_mh) + 1.96 * se), 4)
    else:
        ci_lo, ci_hi = None, None

    return {
        "or_mh": round(or_mh, 4),
        "ci": (ci_lo, ci_hi),
        "chi2_raw": round(chi2_raw, 4),
        "p_raw": p_raw,
        "chi2_cc": round(chi2_cc, 4),
        "p_cc": p_cc,
    }


def compute_cmh_results(stats_list):
    results = {}
    for scope in ("overall", "medqa", "pubmedqa"):
        strata = []
        for s in stats_list:
            b, r = s["baseline"], s["rag"]
            if scope == "overall":
                b_corr, r_corr = b["correct_list"], r["correct_list"]
            elif scope == "medqa":
                b_corr, r_corr = b["correct_medqa"], r["correct_medqa"]
            else:
                b_corr, r_corr = b["correct_pubmedqa"], r["correct_pubmedqa"]
            a = sum(r_corr)
            b_cnt = len(r_corr) - a
            c = sum(b_corr)
            d = len(b_corr) - c
            strata.append((a, b_cnt, c, d))
        results[scope] = cochran_mantel_haenszel(strata)
    return results


def acc(rows):
    return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None


def parse_rate(rows):
    return round(sum(bool(r.get("parsed")) for r in rows) / len(rows), 4) if rows else None


def fallback_count(rows):
    return sum(1 for r in rows if r.get("pred_source") == "logprob_fallback")


def mean_sec(rows):
    return round(sum(r["seconds"] for r in rows) / len(rows), 4) if rows else None


# ---------- per-model stats ----------

# ---------- per-model stats ----------

def model_stats(model_name, p5_path, p8_path=None, p10_path=None):
    b_rows = [json.loads(l) for l in open(p5_path)] if p5_path and os.path.exists(p5_path) else []
    r_rows = [json.loads(l) for l in open(p8_path)] if p8_path and os.path.exists(p8_path) else []
    c_rows = [json.loads(l) for l in open(p10_path)] if p10_path and os.path.exists(p10_path) else []

    # Align by id
    b_by_id = {r["id"]: r for r in b_rows}
    r_by_id = {r["id"]: r for r in r_rows}
    c_by_id = {r["id"]: r for r in c_rows}
    common = sorted(set(b_by_id) & set(r_by_id)) if r_rows else sorted(b_by_id.keys())

    result = {
        "model": model_name,
        "n_common": len(common),
        "b_by_id": b_by_id,
        "r_by_id": r_by_id,
        "c_by_id": c_by_id,
    }

    # Baseline stats
    if b_rows:
        test_ans = [r for r in b_rows if r.get("split") == "test" and not r.get("should_abstain")]
        test_medqa = [r for r in test_ans if r.get("dataset") == "medqa"]
        test_pubmed = [r for r in test_ans if r.get("dataset") == "pubmedqa"]

        corr_all = [r["correct"] for r in test_ans]
        corr_med = [r["correct"] for r in test_medqa]
        corr_pub = [r["correct"] for r in test_pubmed]

        # If model only evaluated on one dataset (e.g. MedQA only for medpsy-4b),
        # acc_overall is None ("n/a") to avoid conflating a partial benchmark with 2-dataset overall.
        has_full_eval = bool(test_medqa and test_pubmed)
        trunc_rate = round(sum(bool(r.get("truncated")) for r in b_rows) / len(b_rows), 4) if b_rows else None

        result["baseline"] = {
            "n_test_answerable": len(test_ans),
            "acc_overall": acc(test_ans) if has_full_eval else None,
            "ci_overall": bootstrap_ci(corr_all) if has_full_eval else (None, None),
            "acc_medqa": acc(test_medqa) if test_medqa else None,
            "ci_medqa": bootstrap_ci(corr_med) if test_medqa else (None, None),
            "acc_pubmedqa": acc(test_pubmed) if test_pubmed else None,
            "ci_pubmedqa": bootstrap_ci(corr_pub) if test_pubmed else (None, None),
            "parse_rate": parse_rate(b_rows),
            "truncation_rate": trunc_rate,
            "fallback_count": fallback_count(b_rows),
            "mean_sec_per_q": mean_sec(test_ans),
            "correct_list": corr_all,
            "correct_medqa": corr_med,
            "correct_pubmedqa": corr_pub,
        }
    else:
        result["baseline"] = None

    # RAG stats
    if r_rows:
        test_ans = [r for r in r_rows if r.get("split") == "test" and not r.get("should_abstain")]
        test_medqa = [r for r in test_ans if r.get("dataset") == "medqa"]
        test_pubmed = [r for r in test_ans if r.get("dataset") == "pubmedqa"]

        corr_all = [r["correct"] for r in test_ans]
        corr_med = [r["correct"] for r in test_medqa]
        corr_pub = [r["correct"] for r in test_pubmed]
        has_full_eval = bool(test_medqa and test_pubmed)
        trunc_rate = round(sum(bool(r.get("truncated")) for r in r_rows) / len(r_rows), 4) if r_rows else None

        result["rag"] = {
            "n_test_answerable": len(test_ans),
            "acc_overall": acc(test_ans) if has_full_eval else None,
            "ci_overall": bootstrap_ci(corr_all) if has_full_eval else (None, None),
            "acc_medqa": acc(test_medqa) if test_medqa else None,
            "ci_medqa": bootstrap_ci(corr_med) if test_medqa else (None, None),
            "acc_pubmedqa": acc(test_pubmed) if test_pubmed else None,
            "ci_pubmedqa": bootstrap_ci(corr_pub) if test_pubmed else (None, None),
            "parse_rate": parse_rate(r_rows),
            "truncation_rate": trunc_rate,
            "fallback_count": fallback_count(r_rows),
            "mean_sec_per_q": mean_sec(test_ans),
            "correct_list": corr_all,
            "correct_medqa": corr_med,
            "correct_pubmedqa": corr_pub,
        }

        # McNemar RAG vs Baseline
        if b_rows:
            b_ans_by_id = {r["id"]: r for r in b_by_id.values()
                           if r.get("split") == "test" and not r.get("should_abstain")}
            r_ans_by_id = {r["id"]: r for r in r_by_id.values()
                           if r.get("split") == "test" and not r.get("should_abstain")}
            shared_ans = sorted(set(b_ans_by_id) & set(r_ans_by_id))

            b_corr = [b_ans_by_id[i]["correct"] for i in shared_ans]
            r_corr = [r_ans_by_id[i]["correct"] for i in shared_ans]
            result["mcnemar_p_overall"] = mcnemar_exact(b_corr, r_corr) if has_full_eval else None

            for ds in ("medqa", "pubmedqa"):
                shared_ds = [i for i in shared_ans if b_ans_by_id[i].get("dataset") == ds]
                if shared_ds:
                    b_ds = [b_ans_by_id[i]["correct"] for i in shared_ds]
                    r_ds = [r_ans_by_id[i]["correct"] for i in shared_ds]
                    result[f"mcnemar_p_{ds}"] = mcnemar_exact(b_ds, r_ds)
                else:
                    result[f"mcnemar_p_{ds}"] = None
        else:
            result["mcnemar_p_overall"] = None
            result["mcnemar_p_medqa"] = None
            result["mcnemar_p_pubmedqa"] = None
    else:
        result["rag"] = None
        result["mcnemar_p_overall"] = None
        result["mcnemar_p_medqa"] = None
        result["mcnemar_p_pubmedqa"] = None

    # Context (+Abstract) stats
    if c_rows:
        test_ans = [r for r in c_rows if r.get("split") == "test" and not r.get("should_abstain")]
        test_pubmed = [r for r in test_ans if r.get("dataset") == "pubmedqa"]
        corr_pub = [r["correct"] for r in test_pubmed]
        trunc_rate = round(sum(bool(r.get("truncated")) for r in c_rows) / len(c_rows), 4) if c_rows else None

        result["context"] = {
            "n_test": len(test_pubmed),
            "acc_pubmedqa": acc(test_pubmed),
            "ci_pubmedqa": bootstrap_ci(corr_pub),
            "parse_rate": parse_rate(c_rows),
            "truncation_rate": trunc_rate,
            "fallback_count": fallback_count(c_rows),
            "mean_sec_per_q": mean_sec(test_pubmed),
            "correct_pubmedqa": corr_pub,
        }
    else:
        result["context"] = None

    return result


def compute_medpsy_comparisons(stats_list):
    """Compute pairwise exact McNemar comparison of MedPsy-4B vs all other baselines on MedQA."""
    mp_stat = next((s for s in stats_list if s["model"] == "medpsy-4b"), None)
    if not mp_stat or not mp_stat.get("b_by_id"):
        return []

    mp_dict = {
        r["id"]: r for r in mp_stat["b_by_id"].values()
        if r.get("split") == "test" and r.get("dataset") == "medqa" and not r.get("should_abstain")
    }
    if not mp_dict:
        return []

    comps = []
    for s in stats_list:
        if s["model"] == "medpsy-4b":
            continue
        other_dict = {
            r["id"]: r for r in s.get("b_by_id", {}).values()
            if r.get("split") == "test" and r.get("dataset") == "medqa" and not r.get("should_abstain")
        }
        shared_ids = sorted(set(mp_dict.keys()) & set(other_dict.keys()))
        if not shared_ids:
            continue

        b = sum(1 for i in shared_ids if mp_dict[i]["correct"] and not other_dict[i]["correct"])
        c = sum(1 for i in shared_ids if not mp_dict[i]["correct"] and other_dict[i]["correct"])
        p = mcnemar_exact([mp_dict[i]["correct"] for i in shared_ids],
                          [other_dict[i]["correct"] for i in shared_ids])

        comps.append({
            "model": s["model"],
            "n_shared": len(shared_ids),
            "medpsy_corr": sum(mp_dict[i]["correct"] for i in shared_ids),
            "other_corr": sum(other_dict[i]["correct"] for i in shared_ids),
            "medpsy_only_wins": b,
            "other_only_wins": c,
            "p_value": p,
            "significant": p < 0.05,
        })
    return comps


# ---------- output ----------

def fmt_acc_ci(acc_val, ci):
    if acc_val is None:
        return "n/a"
    lo, hi = ci if ci else (None, None)
    if lo is None:
        return f"{acc_val:.4f}"
    return f"{acc_val:.4f} [{lo:.3f}, {hi:.3f}]"


def write_markdown(stats_list, out_path):
    lines = []
    lines.append("# Final Results: Baseline vs Full RAG Benchmark\n")
    lines.append(
        "> **Test set, answerable questions only** (should_abstain==False).\n"
        "> 95% CI: percentile bootstrap (1 000 resamples, seed 42).\n"
        "> McNemar: exact two-sided binomial test on discordant pairs.\n"
        "> Sec/Q: mean seconds per question over test-split answerable rows only.\n"
        "> **Note on medpsy-4b**: Evaluated on MedQA test baseline ($N=500$) and PubMedQA test +Abstract ($N=500$). "
        "Missing modes/splits are displayed as `n/a`.\n"
    )
    lines.append("")

    # Main accuracy table
    lines.append("## Accuracy")
    lines.append(
        "| Model | Baseline Overall [95% CI] | Baseline MedQA [95% CI] | Baseline PubMedQA [95% CI] "
        "| RAG Overall [95% CI] | RAG MedQA [95% CI] | RAG PubMedQA [95% CI] "
        "| McNemar p overall | McNemar p MedQA | McNemar p PubMedQA |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in stats_list:
        b = s.get("baseline")
        r = s.get("rag")
        b_ov = fmt_acc_ci(b["acc_overall"], b["ci_overall"]) if b else "n/a"
        b_mq = fmt_acc_ci(b["acc_medqa"], b["ci_medqa"]) if b else "n/a"
        b_pm = fmt_acc_ci(b["acc_pubmedqa"], b["ci_pubmedqa"]) if b else "n/a"

        r_ov = fmt_acc_ci(r["acc_overall"], r["ci_overall"]) if r else "n/a"
        r_mq = fmt_acc_ci(r["acc_medqa"], r["ci_medqa"]) if r else "n/a"
        r_pm = fmt_acc_ci(r["acc_pubmedqa"], r["ci_pubmedqa"]) if r else "n/a"

        p_ov = format_p_value(s.get("mcnemar_p_overall"))
        p_med = format_p_value(s.get("mcnemar_p_medqa"))
        p_pub = format_p_value(s.get("mcnemar_p_pubmedqa"))

        lines.append(
            f"| {s['model']} | {b_ov} | {b_mq} | {b_pm} | {r_ov} | {r_mq} | {r_pm} | {p_ov} | {p_med} | {p_pub} |"
        )

    lines.append("")
    lines.append("## Parse Rate, Truncation, and Speed")
    lines.append(
        "| Model | Baseline Parse Rate | Baseline Truncated | Baseline Fallback | Baseline Sec/Q "
        "| RAG Parse Rate | RAG Truncated | RAG Fallback | RAG Sec/Q |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in stats_list:
        b = s.get("baseline")
        r = s.get("rag")
        b_pr = f"{b['parse_rate']:.4f}" if b and b["parse_rate"] is not None else "n/a"
        b_tr = f"{b['truncation_rate']:.1%}" if b and b["truncation_rate"] is not None else "0.0%"
        b_fb = str(b["fallback_count"]) if b and b["fallback_count"] is not None else "n/a"
        b_sq = f"{b['mean_sec_per_q']:.3f} s" if b and b["mean_sec_per_q"] is not None else "n/a"

        r_pr = f"{r['parse_rate']:.4f}" if r and r["parse_rate"] is not None else "n/a"
        r_tr = f"{r['truncation_rate']:.1%}" if r and r["truncation_rate"] is not None else "0.0%"
        r_fb = str(r["fallback_count"]) if r and r["fallback_count"] is not None else "n/a"
        r_sq = f"{r['mean_sec_per_q']:.3f} s" if r and r["mean_sec_per_q"] is not None else "n/a"

        lines.append(
            f"| {s['model']} | {b_pr} | {b_tr} | {b_fb} | {b_sq} | {r_pr} | {r_tr} | {r_fb} | {r_sq} |"
        )

    # PubMedQA +Abstract section
    ctx_models = [s for s in stats_list if s.get("context")]
    if ctx_models:
        lines.append("")
        lines.append("## PubMedQA +Abstract (Oracle Context Benchmark)")
        lines.append(
            "| Model | PubMedQA +Abstract Acc [95% CI] | Parse Rate | Truncated | Sec/Q |"
        )
        lines.append("|---|---|---|---|---|")
        for s in ctx_models:
            c = s["context"]
            c_acc = fmt_acc_ci(c["acc_pubmedqa"], c["ci_pubmedqa"])
            c_pr = f"{c['parse_rate']:.4f}" if c.get("parse_rate") is not None else "n/a"
            c_tr = f"{c['truncation_rate']:.1%}" if c.get("truncation_rate") is not None else "0.0%"
            c_sq = f"{c['mean_sec_per_q']:.3f} s" if c.get("mean_sec_per_q") is not None else "n/a"
            lines.append(f"| {s['model']} | {c_acc} | {c_pr} | {c_tr} | {c_sq} |")

    # MedPsy-4B Pairwise Comparison Section
    medpsy_comps = compute_medpsy_comparisons(stats_list)
    if medpsy_comps:
        lines.append("")
        lines.append("## MedPsy-4B vs. Standard SLM Baselines (MedQA Test Split, $N=500$)")
        lines.append(
            "> Pairwise McNemar exact two-sided test on discordant pairs for MedQA test ($N=500$).  \n"
            "> - **MedPsy-Only Wins ($b$)**: Questions answered correctly by MedPsy-4B but wrong by baseline.  \n"
            "> - **Baseline-Only Wins ($c$)**: Questions answered correctly by baseline but wrong by MedPsy-4B.\n"
        )
        lines.append(
            "| Comparison | MedPsy-4B Acc | Baseline Acc | MedPsy-Only ($b$) | Baseline-Only ($c$) | McNemar $p$-value | Significant? |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for comp in medpsy_comps:
            p_val = comp["p_value"]
            p_str = format_p_value(p_val)
            is_sig_001 = (p_val < 0.001) if isinstance(p_val, (int, float)) else True
            is_sig_05 = (p_val < 0.05) if isinstance(p_val, (int, float)) else True
            sig_str = "**Yes (p < 0.001)**" if is_sig_001 else ("Yes (p < 0.05)" if is_sig_05 else f"No (p = {p_val:.3f})")
            lines.append(
                f"| MedPsy-4B vs. **{comp['model']}** | {comp['medpsy_corr']/comp['n_shared']:.1%} | "
                f"{comp['other_corr']/comp['n_shared']:.1%} | {comp['medpsy_only_wins']} | "
                f"{comp['other_only_wins']} | {p_str} | {sig_str} |"
            )

    lines.append("")
    lines.append("## Key Findings")
    lines.append(
        "For the baseline-vs-Full-RAG comparisons, no multiple-test correction is needed; "
        "a result is significant if raw $p < 0.05$."
    )
    lines.append("")

    sig_raw = []
    for s in stats_list:
        if not s.get("rag") or not s.get("baseline"):
            continue
        p = s.get("mcnemar_p_overall")
        if p is not None and p < 0.05:
            direction = "RAG > Baseline" if s["rag"]["acc_overall"] > s["baseline"]["acc_overall"] else "Baseline > RAG"
            sig_raw.append(f"  - {s['model']}: p={format_p_value(p)} ({direction}, "
                           f"Baseline {s['baseline']['acc_overall']:.4f} → RAG {s['rag']['acc_overall']:.4f})")
        for ds, key in (("MedQA", "mcnemar_p_medqa"), ("PubMedQA", "mcnemar_p_pubmedqa")):
            p_ds = s.get(key)
            if p_ds is not None and p_ds < 0.05:
                b_acc = s["baseline"]["acc_medqa"] if ds == "MedQA" else s["baseline"]["acc_pubmedqa"]
                r_acc = s["rag"]["acc_medqa"] if ds == "MedQA" else s["rag"]["acc_pubmedqa"]
                direction = "RAG > Baseline" if r_acc > b_acc else "Baseline > RAG"
                sig_raw.append(f"  - {s['model']} [{ds}]: p={format_p_value(p_ds)} ({direction}, "
                               f"Baseline {b_acc:.4f} → RAG {r_acc:.4f})")

    if sig_raw:
        lines.append("**Significant before correction (raw p < 0.05, Full RAG vs Baseline):**")
        lines.extend(sig_raw)
    else:
        lines.append("No standard model achieves p < 0.05 for Full RAG vs Baseline (overall or per dataset).")
    lines.append("")

    # CMH pooled analysis across models with full evaluation
    cmh_candidates = [s for s in stats_list if s.get("baseline") and s.get("rag") and s["baseline"].get("acc_pubmedqa") is not None and s["rag"].get("acc_pubmedqa") is not None]
    if cmh_candidates:
        cmh = compute_cmh_results(cmh_candidates)
        lines.append("## Pooled Cochran–Mantel–Haenszel (CMH) Analysis")
        lines.append("")
        lines.append(
            f"Across all {len(cmh_candidates)} standard SLMs with parametric vs. RAG runs (stratified 2x2 meta-analysis):\n"
        )
        lines.append(
            "| Scope | Mantel–Haenszel Common OR [95% CI] | CMH $\\chi^2$ (df=1) | p-value (raw) | p-value (continuity-corrected) | Significant? |"
        )
        lines.append("|---|---|---|---|---|---|")
        for scope in ("overall", "medqa", "pubmedqa"):
            res = cmh[scope]
            sig_str = "**Yes (p < 0.001, RAG > Baseline)**" if res["p_raw"] < 0.01 else (
                "Yes (p < 0.05)" if res["p_raw"] < 0.05 else f"No (p = {res['p_raw']:.3f})"
            )
            ci_str = f"[{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]" if res["ci"][0] else "-"
            lines.append(
                f"| **{scope.capitalize() if scope != 'medqa' else 'MedQA'}** "
                f"| {res['or_mh']:.4f} {ci_str} "
                f"| {res['chi2_raw']:.4f} "
                f"| {res['p_raw']:.5e} "
                f"| {res['p_cc']:.5e} "
                f"| {sig_str} |"
            )
        lines.append("")

    lines.append("---")
    lines.append(
        f"*Retrieval+rerank cost not included in Sec/Q above: 0.12730 s/q "
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
        "baseline_parse_rate", "baseline_truncation_rate", "baseline_fallback_count", "baseline_sec_per_q",
        "rag_acc_overall", "rag_ci_lo", "rag_ci_hi",
        "rag_acc_medqa", "rag_ci_medqa_lo", "rag_ci_medqa_hi",
        "rag_acc_pubmedqa", "rag_ci_pubmedqa_lo", "rag_ci_pubmedqa_hi",
        "rag_parse_rate", "rag_truncation_rate", "rag_fallback_count", "rag_sec_per_q",
        "context_acc_pubmedqa", "context_ci_pubmedqa_lo", "context_ci_pubmedqa_hi",
        "context_parse_rate", "context_truncation_rate", "context_sec_per_q",
        "mcnemar_p_overall", "mcnemar_p_medqa", "mcnemar_p_pubmedqa",
        "n_test_answerable",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in stats_list:
            b, r, c = s.get("baseline"), s.get("rag"), s.get("context")
            w.writerow({
                "model": s["model"],
                "baseline_acc_overall":      b["acc_overall"] if b else "",
                "baseline_ci_lo":            b["ci_overall"][0] if b and b["ci_overall"] else "",
                "baseline_ci_hi":            b["ci_overall"][1] if b and b["ci_overall"] else "",
                "baseline_acc_medqa":        b["acc_medqa"] if b else "",
                "baseline_ci_medqa_lo":      b["ci_medqa"][0] if b and b["ci_medqa"] else "",
                "baseline_ci_medqa_hi":      b["ci_medqa"][1] if b and b["ci_medqa"] else "",
                "baseline_acc_pubmedqa":     b["acc_pubmedqa"] if b else "",
                "baseline_ci_pubmedqa_lo":   b["ci_pubmedqa"][0] if b and b["ci_pubmedqa"] else "",
                "baseline_ci_pubmedqa_hi":   b["ci_pubmedqa"][1] if b and b["ci_pubmedqa"] else "",
                "baseline_parse_rate":       b["parse_rate"] if b else "",
                "baseline_truncation_rate":  b["truncation_rate"] if b else "",
                "baseline_fallback_count":   b["fallback_count"] if b else "",
                "baseline_sec_per_q":        b["mean_sec_per_q"] if b else "",
                "rag_acc_overall":           r["acc_overall"] if r else "",
                "rag_ci_lo":                 r["ci_overall"][0] if r and r["ci_overall"] else "",
                "rag_ci_hi":                 r["ci_overall"][1] if r and r["ci_overall"] else "",
                "rag_acc_medqa":             r["acc_medqa"] if r else "",
                "rag_ci_medqa_lo":           r["ci_medqa"][0] if r and r["ci_medqa"] else "",
                "rag_ci_medqa_hi":           r["ci_medqa"][1] if r and r["ci_medqa"] else "",
                "rag_acc_pubmedqa":          r["acc_pubmedqa"] if r else "",
                "rag_ci_pubmedqa_lo":        r["ci_pubmedqa"][0] if r and r["ci_pubmedqa"] else "",
                "rag_ci_pubmedqa_hi":        r["ci_pubmedqa"][1] if r and r["ci_pubmedqa"] else "",
                "rag_parse_rate":            r["parse_rate"] if r else "",
                "rag_truncation_rate":       r["truncation_rate"] if r else "",
                "rag_fallback_count":        r["fallback_count"] if r else "",
                "rag_sec_per_q":             r["mean_sec_per_q"] if r else "",
                "context_acc_pubmedqa":      c["acc_pubmedqa"] if c else "",
                "context_ci_pubmedqa_lo":    c["ci_pubmedqa"][0] if c and c["ci_pubmedqa"] else "",
                "context_ci_pubmedqa_hi":    c["ci_pubmedqa"][1] if c and c["ci_pubmedqa"] else "",
                "context_parse_rate":        c["parse_rate"] if c else "",
                "context_truncation_rate":   c["truncation_rate"] if c else "",
                "context_sec_per_q":         c["mean_sec_per_q"] if c else "",
                "mcnemar_p_overall":         s.get("mcnemar_p_overall", ""),
                "mcnemar_p_medqa":           s.get("mcnemar_p_medqa", ""),
                "mcnemar_p_pubmedqa":        s.get("mcnemar_p_pubmedqa", ""),
                "n_test_answerable":         b["n_test_answerable"] if b else (r["n_test_answerable"] if r else 0),
            })
    print(f"Saved CSV: {out_path}")


def print_table(stats_list):
    print("\n" + "=" * 140)
    print("FINAL RESULTS: BASELINE vs FULL RAG BENCHMARK  (test-split, answerable questions only)")
    print("Sec/Q: mean over test-split answerable rows. McNemar: exact two-sided binomial.")
    print("=" * 140)
    hdr = (f"{'Model':<16} {'Mode':<9} {'Overall [95% CI]':<28} "
           f"{'MedQA [95% CI]':<26} {'PubMedQA [95% CI]':<26} "
           f"{'Parse':>6} {'Trunc':>7} {'Fallbk':>7} {'Sec/Q':>7} {'McNemar p':>12}")
    print(hdr)
    print("-" * 140)
    for s in stats_list:
        for mode_key, label in [("baseline", "Baseline"), ("rag", "RAG"), ("context", "+Abstract")]:
            m = s.get(mode_key)
            if not m:
                continue
            ov = f"{m['acc_overall']:.4f} [{m['ci_overall'][0]:.3f},{m['ci_overall'][1]:.3f}]" if m.get("acc_overall") is not None else "n/a"
            med = f"{m['acc_medqa']:.4f} [{m['ci_medqa'][0]:.3f},{m['ci_medqa'][1]:.3f}]" if m.get("acc_medqa") is not None else "n/a"
            pub = f"{m['acc_pubmedqa']:.4f} [{m['ci_pubmedqa'][0]:.3f},{m['ci_pubmedqa'][1]:.3f}]" if m.get("acc_pubmedqa") is not None else "n/a"
            pr = f"{m['parse_rate']:.4f}" if m.get("parse_rate") is not None else "n/a"
            tr = f"{m['truncation_rate']:.1%}" if m.get("truncation_rate") is not None else "0.0%"
            fb = str(m.get("fallback_count", "n/a"))
            sq = f"{m['mean_sec_per_q']:.3f}" if m.get("mean_sec_per_q") is not None else "n/a"
            p = format_p_value(s.get("mcnemar_p_overall")) if (label == "RAG" and s.get("mcnemar_p_overall") is not None) else "-"
            print(
                f"{s['model']:<16} {label:<9} {ov:<28} {med:<26} {pub:<26} "
                f"{pr:>6} {tr:>7} {fb:>7} {sq:>7} {p:>12}"
            )
        print()
    print("=" * 140)

    # MedPsy pairwise McNemar comparison
    medpsy_comps = compute_medpsy_comparisons(stats_list)
    if medpsy_comps:
        print("\nMEDPSY-4B vs OTHER BASELINES (MedQA Test Split, N=500):")
        print("-" * 100)
        print(f"{'Baseline Comparison':<30} {'MedPsy Acc':<12} {'Other Acc':<12} {'b (MedPsy+)':<14} {'c (Other+)':<14} {'McNemar p':<12}")
        print("-" * 100)
        for comp in medpsy_comps:
            p_val = comp["p_value"]
            p_str = format_p_value(p_val)
            print(f"vs {comp['model']:<27} {comp['medpsy_corr']/comp['n_shared']:<12.1%} {comp['other_corr']/comp['n_shared']:<12.1%} {comp['medpsy_only_wins']:<14} {comp['other_only_wins']:<14} {p_str:<12}")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Final model results table.")
    parser.add_argument("--base-dir", default="outputs/kaggle_qa/full")
    parser.add_argument("--context-dir", default="outputs/kaggle_qa/context")
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
        p10 = os.path.join(args.context_dir, "work", "phase10", f"{name}.jsonl") if args.context_dir else None
        has_p5 = os.path.exists(p5)
        has_p8 = os.path.exists(p8)
        has_p10 = p10 and os.path.exists(p10)

        if not has_p5 and not has_p8 and not has_p10:
            print(f"Skipping {name}: no phase5, phase8, or phase10 files found")
            continue

        print(f"Processing {name} (p5={has_p5}, p8={has_p8}, p10={bool(has_p10)})...")
        stats_list.append(model_stats(
            name,
            p5 if has_p5 else None,
            p8 if has_p8 else None,
            p10 if has_p10 else None,
        ))

    if not stats_list:
        print("No models found.")
        return

    # Sort by MedQA baseline accuracy descending
    stats_list.sort(key=lambda s: (s["baseline"]["acc_medqa"] if s.get("baseline") else 0) or 0, reverse=True)

    print_table(stats_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(stats_list, out_dir / "final_results.md")
    write_csv(stats_list, out_dir / "final_results.csv")


if __name__ == "__main__":
    main()
