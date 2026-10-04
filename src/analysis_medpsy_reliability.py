"""Calibrated reliability score and abstention threshold for MedPsy-4B.

Protocol (Rule 6: never tune on test):
  * Everything learned here (feature scaling, logistic-regression weights, thresholds) is fitted
    on the VALIDATION split only (MedQA 200 + unanswerable 50 baseline; PubMedQA 200 context).
  * The frozen model and thresholds are then applied ONCE to the held-out TEST split
    (MedQA 500 + unanswerable 150; PubMedQA 500) and reported with bootstrap 95% CIs.

Reliability score = predicted probability that answering would be unsafe
  (MedQA/unanswerable set: question is unanswerable OR the answer is wrong;
   PubMedQA set: the answer is wrong).
Features (all produced by the model run itself, no extra GPU time):
  truncated, generated-token fraction of the 1,024 cap, max option probability,
  top-1 minus top-2 option probability, normalised option entropy,
  and (MedQA set only, if evidence is available) the top MedCPT rerank score.

Rules compared on test (thresholds chosen on validation):
  none              always answer
  truncated         abstain if the thinking budget is exhausted (pre-specified, no parameters)
  tokens            abstain if generated tokens >= t                   (t: max abstention F1 on val)
  score_f1          abstain if reliability score >= tau                (tau: max abstention F1 on val)
  score_safe        abstain if reliability score >= tau                (tau: max val coverage with
                                                                        val accuracy-when-answering >= TARGET,
                                                                        val coverage >= MIN_COVERAGE)
Metrics: abstention precision/recall/F1 (positive = unanswerable), coverage, accuracy when
answering, unsafe-answer rate (answered AND (wrong OR unanswerable)) over all questions,
AUROC and AURC of each signal, ECE/Brier of raw confidence vs calibrated score.

Outputs:
  outputs/analysis/medpsy_reliability.md
  outputs/analysis/medpsy_reliability.json
  outputs/analysis/medpsy_reliability_model.json   (frozen weights + thresholds, for the chatbot)
  outputs/figures/fig13_reliability.png / .svg

Run (local, CPU, seconds):
  python -m src.analysis_medpsy_reliability
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

TOKEN_CAP = 1024
SAFE_TARGET = {"medqa": 0.95, "pubmedqa": 0.90}   # pre-specified target accuracy-when-answering ("safe" point)
MIN_COVERAGE = 0.50         # pre-specified: a safe threshold may not refuse more than half the answerable questions
FIXED_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6)
L2 = 1.0                    # pre-specified ridge penalty (not tuned)
N_BOOT = 1000
SEED = 42


# ── I/O ────────────────────────────────────────────────────────────────────────

def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def resolve(path_arg, phase, split):
    """Return path_arg if it exists, else search outputs/kaggle_qa/** for a medpsy-4b file of that phase
    containing rows of the requested split."""
    if path_arg and Path(path_arg).exists():
        return Path(path_arg)
    for cand in sorted(glob.glob(f"outputs/kaggle_qa/**/{phase}/medpsy-4b.jsonl", recursive=True)):
        try:
            with open(cand) as fh:
                first = json.loads(next(l for l in fh if l.strip()))
            if first.get("split") == split:
                return Path(cand)
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(f"No medpsy-4b {phase} file with split={split} (tried {path_arg} and outputs/kaggle_qa/**)")


def load_rows(path, split):
    rows = [r for r in read_jsonl(path) if r.get("split") == split]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), f"[CHECK FAILED] duplicate ids in {path}"
    assert rows, f"[CHECK FAILED] no {split} rows in {path}"
    return rows


# ── Features ───────────────────────────────────────────────────────────────────

def option_stats(lp):
    if not lp:
        return 0.0, 0.0, 1.0
    p = np.array(sorted(lp.values(), reverse=True), dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
    top1 = p[0]
    margin = p[0] - (p[1] if len(p) > 1 else 0.0)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(len(p))) if len(p) > 1 else 0.0
    return float(top1), float(margin), ent


def featurise(rows, evidence=None):
    names = ["truncated", "token_frac", "confidence", "margin", "entropy"]
    if evidence is not None:
        names.append("top_rerank")
    X = []
    for r in rows:
        conf, margin, ent = option_stats(r.get("letter_probs"))
        f = [float(bool(r.get("truncated"))),
             min(float(r.get("gen_tokens") or 0) / TOKEN_CAP, 1.0),
             conf, margin, ent]
        if evidence is not None:
            ps = evidence.get(r["id"], {}).get("passages") or []
            f.append(float(ps[0]["rerank_score"]) if ps else 0.0)
        X.append(f)
    return np.array(X, dtype=float), names


def risk_label(r):
    """1 = answering would be unsafe (unanswerable question, or wrong answer)."""
    return int(bool(r.get("should_abstain")) or not bool(r.get("correct")))


# ── Logistic regression (ridge, Newton/IRLS; numpy only) ──────────────────────

def fit_logreg(X, y, l2=L2, iters=100):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = np.eye(Z.shape[1]) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + reg @ w
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"mean": mu.tolist(), "std": sd.tolist(), "weights": w.tolist()}


def predict(model, X):
    Z = np.hstack([np.ones((len(X), 1)), (X - np.array(model["mean"])) / np.array(model["std"])])
    return 1 / (1 + np.exp(-Z @ np.array(model["weights"])))


# ── Metrics ────────────────────────────────────────────────────────────────────

def rule_metrics(should_abstain, correct, abstain):
    sa, c, a = (np.asarray(v, dtype=bool) for v in (should_abstain, correct, abstain))
    tp, fp = int((sa & a).sum()), int((~sa & a).sum())
    fn = int((sa & ~a).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ans = ~sa
    n_ans = int(ans.sum())
    answered = ans & ~a
    cov = answered.sum() / n_ans if n_ans else 0.0
    acc_ans = (c & answered).sum() / answered.sum() if answered.sum() else 0.0
    unsafe = (~a & (sa | ~c)).sum() / len(sa) if len(sa) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "coverage": float(cov),
            "acc_answering": float(acc_ans), "unsafe_rate": float(unsafe),
            "n_abstained": int(a.sum()), "n": int(len(sa))}


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                      # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def risk_coverage(score, y):
    """Answer the lowest-risk questions first; return coverage and risk arrays, and AURC."""
    order = np.argsort(np.asarray(score, float), kind="mergesort")
    yy = np.asarray(y, float)[order]
    k = np.arange(1, len(yy) + 1)
    risk = np.cumsum(yy) / k
    cov = k / len(yy)
    return cov, risk, float(risk.mean())


def ece(prob_ok, ok, bins=10):
    prob_ok, ok = np.asarray(prob_ok, float), np.asarray(ok, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob_ok >= lo) & ((prob_ok < hi) if hi < 1 else (prob_ok <= hi))
        if m.any():
            e += m.mean() * abs(prob_ok[m].mean() - ok[m].mean())
    return float(e)


def brier(prob_ok, ok):
    return float(np.mean((np.asarray(prob_ok, float) - np.asarray(ok, float)) ** 2))


def boot_ci(fn, n, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ── Threshold selection (validation only) ──────────────────────────────────────

def pick_threshold_f1(score, sa, c):
    best = None
    for t in np.unique(score):
        m = rule_metrics(sa, c, score >= t)
        key = (round(m["f1"], 6), m["coverage"])
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def pick_threshold_safe(score, sa, c, target, min_cov=MIN_COVERAGE):
    """Largest val coverage with val accuracy-when-answering >= target; if the target cannot be met
    with coverage >= min_cov, the highest val accuracy subject to coverage >= min_cov."""
    cands = []
    for t in list(np.unique(score)) + [np.inf]:
        m = rule_metrics(sa, c, score >= t)
        if m["coverage"] >= min_cov:
            cands.append((m["acc_answering"] >= target, m["coverage"], m["acc_answering"], float(t)))
    ok = [x for x in cands if x[0]]
    if ok:
        return max(ok, key=lambda x: (x[1], x[2]))[3]
    return max(cands, key=lambda x: (x[2], x[1]))[3]


def threshold_for_coverage(score, sa, cov):
    """Threshold so that a fraction `cov` of the answerable validation questions is answered."""
    if cov >= 1.0:
        return np.inf
    return float(np.quantile(score[~sa], cov, method="higher"))


# ── Main analysis for one task ─────────────────────────────────────────────────

def analyse(name, val_rows, test_rows, evidence, has_unanswerable, target):
    Xv, feats = featurise(val_rows, evidence)
    Xt, _ = featurise(test_rows, evidence)
    yv = np.array([risk_label(r) for r in val_rows])
    yt = np.array([risk_label(r) for r in test_rows])
    sa_v = np.array([bool(r.get("should_abstain")) for r in val_rows])
    sa_t = np.array([bool(r.get("should_abstain")) for r in test_rows])
    c_v = np.array([bool(r.get("correct")) for r in val_rows])
    c_t = np.array([bool(r.get("correct")) for r in test_rows])

    model = fit_logreg(Xv, yv)
    s_v, s_t = predict(model, Xv), predict(model, Xt)
    tok_v, tok_t = Xv[:, feats.index("token_frac")], Xt[:, feats.index("token_frac")]
    conf_t = Xt[:, feats.index("confidence")]

    thr = {"tokens": pick_threshold_f1(tok_v, sa_v, c_v) if has_unanswerable else
           pick_threshold_safe(tok_v, sa_v, c_v, target),
           "score_f1": pick_threshold_f1(s_v, sa_v, c_v) if has_unanswerable else None,
           "score_safe": pick_threshold_safe(s_v, sa_v, c_v, target)}

    def abstain_vec(rule, idx=None):
        sl = slice(None) if idx is None else idx
        if rule == "none":
            return np.zeros(len(test_rows), bool)[sl]
        if rule == "truncated":
            return (Xt[:, feats.index("truncated")] > 0.5)[sl]
        if rule == "tokens":
            return (tok_t >= thr["tokens"])[sl]
        return (s_t >= thr[rule])[sl]

    rules = ["none", "truncated", "tokens", "score_safe"] + (["score_f1"] if has_unanswerable else [])
    results = {}
    for rule in rules:
        point = rule_metrics(sa_t, c_t, abstain_vec(rule))
        cis = {}
        for key in ("f1", "recall", "coverage", "acc_answering", "unsafe_rate"):
            cis[key] = boot_ci(lambda i, k=key, r=rule: rule_metrics(sa_t[i], c_t[i], abstain_vec(r, i))[k],
                               len(test_rows))
        results[rule] = {**point, "ci": cis,
                         "val": rule_metrics(sa_v, c_v, {"none": np.zeros(len(val_rows), bool),
                                                         "truncated": Xv[:, feats.index("truncated")] > 0.5,
                                                         "tokens": tok_v >= thr["tokens"]}.get(
                                                             rule, s_v >= (thr.get(rule) or np.inf)))}

    # Paired bootstrap: reliability score vs truncation rule
    best_rule = "score_f1" if has_unanswerable else "score_safe"
    key = "f1" if has_unanswerable else "unsafe_rate"
    diff = lambda i: (rule_metrics(sa_t[i], c_t[i], abstain_vec(best_rule, i))[key]
                      - rule_metrics(sa_t[i], c_t[i], abstain_vec("truncated", i))[key])
    rng = np.random.default_rng(SEED)
    diffs = [diff(rng.integers(0, len(test_rows), len(test_rows))) for _ in range(N_BOOT)]
    paired = {"rule": best_rule, "metric": key,
              "diff": results[best_rule][key] - results["truncated"][key],
              "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))}

    # Threshold-free signal quality (risk label), calibration of P(answer is safe)
    signals = {"raw confidence": 1 - conf_t, "generated tokens": tok_t, "reliability score": s_t}
    signal_q = {}
    for sname, sc in signals.items():
        cov, risk, aurc = risk_coverage(sc, yt)
        signal_q[sname] = {"auroc": auroc(sc, yt),
                           "auroc_ci": boot_ci(lambda i, s=sc: auroc(s[i], yt[i]), len(yt)),
                           "aurc": aurc, "curve": (cov.tolist(), risk.tolist())}
    ok_t = 1 - yt
    calib = {"raw confidence": {"ece": ece(conf_t, ok_t), "brier": brier(conf_t, ok_t)},
             "reliability score": {"ece": ece(1 - s_t, ok_t), "brier": brier(1 - s_t, ok_t)}}

    # Accuracy-coverage table: thresholds set on validation for fixed target coverages
    fixed = []
    for cov in FIXED_COVERAGES:
        t = threshold_for_coverage(s_v, sa_v, cov)
        m = rule_metrics(sa_t, c_t, s_t >= t if np.isfinite(t) else np.zeros(len(s_t), bool))
        ci = boot_ci(lambda i, tt=t: rule_metrics(sa_t[i], c_t[i], (s_t[i] >= tt) if np.isfinite(tt)
                                                  else np.zeros(len(i), bool))["acc_answering"], len(s_t))
        fixed.append({"target_coverage": cov, "threshold": t, "test_coverage": m["coverage"],
                      "test_acc_answering": m["acc_answering"], "ci": ci, "unsafe_rate": m["unsafe_rate"]})

    return {"name": name, "features": feats, "model": model, "thresholds": thr, "fixed_coverage": fixed,
            "target": target,
            "n_val": len(val_rows), "n_test": len(test_rows), "results": results,
            "paired": paired, "signals": signal_q, "calibration": calib,
            "_plot": {"conf": conf_t, "score": s_t, "ok": ok_t}}


# ── Report ─────────────────────────────────────────────────────────────────────

RULE_LABEL = {"none": "Always answer", "truncated": "Abstain if truncated (pre-specified)",
              "tokens": "Abstain if tokens >= t (t from val)",
              "score_f1": "**Reliability score, max-F1 τ (from val)**",
              "score_safe": "**Reliability score, safe τ (val target acc, coverage ≥ 50%)**"}


def pct(x):
    return f"{100 * x:.1f}%"


def ci_pct(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def section(a, has_unanswerable):
    L = [f"## {a['name']}\n",
         f"Fitted on validation (n={a['n_val']}), evaluated once on test (n={a['n_test']}). "
         f"Features: {', '.join(a['features'])}.\n"]
    if has_unanswerable:
        L += ["| Rule | Abstain F1 [95% CI] | Unanswerable recall [CI] | Coverage [CI] | "
              "Acc when answering [CI] | Unsafe answers [CI] |", "|---|---|---|---|---|---|"]
    else:
        L += ["| Rule | Coverage [CI] | Acc when answering [CI] | Unsafe answers [CI] |", "|---|---|---|---|"]
    for rule, m in a["results"].items():
        ci = m["ci"]
        if has_unanswerable:
            L.append(f"| {RULE_LABEL[rule]} | {m['f1']:.3f} [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}] | "
                     f"{pct(m['recall'])} {ci_pct(ci['recall'])} | {pct(m['coverage'])} {ci_pct(ci['coverage'])} | "
                     f"{pct(m['acc_answering'])} {ci_pct(ci['acc_answering'])} | "
                     f"{pct(m['unsafe_rate'])} {ci_pct(ci['unsafe_rate'])} |")
        else:
            L.append(f"| {RULE_LABEL[rule]} | {pct(m['coverage'])} {ci_pct(ci['coverage'])} | "
                     f"{pct(m['acc_answering'])} {ci_pct(ci['acc_answering'])} | "
                     f"{pct(m['unsafe_rate'])} {ci_pct(ci['unsafe_rate'])} |")
    p = a["paired"]
    if p["metric"] == "f1":
        dtxt = f"abstention F1 {p['diff']:+.3f} [95% CI {p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"
    else:
        dtxt = (f"unsafe-answer rate {100 * p['diff']:+.1f} pts "
                f"[95% CI {100 * p['ci'][0]:+.1f}, {100 * p['ci'][1]:+.1f}]")
    L += ["", f"Safe target for this task: {a['target']:.0%} accuracy when answering (validation).", "",
          f"Paired bootstrap, {RULE_LABEL[p['rule']].strip('*')} minus truncation rule: {dtxt}.", "",
          "**Accuracy vs coverage (reliability score; threshold set on validation for each target coverage)**", "",
          "| Target coverage (val) | Test coverage | Test accuracy when answering [95% CI] | Unsafe answers |",
          "|---|---|---|---|"]
    for f in a["fixed_coverage"]:
        L.append(f"| {f['target_coverage']:.0%} | {pct(f['test_coverage'])} | {pct(f['test_acc_answering'])} "
                 f"{ci_pct(f['ci'])} | {pct(f['unsafe_rate'])} |")
    L += ["",
          "**Signal quality (threshold-free, test; target = unsafe to answer)**", "",
          "| Signal | AUROC [95% CI] | AURC (lower is better) |", "|---|---|---|"]
    for s, q in a["signals"].items():
        L.append(f"| {s} | {q['auroc']:.3f} [{q['auroc_ci'][0]:.3f}, {q['auroc_ci'][1]:.3f}] | {q['aurc']:.3f} |")
    L += ["", "**Calibration of P(answer is safe), test**", "", "| Signal | ECE | Brier |", "|---|---|---|"]
    for s, c in a["calibration"].items():
        L.append(f"| {s} | {c['ece']:.3f} | {c['brier']:.3f} |")
    L += ["", f"Thresholds (validation): " + ", ".join(
        f"{k}={v:.4f}" for k, v in a["thresholds"].items() if v is not None and np.isfinite(v)), ""]
    return L


def plot(analyses, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(analyses) + 1, figsize=(5.2 * (len(analyses) + 1), 4.4))
    for ax, a in zip(axes, analyses):
        for sname, q in a["signals"].items():
            cov, risk = q["curve"]
            ax.plot(cov, risk, label=f"{sname} (AURC {q['aurc']:.3f})")
        ax.set_xlabel("Coverage (fraction answered)")
        ax.set_ylabel("Unsafe-answer rate among answered")
        ax.set_title(f"Risk–coverage: {a['name']}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[-1]
    a = analyses[0]
    bins = np.linspace(0, 1, 11)
    for lab, prob in (("raw confidence", a["_plot"]["conf"]), ("reliability score", 1 - a["_plot"]["score"])):
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (prob >= lo) & (prob <= hi if hi == 1 else prob < hi)
            if m.sum() >= 5:
                xs.append(prob[m].mean()); ys.append(a["_plot"]["ok"][m].mean())
        ax.plot(xs, ys, "o-", label=f"{lab} (ECE {a['calibration'][lab]['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Predicted P(answer is safe)")
    ax.set_ylabel("Observed fraction safe")
    ax.set_title(f"Calibration: {a['name']}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig13_reliability.{ext}", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--val-baseline", default="outputs/kaggle_qa/medpsy_val/work/phase5/medpsy-4b.jsonl")
    ap.add_argument("--val-context", default="outputs/kaggle_qa/medpsy_val/work/phase10/medpsy-4b.jsonl")
    ap.add_argument("--test-baseline", default="outputs/kaggle_qa/full/medpsy-4b/work/phase5/medpsy-4b.jsonl")
    ap.add_argument("--test-context", default="outputs/kaggle_qa/full/medpsy-4b/work/phase10/medpsy-4b.jsonl")
    ap.add_argument("--evidence", default="outputs/kaggle_build/work/phase7/evidence.jsonl",
                    help="Phase 7 evidence (for the top rerank-score feature); skipped if missing")
    ap.add_argument("--no-rerank-feature", action="store_true")
    ap.add_argument("--out-dir", default="outputs/analysis")
    ap.add_argument("--fig-dir", default="outputs/figures")
    args = ap.parse_args()

    vb = load_rows(resolve(args.val_baseline, "phase5", "validation"), "validation")
    tb = load_rows(resolve(args.test_baseline, "phase5", "test"), "test")
    vc = load_rows(resolve(args.val_context, "phase10", "validation"), "validation")
    tc = load_rows(resolve(args.test_context, "phase10", "test"), "test")
    print(f"Loaded val baseline {len(vb)}, test baseline {len(tb)}, val context {len(vc)}, test context {len(tc)}")

    # Integrity checks (Rule 3) and no val/test overlap
    assert {r["id"] for r in vb}.isdisjoint({r["id"] for r in tb}), "[CHECK FAILED] val/test overlap (baseline)"
    assert {r["id"] for r in vc}.isdisjoint({r["id"] for r in tc}), "[CHECK FAILED] val/test overlap (context)"
    assert any(r.get("should_abstain") for r in vb), "[CHECK FAILED] no unanswerable rows in val baseline"
    for rows, lab in ((vb, "val baseline"), (tb, "test baseline"), (vc, "val context"), (tc, "test context")):
        assert all("gen_tokens" in r and "letter_probs" in r for r in rows), f"[CHECK FAILED] fields missing in {lab}"

    evidence = None
    if not args.no_rerank_feature and Path(args.evidence).exists():
        evidence = {r["id"]: r for r in read_jsonl(args.evidence)}
        missing = [r["id"] for r in vb + tb if r["id"] not in evidence]
        if missing:
            print(f"WARNING: {len(missing)} questions lack evidence; dropping the rerank feature")
            evidence = None
    print("Rerank feature:", "ON" if evidence else "OFF")

    a_mq = analyse("MedQA + unanswerable (no retrieval in prompt)", vb, tb, evidence, has_unanswerable=True,
                   target=SAFE_TARGET["medqa"])
    a_pm = analyse("PubMedQA + abstract", vc, tc, None, has_unanswerable=False, target=SAFE_TARGET["pubmedqa"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L = ["# MedPsy-4B calibrated reliability score and abstention threshold\n",
         "**Protocol.** Logistic-regression weights, feature scaling and every threshold were fitted on the "
         "validation split only; the frozen rule was applied once to the held-out test split. "
         f"Ridge penalty ({L2}), safe targets (MedQA {SAFE_TARGET['medqa']:.0%}, PubMedQA "
         f"{SAFE_TARGET['pubmedqa']:.0%} accuracy when answering) and the {MIN_COVERAGE:.0%} minimum coverage "
         "were fixed in advance. CIs are 95% percentile bootstrap (1,000 resamples, seed 42). "
         "*Unsafe answers* = questions the system answered that were wrong or unanswerable, over all questions.\n"]
    L += section(a_mq, True) + section(a_pm, False)
    (out / "medpsy_reliability.md").write_text("\n".join(L))

    def strip(a):
        d = {k: v for k, v in a.items() if k != "_plot"}
        d["signals"] = {s: {k: v for k, v in q.items() if k != "curve"} for s, q in a["signals"].items()}
        return d
    (out / "medpsy_reliability.json").write_text(json.dumps([strip(a_mq), strip(a_pm)], indent=2, default=float))
    frozen = {"token_cap": TOKEN_CAP, "safe_target": SAFE_TARGET, "min_coverage": MIN_COVERAGE,
              "medqa": {"features": a_mq["features"], **a_mq["model"], "thresholds": a_mq["thresholds"]},
              "pubmedqa_context": {"features": a_pm["features"], **a_pm["model"], "thresholds": a_pm["thresholds"]}}
    (out / "medpsy_reliability_model.json").write_text(json.dumps(frozen, indent=2, default=float))
    plot([a_mq, a_pm], Path(args.fig_dir))

    print("\n".join(L))
    print(f"\nWrote {out/'medpsy_reliability.md'}, medpsy_reliability.json, medpsy_reliability_model.json, "
          f"{Path(args.fig_dir)/'fig13_reliability.png'}")


if __name__ == "__main__":
    main()
