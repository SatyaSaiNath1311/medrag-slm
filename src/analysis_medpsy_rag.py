"""MedPsy-4B: baseline vs RAG on the MedQA test split (paired, same 500 questions).

Reports accuracy with bootstrap 95% CI, exact McNemar test, truncation rates, accuracy when
answering under the abstain-if-truncated rule, and seconds per question.

Run:  python -m src.analysis_medpsy_rag
Writes outputs/analysis/medpsy_rag.md
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def read(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def boot_ci(x, n_boot=1000, seed=42):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="outputs/kaggle_qa/full/medpsy-4b/work/phase5/medpsy-4b.jsonl")
    ap.add_argument("--rag", default="outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl")
    ap.add_argument("--out", default="outputs/analysis/medpsy_rag.md")
    args = ap.parse_args()

    base = {r["id"]: r for r in read(args.baseline) if r.get("split") == "test" and r.get("dataset") == "medqa"}
    rag = {r["id"]: r for r in read(args.rag) if r.get("split") == "test" and r.get("dataset") == "medqa"}
    ids = sorted(set(base) & set(rag))
    assert len(ids) == len(rag) == 500, f"[CHECK FAILED] expected 500 paired MedQA questions, got {len(ids)} (rag {len(rag)})"

    cb = np.array([bool(base[i]["correct"]) for i in ids])
    cr = np.array([bool(rag[i]["correct"]) for i in ids])
    tb = np.array([bool(base[i].get("truncated")) for i in ids])
    tr = np.array([bool(rag[i].get("truncated")) for i in ids])
    b = int((cr & ~cb).sum())   # RAG right, baseline wrong
    c = int((cb & ~cr).sum())   # baseline right, RAG wrong
    p = mcnemar_exact(b, c)

    def answered_acc(correct, trunc):
        a = ~trunc
        return correct[a].mean() if a.any() else float("nan"), a.mean()

    ab, cov_b = answered_acc(cb, tb)
    ar, cov_r = answered_acc(cr, tr)

    sb, sr = [base[i].get("seconds") for i in ids], [rag[i].get("seconds") for i in ids]
    sb = [s for s in sb if s is not None]
    sr = [s for s in sr if s is not None]

    L = ["# MedPsy-4B: baseline vs RAG on MedQA test (n = 500, paired)\n",
         "| Mode | Accuracy [95% CI] | Truncated | Acc. when answering (abstain if truncated) | Coverage | s/question |",
         "|---|---|---|---|---|---|"]
    for name, cor, tru, acc_a, cov, secs in (("Baseline (no RAG)", cb, tb, ab, cov_b, sb),
                                              ("RAG (top-5 evidence)", cr, tr, ar, cov_r, sr)):
        lo, hi = boot_ci(cor)
        sec_txt = f"{np.mean(secs):.1f}" if secs else "n/a"
        L.append(f"| {name} | {100*cor.mean():.1f}% [{100*lo:.1f}, {100*hi:.1f}] | {100*tru.mean():.1f}% | "
                 f"{100*acc_a:.1f}% | {100*cov:.1f}% | {sec_txt} |")
    L += ["", f"Exact McNemar (paired): RAG-only correct = {b}, baseline-only correct = {c}, p = {p:.3g}.",
          f"Accuracy change: {100*(cr.mean()-cb.mean()):+.1f} percentage points."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
