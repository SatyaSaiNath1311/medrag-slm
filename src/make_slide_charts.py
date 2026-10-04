"""Presentation charts built directly from the result files (no hard-coded numbers).

Every bar carries its exact value above it; labels never overlap bars, legends sit outside
the plotting area, and each chart is a separate figure (no merged panels).

Inputs (all produced by earlier phases):
  outputs/analysis/final_results.csv                 accuracy and latency for the 6 models
  outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl   MedPsy-4B + RAG (optional; shown as
                                                     "not run" until the RAG_MEDPSY run is downloaded)
  outputs/analysis/medpsy_reliability.json           reliability score results
  outputs/analysis/phase9_abstention.json            RAG + abstain for the 5 standard models
  outputs/analysis/profiling.json                    peak GPU memory for the 5 profiled models

Outputs: outputs/figures/slides/chart*.png (300 dpi) and .svg

Run:  python -m src.make_slide_charts
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = ["qwen3-1.7b", "smollm3-3b", "phi4-mini", "qwen3-4b", "gemma3-4b", "medpsy-4b"]
LABEL = {"qwen3-1.7b": "Qwen3-1.7B", "smollm3-3b": "SmolLM3-3B", "phi4-mini": "Phi-4-mini",
         "qwen3-4b": "Qwen3-4B", "gemma3-4b": "Gemma3-4B", "medpsy-4b": "MedPsy-4B"}
RETRIEVAL_OVERHEAD_S = 0.127   # BM25 + dense + rerank per question, measured in the Phase 6/7 build logs
BLUE, ORANGE, GREEN, GREY = "#2E6FB7", "#E8892B", "#2E9E5B", "#8C8C8C"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 16,
                     "axes.titleweight": "bold", "axes.labelsize": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.3})


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_main(path):
    with open(path, newline="") as fh:
        rows = {r["model"].strip(): r for r in csv.DictReader(fh)}
    missing = [m for m in MODELS if m not in rows]
    assert not missing, f"[CHECK FAILED] models missing from {path}: {missing}"
    return rows


def load_medpsy_rag_rows(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("split") == "test" and r.get("dataset") == "medqa"]
    return rows if len(rows) == 500 else None


def load_medpsy_rag(path):
    p = Path(path)
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("split") == "test" and r.get("dataset") == "medqa"]
    if len(rows) != 500:
        print(f"WARNING: MedPsy RAG file has {len(rows)} MedQA test rows (expected 500); not plotted")
        return None
    return sum(bool(r["correct"]) for r in rows) / len(rows)


def label_bars(ax, bars, values, fmt, pad=4, size=12):
    for bar, v in zip(bars, values):
        if v is None:
            continue
        ax.annotate(fmt(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, pad), textcoords="offset points", ha="center", va="bottom",
                    fontsize=size, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def mark_missing(ax, x, ymax, text="not run"):
    ax.annotate(text, (x, ymax * 0.03), ha="center", va="bottom", fontsize=11, color=GREY, style="italic")


def save(fig, out_dir, name):
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_dir / name}.png")


def chart_medqa(rows, medpsy_rag, out_dir):
    base = [fnum(rows[m]["baseline_acc_medqa"]) for m in MODELS]
    rag = [fnum(rows[m]["rag_acc_medqa"]) for m in MODELS]
    rag[MODELS.index("medpsy-4b")] = medpsy_rag
    base = [v * 100 if v is not None else None for v in base]
    rag = [v * 100 if v is not None else None for v in rag]
    x, w = np.arange(len(MODELS)), 0.38
    fig, ax = plt.subplots(figsize=(13, 6.5))
    b1 = ax.bar(x - w / 2 - 0.02, [v or 0 for v in base], w, label="Baseline (no retrieval)", color=BLUE)
    b2 = ax.bar(x + w / 2 + 0.02, [v or 0 for v in rag], w, label="With RAG (top-5 evidence)", color=ORANGE)
    label_bars(ax, b1, base, lambda v: f"{v:.1f}%")
    label_bars(ax, b2, rag, lambda v: f"{v:.1f}%")
    for i, v in enumerate(rag):
        if v is None:
            mark_missing(ax, x[i] + w / 2 + 0.02, 100)
    ax.set_xticks(x, [LABEL[m] for m in MODELS])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MedQA accuracy: baseline vs RAG (test, n = 500)", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir, "chart1_medqa_baseline_vs_rag")


def chart_pubmedqa(rows, out_dir):
    vals = [fnum(rows[m]["context_acc_pubmedqa"]) * 100 for m in MODELS]
    colors = [GREEN if m == "medpsy-4b" else BLUE for m in MODELS]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(np.arange(len(MODELS)), vals, 0.6, color=colors)
    label_bars(ax, bars, vals, lambda v: f"{v:.1f}%")
    ax.set_xticks(np.arange(len(MODELS)), [LABEL[m] for m in MODELS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("PubMedQA accuracy with the abstract provided (test, n = 500)", pad=20)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir, "chart2_pubmedqa_with_abstract")


def chart_latency(rows, out_dir):
    vals = [fnum(rows[m]["baseline_sec_per_q"]) for m in MODELS]
    colors = [GREEN if m == "medpsy-4b" else BLUE for m in MODELS]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(np.arange(len(MODELS)), vals, 0.6, color=colors)
    label_bars(ax, bars, vals, lambda v: f"{v:.2f} s" if v < 10 else f"{v:.1f} s")
    ax.set_yscale("log")
    ax.set_ylim(0.05, max(vals) * 4)
    ax.set_xticks(np.arange(len(MODELS)), [LABEL[m] for m in MODELS])
    ax.set_ylabel("Seconds per question (log scale)")
    ax.set_title("Latency per question without retrieval", pad=20)
    ax.grid(axis="y", alpha=0.25, which="major")
    save(fig, out_dir, "chart3_latency")


def chart_safety(rel, out_dir):
    res = rel["results"]
    rules = [("none", "Always answer"), ("truncated", "Old rule\n(truncation)"), ("score_f1", "Reliability\nscore")]
    acc = [res[k]["acc_answering"] * 100 for k, _ in rules]
    unsafe = [res[k]["unsafe_rate"] * 100 for k, _ in rules]
    x, w = np.arange(len(rules)), 0.36
    fig, ax = plt.subplots(figsize=(11, 6.5))
    b1 = ax.bar(x - w / 2 - 0.02, acc, w, label="Accuracy when answering", color=GREEN)
    b2 = ax.bar(x + w / 2 + 0.02, unsafe, w, label="Unsafe answers (wrong or unanswerable)", color=ORANGE)
    label_bars(ax, b1, acc, lambda v: f"{v:.1f}%")
    label_bars(ax, b2, unsafe, lambda v: f"{v:.1f}%")
    ax.axhline(90, color=GREY, ls="--", lw=1, zorder=0)
    ax.annotate("90% target", (x[-1] + 0.62, 90), va="center", fontsize=11, color=GREY, annotation_clip=False)
    ax.set_xticks(x, [lbl for _, lbl in rules])
    ax.set_ylim(0, 110)
    ax.set_ylabel("%")
    ax.set_title(f"MedPsy-4B safety on MedQA + unanswerable (test, n = {rel['n_test']})", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir, "chart4_medpsy_safety")


def chart_coverage(rel, out_dir):
    pts = rel["fixed_coverage"]
    cov = [p["test_coverage"] * 100 for p in pts]
    acc = [p["test_acc_answering"] * 100 for p in pts]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(cov, acc, "-o", color=BLUE, lw=2.5, ms=9)
    for c, a in zip(cov, acc):
        ax.annotate(f"{a:.1f}%", (c, a), xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=12, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.axhline(90, color=GREY, ls="--", lw=1, zorder=0)
    ax.annotate("90% target", (min(cov) - 1, 90.4), fontsize=11, color=GREY)
    ax.invert_xaxis()
    ax.set_xlim(max(cov) + 3, min(cov) - 3)
    ax.set_ylim(min(acc) - 5, 101)
    ax.set_xlabel("Coverage: share of answerable questions the model answers (%)")
    ax.set_ylabel("Accuracy when answering (%)")
    ax.set_title("MedPsy-4B accuracy vs coverage (thresholds set on validation, test shown)", pad=20)
    ax.grid(alpha=0.25)
    save(fig, out_dir, "chart5_accuracy_vs_coverage")


def chart_calibration(rel, out_dir):
    names = ["Raw confidence", "Reliability score"]
    ece = [rel["calibration"]["raw confidence"]["ece"], rel["calibration"]["reliability score"]["ece"]]
    auroc = [rel["signals"]["raw confidence"]["auroc"], rel["signals"]["reliability score"]["auroc"]]
    for vals, name, title, ylim, fname in (
            (ece, "ECE (lower is better)", "Calibration error: raw confidence vs reliability score", 0.4,
             "chart6a_calibration_ece"),
            (auroc, "AUROC (higher is better)", "Detecting unsafe answers: raw confidence vs reliability score",
             1.0, "chart6b_auroc")):
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar([0, 1], vals, 0.5, color=[GREY, GREEN])
        label_bars(ax, bars, vals, lambda v: f"{v:.3f}")
        ax.set_xticks([0, 1], names)
        ax.set_ylim(0, ylim * 1.12)
        ax.set_ylabel(name)
        ax.set_title(title, pad=20, fontsize=14)
        ax.grid(axis="y", alpha=0.25)
        save(fig, out_dir, fname)


def chart_three_strategies(rows, medpsy_rag_rows, phase9_path, out_dir):
    """Baseline vs RAG vs RAG + Abstain for all six models, coverage printed inside the abstain bar."""
    p9 = json.loads(Path(phase9_path).read_text())["models"]
    base = [fnum(rows[m]["baseline_acc_medqa"]) for m in MODELS]
    rag = [fnum(rows[m]["rag_acc_medqa"]) for m in MODELS]
    abst, cov = [], []
    for m in MODELS:
        if m == "medpsy-4b":
            if medpsy_rag_rows is None:
                abst.append(None); cov.append(None); continue
            ans = [r for r in medpsy_rag_rows if not r.get("truncated")]
            abst.append(sum(bool(r["correct"]) for r in ans) / len(ans))
            cov.append(len(ans) / len(medpsy_rag_rows))
            rag[MODELS.index(m)] = sum(bool(r["correct"]) for r in medpsy_rag_rows) / len(medpsy_rag_rows)
        else:
            e = next(x for x in p9 if x["model"] == m and x["mode"] == "rag")["fixed_coverage_scaled"]["cov80"]
            abst.append(e["selective_accuracy"]); cov.append(e["coverage_achieved_pct"] / 100)
    pct = lambda vals: [v * 100 if v is not None else None for v in vals]
    base, rag, abst = pct(base), pct(rag), pct(abst)

    x, w = np.arange(len(MODELS)), 0.27
    fig, ax = plt.subplots(figsize=(14, 7))
    b1 = ax.bar(x - w - 0.02, [v or 0 for v in base], w, label="Baseline", color=BLUE)
    b2 = ax.bar(x, [v or 0 for v in rag], w, label="RAG", color=ORANGE)
    b3 = ax.bar(x + w + 0.02, [v or 0 for v in abst], w, label="RAG + Abstain (accuracy when answering)", color=GREEN)
    for bars, vals in ((b1, base), (b2, rag), (b3, abst)):
        label_bars(ax, bars, vals, lambda v: f"{v:.1f}", size=11)
    for bar, c in zip(b3, cov):
        if c is not None:
            ax.annotate(f"cov\n{100 * c:.0f}%", (bar.get_x() + bar.get_width() / 2, 3), ha="center",
                        va="bottom", fontsize=9.5, color="white", fontweight="bold")
    ax.set_xticks(x, [LABEL[m] for m in MODELS])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Six models under three strategies", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.01, -0.04,
             "Baseline and RAG: MedQA test (n = 500). RAG + Abstain, standard models: validation-set confidence "
             "threshold for ~80% coverage, full test set (n = 1,150).\n"
             "RAG + Abstain, MedPsy-4B: abstain when the reasoning budget is exhausted, MedQA test (n = 500). "
             "cov = share of questions answered.", fontsize=10, color="#444444", ha="left", va="top")
    save(fig, out_dir, "chart7_three_strategies")


def chart_latency_modes(rows, medpsy_rag_rows, out_dir):
    """Latency per question, baseline vs RAG, for all six models (log scale, every bar labelled)."""
    base = [fnum(rows[m]["baseline_sec_per_q"]) for m in MODELS]
    rag = [fnum(rows[m]["rag_sec_per_q"]) for m in MODELS]
    if medpsy_rag_rows is not None:
        secs = [r["seconds"] for r in medpsy_rag_rows if r.get("seconds") is not None]
        rag[MODELS.index("medpsy-4b")] = sum(secs) / len(secs) if secs else None
    rag = [v + RETRIEVAL_OVERHEAD_S if v is not None else None for v in rag]
    fmt = lambda v: f"{v:.2f} s" if v < 10 else f"{v:.1f} s"
    x, w = np.arange(len(MODELS)), 0.38
    fig, ax = plt.subplots(figsize=(13, 6.5))
    b1 = ax.bar(x - w / 2 - 0.02, [v or 0 for v in base], w, label="Baseline", color=BLUE)
    b2 = ax.bar(x + w / 2 + 0.02, [v or 1e-9 for v in rag], w, label="With RAG", color=ORANGE)
    label_bars(ax, b1, base, fmt, size=11)
    label_bars(ax, b2, rag, fmt, size=11)
    ax.set_yscale("log")
    ax.set_ylim(0.05, max(v for v in base + rag if v) * 5)
    ax.set_xticks(x, [LABEL[m] for m in MODELS])
    ax.set_ylabel("Seconds per question (log scale)")
    ax.set_title("Latency per question: baseline vs RAG", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.01, -0.03, "RAG latency includes 0.127 s retrieval per question. MedPsy-4B reasons step by step before "
             "answering; the other five models answer directly.",
             fontsize=10, color="#444444", ha="left", va="top")
    save(fig, out_dir, "chart8_latency_baseline_vs_rag")


def chart_vram(profiling_path, out_dir):
    """Peak GPU memory (allocated, summed over GPUs), baseline vs RAG, five profiled models."""
    prof = json.loads(Path(profiling_path).read_text())
    models = [m for m in MODELS if m in prof]

    def peak_gb(m, mode):
        cm = prof[m]["modes"].get(mode, {}).get("cuda_memory", {})
        return sum(v.get("max_memory_allocated_mb", 0) for v in cm.values()) / 1024 if cm else None

    base = [peak_gb(m, "baseline") for m in models]
    rag = [peak_gb(m, "rag") for m in models]
    x, w = np.arange(len(models)), 0.38
    fig, ax = plt.subplots(figsize=(12, 6.5))
    b1 = ax.bar(x - w / 2 - 0.02, [v or 0 for v in base], w, label="Baseline", color=BLUE)
    b2 = ax.bar(x + w / 2 + 0.02, [v or 0 for v in rag], w, label="With RAG", color=ORANGE)
    label_bars(ax, b1, base, lambda v: f"{v:.1f} GB", size=11)
    label_bars(ax, b2, rag, lambda v: f"{v:.1f} GB", size=11)
    ax.axhline(16, color="#C0392B", ls="--", lw=1.2, zorder=0)
    ax.annotate("Single 16 GB T4 limit", (x[0] - 0.45, 16.4), fontsize=11, color="#C0392B")
    ax.set_xticks(x, [LABEL[m] + ("*" if m == "gemma3-4b" else "") for m in models])
    ax.set_ylim(0, 23)
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("Peak GPU memory: baseline vs RAG", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.01, -0.03, "* Gemma3-4B loaded in float32 across two T4 GPUs to avoid numerical overflow. "
             "MedPsy-4B was not memory-profiled; it runs on a single 16 GB T4 in the chatbot.",
             fontsize=10, color="#444444", ha="left", va="top")
    save(fig, out_dir, "chart9_gpu_memory")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--main", default="outputs/analysis/final_results.csv")
    ap.add_argument("--medpsy-rag", default="outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl")
    ap.add_argument("--reliability", default="outputs/analysis/medpsy_reliability.json")
    ap.add_argument("--profiling", default="outputs/analysis/profiling.json")
    ap.add_argument("--phase9", default="outputs/analysis/phase9_abstention.json")
    ap.add_argument("--out-dir", default="outputs/figures/slides")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_main(args.main)
    medpsy_rag = load_medpsy_rag(args.medpsy_rag)
    print("MedPsy-4B + RAG:", f"{100 * medpsy_rag:.1f}%" if medpsy_rag is not None else "not available yet")
    rel = json.loads(Path(args.reliability).read_text())[0]

    chart_medqa(rows, medpsy_rag, out)
    chart_pubmedqa(rows, out)
    chart_latency(rows, out)
    chart_safety(rel, out)
    chart_coverage(rel, out)
    chart_calibration(rel, out)
    medpsy_rag_rows = load_medpsy_rag_rows(args.medpsy_rag)
    chart_three_strategies(rows, medpsy_rag_rows, args.phase9, out)
    chart_latency_modes(rows, medpsy_rag_rows, out)
    chart_vram(args.profiling, out)
    print(f"Done. Charts in {out}/")


if __name__ == "__main__":
    main()
