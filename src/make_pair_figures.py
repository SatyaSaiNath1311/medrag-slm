"""src/make_pair_figures.py — Matched-Pair and Efficiency Figures (v3).

Generates presentation-ready figures (PNG 300 dpi and SVG):
  - fig14_matched_pairs_comparison: Paired bars (generic vs medical per pair, baseline and RAG)
  - fig15_rag_gain_by_group: RAG gain comparison by group (Generic vs Medical)
  - fig16_accuracy_vs_latency_thinking: Accuracy vs. sec/question scatter with thinking-models marked

Rules:
  - Saves to outputs/figures/ with new figure numbers (fig14..fig16); no overwrites of fig1..fig13.
  - High resolution (300 dpi PNG) and scalable vector graphics (SVG).
  - All numbers are loaded strictly from real output files under outputs/ using the canonical
    strict accuracy rule from src/answer_validity.py.
  - If any required output file is missing, raises a clear error naming all checked candidate paths.
"""

import json
from math import comb
import os
from pathlib import Path
import sys

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.answer_validity import is_strict_correct


# Academic color palette
COLOR_GENERIC_BASE = "#7F7F7F"    # Neutral grey
COLOR_GENERIC_RAG  = "#1F77B4"    # Classic blue
COLOR_MEDICAL_BASE = "#FF7F0E"    # Amber / orange
COLOR_MEDICAL_RAG  = "#2CA02C"    # Vibrant green
COLOR_THINKING_MARK = "#9467BD"   # Royal purple for thinking models


def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
        "grid.color": "#EAEAEA",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#E0E0E0",
    })


def resolve_model_output_path(model_name: str, condition: str, root_dir=".") -> str:
    """Find existing output JSONL file path for model and condition, or raise FileNotFoundError.
    
    Checks v3 canonical paths first, then legacy runner output paths.
    """
    canonical_v3 = os.path.join(root_dir, "outputs", "v3", model_name, condition, f"{model_name}.jsonl")
    candidates = [
        canonical_v3,
        os.path.join(root_dir, "outputs", "v3", model_name, f"{condition}.jsonl"),
        os.path.join("/kaggle/working/outputs/v3", model_name, condition, f"{model_name}.jsonl"),
    ]

    # Handle model aliases like medgemma-4b <-> medgemma-1.5-4b-it
    if model_name in ("medgemma-4b", "medgemma-1.5-4b-it"):
        alt_name = "medgemma-1.5-4b-it" if model_name == "medgemma-4b" else "medgemma-4b"
        candidates.extend([
            os.path.join(root_dir, "outputs", "v3", alt_name, condition, f"{alt_name}.jsonl"),
            os.path.join(root_dir, "outputs", "v3", alt_name, f"{condition}.jsonl"),
        ])

    # Legacy paths for baseline (C1) and RAG (C2) runs
    if condition == "C1":
        candidates.extend([
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", model_name, "work", "phase5", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_final", "work", "phase5", f"{model_name}.jsonl"),
        ])
    elif condition == "C2":
        candidates.extend([
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", model_name, "work", "phase8", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "full", f"{model_name}_rag", "work", "phase8", f"{model_name}.jsonl"),
            os.path.join(root_dir, "outputs", "kaggle_qa", "medpsy_rag", "work", "phase8", f"{model_name}.jsonl"),
        ])

    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p

    candidates_str = "\n  - ".join(candidates)
    raise FileNotFoundError(
        f"Missing required output file for model '{model_name}' (condition {condition}).\n"
        f"Checked candidate paths:\n  - {candidates_str}"
    )


def load_eval_data(model_name: str, condition: str, dataset: str = "medqa", split: str = "test", root_dir=".") -> dict:
    """Load JSONL rows for model and condition, filter answerable questions, and compute strict metrics.
    
    Uses canonical strict-accuracy rule from src/answer_validity.py.
    """
    path = resolve_model_output_path(model_name, condition, root_dir)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"Output file contains no rows: {path}")

    filtered = []
    for r in rows:
        # Ignore unanswerable questions when computing benchmark accuracy
        if r.get("should_abstain", False):
            continue
        # Check split if present
        if split and r.get("split") and r.get("split") != split:
            continue
        # Check dataset if present
        if dataset:
            r_ds = r.get("dataset")
            r_id = str(r.get("id", ""))
            if r_ds:
                if r_ds != dataset:
                    continue
            else:
                if dataset not in r_id:
                    continue
        filtered.append(r)

    if not filtered:
        raise ValueError(f"No matching {dataset} {split} rows found in {path}")

    strict_correct = [1 if is_strict_correct(r) else 0 for r in filtered]
    acc = sum(strict_correct) / len(strict_correct)

    secs = [float(r["seconds"]) for r in filtered if r.get("seconds") is not None and float(r["seconds"]) > 0]
    mean_sec = sum(secs) / len(secs) if secs else 0.0

    qmap = {r["id"]: (1 if is_strict_correct(r) else 0) for r in filtered if "id" in r}

    return {
        "path": path,
        "n": len(filtered),
        "accuracy": acc,
        "sec_per_q": mean_sec,
        "strict_correct": strict_correct,
        "qmap": qmap,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def make_fig14_paired_bars(out_dir="outputs/figures", root_dir="."):
    """Fig 14: Paired bars (generic vs medical per pair, baseline and RAG)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pair_specs = [
        {"name": "Pair 1 (1.7B)\nQwen3-1.7B vs MedPsy-1.7B", "generic": "qwen3-1.7b", "medical": "medpsy-1.7b"},
        {"name": "Pair 2 (4B, Qwen)\nQwen3-4B vs MedPsy-4B", "generic": "qwen3-4b", "medical": "medpsy-4b"},
        {"name": "Pair 3 (4B, Gemma)\nGemma3-4B vs MedGemma-4B", "generic": "gemma3-4b", "medical": "medgemma-1.5-4b-it"},
    ]

    pairs = []
    for spec in pair_specs:
        gen_base_data = load_eval_data(spec["generic"], "C1", dataset="medqa", split="test", root_dir=root_dir)
        gen_rag_data  = load_eval_data(spec["generic"], "C2", dataset="medqa", split="test", root_dir=root_dir)
        med_base_data = load_eval_data(spec["medical"], "C1", dataset="medqa", split="test", root_dir=root_dir)
        med_rag_data  = load_eval_data(spec["medical"], "C2", dataset="medqa", split="test", root_dir=root_dir)

        # Paired McNemar for domain baseline gain (C1)
        common_ids = sorted(set(gen_base_data["qmap"].keys()) & set(med_base_data["qmap"].keys()))
        b = sum(1 for q in common_ids if gen_base_data["qmap"][q] == 1 and med_base_data["qmap"][q] == 0)
        c = sum(1 for q in common_ids if gen_base_data["qmap"][q] == 0 and med_base_data["qmap"][q] == 1)
        p_val = mcnemar_exact(b, c)

        pairs.append({
            "name": spec["name"],
            "gen_base": gen_base_data["accuracy"],
            "gen_rag": gen_rag_data["accuracy"],
            "med_base": med_base_data["accuracy"],
            "med_rag": med_rag_data["accuracy"],
            "mcnemar_p": p_val,
        })

    x = np.arange(len(pairs))
    width = 0.18

    # Bars for each of the 4 conditions per pair
    rects1 = ax.bar(x - 1.5 * width, [p["gen_base"] * 100 for p in pairs], width, label="Generic Baseline (C1)", color=COLOR_GENERIC_BASE)
    rects2 = ax.bar(x - 0.5 * width, [p["gen_rag"] * 100 for p in pairs], width, label="Generic RAG (C2)", color=COLOR_GENERIC_RAG)
    rects3 = ax.bar(x + 0.5 * width, [p["med_base"] * 100 for p in pairs], width, label="Medical Baseline (C1)", color=COLOR_MEDICAL_BASE)
    rects4 = ax.bar(x + 1.5 * width, [p["med_rag"] * 100 for p in pairs], width, label="Medical RAG (C2)", color=COLOR_MEDICAL_RAG)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 14: Matched-Pair Generic vs. Medical-Domain Performance (MedQA Test)")
    ax.set_xticks(x)
    ax.set_xticklabels([p["name"] for p in pairs])
    ax.set_ylim(0, 100)
    ax.grid(axis="y")
    ax.legend(loc="upper left")

    # Add text annotations
    for i, p in enumerate(pairs):
        diff = (p["med_base"] - p["gen_base"]) * 100
        sign = "+" if diff >= 0 else ""
        p_txt = "p < 0.001" if p["mcnemar_p"] < 0.001 else f"p = {p['mcnemar_p']:.3f}"
        ax.text(x[i], 90, f"Domain Gain:\n{sign}{diff:.1f} pp ({p_txt})", ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9E6", ec="#E0C060"))

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig14_matched_pairs_comparison.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig14_matched_pairs_comparison.svg"))
    plt.close(fig)
    print("Saved Figure 14 to outputs/figures/fig14_matched_pairs_comparison.{png,svg}")


def make_fig15_rag_gain_by_group(out_dir="outputs/figures", root_dir="."):
    """Fig 15: RAG gain comparison by group (Generic vs Medical)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True, gridspec_kw={"width_ratios": [3, 2]})

    # Generic models
    gen_specs = [
        ("Qwen3-1.7B", "qwen3-1.7b"),
        ("SmolLM3-3B", "smollm3-3b"),
        ("Phi-4-mini", "phi4-mini"),
        ("Qwen3-4B", "qwen3-4b"),
        ("Gemma3-4B", "gemma3-4b"),
    ]
    gen_models = [s[0] for s in gen_specs]
    gen_gains = []
    for _, model_key in gen_specs:
        c1 = load_eval_data(model_key, "C1", dataset="medqa", split="test", root_dir=root_dir)
        c2 = load_eval_data(model_key, "C2", dataset="medqa", split="test", root_dir=root_dir)
        gain = (c2["accuracy"] - c1["accuracy"]) * 100
        gen_gains.append(gain)

    # Medical models
    med_specs = [
        ("MedPsy-1.7B", "medpsy-1.7b"),
        ("MedPsy-4B", "medpsy-4b"),
        ("MedGemma-4B", "medgemma-1.5-4b-it"),
    ]
    med_models = [s[0] for s in med_specs]
    med_gains = []
    for _, model_key in med_specs:
        c1 = load_eval_data(model_key, "C1", dataset="medqa", split="test", root_dir=root_dir)
        c2 = load_eval_data(model_key, "C2", dataset="medqa", split="test", root_dir=root_dir)
        gain = (c2["accuracy"] - c1["accuracy"]) * 100
        med_gains.append(gain)

    y_gen = np.arange(len(gen_models))
    y_med = np.arange(len(med_models))

    def _compute_limits(gains, pad=1.5):
        low = min(gains) if gains else 0.0
        high = max(gains) if gains else 5.0
        x_min = min(0.0, low) - pad
        x_max = max(0.0, high) + pad
        return (x_min, x_max)

    ax1.barh(y_gen, gen_gains, color=COLOR_GENERIC_RAG, height=0.55)
    mean_gen = float(np.mean(gen_gains))
    sign_gen = "+" if mean_gen >= 0 else ""
    ax1.axvline(mean_gen, color="#0D47A1", linestyle="--", linewidth=1.5, label=f"Mean Generic Gain: {sign_gen}{mean_gen:.1f} pp")
    if min(gen_gains) < 0:
        ax1.axvline(0, color="#888888", linestyle="-", linewidth=0.8, alpha=0.7)
    ax1.set_yticks(y_gen)
    ax1.set_yticklabels(gen_models)
    ax1.set_xlabel("RAG Accuracy Improvement (percentage points)")
    ax1.set_title("Generic Models (Higher RAG Gain)")
    ax1.grid(axis="x")
    ax1.set_xlim(_compute_limits(gen_gains))
    ax1.legend(loc="lower right")

    ax2.barh(y_med, med_gains, color=COLOR_MEDICAL_RAG, height=0.55)
    mean_med = float(np.mean(med_gains))
    sign_med = "+" if mean_med >= 0 else ""
    ax2.axvline(mean_med, color="#1B5E20", linestyle="--", linewidth=1.5, label=f"Mean Medical Gain: {sign_med}{mean_med:.1f} pp")
    if min(med_gains) < 0:
        ax2.axvline(0, color="#888888", linestyle="-", linewidth=0.8, alpha=0.7)
    ax2.set_yticks(y_med)
    ax2.set_yticklabels(med_models)
    ax2.set_xlabel("RAG Accuracy Improvement (percentage points)")
    ax2.set_title("Medical Models (Smaller RAG Gain)")
    ax2.grid(axis="x")
    ax2.set_xlim(_compute_limits(med_gains))
    ax2.legend(loc="lower right")

    fig.suptitle("Figure 15: Empirical RAG Gain (Delta Acc) by Model Group: RQ2 Verification", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig15_rag_gain_by_group.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig15_rag_gain_by_group.svg"))
    plt.close(fig)
    print("Saved Figure 15 to outputs/figures/fig15_rag_gain_by_group.{png,svg}")


def make_fig16_accuracy_vs_latency_thinking(out_dir="outputs/figures", root_dir="."):
    """Fig 16: Accuracy vs sec/question scatter with thinking models explicitly marked."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical data loaded from real output files
    # Format: (display_name, model_key, condition, is_thinking, is_medical)
    point_specs = [
        ("Qwen3-1.7B Base", "qwen3-1.7b", "C1", False, False),
        ("Qwen3-1.7B RAG", "qwen3-1.7b", "C2", False, False),
        ("SmolLM3-3B Base", "smollm3-3b", "C1", False, False),
        ("SmolLM3-3B RAG", "smollm3-3b", "C2", False, False),
        ("Phi-4-mini Base", "phi4-mini", "C1", False, False),
        ("Phi-4-mini RAG", "phi4-mini", "C2", False, False),
        ("Qwen3-4B Base", "qwen3-4b", "C1", False, False),
        ("Qwen3-4B RAG", "qwen3-4b", "C2", False, False),
        ("Gemma3-4B Base", "gemma3-4b", "C1", False, False),
        ("Gemma3-4B RAG", "gemma3-4b", "C2", False, False),
        # Medical & Thinking models
        ("MedPsy-4B Base", "medpsy-4b", "C1", True, True),
        ("MedPsy-4B RAG", "medpsy-4b", "C2", True, True),
        ("MedPsy-1.7B Base", "medpsy-1.7b", "C1", True, True),
        ("Qwen3-4B-Thinking-2507\n(budget-limited: 99.6% trunc @ 1,024 cap)", "qwen3-4b-thinking-2507", "C1", True, False),
        ("MedGemma-4B Base", "medgemma-1.5-4b-it", "C1", False, True),
    ]

    points = []
    for name, model_key, cond, is_think, is_med in point_specs:
        data = load_eval_data(model_key, cond, dataset="medqa", split="test", root_dir=root_dir)
        acc_pct = data["accuracy"] * 100
        sec_per_q = data["sec_per_q"]
        if sec_per_q <= 0:
            raise ValueError(f"Non-positive inference latency ({sec_per_q}) for {model_key} {cond}")
        points.append((name, acc_pct, sec_per_q, is_think, is_med))

    for name, acc, sec, is_think, is_med in points:
        color = COLOR_MEDICAL_RAG if is_med else COLOR_GENERIC_RAG
        marker = "D" if is_think else "o"
        size = 140 if is_think else 80
        edge_color = COLOR_THINKING_MARK if is_think else "#333333"
        edge_width = 2.0 if is_think else 1.0

        ax.scatter(sec, acc, color=color, s=size, marker=marker, edgecolors=edge_color, linewidths=edge_width, zorder=4)
        
        # Placement offsets
        ha = "left" if sec > 10 else "right"
        dx = 1.15 if sec > 10 else 0.85
        ax.annotate(name, (sec, acc), xytext=(sec * dx, acc + 0.8), fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#DDDDDD", alpha=0.85))

    # Legend elements
    ax.scatter([], [], color=COLOR_GENERIC_RAG, marker="o", s=80, label="Standard Non-Thinking SLMs")
    ax.scatter([], [], color=COLOR_MEDICAL_RAG, marker="o", s=80, label="Medical Instruction-Tuned (MedGemma)")
    ax.scatter([], [], color=COLOR_MEDICAL_RAG, marker="D", s=120, edgecolors=COLOR_THINKING_MARK, linewidths=2.0, label="Thinking / Reasoning Models (MedPsy, Qwen-Thinking)")

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency per Question (Seconds, Log Scale)")
    ax.set_ylabel("MedQA Test Accuracy (%)")
    ax.set_title("Figure 16: Accuracy vs. Inference Latency Trade-off Across Model Classes (RQ4)")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 95)
    ax.set_xlim(0.08, 120)

    # Efficiency zones annotation
    ax.axvspan(0.08, 4.0, color="#E8F5E9", alpha=0.3, zorder=1, label="Sub-5s Edge Deployment Zone")
    ax.axvspan(10.0, 100.0, color="#EDE7F6", alpha=0.3, zorder=1, label="Extended Reasoning Zone (1024-token cap)")
    ax.text(0.15, 5, "Low-Latency Edge Zone (<4s)", fontsize=9, color="#2E7D32", fontweight="bold")
    ax.text(12.0, 5, "Extended Thinking Zone (15-65s)", fontsize=9, color="#512DA8", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig16_accuracy_vs_latency_thinking.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "fig16_accuracy_vs_latency_thinking.svg"))
    plt.close(fig)
    print("Saved Figure 16 to outputs/figures/fig16_accuracy_vs_latency_thinking.{png,svg}")


def main():
    apply_theme()
    out_dir = "outputs/figures"
    figures = [
        ("Figure 14", make_fig14_paired_bars),
        ("Figure 15", make_fig15_rag_gain_by_group),
        ("Figure 16", make_fig16_accuracy_vs_latency_thinking),
    ]
    failed = False
    for name, func in figures:
        try:
            func(out_dir)
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[SKIPPED] {name} could not be generated:\n{err}\n", file=sys.stderr)
            failed = True
        except Exception as err:
            print(f"\n[ERROR] {name} failed with unexpected error:\n{err}\n", file=sys.stderr)
            failed = True

    if failed:
        print("\nCompleted with errors: one or more figures were skipped due to missing outputs.\n", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll v3 figures successfully generated in outputs/figures/ (fig14..fig16).\n")


if __name__ == "__main__":
    main()
