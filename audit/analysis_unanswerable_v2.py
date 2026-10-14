#!/usr/bin/env python3
"""audit/analysis_unanswerable_v2.py

Evaluates abstention on unanswerable_v2 across all 6 models:
  - Primary MedPsy-4B: Evaluated using standard Phase 5 baseline prompt and decoding from
    outputs/kaggle_qa/unans_v2/medpsy-4b/ (or marked n/a [NOT RUN] if pending)
  - 5 standard SLMs (qwen3-1.7b, phi4-mini, smollm3-3b, qwen3-4b, gemma3-4b):
    Evaluated from outputs/kaggle_qa/unans_v2/<model>/ (or marked n/a [NOT RUN] if pending)
  - Secondary MedPsy-4B: Derived from earlier multi-sample self-consistency greedy trace
    (outputs/sc_val and outputs/sc_test_s3_unanswerable)

Identical Abstention Rule:
  Used identically across all 6 models: A question is classified as abstained if:
    1. Parsing failure or answer not in {"A", "B", "C", "D"}, OR
    2. Generation was truncated before answering (truncated == True), OR
    3. Output text matches spontaneous refusal phrases in final answer segment (after reasoning block), OR
    4. Option confidence < tau (where tau is tuned strictly on the validation split).

Protocol:
  - FIT NOTHING ON TEST: Calibration/threshold tau is fit strictly on validation split (N=260)
  - Frozen evaluation on held-out test split (N=140: 100 unanswerable + 40 controls)
  - Metrics:
      1. Abstention Recall (on unanswerable questions)
      2. False Abstention Rate (FAR, on control questions)
      3. Abstention Precision and F1
      4. Per-category Recall (ambiguous, fabricated_disease, fabricated_drug, false_premise, missing_info, out_of_scope)
      5. Percentile bootstrap 95% CIs (1,000 resamples, seed 42)

Outputs:
  - outputs/analysis/unanswerable_v2_metrics.json
  - docs/tables/table_unanswerable_v2_6models.md
  - docs/tables/table_unanswerable_v2_6models.csv
"""

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = REPO_ROOT / "outputs" / "analysis" / "unanswerable_v2_metrics.json"
TABLES_DIR = REPO_ROOT / "docs" / "tables"

SPONTANEOUS_REGEX = re.compile(
    r"(none of the|cannot be determined|not enough information|insufficient information|abstain)",
    re.IGNORECASE,
)

ALL_MODELS = ["medpsy-4b", "qwen3-1.7b", "phi4-mini", "smollm3-3b", "qwen3-4b", "gemma3-4b"]
CATEGORIES = [
    "ambiguous",
    "fabricated_disease",
    "fabricated_drug",
    "false_premise",
    "missing_info",
    "out_of_scope",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract_confidence(row: Dict[str, Any]) -> float:
    if "confidence" in row and row["confidence"] is not None:
        return float(row["confidence"])
    lp = row.get("letter_probs", {})
    if lp:
        return float(max(lp.values()))
    return 0.0


def extract_final_answer_segment(raw_output: str) -> str:
    """Extract the final answer segment after reasoning/thinking block ends.

    If </think> is present: returns text after the last </think>.
    If <think> is present without </think> (unclosed/truncated thinking):
        returns "" (no final answer segment produced; handled by truncation rule).
    If no thinking block is present:
        returns the full output.
    """
    if not raw_output:
        return ""
    think_close_matches = list(re.finditer(r"</think>", raw_output, re.IGNORECASE))
    if think_close_matches:
        return raw_output[think_close_matches[-1].end():]
    think_open_matches = list(re.finditer(r"<think>", raw_output, re.IGNORECASE))
    if think_open_matches:
        return ""
    return raw_output


def is_abstained_unified(row: Dict[str, Any], tau: Optional[float] = None) -> bool:
    """Unified abstention rule used identically for all 6 models.

    A prediction is deemed an abstention if:
      1. Format failure: parsed is False or pred is None or pred not in {A, B, C, D}.
      2. Truncation: generation hit max token limit without producing a valid answer (truncated == True).
      3. Explicit refusal: regex match for spontaneous unanswerable refusal phrases in the final answer
         segment (text after thinking block ends; if unclosed thinking, segment is empty).
      4. Low confidence: answer confidence < tau (where tau is calibrated strictly on validation split).
    """
    if not row.get("parsed") or row.get("pred") not in ("A", "B", "C", "D"):
        return True
    if row.get("truncated"):
        return True
    raw = str(row.get("raw_output", ""))
    final_seg = extract_final_answer_segment(raw)
    if SPONTANEOUS_REGEX.search(final_seg):
        return True
    if tau is not None:
        conf = extract_confidence(row)
        if conf < tau:
            return True
    return False


def is_abstained_medpsy_sc(row: Dict[str, Any]) -> bool:
    """Secondary MedPsy SC greedy pass rule (for legacy SC comparison row)."""
    g_ans = row.get("greedy_answer")
    g_insuf = bool(row.get("greedy_insufficient_info"))
    g_trunc = bool(row.get("greedy_truncated"))
    return bool(g_insuf or g_ans in ("ABSTAIN", None) or g_trunc)


def tune_tau_on_val(val_rows: List[Dict[str, Any]], step: float = 0.01) -> float:
    """Tune confidence threshold tau on validation split to maximize abstention F1.

    Tie-breaking: among equal-F1 thresholds, chooses the lowest tau (first encountered
    in ascending scan from 0.10 to 0.98).
    Raises ValueError if val_rows is empty.
    """
    if not val_rows:
        raise ValueError("Cannot tune tau: val_rows is empty. Validation split is required.")

    best_tau = None
    best_f1 = -1.0
    for t_int in range(10, 99):
        tau = round(t_int * step, 4)
        tp = sum(1 for r in val_rows if not r.get("control") and is_abstained_unified(r, tau))
        fp = sum(1 for r in val_rows if r.get("control") and is_abstained_unified(r, tau))
        fn = sum(1 for r in val_rows if not r.get("control") and not is_abstained_unified(r, tau))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        # Strict inequality ensures tie-breaking chooses the lowest tau
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    if best_tau is None:
        raise RuntimeError("Tau tuning failed to find a valid threshold on validation rows.")
    return best_tau


def compute_metrics(
    rows: List[Dict[str, Any]],
    abstain_fn,
) -> Dict[str, Any]:
    unans_items = [r for r in rows if not r.get("control", False)]
    ctrl_items = [r for r in rows if r.get("control", False)]

    tp = sum(1 for r in unans_items if abstain_fn(r))
    fn = len(unans_items) - tp
    fp = sum(1 for r in ctrl_items if abstain_fn(r))
    tn = len(ctrl_items) - fp

    rec = tp / len(unans_items) if unans_items else 0.0
    far = fp / len(ctrl_items) if ctrl_items else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # Per category recall on unanswerable questions
    cat_rec = {}
    for cat in CATEGORIES:
        cat_unans = [r for r in unans_items if r.get("category") == cat]
        if cat_unans:
            c_tp = sum(1 for r in cat_unans if abstain_fn(r))
            cat_rec[cat] = {
                "tp": c_tp,
                "total": len(cat_unans),
                "recall": round(c_tp / len(cat_unans), 4),
            }
        else:
            cat_rec[cat] = {"tp": 0, "total": 0, "recall": 0.0}

    return {
        "n_unans": len(unans_items),
        "n_ctrl": len(ctrl_items),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": round(rec, 4),
        "far": round(far, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "category_recall": cat_rec,
    }


def bootstrap_ci(
    rows: List[Dict[str, Any]],
    abstain_fn,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    rng = random.Random(seed)
    n = len(rows)
    recs, fars, f1s = [], [], []
    cat_lists = {cat: [] for cat in CATEGORIES}

    for _ in range(n_resamples):
        sample = [rows[rng.randint(0, n - 1)] for _ in range(n)]
        m = compute_metrics(sample, abstain_fn)
        recs.append(m["recall"])
        fars.append(m["far"])
        f1s.append(m["f1"])
        for cat in CATEGORIES:
            cat_lists[cat].append(m["category_recall"][cat]["recall"])

    recs.sort()
    fars.sort()
    f1s.sort()
    lo_i = int(0.025 * n_resamples)
    hi_i = int(0.975 * n_resamples)

    cis = {
        "recall_ci": (round(recs[lo_i], 4), round(recs[hi_i], 4)),
        "far_ci": (round(fars[lo_i], 4), round(fars[hi_i], 4)),
        "f1_ci": (round(f1s[lo_i], 4), round(f1s[hi_i], 4)),
    }
    for cat in CATEGORIES:
        cat_lists[cat].sort()
        cis[f"cat_{cat}_ci"] = (round(cat_lists[cat][lo_i], 4), round(cat_lists[cat][hi_i], 4))
    return cis


def evaluate_phase5_model(model_name: str) -> Dict[str, Any]:
    """Evaluate any of the 6 models using standard Phase 5 prompts & unified abstention rule."""
    base_dir = REPO_ROOT / "outputs" / "kaggle_qa" / "unans_v2" / model_name
    p5_path = base_dir / f"baseline_{model_name}.jsonl"
    p8_path = base_dir / f"rag_{model_name}.jsonl"

    if not p5_path.exists() and not p8_path.exists():
        return {"status": "NOT_RUN", "model": model_name}

    results = {"status": "AVAILABLE", "model": model_name}

    for mode, path in [("baseline", p5_path), ("rag", p8_path)]:
        if not path.exists():
            results[mode] = {"status": "NOT_RUN"}
            continue
        all_rows = load_jsonl(path)
        val_rows = [r for r in all_rows if r.get("split") == "validation"]
        test_rows = [r for r in all_rows if r.get("split") == "test"]

        if not val_rows:
            raise ValueError(f"Missing validation rows in {path} for model {model_name} mode {mode}. Validation split is required to fit tau.")

        # 1. Fit tau on validation split only using unified rule
        tau_val = tune_tau_on_val(val_rows)

        abstain_fn = lambda r: is_abstained_unified(r, tau=tau_val)
        test_m = compute_metrics(test_rows, abstain_fn)
        test_ci = bootstrap_ci(test_rows, abstain_fn)

        results[mode] = {
            "status": "AVAILABLE",
            "tau_val_fitted": tau_val,
            "test_metrics": test_m,
            "test_ci": test_ci,
        }
    return results


def evaluate_medpsy_sc() -> Optional[Dict[str, Any]]:
    """Secondary evaluation for MedPsy-4B under Self-Consistency protocol."""
    val_path = REPO_ROOT / "outputs" / "sc_val" / "medpsy-4b" / "unanswerable_v2.jsonl"
    test_path = REPO_ROOT / "outputs" / "sc_test_s3_unanswerable" / "outputs" / "sc_test" / "medpsy-4b" / "unanswerable_v2.jsonl"

    if not val_path.exists() or not test_path.exists():
        return None

    val_rows = load_jsonl(val_path)
    test_rows = load_jsonl(test_path)

    val_m = compute_metrics(val_rows, is_abstained_medpsy_sc)
    test_m = compute_metrics(test_rows, is_abstained_medpsy_sc)
    test_ci = bootstrap_ci(test_rows, is_abstained_medpsy_sc)

    return {
        "model": "medpsy-4b",
        "strategy": "Self-Consistency (K=5 Greedy Pass)",
        "val_metrics": val_m,
        "test_metrics": test_m,
        "test_ci": test_ci,
    }


def main():
    print("=" * 80)
    print("Cross-Model Unanswerable v2 Abstention Analysis (6 Models - Unified Rule)")
    print("=" * 80)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}

    # 1. Primary evaluations for all 6 models (Phase 5 prompt + decoding, unified rule)
    print("\n--- Primary Evaluations: Phase 5 Standardized Prompt (Unified Rule) ---")
    for m in ALL_MODELS:
        print(f"Evaluating {m} (Phase 5 standardized prompt)...")
        res = evaluate_phase5_model(m)
        all_data[m] = res
        if res.get("status") == "AVAILABLE":
            for mode in ("baseline", "rag"):
                if res.get(mode, {}).get("status") == "AVAILABLE":
                    tm = res[mode]["test_metrics"]
                    tci = res[mode]["test_ci"]
                    print(f"  [{m} {mode}] Tau: {res[mode]['tau_val_fitted']:.4f} | Recall: {tm['recall']*100:.1f}% {tci['recall_ci']} | FAR: {tm['far']*100:.1f}% | F1: {tm['f1']:.4f}")
        else:
            print(f"  [{m}] Pending runner output in outputs/kaggle_qa/unans_v2/{m}/ (n/a [NOT RUN])")

    # 2. Secondary evaluation for MedPsy-4B (SC-derived greedy trace)
    print("\n--- Secondary Evaluation: MedPsy-4B Self-Consistency Trace ---")
    sc_res = evaluate_medpsy_sc()
    if sc_res:
        all_data["medpsy-4b-sc-secondary"] = sc_res
        tm = sc_res["test_metrics"]
        tci = sc_res["test_ci"]
        print(f"  [medpsy-4b SC secondary] Recall: {tm['recall']*100:.1f}% {tci['recall_ci']} | FAR: {tm['far']*100:.1f}% | F1: {tm['f1']:.4f}")
    else:
        all_data["medpsy-4b-sc-secondary"] = {"status": "NOT_RUN"}
        print("  [medpsy-4b SC secondary] Inputs not found; marking n/a")

    # Save JSON
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(all_data, fh, indent=2)
    print(f"\nSaved metrics JSON to: {OUT_JSON.relative_to(REPO_ROOT)}")

    # Build Markdown and CSV Table
    headers = [
        "Model",
        "Role / Protocol",
        "Mode",
        "Fitted tau (Val)",
        "Decision Strategy",
        "Abstention Recall (Unanswerable)",
        "False Abstention Rate (Controls)",
        "Abstention F1",
        "Ambiguous Recall",
        "Fabricated Entity Recall",
        "False Premise Recall",
    ]
    rows = []

    # 1. Primary MedPsy-4B entry
    m_p5 = all_data.get("medpsy-4b", {})
    if m_p5.get("status") == "AVAILABLE" and m_p5.get("baseline", {}).get("status") == "AVAILABLE":
        b_data = m_p5["baseline"]
        tm = b_data["test_metrics"]
        tci = b_data["test_ci"]
        cr = tm["category_recall"]
        fab_rec = (cr["fabricated_disease"]["tp"] + cr["fabricated_drug"]["tp"]) / max(1, cr["fabricated_disease"]["total"] + cr["fabricated_drug"]["total"])
        tau_val_str = f"{b_data.get('tau_val_fitted'):.4f}" if b_data.get("tau_val_fitted") is not None else "n/a"
        rows.append([
            "medpsy-4b",
            "Primary (Phase 5 Prompt)",
            "Baseline",
            tau_val_str,
            f"Unified Rule (conf < {tau_val_str} abstains)",
            f"{tm['recall']*100:.1f}% [{tci['recall_ci'][0]*100:.1f}%, {tci['recall_ci'][1]*100:.1f}%]",
            f"{tm['far']*100:.1f}% [{tci['far_ci'][0]*100:.1f}%, {tci['far_ci'][1]*100:.1f}%]",
            f"{tm['f1']:.4f} [{tci['f1_ci'][0]:.4f}, {tci['f1_ci'][1]:.4f}]",
            f"{cr['ambiguous']['recall']*100:.1f}%",
            f"{fab_rec*100:.1f}%",
            f"{cr['false_premise']['recall']*100:.1f}%",
        ])
    else:
        rows.append(["medpsy-4b", "Primary (Phase 5 Prompt)", "Baseline", "n/a [NOT RUN]", "Unified Rule (Val-calibrated tau)", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]"])

    # 2. Standard 5 SLMs
    for m in ALL_MODELS[1:]:
        m_entry = all_data.get(m, {})
        for mode in ("Baseline", "RAG"):
            mode_key = mode.lower()
            mode_data = m_entry.get(mode_key, {})
            if mode_data.get("status") == "AVAILABLE":
                tm = mode_data["test_metrics"]
                tci = mode_data["test_ci"]
                cr = tm["category_recall"]
                fab_rec = (cr["fabricated_disease"]["tp"] + cr["fabricated_drug"]["tp"]) / max(1, cr["fabricated_disease"]["total"] + cr["fabricated_drug"]["total"])
                tau_val_str = f"{mode_data.get('tau_val_fitted'):.4f}" if mode_data.get("tau_val_fitted") is not None else "n/a"
                rows.append([
                    m,
                    "Primary (Phase 5/8)",
                    mode,
                    tau_val_str,
                    f"Unified Rule (conf < {tau_val_str} abstains)",
                    f"{tm['recall']*100:.1f}% [{tci['recall_ci'][0]*100:.1f}%, {tci['recall_ci'][1]*100:.1f}%]",
                    f"{tm['far']*100:.1f}% [{tci['far_ci'][0]*100:.1f}%, {tci['far_ci'][1]*100:.1f}%]",
                    f"{tm['f1']:.4f} [{tci['f1_ci'][0]:.4f}, {tci['f1_ci'][1]:.4f}]",
                    f"{cr['ambiguous']['recall']*100:.1f}%",
                    f"{fab_rec*100:.1f}%",
                    f"{cr['false_premise']['recall']*100:.1f}%",
                ])
            else:
                rows.append([m, "Primary (Phase 5/8)", mode, "n/a [NOT RUN]", "Unified Rule (Val-calibrated tau)", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]"])

    # 3. Secondary MedPsy SC-derived entry
    sc_entry = all_data.get("medpsy-4b-sc-secondary", {})
    if sc_entry.get("test_metrics"):
        tm = sc_entry["test_metrics"]
        tci = sc_entry["test_ci"]
        cr = tm["category_recall"]
        fab_rec = (cr["fabricated_disease"]["tp"] + cr["fabricated_drug"]["tp"]) / max(1, cr["fabricated_disease"]["total"] + cr["fabricated_drug"]["total"])
        rows.append([
            "medpsy-4b",
            "Secondary (SC-Derived)",
            "Greedy Pass (K=5)",
            "—",
            "Greedy Refusal / Truncation",
            f"{tm['recall']*100:.1f}% [{tci['recall_ci'][0]*100:.1f}%, {tci['recall_ci'][1]*100:.1f}%]",
            f"{tm['far']*100:.1f}% [{tci['far_ci'][0]*100:.1f}%, {tci['far_ci'][1]*100:.1f}%]",
            f"{tm['f1']:.4f} [{tci['f1_ci'][0]:.4f}, {tci['f1_ci'][1]:.4f}]",
            f"{cr['ambiguous']['recall']*100:.1f}%",
            f"{fab_rec*100:.1f}%",
            f"{cr['false_premise']['recall']*100:.1f}%",
        ])
    else:
        rows.append(["medpsy-4b", "Secondary (SC-Derived)", "Greedy Pass (K=5)", "—", "Greedy Refusal", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]", "n/a [NOT RUN]"])

    md_path = TABLES_DIR / "table_unanswerable_v2_6models.md"
    csv_path = TABLES_DIR / "table_unanswerable_v2_6models.csv"

    footnote = (
        "\n*Footnote*: Evaluated on held-out test split of Unanswerable v2 ($N=140$: 100 unanswerable clinical queries + 40 answerable clinical controls).\n"
        "**Identical Abstention Rule Across All 6 Models**: A question is classified as abstained if and only if "
        "(1) model output fails format parsing or predicts an option outside {A, B, C, D}, "
        "(2) generation was truncated before providing a final answer (`truncated == True`), "
        "(3) generation text matches spontaneous refusal phrases ('none of the', 'cannot be determined', 'not enough information', 'insufficient information', 'abstain') in the final answer segment after reasoning, OR "
        "(4) predicted option confidence falls below validation-calibrated threshold tau. "
        "Threshold tau is tuned strictly on the unanswerable_v2 validation split ($N=260$) to maximize abstention F1 without any test-split leakage (tie-breaking chooses lowest tau).\n"
        "**Reporting Structure**: The primary MedPsy-4B row uses the standardized Phase 5 baseline prompt and greedy decoding identical to the other 5 SLMs (output stored in `outputs/kaggle_qa/unans_v2/medpsy-4b/`). "
        "The secondary MedPsy-4B row reports the multi-sample self-consistency (SC) greedy trace as reference. "
        "95% CIs are computed via 1,000 percentile bootstrap resamples (seed 42). Pending models designated as `n/a [NOT RUN]`.\n"
    )

    # Write Markdown
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Table: Cross-Model Selective Abstention on Unanswerable v2 (Unified Decision Rule)\n\n")
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for r in rows:
            fh.write("| " + " | ".join(r) + " |\n")
        fh.write(footnote)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Saved Markdown table to: {md_path.relative_to(REPO_ROOT)}")
    print(f"Saved CSV table to     : {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
