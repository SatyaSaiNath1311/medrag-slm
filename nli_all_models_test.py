#!/usr/bin/env python3
"""nli_all_models_test.py

Evaluates NLI citation entailment and answer-level grounding support across all 6 models:
  - 5 standard SLMs: qwen3-1.7b, phi4-mini, smollm3-3b, qwen3-4b, gemma3-4b
  - Reasoning model: medpsy-4b (included via pre-computed run / unified evaluation)

Reuses the frozen logic in nli_medpsy_test.py exactly:
  - NLI Model: pritamdeka/PubMedBERT-MNLI-MedNLI
  - Threshold: 0.5
  - Evidence regex: Evidence\s*:\s*((?:\[\s*\d+\s*\][\s,;and]*)+)
  - Truncated-row skipping
  - Dev-validated calibration: PREC = 0.52 (95% CI: 0.335-0.70), FNR = 0.04 (95% CI: 0.007-0.195)
  - Strict assertions ensuring zero parameter divergence from the MedPsy run.

Output:
  outputs/analysis_nli/all_models_test_citation_metrics.json
"""

import collections
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── FROZEN CONFIGURATION (Strict assertion against deviation) ─────────────────
NLI = "pritamdeka/PubMedBERT-MNLI-MedNLI"
THR = 0.5
PREC, PREC_CI = 0.52, (0.335, 0.70)
FNR, FNR_CI = 0.04, (0.007, 0.195)

assert NLI == "pritamdeka/PubMedBERT-MNLI-MedNLI", "NLI model must match frozen MedPsy run"
assert THR == 0.5, "Threshold must match frozen MedPsy run (0.5)"
assert PREC == 0.52 and PREC_CI == (0.335, 0.70), "Dev precision must match frozen MedPsy run"
assert FNR == 0.04 and FNR_CI == (0.007, 0.195), "Dev FNR must match frozen MedPsy run"

EV_RE = re.compile(r"Evidence\s*:\s*((?:\[\s*\d+\s*\][\s,;and]*)+)", re.I)

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "outputs" / "analysis_nli"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "all_models_test_citation_metrics.json"

MODEL_PATHS = {
    "qwen3-1.7b": REPO_ROOT / "outputs" / "kaggle_qa" / "full" / "qwen3-1.7b" / "work" / "phase8" / "qwen3-1.7b.jsonl",
    "phi4-mini":   REPO_ROOT / "outputs" / "kaggle_qa" / "full" / "phi4-mini" / "work" / "phase8" / "phi4-mini.jsonl",
    "smollm3-3b":  REPO_ROOT / "outputs" / "kaggle_qa" / "full" / "smollm3-3b" / "work" / "phase8" / "smollm3-3b.jsonl",
    "qwen3-4b":    REPO_ROOT / "outputs" / "kaggle_qa" / "full" / "qwen3-4b" / "work" / "phase8" / "qwen3-4b.jsonl",
    "gemma3-4b":   REPO_ROOT / "outputs" / "kaggle_qa" / "full" / "gemma3-4b_rag" / "work" / "phase8" / "gemma3-4b.jsonl",
}

MEDPSY_PATH = REPO_ROOT / "outputs" / "kaggle_qa" / "medpsy_rag" / "work" / "phase8" / "medpsy-4b.jsonl"
MEDPSY_EXISTING_METRICS = OUT_DIR / "medpsy_test_citation_metrics.json"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def final_cites(raw_text: str) -> Optional[List[int]]:
    sec = raw_text.rsplit("</think>", 1)[-1] if "</think>" in raw_text else raw_text
    m = EV_RE.findall(sec)
    return sorted({int(n) for n in re.findall(r"\d+", m[-1])}) if m else None


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    dn = 1 + z * z / n
    cc = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (cc - a) / dn), min(1.0, (cc + a) / dn)


def fmt_stat(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    pct = (k / max(n, 1)) * 100.0
    return f"{k}/{n} = {pct:.1f}% (95% CI {lo*100:.1f}%-{hi*100:.1f}%)"


def evaluate_model(
    model_name: str,
    raw_rows: List[Dict[str, Any]],
    qs: Dict[str, Dict[str, Any]],
    ev: Dict[str, Dict[str, Any]],
    tok: AutoTokenizer,
    mdl: AutoModelForSequenceClassification,
    ent_idx: int,
    batch_size: int = 8,
) -> Dict[str, Any]:
    # Filter to MedQA test split rows
    rows = [r for r in raw_rows if r.get("dataset") == "medqa" and r.get("split", "test") == "test"]
    assert len(rows) == 500, f"Expected 500 MedQA test rows for {model_name}, found {len(rows)}"
    assert all(r["id"] in qs and r["id"] in ev for r in rows), f"Missing question or evidence mapping for {model_name}"

    mis = sum(1 for r in rows if [p["chunk_id"] for p in ev[r["id"]]["passages"]] != list(r.get("evidence_chunk_ids", [])))
    print(f"[{model_name}] evidence join mismatches: {mis} / 500")
    assert mis == 0, f"Passage join mismatch in {model_name}; stop"

    n_trunc = 0
    n_noline = 0
    items = []
    for r in rows:
        if r.get("truncated"):
            n_trunc += 1
            continue
        q = qs[r["id"]]
        c = final_cites(r.get("raw_output", ""))
        psg = ev[r["id"]]["passages"]
        c = [x for x in (c or []) if 1 <= x <= len(psg)]
        if r.get("pred") not in q["options"] or not c:
            n_noline += 1
            continue
        items.append((r, q, psg, c))

    N = len(items)
    print(f"[{model_name}] scorable: {N}/500 ({N/500:.1%}) | truncated: {n_trunc} | no usable Evidence line: {n_noline}")

    if N == 0:
        return {
            "scorable": 0,
            "truncated": n_trunc,
            "no_line": n_noline,
            "pair_entail": [0, 0],
            "corrected_est": [0.0, 0.0, 0.0],
            "answer_support": [0, 0],
            "citation_recall_available": [0, 0],
            "citation_recall_cited": [0, 0],
        }

    # Prepare premise-hypothesis pairs
    prem, hyp, key = [], [], []
    for qi, (r, q, psg, c) in enumerate(items):
        h = f"The correct answer to the question is: {q['options'][r['pred']]}."
        for k in range(1, len(psg) + 1):
            prem.append(psg[k - 1]["text"])
            hyp.append(h)
            key.append((qi, k))

    print(f"[{model_name}] scoring {len(prem)} premise-hypothesis pairs on {mdl.device}...")

    # Batch inference
    out_probs = []
    with torch.no_grad():
        for i in range(0, len(prem), batch_size):
            enc = tok(
                prem[i:i + batch_size],
                hyp[i:i + batch_size],
                truncation="only_first",
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(mdl.device)
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, -1)[:, ent_idx].tolist()
            out_probs.extend(probs)

    sc = dict(zip(key, out_probs))

    # Metric 1: Citation precision at pair level
    cited_pairs = [(qi, k) for qi, (r, q, psg, c) in enumerate(items) for k in c]
    k_pair = sum(sc[p] >= THR for p in cited_pairs)
    total_cited = len(cited_pairs)

    rate = k_pair / total_cited if total_cited > 0 else 0.0
    lo_corr = rate * PREC_CI[0] + (1.0 - rate) * FNR_CI[0]
    hi_corr = rate * PREC_CI[1] + (1.0 - rate) * FNR_CI[1]
    est_corr = rate * PREC + (1.0 - rate) * FNR

    # Metric 2: Answer-level grounding support (any cited passage entailed)
    sup = [any(sc[(qi, k)] >= THR for k in c) for qi, (r, q, psg, c) in enumerate(items)]
    cor = [r["pred"] == r["gold"] for (r, q, psg, c) in items]

    # Metric 3: Citation recall
    # Does question have >= 1 entailed retrieved passage among top-5?
    avail = [any(sc[(qi, k)] >= THR for k in range(1, len(psg) + 1)) for qi, (r, q, psg, c) in enumerate(items)]
    k_avail = sum(avail)
    k_cited_entailed = sum(s for s, a in zip(sup, avail) if a)

    print(f"  pair-level entailment : {fmt_stat(k_pair, total_cited)}")
    print(f"  corrected precision   : ~{est_corr:.1%} (range {lo_corr:.1%}-{hi_corr:.1%})")
    print(f"  answer-level support  : {fmt_stat(sum(sup), N)}")
    print(f"  questions with >=1 entailed passage: {fmt_stat(k_avail, N)}")
    print(f"  ...cited an entailed passage       : {fmt_stat(k_cited_entailed, k_avail)}")

    return {
        "scorable": N,
        "truncated": n_trunc,
        "no_line": n_noline,
        "total_cited_passages": total_cited,
        "pair_entail": [k_pair, total_cited],
        "pair_entail_rate": round(rate, 4),
        "corrected_est": [round(est_corr, 4), round(lo_corr, 4), round(hi_corr, 4)],
        "answer_support": [sum(sup), N],
        "answer_support_rate": round(sum(sup) / N, 4) if N else 0.0,
        "answer_support_correct": [sum(s for s, y in zip(sup, cor) if y), sum(cor)],
        "answer_support_wrong": [sum(s for s, y in zip(sup, cor) if not y), N - sum(cor)],
        "citation_recall_available": [k_avail, N],
        "citation_recall_cited": [k_cited_entailed, k_avail],
        "citation_recall_rate": round(k_cited_entailed / k_avail, 4) if k_avail else 0.0,
    }


def main():
    print("=" * 80)
    print("Unified NLI Citation Faithfulness Evaluation across 6 Models")
    print(f"NLI Model : {NLI} (Threshold={THR})")
    print(f"Calib     : Precision={PREC} {PREC_CI}, FNR={FNR} {FNR_CI}")
    print("=" * 80 + "\n")

    # Load shared questions and evidence
    test_jsonl = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase1" / "test.jsonl"
    ev_jsonl = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase7" / "evidence.jsonl"

    if not test_jsonl.exists() or not ev_jsonl.exists():
        sys.exit(f"ERROR: test.jsonl or evidence.jsonl missing. Run build phase first.")

    qs = {r["id"]: r for r in load_jsonl(test_jsonl) if r.get("dataset") == "medqa"}
    ev = {r["id"]: r for r in load_jsonl(ev_jsonl) if r.get("split") == "test" and r.get("dataset") == "medqa"}
    assert len(qs) == 500, f"Expected 500 MedQA test questions, found {len(qs)}"

    # Load NLI model
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading NLI model {NLI} onto {device}...")
    tok = AutoTokenizer.from_pretrained(NLI)
    mdl = AutoModelForSequenceClassification.from_pretrained(NLI).to(device).eval()

    labs = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
    ent_matches = [k for k, v in labs.items() if "entail" in v]
    assert len(ent_matches) == 1, f"Expected 1 entailment label, found {ent_matches}"
    ent_idx = ent_matches[0]

    all_metrics = {}

    # Evaluate the 5 standard SLMs
    for model_name, path in MODEL_PATHS.items():
        print(f"\nProcessing {model_name} from {path.relative_to(REPO_ROOT)}...")
        raw_rows = load_jsonl(path)
        all_metrics[model_name] = evaluate_model(
            model_name=model_name,
            raw_rows=raw_rows,
            qs=qs,
            ev=ev,
            tok=tok,
            mdl=mdl,
            ent_idx=ent_idx,
            batch_size=16 if device != "cpu" else 8,
        )

    # MedPsy-4B: Evaluate or reuse existing verified result
    if MEDPSY_PATH.exists():
        print(f"\nProcessing medpsy-4b from {MEDPSY_PATH.relative_to(REPO_ROOT)}...")
        raw_rows = load_jsonl(MEDPSY_PATH)
        all_metrics["medpsy-4b"] = evaluate_model(
            model_name="medpsy-4b",
            raw_rows=raw_rows,
            qs=qs,
            ev=ev,
            tok=tok,
            mdl=mdl,
            ent_idx=ent_idx,
            batch_size=16 if device != "cpu" else 8,
        )
    elif MEDPSY_EXISTING_METRICS.exists():
        print(f"\nLoading existing verified medpsy-4b metrics from {MEDPSY_EXISTING_METRICS.relative_to(REPO_ROOT)}...")
        with open(MEDPSY_EXISTING_METRICS, "r", encoding="utf-8") as fh:
            all_metrics["medpsy-4b"] = json.load(fh)

    # Save unified JSON
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)

    print("\n" + "=" * 80)
    print(f"SUCCESS: Saved unified 6-model citation metrics to: {OUT_FILE.relative_to(REPO_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
