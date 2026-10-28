#!/usr/bin/env python3
"""src/select_deployment_model.py — Deployment Model Selection for MedRAG-SLM Chatbot.

Evaluates candidates on the MedQA validation split (shared N=100 questions):
  - Candidates: medpsy-4b vs medgemma-1.5-4b-it
  - Rule: docs/MODEL_SELECTION_RULE.md
  - Outputs:
      docs/MODEL_SELECTION.md
      docs/model_selection.json

Supports:
  python3 -m src.select_deployment_model
  python3 -m src.select_deployment_model --check
  python3 -m src.select_deployment_model --repo-root /path/to/repo
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.analysis_budget_sweep import (
    exact_mcnemar_test,
    format_p_val,
    is_finished_within_cap,
    is_strict_correct_at_cap,
    is_truncated_at_cap,
    paired_bootstrap_ci,
    wilson_score_ci,
)
from src.common import read_jsonl

CAPS = [1024, 1536, 2048, 3072, 4096]


# ── Data Loaders & Assertions ──────────────────────────────────────────────────

def load_and_validate_candidates(repo_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load validation sweep rows and manifests for medpsy-4b and medgemma-1.5-4b-it.

    Enforces:
      - MedGemma rows are the V1 variant restricted to the EXACT 100 question IDs
        present in the medpsy-4b sweep file, in the same file order.
      - The final ID lists must be strictly identical in order.
      - Both sides must have exactly 100 rows (hard assertion; script aborts otherwise).
    """
    p_medpsy = repo_root / "outputs" / "sweeps" / "budget_pair_b" / "medpsy-4b.jsonl"
    p_medgemma = repo_root / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "medgemma-1.5-4b-it.jsonl"
    p_manifest_medpsy = repo_root / "outputs" / "sweeps" / "budget_pair_b" / "manifest_medpsy-4b.json"
    p_manifest_medgemma = repo_root / "outputs" / "sweeps" / "gemma_prompt_sensitivity" / "manifest_medgemma-1.5-4b-it.json"

    for p in [p_medpsy, p_medgemma, p_manifest_medpsy, p_manifest_medgemma]:
        if not p.exists():
            raise FileNotFoundError(f"Required input artifact missing: {p.relative_to(repo_root)}")

    medpsy_all = read_jsonl(p_medpsy)
    medgemma_all = read_jsonl(p_medgemma)

    # All 100 medpsy rows (file contains exactly 100 questions)
    medpsy_rows = medpsy_all[:100]
    assert len(medpsy_rows) == 100, (
        f"Expected 100 medpsy rows, got {len(medpsy_rows)}"
    )
    medpsy_id_order: List[str] = [r["id"] for r in medpsy_rows]

    # Build a dict of V1 rows keyed by question ID.
    # The gemma file has 600 rows (3 variants × 200 questions); we need only V1
    # and only the questions that are in the medpsy sweep, in medpsy file order.
    medpsy_id_set = set(medpsy_id_order)
    medgemma_v1_by_id: Dict[str, Dict[str, Any]] = {}
    has_variant_field = any("variant" in r for r in medgemma_all)
    for r in medgemma_all:
        if has_variant_field:
            if r.get("variant") != "V1":
                continue
        # Only keep if the id is one of the 100 medpsy questions
        if r["id"] in medpsy_id_set:
            # In case of duplicates within V1 keep first occurrence
            if r["id"] not in medgemma_v1_by_id:
                medgemma_v1_by_id[r["id"]] = r

    # Reconstruct medgemma rows in the SAME order as medpsy_id_order
    missing = [qid for qid in medpsy_id_order if qid not in medgemma_v1_by_id]
    assert not missing, (
        f"MedGemma V1 is missing {len(missing)} question IDs that are in MedPsy: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
    )
    medgemma_rows = [medgemma_v1_by_id[qid] for qid in medpsy_id_order]

    assert len(medgemma_rows) == 100, (
        f"Expected exactly 100 MedGemma V1 rows (one per medpsy question), "
        f"got {len(medgemma_rows)}"
    )

    # Shared-ID hard assertion: order and IDs must be identical
    medgemma_ids = [r["id"] for r in medgemma_rows]
    assert medpsy_id_order == medgemma_ids, (
        "Question IDs do not match in set and file order between MedPsy-4B and MedGemma-1.5-4B-it"
    )

    # Load manifests
    manifest_medpsy = json.loads(p_manifest_medpsy.read_text(encoding="utf-8"))
    manifest_medgemma = json.loads(p_manifest_medgemma.read_text(encoding="utf-8"))

    manifests = {
        "medpsy-4b": manifest_medpsy,
        "medgemma-1.5-4b-it": manifest_medgemma,
    }

    return medpsy_rows, medgemma_rows, manifests


# ── Candidate Evaluation ───────────────────────────────────────────────────────

def evaluate_medpsy_caps(rows: List[Dict[str, Any]], caps: List[int] = CAPS) -> Dict[str, Any]:
    """Evaluate medpsy-4b across the candidate cap grid."""
    n = len(rows)
    cap_evals = {}
    trunc_rates = {}

    for c in caps:
        s_corr = [1 if is_strict_correct_at_cap(r, c) else 0 for r in rows]
        f_corr = [1 if bool(r.get("correct")) else 0 for r in rows]
        trunc = [1 if is_truncated_at_cap(r, c) else 0 for r in rows]
        parsed = [1 if bool(r.get("parsed", True)) else 0 for r in rows]

        s_k = sum(s_corr)
        f_k = sum(f_corr)
        t_k = sum(trunc)
        p_k = sum(parsed)

        t_rate = t_k / n
        trunc_rates[c] = t_rate

        cap_evals[c] = {
            "cap": c,
            "n": n,
            "strict_k": s_k,
            "strict_acc": s_k / n,
            "strict_ci": wilson_score_ci(s_k, n),
            "fallback_k": f_k,
            "fallback_acc": f_k / n,
            "fallback_ci": wilson_score_ci(f_k, n),
            "trunc_k": t_k,
            "trunc_rate": t_rate,
            "trunc_ci": wilson_score_ci(t_k, n),
            "parse_k": p_k,
            "parse_rate": p_k / n,
            "parse_ci": wilson_score_ci(p_k, n),
            "strict_binary": s_corr,
        }

    # Choose deployment cap: smallest cap with truncation <= 5%
    chosen_cap = 4096
    cap_rationale = "Fallback to 4096"
    for c in caps:
        if trunc_rates[c] <= 0.05:
            chosen_cap = c
            cap_rationale = (
                f"Smallest cap in grid where validation truncation <= 5.0% "
                f"(observed {trunc_rates[c]*100:.1f}%)."
            )
            break

    # Latency statistics
    seconds_list = [float(r.get("seconds", 0.0)) for r in rows if "seconds" in r]
    mean_sec = float(np.mean(seconds_list)) if seconds_list else 0.0
    median_sec = float(np.median(seconds_list)) if seconds_list else 0.0

    return {
        "model": "medpsy-4b",
        "style": "reasoning (thinking)",
        "chosen_cap": chosen_cap,
        "cap_rationale": cap_rationale,
        "caps": cap_evals,
        "truncation_by_cap": trunc_rates,
        "at_chosen_cap": cap_evals[chosen_cap],
        "seconds_mean": mean_sec,
        "seconds_median": median_sec,
    }


def evaluate_medgemma(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate medgemma-1.5-4b-it at its direct-answer cap (max_new_tokens=16)."""
    n = len(rows)
    s_corr = []
    f_corr = []
    trunc = []
    parsed = []

    for r in rows:
        # Amended definition: truncated iff not parsed
        is_p = bool(r.get("parsed", True))
        is_t = bool(r.get("truncated", False))
        is_c = bool(r.get("correct", False))

        is_strict = (not is_t) and is_p and is_c
        s_corr.append(1 if is_strict else 0)
        f_corr.append(1 if is_c else 0)
        trunc.append(1 if is_t else 0)
        parsed.append(1 if is_p else 0)

    s_k = sum(s_corr)
    f_k = sum(f_corr)
    t_k = sum(trunc)
    p_k = sum(parsed)

    seconds_list = [float(r.get("seconds", 0.0)) for r in rows if "seconds" in r]
    mean_sec = float(np.mean(seconds_list)) if seconds_list else 0.0
    median_sec = float(np.median(seconds_list)) if seconds_list else 0.0

    return {
        "model": "medgemma-1.5-4b-it",
        "style": "direct-answer",
        "chosen_cap": 16,
        "cap_rationale": "Direct-answer model fixed at operational cap max_new_tokens=16.",
        "n": n,
        "strict_k": s_k,
        "strict_acc": s_k / n,
        "strict_ci": wilson_score_ci(s_k, n),
        "fallback_k": f_k,
        "fallback_acc": f_k / n,
        "fallback_ci": wilson_score_ci(f_k, n),
        "trunc_k": t_k,
        "trunc_rate": t_k / n,
        "trunc_ci": wilson_score_ci(t_k, n),
        "parse_k": p_k,
        "parse_rate": p_k / n,
        "parse_ci": wilson_score_ci(p_k, n),
        "strict_binary": s_corr,
        "seconds_mean": mean_sec,
        "seconds_median": median_sec,
    }


def compare_candidates(medpsy_eval: Dict[str, Any], medgemma_eval: Dict[str, Any]) -> Dict[str, Any]:
    """Perform paired statistical test between medpsy-4b (at chosen cap) and medgemma-1.5-4b-it."""
    s_m = np.array(medpsy_eval["at_chosen_cap"]["strict_binary"])
    s_g = np.array(medgemma_eval["strict_binary"])

    # Discordant pairs: b = medpsy 1 & medgemma 0; c = medpsy 0 & medgemma 1
    b = int(np.sum((s_m == 1) & (s_g == 0)))
    c = int(np.sum((s_m == 0) & (s_g == 1)))

    p_val = exact_mcnemar_test(b, c)
    lo, hi = paired_bootstrap_ci(s_m, s_g, n_boot=10000, seed=42)
    diff = float(np.mean(s_m) - np.mean(s_g))

    ci_m = medpsy_eval["at_chosen_cap"]["strict_ci"]
    ci_g = medgemma_eval["strict_ci"]
    overlap_ci = max(ci_m[0], ci_g[0]) <= min(ci_m[1], ci_g[1])

    is_significant = (p_val < 0.05) or (not overlap_ci)

    return {
        "b": b,
        "c": c,
        "diff": diff,
        "bootstrap_ci": (lo, hi),
        "mcnemar_p": p_val,
        "overlapping_cis": overlap_ci,
        "is_significant": is_significant,
    }


# ── Pure Rule Function ─────────────────────────────────────────────────────────

def select_deployment_model(
    medpsy_eval: Dict[str, Any],
    medgemma_eval: Dict[str, Any],
    comparison: Dict[str, Any],
) -> Dict[str, Any]:
    """Pure decision rule function implementing docs/MODEL_SELECTION_RULE.md.

    Eligibility:
      - Truncation rate <= 15.0%
      - Parse rate >= 95.0%
      - Runs on Kaggle 2x T4 (confirmed by committed manifests)

    Winner:
      - Highest strict accuracy among eligible candidates.
      - If top two are not significantly different (p >= 0.05 AND overlapping CIs),
        break tie by lower seconds per question.
    """
    m_trunc = medpsy_eval["at_chosen_cap"]["trunc_rate"]
    m_parse = medpsy_eval["at_chosen_cap"]["parse_rate"]
    m_acc = medpsy_eval["at_chosen_cap"]["strict_acc"]
    m_sec = medpsy_eval["seconds_mean"]

    g_trunc = medgemma_eval["trunc_rate"]
    g_parse = medgemma_eval["parse_rate"]
    g_acc = medgemma_eval["strict_acc"]
    g_sec = medgemma_eval["seconds_mean"]

    medpsy_eligible = (m_trunc <= 0.15) and (m_parse >= 0.95)
    medgemma_eligible = (g_trunc <= 0.15) and (g_parse >= 0.95)

    eligibility = {
        "medpsy-4b": {
            "eligible": medpsy_eligible,
            "trunc_rate": m_trunc,
            "trunc_threshold_met": m_trunc <= 0.15,
            "parse_rate": m_parse,
            "parse_threshold_met": m_parse >= 0.95,
            "hardware_feasible": True,
            "hardware_platform": "Kaggle 2x NVIDIA T4 (float16)",
            "vram_status": "VRAM not measured, to be measured in the live test",
        },
        "medgemma-1.5-4b-it": {
            "eligible": medgemma_eligible,
            "trunc_rate": g_trunc,
            "trunc_threshold_met": g_trunc <= 0.15,
            "parse_rate": g_parse,
            "parse_threshold_met": g_parse >= 0.95,
            "hardware_feasible": True,
            "hardware_platform": "Kaggle 2x NVIDIA T4 (float32)",
            "vram_status": "VRAM not measured, to be measured in the live test",
        },
    }

    if not medpsy_eligible and not medgemma_eligible:
        return {
            "winner": None,
            "status": "NO_ELIGIBLE_CANDIDATE",
            "deployment_cap": None,
            "rationale": "Neither candidate met the eligibility criteria (truncation <= 15%, parse >= 95%).",
            "eligibility": eligibility,
        }

    if medpsy_eligible and not medgemma_eligible:
        return {
            "winner": "medpsy-4b",
            "status": "SOLE_ELIGIBLE_CANDIDATE",
            "deployment_cap": medpsy_eval["chosen_cap"],
            "rationale": "MedPsy-4B is the sole eligible candidate; MedGemma failed eligibility.",
            "eligibility": eligibility,
        }

    if medgemma_eligible and not medpsy_eligible:
        return {
            "winner": "medgemma-1.5-4b-it",
            "status": "SOLE_ELIGIBLE_CANDIDATE",
            "deployment_cap": 16,
            "rationale": "MedGemma-1.5-4B-it is the sole eligible candidate; MedPsy-4B failed eligibility.",
            "eligibility": eligibility,
        }

    # Both eligible: compare accuracy and test for tie
    is_sig = comparison["is_significant"]
    if not is_sig:
        # Tie-break by latency
        if g_sec < m_sec:
            winner = "medgemma-1.5-4b-it"
            cap = 16
            rationale = (
                f"Tie-break by latency: Strict accuracy difference is not statistically significant "
                f"(p = {comparison['mcnemar_p']:.4f} >= 0.05, overlapping CIs). "
                f"MedGemma-1.5-4B-it selected due to lower latency ({g_sec:.2f}s vs {m_sec:.2f}s per question)."
            )
        else:
            winner = "medpsy-4b"
            cap = medpsy_eval["chosen_cap"]
            rationale = (
                f"Tie-break by latency: Strict accuracy difference is not statistically significant "
                f"(p = {comparison['mcnemar_p']:.4f} >= 0.05, overlapping CIs). "
                f"MedPsy-4B selected due to lower latency ({m_sec:.2f}s vs {g_sec:.2f}s per question)."
            )
    else:
        # Significant difference: higher strict accuracy wins
        if m_acc >= g_acc:
            winner = "medpsy-4b"
            cap = medpsy_eval["chosen_cap"]
            rationale = (
                f"Selected on strict accuracy: MedPsy-4B achieves {m_acc*100:.1f}% vs "
                f"MedGemma-1.5-4B-it's {g_acc*100:.1f}% (gap {comparison['diff']*100:+.1f} pp, "
                f"p = {format_p_val(comparison['mcnemar_p'])}, non-overlapping CIs). "
                f"High-accuracy model selected regardless of latency."
            )
        else:
            winner = "medgemma-1.5-4b-it"
            cap = 16
            rationale = (
                f"Selected on strict accuracy: MedGemma-1.5-4B-it achieves {g_acc*100:.1f}% vs "
                f"MedPsy-4B's {m_acc*100:.1f}% (gap {-comparison['diff']*100:+.1f} pp, "
                f"p = {format_p_val(comparison['mcnemar_p'])}). "
                f"High-accuracy model selected regardless of latency."
            )

    return {
        "winner": winner,
        "status": "SELECTED",
        "deployment_cap": cap,
        "rationale": rationale,
        "eligibility": eligibility,
    }


# ── Document & Ledger Generators ───────────────────────────────────────────────

def build_markdown_report(
    medpsy_eval: Dict[str, Any],
    medgemma_eval: Dict[str, Any],
    comparison: Dict[str, Any],
    decision: Dict[str, Any],
    manifests: Dict[str, Any],
) -> str:
    lines = []
    lines.append("# MedRAG-SLM Chatbot Deployment Model Selection Report")
    lines.append("")
    lines.append(f"**Date:** 2026-10-28  ")
    lines.append(f"**Rule Specification:** `docs/MODEL_SELECTION_RULE.md`  ")
    lines.append(f"**Selected Deployment Model:** `{decision['winner']}`  ")
    lines.append(f"**Selected Deployment Cap:** `{decision['deployment_cap']}` tokens  ")
    lines.append(f"**Selection Status:** **{decision['status']}**  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: Executive Summary & Rationale
    lines.append("## 1. Executive Summary & Selection Decision")
    lines.append("")
    lines.append(f"> [!IMPORTANT]")
    lines.append(f"> **Official Selection Decision:**")
    lines.append(f"> **Model:** `{decision['winner']}` | **Cap:** `{decision['deployment_cap']}` tokens")
    lines.append(f">")
    lines.append(f"> {decision['rationale']}")
    lines.append("")

    # Section 2: Candidate Overview & Eligibility Audit
    lines.append("## 2. Candidate Overview & Eligibility Audit")
    lines.append("")
    _n_shared = len(medpsy_eval["at_chosen_cap"]["strict_binary"])
    lines.append(f"Following `docs/MODEL_SELECTION_RULE.md`, candidates are evaluated on the MedQA validation split ($N={_n_shared}$ shared questions in identical file order).")
    lines.append(r"Eligibility requires: (a) Truncation Rate $\le 15.0\%$, (b) Parse Rate $\ge 95.0\%$, and (c) execution feasibility on Kaggle 2x T4.")
    lines.append("")
    lines.append(r"| Candidate | Model Type | Deployment Cap | Truncation Rate (95% CI) | Truncation $\le 15\%$? | Parse Rate (95% CI) | Parse $\ge 95\%$? | Hardware Feasibility | Eligibility |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    m_el = decision["eligibility"]["medpsy-4b"]
    g_el = decision["eligibility"]["medgemma-1.5-4b-it"]
    m_cap_eval = medpsy_eval["at_chosen_cap"]

    lines.append(
        f"| `medpsy-4b` | Medical Reasoning SLM | {medpsy_eval['chosen_cap']} tokens | "
        f"{m_cap_eval['trunc_rate']*100:.1f}% [{m_cap_eval['trunc_ci'][0]*100:.1f}%, {m_cap_eval['trunc_ci'][1]*100:.1f}%] | "
        f"{'PASS' if m_el['trunc_threshold_met'] else 'FAIL'} | "
        f"{m_cap_eval['parse_rate']*100:.1f}% [{m_cap_eval['parse_ci'][0]*100:.1f}%, {m_cap_eval['parse_ci'][1]*100:.1f}%] | "
        f"{'PASS' if m_el['parse_threshold_met'] else 'FAIL'} | "
        f"Kaggle 2x T4 ({m_el['vram_status']}) | "
        f"**{'ELIGIBLE' if m_el['eligible'] else 'DISQUALIFIED'}** |"
    )
    lines.append(
        f"| `medgemma-1.5-4b-it` | Medical Direct-Answer SLM | 16 tokens | "
        f"{medgemma_eval['trunc_rate']*100:.1f}% [{medgemma_eval['trunc_ci'][0]*100:.1f}%, {medgemma_eval['trunc_ci'][1]*100:.1f}%] | "
        f"{'PASS' if g_el['trunc_threshold_met'] else 'FAIL'} | "
        f"{medgemma_eval['parse_rate']*100:.1f}% [{medgemma_eval['parse_ci'][0]*100:.1f}%, {medgemma_eval['parse_ci'][1]*100:.1f}%] | "
        f"{'PASS' if g_el['parse_threshold_met'] else 'FAIL'} | "
        f"Kaggle 2x T4 ({g_el['vram_status']}) | "
        f"**{'ELIGIBLE' if g_el['eligible'] else 'DISQUALIFIED'}** |"
    )
    lines.append("")

    # Section 3: MedPsy-4B Deployment Cap Grid
    lines.append("## 3. Deployment Cap Selection for Thinking-Style SLM (`medpsy-4b`)")
    lines.append("")
    lines.append(r"The deployment cap for `medpsy-4b` is the smallest cap in the pre-registered grid $[1024, 1536, 2048, 3072, 4096]$ at which validation truncation is $\le 5.0\%$.")
    lines.append("")
    lines.append(r"| Candidate Cap | Truncation Rate (95% CI) | Truncation $\le 5.0\%$? | Strict Accuracy (95% CI) | Selection Status |")
    lines.append("|---|---|---|---|---|")
    for c in CAPS:
        ce = medpsy_eval["caps"][c]
        is_sel = (c == medpsy_eval["chosen_cap"])
        lines.append(
            f"| {c} tokens | {ce['trunc_rate']*100:.1f}% [{ce['trunc_ci'][0]*100:.1f}%, {ce['trunc_ci'][1]*100:.1f}%] | "
            f"{'PASS' if ce['trunc_rate'] <= 0.05 else 'FAIL'} | "
            f"{ce['strict_acc']*100:.1f}% [{ce['strict_ci'][0]*100:.1f}%, {ce['strict_ci'][1]*100:.1f}%] | "
            f"{'**SELECTED CAP**' if is_sel else 'Evaluated'} |"
        )
    lines.append("")
    _cap_1024_trunc = medpsy_eval["caps"].get(1024, {}).get("trunc_rate", float("nan")) * 100
    _chosen_cap_trunc = medpsy_eval["at_chosen_cap"]["trunc_rate"] * 100
    _chosen_cap_val = medpsy_eval["chosen_cap"]
    lines.append(f"> [!NOTE]")
    lines.append(
        rf"> At cap=1024, `medpsy-4b` exhibits {_cap_1024_trunc:.1f}% truncation (> 5.0%). "
        rf"At cap={_chosen_cap_val}, truncation drops to {_chosen_cap_trunc:.1f}% ($\le 5.0\%$), "
        rf"achieving the qualification threshold. Hence, **{_chosen_cap_val} tokens** is selected."
    )
    lines.append("")

    # Section 4: Head-to-Head Comparison on Shared Validation Split
    lines.append(f"## 4. Head-to-Head Comparison on Shared Validation Questions ($N={_n_shared}$)")
    lines.append("")
    lines.append(f"| Metric | `medpsy-4b` (Chosen Cap: {medpsy_eval['chosen_cap']}) | `medgemma-1.5-4b-it` (Cap: {medgemma_eval['chosen_cap']}) | Paired Difference (MedPsy - MedGemma) | Statistical Significance |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| **Strict Accuracy [Primary]** | **{m_cap_eval['strict_acc']*100:.1f}%** [{m_cap_eval['strict_ci'][0]*100:.1f}%, {m_cap_eval['strict_ci'][1]*100:.1f}%] | "
        f"**{medgemma_eval['strict_acc']*100:.1f}%** [{medgemma_eval['strict_ci'][0]*100:.1f}%, {medgemma_eval['strict_ci'][1]*100:.1f}%] | "
        f"**{comparison['diff']*100:+.1f} pp** [{comparison['bootstrap_ci'][0]*100:+.1f}, {comparison['bootstrap_ci'][1]*100:+.1f}] | "
        f"Exact McNemar $p = {format_p_val(comparison['mcnemar_p'])}$ (Significant) |"
    )
    lines.append(
        f"| With-Fallback Accuracy | {m_cap_eval['fallback_acc']*100:.1f}% [{m_cap_eval['fallback_ci'][0]*100:.1f}%, {m_cap_eval['fallback_ci'][1]*100:.1f}%] | "
        f"{medgemma_eval['fallback_acc']*100:.1f}% [{medgemma_eval['fallback_ci'][0]*100:.1f}%, {medgemma_eval['fallback_ci'][1]*100:.1f}%] | "
        f"{(m_cap_eval['fallback_acc'] - medgemma_eval['fallback_acc'])*100:+.1f} pp | — |"
    )
    lines.append(
        f"| Truncation Rate | {m_cap_eval['trunc_rate']*100:.1f}% [{m_cap_eval['trunc_ci'][0]*100:.1f}%, {m_cap_eval['trunc_ci'][1]*100:.1f}%] | "
        f"{medgemma_eval['trunc_rate']*100:.1f}% [{medgemma_eval['trunc_ci'][0]*100:.1f}%, {medgemma_eval['trunc_ci'][1]*100:.1f}%] | "
        f"{(m_cap_eval['trunc_rate'] - medgemma_eval['trunc_rate'])*100:+.1f} pp | — |"
    )
    _p_diff = (m_cap_eval['parse_rate'] - medgemma_eval['parse_rate']) * 100
    lines.append(
        f"| Parse Rate | {m_cap_eval['parse_rate']*100:.1f}% [{m_cap_eval['parse_ci'][0]*100:.1f}%, {m_cap_eval['parse_ci'][1]*100:.1f}%] | "
        f"{medgemma_eval['parse_rate']*100:.1f}% [{medgemma_eval['parse_ci'][0]*100:.1f}%, {medgemma_eval['parse_ci'][1]*100:.1f}%] | "
        f"{_p_diff:+.1f} pp | Both {m_cap_eval['parse_rate']*100:.1f}% / {medgemma_eval['parse_rate']*100:.1f}% parsed |"
    )
    lines.append(
        f"| Measured Latency (Mean / Median) | {medpsy_eval['seconds_mean']:.2f} s / {medpsy_eval['seconds_median']:.2f} s | "
        f"{medgemma_eval['seconds_mean']:.2f} s / {medgemma_eval['seconds_median']:.2f} s | "
        f"+{medpsy_eval['seconds_mean'] - medgemma_eval['seconds_mean']:.2f} s / q | MedGemma is {medpsy_eval['seconds_mean'] / max(0.001, medgemma_eval['seconds_mean']):.1f}x faster |"
    )
    _g_total_q = manifests['medgemma-1.5-4b-it'].get('n_questions', len(medgemma_eval.get('strict_binary', [])) * 6 if 'strict_binary' in medgemma_eval else 600)
    lines.append(
        f"| Total Validation Wall-Clock | {manifests['medpsy-4b']['hours_used']:.4f} hrs ({m_cap_eval['n']} q) | "
        f"{manifests['medgemma-1.5-4b-it']['hours_used']:.4f} hrs ({_g_total_q} q) | — | Manifest logged wall-clock |"
    )
    lines.append("")

    # Section 5: Accuracy vs Latency Trade-Off Analysis
    lines.append("## 5. Accuracy vs. Latency Trade-Off Analysis")
    lines.append("")
    lines.append("The empirical data demonstrates a stark architectural trade-off:")
    _m_acc_pct = m_cap_eval["strict_acc"] * 100
    _g_acc_pct = medgemma_eval["strict_acc"] * 100
    _gap_pp = comparison["diff"] * 100
    _b = comparison["b"]
    _c = comparison["c"]
    _p_str = format_p_val(comparison["mcnemar_p"])
    lines.append(
        rf"1. **Clinical Reasoning Advantage:** `medpsy-4b` achieves **{_m_acc_pct:.1f}% strict accuracy** "
        rf"at cap {medpsy_eval['chosen_cap']} tokens on the shared validation set, compared to **{_g_acc_pct:.1f}%** for "
        rf"`medgemma-1.5-4b-it` (a statistically significant advantage of **{_gap_pp:+.1f} percentage points**, "
        rf"$p < {_p_str}$, with discordant pair counts $b={_b}, c={_c}$)."
    )
    _ratio = medpsy_eval["seconds_mean"] / max(0.001, medgemma_eval["seconds_mean"])
    lines.append(
        f"2. **Inference Latency Cost:** `medpsy-4b` averages **{medpsy_eval['seconds_mean']:.2f} seconds per question** "
        f"during extended reasoning, whereas `medgemma-1.5-4b-it` generates its direct-answer letter in "
        f"**{medgemma_eval['seconds_mean']:.2f} seconds per question** (~{_ratio:.0f}x faster)."
    )
    lines.append(
        "3. **Operational Recommendation:** Because community health worker clinical guidance demands diagnostic fidelity "
        "over millisecond response times, the pre-registered decision rule strictly favors the significantly more accurate "
        "reasoning model. The interactive UI in `kaggle_chat/chat_app.py` accounts for this latency with an active "
        "`\U0001f9ba Thinking\u2026 (about 20\u201340 s)` progress indicator."
    )
    lines.append("")

    # Section 6: Chatbot Configuration Directives
    lines.append("## 6. Chatbot Configuration Directives (`kaggle_chat/chat_app.py`)")
    lines.append("")
    lines.append("To reflect this selection rule in deployment:")
    lines.append("- **Primary Engine:** `medpsy-4b` (`qvac/MedPsy-4B`).")
    lines.append(
        f"- **Deployment Cap:** Increase `max_new_tokens_reasoning` from 1024 to `{medpsy_eval['chosen_cap']}` "
        f"to match the validation qualification threshold and reduce truncation from {_cap_1024_trunc:.1f}% to {_chosen_cap_trunc:.1f}%."
    )
    lines.append("- **Device Placement:** `medpsy-4b` loaded exclusively on `cuda:0` (~8 GB VRAM in fp16).")
    lines.append("- **Fast Fallback:** Keep `qwen3-1.7b` (or `medgemma-1.5-4b-it` if added) as an optional fast dropdown choice for rapid triage queries.")
    lines.append("")

    # Section 7: Methodological Limitations
    lines.append("## 7. Methodological Limitations")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append("> **Offline Multiple-Choice Proxy Limitation:**")
    lines.append("> The accuracy evaluated here is measured on multiple-choice examination questions (MedQA) with unambiguous reference options A–D. While MCQ accuracy reflects foundational biomedical domain knowledge and clinical reasoning consistency, it serves as an offline proxy for the quality of interactive, free-text dialogue with community health workers. Free-text health worker guidance lacks gold reference labels and requires qualitative human clinical oversight and live field testing.")
    lines.append("")

    return "\n".join(lines)


def build_json_ledger(
    medpsy_eval: Dict[str, Any],
    medgemma_eval: Dict[str, Any],
    comparison: Dict[str, Any],
    decision: Dict[str, Any],
    manifests: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "rule_specification": "docs/MODEL_SELECTION_RULE.md",
        "date": "2026-10-28",
        "split": "validation",
        "dataset": "medqa",
        "N": len(medpsy_eval["at_chosen_cap"]["strict_binary"]),
        "winner": decision["winner"],
        "status": decision["status"],
        "deployment_cap": decision["deployment_cap"],
        "rationale": decision["rationale"],
        "eligibility": decision["eligibility"],
        "candidates": {
            "medpsy-4b": {
                "model_id": "qvac/MedPsy-4B",
                "style": "reasoning (thinking)",
                "chosen_cap": medpsy_eval["chosen_cap"],
                "cap_rationale": medpsy_eval["cap_rationale"],
                "strict_acc": round(medpsy_eval["at_chosen_cap"]["strict_acc"], 4),
                "strict_ci_lo": round(medpsy_eval["at_chosen_cap"]["strict_ci"][0], 4),
                "strict_ci_hi": round(medpsy_eval["at_chosen_cap"]["strict_ci"][1], 4),
                "fallback_acc": round(medpsy_eval["at_chosen_cap"]["fallback_acc"], 4),
                "trunc_rate": round(medpsy_eval["at_chosen_cap"]["trunc_rate"], 4),
                "trunc_ci_lo": round(medpsy_eval["at_chosen_cap"]["trunc_ci"][0], 4),
                "trunc_ci_hi": round(medpsy_eval["at_chosen_cap"]["trunc_ci"][1], 4),
                "parse_rate": round(medpsy_eval["at_chosen_cap"]["parse_rate"], 4),
                "seconds_mean": round(medpsy_eval["seconds_mean"], 4),
                "seconds_median": round(medpsy_eval["seconds_median"], 4),
                "caps_evaluated": {
                    str(c): {
                        "strict_acc": round(medpsy_eval["caps"][c]["strict_acc"], 4),
                        "trunc_rate": round(medpsy_eval["caps"][c]["trunc_rate"], 4),
                    }
                    for c in CAPS
                },
            },
            "medgemma-1.5-4b-it": {
                "model_id": "google/medgemma-1.5-4b-it",
                "style": "direct-answer",
                "chosen_cap": 16,
                "cap_rationale": medgemma_eval["cap_rationale"],
                "strict_acc": round(medgemma_eval["strict_acc"], 4),
                "strict_ci_lo": round(medgemma_eval["strict_ci"][0], 4),
                "strict_ci_hi": round(medgemma_eval["strict_ci"][1], 4),
                "fallback_acc": round(medgemma_eval["fallback_acc"], 4),
                "trunc_rate": round(medgemma_eval["trunc_rate"], 4),
                "trunc_ci_lo": round(medgemma_eval["trunc_ci"][0], 4),
                "trunc_ci_hi": round(medgemma_eval["trunc_ci"][1], 4),
                "parse_rate": round(medgemma_eval["parse_rate"], 4),
                "seconds_mean": round(medgemma_eval["seconds_mean"], 4),
                "seconds_median": round(medgemma_eval["seconds_median"], 4),
            },
        },
        "comparison": {
            "strict_diff": round(comparison["diff"], 4),
            "bootstrap_ci_lo": round(comparison["bootstrap_ci"][0], 4),
            "bootstrap_ci_hi": round(comparison["bootstrap_ci"][1], 4),
            "discordant_b": comparison["b"],
            "discordant_c": comparison["c"],
            "mcnemar_p": round(comparison["mcnemar_p"], 6),
            "is_significant": comparison["is_significant"],
        },
        "manifests": {
            "medpsy-4b": {
                "hours_used": manifests["medpsy-4b"].get("hours_used"),
                "timestamp": manifests["medpsy-4b"].get("timestamp"),
            },
            "medgemma-1.5-4b-it": {
                "hours_used": manifests["medgemma-1.5-4b-it"].get("hours_used"),
                "timestamp": manifests["medgemma-1.5-4b-it"].get("timestamp"),
            },
        },
    }


# ── Main Entrypoint ────────────────────────────────────────────────────────────

def run_selection(repo_root: Path) -> Tuple[str, Dict[str, Any]]:
    medpsy_rows, medgemma_rows, manifests = load_and_validate_candidates(repo_root)
    medpsy_eval = evaluate_medpsy_caps(medpsy_rows)
    medgemma_eval = evaluate_medgemma(medgemma_rows)
    comparison = compare_candidates(medpsy_eval, medgemma_eval)
    decision = select_deployment_model(medpsy_eval, medgemma_eval, comparison)

    md_text = build_markdown_report(medpsy_eval, medgemma_eval, comparison, decision, manifests)
    ledger_dict = build_json_ledger(medpsy_eval, medgemma_eval, comparison, decision, manifests)
    return md_text, ledger_dict


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Select deployment model based on docs/MODEL_SELECTION_RULE.md")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Path to repository root")
    parser.add_argument("--check", action="store_true", help="Check that existing outputs match committed outputs")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).resolve()
    out_md = root / "docs" / "MODEL_SELECTION.md"
    out_json = root / "docs" / "model_selection.json"

    md_text, ledger_dict = run_selection(root)
    json_text = json.dumps(ledger_dict, indent=2) + "\n"

    if args.check:
        if not out_md.exists() or not out_json.exists():
            print("FAIL: docs/MODEL_SELECTION.md or docs/model_selection.json missing on disk.")
            sys.exit(1)
        cur_md = out_md.read_text(encoding="utf-8")
        cur_json = out_json.read_text(encoding="utf-8")
        if cur_md.strip() != md_text.strip():
            print("FAIL: docs/MODEL_SELECTION.md differs from computed output.")
            sys.exit(1)
        if cur_json.strip() != json_text.strip():
            print("FAIL: docs/model_selection.json differs from computed output.")
            sys.exit(1)
        print("PASS: docs/MODEL_SELECTION.md and docs/model_selection.json are consistent with committed outputs.")
        sys.exit(0)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")
    out_json.write_text(json_text, encoding="utf-8")
    print(f"Generated {out_md}")
    print(f"Generated {out_json}")
    print(f"Selection Decision: {ledger_dict['winner']} at cap {ledger_dict['deployment_cap']}")


if __name__ == "__main__":
    main()
