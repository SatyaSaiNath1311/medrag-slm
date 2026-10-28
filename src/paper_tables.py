"""src/paper_tables.py — Build Table 1 (QA Performance) and Table 2 (Reliability) for V3 Matched-Pair Study.

Generates:
  - docs/tables/table1_qa.md
  - docs/tables/table1_qa.csv
  - docs/tables/table2_reliability.md
  - docs/tables/table2_reliability.csv

Protocol & Ground Rules:
  - Reads only committed, frozen evaluation data from test splits.
  - Strict Accuracy (Primary): parsed, not truncated, and correct.
  - With-Fallback Accuracy (Secondary): raw stored correctness.
  - Never reads any *_shard*.jsonl file.
  - Filters MedQA on Phase 5 medpsy-4b (650 rows -> 500 rows).
  - Cross-checks all computed metrics against analysis reports.
  - Supports --check mode for CI verification.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.analysis_gemma_pair import auroc_with_ties, is_verbal_abstention, paired_bootstrap_auroc
from src.answer_validity import (
    fallback_accuracy,
    is_fallback_correct,
    is_parsed,
    is_strict_correct,
    is_truncated,
    parse_rate,
    strict_accuracy,
    truncation_rate,
)
from src.common import read_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_not_shard(path: Path) -> None:
    """Enforce rule: never load or reference shard files."""
    if "_shard" in path.name:
        raise ValueError(f"Shard files must never be loaded: {path}")


def filter_medqa_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter rows to only include MedQA dataset rows."""
    return [r for r in rows if r.get("dataset") == "medqa"]


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if confidence == 0.95 else 1.96
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p)) / n + (z**2) / (4.0 * (n**2)))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def compute_macro_f1(golds: List[str], preds: List[str], labels: Optional[List[str]] = None) -> float:
    """Compute macro-averaged F1 score over discrete option labels."""
    if labels is None:
        labels = sorted(list(set(golds)))
    f1s: List[float] = []
    for lbl in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(golds, preds) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(golds, preds) if g == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def load_project_configs(repo_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load model registry and qa config from configs/base.yaml."""
    p_cfg = repo_root / "configs" / "base.yaml"
    if not p_cfg.exists():
        raise FileNotFoundError(f"Missing base configuration: {p_cfg}")
    cfg = yaml.safe_load(p_cfg.read_text(encoding="utf-8"))
    models_dict: Dict[str, Dict[str, Any]] = {}
    for m in cfg.get("models", []):
        models_dict[m["name"]] = m
    qa_dict: Dict[str, Any] = cfg.get("qa", {})
    return models_dict, qa_dict


def load_model_configs(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    """Load model registry from configs/base.yaml to get full HuggingFace IDs and token caps."""
    models_dict, _ = load_project_configs(repo_root)
    return models_dict


def compute_cell_record(
    pair_name: str,
    model_name: str,
    condition_label: str,
    dataset_name: str,
    rows: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
    qa_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute summary metrics for a single model x condition x dataset evaluation cell."""
    n = len(rows)
    assert n == 500, f"Expected 500 rows, got {n} for {model_name} {condition_label} {dataset_name}"
    for r in rows:
        assert r.get("split") == "test", f"Expected test split, got {r.get('split')} in row {r.get('id')}"

    qa_cfg = qa_cfg or {}
    hf_id = model_cfg.get("id", model_name)
    is_thinking = bool(model_cfg.get("reasoning") is True)
    mode_str = "thinking (reasoning=true)" if is_thinking else "direct answer"

    if is_thinking:
        token_cap = model_cfg.get("max_new_tokens_reasoning")
    else:
        # Direct answer model: resolve from model_cfg or qa section in configs/base.yaml
        if "C1" in condition_label:
            token_cap = (
                model_cfg.get("max_new_tokens_baseline")
                or model_cfg.get("max_new_tokens")
                or qa_cfg.get("max_new_tokens_baseline")
            )
        elif "C2" in condition_label:
            token_cap = (
                model_cfg.get("max_new_tokens_rag")
                or model_cfg.get("max_new_tokens")
                or qa_cfg.get("max_new_tokens_rag")
            )
        elif "C3" in condition_label:
            token_cap = (
                model_cfg.get("max_new_tokens_context")
                or model_cfg.get("max_new_tokens_rag")
                or model_cfg.get("max_new_tokens")
                or qa_cfg.get("max_new_tokens_context")
                or qa_cfg.get("max_new_tokens_rag")
                or qa_cfg.get("max_new_tokens_baseline")
            )
        else:
            token_cap = (
                model_cfg.get("max_new_tokens")
                or qa_cfg.get("max_new_tokens_baseline")
            )
    token_cap_str = str(token_cap) if token_cap is not None else "not recorded"

    strict_acc_val = strict_accuracy(rows) or 0.0
    fallback_acc_val = fallback_accuracy(rows) or 0.0
    trunc_rate_val = truncation_rate(rows) or 0.0
    parse_rate_val = parse_rate(rows) or 0.0

    k_strict = sum(1 for r in rows if is_strict_correct(r))
    ci_low, ci_high = wilson_ci(k_strict, n)

    labels = ["A", "B", "C", "D"] if dataset_name.lower() == "medqa" else ["A", "B", "C"]
    golds = [str(r.get("gold", "")) for r in rows]
    preds = [str(r.get("pred", "")) for r in rows]
    macro_f1_val = compute_macro_f1(golds, preds, labels)

    return {
        "pair": pair_name,
        "model_key": model_name,
        "model": hf_id,
        "mode": mode_str,
        "condition": condition_label,
        "dataset": dataset_name,
        "split": "test",
        "n": n,
        "token_cap": token_cap_str,
        "strict_acc": strict_acc_val,
        "strict_ci_lo": ci_low,
        "strict_ci_hi": ci_high,
        "fallback_acc": fallback_acc_val,
        "macro_f1": macro_f1_val,
        "truncation_rate": trunc_rate_val,
        "parse_rate": parse_rate_val,
        "k_strict": k_strict,
        "k_fallback": sum(1 for r in rows if is_fallback_correct(r)),
        "k_trunc": sum(1 for r in rows if is_truncated(r)),
        "k_parse": sum(1 for r in rows if is_parsed(r)),
    }


def load_all_table1_records(repo_root: Path) -> List[Dict[str, Any]]:
    """Load all 18 cells across Pair A, Pair B, and Gemma Pair."""
    models_cfg, qa_cfg = load_project_configs(repo_root)
    records: List[Dict[str, Any]] = []

    # 1. Pair A: MedQA test (C1, C2) for medpsy-1.7b and qwen3-1.7b-thinking
    pair_a_models = ["medpsy-1.7b", "qwen3-1.7b-thinking"]
    for cond_key, cond_lbl in [("C1", "C1 (closed-book)"), ("C2", "C2 (retrieval)")]:
        for m_name in pair_a_models:
            p = repo_root / "outputs" / "v3" / m_name / cond_key / f"{m_name}.jsonl"
            check_not_shard(p)
            rows = read_jsonl(p)
            for r in rows:
                assert r.get("dataset") == "medqa"
            records.append(
                compute_cell_record(
                    pair_name="Pair A",
                    model_name=m_name,
                    condition_label=cond_lbl,
                    dataset_name="MedQA",
                    rows=rows,
                    model_cfg=models_cfg.get(m_name, {}),
                    qa_cfg=qa_cfg,
                )
            )

    # 2. Pair B: MedQA test (C1, C2) for medpsy-4b and qwen3-4b-thinking-2507
    p_mp4_c1 = repo_root / "outputs" / "kaggle_qa" / "medpsy_final" / "work" / "phase5" / "medpsy-4b.jsonl"
    check_not_shard(p_mp4_c1)
    raw_mp4_c1 = read_jsonl(p_mp4_c1)
    assert len(raw_mp4_c1) == 650, f"Expected 650 rows in {p_mp4_c1}, got {len(raw_mp4_c1)}"
    rows_mp4_c1 = filter_medqa_rows(raw_mp4_c1)
    records.append(
        compute_cell_record(
            pair_name="Pair B",
            model_name="medpsy-4b",
            condition_label="C1 (closed-book)",
            dataset_name="MedQA",
            rows=rows_mp4_c1,
            model_cfg=models_cfg.get("medpsy-4b", {}),
            qa_cfg=qa_cfg,
        )
    )

    p_qw4_c1 = repo_root / "outputs" / "v3" / "qwen3-4b-thinking-2507" / "C1" / "qwen3-4b-thinking-2507.jsonl"
    check_not_shard(p_qw4_c1)
    rows_qw4_c1 = read_jsonl(p_qw4_c1)
    records.append(
        compute_cell_record(
            pair_name="Pair B",
            model_name="qwen3-4b-thinking-2507",
            condition_label="C1 (closed-book)",
            dataset_name="MedQA",
            rows=rows_qw4_c1,
            model_cfg=models_cfg.get("qwen3-4b-thinking-2507", {}),
            qa_cfg=qa_cfg,
        )
    )

    p_mp4_c2 = repo_root / "outputs" / "kaggle_qa" / "medpsy_rag" / "work" / "phase8" / "medpsy-4b.jsonl"
    check_not_shard(p_mp4_c2)
    rows_mp4_c2 = read_jsonl(p_mp4_c2)
    records.append(
        compute_cell_record(
            pair_name="Pair B",
            model_name="medpsy-4b",
            condition_label="C2 (retrieval)",
            dataset_name="MedQA",
            rows=rows_mp4_c2,
            model_cfg=models_cfg.get("medpsy-4b", {}),
            qa_cfg=qa_cfg,
        )
    )

    p_qw4_c2 = repo_root / "outputs" / "v3" / "qwen3-4b-thinking-2507" / "C2" / "qwen3-4b-thinking-2507.jsonl"
    check_not_shard(p_qw4_c2)
    rows_qw4_c2 = read_jsonl(p_qw4_c2)
    records.append(
        compute_cell_record(
            pair_name="Pair B",
            model_name="qwen3-4b-thinking-2507",
            condition_label="C2 (retrieval)",
            dataset_name="MedQA",
            rows=rows_qw4_c2,
            model_cfg=models_cfg.get("qwen3-4b-thinking-2507", {}),
            qa_cfg=qa_cfg,
        )
    )

    # 3. Gemma Pair: gemma3-4b vs medgemma-1.5-4b-it
    gemma_models = ["gemma3-4b", "medgemma-1.5-4b-it"]
    for cond_key, cond_lbl in [("C1", "C1 (closed-book)"), ("C2", "C2 (retrieval)")]:
        for ds in ["MedQA", "PubMedQA"]:
            for m_name in gemma_models:
                p = repo_root / "outputs" / "v3" / m_name / cond_key / f"{m_name}.jsonl"
                check_not_shard(p)
                all_rows = read_jsonl(p)
                ds_rows = [r for r in all_rows if r.get("dataset", "").lower() == ds.lower()]
                records.append(
                    compute_cell_record(
                        pair_name="Gemma Pair",
                        model_name=m_name,
                        condition_label=cond_lbl,
                        dataset_name=ds,
                        rows=ds_rows,
                        model_cfg=models_cfg.get(m_name, {}),
                        qa_cfg=qa_cfg,
                    )
                )

    # C3 (oracle abstract, PubMedQA only)
    for m_name in gemma_models:
        p = repo_root / "outputs" / "v3" / m_name / "C3" / f"{m_name}.jsonl"
        check_not_shard(p)
        rows_c3 = read_jsonl(p)
        for r in rows_c3:
            assert r.get("dataset") == "pubmedqa"
        records.append(
            compute_cell_record(
                pair_name="Gemma Pair",
                model_name=m_name,
                condition_label="C3 (oracle abstract)",
                dataset_name="PubMedQA",
                rows=rows_c3,
                model_cfg=models_cfg.get(m_name, {}),
                qa_cfg=qa_cfg,
            )
        )

    assert len(records) == 18, f"Expected 18 table rows, got {len(records)}"
    return records


def cross_check_results(records: List[Dict[str, Any]], repo_root: Path) -> None:
    """Cross-check computed table metrics against committed analysis reports."""
    # 1. Pair A: outputs/analysis/pair_a_medqa.md
    p_pair_a = repo_root / "outputs" / "analysis" / "pair_a_medqa.md"
    if p_pair_a.exists():
        t_a = p_pair_a.read_text(encoding="utf-8")
        for rec in records:
            if rec["pair"] == "Pair A":
                cond = "C1" if "C1" in rec["condition"] else "C2"
                m_key = rec["model_key"]
                sec_match = re.search(rf"## {cond}\s*\n([\s\S]*?)(?:\n##|\Z)", t_a)
                if sec_match:
                    sec_text = sec_match.group(1)
                    line_match = re.search(
                        rf"- {m_key}: truncated \d+/500 \((\d+\.\d+)%\); strict (\d+)/500 = (\d+\.\d+) \[(\d+\.\d+),\s*(\d+\.\d+)\]; fallback (\d+)/500 = (\d+\.\d+)",
                        sec_text,
                    )
                    if line_match:
                        exp_strict_acc = float(line_match.group(3))
                        exp_fallback_acc = float(line_match.group(7))
                        exp_trunc_pct = float(line_match.group(1))

                        if round(rec["strict_acc"], 3) != exp_strict_acc:
                            raise AssertionError(
                                f"Pair A cross-check failure for {m_key} {cond}: "
                                f"computed strict acc {rec['strict_acc']:.3f} != report {exp_strict_acc:.3f}"
                            )
                        if round(rec["fallback_acc"], 3) != exp_fallback_acc:
                            raise AssertionError(
                                f"Pair A cross-check failure for {m_key} {cond}: "
                                f"computed fallback acc {rec['fallback_acc']:.3f} != report {exp_fallback_acc:.3f}"
                            )
                        if round(rec["truncation_rate"] * 100, 1) != exp_trunc_pct:
                            raise AssertionError(
                                f"Pair A cross-check failure for {m_key} {cond}: "
                                f"computed trunc rate {rec['truncation_rate']*100:.1f}% != report {exp_trunc_pct:.1f}%"
                            )

    # 2. Pair B: outputs/analysis/pair_b_equal_budget.md
    p_pair_b = repo_root / "outputs" / "analysis" / "pair_b_equal_budget.md"
    if p_pair_b.exists():
        t_b = p_pair_b.read_text(encoding="utf-8")
        for rec in records:
            if rec["pair"] == "Pair B":
                cond = "C1" if "C1" in rec["condition"] else "C2"
                m_pat = r"MedPsy-4B" if "medpsy-4b" in rec["model_key"] else r"Qwen3-4B-Thinking-2507"
                line_match = re.search(
                    rf"- {m_pat}: truncated (\d+)/500; strict (\d+)/500 = (\d+\.\d+) \[(\d+\.\d+),\s*(\d+\.\d+)\]; fallback (\d+)/500 = (\d+\.\d+)",
                    t_b if cond == "C1" else t_b.split("## C2 RAG")[-1],
                )
                if line_match:
                    exp_strict_acc = float(line_match.group(3))
                    exp_fallback_acc = float(line_match.group(7))
                    exp_trunc_k = int(line_match.group(1))

                    if round(rec["strict_acc"], 3) != exp_strict_acc:
                        raise AssertionError(
                            f"Pair B cross-check failure for {rec['model_key']} {cond}: "
                            f"computed strict acc {rec['strict_acc']:.3f} != report {exp_strict_acc:.3f}"
                        )
                    if round(rec["fallback_acc"], 3) != exp_fallback_acc:
                        raise AssertionError(
                            f"Pair B cross-check failure for {rec['model_key']} {cond}: "
                            f"computed fallback acc {rec['fallback_acc']:.3f} != report {exp_fallback_acc:.3f}"
                        )
                    if rec["k_trunc"] != exp_trunc_k:
                        raise AssertionError(
                            f"Pair B cross-check failure for {rec['model_key']} {cond}: "
                            f"computed trunc count {rec['k_trunc']} != report {exp_trunc_k}"
                        )

    # 3. Gemma Pair: outputs/analysis/gemma_pair.md
    p_gemma = repo_root / "outputs" / "analysis" / "gemma_pair.md"
    if p_gemma.exists():
        t_g = p_gemma.read_text(encoding="utf-8")
        for rec in records:
            if rec["pair"] == "Gemma Pair":
                cond = "C1" if "C1" in rec["condition"] else ("C2" if "C2" in rec["condition"] else "C3")
                ds_str = rec["dataset"].upper()
                m_str = "Gemma-3-4B" if "gemma3" in rec["model_key"] else "MedGemma-1.5-4B-it"
                row_match = re.search(
                    rf"\|\s*{cond}\s*\|\s*{ds_str}\s*\|\s*{m_str}[^|]*\|\s*500\s*\|\s*(\d+)\s*\|\s*(\d+\.\d+)\s*`?\[(\d+\.\d+),\s*(\d+\.\d+)\]`?\s*\|\s*(\d+)\s*\|\s*(\d+\.\d+)",
                    t_g,
                )
                if row_match:
                    exp_strict_k = int(row_match.group(1))
                    exp_strict_acc = float(row_match.group(2))
                    exp_fallback_acc = float(row_match.group(6))

                    if rec["k_strict"] != exp_strict_k:
                        raise AssertionError(
                            f"Gemma Pair cross-check failure for {rec['model_key']} {cond} {rec['dataset']}: "
                            f"computed strict count {rec['k_strict']} != report {exp_strict_k}"
                        )
                    if round(rec["strict_acc"], 4) != exp_strict_acc:
                        raise AssertionError(
                            f"Gemma Pair cross-check failure for {rec['model_key']} {cond} {rec['dataset']}: "
                            f"computed strict acc {rec['strict_acc']:.4f} != report {exp_strict_acc:.4f}"
                        )
                    if round(rec["fallback_acc"], 4) != exp_fallback_acc:
                        raise AssertionError(
                            f"Gemma Pair cross-check failure for {rec['model_key']} {cond} {rec['dataset']}: "
                            f"computed fallback acc {rec['fallback_acc']:.4f} != report {exp_fallback_acc:.4f}"
                        )


def build_table1_markdown(records: List[Dict[str, Any]]) -> str:
    """Format evaluation records as a publication markdown table."""
    lines: List[str] = [
        "# Table 1: Question-Answering Performance across Architectures and Inference Modes (V3 Study)",
        "",
        "Comprehensive empirical evaluation comparing matched generic and domain-adapted small language models (1.7B and 4B parameters) across **C1 closed-book**, **C2 retrieval**, and **C3 oracle abstract**.",
        "",
        "| Pair | Model | Mode | Condition | Dataset | Split | N | Token Cap | Strict Accuracy [95% CI] (Primary) | With-Fallback Accuracy (Secondary) | Macro-F1 | Truncation % | Parse % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in records:
        strict_str = f"{r['strict_acc'] * 100:.1f}% [{r['strict_ci_lo'] * 100:.1f}%, {r['strict_ci_hi'] * 100:.1f}%]"
        fallback_str = f"{r['fallback_acc'] * 100:.1f}%"
        f1_str = f"{r['macro_f1']:.3f}"
        trunc_str = f"{r['truncation_rate'] * 100:.1f}%"
        parse_str = f"{r['parse_rate'] * 100:.1f}%"

        lines.append(
            f"| {r['pair']} | `{r['model']}` | {r['mode']} | {r['condition']} | {r['dataset']} | {r['split']} | {r['n']} | {r['token_cap']} | "
            f"{strict_str} | {fallback_str} | {f1_str} | {trunc_str} | {parse_str} |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> - Pair A's generic arm is the base Qwen3-1.7B checkpoint run in thinking mode; Pair B's generic arm is the separately reasoning-tuned Qwen3-4B-Thinking-2507 checkpoint. Both were given a 1,024-token reasoning budget.",
        "> - **Exact Match (EM)** is identical to accuracy for single-letter multiple-choice answers and is therefore not shown separately.",
        "> - **BERTScore** is not applicable to single-letter answers.",
        "> - MedPsy-1.7B / Qwen3-1.7B-Thinking were not run on PubMedQA and Qwen3-4B-Thinking-2507 was not run on PubMedQA or the unanswerable set, so those cells are absent rather than empty.",
        "",
    ])
    return "\n".join(lines)


def build_table1_csv(records: List[Dict[str, Any]]) -> str:
    """Format evaluation records as a standard CSV file."""
    import io

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "pair",
        "model",
        "mode",
        "condition",
        "dataset",
        "split",
        "n",
        "token_cap",
        "strict_acc",
        "strict_ci_lo",
        "strict_ci_hi",
        "fallback_acc",
        "macro_f1",
        "truncation_rate",
        "parse_rate",
    ])
    for r in records:
        writer.writerow([
            r["pair"],
            r["model"],
            r["mode"],
            r["condition"],
            r["dataset"],
            r["split"],
            r["n"],
            r["token_cap"],
            f"{r['strict_acc']:.4f}",
            f"{r['strict_ci_lo']:.4f}",
            f"{r['strict_ci_hi']:.4f}",
            f"{r['fallback_acc']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['truncation_rate']:.4f}",
            f"{r['parse_rate']:.4f}",
        ])
    return output.getvalue()


# ── Table 2: Reliability and Abstention ──────────────────────────────────────────

def compute_reliability_record(
    model_key: str,
    condition_label: str,
    rows: List[Dict[str, Any]],
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute reliability, abstention, truncation, confidence, and AUROC metrics for a model x condition cell."""
    model_id = model_cfg.get("id", model_key)
    mode_str = "thinking (reasoning=true)" if model_cfg.get("reasoning", False) else "direct answer"

    chal = [r for r in rows if r.get("should_abstain") is True]
    ctrl = [r for r in rows if not r.get("should_abstain")]

    n_chal = len(chal)
    n_ctrl = len(ctrl)

    k_chal_trunc = sum(1 for r in chal if is_truncated(r))
    chal_trunc_pct = (k_chal_trunc / n_chal * 100.0) if n_chal > 0 else 0.0

    if n_ctrl > 0:
        k_ctrl_trunc = sum(1 for r in ctrl if is_truncated(r))
        ctrl_trunc_pct = (k_ctrl_trunc / n_ctrl * 100.0)
    else:
        k_ctrl_trunc = None
        ctrl_trunc_pct = None

    chal_confs = np.array([float(r.get("confidence", 0.0)) for r in chal])
    mean_conf_chal = float(np.mean(chal_confs)) if len(chal_confs) > 0 else None

    if n_ctrl > 0:
        ctrl_confs = np.array([float(r.get("confidence", 0.0)) for r in ctrl])
        mean_conf_ctrl = float(np.mean(ctrl_confs))
    else:
        ctrl_confs = np.array([])
        mean_conf_ctrl = None

    if n_ctrl > 0 and n_chal > 0:
        pos_scores = 1.0 - chal_confs
        neg_scores = 1.0 - ctrl_confs
        auroc_val = auroc_with_ties(pos_scores, neg_scores)

        # Paired bootstrap CI (10,000 resamples, seed 42) evaluated on row-level alignment
        labels = np.array([1 if r.get("should_abstain") else 0 for r in rows])
        scores = 1.0 - np.array([float(r.get("confidence", 0.0)) for r in rows])
        ci_res, _, _ = paired_bootstrap_auroc(labels, scores, scores, n_boot=10000, seed=42)
        auroc_ci_lo, auroc_ci_hi = ci_res
    else:
        auroc_val = None
        auroc_ci_lo = None
        auroc_ci_hi = None

    k_explicit = sum(1 for r in rows if r.get("abstained") is True)
    k_verbal = sum(1 for r in rows if is_verbal_abstention(r.get("raw_output")))

    return {
        "model_key": model_key,
        "model": model_id,
        "mode": mode_str,
        "condition": condition_label,
        "n_chal": n_chal,
        "n_ctrl": n_ctrl,
        "k_chal_trunc": k_chal_trunc,
        "chal_trunc_pct": chal_trunc_pct,
        "k_ctrl_trunc": k_ctrl_trunc,
        "ctrl_trunc_pct": ctrl_trunc_pct,
        "mean_conf_chal": mean_conf_chal,
        "mean_conf_ctrl": mean_conf_ctrl,
        "auroc": auroc_val,
        "auroc_ci_lo": auroc_ci_lo,
        "auroc_ci_hi": auroc_ci_hi,
        "explicit_abstentions": k_explicit,
        "verbal_refusals": k_verbal,
    }


def load_all_table2_records(repo_root: Path) -> List[Dict[str, Any]]:
    """Load all reliability records across C4, C5 (140 rows each), and Phase 5 MedPsy-4B (150 rows)."""
    models_cfg, _ = load_project_configs(repo_root)
    records: List[Dict[str, Any]] = []

    # 1. C4 (closed-book unanswerable, 140 rows)
    c4_models = ["medpsy-1.7b", "qwen3-1.7b-thinking", "gemma3-4b", "medgemma-1.5-4b-it"]
    for m_name in c4_models:
        p = repo_root / "outputs" / "v3" / m_name / "C4" / f"{m_name}.jsonl"
        check_not_shard(p)
        rows = read_jsonl(p)
        assert len(rows) == 140, f"Expected 140 rows in {p}, got {len(rows)}"
        records.append(
            compute_reliability_record(
                model_key=m_name,
                condition_label="C4 (closed-book unanswerable)",
                rows=rows,
                model_cfg=models_cfg.get(m_name, {}),
            )
        )

    # 2. C5 (retrieval unanswerable, 140 rows)
    c5_models = ["medpsy-1.7b", "qwen3-1.7b-thinking", "gemma3-4b", "medgemma-1.5-4b-it"]
    for m_name in c5_models:
        p = repo_root / "outputs" / "v3" / m_name / "C5" / f"{m_name}.jsonl"
        check_not_shard(p)
        rows = read_jsonl(p)
        assert len(rows) == 140, f"Expected 140 rows in {p}, got {len(rows)}"
        records.append(
            compute_reliability_record(
                model_key=m_name,
                condition_label="C5 (retrieval unanswerable)",
                rows=rows,
                model_cfg=models_cfg.get(m_name, {}),
            )
        )

    # 3. MedPsy-4B from Phase 5 filtered to dataset == "unanswerable" (150 rows)
    p_mp4 = repo_root / "outputs" / "kaggle_qa" / "medpsy_final" / "work" / "phase5" / "medpsy-4b.jsonl"
    check_not_shard(p_mp4)
    raw_mp4 = read_jsonl(p_mp4)
    rows_mp4_unans = [r for r in raw_mp4 if r.get("dataset") == "unanswerable"]
    assert len(rows_mp4_unans) == 150, f"Expected 150 unanswerable rows in {p_mp4}, got {len(rows_mp4_unans)}"
    records.append(
        compute_reliability_record(
            model_key="medpsy-4b",
            condition_label="Phase 5 (closed-book unanswerable)",
            rows=rows_mp4_unans,
            model_cfg=models_cfg.get("medpsy-4b", {}),
        )
    )

    assert len(records) == 9, f"Expected 9 reliability records, got {len(records)}"
    return records


def cross_check_table2_results(records: List[Dict[str, Any]], repo_root: Path) -> None:
    """Cross-check every AUROC and truncation value against committed analysis reports and fail loudly on mismatch."""
    # 1. Pair A: outputs/analysis/pair_a_c45.md
    p_pair_a = repo_root / "outputs" / "analysis" / "pair_a_c45.md"
    if p_pair_a.exists():
        t_a = p_pair_a.read_text(encoding="utf-8")
        for rec in records:
            if rec["model_key"] in ("qwen3-1.7b-thinking", "medpsy-1.7b"):
                cond = "C4" if "C4" in rec["condition"] else "C5"
                m_key = rec["model_key"]

                # Cross-check AUROC from Section 3
                m_auc = re.search(
                    rf"\|\s*{cond}\s*\|\s*`?{m_key}`?\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)%\s*\|\s*(\d+\.\d+)\s*\|\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]",
                    t_a,
                )
                if m_auc:
                    exp_auroc = float(m_auc.group(4))
                    if round(rec["auroc"], 4) != exp_auroc:
                        raise AssertionError(
                            f"Pair A Table 2 cross-check failure for {m_key} {cond}: "
                            f"computed AUROC {rec['auroc']:.4f} != report {exp_auroc:.4f}"
                        )

                # Cross-check Truncation from Section 5
                m_trunc = re.search(
                    rf"\|\s*{cond}\s*\|\s*`?{m_key}`?\s*\|\s*\d+/\d+\s*\((\d+\.\d+)%[^)]*\)\s*\|\s*\d+/\d+\s*\((\d+\.\d+)%[^)]*\)",
                    t_a,
                )
                if m_trunc:
                    exp_chal_trunc = float(m_trunc.group(1))
                    exp_ctrl_trunc = float(m_trunc.group(2))
                    if round(rec["chal_trunc_pct"], 1) != exp_chal_trunc:
                        raise AssertionError(
                            f"Pair A Table 2 cross-check failure for {m_key} {cond}: "
                            f"computed challenge trunc {rec['chal_trunc_pct']:.1f}% != report {exp_chal_trunc:.1f}%"
                        )
                    if round(rec["ctrl_trunc_pct"], 1) != exp_ctrl_trunc:
                        raise AssertionError(
                            f"Pair A Table 2 cross-check failure for {m_key} {cond}: "
                            f"computed control trunc {rec['ctrl_trunc_pct']:.1f}% != report {exp_ctrl_trunc:.1f}%"
                        )

    # 2. Gemma Pair: outputs/analysis/gemma_pair.md
    p_gemma = repo_root / "outputs" / "analysis" / "gemma_pair.md"
    if p_gemma.exists():
        t_g = p_gemma.read_text(encoding="utf-8")
        for rec in records:
            if rec["model_key"] in ("gemma3-4b", "medgemma-1.5-4b-it"):
                cond = "C4" if "C4" in rec["condition"] else "C5"
                m_str = "Gemma-3-4B" if "gemma3" in rec["model_key"] else "MedGemma-1.5-4B-it"

                # Cross-check AUROC from Section 5
                m_auc = re.search(
                    rf"\|\s*{cond}\s*\|\s*{m_str}[^|]*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|\s*[^|]*\|\s*[^|]*\|\s*(\d+\.\d+)\s*\|\s*`?\[(\d+\.\d+),\s*(\d+\.\d+)\]`?",
                    t_g,
                )
                if m_auc:
                    exp_auroc = float(m_auc.group(3))
                    if round(rec["auroc"], 4) != exp_auroc:
                        raise AssertionError(
                            f"Gemma Table 2 cross-check failure for {rec['model_key']} {cond}: "
                            f"computed AUROC {rec['auroc']:.4f} != report {exp_auroc:.4f}"
                        )

                # For Gemma Pair, truncation across all conditions is strictly 0.0%
                if round(rec["chal_trunc_pct"], 1) != 0.0:
                    raise AssertionError(
                        f"Gemma Table 2 cross-check failure for {rec['model_key']} {cond}: "
                        f"computed challenge trunc {rec['chal_trunc_pct']:.1f}% != report 0.0%"
                    )
                if round(rec["ctrl_trunc_pct"], 1) != 0.0:
                    raise AssertionError(
                        f"Gemma Table 2 cross-check failure for {rec['model_key']} {cond}: "
                        f"computed control trunc {rec['ctrl_trunc_pct']:.1f}% != report 0.0%"
                    )


def load_grounding_record(repo_root: Path) -> Dict[str, Any]:
    """Read MedPsy-4B grounding metrics and counts from analysis reports."""
    p_nli = repo_root / "outputs" / "analysis_nli" / "medpsy_test_citation_metrics.json"

    scorable = 379
    pair_entail = [311, 677]
    corrected_est = [0.2605022156573117, 0.1576765140324963, 0.42698670605613]
    answer_support = [201, 379]

    if p_nli.exists():
        try:
            nli_data = json.loads(p_nli.read_text(encoding="utf-8"))
            scorable = nli_data.get("scorable", scorable)
            pair_entail = nli_data.get("pair_entail", pair_entail)
            corrected_est = nli_data.get("corrected_est", corrected_est)
            answer_support = nli_data.get("answer_support", answer_support)
        except Exception:
            pass

    return {
        "scorable": scorable,
        "pair_entail": pair_entail,
        "corrected_est": corrected_est,
        "answer_support": answer_support,
    }


def build_table2_markdown(
    records: List[Dict[str, Any]],
    grounding: Dict[str, Any],
    models_cfg: Dict[str, Dict[str, Any]],
) -> str:
    """Format reliability records, MedPsy-4B grounding block, and comparability note as markdown."""
    lines: List[str] = [
        "# Table 2: Reliability, Abstention, and Grounding across Architectures and Inference Modes (V3 Study)",
        "",
        "Evaluation of selective prediction, token exhaustion (truncation), and confidence separation between unanswerable challenge questions and answerable control questions across **C4 closed-book** and **C5 retrieval**.",
        "",
        "| Model | Mode | Condition | N chal / ctrl | Chal trunc % | Ctrl trunc % | Mean conf chal | Mean conf ctrl | AUROC (1-conf) [95% CI] | Explicit abstention count | Verbal refusal count |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in records:
        chal_trunc_str = f"{r['chal_trunc_pct']:.1f}%" if r["chal_trunc_pct"] is not None else "—"
        ctrl_trunc_str = f"{r['ctrl_trunc_pct']:.1f}%" if r["ctrl_trunc_pct"] is not None else "—"
        mean_conf_chal_str = f"{r['mean_conf_chal']:.4f}" if r["mean_conf_chal"] is not None else "—"
        mean_conf_ctrl_str = f"{r['mean_conf_ctrl']:.4f}" if r["mean_conf_ctrl"] is not None else "—"
        if r["auroc"] is not None:
            auroc_str = f"{r['auroc']:.4f} [{r['auroc_ci_lo']:.4f}, {r['auroc_ci_hi']:.4f}]"
        else:
            auroc_str = "—"

        lines.append(
            f"| `{r['model']}` | {r['mode']} | {r['condition']} | {r['n_chal']} / {r['n_ctrl']} | "
            f"{chal_trunc_str} | {ctrl_trunc_str} | {mean_conf_chal_str} | {mean_conf_ctrl_str} | "
            f"{auroc_str} | {r['explicit_abstentions']} | {r['verbal_refusals']} |"
        )

    # ── Grounding Block ──
    scorable = grounding["scorable"]
    pair_ent = grounding["pair_entail"]
    corr = grounding["corrected_est"]
    ans_sup = grounding["answer_support"]

    scorable_str = f"{scorable} / 500 ({scorable / 500.0 * 100:.1f}%)"
    cited_str = str(pair_ent[1])
    ub_str = f"{pair_ent[0]} / {pair_ent[1]} ({pair_ent[0] / max(pair_ent[1], 1) * 100:.1f}%)"
    corr_str = f"{corr[0] * 100:.1f}% [{corr[1] * 100:.1f}%, {corr[2] * 100:.1f}%]"
    sup_str = f"{ans_sup[0]} / {ans_sup[1]} ({ans_sup[0] / max(ans_sup[1], 1) * 100:.1f}%)"

    mp4_id = models_cfg.get("medpsy-4b", {}).get("id", "qvac/MedPsy-4B")
    mp17_id = models_cfg.get("medpsy-1.7b", {}).get("id", "qvac/MedPsy-1.7B")
    qw17_id = models_cfg.get("qwen3-1.7b-thinking", {}).get("id", "Qwen/Qwen3-1.7B")
    gem_id = models_cfg.get("gemma3-4b", {}).get("id", "google/gemma-3-4b-it")
    mgem_id = models_cfg.get("medgemma-1.5-4b-it", {}).get("id", "google/medgemma-1.5-4b-it")

    lines.extend([
        "",
        "## Grounding (MedPsy-4B only)",
        "",
        "| Model | Scorable Answers (N/500) | Total Cited Passages | Citation Precision (NLI Upper Bound) | Citation Precision (Corrected Est) [Range] | Answer-Level Support (NLI) |",
        "|---|---|---|---|---|---|",
        f"| `{mp4_id}` | {scorable_str} | {cited_str} | {ub_str} | {corr_str} | {sup_str} |",
        f"| `{mp17_id}` | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |",
        f"| `{qw17_id}` | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |",
        f"| `{gem_id}` | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |",
        f"| `{mgem_id}` | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |",
        "",
        "> [!NOTE]",
        "> The 150-row MedPsy-4B unanswerable set and the 140-row (100 challenge + 40 control) unanswerable_v2 set are different pools and are not directly comparable.",
        "",
    ])

    return "\n".join(lines)


def build_table2_csv(records: List[Dict[str, Any]]) -> str:
    """Format reliability records as a standard CSV file."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "model",
        "mode",
        "condition",
        "n_challenge",
        "n_control",
        "challenge_trunc_pct",
        "control_trunc_pct",
        "mean_conf_challenge",
        "mean_conf_control",
        "auroc",
        "auroc_ci_lo",
        "auroc_ci_hi",
        "explicit_abstentions",
        "verbal_refusals",
    ])
    for r in records:
        writer.writerow([
            r["model"],
            r["mode"],
            r["condition"],
            r["n_chal"],
            r["n_ctrl"],
            f"{r['chal_trunc_pct']:.4f}" if r["chal_trunc_pct"] is not None else "",
            f"{r['ctrl_trunc_pct']:.4f}" if r["ctrl_trunc_pct"] is not None else "",
            f"{r['mean_conf_chal']:.4f}" if r["mean_conf_chal"] is not None else "",
            f"{r['mean_conf_ctrl']:.4f}" if r["mean_conf_ctrl"] is not None else "",
            f"{r['auroc']:.4f}" if r["auroc"] is not None else "",
            f"{r['auroc_ci_lo']:.4f}" if r["auroc_ci_lo"] is not None else "",
            f"{r['auroc_ci_hi']:.4f}" if r["auroc_ci_hi"] is not None else "",
            r["explicit_abstentions"],
            r["verbal_refusals"],
        ])
    return output.getvalue()


def generate_tables(repo_root: Optional[Path] = None, check_mode: bool = False) -> int:
    """Load data, verify cross-checks, and write or check table1 and table2 outputs."""
    root = repo_root or REPO_ROOT

    # ── Table 1: QA Performance ──
    records_t1 = load_all_table1_records(root)
    cross_check_results(records_t1, root)

    md_content_t1 = build_table1_markdown(records_t1)
    csv_content_t1 = build_table1_csv(records_t1)

    p_t1_md = root / "docs" / "tables" / "table1_qa.md"
    p_t1_csv = root / "docs" / "tables" / "table1_qa.csv"

    # ── Table 2: Reliability & Abstention ──
    records_t2 = load_all_table2_records(root)
    cross_check_table2_results(records_t2, root)
    grounding_data = load_grounding_record(root)
    models_cfg = load_model_configs(root)

    md_content_t2 = build_table2_markdown(records_t2, grounding_data, models_cfg)
    csv_content_t2 = build_table2_csv(records_t2)

    p_t2_md = root / "docs" / "tables" / "table2_reliability.md"
    p_t2_csv = root / "docs" / "tables" / "table2_reliability.csv"

    if check_mode:
        for p, computed, name in [
            (p_t1_md, md_content_t1, "Table 1 MD"),
            (p_t1_csv, csv_content_t1, "Table 1 CSV"),
            (p_t2_md, md_content_t2, "Table 2 MD"),
            (p_t2_csv, csv_content_t2, "Table 2 CSV"),
        ]:
            if not p.exists():
                print(f"FAIL: Target file does not exist: {p}", file=sys.stderr)
                return 1
            existing = p.read_text(encoding="utf-8")
            if existing != computed:
                print(f"FAIL: {p} ({name}) differs from computed content.", file=sys.stderr)
                return 1

        print("PASS: docs/tables/table1_qa and table2_reliability (md, csv) match committed outputs.")
        return 0

    p_t1_md.parent.mkdir(parents=True, exist_ok=True)
    p_t1_md.write_text(md_content_t1, encoding="utf-8")
    p_t1_csv.write_text(csv_content_t1, encoding="utf-8")
    p_t2_md.write_text(md_content_t2, encoding="utf-8")
    p_t2_csv.write_text(csv_content_t2, encoding="utf-8")
    print(f"Wrote {p_t1_md}")
    print(f"Wrote {p_t1_csv}")
    print(f"Wrote {p_t2_md}")
    print(f"Wrote {p_t2_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Table 1 (QA) and Table 2 (Reliability) for V3 study.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether existing table files match computed contents without modifying them.",
    )
    args = parser.parse_args()
    return generate_tables(check_mode=args.check)


if __name__ == "__main__":
    sys.exit(main())

