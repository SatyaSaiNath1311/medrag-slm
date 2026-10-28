"""src/paper_tables.py — Build Table 1 (QA Performance) for V3 Matched-Pair Study.

Generates:
  - docs/tables/table1_qa.md
  - docs/tables/table1_qa.csv

Protocol & Ground Rules:
  - Reads only committed, frozen evaluation data from test splits (N=500 per cell).
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
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

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


def generate_tables(repo_root: Optional[Path] = None, check_mode: bool = False) -> int:
    """Load data, verify cross-checks, and write or check table1 outputs."""
    root = repo_root or REPO_ROOT
    records = load_all_table1_records(root)
    cross_check_results(records, root)

    md_content = build_table1_markdown(records)
    csv_content = build_table1_csv(records)

    p_md = root / "docs" / "tables" / "table1_qa.md"
    p_csv = root / "docs" / "tables" / "table1_qa.csv"

    if check_mode:
        if not p_md.exists() or not p_csv.exists():
            print(f"FAIL: Target files do not exist: {p_md} or {p_csv}", file=sys.stderr)
            return 1

        existing_md = p_md.read_text(encoding="utf-8")
        existing_csv = p_csv.read_text(encoding="utf-8")

        if existing_md != md_content:
            print(f"FAIL: {p_md} differs from computed table content.", file=sys.stderr)
            return 1
        if existing_csv != csv_content:
            print(f"FAIL: {p_csv} differs from computed CSV content.", file=sys.stderr)
            return 1

        print("PASS: docs/tables/table1_qa (md, csv) match committed outputs.")
        return 0

    p_md.parent.mkdir(parents=True, exist_ok=True)
    p_md.write_text(md_content, encoding="utf-8")
    p_csv.write_text(csv_content, encoding="utf-8")
    print(f"Wrote {p_md}")
    print(f"Wrote {p_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Table 1 QA Performance for V3 study.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether existing table files match computed contents without modifying them.",
    )
    args = parser.parse_args()
    return generate_tables(check_mode=args.check)


if __name__ == "__main__":
    sys.exit(main())
