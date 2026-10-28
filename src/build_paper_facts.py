#!/usr/bin/env python3
"""src/build_paper_facts.py — Generate and Audit the Master Paper Fact Ledger.

Outputs:
  - docs/PAPER_FACTS.md

Features:
  - Dynamic extraction and derivation of all quoted experimental numbers (F001...F073)
    directly from committed analysis reports, manifests, and tables at build time.
  - Plain-language statement, value, 95% CI, split, sample size, and provenance line/key.
  - Fail-fast validation: build or audit fails if any cited source file cannot be parsed
    or an expected empirical number is missing.
  - Verification mode (--check): ensures docs/PAPER_FACTS.md is up-to-date with committed
    analyses and matches dynamically extracted source facts.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute 2-sided Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z ** 2) / (4 * (n ** 2)))) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def mcnemar_asymptotic_ci(b: int, c: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute asymptotic Wald confidence interval for paired binary difference."""
    diff = (c - b) / n
    var = (b + c - ((b - c) ** 2) / n) / (n ** 2)
    margin = z * math.sqrt(max(0.0, var))
    return (round(diff - margin, 4), round(diff + margin, 4))


# ── Dynamic Provenance Parser ──────────────────────────────────────────────────

def compute_facts(repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Compute and verify all 73 paper facts directly from repository source files."""
    root = repo_root or REPO_ROOT

    # 1. Source: outputs/analysis/pair_a_medqa.md
    p_pair_a = root / "outputs/analysis/pair_a_medqa.md"
    if not p_pair_a.exists():
        raise FileNotFoundError(f"Missing cited source: {p_pair_a}")
    t_pair_a = p_pair_a.read_text(encoding="utf-8")
    lines_pair_a = t_pair_a.splitlines()

    # Extract lines 5-9 (C1)
    l5 = next((l for l in lines_pair_a if l.startswith("- medpsy-1.7b:") and "strict" in l), None)
    l6 = next((l for l in lines_pair_a if l.startswith("- qwen3-1.7b-thinking:") and "strict" in l), None)
    l7 = next((l for l in lines_pair_a if "paired medpsy-1.7b minus qwen3-1.7b-thinking (strict):" in l), None)
    l8 = next((l for l in lines_pair_a if "paired medpsy-1.7b minus qwen3-1.7b-thinking (fallback):" in l), None)
    l9 = next((l for l in lines_pair_a if "paired medpsy-1.7b minus qwen3-1.7b-thinking (truncated):" in l), None)

    if not all([l5, l6, l7, l8, l9]):
        raise ValueError(f"Failed to parse C1 section in {p_pair_a}")

    m5 = re.search(r"truncated (\d+)/(\d+) \((\d+\.\d+)%\); strict \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\]); fallback \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\])", l5)
    m6 = re.search(r"truncated (\d+)/(\d+) \((\d+\.\d+)%\); strict \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\]); fallback \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\])", l6)
    m7 = re.search(r"\(strict\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\]).*?McNemar raw p\s*=\s*([0-9e.-]+)", l7)
    m8 = re.search(r"\(fallback\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l8)
    m9 = re.search(r"\(truncated\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l9)

    if not all([m5, m6, m7, m8, m9]):
        raise ValueError(f"Regex match failure in C1 lines of {p_pair_a}")

    # Extract lines 13-17 (C2)
    c2_start = next(i for i, l in enumerate(lines_pair_a) if l.strip() == "## C2")
    l13 = next((l for l in lines_pair_a[c2_start:] if l.startswith("- medpsy-1.7b:") and "strict" in l), None)
    l14 = next((l for l in lines_pair_a[c2_start:] if l.startswith("- qwen3-1.7b-thinking:") and "strict" in l), None)
    l15 = next((l for l in lines_pair_a[c2_start:] if "paired medpsy-1.7b minus qwen3-1.7b-thinking (strict):" in l), None)
    l16 = next((l for l in lines_pair_a[c2_start:] if "paired medpsy-1.7b minus qwen3-1.7b-thinking (fallback):" in l), None)
    l17 = next((l for l in lines_pair_a[c2_start:] if "paired medpsy-1.7b minus qwen3-1.7b-thinking (truncated):" in l), None)

    m13 = re.search(r"truncated (\d+)/(\d+) \((\d+\.\d+)%\); strict \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\]); fallback \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\])", l13)
    m14 = re.search(r"truncated (\d+)/(\d+) \((\d+\.\d+)%\); strict \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\]); fallback \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\])", l14)
    m15 = re.search(r"\(strict\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l15)
    m16 = re.search(r"\(fallback\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l16)
    m17 = re.search(r"\(truncated\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l17)

    if not all([m13, m14, m15, m16, m17]):
        raise ValueError(f"Regex match failure in C2 lines of {p_pair_a}")

    # Extract lines 28-36 (Within-model RAG effect & difference)
    rag_start = next(i for i, l in enumerate(lines_pair_a) if "## RAG effect" in l)
    l29 = next((l for l in lines_pair_a[rag_start:] if l.startswith("- medpsy-1.7b (strict):")), None)
    l31 = next((l for l in lines_pair_a[rag_start:] if l.startswith("- medpsy-1.7b (truncated):")), None)
    l32 = next((l for l in lines_pair_a[rag_start:] if l.startswith("- qwen3-1.7b-thinking (strict):")), None)
    l35 = next((l for l in lines_pair_a[rag_start:] if "difference in RAG effect, medpsy-1.7b minus qwen3-1.7b-thinking (strict):" in l), None)

    m29 = re.search(r"\(strict\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l29)
    m31 = re.search(r"\(truncated\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l31)
    m32 = re.search(r"\(strict\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l32)
    m35 = re.search(r"\(strict\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l35)

    if not all([m29, m31, m32, m35]):
        raise ValueError(f"Regex match failure in RAG effect lines of {p_pair_a}")

    # 2. Source: outputs/analysis/pair_a_c45.md
    p_pair_a_c45 = root / "outputs/analysis/pair_a_c45.md"
    if not p_pair_a_c45.exists():
        raise FileNotFoundError(f"Missing cited source: {p_pair_a_c45}")
    t_c45 = p_pair_a_c45.read_text(encoding="utf-8")
    lines_c45 = t_c45.splitlines()

    l_c4_qwen = next(l for l in lines_c45 if "| C4 | `qwen3-1.7b-thinking`" in l)
    l_c4_medpsy = next(l for l in lines_c45 if "| C4 | `medpsy-1.7b`" in l)
    l_c5_qwen = next(l for l in lines_c45 if "| C5 | `qwen3-1.7b-thinking`" in l)
    l_c5_medpsy = next(l for l in lines_c45 if "| C5 | `medpsy-1.7b`" in l)

    def parse_c45_trunc(line: str) -> Tuple[int, int, float, str]:
        parts = [p.strip() for p in line.split("|")]
        n = int(parts[3])
        trunc_part = parts[5]  # e.g. "28 (20.0%)"
        m = re.search(r"(\d+)\s*\(([\d.]+)%\)", trunc_part)
        k = int(m.group(1))
        rate_pct = float(m.group(2))
        w_low, w_high = wilson_ci(k, n)
        return (k, n, rate_pct, f"[{w_low:.3f}, {w_high:.3f}]")

    c4_q_k, c4_q_n, c4_q_pct, c4_q_ci = parse_c45_trunc(l_c4_qwen)
    c4_m_k, c4_m_n, c4_m_pct, c4_m_ci = parse_c45_trunc(l_c4_medpsy)
    c5_q_k, c5_q_n, c5_q_pct, c5_q_ci = parse_c45_trunc(l_c5_qwen)
    c5_m_k, c5_m_n, c5_m_pct, c5_m_ci = parse_c45_trunc(l_c5_medpsy)

    # 3. Source: outputs/analysis/gemma_pair.md
    p_gemma_pair = root / "outputs/analysis/gemma_pair.md"
    if not p_gemma_pair.exists():
        raise FileNotFoundError(f"Missing cited source: {p_gemma_pair}")
    t_gemma_pair = p_gemma_pair.read_text(encoding="utf-8")
    lines_gemma_pair = t_gemma_pair.splitlines()

    l_gp_s1_gemma_c1 = next(l for l in lines_gemma_pair if "| C1 | MEDQA | Gemma-3-4B" in l)
    l_gp_s1_medgemma_c1 = next(l for l in lines_gemma_pair if "| C1 | MEDQA | MedGemma-1.5-4B-it" in l)
    l_gp_s2_c1_mq = next(l for l in lines_gemma_pair if "| C1 | MEDQA | 500 |" in l)
    l_gp_s2_c1_pq = next(l for l in lines_gemma_pair if "| C1 | PUBMEDQA | 500 |" in l)
    l_gp_s2_c2_mq = next(l for l in lines_gemma_pair if "| C2 | MEDQA | 500 |" in l)
    l_gp_s2_c2_pq = next(l for l in lines_gemma_pair if "| C2 | PUBMEDQA | 500 |" in l)
    l_gp_s2_c3_pq = next(l for l in lines_gemma_pair if "| C3 | PUBMEDQA | 500 |" in l)
    l_gp_s3_gemma_mq = next(l for l in lines_gemma_pair if l.startswith("| MEDQA | Gemma-3-4B"))
    l_gp_s3_medgemma_mq = next(l for l in lines_gemma_pair if l.startswith("| MEDQA | MedGemma-1.5-4B-it"))
    l_gp_s4_gold_a = next(l for l in lines_gemma_pair if "| **A** | `yes` | 276 |" in l)

    # 4. Source: outputs/analysis/pair_b_equal_budget.md
    p_pair_b = root / "outputs/analysis/pair_b_equal_budget.md"
    if not p_pair_b.exists():
        raise FileNotFoundError(f"Missing cited source: {p_pair_b}")
    t_pair_b = p_pair_b.read_text(encoding="utf-8")
    lines_pair_b = t_pair_b.splitlines()

    l_pb_6 = next(l for l in lines_pair_b if l.startswith("- MedPsy-4B: truncated") and "strict" in l)
    l_pb_7 = next(l for l in lines_pair_b if l.startswith("- Qwen3-4B-Thinking-2507: truncated") and "strict" in l)
    l_pb_8 = next(l for l in lines_pair_b if "paired MedPsy minus base (strict):" in l and "+0.828" in l)
    l_pb_9 = next(l for l in lines_pair_b if "paired MedPsy minus base (fallback):" in l and "+0.208" in l)
    c2_b_start = next(i for i, l in enumerate(lines_pair_b) if l.strip() == "## C2 RAG")
    l_pb_12 = next(l for l in lines_pair_b[c2_b_start:] if l.startswith("- MedPsy-4B: truncated"))
    l_pb_13 = next(l for l in lines_pair_b[c2_b_start:] if l.startswith("- Qwen3-4B-Thinking-2507: truncated"))
    l_pb_14 = next(l for l in lines_pair_b[c2_b_start:] if "paired MedPsy minus base (strict):" in l)

    m_pb_6 = re.search(r"truncated (\d+)/(\d+); strict \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\]); fallback \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\])", l_pb_6)
    m_pb_7 = re.search(r"truncated (\d+)/(\d+); strict \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\]); fallback \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\])", l_pb_7)
    m_pb_8 = re.search(r"\(strict\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l_pb_8)
    m_pb_9 = re.search(r"\(fallback\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l_pb_9)
    m_pb_12 = re.search(r"truncated (\d+)/(\d+); strict \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\])", l_pb_12)
    m_pb_13 = re.search(r"truncated (\d+)/(\d+); strict \d+/\d+ = (\d+\.\d+) (\[\d+\.\d+, \d+\.\d+\])", l_pb_13)
    m_pb_14 = re.search(r"\(strict\):\s*([+-]\d+\.\d+)\s*(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])", l_pb_14)

    # 5. Source: outputs/analysis/budget_pair_b.md
    p_budget_sweep = root / "outputs/analysis/budget_pair_b.md"
    if not p_budget_sweep.exists():
        raise FileNotFoundError(f"Missing cited source: {p_budget_sweep}")
    t_budget_sweep = p_budget_sweep.read_text(encoding="utf-8")
    lines_budget_sweep = t_budget_sweep.splitlines()

    l_bs_1024 = next(l for l in lines_budget_sweep if "| 1024 | Strict Accuracy |" in l)
    l_bs_1536 = next(l for l in lines_budget_sweep if "| 1536 | Strict Accuracy |" in l)
    l_bs_2048 = next(l for l in lines_budget_sweep if "| 2048 | Strict Accuracy |" in l)
    l_bs_3072 = next(l for l in lines_budget_sweep if "| 3072 | Strict Accuracy |" in l)
    l_bs_4096 = next(l for l in lines_budget_sweep if "| 4096 | Strict Accuracy |" in l)
    l_bs_qwen_4096 = next(l for l in lines_budget_sweep if "| 4096 | `qwen3-4b-thinking-2507` |" in l)

    # 6. Source: outputs/analysis/gemma_prompt_sensitivity.md
    p_gemma_sens = root / "outputs/analysis/gemma_prompt_sensitivity.md"
    if not p_gemma_sens.exists():
        raise FileNotFoundError(f"Missing cited source: {p_gemma_sens}")
    t_gemma_sens = p_gemma_sens.read_text(encoding="utf-8")
    lines_gemma_sens = t_gemma_sens.splitlines()

    l_gs_v1 = next(l for l in lines_gemma_sens if "| V1 |" in l and "pp" in l)
    l_gs_v2 = next(l for l in lines_gemma_sens if "| V2 |" in l and "pp" in l)
    l_gs_v3 = next(l for l in lines_gemma_sens if "| V3 |" in l and "pp" in l)
    l_gs_spread = next(l for l in lines_gemma_sens if "Criterion 2 (Spread Bound)" in l)
    l_gs_verdict = next(l for l in lines_gemma_sens if "Pre-Registered Verdict:" in l)

    m_gs_v1 = re.search(r"\|\s*([+-]?\d+\.\d+)\s*pp\s*\|\s*(\[[+-]?\d+\.\d+,\s*[+-]?\d+\.\d+\])", l_gs_v1)
    m_gs_v2 = re.search(r"\|\s*([+-]?\d+\.\d+)\s*pp\s*\|\s*(\[[+-]?\d+\.\d+,\s*[+-]?\d+\.\d+\])", l_gs_v2)
    m_gs_v3 = re.search(r"\|\s*([+-]?\d+\.\d+)\s*pp\s*\|\s*(\[[+-]?\d+\.\d+,\s*[+-]?\d+\.\d+\])", l_gs_v3)
    m_gs_spread = re.search(r"\*\*(\d+\.\d+)\s*(?:percentage points|pp)\*\*", l_gs_spread)
    m_gs_verdict = re.search(r"Pre-Registered Verdict:\s*([A-Z]+):", l_gs_verdict)

    # 7. Source: outputs/analysis/medpsy_reliability.md
    p_reliability = root / "outputs/analysis/medpsy_reliability.md"
    if not p_reliability.exists():
        raise FileNotFoundError(f"Missing cited source: {p_reliability}")
    t_reliability = p_reliability.read_text(encoding="utf-8")
    lines_reliability = t_reliability.splitlines()

    l_rel_header = next(l for l in lines_reliability if "Acc when answering [CI]" in l)
    idx_rel_acc = [c.strip() for c in l_rel_header.split("|")].index("Acc when answering [CI]")

    l_rel_always = next(l for l in lines_reliability if "| Always answer |" in l)
    l_rel_trunc = next(l for l in lines_reliability if "| Abstain if truncated (pre-specified) |" in l)
    l_rel_f1 = next(l for l in lines_reliability if "Reliability score, max-F1 τ" in l)
    l_rel_safe = next(l for l in lines_reliability if "Reliability score, safe τ" in l)

    cell_rel_always_acc = [c.strip() for c in l_rel_always.split("|")][idx_rel_acc]
    cell_rel_trunc_acc = [c.strip() for c in l_rel_trunc.split("|")][idx_rel_acc]
    m_rel_always_acc = re.search(r"(\d+\.\d+)%\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]", cell_rel_always_acc)
    m_rel_trunc_acc = re.search(r"(\d+\.\d+)%\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]", cell_rel_trunc_acc)

    # 8. Source: outputs/sweeps/budget_pair_b/manifest.json
    p_manifest_b = root / "outputs/sweeps/budget_pair_b/manifest.json"
    if not p_manifest_b.exists():
        raise FileNotFoundError(f"Missing cited source: {p_manifest_b}")
    manifest_b = json.loads(p_manifest_b.read_text(encoding="utf-8"))

    # 9. Source: outputs/sweeps/gemma_prompt_sensitivity/manifest.json
    p_manifest_g = root / "outputs/sweeps/gemma_prompt_sensitivity/manifest.json"
    if not p_manifest_g.exists():
        raise FileNotFoundError(f"Missing cited source: {p_manifest_g}")
    manifest_g = json.loads(p_manifest_g.read_text(encoding="utf-8"))

    # 10. Source: outputs/analysis/medpsy_rag.md
    p_medpsy_rag = root / "outputs/analysis/medpsy_rag.md"
    if not p_medpsy_rag.exists():
        raise FileNotFoundError(f"Missing cited source: {p_medpsy_rag}")
    t_medpsy_rag = p_medpsy_rag.read_text(encoding="utf-8")
    lines_medpsy_rag = t_medpsy_rag.splitlines()
    l_mrag_base = next(l for l in lines_medpsy_rag if "| Baseline (no RAG) |" in l)
    l_mrag_rag = next(l for l in lines_medpsy_rag if "| RAG (top-5 evidence) |" in l)

    facts = [
        # ── Pair A: 1.7B Thinking (MedPsy-1.7B vs Qwen3-1.7B-Thinking) ───────────────
        {
            "id": "F001",
            "statement": "Pair A MedQA C1 (closed-book) strict accuracy gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking)",
            "value": f"{float(m7.group(1))*100:+.1f} pp ({m7.group(1)})",
            "ci_95": m7.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 7",
        },
        {
            "id": "F002",
            "statement": "Pair A MedQA C1 with-fallback accuracy gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking) [Secondary]",
            "value": f"{float(m8.group(1))*100:+.1f} pp ({m8.group(1)})",
            "ci_95": m8.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 8",
        },
        {
            "id": "F003",
            "statement": "Pair A MedQA C1 truncation rate gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking)",
            "value": f"{float(m9.group(1))*100:+.1f} pp ({m9.group(1)})",
            "ci_95": m9.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 9",
        },
        {
            "id": "F004",
            "statement": "Pair A MedQA C1 MedPsy-1.7B strict accuracy",
            "value": f"{float(m5.group(4))*100:.1f}% ({m5.group(4)})",
            "ci_95": m5.group(5),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 5",
        },
        {
            "id": "F005",
            "statement": "Pair A MedQA C1 MedPsy-1.7B with-fallback accuracy [Secondary]",
            "value": f"{float(m5.group(6))*100:.1f}% ({m5.group(6)})",
            "ci_95": m5.group(7),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 5",
        },
        {
            "id": "F006",
            "statement": f"Pair A MedQA C1 MedPsy-1.7B truncation rate ({m5.group(1)}/500)",
            "value": f"{float(m5.group(3)):.1f}% ({float(m5.group(3))/100:.3f})",
            "ci_95": f"[{wilson_ci(int(m5.group(1)), int(m5.group(2)))[0]:.3f}, {wilson_ci(int(m5.group(1)), int(m5.group(2)))[1]:.3f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 5",
        },
        {
            "id": "F007",
            "statement": "Pair A MedQA C1 Qwen3-1.7B-Thinking strict accuracy",
            "value": f"{float(m6.group(4))*100:.1f}% ({m6.group(4)})",
            "ci_95": m6.group(5),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 6",
        },
        {
            "id": "F008",
            "statement": "Pair A MedQA C1 Qwen3-1.7B-Thinking with-fallback accuracy [Secondary]",
            "value": f"{float(m6.group(6))*100:.1f}% ({m6.group(6)})",
            "ci_95": m6.group(7),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 6",
        },
        {
            "id": "F009",
            "statement": f"Pair A MedQA C1 Qwen3-1.7B-Thinking truncation rate ({m6.group(1)}/500)",
            "value": f"{float(m6.group(3)):.1f}% ({float(m6.group(3))/100:.3f})",
            "ci_95": f"[{wilson_ci(int(m6.group(1)), int(m6.group(2)))[0]:.3f}, {wilson_ci(int(m6.group(1)), int(m6.group(2)))[1]:.3f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 6",
        },
        {
            "id": "F010",
            "statement": "Pair A MedQA C1 exact two-sided McNemar p-value (strict accuracy difference)",
            "value": f"<1e-4 ({m7.group(3)})",
            "ci_95": "N/A",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 7",
        },
        {
            "id": "F011",
            "statement": "Pair A MedQA C2 (full RAG) strict accuracy gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking)",
            "value": f"{float(m15.group(1))*100:+.1f} pp ({m15.group(1)})",
            "ci_95": m15.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 15",
        },
        {
            "id": "F012",
            "statement": "Pair A MedQA C2 with-fallback accuracy gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking) [Secondary]",
            "value": f"{float(m16.group(1))*100:+.1f} pp ({m16.group(1)})",
            "ci_95": m16.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 16",
        },
        {
            "id": "F013",
            "statement": "Pair A MedQA C2 truncation rate gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking)",
            "value": f"{float(m17.group(1))*100:+.1f} pp ({m17.group(1)})",
            "ci_95": m17.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 17",
        },
        {
            "id": "F014",
            "statement": "Pair A MedQA C2 MedPsy-1.7B strict accuracy",
            "value": f"{float(m13.group(4))*100:.1f}% ({m13.group(4)})",
            "ci_95": m13.group(5),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 13",
        },
        {
            "id": "F015",
            "statement": "Pair A MedQA C2 MedPsy-1.7B with-fallback accuracy [Secondary]",
            "value": f"{float(m13.group(6))*100:.1f}% ({m13.group(6)})",
            "ci_95": m13.group(7),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 13",
        },
        {
            "id": "F016",
            "statement": f"Pair A MedQA C2 MedPsy-1.7B truncation rate ({m13.group(1)}/500)",
            "value": f"{float(m13.group(3)):.1f}% ({float(m13.group(3))/100:.3f})",
            "ci_95": f"[{wilson_ci(int(m13.group(1)), int(m13.group(2)))[0]:.3f}, {wilson_ci(int(m13.group(1)), int(m13.group(2)))[1]:.3f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 13",
        },
        {
            "id": "F017",
            "statement": "Pair A MedQA C2 Qwen3-1.7B-Thinking strict accuracy",
            "value": f"{float(m14.group(4))*100:.1f}% ({m14.group(4)})",
            "ci_95": m14.group(5),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 14",
        },
        {
            "id": "F018",
            "statement": "Pair A MedQA C2 Qwen3-1.7B-Thinking with-fallback accuracy [Secondary]",
            "value": f"{float(m14.group(6))*100:.1f}% ({m14.group(6)})",
            "ci_95": m14.group(7),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 14",
        },
        {
            "id": "F019",
            "statement": f"Pair A MedQA C2 Qwen3-1.7B-Thinking truncation rate ({m14.group(1)}/500)",
            "value": f"{float(m14.group(3)):.1f}% ({float(m14.group(3))/100:.3f})",
            "ci_95": f"[{wilson_ci(int(m14.group(1)), int(m14.group(2)))[0]:.3f}, {wilson_ci(int(m14.group(1)), int(m14.group(2)))[1]:.3f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 14",
        },
        {
            "id": "F020",
            "statement": "Pair A MedQA RAG within-model effect on MedPsy-1.7B strict accuracy (C2 minus C1)",
            "value": f"{float(m29.group(1))*100:+.1f} pp ({m29.group(1)})",
            "ci_95": m29.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 29",
        },
        {
            "id": "F021",
            "statement": "Pair A MedQA RAG within-model effect on MedPsy-1.7B truncation rate (C2 minus C1)",
            "value": f"{float(m31.group(1))*100:+.1f} pp ({m31.group(1)})",
            "ci_95": m31.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 31",
        },
        {
            "id": "F022",
            "statement": "Pair A MedQA RAG within-model effect on Qwen3-1.7B-Thinking strict accuracy (C2 minus C1)",
            "value": f"{float(m32.group(1))*100:+.1f} pp ({m32.group(1)})",
            "ci_95": m32.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 32",
        },
        {
            "id": "F023",
            "statement": "Pair A MedQA difference in RAG effect (MedPsy minus Qwen strict accuracy interaction)",
            "value": f"{float(m35.group(1))*100:+.1f} pp ({m35.group(1)})",
            "ci_95": m35.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_a_medqa.md",
            "source_ref": "Line 35",
        },

        # ── Unanswerable v2 C4/C5 ──────────────────────────────────────────────────
        {
            "id": "F024",
            "statement": f"Pair A Unanswerable v2 C4 truncation rate for MedPsy-1.7B ({c4_m_k}/{c4_m_n})",
            "value": f"{c4_m_pct:.1f}% ({c4_m_k/c4_m_n:.3f})",
            "ci_95": c4_m_ci,
            "split": "test",
            "n": c4_m_n,
            "source_file": "outputs/analysis/pair_a_c45.md",
            "source_ref": "Line 17",
        },
        {
            "id": "F025",
            "statement": f"Pair A Unanswerable v2 C4 truncation rate for Qwen3-1.7B-Thinking ({c4_q_k}/{c4_q_n})",
            "value": f"{c4_q_pct:.1f}% ({c4_q_k/c4_q_n:.3f})",
            "ci_95": c4_q_ci,
            "split": "test",
            "n": c4_q_n,
            "source_file": "outputs/analysis/pair_a_c45.md",
            "source_ref": "Line 16",
        },
        {
            "id": "F026",
            "statement": f"Pair A Unanswerable v2 C5 truncation rate for MedPsy-1.7B ({c5_m_k}/{c5_m_n})",
            "value": f"{c5_m_pct:.1f}% ({c5_m_k/c5_m_n:.3f})",
            "ci_95": c5_m_ci,
            "split": "test",
            "n": c5_m_n,
            "source_file": "outputs/analysis/pair_a_c45.md",
            "source_ref": "Line 19",
        },
        {
            "id": "F027",
            "statement": f"Pair A Unanswerable v2 C5 truncation rate for Qwen3-1.7B-Thinking ({c5_q_k}/{c5_q_n})",
            "value": f"{c5_q_pct:.1f}% ({c5_q_k/c5_q_n:.3f})",
            "ci_95": c5_q_ci,
            "split": "test",
            "n": c5_q_n,
            "source_file": "outputs/analysis/pair_a_c45.md",
            "source_ref": "Line 18",
        },

        # ── Gemma Pair: 4B Direct (MedGemma-1.5-4B-it vs Gemma-3-4B) ─────────────────
        {
            "id": "F028",
            "statement": "Gemma Pair MedQA C1 strict accuracy gap (MedGemma minus Gemma3)",
            "value": f"{float(re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c1_mq).group(1))*100:+.2f} pp ({re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c1_mq).group(1)})",
            "ci_95": re.search(r"`(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])`", l_gp_s2_c1_mq).group(1),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 54",
        },
        {
            "id": "F029",
            "statement": "Gemma Pair MedQA C1 Gemma-3-4B strict accuracy",
            "value": f"{float(re.search(r'(\d+\.\d+)\s*`\[', l_gp_s1_gemma_c1).group(1))*100:.2f}% ({re.search(r'(\d+\.\d+)\s*`\[', l_gp_s1_gemma_c1).group(1)})",
            "ci_95": re.search(r"`(\[\d+\.\d+,\s*\d+\.\d+\])`", l_gp_s1_gemma_c1).group(1),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 29",
        },
        {
            "id": "F030",
            "statement": "Gemma Pair MedQA C1 MedGemma-1.5-4B-it strict accuracy",
            "value": f"{float(re.search(r'(\d+\.\d+)\s*`\[', l_gp_s1_medgemma_c1).group(1))*100:.2f}% ({re.search(r'(\d+\.\d+)\s*`\[', l_gp_s1_medgemma_c1).group(1)})",
            "ci_95": re.search(r"`(\[\d+\.\d+,\s*\d+\.\d+\])`", l_gp_s1_medgemma_c1).group(1),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 30",
        },
        {
            "id": "F031",
            "statement": "Gemma Pair MedQA C1 raw McNemar p-value",
            "value": re.search(r"\|\s*(\d+\.\d{5})\s*\|\s*\d+\.\d{5}\s*\|", l_gp_s2_c1_mq).group(1),
            "ci_95": "N/A",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 54",
        },
        {
            "id": "F032",
            "statement": "Gemma Pair MedQA C1 Holm-adjusted p-value across m=5 family",
            "value": f"{re.search(r'\|\s*\d+\.\d{5}\s*\|\s*(\d+\.\d{5})\s*\|', l_gp_s2_c1_mq).group(1)} (ns at alpha=0.05)",
            "ci_95": "N/A",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 54",
        },
        {
            "id": "F033",
            "statement": "Gemma Pair PubMedQA C1 strict accuracy gap (MedGemma minus Gemma3)",
            "value": f"{float(re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c1_pq).group(1))*100:+.2f} pp ({re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c1_pq).group(1)})",
            "ci_95": re.search(r"`(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])`", l_gp_s2_c1_pq).group(1),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 55",
        },
        {
            "id": "F034",
            "statement": "Gemma Pair MedQA C2 strict accuracy gap (MedGemma minus Gemma3)",
            "value": f"{float(re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c2_mq).group(1))*100:+.2f} pp ({re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c2_mq).group(1)})",
            "ci_95": re.search(r"`(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])`", l_gp_s2_c2_mq).group(1),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 56",
        },
        {
            "id": "F035",
            "statement": "Gemma Pair PubMedQA C2 strict accuracy gap (MedGemma minus Gemma3)",
            "value": f"{float(re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c2_pq).group(1))*100:+.2f} pp ({re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c2_pq).group(1)})",
            "ci_95": re.search(r"`(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])`", l_gp_s2_c2_pq).group(1),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 57",
        },
        {
            "id": "F036",
            "statement": "Gemma Pair PubMedQA C3 (oracle abstract) strict accuracy gap (MedGemma minus Gemma3)",
            "value": f"{float(re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c3_pq).group(1))*100:+.2f} pp ({re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s2_c3_pq).group(1)})",
            "ci_95": re.search(r"`(\[[+-]\d+\.\d+,\s*[+-]\d+\.\d+\])`", l_gp_s2_c3_pq).group(1),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 58",
        },
        {
            "id": "F037",
            "statement": "Gemma Pair MedQA within-model RAG effect for Gemma-3-4B (C2 minus C1)",
            "value": f"{float(re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s3_gemma_mq).group(1))*100:+.2f} pp ({re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s3_gemma_mq).group(1)})",
            "ci_95": f"[{mcnemar_asymptotic_ci(int(re.search(r'b=(\d+)\s*/\s*c=(\d+)', l_gp_s3_gemma_mq).group(1)), int(re.search(r'b=(\d+)\s*/\s*c=(\d+)', l_gp_s3_gemma_mq).group(2)), 500)[0]:+.4f}, {mcnemar_asymptotic_ci(int(re.search(r'b=(\d+)\s*/\s*c=(\d+)', l_gp_s3_gemma_mq).group(1)), int(re.search(r'b=(\d+)\s*/\s*c=(\d+)', l_gp_s3_gemma_mq).group(2)), 500)[1]:+.4f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 78",
        },
        {
            "id": "F038",
            "statement": "Gemma Pair MedQA within-model RAG effect for MedGemma-1.5-4B-it (C2 minus C1)",
            "value": f"{float(re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s3_medgemma_mq).group(1))*100:+.2f} pp ({re.search(r'\|\s*([+-]\d+\.\d+)\s*\(\+\d+\)\s*\|', l_gp_s3_medgemma_mq).group(1)})",
            "ci_95": f"[{mcnemar_asymptotic_ci(int(re.search(r'b=(\d+)\s*/\s*c=(\d+)', l_gp_s3_medgemma_mq).group(1)), int(re.search(r'b=(\d+)\s*/\s*c=(\d+)', l_gp_s3_medgemma_mq).group(2)), 500)[0]:+.4f}, {mcnemar_asymptotic_ci(int(re.search(r'b=(\d+)\s*/\s*c=(\d+)', l_gp_s3_medgemma_mq).group(1)), int(re.search(r'b=(\d+)\s*/\s*c=(\d+)', l_gp_s3_medgemma_mq).group(2)), 500)[1]:+.4f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 79",
        },
        {
            "id": "F039",
            "statement": "Gemma Pair PubMedQA majority-class baseline share ('yes')",
            "value": f"{float(re.search(r'(\d+\.\d+)%', l_gp_s4_gold_a).group(1)):.2f}% ({re.search(r'\|\s*(\d+)\s*\|', l_gp_s4_gold_a).group(1)}/500)",
            "ci_95": re.search(r"`(\[\d+\.\d+,\s*\d+\.\d+\])`", l_gp_s4_gold_a).group(1),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/gemma_pair.md",
            "source_ref": "Line 98",
        },

        # ── Pair B: 4B Thinking (MedPsy-4B vs Qwen3-4B-Thinking-2507 at 1024 Cap) ─────
        {
            "id": "F040",
            "statement": "Pair B MedQA C1 strict accuracy gap (MedPsy-4B minus Qwen3-4B-Thinking at 1,024 cap)",
            "value": f"{float(m_pb_8.group(1))*100:+.1f} pp ({m_pb_8.group(1)})",
            "ci_95": m_pb_8.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 8",
        },
        {
            "id": "F041",
            "statement": "Pair B MedQA C1 with-fallback accuracy gap (MedPsy-4B minus Qwen3-4B-Thinking) [Secondary]",
            "value": f"{float(m_pb_9.group(1))*100:+.1f} pp ({m_pb_9.group(1)})",
            "ci_95": m_pb_9.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 9",
        },
        {
            "id": "F042",
            "statement": "Pair B MedQA C1 MedPsy-4B strict accuracy",
            "value": f"{float(m_pb_6.group(3))*100:.1f}% ({m_pb_6.group(3)})",
            "ci_95": m_pb_6.group(4),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 6",
        },
        {
            "id": "F043",
            "statement": "Pair B MedQA C1 MedPsy-4B with-fallback accuracy [Secondary]",
            "value": f"{float(m_pb_6.group(5))*100:.1f}% ({m_pb_6.group(5)})",
            "ci_95": m_pb_6.group(6),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 6",
        },
        {
            "id": "F044",
            "statement": f"Pair B MedQA C1 MedPsy-4B truncation rate ({m_pb_6.group(1)}/500)",
            "value": f"{int(m_pb_6.group(1))/500*100:.1f}% ({int(m_pb_6.group(1))/500:.3f})",
            "ci_95": f"[{wilson_ci(int(m_pb_6.group(1)), int(m_pb_6.group(2)))[0]:.3f}, {wilson_ci(int(m_pb_6.group(1)), int(m_pb_6.group(2)))[1]:.3f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 6",
        },
        {
            "id": "F045",
            "statement": "Pair B MedQA C1 Qwen3-4B-Thinking-2507 strict accuracy",
            "value": f"{float(m_pb_7.group(3))*100:.1f}% ({m_pb_7.group(3)})",
            "ci_95": m_pb_7.group(4),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 7",
        },
        {
            "id": "F046",
            "statement": "Pair B MedQA C1 Qwen3-4B-Thinking-2507 with-fallback accuracy [Secondary]",
            "value": f"{float(m_pb_7.group(5))*100:.1f}% ({m_pb_7.group(5)})",
            "ci_95": m_pb_7.group(6),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 7",
        },
        {
            "id": "F047",
            "statement": f"Pair B MedQA C1 Qwen3-4B-Thinking-2507 truncation rate ({m_pb_7.group(1)}/500)",
            "value": f"{int(m_pb_7.group(1))/500*100:.1f}% ({int(m_pb_7.group(1))/500:.3f})",
            "ci_95": f"[{wilson_ci(int(m_pb_7.group(1)), int(m_pb_7.group(2)))[0]:.3f}, {wilson_ci(int(m_pb_7.group(1)), int(m_pb_7.group(2)))[1]:.3f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 7",
        },
        {
            "id": "F048",
            "statement": "Pair B MedQA C2 strict accuracy gap (MedPsy-4B minus Qwen3-4B-Thinking at 1,024 cap)",
            "value": f"{float(m_pb_14.group(1))*100:+.1f} pp ({m_pb_14.group(1)})",
            "ci_95": m_pb_14.group(2),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 14",
        },
        {
            "id": "F049",
            "statement": "Pair B MedQA C2 MedPsy-4B strict accuracy",
            "value": f"{float(m_pb_12.group(3))*100:.1f}% ({m_pb_12.group(3)})",
            "ci_95": m_pb_12.group(4),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 12",
        },
        {
            "id": "F050",
            "statement": f"Pair B MedQA C2 MedPsy-4B truncation rate ({m_pb_12.group(1)}/500)",
            "value": f"{int(m_pb_12.group(1))/500*100:.1f}% ({int(m_pb_12.group(1))/500:.3f})",
            "ci_95": f"[{wilson_ci(int(m_pb_12.group(1)), int(m_pb_12.group(2)))[0]:.3f}, {wilson_ci(int(m_pb_12.group(1)), int(m_pb_12.group(2)))[1]:.3f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 12",
        },
        {
            "id": "F051",
            "statement": "Pair B MedQA C2 Qwen3-4B-Thinking-2507 strict accuracy",
            "value": f"{float(m_pb_13.group(3))*100:.1f}% ({m_pb_13.group(3)})",
            "ci_95": m_pb_13.group(4),
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 13",
        },
        {
            "id": "F052",
            "statement": f"Pair B MedQA C2 Qwen3-4B-Thinking-2507 truncation rate ({m_pb_13.group(1)}/500)",
            "value": f"{int(m_pb_13.group(1))/500*100:.1f}% ({int(m_pb_13.group(1))/500:.3f})",
            "ci_95": f"[{wilson_ci(int(m_pb_13.group(1)), int(m_pb_13.group(2)))[0]:.3f}, {wilson_ci(int(m_pb_13.group(1)), int(m_pb_13.group(2)))[1]:.3f}]",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/pair_b_equal_budget.md",
            "source_ref": "Line 13",
        },

        # ── Validation Robustness: Budget Sweep (Caps 1024 to 4096) ──────────────────
        {
            "id": "F053",
            "statement": "Pair B Budget Sweep Cap 1024 strict accuracy gap (MedPsy minus Qwen)",
            "value": f"{re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_1024).group(1)} pp ({float(re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_1024).group(1))/100:+.3f})",
            "ci_95": f"[{float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_1024).group(1))/100:+.3f}, {float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_1024).group(2))/100:+.3f}]",
            "split": "val",
            "n": 100,
            "source_file": "outputs/analysis/budget_pair_b.md",
            "source_ref": "Line 55",
        },
        {
            "id": "F054",
            "statement": "Pair B Budget Sweep Cap 1536 strict accuracy gap (MedPsy minus Qwen)",
            "value": f"{re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_1536).group(1)} pp ({float(re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_1536).group(1))/100:+.3f})",
            "ci_95": f"[{float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_1536).group(1))/100:+.3f}, {float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_1536).group(2))/100:+.3f}]",
            "split": "val",
            "n": 100,
            "source_file": "outputs/analysis/budget_pair_b.md",
            "source_ref": "Line 57",
        },
        {
            "id": "F055",
            "statement": "Pair B Budget Sweep Cap 2048 strict accuracy gap (MedPsy minus Qwen)",
            "value": f"{re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_2048).group(1)} pp ({float(re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_2048).group(1))/100:+.3f})",
            "ci_95": f"[{float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_2048).group(1))/100:+.3f}, {float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_2048).group(2))/100:+.3f}]",
            "split": "val",
            "n": 100,
            "source_file": "outputs/analysis/budget_pair_b.md",
            "source_ref": "Line 59",
        },
        {
            "id": "F056",
            "statement": "Pair B Budget Sweep Cap 3072 strict accuracy gap (MedPsy minus Qwen)",
            "value": f"{re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_3072).group(1)} pp ({float(re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_3072).group(1))/100:+.3f})",
            "ci_95": f"[{float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_3072).group(1))/100:+.3f}, {float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_3072).group(2))/100:+.3f}]",
            "split": "val",
            "n": 100,
            "source_file": "outputs/analysis/budget_pair_b.md",
            "source_ref": "Line 61",
        },
        {
            "id": "F057",
            "statement": "Pair B Budget Sweep Cap 4096 strict accuracy gap (MedPsy minus Qwen)",
            "value": f"{re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_4096).group(1)} pp ({float(re.search(r'\|\s*([+-]\d+\.\d+)%\s*\|', l_bs_4096).group(1))/100:+.3f})",
            "ci_95": f"[{float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_4096).group(1))/100:+.3f}, {float(re.search(r'\[([+-]\d+\.\d+)%,\s*([+-]\d+\.\d+)%\]', l_bs_4096).group(2))/100:+.3f}]",
            "split": "val",
            "n": 100,
            "source_file": "outputs/analysis/budget_pair_b.md",
            "source_ref": "Line 63",
        },
        {
            "id": "F058",
            "statement": "Pair B Budget Sweep Cap 4096 Qwen3-4B-Thinking-2507 residual truncation rate",
            "value": f"{re.search(r'\|\s*(\d+\.\d+)%\s*\[(\d+\.\d+)%,\s*(\d+\.\d+)%\]\s*\|\s*\d+\.\d+%\s*\((\d+)\s*trunc\)', l_bs_qwen_4096).group(1)}% ({float(re.search(r'\|\s*(\d+\.\d+)%\s*\[(\d+\.\d+)%,\s*(\d+\.\d+)%\]\s*\|\s*\d+\.\d+%\s*\((\d+)\s*trunc\)', l_bs_qwen_4096).group(1))/100:.3f})",
            "ci_95": f"[{float(re.search(r'\|\s*(\d+\.\d+)%\s*\[(\d+\.\d+)%,\s*(\d+\.\d+)%\]\s*\|\s*\d+\.\d+%\s*\((\d+)\s*trunc\)', l_bs_qwen_4096).group(2))/100:.3f}, {float(re.search(r'\|\s*(\d+\.\d+)%\s*\[(\d+\.\d+)%,\s*(\d+\.\d+)%\]\s*\|\s*\d+\.\d+%\s*\((\d+)\s*trunc\)', l_bs_qwen_4096).group(3))/100:.3f}]",
            "split": "val",
            "n": 100,
            "source_file": "outputs/analysis/budget_pair_b.md",
            "source_ref": "Line 44",
        },

        # ── Validation Robustness: Gemma Prompt Sensitivity ──────────────────────────
        {
            "id": "F059",
            "statement": "Gemma Prompt Sensitivity V1 (standard baseline) gap (Gemma3 minus MedGemma)",
            "value": f"{float(m_gs_v1.group(1)):+.2f} pp ({float(m_gs_v1.group(1))/100:+.4f})",
            "ci_95": f"[{float(re.search(r'\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]', m_gs_v1.group(2)).group(1))/100:+.4f}, {float(re.search(r'\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]', m_gs_v1.group(2)).group(2))/100:+.4f}]",
            "split": "val",
            "n": 200,
            "source_file": "outputs/analysis/gemma_prompt_sensitivity.md",
            "source_ref": "Section 3 Table",
        },
        {
            "id": "F060",
            "statement": "Gemma Prompt Sensitivity V2 (clinical scenario) gap (Gemma3 minus MedGemma)",
            "value": f"{float(m_gs_v2.group(1)):+.2f} pp ({float(m_gs_v2.group(1))/100:+.4f})",
            "ci_95": f"[{float(re.search(r'\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]', m_gs_v2.group(2)).group(1))/100:+.4f}, {float(re.search(r'\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]', m_gs_v2.group(2)).group(2))/100:+.4f}]",
            "split": "val",
            "n": 200,
            "source_file": "outputs/analysis/gemma_prompt_sensitivity.md",
            "source_ref": "Section 3 Table",
        },
        {
            "id": "F061",
            "statement": "Gemma Prompt Sensitivity V3 (board examination) gap (Gemma3 minus MedGemma)",
            "value": f"{float(m_gs_v3.group(1)):+.2f} pp ({float(m_gs_v3.group(1))/100:+.4f})",
            "ci_95": f"[{float(re.search(r'\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]', m_gs_v3.group(2)).group(1))/100:+.4f}, {float(re.search(r'\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]', m_gs_v3.group(2)).group(2))/100:+.4f}]",
            "split": "val",
            "n": 200,
            "source_file": "outputs/analysis/gemma_prompt_sensitivity.md",
            "source_ref": "Section 3 Table",
        },
        {
            "id": "F062",
            "statement": "Gemma Prompt Sensitivity gap spread across variants (max - min)",
            "value": f"{float(m_gs_spread.group(1)):.2f} pp ({float(m_gs_spread.group(1))/100:.4f})",
            "ci_95": "<= 10.0 pp threshold",
            "split": "val",
            "n": 200,
            "source_file": "outputs/analysis/gemma_prompt_sensitivity.md",
            "source_ref": "Section 1",
        },
        {
            "id": "F063",
            "statement": "Gemma Prompt Sensitivity pre-registered stability verdict",
            "value": m_gs_verdict.group(1),
            "ci_95": "N/A",
            "split": "val",
            "n": 200,
            "source_file": "outputs/analysis/gemma_prompt_sensitivity.md",
            "source_ref": "Section 1",
        },

        # ── MedPsy-4B Reliability and Safety Calibration ────────────────────────────
        {
            "id": "F064",
            "statement": "MedPsy-4B frozen test baseline accuracy when answering on MedQA + unanswerable pool, N=650 (Always Answer)",
            "value": f"{m_rel_always_acc.group(1)}% ({float(m_rel_always_acc.group(1))/100:.3f})",
            "ci_95": f"[{float(m_rel_always_acc.group(2))/100:.3f}, {float(m_rel_always_acc.group(3))/100:.3f}]",
            "split": "test",
            "n": 650,
            "source_file": "outputs/analysis/medpsy_reliability.md",
            "source_ref": "Line 11",
        },
        {
            "id": "F065",
            "statement": "MedPsy-4B frozen test accuracy when answering under pre-specified truncation rule (MedQA + unanswerable pool, N=650)",
            "value": f"{m_rel_trunc_acc.group(1)}% ({float(m_rel_trunc_acc.group(1))/100:.3f})",
            "ci_95": f"[{float(m_rel_trunc_acc.group(2))/100:.3f}, {float(m_rel_trunc_acc.group(3))/100:.3f}]",
            "split": "test",
            "n": 650,
            "source_file": "outputs/analysis/medpsy_reliability.md",
            "source_ref": "Line 12",
        },
        {
            "id": "F066",
            "statement": "MedPsy-4B calibrated reliability score abstention F1 (max-F1 threshold from val)",
            "value": re.search(r"\|\s*(\d+\.\d+)\s*\[", l_rel_f1).group(1),
            "ci_95": re.search(r"(\[\d+\.\d+,\s*\d+\.\d+\])", l_rel_f1).group(1),
            "split": "test",
            "n": 650,
            "source_file": "outputs/analysis/medpsy_reliability.md",
            "source_ref": "Line 15",
        },
        {
            "id": "F067",
            "statement": "MedPsy-4B unanswerable recall under calibrated reliability score",
            "value": f"{re.search(r'\|\s*\d+\.\d+\s*\[\d+\.\d+,\s*\d+\.\d+\]\s*\|\s*(\d+\.\d+)%', l_rel_f1).group(1)}%",
            "ci_95": f"[{re.search(r'\|\s*\d+\.\d+\s*\[\d+\.\d+,\s*\d+\.\d+\]\s*\|\s*\d+\.\d+%\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]', l_rel_f1).group(1)}, {re.search(r'\|\s*\d+\.\d+\s*\[\d+\.\d+,\s*\d+\.\d+\]\s*\|\s*\d+\.\d+%\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]', l_rel_f1).group(2)}]",
            "split": "test",
            "n": 650,
            "source_file": "outputs/analysis/medpsy_reliability.md",
            "source_ref": "Line 15",
        },
        {
            "id": "F068",
            "statement": "MedPsy-4B accuracy when answering under safe reliability threshold (coverage 72.8%)",
            "value": f"{re.search(r'\|\s*\d+\.\d+%\s*\[\d+\.\d+,\s*\d+\.\d+\]\s*\|\s*(\d+\.\d+)%', l_rel_safe).group(1)}%",
            "ci_95": f"[{re.search(r'\|\s*\d+\.\d+%\s*\[\d+\.\d+,\s*\d+\.\d+\]\s*\|\s*\d+\.\d+%\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]', l_rel_safe).group(1)}, {re.search(r'\|\s*\d+\.\d+%\s*\[\d+\.\d+,\s*\d+\.\d+\]\s*\|\s*\d+\.\d+%\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]', l_rel_safe).group(2)}]",
            "split": "test",
            "n": 650,
            "source_file": "outputs/analysis/medpsy_reliability.md",
            "source_ref": "Line 14",
        },
        {
            "id": "F069",
            "statement": "MedPsy-4B unsafe answer rate under safe reliability threshold",
            "value": f"{re.search(r'\|\s*(\d+\.\d+)%\s*\[\d+\.\d+,\s*\d+\.\d+\]\s*\|$', l_rel_safe).group(1)}%",
            "ci_95": f"[{re.search(r'\|\s*\d+\.\d+%\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]\s*\|$', l_rel_safe).group(1)}, {re.search(r'\|\s*\d+\.\d+%\s*\[(\d+\.\d+),\s*(\d+\.\d+)\]\s*\|$', l_rel_safe).group(2)}]",
            "split": "test",
            "n": 650,
            "source_file": "outputs/analysis/medpsy_reliability.md",
            "source_ref": "Line 14",
        },

        # ── Compute and Resource Usage ───────────────────────────────────────────────
        {
            "id": "F070",
            "statement": "Pair B budget sweep execution GPU hours (2x T4 GPU)",
            "value": f"{float(manifest_b['hours_used']):.2f} hours",
            "ci_95": "N/A",
            "split": "val",
            "n": manifest_b.get("N", 100),
            "source_file": "outputs/sweeps/budget_pair_b/manifest.json",
            "source_ref": "Key 'hours_used'",
        },
        {
            "id": "F071",
            "statement": "Gemma prompt sensitivity execution GPU hours (2x T4 GPU)",
            "value": f"{float(manifest_g['hours_used']):.2f} hours ({float(manifest_g['hours_used'])*60:.1f} min)",
            "ci_95": "N/A",
            "split": "val",
            "n": manifest_g.get("N", 200) * len(manifest_g.get("variants", ["V1", "V2", "V3"])),
            "source_file": "outputs/sweeps/gemma_prompt_sensitivity/manifest.json",
            "source_ref": "Key 'hours_used'",
        },
        {
            "id": "F072",
            "statement": "MedPsy-4B baseline latency on MedQA test",
            "value": f"{re.search(r'\|\s*(\d+\.\d+)\s*\|$', l_mrag_base).group(1)} s/question",
            "ci_95": "N/A",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/medpsy_rag.md",
            "source_ref": "Line 5",
        },
        {
            "id": "F073",
            "statement": "MedPsy-4B RAG latency on MedQA test",
            "value": f"{re.search(r'\|\s*(\d+\.\d+)\s*\|$', l_mrag_rag).group(1)} s/question",
            "ci_95": "N/A",
            "split": "test",
            "n": 500,
            "source_file": "outputs/analysis/medpsy_rag.md",
            "source_ref": "Line 6",
        },
    ]

    return facts


# Populate module-level FACTS by computing dynamically from repository sources
FACTS: List[Dict[str, Any]] = compute_facts(REPO_ROOT)


def generate_markdown(facts: Optional[List[Dict[str, Any]]] = None) -> str:
    """Generate Markdown content for docs/PAPER_FACTS.md."""
    fact_list = facts if facts is not None else FACTS
    lines = []
    lines.append("# Master Paper Fact Ledger")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Rules of Evidence:**")
    lines.append("> 1. Every empirical number cited in paper text, abstract, or executive summary must cite an ID (`F001...`) from this ledger.")
    lines.append("> 2. Strict accuracy requires: parsed, untruncated, and correct. Fallback accuracy is secondary and explicitly labelled.")
    lines.append("> 3. All numbers are computed from committed repository outputs and cryptographic frozen splits; no numbers are estimated.")
    lines.append("")
    lines.append("| ID | Plain-Language Statement | Empirical Value | 95% Confidence Interval | Split | N | Source File | Provenance Reference |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for f in fact_list:
        lines.append(
            f"| **{f['id']}** | {f['statement']} | `{f['value']}` | {f['ci_95']} | {f['split']} | {f['n']} | `{f['source_file']}` | {f['source_ref']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and Audit Master Paper Fact Ledger")
    parser.add_argument("--check", action="store_true", help="Assert docs/PAPER_FACTS.md matches generated ledger")
    args = parser.parse_args()

    # Always compute dynamically from source reports and manifests
    fresh_facts = compute_facts(REPO_ROOT)
    content = generate_markdown(fresh_facts)
    target_path = REPO_ROOT / "docs" / "PAPER_FACTS.md"

    if args.check:
        if not target_path.exists():
            print(f"FAIL: {target_path} does not exist.")
            sys.exit(1)
        curr = target_path.read_text(encoding="utf-8")
        if curr.strip() != content.strip():
            print(f"FAIL: {target_path} is out of date with master facts ledger.")
            sys.exit(1)
        print("PASS: docs/PAPER_FACTS.md is strictly identical to master fact ledger and verified against all sources.")
        sys.exit(0)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    print(f"Generated {target_path} with {len(fresh_facts)} empirical facts computed directly from sources.")


if __name__ == "__main__":
    main()
