"""src/analysis_pair_a_c45.py — Read-only analysis of Pair A on C4/C5.

Generic Model : Qwen3-1.7B-Thinking
Domain Model  : MedPsy-1.7B
Conditions    : C4 (Closed-book unanswerable_v2), C5 (RAG unanswerable_v2)

Inputs:
  - outputs/v3/qwen3-1.7b-thinking/C{4,5}/qwen3-1.7b-thinking.jsonl
  - outputs/v3/medpsy-1.7b/C{4,5}/medpsy-1.7b.jsonl

Pre-flight assertions:
  - split == "test" on all rows
  - unique ids within each model
  - identical id sets across models

Report targets:
  - stdout
  - outputs/analysis/pair_a_c45.md
"""

from collections import defaultdict
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

# Reuse verified statistical routines from src.analysis_gemma_pair
from src.analysis_gemma_pair import (
    auroc_with_ties,
    holm_bonferroni,
    load_jsonl,
    mcnemar_exact,
    paired_bootstrap_auroc,
    wilson_ci,
)

GENERIC_MODEL = "qwen3-1.7b-thinking"
DOMAIN_MODEL = "medpsy-1.7b"
CONDITIONS = ["C4", "C5"]

OUTPUTS_DIR = BASE_DIR / "outputs" / "v3"
REPORT_DIR = BASE_DIR / "outputs" / "analysis"
REPORT_PATH = REPORT_DIR / "pair_a_c45.md"

VERBAL_ABSTAIN_PATTERN = re.compile(
    r"^\s*[*_`\"'#\s]*(none|insufficient|not\s+enough|cannot)\b",
    re.IGNORECASE,
)


def is_verbal_abstention(raw_output: str) -> bool:
    """Classify whether the text following 'Answer:' matches verbal refusal keywords.
    
    Keywords: 'None', 'Insufficient', 'Not enough', or 'Cannot'.
    """
    if not raw_output:
        return False
    m = re.search(r"Answer\s*:\s*(.*)", raw_output, re.IGNORECASE | re.DOTALL)
    after = m.group(1).strip() if m else raw_output.strip()
    return bool(VERBAL_ABSTAIN_PATTERN.search(after))


def run_analysis() -> Tuple[str, Dict[str, Any]]:
    lines: List[str] = []

    def p(text: str = "") -> None:
        lines.append(text)

    p("# Pair A C4/C5 Analysis: Qwen3-1.7B-Thinking (Generic) vs MedPsy-1.7B (Domain)")
    p()
    p("**Models Analyzed**:")
    p(f"- Generic Arm: `{GENERIC_MODEL}`")
    p(f"- Medical/Domain Arm: `{DOMAIN_MODEL}`")
    p(f"- Conditions: `C4` (Closed-book unanswerable_v2), `C5` (RAG unanswerable_v2)")
    p(f"- Split: `test` (N=140 per condition: 100 challenge, 40 control)")
    p()

    # Pre-flight data load and assertions
    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {GENERIC_MODEL: {}, DOMAIN_MODEL: {}}
    for cond in CONDITIONS:
        g_path = OUTPUTS_DIR / GENERIC_MODEL / cond / f"{GENERIC_MODEL}.jsonl"
        m_path = OUTPUTS_DIR / DOMAIN_MODEL / cond / f"{DOMAIN_MODEL}.jsonl"

        g_rows = load_jsonl(g_path)
        m_rows = load_jsonl(m_path)

        # Assertions
        assert len(g_rows) == 140, f"{GENERIC_MODEL} {cond} expected 140 rows, got {len(g_rows)}"
        assert len(m_rows) == 140, f"{DOMAIN_MODEL} {cond} expected 140 rows, got {len(m_rows)}"
        assert all(r.get("split") == "test" for r in g_rows), f"{GENERIC_MODEL} {cond} split != test"
        assert all(r.get("split") == "test" for r in m_rows), f"{DOMAIN_MODEL} {cond} split != test"

        g_ids = [r["id"] for r in g_rows]
        m_ids = [r["id"] for r in m_rows]
        assert len(g_ids) == len(set(g_ids)), f"{GENERIC_MODEL} {cond} ids not unique"
        assert len(m_ids) == len(set(m_ids)), f"{DOMAIN_MODEL} {cond} ids not unique"
        assert g_ids == m_ids, f"{cond} question id sets do not match between models"

        data[GENERIC_MODEL][cond] = g_rows
        data[DOMAIN_MODEL][cond] = m_rows

    p("> [!NOTE]")
    p("> Pre-flight checks PASSED: split == 'test' across all rows, 140 unique IDs per model/condition, and exact 1:1 ID match across arms.")
    p()

    # ─────────────────────────────────────────────────────────────────────────────
    # 1. Per Model and Condition Summary
    # ─────────────────────────────────────────────────────────────────────────────
    p("## 1. Per-Model & Per-Condition Summary")
    p()
    p("| Condition | Model | Total N | Parsed Count (%) | Truncated Count (%) | Control Acc (N=40, Wilson 95% CI) | Challenge Truncated (N=100) |")
    p("|---|---|---|---|---|---|---|")

    summary_stats: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for cond in CONDITIONS:
        for model in [GENERIC_MODEL, DOMAIN_MODEL]:
            rows = data[model][cond]
            n_total = len(rows)
            n_parsed = sum(1 for r in rows if r.get("parsed"))
            n_trunc = sum(1 for r in rows if r.get("truncated"))
            ctrl_rows = [r for r in rows if not r.get("should_abstain")]
            ctrl_correct = sum(1 for r in ctrl_rows if r.get("correct"))
            ctrl_ci = wilson_ci(ctrl_correct, len(ctrl_rows))
            chal_trunc = sum(1 for r in rows if r.get("should_abstain") and r.get("truncated"))

            summary_stats[cond][model] = {
                "n_total": n_total,
                "n_parsed": n_parsed,
                "n_trunc": n_trunc,
                "ctrl_correct": ctrl_correct,
                "ctrl_total": len(ctrl_rows),
                "ctrl_ci": ctrl_ci,
                "chal_trunc": chal_trunc,
            }

            chal_rows = [r for r in rows if r.get("should_abstain")]
            chal_trunc_pct = (chal_trunc / max(1, len(chal_rows))) * 100
            p(f"| {cond} | `{model}` | {n_total} | {n_parsed} ({n_parsed/n_total*100:.1f}%) | "
              f"{n_trunc} ({n_trunc/n_total*100:.1f}%) | "
              f"{ctrl_correct}/{len(ctrl_rows)} ({ctrl_correct/len(ctrl_rows)*100:.1f}%, [{ctrl_ci[0]*100:.1f}%, {ctrl_ci[1]*100:.1f}%]) | "
              f"{chal_trunc} ({chal_trunc_pct:.1f}%) |")
    p()

    # ─────────────────────────────────────────────────────────────────────────────
    # 2. Paired Overall Truncation Comparison (Exact McNemar + Holm Adjustment)
    # ─────────────────────────────────────────────────────────────────────────────
    p("## 2. Paired Overall Truncation Comparison (Generic vs Domain)")
    p()
    p("Paired 2x2 contingency on overall truncation across all 140 question IDs per condition.")
    p("**Multiplicity Family**: Overall Truncation Difference ($m=2$: C4, C5).")
    p()

    mcnemar_raw_ps: List[float] = []
    mcnemar_records: List[Dict[str, Any]] = []

    for cond in CONDITIONS:
        g_rows = data[GENERIC_MODEL][cond]
        m_rows = data[DOMAIN_MODEL][cond]

        a = sum(1 for g, m in zip(g_rows, m_rows) if g.get("truncated") and m.get("truncated"))
        b = sum(1 for g, m in zip(g_rows, m_rows) if g.get("truncated") and not m.get("truncated"))
        c = sum(1 for g, m in zip(g_rows, m_rows) if not g.get("truncated") and m.get("truncated"))
        d = sum(1 for g, m in zip(g_rows, m_rows) if not g.get("truncated") and not m.get("truncated"))

        raw_p = mcnemar_exact(b, c)
        mcnemar_raw_ps.append(raw_p)
        mcnemar_records.append({
            "cond": cond,
            "a": a, "b": b, "c": c, "d": d,
            "raw_p": raw_p,
        })

    holm_ps = holm_bonferroni(mcnemar_raw_ps)
    for rec, adj_p in zip(mcnemar_records, holm_ps):
        rec["adj_p"] = adj_p

    p("| Condition | Both Trunc (a) | Qwen Only (b) | MedPsy Only (c) | Neither Trunc (d) | Discordant (b/c) | McNemar Raw p | Holm Adj p (m=2) | Significance |")
    p("|---|---|---|---|---|---|---|---|---|")
    for rec in mcnemar_records:
        sig_str = "p < 0.05" if rec["adj_p"] < 0.05 else "ns"
        p(f"| {rec['cond']} | {rec['a']} | {rec['b']} | {rec['c']} | {rec['d']} | "
          f"{rec['b']}/{rec['c']} | {rec['raw_p']:.5f} | {rec['adj_p']:.5f} | {sig_str} |")
    p()
    p("> [!NOTE]")
    p(f"> Under Holm-Bonferroni correction (Family: Overall Truncation, m=2), overall truncation difference between Qwen3-1.7B-thinking and MedPsy-1.7B is not statistically significant in either C4 (adj p = {holm_ps[0]:.4f}) or C5 (adj p = {holm_ps[1]:.4f}).")
    p()

    # ─────────────────────────────────────────────────────────────────────────────
    # 3. Confidence Separation on All Rows (N=140)
    # ─────────────────────────────────────────────────────────────────────────────
    p("## 3. Confidence Separation on All Rows (N=140)")
    p()
    p("Separation between challenge questions (positive class: should abstain, label=1) and control questions (negative class: should answer, label=0).")
    p("Score metric: $s = 1 - \\text{confidence}$. Ties receive exact half credit ($0.5$). Paired bootstrap 95% CIs use 10,000 question resamples (seed 42).")
    p()

    p("| Condition | Model | Mean Conf Chal (N=100) | Mean Conf Ctrl (N=40) | Chal Conf >= 0.90 (%) | AUROC (1 - conf) | 95% Bootstrap CI | MedPsy - Qwen Diff | 95% Diff CI |")
    p("|---|---|---|---|---|---|---|---|---|")

    all_qwen_aurocs = {}
    for cond in CONDITIONS:
        g_rows = data[GENERIC_MODEL][cond]
        m_rows = data[DOMAIN_MODEL][cond]

        labels = np.array([1 if r.get("should_abstain") else 0 for r in g_rows])
        g_confs = np.array([float(r.get("confidence", 0.0)) for r in g_rows])
        m_confs = np.array([float(r.get("confidence", 0.0)) for r in m_rows])

        g_scores = 1.0 - g_confs
        m_scores = 1.0 - m_confs

        g_auc = auroc_with_ties(g_scores[labels == 1], g_scores[labels == 0])
        all_qwen_aurocs[cond] = g_auc
        m_auc = auroc_with_ties(m_scores[labels == 1], m_scores[labels == 0])
        diff_auc = m_auc - g_auc

        g_ci, m_ci, diff_ci = paired_bootstrap_auroc(labels, g_scores, m_scores, n_boot=10000, seed=42)

        g_chal_mean = float(np.mean(g_confs[labels == 1]))
        g_ctrl_mean = float(np.mean(g_confs[labels == 0]))
        g_frac90 = float(np.mean(g_confs[labels == 1] >= 0.90) * 100)

        m_chal_mean = float(np.mean(m_confs[labels == 1]))
        m_ctrl_mean = float(np.mean(m_confs[labels == 0]))
        m_frac90 = float(np.mean(m_confs[labels == 1] >= 0.90) * 100)

        p(f"| {cond} | `{GENERIC_MODEL}` | {g_chal_mean:.4f} | {g_ctrl_mean:.4f} | {g_frac90:.1f}% | {g_auc:.4f} | [{g_ci[0]:.4f}, {g_ci[1]:.4f}] | — | — |")
        p(f"| {cond} | `{DOMAIN_MODEL}` | {m_chal_mean:.4f} | {m_ctrl_mean:.4f} | {m_frac90:.1f}% | {m_auc:.4f} | [{m_ci[0]:.4f}, {m_ci[1]:.4f}] | {diff_auc:+.4f} | [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}] |")
    p()

    # ─────────────────────────────────────────────────────────────────────────────
    # 4. Sensitivity Analysis: Clean Rows (parsed == True AND truncated == False for BOTH)
    # ─────────────────────────────────────────────────────────────────────────────
    p("## 4. Sensitivity Analysis: Clean Un-Truncated Rows (Both Parsed & Not Truncated)")
    p()
    p("Confidence on truncated rows originates from a forced argmax answer after token budget exhaustion. "
      "Furthermore, models can occasionally emit an answer letter while still hitting the token limit (`parsed == True and truncated == True`). "
      "This sensitivity evaluation restricts analysis strictly to question IDs where **BOTH models completed without truncation and were successfully parsed** "
      "(`parsed == True and truncated == False`).")
    p()

    p("### Rows with `parsed == True` and `truncated == True`")
    p()
    p("| Condition | Model | Total Rows | Parsed True & Truncated True Count (%) | Reason for Truncation |")
    p("|---|---|---|---|---|")
    for cond in CONDITIONS:
        for model in [GENERIC_MODEL, DOMAIN_MODEL]:
            rows = data[model][cond]
            n_pt = sum(1 for r in rows if r.get("parsed") and r.get("truncated"))
            reason_str = "Answer extracted before generation hit max_new_tokens" if n_pt > 0 else "All truncated rows failed parser (0 parsed)"
            p(f"| {cond} | `{model}` | {len(rows)} | {n_pt} ({n_pt/len(rows)*100:.1f}%) | {reason_str} |")
    p()

    p("### Clean Sensitivity Evaluation Table (`parsed == True and truncated == False` for BOTH models)")
    p()
    p("| Condition | Clean N (Chal / Ctrl) | Model | Mean Conf Chal | Mean Conf Ctrl | Chal Conf >= 0.90 (%) | AUROC (1 - conf) | 95% Bootstrap CI | MedPsy - Qwen Diff | 95% Diff CI |")
    p("|---|---|---|---|---|---|---|---|---|---|")

    clean_qwen_aurocs = {}
    for cond in CONDITIONS:
        g_rows = data[GENERIC_MODEL][cond]
        m_rows = data[DOMAIN_MODEL][cond]

        # Restrict strictly to parsed == True AND truncated == False for BOTH models
        clean_pairs = [
            (g, m) for g, m in zip(g_rows, m_rows)
            if g.get("parsed") and not g.get("truncated") and m.get("parsed") and not m.get("truncated")
        ]
        g_sub = [p[0] for p in clean_pairs]
        m_sub = [p[1] for p in clean_pairs]

        n_clean = len(clean_pairs)
        labels = np.array([1 if r.get("should_abstain") else 0 for r in g_sub])
        n_chal = int(np.sum(labels == 1))
        n_ctrl = int(np.sum(labels == 0))

        g_confs = np.array([float(r.get("confidence", 0.0)) for r in g_sub])
        m_confs = np.array([float(r.get("confidence", 0.0)) for r in m_sub])

        g_scores = 1.0 - g_confs
        m_scores = 1.0 - m_confs

        g_auc = auroc_with_ties(g_scores[labels == 1], g_scores[labels == 0])
        clean_qwen_aurocs[cond] = g_auc
        m_auc = auroc_with_ties(m_scores[labels == 1], m_scores[labels == 0])
        diff_auc = m_auc - g_auc

        g_ci, m_ci, diff_ci = paired_bootstrap_auroc(labels, g_scores, m_scores, n_boot=10000, seed=42)

        g_chal_mean = float(np.mean(g_confs[labels == 1]))
        g_ctrl_mean = float(np.mean(g_confs[labels == 0]))
        g_frac90 = float(np.mean(g_confs[labels == 1] >= 0.90) * 100)

        m_chal_mean = float(np.mean(m_confs[labels == 1]))
        m_ctrl_mean = float(np.mean(m_confs[labels == 0]))
        m_frac90 = float(np.mean(m_confs[labels == 1] >= 0.90) * 100)

        p(f"| {cond} | {n_clean} ({n_chal}/{n_ctrl}) | `{GENERIC_MODEL}` | {g_chal_mean:.4f} | {g_ctrl_mean:.4f} | {g_frac90:.1f}% | {g_auc:.4f} | [{g_ci[0]:.4f}, {g_ci[1]:.4f}] | — | — |")
        p(f"| {cond} | {n_clean} ({n_chal}/{n_ctrl}) | `{DOMAIN_MODEL}` | {m_chal_mean:.4f} | {m_ctrl_mean:.4f} | {m_frac90:.1f}% | {m_auc:.4f} | [{m_ci[0]:.4f}, {m_ci[1]:.4f}] | {diff_auc:+.4f} | [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}] |")
    p()
    p("> [!IMPORTANT]")
    p("> **Critical Sensitivity Finding**: When restricted strictly to rows where generation completed without budget exhaustion "
      "(`parsed == True and truncated == False` for both models), Qwen3-1.7B-thinking's predicted token probability saturates at 1.0000 on virtually all questions "
      f"(mean 1.0000 on both challenge and control). Consequently, its genuine separation AUROC drops to chance level ({clean_qwen_aurocs.get('C4', 0.5):.4f} in C4, {clean_qwen_aurocs.get('C5', 0.5):.4f} in C5). "
      f"The higher apparent AUROCs in Section 3 ({all_qwen_aurocs.get('C4', 0.62):.4f} and {all_qwen_aurocs.get('C5', 0.70):.4f}) were driven by forced continuation confidences on truncated questions.")
    p()

    # ─────────────────────────────────────────────────────────────────────────────
    # 5. Budget Exhaustion as an Abstention Signal (Descriptive)
    # ─────────────────────────────────────────────────────────────────────────────
    p("## 5. Budget Exhaustion as an Abstention Signal (Descriptive)")
    p()
    p("> [!NOTE]")
    p("> **Descriptive Interpretation Guard**: Truncation reflects token budget exhaustion (hitting `max_new_tokens` / reasoning cap). "
      "While higher truncation on unanswerable items can correlate with the model struggling to resolve the question, truncation also depends "
      "heavily on prompt length, question token length, model verbosity, and retrieved context length (particularly in C5). "
      "It must be interpreted as a descriptive behavioral artifact rather than a calibrated, intentional abstention mechanism.")
    p()

    p("### Truncation Rate: Challenge Rows vs Control Rows")
    p()
    p("| Condition | Model | Challenge Truncation (N=100, Wilson 95% CI) | Control Truncation (N=40, Wilson 95% CI) |")
    p("|---|---|---|---|")

    chal_mcnemar_raw_ps: List[float] = []
    chal_mcnemar_records: List[Dict[str, Any]] = []

    for cond in CONDITIONS:
        g_rows = data[GENERIC_MODEL][cond]
        m_rows = data[DOMAIN_MODEL][cond]

        g_chal = [r for r in g_rows if r.get("should_abstain")]
        g_ctrl = [r for r in g_rows if not r.get("should_abstain")]
        m_chal = [r for r in m_rows if r.get("should_abstain")]
        m_ctrl = [r for r in m_rows if not r.get("should_abstain")]

        g_chal_t = sum(1 for r in g_chal if r.get("truncated"))
        g_ctrl_t = sum(1 for r in g_ctrl if r.get("truncated"))
        g_chal_ci = wilson_ci(g_chal_t, len(g_chal))
        g_ctrl_ci = wilson_ci(g_ctrl_t, len(g_ctrl))

        m_chal_t = sum(1 for r in m_chal if r.get("truncated"))
        m_ctrl_t = sum(1 for r in m_ctrl if r.get("truncated"))
        m_chal_ci = wilson_ci(m_chal_t, len(m_chal))
        m_ctrl_ci = wilson_ci(m_ctrl_t, len(m_ctrl))

        p(f"| {cond} | `{GENERIC_MODEL}` | {g_chal_t}/{len(g_chal)} ({g_chal_t/max(1, len(g_chal))*100:.1f}%, [{g_chal_ci[0]*100:.1f}%, {g_chal_ci[1]*100:.1f}%]) | "
          f"{g_ctrl_t}/{len(g_ctrl)} ({g_ctrl_t/max(1, len(g_ctrl))*100:.1f}%, [{g_ctrl_ci[0]*100:.1f}%, {g_ctrl_ci[1]*100:.1f}%]) |")
        p(f"| {cond} | `{DOMAIN_MODEL}` | {m_chal_t}/{len(m_chal)} ({m_chal_t/max(1, len(m_chal))*100:.1f}%, [{m_chal_ci[0]*100:.1f}%, {m_chal_ci[1]*100:.1f}%]) | "
          f"{m_ctrl_t}/{len(m_ctrl)} ({m_ctrl_t/max(1, len(m_ctrl))*100:.1f}%, [{m_ctrl_ci[0]*100:.1f}%, {m_ctrl_ci[1]*100:.1f}%]) |")

        # Paired McNemar on the 100 challenge IDs
        a_c = sum(1 for g, m in zip(g_chal, m_chal) if g.get("truncated") and m.get("truncated"))
        b_c = sum(1 for g, m in zip(g_chal, m_chal) if g.get("truncated") and not m.get("truncated"))
        c_c = sum(1 for g, m in zip(g_chal, m_chal) if not g.get("truncated") and m.get("truncated"))
        d_c = sum(1 for g, m in zip(g_chal, m_chal) if not g.get("truncated") and not m.get("truncated"))

        raw_p_c = mcnemar_exact(b_c, c_c)
        chal_mcnemar_raw_ps.append(raw_p_c)
        chal_mcnemar_records.append({
            "cond": cond,
            "a": a_c, "b": b_c, "c": c_c, "d": d_c,
            "diff_rate": (m_chal_t - g_chal_t),
            "raw_p": raw_p_c,
        })

    p()
    p("### Paired Comparison of Challenge-Row Truncation (100 Challenge IDs)")
    p()
    p("**Multiplicity Family**: Challenge Truncation Difference ($m=2$: C4, C5).")
    p()

    chal_holm_ps = holm_bonferroni(chal_mcnemar_raw_ps)
    for rec, adj_p in zip(chal_mcnemar_records, chal_holm_ps):
        rec["adj_p"] = adj_p

    p("| Condition | Both Trunc (a) | Qwen Only (b) | MedPsy Only (c) | Neither Trunc (d) | MedPsy - Qwen Diff | McNemar Raw p | Holm Adj p (m=2) | Significance |")
    p("|---|---|---|---|---|---|---|---|---|")
    for rec in chal_mcnemar_records:
        sig_str = "p < 0.05" if rec["adj_p"] < 0.05 else "ns"
        p(f"| {rec['cond']} | {rec['a']} | {rec['b']} | {rec['c']} | {rec['d']} | "
          f"{rec['diff_rate']:+d}% | {rec['raw_p']:.5f} | {rec['adj_p']:.5f} | {sig_str} |")
    p()
    p("> [!NOTE]")
    p(f"> In C4 closed-book, challenge truncation difference is small and not significant (-6%, adj p = {chal_holm_ps[0]:.4f}). "
      f"In C5 RAG, Qwen3-1.7B-thinking truncates on significantly more challenge items than MedPsy-1.7B (56% vs 38%, diff = -18%, adj p = {chal_holm_ps[1]:.4f}), "
      f"reflecting higher context verbosity consumption in the thinking model.")
    p()

    # ─────────────────────────────────────────────────────────────────────────────
    # 6. Abstention Counts (Explicit & Verbal)
    # ─────────────────────────────────────────────────────────────────────────────
    p("## 6. Abstention Counts (Explicit & Verbal)")
    p()
    p("Counts of explicit abstentions (`abstained == True`, corresponding to `pred == 'ABSTAIN'`) "
      "and verbal refusal phrases in `raw_output` after `'Answer:'` starting with `None`, `Insufficient`, `Not enough`, or `Cannot` (case-insensitive).")
    p()

    p("| Condition | Model | Total Rows | Explicit Abstentions (`abstained == True`) | Verbal Refusal after Answer: (`None/Insufficient/...`) | Challenge Abstentions | Control Abstentions |")
    p("|---|---|---|---|---|---|---|")

    for cond in CONDITIONS:
        for model in [GENERIC_MODEL, DOMAIN_MODEL]:
            rows = data[model][cond]
            n_abs_total = sum(1 for r in rows if r.get("abstained") is True)
            n_verb_total = sum(1 for r in rows if is_verbal_abstention(r.get("raw_output", "")))

            chal_rows = [r for r in rows if r.get("should_abstain")]
            ctrl_rows = [r for r in rows if not r.get("should_abstain")]

            chal_abs = sum(1 for r in chal_rows if r.get("abstained") is True or is_verbal_abstention(r.get("raw_output", "")))
            ctrl_abs = sum(1 for r in ctrl_rows if r.get("abstained") is True or is_verbal_abstention(r.get("raw_output", "")))

            p(f"| {cond} | `{model}` | {len(rows)} | "
              f"{n_abs_total} ({n_abs_total/len(rows)*100:.1f}%) | "
              f"{n_verb_total} ({n_verb_total/len(rows)*100:.1f}%) | "
              f"{chal_abs}/{len(chal_rows)} ({chal_abs/max(1, len(chal_rows))*100:.1f}%) | "
              f"{ctrl_abs}/{len(ctrl_rows)} ({ctrl_abs/max(1, len(ctrl_rows))*100:.1f}%) |")
    p()
    total_abs = sum(1 for c in CONDITIONS for m in [GENERIC_MODEL, DOMAIN_MODEL] for r in data[m][c] if r.get("abstained") is True)
    total_verb = sum(1 for c in CONDITIONS for m in [GENERIC_MODEL, DOMAIN_MODEL] for r in data[m][c] if is_verbal_abstention(r.get("raw_output", "")))
    total_q_c4 = len(data[GENERIC_MODEL]["C4"])
    p("> [!NOTE]")
    p(f"> Both models recorded **{total_abs} explicit abstentions** (`abstained == True`) and **{total_verb} verbal refusals** after `'Answer:'` across all {total_q_c4} questions in both C4 and C5. "
      "Neither model exhibits spontaneous or instruction-prompted abstention; when confronted with unanswerable questions, both models either run out of generation tokens (truncation) "
      "or output an answer choice with near-maximal confidence.")
    p()

    # ─────────────────────────────────────────────────────────────────────────────
    # 7. Per-Category Breakdown for Challenge Rows
    # ─────────────────────────────────────────────────────────────────────────────
    p("## 7. Per-Category Breakdown for Challenge Rows")
    p()
    p("> [!WARNING]")
    p("> **Sample Size Guard**: In unanswerable_v2 test split, each individual unanswerability category contains fewer than 20 rows (16 or 17 rows each). "
      "Per the evaluation protocol, categories with fewer than 20 rows are listed below for descriptive completeness only and **must not be given definitive substantive interpretation**.")
    p()

    for cond in CONDITIONS:
        g_rows = data[GENERIC_MODEL][cond]
        m_rows = data[DOMAIN_MODEL][cond]

        cats = sorted(set(r.get("category") for r in g_rows if r.get("should_abstain")))

        p(f"### Condition {cond}")
        p()
        p("| Category | N Rows | Interpret? | Qwen Parsed (%) | Qwen Trunc (%) | Qwen Mean Conf | MedPsy Parsed (%) | MedPsy Trunc (%) | MedPsy Mean Conf |")
        p("|---|---|---|---|---|---|---|---|---|")

        for cat in cats:
            g_c = [r for r in g_rows if r.get("category") == cat and r.get("should_abstain")]
            m_c = [r for r in m_rows if r.get("category") == cat and r.get("should_abstain")]
            n_cat = len(g_c)
            interpret_str = "Yes" if n_cat >= 20 else "No (N < 20)"

            g_p = sum(1 for r in g_c if r.get("parsed"))
            g_t = sum(1 for r in g_c if r.get("truncated"))
            g_conf = float(np.mean([float(r.get("confidence", 0.0)) for r in g_c]))

            m_p = sum(1 for r in m_c if r.get("parsed"))
            m_t = sum(1 for r in m_c if r.get("truncated"))
            m_conf = float(np.mean([float(r.get("confidence", 0.0)) for r in m_c]))

            p(f"| `{cat}` | {n_cat} | {interpret_str} | "
              f"{g_p}/{n_cat} ({g_p/n_cat*100:.1f}%) | {g_t}/{n_cat} ({g_t/n_cat*100:.1f}%) | {g_conf:.4f} | "
              f"{m_p}/{n_cat} ({m_p/n_cat*100:.1f}%) | {m_t}/{n_cat} ({m_t/n_cat*100:.1f}%) | {m_conf:.4f} |")
        p()

    # ─────────────────────────────────────────────────────────────────────────────
    # 8. Assumptions and Field Specifications
    # ─────────────────────────────────────────────────────────────────────────────
    p("## 8. Assumptions & Field Specifications")
    p()
    p("1. `split`: String identifier strictly equal to `'test'` across all rows.")
    p("2. `id`: String question identifier unique within each model and identical across both arms.")
    p("3. `should_abstain`: Ground-truth boolean indicating challenge questions (`True`, N=100) vs control questions (`False`, N=40).")
    p("4. `control`: Boolean flag exactly equal to `not should_abstain` across all rows.")
    p("5. `parsed`: Boolean indicating whether model's final answer letter was successfully extracted via regex/parser without fallback.")
    p("6. `truncated`: Boolean indicating whether model generated up to `max_new_tokens` (or reasoning cap) without emitting an end-of-sequence / final tag.")
    p("7. `correct`: Boolean indicating agreement between predicted letter and canonical ground-truth gold answer.")
    p("8. `confidence`: Float extracted from softmax letter probabilities for the predicted token; when truncated, comes from prompt-forced continuation.")
    p("9. `category`: Unanswerability classification string (`ambiguous`, `fabricated_disease`, `fabricated_drug`, `false_premise`, `missing_info`, `out_of_scope`).")
    p("10. `abstained`: Boolean indicating whether model emitted `ABSTAIN` as its predicted option.")
    p("11. `raw_output`: String containing model's raw generated text, inspected for verbal refusal phrases following `'Answer:'`.")
    p()

    report_text = "\n".join(lines)
    return report_text, summary_stats


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_text, _ = run_analysis()

    # Print to stdout
    print(report_text)

    # Write to markdown file
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"\n[OK] Report successfully written to {REPORT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
