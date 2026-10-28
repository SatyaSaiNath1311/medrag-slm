# MedRAG-SLM Chatbot Deployment Model Selection Report

**Date:** 2026-10-28  
**Rule Specification:** `docs/MODEL_SELECTION_RULE.md`  
**Selected Deployment Model:** `medpsy-4b`  
**Selected Deployment Cap:** `1536` tokens  
**Selection Status:** **SELECTED**  

---

## 1. Executive Summary & Selection Decision

> [!IMPORTANT]
> **Official Selection Decision:**
> **Model:** `medpsy-4b` | **Cap:** `1536` tokens
>
> Selected on strict accuracy: MedPsy-4B achieves 92.0% vs MedGemma-1.5-4B-it's 57.0% (gap +35.0 pp, p = <1e-4, non-overlapping CIs). High-accuracy model selected regardless of latency.

## 2. Candidate Overview & Eligibility Audit

Following `docs/MODEL_SELECTION_RULE.md`, candidates are evaluated on the MedQA validation split ($N=100$ shared questions in identical file order).
Eligibility requires: (a) Truncation Rate $\le 15.0\%$, (b) Parse Rate $\ge 95.0\%$, and (c) execution feasibility on Kaggle 2x T4.

| Candidate | Model Type | Deployment Cap | Truncation Rate (95% CI) | Truncation $\le 15\%$? | Parse Rate (95% CI) | Parse $\ge 95\%$? | Hardware Feasibility | Eligibility |
|---|---|---|---|---|---|---|---|---|
| `medpsy-4b` | Medical Reasoning SLM | 1536 tokens | 2.0% [0.6%, 7.0%] | PASS | 100.0% [96.3%, 100.0%] | PASS | Kaggle 2x T4 (VRAM not measured, to be measured in the live test) | **ELIGIBLE** |
| `medgemma-1.5-4b-it` | Medical Direct-Answer SLM | 16 tokens | 0.0% [0.0%, 3.7%] | PASS | 100.0% [96.3%, 100.0%] | PASS | Kaggle 2x T4 (VRAM not measured, to be measured in the live test) | **ELIGIBLE** |

## 3. Deployment Cap Selection for Thinking-Style SLM (`medpsy-4b`)

The deployment cap for `medpsy-4b` is the smallest cap in the pre-registered grid $[1024, 1536, 2048, 3072, 4096]$ at which validation truncation is $\le 5.0\%$.

| Candidate Cap | Truncation Rate (95% CI) | Truncation $\le 5.0\%$? | Strict Accuracy (95% CI) | Selection Status |
|---|---|---|---|---|
| 1024 tokens | 6.0% [2.8%, 12.5%] | FAIL | 88.0% [80.2%, 93.0%] | Evaluated |
| 1536 tokens | 2.0% [0.6%, 7.0%] | PASS | 92.0% [85.0%, 95.9%] | **SELECTED CAP** |
| 2048 tokens | 1.0% [0.2%, 5.4%] | PASS | 93.0% [86.3%, 96.6%] | Evaluated |
| 3072 tokens | 0.0% [0.0%, 3.7%] | PASS | 94.0% [87.5%, 97.2%] | Evaluated |
| 4096 tokens | 0.0% [0.0%, 3.7%] | PASS | 94.0% [87.5%, 97.2%] | Evaluated |

> [!NOTE]
> At cap=1024, `medpsy-4b` exhibits 6.0% truncation (> 5.0%). At cap=1536, truncation drops to 2.0% ($\le 5.0\%$), achieving the qualification threshold. Hence, **1536 tokens** is selected.

## 4. Head-to-Head Comparison on Shared Validation Questions ($N=100$)

| Metric | `medpsy-4b` (Chosen Cap: 1536) | `medgemma-1.5-4b-it` (Cap: 16) | Paired Difference (MedPsy - MedGemma) | Statistical Significance |
|---|---|---|---|---|
| **Strict Accuracy [Primary]** | **92.0%** [85.0%, 95.9%] | **57.0%** [47.2%, 66.3%] | **+35.0 pp** [+25.0, +45.0] | Exact McNemar $p = <1e-4$ (Significant) |
| With-Fallback Accuracy | 94.0% [87.5%, 97.2%] | 57.0% [47.2%, 66.3%] | +37.0 pp | — |
| Truncation Rate | 2.0% [0.6%, 7.0%] | 0.0% [0.0%, 3.7%] | +2.0 pp | — |
| Parse Rate | 100.0% [96.3%, 100.0%] | 100.0% [96.3%, 100.0%] | +0.0 pp | Both 100.0% / 100.0% parsed |
| Measured Latency (Mean / Median) | 33.58 s / 29.67 s | 0.90 s / 0.90 s | +32.68 s / q | MedGemma is 37.3x faster |
| Total Validation Wall-Clock | 0.9328 hrs (100 q) | 0.1600 hrs (600 q) | — | Manifest logged wall-clock |

## 5. Accuracy vs. Latency Trade-Off Analysis

The empirical data demonstrates a stark architectural trade-off:
1. **Clinical Reasoning Advantage:** `medpsy-4b` achieves **92.0% strict accuracy** at cap 1536 tokens on the shared validation set, compared to **57.0%** for `medgemma-1.5-4b-it` (a statistically significant advantage of **+35.0 percentage points**, $p < <1e-4$, with discordant pair counts $b=37, c=2$).
2. **Inference Latency Cost:** `medpsy-4b` averages **33.58 seconds per question** during extended reasoning, whereas `medgemma-1.5-4b-it` generates its direct-answer letter in **0.90 seconds per question** (~37x faster).
3. **Operational Recommendation:** Because community health worker clinical guidance demands diagnostic fidelity over millisecond response times, the pre-registered decision rule strictly favors the significantly more accurate reasoning model. The interactive UI in `kaggle_chat/chat_app.py` accounts for this latency with an active `🦺 Thinking… (about 20–40 s)` progress indicator.

## 6. Chatbot Configuration Directives (`kaggle_chat/chat_app.py`)

To reflect this selection rule in deployment:
- **Primary Engine:** `medpsy-4b` (`qvac/MedPsy-4B`).
- **Deployment Cap:** Increase `max_new_tokens_reasoning` from 1024 to `1536` to match the validation qualification threshold and reduce truncation from 6.0% to 2.0%.
- **Device Placement:** `medpsy-4b` loaded exclusively on `cuda:0` (~8 GB VRAM in fp16).
- **Fast Fallback:** Keep `qwen3-1.7b` (or `medgemma-1.5-4b-it` if added) as an optional fast dropdown choice for rapid triage queries.

## 7. Methodological Limitations

> [!WARNING]
> **Offline Multiple-Choice Proxy Limitation:**
> The accuracy evaluated here is measured on multiple-choice examination questions (MedQA) with unambiguous reference options A–D. While MCQ accuracy reflects foundational biomedical domain knowledge and clinical reasoning consistency, it serves as an offline proxy for the quality of interactive, free-text dialogue with community health workers. Free-text health worker guidance lacks gold reference labels and requires qualitative human clinical oversight and live field testing.
