# Consolidated V3 Experimental Results & Validation Robustness

> [!NOTE]
> **Strict Accuracy (Primary Metric):** A response is scored as strictly correct (1.0) iff it was parsed successfully, finished naturally without truncation (`truncated == False`), and matches gold. All truncated or unparsed outputs count as incorrect (0.0).
> **With-Fallback Accuracy (Secondary Metric):** Raw benchmark accuracy with heuristic letter recovery after truncation.
> All confidence intervals are 95% Wilson score intervals for proportions and 10,000-resample percentile bootstrap intervals for paired differences.

## 1. Pair A: medpsy-1.7b vs qwen3-1.7b-thinking (MedQA Test, N=500)

**Holm Family:** Single family of 6 hypothesis tests across C1 and C2 (Strict Accuracy, Fallback Accuracy, and Truncation Rate differences).

| Condition | Model | N | Strict Acc (95% CI) [Primary] | With-Fallback Acc [Secondary] | Truncation Rate | Parse Rate | Paired Diff (Med - Gen) | Exact McNemar p | Holm-Adjusted p |
|---|---|---|---|---|---|---|---|---|---|
| C1 | `medpsy-1.7b` | 500 | 60.2% [55.8%, 64.4%] | 77.8% [74.0%, 81.2%] | 32.0% [28.1%, 36.2%] | 70.6% [66.5%, 74.4%] | +29.8% [+24.8%, +34.8%] | <1e-4 | <1e-4 |
| C1 | `qwen3-1.7b-thinking` | 500 | 30.4% [26.5%, 34.6%] | 49.8% [45.4%, 54.2%] | 51.0% [46.6%, 55.4%] | 50.2% [45.8%, 54.6%] | — | — | — |
| C2 | `medpsy-1.7b` | 500 | 45.2% [40.9%, 49.6%] | 78.4% [74.6%, 81.8%] | 46.6% [42.3%, 51.0%] | 71.0% [66.9%, 74.8%] | +12.0% [+7.2%, +16.8%] | <1e-4 | <1e-4 |
| C2 | `qwen3-1.7b-thinking` | 500 | 33.2% [29.2%, 37.4%] | 53.2% [48.8%, 57.5%] | 49.4% [45.0%, 53.8%] | 51.4% [47.0%, 55.8%] | — | — | — |

## 2. Pair A: medpsy-1.7b vs qwen3-1.7b-thinking (Unanswerable v2 Test, N=140)

**Holm Family:** Truncation rate differences across C4 and C5 (m=2).

| Condition | Model | N | Strict Acc (95% CI) [Primary] | With-Fallback Acc [Secondary] | Truncation Rate | Parse Rate | Paired Diff (Med - Gen) | Exact McNemar p | Holm-Adjusted p |
|---|---|---|---|---|---|---|---|---|---|
| C4 | `medpsy-1.7b` | 140 | 25.0% [18.6%, 32.8%] | 28.6% [21.7%, 36.5%] | 20.0% [14.2%, 27.4%] | 84.3% [77.4%, 89.4%] | -3.6% [-7.1%, -0.7%] | 0.0625 | 0.0625 |
| C4 | `qwen3-1.7b-thinking` | 140 | 28.6% [21.7%, 36.5%] | 28.6% [21.7%, 36.5%] | 20.7% [14.8%, 28.2%] | 79.3% [71.8%, 85.2%] | — | — | — |
| C5 | `medpsy-1.7b` | 140 | 23.6% [17.3%, 31.2%] | 28.6% [21.7%, 36.5%] | 32.1% [25.0%, 40.3%] | 79.3% [71.8%, 85.2%] | -2.1% [-5.7%, +1.4%] | 0.4531 | 0.4531 |
| C5 | `qwen3-1.7b-thinking` | 140 | 25.7% [19.2%, 33.5%] | 28.6% [21.7%, 36.5%] | 42.9% [35.0%, 51.1%] | 56.4% [48.2%, 64.4%] | — | — | — |

## 3. Gemma Pair: medgemma-1.5-4b-it vs gemma3-4b (Test Split, N=500 per cell)

**Holm Family:** Five paired strict accuracy tests across C1 MedQA, C1 PubMedQA, C2 MedQA, C2 PubMedQA, C3 PubMedQA (m=5).

| Cell | Model | N | Strict Acc (95% CI) [Primary] | With-Fallback Acc [Secondary] | Truncation Rate | Parse Rate | Paired Diff (Med - Gen) | Exact McNemar p | Holm-Adjusted p |
|---|---|---|---|---|---|---|---|---|---|
| C1_MedQA | `medgemma-1.5-4b-it` | 500 | 54.80% [50.4%, 59.1%] | 54.80% [50.4%, 59.1%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | +6.20% [+0.8%, +11.4%] | 0.0255 | 0.1274 |
| C1_MedQA | `gemma3-4b` | 500 | 48.60% [44.2%, 53.0%] | 48.60% [44.2%, 53.0%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | — | — | — |
| C1_PubMedQA | `medgemma-1.5-4b-it` | 500 | 50.80% [46.4%, 55.2%] | 50.80% [46.4%, 55.2%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | +1.60% [-2.4%, +5.4%] | 0.4797 | 0.9594 |
| C1_PubMedQA | `gemma3-4b` | 500 | 49.20% [44.8%, 53.6%] | 49.20% [44.8%, 53.6%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | — | — | — |
| C2_MedQA | `medgemma-1.5-4b-it` | 500 | 57.20% [52.8%, 61.5%] | 57.20% [52.8%, 61.5%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | +4.20% [-0.4%, +8.8%] | 0.0871 | 0.2614 |
| C2_MedQA | `gemma3-4b` | 500 | 53.00% [48.6%, 57.3%] | 53.00% [48.6%, 57.3%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | — | — | — |
| C2_PubMedQA | `medgemma-1.5-4b-it` | 500 | 51.00% [46.6%, 55.4%] | 51.00% [46.6%, 55.4%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | +0.80% [-3.0%, +4.8%] | 0.7666 | 0.9594 |
| C2_PubMedQA | `gemma3-4b` | 500 | 50.20% [45.8%, 54.6%] | 50.20% [45.8%, 54.6%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | — | — | — |
| C3_PubMedQA | `medgemma-1.5-4b-it` | 500 | 57.00% [52.6%, 61.3%] | 57.00% [52.6%, 61.3%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | +4.00% [+0.4%, +7.6%] | 0.0399 | 0.1594 |
| C3_PubMedQA | `gemma3-4b` | 500 | 53.00% [48.6%, 57.3%] | 53.00% [48.6%, 57.3%] | 0.0% [0.0%, 0.8%] | 100.0% [99.2%, 100.0%] | — | — | — |

## 4. Pair B: medpsy-4b vs qwen3-4b-thinking-2507 (MedQA Test, 1,024 Cap, N=500)

**Holm Family:** Two paired strict accuracy tests across C1 and C2 (m=2).

| Condition | Model | N | Strict Acc (95% CI) [Primary] | With-Fallback Acc [Secondary] | Truncation Rate | Parse Rate | Paired Diff (Med - Gen) | Exact McNemar p | Holm-Adjusted p |
|---|---|---|---|---|---|---|---|---|---|
| C1 | `medpsy-4b` | 500 | 83.2% [79.7%, 86.2%] | 87.6% [84.4%, 90.2%] | 10.6% [8.2%, 13.6%] | 90.4% [87.5%, 92.7%] | +82.8% [+79.4%, +86.0%] | <1e-4 | <1e-4 |
| C1 | `qwen3-4b-thinking-2507` | 500 | 0.4% [0.1%, 1.4%] | 66.8% [62.6%, 70.8%] | 99.6% [98.6%, 99.9%] | 2.0% [1.1%, 3.6%] | — | — | — |
| C2 | `medpsy-4b` | 500 | 72.6% [68.5%, 76.3%] | 89.0% [86.0%, 91.5%] | 21.6% [18.2%, 25.4%] | 83.8% [80.3%, 86.8%] | +72.6% [+68.6%, +76.4%] | <1e-4 | <1e-4 |
| C2 | `qwen3-4b-thinking-2507` | 500 | 0.0% [0.0%, 0.8%] | 60.4% [56.0%, 64.6%] | 100.0% [99.2%, 100.0%] | 0.6% [0.2%, 1.7%] | — | — | — |

## 5. Validation Robustness

### 5.1 Pair B Token-Budget Sweep (MedQA Validation Split, N=100)
All caps evaluated via exact single-prefix derivation under greedy decoding (`max_new_tokens=4096`).

| Cap | MedPsy Strict Acc | Qwen Strict Acc | MedPsy Trunc Rate | Qwen Trunc Rate | Paired Strict Diff | McNemar p | Holm-Adjusted p |
|---|---|---|---|---|---|---|---|
| 1024 | 88.0% | 0.0% | 6.0% | 100.0% | +88.0% | <1e-4 | <1e-4 |
| 1536 | 92.0% | 6.0% | 2.0% | 94.0% | +86.0% | <1e-4 | <1e-4 |
| 2048 | 93.0% | 19.0% | 1.0% | 81.0% | +74.0% | <1e-4 | <1e-4 |
| 3072 | 94.0% | 51.0% | 0.0% | 45.0% | +43.0% | <1e-4 | <1e-4 |
| 4096 | 94.0% | 62.0% | 0.0% | 28.0% | +32.0% | <1e-4 | <1e-4 |

### 5.2 Gemma Prompt-Sensitivity Evaluation (MedQA Validation Split, N=200)
Evaluation across three prompt variations (`docs/GEMMA_PROMPT_SENSITIVITY_RULE.md`): V1 (standard baseline), V2 (clinical scenario), V3 (board examination).

- **Pre-Registered Stability Verdict:** `STABLE` (Gap spread across variants: **2.50 pp**, within pre-registered $\le 10.0$ pp threshold; no sign flip).
- **Format-Confound Status:** None flagged (all variants exhibited $\le 5.0$ pp difference in parse rates).

| Variant | Gemma-3-4B Acc | MedGemma-1.5-4B-it Acc | Gap (Gemma3 - MedGemma) | 95% Bootstrap CI | Exact McNemar p | Status |
|---|---|---|---|---|---|---|
| V1 | 51.00% | 61.00% | -10.00 pp | [-18.00, -2.00] | 0.0225 | Not Confounded |
| V2 | 50.00% | 65.50% | -15.50 pp | [-23.50, -7.00] | 0.0004 | Not Confounded |
| V3 | 51.00% | 63.50% | -12.50 pp | [-20.50, -4.50] | 0.0035 | Not Confounded |

## 6. Limitations Flags (Data-Derived)

| Flag ID | Category | Empirical Observation | Scientific Impact |
|---|---|---|---|
| **L01** | Truncation Ceiling | Qwen3-4B-Thinking-2507 suffered 99.6% (498/500) truncation in C1 and 100.0% (500/500) in C2 at 1,024 cap on frozen test. | Strict accuracy collapsed to 0.4% and 0.0%. Test cap represents budget exhaustion rather than reasoning parity. |
| **L02** | Residual Truncation | Qwen3-4B-Thinking-2507 exhibited 28.0% residual truncation at 4,096 tokens on validation sweep. | Reasoning loops persist even at 4x budget cap; unconstrained budget does not eliminate pathological looping. |
| **L03** | RAG Truncation Inflation | MedPsy-1.7B truncation increased from 32.0% in C1 to 46.6% in C2 (+14.6 pp) upon retrieval injection. | Evidence ingestion inflates thinking length, driving a net -15.0 pp strict accuracy degradation under fixed cap. |
| **L04** | Format Sensitivity | Gemma prompt-sensitivity variants yielded 0.00 pp parse rate difference (all 100.0% parsed). | No format confound observed; direct-answer formatting remained perfectly stable across prompt framings. |
| **L05** | Coverage Gaps | Pair B evaluated strictly on MedQA C1 and C2. Conditions C3, C4, C5 were not evaluated. | Cross-condition comparisons for Pair B on PubMedQA and Unanswerable v2 remain empirical gaps. |

## 7. Coverage Gaps Summary

| Experiment Pair | Condition | Dataset | Split | Status | Rationale |
|---|---|---|---|---|---|
| Pair A (1.7B Thinking) | C1, C2 | MedQA | Test (N=500) | **COMPLETE** | Fully evaluated and committed. |
| Pair A (1.7B Thinking) | C4, C5 | Unanswerable v2 | Test (N=140) | **COMPLETE** | Fully evaluated and committed. |
| Pair A (1.7B Thinking) | C3 | PubMedQA | Test (N=500) | **GAP** | Not executed in V3 evaluation protocol. |
| Gemma Pair (4B Direct) | C1, C2 | MedQA | Test (N=500) | **COMPLETE** | Fully evaluated and committed. |
| Gemma Pair (4B Direct) | C1, C2, C3 | PubMedQA | Test (N=500) | **COMPLETE** | Fully evaluated and committed. |
| Gemma Pair (4B Direct) | C4, C5 | Unanswerable v2 | Test (N=140) | **COMPLETE** | Fully evaluated and committed. |
| Pair B (4B Thinking) | C1, C2 | MedQA | Test (N=500) | **COMPLETE** | Fully evaluated and committed. |
| Pair B (4B Thinking) | C3 | PubMedQA | Test (N=500) | **GAP** | Pre-flight aborted in earlier protocol; omitted from V3. |
| Pair B (4B Thinking) | C4, C5 | Unanswerable v2 | Test (N=140) | **GAP** | Omitted from V3 budget run. |
