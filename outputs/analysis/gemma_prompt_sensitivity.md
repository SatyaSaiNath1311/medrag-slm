# Gemma Prompt-Sensitivity Check: gemma3-4b vs medgemma-1.5-4b-it

**Models Evaluated:** `gemma3-4b` (Generic Baseline) vs `medgemma-1.5-4b-it` (Medical Adaptation)
**Dataset & Split:** MedQA validation split (first 200 questions in seeded order, seed 42)
**Condition & Decoding:** Condition C1 (closed-book), greedy decoding (`do_sample=False`, `max_new_tokens=16`)
**Pre-Registered Protocol:** `docs/GEMMA_PROMPT_SENSITIVITY_RULE.md`

## 1. Pre-Registered Stability Evaluation

> [!IMPORTANT]
> **Pre-Registered Verdict: STABLE: Gap sign is uniform across variants (negative (favoring medgemma-1.5-4b-it)) and gap spread (5.50 pp) is <= 10.0 pp.**

- **Criterion 1 (Sign Uniformity):** PASS (All variants share the same sign).
- **Criterion 2 (Spread Bound):** Observed spread across variants is **5.50 percentage points** (Pre-registered threshold: $\le 10.0$ pp).
- **Format-Confound Status (Section 4.1):** None (all variants <= 5.0 pp parse-rate difference)

## 2. Model x Variant Performance Summary

| Variant | Template Description | Model | Strict Acc (95% CI) | Truncation Rate (95% CI) | Parse Rate (95% CI) |
|---|---|---|---|---|---|
| V1 | V1 (Standard Baseline) | `gemma3-4b` | 51.00% [44.1%, 57.8%] | 0.00% [0.0%, 1.9%] | 100.00% [98.1%, 100.0%] |
| V1 | V1 (Standard Baseline) | `medgemma-1.5-4b-it` | 61.00% [54.1%, 67.5%] | 0.00% [0.0%, 1.9%] | 100.00% [98.1%, 100.0%] |
| V2 | V2 (Clinical Scenario) | `gemma3-4b` | 50.00% [43.1%, 56.9%] | 0.00% [0.0%, 1.9%] | 100.00% [98.1%, 100.0%] |
| V2 | V2 (Clinical Scenario) | `medgemma-1.5-4b-it` | 65.50% [58.7%, 71.7%] | 0.00% [0.0%, 1.9%] | 100.00% [98.1%, 100.0%] |
| V3 | V3 (Board Examination) | `gemma3-4b` | 51.00% [44.1%, 57.8%] | 0.00% [0.0%, 1.9%] | 100.00% [98.1%, 100.0%] |
| V3 | V3 (Board Examination) | `medgemma-1.5-4b-it` | 63.50% [56.6%, 69.9%] | 0.50% [0.1%, 2.8%] | 99.50% [97.2%, 99.9%] |

## 3. Paired Statistical Comparisons (Gemma3 - MedGemma)

Discordant counts: $b$ = Gemma3 correct & MedGemma incorrect; $c$ = Gemma3 incorrect & MedGemma correct.
Differences are evaluated via exact two-sided McNemar tests and paired percentile bootstrap 95% CIs (10,000 resamples, seed 42).
Familywise error rate across the three variants is controlled using a single Holm-Bonferroni family ($m=3$).

| Variant | Gemma3 Acc | MedGemma Acc | Gap ($\Delta$ pp) | 95% Bootstrap CI | Discordant ($b/c$) | Raw McNemar $p$ | Holm-Adjusted $p$ |
|---|---|---|---|---|---|---|---|
| V1 | 51.00% | 61.00% | -10.00 pp | [-18.00, -2.00] | 25 / 45 | 0.0225 | 0.0225 |
| V2 | 50.00% | 65.50% | -15.50 pp | [-23.50, -7.00] | 22 / 53 | 0.0004 | 0.0013 |
| V3 | 51.00% | 63.50% | -12.50 pp | [-20.50, -4.50] | 22 / 47 | 0.0035 | 0.0071 |

### 3.1 Format-Confound Check & Descriptive Sensitivity (Section 4.1)

A variant is flagged `FORMAT-CONFOUNDED` if the parse rates of the two models differ by more than 5.0 percentage points in that variant.
For flagged variants, the gap computed only on questions where both models produced a parsed answer is shown, labelled `DESCRIPTIVE, selected subset`.

No variants were flagged `FORMAT-CONFOUNDED`. Pairwise parse-rate differences:
- **V1:** 0.00 percentage points ($\le 5.0$ pp threshold).
- **V2:** 0.00 percentage points ($\le 5.0$ pp threshold).
- **V3:** 0.50 percentage points ($\le 5.0$ pp threshold).

## 4. Visual Analysis

![Gemma Prompt Sensitivity Gap](outputs/analysis/gemma_prompt_sensitivity.png)

## 5. Scope and Test-Set Integrity

- **Validation Only:** All analyses and inferences in this report are conducted exclusively on the MedQA validation split ($N=200$).
- **Test-Set Independence:** This check does not access or modify test data. The test-set results previously reported (e.g. MedGemma +6.20 pp in Condition C1) remain untouched as the final empirical benchmark.
- **Purpose:** This sensitivity check quantifies the degree of prompt brittleness in the closed-book comparison, testing whether the adaptation advantage is robust to prompt rephrasing.

## 6. Execution Provenance & Manifest

- **Git Commit:** `unknown`
- **Rule Document Hash (sha256):** `unknown`
- **Completed Flag:** `True`
- **Hours Elapsed:** `0.3668`

### Prompt Template Hashes (sha256)
- `V1`: `a932a135e38e26fae1fd28411b863dbc7ac7d57dbfdd392162f765c91391c60e`
- `V2`: `4fe3e08ee35794f9553435895f3b75b764521538e6ca92ce96b66b6265449cb1`
- `V3`: `8e95470b7ae998d737cc4aad563b08c8d382706c96b7007d7f06af8b27c58b07`
