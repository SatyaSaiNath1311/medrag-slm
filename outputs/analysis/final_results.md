# Final Results: Baseline vs Full RAG (5 Models)

> **Test set, answerable questions only** (should_abstain==False).
> 95% CI: percentile bootstrap (1 000 resamples, seed 42).
> McNemar: exact two-sided binomial test on discordant pairs.
> Sec/Q: mean seconds per question over test-split answerable rows only (same rows as accuracy).
> **Significant** = Holm-Bonferroni-adjusted p < 0.05 (not applicable here; no Holm correction is applied to the baseline-vs-RAG test — there is only one comparison per model).


## Accuracy
| Model | Baseline Overall [95% CI] | Baseline MedQA [95% CI] | Baseline PubMedQA [95% CI] | RAG Overall [95% CI] | RAG MedQA [95% CI] | RAG PubMedQA [95% CI] | McNemar p overall | McNemar p MedQA | McNemar p PubMedQA |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-4b | 0.5540 [0.523, 0.584] | 0.5840 [0.538, 0.626] | 0.5240 [0.478, 0.568] | 0.5440 [0.513, 0.574] | 0.6040 [0.560, 0.648] | 0.4840 [0.442, 0.528] | 0.58256 | 0.40749 | 0.12053 |
| phi4-mini | 0.5040 [0.471, 0.535] | 0.5340 [0.492, 0.578] | 0.4740 [0.432, 0.518] | 0.5190 [0.488, 0.550] | 0.5880 [0.542, 0.628] | 0.4500 [0.410, 0.496] | 0.37500 | 0.01240 | 0.35258 |
| gemma3-4b | 0.4890 [0.458, 0.519] | 0.4860 [0.442, 0.526] | 0.4920 [0.450, 0.538] | 0.5160 [0.486, 0.547] | 0.5300 [0.484, 0.574] | 0.5020 [0.460, 0.548] | 0.12994 | 0.08817 | 0.73814 |
| qwen3-1.7b | 0.4620 [0.432, 0.493] | 0.4460 [0.402, 0.488] | 0.4780 [0.434, 0.518] | 0.4790 [0.449, 0.510] | 0.5060 [0.464, 0.550] | 0.4520 [0.410, 0.496] | 0.31446 | 0.01327 | 0.26305 |
| smollm3-3b | 0.4720 [0.441, 0.503] | 0.4160 [0.374, 0.456] | 0.5280 [0.484, 0.574] | 0.4990 [0.469, 0.529] | 0.4720 [0.430, 0.514] | 0.5260 [0.486, 0.572] | 0.07595 | 0.01185 | 1.00000 |

## Parse Rate, Fallback Count, and Speed
| Model | Baseline Parse Rate | Baseline Fallback | Baseline Sec/Q | RAG Parse Rate | RAG Fallback | RAG Sec/Q |
|---|---|---|---|---|---|---|
| qwen3-4b | 0.9975 | 4 | 0.223 s | 0.9988 | 2 | 1.452 s |
| phi4-mini | 1.0000 | 0 | 0.162 s | 1.0000 | 0 | 1.129 s |
| gemma3-4b | 1.0000 | 0 | 0.611 s | 0.9994 | 1 | 3.321 s |
| qwen3-1.7b | 1.0000 | 0 | 0.117 s | 1.0000 | 0 | 0.696 s |
| smollm3-3b | 1.0000 | 0 | 0.162 s | 1.0000 | 0 | 0.939 s |

## Key Findings
'Significant' throughout means **Holm-Bonferroni-adjusted p < 0.05** (where HB correction is applied, i.e., in the adaptive-RAG analysis over 4 strategies).  
For the single baseline-vs-Full-RAG comparison (this table), no multiple-test correction is needed; a result is 'significant before correction' if raw p < 0.05.

**Significant before correction (raw p < 0.05, Full RAG vs Baseline):**
  - phi4-mini [MedQA]: p=0.01240 (RAG > Baseline, Baseline 0.5340 → RAG 0.5880)
  - qwen3-1.7b [MedQA]: p=0.01327 (RAG > Baseline, Baseline 0.4460 → RAG 0.5060)
  - smollm3-3b [MedQA]: p=0.01185 (RAG > Baseline, Baseline 0.4160 → RAG 0.4720)

## Pooled Cochran–Mantel–Haenszel (CMH) Analysis

Across all 5 models (stratified 2x2 meta-analysis, RAG vs Baseline conditional on model):

| Scope | Mantel–Haenszel Common OR [95% CI] | CMH $\chi^2$ (df=1) | p-value (raw) | p-value (continuity-corrected) | Significant? |
|---|---|---|---|---|---|
| **Overall** | 1.0629 [0.9826, 1.1497] | 2.3160 | 1.28053e-01 | 1.33148e-01 | No (p = 0.128) |
| **MedQA** | 1.2090 [1.0812, 1.3519] | 11.0851 | 8.70246e-04 | 9.63507e-04 | **Yes (p < 0.001, RAG > Baseline)** |
| **Pubmedqa** | 0.9364 [0.8380, 1.0463] | 1.3471 | 2.45778e-01 | 2.57485e-01 | No (p = 0.246) |

---
*Retrieval+rerank cost not included in Sec/Q above: 0.12730 s/q (Phase6 23.424 s + Phase7 180.261 s = 203.685 s / 1600 q, from outputs/kaggle_build/medrag-build.log)*