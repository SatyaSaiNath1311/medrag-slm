# Final Results: Baseline vs Full RAG Benchmark

> **Test set, answerable questions only** (should_abstain==False).
> 95% CI: percentile bootstrap (1 000 resamples, seed 42).
> McNemar: exact two-sided binomial test on discordant pairs.
> Sec/Q: mean seconds per question over test-split answerable rows only.
> **Note on medpsy-4b**: Evaluated on MedQA test baseline ($N=500$) and PubMedQA test +Abstract ($N=500$). Missing modes/splits are displayed as `n/a`.


## Accuracy
| Model | Baseline Overall [95% CI] | Baseline MedQA [95% CI] | Baseline PubMedQA [95% CI] | RAG Overall [95% CI] | RAG MedQA [95% CI] | RAG PubMedQA [95% CI] | McNemar p overall | McNemar p MedQA | McNemar p PubMedQA |
|---|---|---|---|---|---|---|---|---|---|
| medpsy-4b | n/a | 0.8760 [0.846, 0.904] | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| qwen3-4b | 0.5540 [0.523, 0.584] | 0.5840 [0.538, 0.626] | 0.5240 [0.478, 0.568] | 0.5440 [0.513, 0.574] | 0.6040 [0.560, 0.648] | 0.4840 [0.442, 0.528] | 0.58256 | 0.40749 | 0.12053 |
| phi4-mini | 0.5040 [0.471, 0.535] | 0.5340 [0.492, 0.578] | 0.4740 [0.432, 0.518] | 0.5190 [0.488, 0.550] | 0.5880 [0.542, 0.628] | 0.4500 [0.410, 0.496] | 0.37500 | 0.01240 | 0.35258 |
| gemma3-4b | 0.4890 [0.458, 0.519] | 0.4860 [0.442, 0.526] | 0.4920 [0.450, 0.538] | 0.5160 [0.486, 0.547] | 0.5300 [0.484, 0.574] | 0.5020 [0.460, 0.548] | 0.12994 | 0.08817 | 0.73814 |
| qwen3-1.7b | 0.4620 [0.432, 0.493] | 0.4460 [0.402, 0.488] | 0.4780 [0.434, 0.518] | 0.4790 [0.449, 0.510] | 0.5060 [0.464, 0.550] | 0.4520 [0.410, 0.496] | 0.31446 | 0.01327 | 0.26305 |
| smollm3-3b | 0.4720 [0.441, 0.503] | 0.4160 [0.374, 0.456] | 0.5280 [0.484, 0.574] | 0.4990 [0.469, 0.529] | 0.4720 [0.430, 0.514] | 0.5260 [0.486, 0.572] | 0.07595 | 0.01185 | 1.00000 |

## Parse Rate, Truncation, and Speed
| Model | Baseline Parse Rate | Baseline Truncated | Baseline Fallback | Baseline Sec/Q | RAG Parse Rate | RAG Truncated | RAG Fallback | RAG Sec/Q |
|---|---|---|---|---|---|---|---|---|
| medpsy-4b | 0.8231 | 18.9% | 0 | 37.303 s | n/a | 0.0% | n/a | n/a |
| qwen3-4b | 0.9975 | 0.0% | 4 | 0.223 s | 0.9988 | 0.0% | 2 | 1.452 s |
| phi4-mini | 1.0000 | 0.0% | 0 | 0.162 s | 1.0000 | 0.0% | 0 | 1.129 s |
| gemma3-4b | 1.0000 | 0.0% | 0 | 0.611 s | 0.9994 | 0.0% | 1 | 3.321 s |
| qwen3-1.7b | 1.0000 | 0.0% | 0 | 0.117 s | 1.0000 | 0.0% | 0 | 0.696 s |
| smollm3-3b | 1.0000 | 0.0% | 0 | 0.162 s | 1.0000 | 0.0% | 0 | 0.939 s |

## PubMedQA +Abstract (Oracle Context Benchmark)
| Model | PubMedQA +Abstract Acc [95% CI] | Parse Rate | Truncated | Sec/Q |
|---|---|---|---|---|
| medpsy-4b | 0.7800 [0.746, 0.814] | 0.9980 | 0.2% | 30.279 s |
| qwen3-4b | 0.7460 [0.710, 0.784] | 1.0000 | 0.0% | 0.406 s |
| phi4-mini | 0.7360 [0.698, 0.776] | 1.0000 | 0.0% | 0.485 s |
| gemma3-4b | 0.6880 [0.650, 0.730] | 1.0000 | 0.0% | 1.422 s |
| qwen3-1.7b | 0.6620 [0.618, 0.704] | 1.0000 | 0.0% | 0.333 s |
| smollm3-3b | 0.7240 [0.686, 0.764] | 1.0000 | 0.0% | 0.464 s |

## MedPsy-4B vs. Standard SLM Baselines (MedQA Test Split, $N=500$)
> Pairwise McNemar exact two-sided test on discordant pairs for MedQA test ($N=500$).  
> - **MedPsy-Only Wins ($b$)**: Questions answered correctly by MedPsy-4B but wrong by baseline.  
> - **Baseline-Only Wins ($c$)**: Questions answered correctly by baseline but wrong by MedPsy-4B.

| Comparison | MedPsy-4B Acc | Baseline Acc | MedPsy-Only ($b$) | Baseline-Only ($c$) | McNemar $p$-value | Significant? |
|---|---|---|---|---|---|---|
| MedPsy-4B vs. **qwen3-4b** | 87.6% | 58.4% | 176 | 30 | 2.49e-26 | **Yes (p < 0.001)** |
| MedPsy-4B vs. **phi4-mini** | 87.6% | 53.4% | 189 | 18 | 3.81e-37 | **Yes (p < 0.001)** |
| MedPsy-4B vs. **gemma3-4b** | 87.6% | 48.6% | 217 | 22 | 1.75e-41 | **Yes (p < 0.001)** |
| MedPsy-4B vs. **qwen3-1.7b** | 87.6% | 44.6% | 235 | 20 | 9.75e-48 | **Yes (p < 0.001)** |
| MedPsy-4B vs. **smollm3-3b** | 87.6% | 41.6% | 250 | 20 | 9.70e-52 | **Yes (p < 0.001)** |

## Key Findings
For the baseline-vs-Full-RAG comparisons, no multiple-test correction is needed; a result is significant if raw $p < 0.05$.

**Significant before correction (raw p < 0.05, Full RAG vs Baseline):**
  - phi4-mini [MedQA]: p=0.01240 (RAG > Baseline, Baseline 0.5340 → RAG 0.5880)
  - qwen3-1.7b [MedQA]: p=0.01327 (RAG > Baseline, Baseline 0.4460 → RAG 0.5060)
  - smollm3-3b [MedQA]: p=0.01185 (RAG > Baseline, Baseline 0.4160 → RAG 0.4720)

## Pooled Cochran–Mantel–Haenszel (CMH) Analysis

Across all 5 standard SLMs with parametric vs. RAG runs (stratified 2x2 meta-analysis):

| Scope | Mantel–Haenszel Common OR [95% CI] | CMH $\chi^2$ (df=1) | p-value (raw) | p-value (continuity-corrected) | Significant? |
|---|---|---|---|---|---|
| **Overall** | 1.0629 [0.9826, 1.1497] | 2.3160 | 1.28053e-01 | 1.33148e-01 | No (p = 0.128) |
| **MedQA** | 1.2090 [1.0812, 1.3519] | 11.0851 | 8.70246e-04 | 9.63507e-04 | **Yes (p < 0.001, RAG > Baseline)** |
| **Pubmedqa** | 0.9364 [0.8380, 1.0463] | 1.3471 | 2.45778e-01 | 2.57485e-01 | No (p = 0.246) |

---
*Retrieval+rerank cost not included in Sec/Q above: 0.12730 s/q (Phase6 23.424 s + Phase7 180.261 s = 203.685 s / 1600 q, from outputs/kaggle_build/medrag-build.log)*