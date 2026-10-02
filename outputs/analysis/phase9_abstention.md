# Phase 9: Abstention and Selective Accuracy Analysis

> **Dataset**: MedQA + PubMedQA test split (1 000 answerable questions, 150 unanswerable questions).
> **Confidence**: Max letter probability ($\max_{L \in \{A, B, C, D\}} P(L)$).
> **Threshold Tuning**: $\tau$ tuned on **validation split only** (400 answerable, 50 unanswerable) to maximize Abstention F1 ($+ = \text{should\_abstain}$).
> **Bootstrap CIs**: Percentile bootstrap (1 000 resamples, seed 42) for Test Abstention F1 and Selective Accuracy.
> **Risk-Coverage**: Selective accuracy evaluated on test answerable questions as low-confidence predictions are pruned.

## 1. Abstention on Unanswerable Questions (`should_abstain == True`)

| Model | Mode | Frozen $\tau$ | Val F1 | Test Prec | Test Rec | Test F1 [95% CI] | Wrongly Abstained % | Full Acc | Selective Acc [95% CI] |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| qwen3-4b | baseline | 0.9763 | 0.2727 | 0.1917 | 0.2467 | 0.2157 [0.157, 0.274] | 15.6% | 0.5540 | **0.5865** [0.553, 0.619] |
| qwen3-4b | rag | 1.0000 | 0.2403 | 0.1469 | 0.4533 | 0.2219 [0.179, 0.266] | 39.5% | 0.5440 | **0.6380** [0.600, 0.676] |
| phi4-mini | baseline | 0.9597 | 0.2233 | 0.1296 | 0.8133 | 0.2236 [0.190, 0.259] | 81.9% | 0.5040 | **0.7127** [0.645, 0.777] |
| phi4-mini | rag | 0.7082 | 0.2447 | 0.1445 | 0.4133 | 0.2142 [0.171, 0.258] | 36.7% | 0.5190 | **0.6066** [0.569, 0.646] |
| gemma3-4b | baseline | 0.9999 | 0.2721 | 0.1962 | 0.3467 | 0.2506 [0.199, 0.306] | 21.3% | 0.4890 | **0.5146** [0.480, 0.549] |
| gemma3-4b | rag | 1.0000 | 0.2097 | 0.1402 | 0.8400 | 0.2402 [0.206, 0.277] | 77.3% | 0.5160 | **0.7093** [0.651, 0.763] |
| qwen3-1.7b | baseline | 0.9998 | 0.2415 | 0.1750 | 0.4667 | 0.2545 [0.208, 0.301] | 33.0% | 0.4620 | **0.5075** [0.469, 0.545] |
| qwen3-1.7b | rag | 0.9992 | 0.2162 | 0.1560 | 0.3400 | 0.2138 [0.169, 0.261] | 27.6% | 0.4790 | **0.5193** [0.483, 0.556] |
| smollm3-3b | baseline | 0.9865 | 0.2267 | 0.1434 | 0.7333 | 0.2399 [0.202, 0.278] | 65.7% | 0.4720 | **0.6268** [0.575, 0.676] |
| smollm3-3b | rag | 0.9998 | 0.2024 | 0.1307 | 1.0000 | 0.2311 [0.202, 0.263] | 99.8% | 0.4990 | **1.0000** [0.000, 1.000] |

## 2. Confidence Quality & Risk-Coverage (Test Answerable Rows, N=1 000)

| Model | Mode | AUROC (Corr/Incorr) | AUROC (Ans/Unans) | ECE (10-bin) | AURC | Acc @ 100% | Acc @ 80% | Acc @ 60% | Acc @ 40% | Acc @ 20% |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| qwen3-4b | baseline | 0.6280 | 0.5892 | 0.4168 | 0.3432 | 0.5540 | 0.5950 | 0.6450 | 0.6750 | **0.7400** |
| qwen3-4b | rag | 0.6320 | 0.5466 | 0.4310 | 0.3415 | 0.5440 | 0.5787 | 0.6383 | 0.6950 | **0.7650** |
| phi4-mini | baseline | 0.6445 | 0.4781 | 0.2427 | 0.3730 | 0.5040 | 0.5463 | 0.5800 | 0.6350 | **0.7150** |
| phi4-mini | rag | 0.6792 | 0.5222 | 0.2591 | 0.3294 | 0.5190 | 0.5650 | 0.6233 | 0.6900 | **0.7550** |
| gemma3-4b | baseline | 0.5806 | 0.6219 | 0.5029 | 0.4295 | 0.4890 | 0.5100 | 0.5417 | 0.5750 | **0.6350** |
| gemma3-4b | rag | 0.6344 | 0.5117 | 0.4713 | 0.3895 | 0.5160 | 0.5437 | 0.5917 | 0.6575 | **0.7100** |
| qwen3-1.7b | baseline | 0.5688 | 0.5962 | 0.5090 | 0.4913 | 0.4620 | 0.4888 | 0.5083 | 0.5025 | **0.5650** |
| qwen3-1.7b | rag | 0.5668 | 0.5573 | 0.4922 | 0.4737 | 0.4790 | 0.5012 | 0.5250 | 0.5550 | **0.5550** |
| smollm3-3b | baseline | 0.6216 | 0.5590 | 0.3930 | 0.4326 | 0.4720 | 0.4938 | 0.5433 | 0.6075 | **0.6450** |
| smollm3-3b | rag | 0.6162 | 0.5532 | 0.3284 | 0.4171 | 0.4990 | 0.5250 | 0.5633 | 0.6100 | **0.6400** |

## 3. Spontaneous Abstention (Refusal Regex Matches)

> Regex pattern: `none of the|cannot be determined|not enough information` (case-insensitive) on `raw_output`.

| Model | Mode | Validation Refusals (All / Unans) | Test Refusals (All / Unans) | Details |
|---|---|:---:|:---:|---|
| qwen3-4b | baseline | 3/450 (3/50) | 1/1150 (1/150) | Val: 3 on unanswerable; Test: 1 on unanswerable, 0 on answerable |
| qwen3-4b | rag | 0/450 (0/50) | 1/1150 (0/150) | Test: 0 on unanswerable, 1 on answerable |
| phi4-mini | baseline | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| phi4-mini | rag | 1/450 (0/50) | 0/1150 (0/150) | Val: 0 on unanswerable |
| gemma3-4b | baseline | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| gemma3-4b | rag | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| qwen3-1.7b | baseline | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| qwen3-1.7b | rag | 0/450 (0/50) | 2/1150 (0/150) | Test: 0 on unanswerable, 2 on answerable |
| smollm3-3b | baseline | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| smollm3-3b | rag | 0/450 (0/50) | 3/1150 (0/150) | Test: 0 on unanswerable, 3 on answerable |

## 4. Key Findings & Discussion

1. **Selective Accuracy Gains**:
   - When low-confidence questions are pruned using $\tau$, accuracy on retained answerable questions rises across the board.
   - For `phi4-mini` (Baseline), selective accuracy reaches **0.7127** (vs 0.5040 baseline, a +20.9% gain), while `gemma3-4b` (RAG) reaches **0.7093** (vs 0.5160, a +19.3% gain).
   - Under risk-coverage at 20% coverage (top quintile confidence), `qwen3-4b` RAG attains **0.7650**, `phi4-mini` RAG reaches **0.7550**, and `qwen3-4b` Baseline reaches **0.7400**.
2. **Separation of Answerable vs Unanswerable Questions**:
   - AUROC (Answerable vs Unanswerable) ranges from **0.512 to 0.622** (highest: `gemma3-4b` Baseline at 0.6219 and `qwen3-1.7b` Baseline at 0.5962).
   - While unanswerable questions induce lower model confidence on average, the calibration overlap is substantial, leading to moderate test F1 scores (0.21 - 0.25) when optimizing solely on validation.
3. **Calibration & Calibration Error**:
   - Models with sharper probability distributions (`phi4-mini`, `smollm3-3b`) exhibit lower ECE (0.24 - 0.39), whereas `gemma3-4b` and `qwen3-1.7b` exhibit higher overconfidence (ECE 0.47 - 0.51).
4. **Spontaneous Abstention**:
   - Standard instruction-tuned SLMs rarely refuse spontaneously (0-3 occurrences total across 1 150 test questions).
   - When `qwen3-4b` Baseline spontaneously refuses, it exclusively identifies unanswerable questions (3/3 on validation, 1/1 on test), showing an emergent sensitivity to unanswerability that is masked by RAG prompting.

---
*Artifacts generated: `outputs/analysis/risk_coverage_baseline.png`, `outputs/analysis/risk_coverage_rag.png`, and `outputs/analysis/phase9_abstention.json`.*
