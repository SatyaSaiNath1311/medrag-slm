# MedPsy-4B Abstention and Thinking Budget Analysis

**Context & Protocol**:
- **Pre-specified budget**: The generation cap of **1,024 tokens** was fixed prior to test-split evaluation.
- **Evaluation Pool**: Held-out test split of $N=650$ questions (500 answerable MedQA + 150 unanswerable questions).
- **Target Task**: Binary abstention detection, where unanswerable questions ($N=150$) require abstention and MedQA questions ($N=500$) require an answer.
- **Statistical Rigor**: All confidence intervals are 95% percentile bootstrap CIs (1,000 resamples, seed 42).

## 1. Abstention Rule Performance Comparison

Comparison of the pre-specified truncation rule against confidence thresholding and the always-abstain baseline:

| Rule | Precision [95% CI] | Recall [95% CI] | F1 Score [95% CI] | Wrongly Refused % [95% CI] | Coverage % [95% CI] | Acc when Answering [95% CI] |
|---|---|---|---|---|---|---|
| **Abstain if Truncated** *(Pre-specified)* | 0.569 [0.478, 0.655] | 0.467 [0.384, 0.544] | **0.513** [0.434, 0.581] | 10.6% [7.9%, 13.2%] | 89.4% [86.8%, 92.1%] | 93.1% [90.6%, 95.4%] |
| **Truncated OR Conf < 0.90** | 0.568 [0.477, 0.653] | 0.473 [0.389, 0.553] | **0.516** [0.437, 0.585] | 10.8% [8.1%, 13.4%] | 89.2% [86.6%, 91.9%] | 93.3% [90.9%, 95.5%] |
| **Conf < 0.90 alone** | 0.656 [0.480, 0.818] | 0.140 [0.086, 0.201] | **0.231** [0.148, 0.315] | 2.2% [1.0%, 3.6%] | 97.8% [96.4%, 99.0%] | 89.2% [86.5%, 91.9%] |
| **Always Abstain** *(Baseline)* | 0.231 [0.199, 0.260] | 1.000 [1.000, 1.000] | **0.375** [0.331, 0.413] | 100.0% [100.0%, 100.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

> **Key Takeaway**: Truncation serves as a natural, calibration-free uncertainty signal. When MedPsy exhausts its 1,024-token thinking budget, it signals insolubility or extreme deliberation, achieving an F1 score of **0.513** compared to the naive always-abstain F1 of **0.375** (+13.8 pts). Answering only when thinking finishes raises MedQA test accuracy from **87.6%** to **93.1%** at **89.4%** coverage.

## 2. Generated-Token Distributions

Analysis of tokens generated inside `<think>...</think>` and answer blocks (cap = 1,024 tokens):

| Subgroup | $N$ | Mean Tokens (SD) | Median Tokens | % reaching 1,024 tokens ($N$) | Interpretation |
|---|---|---|---|---|---|
| **Answerable (MedQA Test)** | 500 | 562.9 (±251.9) | 495.0 | **11.6%** (58/500) | Answerable medical clinical vignettes |
| **Unanswerable Test Pool** | 150 | 839.4 (±219.5) | 990.5 | **46.7%** (70/150) | Questions with removed critical findings / insoluble |
| **MedQA: Correct Predictions** | 438 | 517.2 (±223.4) | 464.5 | **5.7%** (25/438) | Vignettes solved correctly by model |
| **MedQA: Incorrect Predictions** | 62 | 885.6 (±201.3) | 1024.0 | **53.2%** (33/62) | Vignettes failed by model |

> **Note**: Truncated (no final answer within the cap) = 53/500 MedQA, 70/150 unanswerable; 5 MedQA answers reached the cap but still gave a final answer.


> **Findings**:  
> 1. **Unanswerable questions trigger budget exhaustion at 4.4× the rate of answerable ones** (46.7% vs 11.6%).  
> 2. **Wrong answers deliberated longer and hit the cap 5× more frequently than correct answers** (53.2% vs 5.7%).  
> 3. Accuracy on MedQA questions that finished thinking was **93.1%** (416 / 447), whereas accuracy on questions truncated at 1,024 tokens was only **41.5%** (22 / 53).

## 3. Protocol Note

- The 1,024-token cap was set in the configuration before the pilot and before any test evaluation; the truncation rule has no tuned parameters.
