# MedPsy-4B calibrated reliability score and abstention threshold

**Protocol.** Logistic-regression weights, feature scaling and every threshold were fitted on the validation split only; the frozen rule was applied once to the held-out test split. Ridge penalty (1.0), safe targets (MedQA 95%, PubMedQA 90% accuracy when answering) and the 50% minimum coverage were fixed in advance. CIs are 95% percentile bootstrap (1,000 resamples, seed 42). *Unsafe answers* = questions the system answered that were wrong or unanswerable, over all questions.

## MedQA + unanswerable (no retrieval in prompt)

Fitted on validation (n=250), evaluated once on test (n=650). Features: truncated, token_frac, confidence, margin, entropy, top_rerank.

| Rule | Abstain F1 [95% CI] | Unanswerable recall [CI] | Coverage [CI] | Acc when answering [CI] | Unsafe answers [CI] |
|---|---|---|---|---|---|
| Always answer | 0.000 [0.000, 0.000] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 87.6% [84.6, 90.3] | 32.6% [29.1, 36.3] |
| Abstain if truncated (pre-specified) | 0.513 [0.440, 0.578] | 46.7% [38.5, 54.3] | 89.4% [86.9, 92.1] | 93.1% [90.4, 95.3] | 17.1% [14.2, 20.3] |
| Abstain if tokens >= t (t from val) | 0.539 [0.479, 0.591] | 66.7% [58.8, 74.0] | 75.8% [72.3, 79.6] | 96.6% [94.6, 98.2] | 9.7% [7.5, 12.2] |
| **Reliability score, safe τ (val target acc, coverage ≥ 50%)** | 0.537 [0.478, 0.589] | 70.0% [62.6, 76.7] | 72.8% [69.0, 76.6] | 97.5% [95.7, 99.1] | 8.3% [6.3, 10.6] |
| **Reliability score, max-F1 τ (from val)** | 0.555 [0.495, 0.609] | 69.3% [61.7, 76.2] | 75.8% [72.1, 79.5] | 96.6% [94.6, 98.2] | 9.1% [7.1, 11.4] |

Safe target for this task: 95% accuracy when answering (validation).

Paired bootstrap, Reliability score, max-F1 τ (from val) minus truncation rule: abstention F1 +0.042 [95% CI -0.014, +0.104].

**Accuracy vs coverage (reliability score; threshold set on validation for each target coverage)**

| Target coverage (val) | Test coverage | Test accuracy when answering [95% CI] | Unsafe answers |
|---|---|---|---|
| 100% | 100.0% | 87.6% [84.6, 90.3] | 32.6% |
| 90% | 81.4% | 95.1% [92.8, 97.0] | 13.1% |
| 80% | 72.2% | 97.5% [95.7, 99.1] | 8.2% |
| 70% | 64.0% | 97.5% [95.7, 99.1] | 5.8% |
| 60% | 57.2% | 97.9% [96.1, 99.3] | 4.0% |

**Signal quality (threshold-free, test; target = unsafe to answer)**

| Signal | AUROC [95% CI] | AURC (lower is better) |
|---|---|---|
| raw confidence | 0.655 [0.603, 0.705] | 0.282 |
| generated tokens | 0.849 [0.816, 0.877] | 0.120 |
| reliability score | 0.859 [0.828, 0.886] | 0.118 |

**Calibration of P(answer is safe), test**

| Signal | ECE | Brier |
|---|---|---|
| raw confidence | 0.309 | 0.300 |
| reliability score | 0.050 | 0.140 |

Thresholds (validation): tokens=0.7393, score_f1=0.3204, score_safe=0.2942

## PubMedQA + abstract

Fitted on validation (n=200), evaluated once on test (n=500). Features: truncated, token_frac, confidence, margin, entropy.

| Rule | Coverage [CI] | Acc when answering [CI] | Unsafe answers [CI] |
|---|---|---|---|
| Always answer | 100.0% [100.0, 100.0] | 78.0% [74.6, 81.6] | 22.0% [18.4, 25.4] |
| Abstain if truncated (pre-specified) | 99.8% [99.4, 100.0] | 78.2% [74.6, 81.8] | 21.8% [18.2, 25.4] |
| Abstain if tokens >= t (t from val) | 53.4% [49.4, 57.6] | 86.1% [82.0, 90.2] | 7.4% [5.2, 9.8] |
| **Reliability score, safe τ (val target acc, coverage ≥ 50%)** | 52.8% [48.4, 57.0] | 86.0% [81.9, 90.2] | 7.4% [5.2, 9.6] |

Safe target for this task: 90% accuracy when answering (validation).

Paired bootstrap, Reliability score, safe τ (val target acc, coverage ≥ 50%) minus truncation rule: unsafe-answer rate -14.4 pts [95% CI -17.4, -11.2].

**Accuracy vs coverage (reliability score; threshold set on validation for each target coverage)**

| Target coverage (val) | Test coverage | Test accuracy when answering [95% CI] | Unsafe answers |
|---|---|---|---|
| 100% | 100.0% | 78.0% [74.6, 81.6] | 22.0% |
| 90% | 91.2% | 79.8% [76.2, 83.6] | 18.4% |
| 80% | 79.2% | 82.1% [78.3, 85.7] | 14.2% |
| 70% | 73.0% | 84.1% [80.3, 87.9] | 11.6% |
| 60% | 63.8% | 86.5% [82.7, 90.2] | 8.6% |

**Signal quality (threshold-free, test; target = unsafe to answer)**

| Signal | AUROC [95% CI] | AURC (lower is better) |
|---|---|---|
| raw confidence | 0.610 [0.554, 0.668] | 0.157 |
| generated tokens | 0.671 [0.609, 0.727] | 0.154 |
| reliability score | 0.659 [0.597, 0.716] | 0.164 |

**Calibration of P(answer is safe), test**

| Signal | ECE | Brier |
|---|---|---|
| raw confidence | 0.220 | 0.220 |
| reliability score | 0.048 | 0.165 |

Thresholds (validation): tokens=0.4395, score_safe=0.2293
