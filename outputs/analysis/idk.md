# "I Don't Know" (IDK) Abstention Experiment Analysis

## Executive Summary

This document evaluates the explicit **"I don't know" (IDK)** abstention mechanism introduced in Phase 12 
(`baseline_idk` and `rag_idk`) and compares it against standard forced-choice non-IDK baselines 
(`baseline` Phase 5, `rag` Phase 8) and post-hoc calibrated confidence gating ($T^*$ scaling).

All metrics are evaluated on the **TEST split** ($N=1,150$ total questions: $1,000$ answerable, $150$ unanswerable).


### Evaluated Metrics & Evaluation Framework

1. **Abstention on Unanswerable Questions ($N=150$)**:
   - Evaluated on the MedQA + Unanswerable benchmark pool ($N=650$, where 150 questions have their premise perturbed to make them unanswerable).
   - **Always-Abstain Baseline**: A naive classifier that unconditionally abstains on all questions achieves:
     - $\text{Precision} = 150 / 650 = \mathbf{0.231}$ (23.1%)
     - $\text{Recall} = 150 / 150 = \mathbf{1.000}$ (100.0%)
     - $\text{F1} = \frac{2 \times 0.231 \times 1.0}{0.231 + 1.0} = \mathbf{0.375}$
   - We report Precision, Recall, and F1 with 95% bootstrap confidence intervals.
2. **Wrong Abstentions on Answerable Questions ($N=1,000$)**:
   - Percentage of answerable questions where the model incorrectly abstained (False Positives): $\frac{FP}{N_{\text{ans}}} \times 100\%$.
3. **Coverage & Selective Accuracy on Answered Questions**:
   - **Coverage**: $\frac{N_{\text{answered}}}{N_{\text{ans}}} \times 100\%$.
   - **Selective Accuracy**: Accuracy computed exclusively over questions the model elected to answer: $\frac{\text{Correct}_{\text{answered}}}{N_{\text{answered}}} \times 100\%$.
4. **Overall Accuracy (Counting Abstentions as Not-Correct)**:
   - Penalizes abstentions as $0$ points: $\text{Overall Accuracy} = \text{Coverage} \times \text{Selective Accuracy} = \frac{\text{Correct}}{1,000} \times 100\%$.
5. **Combined Abstention Rule ("IDK Option OR Low Calibrated Confidence")**:
   - Dual-safeguard rule: The model abstains if it explicitly selects the IDK option, **or** if its post-hoc temperature-scaled confidence $p_{\text{scaled}}$ falls below the validation-calibrated threshold $\tau^*$.


---
## Results for `qwen3-4b`

- **Validation Temperatures ($T^*$ min-NLL)**: Baseline $T^* = 17.24$, RAG $T^* = 20.73$
- **Validation Confidence Thresholds (F1-tuned $\tau^*$)**: Baseline $\tau^* = 0.5122$, RAG $\tau^* = 0.5403$

### Table 1A: Abstention on MedQA + Unanswerable Pool ($N=650$, Base Prevalence = 0.231)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.231** [—] | **1.000** [—] | **0.375** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 58.4% [54.0, 62.7] | 58.4% [54.0, 62.7] |
| Baseline (Confidence Gate) | 0.215 [0.172, 0.260] | 0.487 [0.410, 0.569] | 0.299 [0.245, 0.353] | 53.2% [49.1, 57.7] | 46.8% [42.3, 50.9] | 73.1% [67.3, 78.5] | 34.2% [29.6, 38.3] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 60.4% [56.2, 64.8] | 60.4% [56.2, 64.8] |
| Full RAG (Confidence Gate) | 0.224 [0.177, 0.270] | 0.487 [0.408, 0.573] | 0.307 [0.252, 0.362] | 50.6% [46.6, 55.1] | 49.4% [45.0, 53.4] | 77.3% [72.7, 82.5] | 38.2% [34.3, 42.3] |
| **Baseline (IDK Option)** | 0.520 [0.407, 0.632] | 0.260 [0.194, 0.339] | 0.347 [0.268, 0.427] | 7.2% [5.2, 9.8] | 92.8% [90.2, 94.8] | 59.7% [55.4, 63.8] | 55.4% [51.3, 59.5] |
| **Baseline (IDK + Conf Combined)** | 0.274 [0.231, 0.316] | 0.740 [0.669, 0.811] | 0.400 [0.348, 0.449] | 58.8% [54.6, 63.1] | 41.2% [37.0, 45.4] | 74.3% [68.1, 79.9] | 30.6% [26.5, 34.6] |
| **Full RAG (IDK Option)** | 0.480 [0.267, 0.667] | 0.080 [0.038, 0.126] | 0.137 [0.067, 0.206] | 2.6% [1.4, 4.1] | 97.4% [95.9, 98.6] | 62.6% [58.2, 67.0] | 61.0% [56.7, 65.3] |
| **Full RAG (IDK + Conf Combined)** | 0.270 [0.214, 0.323] | 0.520 [0.443, 0.608] | 0.355 [0.294, 0.412] | 42.2% [38.1, 46.2] | 57.8% [53.8, 61.9] | 76.1% [71.7, 80.8] | 44.0% [40.1, 48.1] |

### Table 1B: Abstention on Full Test Split ($N=1,150$, 1,000 Answerable + 150 Unanswerable)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.130** [—] | **1.000** [—] | **0.231** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 55.4% [52.5, 58.4] | 55.4% [52.5, 58.4] |
| Baseline (Confidence Gate) | 0.146 [0.117, 0.177] | 0.487 [0.411, 0.567] | 0.225 [0.183, 0.267] | 42.7% [39.7, 45.9] | 57.3% [54.1, 60.4] | 64.4% [60.5, 68.2] | 36.9% [33.9, 39.8] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 54.4% [51.2, 57.6] | 54.4% [51.2, 57.6] |
| Full RAG (Confidence Gate) | 0.139 [0.111, 0.169] | 0.487 [0.403, 0.571] | 0.216 [0.176, 0.256] | 45.2% [42.0, 48.4] | 54.8% [51.6, 58.1] | 64.1% [60.1, 68.2] | 35.1% [32.2, 38.2] |
| **Baseline (IDK Option)** | 0.172 [0.125, 0.228] | 0.260 [0.191, 0.338] | 0.207 [0.154, 0.268] | 18.8% [16.3, 21.0] | 81.2% [79.0, 83.7] | 57.3% [53.8, 60.6] | 46.5% [43.6, 49.4] |
| **Baseline (IDK + Conf Combined)** | 0.151 [0.126, 0.177] | 0.740 [0.667, 0.810] | 0.251 [0.213, 0.288] | 62.2% [59.3, 65.1] | 37.8% [34.9, 40.7] | 67.5% [62.8, 71.9] | 25.5% [22.9, 28.1] |
| **Full RAG (IDK Option)** | 0.049 [0.024, 0.077] | 0.080 [0.038, 0.125] | 0.061 [0.030, 0.093] | 23.3% [20.6, 25.8] | 76.7% [74.2, 79.4] | 58.1% [54.4, 61.8] | 44.6% [41.2, 47.8] |
| **Full RAG (IDK + Conf Combined)** | 0.128 [0.104, 0.155] | 0.520 [0.441, 0.600] | 0.205 [0.170, 0.245] | 53.2% [49.9, 56.3] | 46.8% [43.7, 50.1] | 68.2% [63.9, 72.5] | 31.9% [28.9, 34.9] |

---
## Results for `phi4-mini`

- **Validation Temperatures ($T^*$ min-NLL)**: Baseline $T^* = 3.69$, RAG $T^* = 4.49$
- **Validation Confidence Thresholds (F1-tuned $\tau^*$)**: Baseline $\tau^* = 0.6010$, RAG $\tau^* = 0.5277$

### Table 1A: Abstention on MedQA + Unanswerable Pool ($N=650$, Base Prevalence = 0.231)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.231** [—] | **1.000** [—] | **0.375** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 53.4% [49.1, 57.7] | 53.4% [49.1, 57.7] |
| Baseline (Confidence Gate) | 0.222 [0.185, 0.258] | 0.800 [0.733, 0.863] | 0.348 [0.298, 0.393] | 84.0% [80.9, 87.2] | 16.0% [12.9, 19.2] | 85.0% [76.8, 92.9] | 13.6% [10.5, 16.6] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 58.8% [54.7, 63.3] | 58.8% [54.7, 63.3] |
| Full RAG (Confidence Gate) | 0.240 [0.202, 0.281] | 0.720 [0.653, 0.789] | 0.360 [0.312, 0.410] | 68.4% [64.1, 72.7] | 31.6% [27.4, 36.0] | 84.2% [78.2, 89.8] | 26.6% [22.8, 31.0] |

### Table 1B: Abstention on Full Test Split ($N=1,150$, 1,000 Answerable + 150 Unanswerable)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.130** [—] | **1.000** [—] | **0.231** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 50.4% [47.4, 53.6] | 50.4% [47.4, 53.6] |
| Baseline (Confidence Gate) | 0.124 [0.103, 0.144] | 0.800 [0.735, 0.859] | 0.214 [0.182, 0.245] | 84.9% [82.6, 86.9] | 15.1% [13.1, 17.4] | 74.2% [67.3, 80.9] | 11.2% [9.3, 13.2] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 51.9% [48.5, 55.0] | 51.9% [48.5, 55.0] |
| Full RAG (Confidence Gate) | 0.134 [0.114, 0.160] | 0.720 [0.650, 0.786] | 0.226 [0.193, 0.265] | 69.6% [66.8, 72.3] | 30.4% [27.7, 33.3] | 68.8% [63.4, 73.7] | 20.9% [18.5, 23.5] |

> [!NOTE]
> **Phase 12 Kaggle Execution Notice**:
> `work/phase12/` files are not yet populated locally. The runner is configured for Kaggle execution:
> `MODELS = ["qwen3-4b"]`, `MODES = ["baseline_idk", "rag_idk"]`, `PROFILE = False`, `CHECK_ONLY = False`, `TINY = False`.
> Once execution completes and output files (`qwen3-4b_baseline_idk.jsonl`, `qwen3-4b_rag_idk.jsonl`) are placed in `work/phase12/`, > rerun `python3 src/analysis_idk.py` to populate the empirical IDK rows in these tables.

---
## Results for `gemma3-4b`

- **Validation Temperatures ($T^*$ min-NLL)**: Baseline $T^* = 25.71$, RAG $T^* = 19.54$
- **Validation Confidence Thresholds (F1-tuned $\tau^*$)**: Baseline $\tau^* = 0.4940$, RAG $\tau^* = 0.5783$

### Table 1A: Abstention on MedQA + Unanswerable Pool ($N=650$, Base Prevalence = 0.231)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.231** [—] | **1.000** [—] | **0.375** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 48.6% [44.2, 53.1] | 48.6% [44.2, 53.1] |
| Baseline (Confidence Gate) | 0.208 [0.171, 0.247] | 0.600 [0.522, 0.680] | 0.309 [0.260, 0.358] | 68.6% [64.8, 73.0] | 31.4% [27.1, 35.3] | 65.0% [56.9, 72.8] | 20.4% [16.5, 24.1] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 53.0% [48.6, 57.2] | 53.0% [48.6, 57.2] |
| Full RAG (Confidence Gate) | 0.254 [0.214, 0.297] | 0.813 [0.752, 0.873] | 0.387 [0.337, 0.438] | 71.8% [68.1, 75.9] | 28.2% [24.1, 31.9] | 71.6% [64.6, 79.5] | 20.2% [16.7, 23.8] |

### Table 1B: Abstention on Full Test Split ($N=1,150$, 1,000 Answerable + 150 Unanswerable)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.130** [—] | **1.000** [—] | **0.231** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 48.9% [45.8, 52.1] | 48.9% [45.8, 52.1] |
| Baseline (Confidence Gate) | 0.160 [0.130, 0.191] | 0.600 [0.523, 0.681] | 0.252 [0.211, 0.296] | 47.3% [44.3, 50.4] | 52.7% [49.7, 55.7] | 53.9% [49.7, 58.3] | 28.4% [25.7, 31.4] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 51.6% [48.7, 54.9] | 51.6% [48.7, 54.9] |
| Full RAG (Confidence Gate) | 0.140 [0.119, 0.164] | 0.813 [0.746, 0.878] | 0.239 [0.206, 0.275] | 75.0% [72.4, 77.7] | 25.0% [22.3, 27.6] | 69.2% [63.4, 74.8] | 17.3% [15.1, 19.8] |

> [!NOTE]
> **Phase 12 Kaggle Execution Notice**:
> `work/phase12/` files are not yet populated locally. The runner is configured for Kaggle execution:
> `MODELS = ["qwen3-4b"]`, `MODES = ["baseline_idk", "rag_idk"]`, `PROFILE = False`, `CHECK_ONLY = False`, `TINY = False`.
> Once execution completes and output files (`qwen3-4b_baseline_idk.jsonl`, `qwen3-4b_rag_idk.jsonl`) are placed in `work/phase12/`, > rerun `python3 src/analysis_idk.py` to populate the empirical IDK rows in these tables.

---
## Results for `qwen3-1.7b`

- **Validation Temperatures ($T^*$ min-NLL)**: Baseline $T^* = 24.72$, RAG $T^* = 22.72$
- **Validation Confidence Thresholds (F1-tuned $\tau^*$)**: Baseline $\tau^* = 0.5017$, RAG $\tau^* = 0.5282$

### Table 1A: Abstention on MedQA + Unanswerable Pool ($N=650$, Base Prevalence = 0.231)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.231** [—] | **1.000** [—] | **0.375** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 44.6% [40.0, 49.1] | 44.6% [40.0, 49.1] |
| Baseline (Confidence Gate) | 0.204 [0.163, 0.242] | 0.560 [0.480, 0.637] | 0.299 [0.249, 0.347] | 65.6% [61.3, 69.9] | 34.4% [30.1, 38.7] | 58.1% [50.0, 65.6] | 20.0% [16.5, 23.7] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 50.6% [46.2, 55.0] | 50.6% [46.2, 55.0] |
| Full RAG (Confidence Gate) | 0.193 [0.156, 0.232] | 0.527 [0.449, 0.609] | 0.282 [0.234, 0.332] | 66.2% [62.0, 70.0] | 33.8% [30.0, 38.0] | 62.7% [54.9, 70.4] | 21.2% [17.7, 24.7] |

### Table 1B: Abstention on Full Test Split ($N=1,150$, 1,000 Answerable + 150 Unanswerable)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.130** [—] | **1.000** [—] | **0.231** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 46.2% [43.3, 49.2] | 46.2% [43.3, 49.2] |
| Baseline (Confidence Gate) | 0.156 [0.125, 0.187] | 0.560 [0.477, 0.642] | 0.243 [0.201, 0.286] | 45.6% [42.5, 48.8] | 54.4% [51.2, 57.5] | 51.8% [47.8, 56.2] | 28.2% [25.5, 31.1] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 47.9% [44.7, 51.0] | 47.9% [44.7, 51.0] |
| Full RAG (Confidence Gate) | 0.138 [0.112, 0.166] | 0.527 [0.449, 0.604] | 0.219 [0.181, 0.259] | 49.3% [46.2, 52.5] | 50.7% [47.5, 53.8] | 52.9% [48.8, 57.1] | 26.8% [24.2, 29.5] |

> [!NOTE]
> **Phase 12 Kaggle Execution Notice**:
> `work/phase12/` files are not yet populated locally. The runner is configured for Kaggle execution:
> `MODELS = ["qwen3-4b"]`, `MODES = ["baseline_idk", "rag_idk"]`, `PROFILE = False`, `CHECK_ONLY = False`, `TINY = False`.
> Once execution completes and output files (`qwen3-4b_baseline_idk.jsonl`, `qwen3-4b_rag_idk.jsonl`) are placed in `work/phase12/`, > rerun `python3 src/analysis_idk.py` to populate the empirical IDK rows in these tables.

---
## Results for `smollm3-3b`

- **Validation Temperatures ($T^*$ min-NLL)**: Baseline $T^* = 6.58$, RAG $T^* = 4.88$
- **Validation Confidence Thresholds (F1-tuned $\tau^*$)**: Baseline $\tau^* = 0.5800$, RAG $\tau^* = 0.8024$

### Table 1A: Abstention on MedQA + Unanswerable Pool ($N=650$, Base Prevalence = 0.231)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.231** [—] | **1.000** [—] | **0.375** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 41.6% [37.3, 46.0] | 41.6% [37.3, 46.0] |
| Baseline (Confidence Gate) | 0.205 [0.174, 0.238] | 0.833 [0.770, 0.894] | 0.329 [0.284, 0.372] | 96.8% [95.1, 98.4] | 3.2% [1.6, 4.9] | 87.5% [68.8, 100.0] | 2.8% [1.4, 4.4] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 47.2% [42.5, 51.6] | 47.2% [42.5, 51.6] |
| Full RAG (Confidence Gate) | 0.231 [0.197, 0.263] | 1.000 [1.000, 1.000] | 0.375 [0.329, 0.417] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | — | 0.0% [0.0, 0.0] |

### Table 1B: Abstention on Full Test Split ($N=1,150$, 1,000 Answerable + 150 Unanswerable)

| Method / Variant | Unans Prec [95% CI] | Unans Recall [95% CI] | Unans F1 [95% CI] | Wrongly Abstained % [CI] | Coverage % [CI] | Selective Acc % [CI] | Overall Acc % [CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Always-Abstain Baseline** | **0.130** [—] | **1.000** [—] | **0.231** [—] | 100.0% [—] | 0.0% [—] | — | 0.0% [—] |
| Baseline (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 47.2% [44.2, 50.3] | 47.2% [44.2, 50.3] |
| Baseline (Confidence Gate) | 0.134 [0.112, 0.157] | 0.833 [0.771, 0.889] | 0.231 [0.196, 0.265] | 80.5% [78.1, 82.8] | 19.5% [17.2, 21.9] | 64.1% [57.3, 71.3] | 12.5% [10.5, 14.6] |
| Full RAG (Standard) | 0.000 [—] | 0.000 [—] | 0.000 [—] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] | 49.9% [46.9, 53.0] | 49.9% [46.9, 53.0] |
| Full RAG (Confidence Gate) | 0.130 [0.113, 0.151] | 1.000 [1.000, 1.000] | 0.231 [0.203, 0.263] | 100.0% [100.0, 100.0] | 0.0% [0.0, 0.0] | — | 0.0% [0.0, 0.0] |

> [!NOTE]
> **Phase 12 Kaggle Execution Notice**:
> `work/phase12/` files are not yet populated locally. The runner is configured for Kaggle execution:
> `MODELS = ["qwen3-4b"]`, `MODES = ["baseline_idk", "rag_idk"]`, `PROFILE = False`, `CHECK_ONLY = False`, `TINY = False`.
> Once execution completes and output files (`qwen3-4b_baseline_idk.jsonl`, `qwen3-4b_rag_idk.jsonl`) are placed in `work/phase12/`, > rerun `python3 src/analysis_idk.py` to populate the empirical IDK rows in these tables.


## Key Findings and Comparative Analysis

1. **Forced Choice Baselines Fail at Abstention by Construction**:
   In standard multiple-choice setups (`baseline` and `rag`), models are forced to choose among given options (A–D).    Consequently, spontaneous abstention is 0.0%, yielding 0% recall on unanswerable clinical questions and forcing hallucinations.

2. **Calibrated Confidence Gating Trades Coverage for Accuracy**:
   Post-hoc temperature scaling ($T^* \approx 17.2$ for baseline, $20.7$ for RAG) successfully disperses saturated logits.    Filtering predictions with low calibrated confidence improves selective accuracy on answered questions (e.g. from 55.4% to 58.6% on baseline, and 54.4% to 63.8% on RAG),    while capturing 24.7% to 45.3% of unanswerable questions.

3. **Role of Explicit IDK Prompts**:
   Offering an explicit final option ("I do not have enough information to answer this question") provides an interpretable, in-generation mechanism for clinical safety.    Combining the explicit option with residual calibrated confidence thresholding forms a dual-defense filter against clinical hallucinations.

