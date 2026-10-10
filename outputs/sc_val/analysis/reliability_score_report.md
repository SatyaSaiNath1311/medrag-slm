# MedPsy-4B Combined Reliability Score & Abstention Calibration Report

Evaluates a multi-feature reliability score for MedPsy-4B selective abstention on the **validation split only** (660 questions total):

- **MedQA Validation**: 200 questions ($Y=0$, answerable)
- **PubMedQA Validation**: 200 questions ($Y=0$, answerable context mode)
- **Unanswerable v2 Validation**: 260 questions (200 challenge $Y=1$, 60 controls $Y=0$)

## 1. Out-of-Fold Model Discrimination (Cross-Validation)

A logistic regression model was fit using **Stratified 5-Fold Cross-Validation** on the full 660-row validation dataset.

- **Out-of-Fold AUROC**: **0.7192**
- **Mean Fold AUROC**: **0.7303** (fold scores: 0.7496, 0.7621, 0.7492, 0.7197, 0.6709)

### Fitted Model Coefficients & Feature Importance

| Feature | Standardized Coef | Raw Coef | Odds Ratio (Std) | Interpretation |
|---|---|---|---|---|
| `truncation_rate` | **+0.6775** | +2.8996 | 1.9690 | Strongest positive predictor of unanswerability; thinking token exhaustion indicates model struggle. |
| `distinct_answers` | **+0.4921** | +1.0830 | 1.6357 | Dispersion in candidate answers across SC samples indicates epistemic uncertainty. |
| `greedy_truncated` | **+0.3059** | +1.0130 | 1.3579 | Early truncation in the greedy pass indicates failure to reach resolution. |
| `agreement` | **+0.0261** | +0.1275 | 1.0265 | Higher majority agreement decreases likelihood of abstention. |
| `insufficient_rate` | **+0.0000** | +0.0000 | 1.0000 | Rare in validation generations (< 0.1%); active when explicit abstention is emitted. |
| `greedy_insufficient_info` | **+0.0000** | +0.0000 | 1.0000 | Greedy explicit abstention token emission. |

- **Model Intercept (Standardized)**: -0.8850 (Raw: -2.6586)

## 2. Conformal Threshold Calibration & Method Comparison

Calibration target: **False Abstention Rate (FAR) $\le 5.0\%$ on answerable questions** (MedQA + PubMedQA + 60 controls = 460 items).

| Method | Decision Rule / Threshold | FAR (Answerable) | Unanswerable Recall | Precision | Abstention F1 | Overall Coverage | Selective Accuracy (Answered) |
|---|---|---|---|---|---|---|---|
| **Combined Reliability Score** (Ours) | $\hat{P}(\text{Abstain}) \ge 0.5119$ | **5.0%** (23/460) | **40.0%** (80/200) | **77.7%** | **0.5281** | **84.4%** (557/660) | **87.0%** (380/437) |
| Agreement-Only (Earlier Best) | $\text{Agreement} < 0.9$ | 13.7% (63/460) | 49.5% (99/200) | 61.1% | 0.5470 | 75.5% (498/660) | 89.9% (357/397) |
| Agreement-Only (Conformal FAR $\le 5\%$) | $\text{Agreement} < 0.5$ | 2.2% (10/460) | 18.0% (36/200) | 78.3% | 0.2927 | 93.0% (614/660) | 86.7% (390/450) |

> [!NOTE]
> At the calibrated conformal operating point (FAR $\le 5\%$):
> - The Combined Score detects **40.0%** of unanswerable items compared to only **18.0%** for agreement alone under the same false-abstention constraint.
> - Selective accuracy on retained answerable questions rises from **85.7%** (full coverage) to **87.0%**.

## 3. Recall Comparison at Fixed False Abstention Budgets (1%–15% FAR)

Compares the trade-off between false alarm rate on valid medical questions versus recall on unanswerable challenges:

| Target FAR Budget | Combined Score Recall | Combined Actual FAR | Agreement-Only Recall | Agreement Actual FAR | Recall Advantage (pts) | Combined F1 | Agreement F1 |
|---|---|---|---|---|---|---|---|
| **FAR $\le 1\%$** | **11.5%** | 0.9% | 6.5% | 0.7% | **+5.0%** | 0.2026 | 0.1204 |
| **FAR $\le 2\%$** | **18.0%** | 2.0% | 6.5% | 0.7% | **+11.5%** | 0.2939 | 0.1204 |
| **FAR $\le 5\%$** | **40.0%** | 5.0% | 18.0% | 2.2% | **+22.0%** | 0.5281 | 0.2927 |
| **FAR $\le 10\%$** | **46.5%** | 9.3% | 37.0% | 6.5% | **+9.5%** | 0.5536 | 0.4868 |
| **FAR $\le 15\%$** | **58.0%** | 14.6% | 49.5% | 13.7% | **+8.5%** | 0.6057 | 0.5470 |

## 4. Unanswerable v2 Category Breakdown (New Combined Score)

Breakdown under calibrated decision rule: $\hat{P}(\text{Abstain}) \ge 0.5119$:

| Category | Type | N | Mean Agreement | % Truncated | % Abstained (Recall / FAR) | Control Accuracy |
|---|---|---|---|---|---|---|
| `fabricated_disease` | **Control** | 30 | 1.00 | 0.0% | **0.0%** (0/30) | 100.0% |
| `fabricated_drug` | **Control** | 30 | 1.00 | 0.0% | **0.0%** (0/30) | 100.0% |
| `ambiguous` | **Unanswerable** | 33 | 0.87 | 33.3% | **18.2%** (6/33) | — |
| `fabricated_disease` | **Unanswerable** | 33 | 0.59 | 57.6% | **48.5%** (16/33) | — |
| `fabricated_drug` | **Unanswerable** | 33 | 0.73 | 48.5% | **45.5%** (15/33) | — |
| `false_premise` | **Unanswerable** | 33 | 0.75 | 78.8% | **63.6%** (21/33) | — |
| `missing_info` | **Unanswerable** | 34 | 0.88 | 32.4% | **26.5%** (9/34) | — |
| `out_of_scope` | **Unanswerable** | 34 | 0.82 | 26.5% | **38.2%** (13/34) | — |

### Diagnostic Observations on Difficult Categories

- **Fabricated Disease & Drug Controls**: Falsely abstained at **0.0%** (0/60) with **100.0%** accuracy on real controls.
- **Ambiguous Questions**: Recall is **18.2%** under the conservative 5% global FAR budget. When model agreement is high despite ambiguity, truncation features capture subtle reasoning stress.
- **Missing Information**: Recall is **26.5%** under the 5% FAR constraint.
- **False Premise & Fabricated Diseases**: Show the highest recall (**63.6%** and **48.5%** respectively), driven by elevated truncation and answer dispersion.

## 5. Documented Scope & Methodology Limits

> [!NOTE]
> Evaluated in baseline (non-RAG) mode; retrieval-grounding signals (rerank score, NLI entailment) were not available in this run.

