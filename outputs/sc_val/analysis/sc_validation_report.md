# MedPsy-4B Self-Consistency Validation Report

Evaluates self-consistency (SC, $K=5$, $T=0.7$, Top-$p=0.95$) on validation splits:

- **MedQA Validation**: 200 questions, baseline reasoning mode.
- **PubMedQA Validation**: 200 questions, +abstract context reasoning mode.
- **Unanswerable Challenge v2 Validation**: 260 questions (200 unanswerable, 60 controls).

## 1. Accuracy: Greedy vs. Self-Consistency Majority Vote

- **MedQA Greedy Baseline Source**: hybrid: row['greedy_answer'] (191/200) + fallback reference (9/200)
- **PubMedQA Greedy Baseline Source**: hybrid: row['greedy_answer'] (199/200) + fallback reference (1/200)

| Dataset | N | Greedy Acc [95% CI] | SC Maj-Vote Acc [95% CI] | Δ Acc (pts) [95% CI] | SC Wins / Greedy Wins | McNemar $p$-value | AUROC [95% CI] | ECE |
|---|---|---|---|---|---|---|---|---|
| **MEDQA** | 200 | 90.5% [86.5%, 94.5%] | **91.0%** [87.0%, 95.0%] | **+0.5%** [-2.0%, +3.5%] | 4 / 3 | $p=1.0000$ | 0.7924 [0.666, 0.911] | 0.0330 |
| **PUBMEDQA** | 200 | 73.5% [67.5%, 79.5%] | **76.0%** [70.5%, 81.5%] | **+2.5%** [+0.5%, +5.5%] | 6 / 1 | $p=0.1250$ | 0.5896 [0.523, 0.668] | 0.2090 |

## 2. Abstention Performance on Unanswerable v2 Validation Set

Evaluates 3 decision rules side by side across agreement thresholds $\tau \in [0.2, 1.0]$:

- **(a) Current Rule**: $\text{Agreement} < \tau$ OR any sample truncated OR any sample outputs `INSUFFICIENT INFORMATION`.
- **(b) Fraction Rule**: $\text{Agreement} < \tau$ OR truncation rate $\ge 0.6$ OR insufficient rate $\ge 0.6$.
- **(c) Agreement-Only Rule**: $\text{Agreement} < \tau$ OR $\text{majority\_answer} == \text{'ABSTAIN'}$.

> [!IMPORTANT]
> **★ BEST RULE ON VALIDATION (by F1)**: **(a) Current Rule** at **$\tau = 0.9$**  
> - **F1 Score**: **0.726** [0.669, 0.782]  
> - **Recall**: **57.0%** [50.3%, 64.2%]  
> - **False Abstention Rate on Controls**: **0.0%** [0.0%, 0.0%]  
> - **Coverage**: **56.1%**  

### Side-by-Side Comparison Matrix

| Agreement $\tau$ | (a) Current Rule<br>F1 (Rec / FAR / Cov) | (b) Fraction Rule<br>F1 (Rec / FAR / Cov) | (c) Agreement-Only<br>F1 (Rec / FAR / Cov) | Best Rule at $\tau$ |
|---|---|---|---|---|
| **$\tau = 0.2$** | **0.630** (46.0% / 0.0% / 64.6%) | **0.361** (22.0% / 0.0% / 83.1%) | **0.030** (1.5% / 0.0% / 98.9%) | **(a) Current Rule** |
| **$\tau = 0.3$** | **0.630** (46.0% / 0.0% / 64.6%) | **0.361** (22.0% / 0.0% / 83.1%) | **0.122** (6.5% / 0.0% / 95.0%) | **(a) Current Rule** |
| **$\tau = 0.4$** | **0.630** (46.0% / 0.0% / 64.6%) | **0.361** (22.0% / 0.0% / 83.1%) | **0.122** (6.5% / 0.0% / 95.0%) | **(a) Current Rule** |
| **$\tau = 0.5$** | **0.635** (46.5% / 0.0% / 64.2%) | **0.425** (27.0% / 0.0% / 79.2%) | **0.305** (18.0% / 0.0% / 86.2%) | **(a) Current Rule** |
| **$\tau = 0.6$** | **0.635** (46.5% / 0.0% / 64.2%) | **0.425** (27.0% / 0.0% / 79.2%) | **0.305** (18.0% / 0.0% / 86.2%) | **(a) Current Rule** |
| **$\tau = 0.7$** | **0.689** (52.5% / 0.0% / 59.6%) | **0.592** (42.0% / 0.0% / 67.7%) | **0.540** (37.0% / 0.0% / 71.5%) | **(a) Current Rule** |
| **$\tau = 0.8$** | **0.689** (52.5% / 0.0% / 59.6%) | **0.592** (42.0% / 0.0% / 67.7%) | **0.540** (37.0% / 0.0% / 71.5%) | **(a) Current Rule** |
| **$\tau = 0.9$** | **0.726** (57.0% / 0.0% / 56.1%) | **0.689** (52.5% / 0.0% / 59.6%) | **0.662** (49.5% / 0.0% / 61.9%) | **(a) Current Rule** |
| **$\tau = 1.0$** | **0.726** (57.0% / 0.0% / 56.1%) | **0.689** (52.5% / 0.0% / 59.6%) | **0.662** (49.5% / 0.0% / 61.9%) | **(a) Current Rule** |

### Comprehensive Rule Breakdown across Thresholds

| Rule | $\tau$ | Precision [95% CI] | Recall [95% CI] | False Abstention (Controls) [95% CI] | F1 Score [95% CI] | Coverage | Status |
|---|---|---|---|---|---|---|---|
| (a) Current Rule | **0.2** | 100.0% [100.0%, 100.0%] | 46.0% [38.6%, 52.9%] | 0.0% [0.0%, 0.0%] | **0.630** [0.557, 0.692] | 64.6% | — |
| (b) Fraction Rule | **0.2** | 100.0% [100.0%, 100.0%] | 22.0% [16.4%, 27.9%] | 0.0% [0.0%, 0.0%] | **0.361** [0.282, 0.436] | 83.1% | — |
| (c) Agreement-Only | **0.2** | 100.0% [0.0%, 100.0%] | 1.5% [0.0%, 3.6%] | 0.0% [0.0%, 0.0%] | **0.030** [0.000, 0.069] | 98.9% | — |
| (a) Current Rule | **0.3** | 100.0% [100.0%, 100.0%] | 46.0% [38.6%, 52.9%] | 0.0% [0.0%, 0.0%] | **0.630** [0.557, 0.692] | 64.6% | — |
| (b) Fraction Rule | **0.3** | 100.0% [100.0%, 100.0%] | 22.0% [16.4%, 27.9%] | 0.0% [0.0%, 0.0%] | **0.361** [0.282, 0.436] | 83.1% | — |
| (c) Agreement-Only | **0.3** | 100.0% [100.0%, 100.0%] | 6.5% [3.1%, 10.3%] | 0.0% [0.0%, 0.0%] | **0.122** [0.060, 0.186] | 95.0% | — |
| (a) Current Rule | **0.4** | 100.0% [100.0%, 100.0%] | 46.0% [38.6%, 52.9%] | 0.0% [0.0%, 0.0%] | **0.630** [0.557, 0.692] | 64.6% | — |
| (b) Fraction Rule | **0.4** | 100.0% [100.0%, 100.0%] | 22.0% [16.4%, 27.9%] | 0.0% [0.0%, 0.0%] | **0.361** [0.282, 0.436] | 83.1% | — |
| (c) Agreement-Only | **0.4** | 100.0% [100.0%, 100.0%] | 6.5% [3.1%, 10.3%] | 0.0% [0.0%, 0.0%] | **0.122** [0.060, 0.186] | 95.0% | — |
| (a) Current Rule | **0.5** | 100.0% [100.0%, 100.0%] | 46.5% [39.3%, 53.4%] | 0.0% [0.0%, 0.0%] | **0.635** [0.564, 0.697] | 64.2% | — |
| (b) Fraction Rule | **0.5** | 100.0% [100.0%, 100.0%] | 27.0% [20.7%, 33.2%] | 0.0% [0.0%, 0.0%] | **0.425** [0.343, 0.498] | 79.2% | — |
| (c) Agreement-Only | **0.5** | 100.0% [100.0%, 100.0%] | 18.0% [12.6%, 23.6%] | 0.0% [0.0%, 0.0%] | **0.305** [0.224, 0.381] | 86.2% | — |
| (a) Current Rule | **0.6** | 100.0% [100.0%, 100.0%] | 46.5% [39.3%, 53.4%] | 0.0% [0.0%, 0.0%] | **0.635** [0.564, 0.697] | 64.2% | — |
| (b) Fraction Rule | **0.6** | 100.0% [100.0%, 100.0%] | 27.0% [20.7%, 33.2%] | 0.0% [0.0%, 0.0%] | **0.425** [0.343, 0.498] | 79.2% | — |
| (c) Agreement-Only | **0.6** | 100.0% [100.0%, 100.0%] | 18.0% [12.6%, 23.6%] | 0.0% [0.0%, 0.0%] | **0.305** [0.224, 0.381] | 86.2% | — |
| (a) Current Rule | **0.7** | 100.0% [100.0%, 100.0%] | 52.5% [45.3%, 59.6%] | 0.0% [0.0%, 0.0%] | **0.689** [0.624, 0.747] | 59.6% | — |
| (b) Fraction Rule | **0.7** | 100.0% [100.0%, 100.0%] | 42.0% [35.3%, 49.3%] | 0.0% [0.0%, 0.0%] | **0.592** [0.522, 0.660] | 67.7% | — |
| (c) Agreement-Only | **0.7** | 100.0% [100.0%, 100.0%] | 37.0% [30.6%, 43.8%] | 0.0% [0.0%, 0.0%] | **0.540** [0.468, 0.609] | 71.5% | — |
| (a) Current Rule | **0.8** | 100.0% [100.0%, 100.0%] | 52.5% [45.3%, 59.6%] | 0.0% [0.0%, 0.0%] | **0.689** [0.624, 0.747] | 59.6% | — |
| (b) Fraction Rule | **0.8** | 100.0% [100.0%, 100.0%] | 42.0% [35.3%, 49.3%] | 0.0% [0.0%, 0.0%] | **0.592** [0.522, 0.660] | 67.7% | — |
| (c) Agreement-Only | **0.8** | 100.0% [100.0%, 100.0%] | 37.0% [30.6%, 43.8%] | 0.0% [0.0%, 0.0%] | **0.540** [0.468, 0.609] | 71.5% | — |
| (a) Current Rule | **0.9** | 100.0% [100.0%, 100.0%] | 57.0% [50.3%, 64.2%] | 0.0% [0.0%, 0.0%] | **0.726** [0.669, 0.782] | 56.1% | ★ **BEST RULE** |
| (b) Fraction Rule | **0.9** | 100.0% [100.0%, 100.0%] | 52.5% [45.5%, 59.7%] | 0.0% [0.0%, 0.0%] | **0.689** [0.625, 0.748] | 59.6% | — |
| (c) Agreement-Only | **0.9** | 100.0% [100.0%, 100.0%] | 49.5% [42.9%, 56.6%] | 0.0% [0.0%, 0.0%] | **0.662** [0.600, 0.723] | 61.9% | — |
| (a) Current Rule | **1.0** | 100.0% [100.0%, 100.0%] | 57.0% [50.3%, 64.2%] | 0.0% [0.0%, 0.0%] | **0.726** [0.669, 0.782] | 56.1% | — |
| (b) Fraction Rule | **1.0** | 100.0% [100.0%, 100.0%] | 52.5% [45.5%, 59.7%] | 0.0% [0.0%, 0.0%] | **0.689** [0.625, 0.748] | 59.6% | — |
| (c) Agreement-Only | **1.0** | 100.0% [100.0%, 100.0%] | 49.5% [42.9%, 56.6%] | 0.0% [0.0%, 0.0%] | **0.662** [0.600, 0.723] | 61.9% | — |

## 3. Unanswerable v2 Breakdown by Category

> Evaluated under validation winning rule: **(a) Current Rule at $\tau=0.9$**.

| Category | Type | N | Mean Agreement | % Truncated | % 'INSUFFICIENT INFO' | % Abstained | Majority-Vote Acc (Controls) |
|---|---|---|---|---|---|---|---|
| `fabricated_disease` | **Control** | 30 | 1.00 | 0.0% | 0.0% | 0.0% | 100.0% |
| `fabricated_drug` | **Control** | 30 | 1.00 | 0.0% | 0.0% | 0.0% | 100.0% |
| `ambiguous` | **Unanswerable** | 33 | 0.87 | 33.3% | 0.0% | 36.4% | — |
| `fabricated_disease` | **Unanswerable** | 33 | 0.59 | 57.6% | 0.0% | 72.7% | — |
| `fabricated_drug` | **Unanswerable** | 33 | 0.73 | 48.5% | 0.0% | 60.6% | — |
| `false_premise` | **Unanswerable** | 33 | 0.75 | 78.8% | 0.0% | 87.9% | — |
| `missing_info` | **Unanswerable** | 34 | 0.88 | 32.4% | 0.0% | 38.2% | — |
| `out_of_scope` | **Unanswerable** | 34 | 0.82 | 26.5% | 0.0% | 47.1% | — |

## 4. Paired Comparison of Fabricated vs. Real Controls (Identical Stems)

| Domain | Matched Pairs | Fake Mean Agreement | Control Mean Agreement | Δ Agreement | Wilcoxon $p$-value | Fake Abstention (Recall) | Control False Abstention | Selective Abstention Success (Fake Abst & Ctrl Ans) | Hallucinated on Fake (Both Ans) |
|---|---|---|---|---|---|---|---|---|---|
| **Drug** | 19 | 0.69 | 1.00 | **+0.31** | $p=0.0015$ | 68.4% | 0.0% | **13/19 (68.4%)** | 6/19 (31.6%) |
| **Disease** | 21 | 0.69 | 1.00 | **+0.31** | $p=0.0015$ | 61.9% | 0.0% | **13/21 (61.9%)** | 8/21 (38.1%) |
| **Combined** | 40 | 0.69 | 1.00 | **+0.31** | $p=0.0000$ | 65.0% | 0.0% | **26/40 (65.0%)** | 14/40 (35.0%) |

