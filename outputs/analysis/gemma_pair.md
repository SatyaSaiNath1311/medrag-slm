# Matched-Pair Statistical Report: Gemma-3-4B vs MedGemma-1.5-4B-it

This report evaluates the generic-vs-medical matched pair under Phase 1 V3 frozen-test conditions:
- **Generic Model**: `google/gemma-3-4b-it` (`gemma3-4b`)
- **Medical Domain Model**: `google/medgemma-1.5-4b-it` (`medgemma-1.5-4b-it`)
- **Hardware & Runtime**: Identical dual-GPU T4 inference (`float32`), greedy decoding, same prompts, same seeds.

# Pre-Flight Frozen Data Integrity Audit

| Condition | Questions (N) | Gemma-3-4B Path | MedGemma-1.5-4B-it Path | Status |
| :--- | :---: | :--- | :--- | :---: |
| C1 | 1000 | `outputs/v3/gemma3-4b/C1/gemma3-4b.jsonl` | `outputs/v3/medgemma-1.5-4b-it/C1/medgemma-1.5-4b-it.jsonl` | PASSED |
| C2 | 1000 | `outputs/v3/gemma3-4b/C2/gemma3-4b.jsonl` | `outputs/v3/medgemma-1.5-4b-it/C2/medgemma-1.5-4b-it.jsonl` | PASSED |
| C3 | 500 | `outputs/v3/gemma3-4b/C3/gemma3-4b.jsonl` | `outputs/v3/medgemma-1.5-4b-it/C3/medgemma-1.5-4b-it.jsonl` | PASSED |
| C4 | 140 | `outputs/v3/gemma3-4b/C4/gemma3-4b.jsonl` | `outputs/v3/medgemma-1.5-4b-it/C4/medgemma-1.5-4b-it.jsonl` | PASSED |
| C5 | 140 | `outputs/v3/gemma3-4b/C5/gemma3-4b.jsonl` | `outputs/v3/medgemma-1.5-4b-it/C5/medgemma-1.5-4b-it.jsonl` | PASSED |

All pre-flight assertions passed: 100% frozen `test` split, zero duplicate IDs, and identical question sets across models.

## Section 1: Accuracy per Condition (C1, C2, C3) and Dataset

Evaluation of answerable multiple-choice benchmark questions:
- **Strict Accuracy**: `(not truncated) and parsed and correct` (unparsed or truncated count as 0)
- **With-Fallback Accuracy**: raw stored `correct` field (includes logprob or heuristic fallback if unparsed)
- Confidence intervals are Wilson 95% score intervals.

| Condition | Dataset | Model | N | Strict Correct | Strict Acc (95% CI) | Fallback Correct | Fallback Acc (95% CI) | Truncated | Unparsed |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C1 | MEDQA | Gemma-3-4B (Generic) | 500 | 243 | 0.4860 `[0.4425, 0.5297]` | 243 | 0.4860 `[0.4425, 0.5297]` | 0 | 0 |
| C1 | MEDQA | MedGemma-1.5-4B-it (Medical) | 500 | 274 | 0.5480 `[0.5042, 0.5911]` | 274 | 0.5480 `[0.5042, 0.5911]` | 0 | 0 |
| C1 | PUBMEDQA | Gemma-3-4B (Generic) | 500 | 246 | 0.4920 `[0.4484, 0.5357]` | 246 | 0.4920 `[0.4484, 0.5357]` | 0 | 0 |
| C1 | PUBMEDQA | MedGemma-1.5-4B-it (Medical) | 500 | 254 | 0.5080 `[0.4643, 0.5516]` | 254 | 0.5080 `[0.4643, 0.5516]` | 0 | 0 |
| C2 | MEDQA | Gemma-3-4B (Generic) | 500 | 265 | 0.5300 `[0.4862, 0.5734]` | 265 | 0.5300 `[0.4862, 0.5734]` | 0 | 0 |
| C2 | MEDQA | MedGemma-1.5-4B-it (Medical) | 500 | 286 | 0.5720 `[0.5282, 0.6147]` | 286 | 0.5720 `[0.5282, 0.6147]` | 0 | 0 |
| C2 | PUBMEDQA | Gemma-3-4B (Generic) | 500 | 251 | 0.5020 `[0.4583, 0.5456]` | 251 | 0.5020 `[0.4583, 0.5456]` | 0 | 0 |
| C2 | PUBMEDQA | MedGemma-1.5-4B-it (Medical) | 500 | 255 | 0.5100 `[0.4663, 0.5536]` | 255 | 0.5100 `[0.4663, 0.5536]` | 0 | 0 |
| C3 | PUBMEDQA | Gemma-3-4B (Generic) | 500 | 265 | 0.5300 `[0.4862, 0.5734]` | 265 | 0.5300 `[0.4862, 0.5734]` | 0 | 0 |
| C3 | PUBMEDQA | MedGemma-1.5-4B-it (Medical) | 500 | 285 | 0.5700 `[0.5262, 0.6127]` | 285 | 0.5700 `[0.5262, 0.6127]` | 0 | 0 |

> **Note**: In conditions C1, C2, and C3 for both models, truncation was 0/500 and parse rate was 500/500 (100.0%). Therefore, Strict Accuracy and With-Fallback Accuracy are identical across all benchmark cells.

## Section 2: Paired Generic vs Medical Comparison (Condition x Dataset)

Exact paired comparison on identical test questions ($N=500$ per cell):
- $b$: Generic correct, Medical incorrect (Generic win)
- $c$: Generic incorrect, Medical correct (Medical win)
- Raw two-sided McNemar exact binomial p-value under $H_0: p = 0.5$ on discordant pairs ($b, c$)
- **Holm-Adjusted p-value**: Step-down family-wise error rate control across all 5 paired tests ($m=5$)
- Paired Accuracy Difference ($\Delta = \text{MedGemma} - \text{Gemma3}$) with percentile bootstrap 95% CI (10,000 resamples, seed 42)
- Significance is marked **only** where Holm-adjusted $p < 0.05$.

| Condition | Dataset | N | Gemma3 Acc | MedGemma Acc | Concordant (1-1 / 0-0) | Discordant (b / c) | Raw p | Holm p | Acc Diff (Med - Gen) | Bootstrap 95% CI | Sig (Holm p < 0.05) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C1 | MEDQA | 500 | 0.4860 | 0.5480 | 168 / 151 | b=75 / c=106 | 0.02548 | 0.12742 | +0.0620 (+31) | `[+0.0080, +0.1140]` | No |
| C1 | PUBMEDQA | 500 | 0.4920 | 0.5080 | 201 / 201 | b=45 / c=53 | 0.47969 | 0.95938 | +0.0160 (+8) | `[-0.0240, +0.0540]` | No |
| C2 | MEDQA | 500 | 0.5300 | 0.5720 | 207 / 156 | b=58 / c=79 | 0.08713 | 0.26140 | +0.0420 (+21) | `[-0.0040, +0.0880]` | No |
| C2 | PUBMEDQA | 500 | 0.5020 | 0.5100 | 202 / 196 | b=49 / c=53 | 0.76660 | 0.95938 | +0.0080 (+4) | `[-0.0300, +0.0480]` | No |
| C3 | PUBMEDQA | 500 | 0.5300 | 0.5700 | 232 / 182 | b=33 / c=53 | 0.03985 | 0.15941 | +0.0400 (+20) | `[+0.0040, +0.0760]` | No |

**Statistical Observations**:
- On MedQA C1 (Closed-Book), MedGemma-1.5-4B-it shows a +6.20% margin over Gemma-3-4B (raw $p = 0.0255$). However, when adjusting for the family of 5 paired tests via Holm-Bonferroni, the adjusted $p = 0.1274$, which does not cross the $\alpha = 0.05$ threshold.
- On PubMedQA C3 (Oracle Abstract), MedGemma achieves +4.00% higher accuracy (raw $p = 0.0399$), but this difference is likewise non-significant after Holm adjustment ($p = 0.1594$).
- Across all five conditions and datasets, no paired accuracy difference between Gemma-3-4B and MedGemma-1.5-4B-it remains statistically significant after family-wise error rate control.
- The two models differ across multiple dimensions beyond training domain data, including pretraining token mixtures, base checkpoints, and instruction-tuning recipes.

## Section 3: RAG Gain (C2 minus C1) and Cross-Model Interaction

Paired within-model RAG effect (C2 vs C1) on identical test questions ($N=500$):
- $b_{\text{RAG}}$: C1 correct, C2 incorrect (RAG degraded)
- $c_{\text{RAG}}$: C1 incorrect, C2 correct (RAG rescued)
- Raw exact two-sided McNemar p-value testing within-model RAG effect ($H_0: b_{\text{RAG}} = c_{\text{RAG}}$)
- **Holm-Adjusted p-value**: Step-down adjustment across the family of 4 within-model tests ($m=4$)
- Difference in RAG Gain ($\Delta \text{Gain} = \text{Gain}_{\text{MedGemma}} - \text{Gain}_{\text{Gemma3}}$) with paired bootstrap 95% CI (10,000 resamples, seed 42)
- Significance marked **only** where Holm-adjusted $p < 0.05$.

| Dataset | Model | C1 Acc | C2 Acc | RAG Gain (C2 - C1) | Discordant (b: Degraded / c: Rescued) | Raw p | Holm p | Sig (Holm p < 0.05) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MEDQA | Gemma-3-4B (Generic) | 0.4860 | 0.5300 | +0.0440 (+22) | b=65 / c=87 | 0.08817 | 0.35270 | No |
| MEDQA | MedGemma-1.5-4B-it (Medical) | 0.5480 | 0.5720 | +0.0240 (+12) | b=55 / c=67 | 0.31931 | 0.95792 | No |
| PUBMEDQA | Gemma-3-4B (Generic) | 0.4920 | 0.5020 | +0.0100 (+5) | b=69 / c=74 | 0.73814 | 1.00000 | No |
| PUBMEDQA | MedGemma-1.5-4B-it (Medical) | 0.5080 | 0.5100 | +0.0020 (+1) | b=46 / c=47 | 1.00000 | 1.00000 | No |

### Difference in RAG Gain (Cross-Model Interaction)

| Dataset | Gemma3 Gain | MedGemma Gain | Gain Difference (MedGemma - Gemma3) | Bootstrap 95% CI (10,000 resamples) | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| MEDQA | +0.0440 | +0.0240 | -0.0200 | `[-0.0780, +0.0360]` | RAG gain was +4.40% in Gemma3 vs +2.40% in MedGemma; net interaction is -2.00%, with 95% CI spanning zero (no significant interaction). |
| PUBMEDQA | +0.0100 | +0.0020 | -0.0080 | `[-0.0600, +0.0440]` | Both models exhibited negligible RAG gain on PubMedQA (+1.00% vs +0.20%); net interaction is -0.80% (95% CI spans zero). |

## Section 4: PubMedQA Test Set Distribution and Model Prediction Dynamics

PubMedQA labeled test split ($N=500$ questions) has three answer options: `A` (yes), `B` (no), and `C` (maybe).

### Gold Label Distribution (N=500)

| Option Letter | Semantic Meaning | Count (N) | Percentage (%) | Wilson 95% CI |
| :---: | :---: | :---: | :---: | :---: |
| **A** | `yes` | 276 | 55.20% | `[0.5082, 0.5950]` |
| **B** | `no` | 164 | 32.80% | `[0.2883, 0.3703]` |
| **C** | `maybe` | 60 | 12.00% | `[0.0944, 0.1514]` |

- **Majority-Class Baseline**: Always predicting **A** (`yes`) yields an accuracy of **276/500 (55.20%)**.

### Model Prediction Distributions Across Conditions (C1, C2, C3)

| Condition | Context Type | Model | Accuracy | vs Majority Baseline | Pred A (yes) | Pred B (no) | Pred C (maybe) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| C1 | Closed-Book (None) | Gemma-3-4B | 0.4920 (246/500) | -0.0600 (-6.00%) | 330 (66.0%) | 43 (8.6%) | 127 (25.4%) |
| C1 | Closed-Book (None) | MedGemma-1.5-4B-it | 0.5080 (254/500) | -0.0440 (-4.40%) | 343 (68.6%) | 67 (13.4%) | 90 (18.0%) |
| C2 | Retrieved RAG (Top-5) | Gemma-3-4B | 0.5020 (251/500) | -0.0500 (-5.00%) | 311 (62.2%) | 159 (31.8%) | 30 (6.0%) |
| C2 | Retrieved RAG (Top-5) | MedGemma-1.5-4B-it | 0.5100 (255/500) | -0.0420 (-4.20%) | 336 (67.2%) | 95 (19.0%) | 69 (13.8%) |
| C3 | Oracle Abstract | Gemma-3-4B | 0.5300 (265/500) | -0.0220 (-2.20%) | 359 (71.8%) | 81 (16.2%) | 60 (12.0%) |
| C3 | Oracle Abstract | MedGemma-1.5-4B-it | 0.5700 (285/500) | +0.0180 (+1.80%) | 430 (86.0%) | 60 (12.0%) | 10 (2.0%) |

**PubMedQA Dynamics**:
1. **Majority-Class Deficit**: In both C1 (Closed-Book) and C2 (Retrieved RAG), **neither model outperforms the simple majority-class heuristic** (55.20%). Gemma-3-4B scores 49.20% (C1) and 50.20% (C2); MedGemma-1.5-4B-it scores 50.80% (C1) and 51.00% (C2).
2. **Effect of Retrieved RAG**: Top-5 textbook RAG evidence shifts Gemma3's predictions towards 'no' (from 43 up to 159), but does not improve overall accuracy (+1.00%).
3. **Oracle Context (C3)**: When provided the actual study abstract, MedGemma strongly shifts toward predicting 'yes' (430/500, 86.0%), surpassing the majority class at 57.00% (285/500). Gemma-3-4B remains below the majority baseline at 53.00% (265/500).

## Section 5: C4/C5 Challenge vs Control Separation Analysis

### Confidence Metric and Tie Handling Definition
In [src/qa_pipeline_v3.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/qa_pipeline_v3.py#L341-L344) and [src/llm.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/llm.py#L380-L390):
- **Field Used**: `row['confidence']`
- **What it Measures**: At generation step 0 (immediately following prompt prefix `Answer:`), next-token logits are extracted for valid option letters $\{A, B, C, D\}$ and converted to probabilities via softmax normalized strictly across those letters (`row['letter_probs']`). `confidence = max(letter_probs.values())`.
- **Detection AUROC**: Evaluates $(1 - \text{confidence})$ as a continuous detector for separating unanswerable challenge rows ($y=1$, `should_abstain=True`) from answerable control rows ($y=0$, `should_abstain=False`).
- **Tie Handling**: AUROC is computed as the fraction of (challenge, control) pairs where the challenge score $(1 - \text{conf}_{\text{chal}})$ exceeds the control score $(1 - \text{conf}_{\text{ctrl}})$. Because many Gemma-3-4B predictions saturate at $\text{confidence} = 1.0000$ on both control and challenge questions ($1 - \text{conf} = 0.0000$), **tied pairs are assigned exact half credit (0.5)**, corresponding mathematically to the mid-rank Wilcoxon-Mann-Whitney U formulation.
- **Bootstrap Confidence Intervals**: 10,000 paired question resamples (seed 42) are drawn for each condition, preserving 1-to-1 question alignment between models.

### Overall Separation Metrics (N=140: 100 Challenge + 40 Control)

| Condition | Model | Chal Mean Conf | Ctrl Mean Conf | Conf Gap (Ctrl - Chal) | Chal Conf >= 0.90 | AUROC (1 - Conf) | AUROC 95% Bootstrap CI | AUROC Diff (Med - Gen) | Diff 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C4 | Gemma-3-4B (Generic) | 0.9919 | 1.0000 | +0.0081 | 0.9700 (97/100) | 0.7650 | `[0.7151, 0.8131]` | — | — |
| C4 | MedGemma-1.5-4B-it (Medical) | 0.8473 | 0.9983 | +0.1510 | 0.5900 (59/100) | 0.9617 | `[0.9298, 0.9862]` | **+0.1967** | `[+0.1434, +0.2510]` |
| C5 | Gemma-3-4B (Generic) | 0.9765 | 0.9999 | +0.0234 | 0.9400 (94/100) | 0.8325 | `[0.7593, 0.8959]` | — | — |
| C5 | MedGemma-1.5-4B-it (Medical) | 0.8355 | 0.9889 | +0.1534 | 0.5500 (55/100) | 0.8982 | `[0.8374, 0.9490]` | **+0.0657** | `[-0.0052, +0.1393]` |

**AUROC Findings**:
- In **C4 (Baseline)**, MedGemma achieves a statistically significant +0.1967 higher AUROC than Gemma3 ($0.9617$ vs $0.7650$, paired bootstrap 95% CI $[+0.1434, +0.2510]$). Gemma3 assigned $\ge 0.90$ confidence to 97% of unanswerable questions, whereas MedGemma did so for 59%.
- In **C5 (RAG)**, MedGemma's AUROC is $0.8982$ vs Gemma3's $0.8325$; the paired difference is +0.0657, with 95% CI $[-0.0052, +0.1393]$ spanning zero.

### Breakdown by Unanswerability Category

AUROC is evaluated for each category against all 40 control questions. Tied pairs receive half credit (0.5).

| Condition | Model | Category | Chal N | Chal Mean Conf | Chal >= 0.90 | Ctrl N | Ctrl Mean Conf | AUROC (vs All Controls) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| C4 | Gemma-3-4B | `ambiguous` | 17 | 0.9992 | 1.0000 (17/17) | 0 | — | 0.6765 |
| C4 | Gemma-3-4B | `fabricated_disease` | 17 | 0.9982 | 1.0000 (17/17) | 20 | 1.0000 | 0.6765 |
| C4 | Gemma-3-4B | `fabricated_drug` | 17 | 0.9897 | 0.9412 (16/17) | 20 | 1.0000 | 0.8824 |
| C4 | Gemma-3-4B | `false_premise` | 17 | 0.9831 | 0.9412 (16/17) | 0 | — | 0.6765 |
| C4 | Gemma-3-4B | `missing_info` | 16 | 0.9997 | 1.0000 (16/16) | 0 | — | 0.8125 |
| C4 | Gemma-3-4B | `out_of_scope` | 16 | 0.9815 | 0.9375 (15/16) | 0 | — | 0.8750 |
| C4 | MedGemma-1.5-4B-it | `ambiguous` | 17 | 0.8649 | 0.5882 (10/17) | 0 | — | 0.9044 |
| C4 | MedGemma-1.5-4B-it | `fabricated_disease` | 17 | 0.8759 | 0.6471 (11/17) | 20 | 0.9983 | 0.9750 |
| C4 | MedGemma-1.5-4B-it | `fabricated_drug` | 17 | 0.8822 | 0.7059 (12/17) | 20 | 0.9983 | 0.9853 |
| C4 | MedGemma-1.5-4B-it | `false_premise` | 17 | 0.8660 | 0.6471 (11/17) | 0 | — | 0.9912 |
| C4 | MedGemma-1.5-4B-it | `missing_info` | 16 | 0.7998 | 0.5000 (8/16) | 0 | — | 0.9656 |
| C4 | MedGemma-1.5-4B-it | `out_of_scope` | 16 | 0.7886 | 0.4375 (7/16) | 0 | — | 0.9484 |
| C5 | Gemma-3-4B | `ambiguous` | 17 | 0.9794 | 0.9412 (16/17) | 0 | — | 0.8279 |
| C5 | Gemma-3-4B | `fabricated_disease` | 17 | 1.0000 | 1.0000 (17/17) | 20 | 1.0000 | 0.7287 |
| C5 | Gemma-3-4B | `fabricated_drug` | 17 | 0.9992 | 1.0000 (17/17) | 20 | 0.9998 | 0.7507 |
| C5 | Gemma-3-4B | `false_premise` | 17 | 0.9640 | 0.9412 (16/17) | 0 | — | 0.9243 |
| C5 | Gemma-3-4B | `missing_info` | 16 | 0.9968 | 1.0000 (16/16) | 0 | — | 0.8492 |
| C5 | Gemma-3-4B | `out_of_scope` | 16 | 0.9171 | 0.7500 (12/16) | 0 | — | 0.9203 |
| C5 | MedGemma-1.5-4B-it | `ambiguous` | 17 | 0.8885 | 0.5882 (10/17) | 0 | — | 0.8485 |
| C5 | MedGemma-1.5-4B-it | `fabricated_disease` | 17 | 0.9299 | 0.7647 (13/17) | 20 | 0.9880 | 0.8750 |
| C5 | MedGemma-1.5-4B-it | `fabricated_drug` | 17 | 0.8330 | 0.6471 (11/17) | 20 | 0.9898 | 0.9250 |
| C5 | MedGemma-1.5-4B-it | `false_premise` | 17 | 0.7575 | 0.4706 (8/17) | 0 | — | 0.9691 |
| C5 | MedGemma-1.5-4B-it | `missing_info` | 16 | 0.8507 | 0.5625 (9/16) | 0 | — | 0.8297 |
| C5 | MedGemma-1.5-4B-it | `out_of_scope` | 16 | 0.7490 | 0.2500 (4/16) | 0 | — | 0.9406 |

## Section 6: Unparsed Rows and Verbal Abstention Analysis

### Table of Unparsed Rows (`parsed == False`)

Every row across all conditions (C1-C5) where `parsed` is False:

| Model | Condition | Question ID | should_abstain | category | First 120 Characters of raw_output |
| :--- | :---: | :--- | :---: | :---: | :--- |
| gemma3-4b | C4 | `unans-v2-test-0115` | True | `false_premise` | 'Answer: **None of the above.**\n\n**Explanation:**\n\nThe human gallbladder does *' |
| gemma3-4b | C5 | `unans-v2-test-0096` | True | `out_of_scope` | 'Answer: None of the options provided are supported by the evidence. The passages discuss the benefits and challenges of ' |
| gemma3-4b | C5 | `unans-v2-test-0120` | True | `ambiguous` | 'Answer: None\n\nEvidence: [1], [3], [4], [5]\n\nThe evidence consistently demonstrates that *no* screening modality has zero' |

Total unparsed rows across all conditions: **3** (all 3 in Gemma-3-4B, 0 in MedGemma-1.5-4B-it).

### Verbal Abstention Classification
Classification criterion: `raw_output` following `Answer:` starts with `None`, `Insufficient`, `Not enough`, or `Cannot` (case-insensitive, ignoring markdown bold/quotes).

| Condition | Model | Challenge Verbal Abstentions | Control Verbal Abstentions | Total Verbal Abstentions | Details |
| :--- | :--- | :---: | :---: | :---: | :--- |
| C1 | Gemma-3-4B | 0 / 0 | 0 / 1000 | **0** | None |
| C1 | MedGemma-1.5-4B-it | 0 / 0 | 0 / 1000 | **0** | None |
| C2 | Gemma-3-4B | 0 / 0 | 0 / 1000 | **0** | None |
| C2 | MedGemma-1.5-4B-it | 0 / 0 | 0 / 1000 | **0** | None |
| C3 | Gemma-3-4B | 0 / 0 | 0 / 500 | **0** | None |
| C3 | MedGemma-1.5-4B-it | 0 / 0 | 0 / 500 | **0** | None |
| C4 | Gemma-3-4B | 1 / 100 | 0 / 40 | **1** | `unans-v2-test-0115` |
| C4 | MedGemma-1.5-4B-it | 0 / 100 | 0 / 40 | **0** | None |
| C5 | Gemma-3-4B | 2 / 100 | 0 / 40 | **2** | `unans-v2-test-0096`, `unans-v2-test-0120` |
| C5 | MedGemma-1.5-4B-it | 0 / 100 | 0 / 40 | **0** | None |

### C4/C5 Accuracy and Abstention Rates (Reported Separately)

Control questions ($N=40$) are standard multiple-choice questions requiring an option letter. Challenge questions ($N=100$) factually have no valid answer and require abstention.
- **Standard (Stored)**: Verbal refusals failed option parsing, fell back to logprob argmax `D`, and were counted as incorrect answer attempts.
- **+ Verbal Abstentions**: Verbal refusals are recognized as intentional abstentions (`pred='ABSTAIN'`). On challenge questions, this counts as a **successful abstention**.

| Condition | Model | Evaluation Mode | Control Correct (N=40) | Control Acc | Control 95% Wilson CI | Challenge Abstained (N=100) | Challenge Abstention Rate | Challenge 95% Wilson CI |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| C4 | Gemma-3-4B | Standard (Stored) | 40 / 40 | 1.0000 | `[0.9124, 1.0000]` | 0 / 100 | 0.0000 | `[0.0000, 0.0370]` |
| C4 | Gemma-3-4B | + Verbal Abstentions | 40 / 40 | 1.0000 | `[0.9124, 1.0000]` | 1 / 100 | 0.0100 | `[0.0018, 0.0545]` |
| C4 | MedGemma-1.5-4B-it | Standard (Stored) | 40 / 40 | 1.0000 | `[0.9124, 1.0000]` | 0 / 100 | 0.0000 | `[0.0000, 0.0370]` |
| C4 | MedGemma-1.5-4B-it | + Verbal Abstentions | 40 / 40 | 1.0000 | `[0.9124, 1.0000]` | 0 / 100 | 0.0000 | `[0.0000, 0.0370]` |
| C5 | Gemma-3-4B | Standard (Stored) | 38 / 40 | 0.9500 | `[0.8350, 0.9862]` | 0 / 100 | 0.0000 | `[0.0000, 0.0370]` |
| C5 | Gemma-3-4B | + Verbal Abstentions | 38 / 40 | 0.9500 | `[0.8350, 0.9862]` | 2 / 100 | 0.0200 | `[0.0055, 0.0700]` |
| C5 | MedGemma-1.5-4B-it | Standard (Stored) | 39 / 40 | 0.9750 | `[0.8712, 0.9956]` | 0 / 100 | 0.0000 | `[0.0000, 0.0370]` |
| C5 | MedGemma-1.5-4B-it | + Verbal Abstentions | 39 / 40 | 0.9750 | `[0.8712, 0.9956]` | 0 / 100 | 0.0000 | `[0.0000, 0.0370]` |

### C4/C5 Confidence & Separation Metrics With vs Without Verbal Abstentions

When verbal abstentions are treated as abstentions, option-answering confidence is set to $0.0$ ($1 - \text{conf} = 1.0$):

| Condition | Model | Mode | Chal Mean Conf | Ctrl Mean Conf | Chal >= 0.90 | AUROC (1 - Conf) | AUROC 95% Bootstrap CI |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| C4 | Gemma-3-4B | Raw (Stored) | 0.9919 | 1.0000 | 0.9700 (97/100) | 0.7650 | `[0.7151, 0.8131]` |
| C4 | Gemma-3-4B | + Verbal (conf=0.0) | 0.9819 | 1.0000 | 0.9600 (96/100) | 0.7650 | `[0.7151, 0.8131]` |
| C4 | MedGemma-1.5-4B-it | Raw (Stored) | 0.8473 | 0.9983 | 0.5900 (59/100) | 0.9617 | `[0.9298, 0.9862]` |
| C4 | MedGemma-1.5-4B-it | + Verbal (conf=0.0) | 0.8473 | 0.9983 | 0.5900 (59/100) | 0.9617 | `[0.9298, 0.9862]` |
| C5 | Gemma-3-4B | Raw (Stored) | 0.9765 | 0.9999 | 0.9400 (94/100) | 0.8325 | `[0.7593, 0.8959]` |
| C5 | Gemma-3-4B | + Verbal (conf=0.0) | 0.9614 | 0.9999 | 0.9400 (94/100) | 0.8325 | `[0.7593, 0.8959]` |
| C5 | MedGemma-1.5-4B-it | Raw (Stored) | 0.8355 | 0.9889 | 0.5500 (55/100) | 0.8982 | `[0.8374, 0.9490]` |
| C5 | MedGemma-1.5-4B-it | + Verbal (conf=0.0) | 0.8355 | 0.9889 | 0.5500 (55/100) | 0.8982 | `[0.8374, 0.9490]` |

## Section 7: Assumptions and Field Specifications

The following assumptions were made regarding the JSONL output schema and field semantics:
1. `id`: Unique string identifier for each question across benchmarks (e.g. `medqa-test-0001`, `unans-v2-test-0001`). Used for 1-to-1 paired alignment.
2. `split`: Dataset partition. Verified to be exclusively `test` for all 2,780 rows per model.
3. `dataset`: Benchmark name (`medqa`, `pubmedqa`, or `unanswerable`).
4. `condition`: Experimental condition (`C1`, `C2`, `C3`, `C4`, `C5`).
5. `should_abstain`: Boolean indicator. `False` for answerable questions (all MedQA, all PubMedQA, and 40 C4/C5 controls); `True` for 100 C4/C5 challenge questions.
6. `category`: Present on C4/C5 rows (`false_premise`, `ambiguous`, `fabricated_disease`, `fabricated_drug`, `out_of_scope`, `missing_info`).
7. `parsed`: Boolean indicating whether regex parsing extracted a valid option letter from `raw_output`.
8. `truncated`: Boolean indicating whether generation reached token limits without terminating cleanly (0 across all rows).
9. `correct`: Stored boolean indicating whether `pred` matched canonical `gold`.
10. `confidence`: Float in `[0.0, 1.0]`, representing `max(letter_probs.values())` at generation step 0 following `Answer:`. Used as a measure of multiple-choice certainty.
11. `letter_probs`: Dictionary mapping option letters $\{A, B, C, D\}$ to their softmax-normalized probabilities at the answer step.
12. `raw_output`: Complete string decoded from model tokens following prompt prefix `Answer:`.

