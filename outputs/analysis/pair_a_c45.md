# Pair A C4/C5 Analysis: Qwen3-1.7B-Thinking (Generic) vs MedPsy-1.7B (Domain)

**Models Analyzed**:
- Generic Arm: `qwen3-1.7b-thinking`
- Medical/Domain Arm: `medpsy-1.7b`
- Conditions: `C4` (Closed-book unanswerable_v2), `C5` (RAG unanswerable_v2)
- Split: `test` (N=140 per condition: 100 challenge, 40 control)

> [!NOTE]
> Pre-flight checks PASSED: split == 'test' across all rows, 140 unique IDs per model/condition, and exact 1:1 ID match across arms.

## 1. Per-Model & Per-Condition Summary

| Condition | Model | Total N | Parsed Count (%) | Truncated Count (%) | Control Acc (N=40, Wilson 95% CI) | Challenge Truncated (N=100) |
|---|---|---|---|---|---|---|
| C4 | `qwen3-1.7b-thinking` | 140 | 111 (79.3%) | 29 (20.7%) | 40/40 (100.0%, [91.2%, 100.0%]) | 29 (29.0%) |
| C4 | `medpsy-1.7b` | 140 | 118 (84.3%) | 28 (20.0%) | 40/40 (100.0%, [91.2%, 100.0%]) | 23 (23.0%) |
| C5 | `qwen3-1.7b-thinking` | 140 | 79 (56.4%) | 60 (42.9%) | 40/40 (100.0%, [91.2%, 100.0%]) | 56 (56.0%) |
| C5 | `medpsy-1.7b` | 140 | 111 (79.3%) | 45 (32.1%) | 40/40 (100.0%, [91.2%, 100.0%]) | 38 (38.0%) |

## 2. Paired Overall Truncation Comparison (Generic vs Domain)

Paired 2x2 contingency on overall truncation across all 140 question IDs per condition.
**Multiplicity Family**: Overall Truncation Difference ($m=2$: C4, C5).

| Condition | Both Trunc (a) | Qwen Only (b) | MedPsy Only (c) | Neither Trunc (d) | Discordant (b/c) | McNemar Raw p | Holm Adj p (m=2) | Significance |
|---|---|---|---|---|---|---|---|---|
| C4 | 13 | 16 | 15 | 96 | 16/15 | 1.00000 | 1.00000 | ns |
| C5 | 27 | 33 | 18 | 62 | 33/18 | 0.04887 | 0.09775 | ns |

> [!NOTE]
> Under Holm-Bonferroni correction (Family: Overall Truncation, m=2), overall truncation difference between Qwen3-1.7B-thinking and MedPsy-1.7B is not statistically significant in either C4 (adj p = 1.0000) or C5 (adj p = 0.0977).

## 3. Confidence Separation on All Rows (N=140)

Separation between challenge questions (positive class: should abstain, label=1) and control questions (negative class: should answer, label=0).
Score metric: $s = 1 - \text{confidence}$. Ties receive exact half credit ($0.5$). Paired bootstrap 95% CIs use 10,000 question resamples (seed 42).

| Condition | Model | Mean Conf Chal (N=100) | Mean Conf Ctrl (N=40) | Chal Conf >= 0.90 (%) | AUROC (1 - conf) | 95% Bootstrap CI | MedPsy - Qwen Diff | 95% Diff CI |
|---|---|---|---|---|---|---|---|---|
| C4 | `qwen3-1.7b-thinking` | 0.9801 | 1.0000 | 94.0% | 0.6200 | [0.5800, 0.6632] | — | — |
| C4 | `medpsy-1.7b` | 0.9829 | 0.9985 | 94.0% | 0.6819 | [0.5856, 0.7719] | +0.0619 | [-0.0406, +0.1583] |
| C5 | `qwen3-1.7b-thinking` | 0.9641 | 0.9983 | 91.0% | 0.7027 | [0.6398, 0.7618] | — | — |
| C5 | `medpsy-1.7b` | 0.9329 | 0.9989 | 83.0% | 0.6252 | [0.5285, 0.7176] | -0.0775 | [-0.1780, +0.0214] |

## 4. Sensitivity Analysis: Clean Un-Truncated Rows (Both Parsed & Not Truncated)

Confidence on truncated rows originates from a forced argmax answer after token budget exhaustion. Furthermore, models can occasionally emit an answer letter while still hitting the token limit (`parsed == True and truncated == True`). This sensitivity evaluation restricts analysis strictly to question IDs where **BOTH models completed without truncation and were successfully parsed** (`parsed == True and truncated == False`).

### Rows with `parsed == True` and `truncated == True`

| Condition | Model | Total Rows | Parsed True & Truncated True Count (%) | Reason for Truncation |
|---|---|---|---|---|
| C4 | `qwen3-1.7b-thinking` | 140 | 0 (0.0%) | All truncated rows failed parser (0 parsed) |
| C4 | `medpsy-1.7b` | 140 | 6 (4.3%) | Answer extracted before generation hit max_new_tokens |
| C5 | `qwen3-1.7b-thinking` | 140 | 0 (0.0%) | All truncated rows failed parser (0 parsed) |
| C5 | `medpsy-1.7b` | 140 | 16 (11.4%) | Answer extracted before generation hit max_new_tokens |

### Clean Sensitivity Evaluation Table (`parsed == True and truncated == False` for BOTH models)

| Condition | Clean N (Chal / Ctrl) | Model | Mean Conf Chal | Mean Conf Ctrl | Chal Conf >= 0.90 (%) | AUROC (1 - conf) | 95% Bootstrap CI | MedPsy - Qwen Diff | 95% Diff CI |
|---|---|---|---|---|---|---|---|---|---|
| C4 | 96 (61/35) | `qwen3-1.7b-thinking` | 1.0000 | 1.0000 | 100.0% | 0.5164 | [0.5000, 0.5424] | — | — |
| C4 | 96 (61/35) | `medpsy-1.7b` | 0.9962 | 0.9983 | 100.0% | 0.6155 | [0.4980, 0.7258] | +0.0991 | [-0.0232, +0.2147] |
| C5 | 61 (30/31) | `qwen3-1.7b-thinking` | 1.0000 | 1.0000 | 100.0% | 0.5000 | [0.5000, 0.5000] | — | — |
| C5 | 61 (30/31) | `medpsy-1.7b` | 0.9939 | 0.9994 | 100.0% | 0.5048 | [0.3578, 0.6548] | +0.0048 | [-0.1422, +0.1548] |

> [!IMPORTANT]
> **Critical Sensitivity Finding**: When restricted strictly to rows where generation completed without budget exhaustion (`parsed == True and truncated == False` for both models), Qwen3-1.7B-thinking's predicted token probability saturates at 1.0000 on virtually all questions (mean 1.0000 on both challenge and control). Consequently, its genuine separation AUROC drops to chance level (0.5164 in C4, 0.5000 in C5). The higher apparent AUROCs in Section 3 (0.6200 and 0.7027) were driven by forced continuation confidences on truncated questions.

## 5. Budget Exhaustion as an Abstention Signal (Descriptive)

> [!NOTE]
> **Descriptive Interpretation Guard**: Truncation reflects token budget exhaustion (hitting `max_new_tokens` / reasoning cap). While higher truncation on unanswerable items can correlate with the model struggling to resolve the question, truncation also depends heavily on prompt length, question token length, model verbosity, and retrieved context length (particularly in C5). It must be interpreted as a descriptive behavioral artifact rather than a calibrated, intentional abstention mechanism.

### Truncation Rate: Challenge Rows vs Control Rows

| Condition | Model | Challenge Truncation (N=100, Wilson 95% CI) | Control Truncation (N=40, Wilson 95% CI) |
|---|---|---|---|
| C4 | `qwen3-1.7b-thinking` | 29/100 (29.0%, [21.0%, 38.5%]) | 0/40 (0.0%, [0.0%, 8.8%]) |
| C4 | `medpsy-1.7b` | 23/100 (23.0%, [15.8%, 32.2%]) | 5/40 (12.5%, [5.5%, 26.1%]) |
| C5 | `qwen3-1.7b-thinking` | 56/100 (56.0%, [46.2%, 65.3%]) | 4/40 (10.0%, [4.0%, 23.1%]) |
| C5 | `medpsy-1.7b` | 38/100 (38.0%, [29.1%, 47.8%]) | 7/40 (17.5%, [8.7%, 31.9%]) |

### Paired Comparison of Challenge-Row Truncation (100 Challenge IDs)

**Multiplicity Family**: Challenge Truncation Difference ($m=2$: C4, C5).

| Condition | Both Trunc (a) | Qwen Only (b) | MedPsy Only (c) | Neither Trunc (d) | MedPsy - Qwen Diff | McNemar Raw p | Holm Adj p (m=2) | Significance |
|---|---|---|---|---|---|---|---|---|
| C4 | 13 | 16 | 10 | 61 | -6% | 0.32694 | 0.32694 | ns |
| C5 | 25 | 31 | 13 | 31 | -18% | 0.00956 | 0.01912 | p < 0.05 |

> [!NOTE]
> In C4 closed-book, challenge truncation difference is small and not significant (-6%, adj p = 0.3269). In C5 RAG, Qwen3-1.7B-thinking truncates on significantly more challenge items than MedPsy-1.7B (56% vs 38%, diff = -18%, adj p = 0.0191), reflecting higher context verbosity consumption in the thinking model.

## 6. Abstention Counts (Explicit & Verbal)

Counts of explicit abstentions (`abstained == True`, corresponding to `pred == 'ABSTAIN'`) and verbal refusal phrases in `raw_output` after `'Answer:'` starting with `None`, `Insufficient`, `Not enough`, or `Cannot` (case-insensitive).

| Condition | Model | Total Rows | Explicit Abstentions (`abstained == True`) | Verbal Refusal after Answer: (`None/Insufficient/...`) | Challenge Abstentions | Control Abstentions |
|---|---|---|---|---|---|---|
| C4 | `qwen3-1.7b-thinking` | 140 | 0 (0.0%) | 0 (0.0%) | 0/100 (0.0%) | 0/40 (0.0%) |
| C4 | `medpsy-1.7b` | 140 | 0 (0.0%) | 0 (0.0%) | 0/100 (0.0%) | 0/40 (0.0%) |
| C5 | `qwen3-1.7b-thinking` | 140 | 0 (0.0%) | 0 (0.0%) | 0/100 (0.0%) | 0/40 (0.0%) |
| C5 | `medpsy-1.7b` | 140 | 0 (0.0%) | 0 (0.0%) | 0/100 (0.0%) | 0/40 (0.0%) |

> [!NOTE]
> Both models recorded **0 explicit abstentions** (`abstained == True`) and **0 verbal refusals** after `'Answer:'` across all 140 questions in both C4 and C5. Neither model exhibits spontaneous or instruction-prompted abstention; when confronted with unanswerable questions, both models either run out of generation tokens (truncation) or output an answer choice with near-maximal confidence.

## 7. Per-Category Breakdown for Challenge Rows

> [!WARNING]
> **Sample Size Guard**: In unanswerable_v2 test split, each individual unanswerability category contains fewer than 20 rows (16 or 17 rows each). Per the evaluation protocol, categories with fewer than 20 rows are listed below for descriptive completeness only and **must not be given definitive substantive interpretation**.

### Condition C4

| Category | N Rows | Interpret? | Qwen Parsed (%) | Qwen Trunc (%) | Qwen Mean Conf | MedPsy Parsed (%) | MedPsy Trunc (%) | MedPsy Mean Conf |
|---|---|---|---|---|---|---|---|---|
| `ambiguous` | 17 | No (N < 20) | 16/17 (94.1%) | 1/17 (5.9%) | 0.9971 | 15/17 (88.2%) | 2/17 (11.8%) | 0.9761 |
| `fabricated_disease` | 17 | No (N < 20) | 11/17 (64.7%) | 6/17 (35.3%) | 0.9704 | 12/17 (70.6%) | 5/17 (29.4%) | 0.9924 |
| `fabricated_drug` | 17 | No (N < 20) | 15/17 (88.2%) | 2/17 (11.8%) | 0.9836 | 17/17 (100.0%) | 0/17 (0.0%) | 0.9958 |
| `false_premise` | 17 | No (N < 20) | 4/17 (23.5%) | 13/17 (76.5%) | 0.9678 | 6/17 (35.3%) | 11/17 (64.7%) | 0.9509 |
| `missing_info` | 16 | No (N < 20) | 13/16 (81.2%) | 3/16 (18.8%) | 0.9621 | 13/16 (81.2%) | 4/16 (25.0%) | 0.9962 |
| `out_of_scope` | 16 | No (N < 20) | 12/16 (75.0%) | 4/16 (25.0%) | 0.9999 | 15/16 (93.8%) | 1/16 (6.2%) | 0.9872 |

### Condition C5

| Category | N Rows | Interpret? | Qwen Parsed (%) | Qwen Trunc (%) | Qwen Mean Conf | MedPsy Parsed (%) | MedPsy Trunc (%) | MedPsy Mean Conf |
|---|---|---|---|---|---|---|---|---|
| `ambiguous` | 17 | No (N < 20) | 8/17 (47.1%) | 9/17 (52.9%) | 0.9542 | 9/17 (52.9%) | 8/17 (47.1%) | 0.8755 |
| `fabricated_disease` | 17 | No (N < 20) | 13/17 (76.5%) | 4/17 (23.5%) | 0.9838 | 14/17 (82.4%) | 5/17 (29.4%) | 0.9671 |
| `fabricated_drug` | 17 | No (N < 20) | 9/17 (52.9%) | 8/17 (47.1%) | 0.9781 | 16/17 (94.1%) | 2/17 (11.8%) | 0.9702 |
| `false_premise` | 17 | No (N < 20) | 1/17 (5.9%) | 16/17 (94.1%) | 0.9356 | 9/17 (52.9%) | 12/17 (70.6%) | 0.8757 |
| `missing_info` | 16 | No (N < 20) | 6/16 (37.5%) | 10/16 (62.5%) | 0.9900 | 11/16 (68.8%) | 8/16 (50.0%) | 0.9125 |
| `out_of_scope` | 16 | No (N < 20) | 6/16 (37.5%) | 9/16 (56.2%) | 0.9435 | 16/16 (100.0%) | 3/16 (18.8%) | 0.9991 |

## 8. Assumptions & Field Specifications

1. `split`: String identifier strictly equal to `'test'` across all rows.
2. `id`: String question identifier unique within each model and identical across both arms.
3. `should_abstain`: Ground-truth boolean indicating challenge questions (`True`, N=100) vs control questions (`False`, N=40).
4. `control`: Boolean flag exactly equal to `not should_abstain` across all rows.
5. `parsed`: Boolean indicating whether model's final answer letter was successfully extracted via regex/parser without fallback.
6. `truncated`: Boolean indicating whether model generated up to `max_new_tokens` (or reasoning cap) without emitting an end-of-sequence / final tag.
7. `correct`: Boolean indicating agreement between predicted letter and canonical ground-truth gold answer.
8. `confidence`: Float extracted from softmax letter probabilities for the predicted token; when truncated, comes from prompt-forced continuation.
9. `category`: Unanswerability classification string (`ambiguous`, `fabricated_disease`, `fabricated_drug`, `false_premise`, `missing_info`, `out_of_scope`).
10. `abstained`: Boolean indicating whether model emitted `ABSTAIN` as its predicted option.
11. `raw_output`: String containing model's raw generated text, inspected for verbal refusal phrases following `'Answer:'`.
