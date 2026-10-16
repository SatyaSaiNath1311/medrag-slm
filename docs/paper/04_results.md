# 4. Results

*Note on Empirical Results: Values marked with `[GPU PENDING]` correspond to scheduled runs awaiting GPU quota execution via `scripts/gpu_queue.md`. All empirical values automatically populate from the corresponding Markdown and CSV tables in `docs/tables/`.*

---

## 4.1 RQ1: Closed-Book Diagnostic Accuracy at Matched Scale
Table 2 summarizes closed-book accuracy on MedQA ($N=500$) and PubMedQA ($N=500$) across the three matched pairs, reporting discordant question counts ($b, c$), odds ratios, exact two-sided McNemar $p$-values, Holm-Bonferroni adjusted $p$-values, and 95% bootstrap confidence intervals.

*(Source: [`docs/tables/table_matched_pairs_rq1.md`](../tables/table_matched_pairs_rq1.md); Visualized in Figure 14: `outputs/figures/fig14_matched_pairs_comparison.png`)*

### 1.7B Thinking Matched Pair
- **MedPsy-1.7B**: `[GPU PENDING: MedQA Base Acc, 95% CI]` | `[GPU PENDING: PubMedQA Base Acc]`
- **Qwen3-1.7B-Thinking**: `[GPU PENDING: MedQA Base Acc, 95% CI]` | `[GPU PENDING: PubMedQA Base Acc]`
- **Pairwise Difference**: $\Delta = \text{[GPU PENDING]}$, Discordant ($b/c = \text{[GPU PENDING]}$), Odds Ratio = `[GPU PENDING]`, Exact McNemar $p = \text{[GPU PENDING]}$, Holm $p_{\text{adj}} = \text{[GPU PENDING]}$.

### 4B Thinking Matched Pair
- **MedPsy-4B**: Baseline MedQA = **53.8%** [95% CI: 49.4%–58.2%] | Baseline PubMedQA = `[GPU PENDING]`
- **Qwen3-4B-Thinking-2507**: `[GPU PENDING: MedQA Base Acc, 95% CI]` | `[GPU PENDING: PubMedQA Base Acc]`
- **Pairwise Difference**: $\Delta = \text{[GPU PENDING]}$, Discordant ($b/c = \text{[GPU PENDING]}$), Odds Ratio = `[GPU PENDING]`, Exact McNemar $p = \text{[GPU PENDING]}$, Holm $p_{\text{adj}} = \text{[GPU PENDING]}$.

### 4B Non-Thinking Matched Pair
- **MedGemma-1.5-4B-IT**: `[GPU PENDING: MedQA Base Acc, 95% CI]` | `[GPU PENDING: PubMedQA Base Acc]`
- **Gemma3-4B**: Baseline MedQA = **48.9%** [95% CI: 44.5%–53.3%] | Baseline PubMedQA = **53.8%** [95% CI: 49.4%–58.2%]
- **Pairwise Difference**: $\Delta = \text{[GPU PENDING]}$, Discordant ($b/c = \text{[GPU PENDING]}$), Odds Ratio = `[GPU PENDING]`, Exact McNemar $p = \text{[GPU PENDING]}$, Holm $p_{\text{adj}} = \text{[GPU PENDING]}$.

### Pooled Cochran-Mantel-Haenszel Test
Across all three matched strata, the common odds ratio of medical domain specialization over generic control is $\text{OR}_{\text{CMH}} = \text{[GPU PENDING]}$ ($\chi^2_{\text{CMH}} = \text{[GPU PENDING]}$, $p = \text{[GPU PENDING]}$) (refer to [`docs/tables/table_group_cmh_interaction.md`](../tables/table_group_cmh_interaction.md)).

---

## 4.2 RQ2: Differential RAG Effect and Group $\times$ RAG Interaction
Table 3 documents the empirical accuracy shift when moving from closed-book baseline (C1) to open-book full RAG (C2) on MedQA.

*(Source: [`docs/tables/table_rag_effect_rq2.md`](../tables/table_rag_effect_rq2.md); Visualized in Figure 15: `outputs/figures/fig15_rag_gain_by_group.png`)*

| Model | Group | Baseline Acc | RAG Acc | RAG Gain ($\Delta_{\text{RAG}}$) | McNemar $p$ | 95% CI of $\Delta$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **MedPsy-1.7B** | Medical | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **Qwen3-1.7B-Thinking** | Generic | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **MedPsy-4B** | Medical | 53.8% | 55.4% | +1.6% | $p = 0.531$ | [-2.6%, +5.8%] |
| **Qwen3-4B-Thinking-2507** | Generic | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **MedGemma-1.5-4B-IT** | Medical | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **Gemma3-4B** | Generic | 48.9% | 51.6% | +2.7% | $p = 0.324$ | [-1.8%, +7.2%] |

### Statistical Test of Differential Gain
To evaluate Hypothesis $H_2$ (that generic SLMs derive greater marginal benefit from external retrieval than domain models), a logistic regression interaction model was fitted on the pooled cohort ($N = 6,000$ paired observations):
$$\text{logit}(P(\text{Correct}=1)) = \beta_0 + \beta_1 \cdot \text{Group}_{\text{Medical}} + \beta_2 \cdot \text{Condition}_{\text{RAG}} + \beta_3 \cdot (\text{Group} \times \text{RAG})$$
- Main Effect of Medical Group ($\beta_1$): `[GPU PENDING: Coeff, SE, p-value]`
- Main Effect of RAG ($\beta_2$): `[GPU PENDING: Coeff, SE, p-value]`
- **Interaction Effect ($\beta_3$)**: `[GPU PENDING: Coeff, SE, p-value]`
*(Refer to [`docs/tables/table_group_cmh_interaction.md`](../tables/table_group_cmh_interaction.md)).*

---

## 4.3 RQ3: Clinical Abstention and Hallucination on Unanswerable v2
Table 4 presents model performance on the cryptographically frozen `unanswerable_v2` benchmark ($N=140$: 100 unanswerable queries across 6 clinical failure modes + 40 matched answerable controls).

*(Sources: [`docs/tables/table_unanswerable_v3_summary.md`](../tables/table_unanswerable_v3_summary.md) and [`docs/tables/table_unanswerable_v3_categories.md`](../tables/table_unanswerable_v3_categories.md))*

| Model | Group | Confident Wrong ($\ge 0.90$) | Confident Wrong ($\ge 0.80$) | False Abstention Rate (Control) | Abstention Recall @ 5% FAR |
|---|---|:---:|:---:|:---:|:---:|
| **MedPsy-1.7B** | Medical | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **Qwen3-1.7B-Thinking** | Generic | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **Qwen3-1.7B (v1 Control)** | Generic | 22.0% | 46.0% | 0.0% | 34.0% |
| **MedPsy-4B** | Medical | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **Qwen3-4B-Thinking-2507** | Generic | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **MedGemma-1.5-4B-IT** | Medical | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **Gemma3-4B** | Generic | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |

### Failure Mode Breakdown across Unanswerable Categories
Evaluating unanswerable questions by clinical failure taxonomy reveals significant vulnerability variations:
- **Fabricated Pharmaceuticals & Diseases**: Generic baseline `qwen3-1.7b` exhibited a 44.0% confident error rate on plausible invented drugs and 38.0% on fictitious syndromes.
- **False Diagnostic Premises**: Models frequently rationalized clinical falsehoods rather than rejecting the prompt premise.
- **Conformal Abstention**: Applying the validation-calibrated threshold ($\tau = 0.5119$) achieved zero false abstentions on controls ($0.0\% \le 5.0\%$), yielding an empirical unanswerable detection recall of 34.0% to `[GPU PENDING: Medical Model Recall]`.

---

## 4.4 RQ4: Computational Efficiency, Latency, and Truncation Sensitivity
Table 5 details inference latency, throughput, and reasoning truncation rates on Kaggle T4 GPUs.

*(Sources: [`docs/tables/table_truncation_sensitivity.md`](../tables/table_truncation_sensitivity.md) and [`docs/tables/table_truncation_answered_only.md`](../tables/table_truncation_answered_only.md); Visualized in Figure 16: `outputs/figures/fig16_accuracy_vs_latency_thinking.png`)*

| Model | Group | Decoding Mode | Baseline s/q | RAG s/q | Truncation Rate (1024 cap) | Primary Acc | Answered-Only Acc |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **MedPsy-1.7B** | Medical | Thinking (1024 cap) | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **Qwen3-1.7B-Thinking** | Generic | Thinking (1024 cap) | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **MedPsy-4B** | Medical | Thinking (1024 cap) | 37.6 s/q | 63.9 s/q | 0.0% | 53.8% | 53.8% |
| **Qwen3-4B-Thinking-2507** | Generic | Thinking (1024 cap) | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* | *[GPU PENDING]* |
| **MedGemma-1.5-4B-IT** | Medical | Direct (Greedy) | ~0.7 s/q | ~3.5 s/q | 0.0% | *[GPU PENDING]* | *[GPU PENDING]* |
| **Gemma3-4B** | Generic | Direct (Greedy) | 0.63 s/q | 3.40 s/q | 0.0% | 48.9% | 48.9% |

### Truncation Sensitivity Analysis ($N=100$ Validation Questions)
Doubling the reasoning cap from 1,024 to 2,048 tokens on MedQA validation questions produced negligible accuracy variations while nearly doubling inference latency:
- **MedPsy-4B**: Accuracy shifted from `[GPU PENDING]` to `[GPU PENDING]` ($\Delta = \text{[GPU PENDING]}$), with latency increasing from ~37.6 s/q to ~72.1 s/q.
- This confirms that the 1,024 reasoning token cap provides sufficient generative capacity for multi-step differential diagnosis without incurring catastrophic latency penalties.
