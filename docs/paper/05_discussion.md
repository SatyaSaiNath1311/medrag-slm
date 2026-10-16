# 5. Discussion

*Note on Discussion Framing: Empirical evaluations for v3 matched pairs are currently awaiting GPU quota execution via `scripts/gpu_queue.md`. Consequently, all substantive interpretations below are structured as conditional hypotheses contingent on final statistical outcomes from `docs/tables/`, marked with `[GPU PENDING]`.*

---

## 5.1 Conditional Interpretations for Matched-Pair Domain Effect (RQ1) `[GPU PENDING]`
By strictly controlling for parameter scale and reasoning token budgets (1,024-token cap for thinking models; greedy direct decoding for non-thinking models), our matched-pair protocol isolates the marginal contribution of medical domain adaptation.

- **If Hypothesis $H_1$ is supported** (exact McNemar test with Holm-adjusted $p < 0.05$ across matched pairs in [`docs/tables/table_matched_pairs_rq1.md`](../tables/table_matched_pairs_rq1.md)):
  - It would indicate that domain-specific pre-training and instruction tuning provide genuine diagnostic accuracy advantages that cannot be replicated solely by scaling prompt tokens or reasoning length within generic models of the same parameter tier.
  - In direct decoding models (`MedGemma-1.5-4B-IT` vs. `Gemma3-4B`), this would be characterized by superior diagnostic terminology recall and standard clinical protocol adherence (`[GPU PENDING]`).
  - In chain-of-thought models (`MedPsy` vs. `Qwen3-Thinking`), this would manifest as higher diagnostic differential precision within the standardized 1,024 reasoning token budget (`[GPU PENDING]`).
- **If Hypothesis $H_1$ is rejected or partially supported** (e.g., non-significant differences or instances where generic controls match or exceed domain models):
  - It would substantiate recent observations in published benchmarks (e.g., MedRGB, EIS 2026 Proceedings) that large-scale general web pre-training imbues modern generic SLMs with sufficient latent biomedical knowledge to compete effectively with small specialized models.

---

## 5.2 Differential RAG Dynamics: Conditional Hypotheses on the Equalizer Effect (RQ2) `[GPU PENDING]`
The interaction between model domain and external retrieval ($\text{Group} \times \text{RAG}$) evaluates whether external evidence provides differential utility to generic versus domain-specialized SLMs.

- **If Hypothesis $H_2$ holds** (statistically significant negative interaction coefficient $\beta_3 < 0, p < 0.05$ in the logistic regression model in [`docs/tables/table_group_cmh_interaction.md`](../tables/table_group_cmh_interaction.md)):
  - It would demonstrate that external retrieval functions as a "knowledge equalizer":
    - **Generic SLMs** would experience larger marginal accuracy gains ($\Delta_{\text{RAG}}$) because high-relevance evidence directly compensates for their sparser parametric medical memory (`[GPU PENDING]`).
    - **Medical SLMs** would exhibit smaller marginal gains due to diminishing returns, as external passages largely reiterate facts already encoded within their specialized weights (`[GPU PENDING]`).
  - In clinical engineering contexts where continuous domain fine-tuning is cost-prohibitive, this outcome would support pairing open-weights generic SLMs with high-precision retrieval as an operationally efficient alternative.
- **If Hypothesis $H_2$ is rejected** ($\beta_3 \ge 0$ or non-significant):
  - It would suggest that domain-specialized SLMs possess superior contextual comprehension, enabling them to assimilate and synthesize retrieved evidence more effectively than generic models.

---

## 5.3 Hypothesized Failure Modes and Abstention on Unanswerable Queries (RQ3) `[GPU PENDING]`
Evaluations on the cryptographically frozen `unanswerable_v2` benchmark ($N=140$) will test competing hypotheses regarding clinical hallucination and abstention reliability (refer to [`docs/tables/table_unanswerable_v3_summary.md`](../tables/table_unanswerable_v3_summary.md) and [`docs/tables/table_unanswerable_v3_categories.md`](../tables/table_unanswerable_v3_categories.md)):

- **Hypothesis $H_3$ Expectations**:
  - Medical models are hypothesized to yield lower confident error rates (confidence $\ge 0.90$) than generic models on plausible unanswerable dilemmas (`[GPU PENDING]`).
  - However, an alternative risk is that domain fine-tuning introduces strong clinical inductive biases, predisposing specialized models to rationalize fictitious entities (e.g., inventing pharmacodynamic mechanisms for fabricated pharmaceuticals) rather than recognizing clinical impossibility.
- **Conformal Calibration Imperative**:
  - Regardless of whether domain models achieve lower raw error rates, uncalibrated small models are expected to lack safe verbalized probabilities.
  - If verified, this will confirm that post-hoc distribution-free conformal abstention thresholds calibrated on validation sets under the clinical safety constraint of False Abstention Rate (FAR) $\le 5\%$ (`[GPU PENDING]`) remain mandatory for safe deployment.

---

## 5.4 Operational Feasibility and Latency-Accuracy Trade-offs (RQ4) `[GPU PENDING]`
Deploying small language models in hospital networks requires balancing diagnostic accuracy against compute and latency constraints:

- **Latency Regimes**:
  - Empirical logs from Kaggle T4 runs demonstrate a pronounced bimodal latency distribution: non-thinking models complete inference in ~0.6 to 3.5 seconds per query, whereas thinking models require ~30 to 65 seconds per query due to reasoning token generation.
- **Trade-off Implications (`[GPU PENDING]`)**:
  - The final accuracy-per-second Pareto frontier (visualized in [`outputs/figures/fig16_accuracy_vs_latency_thinking.png`](../figures/fig16_accuracy_vs_latency_thinking.png)) will establish whether the diagnostic gain from extended reasoning justifies the substantial throughput penalty.
  - If the accuracy delta between thinking and non-thinking models is modest, rapid non-thinking SLMs coupled with RAG would be favored for interactive triage, while thinking models would be reserved for asynchronous diagnostic review.
