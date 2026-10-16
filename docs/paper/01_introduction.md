# 1. Introduction

## 1.1 Context and Clinical Motivation
Deploying artificial intelligence in clinical environments requires balancing diagnostic accuracy, factual reliability, and data privacy. While frontier large language models (LLMs) exceeding 70 billion parameters demonstrate impressive medical examination performance, their massive computational footprints preclude on-premises deployment in hospital networks and edge clinical hardware. Under strict privacy regulations (e.g., HIPAA, GDPR), patient-identifiable clinical notes and proprietary institutional guidelines cannot be routed to third-party cloud APIs.

Consequently, Small Language Models (SLMs) containing 1.5 to 4 billion parameters have emerged as viable candidates for local, privacy-preserving clinical decision support. However, deploying SLMs introduces severe technical trade-offs: smaller parametric capacities increase the risk of factual hallucination, reduce reasoning depth, and limit broad biomedical knowledge retention.

Retrieval-Augmented Generation (RAG) provides an established framework to compensate for parametric memory limits by grounding model generation in verified medical literature (e.g., peer-reviewed textbooks and PubMed abstracts). Yet, an open question remains regarding the interaction between domain-specific medical pre-training/fine-tuning and external retrieval: **Does domain specialization render external retrieval redundant, or do domain-adapted SLMs benefit synergistically from structured clinical evidence?**

## 1.2 Confounding in Prior Literature
A critical methodological deficiency in recent literature is the confounding of three distinct variables:
1. **Model scale** (e.g., comparing a 1.7B generic model to an 8B domain model);
2. **Inference reasoning paradigm** (e.g., comparing a direct greedy-decoding generic model against a chain-of-thought "thinking" domain model generating thousands of intermediate reasoning tokens);
3. **Domain adaptation** (pre-training on biomedical corpora vs. general web pre-training).

When domain-specific models demonstrate superior accuracy in unstandardized benchmarks, it is frequently unclear whether performance gains arise from domain-specific clinical knowledge, extended reasoning token budgets, or scale disparities. Furthermore, negative results where generic models match or exceed domain models under high-recall retrieval are rarely systematically tested across matched architectures.

## 1.3 Research Questions and Hypotheses
To rigorously disentangle these factors, this study establishes a 2 $\times$ 3 factorial matched-pair design across three parameter-matched pairs ($\le 4\text{B}$):
- **Pair 1 (1.7B Thinking)**: MedPsy-1.7B vs. Qwen3-1.7B-Thinking (reasoning token cap: 1,024).
- **Pair 2 (4B Thinking)**: MedPsy-4B vs. Qwen3-4B-Thinking-2507 (reasoning token cap: 1,024).
- **Pair 3 (4B Non-Thinking)**: MedGemma-1.5-4B-IT vs. Gemma3-4B (direct decoding).

We formalize four research questions (RQs) and register directional hypotheses in advance:

* **RQ1 (Closed-Book Domain Effect)**: *Does medical-domain adaptation improve closed-book diagnostic accuracy at matched parameter size and reasoning mode?*
  - **Hypothesis $H_{1}$**: Medical-domain SLMs will significantly outperform their matched generic counterparts in parametric baseline settings (exact McNemar test, $\alpha=0.05$, Holm-Bonferroni corrected).
* **RQ2 (Differential RAG Gain)**: *Does external evidence retrieval benefit generic and medical SLMs equally, or do generic SLMs experience a larger accuracy gain?*
  - **Hypothesis $H_{2}$**: The marginal accuracy gain ($\Delta_{\text{RAG}} = \text{Acc}_{\text{RAG}} - \text{Acc}_{\text{Base}}$) will be significantly larger for generic SLMs than domain-specialized SLMs (logistic regression group $\times$ RAG interaction test, $p < 0.05$).
* **RQ3 (Clinical Abstention and Hallucination)**: *Do medical-domain SLMs exhibit lower rates of confident incorrect answers and more reliable abstention on unanswerable clinical queries?*
  - **Hypothesis $H_{3}$**: Medical models will produce fewer confident fabricated answers (confidence $\ge 0.90$ on unanswerable vignettes), but both groups will require calibrated conformal abstention thresholds to satisfy the clinical safety constraint of False Abstention Rate (FAR) $\le 5\%$.
* **RQ4 (Computational Efficiency & Truncation Costs)**: *What computational penalties in inference latency, peak memory, and context truncation accompany domain-specialized reasoning?*
  - **Hypothesis $H_{4}$**: Extended reasoning models will exhibit 10$\times$ to 50$\times$ higher latency per question, creating critical trade-offs between diagnostic accuracy and throughput.

## 1.4 Primary Contributions
1. **Factorial Matched-Pair Framework**: We introduce a controlled benchmark protocol pairing generic and medical SLMs sharing base architectures, token budgets, and prompts, preventing inference-mode and scale confounding.
2. **Empirical RAG Interaction Analysis**: We evaluate 5 standardized conditions ($N=2,140$ questions per model) spanning MedQA (USMLE), PubMedQA, and a 6-category unanswerable benchmark (`unanswerable_v2`), demonstrating how retrieval alters the performance gap between domain and generic SLMs.
3. **Rigorous Statistical Verification**: We conduct exact two-sided McNemar tests, Holm-Bonferroni multiplicity corrections, bootstrap confidence intervals ($B=1,000$), Cochran-Mantel-Haenszel tests, and interaction modeling.
4. **Clinical Safety & Efficiency Trade-offs**: We analyze confident hallucination rates under conformal prediction and quantify the latency and memory overhead of thinking-model reasoning tokens in resource-constrained deployment.
