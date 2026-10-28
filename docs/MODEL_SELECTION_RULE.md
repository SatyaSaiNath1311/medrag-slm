# Pre-Registered Deployment Model Selection Rule for MedRAG-SLM Chatbot

**Candidates:** `medpsy-4b` (`qvac/MedPsy-4B`) vs `medgemma-1.5-4b-it` (`google/medgemma-1.5-4b-it`)  
**Comparators (Non-Candidates):** `qwen3-4b-thinking-2507` and `gemma3-4b` (generic controls, not deployment candidates)  
**Date Registered:** 2026-10-28  
**Status:** Pre-registered before execution of the deployment selection script. Frozen.

---

## 1. Background & Purpose

The MedRAG-SLM project develops an interactive clinical decision support application tailored for Community Health Workers (CHWs) and primary care providers, operating within the resource constraints of a free Kaggle GPU accelerator (2x NVIDIA T4).

To select the final primary clinical reasoning engine for deployment in `kaggle_chat/chat_app.py`, we evaluate the two medical domain small language models (SLMs) developed and evaluated in this project:
1. `medpsy-4b`: A 4-billion parameter medical reasoning SLM exhibiting extended step-by-step thinking traces.
2. `medgemma-1.5-4b-it`: A 4-billion parameter medical domain instruction-tuned SLM producing concise, direct clinical answers.

The generic baseline models (`qwen3-4b-thinking-2507` and `gemma3-4b`) serve as generic controls in the research experiments and are excluded from deployment consideration.

### Candidate Scope: `medpsy-1.7b`
`medpsy-1.7b` was evaluated in Pair A on the frozen test set (Conditions C1, C2, C4, C5 under `outputs/v3/medpsy-1.7b/`). However, no committed validation outputs exist for `medpsy-1.7b` on the validation questions evaluated here. Following the hard research protocol rule that deployment decisions must be determined solely on validation data without inspecting test data, `medpsy-1.7b` cannot be evaluated under this offline rule and is excluded.

---

## 2. Evaluation Data & Split

- **Dataset:** MedQA validation split only.
- **Sample:** First 100 questions in file order from `phase1/validation.jsonl` where `dataset == "medqa"`.
- **Condition:** Condition C1 (closed-book medical QA, no external retrieved context), using greedy decoding (`do_sample=False`).
- **Data Provenance:**
  - `medpsy-4b`: Outputs loaded from `outputs/sweeps/budget_pair_b/medpsy-4b.jsonl` (the single-run 4096-token sweep derivation).
  - `medgemma-1.5-4b-it`: Outputs loaded from `outputs/sweeps/gemma_prompt_sensitivity/medgemma-1.5-4b-it.jsonl`, restricted to Variant 1 (`V1`, standard baseline prompt) and the first 100 questions.
- **Shared-ID Hard Assertion:** The analysis script must assert that the 100 question IDs for `medpsy-4b` and `medgemma-1.5-4b-it` are strictly identical in ID set and file ordering (`assert medpsy_ids == medgemma_ids`). If the assertion fails or any input is missing, the script must abort immediately with a non-zero exit code.
- **Safety Assertion:** A hard assertion verifies that every loaded row has `split == "validation"`. Zero test questions are inspected or loaded.

---

## 3. Disclosure

Disclosure: the sweep and prompt-sensitivity results were seen before this rule was written; the thresholds below are set on truncation and parse rate only, and no accuracy comparison enters eligibility.

---

## 4. Eligibility Criteria

A model is **eligible** for deployment if and only if it satisfies all three of the following independent criteria:

1. **Truncation Rate Threshold:**
   $$\text{TruncationRate}(M, c^*) \le 15.0\%$$
   The model's truncation rate at its deployment cap $c^*$ on the 100 validation questions must not exceed 15.0%.

2. **Parse Rate Threshold:**
   $$\text{ParseRate}(M) \ge 95.0\%$$
   The model must format its output reliably, yielding a parse rate of at least 95.0% (at most 5 unparseable responses out of 100 questions).

3. **Deployment Hardware Feasibility:**
   The model must be capable of loading and executing on the target deployment platform of Kaggle 2x NVIDIA T4 GPUs (16 GB VRAM each). This feasibility is confirmed by committed execution manifests and environment logs:
   - `medpsy-4b`: Successfully executed on Kaggle 2x T4 in `outputs/sweeps/budget_pair_b/manifest_medpsy-4b.json` (float16, batch size 1).
   - `medgemma-1.5-4b-it`: Successfully executed on Kaggle 2x T4 in `outputs/sweeps/gemma_prompt_sensitivity/manifest_medgemma-1.5-4b-it.json` (float32, batch size 8).
   - Memory specification: VRAM was not logged in the committed manifests; therefore, per project policy, memory usage is recorded as *"VRAM not measured, to be measured in the live test"* rather than guessed.

If either candidate fails any eligibility criterion, it is disqualified. If both candidates fail, no model is selected and the script exits with an advisory report.

---

## 5. Deployment Cap Specification

### 5.1 Thinking-Style Models (`medpsy-4b`)
For reasoning SLMs that emit internal thinking traces, the deployment cap $c^*$ is chosen from the candidate grid:
$$\mathcal{C} = [1024, 1536, 2048, 3072, 4096]$$

The deployment cap $c^*$ is defined as the **smallest cap** in $\mathcal{C}$ at which `medpsy-4b` achieves a validation truncation rate of $\le 5.0\%$:
$$c^* = \min \{ c \in \mathcal{C} \mid \text{TruncationRate}(\text{medpsy-4b}, c) \le 0.05 \}$$

The choice of $c^*$ is governed strictly by the truncation threshold ($\le 5.0\%$) and is independent of model accuracy.

### 5.2 Direct-Answer Models (`medgemma-1.5-4b-it`)
Direct-answer models do not produce chain-of-thought tokens and use their standard operational decoding cap of `max_new_tokens = 16`.

### 5.3 Truncation Definitions
- **`medpsy-4b` (thinking-style):** A question is finished within cap $c$ iff `finished_naturally` is True and `n_generated_tokens <= c` (counting EOS). Truncation at cap $c$ is defined as not finished within $c$, computed via `is_finished_within_cap(row, c)` from `src/analysis_budget_sweep.py`.
- **`medgemma-1.5-4b-it` (direct-answer):** A response is truncated iff no valid answer option letter could be parsed (`truncated = not parsed`), following the amended convention established in Phase 2 / Task 2 Fix 3.

---

## 6. Winner Selection Rule

Among all eligible candidates:

1. **Primary Metric (Strict Accuracy):**
   The candidate with the highest strict validation accuracy at its deployment cap is selected as the winning model:
   $$\text{Strict Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\text{finished within cap } c \land \text{parsed} \land \text{correct})$$

2. **Tie-Breaking Rule (Latency vs Accuracy):**
   If the top two eligible candidates are **not significantly different in strict accuracy**, defined strictly as:
   - Paired two-sided exact McNemar test $p \ge 0.05$, **AND**
   - Overlapping Wilson 95% score confidence intervals for strict accuracy,
   
   then the tie is broken in favor of **operational latency**: the candidate with the lower measured seconds per question from committed manifests/rows is selected.
   
   If the accuracy difference is statistically significant ($p < 0.05$ or non-overlapping Wilson CIs), the candidate with the higher strict accuracy is selected regardless of latency.

---

## 7. Reporting Requirements

The evaluation report (`docs/MODEL_SELECTION.md`) and machine-readable ledger (`docs/model_selection.json`) must record:
1. Strict accuracy with Wilson 95% score confidence intervals (primary metric).
2. With-fallback accuracy (secondary metric, treating unparsed/truncated responses as incorrect unless raw match).
3. Truncation rate with Wilson 95% score confidence intervals.
4. Parse rate with Wilson 95% score confidence intervals.
5. Measured latency: mean and median seconds per question from row data, and total wall-clock execution time from manifests.
6. Paired model comparison: discordant pair counts ($b, c$), exact two-sided McNemar $p$-value, and paired percentile bootstrap 95% confidence interval (10,000 resamples, fixed seed 42).
7. Plain-words discussion of the accuracy vs latency trade-off.
8. Methodological limitation: multiple-choice examination accuracy on MedQA is an offline benchmark proxy for the quality of interactive, free-text community health worker dialogue and clinical guidance, which lacks reference gold labels.

---

## 8. Amendment Protocol

This document is frozen upon commit prior to the execution of the model selection script. Any modification to this rule after results are viewed is strictly prohibited as an in-place edit. Any post-hoc analysis or protocol change must be recorded below as a dated, signed amendment detailing the justification.

---

### Amendments
*(None. Pre-registered document).*
