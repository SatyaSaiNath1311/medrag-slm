# Pre-Registered Token-Budget Decision Rule for Pair B
**Models:** `qwen3-4b-thinking-2507` (generic reasoning control) vs `medpsy-4b` (medical reasoning SLM)  
**Date Registered:** 2026-10-27  
**Status:** Pre-registered before execution of the validation token-budget sweep. Frozen.

---

## 1. Background & Rationale
In the frozen test evaluation of Pair B under Condition C1 (closed-book MedQA), `qwen3-4b-thinking-2507` truncated at a rate of ~99.6% when restricted to the default 1,024-token budget cap. As a result, the existing comparison at cap=1024 is heavily budget-limited for the generic reasoning model rather than reflective of intrinsic domain reasoning performance.

To determine an appropriate, fair token cap for the test set, we conduct a token-budget sweep on the **validation split only**. To eliminate confirmation bias and prevent data peeking, the decision rule for fixing the test cap is specified and committed here **prior to executing the sweep**.

Disclosure: the motivation for this sweep comes from the earlier test-set observation of ~99.6% truncation for qwen3-4b-thinking-2507 at the 1,024-token cap. The test cap itself is selected solely from validation data by the rule below; no test accuracy enters the rule.

---

## 2. Experimental Sweep Protocol

### 2.1 Dataset & Split
- **Split:** MedQA validation split only.
- **Source:** First 100 questions in file order from `phase1/validation.jsonl` where `dataset == "medqa"`. (The file is randomly shuffled and not ordered by topic, specialty, or difficulty).
- **Condition:** Condition C1 (closed-book multi-choice QA).
- **Prompt:** Identical C1 prompt format reused across both models via existing prompt builders.
- **Safety Assertion:** A hard assertion verifies that every loaded row has `split == "validation"`. Zero test questions are loaded, inspected, or evaluated.

### 2.2 Decoding & Prefix-Consistency
- **Decoding:** Strictly greedy decoding (`do_sample=False`).
- **Single-Run Derivation:** Under deterministic greedy autoregressive decoding, token generation at step $t$ is fully invariant to future steps. Therefore, generating up to $M = 4096$ tokens produces the exact token prefix sequence that would have been generated at any smaller cap $c \in \{1024, 1536, 2048, 3072\}$. All caps in the grid are derived from the single 4096-token run, so the five caps are mutually consistent by construction. A separate run at a smaller cap could differ in rare cases because of floating-point nondeterminism; this is not tested.

### 2.3 Evaluated Cap Grid
The sweep evaluates five candidate caps:
$$\mathcal{C} = [1024, 1536, 2048, 3072, 4096]$$

### 2.4 Definitions
A question is finished within cap c iff finished_naturally is true and n_generated_tokens <= c (the EOS token is counted). The truncation rate at cap c is the fraction of the 100 questions not finished within c. The decision rule uses this point estimate. A Wilson 95% CI is reported alongside it, with the note that N=100 gives a wide interval near the 20% threshold; the CI does not change the decision.

---

## 3. Pre-Registered Decision Rule for Test Cap Selection

1. **Qualification Criterion:**
   The candidate test cap $c^*$ is the **smallest cap** in $\mathcal{C}$ at which `qwen3-4b-thinking-2507` achieves a validation truncation rate of $\le 20.0\%$:
   $$c^* = \min \{ c \in \mathcal{C} \mid \text{TruncationRate}(\text{qwen3-4b-thinking-2507}, c) \le 0.20 \}$$

2. **Fallback Rule:**
   If no cap in $\mathcal{C}$ achieves a truncation rate $\le 20.0\%$, the maximum cap $c^* = 4096$ shall be selected. In this event, the residual truncation rate at 4096 tokens must be reported explicitly as a methodological limitation, not as parity.

3. **Equal Budget Application:**
   The chosen cap $c^*$ applies identically to both models in Pair B (`qwen3-4b-thinking-2507` and `medpsy-4b`) to guarantee a matched, equal-budget comparison.

4. **Independence from Accuracy:**
   The selection of $c^*$ is governed strictly by the truncation threshold ($\le 20\%$). No cap is chosen, promoted, or rejected by observing model accuracy.

5. **Complete Reporting:**
   All five caps in $\mathcal{C}$ will be reported in full in the sweep analysis report and figures, regardless of which cap is chosen by the rule.

6. **Scope:**
   $c^*$ applies to both models in the Pair B MedQA test runs, closed-book and RAG conditions alike. Truncation in the RAG condition is reported separately and does not alter $c^*$.

7. **Compute Feasibility:**
   $c^*$ is never lowered for compute reasons. If the test run at $c^*$ does not fit one Kaggle quota window it is split into sessions with the same cap and the same fixed question order. If it cannot be completed before the paper deadline, the Pair B comparison is reported at the existing 1,024-token cap with the truncation limitation stated, and $c^*$ is reported as a validation finding.

---

## 4. Amendment Protocol
This document is frozen upon commit prior to the run. Any modification to this rule after results are viewed is strictly prohibited as an in-place edit. Any post-hoc analysis or protocol change must be recorded below as a dated, signed amendment detailing the justification.

---

### Amendments
*(None. Pre-registered document).*
