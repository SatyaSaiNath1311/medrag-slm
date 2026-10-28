# Pre-Registered Prompt-Sensitivity Protocol & Stability Decision Rule for Gemma Pair
**Models:** `gemma3-4b` (`google/gemma-3-4b-it`, generic baseline) vs `medgemma-1.5-4b-it` (`google/medgemma-1.5-4b-it`, medical domain SLM)  
**Date Registered:** 2026-10-28  
**Status:** Pre-registered before execution of the validation prompt-sensitivity run. Frozen.

---

## 1. Background & Rationale
In the frozen test evaluation of Pair C (Gemma-3-4B vs MedGemma-1.5-4B-it) under Condition C1 (closed-book MedQA, $N=500$), MedGemma achieved 54.80% strict accuracy versus Gemma-3's 48.60%, yielding a +6.20 percentage point gap favoring the medical model ($p = 0.0255$ raw McNemar, $p = 0.1274$ Holm-adjusted).

Because instruction-tuned LLMs may exhibit sensitivity to prompt wording, whitespace, and option formatting, we investigate whether this generic-vs-medical performance gap is robust or sensitive to prompt variation. To protect the test set from contamination and avoid p-hacking, this evaluation is performed **strictly on the validation split** under a pre-registered decision rule committed **prior to running any inferences**.

---

## 2. Experimental Protocol

### 2.1 Dataset & Split
- **Split:** MedQA validation split only.
- **Sample:** First 200 questions in file order from `phase1/validation.jsonl` where `dataset == "medqa"`.
- **Sample Randomization:** During Phase 1 dataset preparation, the MedQA training pool was randomly shuffled with seed 42 before the 200 validation questions were extracted. Thus, the file order is already a balanced, seeded random sample.
- **Safety Assertion:** A hard assertion verifies that every loaded row has `split == "validation"`. Zero test questions are loaded or inspected.

### 2.2 Condition & Decoding
- **Condition:** Condition C1 (closed-book multi-choice QA, no external retrieved context).
- **Decoding Configuration:** Strictly greedy decoding (`do_sample=False`), identical to the main C1 experimental evaluation:
  - `max_new_tokens = 16`
  - `batch_size = 8` (with recursive OOM halving via `generate_safe`)
  - `dtype = torch.float32` (both models use Gemma3 loader on dual-GPU T4)
  - `assistant_prefix = "Answer:"` (enforces direct letter generation)

### 2.3 Prompt Variants
Both models receive identical prompts for each variant. The answer-format rules are identical across all three variants: the model is instructed to output the letter corresponding to the correct option, and generation begins after the pre-filled `Answer:` prefix.

#### Variant 1 (V1 — Baseline v3 Prompt, Byte-Identical)
Imported directly from `src.llm.baseline_prompt(q)`:
```
You are a medical expert answering a multiple-choice question.

Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Answer with the letter of the correct option.
```

#### Variant 2 (V2 — Clinical Evaluation Phrasing with Parenthesized Options)
```
The following is a multiple-choice medical examination question. Please evaluate the clinical scenario and select the single best option.

Question: {question}
Options:
(A) {option_A}
(B) {option_B}
(C) {option_C}
(D) {option_D}

Provide the letter corresponding to the correct answer.
```

#### Variant 3 (V3 — Board-Style Presentation with Bracketed Choices)
```
Carefully read the patient presentation below and identify the correct diagnostic or management step.

Question: {question}
Choices:
[A] {option_A}
[B] {option_B}
[C] {option_C}
[D] {option_D}

State the letter of the single correct option.
```

---

## 3. Evaluation Outcomes & Definitions

1. **Strict Accuracy:**
   A response is strictly correct iff it finished naturally (not truncated), was parsed into a valid option letter, and matches the gold answer. Unparsed or truncated responses receive score 0.0.
   $$\text{StrictAcc} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\text{finished\_naturally}_i \land \text{parsed}_i \land \text{correct}_i)$$

2. **Truncation Rate:**
   Fraction of the $N=200$ questions where generation exhausted the 16-token budget without emitting an EOS token (`truncated == True`).

3. **Parse Rate:**
   Fraction of responses from which an option letter was successfully extracted (`parsed == True`).

---

## 4. Pre-Registered Stability Criterion

The paired generic-vs-medical performance gap for variant $V \in \{V_1, V_2, V_3\}$ is defined as:
$$\text{Gap}(V) = \text{StrictAcc}(\text{gemma3-4b}, V) - \text{StrictAcc}(\text{medgemma-1.5-4b-it}, V)$$

### Stability Rule:
The gap is classified as **STABLE** if and only if **both** of the following conditions hold:
1. **Sign Consistency:** The sign of the gap is identical across all three variants:
   $$\text{sign}(\text{Gap}(V_1)) = \text{sign}(\text{Gap}(V_2)) = \text{sign}(\text{Gap}(V_3)) \neq 0$$
   (i.e., the same model wins under all three prompts).
2. **Bounded Spread:** The spread between the maximum and minimum gap across variants does not exceed 10.0 percentage points:
   $$\max_{V \in \{V_1, V_2, V_3\}} \text{Gap}(V) - \min_{V \in \{V_1, V_2, V_3\}} \text{Gap}(V) \le 0.10$$

### Sensitivity Rule:
If the sign flips across any variant, or if the spread exceeds 10.0 percentage points ($> 0.10$), the gap is classified and reported as **PROMPT-SENSITIVE**.

Both outcomes will be reported honestly in the analysis report and final publication tables without post-hoc rationalization.

### 4.1 Format-Confound Flag
A variant is flagged FORMAT-CONFOUNDED if the parse rates of the two models differ by more than 5.0 percentage points in that variant. For flagged variants the report additionally shows the gap computed only on questions where both models produced a parsed answer, labelled DESCRIPTIVE, selected subset. The stability verdict in Section 4 is unchanged by this flag; the report states which variants were flagged next to the verdict.

---

## 5. Statistical Analysis Plan

1. **Confidence Intervals:** Wilson 95% score intervals for strict accuracy and truncation rate for each model $\times$ variant cell ($2 \times 3 = 6$ cells).
2. **Paired Tests:** Exact two-sided McNemar test on discordant pairs ($b, c$) for each variant.
3. **Paired Differences:** Percentile bootstrap 95% confidence intervals for the paired accuracy difference $\text{Gap}(V)$ (10,000 resamples, fixed seed 42).
4. **Multiple Testing Correction:** A single Holm-Bonferroni family applied across the three per-variant strict accuracy differences ($m=3$). Significance is declared only where Holm-adjusted $p < 0.05$.
5. **Spread Table:** Model-level accuracy spread ($\max - \min$) across variants for each model, alongside the gap spread.

---

## 6. Scope Statement
This check is conducted solely on the validation set ($N=200$) as a diagnostic robustness audit. It does not modify, replace, or invalidate the primary frozen test set results, which remain as previously reported.

---

## 7. Amendment Protocol
This document is frozen upon commit prior to running the sensitivity check. In-place edits after results are generated are prohibited. Any subsequent changes must be logged below as dated amendments detailing rationale and date.

---

### Amendments
*(None. Pre-registered document).*

#### Amendment 1 (2026-10-28)
- **Context & Finding**: Sections 2.2 and 3 initially defined truncation as "no EOS within 16 tokens" (`finished_naturally == False` implies `truncated == True`). During an initial $N=3$ smoke run on Kaggle (kernel v63), all 9 generated responses (3 questions $\times$ 3 variants $\times$ 2 models) produced `truncated=True` and `finished_naturally=False` despite `parsed=True` and correct answers. With `max_new_tokens=16` and the `"Answer:"` prefill, both models emit the answer letter on token 1 and continue writing explanatory text until hitting the 16-token cap.
- **Convention Conflict**: In the repo's established C1 direct-answer baseline outputs (`outputs/kaggle_qa/...` and `outputs/v3/gemma3-4b/C1/gemma3-4b.jsonl`), all rows have `truncated=0` / `truncated=False`. The established convention for this 16-token direct-answer task is that a response is considered "truncated" only if no answer letter could be parsed. Under the initial definition, strict accuracy would be 0% for both models across every variant.
- **Redefinition of Truncation**: Truncation for this experiment is formally redefined to match the repo's established C1 convention: a response is considered truncated (`truncated == True`) if and only if no parseable answer letter was produced (`truncated = not parsed`). Under this definition, strict correctness remains $\mathbf{1}(\neg\text{truncated} \land \text{parsed} \land \text{correct})$, which correctly evaluates to 1 when a valid answer letter is parsed and matches the gold label.
- **Diagnostic Logging**: The raw generation indicators `finished_naturally` (whether EOS was emitted before token 16) and `hit_cap_without_eos` (`not finished_naturally`) are preserved and logged in every row as diagnostics only; they do not penalize strict accuracy.
- **Integrity**: This amendment was adopted after diagnosing the $N=3$ smoke run protocol defect and before any full-$N$ ($N=200$) data existed. No evaluation result was consulted.

