# Table 2: Reliability, Abstention, and Grounding across Architectures and Inference Modes (V3 Study)

Evaluation of selective prediction, token exhaustion (truncation), and confidence separation between unanswerable challenge questions and answerable control questions across **C4 closed-book** and **C5 retrieval**.

| Model | Mode | Condition | N chal / ctrl | Chal trunc % | Ctrl trunc % | Mean conf chal | Mean conf ctrl | AUROC (1-conf) [95% CI] | Explicit abstention count | Verbal refusal count |
|---|---|---|---|---|---|---|---|---|---|---|
| `qvac/MedPsy-1.7B` | thinking (reasoning=true) | C4 (closed-book unanswerable) | 100 / 40 | 23.0% | 12.5% | 0.9829 | 0.9985 | 0.6819 [0.5856, 0.7719] | 0 | 0 |
| `Qwen/Qwen3-1.7B` | thinking (reasoning=true) | C4 (closed-book unanswerable) | 100 / 40 | 29.0% | 0.0% | 0.9801 | 1.0000 | 0.6200 [0.5800, 0.6632] | 0 | 0 |
| `google/gemma-3-4b-it` | direct answer | C4 (closed-book unanswerable) | 100 / 40 | 0.0% | 0.0% | 0.9919 | 1.0000 | 0.7650 [0.7151, 0.8131] | 0 | 1 |
| `google/medgemma-1.5-4b-it` | direct answer | C4 (closed-book unanswerable) | 100 / 40 | 0.0% | 0.0% | 0.8473 | 0.9983 | 0.9617 [0.9298, 0.9862] | 0 | 0 |
| `qvac/MedPsy-1.7B` | thinking (reasoning=true) | C5 (retrieval unanswerable) | 100 / 40 | 38.0% | 17.5% | 0.9329 | 0.9989 | 0.6252 [0.5285, 0.7176] | 0 | 0 |
| `Qwen/Qwen3-1.7B` | thinking (reasoning=true) | C5 (retrieval unanswerable) | 100 / 40 | 56.0% | 10.0% | 0.9641 | 0.9983 | 0.7027 [0.6398, 0.7618] | 0 | 0 |
| `google/gemma-3-4b-it` | direct answer | C5 (retrieval unanswerable) | 100 / 40 | 0.0% | 0.0% | 0.9765 | 0.9999 | 0.8325 [0.7593, 0.8959] | 0 | 2 |
| `google/medgemma-1.5-4b-it` | direct answer | C5 (retrieval unanswerable) | 100 / 40 | 0.0% | 0.0% | 0.8355 | 0.9889 | 0.8982 [0.8374, 0.9490] | 0 | 0 |
| `qvac/MedPsy-4B` | thinking (reasoning=true) | Phase 5 (closed-book unanswerable) | 150 / 0 | 46.7% | — | 0.9552 | — | — | 0 | 0 |

## Grounding (MedPsy-4B only)

| Model | Scorable Answers (N/500) | Total Cited Passages | Citation Precision (NLI Upper Bound) | Citation Precision (Corrected Est) [Range] | Answer-Level Support (NLI) |
|---|---|---|---|---|---|
| `qvac/MedPsy-4B` | 379 / 500 (75.8%) | 677 | 311 / 677 (45.9%) | 26.1% [15.8%, 42.7%] | 201 / 379 (53.0%) |
| `qvac/MedPsy-1.7B` | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |
| `Qwen/Qwen3-1.7B` | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |
| `google/gemma-3-4b-it` | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |
| `google/medgemma-1.5-4b-it` | not evaluated | not evaluated | not evaluated | not evaluated | not evaluated |

> [!NOTE]
> The 150-row MedPsy-4B unanswerable set and the 140-row (100 challenge + 40 control) unanswerable_v2 set are different pools and are not directly comparable.
