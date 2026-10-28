# Table 1: Question-Answering Performance across Architectures and Inference Modes (V3 Study)

Comprehensive empirical evaluation comparing matched generic and domain-adapted small language models (1.7B and 4B parameters) across **C1 closed-book**, **C2 retrieval**, and **C3 oracle abstract**.

| Pair | Model | Mode | Condition | Dataset | Split | N | Token Cap | Strict Accuracy [95% CI] (Primary) | With-Fallback Accuracy (Secondary) | Macro-F1 | Truncation % | Parse % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pair A | `qvac/MedPsy-1.7B` | thinking (reasoning=true) | C1 (closed-book) | MedQA | test | 500 | 1024 | 60.2% [55.8%, 64.4%] | 77.8% | 0.777 | 32.0% | 70.6% |
| Pair A | `Qwen/Qwen3-1.7B` | thinking (reasoning=true) | C1 (closed-book) | MedQA | test | 500 | 1024 | 30.4% [26.5%, 34.6%] | 49.8% | 0.496 | 51.0% | 50.2% |
| Pair A | `qvac/MedPsy-1.7B` | thinking (reasoning=true) | C2 (retrieval) | MedQA | test | 500 | 1024 | 45.2% [40.9%, 49.6%] | 78.4% | 0.783 | 46.6% | 71.0% |
| Pair A | `Qwen/Qwen3-1.7B` | thinking (reasoning=true) | C2 (retrieval) | MedQA | test | 500 | 1024 | 33.2% [29.2%, 37.4%] | 53.2% | 0.533 | 49.4% | 51.4% |
| Pair B | `qvac/MedPsy-4B` | thinking (reasoning=true) | C1 (closed-book) | MedQA | test | 500 | 1024 | 83.2% [79.7%, 86.2%] | 87.6% | 0.875 | 10.6% | 90.4% |
| Pair B | `Qwen/Qwen3-4B-Thinking-2507` | thinking (reasoning=true) | C1 (closed-book) | MedQA | test | 500 | 1024 | 0.4% [0.1%, 1.4%] | 66.8% | 0.668 | 99.6% | 2.0% |
| Pair B | `qvac/MedPsy-4B` | thinking (reasoning=true) | C2 (retrieval) | MedQA | test | 500 | 1024 | 72.6% [68.5%, 76.3%] | 89.0% | 0.890 | 21.6% | 83.8% |
| Pair B | `Qwen/Qwen3-4B-Thinking-2507` | thinking (reasoning=true) | C2 (retrieval) | MedQA | test | 500 | 1024 | 0.0% [0.0%, 0.8%] | 60.4% | 0.605 | 100.0% | 0.6% |
| Gemma Pair | `google/gemma-3-4b-it` | direct answer | C1 (closed-book) | MedQA | test | 500 | 16 | 48.6% [44.2%, 53.0%] | 48.6% | 0.483 | 0.0% | 100.0% |
| Gemma Pair | `google/medgemma-1.5-4b-it` | direct answer | C1 (closed-book) | MedQA | test | 500 | 16 | 54.8% [50.4%, 59.1%] | 54.8% | 0.548 | 0.0% | 100.0% |
| Gemma Pair | `google/gemma-3-4b-it` | direct answer | C1 (closed-book) | PubMedQA | test | 500 | 16 | 49.2% [44.8%, 53.6%] | 49.2% | 0.369 | 0.0% | 100.0% |
| Gemma Pair | `google/medgemma-1.5-4b-it` | direct answer | C1 (closed-book) | PubMedQA | test | 500 | 16 | 50.8% [46.4%, 55.2%] | 50.8% | 0.383 | 0.0% | 100.0% |
| Gemma Pair | `google/gemma-3-4b-it` | direct answer | C2 (retrieval) | MedQA | test | 500 | 40 | 53.0% [48.6%, 57.3%] | 53.0% | 0.528 | 0.0% | 100.0% |
| Gemma Pair | `google/medgemma-1.5-4b-it` | direct answer | C2 (retrieval) | MedQA | test | 500 | 40 | 57.2% [52.8%, 61.5%] | 57.2% | 0.570 | 0.0% | 100.0% |
| Gemma Pair | `google/gemma-3-4b-it` | direct answer | C2 (retrieval) | PubMedQA | test | 500 | 40 | 50.2% [45.8%, 54.6%] | 50.2% | 0.366 | 0.0% | 100.0% |
| Gemma Pair | `google/medgemma-1.5-4b-it` | direct answer | C2 (retrieval) | PubMedQA | test | 500 | 40 | 51.0% [46.6%, 55.4%] | 51.0% | 0.391 | 0.0% | 100.0% |
| Gemma Pair | `google/gemma-3-4b-it` | direct answer | C3 (oracle abstract) | PubMedQA | test | 500 | 40 | 53.0% [48.6%, 57.3%] | 53.0% | 0.385 | 0.0% | 100.0% |
| Gemma Pair | `google/medgemma-1.5-4b-it` | direct answer | C3 (oracle abstract) | PubMedQA | test | 500 | 40 | 57.0% [52.6%, 61.3%] | 57.0% | 0.347 | 0.0% | 100.0% |

> [!NOTE]
> - Pair A's generic arm is the base Qwen3-1.7B checkpoint run in thinking mode; Pair B's generic arm is the separately reasoning-tuned Qwen3-4B-Thinking-2507 checkpoint. Both were given a 1,024-token reasoning budget.
> - **Exact Match (EM)** is identical to accuracy for single-letter multiple-choice answers and is therefore not shown separately.
> - **BERTScore** is not applicable to single-letter answers.
> - MedPsy-1.7B / Qwen3-1.7B-Thinking were not run on PubMedQA and Qwen3-4B-Thinking-2507 was not run on PubMedQA or the unanswerable set, so those cells are absent rather than empty.
