# Table 3: Inference Efficiency and Generation Latency across Architectures and Inference Modes (V3 Study)

Empirical measurement of inference runtime, throughput, and memory consumption across **C1 closed-book**, **C2 retrieval**, and **C3 oracle abstract**.

| Model | Mode | Condition | Dataset | N | Mean seconds/question | Median seconds/question | p90 seconds/question | Mean generated tokens | Tokens/second | Peak VRAM | RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `qvac/MedPsy-1.7B` | thinking (reasoning=true) | C1 (closed-book) | MedQA | 500 | 34.4 | 34.2 | 38.9 | 821.1 | 23.8 | not measured | not measured |
| `Qwen/Qwen3-1.7B` | thinking (reasoning=true) | C1 (closed-book) | MedQA | 500 | 34.0 | 34.3 | 39.0 | 805.6 | 23.7 | not measured | not measured |
| `qvac/MedPsy-1.7B` | thinking (reasoning=true) | C2 (retrieval) | MedQA | 500 | 42.8 | 43.2 | 45.7 | 927.7 | 21.7 | not measured | not measured |
| `Qwen/Qwen3-1.7B` | thinking (reasoning=true) | C2 (retrieval) | MedQA | 500 | 38.6 | 39.3 | 43.0 | 827.8 | 21.4 | not measured | not measured |
| `qvac/MedPsy-4B` | thinking (reasoning=true) | C1 (closed-book) | MedQA | 500 | 37.3 | 37.1 | 46.7 | 562.9 | 15.1 | not measured | not measured |
| `Qwen/Qwen3-4B-Thinking-2507` | thinking (reasoning=true) | C1 (closed-book) | MedQA | 500 | 65.0 | 64.6 | 66.6 | 1024.0 | 15.7 | not measured | not measured |
| `qvac/MedPsy-4B` | thinking (reasoning=true) | C2 (retrieval) | MedQA | 500 | 63.9 | 63.4 | 74.7 | 758.3 | 11.9 | not measured | not measured |
| `Qwen/Qwen3-4B-Thinking-2507` | thinking (reasoning=true) | C2 (retrieval) | MedQA | 500 | 88.4 | 88.1 | 92.4 | 1024.0 | 11.6 | not measured | not measured |
| `google/gemma-3-4b-it` | direct answer | C1 (closed-book) | MedQA | 500 | 1.1 | 1.0 | 1.2 | 15.7 | 14.8 | not measured | not measured |
| `google/medgemma-1.5-4b-it` | direct answer | C1 (closed-book) | MedQA | 500 | 1.1 | 1.1 | 1.3 | 16.0 | 14.1 | not measured | not measured |
| `google/gemma-3-4b-it` | direct answer | C1 (closed-book) | PubMedQA | 500 | 0.4 | 0.4 | 0.4 | 7.2 | 19.0 | not measured | not measured |
| `google/medgemma-1.5-4b-it` | direct answer | C1 (closed-book) | PubMedQA | 500 | 0.3 | 0.2 | 0.4 | 2.1 | 7.2 | not measured | not measured |
| `google/gemma-3-4b-it` | direct answer | C2 (retrieval) | MedQA | 500 | 3.8 | 3.7 | 4.4 | 11.3 | 3.0 | not measured | not measured |
| `google/medgemma-1.5-4b-it` | direct answer | C2 (retrieval) | MedQA | 500 | 4.1 | 3.9 | 5.2 | 10.4 | 2.5 | not measured | not measured |
| `google/gemma-3-4b-it` | direct answer | C2 (retrieval) | PubMedQA | 500 | 3.2 | 3.2 | 3.9 | 11.5 | 3.5 | not measured | not measured |
| `google/medgemma-1.5-4b-it` | direct answer | C2 (retrieval) | PubMedQA | 500 | 3.5 | 3.4 | 4.3 | 10.4 | 3.0 | not measured | not measured |
| `google/gemma-3-4b-it` | direct answer | C3 (oracle abstract) | PubMedQA | 500 | 0.6 | 0.6 | 0.6 | 7.0 | 11.8 | not measured | not measured |
| `google/medgemma-1.5-4b-it` | direct answer | C3 (oracle abstract) | PubMedQA | 500 | 0.6 | 0.6 | 0.6 | 7.0 | 11.4 | not measured | not measured |

> [!NOTE]
> - **Tokens/second formula**: Stated as mean generated tokens divided by mean seconds per question ($\text{Tokens/second} = \frac{\text{Mean generated tokens}}{\text{Mean seconds/question}}$).
> - **Peak VRAM and RAM**: Printed as "not measured" for every row because no manifest records them. They will be measured in the live deployment test and timings come from batched offline evaluation on Kaggle 2x T4, so they are not single-request latencies.
> - **Footnote**: All evaluation rows contained recorded seconds and generated token counts (0 rows excluded).

* Footnote: All evaluation rows contained recorded seconds and generated token counts (0 rows excluded).
