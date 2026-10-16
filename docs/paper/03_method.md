# 3. Method

## 3.1 Factorial Matched-Pair Framework
To isolate domain adaptation from confounding effects of parameter scale and reasoning token budgets, we establish three matched pairs of Small Language Models ($\le 4\text{B}$):

| Pair ID | Group | Model Name | HF Repository | Architecture | Parameters | Reasoning Cap | Precision | Gating & License |
|---|---|---|---|---|:---:|:---:|:---:|---|
| **Pair 1** | **Medical** | MedPsy-1.7B | `qvac/MedPsy-1.7B` | Qwen2.5 | 1.7B | 1,024 tokens | `bfloat16` | Open (Apache 2.0) |
| **Pair 1** | **Generic** | Qwen3-1.7B-Thinking | `Qwen/Qwen3-1.7B` | Qwen3 | 1.7B | 1,024 tokens | `bfloat16` | Open (Apache 2.0) |
| **Pair 2** | **Medical** | MedPsy-4B | `qvac/MedPsy-4B` | Qwen2.5 | 4.0B | 1,024 tokens | `bfloat16` | Open (Apache 2.0) |
| **Pair 2** | **Generic** | Qwen3-4B-Thinking-2507 | `Qwen/Qwen3-4B-Thinking-2507` | Qwen3 | 4.0B | 1,024 tokens | `bfloat16` | Open (Apache 2.0) |
| **Pair 3** | **Medical** | MedGemma-1.5-4B-IT | `google/medgemma-1.5-4b-it` | Gemma-3 | 4.0B | N/A (direct) | `float32` | Gated (Google HAI-DEF) |
| **Pair 3** | **Generic** | Gemma3-4B | `google/gemma-3-4b-it` | Gemma-3 | 4.0B | N/A (direct) | `float32` | Open (Gemma License) |

All thinking models (`MedPsy-1.7B`, `Qwen3-1.7B-Thinking`, `MedPsy-4B`, and `Qwen3-4B-Thinking-2507`) are evaluated under an invariant **1,024-token reasoning cap** to isolate reasoning quality from reasoning length. In accordance with numerical stability requirements on Kaggle T4 GPUs, Gemma-based architectures are executed in `float32` to eliminate NaN logit generation.

## 3.2 Evaluation Conditions
Each model is evaluated across five standardized experimental conditions ($N=2,140$ total test evaluations per model):
1. **Condition 1 (C1: MedQA Baseline)**: Closed-book USMLE 4-option multiple choice ($N=500$ test questions). Evaluates parametric diagnostic knowledge without external evidence.
2. **Condition 2 (C2: MedQA Full RAG)**: Open-book USMLE 4-option ($N=500$ test questions) conditioned on top-$k=5$ retrieved textbook passages ($\le 1,800$ context tokens).
3. **Condition 3 (C3: PubMedQA Baseline)**: Closed-book biomedical research questions with ternary options (`yes`, `no`, `maybe`) without abstracts ($N=500$ test questions).
4. **Condition 4 (C4: PubMedQA Context)**: Open-book biomedical questions conditioned on the original author-provided PubMed abstracts ($N=500$ test questions).
5. **Condition 5 (C5: Unanswerable v2)**: Abstention benchmark ($N=140$ test questions: 100 unanswerable queries across 6 clinical dilemma categories + 40 paired answerable controls).

## 3.3 Retrieval Pipeline and Knowledge Corpus
The external knowledge base consists of the **MedRAG Textbooks Corpus**, comprising 18 standard USMLE textbooks chunked into 124,077 passages (111.3 MB).

The invariant retrieval pipeline executes in three stages:
1. **Hybrid Retrieval**:
   - **Dense Retrieval**: `ncbi/MedCPT-Query-Encoder` and `ncbi/MedCPT-Article-Encoder` (768 dimensions), indexed using an inverted file product quantization (IVF-PQ) index in FAISS.
   - **Sparse Retrieval**: BM25 index over lowercase, stopword-filtered passage tokens.
2. **Reciprocal Rank Fusion (RRF)**:
   Scores from dense and sparse retrieval are merged via RRF with constant $k=60$:
   $$RRF(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + r_m(d)}$$
   Yielding a candidate pool of top-$M=20$ passages.
3. **Cross-Encoder Re-Ranking**:
   The top-20 candidates are re-ranked using `ncbi/MedCPT-Cross-Encoder`. The top-$k=5$ highest-scoring passages are formatted with bracketed source identifiers (`[Doc 1]`, ..., `[Doc 5]`) and prepended to the prompt within an 1,800-token context budget.

## 3.4 Inference Protocol and Answer Parsing
All evaluations enforce greedy decoding (temperature = 0.0, top-$p=1.0$, seed = 42).
- For direct models, responses are parsed using strict regex extracting the final valid letter choice.
- For thinking models, responses are parsed by first isolating the designated reasoning container (`<think>...</think>`), checking token length against the 1,024-token cap, and extracting the final option from the post-reasoning conclusion.
- Truncation is explicitly recorded when a reasoning block reaches 1,024 tokens without outputting a valid terminal answer.

## 3.5 Statistical Analysis Protocol
To provide rigorous hypothesis testing, all results are evaluated under the following statistical framework:
1. **Paired Exact McNemar Test**: Evaluates discordant question outcomes ($b$: generic correct / medical incorrect; $c$: generic incorrect / medical correct) via exact two-sided binomial tests:
   $$p = 2 \cdot \sum_{i=c}^{b+c} \binom{b+c}{i} 0.5^{b+c} \quad (\text{for } c \ge b)$$
2. **Holm-Bonferroni Step-Down Correction**: Family-wise error rate is controlled across all primary matched-pair hypotheses at $\alpha = 0.05$.
3. **Bootstrap Confidence Intervals**: 95% confidence intervals are computed via $B=1,000$ stratified bootstrap resamples with fixed seed 42.
4. **Cochran-Mantel-Haenszel (CMH) Test**: Tests conditional independence of accuracy and model group (Generic vs. Medical) stratified across the three matched pairs.
5. **Logistic Regression Interaction Model**: Tests the differential RAG effect ($\text{Group} \times \text{RAG}$) via:
   $$\text{logit}(P(Y=1)) = \beta_0 + \beta_1 \cdot \text{Group} + \beta_2 \cdot \text{RAG} + \beta_3 \cdot (\text{Group} \times \text{RAG})$$
   A negative $\beta_3$ ($p < 0.05$) validates Hypothesis $H_2$ that medical models gain less from RAG than generic models.
6. **Primary vs. Answered-Only Accuracy**: Accuracy is reported both on the full cohort ($N=500$, treating truncated/unparsed as incorrect) and on answered-only subsets to verify that findings are not driven by truncation artifacts.
