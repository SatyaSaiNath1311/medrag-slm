# MedRAG-SLM: Pre-Registered Next-Experiment Specifications

**Project**: Retrieval-Augmented Small Language Models for Reliable Medical Question Answering  
**Status**: Pre-registered experimental designs for post-baseline extensions  
**Protocol Rules**:
1. All hyperparameter tuning, prompt development, and gate exploration must be completed on the **validation split only**.
2. Held-out test splits are evaluated **exactly once** using frozen configurations.
3. Success criteria and evaluation metrics are fixed in advance.
4. Upgrades to retrieval or modeling are designated as separate versions (e.g., v2) and **never overwrite existing frozen results**.

---

## Experiment 1: NLI Citation-Support Metric on the Standard 5 SLMs

### 1.1 Objective & Rationale
In the current project, citation entailment verification using `pritamdeka/PubMedBERT-MNLI-MedNLI` is reported exclusively on `medpsy-4b` ($N=500$ MedQA test RAG questions, where 379 were scorable with a final `Evidence: [...]` line). The other 5 models (`qwen3-1.7b`, `phi4-mini`, `smollm3-3b`, `qwen3-4b`, `gemma3-4b`) already generated RAG answers in Phase 8 containing top-5 evidence citations. This experiment executes the identical frozen NLI scoring protocol on all 5 standard SLMs to construct a unified 6-model citation faithfulness table.

### 1.2 Inputs
- Frozen Phase 8 RAG test outputs:
  - `outputs/kaggle_qa/full/qwen3-1.7b/work/phase8/qwen3-1.7b.jsonl`
  - `outputs/kaggle_qa/full/phi4-mini/work/phase8/phi4-mini.jsonl`
  - `outputs/kaggle_qa/full/smollm3-3b/work/phase8/smollm3-3b.jsonl`
  - `outputs/kaggle_qa/full/qwen3-4b/work/phase8/qwen3-4b.jsonl`
  - `outputs/kaggle_qa/full/gemma3-4b_rag/work/phase8/gemma3-4b.jsonl`
- Test questions: `outputs/kaggle_build/work/phase1/test.jsonl` (MedQA test, $N=500$)
- Top-5 reranked evidence passages: `outputs/kaggle_build/work/phase7/evidence.jsonl`
- Pre-trained NLI Model: `pritamdeka/PubMedBERT-MNLI-MedNLI` (threshold $\tau = 0.5$)
- Calibration Parameters (frozen from `nli_dev.py` validation):
  - Empirical dev precision: $\text{PREC} = 0.52$ (95% CI: $[0.335, 0.70]$)
  - Empirical dev false negative rate: $\text{FNR} = 0.04$ (95% CI: $[0.007, 0.195]$)

### 1.3 Exact Outputs
- Per-model test metrics JSON files:
  - `outputs/analysis_nli/{model}_test_citation_metrics.json`
- Unified multi-model summary table: `docs/tables/table_nli_citation_all_models.md` and `.csv` reporting:
  - % Scorable generations (valid `Evidence: [...]` citation line present)
  - Citation precision upper bound (raw NLI-entailed fraction of cited passages)
  - Citation precision corrected for dev precision / NPV
  - Answer-level grounding support rate (at least one cited passage entails predicted answer)
  - Citation recall upper bound (among questions where at least one retrieved passage is entailed, did the model cite it?)

### 1.4 Pre-Registered Success Criteria
- Valid citation line extraction rate $\ge 85.0\%$ across all 5 models.
- Tests the architectural hypothesis: Medical reasoning fine-tuning (`medpsy-4b`) yields significantly higher corrected citation precision than general-domain small language models of comparable parameter scale (`qwen3-4b`, `gemma3-4b`).

### 1.5 Execution Mode & Compute Budget
- **Kaggle Runner Mode**: None. Evaluated entirely offline via CPU/PyTorch script (`nli_all_models_test.py`).
- **Estimated Compute**: 0 GPU hours (~1.0–1.2 CPU hours per model, ~6 hours total across 5 models).

---

## Experiment 2: Robustness Evaluation: Option-Order Shuffling, Paraphrasing, & Distractor Injection

### 2.1 Objective & Rationale
Small language models frequently exhibit severe position bias (e.g. disproportionate preference for option 'A'), sensitivity to minor syntactic rephrasing of clinical vignettes, and susceptibility to distractor passages in retrieval context. This experiment rigorously quantifies model robustness across three perturbation axes.

### 2.2 Protocol & Experimental Design
1. **Pilot on Validation Split First**:
   - Run a pilot on $N=50$ MedQA validation questions to confirm parsing and assertion integrity.
2. **Single Frozen Evaluation on Test Subset**:
   - Evaluate a fixed random sample of $N=200$ MedQA test questions (`seed=42`).
3. **Perturbation Conditions**:
   - **Condition A (Option Permutation)**: Cyclic shift ($A \to B \to C \to D \to A$) and complete reverse order ($D, C, B, A$). Ground-truth labels shifted accordingly.
   - **Condition B (Vignette Paraphrasing)**: Rule-based syntactic variation of patient presentation headers (e.g., standardizing age, sex, and chief complaint order without altering clinical entities or numerical values).
   - **Condition C (Adversarial Distractor Injection)**: Injecting a high-BM25-overlap distractor passage from an unrelated medical specialty into position 1 or 2 of the top-5 retrieval context.

### 2.3 Inputs
- Frozen sample of $N=200$ MedQA test vignettes from `work/phase1/test.jsonl`.
- Top-5 reranked evidence from `work/phase7/evidence.jsonl`.
- Textbook corpus `work/phase2/corpus.jsonl` for extracting distractor passages.

### 2.4 Exact Outputs
- `outputs/robustness/{model}_{condition}.jsonl`
- Summary report `docs/tables/table_robustness_analysis.md` and `.csv` reporting:
  - Prediction Consistency Rate (% of questions where predicted option text remains identical under permutation).
  - Accuracy Degradation: $\Delta \text{Acc} = \text{Acc}_{\text{perturbed}} - \text{Acc}_{\text{clean}}$.
  - Position Entropy: Shannon entropy of predicted answer choices $H(P) = -\sum_{i} p_i \log_2 p_i$ (ideal uniform = 2.0 bits).

### 2.5 Pre-Registered Success Criteria
- Prediction consistency under option permutation $\ge 85.0\%$ for MedPsy-4B, and $\ge 70.0\%$ for standard SLMs.
- Accuracy degradation under distractor injection $\le 5.0$ percentage points across all models.

### 2.6 Execution Mode & Compute Budget
- **Kaggle Runner Mode**: `STEP = "ROBUSTNESS_EVAL"` on 2x T4 GPUs.
- **Estimated GPU Hours**: ~2.5 GPU hours on Kaggle 2x T4.

---

## Experiment 3: Out-of-Domain Generalization on MedMCQA and MMLU-Medical

### 2.1 Objective & Rationale
Determine whether the comparative performance advantages of MedPsy-4B, RAG gains, and calibrated abstention rules hold when deployed on out-of-distribution medical question benchmarks.

### 3.2 Protocol & Pre-Flight Leakage Guard
1. **Mandatory 13-gram Corpus Contamination Scan**:
   - Extract contiguous 13-word n-grams from all candidate questions and options in MedMCQA and MMLU-Medical.
   - Match against the 124,077-chunk textbook retrieval corpus (`outputs/kaggle_build/work/phase2/corpus.jsonl`).
   - Discard any question with substantive verbatim overlap prior to sampling.
2. **Sampling & Splits**:
   - **MedMCQA**: $N=500$ held-out test sample from `openlifescienceai/medmcqa` (AIIMS/NEET PG medical entrance exam questions).
   - **MMLU-Medical**: $N=500$ held-out test sample aggregated across Clinical Knowledge, Medical Genetics, Anatomy, Professional Medicine, and College Biology.
   - Validation pilot: $N=100$ per dataset for format confirmation.

### 3.3 Inputs
- Cleaned HuggingFace datasets (`openlifescienceai/medmcqa`, `cais/mmlu`).
- Master retrieval index (`outputs/kaggle_build/work/phase3/index/`).

### 3.4 Exact Outputs
- Benchmark splits: `data/generalization/{medmcqa,mmlu_medical}_test.jsonl`.
- Retrieved evidence: `outputs/generalization/work/phase7/evidence.jsonl`.
- Model answer traces: `outputs/generalization/qa/{model}_{dataset}_{mode}.jsonl`.
- Comparison table: `docs/tables/table_generalization_results.md` and `.csv`.

### 3.5 Pre-Registered Success Criteria
- Positive RAG improvement ($\Delta \text{Acc} > 0$) observed on factual, memory-intensive subdomains (Anatomy, Medical Genetics).
- Calibrated conformal reliability threshold ($\tau = 0.5119$) achieves False Abstention Rate $\le 10.0\%$ on answerable out-of-domain medical queries.

### 3.6 Execution Mode & Compute Budget
- **Kaggle Runner Mode**: `STEP = "GENERALIZATION_RUN"` on 2x T4 GPUs.
- **Estimated GPU Hours**: ~4.0 GPU hours on Kaggle 2x T4.

---

## Experiment 4: 4-Bit Quantization Efficiency & Precision Benchmarking

### 4.1 Objective & Rationale
Examine the feasibility of deploying 4B parameter models on resource-constrained clinical workstations or mobile health devices by benchmarking 4-bit NormalFloat (NF4 with double quantization) against native 16-bit float16 baselines.

### 4.2 Protocol & Experimental Design
1. **Quantization Pipeline**:
   - Use `bitsandbytes` NF4 quantization with double quantization (`bnb_4bit_use_double_quant=True`) and float16 compute dtype (`bnb_4bit_compute_dtype=torch.float16`).
2. **Evaluated Models**:
   - `medpsy-4b`
   - `qwen3-4b`
   - `gemma3-4b`
3. **Benchmark Scope**:
   - Held-out MedQA test ($N=500$) across Baseline and Full RAG modes.

### 4.3 Inputs
- HuggingFace model repositories loaded with `BitsAndBytesConfig(load_in_4bit=True)`.
- Test questions: `outputs/kaggle_build/work/phase1/test.jsonl`.
- Top-5 reranked evidence: `outputs/kaggle_build/work/phase7/evidence.jsonl`.

### 4.4 Exact Outputs
- Model predictions: `outputs/quantization/{model}_4bit_{mode}.jsonl`.
- Comparative table `docs/tables/table_quantization_efficiency.md` and `.csv` reporting:
  - Accuracy retention ratio: $\text{Acc}_{\text{4-bit}} / \text{Acc}_{\text{16-bit}}$.
  - Peak GPU memory allocated (GB).
  - Average latency per question (seconds).
  - Generation throughput (tokens/second).

### 4.5 Pre-Registered Success Criteria
- Accuracy retention $\ge 97.0\%$ of original float16 accuracy (MedPsy-4B retains $\ge 85.0\%$ MedQA accuracy).
- Peak GPU VRAM allocated drops below **5.5 GB**, fitting comfortably on consumer 6 GB and 8 GB GPUs (e.g. RTX 3060, RTX 4060, Apple M-series unified memory).

### 4.6 Execution Mode & Compute Budget
- **Kaggle Runner Mode**: `STEP = "QUANTIZATION_EVAL"` on a single T4 GPU (or 2x T4).
- **Estimated GPU Hours**: ~2.0 GPU hours on Kaggle.

---

## Experiment 5: Optional Retrieval Architecture Upgrade (Designated as MedRAG v2)

### 5.1 Objective & Rationale
In the v1 benchmark, MedCPT dense retrieval achieves 30.4% context recall on MedQA top-5 passages (152/500 questions contain the correct answer option text in the top 5 passages). An upgraded hybrid retrieval architecture can significantly increase grounding context precision and downstream reasoning accuracy.

### 5.2 Protocol (Strict Enforcement of Rule 6)
1. **Development on Validation Split Only**:
   - Evaluate candidate retrieval enhancements exclusively on `work/phase1/validation.jsonl` ($N=200$ MedQA val):
     - Dense retriever comparison: `ncbi/MedCPT-Query-Encoder` vs `BAAI/bge-m3`.
     - Chunking strategy: 350 words / 50 overlap vs 200 words / 30 overlap.
     - Fusion formula: Reciprocal Rank Fusion ($k=60$) vs learned linear score interpolation.
2. **Freeze Winning Configuration**:
   - Select the optimal configuration based solely on validation top-5 context recall.
3. **Single Test Evaluation**:
   - Run the frozen v2 retriever exactly once on held-out test splits.
   - Save all outputs into a separate directory: `outputs/retrieval_v2/`.
   - Report as a distinct **"v2 Retrieval Benchmark"** table, preserving the frozen v1 baseline tables intact.

### 5.3 Inputs
- Validation split: `outputs/kaggle_build/work/phase1/validation.jsonl` (for tuning).
- Test split: `outputs/kaggle_build/work/phase1/test.jsonl` (for single frozen evaluation).
- Textbook corpus: `outputs/kaggle_build/work/phase2/corpus.jsonl`.

### 5.4 Exact Outputs
- v2 Evidence candidate pool: `outputs/retrieval_v2/work/phase7/evidence.jsonl`.
- v2 QA generations: `outputs/retrieval_v2/qa/{model}.jsonl`.
- Comparison table: `docs/tables/table_retrieval_v2_comparison.md` and `.csv`.

### 5.5 Pre-Registered Success Criteria
- Validation top-5 context recall increases by at least **+10.0 percentage points** (from 30.4% to $\ge 40.4\%$).
- Downstream MedQA test accuracy improves by $\ge 2.0$ percentage points without degrading PubMedQA accuracy.

### 5.6 Execution Mode & Compute Budget
- **Kaggle Runner Mode**: `STEP = "RETRIEVAL_V2_BUILD"` then `STEP = "RETRIEVAL_V2_QA"`.
- **Estimated GPU Hours**: ~3.5 GPU hours on Kaggle 2x T4.

---

## Complementary Execution Tasks: Resource & Empirical GPU-Hour Budget

Using empirical generation timings directly extracted from the completed runner logs (`outputs/kaggle_qa/full/*/medrag-slm-runner.log`, `outputs/kaggle_qa/medpsy_final/medrag-slm-runner.log`, and `outputs/kaggle_qa/medpsy_rag/medrag-slm-runner.log`), the compute allocations for the remaining benchmark cells are revised below.

### Empirical Latencies from Frozen Benchmark Logs (Kaggle T4 GPUs)
- **`qwen3-1.7b`**: Baseline = 0.1192 s/q | RAG = 0.7009 s/q (combined = 0.8201 s/q)
- **`smollm3-3b`**: Baseline = 0.1657 s/q | RAG = 0.9454 s/q (combined = 1.1111 s/q)
- **`phi4-mini`**: Baseline = 0.1664 s/q | RAG = 1.1366 s/q (combined = 1.3030 s/q)
- **`qwen3-4b`**: Baseline = 0.2282 s/q | RAG = 1.4659 s/q (combined = 1.6941 s/q)
- **`gemma3-4b`**: Baseline = 0.6278 s/q | RAG = 3.3412 s/q (combined = 3.9690 s/q)
- **`medpsy-4b`**:
  - Baseline (MedQA/Unanswerable greedy reasoning): **37.58 s/q** (18.9% truncated at 1,024 thinking cap)
  - Full RAG (MedQA test greedy reasoning + top-5 citations): **63.93 s/q** (21.6% truncated)
  - Gold Context (PubMedQA context reasoning): **30.28 s/q** (0.2% truncated)

---

### Task 4 Compute: Unanswerable v2 Benchmark Matrix ($N=400$ Questions: 260 Val + 140 Test [100 Unanswerable + 40 Controls])

1. **Retrieval & Rerank Pre-Pass (`BUILD_UNANS_V2_EVIDENCE = True`)**:
   - Executes Phase 6 (dense MedCPT + BM25) and Phase 7 (MedCPT-Cross-Encoder) for all 400 questions against the existing 124,077-chunk textbook corpus.
   - Empirical wall-clock duration: ~3.5–4.5 minutes.
   - **Compute**: **~0.08 GPU hours** on 1x/2x T4.

2. **Five Standard SLMs (`UNANS_V2_SLM = True`, one model per kernel push)**:
   - Questions per model: 400 questions across both Baseline (Phase 5) and RAG (Phase 8).
   - Per-model pure inference duration:
     - `qwen3-1.7b`: $400 \times 0.8201\text{ s} = 328\text{ s} \approx 5.5\text{ mins}$ (~0.09 GPU hrs)
     - `smollm3-3b`: $400 \times 1.1111\text{ s} = 444\text{ s} \approx 7.4\text{ mins}$ (~0.12 GPU hrs)
     - `phi4-mini`: $400 \times 1.3030\text{ s} = 521\text{ s} \approx 8.7\text{ mins}$ (~0.14 GPU hrs)
     - `qwen3-4b`: $400 \times 1.6941\text{ s} = 678\text{ s} \approx 11.3\text{ mins}$ (~0.19 GPU hrs)
     - `gemma3-4b`: $400 \times 3.9690\text{ s} = 1,588\text{ s} \approx 26.5\text{ mins}$ (~0.44 GPU hrs)
   - Total pure inference: $3,559\text{ s} \approx 59.3\text{ mins} \approx \mathbf{0.99\text{ GPU hrs}}$.
   - Including model download and weight materialization overhead across 5 kernel pushes: **~1.25 GPU hours**.

3. **MedPsy-4B Standardized Baseline (`MEDPSY_UNANS_V2_BASELINE = True`)**:
   - Evaluates `medpsy-4b` on all 400 unanswerable_v2 questions using the identical Phase 5 standardized baseline prompt and greedy decoding (providing the primary MedPsy entry in Table 7).
   - Dual-GPU parity shards across 2x T4 (shard 0: 200 questions [130 val + 70 test]; shard 1: 200 questions [130 val + 70 test]).
   - At logged baseline rate of 37.58 s/q:
     - Wall-clock time: $200 \times 37.58\text{ s} = 7,516\text{ s} \approx 2.09\text{ hours}$.
     - Total compute: $2.09\text{ h} \times 2\text{ GPUs} = \mathbf{4.18\text{ GPU hours}}$.
- **Total Task 4 Compute**: $0.08 + 1.25 + 4.18 = \mathbf{\sim 5.51\text{ GPU hours}}$.

---

### Task 5 Compute: MedPsy-4B Missing Cells on PubMedQA ($N=500$ Test Questions)

1. **Sub-Task 5A: MedPsy-4B Closed-Book Baseline (`MEDPSY_PUBMEDQA_BASELINE = True`)**:
   - Evaluates closed-book baseline (no context, no retrieval) on 500 PubMedQA test questions.
   - Dual-GPU shards across 2x T4 (250 questions per GPU).
   - At empirical baseline rate of 37.58 s/q:
     - Wall-clock time: $250 \times 37.58\text{ s} = 9,395\text{ s} \approx 2.61\text{ hours}$ (well below 9h kernel limit).
     - Total compute: $2.61\text{ h} \times 2\text{ GPUs} = \mathbf{5.22\text{ GPU hours}}$.

2. **Sub-Task 5B: MedPsy-4B Full RAG (`MEDPSY_PUBMEDQA_RAG = True`)**:
   - Evaluates full open-domain RAG (retrieving top-5 textbook passages) on 500 PubMedQA test questions.
   - Dual-GPU shards across 2x T4 (250 questions per GPU).
   - At empirical RAG rate of 63.93 s/q:
     - Wall-clock time: $250 \times 63.93\text{ s} = 15,983\text{ s} \approx 4.44\text{ hours}$ (well below 9h kernel limit).
     - Total compute: $4.44\text{ h} \times 2\text{ GPUs} = \mathbf{8.88\text{ GPU hours}}$.
- **Total Task 5 Compute**: $5.22 + 8.88 = \mathbf{14.10\text{ GPU hours}}$ (~7.05h combined wall-clock on 2x T4).

---

## Master Experiment Resource Allocation Summary

| Experiment / Task ID | Title / Target | Compute Device | Kaggle Runner Mode Setting | Est. GPU Hours | Wall-Clock Time | Prerequisite |
|---|---|:---:|:---:|:---:|:---:|---|
| **Task 4 (Pre-Pass)** | Unanswerable v2 Retrieval & Rerank | Kaggle 2x T4 | `BUILD_UNANS_V2_EVIDENCE = True` | **~0.1 hrs** | ~4 mins | `medrag-build` output |
| **Task 4 (5 SLMs)** | 5 SLMs on Unanswerable v2 (Base+RAG) | Kaggle 1x/2x T4 | `UNANS_V2_SLM = True` (per model) | **~1.3 hrs** | ~1.0h total | Pre-pass evidence |
| **Task 4 (MedPsy)** | MedPsy-4B Unanswerable v2 Baseline | Kaggle 2x T4 | `MEDPSY_UNANS_V2_BASELINE = True` | **~4.2 hrs** | ~2.1h | `data/unanswerable_v2/` |
| **Task 5A** | MedPsy-4B PubMedQA Baseline (500) | Kaggle 2x T4 | `MEDPSY_PUBMEDQA_BASELINE = True` | **~5.2 hrs** | ~2.6h | `medrag-build` output |
| **Task 5B** | MedPsy-4B PubMedQA Full RAG (500) | Kaggle 2x T4 | `MEDPSY_PUBMEDQA_RAG = True` | **~8.9 hrs** | ~4.4h | Phase 7 evidence |
| **Exp 1** | NLI Citation Support on 5 SLMs | Local CPU | None (`nli_all_models_test.py`) | 0.0 hrs | ~6h (CPU) | Existing Phase 8 test outputs |
| **Exp 2** | Robustness & Perturbation Suite | Kaggle 2x T4 | `ROBUSTNESS_EVAL = True` | ~2.5 hrs | ~1.3h | Phase 1 test split & evidence |
| **Exp 3** | Out-of-Domain Generalization | Kaggle 2x T4 | `GENERALIZATION_RUN = True` | ~4.0 hrs | ~2.0h | 13-gram contamination scan |
| **Exp 4** | 4-Bit Quantization Efficiency | Kaggle 1x/2x T4 | `QUANTIZATION_EVAL = True` | ~2.0 hrs | ~1.5h | `bitsandbytes` installation |
| **Exp 5** | MedRAG v2 Retrieval Upgrade | Kaggle 2x T4 | `RETRIEVAL_V2_BUILD / QA = True` | ~3.5 hrs | ~2.5h | Validation-only recall tuning |
| **Total** | **All Extensions + Missing Cells** | — | — | **~31.7 GPU hrs** | — | **Fits in Kaggle quotas** |

