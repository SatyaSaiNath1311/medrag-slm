# MedRAG-SLM Research Project: Comprehensive Status Report

**Project Title**: Retrieval-Augmented Small Language Models for Reliable Medical Question Answering  
**Scope**: 6 Small Language Models (`qwen3-1.7b`, `phi4-mini`, `smollm3-3b`, `qwen3-4b`, `gemma3-4b`, `medpsy-4b`)  
**Core Benchmark Splits**: Held-out Test ($N=1,150$ for Phase 1; $N=140$ for Unanswerable v2); Validation ($N=450$ for Phase 1; $N=260$ for Unanswerable v2; $N=660$ for SC Validation)  
**Date of Audit**: October 14, 2026  
**Auditor**: Antigravity Pair-Programming Agent (Read-Only Analysis Mode)

---

## 1. Inventory of All Result Sets

The table below catalogs every empirical result set generated across local runs and Kaggle GPU executions (`kaggle_runner/runner.py`), detailing model, inference mode, split, dataset coverage, sample size ($N$), physical output path, last commit SHA associated with the run, and git tracking status.

| Model | Mode | Split | Dataset(s) | $N$ | Output File Path | Associated Git Commit SHA | Git Tracked? |
|---|---|---|---|:---:|---|---|:---:|
| `qwen3-1.7b` | Baseline | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/qwen3-1.7b/work/phase5/qwen3-1.7b.jsonl` | `9602c7051e404dfbe62d2722eea58de63fc00f3c` | Gitignored (`outputs/*`) |
| `qwen3-1.7b` | Full RAG | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/qwen3-1.7b/work/phase8/qwen3-1.7b.jsonl` | `9602c7051e404dfbe62d2722eea58de63fc00f3c` | Gitignored (`outputs/*`) |
| `qwen3-1.7b` | Adaptive | Test (eval) | MedQA (500), PubMedQA (500) [Val tuned: 400] | 1,000 | `outputs/analysis/adaptive_rag_qwen3-1.7b.json` | `2ab5170c722699af7ca5e38d21188f85c0aee273` | **Committed** (`!outputs/analysis/`) |
| `qwen3-1.7b` | +Abstract | Test + Val | PubMedQA (+Study Abstract) | 700 | `outputs/kaggle_qa/context/work/phase10/qwen3-1.7b.jsonl` | `213e486ee860ca3db5a4cfd2f40506136089f2b1` | Gitignored (`outputs/*`) |
| `phi4-mini` | Baseline | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/phi4-mini/work/phase5/phi4-mini.jsonl` | `b64f75f62d31953c4a9dab18b9889d6039a2e889` | Gitignored (`outputs/*`) |
| `phi4-mini` | Full RAG | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/phi4-mini/work/phase8/phi4-mini.jsonl` | `b64f75f62d31953c4a9dab18b9889d6039a2e889` | Gitignored (`outputs/*`) |
| `phi4-mini` | Adaptive | Test (eval) | MedQA (500), PubMedQA (500) [Val tuned: 400] | 1,000 | `outputs/analysis/adaptive_rag_phi4-mini.json` | `2ab5170c722699af7ca5e38d21188f85c0aee273` | **Committed** (`!outputs/analysis/`) |
| `phi4-mini` | +Abstract | Test + Val | PubMedQA (+Study Abstract) | 700 | `outputs/kaggle_qa/context/work/phase10/phi4-mini.jsonl` | `213e486ee860ca3db5a4cfd2f40506136089f2b1` | Gitignored (`outputs/*`) |
| `smollm3-3b` | Baseline | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/smollm3-3b/work/phase5/smollm3-3b.jsonl` | `7c45ce5a912d8b0d8aa10c7c967cbea132537b30` | Gitignored (`outputs/*`) |
| `smollm3-3b` | Full RAG | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/smollm3-3b/work/phase8/smollm3-3b.jsonl` | `7c45ce5a912d8b0d8aa10c7c967cbea132537b30` | Gitignored (`outputs/*`) |
| `smollm3-3b` | Adaptive | Test (eval) | MedQA (500), PubMedQA (500) [Val tuned: 400] | 1,000 | `outputs/analysis/adaptive_rag_smollm3-3b.json` | `2ab5170c722699af7ca5e38d21188f85c0aee273` | **Committed** (`!outputs/analysis/`) |
| `smollm3-3b` | +Abstract | Test + Val | PubMedQA (+Study Abstract) | 700 | `outputs/kaggle_qa/context/work/phase10/smollm3-3b.jsonl` | `213e486ee860ca3db5a4cfd2f40506136089f2b1` | Gitignored (`outputs/*`) |
| `qwen3-4b` | Baseline | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/qwen3-4b/work/phase5/qwen3-4b.jsonl` | `29267fc7d1c69d9c763f6c6fe06663b2504b4339` | Gitignored (`outputs/*`) |
| `qwen3-4b` | Full RAG | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/qwen3-4b/work/phase8/qwen3-4b.jsonl` | `29267fc7d1c69d9c763f6c6fe06663b2504b4339` | Gitignored (`outputs/*`) |
| `qwen3-4b` | Adaptive | Test (eval) | MedQA (500), PubMedQA (500) [Val tuned: 400] | 1,000 | `outputs/analysis/adaptive_rag_qwen3-4b.json` | `2ab5170c722699af7ca5e38d21188f85c0aee273` | **Committed** (`!outputs/analysis/`) |
| `qwen3-4b` | +Abstract | Test + Val | PubMedQA (+Study Abstract) | 700 | `outputs/kaggle_qa/context/work/phase10/qwen3-4b.jsonl` | `213e486ee860ca3db5a4cfd2f40506136089f2b1` | Gitignored (`outputs/*`) |
| `qwen3-4b` | IDK Baseline | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/idk/work/phase12/qwen3-4b_baseline_idk.jsonl` | `3a89019de5776a94a7d26bd23e8ddbddff7444ae` | Gitignored (`outputs/*`) |
| `qwen3-4b` | IDK RAG | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/idk/work/phase12/qwen3-4b_rag_idk.jsonl` | `3a89019de5776a94a7d26bd23e8ddbddff7444ae` | Gitignored (`outputs/*`) |
| `gemma3-4b` | Baseline | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/gemma3-4b/work/phase5/gemma3-4b.jsonl` | `c067731860569cb28892d02ccb9336debc6d0341` | Gitignored (`outputs/*`) |
| `gemma3-4b` | Full RAG | Test + Val | MedQA (500/200), PubMedQA (500/200), Unans v1 (150/50) | 1,600 | `outputs/kaggle_qa/full/gemma3-4b_rag/work/phase8/gemma3-4b.jsonl` | `c067731860569cb28892d02ccb9336debc6d0341` | Gitignored (`outputs/*`) |
| `gemma3-4b` | Adaptive | Test (eval) | MedQA (500), PubMedQA (500) [Val tuned: 400] | 1,000 | `outputs/analysis/adaptive_rag_gemma3-4b.json` | `2ab5170c722699af7ca5e38d21188f85c0aee273` | **Committed** (`!outputs/analysis/`) |
| `gemma3-4b` | +Abstract | Test + Val | PubMedQA (+Study Abstract) | 700 | `outputs/kaggle_qa/context/work/phase10/gemma3-4b.jsonl` | `54c24dab2bd32b91de38fa0de35b7d50eb665f17` | Gitignored (`outputs/*`) |
| `medpsy-4b` | Baseline | Test | MedQA (500), Unanswerable v1 (150) | 650 | `outputs/kaggle_qa/medpsy_final/work/phase5/medpsy-4b.jsonl` | `bcddc01a5e935a97f8a32fe8315f38f7b88019d9` | Gitignored (`outputs/*`) |
| `medpsy-4b` | Baseline | Validation | MedQA (200), Unanswerable v1 (50) | 250 | `outputs/kaggle_qa/medpsy_val/work/phase5/medpsy-4b.jsonl` | `02c271f2bbac27f466c2a542b7e8a6a5eee7db89` | Gitignored (`outputs/*`) |
| `medpsy-4b` | Full RAG | Test | MedQA (500) | 500 | `outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl` | `6c374edd8877af25dc3d9389e42be11194faab97` | Gitignored (`outputs/*`) |
| `medpsy-4b` | +Abstract | Test | PubMedQA (+Study Abstract) | 500 | `outputs/kaggle_qa/medpsy_final/work/phase10/medpsy-4b.jsonl` | `bcddc01a5e935a97f8a32fe8315f38f7b88019d9` | Gitignored (`outputs/*`) |
| `medpsy-4b` | +Abstract | Validation | PubMedQA (+Study Abstract) | 200 | `outputs/kaggle_qa/medpsy_val/work/phase10/medpsy-4b.jsonl` | `02c271f2bbac27f466c2a542b7e8a6a5eee7db89` | Gitignored (`outputs/*`) |
| `medpsy-4b` | Self-Consistency | Validation | MedQA (200), PubMedQA (200), Unanswerable v2 (260) | 660 | `outputs/sc_val/medpsy-4b/{medqa,pubmedqa,unanswerable_v2}.jsonl` | `558708bc387ef274283bb13cecc8c137f6e2ee17` | Gitignored (`outputs/*`) |
| `medpsy-4b` | Self-Consistency | Test (S1) | MedQA ($N=500$) | 500 | `outputs/sc_test/medpsy-4b/outputs/sc_test/medpsy-4b/medqa.jsonl` | `9143924be03e3851dd98e89c20c2dce2ff7488c7` | Gitignored (`outputs/*`) |
| `medpsy-4b` | Self-Consistency | Test (S2) | PubMedQA ($N=500$) | 500 | `outputs/sc_test_s2_pubmedqa/outputs/sc_test/medpsy-4b/pubmedqa.jsonl` | `c4982638d71756d8bb339f01721ae2936cdc3e63` | Gitignored (`outputs/*`) |
| `medpsy-4b` | Self-Consistency | Test (S3) | Unanswerable v2 (100 challenge + 40 control) | 140 | `outputs/sc_test_s3_unanswerable/outputs/sc_test/medpsy-4b/unanswerable_v2.jsonl` | `2f5f0b509c6e02bda5c6b504a3815102f7743bae` | Gitignored (`outputs/*`) |
| `medpsy-4b` | Reliability Score | Test (Eval) | MedQA (500), PubMedQA (500), Unanswerable v2 (140) | 1,140 | `outputs/analysis_test/reliability_test_results.json` | `2f5f0b509c6e02bda5c6b504a3815102f7743bae` | Gitignored (`outputs/*`) |
| `medpsy-4b` | NLI Metric | Test (Eval) | MedQA Test RAG ($N=500$, 379 scorable, 677 cited pairs) | 500 | `outputs/analysis_nli/medpsy_test_citation_metrics.json` | `59da64619e4e61975d467a57f5548f8d42ee7546` | Gitignored (`outputs/*`) |

---

## 2. Six-Model Final Test Coverage Matrix & Empirical Gaps

The table below indicates the existence of **FINAL HELD-OUT TEST RESULTS** for each model across all research dimensions. Every gap is clearly labeled.

| Model | MedQA Test ($N=500$) | PubMedQA Baseline ($N=500$) | PubMedQA +Abstract ($N=500$) | Unanswerable v1 ($N=150$) | Unanswerable v2 ($N=140$) | Self-Consistency (SC) | Calibrated Reliability Score | NLI Citation Support Metric |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **qwen3-1.7b** | **YES** (Base + RAG) | **YES** (Base + RAG) | **YES** | **YES** (Base + RAG) | **GAP** (Never run) | **GAP** (Never run) | **GAP** (Never fitted) | **GAP** (Dev only, no test run) |
| **phi4-mini** | **YES** (Base + RAG) | **YES** (Base + RAG) | **YES** | **YES** (Base + RAG) | **GAP** (Never run) | **GAP** (Never run) | **GAP** (Never fitted) | **GAP** (Dev only, no test run) |
| **smollm3-3b** | **YES** (Base + RAG) | **YES** (Base + RAG) | **YES** | **YES** (Base + RAG) | **GAP** (Never run) | **GAP** (Never run) | **GAP** (Never fitted) | **GAP** (Dev only, no test run) |
| **qwen3-4b** | **YES** (Base + RAG) | **YES** (Base + RAG) | **YES** | **YES** (Base + RAG) | **GAP** (Never run) | **GAP** (Never run) | **GAP** (Never fitted) | **GAP** (Dev only, no test run) |
| **gemma3-4b** | **YES** (Base + RAG) | **YES** (Base + RAG) | **YES** | **YES** (Base + RAG) | **GAP** (Never run) | **GAP** (Never run) | **GAP** (Never fitted) | **GAP** (Dev only, no test run) |
| **medpsy-4b** | **YES** (Base + RAG) | **GAP** (Omitted) | **YES** | **YES** (Baseline only) | **YES** (SC Session 3) | **YES** (S1, S2, S3) | **YES** (Val fitted, Test evaluated) | **YES** (MedQA Test evaluated) |

### Key Observations on Missing Results
1. **MedPsy-4B PubMedQA Baseline (Closed-Book)**: MedPsy-4B was evaluated on PubMedQA *only* in Oracle Context (+Abstract) mode ($N=500$) and SC mode ($N=500$), but was never run in zero-context closed-book baseline mode.
2. **MedPsy-4B RAG Coverage**: MedPsy-4B Full RAG was executed exclusively on MedQA test ($N=500$). It was never executed with RAG on PubMedQA or on Unanswerable v2.
3. **Unanswerable v2 Benchmark**: The new unanswerable benchmark (400 items total, 140 test) was evaluated *only* on MedPsy-4B in SC mode. The 5 standard SLMs were evaluated solely on the older Unanswerable v1 set (150 test items generated by omitting the correct option from MedQA train questions).
4. **Self-Consistency (SC) on Standard SLMs**: SC generation (5 sampled paths per question + greedy pass) was executed solely on MedPsy-4B across 3 Kaggle sessions. It was never executed on the other 5 models due to compute allocation.
5. **NLI Citation Support Evaluation**: NLI-based citation verification (`PubMedBERT-MNLI-MedNLI`) was tested on validation pairs across models (`nli_dev.py`), but test set citation support was executed solely on MedPsy-4B (`nli_medpsy_test.py`).

---

## 3. Codebase Audit: Every Location Reading Test Data

The table below lists every instance in `src/` and root scripts where a held-out test file is read, accompanied by an explicit leakage check flagging whether the script tunes or fits any parameter.

| File | Line(s) | Target Test File Read | Purpose | Reads Test & Fits/Tunes? | Leakage Assessment |
|---|---|---|---|:---:|---|
| [src/phase01_datasets.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/phase01_datasets.py) | 48, 131, 170 | MedQA official test split (HF) | Samples fixed 500 test questions; writes `work/phase1/test.jsonl` | **NO** | Zero tuning; pure sampling with seed 42. |
| [src/phase06_retrieve.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/phase06_retrieve.py) | 51, 77 | `work/phase1/test.jsonl` | Encodes test queries with MedCPT Query Encoder; retrieves BM25 & dense passages | **NO** | Fixed inference; zero fitting. |
| [src/phase07_rerank.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/phase07_rerank.py) | 27, 57 | `work/phase1/test.jsonl` | Reranks top retrieved candidates with MedCPT Cross-Encoder | **NO** | Fixed inference; zero fitting. |
| [src/qa_pipeline.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/qa_pipeline.py) | 282, 295, 639, 829, 835, 841 | `work/phase1/test.jsonl` | Batched LLM inference on test split questions across baseline, RAG, and context modes | **NO** | Generation-only pipeline. |
| [src/final_results.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/final_results.py) | 217, 251, 281, 310, 339, 350 | Phase 5, 8, 10 model outputs (`split == 'test'`) | Computes final benchmark accuracy, bootstrap 95% CIs, McNemar tests | **NO** | Pure evaluation; no thresholds or models fitted. |
| [src/metrics_extended.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/metrics_extended.py) | 297, 303 | `work/phase1/test.jsonl` + Phase 5/8/10 outputs | Computes Macro-F1, Exact Match, option entropy, and confusion matrices | **NO** | Metric reporting only. |
| [src/presentation_tables.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/presentation_tables.py) | 155 | `work/phase1/test.jsonl` + model outputs | Formats presentation tables 1–4 | **NO** | Tabulation only. |
| [src/data_audit.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/data_audit.py) | 55, 145 | `work/phase1/test.jsonl` | Audits test schemas, 5-gram Jaccard leakage, and 13-gram textbook overlap | **NO** | Audit script only. |
| [src/analysis_hallucination.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/analysis_hallucination.py) | 259, 260, 892 | `work/phase1/test.jsonl` + model outputs | Hallucination rates at confidence $\ge 0.90$ and CMH meta-analysis | **NO** | Statistical testing only. |
| [src/analysis_idk.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/analysis_idk.py) | 287 | `work/phase1/test.jsonl` + Phase 12 outputs | Compares baseline vs RAG IDK abstention rates on test questions | **NO** | Evaluation only. |
| [src/analysis_adaptive_rag.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/analysis_adaptive_rag.py) | 243, 341 | Phase 5 & 8 outputs (`split == 'test'`) | Evaluates adaptive gating strategies on held-out test rows | **FLAG (Coupled)** | **Clean Separation Verified**: Gates are tuned strictly on `val_pairs` (lines 277–324). Thresholds are applied frozen to `test_pairs` (lines 341–392). |
| [src/phase09_abstention.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/phase09_abstention.py) | 413, 465 | Phase 5 & 8 outputs (`split == 'test'`) | Evaluates selective accuracy and coverage on held-out test rows | **FLAG (Coupled)** | **Clean Separation Verified**: Temperature $T^*$ and coverage thresholds ($\tau_{80}, \tau_{50}$) are fitted strictly on `val_rows` (lines 416–460). Frozen parameters are applied to `test_rows`. |
| [src/analysis_medpsy_reliability.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/analysis_medpsy_reliability.py) | 78, 235 | `outputs/kaggle_qa/**/medpsy-4b.jsonl` (`split == 'test'`) | Evaluates single-pass reliability score on MedQA test and PubMedQA context test | **FLAG (Coupled)** | **Clean Separation Verified**: Scaler, logistic regression, and thresholds fitted strictly on `split == 'validation'` (lines 124–180). Evaluated frozen on test. |
| [src/reliability_score.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/src/reliability_score.py) | 58–72 | Explicit assertions blocking test files | Evaluates SC validation outputs; fits 5-fold CV logistic regression & conformal threshold | **NO (Blocked)** | **Strict Firewall**: Explicitly crashes if any file containing `test.jsonl` is supplied. Validation-only. |
| [apply_reliability_test.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/apply_reliability_test.py) | 13–17, 45–80 | `outputs/sc_test/**/medqa.jsonl`, `outputs/sc_test_s2_pubmedqa/**/pubmedqa.jsonl`, `outputs/sc_test_s3_unanswerable/**/unanswerable_v2.jsonl` | Loads frozen test outputs and predicts with validation-fitted model | **NO (Firewall)** | Fits model on `outputs/sc_val` only; asserts exact match to validation report coefficients; applies validation threshold $\tau=0.5119$ to test without re-fitting. |
| [nli_dev.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/nli_dev.py) | 11, 29 | `outputs/kaggle_build/work/phase1/validation.jsonl` | Samples 100 validation question pairs across models for NLI validation | **NO** | Never touches test data. |
| [nli_medpsy_test.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/nli_medpsy_test.py) | 11, 12, 13 | `outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl`, `work/phase1/test.jsonl`, `work/phase7/evidence.jsonl` | Evaluates NLI entailment on cited passages for MedQA test | **NO** | Uses frozen model and dev-calibrated precision (0.52) and FNR (0.04); zero fitting on test. |
| [kaggle_runner/runner.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/kaggle_runner/runner.py) | 502, 537, 571, 999, 1071 | `work/phase1/test.jsonl` and `data/unanswerable_v2/test.jsonl` | Kaggle GPU execution harness for test generation runs | **NO** | Generation-only harness. |
| [tests/test_unanswerable_v2.py](file:///Users/satyasainathteeparthi/Documents/MTECH/Phase%201%20Medrag/medrag-slm/tests/test_unanswerable_v2.py) | 44, 198 | `data/unanswerable_v2/test.jsonl` and `test.sha256` | Unit tests verifying schema, count, and cryptographic hash | **NO** | Unit test only. |

> [!IMPORTANT]
> **Zero Data Leakage Found**: Across the entire codebase, **no script tunes, fits, or calibrates any parameter using test data**. In scripts where fitting and evaluation co-exist (`analysis_adaptive_rag.py`, `phase09_abstention.py`, `analysis_medpsy_reliability.py`), the validation rows are strictly separated from test rows prior to fitting. The newer self-consistency reliability evaluation architecture (`reliability_score.py` + `apply_reliability_test.py`) establishes an even stronger decoupling by keeping fitting and testing in completely separate execution files.

---

## 4. Complete Audit of Fitted and Tuned Components

Every learned, calibrated, or tuned component in the research project is cataloged below with its fitting script, source split, parameters, and saved location.

### A. Multi-Feature Self-Consistency Reliability Model (MedPsy-4B)
- **Fitting File**: `src/reliability_score.py` (re-verified in `apply_reliability_test.py`)
- **Split Fitted On**: **Validation Split Only** ($N=660$: 200 MedQA val + 200 PubMedQA val + 260 Unanswerable v2 val)
- **Model Family**: `Pipeline([StandardScaler(), LogisticRegression(C=1.0, random_state=42, max_iter=1000)])`
- **Features Used**: `truncation_rate` (+0.6775), `distinct_answers` (+0.4921), `greedy_truncated` (+0.3059), `agreement` (+0.0261), `insufficient_rate` (0.0000), `greedy_insufficient_info` (0.0000)
- **Fitted Intercept**: Standardized: **-0.8850** (Raw: -2.6586)
- **Validation Discrimination**: Stratified 5-Fold Cross-Validation AUROC = **0.7192** (mean fold: 0.7303)
- **Saved Location**: `outputs/sc_val/analysis/reliability_score_report.md`

### B. Calibrated Conformal Abstention Threshold (MedPsy-4B)
- **Fitting File**: `src/reliability_score.py` (`calibrate_conformal_threshold`)
- **Split Fitted On**: **Validation Split Only** (Target constraint: False Abstention Rate $\le 5.0\%$ on the 460 answerable validation questions)
- **Calibrated Operating Threshold**: **$\tau = 0.511881... \approx 0.5119$**
- **Validation Performance at $\tau$**: False Abstention Rate = 5.0% (23/460); Unanswerable Recall = 40.0% (80/200); Abstention F1 = 0.5281; Coverage = 84.4% (557/660); Selective Accuracy = 87.0% (380/437)
- **Application to Test**: Evaluated strictly frozen in `apply_reliability_test.py` $\implies$ Test FAR = 8.56% (89/1,040), Test Recall = 38.0% (38/100 non-controls), Selective Acc = 95.1% (MedQA), 80.4% (PubMedQA)
- **Saved Location**: `outputs/sc_val/analysis/table_abstention_thresholds.md` and `outputs/analysis_test/reliability_test_results.json`

### C. Earlier Single-Pass MedPsy Reliability Model
- **Fitting File**: `src/analysis_medpsy_reliability.py`
- **Split Fitted On**: **Validation Split Only** (MedQA val 200 + unanswerable v1 val 50; PubMedQA val 200)
- **Features Used**: `truncated`, `token_frac`, `confidence`, `margin`, `entropy`, `top_rerank`
- **Saved Model Weights**: `outputs/analysis/medpsy_reliability_model.json`
- **Thresholds**: MedQA $\tau_{\text{safe}} = 0.2942$; PubMedQA context $\tau_{\text{safe}} = 0.2293$

### D. Adaptive RAG Decision Gates (5 Standard SLMs)
- **Fitting File**: `src/analysis_adaptive_rag.py`
- **Split Fitted On**: **Validation Split Only** ($N=400$ answerable: 200 MedQA val + 200 PubMedQA val)
- **Parameters Fitted**:
  - `rerank_gate`: threshold $\tau$ on `top_rerank_score` (tuned per model: e.g., $\tau \in [5.8, 6.2]$)
  - `confidence_gate`: pick higher confidence between baseline and RAG
  - `combined`: threshold $\tau$ on reranking score plus margin $\delta$ on confidence
- **Saved Location**: `outputs/analysis/adaptive_rag_{model}.json`

### E. Phase 9 Temperature Scaling & Confidence Thresholds (5 Standard SLMs)
- **Fitting File**: `src/phase09_abstention.py`
- **Split Fitted On**: **Validation Split Only** ($N=450$: 400 answerable + 50 unanswerable v1)
- **Parameters Fitted**:
  - Temperature $T^*$ minimizing NLL on validation answerable predictions ($T^* \in [3.7, 26.1]$)
  - Operating thresholds $\tau_{80}$ and $\tau_{50}$ targeting 80% and 50% coverage on validation
  - Maximum F1 threshold $\tau_{F1}$ on validation unanswerables
- **Saved Location**: `outputs/analysis/phase9_abstention.json` and `phase9_abstention.md`

### F. NLI Citation Verification Threshold & Operating Parameters
- **Fitting / Dev File**: `nli_dev.py` and `nli_agreement.py`
- **Split Validated On**: **Validation Split Only** (100 sampled validation questions from `validation.jsonl`)
- **Frozen Configuration**: Model = `pritamdeka/PubMedBERT-MNLI-MedNLI`; Classification threshold $\tau = 0.5$
- **Dev-Calibrated Operating Parameters**: Empirical precision = $13/25 = 0.52$ (95% CI: [0.335, 0.70]); Empirical FNR = $1 - 24/25 = 0.04$ (95% CI: [0.007, 0.195])
- **Saved Location**: `outputs/analysis_nli/dev_pairs_scored.json` and `outputs/analysis_nli/medpsy_test_citation_metrics.json`

### G. Prompt Engineering and Decoding Templates
- **Definition Files**: `src/qa_pipeline.py` and `kaggle_runner/runner.py`
- **Split Developed On**: Pilot experiments and validation sets; frozen prior to all full benchmark test executions.

---

## 5. Uncommitted and Gitignored Outputs Critical to the Project

Inspection of `.gitignore` reveals the following rule:
```gitignore
data/
outputs/*
!outputs/analysis/
!outputs/figures/
__pycache__/
*.pyc
.venv/
.env
kaggle.json
review_unanswerable_v2.md
```

Because `outputs/*` is ignored by default (with exceptions only for `!outputs/analysis/` and `!outputs/figures/`), **the majority of raw experiment outputs and intermediate build artifacts are untracked by Git**.

### Critical Gitignored Artifacts
1. **Raw Question-Answering Generations (Phase 5, 8, 10, 12)**:
   - `outputs/kaggle_qa/full/`: All raw test and validation JSONL files for `qwen3-1.7b`, `phi4-mini`, `smollm3-3b`, `qwen3-4b`, and `gemma3-4b` (~100 MB).
   - `outputs/kaggle_qa/context/`: Raw PubMedQA +Abstract JSONL files for all 6 models (~4 MB).
   - `outputs/kaggle_qa/medpsy_final/` and `medpsy_rag/`: Raw test JSONL files for `medpsy-4b` (~5 MB).
   - `outputs/kaggle_qa/idk/`: Raw IDK abstention experiment outputs (~3 MB).
2. **Self-Consistency Raw Traces**:
   - `outputs/sc_val/medpsy-4b/`: Raw validation JSONL files (660 questions x 5 samples + greedy = ~420 KB).
   - `outputs/sc_test/medpsy-4b/outputs/sc_test/medpsy-4b/medqa.jsonl` (~309 KB).
   - `outputs/sc_test_s2_pubmedqa/outputs/sc_test/medpsy-4b/pubmedqa.jsonl` (~312 KB).
   - `outputs/sc_test_s3_unanswerable/outputs/sc_test/medpsy-4b/unanswerable_v2.jsonl` (~92 KB).
3. **Execution Logs and Timing Sources**:
   - `outputs/kaggle_build/medrag-build.log` (267 KB): Contains exact phase 6 (retrieval) and phase 7 (reranking) wall clock execution times used to calculate the 0.1273 s/question retrieval overhead cited across all tables.
   - `outputs/kaggle_qa/medpsy_final/medrag-slm-runner.log` (504 KB).
   - `outputs/sc_test_s2_pubmedqa/medrag-slm-runner.log` (266 KB).
   - `outputs/sc_test_s3_unanswerable/medrag-slm-runner.log` (260 KB).
4. **Intermediate Knowledge Base and Index Data**:
   - `outputs/kaggle_build/work/phase1/test.jsonl` and `validation.jsonl` (Phase 1 master question splits).
   - `outputs/kaggle_build/work/phase2/corpus.jsonl` (124,077-chunk textbook corpus).
   - `outputs/kaggle_build/work/phase7/evidence.jsonl` (top-5 reranked evidence passages joined in NLI evaluation).
5. **Secondary Analysis Directories**:
   - `outputs/analysis_test/reliability_test_results.json` (2 KB): Ignored because `!outputs/analysis/` does not unignore `outputs/analysis_test/`!
   - `outputs/analysis_nli/` (`medpsy_test_citation_metrics.json`, `dev_pairs_scored.json`, `label_sheet.csv`): Ignored under `outputs/*`.

### Risk Assessment
If the local workspace is wiped or cloned fresh from GitHub:
- The presentation markdown tables in `outputs/analysis/` will be preserved, but **no raw answers can be verified, no audit scripts can re-run, and no new metrics can be extracted without re-running Kaggle GPU inference**.
- **Action Required**: The final test and validation JSONL files, `outputs/analysis_test/`, `outputs/analysis_nli/`, and the build log `medrag-build.log` must either be added to Git using `git add -f` or archived as a permanent release artifact (e.g. Zenodo / OSF / Kaggle Dataset).

---

## 6. Open Research and Technical Questions

The following open questions must be resolved for final paper drafting and audit sign-off:

### Question 1: Archival of Raw Test Output Files
> The repository's `.gitignore` currently leaves all raw test JSONLs (`outputs/kaggle_qa/`, `outputs/sc_test*/`, `outputs/analysis_test/`, `outputs/analysis_nli/`) untracked. Should we update `.gitignore` or create an explicit release archive script to ensure all paper claims are verifiable from the GitHub repository?

### Question 2: Scope of the 5 Standard SLMs on the Unanswerable Benchmark
> In the paper, `medpsy-4b` is evaluated on `unanswerable_v2` ($N=140$), whereas the 5 standard SLMs (`qwen3-1.7b`, `phi4-mini`, `smollm3-3b`, `qwen3-4b`, `gemma3-4b`) were evaluated on `unanswerable_v1` ($N=150$) in Phase 5/8/9. Should the final comparison table report `unanswerable_v1` for the 5 SLMs alongside `unanswerable_v2` for MedPsy (with clear footnoting), or do you plan to run the 5 SLMs on `unanswerable_v2` before submission?

### Question 3: MedPsy-4B Baseline PubMedQA Evaluation
> MedPsy-4B currently has test results for MedQA (baseline and RAG) and PubMedQA (+Abstract oracle context mode), but no closed-book baseline run on PubMedQA without the abstract. Is this intentional because PubMedQA is primarily utilized as an oracle context benchmark in the literature, or should a zero-context PubMedQA baseline be scheduled?

### Question 4: Discrepancy in `data_audit.md` Option Labels [RESOLVED — TASK 2 AUDIT FIX]
> **Resolution Note**: `src/data_audit.py` (lines 210–212 and line 630) was patched to accept the canonical A/B/C letter mapping (`A=yes`, `B=no`, `C=maybe`) generated by `phase01_datasets.py` alongside the literal string values `{"yes", "no", "maybe"}`. Neither the dataset generator nor the QA pipeline was modified. Re-running `python -m src.data_audit` now reports 100.0% Answer Format Compliance across both test and validation splits without false-alarm schema deviations.

### Question 5: Evaluation of NLI Citation Metric on the Other 5 Models
> `nli_medpsy_test.py` currently evaluates citation precision and answer support exclusively on `medpsy-4b` ($N=500$ MedQA test). The 5 standard SLMs already have top-5 evidence citations in their Phase 8 RAG test outputs. Should we execute the frozen NLI evaluation script on the other 5 models' Phase 8 outputs (pure CPU inference, ~1 hour per model) to produce a 6-model citation faithfulness table for the paper?

