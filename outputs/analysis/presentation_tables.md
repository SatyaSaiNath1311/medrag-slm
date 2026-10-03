# Medical RAG Small Language Model (SLM) Benchmark: Presentation Tables

Comprehensive empirical evaluation comparing open-weight small language models (1.7B to 4B parameters) and medical reasoning models across **Baseline (parametric-only)**, **Full RAG**, **Best Adaptive Gate**, and **+Abstract (PubMedQA oracle context)**.

## Table 1: Question-Answering Performance across Architectures and Inference Modes
> **Setup**: Evaluated on the held-out test split ($N=1,000$ answerable: 500 MedQA + 500 PubMedQA).  
> **Metrics**: Accuracy, Exact Match (EM)*, and Macro-F1 across discrete answer option classes ($C=4$ for MedQA, $C=3$ for PubMedQA).  
> **Uncertainty**: 95% percentile bootstrap confidence intervals (1,000 resamples, seed 42).  
> *Notes*:  
> 1. **Exact Match (EM)** is mathematically identical to accuracy for single-token multiple-choice options.  
> 2. **BERTScore** is not applicable to single-letter multiple-choice answers and is omitted.  
> 3. **+Abstract** provides the ground-truth study abstract for PubMedQA (standard oracle context benchmark).

| Model | Variant | Overall Acc [95% CI] | Overall EM | Overall Macro-F1 [95% CI] | MedQA Acc [95% CI] | MedQA EM | MedQA Macro-F1 [95% CI] | PubMedQA Acc [95% CI] | PubMedQA EM | PubMedQA Macro-F1 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| **medpsy-4b** | Baseline | n/a | n/a | n/a | 87.6% [0.846, 0.904] | 87.6% | 0.875 [0.845, 0.903] | n/a | n/a | n/a |
| **medpsy-4b** | +Abstract (PubMedQA) | — | — | — | — | — | — | 78.0% [0.746, 0.814] | 78.0% | 0.630 [0.583, 0.679] |
| **qwen3-4b** | Baseline | 55.4% [0.523, 0.584] | 55.4% | 0.520 [0.486, 0.553] | 58.4% [0.538, 0.626] | 58.4% | 0.582 [0.536, 0.622] | 52.4% [0.478, 0.568] | 52.4% | 0.355 [0.313, 0.397] |
| **qwen3-4b** | Full RAG | 54.4% [0.513, 0.574] | 54.4% | 0.526 [0.490, 0.561] | 60.4% [0.560, 0.648] | 60.4% | 0.597 [0.552, 0.640] | 48.4% [0.442, 0.528] | 48.4% | 0.403 [0.361, 0.446] |
| **qwen3-4b** | Adaptive (Combined Gate) | 58.3% [0.553, 0.614] | 58.3% | 0.555 [0.519, 0.589] | 61.4% [0.572, 0.660] | 61.4% | 0.610 [0.567, 0.656] | 55.2% [0.512, 0.596] | 55.2% | 0.413 [0.372, 0.458] |
| **qwen3-4b** | +Abstract (PubMedQA) | — | — | — | — | — | — | 74.6% [0.710, 0.784] | 74.6% | 0.584 [0.538, 0.631] |
| **phi4-mini** | Baseline | 50.4% [0.471, 0.535] | 50.4% | 0.475 [0.441, 0.508] | 53.4% [0.492, 0.578] | 53.4% | 0.530 [0.486, 0.574] | 47.4% [0.432, 0.518] | 47.4% | 0.376 [0.335, 0.417] |
| **phi4-mini** | Full RAG | 51.9% [0.488, 0.550] | 51.9% | 0.503 [0.469, 0.535] | 58.8% [0.542, 0.628] | 58.8% | 0.584 [0.539, 0.625] | 45.0% [0.410, 0.496] | 45.0% | 0.353 [0.317, 0.392] |
| **phi4-mini** | Adaptive (Confidence Gate) | 55.1% [0.521, 0.583] | 55.1% | 0.526 [0.491, 0.558] | 58.8% [0.544, 0.630] | 58.8% | 0.584 [0.539, 0.627] | 51.4% [0.474, 0.560] | 51.4% | 0.410 [0.369, 0.456] |
| **phi4-mini** | +Abstract (PubMedQA) | — | — | — | — | — | — | 73.6% [0.698, 0.776] | 73.6% | 0.556 [0.518, 0.602] |
| **gemma3-4b** | Baseline | 48.9% [0.458, 0.519] | 48.9% | 0.448 [0.414, 0.482] | 48.6% [0.442, 0.526] | 48.6% | 0.483 [0.439, 0.524] | 49.2% [0.450, 0.538] | 49.2% | 0.369 [0.326, 0.413] |
| **gemma3-4b** | Full RAG | 51.6% [0.486, 0.547] | 51.6% | 0.486 [0.451, 0.519] | 53.0% [0.484, 0.574] | 53.0% | 0.528 [0.481, 0.571] | 50.2% [0.460, 0.548] | 50.2% | 0.366 [0.328, 0.407] |
| **gemma3-4b** | Adaptive (Rerank Gate) | 51.5% [0.485, 0.548] | 51.5% | 0.471 [0.438, 0.504] | 53.0% [0.486, 0.572] | 53.0% | 0.526 [0.479, 0.570] | 50.0% [0.460, 0.546] | 50.0% | 0.361 [0.322, 0.404] |
| **gemma3-4b** | +Abstract (PubMedQA) | — | — | — | — | — | — | 68.8% [0.650, 0.730] | 68.8% | 0.512 [0.475, 0.551] |
| **qwen3-1.7b** | Baseline | 46.2% [0.432, 0.493] | 46.2% | 0.445 [0.410, 0.479] | 44.6% [0.402, 0.488] | 44.6% | 0.444 [0.398, 0.486] | 47.8% [0.434, 0.518] | 47.8% | 0.340 [0.310, 0.367] |
| **qwen3-1.7b** | Full RAG | 47.9% [0.449, 0.510] | 47.9% | 0.468 [0.434, 0.502] | 50.6% [0.464, 0.550] | 50.6% | 0.504 [0.461, 0.547] | 45.2% [0.410, 0.496] | 45.2% | 0.320 [0.292, 0.350] |
| **qwen3-1.7b** | Adaptive (Combined Gate) | 49.9% [0.469, 0.530] | 49.9% | 0.487 [0.452, 0.521] | 51.4% [0.470, 0.560] | 51.4% | 0.512 [0.467, 0.556] | 48.4% [0.440, 0.530] | 48.4% | 0.343 [0.314, 0.375] |
| **qwen3-1.7b** | +Abstract (PubMedQA) | — | — | — | — | — | — | 66.2% [0.618, 0.704] | 66.2% | 0.468 [0.427, 0.511] |
| **smollm3-3b** | Baseline | 47.2% [0.441, 0.503] | 47.2% | 0.435 [0.402, 0.468] | 41.6% [0.374, 0.456] | 41.6% | 0.414 [0.372, 0.456] | 52.8% [0.484, 0.574] | 52.8% | 0.356 [0.327, 0.386] |
| **smollm3-3b** | Full RAG | 49.9% [0.469, 0.529] | 49.9% | 0.473 [0.439, 0.506] | 47.2% [0.430, 0.514] | 47.2% | 0.472 [0.430, 0.513] | 52.6% [0.486, 0.572] | 52.6% | 0.366 [0.335, 0.398] |
| **smollm3-3b** | Adaptive (Rerank Gate) | 50.0% [0.470, 0.531] | 50.0% | 0.465 [0.431, 0.500] | 45.2% [0.410, 0.498] | 45.2% | 0.451 [0.406, 0.496] | 54.8% [0.504, 0.596] | 54.8% | 0.374 [0.344, 0.406] |
| **smollm3-3b** | +Abstract (PubMedQA) | — | — | — | — | — | — | 72.4% [0.686, 0.764] | 72.4% | 0.505 [0.481, 0.529] |

## Table 2: System Reliability, Hallucination, and Evidence Grounding (MedQA)
> **Setup**: Grounding and hallucination evaluation on MedQA test answerable ($N=500$) and unanswerable ($N=150$) sets.  
> **Metrics Definition**:  
> - **Faithfulness**: % of answers whose cited passage explicitly contains the selected option text (string-matching proxy). For Adaptive gates, computed *strictly over final answers routed to RAG*.  
> - **Wrong Answers ($N$)**: Total count of incorrect predictions on the MedQA test set ($N=500$).  
> - **Hallucination Rate (Wrong Answers)**: % of incorrect predictions carrying raw confidence $\ge 0.90$.  
> - **Hallucination Rate (Unanswerable)**: % of unanswerable questions answered with raw confidence $\ge 0.90$.  
> - **Citation Validity %**: % of bracketed citations pointing to valid retrieved passage numbers (1–5).  
> *Notes*:  
> 1. Grounding metrics utilize deterministic string-matching proxies (exact substring or first word removed), not LLM-judged RAGAS.  
> 2. Adaptive gates choose the higher-confidence answer, so the conditional hallucination rate among wrong answers is inflated by construction.

### Retrieval Performance (Shared across all models on MedQA)
| Metric | Value | Scope | Description |
|---|---|---|---|
| **Context Precision** | **12.9%** | Top-5 passages | % of 5 retrieved passages containing the correct gold option text (mean: 0.64 / 5 passages) |
| **Context Recall (Default Rule)** | **30.4%** | Any of top-5 | % of questions with the gold option present in *any* of the 5 retrieved passages (152 / 500) |
| **Strict Verbatim Recall** | **18.8%** | Any of top-5 | % of questions with the verbatim option string present in *any* passage (94 / 500) |

### Model Reliability and Hallucination Breakdown
| Model | Variant | Wrong Answers ($N$) | Faithfulness % (Count / $N_{\text{RAG}}$) | Hallucination Rate: Wrong $\ge 0.90$ % (Count/$N$) | Hallucination Rate: Unanswerable $\ge 0.90$ % (Count/150) | Citation Validity % |
|---|---|---|---|---|---|---|
| **medpsy-4b** | Baseline | 62 | — | 85.5% (53/62) | 86.0% (129/150) | — |
| **qwen3-4b** | Baseline | 208 | — | 82.7% (172/208) | 82.7% (124/150) | — |
| **qwen3-4b** | Full RAG | 198 | 31.2% (156/500) | 82.8% (164/198) | 88.0% (132/150) | 100.0% |
| **qwen3-4b** | Adaptive (Combined Gate) | 193 | 33.3% (117/351) | 95.3% (184/193) | 82.7% (124/150) | 100.0% |
| **phi4-mini** | Baseline | 233 | — | 18.9% (44/233) | 32.7% (49/150) | — |
| **phi4-mini** | Full RAG | 206 | 26.8% (134/500) | 21.8% (45/206) | 33.3% (50/150) | 100.0% |
| **phi4-mini** | Adaptive (Confidence Gate) | 206 | 29.7% (94/316) | 32.5% (67/206) | 32.7% (49/150) | 100.0% |
| **gemma3-4b** | Baseline | 257 | — | 95.3% (245/257) | 95.3% (143/150) | — |
| **gemma3-4b** | Full RAG | 235 | 32.0% (160/500) | 93.6% (220/235) | 94.0% (141/150) | 100.0% |
| **gemma3-4b** | Adaptive (Rerank Gate) | 235 | 38.1% (98/257) | 94.5% (222/235) | 95.3% (143/150) | 100.0% |
| **qwen3-1.7b** | Baseline | 277 | — | 84.8% (235/277) | 88.0% (132/150) | — |
| **qwen3-1.7b** | Full RAG | 247 | 21.0% (105/500) | 86.2% (213/247) | 88.7% (133/150) | 100.0% |
| **qwen3-1.7b** | Adaptive (Combined Gate) | 243 | 25.7% (79/307) | 97.9% (238/243) | 88.0% (132/150) | 100.0% |
| **smollm3-3b** | Baseline | 292 | — | 36.0% (105/292) | 48.0% (72/150) | — |
| **smollm3-3b** | Full RAG | 264 | 21.8% (109/500) | 33.0% (87/264) | 40.7% (61/150) | 100.0% |
| **smollm3-3b** | Adaptive (Rerank Gate) | 274 | 25.0% (80/320) | 33.2% (91/274) | 48.0% (72/150) | 100.0% |

## Table 3: Efficiency, Generation Throughput, and Computational Footprint
> **Setup**: Evaluated on the answerable test questions ($N=1,000$ for full benchmarks, $N=500$ for PubMedQA +Abstract).  
> **Latency Details**:  
> - **Baseline**: Pure parametric forward pass latency.  
> - **Full RAG**: Includes generation pass + **0.127 s retrieval overhead** (BM25 + MedCPT dense (FAISS) + MedCPT cross-encoder rerank from Phase 6 & 7 build logs).  
> - **Adaptive Gates**: Incorporates retrieval overhead and selective single/double generation passes based on gate logic; peak memory corresponds to Full RAG when retrieval is triggered.  
> - **Peak VRAM**: Maximum GPU memory allocated (`torch.cuda.max_memory_allocated`) summed across GPUs (GB, 1 decimal); reserved memory is excluded because it includes cached allocations from earlier modes.  
> - **Peak RAM**: Process resident set size (`peak_process_rss_mb`, GB); all models profiled sequentially in one process, so RSS includes residual allocations from previously loaded models — treat as an upper bound.  
> *Footnotes*:  
> 1. **Timing source**: Latency and throughput are computed over full test runs ($N=1,000$ questions); profiling used 20 questions per mode (10 for context) on Kaggle 2×T4.  
> 2. **gemma3-4b**: Loaded in float32 across 2×T4; RAG used batch size 2 vs 8 for baseline, so peak VRAM is not directly comparable across its modes.

| Model | Variant | Avg Latency (s/q) | End-to-End Tokens/s (includes prompt processing) | Mean Prompt Tokens | Parse Rate | Trunc Rate | Peak VRAM | Peak RAM |
|---|---|---|---|---|---|---|---|---|
| **medpsy-4b** | Baseline | 37.303 s | 15.1 tok/s | 305.8 | 90.4% | 10.6% | pending | pending |
| **medpsy-4b** | +Abstract (PubMedQA) | 30.279 s | 16.1 tok/s | 496.8 | 99.8% | 0.2% | pending | pending |
| **qwen3-4b** | Baseline | 0.223 s | 71.8 tok/s | 161.2 | 100.0% | 0.0% | 8.4 GB | 3.5 GB |
| **qwen3-4b** | Full RAG | 1.579 s | 12.7 tok/s | 1200.1 | 100.0% | 0.0% | 10.2 GB | 3.5 GB |
| **qwen3-4b** | Adaptive (Combined Gate) | 1.802 s | 17.3 tok/s | 843.9 | 100.0% | 0.0% | 10.2 GB | 3.5 GB |
| **qwen3-4b** | +Abstract (PubMedQA) | 0.406 s | 21.8 tok/s | 440.8 | 100.0% | 0.0% | 8.7 GB | 3.5 GB |
| **phi4-mini** | Baseline | 0.162 s | 60.5 tok/s | 146.2 | 100.0% | 0.0% | 8.0 GB | 4.3 GB |
| **phi4-mini** | Full RAG | 1.256 s | 20.1 tok/s | 1138.9 | 100.0% | 0.0% | 9.6 GB | 4.3 GB |
| **phi4-mini** | Adaptive (Confidence Gate) | 1.418 s | 23.9 tok/s | 722.5 | 100.0% | 0.0% | 9.6 GB | 4.3 GB |
| **phi4-mini** | +Abstract (PubMedQA) | 0.485 s | 25.4 tok/s | 411.8 | 100.0% | 0.0% | 8.3 GB | 4.3 GB |
| **gemma3-4b** | Baseline | 0.611 s | 18.7 tok/s | 157.8 | 100.0% | 0.0% | 18.4 GB | 5.8 GB |
| **gemma3-4b** | Full RAG | 3.449 s | 3.4 tok/s | 1144.7 | 100.0% | 0.0% | 17.6 GB | 5.6 GB |
| **gemma3-4b** | Adaptive (Rerank Gate) | 1.842 s | 6.5 tok/s | 555.9 | 100.0% | 0.0% | 17.6 GB | 5.6 GB |
| **gemma3-4b** | +Abstract (PubMedQA) | 1.422 s | 5.0 tok/s | 431.6 | 100.0% | 0.0% | 16.8 GB | 5.6 GB |
| **qwen3-1.7b** | Baseline | 0.117 s | 136.6 tok/s | 161.2 | 100.0% | 0.0% | 4.5 GB | 6.2 GB |
| **qwen3-1.7b** | Full RAG | 0.823 s | 52.3 tok/s | 1200.1 | 100.0% | 0.0% | 5.8 GB | 6.2 GB |
| **qwen3-1.7b** | Adaptive (Combined Gate) | 0.940 s | 59.0 tok/s | 839.9 | 100.0% | 0.0% | 5.8 GB | 6.2 GB |
| **qwen3-1.7b** | +Abstract (PubMedQA) | 0.333 s | 106.4 tok/s | 440.8 | 100.0% | 0.0% | 4.7 GB | 6.2 GB |
| **smollm3-3b** | Baseline | 0.162 s | 98.6 tok/s | 213.5 | 100.0% | 0.0% | 6.4 GB | 5.2 GB |
| **smollm3-3b** | Full RAG | 1.067 s | 42.3 tok/s | 1225.1 | 100.0% | 0.0% | 7.6 GB | 5.2 GB |
| **smollm3-3b** | Adaptive (Rerank Gate) | 0.691 s | 49.6 tok/s | 736.0 | 100.0% | 0.0% | 7.6 GB | 5.2 GB |
| **smollm3-3b** | +Abstract (PubMedQA) | 0.464 s | 32.5 tok/s | 480.2 | 100.0% | 0.0% | 6.6 GB | 5.0 GB |

4 of 5 SLM models run on a single 16 GB T4 including RAG; RAG adds ~1.2–1.8 GB VRAM. MedPsy-4B utilizes test-time reasoning (~1,024 max tokens).

## Table 4: MedPsy-4B Baseline vs SLM Baselines Pairwise Comparison (MedQA Test, $N=500$)
> **Setup**: Exact two-sided binomial McNemar test comparing `medpsy-4b` baseline against standard SLM baselines on the identical held-out MedQA test questions ($N=500$).  
> **Contingency Matrix**: $b$ = MedPsy correct & Other incorrect; $c$ = MedPsy incorrect & Other correct.  
> **Significance**: *** $p < 0.001$, ** $p < 0.01$, * $p < 0.05$.

| Comparison Baseline | MedPsy-4B Acc [95% CI] | Other Model Acc [95% CI] | Δ Acc (pts) | MedPsy+ / Other- ($b$) | MedPsy- / Other+ ($c$) | McNemar $p$-value | Significance |
|---|---|---|---|---|---|---|---|
| vs **qwen3-4b** | 87.6% [0.846, 0.904] | 58.4% [0.540, 0.624] | +29.2 | 176 | 30 | 2.49e-26 | *** |
| vs **phi4-mini** | 87.6% [0.846, 0.904] | 53.4% [0.488, 0.578] | +34.2 | 189 | 18 | 3.81e-37 | *** |
| vs **gemma3-4b** | 87.6% [0.846, 0.904] | 48.6% [0.440, 0.532] | +39.0 | 217 | 22 | 1.75e-41 | *** |
| vs **qwen3-1.7b** | 87.6% [0.846, 0.904] | 44.6% [0.402, 0.488] | +43.0 | 235 | 20 | 9.75e-48 | *** |
| vs **smollm3-3b** | 87.6% [0.846, 0.904] | 41.6% [0.374, 0.460] | +46.0 | 250 | 20 | 9.70e-52 | *** |
