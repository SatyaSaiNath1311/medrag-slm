# Extended Benchmark Metrics: QA Classification, Retrieval, Reliability, and Calibration

> **Evaluation Context**: Held-out test split ($N=1,000$ answerable: 500 MedQA-USMLE + 500 PubMedQA).  
> **Statistical Rigor**: 95% confidence intervals generated via percentile bootstrap (1,000 resamples, seed 42).  
> **Models**: Evaluated across 6 small language models, including `medpsy-4b` (reasoning model, baseline & +Abstract).  
> Missing modes and unsupported experimental splits are gracefully designated as `n/a`.

## 1. QA Classification Performance

> Multi-class precision, recall, and F1 across answer choices.  
> - **MedQA (4-option)**: Labels A, B, C, D.  
> - **PubMedQA (3-option)**: Labels yes (A), no (B), maybe (C).  
> - **Micro-F1**: Mathematically equivalent to overall Accuracy for single-label multiple-choice questions.

### Table 1A: MedQA Test Split ($N=500$) Multi-Class Performance
| Model | Mode | Micro-F1 (Acc) [95% CI] | Macro-F1 [95% CI] | Weighted-F1 [95% CI] | P (A/B/C/D) | R (A/B/C/D) | F1 (A/B/C/D) |
|---|---|---|---|---|---|---|---|
| **medpsy-4b** | Baseline | 0.876 [0.846, 0.904] | 0.875 [0.845, 0.903] | 0.876 [0.845, 0.904] | 0.87/0.85/0.90/0.90 | 0.89/0.92/0.85/0.84 | 0.88/0.88/0.87/0.87 |
| **medpsy-4b** | RAG | n/a | n/a | n/a | n/a | n/a | n/a |
| **medpsy-4b** | Adaptive | n/a | n/a | n/a | n/a | n/a | n/a |
| **medpsy-4b** | +Abstract | n/a | n/a | n/a | n/a | n/a | n/a |
| **qwen3-4b** | Baseline | 0.584 [0.538, 0.626] | 0.582 [0.536, 0.622] | 0.583 [0.537, 0.625] | 0.62/0.60/0.57/0.53 | 0.58/0.66/0.54/0.54 | 0.60/0.63/0.55/0.54 |
| **qwen3-4b** | RAG | 0.604 [0.560, 0.648] | 0.597 [0.552, 0.640] | 0.601 [0.558, 0.646] | 0.60/0.61/0.64/0.56 | 0.65/0.70/0.56/0.48 | 0.63/0.65/0.60/0.52 |
| **qwen3-4b** | Adaptive | 0.614 [0.572, 0.660] | 0.610 [0.567, 0.656] | 0.613 [0.571, 0.658] | 0.62/0.60/0.67/0.55 | 0.62/0.72/0.58/0.52 | 0.62/0.66/0.62/0.54 |
| **qwen3-4b** | +Abstract | n/a | n/a | n/a | n/a | n/a | n/a |
| **phi4-mini** | Baseline | 0.534 [0.492, 0.578] | 0.530 [0.486, 0.574] | 0.535 [0.491, 0.580] | 0.66/0.51/0.50/0.46 | 0.54/0.61/0.51/0.46 | 0.60/0.56/0.51/0.46 |
| **phi4-mini** | RAG | 0.588 [0.542, 0.628] | 0.584 [0.539, 0.625] | 0.590 [0.544, 0.631] | 0.73/0.58/0.58/0.47 | 0.57/0.62/0.63/0.51 | 0.64/0.60/0.61/0.49 |
| **phi4-mini** | Adaptive | 0.588 [0.544, 0.630] | 0.584 [0.539, 0.627] | 0.589 [0.546, 0.632] | 0.73/0.56/0.57/0.50 | 0.57/0.66/0.60/0.50 | 0.64/0.61/0.59/0.50 |
| **phi4-mini** | +Abstract | n/a | n/a | n/a | n/a | n/a | n/a |
| **gemma3-4b** | Baseline | 0.486 [0.442, 0.526] | 0.483 [0.439, 0.524] | 0.486 [0.442, 0.528] | 0.52/0.51/0.49/0.41 | 0.53/0.50/0.44/0.47 | 0.53/0.50/0.47/0.44 |
| **gemma3-4b** | RAG | 0.530 [0.484, 0.574] | 0.528 [0.481, 0.571] | 0.531 [0.485, 0.575] | 0.65/0.52/0.49/0.46 | 0.54/0.60/0.49/0.50 | 0.59/0.56/0.49/0.48 |
| **gemma3-4b** | Adaptive | 0.530 [0.486, 0.572] | 0.526 [0.479, 0.570] | 0.530 [0.484, 0.573] | 0.64/0.52/0.52/0.43 | 0.57/0.61/0.45/0.48 | 0.60/0.57/0.48/0.45 |
| **gemma3-4b** | +Abstract | n/a | n/a | n/a | n/a | n/a | n/a |
| **qwen3-1.7b** | Baseline | 0.446 [0.402, 0.488] | 0.444 [0.398, 0.486] | 0.443 [0.398, 0.486] | 0.50/0.44/0.43/0.43 | 0.34/0.53/0.49/0.44 | 0.40/0.48/0.46/0.43 |
| **qwen3-1.7b** | RAG | 0.506 [0.464, 0.550] | 0.504 [0.461, 0.547] | 0.505 [0.463, 0.549] | 0.56/0.53/0.46/0.48 | 0.44/0.56/0.57/0.45 | 0.50/0.54/0.51/0.47 |
| **qwen3-1.7b** | Adaptive | 0.514 [0.470, 0.560] | 0.512 [0.467, 0.556] | 0.513 [0.468, 0.558] | 0.57/0.54/0.47/0.49 | 0.45/0.56/0.58/0.47 | 0.50/0.55/0.52/0.48 |
| **qwen3-1.7b** | +Abstract | n/a | n/a | n/a | n/a | n/a | n/a |
| **smollm3-3b** | Baseline | 0.416 [0.374, 0.456] | 0.414 [0.372, 0.456] | 0.412 [0.369, 0.454] | 0.54/0.50/0.41/0.31 | 0.26/0.53/0.39/0.52 | 0.35/0.52/0.40/0.39 |
| **smollm3-3b** | RAG | 0.472 [0.430, 0.514] | 0.472 [0.430, 0.513] | 0.472 [0.429, 0.514] | 0.66/0.51/0.50/0.35 | 0.33/0.55/0.45/0.60 | 0.44/0.53/0.48/0.44 |
| **smollm3-3b** | Adaptive | 0.452 [0.410, 0.498] | 0.451 [0.406, 0.496] | 0.450 [0.404, 0.496] | 0.61/0.52/0.47/0.33 | 0.30/0.57/0.42/0.56 | 0.40/0.54/0.45/0.41 |
| **smollm3-3b** | +Abstract | n/a | n/a | n/a | n/a | n/a | n/a |

### Table 1B: PubMedQA Test Split ($N=500$) Multi-Class Performance
| Model | Mode | Micro-F1 (Acc) [95% CI] | Macro-F1 [95% CI] | Weighted-F1 [95% CI] | P (yes/no/maybe) | R (yes/no/maybe) | F1 (yes/no/maybe) |
|---|---|---|---|---|---|---|---|
| **medpsy-4b** | Baseline | n/a | n/a | n/a | n/a | n/a | n/a |
| **medpsy-4b** | RAG | n/a | n/a | n/a | n/a | n/a | n/a |
| **medpsy-4b** | Adaptive | n/a | n/a | n/a | n/a | n/a | n/a |
| **medpsy-4b** | +Abstract | 0.780 [0.746, 0.814] | 0.630 [0.583, 0.679] | 0.760 [0.721, 0.798] | 0.80/0.80/0.38 | 0.88/0.84/0.17 | 0.84/0.82/0.23 |
| **qwen3-4b** | Baseline | 0.524 [0.478, 0.568] | 0.355 [0.313, 0.397] | 0.482 [0.430, 0.532] | 0.59/0.45/0.11 | 0.80/0.21/0.08 | 0.68/0.29/0.09 |
| **qwen3-4b** | RAG | 0.484 [0.442, 0.528] | 0.403 [0.361, 0.446] | 0.489 [0.448, 0.533] | 0.61/0.42/0.17 | 0.55/0.49/0.18 | 0.58/0.45/0.18 |
| **qwen3-4b** | Adaptive | 0.552 [0.512, 0.596] | 0.413 [0.372, 0.458] | 0.532 [0.488, 0.577] | 0.62/0.49/0.17 | 0.74/0.40/0.10 | 0.68/0.44/0.12 |
| **qwen3-4b** | +Abstract | 0.746 [0.710, 0.784] | 0.584 [0.538, 0.631] | 0.722 [0.681, 0.766] | 0.76/0.79/0.31 | 0.91/0.69/0.13 | 0.83/0.74/0.19 |
| **phi4-mini** | Baseline | 0.474 [0.432, 0.518] | 0.376 [0.335, 0.417] | 0.474 [0.430, 0.520] | 0.61/0.47/0.14 | 0.66/0.24/0.27 | 0.63/0.32/0.18 |
| **phi4-mini** | RAG | 0.450 [0.410, 0.496] | 0.353 [0.317, 0.392] | 0.457 [0.416, 0.502] | 0.60/0.39/0.08 | 0.53/0.45/0.08 | 0.56/0.42/0.08 |
| **phi4-mini** | Adaptive | 0.514 [0.474, 0.560] | 0.410 [0.369, 0.456] | 0.513 [0.472, 0.563] | 0.63/0.45/0.15 | 0.65/0.41/0.17 | 0.64/0.43/0.16 |
| **phi4-mini** | +Abstract | 0.736 [0.698, 0.776] | 0.556 [0.518, 0.602] | 0.708 [0.666, 0.755] | 0.78/0.71/0.21 | 0.84/0.80/0.07 | 0.81/0.75/0.10 |
| **gemma3-4b** | Baseline | 0.492 [0.450, 0.538] | 0.369 [0.326, 0.413] | 0.468 [0.422, 0.518] | 0.62/0.53/0.16 | 0.74/0.14/0.33 | 0.67/0.22/0.21 |
| **gemma3-4b** | RAG | 0.502 [0.460, 0.548] | 0.366 [0.328, 0.407] | 0.484 [0.440, 0.531] | 0.58/0.42/0.10 | 0.66/0.41/0.05 | 0.62/0.41/0.07 |
| **gemma3-4b** | Adaptive | 0.500 [0.460, 0.546] | 0.361 [0.322, 0.404] | 0.476 [0.429, 0.527] | 0.61/0.48/0.12 | 0.76/0.18/0.18 | 0.67/0.27/0.14 |
| **gemma3-4b** | +Abstract | 0.688 [0.650, 0.730] | 0.512 [0.475, 0.551] | 0.666 [0.625, 0.712] | 0.72/0.78/0.10 | 0.90/0.56/0.07 | 0.80/0.65/0.08 |
| **qwen3-1.7b** | Baseline | 0.478 [0.434, 0.518] | 0.340 [0.310, 0.367] | 0.451 [0.404, 0.493] | 0.65/0.38/0.00 | 0.43/0.73/0.00 | 0.52/0.50/0.00 |
| **qwen3-1.7b** | RAG | 0.452 [0.410, 0.496] | 0.320 [0.292, 0.350] | 0.433 [0.389, 0.478] | 0.57/0.35/0.00 | 0.49/0.55/0.00 | 0.53/0.43/0.00 |
| **qwen3-1.7b** | Adaptive | 0.484 [0.440, 0.530] | 0.343 [0.314, 0.375] | 0.458 [0.415, 0.503] | 0.61/0.39/0.00 | 0.48/0.66/0.00 | 0.54/0.49/0.00 |
| **qwen3-1.7b** | +Abstract | 0.662 [0.618, 0.704] | 0.468 [0.427, 0.511] | 0.619 [0.570, 0.669] | 0.67/0.64/0.33 | 0.88/0.53/0.03 | 0.76/0.58/0.06 |
| **smollm3-3b** | Baseline | 0.528 [0.484, 0.574] | 0.356 [0.327, 0.386] | 0.493 [0.448, 0.541] | 0.59/0.41/0.00 | 0.69/0.45/0.00 | 0.64/0.43/0.00 |
| **smollm3-3b** | RAG | 0.526 [0.486, 0.572] | 0.366 [0.335, 0.398] | 0.496 [0.453, 0.541] | 0.60/0.43/0.00 | 0.61/0.57/0.00 | 0.61/0.49/0.00 |
| **smollm3-3b** | Adaptive | 0.548 [0.504, 0.596] | 0.374 [0.344, 0.406] | 0.513 [0.468, 0.560] | 0.61/0.45/0.00 | 0.69/0.51/0.00 | 0.65/0.48/0.00 |
| **smollm3-3b** | +Abstract | 0.724 [0.686, 0.764] | 0.505 [0.481, 0.529] | 0.677 [0.634, 0.723] | 0.75/0.68/0.00 | 0.87/0.75/0.00 | 0.80/0.71/0.00 |

### Table 1C: Representative Confusion Matrices (MedQA Baseline vs RAG)
> Matrix convention: Rows indicate **Gold (True)** label; Columns indicate **Predicted** label.

#### **qwen3-4b** (MedQA Baseline)
| True \ Pred | Pred A | Pred B | Pred C | Pred D | Total |
|---|---|---|---|---|---|
| **True A** | 83 | 15 | 25 | 19 | 142 |
| **True B** | 14 | 81 | 17 | 10 | 122 |
| **True C** | 23 | 16 | 71 | 21 | 131 |
| **True D** | 13 | 23 | 12 | 57 | 105 |

#### **qwen3-4b** (MedQA Full RAG)
| True \ Pred | Pred A | Pred B | Pred C | Pred D | Total |
|---|---|---|---|---|---|
| **True A** | 93 | 19 | 16 | 14 | 142 |
| **True B** | 15 | 86 | 11 | 10 | 122 |
| **True C** | 25 | 18 | 73 | 15 | 131 |
| **True D** | 22 | 19 | 14 | 50 | 105 |

#### **phi4-mini** (MedQA Baseline)
| True \ Pred | Pred A | Pred B | Pred C | Pred D | Total |
|---|---|---|---|---|---|
| **True A** | 77 | 24 | 22 | 19 | 142 |
| **True B** | 9 | 75 | 24 | 14 | 122 |
| **True C** | 17 | 24 | 67 | 23 | 131 |
| **True D** | 13 | 23 | 21 | 48 | 105 |

#### **phi4-mini** (MedQA Full RAG)
| True \ Pred | Pred A | Pred B | Pred C | Pred D | Total |
|---|---|---|---|---|---|
| **True A** | 81 | 18 | 21 | 22 | 142 |
| **True B** | 8 | 76 | 18 | 20 | 122 |
| **True C** | 11 | 19 | 83 | 18 | 131 |
| **True D** | 11 | 19 | 21 | 54 | 105 |

#### **medpsy-4b** (MedQA Baseline)
| True \ Pred | Pred A | Pred B | Pred C | Pred D | Total |
|---|---|---|---|---|---|
| **True A** | 127 | 8 | 5 | 2 | 142 |
| **True B** | 6 | 112 | 3 | 1 | 122 |
| **True C** | 8 | 5 | 111 | 7 | 131 |
| **True D** | 5 | 7 | 5 | 88 | 105 |

## 2. Retrieval Benchmark: Phase 6 (Hybrid top-20) vs. Phase 7 (Reranked top-5)

> **Setup**: Evaluated across all $N=500$ MedQA test questions.  
> A retrieved passage is considered **relevant** if it contains the ground-truth option text.  
> Two matching rules are evaluated:  
> 1. **Default Rule**: Exact substring match OR first word removed (if word count $\ge 2$ and rest $\ge 4$ chars).  
> 2. **Strict Verbatim Rule**: Exact case-insensitive substring match only.

### Table 2: Retrieval Ranking Metrics & Cross-Encoder Reranker Gain
| Stage | Matching Rule | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Precision@5 | MRR@10 | MAP@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **Phase 6: Hybrid BM25+MedCPT** | Default Rule | 10.0% | 22.4% | 29.6% | 37.2% | 8.4% | 0.151 | 0.139 | 0.183 |
| **Phase 7: MedCPT Cross-Encoder** | Default Rule | 17.0% | 30.0% | 30.0%* | 30.0%* | 12.5% | 0.220 | 0.211 | 0.237 |
| *Reranker Delta (Gain)* | *Default Rule* | *+7.0%* | *+7.6%* | — | — | *+4.1%* | *+0.069* | *+0.072* | *+0.054* |
|---|---|---|---|---|---|---|---|---|---|
| **Phase 6: Hybrid BM25+MedCPT** | Strict Verbatim | 6.4% | 13.4% | 17.6% | 23.2% | 4.9% | 0.093 | 0.087 | 0.111 |
| **Phase 7: MedCPT Cross-Encoder** | Strict Verbatim | 10.0% | 18.8% | 18.8%* | 18.8%* | 7.3% | 0.133 | 0.129 | 0.146 |
| *Reranker Delta (Gain)* | *Strict Verbatim* | *+3.6%* | *+5.4%* | — | — | *+2.4%* | *+0.040* | *+0.042* | *+0.035* |

*Note: Phase 7 produces top-5 reranked passages. Recall@10 and @20 on Phase 7 represent the top-5 cap.*

## 3. RAG Reliability, Citation Quality, and Hallucination Grounding (MedQA)

> Grounding and citation metrics evaluated across the MedQA test set ($N=500$):  
> - **Faithfulness %**: % of answers where at least one cited passage contains the selected option text.  
> - **Unsupported-Answer Rate**: % of answers where none of the cited passages ground the chosen option ($1 - \text{Faithfulness}$).  
> - **Citation Precision**: Total cited passages containing the chosen option divided by total cited passages.  
> - **Citation Recall**: Total retrieved gold-containing passages that were cited divided by all retrieved gold-containing passages.  
> - **Citation Validity %**: % of bracketed citations referencing a valid passage index (1–5).

### Table 3: RAG Citation & Grounding Breakdown
| Model | Faithfulness % (Count/500) | Unsupported-Answer Rate % | Citation Precision % | Citation Recall % | Citation Validity % |
|---|---|---|---|---|---|
| **medpsy-4b** | n/a | n/a | n/a | n/a | n/a |
| **qwen3-4b** | 30.8% (154/500) | 69.2% | 19.3% | 73.2% | 100.0% |
| **phi4-mini** | 26.4% (132/500) | 73.6% | 20.7% | 51.8% | 100.0% |
| **gemma3-4b** | 31.8% (159/500) | 68.2% | 21.2% | 58.5% | 100.0% |
| **qwen3-1.7b** | 20.4% (102/500) | 79.6% | 21.5% | 45.0% | 100.0% |
| **smollm3-3b** | 21.4% (107/500) | 78.6% | 21.5% | 30.4% | 100.0% |

## 4. Confidence Calibration: ECE, MCE, Brier Score, and NLL

> Calibration computed over all answerable test questions ($N=1,000$ overall: 500 MedQA + 500 PubMedQA).  
> Temperature scaling ($T^*$) fitted on validation NLL is applied where validation tuning exists.  
> - **ECE**: Expected Calibration Error (10 equal-width bins).  
> - **MCE**: Maximum Calibration Error across bins.  
> - **Brier Score**: Mean squared error between confidence and empirical correctness.  
> - **NLL**: Negative log-likelihood of the ground-truth option choice.

### Table 4: Calibration Metrics (Raw vs. Post-Hoc Temperature Scaled)
| Model | Mode | Temp ($T^*$) | Raw ECE | Scaled ECE | Raw MCE | Scaled MCE | Raw Brier | Scaled Brier | Raw NLL | Scaled NLL |
|---|---|---|---|---|---|---|---|---|---|---|
| **medpsy-4b** | Baseline | n/a | 0.115 | n/a | 0.634 | n/a | 0.112 | n/a | 1.011 | n/a |
| **medpsy-4b** | RAG | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| **qwen3-4b** | Baseline | 17.24 | 0.417 | 0.061 | 0.518 | 0.109 | 0.417 | 0.239 | 7.318 | 1.056 |
| **qwen3-4b** | RAG | 20.73 | 0.431 | 0.072 | 0.546 | 0.111 | 0.430 | 0.242 | 8.450 | 1.074 |
| **phi4-mini** | Baseline | 3.69 | 0.243 | 0.031 | 0.345 | 0.112 | 0.299 | 0.236 | 1.505 | 1.092 |
| **phi4-mini** | RAG | 4.49 | 0.259 | 0.049 | 0.391 | 0.102 | 0.298 | 0.229 | 1.610 | 1.060 |
| **gemma3-4b** | Baseline | 25.71 | 0.503 | 0.072 | 0.648 | 0.114 | 0.503 | 0.250 | 9.817 | 1.153 |
| **gemma3-4b** | RAG | 19.54 | 0.471 | 0.039 | 0.763 | 0.144 | 0.469 | 0.240 | 6.854 | 1.098 |
| **qwen3-1.7b** | Baseline | 24.72 | 0.509 | 0.057 | 0.525 | 0.120 | 0.505 | 0.249 | 10.290 | 1.164 |
| **qwen3-1.7b** | RAG | 22.72 | 0.492 | 0.075 | 0.514 | 0.146 | 0.496 | 0.253 | 9.668 | 1.159 |
| **smollm3-3b** | Baseline | 6.58 | 0.393 | 0.035 | 0.516 | 0.135 | 0.404 | 0.237 | 2.402 | 1.144 |
| **smollm3-3b** | RAG | 4.88 | 0.328 | 0.047 | 0.433 | 0.206 | 0.355 | 0.239 | 1.847 | 1.112 |

## 5. System Efficiency, Latency, and Throughput

> Computed over test-split questions.  
> RAG latency includes retrieval + reranking overhead ($+0.1273$ s/q) added to generation time.  
> Throughput represents end-to-end questions processed per second.

### Table 5: Latency and Throughput per Model and Mode
| Model | Mode | Mean Latency (s) | P95 Latency (s) | End-to-End Latency (s) | Throughput (Q/s) |
|---|---|---|---|---|---|
| **medpsy-4b** | Baseline | 37.303 s | 47.551 s | 37.303 s | 0.03 q/s |
| **medpsy-4b** | RAG | n/a | n/a | n/a | n/a |
| **medpsy-4b** | Adaptive | n/a | n/a | n/a | n/a |
| **medpsy-4b** | +Abstract | 30.279 s | 37.588 s | 30.279 s | 0.03 q/s |
| **qwen3-4b** | Baseline | 0.223 s | 0.372 s | 0.223 s | 4.49 q/s |
| **qwen3-4b** | RAG | 1.452 s | 2.014 s | 1.579 s | 0.63 q/s |
| **qwen3-4b** | Adaptive | 1.036 s | 1.995 s | 1.119 s | 0.89 q/s |
| **qwen3-4b** | +Abstract | 0.406 s | 0.540 s | 0.406 s | 2.46 q/s |
| **phi4-mini** | Baseline | 0.162 s | 0.258 s | 0.162 s | 6.17 q/s |
| **phi4-mini** | RAG | 1.129 s | 1.414 s | 1.256 s | 0.80 q/s |
| **phi4-mini** | Adaptive | 0.720 s | 1.351 s | 0.794 s | 1.26 q/s |
| **phi4-mini** | +Abstract | 0.485 s | 0.730 s | 0.485 s | 2.06 q/s |
| **gemma3-4b** | Baseline | 0.611 s | 1.101 s | 0.611 s | 1.64 q/s |
| **gemma3-4b** | RAG | 3.321 s | 4.366 s | 3.449 s | 0.29 q/s |
| **gemma3-4b** | Adaptive | 1.714 s | 4.110 s | 1.765 s | 0.57 q/s |
| **gemma3-4b** | +Abstract | 1.422 s | 1.840 s | 1.422 s | 0.70 q/s |
| **qwen3-1.7b** | Baseline | 0.117 s | 0.155 s | 0.117 s | 8.53 q/s |
| **qwen3-1.7b** | RAG | 0.696 s | 0.885 s | 0.823 s | 1.21 q/s |
| **qwen3-1.7b** | Adaptive | 0.496 s | 0.840 s | 0.579 s | 1.73 q/s |
| **qwen3-1.7b** | +Abstract | 0.333 s | 0.372 s | 0.333 s | 3.01 q/s |
| **smollm3-3b** | Baseline | 0.162 s | 0.233 s | 0.162 s | 6.16 q/s |
| **smollm3-3b** | RAG | 0.939 s | 1.166 s | 1.067 s | 0.94 q/s |
| **smollm3-3b** | Adaptive | 0.564 s | 1.147 s | 0.628 s | 1.59 q/s |
| **smollm3-3b** | +Abstract | 0.464 s | 0.625 s | 0.464 s | 2.16 q/s |

---

> **Methodological Footnote**:  
> 1. Grounding and citation metrics utilize deterministic string-matching proxies (exact substring or first word removed).  
> 2. Micro-F1 is mathematically identical to accuracy for single-label multiple-choice questions.  
> 3. Text generation metrics (BERTScore, ROUGE-L, answer relevancy) are not applicable to single-letter categorical MCQ predictions.
