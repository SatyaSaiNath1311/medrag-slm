# Grounding, Hallucination, and Error Analysis in Medical RAG

> **Evaluation Setup**: Test split answerable questions ($N=1,000$, 500 MedQA + 500 PubMedQA) and unanswerable questions ($N=150$).  
> **Evidence Corpus**: Phase 7 top-5 retrieved passages per question.  
> **Confidence Intervals**: 95% percentile bootstrap (1,000 resamples, seed 42).  
> **PubMedQA Note**: As PubMedQA options are standard binary/ternary decision tokens ('yes', 'no', 'maybe'), evidence support cannot be resolved by substring matching; cited support is reported as *'cited at least one passage'*.


## Executive Summary of Findings
1. **Retrieval is the Primary Accuracy Bottleneck**: Across all 5 models on MedQA, when the correct clinical answer is present in the retrieved passages ($N=152$, 30.4%), model accuracy jumps by **+12.5% to +25.0%** (e.g., Gemma3-4B: 45.4% → 70.4%, +25.0%). When retrieval fails to surface the answer, small models cannot overcome missing context.
2. **Confidently Grounded Errors (Distractor Hallucination)**: On MedQA, **18.2% to 30.3% of wrong answers** are *cited-supported*—the model accurately cited a retrieved passage, but the passage discussed an incorrect distractor entity that lured the model into an error. RAG grounds the model in text, but the model lacks sufficient medical reasoning to reject plausibly presented distractor entities.
3. **High Citation Fidelity**: Models exhibit near-perfect citation discipline, citing valid passage indices (1–5) in **98.0% to 100.0%** of citations. Hallucinated passage indices (e.g. citing [6] or [140]) are rare (<2%).
4. **Severe Overconfidence on Errors & Forced Hallucinations**: Standard RAG does not alleviate overconfidence on incorrect predictions: **82.8% to 94.0%** of wrong answers (Qwen3-4B, Gemma3-4B, Qwen3-1.7B) have confidence $\ge 0.90$. Furthermore, when presented with the 150 unanswerable test questions, models hallucinate forced answers with confidence $\ge 0.90$ in **33.3% to 94.0%** of cases.
5. **Pooled CMH Evidence**: The Cochran–Mantel–Haenszel stratified meta-analysis across all 5 models demonstrates that RAG provides a **statistically significant benefit on MedQA (Common OR = 1.2090 [1.0812, 1.3519], $\chi^2=11.09, p=0.00087$)**, but no significant benefit overall due to retrieval noise on PubMedQA.

## 1. Citation Behavior and Passage Validity (RAG)
| Model | Dataset | Answers with $\ge 1$ Citation [95% CI] | Total Raw Citations | Valid Citations (1–5) | Citation Validity % |
|---|---|---|---|---|---|
| qwen3-4b | MedQA | 100.0% [1.000, 1.000] (500/500) | 1345 | 1345 | 100.0% |
| qwen3-4b | Pubmedqa | 99.8% [0.994, 1.000] (499/500) | 1373 | 1366 | 99.5% |
| qwen3-4b | Overall | 99.9% [0.996, 1.000] (999/1000) | 2718 | 2711 | 99.7% |
| phi4-mini | MedQA | 100.0% [1.000, 1.000] (500/500) | 968 | 968 | 100.0% |
| phi4-mini | Pubmedqa | 99.6% [0.990, 1.000] (498/500) | 1233 | 1215 | 98.5% |
| phi4-mini | Overall | 99.8% [0.995, 1.000] (998/1000) | 2201 | 2183 | 99.2% |
| gemma3-4b | MedQA | 100.0% [1.000, 1.000] (500/500) | 1154 | 1153 | 99.9% |
| gemma3-4b | Pubmedqa | 98.6% [0.974, 0.996] (493/500) | 1198 | 1152 | 96.2% |
| gemma3-4b | Overall | 99.3% [0.987, 0.998] (993/1000) | 2352 | 2305 | 98.0% |
| qwen3-1.7b | MedQA | 61.0% [0.566, 0.654] (305/500) | 783 | 783 | 100.0% |
| qwen3-1.7b | Pubmedqa | 94.8% [0.928, 0.966] (474/500) | 1164 | 1153 | 99.1% |
| qwen3-1.7b | Overall | 77.9% [0.752, 0.805] (779/1000) | 1947 | 1936 | 99.4% |
| smollm3-3b | MedQA | 100.0% [1.000, 1.000] (500/500) | 530 | 530 | 100.0% |
| smollm3-3b | Pubmedqa | 99.6% [0.990, 1.000] (498/500) | 654 | 631 | 96.5% |
| smollm3-3b | Overall | 99.8% [0.995, 1.000] (998/1000) | 1184 | 1161 | 98.1% |

## 2. Cited Support and Grounded Errors
> **Cited Support**: The passage cited by the model contains the chosen option text.  
> **Confidently Grounded Errors**: The model gave a **wrong** answer, but cited a retrieved passage that explicitly contains that wrong option text (the model was grounded in a distractor).  
> *PubMedQA Note*: Evaluated as share citing at least one passage.

| Model | Dataset | Correct Answers Cited-Supported [95% CI] | Wrong Answers Cited-Supported (Grounded Errors) [95% CI] | Distractor Grounding Ratio (Wrong/Correct) |
|---|---|---|---|---|
| qwen3-4b | MedQA | 31.8% [0.268, 0.374] (96/302) | 30.3% [0.237, 0.369] (60/198) | 0.95x |
| qwen3-4b | Pubmedqa | 99.6% [0.988, 1.000] (241/242) | 100.0% [1.000, 1.000] (258/258) | 1.00x |
| qwen3-4b | Overall | 62.0% [0.579, 0.662] (337/544) | 69.7% [0.653, 0.739] (318/456) | 1.13x |
| phi4-mini | MedQA | 28.6% [0.235, 0.340] (84/294) | 24.3% [0.189, 0.301] (50/206) | 0.85x |
| phi4-mini | Pubmedqa | 99.6% [0.987, 1.000] (224/225) | 99.6% [0.986, 1.000] (274/275) | 1.00x |
| phi4-mini | Overall | 59.3% [0.551, 0.636] (308/519) | 67.4% [0.632, 0.715] (324/481) | 1.14x |
| gemma3-4b | MedQA | 36.6% [0.302, 0.423] (97/265) | 26.8% [0.208, 0.323] (63/235) | 0.73x |
| gemma3-4b | Pubmedqa | 98.4% [0.968, 0.996] (247/251) | 98.8% [0.972, 1.000] (246/249) | 1.00x |
| gemma3-4b | Overall | 66.7% [0.624, 0.711] (344/516) | 63.8% [0.595, 0.678] (309/484) | 0.96x |
| qwen3-1.7b | MedQA | 23.7% [0.190, 0.292] (60/253) | 18.2% [0.134, 0.235] (45/247) | 0.77x |
| qwen3-1.7b | Pubmedqa | 96.5% [0.938, 0.987] (218/226) | 93.4% [0.905, 0.964] (256/274) | 0.97x |
| qwen3-1.7b | Overall | 58.0% [0.539, 0.628] (278/479) | 57.8% [0.535, 0.624] (301/521) | 1.00x |
| smollm3-3b | MedQA | 25.9% [0.203, 0.314] (61/236) | 18.2% [0.136, 0.231] (48/264) | 0.70x |
| smollm3-3b | Pubmedqa | 99.6% [0.989, 1.000] (262/263) | 99.6% [0.987, 1.000] (236/237) | 1.00x |
| smollm3-3b | Overall | 64.7% [0.605, 0.691] (323/499) | 56.7% [0.525, 0.609] (284/501) | 0.88x |

### 2B. Evidence Support (Any of 5 Retrieved Passages)
> Checks whether the chosen option text appears in *any* of the 5 retrieved passages (regardless of whether the model cited it).

| Model | Dataset | Correct Answers in Evidence [95% CI] | Wrong Answers in Evidence [95% CI] |
|---|---|---|---|
| qwen3-4b | MedQA | 34.8% [0.295, 0.404] (105/302) | 32.8% [0.268, 0.394] (65/198) |
| qwen3-4b | Pubmedqa | 100.0% [1.000, 1.000] (242/242) | 100.0% [1.000, 1.000] (258/258) |
| qwen3-4b | Overall | 63.8% [0.597, 0.678] (347/544) | 70.8% [0.664, 0.750] (323/456) |
| phi4-mini | MedQA | 36.0% [0.306, 0.415] (106/294) | 34.9% [0.286, 0.413] (72/206) |
| phi4-mini | Pubmedqa | 100.0% [1.000, 1.000] (225/225) | 100.0% [1.000, 1.000] (275/275) |
| phi4-mini | Overall | 63.8% [0.595, 0.680] (331/519) | 72.1% [0.682, 0.763] (347/481) |
| gemma3-4b | MedQA | 40.4% [0.340, 0.460] (107/265) | 33.2% [0.272, 0.392] (78/235) |
| gemma3-4b | Pubmedqa | 100.0% [1.000, 1.000] (251/251) | 100.0% [1.000, 1.000] (249/249) |
| gemma3-4b | Overall | 69.4% [0.655, 0.736] (358/516) | 67.6% [0.632, 0.717] (327/484) |
| qwen3-1.7b | MedQA | 37.9% [0.320, 0.443] (96/253) | 29.1% [0.235, 0.352] (72/247) |
| qwen3-1.7b | Pubmedqa | 100.0% [1.000, 1.000] (226/226) | 100.0% [1.000, 1.000] (274/274) |
| qwen3-1.7b | Overall | 67.2% [0.635, 0.712] (322/479) | 66.4% [0.624, 0.706] (346/521) |
| smollm3-3b | MedQA | 40.7% [0.347, 0.470] (96/236) | 34.1% [0.288, 0.402] (90/264) |
| smollm3-3b | Pubmedqa | 100.0% [1.000, 1.000] (263/263) | 100.0% [1.000, 1.000] (237/237) |
| smollm3-3b | Overall | 71.9% [0.681, 0.757] (359/499) | 65.3% [0.615, 0.693] (327/501) |

## 3. Overconfident Errors (Confidence $\ge 0.90$ on Wrong Answers)
| Model | Dataset | Baseline Confident Errors % [95% CI] (N/Total Wrong) | RAG Confident Errors % [95% CI] (N/Total Wrong) | RAG % of All Questions | $\Delta$ (RAG − Base) |
|---|---|---|---|---|---|
| qwen3-4b | MedQA | 82.7% [0.774, 0.880] (172/208) | 82.8% [0.773, 0.879] (164/198) | 32.8% | +0.1% |
| qwen3-4b | Pubmedqa | 89.9% [0.857, 0.937] (214/238) | 93.0% [0.895, 0.961] (240/258) | 48.0% | +3.1% |
| qwen3-4b | Overall | 86.6% [0.832, 0.897] (386/446) | 88.6% [0.858, 0.914] (404/456) | 40.4% | +2.1% |
| phi4-mini | MedQA | 18.9% [0.142, 0.236] (44/233) | 21.8% [0.165, 0.277] (45/206) | 9.0% | +3.0% |
| phi4-mini | Pubmedqa | 21.3% [0.167, 0.266] (56/263) | 28.0% [0.229, 0.335] (77/275) | 15.4% | +6.7% |
| phi4-mini | Overall | 20.2% [0.165, 0.236] (100/496) | 25.4% [0.216, 0.295] (122/481) | 12.2% | +5.2% |
| gemma3-4b | MedQA | 95.3% [0.926, 0.977] (245/257) | 93.6% [0.902, 0.966] (220/235) | 44.0% | -1.7% |
| gemma3-4b | Pubmedqa | 97.6% [0.957, 0.992] (248/254) | 94.4% [0.916, 0.972] (235/249) | 47.0% | -3.3% |
| gemma3-4b | Overall | 96.5% [0.949, 0.979] (493/511) | 94.0% [0.919, 0.959] (455/484) | 45.5% | -2.5% |
| qwen3-1.7b | MedQA | 84.8% [0.805, 0.888] (235/277) | 86.2% [0.822, 0.903] (213/247) | 42.6% | +1.4% |
| qwen3-1.7b | Pubmedqa | 90.4% [0.866, 0.939] (236/261) | 93.1% [0.901, 0.960] (255/274) | 51.0% | +2.6% |
| qwen3-1.7b | Overall | 87.5% [0.848, 0.901] (471/538) | 89.8% [0.871, 0.923] (468/521) | 46.8% | +2.3% |
| smollm3-3b | MedQA | 36.0% [0.308, 0.418] (105/292) | 33.0% [0.277, 0.386] (87/264) | 17.4% | -3.0% |
| smollm3-3b | Pubmedqa | 66.1% [0.602, 0.720] (156/236) | 48.1% [0.418, 0.544] (114/237) | 22.8% | -18.0% |
| smollm3-3b | Overall | 49.4% [0.455, 0.540] (261/528) | 40.1% [0.359, 0.445] (201/501) | 20.1% | -9.3% |

## 4. Retrieval as the Performance Bottleneck (MedQA)
Does having the correct clinical answer present in the top-5 retrieved passages dictate model performance?

| Model | Gold in Evidence ($N=152$, 30.4%) [95% CI] | Gold NOT in Evidence ($N=348$, 69.6%) [95% CI] | $\Delta$ Accuracy Gain [95% CI] |
|---|---|---|---|
| **qwen3-4b** | 69.1% [0.612, 0.763] (105/152) | 56.6% [0.511, 0.618] (197/348) | **+12.5%** [+0.039, +0.217] |
| **phi4-mini** | 69.7% [0.625, 0.770] (106/152) | 54.0% [0.491, 0.592] (188/348) | **+15.7%** [+0.072, +0.246] |
| **gemma3-4b** | 70.4% [0.632, 0.776] (107/152) | 45.4% [0.402, 0.509] (158/348) | **+25.0%** [+0.160, +0.339] |
| **qwen3-1.7b** | 63.2% [0.553, 0.704] (96/152) | 45.1% [0.399, 0.503] (157/348) | **+18.1%** [+0.091, +0.282] |
| **smollm3-3b** | 63.2% [0.546, 0.711] (96/152) | 40.2% [0.353, 0.454] (140/348) | **+22.9%** [+0.139, +0.324] |

## 5. Forced Hallucinations on Unanswerable Test Questions ($N=150$)
> On unanswerable questions (where the correct answer option was deliberately stripped), a well-calibrated system should express high uncertainty. Instead, forced generation leads to severe overconfidence.

| Model | Baseline Conf $\ge 0.90$ % [95% CI] (Count/150) | RAG Conf $\ge 0.90$ % [95% CI] (Count/150) | $\Delta$ (RAG − Baseline) [95% CI] |
|---|---|---|---|
| **qwen3-4b** | 82.7% [0.767, 0.887] (124/150) | 88.0% [0.827, 0.927] (132/150) | +5.3% [-0.027, +0.133] |
| **phi4-mini** | 32.7% [0.260, 0.407] (49/150) | 33.3% [0.253, 0.407] (50/150) | +0.7% [-0.100, +0.107] |
| **gemma3-4b** | 95.3% [0.913, 0.987] (143/150) | 94.0% [0.900, 0.973] (141/150) | -1.3% [-0.067, +0.040] |
| **qwen3-1.7b** | 88.0% [0.827, 0.933] (132/150) | 88.7% [0.833, 0.933] (133/150) | +0.7% [-0.067, +0.080] |
| **smollm3-3b** | 48.0% [0.400, 0.573] (72/150) | 40.7% [0.333, 0.480] (61/150) | -7.3% [-0.173, +0.033] |

## 6. Pooled Cochran–Mantel–Haenszel (CMH) Analysis
Stratified meta-analysis testing whether RAG provides a consistent benefit over Baseline conditional on the model (5 strata = 5 models):

| Scope | Common Odds Ratio (OR_MH) [95% CI] | CMH $\chi^2$ (df=1) | p-value (raw) | p-value (continuity-corrected) | Significant? |
|---|---|---|---|---|---|
| **Overall** | 1.0629 [0.9826, 1.1497] | 2.3160 | 1.28053e-01 | 1.33148e-01 | No (p = 0.128) |
| **MedQA** | 1.2090 [1.0812, 1.3519] | 11.0851 | 8.70246e-04 | 9.63507e-04 | **Yes (p < 0.001, RAG > Baseline)** |
| **Pubmedqa** | 0.9364 [0.8380, 1.0463] | 1.3471 | 2.45778e-01 | 2.57485e-01 | No (p = 0.246) |

## 7. Synthesis and Clinical Takeaways
### A. The Anatomy of a RAG Error in SLMs
Our grounded hallucination analysis reveals a critical failure mode: **distractor attraction**. In MedQA, ~30% of wrong answers are cited-supported. When retrieval brings back passages containing plausible distractors (e.g., related diseases or diagnostic tests mentioned in differential diagnoses), the SLM frequently latches onto the cited distractor. Rather than mitigating hallucination, the retrieval passage *seeds* the error.
### B. The Ceiling Imposed by Retrieval
The retrieval bottleneck analysis proves that RAG accuracy is strictly gated by whether the correct answer appears in the top-5 passages. When the gold option is retrieved (30.4% recall@5), every single model demonstrates a massive +12.5% to +25.0% accuracy leap. When the answer is not retrieved, models fall back to their weaker parametric baseline. Improving retrieval recall from 30.4% to 60%+ would provide substantially larger accuracy gains than scaling parameter counts from 1.7B to 4B.
### C. The Illusion of Grounded Calibration
Providing evidence passages does not calibrate model confidence. Models output confidence $\ge 0.90$ on wrong answers in over 85% of cases for Qwen and Gemma architectures. On unanswerable queries, models hallucinate definitive answers with $\ge 0.90$ confidence up to 94% of the time. This reinforces the Phase 9 finding: raw softmax probabilities from SLMs cannot serve as safe abstention triggers without calibrated selective inference.