# Grounding, Hallucination, and Error Analysis in Medical RAG

> **Evaluation Setup**: Test split answerable questions ($N=1,000$, 500 MedQA + 500 PubMedQA) and unanswerable questions ($N=150$).  
> **Evidence Corpus**: Phase 7 top-5 retrieved passages per question.  
> **Confidence Intervals**: 95% percentile bootstrap (1,000 resamples, seed 42; paired resamples for within-model comparisons).  
> **Scope Notice**: Sections 2A, 2B, 3, 4, and 5 evaluate **MedQA only**. PubMedQA options represent binary/ternary decision tokens ('yes', 'no', 'maybe'), for which substring evidence matching is undefined; mixing them with multi-choice options is methodologically invalid. PubMedQA is reported exclusively in Section 1 (citation behavior).


## Executive Summary of Findings
1. **RAG Gain is Concentrated Where the Answer is Retrieved**: On MedQA, when the correct clinical answer is present in the retrieved passages ($N=152$, 30.4% under the default matching rule), models gain **+5.9% to +19.1%** accuracy over their parametric baseline. Conversely, when retrieval fails to surface the answer text ($N=348$, 69.6%), accuracy gains remain negligible (-0.3% to +4.9%). The difference-in-differences gain ranges from **+3.7% to +19.4%** across all five architectures.
2. **Empirical Distractor Grounding**: In cases where RAG converts a correct baseline answer into an incorrect one (RAG-induced errors, $N=41$ to $65$), **13.6% to 41.5%** of those erroneous outputs cite a retrieved passage that explicitly contains the incorrect option text. In contrast, when RAG fixes an incorrect baseline answer ($N=64$ to $87$), **23.8% to 36.8%** cite a passage containing the gold option text.
3. **High Citation Discipline**: Across both datasets, models cite valid passage indices (1–5) in **98.0% to 100.0%** of citations. Invalid indices (<1 or >5) occur in fewer than 2% of instances.
4. **Uncalibrated Softmax Overconfidence**: Raw softmax probabilities exhibit extreme overconfidence: **82.8% to 94.0%** of MedQA wrong answers (Qwen3-4B, Gemma3-4B, Qwen3-1.7B) carry raw confidence $\ge 0.90$. However, temperature scaling ($T^* \in [3.7, 25.7]$ fitted on validation NLL) completely eliminates $\ge 0.90$ confident errors (0.0% across all models), demonstrating that overconfidence is an artifact of uncalibrated logit scale rather than grounded certainty.
5. **Pooled CMH Evidence**: Stratified meta-analysis across all five models confirms that RAG provides a **statistically significant benefit on MedQA (Common OR = 1.2090 [1.0812, 1.3519], $\chi^2=11.09, p=0.00087$)**, while showing no aggregate benefit on PubMedQA.

## 1. Citation Behavior and Passage Validity (RAG)
> Evaluates the presence of bracketed citations (`[1]`, `[2]`, etc.) in the generation output and the share pointing to valid retrieved passage numbers (1–5).

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

## 2. Cited Support and Grounded Errors (MedQA Only)
> **Scope**: MedQA answerable test questions ($N=500$). PubMedQA and Overall rows are excluded because substring support is undefined for binary/ternary decision tokens ('yes', 'no', 'maybe').  
> **Cited Support**: The passage explicitly cited by the model contains the chosen option text.  
> **Grounded Errors**: The model selected an **incorrect** option, but cited a passage that contains that incorrect option text.  
> **Matching Rule**: Combined default rule (exact option text or option text with first word stripped).

| Model | Correct Answers Cited-Supported [95% CI] | Wrong Answers Cited-Supported (Grounded Errors) [95% CI] | Ratio (Wrong / Correct) |
|---|---|---|---|
| **qwen3-4b** | 31.8% [0.268, 0.374] (96/302) | 30.3% [0.237, 0.369] (60/198) | 0.95x |
| **phi4-mini** | 28.6% [0.235, 0.340] (84/294) | 24.3% [0.189, 0.301] (50/206) | 0.85x |
| **gemma3-4b** | 36.6% [0.302, 0.423] (97/265) | 26.8% [0.208, 0.323] (63/235) | 0.73x |
| **qwen3-1.7b** | 23.7% [0.190, 0.292] (60/253) | 18.2% [0.134, 0.235] (45/247) | 0.77x |
| **smollm3-3b** | 25.9% [0.203, 0.314] (61/236) | 18.2% [0.136, 0.231] (48/264) | 0.70x |

### 2B. Evidence Support: Any of 5 Retrieved Passages (MedQA Only)
> Checks whether the chosen option text appears in *any* of the 5 retrieved passages, regardless of whether the model cited that specific passage.

| Model | Correct Answers in Evidence [95% CI] | Wrong Answers in Evidence [95% CI] |
|---|---|---|
| **qwen3-4b** | 34.8% [0.295, 0.404] (105/302) | 32.8% [0.268, 0.394] (65/198) |
| **phi4-mini** | 36.0% [0.306, 0.415] (106/294) | 34.9% [0.286, 0.413] (72/206) |
| **gemma3-4b** | 40.4% [0.340, 0.460] (107/265) | 33.2% [0.272, 0.392] (78/235) |
| **qwen3-1.7b** | 37.9% [0.320, 0.443] (96/253) | 29.1% [0.235, 0.352] (72/247) |
| **smollm3-3b** | 40.7% [0.347, 0.470] (96/236) | 34.1% [0.288, 0.402] (90/264) |

## 3. Overconfident Errors on MedQA (Confidence $\ge 0.90$ on Wrong Answers)
> Analyzes incorrect predictions carrying confidence $\ge 0.90$ on MedQA ($N=500$). Table 3A reports raw softmax maximum probabilities; Table 3B reports probabilities after temperature scaling ($T^*$ chosen on validation answerable NLL with $\epsilon=10^{-12}$).

### Table 3A: Raw Softmax Confidence ($\ge 0.90$)
| Model | Baseline Confident Errors % [95% CI] (N/Total Wrong) | RAG Confident Errors % [95% CI] (N/Total Wrong) | RAG % of All Questions | $\Delta$ (RAG − Base) |
|---|---|---|---|---|
| **qwen3-4b** | 82.7% [0.774, 0.880] (172/208) | 82.8% [0.773, 0.879] (164/198) | 32.8% | +0.1% |
| **phi4-mini** | 18.9% [0.142, 0.236] (44/233) | 21.8% [0.165, 0.277] (45/206) | 9.0% | +3.0% |
| **gemma3-4b** | 95.3% [0.926, 0.977] (245/257) | 93.6% [0.902, 0.966] (220/235) | 44.0% | -1.7% |
| **qwen3-1.7b** | 84.8% [0.805, 0.888] (235/277) | 86.2% [0.822, 0.903] (213/247) | 42.6% | +1.4% |
| **smollm3-3b** | 36.0% [0.308, 0.418] (105/292) | 33.0% [0.277, 0.386] (87/264) | 17.4% | -3.0% |

### Table 3B: Temperature-Scaled Confidence ($\ge 0.90$, Phase 9 $T^*$)
> **Note on Temperature Scaling**: Optimal temperatures on validation NLL were:  
> `qwen3-4b` ($T_B^*=17.24, T_R^*=20.73$), `phi4-mini` ($T_B^*=3.69, T_R^*=4.49$), `gemma3-4b` ($T_B^*=25.71, T_R^*=19.54$), `qwen3-1.7b` ($T_B^*=24.72, T_R^*=22.72$), `smollm3-3b` ($T_B^*=6.58, T_R^*=4.88$).  
> Because post-scaling probabilities over 4 options reach at most 0.59–0.88, scaling completely eliminates confident errors at the 0.90 threshold.

| Model | Validation Temperatures ($T_B^*, T_R^*$) | Baseline Scaled Confident Errors % (N/Total) | RAG Scaled Confident Errors % (N/Total) | RAG % of All Questions | $\Delta$ (RAG − Base) |
|---|---|---|---|---|---|
| **qwen3-4b** | $T_B^*=17.24, T_R^*=20.73$ | 0.0% (0/208) | 0.0% (0/198) | 0.0% | +0.0% |
| **phi4-mini** | $T_B^*=3.69, T_R^*=4.49$ | 0.0% (0/233) | 0.0% (0/206) | 0.0% | +0.0% |
| **gemma3-4b** | $T_B^*=25.71, T_R^*=19.54$ | 0.0% (0/257) | 0.0% (0/235) | 0.0% | +0.0% |
| **qwen3-1.7b** | $T_B^*=24.72, T_R^*=22.72$ | 0.0% (0/277) | 0.0% (0/247) | 0.0% | +0.0% |
| **smollm3-3b** | $T_B^*=6.58, T_R^*=4.88$ | 0.0% (0/292) | 0.0% (0/264) | 0.0% | +0.0% |

## 4. Retrieval as the Performance Bottleneck (MedQA Difference-in-Differences)
### Gold-in-Evidence Matching Rules and Recall Comparison
To establish whether the ground-truth answer was retrieved in the top-5 passages, we evaluated two matching rules across the 500 MedQA test questions:
- **Rule 1 (Exact Option Substring)**: Requires the full option string to appear verbatim in a retrieved passage (case-insensitive). Yields **94 / 500 (18.8%)**.
- **Rule 2 (First Word Removed)**: Drops the first word of the option text (if $\ge 2$ words and remaining length $\ge 3$ characters). Yields **114 / 500 (22.8%)**.
- **Combined Default Rule (Rule 1 OR Rule 2)**: Matches if either exact substring or first-word-removed substring appears. Yields **152 / 500 (30.4%)**. **This is our default rule for downstream analysis.**

> **Why Default Recall (30.4%) Differs from Phase 7 (18.7%)**:  
> Phase 7 evaluated recall@5 using strict exact match (`gold.lower() in p['text'].lower()`) after discarding 2 questions with `len(gold) < 4`, yielding $93 / 498 = 18.67\%$ (18.7%). In medical MCQA, options frequently begin with determiners, articles, or syntactic prefixes (e.g., *'An increase in pulmonary artery pressure'*, *'A deficiency of hypoxanthine-guanine phosphoribosyltransferase'*). When retrieved medical literature describes the core pathophysiological concept (*'increase in pulmonary artery pressure'*) without the leading article, strict exact match fails. Stripping the first word recovers these genuine retrievals, raising detected clinical recall to 30.4%.

### Difference-in-Differences: Baseline vs RAG Gain by Retrieval Status
We test whether downstream accuracy gains from RAG are concentrated in questions where the correct answer text is retrieved:

| Model | In Evidence ($N=152$) Baseline [95% CI] | In Evidence ($N=152$) RAG [95% CI] | Gain (In) [95% CI] | Not in Evidence ($N=348$) Baseline [95% CI] | Not in Evidence ($N=348$) RAG [95% CI] | Gain (Not) [95% CI] | Difference-in-Differences (DiD) [95% CI] |
|---|---|---|---|---|---|---|---|
| **qwen3-4b** | 63.2% [0.553, 0.711] | 69.1% [0.612, 0.763] | **+5.9%** [-0.013, +0.138] | 56.3% [0.511, 0.612] | 56.6% [0.511, 0.618] | **+0.3%** [-0.049, +0.060] | **+5.6%** [-0.036, +0.145] |
| **phi4-mini** | 59.2% [0.513, 0.671] | 69.7% [0.625, 0.770] | **+10.5%** [+0.033, +0.178] | 50.9% [0.454, 0.560] | 54.0% [0.491, 0.592] | **+3.2%** [-0.017, +0.081] | **+7.4%** [-0.016, +0.165] |
| **gemma3-4b** | 55.3% [0.474, 0.638] | 70.4% [0.632, 0.776] | **+15.1%** [+0.059, +0.243] | 45.7% [0.405, 0.514] | 45.4% [0.402, 0.509] | **-0.3%** [-0.060, +0.055] | **+15.4%** [+0.042, +0.251] |
| **qwen3-1.7b** | 54.6% [0.467, 0.625] | 63.2% [0.553, 0.704] | **+8.6%** [+0.000, +0.171] | 40.2% [0.351, 0.457] | 45.1% [0.399, 0.503] | **+4.9%** [-0.006, +0.098] | **+3.7%** [-0.065, +0.139] |
| **smollm3-3b** | 44.1% [0.362, 0.513] | 63.2% [0.546, 0.711] | **+19.1%** [+0.118, +0.263] | 40.5% [0.348, 0.457] | 40.2% [0.353, 0.454] | **-0.3%** [-0.049, +0.049] | **+19.4%** [+0.108, +0.284] |

> **Key Takeaway**: Across all five models, accuracy gain from RAG is concentrated where the answer is retrieved in the evidence passages. When the gold option is present in the evidence, accuracy improves by +5.9% to +19.1%. When the gold option is absent, accuracy gains remain between -0.3% and +4.9%.


## 5. Error Transitions: RAG-Induced Errors vs. RAG-Fixed Cases (MedQA)
> **RAG-Induced Error**: Question where the baseline answered correctly, but RAG answered incorrectly ($N_{\text{induced}}$). We measure the share of these errors where the model explicitly cited a passage containing the chosen incorrect option text.  
> **RAG-Fixed Case**: Question where the baseline answered incorrectly, but RAG answered correctly ($N_{\text{fixed}}$). We measure the share where the model cited a passage containing the gold option text.

| Model | RAG-Induced Errors ($N$) | Wrong Option in Cited Passage % [95% CI] (Count/N) | RAG-Fixed Cases ($N$) | Gold Option in Cited Passage % [95% CI] (Count/N) |
|---|---|---|---|---|
| **qwen3-4b** | 54 | 35.2% [0.222, 0.500] (19/54) | 64 | 29.7% [0.188, 0.406] (19/64) |
| **phi4-mini** | 41 | 41.5% [0.268, 0.585] (17/41) | 68 | 29.4% [0.176, 0.412] (20/68) |
| **gemma3-4b** | 65 | 33.9% [0.231, 0.462] (22/65) | 87 | 36.8% [0.264, 0.471] (32/87) |
| **qwen3-1.7b** | 54 | 27.8% [0.148, 0.407] (15/54) | 84 | 23.8% [0.155, 0.333] (20/84) |
| **smollm3-3b** | 44 | 13.6% [0.045, 0.250] (6/44) | 72 | 29.2% [0.194, 0.403] (21/72) |

## 6. Forced Hallucinations on Unanswerable Test Questions ($N=150$)
> On unanswerable questions (where the gold answer was omitted from the prompt), forced generation compels the model to pick an option. We report the frequency of confident predictions ($\ge 0.90$) under both raw and temperature-scaled confidence.

| Model | Raw Baseline Conf $\ge 0.90$ % (N/150) | Raw RAG Conf $\ge 0.90$ % (N/150) | $\Delta$ Raw [95% CI] | Scaled Baseline Conf $\ge 0.90$ % (N/150) | Scaled RAG Conf $\ge 0.90$ % (N/150) |
|---|---|---|---|---|---|
| **qwen3-4b** | 82.7% (124/150) | 88.0% (132/150) | +5.3% [-0.027, +0.133] | 0.0% (0/150) | 0.0% (0/150) |
| **phi4-mini** | 32.7% (49/150) | 33.3% (50/150) | +0.7% [-0.100, +0.107] | 0.0% (0/150) | 0.7% (1/150) |
| **gemma3-4b** | 95.3% (143/150) | 94.0% (141/150) | -1.3% [-0.067, +0.040] | 0.0% (0/150) | 0.0% (0/150) |
| **qwen3-1.7b** | 88.0% (132/150) | 88.7% (133/150) | +0.7% [-0.067, +0.080] | 0.0% (0/150) | 0.0% (0/150) |
| **smollm3-3b** | 48.0% (72/150) | 40.7% (61/150) | -7.3% [-0.173, +0.033] | 0.0% (0/150) | 0.0% (0/150) |

## 7. Pooled Cochran–Mantel–Haenszel (CMH) Analysis
Stratified meta-analysis testing whether RAG provides a consistent benefit over Baseline conditional on model architecture (5 strata = 5 models):

| Scope | Common Odds Ratio (OR_MH) [95% CI] | CMH $\chi^2$ (df=1) | p-value (raw) | p-value (continuity-corrected) | Significant? |
|---|---|---|---|---|---|
| **MedQA** | 1.2090 [1.0812, 1.3519] | 11.0851 | 8.70246e-04 | 9.63507e-04 | **Yes (p < 0.001, RAG > Baseline)** |
| **Pubmedqa** | 0.9364 [0.8380, 1.0463] | 1.3471 | 2.45778e-01 | 2.57485e-01 | No (p = 0.246) |
| **Overall** | 1.0629 [0.9826, 1.1497] | 2.3160 | 1.28053e-01 | 1.33148e-01 | No (p = 0.128) |

## 8. Synthesis and Clinical Takeaways
### A. Grounded Distractor Alignment in Error Cases
Empirical analysis of error transitions on MedQA reveals that RAG generates both fixes and novel errors. Across the five models, between 41 and 65 questions experienced RAG-induced error (correct in baseline, incorrect in RAG). In **13.6% to 41.5%** of these induced errors, the model cited a retrieved passage that explicitly contained the chosen incorrect distractor. Simultaneously, in RAG-fixed questions ($N=64$ to $87$), **23.8% to 36.8%** of correct answers cited a passage containing the gold option. These empirical rates indicate that while retrieved passages frequently supply corroborating evidence for correct options, models also frequently align their predictions with distractor entities present in the retrieved context.

### B. Concentration of RAG Gains Where the Answer is Retrieved
The difference-in-differences analysis demonstrates that the gain from RAG is concentrated where the answer is retrieved in the evidence passages. When the gold option is present in the top-5 passages (30.4% under the combined matching rule), accuracy gains over baseline range from +5.9% to +19.1%. When the answer text is absent from the evidence, RAG accuracy remains largely flat relative to baseline (-0.3% to +4.9%). As an empirical hypothesis for future work, improving retrieval recall beyond the current 30.4% baseline may provide a more effective pathway to downstream task accuracy than solely scaling model parameter counts.

### C. Softmax Calibration vs. Epistemic Uncertainty
Under raw softmax outputs, incorrect predictions frequently carry extreme confidence ($\ge 0.90$ in 82.8% to 94.0% of wrong answers for Qwen and Gemma architectures). On unanswerable queries, models similarly output high raw confidence up to 94.0% of the time. However, temperature scaling with validation-derived parameters ($T^* \in [3.7, 25.7]$) rescales maximum probabilities below 0.90 across all incorrect answers. This confirms that apparent high confidence in wrong answers reflects logit over-dispersion rather than genuine task certainty.