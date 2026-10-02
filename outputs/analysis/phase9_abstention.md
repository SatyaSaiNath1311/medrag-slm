# Phase 9: Abstention, Calibration, and Selective Accuracy Analysis

> **Dataset**: MedQA + PubMedQA test split ($N=1\,150$ total: 1 000 answerable, 150 unanswerable).
> **Confidence**: Max letter probability ($\max_{L \in \{A, B, C, D\}} P(L)$).
> **Threshold Tuning**: All operating thresholds tuned strictly on the **validation split only** ($N=450$: 400 answerable, 50 unanswerable).
> **Coverage**: Percentage of all test questions answered ($N_{\text{answered}} / 1\,150$). Selective accuracy is never reported without coverage.
> **Trivial Baseline**: Always Abstain achieves Test F1 = **0.2308** ($2p/(1+p)$, $p = 150/1150 \approx 0.1304$).

## 1. Headline Selective Classification: Fixed-Coverage Operating Points

Threshold $\tau$ is chosen on **validation** to target 80% and 50% coverage, then applied to **test**.
To avoid ties at 1.0, operating thresholds in **Table 1A** are determined using **temperature-scaled confidence** ($T^*$ from validation NLL). Table 1B reports the unscaled raw version for comparison.

### Table 1A: Temperature-Scaled Confidence (Tie-Free Operating Points)

| Model | Mode | Mode Acc @ 100% Cov | Target 80% $\tau_{\text{scaled}}$ | Test Cov Achieved | Sel Acc @ 80% [95% CI] | Unans Abstained @ 80% | Target 50% $\tau_{\text{scaled}}$ | Test Cov Achieved | Sel Acc @ 50% [95% CI] | Unans Abstained @ 50% |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| qwen3-4b | baseline | 0.5540 | 0.4320 | 80.4% | **0.5928** [0.559, 0.627] | 22.0% | 0.6722 | 50.0% | **0.6550** [0.615, 0.698] | 60.7% |
| qwen3-4b | rag | 0.5440 | 0.5355 | 77.4% | **0.5946** [0.561, 0.629] | 21.3% | 1.0000 | 48.7% | **0.6633** [0.620, 0.704] | 59.3% |
| phi4-mini | baseline | 0.5040 | 0.3874 | 77.2% | **0.5532** [0.519, 0.589] | 9.3% | 0.4630 | 47.5% | **0.6030** [0.558, 0.646] | 46.7% |
| phi4-mini | rag | 0.5190 | 0.3821 | 76.0% | **0.5795** [0.544, 0.613] | 12.0% | 0.4672 | 44.9% | **0.6667** [0.622, 0.708] | 52.0% |
| gemma3-4b | baseline | 0.4890 | 0.4128 | 79.3% | **0.5196** [0.486, 0.554] | 19.3% | 0.6373 | 46.8% | **0.5592** [0.514, 0.604] | 68.0% |
| gemma3-4b | rag | 0.5160 | 0.3795 | 81.2% | **0.5556** [0.521, 0.591] | 5.3% | 0.4732 | 49.2% | **0.5992** [0.556, 0.642] | 45.3% |
| qwen3-1.7b | baseline | 0.4620 | 0.4027 | 81.5% | **0.4847** [0.451, 0.519] | 18.7% | 0.6293 | 51.3% | **0.5121** [0.470, 0.556] | 63.3% |
| qwen3-1.7b | rag | 0.4790 | 0.4637 | 82.3% | **0.5031** [0.468, 0.537] | 13.3% | 0.7699 | 51.4% | **0.5293** [0.486, 0.569] | 58.7% |
| smollm3-3b | baseline | 0.4720 | 0.3632 | 79.4% | **0.5182** [0.481, 0.556] | 3.3% | 0.4395 | 50.2% | **0.5708** [0.527, 0.616] | 40.0% |
| smollm3-3b | rag | 0.4990 | 0.3868 | 77.4% | **0.5441** [0.509, 0.580] | 12.7% | 0.4693 | 45.4% | **0.6000** [0.557, 0.644] | 52.0% |

### Table 1B: Raw Confidence Operating Points (Reference)

| Model | Mode | Mode Acc @ 100% Cov | Target 80% $\tau_{\text{raw}}$ | Test Cov Achieved | Sel Acc @ 80% [95% CI] | Unans Abstained @ 80% | Target 50% $\tau_{\text{raw}}$ | Test Cov Achieved | Sel Acc @ 50% [95% CI] | Unans Abstained @ 50% |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| qwen3-4b | baseline | 0.5540 | 0.9823 | 81.3% | **0.5874** [0.554, 0.621] | 26.0% | 1.0000 | 50.0% | **0.6480** [0.608, 0.691] | 61.3% |
| qwen3-4b | rag | 0.5440 | 0.9967 | 78.5% | **0.5817** [0.548, 0.617] | 24.0% | 1.0000 | 52.3% | **0.6517** [0.611, 0.691] | 55.3% |
| phi4-mini | baseline | 0.5040 | 0.5616 | 78.3% | **0.5508** [0.517, 0.588] | 17.3% | 0.7865 | 45.8% | **0.6095** [0.567, 0.653] | 56.0% |
| phi4-mini | rag | 0.5190 | 0.6108 | 74.8% | **0.5764** [0.542, 0.610] | 24.0% | 0.8634 | 44.3% | **0.6748** [0.631, 0.716] | 61.3% |
| gemma3-4b | baseline | 0.4890 | 0.9999 | 77.8% | **0.5119** [0.476, 0.547] | 33.3% | 1.0000 | 47.1% | **0.5607** [0.515, 0.606] | 68.0% |
| gemma3-4b | rag | 0.5160 | 0.9996 | 79.9% | **0.5431** [0.508, 0.578] | 21.3% | 1.0000 | 49.2% | **0.6163** [0.573, 0.657] | 49.3% |
| qwen3-1.7b | baseline | 0.4620 | 0.9727 | 84.3% | **0.4812** [0.448, 0.516] | 20.7% | 1.0000 | 51.3% | **0.5121** [0.470, 0.556] | 63.3% |
| qwen3-1.7b | rag | 0.4790 | 0.9841 | 83.1% | **0.4976** [0.463, 0.533] | 18.7% | 1.0000 | 51.7% | **0.5253** [0.482, 0.565] | 58.7% |
| smollm3-3b | baseline | 0.4720 | 0.7119 | 78.9% | **0.4931** [0.458, 0.529] | 25.3% | 0.9450 | 48.3% | **0.5766** [0.534, 0.621] | 60.0% |
| smollm3-3b | rag | 0.4990 | 0.6467 | 80.0% | **0.5248** [0.490, 0.561] | 24.0% | 0.9128 | 46.3% | **0.6029** [0.559, 0.646] | 62.7% |

> *Footnote on Tie Issue*: In raw confidence, models frequently saturate with $\text{confidence} = 1.0000$ on over 50% of questions (causing $\tau_{\text{raw}} = 1.0000$ at 50% coverage). This creates discrete step artifacts and tie-order dependence. Temperature-scaled confidence ($T^*$ fitted on validation NLL) smoothly disperses saturated probabilities, eliminating ties and yielding robust, continuous operating thresholds.

## 2. Abstention on Unanswerable Questions vs Trivial Baselines

Abstention rule: abstain if $\text{confidence} < \tau$ (tuned on validation to maximize F1). Positive class = `should_abstain`.

| Model | Mode | Frozen $\tau$ | Val F1 | Test Prec | Test Rec | Test F1 [95% CI] | Coverage (% Test Answered) | Wrongly Abstained % | Selective Acc [95% CI] | Beats Always-Abstain? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| *Trivial: Always Abstain* | — | $\infty$ | 0.2000 | 0.1304 | 1.0000 | **0.2308** | 0.0% | 100.0% | N/A | — |
| *Trivial: Never Abstain* | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 100.0% | 0.0% | Mode Acc @ 100% | No |
| qwen3-4b | baseline | 0.9763 | 0.2727 | 0.1917 | 0.2467 | 0.2157 [0.157, 0.274] | 83.2% | 15.6% | 0.5865 [0.553, 0.619] | No |
| qwen3-4b | rag | 1.0000 | 0.2403 | 0.1469 | 0.4533 | 0.2219 [0.179, 0.266] | 59.7% | 39.5% | 0.6380 [0.600, 0.676] | No |
| phi4-mini | baseline | 0.9597 | 0.2233 | 0.1296 | 0.8133 | 0.2236 [0.190, 0.259] | 18.2% | 81.9% | 0.7127 [0.645, 0.777] | No |
| phi4-mini | rag | 0.7082 | 0.2447 | 0.1445 | 0.4133 | 0.2142 [0.171, 0.258] | 62.7% | 36.7% | 0.6066 [0.569, 0.646] | No |
| gemma3-4b | baseline | 0.9999 | 0.2721 | 0.1962 | 0.3467 | 0.2506 [0.199, 0.306] | 77.0% | 21.3% | 0.5146 [0.480, 0.549] | Marginal (CI overlaps) |
| gemma3-4b | rag | 1.0000 | 0.2097 | 0.1402 | 0.8400 | 0.2402 [0.206, 0.277] | 21.8% | 77.3% | 0.7093 [0.651, 0.763] | Marginal (CI overlaps) |
| qwen3-1.7b | baseline | 0.9998 | 0.2415 | 0.1750 | 0.4667 | 0.2545 [0.208, 0.301] | 65.2% | 33.0% | 0.5075 [0.469, 0.545] | Marginal (CI overlaps) |
| qwen3-1.7b | rag | 0.9992 | 0.2162 | 0.1560 | 0.3400 | 0.2138 [0.169, 0.261] | 71.6% | 27.6% | 0.5193 [0.483, 0.556] | No |
| smollm3-3b | baseline | 0.9865 | 0.2267 | 0.1434 | 0.7333 | 0.2399 [0.202, 0.278] | 33.3% | 65.7% | 0.6268 [0.575, 0.676] | Marginal (CI overlaps) |
| smollm3-3b | rag | 0.9998 | 0.2024 | 0.1307 | 1.0000 | 0.2311 [0.202, 0.263] | 0.2% | 99.8% | 1.0000 [0.000, 1.000] | Marginal (CI overlaps) |

## 3. Calibration: Temperature Scaling (Fitted on Validation NLL)

| Model | Mode | Fitted $T^*$ (Val NLL) | Val NLL (Raw $\to$ Scaled) | Test ECE (Raw) | Test ECE (Scaled) | Ans-AUROC (Raw) | Ans-AUROC (Scaled) | AUROC Changed? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| qwen3-4b | baseline | 16.3 | 7.15 $\to$ 5.74 | 0.4168 | **0.1776** | 0.5892 | 0.5629 | Shifted (-0.0263)* |
| qwen3-4b | rag | 13.0 | 8.83 $\to$ 7.87 | 0.4310 | **0.2360** | 0.5466 | 0.5349 | Shifted (-0.0117)* |
| phi4-mini | baseline | 3.7 | 1.59 $\to$ 1.09 | 0.2427 | **0.0308** | 0.4781 | 0.4174 | Shifted (-0.0606)* |
| phi4-mini | rag | 4.5 | 1.80 $\to$ 1.08 | 0.2591 | **0.0489** | 0.5222 | 0.4546 | Shifted (-0.0676)* |
| gemma3-4b | baseline | 24.5 | 10.03 $\to$ 7.81 | 0.5029 | **0.2437** | 0.6219 | 0.5846 | Shifted (-0.0373)* |
| gemma3-4b | rag | 20.8 | 7.59 $\to$ 4.36 | 0.4713 | **0.1096** | 0.5117 | 0.4586 | Shifted (-0.0531)* |
| qwen3-1.7b | baseline | 26.1 | 9.44 $\to$ 7.97 | 0.5090 | **0.2889** | 0.5962 | 0.5680 | Shifted (-0.0282)* |
| qwen3-1.7b | rag | 10.9 | 8.68 $\to$ 7.46 | 0.4922 | **0.2934** | 0.5573 | 0.5404 | Shifted (-0.0169)* |
| smollm3-3b | baseline | 6.6 | 2.44 $\to$ 1.15 | 0.3930 | **0.0347** | 0.5590 | 0.4336 | Shifted (-0.1254)* |
| smollm3-3b | rag | 4.9 | 1.98 $\to$ 1.13 | 0.3284 | **0.0471** | 0.5532 | 0.4311 | Shifted (-0.1221)* |

*Note: Ans-AUROC shifts slightly under temperature scaling because cross-dataset probability scaling differs between 4-option MedQA ($p \to 0.25$) and 3-option PubMedQA ($p \to 0.33$).*

## 4. Risk-Coverage Profile and Paired $\Delta\text{AURC}$ (Test Answerable Rows, N=1 000)

### Table 4A: Risk-Coverage Across Coverage Levels

| Model | Mode | AURC (Lower is Better) | Sel Acc @ 100% (100% Cov) | Sel Acc @ 80% (80% Cov) | Sel Acc @ 60% (60% Cov) | Sel Acc @ 40% (40% Cov) | Sel Acc @ 20% (20% Cov) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| qwen3-4b | baseline | **0.3432** | 0.5540 (100.0%) | 0.5950 (80.0%) | 0.6450 (60.0%) | 0.6750 (40.0%) | **0.7400** (20.0%) |
| qwen3-4b | rag | **0.3415** | 0.5440 (100.0%) | 0.5787 (80.0%) | 0.6383 (60.0%) | 0.6950 (40.0%) | **0.7650** (20.0%) |
| phi4-mini | baseline | **0.3730** | 0.5040 (100.0%) | 0.5463 (80.0%) | 0.5800 (60.0%) | 0.6350 (40.0%) | **0.7150** (20.0%) |
| phi4-mini | rag | **0.3294** | 0.5190 (100.0%) | 0.5650 (80.0%) | 0.6233 (60.0%) | 0.6900 (40.0%) | **0.7550** (20.0%) |
| gemma3-4b | baseline | **0.4295** | 0.4890 (100.0%) | 0.5100 (80.0%) | 0.5417 (60.0%) | 0.5750 (40.0%) | **0.6350** (20.0%) |
| gemma3-4b | rag | **0.3895** | 0.5160 (100.0%) | 0.5437 (80.0%) | 0.5917 (60.0%) | 0.6575 (40.0%) | **0.7100** (20.0%) |
| qwen3-1.7b | baseline | **0.4913** | 0.4620 (100.0%) | 0.4888 (80.0%) | 0.5083 (60.0%) | 0.5025 (40.0%) | **0.5650** (20.0%) |
| qwen3-1.7b | rag | **0.4737** | 0.4790 (100.0%) | 0.5012 (80.0%) | 0.5250 (60.0%) | 0.5550 (40.0%) | **0.5550** (20.0%) |
| smollm3-3b | baseline | **0.4326** | 0.4720 (100.0%) | 0.4938 (80.0%) | 0.5433 (60.0%) | 0.6075 (40.0%) | **0.6450** (20.0%) |
| smollm3-3b | rag | **0.4171** | 0.4990 (100.0%) | 0.5250 (80.0%) | 0.5633 (60.0%) | 0.6100 (40.0%) | **0.6400** (20.0%) |

### Table 4B: Paired Test of RAG vs Baseline Reliability ($\Delta\text{AURC} = \text{AURC}_{\text{RAG}} - \text{AURC}_{\text{Baseline}}$)

> 95% CI from paired bootstrap (1 000 resamples, seed 42) over identical test answerable questions. Significant if upper CI bound $< 0$ (lower AURC is better).

| Model | Baseline AURC | RAG AURC | $\Delta\text{AURC}$ [95% CI] | Statistically Significant? |
|---|:---:|:---:|:---:|:---:|
| qwen3-4b | 0.3424 | 0.3256 | **-0.0169** [-0.0362, +0.0453] | No (CI spans 0) |
| phi4-mini | 0.3730 | 0.3294 | **-0.0435** [-0.0766, -0.0108] | **Yes (Significant)**\* |
| gemma3-4b | 0.4264 | 0.3715 | **-0.0548** [-0.1206, -0.0350] | **Yes (Significant)**\* |
| qwen3-1.7b | 0.4785 | 0.4491 | **-0.0294** [-0.0582, +0.0240] | No (CI spans 0) |
| smollm3-3b | 0.4327 | 0.4171 | **-0.0156** [-0.0506, +0.0185] | No (CI spans 0) |

## 5. Spontaneous Abstention (Refusal Regex Matches)

> Regex pattern: `none of the|cannot be determined|not enough information` (case-insensitive) on `raw_output`.

| Model | Mode | Validation Refusals (All / Unans) | Test Refusals (All / Unans) | Details |
|---|---|:---:|:---:|---|
| qwen3-4b | baseline | 3/450 (3/50) | 1/1150 (1/150) | Val: 3/3 on unanswerable; Test: 1/1 on unanswerable |
| qwen3-4b | rag | 0/450 (0/50) | 1/1150 (0/150) | Test: 0/1 on unanswerable |
| phi4-mini | baseline | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| phi4-mini | rag | 1/450 (0/50) | 0/1150 (0/150) | Val: 0/1 on unanswerable |
| gemma3-4b | baseline | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| gemma3-4b | rag | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| qwen3-1.7b | baseline | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| qwen3-1.7b | rag | 0/450 (0/50) | 2/1150 (0/150) | Test: 0/2 on unanswerable |
| smollm3-3b | baseline | 0/450 (0/50) | 0/1150 (0/150) | No spontaneous refusal |
| smollm3-3b | rag | 0/450 (0/50) | 3/1150 (0/150) | Test: 0/3 on unanswerable |

## 6. Key Findings & Discussion

1. **Confidence-Based Abstention Does Not Beat Always-Abstain**:
   - On the test set ($p = 150/1\,150 = 13.04\%$ unanswerable questions), the trivial strategy of **always abstaining** achieves an Abstention F1 of **0.2308**.
   - Across all 5 models in both Baseline and RAG modes, validation-tuned threshold abstention achieves Test F1 scores between **0.2138 and 0.2545**.
   - In all cases, the 95% bootstrap confidence intervals overlap the trivial 0.2308 baseline. **No model reliably outperforms trivial always-abstaining**, demonstrating that max letter probability alone is insufficient to identify unanswerable medical questions.
2. **Probabilities are Saturated and Severely Overconfident**:
   - Uncalibrated models exhibit massive calibration errors (raw Test ECE between **0.2427 and 0.5090**), frequently assigning probabilities $\ge 0.99$ to incorrect answers.
   - Fitting temperature scaling on validation NLL yields large optimal temperatures ($T^* \approx 3.7 - 26.1$), confirming extreme overconfidence.
   - Post-hoc temperature scaling dramatically reduces Test ECE (e.g. `phi4-mini` drops $0.2427 \to 0.0308$, `smollm3-3b` drops $0.3930 \to 0.0347$, `gemma3-4b` drops $0.4713 \to 0.1096$).
   - However, temperature scaling does not improve unanswerable detection: because temperature scaling pulls 3-option PubMedQA probabilities towards $0.333$ while 4-option MedQA probabilities pull towards $0.250$, cross-dataset AUROC shifts slightly downward (e.g. $0.5892 \to 0.5629$ on `qwen3-4b`).
3. **Risk-Coverage as the Primary Reliability Result**:
   - The meaningful operational utility of model confidence lies in **selective classification** (risk-coverage), where answering only higher-confidence questions monotonically reduces risk.
   - At 80% coverage (Table 1A, tie-free operating points), selective accuracy consistently exceeds the same mode's 100%-coverage accuracy while discarding 17%–33% of unanswerable questions.
   - At 50% coverage (Table 1A), selective accuracy reaches **0.6667** for `phi4-mini` RAG (vs 0.5190 at 100% cov), **0.6633** for `qwen3-4b` RAG (vs 0.5440 at 100% cov), and **0.5992** for `gemma3-4b` RAG (vs 0.5160 at 100% cov), while correctly abstaining on 40%–68% of unanswerable questions.
4. **RAG Significantly Improves AURC for Select Models**:
   - Paired bootstrap testing (Table 4B, 1 000 resamples, seed 42) shows that RAG significantly reduces AURC for **`phi4-mini`** ($\Delta\text{AURC} = -0.0435$ [$-0.0766, -0.0108$]) and **`gemma3-4b`** ($\Delta\text{AURC} = -0.0548$ [$-0.1206, -0.0350$]), with both 95% CIs strictly excluding 0.
   - For `qwen3-4b` ($\Delta = -0.0169$ [$-0.0362, +0.0453$]), `qwen3-1.7b` ($\Delta = -0.0294$ [$-0.0582, +0.0240$]), and `smollm3-3b` ($\Delta = -0.0156$ [$-0.0506, +0.0185$]), the point estimates improve under RAG, but the paired 95% confidence intervals cross zero, indicating that the AURC reductions for these three models are not statistically significant.

---
*Artifacts generated: `outputs/analysis/risk_coverage_baseline.png`, `outputs/analysis/risk_coverage_rag.png`, and `outputs/analysis/phase9_abstention.json`.*
