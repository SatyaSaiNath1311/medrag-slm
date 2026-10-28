# Master Paper Fact Ledger

> [!IMPORTANT]
> **Rules of Evidence:**
> 1. Every empirical number cited in paper text, abstract, or executive summary must cite an ID (`F001...`) from this ledger.
> 2. Strict accuracy requires: parsed, untruncated, and correct. Fallback accuracy is secondary and explicitly labelled.
> 3. All numbers are computed from committed repository outputs and cryptographic frozen splits; no numbers are estimated.

| ID | Plain-Language Statement | Empirical Value | 95% Confidence Interval | Split | N | Source File | Provenance Reference |
|---|---|---|---|---|---|---|---|
| **F001** | Pair A MedQA C1 (closed-book) strict accuracy gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking) | `+29.8 pp (+0.298)` | [+0.250, +0.346] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 7 |
| **F002** | Pair A MedQA C1 with-fallback accuracy gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking) [Secondary] | `+28.0 pp (+0.280)` | [+0.230, +0.330] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 8 |
| **F003** | Pair A MedQA C1 truncation rate gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking) | `-19.0 pp (-0.190)` | [-0.244, -0.136] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 9 |
| **F004** | Pair A MedQA C1 MedPsy-1.7B strict accuracy | `60.2% (0.602)` | [0.558, 0.644] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 5 |
| **F005** | Pair A MedQA C1 MedPsy-1.7B with-fallback accuracy [Secondary] | `77.8% (0.778)` | [0.740, 0.812] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 5 |
| **F006** | Pair A MedQA C1 MedPsy-1.7B truncation rate (160/500) | `32.0% (0.320)` | [0.280, 0.363] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 5 |
| **F007** | Pair A MedQA C1 Qwen3-1.7B-Thinking strict accuracy | `30.4% (0.304)` | [0.265, 0.346] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 6 |
| **F008** | Pair A MedQA C1 Qwen3-1.7B-Thinking with-fallback accuracy [Secondary] | `49.8% (0.498)` | [0.454, 0.542] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 6 |
| **F009** | Pair A MedQA C1 Qwen3-1.7B-Thinking truncation rate (255/500) | `51.0% (0.510)` | [0.466, 0.554] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 6 |
| **F010** | Pair A MedQA C1 exact two-sided McNemar p-value (strict accuracy difference) | `<1e-4 (5.53e-28)` | N/A | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 7 |
| **F011** | Pair A MedQA C2 (full RAG) strict accuracy gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking) | `+12.0 pp (+0.120)` | [+0.070, +0.170] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 15 |
| **F012** | Pair A MedQA C2 with-fallback accuracy gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking) [Secondary] | `+25.2 pp (+0.252)` | [+0.204, +0.300] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 16 |
| **F013** | Pair A MedQA C2 truncation rate gap (MedPsy-1.7B minus Qwen3-1.7B-Thinking) | `-2.8 pp (-0.028)` | [-0.082, +0.026] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 17 |
| **F014** | Pair A MedQA C2 MedPsy-1.7B strict accuracy | `45.2% (0.452)` | [0.409, 0.496] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 13 |
| **F015** | Pair A MedQA C2 MedPsy-1.7B with-fallback accuracy [Secondary] | `78.4% (0.784)` | [0.746, 0.818] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 13 |
| **F016** | Pair A MedQA C2 MedPsy-1.7B truncation rate (233/500) | `46.6% (0.466)` | [0.423, 0.510] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 13 |
| **F017** | Pair A MedQA C2 Qwen3-1.7B-Thinking strict accuracy | `33.2% (0.332)` | [0.292, 0.374] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 14 |
| **F018** | Pair A MedQA C2 Qwen3-1.7B-Thinking with-fallback accuracy [Secondary] | `53.2% (0.532)` | [0.488, 0.575] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 14 |
| **F019** | Pair A MedQA C2 Qwen3-1.7B-Thinking truncation rate (247/500) | `49.4% (0.494)` | [0.450, 0.538] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 14 |
| **F020** | Pair A MedQA RAG within-model effect on MedPsy-1.7B strict accuracy (C2 minus C1) | `-15.0 pp (-0.150)` | [-0.198, -0.102] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 29 |
| **F021** | Pair A MedQA RAG within-model effect on MedPsy-1.7B truncation rate (C2 minus C1) | `+14.6 pp (+0.146)` | [+0.094, +0.198] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 31 |
| **F022** | Pair A MedQA RAG within-model effect on Qwen3-1.7B-Thinking strict accuracy (C2 minus C1) | `+2.8 pp (+0.028)` | [-0.016, +0.072] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 32 |
| **F023** | Pair A MedQA difference in RAG effect (MedPsy minus Qwen strict accuracy interaction) | `-17.8 pp (-0.178)` | [-0.238, -0.116] | test | 500 | `outputs/analysis/pair_a_medqa.md` | Line 35 |
| **F024** | Pair A Unanswerable v2 C4 truncation rate for MedPsy-1.7B (28/140) | `20.0% (0.200)` | [0.142, 0.274] | test | 140 | `outputs/analysis/pair_a_c45.md` | Line 17 |
| **F025** | Pair A Unanswerable v2 C4 truncation rate for Qwen3-1.7B-Thinking (29/140) | `20.7% (0.207)` | [0.148, 0.282] | test | 140 | `outputs/analysis/pair_a_c45.md` | Line 16 |
| **F026** | Pair A Unanswerable v2 C5 truncation rate for MedPsy-1.7B (45/140) | `32.1% (0.321)` | [0.249, 0.403] | test | 140 | `outputs/analysis/pair_a_c45.md` | Line 19 |
| **F027** | Pair A Unanswerable v2 C5 truncation rate for Qwen3-1.7B-Thinking (60/140) | `42.9% (0.429)` | [0.349, 0.512] | test | 140 | `outputs/analysis/pair_a_c45.md` | Line 18 |
| **F028** | Gemma Pair MedQA C1 strict accuracy gap (MedGemma minus Gemma3) | `+6.20 pp (+0.0620)` | [+0.0080, +0.1140] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 54 |
| **F029** | Gemma Pair MedQA C1 Gemma-3-4B strict accuracy | `48.60% (0.4860)` | [0.4425, 0.5297] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 29 |
| **F030** | Gemma Pair MedQA C1 MedGemma-1.5-4B-it strict accuracy | `54.80% (0.5480)` | [0.5042, 0.5911] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 30 |
| **F031** | Gemma Pair MedQA C1 raw McNemar p-value | `0.02548` | N/A | test | 500 | `outputs/analysis/gemma_pair.md` | Line 54 |
| **F032** | Gemma Pair MedQA C1 Holm-adjusted p-value across m=5 family | `0.12742 (ns at alpha=0.05)` | N/A | test | 500 | `outputs/analysis/gemma_pair.md` | Line 54 |
| **F033** | Gemma Pair PubMedQA C1 strict accuracy gap (MedGemma minus Gemma3) | `+1.60 pp (+0.0160)` | [-0.0240, +0.0540] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 55 |
| **F034** | Gemma Pair MedQA C2 strict accuracy gap (MedGemma minus Gemma3) | `+4.20 pp (+0.0420)` | [-0.0040, +0.0880] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 56 |
| **F035** | Gemma Pair PubMedQA C2 strict accuracy gap (MedGemma minus Gemma3) | `+0.80 pp (+0.0080)` | [-0.0300, +0.0480] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 57 |
| **F036** | Gemma Pair PubMedQA C3 (oracle abstract) strict accuracy gap (MedGemma minus Gemma3) | `+4.00 pp (+0.0400)` | [+0.0040, +0.0760] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 58 |
| **F037** | Gemma Pair MedQA within-model RAG effect for Gemma-3-4B (C2 minus C1) | `+4.40 pp (+0.0440)` | [-0.0060, +0.0940] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 78 |
| **F038** | Gemma Pair MedQA within-model RAG effect for MedGemma-1.5-4B-it (C2 minus C1) | `+2.40 pp (+0.0240)` | [-0.0240, +0.0720] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 79 |
| **F039** | Gemma Pair PubMedQA majority-class baseline share ('yes') | `55.20% (276/500)` | [0.5082, 0.5950] | test | 500 | `outputs/analysis/gemma_pair.md` | Line 98 |
| **F040** | Pair B MedQA C1 strict accuracy gap (MedPsy-4B minus Qwen3-4B-Thinking at 1,024 cap) | `+82.8 pp (+0.828)` | [+0.794, +0.860] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 8 |
| **F041** | Pair B MedQA C1 with-fallback accuracy gap (MedPsy-4B minus Qwen3-4B-Thinking) [Secondary] | `+20.8 pp (+0.208)` | [+0.164, +0.252] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 9 |
| **F042** | Pair B MedQA C1 MedPsy-4B strict accuracy | `83.2% (0.832)` | [0.797, 0.862] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 6 |
| **F043** | Pair B MedQA C1 MedPsy-4B with-fallback accuracy [Secondary] | `87.6% (0.876)` | [0.844, 0.902] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 6 |
| **F044** | Pair B MedQA C1 MedPsy-4B truncation rate (53/500) | `10.6% (0.106)` | [0.081, 0.136] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 6 |
| **F045** | Pair B MedQA C1 Qwen3-4B-Thinking-2507 strict accuracy | `0.4% (0.004)` | [0.001, 0.014] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 7 |
| **F046** | Pair B MedQA C1 Qwen3-4B-Thinking-2507 with-fallback accuracy [Secondary] | `66.8% (0.668)` | [0.626, 0.708] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 7 |
| **F047** | Pair B MedQA C1 Qwen3-4B-Thinking-2507 truncation rate (498/500) | `99.6% (0.996)` | [0.986, 0.999] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 7 |
| **F048** | Pair B MedQA C2 strict accuracy gap (MedPsy-4B minus Qwen3-4B-Thinking at 1,024 cap) | `+72.6 pp (+0.726)` | [+0.688, +0.764] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 14 |
| **F049** | Pair B MedQA C2 MedPsy-4B strict accuracy | `72.6% (0.726)` | [0.685, 0.763] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 12 |
| **F050** | Pair B MedQA C2 MedPsy-4B truncation rate (108/500) | `21.6% (0.216)` | [0.182, 0.254] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 12 |
| **F051** | Pair B MedQA C2 Qwen3-4B-Thinking-2507 strict accuracy | `0.0% (0.000)` | [0.000, 0.008] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 13 |
| **F052** | Pair B MedQA C2 Qwen3-4B-Thinking-2507 truncation rate (500/500) | `100.0% (1.000)` | [0.992, 1.000] | test | 500 | `outputs/analysis/pair_b_equal_budget.md` | Line 13 |
| **F053** | Pair B Budget Sweep Cap 1024 strict accuracy gap (MedPsy minus Qwen) | `+88.0 pp (+0.880)` | [+0.810, +0.940] | val | 100 | `outputs/analysis/budget_pair_b.md` | Line 55 |
| **F054** | Pair B Budget Sweep Cap 1536 strict accuracy gap (MedPsy minus Qwen) | `+86.0 pp (+0.860)` | [+0.790, +0.920] | val | 100 | `outputs/analysis/budget_pair_b.md` | Line 57 |
| **F055** | Pair B Budget Sweep Cap 2048 strict accuracy gap (MedPsy minus Qwen) | `+74.0 pp (+0.740)` | [+0.650, +0.820] | val | 100 | `outputs/analysis/budget_pair_b.md` | Line 59 |
| **F056** | Pair B Budget Sweep Cap 3072 strict accuracy gap (MedPsy minus Qwen) | `+43.0 pp (+0.430)` | [+0.320, +0.530] | val | 100 | `outputs/analysis/budget_pair_b.md` | Line 61 |
| **F057** | Pair B Budget Sweep Cap 4096 strict accuracy gap (MedPsy minus Qwen) | `+32.0 pp (+0.320)` | [+0.220, +0.420] | val | 100 | `outputs/analysis/budget_pair_b.md` | Line 63 |
| **F058** | Pair B Budget Sweep Cap 4096 Qwen3-4B-Thinking-2507 residual truncation rate | `28.0% (0.280)` | [0.198, 0.379] | val | 100 | `outputs/analysis/budget_pair_b.md` | Line 64 |
| **F059** | Gemma Prompt Sensitivity V1 (standard baseline) gap (Gemma3 minus MedGemma) | `-4.50 pp (-0.0450)` | [-0.0900, -0.0050] | val | 200 | `outputs/analysis/gemma_prompt_sensitivity.md` | Section 3 Table |
| **F060** | Gemma Prompt Sensitivity V2 (clinical scenario) gap (Gemma3 minus MedGemma) | `-5.00 pp (-0.0500)` | [-0.1000, -0.0050] | val | 200 | `outputs/analysis/gemma_prompt_sensitivity.md` | Section 3 Table |
| **F061** | Gemma Prompt Sensitivity V3 (board examination) gap (Gemma3 minus MedGemma) | `-2.50 pp (-0.0250)` | [-0.0750, +0.0250] | val | 200 | `outputs/analysis/gemma_prompt_sensitivity.md` | Section 3 Table |
| **F062** | Gemma Prompt Sensitivity gap spread across variants (max - min) | `2.50 pp (0.0250)` | <= 10.0 pp threshold | val | 200 | `outputs/analysis/gemma_prompt_sensitivity.md` | Section 1 |
| **F063** | Gemma Prompt Sensitivity pre-registered stability verdict | `STABLE` | N/A | val | 200 | `outputs/analysis/gemma_prompt_sensitivity.md` | Section 1 |
| **F064** | MedPsy-4B frozen test greedy accuracy on MedQA | `72.88% (0.7288)` | N/A | test | 1272 | `outputs/analysis/medpsy_reliability.md` | Section 1 |
| **F065** | MedPsy-4B self-consistency SC-5 accuracy on MedQA test | `74.06% (0.7406)` | N/A | test | 1272 | `outputs/analysis/medpsy_reliability.md` | Section 1 |
| **F066** | MedPsy-4B calibrated reliability score abstention F1 (max-F1 threshold from val) | `0.555` | [0.495, 0.609] | test | 650 | `outputs/analysis/medpsy_reliability.md` | Line 15 |
| **F067** | MedPsy-4B unanswerable recall under calibrated reliability score | `69.3%` | [61.7, 76.2] | test | 650 | `outputs/analysis/medpsy_reliability.md` | Line 15 |
| **F068** | MedPsy-4B accuracy when answering under safe reliability threshold (coverage 72.8%) | `97.5%` | [95.7, 99.1] | test | 650 | `outputs/analysis/medpsy_reliability.md` | Line 14 |
| **F069** | MedPsy-4B unsafe answer rate under safe reliability threshold | `8.3%` | [6.3, 10.6] | test | 650 | `outputs/analysis/medpsy_reliability.md` | Line 14 |
| **F070** | Pair B budget sweep execution GPU hours (2x T4 GPU) | `6.92 hours` | N/A | val | 100 | `outputs/sweeps/budget_pair_b/manifest.json` | Key 'hours_used' |
| **F071** | Gemma prompt sensitivity execution GPU hours (2x T4 GPU) | `0.37 hours (22.0 min)` | N/A | val | 600 | `outputs/sweeps/gemma_prompt_sensitivity/manifest.json` | Key 'hours_used' |
| **F072** | MedPsy-4B baseline latency on MedQA test | `37.3 s/question` | N/A | test | 500 | `outputs/analysis/medpsy_rag.md` | Line 5 |
| **F073** | MedPsy-4B RAG latency on MedQA test | `63.9 s/question` | N/A | test | 500 | `outputs/analysis/medpsy_rag.md` | Line 6 |
