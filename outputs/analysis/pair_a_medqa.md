# Pair A, MedQA test, 1,024-token cap (generic = qwen3-1.7b-thinking, domain = medpsy-1.7b)
Strict = parsed and not truncated and correct (PRIMARY). Fallback = stored correct, incl. forced letters after truncation (SECONDARY). Truncation is analysed as its own outcome.

## C1
- medpsy-1.7b: truncated 160/500 (32.0%); strict 301/500 = 0.602 [0.558, 0.644]; fallback 389/500 = 0.778 [0.740, 0.812]
- qwen3-1.7b-thinking: truncated 255/500 (51.0%); strict 152/500 = 0.304 [0.265, 0.346]; fallback 249/500 = 0.498 [0.454, 0.542]
- paired medpsy-1.7b minus qwen3-1.7b-thinking (strict): +0.298 [+0.250, +0.346]; medpsy-1.7b-only 176, qwen3-1.7b-thinking-only 27; McNemar raw p = 5.53e-28
- paired medpsy-1.7b minus qwen3-1.7b-thinking (fallback): +0.280 [+0.230, +0.330]; medpsy-1.7b-only 173, qwen3-1.7b-thinking-only 33; McNemar raw p = 4.18e-24
- paired medpsy-1.7b minus qwen3-1.7b-thinking (truncated): -0.190 [-0.244, -0.136]; medpsy-1.7b-only 59, qwen3-1.7b-thinking-only 154; McNemar raw p = 5.72e-11
- DESCRIPTIVE, selected subset where BOTH finished (n=184): medpsy-1.7b 0.918, qwen3-1.7b-thinking 0.690; diff +0.228 [+0.163, +0.293]

## C2
- medpsy-1.7b: truncated 233/500 (46.6%); strict 226/500 = 0.452 [0.409, 0.496]; fallback 392/500 = 0.784 [0.746, 0.818]
- qwen3-1.7b-thinking: truncated 247/500 (49.4%); strict 166/500 = 0.332 [0.292, 0.374]; fallback 266/500 = 0.532 [0.488, 0.575]
- paired medpsy-1.7b minus qwen3-1.7b-thinking (strict): +0.120 [+0.070, +0.170]; medpsy-1.7b-only 112, qwen3-1.7b-thinking-only 52; McNemar raw p = 3.21e-06
- paired medpsy-1.7b minus qwen3-1.7b-thinking (fallback): +0.252 [+0.204, +0.300]; medpsy-1.7b-only 153, qwen3-1.7b-thinking-only 27; McNemar raw p = 1.45e-22
- paired medpsy-1.7b minus qwen3-1.7b-thinking (truncated): -0.028 [-0.082, +0.026]; medpsy-1.7b-only 85, qwen3-1.7b-thinking-only 99; McNemar raw p = 3.38e-01
- DESCRIPTIVE, selected subset where BOTH finished (n=167): medpsy-1.7b 0.886, qwen3-1.7b-thinking 0.713; diff +0.174 [+0.108, +0.246]

## Holm-adjusted p-values (one family: every paired test above)
- C1 strict: raw 5.53e-28, Holm 3.32e-27
- C1 fallback: raw 4.18e-24, Holm 2.09e-23
- C1 truncated: raw 5.72e-11, Holm 1.72e-10
- C2 strict: raw 3.21e-06, Holm 6.42e-06
- C2 fallback: raw 1.45e-22, Holm 5.81e-22
- C2 truncated: raw 3.38e-01, Holm 3.38e-01

## RAG effect (C2 minus C1), paired within model
- medpsy-1.7b (strict): -0.150 [-0.198, -0.102]; C2-only 43, C1-only 118; McNemar p = 2.85e-09
- medpsy-1.7b (fallback): +0.006 [-0.030, +0.042]; C2-only 47, C1-only 44; McNemar p = 8.34e-01
- medpsy-1.7b (truncated): +0.146 [+0.094, +0.198]; C2-only 129, C1-only 56; McNemar p = 8.24e-08
- qwen3-1.7b-thinking (strict): +0.028 [-0.016, +0.072]; C2-only 70, C1-only 56; McNemar p = 2.47e-01
- qwen3-1.7b-thinking (fallback): +0.034 [-0.012, +0.080]; C2-only 75, C1-only 58; McNemar p = 1.65e-01
- qwen3-1.7b-thinking (truncated): -0.016 [-0.066, +0.036]; C2-only 83, C1-only 91; McNemar p = 5.96e-01
- difference in RAG effect, medpsy-1.7b minus qwen3-1.7b-thinking (strict): -0.178 [-0.238, -0.116]
- difference in RAG effect, medpsy-1.7b minus qwen3-1.7b-thinking (fallback): -0.028 [-0.082, +0.026]
