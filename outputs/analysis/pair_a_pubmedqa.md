# Pair A, PubMedQA test, 1,024-token cap (generic = qwen3-1.7b-thinking, domain = medpsy-1.7b)
Strict = parsed and not truncated and correct (PRIMARY). Fallback = stored correct, incl. forced letters after truncation (SECONDARY). Truncation is analysed as its own outcome.

## C1
- medpsy-1.7b: truncated 11/500 (2.2%); strict 245/500 = 0.490 [0.446, 0.534]; fallback 249/500 = 0.498 [0.454, 0.542]
- qwen3-1.7b-thinking: truncated 7/500 (1.4%); strict 234/500 = 0.468 [0.425, 0.512]; fallback 236/500 = 0.472 [0.429, 0.516]
- paired medpsy-1.7b minus qwen3-1.7b-thinking (strict): +0.022 [-0.024, +0.068]; medpsy-1.7b-only 75, qwen3-1.7b-thinking-only 64; McNemar raw p = 3.96e-01
- paired medpsy-1.7b minus qwen3-1.7b-thinking (fallback): +0.026 [-0.018, +0.072]; medpsy-1.7b-only 74, qwen3-1.7b-thinking-only 61; McNemar raw p = 3.02e-01
- paired medpsy-1.7b minus qwen3-1.7b-thinking (truncated): +0.008 [-0.008, +0.024]; medpsy-1.7b-only 11, qwen3-1.7b-thinking-only 7; McNemar raw p = 4.81e-01
- DESCRIPTIVE, selected subset where BOTH finished (n=482): medpsy-1.7b 0.502, qwen3-1.7b-thinking 0.475; diff +0.027 [-0.019, +0.073]

## C2
- medpsy-1.7b: truncated 176/500 (35.2%); strict 159/500 = 0.318 [0.279, 0.360]; fallback 224/500 = 0.448 [0.405, 0.492]
- qwen3-1.7b-thinking: truncated 78/500 (15.6%); strict 166/500 = 0.332 [0.292, 0.374]; fallback 188/500 = 0.376 [0.335, 0.419]
- paired medpsy-1.7b minus qwen3-1.7b-thinking (strict): -0.014 [-0.064, +0.036]; medpsy-1.7b-only 77, qwen3-1.7b-thinking-only 84; McNemar raw p = 6.36e-01
- paired medpsy-1.7b minus qwen3-1.7b-thinking (fallback): +0.072 [+0.024, +0.120]; medpsy-1.7b-only 99, qwen3-1.7b-thinking-only 63; McNemar raw p = 5.79e-03
- paired medpsy-1.7b minus qwen3-1.7b-thinking (truncated): +0.196 [+0.144, +0.248]; medpsy-1.7b-only 144, qwen3-1.7b-thinking-only 46; McNemar raw p = 5.88e-13
- DESCRIPTIVE, selected subset where BOTH finished (n=278): medpsy-1.7b 0.493, qwen3-1.7b-thinking 0.410; diff +0.083 [+0.018, +0.147]

## C3
- medpsy-1.7b: truncated 2/500 (0.4%); strict 259/500 = 0.518 [0.474, 0.561]; fallback 259/500 = 0.518 [0.474, 0.561]
- qwen3-1.7b-thinking: truncated 63/500 (12.6%); strict 153/500 = 0.306 [0.267, 0.348]; fallback 181/500 = 0.362 [0.321, 0.405]
- paired medpsy-1.7b minus qwen3-1.7b-thinking (strict): +0.212 [+0.160, +0.264]; medpsy-1.7b-only 152, qwen3-1.7b-thinking-only 46; McNemar raw p = 1.95e-14
- paired medpsy-1.7b minus qwen3-1.7b-thinking (fallback): +0.156 [+0.104, +0.208]; medpsy-1.7b-only 134, qwen3-1.7b-thinking-only 56; McNemar raw p = 1.47e-08
- paired medpsy-1.7b minus qwen3-1.7b-thinking (truncated): -0.122 [-0.152, -0.094]; medpsy-1.7b-only 1, qwen3-1.7b-thinking-only 62; McNemar raw p = 1.39e-17
- DESCRIPTIVE, selected subset where BOTH finished (n=436): medpsy-1.7b 0.518, qwen3-1.7b-thinking 0.351; diff +0.167 [+0.112, +0.222]

## Holm-adjusted p-values (one family: every paired test above)
- C1 strict: raw 3.96e-01, Holm 1.00e+00
- C1 fallback: raw 3.02e-01, Holm 1.00e+00
- C1 truncated: raw 4.81e-01, Holm 1.00e+00
- C2 strict: raw 6.36e-01, Holm 1.00e+00
- C2 fallback: raw 5.79e-03, Holm 2.90e-02
- C2 truncated: raw 5.88e-13, Holm 4.12e-12
- C3 strict: raw 1.95e-14, Holm 1.56e-13
- C3 fallback: raw 1.47e-08, Holm 8.82e-08
- C3 truncated: raw 1.39e-17, Holm 1.25e-16
