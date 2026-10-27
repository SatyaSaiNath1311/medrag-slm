# Pair A, MedQA test, 1,024-token cap (generic = qwen3-1.7b-thinking, domain = medpsy-1.7b)
Strict = parsed and not truncated and correct (PRIMARY). Fallback = stored correct, incl. forced letters after truncation (SECONDARY). Truncation is analysed as its own outcome.

## C1
- medpsy-1.7b: truncated 160/500 (32.0%); strict 301/500 = 0.602 [0.558, 0.644]; fallback 389/500 = 0.778 [0.740, 0.812]
- qwen3-1.7b-thinking: truncated 255/500 (51.0%); strict 152/500 = 0.304 [0.265, 0.346]; fallback 249/500 = 0.498 [0.454, 0.542]
- paired medpsy-1.7b minus qwen3-1.7b-thinking (strict): +0.298 [+0.250, +0.346]; medpsy-1.7b-only 176, qwen3-1.7b-thinking-only 27; McNemar raw p = 5.53e-28
- paired medpsy-1.7b minus qwen3-1.7b-thinking (fallback): +0.280 [+0.230, +0.330]; medpsy-1.7b-only 173, qwen3-1.7b-thinking-only 33; McNemar raw p = 4.18e-24
- paired medpsy-1.7b minus qwen3-1.7b-thinking (truncated): -0.190 [-0.244, -0.136]; medpsy-1.7b-only 59, qwen3-1.7b-thinking-only 154; McNemar raw p = 5.72e-11
- DESCRIPTIVE, selected subset where BOTH finished (n=184): medpsy-1.7b 0.918, qwen3-1.7b-thinking 0.690; diff +0.228 [+0.163, +0.293]

## Holm-adjusted p-values (one family: every paired test above)
- C1 strict: raw 5.53e-28, Holm 1.66e-27
- C1 fallback: raw 4.18e-24, Holm 8.37e-24
- C1 truncated: raw 5.72e-11, Holm 5.72e-11
