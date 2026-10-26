# Pair B, equal 1,024-token budget, MedQA test (n=500)
Base = Qwen3-4B-Thinking-2507; domain = MedPsy-4B. Same prompt, cap, evidence and fallback rule.
Strict = parsed and not truncated and correct (primary). With-fallback = stored correct, i.e. a forced letter after a truncated thought (secondary).

## C1 closed-book
- MedPsy-4B: truncated 53/500; strict 416/500 = 0.832 [0.797, 0.862]; fallback 438/500 = 0.876 [0.844, 0.902]
- Qwen3-4B-Thinking-2507: truncated 498/500; strict 2/500 = 0.004 [0.001, 0.014]; fallback 334/500 = 0.668 [0.626, 0.708]
- paired MedPsy minus base (strict): +0.828 [+0.794, +0.860]; MedPsy-only correct 414, base-only correct 0, McNemar p = 4.73e-125
- paired MedPsy minus base (fallback): +0.208 [+0.164, +0.252]; MedPsy-only correct 128, base-only correct 24, McNemar p = 2.35e-18

## C2 RAG
- MedPsy-4B: truncated 108/500; strict 363/500 = 0.726 [0.685, 0.763]; fallback 445/500 = 0.890 [0.860, 0.915]
- Qwen3-4B-Thinking-2507: truncated 500/500; strict 0/500 = 0.000 [0.000, 0.008]; fallback 302/500 = 0.604 [0.560, 0.646]
- paired MedPsy minus base (strict): +0.726 [+0.688, +0.764]; MedPsy-only correct 363, base-only correct 0, McNemar p = 1.06e-109
- paired MedPsy minus base (fallback): +0.286 [+0.240, +0.332]; MedPsy-only correct 159, base-only correct 16, McNemar p = 8.46e-31

## RAG gain (C2 minus C1), with-fallback, paired within model
- MedPsy-4B: +0.014 [-0.016, +0.042]; RAG-only correct 30, closed-book-only correct 23, McNemar p = 4.10e-01
- Qwen3-4B-Thinking-2507: -0.064 [-0.108, -0.020]; RAG-only correct 47, closed-book-only correct 79, McNemar p = 5.54e-03
