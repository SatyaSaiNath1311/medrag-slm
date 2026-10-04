# MedPsy-4B: baseline vs RAG on MedQA test (n = 500, paired)

| Mode | Accuracy [95% CI] | Truncated | Acc. when answering (abstain if truncated) | Coverage | s/question |
|---|---|---|---|---|---|
| Baseline (no RAG) | 87.6% [84.8, 90.2] | 10.6% | 93.1% | 89.4% | 37.3 |
| RAG (top-5 evidence) | 89.0% [86.4, 91.6] | 21.6% | 92.6% | 78.4% | 63.9 |

Exact McNemar (paired): RAG-only correct = 30, baseline-only correct = 23, p = 0.41.
Accuracy change: +1.4 percentage points.
