# Presentation Figures

All figures are rendered at 300 DPI (`.png`) and vector graphics (`.svg`) using a unified, colour-blind-safe aesthetic:
- **Baseline**: Slate Grey (`#6C757D`)
- **Full RAG**: Strong Blue (`#1F77B4`)
- **Adaptive Gate**: Green (`#2CA02C`)
- **+Abstract Context**: Vivid Orange (`#FF7F0E`)

---

### Figure Captions for Slides

| File | Figure Title | Slide-Ready One-Sentence Caption |
|---|---|---|
| **fig1_overall_accuracy** | Overall QA Accuracy | Adaptive gating improves accuracy over baseline and full RAG across all models, achieving Holm-significant gains in three architectures (*). |
| **fig2_medqa_pubmedqa** | MedQA vs. PubMedQA Divergence | Retrieval provides a significant, consistent benefit on MedQA across all architectures (pooled CMH OR = 1.21, p < 0.001), but yields zero net gain on PubMedQA. |
| **fig3_pubmedqa_abstract** | PubMedQA Abstract Benchmark | Open-domain textbook RAG fails on study-specific queries, whereas supplying the actual study abstract boosts accuracy by over 20 percentage points above majority baseline. |
| **fig4_accuracy_vs_latency** | Accuracy vs. Latency Trade-Off | Single-generation Rerank Gating recovers the accuracy gains of two-pass confidence gating with near-zero latency overhead over pure retrieval. |
| **fig5_risk_coverage** | Risk-Coverage Abstention Curves | Abstaining on low model confidence monotonically increases selective accuracy from ~52% to >75% as coverage narrows from 100% to 20%. |
| **fig6_retrieval_bottleneck** | Retrieval Bottleneck & DiD | Accuracy gains from RAG are strictly concentrated on the 30% of questions where the gold answer is successfully retrieved, yielding a pooled difference-in-differences of +10.2 points. |
| **fig7_fixed_vs_induced** | MedQA Error Transitions | While RAG induces 41–65 novel errors where models cite incorrect retrieved distractors, fixed questions consistently dominate, producing a net gain of +10 to +30 answers per model. |
| **fig8_overconfidence** | Persistent Overconfidence | Both baseline and RAG exhibit severe overconfidence, assigning >=0.90 probability to 20–95% of wrong answers and unanswerable questions alike. |
| **fig9_calibration** | Phi-4-mini Reliability Diagram | Post-hoc temperature scaling (T=4.49) successfully resolves logit over-dispersion, reducing expected calibration error (ECE) from 25.9% down to 4.9%. |
| **fig10_memory** | Peak VRAM Footprint | Four out of five models run comfortably on a single 16 GB T4 GPU with RAG adding only 1.2–1.8 GB of VRAM overhead. |
