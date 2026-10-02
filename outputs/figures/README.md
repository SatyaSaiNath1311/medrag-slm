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
| **fig1_overall_accuracy** | Overall QA Accuracy | Adaptive gating improves accuracy over baseline for all 5 models; gains are Holm-significant for 3 (*). |
| **fig2_medqa_pubmedqa** | MedQA vs. PubMedQA Divergence | RAG significantly improves MedQA across models (pooled CMH OR = 1.21, p < 0.001) but has no significant effect on PubMedQA (pooled p = 0.25). |
| **fig3_pubmedqa_abstract** | PubMedQA Abstract Benchmark | Textbook RAG does not help study-specific PubMedQA questions; supplying the study abstract raises accuracy by 18–26 points over question-only, to 66–75%. |
| **fig4_accuracy_vs_latency** | Accuracy vs. Latency Trade-Off | Single-generation rerank gating matches or exceeds Full RAG accuracy at 35–50% lower latency; two-pass confidence gating is most accurate but slowest. |
| **fig5_risk_coverage** | Risk-Coverage Abstention Curves | Answering only the most confident questions raises accuracy, up to 76.5% at 20% coverage (Qwen3-4B, RAG). |
| **fig6_retrieval_bottleneck** | Retrieval Bottleneck & DiD | RAG gains are concentrated where the gold answer is retrieved: same direction for all 5 models, pooled difference-in-differences +10.2 points [5.8, 14.6]. |
| **fig7_fixed_vs_induced** | MedQA Error Transitions | On MedQA, RAG fixes 64–87 questions and breaks 41–65, a net gain of +10 to +30 per model. |
| **fig8_overconfidence** | Persistent Overconfidence | Both baseline and RAG exhibit severe overconfidence, assigning >=0.90 probability to 20–95% of wrong answers and unanswerable questions alike. |
| **fig9_calibration** | Phi-4-mini Reliability Diagram | Temperature scaling reduces calibration error (ECE) from 25.9% to 4.9% for Phi-4-mini (RAG). |
| **fig10_memory** | Peak VRAM Footprint | Four out of five models run comfortably on a single 16 GB T4 GPU with RAG adding only 1.2–1.8 GB of VRAM overhead. |
