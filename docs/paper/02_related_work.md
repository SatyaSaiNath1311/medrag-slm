# 2. Related Work

## 2.1 Domain-Specialized Medical Language Models
The application of natural language processing to biomedicine has progressed from masked language models (e.g., BioBERT, ClinicalBERT) to generative foundation models. Frontier general-purpose models such as GPT-4 and Med-PaLM 2 demonstrated that massive parameter scale enables passing scores on the United States Medical Licensing Examination (USMLE). To achieve comparable domain mastery without prohibitive compute costs, specialized models were developed via domain continual pre-training (CPT) and instruction tuning on biomedical corpora.

Notable medical-domain models include:
- **MEDITRON-70B and PMC-LLaMA-13B**: Pre-trained on PubMed articles and clinical guidelines, demonstrating strong factual recall but retaining large parameter footprints.
- **MedGemma-4B and MedGemma-1.5-4B-IT**: Developed by Google Health through instruction-tuning Gemma base architectures on multi-task medical datasets (MedQA, MedMCQA, clinical notes).
- **MedPsy (1.7B and 4B)**: Domain-specialized reasoning models optimized for clinical psychiatry and multi-step diagnostic differential generation, employing chain-of-thought (CoT) tokens.

Despite these developments, prior benchmarks typically compare domain models against generic baselines of unequal scale or distinct model families, obscuring the true marginal gain of domain training.

## 2.2 Retrieval-Augmented Generation in Clinical Decision Support
Retrieval-Augmented Generation (RAG) addresses parametric knowledge decay and hallucination by retrieving relevant text passages from an external corpus to condition model responses. In the medical domain, Xiong et al. introduced the MIRAGE benchmark (ACL Findings 2024), providing a standardized framework across diverse medical QA datasets (MedQA, PubMedQA, BioASQ) and demonstrating that hybrid retrieval (BM25 combined with MedCPT dense embeddings) consistently outperforms single-modality retrieval.

Recent studies examining domain models under RAG report conflicting findings:
- The **MedRGB benchmark** (*arXiv 2411.09213*) evaluated general LLMs and medical fine-tuned models side-by-side with 5 to 20 retrieved documents, finding that domain fine-tuning does not guarantee superior RAG performance; domain LLMs frequently hallucinate when retrieved snippets conflict with internal pre-trained priors.
- **Rare-case clinical retrieval evaluations** (*arXiv 2605.21807*) observed that general-purpose foundation models occasionally outperform domain-adapted models when provided with high-recall retrieval contexts.
- **Benchmarking SLMs for Medical QA** (*EIS 2026 Proceedings*) confirmed that under 10B parameters, domain specialization involves sharp trade-offs between clinical vocabulary accuracy and instructional flexibility.

A detailed synthesis comparing our experimental scope against these published studies is provided in Table 1 (refer to [`docs/tables/table_literature_comparison.md`](../tables/table_literature_comparison.md)).

## 2.3 Hallucination, Unanswerability, and Conformal Abstention
In high-stakes clinical settings, producing a confident incorrect answer is far more dangerous than abstaining. While generic models frequently hallucinate plausible-sounding pharmacological agents, domain-tuned models may overconfidently commit to erroneous diagnoses.

Traditional abstention mechanisms rely on raw softmax probabilities or verbalized confidence scores, both of which are notoriously miscalibrated in small language models. Conformal prediction offers a distribution-free statistical framework to guarantee finite-sample error bounds. By calibrating an abstention non-conformity score on a held-out validation set, one can bound the False Abstention Rate (FAR) below a pre-specified clinical threshold $\alpha$ (e.g., $\le 5\%$).

Prior evaluations of unanswerable medical questions relied heavily on simplistic answer-removal heuristics. In this work, we build upon the six-category taxonomy of `unanswerable_v2`—spanning fabricated pharmaceuticals, fictitious diseases, false diagnostic premises, missing critical patient variables, out-of-scope legal queries, and clinical ambiguity—to benchmark abstention reliability rigorously.
