# 6. Limitations and Ethical Considerations

## 6.1 Limitations

### Multiple-Choice Examination Scope
Our primary quantitative evaluations are grounded in standardized examination benchmarks: MedQA (USMLE 4-option multiple-choice format) and PubMedQA (ternary biomedical research questions). While standardized exams test diagnostic reasoning and biomedical knowledge, they do not fully capture the complexity of real-world clinical practice, which involves unstructured clinical notes, ambiguous patient narratives, non-standard abbreviations, and multimodal telemetry data.

### Retrieval Pipeline Architecture
All evaluations utilize a fixed, hybrid retrieval pipeline combining MedCPT dense retrieval, BM25 sparse retrieval, reciprocal rank fusion, and cross-encoder re-ranking over the MedRAG USMLE Textbooks corpus. While this represents a high-performing medical RAG architecture, we did not explore agentic multi-hop retrieval, dynamic query rewriting, or web-scale external search engines (e.g., live PubMed API querying), which could alter the relative benefit of external retrieval.

### Hardware and Quantization Constraints
All models were evaluated in full 16-bit or 32-bit floating-point precision on cloud NVIDIA T4 GPUs. In practical edge deployments (e.g., clinic workstations, bedside tablets, or mobile nursing carts), models are typically quantized to 4-bit or 8-bit integer formats (e.g., via AWQ, GPTQ, or GGUF). Quantization affects small models non-uniformly and may differentially impact reasoning token generation, chain-of-thought coherence, and logits calibration.

### Fixed Reasoning Token Budget
To ensure rigorous comparability across models, thinking models were evaluated under an invariant 1,024 reasoning token cap. Although our validation sensitivity analysis on 100 MedQA questions demonstrated negligible accuracy differences between 1,024 and 2,048 tokens, highly complex, multi-system clinical cases might benefit from longer reasoning trajectories.

### Synthetic vs. Ecological Unanswerability
The `unanswerable_v2` benchmark provides structured categorization of clinical failure modes (fabricated drugs, fictitious pathologies, false premises, missing parameters). However, real-world unanswerability in electronic health records (EHR) often arises from conflicting lab results, temporal ambiguity, unrecorded patient history, or fragmented clinical handoffs, which present distinct cognitive challenges.

---

## 6.2 Ethical and Clinical Safety Declarations

### Research Prototype Disclaimer
The models, pipelines, and frameworks presented in this research are academic prototypes designed solely for comparative research and computational benchmarking. **None of the models or systems evaluated in this work are certified medical devices**, and they must not be used as autonomous diagnostic or treatment planning systems.

### Clinician-in-the-Loop Imperative
Under no circumstances should clinical decisions—including pharmacotherapy selection, differential diagnosis, or triage categorization—be executed based solely on model-generated outputs. In any potential deployment scenario, model outputs and retrieved evidence must be subject to review by licensed healthcare professionals.

### Data Privacy and Security
All evaluations in this study were conducted on publicly available, de-identified research datasets (USMLE-style vignettes, open PubMed abstracts, and synthetic test suites). No patient-identifiable private health information (PHI) was processed or exposed during this investigation.
