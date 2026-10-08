# medrag-slm
Retrieval-Augmented Small Language Models for Reliable Medical Question Answering:
A Systematic Study of Accuracy, Hallucination, Abstention, and Efficiency Trade-offs

## What runs where
| Where | What | Phases |
|---|---|---|
| Laptop (Antigravity) | write code, tiny CPU check of every phase | all, tiny scale |
| Kaggle notebook **medrag-build** | data, corpus, indexes, retrieval, reranking | 1, 2, 3, 6, 7 |
| Kaggle notebook **medrag-slm-runner** | 5 models: format check, baseline, RAG | 4, 5, 8 |

## Design (frozen)
- Questions: MedQA 500 test / 200 val, PubMedQA 500 / 200, unanswerable 150 / 50 (MedQA with the correct option removed)
- Knowledge base: MedRAG Textbooks (18 USMLE textbooks, ~125k snippets)
- Retrieval: MedCPT dense + BM25, fused with RRF -> top 20 -> MedCPT cross-encoder -> top 5
- Models: Qwen3-1.7B, Gemma 3 4B, Phi-4-mini, Qwen3-4B, SmolLM3-3B; greedy, thinking off, same prompts
- All numbers live in `configs/base.yaml`

## Outputs (work/)
| Folder | Content |
|---|---|
| phase1 | test.jsonl, validation.jsonl |
| phase2 | corpus.jsonl |
| phase3 | chunks.jsonl, dense.faiss, bm25/ |
| phase4 | format check per model |
| phase5 | baseline answers per model |
| phase6 | retrieved.jsonl (top 20) |
| phase7 | evidence.jsonl (top 5 + scores) |
| phase8 | RAG answers per model, summary.json |
Every folder has a manifest.json (counts, settings, file hashes).

## Run order
1. Laptop: `pip install -r requirements.txt` then `./scripts/local_tiny.sh` -> must end with `ALL TINY PHASES PASSED`
2. `git add . && git commit -m "phases 1-8" && git push`
3. `kaggle kernels push -p kaggle_build` (first time: set GPU T4 x2 + Internet On in the notebook settings)
4. When medrag-build is complete: `kaggle kernels push -p kaggle_runner`
5. Download results: `kaggle kernels output satyasainath1311/medrag-slm-runner -p outputs/kaggle_qa`

## Run the demo
### 1. Local Offline Demo (Streamlit)
```bash
pip install -r requirements-app.txt && streamlit run app/demo.py
```

### 2. Live Interactive Chatbot (Kaggle GPU)
```bash
kaggle kernels push -p kaggle_chat
```
Open the kernel in Kaggle, click **Edit → Run All** (GPU enabled + Internet on, with `medrag-build` attached). Gradio generates a live public shareable link (`https://xxxx.gradio.live`). See [`kaggle_chat/README.md`](kaggle_chat/README.md) for details.

## If something fails
| Message | Fix |
|---|---|
| `[CHECK FAILED] ...` | read the message; it names exactly what is wrong |
| medrag-build output not found | in medrag-slm-runner: Add Input -> Your Work -> medrag-build |
| model `FAILED format check` | open work/phase4/<model>.json, look at `unparsed_examples` |
| Non-finite logits / garbage for Gemma | set Gemma `dtype: float32` in configs/base.yaml |
| A model crashed mid-run | set `MODELS = ["<name>"]` in kaggle_runner/runner.py and push again (finished questions are skipped) |

## Adaptive RAG Analysis
The script `src/analysis_adaptive_rag.py` evaluates three adaptive routing strategies that dynamically select between the model's parametric baseline and retrieved RAG predictions per question:
1. **Rerank Gate (`rerank_gate`)**: Use RAG if the top evidence reranker score $\ge \tau$, otherwise fall back to baseline.
2. **Confidence Gate (`confidence_gate`)**: Compare the model's top letter probability between baseline and RAG, choosing the prediction with higher confidence.
3. **Combined Gate (`combined`)**: Use Rerank Gate, but fall back to baseline if RAG confidence is lower than baseline confidence by more than $\delta$.

**Validation-Only Tuning Rule**:
All thresholds ($\tau$ over 101 validation rerank score quantiles, and $\delta \in \{0, 0.05, 0.1, 0.2\}$) are tuned strictly on the `validation` split to maximize validation accuracy. The selected parameters are then frozen and applied to the `test` split unchanged, preventing data leakage and ensuring test results remain unbiased.

## Unanswerable Question Set v2 (`data/unanswerable_v2/`)
A larger, challenging evaluation benchmark created by `src/build_unanswerable_v2.py` for evaluating medical SLM abstention reliability and distinguishing safe abstention from false abstention:
- **300 Unanswerable Questions** across 6 balanced categories (50 each):
  1. `fabricated_drug`: Plausible fake pharmaceuticals built from pharma-style syllables (`-vastin`, `-zolam`, `-ciclib`, etc.) in realistic dosing, adverse-effect, and interaction scenarios (verified 0 matches in textbook corpus).
  2. `fabricated_disease`: Plausible fake syndromes and fictitious pathologies in diagnostic/management dilemmas (verified 0 matches in textbook corpus).
  3. `false_premise`: Questions presupposing medical falsehoods (e.g., *"Why does paracetamol cure Plasmodium falciparum malaria?"*).
  4. `missing_info`: Clinical scenarios that cannot be answered without essential omitted parameters (weight, age, creatinine clearance, vitals, or clinical history).
  5. `out_of_scope`: Non-clinical requests (medical malpractice tort liability, health insurance billing/reimbursement, deterministic individual lifespan predictions, private clinician contact PII).
  6. `ambiguous`: Questions lacking clinical consensus or a single defensible option.
- **100 Matched Answerable Controls**: Near-miss paired questions sharing the exact template structures of the fabricated drug and disease categories, but utilizing real common entities (metformin, amoxicillin, malaria, anemia, etc.) with correct gold options to quantify false abstention rates.
- **Data Splits**:
  - `data/unanswerable_v2/val.jsonl`: 200 unanswerable + 60 controls (for validation threshold selection and abstention tuning).
  - `data/unanswerable_v2/test.jsonl`: 100 unanswerable + 40 controls (frozen test set).
  - `data/unanswerable_v2/test.sha256`: Cryptographic SHA-256 digest of `test.jsonl`.
- **Frozen-Test Rule**:
  The test partition (`data/unanswerable_v2/test.jsonl`) is cryptographically frozen (`test.sha256`) and must **NEVER** be examined, modified, or used for model prompt tuning, threshold calibration, or temperature fitting. All abstention parameters and operating thresholds must be tuned exclusively on `val.jsonl`.


