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

