# MedRAG-SLM: Matched Generic vs Medical-Domain Comparison (Design v3, pre-registered)

**Repo location:** `docs/DESIGN_GENERIC_VS_MEDICAL.md` (commit to GitHub main so every session and Antigravity can read it)
**Status:** Plan written 2026-10-16, before any new GPU run. Nothing in this file has been run yet.
**Title (unchanged):** Retrieval-Augmented Small Language Models for Reliable Medical Question Answering: A Systematic Study of Accuracy, Hallucination, and Efficiency Trade-offs
**Protocol rules (same as `NEXT_EXPERIMENTS.md`):**
1. Prompts, thresholds and settings are tuned on the **validation split only**.
2. Frozen test splits are evaluated **once** per model and condition.
3. Success criteria below are fixed before running.
4. New results are added as **v3**; existing frozen results are never overwritten.

---

## 1. Why this redesign (guide's feedback)

Current study: 5 generic small models (weak results), then MedPsy-4B added at the end. The panel's concern is fair: it looks like a domain model was brought in only to win, and the comparison mixes three effects (domain training, thinking mode, model size).

Fix: make **generic vs medical-domain** a designed factor, with **matched pairs** that share a base model or family, run under identical conditions, and report trade-offs honestly (including where generic models win).

### Pattern in published work (checked via search, 2026-10-16)
| Paper | What they do | Lesson for us |
|---|---|---|
| MedRGB (arXiv 2411.09213) | General LLMs and medical fine-tuned models (MEDITRON-70b, PMC-Llama-13b) side by side, no-retrieval vs RAG with 5 and 20 documents | Same settings for both groups; domain models gave mixed results, not automatic wins |
| Rare-case clinical retrieval benchmark (arXiv 2605.21807) | Sparse, general-purpose and medical fine-tuned models across several retrievers | Report per-group results; general models can beat medical ones |
| Benchmarking SLMs for medical QA in resource-constrained settings (EIS 2026 proceedings) | Domain-specific vs general-purpose models under 10B parameters | Frame the result as a trade-off (comprehensiveness vs hallucination vs coverage), not a winner |
| MedRAG / MIRAGE (Xiong et al., ACL Findings 2024) | Fixed benchmark, many LLM x retriever x corpus combinations | Factorial grid, same questions, same prompts |
| Local medical RAG study (arXiv 2506.20009) | llama3, mistral and medgemma-4b-it with one RAG pipeline | Medical model enters as one arm of a fixed pipeline |

Open each paper and read the methods section before citing. The table above is from abstracts and summaries only.

---

## 2. Research questions and hypotheses

| ID | Question | Hypothesis (direction fixed in advance) |
|---|---|---|
| RQ1 | Does medical-domain training improve closed-book accuracy at matched size/family? | Yes for all 3 pairs |
| RQ2 | Does RAG help generic and medical models equally? | RAG gain is **smaller** for medical models (they already hold the knowledge) |
| RQ3 | Do medical models hallucinate less and abstain better on unanswerable_v2? | Fewer fabricated answers, but abstention stays moderate; no model is "perfect" |
| RQ4 | What does domain training cost in latency, memory and truncation? | Reasoning-style medical models are much slower; report accuracy per second |

Note: results can contradict a hypothesis. They are reported either way.

---

## 3. Models

### 3.1 Matched pairs (primary analysis)
| Pair | Generic arm | Medical arm | Shared base |
|---|---|---|---|
| P1 (1.7B) | Qwen3-1.7B (done, thinking off) | MedPsy-1.7B | Qwen3-1.7B |
| P2 (4B, Qwen) | Qwen3-4B (done, thinking off) | MedPsy-4B (done) | Qwen3-4B-Thinking-2507 |
| P3 (4B, Gemma) | Gemma3-4B (done) | MedGemma-1.5-4B-it | Gemma 3 4B |

Base-model facts for MedPsy come from the model cards (MedPsy-4B is post-trained from Qwen3-4B-Thinking-2507; MedPsy-1.7B from Qwen3-1.7B in thinking mode). **Confirm on the Hugging Face cards before the run:** exact repo IDs, the MedGemma-1.5-4B base model, licences and gating (MedGemma and Gemma are gated, so use the Kaggle web UI route with the HF_TOKEN secret, as for gemma3-4b).

### 3.2 Reasoning-mode controls (to remove the thinking confound)
| Control | Why |
|---|---|
| Qwen3-4B-Thinking-2507 | Same base and thinking style as MedPsy-4B; isolates domain post-training |
| Qwen3-1.7B with thinking ON | Same as MedPsy-1.7B's base mode |

Both controls use the **same thinking-token cap as MedPsy** (1,024 tokens, as in the existing MedPsy runs).

### 3.3 Extra generic references (already run, kept for context)
Phi-4-mini, SmolLM3-3B.

### 3.4 Final model set: 8 core models + 2 references
Generic: Qwen3-1.7B, Qwen3-1.7B-thinking, Qwen3-4B, Qwen3-4B-Thinking-2507, Gemma3-4B (+ Phi-4-mini, SmolLM3-3B).
Medical: MedPsy-1.7B, MedPsy-4B, MedGemma-1.5-4B-it.

---

## 4. Conditions (identical for every model)

| Code | Condition | Data |
|---|---|---|
| C1 | Closed-book baseline | MedQA test (500), PubMedQA test (500, no context) |
| C2 | RAG (v1 pipeline: MedCPT + BM25 + RRF -> top 20 -> MedCPT cross-encoder -> top 5) | MedQA test, PubMedQA test |
| C3 | PubMedQA + abstract (oracle context) | PubMedQA test |
| C4 | unanswerable_v2, baseline | 140 test questions (100 unanswerable + 40 controls), 260 val |
| C5 | unanswerable_v2, RAG (uses `medrag-unans-v2-evidence`) | same |

**Fairness rules**
- Same question files, same top-5 evidence, same answer format and parsing for all models.
- Chat template and generation settings follow each model card, but are **recorded in the manifest**. Greedy decoding for all.
- Thinking models: same cap (1,024 tokens). Report truncation rate. A truncated answer counts as **wrong** for accuracy (primary) and is also reported as "answered-only accuracy" (secondary).
- Sensitivity check on validation only: cap 2,048 tokens on 100 MedQA questions for the 4 thinking models, to show how much truncation affects the ranking.
- Prompts must not be tuned for one group. If a prompt change is needed for format reasons, it is made for all and the generic arm is re-run on validation.

---

## 5. Metrics and statistics

**Accuracy:** per dataset and overall, 95% bootstrap CI (1,000 resamples, seed 42), as in `final_results.md`.
**Within-pair test:** McNemar exact test, generic vs medical, per pair and condition. Holm correction across the 3 pairs.
**RAG effect (RQ2):** per model, McNemar baseline vs RAG; pooled CMH separately for the generic group and the medical group; then compare the two groups' odds ratios (Breslow-Day or logistic regression with `model_type x RAG` interaction).
**Hallucination (RQ3):** on unanswerable_v2 test, rate of confident wrong answers on unanswerable questions (answered instead of abstaining), by category; control questions give the false-abstention rate.
**Abstention:** recall on unanswerable at fixed 5% false-abstention rate (the conformal procedure already in `reliability_score.py`, fitted on validation, applied once to test). Models without self-consistency data use the single-run signals available; state this limit.
**Citation support:** the frozen NLI metric from `nli_all_models_test.py`, reported as an upper bound (dev precision 52%, NPV 96%).
**Efficiency (RQ4):** sec/question, thinking tokens per answer, truncation %, peak VRAM/RAM, accuracy per second.
**Reporting rule:** one table per RQ; no claim without a CI or p-value; no "state of the art" language.

---

## 6. Pre-registered success criteria

| ID | Criterion | If not met |
|---|---|---|
| S1 | Valid-answer parse rate >= 95% for every model and condition (thinking models: >= 80% before truncation counting) | Fix parsing on validation, re-run that model only |
| S2 | RQ1 is "supported" for a pair only if the generic-vs-medical McNemar p < 0.05 after Holm correction **and** the direction matches | Report as not supported |
| S3 | Controls: the medical model's gain over its thinking-mode generic control is reported separately from its gain over the non-thinking generic arm | Always reported |
| S4 | Data leakage audit re-run on every new run set (`audit/`): 0 test-question leakage | Stop and investigate |

---

## 7. Execution plan (GPU needed after weekly quota reset)

### Gate 0: before anything
- [ ] Check remaining Kaggle GPU hours (weekly quota 30 h; 29:59 used as of 2026-10-15).
- [ ] Commit this file to `docs/`.
- [ ] Confirm model IDs, base models and gating on the Hugging Face cards.
- [ ] Share this design with the guide and note any change she asks for **before** running.

### Step A (non-GPU, can start now)
- [ ] Add the new models to `configs/base.yaml` and `kaggle_runner/runner.py` (one entry per model; thinking cap and chat template recorded).
- [ ] Write `src/analysis_pairs.py`: matched-pair McNemar + Holm, RAG-effect CMH by group, interaction test, per-category unanswerable table.
- [ ] Write `src/make_pair_figures.py`: paired bar chart (generic vs medical per pair, baseline and RAG), RAG-gain by group, accuracy vs latency scatter.
- [ ] Tiny CPU smoke test on 8 questions for each new model entry (parsing only).

### Step B (GPU, follow the existing one-model-per-push workflow)
Use the pilot gate you already use: **validation N=50 pilot first, then full.**
1. Finish pending v2 runs: smollm3-3b, phi4-mini, qwen3-4b, gemma3-4b (web UI) on unanswerable_v2; MedPsy-4B on unanswerable_v2 with the Phase 5 prompt; MedPsy-4B PubMedQA closed-book and RAG.
2. New models, in this order (cheapest information first): MedPsy-1.7B, Qwen3-1.7B-thinking, Qwen3-4B-Thinking-2507, MedGemma-1.5-4B-it.
3. For each new model: C1 + C2 + C3 + C4 + C5, downloaded to `outputs/kaggle_qa/full/<model>` before the next push.

**Budget method.** Hours = questions x sec/q / 3600 (divide by parallel GPUs if sharded). Reference from your logs: MedPsy-4B about 37 s/q (baseline) and 30 s/q (PubMedQA + abstract) on the existing setup, so about 1,000 answerable questions is roughly 10 h for a 4B thinking model. The pilot's sec/q for each new model replaces this estimate. Plan across **at least two weekly quotas** and do not start a model that cannot finish inside the remaining hours.

### Step C (after runs)
- [ ] Run `analysis_pairs.py` once on frozen test outputs (no retuning).
- [ ] NLI citation metric on the new models.
- [ ] Leakage audit.
- [ ] Update `final_results.md` and figures; keep v1/v2 tables untouched.

---

## 8. Paper structure (target: IEEE Access)

1. Introduction: health-worker setting, why small models, the generic-vs-medical question.
2. Related work: the papers in section 1 plus MedRAG, MIRAGE-style benchmarking, abstention.
3. Method: data, corpus, retrieval, **matched-pair design**, controls, metrics.
4. Results by RQ (one table each) and a combined efficiency view.
5. Discussion: when does domain training help; RAG's role when Recall@5 is about 30%; abstention limits.
6. Limitations (below). 7. Conclusion.

## 9. Threats to validity (state these in the paper)
- Retrieval Recall@5 about 30%: RAG gains are limited by retrieval, not only by the model.
- Reasoning models are capped at 1,024 thinking tokens; truncation affects accuracy and abstention.
- Medical models may have seen benchmark-style data in training; the leakage audit covers our corpus and splits, not their training data.
- NLI citation metric is an upper bound (validated against an LLM judge, not a human reference).
- Single seed, greedy decoding; paraphrase robustness not yet tested (Experiment 2 in `NEXT_EXPERIMENTS.md` covers it).
- Published numbers are not directly comparable (see `table_literature_comparison.md`).

## 10. Current state snapshot (from the project log, 2026-10-15)
- Done: phases 0-10 for 5 generic models; MedPsy-4B baseline, RAG, self-consistency (val and frozen test); reliability score applied once to test; NLI citation metric on all 6 models; leakage audit passed; v2 evidence pre-pass; qwen3-1.7b on unanswerable_v2.
- Pending: the v2 runs listed in Step B.1, then the new models in Step B.2.
- GPU quota exhausted until the weekly reset.

## 11. Resume checklist for the next session
1. Read this file and `docs/STATUS_REPORT.md`.
2. Check GPU hours; pick the next unchecked item in section 7.
3. Run the validation pilot for that model; only then the full run.
4. Download, verify ids and counts, commit, then the next model.