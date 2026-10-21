# MedRAG-SLM GPU Execution Queue & Schedule Override Runbook

**Hard Deadline**: 2–3 weeks.  
**Optimization Target**: Matched generic-vs-medical comparison (Pairs A, B, C) under identical evaluation conditions.  
**GPU Budget Allocation**: 30.0 hours / week (Kaggle T4 dual-GPU environment).

---

## 1. Schedule Override Priority & 30-Hour Quota Tracking

| Priority Order | Run ID & Description | Model | Split / Conditions | $N$ (Questions) | Est. Wall Hours | Quota Cumulative | Run Status |
|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **(1)** | **Run 1: MedPsy-4B Unified Overnight Push (Stages A, B, C)** | `medpsy-4b` | Stage A: unans_v2 (400)<br>Stage B: pubmedqa_base (500)<br>Stage C: pubmedqa_rag (500) | 1,400 | **8.32 h** | 8.32 h / 30.0 h | **READY TO RUN** |
| · *(1a)* | *Stage A: unanswerable_v2 baseline* | `medpsy-4b` | val (260) + test (140), baseline | 400 | *1.78 h* | *1.78 h* | Stage 1/3 |
| · *(1b)* | *Stage B: PubMedQA closed-book baseline* | `medpsy-4b` | test (500), closed-book baseline | 500 | *2.10 h* | *3.88 h* | Stage 2/3 |
| · *(1c)* | *Stage C: PubMedQA full RAG* | `medpsy-4b` | test (500), full RAG (top-5 MedCPT) | 500 | *4.44 h* | *8.32 h* | Stage 3/3 |
| **(2)** | **Run 2: Qwen3-4B unanswerable_v2** | `qwen3-4b` | val (260) + test (140), base + RAG | 400 | **0.19 h** | 8.51 h / 30.0 h | QUEUED |
| **(3a)** | **Run 3: MedPsy-1.7B Validation Pilot** | `medpsy-1.7b` | val ($N=50$), C1–C5 | 250 | **1.39 h** | 9.90 h / 30.0 h | QUEUED (Gate) |
| **(4a)** | **Run 4: MedGemma-1.5-4B-it Pilot** | `medgemma-1.5-4b-it` | val ($N=50$), C1–C5 | 250 | **0.15 h** | 10.05 h / 30.0 h | QUEUED (Gate, Web UI) |
| **(4b)** | **Run 5: Gemma3-4B unanswerable_v2** | `gemma3-4b` | val (260) + test (140), base + RAG | 400 | **0.20 h** | 10.25 h / 30.0 h | QUEUED |
| **(4c)** | **Run 6: MedGemma-1.5-4B-it Full Test** | `medgemma-1.5-4b-it` | test ($N=500/140$), C1–C5 | 2,140 | **1.10 h** | 11.35 h / 30.0 h | QUEUED (Web UI) |
| **(3b)** | **Run 7: MedPsy-1.7B Full Test** | `medpsy-1.7b` | test ($N=500/140$), C1–C5 | 2,140 | **12.20 h** | 23.55 h / 30.0 h | QUEUED |
| **Buffer** | *Reserved Contingency / Re-runs* | — | — | — | **6.45 h** | 30.00 h / 30.0 h | RESERVED |
| *(5)* | *Run 8: Qwen3-4B-Thinking-2507* | `qwen3-4b-thinking-2507` | val pilot (2.7h) / full (23.8h) | 2,140 | ~26.5 h | Cycle 2 | Cycle 2 Window |
| *(6)* | *Optional: Phi-4-mini, SmolLM3-3B, Qwen3-1.7B thinking* | Various | unanswerable_v2 / thinking | — | — | Extra hours | If hours remain |

---

## 2. Hard Execution Rules

1. **Sequential Progression**: Only execute ONE run at a time. Finish the run, download outputs, verify integrity, review results, and confirm before proceeding.
2. **Code Synchronization**: Kaggle clones the repo from GitHub `main`. **Always commit and push to git BEFORE pushing kernel to Kaggle.**
3. **Dual-GPU Sharding & Resumability**: MedPsy runs utilize 2x T4 GPUs via `NUM_SHARDS=2` and `CUDA_VISIBLE_DEVICES` parallelism in `runner.py`. Output is written incrementally per shard (`append_jsonl` + flush) and is fully resumable across stages. Finished outputs are never overwritten or deleted by later stages.
4. **Primary Accuracy Metric**: Strict accuracy (`(not truncated) and parsed and correct`) is the primary metric. With-fallback is secondary and always labelled.
5. **Read-Only Test Firewall & Frozen Hash Gate**: Pre-flight SHA-256 verification of benchmark inputs occurs before any GPU inference begins. Any hash mismatch immediately aborts the kernel. Ground truth test splits and unanswerable v2 test sets are read-only; no tuning on test.

---

## 3. Run 1 Checklist: MedPsy-4B Unified Overnight Push (Stages A, B, C)

### Overview
- **Objective**: Execute all three critical MedPsy-4B benchmark stages in a single sequential overnight run:
  - **Stage A**: `unanswerable_v2` baseline (Phase 5 standard prompt via `src.llm.baseline_prompt`, 400 questions: 260 val + 140 test).
  - **Stage B**: `pubmedqa` closed-book baseline (test split, 500 questions).
  - **Stage C**: `pubmedqa` full RAG (test split, top-5 MedCPT evidence, 500 questions).
- **Model**: `medpsy-4b` (`qvac/MedPsy-4B`, float16, dual-GPU 2-shard execution).
- **Execution Budget**:
  - Stage A: 1.78 h
  - Stage B: 2.10 h
  - Stage C: 4.44 h
  - **Total**: 8.32 h (Fits within 10.0 h Kaggle kernel limit and leaves 21.68 h weekly quota).
- **Safety**: Incremental per-shard flush, per-stage idempotency/resumability, pre-flight frozen data SHA-256 check.

### Step-by-Step Execution Sequence

#### [ ] 1. Check Configuration in `kaggle_runner/runner.py`
Verify the flags:
```python
MODELS = ["medpsy-4b"]
TINY = False
CHECK_ONLY = False
MEDPSY_UNANS_V2_BASELINE = True
MEDPSY_PUBMEDQA_BASELINE = True
MEDPSY_PUBMEDQA_RAG = True
```
All other modes (`BUILD_UNANS_V2_EVIDENCE`, `UNANS_V2_SLM`, `V3_RUN`) must be `False`.

#### [ ] 2. Pre-Commit Verification & Git Push
Verify local frozen hashes and test suites before pushing:
```bash
shasum -a 256 data/unanswerable_v2/val.jsonl > data/unanswerable_v2/val.sha256
python3 audit/make_frozen_hashes.py
python3 -m unittest tests/test_unanswerable_v2.py
python3 audit/final_leakage_audit.py

git add kaggle_runner/runner.py src/ scripts/ audit/ data/unanswerable_v2/val.sha256
git commit -m "Configure Run 1: MedPsy-4B 3-stage unified overnight push (Stages A, B, C)"
git push origin main
```

#### [ ] 3. Push Kernel to Kaggle
```bash
cd kaggle_runner && kaggle kernels push && cd ..
```

#### [ ] 4. Monitor Kernel Status
```bash
kaggle kernels status satyasainath1311/medrag-slm-runner
```
*(Wait until status transitions from "running" to "complete" — approximately 8 hr 20 min).*

#### [ ] 5. Download Kernel Outputs & Audit Directory Structure
```bash
mkdir -p /tmp/medpsy_overnight_dl
kaggle kernels output satyasainath1311/medrag-slm-runner -p /tmp/medpsy_overnight_dl

# Audit downloaded structure:
ls -R /tmp/medpsy_overnight_dl/outputs/kaggle_qa/

# Copy canonical single-file outputs to repo destinations:
mkdir -p outputs/kaggle_qa/unans_v2/medpsy-4b
mkdir -p outputs/kaggle_qa/medpsy_pubmedqa_baseline
mkdir -p outputs/kaggle_qa/medpsy_pubmedqa_rag

cp /tmp/medpsy_overnight_dl/outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_medpsy-4b.jsonl \
   outputs/kaggle_qa/unans_v2/medpsy-4b/
cp /tmp/medpsy_overnight_dl/outputs/kaggle_qa/medpsy_pubmedqa_baseline/medpsy-4b_pubmedqa_baseline.jsonl \
   outputs/kaggle_qa/medpsy_pubmedqa_baseline/
cp /tmp/medpsy_overnight_dl/outputs/kaggle_qa/medpsy_pubmedqa_rag/medpsy-4b_pubmedqa_rag.jsonl \
   outputs/kaggle_qa/medpsy_pubmedqa_rag/
```

#### [ ] 6. Run Stage Verification Suite
```bash
# Verify individual stages:
python3 scripts/verify_run.py --run medpsy_unans_v2_baseline
python3 scripts/verify_run.py --run medpsy_pubmedqa_baseline
python3 scripts/verify_run.py --run medpsy_pubmedqa_rag

# Or run combined suite:
python3 scripts/verify_run.py --run all_stages
```

#### [ ] 7. Verification Gate (Pass/Fail Criteria)
**The gate is strictly INTEGRITY ONLY.** Truncation rate, parse rate, strict and fallback accuracy are reported in the output summary, not gated. The parse rate criterion (previously $\ge 90\%$) is completely removed.

- [ ] **Stage A (`baseline_medpsy-4b.jsonl`)**:
  - File exists: `outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_medpsy-4b.jsonl`
  - Expected row count: Exactly 400 rows (260 validation, 140 test)
  - Duplicate IDs: 0
  - Missing IDs: 0 against reference `data/unanswerable_v2/{val,test}.jsonl`
  - Unexpected IDs: 0
  - Gold label / answer mismatches: 0
  - Input frozen hash verified: `data/unanswerable_v2/test.jsonl` and `val.jsonl` match canonical SHA-256
- [ ] **Stage B (`medpsy-4b_pubmedqa_baseline.jsonl`)**:
  - File exists: `outputs/kaggle_qa/medpsy_pubmedqa_baseline/medpsy-4b_pubmedqa_baseline.jsonl`
  - Expected row count: Exactly 500 rows
  - Duplicate IDs: 0
  - Missing IDs: 0 against `outputs/kaggle_build/work/phase1/test.jsonl`
  - Unexpected IDs: 0
  - Gold label / answer mismatches: 0
  - Input frozen hash verified: PubMedQA test matches `fe7fe3...`
- [ ] **Stage C (`medpsy-4b_pubmedqa_rag.jsonl`)**:
  - File exists: `outputs/kaggle_qa/medpsy_pubmedqa_rag/medpsy-4b_pubmedqa_rag.jsonl`
  - Expected row count: Exactly 500 rows
  - Duplicate IDs: 0
  - Missing IDs: 0 against `outputs/kaggle_build/work/phase1/test.jsonl`
  - Unexpected IDs: 0
  - Gold label / answer mismatches: 0
  - Input frozen hash verified: PubMedQA test matches `fe7fe3...`
- [ ] **Overall Pass**: `python3 scripts/verify_run.py --run all_stages` exits with code 0.

---

## 4. Next Runs in Schedule Override (Cycle 1 Remaining Quota: 21.68 h)

- **Run 2**: `qwen3-4b` unanswerable_v2 (400 questions, ~0.19 h)
- **Run 3**: `medpsy-1.7b` Validation Pilot ($N=50$, C1–C5, ~1.39 h)
- **Run 4**: `medgemma-1.5-4b-it` Validation Pilot ($N=50$, C1–C5, ~0.15 h) + Gemma3-4B v2 (~0.20 h)
- **Run 6**: `medgemma-1.5-4b-it` Full Test (2,140 questions, ~1.10 h)
- **Run 7**: `medpsy-1.7b` Full Test (2,140 questions, ~12.20 h)
