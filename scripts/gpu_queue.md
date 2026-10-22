# MedRAG-SLM GPU Execution Queue & Schedule Override Runbook

**Hard Deadline**: 2–3 weeks.  
**Optimization Target**: Matched generic-vs-medical comparison (Pairs A, B, C) under identical evaluation conditions.  
**GPU Budget Allocation**: 30.0 hours / week (Kaggle T4 dual-GPU environment).

---

## 1. Schedule Override Priority & 30-Hour Quota Tracking

| Priority Order | Run ID & Description | Model | Split / Conditions | $N$ (Questions) | Est. Wall Hours | Quota Cumulative | Run Status |
|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **(1)** | **Run 1: MedPsy-4B Unified Overnight Push (Stages A, B, C)** | `medpsy-4b` | Stage A: unans_v2 base (400)<br>Stage B: pubmedqa base (500)<br>Stage C: pubmedqa RAG (500) | 1,400 | **8.32 h** | 8.32 h / 30.0 h | **COMPLETED** |
| **(2)** | **Run 2: Unified Two-Stage Push (Stage A2 + Stage D)** | `medpsy-4b`<br>`qwen3-4b` | Stage A2: unans_v2 reasoning (400)<br>Stage D: unans_v2 base+RAG (800) | 1,200 | **1.97 h** | 10.29 h / 30.0 h | **READY TO RUN** |
| · *(2a)* | *Stage A2: MedPsy-4B unans_v2 reasoning baseline* | `medpsy-4b` | val (260) + test (140), reasoning prompt | 400 | *1.78 h* | *8.32 h + 1.78 h* | Stage 1/2 |
| · *(2b)* | *Stage D: Qwen3-4B unans_v2 baseline + RAG* | `qwen3-4b` | val (260) + test (140), base (400) + RAG (400) | 800 | *0.19 h* | *8.32 h + 1.97 h* | Stage 2/2 |
| **(3a)** | **Run 3: MedPsy-1.7B Validation Pilot** | `medpsy-1.7b` | val ($N=50$), C1–C5 | 250 | **1.39 h** | 11.68 h / 30.0 h | QUEUED (Gate) |
| **(4a)** | **Run 4: MedGemma-1.5-4B-it Pilot** | `medgemma-1.5-4b-it` | val ($N=50$), C1–C5 | 250 | **0.15 h** | 11.83 h / 30.0 h | QUEUED (Gate, Web UI) |
| **(4b)** | **Run 5: Gemma3-4B unanswerable_v2** | `gemma3-4b` | val (260) + test (140), base + RAG | 400 | **0.20 h** | 12.03 h / 30.0 h | QUEUED |
| **(4c)** | **Run 6: MedGemma-1.5-4B-it Full Test** | `medgemma-1.5-4b-it` | test ($N=500/140$), C1–C5 | 2,140 | **1.10 h** | 13.13 h / 30.0 h | QUEUED (Web UI) |
| **(3b)** | **Run 7: MedPsy-1.7B Full Test** | `medpsy-1.7b` | test ($N=500/140$), C1–C5 | 2,140 | **12.20 h** | 25.33 h / 30.0 h | QUEUED |
| **Buffer** | *Reserved Contingency / Re-runs* | — | — | — | **4.67 h** | 30.00 h / 30.0 h | RESERVED |
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

## 3. Run 2 Checklist: Unified Two-Stage Push (Stage A2 + Stage D)

### Overview
- **Objective**: Execute Stage A2 (MedPsy-4B unanswerable_v2 reasoning baseline rerun) and Stage D (Qwen3-4B unanswerable_v2 baseline + RAG) sequentially in a single kernel push:
  - **Stage A2**: `medpsy-4b` on `unanswerable_v2`, baseline (N=400: 260 val + 140 test) using the native **reasoning prompt** (`reasoning_baseline_prompt`, `standard_prompt=False`, `<think>...</think>`, ending with `Answer: <LETTER>`). Writes to `outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl`.
  - **Stage D**: `qwen3-4b` on `unanswerable_v2`, baseline AND full RAG (N=400 each, 800 total evaluations) using `UNANS_V2_SLM` mode (identical configuration and evidence to `qwen3-1.7b`). Writes to `outputs/kaggle_qa/unans_v2/qwen3-4b/baseline_qwen3-4b.jsonl` and `rag_qwen3-4b.jsonl`.
- **Execution Budget**:
  - Stage A2: 1.78 h (400 questions @ 32.0 s/q across 2 T4 shards)
  - Stage D: 0.19 h (baseline 92 s + RAG 588 s = 680 s ~ 11.3 min)
  - **Total**: **1.97 h** (Leaves ~19.71 h weekly quota).
- **Safety**: Incremental flush per question/shard, per-stage resumability, pre-flight frozen data SHA-256 check.

### Step-by-Step Execution Sequence

#### [ ] 1. Check Configuration in `kaggle_runner/runner.py`
Verify the flags:
```python
MEDPSY_UNANS_V2_REASONING = True
QWEN3_4B_UNANS_V2 = True
UNANS_MODEL = "qwen3-4b"

# All earlier and conflicting flags must be False:
MEDPSY_UNANS_V2_BASELINE = False
MEDPSY_PUBMEDQA_BASELINE = False
MEDPSY_PUBMEDQA_RAG = False
BUILD_UNANS_V2_EVIDENCE = False
UNANS_V2_SLM = False
V3_RUN = False
```

#### [ ] 2. Pre-Commit Verification & Git Push
Verify local files compile and push:
```bash
python3 -m py_compile kaggle_runner/runner.py scripts/verify_run.py

git add kaggle_runner/runner.py scripts/verify_run.py scripts/gpu_queue.md
git commit -m "Configure Run 2: Two-stage push (Stage A2 MedPsy-4B reasoning + Stage D Qwen3-4B base+RAG)"
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
*(Wait until status transitions from "running" to "complete" — approximately 1 hr 58 min).*

#### [ ] 5. Download Kernel Outputs & Audit Directory Structure
```bash
mkdir -p /tmp/run2_twostage_dl
kaggle kernels output satyasainath1311/medrag-slm-runner -p /tmp/run2_twostage_dl

# Audit downloaded structure:
ls -R /tmp/run2_twostage_dl/outputs/kaggle_qa/unans_v2/

# Copy canonical single-file outputs to repo destinations:
mkdir -p outputs/kaggle_qa/unans_v2/medpsy-4b
mkdir -p outputs/kaggle_qa/unans_v2/qwen3-4b

cp /tmp/run2_twostage_dl/outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl \
   outputs/kaggle_qa/unans_v2/medpsy-4b/
cp /tmp/run2_twostage_dl/outputs/kaggle_qa/unans_v2/qwen3-4b/baseline_qwen3-4b.jsonl \
   outputs/kaggle_qa/unans_v2/qwen3-4b/
cp /tmp/run2_twostage_dl/outputs/kaggle_qa/unans_v2/qwen3-4b/rag_qwen3-4b.jsonl \
   outputs/kaggle_qa/unans_v2/qwen3-4b/
```

#### [ ] 6. Run Stage Verification Suite
```bash
# Verify all outputs from Run 2 together:
python3 scripts/verify_run.py --run two_stage_push

# Or verify individually:
python3 scripts/verify_run.py --run medpsy_unans_v2_reasoning
python3 scripts/verify_run.py --run qwen3_4b_unans_v2_baseline
python3 scripts/verify_run.py --run qwen3_4b_unans_v2_rag
```

#### [ ] 7. Verification Gate (Pass/Fail Criteria)
**The gate is strictly INTEGRITY ONLY.** Truncation rate, parse rate, strict and fallback accuracy are reported in the output summary, not gated.

- [ ] **Stage A2 (`baseline_reasoning_medpsy-4b.jsonl`)**:
  - File exists: `outputs/kaggle_qa/unans_v2/medpsy-4b/baseline_reasoning_medpsy-4b.jsonl`
  - Committed standard-prompt baseline `baseline_medpsy-4b.jsonl` was NOT overwritten
  - Expected row count: Exactly 400 rows (260 validation, 140 test)
  - Duplicate IDs: 0
  - Missing IDs: 0 against reference `data/unanswerable_v2/{val,test}.jsonl`
  - Unexpected IDs: 0
  - Gold label / answer mismatches: 0
  - Input frozen hash verified: `data/unanswerable_v2/test.jsonl` and `val.jsonl` match canonical SHA-256
- [ ] **Stage D Baseline (`baseline_qwen3-4b.jsonl`)**:
  - File exists: `outputs/kaggle_qa/unans_v2/qwen3-4b/baseline_qwen3-4b.jsonl`
  - Expected row count: Exactly 400 rows (260 validation, 140 test)
  - Duplicate IDs: 0
  - Missing IDs: 0 against reference `data/unanswerable_v2/{val,test}.jsonl`
  - Unexpected IDs: 0
  - Gold label / answer mismatches: 0
- [ ] **Stage D RAG (`rag_qwen3-4b.jsonl`)**:
  - File exists: `outputs/kaggle_qa/unans_v2/qwen3-4b/rag_qwen3-4b.jsonl`
  - Expected row count: Exactly 400 rows (260 validation, 140 test)
  - Duplicate IDs: 0
  - Missing IDs: 0 against reference `data/unanswerable_v2/{val,test}.jsonl`
  - Unexpected IDs: 0
  - Gold label / answer mismatches: 0
- [ ] **Overall Pass**: `python3 scripts/verify_run.py --run two_stage_push` exits with code 0.
