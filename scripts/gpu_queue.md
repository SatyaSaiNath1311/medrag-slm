# MedRAG-SLM GPU Execution Queue & Schedule Override Runbook

**Hard Deadline**: 2–3 weeks.  
**Optimization Target**: Matched generic-vs-medical comparison (Pairs A, B, C) under identical evaluation conditions.  
**GPU Budget Allocation**: 30.0 hours / week (Kaggle T4 dual-GPU environment).

---

## 1. Schedule Override Priority & 30-Hour Quota Tracking

| Priority Order | Run ID & Description | Model | Split / Conditions | $N$ (Questions) | Est. Wall Hours | Quota Cumulative | Run Status |
|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **(1)** | **Run 1: MedPsy-4B Unified Overnight Push (Stages A, B, C)** | `medpsy-4b` | Stage A: unans_v2 base (400)<br>Stage B: pubmedqa base (500)<br>Stage C: pubmedqa RAG (500) | 1,400 | **8.32 h** | 8.32 h / 30.0 h | **COMPLETED** |
| **(2)** | **Run 2: Unified Two-Stage Push (Stage A2 + Stage D)** | `medpsy-4b`<br>`qwen3-4b` | Stage A2: unans_v2 reasoning (400)<br>Stage D: unans_v2 base+RAG (800) | 1,200 | **1.97 h** | 10.29 h / 30.0 h | **COMPLETED** |
| · *(2a)* | *Stage A2: MedPsy-4B unans_v2 reasoning baseline* | `medpsy-4b` | val (260) + test (140), reasoning prompt | 400 | *1.78 h* | *8.32 h + 1.78 h* | COMPLETED |
| · *(2b)* | *Stage D: Qwen3-4B unans_v2 baseline + RAG* | `qwen3-4b` | val (260) + test (140), base (400) + RAG (400) | 800 | *0.19 h* | *8.32 h + 1.97 h* | COMPLETED |
| **(3)** | **Run 3: Qwen3-4B-Thinking-2507 MedQA Control (C1 + C2)** | `qwen3-4b-thinking-2507` | Condition 1: MedQA test base (500)<br>Condition 2: MedQA test RAG (500) | 1,000 | **7.05 h** | 24.39 h / 30.0 h | **READY TO RUN** |
| · *(3a)* | *Condition 1: MedQA test baseline (reasoning prompt)* | `qwen3-4b-thinking-2507` | test (N=500), reasoning_baseline_prompt | 500 | *2.61 h (5.22 GPU-h)* | *10.29 h + 5.22 h* | Condition 1/2 |
| · *(3b)* | *Condition 2: MedQA test RAG (reasoning prompt)* | `qwen3-4b-thinking-2507` | test (N=500), reasoning_rag_prompt | 500 | *4.44 h (8.88 GPU-h)* | *10.29 h + 14.10 h* | Condition 2/2 |
| **Buffer** | *Reserved Contingency / Re-runs* | — | — | — | **5.61 h** | 30.00 h / 30.0 h | RESERVED |
| *(4)* | *Deferred to Next Quota: Qwen3-4B-Thinking PubMedQA & unans_v2* | `qwen3-4b-thinking-2507` | PubMedQA (C3/C4) + unanswerable_v2 (C5) | 1,140 | ~4.8 h | Next Cycle | Deferred |
| *(5)* | *MedPsy-1.7B / MedGemma-1.5-4B-it Matched Pairs* | Various | Validation pilots & full evaluations | — | — | Cycle 2 | Cycle 2 Window |

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

---

## 4. Run 3 Checklist: Qwen3-4B-Thinking-2507 MedQA Control (Pair B Thinking Control)

### Overview
- **Objective**: Execute Condition 1 (MedQA test baseline) and Condition 2 (MedQA test full RAG) sequentially for `qwen3-4b-thinking-2507` under identical prompt, decoding, and token budget as `medpsy-4b`.
  - **Condition 1**: MedQA test ($N=500$) baseline using native reasoning prompt (`reasoning_baseline_prompt`, 1024-token thinking cap, greedy decoding). Writes to `outputs/v3/qwen3-4b-thinking-2507/C1/qwen3-4b-thinking-2507.jsonl`.
  - **Condition 2**: MedQA test ($N=500$) full RAG using native reasoning prompt (`reasoning_rag_prompt`, top-5 MedCPT evidence from `phase7/evidence.jsonl`, 1024-token thinking cap, greedy decoding). Writes to `outputs/v3/qwen3-4b-thinking-2507/C2/qwen3-4b-thinking-2507.jsonl`.
- **Compute Budget**:
  - Condition 1: 5.22 GPU-hours (500 questions @ 37.58 s/q across 2 T4 shards $\approx$ 2.61 wall hours).
  - Condition 2: 8.88 GPU-hours (500 questions @ 63.93 s/q across 2 T4 shards $\approx$ 4.44 wall hours).
  - **Total**: **14.10 GPU-hours** ($\approx$ **7.05 wall hours**). Leaves **5.61 h** remaining weekly quota buffer.
- **Safety**: Incremental flush per shard/question, per-condition resumability, pre-flight frozen data SHA-256 check against `audit/frozen_hashes.json`.

### Step-by-Step Execution Sequence

#### [ ] 1. Check Configuration in `kaggle_runner/runner.py`
Verify the flags:
```python
QWEN3_4B_THINKING_MEDQA = True

# All earlier and conflicting flags must be False:
MEDPSY_UNANS_V2_REASONING = False
QWEN3_4B_UNANS_V2 = False
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
python3 -m py_compile kaggle_runner/runner.py scripts/verify_run.py scripts/budget.py

git add kaggle_runner/runner.py scripts/verify_run.py scripts/budget.py scripts/gpu_queue.md
git commit -m "Configure Run 3: Qwen3-4B-Thinking-2507 MedQA baseline + RAG (Pair B thinking control)"
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
*(Wait until status transitions from "running" to "complete" — approximately 7 hr 05 min wall clock).*

#### [ ] 5. Download Kernel Outputs & Audit Directory Structure
```bash
mkdir -p /tmp/run3_qwen3_thinking_dl
kaggle kernels output satyasainath1311/medrag-slm-runner -p /tmp/run3_qwen3_thinking_dl

# Copy canonical single-file outputs to repo destinations:
mkdir -p outputs/v3/qwen3-4b-thinking-2507/C1
mkdir -p outputs/v3/qwen3-4b-thinking-2507/C2

cp /tmp/run3_qwen3_thinking_dl/outputs/v3/qwen3-4b-thinking-2507/C1/qwen3-4b-thinking-2507.jsonl \
   outputs/v3/qwen3-4b-thinking-2507/C1/
cp /tmp/run3_qwen3_thinking_dl/outputs/v3/qwen3-4b-thinking-2507/C2/qwen3-4b-thinking-2507.jsonl \
   outputs/v3/qwen3-4b-thinking-2507/C2/
```

#### [ ] 6. Run Verification Suite
```bash
# Verify both conditions combined:
python3 scripts/verify_run.py --run qwen3_4b_thinking_medqa

# Or verify individually:
python3 scripts/verify_run.py --run qwen3_4b_thinking_medqa_baseline
python3 scripts/verify_run.py --run qwen3_4b_thinking_medqa_rag
```

#### [ ] 7. Verification Gate (Pass/Fail Criteria)
**The gate is strictly INTEGRITY ONLY.** Truncation rate, parse rate, strict and fallback accuracy are reported in the output summary, not gated.

- [ ] **Condition 1 (`C1/qwen3-4b-thinking-2507.jsonl`)**:
  - File exists: `outputs/v3/qwen3-4b-thinking-2507/C1/qwen3-4b-thinking-2507.jsonl`
  - Expected row count: Exactly 500 rows
  - Duplicate IDs: 0
  - Missing IDs: 0 against reference `outputs/kaggle_build/work/phase1/test.jsonl` (MedQA)
  - Unexpected IDs: 0
  - Gold answer mismatches: 0
  - Input frozen hash verified: `outputs/kaggle_build/work/phase1/test.jsonl` matches SHA-256 `fe7fe3...`
- [ ] **Condition 2 (`C2/qwen3-4b-thinking-2507.jsonl`)**:
  - File exists: `outputs/v3/qwen3-4b-thinking-2507/C2/qwen3-4b-thinking-2507.jsonl`
  - Expected row count: Exactly 500 rows
  - Duplicate IDs: 0
  - Missing IDs: 0 against reference `outputs/kaggle_build/work/phase1/test.jsonl` (MedQA)
  - Unexpected IDs: 0
  - Gold answer mismatches: 0
- [ ] **Overall Pass**: `python3 scripts/verify_run.py --run qwen3_4b_thinking_medqa` exits with code 0.

