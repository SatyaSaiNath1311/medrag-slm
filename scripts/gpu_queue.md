# MedRAG-SLM GPU Execution Queue & Runbook

This document specifies the exact, prioritized copy-paste commands to execute once the weekly Kaggle GPU quota resets (30 hours).

> [!IMPORTANT]
> **Safety Rules:**
> - Never run full test sets without running the validation pilot ($N=50$) first.
> - Verify that output files have 100% parse rate and 0 errors before initiating full test runs.
> - Full test runs require setting `export V3_CONFIRM_TEST=1` to prevent accidental execution.
> - Resume skipping is enabled by default: interrupted runs will safely pick up where they left off.

---

## 1. Execution Priority & Quota Allocation

Based on `python3 scripts/budget.py`:
- **Phase A (Finish Pending v2)**: ~1.5 hours (or ~14.4h if completing MedPsy PubMedQA RAG).
- **Phase B (v3 Validation Pilots)**: 5.58 hours across all 4 models.
- **Phase C (v3 Test Execution)**: Prioritized to guarantee core RQ1/RQ2 matched pairs.
  1. `medgemma-1.5-4b-it` (Non-thinking Medical 4B): **~1.1 hours** (extremely fast, completes RQ1 4B non-thinking comparison against `gemma3-4b`).
  2. `medpsy-1.7b` (Thinking Medical 1.7B): **~12.2 hours**.
  3. `qwen3-1.7b-thinking` (Thinking Generic 1.7B Control): **~12.2 hours**.
  4. `qwen3-4b-thinking-2507` (Thinking Generic 4B Control): **~23.8 hours** (second quota cycle or dedicated window).

---

## 2. Phase A: Finish Pending v2 Evaluations

### Step A1: Unanswerable v2 SLM Evaluation (4 Models, < 10 mins)
Runs `smollm3-3b`, `phi4-mini`, `qwen3-4b`, and `gemma3-4b` on the canonical Unanswerable v2 test benchmark ($N=140$).

```bash
# In your Kaggle kernel or local GPU runner:
export UNANS_V2_SLM=1
python3 kaggle_runner/runner.py
```

### Step A2: MedPsy-4B Completions
Completes baseline, context, and RAG on remaining test splits.

```bash
# Baseline Unanswerable v2
export MEDPSY_UNANS_V2_BASELINE=1
python3 kaggle_runner/runner.py

# PubMedQA Baseline (Phase 5)
export MEDPSY_PUBMEDQA_BASELINE=1
python3 kaggle_runner/runner.py

# PubMedQA Full RAG (Phase 8)
export MEDPSY_PUBMEDQA_RAG=1
python3 kaggle_runner/runner.py
```

---

## 3. Phase B: v3 Validation Pilots ($N=50$, Verification Gate)

Run the $N=50$ stratified validation pilot on each new model to verify chat template tokenization, reasoning extraction, and latency.

### Step B1: Pilot MedPsy-1.7B
```bash
export V3_RUN=1
export V3_MODEL="medpsy-1.7b"
export V3_CONDITIONS="C1,C2,C3,C4,C5"
export V3_PILOT=1
export V3_PILOT_N=50
python3 kaggle_runner/runner.py
```
*Verify parse rate:*
```bash
python3 -c "import json; [print(f, sum(1 for line in open(f) if json.loads(line).get('pred_extracted') is not None)/50) for f in Path('outputs/v3/medpsy-1.7b').glob('*_pilot/*.jsonl')]"
```

### Step B2: Pilot Qwen3-1.7B-Thinking
```bash
export V3_RUN=1
export V3_MODEL="qwen3-1.7b-thinking"
export V3_CONDITIONS="C1,C2,C3,C4,C5"
export V3_PILOT=1
export V3_PILOT_N=50
python3 kaggle_runner/runner.py
```

### Step B3: Pilot MedGemma-1.5-4B-IT (Web-UI Run)
> [!NOTE]
> `medgemma-1.5-4b-it` requires Hugging Face authentication (`HF_TOKEN`) and must be launched in the Kaggle Web-UI.
```bash
# Ensure HF_TOKEN is in Kaggle Secrets
export V3_RUN=1
export V3_MODEL="medgemma-1.5-4b-it"
export V3_CONDITIONS="C1,C2,C3,C4,C5"
export V3_PILOT=1
export V3_PILOT_N=50
python3 kaggle_runner/runner.py
```

### Step B4: Pilot Qwen3-4B-Thinking-2507
```bash
export V3_RUN=1
export V3_MODEL="qwen3-4b-thinking-2507"
export V3_CONDITIONS="C1,C2,C3,C4,C5"
export V3_PILOT=1
export V3_PILOT_N=50
python3 kaggle_runner/runner.py
```

---

## 4. Phase C: v3 Full Test Evaluation

After pilots pass with 100% parse rates, run full test evaluations ($N=500$ for MedQA and PubMedQA, $N=140$ for Unanswerable v2).

### Priority 1: MedGemma-1.5-4B-IT (~1.1 GPU hours)
*Fastest run, directly closes the medical vs generic 4B non-thinking pair.*
```bash
export V3_RUN=1
export V3_MODEL="medgemma-1.5-4b-it"
export V3_CONDITIONS="C1,C2,C3,C4,C5"
export V3_PILOT=0
export V3_CONFIRM_TEST=1
python3 kaggle_runner/runner.py
```

### Priority 2: MedPsy-1.7B (~12.2 GPU hours)
*1.7B Medical thinking model.*
```bash
export V3_RUN=1
export V3_MODEL="medpsy-1.7b"
export V3_CONDITIONS="C1,C2,C3,C4,C5"
export V3_PILOT=0
export V3_CONFIRM_TEST=1
python3 kaggle_runner/runner.py
```

### Priority 3: Qwen3-1.7B-Thinking (~12.2 GPU hours)
*1.7B Generic thinking control.*
```bash
export V3_RUN=1
export V3_MODEL="qwen3-1.7b-thinking"
export V3_CONDITIONS="C1,C2,C3,C4,C5"
export V3_PILOT=0
export V3_CONFIRM_TEST=1
python3 kaggle_runner/runner.py
```

### Priority 4: Qwen3-4B-Thinking-2507 (~23.8 GPU hours)
*4B Generic thinking control.*
```bash
export V3_RUN=1
export V3_MODEL="qwen3-4b-thinking-2507"
export V3_CONDITIONS="C1,C2,C3,C4,C5"
export V3_PILOT=0
export V3_CONFIRM_TEST=1
python3 kaggle_runner/runner.py
```

---

## 5. Post-Run Verification & Analysis

Immediately after GPU outputs are downloaded into `outputs/v3/`:

1. **Verify Integrity & Leakage:**
   ```bash
   python3 audit/final_leakage_audit.py
   ```
2. **Run Statistical Matched-Pair Analysis (McNemar, Holm, CMH, Logistic Interaction):**
   ```bash
   python3 src/analysis_pairs.py
   ```
3. **Run Unanswerable v3 Analysis:**
   ```bash
   python3 src/analysis_unanswerable_v3.py
   ```
4. **Generate Figures 14–16:**
   ```bash
   python3 src/make_pair_figures.py
   ```
