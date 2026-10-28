# Reproduction Guide: Phase 1 MedRAG-SLM & V3 Matched-Pair Study

This document specifies the exact software and hardware environments, cryptographic data digests, model identifiers, and commands required to reproduce all analyses, verification gates, and experimental tables from committed repository outputs.

---

## 1. What Is and Is Not Reproducible Without a GPU

- **Fully Reproducible Locally on CPU (Laptop in < 2 minutes):**
  - All statistical tests, paired bootstrap intervals, Wilson score CIs, McNemar tests, and Holm-Bonferroni corrections.
  - Verification gate (`bash scripts/verify_all.sh`) and frozen data cryptographic SHA-256 audits.
  - Table generation: `docs/tables/final_results_v3.md`, `.csv`, `.tex`.
  - Paper fact ledger generation and audit: `docs/PAPER_FACTS.md` (`python3 -m src.build_paper_facts --check`).
  - Report numerical comparison (`python3 scripts/compare_report_numbers.py`).
  - Unit tests (`pytest`).
- **Requires GPU (Kaggle 2x NVIDIA T4, 7.29 GPU hours measured across validation sweeps):**
  - Full model inference generation from Hugging Face weights to produce raw `.jsonl` responses for thinking models (1,024 to 4,096 tokens) and direct-answer direct decoders. Measured GPU-hours are recorded in committed manifests for validation sweeps (6.92 h + 0.37 h = 7.29 h). Earlier test-set runs were completed on Kaggle but did not log elapsed GPU hours to their committed manifests.

---

## 2. Hardware and Software Environment

Read from committed `outputs/sc_test/medpsy-4b/work/env/environment.txt` and Kaggle manifests:

| Component | Specification |
|---|---|
| **Python** | Python 3.10.12+ (Linux / macOS compatible) |
| **PyTorch** | `torch==2.6.0` (Pair B sweep) / `torch==2.10.0+cu128` (Gemma sensitivity) |
| **Transformers** | `transformers==4.49.0` (or `transformers==5.0.0`) |
| **Accelerate** | `accelerate==1.13.0` |
| **SciPy / NumPy** | `scipy>=1.11.0`, `numpy>=1.24.0` |
| **Hardware (Inference)** | 2x NVIDIA Tesla T4 (15.6 GiB VRAM per GPU; 31.2 GiB total) |
| **Decoding Precision** | `float16` for Qwen and MedPsy models; `float32` for Gemma-3 / MedGemma models on dual-GPU |

---

## 3. Cryptographic Benchmark Data & Frozen Hashes

All splits, question IDs, and ground-truth options were built with fixed random seed `42` and are cryptographically frozen.

| Benchmark Partition | Relative Path | Canonical SHA-256 Digest | Questions (N) |
|---|---|---|---|
| **Phase 1 Test Split** | `outputs/kaggle_build/work/phase1/test.jsonl` | `fe7fe3fb88c77c4ffc6574c8716ec0b3f4787fbfaf2c22e2eb5741ac040cc34a` | 1,150 (500 MedQA, 500 PubMedQA, 150 Unans) |
| **Unanswerable v2 Test** | `data/unanswerable_v2/test.jsonl` | `a0e45676dced360b2ac64e8fc33dbb49782c4fd9022e93190e633c9d66396078` | 140 (100 Challenge + 40 Control) |
| **Unanswerable v2 Val** | `data/unanswerable_v2/val.jsonl` | `f5d779faa76e1c6a85f59718be72691a6d676b374070f346e4b332b70ba7b3a4` | 260 (200 Challenge + 60 Control) |

---

## 4. Model Identifiers and Hugging Face Hub Repositories

| Model Name | Canonical Hugging Face Repository ID | Architecture / Family | Reasoning Mode | Cap ($c$) |
|---|---|---|---|---|
| `medpsy-1.7b` | `qvac/MedPsy-1.7B` | Causal LM (Qwen2.5 base) | Thinking enabled | 1,024 tokens |
| `qwen3-1.7b-thinking` | `Qwen/Qwen3-1.7B` | Causal LM | Thinking enabled | 1,024 tokens |
| `gemma3-4b` | `google/gemma-3-4b-it` | Gemma 3 Direct | Direct Answer | 16 tokens |
| `medgemma-1.5-4b-it` | `google/medgemma-1.5-4b-it` | Gemma Direct | Direct Answer | 16 tokens |
| `medpsy-4b` | `qvac/MedPsy-4B` | Causal LM (Qwen2.5 base) | Thinking enabled | 1,024 tokens |
| `qwen3-4b-thinking-2507` | `Qwen/Qwen3-4B-Thinking-2507` | Causal LM | Thinking enabled | 1,024 / 4,096 tokens |

---

## 5. Kaggle Compute Usage (from Manifests)

| Run / Experiment | Split | Models | GPU Hours | Status & Manifest Provenance |
|---|---|---|---|---|
| **Pair B Token-Budget Sweep** | Validation ($N=100$) | `qwen3-4b-thinking-2507`, `medpsy-4b` | 6.92 hrs | **Measured**: `outputs/sweeps/budget_pair_b/manifest.json` (`hours_used`: 6.9229, wall-clock for sequential execution of both models on 2x T4). Per-model manifests: `manifest_medpsy-4b.json` (0.9328 h) and `manifest_qwen3-4b-thinking-2507.json` (6.8788 h). |
| **Gemma Prompt Sensitivity** | Validation ($N=200 \times 3$) | `gemma3-4b`, `medgemma-1.5-4b-it` | 0.37 hrs | **Measured**: `outputs/sweeps/gemma_prompt_sensitivity/manifest.json` (`hours_used`: 0.3668, ~22.0 min, wall-clock for sequential execution of both models across 3 prompt variants on 2x T4). Per-model manifests: `manifest_gemma3-4b.json` (0.1591 h) and `manifest_medgemma-1.5-4b-it.json` (0.1600 h; sum of per-model = 0.3191 h). |
| **Pair A C1 / C2 Test** | Test ($N=500$) | `medpsy-1.7b`, `qwen3-1.7b-thinking` | not recorded in a committed manifest | Manifests under `outputs/v3/medpsy-1.7b/` and `outputs/v3/qwen3-1.7b-thinking/` record configuration, model repo IDs, and timestamps, but do not record `hours_used`. |
| **Gemma Pair C1–C5 Test** | Test ($N=1,140$) | `gemma3-4b`, `medgemma-1.5-4b-it` | not recorded in a committed manifest | Manifests under `outputs/v3/gemma3-4b/manifest.json` and `outputs/v3/medgemma-1.5-4b-it/manifest.json` record configuration and timestamps, but do not record `hours_used`. |
| **Pair B C1 / C2 Test** | Test ($N=500$) | `qwen3-4b-thinking-2507` | not recorded in a committed manifest | No manifest is committed under `outputs/v3/qwen3-4b-thinking-2507/`. |

---

## 6. Exact Reproduction Commands (Run Locally on CPU)

### Step 1: Run Full Verification Gate
```bash
bash scripts/verify_all.sh
```
Expected output:
```
================================================================================
PHASE 1 MEDRAG-SLM: MASTER VERIFICATION GATE
================================================================================
PASS: All pytest unit tests passed
PASS: Frozen benchmark datasets SHA-256 audit verified
PASS: Outputs and sweeps integrity audit (IDs, row counts, manifests, no hf_ tokens, rule hashes)
PASS: docs/PAPER_FACTS.md is strictly identical to master fact ledger
PASS: docs/tables/final_results_v3 (md, csv, tex) match committed outputs
PASS: All V3 mode flags in kaggle_runner/runner.py are set to False
PASS: git working tree is clean under outputs/ and docs/
================================================================================
VERIFICATION SUMMARY: All 7 checks PASSED.
================================================================================
```

### Step 2: Regenerate Consolidated Tables & LaTeX
```bash
python3 -m src.final_results_v3
```

### Step 3: Regenerate Master Paper Facts Ledger
```bash
python3 -m src.build_paper_facts
```

### Step 4: Re-run Individual Analysis Scripts
```bash
python3 src/analysis_pair_a_medqa.py
python3 src/analysis_gemma_pair.py
python3 src/analysis_pair_b.py
python3 src/analysis_budget_sweep.py
python3 src/analysis_gemma_prompt_sensitivity.py
python3 scripts/generate_pair_b_tables.py
```

### Step 5: Verify Numeric Stability Across Report Regenerations
```bash
python3 scripts/compare_report_numbers.py outputs/analysis/budget_pair_b.md outputs/analysis/budget_pair_b.md
python3 scripts/compare_report_numbers.py outputs/analysis/gemma_prompt_sensitivity.md outputs/analysis/gemma_prompt_sensitivity.md
```
