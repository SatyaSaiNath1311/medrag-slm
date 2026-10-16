# Model Cards Verification and Hugging Face Specifications

**Date:** 2026-10-16  
**Reference Document:** `docs/DESIGN_GENERIC_VS_MEDICAL.md`  
**Purpose:** Pre-flight verification of Hugging Face repository IDs, base models, licensing, gating, chat templates, and execution dtypes across the v3 matched-pair study.

---

## 1. Summary Matrix

| Model Identifier | Hugging Face Repo ID | Base Model / Family | Gated? | License / Access Terms | Reasoning / Thinking Mode | Recommended Dtype | Loader / Execution Route |
|---|---|---|:---:|---|---|:---:|---|
| `medpsy-1.7b` | `qvac/MedPsy-1.7B` | Qwen3-1.7B | **No** | Apache 2.0 | Yes (`<think>`, cap=1,024) | `float16` | Standard `AutoModelForCausalLM` |
| `qwen3-1.7b-thinking` | `Qwen/Qwen3-1.7B` | Qwen3-1.7B | **No** | Apache 2.0 | Yes (`enable_thinking=True`, cap=1,024) | `float16` | Standard `AutoModelForCausalLM` |
| `qwen3-4b-thinking-2507` | `Qwen/Qwen3-4B-Thinking-2507` | Qwen3-4B | **No** | Apache 2.0 | Yes (Native thinking, cap=1,024) | `float16` | Standard `AutoModelForCausalLM` |
| `medgemma-1.5-4b-it` | `google/medgemma-1.5-4b-it` | Gemma 3 4B | **YES** | Health AI Developer Foundations (HAI-DEF) | No (Instruction-tuned) | `float32` (T4 dual-GPU) | `loader: gemma3` via Kaggle Web-UI with `HF_TOKEN` |

---

## 2. Detailed Model Card Findings

### 2.1 MedPsy-1.7B
- **Hugging Face ID**: [`qvac/MedPsy-1.7B`](https://huggingface.co/qvac/MedPsy-1.7B)
- **Developer / Organization**: Tether AI / QVAC Psy initiative
- **Base Model**: Post-trained on top of `Qwen/Qwen3-1.7B` in thinking/reasoning mode.
- **Architecture**: `Qwen3ForCausalLM` (1.7 billion parameters).
- **Gating**: Open access. No gate, no license acceptance agreement needed.
- **Chat Template**: Native Jinja chat template with `<think>...</think>` output structure.
- **Thinking Configuration**: Cap at 1,024 tokens (consistent with `medpsy-4b`). Truncated outputs flagged via `truncated=True`.
- **Inference Requirements**: `torch.float16`, fits comfortably on a single T4 GPU (~3.5 GB VRAM).
- **Verification Status**: **CONFIRMED**.

### 2.2 Qwen3-1.7B (Thinking ON Control)
- **Hugging Face ID**: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Developer / Organization**: Qwen Team, Alibaba
- **Base Model**: Direct base/instruct release.
- **Gating**: Open access. Apache 2.0.
- **Chat Template**: Hybrid architecture supporting dynamic reasoning. When invoked with `enable_thinking=True` or prompt instructions, emits `<think>...</think>`.
- **Thinking Configuration**: Thinking cap set to 1,024 tokens to match `medpsy-1.7b`.
- **Inference Requirements**: `torch.float16`, single T4 GPU (~3.5 GB VRAM).
- **Verification Status**: **CONFIRMED**.

### 2.3 Qwen3-4B-Thinking-2507 (Generic Reasoning Control for MedPsy-4B)
- **Hugging Face ID**: [`Qwen/Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- **Developer / Organization**: Qwen Team, Alibaba (version signature 2507)
- **Base Model**: Qwen3-4B architecture with long-context native reasoning.
- **Gating**: Open access. Apache 2.0.
- **Chat Template**: Thinking mode enabled by default; native `<think>...</think>` tags generated before final answers.
- **Thinking Configuration**: Thinking cap set to 1,024 tokens to exactly match `medpsy-4b`.
- **Inference Requirements**: `torch.float16`, single or dual T4 GPU (~8 GB VRAM).
- **Verification Status**: **CONFIRMED**.

### 2.4 MedGemma-1.5-4B-it
- **Hugging Face ID**: [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it)
- **Developer / Organization**: Google Health AI
- **Base Model**: Gemma 3 4B (`google/gemma-3-4b-it`).
- **Gating**: **STRICTLY GATED**. Users must accept Google's Health AI Developer Foundations (HAI-DEF) terms on the Hugging Face model card before weights can be fetched.
- **Kaggle Execution Route**:
  1. Must run in Kaggle Notebook Web UI with Kaggle Secrets containing `HF_TOKEN`.
  2. If mounted from Kaggle Models: Kaggle handle `google/medgemma-1-5/transformers/medgemma-1.5-4b-it` or direct HF fallback.
- **Dtype & Precision**:
  - **Important T4 Lesson from `gemma3-4b`**: `float16` produces numerical overflow (NaN logits) on NVIDIA T4 GPUs for Gemma 3 architecture. Must use `dtype: float32` distributed across both T4 GPUs via `device_map="auto"` or `bfloat16` if hardware supports it.
  - Set `batch_size_rag: 2` to avoid 28 GB combined VRAM exhaustion during multi-passage RAG inference.
- **Thinking Mode**: Instruction-tuned dialogue model (not a thinking model). Generates direct answers without `<think>` blocks.
- **Verification Status**: **CONFIRMED**.

---

## 3. Matched-Pair Validation Checklist

| Pair ID | Generic Arm | Medical Arm | Shared Base Confirmed? | Reasoning Confound Controlled? |
|:---:|---|---|:---:|:---:|
| **P1** | `qwen3-1.7b` (done, non-thinking) | `medpsy-1.7b` | **Yes** (`Qwen/Qwen3-1.7B`) | Controlled by `qwen3-1.7b-thinking` |
| **P2** | `qwen3-4b` (done, non-thinking) | `medpsy-4b` (done) | **Yes** (`Qwen3-4B-Thinking-2507`) | Controlled by `qwen3-4b-thinking-2507` |
| **P3** | `gemma3-4b` (done) | `medgemma-1.5-4b-it` | **Yes** (`Gemma 3 4B`) | Both instruction-tuned (no thinking confound) |

---

## 4. Action Items & User TODOs
- [ ] User must log in to Hugging Face, visit [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it), and accept the Health AI Developer Foundations terms of use.
- [ ] User must verify that their Kaggle account Secret `HF_TOKEN` has read permissions granted for the accepted MedGemma repository prior to launching the MedGemma Kaggle kernel.
