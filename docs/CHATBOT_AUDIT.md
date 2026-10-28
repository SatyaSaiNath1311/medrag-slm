# MedRAG-SLM Chatbot Architecture & Codebase Audit

**Date:** 2026-10-28  
**Scope:** Read-only inspection and technical audit of all chatbot and clinical dialogue modules in the repository (`kaggle_chat/`, `src/encoders.py`, `src/llm.py`, `tests/test_chat_app.py`).

---

## 1. Component Implementation Audit Matrix

| Component | Status | Source File & Functions | Architecture & Runtime Behavior |
|---|---|---|---|
| **Model Loading** | **EXISTS** | `kaggle_chat/chat_app.py`:<br>`get_llm()`, `get_device_placement()` | Loads Hugging Face causal LM (`qvac/MedPsy-4B` default or `Qwen/Qwen3-1.7B` fast) via `src.llm.LLM`. Sets `device="cuda:0"` explicitly (`device_map={"": 0}`), strictly avoiding `device_map="auto"`. Uses `float16` on CUDA, `float32` on CPU. LLM is loaded first during startup to secure GPU VRAM before encoders load. |
| **Emergency Red-Flag Check** | **EXISTS** | `kaggle_chat/chat_app.py`:<br>`check_emergency()`, `is_negated()`, `is_infant_under_3_months()` | Comprehensive clinical screening for 11 emergency conditions (chest pain, severe breathlessness, unconsciousness, convulsions, vomiting blood, stroke signs, anaphylaxis, suicide, poisoning, pregnancy complications, infant <3m fever). Uses clause-bounded negation handling (6-word window, clause delimiters `.,;:!?` and `but`). Prepends `EMERGENCY_PREFIX` referral banner. |
| **Retrieval & Reranking** | **EXISTS** | `kaggle_chat/chat_app.py`:<br>`get_retrieval_system()`, `retrieve_top_passages()`, `rrf_fuse()` | Hybrid sparse + dense retrieval over 124,077 textbook chunks. BM25 (`bm25s` on CPU) + MedCPT FAISS dense index (strictly on CPU RAM) fused via Reciprocal Rank Fusion ($k=60$) to top-20. MedCPT cross-encoder (`encoders.cross_encoder` placed on `cuda:1` if dual-GPU, else CPU) reranks top-20 to top-5. Computes validation 20th percentile rerank threshold. |
| **Prompt Construction** | **EXISTS** | `kaggle_chat/chat_app.py`:<br>`chat_pipeline()`,<br>`kaggle_chat/question_types.py`:<br>`build_hw_prompt()`, `classify_question()` | Dual prompt generation: (1) MCQ mode uses `reasoning_rag_prompt` or `rag_prompt`. (2) Community health worker mode classifies queries (`classify_question` -> general, pregnancy_child, diagnosis_treatment, emergency, greeting, unclear) and formats tailored clinical prompts (`build_hw_prompt`). |
| **Generation Settings** | **PARTIAL** | `kaggle_chat/chat_app.py`:<br>`chat_pipeline()` lines 669–738 | Uses strictly greedy autoregressive decoding (`do_sample=False`). In MCQ mode: `max_new_tokens=1024` for reasoning, `40` for direct. In HW mode: `max_new_tokens=1024` for MedPsy-4B, `350` for Qwen3-1.7B.<br>*(Marked PARTIAL: MedPsy-4B requires cap 1536 per validation selection rule to reach $\le 5\%$ truncation; sampling parameters are not runtime-configurable).* |
| **Citation Handling** | **EXISTS** | `kaggle_chat/chat_app.py`:<br>`chat_pipeline()` lines 762–780,<br>`src/llm.py`: `parse_citations()` | Parses `[n]` bracketed citation markers in the generated text. Filters top-5 passages to display only passages actually cited in the final answer. Formats citations with source textbook title, 200-char preview, and expandable `<details><summary>Expand full passage</summary>` accordion with full chunk text. |
| **Abstention Safety** | **EXISTS** | `kaggle_chat/chat_app.py`:<br>`evaluate_abstention()` | Multi-tier calibrated safety check. Abstains and returns `ABSTAIN_MESSAGE` if: (1) generation was truncated due to budget exhaustion (`truncated=True`), (2) model output explicitly contains `"INSUFFICIENT INFORMATION"`, or (3) top rerank score is below the 20th percentile validation threshold (`top_rerank_score < rerank_threshold_p20`). |
| **Session Handling & Multi-Turn** | **PARTIAL** | `kaggle_chat/chat_app.py`:<br>`normalize_history()`, `build_interface()` | Gradio UI manages in-browser turn history via `gr.Chatbot(type="messages")`. Messages are normalized and rendered.<br>*(Marked PARTIAL: Retrieval and prompt construction evaluate only the current turn query; prior conversation history is not incorporated into retrieval query expansion or LLM context. No server-side session persistence).* |
| **Logging & Observability** | **PARTIAL** | `kaggle_chat/chat_app.py`:<br>`print_gpu_memory()`, stdout timings | Prints per-GPU memory (`torch.cuda.memory_allocated` and `mem_get_info` for `cuda:0` and `cuda:1`) and latency breakdown (`retrieval_sec`, `gen_sec`, `total_sec`) to console.<br>*(Marked PARTIAL: No persistent disk logging of queries, emergency triage flags, or abstention telemetry).* |
| **Execution Environment & Tunnel** | **EXISTS** | `kaggle_chat/chat_app.py`,<br>`kaggle_chat/README.md` | Dual GPU Kaggle execution architecture: MedPsy-4B placed on `cuda:0` (~8 GB in fp16); MedCPT encoders on `cuda:1` (~1.5 GB in fp16); FAISS index on CPU RAM (~4 GB); BM25 index and chunks on CPU RAM. Public internet access via Cloudflare Quick Tunnel (`cloudflared tunnel --url http://localhost:7860`) exposing HTTPS `*.trycloudflare.com` URL. |

---

## 2. Local Test Skipping Analysis (`tests/test_chat_app.py`)

### Exact Reason for Skip
When running `pytest tests/test_chat_app.py` locally without GPU or heavy dependencies, the test suite is skipped with:
```text
SKIPPED [1] tests/test_chat_app.py: gradio is an optional dependency for kaggle_chat.chat_app
```

### Technical Root Cause
1. `kaggle_chat/chat_app.py` imports optional runtime packages at the module level:
   - `gradio`
   - `faiss`
   - `bm25s`
   - `Stemmer`
   - `torch`
   - `transformers`
2. In `kaggle_chat/chat_app.py` (lines 52–60), an inline fallback is present:
   ```python
   try:
       import gradio as gr
   except ImportError:
       import subprocess
       print("Installing Gradio...", flush=True)
       subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
       import gradio as gr
   ```
   If imported in an environment where `gradio` is missing, `chat_app.py` initiates an unconstrained, blocking `pip install` subprocess over the network. In sandboxed or offline test runs, this network call stalls for 4+ minutes and fails collection.
3. To protect test execution, `tests/test_chat_app.py` (lines 24–34) registers safe module-level import guards:
   ```python
   try:
       import pytest
       for _pkg in ["gradio", "faiss", "bm25s", "Stemmer"]:
           pytest.importorskip(_pkg, reason=f"{_pkg} is an optional dependency for kaggle_chat.chat_app")
   except ImportError:
       ...
   ```
   When any of these optional packages are absent, pytest deselects the entire module in ~3 seconds, preserving test-runner performance.

---

## 3. Specifications for Standalone Production Backend API

Based on the existing audited codebase, migrating from the monolithic Gradio script to an enterprise-grade standalone backend service requires:

### 3.1 Framework & Entrypoint
- Replace Gradio Blocks with a high-performance **ASGI framework (FastAPI)** with asynchronous worker support.
- Decouple model inference and retrieval workers into a dedicated worker pool (e.g. Ray Serve or Celery worker process) to prevent blocking HTTP endpoints.

### 3.2 Required REST Endpoints
1. `POST /api/v1/chat`:
   - Request Body:
     ```json
     {
       "session_id": "uuid4",
       "query": "A child has fever for 3 days, rash and red eyes.",
       "options": null,
       "model": "medpsy-4b",
       "stream": true
     }
     ```
   - Response: Server-Sent Events (SSE) streaming tokens or a structured JSON response including answer text, citations, emergency banner, and timing metadata.
2. `POST /api/v1/triage`:
   - Standalone lightweight endpoint executing `check_emergency(query)`.
   - Returns emergency classification in $<5$ ms without loading model weights or running retrieval.
3. `GET /api/v1/health`:
   - System liveness and readiness probe reporting CUDA availability, allocated VRAM per GPU, index vector count, and model load status.

### 3.3 Multi-Turn Context & Query Rewriting
- Integrate a lightweight conversational contextualizer: when a user asks a follow-up query (e.g. *"What is the dosage for infants?"*), rewrite the query using conversation history before querying BM25 and MedCPT to prevent retrieval degradation.

### 3.4 Persistent Telemetry & Structured Logging
- Implement structured JSONL audit logging recording:
  - Timestamp & anonymized session ID
  - Query classification (`mcq`, `general`, `pregnancy_child`, `emergency`)
  - Red-flag detection status & emergency referral rationale
  - Top-5 chunk IDs and cross-encoder rerank scores
  - Abstention decision and criterion
  - Total inference latency and token counts

### 3.5 Operational Config Updates
- Update the default reasoning token cap from `1024` to `1536` in accordance with the pre-registered model selection rule (`docs/MODEL_SELECTION_RULE.md`), reducing expected reasoning truncation from 6.0% to 2.0%.
