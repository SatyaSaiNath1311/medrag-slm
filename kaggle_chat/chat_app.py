"""MedRAG-SLM Live Chatbot Application (Kaggle GPU).

Interactive clinical dialogue powered by:
- Hybrid BM25 + dense MedCPT FAISS retrieval (top-20)
- MedCPT cross-encoder reranker (top-5)
- Evaluated Small Language Models (Qwen3-4B default, Qwen3-1.7B fast)
- Temperature-scaled confidence calibration and smart adaptive gating
- Option-guided MCQ mode (benchmark evaluated) & free-text clinical QA mode

Run on Kaggle with GPU accelerator enabled:
  python kaggle_chat/chat_app.py
"""

from __future__ import annotations

import glob
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Setup Hugging Face token from Kaggle Secrets if available
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception:
    pass

# Ensure Gradio is available
try:
    import gradio as gr
except ImportError:
    import subprocess
    print("Installing Gradio...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr

import bm25s
import faiss
import numpy as np
import Stemmer
import torch

from src import encoders
from src.common import build_query, get_device, load_config, read_jsonl
from src.llm import LLM, baseline_prompt, parse_answer, parse_citations, rag_prompt

# -----------------------------------------------------------------------------
# Calibration Constants (Validation NLL Temperatures T*)
# -----------------------------------------------------------------------------
TEMPERATURES = {
    "qwen3-4b": {"baseline": 17.24, "rag": 20.73},
    "qwen3-1.7b": {"baseline": 24.72, "rag": 22.72},
}

MODEL_CONFIGS = {
    "qwen3-4b": {
        "name": "qwen3-4b",
        "id": "Qwen/Qwen3-4B",
        "dtype": "float16" if torch.cuda.is_available() else "float32",
    },
    "qwen3-1.7b": {
        "name": "qwen3-1.7b",
        "id": "Qwen/Qwen3-1.7B",
        "dtype": "float16" if torch.cuda.is_available() else "float32",
    },
}

# -----------------------------------------------------------------------------
# Path Discovery for Phase 3 (Chunks, BM25, FAISS)
# -----------------------------------------------------------------------------
def find_phase3_dir() -> Path:
    """Locate phase3 build artifacts on Kaggle or locally."""
    # 1. Look in Kaggle input directories
    hits = glob.glob("/kaggle/input/**/phase3/dense.faiss", recursive=True)
    if hits:
        return Path(hits[0]).parent
    hits = glob.glob("/kaggle/input/**/work/phase3/dense.faiss", recursive=True)
    if hits:
        return Path(hits[0]).parent

    # 2. Look in local project workspace
    local_p3 = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase3"
    if (local_p3 / "dense.faiss").exists():
        return local_p3

    raise FileNotFoundError(
        "Could not find phase3 artifacts (dense.faiss, bm25, chunks.jsonl). "
        "On Kaggle, ensure 'medrag-build' is attached: Add Input -> Your Work -> medrag-build."
    )


# -----------------------------------------------------------------------------
# Global Cached Retrieval & Generation State
# -----------------------------------------------------------------------------
_CACHED_RETRIEVAL: Dict[str, Any] = {}
_CACHED_MODELS: Dict[str, LLM] = {}


def get_retrieval_system(cfg_path: Optional[str] = None):
    """Load and cache retrieval models, indexes, and chunks once at startup."""
    global _CACHED_RETRIEVAL
    if _CACHED_RETRIEVAL:
        return _CACHED_RETRIEVAL

    print("\n[Init] Initializing retrieval components...", flush=True)
    t0 = time.time()
    device = get_device()
    p3_dir = find_phase3_dir()
    print(f"  Using phase3 directory: {p3_dir}")

    config_file = cfg_path or (REPO_ROOT / "configs" / "base.yaml")
    cfg = load_config(config_file)

    # 1. Load Chunks
    chunks_file = p3_dir / "chunks.jsonl"
    print(f"  Loading textbook chunks from {chunks_file.name}...")
    chunks = read_jsonl(chunks_file)
    print(f"  Loaded {len(chunks):,} textbook chunks.")

    # 2. Load FAISS Dense Index
    faiss_file = p3_dir / "dense.faiss"
    print(f"  Loading FAISS index from {faiss_file.name}...")
    faiss_index = faiss.read_index(str(faiss_file))

    # 3. Load BM25 Sparse Index
    bm25_dir = p3_dir / "bm25"
    print(f"  Loading BM25 index from {bm25_dir.name}...")
    bm25_index = bm25s.BM25.load(str(bm25_dir))
    stemmer = Stemmer.Stemmer("english")

    # 4. Encoders
    print("  Loading MedCPT query encoder...")
    query_enc = encoders.query_encoder(cfg, device)
    print("  Loading MedCPT cross-encoder reranker...")
    cross_enc = encoders.cross_encoder(cfg, device)

    _CACHED_RETRIEVAL = {
        "cfg": cfg,
        "device": device,
        "chunks": chunks,
        "faiss_index": faiss_index,
        "bm25_index": bm25_index,
        "stemmer": stemmer,
        "query_enc": query_enc,
        "cross_enc": cross_enc,
    }
    print(f"[Init] Retrieval components successfully cached in {time.time() - t0:.1f}s!\n", flush=True)
    return _CACHED_RETRIEVAL


def get_llm(model_key: str) -> LLM:
    """Retrieve or load the requested Small Language Model."""
    global _CACHED_MODELS
    clean_key = "qwen3-1.7b" if "1.7b" in model_key.lower() else "qwen3-4b"

    if clean_key in _CACHED_MODELS:
        return _CACHED_MODELS[clean_key]

    print(f"\n[Model] Loading SLM '{clean_key}' on {get_device()}...", flush=True)
    t0 = time.time()
    mcfg = MODEL_CONFIGS[clean_key]
    device = get_device()
    llm = LLM(mcfg, device)
    _CACHED_MODELS[clean_key] = llm
    print(f"[Model] '{clean_key}' loaded in {time.time() - t0:.1f}s!\n", flush=True)
    return llm


# -----------------------------------------------------------------------------
# Retrieval & Reranking Functions
# -----------------------------------------------------------------------------
def rrf_fuse(dense_ids: List[int], bm25_ids: List[int], rrf_k: int = 60, top_k: int = 20) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of dense and sparse candidate rankings."""
    scores: Dict[int, float] = {}
    d_rank: Dict[int, int] = {}
    b_rank: Dict[int, int] = {}

    for r, idx in enumerate(dense_ids, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + r)
        d_rank[idx] = r
    for r, idx in enumerate(bm25_ids, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + r)
        b_rank[idx] = r

    best = sorted(scores.keys(), key=lambda i: (-scores[i], i))[:top_k]
    return [
        {
            "idx": i,
            "rrf": round(scores[i], 6),
            "dense_rank": d_rank.get(i),
            "bm25_rank": b_rank.get(i),
        }
        for i in best
    ]


def retrieve_top_passages(query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], float]:
    """Execute hybrid retrieval (top-20) and cross-encoder rerank to extract top-5 passages."""
    t0 = time.perf_counter()
    retr = get_retrieval_system()
    cfg = retr["cfg"]
    chunks = retr["chunks"]
    faiss_index = retr["faiss_index"]
    bm25_index = retr["bm25_index"]
    stemmer = retr["stemmer"]
    query_enc = retr["query_enc"]
    cross_enc = retr["cross_enc"]

    # 1. Dense search
    qemb = query_enc.embed([query], label="encoding query")
    _, dense_res = faiss_index.search(qemb, min(50, faiss_index.ntotal))
    dense_ids = [int(i) for i in dense_res[0] if i >= 0]

    # 2. Sparse BM25 search
    tokenized = bm25s.tokenize([query], stopwords="en", stemmer=stemmer, show_progress=False)
    bm25_res, _ = bm25_index.retrieve(tokenized, k=min(50, len(chunks)), show_progress=False)
    bm25_ids = [int(i) for i in bm25_res[0]]

    # 3. Reciprocal Rank Fusion (top-20)
    cands = rrf_fuse(dense_ids, bm25_ids, rrf_k=60, top_k=20)

    # 4. Cross-Encoder Rerank
    flat_q = [query] * len(cands)
    flat_a = [f"{chunks[c['idx']]['title']}. {chunks[c['idx']]['text']}" for c in cands]
    scores = cross_enc.score(flat_q, flat_a, label="reranking candidates")

    order = np.argsort(-scores, kind="stable")[:top_k]
    top_passages = []
    for rank, j in enumerate(order, start=1):
        c = cands[int(j)]
        ch = chunks[c["idx"]]
        top_passages.append(
            {
                "rank": rank,
                "idx": c["idx"],
                "chunk_id": ch["chunk_id"],
                "title": ch["title"],
                "text": ch["text"],
                "rerank_score": round(float(scores[j]), 4),
            }
        )

    retrieval_sec = time.perf_counter() - t0
    return top_passages, retrieval_sec


# -----------------------------------------------------------------------------
# Temperature-Scaling & Calibration
# -----------------------------------------------------------------------------
def apply_temperature_scaling(
    letter_probs: Dict[str, float], T: float, eps: float = 1e-12
) -> Tuple[Dict[str, float], float]:
    """Compute temperature-scaled confidence using validation T* on log-probabilities."""
    if not letter_probs:
        return {}, 0.0

    items = list(letter_probs.items())
    logits = [math.log(max(p, eps)) / T for _, p in items]
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits)
    scaled_probs = {items[i][0]: exp_logits[i] / sum_exp for i in range(len(items))}
    conf = max(scaled_probs.values())
    return scaled_probs, conf


def classify_confidence_tier(conf: float, num_options: int = 4) -> Tuple[str, bool]:
    """Map calibrated probability to High / Medium / Low tiers."""
    # Baseline threshold: random guessing is 1/N
    threshold_low = 0.40 if num_options >= 4 else 0.45
    threshold_high = 0.60 if num_options >= 4 else 0.65

    if conf >= threshold_high:
        return "High", False
    elif conf >= threshold_low:
        return "Medium", False
    else:
        return "Low", True


# -----------------------------------------------------------------------------
# Core Chatbot Pipeline
# -----------------------------------------------------------------------------
def chat_pipeline(
    question: str,
    opt_a: str,
    opt_b: str,
    opt_c: str,
    opt_d: str,
    model_choice: str,
    evidence_setting: str,
    history: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], str, str, str, str, str]:
    """Execute complete clinical QA dialogue turn."""
    if not question.strip():
        return history, "", opt_a, opt_b, opt_c, opt_d

    t_total_start = time.perf_counter()
    clean_model_key = "qwen3-1.7b" if "1.7b" in model_choice.lower() else "qwen3-4b"
    llm = get_llm(clean_model_key)
    temps = TEMPERATURES.get(clean_model_key, {"baseline": 20.0, "rag": 20.0})

    # Parse provided options
    options: Dict[str, str] = {}
    if opt_a.strip():
        options["A"] = opt_a.strip()
    if opt_b.strip():
        options["B"] = opt_b.strip()
    if opt_c.strip():
        options["C"] = opt_c.strip()
    if opt_d.strip():
        options["D"] = opt_d.strip()

    is_mcq = len(options) >= 2

    # 1. Build Query and Execute Hybrid Retrieval
    q_dict = {"question": question.strip(), "options": options}
    search_query = build_query(q_dict, include_options=is_mcq)
    top_passages, retrieval_sec = retrieve_top_passages(search_query, top_k=5)

    # 2. Generation & Gating
    t_gen_start = time.perf_counter()

    if is_mcq:
        # Multiple-choice benchmark mode
        letters = sorted(options.keys())
        b_prompt = baseline_prompt(q_dict)
        r_prompt = rag_prompt(q_dict, top_passages, max_chars=1200)

        # Baseline generation
        b_res = llm.generate_safe([b_prompt], [letters], max_new_tokens=16)[0]
        b_pred = parse_answer(b_res["raw_output"], options) or letters[0]
        b_lp = b_res.get("letter_probs") or {L: (1.0 if L == b_pred else 0.0) for L in letters}
        _, b_conf = apply_temperature_scaling(b_lp, temps["baseline"])

        # RAG generation
        r_res = llm.generate_safe([r_prompt], [letters], max_new_tokens=40)[0]
        r_pred = parse_answer(r_res["raw_output"], options) or letters[0]
        r_lp = r_res.get("letter_probs") or {L: (1.0 if L == r_pred else 0.0) for L in letters}
        _, r_conf = apply_temperature_scaling(r_lp, temps["rag"])

        # Gating Decision
        if evidence_setting == "Never":
            chosen_method = "Baseline (Parametric)"
            chosen_letter = b_pred
            chosen_conf = b_conf
            gate_reason = "Manual override: textbook retrieval disabled."
            cited_passages = []
        elif evidence_setting == "Always":
            chosen_method = "Full RAG"
            chosen_letter = r_pred
            chosen_conf = r_conf
            gate_reason = "Manual override: forced textbook evidence RAG."
            cited_passages = parse_citations(r_res["raw_output"], len(top_passages))
        else:
            # Auto (Smart Confidence Gate)
            if r_conf > b_conf:
                chosen_method = "Smart Gate → Full RAG"
                chosen_letter = r_pred
                chosen_conf = r_conf
                gate_reason = f"RAG confidence ({r_conf*100:.1f}%) > Baseline ({b_conf*100:.1f}%)"
                cited_passages = parse_citations(r_res["raw_output"], len(top_passages))
            else:
                chosen_method = "Smart Gate → Baseline"
                chosen_letter = b_pred
                chosen_conf = b_conf
                gate_reason = f"Baseline confidence ({b_conf*100:.1f}%) ≥ RAG ({r_conf*100:.1f}%)"
                cited_passages = []

        chosen_text = options.get(chosen_letter, "")
        tier, is_low_conf = classify_confidence_tier(chosen_conf, num_options=len(options))
        conf_display = f"{tier} ({chosen_conf*100:.1f}%)"
        answer_text = f"**Option {chosen_letter}**: {chosen_text}"

    else:
        # Free-text mode (not evaluated benchmark)
        ev_text = "\n".join(f"[{i}] ({p['title']}) {p['text'][:1200]}" for i, p in enumerate(top_passages, 1))
        free_prompt = (
            "You are a medical expert providing clinical decision support. "
            "Using the provided evidence passages from authoritative medical textbooks, "
            "provide a direct, concise, and clinically rigorous answer to the question. "
            "Cite supporting passages in square brackets like [1] or [2] where applicable.\n\n"
            f"Evidence:\n{ev_text}\n\n"
            f"Question: {question.strip()}\n\n"
            "Answer:"
        )

        msgs = [{"role": "user", "content": free_prompt}]
        prompt_text = llm.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = llm.tok([prompt_text], return_tensors="pt")
        first_param = next(llm.model.parameters())
        enc = {k: v.to(first_param.device) for k, v in enc.items()}

        with torch.inference_mode():
            out_tokens = llm.model.generate(
                **enc,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=llm.pad_id,
                eos_token_id=sorted(llm.eos),
            )
        gen_ids = out_tokens[0, enc["input_ids"].shape[1]:].tolist()
        raw_free = llm.tok.decode(gen_ids, skip_special_tokens=True).strip()

        chosen_method = "Full RAG (Free-Text)"
        gate_reason = "Free-text clinical QA mode (no option choices provided)."
        conf_display = "N/A (Free-Text Mode)"
        is_low_conf = False
        cited_passages = parse_citations(raw_free, len(top_passages))
        answer_text = f"*free-text mode (not part of the evaluated benchmark)*\n\n{raw_free}"

    gen_sec = time.perf_counter() - t_gen_start
    total_sec = time.perf_counter() - t_total_start

    # 3. Format Expandable Sources
    sources_md_parts = []
    for p in top_passages:
        p_rank = p["rank"]
        is_cited = p_rank in cited_passages
        cite_badge = " ⭐ **[Cited by Model]**" if is_cited else ""
        sources_md_parts.append(
            f"<details><summary><b>[{p_rank}] {p['title']}</b> (Rerank score: {p['rerank_score']:.2f}){cite_badge}</summary>\n\n"
            f"> {p['text']}\n</details>"
        )
    sources_md = "\n\n".join(sources_md_parts)

    # 4. Construct Final Bot Reply
    warning_prefix = "⚠️ **Low confidence — refer to a clinician**\n\n" if is_low_conf else ""

    bot_reply = (
        f"{warning_prefix}### 🩺 Clinical Recommendation\n"
        f"{answer_text}\n\n"
        f"---\n"
        f"**📊 Decision & Confidence Analysis:**\n"
        f"- **Method Selected:** `{chosen_method}` ({gate_reason})\n"
        f"- **Calibrated Confidence:** **{conf_display}**"
        + (f" *(scaled with validation $T^*={temps['rag'] if 'RAG' in chosen_method else temps['baseline']:.2f}$)*" if is_mcq else "")
        + f"\n- **Model:** `{clean_model_key}` &nbsp;|&nbsp; **Evidence Mode:** `{evidence_setting}`\n"
        f"- **Latency:** `{total_sec:.2f}s` *(Retrieval: {retrieval_sec:.2f}s | Generation: {gen_sec:.2f}s)*\n\n"
        f"---\n"
        f"### 📚 Retrieved Textbook Evidence\n"
        f"{sources_md}\n\n"
        f"---\n"
        f"*Decision support only — verify with a clinician.*"
    )

    # User message summary in chat history
    if is_mcq:
        opt_summary = " &nbsp;|&nbsp; ".join(f"**{k}:** {v}" for k, v in sorted(options.items()))
        user_msg = f"**{question.strip()}**\n\n<small>{opt_summary}</small>"
    else:
        user_msg = question.strip()

    updated_history = history + [(user_msg, bot_reply)]
    return updated_history, "", opt_a, opt_b, opt_c, opt_d


# -----------------------------------------------------------------------------
# Gradio UI Construction
# -----------------------------------------------------------------------------
def build_interface() -> Tuple[gr.Blocks, str]:
    """Build the clean Gradio interactive web application."""
    custom_css = """
    .gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .chat-header { margin-bottom: 0.8rem; }
    """

    with gr.Blocks(title="MedRAG-SLM Clinical Chatbot") as demo:
        gr.Markdown(
            "# 🩺 MedRAG-SLM: Live Clinical Decision Support Chatbot\n"
            "**Retrieval-Augmented Small Language Models on Clinical Textbooks with Adaptive Gating.**\n"
            "*Runs live on Kaggle GPU using MedCPT hybrid search and calibrated confidence.*"
        )

        chatbot = gr.Chatbot(label="Clinical Dialogue & Evidence Verification", height=480)

        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Patient Case / Clinical Question",
                    placeholder="Enter a clinical vignette, diagnostic query, or biomedical question...",
                    lines=3,
                )

                with gr.Accordion("Multiple-Choice Options (Optional — for Evaluated Benchmark Mode)", open=True):
                    with gr.Row():
                        opt_a = gr.Textbox(label="Option A", placeholder="e.g. Left anterior descending artery")
                        opt_b = gr.Textbox(label="Option B", placeholder="e.g. Right coronary artery")
                    with gr.Row():
                        opt_c = gr.Textbox(label="Option C", placeholder="e.g. Left circumflex artery")
                        opt_d = gr.Textbox(label="Option D", placeholder="e.g. Left main coronary artery")

            with gr.Column(scale=2):
                model_choice = gr.Dropdown(
                    choices=["Qwen3-4B (default)", "Qwen3-1.7B (fast)"],
                    value="Qwen3-4B (default)",
                    label="SLM Model Architecture",
                )
                evidence_setting = gr.Radio(
                    choices=["Auto (smart gate)", "Always", "Never"],
                    value="Auto (smart gate)",
                    label="Textbook Retrieval Setting",
                    info="Smart gate routes based on model confidence.",
                )
                with gr.Row():
                    submit_btn = gr.Button("🩺 Submit Query", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

        # Clickable Examples
        gr.Markdown("### 💡 Clickable Benchmark Examples")
        examples_data = [
            [
                "A 45-year-old male with a history of hypertension presents to the emergency department with severe retrosternal crushing chest pain radiating to his left shoulder and diaphoresis. An electrocardiogram (ECG) shows ST-segment elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?",
                "Left anterior descending artery",
                "Right coronary artery",
                "Left circumflex artery",
                "Left main coronary artery",
            ],
            [
                "Does gadofosveset-enhanced magnetic resonance angiography improve diagnostic accuracy for carotid artery stenosis compared to digital subtraction angiography?",
                "yes",
                "no",
                "maybe",
                "",
            ],
            [
                "What is the recommended first-line treatment and intramuscular injection dose for an adult experiencing acute anaphylaxis?",
                "",
                "",
                "",
                "",
            ],
        ]

        gr.Examples(
            examples=examples_data,
            inputs=[question_input, opt_a, opt_b, opt_c, opt_d],
            label="Click an example to load clinical vignette & options:",
        )

        # Wire Events
        submit_btn.click(
            fn=chat_pipeline,
            inputs=[question_input, opt_a, opt_b, opt_c, opt_d, model_choice, evidence_setting, chatbot],
            outputs=[chatbot, question_input, opt_a, opt_b, opt_c, opt_d],
        )

        clear_btn.click(
            fn=lambda: ([], "", "", "", "", ""),
            inputs=None,
            outputs=[chatbot, question_input, opt_a, opt_b, opt_c, opt_d],
        )

    return demo, custom_css


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import inspect

    print("=" * 70)
    print("Starting MedRAG-SLM Gradio Chatbot (Kaggle GPU Session)...")
    print("=" * 70)

    # Pre-cache retrieval and default model
    get_retrieval_system()
    get_llm("qwen3-4b")

    app, custom_css = build_interface()

    print("\n" + "=" * 70)
    print("🎉 LAUNCHING GRADIO APP WITH PUBLIC LINK...")
    print("=" * 70 + "\n", flush=True)

    launch_kwargs = {
        "share": True,
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }
    launch_params = inspect.signature(app.launch).parameters
    if "theme" in launch_params:
        launch_kwargs["theme"] = gr.themes.Soft()
    if "css" in launch_params:
        launch_kwargs["css"] = custom_css

    app.queue().launch(**launch_kwargs)
