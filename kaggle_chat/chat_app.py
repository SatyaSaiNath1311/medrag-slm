"""MedRAG-SLM Community Health Worker Live Chatbot (Kaggle GPU).

Interactive clinical dialogue tailored for community health workers, powered by:
- Hybrid BM25 + dense MedCPT FAISS retrieval (top-20)
- MedCPT cross-encoder reranker (top-5)
- Authoritative medical textbook corpus (Harrison, Katzung, Schwartz, Nelson, etc.)
- MedPsy-4B reasoning model (default, 1024 thinking tokens) & Qwen3-1.7B (fast)
- Red-flag emergency screening (immediate 108 referral for acute danger signs)
- Multi-tier abstention safety (truncated thinking budget, INSUFFICIENT INFORMATION, rerank threshold)
- Option-guided MCQ mode (evaluated benchmark) & community health-worker guidance mode

Run on Kaggle with GPU accelerator enabled:
  python kaggle_chat/chat_app.py
"""

from __future__ import annotations

import os

# Configure PyTorch CUDA memory allocator before importing torch to mitigate memory fragmentation on 16GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
from src.llm import (
    ABSTAIN_MESSAGE,
    LLM,
    baseline_prompt,
    parse_answer,
    parse_citations,
    rag_prompt,
    reasoning_rag_prompt,
)

# -----------------------------------------------------------------------------
# Model Configurations
# -----------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "medpsy-4b": {
        "name": "medpsy-4b",
        "id": "qvac/MedPsy-4B",
        "dtype": "float16" if torch.cuda.is_available() else "float32",
        "reasoning": True,
        "max_new_tokens_reasoning": 1024,
    },
    "qwen3-1.7b": {
        "name": "qwen3-1.7b",
        "id": "Qwen/Qwen3-1.7B",
        "dtype": "float16" if torch.cuda.is_available() else "float32",
        "reasoning": False,
    },
}

DEFAULT_MODEL = "medpsy-4b"

# Emergency Referral Prefix
EMERGENCY_PREFIX = "⚠️ **This may be an emergency. Please go to the nearest hospital or call 108 immediately.**\n\n"

# Mandatory Footer
MANDATORY_FOOTER = "*Decision support only — please consult a doctor for medical decisions.*"


# -----------------------------------------------------------------------------
# Emergency & Red-Flag Screening Logic
# -----------------------------------------------------------------------------
NEGATION_WORDS = {"no", "not", "without", "denies", "denied", "absence", "free", "never"}
NEGATION_PHRASES = ["absence of", "free of"]
CLAUSE_DELIMITER_PATTERN = re.compile(r"[.,;:!?\n]|\bbut\b", re.IGNORECASE)


def is_negated(text: str, match_start: int) -> bool:
    """Check if a red-flag match is negated in the preceding 6 words within the same clause.

    Clauses are delimited by punctuation (.,;:!?) or the coordinating conjunction 'but'.
    Negation triggers: 'no', 'not', 'without', 'denies', 'denied', 'absence of', 'free of', 'never'.
    """
    preceding = text[:match_start]
    clauses = CLAUSE_DELIMITER_PATTERN.split(preceding)
    same_clause = clauses[-1] if clauses else ""
    same_clause_lower = same_clause.lower()

    # Check multi-word negation phrases
    for phrase in NEGATION_PHRASES:
        if phrase in same_clause_lower:
            after_phrase = same_clause_lower.split(phrase)[-1]
            words_after = re.findall(r"\b\w+\b", after_phrase)
            if len(words_after) <= 6:
                return True

    # Check single-word negation tokens
    words = re.findall(r"\b\w+\b", same_clause_lower)
    preceding_words = words[-6:]
    for w in preceding_words:
        if w in NEGATION_WORDS:
            return True

    return False


def is_infant_under_3_months(text: str) -> bool:
    """Check whether the text refers to an infant / newborn / baby under 3 months."""
    lower = text.lower()
    if re.search(r"\b(newborn|neonate)s?\b", lower):
        return True
    if re.search(r"\b(baby|infant|child)\s+under\s+(?:3|three)\s+months?\b", lower):
        return True
    # 'baby 1 month old', 'infant 2 months old', 'baby 3 weeks old', 'baby 1 month'
    if re.search(r"\b(baby|infant)\s+(?:is\s+)?(?:[0-2]\s*months?|[1-9]\s*weeks?|[0-9]+\s*days?|1\s*month|2\s*months?)\b", lower):
        return True
    # '1 month old baby', '3-week-old infant'
    if re.search(r"\b(?:[0-2]\s*months?|[1-9]\s*weeks?|[0-9]+\s*days?|1\s*month|2\s*months?)[\s-]old\s+(?:baby|infant)\b", lower):
        return True
    return False


def check_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """Screen user query for acute red-flag medical emergencies.

    Monitored specific red flags:
    1. 'chest pain'
    2. 'difficulty breathing', 'can't breathe', 'cannot breathe', 'severe breathlessness', 'respiratory distress'
    3. 'unconscious', 'not responding', 'unresponsive'
    4. 'seizure', 'fits', 'convulsion'
    5. 'heavy bleeding', 'vomiting blood', 'coughing blood'
    6. Stroke signs: 'face drooping', 'slurred speech', 'sudden weakness on one side'
    7. 'severe allergic reaction', 'swelling of the face', 'swelling of the throat', 'anaphylaxis'
    8. 'suicidal', 'wants to die', 'kill himself', 'kill herself'
    9. 'poisoning', 'swallowed poison', 'overdose'
    10. Pregnancy / pregnant + (bleeding OR severe headache OR headache OR fits OR severe abdominal pain OR swelling with headache)
    11. Infant / newborn / baby under 3 months + fever

    Plain 'fever', 'headache', 'pain', 'cough' alone must NOT trigger.
    Ignores a red-flag phrase if a negation word appears within the preceding 6 words in the same clause.
    """
    if not text or not text.strip():
        return False, None

    lower_text = text.lower()

    # 1. Pregnancy emergency check
    is_pregnant = bool(re.search(r"\b(pregnan(?:t|cy)|gravida|in\s+labor|trimester)\b", lower_text))
    if is_pregnant:
        # Swelling with headache
        has_swelling = bool(re.search(r"\bswelling\b", lower_text))
        headache_matches = list(re.finditer(r"\b(?:severe\s+)?headache\b", lower_text))
        if has_swelling and headache_matches:
            if any(not is_negated(text, m.start()) for m in headache_matches):
                return True, "Pregnancy with swelling and headache"

        preg_complications = [
            ("Pregnancy with bleeding", r"\bbleeding\b"),
            ("Pregnancy with severe headache", r"\bsevere\s+headache\b"),
            ("Pregnancy with headache", r"\bheadache\b"),
            ("Pregnancy with fits", r"\b(fits?|convulsions?)\b"),
            ("Pregnancy with severe abdominal pain", r"\bsevere\s+abdominal\s+pain\b"),
        ]
        for label, pat in preg_complications:
            for m in re.finditer(pat, lower_text):
                if not is_negated(text, m.start()):
                    return True, label

    # 2. Infant / newborn / baby under 3 months + fever
    if is_infant_under_3_months(text):
        fever_matches = list(re.finditer(r"\bfever\b", lower_text))
        if fever_matches and any(not is_negated(text, m.start()) for m in fever_matches):
            return True, "Infant under 3 months with fever"

    # 3. Specific emergency phrases
    red_flag_patterns = [
        ("Chest pain", r"\bchest\s+pain\b"),
        ("Difficulty breathing", r"\b(difficulty\s+breathing|can'?t\s+breathe|cannot\s+breathe|severe\s+breathlessness|respiratory\s+distress)\b"),
        ("Unconsciousness", r"\b(unconscious|not\s+responding|unresponsive)\b"),
        ("Seizure", r"\b(seizures?|fits?|convulsions?)\b"),
        ("Heavy bleeding", r"\b(heavy\s+bleeding|vomiting\s+blood|coughing\s+(?:up\s+)?blood)\b"),
        ("Stroke signs", r"\b(face\s+drooping|facial\s+droop|slurred\s+speech|sudden\s+weakness\s+on\s+one\s+side)\b"),
        ("Severe allergic reaction", r"\b(severe\s+allergic\s+reaction|swelling\s+of\s+(?:the\s+)?face|swelling\s+of\s+(?:the\s+)?throat|anaphylaxis)\b"),
        ("Suicidal thoughts", r"\b(suicidal|wants?\s+to\s+die|kill\s+himself|kill\s+herself|kill\s+myself)\b"),
        ("Poisoning", r"\b(poisoning|swallowed\s+poison|overdose)\b"),
    ]

    for label, pat in red_flag_patterns:
        for m in re.finditer(pat, lower_text):
            if not is_negated(text, m.start()):
                return True, label

    return False, None


# -----------------------------------------------------------------------------
# Abstention & Safety Logic
# -----------------------------------------------------------------------------
def evaluate_abstention(
    truncated: bool,
    model_output: str,
    top_rerank_score: float,
    rerank_threshold_p20: float,
) -> Tuple[bool, Optional[str]]:
    """Determine whether to abstain based on safety and confidence criteria.

    Abstains if ANY:
    1. Generation truncated (thinking budget of 1024 tokens exhausted)
    2. Output explicitly contains 'INSUFFICIENT INFORMATION'
    3. Top rerank score falls below the validation 20th percentile threshold
    """
    if truncated:
        return True, "Generation truncated (thinking budget of 1024 tokens exhausted)"

    if "INSUFFICIENT INFORMATION" in model_output.upper():
        return True, "Model identified insufficient evidence ('INSUFFICIENT INFORMATION')"

    if top_rerank_score < rerank_threshold_p20:
        return True, (
            f"Retrieved passage relevance score ({top_rerank_score:.2f}) is below "
            f"the 20th percentile validation threshold ({rerank_threshold_p20:.2f})"
        )

    return False, None


def strip_thinking(text: str) -> str:
    """Remove hidden thinking text enclosed in <think> tags."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    clean = re.sub(r"</?think>", "", clean, flags=re.IGNORECASE).strip()
    return clean


# -----------------------------------------------------------------------------
# Path Discovery & Dynamic Validation Rerank Threshold
# -----------------------------------------------------------------------------
def find_phase3_dir() -> Path:
    """Locate phase3 build artifacts on Kaggle or locally."""
    hits = glob.glob("/kaggle/input/**/phase3/dense.faiss", recursive=True)
    if hits:
        return Path(hits[0]).parent
    hits = glob.glob("/kaggle/input/**/work/phase3/dense.faiss", recursive=True)
    if hits:
        return Path(hits[0]).parent

    local_p3 = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase3"
    if (local_p3 / "dense.faiss").exists():
        return local_p3

    alt_p3 = REPO_ROOT / "work" / "phase3"
    if (alt_p3 / "dense.faiss").exists():
        return alt_p3

    raise FileNotFoundError(
        "Could not find phase3 artifacts (dense.faiss, bm25, chunks.jsonl). "
        "On Kaggle, ensure 'medrag-build' is attached: Add Input -> Your Work -> medrag-build."
    )


def compute_validation_rerank_threshold_p20(evidence_path: Optional[Path] = None) -> float:
    """Compute the 20th percentile of top rerank scores on validation questions."""
    candidates = []
    if evidence_path and evidence_path.exists():
        candidates.append(evidence_path)

    kaggle_hits = glob.glob("/kaggle/input/**/evidence.jsonl", recursive=True)
    for h in kaggle_hits:
        candidates.append(Path(h))

    candidates.extend([
        REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase7" / "evidence.jsonl",
        REPO_ROOT / "work" / "phase7" / "evidence.jsonl",
    ])

    val_top_scores: List[float] = []
    for cand in candidates:
        if cand.exists():
            print(f"  Reading validation evidence from {cand}...")
            try:
                records = read_jsonl(cand)
                for r in records:
                    if r.get("split") == "validation":
                        passages = r.get("passages", [])
                        if passages and "rerank_score" in passages[0]:
                            val_top_scores.append(float(passages[0]["rerank_score"]))
                if val_top_scores:
                    break
            except Exception as e:
                print(f"  Warning reading {cand}: {e}")

    if val_top_scores:
        p20 = float(np.percentile(val_top_scores, 20))
        p20_rounded = round(p20, 2)
        print(f"  Computed validation rerank threshold (P20): {p20_rounded} (from {len(val_top_scores)} questions)")
        return p20_rounded

    fallback_p20 = 1.85
    print(f"  Validation evidence file not found; using calibrated default P20 threshold: {fallback_p20}")
    return fallback_p20


# -----------------------------------------------------------------------------
# Global Cached Retrieval & Generation State
# -----------------------------------------------------------------------------
def print_gpu_memory(label: str = "") -> None:
    """Print per-GPU memory using torch.cuda.memory_allocated and torch.cuda.mem_get_info for cuda:0 and cuda:1."""
    if not torch.cuda.is_available():
        print(f"[Memory] {label} - CUDA not available (CPU mode)", flush=True)
        return
    n_dev = torch.cuda.device_count()
    print(f"[Memory] {label}:", flush=True)
    for dev_idx in [0, 1]:
        dev_name = f"cuda:{dev_idx}"
        if dev_idx < n_dev:
            alloc_b = torch.cuda.memory_allocated(dev_name)
            free_b, total_b = torch.cuda.mem_get_info(dev_name)
            alloc_mb = alloc_b / (1024 ** 2)
            free_mb = free_b / (1024 ** 2)
            total_mb = total_b / (1024 ** 2)
            used_mb = total_mb - free_mb
            print(
                f"  {dev_name}: torch.cuda.memory_allocated = {alloc_mb:.1f} MB | "
                f"torch.cuda.mem_get_info = free {free_mb:.1f} MB / total {total_mb:.1f} MB (used: {used_mb:.1f} MB)",
                flush=True,
            )
        else:
            print(f"  {dev_name}: Not available (system has {n_dev} GPU(s))", flush=True)


def get_device_placement() -> Tuple[str, str]:
    """Determine explicit device placement for LLM and retrieval encoders.

    Rules:
    - MedPsy-4B (and the qwen3-1.7b fallback): cuda:0 ONLY (or cpu if no GPU).
    - MedCPT query encoder and MedCPT cross-encoder: cuda:1 if available (else CPU).
    - If only one GPU exists, put encoders on CPU.
    - FAISS index: CPU only (faiss.read_index, never index_cpu_to_gpu, never StandardGpuResources).
    """
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    llm_device = "cuda:0"
    encoder_device = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
    return llm_device, encoder_device


_CACHED_RETRIEVAL: Dict[str, Any] = {}
_CACHED_MODELS: Dict[str, LLM] = {}


def get_retrieval_system(cfg_path: Optional[str] = None, device: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache retrieval models, indexes, chunks, and threshold once at startup.

    Encoders are placed on `device` (cuda:1 if multiple GPUs exist, else cpu; NEVER on cuda:0).
    FAISS index is loaded on CPU with faiss.read_index and NEVER moved to GPU.
    """
    global _CACHED_RETRIEVAL
    if _CACHED_RETRIEVAL:
        return _CACHED_RETRIEVAL

    print("\n[Init] Initializing retrieval components and clinical index...", flush=True)
    t0 = time.time()

    if device is None:
        _, encoder_device = get_device_placement()
    else:
        encoder_device = device

    p3_dir = find_phase3_dir()
    print(f"  Using phase3 directory: {p3_dir}")

    config_file = cfg_path or (REPO_ROOT / "configs" / "base.yaml")
    cfg = load_config(config_file)

    # 1. Load Chunks
    chunks_file = p3_dir / "chunks.jsonl"
    print(f"  Loading textbook chunks from {chunks_file.name}...")
    chunks = read_jsonl(chunks_file)
    print(f"  Loaded {len(chunks):,} textbook chunks.")
    print_gpu_memory("After loading textbook chunks (CPU)")

    # 2. Load FAISS Dense Index (Strictly on CPU: faiss.read_index, never index_cpu_to_gpu, never StandardGpuResources)
    faiss_file = p3_dir / "dense.faiss"
    print(f"  Loading FAISS index on CPU from {faiss_file.name}...")
    faiss_index = faiss.read_index(str(faiss_file))
    print(f"  Loaded FAISS CPU index with {faiss_index.ntotal:,} vectors.")
    print_gpu_memory("After loading FAISS index (CPU)")

    # 3. Load BM25 Sparse Index
    bm25_dir = p3_dir / "bm25"
    print(f"  Loading BM25 index from {bm25_dir.name}...")
    bm25_index = bm25s.BM25.load(str(bm25_dir))
    stemmer = Stemmer.Stemmer("english")
    print_gpu_memory("After loading BM25 index (CPU)")

    # 4. Encoders (placed on cuda:1 if multiple GPUs, else cpu; NEVER on cuda:0)
    print(f"  Loading MedCPT query encoder on {encoder_device}...")
    query_enc = encoders.query_encoder(cfg, encoder_device)
    print_gpu_memory(f"After loading query encoder on {encoder_device}")

    print(f"  Loading MedCPT cross-encoder reranker on {encoder_device}...")
    cross_enc = encoders.cross_encoder(cfg, encoder_device)
    print_gpu_memory(f"After loading cross-encoder on {encoder_device}")

    # 5. Validation Rerank Threshold (P20)
    p20_threshold = compute_validation_rerank_threshold_p20()

    _CACHED_RETRIEVAL = {
        "cfg": cfg,
        "device": encoder_device,
        "chunks": chunks,
        "faiss_index": faiss_index,
        "bm25_index": bm25_index,
        "stemmer": stemmer,
        "query_enc": query_enc,
        "cross_enc": cross_enc,
        "rerank_p20": p20_threshold,
    }
    print(f"[Init] Retrieval components successfully cached in {time.time() - t0:.1f}s!\n", flush=True)
    print_gpu_memory("After completing retrieval system init")
    return _CACHED_RETRIEVAL


def get_llm(model_key: str, device: Optional[str] = None) -> LLM:
    """Retrieve or load the requested Language Model (cuda:0 ONLY, or cpu if no GPU)."""
    global _CACHED_MODELS
    clean_key = "qwen3-1.7b" if "1.7b" in model_key.lower() else "medpsy-4b"

    if clean_key in _CACHED_MODELS:
        return _CACHED_MODELS[clean_key]

    target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[Model] Loading SLM '{clean_key}' on {target_device}...", flush=True)
    t0 = time.time()
    mcfg = MODEL_CONFIGS[clean_key]
    llm = LLM(mcfg, target_device)
    _CACHED_MODELS[clean_key] = llm
    print(f"[Model] '{clean_key}' loaded in {time.time() - t0:.1f}s!\n", flush=True)
    print_gpu_memory(f"After loading LLM '{clean_key}' on {target_device}")
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
                "chunk_id": ch.get("chunk_id", f"chunk_{c['idx']}"),
                "title": ch.get("title", ch.get("source", "Medical Textbook")),
                "source": ch.get("source", "Medical Textbook"),
                "text": ch.get("text", ""),
                "rerank_score": round(float(scores[j]), 4),
            }
        )

    retrieval_sec = time.perf_counter() - t0
    return top_passages, retrieval_sec


# -----------------------------------------------------------------------------
# Core Chatbot Pipeline (Streaming Generator)
# -----------------------------------------------------------------------------
def chat_pipeline(
    question: str,
    opt_a: str,
    opt_b: str,
    opt_c: str,
    opt_d: str,
    model_choice: str,
    history: List[Tuple[str, str]],
) -> Iterator[Tuple[List[Tuple[str, str]], str, str, str, str, str]]:
    """Execute complete clinical QA dialogue turn with streaming progress."""
    if not question or not question.strip():
        yield history, "", opt_a, opt_b, opt_c, opt_d
        return

    user_q = question.strip()

    # Parse options if provided
    options: Dict[str, str] = {}
    if opt_a and opt_a.strip():
        options["A"] = opt_a.strip()
    if opt_b and opt_b.strip():
        options["B"] = opt_b.strip()
    if opt_c and opt_c.strip():
        options["C"] = opt_c.strip()
    if opt_d and opt_d.strip():
        options["D"] = opt_d.strip()

    is_mcq = len(options) >= 2

    if is_mcq:
        opt_summary = " &nbsp;|&nbsp; ".join(f"**{k}:** {v}" for k, v in sorted(options.items()))
        user_display = f"**{user_q}**\n\n<small>{opt_summary}</small>"
    else:
        user_display = user_q

    # Yield intermediate state: "Thinking... (about 20-40 s)"
    interim_history = history + [(user_display, "🩺 Thinking… (about 20–40 s)")]
    yield interim_history, "", opt_a, opt_b, opt_c, opt_d

    t_total_start = time.perf_counter()

    # 1. Emergency Red-Flag Screening
    is_emergency, emergency_reason = check_emergency(user_q)
    emergency_banner = EMERGENCY_PREFIX if is_emergency else ""

    # 2. Hybrid Retrieval & Reranking
    retr = get_retrieval_system()
    rerank_p20 = retr.get("rerank_p20", 1.85)

    q_dict = {"question": user_q, "options": options}
    search_query = build_query(q_dict, include_options=is_mcq)
    top_passages, retrieval_sec = retrieve_top_passages(search_query, top_k=5)
    top_rerank_score = top_passages[0]["rerank_score"] if top_passages else -999.0

    # 3. Model Inference
    t_gen_start = time.perf_counter()
    clean_model_key = "qwen3-1.7b" if "1.7b" in model_choice.lower() else "medpsy-4b"
    llm = get_llm(clean_model_key)

    is_reasoning_model = bool(llm.is_reasoning)
    truncated = False
    raw_model_output = ""
    parsed_answer = None

    if is_mcq:
        # Multiple-Choice Exam Question Mode
        letters = sorted(options.keys())
        if is_reasoning_model:
            r_prompt = reasoning_rag_prompt(q_dict, top_passages, max_chars=1200)
            res = llm.generate_reasoning([r_prompt], [letters], max_new_tokens=1024)[0]
            truncated = bool(res.get("truncated", False))
            raw_model_output = res.get("raw_output", "")
            parsed_answer = res.get("_pred") or parse_answer(raw_model_output, options)
        else:
            r_prompt = rag_prompt(q_dict, top_passages, max_chars=1200)
            res = llm.generate_safe([r_prompt], [letters], max_new_tokens=40)[0]
            raw_model_output = res.get("raw_output", "")
            parsed_answer = parse_answer(raw_model_output, options)
    else:
        # Community Health Worker Plain Question Mode
        ev_lines = []
        for i, p in enumerate(top_passages, 1):
            src_name = p.get("title", p.get("source", f"Textbook {i}"))
            snippet = p.get("text", "")[:1000]
            ev_lines.append(f"[{i}] ({src_name})\n{snippet}")
        ev_text = "\n\n".join(ev_lines)

        hw_prompt = (
            "You are a clinical decision support assistant for community health workers. "
            "Answer the following question in simple, clear language using the provided medical textbook evidence.\n"
            "Format your answer in at most 6 short bullet points.\n"
            "Cite supporting evidence using bracketed numbers like [1], [2] where applicable.\n"
            "If the passages and your knowledge are not enough to answer confidently and safely, "
            "say exactly 'INSUFFICIENT INFORMATION'.\n\n"
            f"Evidence:\n{ev_text}\n\n"
            f"Health Worker Question: {user_q}\n\n"
            "Answer:"
        )

        if is_reasoning_model:
            # MedPsy-4B reasoning generation
            wrapped = llm.wrap_reasoning(hw_prompt)
            enc = llm.tok([wrapped], return_tensors="pt", padding=True, add_special_tokens=False)
            first_dev = next(llm.model.parameters()).device
            enc = {k: v.to(first_dev) for k, v in enc.items()}

            from transformers import GenerationConfig
            gcfg = GenerationConfig(
                max_new_tokens=1024,
                do_sample=False,
                eos_token_id=sorted(llm.eos),
                pad_token_id=llm.pad_id,
                return_dict_in_generate=True,
            )
            with torch.inference_mode():
                out = llm.model.generate(**enc, generation_config=gcfg)

            seqs = out.sequences if hasattr(out, "sequences") else out
            prompt_len = enc["input_ids"].shape[1]
            all_gen_ids = seqs[0, prompt_len:].tolist()
            cut = next((i for i, t in enumerate(all_gen_ids) if t in llm.eos), len(all_gen_ids))
            gen_ids = all_gen_ids[:cut]
            raw_model_output = llm.tok.decode(gen_ids, skip_special_tokens=True).strip()

            # Truncation: exhausted the 1024 budget
            truncated = (cut == len(all_gen_ids) and cut >= 1024)
        else:
            # Qwen3-1.7B fast generation
            msgs = [{"role": "user", "content": hw_prompt}]
            prompt_text = llm.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = llm.tok([prompt_text], return_tensors="pt")
            first_dev = next(llm.model.parameters()).device
            enc = {k: v.to(first_dev) for k, v in enc.items()}

            with torch.inference_mode():
                out_tokens = llm.model.generate(
                    **enc,
                    max_new_tokens=250,
                    do_sample=False,
                    pad_token_id=llm.pad_id,
                    eos_token_id=sorted(llm.eos),
                )
            gen_ids = out_tokens[0, enc["input_ids"].shape[1]:].tolist()
            raw_model_output = llm.tok.decode(gen_ids, skip_special_tokens=True).strip()
            truncated = False

    gen_sec = time.perf_counter() - t_gen_start
    total_sec = time.perf_counter() - t_total_start

    # Clean display text (strip hidden think blocks)
    clean_output = strip_thinking(raw_model_output)

    # 4. Abstention Evaluation
    should_abstain, abstain_reason = evaluate_abstention(
        truncated=truncated,
        model_output=clean_output,
        top_rerank_score=top_rerank_score,
        rerank_threshold_p20=rerank_p20,
    )

    # 5. Build Content
    if should_abstain:
        content_body = f"🛡️ **Abstention Advisory:** {ABSTAIN_MESSAGE}\n\n*(Criteria: {abstain_reason})*"
    elif is_mcq:
        chosen_opt = parsed_answer if parsed_answer in options else (letters[0] if letters else "A")
        chosen_text = options.get(chosen_opt, "")
        content_body = f"**Answer:** **Option {chosen_opt}** — {chosen_text}\n\n{clean_output}"
    else:
        content_body = clean_output

    # 6. Format References Section
    cited_nums = parse_citations(clean_output, len(top_passages))
    ref_passages = [p for p in top_passages if p["rank"] in cited_nums] if cited_nums else top_passages

    ref_items = []
    for p in ref_passages:
        r = p["rank"]
        src = p.get("title", p.get("source", "Medical Textbook"))
        text_snippet = p.get("text", "").strip()
        first_200 = text_snippet[:200].replace("\n", " ") + ("..." if len(text_snippet) > 200 else "")
        ref_items.append(
            f"- **[{r}] {src}**: {first_200}\n"
            f"  <details><summary>Expand full passage</summary>\n\n  > {text_snippet}\n  </details>"
        )
    references_block = "### 📚 References:\n" + "\n".join(ref_items)

    # 7. Final Response Assembly
    timing_line = f"⏱️ *Response time: {total_sec:.1f}s (Retrieval: {retrieval_sec:.2f}s | Generation: {gen_sec:.2f}s | Model: {clean_model_key})*"

    final_bot_reply = (
        f"{emergency_banner}"
        f"{content_body}\n\n"
        f"---\n"
        f"{references_block}\n\n"
        f"---\n"
        f"{timing_line}\n\n"
        f"{MANDATORY_FOOTER}"
    )

    final_history = history + [(user_display, final_bot_reply)]
    yield final_history, "", opt_a, opt_b, opt_c, opt_d


# -----------------------------------------------------------------------------
# Gradio UI Construction
# -----------------------------------------------------------------------------
def build_interface() -> Tuple[gr.Blocks, str]:
    """Build the clean Gradio interactive web application for community health workers."""
    custom_css = """
    .gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .chat-header { margin-bottom: 0.8rem; }
    """

    with gr.Blocks(title="MedRAG-SLM Community Health Worker Assistant") as demo:
        gr.Markdown(
            "# 🩺 MedRAG-SLM: Community Health Worker Clinical Assistant\n"
            "**Authoritative Medical Decision Support for Primary Care & Field Health Workers.**\n"
            "*Powered by MedPsy-4B reasoning & MedCPT textbook retrieval with active emergency screening and abstention safety.*"
        )

        chatbot = gr.Chatbot(label="Community Health Worker Dialogue & Clinical Guidance", height=500)

        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Patient Case / Field Health Inquiry",
                    placeholder="Type patient symptoms, maternal checks, child illness, or dosing inquiry in plain English...",
                    lines=3,
                )

                with gr.Accordion("Multiple-choice options (A–D) [Optional — for exam-style questions]", open=False):
                    with gr.Row():
                        opt_a = gr.Textbox(label="Option A", placeholder="e.g. Left anterior descending artery")
                        opt_b = gr.Textbox(label="Option B", placeholder="e.g. Right coronary artery")
                    with gr.Row():
                        opt_c = gr.Textbox(label="Option C", placeholder="e.g. Left circumflex artery")
                        opt_d = gr.Textbox(label="Option D", placeholder="e.g. Left main coronary artery")

            with gr.Column(scale=2):
                model_choice = gr.Dropdown(
                    choices=["medpsy-4b (default)", "qwen3-1.7b (fast)"],
                    value="medpsy-4b (default)",
                    label="SLM Reasoning Engine",
                    info="MedPsy-4B provides clinically validated step-by-step thinking.",
                )
                with gr.Row():
                    submit_btn = gr.Button("🩺 Submit Question", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

        # Clickable Examples
        gr.Markdown("### 💡 Field Clinical Examples (Click to Load)")
        examples_data = [
            [
                "A child has fever for 3 days, rash and red eyes. What could it be and what should I do?",
                "", "", "", "",
            ],
            [
                "Pregnant woman 7 months with leg swelling and headache — what should I check?",
                "", "", "", "",
            ],
            [
                "What is the first-line treatment for uncomplicated malaria?",
                "", "", "", "",
            ],
            [
                "Can I give paracetamol and ibuprofen together to an adult with fever?",
                "", "", "", "",
            ],
            [
                "A 45-year-old male with hypertension presents with sudden severe crushing chest pain radiating to his left shoulder and jaw, diaphoresis, and shortness of breath. ECG reveals ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?",
                "Left anterior descending artery",
                "Right coronary artery",
                "Left circumflex artery",
                "Left main coronary artery",
            ],
        ]

        gr.Examples(
            examples=examples_data,
            inputs=[question_input, opt_a, opt_b, opt_c, opt_d],
            label="Click an example to test emergency triage, field guidance, or MCQ:",
        )

        # Wire Events
        submit_btn.click(
            fn=chat_pipeline,
            inputs=[question_input, opt_a, opt_b, opt_c, opt_d, model_choice, chatbot],
            outputs=[chatbot, question_input, opt_a, opt_b, opt_c, opt_d],
        )

        question_input.submit(
            fn=chat_pipeline,
            inputs=[question_input, opt_a, opt_b, opt_c, opt_d, model_choice, chatbot],
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
    print("Starting MedRAG-SLM Health Worker Chatbot (Kaggle GPU Session)...")
    print("=" * 70)

    llm_device, encoder_device = get_device_placement()
    print(f"[Device Placement] LLM: {llm_device} (cuda:0 only) | Encoders: {encoder_device} | FAISS: CPU")
    print_gpu_memory("Startup memory state before loading")

    # Step 1: Load LLM FIRST on cuda:0 so the model always secures its memory
    print(f"\n--- [Startup Step 1] Loading LLM ({DEFAULT_MODEL}) FIRST on {llm_device} ---", flush=True)
    get_llm(DEFAULT_MODEL, device=llm_device)

    # Step 2: Load retrieval components (CPU FAISS, encoders on cuda:1 if available, else CPU)
    print(f"\n--- [Startup Step 2] Loading retrieval components on {encoder_device} ---", flush=True)
    get_retrieval_system(device=encoder_device)

    app, custom_css = build_interface()

    print("\n" + "=" * 70)
    print("🎉 LAUNCHING GRADIO APP WITH PUBLIC LINK (share=True)...")
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
