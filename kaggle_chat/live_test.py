"""Live deployment test for the MedRAG-SLM chatbot pipeline.

Runs on Kaggle as a script kernel. Measures latency, peak VRAM, and
abstention behaviour for the MCQ and health-worker paths. Does not start Gradio.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

# On Kaggle only live_test.py is uploaded, so clone the repo for the rest of the package.
REPO = "/tmp/medrag-slm"
if os.path.exists("/kaggle/working"):
    import subprocess
    subprocess.run(["rm", "-rf", REPO], check=False)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/SatyaSaiNath1311/medrag-slm.git", REPO],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(REPO, "requirements.txt")], check=False)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
sys.path.insert(0, ".")

import torch

from kaggle_chat.chat_app import (
    CHAT_MAX_NEW_TOKENS_REASONING,
    DEFAULT_MODEL,
    evaluate_abstention,
    generate_mcq_answer,
    generate_plain_answer,
    get_device_placement,
    get_llm,
    get_retrieval_system,
    retrieve_evidence,
    strip_thinking,
)
from kaggle_chat.question_types import classify_question

SEP = "=" * 70

PLAIN_QUESTIONS = [
    "A patient has had a fever and cough for three days. What should I check first?",
    "What is the recommended storage temperature for the vaccine lot numbered XJ-4471-B?",
    "Explain everything a community health worker needs to know about managing chronic illness in a rural clinic.",
]


def run_one(llm, question, options, label):
    """Run one question end to end and print timings and abstention outcome."""
    is_mcq = options is not None
    q_dict = {"question": question, "options": options or {}}
    t0 = time.perf_counter()
    top_passages, top_rerank_score, retrieval_sec = retrieve_evidence(q_dict, is_mcq)
    if is_mcq:
        letters = sorted(options.keys())
        raw, truncated, parsed, gen_sec = generate_mcq_answer(
            llm, q_dict, top_passages, letters, use_reasoning=True
        )
    else:
        qtype = classify_question(question)
        raw, truncated, gen_sec = generate_plain_answer(
            llm, question, top_passages, qtype, use_reasoning=True
        )
        parsed = None
    total = time.perf_counter() - t0
    clean = strip_thinking(raw)
    retr = get_retrieval_system()
    p20 = retr.get("rerank_p20", 1.85)
    abstained, reason = evaluate_abstention(truncated, clean, top_rerank_score, p20)
    print(f"  [{label}] retrieval {retrieval_sec:.1f}s | generation {gen_sec:.1f}s | total {total:.1f}s")
    print(f"           truncated={truncated} rerank_top={top_rerank_score:.2f} parsed={parsed}")
    print(f"           abstained={abstained} reason={reason}")
    print(f"           answer: {clean[:200].replace(chr(10), ' ')}")
    return total


def main():
    print(SEP)
    print("MedRAG-SLM LIVE DEPLOYMENT TEST")
    print(SEP)
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Token cap: {CHAT_MAX_NEW_TOKENS_REASONING}")
    print(f"CUDA devices: {torch.cuda.device_count()}")

    llm_device, encoder_device = get_device_placement()
    print(f"LLM device: {llm_device} | Encoder device: {encoder_device}")

    llm = get_llm(DEFAULT_MODEL, device=llm_device)
    retr = get_retrieval_system(device=encoder_device)
    p20 = retr.get("rerank_p20", 1.85)
    print(f"rerank_p20 in use: {p20}  (1.85 means the hard-coded fallback was used)")

    mcq = []
    for path in ("/kaggle/working/work/phase1/test.jsonl",
                 "outputs/kaggle_build/work/phase1/test.jsonl"):
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    row = json.loads(line)
                    if row.get("dataset") == "medqa" or "options" in row:
                        mcq.append(row)
                    if len(mcq) >= 3:
                        break
            break
    print(f"Loaded {len(mcq)} MCQ questions")

    results = {}
    for i, row in enumerate(mcq, 1):
        for pass_no in ("warmup", "steady"):
            try:
                t = run_one(llm, row["question"], row["options"], f"MCQ{i} {pass_no}")
                if pass_no == "steady":
                    results[f"MCQ{i}"] = t
            except Exception:
                traceback.print_exc()

    for i, q in enumerate(PLAIN_QUESTIONS, 1):
        for pass_no in ("warmup", "steady"):
            try:
                t = run_one(llm, q, None, f"PLAIN{i} {pass_no}")
                if pass_no == "steady":
                    results[f"PLAIN{i}"] = t
            except Exception:
                traceback.print_exc()

    print("\n" + SEP)
    print("PEAK VRAM")
    for d in range(torch.cuda.device_count()):
        print(f"  cuda:{d}  {torch.cuda.max_memory_allocated(d) / 1024**3:.2f} GiB")
    print("\nSTEADY-STATE SECONDS PER QUESTION")
    for k, v in results.items():
        print(f"  {k}: {v:.1f}s")
    if results:
        print(f"  max: {max(results.values()):.1f}s")
    print(SEP)


if __name__ == "__main__":
    main()
