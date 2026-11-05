"""Measure citation marker frequency in the live plain-answer pipeline.

Runs 30 realistic health-worker questions through MedRAGBackend to determine
how often the model emits citation markers [n], to evaluate whether a
zero-citation abstention rule is viable.

Usage (from repo root on Kaggle):
    python3 kaggle_chat/citation_rate_check.py

Logs to outputs/deployment/citation_rate_<YYYY-MM-DD>.log and stdout.
"""
from __future__ import annotations

import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# On Kaggle only live_test.py is uploaded, so clone the repo for the rest of the package.
REPO = "/tmp/medrag-slm"
if os.path.exists("/kaggle/working"):
    import subprocess
    subprocess.run(["rm", "-rf", REPO], check=False)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", "probe/citation-format",
         "https://github.com/SatyaSaiNath1311/medrag-slm.git", REPO],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(REPO, "requirements.txt")], check=False)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
sys.path.insert(0, ".")

# Make kaggle_chat and src importable when run as a script.
REPO_ROOT = Path(REPO) if os.path.exists("/kaggle/working") else Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEP = "=" * 70

# ---------------------------------------------------------------- questions
QUESTIONS = [
    # Symptoms
    "A 35-year-old man has had a persistent dry cough and mild fever for four days. What could it be?",

    # Treatment / care
    "How should a health worker manage mild dehydration from diarrhoea in an adult at home?",

    # Medicine safety
    "Can a 45-year-old adult take paracetamol and ibuprofen together for joint pain?",

    # Pregnancy / child health
    "A pregnant woman in her second trimester has mild leg swelling after standing. What should be checked?",

    # General information
    "What is hypertension and how does high blood pressure affect long-term health?",
]


class _Tee:
    """Write to both a file and the original stdout."""
    def __init__(self, log_fh, stream):
        self._log = log_fh
        self._stream = stream

    def write(self, s):
        self._stream.write(s)
        self._log.write(s)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._stream.fileno()

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def main():
    # ---- log setup ----
    log_dir = REPO_ROOT / "outputs" / "deployment"
    log_dir.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    log_path = log_dir / f"citation_rate_{date_stamp}.log"
    log_fh = open(log_path, "a")
    tee = _Tee(log_fh, sys.stdout)
    sys.stdout = tee
    sys.stderr = _Tee(log_fh, sys.__stderr__)

    print(SEP)
    print("MedRAG-SLM  CITATION RATE CHECK")
    print(SEP)
    print(f"Date : {date_stamp}")
    print(f"Log  : {log_path}")
    print(f"Total Questions : {len(QUESTIONS)}")

    # ---- backend construction (model load) ----
    print(f"\n[backend] loading MedRAGBackend …")
    t0 = time.perf_counter()
    from kaggle_chat.live_backend import MedRAGBackend
    backend = MedRAGBackend()
    load_sec = time.perf_counter() - t0
    print(f"[backend] ready in {load_sec:.1f}s")

    # ---- run questions ----
    results = []
    marker_pattern = re.compile(r"\[\d+\]")

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n{SEP}")
        print(f"  [{i:02d}/{len(QUESTIONS)}] question: {question[:120]}…")
        t_start = time.perf_counter()
        try:
            yielded = []
            for item in backend.stream(question, []):
                yielded.append(item)
                if isinstance(item, str):
                    print(f"           progress: {item}")
            elapsed = time.perf_counter() - t_start

            if not yielded or isinstance(yielded[-1], str) or not hasattr(yielded[-1], "status"):
                print(f"  [{i:02d}/30] ERROR: final stream item is not a reply object")
                record = {
                    "index": i,
                    "question": question,
                    "type_label": "Error",
                    "status": "error",
                    "elapsed": elapsed,
                    "n_references": 0,
                    "has_marker": False,
                    "distinct_markers_count": 0,
                    "markers": [],
                    "error": True,
                }
            else:
                final = yielded[-1]
                status = getattr(final, "status", "unknown")
                type_label = getattr(final, "type_label", "") or "Unknown"
                refs = getattr(final, "references", []) or []
                n_refs = len(refs)
                body = getattr(final, "text", "") or ""
                markers = marker_pattern.findall(body)
                has_marker = len(markers) > 0
                n_distinct = len(set(markers))

                record = {
                    "index": i,
                    "question": question,
                    "type_label": type_label,
                    "status": status,
                    "elapsed": elapsed,
                    "n_references": n_refs,
                    "has_marker": has_marker,
                    "distinct_markers_count": n_distinct,
                    "markers": sorted(set(markers)),
                    "error": False,
                }

                reply_preview = body.replace("\n", " ")[:300]
                print(f"  [{i:02d}/30] elapsed {elapsed:.1f}s")
                print(f"           type_label={type_label} status={status} refs={n_refs}")
                print(f"           has_marker={'yes' if has_marker else 'no'} distinct_markers={n_distinct} markers={record['markers']}")
                print(f"           reply_body: {reply_preview}")

        except Exception:
            elapsed = time.perf_counter() - t_start
            record = {
                "index": i,
                "question": question,
                "type_label": "Error",
                "status": "error",
                "elapsed": elapsed,
                "n_references": 0,
                "has_marker": False,
                "distinct_markers_count": 0,
                "markers": [],
                "error": True,
            }
            print(f"  [{i:02d}/30] EXCEPTION after {elapsed:.1f}s:")
            traceback.print_exc()

        results.append(record)

    # ---- summary ----
    total_q = len(results)
    answered_q = [r for r in results if r["status"] == "answered"]
    n_answered = len(answered_q)
    cited_q = [r for r in answered_q if r["has_marker"]]
    n_cited = len(cited_q)
    citation_rate = (n_cited / n_answered * 100) if n_answered > 0 else 0.0

    print(f"\n{SEP}")
    print("CITATION RATE SUMMARY")
    print(SEP)
    print(f"Total questions               : {total_q}")
    print(f"Total answered                : {n_answered}")
    print(f"Answered with [n] citation    : {n_cited}")
    print(f"Overall Citation Rate         : {citation_rate:.1f}% ({n_cited}/{n_answered})")

    # ---- breakdown by type label ----
    print(f"\n{SEP}")
    print("BREAKDOWN BY QUESTION TYPE")
    print(SEP)
    header = f"{'Type Label':<28} {'Total':>6} {'Answered':>9} {'Cited':>7} {'Citation Rate':>14}"
    print(header)
    print("-" * len(header))

    # Collect distinct type labels in encounter order
    type_labels = []
    for r in results:
        tl = r["type_label"]
        if tl != "Error" and tl not in type_labels:
            type_labels.append(tl)

    for tl in type_labels:
        subset = [r for r in results if r["type_label"] == tl]
        s_total = len(subset)
        s_ans = [r for r in subset if r["status"] == "answered"]
        s_n_ans = len(s_ans)
        s_cited = [r for r in s_ans if r["has_marker"]]
        s_n_cited = len(s_cited)
        s_rate = (s_n_cited / s_n_ans * 100) if s_n_ans > 0 else 0.0
        print(f"{tl:<28} {s_total:>6} {s_n_ans:>9} {s_n_cited:>7} {s_rate:>13.1f}%")

    print("-" * len(header))
    print(f"{'OVERALL':<28} {total_q:>6} {n_answered:>9} {n_cited:>7} {citation_rate:>13.1f}%")
    print(f"\n[backend] model load time: {load_sec:.1f}s")
    print(SEP)

    # ---- cleanup ----
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_fh.close()


if __name__ == "__main__":
    main()

