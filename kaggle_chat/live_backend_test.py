"""Headless exercise of the live MedRAGBackend — no Gradio, no tunnel.

Instantiates the real backend once, runs a fixed battery of test cases through
stream(), and reports timing, abstention, truncation, and citation counts.

Usage (from repo root on Kaggle):
    python3 kaggle_chat/live_backend_test.py

Logs to outputs/deployment/live_backend_test_<YYYY-MM-DD>.log and stdout.
"""
from __future__ import annotations

import os
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
        ["git", "clone", "--depth", "1",
         "https://github.com/SatyaSaiNath1311/medrag-slm.git", REPO],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    os.path.join(REPO, "requirements.txt")], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=False)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
sys.path.insert(0, ".")

# Make kaggle_chat and src importable when run as a script.
REPO_ROOT = Path(REPO) if os.path.exists("/kaggle/working") else Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_chat.chat_ui import Reply

SEP = "=" * 70
TRUNCATION_REASON_MARKER = "Generation truncated"

# ---------------------------------------------------------------- test cases
TEST_CASES = [
    (
        "mcq",
        "A 45-year-old man presents with sudden-onset chest pain radiating to the left arm, "
        "diaphoresis, and nausea. ECG shows ST-segment elevation in leads II, III, and aVF. "
        "Which of the following is the most likely diagnosis?\n"
        "Options: (A) Unstable angina (B) Acute inferior STEMI (C) Pericarditis (D) Aortic dissection",
    ),
    (
        "plain",
        "A 3-year-old child has had a fever of 39°C for two days with mild cough and runny "
        "nose. The child is still drinking and playing. What should the community health "
        "worker check first, and when should the child be referred?",
    ),
    (
        "emergency",
        "A pregnant woman at 34 weeks is unconscious and having convulsions. She has severe "
        "headache and her blood pressure is very high. What should I do immediately?",
    ),
    (
        "unanswerable",
        "What is the recommended adult dose of Zarflinoximab for treatment-resistant "
        "autoimmune hepatic encephalopathy? Include pharmacokinetic parameters.",
    ),
    (
        "long",
        "Provide an exhaustive, step-by-step differential diagnosis for a 60-year-old male "
        "presenting with chronic fatigue, unintentional weight loss, intermittent low-grade "
        "fever, night sweats, diffuse lymphadenopathy, hepatosplenomegaly, and pancytopenia. "
        "For each condition in the differential, explain the pathophysiology in detail, list "
        "every relevant laboratory test and imaging study with expected findings, describe "
        "the full staging workup, outline first-line through third-line treatment protocols "
        "with doses, discuss prognosis stratified by stage, and provide detailed reasoning "
        "for why each diagnosis should or should not be ranked higher. Cover at least: "
        "non-Hodgkin lymphoma, Hodgkin lymphoma, chronic lymphocytic leukaemia, acute "
        "myeloid leukaemia, myelodysplastic syndrome, hairy cell leukaemia, systemic lupus "
        "erythematosus, sarcoidosis, tuberculosis, HIV/AIDS, visceral leishmaniasis, "
        "brucellosis, infective endocarditis, hemophagocytic lymphohistiocytosis, Gaucher "
        "disease, amyloidosis, and metastatic carcinoma of unknown primary.",
    ),
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


def main():
    # ---- log setup ----
    log_dir = REPO_ROOT / "outputs" / "deployment"
    log_dir.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    log_path = log_dir / f"live_backend_test_{date_stamp}.log"
    log_fh = open(log_path, "a")
    tee = _Tee(log_fh, sys.stdout)
    sys.stdout = tee
    sys.stderr = _Tee(log_fh, sys.__stderr__)

    print(SEP)
    print("MedRAG-SLM  LIVE BACKEND TEST")
    print(SEP)
    print(f"Date : {date_stamp}")
    print(f"Log  : {log_path}")

    # ---- backend construction (model load) ----
    print(f"\n[backend] loading MedRAGBackend …")
    t0 = time.perf_counter()
    from kaggle_chat.live_backend import MedRAGBackend
    backend = MedRAGBackend()
    load_sec = time.perf_counter() - t0
    print(f"[backend] ready in {load_sec:.1f}s")

    # ---- run test cases ----
    results = []  # list of dicts for the summary table

    for case_name, question in TEST_CASES:
        print(f"\n{SEP}")
        print(f"  [{case_name}] question: {question[:120]}…")
        row = {
            "case": case_name,
            "elapsed": 0.0,
            "status": "error",
            "n_references": 0,
            "reason": "",
            "error": False,
        }
        try:
            t_start = time.perf_counter()
            yielded = []
            for item in backend.stream(question, []):
                yielded.append(item)
                if not isinstance(item, Reply):
                    print(f"           progress: {item}")
            elapsed = time.perf_counter() - t_start
            row["elapsed"] = elapsed

            if not yielded or not isinstance(yielded[-1], Reply):
                print(f"  [{case_name}] ERROR: final stream item is not a Reply")
                row["error"] = True
                row["status"] = "error"
            else:
                final = yielded[-1]
                status = final.status
                n_refs = len(final.references) if final.references else 0
                if status == "abstained":
                    reason = final.text or ""
                elif status == "emergency" and "(Criteria: " in (final.text or ""):
                    reason = (final.text or "").split("(Criteria: ", 1)[1].rstrip(")*")
                else:
                    reason = ""

                row["status"] = status
                row["n_references"] = n_refs
                row["reason"] = reason

                reply_body = (final.text or "").replace("\n", " ")[:300]
                print(f"  [{case_name}] elapsed {elapsed:.1f}s")
                print(f"           status={status} n_references={n_refs} reason={reason}")
                print(f"           type_label={final.type_label}")
                if final.emergency_reason:
                    print(f"           emergency_reason={final.emergency_reason[:200]}")
                print(f"           reply_body: {reply_body}")

        except Exception:
            row["error"] = True
            row["status"] = "error"
            row["elapsed"] = time.perf_counter() - t_start
            print(f"  [{case_name}] EXCEPTION after {row['elapsed']:.1f}s:")
            traceback.print_exc()

        results.append(row)

    # ---- summary table ----
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    header = f"{'Case':<15} {'Elapsed':>8} {'Status':<12} {'Refs':>6}  {'Reason'}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['case']:<15} {r['elapsed']:>7.1f}s {r['status']:<12} {r['n_references']:>6}  {r['reason']}")

    # ---- truncation path verdict ----
    long_row = next((r for r in results if r["case"] == "long"), None)
    truncation_fired = (
        long_row is not None
        and not long_row["error"]
        and TRUNCATION_REASON_MARKER in long_row["reason"]
    )
    print(f"\nTRUNCATION PATH FIRED: {'yes' if truncation_fired else 'no'}")

    print(f"\n[backend] model load time: {load_sec:.1f}s")
    print(SEP)

    # ---- cleanup ----
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_fh.close()


if __name__ == "__main__":
    main()

