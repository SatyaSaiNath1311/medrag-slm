"""Kaggle runner B: question answering with the 5+1 models (Phases 4, 5, 8).
Input: output of runner A (medrag-build), attached as a notebook input.
Change MODE to "smoke" to re-run the setup smoke test instead.

PILOT mode (PILOT=True):
  Runs only medpsy-4b on a seeded 80-question validation subset (60 MedQA + 20 PubMedQA).
  Modes: baseline + context.  Single GPU (CUDA_VISIBLE_DEVICES=0).
  Output written to {WORK}/pilot/.
  A human-readable pilot_summary.txt is printed after the run.

PARALLEL_GPUS mode (PARALLEL_GPUS=True):
  Splits the model list across two GPUs by launching two subprocesses:
    - GPU 0: first half of MODELS
    - GPU 1: second half of MODELS
  Each subprocess gets CUDA_VISIBLE_DEVICES set so it sees only one T4.
  Results are merged (both write to the same WORK directory; phase dirs are model-specific
  JSONL files so there is no collision).
  Not compatible with PILOT or PROFILE modes.
"""
import glob
import json
import os
import subprocess
import sys

REPO_URL = "https://github.com/SatyaSaiNath1311/medrag-slm.git"
CODE_DIR = "/tmp/medrag-slm"
WORK = "/kaggle/working/work"
MODE = os.environ.get("MODE", "qa")        # "qa" or "smoke"
CHECK_ONLY = False
PROFILE = False
PILOT = True                # run medpsy-4b pilot
PARALLEL_GPUS = False       # split model list across 2 GPUs in parallel subprocesses
TINY = False                # set to True for tiny check on Kaggle
MODELS = ["medpsy-4b"]
MODES = ["baseline", "rag"]

# Override from env var if provided
if "CHECK_ONLY" in os.environ:
    CHECK_ONLY = os.environ["CHECK_ONLY"].lower() in ("1", "true", "yes")
if "PROFILE" in os.environ:
    PROFILE = os.environ["PROFILE"].lower() in ("1", "true", "yes")
if "PILOT" in os.environ:
    PILOT = os.environ["PILOT"].lower() in ("1", "true", "yes")
if "PARALLEL_GPUS" in os.environ:
    PARALLEL_GPUS = os.environ["PARALLEL_GPUS"].lower() in ("1", "true", "yes")
if "TINY" in os.environ:
    TINY = os.environ["TINY"].lower() in ("1", "true", "yes")
if "MODELS" in os.environ:
    MODELS = [m.strip() for m in os.environ["MODELS"].split(",") if m.strip()]
if "MODES" in os.environ:
    MODES = [m.strip() for m in os.environ["MODES"].split(",") if m.strip()]


def print_kaggle_input_configs(max_depth=7):
    base = "/kaggle/input"
    if not os.path.exists(base):
        print(f"{base} does not exist", flush=True)
        return
    print(f"Scanning {base} for config.json (max depth {max_depth}):", flush=True)
    base_depth = base.rstrip(os.path.sep).count(os.path.sep)
    found = []
    for root, dirs, files in os.walk(base):
        cur_depth = root.count(os.path.sep) - base_depth
        if cur_depth >= max_depth:
            dirs.clear()
        if "config.json" in files:
            found.append(root)
            print(f"  found config.json: {root}", flush=True)
    if not found:
        print("  no config.json found under /kaggle/input", flush=True)


print_kaggle_input_configs()


def run(cmd, extra_env=None):
    print(">>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, check=True, env=env)


def run_bg(cmd, extra_env=None):
    """Launch a subprocess in the background and return the Popen object."""
    print(">> [bg]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(cmd, env=env)


# 1. Hugging Face token from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:  # noqa: BLE001
    print("WARNING: HF_TOKEN secret not available:", e)

# 2. Fresh copy of the code
run(["rm", "-rf", CODE_DIR])
run(["git", "clone", "--depth", "1", REPO_URL, CODE_DIR])
os.chdir(CODE_DIR)
run(["git", "log", "-1", "--oneline"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

# Check for kaggle_runner/models.txt if MODELS is still empty
models_file = "kaggle_runner/models.txt"
if not MODELS and os.path.exists(models_file):
    with open(models_file) as f:
        MODELS = [line.strip() for line in f if line.strip() and not line.startswith("#")]
if MODELS:
    print(f"Selected models to run: {MODELS}")
if MODES:
    print(f"Selected modes to run: {MODES}")
if PROFILE:
    print("PROFILE mode enabled: running profiling benchmarks")
if PILOT:
    print("PILOT mode enabled: medpsy-4b pilot on 80-question val subset, single GPU")
if PARALLEL_GPUS:
    print("PARALLEL_GPUS mode enabled: splitting model list across 2 GPUs")
if TINY:
    print("TINY mode enabled: running on tiny scale")

# Save environment freeze and nvidia-smi
ENV_DIR = os.path.join(WORK, "env")
os.makedirs(ENV_DIR, exist_ok=True)
try:
    with open(os.path.join(ENV_DIR, "environment.txt"), "w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
except Exception as e:
    print(f"Warning writing pip freeze: {e}")

try:
    with open(os.path.join(ENV_DIR, "nvidia_smi.txt"), "w") as f:
        subprocess.run(["nvidia-smi"], stdout=f, check=True)
except Exception as e:
    print(f"Warning writing nvidia-smi: {e}")

if MODE == "smoke":
    run([sys.executable, "-m", "src.smoke_test", "--config", "configs/base.yaml",
         "--out", "/kaggle/working/outputs/smoke"])
    sys.exit(0)

# 3. Link runner A's output (phase1 ... phase7) into the work folder
hits = glob.glob("/kaggle/input/**/phase7/evidence.jsonl", recursive=True)
if not hits:
    sys.exit("ERROR: medrag-build output not found. Attach it: Add Input -> Your Work -> medrag-build")
build_dir = os.path.dirname(os.path.dirname(hits[0]))
print("Using build output from:", build_dir)
os.makedirs(WORK, exist_ok=True)
for n in (1, 2, 3, 6, 7):
    src_dir, dst = os.path.join(build_dir, f"phase{n}"), os.path.join(WORK, f"phase{n}")
    if os.path.isdir(src_dir) and not os.path.exists(dst):
        os.symlink(src_dir, dst)


# ── Helper to build the qa_pipeline command ────────────────────────────────────

def build_qa_cmd(models_list, modes_list, split=None, work_dir=None, check_only=False,
                 profile=False, tiny=False, skip_format_check=False):
    cmd = [sys.executable, "-m", "src.qa_pipeline",
           "--config", "configs/base.yaml",
           "--work", work_dir or WORK]
    if check_only:
        cmd.append("--check-only")
    if profile:
        cmd.append("--profile")
    if tiny:
        cmd.append("--tiny")
    if skip_format_check:
        cmd.append("--skip-format-check")
    if models_list:
        cmd += ["--models", *models_list]
    if modes_list:
        cmd += ["--modes", *modes_list]
    if split:
        cmd += ["--split", split]
    return cmd


# ── PILOT mode ─────────────────────────────────────────────────────────────────

if PILOT:
    if PARALLEL_GPUS or PROFILE:
        sys.exit("ERROR: PILOT is not compatible with PARALLEL_GPUS or PROFILE modes.")

    pilot_work = os.path.join(WORK, "pilot")
    os.makedirs(pilot_work, exist_ok=True)

    # Symlink build phases into pilot_work as well (re-use same phase1..7 data)
    for n in (1, 2, 3, 6, 7):
        src_dir = os.path.join(WORK, f"phase{n}")
        dst = os.path.join(pilot_work, f"phase{n}")
        if os.path.isdir(src_dir) and not os.path.exists(dst):
            os.symlink(src_dir, dst)

    pilot_cmd = build_qa_cmd(
        models_list=["medpsy-4b"],
        modes_list=["baseline", "context"],
        split="val_subset",
        work_dir=pilot_work,
        skip_format_check=True,   # reasoning model — skip standard parse-rate format check
    )
    print("\n=== PILOT RUN: medpsy-4b, 80-question val subset, GPU 0 ===", flush=True)
    run(pilot_cmd, extra_env={"CUDA_VISIBLE_DEVICES": "0"})

    # Print pilot summary
    print("\n=== PILOT SUMMARY ===", flush=True)
    for mode, phase in [("baseline", "phase5"), ("context", "phase10")]:
        phase_dir = os.path.join(pilot_work, phase)
        jsonl_path = os.path.join(phase_dir, "medpsy-4b.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"  [{mode}] output not found at {jsonl_path}", flush=True)
            continue
        rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
        n_total = len(rows)
        answerable = [r for r in rows if not r.get("should_abstain")]
        n_ans = len(answerable)
        n_correct = sum(1 for r in answerable if r.get("correct"))
        acc = round(n_correct / n_ans, 4) if n_ans else None
        n_parsed = sum(1 for r in rows if r.get("parsed"))
        n_truncated = sum(1 for r in rows if r.get("truncated"))
        n_refeed = sum(1 for r in rows if r.get("pred_source") == "logprob_refeed")
        medqa_rows = [r for r in answerable if r.get("dataset") == "medqa"]
        pubmedqa_rows = [r for r in answerable if r.get("dataset") == "pubmedqa"]
        acc_medqa = round(sum(r["correct"] for r in medqa_rows) / len(medqa_rows), 4) if medqa_rows else None
        acc_pubmedqa = round(sum(r["correct"] for r in pubmedqa_rows) / len(pubmedqa_rows), 4) if pubmedqa_rows else None
        summary_lines = [
            f"  [{mode}] n={n_total}, accuracy={acc} (MedQA={acc_medqa}, PubMedQA={acc_pubmedqa})",
            f"    parsed={n_parsed}/{n_total}, truncated={n_truncated}, logprob_refeed={n_refeed}",
        ]
        for line in summary_lines:
            print(line, flush=True)
    print("=== END PILOT SUMMARY ===\n", flush=True)
    sys.exit(0)


# ── PARALLEL_GPUS mode ─────────────────────────────────────────────────────────

if PARALLEL_GPUS:
    if not MODELS:
        sys.exit("ERROR: PARALLEL_GPUS requires an explicit MODELS list.")
    if PROFILE or CHECK_ONLY:
        # Profiling and check-only don't benefit from GPU split; run normally
        print("WARNING: PARALLEL_GPUS with PROFILE/CHECK_ONLY — running sequentially on all GPUs.", flush=True)
        PARALLEL_GPUS = False
    else:
        mid = len(MODELS) // 2
        models_gpu0 = MODELS[:mid] if mid > 0 else MODELS
        models_gpu1 = MODELS[mid:] if mid > 0 and mid < len(MODELS) else []

        procs = []
        if models_gpu0:
            cmd0 = build_qa_cmd(models_gpu0, MODES, work_dir=WORK, check_only=CHECK_ONLY, tiny=TINY)
            procs.append(("GPU-0", run_bg(cmd0, extra_env={"CUDA_VISIBLE_DEVICES": "0"})))
        if models_gpu1:
            cmd1 = build_qa_cmd(models_gpu1, MODES, work_dir=WORK, check_only=CHECK_ONLY, tiny=TINY)
            procs.append(("GPU-1", run_bg(cmd1, extra_env={"CUDA_VISIBLE_DEVICES": "1"})))

        print(f"\nWaiting for {len(procs)} GPU subprocess(es)...", flush=True)
        exit_codes = {}
        for label, proc in procs:
            rc = proc.wait()
            exit_codes[label] = rc
            print(f"  {label} finished with exit code {rc}", flush=True)

        any_failed = any(rc != 0 for rc in exit_codes.values())
        print("\nRUNNER B PARALLEL COMPLETE:", exit_codes)
        if any_failed:
            sys.exit(1)
        sys.exit(0)

# ── Sequential run (default) ───────────────────────────────────────────────────

cmd = build_qa_cmd(MODELS, MODES, work_dir=WORK, check_only=CHECK_ONLY, profile=PROFILE, tiny=TINY)
run(cmd)
print("\nRUNNER B COMPLETE: " + ("profiling benchmark" if PROFILE else ("phase 4 format check" if CHECK_ONLY else f"modes {MODES}")))
