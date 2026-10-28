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
import atexit
import glob
import json
import os
import subprocess
import sys
import time

KERNEL_START_TIME = float(os.environ.get("KERNEL_START_TIME", time.time()))
if "KERNEL_START_TIME" not in os.environ:
    os.environ["KERNEL_START_TIME"] = str(KERNEL_START_TIME)
SCRIPT_PATH = os.path.abspath(__file__)

REPO_URL = "https://github.com/SatyaSaiNath1311/medrag-slm.git"
CODE_DIR = "/tmp/medrag-slm"
WORK = "/kaggle/working/work"
# ══════════════════════════════════════════════════════════════════════════════
# RUNNER CONFIGURATION CONSTANTS (Set desired mode before pushing to Kaggle)
# ══════════════════════════════════════════════════════════════════════════════
MODE = "qa"                     # "qa" or "smoke"
CHECK_ONLY = False
PROFILE = False
PILOT = False                   # run medpsy-4b pilot (80 val-subset questions, single GPU)
PARALLEL_GPUS = False           # split model list across 2 GPUs in parallel subprocesses
TINY = False                    # set to True for tiny check on Kaggle
MODELS = ["medpsy-4b"]
MODES = ["baseline", "rag"]

# Prior completed run: RAG_MEDPSY = True

# ── Prior / Completed Modes ───────────────────────────────────────────────────
FINAL_MEDPSY = False            # medpsy-4b full test split, dual-GPU shards (DONE)
VAL_MEDPSY = False              # medpsy-4b full validation split (DONE)
RAG_MEDPSY = False              # medpsy-4b + RAG on MedQA test split (500), dual-GPU shards (DONE)
SC_MODE = False                 # medpsy-4b self-consistency mode
SC_SPLIT = "test"
SC_CONFIRM_TEST = True
SC_K = 5
SC_TEMPERATURE = 0.7
SC_TOP_P = 0.95
SC_PILOT_N = 0
SC_BATCH_SIZE = 5               # batch size for sampling to avoid T4 OOM
SC_DATASETS = "unanswerable_v2"
SC_SHARD = None
SC_NUM_SHARDS = 2

# ── Specialized Task Modes (Tasks 4 & 5) ──────────────────────────────────────
# Set EXACTLY ONE of the following modes to True before running/pushing with `kaggle kernels push`:
BUILD_UNANS_V2_EVIDENCE = False # Task 4 Pre-Pass: produce hybrid retrieval & rerank on unanswerable_v2
UNANS_V2_SLM = False            # Task 4: run 5 SLMs baseline + RAG on unanswerable_v2 (val + test)
UNANS_MODEL = "qwen3-4b"         # Target SLM: 'qwen3-1.7b', 'phi4-mini', 'smollm3-3b', 'qwen3-4b', 'gemma3-4b'
MEDPSY_UNANS_V2_BASELINE = False # Stage A: medpsy-4b on unanswerable_v2 with Phase 5 baseline prompt (COMPLETED & COMMITTED)
MEDPSY_PUBMEDQA_BASELINE = False # Stage B: medpsy-4b closed-book baseline on PubMedQA test (COMPLETED & COMMITTED)
MEDPSY_PUBMEDQA_RAG = False      # Stage C: medpsy-4b RAG on PubMedQA test (COMPLETED & COMMITTED)

# Unified 2-Stage Push: Stage A2 (MedPsy-4B reasoning prompt) + Stage D (Qwen3-4B baseline + RAG)
MEDPSY_UNANS_V2_REASONING = False # Stage A2: medpsy-4b unans_v2 baseline with native reasoning prompt (COMPLETED & COMMITTED)
QWEN3_4B_UNANS_V2 = False         # Stage D: qwen3-4b on unanswerable_v2 baseline + RAG (COMPLETED & COMMITTED)

# ── Pair B Thinking-Mode Control: Qwen3-4B-Thinking-2507 MedQA Baseline + RAG ──
QWEN3_4B_THINKING_MEDQA = False   # Run qwen3-4b-thinking-2507 on MedQA test (N=500) baseline and RAG sequentially

# ── V3 Generic-vs-Medical Matched-Pair Experiment ─────────────────────────────
# One model per push. Set V3_RUN=True, choose model and conditions, then push.
# Pilot (V3_PILOT=True) runs the stratified N=50 val subset first; full run after.
# Output path: outputs/v3/<model>/<condition>/<model>.jsonl
V3_RUN = False
V3_MODEL = "qwen3-1.7b-thinking"
V3_CONDITIONS = "C2"
V3_PILOT = False
V3_PILOT_N = 10
V3_CONFIRM_TEST = False
V3_DATASETS = "medqa"
V3_PARALLEL_MODELS = "medpsy-1.7b,qwen3-1.7b-thinking"

# Truncation sensitivity check (Task 6, validation only, no model inference at full scale)
V3_TRUNC_SENSITIVITY = False # compare cap=1024 vs cap=2048 on 100 MedQA val questions for 4 thinking models

# Token-budget sweep for Pair B (Task 1/5, validation only, greedy cap=4096)
V3_BUDGET_SWEEP = False
V3_SWEEP_N = 100
V3_SWEEP_MAX_HOURS = 10.0
V3_SWEEP_BATCH_SIZE = 1

# Gemma prompt-sensitivity check (Task 2/5, validation only, greedy cap=16, 3 prompt variants)
V3_GEMMA_PROMPT_SENS = True
V3_GEMMA_SENS_N = 3
V3_GEMMA_SENS_MAX_HOURS = 3.0

# Optional legacy env overrides for base runner settings
for flag, val in [
    ("FINAL_MEDPSY", "FINAL_MEDPSY"),
    ("VAL_MEDPSY", "VAL_MEDPSY"),
    ("RAG_MEDPSY", "RAG_MEDPSY"),
    ("SC_MODE", "SC_MODE"),
    ("CHECK_ONLY", "CHECK_ONLY"),
    ("PROFILE", "PROFILE"),
    ("PILOT", "PILOT"),
    ("PARALLEL_GPUS", "PARALLEL_GPUS"),
    ("TINY", "TINY"),
    ("V3_RUN", "V3_RUN"),
    ("V3_TRUNC_SENSITIVITY", "V3_TRUNC_SENSITIVITY"),
    ("V3_BUDGET_SWEEP", "V3_BUDGET_SWEEP"),
    ("V3_GEMMA_PROMPT_SENS", "V3_GEMMA_PROMPT_SENS"),
]:
    if flag in os.environ:
        locals()[val] = os.environ[flag].lower() in ("1", "true", "yes")

if "V3_SWEEP_N" in os.environ:
    V3_SWEEP_N = int(os.environ["V3_SWEEP_N"])
if "V3_SWEEP_MAX_HOURS" in os.environ:
    V3_SWEEP_MAX_HOURS = float(os.environ["V3_SWEEP_MAX_HOURS"])
if "V3_SWEEP_BATCH_SIZE" in os.environ:
    V3_SWEEP_BATCH_SIZE = int(os.environ["V3_SWEEP_BATCH_SIZE"])
if "V3_GEMMA_SENS_N" in os.environ:
    V3_GEMMA_SENS_N = int(os.environ["V3_GEMMA_SENS_N"])
if "V3_GEMMA_SENS_MAX_HOURS" in os.environ:
    V3_GEMMA_SENS_MAX_HOURS = float(os.environ["V3_GEMMA_SENS_MAX_HOURS"])

# Check for unified MedPsy 3-stage overnight push (historical completed run)
_is_medpsy_overnight = (
    MEDPSY_UNANS_V2_BASELINE and MEDPSY_PUBMEDQA_BASELINE and MEDPSY_PUBMEDQA_RAG
    and not (BUILD_UNANS_V2_EVIDENCE or UNANS_V2_SLM or V3_RUN or V3_TRUNC_SENSITIVITY or V3_BUDGET_SWEEP or V3_GEMMA_PROMPT_SENS or MEDPSY_UNANS_V2_REASONING or QWEN3_4B_UNANS_V2 or QWEN3_4B_THINKING_MEDQA)
)

# Check for unified 2-stage push: Stage A2 (MedPsy-4B reasoning baseline) + Stage D (Qwen3-4B base+RAG)
_is_two_stage_push = (
    MEDPSY_UNANS_V2_REASONING and QWEN3_4B_UNANS_V2
    and not (BUILD_UNANS_V2_EVIDENCE or UNANS_V2_SLM or MEDPSY_UNANS_V2_BASELINE or MEDPSY_PUBMEDQA_BASELINE or MEDPSY_PUBMEDQA_RAG or V3_RUN or V3_TRUNC_SENSITIVITY or V3_BUDGET_SWEEP or V3_GEMMA_PROMPT_SENS or QWEN3_4B_THINKING_MEDQA)
)

# Mutual exclusivity guard for specialized modes vs base/legacy runner flags
_active_special_modes = [
    m for m, v in [
        ("BUILD_UNANS_V2_EVIDENCE", BUILD_UNANS_V2_EVIDENCE),
        ("UNANS_V2_SLM", UNANS_V2_SLM),
        ("MEDPSY_UNANS_V2_BASELINE", MEDPSY_UNANS_V2_BASELINE),
        ("MEDPSY_PUBMEDQA_BASELINE", MEDPSY_PUBMEDQA_BASELINE),
        ("MEDPSY_PUBMEDQA_RAG", MEDPSY_PUBMEDQA_RAG),
        ("MEDPSY_UNANS_V2_REASONING", MEDPSY_UNANS_V2_REASONING),
        ("QWEN3_4B_UNANS_V2", QWEN3_4B_UNANS_V2),
        ("QWEN3_4B_THINKING_MEDQA", QWEN3_4B_THINKING_MEDQA),
        ("V3_RUN", V3_RUN),
        ("V3_TRUNC_SENSITIVITY", V3_TRUNC_SENSITIVITY),
        ("V3_BUDGET_SWEEP", V3_BUDGET_SWEEP),
        ("V3_GEMMA_PROMPT_SENS", V3_GEMMA_PROMPT_SENS),
    ] if v
]
if not (_is_medpsy_overnight or _is_two_stage_push) and len(_active_special_modes) > 1:
    sys.exit(f"ERROR: Mutually exclusive runner modes enabled: {_active_special_modes}. Enable only one before pushing.")

if _active_special_modes:
    _disallowed = [
        ("PILOT", PILOT),
        ("FINAL_MEDPSY", FINAL_MEDPSY),
        ("VAL_MEDPSY", VAL_MEDPSY),
        ("RAG_MEDPSY", RAG_MEDPSY),
        ("SC_MODE", SC_MODE),
        ("PROFILE", PROFILE),
        ("PARALLEL_GPUS", PARALLEL_GPUS),
    ]
    # for V3_RUN, V3_TRUNC_SENSITIVITY, V3_BUDGET_SWEEP, and V3_GEMMA_PROMPT_SENS, TINY and CHECK_ONLY are supported.
    if not (V3_RUN or V3_TRUNC_SENSITIVITY or V3_BUDGET_SWEEP or V3_GEMMA_PROMPT_SENS):
        _disallowed.extend([("TINY", TINY), ("CHECK_ONLY", CHECK_ONLY)])
    _conflicting_flags = [m for m, v in _disallowed if v]
    if _conflicting_flags:
        sys.exit(
            f"ERROR: Special mode {_active_special_modes[0]} cannot be run together with flags {_conflicting_flags}. "
            "Set these flags to False when running a special mode."
        )


_active_legacy_modes = [
    m for m, v in [
        ("FINAL_MEDPSY", FINAL_MEDPSY),
        ("VAL_MEDPSY", VAL_MEDPSY),
        ("RAG_MEDPSY", RAG_MEDPSY),
        ("SC_MODE", SC_MODE),
        ("PILOT", PILOT),
    ] if v
]
if len(_active_legacy_modes) > 1:
    sys.exit(f"ERROR: Mutually exclusive legacy modes enabled: {_active_legacy_modes}. Enable only one before pushing.")

if "MODELS" in os.environ:
    MODELS = [m.strip() for m in os.environ["MODELS"].split(",") if m.strip()]
if "MODES" in os.environ:
    MODES = [m.strip() for m in os.environ["MODES"].split(",") if m.strip()]
if "SC_SPLIT" in os.environ:
    SC_SPLIT = os.environ["SC_SPLIT"].strip().lower()
if "SC_SHARD" in os.environ:
    SC_SHARD = int(os.environ["SC_SHARD"])
if "SC_NUM_SHARDS" in os.environ:
    SC_NUM_SHARDS = int(os.environ["SC_NUM_SHARDS"])

# ── SC_MODE Startup safety assertions ──────────────────────────────────────────
if SC_MODE:
    assert SC_SPLIT in ("validation", "test"), f"Invalid SC_SPLIT='{SC_SPLIT}'. Must be 'validation' or 'test'."
    if SC_SPLIT == "test":
        if not SC_CONFIRM_TEST:
            sys.exit(
                "\n" + "=" * 80 + "\n"
                "ERROR: SC_SPLIT is set to 'test', but SC_CONFIRM_TEST is not True.\n"
                "The test set is frozen and this run is final. Retuning after seeing test numbers is forbidden.\n"
                "To proceed with the final frozen test evaluation, set SC_CONFIRM_TEST = True in runner.py.\n"
                + "=" * 80 + "\n"
            )
        assert SC_PILOT_N == 0, (
            f"ERROR: SC_PILOT_N={SC_PILOT_N} is not allowed when SC_SPLIT='test'. "
            "No piloting on the test set; full run only (require SC_PILOT_N == 0)."
        )


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


def _guard_no_token_leak():
    """Ensure hf_token.txt is never copied or symlinked into /kaggle/working or any outputs folder."""
    for base_dir in ("/kaggle/working", "outputs"):
        if os.path.exists(base_dir):
            for root, _, files in os.walk(base_dir, followlinks=False):
                if "hf_token.txt" in files:
                    leak = os.path.join(root, "hf_token.txt")
                    try:
                        os.remove(leak)
                    except Exception:
                        pass
                    raise RuntimeError(f"SECURITY ALERT: {leak} detected in output/working directory and purged.")


atexit.register(_guard_no_token_leak)


if SC_SHARD is None:
    print_kaggle_input_configs()

    # 1. Hugging Face token from Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle Secrets")
    except Exception as e:  # noqa: BLE001
        print("WARNING: HF_TOKEN secret not available:", e)

    # Fallback: load HF_TOKEN from private dataset if not in secrets
    if "HF_TOKEN" not in os.environ:
        token_matches = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)
        if token_matches:
            try:
                with open(token_matches[0], "r", encoding="utf-8") as f_tok:
                    tok_val = f_tok.read().strip()
                if tok_val:
                    os.environ["HF_TOKEN"] = tok_val
                    print("HF_TOKEN loaded from private dataset", flush=True)
            except Exception as e_tok:  # noqa: BLE001
                print("WARNING: Failed to read hf_token.txt from private dataset:", e_tok)

    # Verify token file was not copied into working directory or outputs
    _guard_no_token_leak()

    # 2. Fresh copy of the code
    run(["rm", "-rf", CODE_DIR])
    run(["git", "clone", "--depth", "1", REPO_URL, CODE_DIR])
    os.chdir(CODE_DIR)
    sys.path.insert(0, CODE_DIR)
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
    if FINAL_MEDPSY:
        print("FINAL_MEDPSY mode enabled: medpsy-4b full test split, 2-shard dual-GPU")
    if RAG_MEDPSY:
        print("RAG_MEDPSY mode enabled: medpsy-4b + RAG on MedQA test (500), 2-shard dual-GPU")
    if BUILD_UNANS_V2_EVIDENCE:
        print("BUILD_UNANS_V2_EVIDENCE mode enabled: producing hybrid retrieval (Phase 6) and reranking (Phase 7) on unanswerable_v2")
    if UNANS_V2_SLM:
        print(f"UNANS_V2_SLM mode enabled: model={UNANS_MODEL} baseline+RAG on unanswerable_v2 (val+test)")
    if _is_two_stage_push:
        print("TWO_STAGE_UNANS_PUSH mode enabled: Stage A2 (medpsy-4b reasoning baseline) + Stage D (qwen3-4b baseline+RAG)")
    if QWEN3_4B_THINKING_MEDQA:
        print("QWEN3_4B_THINKING_MEDQA mode enabled: qwen3-4b-thinking-2507 on MedQA test (N=500) baseline + RAG (Pair B thinking control, 2-GPU shards)")
    if _is_medpsy_overnight:
        print("MEDPSY_OVERNIGHT mode enabled: medpsy-4b unified 3-stage push (Stage A: unans_v2 baseline, Stage B: PubMedQA baseline, Stage C: PubMedQA RAG)")
    else:
        if MEDPSY_UNANS_V2_BASELINE:
            print("MEDPSY_UNANS_V2_BASELINE mode enabled: medpsy-4b Phase 5 baseline prompt on unanswerable_v2 (val+test), 2-GPU shards")
        if MEDPSY_PUBMEDQA_BASELINE:
            print("MEDPSY_PUBMEDQA_BASELINE mode enabled: medpsy-4b closed-book baseline on PubMedQA test (500), dual-GPU")
        if MEDPSY_PUBMEDQA_RAG:
            print("MEDPSY_PUBMEDQA_RAG mode enabled: medpsy-4b RAG on PubMedQA test (500), dual-GPU")
    if VAL_MEDPSY:
        print("VAL_MEDPSY mode enabled: medpsy-4b full validation split, 2-shard dual-GPU")
    if PILOT:
        print("PILOT mode enabled: medpsy-4b pilot on 80-question val subset, single GPU")
    if V3_RUN:
        _v3_conditions = [c.strip() for c in V3_CONDITIONS.split(",") if c.strip()]
        _v3_pilot_str = f" [PILOT N={V3_PILOT_N}]" if V3_PILOT else ""
        if V3_PARALLEL_MODELS:
            print(f"V3_RUN mode enabled (PARALLEL): models={V3_PARALLEL_MODELS}, conditions={_v3_conditions}{_v3_pilot_str}")
        else:
            print(f"V3_RUN mode enabled: model={V3_MODEL}, conditions={_v3_conditions}{_v3_pilot_str}")
        if not V3_PILOT and not V3_CONFIRM_TEST:
            sys.exit(
                "\n" + "=" * 80 + "\n"
                "ERROR: V3_RUN on the TEST split requires V3_CONFIRM_TEST=True.\n"
                "The test set is frozen. Set V3_CONFIRM_TEST=True once you are ready for the final run.\n"
                + "=" * 80 + "\n"
            )
        _v3_valid_models = [
            "medpsy-1.7b", "qwen3-1.7b-thinking", "qwen3-4b-thinking-2507", "medgemma-1.5-4b-it",
            "qwen3-1.7b", "qwen3-4b", "gemma3-4b", "medpsy-4b", "phi4-mini", "smollm3-3b",
        ]
        if V3_PARALLEL_MODELS:
            _parallel_models = [m.strip() for m in V3_PARALLEL_MODELS.split(",") if m.strip()]
            if len(_parallel_models) != 2:
                sys.exit(f"ERROR: V3_PARALLEL_MODELS must specify exactly two comma-separated models, got: {V3_PARALLEL_MODELS}")
            _bad_pmodels = [m for m in _parallel_models if m not in _v3_valid_models]
            if _bad_pmodels:
                sys.exit(f"ERROR: V3_PARALLEL_MODELS contains unknown models: {_bad_pmodels}. Valid: {_v3_valid_models}")
        else:
            if V3_MODEL not in _v3_valid_models:
                sys.exit(f"ERROR: V3_MODEL='{V3_MODEL}' is not a valid v3 model. Choose one of: {_v3_valid_models}")
        _v3_valid_conditions = {"C1", "C2", "C3", "C4", "C5"}
        _bad_conds = [c for c in _v3_conditions if c not in _v3_valid_conditions]
        if _bad_conds:
            sys.exit(f"ERROR: V3_CONDITIONS contains unknown conditions: {_bad_conds}. Valid: {_v3_valid_conditions}")
        _v3_valid_datasets = {"all", "medqa", "pubmedqa"}
        if V3_DATASETS not in _v3_valid_datasets:
            sys.exit(f"ERROR: V3_DATASETS='{V3_DATASETS}' is not valid. Choose one of: {sorted(_v3_valid_datasets)}")
        if "C3" in _v3_conditions and V3_DATASETS == "medqa":
            sys.exit("ERROR: Condition C3 requires pubmedqa dataset, but V3_DATASETS='medqa'")
    if V3_TRUNC_SENSITIVITY:
        print("V3_TRUNC_SENSITIVITY mode enabled: cap=1024 vs cap=2048 on 100 MedQA val questions for 4 thinking models")
    if V3_BUDGET_SWEEP:
        print(f"V3_BUDGET_SWEEP mode enabled: models=['qwen3-4b-thinking-2507', 'medpsy-4b'], N={V3_SWEEP_N}, max_hours={V3_SWEEP_MAX_HOURS}, batch_size={V3_SWEEP_BATCH_SIZE}")
    if V3_GEMMA_PROMPT_SENS:
        print(f"V3_GEMMA_PROMPT_SENS mode enabled: models=['gemma3-4b', 'medgemma-1.5-4b-it'], N={V3_GEMMA_SENS_N}, max_hours={V3_GEMMA_SENS_MAX_HOURS}, variants=['V1', 'V2', 'V3']")

    if PARALLEL_GPUS:
        print("PARALLEL_GPUS mode enabled: splitting model list across 2 GPUs")
    if TINY:
        print("TINY mode enabled: running on tiny scale")
    if SC_MODE:
        if SC_SPLIT == "test":
            print("\n" + "#" * 80, flush=True)
            print("=== FINAL TEST EVALUATION — frozen numbers, reliability score already fit on validation, DO NOT retune after seeing these ===", flush=True)
            print("#" * 80 + "\n", flush=True)
        print(f"SC_MODE enabled: medpsy-4b self-consistency (split={SC_SPLIT}, parent coordinator, dual-GPU, datasets={SC_DATASETS}, K={SC_K}, temp={SC_TEMPERATURE}, top_p={SC_TOP_P}, pilot_n={SC_PILOT_N}, batch_size={SC_BATCH_SIZE})")

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
        if SC_MODE:
            print(f"Note: evidence.jsonl not found; SC_MODE requires only {SC_SPLIT} data.", flush=True)
        else:
            sys.exit("ERROR: medrag-build output not found. Attach it: Add Input -> Your Work -> medrag-build")
    else:
        build_dir = os.path.dirname(os.path.dirname(hits[0]))
        print("Using build output from:", build_dir)
        os.makedirs(WORK, exist_ok=True)
        for n in (1, 2, 3, 6, 7):
            src_dir, dst = os.path.join(build_dir, f"phase{n}"), os.path.join(WORK, f"phase{n}")
            if os.path.isdir(src_dir) and not os.path.exists(dst):
                os.symlink(src_dir, dst)
else:
    # Worker subprocess (SC_SHARD is set): skip all one-time setup
    if os.path.exists(CODE_DIR):
        os.chdir(CODE_DIR)
    sys.path.insert(0, CODE_DIR)


# ── Helper to build the qa_pipeline command ────────────────────────────────────

def build_qa_cmd(models_list, modes_list, split=None, work_dir=None, check_only=False,
                 profile=False, tiny=False, skip_format_check=False,
                 shard=None, num_shards=2, datasets=None, standard_prompt=False):
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
    if standard_prompt:
        cmd.append("--standard-prompt")
    if models_list:
        cmd += ["--models", *models_list]
    if modes_list:
        cmd += ["--modes", *modes_list]
    if split:
        cmd += ["--split", split]
    if datasets:
        cmd += ["--datasets", datasets]
    if shard is not None:
        cmd += ["--shard", str(shard), "--num-shards", str(num_shards)]
    return cmd


def verify_frozen_hashes_before_run(code_dir, work_dir):
    """Verify cryptographic SHA-256 integrity of frozen datasets before execution.
    
    Checks unanswerable_v2 val.jsonl, unanswerable_v2 test.jsonl, and phase1/test.jsonl (PubMedQA).
    If any hash does not match the canonical frozen hash, ABORTS the kernel immediately.
    """
    import hashlib
    print("\n" + "=" * 80, flush=True)
    print("=== PRE-FLIGHT FROZEN DATA INTEGRITY AUDIT ===", flush=True)
    print("=" * 80, flush=True)
    
    def get_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    manifest = {}
    for mf in [
        os.path.join(code_dir, "audit", "frozen_hashes.json"),
        os.path.abspath("audit/frozen_hashes.json"),
    ]:
        if os.path.exists(mf):
            try:
                with open(mf) as f:
                    manifest = json.load(f).get("files", {})
                break
            except Exception:
                pass

    EXPECTED_HASHES = {
        "unans_v2_test": "a0e45676dced360b2ac64e8fc33dbb49782c4fd9022e93190e633c9d66396078",
        "pubmedqa_test": "fe7fe3fb88c77c4ffc6574c8716ec0b3f4787fbfaf2c22e2eb5741ac040cc34a",
    }
    if "data/unanswerable_v2/val.jsonl" in manifest:
        EXPECTED_HASHES["unans_v2_val"] = manifest["data/unanswerable_v2/val.jsonl"]["sha256"]
    
    for val_sha_p in [
        os.path.join(code_dir, "data", "unanswerable_v2", "val.sha256"),
        os.path.abspath("data/unanswerable_v2/val.sha256"),
    ]:
        if os.path.exists(val_sha_p):
            try:
                EXPECTED_HASHES["unans_v2_val"] = open(val_sha_p).read().strip().split()[0]
                break
            except Exception:
                pass

    # 1. Locate unanswerable_v2 val.jsonl
    val_candidates = [
        os.path.join(code_dir, "data", "unanswerable_v2", "val.jsonl"),
        os.path.abspath("data/unanswerable_v2/val.jsonl"),
    ]
    val_src = next((p for p in val_candidates if os.path.exists(p)), None)
    if not val_src:
        input_hits = glob.glob("/kaggle/input/**/unanswerable_v2/val.jsonl", recursive=True)
        if input_hits:
            val_src = input_hits[0]
    if not val_src or not os.path.exists(val_src):
        sys.exit("FATAL: unanswerable_v2 val.jsonl not found in code dir or /kaggle/input/**")

    # 2. Locate unanswerable_v2 test.jsonl
    test_candidates = [
        os.path.join(code_dir, "data", "unanswerable_v2", "test.jsonl"),
        os.path.abspath("data/unanswerable_v2/test.jsonl"),
    ]
    test_src = next((p for p in test_candidates if os.path.exists(p)), None)
    if not test_src:
        input_hits = glob.glob("/kaggle/input/**/unanswerable_v2/test.jsonl", recursive=True)
        if input_hits:
            test_src = input_hits[0]
    if not test_src or not os.path.exists(test_src):
        sys.exit("FATAL: unanswerable_v2 test.jsonl not found in code dir or /kaggle/input/**")

    # 3. Locate PubMedQA test file (phase1/test.jsonl)
    p1_candidates = [
        os.path.join(code_dir, "outputs", "kaggle_build", "work", "phase1", "test.jsonl"),
        os.path.abspath("outputs/kaggle_build/work/phase1/test.jsonl"),
        os.path.join(work_dir, "phase1", "test.jsonl"),
    ]
    p1_test_src = next((p for p in p1_candidates if os.path.exists(p)), None)
    if not p1_test_src:
        input_hits = glob.glob("/kaggle/input/**/phase1/test.jsonl", recursive=True)
        if input_hits:
            p1_test_src = input_hits[0]
    if not p1_test_src or not os.path.exists(p1_test_src):
        sys.exit("FATAL: PubMedQA phase1/test.jsonl not found in code dir, work dir or /kaggle/input/**")

    files_to_check = [
        ("unanswerable_v2 test", test_src, EXPECTED_HASHES.get("unans_v2_test")),
        ("unanswerable_v2 val", val_src, EXPECTED_HASHES.get("unans_v2_val")),
        ("PubMedQA phase1 test", p1_test_src, EXPECTED_HASHES.get("pubmedqa_test")),
    ]

    for label, path, expected in files_to_check:
        actual = get_sha256(path)
        print(f"\nAudit: {label} [{path}]", flush=True)
        print(f"  SHA-256 Calculated : {actual}", flush=True)
        if expected:
            print(f"  SHA-256 Expected   : {expected}", flush=True)
            if actual != expected:
                sys.exit(
                    f"\n{'='*80}\n"
                    f"FATAL INTEGRITY ERROR: SHA-256 mismatch for {label}!\n"
                    f"  File path: {path}\n"
                    f"  Calculated: {actual}\n"
                    f"  Expected  : {expected}\n"
                    f"ABORTING KERNEL IMMEDIATELY to prevent benchmark corruption.\n"
                    f"{'='*80}\n"
                )
            print(f"  [PASS] Integrity Verified (Match)", flush=True)
        else:
            print(f"  [INFO] Hash recorded (no static expected string provided)", flush=True)

    print("\n=== PRE-FLIGHT FROZEN DATA INTEGRITY AUDIT PASSED ===", flush=True)
    print("=" * 80 + "\n", flush=True)
    return val_src, test_src, p1_test_src


# ── SC_MODE: Self-Consistency Mode for MedPsy-4B (Validation or Test) ──────────
if SC_MODE:
    import collections
    import math
    import random
    import re
    import time

    # ══════════════════════════════════════════════════════════════════════════
    # PARENT COORDINATOR (SC_SHARD is None): Launch dual-GPU shards & merge
    # ══════════════════════════════════════════════════════════════════════════
    if SC_SHARD is None:
        print("\n" + "=" * 80, flush=True)
        if SC_SPLIT == "test":
            print("=== FINAL TEST EVALUATION — frozen numbers, reliability score already fit on validation, DO NOT retune after seeing these ===", flush=True)
        print(f"=== SC_MODE PARENT: Launching dual-GPU shards (split={SC_SPLIT}) ===", flush=True)
        print(f"  SC_SPLIT       : {SC_SPLIT}", flush=True)
        print(f"  SC_DATASETS    : {SC_DATASETS}", flush=True)
        print(f"  SC_K           : {SC_K}", flush=True)
        print(f"  SC_TEMPERATURE : {SC_TEMPERATURE}", flush=True)
        print(f"  SC_TOP_P       : {SC_TOP_P}", flush=True)
        print(f"  SC_PILOT_N     : {SC_PILOT_N} ({'full run required for test' if SC_SPLIT == 'test' else '0 = full validation set'})", flush=True)
        print(f"  SC_BATCH_SIZE  : {SC_BATCH_SIZE}", flush=True)
        print(f"  SC_NUM_SHARDS  : {SC_NUM_SHARDS}", flush=True)
        print("=" * 80 + "\n", flush=True)

        split_subdir = "sc_test" if SC_SPLIT == "test" else "sc_val"
        sc_out_dir = f"/kaggle/working/outputs/{split_subdir}/medpsy-4b" if os.path.exists("/kaggle/working") else f"outputs/{split_subdir}/medpsy-4b"
        os.makedirs(sc_out_dir, exist_ok=True)
        if os.path.exists("/kaggle/working") and os.getcwd() != "/kaggle/working":
            try:
                os.makedirs(f"outputs/{split_subdir}/medpsy-4b", exist_ok=True)
            except Exception:
                pass

        os.environ["KERNEL_START_TIME"] = str(KERNEL_START_TIME)
        procs = []
        for sid in range(SC_NUM_SHARDS):
            env_shard = os.environ.copy()
            if "HF_TOKEN" in os.environ:
                env_shard["HF_TOKEN"] = os.environ["HF_TOKEN"]
            env_shard["SC_SPLIT"] = SC_SPLIT
            env_shard["CUDA_VISIBLE_DEVICES"] = str(sid)
            env_shard["SC_SHARD"] = str(sid)
            env_shard["SC_NUM_SHARDS"] = str(SC_NUM_SHARDS)
            env_shard["KERNEL_START_TIME"] = str(KERNEL_START_TIME)

            script_file = os.path.join(CODE_DIR, "kaggle_runner", "runner.py")
            if not os.path.exists(script_file):
                script_file = SCRIPT_PATH
            cmd = [sys.executable, script_file]
            print(f">> [Parent] Launching shard {sid} with CUDA_VISIBLE_DEVICES={sid}, SC_SHARD={sid}, SC_NUM_SHARDS={SC_NUM_SHARDS}", flush=True)
            p = subprocess.Popen(cmd, env=env_shard)
            procs.append((sid, p))

        print(f"\n[Parent] Waiting for {len(procs)} shard subprocesses...", flush=True)
        exit_codes = {}
        for sid, p in procs:
            rc = p.wait()
            exit_codes[sid] = rc
            print(f"[Parent] Shard {sid} finished with exit code {rc}", flush=True)

        if any(rc != 0 for rc in exit_codes.values()):
            print(f"\n[Parent] WARNING: One or more shards exited with non-zero code: {exit_codes}", flush=True)

        # ── 3. Merge shards into <dataset>.jsonl ───────────────────────────
        print("\n" + "=" * 80, flush=True)
        print(f"=== SC_MODE PARENT: MERGING SHARDS (split={SC_SPLIT}) ===", flush=True)
        active_sc_datasets = [d.strip().lower() for d in SC_DATASETS.split(",") if d.strip()]

        if SC_SPLIT == "test":
            expected_map = {
                "medqa": 500,
                "pubmedqa": 500,
                "unanswerable_v2": 140,
            }
        else:
            expected_map = {
                "medqa": 200 if SC_PILOT_N == 0 else SC_PILOT_N,
                "pubmedqa": 200 if SC_PILOT_N == 0 else SC_PILOT_N,
                "unanswerable_v2": 260 if SC_PILOT_N == 0 else SC_PILOT_N,
            }

        local_dir = f"outputs/{split_subdir}/medpsy-4b"
        for ds_name in active_sc_datasets:
            shard_rows = []
            for sid in range(SC_NUM_SHARDS):
                sp = os.path.join(sc_out_dir, f"{ds_name}_shard{sid}.jsonl")
                if not os.path.exists(sp):
                    local_sp = os.path.join(local_dir, f"{ds_name}_shard{sid}.jsonl")
                    if os.path.exists(local_sp):
                        sp = local_sp
                if os.path.exists(sp):
                    with open(sp, "r", encoding="utf-8") as fh:
                        part = [json.loads(line) for line in fh if line.strip()]
                    print(f"  [Merge] {ds_name} shard {sid}: {len(part)} rows from {sp}", flush=True)
                    shard_rows.extend(part)
                else:
                    print(f"  [Merge] WARNING: Shard file {sp} not found for {ds_name} shard {sid}!", flush=True)

            # Check duplicate IDs
            ids = [r["id"] for r in shard_rows]
            if len(ids) != len(set(ids)):
                dups = len(ids) - len(set(ids))
                print(f"  [Merge] WARNING: {ds_name} has {dups} duplicate ID(s) across shards!", flush=True)

            # Check expected count
            expected = expected_map.get(ds_name)
            if expected is not None and len(shard_rows) != expected:
                print(f"  [Merge] WARNING: {ds_name} expected {expected} rows (SC_SPLIT={SC_SPLIT}, SC_PILOT_N={SC_PILOT_N}), got {len(shard_rows)} rows!", flush=True)

            # Write merged file
            merged_file = os.path.join(sc_out_dir, f"{ds_name}.jsonl")
            with open(merged_file, "w", encoding="utf-8") as fh:
                for r in shard_rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"  [Merge] Successfully wrote {len(shard_rows)} rows to {merged_file}", flush=True)

            # Duplicate to repo-local dir if different
            if os.path.exists(local_dir) and sc_out_dir != local_dir:
                try:
                    local_merged = os.path.join(local_dir, f"{ds_name}.jsonl")
                    with open(local_merged, "w", encoding="utf-8") as fh:
                        for r in shard_rows:
                            fh.write(json.dumps(r) + "\n")
                except Exception:
                    pass

        print("=" * 80 + "\n", flush=True)
        print("=== SC_MODE EXECUTION FINISHED SUCCESSFULLY ===", flush=True)
        print(f"All merged outputs stored in: {sc_out_dir}", flush=True)
        print("=" * 80 + "\n", flush=True)
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════════
    # SUBPROCESS WORKER (SC_SHARD is set): Process only shard items
    # ══════════════════════════════════════════════════════════════════════════
    assert SC_SHARD is not None, "Parent coordinator must never enter worker logic or load models"
    import torch
    from transformers import GenerationConfig

    from src.common import get_device, load_config
    from src.llm import (
        LLM,
        extract_final_answer_from_reasoning,
        parse_answer,
        reasoning_baseline_prompt,
        reasoning_context_prompt,
    )
    from src.qa_pipeline import ensure_pubmedqa_contexts

    active_sc_datasets = [d.strip().lower() for d in SC_DATASETS.split(",") if d.strip()]

    # ── Load config and extract exact context-mode settings from cfg ───────────
    cfg = load_config("configs/base.yaml")
    qcfg = cfg.get("qa", {})
    mcfg = next(m for m in cfg["models"] if m["name"] == "medpsy-4b")
    is_reasoning = mcfg.get("reasoning", False)

    context_max_chars = qcfg["passage_max_chars"] * 3
    context_max_new_tokens = (
        mcfg.get("max_new_tokens_reasoning")
        if is_reasoning
        else (qcfg.get("max_new_tokens_context") or qcfg["max_new_tokens_rag"])
    )

    print("\n" + "=" * 80, flush=True)
    if SC_SPLIT == "test":
        print("=== FINAL TEST EVALUATION — frozen numbers, reliability score already fit on validation, DO NOT retune after seeing these ===", flush=True)
    print(f"=== SC_MODE SHARD {SC_SHARD}/{SC_NUM_SHARDS} STARTUP: MedPsy-4B Self-Consistency ({SC_SPLIT}) ===", flush=True)
    print(f"  SC_SPLIT       : {SC_SPLIT}", flush=True)
    print(f"  SC_SHARD       : {SC_SHARD} of {SC_NUM_SHARDS}", flush=True)
    print(f"  SC_DATASETS    : {SC_DATASETS} ({active_sc_datasets})", flush=True)
    print(f"  SC_K           : {SC_K}", flush=True)
    print(f"  SC_TEMPERATURE : {SC_TEMPERATURE}", flush=True)
    print(f"  SC_TOP_P       : {SC_TOP_P}", flush=True)
    print(f"  SC_PILOT_N     : {SC_PILOT_N} ({'full run required for test' if SC_SPLIT == 'test' else '0 = full validation set'})", flush=True)
    print(f"  SC_BATCH_SIZE  : {SC_BATCH_SIZE}", flush=True)
    print(f"  Context max_chars     : {context_max_chars}", flush=True)
    print(f"  Context max_new_tokens: {context_max_new_tokens}", flush=True)
    print("=" * 80 + "\n", flush=True)

    # ── 1. Stratified sampling helper for unanswerable_v2 ──────────────────────
    def sample_stratified_unans_v2(rows, n_target, seed=42):
        """Sample n_target questions stratified by category and control flag.
        Ensures all 6 unanswerable categories + 2 control categories are represented.
        """
        if n_target <= 0 or n_target >= len(rows):
            return rows

        rng = random.Random(seed)
        strata = collections.defaultdict(list)
        for r in rows:
            key = (r.get("category"), bool(r.get("control", False)))
            strata[key].append(r)

        # Sort each stratum deterministically by id
        for k in strata:
            strata[k].sort(key=lambda x: x["id"])

        total_len = len(rows)
        quotas = {}
        remainders = []
        for k, items in strata.items():
            exact = (len(items) / total_len) * n_target
            alloc = int(exact)
            rem = exact - alloc
            quotas[k] = alloc
            remainders.append((rem, k))

        leftover = n_target - sum(quotas.values())
        remainders.sort(key=lambda x: x[0], reverse=True)
        for i in range(leftover):
            quotas[remainders[i % len(remainders)][1]] += 1

        selected = []
        for k, items in strata.items():
            q_cnt = quotas[k]
            if q_cnt > 0:
                selected.extend(rng.sample(items, q_cnt))

        selected.sort(key=lambda x: x["id"])
        return selected

    # ── 2. Load unanswerable_v2 data (with strict split safety asserts) ────────
    unans_all = []
    if "unanswerable_v2" in active_sc_datasets:
        filename = "test.jsonl" if SC_SPLIT == "test" else "val.jsonl"
        unans_candidates = [
            os.path.join(CODE_DIR, "data", "unanswerable_v2", filename),
            f"data/unanswerable_v2/{filename}",
            os.path.abspath(f"data/unanswerable_v2/{filename}"),
            f"/kaggle/working/medrag-slm/data/unanswerable_v2/{filename}",
            f"/kaggle/working/data/unanswerable_v2/{filename}",
        ]
        unans_path = next((p for p in unans_candidates if os.path.exists(p)), None)
        if not unans_path:
            input_hits = (
                glob.glob(f"/kaggle/input/**/unanswerable_v2/{filename}", recursive=True)
                or glob.glob(f"/kaggle/input/**/data/unanswerable_v2/{filename}", recursive=True)
            )
            if input_hits:
                unans_path = input_hits[0]

        if not unans_path or not os.path.exists(unans_path):
            searched_fmt = "\n  - ".join(unans_candidates + [f"/kaggle/input/**/unanswerable_v2/{filename}"])
            sys.exit(
                f"\n" + "=" * 80 + "\n"
                f"FATAL ERROR: data/unanswerable_v2/{filename} NOT FOUND!\n"
                f"Searched candidate locations:\n  - {searched_fmt}\n\n"
                f"Troubleshooting / Resolution:\n"
                f"  1. Git clone path: verify 'data/unanswerable_v2/{filename}' is committed and pushed to git repository:\n"
                f"     {REPO_URL}\n"
                f"  2. Kaggle dataset input: if using an attached dataset, ensure it contains 'data/unanswerable_v2/{filename}'.\n"
                f"  3. Working directory: currently '{os.getcwd()}'; CODE_DIR is '{CODE_DIR}'.\n"
                f"=" * 80 + "\n"
            )

        with open(unans_path, "r", encoding="utf-8") as f:
            unans_all = [json.loads(line) for line in f if line.strip()]

        if SC_SPLIT == "test":
            assert unans_path.endswith("test.jsonl"), (
                f"SAFETY ASSERTION VIOLATION: Expected test.jsonl in SC_SPLIT='test' mode, got {unans_path}!"
            )
            assert "val.jsonl" not in unans_path.lower(), (
                f"SAFETY ASSERTION VIOLATION: Path '{unans_path}' contains val.jsonl in SC_SPLIT='test' mode!"
            )
            assert len(unans_all) == 140, f"Expected 140 unanswerable_v2 test rows, got {len(unans_all)}"
            assert all(r.get("split") == "test" for r in unans_all), (
                "SAFETY ASSERTION VIOLATION: Found non-test split rows in unanswerable test set!"
            )
            assert all(r.get("id", "").startswith("unans-v2-test-") for r in unans_all), (
                "SAFETY ASSERTION VIOLATION: Found IDs not starting with 'unans-v2-test-' in unanswerable test set!"
            )
            print(f"Loaded {len(unans_all)} unanswerable_v2 test questions from {unans_path}", flush=True)
        else:
            assert not unans_path.endswith("test.jsonl"), (
                f"SAFETY ASSERTION VIOLATION: Attempted to load test set {unans_path} in validation SC_MODE!"
            )
            assert "test.jsonl" not in unans_path.lower(), (
                f"SAFETY ASSERTION VIOLATION: Path '{unans_path}' contains test.jsonl in validation SC_MODE!"
            )
            assert len(unans_all) == 260, f"Expected 260 unanswerable_v2 val rows, got {len(unans_all)}"
            assert all(r.get("split") == "validation" for r in unans_all), (
                "SAFETY ASSERTION VIOLATION: Found non-validation split rows in unanswerable val set!"
            )
            assert not any(r.get("id", "").startswith("unans-v2-test-") for r in unans_all), (
                "SAFETY ASSERTION VIOLATION: Found test-split IDs in unanswerable val set!"
            )
            print(f"Loaded {len(unans_all)} unanswerable_v2 validation questions from {unans_path}", flush=True)

    # ── 3. Load MedQA and PubMedQA questions ──────────────────────────────────
    medqa_all = []
    pubmedqa_all = []
    if any(ds in active_sc_datasets for ds in ("medqa", "pubmedqa")):
        split_p1_file = "test.jsonl" if SC_SPLIT == "test" else "validation.jsonl"
        candidates = [
            os.path.join(CODE_DIR, "outputs", "kaggle_build", "work", "phase1", split_p1_file),
            f"outputs/kaggle_build/work/phase1/{split_p1_file}",
            os.path.abspath(f"outputs/kaggle_build/work/phase1/{split_p1_file}"),
            os.path.join(WORK, "phase1", split_p1_file),
            f"/kaggle/working/work/phase1/{split_p1_file}",
        ]
        p1_file = next((p for p in candidates if os.path.exists(p)), None)
        if not p1_file:
            hits_p1 = (
                glob.glob(f"/kaggle/input/**/phase1/{split_p1_file}", recursive=True)
                or glob.glob(f"/kaggle/input/**/{split_p1_file}", recursive=True)
            )
            if hits_p1:
                p1_file = hits_p1[0]
        if not p1_file:
            sys.exit(f"ERROR: phase1/{split_p1_file} not found for MedQA / PubMedQA!")

        with open(p1_file, "r", encoding="utf-8") as f:
            p1_rows = [json.loads(line) for line in f if line.strip()]

        medqa_all = [q for q in p1_rows if q.get("dataset") == "medqa" and not q.get("should_abstain")]
        pubmedqa_all = [q for q in p1_rows if q.get("dataset") == "pubmedqa" and not q.get("should_abstain")]

        if SC_SPLIT == "test":
            if "medqa" in active_sc_datasets:
                assert len(medqa_all) == 500, f"Expected 500 MedQA test rows, got {len(medqa_all)}"
            if "pubmedqa" in active_sc_datasets:
                assert len(pubmedqa_all) == 500, f"Expected 500 PubMedQA test rows, got {len(pubmedqa_all)}"
            print(f"Loaded {len(medqa_all)} MedQA test and {len(pubmedqa_all)} PubMedQA test from {p1_file}", flush=True)
        else:
            if "medqa" in active_sc_datasets:
                assert len(medqa_all) == 200, f"Expected 200 MedQA val rows, got {len(medqa_all)}"
            if "pubmedqa" in active_sc_datasets:
                assert len(pubmedqa_all) == 200, f"Expected 200 PubMedQA val rows, got {len(pubmedqa_all)}"
            print(f"Loaded {len(medqa_all)} MedQA val and {len(pubmedqa_all)} PubMedQA val from {p1_file}", flush=True)

    # ── 4. Apply SC_PILOT_N filter if active ───────────────────────────────────
    medqa_run = []
    pubmedqa_run = []
    unans_run = []
    if SC_PILOT_N > 0:
        assert SC_SPLIT == "validation", "Piloting is strictly forbidden on test set (require SC_PILOT_N == 0)"
        if "medqa" in active_sc_datasets:
            medqa_run = medqa_all[:SC_PILOT_N]
        if "pubmedqa" in active_sc_datasets:
            pubmedqa_run = pubmedqa_all[:SC_PILOT_N]
        if "unanswerable_v2" in active_sc_datasets:
            unans_run = sample_stratified_unans_v2(unans_all, SC_PILOT_N, seed=42)
        print(f"\nPILOT MODE ACTIVE (SC_PILOT_N={SC_PILOT_N}):")
        for ds in active_sc_datasets:
            cnt = len(medqa_run) if ds == "medqa" else (len(pubmedqa_run) if ds == "pubmedqa" else len(unans_run))
            print(f"  {ds:<15}: first {cnt} questions")
    else:
        if "medqa" in active_sc_datasets:
            medqa_run = medqa_all
        if "pubmedqa" in active_sc_datasets:
            pubmedqa_run = pubmedqa_all
        if "unanswerable_v2" in active_sc_datasets:
            unans_run = unans_all
        mode_label = "FULL TEST EVALUATION ACTIVE" if SC_SPLIT == "test" else "FULL VALIDATION MODE ACTIVE"
        print(f"\n{mode_label}:")
        for ds in active_sc_datasets:
            cnt = len(medqa_run) if ds == "medqa" else (len(pubmedqa_run) if ds == "pubmedqa" else len(unans_run))
            print(f"  {ds:<15}: all {cnt} questions")

    # ── Ensure PubMedQA contexts (exactly like qa_pipeline context mode) ───────
    if "pubmedqa" in active_sc_datasets:
        ensure_pubmedqa_contexts(pubmedqa_run, cfg)
        for q in pubmedqa_run:
            assert bool(q.get("context")), f"SAFETY ASSERTION VIOLATION: PubMedQA row {q.get('id')} has empty context!"
        print(f"Verified non-empty contexts attached for all {len(pubmedqa_run)} PubMedQA questions.", flush=True)

    # ── 5. Setup output directories ───────────────────────────────────────────
    split_subdir = "sc_test" if SC_SPLIT == "test" else "sc_val"
    sc_out_dir = f"/kaggle/working/outputs/{split_subdir}/medpsy-4b" if os.path.exists("/kaggle/working") else f"outputs/{split_subdir}/medpsy-4b"
    os.makedirs(sc_out_dir, exist_ok=True)
    if os.path.exists("/kaggle/working") and os.getcwd() != "/kaggle/working":
        try:
            os.makedirs(f"outputs/{split_subdir}/medpsy-4b", exist_ok=True)
        except Exception:
            pass

    # ── 6. Load MedPsy-4B model ────────────────────────────────────────────────
    device = get_device()
    print(f"\nLoading MedPsy-4B (reasoning model) on device={device}...", flush=True)
    llm = LLM(mcfg, device=device)

    # ── 7. Generation helpers: Greedy pass + Batched Sampling ──────────────────
    def final_answer_section(gen_text: str) -> str:
        r"""Return text after the last '</think>' if present;
        otherwise text from the last line matching r"(?im)^\s*Answer\s*:" to the end;
        otherwise the last 300 characters.
        """
        think_matches = list(re.finditer(r"</think>", gen_text, re.IGNORECASE))
        if think_matches:
            return gen_text[think_matches[-1].end():]
        ans_matches = list(re.finditer(r"(?im)^\s*Answer\s*:", gen_text))
        if ans_matches:
            return gen_text[ans_matches[-1].start():]
        return gen_text[-300:] if len(gen_text) > 300 else gen_text

    def generate_single_greedy(q, prompt_text, max_new_tokens=1024):
        """1 extra greedy generation (do_sample=False, temp=0) per question for paired SC vs greedy evaluation."""
        wrapped = llm.wrap_reasoning(prompt_text)
        first_dev = next(llm.model.parameters()).device
        enc = llm.tok([wrapped], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(first_dev) for k, v in enc.items()}
        prompt_len = int(enc["attention_mask"][0].sum())

        gcfg_greedy = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=sorted(llm.eos),
            pad_token_id=llm.pad_id,
            return_dict_in_generate=True,
        )

        t0 = time.perf_counter()
        if llm.device == "cuda" or (isinstance(llm.device, str) and str(llm.device).startswith("cuda")):
            torch.cuda.synchronize()
        with torch.inference_mode():
            out = llm.model.generate(**enc, generation_config=gcfg_greedy)
        if llm.device == "cuda" or (isinstance(llm.device, str) and str(llm.device).startswith("cuda")):
            torch.cuda.synchronize()
        g_time = round(time.perf_counter() - t0, 4)

        seqs = out.sequences if hasattr(out, "sequences") else out
        prompt_ids_len = enc["input_ids"].shape[1]
        gen_ids = seqs[0, prompt_ids_len:].tolist()
        cut = next((idx for idx, t in enumerate(gen_ids) if t in llm.eos), len(gen_ids))
        gen_text = llm.tok.decode(gen_ids[:cut], skip_special_tokens=True)
        hit_max = (cut == len(gen_ids))

        has_final_answer = bool(re.search(r"(?:^|\n)Answer\s*:\s*[A-Za-z]", gen_text, re.MULTILINE))
        truncated = bool(hit_max and not has_final_answer)

        ans_sec = final_answer_section(gen_text)
        has_insuf = bool(re.search(r"\bINSUFFICIENT\s+INFORMATION\b", ans_sec, re.IGNORECASE))
        has_insuf_full = bool(re.search(r"\bINSUFFICIENT\s+INFORMATION\b", gen_text, re.IGNORECASE))
        insuf_in_reasoning_only = bool(has_insuf_full and not has_insuf)

        if has_insuf:
            ans = "ABSTAIN"
        else:
            ans = extract_final_answer_from_reasoning(gen_text, q["options"])
            if ans is None:
                ans = parse_answer(gen_text, q["options"])
            if ans == "ABSTAIN":
                has_insuf = True

        return ans, truncated, has_insuf, insuf_in_reasoning_only, g_time, prompt_len

    def generate_sc_k_samples(q, prompt_text, k_samples, temperature, top_p, batch_size, max_new_tokens=1024):
        wrapped = llm.wrap_reasoning(prompt_text)
        first_dev = next(llm.model.parameters()).device
        enc_single = llm.tok([wrapped], return_tensors="pt", padding=True, add_special_tokens=False)
        enc_single = {k: v.to(first_dev) for k, v in enc_single.items()}
        prompt_ids_len = enc_single["input_ids"].shape[1]

        gcfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=sorted(llm.eos),
            pad_token_id=llm.pad_id,
            return_dict_in_generate=True,
        )

        chunks = []
        rem = k_samples
        while rem > 0:
            c = min(batch_size, rem)
            chunks.append(c)
            rem -= c

        raw_outputs = []
        for c in chunks:
            try:
                batch_enc = {
                    "input_ids": enc_single["input_ids"].repeat(c, 1),
                    "attention_mask": enc_single["attention_mask"].repeat(c, 1),
                }
                if llm.device == "cuda" or (isinstance(llm.device, str) and str(llm.device).startswith("cuda")):
                    torch.cuda.synchronize()
                with torch.inference_mode():
                    out = llm.model.generate(**batch_enc, generation_config=gcfg)
                if llm.device == "cuda" or (isinstance(llm.device, str) and str(llm.device).startswith("cuda")):
                    torch.cuda.synchronize()
                seqs = out.sequences if hasattr(out, "sequences") else out
                for i in range(c):
                    gen_ids = seqs[i, prompt_ids_len:].tolist()
                    cut = next((idx for idx, t in enumerate(gen_ids) if t in llm.eos), len(gen_ids))
                    gen_text = llm.tok.decode(gen_ids[:cut], skip_special_tokens=True)
                    raw_outputs.append((gen_text, cut == len(gen_ids)))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
                    print(f"    [OOM Warning] Batch size {c} OOM; falling back to 1 sample at a time", flush=True)
                    torch.cuda.empty_cache()
                    for _ in range(c):
                        with torch.inference_mode():
                            out_single = llm.model.generate(**enc_single, generation_config=gcfg)
                        seqs_single = out_single.sequences if hasattr(out_single, "sequences") else out_single
                        gen_ids = seqs_single[0, prompt_ids_len:].tolist()
                        cut = next((idx for idx, t in enumerate(gen_ids) if t in llm.eos), len(gen_ids))
                        gen_text = llm.tok.decode(gen_ids[:cut], skip_special_tokens=True)
                        raw_outputs.append((gen_text, cut == len(gen_ids)))
                else:
                    raise

        parsed_answers = []
        truncations = []
        insufficient_info = []
        insuf_in_reasoning_only = []

        for gen_text, hit_max in raw_outputs:
            has_final_answer = bool(re.search(r"(?:^|\n)Answer\s*:\s*[A-Za-z]", gen_text, re.MULTILINE))
            truncated = bool(hit_max and not has_final_answer)

            ans_sec = final_answer_section(gen_text)
            has_insuf = bool(re.search(r"\bINSUFFICIENT\s+INFORMATION\b", ans_sec, re.IGNORECASE))
            has_insuf_full = bool(re.search(r"\bINSUFFICIENT\s+INFORMATION\b", gen_text, re.IGNORECASE))
            insuf_reasoning = bool(has_insuf_full and not has_insuf)

            # Treat INSUFFICIENT INFORMATION in final answer section as "ABSTAIN"
            if has_insuf:
                ans = "ABSTAIN"
            else:
                ans = extract_final_answer_from_reasoning(gen_text, q["options"])
                if ans is None:
                    ans = parse_answer(gen_text, q["options"])
                if ans == "ABSTAIN":
                    has_insuf = True

            parsed_answers.append(ans)
            truncations.append(truncated)
            insufficient_info.append(has_insuf)
            insuf_in_reasoning_only.append(insuf_reasoning)

        return parsed_answers, truncations, insufficient_info, insuf_in_reasoning_only

    # ── 8. Run each dataset in sequence (Writing to <dataset>_shard<SC_SHARD>.jsonl) ───
    all_dataset_tasks = [
        ("medqa", medqa_run, "baseline"),
        ("pubmedqa", pubmedqa_run, "context"),
        ("unanswerable_v2", unans_run, "baseline"),
    ]
    dataset_tasks = [t for t in all_dataset_tasks if t[0] in active_sc_datasets]

    # Filter questions for this shard: index % SC_NUM_SHARDS == SC_SHARD
    shard_tasks = []
    for ds_name, questions, prompt_mode in dataset_tasks:
        shard_qs = [q for i, q in enumerate(questions) if i % SC_NUM_SHARDS == SC_SHARD]
        shard_tasks.append((ds_name, shard_qs, prompt_mode))

    total_q_all = sum(len(qs) for _, qs, _ in shard_tasks)
    timing_history = []
    eta_printed = False
    TIME_BUDGET_SEC = 11.0 * 3600.0  # 11 hours maximum
    time_budget_reached = False

    for ds_idx, (ds_name, questions, prompt_mode) in enumerate(shard_tasks):
        if time_budget_reached:
            print(f"[Shard {SC_SHARD}] Skipping {ds_name} due to time budget limit.", flush=True)
            break

        out_file = os.path.join(sc_out_dir, f"{ds_name}_shard{SC_SHARD}.jsonl")
        done_ids = set()
        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            done_ids.add(json.loads(line)["id"])
                        except Exception:
                            pass
        print(f"\n--- [Shard {SC_SHARD}] Running SC for {ds_name} ({len(questions)} on shard, {len(done_ids)} already done) ---", flush=True)

        with open(out_file, "a", encoding="utf-8") as out_fh:
            for q_idx, q in enumerate(questions, 1):
                if q["id"] in done_ids:
                    continue

                # Time budget check before starting each question
                elapsed_sec = time.time() - KERNEL_START_TIME
                if elapsed_sec >= TIME_BUDGET_SEC:
                    rem_in_ds = sum(1 for q_rem in questions if q_rem["id"] not in done_ids)
                    print(
                        f"\n{'='*75}\n"
                        f"[Shard {SC_SHARD}] TIME BUDGET LIMIT REACHED: {elapsed_sec / 3600.0:.2f} hours elapsed since start (limit 11.0h).\n"
                        f"[Shard {SC_SHARD}] Stopping before starting question {q_idx}/{len(questions)}. {rem_in_ds} question(s) remain unstarted in {ds_name}.\n"
                        f"[Shard {SC_SHARD}] Exiting cleanly so outputs are saved.\n"
                        f"{'='*75}\n",
                        flush=True,
                    )
                    time_budget_reached = True
                    break

                if prompt_mode == "context":
                    prompt_text = reasoning_context_prompt(q, max_chars=context_max_chars)
                    cur_max_new = context_max_new_tokens
                else:
                    prompt_text = reasoning_baseline_prompt(q)
                    cur_max_new = 1024

                # 1. Greedy pass (1 extra greedy generation per question)
                g_ans, g_trunc, g_insuf, g_insuf_reasoning, g_time, prompt_tokens = generate_single_greedy(
                    q, prompt_text, max_new_tokens=cur_max_new
                )

                # 2. Self-consistency sampled passes (SC_K samples)
                t0_sc = time.perf_counter()
                parsed_answers, truncations, insufficient_info, insuf_reasoning_only = generate_sc_k_samples(
                    q, prompt_text, SC_K, SC_TEMPERATURE, SC_TOP_P, SC_BATCH_SIZE, max_new_tokens=cur_max_new
                )
                sc_gen_time = round(time.perf_counter() - t0_sc, 4)

                # Count valid answers (including "ABSTAIN" as its own vote)
                valid_answers = [a for a in parsed_answers if a is not None]
                if valid_answers:
                    counts = collections.Counter(valid_answers)
                    majority_answer, majority_count = counts.most_common(1)[0]
                else:
                    majority_answer = None
                    majority_count = 0

                agreement = round(majority_count / SC_K, 4)
                distinct_answers = len(set(valid_answers))
                context_chars = len(q.get("context") or "")

                row = {
                    "id": q["id"],
                    "dataset": ds_name,
                    "category": q.get("category"),
                    "control": bool(q.get("control", False)),
                    "gold": q.get("answer"),
                    "source_id": q.get("source_id"),
                    "prompt_tokens": prompt_tokens,
                    "context_chars": context_chars,
                    "greedy_answer": g_ans,
                    "greedy_truncated": g_trunc,
                    "greedy_insufficient_info": g_insuf,
                    "greedy_insuf_in_reasoning_only": g_insuf_reasoning,
                    "greedy_time": g_time,
                    "parsed_answers": parsed_answers,
                    "truncations": truncations,
                    "insufficient_info": insufficient_info,
                    "insuf_in_reasoning_only": insuf_reasoning_only,
                    "majority_answer": majority_answer,
                    "agreement": agreement,
                    "distinct_answers": distinct_answers,
                    "generation_time": sc_gen_time,
                }

                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                done_ids.add(q["id"])

                # Also duplicate to repo-local dir if different from sc_out_dir
                local_dir = f"outputs/{split_subdir}/medpsy-4b"
                if os.path.exists(local_dir) and sc_out_dir != local_dir:
                    try:
                        local_file = os.path.join(local_dir, f"{ds_name}_shard{SC_SHARD}.jsonl")
                        with open(local_file, "a", encoding="utf-8") as loc_fh:
                            loc_fh.write(json.dumps(row) + "\n")
                            loc_fh.flush()
                    except Exception:
                        pass

                q_total_sec = g_time + sc_gen_time
                timing_history.append(q_total_sec)

                # Print ETA after first 5 questions of the first dataset
                if not eta_printed and ds_idx == 0 and len(timing_history) == 5:
                    eta_printed = True
                    mean_sec = sum(timing_history) / len(timing_history)
                    done_count_shard = 0
                    for d_n, _, _ in shard_tasks:
                        d_path = os.path.join(sc_out_dir, f"{d_n}_shard{SC_SHARD}.jsonl")
                        if os.path.exists(d_path):
                            try:
                                with open(d_path, "r", encoding="utf-8") as f_cnt:
                                    done_count_shard += sum(1 for line in f_cnt if line.strip())
                            except Exception:
                                pass
                    rem_q_shard = max(0, total_q_all - done_count_shard)
                    proj_hours = (mean_sec * rem_q_shard) / 3600.0
                    print(
                        f"\n{'='*75}\n"
                        f"[ETA - Shard {SC_SHARD}] Mean: {mean_sec:.2f} s/question (greedy + SC) | "
                        f"Projected time for remaining {rem_q_shard} questions on Shard {SC_SHARD} across all {len(shard_tasks)} datasets: {proj_hours:.2f} hours "
                        f"({proj_hours * 60:.1f} mins)\n"
                        f"{'='*75}\n",
                        flush=True,
                    )

                if q_idx % 10 == 0 or q_idx == len(questions):
                    print(
                        f"  [Shard {SC_SHARD}][{ds_name}] {q_idx}/{len(questions)} (id: {q['id']}) done: "
                        f"greedy={g_ans}, maj={majority_answer}, agree={agreement:.2f}, "
                        f"sc_time={sc_gen_time:.1f}s, greedy_time={g_time:.1f}s",
                        flush=True,
                    )

        print(f"[Shard {SC_SHARD}][{ds_name}] Complete! Shard output saved to {out_file}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"=== SC_MODE SHARD {SC_SHARD} EXECUTION FINISHED SUCCESSFULLY ===", flush=True)
    print(f"Outputs stored in: {sc_out_dir}", flush=True)
    print("=" * 80 + "\n", flush=True)
    sys.exit(0)


# ── BUILD_UNANS_V2_EVIDENCE mode: Pre-pass retrieval and reranking for unanswerable_v2 ─
if BUILD_UNANS_V2_EVIDENCE:
    if FINAL_MEDPSY or VAL_MEDPSY or PILOT or PARALLEL_GPUS or PROFILE or SC_MODE or UNANS_V2_SLM or MEDPSY_UNANS_V2_BASELINE or MEDPSY_PUBMEDQA_BASELINE or MEDPSY_PUBMEDQA_RAG or RAG_MEDPSY:
        sys.exit("ERROR: BUILD_UNANS_V2_EVIDENCE must run alone.")
    import shutil
    print("\n=== BUILD_UNANS_V2_EVIDENCE STARTUP ===", flush=True)

    v2_work = os.path.join(WORK, "work_unans_v2")
    os.makedirs(v2_work, exist_ok=True)
    v2_p1 = os.path.join(v2_work, "phase1")
    os.makedirs(v2_p1, exist_ok=True)

    val_src = os.path.join(CODE_DIR, "data", "unanswerable_v2", "val.jsonl")
    test_src = os.path.join(CODE_DIR, "data", "unanswerable_v2", "test.jsonl")
    if not os.path.exists(val_src) or not os.path.exists(test_src):
        val_src = os.path.abspath("data/unanswerable_v2/val.jsonl")
        test_src = os.path.abspath("data/unanswerable_v2/test.jsonl")
    if not os.path.exists(val_src) or not os.path.exists(test_src):
        sys.exit(f"ERROR: unanswerable_v2 files not found at {val_src} / {test_src}")

    shutil.copyfile(val_src, os.path.join(v2_p1, "validation.jsonl"))
    shutil.copyfile(test_src, os.path.join(v2_p1, "test.jsonl"))

    for n in (2, 3):
        src_dir = os.path.join(WORK, f"phase{n}")
        dst = os.path.join(v2_work, f"phase{n}")
        if os.path.isdir(src_dir) and not os.path.exists(dst):
            os.symlink(src_dir, dst)

    print("\n--- Running Phase 6 (Hybrid Retrieval) for unanswerable_v2 ---", flush=True)
    run([sys.executable, "-m", "src.phase06_retrieve", "--config", "configs/base.yaml", "--work", v2_work])
    print("\n--- Running Phase 7 (Reranking) for unanswerable_v2 ---", flush=True)
    run([sys.executable, "-m", "src.phase07_rerank", "--config", "configs/base.yaml", "--work", v2_work])

    v2_ev = os.path.join(v2_work, "phase7", "evidence.jsonl")
    if not os.path.exists(v2_ev):
        sys.exit(f"ERROR: Failed to produce v2 evidence at {v2_ev}")

    out_dir = "/kaggle/working/outputs/kaggle_build/unans_v2" if os.path.exists("/kaggle/working") else "outputs/kaggle_build/unans_v2"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(v2_ev, os.path.join(out_dir, "evidence.jsonl"))
    print(f"\n[BUILD_UNANS_V2_EVIDENCE] Success: {v2_ev} -> {os.path.join(out_dir, 'evidence.jsonl')}")
    print("=== BUILD_UNANS_V2_EVIDENCE COMPLETE ===\n", flush=True)
    sys.exit(0)


# ── UNANS_V2_SLM mode: 5 standard SLMs on unanswerable_v2 (baseline + RAG) ───
if UNANS_V2_SLM:
    if FINAL_MEDPSY or VAL_MEDPSY or PILOT or PARALLEL_GPUS or PROFILE or SC_MODE or BUILD_UNANS_V2_EVIDENCE or MEDPSY_UNANS_V2_BASELINE or MEDPSY_PUBMEDQA_BASELINE or MEDPSY_PUBMEDQA_RAG or RAG_MEDPSY:
        sys.exit("ERROR: UNANS_V2_SLM must run alone.")
    import shutil
    target_model = UNANS_MODEL
    print(f"\n=== UNANS_V2_SLM STARTUP (model={target_model}, datasets=unanswerable_v2, modes=baseline+rag) ===", flush=True)

    # 1. Setup work directory for unans_v2
    v2_work = os.path.join(WORK, "work_unans_v2")
    os.makedirs(v2_work, exist_ok=True)
    v2_p1 = os.path.join(v2_work, "phase1")
    os.makedirs(v2_p1, exist_ok=True)

    # Link/copy data/unanswerable_v2/{val,test}.jsonl into phase1/
    val_src = os.path.join(CODE_DIR, "data", "unanswerable_v2", "val.jsonl")
    test_src = os.path.join(CODE_DIR, "data", "unanswerable_v2", "test.jsonl")
    if not os.path.exists(val_src) or not os.path.exists(test_src):
        # Fallback to local repo paths if running outside /tmp
        val_src = os.path.abspath("data/unanswerable_v2/val.jsonl")
        test_src = os.path.abspath("data/unanswerable_v2/test.jsonl")
    if not os.path.exists(val_src) or not os.path.exists(test_src):
        sys.exit(f"ERROR: unanswerable_v2 files not found at {val_src} / {test_src}")

    shutil.copyfile(val_src, os.path.join(v2_p1, "validation.jsonl"))
    shutil.copyfile(test_src, os.path.join(v2_p1, "test.jsonl"))

    # Link phase2 (corpus) and phase3 (indexes) from build directory
    for n in (2, 3):
        src_dir = os.path.join(WORK, f"phase{n}")
        dst = os.path.join(v2_work, f"phase{n}")
        if os.path.isdir(src_dir) and not os.path.exists(dst):
            os.symlink(src_dir, dst)

    # 2. Obtain pre-computed Phase 6/7 evidence from the attached Kaggle dataset.
    #    Phases 6 and 7 are NEVER recomputed here — if the file is absent, exit immediately
    #    so no GPU quota is wasted.
    #    Discovery order:
    #      1. Canonical path  : /kaggle/input/medrag-unans-v2-evidence/evidence.jsonl
    #      2. Glob (slug)     : /kaggle/input/**/medrag-unans-v2-evidence/evidence.jsonl
    #      3. Glob (filtered) : /kaggle/input/**/evidence.jsonl  (paths containing "unans-v2")
    v2_ev = os.path.join(v2_work, "phase7", "evidence.jsonl")
    if not os.path.exists(v2_ev):
        _src_ev = None
        _src_label = None

        # Tier 1: canonical mount path
        _t1 = "/kaggle/input/medrag-unans-v2-evidence/evidence.jsonl"
        if os.path.exists(_t1):
            _src_ev, _src_label = _t1, "canonical"

        # Tier 2: glob on the dataset slug directory name
        if _src_ev is None:
            _t2_hits = glob.glob("/kaggle/input/**/medrag-unans-v2-evidence/evidence.jsonl", recursive=True)
            if _t2_hits:
                _src_ev, _src_label = _t2_hits[0], "glob-slug"

        # Tier 3: any evidence.jsonl whose path contains "unans-v2"
        if _src_ev is None:
            _t3_hits = [
                p for p in glob.glob("/kaggle/input/**/evidence.jsonl", recursive=True)
                if "unans-v2" in p
            ]
            if _t3_hits:
                _src_ev, _src_label = _t3_hits[0], f"glob-filtered ({_t3_hits[0]})"

        if _src_ev is None:
            sys.exit(
                "\n" + "=" * 80 + "\n"
                "ERROR [UNANS_V2_SLM]: Pre-computed evidence file not found.\n"
                "Searched (in order):\n"
                "  1. /kaggle/input/medrag-unans-v2-evidence/evidence.jsonl\n"
                "  2. /kaggle/input/**/medrag-unans-v2-evidence/evidence.jsonl\n"
                "  3. /kaggle/input/**/evidence.jsonl  (filtered to paths containing 'unans-v2')\n"
                "Resolution: attach the 'satyasainath1311/medrag-unans-v2-evidence' dataset\n"
                "  (Kaggle notebook -> Add Data -> Your Datasets -> medrag-unans-v2-evidence).\n"
                "Phases 6 and 7 are intentionally NOT recomputed here to avoid wasting GPU quota.\n"
                + "=" * 80 + "\n"
            )

        os.makedirs(os.path.join(v2_work, "phase7"), exist_ok=True)
        shutil.copyfile(_src_ev, v2_ev)
        print(f"Copied pre-computed evidence [{_src_label}]: {_src_ev} -> {v2_ev}", flush=True)
    else:
        print(f"Using existing v2 retrieval evidence at {v2_ev}", flush=True)

    # Integrity assertion: evidence must contain 400 rows (260 val + 140 test), 5 passages each.
    with open(v2_ev, "r", encoding="utf-8") as _ev_fh:
        _ev_rows = [json.loads(_l) for _l in _ev_fh if _l.strip()]
    _ev_ids = [r.get("id") for r in _ev_rows]
    _ev_val  = sum(1 for i in _ev_ids if i and "val"  in str(i))
    _ev_test = sum(1 for i in _ev_ids if i and "test" in str(i))
    assert len(_ev_rows) == 400, (
        f"ERROR: evidence.jsonl has {len(_ev_rows)} rows; expected 400 (260 val + 140 test)."
    )
    assert _ev_val == 260 and _ev_test == 140, (
        f"ERROR: evidence.jsonl id split mismatch: val={_ev_val}, test={_ev_test} (expected 260, 140)."
    )
    _bad_passages = [r.get("id") for r in _ev_rows if len(r.get("passages", [])) != 5]
    assert not _bad_passages, (
        f"ERROR: {len(_bad_passages)} rows in evidence.jsonl do not have exactly 5 passages: {_bad_passages[:5]}"
    )
    _ev_id_set = set(_ev_ids)
    print(
        f"Evidence integrity OK: {len(_ev_rows)} rows, "
        f"val={_ev_val}, test={_ev_test}, 5 passages each.",
        flush=True,
    )

    # 3. Run QA pipeline (baseline + RAG, split all_needed)
    qa_cmd = build_qa_cmd(
        models_list=[target_model],
        modes_list=["baseline", "rag"],
        split="all_needed",
        work_dir=v2_work,
        skip_format_check=True,
    )
    print(f"\n--- Running QA pipeline for {target_model} on unanswerable_v2 (baseline + RAG) ---", flush=True)
    run(qa_cmd)

    # 4. Save results to target outputs directory: outputs/kaggle_qa/unans_v2/<model>/
    out_dir = f"/kaggle/working/outputs/kaggle_qa/unans_v2/{target_model}" if os.path.exists("/kaggle/working") else f"outputs/kaggle_qa/unans_v2/{target_model}"
    os.makedirs(out_dir, exist_ok=True)

    p5_src = os.path.join(v2_work, "phase5", f"{target_model}.jsonl")
    p8_src = os.path.join(v2_work, "phase8", f"{target_model}.jsonl")
    if os.path.exists(p5_src):
        shutil.copyfile(p5_src, os.path.join(out_dir, f"baseline_{target_model}.jsonl"))
        print(f"Saved baseline output -> {os.path.join(out_dir, f'baseline_{target_model}.jsonl')}")
    if os.path.exists(p8_src):
        shutil.copyfile(p8_src, os.path.join(out_dir, f"rag_{target_model}.jsonl"))
        print(f"Saved RAG output -> {os.path.join(out_dir, f'rag_{target_model}.jsonl')}")

    # Diagnostic disk usage before cleanup
    if os.path.exists("/kaggle/working"):
        print("\n--- Disk usage of /kaggle/working before cleanup ---", flush=True)
        subprocess.run(["du", "-sh", "/kaggle/working"], check=False)
        subprocess.run(["du", "-shL", "/kaggle/working"], check=False)

    # Remove every symlink under /kaggle/working (do not follow links) and purge caches
    if os.path.exists("/kaggle/working"):
        symlinks_to_remove = []
        for root, dirs, files in os.walk("/kaggle/working", followlinks=False):
            for name in dirs + files:
                p = os.path.join(root, name)
                if os.path.islink(p):
                    symlinks_to_remove.append(p)
        for p in symlinks_to_remove:
            try:
                os.unlink(p)
            except OSError:
                pass
        for phase in ("phase2", "phase3"):
            for base in ("/kaggle/working/work_unans_v2", "/kaggle/working/work/work_unans_v2"):
                p = os.path.join(base, phase)
                if os.path.islink(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.cache/pip"), ignore_errors=True)

    if os.path.exists("/kaggle/working"):
        print("\n--- Disk usage of /kaggle/working after cleanup ---", flush=True)
        subprocess.run(["du", "-sh", "/kaggle/working"], check=False)

    print(f"\n=== UNANS_V2_SLM COMPLETE for {target_model} ===\n", flush=True)
    sys.exit(0)


def compute_stage_summary_metrics(rows, is_unans=False):
    n = len(rows)
    trunc = sum(1 for r in rows if r.get("truncated"))
    trunc_rate = (trunc / n * 100) if n else 0.0
    parsed = sum(1 for r in rows if r.get("parsed", True))
    parse_rate = (parsed / n * 100) if n else 0.0

    if is_unans:
        controls = [r for r in rows if not r.get("should_abstain")]
        n_eval = len(controls)
        strict_correct = sum(1 for r in controls if (not r.get("truncated")) and r.get("parsed") and r.get("correct"))
        fallback_correct = sum(1 for r in controls if r.get("correct"))
    else:
        n_eval = n
        strict_correct = sum(1 for r in rows if (not r.get("truncated")) and r.get("parsed") and r.get("correct"))
        fallback_correct = sum(1 for r in rows if r.get("correct"))

    strict_acc = (strict_correct / n_eval) if n_eval else 0.0
    fallback_acc = (fallback_correct / n_eval) if n_eval else 0.0

    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    s_per_q = (sum(secs) / len(secs)) if secs else 0.0

    return {
        "rows": n,
        "n_eval": n_eval,
        "strict_acc": strict_acc,
        "fallback_acc": fallback_acc,
        "trunc_rate": trunc_rate,
        "trunc_count": trunc,
        "parse_rate": parse_rate,
        "parsed_count": parsed,
        "s_per_q": s_per_q,
    }


# ── TWO-STAGE UNIFIED PUSH (Stage A2 + Stage D) ──────────────────────────────
if _is_two_stage_push:
    import glob
    import json
    import math
    import shutil
    import subprocess
    import threading

    NUM_SHARDS = 2
    print("\n" + "=" * 80, flush=True)
    print("=== TWO-STAGE UNIFIED PUSH: SEQUENTIAL EXECUTION ===", flush=True)
    print("  Stage A2: medpsy-4b on unanswerable_v2 baseline (reasoning prompt, val+test, N=400)", flush=True)
    print("  Stage D : qwen3-4b on unanswerable_v2 (baseline + RAG, val+test, N=400 each)", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Pre-Flight Frozen Data Audit (Mandatory Check)
    val_src, test_src, p1_test_src = verify_frozen_hashes_before_run(CODE_DIR, WORK)

    stage_summaries = {}

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE A2: medpsy-4b on unanswerable_v2 baseline (REASONING prompt, val+test, N=400)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print(">>> STARTING STAGE A2: medpsy-4b unanswerable_v2 baseline (reasoning prompt, N=400)", flush=True)
    print("=" * 80, flush=True)

    stage_a2_dir = "/kaggle/working/outputs/kaggle_qa/unans_v2/medpsy-4b" if os.path.exists("/kaggle/working") else "outputs/kaggle_qa/unans_v2/medpsy-4b"
    os.makedirs(stage_a2_dir, exist_ok=True)
    stage_a2_final = os.path.join(stage_a2_dir, "baseline_reasoning_medpsy-4b.jsonl")

    v2_work_a2 = os.path.join(WORK, "work_unans_v2_medpsy")
    os.makedirs(v2_work_a2, exist_ok=True)
    v2_p1_a2 = os.path.join(v2_work_a2, "phase1")
    os.makedirs(v2_p1_a2, exist_ok=True)
    shutil.copyfile(val_src, os.path.join(v2_p1_a2, "validation.jsonl"))
    shutil.copyfile(test_src, os.path.join(v2_p1_a2, "test.jsonl"))

    stage_a2_done = False
    if os.path.exists(stage_a2_final):
        try:
            with open(stage_a2_final) as fh:
                existing_a2 = [json.loads(l) for l in fh if l.strip()]
            if len(existing_a2) == 400:
                print(f"[RESUME] Stage A2 already completed ({len(existing_a2)} rows in {stage_a2_final}). Skipping.", flush=True)
                stage_a2_rows = existing_a2
                stage_a2_done = True
        except Exception:
            pass

    if not stage_a2_done:
        un_errors = {}
        def run_un_base_shard_a2(shard_idx):
            env = {"CUDA_VISIBLE_DEVICES": str(shard_idx)}
            try:
                cmd = build_qa_cmd(
                    models_list=["medpsy-4b"],
                    modes_list=["baseline"],
                    split="all_needed",
                    work_dir=v2_work_a2,
                    skip_format_check=True,
                    shard=shard_idx,
                    num_shards=NUM_SHARDS,
                    standard_prompt=False,  # Native reasoning prompt (<think>...</think> + Answer: <LETTER>)
                )
                print(f"\n[Stage A2 - Shard {shard_idx}] Starting MedPsy unans_v2 reasoning baseline on GPU {shard_idx}...", flush=True)
                run(cmd, extra_env=env)
            except Exception as exc:
                un_errors[shard_idx] = str(exc)

        threads = [threading.Thread(target=run_un_base_shard_a2, args=(i,), name=f"stage-a2-shard-{i}") for i in range(NUM_SHARDS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if un_errors:
            print("\nFATAL ERROR in Stage A2 shard execution:", un_errors, flush=True)
            sys.exit(1)

        print("\n=== MERGING STAGE A2 SHARDS ===", flush=True)
        p5_dir_a2 = os.path.join(v2_work_a2, "phase5")
        stage_a2_rows = []
        seen_ids = set()
        for sid in range(NUM_SHARDS):
            sp = os.path.join(p5_dir_a2, f"medpsy-4b_shard{sid}.jsonl")
            if os.path.exists(sp):
                with open(sp) as fh:
                    for l in fh:
                        if l.strip():
                            r = json.loads(l)
                            if r["id"] not in seen_ids:
                                stage_a2_rows.append(r)
                                seen_ids.add(r["id"])

        # Write to distinct file without touching baseline_medpsy-4b.jsonl
        with open(stage_a2_final, "w", encoding="utf-8") as fh:
            for r in stage_a2_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        print(f"  Stage A2 output saved ({len(stage_a2_rows)} rows) -> {stage_a2_final}", flush=True)

    val_cnt_a2 = sum(1 for r in stage_a2_rows if r.get("split") == "validation")
    test_cnt_a2 = sum(1 for r in stage_a2_rows if r.get("split") == "test")
    m_a2 = compute_stage_summary_metrics(stage_a2_rows, is_unans=True)
    summary_a2_line = (
        f"rows={m_a2['rows']} (val={val_cnt_a2}, test={test_cnt_a2}) | "
        f"strict_acc={m_a2['strict_acc']:.4f} (ctrl) | "
        f"fallback_acc={m_a2['fallback_acc']:.4f} (ctrl) | "
        f"trunc_rate={m_a2['trunc_rate']:.1f}% ({m_a2['trunc_count']}/{m_a2['rows']}) | "
        f"parse_rate={m_a2['parse_rate']:.1f}% ({m_a2['parsed_count']}/{m_a2['rows']}) | "
        f"{m_a2['s_per_q']:.2f} s/q"
    )
    stage_summaries["Stage A2 (MedPsy-4B reasoning baseline)"] = summary_a2_line
    print(f"\n[STAGE A2 SUMMARY] {summary_a2_line}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE D: qwen3-4b on unanswerable_v2 (baseline AND rag, val+test, N=400 each)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print(">>> STARTING STAGE D: qwen3-4b on unanswerable_v2 (baseline + RAG, N=400 each)", flush=True)
    print("=" * 80, flush=True)

    target_model_d = "qwen3-4b"
    stage_d_dir = f"/kaggle/working/outputs/kaggle_qa/unans_v2/{target_model_d}" if os.path.exists("/kaggle/working") else f"outputs/kaggle_qa/unans_v2/{target_model_d}"
    os.makedirs(stage_d_dir, exist_ok=True)
    stage_d_base_final = os.path.join(stage_d_dir, f"baseline_{target_model_d}.jsonl")
    stage_d_rag_final = os.path.join(stage_d_dir, f"rag_{target_model_d}.jsonl")

    v2_work_d = os.path.join(WORK, "work_unans_v2_qwen3_4b")
    os.makedirs(v2_work_d, exist_ok=True)
    v2_p1_d = os.path.join(v2_work_d, "phase1")
    os.makedirs(v2_p1_d, exist_ok=True)
    shutil.copyfile(val_src, os.path.join(v2_p1_d, "validation.jsonl"))
    shutil.copyfile(test_src, os.path.join(v2_p1_d, "test.jsonl"))

    for n in (2, 3):
        src_dir = os.path.join(WORK, f"phase{n}")
        dst = os.path.join(v2_work_d, f"phase{n}")
        if os.path.isdir(src_dir) and not os.path.exists(dst):
            os.symlink(src_dir, dst)

    v2_ev_d = os.path.join(v2_work_d, "phase7", "evidence.jsonl")
    if not os.path.exists(v2_ev_d):
        _src_ev = None
        _src_label = None
        _t1 = "/kaggle/input/medrag-unans-v2-evidence/evidence.jsonl"
        if os.path.exists(_t1):
            _src_ev, _src_label = _t1, "canonical"
        if _src_ev is None:
            _t2_hits = glob.glob("/kaggle/input/**/medrag-unans-v2-evidence/evidence.jsonl", recursive=True)
            if _t2_hits:
                _src_ev, _src_label = _t2_hits[0], "glob-slug"
        if _src_ev is None:
            _t3_hits = [p for p in glob.glob("/kaggle/input/**/evidence.jsonl", recursive=True) if "unans-v2" in p]
            if _t3_hits:
                _src_ev, _src_label = _t3_hits[0], f"glob-filtered ({_t3_hits[0]})"
        if _src_ev is None:
            _local_cands = [
                "outputs/kaggle_build/unans_v2/evidence.jsonl",
                os.path.join(WORK, "work_unans_v2", "phase7", "evidence.jsonl"),
            ]
            for lc in _local_cands:
                if os.path.exists(lc):
                    _src_ev, _src_label = lc, "local-fallback"
                    break
        if _src_ev is None:
            sys.exit(
                "\n" + "=" * 80 + "\n"
                "ERROR [STAGE D]: Pre-computed evidence file not found for unanswerable_v2.\n"
                "Searched /kaggle/input/medrag-unans-v2-evidence/evidence.jsonl and local paths.\n"
                + "=" * 80 + "\n"
            )

        os.makedirs(os.path.join(v2_work_d, "phase7"), exist_ok=True)
        shutil.copyfile(_src_ev, v2_ev_d)
        print(f"Copied pre-computed evidence [{_src_label}]: {_src_ev} -> {v2_ev_d}", flush=True)

    with open(v2_ev_d, "r", encoding="utf-8") as _ev_fh:
        _ev_rows = [json.loads(_l) for _l in _ev_fh if _l.strip()]
    _ev_ids = [r.get("id") for r in _ev_rows]
    _ev_val = sum(1 for i in _ev_ids if i and "val" in str(i))
    _ev_test = sum(1 for i in _ev_ids if i and "test" in str(i))
    assert len(_ev_rows) == 400 and _ev_val == 260 and _ev_test == 140, (
        f"ERROR: evidence.jsonl row mismatch: total={len(_ev_rows)}, val={_ev_val}, test={_ev_test}"
    )
    print(f"Evidence integrity OK: 400 rows (val=260, test=140), 5 passages each.", flush=True)

    stage_d_done = False
    if os.path.exists(stage_d_base_final) and os.path.exists(stage_d_rag_final):
        try:
            with open(stage_d_base_final) as fh_b, open(stage_d_rag_final) as fh_r:
                existing_d_b = [json.loads(l) for l in fh_b if l.strip()]
                existing_d_r = [json.loads(l) for l in fh_r if l.strip()]
            if len(existing_d_b) == 400 and len(existing_d_r) == 400:
                print(f"[RESUME] Stage D already completed (400 baseline, 400 RAG rows). Skipping.", flush=True)
                stage_d_base_rows = existing_d_b
                stage_d_rag_rows = existing_d_r
                stage_d_done = True
        except Exception:
            pass

    if not stage_d_done:
        qa_cmd_d = build_qa_cmd(
            models_list=[target_model_d],
            modes_list=["baseline", "rag"],
            split="all_needed",
            work_dir=v2_work_d,
            skip_format_check=True,
            standard_prompt=False,
        )
        print(f"\n--- Running QA pipeline for {target_model_d} on unanswerable_v2 (baseline + RAG) ---", flush=True)
        run(qa_cmd_d)

        p5_src_d = os.path.join(v2_work_d, "phase5", f"{target_model_d}.jsonl")
        p8_src_d = os.path.join(v2_work_d, "phase8", f"{target_model_d}.jsonl")
        assert os.path.exists(p5_src_d), f"Stage D baseline output missing at {p5_src_d}"
        assert os.path.exists(p8_src_d), f"Stage D RAG output missing at {p8_src_d}"

        shutil.copyfile(p5_src_d, stage_d_base_final)
        shutil.copyfile(p8_src_d, stage_d_rag_final)
        print(f"Saved baseline output -> {stage_d_base_final}", flush=True)
        print(f"Saved RAG output -> {stage_d_rag_final}", flush=True)

        with open(stage_d_base_final) as fh_b, open(stage_d_rag_final) as fh_r:
            stage_d_base_rows = [json.loads(l) for l in fh_b if l.strip()]
            stage_d_rag_rows = [json.loads(l) for l in fh_r if l.strip()]

    m_d_base = compute_stage_summary_metrics(stage_d_base_rows, is_unans=True)
    m_d_rag = compute_stage_summary_metrics(stage_d_rag_rows, is_unans=True)

    summary_d_base_line = (
        f"rows={m_d_base['rows']} (val=260, test=140) | "
        f"strict_acc={m_d_base['strict_acc']:.4f} (ctrl) | "
        f"fallback_acc={m_d_base['fallback_acc']:.4f} (ctrl) | "
        f"trunc_rate={m_d_base['trunc_rate']:.1f}% ({m_d_base['trunc_count']}/{m_d_base['rows']}) | "
        f"parse_rate={m_d_base['parse_rate']:.1f}% ({m_d_base['parsed_count']}/{m_d_base['rows']}) | "
        f"{m_d_base['s_per_q']:.2f} s/q"
    )
    summary_d_rag_line = (
        f"rows={m_d_rag['rows']} (val=260, test=140) | "
        f"strict_acc={m_d_rag['strict_acc']:.4f} (ctrl) | "
        f"fallback_acc={m_d_rag['fallback_acc']:.4f} (ctrl) | "
        f"trunc_rate={m_d_rag['trunc_rate']:.1f}% ({m_d_rag['trunc_count']}/{m_d_rag['rows']}) | "
        f"parse_rate={m_d_rag['parse_rate']:.1f}% ({m_d_rag['parsed_count']}/{m_d_rag['rows']}) | "
        f"{m_d_rag['s_per_q']:.2f} s/q"
    )

    stage_summaries["Stage D (Qwen3-4B Baseline)"] = summary_d_base_line
    stage_summaries["Stage D (Qwen3-4B RAG)"] = summary_d_rag_line

    print(f"\n[STAGE D (BASELINE) SUMMARY] {summary_d_base_line}", flush=True)
    print(f"[STAGE D (RAG) SUMMARY]      {summary_d_rag_line}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL COMBINED SUMMARY ACROSS ALL STAGES
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 96, flush=True)
    print("=== TWO-STAGE UNIFIED PUSH (STAGE A2 & STAGE D): FINAL COMBINED SUMMARY ===", flush=True)
    print("=" * 96, flush=True)
    print(f"{'Stage':<42} {'Rows':<8} {'Strict Acc (ctrl)':<18} {'Fallback Acc (ctrl)':<20} {'Trunc Rate':<12} {'Parse Rate':<12} {'Speed':<10}", flush=True)
    print("-" * 96, flush=True)
    for stg_label, metrics_obj in [
        ("Stage A2: MedPsy-4B unans_v2 (Reasoning)", m_a2),
        ("Stage D : Qwen3-4B unans_v2 Baseline", m_d_base),
        ("Stage D : Qwen3-4B unans_v2 RAG", m_d_rag),
    ]:
        strict_str = f"{metrics_obj['strict_acc']:.4f}"
        fb_str = f"{metrics_obj['fallback_acc']:.4f}"
        trunc_str = f"{metrics_obj['trunc_rate']:.1f}%"
        parse_str = f"{metrics_obj['parse_rate']:.1f}%"
        sq_str = f"{metrics_obj['s_per_q']:.2f} s/q"
        print(f"{stg_label:<42} {metrics_obj['rows']:<8} {strict_str:<18} {fb_str:<20} {trunc_str:<12} {parse_str:<12} {sq_str:<10}", flush=True)
    print("=" * 96, flush=True)
    total_q = m_a2['rows'] + m_d_base['rows'] + m_d_rag['rows']
    print(f"Total Questions Evaluated : {total_q} (400 Stage A2 + 400 Stage D Base + 400 Stage D RAG)", flush=True)
    print("All outputs safely committed to outputs/kaggle_qa/unans_v2/.", flush=True)
    print("=" * 96 + "\n", flush=True)

    # Post-run cleanup (caches only, after all stages finish successfully)
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.cache/pip"), ignore_errors=True)
    sys.exit(0)


# ── PAIR B THINKING CONTROL: QWEN3-4B-THINKING-2507 MEDQA BASELINE + RAG ───────
if QWEN3_4B_THINKING_MEDQA:
    import glob
    import json
    import math
    import shutil
    import subprocess
    import threading

    NUM_SHARDS = 2
    print("\n" + "=" * 80, flush=True)
    print("=== QWEN3-4B-THINKING-2507 MEDQA EVALUATION (Pair B Thinking Control) ===", flush=True)
    print("  Condition 1: MedQA test (N=500) baseline, reasoning_baseline_prompt [est. 5.22 h]", flush=True)
    print("  Condition 2: MedQA test (N=500) RAG, reasoning_rag_prompt            [est. 8.88 h]", flush=True)
    print("  Total estimated: 14.10 h (2-shard dual-GPU wall-clock ~7.05 h, fits in weekly quota)", flush=True)
    print("  PubMedQA and unanswerable_v2 conditions deferred to next quota window.", flush=True)
    print("=" * 80 + "\n", flush=True)

    print("MATCHING REQUIREMENTS AUDIT:", flush=True)
    print("  - Target Model       : Qwen/Qwen3-4B-Thinking-2507 (reasoning=True, float16)", flush=True)
    print("  - Prompt Functions   : Condition 1: reasoning_baseline_prompt | Condition 2: reasoning_rag_prompt", flush=True)
    print("  - Thinking Budget    : 1024 tokens thinking cap (max_new_tokens_reasoning=1024)", flush=True)
    print("  - Decoding Settings  : Greedy (temperature=0.0, do_sample=False)", flush=True)
    print("  - Benchmark Split    : Frozen MedQA test split (N=500 in phase1/test.jsonl)", flush=True)
    print("  - Evidence File      : Frozen Phase 7 MedCPT top-5 evidence (phase7/evidence.jsonl)", flush=True)
    print("  - Answer Parser      : extract_final_answer_from_reasoning (native think tags)", flush=True)
    print("  - Primary Rule       : Strict accuracy: (not truncated) and parsed and correct", flush=True)
    print("  - Secondary Rule     : With-fallback accuracy (raw correct field)", flush=True)
    print("\nDIFFERENCES BETWEEN ARMS (qwen3-4b-thinking-2507 vs medpsy-4b):", flush=True)
    print("  - Domain Training    : medpsy-4b received domain-specific medical instruction post-training; qwen3-4b-thinking-2507 is generic base thinking.", flush=True)
    print("  - Chat Template      : Both utilize Qwen chat template (<think>...</think> thinking tokens).", flush=True)
    print("  - Tokenizer          : Identical vocabulary, special tokens, and thinking tags.", flush=True)
    print("  - Logprob Re-feed    : Identical logprob re-feed mechanism after closed thinking blocks.", flush=True)
    print("=" * 80 + "\n", flush=True)

    # 1. Pre-Flight Frozen Data Audit (Mandatory Check)
    val_src, test_src, p1_test_src = verify_frozen_hashes_before_run(CODE_DIR, WORK)

    # Canonical v3 destination directories
    out_base = (
        "/kaggle/working/outputs/v3/qwen3-4b-thinking-2507"
        if os.path.exists("/kaggle/working")
        else "outputs/v3/qwen3-4b-thinking-2507"
    )
    c1_dir = os.path.join(out_base, "C1")
    os.makedirs(c1_dir, exist_ok=True)
    c1_final = os.path.join(c1_dir, "qwen3-4b-thinking-2507.jsonl")

    c2_dir = os.path.join(out_base, "C2")
    os.makedirs(c2_dir, exist_ok=True)
    c2_final = os.path.join(c2_dir, "qwen3-4b-thinking-2507.jsonl")

    qa_mirror_dir = (
        "/kaggle/working/outputs/kaggle_qa/qwen3_4b_thinking/work"
        if os.path.exists("/kaggle/working")
        else "outputs/kaggle_qa/qwen3_4b_thinking/work"
    )
    os.makedirs(os.path.join(qa_mirror_dir, "phase5"), exist_ok=True)
    os.makedirs(os.path.join(qa_mirror_dir, "phase8"), exist_ok=True)

    condition_summaries = {}

    # ══════════════════════════════════════════════════════════════════════════
    # CONDITION 1: MedQA test (N=500) baseline, reasoning_baseline_prompt
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print(">>> STARTING CONDITION 1: MedQA test (N=500) baseline (reasoning_baseline_prompt)", flush=True)
    print("=" * 80, flush=True)

    c1_done = False
    if os.path.exists(c1_final):
        try:
            with open(c1_final) as fh:
                existing_c1 = [json.loads(l) for l in fh if l.strip()]
            if len(existing_c1) == 500:
                print(f"[RESUME] Condition 1 already completed ({len(existing_c1)} rows in {c1_final}). Skipping.", flush=True)
                c1_rows = existing_c1
                c1_done = True
        except Exception:
            pass

    if not c1_done:
        c1_errors = {}
        def run_c1_shard(shard_idx):
            env = {"CUDA_VISIBLE_DEVICES": str(shard_idx)}
            try:
                cmd = build_qa_cmd(
                    models_list=["qwen3-4b-thinking-2507"],
                    modes_list=["baseline"],
                    datasets="medqa",
                    split="test",
                    work_dir=WORK,
                    skip_format_check=True,
                    shard=shard_idx,
                    num_shards=NUM_SHARDS,
                    standard_prompt=False,
                )
                print(f"\n[Condition 1 - Shard {shard_idx}] Starting MedQA baseline on GPU {shard_idx}...", flush=True)
                run(cmd, extra_env=env)
            except Exception as exc:
                c1_errors[shard_idx] = str(exc)

        threads = [threading.Thread(target=run_c1_shard, args=(i,), name=f"c1-shard-{i}") for i in range(NUM_SHARDS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if c1_errors:
            print("\nFATAL ERROR in Condition 1 shard execution:", c1_errors, flush=True)
            sys.exit(1)

        print("\n=== MERGING CONDITION 1 SHARDS ===", flush=True)
        p5_dir = os.path.join(WORK, "phase5")
        c1_rows = []
        seen_ids_c1 = set()
        for sid in range(NUM_SHARDS):
            sp = os.path.join(p5_dir, f"qwen3-4b-thinking-2507_shard{sid}.jsonl")
            if os.path.exists(sp):
                with open(sp) as fh:
                    for l in fh:
                        if l.strip():
                            r = json.loads(l)
                            if r["id"] not in seen_ids_c1:
                                c1_rows.append(r)
                                seen_ids_c1.add(r["id"])

        if len(c1_rows) != 500:
            print(f"WARNING: Expected 500 rows for Condition 1, got {len(c1_rows)}!", flush=True)

        with open(c1_final, "w", encoding="utf-8") as fh:
            for r in c1_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        print(f"  Condition 1 canonical output saved ({len(c1_rows)} rows) -> {c1_final}", flush=True)
        shutil.copyfile(c1_final, os.path.join(qa_mirror_dir, "phase5", "qwen3-4b-thinking-2507.jsonl"))

    m_c1 = compute_stage_summary_metrics(c1_rows, is_unans=False)
    summary_c1_line = (
        f"rows={m_c1['rows']} | "
        f"strict_acc={m_c1['strict_acc']:.4f} | "
        f"fallback_acc={m_c1['fallback_acc']:.4f} | "
        f"trunc_rate={m_c1['trunc_rate']:.1f}% ({m_c1['trunc_count']}/{m_c1['rows']}) | "
        f"parse_rate={m_c1['parse_rate']:.1f}% ({m_c1['parsed_count']}/{m_c1['rows']}) | "
        f"{m_c1['s_per_q']:.2f} s/q"
    )
    condition_summaries["Condition 1 (MedQA Baseline)"] = summary_c1_line
    print(f"\n[CONDITION 1 SUMMARY] {summary_c1_line}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # CONDITION 2: MedQA test (N=500) RAG, reasoning_rag_prompt
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print(">>> STARTING CONDITION 2: MedQA test (N=500) RAG (reasoning_rag_prompt)", flush=True)
    print("=" * 80, flush=True)

    # Verify Phase 7 evidence file resolution
    ev_path = os.path.join(WORK, "phase7", "evidence.jsonl")
    if not os.path.exists(ev_path):
        ev_hits = glob.glob("/kaggle/input/**/phase7/evidence.jsonl", recursive=True)
        if ev_hits:
            ev_path = ev_hits[0]
            os.makedirs(os.path.join(WORK, "phase7"), exist_ok=True)
            dst_ev = os.path.join(WORK, "phase7", "evidence.jsonl")
            if not os.path.exists(dst_ev):
                shutil.copyfile(ev_path, dst_ev)
                print(f"  Copied evidence to work dir: {dst_ev}", flush=True)
        else:
            for fallback_ev in [
                "outputs/kaggle_build/work/phase7/evidence.jsonl",
                os.path.join(CODE_DIR, "outputs", "kaggle_build", "work", "phase7", "evidence.jsonl"),
            ]:
                if os.path.exists(fallback_ev):
                    os.makedirs(os.path.join(WORK, "phase7"), exist_ok=True)
                    shutil.copyfile(fallback_ev, os.path.join(WORK, "phase7", "evidence.jsonl"))
                    print(f"  Copied local evidence to work dir from: {fallback_ev}", flush=True)
                    break

    c2_done = False
    if os.path.exists(c2_final):
        try:
            with open(c2_final) as fh:
                existing_c2 = [json.loads(l) for l in fh if l.strip()]
            if len(existing_c2) == 500:
                print(f"[RESUME] Condition 2 already completed ({len(existing_c2)} rows in {c2_final}). Skipping.", flush=True)
                c2_rows = existing_c2
                c2_done = True
        except Exception:
            pass

    if not c2_done:
        c2_errors = {}
        def run_c2_shard(shard_idx):
            env = {"CUDA_VISIBLE_DEVICES": str(shard_idx)}
            try:
                cmd = build_qa_cmd(
                    models_list=["qwen3-4b-thinking-2507"],
                    modes_list=["rag"],
                    datasets="medqa",
                    split="test",
                    work_dir=WORK,
                    skip_format_check=True,
                    shard=shard_idx,
                    num_shards=NUM_SHARDS,
                    standard_prompt=False,
                )
                print(f"\n[Condition 2 - Shard {shard_idx}] Starting MedQA RAG on GPU {shard_idx}...", flush=True)
                run(cmd, extra_env=env)
            except Exception as exc:
                c2_errors[shard_idx] = str(exc)

        threads = [threading.Thread(target=run_c2_shard, args=(i,), name=f"c2-shard-{i}") for i in range(NUM_SHARDS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if c2_errors:
            print("\nFATAL ERROR in Condition 2 shard execution:", c2_errors, flush=True)
            sys.exit(1)

        print("\n=== MERGING CONDITION 2 SHARDS ===", flush=True)
        p8_dir = os.path.join(WORK, "phase8")
        c2_rows = []
        seen_ids_c2 = set()
        for sid in range(NUM_SHARDS):
            sp = os.path.join(p8_dir, f"qwen3-4b-thinking-2507_shard{sid}.jsonl")
            if os.path.exists(sp):
                with open(sp) as fh:
                    for l in fh:
                        if l.strip():
                            r = json.loads(l)
                            if r["id"] not in seen_ids_c2:
                                c2_rows.append(r)
                                seen_ids_c2.add(r["id"])

        if len(c2_rows) != 500:
            print(f"WARNING: Expected 500 rows for Condition 2, got {len(c2_rows)}!", flush=True)

        with open(c2_final, "w", encoding="utf-8") as fh:
            for r in c2_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        print(f"  Condition 2 canonical output saved ({len(c2_rows)} rows) -> {c2_final}", flush=True)
        shutil.copyfile(c2_final, os.path.join(qa_mirror_dir, "phase8", "qwen3-4b-thinking-2507.jsonl"))

    m_c2 = compute_stage_summary_metrics(c2_rows, is_unans=False)
    summary_c2_line = (
        f"rows={m_c2['rows']} | "
        f"strict_acc={m_c2['strict_acc']:.4f} | "
        f"fallback_acc={m_c2['fallback_acc']:.4f} | "
        f"trunc_rate={m_c2['trunc_rate']:.1f}% ({m_c2['trunc_count']}/{m_c2['rows']}) | "
        f"parse_rate={m_c2['parse_rate']:.1f}% ({m_c2['parsed_count']}/{m_c2['rows']}) | "
        f"{m_c2['s_per_q']:.2f} s/q"
    )
    condition_summaries["Condition 2 (MedQA Full RAG)"] = summary_c2_line
    print(f"\n[CONDITION 2 SUMMARY] {summary_c2_line}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL COMBINED SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 96, flush=True)
    print("=== QWEN3-4B-THINKING-2507 MEDQA EVALUATION FINAL SUMMARY ===", flush=True)
    print("=" * 96, flush=True)
    print(f"{'Condition':<36} {'N':<6} {'Strict Acc':<12} {'Fallback Acc':<14} {'Trunc Rate':<12} {'Parse Rate':<12} {'Latency':<10}", flush=True)
    print("-" * 96, flush=True)
    for cond_title, metrics_obj in [
        ("Condition 1: MedQA Baseline", m_c1),
        ("Condition 2: MedQA Full RAG", m_c2),
    ]:
        strict_str = f"{metrics_obj['strict_acc']:.4f}"
        fb_str = f"{metrics_obj['fallback_acc']:.4f}"
        trunc_str = f"{metrics_obj['trunc_rate']:.1f}%"
        parse_str = f"{metrics_obj['parse_rate']:.1f}%"
        sq_str = f"{metrics_obj['s_per_q']:.2f} s/q"
        print(f"{cond_title:<36} {metrics_obj['rows']:<6} {strict_str:<12} {fb_str:<14} {trunc_str:<12} {parse_str:<12} {sq_str:<10}", flush=True)
    print("=" * 96, flush=True)
    print(f"Total MedQA Questions Evaluated: {m_c1['rows'] + m_c2['rows']} (500 Baseline + 500 RAG)", flush=True)
    print(f"Canonical Outputs Saved: {c1_final} and {c2_final}", flush=True)
    print("=" * 96 + "\n", flush=True)

    # Post-run cleanup (caches only, after all conditions finish successfully)
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.cache/pip"), ignore_errors=True)
    sys.exit(0)


# ── MEDPSY UNIFIED OVERNIGHT PUSH (Stages A, B, C) ───────────────────────────
if _is_medpsy_overnight:
    import math
    import shutil
    import threading

    NUM_SHARDS = 2
    print("\n" + "=" * 80, flush=True)
    print("=== MEDPSY-4B UNIFIED OVERNIGHT RUNNER: 3 STAGES SEQUENTIAL EXECUTION ===", flush=True)
    print("  Stage A: unanswerable_v2 baseline (Phase 5 baseline prompt, val+test, N=400)", flush=True)
    print("  Stage B: PubMedQA closed-book baseline (test, N=500)", flush=True)
    print("  Stage C: PubMedQA full RAG (test, N=500, top-5 MedCPT evidence)", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Pre-Flight Frozen Data Audit (Mandatory Check)
    val_src, test_src, p1_test_src = verify_frozen_hashes_before_run(CODE_DIR, WORK)

    # ── Ensure WORK/phase1/test.jsonl uses the frozen repo bytes ───────────────
    # WORK/phase1 may be a symlink to /kaggle/input/**/phase1 (read-only filesystem).
    # Replace symlink with a real directory so test.jsonl can be safely overwritten.
    work_p1 = os.path.join(WORK, "phase1")
    if os.path.islink(work_p1):
        target_p1 = os.readlink(work_p1)
        os.unlink(work_p1)
        os.makedirs(work_p1, exist_ok=True)
        if os.path.isdir(target_p1):
            for item in os.listdir(target_p1):
                if item != "test.jsonl":
                    src_item = os.path.join(target_p1, item)
                    dst_item = os.path.join(work_p1, item)
                    try:
                        os.symlink(src_item, dst_item)
                    except OSError:
                        shutil.copy2(src_item, dst_item)
    elif not os.path.exists(work_p1):
        os.makedirs(work_p1, exist_ok=True)

    target_work_test = os.path.join(work_p1, "test.jsonl")
    if os.path.islink(target_work_test) or os.path.exists(target_work_test):
        try:
            os.unlink(target_work_test)
        except OSError:
            pass
    shutil.copyfile(p1_test_src, target_work_test)

    import hashlib
    h_setup = hashlib.sha256()
    with open(target_work_test, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h_setup.update(chunk)
    copied_sha = h_setup.hexdigest()
    print(f"\n[FROZEN DATA SETUP] Copied frozen test.jsonl over {target_work_test}:", flush=True)
    print(f"  Source path : {p1_test_src}", flush=True)
    print(f"  SHA-256     : {copied_sha}\n", flush=True)

    # ── Resolve and audit PubMedQA phase1 validation file context integrity ────
    val_p1_candidates = [
        os.path.join(CODE_DIR, "outputs", "kaggle_build", "work", "phase1", "validation.jsonl"),
        os.path.abspath("outputs/kaggle_build/work/phase1/validation.jsonl"),
        os.path.join(work_p1, "validation.jsonl"),
    ]
    val_p1_src = next((p for p in val_p1_candidates if os.path.exists(p)), None)
    if not val_p1_src:
        input_hits = glob.glob("/kaggle/input/**/phase1/validation.jsonl", recursive=True)
        if input_hits:
            val_p1_src = input_hits[0]
    if not val_p1_src or not os.path.exists(val_p1_src):
        sys.exit("FATAL: PubMedQA phase1/validation.jsonl not found in code dir, work dir or /kaggle/input/**")

    # Overwrite WORK/phase1/validation.jsonl with frozen repo copy as well
    target_work_val = os.path.join(work_p1, "validation.jsonl")
    if os.path.islink(target_work_val) or os.path.exists(target_work_val):
        try:
            os.unlink(target_work_val)
        except OSError:
            pass
    shutil.copyfile(val_p1_src, target_work_val)

    with open(val_p1_src, "r", encoding="utf-8") as fh:
        val_p1_rows = [json.loads(line) for line in fh if line.strip()]

    pm_val_rows = [r for r in val_p1_rows if r.get("id", "").startswith("pubmedqa")]
    missing_ctx = [r for r in pm_val_rows if not r.get("context")]
    if missing_ctx:
        sys.exit(
            f"FATAL: {len(missing_ctx)}/{len(pm_val_rows)} PubMedQA validation rows in {val_p1_src} "
            f"lack a non-empty 'context' field! Example ID: {missing_ctx[0].get('id')}"
        )

    h_val = hashlib.sha256()
    with open(val_p1_src, "rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h_val.update(chunk)
    val_p1_sha = h_val.hexdigest()

    print(f"Pre-flight Validation Context Audit [{val_p1_src}]:", flush=True)
    print(f"  Path used   : {val_p1_src}", flush=True)
    print(f"  SHA-256     : {val_p1_sha}", flush=True)
    print(f"  PubMedQA Rows Checked: {len(pm_val_rows)} (expect 200)", flush=True)
    print(f"  [PASS] all have context\n", flush=True)

    def compute_stage_summary_metrics(rows, is_unans=False):
        n = len(rows)
        trunc = sum(1 for r in rows if r.get("truncated"))
        trunc_rate = (trunc / n * 100) if n else 0.0
        
        if is_unans:
            controls = [r for r in rows if not r.get("should_abstain")]
            n_eval = len(controls)
            strict_correct = sum(1 for r in controls if (not r.get("truncated")) and r.get("parsed") and r.get("correct"))
            fallback_correct = sum(1 for r in controls if r.get("correct"))
        else:
            n_eval = n
            strict_correct = sum(1 for r in rows if (not r.get("truncated")) and r.get("parsed") and r.get("correct"))
            fallback_correct = sum(1 for r in rows if r.get("correct"))
        
        strict_acc = (strict_correct / n_eval) if n_eval else 0.0
        fallback_acc = (fallback_correct / n_eval) if n_eval else 0.0
        
        secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
        s_per_q = (sum(secs) / len(secs)) if secs else 0.0
        
        return {
            "rows": n,
            "n_eval": n_eval,
            "strict_acc": strict_acc,
            "fallback_acc": fallback_acc,
            "trunc_rate": trunc_rate,
            "trunc_count": trunc,
            "s_per_q": s_per_q,
        }

    stage_summaries = {}

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE A: unanswerable_v2 baseline (Phase 5 prompt, val+test, N=400)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print(">>> STARTING STAGE A: unanswerable_v2 baseline (Phase 5 baseline prompt, N=400)", flush=True)
    print("=" * 80, flush=True)

    stage_a_dir = "/kaggle/working/outputs/kaggle_qa/unans_v2/medpsy-4b" if os.path.exists("/kaggle/working") else "outputs/kaggle_qa/unans_v2/medpsy-4b"
    os.makedirs(stage_a_dir, exist_ok=True)
    stage_a_final = os.path.join(stage_a_dir, "baseline_medpsy-4b.jsonl")

    v2_work = os.path.join(WORK, "work_unans_v2")
    os.makedirs(v2_work, exist_ok=True)
    v2_p1 = os.path.join(v2_work, "phase1")
    os.makedirs(v2_p1, exist_ok=True)
    shutil.copyfile(val_src, os.path.join(v2_p1, "validation.jsonl"))
    shutil.copyfile(test_src, os.path.join(v2_p1, "test.jsonl"))

    stage_a_done = False
    if os.path.exists(stage_a_final):
        try:
            with open(stage_a_final) as fh:
                existing_a = [json.loads(l) for l in fh if l.strip()]
            if len(existing_a) == 400:
                print(f"[RESUME] Stage A already completed ({len(existing_a)} rows in {stage_a_final}). Skipping.", flush=True)
                stage_a_rows = existing_a
                stage_a_done = True
        except Exception:
            pass

    if not stage_a_done:
        un_errors = {}
        def run_un_base_shard(shard_idx):
            env = {"CUDA_VISIBLE_DEVICES": str(shard_idx)}
            try:
                cmd = build_qa_cmd(
                    models_list=["medpsy-4b"],
                    modes_list=["baseline"],
                    split="all_needed",
                    work_dir=v2_work,
                    skip_format_check=True,
                    shard=shard_idx,
                    num_shards=NUM_SHARDS,
                    standard_prompt=True,
                )
                print(f"\n[Stage A - Shard {shard_idx}] Starting MedPsy unans_v2 baseline on GPU {shard_idx}...", flush=True)
                run(cmd, extra_env=env)
            except Exception as exc:
                un_errors[shard_idx] = str(exc)

        threads = [threading.Thread(target=run_un_base_shard, args=(i,), name=f"stage-a-shard-{i}") for i in range(NUM_SHARDS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if un_errors:
            print("\nFATAL ERROR in Stage A shard execution:", un_errors, flush=True)
            sys.exit(1)

        print("\n=== MERGING STAGE A SHARDS ===", flush=True)
        p5_dir = os.path.join(v2_work, "phase5")
        stage_a_rows = []
        seen_ids = set()
        for sid in range(NUM_SHARDS):
            sp = os.path.join(p5_dir, f"medpsy-4b_shard{sid}.jsonl")
            if os.path.exists(sp):
                with open(sp) as fh:
                    for l in fh:
                        if l.strip():
                            r = json.loads(l)
                            if r["id"] not in seen_ids:
                                stage_a_rows.append(r)
                                seen_ids.add(r["id"])

        # Write ONLY ONE file under outputs/kaggle_qa/
        with open(stage_a_final, "w", encoding="utf-8") as fh:
            for r in stage_a_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        print(f"  Stage A output saved ({len(stage_a_rows)} rows) -> {stage_a_final}", flush=True)

    val_cnt = sum(1 for r in stage_a_rows if r.get("split") == "validation")
    test_cnt = sum(1 for r in stage_a_rows if r.get("split") == "test")
    m_a = compute_stage_summary_metrics(stage_a_rows, is_unans=True)
    summary_a_line = f"rows={m_a['rows']} (val={val_cnt}, test={test_cnt}) | strict_acc={m_a['strict_acc']:.4f} (ctrl) | fallback_acc={m_a['fallback_acc']:.4f} (ctrl) | trunc_rate={m_a['trunc_rate']:.1f}% ({m_a['trunc_count']}/{m_a['rows']}) | {m_a['s_per_q']:.2f} s/q"
    stage_summaries["Stage A (unanswerable_v2 baseline)"] = summary_a_line
    print(f"\n[STAGE A SUMMARY] {summary_a_line}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════
    # STAGE B: PubMedQA Closed-Book Baseline (test, N=500)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print(">>> STARTING STAGE B: PubMedQA closed-book baseline (test, N=500)", flush=True)
    print("=" * 80, flush=True)

    stage_b_dir = "/kaggle/working/outputs/kaggle_qa/medpsy_pubmedqa_baseline" if os.path.exists("/kaggle/working") else "outputs/kaggle_qa/medpsy_pubmedqa_baseline"
    os.makedirs(stage_b_dir, exist_ok=True)
    stage_b_final = os.path.join(stage_b_dir, "medpsy-4b_pubmedqa_baseline.jsonl")

    pm_base_work = os.path.join(WORK, "work_pubmedqa_baseline")
    os.makedirs(pm_base_work, exist_ok=True)
    pm_base_p1 = os.path.join(pm_base_work, "phase1")
    os.makedirs(pm_base_p1, exist_ok=True)
    shutil.copyfile(p1_test_src, os.path.join(pm_base_p1, "test.jsonl"))
    shutil.copyfile(val_p1_src, os.path.join(pm_base_p1, "validation.jsonl"))

    stage_b_done = False
    if os.path.exists(stage_b_final):
        try:
            with open(stage_b_final) as fh:
                existing_b = [json.loads(l) for l in fh if l.strip()]
            if len(existing_b) == 500:
                print(f"[RESUME] Stage B already completed ({len(existing_b)} rows in {stage_b_final}). Skipping.", flush=True)
                stage_b_rows = existing_b
                stage_b_done = True
        except Exception:
            pass

    if not stage_b_done:
        pm_errors = {}
        def run_pm_base_shard(shard_idx):
            env = {"CUDA_VISIBLE_DEVICES": str(shard_idx)}
            try:
                cmd = build_qa_cmd(
                    models_list=["medpsy-4b"],
                    modes_list=["baseline"],
                    datasets="pubmedqa",
                    split="test",
                    work_dir=pm_base_work,
                    skip_format_check=True,
                    shard=shard_idx,
                    num_shards=NUM_SHARDS,
                )
                print(f"\n[Stage B - Shard {shard_idx}] Starting PubMedQA baseline on GPU {shard_idx}...", flush=True)
                run(cmd, extra_env=env)
            except Exception as exc:
                pm_errors[shard_idx] = str(exc)

        threads = [threading.Thread(target=run_pm_base_shard, args=(i,), name=f"stage-b-shard-{i}") for i in range(NUM_SHARDS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if pm_errors:
            print("\nFATAL ERROR in Stage B shard execution:", pm_errors, flush=True)
            sys.exit(1)

        print("\n=== MERGING STAGE B SHARDS ===", flush=True)
        p5_dir = os.path.join(pm_base_work, "phase5")
        stage_b_rows = []
        seen_ids = set()
        for sid in range(NUM_SHARDS):
            sp = os.path.join(p5_dir, f"medpsy-4b_shard{sid}.jsonl")
            if os.path.exists(sp):
                with open(sp) as fh:
                    for l in fh:
                        if l.strip():
                            r = json.loads(l)
                            if r["id"] not in seen_ids and r.get("dataset") == "pubmedqa":
                                stage_b_rows.append(r)
                                seen_ids.add(r["id"])

        with open(stage_b_final, "w", encoding="utf-8") as fh:
            for r in stage_b_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        print(f"  Stage B output saved ({len(stage_b_rows)} rows) -> {stage_b_final}", flush=True)

    m_b = compute_stage_summary_metrics(stage_b_rows, is_unans=False)
    summary_b_line = f"rows={m_b['rows']} (test) | strict_acc={m_b['strict_acc']:.4f} | fallback_acc={m_b['fallback_acc']:.4f} | trunc_rate={m_b['trunc_rate']:.1f}% ({m_b['trunc_count']}/{m_b['rows']}) | {m_b['s_per_q']:.2f} s/q"
    stage_summaries["Stage B (PubMedQA baseline)"] = summary_b_line
    print(f"\n[STAGE B SUMMARY] {summary_b_line}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════
    # STAGE C: PubMedQA Full RAG (test, N=500, top-5 MedCPT evidence)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80, flush=True)
    print(">>> STARTING STAGE C: PubMedQA full RAG (test, N=500)", flush=True)
    print("=" * 80, flush=True)

    ev_path = os.path.join(WORK, "phase7", "evidence.jsonl")
    if not os.path.exists(ev_path):
        ev_hits = glob.glob("/kaggle/input/**/phase7/evidence.jsonl", recursive=True)
        if ev_hits:
            ev_path = ev_hits[0]
    if not os.path.exists(ev_path):
        sys.exit(f"FATAL ERROR: Phase 7 evidence not found at {ev_path}. Attach medrag-build output.")

    stage_c_dir = "/kaggle/working/outputs/kaggle_qa/medpsy_pubmedqa_rag" if os.path.exists("/kaggle/working") else "outputs/kaggle_qa/medpsy_pubmedqa_rag"
    os.makedirs(stage_c_dir, exist_ok=True)
    stage_c_final = os.path.join(stage_c_dir, "medpsy-4b_pubmedqa_rag.jsonl")

    pm_rag_work = os.path.join(WORK, "work_pubmedqa_rag")
    os.makedirs(pm_rag_work, exist_ok=True)
    pm_rag_p1 = os.path.join(pm_rag_work, "phase1")
    pm_rag_p7 = os.path.join(pm_rag_work, "phase7")
    os.makedirs(pm_rag_p1, exist_ok=True)
    os.makedirs(pm_rag_p7, exist_ok=True)
    shutil.copyfile(p1_test_src, os.path.join(pm_rag_p1, "test.jsonl"))
    shutil.copyfile(ev_path, os.path.join(pm_rag_p7, "evidence.jsonl"))
    shutil.copyfile(val_p1_src, os.path.join(pm_rag_p1, "validation.jsonl"))

    stage_c_done = False
    if os.path.exists(stage_c_final):
        try:
            with open(stage_c_final) as fh:
                existing_c = [json.loads(l) for l in fh if l.strip()]
            if len(existing_c) == 500:
                print(f"[RESUME] Stage C already completed ({len(existing_c)} rows in {stage_c_final}). Skipping.", flush=True)
                stage_c_rows = existing_c
                stage_c_done = True
        except Exception:
            pass

    if not stage_c_done:
        rag_errors = {}
        def run_pm_rag_shard(shard_idx):
            env = {"CUDA_VISIBLE_DEVICES": str(shard_idx)}
            try:
                cmd = build_qa_cmd(
                    models_list=["medpsy-4b"],
                    modes_list=["rag"],
                    datasets="pubmedqa",
                    split="test",
                    work_dir=pm_rag_work,
                    skip_format_check=True,
                    shard=shard_idx,
                    num_shards=NUM_SHARDS,
                )
                print(f"\n[Stage C - Shard {shard_idx}] Starting PubMedQA RAG on GPU {shard_idx}...", flush=True)
                run(cmd, extra_env=env)
            except Exception as exc:
                rag_errors[shard_idx] = str(exc)

        threads = [threading.Thread(target=run_pm_rag_shard, args=(i,), name=f"stage-c-shard-{i}") for i in range(NUM_SHARDS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if rag_errors:
            print("\nFATAL ERROR in Stage C shard execution:", rag_errors, flush=True)
            sys.exit(1)

        print("\n=== MERGING STAGE C SHARDS ===", flush=True)
        p8_dir = os.path.join(pm_rag_work, "phase8")
        stage_c_rows = []
        seen_ids = set()
        for sid in range(NUM_SHARDS):
            sp = os.path.join(p8_dir, f"medpsy-4b_shard{sid}.jsonl")
            if os.path.exists(sp):
                with open(sp) as fh:
                    for l in fh:
                        if l.strip():
                            r = json.loads(l)
                            if r["id"] not in seen_ids and r.get("dataset") == "pubmedqa":
                                stage_c_rows.append(r)
                                seen_ids.add(r["id"])

        with open(stage_c_final, "w", encoding="utf-8") as fh:
            for r in stage_c_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        print(f"  Stage C output saved ({len(stage_c_rows)} rows) -> {stage_c_final}", flush=True)

    m_c = compute_stage_summary_metrics(stage_c_rows, is_unans=False)
    summary_c_line = f"rows={m_c['rows']} (test) | strict_acc={m_c['strict_acc']:.4f} | fallback_acc={m_c['fallback_acc']:.4f} | trunc_rate={m_c['trunc_rate']:.1f}% ({m_c['trunc_count']}/{m_c['rows']}) | {m_c['s_per_q']:.2f} s/q"
    stage_summaries["Stage C (PubMedQA RAG)"] = summary_c_line
    print(f"\n[STAGE C SUMMARY] {summary_c_line}", flush=True)


    # ══════════════════════════════════════════════════════════════════════════
    # FINAL COMBINED SUMMARY ACROSS ALL 3 STAGES
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("=== MEDPSY-4B UNIFIED OVERNIGHT PUSH: FINAL COMBINED SUMMARY ===", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Stage':<38} {'Rows':<8} {'Strict Acc':<12} {'Fallback Acc':<14} {'Trunc Rate':<12} {'Speed':<10}", flush=True)
    print("-" * 90, flush=True)
    for stg_label, metrics_obj in [
        ("Stage A: unanswerable_v2 baseline", m_a),
        ("Stage B: PubMedQA baseline", m_b),
        ("Stage C: PubMedQA RAG", m_c),
    ]:
        strict_str = f"{metrics_obj['strict_acc']:.4f}"
        if "unanswerable" in stg_label:
            strict_str += " (c)"
        fb_str = f"{metrics_obj['fallback_acc']:.4f}"
        if "unanswerable" in stg_label:
            fb_str += " (c)"
        trunc_str = f"{metrics_obj['trunc_rate']:.1f}%"
        sq_str = f"{metrics_obj['s_per_q']:.2f} s/q"
        print(f"{stg_label:<38} {metrics_obj['rows']:<8} {strict_str:<12} {fb_str:<14} {trunc_str:<12} {sq_str:<10}", flush=True)
    print("=" * 90, flush=True)
    total_q = m_a['rows'] + m_b['rows'] + m_c['rows']
    print(f"Total Questions Evaluated : {total_q} across 3 stages", flush=True)
    print("All outputs safely committed to outputs/kaggle_qa/.", flush=True)
    print("=" * 90 + "\n", flush=True)

    # Post-run cleanup (caches only, after all stages finish successfully)
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.cache/pip"), ignore_errors=True)
    sys.exit(0)


# ── RAG_MEDPSY mode: MedPsy-4B + RAG on the MedQA test split ───────────────────
if RAG_MEDPSY:
    if FINAL_MEDPSY or VAL_MEDPSY or PILOT or PARALLEL_GPUS or PROFILE:
        sys.exit("ERROR: RAG_MEDPSY must run alone (set FINAL_MEDPSY, VAL_MEDPSY, PILOT, PARALLEL_GPUS, PROFILE to False).")
    import math
    import threading

    NUM_SHARDS = 2
    EXPECTED_RAG = 500   # MedQA test questions
    ev_path = os.path.join(WORK, "phase7", "evidence.jsonl")
    if not os.path.exists(ev_path):
        sys.exit(f"ERROR: Phase 7 evidence not found at {ev_path}; attach the medrag-build output.")
    print("\n=== RAG_MEDPSY STARTUP (split=test, datasets=medqa, mode=rag) ===", flush=True)
    try:
        with open(os.path.join(WORK, "phase1", "test.jsonl")) as fh:
            mq = [json.loads(l) for l in fh if l.strip()]
        mq = [q for q in mq if q.get("dataset") == "medqa"]
        for sid in range(NUM_SHARDS):
            print(f"shard {sid} rag: {sum(1 for i, _ in enumerate(mq) if i % NUM_SHARDS == sid)} (medqa)", flush=True)
    except Exception as exc:
        print(f"  (could not pre-count shards: {exc})", flush=True)

    rag_errors = {}

    def run_rag_shard(shard_idx):
        env = {"CUDA_VISIBLE_DEVICES": str(shard_idx)}
        try:
            cmd = build_qa_cmd(models_list=["medpsy-4b"], modes_list=["rag"], datasets="medqa",
                               split="test", work_dir=WORK, skip_format_check=True,
                               shard=shard_idx, num_shards=NUM_SHARDS)
            print(f"\n[shard {shard_idx}] Starting RAG on GPU {shard_idx}...", flush=True)
            run(cmd, extra_env=env)
        except Exception as exc:
            rag_errors[shard_idx] = str(exc)

    threads = [threading.Thread(target=run_rag_shard, args=(i,), name=f"rag-shard-{i}") for i in range(NUM_SHARDS)]
    for t in threads:
        t.start()
    print(f"\nWaiting for {len(threads)} shard thread(s)...", flush=True)
    for t in threads:
        t.join()
    if rag_errors:
        print("\nERROR: one or more shards failed:", rag_errors, flush=True)
        sys.exit(1)

    print("\n=== MERGING SHARDS ===", flush=True)
    p8_dir = os.path.join(WORK, "phase8")
    rows = []
    for sid in range(NUM_SHARDS):
        sp = os.path.join(p8_dir, f"medpsy-4b_shard{sid}.jsonl")
        with open(sp) as fh:
            part = [json.loads(l) for l in fh if l.strip()]
        print(f"    shard {sid}: {len(part)} rows from {sp}", flush=True)
        rows.extend(part)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        print(f"WARNING: [MERGE] {len(ids) - len(set(ids))} duplicate ids", flush=True)
    if len(rows) != EXPECTED_RAG:
        print(f"WARNING: [MERGE] rag: expected {EXPECTED_RAG} rows, got {len(rows)}", flush=True)
    out_path = os.path.join(p8_dir, "medpsy-4b.jsonl")
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  merged rag: {len(rows)} rows -> {out_path}", flush=True)

    n = len(rows)
    n_c = sum(1 for r in rows if r.get("correct"))
    p = n_c / n if n else 0.0
    m = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0
    trunc = sum(1 for r in rows if r.get("truncated"))
    secs = [r["seconds"] for r in rows if r.get("seconds") is not None]
    print("\n=== RAG MEDPSY SUMMARY (MedQA test) ===", flush=True)
    print(f"  RAG MedQA accuracy : {p:.4f}  95%CI=({max(0, p - m):.4f}, {min(1, p + m):.4f})  {n_c}/{n} correct", flush=True)
    print(f"  Truncated          : {trunc}/{n} ({100 * trunc / max(n, 1):.1f}%)", flush=True)
    if secs:
        print(f"  s/question         : {sum(secs) / len(secs):.2f}", flush=True)
    print("  (Baseline for comparison: 0.8760 [0.846, 0.904] from the FINAL_MEDPSY run)", flush=True)
    print("=== END RAG MEDPSY SUMMARY ===\n", flush=True)
    sys.exit(0)

if FINAL_MEDPSY or VAL_MEDPSY:
    if FINAL_MEDPSY and VAL_MEDPSY:
        sys.exit("ERROR: set only one of FINAL_MEDPSY / VAL_MEDPSY to True.")
    if PILOT or PARALLEL_GPUS or PROFILE:
        sys.exit("ERROR: FINAL_MEDPSY/VAL_MEDPSY are not compatible with PILOT, PARALLEL_GPUS, or PROFILE modes.")
    MEDPSY_SPLIT = "val" if VAL_MEDPSY else "test"
    MEDPSY_SPLIT_FILE = "validation.jsonl" if VAL_MEDPSY else "test.jsonl"

    import math
    import threading

    NUM_SHARDS = 2

    # Expected question counts per dataset on the test split:
    #   MedQA test: 500 questions
    #   Unanswerable test: 150 questions
    #   PubMedQA test: 500 questions
    #
    # Baseline mode covers MedQA (500) + unanswerable (150) = 650 total.
    # Split evenly by index parity across 2 shards:
    #   shard 0 baseline: 325 (medqa 250, unanswerable 75)
    #   shard 1 baseline: 325 (medqa 250, unanswerable 75)
    #
    # Context mode covers PubMedQA (500) total.
    # Split evenly by index parity across 2 shards:
    #   shard 0 context: 250 (pubmedqa 250)
    #   shard 1 context: 250 (pubmedqa 250)
    print(f"\n=== {'VAL' if VAL_MEDPSY else 'FINAL'}_MEDPSY STARTUP (split={MEDPSY_SPLIT}) ===", flush=True)
    test_jsonl = os.path.join(WORK, "phase1", MEDPSY_SPLIT_FILE)
    if os.path.exists(test_jsonl):
        try:
            with open(test_jsonl) as fh:
                test_qs = [json.loads(l) for l in fh if l.strip()]
            base_qs = [q for q in test_qs if (q.get("dataset") == "medqa" and not q.get("should_abstain")) or q.get("dataset") == "unanswerable" or q.get("should_abstain")]
            ctx_qs = [q for q in test_qs if q.get("dataset") == "pubmedqa" and not q.get("should_abstain")]
            for sid in range(NUM_SHARDS):
                s_b = [q for i, q in enumerate(base_qs) if i % NUM_SHARDS == sid]
                s_b_mq = sum(1 for q in s_b if q.get("dataset") == "medqa" and not q.get("should_abstain"))
                s_b_un = sum(1 for q in s_b if q.get("dataset") == "unanswerable" or q.get("should_abstain"))
                s_c = [q for i, q in enumerate(ctx_qs) if i % NUM_SHARDS == sid]
                s_c_pm = len(s_c)
                print(f"shard {sid} baseline: {len(s_b)} (medqa {s_b_mq}, unanswerable {s_b_un})", flush=True)
                print(f"shard {sid} context: {len(s_c)} (pubmedqa {s_c_pm})", flush=True)
        except Exception:
            for sid in range(NUM_SHARDS):
                print(f"shard {sid} baseline: 325 (medqa 250, unanswerable 75)", flush=True)
                print(f"shard {sid} context: 250 (pubmedqa 250)", flush=True)
    else:
        for sid in range(NUM_SHARDS):
            print(f"shard {sid} baseline: 325 (medqa 250, unanswerable 75)", flush=True)
            print(f"shard {sid} context: 250 (pubmedqa 250)", flush=True)

    # ── Launch two shards in parallel across GPUs ──────────────────────────────
    # For each shard, baseline and context are run as two separate qa_pipeline
    # invocations, sequentially on that shard's GPU.
    shard_errors = {}

    def run_shard_worker(shard_idx):
        gpu_id = str(shard_idx)  # shard 0 → GPU 0, shard 1 → GPU 1
        env = {"CUDA_VISIBLE_DEVICES": gpu_id}
        try:
            # Invocation 1: baseline on medqa,unanswerable
            cmd_b = build_qa_cmd(
                models_list=["medpsy-4b"],
                modes_list=["baseline"],
                datasets="medqa,unanswerable",
                split=MEDPSY_SPLIT,
                work_dir=WORK,
                skip_format_check=True,
                shard=shard_idx,
                num_shards=NUM_SHARDS,
            )
            print(f"\n[shard {shard_idx}] Starting baseline on GPU {gpu_id}...", flush=True)
            run(cmd_b, extra_env=env)

            # Invocation 2: context on pubmedqa
            cmd_c = build_qa_cmd(
                models_list=["medpsy-4b"],
                modes_list=["context"],
                datasets="pubmedqa",
                split=MEDPSY_SPLIT,
                work_dir=WORK,
                skip_format_check=True,
                shard=shard_idx,
                num_shards=NUM_SHARDS,
            )
            print(f"\n[shard {shard_idx}] Starting context on GPU {gpu_id}...", flush=True)
            run(cmd_c, extra_env=env)
        except Exception as exc:
            shard_errors[shard_idx] = str(exc)

    threads = []
    for shard_idx in range(NUM_SHARDS):
        t = threading.Thread(target=run_shard_worker, args=(shard_idx,), name=f"shard-{shard_idx}")
        threads.append(t)
        t.start()

    print(f"\nWaiting for {len(threads)} shard thread(s)...", flush=True)
    for t in threads:
        t.join()

    if shard_errors:
        print("\nERROR: one or more shards failed:", shard_errors, flush=True)
        sys.exit(1)

    # ── Merge shard files into standard output files ───────────────────────────

    def merge_shards(phase_dir_path, model_name, num_shards_merge,
                     expected_count, label):
        """Concatenate shard JSONL files.
        If counts differ from expected or duplicates exist, print a clear WARNING
        with actual numbers but STILL write the merged file — never exit before writing.
        """
        all_rows = []
        for sid in range(num_shards_merge):
            shard_path = os.path.join(phase_dir_path, f"{model_name}_shard{sid}.jsonl")
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Shard file missing: {shard_path}")
            with open(shard_path) as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
            print(f"    shard {sid}: {len(rows)} rows from {shard_path}")
            all_rows.extend(rows)

        ids = [r["id"] for r in all_rows]
        dup_ids = [i for i in ids if ids.count(i) > 1]
        if dup_ids:
            unique_dups = sorted(set(dup_ids))
            print(
                f"WARNING: [MERGE] Duplicate ids in {label}: {len(dup_ids)} duplicates "
                f"({len(unique_dups)} unique: {unique_dups[:10]})",
                flush=True,
            )
        if len(all_rows) != expected_count:
            print(
                f"WARNING: [MERGE] {label}: count mismatch: expected {expected_count} rows, got {len(all_rows)}",
                flush=True,
            )

        out_path = os.path.join(phase_dir_path, f"{model_name}.jsonl")
        with open(out_path, "w") as fh:
            for r in all_rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  merged {label}: {len(all_rows)} rows → {out_path}", flush=True)
        return all_rows

    print("\n=== MERGING SHARDS ===", flush=True)
    # MedQA (500) + unanswerable (150) = 650 baseline rows
    EXPECTED_BASELINE = 650
    # PubMedQA test = 500 context rows
    EXPECTED_CONTEXT  = 500
    if VAL_MEDPSY:
        # Validation split: MedQA 200 + unanswerable 50 baseline; PubMedQA 200 context
        EXPECTED_BASELINE, EXPECTED_CONTEXT = 250, 200

    p5_dir  = os.path.join(WORK, "phase5")
    p10_dir = os.path.join(WORK, "phase10")
    os.makedirs(p5_dir,  exist_ok=True)
    os.makedirs(p10_dir, exist_ok=True)

    baseline_rows = merge_shards(p5_dir,  "medpsy-4b", NUM_SHARDS, EXPECTED_BASELINE, "baseline")
    context_rows  = merge_shards(p10_dir, "medpsy-4b", NUM_SHARDS, EXPECTED_CONTEXT,  "context")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────

    def _ci95(n_correct, n_total):
        """Wilson 95% CI (normal approximation for display)."""
        if n_total == 0:
            return (None, None)
        p = n_correct / n_total
        z = 1.96
        margin = z * math.sqrt(p * (1 - p) / n_total)
        return (round(max(0.0, p - margin), 4), round(min(1.0, p + margin), 4))

    print(f"\n=== {'VAL' if VAL_MEDPSY else 'FINAL'} MEDPSY SUMMARY (split={MEDPSY_SPLIT}) ===", flush=True)

    # ── Baseline: MedQA accuracy ────────────────────────────────────────────────
    medqa_b  = [r for r in baseline_rows if r.get("dataset") == "medqa" and not r.get("should_abstain")]
    n_mq, n_mq_c = len(medqa_b), sum(r["correct"] for r in medqa_b)
    acc_mq   = round(n_mq_c / n_mq, 4) if n_mq else None
    ci_mq    = _ci95(n_mq_c, n_mq)
    # parse / truncation / speed for baseline
    b_parsed = [r for r in baseline_rows if r.get("parsed")]
    b_trunc  = [r for r in baseline_rows if r.get("truncated")]
    b_sec    = [r["seconds"] for r in baseline_rows if r.get("seconds") is not None]
    print(f"  Baseline MedQA     : accuracy={acc_mq}  95%CI={ci_mq}  n={n_mq}/{n_mq_c} correct", flush=True)
    print(f"  Baseline parse rate: {len(b_parsed)}/{len(baseline_rows)} "
          f"({100*len(b_parsed)/len(baseline_rows):.1f}%)", flush=True)
    print(f"  Baseline truncated : {len(b_trunc)}/{len(baseline_rows)} "
          f"({100*len(b_trunc)/len(baseline_rows):.1f}%)", flush=True)
    if b_sec:
        print(f"  Baseline s/question: {round(sum(b_sec)/len(b_sec), 2)}", flush=True)

    # ── Baseline: unanswerable confidence ──────────────────────────────────────
    unans_b = [r for r in baseline_rows if r.get("should_abstain")]
    if unans_b:
        confs = [r["confidence"] for r in unans_b if r.get("confidence") is not None]
        mean_conf  = round(sum(confs) / len(confs), 4) if confs else None
        high_conf  = sum(1 for c in confs if c >= 0.9)
        print(f"  Unanswerable (n={len(unans_b)}): mean_confidence={mean_conf}  "
              f"conf>=0.9: {high_conf}/{len(confs)} ({100*high_conf/len(confs):.1f}%)"
              if confs else f"  Unanswerable (n={len(unans_b)}): no confidence data",
              flush=True)

    # ── Context: PubMedQA accuracy ─────────────────────────────────────────────
    pubmedqa_c = [r for r in context_rows if not r.get("should_abstain")]
    n_pm, n_pm_c = len(pubmedqa_c), sum(r["correct"] for r in pubmedqa_c)
    acc_pm   = round(n_pm_c / n_pm, 4) if n_pm else None
    ci_pm    = _ci95(n_pm_c, n_pm)
    c_parsed = [r for r in context_rows if r.get("parsed")]
    c_trunc  = [r for r in context_rows if r.get("truncated")]
    c_sec    = [r["seconds"] for r in context_rows if r.get("seconds") is not None]
    print(f"  Context PubMedQA   : accuracy={acc_pm}  95%CI={ci_pm}  n={n_pm}/{n_pm_c} correct", flush=True)
    print(f"  Context parse rate : {len(c_parsed)}/{len(context_rows)} "
          f"({100*len(c_parsed)/len(context_rows):.1f}%)", flush=True)
    print(f"  Context truncated  : {len(c_trunc)}/{len(context_rows)} "
          f"({100*len(c_trunc)/len(context_rows):.1f}%)", flush=True)
    if c_sec:
        print(f"  Context s/question : {round(sum(c_sec)/len(c_sec), 2)}", flush=True)

    print("=== END FINAL MEDPSY SUMMARY ===\n", flush=True)
    sys.exit(0)


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


# ── V3_RUN mode ────────────────────────────────────────────────────────────────

if V3_RUN:
    import yaml
    _v3_conditions = [c.strip() for c in V3_CONDITIONS.split(",") if c.strip()]
    _v3_split = "validation" if V3_PILOT else "test"

    if not V3_PILOT:
        verify_frozen_hashes_before_run(CODE_DIR, WORK)

    if V3_PARALLEL_MODELS:
        _parallel_models = [m.strip() for m in V3_PARALLEL_MODELS.split(",") if m.strip()]
        if len(_parallel_models) != 2:
            sys.exit(f"ERROR: V3_PARALLEL_MODELS must specify exactly two comma-separated models, got: {V3_PARALLEL_MODELS}")

        procs = []
        for i, pmodel in enumerate(_parallel_models):
            if V3_PILOT:
                _p_out_base = (
                    f"/kaggle/working/outputs/pilots/{pmodel}"
                    if os.path.exists("/kaggle/working")
                    else f"outputs/pilots/{pmodel}"
                )
            else:
                _p_out_base = (
                    f"/kaggle/working/outputs/v3/{pmodel}"
                    if os.path.exists("/kaggle/working")
                    else f"outputs/v3/{pmodel}"
                )
            os.makedirs(_p_out_base, exist_ok=True)

            cmd_p = [
                sys.executable, "-m", "src.qa_pipeline_v3",
                "--config", "configs/base.yaml",
                "--work", WORK,
                "--model", pmodel,
                "--split", _v3_split,
                "--conditions", ",".join(_v3_conditions),
                "--out", _p_out_base,
            ]
            if V3_DATASETS != "all":
                cmd_p += ["--datasets", V3_DATASETS]
            if V3_PILOT:
                cmd_p += ["--pilot", "--pilot-n", str(V3_PILOT_N)]
            if CHECK_ONLY:
                cmd_p += ["--check-only"]
            if TINY:
                cmd_p += ["--tiny"]

            extra_env = {"CUDA_VISIBLE_DEVICES": str(i)}
            if "HF_TOKEN" in os.environ:
                extra_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
            procs.append((f"{pmodel} (GPU {i})", run_bg(cmd_p, extra_env=extra_env)))

        print(f"\nWaiting for {len(procs)} parallel v3 subprocess(es)...", flush=True)
        exit_codes = {}
        for label, proc in procs:
            rc = proc.wait()
            exit_codes[label] = rc
            print(f"  {label} finished with exit code {rc}", flush=True)

        any_failed = any(rc != 0 for rc in exit_codes.values())
        print("\nV3_RUN PARALLEL COMPLETE:", exit_codes, flush=True)
        if any_failed:
            sys.exit(1)
        sys.exit(0)

    if V3_PILOT:
        _v3_out_base = (
            f"/kaggle/working/outputs/pilots/{V3_MODEL}"
            if os.path.exists("/kaggle/working")
            else f"outputs/pilots/{V3_MODEL}"
        )
    else:
        _v3_out_base = (
            f"/kaggle/working/outputs/v3/{V3_MODEL}"
            if os.path.exists("/kaggle/working")
            else f"outputs/v3/{V3_MODEL}"
        )
    os.makedirs(_v3_out_base, exist_ok=True)

    cmd_v3 = [
        sys.executable, "-m", "src.qa_pipeline_v3",
        "--config", "configs/base.yaml",
        "--work", WORK,
        "--model", V3_MODEL,
        "--split", _v3_split,
        "--conditions", ",".join(_v3_conditions),
        "--out", _v3_out_base,
    ]
    if V3_DATASETS != "all":
        cmd_v3 += ["--datasets", V3_DATASETS]
    if V3_PILOT:
        cmd_v3 += ["--pilot", "--pilot-n", str(V3_PILOT_N)]
    if CHECK_ONLY:
        cmd_v3 += ["--check-only"]
    if TINY:
        cmd_v3 += ["--tiny"]
    run(cmd_v3)
    print(f"\nV3_RUN COMPLETE: model={V3_MODEL} split={_v3_split} conditions={_v3_conditions}", flush=True)
    sys.exit(0)


# ── V3_TRUNC_SENSITIVITY mode ──────────────────────────────────────────────────

if V3_TRUNC_SENSITIVITY:
    import yaml
    with open("configs/base.yaml") as _f:
        _v3_cfg = yaml.safe_load(_f)
    _trunc_n    = _v3_cfg.get("v3", {}).get("truncation_sensitivity_n", 100)
    _trunc_caps = _v3_cfg.get("v3", {}).get("truncation_caps", [1024, 2048])
    _trunc_models = [
        "medpsy-1.7b", "qwen3-1.7b-thinking", "qwen3-4b-thinking-2507", "medpsy-4b",
    ]
    print(f"\nV3_TRUNC_SENSITIVITY: n={_trunc_n}, caps={_trunc_caps}, models={_trunc_models}", flush=True)
    _trunc_out = (
        "/kaggle/working/outputs/v3/trunc_sensitivity"
        if os.path.exists("/kaggle/working")
        else "outputs/v3/trunc_sensitivity"
    )
    os.makedirs(_trunc_out, exist_ok=True)
    cmd_trunc = [
        sys.executable, "-m", "src.truncation_sensitivity",
        "--config", "configs/base.yaml",
        "--work", WORK,
        "--n", str(_trunc_n),
        "--caps", ",".join(str(c) for c in _trunc_caps),
        "--models", ",".join(_trunc_models),
        "--out", _trunc_out,
    ]
    if TINY:
        cmd_trunc += ["--tiny"]
    run(cmd_trunc)
    print(f"\nV3_TRUNC_SENSITIVITY COMPLETE — results in {_trunc_out}", flush=True)
    sys.exit(0)


# ── V3_BUDGET_SWEEP mode ────────────────────────────────────────────────────────
if V3_BUDGET_SWEEP:
    verify_frozen_hashes_before_run(CODE_DIR, WORK)
    _sweep_models = ["qwen3-4b-thinking-2507", "medpsy-4b"]
    _sweep_out = (
        "/kaggle/working/outputs/sweeps/budget_pair_b"
        if os.path.exists("/kaggle/working")
        else "outputs/sweeps/budget_pair_b"
    )
    os.makedirs(_sweep_out, exist_ok=True)
    print(
        f"\nStarting V3_BUDGET_SWEEP: models={_sweep_models}, N={V3_SWEEP_N}, "
        f"max_hours={V3_SWEEP_MAX_HOURS}, batch_size={V3_SWEEP_BATCH_SIZE}",
        flush=True,
    )

    procs = []
    t0_sweep = time.time()
    for i, smodel in enumerate(_sweep_models):
        cmd_s = [
            sys.executable, "-m", "src.qa_pipeline_v3",
            "--config", "configs/base.yaml",
            "--work", WORK,
            "--model", smodel,
            "--split", "validation",
            "--budget-sweep",
            "--sweep-n", str(V3_SWEEP_N),
            "--sweep-max-hours", str(V3_SWEEP_MAX_HOURS),
            "--batch-size", str(V3_SWEEP_BATCH_SIZE),
            "--out", _sweep_out,
        ]
        if CHECK_ONLY:
            cmd_s += ["--check-only"]
        if TINY:
            cmd_s += ["--tiny"]

        extra_env = {"CUDA_VISIBLE_DEVICES": str(i)}
        if "HF_TOKEN" in os.environ:
            extra_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
        procs.append((f"{smodel} (GPU {i})", run_bg(cmd_s, extra_env=extra_env)))

    print(f"\nWaiting for {len(procs)} parallel budget-sweep subprocess(es)...", flush=True)
    exit_codes = {}
    for label, proc in procs:
        rc = proc.wait()
        exit_codes[label] = rc
        print(f"  {label} finished with exit code {rc}", flush=True)

    elapsed_hours = (time.time() - t0_sweep) / 3600.0
    any_failed = any(rc != 0 for rc in exit_codes.values())

    # Build and verify combined manifest.json
    import hashlib
    rule_path = "docs/BUDGET_DECISION_RULE.md"
    rule_hash = ""
    if os.path.exists(rule_path):
        with open(rule_path, "rb") as fh:
            rule_hash = hashlib.sha256(fh.read()).hexdigest()

    def _get_git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return os.environ.get("GIT_COMMIT", "unknown")

    # Check that both model outputs exist and have N rows
    all_completed = (not any_failed)
    for smodel in _sweep_models:
        mf = os.path.join(_sweep_out, f"{smodel}.jsonl")
        if not os.path.exists(mf):
            all_completed = False
            break
        with open(mf, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
            if len(lines) < V3_SWEEP_N:
                all_completed = False

    manifest_path = os.path.join(_sweep_out, "manifest.json")
    manifest_data = {
        "git_commit_hash": _get_git_commit(),
        "rule_file_sha256": rule_hash,
        "generation_config": {
            "max_new_tokens": 4096,
            "do_sample": False,
        },
        "torch_version": "2.6.0" if "torch" not in sys.modules else sys.modules["torch"].__version__,
        "transformers_version": "4.49.0" if "transformers" not in sys.modules else sys.modules["transformers"].__version__,
        "dtype": "float16",
        "batch_size": V3_SWEEP_BATCH_SIZE,
        "N": V3_SWEEP_N,
        "seed": 42,
        "completed": all_completed,
        "hours_used": round(elapsed_hours, 4),
        "models": _sweep_models,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2)
    print(f"Sweep manifest written to {manifest_path} (completed={all_completed})", flush=True)

    print("\nV3_BUDGET_SWEEP COMPLETE:", exit_codes, flush=True)
    if any_failed or not all_completed:
        sys.exit(1)
    sys.exit(0)


# ── V3_GEMMA_PROMPT_SENS mode ──────────────────────────────────────────────────
if V3_GEMMA_PROMPT_SENS:
    verify_frozen_hashes_before_run(CODE_DIR, WORK)
    _sens_models = ["gemma3-4b", "medgemma-1.5-4b-it"]
    _sens_out = (
        "/kaggle/working/outputs/sweeps/gemma_prompt_sensitivity"
        if os.path.exists("/kaggle/working")
        else "outputs/sweeps/gemma_prompt_sensitivity"
    )
    os.makedirs(_sens_out, exist_ok=True)
    print(
        f"\nStarting V3_GEMMA_PROMPT_SENS (SEQUENTIAL): models={_sens_models}, N={V3_GEMMA_SENS_N}, "
        f"max_hours={V3_GEMMA_SENS_MAX_HOURS}, variants=['V1', 'V2', 'V3']",
        flush=True,
    )

    t0_sens = time.time()
    exit_codes = {}

    # Sequential execution: gemma3-4b first, then medgemma-1.5-4b-it.
    # No CUDA_VISIBLE_DEVICES pinning: both GPUs visible so device_map="auto" shards float32 model across T4s.
    for i, smodel in enumerate(_sens_models):
        elapsed_so_far = (time.time() - t0_sens) / 3600.0
        remaining_total = max(0.05, V3_GEMMA_SENS_MAX_HOURS - elapsed_so_far)
        if i == 0:
            model_budget = min(V3_GEMMA_SENS_MAX_HOURS / 2.0, remaining_total)
        else:
            model_budget = remaining_total

        print(
            f"\n[{i+1}/{len(_sens_models)}] Launching {smodel} sequentially "
            f"(budget: {model_budget:.2f}h, total elapsed: {elapsed_so_far:.2f}h)...",
            flush=True,
        )

        cmd_s = [
            sys.executable, "-m", "src.qa_pipeline_v3",
            "--config", "configs/base.yaml",
            "--work", WORK,
            "--model", smodel,
            "--split", "validation",
            "--gemma-prompt-sens",
            "--sens-n", str(V3_GEMMA_SENS_N),
            "--sens-max-hours", f"{model_budget:.4f}",
            "--out", _sens_out,
        ]
        if CHECK_ONLY:
            cmd_s += ["--check-only"]
        if TINY:
            cmd_s += ["--tiny"]

        # Ensure CUDA_VISIBLE_DEVICES is NOT pinned to a single GPU
        sub_env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" in sub_env:
            del sub_env["CUDA_VISIBLE_DEVICES"]
        if "HF_TOKEN" in os.environ:
            sub_env["HF_TOKEN"] = os.environ["HF_TOKEN"]

        print(">>", " ".join(cmd_s), flush=True)
        res = subprocess.run(cmd_s, env=sub_env)
        rc = res.returncode
        exit_codes[smodel] = rc
        print(f"  {smodel} finished with exit code {rc}", flush=True)

    elapsed_hours = (time.time() - t0_sens) / 3600.0
    any_failed = any(rc != 0 for rc in exit_codes.values())

    # Build and verify combined manifest.json
    import hashlib
    rule_path = "docs/GEMMA_PROMPT_SENSITIVITY_RULE.md"
    rule_hash = ""
    if os.path.exists(rule_path):
        with open(rule_path, "rb") as fh:
            rule_hash = hashlib.sha256(fh.read()).hexdigest()

    from src.qa_pipeline_v3 import PROMPT_TEMPLATE_HASHES

    def _get_git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return os.environ.get("GIT_COMMIT", "unknown")

    # Check that both model outputs exist and have N * 3 rows
    all_completed = (not any_failed)
    for smodel in _sens_models:
        mf = os.path.join(_sens_out, f"{smodel}.jsonl")
        if not os.path.exists(mf):
            all_completed = False
            break
        with open(mf, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
            if len(lines) < V3_GEMMA_SENS_N * 3:
                all_completed = False

    manifest_path = os.path.join(_sens_out, "manifest.json")
    manifest_data = {
        "git_commit_hash": _get_git_commit(),
        "rule_file_sha256": rule_hash,
        "prompt_template_hashes": PROMPT_TEMPLATE_HASHES,
        "generation_config": {
            "max_new_tokens": 16,
            "do_sample": False,
        },
        "torch_version": "2.6.0" if "torch" not in sys.modules else sys.modules["torch"].__version__,
        "transformers_version": "4.49.0" if "transformers" not in sys.modules else sys.modules["transformers"].__version__,
        "dtype": "float32",
        "N": V3_GEMMA_SENS_N,
        "variants": ["V1", "V2", "V3"],
        "seed": 42,
        "completed": all_completed,
        "hours_used": round(elapsed_hours, 4),
        "models": _sens_models,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2)
    print(f"Prompt-sensitivity manifest written to {manifest_path} (completed={all_completed})", flush=True)

    print("\nV3_GEMMA_PROMPT_SENS COMPLETE:", exit_codes, flush=True)
    if any_failed or not all_completed:
        sys.exit(1)
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
