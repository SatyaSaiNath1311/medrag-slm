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
import time

KERNEL_START_TIME = float(os.environ.get("KERNEL_START_TIME", time.time()))
if "KERNEL_START_TIME" not in os.environ:
    os.environ["KERNEL_START_TIME"] = str(KERNEL_START_TIME)
SCRIPT_PATH = os.path.abspath(__file__)

REPO_URL = "https://github.com/SatyaSaiNath1311/medrag-slm.git"
CODE_DIR = "/tmp/medrag-slm"
WORK = "/kaggle/working/work"
MODE = os.environ.get("MODE", "qa")        # "qa" or "smoke"
CHECK_ONLY = False
PROFILE = False
FINAL_MEDPSY = False        # full test-split run of medpsy-4b, dual-GPU shards (DONE)
VAL_MEDPSY = False          # full validation-split run of medpsy-4b (250 baseline + 200 context), dual-GPU shards (DONE)
RAG_MEDPSY = False          # medpsy-4b + RAG (top-5 evidence) on the MedQA test split (500), dual-GPU shards
PILOT = False               # run medpsy-4b pilot (80 val-subset questions, single GPU)
PARALLEL_GPUS = False       # split model list across 2 GPUs in parallel subprocesses
TINY = False                # set to True for tiny check on Kaggle
MODELS = ["medpsy-4b"]
MODES = ["baseline", "rag"]

SC_MODE = True       # self-consistency mode for MedPsy-4B on validation
SC_K = 5
SC_TEMPERATURE = 0.7
SC_TOP_P = 0.95
SC_PILOT_N = 4
SC_BATCH_SIZE = 5      # batch size for sampling to avoid T4 OOM
SC_DATASETS = "medqa,pubmedqa,unanswerable_v2"
SC_SHARD = None
SC_NUM_SHARDS = 2

# Override from env var if provided
if "SC_MODE" in os.environ:
    SC_MODE = os.environ["SC_MODE"].lower() in ("1", "true", "yes")
if "SC_K" in os.environ:
    SC_K = int(os.environ["SC_K"])
if "SC_TEMPERATURE" in os.environ:
    SC_TEMPERATURE = float(os.environ["SC_TEMPERATURE"])
if "SC_TOP_P" in os.environ:
    SC_TOP_P = float(os.environ["SC_TOP_P"])
if "SC_PILOT_N" in os.environ:
    SC_PILOT_N = int(os.environ["SC_PILOT_N"])
if "SC_BATCH_SIZE" in os.environ:
    SC_BATCH_SIZE = int(os.environ["SC_BATCH_SIZE"])
if "SC_DATASETS" in os.environ:
    SC_DATASETS = os.environ["SC_DATASETS"]
if "SC_SHARD" in os.environ:
    SC_SHARD = int(os.environ["SC_SHARD"])
if "SC_NUM_SHARDS" in os.environ:
    SC_NUM_SHARDS = int(os.environ["SC_NUM_SHARDS"])
if "CHECK_ONLY" in os.environ:
    CHECK_ONLY = os.environ["CHECK_ONLY"].lower() in ("1", "true", "yes")
if "PROFILE" in os.environ:
    PROFILE = os.environ["PROFILE"].lower() in ("1", "true", "yes")
if "FINAL_MEDPSY" in os.environ:
    FINAL_MEDPSY = os.environ["FINAL_MEDPSY"].lower() in ("1", "true", "yes")
if "VAL_MEDPSY" in os.environ:
    VAL_MEDPSY = os.environ["VAL_MEDPSY"].lower() in ("1", "true", "yes")
if "RAG_MEDPSY" in os.environ:
    RAG_MEDPSY = os.environ["RAG_MEDPSY"].lower() in ("1", "true", "yes")
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


if SC_SHARD is None:
    print_kaggle_input_configs()

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
    if VAL_MEDPSY:
        print("VAL_MEDPSY mode enabled: medpsy-4b full validation split, 2-shard dual-GPU")
    if PILOT:
        print("PILOT mode enabled: medpsy-4b pilot on 80-question val subset, single GPU")
    if PARALLEL_GPUS:
        print("PARALLEL_GPUS mode enabled: splitting model list across 2 GPUs")
    if TINY:
        print("TINY mode enabled: running on tiny scale")
    if SC_MODE:
        print(f"SC_MODE enabled: medpsy-4b self-consistency (parent coordinator, dual-GPU, datasets={SC_DATASETS}, K={SC_K}, temp={SC_TEMPERATURE}, top_p={SC_TOP_P}, pilot_n={SC_PILOT_N}, batch_size={SC_BATCH_SIZE})")

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
            print("Note: evidence.jsonl not found; SC_MODE requires only validation data.", flush=True)
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
                 shard=None, num_shards=2, datasets=None):
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
    if datasets:
        cmd += ["--datasets", datasets]
    if shard is not None:
        cmd += ["--shard", str(shard), "--num-shards", str(num_shards)]
    return cmd


# ── SC_MODE: Self-Consistency Mode for MedPsy-4B on Validation Only ────────────
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
        print("=== SC_MODE PARENT: Launching dual-GPU shards ===", flush=True)
        print(f"  SC_DATASETS    : {SC_DATASETS}", flush=True)
        print(f"  SC_K           : {SC_K}", flush=True)
        print(f"  SC_TEMPERATURE : {SC_TEMPERATURE}", flush=True)
        print(f"  SC_TOP_P       : {SC_TOP_P}", flush=True)
        print(f"  SC_PILOT_N     : {SC_PILOT_N} (0 = full validation set)", flush=True)
        print(f"  SC_BATCH_SIZE  : {SC_BATCH_SIZE}", flush=True)
        print(f"  SC_NUM_SHARDS  : {SC_NUM_SHARDS}", flush=True)
        print("=" * 80 + "\n", flush=True)

        sc_out_dir = "/kaggle/working/outputs/sc_val/medpsy-4b" if os.path.exists("/kaggle/working") else "outputs/sc_val/medpsy-4b"
        os.makedirs(sc_out_dir, exist_ok=True)
        if os.path.exists("/kaggle/working") and os.getcwd() != "/kaggle/working":
            try:
                os.makedirs("outputs/sc_val/medpsy-4b", exist_ok=True)
            except Exception:
                pass

        os.environ["KERNEL_START_TIME"] = str(KERNEL_START_TIME)
        procs = []
        for sid in range(SC_NUM_SHARDS):
            env_shard = os.environ.copy()
            if "HF_TOKEN" in os.environ:
                env_shard["HF_TOKEN"] = os.environ["HF_TOKEN"]
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
        print("=== SC_MODE PARENT: MERGING SHARDS ===", flush=True)
        active_sc_datasets = [d.strip().lower() for d in SC_DATASETS.split(",") if d.strip()]

        expected_map = {
            "medqa": 200 if SC_PILOT_N == 0 else SC_PILOT_N,
            "pubmedqa": 200 if SC_PILOT_N == 0 else SC_PILOT_N,
            "unanswerable_v2": 260 if SC_PILOT_N == 0 else SC_PILOT_N,
        }

        for ds_name in active_sc_datasets:
            shard_rows = []
            for sid in range(SC_NUM_SHARDS):
                sp = os.path.join(sc_out_dir, f"{ds_name}_shard{sid}.jsonl")
                if not os.path.exists(sp):
                    local_sp = os.path.join("outputs/sc_val/medpsy-4b", f"{ds_name}_shard{sid}.jsonl")
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
                print(f"  [Merge] WARNING: {ds_name} expected {expected} rows (SC_PILOT_N={SC_PILOT_N}), got {len(shard_rows)} rows!", flush=True)

            # Write merged file
            merged_file = os.path.join(sc_out_dir, f"{ds_name}.jsonl")
            with open(merged_file, "w", encoding="utf-8") as fh:
                for r in shard_rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"  [Merge] Successfully wrote {len(shard_rows)} rows to {merged_file}", flush=True)

            # Duplicate to repo-local dir if different
            local_dir = "outputs/sc_val/medpsy-4b"
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
    print(f"=== SC_MODE SHARD {SC_SHARD}/{SC_NUM_SHARDS} STARTUP: MedPsy-4B Self-Consistency Validation ===", flush=True)
    print(f"  SC_SHARD       : {SC_SHARD} of {SC_NUM_SHARDS}", flush=True)
    print(f"  SC_DATASETS    : {SC_DATASETS} ({active_sc_datasets})", flush=True)
    print(f"  SC_K           : {SC_K}", flush=True)
    print(f"  SC_TEMPERATURE : {SC_TEMPERATURE}", flush=True)
    print(f"  SC_TOP_P       : {SC_TOP_P}", flush=True)
    print(f"  SC_PILOT_N     : {SC_PILOT_N} (0 = full validation set)", flush=True)
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

    # ── 2. Load unanswerable_v2 validation data (with strict test.jsonl safety assert) ──
    unans_val_all = []
    if "unanswerable_v2" in active_sc_datasets:
        unans_candidates = [
            os.path.join(CODE_DIR, "data", "unanswerable_v2", "val.jsonl"),
            "data/unanswerable_v2/val.jsonl",
            os.path.abspath("data/unanswerable_v2/val.jsonl"),
            "/kaggle/working/medrag-slm/data/unanswerable_v2/val.jsonl",
            "/kaggle/working/data/unanswerable_v2/val.jsonl",
        ]
        unans_val_path = next((p for p in unans_candidates if os.path.exists(p)), None)
        if not unans_val_path:
            input_hits = (
                glob.glob("/kaggle/input/**/unanswerable_v2/val.jsonl", recursive=True)
                or glob.glob("/kaggle/input/**/data/unanswerable_v2/val.jsonl", recursive=True)
            )
            if input_hits:
                unans_val_path = input_hits[0]

        if not unans_val_path or not os.path.exists(unans_val_path):
            searched_fmt = "\n  - ".join(unans_candidates + ["/kaggle/input/**/unanswerable_v2/val.jsonl"])
            sys.exit(
                f"\n" + "=" * 80 + "\n"
                f"FATAL ERROR: data/unanswerable_v2/val.jsonl NOT FOUND!\n"
                f"Searched candidate locations:\n  - {searched_fmt}\n\n"
                f"Troubleshooting / Resolution:\n"
                f"  1. Git clone path: verify 'data/unanswerable_v2/val.jsonl' is committed and pushed to git repository:\n"
                f"     {REPO_URL}\n"
                f"  2. Kaggle dataset input: if using an attached dataset, ensure it contains 'data/unanswerable_v2/val.jsonl'.\n"
                f"  3. Working directory: currently '{os.getcwd()}'; CODE_DIR is '{CODE_DIR}'.\n"
                f"=" * 80 + "\n"
            )

        # STRICT ASSERT: Never load test.jsonl in SC_MODE
        assert not unans_val_path.endswith("test.jsonl"), (
            f"SAFETY ASSERTION VIOLATION: Attempted to load test set {unans_val_path} in SC_MODE!"
        )
        assert "test.jsonl" not in unans_val_path.lower(), (
            f"SAFETY ASSERTION VIOLATION: Path '{unans_val_path}' contains test.jsonl!"
        )

        with open(unans_val_path, "r", encoding="utf-8") as f:
            unans_val_all = [json.loads(line) for line in f if line.strip()]

        assert len(unans_val_all) == 260, f"Expected 260 unanswerable_v2 val rows, got {len(unans_val_all)}"
        assert all(r.get("split") == "validation" for r in unans_val_all), (
            "SAFETY ASSERTION VIOLATION: Found non-validation split rows in unanswerable val set!"
        )
        assert not any(r.get("id", "").startswith("unans-v2-test-") for r in unans_val_all), (
            "SAFETY ASSERTION VIOLATION: Found test-split IDs in unanswerable val set!"
        )
        print(f"Loaded {len(unans_val_all)} unanswerable_v2 validation questions from {unans_val_path}", flush=True)

    # ── 3. Load MedQA (200) and PubMedQA (200) validation questions ────────────
    medqa_val_all = []
    pubmedqa_val_all = []
    if any(ds in active_sc_datasets for ds in ("medqa", "pubmedqa")):
        val_candidates = [
            os.path.join(WORK, "phase1", "validation.jsonl"),
            "/kaggle/working/work/phase1/validation.jsonl",
            os.path.join(CODE_DIR, "outputs", "kaggle_build", "work", "phase1", "validation.jsonl"),
            "outputs/kaggle_build/work/phase1/validation.jsonl",
        ]
        val_file = next((p for p in val_candidates if os.path.exists(p)), None)
        if not val_file:
            hits_v = glob.glob("/kaggle/input/**/phase1/validation.jsonl", recursive=True) or glob.glob("/kaggle/input/**/validation.jsonl", recursive=True)
            if hits_v:
                val_file = hits_v[0]
        if not val_file:
            sys.exit("ERROR: validation.jsonl not found for MedQA / PubMedQA!")

        with open(val_file, "r", encoding="utf-8") as f:
            val_p1_rows = [json.loads(line) for line in f if line.strip()]

        medqa_val_all = [q for q in val_p1_rows if q.get("dataset") == "medqa" and not q.get("should_abstain")]
        pubmedqa_val_all = [q for q in val_p1_rows if q.get("dataset") == "pubmedqa" and not q.get("should_abstain")]

        if "medqa" in active_sc_datasets:
            assert len(medqa_val_all) == 200, f"Expected 200 MedQA val rows, got {len(medqa_val_all)}"
        if "pubmedqa" in active_sc_datasets:
            assert len(pubmedqa_val_all) == 200, f"Expected 200 PubMedQA val rows, got {len(pubmedqa_val_all)}"
        print(f"Loaded {len(medqa_val_all)} MedQA val and {len(pubmedqa_val_all)} PubMedQA val from {val_file}", flush=True)

    # ── 4. Apply SC_PILOT_N filter if active ───────────────────────────────────
    medqa_run = []
    pubmedqa_run = []
    unans_run = []
    if SC_PILOT_N > 0:
        if "medqa" in active_sc_datasets:
            medqa_run = medqa_val_all[:SC_PILOT_N]
        if "pubmedqa" in active_sc_datasets:
            pubmedqa_run = pubmedqa_val_all[:SC_PILOT_N]
        if "unanswerable_v2" in active_sc_datasets:
            unans_run = sample_stratified_unans_v2(unans_val_all, SC_PILOT_N, seed=42)
        print(f"\nPILOT MODE ACTIVE (SC_PILOT_N={SC_PILOT_N}):")
        for ds in active_sc_datasets:
            cnt = len(medqa_run) if ds == "medqa" else (len(pubmedqa_run) if ds == "pubmedqa" else len(unans_run))
            print(f"  {ds:<15}: first {cnt} questions")
    else:
        if "medqa" in active_sc_datasets:
            medqa_run = medqa_val_all
        if "pubmedqa" in active_sc_datasets:
            pubmedqa_run = pubmedqa_val_all
        if "unanswerable_v2" in active_sc_datasets:
            unans_run = unans_val_all
        print(f"\nFULL VALIDATION MODE ACTIVE:")
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
    sc_out_dir = "/kaggle/working/outputs/sc_val/medpsy-4b" if os.path.exists("/kaggle/working") else "outputs/sc_val/medpsy-4b"
    os.makedirs(sc_out_dir, exist_ok=True)
    if os.path.exists("/kaggle/working") and os.getcwd() != "/kaggle/working":
        try:
            os.makedirs("outputs/sc_val/medpsy-4b", exist_ok=True)
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
                local_dir = "outputs/sc_val/medpsy-4b"
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
