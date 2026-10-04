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
FINAL_MEDPSY = False        # full test-split run of medpsy-4b, dual-GPU shards (DONE)
VAL_MEDPSY = False          # full validation-split run of medpsy-4b (250 baseline + 200 context), dual-GPU shards (DONE)
RAG_MEDPSY = True           # medpsy-4b + RAG (top-5 evidence) on the MedQA test split (500), dual-GPU shards
PILOT = False               # run medpsy-4b pilot (80 val-subset questions, single GPU)
PARALLEL_GPUS = False       # split model list across 2 GPUs in parallel subprocesses
TINY = False                # set to True for tiny check on Kaggle
MODELS = ["medpsy-4b"]
MODES = ["baseline", "rag"]

# Override from env var if provided
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


# ── FINAL_MEDPSY mode ──────────────────────────────────────────────────────────


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
