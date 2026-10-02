"""Kaggle runner B: question answering with the 5 models (Phases 4, 5, 8).
Input: output of runner A (medrag-build), attached as a notebook input.
Change MODE to "smoke" to re-run the setup smoke test instead."""
import glob
import os
import subprocess
import sys

REPO_URL = "https://github.com/SatyaSaiNath1311/medrag-slm.git"
CODE_DIR = "/tmp/medrag-slm"
WORK = "/kaggle/working/work"
MODE = os.environ.get("MODE", "qa")        # "qa" or "smoke"
CHECK_ONLY = False
TINY = False                               # set to True for tiny check on Kaggle
MODELS = ["phi4-mini"]

# Override from env var if provided
if "CHECK_ONLY" in os.environ:
    CHECK_ONLY = os.environ["CHECK_ONLY"].lower() in ("1", "true", "yes")
if "TINY" in os.environ:
    TINY = os.environ["TINY"].lower() in ("1", "true", "yes")
if "MODELS" in os.environ:
    MODELS = [m.strip() for m in os.environ["MODELS"].split(",") if m.strip()]


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


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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

# 4. Run phases 4, 5, 8
cmd = [sys.executable, "-m", "src.qa_pipeline", "--config", "configs/base.yaml", "--work", WORK]
if CHECK_ONLY:
    cmd.append("--check-only")
if TINY:
    cmd.append("--tiny")
if MODELS:
    cmd += ["--models", *MODELS]
run(cmd)
print("\nRUNNER B COMPLETE: " + ("phase 4 format check" if CHECK_ONLY else "phases 4, 5, 8"))
