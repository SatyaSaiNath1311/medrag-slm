"""Thin Kaggle runner: download the code from GitHub, install, run one step.
All real logic lives in the GitHub repo, never in this file."""
import os
import subprocess
import sys

REPO_URL = "https://github.com/SatyaSaiNath1311/medrag-slm"  # <- edit once
STEP = ["-m", "src.smoke_test", "--config", "configs/base.yaml", "--out", "/kaggle/working/outputs/smoke"]

CODE_DIR = "/tmp/medrag-slm"


def run(cmd, **kw):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


# 1. Hugging Face token from Kaggle Secrets (attach it once: Add-ons -> Secrets -> HF_TOKEN)
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
run(["git", "log", "-1", "--oneline"])  # records exactly which code version ran

# 3. Install (torch is already on Kaggle and is NOT reinstalled)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

# 4. Run the step
run([sys.executable, *STEP])
