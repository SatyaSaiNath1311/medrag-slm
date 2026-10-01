"""Kaggle runner A: build data, corpus, indexes, retrieval and reranking (Phases 1, 2, 3, 6, 7).
Model-independent. Output is used as input by runner B (medrag-slm-runner)."""
import os
import subprocess
import sys

REPO_URL = "https://github.com/SatyaSaiNath1311/medrag-slm.git"
CODE_DIR = "/tmp/medrag-slm"
WORK = "/kaggle/working/work"
PHASES = ["phase01_datasets", "phase02_corpus", "phase03_index", "phase06_retrieve", "phase07_rerank"]


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


run(["rm", "-rf", CODE_DIR])
run(["git", "clone", "--depth", "1", REPO_URL, CODE_DIR])
os.chdir(CODE_DIR)
run(["git", "log", "-1", "--oneline"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])

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

for phase in PHASES:
    run([sys.executable, "-m", f"src.{phase}", "--config", "configs/base.yaml", "--work", WORK])
print("\nRUNNER A COMPLETE: phases 1, 2, 3, 6, 7")
