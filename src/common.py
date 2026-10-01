"""Shared helpers used by every phase."""
import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import yaml


# ---------- config / args ----------
def phase_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--work", default="outputs/work", help="Folder holding phase1/, phase2/, ... outputs")
    ap.add_argument("--tiny", action="store_true", help="Small, fast version for laptop checks")
    return ap


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def phase_dir(work, n):
    d = Path(work) / f"phase{n}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- files ----------
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Atomic write: a crash never leaves a half-written file behind."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(pdir, info):
    pdir = Path(pdir)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(pdir.iterdir()) if p.is_file() and p.name != "manifest.json"}
    info = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), **info, "files": files}
    (pdir / "manifest.json").write_text(json.dumps(info, indent=2))
    return info


# ---------- text ----------
def norm_space(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def norm_key(s):
    """Key for duplicate detection: lowercase, letters/digits only."""
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


# ---------- checks ----------
def check(condition, message):
    if not condition:
        raise AssertionError(f"[CHECK FAILED] {message}")
    print(f"  [ok] {message}")


def banner(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70, flush=True)


def build_query(q, include_options=True):
    text = q["question"]
    if include_options:
        text += " " + " ".join(q["options"].values())
    return text
