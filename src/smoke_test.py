"""Week-1 smoke test.

Checks the environment and all 5 models BEFORE the real project starts:
  1. Environment: GPU, library versions, Hugging Face token
  2. Access: can each model be downloaded (catches gated-model/licence problems)
  3. Per model: loads, produces finite numbers (catches fp16 problems),
     follows the answer format, no thinking-mode leak, speed and VRAM

Usage:
  Kaggle (real check): python -m src.smoke_test --config configs/base.yaml --out /kaggle/working/outputs
  Laptop (code check): python -m src.smoke_test --config configs/base.yaml --out outputs/local --tiny
"""
import argparse
import gc
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import transformers
import yaml
from packaging import version

# Five simple questions. Only used to check the models work, not for results.
QUESTIONS = [
    {"q": "Deficiency of which vitamin causes scurvy?",
     "options": {"A": "Vitamin A", "B": "Vitamin C", "C": "Vitamin D", "D": "Vitamin K"}, "answer": "B"},
    {"q": "Which organism is the most common cause of community-acquired pneumonia in adults?",
     "options": {"A": "Staphylococcus aureus", "B": "Klebsiella pneumoniae",
                 "C": "Streptococcus pneumoniae", "D": "Pseudomonas aeruginosa"}, "answer": "C"},
    {"q": "What is the antidote for acetaminophen (paracetamol) overdose?",
     "options": {"A": "N-acetylcysteine", "B": "Naloxone", "C": "Flumazenil", "D": "Atropine"}, "answer": "A"},
    {"q": "Peaked T waves on an ECG are a classic sign of which electrolyte abnormality?",
     "options": {"A": "Hypokalemia", "B": "Hyponatremia", "C": "Hypercalcemia", "D": "Hyperkalemia"}, "answer": "D"},
    {"q": "Which hormone is produced by the beta cells of the pancreas?",
     "options": {"A": "Glucagon", "B": "Insulin", "C": "Somatostatin", "D": "Gastrin"}, "answer": "B"},
]

PROMPT = (
    "Answer the following multiple-choice medical question.\n"
    "Reply with only one line in exactly this format: Answer: <letter>\n\n"
    "Question: {q}\n{opts}"
)

ANSWER_RE = re.compile(r"Answer\s*[:\-]?\s*\(?\s*([ABCD])\b", re.IGNORECASE)
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
DTYPE_KWARG = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"


def check_environment():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "warnings": [],
    }
    if not info["cuda_available"]:
        info["warnings"].append("No GPU found. Fine for a --tiny laptop check, NOT for the real test.")
    if any("P100" in g for g in info["gpus"]):
        info["warnings"].append("P100 detected. Switch the Kaggle notebook to GPU T4 x2 (newer PyTorch may not support P100).")
    if not info["hf_token_set"]:
        info["warnings"].append("HF_TOKEN not set. Gemma (gated) will fail to download.")
    return info


def check_access(model_id):
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(model_id, "config.json")
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:300]}"


def load_model(cfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    if cfg.get("loader") == "gemma3":
        from transformers import Gemma3ForConditionalGeneration as ModelClass
    else:
        ModelClass = AutoModelForCausalLM
    kwargs = {DTYPE_KWARG: DTYPES[cfg["dtype"]]}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = ModelClass.from_pretrained(cfg["id"], **kwargs)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    return tok, model


def build_prompt(tok, item):
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    messages = [{"role": "user", "content": PROMPT.format(q=item["q"], opts=opts)}]
    # enable_thinking=False switches off Qwen3/SmolLM3 thinking; other templates ignore it.
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def first_device(model):
    return next(model.parameters()).device


def test_model(cfg, device, max_new_tokens):
    result = {"name": cfg["name"], "id": cfg["id"], "dtype": cfg["dtype"], "status": "FAIL", "problems": []}

    ok, err = check_access(cfg["id"])
    if not ok:
        result["problems"].append(f"Cannot access model: {err}")
        return result

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tok, model = load_model(cfg, device)
    result["load_seconds"] = round(time.perf_counter() - t0, 1)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    answers, gen_tokens, gen_time = [], 0, 0.0
    for i, item in enumerate(QUESTIONS):
        prompt = build_prompt(tok, item)
        # Chat template already contains BOS, so don't add it twice.
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(first_device(model))

        with torch.inference_mode():
            if i == 0:  # numeric health check: catches fp16 overflow (NaN/inf)
                logits = model(**inputs).logits
                if not torch.isfinite(logits).all():
                    result["problems"].append("Non-finite logits (NaN/inf): change dtype in configs/base.yaml")
            t1 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
            gen_time += time.perf_counter() - t1

        new_ids = out[0, inputs["input_ids"].shape[1]:]
        gen_tokens += len(new_ids)
        raw = tok.decode(new_ids, skip_special_tokens=False)
        text = tok.decode(new_ids, skip_special_tokens=True).strip()
        m = ANSWER_RE.search(text)
        pred = m.group(1).upper() if m else None
        answers.append({"expected": item["answer"], "predicted": pred, "output": text[:200]})
        if "<think>" in raw and "</think>" not in raw:
            result["problems"].append(f"Q{i+1}: thinking mode leaked into output")
        if not text:
            result["problems"].append(f"Q{i+1}: empty output")
        elif pred is None:
            result["problems"].append(f"Q{i+1}: answer format not followed -> {text[:80]!r}")

    result["answers"] = answers
    result["format_ok"] = sum(a["predicted"] is not None for a in answers)
    result["correct"] = sum(a["predicted"] == a["expected"] for a in answers)
    result["tokens_per_second"] = round(gen_tokens / gen_time, 1) if gen_time else None
    if device == "cuda":
        result["peak_vram_gb"] = round(sum(torch.cuda.max_memory_allocated(d)
                                           for d in range(torch.cuda.device_count())) / 1e9, 2)
    if not result["problems"]:
        result["status"] = "PASS"

    del model, tok
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="outputs/smoke")
    ap.add_argument("--tiny", action="store_true", help="Quick laptop check with the tiny model only")
    ap.add_argument("--models", nargs="*", help="Test only these model names")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = check_environment()
    device = "cuda" if env["cuda_available"] else "cpu"
    print("=== ENVIRONMENT ===")
    print(json.dumps(env, indent=2))

    models = [cfg["tiny_model"]] if args.tiny else cfg["models"]
    if args.models:
        models = [m for m in models if m["name"] in args.models]

    results = []
    for m in models:
        print(f"\n=== TESTING {m['name']} ({m['id']}, {m['dtype']}) ===", flush=True)
        try:
            r = test_model(m, device, cfg["smoke_test"]["max_new_tokens"])
        except Exception as e:  # noqa: BLE001 - one broken model must not stop the others
            r = {"name": m["name"], "id": m["id"], "status": "FAIL",
                 "problems": [f"Crashed: {type(e).__name__}: {e}"], "traceback": traceback.format_exc()}
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("answers", "traceback")}, indent=2), flush=True)

    # Save exact library versions so the working setup can be locked later.
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out_dir / "environment.txt").write_text(freeze)

    overall = "PASS" if results and all(r["status"] == "PASS" for r in results) else "FAIL"
    report = {"overall": overall, "environment": env, "models": results}
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== SUMMARY ===")
    print(f"{'model':<14}{'status':<8}{'format':<8}{'correct':<9}{'tok/s':<8}{'VRAM GB':<8}")
    for r in results:
        print(f"{r['name']:<14}{r['status']:<8}{str(r.get('format_ok', '-')) + '/5':<8}"
              f"{str(r.get('correct', '-')) + '/5':<9}{str(r.get('tokens_per_second', '-')):<8}"
              f"{str(r.get('peak_vram_gb', '-')):<8}")
        for p in r.get("problems", []):
            print(f"    - {p}")
    print(f"\nOVERALL: {overall}   (full report: {out_dir / 'smoke_report.json'})")


if __name__ == "__main__":
    main()
