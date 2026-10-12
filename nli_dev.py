import json, glob, random, csv, re, collections
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

N_Q_PER_MODEL, SEED = 100, 42
MODEL_CANDIDATES = ["pritamdeka/PubMedBERT-MNLI-MedNLI", "cross-encoder/nli-deberta-v3-base"]
OUT = Path("outputs/analysis_nli"); OUT.mkdir(parents=True, exist_ok=True)
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]

val_q = {r["id"]: r for r in load("outputs/kaggle_build/work/phase1/validation.jsonl") if r["dataset"] == "medqa"}
ev = {r["id"]: r for r in load("outputs/kaggle_build/work/phase7/evidence.jsonl")
      if r["split"] == "validation" and r["dataset"] == "medqa"}
assert isinstance(next(iter(ev.values()))["passages"], list), "passages is not a list"

EV_RE = re.compile(r"Evidence\s*:\s*((?:\[\s*\d+\s*\][\s,;and]*)+)", re.I)
def final_cites(t):
    sec = t.rsplit("</think>", 1)[-1] if "</think>" in t else t
    m = EV_RE.findall(sec)
    return sorted({int(n) for n in re.findall(r"\d+", m[-1])}) if m else None

# ---- build (question, chosen answer, cited passage) pairs from the final Evidence line only ----
rng = random.Random(SEED)
pairs = []
for f in sorted(glob.glob("outputs/kaggle_qa/full/*/work/phase8/*.jsonl")):
    if "shard" in f or "_rag" in f:
        continue
    model = Path(f).stem
    rows = [r for r in load(f) if r["split"] == "validation" and r["dataset"] == "medqa"
            and not r.get("truncated") and r.get("pred") in (val_q.get(r["id"], {}).get("options") or {})]
    rows = sorted(rows, key=lambda r: r["id"])
    for r in rng.sample(rows, min(N_Q_PER_MODEL, len(rows))):
        q, cites = val_q[r["id"]], final_cites(r["raw_output"])
        if not cites:
            continue
        psg = ev[r["id"]]["passages"]
        for c in cites:
            if 1 <= c <= len(psg):
                pairs.append({"pair_id": len(pairs), "model": model, "qid": r["id"], "question": q["question"],
                              "pred": r["pred"], "option_text": q["options"][r["pred"]],
                              "correct": r["pred"] == q["answer"], "cite": c,
                              "chunk_id": psg[c-1].get("chunk_id"), "passage": psg[c-1]["text"]})
print("pairs:", len(pairs), "| per model:", dict(collections.Counter(p["model"] for p in pairs)))

# ---- NLI ----
tok = mdl = ent_idx = None
for name in MODEL_CANDIDATES:
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSequenceClassification.from_pretrained(name).eval()
        labs = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
        idx = [k for k, v in labs.items() if "entail" in v]
        if len(idx) == 1:
            ent_idx, used = idx[0], name; break
        print("skipping", name, "- cannot identify entailment label:", labs)
    except Exception as e:
        print("skipping", name, "-", repr(e)[:150])
assert ent_idx is not None, "no usable NLI model"
print("NLI model:", used, "| entailment label index:", ent_idx)

@torch.no_grad()
def entail_probs(prem, hyp, bs=8):
    out = []
    for i in range(0, len(prem), bs):
        enc = tok(prem[i:i+bs], hyp[i:i+bs], truncation="only_first", max_length=512,
                  padding=True, return_tensors="pt")
        out += torch.softmax(mdl(**enc).logits, -1)[:, ent_idx].tolist()
    return out

hyps = [f"The correct answer to the question is: {p['option_text']}." for p in pairs]
for p, s in zip(pairs, entail_probs([p["passage"] for p in pairs], hyps)):
    p["entail_prob"] = round(s, 4)
json.dump({"nli_model": used, "pairs": pairs}, open(OUT / "dev_pairs_scored.json", "w"))

# ---- diagnostic: does NLI support separate correct from wrong answers (string match did not) ----
def rate(ps): return f"{np.mean([p['entail_prob'] >= 0.5 for p in ps]):.1%} (n={len(ps)})" if ps else "n/a"
print("\nPair-level entailment rate (>=0.5)")
print("  correct answers:", rate([p for p in pairs if p["correct"]]))
print("  wrong answers:  ", rate([p for p in pairs if not p["correct"]]))
by_q = collections.defaultdict(list)
for p in pairs: by_q[(p["model"], p["qid"])].append(p)
sup = [(any(x["entail_prob"] >= 0.5 for x in v), v[0]["correct"]) for v in by_q.values()]
print("Answer-level (any cited passage entails)")
print("  correct answers:", f"{np.mean([s for s, c in sup if c]):.1%} (n={sum(c for _, c in sup)})")
print("  wrong answers:  ", f"{np.mean([s for s, c in sup if not c]):.1%} (n={sum(not c for _, c in sup)})")
for m in sorted({p["model"] for p in pairs}):
    print(" ", m, rate([p for p in pairs if p["model"] == m]))

# ---- 50-pair label sheet (NLI score hidden; sampled half above / half below 0.5 so both classes appear) ----
hi = [p for p in pairs if p["entail_prob"] >= 0.5]; lo = [p for p in pairs if p["entail_prob"] < 0.5]
pick = rng.sample(hi, min(25, len(hi))) + rng.sample(lo, min(25, len(lo)))
rng.shuffle(pick)
with open(OUT / "label_sheet.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["pair_id", "question", "chosen_answer", "passage", "your_label_1_supports_0_not"])
    for p in pick:
        w.writerow([p["pair_id"], p["question"], f"{p['pred']}. {p['option_text']}", p["passage"][:1200], ""])
print("\nwrote", OUT / "label_sheet.csv", "| scores are in dev_pairs_scored.json - do not open before labeling")
