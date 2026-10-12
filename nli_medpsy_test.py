import json, re, math, collections
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI, THR = "pritamdeka/PubMedBERT-MNLI-MedNLI", 0.5               # frozen
PREC, PREC_CI, FNR, FNR_CI = 0.52, (0.335, 0.70), 0.04, (0.007, 0.195)   # dev-validated: precision 13/25, NPV 24/25
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
OUT = Path("outputs/analysis_nli"); OUT.mkdir(parents=True, exist_ok=True)

rows = load("outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl")
qs = {r["id"]: r for r in load("outputs/kaggle_build/work/phase1/test.jsonl") if r["dataset"] == "medqa"}
ev = {r["id"]: r for r in load("outputs/kaggle_build/work/phase7/evidence.jsonl")
      if r["split"] == "test" and r["dataset"] == "medqa"}
assert len(rows) == 500 and all(r["id"] in qs and r["id"] in ev for r in rows)
mis = sum(1 for r in rows if [p["chunk_id"] for p in ev[r["id"]]["passages"]] != list(r["evidence_chunk_ids"]))
print("evidence join mismatches:", mis, "/ 500")
assert mis == 0, "passage join is wrong; stop"

EV_RE = re.compile(r"Evidence\s*:\s*((?:\[\s*\d+\s*\][\s,;and]*)+)", re.I)
def final_cites(t):
    sec = t.rsplit("</think>", 1)[-1] if "</think>" in t else t
    m = EV_RE.findall(sec)
    return sorted({int(n) for n in re.findall(r"\d+", m[-1])}) if m else None

n_trunc = n_noline = 0; items = []
for r in rows:
    if r.get("truncated"): n_trunc += 1; continue
    q = qs[r["id"]]; c = final_cites(r["raw_output"])
    psg = ev[r["id"]]["passages"]
    c = [x for x in (c or []) if 1 <= x <= len(psg)]
    if r.get("pred") not in q["options"] or not c: n_noline += 1; continue
    items.append((r, q, psg, c))
N = len(items)
print(f"scorable: {N}/500 ({N/500:.1%}) | truncated: {n_trunc} | no usable Evidence line: {n_noline}")

tok = AutoTokenizer.from_pretrained(NLI); mdl = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
labs = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
ent = [k for k, v in labs.items() if "entail" in v]; assert len(ent) == 1; ent = ent[0]

@torch.no_grad()
def entail(prem, hyp, bs=8):
    out = []
    for i in range(0, len(prem), bs):
        enc = tok(prem[i:i+bs], hyp[i:i+bs], truncation="only_first", max_length=512, padding=True, return_tensors="pt")
        out += torch.softmax(mdl(**enc).logits, -1)[:, ent].tolist()
    return out

prem, hyp, key = [], [], []
for qi, (r, q, psg, c) in enumerate(items):
    h = f"The correct answer to the question is: {q['options'][r['pred']]}."
    for k in range(1, len(psg) + 1):
        prem.append(psg[k-1]["text"]); hyp.append(h); key.append((qi, k))
print("scoring", len(prem), "pairs (CPU, ~10-20 min)...")
sc = dict(zip(key, entail(prem, hyp)))

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; dn = 1 + z*z/n; cc = p + z*z/(2*n); a = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return max(0, (cc-a)/dn), min(1, (cc+a)/dn)
def fmt(k, n): lo, hi = wilson(k, n); return f"{k}/{n} = {k/max(n,1):.1%} (CI {lo:.1%}-{hi:.1%})"

cited_pairs = [(qi, k) for qi, (r, q, psg, c) in enumerate(items) for k in c]
k_pair = sum(sc[p] >= THR for p in cited_pairs)
print("\n== Citation precision (UPPER BOUND, NLI-entailed share of cited passages) ==")
print("  pair level:", fmt(k_pair, len(cited_pairs)))
rate = k_pair / len(cited_pairs)
lo = rate*PREC_CI[0] + (1-rate)*FNR_CI[0]; hi = rate*PREC_CI[1] + (1-rate)*FNR_CI[1]
est = rate*PREC + (1-rate)*FNR
print(f"  corrected for dev precision/NPV: ~{est:.1%} (range {lo:.1%}-{hi:.1%})")

print("\n== Answer-level support (UPPER BOUND): any cited passage entailed ==")
sup = [any(sc[(qi, k)] >= THR for k in c) for qi, (r, q, psg, c) in enumerate(items)]
cor = [r["pred"] == r["gold"] for (r, q, psg, c) in items]
print("  all answers:  ", fmt(sum(sup), N))
print("  correct:      ", fmt(sum(s for s, y in zip(sup, cor) if y), sum(cor)))
print("  wrong:        ", fmt(sum(s for s, y in zip(sup, cor) if not y), N - sum(cor)))

print("\n== Citation recall (UPPER BOUND) ==")
avail = [any(sc[(qi, k)] >= THR for k in range(1, len(psg)+1)) for qi, (r, q, psg, c) in enumerate(items)]
print("  questions with >=1 entailed retrieved passage:", fmt(sum(avail), N))
print("  ...of those, model cited an entailed passage: ", fmt(sum(s for s, a in zip(sup, avail) if a), sum(avail)))

json.dump({"scorable": N, "truncated": n_trunc, "no_line": n_noline, "pair_entail": [k_pair, len(cited_pairs)],
           "corrected_est": [est, lo, hi], "answer_support": [sum(sup), N]},
          open(OUT / "medpsy_test_citation_metrics.json", "w"), indent=2)
print("\nsaved", OUT / "medpsy_test_citation_metrics.json")
