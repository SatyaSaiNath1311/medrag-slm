import csv, json, math

PREC_MIN, NPV_MIN, THR = 0.75, 0.75, 0.5   # fixed before labeling
d = json.load(open("outputs/analysis_nli/dev_pairs_scored.json"))
score = {p["pair_id"]: p["entail_prob"] for p in d["pairs"]}
rows = list(csv.DictReader(open("outputs/analysis_nli/label_sheet.csv", encoding="utf-8-sig")))
col = "your_label_1_supports_0_not"
lab = []
for r in rows:
    v = r[col].strip()
    assert v in ("0", "1"), f"unlabeled or invalid label, pair_id={r['pair_id']}: {v!r}"
    lab.append((score[int(r["pair_id"])] >= THR, int(v)))

ys = {y for _, y in lab}
assert ys == {0, 1}, f'labels contain only {ys}; need both 0 and 1'

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; dn = 1 + z * z / n; c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - a) / dn), min(1.0, (c + a) / dn)

tp = sum(1 for pos, y in lab if pos and y == 1); fp = sum(1 for pos, y in lab if pos and y == 0)
fn = sum(1 for pos, y in lab if not pos and y == 1); tn = sum(1 for pos, y in lab if not pos and y == 0)
n = len(lab); n_pos, n_neg = tp + fp, tn + fn
prec = tp / n_pos if n_pos else float("nan"); npv = tn / n_neg if n_neg else float("nan")

po = (tp + tn) / n
pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

w_pos = sum(1 for s in score.values() if s >= THR) / len(score)   # real share of NLI-positive pairs
w_acc = w_pos * prec + (1 - w_pos) * npv

print(f"labeled: {n} | human positives: {tp + fn} | NLI-positive: {n_pos} | NLI-negative: {n_neg}")
print(f"confusion  TP={tp} FP={fp} FN={fn} TN={tn}")
lo, hi = wilson(tp, n_pos); print(f"precision (NLI-positive that humans accept): {prec:.1%} (CI {lo:.1%}-{hi:.1%})")
lo, hi = wilson(tn, n_neg); print(f"NPV       (NLI-negative that humans reject): {npv:.1%} (CI {lo:.1%}-{hi:.1%})")
print(f"Cohen's kappa on the sheet: {kappa:.2f} | raw agreement: {po:.1%} (sheet is 50/50 by design)")
print(f"prevalence-weighted accuracy (NLI-positive share {w_pos:.1%} of all 817 pairs): {w_acc:.1%}")
print("\nPASS" if prec >= PREC_MIN and npv >= NPV_MIN else "\nFAIL", f"(criteria: precision >= {PREC_MIN:.0%} and NPV >= {NPV_MIN:.0%})")
