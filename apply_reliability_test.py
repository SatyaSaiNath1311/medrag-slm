import json, math
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.reliability_score import (FEATURE_NAMES, load_validation_dataset,
                                   fit_and_evaluate_cv, calibrate_conformal_threshold)

SEED = 42
VAL_DIR = "outputs/sc_val/medpsy-4b"
assert "sc_val" in VAL_DIR
TEST_FILES = {
    "medqa": "outputs/sc_test/medpsy-4b/outputs/sc_test/medpsy-4b/medqa.jsonl",
    "pubmedqa": "outputs/sc_test_s2_pubmedqa/outputs/sc_test/medpsy-4b/pubmedqa.jsonl",
    "unanswerable_v2": "outputs/sc_test_s3_unanswerable/outputs/sc_test/medpsy-4b/unanswerable_v2.jsonl",
}

# ---- 1. Fit on VALIDATION ONLY (identical to reliability_score.py) ----
val = load_validation_dataset(Path(VAL_DIR))
cv = fit_and_evaluate_cv(val, n_splits=5, seed=SEED)
calib = calibrate_conformal_threshold(val, cv["oof_probs"], target_far=0.05)
thr = calib["threshold"]

X = np.array([[d[f] for f in FEATURE_NAMES] for d in val], float)
y = np.array([1 if d["should_abstain"] else 0 for d in val], int)
pipe = Pipeline([("scaler", StandardScaler()),
                 ("clf", LogisticRegression(random_state=SEED, max_iter=1000))])
pipe.fit(X, y)
coef = dict(zip(FEATURE_NAMES, pipe.named_steps["clf"].coef_[0]))
inter = float(pipe.named_steps["clf"].intercept_[0])

# ---- 2. Reproduction check against the validation report ----
EXPECTED = {"agreement": 0.0261, "distinct_answers": 0.4921, "truncation_rate": 0.6775,
            "greedy_truncated": 0.3059, "insufficient_rate": 0.0, "greedy_insufficient_info": 0.0}
for k, v in EXPECTED.items():
    assert abs(coef[k] - v) < 1e-3, f"coef mismatch {k}: {coef[k]:.4f} vs report {v}"
assert abs(inter - (-0.8850)) < 1e-3, f"intercept mismatch: {inter:.4f}"
print("Reproduction check PASSED (coefficients + intercept match validation report)")
print(f"Validation threshold: {thr:.4f} | val FAR {calib['far_answerable']:.2%} "
      f"({calib['n_false_abst']}/{calib['n_ans']}) | val recall {calib['recall']:.2%} "
      f"({calib['n_true_abst']}/{calib['n_unans']})")

# ---- 3. Load TEST rows (same feature/label rules as the validation loader) ----
def load_test(ds, path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ctrl = bool(r.get("control", False)) if ds == "unanswerable_v2" else False
        should = (not ctrl) if ds == "unanswerable_v2" else False
        tr = r.get("truncations") or []
        ins = r.get("insufficient_info") or []
        rows.append({
            "id": r["id"], "dataset": ds, "category": r.get("category"), "control": ctrl,
            "gold": r.get("gold"), "majority_answer": r.get("majority_answer"),
            "should_abstain": should,
            "agreement": float(r.get("agreement", 0.0)),
            "distinct_answers": float(r.get("distinct_answers", 1.0)),
            "truncation_rate": float(np.mean(tr)) if tr else 0.0,
            "insufficient_rate": float(np.mean(ins)) if ins else 0.0,
            "greedy_truncated": 1.0 if r.get("greedy_truncated") else 0.0,
            "greedy_insufficient_info": 1.0 if r.get("greedy_insufficient_info") else 0.0,
        })
    return rows

test = []
for ds, p in TEST_FILES.items():
    rows = load_test(ds, p)
    print(f"loaded {ds}: {len(rows)} rows")
    test += rows
assert len(test) == 500 + 500 + 140

Xt = np.array([[d[f] for f in FEATURE_NAMES] for d in test], float)
probs = pipe.predict_proba(Xt)[:, 1]          # predict only, no fitting on test
abst = probs >= thr
for d, p, a in zip(test, probs, abst):
    d["p_abstain"] = float(p); d["abstain"] = bool(a)

# ---- 4. Report ----
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; dn = 1 + z*z/n; c = p + z*z/(2*n)
    a = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-a)/dn, (c+a)/dn)

def rate(rows, label):
    n = len(rows); k = sum(r["abstain"] for r in rows)
    lo, hi = wilson(k, n)
    print(f"  {label:28s} {k:4d}/{n:<4d} = {k/max(n,1):6.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    return {"k": k, "n": n, "rate": k / max(n, 1), "ci": [lo, hi]}

res = {"threshold": thr}
ans = [r for r in test if not r["should_abstain"]]
una = [r for r in test if r["should_abstain"]]
print("\n== FALSE ABSTENTION on answerable questions (target <= 5% from validation) ==")
res["far_all"] = rate(ans, "all answerable")
res["far_medqa"] = rate([r for r in ans if r["dataset"] == "medqa"], "MedQA")
res["far_pubmedqa"] = rate([r for r in ans if r["dataset"] == "pubmedqa"], "PubMedQA")
res["far_controls"] = rate([r for r in ans if r["dataset"] == "unanswerable_v2"], "unanswerable_v2 controls")
print("\n== RECALL on unanswerable (non-control) ==")
res["recall_all"] = rate(una, "all unanswerable")
res["recall_by_category"] = {}
for c in sorted({r["category"] for r in una}):
    res["recall_by_category"][c] = rate([r for r in una if r["category"] == c], c)

print("\n== SELECTIVE ACCURACY (SC majority vote, answered questions only) ==")
for ds in ("medqa", "pubmedqa"):
    rows = [r for r in test if r["dataset"] == ds]
    full = sum(r["majority_answer"] == r["gold"] for r in rows) / len(rows)
    kept = [r for r in rows if not r["abstain"]]
    sel = sum(r["majority_answer"] == r["gold"] for r in kept) / max(len(kept), 1)
    print(f"  {ds:10s} no-abstain acc {full:.1%} | answered {len(kept)}/{len(rows)} "
          f"(coverage {len(kept)/len(rows):.1%}) | selective acc {sel:.1%}")
    res[f"selective_{ds}"] = {"full_acc": full, "coverage": len(kept)/len(rows), "selective_acc": sel}

out = Path("outputs/analysis_test"); out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / "reliability_test_results.json", "w"), indent=2)
print("\nsaved outputs/analysis_test/reliability_test_results.json")
