import json, math, os, random
random.seed(42)
def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
def strict(r): return bool(r.get("parsed")) and not r.get("truncated") and bool(r.get("correct"))
def fallback(r): return bool(r.get("correct"))
def paired(A, B, ids, f):
    a = [f(A[i]) for i in ids]; b = [f(B[i]) for i in ids]; n = len(ids)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    diffs = []
    for _ in range(10000):
        s = random.choices(range(n), k=n)
        diffs.append(sum(a[j] - b[j] for j in s) / n)
    diffs.sort()
    return sum(a)/n, sum(b)/n, only_a, only_b, mcnemar(only_a, only_b), diffs[250], diffs[9749]
MP = {"C1": "outputs/kaggle_qa/medpsy_final/work/phase5/medpsy-4b.jsonl",
      "C2": "outputs/kaggle_qa/medpsy_rag/work/phase8/medpsy-4b.jsonl"}
QW = {c: f"outputs/v3/qwen3-4b-thinking-2507/{c}/qwen3-4b-thinking-2507.jsonl" for c in ("C1", "C2")}
M = {c: load(p) for c, p in MP.items()}
Q = {c: load(p) for c, p in QW.items()}
ids = sorted(set(Q["C1"]) & set(Q["C2"]) & set(M["C1"]) & set(M["C2"]))
assert len(ids) == 500, len(ids)
lines = [f"# Pair B, equal 1,024-token budget, MedQA test (n={len(ids)})",
         "Base = Qwen3-4B-Thinking-2507; domain = MedPsy-4B. Same prompt, cap, evidence and fallback rule.",
         "Strict = parsed and not truncated and correct (primary). With-fallback = stored correct, i.e. a forced letter after a truncated thought (secondary).", ""]
for cond, name in (("C1", "closed-book"), ("C2", "RAG")):
    lines.append(f"## {cond} {name}")
    for label, S in (("MedPsy-4B", M[cond]), ("Qwen3-4B-Thinking-2507", Q[cond])):
        t = sum(1 for i in ids if S[i].get("truncated"))
        parts = []
        for mname, f in (("strict", strict), ("fallback", fallback)):
            k = sum(1 for i in ids if f(S[i])); lo, hi = wilson(k, len(ids))
            parts.append(f"{mname} {k}/{len(ids)} = {k/len(ids):.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"- {label}: truncated {t}/{len(ids)}; " + "; ".join(parts))
    for mname, f in (("strict", strict), ("fallback", fallback)):
        am, ab, x, y, p, lo, hi = paired(M[cond], Q[cond], ids, f)
        lines.append(f"- paired MedPsy minus base ({mname}): {am-ab:+.3f} [{lo:+.3f}, {hi:+.3f}]; MedPsy-only correct {x}, base-only correct {y}, McNemar p = {p:.2e}")
    lines.append("")
lines.append("## RAG gain (C2 minus C1), with-fallback, paired within model")
for label, S in (("MedPsy-4B", M), ("Qwen3-4B-Thinking-2507", Q)):
    a2, a1, x, y, p, lo, hi = paired(S["C2"], S["C1"], ids, fallback)
    lines.append(f"- {label}: {a2-a1:+.3f} [{lo:+.3f}, {hi:+.3f}]; RAG-only correct {x}, closed-book-only correct {y}, McNemar p = {p:.2e}")
os.makedirs("outputs/analysis", exist_ok=True)
open("outputs/analysis/pair_b_equal_budget.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
