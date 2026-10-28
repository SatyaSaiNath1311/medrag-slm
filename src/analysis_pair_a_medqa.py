import json, math, os, random
from pathlib import Path
random.seed(42)
M, Q = "medpsy-1.7b", "qwen3-1.7b-thinking"

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c - h, c + h
def mcn(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0]*len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj
strict = lambda r: bool(r.get("parsed")) and not r.get("truncated") and bool(r.get("correct"))
fb = lambda r: bool(r.get("correct"))
tr = lambda r: bool(r.get("truncated"))
def boot(fn, n, reps=10000):
    d = []
    for _ in range(reps):
        s = random.choices(range(n), k=n); d.append(fn(s))
    d.sort(); return d[250], d[9749]
def paired(a, b):
    n = len(a)
    x = sum(1 for u, v in zip(a, b) if u and not v); y = sum(1 for u, v in zip(a, b) if v and not u)
    lo, hi = boot(lambda s: sum(a[j] - b[j] for j in s) / n, n)
    return sum(a)/n - sum(b)/n, lo, hi, x, y, mcn(x, y)
def run_analysis(repo_root=".", out_path="outputs/analysis/pair_a_medqa.md", expected_n=500):
    repo_root = Path(repo_root)
    def load(m, c):
        p = repo_root / f"outputs/v3/{m}/{c}/{m}.jsonl"
        if not p.exists():
            return None
        rows = {r["id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}
        assert all(r.get("split") == "test" and r.get("dataset") == "medqa" for r in rows.values()), str(p)
        return rows

    D = {c: (load(M, c), load(Q, c)) for c in ("C1", "C2")}
    D = {c: v for c, v in D.items() if v[0] is not None and v[1] is not None}
    L = ["# Pair A, MedQA test, 1,024-token cap (generic = qwen3-1.7b-thinking, domain = medpsy-1.7b)",
         "Strict = parsed and not truncated and correct (PRIMARY). Fallback = stored correct, incl. forced letters after truncation (SECONDARY). Truncation is analysed as its own outcome.", ""]
    fam = []
    for c, (m, q) in D.items():
        ids = sorted(set(m) & set(q)); n = len(ids)
        if expected_n is not None:
            assert n == expected_n, (c, n, expected_n)
        L.append(f"## {c}")
        for name, S in ((M, m), (Q, q)):
            t = sum(tr(S[i]) for i in ids); s = sum(strict(S[i]) for i in ids); f = sum(fb(S[i]) for i in ids)
            L.append(f"- {name}: truncated {t}/{n} ({t/n:.1%}); strict {s}/{n} = {s/n:.3f} [{wilson(s,n)[0]:.3f}, {wilson(s,n)[1]:.3f}]; fallback {f}/{n} = {f/n:.3f} [{wilson(f,n)[0]:.3f}, {wilson(f,n)[1]:.3f}]")
        for label, fn in (("strict", strict), ("fallback", fb), ("truncated", tr)):
            d, lo, hi, x, y, p = paired([fn(m[i]) for i in ids], [fn(q[i]) for i in ids])
            fam.append((f"{c} {label}", p))
            L.append(f"- paired {M} minus {Q} ({label}): {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; {M}-only {x}, {Q}-only {y}; McNemar raw p = {p:.2e}")
        both = [i for i in ids if not tr(m[i]) and not tr(q[i]) and m[i].get("parsed") and q[i].get("parsed")]
        if both:
            a = [bool(m[i].get("correct")) for i in both]; b = [bool(q[i].get("correct")) for i in both]
            d, lo, hi, x, y, p = paired(a, b)
            L.append(f"- DESCRIPTIVE, selected subset where BOTH finished (n={len(both)}): {M} {sum(a)/len(both):.3f}, {Q} {sum(b)/len(both):.3f}; diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        L.append("")
    adj = holm([p for _, p in fam])
    L.append("## Holm-adjusted p-values (one family: every paired test above)")
    for (lab, p), a in zip(fam, adj):
        L.append(f"- {lab}: raw {p:.2e}, Holm {a:.2e}")
    if len(D) == 2:
        ids = sorted(set.intersection(*[set(D[c][k]) for c in D for k in (0, 1)])); n = len(ids)
        L.append(""); L.append("## RAG effect (C2 minus C1), paired within model")
        for name, k in ((M, 0), (Q, 1)):
            for label, fn in (("strict", strict), ("fallback", fb), ("truncated", tr)):
                d, lo, hi, x, y, p = paired([fn(D["C2"][k][i]) for i in ids], [fn(D["C1"][k][i]) for i in ids])
                L.append(f"- {name} ({label}): {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; C2-only {x}, C1-only {y}; McNemar p = {p:.2e}")
        for label, fn in (("strict", strict), ("fallback", fb)):
            gm = [fn(D["C2"][0][i]) - fn(D["C1"][0][i]) for i in ids]; gq = [fn(D["C2"][1][i]) - fn(D["C1"][1][i]) for i in ids]
            lo, hi = boot(lambda s: sum(gm[j] - gq[j] for j in s) / n, n)
            L.append(f"- difference in RAG effect, {M} minus {Q} ({label}): {(sum(gm)-sum(gq))/n:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L) + "\n"


def main():
    report = run_analysis()
    print(report)


if __name__ == "__main__":
    main()

