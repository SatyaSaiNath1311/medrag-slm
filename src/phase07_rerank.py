"""Phase 7: reranking with the MedCPT cross-encoder; keep the best passages as evidence.

Model-independent: run once, shared by all 5 LLMs.

Output (work/phase7/):
  evidence.jsonl  {id, split, dataset, passages:[{rank, idx, chunk_id, title, text, rerank_score, rrf_rank}],
                   all_scores:[20 scores, for analysis]}
  manifest.json
"""
import numpy as np

from src import encoders
from src.common import (banner, check, get_device, load_config, phase_args, phase_dir, read_jsonl,
                        set_seed, write_jsonl, write_manifest)
from src.phase06_retrieve import answer_in_passages


def main():
    args = phase_args("Phase 7: reranking").parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    top_k = cfg["rerank"]["top_k"]
    device = get_device()
    out = phase_dir(args.work, 7)
    p1, p3, p6 = (phase_dir(args.work, n) for n in (1, 3, 6))

    questions = {q["id"]: q for q in read_jsonl(p1 / "test.jsonl") + read_jsonl(p1 / "validation.jsonl")}
    chunks = read_jsonl(p3 / "chunks.jsonl")
    retrieved = read_jsonl(p6 / "retrieved.jsonl")

    banner(f"PHASE 7: reranking on {device}")
    flat_q, flat_a, owner = [], [], []
    for qi, r in enumerate(retrieved):
        for c in r["candidates"]:
            ch = chunks[c["idx"]]
            flat_q.append(r["query"])
            flat_a.append(f"{ch['title']}. {ch['text']}")
            owner.append(qi)
    scores = encoders.cross_encoder(cfg, device).score(flat_q, flat_a, label="scoring pairs")

    rows, pos, hits = [], 0, []
    for r in retrieved:
        n = len(r["candidates"])
        s = scores[pos:pos + n]
        pos += n
        order = np.argsort(-s, kind="stable")[:top_k]
        passages = []
        for rank, j in enumerate(order, 1):
            c = r["candidates"][int(j)]
            ch = chunks[c["idx"]]
            passages.append({"rank": rank, "idx": c["idx"], "chunk_id": ch["chunk_id"], "title": ch["title"],
                             "text": ch["text"], "rerank_score": round(float(s[j]), 4), "rrf_rank": int(j) + 1})
        rows.append({"id": r["id"], "split": r["split"], "dataset": r["dataset"], "passages": passages,
                     "all_scores": [round(float(x), 4) for x in s]})
        q = questions[r["id"]]
        h = answer_in_passages(q, passages)
        if h is not None and q["split"] == "test":
            hits.append(h)

    banner("PHASE 7: checks")
    check(pos == len(scores) == len(owner), "every candidate was scored exactly once")
    check(bool(np.isfinite(scores).all()), "all rerank scores are finite")
    expected = min(top_k, len(retrieved[0]["candidates"]))
    check(all(len(r["passages"]) == expected for r in rows), f"every question has {expected} evidence passages")
    check(len(rows) == len(questions), f"evidence for all {len(questions)} questions")
    rate = sum(hits) / len(hits) if hits else None
    if rate is not None:
        print(f"  [info] MedQA test: correct option text found in top {expected} evidence: {rate:.1%}")

    write_jsonl(out / "evidence.jsonl", rows)
    write_manifest(out, {"phase": 7, "tiny": args.tiny, "settings": cfg["rerank"],
                         "medqa_test_answer_in_evidence": rate})
    print(f"\nPHASE 7 DONE -> {out}")


if __name__ == "__main__":
    main()
