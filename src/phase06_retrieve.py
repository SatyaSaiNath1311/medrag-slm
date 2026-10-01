"""Phase 6: hybrid retrieval (dense MedCPT + BM25, fused with Reciprocal Rank Fusion).

Model-independent: run once, shared by all 5 LLMs.

Output (work/phase6/):
  retrieved.jsonl  {id, split, dataset, query, candidates:[{idx, chunk_id, rrf, dense_rank, bm25_rank}]}
  manifest.json    includes answer-in-top-k rate (sanity signal for retrieval quality)
"""
import bm25s
import faiss
import Stemmer

from src import encoders
from src.common import (banner, build_query, check, get_device, load_config, phase_args, phase_dir,
                        read_jsonl, set_seed, write_jsonl, write_manifest)


def rrf_fuse(dense_ids, bm25_ids, rrf_k, top_k):
    scores, d_rank, b_rank = {}, {}, {}
    for r, i in enumerate(dense_ids, 1):
        scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + r)
        d_rank[i] = r
    for r, i in enumerate(bm25_ids, 1):
        scores[i] = scores.get(i, 0.0) + 1.0 / (rrf_k + r)
        b_rank[i] = r
    best = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
    return [{"idx": i, "rrf": round(scores[i], 6), "dense_rank": d_rank.get(i), "bm25_rank": b_rank.get(i)}
            for i in best]


def answer_in_passages(q, passages):
    """Proxy for retrieval recall: does the correct option text appear in a retrieved passage?
    Only meaningful for MedQA (option texts are specific medical terms)."""
    if q["dataset"] != "medqa":
        return None
    gold = q["options"][q["answer"]].lower()
    if len(gold) < 4:
        return None
    return any(gold in p["text"].lower() for p in passages)


def main():
    args = phase_args("Phase 6: retrieval").parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    rcfg = cfg["retrieval"]
    device = get_device()
    out = phase_dir(args.work, 6)
    p1, p3 = phase_dir(args.work, 1), phase_dir(args.work, 3)

    questions = read_jsonl(p1 / "test.jsonl") + read_jsonl(p1 / "validation.jsonl")
    chunks = read_jsonl(p3 / "chunks.jsonl")
    queries = [build_query(q, rcfg["include_options"]) for q in questions]
    print(f"{len(questions)} questions, {len(chunks)} chunks")

    banner("PHASE 6: dense search")
    qemb = encoders.query_encoder(cfg, device).embed(queries, label="encoding queries")
    index = faiss.read_index(str(p3 / "dense.faiss"))
    _, dense = index.search(qemb, min(rcfg["dense_k"], index.ntotal))

    banner("PHASE 6: BM25 search")
    stemmer = Stemmer.Stemmer("english")
    bm25 = bm25s.BM25.load(str(p3 / "bm25"))
    bm25_res, _ = bm25.retrieve(bm25s.tokenize(queries, stopwords="en", stemmer=stemmer, show_progress=False),
                                k=min(rcfg["bm25_k"], len(chunks)), show_progress=False)

    banner("PHASE 6: fusion")
    rows, hits = [], []
    for qi, q in enumerate(questions):
        cands = rrf_fuse([int(i) for i in dense[qi] if i >= 0], [int(i) for i in bm25_res[qi]],
                         rcfg["rrf_k"], rcfg["top_k"])
        for c in cands:
            c["chunk_id"] = chunks[c["idx"]]["chunk_id"]
        rows.append({"id": q["id"], "split": q["split"], "dataset": q["dataset"], "query": queries[qi],
                     "candidates": cands})
        h = answer_in_passages(q, [chunks[c["idx"]] for c in cands])
        if h is not None and q["split"] == "test":
            hits.append(h)

    banner("PHASE 6: checks")
    expected = min(rcfg["top_k"], len(chunks))
    check(len(rows) == len(questions), f"retrieval done for all {len(questions)} questions")
    check(all(len(r["candidates"]) == expected for r in rows), f"every question has {expected} candidates")
    check(all(len({c['idx'] for c in r['candidates']}) == len(r['candidates']) for r in rows),
          "no duplicate candidates")
    rate = sum(hits) / len(hits) if hits else None
    print(f"  [info] MedQA test: correct option text found in top {expected}: "
          f"{rate:.1%}" if rate is not None else "  [info] no MedQA items to measure")

    write_jsonl(out / "retrieved.jsonl", rows)
    write_manifest(out, {"phase": 6, "tiny": args.tiny, "n_questions": len(rows), "settings": rcfg,
                         "medqa_test_answer_in_topk": rate})
    print(f"\nPHASE 6 DONE -> {out}")


if __name__ == "__main__":
    main()
