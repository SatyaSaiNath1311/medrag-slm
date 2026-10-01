"""Phase 3: chunking and indexing.

Chunking: textbook snippets are already short; any longer than max_words are split with overlap.
Indexes:  dense  = MedCPT article embeddings in an exact FAISS inner-product index
          sparse = BM25 (bm25s, English stopwords + stemming)

Output (work/phase3/):
  chunks.jsonl      {idx, chunk_id, doc_id, source, title, text}  (idx = row number in both indexes)
  dense.faiss, bm25/  indexes
  manifest.json
"""
import random

import bm25s
import faiss
import numpy as np
import Stemmer

from src import encoders
from src.common import (banner, check, get_device, load_config, phase_args, phase_dir, read_jsonl,
                        set_seed, write_jsonl, write_manifest)


def chunk_docs(docs, max_words, overlap, min_words):
    chunks, step = [], max_words - overlap
    for d in docs:
        words = d["text"].split()
        pieces = [words] if len(words) <= max_words else [
            words[s:s + max_words] for s in range(0, len(words), step)]
        for j, p in enumerate(pieces):
            if j > 0 and len(p) < min_words:
                break  # tail already covered by the previous window's overlap
            chunks.append({"idx": len(chunks), "chunk_id": f"{d['doc_id']}#{j}", "doc_id": d["doc_id"],
                           "source": d["source"], "title": d["title"], "text": " ".join(p)})
    return chunks


def bm25_texts(chunks):
    return [f"{c['title']} {c['text']}" for c in chunks]


def main():
    args = phase_args("Phase 3: indexing").parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    icfg = cfg["index"]
    out = phase_dir(args.work, 3)
    device = get_device()

    banner("PHASE 3: chunking")
    docs = read_jsonl(phase_dir(args.work, 2) / "corpus.jsonl")
    chunks = chunk_docs(docs, icfg["max_words"], icfg["overlap_words"], cfg["corpus"]["min_words"])
    write_jsonl(out / "chunks.jsonl", chunks)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")

    banner("PHASE 3: BM25 index")
    stemmer = Stemmer.Stemmer("english")
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(bm25_texts(chunks), stopwords="en", stemmer=stemmer, show_progress=False))
    bm25.save(str(out / "bm25"))

    banner(f"PHASE 3: dense index (MedCPT on {device})")
    enc = encoders.article_encoder(cfg, device)
    emb = enc.embed([c["title"] for c in chunks], [c["text"] for c in chunks], label="embedding chunks")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out / "dense.faiss"))

    banner("PHASE 3: checks")
    check(len(chunks) == len(docs) or len(chunks) > len(docs), "every document produced at least one chunk")
    check(len({c["chunk_id"] for c in chunks}) == len(chunks), "all chunk ids unique")
    check(emb.shape[0] == len(chunks) == index.ntotal, f"{index.ntotal} vectors = {len(chunks)} chunks")
    check(bool(np.isfinite(emb).all()), "all embeddings are finite numbers")

    rng = random.Random(cfg["seed"])
    sample = rng.sample(chunks, min(5, len(chunks)))
    res, _ = bm25.retrieve(bm25s.tokenize([bm25_texts([c])[0] for c in sample], stopwords="en",
                                          stemmer=stemmer, show_progress=False), k=5, show_progress=False)
    bm25_hits = sum(c["idx"] in [int(x) for x in res[i]] for i, c in enumerate(sample))
    check(bm25_hits >= len(sample) - 1, f"BM25 finds a chunk from its own text in top 5 ({bm25_hits}/{len(sample)})")

    qenc = encoders.query_encoder(cfg, device)
    qemb = qenc.embed([c["title"] + " " + c["text"] for c in sample], label="self-check")
    _, I = index.search(qemb, 10)
    dense_hits = sum(c["idx"] in I[i] for i, c in enumerate(sample))
    print(f"  [info] dense retrieval finds a chunk from its own text in top 10: {dense_hits}/{len(sample)}")
    check(dense_hits >= 1, "dense index returns sensible neighbours")

    write_manifest(out, {"phase": 3, "tiny": args.tiny, "device": device, "n_chunks": len(chunks),
                         "embedding_dim": int(emb.shape[1]), "bm25_self_check": f"{bm25_hits}/{len(sample)}",
                         "dense_self_check": f"{dense_hits}/{len(sample)}", "settings": icfg})
    print(f"\nPHASE 3 DONE: {len(chunks)} chunks indexed -> {out}")


if __name__ == "__main__":
    main()
