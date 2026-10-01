"""Phase 2: medical knowledge base.

Source: MedRAG Textbooks (18 standard USMLE textbooks, pre-split into snippets of <= 1000 characters).
Cleaning: normalise whitespace, drop very short snippets, drop exact duplicates, keep metadata.

Output (work/phase2/):
  corpus.jsonl   {doc_id, source, title, text}
  manifest.json
"""
import hashlib
from collections import Counter
from itertools import islice

from datasets import load_dataset

from src.common import (banner, check, load_config, norm_key, norm_space, phase_args, phase_dir,
                        set_seed, write_jsonl, write_manifest)


def main():
    args = phase_args("Phase 2: knowledge base").parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ccfg = cfg["corpus"]
    out = phase_dir(args.work, 2)

    banner("PHASE 2: loading corpus")
    if args.tiny:
        stream = load_dataset(ccfg["hf"], split="train", streaming=True)
        raw = list(islice(stream, ccfg["tiny_rows"]))
    else:
        raw = load_dataset(ccfg["hf"], split="train")
    print(f"raw snippets: {len(raw)}")

    banner("PHASE 2: cleaning")
    stats, seen, rows = Counter(), set(), []
    for r in raw:
        text = norm_space(r.get("content"))
        title = norm_space(r.get("title"))
        if len(text.split()) < ccfg["min_words"]:
            stats["dropped_short"] += 1
            continue
        h = hashlib.md5(norm_key(text).encode()).hexdigest()
        if h in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(h)
        rows.append({"doc_id": str(r["id"]), "source": title, "title": title, "text": text})
    stats["raw"] = len(raw)
    stats["kept"] = len(rows)

    banner("PHASE 2: checks")
    check(len({r["doc_id"] for r in rows}) == len(rows), "all doc ids unique")
    check(all(r["text"] and r["source"] for r in rows), "no empty text or source")
    minimum = ccfg["tiny_min_docs"] if args.tiny else ccfg["min_docs"]
    check(len(rows) >= minimum, f"{len(rows)} documents kept (minimum {minimum})")

    write_jsonl(out / "corpus.jsonl", rows)
    per_source = dict(Counter(r["source"] for r in rows).most_common())
    write_manifest(out, {"phase": 2, "tiny": args.tiny, "source_dataset": ccfg["hf"],
                         "stats": dict(stats), "documents_per_source": per_source})
    print("Stats:", dict(stats))
    print("Sources:", per_source)
    print(f"\nPHASE 2 DONE: {len(rows)} documents -> {out}")


if __name__ == "__main__":
    main()
