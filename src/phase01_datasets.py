"""Phase 1: dataset loading, cleaning, splitting.

Output (work/phase1/):
  test.jsonl, validation.jsonl   one question per line:
    {id, split, dataset, question, options{A..}, answer (letter or null), should_abstain, source_id}
  manifest.json                  counts, label distribution, file hashes

Datasets:
  medqa        MedQA-USMLE 4-option (test drawn from official test, validation from official train)
  pubmedqa     PubMedQA expert-labelled 1k (yes/no/maybe -> A/B/C); test and validation disjoint
  unanswerable MedQA train questions with the correct option REMOVED (3 options left).
               Correct behaviour is to abstain. Used to measure safe abstention.
"""
import random
from collections import Counter

from datasets import load_dataset

from src.common import (banner, check, load_config, norm_key, norm_space, phase_args, phase_dir,
                        set_seed, write_jsonl, write_manifest)

LETTERS = "ABCDEFGH"


def clean_mcq(rows, n_options, stats, tag):
    """Drop malformed rows and duplicate questions."""
    out, seen = [], set()
    for r in rows:
        opts = r["options"]
        bad = (not r["question"] or len(opts) != n_options or any(not v for v in opts.values())
               or r["answer"] not in opts)
        if bad:
            stats[f"{tag}_dropped_malformed"] += 1
            continue
        k = norm_key(r["question"])
        if k in seen:
            stats[f"{tag}_dropped_duplicate"] += 1
            continue
        seen.add(k)
        out.append(r)
    return out


def load_medqa(cfg, stats):
    ds = load_dataset(cfg["data"]["medqa_hf"])
    check("train" in ds and "test" in ds, f"MedQA has train and test splits (found {list(ds)})")
    out = {}
    for split in ("train", "test"):
        rows = []
        for i, r in enumerate(ds[split]):
            opts = r["options"]
            if isinstance(opts, list):  # some mirrors store [{key, value}]
                opts = {o["key"]: o["value"] for o in opts}
            opts = {k: norm_space(v) for k, v in sorted(opts.items())}
            rows.append({"id": f"medqa-{split}-{i:05d}", "dataset": "medqa",
                         "question": norm_space(r["question"]), "options": opts,
                         "answer": r.get("answer_idx"), "should_abstain": False, "source_id": None})
        stats[f"medqa_{split}_raw"] = len(rows)
        out[split] = clean_mcq(rows, 4, stats, f"medqa_{split}")
    # Leakage guard: no training question may duplicate a test question.
    test_keys = {norm_key(r["question"]) for r in out["test"]}
    before = len(out["train"])
    out["train"] = [r for r in out["train"] if norm_key(r["question"]) not in test_keys]
    stats["medqa_train_dropped_overlap_with_test"] = before - len(out["train"])
    return out


def load_pubmedqa(cfg, stats):
    ds = load_dataset(cfg["data"]["pubmedqa_hf"], cfg["data"]["pubmedqa_config"], split="train")
    mapping = {"yes": "A", "no": "B", "maybe": "C"}
    rows = []
    for r in ds:
        rows.append({"id": f"pubmedqa-{r['pubid']}", "dataset": "pubmedqa",
                     "question": norm_space(r["question"]),
                     "options": {"A": "yes", "B": "no", "C": "maybe"},
                     "answer": mapping.get(str(r["final_decision"]).strip().lower()),
                     "should_abstain": False, "source_id": None})
    stats["pubmedqa_raw"] = len(rows)
    return clean_mcq(rows, 3, stats, "pubmedqa")


def make_unanswerable(r, new_id):
    wrong = [v for k, v in r["options"].items() if k != r["answer"]]
    return {"id": new_id, "dataset": "unanswerable", "question": r["question"],
            "options": dict(zip(LETTERS, wrong)), "answer": None, "should_abstain": True,
            "source_id": r["id"]}


def usable_for_unanswerable(r):
    texts = [v.lower() for v in r["options"].values()]
    if any("all of the" in t or "none of the" in t or "both" == t[:4] for t in texts):
        return False  # removing the answer would make these options ill-defined
    keys = [norm_key(t) for t in texts]
    return len(set(keys)) == len(keys)


def main():
    args = phase_args("Phase 1: datasets").parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    sizes = cfg["data"]["tiny_sizes" if args.tiny else "sizes"]
    rng = random.Random(cfg["seed"])
    stats = Counter()
    out = phase_dir(args.work, 1)

    banner("PHASE 1: loading datasets")
    medqa = load_medqa(cfg, stats)
    pubmed = load_pubmedqa(cfg, stats)

    banner("PHASE 1: sampling fixed splits")
    medqa_test = rng.sample(medqa["test"], sizes["medqa_test"])
    train_pool = list(medqa["train"])
    rng.shuffle(train_pool)
    medqa_val = train_pool[:sizes["medqa_val"]]
    unans_pool = [r for r in train_pool[sizes["medqa_val"]:] if usable_for_unanswerable(r)]
    n_ut, n_uv = sizes["unans_test"], sizes["unans_val"]
    unans_test = [make_unanswerable(r, f"unans-test-{i:04d}") for i, r in enumerate(unans_pool[:n_ut])]
    unans_val = [make_unanswerable(r, f"unans-val-{i:04d}") for i, r in enumerate(unans_pool[n_ut:n_ut + n_uv])]

    pubmed_shuffled = list(pubmed)
    rng.shuffle(pubmed_shuffled)
    pub_test = pubmed_shuffled[:sizes["pubmedqa_test"]]
    pub_val = pubmed_shuffled[sizes["pubmedqa_test"]:sizes["pubmedqa_test"] + sizes["pubmedqa_val"]]

    test = [dict(r, split="test") for r in medqa_test + pub_test + unans_test]
    val = [dict(r, split="validation") for r in medqa_val + pub_val + unans_val]

    banner("PHASE 1: checks")
    for name, rows, n in [("medqa test", medqa_test, sizes["medqa_test"]),
                          ("medqa validation", medqa_val, sizes["medqa_val"]),
                          ("pubmedqa test", pub_test, sizes["pubmedqa_test"]),
                          ("pubmedqa validation", pub_val, sizes["pubmedqa_val"]),
                          ("unanswerable test", unans_test, n_ut),
                          ("unanswerable validation", unans_val, n_uv)]:
        check(len(rows) == n, f"{name}: {len(rows)} questions (expected {n})")
    all_rows = test + val
    check(len({r["id"] for r in all_rows}) == len(all_rows), "all ids unique")
    tk = {norm_key(r["question"]) for r in test if r["dataset"] != "unanswerable"}
    vk = {norm_key(r["question"]) for r in val if r["dataset"] != "unanswerable"}
    check(not (tk & vk), "no question text shared between test and validation")
    uk = {norm_key(r["question"]) for r in unans_test + unans_val}
    check(not (uk & (tk | vk)), "unanswerable questions do not overlap answerable ones")
    check(all(r["answer"] in r["options"] for r in all_rows if not r["should_abstain"]),
          "every answerable question has a valid answer letter")
    check(all(r["answer"] is None and len(r["options"]) == 3 for r in unans_test + unans_val),
          "every unanswerable question has no answer and 3 options")

    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "validation.jsonl", val)
    labels = {f"{s}/{d}": dict(Counter(str(r["answer"]) for r in rows if r["dataset"] == d))
              for s, rows in (("test", test), ("validation", val))
              for d in ("medqa", "pubmedqa", "unanswerable")}
    info = write_manifest(out, {"phase": 1, "tiny": args.tiny, "sizes": sizes,
                                "stats": dict(stats), "label_distribution": labels})
    print("\nLabel distribution:", labels)
    print("Cleaning stats:", dict(stats))
    print(f"\nPHASE 1 DONE: {len(test)} test + {len(val)} validation questions -> {out}")
    return info


if __name__ == "__main__":
    main()
