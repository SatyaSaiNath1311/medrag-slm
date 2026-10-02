"""Phases 4, 5 and 8 in one loop, so each model is downloaded and loaded only once.

For each of the 5 models:
  Phase 4  format check: 20 validation questions, baseline + RAG prompts; parse rate must be >= threshold.
           A model that fails is skipped (no GPU time wasted) and reported.
  Phase 5  Baseline QA   (no retrieval)        -> work/phase5/<model>.jsonl
  Phase 8  RAG QA        (top-5 evidence)      -> work/phase8/<model>.jsonl
Runs on test AND validation (validation is needed later to tune abstention in Phase 9).
Resumable: questions already in an output file are skipped.

Usage:
  python -m src.qa_pipeline --work <dir>                      all 5 models
  python -m src.qa_pipeline --work <dir> --models qwen3-4b    one model
  python -m src.qa_pipeline --work <dir> --tiny               laptop check with the tiny model
"""
import json
import random
import time
from pathlib import Path

from src.common import (append_jsonl, banner, get_device, load_config, phase_args, phase_dir, read_jsonl,
                        set_seed, write_manifest)
from src.llm import LLM, baseline_prompt, parse_answer, parse_citations, rag_prompt


def sample_stratified_tiny(rows, n_per_group, seed):
    rng = random.Random(seed)
    medqa = [r for r in rows if r["dataset"] == "medqa" and not r["should_abstain"]]
    pubmed = [r for r in rows if r["dataset"] == "pubmedqa" and not r["should_abstain"]]
    abstain = [r for r in rows if r["should_abstain"]]
    selected = []
    for group in (medqa, pubmed, abstain):
        n = min(n_per_group, len(group))
        selected.extend(rng.sample(group, n))
    return selected


def run_mode(llm, mode, questions, evidence, qcfg, out_path):
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    if not todo:
        print(f"  {mode}: already complete ({len(done)} rows)")
        return
    prompts = {q["id"]: (baseline_prompt(q) if mode == "baseline"
                         else rag_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"]))
               for q in todo}
    todo.sort(key=lambda q: -len(prompts[q["id"]]))  # longest first: memory problems appear immediately
    max_new = qcfg[f"max_new_tokens_{mode}"]
    bs, t0 = qcfg["batch_size"], time.time()
    for s in range(0, len(todo), bs):
        batch = todo[s:s + bs]
        res = llm.generate_safe([prompts[q["id"]] for q in batch], [list(q["options"]) for q in batch], max_new)
        rows = []
        for q, r in zip(batch, res):
            parsed_pred = parse_answer(r["raw_output"], q["options"])
            parsed = parsed_pred is not None
            lp = r["letter_probs"]
            prob_pred = max(lp, key=lp.get) if lp else None

            if parsed:
                pred = parsed_pred
                pred_source = "parsed"
            elif prob_pred is not None:
                pred = prob_pred
                pred_source = "logprob_fallback"
            else:
                pred = None
                pred_source = "none"

            correct = pred is not None and pred == q["answer"]
            row = {"id": q["id"], "split": q["split"], "dataset": q["dataset"], "model": llm.cfg["name"],
                   "mode": mode, "should_abstain": q["should_abstain"], "gold": q["answer"], "pred": pred,
                   "correct": correct, "parsed": parsed, "pred_source": pred_source,
                   "letter_probs": lp, "confidence": max(lp.values()) if lp else None,
                   "prob_pred": prob_pred, **r,
                   "pred": pred, "correct": correct, "parsed": parsed, "pred_source": pred_source}
            if mode == "rag":
                passages = evidence[q["id"]]["passages"]
                cites = parse_citations(r["raw_output"], len(passages))
                row["citations"] = cites
                row["cited_chunk_ids"] = [passages[c - 1]["chunk_id"] for c in cites]
                row["evidence_chunk_ids"] = [p["chunk_id"] for p in passages]
                row["top_rerank_score"] = passages[0]["rerank_score"]
            rows.append(row)
        append_jsonl(out_path, rows)
        n = min(s + bs, len(todo))
        if (s // bs) % 10 == 0 or n == len(todo):
            print(f"  {mode}: {n}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)


def summarise(path):
    rows = read_jsonl(path)
    test = [r for r in rows if r["split"] == "test"]
    ans = [r for r in test if not r["should_abstain"]]
    acc = lambda rs: round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None
    rate = lambda rs, fn: round(sum(fn(r) for r in rs) / len(rs), 4) if rs else None

    test_acc = acc(ans)
    recomputed_acc = round(sum(r["correct"] for r in ans) / len(ans), 4) if ans else None
    assert test_acc == recomputed_acc, f"Summary accuracy {test_acc} != recomputed {recomputed_acc}"

    datasets = sorted(list({r["dataset"] for r in rows}))
    by_dataset = {}
    for d in datasets:
        d_rows = [r for r in rows if r["dataset"] == d]
        d_ans = [r for r in d_rows if r["split"] == "test" and not r["should_abstain"]]
        by_dataset[d] = {
            "n_rows": len(d_rows),
            "accuracy": acc(d_ans),
            "parse_rate": rate(d_rows, lambda r: bool(r.get("parsed"))),
            "fallback_rate": rate(d_rows, lambda r: r.get("pred_source") == "logprob_fallback"),
        }

    return {"n_rows": len(rows),
            "test_accuracy_all_answerable": test_acc,
            "test_accuracy_medqa": acc([r for r in ans if r["dataset"] == "medqa"]),
            "test_accuracy_pubmedqa": acc([r for r in ans if r["dataset"] == "pubmedqa"]),
            "parse_rate": rate(rows, lambda r: bool(r.get("parsed"))),
            "fallback_rate": rate(rows, lambda r: r.get("pred_source") == "logprob_fallback"),
            "letter_prob_rate": rate(rows, lambda r: r.get("letter_probs") is not None),
            "mean_seconds_per_question": round(sum(r["seconds"] for r in rows) / len(rows), 4) if rows else None,
            "by_dataset": by_dataset}


def format_check(llm, val, evidence, qcfg, n, threshold):
    rng = random.Random(0)
    sample = rng.sample(val, min(n, len(val)))
    report = {}
    for mode in ("baseline", "rag"):
        prompts = [baseline_prompt(q) if mode == "baseline"
                   else rag_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"]) for q in sample]
        res = []
        for s in range(0, len(sample), qcfg["batch_size"]):
            res += llm.generate_safe(prompts[s:s + qcfg["batch_size"]],
                                     [list(q["options"]) for q in sample[s:s + qcfg["batch_size"]]],
                                     qcfg[f"max_new_tokens_{mode}"])
        parsed = [parse_answer(r["raw_output"], q["options"]) is not None for q, r in zip(sample, res)]
        rate = sum(parsed) / len(parsed)
        fallback_count = sum(not p and r.get("letter_probs") is not None for p, r in zip(parsed, res))
        report[mode] = {"parse_rate": round(rate, 3),
                        "fallback_count": fallback_count,
                        "letter_prob_rate": round(sum(r["letter_probs"] is not None for r in res) / len(res), 3),
                        "examples": [r["raw_output"][:150] for r in res[:2]],
                        "unparsed_examples": [r["raw_output"][:150] for r, p in zip(res, parsed) if not p][:3]}
        print(f"  check {mode}: parse rate {rate:.0%}, fallback count {fallback_count}, examples: {report[mode]['examples'][:2]}")
    report["passed"] = all(report[m]["parse_rate"] >= threshold for m in ("baseline", "rag"))
    return report


def delete_from_cache(model_id):
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        revs = [r.commit_hash for repo in info.repos if repo.repo_id == model_id for r in repo.revisions]
        if revs:
            info.delete_revisions(*revs).execute()
            print(f"  removed {model_id} from disk cache")
    except Exception as e:  # noqa: BLE001
        print(f"  (cache cleanup skipped: {e})")


def main():
    ap = phase_args("Phases 4, 5, 8: model check, baseline QA, RAG QA")
    ap.add_argument("--models", nargs="*", help="Only these model names")
    ap.add_argument("--keep-cache", action="store_true", help="Do not delete model files after use")
    ap.add_argument("--strict", action="store_true", help="Exit with an error if any model did not finish")
    ap.add_argument("--dry-run", action="store_true", help="Print selected models and exit without running")
    ap.add_argument("--check-only", action="store_true", help="Run only Phase 4 format check for selected models and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    qcfg = dict(cfg["qa"])
    device = get_device()
    p1, p7 = phase_dir(args.work, 1), phase_dir(args.work, 7)
    p4, p5, p8 = (phase_dir(args.work, n) for n in (4, 5, 8))

    test, val = read_jsonl(p1 / "test.jsonl"), read_jsonl(p1 / "validation.jsonl")
    evidence = {r["id"]: r for r in read_jsonl(p7 / "evidence.jsonl")}
    questions = test + val
    missing = [q["id"] for q in questions if q["id"] not in evidence]
    if missing:
        raise AssertionError(f"[CHECK FAILED] {len(missing)} questions have no Phase 7 evidence")

    all_models = {m["name"]: m for m in cfg["models"] + [cfg["tiny_model"]]}
    if args.models:
        unknown = [name for name in args.models if name not in all_models]
        if unknown:
            raise ValueError(f"Unknown model name(s): {unknown}. Available models: {list(all_models.keys())}")
        models = [all_models[name] for name in args.models]
    elif args.tiny:
        models = [cfg["tiny_model"]]
    else:
        models = cfg["models"]

    if args.dry_run:
        mode_str = " (check-only)" if args.check_only else ""
        print(f"Selected {len(models)} model(s){mode_str}: {[m['name'] for m in models]}")
        return

    if args.check_only:
        n_check = qcfg["check_questions"]
        threshold = qcfg["min_parse_rate"]
    else:
        n_check = 5 if args.tiny else qcfg["check_questions"]
        threshold = qcfg["tiny_min_parse_rate"] if args.tiny else qcfg["min_parse_rate"]

    if args.tiny:
        test = sample_stratified_tiny(test, qcfg["tiny_questions"], cfg["seed"])
        val = sample_stratified_tiny(val, qcfg["tiny_questions"], cfg["seed"] + 1)
        questions = test + val

    status = {}
    check_reports = {}
    for m in models:
        banner(f"MODEL {m['name']} ({m['id']}, {m['dtype']}) on {device}")
        try:
            t0 = time.time()
            llm = LLM(m, device)
            print(f"  loaded in {time.time() - t0:.0f}s")
            check = format_check(llm, val, evidence, qcfg, n_check, threshold)
            check_reports[m["name"]] = check
            (p4 / f"{m['name']}.json").write_text(json.dumps(check, indent=2))
            if not check["passed"]:
                status[m["name"]] = "FAILED format check (see phase4/%s.json)" % m["name"]
                print(f"  !! {status[m['name']]}")
            else:
                if not args.check_only:
                    run_mode(llm, "baseline", questions, evidence, qcfg, p5 / f"{m['name']}.jsonl")
                    run_mode(llm, "rag", questions, evidence, qcfg, p8 / f"{m['name']}.jsonl")
                status[m["name"]] = "DONE"
            llm.close()
        except Exception as e:  # noqa: BLE001 - one broken model must not stop the others
            import traceback
            traceback.print_exc()
            status[m["name"]] = f"ERROR: {type(e).__name__}: {str(e)[:200]}"
        if device == "cuda" and not args.keep_cache:
            delete_from_cache(m["id"])

    if args.check_only:
        banner("PHASE 4 FORMAT CHECK SUMMARY")
        print(f"{'model':<14}{'baseline parse':<16}{'RAG parse':<12}{'fallback count':<16}{'status':<10}")
        print("-" * 68)
        for m in models:
            name = m["name"]
            chk = check_reports.get(name, {})
            b_pr = str(chk.get("baseline", {}).get("parse_rate", "-"))
            r_pr = str(chk.get("rag", {}).get("parse_rate", "-"))
            b_fb = chk.get("baseline", {}).get("fallback_count", 0)
            r_fb = chk.get("rag", {}).get("fallback_count", 0)
            total_fb = f"{b_fb}/{r_fb}" if (b_fb or r_fb) else "0"
            st = "PASS" if chk.get("passed") else ("FAIL" if name in check_reports else "ERROR")
            print(f"{name:<14}{b_pr:<16}{r_pr:<12}{total_fb:<16}{st:<10}")
            if name in check_reports:
                print("  Baseline example outputs:")
                for ex in chk.get("baseline", {}).get("examples", [])[:2]:
                    print(f"    - {ex!r}")
                print("  RAG example outputs:")
                for ex in chk.get("rag", {}).get("examples", [])[:2]:
                    print(f"    - {ex!r}")
                print()
        write_manifest(p4, {"phase": 4, "tiny": args.tiny, "check_only": True, "device": device,
                            "status": status, "settings": qcfg})
        ok = all(v == "DONE" for v in status.values())
        print(f"\nPHASE 4 FORMAT CHECK {'COMPLETE' if ok else 'FINISHED WITH PROBLEMS'}")
        if args.strict and not ok:
            raise SystemExit(1)
        return

    banner("RESULTS SO FAR (test set)")
    summary = {}
    print(f"{'model':<14}{'status':<10}{'baseline acc':<14}{'RAG acc':<10}{'parse B/R':<14}{'sec/q B/R':<12}")
    for m in models:
        name = m["name"]
        b = summarise(p5 / f"{name}.jsonl") if (p5 / f"{name}.jsonl").exists() else {}
        r = summarise(p8 / f"{name}.jsonl") if (p8 / f"{name}.jsonl").exists() else {}

        if b:
            b_ans = [row for row in read_jsonl(p5 / f"{name}.jsonl") if row["split"] == "test" and not row["should_abstain"]]
            b_recomputed = round(sum(row["correct"] for row in b_ans) / len(b_ans), 4) if b_ans else None
            assert b.get("test_accuracy_all_answerable") == b_recomputed, (
                f"Baseline accuracy mismatch: {b.get('test_accuracy_all_answerable')} != {b_recomputed}"
            )
        if r:
            r_ans = [row for row in read_jsonl(p8 / f"{name}.jsonl") if row["split"] == "test" and not row["should_abstain"]]
            r_recomputed = round(sum(row["correct"] for row in r_ans) / len(r_ans), 4) if r_ans else None
            assert r.get("test_accuracy_all_answerable") == r_recomputed, (
                f"RAG accuracy mismatch: {r.get('test_accuracy_all_answerable')} != {r_recomputed}"
            )

        summary[name] = {"status": status.get(name), "baseline": b, "rag": r}
        print(f"{name:<14}{('DONE' if status.get(name) == 'DONE' else 'PROBLEM'):<10}"
              f"{str(b.get('test_accuracy_all_answerable')):<14}{str(r.get('test_accuracy_all_answerable')):<10}"
              f"{str(b.get('parse_rate')) + '/' + str(r.get('parse_rate')):<14}"
              f"{str(b.get('mean_seconds_per_question')) + '/' + str(r.get('mean_seconds_per_question')):<12}")
        if status.get(name) != "DONE":
            print(f"    -> {status.get(name)}")
    Path(p8 / "summary.json").write_text(json.dumps(summary, indent=2))
    for p, n in ((p4, 4), (p5, 5), (p8, 8)):
        write_manifest(p, {"phase": n, "tiny": args.tiny, "device": device, "status": status, "settings": qcfg})
    ok = all(v == "DONE" for v in status.values())
    print(f"\nPHASES 4, 5, 8 {'COMPLETE' if ok else 'FINISHED WITH PROBLEMS (see above)'}")
    if args.strict and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
