"""Phases 4, 5 and 8 in one loop, so each model is downloaded and loaded only once.

For each of the 5+1 models:
  Phase 4  format check: 20 validation questions, baseline + RAG prompts; parse rate must be >= threshold.
           A model that fails is skipped (no GPU time wasted) and reported.
  Phase 5  Baseline QA   (no retrieval)        -> work/phase5/<model>.jsonl
  Phase 8  RAG QA        (top-5 evidence)      -> work/phase8/<model>.jsonl
Runs on test AND validation (validation is needed later to tune abstention in Phase 9).
Resumable: questions already in an output file are skipped.

Reasoning model (medpsy-4b):
  Uses generate_reasoning_safe instead of generate_safe.  Prompts are built by the
  reasoning_* prompt functions.  The row schema is identical to standard models
  (same fields: letter_probs, confidence, pred, pred_source, parsed, gen_tokens, seconds, …)
  so all downstream analysis scripts work unchanged.

  Extra field in reasoning rows: "truncated" (bool) — true when the model hit
  max_new_tokens_reasoning without producing a final "Answer:" line.

Split flag (--split):
  test         run on test questions only
  val_subset   run on a 80-question stratified sample of the validation set (seed 42):
               60 MedQA + 20 PubMedQA, used for the PILOT run of medpsy-4b
  all_needed   run on test + validation (default, same as previous behaviour)

Pilot mode is controlled externally by runner.py (PILOT=True): it selects medpsy-4b,
modes=["baseline","context"], split="val_subset", single GPU.

Usage:
  python -m src.qa_pipeline --work <dir>                      all 5+1 models
  python -m src.qa_pipeline --work <dir> --models qwen3-4b    one model
  python -m src.qa_pipeline --work <dir> --tiny               laptop check with the tiny model
  python -m src.qa_pipeline --work <dir> --split val_subset --models medpsy-4b
"""
import json
import random
import threading
import time
from pathlib import Path

import psutil
import torch

from src.common import (append_jsonl, banner, get_device, load_config, phase_args, phase_dir, read_jsonl,
                        set_seed, write_manifest)
from src.llm import (LLM, baseline_idk_prompt, baseline_prompt, context_prompt, get_idk_letter, get_idk_options,
                     parse_answer, parse_citations, rag_idk_prompt, rag_prompt,
                     reasoning_baseline_prompt, reasoning_context_prompt, reasoning_rag_prompt,
                     extract_final_answer_from_reasoning)


class RSSProfiler:
    def __init__(self, interval_sec: float = 0.2):
        self.interval_sec = interval_sec
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self._stop_event = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop_event.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            self._stop_event.wait(self.interval_sec)

    def start(self):
        try:
            self.peak_rss = self.process.memory_info().rss
        except Exception:
            self.peak_rss = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            rss = self.process.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
        except Exception:
            pass
        return self.peak_rss


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


def sample_val_subset(val_rows, seed=42):
    """Stratified 80-question sample of the validation set for the PILOT run.

    Draws 60 MedQA (non-abstain) + 20 PubMedQA (non-abstain) questions, sorted
    by id for reproducibility, then sampled with the fixed seed.
    Used only when --split val_subset is requested.
    """
    rng = random.Random(seed)
    medqa = sorted([r for r in val_rows if r.get("dataset") == "medqa" and not r.get("should_abstain")],
                   key=lambda x: x["id"])
    pubmedqa = sorted([r for r in val_rows if r.get("dataset") == "pubmedqa" and not r.get("should_abstain")],
                      key=lambda x: x["id"])
    n_medqa = min(60, len(medqa))
    n_pubmedqa = min(20, len(pubmedqa))
    return rng.sample(medqa, n_medqa) + rng.sample(pubmedqa, n_pubmedqa)


def run_mode(llm, mode, questions, evidence, qcfg, out_path):
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    if not todo:
        print(f"  {mode}: already complete ({len(done)} rows)")
        return

    is_reasoning = getattr(llm, "is_reasoning", False)

    if mode == "baseline":
        if is_reasoning:
            prompts = {q["id"]: reasoning_baseline_prompt(q) for q in todo}
        else:
            prompts = {q["id"]: baseline_prompt(q) for q in todo}
        options_map = {q["id"]: q["options"] for q in todo}
        letters_map = {q["id"]: list(q["options"]) for q in todo}
    elif mode == "rag":
        if is_reasoning:
            prompts = {q["id"]: reasoning_rag_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"]) for q in todo}
        else:
            prompts = {q["id"]: rag_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"]) for q in todo}
        options_map = {q["id"]: q["options"] for q in todo}
        letters_map = {q["id"]: list(q["options"]) for q in todo}
    elif mode == "context":
        max_chars = qcfg["passage_max_chars"] * 3
        if is_reasoning:
            prompts = {q["id"]: reasoning_context_prompt(q, max_chars) for q in todo}
        else:
            prompts = {q["id"]: context_prompt(q, max_chars) for q in todo}
        options_map = {q["id"]: q["options"] for q in todo}
        letters_map = {q["id"]: list(q["options"]) for q in todo}
    elif mode == "baseline_idk":
        prompts = {q["id"]: baseline_idk_prompt(q) for q in todo}
        options_map = {q["id"]: get_idk_options(q) for q in todo}
        letters_map = {q["id"]: list(get_idk_options(q).keys()) for q in todo}
    elif mode == "rag_idk":
        prompts = {q["id"]: rag_idk_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"]) for q in todo}
        options_map = {q["id"]: get_idk_options(q) for q in todo}
        letters_map = {q["id"]: list(get_idk_options(q).keys()) for q in todo}
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Sort longest prompt first: memory problems appear immediately
    todo.sort(key=lambda q: -len(prompts[q["id"]]))

    if is_reasoning:
        # Reasoning model: always use model-level batch_size (default 8), max_new_tokens_reasoning
        max_new = llm.cfg.get("max_new_tokens_reasoning") or qcfg.get("max_new_tokens_reasoning") or qcfg["max_new_tokens_rag"]
        bs = llm.cfg.get("batch_size") or qcfg["batch_size"]
        source_desc = "model config (batch_size)" if llm.cfg.get("batch_size") else "qa config"
        print(f"  {mode} [reasoning]: batch_size={bs} (from {source_desc}), max_new={max_new}", flush=True)
    else:
        if "baseline" in mode:
            max_new = qcfg.get(f"max_new_tokens_{mode}") or qcfg["max_new_tokens_baseline"]
            bs = llm.cfg.get("batch_size_baseline") or qcfg["batch_size"]
            source_desc = "model config (batch_size_baseline)" if llm.cfg.get("batch_size_baseline") else "qa config"
        else:
            max_new = qcfg.get(f"max_new_tokens_{mode}") or qcfg["max_new_tokens_rag"]
            bs = llm.cfg.get("batch_size_rag") or qcfg["batch_size"]
            source_desc = "model config (batch_size_rag)" if llm.cfg.get("batch_size_rag") else "qa config"
        print(f"  {mode}: batch_size={bs} (from {source_desc})", flush=True)

    t0 = time.time()
    is_idk = "idk" in mode
    for s in range(0, len(todo), bs):
        batch = todo[s:s + bs]
        batch_prompts = [prompts[q["id"]] for q in batch]
        batch_letters = [letters_map[q["id"]] for q in batch]

        if is_reasoning:
            res = llm.generate_reasoning_safe(batch_prompts, batch_letters, max_new)
        else:
            res = llm.generate_safe(batch_prompts, batch_letters, max_new)

        rows = []
        for q, r in zip(batch, res):
            opts = options_map[q["id"]]
            idk_let = get_idk_letter(q) if is_idk else None

            if is_reasoning:
                # Fields are pre-computed inside generate_reasoning
                pred = r.pop("_pred", None)
                pred_source = r.pop("_pred_source", "none")
                parsed = r.pop("_parsed", False)
            else:
                parsed_pred = parse_answer(r["raw_output"], opts, idk_letter=idk_let)
                parsed = parsed_pred is not None
                lp = r["letter_probs"]
                prob_pred_raw = max(lp, key=lp.get) if lp else None

                if is_idk and prob_pred_raw == idk_let:
                    prob_pred = "ABSTAIN"
                else:
                    prob_pred = prob_pred_raw

                if parsed:
                    pred = parsed_pred
                    pred_source = "parsed"
                elif prob_pred is not None:
                    pred = prob_pred
                    pred_source = "logprob_fallback"
                else:
                    pred = None
                    pred_source = "none"

            lp = r["letter_probs"]
            abstained = (pred == "ABSTAIN")
            correct = pred is not None and pred == q["answer"]
            row = {"id": q["id"], "split": q["split"], "dataset": q["dataset"], "model": llm.cfg["name"],
                   "mode": mode, "should_abstain": q["should_abstain"], "gold": q["answer"], "pred": pred,
                   "correct": correct, "parsed": parsed, "pred_source": pred_source,
                   "letter_probs": lp, "confidence": max(lp.values()) if lp else None,
                   "prob_pred": max(lp, key=lp.get) if lp else None,
                   "load_source": getattr(llm, "load_source", None),
                   "abstained": abstained, **r}
            # Reasoning-specific field (set to False for non-reasoning models for schema consistency)
            if "truncated" not in row:
                row["truncated"] = False
            if mode in ("rag", "rag_idk"):
                passages = evidence[q["id"]]["passages"]
                cites = parse_citations(r["raw_output"], len(passages))
                row["citations"] = cites
                row["cited_chunk_ids"] = [passages[c - 1]["chunk_id"] for c in cites]
                row["evidence_chunk_ids"] = [p["chunk_id"] for p in passages]
                row["top_rerank_score"] = passages[0]["rerank_score"] if passages else None
            elif mode == "context":
                row["citations"] = parse_citations(r["raw_output"], 1)
            rows.append(row)
        append_jsonl(out_path, rows)
        n_done = min(s + bs, len(todo))
        elapsed = time.time() - t0
        eta_str = ""
        if n_done > 0 and n_done < len(todo):
            eta_sec = elapsed / n_done * (len(todo) - n_done)
            eta_str = f"  ETA {eta_sec:.0f}s"
        if (s // bs) % 10 == 0 or n_done == len(todo):
            print(f"  {mode}: {n_done}/{len(todo)}  ({elapsed:.0f}s){eta_str}", flush=True)


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


def sample_format_check(val, seed):
    rng = random.Random(seed)
    medqa = sorted([q for q in val if q["dataset"] == "medqa" and not q["should_abstain"]], key=lambda x: x["id"])
    pubmed = sorted([q for q in val if q["dataset"] == "pubmedqa" and not q["should_abstain"]], key=lambda x: x["id"])
    unans = sorted([q for q in val if q["should_abstain"]], key=lambda x: x["id"])
    return (rng.sample(medqa, min(8, len(medqa))) +
            rng.sample(pubmed, min(8, len(pubmed))) +
            rng.sample(unans, min(4, len(unans))))


def format_check(llm, val, evidence, qcfg, threshold, seed):
    is_reasoning = getattr(llm, "is_reasoning", False)
    sample = sample_format_check(val, seed)
    report = {"load_source": getattr(llm, "load_source", None)}
    for mode in ("baseline", "rag"):
        if is_reasoning:
            prompts = [reasoning_baseline_prompt(q) if mode == "baseline"
                       else reasoning_rag_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"])
                       for q in sample]
        else:
            prompts = [baseline_prompt(q) if mode == "baseline"
                       else rag_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"]) for q in sample]
        res = []
        for s in range(0, len(sample), qcfg["batch_size"]):
            letters_batch = [list(q["options"]) for q in sample[s:s + qcfg["batch_size"]]]
            if is_reasoning:
                max_new = llm.cfg.get("max_new_tokens_reasoning") or qcfg["max_new_tokens_rag"]
                res += llm.generate_reasoning_safe(prompts[s:s + qcfg["batch_size"]], letters_batch, max_new)
            else:
                res += llm.generate_safe(prompts[s:s + qcfg["batch_size"]], letters_batch,
                                         qcfg[f"max_new_tokens_{mode}"])
        rows = []
        for q, r in zip(sample, res):
            if is_reasoning:
                pred = r.pop("_pred", None)
                pred_source = r.pop("_pred_source", "none")
                parsed = r.pop("_parsed", False)
            else:
                parsed_pred = parse_answer(r["raw_output"], q["options"])
                parsed = parsed_pred is not None
                lp = r.get("letter_probs")
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

            if mode == "rag":
                passages = evidence[q["id"]]["passages"]
                cites = parse_citations(r["raw_output"], len(passages))
            else:
                cites = []

            rows.append({
                "id": q["id"],
                "dataset": q["dataset"],
                "should_abstain": q["should_abstain"],
                "raw_output": r["raw_output"],
                "pred": pred,
                "parsed": parsed,
                "pred_source": pred_source,
                "citations": cites,
                "load_source": getattr(llm, "load_source", None),
            })

        parsed_flags = [row["parsed"] for row in rows]
        rate = sum(parsed_flags) / len(parsed_flags) if parsed_flags else 0.0
        fallback_count = sum(1 for row in rows if not row["parsed"] and row["pred_source"] == "logprob_fallback")

        datasets = sorted(list({q["dataset"] for q in sample}))
        parse_rate_by_dataset = {}
        for d in datasets:
            d_rows = [row for row in rows if row["dataset"] == d]
            parse_rate_by_dataset[d] = round(sum(1 for row in d_rows if row["parsed"]) / len(d_rows), 3) if d_rows else None

        report[mode] = {
            "parse_rate": round(rate, 3),
            "parse_rate_by_dataset": parse_rate_by_dataset,
            "fallback_count": fallback_count,
            "letter_prob_rate": round(sum(r.get("letter_probs") is not None for r in res) / len(res), 3) if res else 0.0,
            "examples": [r["raw_output"][:150] for r in res[:2]],
            "unparsed_examples": [r["raw_output"][:150] for r, p in zip(res, parsed_flags) if not p][:3],
            "rows": rows,
        }
        print(f"  check {mode}: parse rate {rate:.0%} (by dataset: {parse_rate_by_dataset}), fallback count {fallback_count}, examples: {report[mode]['examples'][:2]}")
    report["passed"] = all(report[m]["parse_rate"] >= threshold for m in ("baseline", "rag"))
    return report


def ensure_pubmedqa_contexts(questions, cfg):
    """Ensure every PubMedQA question has non-empty context attached."""
    pubmed_rows = [q for q in questions if q.get("dataset") == "pubmedqa"]
    missing = [q for q in pubmed_rows if not q.get("context")]
    if missing:
        from datasets import load_dataset
        ds = load_dataset(cfg["data"]["pubmedqa_hf"], cfg["data"]["pubmedqa_config"], split="train")
        from src.phase01_datasets import format_pubmedqa_context
        ctx_map = {str(r["pubid"]): format_pubmedqa_context(r.get("context")) for r in ds}
        for q in missing:
            pubid = str(q.get("source_id") or q["id"].replace("pubmedqa-", ""))
            q["context"] = ctx_map.get(pubid, "")
    for q in pubmed_rows:
        assert bool(q.get("context")), f"PubMedQA row {q.get('id')} has no context attached"


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


def run_profiling(models, cfg, qcfg, questions, evidence, work_dir, device, keep_cache=False):
    banner("PROFILING BENCHMARK")
    profile_dir = Path(work_dir) / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    out_path = profile_dir / "profiling.json"

    # Sample 20 test questions: 10 MedQA + 10 PubMedQA, fixed seed
    rng = random.Random(42)
    test_medqa = sorted(
        [q for q in questions if q.get("split") == "test" and q.get("dataset") == "medqa" and not q.get("should_abstain")],
        key=lambda q: q["id"],
    )
    test_pubmedqa = sorted(
        [q for q in questions if q.get("split") == "test" and q.get("dataset") == "pubmedqa" and not q.get("should_abstain")],
        key=lambda q: q["id"],
    )
    sample_medqa = rng.sample(test_medqa, min(10, len(test_medqa)))
    sample_pubmedqa = rng.sample(test_pubmedqa, min(10, len(test_pubmedqa)))
    questions_20 = sample_medqa + sample_pubmedqa

    print(f"Profiling sample: {len(sample_medqa)} MedQA + {len(sample_pubmedqa)} PubMedQA = {len(questions_20)} total questions (seed 42)")

    profiling_results = {}
    if out_path.exists():
        try:
            profiling_results = json.loads(out_path.read_text())
        except Exception:
            pass

    rss_profiler = RSSProfiler(interval_sec=0.2)

    for m in models:
        banner(f"PROFILING MODEL: {m['name']} ({m['id']}, {m['dtype']}) on {device}")
        m_profile = {"model_id": m["id"], "modes": {}}

        # Measure model load time
        t_load0 = time.time()
        llm = LLM(m, device)
        load_time = time.time() - t_load0
        m_profile["model_load_time_sec"] = round(load_time, 3)
        print(f"  Model loaded in {load_time:.2f}s")

        is_reasoning = getattr(llm, "is_reasoning", False)

        # Run modes: baseline, rag, context (PubMedQA only)
        for mode in ("baseline", "rag", "context"):
            if mode == "baseline":
                mode_questions = questions_20
                if is_reasoning:
                    prompts = {q["id"]: reasoning_baseline_prompt(q) for q in mode_questions}
                else:
                    prompts = {q["id"]: baseline_prompt(q) for q in mode_questions}
                bs = m.get("batch_size_baseline") or m.get("batch_size") or qcfg["batch_size"]
                max_new = (m.get("max_new_tokens_reasoning") if is_reasoning
                           else qcfg.get("max_new_tokens_baseline") or qcfg["max_new_tokens_rag"])
            elif mode == "rag":
                mode_questions = questions_20
                if is_reasoning:
                    prompts = {q["id"]: reasoning_rag_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"])
                               for q in mode_questions}
                else:
                    prompts = {q["id"]: rag_prompt(q, evidence[q["id"]]["passages"], qcfg["passage_max_chars"])
                               for q in mode_questions}
                bs = m.get("batch_size_rag") or m.get("batch_size") or qcfg["batch_size"]
                max_new = (m.get("max_new_tokens_reasoning") if is_reasoning
                           else qcfg.get("max_new_tokens_rag") or qcfg["max_new_tokens_rag"])
            elif mode == "context":
                mode_questions = sample_pubmedqa  # PubMedQA only
                max_chars = qcfg["passage_max_chars"] * 3
                if is_reasoning:
                    prompts = {q["id"]: reasoning_context_prompt(q, max_chars) for q in mode_questions}
                else:
                    prompts = {q["id"]: context_prompt(q, max_chars) for q in mode_questions}
                bs = m.get("batch_size_rag") or m.get("batch_size") or qcfg["batch_size"]
                max_new = (m.get("max_new_tokens_reasoning") if is_reasoning
                           else qcfg.get("max_new_tokens_context") or qcfg["max_new_tokens_rag"])

            sorted_questions = sorted(mode_questions, key=lambda q: -len(prompts[q["id"]]))

            # Reset peak memory stats before each mode
            if torch.cuda.is_available():
                for dev_idx in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(dev_idx)

            rss_profiler.start()
            t_mode0 = time.time()

            for s in range(0, len(sorted_questions), bs):
                batch = sorted_questions[s:s + bs]
                batch_prompts = [prompts[q["id"]] for q in batch]
                batch_options = [list(q["options"]) for q in batch]
                if is_reasoning:
                    llm.generate_reasoning_safe(batch_prompts, batch_options, max_new)
                else:
                    llm.generate_safe(batch_prompts, batch_options, max_new)

            mode_duration = time.time() - t_mode0
            peak_rss = rss_profiler.stop()
            mean_s_per_q = mode_duration / len(sorted_questions) if sorted_questions else 0.0

            # Record CUDA memory per GPU
            cuda_mem = {}
            if torch.cuda.is_available():
                for dev_idx in range(torch.cuda.device_count()):
                    alloc_bytes = torch.cuda.max_memory_allocated(dev_idx)
                    res_bytes = torch.cuda.max_memory_reserved(dev_idx)
                    cuda_mem[f"cuda:{dev_idx}"] = {
                        "max_memory_allocated_bytes": alloc_bytes,
                        "max_memory_allocated_mb": round(alloc_bytes / (1024 * 1024), 2),
                        "max_memory_reserved_bytes": res_bytes,
                        "max_memory_reserved_mb": round(res_bytes / (1024 * 1024), 2),
                    }

            m_profile["modes"][mode] = {
                "num_questions": len(sorted_questions),
                "mean_seconds_per_question": round(mean_s_per_q, 4),
                "total_seconds": round(mode_duration, 2),
                "peak_process_rss_bytes": peak_rss,
                "peak_process_rss_mb": round(peak_rss / (1024 * 1024), 2),
                "cuda_memory": cuda_mem,
            }

            vram_summary = ", ".join(f"{k}: {v['max_memory_reserved_mb']:.1f}MB" for k, v in cuda_mem.items()) or "N/A (CPU)"
            print(f"  [{mode}] {len(sorted_questions)} questions in {mode_duration:.2f}s "
                  f"({mean_s_per_q:.3f} s/q) | Peak RSS: {peak_rss / (1024*1024):.1f} MB | VRAM: {vram_summary}", flush=True)

        llm.close()
        if device == "cuda" and not keep_cache:
            delete_from_cache(m["id"])

        profiling_results[m["name"]] = m_profile
        out_path.write_text(json.dumps(profiling_results, indent=2))
        print(f"Updated {out_path} with profile for {m['name']}")

    banner("PROFILING COMPLETE")
    print(f"Saved profiling benchmark results to: {out_path}")


def main():
    ap = phase_args("Phases 4, 5, 8, 10: model check, baseline QA, RAG QA, context QA")
    ap.add_argument("--models", nargs="*", help="Only these model names")
    ap.add_argument("--modes", nargs="+", choices=["baseline", "rag", "context", "baseline_idk", "rag_idk"], default=None,
                    help="Which QA modes to run (choices: baseline, rag, context, baseline_idk, rag_idk; default: baseline rag).")
    ap.add_argument("--profile", action="store_true", help="Run profiling benchmark and write work/profile/profiling.json")
    ap.add_argument("--keep-cache", action="store_true", help="Do not delete model files after use")
    ap.add_argument("--strict", action="store_true", help="Exit with an error if any model did not finish")
    ap.add_argument("--dry-run", action="store_true", help="Print selected models and exit without running")
    ap.add_argument("--check-only", action="store_true", help="Run only Phase 4 format check for selected models and exit")
    ap.add_argument("--split", choices=["test", "val_subset", "all_needed"], default="all_needed",
                    help=("Questions to run: 'test' = test split only; "
                          "'val_subset' = stratified 80-question validation sample (seed 42, PILOT use); "
                          "'all_needed' = test + validation (default)."))
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    qcfg = dict(cfg["qa"])
    device = get_device()
    p1 = phase_dir(args.work, 1)
    p4, p5, p8 = (phase_dir(args.work, n) for n in (4, 5, 8))
    p10 = phase_dir(args.work, 10)
    p12 = phase_dir(args.work, 12)

    # Determine which QA modes to run
    run_modes = list(args.modes) if args.modes else ["baseline", "rag"]
    skip_checks = all(m in ("rag", "context", "baseline_idk", "rag_idk") for m in run_modes)

    # Model selection (before data load so --dry-run doesn't need the work dir)
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
        mode_str = " (profile)" if args.profile else (" (check-only)" if args.check_only else "")
        print(f"Selected {len(models)} model(s){mode_str}: {[m['name'] for m in models]}")
        print(f"QA modes: {run_modes}" + ("  [skip Phase 4 + Phase 5]" if skip_checks else ""))
        print(f"Split: {args.split}")
        return

    print(f"QA modes to run: {run_modes}" + ("  [skip Phase 4 + Phase 5]" if skip_checks else ""), flush=True)
    print(f"Split: {args.split}", flush=True)

    test, val = read_jsonl(p1 / "test.jsonl"), read_jsonl(p1 / "validation.jsonl")
    ensure_pubmedqa_contexts(test + val, cfg)
    val_pool = list(val)

    # Build question set based on --split
    if args.split == "test":
        questions = test
    elif args.split == "val_subset":
        questions = sample_val_subset(val, seed=cfg["seed"])
        print(f"  val_subset: {len(questions)} questions (60 MedQA + 20 PubMedQA, seed {cfg['seed']})", flush=True)
    else:  # all_needed
        questions = test + val

    if "rag" in run_modes or "rag_idk" in run_modes or args.check_only or args.profile:
        p7 = phase_dir(args.work, 7)
        if (p7 / "evidence.jsonl").exists():
            evidence = {r["id"]: r for r in read_jsonl(p7 / "evidence.jsonl")}
            missing = [q["id"] for q in questions if q["id"] not in evidence]
            if missing and not args.profile:
                raise AssertionError(f"[CHECK FAILED] {len(missing)} questions have no Phase 7 evidence")
        else:
            if not args.profile:
                raise AssertionError(f"[CHECK FAILED] Phase 7 evidence not found at {p7 / 'evidence.jsonl'}")
            evidence = {}
    else:
        evidence = {}

    if args.profile:
        run_profiling(models, cfg, qcfg, questions, evidence, args.work, device, keep_cache=args.keep_cache)
        return

    threshold = qcfg["tiny_min_parse_rate"] if (args.tiny and not args.check_only) else qcfg["min_parse_rate"]

    if args.tiny:
        test = sample_stratified_tiny(test, qcfg["tiny_questions"], cfg["seed"])
        val = sample_stratified_tiny(val, qcfg["tiny_questions"], cfg["seed"] + 1)
        questions = test + val

    status = {}
    check_reports = {}
    load_sources = {}
    for m in models:
        banner(f"MODEL {m['name']} ({m['id']}, {m['dtype']}) on {device}")
        try:
            t0 = time.time()
            llm = LLM(m, device)
            print(f"  loaded in {time.time() - t0:.0f}s")
            load_sources[m["name"]] = llm.load_source

            if skip_checks or args.check_only:
                if args.check_only:
                    check = format_check(llm, val_pool, evidence, qcfg, threshold, seed=cfg["seed"])
                    check_reports[m["name"]] = check
                    (p4 / f"{m['name']}.json").write_text(json.dumps(check, indent=2))
                    if not check["passed"]:
                        status[m["name"]] = "FAILED format check (see phase4/%s.json)" % m["name"]
                        print(f"  !! {status[m['name']]}")
                        llm.close()
                        continue
                if not args.check_only:
                    print(f"  --modes {run_modes}: skipping Phase 4 (format check) and Phase 5 (baseline)", flush=True)
                    for mode in run_modes:
                        if mode == "baseline":
                            p_out, q_subset = p5, questions
                            out_file = p_out / f"{m['name']}.jsonl"
                        elif mode == "rag":
                            p_out, q_subset = p8, questions
                            out_file = p_out / f"{m['name']}.jsonl"
                        elif mode == "context":
                            p_out, q_subset = p10, [q for q in questions if q["dataset"] == "pubmedqa"]
                            out_file = p_out / f"{m['name']}.jsonl"
                        elif mode in ("baseline_idk", "rag_idk"):
                            p_out, q_subset = p12, questions
                            out_file = p_out / f"{m['name']}_{mode}.jsonl"
                        run_mode(llm, mode, q_subset, evidence, qcfg, out_file)
                status[m["name"]] = "DONE"
            else:
                check = format_check(llm, val_pool, evidence, qcfg, threshold, seed=cfg["seed"])
                check_reports[m["name"]] = check
                (p4 / f"{m['name']}.json").write_text(json.dumps(check, indent=2))
                if not check["passed"]:
                    status[m["name"]] = "FAILED format check (see phase4/%s.json)" % m["name"]
                    print(f"  !! {status[m['name']]}")
                else:
                    for mode in run_modes:
                        if mode == "baseline":
                            p_out, q_subset = p5, questions
                            out_file = p_out / f"{m['name']}.jsonl"
                        elif mode == "rag":
                            p_out, q_subset = p8, questions
                            out_file = p_out / f"{m['name']}.jsonl"
                        elif mode == "context":
                            p_out, q_subset = p10, [q for q in questions if q["dataset"] == "pubmedqa"]
                            out_file = p_out / f"{m['name']}.jsonl"
                        elif mode in ("baseline_idk", "rag_idk"):
                            p_out, q_subset = p12, questions
                            out_file = p_out / f"{m['name']}_{mode}.jsonl"
                        run_mode(llm, mode, q_subset, evidence, qcfg, out_file)
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
                            "status": status, "load_sources": load_sources, "settings": qcfg})
        ok = all(v == "DONE" for v in status.values())
        print(f"\nPHASE 4 FORMAT CHECK {'COMPLETE' if ok else 'FINISHED WITH PROBLEMS'}")
        if args.strict and not ok:
            raise SystemExit(1)
        return

    banner("RESULTS SO FAR (test set)")
    summary = {}
    print(f"{'model':<14}{'status':<10}{'baseline acc':<14}{'RAG acc':<10}{'context acc':<14}{'parse rate':<14}")
    for m in models:
        name = m["name"]
        b = summarise(p5 / f"{name}.jsonl") if (p5 / f"{name}.jsonl").exists() else {}
        r = summarise(p8 / f"{name}.jsonl") if (p8 / f"{name}.jsonl").exists() else {}
        c = summarise(p10 / f"{name}.jsonl") if (p10 / f"{name}.jsonl").exists() else {}
        b_idk = summarise(p12 / f"{name}_baseline_idk.jsonl") if (p12 / f"{name}_baseline_idk.jsonl").exists() else {}
        r_idk = summarise(p12 / f"{name}_rag_idk.jsonl") if (p12 / f"{name}_rag_idk.jsonl").exists() else {}

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
        if c:
            c_ans = [row for row in read_jsonl(p10 / f"{name}.jsonl") if row["split"] == "test" and not row["should_abstain"]]
            c_recomputed = round(sum(row["correct"] for row in c_ans) / len(c_ans), 4) if c_ans else None
            assert c.get("test_accuracy_all_answerable") == c_recomputed, (
                f"Context accuracy mismatch: {c.get('test_accuracy_all_answerable')} != {c_recomputed}"
            )

        summary[name] = {"status": status.get(name), "baseline": b, "rag": r, "context": c,
                         "baseline_idk": b_idk, "rag_idk": r_idk}
        b_acc = str(b.get('test_accuracy_all_answerable', '-'))
        r_acc = str(r.get('test_accuracy_all_answerable', '-'))
        c_acc = str(c.get('test_accuracy_pubmedqa', '-'))
        pr_parts = [str(x.get('parse_rate', '-')) for x in (b, r, c) if x]
        pr_str = "/".join(pr_parts) if pr_parts else "-"
        print(f"{name:<14}{('DONE' if status.get(name) == 'DONE' else 'PROBLEM'):<10}"
              f"{b_acc:<14}{r_acc:<10}{c_acc:<14}{pr_str:<14}")
        if status.get(name) != "DONE":
            print(f"    -> {status.get(name)}")
    if p8.exists():
        Path(p8 / "summary.json").write_text(json.dumps(summary, indent=2))
    if p10.exists():
        Path(p10 / "summary.json").write_text(json.dumps(summary, indent=2))
    if p12.exists():
        Path(p12 / "summary.json").write_text(json.dumps(summary, indent=2))
    batch_sizes_used = {
        m["name"]: {
            "baseline":     m.get("batch_size_baseline") or m.get("batch_size") or qcfg["batch_size"],
            "rag":          m.get("batch_size_rag")      or m.get("batch_size") or qcfg["batch_size"],
            "context":      m.get("batch_size_rag")      or m.get("batch_size") or qcfg["batch_size"],
            "baseline_idk": m.get("batch_size_baseline") or m.get("batch_size") or qcfg["batch_size"],
            "rag_idk":      m.get("batch_size_rag")      or m.get("batch_size") or qcfg["batch_size"],
        }
        for m in models
    }
    manifest_phases = [(p4, 4), (p5, 5), (p8, 8)]
    if "context" in run_modes or any((p10 / f"{m['name']}.jsonl").exists() for m in models):
        manifest_phases.append((p10, 10))
    if any(m in run_modes for m in ("baseline_idk", "rag_idk")) or any(
        (p12 / f"{m['name']}_{mode}.jsonl").exists() for m in models for mode in ("baseline_idk", "rag_idk")
    ):
        manifest_phases.append((p12, 12))
    for p, n in manifest_phases:
        write_manifest(p, {"phase": n, "tiny": args.tiny, "device": device, "status": status,
                            "load_sources": load_sources, "settings": qcfg,
                            "batch_sizes_used": batch_sizes_used, "modes": run_modes,
                            "split": args.split})
    ok = all(v == "DONE" for v in status.values())
    mode_desc = ", ".join(run_modes)
    print(f"\nQA PIPELINE ({mode_desc}) {'COMPLETE' if ok else 'FINISHED WITH PROBLEMS (see above)'}")
    if args.strict and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
