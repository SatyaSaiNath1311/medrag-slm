"""Small-LLM loading, prompts, batched generation, answer parsing.

Identical settings for all standard models: own chat template, thinking off, greedy decoding,
same prompts, same output format. Besides the text, we record the probability the model
gives each option letter at the answer position (needed for abstention in Phase 9).

For reasoning models (cfg["reasoning"] = true, e.g. medpsy-4b):
  - Chat template is applied with enable_thinking=True (the model may emit <think>…</think>).
  - The system instruction asks the model to "Think briefly, then end with a final line
    exactly 'Answer: <LETTER>'" (RAG modes also: "Evidence: [n] [n]").
  - After full generation we ALWAYS compute letter probabilities by re-feeding
    (prompt + generated text + "\\nAnswer:") and reading next-token logits for the
    valid option letters, renormalised over those letters only. This guarantees
    letter_probs is always populated for reasoning outputs.
  - If generation hit max_new_tokens without a final "Answer:" line we first close any
    open <think> block before the re-feed (truncated=True flag).
  - pred = parsed final letter if present in generated text, else argmax of letter_probs;
    pred_source is "parsed" / "logprob_refeed" / "none" accordingly.
  - Row schema is identical to non-reasoning rows:
    letter_probs, confidence, gen_tokens, truncated, seconds, prompt_tokens,
    raw_output, parsed, pred, pred_source, citations (for rag/context).
"""
import re
import time

PROMPT_HEAD = "You are a medical expert answering a multiple-choice question."
FORMAT_BASELINE = "Answer with the letter of the correct option."
FORMAT_RAG = ("Answer with the letter of the correct option. Then, on a new line, write \"Evidence:\" followed by "
              "the numbers of the passages that support your answer in square brackets.")
# The assistant reply is pre-filled with this text, so every model continues directly with the letter.
# Same for all models; it removes formatting differences and fixes the position of the answer.
ANSWER_PREFIX = "Answer:"

# Reasoning-specific instructions (appended to the normal question block)
REASONING_INSTRUCTION = (
    "Think briefly about the question, then end your response with a final line containing "
    "exactly 'Answer: <LETTER>' (one letter, capital, no other text on that line)."
)
REASONING_RAG_INSTRUCTION = (
    "Think briefly about the question, then end your response with two final lines: "
    "'Answer: <LETTER>' and 'Evidence: [n] [n]' (passage numbers in square brackets)."
)

# Constant used across demo, chatbot, and pipeline when the model abstains or confidence is low
ABSTAIN_MESSAGE = "I don't have enough information to answer this confidently. Please consult a doctor."
IDK_TEXT = "I do not have enough information to answer this question."


def _options_block(q):
    return "\n".join(f"{k}. {v}" for k, v in q["options"].items())


def get_idk_letter(q):
    """Next option letter: D for PubMedQA (3 options), E for MedQA and unanswerable."""
    if q.get("dataset") == "pubmedqa":
        return "D"
    return "E"


def get_idk_options(q):
    """Return options dictionary augmented with the IDK option."""
    opts = dict(q["options"])
    opts[get_idk_letter(q)] = IDK_TEXT
    return opts


# ── Standard (non-reasoning) prompts ──────────────────────────────────────────

def baseline_prompt(q):
    return f"{PROMPT_HEAD}\n\nQuestion: {q['question']}\nOptions:\n{_options_block(q)}\n\n{FORMAT_BASELINE}"


def rag_prompt(q, passages, max_chars):
    ev = "\n".join(f"[{i}] ({p['title']}) {p['text'][:max_chars]}" for i, p in enumerate(passages, 1))
    return (f"{PROMPT_HEAD} Use the numbered evidence passages.\n\nEvidence:\n{ev}\n\n"
            f"Question: {q['question']}\nOptions:\n{_options_block(q)}\n\n{FORMAT_RAG}")


def baseline_idk_prompt(q):
    letter = get_idk_letter(q)
    opts = get_idk_options(q)
    opts_block = "\n".join(f"{k}. {v}" for k, v in opts.items())
    instruction = f"If none of the options is correct or you do not have enough information to answer, choose {letter}."
    return f"{PROMPT_HEAD}\n\nQuestion: {q['question']}\nOptions:\n{opts_block}\n\n{FORMAT_BASELINE} {instruction}"


def rag_idk_prompt(q, passages, max_chars):
    letter = get_idk_letter(q)
    opts = get_idk_options(q)
    opts_block = "\n".join(f"{k}. {v}" for k, v in opts.items())
    ev = "\n".join(f"[{i}] ({p['title']}) {p['text'][:max_chars]}" for i, p in enumerate(passages, 1))
    instruction = f"If none of the options is correct or you do not have enough information to answer, choose {letter}."
    return (f"{PROMPT_HEAD} Use the numbered evidence passages.\n\nEvidence:\n{ev}\n\n"
            f"Question: {q['question']}\nOptions:\n{opts_block}\n\n{FORMAT_RAG} {instruction}")


def context_prompt(q, max_chars=None):
    """PubMedQA standard context prompt: single evidence passage containing the abstract."""
    abstract = q.get("context") or ""
    if max_chars is not None and len(abstract) > max_chars:
        abstract = abstract[:max_chars]
    ev = f"[1] (Abstract) {abstract}"
    return (f"{PROMPT_HEAD} Use the numbered evidence passages.\n\nEvidence:\n{ev}\n\n"
            f"Question: {q['question']}\nOptions:\n{_options_block(q)}\n\n{FORMAT_RAG}")


# ── Reasoning-model prompts ────────────────────────────────────────────────────

def reasoning_baseline_prompt(q):
    """Reasoning prompt for baseline mode (no evidence)."""
    return (f"{PROMPT_HEAD}\n\nQuestion: {q['question']}\nOptions:\n{_options_block(q)}\n\n"
            f"{FORMAT_BASELINE}\n{REASONING_INSTRUCTION}")


def reasoning_rag_prompt(q, passages, max_chars):
    """Reasoning prompt for RAG mode (numbered evidence passages)."""
    ev = "\n".join(f"[{i}] ({p['title']}) {p['text'][:max_chars]}" for i, p in enumerate(passages, 1))
    return (f"{PROMPT_HEAD} Use the numbered evidence passages.\n\nEvidence:\n{ev}\n\n"
            f"Question: {q['question']}\nOptions:\n{_options_block(q)}\n\n"
            f"{FORMAT_RAG}\n{REASONING_RAG_INSTRUCTION}")


def reasoning_context_prompt(q, max_chars=None):
    """Reasoning prompt for context mode (study abstract, PubMedQA only)."""
    abstract = q.get("context") or ""
    if max_chars is not None and len(abstract) > max_chars:
        abstract = abstract[:max_chars]
    ev = f"[1] (Abstract) {abstract}"
    return (f"{PROMPT_HEAD} Use the numbered evidence passages.\n\nEvidence:\n{ev}\n\n"
            f"Question: {q['question']}\nOptions:\n{_options_block(q)}\n\n"
            f"{FORMAT_RAG}\n{REASONING_RAG_INSTRUCTION}")


# ── Think-block helpers ────────────────────────────────────────────────────────

_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def close_open_think_block(text: str) -> str:
    """If the text has an unclosed <think> block, close it.

    Used when a reasoning model hits max_new_tokens without producing a final
    'Answer:' line — we close the think block before the re-feed so the model
    sees a clean context ending with </think> and can produce the answer letter.
    """
    opens = len(_THINK_OPEN_RE.findall(text))
    closes = len(_THINK_CLOSE_RE.findall(text))
    if opens > closes:
        return text + "\n</think>"
    return text


def extract_final_answer_from_reasoning(text: str, options) -> str | None:
    """Parse the last 'Answer: X' line from a reasoning model's full output.

    We specifically look at the *last* occurrence because a reasoning model may
    discuss options ("If the answer is A…") in its think block.
    """
    # Find all "Answer: X" occurrences, take the last one
    matches = list(re.finditer(r"answer\W*([A-Za-z])\b", text, re.IGNORECASE))
    for m in reversed(matches):
        letter = m.group(1).upper()
        if letter in options:
            return letter
    return None


# ── Standard parsing (shared) ──────────────────────────────────────────────────

def parse_answer(text, options, idk_letter=None):
    t = re.sub(r"</?\\s*(?:letter|answer)\\s*>", " ", text, flags=re.I).strip()
    m = re.search(r"answer(?:\s+is)?\W*([A-Za-z])\b", t, re.I)
    if m and m.group(1).upper() in options:
        res = m.group(1).upper()
        if (idk_letter and res == idk_letter) or options.get(res) == IDK_TEXT:
            return "ABSTAIN"
        return res
    for letter, value in options.items():  # e.g. "Answer: yes" for PubMedQA
        if re.search(r"answer\W*" + re.escape(value.lower()) + r"\b", t.lower()):
            if (idk_letter and letter == idk_letter) or value == IDK_TEXT:
                return "ABSTAIN"
            return letter
    m = re.match(r"^\W*([A-Za-z])(?:[.):]|\s*$|\n)", t)  # bare "B", "B.", "(B)"
    if m and m.group(1).upper() in options:
        res = m.group(1).upper()
        if (idk_letter and res == idk_letter) or options.get(res) == IDK_TEXT:
            return "ABSTAIN"
        return res
    if IDK_TEXT.lower() in t.lower():
        return "ABSTAIN"
    return None


def parse_citations(text, n_passages):
    if not text:
        return []
    nums = set()
    for b in re.findall(r"\[([^\]]+)\]", text):
        for n in re.findall(r"\b\d+\b", b):
            nums.add(int(n))
    if not nums:
        m = re.search(r"evidence\s*:(.*)", text, re.I | re.DOTALL)
        if m:
            first_line = m.group(1).strip().split("\n")[0]
            for n in re.findall(r"\b\d+\b", first_line):
                nums.add(int(n))
    return sorted(n for n in nums if 1 <= n <= n_passages)


# ── LLM class ─────────────────────────────────────────────────────────────────

class LLM:
    def __init__(self, mcfg, device):
        import glob
        import os
        import torch
        import transformers
        from packaging import version
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.cfg, self.device = torch, mcfg, device
        self.is_reasoning = bool(mcfg.get("reasoning", False))

        load_path = None
        load_source = None

        if mcfg.get("kaggle_path_glob"):
            # a. the glob
            pattern = mcfg["kaggle_path_glob"]
            print(f"  attempting glob: {pattern}", flush=True)
            matches = [p for p in glob.glob(pattern)
                       if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json"))]
            if matches:
                load_path = matches[0]
                load_source = f"kaggle-models {load_path}"
                print(f"    found via glob: {load_path}", flush=True)
            else:
                print("    glob did not match any directory containing config.json", flush=True)

            # b. a recursive search of /kaggle/input for a directory with config.json whose path contains "gemma-3-4b-it" (case-insensitive)
            if load_path is None and os.path.exists("/kaggle/input"):
                print("  attempting recursive search in /kaggle/input for 'gemma-3-4b-it' with config.json...", flush=True)
                for root, dirs, files in os.walk("/kaggle/input"):
                    if "gemma-3-4b-it" in root.lower() and "config.json" in files:
                        load_path = root
                        load_source = f"kaggle-models {load_path}"
                        print(f"    found via recursive search: {load_path}", flush=True)
                        break
                if load_path is None:
                    print("    recursive search found no matching directory with config.json", flush=True)

            # c. if running on Kaggle (/kaggle/input exists): path = kagglehub.model_download(...)
            if load_path is None and os.path.exists("/kaggle/input"):
                handle = mcfg.get("kaggle_handle", "google/gemma-3/transformers/gemma-3-4b-it")
                print(f"  attempting kagglehub.model_download({handle!r})...", flush=True)
                try:
                    import kagglehub
                    hub_dir = kagglehub.model_download(handle)
                    if hub_dir and os.path.isdir(hub_dir):
                        if os.path.exists(os.path.join(hub_dir, "config.json")):
                            load_path = hub_dir
                        else:
                            for root, dirs, files in os.walk(hub_dir):
                                if "config.json" in files:
                                    load_path = root
                                    break
                    if load_path:
                        load_source = f"kagglehub {load_path}"
                        print(f"    loaded via kagglehub: {load_path}", flush=True)
                    else:
                        print(f"    kagglehub downloaded to {hub_dir} but no config.json found", flush=True)
                except Exception as e:
                    print(f"    kagglehub.model_download failed: {e}", flush=True)

            # d. otherwise HF with token; if no HF_TOKEN either, fail immediately
            if load_path is None:
                hf_token = os.environ.get("HF_TOKEN")
                print("  attempting HF fallback...", flush=True)
                if not hf_token:
                    raise RuntimeError("Gemma not available from Kaggle Models (mount/kagglehub) and no HF token")
                load_path = mcfg["id"]
                load_source = f"hf {mcfg['id']}"
        else:
            load_path = mcfg["id"]
            load_source = f"hf {mcfg['id']}"

        print(f"  source: {load_source}", flush=True)
        self.load_source = load_source

        self.tok = AutoTokenizer.from_pretrained(load_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if mcfg.get("loader") == "gemma3":
            from transformers import Gemma3ForConditionalGeneration as cls
        else:
            cls = AutoModelForCausalLM
        dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
        kwargs = {key: dtypes[mcfg["dtype"]]}
        if device == "cuda":
            many = torch.cuda.device_count() > 1 and mcfg["dtype"] == "float32"
            kwargs["device_map"] = "auto" if many else {"": 0}
        elif isinstance(device, str) and device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[-1])
                kwargs["device_map"] = {"": gpu_id}
            except ValueError:
                kwargs["device_map"] = {"": device}
        elif isinstance(device, int):
            kwargs["device_map"] = {"": device}
        self.model = cls.from_pretrained(load_path, **kwargs)
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.eos = {e for e in eos + [self.tok.eos_token_id] if e is not None}
        self.pad_id = self.tok.pad_token_id
        self._letter_ids = {}

    # ── Chat-template helpers ──────────────────────────────────────────────────

    def wrap(self, user_text):
        """Standard (non-reasoning) wrap: thinking disabled, answer prefix pre-filled."""
        msgs = [{"role": "user", "content": user_text}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        return text + ANSWER_PREFIX

    def wrap_reasoning(self, user_text):
        """Reasoning wrap: thinking enabled, no answer prefix (model generates freely)."""
        msgs = [{"role": "user", "content": user_text}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=True)
        except TypeError:
            # Fallback: tokenizer does not support enable_thinking kwarg
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text

    # ── Letter-probability helpers ─────────────────────────────────────────────

    def letter_ids(self, letter):
        if letter not in self._letter_ids:
            ids = set()
            for s in (letter, " " + letter):
                enc = self.tok.encode(s, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
            self._letter_ids[letter] = sorted(ids)
        return self._letter_ids[letter]

    def _answer_step(self, gen_ids, letters):
        """Index of the generation step that holds the answer.
        Preferred: the step right after "Answer:" / "answer is" where a letter was written.
        Fallback: the first such step (e.g. the model wrote "yes" instead of "A")."""
        first = None
        for i in range(len(gen_ids)):
            prev = ANSWER_PREFIX + self.tok.decode(gen_ids[:i], skip_special_tokens=True)
            if not re.search(r"answer(?:\s+is)?\W*$", prev, re.I):
                continue
            new = (ANSWER_PREFIX + self.tok.decode(gen_ids[:i + 1], skip_special_tokens=True))[len(prev):].strip()
            if new.lower() == "is":  # "answer" followed by "is": the answer comes next
                continue
            if first is None:
                first = i
            if new and new[0].upper() in letters and (len(new) == 1 or not new[1].isalpha()):
                return i, "letter"
        return (first, "position") if first is not None else (None, None)

    def _letter_probs(self, gen_ids, step_logits, letters):
        """Probability of each option letter at the answer step, renormalised over the options."""
        i, how = self._answer_step(gen_ids, letters)
        if i is None:
            return None, None
        probs = self.torch.softmax(step_logits[i].float(), dim=-1)
        raw = {L: float(sum(probs[t] for t in self.letter_ids(L))) for L in letters}
        total = sum(raw.values())
        if total <= 0:
            return None, None
        return {L: round(v / total, 6) for L, v in raw.items()}, how

    def _letter_probs_refeed(self, prompt_wrapped: str, generated_text: str, letters):
        """Compute option-letter probabilities by re-feeding the full context.

        Constructs: prompt_wrapped + generated_text + "\\nAnswer:" and reads the
        next-token logits for each valid option letter, renormalised over those letters.
        This is used for reasoning models where the answer may appear anywhere in the
        generated text, and also as a fallback when the answer step is not found.

        Returns (letter_probs_dict, "refeed") or (None, None) on failure.
        """
        torch = self.torch
        refeed_text = prompt_wrapped + generated_text + "\nAnswer:"
        try:
            enc = self.tok(refeed_text, return_tensors="pt", add_special_tokens=False)
            first_dev = next(self.model.parameters()).device
            enc = {k: v.to(first_dev) for k, v in enc.items()}
            with torch.inference_mode():
                out = self.model(**enc)
            logits = out.logits[0, -1, :]  # last position
            probs = torch.softmax(logits.float(), dim=-1)
            raw = {L: float(sum(probs[t] for t in self.letter_ids(L))) for L in letters}
            total = sum(raw.values())
            if total <= 0:
                return None, None
            return {L: round(v / total, 6) for L, v in raw.items()}, "refeed"
        except Exception:  # noqa: BLE001
            return None, None

    # ── Standard generation (non-reasoning) ────────────────────────────────────

    def generate(self, prompts, letters_list, max_new_tokens):
        """Greedy generation for a batch. Returns one dict per prompt."""
        torch = self.torch
        from transformers import GenerationConfig
        texts = [self.wrap(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        first = next(self.model.parameters()).device
        enc = {k: v.to(first) for k, v in enc.items()}
        gcfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=sorted(self.eos),
                                pad_token_id=self.pad_id, return_dict_in_generate=True, output_scores=True)
        if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**enc, generation_config=gcfg)
        if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        L = enc["input_ids"].shape[1]
        results = []
        for b in range(len(prompts)):
            ids = out.sequences[b, L:].tolist()
            cut = next((i for i, t in enumerate(ids) if t in self.eos), len(ids))
            ids = ids[:cut]
            step_logits = [s[b] for s in out.scores[:len(ids)]]
            lp, how = self._letter_probs(ids, step_logits, letters_list[b])
            results.append({
                "raw_output": (ANSWER_PREFIX + self.tok.decode(ids, skip_special_tokens=True)).strip(),
                "letter_probs": lp,
                "prob_source": how,
                "prompt_tokens": int(enc["attention_mask"][b].sum()),
                "gen_tokens": len(ids),
                "seconds": round(seconds / len(prompts), 4),
            })
        return results

    def generate_safe(self, prompts, letters_list, max_new_tokens):
        """Like generate, but halves the batch on GPU out-of-memory instead of crashing.

        Catches both torch.cuda.OutOfMemoryError and RuntimeError containing
        "out of memory" (older PyTorch / CUDA versions surface it as the latter).
        Halves the batch and retries recursively down to batch size 1; re-raises
        only if a single-prompt call still fails.
        """
        def _is_oom(exc):
            if isinstance(exc, self.torch.cuda.OutOfMemoryError):
                return True
            if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
                return True
            return False

        try:
            return self.generate(prompts, letters_list, max_new_tokens)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            self.torch.cuda.empty_cache()
            if len(prompts) == 1:
                print(f"    OOM on single prompt – re-raising", flush=True)
                raise
            h = len(prompts) // 2
            print(f"    OOM at batch {len(prompts)}, retrying as {h} + {len(prompts) - h}", flush=True)
            return (self.generate_safe(prompts[:h], letters_list[:h], max_new_tokens)
                    + self.generate_safe(prompts[h:], letters_list[h:], max_new_tokens))

    # ── Reasoning generation ───────────────────────────────────────────────────

    def generate_reasoning(self, prompts, letters_list, max_new_tokens):
        """Greedy generation for reasoning models (one item at a time for correctness).

        Processing order (longest prompt first) is handled by the caller.
        For each prompt:
          1. Apply chat template with enable_thinking=True (no ANSWER_PREFIX pre-fill).
          2. Generate greedily up to max_new_tokens.
          3. Detect truncation (hit cap without a final "Answer:" line).
          4. If truncated, close any open <think> block.
          5. Compute letter probabilities via _letter_probs_refeed.
          6. Parse the final "Answer: X" from generated text (last occurrence).
          7. Build and return the result dict with the standard schema.

        We process one item at a time (batch=1) to avoid cross-contamination of
        the think-block-close refeed and to keep memory predictable.
        Left-padding is set on the tokenizer already.
        """
        torch = self.torch
        from transformers import GenerationConfig
        results = []
        t0 = time.perf_counter()
        first_dev = next(self.model.parameters()).device

        for prompt, letters in zip(prompts, letters_list):
            wrapped = self.wrap_reasoning(prompt)
            enc = self.tok([wrapped], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(first_dev) for k, v in enc.items()}
            prompt_len = int(enc["attention_mask"][0].sum())

            gcfg = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=sorted(self.eos),
                pad_token_id=self.pad_id,
                return_dict_in_generate=True,   # ask for ModelOutput; some models return Tensor anyway
            )
            if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
                torch.cuda.synchronize()
            with torch.inference_mode():
                out = self.model.generate(**enc, generation_config=gcfg)
            if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
                torch.cuda.synchronize()

            # Normalise: some models return a plain Tensor even when return_dict_in_generate=True
            seqs = out.sequences if hasattr(out, "sequences") else out
            prompt_ids_len = enc["input_ids"].shape[1]
            all_gen_ids = seqs[0, prompt_ids_len:].tolist()
            cut = next((i for i, t in enumerate(all_gen_ids) if t in self.eos), len(all_gen_ids))
            gen_ids = all_gen_ids[:cut]
            gen_text = self.tok.decode(gen_ids, skip_special_tokens=True)

            # Detect whether we ended without a final "Answer:" line
            # truncated = hit the cap (cut == full generated length) and no final answer found
            has_final_answer = bool(re.search(r"(?:^|\n)Answer\s*:\s*[A-Za-z]", gen_text, re.MULTILINE))
            truncated = (cut == len(all_gen_ids) and not has_final_answer)

            # For re-feed: if truncated close any open think block
            refeed_generated = close_open_think_block(gen_text) if truncated else gen_text

            # Always compute letter probs via refeed
            lp, lp_source = self._letter_probs_refeed(wrapped, refeed_generated, letters)

            # Parse the final answer letter from generated text
            parsed_letter = extract_final_answer_from_reasoning(gen_text, letters)

            if parsed_letter is not None:
                pred = parsed_letter
                pred_source = "parsed"
                parsed = True
            elif lp is not None:
                pred = max(lp, key=lp.get)
                pred_source = "logprob_refeed"
                parsed = False
            else:
                pred = None
                pred_source = "none"
                parsed = False

            results.append({
                "raw_output": gen_text,
                "letter_probs": lp,
                "prob_source": lp_source,
                "prompt_tokens": prompt_len,
                "gen_tokens": len(gen_ids),
                "truncated": truncated,
                "seconds": round((time.perf_counter() - t0) / max(len(prompts), 1), 4),
                # Convenience fields used by run_mode
                "_pred": pred,
                "_pred_source": pred_source,
                "_parsed": parsed,
            })

        # Fix per-question seconds (total elapsed / count, done after loop)
        total_sec = time.perf_counter() - t0
        n = len(results)
        for r in results:
            r["seconds"] = round(total_sec / n, 4)

        return results

    def generate_reasoning_safe(self, prompts, letters_list, max_new_tokens):
        """OOM-safe wrapper around generate_reasoning (same halving strategy)."""
        def _is_oom(exc):
            if isinstance(exc, self.torch.cuda.OutOfMemoryError):
                return True
            if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
                return True
            return False

        try:
            return self.generate_reasoning(prompts, letters_list, max_new_tokens)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            self.torch.cuda.empty_cache()
            if len(prompts) == 1:
                print(f"    OOM on single reasoning prompt – re-raising", flush=True)
                raise
            h = len(prompts) // 2
            print(f"    OOM at reasoning batch {len(prompts)}, retrying as {h} + {len(prompts) - h}", flush=True)
            return (self.generate_reasoning_safe(prompts[:h], letters_list[:h], max_new_tokens)
                    + self.generate_reasoning_safe(prompts[h:], letters_list[h:], max_new_tokens))

    def close(self):
        import gc
        del self.model
        gc.collect()
        if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
            self.torch.cuda.empty_cache()
