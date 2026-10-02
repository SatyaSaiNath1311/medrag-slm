"""Small-LLM loading, prompts, batched generation, answer parsing.

Identical settings for all 5 models: own chat template, thinking off, greedy decoding,
same prompts, same output format. Besides the text, we record the probability the model
gives each option letter at the answer position (needed for abstention in Phase 9).
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


# ---------- parsing ----------
def parse_answer(text, options, idk_letter=None):
    t = re.sub(r"</?\s*(?:letter|answer)\s*>", " ", text, flags=re.I).strip()
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


# ---------- model ----------
class LLM:
    def __init__(self, mcfg, device):
        import glob
        import os
        import torch
        import transformers
        from packaging import version
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.cfg, self.device = torch, mcfg, device

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
        self.model = cls.from_pretrained(load_path, **kwargs)
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.eos = {e for e in eos + [self.tok.eos_token_id] if e is not None}
        self.pad_id = self.tok.pad_token_id
        self._letter_ids = {}

    def wrap(self, user_text):
        msgs = [{"role": "user", "content": user_text}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        return text + ANSWER_PREFIX

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
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**enc, generation_config=gcfg)
        if self.device == "cuda":
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

    def close(self):
        import gc
        del self.model
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
