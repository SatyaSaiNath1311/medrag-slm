"""Unit tests for medpsy-4b reasoning model integration.

Tests cover (no GPU / no model loading required):
  1. reasoning_baseline_prompt     — correct structure, letters, instruction
  2. reasoning_rag_prompt          — evidence block present, reasoning instruction
  3. reasoning_context_prompt      — abstract block, PubMedQA structure
  4. close_open_think_block        — closes unclosed <think>; leaves balanced text unchanged
  5. extract_final_answer_from_reasoning — last "Answer: X" parsing, think-block noise ignored
  6. sample_val_subset             — 60 MedQA + 20 PubMedQA, seeded, no overlap with non-members
  7. val_subset in PILOT config    — pilot selects medpsy-4b, baseline + context modes
"""

import random
import unittest

from src.llm import (
    close_open_think_block,
    extract_final_answer_from_reasoning,
    reasoning_baseline_prompt,
    reasoning_context_prompt,
    reasoning_rag_prompt,
    REASONING_INSTRUCTION,
    REASONING_RAG_INSTRUCTION,
    PROMPT_HEAD,
)
from src.qa_pipeline import sample_val_subset


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_medqa_q(idx=1, answer="B"):
    return {
        "id": f"medqa-val-{idx:04d}",
        "split": "val",
        "dataset": "medqa",
        "question": f"What is the primary treatment for condition {idx}?",
        "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
        "answer": answer,
        "should_abstain": False,
    }


def _make_pubmedqa_q(idx=1, answer="A"):
    return {
        "id": f"pubmedqa-val-{idx:04d}",
        "split": "val",
        "dataset": "pubmedqa",
        "question": f"Does treatment {idx} improve outcomes?",
        "options": {"A": "yes", "B": "no", "C": "maybe"},
        "answer": answer,
        "should_abstain": False,
        "context": f"Abstract text about treatment {idx}.",
    }


def _make_passage(i, title=None):
    return {"rank": i, "title": title or f"Source {i}", "text": f"Evidence snippet number {i}."}


# ── Test class ─────────────────────────────────────────────────────────────────

class TestReasoningPrompts(unittest.TestCase):

    def setUp(self):
        self.medqa_q = _make_medqa_q()
        self.pubmedqa_q = _make_pubmedqa_q()
        self.passages = [_make_passage(i) for i in range(1, 4)]

    # ── 1. reasoning_baseline_prompt ──────────────────────────────────────────

    def test_baseline_prompt_contains_question(self):
        prompt = reasoning_baseline_prompt(self.medqa_q)
        self.assertIn(self.medqa_q["question"], prompt)

    def test_baseline_prompt_contains_all_options(self):
        prompt = reasoning_baseline_prompt(self.medqa_q)
        for letter, text in self.medqa_q["options"].items():
            self.assertIn(f"{letter}. {text}", prompt)

    def test_baseline_prompt_has_reasoning_instruction(self):
        prompt = reasoning_baseline_prompt(self.medqa_q)
        self.assertIn(REASONING_INSTRUCTION, prompt)

    def test_baseline_prompt_starts_with_prompt_head(self):
        prompt = reasoning_baseline_prompt(self.medqa_q)
        self.assertTrue(prompt.startswith(PROMPT_HEAD))

    def test_baseline_prompt_no_evidence_block(self):
        prompt = reasoning_baseline_prompt(self.medqa_q)
        self.assertNotIn("Evidence:", prompt)

    # ── 2. reasoning_rag_prompt ───────────────────────────────────────────────

    def test_rag_prompt_contains_numbered_evidence(self):
        prompt = reasoning_rag_prompt(self.medqa_q, self.passages, max_chars=200)
        for i, p in enumerate(self.passages, 1):
            self.assertIn(f"[{i}]", prompt)
            self.assertIn(p["title"], prompt)

    def test_rag_prompt_has_rag_instruction(self):
        prompt = reasoning_rag_prompt(self.medqa_q, self.passages, max_chars=200)
        self.assertIn(REASONING_RAG_INSTRUCTION, prompt)

    def test_rag_prompt_truncates_passage_text(self):
        long_passage = [{"rank": 1, "title": "T", "text": "x" * 500}]
        prompt = reasoning_rag_prompt(self.medqa_q, long_passage, max_chars=100)
        self.assertIn("x" * 100, prompt)
        self.assertNotIn("x" * 101, prompt)

    def test_rag_prompt_contains_question(self):
        prompt = reasoning_rag_prompt(self.medqa_q, self.passages, max_chars=200)
        self.assertIn(self.medqa_q["question"], prompt)

    # ── 3. reasoning_context_prompt ───────────────────────────────────────────

    def test_context_prompt_contains_abstract(self):
        prompt = reasoning_context_prompt(self.pubmedqa_q)
        self.assertIn(self.pubmedqa_q["context"], prompt)

    def test_context_prompt_labels_passage_as_abstract(self):
        prompt = reasoning_context_prompt(self.pubmedqa_q)
        self.assertIn("[1] (Abstract)", prompt)

    def test_context_prompt_has_rag_instruction(self):
        prompt = reasoning_context_prompt(self.pubmedqa_q)
        self.assertIn(REASONING_RAG_INSTRUCTION, prompt)

    def test_context_prompt_truncates_abstract(self):
        long_q = dict(self.pubmedqa_q)
        long_q["context"] = "A" * 600
        prompt = reasoning_context_prompt(long_q, max_chars=200)
        self.assertIn("A" * 200, prompt)
        self.assertNotIn("A" * 201, prompt)

    def test_context_prompt_no_abstract_still_works(self):
        q = dict(self.pubmedqa_q)
        del q["context"]
        prompt = reasoning_context_prompt(q)
        self.assertIn("[1] (Abstract)", prompt)

    # ── 4. close_open_think_block ─────────────────────────────────────────────

    def test_closes_unclosed_think(self):
        text = "<think>\nI am thinking..."
        result = close_open_think_block(text)
        self.assertTrue(result.endswith("</think>"))
        self.assertIn("<think>", result)

    def test_does_not_close_balanced_think(self):
        text = "<think>\nThinking here.\n</think>\nSome conclusion."
        result = close_open_think_block(text)
        self.assertEqual(result, text)

    def test_empty_string_unchanged(self):
        self.assertEqual(close_open_think_block(""), "")

    def test_no_think_block_unchanged(self):
        text = "Answer: B"
        self.assertEqual(close_open_think_block(text), text)

    def test_multiple_balanced_blocks_unchanged(self):
        text = "<think>first</think><think>second</think>Done."
        self.assertEqual(close_open_think_block(text), text)

    def test_one_unclosed_among_balanced(self):
        text = "<think>closed</think><think>unclosed"
        result = close_open_think_block(text)
        self.assertTrue(result.endswith("</think>"))

    # ── 5. extract_final_answer_from_reasoning ────────────────────────────────

    def test_extracts_last_answer_letter(self):
        text = "<think>If answer is A, then...</think>\nAnswer: B"
        options = {"A": "opt A", "B": "opt B", "C": "opt C", "D": "opt D"}
        self.assertEqual(extract_final_answer_from_reasoning(text, options), "B")

    def test_returns_none_when_no_valid_letter(self):
        text = "<think>thinking</think>\nConclusion: none of the above."
        options = {"A": "opt A", "B": "opt B"}
        self.assertIsNone(extract_final_answer_from_reasoning(text, options))

    def test_picks_last_occurrence_not_first(self):
        text = "I think Answer: A, but actually Answer: C is correct."
        options = {"A": "opt A", "B": "opt B", "C": "opt C", "D": "opt D"}
        self.assertEqual(extract_final_answer_from_reasoning(text, options), "C")

    def test_only_valid_option_letter_accepted(self):
        text = "Answer: Z"
        options = {"A": "opt A", "B": "opt B"}
        self.assertIsNone(extract_final_answer_from_reasoning(text, options))

    def test_case_insensitive_answer_prefix(self):
        text = "answer: d"
        options = {"A": "opt A", "B": "opt B", "C": "opt C", "D": "opt D"}
        self.assertEqual(extract_final_answer_from_reasoning(text, options), "D")

    def test_pubmedqa_options(self):
        text = "The evidence suggests yes.\nAnswer: A"
        options = {"A": "yes", "B": "no", "C": "maybe"}
        self.assertEqual(extract_final_answer_from_reasoning(text, options), "A")

    # ── 6. sample_val_subset ──────────────────────────────────────────────────

    def _make_val_pool(self, n_medqa=100, n_pubmedqa=100):
        pool = []
        for i in range(1, n_medqa + 1):
            pool.append(_make_medqa_q(idx=i))
        for i in range(1, n_pubmedqa + 1):
            pool.append(_make_pubmedqa_q(idx=i))
        return pool

    def test_sample_val_subset_sizes(self):
        pool = self._make_val_pool(100, 100)
        subset = sample_val_subset(pool, seed=42)
        medqa = [r for r in subset if r["dataset"] == "medqa"]
        pubmedqa = [r for r in subset if r["dataset"] == "pubmedqa"]
        self.assertEqual(len(medqa), 60)
        self.assertEqual(len(pubmedqa), 20)
        self.assertEqual(len(subset), 80)

    def test_sample_val_subset_seed_reproducible(self):
        pool = self._make_val_pool(100, 100)
        s1 = sample_val_subset(pool, seed=42)
        s2 = sample_val_subset(pool, seed=42)
        self.assertEqual([r["id"] for r in s1], [r["id"] for r in s2])

    def test_sample_val_subset_different_seeds_differ(self):
        pool = self._make_val_pool(100, 100)
        s1 = sample_val_subset(pool, seed=42)
        s2 = sample_val_subset(pool, seed=99)
        self.assertNotEqual([r["id"] for r in s1], [r["id"] for r in s2])

    def test_sample_val_subset_no_duplicates(self):
        pool = self._make_val_pool(100, 100)
        subset = sample_val_subset(pool, seed=42)
        ids = [r["id"] for r in subset]
        self.assertEqual(len(ids), len(set(ids)))

    def test_sample_val_subset_caps_at_available(self):
        pool = self._make_val_pool(n_medqa=30, n_pubmedqa=10)
        subset = sample_val_subset(pool, seed=42)
        medqa = [r for r in subset if r["dataset"] == "medqa"]
        pubmedqa = [r for r in subset if r["dataset"] == "pubmedqa"]
        self.assertLessEqual(len(medqa), 60)
        self.assertLessEqual(len(pubmedqa), 20)

    def test_sample_val_subset_all_from_val(self):
        pool = self._make_val_pool(100, 100)
        pool_ids = {r["id"] for r in pool}
        subset = sample_val_subset(pool, seed=42)
        for r in subset:
            self.assertIn(r["id"], pool_ids)

    def test_sample_val_subset_only_non_abstain(self):
        pool = self._make_val_pool(80, 80)
        # Add some abstain rows
        for i in range(10):
            pool.append({
                "id": f"unans-val-{i:04d}",
                "split": "val",
                "dataset": "medqa",
                "question": "Abstain question",
                "options": {"A": "X", "B": "Y", "C": "Z"},
                "answer": None,
                "should_abstain": True,
            })
        subset = sample_val_subset(pool, seed=42)
        for r in subset:
            self.assertFalse(r.get("should_abstain", False),
                             msg=f"should_abstain=True row found in subset: {r['id']}")

    # ── 7. PILOT config constants ──────────────────────────────────────────────

    def test_pilot_model_is_medpsy(self):
        """Verify that the PILOT constant in runner refers to medpsy-4b."""
        # We just verify the model name string is present in runner.py
        import pathlib
        runner_text = pathlib.Path("kaggle_runner/runner.py").read_text()
        self.assertIn("medpsy-4b", runner_text)
        self.assertIn("PILOT", runner_text)

    def test_pilot_modes_are_baseline_and_context(self):
        import pathlib
        runner_text = pathlib.Path("kaggle_runner/runner.py").read_text()
        self.assertIn('"baseline", "context"', runner_text)
        self.assertIn("val_subset", runner_text)

    def test_pilot_uses_single_gpu(self):
        import pathlib
        runner_text = pathlib.Path("kaggle_runner/runner.py").read_text()
        self.assertIn("CUDA_VISIBLE_DEVICES", runner_text)
        self.assertIn('"0"', runner_text)

    def test_pilot_passes_skip_format_check(self):
        """PILOT build_qa_cmd must include --skip-format-check."""
        import pathlib
        runner_text = pathlib.Path("kaggle_runner/runner.py").read_text()
        self.assertIn("--skip-format-check", runner_text)
        self.assertIn("skip_format_check=True", runner_text)


# ── Mock-model tests for generate_reasoning output normalisation ───────────────

class _FakeTokenizer:
    """Minimal tokenizer stub sufficient for generate_reasoning."""

    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.padding_side = "left"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 1
        # fixed encoding map used by letter_ids
        self._enc = {
            "A": [65], " A": [265],
            "B": [66], " B": [266],
            "C": [67], " C": [267],
            "D": [68], " D": [268],
        }

    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=None):
        import torch
        # Always return a fixed 5-token prompt so we can predict offset
        ids = torch.zeros(1, 5, dtype=torch.long)
        mask = torch.ones(1, 5, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        return messages[0]["content"] + "\n"

    def decode(self, ids, skip_special_tokens=True):
        # ids [66] → "B", ids [67] → "C", etc.
        mapping = {65: "A", 66: "B", 67: "C", 68: "D"}
        return "".join(mapping.get(i, "") for i in ids)

    def encode(self, text, add_special_tokens=False):
        return self._enc.get(text, [999])


class _FakeModel:
    """Minimal model stub.  generate_mode controls the return type."""

    def __init__(self, gen_ids, mode="plain_tensor"):
        """
        gen_ids  — token ids to 'generate' (list[int]), appended after the 5-prompt tokens.
        mode     — "plain_tensor"  : return a plain Tensor (shape [1, prompt+gen])
                   "model_output"  : return an object with .sequences attribute
        """
        import torch
        self._gen_ids = gen_ids
        self._mode = mode
        self._param = torch.zeros(1, requires_grad=False)

    def parameters(self):
        yield self._param

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, generation_config=None):
        import torch
        prompt_len = input_ids.shape[1]
        gen_t = torch.tensor(self._gen_ids, dtype=torch.long).unsqueeze(0)
        full = torch.cat([input_ids, gen_t], dim=1)
        if self._mode == "plain_tensor":
            return full
        else:
            class _Out:
                pass
            o = _Out()
            o.sequences = full
            return o

    # forward pass used by _letter_probs_refeed
    def __call__(self, input_ids=None, attention_mask=None):
        import torch
        # Return logits where token 66 ("B") has the highest score at last position
        vocab = 32000
        logits = torch.zeros(1, input_ids.shape[1], vocab)
        logits[0, -1, 66] = 10.0   # "B" wins
        class _ModelOut:
            pass
        o = _ModelOut()
        o.logits = logits
        return o


def _make_fake_llm(gen_ids, generate_mode="plain_tensor"):
    """Build an LLM-like object using _FakeModel and _FakeTokenizer, bypassing __init__."""
    import torch
    from src import llm as llm_module
    obj = object.__new__(llm_module.LLM)
    obj.tok = _FakeTokenizer()
    obj.model = _FakeModel(gen_ids, mode=generate_mode)
    obj.cfg = {"name": "test-model", "reasoning": True}
    obj.device = "cpu"
    obj.torch = torch
    obj.is_reasoning = True
    obj.eos = {1}          # eos_token_id = 1
    obj.pad_id = 0
    obj._letter_ids = {}
    return obj


class TestGenerateReasoningOutputShape(unittest.TestCase):
    """Verify that generate_reasoning handles both plain-Tensor and .sequences outputs."""

    # gen_ids that spell "\nAnswer: B" (token 66 = 'B', 1 = eos)
    GEN_IDS = [10, 65, 110, 115, 119, 101, 114, 58, 32, 66, 1]  # arbitrary ids ending in 66 then eos
    # Simpler: just use token 66 (B) and eos (1)
    GEN_IDS_SIMPLE = [66, 1]

    def _run_and_check(self, llm_obj):
        """Run generate_reasoning with a single prompt and check the returned structure."""
        prompt = "What is the answer?"
        letters = ["A", "B", "C", "D"]
        results = llm_obj.generate_reasoning([prompt], [letters], max_new_tokens=32)
        self.assertEqual(len(results), 1)
        r = results[0]
        # Schema checks
        self.assertIn("raw_output", r)
        self.assertIn("gen_tokens", r)
        self.assertIn("prompt_tokens", r)
        self.assertIn("truncated", r)
        self.assertIn("seconds", r)
        self.assertIn("_pred", r)
        self.assertIn("_pred_source", r)
        self.assertIn("_parsed", r)
        # gen_tokens should equal len(GEN_IDS_SIMPLE) - 1 (eos trimmed)
        self.assertEqual(r["gen_tokens"], 1)  # [66] after eos strip
        return r

    def test_plain_tensor_output(self):
        """model.generate returns a plain Tensor — must not raise AttributeError."""
        llm = _make_fake_llm(self.GEN_IDS_SIMPLE, generate_mode="plain_tensor")
        r = self._run_and_check(llm)
        # pred should come from logprob_refeed (fake decode of [66] → "B" may not parse)
        self.assertIn(r["_pred_source"], ("parsed", "logprob_refeed", "none"))

    def test_model_output_with_sequences(self):
        """model.generate returns an object with .sequences — must work identically."""
        llm = _make_fake_llm(self.GEN_IDS_SIMPLE, generate_mode="model_output")
        r = self._run_and_check(llm)
        self.assertIn(r["_pred_source"], ("parsed", "logprob_refeed", "none"))

    def test_both_paths_produce_same_gen_tokens(self):
        """The two return-type paths must produce the same gen_tokens count."""
        r_tensor = _make_fake_llm(self.GEN_IDS_SIMPLE, "plain_tensor").generate_reasoning(
            ["Q?"], [["A", "B", "C", "D"]], 32)
        r_obj = _make_fake_llm(self.GEN_IDS_SIMPLE, "model_output").generate_reasoning(
            ["Q?"], [["A", "B", "C", "D"]], 32)
        self.assertEqual(r_tensor[0]["gen_tokens"], r_obj[0]["gen_tokens"])

    def test_eos_stripped_correctly(self):
        """EOS token must not appear in gen_tokens count."""
        # gen_ids = [66, 1] → after eos strip → [66] → gen_tokens=1
        llm = _make_fake_llm([66, 1], "plain_tensor")
        results = llm.generate_reasoning(["Q?"], [["A", "B", "C", "D"]], 32)
        self.assertEqual(results[0]["gen_tokens"], 1)

    def test_truncated_flag_set_when_no_answer_line(self):
        """If all gen_ids were consumed without an Answer: line, truncated=True."""
        # GEN IDS that don't include any recognisable "Answer:" pattern
        # and eos never appears (so cut == len(all_gen_ids))
        # Use a single non-eos token: [99]
        llm = _make_fake_llm([99], "plain_tensor")
        results = llm.generate_reasoning(["Q?"], [["A", "B", "C", "D"]], 32)
        self.assertTrue(results[0]["truncated"])

    def test_truncated_false_when_eos_hit(self):
        """If EOS is hit before cap, truncated must be False even without Answer: line."""
        # [99, 1] → cut=1, all_gen_ids has 2 items; cut(1) != len(2) → not truncated
        llm = _make_fake_llm([99, 1], "plain_tensor")
        results = llm.generate_reasoning(["Q?"], [["A", "B", "C", "D"]], 32)
        self.assertFalse(results[0]["truncated"])


if __name__ == "__main__":
    unittest.main()
