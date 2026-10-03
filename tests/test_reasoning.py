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


if __name__ == "__main__":
    unittest.main()
