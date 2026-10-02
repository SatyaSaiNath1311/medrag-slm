import unittest
from src.llm import FORMAT_RAG, PROMPT_HEAD, context_prompt, parse_answer, parse_citations


class TestContextPrompt(unittest.TestCase):
    def setUp(self):
        self.question = {
            "id": "pubmedqa-12345",
            "dataset": "pubmedqa",
            "question": "Does treatment X improve outcome Y?",
            "options": {"A": "yes", "B": "no", "C": "maybe"},
            "answer": "A",
            "context": "BACKGROUND: Treatment X was investigated. RESULTS: Outcome Y was significantly improved.",
        }

    def test_context_prompt_structure(self):
        prompt = context_prompt(self.question)
        self.assertIn(PROMPT_HEAD, prompt)
        self.assertIn("Use the numbered evidence passages.", prompt)
        self.assertIn("Evidence:\n[1] (Abstract) BACKGROUND: Treatment X was investigated. RESULTS: Outcome Y was significantly improved.", prompt)
        self.assertIn("Question: Does treatment X improve outcome Y?", prompt)
        self.assertIn("Options:\nA. yes\nB. no\nC. maybe", prompt)
        self.assertIn(FORMAT_RAG, prompt)

    def test_context_prompt_truncation(self):
        max_chars = 40
        prompt = context_prompt(self.question, max_chars=max_chars)
        expected_abstract = self.question["context"][:max_chars]
        self.assertIn(f"[1] (Abstract) {expected_abstract}", prompt)
        self.assertNotIn(self.question["context"], prompt)

    def test_context_prompt_empty_context(self):
        q_empty = {
            "id": "pubmedqa-99999",
            "dataset": "pubmedqa",
            "question": "Is this a test?",
            "options": {"A": "yes", "B": "no", "C": "maybe"},
            "context": None,
        }
        prompt = context_prompt(q_empty)
        self.assertIn("[1] (Abstract) ", prompt)

    def test_parse_context_output(self):
        output = "A\nEvidence: [1]"
        self.assertEqual(parse_answer(output, self.question["options"]), "A")
        self.assertEqual(parse_citations(output, 1), [1])


if __name__ == "__main__":
    unittest.main()
