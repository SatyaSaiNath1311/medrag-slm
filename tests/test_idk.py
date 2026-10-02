"""Unit tests for 'I don't know' (IDK) prompt generation and abstention parsing."""

import unittest

from src.llm import (
    ABSTAIN_MESSAGE,
    IDK_TEXT,
    baseline_idk_prompt,
    get_idk_letter,
    get_idk_options,
    parse_answer,
    rag_idk_prompt,
)


class TestIDK(unittest.TestCase):
    def setUp(self):
        self.medqa_q = {
            "id": "medqa-test-0001",
            "dataset": "medqa",
            "question": "What is the primary etiology?",
            "options": {"A": "First option", "B": "Second option", "C": "Third option", "D": "Fourth option"},
        }
        self.pubmedqa_q = {
            "id": "pubmedqa-0001",
            "dataset": "pubmedqa",
            "question": "Does medication X improve survival?",
            "options": {"A": "yes", "B": "no", "C": "maybe"},
        }
        self.unans_q = {
            "id": "unans-test-0001",
            "dataset": "unanswerable",
            "question": "What is the most likely diagnosis?",
            "options": {"A": "Inaccurate 1", "B": "Inaccurate 2", "C": "Inaccurate 3"},
        }
        self.passages = [
            {"rank": 1, "title": "Harrison", "text": "Clinical snippet supporting medicine."},
            {"rank": 2, "title": "Katzung", "text": "Pharmacological properties and mechanism."},
        ]

    def test_idk_letters_and_options(self):
        # Letters
        self.assertEqual(get_idk_letter(self.medqa_q), "E")
        self.assertEqual(get_idk_letter(self.unans_q), "E")
        self.assertEqual(get_idk_letter(self.pubmedqa_q), "D")

        # Options dictionary
        med_opts = get_idk_options(self.medqa_q)
        self.assertIn("E", med_opts)
        self.assertEqual(med_opts["E"], IDK_TEXT)
        self.assertEqual(len(med_opts), 5)

        pqa_opts = get_idk_options(self.pubmedqa_q)
        self.assertIn("D", pqa_opts)
        self.assertEqual(pqa_opts["D"], IDK_TEXT)
        self.assertEqual(len(pqa_opts), 4)

        unans_opts = get_idk_options(self.unans_q)
        self.assertIn("E", unans_opts)
        self.assertEqual(unans_opts["E"], IDK_TEXT)

    def test_idk_prompt_instructions(self):
        # MedQA baseline IDK
        b_prompt = baseline_idk_prompt(self.medqa_q)
        self.assertIn("E. I do not have enough information to answer this question.", b_prompt)
        self.assertIn("If none of the options is correct or you do not have enough information to answer, choose E.", b_prompt)

        # PubMedQA baseline IDK
        p_b_prompt = baseline_idk_prompt(self.pubmedqa_q)
        self.assertIn("D. I do not have enough information to answer this question.", p_b_prompt)
        self.assertIn("If none of the options is correct or you do not have enough information to answer, choose D.", p_b_prompt)

        # MedQA RAG IDK
        r_prompt = rag_idk_prompt(self.medqa_q, self.passages, max_chars=100)
        self.assertIn("E. I do not have enough information to answer this question.", r_prompt)
        self.assertIn("If none of the options is correct or you do not have enough information to answer, choose E.", r_prompt)
        self.assertIn("[1] (Harrison)", r_prompt)

    def test_idk_answer_parsing_as_abstain(self):
        med_opts = get_idk_options(self.medqa_q)
        pqa_opts = get_idk_options(self.pubmedqa_q)

        # Explicit letter E parsed as ABSTAIN
        self.assertEqual(parse_answer("Answer: E", med_opts), "ABSTAIN")
        self.assertEqual(parse_answer("**E. I do not have enough information**", med_opts), "ABSTAIN")
        self.assertEqual(parse_answer("E", med_opts), "ABSTAIN")
        self.assertEqual(parse_answer("Answer: E.", med_opts), "ABSTAIN")

        # Explicit letter D parsed as ABSTAIN for PubMedQA
        self.assertEqual(parse_answer("Answer: D", pqa_opts), "ABSTAIN")
        self.assertEqual(parse_answer("D. I do not have enough information", pqa_opts), "ABSTAIN")

        # Verbatim option text matched as ABSTAIN
        self.assertEqual(parse_answer("Answer: I do not have enough information to answer this question.", med_opts), "ABSTAIN")
        self.assertEqual(parse_answer("I do not have enough information to answer this question.", med_opts), "ABSTAIN")

        # Regular options parse normally
        self.assertEqual(parse_answer("Answer: B", med_opts), "B")
        self.assertEqual(parse_answer("Answer: A", pqa_opts), "A")
        self.assertEqual(parse_answer("Answer: yes", pqa_opts), "A")

    def test_abstain_message_constant(self):
        expected = "I don't have enough information to answer this confidently. Please consult a doctor."
        self.assertEqual(ABSTAIN_MESSAGE, expected)


if __name__ == "__main__":
    unittest.main()
