import unittest
from src.llm import parse_answer, parse_citations


class TestParsing(unittest.TestCase):
    def test_parse_answer(self):
        medqa_options = {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"}
        pubmedqa_options = {"A": "yes", "B": "no", "C": "maybe"}

        # bold "**B. …**"
        self.assertEqual(parse_answer("**B. Some text**", medqa_options), "B")
        self.assertEqual(parse_answer("**B.**", medqa_options), "B")

        # "Answer: C. maybe"
        self.assertEqual(parse_answer("Answer: C. maybe", medqa_options), "C")

        # "Answer: yes" for PubMedQA
        self.assertEqual(parse_answer("Answer: yes", pubmedqa_options), "A")

        # "None of the options…" -> None
        self.assertIsNone(parse_answer("None of the options are correct", medqa_options))
        self.assertIsNone(parse_answer("Answer: None of the options are correct", medqa_options))

    def test_parse_citations(self):
        n_passages = 5
        # "[2][3][4]"
        self.assertEqual(parse_citations("[2][3][4]", n_passages), [2, 3, 4])
        # "[1] [4]"
        self.assertEqual(parse_citations("[1] [4]", n_passages), [1, 4])
        # "[1, 2, 3, 4]"
        self.assertEqual(parse_citations("[1, 2, 3, 4]", n_passages), [1, 2, 3, 4])
        # "[1,2]"
        self.assertEqual(parse_citations("[1,2]", n_passages), [1, 2])
        # "Evidence: [4] (Pharmacology_Katzung) <copied passage text>"
        self.assertEqual(
            parse_citations("Evidence: [4] (Pharmacology_Katzung) <copied passage text>", n_passages),
            [4],
        )
        # "Evidence:\n[1] text…"
        self.assertEqual(parse_citations("Evidence:\n[1] text…", n_passages), [1])

        # Ignore numbers outside 1..n_passages and deduplicate
        self.assertEqual(parse_citations("[0, 1, 3, 99]", n_passages), [1, 3])
        self.assertEqual(parse_citations("[2, 2, 4][4]", n_passages), [2, 4])


if __name__ == "__main__":
    unittest.main()
