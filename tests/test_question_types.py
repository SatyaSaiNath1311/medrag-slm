"""Unit tests for the chatbot question-type router (pure Python, no GPU)."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_chat.question_types import (QUESTION_TYPES, TYPE_LABELS, build_hw_prompt,  # noqa: E402
                                        classify_question)


class TestClassify(unittest.TestCase):
    CASES = [
        ("hi", "greeting"),
        ("Good morning!", "greeting"),
        ("thank you so much", "greeting"),
        ("what can you do", "greeting"),
        ("fever", "unclear"),
        ("help me", "unclear"),
        ("", "unclear"),
        ("Pregnant woman with headache and blurred vision, what should I do?", "pregnancy_child"),
        ("6 month old baby not feeding well", "pregnancy_child"),
        ("Child with fever and cough for 3 days", "pregnancy_child"),
        ("Is ibuprofen safe to take with blood pressure tablets?", "medicine_safety"),
        ("What are the side effects of metformin?", "medicine_safety"),
        ("How to treat a minor burn at home?", "treatment"),
        ("First aid for a snake bite", "treatment"),
        ("Man aged 40 with chest pain and sweating", "symptoms"),
        ("Woman has burning urine and fever since 2 days", "symptoms"),
        ("What is tuberculosis and how does it spread?", "general_info"),
        ("Explain how vaccines work in the body", "general_info"),
    ]

    def test_cases(self):
        for text, expected in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(classify_question(text), expected)

    def test_emergency_overrides_short_or_greeting(self):
        self.assertEqual(classify_question("unconscious", is_emergency=True), "symptoms")
        self.assertEqual(classify_question("hi baby not breathing", is_emergency=True), "pregnancy_child")

    def test_greeting_with_medical_content_is_not_greeting(self):
        self.assertNotEqual(classify_question("hello my child has fever"), "greeting")

    def test_all_types_have_labels(self):
        self.assertEqual(set(QUESTION_TYPES), set(TYPE_LABELS))


class TestPrompts(unittest.TestCase):
    HEADINGS = {
        "symptoms": ["**Possible causes**", "**What to check**", "**First steps**", "**Refer urgently if**"],
        "treatment": ["**First steps**", "**What NOT to do**", "**Medicines**", "**Refer urgently if**"],
        "medicine_safety": ["**Short answer**", "**Cautions**", "**Side effects to watch**"],
        "pregnancy_child": ["**Danger signs - refer immediately if**", "**What to do now**", "**Medicines**"],
        "general_info": ["**In short**", "**Key points**", "**When to see a doctor**"],
    }

    def test_prompt_structure(self):
        for qtype, heads in self.HEADINGS.items():
            p = build_hw_prompt(qtype, "[1] (Textbook)\nsome text", "my question")
            for h in heads:
                self.assertIn(h, p)
            self.assertIn("INSUFFICIENT INFORMATION", p)
            self.assertIn("Never state doses", p)
            self.assertIn("Health worker question: my question", p)
            self.assertTrue(p.rstrip().endswith("Answer:"))

    def test_pregnancy_child_lists_danger_signs_first(self):
        p = build_hw_prompt("pregnancy_child", "ev", "q")
        self.assertLess(p.index("Danger signs"), p.index("What this could mean"))

    def test_unknown_type_falls_back(self):
        self.assertIn("**In short**", build_hw_prompt("nonsense", "ev", "q"))


if __name__ == "__main__":
    unittest.main()
