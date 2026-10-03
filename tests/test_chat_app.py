"""Unit tests for MedRAG-SLM Chatbot offline logic (no GPU, no model inference).

Tests:
1. Emergency red-flag screening (check_emergency) across all critical clinical conditions
   with clause-based negation handling.
2. Abstention decision logic (evaluate_abstention) across truncation, insufficient info, and rerank thresholds.
3. Thinking text stripping (strip_thinking) ensuring <think>...</think> is never exposed to users.
"""

import sys
import unittest
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_chat.chat_app import check_emergency, evaluate_abstention, strip_thinking
from src.llm import ABSTAIN_MESSAGE


class TestChatAppLogic(unittest.TestCase):
    """Test offline safety screening, abstention evaluation, and text sanitization."""

    # -------------------------------------------------------------------------
    # 1. Emergency Red-Flag Screening Tests (Required Cases)
    # -------------------------------------------------------------------------
    def test_required_emergency_cases(self):
        """Verify the specified emergency clinical cases trigger emergency referrals."""
        # 1. Pregnant woman with swelling and headache
        matched, reason = check_emergency("Pregnant woman 7 months with leg swelling and headache — what should I check?")
        self.assertTrue(matched, "Expected emergency for pregnant woman with leg swelling and headache")
        self.assertIn("Pregnancy", reason)

        # 2. Chest pain and sweating
        matched, reason = check_emergency("My father has chest pain and is sweating")
        self.assertTrue(matched, "Expected emergency for chest pain and sweating")
        self.assertIn("Chest pain", reason)

        # 3. Baby 1 month old with high fever
        matched, reason = check_emergency("Baby 1 month old with high fever")
        self.assertTrue(matched, "Expected emergency for baby 1 month old with high fever")
        self.assertIn("Infant", reason)

    def test_required_non_emergency_cases(self):
        """Verify the specified non-emergency queries do NOT trigger false alarms."""
        non_emergencies = [
            # Negation: "without fever or respiratory distress"
            "A child has a mild runny nose and cough for 2 days without fever or respiratory distress.",
            # Routine malaria treatment inquiry
            "What is the first-line treatment for uncomplicated malaria?",
            # Adult fever antipyretic query
            "Can I give paracetamol and ibuprofen together to an adult with fever?",
            # Child with fever and rash (not infant under 3 months, no red flags)
            "A child has fever for 3 days, rash and red eyes. What could it be and what should I do?",
            # Negation: "Patient denies chest pain; mild cough"
            "Patient denies chest pain; mild cough",
        ]
        for query in non_emergencies:
            matched, reason = check_emergency(query)
            self.assertFalse(matched, f"Query falsely flagged as emergency: '{query}' (reason: {reason})")

    # -------------------------------------------------------------------------
    # Additional Red-Flag Emergency Coverage Tests
    # -------------------------------------------------------------------------
    def test_additional_emergencies(self):
        cases = [
            ("The patient is unconscious and unresponsive", "Unconsciousness"),
            ("A 4-year-old child had a seizure and convulsions", "Seizure"),
            ("Patient has heavy bleeding and vomiting blood", "Heavy bleeding"),
            ("Elderly woman with sudden face drooping and slurred speech", "Stroke signs"),
            ("Patient has severe allergic reaction with swelling of the throat", "Severe allergic reaction"),
            ("Teenager expressing suicidal thoughts and wants to die", "Suicidal thoughts"),
            ("Child swallowed poison, suspected organophosphate overdose", "Poisoning"),
            ("Pregnant woman with bleeding and severe abdominal pain", "Pregnancy with bleeding"),
        ]
        for text, expected_label in cases:
            matched, reason = check_emergency(text)
            self.assertTrue(matched, f"Failed to detect emergency for: '{text}'")

    def test_clause_bounded_negation(self):
        # Negation stops at "but" or punctuation
        text = "Patient has no fever, but has chest pain"
        matched, reason = check_emergency(text)
        self.assertTrue(matched, "Negation before 'but' must not negate subsequent emergency clause")
        self.assertIn("Chest pain", reason)

        # Plain symptoms alone must not trigger
        plain_symptoms = [
            "Patient has mild headache for 2 hours",
            "Adult with fever of 38°C and body pain",
            "Dry cough for 3 days without other symptoms",
            "Mild knee pain after walking",
        ]
        for s in plain_symptoms:
            matched, reason = check_emergency(s)
            self.assertFalse(matched, f"Plain symptom alone should not trigger emergency: '{s}'")

    # -------------------------------------------------------------------------
    # 2. Abstention Decision Logic Tests
    # -------------------------------------------------------------------------
    def test_abstention_on_truncation(self):
        # Truncation = thinking budget exhausted (validated Phase 9 rule)
        should_abstain, reason = evaluate_abstention(
            truncated=True,
            model_output="Treatment options include antibiotics...",
            top_rerank_score=5.5,
            rerank_threshold_p20=1.85,
        )
        self.assertTrue(should_abstain)
        self.assertIn("truncated", reason.lower())

    def test_abstention_on_insufficient_information(self):
        # Model explicitly outputs "INSUFFICIENT INFORMATION"
        should_abstain, reason = evaluate_abstention(
            truncated=False,
            model_output="Based on the passages, INSUFFICIENT INFORMATION is available to advise on this dosage.",
            top_rerank_score=4.8,
            rerank_threshold_p20=1.85,
        )
        self.assertTrue(should_abstain)
        self.assertIn("insufficient information", reason.lower())

    def test_abstention_on_low_rerank_score(self):
        # Top rerank score falls below validation 20th percentile
        should_abstain, reason = evaluate_abstention(
            truncated=False,
            model_output="Here are 4 points on clinical care:\n- Rest\n- Hydration",
            top_rerank_score=0.45,
            rerank_threshold_p20=1.85,
        )
        self.assertTrue(should_abstain)
        self.assertIn("below", reason.lower())

    def test_no_abstention_when_criteria_met(self):
        # Clean answer with high rerank score and complete generation
        should_abstain, reason = evaluate_abstention(
            truncated=False,
            model_output=(
                "- Artemether-lumefantrine is the first-line artemisinin-based combination therapy [1].\n"
                "- Administer a 3-day oral course with fatty food or milk [2].\n"
                "- Check rapid diagnostic test (RDT) confirmation before initiating treatment [1]."
            ),
            top_rerank_score=6.20,
            rerank_threshold_p20=1.85,
        )
        self.assertFalse(should_abstain)
        self.assertIsNone(reason)

    # -------------------------------------------------------------------------
    # 3. Thinking Text Sanitization Tests
    # -------------------------------------------------------------------------
    def test_strip_thinking_tags(self):
        raw = (
            "<think>\n"
            "The user is asking about uncomplicated malaria in a non-pregnant adult.\n"
            "Harrison chapter 250 recommends artemisinin-based combination therapy (ACT).\n"
            "</think>\n"
            "- First-line therapy is artemether-lumefantrine (ACT) [1].\n"
            "- Give oral tablets twice daily for 3 days [1]."
        )
        cleaned = strip_thinking(raw)
        self.assertNotIn("<think>", cleaned)
        self.assertNotIn("</think>", cleaned)
        self.assertNotIn("The user is asking about", cleaned)
        self.assertTrue(cleaned.startswith("- First-line therapy"))

    def test_strip_unclosed_thinking(self):
        raw = "<think>Partial thinking block without closure\n- Clinical step 1"
        cleaned = strip_thinking(raw)
        self.assertNotIn("<think>", cleaned)
        self.assertIn("Clinical step 1", cleaned)


if __name__ == "__main__":
    unittest.main()
