"""Unit tests for MedRAG-SLM Chatbot offline logic (no GPU, no model inference).

Tests:
1. Emergency red-flag screening (check_emergency) across all critical clinical conditions.
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
    # 1. Emergency Red-Flag Screening Tests
    # -------------------------------------------------------------------------
    def test_emergency_chest_pain(self):
        matched, reason = check_emergency("A 56-year-old male presents with acute crushing chest pain and sweating.")
        self.assertTrue(matched)
        self.assertIn("Chest pain", reason)

    def test_emergency_difficulty_breathing(self):
        matched, reason = check_emergency("Child is gasping with severe shortness of breath and cannot breathe.")
        self.assertTrue(matched)
        self.assertIn("Difficulty breathing", reason)

    def test_emergency_unconsciousness(self):
        matched, reason = check_emergency("The patient suddenly collapsed and is completely unresponsive.")
        self.assertTrue(matched)
        self.assertIn("Unconsciousness", reason)

    def test_emergency_seizure(self):
        matched, reason = check_emergency("A 4-year-old boy is having a continuous seizure and active convulsions.")
        self.assertTrue(matched)
        self.assertIn("Seizure", reason)

    def test_emergency_heavy_bleeding(self):
        matched, reason = check_emergency("A trauma victim has heavy bleeding and massive blood loss from a wound.")
        self.assertTrue(matched)
        self.assertIn("Heavy bleeding", reason)

    def test_emergency_stroke_signs(self):
        matched, reason = check_emergency("Elderly woman noticed sudden facial droop, arm weakness, and slurred speech.")
        self.assertTrue(matched)
        self.assertIn("Stroke signs", reason)

    def test_emergency_allergic_reaction(self):
        matched, reason = check_emergency("Patient is developing anaphylaxis with rapid swelling of the tongue and lips.")
        self.assertTrue(matched)
        self.assertIn("allergic", reason.lower())

    def test_emergency_suicidal_thoughts(self):
        matched, reason = check_emergency("A teenager is expressing suicidal thoughts and wanting to end life.")
        self.assertTrue(matched)
        self.assertIn("Suicidal", reason)

    def test_emergency_poisoning(self):
        matched, reason = check_emergency("Child accidentally swallowed chemicals, suspected pesticide poisoning.")
        self.assertTrue(matched)
        self.assertIn("Poisoning", reason)

    def test_emergency_pregnancy_complications(self):
        # Bleeding in pregnancy
        matched1, _ = check_emergency("A pregnant woman at 30 weeks gestation presents with vaginal bleeding.")
        self.assertTrue(matched1)

        # Severe headache / preeclampsia signs in pregnancy
        matched2, _ = check_emergency("Pregnant woman 7 months with severe headache and fits.")
        self.assertTrue(matched2)

    def test_emergency_infant_high_fever(self):
        matched, reason = check_emergency("A 2-week-old newborn has a very high fever of 39.5°C and poor feeding.")
        self.assertTrue(matched)
        self.assertIn("fever in infant", reason.lower())

    def test_non_emergency_queries(self):
        # Non-emergencies must not trigger false positive red-flag warnings
        queries = [
            "Can I give paracetamol and ibuprofen together to an adult with mild fever?",
            "What is the first-line oral rehydration solution preparation for mild diarrhea?",
            "What are the typical dietary recommendations for an adult with iron-deficiency anemia?",
            "How should I dress a superficial scrape on the knee?",
            "A child has a mild runny nose and cough for 2 days without fever or respiratory distress.",
        ]
        for q in queries:
            matched, reason = check_emergency(q)
            self.assertFalse(matched, f"Query incorrectly flagged as emergency: '{q}' (reason: {reason})")

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
