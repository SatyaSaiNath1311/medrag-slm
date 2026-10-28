"""Emergency red-flag screening logic for clinical triage."""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Emergency Referral Prefix
EMERGENCY_PREFIX = "⚠️ **This may be an emergency. Please go to the nearest hospital or call 108 immediately.**\n\n"

# Mandatory Footer
MANDATORY_FOOTER = "*Decision support only — please consult a doctor for medical decisions.*"


# -----------------------------------------------------------------------------
# Emergency & Red-Flag Screening Logic
# -----------------------------------------------------------------------------
NEGATION_WORDS = {"no", "not", "without", "denies", "denied", "absence", "free", "never"}
NEGATION_PHRASES = ["absence of", "free of"]
CLAUSE_DELIMITER_PATTERN = re.compile(r"[.,;:!?\n]|\bbut\b", re.IGNORECASE)


def is_negated(text: str, match_start: int) -> bool:
    """Check if a red-flag match is negated in the preceding 6 words within the same clause.

    Clauses are delimited by punctuation (.,;:!?) or the coordinating conjunction 'but'.
    Negation triggers: 'no', 'not', 'without', 'denies', 'denied', 'absence of', 'free of', 'never'.
    """
    preceding = text[:match_start]
    clauses = CLAUSE_DELIMITER_PATTERN.split(preceding)
    same_clause = clauses[-1] if clauses else ""
    same_clause_lower = same_clause.lower()

    # Check multi-word negation phrases
    for phrase in NEGATION_PHRASES:
        if phrase in same_clause_lower:
            after_phrase = same_clause_lower.split(phrase)[-1]
            words_after = re.findall(r"\b\w+\b", after_phrase)
            if len(words_after) <= 6:
                return True

    # Check single-word negation tokens
    words = re.findall(r"\b\w+\b", same_clause_lower)
    preceding_words = words[-6:]
    for w in preceding_words:
        if w in NEGATION_WORDS:
            return True

    return False


def is_infant_under_3_months(text: str) -> bool:
    """Check whether the text refers to an infant / newborn / baby under 3 months."""
    lower = text.lower()
    if re.search(r"\b(newborn|neonate)s?\b", lower):
        return True
    if re.search(r"\b(baby|infant|child)\s+under\s+(?:3|three)\s+months?\b", lower):
        return True
    # 'baby 1 month old', 'infant 2 months old', 'baby 3 weeks old', 'baby 1 month'
    if re.search(r"\b(baby|infant)\s+(?:is\s+)?(?:[0-2]\s*months?|[1-9]\s*weeks?|[0-9]+\s*days?|1\s*month|2\s*months?)\b", lower):
        return True
    # '1 month old baby', '3-week-old infant'
    if re.search(r"\b(?:[0-2]\s*months?|[1-9]\s*weeks?|[0-9]+\s*days?|1\s*month|2\s*months?)[\s-]old\s+(?:baby|infant)\b", lower):
        return True
    return False


def check_emergency(text: str) -> Tuple[bool, Optional[str]]:
    """Screen user query for acute red-flag medical emergencies.

    Monitored specific red flags:
    1. 'chest pain'
    2. 'difficulty breathing', 'can't breathe', 'cannot breathe', 'severe breathlessness', 'respiratory distress'
    3. 'unconscious', 'not responding', 'unresponsive'
    4. 'seizure', 'fits', 'convulsion'
    5. 'heavy bleeding', 'vomiting blood', 'coughing blood'
    6. Stroke signs: 'face drooping', 'slurred speech', 'sudden weakness on one side'
    7. 'severe allergic reaction', 'swelling of the face', 'swelling of the throat', 'anaphylaxis'
    8. 'suicidal', 'wants to die', 'kill himself', 'kill herself'
    9. 'poisoning', 'swallowed poison', 'overdose'
    10. Pregnancy / pregnant + (bleeding OR severe headache OR headache OR fits OR severe abdominal pain OR swelling with headache)
    11. Infant / newborn / baby under 3 months + fever

    Plain 'fever', 'headache', 'pain', 'cough' alone must NOT trigger.
    Ignores a red-flag phrase if a negation word appears within the preceding 6 words in the same clause.
    """
    if not text or not text.strip():
        return False, None

    lower_text = text.lower()

    # 1. Pregnancy emergency check
    is_pregnant = bool(re.search(r"\b(pregnan(?:t|cy)|gravida|in\s+labor|trimester)\b", lower_text))
    if is_pregnant:
        # Swelling with headache
        has_swelling = bool(re.search(r"\bswelling\b", lower_text))
        headache_matches = list(re.finditer(r"\b(?:severe\s+)?headache\b", lower_text))
        if has_swelling and headache_matches:
            if any(not is_negated(text, m.start()) for m in headache_matches):
                return True, "Pregnancy with swelling and headache"

        preg_complications = [
            ("Pregnancy with bleeding", r"\bbleeding\b"),
            ("Pregnancy with severe headache", r"\bsevere\s+headache\b"),
            ("Pregnancy with headache", r"\bheadache\b"),
            ("Pregnancy with fits", r"\b(fits?|convulsions?)\b"),
            ("Pregnancy with severe abdominal pain", r"\bsevere\s+abdominal\s+pain\b"),
        ]
        for label, pat in preg_complications:
            for m in re.finditer(pat, lower_text):
                if not is_negated(text, m.start()):
                    return True, label

    # 2. Infant / newborn / baby under 3 months + fever
    if is_infant_under_3_months(text):
        fever_matches = list(re.finditer(r"\bfever\b", lower_text))
        if fever_matches and any(not is_negated(text, m.start()) for m in fever_matches):
            return True, "Infant under 3 months with fever"

    # 3. Specific emergency phrases
    red_flag_patterns = [
        ("Chest pain", r"\bchest\s+pain\b"),
        ("Difficulty breathing", r"\b(difficulty\s+breathing|can'?t\s+breathe|cannot\s+breathe|severe\s+breathlessness|respiratory\s+distress)\b"),
        ("Unconsciousness", r"\b(unconscious|not\s+responding|unresponsive)\b"),
        ("Seizure", r"\b(seizures?|fits?|convulsions?)\b"),
        ("Heavy bleeding", r"\b(heavy\s+bleeding|vomiting\s+blood|coughing\s+(?:up\s+)?blood)\b"),
        ("Stroke signs", r"\b(face\s+drooping|facial\s+droop|slurred\s+speech|sudden\s+weakness\s+on\s+one\s+side)\b"),
        ("Severe allergic reaction", r"\b(severe\s+allergic\s+reaction|swelling\s+of\s+(?:the\s+)?face|swelling\s+of\s+(?:the\s+)?throat|anaphylaxis)\b"),
        ("Suicidal thoughts", r"\b(suicidal|wants?\s+to\s+die|kill\s+himself|kill\s+herself|kill\s+myself)\b"),
        ("Poisoning", r"\b(poisoning|swallowed\s+poison|overdose)\b"),
    ]

    for label, pat in red_flag_patterns:
        for m in re.finditer(pat, lower_text):
            if not is_negated(text, m.start()):
                return True, label

    return False, None
