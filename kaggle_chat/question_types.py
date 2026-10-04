"""Question-type routing and structured answer prompts for the health-worker chatbot.

Pure Python (no torch / gradio) so it can be unit-tested anywhere.

Types (checked in this priority order):
  greeting          hi / thanks / who are you            -> instant canned reply, no model call
  unclear           too short to answer safely           -> instant clarifying questions, no model call
  pregnancy_child   pregnancy, newborn, infant, child    -> danger signs FIRST, no medicine without doctor
  medicine_safety   dose, tablet, side effect, safe to take -> cautions, never invent doses
  treatment         how to treat / manage / first aid    -> first steps, what not to do, referral
  symptoms          complaint-style question             -> likely causes, what to check, first steps, referral
  general_info      everything else ("what is …")        -> short explanation, key points, when to see a doctor

An emergency red flag (detected elsewhere) always overrides greeting/unclear so a
one-word "unconscious" is never answered with a canned reply.
"""
import re
from typing import Dict

QUESTION_TYPES = ("greeting", "unclear", "pregnancy_child", "medicine_safety",
                  "treatment", "symptoms", "general_info")

TYPE_LABELS: Dict[str, str] = {
    "greeting": "Greeting",
    "unclear": "Needs more detail",
    "pregnancy_child": "Pregnancy / child health",
    "medicine_safety": "Medicine safety",
    "treatment": "Treatment / care",
    "symptoms": "Symptoms",
    "general_info": "General information",
}

_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|namaste|namaskar|good\s+(morning|afternoon|evening|night)|"
    r"thanks?|thank\s+you|thank\s+u|ok(ay)?|bye|who\s+are\s+you|what\s+can\s+you\s+do|"
    r"how\s+are\s+you)\b[\s!.,?]*(\w+[\s!.,?]*){0,3}$",
    re.IGNORECASE,
)

_PREG_CHILD = [
    r"pregnan\w*", r"antenatal", r"prenatal", r"postnatal", r"post-?partum", r"labou?r pains?",
    r"delivery", r"miscarriage", r"trimester", r"breast-?feed\w*", r"lactat\w*",
    r"new-?born", r"neonat\w*", r"infant", r"baby", r"babies", r"toddler", r"child", r"children",
    r"kid", r"kids", r"\d+\s*(day|week|month)s?[\s-]*old", r"\d+\s*-?\s*(year|yr)s?[\s-]*old\s+(boy|girl)",
]
_MEDICINE = [
    r"dose", r"dosage", r"doses", r"tablets?", r"pills?", r"syrup", r"capsules?", r"injection",
    r"medicines?", r"medications?", r"drugs?", r"side[\s-]?effects?", r"safe\s+to\s+(take|give|use)",
    r"can\s+(i|we|he|she|they)\s+(take|give)", r"overdose", r"\bmg\b", r"interaction", r"together\s+with",
    r"paracetamol", r"acetaminophen", r"ibuprofen", r"aspirin", r"antibiotic\w*", r"amoxicillin",
    r"metformin", r"insulin", r"\bors\b", r"iron\s+(tablet|folic)", r"folic\s+acid", r"expired",
]
_TREATMENT = [
    r"treat\w*", r"manage\w*", r"\bcure\b", r"first[\s-]aid", r"what\s+(should|can|do)\s+(i|we)\s+do",
    r"how\s+(to|do\s+i|should\s+i)\s+(care|help|stop|reduce|handle)", r"home\s+(care|remed\w*)",
    r"what\s+to\s+give", r"remed(y|ies)",
]
_SYMPTOMS = [
    r"pain", r"ache", r"fever", r"cough", r"vomit\w*", r"diarrh\w*", r"loose\s+motions?", r"rash",
    r"swell\w*", r"bleed\w*", r"headache", r"breath\w*", r"dizz\w*", r"weak\w*", r"itch\w*",
    r"burn\w*", r"wound", r"jaundice", r"yellow", r"chills?", r"faint\w*", r"seizure", r"fits?",
    r"symptoms?", r"suffer\w*", r"complain\w*", r"since\s+\d+\s+(day|week)s?", r"for\s+\d+\s+(day|week)s?",
    r"what\s+(could|can|might)\s+(it|this)\s+be", r"cause[sd]?",
]


def _any(patterns, text):
    return any(re.search(rf"\b{p}\b", text, re.IGNORECASE) for p in patterns)


def classify_question(text: str, is_emergency: bool = False) -> str:
    """Rule-based question type. Deterministic, explainable, no model call."""
    t = (text or "").strip()
    words = re.findall(r"[A-Za-z0-9]+", t)
    medical = _any(_PREG_CHILD + _MEDICINE + _TREATMENT + _SYMPTOMS, t)

    if not is_emergency:
        if not words:
            return "unclear"
        if _GREETING_RE.match(t) and not medical:
            return "greeting"
        if len(words) <= 2:
            return "unclear"
    if _any(_PREG_CHILD, t):
        return "pregnancy_child"
    if _any(_MEDICINE, t):
        return "medicine_safety"
    if _any(_TREATMENT, t):
        return "treatment"
    if _any(_SYMPTOMS, t) or is_emergency:
        return "symptoms"
    return "general_info"


GREETING_REPLY = (
    "Namaste! I am a medical decision-support assistant for health workers.\n\n"
    "You can ask me about:\n"
    "- **Symptoms** (e.g. *\"Child with fever and cough for 3 days, what could it be?\"*)\n"
    "- **Treatment and first aid** (e.g. *\"How to manage dehydration from diarrhoea?\"*)\n"
    "- **Medicine safety** (e.g. *\"Is ibuprofen safe in pregnancy?\"*)\n"
    "- **Pregnancy and child danger signs**\n\n"
    "Please include age, main problem, how long it has lasted, and any other symptoms. "
    "I answer from medical textbooks with references, and I will tell you when to refer to a doctor."
)

UNCLEAR_REPLY = (
    "I need a little more detail to answer safely. Please tell me:\n"
    "- **Who** is the patient? (age, sex, pregnant or not)\n"
    "- **What** is the main problem?\n"
    "- **How long** has it been going on?\n"
    "- **Other symptoms** (fever, vomiting, bleeding, breathing difficulty, drowsiness?)\n\n"
    "Example: *\"30-year-old woman, fever and burning urine for 2 days, not pregnant.\"*"
)

_COMMON_RULES = (
    "Rules:\n"
    "- Use simple words a community health worker understands. Short bullet points, at most 8 in total.\n"
    "- Use ONLY the evidence passages and well-established medical knowledge.\n"
    "- Cite a passage as [n] only if it directly supports that exact line; otherwise no citation.\n"
    "- Never state doses, numbers or timelines that are not written in the passages.\n"
    "- Do not diagnose with certainty; say 'possible' or 'likely'.\n"
    "- If the passages and your knowledge are not enough to answer safely, reply with exactly "
    "'INSUFFICIENT INFORMATION' and nothing else.\n"
)

_TEMPLATES: Dict[str, str] = {
    "symptoms": (
        "Answer using exactly these headings, in this order:\n"
        "**Possible causes** - 2 to 3 causes, most likely first.\n"
        "**What to check** - simple things to look at or ask (temperature, breathing, hydration, etc.).\n"
        "**First steps** - safe care the health worker can give now.\n"
        "**Refer urgently if** - danger signs that need a doctor or hospital today.\n"
    ),
    "treatment": (
        "Answer using exactly these headings, in this order:\n"
        "**First steps** - practical care, in the order to do them.\n"
        "**What NOT to do** - common harmful mistakes to avoid.\n"
        "**Medicines** - name medicines only if supported by the passages; for doses say "
        "'follow the label or a doctor's prescription'.\n"
        "**Refer urgently if** - danger signs that need a doctor or hospital.\n"
    ),
    "medicine_safety": (
        "Answer using exactly these headings, in this order:\n"
        "**Short answer** - one line: generally safe / use with caution / avoid, and for whom.\n"
        "**Cautions** - who should not take it (pregnancy, children, kidney/liver disease, allergies, other medicines).\n"
        "**Side effects to watch** - the important ones only.\n"
        "**Check with a doctor before giving if** - situations that need a prescriber.\n"
        "Never give a dose unless it is written in the passages; otherwise say 'follow the label or a doctor's prescription'.\n"
    ),
    "pregnancy_child": (
        "This concerns a pregnant woman, newborn, infant or child: be extra careful.\n"
        "Answer using exactly these headings, in this order:\n"
        "**Danger signs - refer immediately if** - list these FIRST.\n"
        "**What this could mean** - 1 to 2 possible causes.\n"
        "**What to do now** - safe first steps only.\n"
        "**Medicines** - say: do not give any medicine without a doctor's advice, unless the passages "
        "clearly support it as safe for this group.\n"
    ),
    "general_info": (
        "Answer using exactly these headings, in this order:\n"
        "**In short** - 1 to 2 simple sentences.\n"
        "**Key points** - up to 4 bullets (causes, prevention, care).\n"
        "**When to see a doctor** - 1 to 2 bullets.\n"
    ),
}


def build_hw_prompt(qtype: str, evidence_text: str, question: str) -> str:
    """Prompt for a model-answered question type (not greeting/unclear)."""
    if qtype not in _TEMPLATES:
        qtype = "general_info"
    return (
        "You are a clinical decision-support assistant for community health workers in places "
        "where a doctor may not be available.\n\n"
        f"Question type: {TYPE_LABELS[qtype]}.\n"
        f"{_TEMPLATES[qtype]}\n"
        f"{_COMMON_RULES}\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Health worker question: {question}\n\n"
        "Answer:"
    )
