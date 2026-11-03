"""Live backend for kaggle_chat/chat_ui.py: runs the real MedPsy RAG pipeline.

Same interface as MockBackend in chat_ui.py:

    backend.is_demo -> False
    backend.stream(question, history) yields progress strings, then exactly one Reply, last.

All pipeline pieces (device placement, model/retrieval loading, retrieval, generation,
thinking-strip, abstention, emergency screening) are reused from chat_app.py / red_flags.py.

Run on Kaggle:   python3 kaggle_chat/live_backend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `kaggle_chat` and `src` importable when this file is run directly as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_chat.chat_ui import ABSTAIN_TEXT, Reply, Reference, _load_router
from kaggle_chat.chat_app import (
    get_llm,
    get_retrieval_system,
    get_device_placement,
    retrieve_evidence,
    generate_plain_answer,
    evaluate_abstention,
    strip_thinking,
    DEFAULT_MODEL,
)
from kaggle_chat.red_flags import check_emergency
from src.llm import parse_citations

SNIPPET_CHARS = 300

# chat_app.py has no named constant for this note; it is inlined in chat_pipeline()
# (kaggle_chat/chat_app.py, `child_preg_note`). Copied verbatim; keep the two in sync.
CHILD_PREG_NOTE = ("\n\n**Pregnant women, newborns and young children can get worse quickly. "
                   "If in doubt, refer to a doctor today.**")


def _shorten(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


class MedRAGBackend:
    """Real MedPsy pipeline: emergency screen -> route -> retrieve -> generate -> abstain check."""
    is_demo = False

    def __init__(self, model_key: str = DEFAULT_MODEL):
        llm_device, encoder_device = get_device_placement()
        # LLM first so it secures cuda:0 memory (same order as chat_app.py startup)
        self.llm = get_llm(model_key, device=llm_device)
        self.retr = get_retrieval_system(device=encoder_device)
        self.rerank_p20 = self.retr.get("rerank_p20", 1.85)
        self._classify, self._labels = _load_router()

    @staticmethod
    def _refs(top_passages) -> list:
        refs = []
        for p in top_passages:
            source = p.get("source", "") or ""
            title = p.get("title") or source or "Medical reference"
            refs.append(Reference(
                title=title,
                snippet=_shorten(p.get("text", "")),
                source=source if source != title else "",
            ))
        return refs

    def stream(self, question, history):
        yield "Checking for danger signs…"
        # Same as chat_pipeline(): screen first, but do NOT stop; the flag is carried to the final Reply.
        # The UI draws the emergency banner itself for status "emergency", so EMERGENCY_PREFIX is not
        # put into the answer text (it would show twice).
        is_emergency, emergency_reason = check_emergency(question)
        emergency_reason = (emergency_reason or "") if is_emergency else ""

        qtype = self._classify(question, is_emergency)
        label = self._labels.get(qtype, "")
        if qtype == "greeting":
            yield Reply("info", "Hello. I can help with symptoms, treatment steps, medicine safety, and pregnancy and child "
                                "danger signs. Tell me the patient's age, the main problem, how long it has lasted, and any "
                                "danger signs.", type_label=label)
            return
        if qtype == "unclear":
            yield Reply("info", "I need a little more detail to answer safely. Please tell me:\n"
                                "- The patient's age and sex\n- The main problem and how long it has lasted\n"
                                "- Any danger signs (trouble breathing, convulsions, heavy bleeding, very drowsy)",
                        type_label=label)
            return

        yield "Searching medical references…"
        q_dict = {"question": question, "options": {}}
        top_passages, top_rerank_score, _ = retrieve_evidence(q_dict, is_mcq=False)

        yield "Reading the evidence…"
        raw_output, truncated, _ = generate_plain_answer(self.llm, question, top_passages, qtype, use_reasoning=True)
        clean_text = strip_thinking(raw_output)

        should_abstain, abstain_reason = evaluate_abstention(truncated, clean_text, top_rerank_score, self.rerank_p20)
        if should_abstain:
            if is_emergency:
                # The UI shows only one banner per status; keep the emergency banner and state the
                # abstention in the body so neither message is lost.
                yield Reply("emergency", f"**{ABSTAIN_TEXT}**\n\n*(Criteria: {abstain_reason})*",
                            type_label=label, emergency_reason=emergency_reason)
            else:
                yield Reply("abstained", abstain_reason or "", type_label=label)
            return

        text = clean_text
        if qtype == "pregnancy_child":
            text += CHILD_PREG_NOTE

        # Same rule as chat_pipeline(): show only the passages the answer cites.
        cited = parse_citations(clean_text, len(top_passages))
        refs = self._refs([p for p in top_passages if p["rank"] in cited])

        yield Reply("emergency" if is_emergency else "answered", text, references=refs, type_label=label,
                    emergency_reason=emergency_reason)


if __name__ == "__main__":
    import inspect

    from kaggle_chat.chat_ui import build_ui

    ui = build_ui(MedRAGBackend())
    launch_kwargs = {"server_name": "0.0.0.0", "server_port": 7860, "share": False}
    launch_params = inspect.signature(ui.launch).parameters
    ui.launch(**{k: v for k, v in launch_kwargs.items() if k in launch_params})
