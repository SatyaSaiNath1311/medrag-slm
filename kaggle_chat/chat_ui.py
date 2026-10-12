"""Clean chat UI for the health-worker assistant (Gradio; Gradio is imported only inside build_ui).

Backends plug in through one small interface, so the same UI runs on a Mac with canned demo
answers (MockBackend) and later on Kaggle with the real MedPsy pipeline:

    class Backend:
        is_demo: bool
        def stream(self, question, history):      # history: list of {"role", "content"}
            yield "Checking for danger signs..."  # progress text, any number of times
            yield Reply(...)                      # final answer, exactly once, last

Run locally:   python3 kaggle_chat/chat_ui.py
Self-test:     python3 kaggle_chat/chat_ui.py --selftest   (no Gradio needed)
"""
from __future__ import annotations

import html
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List

APP_TITLE = "Health Worker Assistant"
APP_SUBTITLE = "Answers with references for community health workers. It will say so when it is unsure."
EMERGENCY_TEXT = "This may be an emergency. Go to the nearest hospital or call 108 immediately."
ABSTAIN_TEXT = "I don't have enough information to answer this confidently. Please consult a doctor."
FOOTER_TEXT = "Decision support only \u2014 please consult a doctor for medical decisions."
MAX_QUESTION_CHARS = 2000


@dataclass
class Reference:
    title: str
    snippet: str = ""
    source: str = ""


@dataclass
class Reply:
    status: str                       # answered | abstained | emergency | info | error
    text: str = ""
    references: List[Reference] = field(default_factory=list)
    type_label: str = ""
    emergency_reason: str = ""


# ---------------------------------------------------------------- safe markdown -> html
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])")
_BULLET = re.compile(r"^\s*[-*\u2022]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_HEAD = re.compile(r"^\s*#{1,6}\s+(.*)$")
_RULE = re.compile(r"^\s*-{3,}\s*$")


def _inline(s: str) -> str:
    s = html.escape(s, quote=True)          # escape first, then add our own tags
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _ITAL.sub(r"<em>\1</em>", s)
    return s


def md_to_html(text: str) -> str:
    out, para, items = [], [], []
    kind = None

    def flush_para():
        if para:
            out.append("<p>" + "<br>".join(_inline(x) for x in para) + "</p>")
            para.clear()

    def flush_list():
        nonlocal kind
        if items:
            tag = "ol" if kind == "ol" else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + f"</{tag}>")
            items.clear()
        kind = None

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        if not line.strip() or _RULE.match(line):
            flush_para(); flush_list(); continue
        mh = _HEAD.match(line)
        if mh:
            flush_para(); flush_list()
            out.append(f"<h4>{_inline(mh.group(1))}</h4>"); continue
        mb, mn = _BULLET.match(line), _NUMBERED.match(line)
        if mb or mn:
            flush_para()
            k = "ul" if mb else "ol"
            if kind not in (None, k):
                flush_list()
            kind = k
            items.append((mb or mn).group(1)); continue
        flush_list(); para.append(line)
    flush_para(); flush_list()
    return "".join(out)


# ---------------------------------------------------------------- html rendering
def _refs_html(refs: List[Reference]) -> str:
    if not refs:
        return ""
    lis = []
    for r in refs:
        src = f' <span class="hw-ref-src">{html.escape(r.source)}</span>' if r.source else ""
        snip = (r.snippet or "").strip()
        if len(snip) > 320:
            snip = snip[:320].rsplit(" ", 1)[0] + "\u2026"
        body = f'<div class="hw-ref-snip">{html.escape(snip)}</div>' if snip else ""
        lis.append(f'<li><div class="hw-ref-title">{html.escape(r.title)}{src}</div>{body}</li>')
    return f'<details class="hw-refs"><summary>References ({len(refs)})</summary><ol>{"".join(lis)}</ol></details>'


def _card_html(reply: Reply) -> str:
    st, parts = reply.status, []
    if st == "emergency":
        sub = f'<div class="hw-banner-sub">{html.escape(reply.emergency_reason)}</div>' if reply.emergency_reason else ""
        parts.append(f'<div class="hw-banner hw-emergency" role="alert"><strong>\u26a0 {html.escape(EMERGENCY_TEXT)}</strong>{sub}</div>')
    elif st == "abstained":
        parts.append(f'<div class="hw-banner hw-abstain" role="status"><strong>{html.escape(ABSTAIN_TEXT)}</strong></div>')
    elif st == "error":
        parts.append('<div class="hw-banner hw-error" role="alert"><strong>Something went wrong. Please try again, or consult a doctor.</strong></div>')
    chips = []
    if reply.type_label:
        chips.append(f'<span class="hw-chip">{html.escape(reply.type_label)}</span>')
    if st == "answered":
        n = len(reply.references)
        if n:
            chips.append('<span class="hw-chip hw-ok">Answered \u00b7 ' + f"{n} reference" + ("s" if n != 1 else "") + "</span>")
        else:
            chips.append('<span class="hw-chip">No references found</span>')
    if chips:
        parts.append(f'<div class="hw-chips">{"".join(chips)}</div>')
    if reply.text.strip():
        parts.append(f'<div class="hw-answer">{md_to_html(reply.text)}</div>')
    parts.append(_refs_html(reply.references))
    if st in ("answered", "abstained", "emergency"):
        parts.append(f'<div class="hw-foot">{html.escape(FOOTER_TEXT)}</div>')
    return f'<div class="hw-card hw-s-{st}">' + "".join(parts) + "</div>"


WELCOME_HTML = (
    '<div class="hw-scroll"><div class="hw-inner"><div class="hw-card"><div class="hw-answer">'
    "<p><strong>How to ask</strong></p><ul>"
    "<li>Say who the patient is: age, sex, and whether pregnant.</li>"
    "<li>Give the main problem and how long it has lasted.</li>"
    "<li>Mention any danger signs: trouble breathing, convulsions, heavy bleeding, very drowsy.</li></ul>"
    "<p>Answers come with references. If I am not sure, I will say so and suggest seeing a doctor.</p>"
    "</div></div></div></div>"
)


def render_turns(turns: list) -> str:
    if not turns:
        return WELCOME_HTML
    out = []
    for t in turns:
        q = html.escape(t["q"]).replace("\n", "<br>")
        if t.get("reply") is not None:
            body = _card_html(t["reply"])
        else:
            prog = html.escape(t.get("progress") or "Working on it\u2026")
            body = f'<div class="hw-progress"><span class="hw-dots"><i></i><i></i><i></i></span> {prog}</div>'
        out.append(
            '<div class="hw-turn">'
            f'<div class="hw-msg hw-user"><div class="hw-who">You</div><div class="hw-bubble">{q}</div></div>'
            f'<div class="hw-msg hw-bot"><div class="hw-who">Assistant</div>{body}</div></div>'
        )
    return '<div class="hw-scroll"><div class="hw-inner">' + "".join(out) + "</div></div>"


def header_html(title: str, demo: bool) -> str:
    demo_html = ""
    if demo:
        demo_html = ('<div class="hw-demo"><strong>DEMO MODE</strong> \u2014 sample answers only, not medical advice. '
                     "Try: a child fever question, \u201cunconscious\u201d, or \u201cabstain\u201d.</div>")
    return (f'<div class="hw-header"><div class="hw-title">{html.escape(title)}</div>'
            f'<div class="hw-sub">{html.escape(APP_SUBTITLE)}</div>{demo_html}</div>')


def footer_html() -> str:
    return ('<div class="hw-disclaimer">Decision support only. It does not replace a doctor. '
            "In an emergency go to the nearest hospital or call 108.</div>")


CSS = """
:root{--hw-text:#1f2937;--hw-muted:#6b7280;--hw-line:#e5e7eb;--hw-panel:#f6f8fa;--hw-accent:#0f6e6e;--hw-user:#e3f0f0;
--hw-red-bg:#fdecec;--hw-red-bd:#e5484d;--hw-red-tx:#8a1c1c;--hw-amb-bg:#fff6e0;--hw-amb-bd:#e0a100;--hw-amb-tx:#6b4a00;
--hw-grn-bg:#e8f6ee;--hw-grn-tx:#1a6b3c}
.dark{--hw-text:#e5e7eb;--hw-muted:#9ca3af;--hw-line:#2c3748;--hw-panel:#1b2433;--hw-accent:#4fb3b3;--hw-user:#173236;
--hw-red-bg:#3a1d1f;--hw-red-tx:#ffb4b4;--hw-amb-bg:#3a2f12;--hw-amb-tx:#ffd98a;--hw-grn-bg:#15301f;--hw-grn-tx:#8fd9ac}
.gradio-container{max-width:880px !important;margin:0 auto !important}
.hw-header{padding:14px 4px 6px}
.hw-title{font-size:1.45rem;font-weight:650;color:var(--hw-text);letter-spacing:-.01em}
.hw-sub{color:var(--hw-muted);font-size:.95rem;margin-top:2px}
.hw-demo{margin-top:10px;padding:8px 12px;border-radius:8px;background:var(--hw-amb-bg);border:1px solid var(--hw-amb-bd);color:var(--hw-amb-tx);font-size:.88rem}
.hw-scroll{display:flex;flex-direction:column-reverse;min-height:260px;max-height:62vh;overflow-y:auto;padding:6px 2px}
.hw-inner{display:flex;flex-direction:column;gap:20px}
.hw-turn{display:flex;flex-direction:column;gap:10px}
.hw-who{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--hw-muted);margin-bottom:4px}
.hw-user .hw-who{text-align:right}
.hw-user .hw-bubble{width:fit-content;max-width:92%;margin-left:auto;background:var(--hw-user);color:var(--hw-text);border-radius:12px 12px 4px 12px;padding:10px 14px;overflow-wrap:anywhere}
.hw-card{background:var(--hw-panel);border:1px solid var(--hw-line);border-radius:12px;padding:14px 16px;color:var(--hw-text);line-height:1.55}
.hw-s-emergency{border-color:var(--hw-red-bd)}
.hw-banner{padding:10px 12px;border-radius:8px;margin-bottom:10px;border:1px solid}
.hw-emergency,.hw-error{background:var(--hw-red-bg);border-color:var(--hw-red-bd);color:var(--hw-red-tx)}
.hw-abstain{background:var(--hw-amb-bg);border-color:var(--hw-amb-bd);color:var(--hw-amb-tx)}
.hw-banner-sub{font-size:.85rem;margin-top:4px;opacity:.9}
.hw-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.hw-chip{font-size:.75rem;padding:2px 9px;border-radius:999px;border:1px solid var(--hw-line);color:var(--hw-muted)}
.hw-chip.hw-ok{background:var(--hw-grn-bg);color:var(--hw-grn-tx);border-color:transparent}
.hw-answer p{margin:0 0 .6em}
.hw-answer ul,.hw-answer ol{margin:0 0 .7em 1.2em;padding:0}
.hw-answer h4{margin:.9em 0 .3em;font-size:.98rem}
.hw-refs{margin-top:10px;border-top:1px solid var(--hw-line);padding-top:8px;font-size:.9rem}
.hw-refs summary{cursor:pointer;font-weight:600;color:var(--hw-accent)}
.hw-refs ol{margin:8px 0 0 1.2em;padding:0}
.hw-refs li{margin-bottom:8px}
.hw-ref-src{color:var(--hw-muted);font-weight:400;font-size:.82rem}
.hw-ref-snip{color:var(--hw-muted);font-size:.85rem;margin-top:2px}
.hw-foot{margin-top:10px;font-size:.78rem;color:var(--hw-muted)}
.hw-progress{display:flex;align-items:center;gap:10px;padding:12px 14px;border:1px dashed var(--hw-line);border-radius:12px;color:var(--hw-muted)}
.hw-dots i{display:inline-block;width:6px;height:6px;margin-right:3px;border-radius:50%;background:var(--hw-accent);animation:hw-blink 1.2s infinite both}
.hw-dots i:nth-child(2){animation-delay:.2s}
.hw-dots i:nth-child(3){animation-delay:.4s}
@keyframes hw-blink{0%,80%,100%{opacity:.2}40%{opacity:1}}
@media (prefers-reduced-motion:reduce){.hw-dots i{animation:none;opacity:.7}}
.hw-disclaimer{margin-top:8px;font-size:.8rem;color:var(--hw-muted);text-align:center}
"""


# ---------------------------------------------------------------- demo backend
def _load_router():
    try:
        try:
            from question_types import classify_question, TYPE_LABELS
        except ImportError:
            from kaggle_chat.question_types import classify_question, TYPE_LABELS
        return (lambda q, e: classify_question(q, is_emergency=e)), dict(TYPE_LABELS)
    except Exception:
        def fallback(q, e):
            if re.match(r"^\s*(hi|hello|hey|thanks?|namaste)\b", q, re.I):
                return "greeting"
            return "unclear" if len(q.split()) < 3 else "general_info"
        return fallback, {}


class MockBackend:
    """Canned answers that only demonstrate the layout. Not medical content."""
    is_demo = True
    _EMERGENCY = re.compile(
        r"unconscious|not breathing|stopped breathing|severe bleeding|bleeding heavily|convuls|seizure|chest pain|choking", re.I)

    def __init__(self, delay: float = 0.6):
        self.delay = delay
        self._classify, self._labels = _load_router()

    def _refs(self):
        return [Reference("Demo source 1", "Sample passage text appears here, shortened to a few lines.", "Textbook (demo)"),
                Reference("Demo source 2", "A second sample passage that the answer would cite.", "Guideline (demo)")]

    def stream(self, question, history):
        yield "Checking for danger signs\u2026"
        time.sleep(self.delay)
        emergency = bool(self._EMERGENCY.search(question))
        if "abstain" in question.lower() and not emergency:
            yield "Searching medical references\u2026"; time.sleep(self.delay)
            yield Reply("abstained", "Demo: this is how an abstention looks. In the live app it appears when the evidence is "
                                     "too weak or the answer was cut off.")
            return
        qtype = self._classify(question, emergency)
        label = self._labels.get(qtype, "")
        if emergency:
            yield Reply("emergency",
                        "- Stay with the person and check whether they are breathing.\n"
                        "- Do not give food or drink.\n"
                        "- Arrange transport to the nearest hospital now.\n\n*Demo text, not medical advice.*",
                        emergency_reason="Demo: an emergency keyword was found in the question.", type_label=label)
            return
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
        yield "Searching medical references\u2026"; time.sleep(self.delay)
        yield "Reading the evidence\u2026"; time.sleep(self.delay)
        if qtype == "pregnancy_child":
            text = ("**Danger signs \u2014 refer urgently if any are present**\n"
                    "- Convulsions, or very drowsy and cannot be woken\n- Unable to drink or breastfeed\n"
                    "- Fast or difficult breathing\n- Heavy vaginal bleeding in pregnancy\n"
                    "- Severe headache with blurred vision in pregnancy\n\n"
                    "**What to do now**\n- Check for the signs above first.\n"
                    "- Do not give medicines without a doctor's advice.\n- Arrange referral if any danger sign is present.\n\n"
                    "*Demo content showing the layout.*")
        elif qtype == "medicine_safety":
            text = ("I cannot give a dose in this demo.\n\n**Before giving any medicine**\n"
                    "- Check the label and the national guideline.\n"
                    "- Ask about allergies, pregnancy, other medicines, age and weight.\n"
                    "- If unsure, ask a pharmacist or doctor first.\n\n*Demo content showing the layout.*")
        else:
            text = ("**This is demo text** \u2014 it shows how an answer is laid out. In the live app the content comes from the "
                    "model and the retrieved references.\n\n### Likely causes\n- Sample item\n- Sample item\n\n"
                    "### What to check\n- Sample item\n\n### First steps\n- Sample item\n\n### When to refer\n- Sample item")
        yield Reply("answered", text, self._refs(), label)


# ---------------------------------------------------------------- Gradio app
QUICK_START = [
    ("Child with fever", "A 4-year-old child has had fever for 3 days. What should I check?"),
    ("Diarrhoea care", "How do I manage mild dehydration from diarrhoea?"),
    ("Medicine in pregnancy", "Is it safe to give paracetamol during pregnancy?"),
    ("Pregnancy danger signs", "What are the danger signs in a pregnant woman?"),
    ("Hypertension", "What is hypertension and when should the patient be referred?"),
]


def build_ui(backend, title: str = APP_TITLE):
    import gradio as gr

    def respond(message, turns):
        turns = list(turns or [])
        q = (message or "").strip()[:MAX_QUESTION_CHARS]
        if not q:
            yield render_turns(turns), turns, message
            return
        history = []
        for t in turns:
            if t.get("reply") is not None:
                history += [{"role": "user", "content": t["q"]}, {"role": "assistant", "content": t["reply"].text}]
        turn = {"q": q, "reply": None, "progress": "Starting\u2026"}
        turns.append(turn)
        yield render_turns(turns), turns, ""
        final = None
        try:
            for item in backend.stream(q, history):
                if isinstance(item, Reply):
                    final = item
                    break
                turn["progress"] = str(item)
                yield render_turns(turns), turns, ""
        except Exception as exc:
            print(f"[chat_ui] backend error: {exc!r}", file=sys.stderr)
        turn["reply"] = final or Reply("error")
        yield render_turns(turns), turns, ""

    with gr.Blocks(title=title) as ui:
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(header_html(title, bool(getattr(backend, "is_demo", False))))
        state = gr.State([])
        chat = gr.HTML(render_turns([]))
        with gr.Row():
            chips = [gr.Button(label, size="sm", variant="secondary") for label, _ in QUICK_START]
        box = gr.Textbox(placeholder="Describe the patient and your question\u2026  (Enter to send, Shift+Enter for a new line)",
                         lines=1, max_lines=5, show_label=False, container=False)
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("New conversation", variant="secondary")
        gr.HTML(footer_html())

        for btn, (_, text) in zip(chips, QUICK_START):
            btn.click(lambda t=text: t, None, box)
        send.click(respond, [box, state], [chat, state, box])
        box.submit(respond, [box, state], [chat, state, box])
        clear.click(lambda: (render_turns([]), []), None, [chat, state])
    return ui


def _selftest():
    out = md_to_html("<script>alert(1)</script> **bold**\n- one\n- two\n\n1. a\n2. b")
    assert "<script>" not in out and "&lt;script&gt;" in out and "<strong>bold</strong>" in out
    assert "<ul>" in out and "<ol>" in out
    page = render_turns([{"q": "<img src=x onerror=alert(1)>",
                          "reply": Reply("answered", "ok", [Reference("T<b>", "s")])}])
    assert "<img" not in page and "<b>" not in page
    m = MockBackend(delay=0)
    for q, status in [("hello", "info"), ("my father is unconscious", "emergency"), ("abstain test", "abstained"),
                      ("A 4-year-old child has had fever for 3 days", "answered")]:
        last = list(m.stream(q, []))[-1]
        assert isinstance(last, Reply) and last.status == status, (q, last)
    for st in ("answered", "abstained", "emergency", "info", "error"):
        assert "hw-card" in render_turns([{"q": "x", "reply": Reply(st, "t")}])
    assert "role=\"alert\"" in render_turns([{"q": "x", "reply": Reply("emergency", "t")}])
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        ui = build_ui(MockBackend())
        ui.queue()
        ui.launch(server_name="127.0.0.1", inbrowser=True)
