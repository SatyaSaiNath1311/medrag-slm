"""MedRAG-SLM Local Streamlit Demo Application.

Uses ONLY saved outputs — no model loading, no GPU, works offline.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "qwen3-4b": "Qwen3-4B",
    "phi4-mini": "Phi-4-mini (3.8B)",
    "gemma3-4b": "Gemma-3-4B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "smollm3-3b": "SmolLM3-3B",
}

FIGURES_ORDER = [
    ("fig1_overall_accuracy", "Overall QA Accuracy across 5 SLMs"),
    ("fig2_medqa_pubmedqa", "MedQA vs. PubMedQA Divergence"),
    ("fig3_pubmedqa_abstract", "PubMedQA Abstract Benchmark"),
    ("fig4_accuracy_vs_latency", "Accuracy vs. Latency Trade-Off"),
    ("fig5_risk_coverage", "Risk-Coverage Abstention Curves"),
    ("fig6_retrieval_bottleneck", "Retrieval Bottleneck & Difference-in-Differences"),
    ("fig7_fixed_vs_induced", "MedQA Error Transitions (Fixed vs. Induced)"),
    ("fig8_overconfidence", "Persistent Overconfidence in SLMs"),
    ("fig9_calibration", "Phi-4-mini Calibration & Reliability Diagram"),
    ("fig10_memory", "Peak VRAM & Hardware Footprint"),
]


# -----------------------------------------------------------------------------
# Cached Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_test_questions() -> List[Dict[str, Any]]:
    """Load test split questions from phase 1."""
    path = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase1" / "test.jsonl"
    questions = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    return questions


@st.cache_data(show_spinner=False)
def load_evidence_dict() -> Dict[str, List[Dict[str, Any]]]:
    """Load top-5 retrieved & reranked passages per question from phase 7 evidence."""
    path = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase7" / "evidence.jsonl"
    evidence_by_id: Dict[str, List[Dict[str, Any]]] = {}
    missing_chunk_ids: set[str] = set()

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = row.get("id")
                passages = row.get("passages", [])
                evidence_by_id[qid] = passages
                for p in passages:
                    if not p.get("text"):
                        missing_chunk_ids.add(p.get("chunk_id"))

    # Fallback lookup in chunks.jsonl if text was ever stripped
    if missing_chunk_ids:
        chunk_path = REPO_ROOT / "outputs" / "kaggle_build" / "work" / "phase3" / "chunks.jsonl"
        if chunk_path.exists():
            chunk_lookup = {}
            with open(chunk_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    c = json.loads(line)
                    cid = c.get("chunk_id")
                    if cid in missing_chunk_ids:
                        chunk_lookup[cid] = c
                        if len(chunk_lookup) == len(missing_chunk_ids):
                            break
            for passages in evidence_by_id.values():
                for p in passages:
                    if not p.get("text") and p.get("chunk_id") in chunk_lookup:
                        p["text"] = chunk_lookup[p["chunk_id"]].get("text", "")
                        p["title"] = p.get("title") or chunk_lookup[p["chunk_id"]].get("title", "")

    return evidence_by_id


@st.cache_data(show_spinner=False)
def load_model_outputs(model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load test predictions for Baseline (phase 5), RAG (phase 8), and Context (phase 10)."""
    p5_path = REPO_ROOT / "outputs" / "kaggle_qa" / "full" / model_name / "work" / "phase5" / f"{model_name}.jsonl"
    p8_path = REPO_ROOT / "outputs" / "kaggle_qa" / "full" / model_name / "work" / "phase8" / f"{model_name}.jsonl"
    p10_path = REPO_ROOT / "outputs" / "kaggle_qa" / "context" / "work" / "phase10" / f"{model_name}.jsonl"

    baseline_by_id = {}
    if p5_path.exists():
        with open(p5_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("split") == "test":
                        baseline_by_id[r["id"]] = r

    rag_by_id = {}
    if p8_path.exists():
        with open(p8_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("split") == "test":
                        rag_by_id[r["id"]] = r

    context_by_id = {}
    if p10_path.exists():
        with open(p10_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("split") == "test":
                        context_by_id[r["id"]] = r

    return baseline_by_id, rag_by_id, context_by_id


@st.cache_data(show_spinner=False)
def load_gate_settings(model_name: str) -> Dict[str, Any]:
    """Load tuned gate thresholds from adaptive_rag_<model>.json."""
    path = REPO_ROOT / "outputs" / "analysis" / f"adaptive_rag_{model_name}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_captions() -> Dict[str, str]:
    """Parse figure captions from outputs/figures/README.md."""
    path = REPO_ROOT / "outputs" / "figures" / "README.md"
    captions = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("| **fig"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 4:
                        fig_key = parts[1].replace("*", "").strip()
                        caption = parts[3].strip()
                        captions[fig_key] = caption
    return captions


@st.cache_data(show_spinner=False)
def load_presentation_tables_markdown() -> str:
    """Load benchmark presentation tables markdown."""
    path = REPO_ROOT / "outputs" / "analysis" / "presentation_tables.md"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Presentation tables file not found."


def render_image(image_path: Path | str, caption: str = ""):
    """Render image using width='stretch' or fallback to use_container_width."""
    try:
        st.image(str(image_path), caption=caption, width="stretch")
    except (TypeError, ValueError):
        st.image(str(image_path), caption=caption, use_container_width=True)


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="MedRAG-SLM Demo",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Clean, high-contrast light theme CSS (no hard-coded dark blues)
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #1A1A1A;
            margin-bottom: 0.25rem;
            line-height: 1.2;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #4A4A4A;
            margin-bottom: 1.25rem;
            line-height: 1.5;
        }
        .question-text-box {
            font-size: 18px;
            font-weight: 400;
            color: #1A1A1A;
            line-height: 1.65;
            margin-bottom: 1.2rem;
        }
        .option-box {
            font-size: 16px;
            padding: 0.65rem 0.95rem;
            margin-bottom: 0.45rem;
            border-radius: 6px;
            background-color: #F4F6F8;
            border: 1px solid #D0D5DD;
            color: #1A1A1A;
            font-weight: 400;
            line-height: 1.5;
        }
        .option-box-correct {
            font-size: 16px;
            padding: 0.65rem 0.95rem;
            margin-bottom: 0.45rem;
            border-radius: 6px;
            background-color: #E6F4EA;
            border: 1.5px solid #28A745;
            color: #0F5132;
            font-weight: 600;
            line-height: 1.5;
        }
        .badge-correct {
            display: inline-block;
            background-color: #28A745;
            color: #FFFFFF;
            font-weight: 700;
            padding: 0.22rem 0.65rem;
            border-radius: 4px;
            font-size: 0.85rem;
            margin-bottom: 0.45rem;
        }
        .badge-wrong {
            display: inline-block;
            background-color: #DC3545;
            color: #FFFFFF;
            font-weight: 700;
            padding: 0.22rem 0.65rem;
            border-radius: 4px;
            font-size: 0.85rem;
            margin-bottom: 0.45rem;
        }
        .col-heading {
            font-size: 1.22rem;
            font-weight: 700;
            color: #1A1A1A;
            margin-bottom: 0.6rem;
            line-height: 1.3;
        }
        .col-subheading {
            font-size: 0.92rem;
            font-weight: 500;
            color: #4A4A4A;
        }
        .cited-highlight {
            background-color: #EFF6FF;
            border-left: 4px solid #1F77B4;
            padding: 0.6rem 0.85rem;
            margin-bottom: 0.6rem;
            border-radius: 4px;
            color: #1A1A1A;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🩺 MedRAG-SLM: Clinical QA & Adaptive RAG Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Systematic empirical study of accuracy, evidence grounding, abstention, and gating across 5 Small Language Models. (Offline, Saved Outputs Only)</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🩺 Live Example", "📊 Results", "⚙️ How It Works"])

    # Load master questions & evidence
    all_questions = load_test_questions()
    evidence_by_id = load_evidence_dict()

    # -------------------------------------------------------------------------
    # TAB 1: Live Example
    # -------------------------------------------------------------------------
    with tab1:
        # Sidebar Controls
        st.sidebar.markdown("### ⚙️ Demo Controls")
        selected_model = st.sidebar.selectbox(
            "Model",
            options=list(MODELS.keys()),
            format_func=lambda x: MODELS[x],
            index=0,
            help="Select the Small Language Model to inspect.",
        )

        dataset_choice = st.sidebar.selectbox(
            "Dataset",
            options=["MedQA", "PubMedQA", "Unanswerable"],
            index=0,
            help="MedQA (USMLE clinical board MCQs), PubMedQA (biomedical research), or Unanswerable (adversarial questions with correct option removed).",
        )

        filter_choice = st.sidebar.selectbox(
            "Filter",
            options=["All", "RAG fixed it", "RAG broke it", "Both wrong", "Both right"],
            index=0,
            help="Filter questions by model behavior transition between Baseline and Full RAG.",
        )

        # Load predictions and gate settings for chosen model
        baseline_by_id, rag_by_id, context_by_id = load_model_outputs(selected_model)
        gate_settings = load_gate_settings(selected_model)
        tau = gate_settings.get("tuning_validation", {}).get("rerank_gate", {}).get("best_tau", 5.0)

        # Filter candidate question IDs
        filtered_ids: List[str] = []
        for q in all_questions:
            qid = q["id"]
            q_dataset = q.get("dataset")
            should_ab = q.get("should_abstain", False)

            # Match dataset
            if dataset_choice == "MedQA":
                if q_dataset != "medqa" or should_ab:
                    continue
            elif dataset_choice == "PubMedQA":
                if q_dataset != "pubmedqa" or should_ab:
                    continue
            elif dataset_choice == "Unanswerable":
                if not should_ab and q_dataset != "unanswerable":
                    continue

            # Model outputs
            b_row = baseline_by_id.get(qid)
            r_row = rag_by_id.get(qid)
            if not b_row or not r_row:
                continue

            b_correct = b_row.get("correct", False)
            r_correct = r_row.get("correct", False)

            # Match filter
            if filter_choice == "All":
                filtered_ids.append(qid)
            elif filter_choice == "RAG fixed it":
                if not b_correct and r_correct:
                    filtered_ids.append(qid)
            elif filter_choice == "RAG broke it":
                if b_correct and not r_correct:
                    filtered_ids.append(qid)
            elif filter_choice == "Both wrong":
                if not b_correct and not r_correct:
                    filtered_ids.append(qid)
            elif filter_choice == "Both right":
                if b_correct and r_correct:
                    filtered_ids.append(qid)

        st.sidebar.caption(f"**Found**: {len(filtered_ids)} matching questions")

        if not filtered_ids:
            st.warning(
                f"No questions match filter **'{filter_choice}'** for dataset **'{dataset_choice}'** on model **{MODELS[selected_model]}**."
            )
        else:
            # Handle session state for question index navigation
            filter_key = f"{selected_model}_{dataset_choice}_{filter_choice}"
            if "last_filter_key" not in st.session_state or st.session_state["last_filter_key"] != filter_key:
                st.session_state["last_filter_key"] = filter_key
                st.session_state["q_idx"] = 0

            # Random Question Button
            col_rand, col_info = st.sidebar.columns([1, 1])
            with col_rand:
                if st.button("🎲 Random", help="Jump to a random question in this filter"):
                    st.session_state["q_idx"] = random.randint(0, len(filtered_ids) - 1)
                    st.rerun()

            # Question selector
            cur_idx = min(st.session_state.get("q_idx", 0), len(filtered_ids) - 1)
            q_num = st.sidebar.number_input(
                "Question #",
                min_value=1,
                max_value=len(filtered_ids),
                value=cur_idx + 1,
                step=1,
                help="Enter question index.",
            )
            st.session_state["q_idx"] = q_num - 1
            selected_qid = filtered_ids[st.session_state["q_idx"]]

            # Active question row
            q_data = next((q for q in all_questions if q["id"] == selected_qid), None)
            b_data = baseline_by_id.get(selected_qid, {})
            r_data = rag_by_id.get(selected_qid, {})
            ctx_data = context_by_id.get(selected_qid, {})
            passages = evidence_by_id.get(selected_qid, [])

            if q_data:
                is_unans = q_data.get("should_abstain", False)
                gold_answer = q_data.get("answer")
                options = q_data.get("options", {})

                # Render Question Card (Everything INSIDE a single bordered container)
                with st.container(border=True):
                    st.markdown(
                        f"<div style='font-size: 0.92rem; color: #4A4A4A; margin-bottom: 0.6rem;'>"
                        f"<b>Question {st.session_state['q_idx'] + 1} of {len(filtered_ids)}</b> &nbsp;|&nbsp; "
                        f"<b>ID:</b> <code>{selected_qid}</code> &nbsp;|&nbsp; "
                        f"<b>Dataset:</b> <code>{q_data.get('dataset', '').upper()}</code>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Question text inside card: readable ~18px, normal weight
                    st.markdown(
                        f'<div class="question-text-box">{q_data.get("question")}</div>',
                        unsafe_allow_html=True,
                    )

                    # Reveal correct answer toggle
                    reveal_answer = st.toggle("👁️ Reveal correct answer", value=False, key="reveal_toggle")

                    st.markdown("<div style='font-weight: 600; color: #1A1A1A; margin-bottom: 0.4rem;'>Options:</div>", unsafe_allow_html=True)
                    for opt_key in sorted(options.keys()):
                        opt_val = options[opt_key]
                        if reveal_answer and not is_unans and opt_key == gold_answer:
                            st.markdown(
                                f'<div class="option-box-correct">✅ <b>{opt_key}.</b> {opt_val} &nbsp; <i>(Correct Answer)</i></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="option-box"><b>{opt_key}.</b> {opt_val}</div>',
                                unsafe_allow_html=True,
                            )

                    if reveal_answer:
                        if is_unans:
                            st.warning(
                                "⚠️ **Adversarial Unanswerable Question**: None of the listed options is medically correct! "
                                "The true gold option was intentionally removed to test if the model detects absent evidence or hallucinates with unwarranted confidence."
                            )
                        else:
                            st.success(f"**Gold Answer:** Option **{gold_answer}** — {options.get(gold_answer, '')}")

                # Unanswerable warning banner
                if is_unans:
                    st.warning(
                        "⚠️ **No option is correct here (Unanswerable Question).** "
                        "Notice the model confidence below: even when faced with impossible choices, SLMs frequently output 90–100% confidence."
                    )

                # -------------------------------------------------------------
                # Three Columns: (1) Baseline | (2) RAG | (3) Smart Gate
                # -------------------------------------------------------------
                col1, col2, col3 = st.columns([1, 1.3, 1])

                # (1) WITHOUT BOOK (BASELINE)
                with col1:
                    st.markdown(
                        '<div class="col-heading">📚 (1) Without Book<br><span class="col-subheading">Parametric Baseline</span></div>',
                        unsafe_allow_html=True,
                    )
                    b_pred = b_data.get("pred", "—")
                    b_conf = float(b_data.get("confidence") or 0.0)
                    b_correct = b_data.get("correct", False)

                    # Only show correctness badge when reveal_answer is ON
                    if reveal_answer:
                        if is_unans:
                            st.markdown('<span class="badge-wrong">❌ Hallucinated</span>', unsafe_allow_html=True)
                        elif b_correct:
                            st.markdown('<span class="badge-correct">✅ Correct</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-wrong">❌ Incorrect</span>', unsafe_allow_html=True)

                    st.markdown(f"**Predicted:** Option **{b_pred}** ({options.get(b_pred, '')})")
                    st.markdown(f"**Confidence:** `{b_conf * 100:.1f}%`")
                    st.progress(min(max(b_conf, 0.0), 1.0))

                    # Letter probability breakdown
                    letter_probs = b_data.get("letter_probs", {})
                    if letter_probs:
                        prob_str = " &nbsp;|&nbsp; ".join([f"**{k}:** {v*100:.1f}%" for k, v in sorted(letter_probs.items())])
                        st.caption(f"Probabilities: {prob_str}")

                    b_sec = b_data.get("seconds", 0.0)
                    st.caption(f"⏱️ Latency: {b_sec:.2f}s &nbsp;|&nbsp; Tokens: {b_data.get('gen_tokens', 0)}")

                # (2) WITH TEXTBOOK (RAG)
                with col2:
                    st.markdown(
                        '<div class="col-heading">📖 (2) With Textbook<br><span class="col-subheading">Full RAG (Top-5 Passages)</span></div>',
                        unsafe_allow_html=True,
                    )
                    r_pred = r_data.get("pred", "—")
                    r_conf = float(r_data.get("confidence") or 0.0)
                    r_correct = r_data.get("correct", False)
                    citations = r_data.get("citations") or []
                    top_rerank = r_data.get("top_rerank_score")

                    # Only show correctness badge when reveal_answer is ON
                    if reveal_answer:
                        if is_unans:
                            st.markdown('<span class="badge-wrong">❌ Hallucinated</span>', unsafe_allow_html=True)
                        elif r_correct:
                            st.markdown('<span class="badge-correct">✅ Correct</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-wrong">❌ Incorrect</span>', unsafe_allow_html=True)

                    st.markdown(f"**Predicted:** Option **{r_pred}** ({options.get(r_pred, '')})")
                    st.markdown(f"**Confidence:** `{r_conf * 100:.1f}%`")
                    st.progress(min(max(r_conf, 0.0), 1.0))

                    cite_display = ", ".join([f"Passage [{c}]" for c in citations]) if citations else "None"
                    st.caption(f"📌 **Citations:** {cite_display}")
                    if top_rerank is not None:
                        st.caption(f"⭐ **Top Rerank Score:** `{top_rerank:.2f}`")

                    # Retrieved Passages Expanders
                    st.markdown("**Retrieved Passages:**")
                    if passages:
                        for p in passages:
                            p_rank = p.get("rank")
                            p_title = p.get("title", "Clinical Textbook")
                            p_score = p.get("rerank_score", 0.0)
                            p_text = p.get("text", "")
                            is_cited = p_rank in citations

                            exp_label = f"{'⭐ [CITED] ' if is_cited else ''}Passage {p_rank}: {p_title} (Score: {p_score:.2f})"
                            with st.expander(exp_label, expanded=is_cited):
                                if is_cited:
                                    st.markdown(
                                        '<div class="cited-highlight">⭐ <b>Cited by Model:</b> The model explicitly referenced this passage in its answer output.</div>',
                                        unsafe_allow_html=True,
                                    )
                                st.write(p_text)
                    else:
                        st.caption("No evidence passages found for this question.")

                    r_sec = r_data.get("seconds", 0.0)
                    st.caption(f"⏱️ Latency: {r_sec:.2f}s &nbsp;|&nbsp; Tokens: {r_data.get('gen_tokens', 0)}")

                # (3) SMART GATE
                with col3:
                    st.markdown(
                        '<div class="col-heading">⚡ (3) Smart Gate<br><span class="col-subheading">Adaptive Dynamic Routing</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Confidence Gate Decision
                    conf_chose_rag = r_conf > b_conf
                    conf_pred = r_pred if conf_chose_rag else b_pred
                    conf_correct = (conf_pred == gold_answer) and not is_unans

                    st.markdown("##### 1. Confidence Gate")
                    st.markdown(f"**Selected:** `{'RAG' if conf_chose_rag else 'Baseline'}` (Option **{conf_pred}**)")

                    # Only show correctness badge when reveal_answer is ON
                    if reveal_answer:
                        if is_unans:
                            st.markdown('<span class="badge-wrong">❌ Incorrect</span>', unsafe_allow_html=True)
                        elif conf_correct:
                            st.markdown('<span class="badge-correct">✅ Correct</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-wrong">❌ Incorrect</span>', unsafe_allow_html=True)

                    if conf_chose_rag:
                        st.caption(f"💡 *Reason:* RAG confidence ({r_conf*100:.1f}%) > Baseline ({b_conf*100:.1f}%) → Routed to RAG.")
                    else:
                        st.caption(f"💡 *Reason:* Baseline confidence ({b_conf*100:.1f}%) ≥ RAG ({r_conf*100:.1f}%) → Kept Baseline.")

                    st.markdown("---")

                    # Rerank Gate Decision
                    rerank_chose_rag = top_rerank is not None and top_rerank >= tau
                    rerank_pred = r_pred if rerank_chose_rag else b_pred
                    rerank_correct = (rerank_pred == gold_answer) and not is_unans

                    st.markdown("##### 2. Rerank Gate")
                    st.markdown(f"**Selected:** `{'RAG' if rerank_chose_rag else 'Baseline'}` (Option **{rerank_pred}**)")

                    # Only show correctness badge when reveal_answer is ON
                    if reveal_answer:
                        if is_unans:
                            st.markdown('<span class="badge-wrong">❌ Incorrect</span>', unsafe_allow_html=True)
                        elif rerank_correct:
                            st.markdown('<span class="badge-correct">✅ Correct</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-wrong">❌ Incorrect</span>', unsafe_allow_html=True)

                    score_disp = f"{top_rerank:.2f}" if top_rerank is not None else "N/A"
                    if rerank_chose_rag:
                        st.caption(f"💡 *Reason:* Top rerank score {score_disp} ≥ threshold {tau:.2f} (high relevance) → Routed to RAG.")
                    else:
                        st.caption(f"💡 *Reason:* Top rerank score {score_disp} < threshold {tau:.2f} (insufficient relevance) → Kept Baseline.")

                    st.caption(f"🎯 *Tuned Threshold (Validation):* $\\tau = {tau:.2f}$")

                # PubMedQA Extra Row: With Study Abstract
                if dataset_choice == "PubMedQA":
                    st.markdown("---")
                    st.markdown("### 🔬 Extra Benchmark: With Study Abstract (+Abstract Oracle Context)")
                    ctx_pred = ctx_data.get("pred", "—")
                    ctx_conf = float(ctx_data.get("confidence") or 0.0)
                    ctx_correct = ctx_data.get("correct", False)

                    c_a1, c_a2 = st.columns([1, 2])
                    with c_a1:
                        if reveal_answer:
                            if ctx_correct:
                                st.markdown('<span class="badge-correct">✅ Correct</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-wrong">❌ Incorrect</span>', unsafe_allow_html=True)
                        st.markdown(f"**Predicted:** Option **{ctx_pred}** ({options.get(ctx_pred, '')})")
                        st.markdown(f"**Confidence:** `{ctx_conf * 100:.1f}%`")
                        st.progress(min(max(ctx_conf, 0.0), 1.0))
                    with c_a2:
                        abstract_text = q_data.get("context") or "No study abstract available."
                        with st.expander("📑 View Full Study Abstract (Original Paper Context)", expanded=False):
                            st.write(abstract_text)

    # -------------------------------------------------------------------------
    # TAB 2: Results
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 📊 Benchmark Results & Empirical Findings")
        st.markdown(
            "Empirical results across 5 small open-weight language models evaluated under uniform prompt structure, "
            "sampling parameters (greedy, temperature 0), and identical test sets."
        )

        captions = load_captions()

        # Display 10 figures in order
        for idx, (fig_key, title) in enumerate(FIGURES_ORDER, start=1):
            png_path = REPO_ROOT / "outputs" / "figures" / f"{fig_key}.png"
            caption_text = captions.get(fig_key, "Caption from outputs/figures/README.md")

            st.markdown(f"#### Figure {idx}: {title}")
            if png_path.exists():
                render_image(png_path)
                st.info(f"**Finding:** {caption_text}")
            else:
                st.warning(f"Figure file not found: `{png_path.name}`")
            st.markdown("---")

        # Expander with Presentation Tables
        with st.expander("📋 Presentation Tables (Full Empirical Benchmark Data)", expanded=False):
            tables_md = load_presentation_tables_markdown()
            st.markdown(tables_md)

    # -------------------------------------------------------------------------
    # TAB 3: How It Works
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### ⚙️ How It Works: Medical RAG Architecture & Gating Pipeline")

        arch_img_path = REPO_ROOT / "app" / "assets" / "architecture.png"
        if arch_img_path.exists():
            render_image(
                arch_img_path,
                caption="End-to-End Medical RAG Architecture: Dense/Sparse Retrieval, Cross-Encoder Reranking, and Adaptive Gating.",
            )
        else:
            st.warning(
                "⚠️ Architecture diagram image (`app/assets/architecture.png`) not found. "
                "Place an architecture image in `app/assets/architecture.png` to display here."
            )

        st.markdown("---")
        st.markdown("#### 🔬 Six Plain-English Steps of the Pipeline")

        steps = [
            (
                "1. Questions",
                "Clinical board questions from USMLE Step 1/2/3 (MedQA) and biomedical research study questions (PubMedQA). "
                "Adversarial Unanswerable questions (with the gold option removed) evaluate model overconfidence and hallucination resistance.",
            ),
            (
                "2. Textbook Library",
                "A curated corpus of 18 premier medical textbooks (Harrison's Internal Medicine, Schwartz's Surgery, Katzung's Pharmacology, etc.) "
                "chunked into ~125,000 dense semantic passages (~1,000 characters each).",
            ),
            (
                "3. Search (Hybrid Dual-Engine)",
                "Each query undergoes parallel dual-stage retrieval: sparse BM25 (capturing precise drug names, rare syndromes, and exact clinical entities) "
                "and dense MedCPT FAISS embeddings (capturing clinical semantics). Candidates are fused via Reciprocal Rank Fusion (RRF) to produce top-20 candidates.",
            ),
            (
                "4. Rerank",
                "A MedCPT cross-encoder scores relevance by performing full cross-attention between question and candidate text, "
                "filtering to the top 5 most relevant clinical evidence passages with calibrated rerank scores.",
            ),
            (
                "5. Small Model Answers",
                "A compact open-weight SLM (1.7B to 4B parameters: Qwen3, Phi-4-mini, Gemma-3, SmolLM3) predicts the answer letter under both Baseline "
                "(parametric-only) and Full RAG (augmented with the 5 retrieved textbook passages and instructions to output bracketed citations).",
            ),
            (
                "6. Smart Gate & Confidence",
                "Dynamic gating routes each query to the optimal answer path: Rerank Gate avoids noisy retrieval by checking if the top rerank score meets "
                "threshold $\\tau$ (saving 35–50% inference latency), while Confidence Gate picks the higher-confidence output. Temperature scaling calibrates probabilities "
                "for safe clinical abstention.",
            ),
        ]

        for step_title, step_desc in steps:
            st.markdown(f"**{step_title}**")
            st.write(step_desc)
            st.write("")


if __name__ == "__main__":
    main()
