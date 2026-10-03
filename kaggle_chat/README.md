# MedRAG-SLM Community Health Worker Live Chatbot (Kaggle GPU)

This directory contains the Gradio-based interactive clinical decision support application tailored for **Community Health Workers (CHWs)** and primary care providers.

The chatbot runs on a free **Kaggle GPU accelerator (T4)** with:
- **Authoritative Textbook Knowledge Base**: 124,078 chunks across 18 medical textbooks (Harrison's Internal Medicine, Katzung Pharmacology, Schwartz's Surgery, Nelson Pediatrics, Novak Gynecology, etc.).
- **Hybrid Retrieval & Reranking**: BM25 sparse lexical search + MedCPT dense FAISS embeddings (top-20) reranked by MedCPT Cross-Encoder (top-5).
- **MedPsy-4B Clinical Reasoning**: 1,024-token reasoning budget with concise health-worker bullet points.
- **Active Red-Flag Emergency Screening**: Detects acute danger signs (chest pain, severe breathlessness, convulsions, pregnancy complications, infant high fever) and immediately outputs a call-108 emergency referral banner.
- **Three-Tier Abstention Safety**: Seamlessly falls back to `ABSTAIN_MESSAGE` ("*I don't have enough information to answer this confidently. Please consult a doctor.*") if thinking budget is exhausted, the model signals `"INSUFFICIENT INFORMATION"`, or the top retrieval score falls below the validation 20th percentile.
- **Expandable References**: Textbook sources cited with [n], title, and first 200 characters with expandable full text.

---

## How to Start in a Kaggle Interactive Notebook

Follow these steps to launch the live demo on Kaggle with a public Gradio URL:

### Step 1: Open an Interactive Notebook Session
1. Navigate to [Kaggle](https://www.kaggle.com/) and create a new notebook (or open your existing `medrag-slm` notebook).
2. Configure **Notebook Settings** in the right-hand sidebar:
   - **Accelerator**: `GPU T4 x2` (or `GPU T4 x1`).
   - **Internet**: **On** (required to download Hugging Face model weights).
   - **Environment**: Python 3.10 / PyTorch 2.x default.

### Step 2: Add the `medrag-build` Output Dataset
1. In the right-hand panel under **Input**, click **Add Input**.
2. Select **Your Work** (or search for `medrag-build`).
3. Attach `medrag-build` so that the indexed files (`dense.faiss`, `bm25/`, `chunks.jsonl`) are available at `/kaggle/input/**/phase3`.

### Step 3: Run the Two Execution Cells

#### Cell 1: Environment Setup & Code Sync
```python
# Cell 1: Setup dependencies and clone repository
!pip install -q gradio bm25s PyStemmer faiss-gpu-cu12 transformers accelerate
!git clone https://github.com/SatyaSaiNath1311/medrag-slm.git /kaggle/working/medrag-slm
%cd /kaggle/working/medrag-slm
```

#### Cell 2: Launch the Chatbot
```python
# Cell 2: Start the Gradio live session with public link
!python kaggle_chat/chat_app.py
```

---

## Finding the Public Gradio Link

Once the retrieval index and model are loaded into GPU VRAM (approx. 40–60 seconds), Gradio will output:

```text
======================================================================
🎉 LAUNCHING GRADIO APP WITH PUBLIC LINK (share=True)...
======================================================================
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxxxxxx.gradio.live
```

- **Open the `https://xxxxxxxx.gradio.live` link** in any web browser on your computer or mobile phone.
- The link is secured by Gradio's global HTTPS tunnel.

---

## ⚠️ Important Session Lifetime Notice

- **Session Active Period**: The public `.gradio.live` link remains active **only while the Kaggle interactive session is running**.
- If the Kaggle tab is closed, disconnected for 20+ minutes of inactivity, or reaches the 12-hour session execution cap, the tunnel will terminate.
- You can restart the app at any time by pressing **Run** in the Kaggle notebook.

---

## 📹 How to Record a Backup Screen Video for Project Defense

Because live cloud GPU sessions can time out or experience network drops during presentations, it is highly recommended to record a 1-to-2 minute backup walkthrough:

1. **Start Screen Recording**:
   - **macOS**: Press `Cmd + Shift + 5`, select the browser window containing the Gradio chatbot, and click **Record**.
   - **Windows**: Press `Win + Alt + R` (Xbox Game Bar) or use OBS Studio.
2. **Demonstrate Key Capabilities**:
   - **Emergency Detection**: Click the chest pain or pregnancy complication example and show the immediate `"⚠️ This may be an emergency. Please go to the nearest hospital or call 108 immediately."` warning banner.
   - **Health Worker Guidance**: Ask a clinical question (e.g. child fever, rash, and red eyes or paracetamol + ibuprofen co-administration).
   - **Progress & Speed**: Note the `"🩺 Thinking… (about 20–40 s)"` indicator followed by the concise bulleted advice and response latency timer.
   - **Expandable References**: Expand the `[1]`, `[2]` textbook passage cards to demonstrate evidence grounded in Nelson/Harrison textbooks.
   - **Abstention Safety**: Submit an ungrounded or ambiguous query to show the calibrated `"🛡️ Abstention Advisory: I don't have enough information to answer this confidently. Please consult a doctor."` response.
3. Save the resulting `.mp4` / `.mov` file to your project portfolio.
