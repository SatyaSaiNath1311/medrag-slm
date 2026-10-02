# MedRAG-SLM Live Chatbot (Kaggle GPU)

This directory contains the Gradio-based interactive medical chatbot demo running with live GPU acceleration, full hybrid BM25 + dense MedCPT FAISS retrieval, cross-encoder reranking, and adaptive confidence gating across clinical textbooks.

---

## How to Start the Chatbot in a Kaggle Interactive Session

1. **Push or Open the Kernel on Kaggle**:
   - Push from terminal:
     ```bash
     kaggle kernels push -p kaggle_chat
     ```
   - Or open the kernel in your browser at `https://www.kaggle.com/code/satyasainath1311/medrag-slm-chat`.

2. **Open in Interactive Edit Mode**:
   - Click **Edit** (or **Edit Notebook/Script**) in the top right to start a live Kaggle interactive session.

3. **Verify Session Settings** (in the right-hand panel):
   - **Accelerator**: GPU (either `GPU T4 x2` or `GPU T4 x1`).
   - **Internet**: **On** (required to load Hugging Face model weights).
   - **Input**: Ensure `medrag-build` is attached under **Input** (`Add Input -> Your Work -> medrag-build`).

4. **Run the Chatbot**:
   - Click **Run All** (or run `python kaggle_chat/chat_app.py` in the interactive console).
   - At startup, the app loads:
     - The textbook chunks (`chunks.jsonl`)
     - BM25 and FAISS dense indexes (`phase3/`)
     - MedCPT query encoder and cross-encoder reranker
     - The default model (`Qwen3-4B`) in `float16` on GPU.

---

## Where the Public Link Appears

Once initialization finishes (typically 30–60 seconds), Gradio generates a secure public tunnel and prints:

```text
======================================================================
🎉 GRADIO PUBLIC LINK GENERATED:
👉 Running on public URL: https://xxxxxxxx.gradio.live
======================================================================
```

- Click this URL in the Kaggle console output to interact with the medical chatbot in any browser on desktop or mobile.

---

## Important Notes on Session Lifetime

- **Session Lifetime**: The public Gradio link is active **only while the Kaggle interactive session is running**.
- If the Kaggle browser tab is closed or reaches the 12-hour session timeout, the server will terminate.
- You can restart the session at any time by clicking **Edit → Run All**.
