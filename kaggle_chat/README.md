# MedRAG-SLM Community Health Worker Live Chatbot (Kaggle GPU)

This directory contains the Gradio-based interactive clinical decision support application tailored for **Community Health Workers (CHWs)** and primary care providers.

The chatbot runs on a free **Kaggle GPU accelerator (T4)** with:
- **Authoritative Textbook Knowledge Base**: 124,077 chunks across 18 medical textbooks (Harrison's Internal Medicine, Katzung Pharmacology, Schwartz's Surgery, Nelson Pediatrics, Novak Gynecology, etc.).
- **Hybrid Retrieval & Reranking**: BM25 sparse lexical search + MedCPT dense FAISS embeddings (top-20) reranked by MedCPT Cross-Encoder (top-5).
- **MedPsy-4B Clinical Reasoning**: 1,024-token reasoning budget with concise health-worker bullet points.
- **Optimized 2x T4 Memory Architecture**: MedPsy-4B is loaded first and placed exclusively on `cuda:0` (~8 GB in fp16). MedCPT encoders run on `cuda:1` (or CPU on 1-GPU setups), with FAISS strictly on CPU and `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to prevent VRAM fragmentation and CUDA OOM.
- **Active Red-Flag Emergency Screening**: Detects acute danger signs (chest pain, severe breathlessness, convulsions, pregnancy complications, infant high fever) and immediately outputs a call-108 emergency referral banner.
- **Three-Tier Abstention Safety**: Seamlessly falls back to `ABSTAIN_MESSAGE` ("*I don't have enough information to answer this confidently. Please consult a doctor.*") if thinking budget is exhausted, the model signals `"INSUFFICIENT INFORMATION"`, or the top retrieval score falls below the validation 20th percentile.
- **Expandable References**: Textbook sources cited with [n], title, and first 200 characters with expandable full text.

---

## How to Start in a Kaggle Interactive Notebook

> [!NOTE]
> **Why Cloudflare Tunnel?** Gradio's built-in `share=True` tunnel (`*.gradio.live`) connects via port 7000, which is blocked by Kaggle's outbound firewall and results in a port 7000 timeout. The reliable, tested method is running the app locally on port 7860 in the background and exposing it via a Cloudflare Quick Tunnel (`*.trycloudflare.com`), which connects via standard outbound HTTPS (port 443).

### Step 1: Open an Interactive Notebook Session
1. Navigate to [Kaggle](https://www.kaggle.com/) and create a new notebook (or open your existing `medrag-slm` notebook).
2. Configure **Notebook Settings** in the right-hand sidebar:
   - **Accelerator**: `GPU T4 x2` (or `GPU T4 x1`).
   - **Internet**: **On** (required to download weights and Cloudflare binary).
   - **Environment**: Python 3.10 / PyTorch 2.x default.
3. Check GPU status to ensure both GPUs are fresh and empty:
   ```bash
   !nvidia-smi
   ```
   Both GPUs should have near 0 MB allocated. If a previous run left processes in memory, restart the session via **Session Options -> Restart Session**.

### Step 2: Add the `medrag-build` Output Dataset
1. In the right-hand panel under **Input**, click **Add Input**.
2. Select **Your Work** (or search for `medrag-build`).
3. Attach `medrag-build` so that the indexed files (`dense.faiss`, `bm25/`, `chunks.jsonl`) are available at `/kaggle/input/**/phase3`.

---

### Step 3: Run the Execution Cells

#### Cell 1: Environment Setup & Fresh Clone
```python
# Cell 1: Fresh clone and dependencies (installs tested Gradio 5.x via requirements-app.txt)
!rm -rf /kaggle/working/medrag-slm
!git clone --depth 1 https://github.com/SatyaSaiNath1311/medrag-slm.git /kaggle/working/medrag-slm
%cd /kaggle/working/medrag-slm
!pip install -q -r requirements.txt -r requirements-app.txt
```
> [!TIP]
> `requirements-app.txt` pins Gradio (`gradio==5.20.0`), ensuring full compatibility with Gradio 5.x `messages` format dictionaries (`{"role": "user"|"assistant", "content": "..."}`) and preventing interface errors.

#### Cell 2: Start the App in the Background
```python
# Cell 2: Start the app in the background
!nohup python kaggle_chat/chat_app.py > chat.log 2>&1 &
```

#### Cell 3: Wait ~75 s and Verify Local Server
```python
# Cell 3: Wait ~75 s for model and retrieval components to initialize, then verify
import time
print("Waiting ~75 seconds for MedPsy-4B and retrieval index to load...")
time.sleep(75)

!tail -3 chat.log
```
Expected output:
```text
Running on local URL:  http://0.0.0.0:7860
```

#### Cell 4: Expose via Cloudflare Quick Tunnel & Get Public URL
```python
# Cell 4: Download cloudflared, start tunnel, and print public link
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /kaggle/working/cloudflared
!chmod +x /kaggle/working/cloudflared
!nohup /kaggle/working/cloudflared tunnel --url http://localhost:7860 --no-autoupdate > cf.log 2>&1 &

import time
time.sleep(15)

# Grep and print the public https://*.trycloudflare.com link
!grep -o 'https://[-0-9a-zA-Z.]*\.trycloudflare\.com' cf.log | head -n 1
```

Click the printed `https://*.trycloudflare.com` link to open the live interactive chatbot interface in any browser!

---

## 🛠️ Troubleshooting

- **If Cell 3 shows an error or does not show `Running on local URL`**:
  ```bash
  !tail -30 chat.log
  ```
  Inspect the trace. Common causes:
  - Missing phase3 directory: check that `medrag-build` is attached under `/kaggle/input`.
  - Memory collision: run `!nvidia-smi`. If an old session process is holding VRAM, restart the notebook session.

- **If Cell 4 prints nothing or tunnel fails**:
  ```bash
  !tail -20 cf.log
  ```
  Ensure Kaggle **Internet** is toggled **On** in the notebook sidebar. You can also re-run the `grep` command after a few seconds if the tunnel was still connecting.

- **Always Start from a Fresh Session**:
  Before launching, verify with:
  ```bash
  !nvidia-smi
  ```
  Both GPU 0 and GPU 1 should show near 0 MB used. If JAX or a previous PyTorch session occupied memory, click **Run -> Restart Session** to start with clean GPU VRAM.

---

## ⚠️ Important Session Lifetime Notice

- **Session Active Period**: The public `.trycloudflare.com` link remains active **only while the Kaggle interactive session is running**.
- If the Kaggle tab is closed, disconnected for 20+ minutes of inactivity, or reaches the 12-hour session execution cap, the session terminates.
- You can restart the app at any time by re-running Cells 2, 3, and 4.

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
