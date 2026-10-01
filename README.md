# medrag-slm: setup guide

Do these steps in order. Each step has a check. Don't move on until the check passes.

## Step 1: Accounts (one time, ~20 min)
1. **GitHub**: create an account and a new **public** repository named `medrag-slm`.
2. **Kaggle**: verify your phone number (Settings), which is required for GPU and internet.
   Then Settings -> API -> Create New Token, which downloads `kaggle.json`.
3. **Hugging Face**: create an account, then Settings -> Access Tokens -> new token (Read).
   Open https://huggingface.co/google/gemma-3-4b-it and accept the licence.

Check: you have `kaggle.json`, an HF token, and the Gemma page shows you have access.

## Step 2: Laptop setup (one time)
1. Install Python 3.11 and Git.
2. Open this folder in Antigravity, then in its terminal:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt kaggle
   ```
3. Put `kaggle.json` in `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).
4. Add `PROJECT_RULES.md` to Antigravity's rules, or tell the agent to read it at the start of each task.

Check: `kaggle kernels list --mine` runs without an error.

## Step 3: Local code check (tiny model, CPU)
```
python -m src.smoke_test --config configs/base.yaml --out outputs/local --tiny
```
Check: the summary shows `OVERALL: PASS`. This only proves the code runs, not the real models.

## Step 4: Push the code to GitHub
```
git init
git add .
git commit -m "setup + smoke test"
git branch -M main
git remote add origin https://github.com/<you>/medrag-slm.git
git push -u origin main
```
Check: the files are visible on github.com.

## Step 5: Real test on Kaggle GPU
1. Edit `kaggle_runner/kernel-metadata.json`: replace `YOUR_KAGGLE_USERNAME`.
2. Edit `kaggle_runner/runner.py`: replace `YOUR_GITHUB_USERNAME`.
3. Push it:
   ```
   kaggle kernels push -p kaggle_runner
   ```
4. **One time, on the Kaggle website**: open the `medrag-slm-runner` notebook -> Edit.
   - Settings: Accelerator = **GPU T4 x2**, Internet = **On**
   - Add-ons -> Secrets -> add `HF_TOKEN` (your Hugging Face token) and tick it for this notebook
   - Click **Save Version** -> Save & Run All
5. Watch progress and download the results:
   ```
   kaggle kernels status <you>/medrag-slm-runner
   kaggle kernels output <you>/medrag-slm-runner -p outputs/kaggle
   ```

Check: `outputs/kaggle/smoke/smoke_report.json` says `"overall": "PASS"`, with all 5 models PASS.

## Step 6: If something fails

| Problem in the report | Fix |
|---|---|
| Cannot access model / GatedRepoError | HF_TOKEN secret not attached, or Gemma licence not accepted |
| Non-finite logits (NaN/inf) | In `configs/base.yaml` set that model's `dtype: float32` (needs T4 x2) |
| P100 detected | Switch the notebook to GPU T4 x2 |
| Answer format not followed | Note which model; we adjust the prompt before the real runs |
| Thinking mode leaked | Note which model; we fix its template settings |
| CUDA out of memory | Re-run that model alone: add `"--models", "<name>"` to STEP in runner.py |

Fix, push to GitHub, and re-run the notebook. Re-test a single model with `--models <name>`.

## Step 7: Lock the working setup
Once everything is PASS, copy the transformers/accelerate/huggingface_hub lines
from `outputs/kaggle/smoke/environment.txt` into `requirements.txt` with exact versions (`==`).
Commit. From now on the environment never changes, so no new setup errors.

Setup is complete. Next: Phase 1 (datasets).
