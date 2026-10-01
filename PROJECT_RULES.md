# Project rules (for the Antigravity agent)

Project: Retrieval-Augmented Small Language Models for Reliable Medical Question Answering.
Goal: keep it simple. Compare 5 models x 3 configs (Baseline, +RAG, +RAG+Abstain) on the key proof-point metrics.

1. All numbers (seeds, k, chunk size, thresholds, model IDs) live in configs/base.yaml. Never hard-code them.
2. Every phase is a script src/phaseXX_*.py: reads only the previous phase's saved output, writes to outputs/.
3. Every phase ends with assertions that fail loudly (row counts, no empty fields, expected columns).
4. Every model answer is saved to JSONL so metrics never need models to be re-run.
5. Long runs must be resumable: skip question IDs already in the output file.
6. Never tune on the test set. Tuning uses the validation set only.
7. Never add torch to requirements.txt (it is preinstalled on Kaggle).
8. Test locally with --tiny (Qwen3-0.6B, CPU, 5 questions) before sending anything to Kaggle.
9. GPU runs happen on Kaggle via kaggle_runner/. Change only the STEP line in runner.py.
