#!/bin/bash
# Run every phase on a tiny scale on your laptop (CPU). Proves the code works before using Kaggle GPU time.
# Usage: bash scripts/local_tiny.sh            (all phases)
#        bash scripts/local_tiny.sh 6          (start from phase 6: retrieval, rerank, models)
set -e
W=outputs/tiny
ALL="phase01_datasets phase02_corpus phase03_index phase06_retrieve phase07_rerank qa_pipeline"
case "$1" in
  6) STEPS="phase06_retrieve phase07_rerank qa_pipeline" ;;
  qa) STEPS="qa_pipeline" ;;
  *) STEPS="$ALL" ;;
esac
PYTHON="python"
if ! command -v python &> /dev/null; then
  if [ -f "./.venv/bin/python" ]; then
    PYTHON="./.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

for p in $STEPS; do
  echo; echo "########## $p ##########"
  EXTRA=""; [ "$p" = "qa_pipeline" ] && EXTRA="--strict"
  $PYTHON -m src.$p --config configs/base.yaml --work $W --tiny $EXTRA
done
echo; echo "ALL TINY PHASES PASSED"
