#!/usr/bin/env bash
# scripts/verify_all.sh — Master Verification Gate for Phase 1 MedRAG SLM.
#
# Runs in under two minutes with no GPU and no network:
#   1. pytest (explicit timeout-safe test suite)
#   2. Frozen data cryptographic SHA-256 audit
#   3. Outputs & Sweeps integrity audit (IDs, uniqueness, counts, manifests, no hf_ tokens, rule hashes)
#   4. Paper Facts ledger check (python3 -m src.build_paper_facts --check)
#   5. Consolidated Results tables check (python3 -m src.final_results_v3 --check)
#   6. Kaggle runner default mode flags check (text parse, all V3 flags False)
#   7. Clean working tree check under outputs/ and docs/
#
# Prints elapsed time of each gate, exactly one PASS/FAIL line per check, and a final summary.
# Exits non-zero on any failure.

set -u

TOTAL_CHECKS=7
PASSED_CHECKS=0
FAILED_CHECKS=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GATE_TOTAL_START=$(date +%s)

report_pass() {
    local check_name="$1"
    local elapsed="${2:-}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    if [ -n "$elapsed" ]; then
        echo "PASS: $check_name (${elapsed}s)"
    else
        echo "PASS: $check_name"
    fi
}

report_fail() {
    local check_name="$1"
    local reason="${2:-}"
    local elapsed="${3:-}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    local time_suffix=""
    if [ -n "$elapsed" ]; then
        time_suffix=" (${elapsed}s)"
    fi
    if [ -n "$reason" ]; then
        echo "FAIL: $check_name ($reason)$time_suffix"
    else
        echo "FAIL: $check_name$time_suffix"
    fi
}

echo "================================================================================"
echo "PHASE 1 MEDRAG-SLM: MASTER VERIFICATION GATE"
echo "================================================================================"

# Check 1: pytest (explicit timeout-safe test list; -m "not slow" by default, --full for all tests)
FULL_RUN=false
for arg in "$@"; do
    if [ "$arg" = "--full" ]; then
        FULL_RUN=true
    fi
done

PYTEST_ARGS=("-q")
if [ "$FULL_RUN" = false ]; then
    PYTEST_ARGS+=("-m" "not slow")
fi

T0=$(date +%s)
TEST_FILES=(
    "tests/test_answer_validity.py"
    "tests/test_budget_sweep.py"
    "tests/test_chat_app.py"
    "tests/test_context_prompt.py"
    "tests/test_final_results.py"
    "tests/test_gemma_prompt_sens.py"
    "tests/test_idk.py"
    "tests/test_pairs.py"
    "tests/test_parsing.py"
    "tests/test_question_types.py"
    "tests/test_reasoning.py"
    "tests/test_sharding.py"
    "tests/test_unanswerable_v2.py"
    "tests/test_v3_datasets.py"
)
if pytest "${PYTEST_ARGS[@]}" "${TEST_FILES[@]}" > /dev/null 2>&1; then
    T1=$(date +%s)
    report_pass "All pytest unit tests passed" "$((T1 - T0))"
else
    T1=$(date +%s)
    report_fail "pytest unit tests" "one or more tests failed" "$((T1 - T0))"
fi

# Check 2: Frozen-Data SHA-256 Audit
T0=$(date +%s)
check_sha() {
    local file="$1"
    local expected="$2"
    if [ ! -f "$file" ]; then
        return 1
    fi
    local actual
    actual=$(python3 -c "import hashlib; print(hashlib.sha256(open('$file', 'rb').read()).hexdigest())")
    [ "$actual" = "$expected" ]
}

SHA_PASS=true
check_sha "outputs/kaggle_build/work/phase1/test.jsonl" "fe7fe3fb88c77c4ffc6574c8716ec0b3f4787fbfaf2c22e2eb5741ac040cc34a" || SHA_PASS=false
check_sha "data/unanswerable_v2/test.jsonl" "a0e45676dced360b2ac64e8fc33dbb49782c4fd9022e93190e633c9d66396078" || SHA_PASS=false
check_sha "data/unanswerable_v2/val.jsonl" "f5d779faa76e1c6a85f59718be72691a6d676b374070f346e4b332b70ba7b3a4" || SHA_PASS=false

T1=$(date +%s)
if [ "$SHA_PASS" = true ]; then
    report_pass "Frozen benchmark datasets SHA-256 audit verified" "$((T1 - T0))"
else
    report_fail "Frozen benchmark datasets SHA-256 audit" "hash mismatch on test or val partition" "$((T1 - T0))"
fi

# Check 3: Outputs & Sweeps Integrity Audit
T0=$(date +%s)
if python3 scripts/audit_outputs.py > /dev/null 2>&1; then
    T1=$(date +%s)
    report_pass "Outputs and sweeps integrity audit (IDs, row counts, manifests, no hf_ tokens, rule hashes)" "$((T1 - T0))"
else
    T1=$(date +%s)
    report_fail "Outputs and sweeps integrity audit" "integrity defect found in outputs/v3 or outputs/sweeps" "$((T1 - T0))"
fi

# Check 4: Paper Facts Ledger Consistency (--check)
T0=$(date +%s)
if python3 -m src.build_paper_facts --check > /dev/null 2>&1; then
    T1=$(date +%s)
    report_pass "docs/PAPER_FACTS.md is strictly identical to master fact ledger" "$((T1 - T0))"
else
    T1=$(date +%s)
    report_fail "docs/PAPER_FACTS.md consistency check" "regenerate via python3 -m src.build_paper_facts" "$((T1 - T0))"
fi

# Check 5: Consolidated Final Results Tables Consistency (--check)
T0=$(date +%s)
if python3 -m src.final_results_v3 --check > /dev/null 2>&1; then
    T1=$(date +%s)
    report_pass "docs/tables/final_results_v3 (md, csv, tex) match committed outputs" "$((T1 - T0))"
else
    T1=$(date +%s)
    report_fail "docs/tables/final_results_v3 consistency check" "regenerate via python3 -m src.final_results_v3" "$((T1 - T0))"
fi

# Check 6: All V3 Runner Default Flags are False (parsed as text, never imported)
T0=$(date +%s)
RUNNER_FLAGS_PASS=$(python3 - << 'EOF'
import ast
import sys

runner_path = "kaggle_runner/runner.py"
with open(runner_path, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=runner_path)

v3_flags = [
    "V3_RUN", "V3_PILOT", "V3_CONFIRM_TEST",
    "V3_TRUNC_SENSITIVITY", "V3_BUDGET_SWEEP",
    "V3_GEMMA_PROMPT_SENS", "QWEN3_4B_THINKING_MEDQA"
]

violations = []
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in v3_flags:
                if isinstance(node.value, ast.Constant) and node.value.value is not False:
                    violations.append(f"{target.id}={node.value.value}")

if violations:
    print(f"Violations: {violations}")
    sys.exit(1)
print("OK")
sys.exit(0)
EOF
)

T1=$(date +%s)
if [ "$RUNNER_FLAGS_PASS" = "OK" ]; then
    report_pass "All V3 mode flags in kaggle_runner/runner.py are set to False" "$((T1 - T0))"
else
    report_fail "Kaggle runner flags audit" "$RUNNER_FLAGS_PASS" "$((T1 - T0))"
fi

# Check 7: Git Working Tree Clean under outputs/ and docs/
T0=$(date +%s)
STATUS_OUTPUT=$(git status --porcelain outputs/ docs/)
T1=$(date +%s)
if [ -z "$STATUS_OUTPUT" ]; then
    report_pass "git working tree is clean under outputs/ and docs/" "$((T1 - T0))"
else
    report_fail "git working tree clean check" "uncommitted changes under outputs/ or docs/" "$((T1 - T0))"
fi

# Final Summary
GATE_TOTAL_END=$(date +%s)
TOTAL_TIME=$((GATE_TOTAL_END - GATE_TOTAL_START))
echo "================================================================================"
if [ "$FAILED_CHECKS" -eq 0 ]; then
    echo "VERIFICATION SUMMARY: All $PASSED_CHECKS checks PASSED in ${TOTAL_TIME}s."
    echo "================================================================================"
    exit 0
else
    echo "VERIFICATION SUMMARY: $FAILED_CHECKS of $TOTAL_CHECKS checks FAILED in ${TOTAL_TIME}s."
    echo "================================================================================"
    exit 1
fi
