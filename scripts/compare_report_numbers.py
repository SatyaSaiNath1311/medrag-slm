#!/usr/bin/env python3
"""scripts/compare_report_numbers.py — Compare numeric tokens between two markdown reports.

Given an old and a new report path, extracts every numeric token (ignoring
path and provenance lines) and exits non-zero if any number other than
the intentionally changed p-value formatting (<1e-4 vs 0.0000) differs.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Patterns identifying lines to ignore (provenance, file paths, hashes, etc.)
IGNORE_PATTERNS = [
    re.compile(r"^\s*-\s*\*\*Git Commit:\*\*", re.IGNORECASE),
    re.compile(r"^\s*-\s*\*\*Rule Document Hash.*:\*\*", re.IGNORECASE),
    re.compile(r"^\s*-\s*\*\*Reference File:\*\*", re.IGNORECASE),
    re.compile(r"^\s*-\s*\*\*Hours Elapsed:\*\*", re.IGNORECASE),
    re.compile(r"^\s*-\s*\*\*Completed Flag:\*\*", re.IGNORECASE),
    re.compile(r"^\s*-\s*\*\*Timestamp.*:\*\*", re.IGNORECASE),
    re.compile(r"^\s*-\s*`[a-zA-Z0-9_-]+`:\s*`[a-f0-9]{40,64}`", re.IGNORECASE),
    re.compile(r"!\[.*?\]\(.*?\)", re.IGNORECASE),  # image links
    re.compile(r"/Users/|/home/|outputs/|\.jsonl|\.png|\.svg|\.md|\.py", re.IGNORECASE),
]

# Regex to find numeric tokens, including percentages, signs, scientific notation, and <1e-4
# Matches tokens like: 100, +88.0%, -94.0%, 0.0000, <1e-4, 1e-4, 1.9599, etc.
NUM_TOKEN_RE = re.compile(r"(<1e-4|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?%?)")


def is_ignored_line(line: str) -> bool:
    """Determine if a line should be excluded from numeric comparison."""
    for pattern in IGNORE_PATTERNS:
        if pattern.search(line):
            return True
    return False


def extract_numbers(file_path: Path) -> List[Tuple[int, str, str]]:
    """Extract (line_num, token, raw_line) for all non-ignored numeric tokens."""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if is_ignored_line(line):
                continue
            # Also ignore section headers for execution provenance
            if re.search(r"## \d+\.\s*(Execution Provenance|Provenance)", line, re.IGNORECASE):
                continue
            tokens = NUM_TOKEN_RE.findall(line)
            for token in tokens:
                results.append((idx, token.strip(), line.strip()))
    return results


def normalize_token(t: str) -> Tuple[str, float]:
    """Return a representation for comparison: category ('val' or 'lt_1e-4') and numeric value."""
    clean = t.replace("%", "").replace("+", "").strip()
    if clean == "<1e-4":
        return ("lt_1e-4", 1e-4)
    try:
        val = float(clean)
        return ("val", val)
    except ValueError:
        return ("str", 0.0)


def tokens_match(tok_old: str, tok_new: str) -> bool:
    """Check if tokens match or represent allowed p-value change (<1e-4 vs 0.0000)."""
    if tok_old == tok_new:
        return True

    cat_old, val_old = normalize_token(tok_old)
    cat_new, val_new = normalize_token(tok_new)

    # Allowed intentional change: 0.0000 in old report changed to <1e-4 in new report
    if (tok_old == "0.0000" or val_old == 0.0) and (tok_new == "<1e-4" or cat_new == "lt_1e-4"):
        return True
    if (tok_new == "0.0000" or val_new == 0.0) and (tok_old == "<1e-4" or cat_old == "lt_1e-4"):
        return True

    if cat_old == "val" and cat_new == "val":
        return abs(val_old - val_new) < 1e-7

    return False


def compare_reports(old_path: Path, new_path: Path) -> bool:
    """Compare two reports, printing differences and returning True if passed."""
    old_nums = extract_numbers(old_path)
    new_nums = extract_numbers(new_path)

    if len(old_nums) != len(new_nums):
        print(f"FAIL: Token count mismatch between {old_path} ({len(old_nums)}) and {new_path} ({len(new_nums)})")
        min_len = min(len(old_nums), len(new_nums))
        for i in range(min_len):
            o_line, o_tok, o_text = old_nums[i]
            n_line, n_tok, n_text = new_nums[i]
            if not tokens_match(o_tok, n_tok):
                print(f"  First mismatch at token #{i+1}:")
                print(f"    Old (L{o_line}): '{o_tok}' in `{o_text}`")
                print(f"    New (L{n_line}): '{n_tok}' in `{n_text}`")
                break
        return False

    mismatches = []
    for i, ((o_line, o_tok, o_text), (n_line, n_tok, n_text)) in enumerate(zip(old_nums, new_nums)):
        if not tokens_match(o_tok, n_tok):
            mismatches.append((i + 1, o_line, o_tok, o_text, n_line, n_tok, n_text))

    if mismatches:
        print(f"FAIL: Found {len(mismatches)} numeric mismatches between {old_path} and {new_path}:")
        for idx, o_line, o_tok, o_text, n_line, n_tok, n_text in mismatches[:10]:
            print(f"  Token #{idx}:")
            print(f"    Old (L{o_line}): '{o_tok}' in `{o_text}`")
            print(f"    New (L{n_line}): '{n_tok}' in `{n_text}`")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more.")
        return False

    print(f"PASS: All {len(old_nums)} numeric tokens match between {old_path.name} and {new_path.name}.")
    return True


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <old_report.md> <new_report.md>")
        sys.exit(2)

    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])

    if not old_path.exists():
        print(f"Error: old report does not exist: {old_path}")
        sys.exit(1)
    if not new_path.exists():
        print(f"Error: new report does not exist: {new_path}")
        sys.exit(1)

    ok = compare_reports(old_path, new_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
