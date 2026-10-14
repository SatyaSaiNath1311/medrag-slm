#!/usr/bin/env python3
"""audit/check_logs_for_secrets.py

Scans every file listed in .gitignore whitelist exceptions (and files within
whitelisted directories) for inadvertently logged API keys, credentials, or tokens:
  - HuggingFace tokens (hf_...)
  - Kaggle API keys and credential blocks
  - OpenAI / Anthropic API keys (sk-...)
  - GitHub Personal Access Tokens (ghp_..., github_pat_...)
  - Generic Bearer tokens, private keys, and credential assignments

Exits:
  0 on clean scan (no secrets detected)
  1 on any secret hit (fails CI / pre-commit audit)

Usage:
  python audit/check_logs_for_secrets.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

SECRET_PATTERNS: Dict[str, re.Pattern] = {
    "HuggingFace Token": re.compile(r"\bhf_[A-Za-z0-9]{34,}\b"),
    "Kaggle API Key": re.compile(r'("key"\s*:\s*"[0-9a-f]{32}"|KAGGLE_KEY\s*=\s*["\'][0-9a-f]{32}["\'])', re.IGNORECASE),
    "Kaggle Credentials Block": re.compile(r'"username"\s*:\s*"[^"]+",\s*"key"\s*:\s*"[^"]+"', re.IGNORECASE),
    "OpenAI API Key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Anthropic API Key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "GitHub PAT": re.compile(r"\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9]{22}_[A-Za-z0-9]{59})\b"),
    "AWS Access Key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Private Key Header": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "Generic Bearer Token": re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{30,}"),
    "API Key Assignment": re.compile(r'(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*["\'][A-Za-z0-9_\-]{20,}["\']', re.IGNORECASE),
}

# Whitelist benign strings that match generic regexes (e.g., standard format strings or public hashes)
BENIGN_WHITELIST: Set[str] = {
    # Add explicit false-positive substrings here if strictly required
}


def parse_gitignore_whitelist(gitignore: Path) -> List[Path]:
    """Parse .gitignore for whitelist exceptions (lines starting with !)."""
    if not gitignore.exists():
        return []

    whitelisted_paths: List[Path] = []
    with open(gitignore, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("!"):
                pattern = line[1:].strip()
                # Resolve relative to repo root
                p = REPO_ROOT / pattern
                whitelisted_paths.append(p)
    return whitelisted_paths


def collect_target_files(whitelist_patterns: List[Path]) -> List[Path]:
    """Collect all existing files that fall under whitelist patterns."""
    collected: Set[Path] = set()

    for p in whitelist_patterns:
        # Check if direct file exists
        if p.is_file():
            collected.add(p.resolve())
        elif p.is_dir():
            for child in p.rglob("*"):
                if child.is_file():
                    collected.add(child.resolve())
        else:
            # Handle potential wildcard patterns in whitelist
            parent = p.parent
            if parent.exists():
                pattern = p.name
                for match in parent.glob(pattern):
                    if match.is_file():
                        collected.add(match.resolve())
                    elif match.is_dir():
                        for child in match.rglob("*"):
                            if child.is_file():
                                collected.add(child.resolve())

    # Sort files deterministically
    return sorted(collected)


def mask_secret(secret_str: str) -> str:
    """Mask secret value for safe terminal printing."""
    s = secret_str.strip()
    if len(s) <= 8:
        return "[REDACTED]"
    return s[:4] + "..." + s[-4:] + " [REDACTED]"


def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan a single file for secret patterns. Returns (line_no, rule_name, masked_snippet)."""
    hits = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                for rule_name, regex in SECRET_PATTERNS.items():
                    matches = regex.findall(line)
                    for m in matches:
                        match_text = m if isinstance(m, str) else m[0]
                        if match_text in BENIGN_WHITELIST:
                            continue
                        hits.append((line_no, rule_name, mask_secret(match_text)))
    except Exception as exc:
        print(f"WARNING: Could not read {file_path.relative_to(REPO_ROOT)}: {exc}")
    return hits


def main():
    print("=" * 80)
    print("Scanning Whitelisted Files & Logs for Inadvertent Secrets")
    print("=" * 80)

    if not GITIGNORE_PATH.exists():
        print(f"ERROR: .gitignore not found at {GITIGNORE_PATH}")
        sys.exit(1)

    whitelist_patterns = parse_gitignore_whitelist(GITIGNORE_PATH)
    print(f"Found {len(whitelist_patterns)} whitelist rule(s) in .gitignore:")
    for pat in whitelist_patterns:
        try:
            rel = pat.relative_to(REPO_ROOT)
        except ValueError:
            rel = pat
        print(f"  !{rel}")

    target_files = collect_target_files(whitelist_patterns)
    print(f"\nExpanded to {len(target_files)} target file(s) on disk to audit.")
    print("-" * 80)

    total_hits = 0
    scanned_count = 0

    for file_path in target_files:
        rel_path = file_path.relative_to(REPO_ROOT).as_posix()
        hits = scan_file(file_path)
        scanned_count += 1
        if hits:
            total_hits += len(hits)
            print(f"\n[FAIL] Potential secret detected in: {rel_path}")
            for line_no, rule, masked in hits:
                print(f"  Line {line_no:5d} | {rule:<25} | Match: {masked}")

    print("\n" + "=" * 80)
    if total_hits > 0:
        print(f"[SECURITY ALERT] {total_hits} potential secret pattern match(es) detected across {scanned_count} files!")
        print("Scrub or redact the tokens before committing or pushing git artifacts.")
        print("=" * 80)
        sys.exit(1)
    else:
        print(f"[PASS] Audited {scanned_count} whitelisted file(s). No secrets or credentials detected.")
        print("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
