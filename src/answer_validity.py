"""src/answer_validity.py — Unified Answer Validity and Accuracy Evaluation.

Canonical definitions:
  1. Strict Correctness (Primary):
     A row counts as correct if and only if:
       - NOT row.get("truncated", False)
       - AND bool(row.get("parsed", True))
       - AND bool(row.get("correct", False))
     Any truncated generation or unparsed response counts strictly as incorrect (0).

  2. With-Fallback Correctness (Secondary, labelled):
     Considers raw stored correctness (including logprob or heuristic fallback answers):
       - bool(row.get("correct", False))

  3. Truncation Detection:
     bool(row.get("truncated", False))
"""

from typing import Any, Dict, List, Optional


def is_strict_correct(row: Dict[str, Any]) -> bool:
    """Return True iff the row was not truncated, was successfully parsed, and was correct."""
    is_trunc = bool(row.get("truncated", False))
    is_prsd = bool(row.get("parsed", True))
    is_corr = bool(row.get("correct", False))
    return (not is_trunc) and is_prsd and is_corr


def is_fallback_correct(row: Dict[str, Any]) -> bool:
    """Return True if row has correct=True, including fallback/unparsed letters."""
    return bool(row.get("correct", False))


def is_truncated(row: Dict[str, Any]) -> bool:
    """Return True iff the row generation was truncated."""
    return bool(row.get("truncated", False))


def is_parsed(row: Dict[str, Any]) -> bool:
    """Return True iff the row response was successfully parsed into an option."""
    return bool(row.get("parsed", True))


def strict_correct_list(rows: List[Dict[str, Any]]) -> List[int]:
    """Return list of 1s and 0s indicating strict correctness."""
    return [1 if is_strict_correct(r) else 0 for r in rows]


def fallback_correct_list(rows: List[Dict[str, Any]]) -> List[int]:
    """Return list of 1s and 0s indicating with-fallback correctness."""
    return [1 if is_fallback_correct(r) else 0 for r in rows]


def strict_accuracy(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Compute strict accuracy over rows."""
    if not rows:
        return None
    return round(sum(1 for r in rows if is_strict_correct(r)) / len(rows), 4)


def fallback_accuracy(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Compute with-fallback accuracy over rows."""
    if not rows:
        return None
    return round(sum(1 for r in rows if is_fallback_correct(r)) / len(rows), 4)


def truncation_rate(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Compute truncation percentage/rate over the exact same rows as accuracy."""
    if not rows:
        return None
    return round(sum(1 for r in rows if is_truncated(r)) / len(rows), 4)


def parse_rate(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Compute parse rate over the exact same rows."""
    if not rows:
        return None
    return round(sum(1 for r in rows if is_parsed(r)) / len(rows), 4)
