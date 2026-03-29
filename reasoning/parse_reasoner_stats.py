"""
reasoning/parse_reasoner_stats.py

Robust Konclude log/stats parser.

Parses either:
- a stats file created by --write-statistics
- OR fallback text (stdout/stderr) when the file is missing

Extracts:
  - consistency (True/False)   # NEVER None — resolved from evidence
  - parse_errors count
  - processing_failures count
  - unsat_classes (if IRIs are reported)
  - unsat_count (if only counts are reported)
  - disjoint_violations (pairs if reported)
  - raw_text (full log for debugging)

CHANGE from original:
  - consistency is now always bool (True/False), never None
  - Logic: explicit "inconsistent"-> False
           unsat_count > 0 or violations-> False
           otherwise-> True (assume consistent if no errors found)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# --- regexes ---
RE_UNSAT_1 = re.compile(r"UnsatisfiableClass\S*[:\s]+([^\s]+)")
RE_UNSAT_2 = re.compile(r"Class\s+([^\s]+)\s+is\s+unsatisfiable", re.IGNORECASE)
RE_UNSATCOUNT = re.compile(r"UnsatisfiableClasses\s*=\s*(\d+)", re.IGNORECASE)

RE_DISJ_1 = re.compile(r"DisjointViolation\S*[:\s]+([^\s]+)\s+([^\s]+)")
RE_DISJ_2 = re.compile(r"Disjoint(?:Class)?\s*[:\-]?\s*([^\s]+)\s+(?:vs|and)\s+([^\s]+)", re.IGNORECASE)

RE_INCONSISTENT = re.compile(r"is\s+inconsistent\.", re.IGNORECASE)

RE_PARSE_ERR = re.compile(
    r"OWL2/XML Ontology node not found|Couldn't match parameters|extract minimal required|parse error|XML error",
    re.IGNORECASE,
)
RE_PROC_FAIL = re.compile(
    r"processing step failed|fatal|exception|stack trace|cannot open|no such file|unknown option|error:",
    re.IGNORECASE,
)

def _unique(items: List[str]) -> List[str]:
    # preserve uniqueness without assuming order matters too much
    return list(dict.fromkeys(items))

def _extract(text: str) -> Dict[str, Any]:
    unsat_classes: List[str] = []
    disj: List[Tuple[str, str]] = []

    # Unsat class IRIs (if present)
    unsat_classes += RE_UNSAT_1.findall(text)
    unsat_classes += RE_UNSAT_2.findall(text)
    unsat_classes = _unique([u.strip() for u in unsat_classes if u.strip()])

    # Disjoint violations (pairs)
    disj += RE_DISJ_1.findall(text)
    disj += RE_DISJ_2.findall(text)
    disj = [(a.strip(), b.strip()) for (a, b) in disj if a.strip() and b.strip()]

    # Unsat count (if present)
    unsat_count = None
    mcount = RE_UNSATCOUNT.search(text)
    if mcount:
        try:
            unsat_count = int(mcount.group(1))
        except ValueError:
            unsat_count = None

    # parse/processing errors counts
    parse_errs = len(RE_PARSE_ERR.findall(text))
    proc_fails = len(RE_PROC_FAIL.findall(text))

    # --- Consistency resolution (NEW LOGIC) ---
    # 1. Explicit "is inconsistent"-> False
    if RE_INCONSISTENT.search(text):
        consistency = False
    # 2. unsat_count > 0 or violations exist-> False
    elif (unsat_count is not None and unsat_count > 0) or len(unsat_classes) > 0 or len(disj) > 0:
        consistency = False
    # 3. Otherwise-> assume consistent (True)
    else:
        consistency = True

    return {
        "unsat_classes": unsat_classes,
        "unsat_count": unsat_count,
        "disjoint_violations": disj,
        "consistency": consistency,
        "parse_errors": parse_errs,
        "processing_failures": proc_fails,
        "raw_text": text,
    }

def parse_stats_file(
    stats_path: Optional[str],
    fallback_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse reasoner stats.

    - If stats_path exists-> parse it
    - Else if fallback_text given-> parse it
    - Else-> return a truthful report indicating missing statistics

    Returns consistency as bool (True/False), never None.
    """
    if stats_path:
        p = Path(stats_path)
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="ignore")
            return _extract(text)

    if fallback_text:
        print("[parse_reasoner_stats] WARNING: stats file missing, parsing fallback text (stdout/stderr).")
        return _extract(fallback_text)


    # Truthful "no stats available" report (not fake)
    # consistency=False because we cannot confirm it's consistent without stats
    return {
        "unsat_classes": [],
        "unsat_count": None,
        "disjoint_violations": [],
        "consistency": False,  # conservative: cannot confirm consistency
        "parse_errors": 0,
        "processing_failures": 1,  # indicates: could not obtain reasoner stats output
        "raw_text": "",
    }