from __future__ import annotations
from typing import Optional, Dict, Any

from rdflib import Graph


def extract_metrics(owl_path: str, report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract metrics from an OWL file + optional reasoner report.

    IMPORTANT:
    This function assumes owl_path is RDFLib-parseable (RDF/XML or TTL).
    Therefore env_repair MUST pass only:
      - the base OWL we control (merged_kg_reasoned.owl)
      - repaired OWL produced by apply_fix (rdflib serialize)
    and NOT Konclude's reasoned output OWL.
    """
    g = Graph()

    # Let RDFLib auto-detect format; if you still hit format issues, we can force format="xml"
    g.parse(owl_path)

    triples = len(g)

    if report is None:
        return {
            "unsat": None,
            "disj": None,
            "parse": 0,
            "proc": 1,   # indicates: no stats available / couldn't parse reasoner info
            "triples": triples,
        }

    return {
        "unsat": len(report.get("unsat_classes", [])),
        "disj": len(report.get("disjoint_violations", [])),
        "parse": report.get("parse_errors", 0),
        "proc": report.get("processing_failures", 0),
        "triples": triples,
    }
