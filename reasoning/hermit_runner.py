# reasoning/hermit_runner.py
"""
HermiT reasoner runner (owlready2-based).

Wraps owlready2's sync_reasoner() and returns a normalized report dict
matching the same schema as Konclude.
"""
import config
from rdflib import Graph
from owlready2 import (
    get_ontology,
    sync_reasoner,
    OwlReadyInconsistentOntologyError,
)
from typing import Dict, Any
from pathlib import Path
import uuid
import time
from reasoning.axiom_extractor import extract_inconsistency_explanation, extract_all_inconsistencies
import owlready2 as _owlready2
_owlready2.JAVA_MEMORY = 1024


def run(
    input_owl: str,
    out_dir: str = None,
) -> Dict[str, Any]:
    """
    Run HermiT reasoner via owlready2 and return a normalized report.

    Args:
        input_owl: Path to input OWL file.
        out_dir:   Output directory for reasoned OWL + logs.
                   Defaults to config.REASONER_OUT_DIR.

    Returns:
        Normalized report dict with schema:
        {
            "is_consistent":       bool,
            "unsat_classes":       list[str],
            "disjoint_violations": list[tuple],
            "raw_logs_path":       str,
            "stats": {
                "reasoner":         "hermit",
                "duration_seconds": float,
                "triple_count":     int,
            }
        }

    Raises:
        FileNotFoundError: If input_owl does not exist.
        RuntimeError:      On unexpected non-recoverable reasoner error.
    """
    start_time = time.perf_counter()
    in_path = Path(input_owl).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input OWL not found: {in_path}")

    if out_dir is None:
        out_dir = str(config.REASONER_OUT_DIR)
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex[:8]
    out_owl = out_dir_path / f"hermit_reasoned_{uid}.owl"
    raw_log_path = out_dir_path / f"hermit_raw_{uid}.log"

    # Load ontology
    print(f"[HermiT] Loading: {in_path}")
    onto = get_ontology(str(in_path)).load()

    is_consistent = True
    unsat_classes = []
    disjoint_violations = []
    log_messages = []

    with onto:
        try:
            sync_reasoner(debug=0)
            log_messages.append("[HermiT] Ontology is CONSISTENT ✓")
            print("[HermiT] Ontology is CONSISTENT ✓")

        except OwlReadyInconsistentOntologyError as exc:
            is_consistent = False
            log_messages.append(f"[HermiT] Ontology is INCONSISTENT — {exc}")
            print(f"[HermiT] Ontology is INCONSISTENT — {exc}")

            # Attempt to extract unsatisfiable classes from the exception message
            # owlready2 doesn't provide structured access to unsat classes,
            # so we parse the exception string
            exc_str = str(exc)
            if "unsatisfiable" in exc_str.lower():
                # Extract class IRIs from exception message if present
                # This is a best-effort parse — format may vary
                import re
                matches = re.findall(r"http[s]?://[^\s]+", exc_str)
                unsat_classes = [m for m in matches if "#" in m or "/" in m]

        # except Exception as exc:
        #     # HermiT sometimes raises non-fatal Java warnings
        #     log_messages.append(f"[HermiT] Non-fatal error (continuing): {exc}")
        #     print(f"[HermiT] Non-fatal error (continuing): {exc}")
        # except Exception as exc:
        #     # This is a real reasoner failure (often JVM memory/pagefile). Don't pretend we know consistency.
        #     log_messages.append(f"[HermiT] REASONER FAILED: {exc}")
        #     print(f"[HermiT] REASONER FAILED: {exc}")

        #     # Write logs (so you can inspect later)
        #     raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        #     raw_log_path.write_text("\n".join(log_messages), encoding="utf-8")

        #     return {
        #         "consistency": False,              # treat as failure / not-consistent for the RL loop
        #         "unsat_classes": [],
        #         "disjoint_violations": [],
        #         "duration_s": time.time() - t0,
        #         "raw_log_path": str(raw_log_path),
        #         "reasoner_error": True,
        #         "reasoner_error_msg": str(exc),
        #     }
        except Exception as exc:
            log_messages.append(f"[HermiT] REASONER FAILED: {exc}")
            print(f"[HermiT] REASONER FAILED: {exc}")

            raw_log_path.parent.mkdir(parents=True, exist_ok=True)
            raw_log_path.write_text("\n".join(log_messages), encoding="utf-8")

            elapsed = time.perf_counter() - start_time

            return {
                "is_consistent": False,              # or None if you want to distinguish “unknown”
                "unsat_classes": [],
                "disjoint_violations": [],
                "raw_logs_path": str(raw_log_path),
                "stats": {
                    "reasoner": "hermit",
                    "duration_seconds": elapsed,
                    "triple_count": 0,
                },
                "reasoner_error": True,
                "reasoner_error_msg": str(exc),
            }

        # --- Extract ALL evidence for inconsistencies ---
    evidence = None
    all_issues = None
    if not is_consistent:
        # Comprehensive scan: find every violation
        all_issues = extract_all_inconsistencies(str(in_path))

        if all_issues and all_issues["total_violations"] > 0:
            evidence = all_issues["primary_evidence"]

            # Populate disjoint_violations from all found violations
            dv_list = [v for v in all_issues["all_violations"]
                       if v["error_type"] in ("disjoint_violation", "transitive_disjoint_violation")]
            if dv_list:
                disjoint_violations = [
                    (v["entity"], v.get("classes", ["", ""])
                     [0], v.get("classes", ["", ""])[1])
                    for v in dv_list
                ]

            # Populate unsat_classes from evidence when available
            for v in all_issues["all_violations"]:
                if v["error_type"] == "unsat_class":
                    iri = v.get("entity_iri") or v.get("entity")
                    if iri and iri not in unsat_classes:
                        unsat_classes.append(iri)

            print(f"\n[HermiT] === INCONSISTENCY REPORT ===")
            print(all_issues["summary_text"])
            print(f"[HermiT] === END REPORT ===\n")
        else:
            # Fall back to single-evidence extraction
            expl = extract_inconsistency_explanation(str(in_path))
            if expl:
                evidence = expl
                print(f"[HermiT] Evidence extracted: type={expl.get('error_type', '?')}, "
                      f"entity={expl.get('entity_label', expl.get('entity', '?'))}")

    # Save reasoned ontology
    print(f"[HermiT] Saving reasoned ontology-> {out_owl}")
    onto.save(file=str(out_owl), format="rdfxml")

    # Count triples in reasoned OWL
    g = Graph()
    g.parse(str(out_owl))
    triple_count = len(g)

    # Save raw logs
    with open(raw_log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_messages))

    elapsed = time.perf_counter() - start_time

    # Build normalized report
    report = {
        "is_consistent":       is_consistent,
        "unsat_classes":       unsat_classes,
        "disjoint_violations": disjoint_violations,
        "raw_logs_path":       str(raw_log_path),
        "stats": {
            "reasoner":         "hermit",
            "duration_seconds": elapsed,
            "triple_count":     triple_count,
        },
    }

    # Include full evidence dict so callers can see the actual cause
    if evidence:
        report["evidence"] = evidence
    # Include comprehensive report
    if all_issues:
        report["all_issues"] = all_issues

    return report
