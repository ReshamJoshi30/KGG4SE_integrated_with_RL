# reasoning/axiom_extractor.py
"""
Extract evidence for WHY an ontology is inconsistent.

Two methods:
  1. Parse HermiT exception message  (fast, often empty)
  2. Heuristic graph search          (robust, ordered by specificity)

Heuristic checks in priority order:
  A. Disjoint violations       — individual typed into two disjoint classes
  B. Domain violations         — subject of property not typed as domain class
  C. Range violations          — object of property not typed as range class
  D. Class-as-subject          — an owl:Class URI used as the subject of a
                                 non-meta property assertion (most common cause
                                 of "Consistent:False / Unsat:0")
  E. Functional property       — property with cardinality 1 has two fillers
  F. Property type mismatch    — owl:ObjectProperty used with a Literal value

Bug fixes vs original:
  - _find_property_violations: namespace skip-list was too broad and silently
    discarded ALL custom properties (anything with "owl" in the URI string,
    which matched e.g. "http://cpsagila.cs.uni-kl.de/.../GENIALOwl#...").
    Fixed to only skip the three canonical W3C namespaces.
  - _find_suspicious_individuals: threshold of >5 triples fired on every
    normal individual and returned a useless "complex_individual" that mapped
    to a no_op repair, blocking the RL reward signal entirely.
    Removed; replaced by targeted checks D and E above.
  - Added domain/range violation detection (checks C and B).
  - Added class-as-subject detection (check D).
  - Added functional property violation detection (check E).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import owlready2
from owlready2 import OwlReadyInconsistentOntologyError
from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection
from rdflib.namespace import XSD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# W3C meta-namespaces — URIs from these are infrastructure, not domain data
# ---------------------------------------------------------------------------
_META_NS = (
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
)


def _is_meta(uri: str) -> bool:
    """Return True iff *uri* belongs to an RDF/OWL/RDFS meta-namespace."""
    return any(uri.startswith(ns) for ns in _META_NS)


def _local(uri: str) -> str:
    """Extract local name from a URI string."""
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Base ontology namespace — types from outside this NS on minted individuals are suspicious
_BASE_ONTO_NS = "http://cpsagila.cs.uni-kl.de/GENIALOnt"
_IND_NS = "http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/"


def _find_external_type_violations(g: Graph, onto_path=None) -> dict | None:
    """
    Detect minted individuals (in the IND namespace) that are typed as
    classes from EXTERNAL namespaces (outside the base ontology NS).

    External-namespace types cause hidden inconsistencies because HermiT
    follows the external class's subclass/disjoint axioms, which can conflict
    with the base ontology in ways that produce Consistent:False with unsat=0.

    Also detects individuals typed as classes from w3id.org/gbo/ or sosa/
    namespaces that are known to conflict with GENIALOnt structure.
    """
    # Minted individual namespace
    minted_inds = {str(s) for s in g.subjects(RDF.type, None)
                   if isinstance(s, URIRef) and str(s).startswith(_IND_NS)}

    # Suspicious external namespaces (not the base ontology)
    _EXTERNAL_NS = (
        "http://w3id.org/gbo/",
        "http://www.w3.org/ns/sosa/",
        "http://www.ontology-of-units-of-measure.org/",
        "http://purl.obolibrary.org/obo/",
    )

    violations = []
    for ind, _, type_uri in g.triples((None, RDF.type, None)):
        if not isinstance(ind, URIRef) or not isinstance(type_uri, URIRef):
            continue
        ind_str = str(ind)
        t_str = str(type_uri)

        # Skip meta classes
        if "w3.org/2002/07/owl" in t_str or "w3.org/1999/02/22-rdf" in t_str:
            continue
        # Skip types from the base ontology — those are safe
        if t_str.startswith(_BASE_ONTO_NS):
            continue
        # Flag types from known-problematic external namespaces
        if any(t_str.startswith(ns) for ns in _EXTERNAL_NS):
            ind_name = ind_str.split("/")[-1].split("#")[-1]
            type_name = t_str.split("#")[-1].split("/")[-1]
            ns_root = "/".join(t_str.split("/")[:5])
            violations.append((ind_str, ind_name, t_str, type_name, ns_root))

    if not violations:
        return None

    # Sort by individual name for determinism
    violations.sort(key=lambda x: x[1])
    ind_str, ind_name, type_str, type_name, ns = violations[0]

    # Count how many triples reference this individual
    ind_uri = URIRef(ind_str)
    n_triples = sum(1 for _ in g.triples((ind_uri, None, None)))
    n_triples += sum(1 for _ in g.triples((None, None, ind_uri)))

    logger.debug(
        "[axiom_extractor] External type: <%s> typed as <%s> (ns: %s), "
        "%d triples",
        ind_name, type_name, ns, n_triples,
    )

    return {
        "error_type": "external_type_violation",
        "entity": ind_str,
        "entity_label": ind_name,
        "evidence": [
            f"Individual <{ind_name}> is typed as <{type_name}> from external namespace {ns}",
            f"External classes import axioms that conflict with base ontology",
            f"Fix: remove the external rdf:type assertion or drop the individual",
            f"Total violations found: {len(violations)}",
        ],
        # (ind_uri, type_uri)
        "external_types": [(v[0], v[2]) for v in violations],
        "actions": [
            {
                "type": "remove_external_type",
                "entity": ind_str,
                "type_uri": type_str,
                "risk": "low",
                "risk_penalty": 0.5,
                "description": f"Remove external type <{type_name}> from <{ind_name}>",
            },
            {
                "type": "drop_entity",
                "entity": ind_str,
                "risk": "medium",
                "risk_penalty": 2.0,
                "description": f"Drop individual <{ind_name}> entirely ({n_triples} triples)",
            },
        ],
    }


def extract_inconsistency_explanation(owl_path: str) -> Optional[Dict[str, Any]]:
    """
    Return the first specific explanation found for why *owl_path* is
    inconsistent, or a generic fallback dict if nothing is found.

    Returns:
        Dict with keys: error_type, entity, evidence (list[str])
        Never returns None — callers can always read error_type.
    """
    # --- Method 1: parse HermiT exception message ---
    evidence = _try_exception_parsing(owl_path)
    if evidence and evidence.get("error_type") not in (
        "inconsistency_from_exception",
        "unknown_inconsistency",
    ):
        print(
            f"[axiom_extractor] Found via exception parsing: {evidence['error_type']}")
        return evidence

    print("[axiom_extractor] Method 1 too generic, trying Method 2 (heuristic search)...")

    # --- Method 2: heuristic graph scan ---
    evidence = _heuristic_search(owl_path)
    if evidence:
        print(
            f"[axiom_extractor] Found via heuristic search: {evidence['error_type']}")
        return evidence

    # Method 6: external namespace type violations (hidden inconsistencies)
    print("[axiom_extractor] Trying Method 6 (external namespace types)...")
    try:
        g6 = Graph()
        g6.parse(str(owl_path))
        ext_evidence = _find_external_type_violations(g6)
        if ext_evidence:
            print(f"[axiom_extractor] Found via Method 6: external_type_violation")
            return ext_evidence
    except Exception as _e6:
        pass

    print("[axiom_extractor] WARNING: No specific evidence found, returning generic inconsistency")
    return {
        "error_type": "unknown_inconsistency",
        "entity": "Unknown",
        "entity_label": "Unknown",
        "evidence": [
            "Ontology is inconsistent but cause unknown — "
            "may require manual inspection"
        ],
        "actions": [
            {"type": "drop_entity", "entity": "Unknown",
                "risk": "high", "risk_penalty": 5.0},
            {"type": "flag_for_review", "entity": "Unknown",
                "risk": "none", "risk_penalty": 0.0},
        ]
    }


# ---------------------------------------------------------------------------
# Method 1 — HermiT exception parsing
# ---------------------------------------------------------------------------

def _try_exception_parsing(owl_path: str) -> Optional[Dict[str, Any]]:
    """Trigger HermiT and try to extract a diagnosis from the exception."""
    try:
        onto = owlready2.get_ontology(f"file://{owl_path}").load()
        with onto:
            owlready2.sync_reasoner_hermit(onto, debug=1)
        return None  # consistent — shouldn't happen here

    except OwlReadyInconsistentOntologyError as exc:
        msg = str(exc)
        print(f"[axiom_extractor] HermiT exception: {msg[:200]}")

        if "disjoint" in msg.lower():
            return {
                "error_type": "disjoint_violation",
                "entity": "extracted_from_error",
                "evidence": [f"Inconsistency detected: {msg[:150]}"],
            }
        if "restriction" in msg.lower() or "cardinality" in msg.lower():
            return {
                "error_type": "property_restriction_violation",
                "entity": "extracted_from_error",
                "evidence": [f"Restriction violation: {msg[:150]}"],
            }
        # Generic — will fall through to Method 2
        return {
            "error_type": "inconsistency_from_exception",
            "entity": "Unknown",
            "evidence": [f"HermiT error: {msg[:200]}"],
        }

    except Exception as exc:
        print(f"[axiom_extractor] Exception parsing failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Method 2 — heuristic graph scan
# ---------------------------------------------------------------------------

def _heuristic_search(owl_path: str) -> Optional[Dict[str, Any]]:
    """
    Scan the merged graph with rdflib and return the first violation found.

    Checks are ordered from most-specific (cheapest to confirm) to least.
    """
    try:
        g = Graph()
        g.parse(owl_path)
    except Exception as exc:
        print(f"[axiom_extractor] Could not parse graph: {exc}")
        return None

    # Pre-compute shared sets once
    all_classes: set[str] = {
        str(s) for s in g.subjects(RDF.type, OWL.Class)
        if isinstance(s, URIRef)
    }
    all_individuals: dict[str, list[str]
                          ] = _collect_individuals(g, all_classes)

    # A — disjoint violations (most common, highest reward)
    result = _find_disjoint_violations(g, all_individuals)
    if result:
        return result

    # A2 — transitive disjoint violations (via equivalentClass/intersectionOf)
    result = _find_transitive_disjoint_violations(g, all_individuals)
    if result:
        return result

    # B — domain violations
    result = _find_domain_violations(g, all_individuals)
    if result:
        return result

    # C — range violations
    result = _find_range_violations(g, all_individuals)
    if result:
        return result

    # D — class URI used as subject of a property assertion
    result = _find_class_as_subject(g, all_classes)
    if result:
        return result

    # E — functional property used twice on same subject
    result = _find_functional_violations(g)
    if result:
        return result

    # F — object property used with a Literal
    result = _find_property_type_mismatch(g)
    if result:
        return result

    return None


# ---------------------------------------------------------------------------
# Comprehensive scan — find ALL violations (not just the first)
# ---------------------------------------------------------------------------

def _find_all_disjoint_violations(
    g: Graph,
    all_individuals: dict[str, list[str]],
) -> List[Dict[str, Any]]:
    """Find ALL individuals typed into two disjoint classes."""
    disjoint_pairs: set[tuple[str, str]] = set()

    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            disjoint_pairs.add((str(s), str(o)))
            disjoint_pairs.add((str(o), str(s)))

    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for _, _, head in g.triples((adc, OWL.members, None)):
            try:
                members = [m for m in Collection(
                    g, head) if isinstance(m, URIRef)]
            except Exception:
                members = []
            for i, m1 in enumerate(members):
                for m2 in members[i + 1:]:
                    disjoint_pairs.add((str(m1), str(m2)))
                    disjoint_pairs.add((str(m2), str(m1)))

    if not disjoint_pairs:
        return []

    results = []
    for ind_iri, classes in all_individuals.items():
        if len(classes) < 2:
            continue
        for i, c1 in enumerate(classes):
            for c2 in classes[i + 1:]:
                if (c1, c2) in disjoint_pairs or (c2, c1) in disjoint_pairs:
                    ind_name = _local(ind_iri)
                    c1_name = _local(c1)
                    c2_name = _local(c2)
                    results.append({
                        "error_type":   "disjoint_violation",
                        "entity":       ind_iri,
                        "entity_label": ind_name,
                        "classes":      [c1, c2],
                        "class_labels": [c1_name, c2_name],
                        "evidence": [
                            f"Individual {ind_name} is typed as both "
                            f"{c1_name} and {c2_name}",
                            f"{c1_name} and {c2_name} are declared disjoint",
                            f"ClassAssertion({c1_name}, {ind_name})",
                            f"ClassAssertion({c2_name}, {ind_name})",
                            f"DisjointClasses({c1_name}, {c2_name})",
                        ],
                    })
    return results


def _find_all_transitive_disjoint_violations(
    g: Graph,
    all_individuals: dict[str, list[str]],
) -> List[Dict[str, Any]]:
    """Find ALL transitive disjoint violations via class hierarchy."""
    sub_of = _build_enhanced_superclass_map(g)

    disjoint_pairs: set[tuple[str, str]] = set()
    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            disjoint_pairs.add((str(s), str(o)))
            disjoint_pairs.add((str(o), str(s)))
    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for _, _, head in g.triples((adc, OWL.members, None)):
            try:
                members = [m for m in Collection(
                    g, head) if isinstance(m, URIRef)]
            except Exception:
                members = []
            for i, m1 in enumerate(members):
                for m2 in members[i + 1:]:
                    disjoint_pairs.add((str(m1), str(m2)))
                    disjoint_pairs.add((str(m2), str(m1)))

    if not disjoint_pairs:
        return []

    # Track already-found direct violations to avoid duplicates
    results = []
    seen = set()
    for ind_iri, classes in all_individuals.items():
        if len(classes) < 2:
            continue
        type_ancestors: dict[str, set[str]] = {}
        for c in classes:
            type_ancestors[c] = _all_ancestors(c, sub_of) | {c}

        class_list = list(type_ancestors.keys())
        for i, t1 in enumerate(class_list):
            for t2 in class_list[i + 1:]:
                for a1 in type_ancestors[t1]:
                    for a2 in type_ancestors[t2]:
                        if (a1, a2) in disjoint_pairs:
                            key = (ind_iri, min(a1, a2), max(a1, a2))
                            if key in seen:
                                continue
                            seen.add(key)
                            ind_name = _local(ind_iri)
                            t1_name = _local(t1)
                            t2_name = _local(t2)
                            a1_name = _local(a1)
                            a2_name = _local(a2)
                            results.append({
                                "error_type":   "transitive_disjoint_violation",
                                "entity":       ind_iri,
                                "entity_label": ind_name,
                                "classes":      [t1, t2],
                                "class_labels": [t1_name, t2_name],
                                "disjoint_ancestors": [a1, a2],
                                "disjoint_ancestor_labels": [a1_name, a2_name],
                                "evidence": [
                                    f"Individual {ind_name} is typed as both "
                                    f"{t1_name} and {t2_name}",
                                    f"{t1_name} is a (transitive) subclass of "
                                    f"{a1_name}",
                                    f"{t2_name} is a (transitive) subclass of "
                                    f"{a2_name}",
                                    f"{a1_name} and {a2_name} are declared disjoint",
                                ],
                            })
    return results


def _find_all_domain_violations(
    g: Graph,
    all_individuals: dict[str, list[str]],
) -> List[Dict[str, Any]]:
    """Find ALL domain violations."""
    results = []
    for prop_uri in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_str = str(prop_uri)
        if _is_meta(prop_str):
            continue
        domains = [str(d) for d in g.objects(prop_uri, RDFS.domain)]
        if not domains:
            continue
        for subj, _, _ in g.triples((None, prop_uri, None)):
            if not isinstance(subj, URIRef):
                continue
            subj_str = str(subj)
            subj_types = set(all_individuals.get(subj_str, []))
            for domain in domains:
                if domain not in subj_types:
                    prop_name = _local(prop_str)
                    subj_name = _local(subj_str)
                    dom_name = _local(domain)
                    results.append({
                        "error_type":    "domain_violation",
                        "entity":        subj_str,
                        "entity_label":  subj_name,
                        "property":      prop_str,
                        "property_label": prop_name,
                        "expected_domain": domain,
                        "evidence": [
                            f"{subj_name} uses property {prop_name}",
                            f"but is not typed as {dom_name} (the declared domain)",
                        ],
                    })
    return results


def _find_all_range_violations(
    g: Graph,
    all_individuals: dict[str, list[str]],
) -> List[Dict[str, Any]]:
    """Find ALL range violations."""
    results = []
    for prop_uri in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_str = str(prop_uri)
        if _is_meta(prop_str):
            continue
        ranges = [str(r) for r in g.objects(prop_uri, RDFS.range)]
        if not ranges:
            continue
        for _, _, obj in g.triples((None, prop_uri, None)):
            if not isinstance(obj, URIRef):
                continue
            obj_str = str(obj)
            obj_types = set(all_individuals.get(obj_str, []))
            for rng in ranges:
                if rng not in obj_types:
                    prop_name = _local(prop_str)
                    obj_name = _local(obj_str)
                    rng_name = _local(rng)
                    results.append({
                        "error_type":    "range_violation",
                        "entity":        obj_str,
                        "entity_label":  obj_name,
                        "property":      prop_str,
                        "property_label": prop_name,
                        "expected_range": rng,
                        "evidence": [
                            f"{obj_name} is the object of property {prop_name}",
                            f"but is not typed as {rng_name} (the declared range)",
                        ],
                    })
    return results


def _find_all_class_as_subject(
    g: Graph,
    all_classes: set[str],
) -> List[Dict[str, Any]]:
    """Find ALL class-as-subject violations."""
    _CLASS_META_PROPS = {
        str(RDF.type), str(RDFS.subClassOf), str(OWL.equivalentClass),
        str(OWL.disjointWith), str(RDFS.label), str(RDFS.comment),
        str(OWL.deprecated), str(OWL.versionInfo), str(OWL.priorVersion),
        str(OWL.backwardCompatibleWith), str(OWL.incompatibleWith),
    }
    _ANNOTATION_NS = (
        "http://purl.obolibrary.org/obo/IAO_",
        "http://purl.obolibrary.org/obo/",
        "http://www.w3.org/2004/02/skos/",
        "http://purl.org/dc/",
        "http://schema.org/",
        "http://xmlns.com/foaf/",
        "http://www.geneontology.org/formats/oboInOwl#",
        "http://purl.org/vocab/",
        "http://creativecommons.org/",
    )

    results = []
    seen = set()
    for s, p, o in g:
        if not isinstance(s, URIRef):
            continue
        s_str = str(s)
        p_str = str(p)
        if p_str in _CLASS_META_PROPS or _is_meta(p_str):
            continue
        if any(p_str.startswith(ns) for ns in _ANNOTATION_NS):
            continue
        if (URIRef(p_str), RDF.type, OWL.AnnotationProperty) in g:
            continue
        if s_str in all_classes and s_str not in seen:
            seen.add(s_str)
            results.append({
                "error_type":   "class_used_as_individual",
                "entity":       s_str,
                "entity_label": _local(s_str),
                "property":     p_str,
                "property_label": _local(p_str),
                "evidence": [
                    f"Class <{_local(s_str)}> is the subject of property {_local(p_str)}",
                    "Classes cannot be used as individuals in property assertions",
                ],
            })
    return results


def _find_all_functional_violations(g: Graph) -> List[Dict[str, Any]]:
    """Find ALL functional property violations."""
    functional_props = set(
        str(p) for p in g.subjects(RDF.type, OWL.FunctionalProperty)
        if not _is_meta(str(p))
    )
    functional_props.update(
        str(p) for p in g.subjects(RDF.type, OWL.InverseFunctionalProperty)
        if not _is_meta(str(p))
    )
    if not functional_props:
        return []

    fillers: dict[tuple[str, str], list[str]] = {}
    for s, p, o in g:
        p_str = str(p)
        if p_str not in functional_props:
            continue
        key = (str(s), p_str)
        fillers.setdefault(key, []).append(str(o))

    results = []
    for (subj, prop), values in fillers.items():
        unique_vals = list(dict.fromkeys(values))
        if len(unique_vals) > 1:
            results.append({
                "error_type":    "functional_property_violation",
                "entity":        subj,
                "entity_label":  _local(subj),
                "property":      prop,
                "property_label": _local(prop),
                "fillers":       unique_vals[:5],
                "evidence": [
                    f"{_local(prop)} is a FunctionalProperty (max one filler per subject)",
                    f"But {_local(subj)} has {len(unique_vals)} fillers: "
                    f"{', '.join(_local(v) for v in unique_vals[:3])}...",
                ],
            })
    return results


def _find_all_property_type_mismatches(g: Graph) -> List[Dict[str, Any]]:
    """Find ALL object property / literal type mismatches."""
    obj_props = set(
        str(p) for p in g.subjects(RDF.type, OWL.ObjectProperty)
        if not _is_meta(str(p))
    )
    results = []
    for s, p, o in g:
        if not isinstance(o, Literal):
            continue
        p_str = str(p)
        if p_str not in obj_props:
            continue
        results.append({
            "error_type":    "property_type_violation",
            "entity":        str(s),
            "entity_label":  _local(str(s)),
            "property":      p_str,
            "property_label": _local(p_str),
            "evidence": [
                f"{_local(p_str)} is declared owl:ObjectProperty",
                f"but {_local(str(s))} uses it with literal value: '{str(o)}'",
            ],
        })
    return results


def extract_all_inconsistencies(owl_path: str) -> Dict[str, Any]:
    """
    Comprehensive scan: find ALL violations in the ontology, not just the first.

    Returns a summary dict with:
      - total_violations: int
      - by_type: {error_type: count}
      - all_violations: [list of all violation dicts]
      - primary_evidence: the first (most important) violation for RL
      - summary_text: human-readable multi-line summary
    """
    try:
        g = Graph()
        g.parse(owl_path)
    except Exception as exc:
        logger.error("[axiom_extractor] Could not parse graph: %s", exc)
        return {
            "total_violations": 0,
            "by_type": {},
            "all_violations": [],
            "primary_evidence": None,
            "summary_text": f"Could not parse ontology: {exc}",
        }

    # Pre-compute shared sets
    all_classes: set[str] = {
        str(s) for s in g.subjects(RDF.type, OWL.Class)
        if isinstance(s, URIRef)
    }
    all_individuals = _collect_individuals(g, all_classes)

    # Run ALL checks and collect every violation
    all_violations: List[Dict[str, Any]] = []

    # A — direct disjoint violations
    all_violations.extend(_find_all_disjoint_violations(g, all_individuals))
    # A2 — transitive disjoint violations
    all_violations.extend(
        _find_all_transitive_disjoint_violations(g, all_individuals))
    # B — domain violations
    all_violations.extend(_find_all_domain_violations(g, all_individuals))
    # C — range violations
    all_violations.extend(_find_all_range_violations(g, all_individuals))
    # D — class-as-subject
    all_violations.extend(_find_all_class_as_subject(g, all_classes))
    # E — functional property violations
    all_violations.extend(_find_all_functional_violations(g))
    # F — property type mismatch
    all_violations.extend(_find_all_property_type_mismatches(g))

    # External namespace types
    ext = _find_external_type_violations(g)
    if ext:
        all_violations.append(ext)

    # Count by type
    by_type: Dict[str, int] = {}
    for v in all_violations:
        etype = v.get("error_type", "unknown")
        by_type[etype] = by_type.get(etype, 0) + 1

    # Build human-readable summary
    summary_lines = [
        f"Total violations found: {len(all_violations)}",
    ]
    for etype, count in sorted(by_type.items()):
        summary_lines.append(f"  {etype}: {count}")
    if all_violations:
        summary_lines.append("")
        summary_lines.append("Detailed violations:")
        for idx, v in enumerate(all_violations, 1):
            entity = v.get("entity_label", v.get("entity", "?"))
            etype = v.get("error_type", "?")
            ev_lines = v.get("evidence", [])
            summary_lines.append(f"  [{idx}] {etype} — {entity}")
            for line in ev_lines[:2]:
                summary_lines.append(f"      {line}")

    primary = all_violations[0] if all_violations else None

    return {
        "total_violations": len(all_violations),
        "by_type": by_type,
        "all_violations": all_violations,
        "primary_evidence": primary,
        "summary_text": "\n".join(summary_lines),
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_enhanced_superclass_map(g: Graph) -> dict[str, set[str]]:
    """
    Build superclass map that follows BOTH:
      1. Direct  rdfs:subClassOf  <URI>-> <URI>
      2. owl:equivalentClass-> owl:intersectionOf  (extract URI members
         as implicit superclasses)

    Returns {class_iri: {direct_parent_iris}}.
    """
    sub_of: dict[str, set[str]] = {}

    # 1. Direct rdfs:subClassOf
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            sub_of.setdefault(str(s), set()).add(str(o))

    # 2. owl:equivalentClass-> owl:intersectionOf members
    for cls, _, bnode in g.triples((None, OWL.equivalentClass, None)):
        if not isinstance(cls, URIRef):
            continue
        cls_str = str(cls)
        if isinstance(bnode, URIRef):
            # Direct equivalence to another named class
            sub_of.setdefault(cls_str, set()).add(str(bnode))
            sub_of.setdefault(str(bnode), set()).add(cls_str)
            continue
        # BNode: check intersectionOf
        for _, _, iof_head in g.triples((bnode, OWL.intersectionOf, None)):
            try:
                for m in Collection(g, iof_head):
                    if isinstance(m, URIRef):
                        sub_of.setdefault(cls_str, set()).add(str(m))
            except Exception:
                pass

    return sub_of


def _all_ancestors(
    cls: str,
    sub_of: dict[str, set[str]],
    _visited: set[str] | None = None,
) -> set[str]:
    """Compute transitive closure of superclasses for *cls*."""
    if _visited is None:
        _visited = set()
    if cls in _visited:
        return set()
    _visited.add(cls)
    parents = sub_of.get(cls, set())
    result = set(parents)
    for p in list(parents):
        result |= _all_ancestors(p, sub_of, _visited)
    return result


def _collect_individuals(
    g: Graph,
    all_classes: set[str],
) -> dict[str, list[str]]:
    """
    Return {individual_iri: [class_iri, ...]} for every rdf:type assertion
    that points at a non-meta class.
    """
    individuals: dict[str, list[str]] = {}
    for s, _, o in g.triples((None, RDF.type, None)):
        if not isinstance(o, URIRef):
            continue
        o_str = str(o)
        if _is_meta(o_str):
            continue
        individuals.setdefault(str(s), []).append(o_str)
    return individuals


# ---------------------------------------------------------------------------
# Check A — disjoint violations
# ---------------------------------------------------------------------------

def _find_disjoint_violations(
    g: Graph,
    all_individuals: dict[str, list[str]],
) -> Optional[Dict[str, Any]]:
    """Find individuals typed into two disjoint classes."""

    disjoint_pairs: set[tuple[str, str]] = set()

    # owl:disjointWith pairs
    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            disjoint_pairs.add((str(s), str(o)))
            disjoint_pairs.add((str(o), str(s)))

    # owl:AllDisjointClasses groups
    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for _, _, head in g.triples((adc, OWL.members, None)):
            try:
                members = [
                    m for m in Collection(g, head) if isinstance(m, URIRef)
                ]
            except Exception:
                members = []
            for i, m1 in enumerate(members):
                for m2 in members[i + 1:]:
                    disjoint_pairs.add((str(m1), str(m2)))
                    disjoint_pairs.add((str(m2), str(m1)))

    if not disjoint_pairs:
        print("[axiom_extractor] No disjoint axioms found")
        return None

    for ind_iri, classes in all_individuals.items():
        if len(classes) < 2:
            continue
        for i, c1 in enumerate(classes):
            for c2 in classes[i + 1:]:
                if (c1, c2) in disjoint_pairs or (c2, c1) in disjoint_pairs:
                    ind_name = _local(ind_iri)
                    c1_name = _local(c1)
                    c2_name = _local(c2)
                    return {
                        "error_type":   "disjoint_violation",
                        "entity":       ind_iri,
                        "entity_label": ind_name,
                        "classes":      [c1, c2],
                        "class_labels": [c1_name, c2_name],
                        "evidence": [
                            f"Individual {ind_name} is typed as both "
                            f"{c1_name} and {c2_name}",
                            f"{c1_name} and {c2_name} are declared disjoint",
                            f"ClassAssertion({c1_name}, {ind_name})",
                            f"ClassAssertion({c2_name}, {ind_name})",
                            f"DisjointClasses({c1_name}, {c2_name})",
                        ],
                    }

    print("[axiom_extractor] No disjoint violations found in individuals")
    return None


# ---------------------------------------------------------------------------
# Check A2 — transitive disjoint violations (via class hierarchy)
# ---------------------------------------------------------------------------

def _find_transitive_disjoint_violations(
    g: Graph,
    all_individuals: dict[str, list[str]],
) -> Optional[Dict[str, Any]]:
    """
    Detect individuals whose types conflict through the TRANSITIVE
    subclass hierarchy.

    The standard disjoint check (A) only catches DIRECT conflicts:
    individual typed as both ClassX and ClassY where DisjointClasses(X,Y).

    This check also catches INDIRECT conflicts, e.g.:
      - ind typed as  processing_element  AND  electronic_control_unit
      - processing_element-> hardware_part-> hardware_element-> non_system_level_element
      - electronic_control_unit-> system-> system_level_element
      - DisjointClasses(non_system_level_element, system_level_element)

    These are invisible to the simple check because the disjoint pair
    (non_system_level_element, system_level_element) is deep in the
    ancestor chain, and the equivalentClass/intersectionOf definitions
    create implicit subClassOf links that rdfs:subClassOf alone misses.
    """
    # Build enhanced hierarchy (includes equivalentClass→intersectionOf)
    sub_of = _build_enhanced_superclass_map(g)

    # Collect all disjoint pairs
    disjoint_pairs: set[tuple[str, str]] = set()
    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            disjoint_pairs.add((str(s), str(o)))
            disjoint_pairs.add((str(o), str(s)))
    for adc in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for _, _, head in g.triples((adc, OWL.members, None)):
            try:
                members = [m for m in Collection(
                    g, head) if isinstance(m, URIRef)]
            except Exception:
                members = []
            for i, m1 in enumerate(members):
                for m2 in members[i + 1:]:
                    disjoint_pairs.add((str(m1), str(m2)))
                    disjoint_pairs.add((str(m2), str(m1)))

    if not disjoint_pairs:
        return None

    # Check each individual with ≥2 types
    for ind_iri, classes in all_individuals.items():
        if len(classes) < 2:
            continue

        # Compute ancestor set for each type
        type_ancestors: dict[str, set[str]] = {}
        for c in classes:
            type_ancestors[c] = _all_ancestors(c, sub_of) | {c}

        # Check all pairs of types for transitive disjointness
        class_list = list(type_ancestors.keys())
        for i, t1 in enumerate(class_list):
            for t2 in class_list[i + 1:]:
                for a1 in type_ancestors[t1]:
                    for a2 in type_ancestors[t2]:
                        if (a1, a2) in disjoint_pairs:
                            ind_name = _local(ind_iri)
                            t1_name = _local(t1)
                            t2_name = _local(t2)
                            a1_name = _local(a1)
                            a2_name = _local(a2)

                            logger.info(
                                "[axiom_extractor] Transitive disjoint: "
                                "%s typed as %s (→%s) AND %s (→%s), "
                                "but %s ⊥ %s",
                                ind_name, t1_name, a1_name,
                                t2_name, a2_name, a1_name, a2_name,
                            )

                            return {
                                "error_type":   "transitive_disjoint_violation",
                                "entity":       ind_iri,
                                "entity_label": ind_name,
                                "classes":      [t1, t2],
                                "class_labels": [t1_name, t2_name],
                                "disjoint_ancestors": [a1, a2],
                                "disjoint_ancestor_labels": [a1_name, a2_name],
                                "evidence": [
                                    f"Individual {ind_name} is typed as both "
                                    f"{t1_name} and {t2_name}",
                                    f"{t1_name} is a (transitive) subclass of "
                                    f"{a1_name}",
                                    f"{t2_name} is a (transitive) subclass of "
                                    f"{a2_name}",
                                    f"{a1_name} and {a2_name} are declared "
                                    f"disjoint",
                                    f"Fix: remove one of the rdf:type assertions "
                                    f"({t1_name} or {t2_name}) from {ind_name}",
                                ],
                                "actions": [
                                    {
                                        "action_type":   "remove_type_assertion",
                                        "entity":        ind_iri,
                                        "class":         t1,
                                        "description":   f"Remove type {t1_name} "
                                                         f"from {ind_name}",
                                        "risk":          "low",
                                        "risk_penalty":  1.0,
                                    },
                                    {
                                        "action_type":   "remove_type_assertion",
                                        "entity":        ind_iri,
                                        "class":         t2,
                                        "description":   f"Remove type {t2_name} "
                                                         f"from {ind_name}",
                                        "risk":          "low",
                                        "risk_penalty":  1.0,
                                    },
                                    {
                                        "action_type":   "drop_entity",
                                        "target":        ind_iri,
                                        "description":   f"Drop {ind_name} entirely",
                                        "risk":          "medium",
                                        "risk_penalty":  2.0,
                                    },
                                ],
                            }

    print("[axiom_extractor] No transitive disjoint violations found")
    return None


# ---------------------------------------------------------------------------
# Check B — domain violations
# ---------------------------------------------------------------------------

def _find_domain_violations(
    g: Graph,
    all_individuals: dict[str, list[str]],
) -> Optional[Dict[str, Any]]:
    """
    Find property assertions where the subject is not typed as the
    property's rdfs:domain class.
    """
    for prop_uri in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_str = str(prop_uri)
        if _is_meta(prop_str):
            continue
        domains = [str(d) for d in g.objects(prop_uri, RDFS.domain)]
        if not domains:
            continue
        for subj, _, _ in g.triples((None, prop_uri, None)):
            if not isinstance(subj, URIRef):
                continue
            subj_str = str(subj)
            subj_types = set(all_individuals.get(subj_str, []))
            for domain in domains:
                if domain not in subj_types:
                    prop_name = _local(prop_str)
                    subj_name = _local(subj_str)
                    dom_name = _local(domain)
                    return {
                        "error_type":    "domain_violation",
                        "entity":        subj_str,
                        "entity_label":  subj_name,
                        "property":      prop_str,
                        "property_label": prop_name,
                        "expected_domain": domain,
                        "evidence": [
                            f"{subj_name} uses property {prop_name}",
                            f"but is not typed as {dom_name} "
                            f"(the declared domain)",
                            f"PropertyAssertion({prop_name}, {subj_name}, ...)",
                            f"Domain({prop_name}, {dom_name})",
                            f"Missing: ClassAssertion({dom_name}, {subj_name})",
                        ],
                        "actions_hint": "add_type_assertion_or_remove_property",
                    }

    print("[axiom_extractor] No domain violations found")
    return None


# ---------------------------------------------------------------------------
# Check C — range violations
# ---------------------------------------------------------------------------

def _find_range_violations(
    g: Graph,
    all_individuals: dict[str, list[str]],
) -> Optional[Dict[str, Any]]:
    """
    Find property assertions where the object is not typed as the
    property's rdfs:range class.
    """
    for prop_uri in g.subjects(RDF.type, OWL.ObjectProperty):
        prop_str = str(prop_uri)
        if _is_meta(prop_str):
            continue
        ranges = [str(r) for r in g.objects(prop_uri, RDFS.range)]
        if not ranges:
            continue
        for _, _, obj in g.triples((None, prop_uri, None)):
            if not isinstance(obj, URIRef):
                continue
            obj_str = str(obj)
            obj_types = set(all_individuals.get(obj_str, []))
            for rng in ranges:
                if rng not in obj_types:
                    prop_name = _local(prop_str)
                    obj_name = _local(obj_str)
                    rng_name = _local(rng)
                    return {
                        "error_type":    "range_violation",
                        "entity":        obj_str,
                        "entity_label":  obj_name,
                        "property":      prop_str,
                        "property_label": prop_name,
                        "expected_range": rng,
                        "evidence": [
                            f"{obj_name} is the object of property {prop_name}",
                            f"but is not typed as {rng_name} "
                            f"(the declared range)",
                            f"PropertyAssertion({prop_name}, ..., {obj_name})",
                            f"Range({prop_name}, {rng_name})",
                            f"Missing: ClassAssertion({rng_name}, {obj_name})",
                        ],
                        "actions_hint": "add_type_assertion_or_remove_property",
                    }

    print("[axiom_extractor] No range violations found")
    return None


# ---------------------------------------------------------------------------
# Check D — class URI used as subject of a domain property assertion
# ---------------------------------------------------------------------------

def _find_class_as_subject(
    g: Graph,
    all_classes: set[str],
) -> Optional[Dict[str, Any]]:
    """
    Detect the most common cause of 'Consistent:False / Unsat:0':
    an owl:Class IRI appearing as the *subject* of a non-meta property.

    This happens when align_triples maps an entity to a class URI and
    then uses it directly in a property assertion instead of minting
    an individual first.
    """
    # Properties that are legitimately used with class subjects.
    # Includes OWL structural predicates AND annotation properties —
    # IAO_*, skos:*, dc:*, dcterms:*, schema:* are all used to document
    # classes and must not be treated as individual assertions.
    _CLASS_META_PROPS = {
        str(RDF.type),
        str(RDFS.subClassOf),
        str(OWL.equivalentClass),
        str(OWL.disjointWith),
        str(RDFS.label),
        str(RDFS.comment),
        str(OWL.deprecated),
        str(OWL.versionInfo),
        str(OWL.priorVersion),
        str(OWL.backwardCompatibleWith),
        str(OWL.incompatibleWith),
    }

    # Namespace prefixes whose properties are always annotation/documentation
    _ANNOTATION_NS = (
        "http://purl.obolibrary.org/obo/IAO_",   # IAO annotation properties
        "http://purl.obolibrary.org/obo/",        # OBO annotations generally
        "http://www.w3.org/2004/02/skos/",        # SKOS
        # Dublin Core dc: and dcterms:
        "http://purl.org/dc/",
        "http://schema.org/",                     # schema.org
        "http://xmlns.com/foaf/",                 # FOAF
        "http://www.geneontology.org/formats/oboInOwl#",  # OBO in OWL
        "http://purl.org/vocab/",
        "http://creativecommons.org/",
    )

    for s, p, o in g:
        if not isinstance(s, URIRef):
            continue
        s_str = str(s)
        p_str = str(p)
        if p_str in _CLASS_META_PROPS:
            continue
        if _is_meta(p_str):
            continue
        # Skip annotation property namespaces
        if any(p_str.startswith(ns) for ns in _ANNOTATION_NS):
            continue
        # Also skip if the property is declared as owl:AnnotationProperty
        if (URIRef(p_str), RDF.type, OWL.AnnotationProperty) in g:
            continue
        if s_str in all_classes:
            prop_name = _local(p_str)
            class_name = _local(s_str)
            return {
                "error_type":   "class_used_as_individual",
                "entity":       s_str,
                "entity_label": class_name,
                "property":     p_str,
                "property_label": prop_name,
                "evidence": [
                    f"Class <{class_name}> is the subject of property "
                    f"{prop_name}",
                    "Classes cannot be used as individuals in property "
                    "assertions — an individual must be created first",
                    f"Fix: mint an individual of type {class_name} and "
                    "use it as the subject instead",
                    "Root cause: align_triples returned a class URI from "
                    "ent_map / best_match without calling ensure_individual()",
                ],
                "actions_hint": "mint_individual_for_class",
            }

    print("[axiom_extractor] No class-as-subject violations found")
    return None


# ---------------------------------------------------------------------------
# Check E — functional property used with two different fillers
# ---------------------------------------------------------------------------

def _find_functional_violations(g: Graph) -> Optional[Dict[str, Any]]:
    """
    Find subjects that have two distinct values for a functional property
    (owl:FunctionalProperty or owl:InverseFunctionalProperty).
    """
    functional_props = set(
        str(p) for p in g.subjects(RDF.type, OWL.FunctionalProperty)
        if not _is_meta(str(p))
    )
    functional_props.update(
        str(p) for p in g.subjects(RDF.type, OWL.InverseFunctionalProperty)
        if not _is_meta(str(p))
    )

    if not functional_props:
        return None

    # {(subject, prop): [filler, ...]}
    fillers: dict[tuple[str, str], list[str]] = {}
    for s, p, o in g:
        p_str = str(p)
        if p_str not in functional_props:
            continue
        key = (str(s), p_str)
        fillers.setdefault(key, []).append(str(o))

    for (subj, prop), values in fillers.items():
        # preserves order, deduplicates
        unique_vals = list(dict.fromkeys(values))
        if len(unique_vals) > 1:
            prop_name = _local(prop)
            subj_name = _local(subj)
            return {
                "error_type":    "functional_property_violation",
                "entity":        subj,
                "entity_label":  subj_name,
                "property":      prop,
                "property_label": prop_name,
                "fillers":       unique_vals[:5],
                "evidence": [
                    f"{prop_name} is a FunctionalProperty "
                    f"(max one filler per subject)",
                    f"But {subj_name} has {len(unique_vals)} fillers: "
                    f"{', '.join(_local(v) for v in unique_vals[:3])}...",
                    "Fix: remove all but one PropertyAssertion for this "
                    "subject+property combination",
                ],
            }

    print("[axiom_extractor] No functional property violations found")
    return None


# ---------------------------------------------------------------------------
# Check F — object property used with a Literal value
# ---------------------------------------------------------------------------

def _find_property_type_mismatch(g: Graph) -> Optional[Dict[str, Any]]:
    """
    Find owl:ObjectProperty assertions where the object is a Literal
    (should be an IRI / individual).

    BUG FIX: The original filter
        if any(ns in str(p) for ns in ["rdf-syntax", "owl", "rdfs", ...])
    matched ANY URI containing the substring "owl", which silently
    discarded properties from namespaces like:
        http://cpsagila.cs.uni-kl.de/GENIALOnt#hasOwlProperty
    The fix checks only exact W3C namespace *prefixes*.
    """
    obj_props = set(
        str(p) for p in g.subjects(RDF.type, OWL.ObjectProperty)
        if not _is_meta(str(p))
    )

    for s, p, o in g:
        if not isinstance(o, Literal):
            continue
        p_str = str(p)
        if p_str not in obj_props:
            continue
        prop_name = _local(p_str)
        subj_name = _local(str(s))
        return {
            "error_type":    "property_type_violation",
            "entity":        str(s),
            "entity_label":  subj_name,
            "property":      p_str,
            "property_label": prop_name,
            "evidence": [
                f"{prop_name} is declared owl:ObjectProperty",
                f"but {subj_name} uses it with literal value: '{str(o)}'",
                "ObjectProperty can only relate individuals to individuals",
                "Fix: change to owl:DatatypeProperty or replace "
                "the literal with an individual IRI",
            ],
        }

    print("[axiom_extractor] No property type mismatch found")
    return None
