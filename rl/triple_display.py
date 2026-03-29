"""
rl/triple_display.py

Human-readable display of KG inconsistencies and repair actions.

Formats the current error object and the surrounding graph neighbourhood
(loaded from the live OWL file) so that a human annotator can understand:
  - What the error actually means in plain language
  - Which concrete triples are responsible
  - Exactly what each candidate repair action will do to the graph

Used by HumanLoop.ask_user() to replace the previous terse action-type listing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# IRI shortening helpers
# ---------------------------------------------------------------------------

_KNOWN_NS: Dict[str, str] = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
    "http://www.w3.org/2000/01/rdf-schema#":        "rdfs:",
    "http://www.w3.org/2002/07/owl#":               "owl:",
    "http://www.w3.org/2001/XMLSchema#":            "xsd:",
    "http://cpsagila.cs.uni-kl.de/GENIALOnt#":     "",        # local ontology classes
    "http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/": "ind:",   # minted individuals
}


def shorten(iri: Any) -> str:
    """
    Turn a full IRI into a short, human-readable label.

    Examples
    --------
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" ->  "rdf:type"
    "http://cpsagila.cs.uni-kl.de/GENIALOnt#Sensor"  ->  "Sensor"
    "http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/Sensor_001" ->  "ind:Sensor_001"
    """
    s = str(iri)
    for prefix, short in _KNOWN_NS.items():
        if s.startswith(prefix):
            return short + s[len(prefix):]
    # Fallback: fragment or last path segment
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def _fmt_triple(s: Any, p: Any, o: Any, width: int = 60) -> str:
    """Format a single (s, p, o) triple as a readable line."""
    ss = shorten(s)
    ps = shorten(p)
    os = shorten(o)
    return f"    {ss:<28}  {ps:<20}  ->  {os}"


# ---------------------------------------------------------------------------
# Graph neighbourhood loader
# ---------------------------------------------------------------------------

def _load_neighbourhood(entity_iri: str, owl_path: str,
                         max_triples: int = 10) -> List[Tuple]:
    """
    Return up to *max_triples* RDF triples that directly involve *entity_iri*
    (as subject or object) from the OWL file at *owl_path*.

    Returns an empty list if the file cannot be parsed or rdflib is missing.
    """
    try:
        import rdflib
    except ImportError:
        return []

    try:
        g = rdflib.Graph()
        g.parse(str(owl_path))
        ref = rdflib.URIRef(entity_iri)
        triples: List[Tuple] = []
        for t in g.triples((ref, None, None)):
            triples.append(t)
            if len(triples) >= max_triples:
                break
        # Also include triples where entity appears as object
        remaining = max_triples - len(triples)
        for t in g.triples((None, None, ref)):
            triples.append(t)
            remaining -= 1
            if remaining <= 0:
                break
        return triples
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Action-effect descriptions
# ---------------------------------------------------------------------------

def _action_effect(action: Dict[str, Any]) -> str:
    """
    Return a one-line plain-English description of what the action will do
    to the graph (in terms of triples added / removed).
    """
    atype = action.get("action_type", "unknown")

    entity   = shorten(action.get("target_entity") or action.get("target") or
                       action.get("entity", "?"))
    cls      = shorten(action.get("target_class") or action.get("class", ""))
    prop     = shorten(action.get("target_property") or action.get("property", ""))
    val      = shorten(action.get("target_value", ""))
    new_iri  = shorten(action.get("new_iri", ""))
    bad_type = shorten(action.get("bad_type", ""))

    effects = {
        "drop_entity":
            f"DELETE all triples where <{entity}> is subject or object",
        "remap_entity":
            f"REPLACE every occurrence of <{entity}> with <{new_iri}>",
        "drop_class_assertion":
            f"DELETE triple:  {entity}  rdf:type  {cls}",
        "remove_type_assertion":
            f"DELETE triple:  {entity}  rdf:type  {cls or '?'}",
        "add_type_assertion":
            f"ADD triple:     {entity}  rdf:type  {cls}",
        "remove_disjoint_axiom":
            f"DELETE owl:disjointWith axiom between the two classes",
        "weaken_relation":
            f"DELETE owl:disjointWith axiom between the two classes",
        "remove_property_assertion":
            f"DELETE all triples:  {entity}  {prop}  ?",
        "remove_specific_property_assertion":
            f"DELETE triple:  {entity}  {prop}  {val}",
        "mint_individual_for_class":
            f"CREATE new individual of class <{entity}>, redirect <{prop}> assertions",
        "remove_type":
            f"DELETE triple:  {entity}  rdf:type  {bad_type}",
        "remove_external_type":
            f"DELETE external rdf:type  {bad_type}  from  {entity}",
        "no_op":
            "Skip — no graph change (manual inspection needed)",
    }
    return effects.get(atype, f"Unknown action type: {atype}")


# ---------------------------------------------------------------------------
# Risk badge
# ---------------------------------------------------------------------------

_RISK_BADGES = {
    "low":      "[ LOW  ]",
    "medium":   "[  MED ]",
    "high":     "[ HIGH ]",
    "critical": "[CRIT! ]",
    "none":     "[  --- ]",
}


def _risk_badge(risk: str) -> str:
    return _RISK_BADGES.get(risk.lower(), f"[{risk[:6].upper():^6}]")


# ---------------------------------------------------------------------------
# Main formatter
# ---------------------------------------------------------------------------

def format_error_for_human(
    error_obj: Dict[str, Any],
    owl_path: Optional[str] = None,
    agent_action: Optional[int] = None,
) -> str:
    """
    Build a multi-line string that presents the inconsistency and all repair
    actions in plain English for a human annotator.

    Parameters
    ----------
    error_obj    : The current error dict from RepairEnv._current_error
    owl_path     : Path to the current (broken) OWL file; used to pull
                   the graph neighbourhood.  Pass None to skip that section.
    agent_action : Index of the action the RL agent chose (highlighted).

    Returns
    -------
    str  — ready to print to stdout
    """
    W = 68  # console width
    lines: List[str] = []
    add = lines.append

    etype        = error_obj.get("error_type", "unknown")
    entity       = error_obj.get("entity") or error_obj.get("violating_entity", "")
    entity_label = error_obj.get("entity_label") or shorten(entity) or "?"
    evidence     = error_obj.get("evidence", [])
    actions      = error_obj.get("actions", [])
    class_labels = error_obj.get("class_labels", [])
    classes      = error_obj.get("classes", [])

    # ── Header ──────────────────────────────────────────────────────────────
    add("\n" + "=" * W)
    add(f"  INCONSISTENCY TYPE:  {etype.upper().replace('_', ' ')}")
    add("=" * W)

    # ── Entity ──────────────────────────────────────────────────────────────
    add(f"\n  Entity:  {entity_label}")
    if entity and entity != entity_label:
        add(f"    IRI:   {entity}")

    # ── Type-specific context ───────────────────────────────────────────────
    if etype in ("disjoint_violation", "transitive_disjoint_violation"):
        intended = error_obj.get("intended_class")
        add("\n  Conflicting type assignments (both cannot be true together):")
        for i, (cls_iri, cls_lbl) in enumerate(
                zip(classes, class_labels or [shorten(c) for c in classes])):
            marker = "  [PRIMARY IDENTITY]" if cls_lbl == intended else "  [LIKELY INCORRECT]" if intended else ""
            add(f"    {entity_label:<28}  rdf:type  ->  {cls_lbl}{marker}")
        if len(class_labels) >= 2:
            add(f"\n  Why it fails:")
            add(f"    {class_labels[0]}  owl:disjointWith  {class_labels[1]}")
        anc = error_obj.get("disjoint_ancestor_labels", [])
        if anc and len(anc) >= 2:
            add(f"\n  (Conflict is via ancestor classes: "
                f"{anc[0]}  _|_  {anc[1]})")
        if intended:
            add(f"\n  Recommendation: keep '{intended}' (matches entity name),"
                f" remove the other type.")

    elif etype == "class_used_as_individual":
        prop = shorten(error_obj.get("property", "?"))
        add(f"\n  Problem:  The class <{entity_label}> is being used as if it")
        add(f"            were an individual -- it appears as the subject of")
        add(f"            the property  <{prop}>.")
        add(f"            Classes cannot have property assertions; only")
        add(f"            individuals can.")

    elif etype in ("domain_violation", "range_violation"):
        prop = shorten(error_obj.get("property", "?"))
        side = "subject" if etype == "domain_violation" else "object"
        add(f"\n  Problem:  <{entity_label}> is the {side} of property <{prop}>")
        add(f"            but does not satisfy the property's required type.")

    elif etype == "functional_property_violation":
        prop = shorten(error_obj.get("property", "?"))
        add(f"\n  Problem:  Functional property <{prop}> on <{entity_label}>")
        add(f"            has more than one distinct value, which violates")
        add(f"            the functional constraint (max one value allowed).")

    elif etype == "property_type_violation":
        prop = shorten(error_obj.get("property", "?"))
        add(f"\n  Problem:  ObjectProperty <{prop}> on <{entity_label}> is used")
        add(f"            with a Literal (plain string/number) value instead")
        add(f"            of an individual IRI.")

    elif etype == "external_type_violation":
        bad = shorten(error_obj.get("bad_type", "?"))
        add(f"\n  Problem:  <{entity_label}> is typed as the external class")
        add(f"            <{bad}>.  External-namespace classes bring in axioms")
        add(f"            (e.g. disjointness) that conflict with local types.")

    # ── Evidence lines ───────────────────────────────────────────────────────
    if evidence:
        add("\n  Evidence:")
        for ev in evidence[:5]:
            add(f"    * {ev}")

    # ── Graph neighbourhood ──────────────────────────────────────────────────
    if owl_path and entity:
        neighbourhood = _load_neighbourhood(entity, owl_path, max_triples=8)
        if neighbourhood:
            add(f"\n  Live graph neighbourhood (triples involving {entity_label}):")
            for t in neighbourhood:
                add(_fmt_triple(*t))
        else:
            add(f"\n  (Could not load graph neighbourhood from OWL file)")

    # ── Action menu ─────────────────────────────────────────────────────────
    add("\n" + "-" * W)
    add("  AVAILABLE REPAIR ACTIONS")
    add("-" * W)

    for i, action in enumerate(actions):
        if not isinstance(action, dict):
            add(f"  [{i}]  {action}")
            continue

        risk    = action.get("risk", "unknown")
        badge   = _risk_badge(risk)
        desc    = action.get("description", action.get("action_type", "?"))
        effect  = _action_effect(action)
        penalty = action.get("risk_penalty", 0.0)

        agent_marker = "  <-- RL agent's choice" if i == agent_action else ""
        add(f"\n  [{i}]  {badge}  {desc}{agent_marker}")
        add(f"        Effect  : {effect}")
        add(f"        Penalty : -{penalty:.1f} from reward")

    add("-" * W)
    return "\n".join(lines)
