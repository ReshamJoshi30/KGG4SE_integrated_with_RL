# qa/apply_fix.py
"""
Apply repair actions to OWL graphs.

FIX — three new action types added to support the fixed repair_candidates:

  1. add_type_assertion             (for domain/range violations)
  2. remove_property_assertion      (all assertions for entity+property)
  3. remove_specific_property_assertion  (one exact s+p+o triple)
  4. mint_individual_for_class      (for class_used_as_individual)

All existing action types unchanged (backward compatible).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from rdflib import Graph, Literal, OWL, RDF, URIRef

import config


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def _load_graph(path: str) -> Graph:
    g = Graph()
    g.parse(path)
    return g


def _save_graph(
    g: Graph,
    out_dir: str,
    step_id: Optional[int] = None,
    prefix: str = "repaired",
) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    fname = (
        f"{prefix}_step_{step_id:04d}_{uid}.owl"
        if step_id is not None
        else f"{prefix}_{uid}.owl"
    )
    out_path = Path(out_dir) / fname
    g.serialize(destination=str(out_path), format="xml")
    return str(out_path)


# ---------------------------------------------------------------------------
# Atomic graph operations
# ---------------------------------------------------------------------------

def _remap_entity(g: Graph, old_iri: str, new_iri: str) -> None:
    """Replace all occurrences of old_iri with new_iri."""
    old_u = URIRef(old_iri)
    new_u = URIRef(new_iri)
    to_add, to_remove = [], []
    for s, p, o in g:
        if s == old_u:
            to_remove.append((s, p, o))
            to_add.append((new_u, p, o))
        if o == old_u:
            to_remove.append((s, p, o))
            to_add.append((s, p, new_u))
    for t in to_remove:
        g.remove(t)
    for t in to_add:
        g.add(t)


def _drop_entity(g: Graph, iri: str) -> None:
    """Remove ALL triples involving this entity (subject or object)."""
    u = URIRef(iri)
    bad = [(s, p, o) for s, p, o in g if s == u or o == u]
    for t in bad:
        g.remove(t)
    print(f"[apply_fix] Dropped entity <{iri}> ({len(bad)} triples removed)")


def _drop_class_assertion(g: Graph, entity_iri: str, class_iri: str) -> None:
    """Remove exactly: entity rdf:type class."""
    g.remove((URIRef(entity_iri), RDF.type, URIRef(class_iri)))
    print(f"[apply_fix] Removed ClassAssertion(<{class_iri}>, <{entity_iri}>)")


def _remove_disjoint_axiom(g: Graph, class1_iri: str, class2_iri: str) -> None:
    """Remove owl:disjointWith in both directions."""
    c1, c2 = URIRef(class1_iri), URIRef(class2_iri)
    g.remove((c1, OWL.disjointWith, c2))
    g.remove((c2, OWL.disjointWith, c1))
    print(
        f"[apply_fix] Removed DisjointClasses(<{class1_iri}>, <{class2_iri}>)")


# NEW -----------------------------------------------------------------------

def _add_type_assertion(g: Graph, entity_iri: str, class_iri: str) -> None:
    """Add: entity rdf:type class (idempotent)."""
    triple = (URIRef(entity_iri), RDF.type, URIRef(class_iri))
    if triple not in g:
        g.add(triple)
        print(
            f"[apply_fix] Added ClassAssertion(<{class_iri}>, <{entity_iri}>)")
    else:
        print(f"[apply_fix] ClassAssertion already present, skipping")


def _remove_property_assertion(
    g: Graph, entity_iri: str, property_iri: str
) -> None:
    """Remove ALL triples: entity property ?."""
    subj = URIRef(entity_iri)
    prop = URIRef(property_iri)
    bad = list(g.triples((subj, prop, None)))
    for t in bad:
        g.remove(t)
    print(f"[apply_fix] Removed {len(bad)} assertion(s) of "
          f"<{property_iri}> on <{entity_iri}>")


def _remove_specific_property_assertion(
    g: Graph, entity_iri: str, property_iri: str, value: str
) -> None:
    """Remove exactly: entity property value (IRI or literal)."""
    subj = URIRef(entity_iri)
    prop = URIRef(property_iri)
    # Try as URIRef first, then Literal
    for obj in (URIRef(value), Literal(value)):
        if (subj, prop, obj) in g:
            g.remove((subj, prop, obj))
            print(f"[apply_fix] Removed specific assertion "
                  f"<{entity_iri}> <{property_iri}> {value!r}")
            return
    print(f"[apply_fix] Specific assertion not found (no-op)")


def _mint_individual_for_class(
    g: Graph, class_iri: str, property_iri: str
) -> None:
    """
    Repair the class-used-as-individual pattern.

    1. Find all triples where the class IRI is the subject of property_iri.
    2. Mint a new individual of that class.
    3. Redirect those triples to use the individual instead.
    4. Remove the original class-as-subject triples.
    """
    import hashlib
    cls_u = URIRef(class_iri)
    prop_u = URIRef(property_iri)
    cls_name = class_iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    # Mint a stable individual URI
    uid = hashlib.md5(f"mint|{cls_name}".encode()).hexdigest()[:8]
    ind_uri = URIRef(f"{config.BASE_INDIVIDUAL_NS}{cls_name}_{uid}")

    # Ensure the individual is typed
    if (ind_uri, RDF.type, cls_u) not in g:
        g.add((ind_uri, RDF.type, cls_u))

    # Find and redirect all property assertions where class is subject
    to_redirect = list(g.triples((cls_u, prop_u, None)))
    for s, p, o in to_redirect:
        g.remove((s, p, o))
        g.add((ind_uri, p, o))

    print(f"[apply_fix] Minted individual <{ind_uri}> of type <{class_iri}>, "
          f"redirected {len(to_redirect)} triple(s)")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_fix(
    current_owl: str,
    error_obj: Dict[str, Any],
    action_idx: int,
    onto_idx=None,
    step_id: Optional[int] = None,
    out_dir: str = "outputs/rl_repair_steps",
) -> str:
    """
    Apply one repair action and save a new OWL file.

    Returns path to the newly saved OWL file.
    """
    g = _load_graph(current_owl)
    actions = error_obj.get("actions", []) or []

    if not (0 <= action_idx < len(actions)):
        print(f"[apply_fix] Invalid action_idx {action_idx} "
              f"(len={len(actions)}), performing no-op")
        return _save_graph(g, out_dir=out_dir, step_id=step_id)

    action = actions[action_idx]

    # --- Structured dict actions ---
    if isinstance(action, dict):
        # Support both "action_type" (standard) and "type" (repair_candidates legacy)
        atype = action.get("action_type", "") or action.get("type", "")

        if atype == "drop_class_assertion":
            _drop_class_assertion(
                g, action["target_entity"], action["target_class"]
            )

        elif atype == "remove_type_assertion":          # alias
            _drop_class_assertion(g, action["entity"], action["class"])

        elif atype == "drop_entity":
            # Support both "target" (old) and "entity" (new) key names
            iri = action.get("entity") or action.get("target", "Unknown")
            _drop_entity(g, iri)

        elif atype == "drop_both_entities":
            for t in action.get("targets", []):
                _drop_entity(g, t)

        elif atype == "remap_entity":
            _remap_entity(g, action["old_iri"], action["new_iri"])

        elif atype == "remove_disjoint_axiom":
            classes = action.get("target_classes", [])
            if len(classes) >= 2:
                _remove_disjoint_axiom(g, classes[0], classes[1])

        elif atype == "weaken_relation":                # alias
            targets = action.get("targets", [])
            if len(targets) >= 2:
                _remove_disjoint_axiom(g, targets[0], targets[1])

        # NEW action types
        elif atype == "add_type_assertion":
            _add_type_assertion(
                g, action["target_entity"], action["target_class"]
            )

        elif atype == "remove_property_assertion":
            _remove_property_assertion(
                g, action["target_entity"], action["target_property"]
            )

        elif atype == "remove_specific_property_assertion":
            _remove_specific_property_assertion(
                g,
                action["target_entity"],
                action["target_property"],
                action["target_value"],
            )

        elif atype == "mint_individual_for_class":
            _mint_individual_for_class(
                g, action["target_class"], action["target_property"]
            )

        # Handle actions from external_type_violation
        elif atype == "remove_type":
            entity = action.get("entity", "")
            bad_type = action.get("bad_type", "")
            if entity and bad_type:
                from rdflib import RDF as _RDF
                u = URIRef(entity)
                bt = URIRef(bad_type)
                removed = 0
                for t in list(g.triples((u, _RDF.type, bt))):
                    g.remove(t)
                    removed += 1
                print(f"[apply_fix] remove_type: removed {removed} rdf:type triple(s) "
                      f"from <{entity.split('/')[-1]}>")

        elif atype == "remove_external_type":
            # Remove a single rdf:type assertion for an external-namespace class
            ind_uri = action.get("entity")
            type_uri = action.get("type_uri")
            if ind_uri and type_uri:
                triple = (URIRef(ind_uri), RDF.type, URIRef(type_uri))
                if triple in g:
                    g.remove(triple)
                    type_name = type_uri.split("#")[-1].split("/")[-1]
                    ind_name = ind_uri.split("/")[-1]
                    print(
                        f"[apply_fix] Removed external type <{type_name}> from <{ind_name}>")
                else:
                    print(
                        f"[apply_fix] remove_external_type: triple not found (already removed?)")

        elif atype == "no_op":
            print("[apply_fix] no_op — manual inspection needed")

        else:
            print(f"[apply_fix] Unknown action_type: {atype!r}, no-op")

    # --- Legacy string actions (backward compatible) ---
    elif isinstance(action, str):
        etype = error_obj.get("error_type", "")
        if etype == "unsat_class":
            iri = error_obj.get("entity_iri")
            suggs = error_obj.get("suggestions", []) or []
            if iri and action == "drop_entity":
                _drop_entity(g, iri)
            elif iri and action.startswith("map_to_suggestion_"):
                try:
                    idx = int(action.rsplit("_", 1)[-1])
                    if 0 <= idx < len(suggs):
                        _remap_entity(g, iri, suggs[idx])
                except ValueError:
                    pass

        elif etype == "disjoint_violation":
            pair = error_obj.get("entity_pair")
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a, b = pair
                if action == "drop_a":
                    _drop_entity(g, a)
                elif action == "drop_b":
                    _drop_entity(g, b)
                elif action == "drop_both":
                    _drop_entity(g, a)
                    _drop_entity(g, b)
                elif action == "weaken_relation":
                    _remove_disjoint_axiom(g, a, b)

    return _save_graph(g, out_dir=out_dir, step_id=step_id)
