# qa/repair_candidates.py
"""
Given a parsed reasoner report and an OntologyIndex, produce a list of
repair tasks the RL agent can act on.

FIX — two bugs in the original _make_evidence_based_repairs():

  1. Only handled "disjoint_violation" and generic fallback.
     New evidence types from the fixed axiom_extractor
     ("domain_violation", "range_violation", "class_used_as_individual",
     "functional_property_violation", "property_type_violation") all fell
     through to the generic no_op branch, giving the RL agent nothing to
     learn from.

  2. The generic fallback produced a single "no_op" action.
     The RL reward signal was always 0 because the agent was never given
     a real action to apply.  The fallback now produces a "skip_entity"
     action that at least removes the offending triple.
"""

from __future__ import annotations

from typing import Any, Dict, List

from alignment.ontology_index import OntologyIndex


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def make_repair_candidates(
    report: Dict[str, Any],
    onto_idx: OntologyIndex,
) -> List[Dict[str, Any]]:
    """
    Generate repair candidates from a reasoner report.

    Priority order:
      1. Evidence extracted by axiom_extractor  (most specific)
      2. unsat_classes from reasoner output
      3. disjoint_violations from reasoner output

    Returns list of error dicts, each with an "actions" list of structured
    action dicts understood by apply_fix().
    """
    repairs: list[dict] = []

    if report is None:
        report = {}

    # --- Priority 1: evidence-based (axiom_extractor output) ---
    evidence = report.get("evidence")
    if evidence:
        repairs.extend(_make_evidence_based_repairs(evidence, onto_idx))

    # --- Priority 2: unsat classes ---
    for cls_iri in report.get("unsat_classes", []):
        tail = cls_iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        suggestions = onto_idx.suggest(tail, topn=3)

        actions: list[dict] = [
            {
                "action_type": "drop_entity",
                "target":      cls_iri,
                "description": f"Remove class {tail} entirely",
                "risk":        "high",
                "risk_penalty": 5.0,
            }
        ]
        for sugg_iri in suggestions:
            sugg_name = sugg_iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            actions.append({
                "action_type": "remap_entity",
                "old_iri":     cls_iri,
                "new_iri":     sugg_iri,
                "description": f"Remap {tail} -> {sugg_name}",
                "risk":        "medium",
                "risk_penalty": 2.0,
            })

        repairs.append({
            "error_type":   "unsat_class",
            "entity_iri":   cls_iri,
            "entity_label": tail,
            "suggestions":  suggestions,
            "actions":      actions,
        })

    # --- Priority 3: disjoint violations ---
    for item in report.get("disjoint_violations", []):
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            _entity, a, b = item[0], item[1], item[2]
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            _entity = None
            a, b = item[0], item[1]
        else:
            continue

        at = a.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        bt = b.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

        actions = []
        if _entity:
            actions.extend([
                {
                    "action_type": "remove_type_assertion",
                    "entity":      _entity,
                    "class":       a,
                    "description": f"Remove type {at} from "
                                   f"{_entity.rsplit('/', 1)[-1]}",
                    "risk":        "low",
                    "risk_penalty": 1.0,
                },
                {
                    "action_type": "remove_type_assertion",
                    "entity":      _entity,
                    "class":       b,
                    "description": f"Remove type {bt} from "
                                   f"{_entity.rsplit('/', 1)[-1]}",
                    "risk":        "low",
                    "risk_penalty": 1.0,
                },
            ])
        actions.append({
            "action_type": "weaken_relation",
            "targets":     [a, b],
            "description": f"Weaken disjoint constraint between {at} and {bt}",
            "risk":        "high",
            "risk_penalty": 8.0,
        })

        repairs.append({
            "error_type":      "disjoint_violation",
            "violating_entity": _entity,
            "class_pair":      (a, b),
            "labels":          (at, bt),
            "actions":         actions,
        })

    # --- Fallback: if still inconsistent but no specific violations found ---
    # This happens after partial repairs where rdflib can't see the remaining issue.
    # Try to infer the culprit entity from recent evidence or previous candidates.
    if not repairs and not report.get("is_consistent", True):
        evidence = report.get("evidence") or {}
        entity = evidence.get("entity")

        # If evidence is empty, check if there's a "last_entity" hint in the report
        # (env_repair can inject this from the previous step's entity)
        if not entity or entity == "Unknown":
            entity = report.get("last_entity", "Unknown")

        label = entity.rsplit(
            "/", 1)[-1].rsplit("#", 1)[-1] if entity != "Unknown" else "Unknown"

        repairs.append({
            "error_type":   "unknown_inconsistency",
            "entity":       entity,
            "entity_label": label,
            "evidence":     evidence.get("evidence", [f"Partial repair incomplete for {label}"]),
            "actions": [
                {
                    "action_type":  "drop_entity",
                    "entity":       entity,
                    "description":  f"Drop <{label}> entirely (fallback after partial repair)",
                    "risk":         "high",
                    "risk_penalty": 3.0,  # lower penalty — this is a continuation
                },
            ],
        })

    return repairs


# ---------------------------------------------------------------------------
# Evidence-based repairs (all types the fixed axiom_extractor can return)
# ---------------------------------------------------------------------------

def _intended_class_index(entity_label: str, class_labels: List[str]) -> "int | None":
    """
    Detect which of the two conflicting class labels is the entity's
    *intended* (primary) type by name-matching against the entity label.

    Strategy
    --------
    1. Normalise both entity label and class labels (lower-case, strip digits
       and UUID fragments, collapse underscores).
    2. If the entity label *starts with* or *contains* one of the class names,
       that class is the intended type.
    3. If neither or both match, return None (cannot determine preference).

    Examples
    --------
    entity_label="electronic_control_unit_f46b9510"
    class_labels=["processing_element", "electronic_control_unit"]
       -> returns 1  (electronic_control_unit matches)

    entity_label="temperature_sensor_a1b2"
    class_labels=["sensor", "actuator"]
       -> returns 0  (sensor matches)
    """
    import re

    def _norm(s: str) -> str:
        # lower-case, remove UUID/hex fragments, collapse underscores
        s = s.lower()
        s = re.sub(r'_?[0-9a-f]{6,}', '', s)   # strip UUID/hex suffixes
        s = re.sub(r'_+', '_', s).strip('_')
        return s

    entity_norm = _norm(entity_label)
    scores = []
    for lbl in class_labels:
        cls_norm = _norm(lbl)
        if not cls_norm:
            scores.append(0)
            continue
        if entity_norm.startswith(cls_norm):
            scores.append(2)   # strongest match
        elif cls_norm in entity_norm:
            scores.append(1)   # partial match
        else:
            scores.append(0)

    best = max(scores)
    if best == 0:
        return None   # no name match, cannot determine preference
    # If two classes score equally, cannot determine preference
    if scores.count(best) > 1:
        return None
    return scores.index(best)


def _make_evidence_based_repairs(
    evidence: Dict[str, Any],
    onto_idx: OntologyIndex,
) -> List[Dict[str, Any]]:
    """
    Dispatch on evidence["error_type"] and produce targeted repair actions.

    NEW types handled (were missing in original, fell through to no_op):
      - domain_violation
      - range_violation
      - class_used_as_individual   <- most common cause of Consistent:False/Unsat:0
      - functional_property_violation
      - property_type_violation
    """
    repairs: list[dict] = []
    etype = evidence.get("error_type", "unknown")

    # ------------------------------------------------------------------ #
    # Disjoint violation — we have the specific individual and both classes
    # ------------------------------------------------------------------ #
    if etype in ("disjoint_violation", "transitive_disjoint_violation"):
        entity = evidence.get("entity")
        entity_label = evidence.get("entity_label", "Unknown")
        classes = evidence.get("classes", [])
        class_labels = evidence.get("class_labels", [])

        # For transitive violations, include the ancestor disjoint pair info
        disjoint_anc_labels = evidence.get("disjoint_ancestor_labels", [])
        extra_info = ""
        if disjoint_anc_labels and len(disjoint_anc_labels) >= 2:
            extra_info = (
                f" (via ancestors {disjoint_anc_labels[0]} _|_ "
                f"{disjoint_anc_labels[1]})"
            )

        if not entity or len(classes) < 2:
            return repairs

        # ── Smart ordering: detect which class is the "intended" type ──────
        # If the entity label contains one of the class labels, that class
        # is the entity's primary identity (e.g. electronic_control_unit_f46b9
        #-> electronic_control_unit is the intended type).
        # Put "remove the OTHER (non-matching) class" first with lower risk
        # so both human and RL see the preferred action upfront.
        intended_idx = _intended_class_index(entity_label, class_labels)

        # Build both removal actions with smart labelling
        remove_actions = []
        for i in range(2):
            other_i = 1 - i
            lbl   = class_labels[i] if i < len(class_labels) else "?"
            is_intended   = (i == intended_idx)
            is_conflicting = (i != intended_idx and intended_idx is not None)

            if is_conflicting:
                # This is the class that should be removed — preferred action
                desc = (
                    f"[RECOMMENDED] Remove '{lbl}' type from {entity_label} "
                    f"-- entity name matches '{class_labels[intended_idx]}', "
                    f"'{lbl}' was likely an incorrect additional type"
                    f"{extra_info}"
                )
                risk        = "low"
                risk_penalty = 0.5   # lower penalty: this is the right move
            elif is_intended:
                # Removing the intended type is riskier — warn the human
                desc = (
                    f"[CAUTION] Remove '{lbl}' type from {entity_label} "
                    f"-- this is the entity's primary identity; "
                    f"only remove if truly wrong"
                    f"{extra_info}"
                )
                risk        = "medium"
                risk_penalty = 3.0
            else:
                # No name match on either side — treat equally
                desc = (
                    f"Remove '{lbl}' type from {entity_label}{extra_info}"
                )
                risk        = "low"
                risk_penalty = 1.0

            remove_actions.append({
                "action_type":   "drop_class_assertion",
                "target_entity": entity,
                "target_class":  classes[i],
                "description":   desc,
                "risk":          risk,
                "risk_penalty":  risk_penalty,
                "intended_type": is_intended,
            })

        # Sort: conflicting (recommended) first, intended (caution) second
        if intended_idx is not None:
            remove_actions.sort(key=lambda a: a.get("intended_type", False))

        repairs.append({
            "error_type":       etype,
            "entity":           entity,
            "entity_label":     entity_label,
            "classes":          classes,
            "class_labels":     class_labels,
            "intended_class":   (class_labels[intended_idx]
                                 if intended_idx is not None else None),
            "evidence":         evidence.get("evidence", []),
            "actions": remove_actions + [
                {
                    "action_type":  "drop_entity",
                    "target":       entity,
                    "description":  (
                        f"[LAST RESORT] Remove {entity_label} entirely"
                        f"{extra_info}"
                    ),
                    "risk":         "high",
                    "risk_penalty": 5.0,
                },
            ],
        })

    # ------------------------------------------------------------------ #
    # Domain / range violations — add a missing rdf:type or remove the triple
    # ------------------------------------------------------------------ #
    elif etype in ("domain_violation", "range_violation"):
        entity = evidence.get("entity")
        entity_label = evidence.get("entity_label", "Unknown")
        prop = evidence.get("property")
        prop_label = evidence.get("property_label", "?")
        expected_type = evidence.get(
            "expected_domain" if etype == "domain_violation" else "expected_range"
        )
        type_label = expected_type.rsplit(
            "#", 1)[-1].rsplit("/", 1)[-1] if expected_type else "?"

        if not entity or not prop:
            return repairs

        actions = []
        # Best fix: add the missing rdf:type assertion
        if expected_type:
            actions.append({
                "action_type":   "add_type_assertion",
                "target_entity": entity,
                "target_class":  expected_type,
                "description":   f"Declare {entity_label} as {type_label} "
                                 f"(satisfies {etype.replace('_', ' ')})",
                "risk":        "low",
                "risk_penalty": 0.5,
            })
        # Fallback: remove the offending property assertion
        actions.append({
            "action_type":    "remove_property_assertion",
            "target_entity":  entity,
            "target_property": prop,
            "description":    f"Remove use of {prop_label} on {entity_label}",
            "risk":           "medium",
            "risk_penalty":    2.0,
        })

        repairs.append({
            "error_type":   etype,
            "entity":       entity,
            "entity_label": entity_label,
            "property":     prop,
            "evidence":     evidence.get("evidence", []),
            "actions":      actions,
        })

    # ------------------------------------------------------------------ #
    # Class used as individual — the ROOT CAUSE of Consistent:False/Unsat:0
    # The class URI appeared as the subject of a property assertion.
    # Fix: mint a proper individual of that class and redirect the triple.
    # ------------------------------------------------------------------ #
    elif etype == "class_used_as_individual":
        entity = evidence.get("entity")          # the class URI
        entity_label = evidence.get("entity_label", "Unknown")
        prop = evidence.get("property")
        prop_label = evidence.get("property_label", "?")

        if not entity or not prop:
            return repairs

        repairs.append({
            "error_type":   "class_used_as_individual",
            "entity":       entity,
            "entity_label": entity_label,
            "property":     prop,
            "evidence":     evidence.get("evidence", []),
            "actions": [
                {
                    # Mint an individual of the class and re-route the triple
                    "action_type":   "mint_individual_for_class",
                    "target_class":  entity,
                    "target_property": prop,
                    "description":   f"Mint individual of {entity_label} "
                                     f"and redirect {prop_label} assertion",
                    "risk":        "low",
                    "risk_penalty": 0.5,
                },
                {
                    # Safer fallback: just drop all triples with the class as subject
                    "action_type": "drop_entity",
                    "target":      entity,
                    "description": f"Remove all assertions using class "
                                   f"{entity_label} as subject",
                    "risk":        "medium",
                    "risk_penalty": 3.0,
                },
            ],
        })

    # ------------------------------------------------------------------ #
    # Functional property violation — remove extra fillers
    # ------------------------------------------------------------------ #
    elif etype == "functional_property_violation":
        entity = evidence.get("entity")
        entity_label = evidence.get("entity_label", "Unknown")
        prop = evidence.get("property")
        prop_label = evidence.get("property_label", "?")
        fillers = evidence.get("fillers", [])

        if not entity or not prop:
            return repairs

        actions = []
        # Remove all but the first filler (keep the one we trust most)
        for extra in fillers[1:]:
            extra_label = extra.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            actions.append({
                "action_type":    "remove_specific_property_assertion",
                "target_entity":  entity,
                "target_property": prop,
                "target_value":   extra,
                "description":    f"Remove extra filler {extra_label} "
                                  f"from {prop_label}",
                "risk":         "low",
                "risk_penalty":  1.0,
            })
        # Nuclear option
        actions.append({
            "action_type":    "remove_property_assertion",
            "target_entity":  entity,
            "target_property": prop,
            "description":    f"Remove ALL {prop_label} assertions on "
                              f"{entity_label}",
            "risk":         "medium",
            "risk_penalty":  3.0,
        })

        repairs.append({
            "error_type":   "functional_property_violation",
            "entity":       entity,
            "entity_label": entity_label,
            "property":     prop,
            "evidence":     evidence.get("evidence", []),
            "actions":      actions,
        })

    # ------------------------------------------------------------------ #
    # Property type mismatch — object property used with a Literal
    # ------------------------------------------------------------------ #
    elif etype == "property_type_violation":
        entity = evidence.get("entity")
        entity_label = evidence.get("entity_label", "Unknown")
        prop = evidence.get("property")
        prop_label = evidence.get("property_label", "?")

        if not entity or not prop:
            return repairs

        repairs.append({
            "error_type":   "property_type_violation",
            "entity":       entity,
            "entity_label": entity_label,
            "property":     prop,
            "evidence":     evidence.get("evidence", []),
            "actions": [
                {
                    "action_type":    "remove_property_assertion",
                    "target_entity":  entity,
                    "target_property": prop,
                    "description":    f"Remove literal value from "
                                      f"ObjectProperty {prop_label}",
                    "risk":         "medium",
                    "risk_penalty":  2.0,
                },
                {
                    "action_type": "drop_entity",
                    "target":      entity,
                    "description": f"Remove {entity_label} and all "
                                   f"its assertions",
                    "risk":        "high",
                    "risk_penalty": 5.0,
                },
            ],
        })

    # ------------------------------------------------------------------ #
    # External type violation — individual typed as external-namespace class
    # Best fix: remove the offending rdf:type assertion
    # ------------------------------------------------------------------ #
    elif etype == "external_type_violation":
        entity = evidence.get("entity", "Unknown")
        entity_label = evidence.get("entity_label", "Unknown")
        # The bad type URI is embedded in the first evidence line
        ev_lines = evidence.get("evidence", [])
        bad_type = evidence.get("bad_type", "")
        if not bad_type and ev_lines:
            # Parse it from "Individual <X> is typed as <Y> from external namespace Z"
            import re
            m = re.search(r"typed as <([^>]+)>",
                          ev_lines[0] if ev_lines else "")
            if m:
                bad_type = m.group(1)

        repairs.append({
            "error_type":   "external_type_violation",
            "entity":       entity,
            "entity_label": entity_label,
            "bad_type":     bad_type,
            "evidence":     ev_lines,
            "actions": [
                {
                    "action_type":  "remove_type",
                    "entity":       entity,
                    "bad_type":     bad_type,
                    "description":  f"Remove external rdf:type <{bad_type.split('/')[-1]}> "
                                    f"from <{entity_label}>",
                    "risk":         "low",
                    "risk_penalty": 0.5,
                },
                {
                    "action_type":  "drop_entity",
                    "entity":       entity,
                    "description":  f"Drop individual <{entity_label}> entirely",
                    "risk":         "medium",
                    "risk_penalty": 2.0,
                },
            ],
        })

    # ------------------------------------------------------------------ #
    # Generic / unknown — at least give the agent ONE real action
    # ------------------------------------------------------------------ #
    else:
        entity = evidence.get("entity", "Unknown")
        label = entity.rsplit("#", 1)[-1].rsplit("/", 1)[-1] if entity else "?"

        repairs.append({
            "error_type": etype,
            "entity":     entity,
            "evidence":   evidence.get("evidence", []),
            "actions": [
                {
                    # Drop the offending entity as a best-effort repair
                    "action_type": "drop_entity",
                    "target":      entity,
                    "description": f"Remove {label} (generic fallback)",
                    "risk":        "high",
                    "risk_penalty": 5.0,
                },
                {
                    "action_type": "no_op",
                    "description": "Skip — manual inspection needed",
                    "risk":        "none",
                    "risk_penalty": 0.0,
                },
            ],
        })

    return repairs
