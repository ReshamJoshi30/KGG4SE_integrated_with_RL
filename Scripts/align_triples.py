# Scripts/align_triples.py
"""
Triple alignment step.

Maps raw cleaned triples to ontology URIs using:
  1. CSV relation/entity maps   (exact lookup — highest priority)
  2. Exact label/localname match against ontology index
  3. Fuzzy match via RapidFuzz  (optional — falls back gracefully)

Writes:
  - Aligned triples as Turtle (.ttl)
  - Alignment report as CSV
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from rdflib import Graph, Namespace, OWL, RDF, RDFS, URIRef

import config
from core.base_step import PipelineStep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional fuzzy matching
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import process as _fuzz_process
    _FUZZY_AVAILABLE = True
    logger.debug("RapidFuzz available — fuzzy matching enabled")
except ImportError:
    _fuzz_process = None          # type: ignore[assignment]
    _FUZZY_AVAILABLE = False
    logger.debug(
        "RapidFuzz not installed — falling back to exact matching only")


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """
    Normalize a string for index lookup:
    lowercase, collapse underscores/hyphens to spaces, collapse whitespace,
    and strip leading/trailing whitespace.

    Args:
        s: Raw string.

    Returns:
        Normalized string.
    """
    s = str(s).strip().lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+",    " ", s)
    return s.strip()


def _localname(uri: URIRef) -> str:
    """
    Extract the local name from a URI.

    Args:
        uri: RDF URI reference.

    Returns:
        Local name string.
    """
    u = str(uri)
    if "#" in u:
        return u.rsplit("#", 1)[1]
    return u.rstrip("/").rsplit("/", 1)[-1]


def load_csv_map(path: Path, key_col: str, val_col: str) -> dict[str, str]:
    """
    Load a two-column CSV into a lowercase-keyed dict.

    Args:
        path:    Path to CSV file. Returns empty dict if file does not exist.
        key_col: Column name to use as dict key.
        val_col: Column name to use as dict value (URI string).

    Returns:
        Dict mapping normalized key-> URI string.
    """
    mapping: dict[str, str] = {}
    if not path.exists():
        logger.warning("Map file not found (skipping): %s", path)
        return mapping

    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get(key_col) or "").strip()
            uri = (row.get(val_col) or "").strip()
            if raw and uri:
                mapping[_norm(raw)] = uri

    logger.info("Loaded %d mapping(s) from %s", len(mapping), path)
    return mapping


def build_ontology_index(g: Graph) -> tuple[dict[str, URIRef], dict[str, URIRef]]:
    """
    Build normalized lookup indices from an OWL ontology graph.

    Indices are keyed by normalized label / local name strings,
    mapped to URIRefs.

    Args:
        g: Parsed RDF graph containing OWL classes and properties.

    Returns:
        Tuple of (entities_index, properties_index).
    """
    entities: dict[str, URIRef] = {}
    props:    dict[str, URIRef] = {}

    # Classes
    for cls in g.subjects(RDF.type, OWL.Class):
        if isinstance(cls, URIRef):
            # Index by local name
            local = _localname(cls)
            entities[_norm(local)] = cls

            # Index by rdfs:label
            for label_obj in g.objects(cls, RDFS.label):
                label_str = str(label_obj)
                entities[_norm(label_str)] = cls

    # Object + Datatype properties
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(prop, URIRef):
            local = _localname(prop)
            props[_norm(local)] = prop
            for label_obj in g.objects(prop, RDFS.label):
                props[_norm(str(label_obj))] = prop

    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(prop, URIRef):
            local = _localname(prop)
            props[_norm(local)] = prop
            for label_obj in g.objects(prop, RDFS.label):
                props[_norm(str(label_obj))] = prop

    logger.info(
        "Ontology index built — entities: %d | properties: %d",
        len(entities), len(props),
    )

    return entities, props


def best_match(
    raw: str,
    index: dict[str, URIRef],
    cutoff: int = config.FUZZY_CUTOFF,
) -> tuple[URIRef | None, float, str]:
    """
    Find the best ontology URI match for a raw text string.

    Attempt order:
      1. Exact match (score = 100.0)
      2. Fuzzy match if RapidFuzz is available (score >= cutoff)

    Args:
        raw:    Raw text to match.
        index:  Ontology index (normalized key-> URI).
        cutoff: Minimum fuzzy match score (0–100).

    Returns:
        Tuple of (matched_uri, score, method) where method is
        "exact" | "fuzzy" | "none".
    """
    norm = _norm(raw)

    # Exact match
    if norm in index:
        return index[norm], 100.0, "exact"

    # Fuzzy match
    if not _FUZZY_AVAILABLE or not index:
        return None, 0.0, "none"

    # Use RapidFuzz
    result = _fuzz_process.extractOne(
        norm,
        index.keys(),
        score_cutoff=cutoff,
    )

    if result is None:
        return None, 0.0, "none"

    matched_key, score, _ = result
    return index[matched_key], float(score), "fuzzy"


# ---------------------------------------------------------------------------
# FIX 1: helper to detect class URIs (needed by ensure_individual + ent_map guard)
# ---------------------------------------------------------------------------

def _is_class_uri(uri: URIRef, entities: dict[str, URIRef]) -> bool:
    """
    Return True iff *uri* is an OWL class in the ontology index.

    Individuals minted under BASE_INDIVIDUAL_NS are never classes.
    """
    if str(uri).startswith(config.BASE_INDIVIDUAL_NS):
        return False
    return uri in entities.values()


def _build_domain_range_index(g: Graph) -> dict:
    """
    Build {property_uri: {"domain": [class_uri,...], "range": [class_uri,...]}}
    from the ontology graph. Used to auto-satisfy domain/range constraints.
    """
    index = {}
    for p in g.subjects(RDF.type, OWL.ObjectProperty):
        p_str = str(p)
        domains = [str(d) for d in g.objects(p, RDFS.domain)
                   if str(d).startswith("http")]
        ranges = [str(r) for r in g.objects(p, RDFS.range)
                  if str(r).startswith("http")]
        if domains or ranges:
            index[p_str] = {"domain": domains, "range": ranges}
    return index


def _build_subclass_index(g: Graph) -> dict[str, set[str]]:
    """
    Build {class_uri: {all_ancestor_class_uris}} (transitive closure).
    Used to check if a type already satisfies a domain/range requirement
    through inheritance.
    """
    direct = {}
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            direct.setdefault(str(s), set()).add(str(o))

    # Transitive closure via DFS
    memo: dict[str, set[str]] = {}

    def ancestors(cls: str) -> set[str]:
        if cls in memo:
            return memo[cls]
        result = {cls}
        for parent in direct.get(cls, set()):
            result |= ancestors(parent)
        memo[cls] = result
        return result

    return {cls: ancestors(cls) for cls in direct}


def _satisfies_type(
    ind_uri: URIRef,
    required_class: str,
    graph: Graph,
    subclass_idx: dict[str, set[str]],
) -> bool:
    """
    Return True if *ind_uri* is already typed as *required_class* or any
    subclass of it (direct or inferred via rdfs:subClassOf).
    """
    for _, _, t in graph.triples((ind_uri, RDF.type, None)):
        t_str = str(t)
        if t_str == required_class:
            return True
        # Check if the type is a subclass of required_class
        if required_class in subclass_idx.get(t_str, set()):
            return True
    return False


def _ensure_domain_range(
    subj: URIRef,
    prop: URIRef,
    obj: URIRef,
    graph: Graph,
    domain_range_idx: dict,
    subclass_idx: dict[str, set[str]],
) -> bool:
    """
    Check domain/range constraints and auto-add missing rdf:type to
    satisfy them.  Always returns True (never skips a triple).

    The reasoner step downstream is responsible for detecting real
    disjoint conflicts — the alignment step should maximise recall so
    the RL repair loop has material to work with.
    """
    entry = domain_range_idx.get(str(prop))
    if not entry:
        return True  # no constraints — always safe

    prop_name = str(prop).split("#")[-1].split("/")[-1]

    # --- Domain check: auto-add missing type to subject ---
    for dom in entry.get("domain", []):
        if not _satisfies_type(subj, dom, graph, subclass_idx):
            graph.add((subj, RDF.type, URIRef(dom)))
            logger.debug(
                "Auto-typed subject <%s> as <%s> to satisfy domain of <%s>",
                str(subj).split("/")[-1], dom.split("#")[-1], prop_name,
            )

    # --- Range check: auto-add missing type to object ---
    if isinstance(obj, URIRef):
        for rng in entry.get("range", []):
            if not _satisfies_type(obj, rng, graph, subclass_idx):
                graph.add((obj, RDF.type, URIRef(rng)))
                logger.debug(
                    "Auto-typed object <%s> as <%s> (range of <%s>)",
                    str(obj).split("/")[-1], rng.split("#")[-1], prop_name,
                )

    return True  # never skip — let the reasoner detect real conflicts


# ---------------------------------------------------------------------------
# FIX 2: ensure_individual — stable hash-based IRI, no counter reset bug
# ---------------------------------------------------------------------------

def ensure_individual(
    uri: URIRef,
    entity_type: str,
    entities: dict[str, URIRef],
    aligned_graph: Graph,
    base_ns: str,
    counter: dict,                  # kept for API compat, no longer used
    source_text: str = "",
) -> URIRef:
    """
    Convert a class URI to a stable individual URI.

    FIX — replaces integer counter with MD5(class_name + source_text).
    Same source text always produces the same IRI across pipeline runs,
    so re-runs never accumulate duplicate individuals that violate
    functional-property or cardinality constraints.
    """
    import hashlib as _hashlib
    if _is_class_uri(uri, entities):
        class_name = _localname(uri)
        salt = f"{class_name}|{source_text.strip().lower()}"
        uid = _hashlib.md5(salt.encode("utf-8")).hexdigest()[:8]
        ind_uri = URIRef(f"{base_ns}{class_name}_{uid}")
        if (ind_uri, RDF.type, uri) not in aligned_graph:
            aligned_graph.add((ind_uri, RDF.type, uri))
            logger.debug(
                "Minted individual <%s> of type <%s> (source=%r)",
                ind_uri, uri, source_text,
            )
        return ind_uri
    return uri


def _ent_map_uri_is_safe(uri_str: str, entities: dict[str, URIRef]) -> bool:
    """
    Return True if a URI from entity_map.csv is safe to use directly.

    A URI is safe if it appears in the ontology's entity index OR
    belongs to the project's own GBO / GENIALOnt namespaces.
    Only truly alien namespaces (not part of the loaded ontology) are rejected.
    """
    uri_ref = URIRef(uri_str)
    if uri_ref in entities.values():
        return True
    # Accept URIs from the project's own namespaces
    if uri_str.startswith("http://w3id.org/gbo"):
        return True
    if uri_str.startswith("http://cpsagila.cs.uni-kl.de/GENIALOnt"):
        return True
    return False


# Module-level memo so it persists across calls within one pipeline run
_EXT_ANCESTOR_MEMO: dict[str, bool] = {}


def _class_has_external_ancestor(
    cls_uri: str,
    onto_graph: Graph,
    external_ns: tuple = (
        "http://w3id.org/gbo/",
        "http://www.w3.org/ns/sosa/",
        "http://www.ontology-of-units-of-measure.org/",
    ),
    _depth: int = 0,
) -> bool:
    """
    Return True if cls_uri or ANY transitive rdfs:subClassOf ancestor
    has a URI from a known-problematic external namespace.

    Uses full transitive closure (depth-limited to 20) with memoisation.
    """
    if _depth > 20:
        return False  # cycle guard

    if cls_uri in _EXT_ANCESTOR_MEMO:
        return _EXT_ANCESTOR_MEMO[cls_uri]

    # Check self
    if any(cls_uri.startswith(ns) for ns in external_ns):
        _EXT_ANCESTOR_MEMO[cls_uri] = True
        return True

    # Check all parents recursively
    for parent in onto_graph.objects(URIRef(cls_uri), RDFS.subClassOf):
        if isinstance(parent, URIRef):
            if _class_has_external_ancestor(str(parent), onto_graph,
                                            external_ns, _depth + 1):
                _EXT_ANCESTOR_MEMO[cls_uri] = True
                return True

    _EXT_ANCESTOR_MEMO[cls_uri] = False
    return False


def align_triples(
    triples:    list[tuple[str, str, str]],
    entities:   dict[str, URIRef],
    props:      dict[str, URIRef],
    rel_map:    dict[str, str],
    ent_map:    dict[str, str],
    base_ns:    str = config.BASE_INDIVIDUAL_NS,
    fuzzy_cutoff: int = config.FUZZY_CUTOFF,
    allow_create: bool = config.ALLOW_CREATE_INDIVIDUALS,
    ontology_path: str | Path | None = None,
) -> tuple[Graph, list[list]]:
    """
    Align raw (subject, predicate, object) text triples to ontology URIs.

    Args:
        ontology_path: Path to the OWL ontology file for domain/range
                       index building. Falls back to config default if
                       not provided.
    """
    import hashlib as _hashlib

    aligned_graph = Graph()
    aligned_graph.bind("rdfs", RDFS)
    aligned_graph.bind("owl",  OWL)
    aligned_graph.bind("rdf",  RDF)

    # Build domain/range and subclass indexes for constraint satisfaction
    _ont_src = str(ontology_path) if ontology_path else str(
        config.DEFAULT_PATHS["align_triples"]["ontology"])
    _onto_graph = Graph()
    try:
        _onto_graph.parse(_ont_src, format="xml")
    except Exception:
        try:
            _onto_graph.parse(_ont_src, format="turtle")
        except Exception:
            pass  # indexes will be empty — domain checks skipped gracefully

    _domain_range_idx = _build_domain_range_index(_onto_graph)
    _subclass_idx = _build_subclass_index(_onto_graph)
    logger.debug("Domain/range index: %d properties", len(_domain_range_idx))

    report_rows = [[
        "subject_text", "predicate_text", "object_text",
        "subject_uri",  "predicate_uri",  "object_uri",
        "s_score", "p_score", "o_score", "status",
    ]]

    aligned = skipped = created = 0
    individual_counter = {}   # kept for log counter only

    for s_txt, p_txt, o_txt in triples:
        if not (s_txt and p_txt and o_txt):
            report_rows.append(
                [s_txt, p_txt, o_txt, "", "", "", "", "", "", "skip:empty"])
            skipped += 1
            continue

        # --- Predicate resolution ---
        p_uri:   URIRef | None = None
        p_score: float = 0.0

        mapped_pred = rel_map.get(str(p_txt).strip().lower())
        if mapped_pred:
            p_uri, p_score = URIRef(mapped_pred), 100.0
        else:
            p_uri, p_score, _ = best_match(p_txt, props, cutoff=fuzzy_cutoff)

        if not p_uri:
            report_rows.append([
                s_txt, p_txt, o_txt, "", "", "",
                "", p_score, "", "skip:predicate_unmatched",
            ])
            skipped += 1
            logger.debug("Predicate unmatched: '%s'", p_txt)
            continue

        # --- Subject resolution ---
        # FIX 3: ent_map may map text to a class URI — guard immediately
        s_uri:   URIRef | None = None
        s_score: float = 0.0

        mapped_subj = ent_map.get(str(s_txt).strip().lower())
        if mapped_subj and _ent_map_uri_is_safe(mapped_subj, entities):
            s_uri = URIRef(mapped_subj)
            s_score = 100.0
            if _is_class_uri(s_uri, entities):
                s_uri = ensure_individual(
                    s_uri, "subject", entities, aligned_graph,
                    base_ns, individual_counter, source_text=s_txt,
                )
        else:
            if mapped_subj and not _ent_map_uri_is_safe(mapped_subj, entities):
                logger.warning(
                    "ent_map subject URI <%s> is not in ontology index — "
                    "falling back to fuzzy match for %r",
                    mapped_subj, s_txt,
                )
            s_uri, s_score, _ = best_match(
                s_txt, entities, cutoff=fuzzy_cutoff)

        # --- Object resolution ---
        # FIX 4: same guard for object position
        o_uri:   URIRef | None = None
        o_score: float = 0.0

        mapped_obj = ent_map.get(str(o_txt).strip().lower())
        if mapped_obj and _ent_map_uri_is_safe(mapped_obj, entities):
            o_uri = URIRef(mapped_obj)
            o_score = 100.0
            if _is_class_uri(o_uri, entities):
                o_uri = ensure_individual(
                    o_uri, "object", entities, aligned_graph,
                    base_ns, individual_counter, source_text=o_txt,
                )
        else:
            if mapped_obj and not _ent_map_uri_is_safe(mapped_obj, entities):
                logger.warning(
                    "ent_map object URI <%s> is not in ontology index — "
                    "falling back to fuzzy match for %r",
                    mapped_obj, o_txt,
                )
            o_uri, o_score, _ = best_match(
                o_txt, entities, cutoff=fuzzy_cutoff)

        status = "ok"

        # --- Handle unmatched subject ---
        if not s_uri:
            if allow_create:
                s_uri = URIRef(
                    base_ns + re.sub(r"[^A-Za-z0-9]+", "_", s_txt).strip("_")
                )
                created += 1
                status = "ok:subject_minted"
                # FIX 5: minted subjects need rdf:type or HermiT fires domain violations
                _s_cls, _s_sc, _ = best_match(s_txt, entities, cutoff=50)
                _s_type = _s_cls if _s_cls else OWL.Thing
                aligned_graph.add((s_uri, RDF.type, _s_type))
                logger.debug("Minted subject <%s> typed as <%s>",
                             s_uri, _s_type)
            else:
                report_rows.append([
                    s_txt, p_txt, o_txt,
                    "", str(p_uri), "",
                    s_score, p_score, o_score,
                    "skip:subject_unmatched",
                ])
                skipped += 1
                continue

        # --- Handle unmatched object ---
        if not o_uri:
            if allow_create:
                o_uri = URIRef(
                    base_ns + re.sub(r"[^A-Za-z0-9]+", "_", o_txt).strip("_")
                )
                created += 1
                status = status + "|ok:object_minted" if status != "ok" else "ok:object_minted"
                # FIX 6: same — minted objects need rdf:type
                _o_cls, _o_sc, _ = best_match(o_txt, entities, cutoff=50)
                _o_type = _o_cls if _o_cls else OWL.Thing
                aligned_graph.add((o_uri, RDF.type, _o_type))
                logger.debug("Minted object <%s> typed as <%s>",
                             o_uri, _o_type)
            else:
                report_rows.append([
                    s_txt, p_txt, o_txt,
                    str(s_uri), str(p_uri), "",
                    s_score, p_score, o_score,
                    "skip:object_unmatched",
                ])
                skipped += 1
                continue

        # FIX 7: pass source_text so hash is stable across runs
        s_ind = ensure_individual(
            s_uri, "subject", entities, aligned_graph,
            base_ns, individual_counter, source_text=s_txt,
        )
        o_ind = ensure_individual(
            o_uri, "object", entities, aligned_graph,
            base_ns, individual_counter, source_text=o_txt,
        )

        # FIX 8: guarantee every individual has at least one rdf:type.
        # If ensure_individual returned a URI with no type (e.g. ent_map
        # pointed at a URI not found in entities.values()), add one now.
        for _ind, _txt in ((s_ind, s_txt), (o_ind, o_txt)):
            if not list(aligned_graph.triples((_ind, RDF.type, None))):
                _cls, _sc, _ = best_match(_txt, entities, cutoff=50)
                _t = _cls if _cls else OWL.Thing
                aligned_graph.add((_ind, RDF.type, _t))
                logger.debug(
                    "FIX8: Added missing rdf:type <%s> to <%s>",
                    str(_t).split("#")[-1], str(_ind).split("/")[-1],
                )

        # Domain/range constraint check — auto-adds missing types.
        _safe = _ensure_domain_range(
            s_ind, p_uri, o_ind,
            aligned_graph, _domain_range_idx, _subclass_idx,
        )

        if not _safe:
            skipped += 1
            report_rows.append([
                s_txt, p_txt, o_txt,
                str(s_ind), str(p_uri), str(o_ind),
                s_score, p_score, o_score,
                "skip:domain_range_conflict",
            ])
            continue

        # Add the assertion with individuals
        aligned_graph.add((s_ind, p_uri, o_ind))
        aligned += 1

        report_rows.append([
            s_txt, p_txt, o_txt,
            str(s_ind), str(p_uri), str(o_ind),
            s_score, p_score, o_score,
            status,
        ])

    total = aligned + skipped
    rate = (aligned / total * 100) if total > 0 else 0.0

    logger.info(
        "Alignment complete — aligned: %d | skipped: %d | "
        "minted: %d | individuals created: %d | rate: %.1f%%",
        aligned, skipped, created, sum(individual_counter.values()), rate,
    )

    return aligned_graph, report_rows


# ---------------------------------------------------------------------------
# Strategy class  (implements PipelineStep interface)
# ---------------------------------------------------------------------------

class AlignTriplesStep(PipelineStep):
    """
    Pipeline step: cleaned triples JSON-> aligned Turtle + CSV report.

    Kwargs:
        input_path    (Path | str): Cleaned triples JSON.
        output_path   (Path | str): Output Turtle (.ttl) file.
        ontology_path (Path | str): OWL ontology file.
        rel_map_path  (Path | str): Predicate CSV map file.
        ent_map_path  (Path | str): Entity CSV map file.
        report_path   (Path | str): Alignment report CSV.
        fuzzy_cutoff  (int)       : Fuzzy match threshold (0–100).
        allow_create  (bool)      : Mint URIs for unmatched entities.
    """

    name: str = "align_triples"

    def run(self, **kwargs: Any) -> Path:
        """
        Execute the align-triples step.

        Returns:
            Path to the written Turtle file.

        Raises:
            FileNotFoundError: If any required input file is missing.
            ValueError:        If the triples JSON has an unsupported shape.
        """
        defaults = config.DEFAULT_PATHS["align_triples"]
        input_path = Path(kwargs.get("input_path",    defaults["input"]))
        output_path = Path(kwargs.get("output_path",   defaults["output"]))
        ont_path = Path(kwargs.get("ontology_path", defaults["ontology"]))
        rel_map_path = Path(kwargs.get("rel_map_path",  defaults["rel_map"]))
        ent_map_path = Path(kwargs.get("ent_map_path",  defaults["ent_map"]))
        report_path = Path(kwargs.get("report_path",   defaults["report"]))
        fuzzy_cutoff = int(kwargs.get("fuzzy_cutoff",   config.FUZZY_CUTOFF))
        allow_create = bool(kwargs.get(
            "allow_create",  config.ALLOW_CREATE_INDIVIDUALS))

        for required in (input_path, ont_path):
            if not required.exists():
                raise FileNotFoundError(f"Required file not found: {required}")

        # Load ontology
        logger.info("[%s] Parsing ontology: %s", self.name, ont_path)
        g = Graph()
        try:
            g.parse(str(ont_path), format="xml")
            logger.debug("Ontology parsed as RDF/XML")
        except Exception:
            g.parse(str(ont_path), format="turtle")
            logger.debug("Ontology parsed as Turtle")

        entities, props = build_ontology_index(g)

        # Load CSV maps
        rel_map = load_csv_map(rel_map_path, "raw", "pref_uri")
        ent_map = load_csv_map(ent_map_path, "raw", "pref_uri")

        # Load triples JSON
        logger.info("[%s] Reading: %s", self.name, input_path)
        with open(input_path, "r", encoding="utf-8") as fh:
            raw_triples = json.load(fh)

        if not isinstance(raw_triples, list):
            raise ValueError(
                f"Expected a JSON list in {input_path}, "
                f"got {type(raw_triples).__name__}"
            )

        logger.info("[%s] Loaded %d triple(s)", self.name, len(raw_triples))

        # Convert to tuples
        triples_tuples = [
            (t.get("subject", ""), t.get("predicate", ""), t.get("object", ""))
            for t in raw_triples
        ]

        # Align
        aligned_graph, report_rows = align_triples(
            triples_tuples,
            entities,
            props,
            rel_map,
            ent_map,
            base_ns=config.BASE_INDIVIDUAL_NS,
            fuzzy_cutoff=fuzzy_cutoff,
            allow_create=allow_create,
            ontology_path=ont_path,
        )

        # Write Turtle
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_graph.serialize(destination=str(output_path), format="turtle")
        logger.info("[%s] Aligned triples-> %s", self.name, output_path)

        # Write CSV report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(report_rows)

        logger.info("[%s] Alignment report-> %s", self.name, report_path)

        return output_path
