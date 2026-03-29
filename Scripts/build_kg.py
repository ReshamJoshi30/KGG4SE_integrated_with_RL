# Scripts/build_kg.py
"""
Knowledge Graph build step.

Merges the base ontology with the aligned triples (Turtle) into a
single RDF graph and serializes it as both OWL/XML and Turtle.
"""

import logging
from pathlib import Path
from typing import Any

from rdflib import Graph

import config
from core.base_step import PipelineStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure function  (independently testable)
# ---------------------------------------------------------------------------

def merge_graphs(
    ontology_path: Path,
    aligned_ttl_path: Path,
) -> Graph:
    """
    Parse and merge the base ontology with aligned triples.

    Args:
        ontology_path:    Path to the base OWL ontology file.
        aligned_ttl_path: Path to the aligned triples Turtle file.

    Returns:
        Merged RDF Graph.

    Raises:
        FileNotFoundError: If either input file does not exist.
        Exception:         If parsing fails for both RDF/XML and Turtle.
    """
    for path in (ontology_path, aligned_ttl_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load base ontology — try RDF/XML first, fall back to Turtle
    logger.info("Loading base ontology: %s", ontology_path)
    base_graph = Graph()
    try:
        base_graph.parse(str(ontology_path), format="xml")
        logger.debug("Base ontology parsed as RDF/XML")
    except Exception:
        base_graph.parse(str(ontology_path), format="turtle")
        logger.debug("Base ontology parsed as Turtle")

    logger.info("Base ontology triples: %d", len(base_graph))

    # Load aligned triples
    logger.info("Loading aligned triples: %s", aligned_ttl_path)
    aligned_graph = Graph()
    aligned_graph.parse(str(aligned_ttl_path), format="turtle")
    logger.info("Aligned triples: %d", len(aligned_graph))

    # Merge
    merged = base_graph + aligned_graph
    logger.info("Merged graph triples: %d", len(merged))

    return merged


def serialize_graph(
    graph: Graph,
    output_owl: Path,
    output_ttl: Path,
) -> None:
    """
    Serialize merged RDF graph to OWL/XML and Turtle.
    
    FIXED: Use rdflib directly for both formats.
    The owlready2 round-trip was causing ontology collapse by incorrectly
    inferring that all classes are equivalent.

    Args:
        graph:      The RDF graph to serialize.
        output_owl: Destination path for OWL/XML output.
        output_ttl: Destination path for Turtle output.
    """
    output_owl.parent.mkdir(parents=True, exist_ok=True)
    output_ttl.parent.mkdir(parents=True, exist_ok=True)

    # Turtle for debugging/visualization
    graph.serialize(destination=str(output_ttl), format="turtle")
    logger.info("Merged TTL saved-> %s", output_ttl)
    
    # OWL/XML for Konclude - direct rdflib serialization
    # This avoids the owlready2 load/save cycle that was causing collapse
    graph.serialize(destination=str(output_owl), format="xml")
    logger.info("Merged OWL saved-> %s", output_owl)


# ---------------------------------------------------------------------------
# Strategy class  (implements PipelineStep interface)
# ---------------------------------------------------------------------------

class BuildKGStep(PipelineStep):
    """
    Pipeline step: ontology + aligned triples-> merged KG (OWL + TTL).

    Kwargs:
        ontology_path    (Path | str): Base OWL ontology file.
                                       Default: config.DEFAULT_PATHS["build_kg"]["ontology"]
        input_path       (Path | str): Aligned triples Turtle file.
                                       Default: config.DEFAULT_PATHS["build_kg"]["input"]
        output_owl_path  (Path | str): Output OWL/XML file.
                                       Default: config.DEFAULT_PATHS["build_kg"]["output_owl"]
        output_ttl_path  (Path | str): Output Turtle file.
                                       Default: config.DEFAULT_PATHS["build_kg"]["output_ttl"]
    """

    name: str = "build_kg"

    def run(self, **kwargs: Any) -> Path:
        """
        Execute the build-kg step.

        Returns:
            Path to the written OWL file (primary output).

        Raises:
            FileNotFoundError: If ontology or aligned triples file is missing.
        """
        defaults        = config.DEFAULT_PATHS["build_kg"]
        ontology_path   = Path(kwargs.get("ontology_path",   defaults["ontology"]))
        input_path      = Path(kwargs.get("input_path",      defaults["input"]))
        output_owl_path = Path(kwargs.get("output_owl_path", defaults["output_owl"]))
        output_ttl_path = Path(kwargs.get("output_ttl_path", defaults["output_ttl"]))

        logger.info(
            "[%s] ontology=%s | aligned=%s",
            self.name, ontology_path, input_path,
        )

        merged = merge_graphs(
            ontology_path=ontology_path,
            aligned_ttl_path=input_path,
        )

        serialize_graph(
            graph=merged,
            output_owl=output_owl_path,
            output_ttl=output_ttl_path,
        )

        return output_owl_path