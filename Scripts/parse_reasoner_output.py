# Scripts/parse_reasoner_output.py
"""
Reasoner output parsing step.

Parses the reasoned OWL file (RDF/XML) into a flat JSON list of
subject / predicate / object triples for downstream consumption
by the quality check and RL modules.
"""

import json
import logging
from pathlib import Path
from typing import Any

from rdflib import Graph

import config
from core.base_step import PipelineStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions  (independently testable)
# ---------------------------------------------------------------------------

def parse_rdf_to_triples(graph: Graph) -> list[dict]:
    """
    Convert an RDF graph into a flat list of triple dicts.

    Each triple is represented as:
    ``{"subject": str, "predicate": str, "object": str}``

    All URIs and literals are cast to plain strings so the output
    is JSON-serializable without further processing.

    Args:
        graph: Parsed RDF graph.

    Returns:
        List of triple dicts.
    """
    triples = [
        {
            "subject":   str(s),
            "predicate": str(p),
            "object":    str(o),
        }
        for s, p, o in graph
    ]
    logger.debug("Parsed %d triple(s) from RDF graph", len(triples))
    return triples


def load_reasoned_owl(input_path: Path) -> Graph:
    """
    Load a reasoned OWL file into an RDF graph.

    Attempts RDF/XML first, falls back to Turtle.

    Args:
        input_path: Path to the reasoned OWL file.

    Returns:
        Parsed RDF Graph.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        RuntimeError:      If the file cannot be parsed in any format.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Reasoned OWL file not found: {input_path}")

    graph = Graph()

    try:
        graph.parse(str(input_path), format="xml")
        logger.debug("Reasoned OWL parsed as RDF/XML")
        return graph
    except Exception as xml_exc:
        logger.debug("RDF/XML parse failed (%s), trying Turtle...", xml_exc)

    try:
        graph.parse(str(input_path), format="turtle")
        logger.debug("Reasoned OWL parsed as Turtle")
        return graph
    except Exception as ttl_exc:
        raise RuntimeError(
            f"Could not parse {input_path} as RDF/XML or Turtle.\n"
            f"RDF/XML error : {xml_exc}\n"
            f"Turtle error  : {ttl_exc}"
        ) from ttl_exc


# ---------------------------------------------------------------------------
# Strategy class  (implements PipelineStep interface)
# ---------------------------------------------------------------------------

class ParseReasonerOutputStep(PipelineStep):
    """
    Pipeline step: reasoned OWL-> flat triples JSON.

    Kwargs:
        input_path  (Path | str): Reasoned OWL file.
                                  Default: config.DEFAULT_PATHS["parse_reasoner"]["input"]
        output_path (Path | str): Output JSON file.
                                  Default: config.DEFAULT_PATHS["parse_reasoner"]["output"]
    """

    name: str = "parse_reasoner"

    def run(self, **kwargs: Any) -> Path:
        """
        Execute the parse-reasoner-output step.

        Returns:
            Path to the written triples JSON file.

        Raises:
            FileNotFoundError: If the reasoned OWL file is missing.
            RuntimeError:      If the OWL file cannot be parsed.
        """
        defaults    = config.DEFAULT_PATHS["parse_reasoner"]
        input_path  = Path(kwargs.get("input_path",  defaults["input"]))
        output_path = Path(kwargs.get("output_path", defaults["output"]))

        logger.info(
            "[%s] input=%s | output=%s",
            self.name, input_path, output_path,
        )

        graph   = load_reasoned_owl(input_path)
        triples = parse_rdf_to_triples(graph)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(triples, fh, indent=2, ensure_ascii=False)

        logger.info(
            "[%s] Parsed %d triple(s)-> %s",
            self.name, len(triples), output_path,
        )

        return output_path