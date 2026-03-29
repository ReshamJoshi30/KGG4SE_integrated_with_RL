# Scripts/check_quality.py
"""
Quality check step.

Computes basic statistics and duplicate analysis over the reasoned
triples JSON and writes a structured quality report.

Designed to be the final step in the pipeline before the RL module.
The report is intentionally schema-stable so the RL module can
consume it programmatically.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import config
from core.base_step import PipelineStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions  (independently testable)
# ---------------------------------------------------------------------------

def compute_stats(triples: list[dict]) -> dict:
    """
    Compute basic coverage statistics over a list of triple dicts.

    Args:
        triples: List of dicts with keys: subject, predicate, object.

    Returns:
        Dict with keys:
          - total_triples     (int)
          - unique_subjects   (int)
          - unique_predicates (int)
          - unique_objects    (int)
          - unique_entities   (int)  subjects ∪ objects
    """
    subjects   = [t["subject"]   for t in triples]
    predicates = [t["predicate"] for t in triples]
    objects    = [t["object"]    for t in triples]

    return {
        "total_triples":     len(triples),
        "unique_subjects":   len(set(subjects)),
        "unique_predicates": len(set(predicates)),
        "unique_objects":    len(set(objects)),
        "unique_entities":   len(set(subjects) | set(objects)),
    }


def find_duplicates(triples: list[dict]) -> list[tuple]:
    """
    Identify duplicate (subject, predicate, object) triples.

    Args:
        triples: List of triple dicts.

    Returns:
        List of (subject, predicate, object) tuples that appear
        more than once. Empty list if no duplicates.
    """
    triple_tuples = [
        (t["subject"], t["predicate"], t["object"])
        for t in triples
    ]
    return [
        item
        for item, count in Counter(triple_tuples).items()
        if count > 1
    ]


def top_n(triples: list[dict], field: str, n: int = 10) -> list[dict]:
    """
    Return the top-N most frequent values for a given triple field.

    Args:
        triples: List of triple dicts.
        field:   One of "subject", "predicate", "object".
        n:       Number of top entries to return.

    Returns:
        List of dicts with keys ``value`` and ``count``,
        sorted descending by count.
    """
    counts = Counter(t[field] for t in triples)
    return [
        {"value": value, "count": count}
        for value, count in counts.most_common(n)
    ]


def build_quality_report(
    triples:    list[dict],
    sample_size: int = 10,
    top_n_size:  int = 10,
) -> dict:
    """
    Build a full quality report dict from a list of triples.

    Report schema:
    ::

        {
          "stats": { ... },
          "duplicate_triples": [ [s, p, o], ... ],
          "top_subjects":    [ {value, count}, ... ],
          "top_predicates":  [ {value, count}, ... ],
          "top_objects":     [ {value, count}, ... ],
          "sample_triples":  [ {subject, predicate, object}, ... ],
        }

    Args:
        triples:     List of triple dicts.
        sample_size: Number of sample triples to include.
        top_n_size:  Number of top-N entries per field.

    Returns:
        Quality report dict.
    """
    stats      = compute_stats(triples)
    duplicates = find_duplicates(triples)

    report = {
        "stats":            stats,
        "duplicate_triples": duplicates,
        "top_subjects":    top_n(triples, "subject",   n=top_n_size),
        "top_predicates":  top_n(triples, "predicate", n=top_n_size),
        "top_objects":     top_n(triples, "object",    n=top_n_size),
        "sample_triples":  triples[:sample_size],
    }

    logger.info(
        "Quality report — total: %d | unique subjects: %d | "
        "unique predicates: %d | duplicates: %d",
        stats["total_triples"],
        stats["unique_subjects"],
        stats["unique_predicates"],
        len(duplicates),
    )

    if duplicates:
        logger.warning("%d duplicate triple(s) detected", len(duplicates))

    return report


# ---------------------------------------------------------------------------
# Strategy class  (implements PipelineStep interface)
# ---------------------------------------------------------------------------

class CheckQualityStep(PipelineStep):
    """
    Pipeline step: reasoned triples JSON-> quality report JSON.

    Kwargs:
        input_path  (Path | str): Reasoned triples JSON file.
                                  Default: config.DEFAULT_PATHS["check_quality"]["input"]
        output_path (Path | str): Quality report JSON file.
                                  Default: config.DEFAULT_PATHS["check_quality"]["output"]
        sample_size (int)       : Number of sample triples in report.
                                  Default: 10
        top_n_size  (int)       : Number of top-N entries per field.
                                  Default: 10
    """

    name: str = "check_quality"

    def run(self, **kwargs: Any) -> Path:
        """
        Execute the check-quality step.

        Returns:
            Path to the written quality report JSON file.

        Raises:
            FileNotFoundError: If the reasoned triples JSON is missing.
            ValueError:        If the JSON is not a list.
        """
        defaults    = config.DEFAULT_PATHS["check_quality"]
        input_path  = Path(kwargs.get("input_path",  defaults["input"]))
        output_path = Path(kwargs.get("output_path", defaults["output"]))
        sample_size = int(kwargs.get("sample_size", 10))
        top_n_size  = int(kwargs.get("top_n_size",  10))

        if not input_path.exists():
            raise FileNotFoundError(f"Reasoned triples file not found: {input_path}")

        logger.info("[%s] Reading: %s", self.name, input_path)

        with open(input_path, "r", encoding="utf-8") as fh:
            triples = json.load(fh)

        if not isinstance(triples, list):
            raise ValueError(
                f"Expected a JSON list in {input_path}, "
                f"got {type(triples).__name__}"
            )

        logger.info("[%s] Loaded %d triple(s)", self.name, len(triples))

        report = build_quality_report(
            triples,
            sample_size=sample_size,
            top_n_size=top_n_size,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)

        logger.info(
            "[%s] Quality report-> %s",
            self.name, output_path,
        )

        return output_path