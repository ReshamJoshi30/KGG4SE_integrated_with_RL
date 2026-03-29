# Scripts/clean_triples.py
"""
Triple cleaning step.

Normalizes, deduplicates, and filters raw LLM-extracted triples:
  - Lowercases and underscores entity names
  - Applies alias normalization
  - Splits compound objects on "and"
  - Drops clause-like entities that are too long to align
  - Removes duplicate triples
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import config
from core.base_step import PipelineStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions  (independently testable)
# ---------------------------------------------------------------------------

def normalize_entity(entity: str) -> str:
    """
    Normalize an entity name to a clean underscore-separated token.

    Steps:
      - Lowercase and strip
      - Replace smart quotes and non-ASCII artifacts
      - Keep only letters, digits, spaces, underscores
      - Convert spaces to underscores
      - Collapse repeated underscores and strip leading/trailing ones

    Args:
        entity: Raw entity string from LLM output.

    Returns:
        Normalized entity string, e.g. ``"temperature_sensor"``.
    """
    entity = str(entity).strip().lower()
    entity = entity.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    entity = re.sub(r"[^a-z0-9_ ]+", "", entity)
    entity = entity.replace(" ", "_")
    entity = re.sub(r"_+", "_", entity).strip("_")
    return entity


def split_object_on_and(obj_norm: str) -> list[str]:
    """
    Split a normalized object token on the word "and" into multiple objects.

    Example:
        ``"data_and_instructions"``-> ``["data", "instructions"]``

    Args:
        obj_norm: Normalized object string.

    Returns:
        List of one or more normalized object strings.
        Returns the original as a single-item list if no split occurs.
    """
    if not obj_norm:
        return []

    txt = obj_norm.replace("_and_", " and ").replace("&", " and ")
    parts = [p.strip() for p in re.split(r"\band\b", txt) if p.strip()]

    if len(parts) <= 1:
        return [obj_norm]

    result = []
    for part in parts:
        normalized = normalize_entity(part)
        if normalized:
            result.append(normalized)

    return result if result else [obj_norm]


def looks_like_clause(entity_norm: str, max_tokens: int = config.CLEAN_MAX_TOKENS) -> bool:
    """
    Heuristic: return True if the entity looks like a long descriptive clause
    rather than a short noun phrase.

    Token count is derived from the number of underscores + 1.

    Args:
        entity_norm: Normalized entity string.
        max_tokens:  Maximum allowed token count. Default from config.

    Returns:
        True if the entity exceeds the token limit.
    """
    if not entity_norm:
        return True
    token_count = entity_norm.count("_") + 1
    return token_count > max_tokens


def clean_triples(
    raw_triples: list[dict],
    aliases: dict | None = None,
    max_tokens: int = config.CLEAN_MAX_TOKENS,
) -> list[dict]:
    """
    Clean and deduplicate a list of raw triple dicts.

    Processing order per triple:
      1. Normalize subject, predicate, object
      2. Apply alias substitution on subject and object
      3. Skip if any part is empty after normalization
      4. Split object on "and" into multiple triples
      5. Drop objects that look like long clauses
      6. Deduplicate by (subject, predicate, object) key

    Args:
        raw_triples: List of dicts with keys: subject, predicate, object.
        aliases:     Optional alias map. Defaults to config.CLEAN_ALIASES.
        max_tokens:  Max tokens in entity name. Defaults to config.CLEAN_MAX_TOKENS.

    Returns:
        List of cleaned, deduplicated triple dicts.
    """
    if aliases is None:
        aliases = config.CLEAN_ALIASES

    cleaned: list[dict] = []
    seen:    set[tuple]  = set()
    skipped_empty    = 0
    skipped_clause   = 0
    skipped_duplicate = 0

    for triple in raw_triples:
        subj = normalize_entity(triple.get("subject",   ""))
        pred = normalize_entity(triple.get("predicate", ""))
        obj_raw = normalize_entity(triple.get("object", ""))

        # Alias substitution — subject
        subj = aliases.get(subj, subj)

        if not subj or not pred or not obj_raw:
            skipped_empty += 1
            logger.debug("Skipped empty triple: %s | %s | %s", subj, pred, obj_raw)
            continue

        # Split compound objects
        obj_list = split_object_on_and(obj_raw)

        for obj in obj_list:
            # Alias substitution — object
            obj = aliases.get(obj, obj)

            if not obj:
                skipped_empty += 1
                continue

            if looks_like_clause(obj, max_tokens=max_tokens):
                skipped_clause += 1
                logger.debug("Skipped clause-like object: '%s'", obj)
                continue

            triple_key = (subj, pred, obj)
            if triple_key in seen:
                skipped_duplicate += 1
                continue

            seen.add(triple_key)
            cleaned.append({"subject": subj, "predicate": pred, "object": obj})

    logger.info(
        "Clean summary — kept: %d | skipped empty: %d | "
        "skipped clause: %d | skipped duplicate: %d",
        len(cleaned), skipped_empty, skipped_clause, skipped_duplicate,
    )

    return cleaned


# ---------------------------------------------------------------------------
# Strategy class  (implements PipelineStep interface)
# ---------------------------------------------------------------------------

class CleanTriplesStep(PipelineStep):
    """
    Pipeline step: raw triples JSON-> cleaned triples JSON.

    Kwargs:
        input_path  (Path | str): Raw triples JSON file.
                                  Default: config.DEFAULT_PATHS["clean_triples"]["input"]
        output_path (Path | str): Cleaned triples JSON file.
                                  Default: config.DEFAULT_PATHS["clean_triples"]["output"]
        max_tokens  (int)       : Max entity token length before clause filter triggers.
                                  Default: config.CLEAN_MAX_TOKENS
        aliases     (dict)      : Custom alias map override.
                                  Default: config.CLEAN_ALIASES
    """

    name: str = "clean_triples"

    def run(self, **kwargs: Any) -> Path:
        """
        Execute the clean-triples step.

        Returns:
            Path to the written cleaned triples JSON file.

        Raises:
            FileNotFoundError : If input JSON does not exist.
            ValueError        : If input JSON has an unexpected structure.
        """
        input_path  = Path(kwargs.get("input_path",  config.DEFAULT_PATHS["clean_triples"]["input"]))
        output_path = Path(kwargs.get("output_path", config.DEFAULT_PATHS["clean_triples"]["output"]))
        max_tokens  = int(kwargs.get("max_tokens", config.CLEAN_MAX_TOKENS))
        aliases     = kwargs.get("aliases", config.CLEAN_ALIASES)

        if not input_path.exists():
            raise FileNotFoundError(f"Raw triples file not found: {input_path}")

        logger.info("[%s] Reading: %s", self.name, input_path)

        with open(input_path, "r", encoding="utf-8") as fh:
            raw_triples = json.load(fh)

        if not isinstance(raw_triples, list):
            raise ValueError(
                f"Expected a JSON list in {input_path}, "
                f"got {type(raw_triples).__name__}"
            )

        logger.info("[%s] Loaded %d raw triple(s)", self.name, len(raw_triples))

        cleaned = clean_triples(
            raw_triples,
            aliases=aliases,
            max_tokens=max_tokens,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(cleaned, fh, indent=2, ensure_ascii=False)

        logger.info(
            "[%s] %d-> %d triple(s) after cleaning-> %s",
            self.name, len(raw_triples), len(cleaned), output_path,
        )

        return output_path