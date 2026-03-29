# llm_kg/generate_triples.py
"""
Triple generation step.
Sends text corpus to a local Ollama/LLM instance and extracts
Subject | Predicate | Object triples.

Merges the original main.py + llm_kg/generate_triples.py into one module.
No hardcoded paths — all config comes from config.py or kwargs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

import config
from core.base_step import PipelineStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_garbage(term: str) -> bool:
    """
    Returns True if the term contains noise / financial / market data.

    Args:
        term: Raw entity string to check.

    Returns:
        True if the term should be filtered out.
    """
    t = term.lower().strip()
    for g in config.GARBAGE_TERMS:
        if g in t:
            return True
    # Filter numeric-heavy tokens  e.g. "usd_75_bn", "75_percent"
    digit_count = sum(1 for c in t if c.isdigit())
    if digit_count > 2:
        return True
    return False


def _query_openai(
    prompt: str,
    model: str = config.OPENAI_MODEL,
    api_key: str = "",
    timeout: int = config.LLM_TIMEOUT,
) -> str:
    """
    Send a prompt to the OpenAI Chat Completions API and return the text.

    Requires:  pip install openai

    Args:
        prompt:  Full prompt string.
        model:   OpenAI model ID ("gpt-4o-mini" or "gpt-4o").
        api_key: OpenAI API key.  Falls back to OPENAI_API_KEY env var.
        timeout: Request timeout in seconds.

    Returns:
        Raw text response from the model.

    Raises:
        RuntimeError: If the openai package is missing or the call fails.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI package not installed. Run:  pip install openai"
        )

    key = api_key or config.OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment "
            "variable:  set OPENAI_API_KEY=sk-..."
        )

    client = OpenAI(api_key=key, timeout=timeout)
    logger.debug("Sending prompt to OpenAI (model=%s)", model)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge graph expert specialising in "
                        "automotive electrical systems. Extract precise "
                        "Subject | Predicate | Object triples from the text. "
                        "Output ONLY the triples, one per line, no other text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,   # deterministic for reproducibility
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    return response.choices[0].message.content or ""


def _query_ollama(
    prompt: str,
    model: str = config.MODEL_NAME,
    ollama_url: str = config.OLLAMA_URL,
    timeout: int = config.LLM_TIMEOUT,
) -> str:
    """
    Sends a prompt to Ollama with streaming disabled.

    Args:
        prompt:     Full prompt string.
        model:      Ollama model name.
        ollama_url: Ollama API endpoint.
        timeout:    Request timeout in seconds.

    Returns:
        Raw LLM response string.

    Raises:
        RuntimeError: On HTTP error or unexpected response shape.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}

    logger.debug("Sending prompt to Ollama (model=%s, url=%s)",
                 model, ollama_url)

    try:
        resp = requests.post(ollama_url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse Ollama JSON response.\nRaw (first 500 chars):\n{resp.text[:500]}"
        ) from exc

    if "response" not in data:
        raise RuntimeError(
            f"No 'response' field in Ollama output. Full JSON:\n{data}"
        )

    return data["response"]


def _parse_raw_triples(raw_text: str) -> list[dict]:
    """
    Parses raw LLM output into a list of triple dicts.
    Filters garbage terms.

    Args:
        raw_text: Raw multiline string from the LLM.

    Returns:
        List of dicts with keys: subject, predicate, object.
    """
    triples: list[dict] = []
    filtered = 0

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or not all(parts):
            continue

        subject, predicate, obj = parts

        if _is_garbage(subject) or _is_garbage(obj):
            filtered += 1
            logger.debug("Filtered garbage triple: %s | %s | %s",
                         subject, predicate, obj)
            continue

        triples.append(
            {"subject": subject, "predicate": predicate, "object": obj})

    if filtered:
        logger.info(
            "Removed %d garbage triple(s) (market/financial data)", filtered)

    return triples


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars*, breaking at the
    last sentence-ending punctuation (.!?) or newline before the limit
    so that sentences are not split mid-way.

    Args:
        text:      Full corpus text.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks covering the entire input.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to break at the last sentence boundary within the window
        segment = text[start:end]
        break_at = max(
            segment.rfind(". "),
            segment.rfind("\n"),
            segment.rfind("! "),
            segment.rfind("? "),
        )
        if break_at > max_chars // 4:  # only use if reasonably far in
            end = start + break_at + 1
        chunks.append(text[start:end])
        start = end

    return chunks


def generate_triples_from_text(
    text: str,
    model: str = config.MODEL_NAME,
    ollama_url: str = config.OLLAMA_URL,
    timeout: int = config.LLM_TIMEOUT,
    max_chars: int = config.LLM_MAX_CHARS,
    max_chunks: int = config.LLM_MAX_CHUNKS,
) -> list[dict]:
    """
    End-to-end: text-> LLM-> parsed triple list.

    Splits the corpus into chunks of *max_chars* so the entire text is
    processed, then aggregates triples from every chunk.

    Args:
        text:       Input corpus text.
        model:      Ollama model name.
        ollama_url: Ollama API endpoint.
        timeout:    Request timeout in seconds.
        max_chars:  Max characters per LLM call.
        max_chunks: Cap on number of chunks to process (0 = all).

    Returns:
        List of triple dicts. Falls back to FALLBACK_TRIPLE if LLM
        returns nothing usable across all chunks.
    """
    chunks = _chunk_text(text, max_chars)
    total_chunks = len(chunks)
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    logger.info(
        "Generating triples from text (%d chars, %d/%d chunk(s) of ≤%d)",
        len(text), len(chunks), total_chunks, max_chars,
    )

    all_triples: list[dict] = []
    for idx, chunk in enumerate(chunks, 1):
        prompt = config.PROMPT_TEMPLATE.replace("{TEXT}", chunk)
        logger.info("Processing chunk %d/%d (%d chars)",
                    idx, len(chunks), len(chunk))

        if config.USE_OPENAI:
            raw_output = _query_openai(
                prompt, model=config.OPENAI_MODEL, timeout=timeout)
        else:
            raw_output = _query_ollama(
                prompt, model=model, ollama_url=ollama_url, timeout=timeout)
        triples = _parse_raw_triples(raw_output)
        logger.info("Chunk %d/%d-> %d triple(s)",
                    idx, len(chunks), len(triples))
        all_triples.extend(triples)

    if not all_triples:
        logger.warning(
            "LLM returned no usable triples — using fallback triple: %s",
            config.FALLBACK_TRIPLE,
        )
        all_triples = [config.FALLBACK_TRIPLE.copy()]

    return all_triples


# ---------------------------------------------------------------------------
# Strategy class  (implements PipelineStep interface)
# ---------------------------------------------------------------------------

class GenerateTriplesStep(PipelineStep):
    """
    Pipeline step: read corpus text-> generate triples-> write JSON.

    Kwargs:
        input_path  (Path | str): Path to corpus .txt or .csv file.
        output_path (Path | str): Destination .json file for triples.
        model       (str):        Ollama model name override.
        ollama_url  (str):        Ollama endpoint override.
        timeout     (int):        Request timeout override.
        max_chars   (int):        Text crop limit override.
    """

    name: str = "generate_triples"

    def run(self, **kwargs: Any) -> Path:
        """
        Execute the generate-triples step.

        Returns:
            Path to the written output JSON file.

        Raises:
            FileNotFoundError: If input_path does not exist.
            RuntimeError:      On LLM communication failure.
        """
        input_path = Path(kwargs.get(
            "input_path",  config.DEFAULT_PATHS["generate_triples"]["input"]))
        output_path = Path(kwargs.get(
            "output_path", config.DEFAULT_PATHS["generate_triples"]["output"]))
        model = kwargs.get("model",       config.MODEL_NAME)
        ollama_url = kwargs.get("ollama_url",  config.OLLAMA_URL)
        timeout = int(kwargs.get("timeout", config.LLM_TIMEOUT))
        max_chars = int(kwargs.get("max_chars", config.LLM_MAX_CHARS))
        max_chunks = int(kwargs.get("max_chunks", config.LLM_MAX_CHUNKS))

        if not input_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {input_path}")

        logger.info("[%s] Reading corpus: %s", self.name, input_path)
        corpus_text = input_path.read_text(encoding="utf-8")

        triples = generate_triples_from_text(
            corpus_text,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            max_chars=max_chars,
            max_chunks=max_chunks,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(triples, fh, indent=2, ensure_ascii=False)

        logger.info(
            "[%s] Extracted %d triple(s)-> %s",
            self.name, len(triples), output_path,
        )
        return output_path
