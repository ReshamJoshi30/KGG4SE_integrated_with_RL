# Scripts/prepare_corpus.py
"""
Corpus preparation step.

Reads a CSV file containing a 'title' column and writes one line
per entry to a plain-text corpus file consumed by the triple
generation step.

Supported CSV column override via kwargs so the step is reusable
for CSVs with different schemas.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

import config
from core.base_step import PipelineStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure function  (independently testable)
# ---------------------------------------------------------------------------

def csv_to_txt(
    input_path: Path,
    output_path: Path,
    text_column: str = "title",
) -> int:
    """
    Convert a CSV file to a plain-text corpus file.

    Args:
        input_path:  Path to the source CSV file.
        output_path: Path to the destination .txt file.
        text_column: Name of the CSV column to extract text from.

    Returns:
        Number of lines written.

    Raises:
        FileNotFoundError : If ``input_path`` does not exist.
        ValueError        : If ``text_column`` is not found in the CSV.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus CSV not found: {input_path}")

    logger.debug("Reading CSV: %s", input_path)
    df = pd.read_csv(input_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_column].dropna().tolist()

    if not texts:
        logger.warning("No non-null values found in column '%s'", text_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        for entry in texts:
            fh.write(str(entry).strip() + "\n")

    logger.info(
        "Corpus written: %d lines-> %s",
        len(texts),
        output_path,
    )
    return len(texts)


# ---------------------------------------------------------------------------
# Strategy class  (implements PipelineStep interface)
# ---------------------------------------------------------------------------

class PrepareCorpusStep(PipelineStep):
    """
    Pipeline step: CSV-> plain-text corpus.

    Kwargs:
        input_path  (Path | str): Source CSV file.
                                  Default: config.DEFAULT_PATHS["prepare_corpus"]["input"]
        output_path (Path | str): Destination .txt file.
                                  Default: config.DEFAULT_PATHS["prepare_corpus"]["output"]
        text_column (str)       : CSV column to extract.
                                  Default: "title"
    """

    name: str = "prepare_corpus"

    def run(self, **kwargs: Any) -> Path:
        """
        Execute the prepare-corpus step.

        Returns:
            Path to the written corpus .txt file.

        Raises:
            FileNotFoundError: If input CSV does not exist.
            ValueError:        If the specified text column is missing.
        """
        input_path  = Path(kwargs.get("input_path",  config.DEFAULT_PATHS["prepare_corpus"]["input"]))
        output_path = Path(kwargs.get("output_path", config.DEFAULT_PATHS["prepare_corpus"]["output"]))
        text_column = str(kwargs.get("text_column", "title"))

        logger.info(
            "[%s] input=%s | output=%s | column=%s",
            self.name, input_path, output_path, text_column,
        )

        csv_to_txt(
            input_path=input_path,
            output_path=output_path,
            text_column=text_column,
        )

        return output_path