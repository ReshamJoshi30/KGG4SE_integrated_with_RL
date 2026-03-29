# Scripts/run_reasoner.py
"""
Reasoner step.

Loads the merged KG OWL file, runs the HermiT reasoner via owlready2,
and saves the inferred ontology.

Consistency result is returned so the pipeline orchestrator or RL
module can act on it — inconsistencies are treated as repair candidates,
not hard failures.
"""

import logging
import re
from pathlib import Path
from typing import Any

from owlready2 import (
    get_ontology,
    sync_reasoner,
    OwlReadyInconsistentOntologyError,
)

import config
from core.base_step import PipelineStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions  (independently testable)
# ---------------------------------------------------------------------------

def check_for_ontology_collapse(owl_path: Path) -> tuple[bool, str]:
    """
    Check if reasoned OWL contains ontology collapse.

    Detects the giant EquivalentClasses block that indicates all classes
    have collapsed to owl:Nothing due to logical contradictions.

    Args:
        owl_path: Path to the reasoned OWL file

    Returns:
        (collapsed: bool, message: str)
    """
    if not owl_path.exists():
        return False, "File does not exist"

    try:
        with open(owl_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.warning("Could not read OWL file for collapse check: %s", e)
        return False, f"Read error: {e}"

    if '<EquivalentClasses>' not in content:
        return False, "No EquivalentClasses blocks found"

    # Count classes in EquivalentClasses blocks
    equiv_blocks = re.findall(
        r'<EquivalentClasses>(.*?)</EquivalentClasses>',
        content,
        re.DOTALL
    )

    for idx, block in enumerate(equiv_blocks):
        class_count = block.count('<Class IRI=')

        # Threshold: more than 50 classes in one equivalence = collapse
        if class_count > 50:
            classes = re.findall(r'<Class IRI="([^"]+)"', block)[:10]
            class_names = [c.split('#')[-1].split('/')[-1]
                           for c in classes[:5]]

            return True, (
                f"ONTOLOGY COLLAPSE DETECTED in EquivalentClasses block {idx+1}: "
                f"{class_count} classes equivalent, including: "
                f"{', '.join(class_names)}... "
                f"This indicates class-level assertions or circular reasoning in aligned triples."
            )

    return False, f"Found {len(equiv_blocks)} normal EquivalentClasses blocks"


def run_hermit(
    input_path: Path,
    output_path: Path,
) -> bool:
    """
    Load a merged KG OWL file, run HermiT reasoning, and save the result.

    Args:
        input_path:  Path to the merged KG OWL file.
        output_path: Destination path for the reasoned OWL file.

    Returns:
        True  if the ontology is consistent.
        False if the ontology is inconsistent (repair candidates exist).

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        RuntimeError:      On unexpected reasoner failure (non-inconsistency error).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Merged KG file not found: {input_path}")

    logger.info("Loading merged KG: %s", input_path)
    onto = get_ontology(str(input_path.resolve())).load()

    logger.info("Running HermiT reasoner...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    consistent = True

    with onto:
        try:
            sync_reasoner()
            logger.info("Ontology is CONSISTENT ✓")

        except OwlReadyInconsistentOntologyError:
            consistent = False
            logger.warning(
                "Ontology is INCONSISTENT — "
                "real errors detected, RL has repair candidates ⚠"
            )

        except Exception as exc:
            msg = str(exc).lower()
            # Only swallow known non-fatal Java / reasoner warnings.
            # Let real errors (OOM, missing JAR, ClassNotFound, etc.) propagate.
            non_fatal_keywords = [
                "java",
                "warning",
                "deprecated",
                "ignoring",
                "timer",
                "heap",  # GC info printed to stderr, not an error
            ]
            if any(kw in msg for kw in non_fatal_keywords):
                logger.warning(
                    "HermiT non-fatal warning (continuing): %s", exc)
            else:
                raise RuntimeError(
                    f"Unexpected reasoner error (not a known Java warning): {exc}"
                ) from exc

    logger.info("Saving reasoned KG-> %s", output_path)
    onto.save(file=str(output_path), format="rdfxml")

    # Check for ontology collapse
    collapsed, message = check_for_ontology_collapse(output_path)
    if collapsed:
        logger.error("⚠️  %s", message)
        logger.error(
            "RECOMMENDED FIXES:\n"
            "  1. Check outputs/reports/alignment_report.csv for issues\n"
            "  2. Verify aligned triples use individuals, not classes\n"
            "  3. Look for circular property assertions\n"
            "  4. Re-run with ALLOW_CREATE_INDIVIDUALS=True in config.py"
        )
        consistent = False  # Mark as inconsistent
    else:
        logger.info("✓ Collapse check passed: %s", message)

    if consistent:
        logger.info(
            "Consistent ontology saved — "
            "run inject_errors.py to create test errors for RL"
        )
    else:
        logger.info(
            "Inconsistent or collapsed ontology saved — "
            "RL repair module can now process repair candidates"
        )

    return consistent


# ---------------------------------------------------------------------------
# Strategy class  (implements PipelineStep interface)
# ---------------------------------------------------------------------------

class RunReasonerStep(PipelineStep):
    """
    Pipeline step: merged KG OWL-> reasoned KG OWL.

    Kwargs:
        input_path  (Path | str): Merged KG OWL file.
                                  Default: config.DEFAULT_PATHS["run_reasoner"]["input"]
        output_path (Path | str): Reasoned KG OWL file.
                                  Default: config.DEFAULT_PATHS["run_reasoner"]["output"]

    Notes:
        An inconsistent ontology does NOT fail the step — it returns
        the output path normally and logs a warning. The consistency
        result is stored in ``self.last_consistent`` for the pipeline
        orchestrator or RL module to inspect.
    """

    name: str = "run_reasoner"

    def __init__(self) -> None:
        self.last_consistent: bool | None = None

    def run(self, **kwargs: Any) -> Path:
        """
        Execute the run-reasoner step.

        Returns:
            Path to the written reasoned OWL file.

        Raises:
            FileNotFoundError: If the merged KG OWL file is missing.
            RuntimeError:      On unexpected non-recoverable reasoner error.
        """
        defaults = config.DEFAULT_PATHS["run_reasoner"]
        input_path = Path(kwargs.get("input_path",  defaults["input"]))
        output_path = Path(kwargs.get("output_path", defaults["output"]))

        logger.info(
            "[%s] input=%s | output=%s",
            self.name, input_path, output_path,
        )

        self.last_consistent = run_hermit(
            input_path=input_path,
            output_path=output_path,
        )

        return output_path
