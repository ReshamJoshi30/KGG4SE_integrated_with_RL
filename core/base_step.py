# core/base_step.py
"""
Abstract base class for all pipeline steps.

Every step in the KG pipeline inherits from PipelineStep and implements
the run(**kwargs) method. This enforces a consistent interface across
all steps and makes the pipeline:

  - Swappable   : swap any step implementation without touching orchestrator
  - Testable    : each step is independently unit-testable
  - REST-ready  : kwargs map directly to HTTP request body params
  - Extensible  : add new steps without modifying existing ones
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PipelineStep(ABC):
    """
    Strategy interface for a single pipeline step.

    Subclasses must:
      - Set a unique ``name`` class attribute.
      - Implement ``run(**kwargs) -> Path``.

    All input/output paths and runtime overrides are passed via kwargs
    so the interface stays stable as steps evolve.

    Example
    -------
    class MyStep(PipelineStep):
        name = "my_step"

        def run(self, **kwargs: Any) -> Path:
            input_path  = Path(kwargs.get("input_path", ...))
            output_path = Path(kwargs.get("output_path", ...))
            # ... do work ...
            return output_path
    """

    #: Unique step identifier — used in logs and CLI routing.
    name: str = "base"

    @abstractmethod
    def run(self, **kwargs: Any) -> Path:
        """
        Execute this pipeline step.

        Args:
            **kwargs: Step-specific parameters.
                      Each step documents its own supported kwargs.
                      Common conventions:
                        input_path  (Path | str) : primary input file
                        output_path (Path | str) : primary output file

        Returns:
            Path to the primary output file produced by this step.

        Raises:
            FileNotFoundError : If a required input file is missing.
            RuntimeError      : On any unrecoverable processing error.
        """

    def execute(self, **kwargs: Any) -> Path:
        """
        Wrapper around ``run()`` that adds:
          - Structured entry / exit logging with step name and elapsed time.
          - Consistent error formatting — catches all exceptions,
            logs them at ERROR level, then re-raises so the pipeline
            orchestrator can decide whether to stop or continue.

        Args:
            **kwargs: Forwarded verbatim to ``run()``.

        Returns:
            Path returned by ``run()``.

        Raises:
            Re-raises any exception raised inside ``run()``.
        """
        logger.info("=== Step [%s] START ===", self.name)
        start = time.perf_counter()

        try:
            result = self.run(**kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "=== Step [%s] FAILED after %.2fs — %s: %s ===",
                self.name,
                elapsed,
                type(exc).__name__,
                exc,
            )
            raise

        elapsed = time.perf_counter() - start
        logger.info("=== Step [%s] DONE (%.2fs) ===", self.name, elapsed)
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"