# reasoning/reasoner_wrapper.py
"""
Unified reasoner interface.

Routes to Konclude or HermiT runners and always returns a normalized report dict.

Usage:
    from reasoning import run_reasoner

    report = run_reasoner("merged_kg.owl", reasoner="konclude")
    report = run_reasoner("merged_kg.owl", reasoner="hermit")

Both return the same schema:
{
    "is_consistent":       bool,
    "unsat_classes":       list[str],
    "disjoint_violations": list[tuple],
    "raw_logs_path":       str,
    "stats": {
        "reasoner":         str,
        "duration_seconds": float,
        "triple_count":     int,
    }
}
"""

from typing import Dict, Any

import config
from reasoning import konclude_runner, hermit_runner


def run_reasoner(
    input_owl: str,
    reasoner: str = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run an OWL reasoner and return a normalized report.

    Args:
        input_owl: Path to input OWL file.
        reasoner:  Which reasoner to use ("konclude" or "hermit").
                   Defaults to config.DEFAULT_REASONER.
        **kwargs:  Additional arguments passed to the specific runner.

    Returns:
        Normalized report dict (see module docstring for schema).

    Raises:
        ValueError:        If reasoner is not "konclude" or "hermit".
        FileNotFoundError: If input_owl does not exist or reasoner binary not found.
    """
    if reasoner is None:
        reasoner = config.DEFAULT_REASONER

    reasoner = reasoner.lower().strip()

    if reasoner == "konclude":
        return konclude_runner.run(input_owl, **kwargs)
    elif reasoner == "hermit":
        return hermit_runner.run(input_owl, **kwargs)
    else:
        raise ValueError(
            f"Unknown reasoner: '{reasoner}'. "
            f"Supported: 'konclude', 'hermit'"
        )