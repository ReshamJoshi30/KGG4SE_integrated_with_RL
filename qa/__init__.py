# qa/__init__.py
"""
Quality assurance and repair module.

Provides repair candidate generation and fix application for KG repair tasks.
"""

from qa.repair_candidates import make_repair_candidates
from qa.apply_fix import apply_fix

__all__ = ["make_repair_candidates", "apply_fix"]