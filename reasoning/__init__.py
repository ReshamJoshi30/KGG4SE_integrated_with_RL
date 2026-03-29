# reasoning/__init__.py
"""
Reasoner integration module.

Provides a unified interface to run multiple OWL reasoners
(Konclude, HermiT) and returns normalized reports in a stable schema.

Usage:
    from reasoning import run_reasoner

    report = run_reasoner("merged_kg.owl", reasoner="konclude")
"""

from reasoning.reasoner_wrapper import run_reasoner

__all__ = ["run_reasoner"]