# llm_kg/__init__.py
"""
LLM triple generation module.

Exports:
    GenerateTriplesStep     — pipeline step class.
    generate_triples_from_text — pure function for direct use or testing.
"""

from llm_kg.generate_triples import GenerateTriplesStep, generate_triples_from_text

__all__ = [
    "GenerateTriplesStep",
    "generate_triples_from_text",
]