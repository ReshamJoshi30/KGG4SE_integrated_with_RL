# core/__init__.py
"""
Core abstractions for the KG pipeline.

Exports:
    PipelineStep — abstract base class for all pipeline steps.
"""

from core.base_step import PipelineStep

__all__ = ["PipelineStep"]