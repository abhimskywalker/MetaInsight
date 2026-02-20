"""Python implementation scaffold for MetaInsight.

This package provides a clean-room, Python-first foundation for the core
MetaInsight workflows. The API mirrors the existing R package entry points
in a progressive way so features can be ported module-by-module.
"""

from .core import MetaInsightBundle, MetaInsightConfig, load_data, setup_configure

__all__ = [
    "MetaInsightBundle",
    "MetaInsightConfig",
    "load_data",
    "setup_configure",
]

__version__ = "0.1.0"
