"""
LangStruct: LLM-powered structured information extraction using DSPy optimization.

A next-generation text extraction library that improves upon existing solutions
by leveraging DSPy's self-optimizing framework instead of manual prompt engineering.
"""

from .api import LangStruct
from .core.chunking import ChunkingConfig
from .core.export_utils import ExportUtilities
from .core.refinement import Budget, Refine, RefinementStrategy
from .core.schema_generator import schema_from_example, schema_from_examples
from .core.schemas import ExtractionResult, Field, ParsedQuery, Schema
from .exceptions import (
    ConfigurationError,
    ExtractionError,
    LangStructError,
    PersistenceError,
    ValidationError,
)
from .optimizers.metrics import ExtractionMetrics
from .visualization.html_viz import HTMLVisualizer, save_visualization, visualize

__version__ = "0.1.0"
__all__ = [
    "LangStruct",
    "ParsedQuery",
    "Schema",
    "Field",
    "ExtractionResult",
    "ChunkingConfig",
    "ExportUtilities",
    "Refine",
    "Budget",
    "RefinementStrategy",
    "HTMLVisualizer",
    "visualize",
    "save_visualization",
    "ExtractionMetrics",
    "schema_from_example",
    "schema_from_examples",
    # Exceptions
    "LangStructError",
    "ConfigurationError",
    "ExtractionError",
    "PersistenceError",
    "ValidationError",
]
