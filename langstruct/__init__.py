"""
LangStruct: LLM-powered structured information extraction using DSPy optimization.

A next-generation text extraction library that improves upon existing solutions
by leveraging DSPy's self-optimizing framework instead of manual prompt engineering.
"""

import logging

# Configure logging format with timestamps for all langstruct loggers
# Child loggers (like langstruct.core.modules) propagate to parent loggers,
# so configuring the langstruct logger ensures all child loggers get timestamps
_langstruct_logger = logging.getLogger("langstruct")

# Only configure if not already set up (to avoid duplicates on re-imports)
if not _langstruct_logger.handlers:
    _logging_handler = logging.StreamHandler()
    _logging_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _logging_handler.setFormatter(_logging_formatter)
    _langstruct_logger.addHandler(_logging_handler)
    _langstruct_logger.setLevel(logging.INFO)
    # Don't propagate to root to avoid duplicate logs
    _langstruct_logger.propagate = False

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
