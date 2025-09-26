"""Core extraction functionality."""

from .chunking import TextChunker
from .export_utils import ExportUtilities  # This adds methods to ExtractionResult
from .grounding import SourceGrounder
from .modules import EntityExtractor, ExtractionPipeline
from .schema_generator import SchemaGenerator, schema_from_example, schema_from_examples
from .schemas import ExtractionResult, Field, Schema
from .signatures import ExtractEntities, ExtractWithSources
from .validation import ExtractionValidator, ValidationIssue, ValidationReport

__all__ = [
    "Schema",
    "Field",
    "ExtractionResult",
    "ExtractEntities",
    "ExtractWithSources",
    "EntityExtractor",
    "ExtractionPipeline",
    "SourceGrounder",
    "TextChunker",
    "SchemaGenerator",
    "schema_from_example",
    "schema_from_examples",
    "ExtractionValidator",
    "ValidationReport",
    "ValidationIssue",
]
