"""Pydantic schemas for type-safe extraction definitions and results."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import field_validator
from typing_extensions import Annotated

# Re-export Field from pydantic for user convenience
Field = PydanticField

# Type alias for any Pydantic model used as a schema
if TYPE_CHECKING:
    Schema = BaseModel  # For backward compatibility in type hints
else:
    Schema = BaseModel


class SourceSpan(BaseModel):
    """Represents a location in the source text."""

    start: int = Field(description="Character start position in source text")
    end: int = Field(description="Character end position in source text")
    text: str = Field(description="Actual text content at this location")

    model_config = ConfigDict(frozen=True)

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be greater than start")
        return v


class ExtractedEntity(BaseModel):
    """Base class for extracted entities with source grounding."""

    value: Any = Field(description="The extracted value")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this extraction (0-1)",
    )
    source: Optional[SourceSpan] = Field(
        default=None, description="Source location where this entity was found"
    )


class ExtractionResult(BaseModel):
    """Complete result of an extraction operation."""

    entities: Dict[str, Any] = Field(
        description="Extracted entities organized by field name"
    )
    sources: Dict[str, List[SourceSpan]] = Field(
        default_factory=dict, description="Source spans for each extracted field"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall extraction confidence score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the extraction"
    )

    model_config = ConfigDict(extra="allow")

    def validate_quality(
        self,
        schema: Optional[BaseModel] = None,
        text: Optional[str] = None,
        previous_results: Optional[List["ExtractionResult"]] = None,
    ) -> "ValidationReport":
        """Validate extraction quality and get improvement suggestions.

        Args:
            schema: Schema used for extraction (required for validation)
            text: Original text that was extracted from
            previous_results: Previous extraction results for consistency checking

        Returns:
            ValidationReport with issues and suggestions

        Example:
            >>> result = extractor.extract(text)
            >>> report = result.validate_quality(schema=PersonSchema, text=text)
            >>> # if not report.is_valid:
            >>> #     inspect report.suggestions for guidance
        """
        if schema is None:
            # Import here to avoid circular import
            from .validation import (
                IssueType,
                Severity,
                ValidationIssue,
                ValidationReport,
            )

            return ValidationReport(
                is_valid=True,
                score=self.confidence,
                issues=[
                    ValidationIssue(
                        type=IssueType.SCHEMA_DESIGN,
                        severity=Severity.INFO,
                        message="No schema provided for validation",
                        suggestion="Pass schema parameter for detailed validation",
                    )
                ],
                suggestions=["Pass schema parameter for detailed validation"],
                summary=f"Basic validation: {self.confidence:.1%} confidence",
            )

        # Import here to avoid circular import
        from .validation import ExtractionValidator

        validator = ExtractionValidator(schema)
        return validator.validate(self, text, previous_results)


# Legacy Schema class for backward compatibility
# Users should now use pydantic.BaseModel directly
class Schema(BaseModel):
    """Legacy base class for extraction schemas.

    DEPRECATED: Use pydantic.BaseModel directly instead.
    This class is kept for backward compatibility only.

    Example:
        # OLD (deprecated):
        class PersonSchema(Schema):
            name: str = Field(description="Full name of the person")

        # NEW (recommended):
        from pydantic import BaseModel, Field

        class PersonSchema(BaseModel):
            name: str = Field(description="Full name of the person")
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Get JSON schema representation for LLM consumption.

        DEPRECATED: Use cls.model_json_schema() directly.
        """
        from .schema_utils import get_json_schema

        return get_json_schema(cls)

    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        """Get field descriptions for prompt engineering.

        DEPRECATED: Use langstruct.core.schema_utils.get_field_descriptions() instead.
        """
        from .schema_utils import get_field_descriptions

        return get_field_descriptions(cls)

    @classmethod
    def get_example_format(cls) -> str:
        """Get example JSON format for few-shot prompting.

        DEPRECATED: Use langstruct.core.schema_utils.get_example_format() instead.
        """
        from .schema_utils import get_example_format

        return get_example_format(cls)


class ChunkResult(BaseModel):
    """Result of processing a single text chunk."""

    chunk_id: int = Field(description="Unique identifier for this chunk")
    chunk_text: str = Field(description="The actual text content of this chunk")
    start_offset: int = Field(
        description="Character offset where this chunk starts in original text"
    )
    end_offset: int = Field(
        description="Character offset where this chunk ends in original text"
    )
    extraction: ExtractionResult = Field(
        description="Extraction results for this chunk"
    )

    model_config = ConfigDict(frozen=True)


class ParsedQuery(BaseModel):
    """Result of parsing a natural language query into structured components."""

    semantic_terms: List[str] = Field(description="Terms for semantic/embedding search")
    structured_filters: Dict[str, Any] = Field(
        description="Filters for metadata search"
    )
    confidence: float = Field(description="Confidence in the parsing", ge=0.0, le=1.0)
    explanation: str = Field(
        description="Human-readable explanation of what was parsed"
    )
    raw_query: str = Field(description="Original query")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parsing metadata"
    )

    model_config = ConfigDict(frozen=True)
