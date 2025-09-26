"""Tests for core schema functionality."""

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from langstruct.core.schema_utils import (
    get_example_format,
    get_field_descriptions,
    get_json_schema,
)
from langstruct.core.schemas import (
    ChunkResult,
    ExtractedEntity,
    ExtractionResult,
    SourceSpan,
)


class PersonTestSchema(BaseModel):
    """Test schema for testing purposes."""

    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    email: Optional[str] = Field(default=None, description="Email address")
    skills: List[str] = Field(description="List of skills")


class TestSourceSpan:
    """Tests for SourceSpan model."""

    def test_source_span_creation(self):
        """Test creating valid SourceSpan."""
        span = SourceSpan(start=0, end=10, text="test text")
        assert span.start == 0
        assert span.end == 10
        assert span.text == "test text"

    def test_source_span_validation(self):
        """Test SourceSpan validation."""
        # Valid span
        span = SourceSpan(start=5, end=15, text="valid span")
        assert span.start < span.end

        # Invalid span - end <= start
        with pytest.raises(ValidationError):
            SourceSpan(start=10, end=5, text="invalid")

        with pytest.raises(ValidationError):
            SourceSpan(start=10, end=10, text="invalid")

    def test_source_span_immutable(self):
        """Test that SourceSpan is immutable."""
        span = SourceSpan(start=0, end=10, text="test")

        with pytest.raises(ValidationError):
            span.start = 5  # Should fail as model is frozen


class TestExtractedEntity:
    """Tests for ExtractedEntity model."""

    def test_extracted_entity_creation(self):
        """Test creating ExtractedEntity with defaults."""
        entity = ExtractedEntity(value="test value")
        assert entity.value == "test value"
        assert entity.confidence == 1.0
        assert entity.source is None

    def test_extracted_entity_with_source(self):
        """Test ExtractedEntity with source information."""
        span = SourceSpan(start=0, end=5, text="test")
        entity = ExtractedEntity(value="test value", confidence=0.8, source=span)
        assert entity.confidence == 0.8
        assert entity.source == span

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        ExtractedEntity(value="test", confidence=0.0)
        ExtractedEntity(value="test", confidence=0.5)
        ExtractedEntity(value="test", confidence=1.0)

        # Invalid confidence scores
        with pytest.raises(ValidationError):
            ExtractedEntity(value="test", confidence=-0.1)

        with pytest.raises(ValidationError):
            ExtractedEntity(value="test", confidence=1.1)


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_extraction_result_creation(self):
        """Test creating basic ExtractionResult."""
        result = ExtractionResult(
            entities={"name": "John", "age": 25}, sources={}, confidence=0.9
        )

        assert result.entities == {"name": "John", "age": 25}
        assert result.confidence == 0.9
        assert result.sources == {}
        assert result.metadata == {}

    def test_extraction_result_with_sources(self):
        """Test ExtractionResult with source information."""
        sources = {
            "name": [SourceSpan(start=0, end=4, text="John")],
            "age": [SourceSpan(start=10, end=12, text="25")],
        }

        result = ExtractionResult(
            entities={"name": "John", "age": 25}, sources=sources, confidence=0.9
        )

        assert result.sources == sources
        assert len(result.sources["name"]) == 1
        assert result.sources["name"][0].text == "John"

    def test_extraction_result_metadata(self):
        """Test ExtractionResult with metadata."""
        metadata = {"pipeline": "test", "chunks": 2, "processing_time": 1.5}

        result = ExtractionResult(
            entities={"name": "John"}, sources={}, confidence=0.8, metadata=metadata
        )

        assert result.metadata == metadata
        assert result.metadata["pipeline"] == "test"

    def test_confidence_validation(self):
        """Test confidence validation in ExtractionResult."""
        # Valid confidence
        ExtractionResult(entities={}, sources={}, confidence=0.5)

        # Invalid confidence
        with pytest.raises(ValidationError):
            ExtractionResult(entities={}, sources={}, confidence=-0.1)

        with pytest.raises(ValidationError):
            ExtractionResult(entities={}, sources={}, confidence=1.5)


class TestSchema:
    """Tests for user-defined schema models."""

    def test_schema_creation(self):
        """Test creating schema instances."""
        schema = PersonTestSchema(
            name="John Doe", age=30, email="john@example.com", skills=["Python", "ML"]
        )

        assert schema.name == "John Doe"
        assert schema.age == 30
        assert schema.email == "john@example.com"
        assert schema.skills == ["Python", "ML"]

    def test_schema_optional_fields(self):
        """Test schema with optional fields."""
        # Email is optional, should work without it
        schema = PersonTestSchema(name="Jane Doe", age=25, skills=["JavaScript"])

        assert schema.name == "Jane Doe"
        assert schema.email is None
        assert schema.skills == ["JavaScript"]

    def test_schema_validation(self):
        """Test schema field validation."""
        # Valid schema
        PersonTestSchema(name="John", age=30, skills=[])

        # Invalid types
        with pytest.raises(ValidationError):
            PersonTestSchema(name=123, age=30, skills=[])  # name must be string

        with pytest.raises(ValidationError):
            PersonTestSchema(name="John", age="thirty", skills=[])  # age must be int

        with pytest.raises(ValidationError):
            PersonTestSchema(
                name="John", age=30, skills="Python"
            )  # skills must be list

    def test_get_json_schema(self):
        """Test JSON schema generation."""
        json_schema = get_json_schema(PersonTestSchema)

        assert "properties" in json_schema
        assert "name" in json_schema["properties"]
        assert "age" in json_schema["properties"]
        assert "email" in json_schema["properties"]
        assert "skills" in json_schema["properties"]

        # Check field types
        assert json_schema["properties"]["name"]["type"] == "string"
        assert json_schema["properties"]["age"]["type"] == "integer"
        assert json_schema["properties"]["skills"]["type"] == "array"

    def test_get_field_descriptions(self):
        """Test field descriptions extraction."""
        descriptions = get_field_descriptions(PersonTestSchema)

        expected = {
            "name": "Full name",
            "age": "Age in years",
            "email": "Email address",
            "skills": "List of skills",
        }

        assert descriptions == expected

    def test_get_example_format(self):
        """Test example format generation."""
        example = get_example_format(PersonTestSchema)

        # Should be valid JSON
        import json

        parsed = json.loads(example)

        assert "name" in parsed
        assert "age" in parsed
        assert "email" in parsed
        assert "skills" in parsed

        # Check example types match schema
        assert isinstance(parsed["name"], str)
        assert isinstance(parsed["age"], int)
        assert isinstance(parsed["skills"], list)


class TestChunkResult:
    """Tests for ChunkResult model."""

    def test_chunk_result_creation(self, sample_extraction_result):
        """Test creating ChunkResult."""
        chunk_result = ChunkResult(
            chunk_id=0,
            chunk_text="Sample text",
            start_offset=0,
            end_offset=11,
            extraction=sample_extraction_result,
        )

        assert chunk_result.chunk_id == 0
        assert chunk_result.chunk_text == "Sample text"
        assert chunk_result.start_offset == 0
        assert chunk_result.end_offset == 11
        assert chunk_result.extraction == sample_extraction_result

    def test_chunk_result_immutable(self, sample_extraction_result):
        """Test that ChunkResult is immutable."""
        chunk_result = ChunkResult(
            chunk_id=0,
            chunk_text="Sample text",
            start_offset=0,
            end_offset=11,
            extraction=sample_extraction_result,
        )

        with pytest.raises(ValidationError):
            chunk_result.chunk_id = 1  # Should fail as model is frozen
