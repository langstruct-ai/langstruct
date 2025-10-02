"""Integration tests for LangStruct API."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from langstruct import LangStruct

# Decorator for integration tests
integration_test = pytest.mark.integration


class TestLangStructAPI:
    """Tests for main LangStruct API class."""

    def test_basic_initialization(self, person_schema, mock_extraction_pipeline):
        """Test basic LangStruct initialization."""
        extractor = LangStruct(schema=person_schema)

        assert issubclass(extractor.schema, person_schema)
        assert extractor.schema is not person_schema
        assert extractor.use_sources is True  # Default
        assert extractor.optimizer is None

    def test_initialization_with_options(self, person_schema, mock_extraction_pipeline):
        """Test LangStruct initialization with custom options."""
        extractor = LangStruct(schema=person_schema, model="gpt-4o", use_sources=False)

        assert issubclass(extractor.schema, person_schema)
        assert extractor.optimizer is None
        assert extractor.use_sources is False

    @integration_test
    def test_extract_basic(self, person_schema, requires_api_key, sample_person_text):
        """Test basic extraction functionality."""
        extractor = LangStruct(schema=person_schema, model="gemini/gemini-2.5-flash")
        result = extractor.extract(sample_person_text, validate=False)

        assert result is not None
        assert "name" in result.entities
        assert result.confidence > 0

    def test_extract_empty_text(self, person_schema, mock_extraction_pipeline):
        """Test extraction with empty text."""
        extractor = LangStruct(schema=person_schema)
        result = extractor.extract("", validate=False)

        assert result.entities == {}
        assert result.sources == {}

    def test_extract_with_validation(
        self, person_schema, mock_extraction_pipeline, sample_person_text
    ):
        """Test extraction with validation enabled."""
        extractor = LangStruct(schema=person_schema)

        # Mock validation to avoid issues
        with patch(
            "langstruct.core.schemas.ExtractionResult.validate_quality"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(
                score=0.9,
                summary="Good extraction",
                issues=[],
                suggestions=[],
                has_warnings=False,
                has_errors=False,
            )

            result = extractor.extract(sample_person_text, validate=True)

            assert result is not None
            assert "validation_score" in result.metadata
            assert "validation_summary" in result.metadata

    def test_extract_multiple(self, person_schema, mock_extraction_pipeline):
        """Test batch extraction."""
        texts = ["Text 1", "Text 2", "Text 3"]
        extractor = LangStruct(schema=person_schema)

        results = extractor.extract(texts, validate=False)

        assert len(results) == 3
        for result in results:
            assert result is not None
            assert isinstance(result.entities, dict)

    def test_example_constructor(self, person_example_data, mock_extraction_pipeline):
        """Test LangStruct constructor with example."""

        # Create a test LM that mimics real DSPy LM interface
        class TestLM:
            def __init__(self, model="test-model"):
                self.model = model
                self.kwargs = {"model": model}
                self._test_model_name = f"test-{model}"

            def __call__(self, *args, **kwargs):
                return MagicMock()

        with patch(
            "langstruct.providers.llm_factory.LLMFactory.create_lm"
        ) as mock_create:
            mock_lm = TestLM("gemini-flash")
            mock_create.return_value = mock_lm

            extractor = LangStruct(
                example=person_example_data, model="gemini/gemini-2.5-flash"
            )

            assert extractor is not None
            assert extractor.schema is not None

            # Should be able to extract
            result = extractor.extract("Test text", validate=False)
            assert result is not None

    def test_examples_constructor(
        self, company_examples_data, mock_extraction_pipeline
    ):
        """Test LangStruct constructor with examples."""

        # Create a test LM that mimics real DSPy LM interface
        class TestLM:
            def __init__(self, model="test-model"):
                self.model = model
                self.kwargs = {"model": model}
                self._test_model_name = f"test-{model}"

            def __call__(self, *args, **kwargs):
                return MagicMock()

        with patch(
            "langstruct.providers.llm_factory.LLMFactory.create_lm"
        ) as mock_create:
            mock_lm = TestLM("gemini-flash")
            mock_create.return_value = mock_lm

            extractor = LangStruct(
                examples=company_examples_data, model="gemini/gemini-2.5-flash"
            )

            assert extractor is not None
            assert extractor.schema is not None

            # Should be able to extract
            result = extractor.extract("Test text", validate=False)
            assert result is not None

    def test_constructor_with_schema(self, person_schema, mock_extraction_pipeline):
        """Test LangStruct constructor with existing schema."""
        extractor = LangStruct(schema=person_schema)

        assert issubclass(extractor.schema, person_schema)
        assert extractor.optimizer is None
        assert extractor.use_sources is True  # Should be enabled by auto

    def test_schema_wrapping_enforces_extra_forbid(self, mock_extraction_pipeline):
        """Ensure wrapped schemas reject unexpected fields."""

        class SimpleSchema(BaseModel):
            name: str
            age: int

        extractor = LangStruct(schema=SimpleSchema)
        wrapped = extractor.schema

        with pytest.raises(ValidationError):
            wrapped(name="Bob", age=42, extra="oops")

    def test_schema_wrapping_validates_assignment(self, mock_extraction_pipeline):
        """Ensure wrapped schemas validate attribute assignment."""

        class SimpleSchema(BaseModel):
            name: str
            age: int

        extractor = LangStruct(schema=SimpleSchema)
        instance = extractor.schema(name="Alice", age=30)

        with pytest.raises(ValidationError):
            instance.age = "thirty"

    def test_schema_wrapping_strips_whitespace(self, mock_extraction_pipeline):
        """Ensure wrapped schemas strip leading/trailing whitespace."""

        class SimpleSchema(BaseModel):
            name: str
            title: str

        extractor = LangStruct(schema=SimpleSchema)
        instance = extractor.schema(name="   Alice  ", title="  Engineer   ")

        assert instance.name == "Alice"
        assert instance.title == "Engineer"

    def test_constructor_with_example(
        self, person_example_data, mock_extraction_pipeline
    ):
        """Test LangStruct constructor with example."""
        extractor = LangStruct(example=person_example_data)

        assert extractor.schema is not None
        assert extractor.optimizer is None
        assert extractor.use_sources is True

    def test_constructor_no_input(self, mock_extraction_pipeline):
        """Test LangStruct constructor with no input (should fail)."""
        with pytest.raises(
            ValueError, match="Must provide either schema, example, or examples"
        ):
            LangStruct()

    def test_schema_info_property(self, person_schema, mock_extraction_pipeline):
        """Test schema_info property."""
        extractor = LangStruct(schema=person_schema)
        schema_info = extractor.schema_info

        assert "fields" in schema_info
        assert "descriptions" in schema_info
        assert "json_schema" in schema_info
        assert "example_format" in schema_info

        assert "name" in schema_info["fields"]
        assert "age" in schema_info["fields"]

    def test_export_batch(self, person_schema, mock_extraction_pipeline):
        """Test export_batch method."""
        extractor = LangStruct(schema=person_schema)

        # Create some mock results
        results = [
            extractor.extract("Text 1", validate=False),
            extractor.extract("Text 2", validate=False),
        ]

        import os
        import tempfile

        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            extractor.export_batch(results, csv_path, format="csv")
            assert os.path.exists(csv_path)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

        # Test JSON export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            extractor.export_batch(results, json_path, format="json")
            assert os.path.exists(json_path)
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_confidence_threshold(
        self, person_schema, mock_extraction_pipeline, sample_person_text
    ):
        """Test confidence threshold filtering."""
        extractor = LangStruct(schema=person_schema)

        # Mock low confidence result
        with patch.object(extractor.pipeline, "__call__") as mock_pipeline:
            from langstruct.core.schemas import ExtractionResult

            mock_pipeline.return_value = ExtractionResult(
                entities={"name": "Test"}, sources={}, confidence=0.3  # Low confidence
            )

            # Should warn about low confidence
            with pytest.warns(UserWarning):
                result = extractor.extract(
                    sample_person_text, confidence_threshold=0.5, validate=False
                )

            assert result.confidence == 0.3  # Result should still be returned

    def test_source_grounding_override(
        self, person_schema, mock_extraction_pipeline, sample_person_text
    ):
        """Test source grounding override parameter."""
        extractor = LangStruct(schema=person_schema, use_sources=False)

        # Override to enable sources
        result = extractor.extract(
            sample_person_text, return_sources=True, validate=False
        )

        # Should have used sources for this extraction
        assert result is not None

        # Override to disable sources
        result = extractor.extract(
            sample_person_text, return_sources=False, validate=False
        )

        # Should not have used sources for this extraction
        assert result is not None

    def test_repr(self, person_schema, mock_extraction_pipeline):
        """Test __repr__ method."""
        extractor = LangStruct(schema=person_schema)
        repr_str = repr(extractor)

        assert "LangStruct" in repr_str
        assert "PersonSchema" in repr_str
        assert "optimizer_initialized=False" in repr_str

    def test_save_load_basic_functionality(
        self, person_schema, mock_extraction_pipeline
    ):
        """Test basic save/load functionality."""
        import tempfile
        from pathlib import Path

        extractor = LangStruct(schema=person_schema)

        # Test save and load functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"

            # Save should work
            extractor.save(str(save_path))

            # Verify save directory and files were created
            assert save_path.exists()
            assert save_path.is_dir()
            assert (save_path / "langstruct_metadata.json").exists()
            assert (save_path / "pipeline.json").exists()

            # Load should work and return a valid extractor
            loaded = LangStruct.load(str(save_path))
            assert loaded is not None
            assert issubclass(loaded.schema, person_schema)

    def test_optimize_raises_for_invalid_optimizer(
        self, person_schema, mock_extraction_pipeline
    ):
        """Ensure invalid optimizer names raise when optimization runs."""
        extractor = LangStruct(schema=person_schema, optimizer="invalid")

        with pytest.raises(ValueError, match="Unknown optimizer"):
            extractor.optimize(["text"])

    def test_optimization_default_disabled(
        self, person_schema, mock_extraction_pipeline
    ):
        """Test that optimization is disabled by default."""
        extractor = LangStruct(schema=person_schema)

        # Optimization should be disabled by default now
        assert extractor.optimizer is None

    def test_evaluate_placeholder(self, person_schema, mock_extraction_pipeline):
        """Test evaluate method (currently placeholder)."""
        extractor = LangStruct(schema=person_schema)

        texts = ["Text 1", "Text 2"]
        expected_results = [{"name": "John"}, {"name": "Jane"}]

        # Should not fail and return scores dict
        with patch("langstruct.optimizers.metrics.ExtractionMetrics") as mock_metrics:
            mock_instance = MagicMock()
            mock_instance.calculate_accuracy.return_value = 0.8
            mock_instance.calculate_f1.return_value = 0.75
            mock_metrics.return_value = mock_instance

            scores = extractor.evaluate(texts, expected_results)

            assert isinstance(scores, dict)
            assert "accuracy" in scores
            assert "f1" in scores


class TestLangStructErrorHandling:
    """Tests for error handling in LangStruct."""

    def test_invalid_model_name(self, person_schema):
        """Test error handling for invalid model names."""
        # Mock LLMFactory to raise an error
        with patch(
            "langstruct.providers.llm_factory.LLMFactory.create_lm"
        ) as mock_create:
            mock_create.side_effect = ValueError("Unsupported model")

            with pytest.raises(ValueError, match="Unsupported model"):
                LangStruct(schema=person_schema, model="invalid-model")

    def test_empty_schema(self):
        """Test error handling for invalid schema."""

        class EmptySchema(BaseModel):
            pass  # No fields

        # Should still work but may have validation issues
        extractor = LangStruct(schema=EmptySchema)
        assert issubclass(extractor.schema, EmptySchema)

    def test_evaluate_mismatched_lengths(self, person_schema, mock_extraction_pipeline):
        """Test evaluate with mismatched input lengths."""
        extractor = LangStruct(schema=person_schema)

        texts = ["Text 1", "Text 2"]
        expected_results = [{"name": "John"}]  # Different length

        with pytest.raises(ValueError, match="must match number"):
            extractor.evaluate(texts, expected_results)


class TestLangStructIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_with_validation(self, mock_extraction_pipeline):
        """Test complete workflow from schema generation to export."""
        # 1. Generate schema from example
        example = {"name": "John Doe", "age": 30, "city": "Boston"}
        extractor = LangStruct(example=example)

        # 2. Extract from text
        text = "Jane Smith is 25 years old and lives in New York."

        with patch(
            "langstruct.core.schemas.ExtractionResult.validate_quality"
        ) as mock_validate:
            mock_validate.return_value = MagicMock(
                score=0.8,
                summary="Good extraction",
                issues=[],
                suggestions=[],
                has_warnings=False,
                has_errors=False,
            )

            result = extractor.extract(text, validate=True)

        # 3. Verify result structure
        assert result is not None
        assert "name" in result.entities
        assert result.confidence > 0
        assert "validation_score" in result.metadata

        # 4. Export result
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result.save_json(temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_batch_processing_workflow(self, mock_extraction_pipeline):
        """Test batch processing with export."""
        # Generate schema
        examples = [
            {"company": "Apple", "founded": 1976, "location": "Cupertino"},
            {"company": "Google", "founded": 1998, "location": "Mountain View"},
        ]
        extractor = LangStruct(examples=examples)

        # Batch process
        texts = [
            "Microsoft was founded in 1975 in Redmond.",
            "Amazon started in 1994 in Seattle.",
            "Facebook began in 2004 in Menlo Park.",
        ]

        results = extractor.extract(texts, validate=False)
        assert len(results) == 3

        # Export batch results
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            extractor.export_batch(results, csv_path)
            assert os.path.exists(csv_path)

            # Verify CSV content
            import csv

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3
            assert all("company" in row for row in rows)

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_auto_configuration_workflow(self, mock_extraction_pipeline):
        """Test auto-configuration with defaults."""
        example = {"product": "iPhone", "price": 999, "available": True}

        # Use auto configuration
        extractor = LangStruct(example=example)

        # Verify default settings
        assert extractor.optimizer is None
        assert extractor.use_sources is True

        # Should work for extraction
        result = extractor.extract(
            "The iPad costs $799 and is available.", validate=False
        )
        assert result is not None
