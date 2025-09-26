"""Performance tests for LangStruct."""

import time
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from langstruct import LangStruct


class BenchmarkSchema(BaseModel):
    """Schema for performance testing."""

    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    location: str = Field(description="Location")
    occupation: str = Field(description="Job title")
    company: str = Field(description="Company name")


@pytest.fixture
def fast_mock_pipeline():
    """Mock pipeline that responds quickly."""

    def mock_pipeline_factory(schema, **kwargs):
        pipeline = MagicMock()

        def fast_extract(text):
            from langstruct.core.schemas import ExtractionResult

            return ExtractionResult(
                entities={
                    "name": "Mock Name",
                    "age": 30,
                    "location": "Mock City",
                    "occupation": "Mock Job",
                    "company": "Mock Corp",
                },
                sources={},
                confidence=0.85,
                metadata={"processing_time": 0.01},
            )

        pipeline.__call__ = fast_extract
        pipeline.extractor = MagicMock()
        pipeline.extractor.use_sources = True

        return pipeline

    return mock_pipeline_factory


class TestPerformanceBaseline:
    """Baseline performance tests."""

    def test_schema_generation_performance(self):
        """Test schema generation performance."""
        example = {
            "name": "John Doe",
            "age": 30,
            "location": "New York",
            "skills": ["Python", "JavaScript", "SQL"],
            "experience": 5.5,
            "active": True,
        }

        # Time schema generation
        start_time = time.time()
        for _ in range(100):  # Generate 100 schemas
            schema = LangStruct(example=example, schema_name=f"TestSchema{_}")
        end_time = time.time()

        avg_time = (end_time - start_time) / 100

        # Should generate schema in reasonable time
        assert (
            avg_time < 0.05
        ), f"Schema generation too slow: {avg_time:.4f}s per schema"

    def test_multiple_examples_performance(self):
        """Test performance with multiple examples."""
        examples = [
            {"name": "John", "age": 30, "city": "NYC"},
            {"name": "Jane", "age": 25, "city": "LA", "skills": ["Python"]},
            {"name": "Bob", "age": 35, "city": "Chicago", "experience": 10},
        ]

        start_time = time.time()
        for _ in range(50):
            schema = LangStruct(examples=examples, schema_name=f"MultiSchema{_}")
        end_time = time.time()

        avg_time = (end_time - start_time) / 50
        assert (
            avg_time < 0.02
        ), f"Multi-example schema generation too slow: {avg_time:.4f}s"

    @patch("langstruct.core.modules.ExtractionPipeline")
    def test_extraction_initialization_performance(
        self, mock_pipeline_class, fast_mock_pipeline
    ):
        """Test LangStruct initialization performance."""
        mock_pipeline_class.side_effect = fast_mock_pipeline

        start_time = time.time()
        for _ in range(50):
            extractor = LangStruct(schema=BenchmarkSchema, model="gpt-5-mini")
        end_time = time.time()

        avg_time = (end_time - start_time) / 50
        assert avg_time < 0.05, f"Extraction initialization too slow: {avg_time:.4f}s"

    def test_validation_performance(self):
        """Test validation performance."""
        from langstruct.core.schemas import ExtractionResult
        from langstruct.core.validation import ExtractionValidator

        validator = ExtractionValidator(BenchmarkSchema)

        result = ExtractionResult(
            entities={
                "name": "John Doe",
                "age": 30,
                "location": "NYC",
                "occupation": "Developer",
                "company": "TechCorp",
            },
            sources={},
            confidence=0.85,
        )

        start_time = time.time()
        for _ in range(100):
            report = validator.validate(result)
        end_time = time.time()

        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01, f"Validation too slow: {avg_time:.4f}s per validation"

    def test_export_performance(self):
        """Test export utilities performance."""
        from langstruct.core.export_utils import ExportUtilities
        from langstruct.core.schemas import ExtractionResult

        # Create test results
        results = []
        for i in range(1000):
            result = ExtractionResult(
                entities={
                    "name": f"Person {i}",
                    "age": 25 + (i % 50),
                    "location": f"City {i % 10}",
                    "occupation": f"Job {i % 20}",
                    "company": f"Company {i % 100}",
                },
                sources={},
                confidence=0.8 + (i % 20) * 0.01,
            )
            results.append(result)

        # Test dict conversion performance
        start_time = time.time()
        dicts = [ExportUtilities.to_dict(result) for result in results]
        dict_time = time.time() - start_time

        assert (
            dict_time < 1.0
        ), f"Dict conversion too slow: {dict_time:.2f}s for 1000 results"

        # Test JSON conversion performance
        start_time = time.time()
        json_strs = [
            ExportUtilities.to_json(result, indent=None) for result in results[:100]
        ]
        json_time = time.time() - start_time

        assert (
            json_time < 1.0
        ), f"JSON conversion too slow: {json_time:.2f}s for 100 results"


@pytest.mark.skipif(
    not hasattr(pytest, "benchmark"), reason="pytest-benchmark not available"
)
class TestBenchmarks:
    """Benchmarking tests using pytest-benchmark."""

    def test_schema_generation_benchmark(self, benchmark):
        """Benchmark schema generation."""
        example = {"name": "John", "age": 30, "location": "NYC"}

        result = benchmark(lambda x: LangStruct(example=x), example)
        assert result is not None

    def test_validation_benchmark(self, benchmark):
        """Benchmark validation."""
        from langstruct.core.schemas import ExtractionResult
        from langstruct.core.validation import ExtractionValidator

        validator = ExtractionValidator(BenchmarkSchema)
        result = ExtractionResult(
            entities={"name": "John", "age": 30}, sources={}, confidence=0.8
        )

        report = benchmark(validator.validate, result)
        assert report is not None

    def test_export_benchmark(self, benchmark):
        """Benchmark export utilities."""
        from langstruct.core.export_utils import ExportUtilities
        from langstruct.core.schemas import ExtractionResult

        result = ExtractionResult(
            entities={"name": "John", "age": 30}, sources={}, confidence=0.8
        )

        output = benchmark(ExportUtilities.to_dict, result)
        assert output is not None


class TestMemoryUsage:
    """Tests for memory usage patterns."""

    def test_schema_generation_memory(self):
        """Test that schema generation doesn't leak memory."""
        import gc
        import sys

        # Get baseline memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Generate many schemas
        for i in range(100):
            schema = LangStruct(
                example={"field1": "value", "field2": i}, schema_name=f"TestSchema{i}"
            )
            # Don't keep reference to schema

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not have significantly more objects
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Too many objects created: {object_growth}"

    @patch("langstruct.core.modules.ExtractionPipeline")
    def test_extractor_memory(self, mock_pipeline_class, fast_mock_pipeline):
        """Test extractor memory usage."""
        mock_pipeline_class.side_effect = fast_mock_pipeline

        import gc

        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create many extractors
        extractors = []
        for i in range(50):
            extractor = LangStruct(schema=BenchmarkSchema)
            extractors.append(extractor)

        # Clear references
        extractors.clear()
        gc.collect()

        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects

        # Should clean up properly
        assert (
            object_growth < 500
        ), f"Memory leak detected: {object_growth} objects retained"


class TestScalability:
    """Tests for scalability with large inputs."""

    def test_large_text_chunking(self):
        """Test performance with large text inputs."""
        from langstruct.core.chunking import ChunkingConfig, TextChunker

        # Create large text (100KB)
        large_text = "This is a test sentence. " * 4000  # ~100KB

        config = ChunkingConfig(max_tokens=1000, overlap_tokens=100)
        chunker = TextChunker(config)

        start_time = time.time()
        chunks = chunker.chunk_text(large_text)
        chunk_time = time.time() - start_time

        assert (
            chunk_time < 1.0
        ), f"Text chunking too slow: {chunk_time:.2f}s for 100KB text"
        assert len(chunks) > 1, "Should create multiple chunks for large text"

    def test_many_fields_schema(self):
        """Test performance with schemas having many fields."""
        # Create example with many fields
        large_example = {f"field_{i}": f"value_{i}" for i in range(50)}

        start_time = time.time()
        schema = LangStruct(example=large_example, schema_name="LargeSchema")
        generation_time = time.time() - start_time

        assert (
            generation_time < 0.1
        ), f"Large schema generation too slow: {generation_time:.2f}s"

        # Verify schema was created correctly
        from langstruct.core.schema_utils import get_field_descriptions

        field_descriptions = get_field_descriptions(schema.schema)
        assert len(field_descriptions) == 50

    def test_batch_processing_scalability(self):
        """Test batch processing with many texts."""
        from langstruct.core.schemas import ExtractionResult

        # Simulate batch processing results
        results = []
        for i in range(1000):
            result = ExtractionResult(
                entities={f"field_{j}": f"value_{i}_{j}" for j in range(5)},
                sources={},
                confidence=0.8,
            )
            results.append(result)

        # Test batch export performance
        from langstruct.core.export_utils import ExportUtilities

        start_time = time.time()
        dicts = [ExportUtilities.to_dict(r, include_metadata=False) for r in results]
        export_time = time.time() - start_time

        assert (
            export_time < 2.0
        ), f"Batch export too slow: {export_time:.2f}s for 1000 results"
        assert len(dicts) == 1000


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    @pytest.fixture(autouse=True)
    def setup_performance_baseline(self):
        """Set up performance baselines."""
        self.baselines = {
            "schema_generation": 0.01,  # 10ms per schema
            "validation": 0.005,  # 5ms per validation
            "dict_export": 0.001,  # 1ms per dict conversion
            "json_export": 0.01,  # 10ms per JSON conversion
        }

    def test_schema_generation_regression(self):
        """Test for schema generation performance regression."""
        example = {"name": "John", "age": 30, "skills": ["Python"]}

        start_time = time.time()
        for _ in range(10):
            LangStruct(example=example, schema_name=f"Schema{_}")
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        baseline = self.baselines["schema_generation"]

        # Allow 50% variance from baseline
        assert avg_time < baseline * 1.5, (
            f"Performance regression in schema generation: "
            f"{avg_time:.4f}s vs baseline {baseline:.4f}s"
        )

    def test_validation_regression(self):
        """Test for validation performance regression."""
        from langstruct.core.schemas import ExtractionResult
        from langstruct.core.validation import ExtractionValidator

        validator = ExtractionValidator(BenchmarkSchema)
        result = ExtractionResult(
            entities={"name": "John", "age": 30}, sources={}, confidence=0.8
        )

        start_time = time.time()
        for _ in range(20):
            validator.validate(result)
        end_time = time.time()

        avg_time = (end_time - start_time) / 20
        baseline = self.baselines["validation"]

        assert avg_time < baseline * 1.5, (
            f"Performance regression in validation: "
            f"{avg_time:.4f}s vs baseline {baseline:.4f}s"
        )

    def test_export_regression(self):
        """Test for export performance regression."""
        from langstruct.core.export_utils import ExportUtilities
        from langstruct.core.schemas import ExtractionResult

        result = ExtractionResult(
            entities={"name": "John", "age": 30, "location": "NYC"},
            sources={},
            confidence=0.8,
        )

        # Test dict export
        start_time = time.time()
        for _ in range(100):
            ExportUtilities.to_dict(result)
        dict_time = (time.time() - start_time) / 100

        # Test JSON export
        start_time = time.time()
        for _ in range(50):
            ExportUtilities.to_json(result, indent=None)
        json_time = (time.time() - start_time) / 50

        dict_baseline = self.baselines["dict_export"]
        json_baseline = self.baselines["json_export"]

        assert (
            dict_time < dict_baseline * 1.5
        ), f"Dict export regression: {dict_time:.4f}s vs {dict_baseline:.4f}s"
        assert (
            json_time < json_baseline * 1.5
        ), f"JSON export regression: {json_time:.4f}s vs {json_baseline:.4f}s"
