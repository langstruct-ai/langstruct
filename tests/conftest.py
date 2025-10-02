"""Pytest configuration and shared fixtures for LangStruct tests."""

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import BaseModel, Field

from langstruct import LangStruct
from langstruct.core.schemas import ExtractionResult, SourceSpan


# Test schemas
class PersonSchema(BaseModel):
    """Test schema for person information."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    location: str = Field(description="Current location")


class CompanySchema(BaseModel):
    """Test schema for company information."""

    company_name: str = Field(description="Full legal name of the company")
    founded_year: int = Field(description="Year when company was founded")
    headquarters: str = Field(description="Primary headquarters location")
    employees: int = Field(description="Number of employees")


@pytest.fixture
def person_schema():
    """Provide PersonSchema for tests."""
    return PersonSchema


@pytest.fixture
def company_schema():
    """Provide CompanySchema for tests."""
    return CompanySchema


@pytest.fixture
def sample_person_text():
    """Provide sample text for person extraction."""
    return """
    Dr. Sarah Johnson is a 34-year-old cardiologist working at Boston General Hospital.
    She completed her medical degree at Harvard Medical School and has been practicing
    medicine for over 8 years. Sarah currently lives in Cambridge, Massachusetts,
    with her family.
    """


@pytest.fixture
def sample_company_text():
    """Provide sample text for company extraction."""
    return """
    Microsoft Corporation was founded in 1975 and is headquartered in Redmond, Washington.
    The technology company employs approximately 238,000 people worldwide and is led by
    CEO Satya Nadella.
    """


@pytest.fixture
def sample_extraction_result():
    """Provide a sample ExtractionResult for testing."""
    return ExtractionResult(
        entities={
            "name": "Dr. Sarah Johnson",
            "age": 34,
            "location": "Cambridge, Massachusetts",
        },
        sources={
            "name": [SourceSpan(start=5, end=19, text="Sarah Johnson")],
            "age": [SourceSpan(start=25, end=27, text="34")],
            "location": [
                SourceSpan(start=180, end=202, text="Cambridge, Massachusetts")
            ],
        },
        confidence=0.92,
        metadata={
            "pipeline": "langstruct",
            "total_chunks": 1,
            "original_text_length": 300,
        },
    )


@pytest.fixture
def mock_dspy_lm():
    """Provide a test DSPy language model."""

    class TestLM:
        def __init__(self):
            self.model = "test-model"
            self.kwargs = {"model": "test-model"}
            self._test_model_name = "test-model"

        def __call__(self, *args, **kwargs):
            # Return a mock response that looks like DSPy response
            return MagicMock(
                entities='{"name": "Test Name", "age": 25, "location": "Test City"}',
                sources='{"name": [{"start": 0, "end": 9, "text": "Test Name"}]}',
                is_valid=True,
                feedback="The extraction looks good",
            )

    return TestLM()


@pytest.fixture
def person_example_data():
    """Provide example data for schema generation."""
    return {"name": "John Doe", "age": 30, "location": "New York, NY"}


@pytest.fixture
def company_examples_data():
    """Provide multiple examples for schema generation."""
    return [
        {
            "company": "Apple Inc.",
            "founded": 1976,
            "headquarters": "Cupertino, CA",
            "employees": 154000,
        },
        {
            "company": "Google LLC",
            "founded": 1998,
            "headquarters": "Mountain View, CA",
            "employees": 174000,
            "ceo": "Sundar Pichai",  # Optional field
        },
    ]


# Test configuration - use Gemini 2.5 Flash for testing
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RUN_INTEGRATION_TESTS = GOOGLE_API_KEY is not None or OPENAI_API_KEY is not None


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment - either real API or mocks."""
    if RUN_INTEGRATION_TESTS:
        # Use real API - configure DSPy
        import dspy

        if GOOGLE_API_KEY:
            dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash-lite"))
            print(f"\n✅ Running tests with Gemini 2.5 Flash")
        elif OPENAI_API_KEY:
            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
            print(f"\n✅ Running tests with OpenAI GPT-4o-mini")
    else:
        print(f"\n⚠️  No API key found - some tests will be skipped")
        print(
            f"   Set GOOGLE_API_KEY (preferred) or OPENAI_API_KEY to run integration tests"
        )


@pytest.fixture
def requires_api_key():
    """Mark tests that require an API key."""
    if not RUN_INTEGRATION_TESTS:
        pytest.skip("Requires GOOGLE_API_KEY or OPENAI_API_KEY environment variable")


# Add pytest marker for integration tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring API key"
    )


@pytest.fixture(autouse=True)
def mock_expensive_calls(monkeypatch):
    """Mock only expensive operations, allow real API calls for basic extraction."""
    if not RUN_INTEGRATION_TESTS:
        # Only mock when no API key available
        from langstruct import api as langstruct_api

        BaseLM = langstruct_api.dspy.BaseLM

        class TestLM(BaseLM):
            """Test LM that properly mimics real DSPy LM interface for tests."""

            def __init__(self, model: str = "test-model", **kwargs):
                self.model = model
                self.kwargs = dict(kwargs) or {"model": model}
                # Special marker for persistence layer to identify test models
                self._test_model_name = (
                    f"test-{model}" if not model.startswith("test-") else model
                )

            def __call__(self, *args, **kwargs):
                return MagicMock()

        def _fake_create_lm(model_name: str, *args, **kwargs):
            if isinstance(model_name, BaseLM):
                return model_name
            resolved_model = model_name or kwargs.get("model", "test-model")
            return TestLM(resolved_model, **kwargs)

        monkeypatch.setattr(
            "langstruct.providers.llm_factory.LLMFactory.create_lm",
            _fake_create_lm,
        )
        monkeypatch.setattr(
            "langstruct.core.modules.ExtractionPipeline", MockExtractionPipeline
        )


class MockExtractionPipeline:
    """Mock extraction pipeline for testing."""

    def __init__(self, schema):
        self.schema = schema
        self.extractor = MagicMock()
        self.extractor.use_sources = True

    def __call__(self, text: str) -> ExtractionResult:
        """Mock extraction that returns predictable results."""
        # Generate mock results based on schema
        entities = {}
        sources = {}

        from langstruct.core.schema_utils import get_field_descriptions

        field_descriptions = get_field_descriptions(self.schema)
        for field_name in field_descriptions.keys():
            if field_name == "name":
                entities[field_name] = "Mock Name"
                sources[field_name] = [SourceSpan(start=0, end=9, text="Mock Name")]
            elif field_name == "age":
                entities[field_name] = 25
                sources[field_name] = [SourceSpan(start=10, end=12, text="25")]
            elif field_name in ["location", "headquarters", "city"]:
                entities[field_name] = "Mock Location"
                sources[field_name] = [
                    SourceSpan(start=20, end=33, text="Mock Location")
                ]
            elif field_name in ["company_name", "company"]:
                entities[field_name] = "Mock Company"
                sources[field_name] = [SourceSpan(start=0, end=12, text="Mock Company")]
            elif field_name in ["founded_year", "founded"]:
                entities[field_name] = 2000
                sources[field_name] = [SourceSpan(start=15, end=19, text="2000")]
            elif field_name == "employees":
                entities[field_name] = 1000
                sources[field_name] = [SourceSpan(start=25, end=29, text="1000")]

        return ExtractionResult(
            entities=entities,
            sources=sources,
            confidence=0.85,
            metadata={
                "pipeline": "langstruct",
                "total_chunks": 1,
                "original_text_length": len(text),
            },
        )

    def save(self, path: str, save_program: bool = False):
        """Mock save method for testing."""
        import json
        from pathlib import Path

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a simple mock save file
        mock_data = {"schema_name": self.schema.__name__, "mock_pipeline": True}

        with open(save_path, "w") as f:
            json.dump(mock_data, f)

    def load(self, path: str):
        """Mock load method for testing."""
        # This is a mock, so we don't actually need to load anything
        pass


@pytest.fixture
def mock_extraction_pipeline(monkeypatch):
    """Replace ExtractionPipeline with mock for testing."""

    def mock_pipeline_factory(*args, **kwargs):
        schema = args[0] if args else kwargs.get("schema")
        return MockExtractionPipeline(schema)

    monkeypatch.setattr(
        "langstruct.core.modules.ExtractionPipeline", mock_pipeline_factory
    )
    return mock_pipeline_factory
