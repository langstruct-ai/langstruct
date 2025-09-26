"""Tests for schema auto-generation functionality."""

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, ValidationError

from langstruct.core.schema_generator import (
    SchemaGenerator,
    schema_from_example,
    schema_from_examples,
)
from langstruct.core.schema_utils import get_field_descriptions, get_json_schema


class TestSchemaGenerator:
    """Tests for SchemaGenerator class."""

    def test_from_example_basic(self, person_example_data):
        """Test basic schema generation from single example."""
        GeneratedSchema = SchemaGenerator.from_example(
            person_example_data, "PersonSchema"
        )

        # Check class creation
        assert GeneratedSchema.__name__ == "PersonSchema"
        assert issubclass(GeneratedSchema, BaseModel)

        # Check fields
        field_descriptions = get_field_descriptions(GeneratedSchema)
        assert "name" in field_descriptions
        assert "age" in field_descriptions
        assert "location" in field_descriptions

        # Check JSON schema types
        json_schema = get_json_schema(GeneratedSchema)
        properties = json_schema["properties"]

        assert properties["name"]["type"] == "string"
        assert properties["age"]["type"] == "integer"
        assert properties["location"]["type"] == "string"

    def test_from_example_with_descriptions(self, person_example_data):
        """Test schema generation with custom descriptions."""
        custom_descriptions = {
            "name": "Custom name description",
            "age": "Custom age description",
        }

        GeneratedSchema = SchemaGenerator.from_example(
            person_example_data, descriptions=custom_descriptions
        )

        field_descriptions = get_field_descriptions(GeneratedSchema)
        assert field_descriptions["name"] == "Custom name description"
        assert field_descriptions["age"] == "Custom age description"
        # Should have auto-generated description for location
        assert "location" in field_descriptions["location"]

    def test_from_example_complex_types(self):
        """Test schema generation with complex data types."""
        example = {
            "name": "John Doe",
            "age": 30,
            "skills": ["Python", "JavaScript"],
            "metadata": {"active": True, "score": 95.5},
            "tags": [],
            "optional_field": None,
        }

        GeneratedSchema = SchemaGenerator.from_example(example)
        json_schema = get_json_schema(GeneratedSchema)
        properties = json_schema["properties"]

        # Check inferred types
        assert properties["name"]["type"] == "string"
        assert properties["age"]["type"] == "integer"
        assert properties["skills"]["type"] == "array"
        assert properties["metadata"]["type"] == "object"

        # Test instance creation
        instance = GeneratedSchema(
            name="Test",
            age=25,
            skills=["test"],
            metadata={"key": "value"},
            tags=[],
            optional_field="value",
        )
        assert instance.name == "Test"

    def test_from_examples_multiple(self, company_examples_data):
        """Test schema generation from multiple examples."""
        GeneratedSchema = SchemaGenerator.from_examples(
            company_examples_data, "CompanySchema"
        )

        # Check class creation
        assert GeneratedSchema.__name__ == "CompanySchema"

        field_descriptions = get_field_descriptions(GeneratedSchema)
        json_schema = get_json_schema(GeneratedSchema)

        # Check all fields from both examples are present
        assert "company" in field_descriptions
        assert "founded" in field_descriptions
        assert "headquarters" in field_descriptions
        assert "employees" in field_descriptions
        assert "ceo" in field_descriptions  # Only in second example

        # CEO should be optional since it's not in first example
        properties = json_schema["properties"]
        # The way we check for optional fields depends on implementation
        # but the field should exist
        assert "ceo" in properties

    def test_from_examples_empty_list(self):
        """Test error handling for empty examples list."""
        with pytest.raises(ValueError, match="At least one example is required"):
            SchemaGenerator.from_examples([])

    def test_type_inference_strings(self):
        """Test type inference for string values."""
        examples = [
            {"email": "john@example.com"},
            {"phone": "555-1234"},
            {"name": "John Doe"},
            {"address": "123 Main St"},
            {"url": "https://example.com"},
        ]

        for example in examples:
            GeneratedSchema = SchemaGenerator.from_example(example)
            json_schema = get_json_schema(GeneratedSchema)
            field_name = list(example.keys())[0]
            assert json_schema["properties"][field_name]["type"] == "string"

    def test_type_inference_numbers(self):
        """Test type inference for numeric values."""
        int_example = {"count": 42}
        float_example = {"price": 19.99}

        # Integer
        IntSchema = SchemaGenerator.from_example(int_example)
        int_schema = get_json_schema(IntSchema)
        assert int_schema["properties"]["count"]["type"] == "integer"

        # Float
        FloatSchema = SchemaGenerator.from_example(float_example)
        float_schema = get_json_schema(FloatSchema)
        assert float_schema["properties"]["price"]["type"] == "number"

    def test_type_inference_mixed_numbers(self):
        """Test type inference with mixed int/float values."""
        examples = [{"value": 42}, {"value": 19.99}]  # int  # float

        GeneratedSchema = SchemaGenerator.from_examples(examples)
        json_schema = get_json_schema(GeneratedSchema)

        # Should infer as float when mixing int and float
        assert json_schema["properties"]["value"]["type"] == "number"

    def test_type_inference_lists(self):
        """Test type inference for list values."""
        examples = [
            {"skills": ["Python", "JavaScript"]},
            {"tags": ["web", "api"]},
            {"numbers": [1, 2, 3]},
            {"empty_list": []},
        ]

        for example in examples:
            GeneratedSchema = SchemaGenerator.from_example(example)
            json_schema = get_json_schema(GeneratedSchema)
            field_name = list(example.keys())[0]
            assert json_schema["properties"][field_name]["type"] == "array"

    def test_description_generation_names(self):
        """Test automatic description generation for name fields."""
        example = {
            "name": "John Doe",
            "first_name": "John",
            "last_name": "Doe",
            "company_name": "Acme Corp",
        }

        GeneratedSchema = SchemaGenerator.from_example(example)
        descriptions = get_field_descriptions(GeneratedSchema)

        # Should contain "name" in descriptions for name fields
        for field in ["name", "first_name", "last_name", "company_name"]:
            assert (
                "name" in descriptions[field].lower()
                or "title" in descriptions[field].lower()
            )

    def test_description_generation_locations(self):
        """Test automatic description generation for location fields."""
        example = {
            "address": "123 Main St",
            "location": "New York",
            "headquarters": "San Francisco",
        }

        GeneratedSchema = SchemaGenerator.from_example(example)
        descriptions = get_field_descriptions(GeneratedSchema)

        # Should contain location-related terms
        for field in ["address", "location", "headquarters"]:
            desc = descriptions[field].lower()
            assert any(term in desc for term in ["address", "location", "headquarters"])

    def test_description_generation_contact(self):
        """Test description generation for contact fields."""
        example = {
            "email": "john@example.com",
            "phone": "555-1234",
            "url": "https://example.com",
        }

        GeneratedSchema = SchemaGenerator.from_example(example)
        descriptions = get_field_descriptions(GeneratedSchema)

        assert "email" in descriptions["email"].lower()
        assert "phone" in descriptions["phone"].lower()
        assert (
            "url" in descriptions["url"].lower()
            or "link" in descriptions["url"].lower()
        )

    def test_none_value_handling(self):
        """Test handling of None values in examples."""
        example = {"required_field": "value", "optional_field": None}

        GeneratedSchema = SchemaGenerator.from_example(example)

        # Should be able to create instance with None for optional field
        instance = GeneratedSchema(required_field="test", optional_field=None)
        assert instance.optional_field is None

    def test_multiple_examples_optional_detection(self):
        """Test detection of optional fields from multiple examples."""
        examples = [
            {"name": "John", "age": 30, "email": "john@example.com"},
            {"name": "Jane", "age": 25},  # No email
            {
                "name": "Bob",
                "age": 40,
                "email": "bob@example.com",
                "phone": "555-1234",
            },  # Has phone
        ]

        GeneratedSchema = SchemaGenerator.from_examples(examples)

        # Should be able to create instances with missing optional fields
        instance1 = GeneratedSchema(name="Test1", age=20)  # No email, phone
        assert instance1.name == "Test1"
        assert instance1.age == 20

        instance2 = GeneratedSchema(name="Test2", age=25, email="test@example.com")
        assert instance2.email == "test@example.com"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_schema_from_example(self, person_example_data):
        """Test schema_from_example convenience function."""
        GeneratedSchema = schema_from_example(person_example_data, "TestSchema")

        assert GeneratedSchema.__name__ == "TestSchema"
        assert issubclass(GeneratedSchema, BaseModel)

        # Should work the same as SchemaGenerator.from_example
        field_descriptions = get_field_descriptions(GeneratedSchema)
        assert "name" in field_descriptions
        assert "age" in field_descriptions
        assert "location" in field_descriptions

    def test_schema_from_examples(self, company_examples_data):
        """Test schema_from_examples convenience function."""
        GeneratedSchema = schema_from_examples(company_examples_data, "CompanySchema")

        assert GeneratedSchema.__name__ == "CompanySchema"
        assert issubclass(GeneratedSchema, BaseModel)

        # Should work the same as SchemaGenerator.from_examples
        field_descriptions = get_field_descriptions(GeneratedSchema)
        assert "company" in field_descriptions
        assert "founded" in field_descriptions
