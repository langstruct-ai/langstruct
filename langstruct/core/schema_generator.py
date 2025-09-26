"""Schema auto-generation from example data for improved DX."""

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import create_model
from typing_extensions import get_args, get_origin

from .schema_utils import _LangStructBaseModel, ensure_schema_class


class SchemaGenerator:
    """Generates Pydantic schemas from example data."""

    @staticmethod
    def from_example(
        example_data: Dict[str, Any],
        schema_name: str = "GeneratedSchema",
        descriptions: Optional[Dict[str, str]] = None,
    ) -> Type[BaseModel]:
        """Generate a Schema class from example data.

        Args:
            example_data: Dictionary with example field values
            schema_name: Name for the generated schema class
            descriptions: Optional custom descriptions for fields

        Returns:
            Generated Schema class

        Example:
            >>> example = {"name": "John Doe", "age": 25, "skills": ["Python", "ML"]}
            >>> PersonSchema = SchemaGenerator.from_example(example, "PersonSchema")
            >>> extractor = LangStruct(schema=PersonSchema)
        """
        descriptions = descriptions or {}
        fields = {}

        for field_name, example_value in example_data.items():
            field_type = SchemaGenerator._infer_type(example_value)
            field_desc = descriptions.get(
                field_name,
                SchemaGenerator._generate_description(field_name, example_value),
            )

            fields[field_name] = (field_type, PydanticField(description=field_desc))

        # Create the schema class dynamically
        GeneratedSchema = create_model(
            schema_name, __base__=_LangStructBaseModel, **fields
        )

        return ensure_schema_class(GeneratedSchema)

    @staticmethod
    def from_examples(
        examples: List[Dict[str, Any]],
        schema_name: str = "GeneratedSchema",
        descriptions: Optional[Dict[str, str]] = None,
    ) -> Type[BaseModel]:
        """Generate schema from multiple examples for better type inference.

        Args:
            examples: List of example dictionaries
            schema_name: Name for the generated schema class
            descriptions: Optional custom descriptions for fields

        Returns:
            Generated Schema class
        """
        if not examples:
            raise ValueError("At least one example is required")

        # Merge all field names from all examples
        all_fields = set()
        for example in examples:
            all_fields.update(example.keys())

        # Analyze each field across all examples
        field_analysis = {}
        for field_name in all_fields:
            values = [ex.get(field_name) for ex in examples if field_name in ex]
            non_none_values = [v for v in values if v is not None]

            if non_none_values:
                field_analysis[field_name] = {
                    "values": non_none_values,
                    "is_optional": len(non_none_values) < len(examples),
                    "example_value": non_none_values[0],
                }
            else:
                field_analysis[field_name] = {
                    "values": [],
                    "is_optional": True,
                    "example_value": None,
                }

        descriptions = descriptions or {}
        fields = {}

        for field_name, analysis in field_analysis.items():
            if analysis["values"]:
                # Infer type from multiple values
                field_type = SchemaGenerator._infer_type_from_multiple(
                    analysis["values"]
                )

                # Make optional if not present in all examples
                if analysis["is_optional"]:
                    field_type = Optional[field_type]

            else:
                # No values found, default to optional string
                field_type = Optional[str]
                analysis["is_optional"] = True

            field_desc = descriptions.get(
                field_name,
                SchemaGenerator._generate_description(
                    field_name, analysis["example_value"]
                ),
            )

            # Set default value for optional fields
            if analysis["is_optional"]:
                fields[field_name] = (
                    field_type,
                    PydanticField(default=None, description=field_desc),
                )
            else:
                fields[field_name] = (field_type, PydanticField(description=field_desc))

        # Create the schema class dynamically
        GeneratedSchema = create_model(
            schema_name, __base__=_LangStructBaseModel, **fields
        )

        return ensure_schema_class(GeneratedSchema)

    @staticmethod
    def _create_schema_from_json(
        json_schema: Dict[str, Any],
        schema_name: str,
        field_descriptions: Dict[str, str],
    ) -> Type[BaseModel]:
        """Create a Schema class from a JSON schema definition.

        This is used to reconstruct schemas when loading saved extractors.

        Args:
            json_schema: JSON schema definition
            schema_name: Name for the schema class
            field_descriptions: Field descriptions

        Returns:
            Generated Schema class
        """
        fields = {}

        # Extract properties from JSON schema
        properties = json_schema.get("properties", {})
        required_fields = set(json_schema.get("required", []))

        for field_name, field_def in properties.items():
            # Convert JSON schema type to Python type
            field_type = SchemaGenerator._json_type_to_python_type(field_def)

            # Make optional if not in required fields
            if field_name not in required_fields:
                field_type = Optional[field_type]

            # Get description
            description = field_descriptions.get(
                field_name, field_def.get("description", f"{field_name} field")
            )

            # Create field definition
            if field_name not in required_fields:
                fields[field_name] = (
                    field_type,
                    PydanticField(default=None, description=description),
                )
            else:
                fields[field_name] = (
                    field_type,
                    PydanticField(description=description),
                )

        # Create the schema class dynamically
        GeneratedSchema = create_model(
            schema_name, __base__=_LangStructBaseModel, **fields
        )

        return ensure_schema_class(GeneratedSchema)

    @staticmethod
    def _json_type_to_python_type(field_def: Dict[str, Any]) -> Type:
        """Convert JSON schema type definition to Python type."""
        json_type = field_def.get("type", "string")

        if json_type == "string":
            return str
        elif json_type == "integer":
            return int
        elif json_type == "number":
            return float
        elif json_type == "boolean":
            return bool
        elif json_type == "array":
            # Get items type if specified
            items_def = field_def.get("items", {})
            if items_def:
                item_type = SchemaGenerator._json_type_to_python_type(items_def)
                return List[item_type]
            else:
                return List[str]  # Default to List[str]
        elif json_type == "object":
            return Dict[str, Any]
        else:
            # Default to string for unknown types
            return str

    @staticmethod
    def _infer_type(value: Any) -> Type:
        """Infer Python type from a single value."""
        if value is None:
            return Optional[str]  # Default to optional string for None values

        value_type = type(value)

        # Handle basic types
        if value_type in (str, int, float, bool):
            return value_type

        # Handle lists
        elif isinstance(value, list):
            if not value:
                return List[str]  # Default to List[str] for empty lists

            # Infer type from first element (could be improved to check all elements)
            element_type = SchemaGenerator._infer_type(value[0])
            return List[element_type]

        # Handle dictionaries
        elif isinstance(value, dict):
            return Dict[str, Any]

        # Handle date/datetime
        elif isinstance(value, (date, datetime)):
            return str  # Convert to string for LLM compatibility

        # Default to string for unknown types
        else:
            return str

    @staticmethod
    def _infer_type_from_multiple(values: List[Any]) -> Type:
        """Infer type from multiple example values."""
        if not values:
            return str

        # Get types of all values
        types = [type(v) for v in values if v is not None]

        if not types:
            return Optional[str]

        # If all values have the same type, use that
        unique_types = set(types)
        if len(unique_types) == 1:
            return SchemaGenerator._infer_type(values[0])

        # Handle numeric type mixing (int + float = float)
        if unique_types == {int, float}:
            return float

        # If types are mixed, check for common patterns
        if all(isinstance(v, (str, int, float, bool)) for v in values):
            # Mixed basic types - use string as common denominator
            return str

        # For lists, try to find common element type
        if all(isinstance(v, list) for v in values):
            all_elements = []
            for lst in values:
                all_elements.extend(lst)

            if all_elements:
                element_type = SchemaGenerator._infer_type_from_multiple(all_elements)
                return List[element_type]
            else:
                return List[str]

        # Default to string for complex mixed types
        return str

    @staticmethod
    def _generate_description(field_name: str, example_value: Any) -> str:
        """Generate a helpful description for a field based on its name and example."""
        # Clean field name for description
        clean_name = field_name.replace("_", " ").replace("-", " ").title()

        # Type-based descriptions
        if isinstance(example_value, str):
            if "email" in field_name.lower():
                return f"Email address"
            elif "name" in field_name.lower():
                return f"Name or title"
            elif "address" in field_name.lower() or "location" in field_name.lower():
                return f"Address or location"
            elif "phone" in field_name.lower():
                return f"Phone number"
            elif "url" in field_name.lower() or "link" in field_name.lower():
                return f"URL or web link"
            else:
                return f"{clean_name} as text"

        elif isinstance(example_value, int):
            if "age" in field_name.lower():
                return f"Age in years"
            elif "year" in field_name.lower():
                return f"Year (numeric)"
            elif "count" in field_name.lower() or "number" in field_name.lower():
                return f"Count or quantity"
            else:
                return f"{clean_name} (integer number)"

        elif isinstance(example_value, float):
            if "price" in field_name.lower() or "cost" in field_name.lower():
                return f"Price or cost amount"
            elif "score" in field_name.lower() or "rating" in field_name.lower():
                return f"Score or rating value"
            else:
                return f"{clean_name} (decimal number)"

        elif isinstance(example_value, bool):
            return f"True/false flag for {clean_name.lower()}"

        elif isinstance(example_value, list):
            if "skill" in field_name.lower():
                return f"List of skills or abilities"
            elif "tag" in field_name.lower():
                return f"List of tags or categories"
            else:
                return f"List of {clean_name.lower()}"

        elif isinstance(example_value, dict):
            return f"{clean_name} details (structured data)"

        else:
            return f"{clean_name}"


# Convenience functions for direct use
def schema_from_example(
    example_data: Dict[str, Any],
    schema_name: str = "GeneratedSchema",
    descriptions: Optional[Dict[str, str]] = None,
) -> Type[BaseModel]:
    """Generate a Schema class from example data (convenience function)."""
    return SchemaGenerator.from_example(example_data, schema_name, descriptions)


def schema_from_examples(
    examples: List[Dict[str, Any]],
    schema_name: str = "GeneratedSchema",
    descriptions: Optional[Dict[str, str]] = None,
) -> Type[BaseModel]:
    """Generate a Schema class from multiple examples (convenience function)."""
    return SchemaGenerator.from_examples(examples, schema_name, descriptions)
