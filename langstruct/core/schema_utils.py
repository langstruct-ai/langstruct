"""Utility functions for working with Pydantic schemas in LangStruct."""

import json
from typing import Any, Dict, Type
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict


def get_json_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Get JSON schema representation for LLM consumption.

    Args:
        model_class: Any Pydantic BaseModel class

    Returns:
        JSON schema dictionary
    """
    return model_class.model_json_schema()


def get_field_descriptions(model_class: Type[BaseModel]) -> Dict[str, str]:
    """Get field descriptions for prompt engineering.

    Args:
        model_class: Any Pydantic BaseModel class

    Returns:
        Dictionary mapping field names to descriptions
    """
    schema = get_json_schema(model_class)
    descriptions = {}

    if "properties" in schema:
        for field_name, field_info in schema["properties"].items():
            descriptions[field_name] = field_info.get("description", "")

    return descriptions


def get_example_format(model_class: Type[BaseModel]) -> str:
    """Get example JSON format for few-shot prompting.

    Args:
        model_class: Any Pydantic BaseModel class

    Returns:
        JSON string with example format
    """
    # Create a simple example based on field types
    example = {}
    schema = get_json_schema(model_class)

    if "properties" in schema:
        for field_name, field_info in schema["properties"].items():
            field_type = field_info.get("type", "string")
            if field_type == "string":
                example[field_name] = f"<{field_name}>"
            elif field_type == "integer":
                example[field_name] = 0
            elif field_type == "number":
                example[field_name] = 0.0
            elif field_type == "boolean":
                example[field_name] = True
            elif field_type == "array":
                example[field_name] = []
            else:
                example[field_name] = {}

    return json.dumps(example, indent=2)


def validate_schema_class(model_class: Type[BaseModel]) -> None:
    """Validate that a class is a proper Pydantic model for LangStruct.

    Args:
        model_class: Class to validate

    Raises:
        TypeError: If the class is not a valid Pydantic BaseModel
    """
    if not isinstance(model_class, type):
        raise TypeError(f"Expected a class, got {type(model_class)}")

    if not issubclass(model_class, BaseModel):
        raise TypeError(
            f"Schema class must inherit from pydantic.BaseModel, got {model_class}"
        )

    # Ensure the model has at least one field
    schema = get_json_schema(model_class)
    if not schema.get("properties"):
        raise ValueError(
            f"Schema class {model_class.__name__} must have at least one field"
        )


def is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic BaseModel class.

    Args:
        obj: Object to check

    Returns:
        True if obj is a Pydantic BaseModel class
    """
    return isinstance(obj, type) and issubclass(obj, BaseModel)


REQUIRED_SCHEMA_CONFIG = ConfigDict(
    extra="forbid", validate_assignment=True, str_strip_whitespace=True
)


class _LangStructBaseModel(BaseModel):
    """Internal base model providing LangStruct's strict defaults."""

    model_config = REQUIRED_SCHEMA_CONFIG


_WRAPPED_SCHEMA_CACHE: "WeakKeyDictionary[type[BaseModel], type[BaseModel]]" = (
    WeakKeyDictionary()
)


def ensure_schema_class(model_class: Type[BaseModel]) -> Type[BaseModel]:
    """Wrap the schema class to enforce LangStruct's strict validation defaults."""

    if getattr(model_class, "__langstruct_wrapped__", False):
        return model_class

    cached = _WRAPPED_SCHEMA_CACHE.get(model_class)
    if cached is not None:
        return cached

    # Only wrap when configuration differs from required defaults
    current_config = dict(model_class.model_config)
    needs_wrap = any(
        current_config.get(key) != value
        for key, value in REQUIRED_SCHEMA_CONFIG.items()
    )

    if not needs_wrap:
        return model_class

    merged_config = {**current_config, **REQUIRED_SCHEMA_CONFIG}

    namespace: Dict[str, Any] = {
        "model_config": ConfigDict(**merged_config),
        "__langstruct_wrapped__": True,
        "__module__": model_class.__module__,
    }

    Wrapped = type(model_class.__name__, (model_class,), namespace)

    Wrapped.__qualname__ = model_class.__qualname__
    Wrapped.__doc__ = model_class.__doc__

    # Ensure Pydantic rebuilds caches with new configuration
    Wrapped.model_rebuild(force=True)
    _WRAPPED_SCHEMA_CACHE[model_class] = Wrapped
    return Wrapped
