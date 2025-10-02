"""Persistence utilities for saving and loading LangStruct extractors."""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import dspy
from pydantic import BaseModel

from ..exceptions import PersistenceError
from .chunking import ChunkingConfig
from .refinement import Budget, Refine, RefinementStrategy
from .schema_utils import get_field_descriptions, get_json_schema, validate_schema_class


class ExtractorMetadata(BaseModel):
    """Metadata for saved LangStruct extractors."""

    langstruct_version: str
    dspy_version: str
    schema_type: str  # "predefined" or "dynamic"
    schema_name: str
    schema_module: Optional[str] = None  # For predefined schemas
    schema_json_data: Dict[str, Any]  # JSON schema representation
    schema_fields: Dict[str, str]  # Field descriptions
    model_name: str
    lm_config: Dict[str, Any]  # LM config without API keys
    chunking_config: Dict[str, Any]
    use_sources: bool
    optimization_applied: bool = False
    optimizer_name: Optional[str] = None
    refinement_applied: bool = False
    created_timestamp: str


class ExtractorPersistence:
    """Handles saving and loading of LangStruct extractors."""

    CURRENT_VERSION = "0.1.0"
    REQUIRED_FILES = ["langstruct_metadata.json", "pipeline.json"]

    @classmethod
    def save_extractor(
        cls,
        extractor: "LangStruct",  # Type hint as string to avoid circular import
        path: Union[str, Path],
    ) -> None:
        """Save a LangStruct extractor to disk.

        Args:
            extractor: LangStruct instance to save
            path: Directory path to save to (will be created if doesn't exist)

        Raises:
            PersistenceError: If saving fails
        """
        save_path = Path(path)

        try:
            # Create directory if it doesn't exist
            save_path.mkdir(parents=True, exist_ok=True)

            # Save DSPy pipeline state using native functionality
            pipeline_path = save_path / "pipeline.json"
            extractor.pipeline.save(str(pipeline_path), save_program=False)

            # Prepare metadata
            metadata = cls._extract_metadata(extractor)

            # Save metadata
            metadata_path = save_path / "langstruct_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, indent=2, default=str)

            # Save optimizer state if present
            if extractor.optimizer is not None:
                optimizer_path = save_path / "optimizer_state.json"
                optimizer_state = cls._extract_optimizer_state(extractor.optimizer)
                with open(optimizer_path, "w", encoding="utf-8") as f:
                    json.dump(optimizer_state, f, indent=2)

            # Save refinement config if present
            if extractor.refine_config is not None:
                refinement_path = save_path / "refinement_config.json"
                refinement_state = cls._extract_refinement_state(
                    extractor.refine_config
                )
                with open(refinement_path, "w", encoding="utf-8") as f:
                    json.dump(refinement_state, f, indent=2)

        except Exception as e:
            raise PersistenceError(f"Failed to save extractor: {str(e)}") from e

    @classmethod
    def load_extractor(cls, path: Union[str, Path]) -> "LangStruct":
        """Load a LangStruct extractor from disk.

        Args:
            path: Directory path to load from

        Returns:
            Loaded LangStruct instance

        Raises:
            PersistenceError: If loading fails
        """
        load_path = Path(path)

        # Validate save directory
        cls._validate_save_directory(load_path)

        try:
            # Load metadata
            metadata_path = load_path / "langstruct_metadata.json"
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            metadata = ExtractorMetadata(**metadata_dict)

            # Check version compatibility
            cls._check_version_compatibility(metadata)

            # Validate API key availability for the model
            cls._validate_api_key_availability(metadata.model_name)

            # Validate DSPy pipeline file
            pipeline_path = load_path / "pipeline.json"
            cls._validate_pipeline_file(pipeline_path)

            # Reconstruct schema
            try:
                schema_class = cls._reconstruct_schema(metadata)
            except Exception as e:
                raise PersistenceError(f"Failed to reconstruct schema: {e}") from e

            # Recreate LangStruct instance
            from ..api import LangStruct  # Import here to avoid circular import

            # Create base extractor without optimization/refinement first
            try:
                chunking_config = cls._restore_chunking_config(metadata.chunking_config)

                extractor = LangStruct(
                    schema=schema_class,
                    model=metadata.model_name,
                    chunking_config=chunking_config,
                    use_sources=metadata.use_sources,
                    **metadata.lm_config,
                )  # Optimizer state restored separately
            except Exception as e:
                raise PersistenceError(
                    f"Failed to recreate LangStruct instance. This may be due to missing API keys, "
                    f"invalid model configuration, or incompatible settings. Error: {e}"
                ) from e

            # Load DSPy pipeline state
            pipeline_path = load_path / "pipeline.json"
            try:
                extractor.pipeline.load(str(pipeline_path))
            except Exception as e:
                raise PersistenceError(
                    f"Failed to load DSPy pipeline state from {pipeline_path}. "
                    f"The save file may be corrupted or incompatible with the current DSPy version. "
                    f"Error: {e}"
                ) from e

            # Restore optimizer if present
            optimizer_path = load_path / "optimizer_state.json"
            if optimizer_path.exists() and metadata.optimization_applied:
                cls._restore_optimizer_state(
                    extractor, optimizer_path, metadata.optimizer_name
                )

            # Restore refinement config if present
            refinement_path = load_path / "refinement_config.json"
            if refinement_path.exists() and metadata.refinement_applied:
                cls._restore_refinement_state(extractor, refinement_path)

            return extractor

        except Exception as e:
            raise PersistenceError(f"Failed to load extractor: {str(e)}") from e

    @classmethod
    def _extract_metadata(cls, extractor: "LangStruct") -> ExtractorMetadata:
        """Extract metadata from a LangStruct instance."""
        import datetime

        import langstruct

        # Determine schema type and get information
        schema_type, schema_module = cls._get_schema_info(extractor.schema)

        # Extract model configuration (excluding API keys)
        model_config = cls._get_safe_model_config(extractor.lm)

        return ExtractorMetadata(
            langstruct_version=(
                langstruct.__version__
                if hasattr(langstruct, "__version__")
                else cls.CURRENT_VERSION
            ),
            dspy_version=(
                dspy.__version__ if hasattr(dspy, "__version__") else "unknown"
            ),
            schema_type=schema_type,
            schema_name=extractor.schema.__name__,
            schema_module=schema_module,
            schema_json_data=get_json_schema(extractor.schema),
            schema_fields=get_field_descriptions(extractor.schema),
            model_name=cls._get_model_name(extractor.lm),
            lm_config=model_config,
            chunking_config=extractor.chunking_config.model_dump(),
            use_sources=extractor.use_sources,
            optimization_applied=extractor.optimizer is not None,
            optimizer_name=extractor.optimizer_name if extractor.optimizer else None,
            refinement_applied=extractor.refine_config is not None,
            created_timestamp=datetime.datetime.now().isoformat(),
        )

    @classmethod
    def _get_schema_info(
        cls, schema_class: Type[BaseModel]
    ) -> tuple[str, Optional[str]]:
        """Determine if schema is predefined or dynamic and get module info."""
        # Check if schema has a proper module (predefined) or was dynamically created
        module = getattr(schema_class, "__module__", None)

        if (
            module
            and module != "__main__"
            and not module.startswith("langstruct.core.schema_generator")
        ):
            return "predefined", module
        else:
            return "dynamic", None

    @classmethod
    def _get_safe_model_config(cls, lm: dspy.LM) -> Dict[str, Any]:
        """Extract safe model configuration (no API keys)."""
        config = {}

        # Extract common safe parameters
        if hasattr(lm, "kwargs") and hasattr(lm.kwargs, "items"):
            safe_keys = {
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "frequency_penalty",
                "presence_penalty",
                "base_url",
                "timeout",
                "max_retries",
            }
            try:
                for key, value in lm.kwargs.items():
                    if key in safe_keys:
                        config[key] = value
            except (TypeError, AttributeError):
                # Handle case where lm.kwargs is not iterable (e.g., in mocks)
                pass

        return config

    @classmethod
    def _get_model_name(cls, lm: dspy.LM) -> str:
        """Extract model name from DSPy LM instance."""
        # Check for special test model marker first (used by test doubles)
        if hasattr(lm, "_test_model_name"):
            return lm._test_model_name

        # Check for model attribute
        if hasattr(lm, "model") and isinstance(lm.model, str):
            return lm.model

        # Check kwargs for model name
        if hasattr(lm, "kwargs") and hasattr(lm.kwargs, "get"):
            try:
                model_name = lm.kwargs.get("model")
                if isinstance(model_name, str) and model_name != "unknown":
                    return model_name
            except (TypeError, AttributeError):
                pass

        # Default fallback
        return "gemini/gemini-2.5-flash"

    @classmethod
    def _extract_optimizer_state(cls, optimizer) -> Dict[str, Any]:
        """Extract optimizer state for serialization."""
        # Store basic optimizer configuration
        state = {
            "type": type(optimizer).__name__,
        }

        # Add type-specific state
        if hasattr(optimizer, "auto"):
            state["auto"] = optimizer.auto
        if hasattr(optimizer, "num_threads"):
            state["num_threads"] = optimizer.num_threads
        if hasattr(optimizer, "kwargs"):
            state["kwargs"] = optimizer.kwargs

        return state

    @classmethod
    def _extract_refinement_state(cls, refine_config: Refine) -> Dict[str, Any]:
        """Extract refinement configuration for serialization."""
        return refine_config.model_dump()

    @classmethod
    def _restore_chunking_config(cls, config_data: Dict[str, Any]) -> ChunkingConfig:
        """Rehydrate ChunkingConfig from persisted metadata."""
        if isinstance(config_data, ChunkingConfig):
            return config_data

        if not config_data:
            return ChunkingConfig()

        try:
            return ChunkingConfig(**config_data)
        except Exception as exc:
            raise PersistenceError(
                f"Invalid chunking configuration in metadata: {exc}"
            ) from exc

    @classmethod
    def _validate_save_directory(cls, path: Path) -> None:
        """Validate that save directory contains required files."""
        if not path.exists():
            raise PersistenceError(f"Save directory does not exist: {path}")

        if not path.is_dir():
            raise PersistenceError(f"Path is not a directory: {path}")

        missing_files = []
        for required_file in cls.REQUIRED_FILES:
            if not (path / required_file).exists():
                missing_files.append(required_file)

        if missing_files:
            raise PersistenceError(f"Missing required files: {missing_files}")

    @classmethod
    def _check_version_compatibility(cls, metadata: ExtractorMetadata) -> None:
        """Check if saved extractor is compatible with current version."""

        # Parse version numbers for better comparison
        def parse_version(version_str: str) -> tuple:
            try:
                parts = version_str.split(".")
                return tuple(
                    int(part) for part in parts[:3]
                )  # Take only major.minor.patch
            except (ValueError, AttributeError):
                return (0, 0, 0)  # Default for unknown versions

        saved_version = parse_version(metadata.langstruct_version)
        current_version = parse_version(cls.CURRENT_VERSION)

        # Check for major version differences (incompatible)
        if saved_version[0] != current_version[0]:
            raise PersistenceError(
                f"Major version mismatch: saved extractor is version {metadata.langstruct_version} "
                f"but current LangStruct is version {cls.CURRENT_VERSION}. "
                f"Major version differences are not supported."
            )

        # Warn about minor version differences
        if saved_version[1] != current_version[1]:
            warnings.warn(
                f"Minor version difference: saved extractor is version {metadata.langstruct_version} "
                f"but current LangStruct is version {cls.CURRENT_VERSION}. "
                f"Loading should work but some features may behave differently.",
                UserWarning,
            )

        # Just log patch version differences (usually fine)
        if saved_version[2] != current_version[2]:
            pass  # Patch differences are usually backwards compatible

    @classmethod
    def _reconstruct_schema(cls, metadata: ExtractorMetadata) -> Type[BaseModel]:
        """Reconstruct schema class from metadata."""
        if metadata.schema_type == "predefined" and metadata.schema_module:
            try:
                # Try to import the original schema class
                import importlib

                module = importlib.import_module(metadata.schema_module)
                schema_class = getattr(module, metadata.schema_name)

                # Validate it's actually a BaseModel subclass
                if not issubclass(schema_class, BaseModel):
                    raise ValueError(
                        f"Class {metadata.schema_name} is not a BaseModel subclass"
                    )

                return schema_class
            except (ImportError, AttributeError, ValueError) as e:
                warnings.warn(
                    f"Could not import original schema {metadata.schema_module}.{metadata.schema_name} ({e}). "
                    f"Falling back to dynamic reconstruction."
                )

        # Dynamic reconstruction from JSON schema
        try:
            from .schema_generator import SchemaGenerator

            # Validate we have required data for reconstruction
            if not metadata.schema_json_data:
                raise ValueError("Schema JSON is empty or missing")
            if not metadata.schema_name:
                raise ValueError("Schema name is missing")

            return SchemaGenerator._create_schema_from_json(
                metadata.schema_json_data,
                metadata.schema_name,
                metadata.schema_fields or {},
            )
        except Exception as e:
            raise PersistenceError(
                f"Failed to reconstruct schema '{metadata.schema_name}' from saved data. "
                f"The schema definition may be corrupted or incompatible. Error: {e}"
            ) from e

    @classmethod
    def _restore_optimizer_state(
        cls, extractor: "LangStruct", optimizer_path: Path, optimizer_name: str
    ) -> None:
        """Restore optimizer state."""
        with open(optimizer_path, "r", encoding="utf-8") as f:
            optimizer_state = json.load(f)

        # Recreate optimizer based on saved state
        if optimizer_name == "miprov2":
            from ..optimizers.mipro import MIPROv2Optimizer

            extractor.optimizer = MIPROv2Optimizer(
                auto=optimizer_state.get("auto", "light"),
                num_threads=optimizer_state.get("num_threads", 4),
                **optimizer_state.get("kwargs", {}),
            )

    @classmethod
    def _restore_refinement_state(
        cls, extractor: "LangStruct", refinement_path: Path
    ) -> None:
        """Restore refinement configuration."""
        with open(refinement_path, "r", encoding="utf-8") as f:
            refinement_data = json.load(f)

        extractor.refine_config = Refine(**refinement_data)

        # Recreate refinement engine
        from .refinement import RefinementEngine

        extractor.refinement_engine = RefinementEngine(
            extractor.schema, extractor.pipeline.extractor
        )

    @classmethod
    def _validate_api_key_availability(cls, model_name: str) -> None:
        """Validate that required API keys are available for the model."""
        from ..providers.llm_factory import LLMFactory

        # Skip validation for local models and test models
        if (
            model_name.startswith("test-")
            or model_name == "mock-model"
            or LLMFactory.is_local_model(model_name)
        ):
            return

        # Check for common API key requirements
        required_key = None
        model_lower = model_name.lower()

        if "gpt" in model_lower or "openai" in model_lower:
            required_key = "OPENAI_API_KEY"
        elif "claude" in model_lower or "anthropic" in model_lower:
            required_key = "ANTHROPIC_API_KEY"
        elif "gemini" in model_lower or "google" in model_lower:
            required_key = "GOOGLE_API_KEY"
        elif "groq" in model_lower:
            required_key = "GROQ_API_KEY"

        if required_key and not os.getenv(required_key):
            raise PersistenceError(
                f"Model '{model_name}' requires the {required_key} environment variable to be set. "
                f"Please set your API key before loading the extractor."
            )

    @classmethod
    def _validate_pipeline_file(cls, pipeline_path: Path) -> None:
        """Validate that the DSPy pipeline file is properly formatted."""
        try:
            with open(pipeline_path, "r", encoding="utf-8") as f:
                pipeline_data = json.load(f)

            # Basic validation that it looks like a DSPy save file
            if not isinstance(pipeline_data, dict):
                raise PersistenceError(
                    f"Pipeline file {pipeline_path} is not a valid JSON object"
                )

            # DSPy save files typically have certain structure
            # This is a basic check - DSPy's load will do more thorough validation
            if not pipeline_data:
                raise PersistenceError(f"Pipeline file {pipeline_path} is empty")

        except json.JSONDecodeError as e:
            raise PersistenceError(
                f"Pipeline file {pipeline_path} contains invalid JSON: {e}"
            )
        except FileNotFoundError:
            raise PersistenceError(f"Pipeline file {pipeline_path} not found")
        except Exception as e:
            raise PersistenceError(
                f"Error validating pipeline file {pipeline_path}: {e}"
            )
