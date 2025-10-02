"""Tests for LangStruct save/load persistence functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from langstruct import LangStruct
from langstruct.core.chunking import ChunkingConfig
from langstruct.core.schema_utils import get_field_descriptions
from langstruct.exceptions import PersistenceError
from langstruct.providers.llm_factory import LLMFactory


@pytest.fixture(autouse=True)
def persistence_default_model(monkeypatch):
    """Force LangStruct.load to use the mock model when defaulting."""

    monkeypatch.setattr(LLMFactory, "get_default_model", lambda: "mock-model")


class PersonSchema(BaseModel):
    """Test schema for persistence tests."""

    name: str = Field(description="Person's full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job or profession")


class TestBasicPersistence:
    """Test basic save and load functionality."""

    def test_save_and_load_dynamic_schema(self):
        """Test saving and loading with dynamically generated schema."""
        # Create extractor with dynamic schema
        extractor = LangStruct(example={"name": "Alice", "age": 30, "city": "Boston"})

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"

            # Save extractor
            extractor.save(str(save_path))

            # Verify save directory structure
            assert save_path.exists()
            assert (save_path / "langstruct_metadata.json").exists()
            assert (save_path / "pipeline.json").exists()

            # Load extractor
            loaded_extractor = LangStruct.load(str(save_path))

            # Test that loaded extractor works
            text = "Hi, I'm Bob and I'm 25 years old from New York."
            result = loaded_extractor.extract(text)

            assert "name" in result.entities
            assert "age" in result.entities
            assert "city" in result.entities

    def test_save_and_load_predefined_schema(self):
        """Test saving and loading with predefined schema."""
        # Create extractor with predefined schema
        extractor = LangStruct(schema=PersonSchema)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"

            # Save extractor
            extractor.save(str(save_path))

            # Load extractor
            loaded_extractor = LangStruct.load(str(save_path))

            # Verify schema fields are preserved
            original_fields = get_field_descriptions(extractor.schema)
            loaded_fields = get_field_descriptions(loaded_extractor.schema)
            assert original_fields == loaded_fields

    def test_save_creates_directory_structure(self):
        """Test that save creates the correct directory structure."""
        extractor = LangStruct(example={"name": "Alice", "age": 30})

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "nested" / "test_extractor"

            # Save to non-existent nested directory
            extractor.save(str(save_path))

            # Verify directory was created
            assert save_path.exists()
            assert save_path.is_dir()

    def test_metadata_content(self):
        """Test that metadata contains expected information."""
        extractor = LangStruct(example={"name": "Alice", "age": 30})

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))

            # Read and verify metadata
            metadata_path = save_path / "langstruct_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            required_fields = [
                "langstruct_version",
                "dspy_version",
                "schema_type",
                "schema_name",
                "schema_json_data",
                "schema_fields",
                "model_name",
                "lm_config",
                "chunking_config",
                "use_sources",
                "created_timestamp",
            ]

            for field in required_fields:
                assert field in metadata

            assert metadata["schema_type"] == "dynamic"
            assert metadata["schema_name"] == "GeneratedSchema"

    def test_chunking_config_round_trip(self):
        """Test that custom chunking configuration is preserved across save/load."""

        custom_config = ChunkingConfig(
            max_tokens=512,
            overlap_tokens=64,
            preserve_paragraphs=False,
            preserve_sentences=True,
        )

        extractor = LangStruct(schema=PersonSchema, chunking_config=custom_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))

            loaded_extractor = LangStruct.load(str(save_path))

            assert isinstance(loaded_extractor.chunking_config, ChunkingConfig)
            assert (
                loaded_extractor.chunking_config.model_dump()
                == custom_config.model_dump()
            )


class TestErrorHandling:
    """Test error handling in save/load operations."""

    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        with pytest.raises(PersistenceError) as exc_info:
            LangStruct.load("/nonexistent/path")

        assert "does not exist" in str(exc_info.value)

    def test_load_missing_metadata(self):
        """Test loading with missing metadata file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "incomplete_extractor"
            save_path.mkdir()

            # Create pipeline file but not metadata
            (save_path / "pipeline.json").write_text("{}")

            with pytest.raises(PersistenceError) as exc_info:
                LangStruct.load(str(save_path))

            assert "Missing required files" in str(exc_info.value)

    def test_load_missing_pipeline(self):
        """Test loading with missing pipeline file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "incomplete_extractor"
            save_path.mkdir()

            # Create metadata but not pipeline
            metadata = {
                "langstruct_version": "0.1.0",
                "dspy_version": "3.0.0",
                "schema_type": "dynamic",
                "schema_name": "TestSchema",
                "schema_json": {"type": "object", "properties": {}},
                "schema_fields": {},
                "model_name": "test-model",
                "lm_config": {},
                "chunking_config": {},
                "use_sources": True,
                "optimization_applied": False,
                "refinement_applied": False,
                "created_timestamp": "2024-01-01T00:00:00",
            }

            (save_path / "langstruct_metadata.json").write_text(json.dumps(metadata))

            with pytest.raises(PersistenceError) as exc_info:
                LangStruct.load(str(save_path))

            assert "Missing required files" in str(exc_info.value)

    def test_load_corrupted_metadata(self):
        """Test loading with corrupted metadata file."""
        extractor = LangStruct(example={"name": "Alice", "age": 30})

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))

            # Corrupt metadata file
            (save_path / "langstruct_metadata.json").write_text("invalid json {")

            with pytest.raises(PersistenceError):
                LangStruct.load(str(save_path))

    def test_load_corrupted_pipeline(self):
        """Test loading with corrupted pipeline file."""
        extractor = LangStruct(example={"name": "Alice", "age": 30})

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))

            # Corrupt pipeline file
            (save_path / "pipeline.json").write_text("invalid json {")

            with pytest.raises(PersistenceError) as exc_info:
                LangStruct.load(str(save_path))

            assert "invalid JSON" in str(exc_info.value)

    def test_version_compatibility_major_diff(self):
        """Test loading with major version difference."""
        extractor = LangStruct(example={"name": "Alice", "age": 30})

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))

            # Modify metadata to simulate major version difference
            metadata_path = save_path / "langstruct_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            metadata["langstruct_version"] = "99.0.0"

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            with pytest.raises(PersistenceError) as exc_info:
                LangStruct.load(str(save_path))

            assert "Major version mismatch" in str(exc_info.value)

    @patch.dict(os.environ, {}, clear=True)
    def test_api_key_validation(self):
        """Test API key validation during load."""
        # This test temporarily clears environment variables
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"

            # Create minimal metadata for OpenAI model
            metadata = {
                "langstruct_version": "0.1.0",
                "dspy_version": "3.0.0",
                "schema_type": "dynamic",
                "schema_name": "TestSchema",
                "schema_json_data": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
                "schema_fields": {"name": "Name field"},
                "model_name": "gpt-4",  # Requires OPENAI_API_KEY
                "lm_config": {},
                "chunking_config": {},
                "use_sources": True,
                "optimization_applied": False,
                "refinement_applied": False,
                "created_timestamp": "2024-01-01T00:00:00",
            }

            save_path.mkdir()
            (save_path / "langstruct_metadata.json").write_text(json.dumps(metadata))
            (save_path / "pipeline.json").write_text("{}")

            with pytest.raises(PersistenceError) as exc_info:
                LangStruct.load(str(save_path))

            assert "OPENAI_API_KEY" in str(exc_info.value)


class TestSchemaReconstruction:
    """Test schema reconstruction functionality."""

    def test_dynamic_schema_reconstruction(self):
        """Test reconstruction of dynamically generated schemas."""
        example = {"name": "Alice", "age": 30, "skills": ["Python", "ML"]}
        extractor = LangStruct(example=example)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))
            loaded_extractor = LangStruct.load(str(save_path))

            # Check that schema fields match
            original_fields = get_field_descriptions(extractor.schema)
            loaded_fields = get_field_descriptions(loaded_extractor.schema)

            assert set(original_fields.keys()) == set(loaded_fields.keys())

    def test_predefined_schema_fallback(self):
        """Test fallback to dynamic reconstruction when predefined schema can't be imported."""
        extractor = LangStruct(schema=PersonSchema)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))

            # Modify metadata to reference a non-existent module
            metadata_path = save_path / "langstruct_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            metadata["schema_module"] = "nonexistent.module"

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Should still load via dynamic reconstruction
            loaded_extractor = LangStruct.load(str(save_path))

            # Fields should be preserved
            original_fields = set(get_field_descriptions(extractor.schema).keys())
            loaded_fields = set(get_field_descriptions(loaded_extractor.schema).keys())
            assert original_fields == loaded_fields


class TestAdvancedFeatures:
    """Test persistence of advanced features like optimization and refinement."""

    def test_save_with_refinement_config(self):
        """Test saving extractor with refinement configuration."""
        from langstruct.core.refinement import Refine

        extractor = LangStruct(
            example={"name": "Alice", "age": 30},
            refine=Refine(strategy="bon", n_candidates=3),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))

            # Check refinement config is saved
            assert (save_path / "refinement_config.json").exists()

            # Load and verify refinement config is restored
            loaded_extractor = LangStruct.load(str(save_path))
            assert loaded_extractor.refine_config is not None
            assert loaded_extractor.refine_config.strategy == "bon"
            assert loaded_extractor.refine_config.n_candidates == 3

    def test_metadata_with_optimization_flag(self):
        """Test that optimization flag is correctly saved in metadata."""
        extractor = LangStruct(example={"name": "Alice", "age": 30})
        extractor.optimizer = object()  # simulate optimization having run

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_extractor"
            extractor.save(str(save_path))

            # Check metadata includes optimization info
            metadata_path = save_path / "langstruct_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            assert metadata["optimization_applied"] == True
            assert metadata["optimizer_name"] == "miprov2"  # Default optimizer


if __name__ == "__main__":
    pytest.main([__file__])
