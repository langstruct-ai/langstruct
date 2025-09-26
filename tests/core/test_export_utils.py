"""Tests for export utilities."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from langstruct.core.export_utils import ExportUtilities
from langstruct.core.schemas import ExtractionResult, SourceSpan


class TestExportUtilities:
    """Tests for ExportUtilities class."""

    def test_to_dict_basic(self, sample_extraction_result):
        """Test basic dictionary conversion."""
        result_dict = ExportUtilities.to_dict(sample_extraction_result)

        # Should include all entities
        assert result_dict["name"] == "Dr. Sarah Johnson"
        assert result_dict["age"] == 34
        assert result_dict["location"] == "Cambridge, Massachusetts"

        # Should include confidence
        assert result_dict["_confidence"] == 0.92

    def test_to_dict_with_metadata(self, sample_extraction_result):
        """Test dictionary conversion with metadata."""
        result_dict = ExportUtilities.to_dict(
            sample_extraction_result, include_metadata=True
        )

        # Should include metadata fields with prefix
        assert "_meta_pipeline" in result_dict
        assert "_meta_total_chunks" in result_dict
        assert "_meta_original_text_length" in result_dict

        assert result_dict["_meta_pipeline"] == "langstruct"
        assert result_dict["_meta_total_chunks"] == 1

    def test_to_dict_with_sources(self, sample_extraction_result):
        """Test dictionary conversion with source information."""
        result_dict = ExportUtilities.to_dict(
            sample_extraction_result, include_sources=True
        )

        # Should include source count and location info
        assert "_sources_name_count" in result_dict
        assert "_sources_name_start" in result_dict
        assert "_sources_name_end" in result_dict

        assert result_dict["_sources_name_count"] == 1
        assert result_dict["_sources_name_start"] == 5
        assert result_dict["_sources_name_end"] == 19

    def test_to_dict_exclude_metadata(self, sample_extraction_result):
        """Test dictionary conversion without metadata."""
        result_dict = ExportUtilities.to_dict(
            sample_extraction_result, include_metadata=False
        )

        # Should not have metadata fields
        metadata_fields = [k for k in result_dict.keys() if k.startswith("_meta_")]
        assert len(metadata_fields) == 0

    def test_to_dict_exclude_sources(self, sample_extraction_result):
        """Test dictionary conversion without sources."""
        result_dict = ExportUtilities.to_dict(
            sample_extraction_result, include_sources=False
        )

        # Should not have source fields
        source_fields = [k for k in result_dict.keys() if k.startswith("_sources_")]
        assert len(source_fields) == 0

    def test_to_json_basic(self, sample_extraction_result):
        """Test basic JSON conversion."""
        json_str = ExportUtilities.to_json(sample_extraction_result)

        # Should be valid JSON
        data = json.loads(json_str)

        assert "entities" in data
        assert "confidence" in data
        assert data["entities"]["name"] == "Dr. Sarah Johnson"
        assert data["confidence"] == 0.92

    def test_to_json_with_sources(self, sample_extraction_result):
        """Test JSON conversion with sources."""
        json_str = ExportUtilities.to_json(
            sample_extraction_result, include_sources=True
        )

        data = json.loads(json_str)

        assert "sources" in data
        assert "name" in data["sources"]
        assert len(data["sources"]["name"]) == 1
        assert data["sources"]["name"][0]["start"] == 5
        assert data["sources"]["name"][0]["end"] == 19
        assert data["sources"]["name"][0]["text"] == "Sarah Johnson"

    def test_to_json_compact(self, sample_extraction_result):
        """Test compact JSON conversion."""
        compact_json = ExportUtilities.to_json(sample_extraction_result, indent=None)

        indented_json = ExportUtilities.to_json(sample_extraction_result, indent=2)

        # Compact should be shorter
        assert len(compact_json) < len(indented_json)

        # Both should parse to same data
        assert json.loads(compact_json) == json.loads(indented_json)

    def test_to_csv_row(self, sample_extraction_result):
        """Test CSV row conversion."""
        row_dict = ExportUtilities.to_csv_row(sample_extraction_result)

        # Should be a flat dictionary suitable for CSV
        assert isinstance(row_dict, dict)
        assert "name" in row_dict
        assert "age" in row_dict
        assert "_confidence" in row_dict

        # All values should be serializable
        for value in row_dict.values():
            assert isinstance(value, (str, int, float, bool, type(None)))

    def test_save_json(self, sample_extraction_result):
        """Test saving JSON to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            ExportUtilities.save_json(sample_extraction_result, temp_path)

            # File should exist and contain valid JSON
            assert os.path.exists(temp_path)

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "entities" in data
            assert data["entities"]["name"] == "Dr. Sarah Johnson"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_csv_single_result(self, sample_extraction_result):
        """Test saving single result to CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            ExportUtilities.save_csv([sample_extraction_result], temp_path)

            # File should exist
            assert os.path.exists(temp_path)

            # Should be readable as CSV
            import csv

            with open(temp_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert "name" in rows[0]
            assert rows[0]["name"] == "Dr. Sarah Johnson"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_csv_multiple_results(self, sample_extraction_result):
        """Test saving multiple results to CSV."""
        # Create second result
        result2 = ExtractionResult(
            entities={"name": "John Doe", "age": 25, "location": "New York"},
            sources={},
            confidence=0.8,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            ExportUtilities.save_csv([sample_extraction_result, result2], temp_path)

            import csv

            with open(temp_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["name"] == "Dr. Sarah Johnson"
            assert rows[1]["name"] == "John Doe"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_csv_empty_list(self):
        """Test saving empty results list to CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            ExportUtilities.save_csv([], temp_path)

            # Should create empty CSV with header
            assert os.path.exists(temp_path)

            import csv

            with open(temp_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 0  # No data rows
            assert reader.fieldnames == ["entities", "confidence"]  # Default header

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.skipif(
        not ExportUtilities.__dict__.get("PANDAS_AVAILABLE", True),
        reason="pandas not available",
    )
    def test_results_to_dataframe(self, sample_extraction_result):
        """Test converting results to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        df = ExportUtilities.results_to_dataframe([sample_extraction_result])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "name" in df.columns
        assert "age" in df.columns
        assert "_confidence" in df.columns

        assert df.iloc[0]["name"] == "Dr. Sarah Johnson"
        assert df.iloc[0]["age"] == 34

    @pytest.mark.skipif(
        not ExportUtilities.__dict__.get("PANDAS_AVAILABLE", True),
        reason="pandas not available",
    )
    def test_save_dataframe_formats(self, sample_extraction_result):
        """Test saving DataFrame in different formats."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        results = [sample_extraction_result]

        # Test CSV format
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            ExportUtilities.save_dataframe(results, csv_path, format="csv")
            assert os.path.exists(csv_path)

            # Should be readable as CSV
            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert "name" in df.columns

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

        # Test JSON format
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            ExportUtilities.save_dataframe(results, json_path, format="json")
            assert os.path.exists(json_path)

            # Should be readable as JSON
            with open(json_path, "r") as f:
                data = json.load(f)
            assert len(data) == 1
            assert "name" in data[0]

        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_save_dataframe_unsupported_format(self, sample_extraction_result):
        """Test error handling for unsupported format."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="Unsupported format"):
                ExportUtilities.save_dataframe(
                    [sample_extraction_result], f.name, format="unsupported"
                )


class TestExtractionResultMethods:
    """Tests for methods added to ExtractionResult class."""

    def test_to_dict_method(self, sample_extraction_result):
        """Test to_dict method on ExtractionResult."""
        result_dict = sample_extraction_result.to_dict()

        assert isinstance(result_dict, dict)
        assert "name" in result_dict
        assert "_confidence" in result_dict

    def test_to_json_method(self, sample_extraction_result):
        """Test to_json method on ExtractionResult."""
        json_str = sample_extraction_result.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "entities" in data
        assert "confidence" in data

    def test_save_json_method(self, sample_extraction_result):
        """Test save_json method on ExtractionResult."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            sample_extraction_result.save_json(temp_path)

            assert os.path.exists(temp_path)
            with open(temp_path, "r") as f:
                data = json.load(f)
            assert "entities" in data

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.skipif(
        not ExportUtilities.__dict__.get("PANDAS_AVAILABLE", True),
        reason="pandas not available",
    )
    def test_to_dataframe_method(self, sample_extraction_result):
        """Test to_dataframe method on ExtractionResult."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        df = sample_extraction_result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "name" in df.columns

    def test_method_parameters(self, sample_extraction_result):
        """Test that method parameters work correctly."""
        # Test with different parameters
        dict_with_meta = sample_extraction_result.to_dict(include_metadata=True)
        dict_without_meta = sample_extraction_result.to_dict(include_metadata=False)

        meta_keys = [k for k in dict_with_meta.keys() if k.startswith("_meta_")]
        no_meta_keys = [k for k in dict_without_meta.keys() if k.startswith("_meta_")]

        assert len(meta_keys) > 0
        assert len(no_meta_keys) == 0

    def test_methods_exist(self, sample_extraction_result):
        """Test that all expected methods are added to ExtractionResult."""
        # Check that methods exist
        assert hasattr(sample_extraction_result, "to_dict")
        assert hasattr(sample_extraction_result, "to_json")
        assert hasattr(sample_extraction_result, "save_json")
        assert hasattr(sample_extraction_result, "to_dataframe")

        # Check that they are callable
        assert callable(sample_extraction_result.to_dict)
        assert callable(sample_extraction_result.to_json)
        assert callable(sample_extraction_result.save_json)
        assert callable(sample_extraction_result.to_dataframe)
