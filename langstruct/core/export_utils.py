"""Export utilities for extraction results in multiple formats."""

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..exceptions import ExtractionError
from .schemas import ExtractionResult, SourceSpan


def _validate_file_path(file_path: Union[str, Path]) -> Path:
    """Validate and sanitize file paths to prevent path traversal attacks.

    Args:
        file_path: Path to validate

    Returns:
        Validated Path object

    Raises:
        ValueError: If path contains path traversal patterns
    """
    # Convert to Path object and resolve
    path = Path(file_path).resolve()

    # Check for suspicious patterns in the original path string before resolution
    path_str = str(file_path)
    if ".." in path_str:
        raise ExtractionError(
            f"Invalid file path: path traversal pattern '..' detected in '{file_path}'"
        )

    # Allow absolute paths but ensure they don't contain path traversal patterns
    # This allows temporary files and other legitimate absolute paths
    return path


class ExportUtilities:
    """Utilities for exporting extraction results to various formats."""

    @staticmethod
    def to_dict(
        result: ExtractionResult,
        include_metadata: bool = True,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """Convert ExtractionResult to a flat dictionary.

        Args:
            result: ExtractionResult to convert
            include_metadata: Whether to include metadata
            include_sources: Whether to include source information

        Returns:
            Dictionary representation
        """
        output = dict(result.entities)
        output["_confidence"] = result.confidence

        if include_metadata and result.metadata:
            for key, value in result.metadata.items():
                output[f"_meta_{key}"] = value

        if include_sources and result.sources:
            # Add source count for each field
            for field_name, spans in result.sources.items():
                output[f"_sources_{field_name}_count"] = len(spans)
                if spans:
                    # Add first source location
                    output[f"_sources_{field_name}_start"] = spans[0].start
                    output[f"_sources_{field_name}_end"] = spans[0].end

        return output

    @staticmethod
    def to_json(
        result: ExtractionResult,
        include_metadata: bool = True,
        include_sources: bool = True,
        indent: Optional[int] = 2,
    ) -> str:
        """Convert ExtractionResult to JSON string.

        Args:
            result: ExtractionResult to convert
            include_metadata: Whether to include metadata
            include_sources: Whether to include source information
            indent: JSON indentation (None for compact)

        Returns:
            JSON string
        """
        data = {"entities": result.entities, "confidence": result.confidence}

        if include_metadata:
            data["metadata"] = result.metadata

        if include_sources:
            # Convert SourceSpan objects to dictionaries
            sources_dict = {}
            for field_name, spans in result.sources.items():
                sources_dict[field_name] = [
                    {"start": span.start, "end": span.end, "text": span.text}
                    for span in spans
                ]
            data["sources"] = sources_dict

        return json.dumps(data, indent=indent, default=str)

    @staticmethod
    def to_csv_row(
        result: ExtractionResult,
        include_metadata: bool = False,
        include_sources: bool = False,
    ) -> Dict[str, Any]:
        """Convert ExtractionResult to CSV row dictionary.

        Args:
            result: ExtractionResult to convert
            include_metadata: Whether to include metadata columns
            include_sources: Whether to include source columns

        Returns:
            Dictionary suitable for CSV writing
        """
        return ExportUtilities.to_dict(result, include_metadata, include_sources)

    @staticmethod
    def results_to_dataframe(
        results: List[ExtractionResult],
        include_metadata: bool = False,
        include_sources: bool = False,
    ) -> "pd.DataFrame":
        """Convert list of ExtractionResults to pandas DataFrame.

        Args:
            results: List of ExtractionResult objects
            include_metadata: Whether to include metadata columns
            include_sources: Whether to include source columns

        Returns:
            pandas DataFrame

        Raises:
            ImportError: If pandas is not installed
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for DataFrame export. Install with: pip install pandas"
            )

        rows = [
            ExportUtilities.to_dict(result, include_metadata, include_sources)
            for result in results
        ]

        return pd.DataFrame(rows)

    @staticmethod
    def save_json(
        result: ExtractionResult,
        file_path: Union[str, Path],
        include_metadata: bool = True,
        include_sources: bool = True,
        indent: Optional[int] = 2,
    ) -> None:
        """Save ExtractionResult to JSON file.

        Args:
            result: ExtractionResult to save
            file_path: Path to save file
            include_metadata: Whether to include metadata
            include_sources: Whether to include source information
            indent: JSON indentation
        """
        validated_path = _validate_file_path(file_path)
        json_str = ExportUtilities.to_json(
            result, include_metadata, include_sources, indent
        )

        with open(validated_path, "w", encoding="utf-8") as f:
            f.write(json_str)

    @staticmethod
    def save_csv(
        results: List[ExtractionResult],
        file_path: Union[str, Path],
        include_metadata: bool = False,
        include_sources: bool = False,
    ) -> None:
        """Save list of ExtractionResults to CSV file.

        Args:
            results: List of ExtractionResult objects
            file_path: Path to save file
            include_metadata: Whether to include metadata columns
            include_sources: Whether to include source columns
        """
        validated_path = _validate_file_path(file_path)

        if not results:
            # Create empty CSV file
            with open(validated_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["entities", "confidence"])
            return

        # Convert all results to dictionaries
        rows = [
            ExportUtilities.to_dict(result, include_metadata, include_sources)
            for result in results
        ]

        # Get all unique keys for header
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())

        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)

        with open(validated_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def save_dataframe(
        results: List[ExtractionResult],
        file_path: Union[str, Path],
        format: str = "csv",
        include_metadata: bool = False,
        include_sources: bool = False,
        **kwargs,
    ) -> None:
        """Save results as DataFrame in various formats.

        Args:
            results: List of ExtractionResult objects
            file_path: Path to save file
            format: Output format ('csv', 'excel', 'json', 'parquet')
            include_metadata: Whether to include metadata columns
            include_sources: Whether to include source columns
            **kwargs: Additional arguments for pandas export methods
        """
        validated_path = _validate_file_path(file_path)
        df = ExportUtilities.results_to_dataframe(
            results, include_metadata, include_sources
        )

        format = format.lower()
        if format == "csv":
            df.to_csv(validated_path, index=False, **kwargs)
        elif format == "excel":
            df.to_excel(validated_path, index=False, **kwargs)
        elif format == "json":
            df.to_json(validated_path, orient="records", **kwargs)
        elif format == "parquet":
            df.to_parquet(validated_path, **kwargs)
        else:
            raise ValueError(
                f"Unsupported format: {format}. Use: csv, excel, json, parquet"
            )

    @staticmethod
    def save_annotated_documents(
        results: List[ExtractionResult],
        file_path: Union[str, Path],
        include_metadata: bool = True,
        include_sources: bool = True,
    ) -> None:
        """Save extraction results to JSONL format for LLM data workflows.

        This format is compatible with popular LLM training and analysis tools.
        Each line contains a complete JSON object representing one extraction result.

        Args:
            results: List of ExtractionResult objects to save
            file_path: Path to save JSONL file
            include_metadata: Whether to include metadata in output
            include_sources: Whether to include source grounding information

        Example:
            >>> results = extractor.extract(texts)
            >>> ExportUtilities.save_annotated_documents(results, "extractions.jsonl")
        """
        validated_path = _validate_file_path(file_path)

        with open(validated_path, "w", encoding="utf-8") as f:
            for result in results:
                json_line = ExportUtilities.to_json(
                    result,
                    include_metadata=include_metadata,
                    include_sources=include_sources,
                    indent=None,  # Compact format for JSONL
                )
                f.write(json_line + "\n")

    @staticmethod
    def load_annotated_documents(file_path: Union[str, Path]) -> List[ExtractionResult]:
        """Load extraction results from JSONL format.

        Args:
            file_path: Path to JSONL file to load

        Returns:
            List of ExtractionResult objects

        Example:
            >>> results = ExportUtilities.load_annotated_documents("extractions.jsonl")
            >>> # Loaded N extraction results
        """
        validated_path = _validate_file_path(file_path)

        if not validated_path.exists():
            raise FileNotFoundError(f"File not found: {validated_path}")

        results = []
        with open(validated_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    data = json.loads(line)

                    # Convert sources back to SourceSpan objects
                    sources = {}
                    if "sources" in data:
                        for field_name, spans_data in data["sources"].items():
                            sources[field_name] = [
                                SourceSpan(
                                    start=span["start"],
                                    end=span["end"],
                                    text=span["text"],
                                )
                                for span in spans_data
                            ]

                    result = ExtractionResult(
                        entities=data.get("entities", {}),
                        sources=sources,
                        confidence=data.get("confidence", 1.0),
                        metadata=data.get("metadata", {}),
                    )
                    results.append(result)

                except json.JSONDecodeError as e:
                    raise ExtractionError(f"Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    raise ExtractionError(f"Error processing line {line_num}: {e}")

        return results


# Add convenience methods to ExtractionResult class
def _add_export_methods_to_extraction_result():
    """Add export methods to ExtractionResult class."""

    def to_dict(
        self, include_metadata: bool = True, include_sources: bool = True
    ) -> Dict[str, Any]:
        """Export as dictionary."""
        return ExportUtilities.to_dict(self, include_metadata, include_sources)

    def to_json(
        self,
        include_metadata: bool = True,
        include_sources: bool = True,
        indent: Optional[int] = 2,
    ) -> str:
        """Export as JSON string."""
        return ExportUtilities.to_json(self, include_metadata, include_sources, indent)

    def save_json(
        self,
        file_path: Union[str, Path],
        include_metadata: bool = True,
        include_sources: bool = True,
    ) -> None:
        """Save to JSON file."""
        ExportUtilities.save_json(self, file_path, include_metadata, include_sources)

    def to_dataframe(
        self, include_metadata: bool = False, include_sources: bool = False
    ) -> "pd.DataFrame":
        """Convert single result to DataFrame."""
        return ExportUtilities.results_to_dataframe(
            [self], include_metadata, include_sources
        )

    # Add methods to ExtractionResult class
    ExtractionResult.to_dict = to_dict
    ExtractionResult.to_json = to_json
    ExtractionResult.save_json = save_json
    ExtractionResult.to_dataframe = to_dataframe


# Automatically add methods when module is imported
_add_export_methods_to_extraction_result()
