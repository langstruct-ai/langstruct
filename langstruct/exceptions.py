"""Custom exception classes for LangStruct.

Simplified exception hierarchy with clear, actionable error messages.
"""


class LangStructError(Exception):
    """Base exception for all LangStruct errors.

    All LangStruct exceptions inherit from this class, making it easy to
    catch any LangStruct-related error.
    """

    pass


class ConfigurationError(LangStructError):
    """Raised when there are setup or configuration issues.

    This includes problems with:
    - Schema definition or generation
    - Model configuration or initialization
    - File operations (save/load)
    - Invalid parameter combinations

    Examples:
        - Missing required parameters
        - Invalid model names
        - Schema generation failures
        - File permission issues
    """

    pass


class ExtractionError(LangStructError):
    """Raised when the extraction process fails.

    This includes problems with:
    - Text extraction or processing
    - Chunking large documents
    - Optimization failures
    - Export operations

    Examples:
        - LLM API failures
        - Chunking configuration errors
        - Optimization convergence issues
        - Export format problems
    """

    pass


class ValidationError(LangStructError):
    """Raised when data validation fails.

    This includes problems with:
    - Schema validation errors
    - Invalid extraction results
    - Type mismatches
    - Required field violations

    Examples:
        - Missing required fields in extraction
        - Type conversion failures
        - Schema constraint violations
        - Invalid input data format
    """

    pass


class PersistenceError(LangStructError):
    """Raised when saving or loading extractors fails.

    This includes problems with:
    - File I/O operations during save/load
    - Missing or corrupted save files
    - Version compatibility issues
    - Schema reconstruction failures

    Examples:
        - Permission denied when writing save files
        - Required save files missing during load
        - Incompatible LangStruct versions
        - Failed schema reconstruction from saved data
    """

    pass
