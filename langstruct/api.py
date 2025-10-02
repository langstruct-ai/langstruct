"""Main LangStruct API for LLM-powered structured information extraction."""

import inspect
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, overload

import dspy

from .core import modules as core_modules
from .core.chunking import ChunkingConfig
from .core.export_utils import ExportUtilities
from .core.modules import QueryParser
from .core.refinement import Budget, Refine, RefinementEngine
from .core.schema_generator import SchemaGenerator
from .core.schema_utils import ensure_schema_class
from .core.schemas import ChunkResult, ExtractionResult, ParsedQuery, Schema
from .exceptions import (
    ConfigurationError,
    ExtractionError,
    LangStructError,
    PersistenceError,
    ValidationError,
)
from .optimizers.metrics import ExtractionMetrics
from .optimizers.mipro import MIPROv2Optimizer
from .parallel import ParallelProcessor, ProcessingResult
from .providers.llm_factory import LLMFactory
from .visualization.html_viz import save_visualization, visualize


class LangStruct:
    """Main interface for structured information extraction using DSPy optimization.

    LangStruct provides a simple API for extracting structured information from
    unstructured text using large language models, with automatic prompt optimization
    and source grounding capabilities.

    Example:
        ```python
        from pydantic import BaseModel, Field
        from langstruct import LangStruct

        class PersonSchema(BaseModel):
            name: str = Field(description="Full name of the person")
            age: int = Field(description="Age in years")
            location: str = Field(description="Current location")

        # Create extractor
        extractor = LangStruct(
            schema=PersonSchema,
            model="gpt-5-mini"
        )

        # Extract with source grounding
        result = extractor.extract(text, return_sources=True)
        # result.entities -> {'name': 'John Doe', 'age': 30, 'location': 'New York'}
        # result.sources  -> Source spans for each field
        ```
    """

    def __init__(
        self,
        schema: Optional[Type[Schema]] = None,
        model: Optional[Union[str, dspy.LM]] = None,
        optimizer: str = "miprov2",
        chunking_config: Optional[ChunkingConfig] = None,
        use_sources: bool = True,
        example: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        schema_name: str = "GeneratedSchema",
        descriptions: Optional[Dict[str, str]] = None,
        refine: Union[bool, Refine, Dict[str, Any]] = False,
        **llm_kwargs,
    ):
        """Initialize LangStruct extractor with smart defaults and auto-configuration.

        This is the main way to create a LangStruct extractor. You can either provide
        a pre-defined schema or let LangStruct auto-generate one from examples.

        Args:
            schema: Pydantic schema defining the extraction structure (optional)
            model: Model name or DSPy LM instance (defaults to "gpt-5-mini"; pass
                "gpt-5-mini"/"gpt-5-pro" for the latest OpenAI models)
            optimizer: Optimizer to use when optimize() runs (default: "miprov2")
            chunking_config: Configuration for text chunking
            use_sources: Whether to include source grounding (default: True)
            example: Single example dict for auto schema generation (optional)
            examples: Multiple example dicts for auto schema generation (optional)
            schema_name: Name for auto-generated schema class
            descriptions: Custom descriptions for auto-generated fields
            refine: Refinement configuration (bool, Refine object, or dict). False disables refinement.
            **llm_kwargs: Additional arguments for LM initialization

        Examples:
            >>> # With existing schema
            >>> extractor = LangStruct(schema=PersonSchema)

            >>> # Auto-generate schema from single example
            >>> extractor = LangStruct(example={"name": "John", "age": 25})

            >>> # Auto-generate schema from multiple examples (better type inference)
            >>> extractor = LangStruct(examples=[
            ...     {"name": "John", "age": 25, "location": "Boston"},
            ...     {"name": "Jane", "skills": ["Python", "ML"]}
            ... ])
        """
        # Handle schema generation from examples if needed
        if schema is None:
            if example is not None:
                schema = SchemaGenerator.from_example(
                    example, schema_name, descriptions
                )
            elif examples is not None:
                schema = SchemaGenerator.from_examples(
                    examples, schema_name, descriptions
                )
            else:
                raise ValueError("Must provide either schema, example, or examples")
        elif example is not None or examples is not None:
            raise ValueError(
                "Cannot provide both schema and example(s). Use one or the other."
            )

        schema = ensure_schema_class(schema)

        self.schema = schema
        self.optimizer_name = optimizer
        self.chunking_config = chunking_config or ChunkingConfig()
        self.use_sources = use_sources
        # Store original example for query parsing
        self.source_example = example or examples

        # Parse refinement configuration
        self.refine_config = self._parse_refine_config(refine)

        # Set up the language model
        if isinstance(model, dspy.LM):
            self.lm = model
        elif model is not None:
            # Model specified, create new LM (factory handles sensible defaults)
            self.lm = LLMFactory.create_lm(model, **llm_kwargs)
        else:
            # No model specified, try to use DSPy's configured LM or fallback to default
            try:
                # Check if DSPy has a configured LM
                current_lm = dspy.settings.lm
                if current_lm is not None:
                    self.lm = current_lm
                else:
                    # Auto-detect available model based on API keys
                    default_model = LLMFactory.get_default_model()
                    self.lm = LLMFactory.create_lm(default_model, **llm_kwargs)
            except (AttributeError, Exception):
                # Auto-detect available model based on API keys
                default_model = LLMFactory.get_default_model()
                self.lm = LLMFactory.create_lm(default_model, **llm_kwargs)

        # Configure DSPy
        dspy.configure(lm=self.lm)

        # Initialize the extraction pipeline (robust to monkeypatched constructors)
        pipeline_cls = core_modules.ExtractionPipeline
        try:
            # Inspect __init__ directly to get actual parameters (not dspy.Module's *args, **kwargs)
            sig = inspect.signature(pipeline_cls.__init__)
        except (TypeError, ValueError):
            # Fallback if signature can't be inspected (e.g., C-extensions or mocks)
            sig = None

        if sig is not None:
            supported_kwargs = {}
            params = sig.parameters
            # Prefer keyword args when supported
            if "schema" in params:
                supported_kwargs["schema"] = schema
            if "chunking_config" in params:
                supported_kwargs["chunking_config"] = chunking_config
            if "use_sources" in params:
                supported_kwargs["use_sources"] = use_sources

            if supported_kwargs:
                self.pipeline = pipeline_cls(**supported_kwargs)
            else:
                # Fall back to positional schema only
                self.pipeline = pipeline_cls(schema)
        else:
            # If signature unavailable, try most general form, then fallback
            try:
                self.pipeline = pipeline_cls(
                    schema=schema,
                    chunking_config=chunking_config,
                    use_sources=use_sources,
                )
            except TypeError:
                self.pipeline = pipeline_cls(schema)

        # Optimizer is created lazily when optimize() is called
        self.optimizer = None

        # Initialize refinement engine if requested
        self.refinement_engine = None
        if self.refine_config:
            self.refinement_engine = RefinementEngine(
                self.schema, self.pipeline.extractor
            )

    @overload
    def extract(
        self,
        text: str,
        confidence_threshold: float = 0.0,
        validate: bool = True,
        debug: bool = False,
        return_sources: Optional[bool] = None,
        refine: Union[bool, Refine, Dict[str, Any], None] = None,
        **kwargs,
    ) -> ExtractionResult: ...

    @overload
    def extract(
        self,
        text: List[str],
        confidence_threshold: float = 0.0,
        validate: bool = True,
        debug: bool = False,
        return_sources: Optional[bool] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = False,
        rate_limit: Optional[int] = None,
        retry_failed: bool = True,
        refine: Union[bool, Refine, Dict[str, Any], None] = None,
        **kwargs,
    ) -> List[ExtractionResult]: ...

    def extract(
        self,
        text: Union[str, List[str]],
        confidence_threshold: float = 0.0,
        validate: bool = True,
        debug: bool = False,
        return_sources: Optional[bool] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = False,
        rate_limit: Optional[int] = None,
        retry_failed: bool = True,
        refine: Union[bool, Refine, Dict[str, Any], None] = None,
        **kwargs,
    ) -> Union[ExtractionResult, List[ExtractionResult]]:
        """Extract structured information from text or list of texts.

        This method handles both single text and batch processing with automatic
        parallelization for lists.

        Args:
            text: Input text or list of texts to extract from
            confidence_threshold: Minimum confidence score to accept results
            validate: Whether to run quality validation and show suggestions
            debug: Whether to show detailed validation warnings and suggestions (default: False)
            return_sources: Override source grounding for this call
            max_workers: Maximum parallel workers for batch processing (list input only)
            show_progress: Show progress bar for batch processing (requires tqdm)
            rate_limit: API calls per minute limit for batch processing
            retry_failed: Whether to raise exception on failures in batch processing
            refine: Refinement configuration to boost accuracy by 15-30%. Can be:
                   - bool: True enables default refinement, False disables
                   - Refine: Custom refinement configuration object
                   - dict: Refinement config dict (e.g., {"strategy": "bon", "n_candidates": 5})
            **kwargs: Additional extraction parameters

        Returns:
            ExtractionResult for single text, or List[ExtractionResult] for multiple texts

        Examples:
            >>> # Basic extraction
            >>> result = extractor.extract("John is 25 years old")

            >>> # With refinement for higher accuracy
            >>> result = extractor.extract("John is 25 years old", refine=True)

            >>> # Enable debug mode for detailed validation feedback
            >>> result = extractor.extract("John is 25 years old", debug=True)

            >>> # Custom refinement configuration
            >>> result = extractor.extract(text, refine={
            ...     "strategy": "bon_then_refine",
            ...     "n_candidates": 5,
            ...     "budget": {"max_calls": 10}
            ... })

            >>> # Batch extraction with refinement
            >>> results = extractor.extract(
            ...     texts=["doc1", "doc2", "doc3"],
            ...     refine=True,
            ...     max_workers=5,
            ...     show_progress=True
            ... )
        """
        # Handle list input with parallel processing
        if isinstance(text, list):
            return self._extract_parallel(
                texts=text,
                confidence_threshold=confidence_threshold,
                validate=validate,
                debug=debug,
                return_sources=return_sources,
                max_workers=max_workers,
                show_progress=show_progress,
                rate_limit=rate_limit,
                retry_failed=retry_failed,
                refine=refine,
            )

        # Handle single text input
        return self._extract_single(
            text, confidence_threshold, validate, debug, return_sources, refine
        )

    def _extract_parallel(
        self,
        texts: List[str],
        confidence_threshold: float = 0.0,
        validate: bool = True,
        debug: bool = False,
        return_sources: Optional[bool] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = False,
        rate_limit: Optional[int] = None,
        retry_failed: bool = True,
        refine: Union[bool, Refine, Dict[str, Any], None] = None,
    ) -> List[ExtractionResult]:
        """Extract from multiple texts in parallel.

        Args:
            texts: List of texts to process
            confidence_threshold: Minimum confidence score
            validate: Whether to validate results
            return_sources: Override source grounding
            max_workers: Maximum parallel workers
            show_progress: Show progress bar
            rate_limit: API calls per minute limit
            retry_failed: Whether to raise on failures
            refine: Override refinement config for this call

        Returns:
            List of ExtractionResult objects
        """
        processor = ParallelProcessor(
            max_workers=max_workers,
            rate_limit=rate_limit,
            retry_attempts=3,
            retry_delay=1.0,
        )

        # Create processing function
        def process_fn(text: str) -> ExtractionResult:
            return self._extract_single(
                text=text,
                confidence_threshold=confidence_threshold,
                validate=validate,
                debug=debug,
                return_sources=return_sources,
                refine=refine,
            )

        # Process in parallel
        result = processor.process_batch(
            items=texts,
            process_fn=process_fn,
            show_progress=show_progress,
            desc="Extracting",
        )

        # Handle failures
        if retry_failed:
            result.raise_if_failed()
        elif result.failed:
            warnings.warn(
                f"{len(result.failed)} extraction(s) failed. "
                f"Success rate: {result.success_rate:.1f}%",
                UserWarning,
            )

        return result.get_results()

    def extract_batch(
        self,
        texts: List[str],
        confidence_threshold: float = 0.0,
        validate: bool = True,
        debug: bool = False,
        return_sources: Optional[bool] = None,
        max_workers: int = 10,
        show_progress: bool = True,
        rate_limit: Optional[int] = None,
        return_failures: bool = False,
    ) -> Union[List[ExtractionResult], ProcessingResult]:
        """Batch extract with explicit parallel processing control.

        This method provides more control over batch processing than the standard
        extract() method when called with a list.

        Args:
            texts: List of texts to extract from
            confidence_threshold: Minimum confidence score to accept
            validate: Whether to run validation
            debug: Whether to show detailed validation warnings and suggestions (default: False)
            return_sources: Override source grounding
            max_workers: Number of parallel workers (default: 10)
            show_progress: Show progress bar (default: True)
            rate_limit: API calls per minute limit
            return_failures: If True, returns ProcessingResult with successes/failures

        Returns:
            List[ExtractionResult] if return_failures=False (raises on any failure)
            ProcessingResult if return_failures=True (includes failures)

        Examples:
            >>> # Simple batch with progress
            >>> results = extractor.extract_batch(texts, show_progress=True)

            >>> # Get detailed results including failures
            >>> result = extractor.extract_batch(texts, return_failures=True)
            >>> # Inspect result.successful / result.failed for counts
        """
        processor = ParallelProcessor(
            max_workers=max_workers,
            rate_limit=rate_limit,
            retry_attempts=3,
            retry_delay=1.0,
        )

        # Create processing function
        def process_fn(text: str) -> ExtractionResult:
            return self._extract_single(
                text=text,
                confidence_threshold=confidence_threshold,
                validate=validate,
                debug=debug,
                return_sources=return_sources,
                refine=refine,
            )

        # Process in parallel
        result = processor.process_batch(
            items=texts,
            process_fn=process_fn,
            show_progress=show_progress,
            desc="Batch extracting",
        )

        # Return based on preference
        if return_failures:
            return result
        else:
            result.raise_if_failed()
            return result.get_results()

    def _extract_single(
        self,
        text: str,
        confidence_threshold: float = 0.0,
        validate: bool = True,
        debug: bool = False,
        return_sources: Optional[bool] = None,
        refine: Union[bool, Refine, Dict[str, Any], None] = None,
    ) -> ExtractionResult:
        """Extract from a single text (internal helper method)."""
        if not text.strip():
            return ExtractionResult(entities={}, sources={})

        # Optionally override source grounding for this call only
        overridden = False
        previous_use_sources = None
        try:
            if (
                return_sources is not None
                and hasattr(self.pipeline, "extractor")
                and hasattr(self.pipeline.extractor, "use_sources")
            ):
                previous_use_sources = getattr(self.pipeline.extractor, "use_sources")
                setattr(self.pipeline.extractor, "use_sources", bool(return_sources))
                overridden = True

            # Run extraction pipeline (call bound __call__ so tests can patch it)
            result = self.pipeline.__call__(text)
        finally:
            # Restore previous setting if we overrode it
            if (
                overridden
                and previous_use_sources is not None
                and hasattr(self.pipeline, "extractor")
                and hasattr(self.pipeline.extractor, "use_sources")
            ):
                setattr(self.pipeline.extractor, "use_sources", previous_use_sources)

        # Apply refinement if requested
        refine_trace = None
        if refine is not None or self.refine_config:
            # Parse refine configuration for this call
            effective_refine = (
                self._parse_refine_config(refine)
                if refine is not None
                else self.refine_config
            )

            if effective_refine:
                # Lazily initialize refinement engine if not already created
                if self.refinement_engine is None:
                    self.refinement_engine = RefinementEngine(
                        self.schema, self.pipeline.extractor
                    )

                # Run refinement process
                refined_result, trace = self.refinement_engine(text, effective_refine)
                result = refined_result
                refine_trace = trace

                # Add refinement metadata
                result.metadata.update(
                    {
                        "refinement_applied": True,
                        "refinement_strategy": effective_refine.strategy,
                        "candidates_generated": len(trace.candidates),
                        "refinement_steps": len(trace.refine_diffs),
                        "refinement_budget_used": trace.budget_used,
                    }
                )

        # Filter by confidence threshold
        if result.confidence < confidence_threshold:
            warnings.warn(
                f"Extraction confidence ({result.confidence:.2f}) below threshold "
                f"({confidence_threshold:.2f}). Consider lowering threshold or "
                f"optimizing the extractor.",
                UserWarning,
            )

        # Add original text to metadata for visualization
        result.metadata["original_text"] = text

        # Run validation if requested
        if validate:
            validation_report = result.validate_quality(schema=self.schema, text=text)

            # Add validation info to metadata
            result.metadata.update(
                {
                    "validation_score": validation_report.score,
                    "validation_summary": validation_report.summary,
                    "validation_issues_count": len(validation_report.issues),
                }
            )

            # Show suggestions if there are issues and debug is enabled
            if (
                debug
                and validation_report.suggestions
                and (validation_report.has_warnings or validation_report.has_errors)
            ):

                def _fmt_suggestion(s: str) -> str:
                    s_clean = s.strip()
                    # Avoid double emoji prefix if suggestion already starts with an emoji/bullet
                    if s_clean.startswith("💡") or s_clean.startswith("•"):
                        return f"  {s_clean}"
                    return f"  💡 {s_clean}"

                suggestions_text = "\n".join(
                    _fmt_suggestion(s) for s in validation_report.suggestions[:3]
                )
                warnings.warn(
                    f"Extraction validation found issues:\n{validation_report.summary}\n\n"
                    f"Suggestions:\n{suggestions_text}\n\n"
                    f"Use result.validate_quality() for detailed analysis."
                )

        return result

    def optimize(
        self,
        texts: List[str],
        expected_results: Optional[List[Dict]] = None,
        validation_split: float = 0.2,
    ) -> "LangStruct":
        """Optimize extraction performance on provided data.

        Args:
            texts: Training texts for optimization
            expected_results: Optional ground truth results for supervised optimization
            validation_split: Fraction of data to use for validation

        Returns:
            Self for method chaining
        """
        if not self.optimizer:
            self._initialize_optimizer()

        if not texts:
            warnings.warn("No training data provided for optimization.")
            return self

        # Split data for training/validation
        split_idx = int(len(texts) * (1 - validation_split))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]

        train_expected = None
        val_expected = None
        if expected_results:
            train_expected = expected_results[:split_idx]
            val_expected = expected_results[split_idx:]

        # Run optimization
        optimized_pipeline = self.optimizer.optimize(
            pipeline=self.pipeline,
            train_texts=train_texts,
            val_texts=val_texts or train_texts,  # Use train if no val data
            train_expected=train_expected,
            val_expected=val_expected,
        )

        self.pipeline = optimized_pipeline
        return self

    def evaluate(
        self,
        texts: List[str],
        expected_results: List[Dict],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate extraction performance on test data.

        Args:
            texts: Test texts
            expected_results: Ground truth results
            metrics: List of metrics to compute ("accuracy", "f1", "precision", "recall")

        Returns:
            Dictionary of metric scores
        """
        if len(texts) != len(expected_results):
            raise ValueError("Number of texts must match number of expected results")

        metrics = metrics or ["accuracy", "f1"]
        evaluator = ExtractionMetrics(self.schema)

        # Run extractions
        predictions = self.extract(texts)

        # Calculate metrics
        scores = {}
        for metric in metrics:
            if hasattr(evaluator, f"calculate_{metric}"):
                score = getattr(evaluator, f"calculate_{metric}")(
                    predictions, expected_results
                )
                scores[metric] = score
            else:
                warnings.warn(f"Unknown metric: {metric}")

        return scores

    def export_batch(
        self,
        results: List[ExtractionResult],
        file_path: str,
        format: str = "csv",
        include_metadata: bool = False,
        include_sources: bool = False,
        **kwargs,
    ) -> None:
        """Export batch extraction results to file.

        Args:
            results: List of ExtractionResult objects to export
            file_path: Path to save file
            format: Export format ('csv', 'json', 'excel', 'parquet')
            include_metadata: Whether to include metadata columns
            include_sources: Whether to include source information
            **kwargs: Additional format-specific arguments

        Example:
            >>> results = extractor.extract(texts)
            >>> extractor.export_batch(results, "output.csv")
            >>> extractor.export_batch(results, "output.xlsx", format="excel")
        """
        format = format.lower()

        if format == "json":
            # For JSON, export as array of objects
            json_data = [
                result.to_dict(include_metadata, include_sources) for result in results
            ]
            import json

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, default=str)

        elif format == "csv":
            ExportUtilities.save_csv(
                results, file_path, include_metadata, include_sources
            )

        else:
            # Use DataFrame for other formats (excel, parquet)
            ExportUtilities.save_dataframe(
                results, file_path, format, include_metadata, include_sources, **kwargs
            )

    def save_annotated_documents(
        self,
        results: List[ExtractionResult],
        file_path: Union[str, Path],
        include_metadata: bool = True,
        include_sources: bool = True,
    ) -> None:
        """Save extraction results to JSONL format for LLM data workflows.

        This format matches LangExtract's save_annotated_documents functionality
        and is compatible with popular LLM training and analysis tools.

        Args:
            results: List of ExtractionResult objects to save
            file_path: Path to save JSONL file
            include_metadata: Whether to include metadata in output
            include_sources: Whether to include source grounding information

        Example:
            >>> results = extractor.extract(texts)
            >>> extractor.save_annotated_documents(results, "extractions.jsonl")
        """
        ExportUtilities.save_annotated_documents(
            results, file_path, include_metadata, include_sources
        )

    def load_annotated_documents(
        self, file_path: Union[str, Path]
    ) -> List[ExtractionResult]:
        """Load extraction results from JSONL format.

        Args:
            file_path: Path to JSONL file to load

        Returns:
            List of ExtractionResult objects

        Example:
            >>> results = extractor.load_annotated_documents("extractions.jsonl")
            >>> # Loaded N extraction results
        """
        return ExportUtilities.load_annotated_documents(file_path)

    def visualize(
        self,
        results: Union[str, Path, List[ExtractionResult]],
        file_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Optional[str]:
        """Generate interactive HTML visualization of extraction results.

        Similar to LangExtract's visualize() function, this creates an interactive
        HTML page showing extracted entities with source highlighting.

        Args:
            results: Either path to JSONL file or list of ExtractionResult objects
            file_path: Optional path to save HTML file (if None, returns HTML string)
            **kwargs: Additional arguments for visualization (title, show_confidence, etc.)

        Returns:
            HTML string if file_path is None, otherwise None

        Example:
            >>> results = extractor.extract(texts)
            >>> html = extractor.visualize(results)  # Returns HTML string
            >>> extractor.visualize(results, "results.html")  # Saves to file
            >>> extractor.visualize("extractions.jsonl", "results.html")  # Load and save
        """
        if file_path:
            save_visualization(results, file_path, **kwargs)
            return None
        else:
            return visualize(results, **kwargs)

    def save(self, path: str) -> None:
        """Save the extractor to disk.

        Saves the complete extractor state including:
        - Schema definition and field descriptions
        - Model configuration (without API keys for security)
        - DSPy pipeline state (signatures, examples, optimizations)
        - Chunking configuration
        - Optimizer state (if optimization was applied)
        - Refinement configuration (if refinement was configured)

        The extractor is saved as a directory containing multiple files
        for better organization and debugging.

        Args:
            path: Directory path to save the extractor to (will be created if needed)

        Example:
            >>> extractor = LangStruct(schema=PersonSchema)
            >>> extractor.optimize(train_texts, expected_results)
            >>> extractor.save("./my_extractor")
            >>> # Creates directory with all extractor components
        """
        from .core.persistence import ExtractorPersistence

        ExtractorPersistence.save_extractor(self, path)

    @classmethod
    def load(cls, path: str) -> "LangStruct":
        """Load a previously saved extractor from disk.

        Reconstructs a complete LangStruct extractor from saved state including:
        - Schema (either original class or dynamically reconstructed)
        - Model configuration (API keys must be available in environment)
        - Optimized DSPy pipeline state with learned examples and prompts
        - All configurations (chunking, refinement, etc.)

        Args:
            path: Directory path containing the saved extractor

        Returns:
            Fully reconstructed LangStruct instance ready for extraction

        Raises:
            PersistenceError: If loading fails due to missing files,
                             version incompatibility, or invalid state

        Example:
            >>> # Save an extractor
            >>> extractor.save("./my_extractor")
            >>>
            >>> # Load it back (API keys must be available)
            >>> loaded_extractor = LangStruct.load("./my_extractor")
            >>> result = loaded_extractor.extract("New text")
        """
        from .core.persistence import ExtractorPersistence

        return ExtractorPersistence.load_extractor(path)

    def _initialize_optimizer(self) -> None:
        """Initialize the appropriate optimizer."""
        if self.optimizer_name.lower() == "miprov2":
            self.optimizer = MIPROv2Optimizer()
        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer_name}. Only 'miprov2' is supported."
            )

    def _parse_refine_config(
        self, refine: Union[bool, Refine, Dict[str, Any]]
    ) -> Optional[Refine]:
        """Parse refinement configuration into Refine object."""
        if refine is False or refine is None:
            return None
        elif refine is True:
            # Use defaults
            return Refine()
        elif isinstance(refine, Refine):
            return refine
        elif isinstance(refine, dict):
            return Refine(**refine)
        else:
            raise ValueError(
                f"Invalid refine configuration: {refine}. Must be bool, Refine object, or dict."
            )

    @property
    def schema_info(self) -> Dict[str, Any]:
        """Get information about the extraction schema.

        Returns:
            Dictionary with schema fields, descriptions, and example format
        """
        from .core.schema_utils import (
            get_example_format,
            get_field_descriptions,
            get_json_schema,
        )

        return {
            "fields": list(get_field_descriptions(self.schema).keys()),
            "descriptions": get_field_descriptions(self.schema),
            "json_schema": get_json_schema(self.schema),
            "example_format": get_example_format(self.schema),
        }

    @overload
    def query(self, query: str, explain: bool = True) -> ParsedQuery: ...

    @overload
    def query(
        self,
        query: List[str],
        explain: bool = True,
        max_workers: Optional[int] = None,
        show_progress: bool = False,
        rate_limit: Optional[int] = None,
        retry_failed: bool = True,
    ) -> List[ParsedQuery]: ...

    def query(
        self,
        query: Union[str, List[str]],
        explain: bool = True,
        max_workers: Optional[int] = None,
        show_progress: bool = False,
        rate_limit: Optional[int] = None,
        retry_failed: bool = True,
    ) -> Union[ParsedQuery, List[ParsedQuery]]:
        """Parse natural language query/queries into structured components for RAG.

        Uses LLM intelligence to convert natural language queries into both
        semantic search terms and structured metadata filters, enabling
        precise RAG retrieval. Handles both single queries and batch processing.

        Args:
            query: Natural language query or list of queries to parse
            explain: Whether to generate human-readable explanation
            max_workers: Maximum parallel workers for batch processing (list input only)
            show_progress: Show progress bar for batch processing (requires tqdm)
            rate_limit: API calls per minute limit for batch processing
            retry_failed: Whether to raise exception on failures in batch processing

        Returns:
            ParsedQuery for single query, or List[ParsedQuery] for multiple queries

        Examples:
            >>> # Single query
            >>> ls = LangStruct(example={"company": "Apple", "revenue": 100.0})
            >>> result = ls.query("Q3 tech companies over $100B revenue")
            >>> # result.semantic_terms -> ["tech companies"]
            >>> # result.structured_filters -> {"revenue": {"$gte": 100.0}}

            >>> # Batch queries with parallel processing
            >>> queries = ["query1", "query2", "query3"]
            >>> results = ls.query(queries, max_workers=5, show_progress=True)
        """
        # Handle list input with parallel processing
        if isinstance(query, list):
            return self._query_parallel(
                queries=query,
                explain=explain,
                max_workers=max_workers,
                show_progress=show_progress,
                rate_limit=rate_limit,
                retry_failed=retry_failed,
            )

        # Handle single query
        return self._query_single(query, explain)

    def _query_single(self, query: str, explain: bool = True) -> ParsedQuery:
        """Parse a single query (internal helper method)."""
        if not query.strip():
            return ParsedQuery(
                semantic_terms=[],
                structured_filters={},
                confidence=0.0,
                explanation="Empty query provided",
                raw_query=query,
                metadata={},
            )

        try:
            # Initialize query parser module with LLM intelligence
            if not hasattr(self, "_query_parser"):
                self._query_parser = QueryParser(self.schema)

            # Let the LLM handle ALL parsing - no regex, no hardcoded rules
            parsed = self._query_parser(query)

            # Add explanation if not already present or if explicitly requested
            if explain and not parsed.explanation:
                parsed.explanation = self._query_parser._generate_explanation(
                    query, parsed.semantic_terms, parsed.structured_filters
                )

            return parsed

        except Exception as e:
            warnings.warn(f"Query parsing failed: {e}", UserWarning)
            # Fallback to treating entire query as semantic search
            return ParsedQuery(
                semantic_terms=[query],
                structured_filters={},
                confidence=0.0,
                explanation=f"Parsing failed, treating as semantic search: {e}",
                raw_query=query,
                metadata={"error": str(e)},
            )

    def _query_parallel(
        self,
        queries: List[str],
        explain: bool = True,
        max_workers: Optional[int] = None,
        show_progress: bool = False,
        rate_limit: Optional[int] = None,
        retry_failed: bool = True,
    ) -> List[ParsedQuery]:
        """Parse multiple queries in parallel.

        Args:
            queries: List of queries to parse
            explain: Whether to generate explanations
            max_workers: Maximum parallel workers
            show_progress: Show progress bar
            rate_limit: API calls per minute limit
            retry_failed: Whether to raise on failures

        Returns:
            List of ParsedQuery objects
        """
        processor = ParallelProcessor(
            max_workers=max_workers,
            rate_limit=rate_limit,
            retry_attempts=3,
            retry_delay=1.0,
        )

        # Create processing function
        def process_fn(query: str) -> ParsedQuery:
            return self._query_single(query, explain)

        # Process in parallel
        result = processor.process_batch(
            items=queries,
            process_fn=process_fn,
            show_progress=show_progress,
            desc="Parsing queries",
        )

        # Handle failures
        if retry_failed:
            result.raise_if_failed()
        elif result.failed:
            warnings.warn(
                f"{len(result.failed)} query parsing(s) failed. "
                f"Success rate: {result.success_rate:.1f}%",
                UserWarning,
            )

        return result.get_results()

    def query_batch(
        self,
        queries: List[str],
        explain: bool = True,
        max_workers: int = 10,
        show_progress: bool = True,
        rate_limit: Optional[int] = None,
        return_failures: bool = False,
    ) -> Union[List[ParsedQuery], ProcessingResult]:
        """Batch query parsing with explicit parallel processing control.

        This method provides more control over batch query processing than the
        standard query() method when called with a list.

        Args:
            queries: List of queries to parse
            explain: Whether to generate explanations
            max_workers: Number of parallel workers (default: 10)
            show_progress: Show progress bar (default: True)
            rate_limit: API calls per minute limit
            return_failures: If True, returns ProcessingResult with successes/failures

        Returns:
            List[ParsedQuery] if return_failures=False (raises on any failure)
            ProcessingResult if return_failures=True (includes failures)

        Examples:
            >>> # Simple batch with progress
            >>> results = extractor.query_batch(queries, show_progress=True)

            >>> # Get detailed results including failures
            >>> result = extractor.query_batch(queries, return_failures=True)
            >>> # Inspect result.successful / result.failed for counts
        """
        processor = ParallelProcessor(
            max_workers=max_workers,
            rate_limit=rate_limit,
            retry_attempts=3,
            retry_delay=1.0,
        )

        # Create processing function
        def process_fn(query: str) -> ParsedQuery:
            return self._query_single(query, explain)

        # Process in parallel
        result = processor.process_batch(
            items=queries,
            process_fn=process_fn,
            show_progress=show_progress,
            desc="Batch parsing queries",
        )

        # Return based on preference
        if return_failures:
            return result
        else:
            result.raise_if_failed()
            return result.get_results()

    def __repr__(self) -> str:
        return (
            f"LangStruct(schema={self.schema.__name__}, "
            f"model={self.lm.__class__.__name__}, "
            f"optimizer_initialized={self.optimizer is not None})"
        )
