"""GEPA optimizer integration for reflective prompt optimization with feedback."""

import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import dspy
from dspy.teleprompt import GEPA

from ..core.modules import ExtractionPipeline
from ..core.schemas import ExtractionResult

logger = logging.getLogger(__name__)


def extraction_metric_with_feedback(
    example: dspy.Example,
    pred: ExtractionResult,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    """Feedback metric function for GEPA optimizer.

    Unlike standard metrics that return a float, GEPA metrics return a
    dspy.Prediction with both a score and textual feedback explaining
    the evaluation.

    Args:
        example: Input example with expected output
        pred: Predicted extraction result
        trace: Execution trace (optional)
        pred_name: Prediction name (optional, used by GEPA)
        pred_trace: Prediction trace (optional, used by GEPA)

    Returns:
        dspy.Prediction with score (0-1) and feedback (str)
    """
    # Handle cases where prediction is None or empty
    if pred is None:
        feedback = (
            "Extraction returned None. This indicates a critical failure in the pipeline. "
            "Check the extraction module configuration and ensure the LM is responding."
        )
        return dspy.Prediction(score=0.0, feedback=feedback)

    # Extract the ExtractionResult from the prediction
    # The wrapper wraps ExtractionResult in a dspy.Prediction
    extraction_result = None
    if hasattr(pred, "extraction_result"):
        extraction_result = pred.extraction_result
    elif hasattr(pred, "entities"):
        # Direct ExtractionResult (shouldn't happen with wrapper, but handle it)
        extraction_result = pred

    # Handle cases with no expected output
    if not hasattr(example, "expected") or not example.expected:
        # If no expected output, use confidence as proxy
        if extraction_result and hasattr(extraction_result, "confidence"):
            confidence = extraction_result.confidence
        elif hasattr(pred, "confidence"):
            confidence = pred.confidence
        else:
            confidence = 0.5
        feedback = (
            f"No ground truth provided. Extraction confidence: {confidence:.1%}. "
            "Cannot provide detailed feedback without expected results."
        )
        return dspy.Prediction(score=confidence, feedback=feedback)

    expected = example.expected

    # Extract entities from the prediction
    if extraction_result and hasattr(extraction_result, "entities"):
        actual = extraction_result.entities
    elif hasattr(pred, "entities"):
        actual = pred.entities
    elif isinstance(pred, dict):
        actual = pred
    else:
        # If pred is some other type, try to extract what we can
        actual = {}

    # Handle empty extraction
    if not actual or (isinstance(actual, dict) and len(actual) == 0):
        required_fields = list(expected.keys()) if isinstance(expected, dict) else []
        feedback = (
            "Extraction failed - no entities were extracted. "
            f"Required fields are missing: {required_fields}. "
            "Suggestions: (1) Check if the text contains the required information, "
            "(2) Ensure the extraction prompt is clear about what to extract, "
            "(3) Verify the schema field descriptions are accurate."
        )
        return dspy.Prediction(score=0.0, feedback=feedback)

    # Calculate field-level metrics
    expected_fields = set(expected.keys()) if isinstance(expected, dict) else set()
    actual_fields = set(actual.keys()) if isinstance(actual, dict) else set()

    if not expected_fields:
        confidence = pred.confidence if hasattr(pred, "confidence") else 0.5
        feedback = (
            f"No expected fields to compare against. "
            f"Extracted {len(actual_fields)} fields with confidence {confidence:.1%}."
        )
        return dspy.Prediction(score=confidence, feedback=feedback)

    # Calculate field recall
    missing_fields = expected_fields - actual_fields
    extra_fields = actual_fields - expected_fields
    matched_fields = expected_fields & actual_fields

    field_recall = (
        len(matched_fields) / len(expected_fields) if expected_fields else 0.0
    )

    # Calculate value accuracy for matched fields
    value_matches = []
    value_mismatches = []

    for field in matched_fields:
        expected_val = str(expected.get(field, "")).lower().strip()
        actual_val = str(actual.get(field, "")).lower().strip()

        # Check if values match
        if (
            expected_val == actual_val
            or expected_val in actual_val
            or actual_val in expected_val
        ):
            value_matches.append(field)
        else:
            value_mismatches.append((field, expected_val, actual_val))

    value_accuracy = len(value_matches) / len(matched_fields) if matched_fields else 0.0

    # Combine field recall and value accuracy for overall score
    score = (field_recall * 0.6) + (value_accuracy * 0.4)

    # Boost score if confidence is high
    confidence = None
    if extraction_result and hasattr(extraction_result, "confidence"):
        confidence = extraction_result.confidence
    elif hasattr(pred, "confidence"):
        confidence = pred.confidence

    if confidence is not None:
        score = (score * 0.8) + (confidence * 0.2)

    # Generate detailed feedback
    feedback_parts = []

    # Overall assessment
    if score >= 0.9:
        feedback_parts.append(f"✅ Excellent extraction (score: {score:.1%}).")
    elif score >= 0.7:
        feedback_parts.append(
            f"✓ Good extraction (score: {score:.1%}), but room for improvement."
        )
    elif score >= 0.5:
        feedback_parts.append(
            f"⚠ Partial extraction (score: {score:.1%}). Significant issues found."
        )
    else:
        feedback_parts.append(
            f"❌ Poor extraction (score: {score:.1%}). Major improvements needed."
        )

    # Field coverage feedback
    if missing_fields:
        feedback_parts.append(
            f"\nMissing fields ({len(missing_fields)}): {', '.join(sorted(missing_fields))}. "
            "Ensure these fields are clearly mentioned in the extraction prompt and that "
            "the input text actually contains this information."
        )

    if extra_fields:
        feedback_parts.append(
            f"\nExtra fields extracted ({len(extra_fields)}): {', '.join(sorted(extra_fields))}. "
            "These were not expected. Consider refining the schema to be more specific."
        )

    # Value accuracy feedback
    if value_matches:
        feedback_parts.append(
            f"\nCorrectly extracted values ({len(value_matches)}): {', '.join(sorted(value_matches))}."
        )

    if value_mismatches:
        mismatch_details = []
        for field, expected_val, actual_val in value_mismatches[:3]:  # Show top 3
            mismatch_details.append(
                f"  - {field}: got '{actual_val}' but expected '{expected_val}'"
            )

        feedback_parts.append(
            f"\nValue mismatches ({len(value_mismatches)}):\n"
            + "\n".join(mismatch_details)
        )

        if len(value_mismatches) > 3:
            feedback_parts.append(f"  ... and {len(value_mismatches) - 3} more")

        feedback_parts.append(
            "\nSuggestions for value mismatches: "
            "(1) Add examples showing the exact format expected, "
            "(2) Clarify field descriptions to specify format/type, "
            "(3) Use more precise extraction instructions."
        )

    # Confidence feedback
    if confidence is not None:
        if confidence < 0.6:
            feedback_parts.append(
                f"\n⚠ Low confidence ({confidence:.1%}). The model is uncertain. "
                "Consider: (1) Providing clearer instructions, (2) Adding few-shot examples, "
                "(3) Using a more capable model."
            )
        elif confidence >= 0.9:
            feedback_parts.append(
                f"\n✓ High confidence ({confidence:.1%}). The model is confident in this extraction."
            )

    # Actionable recommendations
    if score < 0.7:
        feedback_parts.append(
            "\n💡 Key improvements to try: "
            "(1) Add 2-3 high-quality examples showing correct extractions, "
            "(2) Make field descriptions more specific and detailed, "
            "(3) Break complex fields into simpler sub-fields, "
            "(4) Ensure the source text actually contains the required information."
        )

    feedback = "".join(feedback_parts)
    return dspy.Prediction(score=min(1.0, score), feedback=feedback)


class GEPAOptimizer:
    """DSPy GEPA optimizer for reflective prompt optimization with feedback.

    GEPA (Genetic-Pareto) uses reflective prompt evolution guided by rich
    textual feedback to optimize extraction quality. It's particularly effective
    for complex reasoning tasks where understanding *why* extractions succeed
    or fail is valuable for improvement.
    """

    def __init__(
        self,
        auto: Literal["light", "medium", "heavy"] = "light",
        num_threads: int = 4,
        reflection_lm: Optional[Union[str, dspy.LM]] = None,
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
        use_merge: bool = False,
        track_stats: bool = False,
        track_best_outputs: bool = False,
        max_full_evals: Optional[int] = None,
        max_metric_calls: Optional[int] = None,
        **kwargs,
    ):
        """Initialize GEPA optimizer.

        Args:
            auto: Optimization budget level ("light", "medium", "heavy")
            num_threads: Number of threads for parallel evaluation
            reflection_lm: Model for generating reflections (defaults to main LM)
            reflection_minibatch_size: Batch size for reflection process
            candidate_selection_strategy: How to select candidates ("pareto" or "current_best")
            use_merge: Whether to merge successful program variants
            track_stats: Whether to track detailed optimization statistics
            track_best_outputs: Whether to track best outputs during optimization
            max_full_evals: Maximum number of full evaluations (overrides auto)
            max_metric_calls: Maximum metric calls (overrides auto)
            **kwargs: Additional arguments for GEPA
        """
        self.auto = auto
        self.num_threads = num_threads
        self.reflection_lm = reflection_lm
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        self.use_merge = use_merge
        self.track_stats = track_stats
        self.track_best_outputs = track_best_outputs
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls
        self.kwargs = kwargs

        # Optimizer is initialized during optimize()
        self.optimizer = None

    def optimize(
        self,
        pipeline: ExtractionPipeline,
        train_texts: List[str],
        train_expected: Optional[List[Dict]] = None,
        val_texts: Optional[List[str]] = None,
        val_expected: Optional[List[Dict]] = None,
        requires_permission_to_run: bool = False,
    ) -> ExtractionPipeline:
        """Optimize extraction pipeline using GEPA with feedback-driven evolution.

        Args:
            pipeline: Extraction pipeline to optimize
            train_texts: Training texts
            train_expected: Expected results for training (required for GEPA)
            val_texts: Validation texts (optional)
            val_expected: Expected results for validation (optional)
            requires_permission_to_run: Whether permission is required

        Returns:
            Optimized extraction pipeline
        """
        if not train_texts:
            warnings.warn("No training texts provided, returning original pipeline")
            return pipeline

        if not train_expected:
            warnings.warn(
                "GEPA works best with expected results for generating feedback. "
                "No expected results provided - optimization may be less effective."
            )

        # Create DSPy examples from training data
        trainset = []
        for i, text in enumerate(train_texts):
            example_kwargs = {"text": text}
            if train_expected and i < len(train_expected):
                example_kwargs["expected"] = train_expected[i]
            trainset.append(dspy.Example(**example_kwargs).with_inputs("text"))

        # Create validation set if provided
        valset = None
        if val_texts:
            valset = []
            for i, text in enumerate(val_texts):
                example_kwargs = {"text": text}
                if val_expected and i < len(val_expected):
                    example_kwargs["expected"] = val_expected[i]
                valset.append(dspy.Example(**example_kwargs).with_inputs("text"))

        # Configure reflection model - GEPA requires one
        reflection_lm = self.reflection_lm
        if isinstance(reflection_lm, str):
            reflection_lm = dspy.LM(reflection_lm)
        elif reflection_lm is None:
            # Default to the current DSPy LM if none provided
            # GEPA requires a reflection_lm, so we must provide one
            try:
                reflection_lm = dspy.settings.lm
                if reflection_lm is None:
                    logger.warning(
                        "No reflection_lm specified and no default LM configured. "
                        "GEPA may fail. Consider providing reflection_lm parameter."
                    )
            except (AttributeError, Exception):
                logger.warning(
                    "Could not access default LM for reflection. "
                    "GEPA optimization may fail without a reflection_lm."
                )

        # Initialize GEPA optimizer with feedback metric
        optimizer_kwargs = {
            "metric": extraction_metric_with_feedback,
            "num_threads": self.num_threads,
            "reflection_minibatch_size": self.reflection_minibatch_size,
            "candidate_selection_strategy": self.candidate_selection_strategy,
            "use_merge": self.use_merge,
            "track_stats": self.track_stats,
            "track_best_outputs": self.track_best_outputs,
            **self.kwargs,
        }

        # Include format failures in reflective feedback so early runs still provide signal.
        optimizer_kwargs.setdefault("add_format_failure_as_feedback", True)

        # Set budget - prefer explicit params over auto
        if self.max_full_evals is not None:
            optimizer_kwargs["max_full_evals"] = self.max_full_evals
        elif self.max_metric_calls is not None:
            optimizer_kwargs["max_metric_calls"] = self.max_metric_calls
        else:
            optimizer_kwargs["auto"] = self.auto

        # Always set reflection_lm (GEPA requires it)
        if reflection_lm is not None:
            optimizer_kwargs["reflection_lm"] = reflection_lm

        self.optimizer = GEPA(**optimizer_kwargs)

        # Wrapper needed to convert ExtractionResult to dspy.Prediction
        # GEPA's trace system requires dspy.Prediction at the top level
        class DSPyWrapper(dspy.Module):
            """Wrapper that makes ExtractionPipeline compatible with GEPA.

            Key requirements:
            1. Return dspy.Prediction (GEPA needs this for trace analysis)
            2. Include ExtractionResult data (our metric needs this)
            3. Expose internal predictors (GEPA optimizes these)
            """

            def __init__(self, extraction_pipeline):
                super().__init__()
                # Store the pipeline - this makes it discoverable by named_predictors()
                # since ExtractionPipeline is a dspy.Module with internal predictors
                self.pipeline = extraction_pipeline

            def forward(self, text: str) -> dspy.Prediction:
                """Forward pass that returns dspy.Prediction for GEPA."""
                try:
                    # Call the pipeline normally
                    extraction_result = self.pipeline(text)

                    # Convert to dspy.Prediction for GEPA's trace system
                    # Store the ExtractionResult so our metric can access it
                    return dspy.Prediction(
                        extraction_result=extraction_result,
                        # Also expose key fields directly for compatibility
                        entities=extraction_result.entities,
                        sources=extraction_result.sources,
                        confidence=extraction_result.confidence,
                    )
                except Exception as e:
                    logger.error(f"Extraction failed: {e}", exc_info=True)
                    from ..core.schemas import ExtractionResult

                    empty = ExtractionResult(entities={}, sources={}, confidence=0.0)
                    return dspy.Prediction(
                        extraction_result=empty,
                        entities={},
                        sources={},
                        confidence=0.0,
                    )

        wrapped_pipeline = DSPyWrapper(pipeline)

        try:
            # Compile the optimized program
            logger.info(
                "Optimizing extraction pipeline with GEPA (%s budget)...",
                (
                    self.auto
                    if not self.max_full_evals
                    else f"{self.max_full_evals} evals"
                ),
            )
            logger.info("Training examples: %d", len(trainset))
            if valset:
                logger.info("Validation examples: %d", len(valset))

            compile_kwargs = {
                "trainset": trainset,
            }

            if valset:
                compile_kwargs["valset"] = valset

            # Note: GEPA.compile() doesn't support requires_permission_to_run
            # It's handled differently than MIPROv2
            optimized_wrapper = self.optimizer.compile(
                student=wrapped_pipeline, **compile_kwargs
            )

            # Extract the optimized pipeline from the wrapper
            optimized_pipeline = optimized_wrapper.pipeline

            logger.info("GEPA optimization completed successfully")
            return optimized_pipeline

        except Exception as e:
            warnings.warn(
                f"GEPA optimization failed: {str(e)}. Returning original pipeline."
            )
            return pipeline
