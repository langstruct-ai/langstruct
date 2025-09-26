"""MIPROv2 optimizer integration for automatic prompt optimization."""

import warnings
from typing import Any, Dict, List, Optional, Union

import dspy
from dspy.teleprompt import MIPROv2

from ..core.modules import ExtractionPipeline
from ..core.schemas import ExtractionResult


def extraction_metric(
    example: dspy.Example, pred: ExtractionResult, trace=None
) -> float:
    """Metric function for evaluating extraction quality.

    Args:
        example: Input example with expected output
        pred: Predicted extraction result
        trace: Execution trace (optional)

    Returns:
        Score between 0 and 1
    """
    if not hasattr(example, "expected") or not example.expected:
        # If no expected output, use confidence as proxy
        return pred.confidence if hasattr(pred, "confidence") else 0.5

    expected = example.expected
    actual = pred.entities if hasattr(pred, "entities") else pred

    if not actual:
        return 0.0

    # Calculate field overlap
    expected_fields = set(expected.keys()) if isinstance(expected, dict) else set()
    actual_fields = set(actual.keys()) if isinstance(actual, dict) else set()

    if not expected_fields:
        return pred.confidence if hasattr(pred, "confidence") else 0.5

    # Field recall: how many expected fields were extracted
    field_recall = (
        len(expected_fields & actual_fields) / len(expected_fields)
        if expected_fields
        else 0.0
    )

    # Value accuracy: how many extracted values match expected
    value_matches = 0
    total_values = 0

    for field in expected_fields & actual_fields:
        total_values += 1
        expected_val = str(expected.get(field, "")).lower().strip()
        actual_val = str(actual.get(field, "")).lower().strip()

        # Exact match or substring match
        if (
            expected_val == actual_val
            or expected_val in actual_val
            or actual_val in expected_val
        ):
            value_matches += 1

    value_accuracy = value_matches / total_values if total_values > 0 else 0.0

    # Combine field recall and value accuracy
    score = (field_recall * 0.6) + (value_accuracy * 0.4)

    # Boost score if confidence is high
    if hasattr(pred, "confidence"):
        score = (score * 0.8) + (pred.confidence * 0.2)

    return min(1.0, score)


class MIPROv2Optimizer:
    """DSPy MIPROv2 optimizer for joint instruction and example optimization."""

    def __init__(
        self,
        auto: str = "light",
        num_threads: int = 4,
        prompt_model: Optional[Union[str, dspy.LM]] = None,
        **kwargs,
    ):
        """Initialize MIPROv2 optimizer.

        Args:
            auto: Optimization level ("light", "medium", "heavy")
            num_threads: Number of threads for parallel optimization
            prompt_model: Model for generating instruction candidates
            **kwargs: Additional arguments for MIPROv2
        """
        self.auto = auto
        self.num_threads = num_threads
        self.prompt_model = prompt_model
        self.kwargs = kwargs

        # Initialize the DSPy MIPROv2 optimizer
        self.optimizer = None

    def optimize(
        self,
        pipeline: ExtractionPipeline,
        train_texts: List[str],
        train_expected: Optional[List[Dict]] = None,
        val_texts: Optional[List[str]] = None,
        val_expected: Optional[List[Dict]] = None,
        max_bootstrapped_demos: int = 2,
        max_labeled_demos: int = 2,
        requires_permission_to_run: bool = False,
    ) -> ExtractionPipeline:
        """Optimize extraction pipeline using MIPROv2.

        Args:
            pipeline: Extraction pipeline to optimize
            train_texts: Training texts
            train_expected: Expected results for training
            val_texts: Validation texts (optional)
            val_expected: Expected results for validation (optional)
            max_bootstrapped_demos: Maximum bootstrapped demonstrations
            max_labeled_demos: Maximum labeled demonstrations
            requires_permission_to_run: Whether permission is required

        Returns:
            Optimized extraction pipeline
        """
        if not train_texts:
            warnings.warn("No training texts provided, returning original pipeline")
            return pipeline

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

        # Configure prompt model if provided
        prompt_model = self.prompt_model
        if isinstance(prompt_model, str):
            prompt_model = dspy.LM(prompt_model)

        # Initialize MIPROv2 optimizer
        optimizer_kwargs = {
            "metric": extraction_metric,
            "auto": self.auto,
            "num_threads": self.num_threads,
            **self.kwargs,
        }

        if prompt_model:
            optimizer_kwargs["prompt_model"] = prompt_model

        self.optimizer = MIPROv2(**optimizer_kwargs)

        # Create a wrapper to make the pipeline compatible with DSPy optimization
        import logging

        logger = logging.getLogger(__name__)

        class ExtractorWrapper(dspy.Module):
            def __init__(self, extraction_pipeline):
                super().__init__()
                self.pipeline = extraction_pipeline

            def forward(self, text: str) -> ExtractionResult:
                return self.pipeline(text)

        wrapped_pipeline = ExtractorWrapper(pipeline)

        try:
            # Compile the optimized program
            logger.info(
                "Optimizing extraction pipeline with MIPROv2 (%s mode)...", self.auto
            )
            logger.info("Training examples: %d", len(trainset))
            if valset:
                logger.info("Validation examples: %d", len(valset))

            compile_kwargs = {
                "max_bootstrapped_demos": max_bootstrapped_demos,
                "max_labeled_demos": max_labeled_demos,
                "requires_permission_to_run": requires_permission_to_run,
            }

            if valset:
                compile_kwargs["valset"] = valset

            optimized_wrapper = self.optimizer.compile(
                wrapped_pipeline, trainset=trainset, **compile_kwargs
            )

            # Extract the optimized pipeline
            optimized_pipeline = optimized_wrapper.pipeline

            logger.info("MIPROv2 optimization completed successfully")
            return optimized_pipeline

        except Exception as e:
            warnings.warn(
                f"MIPROv2 optimization failed: {str(e)}. Returning original pipeline."
            )
            return pipeline
