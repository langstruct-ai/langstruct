"""Evaluation metrics for extraction quality assessment."""

import json
from typing import Any, Dict, List, Type

from pydantic import BaseModel

from ..core.schema_utils import get_field_descriptions
from ..core.schemas import ExtractionResult


class ExtractionMetrics:
    """Metrics for evaluating extraction quality."""

    def __init__(self, schema: Type[BaseModel]):
        """Initialize metrics calculator.

        Args:
            schema: Expected schema for extractions
        """
        self.schema = schema
        self.field_names = set(get_field_descriptions(schema).keys())

    def calculate_accuracy(
        self, predictions: List[ExtractionResult], expected: List[Dict[str, Any]]
    ) -> float:
        """Calculate field-level extraction accuracy.

        Args:
            predictions: List of extraction results
            expected: List of expected results

        Returns:
            Accuracy score between 0 and 1
        """
        if len(predictions) != len(expected):
            raise ValueError("Predictions and expected results must have same length")

        total_fields = 0
        correct_fields = 0

        for pred, exp in zip(predictions, expected):
            for field_name in self.field_names:
                total_fields += 1
                pred_value = pred.entities.get(field_name)
                exp_value = exp.get(field_name)

                if self._values_match(pred_value, exp_value):
                    correct_fields += 1

        return correct_fields / total_fields if total_fields > 0 else 0.0

    def calculate_f1(
        self, predictions: List[ExtractionResult], expected: List[Dict[str, Any]]
    ) -> float:
        """Calculate F1 score for extraction.

        Args:
            predictions: List of extraction results
            expected: List of expected results

        Returns:
            F1 score between 0 and 1
        """
        precision = self.calculate_precision(predictions, expected)
        recall = self.calculate_recall(predictions, expected)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_precision(
        self, predictions: List[ExtractionResult], expected: List[Dict[str, Any]]
    ) -> float:
        """Calculate precision for extraction.

        Args:
            predictions: List of extraction results
            expected: List of expected results

        Returns:
            Precision score between 0 and 1
        """
        if len(predictions) != len(expected):
            raise ValueError("Predictions and expected results must have same length")

        total_predicted = 0
        correct_predicted = 0

        for pred, exp in zip(predictions, expected):
            for field_name in self.field_names:
                pred_value = pred.entities.get(field_name)
                exp_value = exp.get(field_name)

                if pred_value is not None and str(pred_value).strip():
                    total_predicted += 1
                    if self._values_match(pred_value, exp_value):
                        correct_predicted += 1

        return correct_predicted / total_predicted if total_predicted > 0 else 0.0

    def calculate_recall(
        self, predictions: List[ExtractionResult], expected: List[Dict[str, Any]]
    ) -> float:
        """Calculate recall for extraction.

        Args:
            predictions: List of extraction results
            expected: List of expected results

        Returns:
            Recall score between 0 and 1
        """
        if len(predictions) != len(expected):
            raise ValueError("Predictions and expected results must have same length")

        total_expected = 0
        correct_predicted = 0

        for pred, exp in zip(predictions, expected):
            for field_name in self.field_names:
                pred_value = pred.entities.get(field_name)
                exp_value = exp.get(field_name)

                if exp_value is not None and str(exp_value).strip():
                    total_expected += 1
                    if self._values_match(pred_value, exp_value):
                        correct_predicted += 1

        return correct_predicted / total_expected if total_expected > 0 else 0.0

    def calculate_confidence_calibration(
        self, predictions: List[ExtractionResult], expected: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence calibration metrics.

        Args:
            predictions: List of extraction results
            expected: List of expected results

        Returns:
            Dictionary with calibration metrics
        """
        # TODO: Implement confidence calibration
        # This would measure how well confidence scores correlate with accuracy

        confidences = [pred.confidence for pred in predictions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        accuracy = self.calculate_accuracy(predictions, expected)

        return {
            "average_confidence": avg_confidence,
            "accuracy": accuracy,
            "confidence_accuracy_gap": abs(avg_confidence - accuracy),
        }

    def _values_match(self, predicted: Any, expected: Any) -> bool:
        """Check if predicted and expected values match."""
        # Handle None values
        if predicted is None and expected is None:
            return True
        if predicted is None or expected is None:
            return False

        # Convert to strings for comparison
        pred_str = str(predicted).strip().lower()
        exp_str = str(expected).strip().lower()

        # Exact match
        if pred_str == exp_str:
            return True

        # For numeric values, try numeric comparison
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            return abs(pred_num - exp_num) < 1e-6
        except (ValueError, TypeError):
            pass

        # For lists, compare as sets (order doesn't matter)
        if isinstance(predicted, list) and isinstance(expected, list):
            try:
                return set(str(x).strip().lower() for x in predicted) == set(
                    str(x).strip().lower() for x in expected
                )
            except:
                pass

        return False
