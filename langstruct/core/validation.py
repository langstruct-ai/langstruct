"""Validation and suggestion system for extraction quality assessment."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from .schema_utils import get_field_descriptions, get_json_schema
from .schemas import ExtractionResult, SourceSpan


class IssueType(Enum):
    """Types of validation issues."""

    LOW_CONFIDENCE = "low_confidence"
    MISSING_FIELDS = "missing_fields"
    EMPTY_EXTRACTION = "empty_extraction"
    NO_SOURCE_GROUNDING = "no_source_grounding"
    WEAK_SOURCE_GROUNDING = "weak_source_grounding"
    TYPE_MISMATCH = "type_mismatch"
    INCONSISTENT_RESULTS = "inconsistent_results"
    SCHEMA_DESIGN = "schema_design"


class Severity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue with suggested fixes."""

    type: IssueType
    severity: Severity
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    fix_code: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report with issues and suggestions."""

    is_valid: bool
    score: float  # Overall quality score 0-1
    issues: List[ValidationIssue]
    suggestions: List[str]
    summary: str

    @property
    def has_errors(self) -> bool:
        """Check if report has any errors or critical issues."""
        return any(
            issue.severity in [Severity.ERROR, Severity.CRITICAL]
            for issue in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """Check if report has any warnings."""
        return any(issue.severity == Severity.WARNING for issue in self.issues)


class ExtractionValidator:
    """Validates extraction results and provides improvement suggestions."""

    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema
        self.field_names = set(get_field_descriptions(schema).keys())
        self.field_descriptions = get_field_descriptions(schema)

    def validate(
        self,
        result: ExtractionResult,
        text: Optional[str] = None,
        previous_results: Optional[List[ExtractionResult]] = None,
    ) -> ValidationReport:
        """Validate an extraction result and generate suggestions.

        Args:
            result: ExtractionResult to validate
            text: Original text (for advanced validation)
            previous_results: Previous results for consistency checking

        Returns:
            ValidationReport with issues and suggestions
        """
        issues = []

        # Basic validation checks
        issues.extend(self._check_confidence(result))
        issues.extend(self._check_completeness(result))
        issues.extend(self._check_empty_extraction(result))
        issues.extend(self._check_source_grounding(result))
        issues.extend(self._check_type_consistency(result))

        # Advanced checks if text is provided
        if text:
            issues.extend(self._check_text_length_vs_results(result, text))
            issues.extend(self._check_source_alignment(result, text))

        # Consistency checks if previous results provided
        if previous_results:
            issues.extend(self._check_result_consistency(result, previous_results))

        # Schema design checks
        issues.extend(self._check_schema_design())

        # Calculate overall score
        score = self._calculate_quality_score(result, issues)

        # Determine if valid (no errors/critical issues)
        is_valid = not any(
            issue.severity in [Severity.ERROR, Severity.CRITICAL] for issue in issues
        )

        # Generate summary and suggestions
        suggestions = self._generate_suggestions(issues)
        summary = self._generate_summary(result, issues, score)

        return ValidationReport(
            is_valid=is_valid,
            score=score,
            issues=issues,
            suggestions=suggestions,
            summary=summary,
        )

    def _check_confidence(self, result: ExtractionResult) -> List[ValidationIssue]:
        """Check extraction confidence levels."""
        issues = []

        if result.confidence < 0.3:
            issues.append(
                ValidationIssue(
                    type=IssueType.LOW_CONFIDENCE,
                    severity=Severity.CRITICAL,
                    message=f"Very low confidence ({result.confidence:.1%}) indicates poor extraction quality",
                    suggestion="Consider improving schema descriptions or using different model",
                    fix_code="extractor = LangStruct(schema=YourSchema, model='gpt-5-mini')",
                )
            )
        elif result.confidence < 0.6:
            issues.append(
                ValidationIssue(
                    type=IssueType.LOW_CONFIDENCE,
                    severity=Severity.WARNING,
                    message=f"Low confidence ({result.confidence:.1%}) may indicate extraction issues",
                    suggestion="Review field descriptions and consider adding examples",
                    fix_code="# Add more descriptive field descriptions\nname: str = Field(description='Full legal name of the person')",
                )
            )

        return issues

    def _check_completeness(self, result: ExtractionResult) -> List[ValidationIssue]:
        """Check if all expected fields were extracted."""
        issues = []
        extracted_fields = set(result.entities.keys())
        missing_fields = self.field_names - extracted_fields

        if missing_fields:
            issues.append(
                ValidationIssue(
                    type=IssueType.MISSING_FIELDS,
                    severity=Severity.WARNING,
                    message=f"Missing fields: {', '.join(missing_fields)}",
                    suggestion="Check if these fields exist in the text or make them optional",
                    fix_code=f"# Make fields optional:\n{', '.join(f'{field}: Optional[str]' for field in missing_fields)}",
                )
            )

        return issues

    def _check_empty_extraction(
        self, result: ExtractionResult
    ) -> List[ValidationIssue]:
        """Check for completely empty extractions."""
        issues = []
        non_empty_values = [
            v for v in result.entities.values() if v is not None and str(v).strip()
        ]

        if not non_empty_values:
            issues.append(
                ValidationIssue(
                    type=IssueType.EMPTY_EXTRACTION,
                    severity=Severity.CRITICAL,
                    message="No valid data extracted from text",
                    suggestion="Text may not contain expected information or schema is mismatched",
                    fix_code="# Try a simpler schema or check text content",
                )
            )
        elif len(non_empty_values) < len(self.field_names) * 0.5:
            issues.append(
                ValidationIssue(
                    type=IssueType.EMPTY_EXTRACTION,
                    severity=Severity.WARNING,
                    message=f"Only {len(non_empty_values)}/{len(self.field_names)} fields extracted successfully",
                    suggestion="Consider schema simplification or better field descriptions",
                )
            )

        return issues

    def _check_source_grounding(
        self, result: ExtractionResult
    ) -> List[ValidationIssue]:
        """Check source grounding quality."""
        issues = []

        if not result.sources:
            issues.append(
                ValidationIssue(
                    type=IssueType.NO_SOURCE_GROUNDING,
                    severity=Severity.INFO,
                    message="No source grounding information available",
                    suggestion="Enable source grounding for better verification",
                    fix_code="extractor = LangStruct(schema=YourSchema, use_sources=True)",
                )
            )
            return issues

        # Check grounding coverage
        grounded_fields = len(
            [field for field, spans in result.sources.items() if spans]
        )
        extracted_fields = len([v for v in result.entities.values() if v is not None])

        if extracted_fields > 0 and grounded_fields / extracted_fields < 0.5:
            issues.append(
                ValidationIssue(
                    type=IssueType.WEAK_SOURCE_GROUNDING,
                    severity=Severity.WARNING,
                    message=f"Only {grounded_fields}/{extracted_fields} fields have source grounding",
                    suggestion="Extracted values may not be well-grounded in source text",
                )
            )

        return issues

    def _check_type_consistency(
        self, result: ExtractionResult
    ) -> List[ValidationIssue]:
        """Check if extracted values match expected types."""
        issues = []
        schema_fields = get_json_schema(self.schema).get("properties", {})

        for field_name, value in result.entities.items():
            if value is None:
                continue

            expected_type = schema_fields.get(field_name, {}).get("type")
            if not expected_type:
                continue

            # Check type consistency
            if expected_type == "integer" and not isinstance(value, int):
                try:
                    int(str(value))
                except ValueError:
                    issues.append(
                        ValidationIssue(
                            type=IssueType.TYPE_MISMATCH,
                            severity=Severity.WARNING,
                            message=f"Field '{field_name}' expected integer, got: {type(value).__name__}",
                            field=field_name,
                            suggestion="Check field description or validation logic",
                        )
                    )

        return issues

    def _check_text_length_vs_results(
        self, result: ExtractionResult, text: str
    ) -> List[ValidationIssue]:
        """Check if results are appropriate for text length."""
        issues = []
        text_length = len(text.strip())
        non_empty_fields = len(
            [v for v in result.entities.values() if v is not None and str(v).strip()]
        )

        if text_length < 100 and non_empty_fields > 3:
            issues.append(
                ValidationIssue(
                    type=IssueType.INCONSISTENT_RESULTS,
                    severity=Severity.WARNING,
                    message="Many fields extracted from short text - may indicate over-extraction",
                    suggestion="Verify extraction accuracy for short texts",
                )
            )
        elif text_length > 1000 and non_empty_fields < 2:
            issues.append(
                ValidationIssue(
                    type=IssueType.INCONSISTENT_RESULTS,
                    severity=Severity.WARNING,
                    message="Few fields extracted from long text - may indicate under-extraction",
                    suggestion="Check if text contains expected information",
                )
            )

        return issues

    def _check_source_alignment(
        self, result: ExtractionResult, text: str
    ) -> List[ValidationIssue]:
        """Check if source spans align properly with extracted values."""
        issues = []

        for field_name, spans in result.sources.items():
            extracted_value = result.entities.get(field_name)
            if not spans or extracted_value is None:
                continue

            for span in spans:
                # Check if span is within text bounds
                if span.start < 0 or span.end > len(text):
                    issues.append(
                        ValidationIssue(
                            type=IssueType.WEAK_SOURCE_GROUNDING,
                            severity=Severity.ERROR,
                            message=f"Source span for '{field_name}' is out of bounds",
                            field=field_name,
                        )
                    )
                    continue

                # Check if span text matches extracted value
                actual_text = text[span.start : span.end]
                if actual_text != span.text:
                    issues.append(
                        ValidationIssue(
                            type=IssueType.WEAK_SOURCE_GROUNDING,
                            severity=Severity.WARNING,
                            message=f"Source span text mismatch for '{field_name}'",
                            field=field_name,
                        )
                    )

        return issues

    def _check_result_consistency(
        self, result: ExtractionResult, previous_results: List[ExtractionResult]
    ) -> List[ValidationIssue]:
        """Check consistency with previous results."""
        issues = []

        if len(previous_results) < 2:
            return issues

        # Check confidence consistency
        confidences = [r.confidence for r in previous_results] + [result.confidence]
        avg_confidence = sum(confidences) / len(confidences)

        if abs(result.confidence - avg_confidence) > 0.3:
            issues.append(
                ValidationIssue(
                    type=IssueType.INCONSISTENT_RESULTS,
                    severity=Severity.INFO,
                    message=f"Confidence ({result.confidence:.1%}) differs significantly from average ({avg_confidence:.1%})",
                    suggestion="This result may be unusually good or poor quality",
                )
            )

        return issues

    def _check_schema_design(self) -> List[ValidationIssue]:
        """Check for potential schema design issues."""
        issues = []

        # Check for very short field descriptions
        short_descriptions = [
            field
            for field, desc in self.field_descriptions.items()
            if len(desc.strip()) < 10
        ]

        if short_descriptions:
            issues.append(
                ValidationIssue(
                    type=IssueType.SCHEMA_DESIGN,
                    severity=Severity.INFO,
                    message=f"Fields with short descriptions: {', '.join(short_descriptions)}",
                    suggestion="More detailed descriptions improve extraction accuracy",
                    fix_code="# Use detailed descriptions:\nname: str = Field(description='Full legal name including first and last name')",
                )
            )

        # Check for too many fields (complexity)
        if len(self.field_names) > 10:
            issues.append(
                ValidationIssue(
                    type=IssueType.SCHEMA_DESIGN,
                    severity=Severity.INFO,
                    message=f"Schema has {len(self.field_names)} fields - consider breaking into smaller schemas",
                    suggestion="Complex schemas may reduce extraction accuracy",
                )
            )

        return issues

    def _calculate_quality_score(
        self, result: ExtractionResult, issues: List[ValidationIssue]
    ) -> float:
        """Calculate overall quality score from 0 to 1."""
        base_score = result.confidence

        # Penalty for issues
        penalty = 0
        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                penalty += 0.3
            elif issue.severity == Severity.ERROR:
                penalty += 0.2
            elif issue.severity == Severity.WARNING:
                penalty += 0.1
            # INFO issues don't affect score

        # Bonus for completeness
        non_empty_count = len(
            [v for v in result.entities.values() if v is not None and str(v).strip()]
        )
        completeness_bonus = (non_empty_count / len(self.field_names)) * 0.1

        # Bonus for source grounding
        grounding_bonus = (
            0.05
            if result.sources and any(spans for spans in result.sources.values())
            else 0
        )

        final_score = max(
            0, min(1, base_score - penalty + completeness_bonus + grounding_bonus)
        )
        return final_score

    def _generate_suggestions(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable suggestions from issues."""
        suggestions = []

        # Group issues by type for better suggestions
        issue_types = set(issue.type for issue in issues)

        if IssueType.LOW_CONFIDENCE in issue_types:
            suggestions.append("🎯 Try a more powerful model (e.g. gpt-5-mini)")
            suggestions.append("📝 Add more detailed field descriptions")
            suggestions.append(
                "🔄 Run extractor.optimize(...) with representative data"
            )

        if IssueType.MISSING_FIELDS in issue_types:
            suggestions.append("❓ Make optional fields Optional[type] in schema")
            suggestions.append("🔍 Verify the text contains all expected information")

        if IssueType.EMPTY_EXTRACTION in issue_types:
            suggestions.append("🎨 Simplify schema or check text-schema alignment")
            suggestions.append("📋 Try auto-generating schema from examples")

        if IssueType.NO_SOURCE_GROUNDING in issue_types:
            suggestions.append("🎯 Enable source grounding: use_sources=True")

        if IssueType.SCHEMA_DESIGN in issue_types:
            suggestions.append("📖 Use detailed field descriptions (10+ words)")
            suggestions.append("🔧 Consider breaking complex schemas into simpler ones")

        # Add specific suggestions from issues
        for issue in issues:
            if issue.suggestion and issue.suggestion not in suggestions:
                suggestions.append(f"💡 {issue.suggestion}")

        return suggestions[:5]  # Limit to top 5 suggestions

    def _generate_summary(
        self, result: ExtractionResult, issues: List[ValidationIssue], score: float
    ) -> str:
        """Generate a summary of the validation results."""
        issue_counts = {
            Severity.CRITICAL: len(
                [i for i in issues if i.severity == Severity.CRITICAL]
            ),
            Severity.ERROR: len([i for i in issues if i.severity == Severity.ERROR]),
            Severity.WARNING: len(
                [i for i in issues if i.severity == Severity.WARNING]
            ),
            Severity.INFO: len([i for i in issues if i.severity == Severity.INFO]),
        }

        if score >= 0.8:
            quality = "excellent"
        elif score >= 0.6:
            quality = "good"
        elif score >= 0.4:
            quality = "fair"
        else:
            quality = "poor"

        summary = f"Extraction quality: {quality} (score: {score:.1%}, confidence: {result.confidence:.1%})"

        if issue_counts[Severity.CRITICAL] > 0:
            summary += f" - {issue_counts[Severity.CRITICAL]} critical issues"
        elif issue_counts[Severity.ERROR] > 0:
            summary += f" - {issue_counts[Severity.ERROR]} errors"
        elif issue_counts[Severity.WARNING] > 0:
            summary += f" - {issue_counts[Severity.WARNING]} warnings"
        else:
            summary += " - no issues found"

        return summary
