"""Tests for validation and suggestion system."""

from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from langstruct.core.schemas import ExtractionResult, SourceSpan
from langstruct.core.validation import (
    ExtractionValidator,
    IssueType,
    Severity,
    ValidationIssue,
    ValidationReport,
)


class ValidationTestSchema(BaseModel):
    """Test schema for validation testing."""

    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    location: Optional[str] = Field(description="Current location or address")
    email: Optional[str] = Field(description="Email address")


class TestExtractionValidator:
    """Tests for ExtractionValidator class."""

    def test_validator_creation(self):
        """Test creating ExtractionValidator."""
        validator = ExtractionValidator(ValidationTestSchema)

        assert validator.schema == ValidationTestSchema
        assert "name" in validator.field_names
        assert "age" in validator.field_names
        assert "location" in validator.field_names
        assert "email" in validator.field_names

    def test_validate_high_confidence(self):
        """Test validation of high-confidence result."""
        validator = ExtractionValidator(ValidationTestSchema)

        result = ExtractionResult(
            entities={
                "name": "John Doe",
                "age": 30,
                "location": "New York",
                "email": "john@example.com",
            },
            sources={
                "name": [SourceSpan(start=0, end=8, text="John Doe")],
                "age": [SourceSpan(start=15, end=17, text="30")],
                "location": [SourceSpan(start=25, end=33, text="New York")],
            },
            confidence=0.95,
        )

        report = validator.validate(result)

        assert report.is_valid
        assert report.score >= 0.9
        assert not report.has_errors
        assert (
            len(
                [
                    i
                    for i in report.issues
                    if i.severity in [Severity.ERROR, Severity.CRITICAL]
                ]
            )
            == 0
        )

    def test_validate_low_confidence(self):
        """Test validation of low-confidence result."""
        validator = ExtractionValidator(ValidationTestSchema)

        result = ExtractionResult(
            entities={"name": "John", "age": 25},
            sources={},
            confidence=0.25,  # Very low confidence
        )

        report = validator.validate(result)

        assert not report.is_valid  # Should be invalid due to critical issues
        assert report.score < 0.5
        assert report.has_errors or any(
            i.severity == Severity.CRITICAL for i in report.issues
        )

        # Should have low confidence issue
        low_conf_issues = [
            i for i in report.issues if i.type == IssueType.LOW_CONFIDENCE
        ]
        assert len(low_conf_issues) > 0
        assert low_conf_issues[0].severity == Severity.CRITICAL

    def test_validate_missing_fields(self):
        """Test validation with missing required fields."""
        validator = ExtractionValidator(ValidationTestSchema)

        result = ExtractionResult(
            entities={"name": "John"},  # Missing age (required)
            sources={},
            confidence=0.8,
        )

        report = validator.validate(result)

        # Should detect missing fields
        missing_field_issues = [
            i for i in report.issues if i.type == IssueType.MISSING_FIELDS
        ]
        assert len(missing_field_issues) > 0
        assert "age" in missing_field_issues[0].message

    def test_validate_empty_extraction(self):
        """Test validation of completely empty extraction."""
        validator = ExtractionValidator(ValidationTestSchema)

        result = ExtractionResult(
            entities={}, sources={}, confidence=0.1  # Completely empty
        )

        report = validator.validate(result)

        assert not report.is_valid
        assert report.has_errors

        # Should have empty extraction issue
        empty_issues = [
            i for i in report.issues if i.type == IssueType.EMPTY_EXTRACTION
        ]
        assert len(empty_issues) > 0
        assert empty_issues[0].severity == Severity.CRITICAL

    def test_validate_no_source_grounding(self):
        """Test validation without source grounding."""
        validator = ExtractionValidator(ValidationTestSchema)

        result = ExtractionResult(
            entities={"name": "John", "age": 30},
            sources={},  # No source grounding
            confidence=0.8,
        )

        report = validator.validate(result)

        # Should note lack of source grounding
        no_source_issues = [
            i for i in report.issues if i.type == IssueType.NO_SOURCE_GROUNDING
        ]
        assert len(no_source_issues) > 0
        assert no_source_issues[0].severity == Severity.INFO

    def test_validate_weak_source_grounding(self):
        """Test validation with weak source grounding."""
        validator = ExtractionValidator(ValidationTestSchema)

        result = ExtractionResult(
            entities={"name": "John", "age": 30, "location": "NYC"},
            sources={
                "name": [SourceSpan(start=0, end=4, text="John")]
                # Missing sources for age and location
            },
            confidence=0.8,
        )

        report = validator.validate(result)

        # Should detect weak grounding
        weak_grounding_issues = [
            i for i in report.issues if i.type == IssueType.WEAK_SOURCE_GROUNDING
        ]
        assert len(weak_grounding_issues) > 0

    def test_validate_type_mismatch(self):
        """Test validation with type mismatches."""
        validator = ExtractionValidator(ValidationTestSchema)

        result = ExtractionResult(
            entities={"name": "John", "age": "thirty"},  # Should be int, not string
            sources={},
            confidence=0.8,
        )

        report = validator.validate(result)

        # Should detect type mismatch
        type_issues = [i for i in report.issues if i.type == IssueType.TYPE_MISMATCH]
        assert len(type_issues) > 0
        assert "age" in type_issues[0].message

    def test_validate_with_text_short(self):
        """Test validation with very short text."""
        validator = ExtractionValidator(ValidationTestSchema)

        text = "John, 30"  # Very short text
        result = ExtractionResult(
            entities={
                "name": "John",
                "age": 30,
                "location": "NYC",
                "email": "john@example.com",
            },
            sources={},
            confidence=0.8,
        )

        report = validator.validate(result, text=text)

        # Should warn about many fields from short text
        inconsistent_issues = [
            i for i in report.issues if i.type == IssueType.INCONSISTENT_RESULTS
        ]
        assert len(inconsistent_issues) > 0
        assert "short text" in inconsistent_issues[0].message.lower()

    def test_validate_with_text_long(self):
        """Test validation with very long text but few results."""
        validator = ExtractionValidator(ValidationTestSchema)

        text = "This is a very long text " * 50  # Very long text
        result = ExtractionResult(
            entities={"name": "John"},  # Only one field extracted
            sources={},
            confidence=0.8,
        )

        report = validator.validate(result, text=text)

        # Should warn about few fields from long text
        inconsistent_issues = [
            i for i in report.issues if i.type == IssueType.INCONSISTENT_RESULTS
        ]
        assert len(inconsistent_issues) > 0
        assert "long text" in inconsistent_issues[0].message.lower()

    def test_validate_source_alignment(self):
        """Test validation of source span alignment."""
        validator = ExtractionValidator(ValidationTestSchema)

        text = "John Doe is 30 years old"

        # Test with invalid source span (out of bounds)
        result = ExtractionResult(
            entities={"name": "John Doe"},
            sources={
                "name": [
                    SourceSpan(start=0, end=100, text="John Doe")
                ]  # End beyond text
            },
            confidence=0.8,
        )

        report = validator.validate(result, text=text)

        # Should detect out of bounds span
        grounding_issues = [
            i for i in report.issues if i.type == IssueType.WEAK_SOURCE_GROUNDING
        ]
        assert len(grounding_issues) > 0
        assert "out of bounds" in grounding_issues[0].message

    def test_validate_source_text_mismatch(self):
        """Test validation with mismatched source text."""
        validator = ExtractionValidator(ValidationTestSchema)

        text = "John Doe is 30 years old"
        result = ExtractionResult(
            entities={"name": "John Doe"},
            sources={
                "name": [
                    SourceSpan(start=0, end=8, text="Jane Doe")
                ]  # Wrong text in span
            },
            confidence=0.8,
        )

        report = validator.validate(result, text=text)

        # Should detect text mismatch
        grounding_issues = [
            i for i in report.issues if i.type == IssueType.WEAK_SOURCE_GROUNDING
        ]
        assert len(grounding_issues) > 0
        assert "mismatch" in grounding_issues[0].message

    def test_validate_result_consistency(self):
        """Test validation with previous results for consistency."""
        validator = ExtractionValidator(ValidationTestSchema)

        # Previous results with high confidence
        previous_results = [
            ExtractionResult(entities={}, sources={}, confidence=0.9),
            ExtractionResult(entities={}, sources={}, confidence=0.85),
            ExtractionResult(entities={}, sources={}, confidence=0.88),
        ]

        # Current result with very different confidence
        current_result = ExtractionResult(
            entities={"name": "John"},
            sources={},
            confidence=0.3,  # Much lower than average
        )

        report = validator.validate(current_result, previous_results=previous_results)

        # Should note confidence inconsistency
        inconsistent_issues = [
            i for i in report.issues if i.type == IssueType.INCONSISTENT_RESULTS
        ]
        assert len(inconsistent_issues) > 0
        assert "confidence" in inconsistent_issues[0].message.lower()

    def test_schema_design_checks(self):
        """Test schema design validation."""

        # Create schema with poor descriptions
        class PoorSchema(BaseModel):
            name: str = Field(description="Name")  # Too short
            age: int = Field(description="Age")  # Too short
            field1: str = Field(description="X")  # Very short
            field2: str = Field(description="Y")
            field3: str = Field(description="Z")
            field4: str = Field(description="A")
            field5: str = Field(description="B")
            field6: str = Field(description="C")
            field7: str = Field(description="D")
            field8: str = Field(description="E")
            field9: str = Field(description="F")
            field10: str = Field(description="G")
            field11: str = Field(description="H")  # 11+ fields total

        validator = ExtractionValidator(PoorSchema)

        result = ExtractionResult(entities={"name": "John"}, sources={}, confidence=0.8)

        report = validator.validate(result)

        # Should detect schema design issues
        design_issues = [i for i in report.issues if i.type == IssueType.SCHEMA_DESIGN]
        assert len(design_issues) > 0

        # Should warn about short descriptions and too many fields
        short_desc_found = any(
            "short descriptions" in issue.message for issue in design_issues
        )
        too_many_fields_found = any(
            "fields" in issue.message for issue in design_issues
        )

        assert short_desc_found or too_many_fields_found

    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        validator = ExtractionValidator(ValidationTestSchema)

        # Test perfect result
        perfect_result = ExtractionResult(
            entities={
                "name": "John",
                "age": 30,
                "location": "NYC",
                "email": "john@example.com",
            },
            sources={
                "name": [SourceSpan(start=0, end=4, text="John")],
                "age": [SourceSpan(start=10, end=12, text="30")],
            },
            confidence=0.95,
        )

        perfect_report = validator.validate(perfect_result)
        assert perfect_report.score >= 0.9

        # Test poor result
        poor_result = ExtractionResult(entities={}, sources={}, confidence=0.1)  # Empty

        poor_report = validator.validate(poor_result)
        assert poor_report.score <= 0.3

    def test_suggestions_generation(self):
        """Test suggestion generation."""
        validator = ExtractionValidator(ValidationTestSchema)

        result = ExtractionResult(
            entities={"name": "John"}, sources={}, confidence=0.4  # Low confidence
        )

        report = validator.validate(result)

        # Should have suggestions
        assert len(report.suggestions) > 0

        # Should contain actionable suggestions
        suggestions_text = " ".join(report.suggestions).lower()
        assert any(
            keyword in suggestions_text
            for keyword in ["model", "description", "optimization", "field"]
        )

    def test_summary_generation(self):
        """Test summary generation."""
        validator = ExtractionValidator(ValidationTestSchema)

        # Good result
        good_result = ExtractionResult(
            entities={"name": "John", "age": 30}, sources={}, confidence=0.9
        )

        good_report = validator.validate(good_result)
        assert (
            "excellent" in good_report.summary.lower()
            or "good" in good_report.summary.lower()
        )

        # Poor result
        poor_result = ExtractionResult(entities={}, sources={}, confidence=0.1)

        poor_report = validator.validate(poor_result)
        assert (
            "poor" in poor_report.summary.lower()
            or "critical" in poor_report.summary.lower()
        )


class TestValidationReport:
    """Tests for ValidationReport model."""

    def test_validation_report_creation(self):
        """Test creating ValidationReport."""
        issues = [
            ValidationIssue(
                type=IssueType.LOW_CONFIDENCE,
                severity=Severity.WARNING,
                message="Test warning",
            )
        ]

        report = ValidationReport(
            is_valid=True,
            score=0.8,
            issues=issues,
            suggestions=["Test suggestion"],
            summary="Test summary",
        )

        assert report.is_valid
        assert report.score == 0.8
        assert len(report.issues) == 1
        assert report.suggestions == ["Test suggestion"]
        assert report.summary == "Test summary"

    def test_has_errors_property(self):
        """Test has_errors property."""
        # Report with no errors
        no_error_report = ValidationReport(
            is_valid=True,
            score=0.8,
            issues=[
                ValidationIssue(
                    type=IssueType.LOW_CONFIDENCE,
                    severity=Severity.INFO,
                    message="Info message",
                )
            ],
            suggestions=[],
            summary="Summary",
        )
        assert not no_error_report.has_errors

        # Report with errors
        error_report = ValidationReport(
            is_valid=False,
            score=0.3,
            issues=[
                ValidationIssue(
                    type=IssueType.EMPTY_EXTRACTION,
                    severity=Severity.CRITICAL,
                    message="Critical error",
                )
            ],
            suggestions=[],
            summary="Summary",
        )
        assert error_report.has_errors

    def test_has_warnings_property(self):
        """Test has_warnings property."""
        # Report with warnings
        warning_report = ValidationReport(
            is_valid=True,
            score=0.7,
            issues=[
                ValidationIssue(
                    type=IssueType.MISSING_FIELDS,
                    severity=Severity.WARNING,
                    message="Warning message",
                )
            ],
            suggestions=[],
            summary="Summary",
        )
        assert warning_report.has_warnings

        # Report without warnings
        no_warning_report = ValidationReport(
            is_valid=True, score=0.9, issues=[], suggestions=[], summary="Summary"
        )
        assert not no_warning_report.has_warnings


class TestValidationIssue:
    """Tests for ValidationIssue model."""

    def test_validation_issue_creation(self):
        """Test creating ValidationIssue."""
        issue = ValidationIssue(
            type=IssueType.LOW_CONFIDENCE,
            severity=Severity.WARNING,
            message="Test message",
            field="test_field",
            suggestion="Test suggestion",
            fix_code="test_code",
        )

        assert issue.type == IssueType.LOW_CONFIDENCE
        assert issue.severity == Severity.WARNING
        assert issue.message == "Test message"
        assert issue.field == "test_field"
        assert issue.suggestion == "Test suggestion"
        assert issue.fix_code == "test_code"

    def test_validation_issue_minimal(self):
        """Test creating ValidationIssue with minimal fields."""
        issue = ValidationIssue(
            type=IssueType.SCHEMA_DESIGN,
            severity=Severity.INFO,
            message="Minimal issue",
        )

        assert issue.type == IssueType.SCHEMA_DESIGN
        assert issue.severity == Severity.INFO
        assert issue.message == "Minimal issue"
        assert issue.field is None
        assert issue.suggestion is None
        assert issue.fix_code is None
