"""Tests for refinement feedback flow - verifying judge feedback is passed to refiner."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from langstruct.core.refinement import (
    Budget,
    BuiltinJudge,
    CandidateResult,
    CustomJudge,
    ExtractionRefiner,
    Refine,
    RefinementEngine,
    RefinementStrategy,
    RefinementTrace,
)
from langstruct.core.schemas import ExtractionResult, SourceSpan
from langstruct.core.signatures import JudgeScoreItem, JudgeScores


class PersonTestSchema(BaseModel):
    """Test schema for refinement tests."""

    name: str = Field(description="Name of the person")
    age: int = Field(description="Age in years")


@pytest.fixture
def sample_extraction():
    """Sample extraction result for testing."""
    return ExtractionResult(
        entities={"name": "John Doe", "age": 30},
        sources={
            "name": [SourceSpan(start=0, end=8, text="John Doe")],
            "age": [SourceSpan(start=15, end=17, text="30")],
        },
        confidence=0.85,
        metadata={},
    )


@pytest.fixture
def sample_candidates(sample_extraction):
    """Multiple candidate extractions for testing."""
    candidate1 = sample_extraction
    candidate2 = ExtractionResult(
        entities={"name": "Jane Doe", "age": 25},
        sources={
            "name": [SourceSpan(start=0, end=8, text="Jane Doe")],
            "age": [SourceSpan(start=15, end=17, text="25")],
        },
        confidence=0.75,
        metadata={},
    )
    return [candidate1, candidate2]


class TestFindingsNormalization:
    """Test that the findings field validator normalizes LLM output variations."""

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            ("NO_ISSUES", "NO_ISSUES"),
            ("ISSUES", "ISSUES"),
            ("no_issues", "NO_ISSUES"),
            ("No Issues", "NO_ISSUES"),
            ("no-issues", "NO_ISSUES"),
            ("No_Issues", "NO_ISSUES"),
            ("  NO_ISSUES  ", "NO_ISSUES"),
            ("noissues", "NO_ISSUES"),
            ("none", "NO_ISSUES"),
            ("FINISHED_GENERATION", "NO_ISSUES"),
            ("finished_generation", "NO_ISSUES"),
            ("FINISHED", "NO_ISSUES"),
            ("COMPLETE", "NO_ISSUES"),
            ("COMPLETED", "NO_ISSUES"),
            ("issues", "ISSUES"),
            ("Issues", "ISSUES"),
            ("some problems found", "ISSUES"),
        ],
    )
    def test_findings_normalizes_variants(self, raw_value, expected):
        """JudgeScoreItem should normalize common LLM output variations."""
        item = JudgeScoreItem(
            score=0.9,
            reasoning="Test",
            feedback="Test",
            findings=raw_value,
        )
        assert item.findings == expected


class TestCandidateResultFeedback:
    """Test that CandidateResult properly stores feedback."""

    def test_candidate_result_stores_feedback(self, sample_extraction):
        """CandidateResult should store a feedback field."""
        # Should work with feedback
        result = CandidateResult(
            extraction=sample_extraction,
            score=0.9,
            reasoning="Good extraction",
            feedback="Consider verifying the age from additional context",
            candidate_id=0,
        )
        assert result.feedback == "Consider verifying the age from additional context"

    def test_candidate_result_feedback_is_required(self, sample_extraction):
        """CandidateResult should fail without feedback."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            CandidateResult(
                extraction=sample_extraction,
                score=0.9,
                reasoning="Good extraction",
                # Missing feedback
                candidate_id=0,
            )


class TestJudgeFeedbackReturn:
    """Test that judges return feedback in their tuples."""

    def test_builtin_judge_returns_feedback_tuple(self, sample_candidates):
        """BuiltinJudge should return (score, reasoning, feedback, findings) tuples."""
        judge = BuiltinJudge(PersonTestSchema)

        # Mock the DSPy judge call with structured JudgeScores output
        mock_judge_scores = JudgeScores(
            scores=[
                JudgeScoreItem(
                    score=0.9,
                    reasoning="Good extraction with accurate values",
                    feedback="Consider adding source span for middle name",
                    findings="ISSUES",
                ),
                JudgeScoreItem(
                    score=0.7,
                    reasoning="Missing some details",
                    feedback="Verify age against document header",
                    findings="ISSUES",
                ),
            ]
        )

        with patch.object(judge, "judge") as mock_judge:
            mock_result = MagicMock()
            mock_result.scores = mock_judge_scores
            mock_judge.return_value = mock_result

            scores = judge.forward(
                "Sample text about John Doe, age 30", sample_candidates
            )

            assert len(scores) == 2
            # Each score should be a 4-tuple: (score, reasoning, feedback, findings)
            assert len(scores[0]) == 4
            assert len(scores[1]) == 4

            # Verify feedback is extracted
            assert scores[0][2] == "Consider adding source span for middle name"
            assert scores[1][2] == "Verify age against document header"
            # Verify findings
            assert scores[0][3] == "ISSUES"
            assert scores[1][3] == "ISSUES"

    def test_custom_judge_returns_feedback_tuple(self, sample_candidates):
        """CustomJudge should return (score, reasoning, feedback, findings) tuples."""
        custom_rubric = "Focus on name accuracy"
        judge = CustomJudge(PersonTestSchema, custom_rubric)

        mock_judge_scores = JudgeScores(
            scores=[
                JudgeScoreItem(
                    score=0.85,
                    reasoning="Name matches well",
                    feedback="Double-check spelling of last name",
                    findings="ISSUES",
                ),
                JudgeScoreItem(
                    score=0.65,
                    reasoning="Name partially correct",
                    feedback="Extract full legal name from signature block",
                    findings="ISSUES",
                ),
            ]
        )

        with patch.object(judge, "judge") as mock_judge:
            mock_result = MagicMock()
            mock_result.scores = mock_judge_scores
            mock_judge.return_value = mock_result

            scores = judge.forward("Sample text", sample_candidates)

            assert len(scores) == 2
            assert len(scores[0]) == 4
            assert scores[0][2] == "Double-check spelling of last name"
            assert scores[1][2] == "Extract full legal name from signature block"

    def test_judge_fallback_provides_default_feedback(self, sample_candidates):
        """Judge should provide default feedback when call fails."""
        judge = BuiltinJudge(PersonTestSchema)

        with patch.object(judge, "judge") as mock_judge:
            mock_judge.side_effect = Exception("LLM call failed")

            scores = judge.forward("Sample text", sample_candidates)

            assert len(scores) == 2
            # Fallback should include feedback and default ISSUES findings
            assert scores[0][2] == "No feedback available"
            assert scores[1][2] == "No feedback available"
            assert scores[0][3] == "ISSUES"
            assert scores[1][3] == "ISSUES"


class TestRefinementEngineFeedbackFlow:
    """Test that RefinementEngine passes feedback correctly."""

    def test_feedback_stored_in_trace_candidates(self, sample_candidates):
        """Feedback should be stored in trace.candidates."""
        # Create a mock extractor
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Mock the judge to return scores with feedback and findings
        mock_scores = [
            (0.9, "Good extraction", "Improve source span accuracy", "ISSUES"),
            (0.7, "Partial extraction", "Add missing fields", "ISSUES"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                config = Refine(
                    strategy=RefinementStrategy.BON,
                    n_candidates=2,
                    max_refine_steps=1,  # Must be >= 1
                )

                result, trace = engine.forward("Sample text", config)

                # Verify feedback is stored in candidates
                assert len(trace.candidates) == 2
                assert trace.candidates[0].feedback == "Improve source span accuracy"
                assert trace.candidates[1].feedback == "Add missing fields"

    def test_feedback_passed_to_refiner(self, sample_candidates):
        """Best candidate's feedback should be passed to refiner as issues."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Mock scores - candidate 0 has higher score, with ISSUES findings
        mock_scores = [
            (0.9, "Good extraction", "Check name spelling in header", "ISSUES"),
            (0.7, "Partial extraction", "Add missing fields", "ISSUES"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                with patch.object(engine.refiner, "forward") as mock_refiner:
                    # Make refiner return a result
                    mock_refiner.return_value = sample_candidates[0]

                    config = Refine(
                        strategy=RefinementStrategy.BON_THEN_REFINE,
                        n_candidates=2,
                        max_refine_steps=1,
                    )

                    result, trace = engine.forward("Sample text", config)

                    # Verify refiner was called with feedback as issues
                    mock_refiner.assert_called_once()
                    call_args = mock_refiner.call_args
                    # issues should be the feedback from the best candidate (index 0)
                    assert (
                        call_args.kwargs.get("issues")
                        == "Check name spelling in header"
                    )

    def test_feedback_tracked_in_refine_diffs(self, sample_candidates):
        """Feedback used should be tracked in trace.refine_diffs."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Need 2+ candidates for judge to be called
        mock_scores = [
            (0.9, "Good", "Verify age from birth date", "ISSUES"),
            (0.7, "Partial", "Check name", "ISSUES"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                with patch.object(
                    engine.refiner, "forward", return_value=sample_candidates[0]
                ):
                    with patch.object(
                        engine.refiner,
                        "_detect_issues",
                        return_value="No issues detected",
                    ):
                        config = Refine(
                            strategy=RefinementStrategy.BON_THEN_REFINE,
                            n_candidates=2,  # Need 2+ for judge to be called
                            max_refine_steps=1,
                        )

                        result, trace = engine.forward("Sample text", config)

                        # Verify feedback_used is tracked
                        assert len(trace.refine_diffs) == 1
                        assert (
                            trace.refine_diffs[0]["feedback_used"]
                            == "Verify age from birth date"
                        )

    def test_single_candidate_refine_strategy_runs_judge(self, sample_candidates):
        """Single candidate with REFINE strategy should still run the judge."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Judge returns ISSUES for the single candidate
        mock_scores = [
            (0.8, "Good but could improve", "Check age field", "ISSUES"),
        ]

        config = Refine(
            strategy=RefinementStrategy.REFINE,  # No BON, just refine
            n_candidates=1,
            max_refine_steps=1,
        )

        with patch.object(engine.builtin_judge, "forward", return_value=mock_scores):
            with patch.object(
                engine.refiner, "forward", return_value=sample_candidates[0]
            ):
                result, trace = engine.forward("Sample text", config)

                # Judge should have been called even for single candidate
                assert len(trace.candidates) == 1
                assert trace.candidates[0].feedback == "Check age field"
                assert trace.candidates[0].findings == "ISSUES"


class TestExtractionRefinerReceivesFeedback:
    """Test that ExtractionRefiner properly uses the issues/feedback parameter."""

    def test_refiner_uses_issues_parameter(self, sample_extraction):
        """Refiner should use the issues parameter when provided."""
        refiner = ExtractionRefiner(PersonTestSchema)

        feedback = "The age value seems incorrect, verify against document"

        with patch.object(refiner, "refine") as mock_refine:
            mock_result = MagicMock()
            mock_result.refined_extraction = json.dumps({"name": "John Doe", "age": 31})
            mock_refine.return_value = mock_result

            result = refiner.forward("Sample text", sample_extraction, issues=feedback)

            # Verify the refine signature was called with the feedback as issues
            mock_refine.assert_called_once()
            call_kwargs = mock_refine.call_args.kwargs
            assert call_kwargs.get("issues") == feedback

    def test_refiner_auto_detects_issues_when_none_provided(self, sample_extraction):
        """Refiner should auto-detect issues when none provided."""
        refiner = ExtractionRefiner(PersonTestSchema)

        # Create extraction with potential issue (empty value)
        extraction_with_issue = ExtractionResult(
            entities={"name": "", "age": 30},  # Empty name
            sources={},
            confidence=0.5,
            metadata={},
        )

        detected = refiner._detect_issues(
            "Sample text about John", extraction_with_issue
        )

        # Should detect the missing name
        assert "name" in detected.lower() or "missing" in detected.lower()


class TestParseRetryWithFeedback:
    """Test that JSON parse failures are retried with error feedback."""

    def test_refiner_retries_on_json_parse_failure(self, sample_extraction):
        """Refiner should retry with error feedback when JSON parsing fails."""
        refiner = ExtractionRefiner(PersonTestSchema)

        call_count = 0

        def mock_refine_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                # First call returns invalid JSON
                mock_result.refined_extraction = "not valid json {{"
            else:
                # Second call returns valid JSON
                mock_result.refined_extraction = json.dumps(
                    {"name": "John Doe", "age": 31}
                )
            return mock_result

        with patch.object(refiner, "refine", side_effect=mock_refine_side_effect):
            result = refiner.forward("Sample text", sample_extraction, issues="Fix age")

            # Should have retried and succeeded
            assert call_count == 2
            assert result.entities == {"name": "John Doe", "age": 31}
            assert result.metadata.get("parse_retries") == 1

    def test_refiner_passes_error_in_issues_on_retry(self, sample_extraction):
        """Refiner should include parse error in issues for retry attempts."""
        refiner = ExtractionRefiner(PersonTestSchema)

        captured_issues = []

        def mock_refine_side_effect(**kwargs):
            captured_issues.append(kwargs.get("issues", ""))
            mock_result = MagicMock()
            if len(captured_issues) == 1:
                mock_result.refined_extraction = "bad json"
            else:
                mock_result.refined_extraction = json.dumps({"name": "John", "age": 30})
            return mock_result

        with patch.object(refiner, "refine", side_effect=mock_refine_side_effect):
            refiner.forward("Sample text", sample_extraction, issues="Fix age")

            # Second call should include error feedback
            assert len(captured_issues) == 2
            assert "PREVIOUS ATTEMPT FAILED" in captured_issues[1]
            assert "JSON parsing failed" in captured_issues[1]

    def test_refiner_returns_original_after_max_retries(self, sample_extraction):
        """Refiner should return original extraction after exhausting retries."""
        refiner = ExtractionRefiner(PersonTestSchema)

        with patch.object(refiner, "refine") as mock_refine:
            mock_result = MagicMock()
            mock_result.refined_extraction = "always bad json"
            mock_refine.return_value = mock_result

            with pytest.warns(UserWarning, match="failed to parse after 3 retries"):
                result = refiner.forward(
                    "Sample text", sample_extraction, issues="Fix age"
                )

            # Should return original after 3 retries
            assert mock_refine.call_count == 3
            assert result == sample_extraction

    def test_refiner_no_retry_metadata_on_first_success(self, sample_extraction):
        """No parse_retries metadata when first attempt succeeds."""
        refiner = ExtractionRefiner(PersonTestSchema)

        with patch.object(refiner, "refine") as mock_refine:
            mock_result = MagicMock()
            mock_result.refined_extraction = json.dumps({"name": "John", "age": 30})
            mock_refine.return_value = mock_result

            result = refiner.forward("Sample text", sample_extraction, issues="Fix age")

            assert mock_refine.call_count == 1
            assert "parse_retries" not in result.metadata

    def test_builtin_judge_fallback_on_exception(self, sample_candidates):
        """BuiltinJudge should return fallback scores when judge call raises."""
        judge = BuiltinJudge(PersonTestSchema)

        with patch.object(judge, "judge") as mock_judge:
            mock_judge.side_effect = Exception("LLM error")

            scores = judge.forward("Sample text", sample_candidates)

            assert len(scores) == 2
            assert scores[0][0] == 0.5
            assert scores[0][3] == "ISSUES"

    def test_custom_judge_fallback_on_exception(self, sample_candidates):
        """CustomJudge should return fallback scores when judge call raises."""
        judge = CustomJudge(PersonTestSchema, "Custom rubric")

        with patch.object(judge, "judge") as mock_judge:
            mock_judge.side_effect = Exception("LLM error")

            scores = judge.forward("Sample text", sample_candidates)

            assert len(scores) == 2
            assert scores[0][0] == 0.5
            assert scores[0][3] == "ISSUES"


class TestFindingsSkipRefinement:
    """Test that NO_ISSUES findings skip refinement and ISSUES proceeds."""

    def test_no_issues_skips_refinement(self, sample_candidates):
        """When judge returns NO_ISSUES, refiner should NOT be called."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Judge says NO_ISSUES for the best candidate
        mock_scores = [
            (0.95, "Perfect extraction", "No improvements needed", "NO_ISSUES"),
            (0.7, "Partial extraction", "Add missing fields", "ISSUES"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                with patch.object(engine.refiner, "forward") as mock_refiner:
                    config = Refine(
                        strategy=RefinementStrategy.BON_THEN_REFINE,
                        n_candidates=2,
                        max_refine_steps=1,
                    )

                    result, trace = engine.forward("Sample text", config)

                    # Refiner should NOT have been called
                    mock_refiner.assert_not_called()
                    # Should return the best candidate directly
                    assert result == sample_candidates[0]
                    # No refine diffs should exist
                    assert len(trace.refine_diffs) == 0

    def test_issues_proceeds_with_refinement(self, sample_candidates):
        """When judge returns ISSUES, refiner should be called."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Judge says ISSUES for the best candidate
        mock_scores = [
            (0.9, "Good extraction", "Fix the age field", "ISSUES"),
            (0.7, "Partial extraction", "Add missing fields", "ISSUES"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                with patch.object(engine.refiner, "forward") as mock_refiner:
                    mock_refiner.return_value = sample_candidates[0]

                    config = Refine(
                        strategy=RefinementStrategy.BON_THEN_REFINE,
                        n_candidates=2,
                        max_refine_steps=1,
                    )

                    result, trace = engine.forward("Sample text", config)

                    # Refiner SHOULD have been called
                    mock_refiner.assert_called_once()

    def test_findings_stored_in_trace(self, sample_candidates):
        """Findings should be stored in trace.candidates."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        mock_scores = [
            (0.95, "Perfect", "No issues", "NO_ISSUES"),
            (0.7, "Partial", "Fix name", "ISSUES"),
        ]

        with patch.object(
            engine, "_generate_candidates", return_value=sample_candidates
        ):
            with patch.object(
                engine.builtin_judge, "forward", return_value=mock_scores
            ):
                config = Refine(
                    strategy=RefinementStrategy.BON,
                    n_candidates=2,
                    max_refine_steps=1,
                )

                result, trace = engine.forward("Sample text", config)

                assert len(trace.candidates) == 2
                assert trace.candidates[0].findings == "NO_ISSUES"
                assert trace.candidates[1].findings == "ISSUES"

    def test_single_candidate_no_issues_skips_refinement(self, sample_candidates):
        """Single candidate with NO_ISSUES should skip refinement."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Judge returns NO_ISSUES for the single candidate
        mock_scores = [
            (0.95, "Perfect extraction", "No improvements needed", "NO_ISSUES"),
        ]

        config = Refine(
            strategy=RefinementStrategy.REFINE,
            n_candidates=1,
            max_refine_steps=1,
        )

        with patch.object(engine.builtin_judge, "forward", return_value=mock_scores):
            with patch.object(engine.refiner, "forward") as mock_refiner:
                result, trace = engine.forward("Sample text", config)

                # Refiner should NOT have been called
                mock_refiner.assert_not_called()
                assert len(trace.candidates) == 1
                assert trace.candidates[0].findings == "NO_ISSUES"
                assert len(trace.refine_diffs) == 0


class TestJudgeAlwaysRunsBeforeRefinement:
    """Test that the judge runs for all strategies that include refinement,
    even with a single candidate, so NO_ISSUES can gate the refiner."""

    def test_refine_strategy_single_candidate_runs_judge(self, sample_candidates):
        """REFINE strategy with 1 candidate should still call the judge."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        mock_scores = [
            (0.85, "Good extraction", "Minor improvements", "ISSUES"),
        ]

        config = Refine(
            strategy=RefinementStrategy.REFINE,
            n_candidates=1,
            max_refine_steps=1,
        )

        with patch.object(
            engine.builtin_judge, "forward", return_value=mock_scores
        ) as mock_judge_forward:
            with patch.object(
                engine.refiner, "forward", return_value=sample_candidates[0]
            ):
                result, trace = engine.forward("Sample text", config)

                # Judge MUST have been called
                mock_judge_forward.assert_called_once()
                assert trace.candidates[0].score == 0.85
                assert trace.candidates[0].feedback == "Minor improvements"

    def test_bon_then_refine_single_candidate_runs_judge(self, sample_candidates):
        """BON_THEN_REFINE with 1 candidate should still call the judge."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        mock_scores = [
            (0.9, "Solid extraction", "Check dates", "ISSUES"),
        ]

        config = Refine(
            strategy=RefinementStrategy.BON_THEN_REFINE,
            n_candidates=1,
            max_refine_steps=1,
        )

        with patch.object(
            engine.builtin_judge, "forward", return_value=mock_scores
        ) as mock_judge_forward:
            with patch.object(
                engine.refiner, "forward", return_value=sample_candidates[0]
            ):
                result, trace = engine.forward("Sample text", config)

                mock_judge_forward.assert_called_once()

    def test_bon_only_single_candidate_skips_judge(self, sample_candidates):
        """BON strategy with 1 candidate should NOT call the judge (no refinement to gate)."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        config = Refine(
            strategy=RefinementStrategy.BON,
            n_candidates=1,
            max_refine_steps=1,
        )

        with patch.object(engine.builtin_judge, "forward") as mock_judge_forward:
            result, trace = engine.forward("Sample text", config)

            # Judge should NOT be called — BON-only with 1 candidate has no use for it
            mock_judge_forward.assert_not_called()
            assert trace.candidates[0].reasoning == "Single candidate (no judge)"

    def test_single_candidate_judge_no_issues_saves_refiner_call(
        self, sample_candidates
    ):
        """When judge says NO_ISSUES for a single candidate, refiner is never invoked."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        mock_scores = [
            (0.98, "Perfect — no changes needed", "None", "NO_ISSUES"),
        ]

        config = Refine(
            strategy=RefinementStrategy.REFINE,
            n_candidates=1,
            max_refine_steps=3,  # Would do 3 steps if ISSUES
        )

        with patch.object(engine.builtin_judge, "forward", return_value=mock_scores):
            with patch.object(engine.refiner, "forward") as mock_refiner:
                result, trace = engine.forward("Sample text", config)

                mock_refiner.assert_not_called()
                assert result == sample_candidates[0]
                assert trace.candidates[0].findings == "NO_ISSUES"
                # Budget should reflect judge call but no refiner calls
                assert trace.budget_used["calls"] == 2  # 1 extract + 1 judge

    def test_single_candidate_judge_issues_proceeds_to_refine(self, sample_candidates):
        """When judge says ISSUES for a single candidate, refiner should be called."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        mock_scores = [
            (0.7, "Missing fields", "Add the age field", "ISSUES"),
        ]

        config = Refine(
            strategy=RefinementStrategy.REFINE,
            n_candidates=1,
            max_refine_steps=1,
        )

        with patch.object(engine.builtin_judge, "forward", return_value=mock_scores):
            with patch.object(engine.refiner, "forward") as mock_refiner:
                mock_refiner.return_value = sample_candidates[0]

                result, trace = engine.forward("Sample text", config)

                mock_refiner.assert_called_once()
                # Refiner should receive the judge's feedback
                call_args = mock_refiner.call_args
                assert call_args.kwargs.get("issues") == "Add the age field"

    def test_judge_fallback_defaults_to_issues_proceeds_to_refine(
        self, sample_candidates
    ):
        """When judge call fails (fallback), findings default to ISSUES and refiner runs."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        config = Refine(
            strategy=RefinementStrategy.REFINE,
            n_candidates=1,
            max_refine_steps=1,
        )

        # Make the judge's inner DSPy call fail — fallback returns ISSUES
        with patch.object(engine.builtin_judge, "judge") as mock_dspy_judge:
            mock_dspy_judge.side_effect = Exception("LLM error")

            with patch.object(engine.refiner, "forward") as mock_refiner:
                mock_refiner.return_value = sample_candidates[0]

                result, trace = engine.forward("Sample text", config)

                # Fallback should have given ISSUES, so refiner runs
                mock_refiner.assert_called_once()
                assert trace.candidates[0].findings == "ISSUES"

    def test_empty_extraction_no_issues_skips_refinement(self, sample_candidates):
        """Empty extraction judged as NO_ISSUES should skip refinement."""
        empty_extraction = ExtractionResult(
            entities={"metrics": []},
            sources={},
            confidence=0.9,
            metadata={},
        )
        mock_extractor = MagicMock()
        mock_extractor.return_value = empty_extraction

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        # Judge recognises the empty result is correct per schema exclusions
        mock_scores = [
            (
                1.0,
                "Empty extraction is correct — data doesn't match schema requirements",
                "No improvements needed",
                "NO_ISSUES",
            ),
        ]

        config = Refine(
            strategy=RefinementStrategy.REFINE,
            n_candidates=1,
            max_refine_steps=1,
        )

        with patch.object(engine.builtin_judge, "forward", return_value=mock_scores):
            with patch.object(engine.refiner, "forward") as mock_refiner:
                result, trace = engine.forward("Sample text", config)

                mock_refiner.assert_not_called()
                assert result.entities == {"metrics": []}
                assert trace.candidates[0].findings == "NO_ISSUES"

    def test_custom_judge_no_issues_skips_refinement(self, sample_candidates):
        """Custom judge returning NO_ISSUES should also skip refinement."""
        mock_extractor = MagicMock()
        mock_extractor.return_value = sample_candidates[0]

        engine = RefinementEngine(PersonTestSchema, mock_extractor)

        mock_scores = [
            (0.95, "Meets custom criteria", "No improvements", "NO_ISSUES"),
        ]

        config = Refine(
            strategy=RefinementStrategy.BON_THEN_REFINE,
            n_candidates=1,
            judge="Custom rubric: prefer exact dollar amounts",
            max_refine_steps=2,
        )

        # Need to patch the _get_judge path for custom rubric
        with patch.object(engine, "_get_judge") as mock_get_judge:
            mock_custom = MagicMock()
            mock_custom.return_value = mock_scores
            mock_get_judge.return_value = mock_custom

            with patch.object(engine.refiner, "forward") as mock_refiner:
                result, trace = engine.forward("Sample text", config)

                mock_refiner.assert_not_called()
                assert trace.candidates[0].findings == "NO_ISSUES"
