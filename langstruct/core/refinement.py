"""DSPy refinement system for improving extraction quality through Best-of-N, refinement, and committee scoring."""

import concurrent.futures
import json
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import dspy
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from .constants import MAX_PARSE_RETRIES
from .schema_utils import get_field_descriptions, get_json_schema
from .schemas import ExtractionResult, SourceSpan
from .signatures import (
    ExtractEntities,
    ExtractWithSources,
    JudgeExtractions,
    JudgeScoreItem,
    JudgeScores,
    RefineExtraction,
)


class RefinementStrategy(str, Enum):
    """Available refinement strategies.

    Each strategy offers different trade-offs between accuracy, cost, and speed:

    - BON: Fastest, moderate cost, good accuracy improvement
    - REFINE: Medium speed/cost, targets specific issues
    - BON_THEN_REFINE: Highest accuracy, highest cost, slowest
    """

    BON = "bon"  # Best-of-N candidate selection only
    REFINE = "refine"  # Iterative refinement only
    BON_THEN_REFINE = (
        "bon_then_refine"  # Best-of-N followed by refinement (recommended)
    )


@dataclass
class Budget:
    """Budget controls for refinement operations.

    Prevents runaway costs by limiting LLM API usage. When limits are exceeded,
    refinement falls back gracefully to the best candidate found so far.

    Args:
        max_calls: Maximum number of LLM API calls allowed (default: unlimited)
        max_tokens: Maximum number of tokens to consume (default: unlimited)

    Examples:
        Conservative budget:
        >>> budget = Budget(max_calls=5, max_tokens=5000)

        Production budget:
        >>> budget = Budget(max_calls=10, max_tokens=15000)

        Per-document budget for batching:
        >>> budget = Budget(max_calls=3)  # 3 calls max per document
    """

    max_calls: Optional[int] = None
    max_tokens: Optional[int] = None

    def check_exceeded(self, calls_used: int, tokens_used: int) -> bool:
        """Check if budget has been exceeded."""
        if self.max_calls and calls_used >= self.max_calls:
            return True
        if self.max_tokens and tokens_used >= self.max_tokens:
            return True
        return False


class Refine(BaseModel):
    """Configuration for DSPy refinement system.

    Refinement improves extraction accuracy by generating multiple candidates
    and using Best-of-N selection plus iterative improvement. Typically provides
    15-30% accuracy improvement at 2-5x cost increase.

    Args:
        strategy: Refinement approach - "bon" (Best-of-N only), "refine" (iterative only),
                 or "bon_then_refine" (combined, recommended)
        n_candidates: Number of extraction candidates to generate for Best-of-N (1-10)
        judge: Custom scoring rubric as plain English. If None, uses built-in rubric
               based on faithfulness, completeness, and source quality
        max_refine_steps: Maximum iterative improvement steps (1-5)
        temperature: Temperature for diverse candidate generation (0.0-2.0)
        budget: Budget limits to prevent runaway costs

    Examples:
        Basic refinement with defaults:
        >>> refine = Refine()
        >>> result = extractor.extract(text, refine=refine)

        Custom configuration:
        >>> refine = Refine(
        ...     strategy="bon_then_refine",
        ...     n_candidates=5,
        ...     judge="Prefer exact monetary amounts over rounded numbers",
        ...     budget=Budget(max_calls=10)
        ... )

        Method-level usage:
        >>> result = extractor.extract(text, refine={
        ...     "strategy": "bon",
        ...     "n_candidates": 3
        ... })
    """

    strategy: RefinementStrategy = Field(
        default=RefinementStrategy.BON_THEN_REFINE,
        description="Refinement strategy to use",
    )
    n_candidates: int = Field(
        default=3, ge=1, le=10, description="Number of candidates for Best-of-N"
    )
    judge: Optional[str] = Field(
        default=None, description="Custom judge rubric (uses built-in if None)"
    )
    max_refine_steps: int = Field(
        default=1, ge=1, le=5, description="Maximum refinement iterations"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for candidate generation"
    )
    budget: Optional[Budget] = Field(default=None, description="Budget constraints")

    def __bool__(self) -> bool:
        """Allow Refine() to be used in boolean context."""
        return True


class CandidateResult(BaseModel):
    """A single extraction candidate with scoring information."""

    extraction: ExtractionResult
    score: float = Field(ge=0.0, le=1.0, description="Judge score")
    reasoning: Optional[str] = Field(default=None, description="Judge reasoning")
    feedback: str = Field(
        description="Actionable feedback for improving the extraction"
    )
    findings: Literal["NO_ISSUES", "ISSUES"] = Field(
        default="ISSUES",
        description="NO_ISSUES if extraction is correct and complete, ISSUES if problems were found",
    )
    candidate_id: int = Field(description="Candidate identifier")


class RefinementTrace(BaseModel):
    """Trace information for refinement process."""

    candidates: List[CandidateResult] = Field(default_factory=list)
    chosen_candidate: Optional[int] = Field(default=None)
    refine_diffs: List[Dict[str, Any]] = Field(default_factory=list)
    budget_used: Dict[str, int] = Field(default_factory=dict)


class BuiltinJudge(dspy.Module):
    """Built-in judge that uses schema and source information for scoring."""

    def __init__(self, schema: Type[BaseModel]):
        super().__init__()
        self.schema = schema
        self.judge = dspy.ChainOfThought(JudgeExtractions)

    def forward(
        self, text: str, candidates: List[ExtractionResult]
    ) -> List[Tuple[float, str, str, str]]:
        """Score candidates using built-in rubric.

        Returns list of (score, reasoning, feedback, findings) tuples.
        """
        if not candidates:
            return []

        # Prepare candidates for judging
        candidates_json = json.dumps(
            [
                {
                    "candidate_id": i,
                    "entities": candidate.entities,
                    "sources": {
                        field: [
                            {"start": span.start, "end": span.end, "text": span.text}
                            for span in spans
                        ]
                        for field, spans in candidate.sources.items()
                    },
                    "confidence": candidate.confidence,
                }
                for i, candidate in enumerate(candidates)
            ],
            indent=2,
        )

        schema_json = json.dumps(get_json_schema(self.schema), indent=2)

        # Built-in rubric focuses on faithfulness and completeness
        built_in_rubric = (
            "Score candidates based on:\n"
            "1. Faithfulness: Extracted values must exactly match text in source spans\n"
            "2. Completeness: All required fields should be filled when data is available\n"
            "3. Source quality: Source spans should contain the complete extracted values\n"
            "4. No hallucination: Never extract values not present in the original text\n"
            "5. Schema compliance: Respect schema exclusions and constraints — an empty extraction "
            "is correct when the source data does not match schema requirements\n"
            "Prefer candidates that exactly quote from the text over those that paraphrase.\n\n"
            "For each candidate, provide specific actionable feedback on how to improve the extraction.\n"
            "Set findings to NO_ISSUES if the extraction is correct and complete "
            "(including correctly empty results when data doesn't match schema requirements), "
            "or ISSUES if there are actual problems that need fixing."
        )

        try:
            result = self.judge(
                text=text,
                candidates=candidates_json,
                schema_spec=schema_json,
                rubric=built_in_rubric,
            )

            # DSPy returns a typed JudgeScores instance
            judge_scores: JudgeScores = result.scores
            return [
                (
                    item.score,
                    item.reasoning,
                    item.feedback,
                    item.findings,
                )
                for item in judge_scores.scores
            ]
        except Exception as e:
            logger.warning("Judge call failed: %s", str(e))
            return [
                (0.5, "Judge call failed", "No feedback available", "ISSUES")
            ] * len(candidates)


class CustomJudge(dspy.Module):
    """Custom judge using user-provided rubric."""

    def __init__(self, schema: Type[BaseModel], rubric: str):
        super().__init__()
        self.schema = schema
        self.rubric = rubric
        self.judge = dspy.ChainOfThought(JudgeExtractions)

    def forward(
        self, text: str, candidates: List[ExtractionResult]
    ) -> List[Tuple[float, str, str, str]]:
        """Score candidates using custom rubric.

        Returns list of (score, reasoning, feedback, findings) tuples.
        """
        if not candidates:
            return []

        candidates_json = json.dumps(
            [
                {
                    "candidate_id": i,
                    "entities": candidate.entities,
                    "sources": {
                        field: [
                            {"start": span.start, "end": span.end, "text": span.text}
                            for span in spans
                        ]
                        for field, spans in candidate.sources.items()
                    },
                    "confidence": candidate.confidence,
                }
                for i, candidate in enumerate(candidates)
            ],
            indent=2,
        )

        schema_json = json.dumps(get_json_schema(self.schema), indent=2)

        # Append feedback instruction to custom rubric
        rubric_with_feedback = (
            f"{self.rubric}\n\n"
            "For each candidate, provide specific actionable feedback on how to improve the extraction.\n"
            "Set findings to NO_ISSUES if the extraction is correct and complete "
            "(including correctly empty results when data doesn't match schema requirements), "
            "or ISSUES if there are actual problems that need fixing."
        )

        try:
            result = self.judge(
                text=text,
                candidates=candidates_json,
                schema_spec=schema_json,
                rubric=rubric_with_feedback,
            )

            # DSPy returns a typed JudgeScores instance
            judge_scores: JudgeScores = result.scores
            return [
                (
                    item.score,
                    item.reasoning,
                    item.feedback,
                    item.findings,
                )
                for item in judge_scores.scores
            ]
        except Exception as e:
            logger.warning("Custom judge call failed: %s", str(e))
            return [
                (0.5, "Judge call failed", "No feedback available", "ISSUES")
            ] * len(candidates)


class ExtractionRefiner(dspy.Module):
    """Iterative refinement module for improving extractions."""

    def __init__(self, schema: Type[BaseModel]):
        super().__init__()
        self.schema = schema
        self.refine = dspy.ChainOfThought(RefineExtraction)

    def forward(
        self,
        text: str,
        current_extraction: ExtractionResult,
        issues: Optional[str] = None,
        max_parse_retries: int = MAX_PARSE_RETRIES,
    ) -> ExtractionResult:
        """Refine an extraction by addressing specific issues.

        Args:
            text: Original text
            current_extraction: Current extraction to refine
            issues: Specific issues to address (auto-detected if None)
            max_parse_retries: Max retries when LLM returns invalid JSON

        Returns:
            Refined extraction result
        """
        schema_json = json.dumps(get_json_schema(self.schema), indent=2)
        current_json = json.dumps(current_extraction.entities)

        # Auto-detect issues if not provided
        if issues is None:
            issues = self._detect_issues(text, current_extraction)

        current_issues = issues
        for attempt in range(max_parse_retries):
            result = self.refine(
                text=text,
                current_extraction=current_json,
                schema_spec=schema_json,
                issues=current_issues,
            )

            try:
                refined_entities = json.loads(result.refined_extraction)
            except json.JSONDecodeError as e:
                error_msg = (
                    f"JSON parsing failed: {e}\n\n"
                    f"Invalid output:\n{result.refined_extraction[:500]}"
                )
                current_issues = (
                    f"{issues}\n\n"
                    f"PREVIOUS ATTEMPT FAILED - You MUST fix this error:\n{error_msg}"
                )
                if attempt < max_parse_retries - 1:
                    logger.warning(
                        "Refinement parse failed, retry %d/%d: %s",
                        attempt + 1,
                        max_parse_retries,
                        str(e),
                    )
                    continue
                else:
                    warnings.warn(
                        f"Refinement failed to parse after {max_parse_retries} retries, "
                        "returning original"
                    )
                    return current_extraction

            # Only boost confidence if refinement actually changed entities
            entities_changed = refined_entities != current_extraction.entities
            new_confidence = (
                min(current_extraction.confidence + 0.1, 1.0)
                if entities_changed
                else current_extraction.confidence
            )

            # Create new extraction result with refined entities
            # Keep original sources for now (could be improved with source refinement)
            return ExtractionResult(
                entities=refined_entities,
                sources=current_extraction.sources,  # TODO: Could refine sources too
                confidence=new_confidence,
                metadata={
                    **current_extraction.metadata,
                    "refined": True,
                    "refinement_issues": issues,
                    **({"parse_retries": attempt} if attempt > 0 else {}),
                },
            )

        # Should not reach here, but just in case
        return current_extraction

    def _detect_issues(self, text: str, extraction: ExtractionResult) -> str:
        """Auto-detect issues in current extraction."""
        issues = []

        schema_fields = get_field_descriptions(self.schema)

        # Check for fields defined in the schema but completely missing from extraction
        for field_name in schema_fields:
            if field_name not in extraction.entities:
                issues.append(
                    f"Missing field '{field_name}' (expected by schema but not present in extraction)"
                )

        # Check for empty/null values in fields that are present
        for field_name, value in extraction.entities.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                if field_name in schema_fields:
                    issues.append(f"Missing value for required field: {field_name}")
            elif isinstance(value, (list, dict)) and not value:
                if field_name in schema_fields:
                    issues.append(f"Empty collection for required field: {field_name}")

        # Check source-value alignment
        for field_name, value in extraction.entities.items():
            if value and field_name in extraction.sources:
                spans = extraction.sources[field_name]
                if spans and not any(
                    str(value).lower() in span.text.lower() for span in spans
                ):
                    issues.append(
                        f"Value '{value}' for field '{field_name}' not found in source spans"
                    )

        return "; ".join(issues) if issues else "General quality improvement needed"


class RefinementEngine(dspy.Module):
    """Main refinement engine that orchestrates the entire process."""

    def __init__(self, schema: Type[BaseModel], extractor_module):
        super().__init__()
        self.schema = schema
        self.extractor = extractor_module  # The original extraction module
        self.builtin_judge = BuiltinJudge(schema)
        self.refiner = ExtractionRefiner(schema)

    def forward(
        self, text: str, refine_config: Refine, run_validation: bool = False
    ) -> Tuple[ExtractionResult, RefinementTrace]:
        """Run refinement process based on configuration.

        Args:
            text: Input text to extract from
            refine_config: Refinement configuration
            run_validation: Whether to run LLM validation inside EntityExtractor
                (default False to avoid wasted LLM calls during refinement)

        Returns:
            Tuple of (best_result, trace_info)
        """
        logger.info(
            "RefinementEngine.forward starting config=%s",
            {
                "strategy": refine_config.strategy.value,
                "n_candidates": refine_config.n_candidates,
                "max_refine_steps": refine_config.max_refine_steps,
                "temperature": refine_config.temperature,
                "has_custom_judge": refine_config.judge is not None,
                "budget": (
                    {
                        "max_calls": (
                            refine_config.budget.max_calls
                            if refine_config.budget
                            else None
                        ),
                        "max_tokens": (
                            refine_config.budget.max_tokens
                            if refine_config.budget
                            else None
                        ),
                    }
                    if refine_config.budget
                    else None
                ),
                "schema": getattr(self.schema, "__name__", str(self.schema)),
                "text_length": len(text),
            },
        )
        trace = RefinementTrace()
        calls_used = 0
        tokens_used = 0  # TODO: Implement token counting

        # Check budget before starting
        if refine_config.budget and refine_config.budget.check_exceeded(
            calls_used, tokens_used
        ):
            # Return basic extraction if budget exceeded
            basic_result = self.extractor(text, run_validation=run_validation)
            return basic_result, trace

        candidates = []

        # Step 1: Generate candidates (for BON strategies)
        if refine_config.strategy in [
            RefinementStrategy.BON,
            RefinementStrategy.BON_THEN_REFINE,
        ]:
            logger.info(
                "RefinementEngine generating candidates n_candidates=%d temperature=%.2f",
                refine_config.n_candidates,
                refine_config.temperature,
            )
            candidates = self._generate_candidates(
                text,
                refine_config.n_candidates,
                refine_config.temperature,
                run_validation=run_validation,
            )
            calls_used += len(candidates)
            logger.info(
                "RefinementEngine candidates generated count=%d calls_used=%d",
                len(candidates),
                calls_used,
            )

            # Check budget after candidate generation
            if refine_config.budget and refine_config.budget.check_exceeded(
                calls_used, tokens_used
            ):
                warnings.warn(
                    "Budget exceeded after candidate generation, using first candidate"
                )
                best_result = (
                    candidates[0]
                    if candidates
                    else self.extractor(text, run_validation=run_validation)
                )
                trace.budget_used = {"calls": calls_used, "tokens": tokens_used}
                return best_result, trace
        else:
            # For refine-only strategy, start with single extraction
            candidates = [self.extractor(text, run_validation=run_validation)]
            calls_used += 1

        # Step 2: Judge candidates and select best
        # Always run the judge when the strategy includes refinement (even for
        # a single candidate) so that NO_ISSUES findings can skip the expensive
        # refiner call.  For BON-only with a single candidate, skip judging
        # since there is nothing to compare and no refinement to gate.
        run_judge = len(candidates) > 1 or refine_config.strategy in [
            RefinementStrategy.REFINE,
            RefinementStrategy.BON_THEN_REFINE,
        ]

        if run_judge:
            logger.info(
                "RefinementEngine judging candidates count=%d custom_judge=%s",
                len(candidates),
                refine_config.judge is not None,
            )
            judge = self._get_judge(refine_config.judge)
            scores = judge(text, candidates)
            calls_used += 1  # Judge call
            logger.info(
                "RefinementEngine judging complete scores=%s findings=%s calls_used=%d",
                [round(s[0], 3) for s in scores],
                [s[3] for s in scores],
                calls_used,
            )

            # Create candidate results with scores
            for i, (candidate, (score, reasoning, feedback, findings)) in enumerate(
                zip(candidates, scores)
            ):
                trace.candidates.append(
                    CandidateResult(
                        extraction=candidate,
                        score=score,
                        reasoning=reasoning,
                        feedback=feedback,
                        findings=findings,
                        candidate_id=i,
                    )
                )

            # Select best candidate
            best_idx = max(range(len(scores)), key=lambda i: scores[i][0])
            best_candidate = candidates[best_idx]
            best_feedback = scores[best_idx][2]  # Get feedback for the best candidate
            best_findings = scores[best_idx][3]  # Get findings for the best candidate
            trace.chosen_candidate = best_idx
        else:
            best_candidate = candidates[0]
            best_feedback = "General quality improvement needed"
            best_findings = "ISSUES"
            trace.candidates.append(
                CandidateResult(
                    extraction=best_candidate,
                    score=best_candidate.confidence,
                    reasoning="Single candidate (no judge)",
                    feedback=best_feedback,
                    findings=best_findings,
                    candidate_id=0,
                )
            )
            trace.chosen_candidate = 0

        # Step 3: Refinement (for refine strategies)
        if refine_config.strategy in [
            RefinementStrategy.REFINE,
            RefinementStrategy.BON_THEN_REFINE,
        ]:
            # Skip refinement if judge found no issues with best candidate
            if best_findings == "NO_ISSUES":
                logger.info(
                    "RefinementEngine skipping refinement: judge found NO_ISSUES for best candidate"
                )
                trace.budget_used = {"calls": calls_used, "tokens": tokens_used}
                return best_candidate, trace

            logger.info(
                "RefinementEngine starting refinement steps max_steps=%d feedback=%s",
                refine_config.max_refine_steps,
                best_feedback[:200] if best_feedback else None,
            )
            current_result = best_candidate
            current_feedback = best_feedback

            for step in range(refine_config.max_refine_steps):
                # Check budget before refinement step
                if refine_config.budget and refine_config.budget.check_exceeded(
                    calls_used, tokens_used
                ):
                    logger.info(
                        "RefinementEngine budget exceeded at step=%d calls_used=%d",
                        step,
                        calls_used,
                    )
                    warnings.warn(f"Budget exceeded at refinement step {step}")
                    break

                logger.info(
                    "RefinementEngine refine step=%d/%d starting with feedback=%s",
                    step + 1,
                    refine_config.max_refine_steps,
                    current_feedback[:100] if current_feedback else None,
                )
                prev_entities = current_result.entities.copy()
                # Pass judge feedback as issues to refiner
                refined_result = self.refiner(
                    text, current_result, issues=current_feedback
                )
                calls_used += 1

                # Track changes
                diff = self._compute_diff(prev_entities, refined_result.entities)
                trace.refine_diffs.append(
                    {
                        "step": step,
                        "changes": diff,
                        "confidence_change": refined_result.confidence
                        - current_result.confidence,
                        "feedback_used": current_feedback,
                    }
                )
                # Update feedback for next iteration (use detected issues from refiner)
                current_feedback = self.refiner._detect_issues(text, refined_result)
                logger.info(
                    "RefinementEngine refine step=%d/%d complete changes=%d confidence_delta=%.3f calls_used=%d",
                    step + 1,
                    refine_config.max_refine_steps,
                    len(diff),
                    refined_result.confidence - current_result.confidence,
                    calls_used,
                )

                current_result = refined_result

                # Early stopping if no changes
                if not diff:
                    logger.info(
                        "RefinementEngine early stopping at step=%d (no changes)",
                        step + 1,
                    )
                    break

            best_candidate = current_result

        trace.budget_used = {"calls": calls_used, "tokens": tokens_used}
        logger.info(
            "RefinementEngine.forward complete budget_used=%s chosen_candidate=%s final_confidence=%.3f",
            trace.budget_used,
            trace.chosen_candidate,
            best_candidate.confidence,
        )
        return best_candidate, trace

    def _generate_candidates(
        self,
        text: str,
        n_candidates: int,
        temperature: float,
        run_validation: bool = False,
    ) -> List[ExtractionResult]:
        """Generate multiple extraction candidates with diversity.

        Candidates are generated in parallel using ThreadPoolExecutor for
        improved performance when using API-based LLMs.
        """
        if n_candidates == 1:
            # No need for parallel execution with single candidate
            return [self.extractor(text, run_validation=run_validation)]

        def extract_candidate(candidate_idx: int) -> Tuple[int, ExtractionResult]:
            """Extract a single candidate, returning index for ordering."""
            logger.info(
                "RefinementEngine generating candidate %d/%d",
                candidate_idx + 1,
                n_candidates,
            )
            result = self.extractor(text, run_validation=run_validation)
            logger.info(
                "RefinementEngine candidate %d/%d complete confidence=%.3f",
                candidate_idx + 1,
                n_candidates,
                result.confidence,
            )
            return (candidate_idx, result)

        # Generate candidates in parallel
        candidates: List[Optional[ExtractionResult]] = [None] * n_candidates
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_candidates
        ) as executor:
            futures = [
                executor.submit(extract_candidate, i) for i in range(n_candidates)
            ]
            for future in concurrent.futures.as_completed(futures):
                idx, result = future.result()
                candidates[idx] = result

        # Filter out any None values (shouldn't happen, but defensive)
        return [c for c in candidates if c is not None]

    def _get_judge(self, custom_rubric: Optional[str]):
        """Get appropriate judge based on configuration."""
        if custom_rubric:
            if (
                not hasattr(self, "_custom_judge")
                or self._custom_judge.rubric != custom_rubric
            ):
                self._custom_judge = CustomJudge(self.schema, custom_rubric)
            return self._custom_judge
        else:
            return self.builtin_judge

    def _compute_diff(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute differences between two entity dictionaries."""
        changes = {}

        # Check for changed values
        for key in set(before.keys()) | set(after.keys()):
            before_val = before.get(key)
            after_val = after.get(key)

            if before_val != after_val:
                changes[key] = {
                    "before": before_val,
                    "after": after_val,
                    "change_type": "modified" if key in before else "added",
                }

        # Check for removed keys
        for key in before.keys() - after.keys():
            changes[key] = {
                "before": before[key],
                "after": None,
                "change_type": "removed",
            }

        return changes
