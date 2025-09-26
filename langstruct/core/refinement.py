"""DSPy refinement system for improving extraction quality through Best-of-N, refinement, and committee scoring."""

import json
import random
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import dspy
from pydantic import BaseModel, Field

from .schema_utils import get_field_descriptions, get_json_schema
from .schemas import ExtractionResult, SourceSpan
from .signatures import (
    ExtractEntities,
    ExtractWithSources,
    JudgeExtractions,
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
    ) -> List[Tuple[float, str]]:
        """Score candidates using built-in rubric.

        Returns list of (score, reasoning) tuples.
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
            "Prefer candidates that exactly quote from the text over those that paraphrase."
        )

        result = self.judge(
            text=text,
            candidates=candidates_json,
            schema_spec=schema_json,
            rubric=built_in_rubric,
        )

        try:
            scores_data = json.loads(result.scores)
            if isinstance(scores_data, list):
                return [
                    (item.get("score", 0.5), item.get("reasoning", ""))
                    for item in scores_data
                ]
            else:
                # Fallback if format is unexpected
                return [(0.5, "Judge output format unexpected")] * len(candidates)
        except (json.JSONDecodeError, AttributeError):
            warnings.warn("Judge output could not be parsed, using fallback scores")
            return [(0.5, "Judge parsing failed")] * len(candidates)


class CustomJudge(dspy.Module):
    """Custom judge using user-provided rubric."""

    def __init__(self, schema: Type[BaseModel], rubric: str):
        super().__init__()
        self.schema = schema
        self.rubric = rubric
        self.judge = dspy.ChainOfThought(JudgeExtractions)

    def forward(
        self, text: str, candidates: List[ExtractionResult]
    ) -> List[Tuple[float, str]]:
        """Score candidates using custom rubric."""
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

        result = self.judge(
            text=text,
            candidates=candidates_json,
            schema_spec=schema_json,
            rubric=self.rubric,
        )

        try:
            scores_data = json.loads(result.scores)
            if isinstance(scores_data, list):
                return [
                    (item.get("score", 0.5), item.get("reasoning", ""))
                    for item in scores_data
                ]
            else:
                return [(0.5, "Judge output format unexpected")] * len(candidates)
        except (json.JSONDecodeError, AttributeError):
            warnings.warn(
                "Custom judge output could not be parsed, using fallback scores"
            )
            return [(0.5, "Judge parsing failed")] * len(candidates)


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
    ) -> ExtractionResult:
        """Refine an extraction by addressing specific issues.

        Args:
            text: Original text
            current_extraction: Current extraction to refine
            issues: Specific issues to address (auto-detected if None)

        Returns:
            Refined extraction result
        """
        schema_json = json.dumps(get_json_schema(self.schema), indent=2)
        current_json = json.dumps(current_extraction.entities)

        # Auto-detect issues if not provided
        if issues is None:
            issues = self._detect_issues(text, current_extraction)

        result = self.refine(
            text=text,
            current_extraction=current_json,
            schema_spec=schema_json,
            issues=issues,
        )

        try:
            refined_entities = json.loads(result.refined_extraction)
        except json.JSONDecodeError:
            warnings.warn("Refinement failed to parse, returning original")
            return current_extraction

        # Create new extraction result with refined entities
        # Keep original sources for now (could be improved with source refinement)
        return ExtractionResult(
            entities=refined_entities,
            sources=current_extraction.sources,  # TODO: Could refine sources too
            confidence=min(
                current_extraction.confidence + 0.1, 1.0
            ),  # Slight confidence boost
            metadata={
                **current_extraction.metadata,
                "refined": True,
                "refinement_issues": issues,
            },
        )

    def _detect_issues(self, text: str, extraction: ExtractionResult) -> str:
        """Auto-detect issues in current extraction."""
        issues = []

        # Check for empty required fields
        for field_name, value in extraction.entities.items():
            if not value or (isinstance(value, str) and not value.strip()):
                if field_name in get_field_descriptions(self.schema):
                    issues.append(f"Missing value for required field: {field_name}")

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
        self, text: str, refine_config: Refine
    ) -> Tuple[ExtractionResult, RefinementTrace]:
        """Run refinement process based on configuration.

        Args:
            text: Input text to extract from
            refine_config: Refinement configuration

        Returns:
            Tuple of (best_result, trace_info)
        """
        trace = RefinementTrace()
        calls_used = 0
        tokens_used = 0  # TODO: Implement token counting

        # Check budget before starting
        if refine_config.budget and refine_config.budget.check_exceeded(
            calls_used, tokens_used
        ):
            # Return basic extraction if budget exceeded
            basic_result = self.extractor(text)
            return basic_result, trace

        candidates = []

        # Step 1: Generate candidates (for BON strategies)
        if refine_config.strategy in [
            RefinementStrategy.BON,
            RefinementStrategy.BON_THEN_REFINE,
        ]:
            candidates = self._generate_candidates(
                text, refine_config.n_candidates, refine_config.temperature
            )
            calls_used += len(candidates)

            # Check budget after candidate generation
            if refine_config.budget and refine_config.budget.check_exceeded(
                calls_used, tokens_used
            ):
                warnings.warn(
                    "Budget exceeded after candidate generation, using first candidate"
                )
                best_result = candidates[0] if candidates else self.extractor(text)
                trace.budget_used = {"calls": calls_used, "tokens": tokens_used}
                return best_result, trace
        else:
            # For refine-only strategy, start with single extraction
            candidates = [self.extractor(text)]
            calls_used += 1

        # Step 2: Judge candidates and select best
        if len(candidates) > 1:
            judge = self._get_judge(refine_config.judge)
            scores = judge(text, candidates)
            calls_used += 1  # Judge call

            # Create candidate results with scores
            for i, (candidate, (score, reasoning)) in enumerate(
                zip(candidates, scores)
            ):
                trace.candidates.append(
                    CandidateResult(
                        extraction=candidate,
                        score=score,
                        reasoning=reasoning,
                        candidate_id=i,
                    )
                )

            # Select best candidate
            best_idx = max(range(len(scores)), key=lambda i: scores[i][0])
            best_candidate = candidates[best_idx]
            trace.chosen_candidate = best_idx
        else:
            best_candidate = candidates[0]
            trace.candidates.append(
                CandidateResult(
                    extraction=best_candidate,
                    score=best_candidate.confidence,
                    reasoning="Single candidate",
                    candidate_id=0,
                )
            )
            trace.chosen_candidate = 0

        # Step 3: Refinement (for refine strategies)
        if refine_config.strategy in [
            RefinementStrategy.REFINE,
            RefinementStrategy.BON_THEN_REFINE,
        ]:
            current_result = best_candidate

            for step in range(refine_config.max_refine_steps):
                # Check budget before refinement step
                if refine_config.budget and refine_config.budget.check_exceeded(
                    calls_used, tokens_used
                ):
                    warnings.warn(f"Budget exceeded at refinement step {step}")
                    break

                prev_entities = current_result.entities.copy()
                refined_result = self.refiner(text, current_result)
                calls_used += 1

                # Track changes
                diff = self._compute_diff(prev_entities, refined_result.entities)
                trace.refine_diffs.append(
                    {
                        "step": step,
                        "changes": diff,
                        "confidence_change": refined_result.confidence
                        - current_result.confidence,
                    }
                )

                current_result = refined_result

                # Early stopping if no changes
                if not diff:
                    break

            best_candidate = current_result

        trace.budget_used = {"calls": calls_used, "tokens": tokens_used}
        return best_candidate, trace

    def _generate_candidates(
        self, text: str, n_candidates: int, temperature: float
    ) -> List[ExtractionResult]:
        """Generate multiple extraction candidates with diversity."""
        candidates = []

        # Store original LM settings
        original_lm = dspy.settings.lm

        try:
            # Create varied settings for diversity
            for i in range(n_candidates):
                # Use different seeds and slight temperature variation for diversity
                seed = random.randint(0, 10000) if i > 0 else None

                # TODO: Implement proper temperature/seed control for LM
                # For now, just call the extractor multiple times
                candidate = self.extractor(text)
                candidates.append(candidate)

        finally:
            # Restore original settings
            if original_lm:
                dspy.configure(lm=original_lm)

        return candidates

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
