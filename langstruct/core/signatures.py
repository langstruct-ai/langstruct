"""DSPy signatures for structured extraction tasks."""

from typing import Any, Dict, List, Literal

import dspy
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated


class JudgeScoreItem(BaseModel):
    """Score for a single extraction candidate."""

    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1")
    reasoning: str = Field(description="Explanation of the score")
    feedback: str = Field(description="Specific actionable improvements")
    findings: Literal["NO_ISSUES", "ISSUES"] = Field(
        description=(
            "NO_ISSUES if the extraction is correct and complete (including correctly empty "
            "extractions when the data doesn't match schema requirements), "
            "ISSUES if problems were found that need fixing"
        )
    )

    @field_validator("findings", mode="before")
    @classmethod
    def normalize_findings(cls, v: Any) -> str:
        """Normalize LLM output variations to exact enum values."""
        if isinstance(v, str):
            normalized = v.strip().upper().replace(" ", "_").replace("-", "_")
            if normalized in (
                "NO_ISSUES",
                "NOISSUES",
                "NO_ISSUE",
                "NONE",
                "FINISHED_GENERATION",
                "FINISHED",
                "COMPLETE",
                "COMPLETED",
            ):
                return "NO_ISSUES"
            return "ISSUES"
        return "ISSUES"


class JudgeScores(BaseModel):
    """Judgment results for multiple candidates."""

    scores: List[JudgeScoreItem] = Field(description="Score for each candidate")


class ExtractEntities(dspy.Signature):
    """Extract structured entities from unstructured text.

    Given input text and a schema specification, identify and extract
    relevant entities that match the schema requirements.
    """

    text: Annotated[str, dspy.InputField(desc="Input text to extract from")]
    schema_spec: Annotated[
        str, dspy.InputField(desc="JSON schema defining expected output structure")
    ]
    entities: Annotated[
        str,
        dspy.OutputField(
            desc='Extracted entity VALUES as JSON matching the schema (e.g., {"name": "John", "age": 25})'
        ),
    ]


class ExtractWithSources(dspy.Signature):
    """Extract structured entities with source location grounding.

    Extract entities from text while maintaining precise mappings to
    source locations for verification and trust.
    """

    text: Annotated[str, dspy.InputField(desc="Input text to extract from")]
    schema_spec: Annotated[
        str, dspy.InputField(desc="JSON schema defining expected output structure")
    ]
    entities: Annotated[
        str,
        dspy.OutputField(
            desc='Extracted entity VALUES as JSON matching the schema (e.g., {"name": "John", "age": 25})'
        ),
    ]
    sources: Annotated[
        str,
        dspy.OutputField(
            desc="Source location mappings as JSON with start/end positions for each field"
        ),
    ]


class ValidateExtraction(dspy.Signature):
    """Validate extracted entities against schema and text.

    Verify that extracted entities are accurate, complete, and properly
    grounded in the source text.

    """

    text: Annotated[str, dspy.InputField(desc="Original source text")]
    entities: Annotated[str, dspy.InputField(desc="Extracted entities as JSON")]
    schema_spec: Annotated[str, dspy.InputField(desc="Expected schema specification")]
    is_valid: Annotated[bool, dspy.OutputField(desc="Whether extraction is valid")]
    feedback: Annotated[
        str, dspy.OutputField(desc="Validation feedback and suggestions")
    ]


class SummarizeExtraction(dspy.Signature):
    """Summarize extraction results across multiple text chunks.

    Combine and consolidate entities extracted from multiple text segments
    while removing duplicates and resolving conflicts.
    """

    extractions: Annotated[
        str, dspy.InputField(desc="List of extraction results as JSON")
    ]
    schema_spec: Annotated[str, dspy.InputField(desc="Expected output schema")]
    summary: Annotated[
        str, dspy.OutputField(desc="Consolidated extraction summary as JSON")
    ]
    confidence: Annotated[
        float, dspy.OutputField(desc="Overall confidence score (0-1)")
    ]


class ParseQuery(dspy.Signature):
    """Parse natural language query into semantic and structured components.

    Intelligently decompose a natural language query into:
    - Semantic terms for embedding-based similarity search
    - Structured filters for exact metadata matching

    The LLM should understand comparisons (over, above, below, less than),
    temporal references (Q3 2024, recent, latest), entity mentions,
    and map them to appropriate schema fields.
    """

    query: Annotated[str, dspy.InputField(desc="Natural language query to parse")]
    schema_spec: Annotated[
        str, dspy.InputField(desc="JSON schema defining available fields for filtering")
    ]
    semantic_terms: Annotated[
        str, dspy.OutputField(desc="JSON array of conceptual terms for semantic search")
    ]
    structured_filters: Annotated[
        str,
        dspy.OutputField(
            desc="JSON object of exact filters with operators like $gte, $lt, $in, $eq"
        ),
    ]


class RefineExtraction(dspy.Signature):
    """Refine an existing extraction by addressing specific issues.

    Take a current extraction and improve it by fixing identified issues
    like missing fields, incorrect values, or source misalignments.
    Focus on repair rather than complete re-extraction.
    """

    text: Annotated[str, dspy.InputField(desc="Original source text")]
    current_extraction: Annotated[
        str, dspy.InputField(desc="Current extraction as JSON")
    ]
    schema_spec: Annotated[str, dspy.InputField(desc="Expected schema specification")]
    issues: Annotated[
        str, dspy.InputField(desc="Specific issues to address in refinement")
    ]
    refined_extraction: Annotated[
        str, dspy.OutputField(desc="Improved extraction as JSON matching schema")
    ]


class JudgeExtractions(dspy.Signature):
    """Judge and score multiple extraction candidates.

    Evaluate extraction candidates against a rubric and provide scores,
    reasoning, and actionable feedback. Focus on faithfulness to source text,
    completeness, and accuracy of extracted information.

    For each candidate, set findings to NO_ISSUES if the extraction is correct
    and complete with no problems found, or ISSUES if there are problems that
    need fixing.
    """

    text: Annotated[str, dspy.InputField(desc="Original source text")]
    candidates: Annotated[
        str,
        dspy.InputField(desc="JSON array of extraction candidates with their sources"),
    ]
    schema_spec: Annotated[str, dspy.InputField(desc="Expected schema specification")]
    rubric: Annotated[str, dspy.InputField(desc="Scoring rubric and criteria")]
    scores: Annotated[
        JudgeScores,
        dspy.OutputField(
            desc="Structured judgment with scores, feedback, and findings for each candidate"
        ),
    ]
