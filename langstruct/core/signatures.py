"""DSPy signatures for structured extraction tasks."""

from typing import Any, Dict, List

import dspy
from typing_extensions import Annotated


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

    Evaluate extraction candidates against a rubric and provide scores
    and reasoning. Focus on faithfulness to source text, completeness,
    and accuracy of extracted information.
    """

    text: Annotated[str, dspy.InputField(desc="Original source text")]
    candidates: Annotated[
        str,
        dspy.InputField(desc="JSON array of extraction candidates with their sources"),
    ]
    schema_spec: Annotated[str, dspy.InputField(desc="Expected schema specification")]
    rubric: Annotated[str, dspy.InputField(desc="Scoring rubric and criteria")]
    scores: Annotated[
        str,
        dspy.OutputField(
            desc="JSON array with score (0-1) and reasoning for each candidate"
        ),
    ]
