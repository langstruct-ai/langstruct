"""Core DSPy extraction modules implementing the extraction pipeline."""

import json
from typing import Any, Dict, List, Optional, Type

import dspy
from pydantic import BaseModel, ValidationError

from .chunking import ChunkingConfig, TextChunk, TextChunker
from .grounding import SourceGrounder
from .schema_utils import get_field_descriptions, get_json_schema
from .schemas import ChunkResult, ExtractionResult, ParsedQuery, SourceSpan
from .signatures import (
    ExtractEntities,
    ExtractWithSources,
    ParseQuery,
    SummarizeExtraction,
    ValidateExtraction,
)


class EntityExtractor(dspy.Module):
    """Core entity extraction module using DSPy Chain of Thought."""

    def __init__(self, schema: Type[BaseModel], use_sources: bool = True):
        super().__init__()
        self.schema = schema
        self.use_sources = use_sources

        # Initialize DSPy modules
        if use_sources:
            self.extract = dspy.ChainOfThought(ExtractWithSources)
        else:
            self.extract = dspy.ChainOfThought(ExtractEntities)

        self.validate = dspy.ChainOfThought(ValidateExtraction)
        self.grounder = SourceGrounder()

    def forward(self, text: str, chunk_offset: int = 0) -> ExtractionResult:
        """Extract structured entities from text with validation."""
        schema_json = json.dumps(get_json_schema(self.schema), indent=2)

        # Perform extraction
        if self.use_sources:
            result = self.extract(text=text, schema_spec=schema_json)
            entities_json = result.entities
            sources_json = getattr(result, "sources", "{}")
        else:
            result = self.extract(text=text, schema_spec=schema_json)
            entities_json = result.entities
            sources_json = "{}"

        # Parse extracted entities
        try:
            entities_dict = json.loads(entities_json)
            sources_dict = json.loads(sources_json)
        except json.JSONDecodeError:
            # Fallback to empty result if JSON parsing fails
            entities_dict = {}
            sources_dict = {}

        # Validate extraction using DSPy
        validation = self.validate(
            text=text, entities=entities_json, schema_spec=schema_json
        )

        # Ground entities to source locations
        if not sources_dict:  # If LLM didn't provide sources, compute them
            grounded_sources = self.grounder.ground_entities(
                text, entities_dict, chunk_offset
            )
        else:
            # Convert LLM-provided sources to SourceSpan objects
            # Pass text so we can extract span text if LLM didn't provide it
            grounded_sources = self._parse_llm_sources(sources_dict, chunk_offset, text)

            # Validate LLM-provided sources and fall back to grounding if invalid
            grounded_sources = self._validate_and_fix_sources(
                grounded_sources, text, entities_dict, chunk_offset
            )

        # Calculate overall confidence
        confidence = self._calculate_confidence(validation.is_valid, entities_dict)

        return ExtractionResult(
            entities=entities_dict,
            sources=grounded_sources,
            confidence=confidence,
            metadata={
                "validation_feedback": validation.feedback,
                "is_valid": validation.is_valid,
                "chunk_offset": chunk_offset,
            },
        )

    def _parse_llm_sources(
        self, sources_dict: Dict, chunk_offset: int, text: str = None
    ) -> Dict[str, List[SourceSpan]]:
        """Parse source information provided by LLM into SourceSpan objects.

        Args:
            sources_dict: Source information from LLM
            chunk_offset: Offset to add to positions
            text: Original text to extract span text from if not provided by LLM
        """
        parsed_sources = {}

        for field_name, source_list in sources_dict.items():
            spans = []
            if isinstance(source_list, list):
                for source_info in source_list:
                    try:
                        if isinstance(source_info, dict):
                            # LLM provides positions relative to the chunk text
                            local_start = source_info.get("start", 0)
                            local_end = source_info.get("end", 0)

                            # Get text from LLM or extract from original text
                            span_text = source_info.get("text", "")
                            if (
                                not span_text
                                and text
                                and 0 <= local_start < len(text)
                                and local_start < local_end <= len(text)
                            ):
                                # Extract text from original chunk if not provided
                                span_text = text[local_start:local_end]

                            # Add chunk offset for document-level positions
                            start = local_start + chunk_offset
                            end = local_end + chunk_offset

                            span = SourceSpan(start=start, end=end, text=span_text)
                            spans.append(span)
                    except (KeyError, TypeError, ValidationError):
                        continue  # Skip invalid source entries

            parsed_sources[field_name] = spans

        return parsed_sources

    def _validate_and_fix_sources(
        self,
        sources: Dict[str, List[SourceSpan]],
        text: str,
        entities_dict: Dict,
        chunk_offset: int,
    ) -> Dict[str, List[SourceSpan]]:
        """Validate LLM-provided sources and fix incorrect ones.

        If the LLM provides sources that don't match the entity values,
        fall back to automatic grounding for those fields.
        """
        fixed_sources = {}

        for field_name, value in entities_dict.items():
            if value is None:
                continue

            field_spans = sources.get(field_name, [])

            # Check if any of the spans actually contain the entity value
            valid_spans = []
            for span in field_spans:
                if span.text:
                    # Check if the span text reasonably matches the entity value
                    value_str = str(value).lower().strip()
                    span_text_lower = span.text.lower().strip()

                    # Check for various matching criteria
                    is_valid = False

                    # Exact match or contains
                    if value_str in span_text_lower or span_text_lower in value_str:
                        is_valid = True
                    # For multi-word values, check if any significant words match
                    elif len(value_str.split()) > 1:
                        value_words = [
                            w for w in value_str.split() if len(w) > 2
                        ]  # Skip short words
                        # Check if any significant word from value appears in span
                        if value_words and any(
                            word in span_text_lower for word in value_words
                        ):
                            is_valid = True
                    # For numbers, check exact match
                    elif value_str.isdigit() and value_str in span_text_lower:
                        is_valid = True

                    if is_valid:
                        valid_spans.append(span)

            # If no valid spans found OR spans don't contain full value, try automatic grounding
            if value:
                # Check if any span contains the complete value
                has_complete_match = any(
                    str(value).lower() in span.text.lower()
                    for span in valid_spans
                    if span.text
                )

                # If no complete match, try to find better spans via grounding
                if not has_complete_match:
                    grounded = self.grounder.ground_entities(
                        text, {field_name: value}, chunk_offset
                    )
                    if field_name in grounded and grounded[field_name]:
                        # Use grounded spans if they're better (contain full value)
                        better_spans = [
                            span
                            for span in grounded[field_name]
                            if span.text and str(value).lower() in span.text.lower()
                        ]
                        if better_spans:
                            fixed_sources[field_name] = better_spans
                        elif valid_spans:
                            fixed_sources[field_name] = valid_spans
                        else:
                            fixed_sources[field_name] = (
                                grounded[field_name]
                                if grounded[field_name]
                                else field_spans
                            )
                    else:
                        fixed_sources[field_name] = (
                            valid_spans if valid_spans else field_spans
                        )
                else:
                    fixed_sources[field_name] = valid_spans
            else:
                fixed_sources[field_name] = field_spans

        return fixed_sources

    def _calculate_confidence(self, is_valid: bool, entities_dict: Dict) -> float:
        """Calculate overall extraction confidence score."""
        base_confidence = 0.8 if is_valid else 0.4

        # Adjust based on completeness (non-empty extractions)
        non_empty_count = sum(
            1 for v in entities_dict.values() if v is not None and str(v).strip()
        )
        total_fields = len(get_field_descriptions(self.schema))

        if total_fields > 0:
            completeness_bonus = 0.2 * (non_empty_count / total_fields)
        else:
            completeness_bonus = 0

        return min(1.0, base_confidence + completeness_bonus)


class TextChunkerModule(dspy.Module):
    """Text chunking module for processing long documents."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        super().__init__()
        self.chunker = TextChunker(config)

    def forward(self, text: str) -> List[TextChunk]:
        """Chunk text into manageable pieces."""
        return self.chunker.chunk_text(text)


class ResultAggregator(dspy.Module):
    """Aggregates extraction results from multiple text chunks."""

    def __init__(self, schema: Type[BaseModel]):
        super().__init__()
        self.schema = schema
        self.summarize = dspy.ChainOfThought(SummarizeExtraction)

    def forward(self, chunk_results: List[ChunkResult]) -> ExtractionResult:
        """Combine results from multiple chunks into a single result."""
        if not chunk_results:
            return ExtractionResult(entities={}, sources={})

        if len(chunk_results) == 1:
            return chunk_results[0].extraction

        # Prepare extractions for summarization
        extractions_json = json.dumps(
            [
                {
                    "chunk_id": cr.chunk_id,
                    "entities": cr.extraction.entities,
                    "confidence": cr.extraction.confidence,
                }
                for cr in chunk_results
            ],
            indent=2,
        )

        schema_json = json.dumps(get_json_schema(self.schema), indent=2)

        # Use DSPy to intelligently combine extractions
        summary_result = self.summarize(
            extractions=extractions_json, schema_spec=schema_json
        )

        try:
            combined_entities = json.loads(summary_result.summary)
        except json.JSONDecodeError:
            # Fallback: conservatively merge entities across chunks
            combined_entities = self._conservative_merge(chunk_results)

        # Combine source locations from all chunks
        combined_sources = self._merge_sources(chunk_results)

        # Use the confidence from DSPy or calculate average
        try:
            overall_confidence = float(summary_result.confidence)
        except (ValueError, TypeError):
            overall_confidence = sum(
                cr.extraction.confidence for cr in chunk_results
            ) / len(chunk_results)

        return ExtractionResult(
            entities=combined_entities,
            sources=combined_sources,
            confidence=overall_confidence,
            metadata={
                "num_chunks": len(chunk_results),
                "chunk_ids": [cr.chunk_id for cr in chunk_results],
            },
        )

    def _merge_sources(
        self, chunk_results: List[ChunkResult]
    ) -> Dict[str, List[SourceSpan]]:
        """Merge source spans from multiple chunks."""
        merged_sources = {}

        for chunk_result in chunk_results:
            for field_name, spans in chunk_result.extraction.sources.items():
                if field_name not in merged_sources:
                    merged_sources[field_name] = []
                merged_sources[field_name].extend(spans)

        # Sort spans by start position
        for field_name in merged_sources:
            merged_sources[field_name].sort(key=lambda span: span.start)

        return merged_sources

    def _conservative_merge(self, chunk_results: List[ChunkResult]) -> Dict[str, Any]:
        """Conservatively merge entities across chunks when summarization fails.

        Strategy:
        - For scalar values: choose the non-empty value from the highest-confidence chunk.
        - For lists: union unique items preserving order of first appearance.
        - For dicts: shallow-merge keys using the same scalar/list rules recursively.
        """
        from collections import OrderedDict

        def merge_values(values: List[Any], confidences: List[float]) -> Any:
            # Determine type by first non-None value
            first_non_none = next((v for v in values if v is not None), None)
            if isinstance(first_non_none, list):
                seen = OrderedDict()
                for v in values:
                    if isinstance(v, list):
                        for item in v:
                            key = str(item)
                            if key not in seen:
                                seen[key] = item
                return list(seen.values())
            if isinstance(first_non_none, dict):
                keys = set()
                for v in values:
                    if isinstance(v, dict):
                        keys.update(v.keys())
                merged = {}
                for k in keys:
                    sub_values = [v.get(k) for v in values if isinstance(v, dict)]
                    merged[k] = merge_values(sub_values, confidences)
                return merged
            # Scalars: pick from highest-confidence non-empty
            best_idx = None
            for i, v in sorted(
                enumerate(values), key=lambda x: confidences[x[0]], reverse=True
            ):
                if v is not None and str(v).strip():
                    best_idx = i
                    break
            return values[best_idx] if best_idx is not None else first_non_none

        # Collect values per field
        field_names = set()
        for cr in chunk_results:
            field_names.update(cr.extraction.entities.keys())

        merged: Dict[str, Any] = {}
        confidences = [cr.extraction.confidence for cr in chunk_results]
        for field in field_names:
            values = [cr.extraction.entities.get(field) for cr in chunk_results]
            merged[field] = merge_values(values, confidences)

        return merged


class ExtractionPipeline(dspy.Module):
    """Complete extraction pipeline combining all modules."""

    def __init__(
        self,
        schema: Type[BaseModel],
        chunking_config: Optional[ChunkingConfig] = None,
        use_sources: bool = True,
    ):
        super().__init__()
        self.schema = schema
        self.chunker = TextChunkerModule(chunking_config)
        self.extractor = EntityExtractor(schema, use_sources)
        self.aggregator = ResultAggregator(schema)

    def forward(self, text: str) -> ExtractionResult:
        """Run the complete extraction pipeline on input text."""
        # Step 1: Chunk the text
        chunks = self.chunker(text)

        # Step 2: Extract from each chunk
        chunk_results = []
        for chunk in chunks:
            extraction = self.extractor(chunk.text, chunk.start_offset)
            chunk_result = ChunkResult(
                chunk_id=chunk.id,
                chunk_text=chunk.text,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                extraction=extraction,
            )
            chunk_results.append(chunk_result)

        # Step 3: Aggregate results
        final_result = self.aggregator(chunk_results)

        # Add pipeline metadata
        final_result.metadata.update(
            {
                "pipeline": "langstruct",
                "total_chunks": len(chunks),
                "original_text_length": len(text),
            }
        )

        return final_result


class QueryParser(dspy.Module):
    """Query parsing module using DSPy Chain of Thought for intelligent parsing.

    This module uses an LLM to parse natural language queries into:
    - Semantic terms for embedding-based search
    - Structured filters for exact metadata matching

    No regex patterns or hardcoded rules - pure LLM intelligence.
    """

    def __init__(self, schema: Type[BaseModel]):
        """Initialize query parser with target schema.

        Args:
            schema: The schema that defines available fields for filtering
        """
        super().__init__()
        self.schema = schema
        self.parse = dspy.ChainOfThought(ParseQuery)

    def __call__(self, query: str) -> ParsedQuery:
        """Make the module callable directly."""
        return self.forward(query)

    def forward(self, query: str) -> ParsedQuery:
        """Parse natural language query using LLM intelligence.

        Args:
            query: Natural language query to parse

        Returns:
            ParsedQuery with semantic terms and structured filters
        """
        # Get schema specification as JSON
        schema_json = json.dumps(get_json_schema(self.schema), indent=2)

        # Let the LLM handle ALL parsing logic intelligently
        result = self.parse(query=query, schema_spec=schema_json)

        # Parse LLM output
        try:
            semantic_terms = json.loads(result.semantic_terms)
            if not isinstance(semantic_terms, list):
                semantic_terms = [semantic_terms] if semantic_terms else []
        except (json.JSONDecodeError, AttributeError):
            semantic_terms = []

        try:
            structured_filters = json.loads(result.structured_filters)
            if not isinstance(structured_filters, dict):
                structured_filters = {}
        except (json.JSONDecodeError, AttributeError):
            structured_filters = {}

        # Calculate confidence based on how well the LLM parsed the query
        confidence = self._calculate_confidence(semantic_terms, structured_filters)

        # Generate human-readable explanation
        explanation = self._generate_explanation(
            query, semantic_terms, structured_filters
        )

        return ParsedQuery(
            semantic_terms=semantic_terms,
            structured_filters=structured_filters,
            confidence=confidence,
            explanation=explanation,
            raw_query=query,
            metadata={"schema_used": self.schema.__name__, "parsed_by": "llm"},
        )

    def _calculate_confidence(
        self, semantic_terms: List[str], structured_filters: Dict
    ) -> float:
        """Calculate confidence score for the parsing.

        Args:
            semantic_terms: Parsed semantic search terms
            structured_filters: Parsed structured filters

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence if we got any output
        confidence = 0.5

        # Increase confidence for semantic terms
        if semantic_terms:
            confidence += 0.25

        # Increase confidence for structured filters
        if structured_filters:
            confidence += 0.25

        return min(confidence, 1.0)

    def _generate_explanation(
        self, query: str, semantic_terms: List[str], structured_filters: Dict
    ) -> str:
        """Generate human-readable explanation of the parsing.

        Args:
            query: Original query
            semantic_terms: Parsed semantic terms
            structured_filters: Parsed filters

        Returns:
            Human-readable explanation
        """
        parts = []

        # Explain semantic search
        if semantic_terms:
            parts.append(f"Searching for: {', '.join(semantic_terms)}")

        # Explain filters
        if structured_filters:
            parts.append("With filters:")
            for key, value in structured_filters.items():
                if isinstance(value, dict):
                    # Handle operators like $gte, $lt, etc.
                    for op, val in value.items():
                        op_text = {
                            "$gte": "≥",
                            "$gt": ">",
                            "$lte": "≤",
                            "$lt": "<",
                            "$eq": "=",
                            "$ne": "≠",
                            "$in": "in",
                        }.get(op, op)
                        if op == "$in":
                            parts.append(f"  • {key} {op_text} {val}")
                        else:
                            parts.append(f"  • {key} {op_text} {val}")
                else:
                    parts.append(f"  • {key} = {value}")

        if not parts:
            parts = [f"Treating entire query as semantic search: '{query}'"]

        return "\n".join(parts)
