"""Tests for ResultAggregator conservative merge and source merging."""

from typing import Dict, List

from pydantic import BaseModel, Field

from langstruct.core.modules import ResultAggregator
from langstruct.core.schemas import ChunkResult, ExtractionResult, SourceSpan


class MergeTestSchema(BaseModel):
    name: str = Field(description="Name")
    age: int = Field(description="Age")
    skills: List[str] = Field(description="Skills")
    meta: Dict[str, str] = Field(description="Metadata")


def _mk_chunk_result(
    chunk_id: int,
    text: str,
    start: int,
    end: int,
    entities: Dict,
    confidence: float,
    sources: Dict[str, List[SourceSpan]] | None = None,
) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        chunk_text=text,
        start_offset=start,
        end_offset=end,
        extraction=ExtractionResult(
            entities=entities,
            sources=sources or {},
            confidence=confidence,
        ),
    )


def test_conservative_merge_scalars_lists_dicts():
    aggr = ResultAggregator(MergeTestSchema)

    # Chunk 0: lower confidence, has some values
    c0 = _mk_chunk_result(
        0,
        "Text 0",
        0,
        6,
        entities={
            "name": "Alice",
            "age": 29,
            "skills": ["python"],
            "meta": {"role": "dev"},
        },
        confidence=0.6,
    )

    # Chunk 1: higher confidence, overrides scalar age, extends list, adds dict key
    c1 = _mk_chunk_result(
        1,
        "Text 1",
        6,
        12,
        entities={
            "name": "Alice",  # same
            "age": 30,  # prefer this (higher-confidence)
            "skills": ["ml", "python"],  # union should include both unique values
            "meta": {"team": "ai"},  # merged with previous
        },
        confidence=0.9,
    )

    merged = aggr._conservative_merge([c0, c1])

    assert merged["name"] == "Alice"
    assert merged["age"] == 30  # picked from higher-confidence chunk
    assert set(merged["skills"]) == {"python", "ml"}
    # shallow dict merge: keys from both
    assert merged["meta"]["role"] == "dev"
    assert merged["meta"]["team"] == "ai"


def test_merge_sources_sorts_and_concatenates():
    aggr = ResultAggregator(MergeTestSchema)

    s1 = SourceSpan(start=10, end=15, text="Alice")
    s2 = SourceSpan(start=0, end=5, text="Alice")

    c0 = _mk_chunk_result(
        0,
        "Text 0",
        0,
        6,
        entities={"name": "Alice", "age": 29, "skills": [], "meta": {}},
        confidence=0.7,
        sources={"name": [s1]},
    )

    c1 = _mk_chunk_result(
        1,
        "Text 1",
        6,
        12,
        entities={"name": "Alice", "age": 29, "skills": [], "meta": {}},
        confidence=0.8,
        sources={"name": [s2]},
    )

    merged_sources = aggr._merge_sources([c0, c1])
    assert "name" in merged_sources
    # Should be sorted by start
    assert [span.start for span in merged_sources["name"]] == [0, 10]
