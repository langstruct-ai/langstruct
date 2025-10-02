"""Slow integration tests that hit real LLM providers when API keys are configured."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import pytest

from langstruct import LangStruct
from langstruct.core.chunking import ChunkingConfig
from langstruct.core.refinement import Budget, Refine


@pytest.fixture(scope="module")
def optimization_dataset() -> Tuple[List[str], List[Dict[str, object]]]:
    """Provide lightweight training data for integration optimization runs."""
    texts = [
        """\
        Alice Johnson is a 29-year-old data scientist based in Seattle, Washington.
        She leads the analytics team at BlueSky Labs and mentors junior engineers.
        """.strip(),
    ]

    labels = [
        {"name": "Alice Johnson", "age": 29, "location": "Seattle, Washington"},
    ]

    return texts, labels


@pytest.fixture
def optimized_person_extractor(
    person_schema,
    optimization_dataset,
    requires_api_key,
):
    """Create a LangStruct instance that has been optimized against the dataset."""
    texts, labels = optimization_dataset

    extractor = LangStruct(
        schema=person_schema,
        optimizer="miprov2",
        use_sources=False,  # keep requests smaller for integration runs
    )

    extractor.optimize(
        texts=texts,
        expected_results=labels,
        validation_split=0.0,
    )

    return {
        "extractor": extractor,
        "train_texts": texts,
        "expected_results": labels,
    }


@pytest.mark.integration
def test_integration_optimize_smoke(optimized_person_extractor):
    """End-to-end smoke test covering optimize() and extraction afterwards."""
    bundle = optimized_person_extractor
    extractor: LangStruct = bundle["extractor"]

    test_text = (
        "Dr. Emily Davis is a 38-year-old physician based in Austin, Texas, "
        "where she leads the cardiology program at Central Health."
    )

    result = extractor.extract(test_text, validate=False, return_sources=False)

    assert isinstance(result.entities, dict)
    assert extractor.optimizer is not None
    assert getattr(extractor.optimizer, "optimizer", None) is not None
    assert 0.0 <= result.confidence <= 1.0
    assert any(str(v).strip() for v in result.entities.values())
    assert result.metadata.get("pipeline") == "langstruct"


@pytest.mark.integration
def test_integration_save_load_after_optimization(optimized_person_extractor, tmp_path):
    """Ensure optimized extractors persist and reload correctly."""
    bundle = optimized_person_extractor
    extractor: LangStruct = bundle["extractor"]
    texts: List[str] = bundle["train_texts"]

    save_path = tmp_path / "optimized_extractor"
    extractor.save(str(save_path))

    metadata_path = save_path / "langstruct_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    assert metadata["optimization_applied"] is True
    assert metadata["optimizer_name"] == "miprov2"

    loaded = LangStruct.load(str(save_path))
    loaded_result = loaded.extract(texts[0], validate=False, return_sources=False)

    assert isinstance(loaded_result.entities, dict)
    assert loaded.optimizer is not None
    assert any(str(v).strip() for v in loaded_result.entities.values())


@pytest.mark.integration
def test_integration_chunked_sources(person_schema, requires_api_key):
    """Validate extraction with source grounding across multiple chunks."""
    chunk_config = ChunkingConfig(
        max_tokens=12,
        overlap_tokens=4,
        min_chunk_tokens=3,
        preserve_paragraphs=False,
        preserve_sentences=False,
    )

    extractor = LangStruct(schema=person_schema, chunking_config=chunk_config)

    long_text = (
        "Charlotte Rivera is a 41-year-old neurologist based in San Diego, "
        "California. She leads the neuroscience unit at Horizon Medical Center. "
        "Outside of work, Charlotte mentors students at the local university."
    )

    result = extractor.extract(long_text, validate=False, return_sources=True)

    assert isinstance(result.entities, dict)
    assert result.sources
    assert result.metadata.get("total_chunks", 1) > 1
    assert any(spans for spans in result.sources.values())


@pytest.mark.integration
def test_integration_query_parsing(person_schema, requires_api_key):
    """Ensure query() returns structured output using the query parser."""
    extractor = LangStruct(schema=person_schema)

    query = "cardiologists in Seattle over 30"
    parsed = extractor.query(query, explain=False)

    assert parsed.raw_query == query
    assert 0.0 <= parsed.confidence <= 1.0
    assert parsed.metadata.get("parsed_by") == "llm"


@pytest.mark.integration
def test_integration_refinement_flow(person_schema, requires_api_key):
    """Exercise refinement engine with conservative budget to limit cost."""
    refine_config = Refine(
        strategy="bon",
        n_candidates=1,
        max_refine_steps=1,
        temperature=0.3,
        budget=Budget(max_calls=1),
    )

    extractor = LangStruct(
        schema=person_schema,
        refine=refine_config,
        use_sources=False,
    )

    text = (
        "Dr. Olivia Chen is a 36-year-old cardiologist working at Bayview Medical "
        "Center in San Francisco, California."
    )

    result = extractor.extract(text, validate=False, return_sources=False)

    assert result.metadata.get("refinement_applied")
    assert result.metadata.get("refinement_strategy") == refine_config.strategy
