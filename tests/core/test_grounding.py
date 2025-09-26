"""Unit tests for SourceGrounder exact/fuzzy/semantic matching and validation."""

from langstruct.core.grounding import SourceGrounder
from langstruct.core.schemas import SourceSpan


def test_exact_matches_basic():
    text = "Alice went to Wonderland. Alice met the Hatter."
    g = SourceGrounder()

    matches = g._find_exact_matches(text, "Alice")
    # Two occurrences
    assert len(matches) == 2
    # First occurrence at start
    assert matches[0].span.start == 0
    assert matches[0].span.text == "Alice"


def test_ground_entities_with_offset():
    text = "Hello Alice"
    g = SourceGrounder()
    entities = {"name": "Alice"}

    grounded = g.ground_entities(text, entities, chunk_offset=100)
    assert "name" in grounded
    assert len(grounded["name"]) == 1
    span = grounded["name"][0]
    # Offset applied
    assert span.start == text.find("Alice") + 100
    assert span.text.lower() == "alice"


def test_validate_grounding_exact():
    text = "The value is 42."
    span = SourceSpan(start=text.find("42"), end=text.find("42") + 2, text="42")
    g = SourceGrounder()

    score = g.validate_grounding(text, span, expected_value="42")
    assert score == 1.0
