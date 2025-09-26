"""Tests for LLM-based query parsing functionality."""

from unittest.mock import Mock, patch

import pytest

from langstruct import LangStruct, ParsedQuery
from langstruct.core.modules import QueryParser


class TestQueryParsing:
    """Test suite for query parsing functionality."""

    def test_query_method_exists(self):
        """Test that query() method exists on LangStruct."""
        schema = {"company": "Apple", "revenue": 100.0}
        ls = LangStruct(example=schema)
        assert hasattr(ls, "query")
        assert callable(ls.query)

    def test_query_returns_parsed_query(self):
        """Test that query() returns a ParsedQuery object."""
        schema = {"company": "Apple", "revenue": 100.0}
        ls = LangStruct(example=schema)

        # Mock the QueryParser to avoid LLM calls in tests
        with patch("langstruct.api.QueryParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_result = ParsedQuery(
                semantic_terms=["test"],
                structured_filters={},
                confidence=0.9,
                explanation="Test explanation",
                raw_query="test query",
                metadata={},
            )
            mock_parser.return_value = mock_result

            result = ls.query("test query")

            assert isinstance(result, ParsedQuery)
            assert result.semantic_terms == ["test"]
            assert result.confidence == 0.9

    def test_empty_query_handling(self):
        """Test that empty queries are handled gracefully."""
        schema = {"company": "Apple", "revenue": 100.0}
        ls = LangStruct(example=schema)

        result = ls.query("")
        assert isinstance(result, ParsedQuery)
        assert result.semantic_terms == []
        assert result.structured_filters == {}
        assert result.confidence == 0.0

        result = ls.query("   ")
        assert isinstance(result, ParsedQuery)
        assert result.semantic_terms == []
        assert result.structured_filters == {}
        assert result.confidence == 0.0

    def test_query_parser_initialization(self):
        """Test that QueryParser is initialized correctly."""
        schema = {"company": "Apple", "revenue": 100.0}
        ls = LangStruct(example=schema)

        # First query should initialize the parser
        with patch("langstruct.api.QueryParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_parser.return_value = ParsedQuery(
                semantic_terms=["test"],
                structured_filters={},
                confidence=0.9,
                explanation="",
                raw_query="test",
                metadata={},
            )

            # Clear any existing parser
            if hasattr(ls, "_query_parser"):
                delattr(ls, "_query_parser")

            ls.query("test")

            # Verify QueryParser was initialized with the schema
            mock_parser_class.assert_called_once_with(ls.schema)

    def test_query_error_handling(self):
        """Test that query parsing errors are handled gracefully."""
        schema = {"company": "Apple", "revenue": 100.0}
        ls = LangStruct(example=schema)

        # Mock the parser to raise an exception
        with patch("langstruct.api.QueryParser") as mock_parser_class:
            mock_parser_class.side_effect = Exception("Test error")

            result = ls.query("test query")

            # Should fall back to treating entire query as semantic
            assert isinstance(result, ParsedQuery)
            assert result.semantic_terms == ["test query"]
            assert result.structured_filters == {}
            assert result.confidence == 0.0
            assert "error" in result.metadata


class TestQueryParserModule:
    """Test suite for QueryParser module."""

    def test_query_parser_module_initialization(self):
        """Test QueryParser module initialization."""
        from pydantic import BaseModel

        class QueryTestSchema(BaseModel):
            company: str
            revenue: float

        parser = QueryParser(QueryTestSchema)
        assert parser.schema == QueryTestSchema
        assert hasattr(parser, "parse")

    def test_query_parser_callable(self):
        """Test that QueryParser can be called directly."""
        from pydantic import BaseModel

        class QueryTestSchema(BaseModel):
            company: str
            revenue: float

        parser = QueryParser(QueryTestSchema)

        # Mock the parse ChainOfThought
        with patch.object(parser, "parse") as mock_parse:
            mock_parse.return_value = Mock(
                semantic_terms='["test"]', structured_filters='{"company": "Apple"}'
            )

            result = parser("test query")

            assert isinstance(result, ParsedQuery)
            assert "llm" in result.metadata.get("parsed_by", "")

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        from pydantic import BaseModel

        class QueryTestSchema(BaseModel):
            company: str
            revenue: float

        parser = QueryParser(QueryTestSchema)

        # Test with both semantic terms and filters
        confidence = parser._calculate_confidence(
            ["term1", "term2"], {"company": "Apple", "revenue": 100}
        )
        assert confidence == 1.0

        # Test with only semantic terms
        confidence = parser._calculate_confidence(["term1"], {})
        assert confidence == 0.75

        # Test with only filters
        confidence = parser._calculate_confidence([], {"company": "Apple"})
        assert confidence == 0.75

        # Test with neither
        confidence = parser._calculate_confidence([], {})
        assert confidence == 0.5

    def test_explanation_generation(self):
        """Test human-readable explanation generation."""
        from pydantic import BaseModel

        class QueryTestSchema(BaseModel):
            company: str
            revenue: float

        parser = QueryParser(QueryTestSchema)

        # Test with semantic terms and filters
        explanation = parser._generate_explanation(
            "test query",
            ["tech companies", "AI"],
            {"company": "Apple", "revenue": {"$gte": 100}},
        )

        assert "Searching for: tech companies, AI" in explanation
        assert "company = Apple" in explanation
        assert "revenue ≥ 100" in explanation

        # Test with no results
        explanation = parser._generate_explanation("test", [], {})
        assert "Treating entire query as semantic search" in explanation


class TestLLMBasedParsing:
    """Test that parsing uses LLM, not regex."""

    def test_no_regex_imports(self):
        """Verify that api.py doesn't import regex."""
        import inspect

        import langstruct.api as api_module

        source = inspect.getsource(api_module)

        # Should not have regex imports
        assert "import re\n" not in source
        assert "from re import" not in source

    def test_no_hardcoded_patterns(self):
        """Verify no hardcoded patterns in query method."""
        import inspect

        import langstruct.api as api_module

        # Check both query and _query_single methods since query delegates to _query_single
        query_source = inspect.getsource(api_module.LangStruct.query)
        query_single_source = inspect.getsource(api_module.LangStruct._query_single)

        # Should not have regex patterns in either method
        combined_source = query_source + query_single_source
        assert "r'(Q[1-4])" not in combined_source
        assert ".replace('$', '')" not in combined_source
        assert "over|above|greater than" not in combined_source

        # Should use QueryParser (in _query_single method)
        assert (
            "QueryParser" in query_single_source
            or "_query_parser" in query_single_source
        )
