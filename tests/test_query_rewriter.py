"""Tests for QueryRewriter component.

Tests LLM-powered query rewriting for improved search retrieval.
"""

from unittest.mock import MagicMock, Mock

import pytest

from src.services.rag.query_rewriter import QueryRewriter


class TestQueryRewriter:
    """Test suite for QueryRewriter."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = Mock()
        response = Mock()
        response.text = "What were the economic policies discussed in the rally speeches?"
        llm.generate_content = MagicMock(return_value=response)
        return llm

    @pytest.fixture
    def rewriter(self, mock_llm):
        """Create a QueryRewriter with a mock LLM."""
        return QueryRewriter(llm=mock_llm, enabled=True)

    @pytest.fixture
    def disabled_rewriter(self, mock_llm):
        """Create a disabled QueryRewriter."""
        return QueryRewriter(llm=mock_llm, enabled=False)

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------

    def test_rewrite_returns_rewritten_query(self, rewriter, mock_llm):
        """Test that rewrite returns the LLM-generated rewrite."""
        result = rewriter.rewrite("whats the econimc polcies?")
        assert result == "What were the economic policies discussed in the rally speeches?"
        mock_llm.generate_content.assert_called_once()

    def test_rewrite_passes_temperature_zero(self, rewriter, mock_llm):
        """Test that rewrite uses temperature=0 for deterministic output."""
        rewriter.rewrite("some query")
        call_kwargs = mock_llm.generate_content.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    def test_rewrite_passes_max_tokens_256(self, rewriter, mock_llm):
        """Test that rewrite limits output tokens to 256."""
        rewriter.rewrite("some query")
        call_kwargs = mock_llm.generate_content.call_args
        assert call_kwargs.kwargs["max_tokens"] == 256

    def test_prompt_contains_original_query(self, rewriter, mock_llm):
        """Test that the prompt includes the original query."""
        rewriter.rewrite("what about the wall?")
        prompt_arg = mock_llm.generate_content.call_args.args[0]
        assert "what about the wall?" in prompt_arg

    # ------------------------------------------------------------------
    # Disabled / passthrough behaviour
    # ------------------------------------------------------------------

    def test_disabled_rewriter_returns_original(self, disabled_rewriter, mock_llm):
        """Test that a disabled rewriter returns the original query unchanged."""
        result = disabled_rewriter.rewrite("some typo query")
        assert result == "some typo query"
        mock_llm.generate_content.assert_not_called()

    def test_empty_query_returns_unchanged(self, rewriter, mock_llm):
        """Test that an empty query is returned unchanged."""
        result = rewriter.rewrite("")
        assert result == ""
        mock_llm.generate_content.assert_not_called()

    def test_whitespace_only_query_returns_unchanged(self, rewriter, mock_llm):
        """Test that a whitespace-only query is returned unchanged."""
        result = rewriter.rewrite("   ")
        assert result == "   "
        mock_llm.generate_content.assert_not_called()

    # ------------------------------------------------------------------
    # Error handling / fallback
    # ------------------------------------------------------------------

    def test_llm_error_returns_original_query(self, rewriter, mock_llm):
        """Test that an LLM error falls back to the original query."""
        mock_llm.generate_content.side_effect = RuntimeError("API timeout")
        result = rewriter.rewrite("what about the wall?")
        assert result == "what about the wall?"

    def test_empty_llm_response_returns_original(self, rewriter, mock_llm):
        """Test that an empty LLM response falls back to the original query."""
        empty_response = Mock()
        empty_response.text = ""
        mock_llm.generate_content.return_value = empty_response
        result = rewriter.rewrite("what about the wall?")
        assert result == "what about the wall?"

    def test_suspiciously_long_rewrite_returns_original(self, rewriter, mock_llm):
        """Test that a rewrite >5x the original length is rejected."""
        long_response = Mock()
        long_response.text = "A " * 500  # way too long
        mock_llm.generate_content.return_value = long_response
        result = rewriter.rewrite("short query")
        assert result == "short query"

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    def test_extract_text_with_text_attribute(self):
        """Test _extract_text with standard .text response."""
        response = Mock()
        response.text = "rewritten query"
        assert QueryRewriter._extract_text(response) == "rewritten query"

    def test_extract_text_without_text_attribute(self):
        """Test _extract_text fallback to str()."""
        result = QueryRewriter._extract_text("plain string response")
        assert result == "plain string response"

    def test_rewrite_strips_whitespace(self, rewriter, mock_llm):
        """Test that rewritten queries are stripped of leading/trailing whitespace."""
        padded_response = Mock()
        padded_response.text = "  cleaned query  "
        mock_llm.generate_content.return_value = padded_response
        result = rewriter.rewrite("messy query")
        assert result == "cleaned query"

    # ------------------------------------------------------------------
    # Unchanged query (no-op rewrite)
    # ------------------------------------------------------------------

    def test_identical_rewrite_returns_original(self, rewriter, mock_llm):
        """Test that when the LLM returns the same text, it's returned unchanged."""
        same_response = Mock()
        same_response.text = "already good query"
        mock_llm.generate_content.return_value = same_response
        result = rewriter.rewrite("already good query")
        assert result == "already good query"

    # ------------------------------------------------------------------
    # Integration with RAG pipeline patterns
    # ------------------------------------------------------------------

    def test_rewrite_with_typo_correction(self, rewriter, mock_llm):
        """Test typical typo-correction scenario."""
        corrected = Mock()
        corrected.text = "What did Trump say about immigration?"
        mock_llm.generate_content.return_value = corrected

        result = rewriter.rewrite("waht did trump say abotu immigraton?")
        assert result == "What did Trump say about immigration?"

    def test_rewrite_with_abbreviation_expansion(self, rewriter, mock_llm):
        """Test abbreviation expansion scenario."""
        expanded = Mock()
        expanded.text = "What did Trump say about the Republican Party and the Democratic Party?"
        mock_llm.generate_content.return_value = expanded

        result = rewriter.rewrite("what about GOP and dems?")
        assert result == "What did Trump say about the Republican Party and the Democratic Party?"
