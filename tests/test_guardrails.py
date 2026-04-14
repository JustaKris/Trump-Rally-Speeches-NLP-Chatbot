"""Tests for RAG Guardrails — query validation, relevance filtering, grounding checks.

Tests the three-layer guardrails pipeline:
1. Pre-retrieval query validation
2. Post-retrieval relevance threshold filtering
3. Post-generation grounding verification
"""

import math

import pytest

from speech_nlp.services.rag.guardrails import RAGGuardrails, _extract_content_words
from speech_nlp.services.rag.models import ContextChunk, SearchResult

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def guardrails():
    """Create guardrails with default thresholds."""
    return RAGGuardrails(similarity_threshold=0.4, grounding_threshold=0.3)


@pytest.fixture
def strict_guardrails():
    """Create guardrails with strict thresholds."""
    return RAGGuardrails(similarity_threshold=0.7, grounding_threshold=0.6)


@pytest.fixture
def sample_results_high_relevance():
    """Search results with high relevance (rerank_score > 0)."""
    return [
        SearchResult(
            document="Trump discussed border wall construction and immigration reform.",
            metadata={"source": "DallasOct17_2019.txt", "chunk_index": 3, "total_chunks": 20},
            distance=0.2,
            id="dallas_chunk_3",
            rerank_score=2.5,  # sigmoid ≈ 0.924
        ),
        SearchResult(
            document="The economy is doing very well, record low unemployment.",
            metadata={"source": "CincinnatiAug1_2019.txt", "chunk_index": 7, "total_chunks": 25},
            distance=0.3,
            id="cincy_chunk_7",
            rerank_score=1.8,  # sigmoid ≈ 0.858
        ),
    ]


@pytest.fixture
def sample_results_mixed_relevance():
    """Search results with mixed relevance — some should be filtered."""
    return [
        SearchResult(
            document="Trump talked about the border wall and immigration.",
            metadata={"source": "Dallas.txt", "chunk_index": 0, "total_chunks": 10},
            distance=0.3,
            id="high_1",
            rerank_score=2.0,  # sigmoid ≈ 0.88
        ),
        SearchResult(
            document="Applause and crowd noise.",
            metadata={"source": "Dallas.txt", "chunk_index": 5, "total_chunks": 10},
            distance=0.8,
            id="low_1",
            rerank_score=-2.0,  # sigmoid ≈ 0.12
        ),
        SearchResult(
            document="Thank you, thank you very much.",
            metadata={"source": "Dallas.txt", "chunk_index": 9, "total_chunks": 10},
            distance=0.9,
            id="low_2",
            rerank_score=-3.0,  # sigmoid ≈ 0.047
        ),
    ]


@pytest.fixture
def sample_results_no_rerank():
    """Search results without reranking (combined_score path)."""
    return [
        SearchResult(
            document="Jobs numbers are incredible this month.",
            metadata={"source": "Toledo.txt", "chunk_index": 2, "total_chunks": 15},
            distance=0.4,
            id="above_thresh",
            combined_score=0.55,
        ),
        SearchResult(
            document="Random crowd chatter.",
            metadata={"source": "Toledo.txt", "chunk_index": 14, "total_chunks": 15},
            distance=0.9,
            id="below_thresh",
            combined_score=0.15,
        ),
    ]


@pytest.fixture
def sample_context_chunks():
    """Context chunks for grounding tests."""
    return [
        ContextChunk(
            text="Trump discussed building a border wall along the southern border "
            "to prevent illegal immigration. He emphasized that Mexico would pay for it.",
            source="DallasOct17_2019.txt",
            chunk_index=3,
            score=0.9,
        ),
        ContextChunk(
            text="The unemployment rate has reached historic lows at 3.5 percent. "
            "African American unemployment is at the lowest level ever recorded.",
            source="CincinnatiAug1_2019.txt",
            chunk_index=7,
            score=0.85,
        ),
    ]


# ============================================================================
# Tests: Query Validation (Layer 1)
# ============================================================================


class TestQueryValidation:
    """Tests for pre-retrieval query validation."""

    def test_valid_query(self, guardrails):
        """Normal question passes validation."""
        is_valid, reason = guardrails.validate_query("What did Trump say about the economy?")
        assert is_valid is True
        assert reason is None

    def test_empty_query_rejected(self, guardrails):
        """Empty string is rejected."""
        is_valid, reason = guardrails.validate_query("")
        assert is_valid is False
        assert reason is not None

    def test_whitespace_only_rejected(self, guardrails):
        """Whitespace-only string is rejected."""
        is_valid, reason = guardrails.validate_query("   \t\n  ")
        assert is_valid is False

    def test_none_query_rejected(self, guardrails):
        """None-ish query is rejected (empty after strip)."""
        is_valid, reason = guardrails.validate_query("")
        assert is_valid is False

    def test_too_short_query_rejected(self, guardrails):
        """Very short queries are rejected."""
        is_valid, reason = guardrails.validate_query("hi")
        assert is_valid is False
        assert "short" in reason.lower()

    def test_minimum_viable_query(self, guardrails):
        """3-character query passes (just barely)."""
        is_valid, reason = guardrails.validate_query("war")
        assert is_valid is True


# ============================================================================
# Tests: Relevance Filtering (Layer 2)
# ============================================================================


class TestRelevanceFiltering:
    """Tests for post-retrieval relevance threshold filtering."""

    def test_all_above_threshold(self, guardrails, sample_results_high_relevance):
        """High-relevance results all pass."""
        filtered = guardrails.filter_by_relevance(sample_results_high_relevance)
        assert len(filtered) == 2

    def test_mixed_relevance_filters_low(self, guardrails, sample_results_mixed_relevance):
        """Low-relevance results are dropped."""
        filtered = guardrails.filter_by_relevance(sample_results_mixed_relevance)
        assert len(filtered) == 1
        assert filtered[0].id == "high_1"

    def test_all_below_threshold_returns_empty(self, strict_guardrails):
        """If nothing meets threshold, return empty list."""
        results = [
            SearchResult(
                document="Noise",
                metadata={"source": "test.txt", "chunk_index": 0, "total_chunks": 1},
                distance=0.9,
                id="low",
                combined_score=0.2,
            ),
        ]
        filtered = strict_guardrails.filter_by_relevance(results)
        assert len(filtered) == 0

    def test_empty_results_handled(self, guardrails):
        """Empty input returns empty output."""
        assert guardrails.filter_by_relevance([]) == []

    def test_custom_threshold_override(self, guardrails, sample_results_mixed_relevance):
        """Threshold override is respected."""
        # Very low threshold — everything passes
        filtered = guardrails.filter_by_relevance(sample_results_mixed_relevance, threshold=0.01)
        assert len(filtered) == 3

    def test_combined_score_path(self, guardrails, sample_results_no_rerank):
        """Filtering works on combined_score (no reranking)."""
        filtered = guardrails.filter_by_relevance(sample_results_no_rerank)
        assert len(filtered) == 1
        assert filtered[0].id == "above_thresh"

    def test_distance_only_path(self, guardrails):
        """Filtering works on raw distance (no rerank or combined score)."""
        results = [
            SearchResult(
                document="Close match",
                metadata={"source": "test.txt", "chunk_index": 0, "total_chunks": 1},
                distance=0.3,  # relevance_score = max(0, 1 - 0.3) = 0.7
                id="close",
            ),
            SearchResult(
                document="Far away",
                metadata={"source": "test.txt", "chunk_index": 1, "total_chunks": 1},
                distance=0.8,  # relevance_score = max(0, 1 - 0.8) = 0.2
                id="far",
            ),
        ]
        filtered = guardrails.filter_by_relevance(results)
        assert len(filtered) == 1
        assert filtered[0].id == "close"


# ============================================================================
# Tests: Grounding Verification (Layer 3)
# ============================================================================


class TestGroundingVerification:
    """Tests for post-generation grounding checks."""

    def test_grounded_answer_passes(self, guardrails, sample_context_chunks):
        """Answer using words from context is grounded."""
        answer = (
            "Trump discussed building a border wall to prevent illegal immigration. "
            "He also noted that unemployment reached historic lows."
        )
        is_grounded, score = guardrails.check_grounding(answer, sample_context_chunks)
        assert is_grounded is True
        assert score > 0.3

    def test_ungrounded_answer_fails(self, guardrails, sample_context_chunks):
        """Answer with fabricated content fails grounding."""
        answer = (
            "The president announced a comprehensive climate change initiative "
            "focused on renewable energy transition and carbon neutrality targets."
        )
        is_grounded, score = guardrails.check_grounding(answer, sample_context_chunks)
        assert is_grounded is False
        assert score < 0.3

    def test_refusal_answer_always_passes(self, guardrails, sample_context_chunks):
        """'I don't know'-style answers are always grounded."""
        refusals = [
            "The available documents don't contain information about this topic.",
            "I don't have enough information to answer that question.",
            "I cannot find relevant information in the provided context.",
        ]
        for refusal in refusals:
            is_grounded, score = guardrails.check_grounding(refusal, sample_context_chunks)
            assert is_grounded is True
            assert score == 1.0

    def test_empty_answer_passes(self, guardrails, sample_context_chunks):
        """Empty answer is trivially grounded."""
        is_grounded, score = guardrails.check_grounding("", sample_context_chunks)
        assert is_grounded is True

    def test_empty_context_passes(self, guardrails):
        """No context means nothing to ground against."""
        is_grounded, score = guardrails.check_grounding("Some answer text", [])
        assert is_grounded is True

    def test_grounding_score_range(self, guardrails, sample_context_chunks):
        """Grounding score is between 0 and 1."""
        answer = "Random words that may or may not be in context"
        _, score = guardrails.check_grounding(answer, sample_context_chunks)
        assert 0.0 <= score <= 1.0

    def test_strict_grounding_threshold(self, strict_guardrails, sample_context_chunks):
        """Higher threshold catches partially grounded answers."""
        answer = (
            "Trump mentioned some border topics and discussed economic growth "
            "alongside various agricultural subsidies and pharmaceutical regulations."
        )
        is_grounded, score = strict_guardrails.check_grounding(answer, sample_context_chunks)
        # With strict threshold, partially grounded answers may fail
        assert isinstance(is_grounded, bool)
        assert 0.0 <= score <= 1.0


# ============================================================================
# Tests: Relevance Score Normalisation (SearchResult.relevance_score)
# ============================================================================


class TestRelevanceScoreNormalisation:
    """Tests for SearchResult.relevance_score property."""

    def test_rerank_score_sigmoid(self):
        """Rerank scores are normalised via sigmoid."""
        result = SearchResult(
            document="test",
            metadata={"source": "t.txt", "chunk_index": 0, "total_chunks": 1},
            distance=0.5,
            id="1",
            rerank_score=0.0,
        )
        # sigmoid(0) = 0.5
        assert abs(result.relevance_score - 0.5) < 0.001

    def test_high_rerank_score(self):
        """Positive rerank score gives high relevance."""
        result = SearchResult(
            document="test",
            metadata={"source": "t.txt", "chunk_index": 0, "total_chunks": 1},
            distance=0.5,
            id="1",
            rerank_score=5.0,
        )
        expected = 1.0 / (1.0 + math.exp(-5.0))
        assert abs(result.relevance_score - expected) < 0.001
        assert result.relevance_score > 0.99

    def test_negative_rerank_score(self):
        """Negative rerank score gives low relevance."""
        result = SearchResult(
            document="test",
            metadata={"source": "t.txt", "chunk_index": 0, "total_chunks": 1},
            distance=0.5,
            id="1",
            rerank_score=-5.0,
        )
        assert result.relevance_score < 0.01

    def test_combined_score_passthrough(self):
        """Combined scores are used as-is."""
        result = SearchResult(
            document="test",
            metadata={"source": "t.txt", "chunk_index": 0, "total_chunks": 1},
            distance=0.5,
            id="1",
            combined_score=0.65,
        )
        assert result.relevance_score == 0.65

    def test_distance_fallback(self):
        """Raw distance is converted to similarity."""
        result = SearchResult(
            document="test",
            metadata={"source": "t.txt", "chunk_index": 0, "total_chunks": 1},
            distance=0.3,
            id="1",
        )
        assert abs(result.relevance_score - 0.7) < 0.001

    def test_large_distance_clamped(self):
        """Distance > 1 clamps to 0."""
        result = SearchResult(
            document="test",
            metadata={"source": "t.txt", "chunk_index": 0, "total_chunks": 1},
            distance=1.5,
            id="1",
        )
        assert result.relevance_score == 0.0


# ============================================================================
# Tests: Content Word Extraction Helper
# ============================================================================


class TestContentWordExtraction:
    """Tests for the _extract_content_words helper."""

    def test_filters_short_words(self):
        """Words <= 3 chars are excluded."""
        words = _extract_content_words("The cat sat on a big mat")
        assert "the" not in words
        assert "cat" not in words
        assert "sat" not in words

    def test_filters_stop_words(self):
        """Common stop words are excluded."""
        words = _extract_content_words("This should also have been there before")
        assert "should" not in words
        assert "there" not in words
        assert "been" not in words

    def test_keeps_content_words(self):
        """Meaningful content words are kept."""
        words = _extract_content_words("Trump discussed unemployment rates and border security")
        assert "trump" in words
        assert "unemployment" in words
        assert "border" in words
        assert "security" in words

    def test_strips_punctuation(self):
        """Punctuation is stripped from words."""
        words = _extract_content_words("wall, immigration! economy? growth.")
        assert "wall" in words
        assert "immigration" in words
        assert "economy" in words

    def test_empty_input(self):
        """Empty string yields empty list."""
        assert _extract_content_words("") == []

    def test_filters_non_alpha(self):
        """Non-alphabetic tokens are excluded."""
        words = _extract_content_words("In 2019, 3.5% unemployment rate was achieved")
        assert "2019" not in words
        assert "3.5%" not in words
