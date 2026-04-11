"""RAG Guardrails — relevance filtering, query validation, and grounding checks.

Prevents hallucination by filtering low-relevance results before they reach the LLM,
validates queries for minimum viability, and verifies that generated answers stay
grounded in the retrieved context.
"""

import logging
from typing import List, Optional, Tuple

from .models import ContextChunk, SearchResult

logger = logging.getLogger(__name__)

# Common English stop words excluded from grounding overlap checks
_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "been",
        "before",
        "between",
        "both",
        "came",
        "come",
        "could",
        "does",
        "each",
        "even",
        "from",
        "going",
        "gotten",
        "have",
        "here",
        "into",
        "just",
        "know",
        "like",
        "made",
        "make",
        "many",
        "more",
        "most",
        "much",
        "must",
        "only",
        "other",
        "over",
        "really",
        "said",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "through",
        "very",
        "want",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
        "your",
        "based",
        "rally",
        "speech",
        "document",
        "available",
        "information",
        "according",
        "mentioned",
        "discussed",
        "context",
        "provided",
        "question",
    }
)


class RAGGuardrails:
    """Guardrails for the RAG pipeline.

    Three-layer protection:
    1. **Pre-retrieval** — validate the incoming query
    2. **Post-retrieval** — filter results below a relevance threshold
    3. **Post-generation** — verify the answer is grounded in context
    """

    def __init__(
        self,
        similarity_threshold: float = 0.01,
        grounding_threshold: float = 0.3,
    ):
        """Initialise guardrails with configurable thresholds."""
        self.similarity_threshold = similarity_threshold
        self.grounding_threshold = grounding_threshold
        logger.info(
            f"RAGGuardrails initialized: similarity_threshold={similarity_threshold}, "
            f"grounding_threshold={grounding_threshold}"
        )

    # ------------------------------------------------------------------
    # Layer 1: Pre-retrieval query validation
    # ------------------------------------------------------------------

    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check whether a query is viable for retrieval.

        Returns:
            (is_valid, rejection_reason) — reason is None when valid.
        """
        stripped = query.strip() if query else ""

        if not stripped:
            return False, "Please provide a question."

        if len(stripped) < 3:
            return False, "Query is too short — please ask a more detailed question."

        return True, None

    # ------------------------------------------------------------------
    # Layer 2: Post-retrieval relevance filtering
    # ------------------------------------------------------------------

    def filter_by_relevance(
        self,
        results: List[SearchResult],
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Drop search results whose normalised relevance score is below *threshold*.

        Uses ``SearchResult.relevance_score`` which applies sigmoid normalisation
        to cross-encoder logits so that all scoring paths (semantic, hybrid,
        re-ranked) produce a comparable 0-1 value.

        Args:
            results: Raw search results (already re-ranked if applicable).
            threshold: Override the instance default.

        Returns:
            Filtered list (may be empty).
        """
        if not results:
            return results

        thresh = threshold if threshold is not None else self.similarity_threshold
        filtered = [r for r in results if r.relevance_score >= thresh]

        dropped = len(results) - len(filtered)
        if dropped:
            logger.info(
                f"Relevance filter: kept {len(filtered)}/{len(results)} results "
                f"(threshold={thresh:.2f}, dropped {dropped})"
            )

        if not filtered:
            scores = [round(r.relevance_score, 3) for r in results]
            logger.warning(
                f"All {len(results)} results below relevance threshold {thresh} (scores: {scores})"
            )

        return filtered

    # ------------------------------------------------------------------
    # Layer 3: Post-generation grounding verification
    # ------------------------------------------------------------------

    def check_grounding(
        self,
        answer: str,
        context_chunks: List[ContextChunk],
    ) -> Tuple[bool, float]:
        """Verify that the generated answer is grounded in the retrieved context.

        Uses token-overlap between the answer's content words and the combined
        context text.  This is a lightweight heuristic — not a full NLI check —
        but catches obvious cases where the LLM ignores context entirely.

        Returns:
            (is_grounded, overlap_score) where overlap_score is 0-1.
        """
        if not answer or not context_chunks:
            return True, 1.0  # Nothing to verify

        # Short-circuit for "I don't know"-style answers
        refusal_phrases = [
            "don't contain information",
            "don't have enough information",
            "no relevant information",
            "unable to generate",
            "not in the context",
            "cannot find",
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in refusal_phrases):
            return True, 1.0  # Refusal is always grounded

        context_text = " ".join(c.text for c in context_chunks).lower()
        content_words = _extract_content_words(answer)

        if not content_words:
            return True, 1.0

        grounded_count = sum(1 for w in content_words if w in context_text)
        score = grounded_count / len(content_words)

        is_grounded = score >= self.grounding_threshold

        if not is_grounded:
            logger.warning(
                f"Grounding check failed: {score:.2f} < {self.grounding_threshold} "
                f"({grounded_count}/{len(content_words)} content words found in context)"
            )

        return is_grounded, round(score, 3)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _extract_content_words(text: str) -> List[str]:
    """Extract meaningful content words from text for grounding comparison.

    Strips punctuation, filters stop words and short tokens.
    """
    words = text.lower().split()
    return [
        w.strip(".,!?;:\"'()[]{}—–-")
        for w in words
        if len(w.strip(".,!?;:\"'()[]{}—–-")) > 3
        and w.strip(".,!?;:\"'()[]{}—–-").lower() not in _STOP_WORDS
        and w.strip(".,!?;:\"'()[]{}—–-").isalpha()
    ]
