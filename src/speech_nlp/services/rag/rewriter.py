"""Query Rewriter — LLM-powered query optimization for better retrieval.

Rewrites user queries to be more effective for semantic search over political
speech transcripts. Fixes typos, expands abbreviations, adds relevant terms,
and reformulates vague phrasing to improve retrieval recall.
"""

import logging

from speech_nlp.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
You are a search query cleaner for a political speech transcript database.
Your job is to fix surface-level issues in the query so semantic search works well.

Rules:
- Fix typos, spelling mistakes, and grammatical errors
- Expand abbreviations and acronyms (e.g. "GOP" → "Republican Party")
- If the query is already clear and well-formed, return it EXACTLY as-is
- Do NOT add extra terms, synonyms, or related concepts
- Do NOT broaden or narrow the scope of the query
- Do NOT restructure a query that is already a coherent question
- Keep the rewrite as close to the original as possible
- Return ONLY the cleaned query — no explanation, no preamble, no quotes
- Do NOT answer the question — just clean it up

Original query: {query}

Cleaned query:"""


class QueryRewriter:
    """Rewrites user queries using an LLM to improve search retrieval.

    Designed to sit in the RAG pipeline between query validation and search,
    transparently improving query quality before it hits the embedding model.
    """

    def __init__(self, llm: LLMProvider, enabled: bool = True) -> None:
        """Initialize the query rewriter.

        Args:
            llm: LLM provider instance for query rewriting.
            enabled: Whether query rewriting is active. When disabled,
                     ``rewrite()`` returns the original query unchanged.
        """
        self.llm = llm
        self.enabled = enabled

    def rewrite(self, query: str) -> str:
        """Rewrite a query for better semantic search retrieval.

        Args:
            query: The original user query.

        Returns:
            The rewritten query, or the original if rewriting is disabled
            or fails.
        """
        if not self.enabled:
            logger.debug("Query rewriting disabled — passing through unchanged")
            return query

        original = query.strip()
        if not original:
            return query

        logger.debug("Rewriter input: %r", original)

        try:
            prompt = _REWRITE_PROMPT.format(query=original)
            response = self.llm.generate_content(
                prompt,
                temperature=0.0,  # deterministic rewrites
                max_tokens=256,
            )

            rewritten = self._extract_text(response).strip()

            if not rewritten:
                logger.warning("LLM returned empty rewrite — using original query")
                return original

            # Guard against the LLM returning an answer instead of a rewrite
            if len(rewritten) > len(original) * 5:
                logger.warning(
                    "Rewrite suspiciously long (%d chars vs original %d) — using original",
                    len(rewritten),
                    len(original),
                )
                return original

            if rewritten != original:
                logger.info("Query rewritten: %r → %r", original, rewritten)
            else:
                logger.debug("Query unchanged after rewrite: %r", original)

            return rewritten

        except Exception as e:
            logger.warning("Query rewrite failed (%s) — using original query", e)
            return original

    @staticmethod
    def _extract_text(response: object) -> str:
        """Extract text from an LLM response object.

        Handles the common ``.text`` attribute used by Gemini/OpenAI wrappers.
        """
        if hasattr(response, "text"):
            return response.text  # type: ignore[union-attr]
        return str(response)
