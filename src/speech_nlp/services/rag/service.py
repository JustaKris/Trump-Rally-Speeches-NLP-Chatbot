"""RAG (Retrieval-Augmented Generation) Service - Refactored Version.

This is the streamlined version that delegates to modular components.
Provides semantic search and question-answering capabilities over text documents
using ChromaDB for vector storage, sentence-transformers for embeddings,
LLM providers for answer generation, and hybrid search for improved retrieval.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from speech_nlp.services.llm.base import LLMProvider
from speech_nlp.services.rag.chunking import DocumentLoader
from speech_nlp.services.rag.confidence import ConfidenceCalculator
from speech_nlp.services.rag.entities import EntityAnalyzer
from speech_nlp.services.rag.guardrails import RAGGuardrails
from speech_nlp.services.rag.models import EntityMatch
from speech_nlp.services.rag.rewriter import QueryRewriter
from speech_nlp.services.rag.search import SearchEngine

if TYPE_CHECKING:
    from speech_nlp.services.cache import CacheService

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logger = logging.getLogger(__name__)


class RAGService:
    """Refactored RAG service that delegates to modular components.

    This version maintains backward compatibility while using the new
    modular architecture under the hood.
    """

    def __init__(
        self,
        collection_name: str = "speeches",
        persist_directory: str = "./data/chromadb",
        embedding_model: str = "all-mpnet-base-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 2048,
        chunk_overlap: int = 150,
        llm_service: Optional[LLMProvider] = None,
        use_reranking: bool = True,
        use_hybrid_search: bool = True,
        chunking_strategy: Literal["fixed", "semantic"] = "semantic",
        semantic_min_chunk_size: int = 256,
        semantic_similarity_threshold: Optional[float] = None,
        semantic_breakpoint_percentile: float = 90.0,
        guardrails_enabled: bool = True,
        similarity_threshold: float = 0.01,
        grounding_threshold: float = 0.3,
        query_rewriting_enabled: bool = True,
        cache_service: Optional["CacheService"] = None,
        use_ner: bool = True,
        ner_model: str = "en_core_web_sm",
    ):
        """Initialize RAG service with modular components.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for ChromaDB persistence
            embedding_model: HuggingFace model for embeddings
            reranker_model: Cross-encoder model for re-ranking
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            llm_service: Optional LLM service instance
            use_reranking: Use cross-encoder for re-ranking results
            use_hybrid_search: Combine semantic and keyword search
            chunking_strategy: "fixed" or "semantic" chunking
            semantic_min_chunk_size: Minimum chunk size for semantic chunking
            semantic_similarity_threshold: Absolute similarity threshold (or None for percentile)
            semantic_breakpoint_percentile: Percentile for breakpoint detection
            guardrails_enabled: Enable relevance filtering and grounding checks
            similarity_threshold: Min normalised relevance score to keep a result (0-1)
            grounding_threshold: Min token-overlap ratio for grounding verification (0-1)
            query_rewriting_enabled: Enable LLM-powered query rewriting for better retrieval
            cache_service: Optional cache service for response caching
            use_ner: Enable spaCy NER for entity extraction (falls back to heuristics if unavailable)
            ner_model: spaCy model name to use for NER (default ``en_core_web_sm``)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.use_reranking = use_reranking
        self.use_hybrid_search = use_hybrid_search
        self.guardrails_enabled = guardrails_enabled
        # Store chunk parameters as properties for backward compatibility
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.debug(f"Initializing RAG service: collection={collection_name}")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Set LLM service
        self.llm = llm_service
        if llm_service:
            logger.info("RAG service using LLM for answer generation")
        else:
            logger.info("RAG service using extraction-based answers (no LLM)")

        # Initialize ChromaDB client with persistence
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Text documents for semantic search"},
        )

        # Initialize modular components
        self.document_loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            embedding_model=self.embedding_model,
            min_chunk_size=semantic_min_chunk_size,
            similarity_threshold=semantic_similarity_threshold,
            breakpoint_percentile=semantic_breakpoint_percentile,
        )

        self.search_engine = SearchEngine(
            embedding_model=self.embedding_model,
            collection=self.collection,
            reranker_model=reranker_model,
            use_reranking=use_reranking,
            use_hybrid_search=use_hybrid_search,
        )

        self.entity_analyzer = EntityAnalyzer(
            collection=self.collection,
            sentiment_analyzer=None,  # Can be set later via property
            use_ner=use_ner,
            ner_model=ner_model,
        )

        self.confidence_calculator = ConfidenceCalculator()

        # Initialize guardrails (relevance filtering + grounding verification)
        self.guardrails: Optional[RAGGuardrails] = None
        if guardrails_enabled:
            self.guardrails = RAGGuardrails(
                similarity_threshold=similarity_threshold,
                grounding_threshold=grounding_threshold,
            )

        # Initialize query rewriter (LLM-powered query optimization)
        self.query_rewriter: Optional[QueryRewriter] = None
        if query_rewriting_enabled and llm_service:
            self.query_rewriter = QueryRewriter(llm=llm_service, enabled=True)
            logger.info("✓ Query rewriter enabled")
        elif query_rewriting_enabled and not llm_service:
            logger.info("Query rewriting requested but no LLM available — disabled")

        # Initialize response cache (Redis-backed with in-memory fallback)
        self.cache_service: Optional["CacheService"] = cache_service
        if cache_service:
            logger.info("✓ Response caching enabled")

        # Check if collection has existing data
        count = self.collection.count()
        if count > 0:
            logger.info(f"✓ RAG service initialized with {count} existing chunks")
            # Initialize BM25 with existing documents if using hybrid search
            if use_hybrid_search:
                self._initialize_bm25_from_collection()

    def _initialize_bm25_from_collection(self):
        """Initialize BM25 index from existing ChromaDB collection."""
        try:
            # Get all documents from collection (ChromaDB returns dict with 'documents' key)
            result = self.collection.get()
            if result and result.get("documents"):
                documents = result["documents"]
                # Ensure documents is a list of strings
                if isinstance(documents, list) and all(isinstance(d, str) for d in documents):
                    ids: List[str] = result.get("ids") or []
                    # Explicit cast: ChromaDB Metadata is Dict[str, str|int|float|bool]
                    # which satisfies Dict[str, Any] — annotate to satisfy the type checker.
                    metadatas: List[Dict[str, Any]] = result.get("metadatas") or []  # type: ignore[assignment]
                    self.search_engine.initialize_bm25(documents, ids=ids, metadatas=metadatas)
                    logger.info(f"✓ BM25 index initialized with {len(documents)} documents")
        except Exception as e:
            logger.warning(f"Could not initialize BM25 from existing collection: {e}")

    @property
    def sentiment_analyzer(self):
        """Get sentiment analyzer."""
        return self.entity_analyzer.sentiment_analyzer

    @sentiment_analyzer.setter
    def sentiment_analyzer(self, analyzer):
        """Set sentiment analyzer for entity analysis."""
        self.entity_analyzer.sentiment_analyzer = analyzer

    def load_documents(self, data_dir: str = "data/Donald Trump Rally Speeches") -> int:
        """Load and index all text documents from a directory.

        Args:
            data_dir: Directory containing text files

        Returns:
            Number of documents indexed
        """
        # Use DocumentLoader to load and chunk documents
        chunks, metadatas, ids = self.document_loader.load_from_directory(data_dir)

        if not chunks:
            logger.warning(f"No documents found in {data_dir}")
            return 0

        logger.info(f"Loaded {len(chunks)} chunks from {data_dir}")

        # Generate embeddings for all chunks
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            chunks, show_progress_bar=True, convert_to_numpy=True
        ).tolist()

        # Add to ChromaDB
        logger.info("Adding documents to vector store...")
        # Type ignore needed due to ChromaDB's loose typing
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,  # type: ignore[arg-type]
            ids=ids,
        )

        # Initialize BM25 for hybrid search
        if self.use_hybrid_search:
            self.search_engine.initialize_bm25(chunks)
            logger.info("✓ BM25 index initialized")

        count = self.collection.count()
        logger.info(f"✓ Total documents in collection: {count}")

        # Count unique source files
        unique_sources = len({m["source"] for m in metadatas})
        return unique_sources

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using the configured search strategy.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Delegate to SearchEngine
            results = self.search_engine.search(query, top_k)

            # Convert SearchResult objects to dictionaries for backward compatibility
            return [
                {
                    "document": r.document,
                    "metadata": r.metadata,
                    "distance": r.distance,
                    "id": r.id,
                    "rerank_score": r.rerank_score,
                    "combined_score": r.combined_score,
                    "sources": r.sources,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _no_relevant_info_response(
        self, entity_matches: List[EntityMatch], reason: str
    ) -> Dict[str, Any]:
        """Standard response when no relevant information is found."""
        logger.info("No relevant info: %s (entities=%s)", reason, [m.text for m in entity_matches])
        return {
            "answer": "I don't have enough information to answer that question.",
            "context": [],
            "confidence": "low",
            "confidence_score": 0.0,
            "confidence_explanation": reason,
            "confidence_factors": {
                "retrieval_score": 0.0,
                "consistency": 0.0,
                "chunk_coverage": 0.0,
                "entity_coverage": 0.0,
            },
            "sources": [],
            "entities": [{"text": m.text, "label": m.label} for m in entity_matches],
            "llm_powered": self.llm is not None,
            "guardrails": {"enabled": self.guardrails_enabled, "triggered": True, "reason": reason},
        }

    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question using RAG (retrieval + generation).

        Pipeline:
        0. (Cache) Check for cached response
        1. (Guardrails) Validate the incoming query
        2. (Query Rewriting) Rewrite query for better retrieval
        3. Extract entities from the question
        4. Search for relevant context (fetch extra candidates when filtering)
        5. (Guardrails) Filter results below the relevance threshold
        6. Generate an answer via LLM (or extraction fallback)
        7. (Guardrails) Verify the answer is grounded in context
        8. Calculate confidence and build response
        9. (Cache) Store response for future queries

        Args:
            question: Question to answer
            top_k: Number of context chunks to retrieve

        Returns:
            Dictionary with answer, confidence, sources, and metadata
        """
        logger.info(f"Question: {question}")

        # --- Layer 0: Check cache for existing response ---
        if self.cache_service:
            cached_response = self.cache_service.get_response(question, top_k)
            if cached_response is not None:
                logger.info("Cache HIT - returning cached response")
                return cached_response

        # --- Layer 1: Pre-retrieval query validation ---
        if self.guardrails:
            is_valid, rejection_reason = self.guardrails.validate_query(question)
            if not is_valid:
                return self._no_relevant_info_response(
                    entity_matches=[], reason=rejection_reason or "Invalid query"
                )

        # --- Query rewriting for better retrieval ---
        original_question = question
        search_query = question  # query used for search (may be rewritten)
        if self.query_rewriter:
            search_query = self.query_rewriter.rewrite(question)

        # Extract entities from original question (user intent, not rewritten)
        entity_matches: List[EntityMatch] = self.entity_analyzer.extract_entities_with_types(
            question
        )
        entities = [m.text for m in entity_matches]  # plain strings for backward-compat internals
        logger.debug(f"Extracted entities: {[(m.text, m.label) for m in entity_matches]}")

        # Fetch extra candidates when guardrails will filter
        fetch_k = top_k * 2 if self.guardrails else top_k
        results = self.search_engine.search(search_query, fetch_k)

        if not results:
            return self._no_relevant_info_response(
                entity_matches=entity_matches, reason="No relevant documents found"
            )

        # --- Layer 2: Post-retrieval relevance filtering ---
        pre_filter_count = len(results)
        if self.guardrails:
            results = self.guardrails.filter_by_relevance(results)
            if not results:
                return self._no_relevant_info_response(
                    entity_matches=entity_matches,
                    reason=(
                        f"None of the {pre_filter_count} retrieved chunks met the "
                        f"relevance threshold — the documents may not cover this topic"
                    ),
                )

        # Keep only top_k after filtering
        results = results[:top_k]

        # Convert to ContextChunk objects for confidence calculation
        from speech_nlp.services.rag.models import ContextChunk

        context_chunks = [ContextChunk.from_search_result(r) for r in results]

        # Generate answer using LLM or extraction
        if self.llm:
            answer = self._generate_llm_answer(question, context_chunks)
        else:
            # Fallback: use the most relevant chunk
            answer = context_chunks[0].text

        # --- Layer 3: Post-generation grounding verification ---
        grounding_score: Optional[float] = None
        grounding_passed: Optional[bool] = None
        if self.guardrails and self.llm:
            grounding_passed, grounding_score = self.guardrails.check_grounding(
                answer, context_chunks
            )
            logger.debug(
                "Grounding check: score=%.3f passed=%s",
                grounding_score or 0.0,
                grounding_passed,
            )
            if not grounding_passed:
                answer += (
                    "\n\n⚠️ Note: This answer may contain information not directly "
                    "found in the available speech transcripts."
                )

        # Calculate confidence — pass typed entity matches for label-aware scoring
        confidence_result = self.confidence_calculator.calculate(
            question=question,
            search_results=results,
            context_chunks=context_chunks,
            entity_matches=entity_matches,
        )

        # Get entity statistics — always run without sentiment (fast corpus scan)
        entity_stats = None
        if entities:
            try:
                entity_stats = self.entity_analyzer.get_statistics(
                    entities, include_sentiment=False, include_associations=True
                )
            except Exception as e:
                logger.warning(f"Could not get entity statistics: {e}")

        # Build context for response (convert ContextChunk to dict)
        context_list = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score,
                "location": chunk.location,
                "date": chunk.date,
                "year": chunk.year,
            }
            for chunk in context_chunks
        ]

        # Build confidence factors dict
        confidence_factors_dict = {
            "retrieval_score": confidence_result.factors.retrieval_score,
            "consistency": confidence_result.factors.consistency,
            "chunk_coverage": confidence_result.factors.chunk_coverage,
            "entity_coverage": confidence_result.factors.entity_coverage,
        }

        # Build guardrails metadata
        guardrails_meta: Dict[str, Any] = {
            "enabled": self.guardrails_enabled,
            "triggered": False,
        }
        if self.guardrails:
            guardrails_meta["relevance_filtered"] = pre_filter_count - len(results)
            if grounding_score is not None:
                guardrails_meta["grounding_score"] = grounding_score
                guardrails_meta["grounding_passed"] = grounding_passed

        # Build response
        response = {
            "answer": answer,
            "context": context_list,
            "confidence": confidence_result.level,
            "confidence_score": confidence_result.score,
            "confidence_explanation": confidence_result.explanation,
            "confidence_factors": confidence_factors_dict,
            "sources": sorted({c.source for c in context_chunks}),  # Deduplicate and sort
            "entities": [{"text": m.text, "label": m.label} for m in entity_matches],
            "llm_powered": self.llm is not None,
            "guardrails": guardrails_meta,
        }

        # Include query rewriting metadata when the query was rewritten
        if search_query != original_question:
            response["query_rewriting"] = {
                "enabled": True,
                "original_query": original_question,
                "rewritten_query": search_query,
            }
        elif self.query_rewriter:
            response["query_rewriting"] = {"enabled": True, "rewritten": False}

        if entity_stats:
            # Serialize EntityStatistics to plain dicts and inject the NER label.
            # This makes each entry self-contained — the UI doesn't need to
            # cross-reference data.entities to find the entity type colour.
            label_map = {m.text: m.label for m in entity_matches}
            response["entity_statistics"] = {
                name: {
                    **stats.model_dump(exclude_none=False),
                    "label": label_map.get(name, "UNKNOWN"),
                }
                for name, stats in entity_stats.items()
            }

        # --- Layer 9: Cache the response for future queries ---
        if self.cache_service:
            self.cache_service.cache_response(question, top_k, response)
            response["cached"] = False  # Mark as fresh (not from cache)

        return response

    def _generate_llm_answer(self, question: str, context_chunks: List) -> str:
        """Generate answer using LLM with retrieved context.

        Args:
            question: User's question
            context_chunks: Retrieved context chunks (ContextChunk objects)

        Returns:
            Generated answer
        """
        # Convert ContextChunk objects to dicts for LLM service
        context_dicts = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "location": chunk.location,
                "date": chunk.date,
            }
            for chunk in context_chunks
        ]

        if self.llm is None:
            # No LLM available, use most relevant chunk
            return context_chunks[0].text if context_chunks else "Unable to generate answer."

        try:
            # Type narrowed: self.llm is not None here
            result = self.llm.generate_answer(question=question, context_chunks=context_dicts)
            answer = result.get("answer", "Unable to generate answer.")
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            # Fallback to most relevant chunk
            return context_chunks[0].text if context_chunks else "Unable to generate answer."

    def get_entity_statistics(
        self,
        entities: List[str],
        include_sentiment: bool = False,
        include_associations: bool = True,
    ) -> Dict[str, Any]:
        """Get statistics about entities across the corpus.

        Args:
            entities: List of entity names
            include_sentiment: Include sentiment analysis
            include_associations: Include term associations

        Returns:
            Dictionary with entity statistics
        """
        return self.entity_analyzer.get_statistics(
            entities, include_sentiment=include_sentiment, include_associations=include_associations
        )

    def clear(self):
        """Clear all documents from the collection."""
        logger.warning("Clearing all documents from collection")
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Text documents for semantic search"},
        )
        # Reset BM25
        self.search_engine.bm25 = None
        self.search_engine.bm25_corpus = []
        logger.info("✓ Collection cleared")

    def clear_collection(self) -> bool:
        """Clear all documents from the collection.

        Returns:
            True if successful
        """
        self.clear()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with collection statistics including:
            - collection_name: Name of the collection
            - total_chunks: Number of chunks in collection
            - unique_sources: Number of unique source files
            - sources: List of source filenames
            - embedding_model: Name of embedding model
            - chunk_size: Chunk size in characters
            - chunk_overlap: Chunk overlap in characters
        """
        count = self.collection.count()

        # Get unique sources
        sources = []
        unique_sources = 0
        if count > 0:
            try:
                result = self.collection.get()
                if result and result.get("metadatas"):
                    metadatas = result["metadatas"]
                    if metadatas:
                        source_set = {
                            m.get("source", "") for m in metadatas if m and isinstance(m, dict)
                        }
                        sources = sorted(source_set)
                        unique_sources = len(sources)
            except Exception as e:
                logger.warning(f"Could not get source statistics: {e}")

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "unique_sources": unique_sources,
            "sources": sources,
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def count(self) -> int:
        """Get total number of chunks in collection."""
        return self.collection.count()

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled.

        Returns:
            Dictionary with cache statistics, or None if caching is disabled
        """
        if self.cache_service:
            return self.cache_service.get_stats()
        return None

    def clear_cache(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of cached entries cleared, or 0 if caching is disabled
        """
        if self.cache_service:
            return self.cache_service.clear_all()
        return 0
