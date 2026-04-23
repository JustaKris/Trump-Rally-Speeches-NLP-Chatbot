"""Confidence calculator component for RAG service.

Calculates confidence scores and generates human-readable explanations
for RAG answer quality based on multiple factors.
"""

import logging
from typing import Dict, List, Optional

from speech_nlp.services.rag.models import (
    ConfidenceFactors,
    ConfidenceResult,
    ContextChunk,
    EntityMatch,
    SearchResult,
)

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """Calculates confidence scores for RAG answers.

    Uses multiple factors including retrieval quality, score consistency,
    chunk coverage, and entity mention frequency to assess answer confidence.
    """

    # Weight factors for confidence calculation
    RETRIEVAL_WEIGHT = 0.4  # 40% - Average retrieval score
    CONSISTENCY_WEIGHT = 0.25  # 25% - Score consistency (low variance)
    CHUNK_COVERAGE_WEIGHT = 0.2  # 20% - Number of supporting chunks
    ENTITY_COVERAGE_WEIGHT = 0.15  # 15% - Entity mention frequency

    # Specificity weights per spaCy entity label.
    # Higher = rarer / more query-specific → stronger confidence signal.
    # PERSON/NORP appear in almost every political-speech chunk (e.g. "Trump",
    # "Democrats") so they carry a weaker signal than a named law or event.
    LABEL_SPECIFICITY: Dict[str, float] = {
        "LAW": 0.95,          # "Tax Cuts and Jobs Act" — very specific
        "EVENT": 0.90,        # "Election Day" — specific
        "FAC": 0.90,          # Named facilities — specific
        "PRODUCT": 0.85,      # Named products — specific
        "WORK_OF_ART": 0.85,  # Named titles — specific
        "ORG": 0.75,          # Organizations — moderately specific
        "GPE": 0.65,          # Countries/cities — moderately specific
        "NORP": 0.55,         # "Democrats", "Chinese" — frequent in political speech
        "PERSON": 0.45,       # "Trump", "Biden" — appear in virtually every chunk
        "UNKNOWN": 0.50,      # Heuristic fallback — no type info
    }

    # Thresholds for confidence levels
    HIGH_THRESHOLD = 0.7
    MEDIUM_THRESHOLD = 0.4

    def __init__(self):
        """Initialize confidence calculator."""
        logger.debug("ConfidenceCalculator initialized")

    def calculate(
        self,
        question: str,
        context_chunks: List[ContextChunk],
        search_results: List[SearchResult],
        entity_matches: Optional[List[EntityMatch]] = None,
    ) -> ConfidenceResult:
        """Calculate confidence using multiple factors for accurate assessment.

        Factors considered:
        1. Average retrieval score (semantic similarity) - 40%
        2. Score consistency (low variance = more confident) - 25%
        3. Number of supporting chunks - 20%
        4. Entity mention frequency, weighted by label specificity - 15%

        Args:
            question: Original question
            context_chunks: Retrieved context with scores
            search_results: Raw search results
            entity_matches: Typed entities from spaCy NER (optional).  When
                provided, entity coverage is weighted by label specificity so
                ubiquitous entities (PERSON) inflate the score less than
                query-specific ones (LAW, EVENT).

        Returns:
            ConfidenceResult with score, level, and explanation
        """
        if not context_chunks:
            return ConfidenceResult(
                score=0.0,
                level="low",
                explanation="No relevant context found for the question.",
                factors=ConfidenceFactors(
                    retrieval_score=0.0,
                    consistency=0.0,
                    chunk_coverage=0,
                    entity_coverage=None,
                ),
            )

        # Factor 1: Average retrieval score (40% weight)
        scores = [c.score for c in context_chunks if c.score is not None]
        if not scores:
            scores = [0.0]

        # Normalize scores to [0, 1] range (BM25 can produce negative scores)
        # For negative scores, we clamp to 0. For very high scores, we normalize.
        max_score = max(scores)
        if max_score > 1.0:
            # Normalize all scores by dividing by max
            normalized_scores = [max(0.0, min(1.0, s / max_score)) for s in scores]
        else:
            # Clamp negative scores to 0, keep positive scores as-is
            normalized_scores = [max(0.0, min(1.0, s)) for s in scores]

        avg_score = sum(normalized_scores) / len(normalized_scores)
        score_factor = avg_score * self.RETRIEVAL_WEIGHT

        # Factor 2: Score consistency - low variance is good (25% weight)
        if len(normalized_scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in normalized_scores) / len(normalized_scores)
            consistency = max(0, 1 - variance)  # Convert variance to consistency
        else:
            consistency = 1.0
        consistency_factor = consistency * self.CONSISTENCY_WEIGHT

        # Factor 3: Number of supporting chunks (20% weight)
        chunk_count_score = min(len(context_chunks) / 10.0, 1.0)  # Normalize to max 10
        chunk_factor = chunk_count_score * self.CHUNK_COVERAGE_WEIGHT

        # Factor 4: Entity mention frequency, weighted by label specificity (15% weight)
        # Use pre-extracted typed entities when available; fall back to heuristics.
        if entity_matches is not None:
            entities = [m.text for m in entity_matches]
            entity_coverage = (
                self._calculate_entity_coverage_typed(entity_matches, context_chunks)
                if entity_matches
                else None
            )
        else:
            entities = self._extract_entities(question)
            entity_coverage = (
                self._calculate_entity_coverage(entities, context_chunks) if entities else None
            )

        # Use neutral score if no entities
        entity_factor = (
            entity_coverage * self.ENTITY_COVERAGE_WEIGHT
            if entity_coverage is not None
            else 0.5 * self.ENTITY_COVERAGE_WEIGHT
        )

        # Combine all factors
        final_score = score_factor + consistency_factor + chunk_factor + entity_factor

        # Determine confidence level
        if final_score >= self.HIGH_THRESHOLD:
            level = "high"
        elif final_score >= self.MEDIUM_THRESHOLD:
            level = "medium"
        else:
            level = "low"

        logger.debug(
            "Confidence: %.3f (%s) — retrieval=%.3f consistency=%.3f chunks=%d entities=%s",
            final_score,
            level,
            avg_score,
            consistency,
            len(context_chunks),
            f"{entity_coverage:.3f}" if entity_coverage is not None else "n/a",
        )

        # Generate human-readable explanation
        explanation = self._generate_explanation(
            level=level,
            score=final_score,
            retrieval_score=avg_score,
            consistency=consistency,
            chunk_count=len(context_chunks),
            entity_coverage=entity_coverage,
            entities=entities,
        )

        return ConfidenceResult(
            score=round(final_score, 3),
            level=level,
            explanation=explanation,
            factors=ConfidenceFactors(
                retrieval_score=round(avg_score, 3),
                consistency=round(consistency, 3),
                chunk_coverage=round(chunk_count_score, 3),  # Use normalized score
                entity_coverage=round(entity_coverage, 3) if entity_coverage else None,
            ),
        )

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities from question.

        Simple heuristic: capitalized words longer than 2 chars.

        Args:
            text: Question text

        Returns:
            List of potential entity names
        """
        words = text.split()
        entities = [w.strip(".,!?;:\"'") for w in words if len(w) > 2 and w[0].isupper()]
        return entities

    def _calculate_entity_coverage(
        self, entities: List[str], context_chunks: List[ContextChunk]
    ) -> float:
        """Calculate what percentage of chunks mention the entities.

        Args:
            entities: Entity names to look for
            context_chunks: Context chunks to search in

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        entity_mentions = 0

        for chunk in context_chunks:
            text_lower = chunk.text.lower()
            if any(entity.lower() in text_lower for entity in entities):
                entity_mentions += 1

        return entity_mentions / len(context_chunks) if context_chunks else 0.0

    def _calculate_entity_coverage_typed(
        self,
        entity_matches: List[EntityMatch],
        context_chunks: List[ContextChunk],
    ) -> float:
        """Calculate entity coverage weighted by label specificity.

        Entities that appear in virtually every political-speech chunk (e.g.
        "Trump" with label PERSON) carry a lower specificity weight than rare,
        query-specific entities such as named laws (LAW) or events (EVENT).
        This prevents generic queries from receiving an artificially inflated
        entity-coverage score.

        Per-entity score = chunk_coverage × label_specificity_weight.
        Final score = mean of per-entity scores (0.0 – 1.0).

        Args:
            entity_matches: Typed entities extracted by spaCy NER.
            context_chunks: Retrieved context chunks.

        Returns:
            Weighted coverage ratio (0.0 to 1.0).
        """
        if not context_chunks or not entity_matches:
            return 0.0

        per_entity_scores: List[float] = []
        for match in entity_matches:
            # Raw per-chunk coverage for this entity
            mentions = sum(
                1
                for chunk in context_chunks
                if match.text.lower() in chunk.text.lower()
            )
            raw_coverage = mentions / len(context_chunks)

            # Scale by label specificity (defaults to 0.5 for unknown labels)
            specificity = self.LABEL_SPECIFICITY.get(match.label, 0.50)
            per_entity_scores.append(raw_coverage * specificity)

        return sum(per_entity_scores) / len(per_entity_scores)

    def _generate_explanation(
        self,
        level: str,
        score: float,
        retrieval_score: float,
        consistency: float,
        chunk_count: int,
        entity_coverage: Optional[float],
        entities: Optional[List[str]],
    ) -> str:
        """Generate human-readable explanation for confidence level.

        Args:
            level: Confidence level (high/medium/low)
            score: Overall confidence score
            retrieval_score: Average semantic similarity
            consistency: Score consistency
            chunk_count: Number of chunks
            entity_coverage: Entity coverage ratio
            entities: Detected entities

        Returns:
            Human-readable explanation string
        """
        explanations = []

        # Main score explanation
        explanations.append(f"Overall confidence is {level.upper()} (score: {score:.2f})")

        # Retrieval quality
        if retrieval_score > 0.8:
            explanations.append(f"excellent semantic match (similarity: {retrieval_score:.2f})")
        elif retrieval_score > 0.6:
            explanations.append(f"good semantic match (similarity: {retrieval_score:.2f})")
        elif retrieval_score > 0.4:
            explanations.append(f"moderate semantic match (similarity: {retrieval_score:.2f})")
        else:
            explanations.append(f"weak semantic match (similarity: {retrieval_score:.2f})")

        # Consistency
        if consistency > 0.9:
            explanations.append(f"very consistent results (consistency: {consistency:.2f})")
        elif consistency > 0.7:
            explanations.append(f"consistent results (consistency: {consistency:.2f})")
        else:
            explanations.append(f"varied results (consistency: {consistency:.2f})")

        # Chunk coverage
        explanations.append(f"{chunk_count} supporting context chunks")

        # Entity coverage
        if entities and entity_coverage is not None:
            entity_names = ", ".join(entities[:2])  # First 2 entities
            if entity_coverage > 0.8:
                explanations.append(f"'{entity_names}' mentioned in all retrieved chunks")
            elif entity_coverage > 0.5:
                explanations.append(
                    f"'{entity_names}' mentioned in most chunks ({entity_coverage:.0%})"
                )
            else:
                explanations.append(
                    f"'{entity_names}' mentioned in some chunks ({entity_coverage:.0%})"
                )

        # Join with appropriate separators
        base = explanations[0]
        if len(explanations) > 1:
            details = ", ".join(explanations[1:])
            return f"{base} based on {details}."
        return f"{base}."
