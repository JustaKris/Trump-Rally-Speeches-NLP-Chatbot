"""Entity analyzer component for RAG service.

Handles entity extraction, statistics gathering, sentiment analysis,
and finding associated terms for named entities in the corpus.

Named entity recognition uses spaCy (``en_core_web_sm`` by default) when the
``ner`` optional dependency group is installed.  If spaCy or the requested
model is unavailable the analyzer transparently falls back to the original
capitalisation-heuristic approach so the rest of the pipeline is unaffected.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from chromadb.api.types import IncludeEnum

from speech_nlp.constants import ENTITY_QUESTION_WORDS, ENTITY_STOPWORDS
from speech_nlp.services.rag.models import EntityMatch, EntitySentiment, EntityStatistics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NER label configuration
# ---------------------------------------------------------------------------

# spaCy entity labels that are meaningful for political-speech analysis.
# Labels not in this set (DATE, TIME, CARDINAL, ORDINAL, PERCENT, QUANTITY,
# MONEY) are dropped to keep entity lists clean and relevant.
RELEVANT_NER_LABELS: frozenset[str] = frozenset(
    {
        "PERSON",       # Politicians, journalists, named individuals
        "ORG",          # Companies, media outlets, parties, agencies
        "GPE",          # Countries, states, cities (geopolitical entities)
        "NORP",         # Nationalities, political/religious groups
        "FAC",          # Buildings, airports, bridges, highways
        "EVENT",        # Named events (elections, summits)
        "LAW",          # Named laws, amendments, acts
        "PRODUCT",      # Products, vehicles, software
        "WORK_OF_ART",  # Titles of books, songs, TV shows
    }
)

# Human-readable descriptions for NER labels (used in API responses)
NER_LABEL_DESCRIPTIONS: Dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "NORP": "Nationality / Group",
    "FAC": "Facility",
    "EVENT": "Event",
    "LAW": "Law / Document",
    "PRODUCT": "Product",
    "WORK_OF_ART": "Work of Art",
    "UNKNOWN": "Unknown",
}


class EntityAnalyzer:
    """Analyzes named entities in text and corpus.

    Provides entity extraction, frequency statistics, sentiment analysis,
    and term association discovery.

    When spaCy is installed (``uv sync --group ner``) the analyzer uses the
    configured spaCy model (default ``en_core_web_sm``) for proper NER,
    handling multi-word entities and entity-type classification.  If spaCy
    is not available it falls back to the original capitalisation heuristics
    transparently — no configuration change is required.
    """

    def __init__(
        self,
        collection=None,
        sentiment_analyzer=None,
        use_ner: bool = True,
        ner_model: str = "en_core_web_sm",
    ):
        """Initialize entity analyzer.

        Args:
            collection: ChromaDB collection for corpus-wide analysis
            sentiment_analyzer: Optional sentiment analyzer for entity sentiment
            use_ner: Use spaCy NER when available (True by default)
            ner_model: spaCy model name to load (default ``en_core_web_sm``)
        """
        self.collection = collection
        self.sentiment_analyzer = sentiment_analyzer
        self._use_ner = use_ner
        self._ner_model_name = ner_model
        self._nlp = None  # Lazy-loaded spaCy pipeline

        # Use centralized stopwords from constants
        self.stopwords = ENTITY_STOPWORDS

        if use_ner:
            logger.debug(f"EntityAnalyzer: NER enabled (model={ner_model})")
        else:
            logger.debug("EntityAnalyzer: NER disabled, using heuristics")

    # ------------------------------------------------------------------
    # spaCy lazy-load
    # ------------------------------------------------------------------

    def _get_nlp(self):
        """Lazy-load and cache the spaCy NLP pipeline.

        Returns the spaCy ``Language`` object on success, or ``None`` if
        spaCy / the requested model is unavailable (triggers fallback).
        """
        if not self._use_ner:
            return None

        # Already loaded successfully
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy  # type: ignore[import-not-found]

            self._nlp = spacy.load(  # type: ignore[assignment]
                self._ner_model_name,
                # Only load the NER component — disables parser/tagger for speed
                exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
            )
            logger.info(f"✓ spaCy NER model loaded: {self._ner_model_name}")
            return self._nlp

        except ImportError:
            logger.warning(
                "spaCy not installed — falling back to capitalisation heuristics. "
                "Install the 'ner' dependency group to enable proper NER: "
                "uv sync --group ner && python -m spacy download en_core_web_sm"
            )
        except OSError:
            logger.warning(
                f"spaCy model '{self._ner_model_name}' not found — "
                "falling back to capitalisation heuristics. "
                f"Run: python -m spacy download {self._ner_model_name}"
            )
        except Exception as e:
            logger.warning(f"Failed to load spaCy model '{self._ner_model_name}': {e}")

        # Disable NER so we don't retry on every call
        self._use_ner = False
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entity strings from text.

        Uses spaCy NER when available; falls back to capitalisation heuristics.
        Returns a deduplicated list of entity text strings (backward-compatible).

        Args:
            text: Text to extract entities from

        Returns:
            Deduplicated list of entity name strings
        """
        matches = self.extract_entities_with_types(text)
        return list({m.text for m in matches})

    def extract_entities_with_types(self, text: str) -> List[EntityMatch]:
        """Extract named entities with their types from text.

        Uses spaCy NER when available (multi-word entities, type labels);
        falls back to capitalisation heuristics that emit label ``"UNKNOWN"``.

        Args:
            text: Text to extract entities from

        Returns:
            List of :class:`EntityMatch` objects with ``text`` and ``label``
        """
        if not text:
            return []

        nlp = self._get_nlp()
        if nlp is not None:
            return self._extract_spacy(text, nlp)
        return self._extract_heuristic(text)

    # ------------------------------------------------------------------
    # NER backends
    # ------------------------------------------------------------------

    def _extract_spacy(self, text: str, nlp) -> List[EntityMatch]:
        """Extract entities using spaCy NER.

        Args:
            text: Input text
            nlp: Loaded spaCy Language pipeline

        Returns:
            Filtered, deduplicated list of EntityMatch objects
        """
        doc = nlp(text)
        seen: dict[str, str] = {}  # text → label (keeps first occurrence)

        for ent in doc.ents:
            label = ent.label_
            if label not in RELEVANT_NER_LABELS:
                continue

            clean = ent.text.strip()
            if len(clean) < 2:
                continue
            if clean.lower() in ENTITY_QUESTION_WORDS:
                continue

            # Keep the first label seen for a given entity text
            if clean not in seen:
                seen[clean] = label

        entities = [EntityMatch(text=t, label=lbl) for t, lbl in seen.items()]
        logger.debug(f"spaCy NER: extracted {len(entities)} entities from text")
        return entities

    def _extract_heuristic(self, text: str) -> List[EntityMatch]:
        """Extract entity candidates using capitalisation heuristics.

        Fallback when spaCy is unavailable.  All entities receive label
        ``"UNKNOWN"`` since no type information is available.

        Args:
            text: Input text

        Returns:
            List of EntityMatch objects with label ``"UNKNOWN"``
        """
        words = text.split()
        seen: set[str] = set()

        for word in words:
            clean = word.strip(".,!?;:\"'")
            if clean and clean[0].isupper() and len(clean) > 2:
                if clean.lower() not in ENTITY_QUESTION_WORDS:
                    seen.add(clean)

        entities = [EntityMatch(text=t, label="UNKNOWN") for t in seen]
        logger.debug(f"Heuristic NER: extracted {len(entities)} entities from text")
        return entities

    # ------------------------------------------------------------------
    # Corpus statistics
    # ------------------------------------------------------------------

    def get_statistics(
        self,
        entities: List[str],
        include_sentiment: bool = False,
        include_associations: bool = True,
    ) -> Dict[str, EntityStatistics]:
        """Get comprehensive statistics about entity mentions across the corpus.

        Args:
            entities: List of entity names to analyze
            include_sentiment: Calculate sentiment for entity mentions
            include_associations: Find commonly associated terms

        Returns:
            Dict mapping entity names to their statistics
        """
        if not entities or self.collection is None:
            return {}

        stats = {}
        all_docs = self.collection.get(include=[IncludeEnum.documents, IncludeEnum.metadatas])

        if not all_docs["documents"]:
            return {}

        for entity in entities:
            entity_stats = self._analyze_single_entity(
                entity, all_docs, include_sentiment, include_associations
            )
            if entity_stats:
                stats[entity] = entity_stats

        logger.info(f"Generated statistics for {len(stats)} entities")
        return stats

    def _analyze_single_entity(
        self,
        entity: str,
        all_docs: Dict[str, Any],
        include_sentiment: bool,
        include_associations: bool,
    ) -> Optional[EntityStatistics]:
        """Analyze a single entity across the corpus.

        Args:
            entity: Entity name
            all_docs: All documents from collection
            include_sentiment: Include sentiment analysis
            include_associations: Include term associations

        Returns:
            EntityStatistics if entity found, None otherwise
        """
        entity_lower = entity.lower()
        mentions = 0
        speeches_with_entity = set()
        entity_contexts: List[str] = []

        # Count mentions across all documents
        for i, doc in enumerate(all_docs["documents"]):
            doc_lower = doc.lower()
            count = doc_lower.count(entity_lower)

            if count > 0:
                mentions += count

                # Track which speeches mention this entity
                if all_docs["metadatas"] and i < len(all_docs["metadatas"]):
                    source = all_docs["metadatas"][i].get("source", "unknown")
                    speeches_with_entity.add(source)

                # Store context for analysis (limit to avoid memory issues)
                if len(entity_contexts) < 100:
                    entity_contexts.append(doc)

        # Return None if entity not found
        if mentions == 0:
            return None

        # Calculate percentage of corpus (based on number of speeches)
        total_doc_count = self.collection.count() if self.collection else len(all_docs["documents"])
        speech_count = len(speeches_with_entity)
        percentage = (speech_count / total_doc_count * 100) if total_doc_count > 0 else 0

        # Build base statistics
        entity_data: Dict[str, Any] = {
            "mention_count": mentions,
            "speech_count": len(speeches_with_entity),
            "corpus_percentage": round(percentage, 2),
            "speeches": sorted(speeches_with_entity)[:10],  # Limit to first 10
        }

        # Add sentiment analysis if requested
        if include_sentiment and entity_contexts:
            sentiment = self.analyze_sentiment(entity, entity_contexts)
            entity_data["sentiment"] = sentiment

        # Add associated terms if requested
        if include_associations and entity_contexts:
            associations = self.find_associations(entity, entity_contexts)
            entity_data["associated_terms"] = associations

        return EntityStatistics(**entity_data)

    # ------------------------------------------------------------------
    # Sentiment & associations
    # ------------------------------------------------------------------

    def analyze_sentiment(self, entity: str, contexts: List[str]) -> EntitySentiment:
        """Analyze sentiment of text chunks mentioning an entity.

        Args:
            entity: Entity name
            contexts: Text chunks containing the entity

        Returns:
            EntitySentiment with average score and classification
        """
        if not self.sentiment_analyzer:
            logger.warning("Sentiment analyzer not available")
            return EntitySentiment(
                average_score=0.0,
                classification="Unknown",
                sample_size=0,
            )

        try:
            # Analyze sentiment for each context
            sentiments = []

            for context in contexts[:50]:  # Limit to 50 contexts for performance
                try:
                    result = self.sentiment_analyzer.analyze_sentiment(context)

                    # Convert to numeric score: positive=1, neutral=0, negative=-1
                    if result.get("dominant") == "positive":
                        sentiments.append(result.get("positive", 0))
                    elif result.get("dominant") == "negative":
                        sentiments.append(-result.get("negative", 0))
                    else:
                        sentiments.append(0)
                except Exception:  # nosec B112
                    # Skip failed analyses - intentional pattern for robust error handling
                    continue

            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)

                # Classify overall sentiment
                if avg_sentiment > 0.2:
                    classification = "Positive"
                elif avg_sentiment < -0.2:
                    classification = "Negative"
                else:
                    classification = "Neutral"

                return EntitySentiment(
                    average_score=round(avg_sentiment, 2),
                    classification=classification,
                    sample_size=len(sentiments),
                )

        except Exception as e:
            logger.warning(f"Sentiment analysis failed for {entity}: {e}")

        return EntitySentiment(
            average_score=0.0,
            classification="Unknown",
            sample_size=0,
        )

    def find_associations(self, entity: str, contexts: List[str], top_n: int = 5) -> List[str]:
        """Find most common terms associated with an entity.

        Args:
            entity: Entity name
            contexts: Text chunks containing the entity
            top_n: Number of top associations to return

        Returns:
            List of most common associated terms
        """
        entity_lower = entity.lower()
        words = []

        for context in contexts[:50]:  # Limit for performance
            context_lower = context.lower()

            if entity_lower in context_lower:
                # Extract words near the entity (window-based approach)
                words_in_context = re.findall(r"\b[a-z]{3,}\b", context_lower)
                words.extend(
                    [w for w in words_in_context if w not in self.stopwords and w != entity_lower]
                )

        # Count and return most common
        if words:
            word_counts = Counter(words)
            return [word for word, _ in word_counts.most_common(top_n)]

        return []
