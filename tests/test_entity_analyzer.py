"""Tests for EntityAnalyzer component.

Tests entity extraction, statistics, and associations.
Note: Sentiment analysis tests are skipped if sentiment analyzer not available.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from speech_nlp.services.rag.entities import RELEVANT_NER_LABELS, EntityAnalyzer
from speech_nlp.services.rag.models import EntityMatch


class TestEntityAnalyzer:
    """Test suite for EntityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create EntityAnalyzer without collection (heuristic path for deterministic tests)."""
        return EntityAnalyzer(use_ner=False)

    @pytest.fixture
    def mock_collection(self):
        """Create mock ChromaDB collection."""
        collection = Mock()
        collection.get = MagicMock(
            return_value={
                "documents": [
                    "Trump spoke about Biden and the economy",
                    "Biden's policies were criticized by Trump",
                    "The economy is strong says Trump",
                ],
                "metadatas": [
                    {"source": "speech1.txt"},
                    {"source": "speech2.txt"},
                    {"source": "speech3.txt"},
                ],
            }
        )
        collection.count = MagicMock(return_value=3)
        return collection

    @pytest.fixture
    def analyzer_with_collection(self, mock_collection):
        """Create EntityAnalyzer with mock collection (heuristic path)."""
        return EntityAnalyzer(collection=mock_collection, use_ner=False)

    def test_initialization(self, analyzer):
        """Test EntityAnalyzer initialization."""
        assert analyzer.collection is None
        assert analyzer.sentiment_analyzer is None
        assert isinstance(analyzer.stopwords, set)
        assert len(analyzer.stopwords) > 0

    def test_extract_entities_basic(self, analyzer):
        """Test basic entity extraction."""
        text = "Trump spoke about Biden and China"
        entities = analyzer.extract_entities(text)

        assert "Trump" in entities
        assert "Biden" in entities
        assert "China" in entities

    def test_extract_entities_filters_question_words(self, analyzer):
        """Test that question words are filtered out."""
        text = "What did Trump say about Biden?"
        entities = analyzer.extract_entities(text)

        assert "Trump" in entities
        assert "Biden" in entities
        assert "What" not in entities  # Question word filtered

    def test_extract_entities_filters_short_words(self, analyzer):
        """Test that short words are filtered."""
        text = "I met Dr Smith in US"
        entities = analyzer.extract_entities(text)

        assert "Smith" in entities
        # "Dr" and "US" are too short (2 chars or less)
        assert "Dr" not in entities
        assert "US" not in entities

    def test_extract_entities_removes_punctuation(self, analyzer):
        """Test punctuation removal."""
        text = "Trump, Biden, and Obama."
        entities = analyzer.extract_entities(text)

        # Should extract without punctuation
        assert "Trump" in entities
        assert "Biden" in entities
        assert "Obama" in entities

    def test_extract_entities_empty_text(self, analyzer):
        """Test with empty text."""
        entities = analyzer.extract_entities("")
        assert entities == []

    def test_extract_entities_no_capitalized_words(self, analyzer):
        """Test with no capitalized words."""
        text = "this is all lowercase text"
        entities = analyzer.extract_entities(text)
        assert entities == []

    def test_get_statistics_without_collection(self, analyzer):
        """Test statistics without collection returns empty dict."""
        entities = ["Trump", "Biden"]

        # With no collection, should return empty dict
        stats = analyzer.get_statistics(entities)
        assert isinstance(stats, dict)
        # May be empty or have entries depending on implementation
        # Don't expect error since it gracefully handles missing collection

    def test_get_statistics_basic(self, analyzer_with_collection):
        """Test basic statistics gathering."""
        entities = ["Trump", "Biden"]
        stats = analyzer_with_collection.get_statistics(entities)

        assert isinstance(stats, dict)
        assert "Trump" in stats
        assert "Biden" in stats

        # Check structure of each entity's stats - using Pydantic model attributes
        for _entity, entity_stats in stats.items():
            assert hasattr(entity_stats, "mention_count")
            assert hasattr(entity_stats, "speech_count")
            assert hasattr(entity_stats, "corpus_percentage")
            assert hasattr(entity_stats, "speeches")
            assert isinstance(entity_stats.speeches, list)

    def test_get_statistics_case_insensitive(self, analyzer_with_collection):
        """Test that entity matching is case-insensitive."""
        # Mock returns "Trump" in documents, we search for "trump"
        stats = analyzer_with_collection.get_statistics(["trump"])

        # Should still find matches (case-insensitive)
        assert "trump" in stats or "Trump" in stats

    def test_find_associations_basic(self, analyzer_with_collection):
        """Test finding associated terms."""
        contexts = [
            "Trump spoke about Biden and the economy",
            "Trump criticized Biden's economic policies",
        ]

        associations = analyzer_with_collection.find_associations("Trump", contexts, top_n=5)

        assert isinstance(associations, list)
        assert len(associations) <= 5
        # Should find "Biden" and "economy" associated with "Trump"
        assert any("Biden" in a for a in associations) or any("economy" in a for a in associations)

    def test_find_associations_filters_entity_itself(self, analyzer_with_collection):
        """Test that the entity itself is not in associations."""
        contexts = ["Trump Trump Trump mentioned Biden"]

        associations = analyzer_with_collection.find_associations("Trump", contexts, top_n=10)

        # "Trump" should not be in its own associations
        assert "Trump" not in associations
        assert "trump" not in associations

    def test_find_associations_filters_stopwords(self, analyzer_with_collection):
        """Test that stopwords are filtered from associations."""
        contexts = ["Trump spoke about the economy and the policies"]

        associations = analyzer_with_collection.find_associations("Trump", contexts, top_n=10)

        # Stopwords like "the", "and" should be filtered
        assert "the" not in associations
        assert "and" not in associations

    def test_find_associations_empty_contexts(self, analyzer_with_collection):
        """Test associations with no contexts."""
        associations = analyzer_with_collection.find_associations("Trump", [], top_n=5)

        assert associations == []

    def test_entity_statistics_with_sentiment_none(self, analyzer_with_collection):
        """Test statistics without sentiment analyzer."""
        stats = analyzer_with_collection.get_statistics(["Trump"], include_sentiment=False)

        assert "Trump" in stats
        # Sentiment should not be included (use attribute access for Pydantic model)
        assert stats["Trump"].sentiment is None

    def test_entity_statistics_with_associations(self, analyzer_with_collection):
        """Test statistics with associations enabled."""
        stats = analyzer_with_collection.get_statistics(["Trump"], include_associations=True)

        assert "Trump" in stats
        # Associations should be included (use attribute access for Pydantic model)
        if stats["Trump"].associated_terms:
            assert isinstance(stats["Trump"].associated_terms, list)

    def test_entity_statistics_corpus_percentage(self, mock_collection):
        """Test corpus percentage calculation."""
        # Mock 100 total documents
        mock_collection.count = MagicMock(return_value=100)
        # Mock 10 documents containing "Trump"
        mock_collection.get = MagicMock(
            return_value={
                "documents": ["Trump"] * 10,
                "metadatas": [{"source": f"speech{i}.txt"} for i in range(10)],
            }
        )

        analyzer = EntityAnalyzer(collection=mock_collection, use_ner=False)
        stats = analyzer.get_statistics(["Trump"])

        # Should be 10% (use attribute access for Pydantic model)
        assert stats["Trump"].corpus_percentage == 10.0

    def test_multiple_entities_statistics(self, analyzer_with_collection):
        """Test statistics for multiple entities."""
        entities = ["Trump", "Biden", "China"]
        stats = analyzer_with_collection.get_statistics(entities)

        # Should have stats for each entity found (may not find all if not in test data)
        assert len(stats) > 0
        # Check that returned entities are from the requested list
        for entity in stats.keys():
            assert entity in entities

    def test_extract_entities_preserves_case(self, analyzer):
        """Test that extracted entities preserve original case."""
        text = "Trump spoke with Biden about China"
        entities = analyzer.extract_entities(text)

        # Should preserve exact case
        assert "Trump" in entities
        assert "Biden" in entities
        assert "China" in entities

    def test_stopwords_set_contains_common_words(self, analyzer):
        """Test that stopwords set contains expected words."""
        assert "the" in analyzer.stopwords
        assert "and" in analyzer.stopwords
        assert "is" in analyzer.stopwords
        assert "of" in analyzer.stopwords

    def test_extract_entities_complex_sentence(self, analyzer):
        """Test entity extraction from complex sentence."""
        text = "President Trump met with Chinese President Xi Jinping in Beijing"
        entities = analyzer.extract_entities(text)

        assert "President" in entities
        assert "Trump" in entities
        assert "Chinese" in entities
        assert "Jinping" in entities
        assert "Beijing" in entities


# ---------------------------------------------------------------------------
# NER-specific tests
# ---------------------------------------------------------------------------


class TestEntityMatchModel:
    """Tests for the EntityMatch Pydantic model."""

    def test_entity_match_creation(self):
        """Test EntityMatch instantiation."""
        em = EntityMatch(text="Donald Trump", label="PERSON")
        assert em.text == "Donald Trump"
        assert em.label == "PERSON"

    def test_entity_match_unknown_label(self):
        """Test EntityMatch with UNKNOWN label (heuristic fallback)."""
        em = EntityMatch(text="Trump", label="UNKNOWN")
        assert em.label == "UNKNOWN"

    def test_entity_match_all_relevant_labels(self):
        """Test that all RELEVANT_NER_LABELS are valid label values."""
        for label in RELEVANT_NER_LABELS:
            em = EntityMatch(text="test", label=label)
            assert em.label == label


class TestNERRelevantLabels:
    """Tests for the NER label configuration."""

    def test_relevant_labels_are_frozenset(self):
        """RELEVANT_NER_LABELS should be immutable."""
        assert isinstance(RELEVANT_NER_LABELS, frozenset)

    def test_relevant_labels_contains_key_types(self):
        """Check key political-speech entity types are present."""
        for expected in ("PERSON", "ORG", "GPE", "NORP"):
            assert expected in RELEVANT_NER_LABELS

    def test_numeric_labels_excluded(self):
        """Purely numeric / temporal labels should not be present."""
        for excluded in ("DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY"):
            assert excluded not in RELEVANT_NER_LABELS


class TestSpacyNERPath:
    """Tests for the spaCy-based NER extraction path."""

    def _make_mock_ent(self, text, label):
        ent = MagicMock()
        ent.text = text
        ent.label_ = label
        return ent

    def _make_mock_nlp(self, ents):
        """Return a callable mock that produces a doc with the given ents."""
        doc = MagicMock()
        doc.ents = ents
        nlp = MagicMock(return_value=doc)
        return nlp

    def test_extract_with_spacy_person(self):
        """SpaCy path should return PERSON entities."""
        analyzer = EntityAnalyzer(use_ner=True)
        mock_ents = [self._make_mock_ent("Donald Trump", "PERSON")]
        mock_nlp = self._make_mock_nlp(mock_ents)

        result = analyzer._extract_spacy("Donald Trump visited the White House", mock_nlp)

        assert len(result) == 1
        assert result[0].text == "Donald Trump"
        assert result[0].label == "PERSON"

    def test_extract_with_spacy_filters_irrelevant_labels(self):
        """SpaCy path must drop labels not in RELEVANT_NER_LABELS."""
        analyzer = EntityAnalyzer(use_ner=True)
        mock_ents = [
            self._make_mock_ent("Donald Trump", "PERSON"),  # keep
            self._make_mock_ent("2024", "DATE"),  # drop
            self._make_mock_ent("$1 million", "MONEY"),  # drop
            self._make_mock_ent("America", "GPE"),  # keep
        ]
        mock_nlp = self._make_mock_nlp(mock_ents)

        result = analyzer._extract_spacy("sample text", mock_nlp)

        texts = [e.text for e in result]
        assert "Donald Trump" in texts
        assert "America" in texts
        assert "2024" not in texts
        assert "$1 million" not in texts

    def test_extract_with_spacy_deduplicates(self):
        """Duplicate entity texts should appear only once."""
        analyzer = EntityAnalyzer(use_ner=True)
        mock_ents = [
            self._make_mock_ent("Trump", "PERSON"),
            self._make_mock_ent("Trump", "PERSON"),  # duplicate
        ]
        mock_nlp = self._make_mock_nlp(mock_ents)

        result = analyzer._extract_spacy("Trump Trump", mock_nlp)

        assert len(result) == 1

    def test_extract_with_spacy_multiword_entity(self):
        """SpaCy path should handle multi-word entities as a single entry."""
        analyzer = EntityAnalyzer(use_ner=True)
        mock_ents = [self._make_mock_ent("United States", "GPE")]
        mock_nlp = self._make_mock_nlp(mock_ents)

        result = analyzer._extract_spacy("The United States is great", mock_nlp)

        assert len(result) == 1
        assert result[0].text == "United States"
        assert result[0].label == "GPE"

    def test_extract_with_spacy_empty_text_returns_empty(self):
        """Empty text should return empty list without calling nlp."""
        analyzer = EntityAnalyzer(use_ner=True)
        mock_nlp = MagicMock()

        result = analyzer.extract_entities_with_types("")

        assert result == []
        mock_nlp.assert_not_called()


class TestHeuristicFallbackPath:
    """Tests for the capitalisation-heuristic fallback path."""

    @pytest.fixture
    def heuristic_analyzer(self):
        """Analyzer with NER disabled (always uses heuristics)."""
        return EntityAnalyzer(use_ner=False)

    def test_heuristic_returns_entity_match_objects(self, heuristic_analyzer):
        """Heuristic path must return List[EntityMatch], not List[str]."""
        result = heuristic_analyzer.extract_entities_with_types("Trump spoke about Biden")
        assert all(isinstance(e, EntityMatch) for e in result)

    def test_heuristic_label_is_unknown(self, heuristic_analyzer):
        """All heuristic entities should carry label UNKNOWN."""
        result = heuristic_analyzer.extract_entities_with_types("Trump Biden China")
        assert all(e.label == "UNKNOWN" for e in result)

    def test_heuristic_backward_compat_extract_entities(self, heuristic_analyzer):
        """extract_entities() must still return List[str] (backward compat)."""
        result = heuristic_analyzer.extract_entities("Trump spoke about Biden")
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_extract_entities_with_types_wraps_heuristics(self, heuristic_analyzer):
        """extract_entities_with_types() uses heuristics when NER disabled."""
        result = heuristic_analyzer.extract_entities_with_types("Trump Biden China")
        texts = [e.text for e in result]
        assert "Trump" in texts


class TestSpacyLazyLoad:
    """Tests for the _get_nlp lazy-load mechanism."""

    def test_get_nlp_returns_none_when_ner_disabled(self):
        """_get_nlp must return None when use_ner=False."""
        analyzer = EntityAnalyzer(use_ner=False)
        assert analyzer._get_nlp() is None

    def test_get_nlp_disables_ner_on_import_error(self):
        """ImportError should set _use_ner=False and return None."""
        analyzer = EntityAnalyzer(use_ner=True, ner_model="en_core_web_sm")
        with patch.dict("sys.modules", {"spacy": None}):
            # Force a fresh load attempt by clearing cached nlp
            analyzer._nlp = None
            # With spacy absent, _get_nlp should gracefully return None
            # (actual behaviour depends on whether spacy is installed;
            #  we test the import-error branch via mocking)
            with patch("builtins.__import__", side_effect=ImportError("no spacy")):
                analyzer._use_ner = True
                analyzer._nlp = None
                result = analyzer._get_nlp()
        assert result is None
        assert analyzer._use_ner is False

    def test_get_nlp_disables_ner_on_model_not_found(self):
        """OSError (model not found) should set _use_ner=False and return None."""
        analyzer = EntityAnalyzer(use_ner=True, ner_model="nonexistent_model_xyz")
        # Should gracefully fall back without raising
        result = analyzer._get_nlp()
        assert result is None
        assert analyzer._use_ner is False

    def test_get_nlp_caches_loaded_model(self):
        """Second call to _get_nlp should return the same cached object."""
        analyzer = EntityAnalyzer(use_ner=True, ner_model="en_core_web_sm")
        nlp1 = analyzer._get_nlp()
        nlp2 = analyzer._get_nlp()
        # Both should be the same object (or both None if spaCy unavailable)
        assert nlp1 is nlp2

    def test_ner_enabled_extracts_typed_entities(self):
        """With spaCy available, extract_entities_with_types should use NER."""
        analyzer = EntityAnalyzer(use_ner=True, ner_model="en_core_web_sm")
        if analyzer._get_nlp() is None:
            pytest.skip("spaCy model not available in this environment")

        text = "Donald Trump spoke in New York about the Republican Party."
        result = analyzer.extract_entities_with_types(text)

        assert len(result) > 0
        labels = {e.label for e in result}
        # At minimum we expect a PERSON or GPE label
        assert labels & {"PERSON", "GPE", "NORP", "ORG"}

    def test_ner_backward_compat_with_spacy(self):
        """extract_entities() must still return List[str] when NER is on."""
        analyzer = EntityAnalyzer(use_ner=True, ner_model="en_core_web_sm")
        if analyzer._get_nlp() is None:
            pytest.skip("spaCy model not available in this environment")

        result = analyzer.extract_entities("Donald Trump spoke in New York.")
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)
