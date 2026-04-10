"""Tests for DocumentLoader component.

Tests document loading, chunking, metadata generation, and semantic chunking.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.services.rag.document_loader import DocumentLoader


class TestDocumentLoader:
    """Test suite for DocumentLoader (fixed strategy)."""

    @pytest.fixture
    def loader(self):
        """Create DocumentLoader instance with fixed strategy."""
        return DocumentLoader(chunk_size=500, chunk_overlap=50, chunking_strategy="fixed")

    def test_initialization(self, loader):
        """Test DocumentLoader initialization."""
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 50
        assert loader.text_splitter is not None
        assert loader.chunking_strategy == "fixed"

    def test_chunk_text(self, loader):
        """Test text chunking functionality."""
        text = "This is a test. " * 100  # Create long text
        chunks = loader.chunk_text(text)

        assert len(chunks) > 1  # Should create multiple chunks
        for chunk in chunks:
            assert len(chunk) <= 500  # No chunk should exceed max size
            assert len(chunk) > 0  # No empty chunks

    def test_chunk_text_short_input(self, loader):
        """Test chunking with input shorter than chunk size."""
        text = "Short text"
        chunks = loader.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_empty(self, loader):
        """Test chunking empty text."""
        chunks = loader.chunk_text("")
        assert chunks == []

    def test_load_from_directory_existing(self, loader):
        """Test loading from real speeches directory."""
        data_dir = "data/Donald Trump Rally Speeches"

        if not Path(data_dir).exists():
            pytest.skip(f"Data directory {data_dir} not found")

        chunks, metadatas, ids = loader.load_from_directory(data_dir)

        # Verify outputs
        assert len(chunks) > 0
        assert len(chunks) == len(metadatas)
        assert len(chunks) == len(ids)

        # Verify metadata structure
        for metadata in metadatas:
            assert "source" in metadata
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert isinstance(metadata["source"], str)
            assert isinstance(metadata["chunk_index"], int)
            assert isinstance(metadata["total_chunks"], int)

        # Verify ID format
        for chunk_id in ids:
            assert isinstance(chunk_id, str)
            assert "_chunk_" in chunk_id

    def test_load_from_nonexistent_directory(self, loader):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            loader.load_from_directory("nonexistent/path")

    def test_load_from_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        loader = DocumentLoader(chunking_strategy="fixed")
        chunks, metadatas, ids = loader.load_from_directory(str(tmp_path))

        assert chunks == []
        assert metadatas == []
        assert ids == []

    def test_load_single_file(self, tmp_path):
        """Test loading a single file."""
        # Create test file
        test_file = tmp_path / "test_speech.txt"
        test_content = "This is a test speech. " * 50
        test_file.write_text(test_content, encoding="utf-8")

        loader = DocumentLoader(chunk_size=100, chunk_overlap=20, chunking_strategy="fixed")
        chunks, metadatas, ids = loader.load_from_directory(str(tmp_path))

        assert len(chunks) > 0
        assert all(m["source"] == "test_speech.txt" for m in metadatas)
        assert all("test_speech_chunk_" in chunk_id for chunk_id in ids)

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        loader = DocumentLoader(chunk_size=100, chunk_overlap=20, chunking_strategy="fixed")
        text = "Word " * 100  # Repetitive text to test overlap

        chunks = loader.chunk_text(text)

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            # At least some content should appear in both chunks
            assert chunks[0][-20:] != chunks[1][:20]  # Not exact same, but related

    def test_metadata_chunk_indices(self, tmp_path):
        """Test that chunk indices are sequential and correct."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Word " * 1000, encoding="utf-8")

        loader = DocumentLoader(chunk_size=100, chunk_overlap=10, chunking_strategy="fixed")
        _, metadatas, _ = loader.load_from_directory(str(tmp_path))

        # Get all chunk indices for the file
        indices = [m["chunk_index"] for m in metadatas]
        total_chunks = metadatas[0]["total_chunks"]

        # Should be sequential starting from 0
        assert indices == list(range(len(indices)))
        # total_chunks should match actual count
        assert total_chunks == len(indices)

    def test_multiple_files(self, tmp_path):
        """Test loading multiple files."""
        # Create multiple test files
        for i in range(3):
            test_file = tmp_path / f"speech_{i}.txt"
            test_file.write_text(f"Speech {i} content. " * 50, encoding="utf-8")

        loader = DocumentLoader(chunk_size=200, chunk_overlap=20, chunking_strategy="fixed")
        chunks, metadatas, ids = loader.load_from_directory(str(tmp_path))

        # Should have chunks from all 3 files
        sources = {m["source"] for m in metadatas}
        assert len(sources) == 3
        assert "speech_0.txt" in sources
        assert "speech_1.txt" in sources
        assert "speech_2.txt" in sources


class TestSemanticChunking:
    """Test suite for semantic chunking strategy."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock SentenceTransformer that returns deterministic embeddings."""
        model = MagicMock()

        def _encode(sentences, convert_to_numpy=True, show_progress_bar=False):
            """Return distinct embeddings that simulate topic shifts."""
            n = len(sentences)
            dim = 32
            rng = np.random.RandomState(42)
            embeddings = rng.randn(n, dim).astype(np.float32)
            # Normalise to unit vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms

        model.encode = MagicMock(side_effect=_encode)
        return model

    @pytest.fixture
    def semantic_loader(self, mock_embedding_model):
        """Create a DocumentLoader with semantic strategy."""
        return DocumentLoader(
            chunk_size=2048,
            chunk_overlap=150,
            chunking_strategy="semantic",
            embedding_model=mock_embedding_model,
            min_chunk_size=50,
            breakpoint_percentile=80.0,
        )

    def test_semantic_initialization(self, semantic_loader):
        """Test semantic DocumentLoader initializes correctly."""
        assert semantic_loader.chunking_strategy == "semantic"
        assert semantic_loader.embedding_model is not None
        assert semantic_loader.breakpoint_percentile == 80.0

    def test_semantic_fallback_without_model(self):
        """Test that semantic strategy falls back to fixed if no model provided."""
        loader = DocumentLoader(chunking_strategy="semantic", embedding_model=None)
        assert loader.chunking_strategy == "fixed"

    def test_semantic_chunking_produces_chunks(self, semantic_loader):
        """Test that semantic chunking produces non-empty chunks."""
        # Multi-sentence text with distinct topics
        text = (
            "The economy is growing rapidly. Jobs are coming back to America. "
            "Unemployment is at record lows. "
            "We need to secure the border. The wall is being built. "
            "Immigration reform is essential. "
            "Our military is the strongest in the world. "
            "We are rebuilding our armed forces. Veterans deserve better care."
        )
        chunks = semantic_loader.chunk_text(text)

        assert len(chunks) >= 1
        # All original content should be present
        combined = " ".join(chunks)
        assert "economy" in combined
        assert "border" in combined
        assert "military" in combined

    def test_semantic_chunking_respects_max_size(self, mock_embedding_model):
        """Test that no chunk exceeds the max chunk size."""
        loader = DocumentLoader(
            chunk_size=200,
            chunk_overlap=20,
            chunking_strategy="semantic",
            embedding_model=mock_embedding_model,
            min_chunk_size=30,
        )
        # Generate a long text
        text = ". ".join([f"Sentence number {i} with some filler content here" for i in range(50)])
        chunks = loader.chunk_text(text)

        for chunk in chunks:
            assert len(chunk) <= 200, f"Chunk exceeds max size: {len(chunk)} chars"

    def test_semantic_chunking_short_text(self, semantic_loader):
        """Test semantic chunking with very short text falls back gracefully."""
        text = "Just one sentence."
        chunks = semantic_loader.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_semantic_chunking_empty_text(self, semantic_loader):
        """Test semantic chunking with empty string."""
        assert semantic_loader.chunk_text("") == []

    def test_consecutive_cosine_similarities(self):
        """Test cosine similarity computation between consecutive embeddings."""
        # Two identical vectors → similarity = 1.0
        embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sims = DocumentLoader._consecutive_cosine_similarities(embeddings)

        assert len(sims) == 2
        assert abs(sims[0] - 1.0) < 1e-5  # identical → ~1.0
        assert abs(sims[1] - 0.0) < 1e-5  # orthogonal → ~0.0

    def test_detect_breakpoints_percentile(self, semantic_loader):
        """Test breakpoint detection using percentile."""
        # Simulate similarities with a clear drop
        sims = np.array([0.95, 0.92, 0.30, 0.90, 0.88, 0.25, 0.93])
        breakpoints = semantic_loader._detect_breakpoints(sims)

        # Indices 2 and 5 have the lowest similarities
        assert 2 in breakpoints
        assert 5 in breakpoints

    def test_detect_breakpoints_absolute_threshold(self):
        """Test breakpoint detection using absolute similarity threshold."""
        loader = DocumentLoader(
            chunking_strategy="fixed",  # doesn't matter for this unit test
            similarity_threshold=0.5,
        )
        sims = np.array([0.95, 0.40, 0.90, 0.30])
        breakpoints = loader._detect_breakpoints(sims)

        assert 1 in breakpoints  # 0.40 < 0.5
        assert 3 in breakpoints  # 0.30 < 0.5
        assert 0 not in breakpoints  # 0.95 >= 0.5
        assert 2 not in breakpoints  # 0.90 >= 0.5

    def test_group_sentences(self):
        """Test sentence grouping at breakpoints."""
        sentences = ["A.", "B.", "C.", "D.", "E."]
        breakpoints = [1, 3]  # breaks after index 1 and 3
        groups = DocumentLoader._group_sentences(sentences, breakpoints)

        assert len(groups) == 3
        assert groups[0] == "A. B."
        assert groups[1] == "C. D."
        assert groups[2] == "E."

    def test_enforce_size_constraints_merge_small(self, semantic_loader):
        """Test that small groups get merged."""
        # Set min_chunk_size so 'tiny' groups merge
        semantic_loader.min_chunk_size = 20
        groups = ["Hello world.", "A.", "This is a longer sentence that should stand alone."]
        result = semantic_loader._enforce_size_constraints(groups)

        # "A." (2 chars) should be merged with previous
        assert len(result) <= len(groups)
        combined = " ".join(result)
        assert "A." in combined

    def test_enforce_size_constraints_split_oversized(self, mock_embedding_model):
        """Test that oversized groups get split by the fallback splitter."""
        loader = DocumentLoader(
            chunk_size=50,
            chunk_overlap=10,
            chunking_strategy="semantic",
            embedding_model=mock_embedding_model,
            min_chunk_size=10,
        )
        oversized = "Word " * 100  # 500 chars, well over chunk_size=50
        result = loader._enforce_size_constraints([oversized])

        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 50

    def test_semantic_load_from_directory(self, tmp_path, mock_embedding_model):
        """Test full directory loading with semantic chunking."""
        # Create test file
        test_file = tmp_path / "speech.txt"
        test_file.write_text(
            "The economy is strong. Jobs are plentiful. "
            "We are building the wall. Border security matters. "
            "Our military is powerful. We support our veterans.",
            encoding="utf-8",
        )

        loader = DocumentLoader(
            chunk_size=2048,
            chunk_overlap=100,
            chunking_strategy="semantic",
            embedding_model=mock_embedding_model,
            min_chunk_size=30,
            breakpoint_percentile=70.0,
        )
        chunks, metadatas, ids = loader.load_from_directory(str(tmp_path))

        assert len(chunks) >= 1
        assert len(chunks) == len(metadatas) == len(ids)
        assert all(m["source"] == "speech.txt" for m in metadatas)
        assert all("speech_chunk_" in cid for cid in ids)
