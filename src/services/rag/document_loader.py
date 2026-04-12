"""Document loader component for RAG service.

Handles loading text documents from disk, chunking them into appropriate sizes,
and preparing them for embedding and vector storage.

Supports two chunking strategies:
- "fixed": Mechanical splitting at character boundaries (RecursiveCharacterTextSplitter)
- "semantic": Embedding-based splitting that detects topic boundaries between sentences
"""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename metadata extraction
# ---------------------------------------------------------------------------

_MONTH_ABBR = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

_FILENAME_RE = re.compile(rf"^(.+?)({_MONTH_ABBR})(\d{{1,2}})_(\d{{4}})\.txt$")

_MONTH_TO_INT: Dict[str, int] = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def extract_speech_metadata(filename: str) -> Dict[str, Any]:
    """Extract structured metadata from a speech transcript filename.

    Expected pattern: ``{Location}{MonthDay}_{Year}.txt``
    Examples: ``BattleCreekDec19_2019.txt``, ``Winston-SalemSep8_2020.txt``

    Args:
        filename: Name of the text file (not the full path).

    Returns:
        Dict with ``location``, ``year``, ``month``, ``day``, and ``date``
        (ISO-format string).  Returns an empty dict if the filename does not
        match the expected pattern.
    """
    match = _FILENAME_RE.match(filename)
    if not match:
        return {}

    raw_location, month_abbr, day_str, year_str = match.groups()

    # Convert CamelCase to spaced words (e.g. "BattleCreek" → "Battle Creek")
    # Preserves hyphens (e.g. "Winston-Salem" stays as-is)
    location = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw_location)

    year = int(year_str)
    month = _MONTH_TO_INT[month_abbr]
    day = int(day_str)

    return {
        "location": location,
        "year": year,
        "month": month,
        "day": day,
        "date": f"{year:04d}-{month:02d}-{day:02d}",
    }


# ---------------------------------------------------------------------------
# Sentence tokenization helper (lazy NLTK import)
# ---------------------------------------------------------------------------

_sent_tokenizer = None


def _get_sent_tokenizer():
    """Lazy-load the NLTK Punkt sentence tokenizer."""
    global _sent_tokenizer
    if _sent_tokenizer is None:
        import nltk

        try:
            from nltk.tokenize import PunktSentenceTokenizer

            _sent_tokenizer = PunktSentenceTokenizer()
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import PunktSentenceTokenizer

            _sent_tokenizer = PunktSentenceTokenizer()
    return _sent_tokenizer


class DocumentLoader:
    """Loads and chunks documents for RAG indexing.

    Supports two chunking strategies:
    - **fixed** (default): Uses RecursiveCharacterTextSplitter with configurable
      chunk_size / chunk_overlap.  Fast and deterministic.
    - **semantic**: Embeds each sentence with the provided SentenceTransformer,
      detects topic-shift breakpoints via cosine-similarity drops, and groups
      sentences into coherent chunks.  Falls back to fixed splitting for any
      group that exceeds *chunk_size*.
    """

    def __init__(
        self,
        chunk_size: int = 2048,
        chunk_overlap: int = 150,
        chunking_strategy: Literal["fixed", "semantic"] = "fixed",
        embedding_model: Optional["SentenceTransformer"] = None,
        min_chunk_size: int = 256,
        similarity_threshold: Optional[float] = None,
        breakpoint_percentile: float = 90.0,
    ):
        """Initialize document loader.

        Args:
            chunk_size: Maximum size of text chunks in characters (~512-768 tokens).
            chunk_overlap: Overlap between chunks in characters (~100-150 tokens).
                Used by fixed splitting and as a fallback for oversized semantic groups.
            chunking_strategy: "fixed" for character-based splitting,
                "semantic" for embedding-based splitting.
            embedding_model: SentenceTransformer instance (required for semantic strategy).
            min_chunk_size: Minimum chunk size in characters.  Semantic groups
                smaller than this are merged with their neighbours.
            similarity_threshold: Absolute cosine-similarity threshold for breakpoints.
                If *None* (default), breakpoints are detected automatically using
                *breakpoint_percentile* on the similarity distribution.
            breakpoint_percentile: Percentile of similarity drops that count as
                breakpoints (0-100).  Higher = fewer, larger chunks.
                Only used when *similarity_threshold* is None.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.breakpoint_percentile = breakpoint_percentile

        if chunking_strategy == "semantic" and embedding_model is None:
            logger.warning(
                "Semantic chunking requested but no embedding_model provided — "
                "falling back to fixed chunking"
            )
            self.chunking_strategy = "fixed"

        # Fixed splitter (also used as fallback for oversized semantic groups)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            f"DocumentLoader initialized: strategy={self.chunking_strategy}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_directory(self, data_dir: str) -> Tuple[List[str], List[dict], List[str]]:
        """Load and chunk all text documents from a directory.

        Args:
            data_dir: Directory containing text files

        Returns:
            Tuple of (chunks, metadatas, chunk_ids)

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        all_chunks = []
        all_metadatas = []
        all_ids = []
        files_processed = 0

        for file_path in data_path.glob("*.txt"):
            try:
                chunks, metadatas, ids = self._load_file(file_path)
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                all_ids.extend(ids)
                files_processed += 1
                logger.debug(f"Loaded {len(chunks)} chunks from {file_path.name}")

            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                continue

        logger.info(f"Loaded {len(all_chunks)} chunks from {files_processed} documents")
        return all_chunks, all_metadatas, all_ids

    def _load_file(self, file_path: Path) -> Tuple[List[str], List[dict], List[str]]:
        """Load and chunk a single text file.

        Args:
            file_path: Path to text file

        Returns:
            Tuple of (chunks, metadatas, chunk_ids)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split document into chunks using the configured strategy
        chunks = self.chunk_text(content)

        # Extract structured metadata from filename (location, date, year, etc.)
        speech_metadata = extract_speech_metadata(file_path.name)

        # Create metadata for each chunk
        metadatas = []
        chunk_ids = []

        for i, _chunk in enumerate(chunks):
            chunk_id = f"{file_path.stem}_chunk_{i}"
            chunk_ids.append(chunk_id)

            metadata: Dict[str, Any] = {
                "source": file_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            metadata.update(speech_metadata)

            metadatas.append(metadata)

        return chunks, metadatas, chunk_ids

    def chunk_text(self, text: str) -> List[str]:
        """Chunk a single text string using the configured strategy.

        Useful for processing ad-hoc text without file I/O.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return [] if not text else self.text_splitter.split_text(text)

        if self.chunking_strategy == "semantic":
            return self._semantic_split_text(text)
        return self.text_splitter.split_text(text)

    # ------------------------------------------------------------------
    # Semantic chunking internals
    # ------------------------------------------------------------------

    def _semantic_split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks.

        Algorithm:
        1. Tokenize into sentences (NLTK Punkt).
        2. Embed every sentence with the shared SentenceTransformer.
        3. Compute cosine similarity between consecutive sentence embeddings.
        4. Detect breakpoints where similarity drops below a threshold
           (either an absolute value or a percentile-based value).
        5. Group sentences between breakpoints.
        6. Merge small groups with neighbours; split oversized groups
           using the fixed RecursiveCharacterTextSplitter as a fallback.
        """
        # 1. Sentence tokenization
        tokenizer = _get_sent_tokenizer()
        sentences: List[str] = tokenizer.tokenize(text)

        if len(sentences) <= 1:
            # Single sentence or very short text — use fixed splitter
            return self.text_splitter.split_text(text)

        # 2. Embed sentences
        embeddings = self.embedding_model.encode(  # type: ignore[union-attr]
            sentences, convert_to_numpy=True, show_progress_bar=False
        )

        # 3. Cosine similarity between consecutive sentences
        similarities = self._consecutive_cosine_similarities(embeddings)

        # 4. Detect breakpoints
        breakpoints = self._detect_breakpoints(similarities)

        # 5. Group sentences by breakpoints
        groups = self._group_sentences(sentences, breakpoints)

        # 6. Enforce size constraints (merge small, split large)
        chunks = self._enforce_size_constraints(groups)

        logger.debug(f"Semantic chunking: {len(sentences)} sentences → {len(chunks)} chunks")
        return chunks

    @staticmethod
    def _consecutive_cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between each pair of consecutive embeddings."""
        # Normalise rows to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        normed = embeddings / norms

        # Dot product of consecutive normalised vectors = cosine similarity
        similarities = np.sum(normed[:-1] * normed[1:], axis=1)
        return similarities

    def _detect_breakpoints(self, similarities: np.ndarray) -> List[int]:
        """Return indices where a new chunk should start.

        An index *i* in the breakpoints list means *there is a topic shift
        between sentence i and sentence i+1*.

        If ``similarity_threshold`` is set, any similarity below it triggers a
        breakpoint.  Otherwise, the ``breakpoint_percentile`` is used over the
        *dissimilarity* distribution (1 − similarity).
        """
        if len(similarities) == 0:
            return []

        if self.similarity_threshold is not None:
            threshold = self.similarity_threshold
        else:
            # Use percentile on dissimilarity (higher dissim → more likely breakpoint)
            dissimilarities = 1.0 - similarities
            threshold_dissim = float(np.percentile(dissimilarities, self.breakpoint_percentile))
            # Convert back to similarity threshold
            threshold = 1.0 - threshold_dissim

        breakpoints = [i for i, sim in enumerate(similarities) if sim < threshold]
        return breakpoints

    @staticmethod
    def _group_sentences(sentences: List[str], breakpoints: List[int]) -> List[str]:
        """Group sentences into chunks based on breakpoint indices."""
        groups: List[str] = []
        start = 0
        for bp in sorted(breakpoints):
            # bp is the index of the *gap* between sentence[bp] and sentence[bp+1]
            end = bp + 1
            group_text = " ".join(sentences[start:end]).strip()
            if group_text:
                groups.append(group_text)
            start = end

        # Remainder after last breakpoint
        tail = " ".join(sentences[start:]).strip()
        if tail:
            groups.append(tail)

        return groups

    def _enforce_size_constraints(self, groups: List[str]) -> List[str]:
        """Merge small groups and split oversized groups.

        - Groups shorter than ``min_chunk_size`` are merged with the previous
          group (or the next one if they are the first group).
        - Groups longer than ``chunk_size`` are split using the fixed
          RecursiveCharacterTextSplitter as a deterministic fallback.
        """
        if not groups:
            return groups

        # --- Merge small groups ---
        merged: List[str] = []
        for group in groups:
            if merged and len(group) < self.min_chunk_size:
                merged[-1] = merged[-1] + " " + group
            else:
                merged.append(group)

        # Edge case: if the very first group was small and nothing to prepend to
        if len(merged) > 1 and len(merged[0]) < self.min_chunk_size:
            merged[1] = merged[0] + " " + merged[1]
            merged.pop(0)

        # Edge case: if the very last group is still small, merge it back
        if len(merged) > 1 and len(merged[-1]) < self.min_chunk_size:
            merged[-2] = merged[-2] + " " + merged[-1]
            merged.pop(-1)

        # --- Split oversized groups ---
        final: List[str] = []
        for group in merged:
            if len(group) <= self.chunk_size:
                final.append(group)
            else:
                # Fallback to fixed splitting for this group
                sub_chunks = self.text_splitter.split_text(group)
                final.extend(sub_chunks)

        return final
