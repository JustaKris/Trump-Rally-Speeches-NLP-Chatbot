"""NLP Service for text analysis and processing.

Provides n-gram analysis and other basic NLP operations.
This service aggregates functionality from utils for cleaner API separation.
"""

import logging
import re
from typing import Any, Dict, Optional

from speech_nlp.utils.io import (
    get_dataset_statistics,
    get_project_root,
    load_speeches_from_directory,
)
from speech_nlp.utils.text import extract_ngrams, get_stopwords, tokenize_text

logger = logging.getLogger(__name__)


class NLPService:
    """Service for NLP text analysis operations.

    Provides methods for n-gram analysis and other text processing tasks.
    """

    def __init__(self):
        """Initialize NLP service."""
        self.stopwords = get_stopwords()
        logger.info("NLP service initialized")

    def extract_ngrams(self, text: str, n: int = 2, top_n: int = 20) -> Dict[str, Any]:
        """Extract n-grams from text.

        Args:
            text: Input text
            n: N-gram size (2 for bigrams, 3 for trigrams, etc.)
            top_n: Number of top n-grams to return

        Returns:
            Dictionary with n-gram statistics
        """
        from collections import Counter

        tokens = tokenize_text(text)
        # Remove stopwords for better n-grams
        tokens = [t for t in tokens if t not in self.stopwords and t.isalpha()]

        ngrams = extract_ngrams(tokens, n=n)

        # Count and rank n-grams
        ngram_counts = Counter(ngrams)
        top_ngrams = ngram_counts.most_common(top_n)

        return {
            "n": n,
            "total_ngrams": len(ngrams),
            "unique_ngrams": len(list(set(ngrams))),
            "top_ngrams": [{"ngram": ngram, "count": count} for ngram, count in top_ngrams],
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the speech dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return get_dataset_statistics()

    def list_speeches(self) -> Dict[str, Any]:
        """List all speeches in the dataset.

        Returns:
            Dictionary with speech list and count
        """
        df = load_speeches_from_directory()
        speeches = df[["filename", "location", "month", "year", "word_count"]].to_dict("records")
        return {"total": len(speeches), "speeches": speeches}

    def get_speech_text(self, filename: str) -> Optional[Dict[str, Any]]:
        """Return the full text and metadata for a single speech file.

        Security: only filenames composed of alphanumeric characters, hyphens,
        and underscores followed by ``.txt`` are accepted.  The resolved path
        is also checked to be inside the speeches directory, guarding against
        any edge-case bypass.

        Args:
            filename: Bare filename of the speech (e.g. ``CincinnatiAug1_2019.txt``).

        Returns:
            Dict with ``filename``, ``location``, ``month``, ``year``,
            ``word_count``, and ``content``, or ``None`` if the file is not
            found or the filename fails validation.
        """
        # Allow only safe filenames: word-chars / hyphens + .txt
        if not re.match(r"^[A-Za-z0-9_\-]+\.txt$", filename):
            logger.warning("Speech request rejected — unsafe filename: %r", filename)
            return None

        speeches_dir = get_project_root() / "data" / "Donald Trump Rally Speeches"
        file_path = (speeches_dir / filename).resolve()

        # Confirm resolved path is still inside the speeches directory
        if not str(file_path).startswith(str(speeches_dir.resolve())):
            logger.warning("Speech request rejected — path traversal attempt: %r", filename)
            return None

        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        # Parse metadata from filename (same logic as load_speeches_from_directory)
        name_parts = filename.replace(".txt", "").split("_")
        location, month, year = filename, "", ""
        if len(name_parts) == 2:
            location_date, year = name_parts[0], name_parts[1]
            match = re.search(r"([A-Z][a-z]+)(\d+)", location_date)
            if match:
                location_month = location_date[: match.start(2)]
                location = location_month[:-3] if len(location_month) > 3 else location_month
                month = location_month[-3:] if len(location_month) > 3 else ""
            else:
                location = location_date

        return {
            "filename": filename,
            "location": location,
            "month": month,
            "year": year,
            "word_count": len(content.split()),
            "content": content,
        }
