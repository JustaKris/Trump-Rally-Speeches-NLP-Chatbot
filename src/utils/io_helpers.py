"""Utility functions for data loading and statistics.

This module provides helper functions for loading speech data and
computing aggregate statistics.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_speeches_from_directory(data_dir: Optional[Any] = None) -> pd.DataFrame:
    """Load all speeches from the data directory.

    Args:
        data_dir: Path to directory containing speech text files.
                 If None, uses default data/Donald Trump Rally Speeches/

    Returns:
        DataFrame with columns: filename, location, month, year, content, word_count
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "Donald Trump Rally Speeches"
    else:
        data_dir = Path(data_dir)

    speeches = []

    for file_path in sorted(data_dir.glob("*.txt")):
        # Parse filename (e.g., "GreenvilleJul17_2019.txt")
        filename = file_path.name
        name_parts = filename.replace(".txt", "").split("_")

        if len(name_parts) == 2:
            location_date = name_parts[0]
            year = name_parts[1]

            # Extract location and month from the first part
            # Find where the month starts (first uppercase letter after beginning)
            import re

            match = re.search(r"([A-Z][a-z]+)(\d+)", location_date)
            if match:
                location_month = location_date[: match.start(2)]
                # Extract location (everything before last 3 chars which is month abbreviation)
                location = location_month[:-3] if len(location_month) > 3 else location_month
                month = location_month[-3:] if len(location_month) > 3 else ""
            else:
                location = location_date
                month = ""

            # Read content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            speeches.append(
                {
                    "filename": filename,
                    "location": location,
                    "month": month,
                    "year": year,
                    "content": content,
                    "word_count": len(content.split()),
                }
            )

    return pd.DataFrame(speeches)


def get_dataset_statistics() -> Dict[str, Any]:
    """Get aggregate statistics about the speech dataset.

    Returns:
        Dictionary with dataset statistics
    """
    df = load_speeches_from_directory()

    return {
        "total_speeches": len(df),
        "total_words": int(df["word_count"].sum()),
        "avg_words_per_speech": float(df["word_count"].mean()),
        "date_range": {
            "start": f"{df['month'].iloc[0]} {df['year'].iloc[0]}",
            "end": f"{df['month'].iloc[-1]} {df['year'].iloc[-1]}",
        },
        "years": df["year"].unique().tolist(),
        "locations": df["location"].unique().tolist(),
    }
