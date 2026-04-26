#!/usr/bin/env python3
"""Pre-download all HuggingFace models specified in configuration.

This script reads the application configuration and downloads all required
models to cache them in the Docker image, avoiding runtime downloads.

Usage:
    python scripts/download_models.py [--config-file CONFIG_PATH]
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from speech_nlp.config.logging import configure_logging, get_logger
from speech_nlp.config.settings import Settings

# Configure logging
configure_logging(level="INFO", use_json=False)
logger = get_logger(__name__)


def download_transformers_model(model_name: str) -> None:
    """Download a transformers model and tokenizer.

    Args:
        model_name: HuggingFace model identifier
    """
    logger.info(f"Downloading transformers model: {model_name}")
    try:
        # Model names are from configuration, not user input
        AutoTokenizer.from_pretrained(model_name)  # nosec B615
        AutoModelForSequenceClassification.from_pretrained(model_name)  # nosec B615
        logger.info(f"✓ Successfully downloaded: {model_name}")
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        raise


def download_sentence_transformer(model_name: str) -> None:
    """Download a sentence-transformers model.

    Args:
        model_name: SentenceTransformer model identifier
    """
    logger.info(f"Downloading sentence-transformers model: {model_name}")
    try:
        SentenceTransformer(model_name)
        logger.info(f"✓ Successfully downloaded: {model_name}")
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        raise


def download_cross_encoder(model_name: str) -> None:
    """Download a cross-encoder model.

    Args:
        model_name: CrossEncoder model identifier
    """
    logger.info(f"Downloading cross-encoder model: {model_name}")
    try:
        CrossEncoder(model_name)
        logger.info(f"✓ Successfully downloaded: {model_name}")
    except Exception as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        raise


def download_spacy_model(model_name: str) -> None:
    """Download a spaCy NER model.

    Attempts to load the model first; if not found, downloads it via
    ``spacy.cli.download``.  Skips gracefully if spaCy is not installed.

    Args:
        model_name: spaCy model name (e.g. ``en_core_web_sm``)
    """
    logger.info(f"Downloading spaCy model: {model_name}")
    try:
        import spacy  # type: ignore[import-not-found]

        # Try loading first — might already be installed
        try:
            spacy.load(model_name)
            logger.info(f"✓ spaCy model already available: {model_name}")
            return
        except OSError:
            pass  # Not installed yet — fall through to download

        # Use the current Python executable's pip to install the model package.
        # This works in Docker where no standalone pip/uv binary is on PATH,
        # because sys.executable always points to the active interpreter.
        subprocess.check_call(  # nosec B603
            [sys.executable, "-m", "pip", "install", "--quiet", model_name],
        )
        logger.info(f"✓ Successfully downloaded spaCy model: {model_name}")
    except ImportError:
        logger.warning(
            "spaCy not installed — skipping NER model download. "
            "Install the 'ner' optional group to enable NER: uv sync --group ner"
        )
    except Exception as e:
        logger.error(f"✗ Failed to download spaCy model {model_name}: {e}")
        raise


def main():
    """Download all models specified in configuration."""
    parser = argparse.ArgumentParser(
        description="Pre-download HuggingFace models from configuration"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to config file (defaults to environment-based config)",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Pre-downloading HuggingFace models for offline use")
    logger.info("=" * 70)

    # Load settings
    if args.config_file:
        # Custom config file handling would go here
        logger.warning("Custom config file not yet implemented, using default settings")

    settings = Settings()

    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Config file: configs/{settings.environment}.yaml")
    logger.info("")

    # Download sentiment models
    logger.info("Downloading sentiment analysis models...")
    download_transformers_model(settings.models.sentiment_model_name)
    download_transformers_model(settings.models.emotion_model_name)
    logger.info("")

    # Download embedding model
    logger.info("Downloading embedding models...")
    download_sentence_transformer(settings.models.embedding_model_name)
    logger.info("")

    # Download reranker model
    logger.info("Downloading reranker models...")
    download_cross_encoder(settings.models.reranker_model_name)
    logger.info("")

    # Download spaCy NER model (optional — skips if spaCy not installed)
    if settings.models.ner_enabled:
        logger.info("Downloading NER models...")
        download_spacy_model(settings.models.ner_model_name)
        logger.info("")

    logger.info("=" * 70)
    logger.info("✓ All models downloaded successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
