#!/usr/bin/env python3
"""Pre-download all HuggingFace models specified in configuration.

This script reads the application configuration and downloads all required
models to cache them in the Docker image, avoiding runtime downloads.

Usage:
    python scripts/download_models.py [--config-file CONFIG_PATH]
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import Settings
from core.logging_config import configure_logging, get_logger

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

    logger.info("=" * 70)
    logger.info("✓ All models downloaded successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
