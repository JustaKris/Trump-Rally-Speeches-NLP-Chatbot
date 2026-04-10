"""Quick smoke test for semantic chunking on real speech data."""

import statistics
import sys

from sentence_transformers import SentenceTransformer

from src.services.rag.document_loader import DocumentLoader

DATA_DIR = "data/Donald Trump Rally Speeches"


def main():
    """Run smoke test comparing fixed vs semantic chunking on real speech data."""
    print("Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("Creating semantic DocumentLoader...")
    loader = DocumentLoader(
        chunk_size=2048,
        chunk_overlap=150,
        chunking_strategy="semantic",
        embedding_model=model,
        min_chunk_size=256,
        breakpoint_percentile=90.0,
    )

    print("Loading speeches with semantic chunking...")
    chunks, metadatas, ids = loader.load_from_directory(DATA_DIR)

    sources = {m["source"] for m in metadatas}
    print(f"\nResult: {len(chunks)} chunks from {len(sources)} files")

    sizes = [len(c) for c in chunks]
    print(
        f"Chunk sizes: min={min(sizes)}, max={max(sizes)}, "
        f"mean={statistics.mean(sizes):.0f}, median={statistics.median(sizes):.0f}"
    )

    # Compare with fixed chunking
    print("\n--- Comparing with fixed chunking ---")
    fixed_loader = DocumentLoader(chunk_size=2048, chunk_overlap=150, chunking_strategy="fixed")
    fixed_chunks, _, _ = fixed_loader.load_from_directory(DATA_DIR)
    fixed_sizes = [len(c) for c in fixed_chunks]
    print(
        f"Fixed:    {len(fixed_chunks)} chunks, min={min(fixed_sizes)}, "
        f"max={max(fixed_sizes)}, mean={statistics.mean(fixed_sizes):.0f}"
    )
    print(
        f"Semantic: {len(chunks)} chunks, min={min(sizes)}, "
        f"max={max(sizes)}, mean={statistics.mean(sizes):.0f}"
    )

    # Show first 3 semantic chunks (truncated)
    print("\n--- Sample semantic chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        print(f"\nChunk {i} ({len(chunk)} chars):\n{preview}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
