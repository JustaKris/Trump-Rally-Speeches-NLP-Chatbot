# ====================
# Stage 1: Builder
# ====================
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files for layer caching
COPY pyproject.toml uv.lock /app/

# Create a venv for the app dependencies
RUN uv venv /opt/venv

# Export production deps, strip GPU packages, install CPU-only alternatives
RUN uv export --frozen --no-hashes --no-emit-project \
        --no-group dev --no-group docs --no-group notebooks \
        > requirements.txt && \
    # Remove GPU packages and torch (will install CPU-only torch separately)
    sed -i -E '/^(torch==|torchvision|torchaudio|triton|nvidia[-_])/d' requirements.txt && \
    # Install CPU-only PyTorch (~170MB vs ~2.5GB with CUDA)
    uv pip install --python /opt/venv/bin/python --no-cache \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    # Install remaining production dependencies from PyPI
    uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt

# ====================
# Stage 2: Runtime
# ====================
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Allow Python to find the src package
ENV PYTHONPATH="/app/src"

# Copy app code
COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Pre-download all HuggingFace models based on configuration
# Models are cached in the image for offline use (no runtime downloads)
ENV ENVIRONMENT=production
RUN python scripts/download_models.py

# Minimal NLTK data
RUN python -m nltk.downloader punkt stopwords punkt_tab

EXPOSE ${PORT}

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
