# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only what's needed to compile deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY dataset_builder/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="SynthDataLab" \
      org.opencontainers.image.description="Industrial-Grade Synthetic Data Pipeline" \
      org.opencontainers.image.source="https://github.com/your-org/synthdatalab"

WORKDIR /app

# Copy installed packages from builder (no gcc in final image)
COPY --from=builder /install /usr/local

# Copy application source
COPY dataset_builder/ ./dataset_builder/

# Create a non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser \
    && mkdir -p dataset_builder/data/logs \
    && chown -R appuser:appuser /app

USER appuser

# Data directory is a volume so outputs persist across container restarts
VOLUME ["/app/dataset_builder/data"]

# Default env — override at runtime via --env-file or -e flags
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_MODEL=qwen3:4b \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/dataset_builder

ENTRYPOINT ["python3", "main.py"]
CMD ["--help"]
