FROM ghcr.io/astral-sh/uv:0.6-python3.13-bookworm-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy source and data
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY api.py main.py ./

# Install the project itself
RUN uv sync --frozen

# Build ChromaDB vector stores at build time (data is static)
RUN uv run python scripts/load_documents.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
