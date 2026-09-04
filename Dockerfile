FROM python:3.12-slim-trixie


COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY rag/ ./rag/
COPY api/ ./api/
COPY ui/ ./ui/
COPY data/*.txt ./data/

ENV PATH="/app/.venv/bin:$PATH"