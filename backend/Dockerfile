# syntax=docker/dockerfile:1
# Multi-stage build using uv. Pattern verified against astral-sh/uv-docker-example.

# ── Builder: resolve and install dependencies into /app/.venv ────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Install deps first (cached layer) without the project code.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then copy the source. `package = false` means uv installs only dependencies.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ── Runtime: slim image with just the venv + source (no uv) ──────────────────
# Base MUST match the builder's Python (3.12, bookworm) or the venv paths break.
FROM python:3.12-slim-bookworm
RUN groupadd --system --gid 999 nonroot \
    && useradd --system --gid 999 --uid 999 --create-home nonroot
COPY --from=builder --chown=nonroot:nonroot /app /app
ENV PATH="/app/.venv/bin:$PATH"
USER nonroot
WORKDIR /app
EXPOSE 8000
CMD ["fastapi", "run", "src/api/main.py", "--host", "0.0.0.0", "--port", "8000"]
