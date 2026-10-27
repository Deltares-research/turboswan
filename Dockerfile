FROM containers.deltares.nl/docker-proxy/python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# TODO: remove, should be on Nexus and installed through extra index-url, see MLPS-245
COPY deltares_ai_platform_sdk ./deltares_ai_platform_sdk

# Dependencies only (cached layer; reinstalled when pyproject.toml or uv.lock change)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --extra gpu --no-install-project --no-editable

# Versioned turboswan package from src/ (readme is required by the build metadata)
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra gpu --no-editable

# Configs are available to the code at /app/configs (relative to the working dir)
COPY configs ./configs
