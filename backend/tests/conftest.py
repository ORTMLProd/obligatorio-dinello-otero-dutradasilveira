"""Shared pytest fixtures."""

from __future__ import annotations

# Pre-load XGBoost's native C extension before any test module (including
# test_clip_model) imports torch. On macOS ARM, torch initialises Metal/MPS
# dispatch queues; if XGBoost's OpenMP (libomp) then loads into the same process
# it segfaults. Loading XGBoost first avoids the clash.
try:
    import xgboost as _xgb  # noqa: F401  (import for side-effect only)
except ImportError:
    pass

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    """A TestClient bound to the FastAPI app."""
    return TestClient(app)
