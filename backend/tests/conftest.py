"""Shared pytest fixtures."""

from __future__ import annotations

import os

# torch and xgboost each bundle their own OpenMP runtime (libomp). On macOS ARM,
# running a torch CPU conv (the ResNet18 forward in test_clip_model) in the same
# process as xgboost segfaults: the two OpenMP runtimes clash. Forcing a single
# OpenMP thread before either native lib loads avoids it. Must run before the imports
# below. The standalone training script does not load this conftest, so real training
# keeps full threading.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Pre-load XGBoost's native extension before any test module imports torch, so the
# load order is deterministic alongside the single-thread setting above.
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
