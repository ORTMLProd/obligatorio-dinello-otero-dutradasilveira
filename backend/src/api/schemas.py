"""Pydantic schemas — the strict contract of the API (invariant 4).

Every response/request model forbids extra fields (``extra="forbid"``). The same
types back ``/predict`` and ``/predict/batch`` once those arrive in Fase 2.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: str
    version: str


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model (stub in Fase 0)."""

    # ``protected_namespaces=()`` silences pydantic's warning about the ``model_`` prefix.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_loaded: bool
    version: str | None
    message: str
