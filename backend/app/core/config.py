"""Explicit, server-only local application settings."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    repository_root: Path = Path(__file__).resolve().parents[3]
    load_model: bool = True
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173", "http://localhost:5173",
    )

    @field_validator("cors_origins")
    @classmethod
    def explicit_origins(cls, origins: tuple[str, ...]) -> tuple[str, ...]:
        for origin in origins:
            parsed = urlsplit(origin)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.path
                    or parsed.query or parsed.fragment or "*" in origin):
                raise ValueError("CORS origins must be explicit HTTP(S) origins")
            _ = parsed.port  # Reject malformed ports.
        return origins

    @classmethod
    def from_environment(cls) -> "Settings":
        enabled = os.getenv("GRAPHNIDS_LOAD_MODEL", "true").lower()
        if enabled not in {"true", "false"}:
            raise ValueError("GRAPHNIDS_LOAD_MODEL must be true or false")
        raw = os.getenv("GRAPHNIDS_CORS_ORIGINS")
        options = {"load_model": enabled == "true"}
        if raw is not None:
            values = json.loads(raw)
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError("GRAPHNIDS_CORS_ORIGINS must be a JSON string array")
            options["cors_origins"] = tuple(values)
        return cls(**options)
