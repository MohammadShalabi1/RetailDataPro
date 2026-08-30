from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app


def test_production_disables_developer_backend_routes(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    app = create_app()

    try:
        client = TestClient(app)
        assert client.get("/api/observability/traces").status_code == 404
        assert client.get("/api/evaluations/latest").status_code == 404
        assert client.post("/api/insights/generate").status_code == 404
        assert client.post("/api/reports/weekly-brief").status_code == 404
    finally:
        get_settings.cache_clear()


def test_client_ui_does_not_render_obvious_internal_labels() -> None:
    root = Path(__file__).resolve().parents[2]
    visible_ui = "\n".join(
        [
            (root / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8"),
            (root / "frontend" / "src" / "styles.css").read_text(encoding="utf-8"),
        ]
    ).lower()
    forbidden = [
        "cache hit",
        "cache miss",
        "rrf",
        "embedding",
        "pgvector",
        "multi_source",
        "tool gateway",
        "model router",
        "token count",
        "chunk_id",
        "source_id",
        "sql query",
        "trace",
    ]

    assert [term for term in forbidden if term in visible_ui] == []
