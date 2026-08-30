from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class ClientContext:
    client_id: str


def get_client_context(
    x_client_id: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> ClientContext:
    if settings.is_development and settings.allow_dev_client_header and x_client_id:
        return ClientContext(client_id=x_client_id[:80])
    return ClientContext(client_id=settings.private_client_id)


def require_development(settings: Settings = Depends(get_settings)) -> None:
    if not settings.is_development:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
