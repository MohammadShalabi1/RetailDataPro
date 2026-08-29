from __future__ import annotations

import ipaddress
import socket
from enum import Enum
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field


class WebsiteFetchStatus(str, Enum):
    success = "success"
    blocked = "blocked"
    fetch_error = "fetch_error"


class WebsiteFetchRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2_000)


class WebsiteFetchResult(BaseModel):
    status: WebsiteFetchStatus
    url: str
    content: str = ""
    content_type: str | None = None
    error_code: str | None = None


class SafeWebsiteFetcher:
    def __init__(
        self,
        timeout_seconds: float = 5.0,
        max_response_bytes: int = 500_000,
        allowed_content_types: tuple[str, ...] = ("text/html", "text/plain", "application/json"),
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._allowed_content_types = allowed_content_types

    async def fetch(self, request: WebsiteFetchRequest) -> WebsiteFetchResult:
        safety_error = _validate_public_url(request.url)
        if safety_error:
            return WebsiteFetchResult(status=WebsiteFetchStatus.blocked, url=request.url, error_code=safety_error)

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds, follow_redirects=False) as client:
                response = await client.get(request.url)
                if response.is_redirect:
                    redirect_url = response.headers.get("location", "")
                    redirect_error = _validate_public_url(str(response.url.join(redirect_url)))
                    if redirect_error:
                        return WebsiteFetchResult(status=WebsiteFetchStatus.blocked, url=request.url, error_code="dangerous_redirect")
                    response = await client.get(str(response.url.join(redirect_url)))
                content_type = response.headers.get("content-type", "").split(";")[0].lower()
                if not any(content_type.startswith(allowed) for allowed in self._allowed_content_types):
                    return WebsiteFetchResult(status=WebsiteFetchStatus.blocked, url=str(response.url), error_code="unsupported_content_type")
                content = response.content[: self._max_response_bytes].decode(response.encoding or "utf-8", errors="replace")
                return WebsiteFetchResult(status=WebsiteFetchStatus.success, url=str(response.url), content=content, content_type=content_type)
        except Exception:
            return WebsiteFetchResult(status=WebsiteFetchStatus.fetch_error, url=request.url, error_code="fetch_error")


def _validate_public_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "unsupported_protocol"
    if not parsed.hostname:
        return "missing_host"
    host = parsed.hostname.lower()
    if host in {"localhost", "0.0.0.0"}:
        return "private_host"
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return "dns_lookup_failed"
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return "private_network"
    return None
