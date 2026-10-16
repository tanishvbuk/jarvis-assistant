"""Fetching from public web APIs and feeds: one small helper, no extra dependencies."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

USER_AGENT = "Jarvis/0.1 (personal assistant; private, non-commercial)"
MAX_BYTES = 3_000_000


def fetch_url(url: str, timeout: float = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec: URLs come from config.toml or fixed APIs
        data = r.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f"response over {MAX_BYTES} bytes")
    return data


def get_json(url: str, params: dict | None = None, timeout: float = 20):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    return json.loads(fetch_url(url, timeout))


def request_json(method: str, url: str, body=None, headers: dict | None = None, timeout: float = 20):
    """A JSON request with a JSON body (or none) and extra headers, e.g. a session cookie."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"User-Agent": USER_AGENT, "Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec: the URL comes from config.toml
        raw = r.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError(f"response over {MAX_BYTES} bytes")
    return json.loads(raw) if raw else None
