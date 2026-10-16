"""Web search, so Jarvis can answer from the live web instead of guessing.

One tool, `web_search`, backed by Tavily (https://tavily.com: a search API built for AI assistants; the free plan
is 1,000 credits a month with no card, and it stops, rather than charges, when they run out). It returns a handful of
results with a short extract of each page. Only the query leaves this machine, so Jarvis is told never to put
his personal details in it. What comes back is wrapped as untrusted data (web pages are where hidden instructions live)
and the tool can only read: it never opens a page, follows a link or acts on one.

The key is TAVILY_API_KEY in .env. A local counter (data/websearch_state.json) keeps Jarvis under `monthly_limit`
searches a month so a loop can't use up the allowance. Settings: [websearch] in config.toml.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.parse
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

from .. import events
from ..llm import ToolSpec
from ..persona import LOCAL
from .base import Tool, ToolError, read_state, untrusted, write_state
from .web import request_json

log = logging.getLogger(__name__)

URL = "https://api.tavily.com/search"
EXTRACT_CHARS = 320
MAX_RESULTS = 5
CACHE_S = 600


def short_date(text) -> str:
    """Tavily's published_date as "9 Oct" ("Fri, 09 Oct 2026 10:00:00 GMT" or ISO), or "" if it isn't a date."""
    text = str(text or "").strip()
    for parse in (parsedate_to_datetime, lambda t: datetime.fromisoformat(t.replace("Z", "+00:00"))):
        try:
            d = parse(text)
            return f"{d.day} {d.strftime('%b')}"
        except (TypeError, ValueError, IndexError):
            continue
    return ""


def domain_of(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


class WebSearchConnector:
    name = "websearch"

    def __init__(self, api_key: str, *, data_dir: Path | None = None, max_results: int = MAX_RESULTS,
                 monthly_limit: int = 900, hud_card: bool = True, request: Callable = request_json,
                 now: Callable[[], datetime] | None = None, clock: Callable[[], float] = time.monotonic):
        self.api_key = api_key
        self.max_results = max(1, min(10, max_results))
        self.monthly_limit = monthly_limit
        self.hud_card = hud_card
        self._request = request
        self._now = now or (lambda: datetime.now(LOCAL))
        self._clock = clock
        self.state_file = data_dir / "websearch_state.json" if data_dir else None
        self._cache: dict[tuple, tuple[float, str, dict]] = {}

    # ---- the tool

    def tools(self) -> list[Tool]:
        spec = ToolSpec(
            name="web_search",
            description=("Search the web for current or unfamiliar facts: news, scores, prices, opening times, recent "
                         "releases, anything you aren't sure of. Use it rather than guess. Say where the answer came "
                         "from. Never put his personal details (emails, address, health) in the query."),
            parameters={"type": "object", "properties": {
                "query": {"type": "string", "description": "a short search phrase, as you'd type it"},
                "news": {"type": "boolean", "description": "true for recent news and events"}},
                "required": ["query"]})
        return [Tool(spec=spec, run=self.web_search, notice="Searching the web", focus="search", reads_outside=True)]

    def web_search(self, args: dict) -> str:
        query = " ".join(str(args.get("query") or "").split())[:300]
        if len(query) < 2:
            raise ToolError("what should I search for?")
        news = bool(args.get("news"))
        key = (query.lower(), news)
        hit = self._cache.get(key)
        if hit and self._clock() - hit[0] < CACHE_S:
            self._publish(hit[2])
            return hit[1]
        self._spend()
        body = {"query": query, "search_depth": "basic", "max_results": self.max_results,
                "topic": "news" if news else "general", "include_answer": False}
        if news:
            body["time_range"] = "week"
        try:
            answer = self._request("POST", URL, body, {"Authorization": f"Bearer {self.api_key}"}) or {}
        except urllib.error.HTTPError as e:
            log.warning("web search failed: HTTP %s", e.code)
            if e.code in (401, 403):
                raise ToolError("the search service refused Jarvis's key (check TAVILY_API_KEY).") from e
            if e.code in (429, 432, 433):
                raise ToolError("the search service says its allowance is used up or it's busy; try later.") from e
            raise ToolError("the search service answered with an error.") from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.warning("web search failed: %s", type(e).__name__)
            raise ToolError("the search service isn't answering (is the internet up?).") from e
        results = [r for r in answer.get("results") or [] if r.get("url")][: self.max_results]
        card = {"id": "search", "query": query, "news": news, "left": self.left(),
                "items": [{"title": _clean(r.get("title"), 110), "site": domain_of(r["url"]),
                           "date": short_date(r.get("published_date"))} for r in results]}
        self._publish(card)
        if not results:
            text = untrusted("web search", f'No results for "{query}".')
        else:
            lines = [f'Web search for "{query}" ({"news, past week" if news else "general"}), {len(results)} results (name the outlet you rely on, e.g. "according to Insider Gaming"; if they disagree or are old, say so):']
            for i, r in enumerate(results, 1):
                date = f" | {short_date(r.get('published_date'))}" if short_date(r.get("published_date")) else ""
                lines.append(f"{i}. {_clean(r.get('title'), 110)} | {domain_of(r['url'])}{date} | "
                             f"{_clean(r.get('content'), EXTRACT_CHARS)}")
            text = untrusted("web search results", "\n".join(lines))
        self._cache[key] = (self._clock(), text, card)
        return text

    # ---- the monthly allowance

    def _used(self) -> int:
        s = read_state(self.state_file)
        return int(s.get("used", 0)) if s.get("month") == self._month() else 0

    def _month(self) -> str:
        return self._now().strftime("%Y-%m")

    def left(self) -> int:
        return max(0, self.monthly_limit - self._used())

    def _spend(self) -> None:
        used = self._used()
        if used >= self.monthly_limit:
            raise ToolError(f"Jarvis has used its {self.monthly_limit} web searches for this month; "
                            "they start again on the first.")
        write_state(self.state_file, {"month": self._month(), "used": used + 1})

    def _publish(self, card: dict) -> None:
        if self.hud_card:
            events.publish("card", **{**card, "left": self.left()})

    def status(self) -> str:
        return f"web search (Tavily, {self.left()} of {self.monthly_limit} left this month)"


def _clean(text, limit: int) -> str:
    text = " ".join(str(text or "").lstrip("# ").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def websearch_from_config(cfg) -> WebSearchConnector | None:
    ws = cfg.websearch
    if not ws.get("enabled", True):
        return None
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        return None
    return WebSearchConnector(key, data_dir=cfg.root / "data", max_results=int(ws.get("max_results", MAX_RESULTS)),
                              monthly_limit=int(ws.get("monthly_limit", 900)), hud_card=bool(ws.get("hud_card", True)))
