"""Mobile games news digest from public feeds, read-only.

Sources are listed in config.toml under [games.sources.*]:
  - kind "rss": an RSS or Atom feed (GamesPress, Gamezebo, Droid Gamers, ...).
  - kind "pokemongo": the Pokémon GO news page, which has no feed. Each post card
    on it has a link, a title and a timestamp; only those are read.
A source with role "covered" (a reference outlet) is never reported: its recent
headlines are only used to mark other items as already covered.

Feeds are polled in the background (tick()), at each source's own interval.
What's been seen is kept in data/games.json for keep_days, so nothing is
reported twice and the digest can say what's new since it was last read.
Everything fetched is untrusted text.
"""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable

from .. import events
from ..llm import ToolSpec
from ..persona import LOCAL, spoken
from .base import Tool, untrusted
from .web import fetch_url

log = logging.getLogger(__name__)

SUMMARY_CHARS = 220
MAX_LISTED = 10
NEW_WITHIN = timedelta(days=7)  # an item published longer ago than this never counts as new

MOBILE = re.compile(r"\b(ios|ipados|android|iphone|ipad|mobile|smartphones?|app ?store|google play|play store|"
                    r"apple arcade|netflix games)\b", re.I)
# Guides, codes and puzzle answers are not news.
GUIDES = re.compile(r"\b(tier lists?|codes|today'?s? answers?|guides?|walkthrough|locations?|cheats)\b|^how to\b|"
                    r"\[(best|how)\b", re.I)
# Words too common in games headlines to say two headlines are about the same game.
COMMON = set("""the and for with from into your you our out now new its this that are was will has have get gets
game games gaming mobile ios android iphone ipad update updates event events launch launches launched launching
release releases released season seasons available coming soon trailer free week weekly best top review reviews
first part global worldwide pre-register pre-registration preregistration today players player edition version
anniversary collab collaboration celebrates celebrate brings adds reveals revealed announced announces announce
official how what why when who all more just big here live store app apps play pc console switch steam xbox
playstation nintendo january february march april may june july august september october november
december monday tuesday wednesday thursday friday saturday sunday""".split())


@dataclass
class Source:
    name: str
    label: str
    kind: str                 # "rss" or "pokemongo"
    url: str
    every_minutes: int = 180
    mobile_filter: bool = False
    role: str = "news"        # "news" or "covered"


def sources_from_config(games: dict) -> list[Source]:
    out = []
    for name, s in (games.get("sources") or {}).items():
        if not s.get("url") or s.get("enabled", True) is False:
            continue
        out.append(Source(name=name, label=s.get("label", name), kind=s.get("kind", "rss"), url=s["url"],
                          every_minutes=int(s.get("every_minutes", 180)),
                          mobile_filter=bool(s.get("mobile_filter", False)), role=s.get("role", "news")))
    return out


# ------------------------------------------------------------------ fetching and parsing

def _text(s: str | None) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _when(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s.strip())
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=LOCAL)


def parse_feed(data: bytes) -> list[dict]:
    """RSS 2.0 or Atom -> [{"id", "title", "link", "summary", "published"}]."""
    root = ET.fromstring(data)
    out = []
    if root.tag.endswith("rss") or root.find("channel") is not None:
        for it in root.iter("item"):
            link = (it.findtext("link") or "").strip()
            out.append({"id": (it.findtext("guid") or link).strip(), "title": _text(it.findtext("title")),
                        "link": link, "summary": _text(it.findtext("description")),
                        "published": _when(it.findtext("pubDate"))})
    else:  # Atom
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in root.iterfind("a:entry", ns):
            link_el = e.find("a:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            out.append({"id": (e.findtext("a:id", "", ns) or link).strip(), "title": _text(e.findtext("a:title", "", ns)),
                        "link": link, "summary": _text(e.findtext("a:summary", "", ns) or e.findtext("a:content", "", ns)),
                        "published": _when(e.findtext("a:published", "", ns) or e.findtext("a:updated", "", ns))})
    return [i for i in out if i["title"] and i["id"]]


POGO_CARD = re.compile(r'<a\s+href="(/news/[^"#?]+)"[^>]*class="[^"]*newsCard[^"]*"[^>]*>(.*?)</a>', re.S)


def parse_pokemongo(data: bytes, base: str = "https://pokemongo.com") -> list[dict]:
    """The news page's post cards: link, title (the card's last line of text) and timestamp."""
    page = data.decode("utf-8", errors="replace")
    out = []
    for href, inner in POGO_CARD.findall(page):
        stamp = re.search(r'timestamp="(\d{10,13})"', inner)
        published = None
        if stamp:
            n = int(stamp.group(1))
            published = datetime.fromtimestamp(n / 1000 if n > 10**11 else n, LOCAL)
        lines = [t for t in (_text(x) for x in re.split(r"<[^>]+>", inner)) if t]
        if lines:
            out.append({"id": href, "title": lines[-1], "link": base + href, "summary": "", "published": published})
    return out


# ------------------------------------------------------------------ "already covered" matching

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def key_words(title: str) -> set[str]:
    """Distinctive capitalised words: likely parts of a game's or company's name."""
    words = [re.sub(r"['’]s$", "", w) for w in re.findall(r"[^\W_][\w'’]*", title)]
    return {w.lower() for w in words
            if w[0].isupper() and len(w) >= 3 and w.lower() not in COMMON}


def is_covered(item: dict, headlines: list[str]) -> bool:
    """True if a recent reference-outlet headline seems to be about the same game."""
    if item.get("source") == "gamespress":  # GamesPress titles are usually just the game's name
        name = _norm(item["title"])
        if len(name) >= 4 and any(re.search(rf"\b{re.escape(name)}\b", _norm(h)) for h in headlines):
            return True
    mine = key_words(item["title"])
    return len(mine) >= 2 and any(len(mine & key_words(h)) >= 2 for h in headlines)


# ------------------------------------------------------------------ the connector

class GamesConnector:
    name = "games"

    def __init__(self, sources: list[Source], store: Path, *, keep_days: int = 30, hud_card: bool = True,
                 fetch: Callable[[str], bytes] = fetch_url, now: Callable[[], datetime] | None = None):
        self.sources = {s.name: s for s in sources}
        self.store = store
        self.keep = timedelta(days=keep_days)
        self.hud_card = hud_card
        self._fetch = fetch
        self._now = now or (lambda: datetime.now(LOCAL))
        self._lock = threading.RLock()
        self._failed: dict[str, str] = {}
        self.state = self._load()

    # ---- storage

    def _load(self) -> dict:
        state = {"items": {}, "covered": {}, "polled": {}, "last_digest": None, "mentioned_day": None}
        try:
            state.update(json.loads(self.store.read_text(encoding="utf-8")))
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as e:
            log.warning("games: couldn't read %s (%s); starting afresh", self.store, type(e).__name__)
        return state

    def _save(self) -> None:
        self.store.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.store)

    # ---- polling

    def tick(self) -> None:
        """Poll whichever sources are due, then refresh the HUD card. Called from a background thread."""
        now = self._now()
        due = [s for s in self.sources.values()
               if not self.state["polled"].get(s.name)
               or now - datetime.fromisoformat(self.state["polled"][s.name]) >= timedelta(minutes=s.every_minutes)]
        if not due:
            return
        for s in due:
            self.poll(s, now)
        with self._lock:
            self._prune(now)
            self._save()
        if self.hud_card:
            events.publish("card", **self.card())

    def poll(self, source: Source, now: datetime) -> int:
        try:
            data = self._fetch(source.url)
            found = parse_pokemongo(data) if source.kind == "pokemongo" else parse_feed(data)
        except Exception as e:
            log.warning("games: %s failed: %s", source.name, type(e).__name__)
            self._failed[source.name] = type(e).__name__
            with self._lock:  # try again at the next interval, not every tick
                self.state["polled"][source.name] = now.isoformat()
            return 0
        self._failed.pop(source.name, None)
        added = 0
        with self._lock:
            self.state["polled"][source.name] = now.isoformat()
            if source.role == "covered":
                for f in found:
                    self.state["covered"].setdefault(f["id"], {"title": f["title"], "seen": now.isoformat()})
                return 0
            for f in found:
                key = f"{source.name}:{f['id']}"
                if key in self.state["items"]:
                    continue
                text = f"{f['title']} {f['summary']}"
                if source.mobile_filter and not MOBILE.search(text):
                    continue
                if GUIDES.search(f["title"]):
                    continue
                summary = f["summary"]
                if len(summary) > SUMMARY_CHARS:
                    summary = summary[:SUMMARY_CHARS - 1].rstrip() + "…"
                self.state["items"][key] = {
                    "source": source.name, "title": f["title"], "summary": summary, "link": f["link"],
                    "published": f["published"].isoformat() if f["published"] else None,
                    "seen": now.isoformat()}
                added += 1
        log.info("games: %s: %d new", source.name, added)
        return added

    def _prune(self, now: datetime) -> None:
        cutoff = now - self.keep
        for bucket in ("items", "covered"):
            self.state[bucket] = {k: v for k, v in self.state[bucket].items()
                                  if datetime.fromisoformat(v["seen"]) >= cutoff}

    # ---- selecting

    def _published(self, item: dict) -> datetime:
        return datetime.fromisoformat(item["published"] or item["seen"])

    def new_items(self) -> list[dict]:
        """Items first seen since the last digest (or in the last day, before the first digest)."""
        now = self._now()
        last = datetime.fromisoformat(self.state["last_digest"]) if self.state["last_digest"] else None
        out = []
        for it in self.state["items"].values():
            if self._published(it) < now - NEW_WITHIN:
                continue
            if last and datetime.fromisoformat(it["seen"]) <= last:
                continue
            if not last and self._published(it) < now - timedelta(days=1):
                continue
            out.append(it)
        return out

    def _ranked(self, items: list[dict]) -> list[tuple[dict, bool]]:
        headlines = [c["title"] for c in self.state["covered"].values()]
        marked = [(it, is_covered(it, headlines)) for it in items]
        # Not yet on the reference outlet first, then newest first.
        marked.sort(key=lambda x: self._published(x[0]), reverse=True)
        marked.sort(key=lambda x: x[1])
        return marked

    # ---- the tool

    def tools(self) -> list[Tool]:
        news = [s for s in self.sources.values() if s.role == "news"]
        if not news:
            return []
        spec = ToolSpec(
            name="games_news",
            description=(
                "Mobile games news from press releases, Pokémon GO news and mobile games sites, each marked if "
                "The reference outlet seems to have covered it. scope new (default): since the last digest; "
                "week: the last 7 days."),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["new", "today", "week"]},
                    "source": {"type": "string", "enum": [s.name for s in news]},
                    "search": {"type": "string", "description": "e.g. a game's name"},
                },
            },
        )
        return [Tool(spec=spec, run=self.games_news, notice="Checking games news", focus="games", reads_outside=True)]

    def games_news(self, args: dict) -> str:
        scope = args.get("scope") if args.get("scope") in ("new", "today", "week") else "new"
        source = args.get("source") or None
        search = str(args.get("search") or "").strip().lower()
        now = self._now()
        with self._lock:
            if not self.state["polled"]:
                return untrusted("games news feeds", "The feeds haven't been checked yet; try again in a minute.")
            if scope == "new":
                items = self.new_items()
                since = self.state["last_digest"]
                head = ("since the last digest on " + self._say(datetime.fromisoformat(since), now)
                        if since else "in the last day")
            else:
                days = 1 if scope == "today" else 7
                start = now.replace(hour=0, minute=0, second=0, microsecond=0) if days == 1 else now - timedelta(days=7)
                items = [it for it in self.state["items"].values() if self._published(it) >= start]
                head = "today" if days == 1 else "in the last 7 days"
            if source:
                items = [it for it in items if it["source"] == source]
            if search:
                items = [it for it in items if search in f"{it['title']} {it['summary']}".lower()]
            ranked = self._ranked(items)
            if scope == "new" and not source and not search:
                self.state["last_digest"] = now.isoformat()
                self._save()
        covered = sum(1 for _, c in ranked if c)
        lines = [f"Mobile games news: {len(ranked)} item{'s' if len(ranked) != 1 else ''} {head}"
                 + (f" ({covered} already covered by the reference outlet)" if covered else "") + "."]
        for i, (it, c) in enumerate(ranked[:MAX_LISTED], 1):
            label = self.sources[it["source"]].label if it["source"] in self.sources else it["source"]
            summary = f": {it['summary']}" if it["summary"] else ""
            lines.append(f"{i}. [{label}, {self._say(self._published(it), now)}]"
                         f"{' [covered]' if c else ''} {it['title']}{summary}")
        if len(ranked) > MAX_LISTED:
            lines.append(f"(and {len(ranked) - MAX_LISTED} more, mostly already covered or older)")
        if ranked:
            lines.append("Items from different sites may be the same story: mention each game once.")
        if self._failed:
            lines.append("Couldn't reach at the last check: " + ", ".join(
                self.sources[n].label for n in self._failed if n in self.sources) + ".")
        result = untrusted("games news feeds", "\n".join(lines))
        if scope == "new" and self.hud_card:
            events.publish("card", **self.card())
        return result

    @staticmethod
    def _say(t: datetime, now: datetime) -> str:
        t = t.astimezone(LOCAL)
        if t.date() == now.date():
            return f"today {t:%H:%M}"
        if (now.date() - t.date()).days == 1:
            return f"yesterday {t:%H:%M}"
        return spoken(t, "%a %d %b")

    # ---- the daily mention and the HUD card

    def daily_note(self) -> str | None:
        """Once a day, a line for the model to mention at the start of a reply. None if nothing to say."""
        today = self._now().date().isoformat()
        with self._lock:
            if self.state["mentioned_day"] == today or not self.state["polled"]:
                return None
            ranked = self._ranked(self.new_items())
            if not ranked:
                return None
            self.state["mentioned_day"] = today
            self._save()
        fresh = sum(1 for _, c in ranked if not c)
        return (f"There {'is' if len(ranked) == 1 else 'are'} {len(ranked)} new mobile games news "
                f"item{'s' if len(ranked) != 1 else ''} since the last digest"
                + (f", {fresh} not yet covered by the reference outlet" if fresh != len(ranked) else "")
                + ". Mention this in one short sentence and offer to run through them; don't list them.")

    def card(self) -> dict:
        """Counts and the top few headlines for the HUD's digest card (screen only)."""
        with self._lock:
            ranked = self._ranked(self.new_items())
        return {"id": "games", "new": len(ranked), "uncovered": sum(1 for _, c in ranked if not c),
                "items": [{"title": it["title"], "source": self.sources[it["source"]].label
                           if it["source"] in self.sources else it["source"], "covered": c}
                          for it, c in ranked[:8]],
                "updated": self._now().strftime("%H:%M")}
