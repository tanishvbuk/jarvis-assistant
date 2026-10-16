"""Gmail, read-only, across several accounts.

Each account has a filter in config.local.toml (gmail_query), such as
"in:inbox category:primary" to skip promotions. The model can narrow a search
(a sender, a subject, unread only) but never widen it past that filter.

By default only mail from people is shown: Gmail's automated categories
(Updates, Social, Forums: receipts, alerts, newsletters) are counted, not listed,
unless the model asks for them. Promotions never get past the account filter.

Only the sender, subject, date and Gmail's short snippet are fetched, except
for the embargo finder: it downloads the bodies of emails that mention an
embargo, but only the sentences about the embargo leave this module. Nothing
here can send, delete or label mail.
"""

from __future__ import annotations

import base64
import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

from .. import events
from ..llm import ToolSpec
from ..persona import LOCAL, spoken
from .base import DailyNote, Tool, ToolError, untrusted
from .google import Clients, GoogleAccount, GoogleAuth

log = logging.getLogger(__name__)

SNIPPET_CHARS = 160
EMBARGO_QUERY = "embargo OR embargoed"
EMBARGO_CHARS = 280
CARD_S = 600            # how often the HUD's mail card is refreshed (while a page is on screen)
COUNT_CAP = 50
NOTIFICATIONS = ("category:updates", "category:social", "category:forums")


class GmailClient:
    """The few Gmail API calls Jarvis makes. Tests replace this with a fake."""

    def __init__(self, creds):
        from googleapiclient.discovery import build

        self._svc = build("gmail", "v1", credentials=creds, cache_discovery=False)

    def count(self, query: str, cap: int) -> int:
        """How many messages match, up to `cap` (ids only, nothing about the messages)."""
        r = self._svc.users().messages().list(userId="me", q=query, maxResults=cap).execute()
        return len(r.get("messages", []))

    def search(self, query: str, limit: int) -> list[dict]:
        """Newest first: [{"from", "subject", "date_ms", "snippet", "unread"}]."""
        msgs = self._svc.users().messages()
        ids = [m["id"] for m in msgs.list(userId="me", q=query, maxResults=limit).execute().get("messages", [])]
        if not ids:
            return []
        found: dict[str, dict] = {}

        def got(request_id, response, exception):
            if exception is None:
                found[request_id] = response
            else:
                log.warning("Gmail: one message couldn't be read: %s", exception)

        batch = self._svc.new_batch_http_request(callback=got)  # one HTTP round trip for all of them
        for i in ids:
            batch.add(msgs.get(userId="me", id=i, format="metadata", metadataHeaders=["From", "Subject"]),
                      request_id=i)
        batch.execute()
        out = []
        for i in ids:
            m = found.get(i)
            if not m:
                continue
            headers = {h["name"].lower(): h["value"] for h in m.get("payload", {}).get("headers", [])}
            out.append({"from": headers.get("from", ""), "subject": headers.get("subject", ""),
                        "date_ms": int(m.get("internalDate", 0)), "snippet": m.get("snippet", ""),
                        "unread": "UNREAD" in m.get("labelIds", [])})
        return out

    def search_bodies(self, query: str, limit: int) -> list[dict]:
        """Like search(), but with each message's plain text instead of the snippet."""
        msgs = self._svc.users().messages()
        ids = [m["id"] for m in msgs.list(userId="me", q=query, maxResults=limit).execute().get("messages", [])]
        out = []
        for i in ids:
            m = msgs.get(userId="me", id=i, format="full").execute()
            headers = {h["name"].lower(): h["value"] for h in m.get("payload", {}).get("headers", [])}
            out.append({"from": headers.get("from", ""), "subject": headers.get("subject", ""),
                        "date_ms": int(m.get("internalDate", 0)), "text": body_text(m.get("payload", {}))})
        return out


def body_text(payload: dict) -> str:
    """The message's text: text/plain if there is one, else text/html with the tags removed."""
    plain, rich = [], []

    def walk(part: dict) -> None:
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if data and mime in ("text/plain", "text/html"):
            text = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")
            (plain if mime == "text/plain" else rich).append(text)
        for p in part.get("parts") or []:
            walk(p)

    walk(payload)
    if plain:
        return "\n".join(plain)
    text = "\n".join(rich)
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n", text)
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


def embargo_lines(text: str, limit: int = 2) -> list[str]:
    """The sentences (or lines) that mention an embargo, trimmed."""
    out: list[str] = []
    for m in re.finditer(r"(?i)embargo", text):
        start = max(text.rfind("\n", 0, m.start()), text.rfind(". ", 0, m.start()) + 1, m.start() - 200)
        ends = [e for e in (text.find("\n", m.end()), text.find(". ", m.end())) if e != -1]
        end = min(ends + [m.end() + 200])
        line = _clean(text[start:end + 1], EMBARGO_CHARS)
        if line and line not in out:
            out.append(line)
        if len(out) >= limit:
            break
    return out


def _clean(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", html.unescape(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _when(ms: int, now: datetime) -> str:
    t = datetime.fromtimestamp(ms / 1000, LOCAL)
    if t.date() == now.date():
        return f"today {t:%H:%M}"
    if (now.date() - t.date()).days == 1:
        return f"yesterday {t:%H:%M}"
    return spoken(t, "%a %d %b %H:%M")


class GmailConnector:
    name = "gmail"

    def __init__(self, accounts: list[GoogleAccount], auth: GoogleAuth, *, days: int = 2, max_results: int = 10,
                 cache_seconds: float = 120, hud_card: bool = True, client_factory: Callable | None = None, state_file: Path | None = None,
                 clock: Callable[[], float] = time.monotonic, now: Callable[[], datetime] | None = None):
        self.accounts = {a.name: a for a in accounts if "gmail" in a.services}
        self.auth = auth
        self.days = days
        self.max_results = max_results
        self.cache_seconds = cache_seconds
        self.hud_card = hud_card
        self._card_at = float("-inf")
        self._clients = Clients(auth, client_factory or GmailClient)
        self._cache: dict[tuple, tuple[float, list[dict]]] = {}
        self._clock = clock
        self._now = now or (lambda: datetime.now(LOCAL))
        self._embargo_note = DailyNote(state_file, "embargo_day", self._embargo_count,
                                       lambda: self._now().date(), clock)

    def tools(self) -> list[Tool]:
        if not self.accounts:
            return []
        described = "; ".join(f"{a.name}" + (f" ({a.about})" if a.about else "") for a in self.accounts.values())
        spec = ToolSpec(
            name="check_email",
            description=(
                "His recent email, read-only: sender, subject, time and a preview, newest first. Lists mail from "
                "people; automated notifications are only counted unless include_notifications (e.g. for an "
                f"order, delivery or payment). Accounts: {described}. Omit account for all."),
            parameters={
                "type": "object",
                "properties": {
                    "account": {"type": "string", "enum": list(self.accounts)},
                    "search": {"type": "string", "description": "Gmail terms to narrow, e.g. from:etsy"},
                    "unread_only": {"type": "boolean"},
                    "include_notifications": {"type": "boolean"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 30, "description": f"default {self.days}"},
                },
            },
        )
        embargo = ToolSpec(
            name="find_embargoes",
            description=(
                "Emails mentioning a press embargo (he's a games journalist): sender, subject, arrival and only "
                "the embargo sentences. To diary one, use calendar_change, one event per call."),
            parameters={"type": "object", "properties": {
                "days": {"type": "integer", "minimum": 1, "maximum": 30, "description": "default 14"},
                "account": {"type": "string", "enum": list(self.accounts)}}})
        return [Tool(spec=spec, run=self.check_email, notice="Checking email", focus="mail", reads_outside=True),
                Tool(spec=embargo, run=self.find_embargoes, notice="Looking for embargoes", reads_outside=True)]

    # ------------------------------------------------------------------ the tool

    def check_email(self, args: dict) -> str:
        name = args.get("account") or None
        if name and name not in self.accounts:
            raise ToolError(f"there's no email account called {name!r}; the accounts are {', '.join(self.accounts)}.")
        try:
            days = max(1, min(30, int(args.get("days") or self.days)))
        except (TypeError, ValueError):
            days = self.days
        search = re.sub(r"[(){}]", " ", str(args.get("search") or "")).strip()
        unread_only = bool(args.get("unread_only"))
        notes = bool(args.get("include_notifications"))
        targets = [self.accounts[name]] if name else list(self.accounts.values())

        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            sections = list(pool.map(lambda a: self._section(a, days, search, unread_only, notes), targets))
        return untrusted("Gmail", "\n\n".join(sections))

    def _section(self, account: GoogleAccount, days: int, search: str, unread_only: bool,
                 notifications: bool = False) -> str:
        label = f"Account {account.name}" + (f" ({account.about})" if account.about else "")
        query = self.build_query(account, days, search, unread_only, notifications)
        hidden = 0
        try:
            messages = self._fetch(account, "search", query)
            if not notifications:
                only = self.build_query(account, days, search, unread_only) + " {" + " ".join(NOTIFICATIONS) + "}"
                hidden = self._fetch(account, "count", only)
        except ToolError as e:
            return f"{label}: unavailable: {e}"
        except Exception as e:
            log.warning("Gmail search failed for %s: %s", account.name, type(e).__name__)
            return f"{label}: unavailable: Gmail didn't respond properly. Details are in Jarvis's log."
        kind = "messages" if notifications else "messages from people"
        what = ("unread " if unread_only else "") + f"{kind} from the last {days} day{'s' if days > 1 else ''}"
        skipped = (f"\n(Also {'50+' if hidden >= 50 else hidden} automated notification{'s' if hidden != 1 else ''} "
                   "not listed.)") if hidden else ""
        if not messages:
            return f"{label}: no {what}" + (" matching the search." if search else ".") + skipped
        now = self._now()
        lines = [f"{label}: {len(messages)} {what}" + (" (showing the newest)" if len(messages) >= self.max_results
                                                      else "") + ":"]
        for i, m in enumerate(messages, 1):
            flag = " [unread]" if m["unread"] else ""
            lines.append(f"{i}. {_when(m['date_ms'], now)}{flag} | From: {_clean(m['from'], 80)} | "
                         f"Subject: {_clean(m['subject'], 120) or '(none)'} | {_clean(m['snippet'], SNIPPET_CHARS)}")
        return "\n".join(lines) + skipped

    def find_embargoes(self, args: dict) -> str:
        try:
            days = max(1, min(30, int(args.get("days") or 14)))
        except (TypeError, ValueError):
            days = 14
        name = args.get("account") or None
        if name and name not in self.accounts:
            raise ToolError(f"there's no email account called {name!r}.")
        targets = [self.accounts[name]] if name else list(self.accounts.values())
        now = self._now()
        lines: list[str] = []
        for a in targets:
            query = self.build_query(a, days, EMBARGO_QUERY)
            try:
                found = self._fetch(a, "bodies", query)
            except ToolError as e:
                lines.append(f"Account {a.name}: unavailable: {e}")
                continue
            except Exception as e:
                log.warning("embargo search failed for %s: %s", a.name, type(e).__name__)
                lines.append(f"Account {a.name}: unavailable: Gmail didn't respond properly.")
                continue
            for m in found:
                said = embargo_lines(m["text"]) or embargo_lines(m["subject"])
                if said:
                    lines.append(f"- [{a.name}, received {_when(m['date_ms'], now)}] From: {_clean(m['from'], 80)} | "
                                 f"Subject: {_clean(m['subject'], 120)} | Embargo: " + " / ".join(said))
        head = (f"Today is {spoken(now, '%A %d %B %Y')}. Emails mentioning an embargo in the last {days} days: "
                f"{sum(l.startswith('- ') for l in lines)}. Only offer embargoes that haven't lifted yet (compare each "
                "with today's date); if every one has already lifted, say so.")
        return untrusted("Gmail (embargo sentences only)", "\n".join([head] + lines))

    def daily_note(self) -> str | None:
        """Once a day: how many emails from the last day mention an embargo (counted in the background)."""
        return self._embargo_note.take()

    def tick(self) -> None:
        self._embargo_note.prepare()
        if self.hud_card and self.accounts and events.watched() and self._clock() - self._card_at >= CARD_S:
            self._card_at = self._clock()
            events.publish("card", **self.card())

    def card(self) -> dict:
        """Counts for the HUD's mail card (screen only): mail from people and automated notifications over the last
        few days, per account. Only counts are fetched, nothing about any message."""
        rows, problems = [], []
        for a in self.accounts.values():
            try:
                people = self._fetch(a, "count", self.build_query(a, self.days, notifications=False))
                auto = self._fetch(a, "count", self.build_query(a, self.days) + " {" + " ".join(NOTIFICATIONS) + "}")
            except ToolError:
                problems.append(a.name)
                continue
            except Exception as e:
                log.warning("mail card failed for %s: %s", a.name, type(e).__name__)
                problems.append(a.name)
                continue
            rows.append({"name": a.name, "people": people, "auto": auto})
        total = lambda k: sum(r[k] for r in rows)  # noqa: E731
        return {"id": "mail", "days": self.days, "people": total("people"), "auto": total("auto"),
                "capped": any(r[k] >= COUNT_CAP for r in rows for k in ("people", "auto")),
                "accounts": rows, "problems": problems}

    def _embargo_count(self) -> str | None:
        count = 0
        for a in self.accounts.values():
            try:
                count += self._fetch(a, "count", self.build_query(a, 1, EMBARGO_QUERY))
            except Exception as e:
                log.warning("embargo count failed for %s: %s", a.name, type(e).__name__)
        if not count:
            return None
        return (f"{count} email{'s' if count != 1 else ''} from the last day mention a press embargo. Mention it in "
                "one short sentence and offer to look them up and add them to the calendar.")

    @staticmethod
    def build_query(account: GoogleAccount, days: int, search: str = "", unread_only: bool = False,
                    notifications: bool = True) -> str:
        """The account's own filter always applies; the model's terms can only narrow it."""
        parts = [f"({account.gmail_query})" if account.gmail_query else "", f"newer_than:{days}d"]
        if not notifications:
            parts += [f"-{c}" for c in NOTIFICATIONS]
        if unread_only:
            parts.append("is:unread")
        if search:
            parts.append(f"({search})")
        return " ".join(p for p in parts if p)

    def _fetch(self, account: GoogleAccount, what: str, query: str):
        """client.search() or client.count(), cached per account and query."""
        key = (account.name, what, query)
        hit = self._cache.get(key)
        if hit and self._clock() - hit[0] < self.cache_seconds:
            return hit[1]
        client = self._clients.get(account)
        if what == "search":
            result = client.search(query, self.max_results)
        elif what == "bodies":
            result = client.search_bodies(query, 15)
        else:
            result = client.count(query, 50)
        self._cache[key] = (self._clock(), result)
        return result
