"""Currency conversion from the European Central Bank's daily reference rates, through Frankfurter
(https://frankfurter.dev: free, no key, no account). One tool, `convert_currency`.

About 30 major currencies, no crypto, and the rates are fixed once each working day (around 16:00 CET), so this is
the official daily rate, not the live market rate a card or bureau will give. The answer says which day's rate it is.
Only currency codes and a date leave the PC.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.error
from datetime import date, datetime
from typing import Callable

from ..llm import ToolSpec
from ..persona import LOCAL, spoken
from .base import Tool, ToolError
from .web import get_json

log = logging.getLogger(__name__)

BASE = "https://api.frankfurter.dev/v1"
CACHE_S = 3600


def money(x: float) -> str:
    """1234.5 -> "1,234.50"; very small amounts keep their significant digits."""
    return f"{x:,.2f}" if abs(x) >= 0.1 or x == 0 else f"{x:.4g}"


class CurrencyConnector:
    name = "currency"

    def __init__(self, get: Callable = get_json, now: Callable[[], datetime] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._get = get
        self._now = now or (lambda: datetime.now(LOCAL))
        self._clock = clock
        self._cache: dict[tuple, tuple[float, object]] = {}

    def tools(self) -> list[Tool]:
        spec = ToolSpec(
            name="convert_currency",
            description=("Convert an amount between currencies at the ECB's daily rate (about 30 major currencies, no "
                         "crypto). Use ISO codes: GBP, USD, EUR, INR. Say the rate is the day's official rate."),
            parameters={"type": "object", "properties": {
                "amount": {"type": "number"},
                "from": {"type": "string", "description": "e.g. USD"},
                "to": {"type": "string", "description": "e.g. GBP"},
                "date": {"type": "string", "description": "YYYY-MM-DD for a past rate; default latest"}},
                "required": ["amount", "from", "to"]})
        return [Tool(spec=spec, run=self.convert_currency, notice="Checking the exchange rate")]

    def _fetch(self, path: str, params: dict | None = None):
        key = (path, tuple(sorted((params or {}).items())))
        hit = self._cache.get(key)
        if hit and self._clock() - hit[0] < CACHE_S:
            return hit[1]
        try:
            data = self._get(BASE + path, params)
        except urllib.error.HTTPError as e:
            if e.code in (404, 422):
                raise ToolError("the rate service has no rate for that (a currency it doesn't cover, or a date "
                                "before 1999).") from e
            log.warning("currency lookup failed: HTTP %s", e.code)
            raise ToolError("the exchange-rate service answered with an error.") from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.warning("currency lookup failed: %s", type(e).__name__)
            raise ToolError("the exchange-rate service isn't answering (is the internet up?).") from e
        self._cache[key] = (self._clock(), data)
        return data

    def convert_currency(self, args: dict) -> str:
        try:
            amount = float(args.get("amount"))
        except (TypeError, ValueError):
            raise ToolError("how much?") from None
        src, dst = (str(args.get(k) or "").strip().upper() for k in ("from", "to"))
        if not (re.fullmatch(r"[A-Z]{3}", src) and re.fullmatch(r"[A-Z]{3}", dst)):
            raise ToolError("use three-letter currency codes such as GBP, USD, EUR.")
        when = "latest"
        if args.get("date"):
            try:
                d = date.fromisoformat(str(args["date"])[:10])
            except ValueError:
                raise ToolError(f"{args['date']!r} isn't a date; use YYYY-MM-DD.") from None
            if d > self._now().date():
                raise ToolError("that date is in the future.")
            when = d.isoformat()
        if src == dst:
            return f"{money(amount)} {src} is {money(amount)} {dst}."
        known = self._fetch("/currencies")
        missing = [c for c in (src, dst) if c not in known]
        if missing:
            raise ToolError(f"{', '.join(missing)} isn't covered by the European Central Bank's rates (about 30 major "
                            "currencies, no crypto). A web search may find a rate.")
        data = self._fetch("/" + when, {"base": src, "symbols": dst})
        try:
            rate, day = float(data["rates"][dst]), date.fromisoformat(data["date"])
        except (KeyError, TypeError, ValueError):
            raise ToolError("the rate service's answer wasn't usable.") from None
        return (f"{money(amount)} {src} = {money(amount * rate)} {dst}, at {rate:.5g} {dst} per {src}: the European "
                f"Central Bank's reference rate for {spoken(day, '%A %d %B %Y')}. These are set once on working days, "
                "so a card or bureau rate will differ slightly.")


def currency_from_config(cfg) -> CurrencyConnector | None:
    return CurrencyConnector() if cfg.currency.get("enabled", False) else None  # on via [currency] in config.toml
