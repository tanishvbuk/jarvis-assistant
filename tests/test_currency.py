from __future__ import annotations

import io
import urllib.error
from datetime import datetime

import pytest

from jarvis.connectors.base import ToolError
from jarvis.connectors.currency import CurrencyConnector, money
from jarvis.persona import LOCAL

NOW = datetime(2026, 10, 15, 9, 0, tzinfo=LOCAL)
CURRENCIES = {"GBP": "British Pound", "USD": "US Dollar", "EUR": "Euro", "INR": "Indian Rupee"}


class Fake:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    def __call__(self, url, params=None):
        self.calls.append((url.removeprefix("https://api.frankfurter.dev/v1"), params))
        if self.error:
            raise self.error
        if url.endswith("/currencies"):
            return CURRENCIES
        return {"amount": 1.0, "base": params["base"], "date": "2026-10-14", "rates": {params["symbols"]: 0.75377}}


def make(**kw):
    fake = Fake(**kw)
    c = CurrencyConnector(get=fake, now=lambda: NOW, clock=lambda: 0.0)
    c.fake = fake
    return c


def test_converts_at_the_ecb_rate_and_says_whose_rate_and_which_day():
    c = make()
    out = c.convert_currency({"amount": 50, "from": "usd", "to": "GBP"})
    assert out.startswith("50.00 USD = 37.69 GBP, at 0.75377 GBP per USD")
    assert "European Central Bank's reference rate for Wednesday 14 October 2026" in out
    assert ("/latest", {"base": "USD", "symbols": "GBP"}) in c.fake.calls


def test_past_dates_and_the_cache():
    c = make()
    c.convert_currency({"amount": 1, "from": "GBP", "to": "INR", "date": "2026-10-01"})
    c.convert_currency({"amount": 9, "from": "GBP", "to": "INR", "date": "2026-10-01"})
    rate_calls = [x for x in c.fake.calls if x[0] != "/currencies"]
    assert rate_calls == [("/2026-10-01", {"base": "GBP", "symbols": "INR"})]   # the second was cached


@pytest.mark.parametrize("args,words", [
    ({"amount": "lots", "from": "USD", "to": "GBP"}, "how much"),
    ({"amount": 5, "from": "dollars", "to": "GBP"}, "three-letter"),
    ({"amount": 5, "from": "USD", "to": "GBP", "date": "2027-01-01"}, "future"),
    ({"amount": 5, "from": "USD", "to": "GBP", "date": "yesterday"}, "isn't a date"),
    ({"amount": 5, "from": "BTC", "to": "GBP"}, "BTC isn't covered"),
])
def test_bad_requests_get_a_plain_message(args, words):
    with pytest.raises(ToolError, match=words):
        make().convert_currency(args)


def test_same_currency_needs_no_lookup():
    c = make()
    assert c.convert_currency({"amount": 5, "from": "GBP", "to": "gbp"}) == "5.00 GBP is 5.00 GBP."
    assert c.fake.calls == []


def test_service_down():
    with pytest.raises(ToolError, match="isn't answering"):
        make(error=urllib.error.URLError("down")).convert_currency({"amount": 5, "from": "USD", "to": "GBP"})
    err = urllib.error.HTTPError("u", 500, "x", {}, io.BytesIO(b""))
    with pytest.raises(ToolError, match="error"):
        make(error=err).convert_currency({"amount": 5, "from": "USD", "to": "GBP"})


def test_money_format():
    assert money(1234.5) == "1,234.50" and money(0.0004567) == "0.0004567" and money(0) == "0.00"
