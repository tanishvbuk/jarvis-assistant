"""Games digest with invented feeds (no network)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from jarvis import events
from jarvis.connectors.games import (GamesConnector, Source, is_covered, key_words, parse_feed, parse_pokemongo,
                                     sources_from_config)
from jarvis.persona import LOCAL

T0 = datetime(2026, 10, 10, 9, 0, tzinfo=LOCAL)


def rss(*items):
    body = "".join(f"<item><title>{t}</title><link>https://x.example/{i}</link><guid>{t}</guid>"
                   f"<description>{d}</description><pubDate>{p}</pubDate></item>" for i, (t, d, p) in enumerate(items))
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>{body}</channel></rss>'.encode()


FRI = "Fri, 09 Oct 2026 15:00:00 +0100"
SAT = "Sat, 10 Oct 2026 08:30:00 +0100"
OLD = "Mon, 01 Jun 2026 10:00:00 +0100"

FEEDS = {
    "https://gp": rss(("Starfall Tactics", "Out now on iOS and Android. &lt;b&gt;Free&lt;/b&gt;", SAT),
                      ("Big Console Shooter", "Coming to PS5 and Xbox", SAT),
                      ("Moonlight Bakery", "The cosy sim lands on mobile today", FRI),
                      ("Ancient Mobile Thing", "An iPhone game", OLD)),
    "https://zebo": rss(("Hollow Lantern Saga gets a spooky Halloween update on Android", "", SAT)),
    "https://pg": rss(("Moonlight Bakery is out now and it's lovely", "", FRI)),
    "https://pogo": b"""<div><a href="/news/harvest-2026" class="_newsCard_x1"><div><pg-date-format timestamp="1791612000000"
        class="d"></pg-date-format></div><div class="t">Harvest Festival &amp; Shiny Applin</div></a>
        <a href="/other" class="nav">Not a card</a></div>""",
}

SOURCES = [Source("gamespress", "GamesPress", "rss", "https://gp", 60, mobile_filter=True),
           Source("gamezebo", "Gamezebo", "rss", "https://zebo", 180),
           Source("pokemongo", "Pokémon GO news", "pokemongo", "https://pogo", 360),
           Source("reference", "Reference outlet", "rss", "https://pg", 60, role="covered")]


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


@pytest.fixture
def games(tmp_path):
    fetched = []

    def fetch(url):
        fetched.append(url)
        if url == "https://down":
            raise OSError("no route")
        return FEEDS[url]

    clock = Clock()
    g = GamesConnector(SOURCES, tmp_path / "games.json", fetch=fetch, now=clock)
    g.fetched, g.clock = fetched, clock
    return g


def test_parsers():
    items = parse_feed(FEEDS["https://gp"])
    assert items[0]["title"] == "Starfall Tactics" and items[0]["summary"] == "Out now on iOS and Android. Free"
    assert items[0]["published"] == datetime(2026, 10, 10, 8, 30, tzinfo=LOCAL)
    atom = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>a1</id><title>Hi</title>
               <link href="https://a/1"/><updated>2026-10-10T08:00:00Z</updated><summary>S</summary></entry></feed>"""
    assert parse_feed(atom)[0]["link"] == "https://a/1"
    (post,) = parse_pokemongo(FEEDS["https://pogo"])
    assert post["title"] == "Harvest Festival & Shiny Applin" and post["link"] == "https://pokemongo.com/news/harvest-2026"
    assert post["published"].year == 2026


def test_polling_filters_mobile_and_never_reports_the_reference_outlet(games):
    games.tick()
    titles = {i["title"] for i in games.state["items"].values()}
    assert "Starfall Tactics" in titles and "Moonlight Bakery" in titles
    assert "Big Console Shooter" not in titles          # not mobile
    assert "Moonlight Bakery is out now and it's lovely" not in titles  # reference outlet: reference only
    assert len(games.state["covered"]) == 1
    assert len(games.fetched) == 4


def test_sources_polled_at_their_own_interval_and_no_duplicates(games):
    games.tick()
    games.clock.t += timedelta(minutes=61)
    games.tick()
    assert games.fetched[4:] == ["https://gp", "https://pg"]  # the hourly ones only
    assert len(games.state["items"]) == 5


def test_digest_lists_new_items_uncovered_first_then_marks_them_read(games):
    games.tick()
    out = games.games_news({})
    assert "<untrusted_data" in out and "Mobile games news: 4 items in the last day (1 already covered" in out
    lines = [l for l in out.splitlines() if l[:2] in ("1.", "2.", "3.", "4.")]
    assert "[covered] Moonlight Bakery" in lines[-1]
    assert "Ancient Mobile Thing" not in out                       # published months ago
    assert "Starfall Tactics: Out now on iOS and Android. Free" in out
    again = games.games_news({})
    assert "0 items since the last digest on today 09:00" in again
    games.clock.t += timedelta(hours=3, minutes=1)  # Gamezebo is checked every 3 hours
    games._fetch = lambda url: (rss(("Pixel Golf Kings tees off on iOS", "", "Sat, 10 Oct 2026 09:30:00 +0100"))
                                if url == "https://zebo" else FEEDS[url])
    games.tick()
    assert "Pixel Golf Kings" in games.games_news({})


def test_filters_by_source_search_and_scope_without_marking(games):
    games.tick()
    assert "1 item today" in games.games_news({"scope": "today", "source": "pokemongo"})
    assert "Starfall" in games.games_news({"scope": "week", "search": "starfall"})
    assert games.state["last_digest"] is None


def test_state_survives_a_restart(games, tmp_path):
    games.tick()
    games.games_news({})
    again = GamesConnector(SOURCES, tmp_path / "games.json", fetch=lambda u: FEEDS[u], now=games.clock)
    assert again.state["last_digest"] == T0.isoformat() and len(again.state["items"]) == 5  # incl. the old one, never reported


def test_failed_source_is_reported_and_retried_at_its_interval(tmp_path):
    clock = Clock()
    calls = []

    def fetch(url):
        calls.append(url)
        raise OSError("down")

    g = GamesConnector([Source("gamespress", "GamesPress", "rss", "https://down", 60)], tmp_path / "g.json",
                       fetch=fetch, now=clock)
    g.tick()
    g.tick()
    assert calls == ["https://down"]
    assert "Couldn't reach at the last check: GamesPress." in g.games_news({})


def test_daily_note_once_a_day_and_only_with_news(games):
    assert games.daily_note() is None            # nothing polled yet
    games.tick()
    note = games.daily_note()
    assert note.startswith("There are 4 new mobile games news items since the last digest, 3 not yet covered")
    assert games.daily_note() is None            # once a day
    games.clock.t += timedelta(days=1)
    games.games_news({})
    assert games.daily_note() is None            # nothing new


def test_card(games):
    seen = []
    off = events.subscribe(seen.append)
    try:
        games.tick()
    finally:
        off()
    (card,) = [e for e in seen if e["t"] == "card"]
    assert card["id"] == "games" and card["new"] == 4 and card["uncovered"] == 3
    assert card["items"][0]["covered"] is False and len(card["items"]) == 4


def test_covered_matching():
    pg = ["Moonlight Bakery is out now and it's lovely", "Hollow Lantern Saga review - a spooky treat"]
    assert is_covered({"source": "gamespress", "title": "Moonlight Bakery"}, pg)
    assert not is_covered({"source": "gamespress", "title": "Moonlight"}, ["Moonlighting returns"])
    assert is_covered({"source": "gamezebo", "title": "Hollow Lantern Saga gets a Halloween update"}, pg)
    assert not is_covered({"source": "gamezebo", "title": "Hollow Knight comes to iOS"}, pg)
    assert key_words("Pokémon GO's Harvest Festival is NEW on Android") == {"pokémon", "harvest", "festival"}
    assert not is_covered({"source": "gamezebo", "title": "Phoodle Answer, October 9th 2026"},
                          ["Waffle Answer, October 9th 2026"])  # dates and numbers don't count


def test_guides_are_not_news(games):
    games._fetch = lambda url: rss(("Anime Zero Tier List [Best Characters]", "", SAT),
                                   ("Bloons Blitz codes October 2026", "", SAT),
                                   ("How to get the Soryu Scroll in Slayers 2", "", SAT),
                                   ("Ubisoft is closing Rainbow Six Mobile", "", SAT)) if url == "https://zebo" else FEEDS[url]
    games.tick()
    assert [i["title"] for i in games.state["items"].values() if i["source"] == "gamezebo"] == [
        "Ubisoft is closing Rainbow Six Mobile"]


def test_sources_from_config():
    srcs = sources_from_config({"sources": {"a": {"url": "https://a", "mobile_filter": True},
                                            "b": {"url": "https://b", "enabled": False}, "c": {}}})
    assert [(s.name, s.kind, s.mobile_filter, s.every_minutes) for s in srcs] == [("a", "rss", True, 180)]
