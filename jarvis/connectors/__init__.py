"""Read-only connectors to the user's data. Each one offers the model a few tools."""

from __future__ import annotations

from ..config import Config
from .base import Connector, Tool, ToolError, untrusted
from .google import GoogleAccount, GoogleAuth, accounts_from_config

__all__ = ["build_connectors", "connector_status", "google_auth", "Connector", "Tool", "ToolError", "untrusted"]


def google_auth(cfg: Config) -> GoogleAuth:
    return GoogleAuth(cfg.root / cfg.google.get("tokens_dir", "tokens"))


def build_connectors(cfg: Config) -> list[Connector]:
    """The connectors switched on in config.toml that have something configured."""
    out: list[Connector] = []
    if cfg.google.get("enabled", True):
        accounts = accounts_from_config(cfg.google)
        if cfg.gmail.get("enabled", True) and any("gmail" in a.services for a in accounts):
            from .gmail import GmailConnector

            gmail = GmailConnector(accounts, google_auth(cfg), days=int(cfg.gmail.get("days", 2)),
                                   max_results=int(cfg.gmail.get("max_results", 10)),
                                   cache_seconds=float(cfg.gmail.get("cache_seconds", 120)),
                                   hud_card=bool(cfg.gmail.get("hud_card", True)),
                                   state_file=cfg.root / "data" / "gmail_state.json")
            out.append(gmail)
        if cfg.calendar.get("enabled", True) and any("calendar" in a.services for a in accounts):
            from .calendar import CalendarConnector

            out.append(CalendarConnector(accounts, google_auth(cfg), hud_card=bool(cfg.calendar.get("hud_card", True)),
                                         card_minutes=float(cfg.calendar.get("card_minutes", 10))))
        if cfg.meal_plan.get("enabled", True):
            from .meal_plan import MealPlanConnector, plans_from_config

            plans = plans_from_config(cfg.meal_plan)
            if plans:
                out.append(MealPlanConnector(plans, accounts, google_auth(cfg),
                                             cache_seconds=float(cfg.meal_plan.get("cache_seconds", 600)),
                                             hud_card=bool(cfg.meal_plan.get("hud_card", True))))
    if cfg.google.get("enabled", True) and cfg.tasks.get("enabled", True):
        owner = next((a for a in accounts_from_config(cfg.google) if "tasks" in a.services), None)
        if owner:
            from .tasks import TasksConnector

            out.append(TasksConnector(owner, google_auth(cfg), state_file=cfg.root / "data" / "tasks_state.json",
                                      hud_card=bool(cfg.tasks.get("hud_card", True))))
    from .opengym import opengym_from_config

    gym = opengym_from_config(cfg, meal_plan=next((c for c in out if c.name == "meal_plan"), None))
    if gym:
        out.append(gym)
    from .currency import currency_from_config

    money = currency_from_config(cfg)
    if money:
        out.append(money)
    from .websearch import websearch_from_config

    web = websearch_from_config(cfg)
    if web:
        out.append(web)
    return out


def connector_status(cfg: Config, connectors: list[Connector]) -> str:
    """One line for the console, e.g. 'gmail (personal, shop; not signed in: work)'. No network calls."""
    if not connectors:
        return "none"
    auth = google_auth(cfg)
    parts = []
    for c in connectors:
        if hasattr(c, "plans"):  # meal plans: name the plans, not the accounts
            parts.append(f"{c.name} ({', '.join(c.plans)})")
            continue
        if c.name == "tasks":
            parts.append(f"to-do list ({c.account.name}" + ("" if auth.has_scope(c.account, "tasks")
                                                          else "; sign in again to use it") + ")")
            continue
        if c.name in ("opengym", "websearch"):
            parts.append(c.status())
            continue
        if c.name == "currency":
            parts.append("currency (ECB daily rates)")
            continue
        if c.name == "calendar":
            needs = [n for n in c.writable if not auth.has_scope(c.accounts[n], "calendar_write")]
            parts.append(f"calendar ({', '.join(c.accounts)}; can add to: {', '.join(c.writable) or 'none'}"
                         + (f"; sign in again to allow adding: {', '.join(needs)}" if needs else "") + ")")
            continue
        accounts: list[GoogleAccount] = list(getattr(c, "accounts", {}).values())
        ready = [a.name for a in accounts if auth.signed_in(a)]
        waiting = [a.name for a in accounts if not auth.signed_in(a)]
        detail = ", ".join(ready) or "no accounts signed in"
        if waiting and ready:
            detail += f"; not signed in: {', '.join(waiting)}"
        parts.append(f"{c.name} ({detail})")
    return ", ".join(parts)
