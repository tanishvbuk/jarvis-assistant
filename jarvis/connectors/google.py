"""Google sign-in for several accounts, read-only.

All accounts sign in through one "Desktop app" OAuth client whose secret file is
tokens/client_secret.json. Each account then gets its own token file,
tokens/google-<name>.json. Both stay out of Git (tokens/ is gitignored).

Accounts are listed in config.local.toml, for example:

    [google.accounts.personal]
    email = "someone@gmail.com"
    about = "personal mail"
    services = ["gmail", "calendar", "sheets"]
    gmail_query = "in:inbox category:primary"
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .base import ToolError

log = logging.getLogger(__name__)

# Read-only, except calendar_write (only where an account lists it). Adding a service to an
# account means signing that account in again.
SCOPES = {
    "gmail": "https://www.googleapis.com/auth/gmail.readonly",
    "calendar": "https://www.googleapis.com/auth/calendar.readonly",
    "calendar_write": "https://www.googleapis.com/auth/calendar.events",  # add/move/delete events, after a spoken yes
    "sheets": "https://www.googleapis.com/auth/spreadsheets.readonly",
    "tasks": "https://www.googleapis.com/auth/tasks",  # the to-do list: add, complete, move; delete after a yes
}


class NeedsLogin(ToolError):
    pass


@dataclass
class GoogleAccount:
    name: str
    email: str
    about: str = ""
    services: list[str] = field(default_factory=lambda: ["gmail"])
    gmail_query: str = "in:inbox category:primary"

    @property
    def scopes(self) -> list[str]:
        return [SCOPES[s] for s in self.services if s in SCOPES]


def accounts_from_config(google: dict) -> list[GoogleAccount]:
    out = []
    for name, a in (google.get("accounts") or {}).items():
        unknown = [s for s in a.get("services", ["gmail"]) if s not in SCOPES]
        if unknown:
            log.warning("Google account %s: unknown services %s ignored", name, unknown)
        out.append(GoogleAccount(name=name, email=a.get("email", ""), about=a.get("about", ""),
                                 services=list(a.get("services", ["gmail"])),
                                 gmail_query=a.get("gmail_query", "in:inbox category:primary")))
    return out


class Clients:
    """One API client per account for a connector, built on first use.

    Google's client library isn't thread-safe, and the turn, the background jobs and
    HUD refreshes can all reach the same client, so each one is used by one thread at
    a time. A sign-in that expires while Jarvis runs becomes NeedsLogin, the same for
    every connector, and the client is rebuilt after the next sign-in.
    """

    def __init__(self, auth, factory: Callable):
        self.auth = auth
        self.factory = factory
        self._made: dict[str, _OneAtATime] = {}
        self._lock = threading.Lock()

    def get(self, account: GoogleAccount) -> _OneAtATime:
        with self._lock:
            client = self._made.get(account.name)
        if client is None:
            made = _OneAtATime(self.factory(self.auth.credentials(account)), account.name,
                               lambda: self.drop(account))
            with self._lock:
                client = self._made.setdefault(account.name, made)
        return client

    def drop(self, account: GoogleAccount) -> None:
        with self._lock:
            self._made.pop(account.name, None)


class _OneAtATime:
    """Wraps a client so its calls never overlap."""

    def __init__(self, client, account: str, on_expired: Callable[[], None]):
        self._client, self._account, self._on_expired = client, account, on_expired
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            from google.auth.exceptions import RefreshError

            with self._lock:
                try:
                    return attr(*args, **kwargs)
                except RefreshError as e:  # revoked or expired while Jarvis was running
                    log.warning("Google refresh failed for %s: %s", self._account, e)
                    self._on_expired()
                    raise NeedsLogin(f"the {self._account} Google sign-in has expired or was revoked "
                                     f"(run: python -m jarvis --google-login {self._account}).") from e

        return call


class GoogleAuth:
    def __init__(self, tokens_dir: Path):
        self.dir = tokens_dir
        self.client_secret = tokens_dir / "client_secret.json"

    def token_path(self, account: GoogleAccount) -> Path:
        return self.dir / f"google-{account.name}.json"

    def signed_in(self, account: GoogleAccount) -> bool:
        return self.token_path(account).exists()

    def credentials(self, account: GoogleAccount):
        """Saved credentials for an account, refreshed if needed. Raises NeedsLogin if there are none."""
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        path = self.token_path(account)
        how = f"run: python -m jarvis --google-login {account.name}"
        if not path.exists():
            raise NeedsLogin(f"the {account.name} Google account isn't signed in yet ({how}).")
        # The token's own scopes, not the configured ones: a newly added service shouldn't break
        # what already works before the account signs in again (has_scope() tells them apart).
        creds = Credentials.from_authorized_user_file(str(path))
        if not creds.valid:
            if not creds.refresh_token:
                raise NeedsLogin(f"the {account.name} Google sign-in has expired ({how}).")
            try:
                creds.refresh(Request())
            except RefreshError as e:  # revoked, expired (7 days in Testing), or unused for months
                log.warning("Google refresh failed for %s: %s", account.name, e)
                raise NeedsLogin(f"the {account.name} Google sign-in has expired or was revoked ({how}).") from e
            path.write_text(creds.to_json(), encoding="utf-8")
        return creds

    def has_scope(self, account: GoogleAccount, service: str) -> bool:
        """Whether the saved sign-in already includes a service's permission (no network)."""
        import json

        try:
            data = json.loads(self.token_path(account).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return SCOPES.get(service) in set(data.get("scopes") or [])

    def login(self, account: GoogleAccount) -> str:
        """Open the browser for Google's consent page; save the token. Returns a message for the user."""
        from google_auth_oauthlib.flow import InstalledAppFlow

        if not self.client_secret.exists():
            raise NeedsLogin(f"{self.client_secret} is missing. Follow the Google setup steps in the README first.")
        if not account.scopes:
            raise NeedsLogin(f"account {account.name} has no services listed in config.local.toml.")
        flow = InstalledAppFlow.from_client_secrets_file(str(self.client_secret), account.scopes)
        creds = flow.run_local_server(
            port=0, login_hint=account.email or None, prompt="consent",
            authorization_prompt_message="Opening Google sign-in in your browser. If it doesn't open, visit:\n{url}",
            success_message="Jarvis is signed in. You can close this tab.",
        )
        granted = set(getattr(creds, "granted_scopes", None) or account.scopes)
        missing = [s for s in account.services if SCOPES.get(s) not in granted]
        if "gmail" in account.services and "gmail" not in missing:
            from googleapiclient.discovery import build

            profile = build("gmail", "v1", credentials=creds, cache_discovery=False).users().getProfile(
                userId="me").execute()
            actual = profile.get("emailAddress", "")
            if account.email and actual.lower() != account.email.lower():
                raise NeedsLogin(f"you signed in as {actual}, but account {account.name} is set to {account.email}. "
                                 "Nothing was saved; run the command again and pick the right account.")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.token_path(account).write_text(creds.to_json(), encoding="utf-8")
        note = f"Signed in: {account.name}" + (f" ({account.email})" if account.email else "") + "."
        if missing:
            note += (f" Note: you didn't tick access for {', '.join(missing)} on Google's page, so those won't work"
                     " for this account. Run the command again to change that.")
        return note
