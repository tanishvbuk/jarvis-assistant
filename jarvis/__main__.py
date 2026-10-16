"""Entry point.

    python -m jarvis                  push-to-talk voice loop
    python -m jarvis --text           type instead of talk (no microphone needed)
    python -m jarvis --text --speak   type, and hear the replies
    python -m jarvis --list-voices    Windows voices installed
    python -m jarvis --list-devices   microphones and speakers
    python -m jarvis --provider anthropic   override llm.provider for this run
    python -m jarvis --no-hud         voice loop without the screen
    python -m jarvis --no-wake        voice loop without "hey Jarvis" (push-to-talk only)
    python -m jarvis --text --hud     text mode with the screen
    python -m jarvis --google-login personal   sign a Google account in (read-only); "all" for every account
    python -m jarvis --google-status  check each Google account's sign-in
    python -m jarvis --opengym-pair CODE   pair Jarvis with the openGym gym tracker (code from its Settings)
    python -m jarvis --install-shortcut   a Start-menu "Jarvis" to pin to the taskbar (one click: Jarvis + HUD)
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from datetime import datetime

from . import events
from .assistant import Assistant
from .connectors import build_connectors, connector_status
from .skills import build_skills
from .config import Config, ConfigError, load_config
from .connectors.briefing import briefing_from_config
from .llm import LLMError, make_backend
from .persona import LOCAL
from .turnlog import TurnLog, TurnRecord

log = logging.getLogger("jarvis")


def setup_logging(cfg: Config) -> None:
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(cfg.log_dir / "jarvis.log", encoding="utf-8")],
    )
    if sys.stderr is not None:  # none when started from the shortcut (pythonw)
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(console)
    # The HTTP libraries log every request URL at INFO; keep the log about Jarvis.
    for noisy in ("httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def greeting() -> str:
    hour = datetime.now(LOCAL).hour
    if hour < 5:
        return "Working late, sir. Jarvis online."
    part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
    return f"Good {part}, sir. Jarvis online."


def timings(rec: TurnRecord) -> str:
    def f(v):
        return "-" if v is None else f"{v:.2f}s"

    cost = f", ~£{rec.est_cost_gbp:.5f}" if rec.est_cost_gbp else ""
    return (f"  [record {f(rec.record_s)} | transcribe {f(rec.transcribe_s)} | first token {f(rec.first_token_s)} | "
            f"first audio {f(rec.first_audio_s)} | tokens {rec.input_tokens}/{rec.output_tokens}{cost}]")


def end_of_turn(rec: TurnRecord) -> None:
    """Tell the HUD how the turn went (timings only, no text)."""
    events.publish("turn", first_audio_s=rec.first_audio_s or rec.first_token_s, total_s=rec.total_s)


def make_assistant(cfg: Config) -> tuple[Assistant, list]:
    """The assistant with every connector's and skill's tools. Returns it and the connectors + skills."""
    connectors = build_connectors(cfg)
    print(f"Connectors: {connector_status(cfg, connectors)}")
    connectors += build_skills(cfg)
    timers = next((c for c in connectors if c.name == "timers"), None)
    for c in connectors:
        if c.name == "tasks":
            c.timers = timers  # reminders an hour before timed deadlines
    briefing = briefing_from_config(cfg, connectors)
    notes = [c.daily_note for c in connectors if hasattr(c, "daily_note")]
    if briefing:
        notes.insert(0, briefing.daily_note)  # first: on the morning's first turn it folds the other once-a-day notes in
        connectors.append(briefing)
    return Assistant(cfg, make_backend(cfg.provider), tools=[t for c in connectors for t in c.tools()],
                     notes=notes + [c.pending_note for c in connectors if hasattr(c, "pending_note")],
                     preludes=[briefing.prelude] if briefing else []), connectors


def start_timers(connectors: list, alert) -> None:
    """Start the timers skill's alarm clock, if it's on; alert(text) speaks or prints when one goes off."""
    for c in connectors:
        if c.name == "timers":
            c.start(alert)


def stop_timers(connectors: list) -> None:
    for c in connectors:
        if c.name == "timers":
            c.stop()


class Jobs:
    """Background work for connectors, every 30 seconds: polling feeds, refreshing HUD cards.

    Each connector's tick() decides what is due. A HUD page coming on screen runs them at
    once, so cards skipped while nobody was looking catch up straight away. set() stops it.
    """

    def __init__(self, connectors: list):
        self._jobs = [c for c in connectors if hasattr(c, "tick")]
        self._wake = threading.Event()
        self._stopped = False
        self._unsubscribe = events.subscribe(self._on_event)
        if self._jobs:
            threading.Thread(target=self._run, name="connector-jobs", daemon=True).start()

    def _on_event(self, event: dict) -> None:
        if event["t"] == "watched" and event.get("v"):
            self._wake.set()

    def _run(self) -> None:
        while not self._stopped:
            for c in self._jobs:
                try:
                    c.tick()
                except Exception:
                    log.exception("Background job for %s failed", c.name)
            self._wake.wait(30)
            self._wake.clear()

    def set(self) -> None:
        self._stopped = True
        self._unsubscribe()
        self._wake.set()


def start_jobs(connectors: list) -> Jobs:
    return Jobs(connectors)


# ---------------------------------------------------------------- HUD

def start_hud(cfg: Config, on_talk=None, **info):
    """Start the HUD web server; returns it, or None if it couldn't start (Jarvis carries on without)."""
    from .hud import Hud

    hud = Hud(int(cfg.hud.get("port", 8765)), on_talk=on_talk, info=info)
    try:
        url = hud.start()
    except OSError as e:
        print(f"  HUD not started: port {hud.port} is busy ({e.strerror or e}). Change [hud] port in config.toml.")
        log.warning("HUD not started: %s", e)
        return None
    print(f"HUD: {url}")
    show_hud(url, hud, cfg.hud, log_file=cfg.log_dir / "hud-window.log", widget=cfg.widget,
             data_dir=cfg.root / "data")
    return hud


def show_hud(url: str, hud, settings: dict, log_file=None, widget: dict | None = None, data_dir=None) -> None:
    """Open the HUD as configured: its own window ("app"), the small desktop widget ("widget"), a browser tab
    ("browser"), or not at all ("none").

    With [hud] close_quits (the default), closing the HUD window, or the widget, quits Jarvis. The full HUD that
    the widget opens on request never does: it closes itself after every use.
    """
    import _thread
    import webbrowser

    mode = str(settings.get("window", "app")).lower()
    if mode == "widget":
        from .hud.widget import WidgetHost
        from .hud.window import HudWindow, widget_args

        def widget_failed() -> None:
            print("  The widget needs Microsoft Edge WebView2; opening the full HUD in the browser instead.")
            log.warning("Widget window failed to start; using the browser")
            webbrowser.open(url)

        def hud_failed() -> None:
            print("  The HUD window needs Microsoft Edge WebView2; opening it in the browser instead.")
            webbrowser.open(url + "?auto=1")

        def widget_closed() -> None:
            log.info("Widget closed; quitting")
            _thread.interrupt_main()

        host = WidgetHost(hud, lambda u: HudWindow(u, on_failed=hud_failed, log_file=log_file,
                                                    extra_args=["--ondemand"]))
        hud.widget_host = host
        state_file = (data_dir / "widget.json") if data_dir else None
        page = url + "widget"
        hud.window = HudWindow(page, on_failed=widget_failed, log_file=log_file,
                               on_closed=widget_closed if (widget or {}).get("close_quits", True) else None,
                               extra_args=widget_args(widget or {}, state_file))
    elif mode == "app":
        from .hud.window import HudWindow

        def fall_back() -> None:
            print("  HUD window unavailable (needs Microsoft Edge WebView2); opening it in the browser instead.")
            log.warning("HUD window failed to start; using the browser")
            webbrowser.open(url)

        def closed() -> None:
            log.info("HUD window closed; quitting")
            _thread.interrupt_main()  # the voice loop's Ctrl+C path: tidy shutdown

        hud.window = HudWindow(url, fullscreen=bool(settings.get("fullscreen", False)), on_failed=fall_back,
                               on_closed=closed if settings.get("close_quits", True) else None, log_file=log_file)
    elif mode == "browser":
        webbrowser.open(url)


# ---------------------------------------------------------------- text mode

def run_text(cfg: Config, speak: bool, hud: bool = False) -> None:
    assistant, connectors = make_assistant(cfg)
    briefing = next((c for c in connectors if c.name == "briefing"), None)
    turnlog = TurnLog(cfg.log_dir)
    speech = None
    if speak:
        from .tts import SpeechQueue, voice_factory

        speech = SpeechQueue(voice_factory(cfg))
        print(f"Voice: {speech.voice_name}")
    screen = start_hud(cfg, brain=f"{cfg.provider.name} / {cfg.provider.model}", ears="keyboard",
                       voice=speech.voice_name if speech else "silent") if hud else None
    jobs = start_jobs(connectors)

    def alert(text: str) -> None:
        print(f"\n*** {text}")
        events.publish("notice", text="Timer")
        if speech:
            if not speech.speaking:
                speech.new_turn()
            speech.say(text)

    start_timers(connectors, alert)
    print(f"Jarvis (text mode) using {cfg.provider.name} / {cfg.provider.model}. Type 'quit' to leave.\n")

    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"quit", "exit", "bye"}:
            break
        rec = TurnRecord(mode="text", voice=speech.voice_name if speech else "")
        t0 = time.perf_counter()
        print("Jarvis:", end="", flush=True)
        if speech:
            speech.new_turn()
        events.publish("heard", text=line)
        events.state("thinking")

        def say(sentence: str) -> None:
            print(" " + sentence, end="", flush=True)
            if speech:
                speech.say(sentence)  # the speech queue tells the HUD as it speaks
            else:
                events.state("speaking")
                events.publish("say", text=sentence)

        said = screen.widget_host.show_me(line) if screen and getattr(screen, "widget_host", None) else None
        if said:
            say(said)
        else:
            if briefing:
                briefing.activate()
            assistant.respond(line, say, rec)
        print()
        if speech:
            speech.wait()
            if speech.first_audio_at:
                rec.first_audio_s = speech.first_audio_at - t0
        rec.total_s = time.perf_counter() - t0
        turnlog.write(rec)
        events.state("idle")
        end_of_turn(rec)
        print(timings(rec))
    stop_timers(connectors)
    if speech:
        speech.close()
    jobs.set()
    if screen:
        screen.stop()


# ---------------------------------------------------------------- voice mode

_instance_lock = None  # held for the life of the process


def claim_single_instance() -> bool:
    """Two voice loops would both answer F9, so only one may run at a time."""
    global _instance_lock
    if sys.platform != "win32":
        return True
    import win32api
    import win32event
    import winerror

    _instance_lock = win32event.CreateMutex(None, False, "Local\\JarvisVoiceLoop")
    return win32api.GetLastError() != winerror.ERROR_ALREADY_EXISTS


def make_wake_listener(cfg: Config, presses: queue.Queue, busy, device):
    """The "hey Jarvis" listener, or None (with a note) if its models are missing or it won't load."""
    model_dir = cfg.root / "models" / "openwakeword"
    if not (model_dir / "hey_jarvis_v0.1.onnx").exists():
        print("  Wake word off: models/openwakeword is missing (see the README).")
        return None
    try:
        from .audio.wake import WakeListener

        return WakeListener(on_wake=lambda: presses.put(("wake",)),
                            on_utterance=lambda audio, secs, by_phrase: presses.put(("utterance", audio, secs,
                                                                                       by_phrase)),
                            on_nothing=lambda: presses.put(("nothing",)), busy=busy, model_dir=model_dir,
                            threshold=float(cfg.audio.get("wake_threshold", 0.5)), device=device)
    except Exception as e:
        log.exception("Wake word failed to load")
        print(f"  Wake word off: it failed to load ({type(e).__name__}); details in the log.")
        return None


def run_voice(cfg: Config, hud: bool = True, wake: bool = True) -> None:
    if not claim_single_instance():
        from .hud.window import show_existing

        if show_existing():  # clicked the shortcut again: bring the HUD forward instead
            return
        print("Jarvis is already running in another window. Use that one, or press Ctrl+C there first.")
        if sys.stdout is None:
            from .launcher import message_box

            message_box("Jarvis is already running.")
        return
    from .audio.capture import PushToTalk, Recorder
    from .stt import make_transcriber
    from .tts import SpeechQueue, voice_factory

    sample_rate = int(cfg.audio.get("sample_rate", 16000))
    key = cfg.audio.get("push_to_talk_key", "f9")
    assistant, connectors = make_assistant(cfg)
    briefing = next((c for c in connectors if c.name == "briefing"), None)
    stt = make_transcriber(cfg)
    turnlog = TurnLog(cfg.log_dir)
    turn: threading.Thread | None = None
    recording_since: float | None = None

    def maybe_idle() -> None:
        """Speech has run out: the HUD goes idle unless a turn or a recording is under way."""
        if recording_since is None and not (turn and turn.is_alive()):
            events.state("idle")

    print("Loading voice...")
    speech = SpeechQueue(voice_factory(cfg), on_drained=maybe_idle)
    device = cfg.audio.get("input_device") or None
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    recorder = Recorder(sample_rate, float(cfg.audio.get("max_record_seconds", 30)), device)

    # "down"/"up" from the push-to-talk key and the HUD's on-screen ring; ("wake",), ("utterance", audio, secs)
    # and ("nothing",) from the wake word listener.
    presses: queue.Queue = queue.Queue()
    ptt = PushToTalk(key, lambda: presses.put("down"), lambda: presses.put("up"))
    listener = make_wake_listener(
        cfg, presses, device=device,
        busy=lambda: recording_since is not None or bool(turn and turn.is_alive()) or speech.speaking) if wake else None
    screen = start_hud(cfg, on_talk=presses.put, brain=f"{cfg.provider.name} / {cfg.provider.model}",
                       ears=stt.name, voice=speech.voice_name, wake=listener is not None) if hud else None
    jobs = start_jobs(connectors)

    def alert(text: str) -> None:
        """A timer went off: chime (if he's quiet), then say it after anything he's already saying."""
        print(f"  *** {text}")
        events.publish("notice", text="Timer")
        if not speech.speaking and recording_since is None:
            speech.new_turn()
            try:
                from .audio.chime import play_chime

                play_chime()
            except Exception:
                log.exception("Chime failed")
        speech.say(text)

    start_timers(connectors, alert)
    print(f"\nJarvis ready: brain {cfg.provider.name} / {cfg.provider.model}, ears {stt.name}, voice {speech.voice_name}.")
    if listener:
        print(f'Say "Hey Jarvis" and ask, or hold {key.upper()} and speak. Press {key.upper()} while he talks to '
              "interrupt. Ctrl+C here to quit.\n")
    else:
        print(f"Hold {key.upper()} and speak; release when done. Press {key.upper()} while he talks to interrupt. "
              "Ctrl+C here to quit.\n")
    speech.say(greeting())
    if listener:
        try:
            listener.start()
        except Exception as e:
            log.exception("Wake word microphone failed")
            print(f"  Wake word off: the microphone wouldn't open ({e}).")
            listener = None

    def do_turn(audio, record_s: float, released_at: float, cancel: threading.Event, woken: str = "") -> None:
        try:
            run_turn(audio, record_s, released_at, cancel, woken)
        finally:
            # An interrupted turn leaves the state to the new recording; an apology still
            # being spoken goes idle through maybe_idle when it finishes.
            if not cancel.is_set() and not speech.speaking:
                events.state("idle")

    def run_turn(audio, record_s: float, released_at: float, cancel: threading.Event, woken: str) -> None:
        rec = TurnRecord(mode="voice", stt=stt.name, voice=speech.voice_name, record_s=record_s)
        speech.new_turn()
        try:
            text = stt.transcribe(audio, sample_rate)
        except LLMError as e:
            log.warning("Transcription failed: %s", e)
            rec.error = f"stt:{type(e).__name__}"
            speech.say("I'm afraid I couldn't make that out, sir. The transcription service didn't respond.")
            turnlog.write(rec)
            return
        except Exception:
            log.exception("Transcription failed")
            rec.error = "stt:Exception"
            speech.say("My hearing appears to be malfunctioning, sir. The details are in the log.")
            turnlog.write(rec)
            return
        rec.transcribe_s = time.perf_counter() - released_at
        if cancel.is_set():
            return
        if woken:
            from .audio.wake import heard_wake_phrase, strip_wake_phrase

            if woken == "phrase" and not heard_wake_phrase(text):
                # The model thought it heard "hey Jarvis" but the transcript says otherwise: a false alarm.
                print("  (woke by mistake; ignored)")
                log.info("Wake word false alarm ignored")
                events.publish("notice", text="Ignored: no 'hey Jarvis'")
                return
            text = strip_wake_phrase(text)
            if not text:  # just "Hey Jarvis": ask, then listen once more without the phrase
                speech.say("Yes, sir?")
                speech.wait()
                if listener and not cancel.is_set():
                    events.state("listening")
                    listener.listen_now()
                return
        if not text:
            print("  (heard nothing)")
            events.publish("notice", text="Heard nothing")
            return
        print(f"You: {text}")
        events.publish("heard", text=text)  # on screen only; never logged
        host = getattr(screen, "widget_host", None)
        said = host.show_me(text) if host else None  # plain "show me": opens the full HUD, no model call
        if said:
            print(f"Jarvis: {said}")
            speech.say(said)
            speech.wait()
            turnlog.write(rec)
            return
        if briefing:
            briefing.activate()  # the first time this morning: the briefing screen, and the briefing in this turn
        reply = assistant.respond(text, speech.say, rec, cancel)
        print(f"Jarvis: {reply}")
        speech.wait()
        if speech.first_audio_at:
            rec.first_audio_s = speech.first_audio_at - released_at
        rec.total_s = time.perf_counter() - released_at
        turnlog.write(rec)
        end_of_turn(rec)
        print(timings(rec))

    ptt.start()
    cancel = threading.Event()
    try:
        while True:
            try:
                ev = presses.get(timeout=0.2)  # timeout keeps Ctrl+C responsive on Windows
            except queue.Empty:
                continue
            if isinstance(ev, tuple):  # from the wake word listener
                if ev[0] == "wake":
                    events.state("listening")
                    print("  (hey Jarvis) listening...", flush=True)
                elif ev[0] == "utterance" and recording_since is None and not (turn and turn.is_alive()):
                    _, audio, record_s, by_phrase = ev
                    events.state("thinking")
                    cancel = threading.Event()
                    woken = "phrase" if by_phrase else "follow-up"
                    turn = threading.Thread(target=do_turn, args=(audio, record_s, time.perf_counter(), cancel, woken),
                                            daemon=True)
                    turn.start()
                elif ev[0] == "nothing":
                    print("  (didn't catch that)")
                    events.publish("notice", text="Didn't catch that")
                    if not speech.speaking:
                        events.state("idle")
                continue
            if ev == "down" and recording_since is None:
                if listener:
                    listener.cancel()  # push-to-talk wins over a half-heard request
                recording_since = time.perf_counter()  # set first, so speech stopping doesn't flash idle
                events.state("listening")
                cancel.set()
                speech.stop()
                if turn and turn.is_alive():
                    turn.join(timeout=5)
                try:
                    recorder.start()
                except Exception as e:
                    recording_since = None
                    print(f"  Microphone error: {e}")
                    log.error("Microphone error: %s", e)
                    events.publish("notice", text="Microphone error")
                    events.state("idle")
                    continue
                recording_since = time.perf_counter()
                print("  listening...", flush=True)
            elif ev == "up" and recording_since is not None:
                audio = recorder.stop()
                released_at = time.perf_counter()
                record_s = released_at - recording_since
                if len(audio) < 0.3 * sample_rate:
                    recording_since = None
                    print("  (too short; hold the key while you speak)")
                    events.publish("notice", text="Too short: hold while you speak")
                    events.state("idle")
                    continue
                events.state("thinking")
                cancel = threading.Event()
                turn = threading.Thread(target=do_turn, args=(audio, record_s, released_at, cancel), daemon=True)
                turn.start()
                recording_since = None  # after the turn is alive, so maybe_idle never sees a gap
    except KeyboardInterrupt:
        print("\nGoodbye, sir.")
    finally:
        ptt.stop()
        if listener:
            listener.stop()
        stop_timers(connectors)
        speech.close()
        jobs.set()
        if screen:
            screen.stop()


# ---------------------------------------------------------------- helpers

def list_voices(cfg: Config) -> None:
    from .tts.sapi import list_voices as sapi_voices

    print("Windows (SAPI) voices:")
    for v in sapi_voices():
        print("  " + v)
    piper_dir = cfg.root / "models" / "piper"
    found = sorted(piper_dir.glob("*.onnx")) if piper_dir.exists() else []
    print("\nPiper voices in models/piper:")
    print("\n".join("  " + p.name for p in found) or "  (none downloaded yet)")


def list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())
    print("\n> = default input, < = default output. Put a number in config.toml [audio] input_device to choose.")


def _google_account(cfg: Config, name: str):
    from .connectors.google import accounts_from_config

    accounts = {a.name: a for a in accounts_from_config(cfg.google)}
    if name not in accounts:
        known = ", ".join(accounts) or "none yet"
        raise ConfigError(f"No Google account called {name!r} in config.local.toml (accounts: {known}).")
    return accounts[name]


def google_login(cfg: Config, name: str) -> None:
    """Sign one account in, or every account with name "all" (handy while the app is in Testing,
    where Google signs everyone out after 7 days)."""
    from .connectors import google_auth
    from .connectors.google import NeedsLogin, accounts_from_config

    accounts = accounts_from_config(cfg.google) if name == "all" else [_google_account(cfg, name)]
    print("Google will say the app isn't verified: click Continue (or Advanced, then 'Go to Jarvis'). "
          "It's your own app, so this is expected.")
    for account in accounts:
        print(f"\nSigning in {account.name} ({account.email or 'any account'}) for: "
              f"{', '.join(account.services)}.")
        changes = [s for s in account.services if s in ("calendar_write", "tasks")]
        print("  Read-only, except " + " and ".join(changes) + ", which can make changes (deleting asks you first)."
              if changes else "  All read-only.")
        if len(accounts) > 1:
            try:
                input("Press Enter to open the browser (Ctrl+C to stop)...")
            except (EOFError, KeyboardInterrupt):
                print("\nStopped.")
                return
        try:
            print(google_auth(cfg).login(account))
        except NeedsLogin as e:
            print(f"Not signed in: {e}")


def google_status(cfg: Config) -> None:
    from .connectors import google_auth
    from .connectors.base import ToolError
    from .connectors.google import accounts_from_config

    auth = google_auth(cfg)
    accounts = accounts_from_config(cfg.google)
    print(f"Client secret: {'found' if auth.client_secret.exists() else 'MISSING'} ({auth.client_secret})")
    if not accounts:
        print("No accounts in config.local.toml yet.")
    for a in accounts:
        try:
            auth.credentials(a)
            if "gmail" in a.services:
                from googleapiclient.discovery import build

                build("gmail", "v1", credentials=auth.credentials(a), cache_discovery=False).users().getProfile(
                    userId="me").execute()
            print(f"  {a.name}: OK ({', '.join(a.services)})")
        except ToolError as e:
            print(f"  {a.name}: {e}")
        except Exception as e:
            print(f"  {a.name}: signed in, but Google refused a test request ({type(e).__name__}: {e})")


def opengym_pair(cfg: Config, code: str) -> None:
    """Pair Jarvis with openGym using the code from Settings > Account > Pair the mobile app (see connectors/opengym.py)."""
    from .connectors.base import ToolError
    from .connectors.opengym import OpenGymClient

    path = cfg.root / cfg.opengym.get("session_file", "tokens/opengym.json")
    client = OpenGymClient(cfg.opengym.get("url", "http://127.0.0.1:8080"), path)
    try:
        session = client.pair(code)
    except ToolError as e:
        print(f"Not paired: {e}")
        return
    try:
        client.get_state()
        check = "openGym accepted it."
    except ToolError as e:
        check = f"Saved, but the test request failed: {e}"
    print(f"Paired with openGym as {session['user']} until {session['expires']} (saved to {path}). {check}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jarvis", description="Jarvis voice assistant")
    parser.add_argument("--text", action="store_true", help="type instead of talk")
    parser.add_argument("--speak", action="store_true", help="with --text: also speak the replies")
    parser.add_argument("--provider", help="override llm.provider from config.toml")
    parser.add_argument("--hud", action="store_true", help="with --text: also show the HUD")
    parser.add_argument("--no-hud", action="store_true", help="voice mode without the HUD")
    parser.add_argument("--no-wake", action="store_true", help='voice mode without "hey Jarvis"')
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--google-login", metavar="ACCOUNT", help="sign a Google account in (name from config.local.toml, or all)")
    parser.add_argument("--google-status", action="store_true", help="check each Google account's sign-in")
    parser.add_argument("--opengym-pair", metavar="CODE", help="pair with openGym (Settings > Account > Pair the mobile app)")
    parser.add_argument("--install-shortcut", action="store_true",
                        help='create a Start-menu "Jarvis" shortcut (then right-click it: Pin to taskbar)')
    args = parser.parse_args(argv)

    try:
        cfg = load_config(provider=args.provider)
        setup_logging(cfg)
        if args.list_voices:
            list_voices(cfg)
        elif args.list_devices:
            list_devices()
        elif args.google_login:
            google_login(cfg, args.google_login)
        elif args.google_status:
            google_status(cfg)
        elif args.opengym_pair:
            opengym_pair(cfg, args.opengym_pair)
        elif args.install_shortcut:
            from .launcher import install_shortcut

            path = install_shortcut(cfg.root)
            print(f"Created {path}.\nTo pin it: press the Windows key, type Jarvis, right-click it, Pin to taskbar.")
        elif args.text:
            run_text(cfg, args.speak, hud=args.hud)
        else:
            run_voice(cfg, hud=bool(cfg.hud.get("enabled", True)) and not args.no_hud,
                      wake=bool(cfg.audio.get("wake_word", True)) and not args.no_wake)
    except ConfigError as e:
        print(f"Setup problem: {e}")
        if sys.stdout is None:  # started from the shortcut: there's no console to read
            from .launcher import message_box

            message_box(f"Jarvis couldn't start: {e}")
        return 2
    except Exception:
        if sys.stdout is not None:
            raise
        log.exception("Jarvis stopped unexpectedly")
        from .launcher import message_box

        message_box("Jarvis stopped unexpectedly. The details are in logs\\jarvis.log.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
