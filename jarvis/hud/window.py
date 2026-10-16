"""The HUD in its own window, without a browser.

Runs as a separate process (pywebview must own its main thread, and Jarvis's main
thread runs the voice loop):

    python -m jarvis.hud.window http://127.0.0.1:8765/ [--fullscreen] [--parent PID]

Uses Microsoft Edge WebView2, which ships with Windows 11. Exits with code 3 if
WebView2 can't be used, so Jarvis can fall back to the browser; 0 when the user
closes the window (Jarvis then quits, if [hud] close_quits).

It closes itself if Jarvis's process goes away (a crash, or the terminal being
closed), tells the page when it's minimised so the animation stops, and comes
to the front when the Jarvis shortcut is clicked again (a named Windows event).
Its errors go to logs/hud-window.log.

With --mode widget it is the small desktop widget instead: frameless, remembers its place, sits behind
the other windows and comes forward for a peek on a hotkey. The full HUD is then opened on demand by
the Jarvis process as a second, ordinary window (--ondemand), which closes itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

NO_WEBVIEW = 3
APP_ID = "Jarvis.Assistant"          # groups the window under the pinned Jarvis shortcut on the taskbar
PEEK_SECONDS = 10                    # how long a peek (hotkey, second launch) stays in front
SHOW_EVENT = "Local\\JarvisShowHud"  # set by a second launch: bring the window to the front


class HudWindow:
    """Starts the window process.

    on_failed() runs if it can't start (e.g. no WebView2); on_closed() if the user closes it.
    """

    def __init__(self, url: str, fullscreen: bool = False, on_failed: Callable[[], None] | None = None,
                 on_closed: Callable[[], None] | None = None, log_file: Path | None = None,
                 extra_args: list[str] | None = None):
        args = [sys.executable, "-m", "jarvis.hud.window", url, "--parent", str(os.getpid())]
        if fullscreen:
            args.append("--fullscreen")
        args += list(extra_args or [])
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        err = subprocess.DEVNULL
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            err = open(log_file, "a", encoding="utf-8")  # noqa: SIM115 (kept open for the child's lifetime)
        self._proc = subprocess.Popen(args, creationflags=flags, stdout=subprocess.DEVNULL, stderr=err)
        self._started = time.monotonic()
        self._on_failed = on_failed
        self._on_closed = on_closed
        threading.Thread(target=self._watch, name="hud-window", daemon=True).start()

    def _watch(self) -> None:
        code = self._proc.wait()
        if code != 0:
            # A quick failure means the window never worked; later ones are logged by the window itself.
            if time.monotonic() - self._started < 20 and self._on_failed:
                self._on_failed()
        elif self._on_closed:
            self._on_closed()

    @property
    def running(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        self._on_failed = self._on_closed = None
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def show_existing() -> bool:
    """Ask a running HUD window to come to the front. False if there isn't one."""
    if sys.platform != "win32":
        return False
    try:
        import win32event

        handle = win32event.OpenEvent(win32event.EVENT_MODIFY_STATE, False, SHOW_EVENT)
    except Exception:
        return False
    win32event.SetEvent(handle)
    return True


# ------------------------------------------------------------------ inside the window process

def _wait_for_parent(pid: int, on_gone: Callable[[], None]) -> None:
    """Call on_gone() when the process `pid` ends (Windows: wait on its handle; elsewhere: poll)."""
    if sys.platform == "win32":
        import ctypes

        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if handle:
            ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
            ctypes.windll.kernel32.CloseHandle(handle)
            on_gone()
            return
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            on_gone()
            return
        time.sleep(2)


def _listen_for_show(window, on_show: Callable[[], None] | None = None) -> None:
    import win32event

    event = win32event.CreateEvent(None, False, False, SHOW_EVENT)
    while True:
        win32event.WaitForSingleObject(event, win32event.INFINITE)
        if on_show:
            try:
                on_show()
            except Exception as e:
                print(f"Couldn't show the widget: {e}", file=sys.stderr)
            continue
        try:
            window.restore()
            window.show()
            window.on_top = True   # a brief always-on-top is the reliable way to come to the front
            window.on_top = False
        except Exception as e:
            print(f"Couldn't bring the window forward: {e}", file=sys.stderr)


# ------------------------------------------------------------------ the desktop widget

def widget_args(settings: dict, state_file: Path) -> list[str]:
    """Command-line options for the widget window process, from the [widget] section of the config."""
    args = ["--mode", "widget", "--state", str(state_file),
            "--orb", str(int(settings.get("orb", 320))),
            "--margin", str(int(settings.get("margin", 24))), "--hotkey", str(settings.get("hotkey", "ctrl+alt+j"))]
    args.append("--collapse")  # the widget is always the round sphere
    if settings.get("on_top", False):
        args.append("--on-top")
    if settings.get("transparent", False):
        args.append("--transparent")
    return args


def parse_hotkey(spec: str) -> tuple[int, int] | None:
    """"ctrl+alt+j" -> (modifier flags, virtual-key code) for RegisterHotKey; None if it can't be read."""
    mods = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
    flags, key = 0, None
    for part in (p.strip().lower() for p in spec.split("+") if p.strip()):
        if part in mods:
            flags |= mods[part]
        elif len(part) == 1 and part.isalnum():
            key = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
            key = 0x70 + int(part[1:]) - 1
        else:
            return None
    return (flags, key) if key and flags else None


def load_place(state_file: Path) -> dict:
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return {k: int(data[k]) for k in ("r", "y") if k in data}
    except (OSError, ValueError, TypeError):
        return {}


def save_place(state_file: Path, place: dict) -> None:
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(place), encoding="utf-8")
    except OSError:
        pass


def start_position(width: int, margin: int, place: dict, screen_w: int) -> tuple[int, int]:
    """Where the widget opens: where he left it, else the top right of the primary monitor.

    The place is kept as the window's right edge ("r") and top ("y"): the widget grows and shrinks leftwards.
    """
    if "r" in place and "y" in place:
        return max(0, place["r"] - width), place["y"]
    return max(0, screen_w - width - margin), margin


class WidgetControl:
    """What the widget page can ask of its window (exposed to the page as window.pywebview.api)."""

    HWND_BOTTOM, HWND_TOPMOST, HWND_NOTOPMOST = 1, -1, -2

    def __init__(self, on_top: bool, state_file: Path | None):
        self._win = None
        self._top = on_top
        self._peeking = False
        self._hidden = False
        self._state_file = state_file
        self._place: dict = {}
        self._timer: threading.Timer | None = None
        self._rounded = False
        self.circle = False
        self._clip: tuple | None = None
        self._peek_timer: threading.Timer | None = None

    # ---- z-order
    def _hwnd(self) -> int:
        try:
            return int(self._win.native.Handle.ToInt64())
        except Exception:
            return 0

    def _set_pos(self, after: int) -> None:
        hwnd = self._hwnd()
        if not hwnd or sys.platform != "win32":
            return
        import win32con
        import win32gui

        flags = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER
        try:
            win32gui.SetWindowPos(hwnd, after, 0, 0, 0, 0, flags)
        except Exception as e:
            print(f"SetWindowPos failed: {e}", file=sys.stderr)

    def style(self) -> None:
        """No activation on click (so clicking never pulls it forward) and no taskbar button."""
        hwnd = self._hwnd()
        if not hwnd or sys.platform != "win32":
            return
        import win32con
        import win32gui

        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        # APPWINDOW would put it in Alt+Tab and on the taskbar whatever TOOLWINDOW says; WinForms sets it.
        want = (ex | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOOLWINDOW) & ~win32con.WS_EX_APPWINDOW
        if want != ex:
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, want)
            win32gui.SetWindowPos(hwnd, 0, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOZORDER
                                  | win32con.SWP_NOACTIVATE | win32con.SWP_FRAMECHANGED)
        self._plain_frame(hwnd)
        if not self._rounded:
            self._rounded = True
            self.round_corners()

    @staticmethod
    def _plain_frame(hwnd: int) -> None:
        """Windows 11 paints a translucent dark box over the whole rectangle of a frameless window (the shadow/backdrop
        of its invisible frame), which showed as a square around the round widget. Switch those DWM effects off."""
        import ctypes
        from ctypes import wintypes

        dwm = ctypes.windll.dwmapi
        for attribute, value in ((38, 1),            # system backdrop: none
                                 (33, 1),            # corner preference: don't round
                                 (34, 0xFFFFFFFE),   # border colour: none
                                 (2, 1)):            # non-client rendering: disabled
            v = ctypes.c_int(value - (1 << 32) if value > 0x7FFFFFFF else value)
            dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), attribute, ctypes.byref(v), 4)

    def round_corners(self, force: bool = True) -> None:
        """Rounded corners (or a circle) by clipping the window: transparent windows don't work with WebView2 here.

        With force=False it only acts when the window's size has changed since the last clip (the 2-second check
        calls it that way, because the window can be resized after it starts and a stale clip shows a square)."""
        hwnd = self._hwnd()
        if not hwnd or sys.platform != "win32":
            return
        import ctypes

        import win32gui

        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if not force and self._clip == (right - left, bottom - top, self.circle):
                return
            self._clip = (right - left, bottom - top, self.circle)
            scale = ctypes.windll.user32.GetDpiForWindow(hwnd) / 96 or 1
            d = int(28 * scale)
            if self.circle:
                region = ctypes.windll.gdi32.CreateEllipticRgn(0, 0, right - left + 1, bottom - top + 1)
            else:
                region = ctypes.windll.gdi32.CreateRoundRectRgn(0, 0, right - left + 1, bottom - top + 1, d, d)
            ctypes.windll.user32.SetWindowRgn(hwnd, region, True)
        except Exception as e:
            print(f"Couldn't round the corners: {e}", file=sys.stderr)

    def settle(self) -> None:
        """Put the window where it belongs when nobody asked for a peek: behind the others (or on top)."""
        if self._hidden or self._peeking:
            return
        self.style()  # cheap when nothing changed; undoes anything that put the window back in Alt+Tab
        self.round_corners(force=False)
        self._set_pos(self.HWND_TOPMOST if self._top else self.HWND_BOTTOM)

    # ---- the hotkey and the menu
    def peek(self) -> None:
        """Hotkey: bring it forward for a look; the next press sends it back."""
        if self._hidden:
            self._hidden = False
            self._win.show()
            self.pause(False)
        if self._peek_timer:
            self._peek_timer.cancel()
        if self._peeking:
            self._peeking = False
            self._set_pos(self.HWND_NOTOPMOST)
            self.settle()
        else:
            self._peeking = True
            self._set_pos(self.HWND_TOPMOST)
            # a peek is a glance: it goes back by itself, so it can never stay stuck on top
            self._peek_timer = threading.Timer(PEEK_SECONDS, self._end_peek)
            self._peek_timer.daemon = True
            self._peek_timer.start()

    def _end_peek(self) -> None:
        if self._peeking:
            self.peek()

    def hide(self) -> None:
        self._hidden, self._peeking = True, False
        self.pause(True)
        self._win.hide()

    def toggle_top(self) -> bool:
        self._top = not self._top
        self._peeking = False
        self._set_pos(self.HWND_NOTOPMOST)
        self.settle()
        return self._top

    def quit(self) -> None:
        self._win.destroy()

    def resize(self, width: int, height: int, circle: bool = False) -> None:
        """The page asks for another size: the small sphere, or the panel with the reply. The right edge stays put."""
        width, height = max(100, min(int(width), 700)), max(100, min(int(height), 520))
        if (width, height, circle) == (self._win.width, self._win.height, self.circle):
            return
        right = self._win.x + self._win.width
        self.circle = bool(circle)
        self._win.resize(width, height)
        self._win.move(right - width, self._win.y)
        threading.Timer(0.25, self.round_corners).start()

    def pause(self, paused: bool) -> None:
        try:
            self._win.evaluate_js(f"window.hudPause && window.hudPause({'true' if paused else 'false'})")
        except Exception:
            pass

    # ---- remembering the place
    def moved(self, x: int, y: int) -> None:
        self._place.update(r=int(x) + int(self._win.width), y=int(y))
        self._later_save()

    def _later_save(self) -> None:
        if not self._state_file:
            return
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(1.0, save_place, args=(self._state_file, dict(self._place)))
        self._timer.daemon = True
        self._timer.start()


def _listen_for_hotkey(spec: str, on_press: Callable[[], None]) -> None:
    """A global hotkey through RegisterHotKey, with its own message loop on this thread."""
    parsed = parse_hotkey(spec)
    if not parsed or sys.platform != "win32":
        print(f"Widget hotkey {spec!r} not understood; none registered.", file=sys.stderr)
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    if not user32.RegisterHotKey(None, 1, parsed[0] | 0x4000, parsed[1]):  # 0x4000: don't repeat while held
        print(f"Widget hotkey {spec!r} is taken by another program.", file=sys.stderr)
        return
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        if msg.message == 0x0312:  # WM_HOTKEY
            try:
                on_press()
            except Exception as e:
                print(f"Hotkey action failed: {e}", file=sys.stderr)


def _make_webview_transparent(window) -> None:
    """pywebview asks WebView2 for a transparent background from the wrong thread and it silently stays white;
    doing it on the form's own (UI) thread makes the page's see-through pixels really see through."""
    try:
        import System
        from System import Action

        def apply() -> None:
            window.native.browser.webview.DefaultBackgroundColor = System.Drawing.Color.Transparent

        window.native.Invoke(Action(apply))
    except Exception as e:
        print(f"Couldn't make the page transparent: {type(e).__name__}: {e}", file=sys.stderr)


def _primary_width() -> int:
    if sys.platform == "win32":
        import ctypes

        return int(ctypes.windll.user32.GetSystemMetrics(0)) or 1920
    return 1920


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jarvis.hud.window")
    parser.add_argument("url")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--parent", type=int, default=0)
    parser.add_argument("--mode", choices=["hud", "widget"], default="hud")
    parser.add_argument("--ondemand", action="store_true", help="the full HUD opened from the widget: no show-event")
    parser.add_argument("--state", default="")
    parser.add_argument("--width", type=int, default=380)
    parser.add_argument("--height", type=int, default=170)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--hotkey", default="ctrl+alt+j")
    parser.add_argument("--orb", type=int, default=320)
    parser.add_argument("--collapse", action="store_true")
    parser.add_argument("--on-top", action="store_true")
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args(argv)
    try:
        import webview
    except ImportError:
        return NO_WEBVIEW
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    control: WidgetControl | None = None
    if args.mode == "widget":
        state_file = Path(args.state) if args.state else None
        place = load_place(state_file) if state_file else {}
        width, height = (args.orb, args.orb) if args.collapse else (args.width, args.height)
        x, y = start_position(width, args.margin, place, _primary_width())
        control = WidgetControl(args.on_top, state_file)
        control._place = dict(place)
        control.circle = args.collapse
        window = webview.create_window("Jarvis", args.url, js_api=control, width=width, height=height,
                                       x=x, y=y, min_size=(100, 100), resizable=False, frameless=True,
                                       easy_drag=False, shadow=False, on_top=args.on_top, focus=False,
                                       transparent=args.transparent, background_color="#02090d")
        control._win = window
        window.events.moved += control.moved
        window.events.shown += lambda: control.settle()
    else:
        window = webview.create_window("Jarvis", args.url, width=1280, height=720, min_size=(640, 360),
                                       fullscreen=args.fullscreen, background_color="#000000")

    def pause(paused: bool) -> None:  # stop the animation while nobody can see it
        try:
            window.evaluate_js(f"window.hudPause && window.hudPause({'true' if paused else 'false'})")
        except Exception:
            pass

    window.events.minimized += lambda: pause(True)
    window.events.restored += lambda: pause(False)
    window.events.maximized += lambda: pause(False)

    def keep_behind() -> None:
        while True:  # a new window can reorder things; this does nothing when the order is already right
            time.sleep(2)
            control.settle()

    def started() -> None:
        if args.parent:
            threading.Thread(target=_wait_for_parent, args=(args.parent, window.destroy), daemon=True).start()
        if control:
            time.sleep(0.5)
            if args.transparent:
                _make_webview_transparent(window)
            if args.collapse:  # frameless creation comes out a little short on one side; make it truly square
                window.resize(args.orb, args.orb)
                window.move(x, y)
                for delay in (0.7, 2.0):  # the resize lands on the UI thread a moment later; clip to the final size
                    threading.Timer(delay, control.round_corners).start()
            control.style()
            control.settle()
            threading.Thread(target=_listen_for_hotkey, args=(args.hotkey, control.peek), daemon=True).start()
            threading.Thread(target=keep_behind, daemon=True).start()
            if sys.platform == "win32":  # a second launch of Jarvis = a peek at the widget
                threading.Thread(target=_listen_for_show, args=(window, control.peek), daemon=True).start()
        elif sys.platform == "win32" and not args.ondemand:
            threading.Thread(target=_listen_for_show, args=(window,), daemon=True).start()

    try:
        # Edge WebView2 only: pywebview's other Windows backend is old Internet Explorer, which can't run the HUD.
        webview.start(started, gui="edgechromium", private_mode=True)
    except Exception as e:
        print(f"WebView2 failed: {type(e).__name__}: {e}", file=sys.stderr)
        return NO_WEBVIEW
    return 0


if __name__ == "__main__":
    sys.exit(main())
