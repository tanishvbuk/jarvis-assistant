"""The HUD: a full-screen local web page showing what Jarvis is doing."""

__all__ = ["Hud"]


def __getattr__(name):
    # Imported lazily so the window process (python -m jarvis.hud.window) doesn't load the web server.
    if name == "Hud":
        from .server import Hud

        return Hud
    raise AttributeError(name)
