"""One-click start: the Jarvis icon and a Start-menu shortcut you can pin to the taskbar.

    python -m jarvis --install-shortcut

creates "Jarvis" in the Start menu: it starts Jarvis without a console window
(pythonw), the HUD opens in its own window, and closing that window quits
Jarvis. Clicking it again while Jarvis runs brings the window to the front.
The shortcut and the window share an AppUserModelID, so the running window sits
under the pinned icon on the taskbar. (Windows doesn't let programs pin
themselves: right-click the shortcut, then "Pin to taskbar".)
"""

from __future__ import annotations

import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

from .hud.window import APP_ID

ICON_SIZES = (256, 64, 48, 32, 24, 16)


# ------------------------------------------------------------------ the icon (drawn here: no image library needed)

def _draw(size: int) -> np.ndarray:
    """An arc-reactor ring, RGBA, anti-aliased by drawing at 4x and averaging."""
    ss = 4
    n = size * ss
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1  # -1..1
    r = np.hypot(x, y)
    a = (np.degrees(np.arctan2(y, x)) + 360) % 360
    rgba = np.zeros((n, n, 4))

    def paint(mask, colour, alpha=1.0):
        for i, c in enumerate(colour):
            rgba[..., i] = np.where(mask, c, rgba[..., i])
        rgba[..., 3] = np.where(mask, np.maximum(rgba[..., 3], alpha), rgba[..., 3])

    cyan, ice, amber, dark = (0.16, 0.86, 1.0), (0.85, 0.98, 1.0), (1.0, 0.66, 0.2), (0.02, 0.05, 0.08)
    paint(r < 0.98, dark)                                                     # the dark disc
    paint((r > 0.84) & (r < 0.93), cyan)                                      # outer ring
    paint((r > 0.84) & (r < 0.93) & (a > 300) & (a < 345), amber)             # a short amber arc
    small = size <= 24
    if not small:
        paint((r > 0.64) & (r < 0.74) & ((a % 30) < 22), cyan)                # segmented middle ring
    glow = np.clip(1 - r / 0.42, 0, 1)
    core = r < (0.46 if small else 0.40)
    for i, c in enumerate(ice):
        rgba[..., i] = np.where(core, c * (0.55 + 0.45 * glow) + cyan[i] * (1 - glow) * 0.6, rgba[..., i])
    rgba[..., 3] = np.where(core, 1.0, rgba[..., 3])
    if not small:
        paint((r > 0.16) & (r < 0.22), cyan)                                  # a ring inside the core
    out = rgba.reshape(size, ss, size, ss, 4).mean(axis=(1, 3))
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def _png(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[row].tobytes() for row in range(h))  # filter type 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def make_icon(path: Path) -> Path:
    """Write a Windows .ico with PNG images at the usual sizes."""
    images = [_png(_draw(s)) for s in ICON_SIZES]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for s, data in zip(ICON_SIZES, images):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + entries + b"".join(images))
    return path


# ------------------------------------------------------------------ the shortcut

def start_menu_dir() -> Path:
    return Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def install_shortcut(root: Path, folder: Path | None = None) -> Path:
    """Create Jarvis.lnk (in the Start menu by default). Returns its path."""
    import pythoncom
    from win32com.propsys import propsys, pscon
    from win32com.shell import shell

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        raise FileNotFoundError(f"{pythonw} is missing")
    icon = make_icon(root / "jarvis" / "hud" / "static" / "jarvis.ico")
    link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER,
                                      shell.IID_IShellLink)
    link.SetPath(str(pythonw))
    link.SetArguments("-m jarvis")
    link.SetWorkingDirectory(str(root))
    link.SetIconLocation(str(icon), 0)
    link.SetDescription("Jarvis, your voice assistant")
    store = link.QueryInterface(propsys.IID_IPropertyStore)
    store.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(APP_ID, pythoncom.VT_LPWSTR))
    store.Commit()
    target = (folder or start_menu_dir()) / "Jarvis.lnk"
    target.parent.mkdir(parents=True, exist_ok=True)
    link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(target), 0)
    return target


def message_box(text: str, title: str = "Jarvis") -> None:
    """For when there's no console to print to (started from the shortcut)."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, title, 0x40)  # MB_ICONINFORMATION
