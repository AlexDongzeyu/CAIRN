"""File-backed logging.

The interactive terminal in this environment drops output intermittently, so every
experiment writes its own log and results are read back from disk rather than scraped
from the console.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "data" / "logs"


def make_logger(name: str):
    """Append-mode logger with a run banner, written lazily.

    Truncating at import means that merely importing a module - which the coverage audit
    and the test suite both do - destroys the log of a completed run. Logs are evidence,
    so runs are separated by a banner instead of overwritten.

    The banner is written on the first actual log line rather than at import, because
    importing a module to reuse one helper (the test suite does this) would otherwise
    stamp an empty run into the record of a real one.
    """
    LOGDIR.mkdir(parents=True, exist_ok=True)
    path = LOGDIR / f"{name}.log"
    started = False

    def _banner() -> None:
        nonlocal started
        if not started:
            started = True
            with path.open("a", encoding="utf-8") as f:
                f.write(f"\n===== run {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    def log(msg: str) -> None:
        _banner()
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            # The Windows console is cp1252. Titles fetched from bibliographic APIs carry
            # characters it cannot encode (Crossref returns U+2010 in "inter-rater"), and a
            # crash in the *display* path would otherwise destroy a completed run. The file
            # is UTF-8 and already holds the exact text; only the echo degrades.
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)

    def excepthook(exc_type, exc, tb):
        import traceback

        _banner()
        with path.open("a", encoding="utf-8") as f:
            f.write("TRACEBACK\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook
    return log
