"""Logging configuration for the application."""

import sys
import threading
from pathlib import Path

from loguru import logger

from utils.config import LOGS_DIR, LOG_LEVEL


class CyclicLogSink:
    """
    Simple cyclic file sink that keeps only the most recent N log lines
    and keeps the newest entries at the top of the file.

    This is optimized for human inspection rather than raw throughput:
    we cap the file to a small fixed number of lines (e.g., 2000), and each
    write rewrites at most that many lines.
    """

    def __init__(self, path: Path, max_lines: int = 2000):
        self._path = path
        self._max_lines = max_lines
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        # Loguru passes a formatted string here because we register this sink
        # with 'format=...' on logger.add.
        text = str(message)
        if not text.endswith("\n"):
            text += "\n"

        with self._lock:
            if self._path.exists():
                existing = self._path.read_text(encoding="utf-8").splitlines(keepends=True)
            else:
                existing = []

            # Prepend newest entry so latest logs are at the top.
            new_lines = [text] + existing
            if len(new_lines) > self._max_lines:
                new_lines = new_lines[: self._max_lines]

            self._path.write_text("".join(new_lines), encoding="utf-8")


# Remove default handler
logger.remove()

# Console handler (unchanged)
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=LOG_LEVEL,
    colorize=True,
)

# File handler: cyclic log with newest entries at the top and a hard cap on line count.
log_file = LOGS_DIR / "app.log"
cyclic_sink = CyclicLogSink(log_file, max_lines=2000)
logger.add(
    cyclic_sink,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=LOG_LEVEL,
)

__all__ = ["logger"]

