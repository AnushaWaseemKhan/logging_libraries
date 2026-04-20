from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from .base import LoggerBase


class LoguruAdapter(LoggerBase):
    def __init__(self, *, name: str, level, output: str, log_path: Path | None):
        super().__init__(name=name, level=level, output=output, log_path=log_path)

        # Important: loguru logger is global. Removing handlers affects all uses.
        # For this experiment (single-threaded, one adapter active at a time) it's okay.
        _logger.remove()  # remove default handler(s)

        if output == "file":
            if log_path is None:
                raise ValueError("log_path is required when output='file'")
            sink = str(log_path)
            self.sink_id = _logger.add(sink, mode="a", enqueue=False)
        else:
            self.sink_id = _logger.add(sys.stdout, enqueue=False)

        self._logger = _logger

    def log(self, level, msg: str) -> None:
        # Your `level.name` is like "debug"/"info"/...
        # Loguru expects "DEBUG"/"INFO"/...
        self._logger.log(level.name.upper(), msg)

    def close(self) -> None:
        try:
            self._logger.remove(self.sink_id)
        except Exception:
            pass