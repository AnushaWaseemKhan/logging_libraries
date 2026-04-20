from __future__ import annotations

import sys
from pathlib import Path

import logbook

from .base import LoggerBase


class LogbookAdapter(LoggerBase):
    def __init__(self, *, name: str, level, output: str, log_path: Path | None):
        super().__init__(name=name, level=level, output=output, log_path=log_path)

        self.lb_level = getattr(logbook, level.name.upper(), logbook.INFO)

        # Logbook logger
        self.logger = logbook.Logger(f"bench.{name}", level=self.lb_level)

        # Handler
        if output == "file":
            if log_path is None:
                raise ValueError("log_path is required when output='file'")
            self.handler = logbook.FileHandler(str(log_path), mode="a", level=self.lb_level)
        else:
            self.handler = logbook.StreamHandler(sys.stdout, level=self.lb_level)

        # Activate handler (equivalent to: with handler: ...)
        self.handler.push_application()

    def log(self, level, msg: str) -> None:
        # logbook logs via methods: logger.debug/info/warning/error/critical
        getattr(self.logger, level.name)(msg)

    def close(self) -> None:
        # Deactivate + close handler
        try:
            self.handler.pop_application()
        finally:
            try:
                self.handler.close()
            except Exception:
                pass