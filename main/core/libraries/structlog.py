from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

from .base import LoggerBase


class StructlogAdapter(LoggerBase):
    def __init__(self, *, name: str, level, output: str, log_path: Path | None):
        super().__init__(name=name, level=level, output=output, log_path=log_path)

        # Use a per-library logger name so different libs don't fight over handlers
        self._logger_name = f"bench.{name}"

        base_logger = logging.getLogger(self._logger_name)

        # Remove + close any previous handlers (important across many runs)
        for h in list(base_logger.handlers):
            base_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        base_logger.setLevel(level.num)
        base_logger.propagate = False

        if output == "file":
            if log_path is None:
                raise ValueError("log_path is required when output='file'")
            handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        else:
            handler = logging.StreamHandler(sys.stdout)

        base_logger.addHandler(handler)

        # Configure structlog to render JSON and route via stdlib logging
        structlog.configure(
            processors=[structlog.processors.JSONRenderer()],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Get the structlog logger that matches the stdlib logger name above
        self._log = structlog.get_logger(self._logger_name)

        self._base_logger = base_logger
        self._handler = handler

    def log(self, level, msg: str) -> None:
        # level.name should be: "debug", "info", "warning", "error", "critical"
        getattr(self._log, level.name)(
            "event",
            message=msg,
        )

    def close(self) -> None:
        try:
            self._handler.close()
        finally:
            try:
                self._base_logger.removeHandler(self._handler)
            except Exception:
                pass