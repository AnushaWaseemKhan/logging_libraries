from __future__ import annotations

import logging
import sys
from pathlib import Path

from pythonjsonlogger import jsonlogger

from .base import LoggerBase


class PythonJsonLoggerAdapter(LoggerBase):
    def __init__(self, *, name: str, level, output: str, log_path: Path | None):
        super().__init__(name=name, level=level, output=output, log_path=log_path)

        logger = logging.getLogger(f"bench.{name}")

        # Remove + close old handlers (important for repeated runs)
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        logger.setLevel(level.num)
        logger.propagate = False

        if output == "file":
            if log_path is None:
                raise ValueError("log_path is required when output='file'")
            handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        else:
            handler = logging.StreamHandler(sys.stdout)

        # JSON formatter instead of plain text
        formatter = jsonlogger.JsonFormatter()
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        self.logger = logger
        self.handler = handler

    def log(self, level, msg: str) -> None:
        self.logger.log(level.num, msg)

    def close(self) -> None:
        try:
            self.handler.close()
        finally:
            try:
                self.logger.removeHandler(self.handler)
            except Exception:
                pass