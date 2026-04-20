from __future__ import annotations

import sys
from pathlib import Path

import picologging as logging

from .base import LoggerBase


class PicoLoggingAdapter(LoggerBase):
    def __init__(self, *, name: str, level, output: str, log_path: Path | None):
        super().__init__(name=name, level=level, output=output, log_path=log_path)

        logger = logging.getLogger(f"bench.{name}")

        # Remove + close old handlers
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
            handler = logging.FileHandler(str(log_path), mode="a")
        else:
            handler = logging.StreamHandler(sys.stdout)

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