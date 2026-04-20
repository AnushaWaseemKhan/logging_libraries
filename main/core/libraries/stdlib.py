from __future__ import annotations

import logging
import sys
from pathlib import Path

from .base import LoggerBase


class Stdlibrary(LoggerBase):
    def __init__(self, *, name: str, level, output: str, log_path: Path | None):
        super().__init__(name=name, level=level, output=output, log_path=log_path)

        logger = logging.getLogger(f"bench.{name}")

        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

        logger.setLevel(level.num)
        logger.propagate = False

        if output == "file":
            if log_path is None:
                raise ValueError("log_path is required when output='file'")
            handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        else:
            handler = logging.StreamHandler(sys.stdout)

        logger.addHandler(handler)

        self.logger = logger
        self.handler = handler

    def log(self, level, msg: str) -> None:
        self.logger.log(level.num, msg)

    def close(self) -> None:
        self.logger.removeHandler(self.handler)
        self.handler.close()