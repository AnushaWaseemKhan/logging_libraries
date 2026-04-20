from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class LoggerBase(ABC):
    def __init__(self, *, name: str, level, output: str, log_path: Path | None):
        self.name = name
        self.level = level
        self.output = output
        self.log_path = log_path

    @abstractmethod
    def log(self, level, msg):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Cleanup resources (handlers/files/etc.)."""
        raise NotImplementedError