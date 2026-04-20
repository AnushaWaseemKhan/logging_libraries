from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass(frozen=True)
class Level:
    name: str   # "info"
    num: int    # 20


LEVELS = {
    "debug": Level("debug", logging.DEBUG),
    "info": Level("info", logging.INFO),
    "warning": Level("warning", logging.WARNING),
    "error": Level("error", logging.ERROR),
    "critical": Level("critical", logging.CRITICAL),
}


def prepare_log(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def make_message(size):
    return "X" * int(size)


def get_level(statement):
    """
    Single function:
      returns Level(name, num)
    """
    s = str(statement).lower()
    return LEVELS.get(s, LEVELS["info"])