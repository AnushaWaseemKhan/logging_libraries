from __future__ import annotations

import asyncio
import threading

from aiologger import Logger
from aiologger.handlers.streams import AsyncStreamHandler

from .base import LoggerBase


class Aiologger(LoggerBase):
    def __init__(self, *, name: str, level, output: str, log_path=None):
        super().__init__(name=name, level=level, output=output, log_path=log_path)

        if hasattr(level, "num"):
            logger_level = level.num
        elif isinstance(level, (int, str)):
            logger_level = level
        else:
            raise TypeError(f"Unsupported level type for aiologger: {level!r}")

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        self.logger = Logger(name=f"bench.{name}", level=logger_level)
        self.logger.add_handler(AsyncStreamHandler())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _log_on_loop(self, method_name: str, msg: str) -> None:
        log_call = getattr(self.logger, method_name, None)
        if log_call is None:
            raise AttributeError(f"Invalid log level: {method_name}")
        await log_call(msg)

    def log(self, level, msg: str) -> None:
        method_name = level.name.lower()
        future = asyncio.run_coroutine_threadsafe(
            self._log_on_loop(method_name, msg),
            self.loop,
        )
        future.result()

    async def _shutdown_on_loop(self) -> None:
        await self.logger.shutdown()

    def close(self) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._shutdown_on_loop(),
                self.loop,
            )
            future.result()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()
