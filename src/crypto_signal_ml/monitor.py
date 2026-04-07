"""Unified model-service runner for signal refresh and live monitoring."""

from __future__ import annotations

import logging
import os
from threading import Event, Thread
from typing import Any, Callable

import uvicorn

from .app import SignalGenerationApp
from .config import TrainingConfig
from .engine_api import create_app


LOGGER = logging.getLogger(__name__)


class SignalMonitorService:
    """Run signal generation on a schedule while serving the live engine API."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        signal_generation_app_factory: Callable[..., SignalGenerationApp] = SignalGenerationApp,
        app_factory: Callable[..., Any] = create_app,
    ) -> None:
        self.config = config or TrainingConfig()
        self.signal_generation_app_factory = signal_generation_app_factory
        self.app_factory = app_factory
        self._stop_event = Event()
        self._refresh_thread: Thread | None = None

    def _configure_logging(self) -> None:
        """Initialize basic logging when the process has no handlers yet."""

        root_logger = logging.getLogger()
        if root_logger.handlers:
            return

        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    def _run_signal_generation(self, cycle_label: str) -> None:
        """Generate and publish the latest watchlist and signal snapshots."""

        LOGGER.info("Starting %s signal refresh cycle.", cycle_label)
        results = self.signal_generation_app_factory(config=self.config).run()
        LOGGER.info(
            "%s signal refresh complete: %s signals, %s actionable, latest=%s.",
            cycle_label.capitalize(),
            int(results.get("signalsGenerated", 0)),
            int(results.get("actionableSignalsGenerated", 0)),
            results.get("signalName") or "none",
        )

    def _refresh_loop(self, interval_seconds: int) -> None:
        """Keep refreshing the published signal snapshot on a fixed interval."""

        while not self._stop_event.wait(interval_seconds):
            try:
                self._run_signal_generation(cycle_label="scheduled")
            except Exception:
                LOGGER.exception("Scheduled signal refresh failed.")

    def _start_background_refresh(self) -> None:
        """Launch the periodic signal-refresh worker when enabled."""

        interval_seconds = int(getattr(self.config, "signal_monitor_refresh_interval_seconds", 0) or 0)
        if interval_seconds <= 0:
            LOGGER.info("Signal monitor background refresh is disabled.")
            return

        self._refresh_thread = Thread(
            target=self._refresh_loop,
            args=(interval_seconds,),
            name="signal-monitor-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        LOGGER.info(
            "Signal monitor background refresh started with a %s-second interval.",
            interval_seconds,
        )

    def serve(self) -> None:
        """Run the full model-side service lifecycle in one process."""

        self._configure_logging()

        if bool(getattr(self.config, "signal_monitor_run_initial_generation", True)):
            try:
                self._run_signal_generation(cycle_label="initial")
            except Exception:
                LOGGER.exception("Initial signal refresh failed; continuing with the live API.")

        self._start_background_refresh()

        host = os.getenv("AI_ENGINE_HOST", "127.0.0.1")
        port = int(os.getenv("AI_ENGINE_PORT", "8100"))
        internal_api_key = str(os.getenv("AI_ENGINE_INTERNAL_API_KEY", "")).strip()
        app = self.app_factory(
            config=self.config,
            require_internal_api_key=bool(internal_api_key),
            internal_api_key=internal_api_key or None,
        )

        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
            )
        finally:
            self._stop_event.set()
